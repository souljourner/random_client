# FakeShield Detection Server — Setup & Usage

## Prerequisites

- Linux machine with an NVIDIA GPU (tested on RTX 3090 Ti)
- [Miniforge](https://github.com/conda-forge/miniforge) or Mambaforge installed (`mamba` available on PATH)
- ~26 GB free disk for model weights, ~30 GB VRAM for 8-bit inference

## 1. Initial Setup

```bash
cd server
bash setup.sh
```

This will:
1. Create a `fakeshield_server` conda environment (Python 3.10, CUDA 11.8, PyTorch 2.1)
2. Verify GPU is detected and bitsandbytes loads correctly
3. Clone the FakeShield repo, patch known issues, and install DTE-FDM dependencies
4. Prompt to download ~26 GB of model weights from HuggingFace

To nuke a broken environment and start fresh:

```bash
bash setup.sh --clean
```

## 2. Configuration

Edit `server/config.yaml` as needed:

```yaml
model:
  weight_path: "./weights/fakeshield-v1-22b/DTE-FDM"   # LLaVA model weights
  dtg_path: "./weights/fakeshield-v1-22b/DTG.pth"       # ResNet50 domain classifier
  load_8bit: true                                        # 8-bit quantization (recommended)
  load_4bit: false
  device: "cuda:0"
  temperature: 0.2
  max_new_tokens: 4096

server:
  host: "0.0.0.0"        # bind address
  port: 8000              # listen port
  max_image_size_mb: 20   # reject images larger than this
```

You can also override the config path via the `FAKESHIELD_CONFIG` environment variable.

## 3. Start the Server

```bash
cd server
conda activate fakeshield_server
export PYTHONPATH="${PWD}/FakeShield/DTE-FDM:${PYTHONPATH:-}"
uvicorn server:app --host 0.0.0.0 --port 8000
```

Or equivalently:

```bash
cd server
conda activate fakeshield_server
export PYTHONPATH="${PWD}/FakeShield/DTE-FDM:${PYTHONPATH:-}"
python server.py
```

On startup, the server loads the DTG (ResNet50) and DTE-FDM (LLaVA 13B) models into GPU memory. This takes 1-2 minutes. The server begins accepting requests once loading completes.

## 4. API Endpoints

### `GET /v1/health`

Check if the model is loaded and ready.

```bash
curl http://localhost:8000/v1/health
```

Response:
```json
{"status": "ready", "model": "fakeshield-v1-dte-fdm", "device": "cuda:0"}
```

Returns `503` if the model is still loading.

### `POST /v1/detect`

Submit a base64-encoded image for forgery detection.

```bash
curl -X POST http://localhost:8000/v1/detect \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$(base64 -w0 photo.jpg)\", \"filename\": \"photo.jpg\"}"
```

Request body:
```json
{"image": "<base64-encoded bytes>", "filename": "photo.jpg"}
```

Response (200):
```json
{
  "is_altered": 1.0,
  "explanation": "The image shows signs of tampering...",
  "domain_tag": "Photoshop",
  "model": "fakeshield-v1-dte-fdm"
}
```

| Field | Description |
|---|---|
| `is_altered` | `1.0` = forged, `0.0` = authentic, `-1.0` = parse error |
| `explanation` | Model's natural language explanation |
| `domain_tag` | One of: `AIGC inpainting`, `DeepFake`, `Photoshop` |
| `model` | Model identifier |

Error codes:
- **400** — invalid base64, corrupt image, or image exceeds size limit
- **500** — model inference failure
- **503** — model still loading

## 5. Verify It Works

```bash
# Check health
curl http://localhost:8000/v1/health

# Test with a synthetic image
cd /tmp && python3 -c "from PIL import Image; Image.new('RGB', (100,100), 'red').save('test.jpg')"
curl -s -X POST http://localhost:8000/v1/detect \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$(base64 -w0 test.jpg)\", \"filename\": \"test.jpg\"}" | python3 -m json.tool
```

## 6. Running in Production

To run the server in the background:

```bash
cd server
conda activate fakeshield_server
export PYTHONPATH="${PWD}/FakeShield/DTE-FDM:${PYTHONPATH:-}"
nohup uvicorn server:app --host 0.0.0.0 --port 8000 >> server.log 2>&1 &
```

Monitor with:
```bash
tail -f server.log
curl http://localhost:8000/v1/health
```

## Notes

- Inference is serialized with an `asyncio.Lock` — only one image is processed at a time on the GPU, but the event loop stays responsive for health checks
- The server writes incoming images to temp files (cleaned up after each request)
- On macOS `base64` doesn't have `-w0`; use `base64 -i photo.jpg` instead when testing locally
