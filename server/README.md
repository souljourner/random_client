# TruFor Detection Server — Setup & Usage

## Prerequisites

- Linux machine with NVIDIA GPU, or macOS with Apple Silicon
- [Miniforge](https://github.com/conda-forge/miniforge) installed (`mamba` available on PATH)
- ~500 MB free disk for model weights

## 1. Initial Setup

```bash
cd server
bash setup.sh
```

This will:
1. Create a `trufor_server` environment (Python 3.10, PyTorch 2.2+)
2. Install CUDA support automatically if an NVIDIA GPU is detected
3. Verify compute device (CUDA, MPS, or CPU)
4. Clone the TruFor repo
5. Prompt to download ~200 MB of model weights

To nuke a broken environment and start fresh:

```bash
bash setup.sh --clean
```

## 2. Configuration

Edit `server/config.yaml` as needed:

```yaml
model:
  weight_path: "./weights/weights/trufor.pth.tar"
  device: "auto"           # auto-detects CUDA, MPS, or CPU
  score_threshold: 0.5

server:
  host: "0.0.0.0"          # bind address
  port: 8000                # listen port
  max_image_size_mb: 20     # reject images larger than this
```

You can also override the config path via the `TRUFOR_CONFIG` environment variable.

## 3. Start the Server

```bash
cd server
mamba activate trufor_server
export PYTHONPATH="${PWD}/TruFor/test_docker/src:${PYTHONPATH:-}"
uvicorn server:app --host 0.0.0.0 --port 8000
```

Or equivalently:

```bash
cd server
mamba activate trufor_server
export PYTHONPATH="${PWD}/TruFor/test_docker/src:${PYTHONPATH:-}"
python server.py
```

On startup, the server loads TruFor (~68.7M params) into device memory. This takes a few seconds. The server begins accepting requests once loading completes.

## 4. API Endpoints

### `GET /v1/health`

Check if the model is loaded and ready.

```bash
curl http://localhost:8000/v1/health
```

Response:
```json
{"status": "ready", "model": "trufor-v1", "device": "cuda:0"}
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
  "is_altered": 0.73,
  "explanation": "Detection score 0.73 (threshold 0.5). 12.4% of pixels flagged as potentially manipulated.",
  "domain_tag": "",
  "model": "trufor-v1"
}
```

| Field | Description |
|---|---|
| `is_altered` | Float 0-1. Higher = more likely tampered. `-1.0` on error. |
| `explanation` | Human-readable summary of detection score and pixel analysis |
| `domain_tag` | Empty string (TruFor does not classify forgery type) |
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
mamba activate trufor_server
export PYTHONPATH="${PWD}/TruFor/test_docker/src:${PYTHONPATH:-}"
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
