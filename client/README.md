# FakeShield Client — Setup & Usage

## Prerequisites

- macOS (or any machine with network access to the GPU server)
- [Miniforge](https://github.com/conda-forge/miniforge) installed (`mamba` or `conda` available)
- Images stored as local files on this machine
- FakeShield server running and reachable

## 1. Setup

```bash
cd client
mamba env create -f environment.yaml
conda activate fakeshield_client
```

To update an existing environment:

```bash
mamba env update -n fakeshield_client -f environment.yaml --prune
```

## 2. Configuration

Edit `client/config.yaml`:

```yaml
server:
  url: "http://192.168.4.88:8000"   # GPU server address
  timeout: 120                       # seconds per request
  retries: 3                         # retry attempts on failure

checkpoint_file: "../output/checkpoint.json"
```

## 3. Prepare Input CSV

Create a CSV with columns `ticketId` and `image_path`:

```csv
ticketId,image_path
T001,/Users/jzhu/images/photo1.jpg
T002,/Users/jzhu/images/photo2.png
T003,/Users/jzhu/images/nonexistent.jpg
```

## 4. Run

```bash
conda activate fakeshield_client
cd client
python client.py --input ../input/sample.csv
```

### CLI Options

| Flag | Description |
|---|---|
| `--input PATH` | **(required)** Path to input CSV |
| `--output PATH` | Output CSV path (default: `../output/results.csv`) |
| `--config PATH` | Config file (default: `config.yaml`) |
| `--limit N` | Process only first N rows |
| `--no-resume` | Ignore checkpoint, start fresh |
| `--dry-run` | Validate image files without sending to server |
| `--verbose` / `-v` | Enable debug logging |

### Examples

```bash
# Dry run — validate images exist and are readable, no server needed
python client.py --input ../input/sample.csv --dry-run --verbose

# Process first 10 images
python client.py --input ../input/sample.csv --limit 10

# Full run, start fresh (ignore previous checkpoint)
python client.py --input ../input/sample.csv --no-resume

# Custom output path
python client.py --input ../input/sample.csv --output ../output/batch42.csv
```

## 5. Output Format

Results are written to CSV:

```csv
ticketId,image_path,is_altered,explanation,domain_tag
T001,/path/to/img1.jpg,1.0,"Signs of manipulation...",Photoshop
T002,/path/to/img2.jpg,0.0,"Image appears authentic.",AIGC inpainting
T003,/path/to/broken.jpg,-1.0,"[ERROR] File not found",
```

| `is_altered` Value | Meaning |
|---|---|
| `1.0` | Forged / tampered |
| `0.0` | Authentic |
| `-1.0` | Error (file missing, server failure, parse error) |

## 6. Checkpoint & Resume

- After each image, progress is saved to `../output/checkpoint.json`
- If interrupted (Ctrl+C), restart the same command — it skips already-completed tickets
- Use `--no-resume` to discard the checkpoint and reprocess everything

## 7. Troubleshooting

**Check server is reachable:**
```bash
curl http://192.168.4.88:8000/v1/health
```

**Test a single image manually (macOS):**
```bash
curl -s -X POST http://192.168.4.88:8000/v1/detect \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$(base64 -i /path/to/image.jpg)\", \"filename\": \"image.jpg\"}" \
  | python3 -m json.tool
```

**Common errors:**
- `[ERROR] File not found` — the `image_path` in the CSV doesn't exist; use absolute paths
- `[ERROR] Connection refused` — server is down or wrong IP; check `config.yaml` and `curl /v1/health`
- `[ERROR] 503` — server is still loading the model; wait 1-2 minutes and retry
