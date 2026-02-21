"""Client-side pipeline: read images, send to server, collect results."""

import base64
import csv
import io
import json
import logging
import os
import tempfile
import time

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_input_csv(path: str) -> list[dict]:
    """Load input CSV. Expected columns: ticketId, image_path."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "ticketId" not in row or "image_path" not in row:
                raise ValueError(f"CSV must have 'ticketId' and 'image_path' columns. Found: {list(row.keys())}")
            rows.append({"ticketId": row["ticketId"].strip(), "image_path": row["image_path"].strip()})
    return rows


def load_checkpoint(path: str) -> set[str]:
    """Load set of completed image paths from checkpoint file."""
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return set(data.get("completed", []))


def save_checkpoint(path: str, completed: set[str]):
    """Save completed image paths to checkpoint file (atomic write)."""
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=dir_name or ".", suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump({"completed": sorted(completed)}, f)
        os.replace(tmp_path, path)
    except BaseException:
        os.unlink(tmp_path)
        raise


def append_results_csv(path: str, results: list[dict], write_header: bool = False):
    """Append results to output CSV."""
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    mode = "w" if write_header else "a"
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ticketId", "image_path", "is_altered", "explanation", "domain_tag"])
        if write_header:
            writer.writeheader()
        writer.writerows(results)


def _check_server_health(server_url: str, timeout: int = 10):
    """Check that the server is ready. Raises on failure."""
    health_url = f"{server_url.rstrip('/')}/v1/health"
    try:
        resp = requests.get(health_url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "ready":
            raise RuntimeError(f"Server not ready: {data}")
        logger.info("Server healthy: %s", data)
    except requests.ConnectionError:
        raise RuntimeError(f"Cannot connect to server at {health_url}")
    except requests.Timeout:
        raise RuntimeError(f"Server health check timed out ({timeout}s)")


def _is_url(path: str) -> bool:
    """Check if a path is a URL."""
    return path.startswith("http://") or path.startswith("https://")


def _read_image_bytes(image_path: str) -> bytes:
    """Read image bytes from a local path or URL."""
    if _is_url(image_path):
        resp = requests.get(image_path, timeout=30)
        resp.raise_for_status()
        return resp.content
    with open(image_path, "rb") as f:
        return f.read()


def _send_image(image_path: str, server_url: str, timeout: int, retries: int) -> dict:
    """Read image (local or URL), base64-encode, POST to server. Returns response JSON.

    Returns a dict with keys: is_altered, explanation, domain_tag, model.
    On failure after all retries, raises an exception.
    """
    detect_url = f"{server_url.rstrip('/')}/v1/detect"

    image_bytes = _read_image_bytes(image_path)

    encoded = base64.b64encode(image_bytes).decode("ascii")
    filename = os.path.basename(image_path.rstrip("/").split("?")[0])
    payload = {"image": encoded, "filename": filename}

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(detect_url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            last_error = e
            status = e.response.status_code if e.response is not None else "?"
            detail = ""
            try:
                detail = e.response.json().get("detail", "")
            except Exception:
                pass
            logger.warning(
                "attempt=%d/%d HTTP %s for %s: %s",
                attempt, retries, status, image_path, detail,
            )
            # Don't retry 400 (bad request) — it won't succeed on retry
            if e.response is not None and e.response.status_code == 400:
                raise
        except (requests.ConnectionError, requests.Timeout) as e:
            last_error = e
            logger.warning("attempt=%d/%d error for %s: %s", attempt, retries, image_path, e)

        if attempt < retries:
            backoff = 2 ** attempt
            logger.debug("Retrying in %ds...", backoff)
            time.sleep(backoff)

    raise RuntimeError(f"Failed after {retries} attempts: {last_error}")


def run_pipeline(
    input_csv: str,
    output_csv: str,
    config: dict,
    limit: int | None = None,
    no_resume: bool = False,
    dry_run: bool = False,
):
    """Execute the client-side detection pipeline.

    Args:
        input_csv: path to input CSV file (ticketId, image_path)
        output_csv: path to output CSV file
        config: parsed client config.yaml dict
        limit: process only first N rows
        no_resume: ignore checkpoint, start fresh
        dry_run: validate CSV and test first 3 images only
    """
    server_url = config["server"]["url"]
    timeout = config["server"].get("timeout", 120)
    retries = config["server"].get("retries", 3)
    checkpoint_path = config.get("checkpoint_file", "checkpoint.json")

    # 1. Load input
    logger.info("Loading input CSV: %s", input_csv)
    rows = load_input_csv(input_csv)
    logger.info("Found %d rows in input", len(rows))

    if limit is not None:
        rows = rows[:limit]
        logger.info("Limited to %d rows", len(rows))

    if dry_run:
        rows = rows[:3]
        logger.info("Dry run: processing first %d rows", len(rows))

    # 2. Load checkpoint
    if no_resume:
        completed = set()
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        logger.info("Starting fresh (--no-resume)")
    else:
        completed = load_checkpoint(checkpoint_path)
        logger.info("Checkpoint: %d tickets already processed", len(completed))

    # 3. Filter completed
    pending = [r for r in rows if r["image_path"] not in completed]
    logger.info("%d images pending", len(pending))

    if not pending:
        logger.info("Nothing to process. All tickets already completed.")
        return

    # 4. Write CSV header if starting fresh
    write_header = not os.path.exists(output_csv) or no_resume
    if write_header:
        append_results_csv(output_csv, [], write_header=True)

    # 5. Check server health
    if not dry_run:
        _check_server_health(server_url)

    # 6. Process images one at a time
    total_processed = 0
    total_start = time.time()

    for row in tqdm(pending, desc="Processing", unit="img"):
        tid = row["ticketId"]
        image_path = row["image_path"]

        if not _is_url(image_path) and not os.path.exists(image_path):
            result = {
                "ticketId": tid,
                "image_path": image_path,
                "is_altered": -1.0,
                "explanation": "[ERROR] File not found",
                "domain_tag": "",
            }
        elif dry_run:
            # In dry-run mode, just validate the file exists and is an image
            try:
                from PIL import Image
                if _is_url(image_path):
                    image_bytes = _read_image_bytes(image_path)
                    img = Image.open(io.BytesIO(image_bytes))
                    img.verify()
                    img.close()
                else:
                    with Image.open(image_path) as img:
                        img.verify()
                result = {
                    "ticketId": tid,
                    "image_path": image_path,
                    "is_altered": -1.0,
                    "explanation": "[DRY_RUN] Image valid, skipping inference",
                    "domain_tag": "",
                }
            except Exception as e:
                result = {
                    "ticketId": tid,
                    "image_path": image_path,
                    "is_altered": -1.0,
                    "explanation": f"[DRY_RUN] Invalid image: {e}",
                    "domain_tag": "",
                }
        else:
            try:
                resp = _send_image(image_path, server_url, timeout, retries)
                result = {
                    "ticketId": tid,
                    "image_path": image_path,
                    "is_altered": resp["is_altered"],
                    "explanation": resp["explanation"],
                    "domain_tag": resp.get("domain_tag", ""),
                }
            except Exception as e:
                logger.error("ticket=%s inference error: %s", tid, e, exc_info=True)
                result = {
                    "ticketId": tid,
                    "image_path": image_path,
                    "is_altered": -1.0,
                    "explanation": f"[ERROR] {e}",
                    "domain_tag": "",
                }

        # Write result and checkpoint after each image
        append_results_csv(output_csv, [result])
        completed.add(image_path)
        save_checkpoint(checkpoint_path, completed)
        total_processed += 1

    elapsed = time.time() - total_start
    rate = total_processed / elapsed if elapsed > 0 else 0
    logger.info(
        "Pipeline complete. %d tickets processed in %.1fs (%.1f img/min). Output: %s",
        total_processed, elapsed, rate * 60, output_csv,
    )
