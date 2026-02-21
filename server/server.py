"""TruFor detection server — FastAPI app serving the TruFor model."""

import asyncio
import base64
import logging
import os
import tempfile
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

from src.model import TruForDetector

logger = logging.getLogger(__name__)

# ── Load config ───────────────────────────────────────────────────────
_CONFIG_PATH = os.environ.get("TRUFOR_CONFIG", "config.yaml")
with open(_CONFIG_PATH, encoding="utf-8") as _f:
    CONFIG = yaml.safe_load(_f)

MAX_IMAGE_BYTES = CONFIG.get("server", {}).get("max_image_size_mb", 20) * 1024 * 1024


# ── Pydantic models ──────────────────────────────────────────────────
class DetectRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image bytes")
    filename: str = Field(default="image.jpg", description="Original filename (for logging)")


class DetectResponse(BaseModel):
    is_altered: float
    explanation: str
    domain_tag: str
    model: str = "trufor-v1"


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str


# ── App lifecycle ────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the TruFor model at startup."""
    logger.info("Loading TruFor model...")
    detector = TruForDetector(CONFIG)
    detector.load()

    logger.info("Model device: %s", detector.device)

    app.state.detector = detector
    app.state.inference_lock = asyncio.Lock()
    app.state.ready = True
    logger.info("Model loaded — server ready")
    yield
    app.state.ready = False
    logger.info("Shutting down")


app = FastAPI(title="TruFor Detection API", lifespan=lifespan)


# ── Routes ───────────────────────────────────────────────────────────
@app.get("/v1/health", response_model=HealthResponse)
async def health():
    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Model is still loading")
    return HealthResponse(
        status="ready",
        model="trufor-v1",
        device=app.state.detector.device,
    )


@app.post("/v1/detect", response_model=DetectResponse)
async def detect(req: DetectRequest):
    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Model is still loading")

    # Decode base64
    try:
        image_bytes = base64.b64decode(req.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Image too large ({len(image_bytes)} bytes). Max: {MAX_IMAGE_BYTES} bytes",
        )

    # Write to temp file and validate
    tmp_path = None
    try:
        suffix = os.path.splitext(req.filename)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        # Validate it's a real image
        try:
            with Image.open(tmp_path) as img:
                img.verify()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Run inference (serialized via lock, offloaded to thread)
        async with app.state.inference_lock:
            try:
                result = await asyncio.to_thread(
                    app.state.detector.detect, tmp_path
                )
            except Exception as e:
                logger.error("Inference failed for %s: %s", req.filename, e, exc_info=True)
                raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

        return DetectResponse(
            is_altered=result["score"],
            explanation=result["explanation"],
            domain_tag="",
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    host = CONFIG.get("server", {}).get("host", "0.0.0.0")
    port = CONFIG.get("server", {}).get("port", 8000)
    uvicorn.run(app, host=host, port=port)
