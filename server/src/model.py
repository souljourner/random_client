"""TruFor forgery detection model loading and inference.

Wraps the TruFor (CVPR 2023) CNN-based forgery detector for inference.
Requires the TruFor source code to be available on PYTHONPATH
(cloned during setup).
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


def _build_trufor_config():
    """Build a YACS CfgNode matching TruFor's default config structure."""
    from yacs.config import CfgNode as CN

    cfg = CN()

    # Top-level settings
    cfg.OUTPUT_DIR = "weights"
    cfg.LOG_DIR = "log"
    cfg.GPUS = (0,)
    cfg.WORKERS = 4

    # CUDNN
    cfg.CUDNN = CN()
    cfg.CUDNN.BENCHMARK = True
    cfg.CUDNN.DETERMINISTIC = False
    cfg.CUDNN.ENABLED = True

    # Model
    cfg.MODEL = CN()
    cfg.MODEL.NAME = "detconfcmx"
    cfg.MODEL.PRETRAINED = ""
    cfg.MODEL.MODS = ("RGB", "NP++")

    # MODEL.EXTRA — the nested config that builder_np_conf.py reads
    cfg.MODEL.EXTRA = CN(new_allowed=True)
    cfg.MODEL.EXTRA.BACKBONE = "mit_b2"
    cfg.MODEL.EXTRA.DECODER = "MLPDecoder"
    cfg.MODEL.EXTRA.DECODER_EMBED_DIM = 512
    cfg.MODEL.EXTRA.CONF = True
    cfg.MODEL.EXTRA.DETECTION = "confpool"
    cfg.MODEL.EXTRA.MODULES = ["NP++", "backbone", "loc_head", "conf_head", "det_head"]
    cfg.MODEL.EXTRA.FIX_MODULES = ["NP++"]
    cfg.MODEL.EXTRA.PREPRC = "imagenet"
    cfg.MODEL.EXTRA.NP_WEIGHTS = None
    cfg.MODEL.EXTRA.NP_OUT_CHANNELS = 1
    cfg.MODEL.EXTRA.BN_EPS = 0.001
    cfg.MODEL.EXTRA.BN_MOMENTUM = 0.1

    # Loss (required by config structure)
    cfg.LOSS = CN()
    cfg.LOSS.USE_OHEM = False
    cfg.LOSS.LOSSES = [["LOC", 1.0, "cross_entropy"]]
    cfg.LOSS.SMOOTH = 0

    # Dataset
    cfg.DATASET = CN()
    cfg.DATASET.ROOT = ""
    cfg.DATASET.TRAIN = []
    cfg.DATASET.VALID = []
    cfg.DATASET.NUM_CLASSES = 2
    cfg.DATASET.CLASS_WEIGHTS = [0.5, 2.5]

    # Train
    cfg.TRAIN = CN()
    cfg.TRAIN.IMAGE_SIZE = [512, 512]
    cfg.TRAIN.LR = 0.01
    cfg.TRAIN.OPTIMIZER = "sgd"
    cfg.TRAIN.MOMENTUM = 0.9
    cfg.TRAIN.WD = 0.0001
    cfg.TRAIN.NESTEROV = False
    cfg.TRAIN.IGNORE_LABEL = -1
    cfg.TRAIN.BEGIN_EPOCH = 0
    cfg.TRAIN.END_EPOCH = 100
    cfg.TRAIN.STOP_EPOCH = -1
    cfg.TRAIN.EXTRA_EPOCH = 0
    cfg.TRAIN.RESUME = True
    cfg.TRAIN.PRETRAINING = ""
    cfg.TRAIN.AUG = None
    cfg.TRAIN.BATCH_SIZE_PER_GPU = 18
    cfg.TRAIN.SHUFFLE = True
    cfg.TRAIN.NUM_SAMPLES = 0

    # Valid
    cfg.VALID = CN()
    cfg.VALID.IMAGE_SIZE = None
    cfg.VALID.AUG = None
    cfg.VALID.FIRST_VALID = True
    cfg.VALID.MAX_SIZE = None
    cfg.VALID.BEST_KEY = "avg_mIoU"

    # Test
    cfg.TEST = CN()
    cfg.TEST.MODEL_FILE = ""

    return cfg


class TruForDetector:
    """Wraps TruFor for forgery detection inference."""

    def __init__(self, config: dict):
        device = config["model"].get("device", "auto")
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda:0"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.weights_path = config["model"]["weight_path"]
        self.score_threshold = config["model"].get("score_threshold", 0.5)
        self.model = None

    def load(self):
        """Load model weights. Call once before inference."""
        from models.cmx.builder_np_conf import myEncoderDecoder

        logger.info("Building TruFor model...")
        cfg = _build_trufor_config()
        self.model = myEncoderDecoder(cfg=cfg)

        logger.info("Loading weights from %s", self.weights_path)
        checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        result = self.model.load_state_dict(checkpoint["state_dict"], strict=False)
        if result.missing_keys:
            logger.warning("Missing keys: %s", result.missing_keys)
        if result.unexpected_keys:
            logger.warning("Unexpected keys: %s", result.unexpected_keys)
        self.model.to(self.device)
        self.model.eval()
        logger.info("TruFor model loaded successfully on %s", self.device)

    def detect(self, image_path: str) -> dict:
        """Run forgery detection on a single image.

        Args:
            image_path: path to the image file on disk

        Returns:
            dict with 'score' (float 0-1) and 'explanation' (str)
        """
        MAX_EDGE = 2048
        pil_img = Image.open(image_path).convert("RGB")
        w, h = pil_img.size
        if max(w, h) > MAX_EDGE:
            scale = MAX_EDGE / max(w, h)
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            logger.info("Resized %s from %dx%d to %dx%d", image_path, w, h, *pil_img.size)
        img = np.array(pil_img)
        # HWC -> CHW, divide by 256.0 (matches TruFor training pipeline)
        rgb = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float) / 256.0
        rgb = rgb.unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred, conf, det, npp = self.model(rgb)

        score = torch.sigmoid(det).item()
        loc_map = F.softmax(torch.squeeze(pred, 0), dim=0)[1].cpu().numpy()

        # Compute % of pixels flagged
        pct_flagged = float((loc_map > self.score_threshold).mean() * 100)

        if score > self.score_threshold:
            explanation = (
                f"Detection score {score:.2f} (threshold {self.score_threshold}). "
                f"{pct_flagged:.1f}% of pixels flagged as potentially manipulated."
            )
        else:
            explanation = (
                f"Detection score {score:.2f} (threshold {self.score_threshold}). "
                f"Image appears authentic."
            )

        # Clean up device memory
        del rgb, pred, conf, det, npp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"score": score, "explanation": explanation}
