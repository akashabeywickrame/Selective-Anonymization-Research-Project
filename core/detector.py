# =============================================================================
# core/detector.py — YOLOv8n-face detection wrapper
# =============================================================================
#
# Output per frame: list of dicts, each with keys:
#   bbox        : [x1, y1, x2, y2]  (pixel coords, absolute)
#   confidence  : float
#   class_id    : int  (always 0 = face)
# =============================================================================

import numpy as np
from ultralytics import YOLO
from pathlib import Path

import sys, os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    MODEL_PATH, DEVICE, DETECTION_CONFIDENCE,
    DETECTION_IOU, INPUT_SIZE, MAX_FACES
)

# Model weights — multiple mirrors in priority order
_WEIGHTS_URLS = [
    # Mirror 1: Hugging Face (most reliable)
    "https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt",
    # Mirror 2: GitHub releases (backup)
    "https://github.com/akanametov/yolo-face/releases/download/v0.0.3/yolov8n-face.pt",
]


class FaceDetector:
    """
    Wraps YOLOv8n-face for lightweight, accurate face detection.

    Usage:
        detector = FaceDetector()
        detections = detector.detect(frame)   # frame: np.ndarray BGR
    """

    def __init__(self):
        weights = self._resolve_weights()
        self.model = YOLO(str(weights))
        self.model.to(DEVICE)
        print(f"[FaceDetector] Loaded {weights.name} on {DEVICE.upper()}")

    # ── public ─────────────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run inference on a single BGR frame.

        Returns
        -------
        list of dicts:
            [{"bbox": [x1,y1,x2,y2], "confidence": float, "class_id": 0}, ...]
        """
        results = self.model.predict(
            source     = frame,
            imgsz      = INPUT_SIZE,
            conf       = DETECTION_CONFIDENCE,
            iou        = DETECTION_IOU,
            max_det    = MAX_FACES,
            device     = DEVICE,
            verbose    = False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf            = float(box.conf[0])
                cls             = int(box.cls[0])
                detections.append({
                    "bbox"      : [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": round(conf, 4),
                    "class_id"  : cls,
                })

        return detections

    # ── internal ───────────────────────────────────────────────────────────────
    @staticmethod
    def _resolve_weights() -> Path:
        """
        Use local weights if present, otherwise try multiple download mirrors.
        If all fail, prints manual download instructions and exits cleanly.
        """
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        if MODEL_PATH.exists():
            print(f"[FaceDetector] Using cached weights → {MODEL_PATH}")
            return MODEL_PATH

        import urllib.request

        print(f"[FaceDetector] Weights not found at {MODEL_PATH}")

        for i, url in enumerate(_WEIGHTS_URLS, 1):
            print(f"[FaceDetector] Trying mirror {i}/{len(_WEIGHTS_URLS)}: {url}")
            try:
                def _progress(block, block_size, total):
                    downloaded = block * block_size
                    if total > 0:
                        pct = min(downloaded / total * 100, 100)
                        print(f"\r  Downloading … {pct:.1f}%", end="", flush=True)

                urllib.request.urlretrieve(url, MODEL_PATH, reporthook=_progress)
                print(f"\n[FaceDetector] Saved → {MODEL_PATH}")
                return MODEL_PATH

            except Exception as e:
                print(f"\n[FaceDetector] Mirror {i} failed: {e}")
                if MODEL_PATH.exists():
                    MODEL_PATH.unlink()   # remove partial download

        # ── all mirrors failed → manual instructions ───────────────────────────
        print("\n" + "="*60)
        print("[FaceDetector] ERROR: All download mirrors failed.")
        print("Please download the model manually:")
        print()
        print("  1. Open this URL in your browser:")
        print("     https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt")
        print()
        print("  2. Save the file as:  yolov8n-face.pt")
        print(f"  3. Place it at:       {MODEL_PATH}")
        print()
        print("  Then run:  python main.py  again.")
        print("="*60)
        raise SystemExit(1)