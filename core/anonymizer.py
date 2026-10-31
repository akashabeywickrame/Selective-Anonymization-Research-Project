# =============================================================================
# core/anonymizer.py — Stage 4: conditional anonymization
# =============================================================================
#
# Applies privacy protection to faces. Two modes, so the evaluation can compare
# them head-to-head:
#
#   • SelectiveAnonymizer — the proposed system. Blurs ONLY faces the decision
#                           layer flagged as bystanders; interaction partners are
#                           left untouched to preserve AR utility.
#
#   • BlanketAnonymizer   — the baseline. Blurs EVERY detected face regardless of
#                           context (the Google-Maps / news-footage approach).
#
# Both write directly onto a copy of the frame so the original is never mutated.
# =============================================================================

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    ANON_METHOD, ANON_BLUR_KERNEL, ANON_PIXELATE_SIZE, ANON_PADDING,
)


# =============================================================================
# Low-level region obfuscation
# =============================================================================

def _clip_bbox(bbox, w, h):
    """Pad the bbox, then clamp it to the frame so slicing is always valid."""
    x1, y1, x2, y2 = bbox

    pad_x = int((x2 - x1) * ANON_PADDING)
    pad_y = int((y2 - y1) * ANON_PADDING)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    return x1, y1, x2, y2


def _obfuscate_region(frame: np.ndarray, bbox: list[int]) -> None:
    """Blur or pixelate the bbox region of `frame` IN PLACE."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = _clip_bbox(bbox, w, h)

    if x2 <= x1 or y2 <= y1:
        return  # degenerate region — nothing to do

    roi = frame[y1:y2, x1:x2]

    if ANON_METHOD == "pixelate":
        small = cv2.resize(
            roi,
            (max(1, roi.shape[1] // ANON_PIXELATE_SIZE),
             max(1, roi.shape[0] // ANON_PIXELATE_SIZE)),
            interpolation=cv2.INTER_LINEAR,
        )
        roi[:] = cv2.resize(small, (roi.shape[1], roi.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
    else:  # default: Gaussian blur
        k = ANON_BLUR_KERNEL | 1   # force odd kernel size
        roi[:] = cv2.GaussianBlur(roi, (k, k), 0)


# =============================================================================
# Proposed system — selective anonymization
# =============================================================================

class SelectiveAnonymizer:
    """Blurs only faces flagged as bystanders (interacting == False)."""

    def apply(self, frame: np.ndarray, decided_tracks: list[dict]) -> np.ndarray:
        """
        Parameters
        ----------
        frame          : BGR frame
        decided_tracks : output of InteractionClassifier.classify()

        Returns
        -------
        New frame with bystander faces obfuscated.
        """
        canvas = frame.copy()
        for t in decided_tracks:
            if not t.get("interacting", False):   # bystander → protect
                _obfuscate_region(canvas, t["bbox"])
        return canvas


# =============================================================================
# Baseline — blanket anonymization
# =============================================================================

class BlanketAnonymizer:
    """Blurs every face — the context-blind baseline we compare against."""

    def apply(self, frame: np.ndarray, tracks: list[dict]) -> np.ndarray:
        canvas = frame.copy()
        for t in tracks:
            _obfuscate_region(canvas, t["bbox"])
        return canvas
