# =============================================================================
# core/visualizer.py — Draw bounding boxes + track IDs (debug mode)
# =============================================================================

import cv2
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    BOX_COLOR, BOX_THICKNESS,
    LABEL_COLOR, LABEL_FONT_SCALE, LABEL_THICKNESS
)

# Palette: 20 distinct colors so each Face ID gets its own color
_PALETTE = [
    (255,  56,  56), (255, 157,  51), (255, 112, 31), (255, 178,  29),
    ( 51, 255,  56), ( 51, 255, 255), ( 56, 255, 157), ( 31, 112, 255),
    (178,  29, 255), (255,  51, 178), ( 56, 157, 255), (157, 255,  56),
    (255,  56, 157), ( 29, 255, 178), (157,  56, 255), (255, 255,  51),
    (255,  56, 255), ( 56, 255,  56), ( 56,  56, 255), (112, 255,  31),
]


def _color(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


class Visualizer:
    """
    Draws tracked face bounding boxes + Face-ID labels onto a frame copy.

    Usage:
        vis   = Visualizer()
        debug = vis.draw(frame, tracks)   # returns annotated copy
    """

    def draw(
        self,
        frame: np.ndarray,
        tracks: list[dict],
        show_conf: bool = True,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        frame      : BGR frame
        tracks     : output of FaceTracker.update()
        show_conf  : whether to append confidence to the label

        Returns
        -------
        Annotated copy of the frame (original is not modified)
        """
        canvas = frame.copy()

        for t in tracks:
            tid             = t["track_id"]
            x1, y1, x2, y2 = t["bbox"]
            conf            = t["confidence"]
            color           = _color(tid)

            # ── bounding box ───────────────────────────────────────────────────
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, BOX_THICKNESS)

            # ── label ──────────────────────────────────────────────────────────
            label = f"ID:{tid}"
            if show_conf:
                label += f"  {conf:.2f}"

            (tw, th), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                LABEL_FONT_SCALE,
                LABEL_THICKNESS,
            )

            # filled background rectangle behind text for readability
            label_y = max(y1, th + 6)
            cv2.rectangle(
                canvas,
                (x1, label_y - th - 6),
                (x1 + tw + 4, label_y + baseline - 4),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                canvas,
                label,
                (x1 + 2, label_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                LABEL_FONT_SCALE,
                (0, 0, 0),          # black text on colored background
                LABEL_THICKNESS,
                cv2.LINE_AA,
            )

        # ── HUD: face count ────────────────────────────────────────────────────
        hud = f"Tracked faces: {len(tracks)}"
        cv2.putText(
            canvas, hud, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA
        )

        return canvas