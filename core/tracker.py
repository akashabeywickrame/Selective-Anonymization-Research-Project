# =============================================================================
# core/tracker.py — ByteTrack face tracker (via boxmot)
# =============================================================================
#
# Why ByteTrack?
#   - Motion-only tracking: no face embedding model needed → very lightweight
#   - Handles fast motion and temporary occlusion (keeps ID alive for
#     TRACK_MAX_AGE frames even when face is invisible)
#   - Stable IDs: critical for Stage 3 classifier (per-face history)
#
# Output per frame: list of dicts, each with keys:
#   track_id    : int   — stable across frames
#   bbox        : [x1, y1, x2, y2]
#   confidence  : float
# =============================================================================

import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DEVICE, TRACK_MAX_AGE, TRACK_MIN_HITS, TRACK_IOU_THRESHOLD
)


class FaceTracker:
    """
    Wraps ByteTrack from boxmot to assign stable Face IDs.

    Usage:
        tracker = FaceTracker(frame_size=(width, height))
        tracks  = tracker.update(detections, frame)
    """

    def __init__(self, frame_size: tuple[int, int]):
        from boxmot import ByteTrack

        self.tracker = ByteTrack(
            track_thresh  = 0.45,          # detection confidence threshold
            match_thresh  = TRACK_IOU_THRESHOLD,
            track_buffer  = TRACK_MAX_AGE, # frames to keep lost track alive
            frame_rate    = 30,
        )

        self._frame_size = frame_size
        print(
            f"[FaceTracker] ByteTrack ready | "
            f"max_age={TRACK_MAX_AGE} | iou={TRACK_IOU_THRESHOLD}"
        )

    # ── public ─────────────────────────────────────────────────────────────────
    def update(
        self,
        detections: list[dict],
        frame: np.ndarray,
    ) -> list[dict]:
        """
        Feed current-frame detections into ByteTrack.

        Parameters
        ----------
        detections : output of FaceDetector.detect()
        frame      : current BGR frame (needed by boxmot internally)

        Returns
        -------
        list of dicts:
            [{"track_id": int, "bbox": [x1,y1,x2,y2], "confidence": float}, ...]
        """
        if not detections:
            # still call update so ByteTrack ages its internal tracks
            dets_np = np.empty((0, 6), dtype=np.float32)
        else:
            # boxmot expects shape (N, 6): [x1, y1, x2, y2, conf, class_id]
            dets_np = np.array(
                [
                    [
                        d["bbox"][0], d["bbox"][1],
                        d["bbox"][2], d["bbox"][3],
                        d["confidence"],
                        d["class_id"],
                    ]
                    for d in detections
                ],
                dtype=np.float32,
            )

        tracked = self.tracker.update(dets_np, frame)  # shape (N, 7)

        tracks = []
        if tracked is not None and len(tracked) > 0:
            for t in tracked:
                # boxmot returns: [x1, y1, x2, y2, track_id, conf, class_id, ...]
                x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
                track_id        = int(t[4])
                conf            = float(t[5])
                tracks.append({
                    "track_id"  : track_id,
                    "bbox"      : [x1, y1, x2, y2],
                    "confidence": round(conf, 4),
                })

        return tracks