# =============================================================================
# core/tracker.py — SORT Tracker (zero external dependencies beyond numpy)
# =============================================================================
#
# SORT = Simple Online and Realtime Tracking
# Uses Kalman Filter for motion prediction + Hungarian algorithm for assignment
# No face embedding model needed — pure motion-based tracking
#
# Why replaced boxmot:
#   boxmot has Python 3.12/3.13 compatibility issues on Windows.
#   This implementation is self-contained, faster to install, and equally
#   effective for Phase 1 face tracking.
# =============================================================================

import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TRACK_MAX_AGE, TRACK_MIN_HITS, TRACK_IOU_THRESHOLD


# =============================================================================
# Kalman Filter — predicts face position between detections
# =============================================================================

class KalmanBoxTracker:
    """
    Tracks a single face bounding box using a Kalman Filter.
    State: [x1, y1, x2, y2, vx, vy, vw, vh]
           position + velocity of the bounding box
    """
    _count = 0

    def __init__(self, bbox: list[int]):
        from filterpy.kalman import KalmanFilter

        self.kf = KalmanFilter(dim_x=8, dim_z=4)

        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0,0],
            [0,1,0,0,0,1,0,0],
            [0,0,1,0,0,0,1,0],
            [0,0,0,1,0,0,0,1],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,1],
        ], dtype=float)

        # Measurement matrix (we observe x1,y1,x2,y2 directly)
        self.kf.H = np.array([
            [1,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0],
        ], dtype=float)

        # Measurement noise
        self.kf.R[2:, 2:] *= 10.0
        # Covariance — high uncertainty in velocity initially
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P         *= 10.0
        # Process noise
        self.kf.Q[4:, 4:] *= 0.01

        # Initialise state with first detection
        self.kf.x[:4] = np.array(bbox, dtype=float).reshape((4, 1))

        self.id           = KalmanBoxTracker._count
        KalmanBoxTracker._count += 1

        self.hits         = 1
        self.hit_streak   = 1
        self.age          = 0
        self.time_since_update = 0
        self.confidence   = 1.0

    def predict(self) -> np.ndarray:
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.kf.x[:4].flatten()

    def update(self, bbox: list[int], confidence: float):
        self.kf.update(np.array(bbox, dtype=float).reshape((4, 1)))
        self.hits              += 1
        self.hit_streak        += 1
        self.time_since_update  = 0
        self.confidence         = confidence

    def get_state(self) -> list[int]:
        s = self.kf.x[:4].flatten()
        return [int(s[0]), int(s[1]), int(s[2]), int(s[3])]


# =============================================================================
# IoU + Hungarian assignment helpers
# =============================================================================

def _iou(bb1: list, bb2: list) -> float:
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    xx1 = max(bb1[0], bb2[0])
    yy1 = max(bb1[1], bb2[1])
    xx2 = min(bb1[2], bb2[2])
    yy2 = min(bb1[3], bb2[3])

    w = max(0, xx2 - xx1)
    h = max(0, yy2 - yy1)
    intersection = w * h

    area1 = (bb1[2]-bb1[0]) * (bb1[3]-bb1[1])
    area2 = (bb2[2]-bb2[0]) * (bb2[3]-bb2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def _assign(detections: list, trackers: list, iou_thresh: float):
    """
    Hungarian algorithm to match detections to existing tracks.
    Returns (matched, unmatched_dets, unmatched_trks) indices.
    """
    if not trackers:
        return [], list(range(len(detections))), []

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=float)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = _iou(det["bbox"], trk)

    # Hungarian — minimise cost = maximise IoU
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    matched, unmatched_dets, unmatched_trks = [], [], []

    for d in range(len(detections)):
        if d not in row_ind:
            unmatched_dets.append(d)

    for t in range(len(trackers)):
        if t not in col_ind:
            unmatched_trks.append(t)

    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] < iou_thresh:
            unmatched_dets.append(r)
            unmatched_trks.append(c)
        else:
            matched.append((r, c))

    return matched, unmatched_dets, unmatched_trks


# =============================================================================
# SORT Tracker
# =============================================================================

class FaceTracker:
    """
    SORT-based multi-face tracker.
    Assigns stable Face IDs using Kalman prediction + IoU matching.

    Usage:
        tracker = FaceTracker(frame_size=(width, height))
        tracks  = tracker.update(detections, frame)
    """

    def __init__(self, frame_size: tuple[int, int]):
        # Reset global ID counter for each new video
        KalmanBoxTracker._count = 0

        self.trackers  : list[KalmanBoxTracker] = []
        self.frame_size = frame_size

        print(
            f"[FaceTracker] SORT ready | "
            f"max_age={TRACK_MAX_AGE} | "
            f"min_hits={TRACK_MIN_HITS} | "
            f"iou={TRACK_IOU_THRESHOLD}"
        )

    def update(
        self,
        detections: list[dict],
        frame      = None,          # kept for API compatibility
    ) -> list[dict]:
        """
        Parameters
        ----------
        detections : output of FaceDetector.detect()
        frame      : unused (kept for interface compatibility with boxmot)

        Returns
        -------
        list of dicts:
            [{"track_id": int, "bbox": [x1,y1,x2,y2], "confidence": float}]
        """
        # 1. Predict new positions for all existing tracks
        predicted_boxes = []
        for trk in self.trackers:
            predicted_boxes.append(trk.predict().tolist())

        # 2. Match detections to predicted tracks
        matched, unmatched_dets, unmatched_trks = _assign(
            detections, predicted_boxes, TRACK_IOU_THRESHOLD
        )

        # 3. Update matched tracks
        for det_idx, trk_idx in matched:
            self.trackers[trk_idx].update(
                detections[det_idx]["bbox"],
                detections[det_idx]["confidence"],
            )

        # 4. Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self.trackers.append(
                KalmanBoxTracker(detections[det_idx]["bbox"])
            )
            self.trackers[-1].confidence = detections[det_idx]["confidence"]

        # 5. Remove dead tracks (not seen for too long)
        self.trackers = [
            t for t in self.trackers
            if t.time_since_update <= TRACK_MAX_AGE
        ]

        # 6. Return only confirmed tracks (seen enough times)
        results = []
        for trk in self.trackers:
            if (
                trk.time_since_update == 0 and     # updated this frame
                trk.hits >= TRACK_MIN_HITS          # confirmed track
            ):
                results.append({
                    "track_id"  : trk.id,
                    "bbox"      : trk.get_state(),
                    "confidence": round(trk.confidence, 4),
                })

        return results