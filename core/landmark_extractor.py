# =============================================================================
# core/landmark_extractor.py — MediaPipe Face Mesh on YOLO-detected crops
# =============================================================================
#
# YOLO + SORT handle detection and stable track IDs.  This module layers
# landmark-based cues on top: given a frame and a bounding box from SORT, it
# crops the face region, runs MediaPipe Face Mesh (static_image_mode=True so
# Face Mesh does NOT attempt its own tracking — SORT owns that), and returns
# the two Phase 2b cues:
#
#   yaw  — head-pose yaw in degrees via the PnP problem (solvePnP + RQDecomp).
#           Negative = face turned left, positive = right.  |yaw| > ~40° means
#           the person is not looking at the camera wearer → bystander.
#
#   mar  — Mouth Aspect Ratio, averaged over the inner lip landmarks.
#           MAR ≈ 0 → mouth closed.  MAR ≈ 0.1–0.3+ → mouth open / talking.
#           Used by the V-VAD (Visual Voice Activity Detection) cue in the
#           classifier's rolling window.
#
# Falls back to None when Face Mesh finds no face in the crop (e.g. the face
# is too small or partially occluded).  The classifier treats None as "cue
# unavailable" and relies on Phase 2a geometry only.
# =============================================================================

import math
import cv2
import numpy as np

from pathlib import Path

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker, FaceLandmarkerOptions, RunningMode,
    )
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

LANDMARKER_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


# ── Landmark indices (MediaPipe 468-point model) ──────────────────────────────

# 6 key points for head-pose estimation via solvePnP
_POSE_IDX = [4, 152, 33, 263, 61, 291]

# Corresponding 3D face model coordinates (mm, generic neutral face)
_FACE_3D = np.array([
    [0.0,    0.0,    0.0],     # 4   Nose tip
    [0.0,  -330.0,  -65.0],    # 152 Chin
    [-165.0, 170.0, -135.0],   # 33  Left eye outer corner
    [165.0,  170.0, -135.0],   # 263 Right eye outer corner
    [-150.0,-150.0, -125.0],   # 61  Left mouth corner
    [150.0, -150.0, -125.0],   # 291 Right mouth corner
], dtype=np.float64)

# Inner lip landmarks for Mouth Aspect Ratio
_LIP_UPPER = [82, 13, 312]    # three upper inner lip points
_LIP_LOWER = [87, 14, 317]    # three lower inner lip points
_LIP_LEFT  = 78
_LIP_RIGHT = 308


class LandmarkExtractor:
    """
    Extracts head-pose yaw and mouth aspect ratio from a single face crop.

    Usage:
        extractor = LandmarkExtractor()
        result    = extractor.extract(frame, bbox)
        # result → {"yaw": float, "mar": float}  or  None if no face found
        extractor.close()

    Context-manager form:
        with LandmarkExtractor() as ext:
            result = ext.extract(frame, bbox)
    """

    def __init__(self):
        if not _MP_AVAILABLE:
            raise ImportError(
                "mediapipe is not installed. "
                "Run: pip install mediapipe"
            )
        if not LANDMARKER_MODEL.exists():
            raise FileNotFoundError(
                f"MediaPipe model not found at {LANDMARKER_MODEL}. Download "
                "face_landmarker.task from the MediaPipe model page into models/."
            )
        # IMAGE mode: SORT owns tracking, each crop is processed independently
        self._mesh = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(LANDMARKER_MODEL)),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
        ))

    # ── public API ────────────────────────────────────────────────────────────

    def extract(self, frame: np.ndarray, bbox: list[int]) -> dict | None:
        """
        Run Face Mesh on the face crop defined by bbox.

        Parameters
        ----------
        frame : BGR frame (full resolution from YOLO/SORT pipeline)
        bbox  : [x1, y1, x2, y2] absolute pixel coordinates

        Returns
        -------
        {"yaw": float (degrees), "mar": float (0..~0.4)} or None on failure.
        """
        crop, offset = self._crop_with_padding(frame, bbox)
        if crop is None or crop.size == 0:
            return None

        crop_rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        result   = self._mesh.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
        )

        if not result.face_landmarks:
            return None

        lm   = result.face_landmarks[0]
        ch, cw = crop.shape[:2]

        # Convert normalized landmark coordinates to pixel coordinates in crop space
        pts = [(lm[i].x * cw, lm[i].y * ch) for i in range(len(lm))]

        yaw = self._estimate_yaw(pts, cw, ch)
        mar = self._compute_mar(pts)

        return {"yaw": round(yaw, 2), "mar": round(mar, 4)}

    def close(self):
        self._mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── internals ─────────────────────────────────────────────────────────────

    def _crop_with_padding(
        self, frame: np.ndarray, bbox: list[int], pad_ratio: float = 0.15
    ) -> tuple[np.ndarray | None, tuple[int, int]]:
        """
        Crop the face region from the full frame, adding a small padding margin
        so Face Mesh can see the chin and forehead which YOLO often clips.
        Returns (crop_bgr, (x_offset, y_offset)) or (None, (0,0)).
        """
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]

        pad = int(pad_ratio * max(x2 - x1, y2 - y1, 1))
        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(fw, x2 + pad)
        y2p = min(fh, y2 + pad)

        if x2p <= x1p or y2p <= y1p:
            return None, (0, 0)

        return frame[y1p:y2p, x1p:x2p], (x1p, y1p)

    def _estimate_yaw(
        self, pts: list[tuple[float, float]], img_w: int, img_h: int
    ) -> float:
        """
        Estimate head-pose yaw via the Perspective-n-Point (PnP) problem.

        Projects 6 known 3D face landmarks onto the 2D crop and solves for
        the camera-relative rotation.  The y-angle from RQ decomposition of
        the rotation matrix gives yaw (left-right head turn).
        """
        pts2d = np.array(
            [[pts[i][0], pts[i][1]] for i in _POSE_IDX],
            dtype=np.float64,
        )

        focal  = float(img_w)
        cam    = np.array(
            [[focal, 0, img_w / 2.0],
             [0, focal, img_h / 2.0],
             [0, 0, 1.0]],
            dtype=np.float64,
        )
        dist   = np.zeros((4, 1), dtype=np.float64)

        ok, rvec, _ = cv2.solvePnP(
            _FACE_3D, pts2d, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return 0.0

        rmat, _ = cv2.Rodrigues(rvec)

        # RQDecomp3x3 → (rx, ry, rz) Euler angles in degrees
        # ry (index 1) = yaw (rotation around vertical axis = left/right turn)
        angles = cv2.RQDecomp3x3(rmat)[0]
        return float(angles[1])

    def _compute_mar(self, pts: list[tuple[float, float]]) -> float:
        """
        Mouth Aspect Ratio (MAR) — adapted for MediaPipe 468 inner lip landmarks.

        MAR = mean vertical opening / horizontal mouth width.
        Near zero when closed; rises to ~0.1–0.3+ when speaking.
        """
        top   = np.mean([pts[i] for i in _LIP_UPPER], axis=0)
        bot   = np.mean([pts[i] for i in _LIP_LOWER], axis=0)
        left  = np.array(pts[_LIP_LEFT])
        right = np.array(pts[_LIP_RIGHT])

        vertical   = float(np.linalg.norm(top - bot))
        horizontal = float(np.linalg.norm(left - right))

        return vertical / (horizontal + 1e-6)
