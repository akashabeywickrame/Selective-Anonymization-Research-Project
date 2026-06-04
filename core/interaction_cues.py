# =============================================================================
# core/interaction_cues.py — Egocentric interaction cues from face geometry
# =============================================================================
#
# Stage 3a of the pipeline (the "logic layer").
#
# Given a tracked face, compute the egocentric cues the decision layer uses to
# tell an INTERACTION PARTNER apart from an incidental BYSTANDER.
#
# Phase 2a cues (bounding-box only — no landmarks needed):
#   • proximity  — how much of the frame the face covers (area ratio)
#   • centrality — how close the face is to the optical center (center bias)
#
# Phase 2b cues (require MediaPipe Face Mesh landmarks — added later):
#   • orientation — head-pose yaw/pitch (looking at wearer vs. looking away)
#   • v_vad       — mouth movement over a temporal window (is the person talking)
#
# The proposal's "center bias as attention" assumption lives here: the wearer
# naturally points their head at whomever they are interacting with, so a large,
# centered face is a strong proxy for an active conversation partner.
# =============================================================================

import math


def compute_cues(bbox: list[int], frame_size: tuple[int, int]) -> dict:
    """
    Compute egocentric interaction cues for a single tracked face.

    Parameters
    ----------
    bbox       : [x1, y1, x2, y2] in absolute pixel coordinates
    frame_size : (width, height) of the frame the bbox lives in

    Returns
    -------
    dict with keys:
        area_ratio   : float  — face area / frame area        (0.0 … 1.0)
        center_dist  : float  — normalised distance of face center from frame
                                center, where 0.0 = dead center and 1.0 = corner
    """
    fw, fh = frame_size
    x1, y1, x2, y2 = bbox

    # ── proximity: fraction of the frame the face occupies ───────────────────
    face_w = max(0, x2 - x1)
    face_h = max(0, y2 - y1)
    face_area  = face_w * face_h
    frame_area = fw * fh
    area_ratio = face_area / frame_area if frame_area > 0 else 0.0

    # ── centrality: distance of the face center from the optical center ──────
    face_cx = (x1 + x2) / 2.0
    face_cy = (y1 + y2) / 2.0
    frame_cx = fw / 2.0
    frame_cy = fh / 2.0

    dx = face_cx - frame_cx
    dy = face_cy - frame_cy
    # normalise by the half-diagonal so the corner of the frame maps to ~1.0
    half_diag = math.hypot(frame_cx, frame_cy)
    center_dist = math.hypot(dx, dy) / half_diag if half_diag > 0 else 0.0

    return {
        "area_ratio" : round(area_ratio, 5),
        "center_dist": round(center_dist, 4),
    }
