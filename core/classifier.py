# =============================================================================
# core/classifier.py — Interaction-aware decision layer
# =============================================================================
#
# Stage 3b of the pipeline.
#
# Takes the per-face cues from interaction_cues.py and decides, for each tracked
# face ID, whether it is an INTERACTION PARTNER (preserve) or a BYSTANDER
# (anonymize) — this binary decision IS the research contribution.
#
# Three design choices worth defending at review:
#
#   1. Per-track temporal state. Because faces carry a stable track_id from the
#      SORT tracker, the classifier keeps a short history per ID and only flips a
#      face's label after the cue holds for several consecutive frames
#      (hysteresis). This stops a single noisy frame from un-blurring a bystander
#      or blurring a conversation partner mid-sentence.
#
#   2. "Bystander by default." A face is treated as a bystander unless it earns
#      interaction-partner status. This is the privacy-safe default: when unsure,
#      protect. It biases the system toward a high Privacy Success Rate (PSR).
#
#   3. Landmark cues as veto + boost. Head-pose yaw acts as a hard veto — a face
#      turned away from the camera wearer cannot be a conversation partner
#      regardless of proximity. V-VAD (mouth aspect ratio over a rolling window)
#      acts as a soft boost — a visibly talking face qualifies as a partner even
#      if it is slightly off the optical center.
# =============================================================================

from collections import deque
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    INTERACT_MIN_AREA_RATIO, INTERACT_MAX_CENTER_DIST,
    INTERACT_ENTER_FRAMES, INTERACT_EXIT_FRAMES, INTERACT_LATCH,
    INTERACT_BIG_AREA_RATIO,
    LANDMARK_ENABLED,
    INTERACT_MAX_YAW, VVAD_WINDOW_FRAMES, VVAD_MIN_MAR, VVAD_MIN_ACTIVE_RATIO,
)
from core.interaction_cues import compute_cues


class _TrackState:
    """Rolling interaction state for a single track_id."""

    __slots__ = ("interacting", "enter_streak", "exit_streak", "mar_history")

    def __init__(self):
        self.interacting  = False              # current committed label
        self.enter_streak = 0                  # consecutive frames cues say "interacting"
        self.exit_streak  = 0                  # consecutive frames cues say "bystander"
        self.mar_history  = deque(maxlen=VVAD_WINDOW_FRAMES)  # rolling MAR window


class InteractionClassifier:
    """
    Classifies tracked faces as interaction partner vs. bystander.

    Usage:
        clf      = InteractionClassifier(frame_size=(w, h))
        decided  = clf.classify(tracks, frame, extractor)

    Each returned dict extends the input track with:
        cues         : {"area_ratio", "center_dist", "yaw", "mar"}
        interacting  : bool   — True = preserve, False = anonymize
    """

    def __init__(self, frame_size: tuple[int, int]):
        self.frame_size = frame_size
        self._states: dict[int, _TrackState] = {}

    # ── single-frame instantaneous decision ───────────────────────────────────

    def _cues_say_interacting(self, cues: dict, state: _TrackState) -> bool:
        """
        Combine Phase 2a geometry cues with Phase 2b landmark cues.

        Logic (evaluated in priority order):
          1. Yaw veto   — if abs(yaw) > MAX_YAW the person is not facing the
                          wearer; classify as bystander regardless of proximity.
          2. Proximity + centrality — the Phase 2a rule: close AND centered.
          3. V-VAD boost — if the person is visibly talking they qualify as a
                           partner even if slightly off-center (center_dist
                           threshold is relaxed by 20%).
        """
        area_ratio  = cues["area_ratio"]
        center_dist = cues["center_dist"]
        yaw         = cues["yaw"]   # None when landmarks unavailable
        mar         = cues["mar"]   # None when landmarks unavailable

        # 1. Head-pose veto (only when landmark data is available)
        if yaw is not None and abs(yaw) > INTERACT_MAX_YAW:
            return False

        # 2. Base proximity + centrality rule (Phase 2a)
        proximity_ok  = area_ratio  >= INTERACT_MIN_AREA_RATIO
        centrality_ok = center_dist <= INTERACT_MAX_CENTER_DIST
        geometry_pass = proximity_ok and centrality_ok

        # 3. V-VAD boost: update rolling window, check if actively talking
        if mar is not None:
            state.mar_history.append(mar)

        talking = False
        if len(state.mar_history) >= max(1, VVAD_WINDOW_FRAMES // 3):
            active_frames = sum(1 for m in state.mar_history if m >= VVAD_MIN_MAR)
            active_ratio  = active_frames / len(state.mar_history)
            talking = active_ratio >= VVAD_MIN_ACTIVE_RATIO

        # Talking person qualifies even if slightly off-center
        if talking and proximity_ok and center_dist <= INTERACT_MAX_CENTER_DIST * 1.2:
            return True

        return geometry_pass

    def classify(
        self,
        tracks: list[dict],
        frame: "np.ndarray | None" = None,
        extractor: "object | None" = None,
    ) -> list[dict]:
        """
        Parameters
        ----------
        tracks    : list of track dicts from FaceTracker.update()
        frame     : current BGR frame — needed for landmark extraction.
                    Pass None to skip landmarks (Phase 2a only).
        extractor : LandmarkExtractor instance, or None to skip landmarks.
        """
        results  = []
        seen_ids = set()

        use_landmarks = (
            LANDMARK_ENABLED
            and frame is not None
            and extractor is not None
        )

        for t in tracks:
            tid  = t["track_id"]
            seen_ids.add(tid)

            # Extract landmarks for this face crop (Phase 2b)
            lm = None
            if use_landmarks:
                try:
                    lm = extractor.extract(frame, t["bbox"])
                except Exception:
                    lm = None   # graceful fallback to Phase 2a

            cues  = compute_cues(t["bbox"], self.frame_size, landmarks=lm)
            state = self._states.setdefault(tid, _TrackState())

            # ── update hysteresis streaks ─────────────────────────────────────
            if self._cues_say_interacting(cues, state):
                state.enter_streak += 1
                state.exit_streak   = 0
            else:
                state.exit_streak  += 1
                state.enter_streak  = 0

            # ── commit label change ───────────────────────────────────────────
            # A very large face is confirmed instantly (proximity dominates);
            # otherwise cues must hold for ENTER_FRAMES consecutive frames.
            big_face = cues["area_ratio"] >= INTERACT_BIG_AREA_RATIO

            if not state.interacting and (
                big_face or state.enter_streak >= INTERACT_ENTER_FRAMES
            ):
                state.interacting = True
            elif state.interacting and not INTERACT_LATCH and \
                    state.exit_streak >= INTERACT_EXIT_FRAMES:
                state.interacting = False

            out = dict(t)
            out["cues"]        = cues
            out["interacting"] = state.interacting
            results.append(out)

        # ── forget disappeared tracks ─────────────────────────────────────────
        for tid in list(self._states.keys()):
            if tid not in seen_ids:
                del self._states[tid]

        return results
