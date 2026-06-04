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
# Two design choices worth defending at review:
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
# =============================================================================

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    INTERACT_MIN_AREA_RATIO, INTERACT_MAX_CENTER_DIST,
    INTERACT_ENTER_FRAMES, INTERACT_EXIT_FRAMES, INTERACT_LATCH,
    INTERACT_BIG_AREA_RATIO,
)
from core.interaction_cues import compute_cues


class _TrackState:
    """Rolling interaction state for a single track_id."""

    __slots__ = ("interacting", "enter_streak", "exit_streak")

    def __init__(self):
        self.interacting  = False   # current committed label
        self.enter_streak = 0       # consecutive frames cues say "interacting"
        self.exit_streak  = 0       # consecutive frames cues say "bystander"


class InteractionClassifier:
    """
    Classifies tracked faces as interaction partner vs. bystander.

    Usage:
        clf      = InteractionClassifier(frame_size=(w, h))
        decided  = clf.classify(tracks)   # tracks from FaceTracker.update()

    Each returned dict extends the input track with:
        cues         : {"area_ratio", "center_dist"}
        interacting  : bool   — True = preserve, False = anonymize
    """

    def __init__(self, frame_size: tuple[int, int]):
        self.frame_size = frame_size
        self._states: dict[int, _TrackState] = {}

    # ── single-frame instantaneous test of the geometric cues ────────────────
    def _cues_say_interacting(self, cues: dict) -> bool:
        return (
            cues["area_ratio"]  >= INTERACT_MIN_AREA_RATIO and
            cues["center_dist"] <= INTERACT_MAX_CENTER_DIST
        )

    def classify(self, tracks: list[dict]) -> list[dict]:
        results = []
        seen_ids = set()

        for t in tracks:
            tid  = t["track_id"]
            seen_ids.add(tid)

            cues = compute_cues(t["bbox"], self.frame_size)
            state = self._states.setdefault(tid, _TrackState())

            # ── update the hysteresis streaks ────────────────────────────────
            if self._cues_say_interacting(cues):
                state.enter_streak += 1
                state.exit_streak   = 0
            else:
                state.exit_streak  += 1
                state.enter_streak  = 0

            # ── commit a label change ────────────────────────────────────────
            # A very large face is confirmed instantly (proximity dominates);
            # otherwise the cues must hold for ENTER_FRAMES consecutive frames.
            big_face = cues["area_ratio"] >= INTERACT_BIG_AREA_RATIO

            if not state.interacting and (
                big_face or state.enter_streak >= INTERACT_ENTER_FRAMES
            ):
                state.interacting = True
            elif state.interacting and not INTERACT_LATCH and \
                    state.exit_streak >= INTERACT_EXIT_FRAMES:
                # only revert when latching is OFF — otherwise partner is sticky
                state.interacting = False

            out = dict(t)
            out["cues"]        = cues
            out["interacting"] = state.interacting
            results.append(out)

        # ── forget tracks that have disappeared (free memory, reset IDs) ─────
        for tid in list(self._states.keys()):
            if tid not in seen_ids:
                del self._states[tid]

        return results
