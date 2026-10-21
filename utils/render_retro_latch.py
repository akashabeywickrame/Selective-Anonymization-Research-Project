# =============================================================================
# utils/render_retro_latch.py — Offline "once a partner, always a partner"
# =============================================================================
#
# The live pipeline latches forward in time: once a track is confirmed as an
# interaction partner it is never re-blurred. But the frames BEFORE
# confirmation (while the cues were still accumulating) stay blurred.
#
# For recorded video the whole track is known, so this second pass preserves a
# confirmed partner for every frame of its track, including the frames before
# confirmation. It reads the decisions already recorded in the tracking log
# (no re-detection). With FRIEND_GROUP_ENABLED it also preserves people who stay
# in view alongside a confirmed partner (friends who are not speaking). Writes:
#     output/<category>/<video>_selective_retro.mp4
#
# Usage:
#   python -m utils.render_retro_latch interaction video10.mp4
#   python -m utils.render_retro_latch --all
# =============================================================================

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    INTERACTION_INPUT_DIR, BYSTANDER_INPUT_DIR,
    INTERACTION_OUTPUT_DIR, BYSTANDER_OUTPUT_DIR, INTERACT_RETRO_LATCH,
    FRIEND_GROUP_ENABLED, FRIEND_MIN_SECONDS, FRIEND_MIN_COPRESENCE,
)
from collections import defaultdict
from core.anonymizer import SelectiveAnonymizer
from utils.restore_outputs_from_log import load_decisions, _writer
from utils.video_loader import VideoLoader


def find_friends(decisions, partners, fps):
    """Tracks that stay in view long enough while a confirmed partner is present."""
    frames = defaultdict(set)
    for fi, rows in decisions.items():
        for d in rows:
            frames[d["track_id"]].add(fi)
    partner_frames = set().union(*(frames[t] for t in partners)) if partners else set()
    friends = set()
    for tid, f in frames.items():
        if tid in partners or len(f) / fps < FRIEND_MIN_SECONDS:
            continue
        if len(f & partner_frames) / len(f) >= FRIEND_MIN_COPRESENCE:
            friends.add(tid)
    return friends


def apply_retro_latch(decisions, fps=30.0):
    """Mark every frame of a track as interacting if the track was ever confirmed
    (or, with the friend-group rule, belongs to the wearer's group)."""
    partners = {d["track_id"] for rows in decisions.values() for d in rows if d["interacting"]}
    if FRIEND_GROUP_ENABLED:
        partners |= find_friends(decisions, partners, fps)
    for rows in decisions.values():
        for d in rows:
            if d["track_id"] in partners:
                d["interacting"] = True
    return partners


def render(category, video):
    src_dir = INTERACTION_INPUT_DIR if category == "interaction" else BYSTANDER_INPUT_DIR
    out_dir = INTERACTION_OUTPUT_DIR if category == "interaction" else BYSTANDER_OUTPUT_DIR
    decisions = load_decisions(category, video)
    if not decisions:
        print(f"[Retro] no log rows for {category}/{video}, skipping")
        return
    out = out_dir / f"{Path(video).stem}_selective_retro.mp4"
    tmp = out.with_suffix(".rendering.mp4")
    selective = SelectiveAnonymizer()
    with VideoLoader(src_dir / video) as loader:
        partners = apply_retro_latch(decisions, loader.fps) if INTERACT_RETRO_LATCH else set()
        w, h = loader.frame_size
        wr = _writer(tmp, w, h, loader.fps)
        n = 0
        for fi, frame in enumerate(loader):
            wr.stdin.write(selective.apply(frame, decisions.get(fi, [])).tobytes())
            n += 1
        wr.stdin.close()
        wr.wait()
    tmp.replace(out)
    print(f"[Retro] {category}/{video}: {n} frames, {len(partners)} partner track(s) -> {out}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--all"]:
        for cat, d in (("bystander", BYSTANDER_INPUT_DIR), ("interaction", INTERACTION_INPUT_DIR)):
            for vp in sorted(d.glob("*.mp4"), key=lambda p: int(p.stem[5:])):
                render(cat, vp.name)
    elif len(sys.argv) == 3 and sys.argv[1] in ("interaction", "bystander"):
        render(sys.argv[1], sys.argv[2])
    else:
        sys.exit("Usage: python -m utils.render_retro_latch <interaction|bystander> <video.mp4> | --all")
