# =============================================================================
# eval/verify_clip_predictions.py — Read the system decision from its OUTPUT
# =============================================================================
#
# Usage:
#   python eval/verify_clip_predictions.py
#
# WHY
# ---
# generate_clip_package.py takes each clip's system decision from the tracking
# log (the face was flagged "interacting" in >= 50% of the window). This script
# checks what the system ACTUALLY produced: it opens the selective output video
# and measures, frame by frame, whether the target face was blurred. That covers
# all four layers end to end (detection -> tracking -> classification -> blur)
# and catches cases the log cannot, e.g. a bystander's blur box spilling over a
# partner's face.
#
# HOW (no thresholds to tune)
# ---------------------------
# For each frame in the clip window where the target face is logged, crop the
# inner part of its box from three frames of the same index:
#     original video (resized to the processing size, as the pipeline does)
#     *_selective.mp4  (the system's output)
#     *_blanket.mp4    (every detected face blurred)
# The face counts as BLURRED in that frame if the selective crop is closer to
# the blanket crop than to the original crop (mean absolute difference).
#
# Clip decision = face left unblurred in >= 50% of its measured frames.
# Results are written back into eval/clip_package/clip_map.csv:
#     frac_unblurred_log, system_prediction_log   (from the tracking log)
#     frac_unblurred_video, system_prediction      (verified from the output)
# =============================================================================

import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    LOGS_DIR, INTERACTION_INPUT_DIR, BYSTANDER_INPUT_DIR,
    INTERACTION_OUTPUT_DIR, BYSTANDER_OUTPUT_DIR, FRAME_WIDTH, FRAME_HEIGHT,
)

LOG_PATH = LOGS_DIR / "tracking_log.csv"
MAP_FILE = Path(__file__).resolve().parent / "clip_package" / "clip_map.csv"
SHRINK   = 0.15     # ignore the outer 15% of the box on each side
MIN_SIDE = 8        # skip crops smaller than this (pixels)


def _boxes(wanted):
    """(category, video, track) -> {frame: (x1, y1, x2, y2)} for wanted tracks."""
    out = defaultdict(dict)
    with open(LOG_PATH, newline="") as f:
        for r in csv.DictReader(f):
            k = (r["category"], r["video"], int(r["face_id"]))
            if k in wanted:
                out[k][int(r["frame"])] = (int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"]))
    return out


def _crop(img, box):
    x1, y1, x2, y2 = box
    dx, dy = int((x2 - x1) * SHRINK), int((y2 - y1) * SHRINK)
    x1, y1, x2, y2 = max(0, x1 + dx), max(0, y1 + dy), x2 - dx, y2 - dy
    if x2 - x1 < MIN_SIDE or y2 - y1 < MIN_SIDE:
        return None
    return img[y1:y2, x1:x2].astype(np.int16)


def _measure(row, boxes):
    cat, vid = row["category"], row["video"]
    stem = Path(vid).stem
    src_dir = INTERACTION_INPUT_DIR if cat == "interaction" else BYSTANDER_INPUT_DIR
    out_dir = INTERACTION_OUTPUT_DIR if cat == "interaction" else BYSTANDER_OUTPUT_DIR
    caps = {
        "orig": cv2.VideoCapture(str(src_dir / vid)),
        "sel":  cv2.VideoCapture(str(out_dir / f"{stem}_selective.mp4")),
        "blk":  cv2.VideoCapture(str(out_dir / f"{stem}_blanket.mp4")),
    }
    start, end = int(row["start_frame"]), int(row["end_frame"])
    for c in caps.values():
        c.set(cv2.CAP_PROP_POS_FRAMES, start)

    unblurred = measured = 0
    for fi in range(start, end + 1):
        frames = {}
        for name, c in caps.items():
            ok, img = c.read()
            if not ok:
                break
            frames[name] = img
        if len(frames) < 3:
            break
        if fi not in boxes:
            continue
        orig = frames["orig"]
        if orig.shape[1] != FRAME_WIDTH or orig.shape[0] != FRAME_HEIGHT:
            orig = cv2.resize(orig, (FRAME_WIDTH, FRAME_HEIGHT))
        o, s, b = (_crop(x, boxes[fi]) for x in (orig, frames["sel"], frames["blk"]))
        if o is None:
            continue
        d_orig = np.abs(s - o).mean()
        d_blk = np.abs(s - b).mean()
        measured += 1
        unblurred += d_orig < d_blk
    for c in caps.values():
        c.release()
    return (unblurred / measured if measured else None), measured


def main():
    rows = list(csv.DictReader(open(MAP_FILE, newline="")))
    wanted = {(r["category"], r["video"], int(r["track_id"])) for r in rows}
    print(f"[Verify] Loading boxes for {len(wanted)} tracks...")
    boxes = _boxes(wanted)

    changed = []
    for r in rows:
        # keep the log-based decision; first run stores it under *_log
        if "system_prediction_log" not in r:
            r["frac_unblurred_log"] = r.pop("frac_unblurred")
            r["system_prediction_log"] = r["system_prediction"]
        frac, n = _measure(r, boxes[(r["category"], r["video"], int(r["track_id"]))])
        r["frames_measured"] = n
        r["frac_unblurred_video"] = "" if frac is None else round(frac, 3)
        if frac is None:
            r["system_prediction"] = r["system_prediction_log"]   # nothing to measure
        else:
            r["system_prediction"] = 1 if frac >= 0.5 else 0
        if str(r["system_prediction"]) != str(r["system_prediction_log"]):
            changed.append(r)
        print(f"  {r['clip_id']}  log={r['frac_unblurred_log']:>5}  video={r['frac_unblurred_video']!s:>5}"
              f"  n={n:>3}  pred {r['system_prediction_log']}->{r['system_prediction']}")

    fields = [k for k in rows[0] if k not in ("system_prediction",)] + ["system_prediction"]
    with open(MAP_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\n[Verify] {len(rows)} clips checked; {len(changed)} decision(s) differ from the log:")
    for r in changed:
        print(f"  {r['clip_id']}  {r['category']}/{r['video']} track {r['track_id']}: "
              f"log {r['frac_unblurred_log']} vs video {r['frac_unblurred_video']}")
    print(f"  Updated {MAP_FILE}")


if __name__ == "__main__":
    main()
