# =============================================================================
# utils/restore_outputs_from_log.py — Rebuild broken output videos from the log
# =============================================================================
#
# Some output videos were left unfinalised ("moov atom not found") when a run
# was interrupted; main.py's resume logic then skipped them because the files
# existed. The tracking log, however, is complete for those videos.
#
# This script re-renders the four outputs (selective, blanket, compare, debug)
# for one video using the decisions ALREADY RECORDED in the log, so the result
# is exactly what the Phase 2b system decided. Unlike rerender_from_log.py it
# does not re-run the classifier (which would drop the landmark cues), and it
# keys tracks by category as well as video name (bystander/video1 and
# interaction/video1 are different clips).
#
# Usage:
#   python -m utils.restore_outputs_from_log interaction video3.mp4
# =============================================================================

import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import imageio_ffmpeg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    LOGS_DIR, INTERACTION_INPUT_DIR, BYSTANDER_INPUT_DIR,
    INTERACTION_OUTPUT_DIR, BYSTANDER_OUTPUT_DIR,
)
from core.anonymizer import SelectiveAnonymizer, BlanketAnonymizer
from core.visualizer import Visualizer
from main import build_comparison
from utils.video_loader import VideoLoader

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def _writer(path, w, h, fps):
    return subprocess.Popen([
        FFMPEG, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.3f}", "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(path),
    ], stdin=subprocess.PIPE)


def _float(v):
    return float(v) if v not in ("", None) else None


def load_decisions(category, video):
    """frame -> list of decided-track dicts, as the classifier produced them."""
    by_frame = defaultdict(list)
    with open(LOGS_DIR / "tracking_log.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["category"] != category or r["video"] != video:
                continue
            by_frame[int(r["frame"])].append({
                "track_id": int(r["face_id"]),
                "bbox": [int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])],
                "confidence": _float(r["confidence"]),
                "interacting": r["interacting"] == "1",
                "cues": {k: _float(r[k]) for k in ("area_ratio", "center_dist", "yaw", "mar")},
            })
    return by_frame


def restore(category, video):
    src_dir = INTERACTION_INPUT_DIR if category == "interaction" else BYSTANDER_INPUT_DIR
    out_dir = INTERACTION_OUTPUT_DIR if category == "interaction" else BYSTANDER_OUTPUT_DIR
    stem = Path(video).stem
    decisions = load_decisions(category, video)
    if not decisions:
        sys.exit(f"No log rows for {category}/{video}")

    selective, blanket, vis = SelectiveAnonymizer(), BlanketAnonymizer(), Visualizer()
    with VideoLoader(src_dir / video) as loader:
        w, h = loader.frame_size
        fps = loader.fps
        tmp = {name: out_dir / f"{stem}_{name}.restoring.mp4"
               for name in ("selective", "blanket", "compare", "debug")}
        writers = {
            "selective": _writer(tmp["selective"], w, h, fps),
            "blanket":   _writer(tmp["blanket"], w, h, fps),
            "compare":   _writer(tmp["compare"], w * 2 + 2, h, fps),
            "debug":     _writer(tmp["debug"], w, h, fps),
        }
        n = 0
        for fi, frame in enumerate(loader):
            decided = decisions.get(fi, [])
            sel = selective.apply(frame, decided)
            bla = blanket.apply(frame, decided)
            writers["selective"].stdin.write(sel.tobytes())
            writers["blanket"].stdin.write(bla.tobytes())
            writers["compare"].stdin.write(build_comparison(bla, sel).tobytes())
            writers["debug"].stdin.write(vis.draw_decisions(frame, decided).tobytes())
            n += 1
            if n % 900 == 0:
                print(f"  {category}/{video}: frame {n}")
    for p in writers.values():
        p.stdin.close()
        p.wait()

    # Replace the broken files only after every new one is complete
    for name, path in tmp.items():
        path.replace(out_dir / f"{stem}_{name}.mp4")
    print(f"[Restore] {category}/{video}: {n} frames -> {out_dir}/{stem}_*.mp4")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("interaction", "bystander"):
        sys.exit("Usage: python -m utils.restore_outputs_from_log <interaction|bystander> <video.mp4>")
    restore(sys.argv[1], sys.argv[2])
