# =============================================================================
# utils/rerender_from_log.py — Re-render outputs from the CSV log (no detection)
# =============================================================================
#
# The expensive part of the pipeline is YOLO face detection (~2 FPS on CPU).
# But every detection is already saved in logs/tracking_log.csv with its bbox
# per frame. So when we only change the DECISION LOGIC (classifier thresholds,
# latching, anonymization), we can re-render the output videos straight from the
# log + the original frames — in minutes instead of re-detecting for ~75 min.
#
# It reuses the REAL InteractionClassifier and anonymizers, so the result is
# identical to what a full re-run would produce, and writes H.264 directly via
# the bundled ffmpeg so the videos play on Windows.
#
# Usage:
#   python -m utils.rerender_from_log data/interaction/video1.mp4 video1.mp4
#       arg1 = path to the original source video
#       arg2 = the `video` name as stored in the CSV (defaults to filename)
# =============================================================================

import sys
import csv
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import imageio_ffmpeg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import LOGS_DIR, OUTPUT_DIR
from core.classifier import InteractionClassifier
from core.anonymizer import SelectiveAnonymizer, BlanketAnonymizer
from core.visualizer import Visualizer

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def _h264_writer(path: Path, w: int, h: int, fps: float):
    """Open an ffmpeg subprocess that accepts raw BGR frames and writes H.264."""
    return subprocess.Popen(
        [FFMPEG, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         str(path)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _banner(img, text, color):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 32), (0, 0, 0), cv2.FILLED)
    cv2.putText(out, text, (10, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return out


def load_tracks_by_frame(video_name: str):
    """Read the CSV and group detections by frame index."""
    log = LOGS_DIR / "tracking_log.csv"
    by_frame = defaultdict(list)
    for r in csv.DictReader(open(log)):
        if r["video"] != video_name:
            continue
        by_frame[int(r["frame"])].append({
            "track_id"  : int(r["face_id"]),
            "bbox"      : [int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])],
            "confidence": float(r["confidence"]),
        })
    return by_frame


def rerender(src_video: Path, video_name: str):
    cap = cv2.VideoCapture(str(src_video))
    if not cap.isOpened():
        print(f"[rerender] cannot open {src_video}")
        return

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # NOTE: the log was produced AFTER the loader resized frames, so honour the
    # same resize here (the pipeline resizes to FRAME_WIDTH x FRAME_HEIGHT).
    from config import FRAME_WIDTH, FRAME_HEIGHT
    if FRAME_WIDTH > 0 and FRAME_HEIGHT > 0:
        w, h = FRAME_WIDTH, FRAME_HEIGHT

    by_frame   = load_tracks_by_frame(video_name)
    classifier = InteractionClassifier(frame_size=(w, h))
    selective  = SelectiveAnonymizer()
    blanket    = BlanketAnonymizer()
    vis        = Visualizer()

    out_dir = OUTPUT_DIR / src_video.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src_video.stem

    sel_w = _h264_writer(out_dir / f"{stem}_selective_h264.mp4", w, h, fps)
    cmp_w = _h264_writer(out_dir / f"{stem}_compare_h264.mp4", w * 2, h, fps)
    dbg_w = _h264_writer(out_dir / f"{stem}_debug_h264.mp4", w, h, fps)

    idx = 0
    n_partner_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame.shape[1], frame.shape[0]) != (w, h):
            frame = cv2.resize(frame, (w, h))

        tracks  = by_frame.get(idx, [])
        decided = classifier.classify(tracks)

        sel = selective.apply(frame, decided)
        bla = blanket.apply(frame, decided)
        dbg = vis.draw_decisions(frame, decided)

        combo = cv2.hconcat([
            _banner(bla, "BASELINE: Blanket Blur", (0, 165, 255)),
            _banner(sel, "PROPOSED: Selective",    (0, 255, 0)),
        ])
        combo[:, w-1:w+1] = 255

        sel_w.stdin.write(sel.tobytes())
        cmp_w.stdin.write(combo.tobytes())
        dbg_w.stdin.write(dbg.tobytes())

        if any(d.get("interacting") for d in decided):
            n_partner_frames += 1
        idx += 1
        if idx % 500 == 0:
            print(f"  re-rendered {idx} frames…")

    cap.release()
    for p in (sel_w, cmp_w, dbg_w):
        p.stdin.close(); p.wait()

    print(f"\n[rerender] Done — {idx} frames")
    print(f"  Frames keeping a partner unblurred: {n_partner_frames} "
          f"({100*n_partner_frames/max(idx,1):.1f}%)")
    print(f"  Outputs → {out_dir}/{stem}_*_h264.mp4")


def main():
    if len(sys.argv) < 2:
        print("usage: python -m utils.rerender_from_log <src_video> [csv_video_name]")
        return
    src = Path(sys.argv[1])
    name = sys.argv[2] if len(sys.argv) > 2 else src.name
    rerender(src, name)


if __name__ == "__main__":
    main()
