# =============================================================================
# main.py — Phase 1: Face Detection + Tracking Pipeline
# =============================================================================
#
# Usage:
#   python main.py                          # process ALL videos in data/
#   python main.py --category bystander     # process only bystander videos
#   python main.py --category interaction   # process only interaction videos
#   python main.py --video data/bystander/video1.mp4  # single video
# =============================================================================

import cv2
import numpy as np
import time
import argparse
import csv
from pathlib import Path

from config import (
    BYSTANDER_INPUT_DIR, INTERACTION_INPUT_DIR,
    BYSTANDER_OUTPUT_DIR, INTERACTION_OUTPUT_DIR,
    LOGS_DIR, DEBUG_MODE, FOURCC, TARGET_FPS
)
from core.detector   import FaceDetector
from core.tracker    import FaceTracker
from core.classifier import InteractionClassifier
from core.anonymizer import SelectiveAnonymizer, BlanketAnonymizer
from core.visualizer import Visualizer
from utils.video_loader import VideoLoader


# =============================================================================
# Directory setup
# =============================================================================

def setup_dirs():
    for d in [
        BYSTANDER_INPUT_DIR, INTERACTION_INPUT_DIR,
        BYSTANDER_OUTPUT_DIR, INTERACTION_OUTPUT_DIR,
        LOGS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CSV logger
# =============================================================================

class TrackingLogger:
    """
    Writes per-frame tracking data to a CSV file.

    Columns:
        video, category, frame, face_id, x1, y1, x2, y2, confidence,
        area_ratio, center_dist, interacting
    """

    def __init__(self, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file   = open(log_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "video", "category", "frame",
            "face_id", "x1", "y1", "x2", "y2", "confidence",
            "area_ratio", "center_dist", "interacting",
        ])
        print(f"[Logger] Writing to {log_path}")

    def log(self, video_name: str, category: str, frame_idx: int, tracks: list):
        for t in tracks:
            x1, y1, x2, y2 = t["bbox"]
            cues = t.get("cues", {})
            self._writer.writerow([
                video_name, category, frame_idx,
                t["track_id"], x1, y1, x2, y2, t["confidence"],
                cues.get("area_ratio", ""), cues.get("center_dist", ""),
                int(t.get("interacting", False)),
            ])

    def close(self):
        self._file.close()


# =============================================================================
# Side-by-side comparison frame
# =============================================================================

def build_comparison(blanket_frame, selective_frame):
    """
    Stack the baseline (blanket-blur) and proposed (selective) frames
    horizontally with a labelled banner on each, so the privacy/utility
    difference is obvious in a single clip.

        [  BASELINE: Blanket Blur  |  PROPOSED: Selective  ]
    """
    h = blanket_frame.shape[0]

    def _banner(img, text, color):
        out = img.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 32), (0, 0, 0), cv2.FILLED)
        cv2.putText(
            out, text, (10, 23),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
        )
        return out

    left  = _banner(blanket_frame,   "BASELINE: Blanket Blur", (0, 165, 255))
    right = _banner(selective_frame, "PROPOSED: Selective",    (0, 255, 0))

    # 2px white divider between the two panes
    divider = np.full((h, 2, 3), 255, dtype=left.dtype)
    return cv2.hconcat([left, divider, right])


# =============================================================================
# Single video processor
# =============================================================================

def process_video(
    video_path : Path,
    output_dir : Path,
    category   : str,
    detector   : FaceDetector,
    visualizer : Visualizer,
    logger     : TrackingLogger,
):
    print(f"\n{'='*60}")
    print(f"[Pipeline] Processing : {video_path.name}  [{category}]")
    print(f"{'='*60}")

    with VideoLoader(video_path) as loader:
        w, h = loader.frame_size
        fps  = loader.fps

        # ── per-video pipeline components (state restarts for each video) ──────
        tracker    = FaceTracker(frame_size=(w, h))      # IDs restart from 0
        classifier = InteractionClassifier(frame_size=(w, h))
        selective  = SelectiveAnonymizer()
        blanket    = BlanketAnonymizer()

        # ── video writers ──────────────────────────────────────────────────────
        fourcc = cv2.VideoWriter_fourcc(*FOURCC)

        selective_path = output_dir / f"{video_path.stem}_selective.mp4"
        blanket_path   = output_dir / f"{video_path.stem}_blanket.mp4"
        compare_path   = output_dir / f"{video_path.stem}_compare.mp4"
        debug_path     = output_dir / f"{video_path.stem}_debug.mp4"

        selective_writer = cv2.VideoWriter(str(selective_path), fourcc, fps, (w, h))
        blanket_writer   = cv2.VideoWriter(str(blanket_path),   fourcc, fps, (w, h))
        # side-by-side comparison: two panes (w each) + 2px divider
        compare_writer   = cv2.VideoWriter(str(compare_path), fourcc, fps, (w * 2 + 2, h))
        debug_writer     = cv2.VideoWriter(str(debug_path), fourcc, fps, (w, h)) \
                           if DEBUG_MODE else None

        # ── metrics ────────────────────────────────────────────────────────────
        frame_idx       = 0
        total_det       = 0
        total_tracked   = 0
        total_partners  = 0
        total_bystander = 0
        t_start         = time.perf_counter()

        for frame in loader:
            # 1. Detect
            detections = detector.detect(frame)

            # 2. Track  → stable face IDs
            tracks = tracker.update(detections, frame)

            # 3. Classify → interacting partner vs. bystander (the contribution)
            decided = classifier.classify(tracks)

            # 4. Log per-face decision + cues
            logger.log(video_path.name, category, frame_idx, decided)

            # 5. Anonymize — proposed (selective) and baseline (blanket)
            selective_frame = selective.apply(frame, decided)
            blanket_frame   = blanket.apply(frame, decided)

            selective_writer.write(selective_frame)
            blanket_writer.write(blanket_frame)

            # 6. Side-by-side comparison: [ baseline | proposed ]
            comparison = build_comparison(blanket_frame, selective_frame)
            compare_writer.write(comparison)

            # 7. Debug overlay (green = partner, red = bystander)
            if debug_writer:
                debug_writer.write(visualizer.draw_decisions(frame, decided))

            # 8. Tally + console progress
            n_partner = sum(1 for d in decided if d.get("interacting"))
            total_det       += len(detections)
            total_tracked   += len(decided)
            total_partners  += n_partner
            total_bystander += len(decided) - n_partner
            frame_idx       += 1

            if frame_idx % 30 == 0:
                elapsed = time.perf_counter() - t_start
                cur_fps = frame_idx / elapsed if elapsed > 0 else 0
                print(
                    f"  Frame {frame_idx:>5} | "
                    f"Det: {len(detections):>2} | "
                    f"Partners: {n_partner:>2} | "
                    f"Bystanders: {len(decided) - n_partner:>2} | "
                    f"FPS: {cur_fps:>5.1f}"
                )

        # ── cleanup ────────────────────────────────────────────────────────────
        selective_writer.release()
        blanket_writer.release()
        compare_writer.release()
        if debug_writer:
            debug_writer.release()

        elapsed = time.perf_counter() - t_start
        avg_fps = frame_idx / elapsed if elapsed > 0 else 0

        print(f"\n[Pipeline] Done — {video_path.name}")
        print(f"  Frames processed   : {frame_idx}")
        print(f"  Avg FPS            : {avg_fps:.1f}")
        print(f"  Total detections   : {total_det}")
        print(f"  Partner faces      : {total_partners}")
        print(f"  Bystander faces    : {total_bystander}")
        print(f"  Selective output   : {selective_path}")
        print(f"  Blanket  output    : {blanket_path}")
        print(f"  Comparison output  : {compare_path}")
        if DEBUG_MODE:
            print(f"  Debug output       : {debug_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 1 — Face Tracking Pipeline")
    parser.add_argument(
        "--category",
        choices=["bystander", "interaction", "all"],
        default="all",
        help="Which video category to process (default: all)",
    )
    parser.add_argument(
    "--video",
    type=str,
    default=None,
    help="Path to a single video file (overrides --category)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show real-time tracking window while processing",
    )
    args = parser.parse_args()

    setup_dirs()

    # ── shared instances (loaded once, reused across all videos) ───────────────
    detector   = FaceDetector()
    visualizer = Visualizer()

    # ── build video list ───────────────────────────────────────────────────────
    if args.video:
        video_path = Path(args.video)
        # infer category from folder name
        if "bystander" in str(video_path).lower():
            jobs = [(video_path, BYSTANDER_OUTPUT_DIR, "bystander")]
        else:
            jobs = [(video_path, INTERACTION_OUTPUT_DIR, "interaction")]
    else:
        jobs = []
        if args.category in ("bystander", "all"):
            for vp in sorted(BYSTANDER_INPUT_DIR.glob("*.mp4")):
                jobs.append((vp, BYSTANDER_OUTPUT_DIR, "bystander"))
        if args.category in ("interaction", "all"):
            for vp in sorted(INTERACTION_INPUT_DIR.glob("*.mp4")):
                jobs.append((vp, INTERACTION_OUTPUT_DIR, "interaction"))

    if not jobs:
        print("[Pipeline] No videos found. Place .mp4 files in data/bystander/ or data/interaction/")
        return

    print(f"[Pipeline] Found {len(jobs)} video(s) to process")

    # ── single shared CSV log for entire run ───────────────────────────────────
    log_path = LOGS_DIR / "tracking_log.csv"
    logger   = TrackingLogger(log_path)

    # ── process each video ─────────────────────────────────────────────────────
    pipeline_start = time.perf_counter()

    for video_path, output_dir, category in jobs:
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            process_video(
                video_path, output_dir, category,
                detector, visualizer, logger
            )
        except Exception as e:
            print(f"[Pipeline] ERROR on {video_path.name}: {e}")

    logger.close()

    total_elapsed = time.perf_counter() - pipeline_start
    print(f"\n[Pipeline] All done in {total_elapsed:.1f}s")
    print(f"[Pipeline] Tracking log → {log_path}")


if __name__ == "__main__":
    main()