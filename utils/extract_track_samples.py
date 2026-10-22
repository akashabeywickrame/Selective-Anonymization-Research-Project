# =============================================================================
# utils/extract_track_samples.py — Extract one face crop per track ID
# =============================================================================
#
# Usage:
#   python utils/extract_track_samples.py
#   python utils/extract_track_samples.py --category interaction
#   python utils/extract_track_samples.py --video video1.mp4
#
# For each unique (video, track_id) in the tracking log, this script finds
# the frame where the face is clearest (highest area_ratio = biggest / closest
# face in the frame), crops it with padding, and saves it as a PNG:
#
#   eval/track_samples/
#       video1__track0.png
#       video1__track2.png
#       ...
#
# Open that folder, look at the crops, then fill in:
#   eval/ground_truth/interactions_labels.csv
#   eval/ground_truth/bystander_labels.csv
#
# CSV format:
#   video, track_id, ground_truth
#   video1.mp4, 0, 0          <- 0 = bystander (should be blurred)
#   video1.mp4, 2, 1          <- 1 = interaction partner (should be preserved)
# =============================================================================

import csv
import cv2
import argparse
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    LOGS_DIR, DATA_DIR,
    BYSTANDER_INPUT_DIR, INTERACTION_INPUT_DIR,
)

# Output folder for face crops
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "eval" / "track_samples"

# How much to pad the bounding box when cropping (fraction of face size)
CROP_PADDING = 0.35


def _best_frame_per_track(log_path: Path) -> dict:
    """
    Read tracking_log.csv and, for each (video, track_id), return the row
    that has the highest area_ratio (clearest/closest view of the face).

    Returns
    -------
    dict keyed by (video_name, track_id) → row dict with:
        frame, x1, y1, x2, y2, category, area_ratio
    """
    best = {}

    with open(log_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["video"], int(row["face_id"]))
            area = float(row["area_ratio"]) if row["area_ratio"] else 0.0
            if key not in best or area > float(best[key]["area_ratio"]):
                best[key] = row

    return best


def _find_video(video_name: str, category: str) -> Path | None:
    """Locate the source video file given its name and category."""
    if category == "interaction":
        p = INTERACTION_INPUT_DIR / video_name
    elif category == "bystander":
        p = BYSTANDER_INPUT_DIR / video_name
    else:
        # Try both
        p = INTERACTION_INPUT_DIR / video_name
        if not p.exists():
            p = BYSTANDER_INPUT_DIR / video_name

    return p if p.exists() else None


def _crop_face(frame, x1, y1, x2, y2, padding=CROP_PADDING):
    """Return a padded crop of the face region, clamped to frame bounds."""
    h, w = frame.shape[:2]
    pw = int((x2 - x1) * padding)
    ph = int((y2 - y1) * padding)
    cx1 = max(0, x1 - pw)
    cy1 = max(0, y1 - ph)
    cx2 = min(w, x2 + pw)
    cy2 = min(h, y2 + ph)
    return frame[cy1:cy2, cx1:cx2]


def extract_samples(filter_video: str = None, filter_category: str = None):
    log_path = LOGS_DIR / "tracking_log.csv"
    if not log_path.exists():
        print(f"[Sampler] ERROR: tracking log not found at {log_path}")
        print("  Run main.py first to generate the log.")
        return

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    best_rows = _best_frame_per_track(log_path)
    print(f"[Sampler] Found {len(best_rows)} unique tracks in log")

    # Group by video so we only open each video file once
    by_video = defaultdict(list)
    for (video_name, track_id), row in best_rows.items():
        if filter_video and video_name != filter_video:
            continue
        if filter_category and row["category"] != filter_category:
            continue
        by_video[video_name].append((track_id, row))

    total_saved = 0

    for video_name, tracks in sorted(by_video.items()):
        # Find the source video
        category = tracks[0][1]["category"]
        video_path = _find_video(video_name, category)

        if video_path is None:
            print(f"[Sampler] WARNING: source video not found for {video_name} — skipping")
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[Sampler] WARNING: cannot open {video_path} — skipping")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"\n[Sampler] {video_name}  ({len(tracks)} tracks, {total_frames} frames)")

        # Sort tracks by frame number so we seek forward only
        tracks_sorted = sorted(tracks, key=lambda t: int(t[1]["frame"]))

        for track_id, row in tracks_sorted:
            frame_idx = int(row["frame"])
            x1 = int(row["x1"]); y1 = int(row["y1"])
            x2 = int(row["x2"]); y2 = int(row["y2"])
            system_label = int(row["interacting"])

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"  Track {track_id}: could not read frame {frame_idx} — skipped")
                continue

            crop = _crop_face(frame, x1, y1, x2, y2)
            if crop.size == 0:
                continue

            # Filename encodes video + track so it's unambiguous
            stem = Path(video_name).stem
            label_str = "PARTNER" if system_label else "BYSTANDER"
            out_name = f"{stem}__track{track_id}__{label_str}.png"
            out_path = SAMPLES_DIR / out_name

            cv2.imwrite(str(out_path), crop)
            print(f"  Track {track_id:>3}  frame {frame_idx:>5}  "
                  f"area={float(row['area_ratio']):.4f}  "
                  f"system={label_str:>9}  → {out_name}")
            total_saved += 1

        cap.release()

    print(f"\n[Sampler] Done — {total_saved} crops saved to: {SAMPLES_DIR}")
    print()
    print("="*60)
    print("NEXT STEP: Open the folder above and fill in ground truth.")
    print()
    print("The filename tells you the system's current decision:")
    print("  __PARTNER  → system kept this face visible")
    print("  __BYSTANDER → system blurred this face")
    print()
    print("You decide: is that correct?")
    print()
    print("Edit the CSV files in eval/ground_truth/:")
    print("  interactions_labels.csv  — for interaction category videos")
    print("  bystander_labels.csv     — for bystander category videos")
    print()
    print("CSV format (add one row per track):")
    print("  video,track_id,ground_truth")
    print("  video1.mp4,0,0    <- 0 = bystander (should be blurred)")
    print("  video1.mp4,2,1    <- 1 = partner   (should be preserved)")
    print("="*60)


# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract one face crop per track ID for ground-truth annotation"
    )
    parser.add_argument("--video",    type=str, default=None,
                        help="Process only this video (e.g. video1.mp4)")
    parser.add_argument("--category", choices=["interaction", "bystander"],
                        default=None, help="Filter by category")
    args = parser.parse_args()

    extract_samples(filter_video=args.video, filter_category=args.category)
