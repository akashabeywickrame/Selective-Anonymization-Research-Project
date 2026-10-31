# =============================================================================
# eval/annotate.py — Interactive ground-truth annotation tool
# =============================================================================
#
# Usage:
#   python eval/annotate.py                        # annotate all videos in log
#   python eval/annotate.py --video video1.mp4     # annotate one video
#   python eval/annotate.py --sheet                # generate contact sheet only
#
# HOW IT WORKS
# ------------
# 1. Generates a contact sheet image (eval/track_samples/contact_sheet_<video>.png)
#    showing every unique track ID as a face crop — open this to see who is who.
#
# 2. Plays the debug video so you have full context while annotating.
#
# 3. For each track, shows the crop in a window and asks:
#      p = interaction partner (preserve — label 1)
#      b = bystander           (blur     — label 0)
#      s = skip (unsure)
#
# LABELLING RULE
# --------------
#   - If someone EVER talks to the wearer during the video → partner (p)
#   - Only bystander if they NEVER interact at any point
#   - Tracks already confirmed as partner by the system (LATCH) are pre-filled
#     as partner — you can override if wrong.
#
# Results are saved automatically to:
#   eval/ground_truth/interactions_labels.csv
#   eval/ground_truth/bystander_labels.csv
# =============================================================================

import csv
import cv2
import argparse
import sys
import os
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    LOGS_DIR, INTERACTION_INPUT_DIR, BYSTANDER_INPUT_DIR,
)

GROUND_TRUTH_DIR = Path(__file__).resolve().parent / "ground_truth"
SAMPLES_DIR      = Path(__file__).resolve().parent / "track_samples"
INTERACTIONS_GT  = GROUND_TRUTH_DIR / "interactions_labels.csv"
BYSTANDER_GT     = GROUND_TRUTH_DIR / "bystander_labels.csv"
LOG_PATH         = LOGS_DIR / "tracking_log.csv"

CROP_PADDING = 0.35
SHEET_COLS   = 8       # tracks per row in the contact sheet
CELL_SIZE    = 120     # px per cell in the contact sheet


# =============================================================================
# Load tracking log
# =============================================================================

def _load_log():
    """
    Returns:
      best_frame : {(video, track_id) -> row}  — best (largest) frame per track
      ever_partner : {(video, track_id) -> bool} — True if system ever said partner
      track_categories : {(video, track_id) -> category}
    """
    best_frame    = {}
    ever_partner  = defaultdict(bool)
    track_cats    = {}

    with open(LOG_PATH, newline="") as f:
        for row in csv.DictReader(f):
            key  = (row["video"], int(row["face_id"]))
            area = float(row["area_ratio"]) if row["area_ratio"] else 0.0

            if key not in best_frame or area > float(best_frame[key]["area_ratio"]):
                best_frame[key] = row

            if row["interacting"] == "1":
                ever_partner[key] = True

            track_cats[key] = row["category"]

    return best_frame, ever_partner, track_cats


def _find_video(video_name, category):
    p = (INTERACTION_INPUT_DIR if category == "interaction"
         else BYSTANDER_INPUT_DIR) / video_name
    if not p.exists():
        # try other category
        alt = (BYSTANDER_INPUT_DIR if category == "interaction"
               else INTERACTION_INPUT_DIR) / video_name
        return alt if alt.exists() else None
    return p


def _crop_face(frame, x1, y1, x2, y2, pad=CROP_PADDING):
    h, w = frame.shape[:2]
    pw = int((x2 - x1) * pad)
    ph = int((y2 - y1) * pad)
    return frame[max(0, y1-ph):min(h, y2+ph),
                 max(0, x1-pw):min(w, x2+pw)]


# =============================================================================
# Contact sheet
# =============================================================================

def build_contact_sheet(video_name, best_frame, ever_partner):
    """
    Build a single image grid: one cell per track ID, labelled with the ID
    and the system's current decision (PARTNER / BYSTANDER).
    """
    tracks = [(tid, row) for (vid, tid), row in best_frame.items()
              if vid == video_name]
    tracks.sort(key=lambda x: x[0])

    if not tracks:
        return None

    # Find source video
    category = best_frame[(video_name, tracks[0][0])]["category"]
    video_path = _find_video(video_name, category)
    if not video_path:
        print(f"[Sheet] Source video not found for {video_name}")
        return None

    cap = cv2.VideoCapture(str(video_path))

    cells = []
    for tid, row in tracks:
        frame_idx = int(row["frame"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            cells.append(None)
            continue

        crop = _crop_face(frame, int(row["x1"]), int(row["y1"]),
                          int(row["x2"]), int(row["y2"]))
        if crop.size == 0:
            cells.append(None)
            continue

        # Resize to cell size
        cell = cv2.resize(crop, (CELL_SIZE, CELL_SIZE))

        # Add coloured border: green = system said partner, red = bystander
        is_partner = ever_partner.get((video_name, tid), False)
        color = (0, 180, 0) if is_partner else (0, 0, 200)
        cell = cv2.copyMakeBorder(cell, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=color)

        # Label: track ID + system decision
        label = f"ID:{tid}"
        decision = "PTR" if is_partner else "BYS"
        cv2.rectangle(cell, (0, CELL_SIZE-22), (CELL_SIZE+6, CELL_SIZE+6),
                      (0, 0, 0), cv2.FILLED)
        cv2.putText(cell, label, (4, CELL_SIZE - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(cell, decision, (CELL_SIZE - 34, CELL_SIZE - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (0, 255, 0) if is_partner else (0, 100, 255), 1, cv2.LINE_AA)

        cells.append((tid, cell))

    cap.release()

    valid = [(tid, c) for tid, c in cells if c is not None]
    if not valid:
        return None

    n_cols = SHEET_COLS
    n_rows = (len(valid) + n_cols - 1) // n_cols
    cell_h, cell_w = valid[0][1].shape[:2]

    # Header banner
    header_h = 40
    sheet_w   = cell_w * n_cols
    sheet_h   = header_h + cell_h * n_rows
    sheet = cv2.copyMakeBorder(
        cv2.Mat if False else
        __import__('numpy').zeros((sheet_h, sheet_w, 3), dtype='uint8'),
        0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30)
    )

    # Draw header
    cv2.putText(sheet,
        f"{video_name}  —  {len(valid)} tracks   "
        f"[GREEN border = system said PARTNER | RED border = system said BYSTANDER]",
        (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # Place cells
    for i, (tid, cell) in enumerate(valid):
        row_i = i // n_cols
        col_i = i %  n_cols
        y_off = header_h + row_i * cell_h
        x_off = col_i * cell_w
        sheet[y_off:y_off+cell_h, x_off:x_off+cell_w] = cell

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    stem      = Path(video_name).stem
    out_path  = SAMPLES_DIR / f"contact_sheet_{stem}.png"
    cv2.imwrite(str(out_path), sheet)
    print(f"[Sheet] Saved → {out_path}")
    return out_path


# =============================================================================
# Interactive annotator
# =============================================================================

def annotate_video(video_name, best_frame, ever_partner, track_cats,
                   existing_labels):
    """
    Show each track's face crop one by one and ask the user to label it.
    Returns dict: track_id -> label (0 or 1)
    """
    tracks = sorted(
        [(tid, row) for (vid, tid), row in best_frame.items() if vid == video_name],
        key=lambda x: x[0]
    )
    if not tracks:
        return {}

    category   = track_cats.get((video_name, tracks[0][0]), "interaction")
    video_path = _find_video(video_name, category)

    results = dict(existing_labels)   # start from any already-saved labels

    print(f"\n{'='*60}")
    print(f" Annotating: {video_name}  ({len(tracks)} tracks)")
    print(f" Category  : {category}")
    print(f"{'='*60}")
    print(" Controls:  p = Partner   b = Bystander   s = Skip")
    print(" Press ESC or q in the image window to quit early.")
    print(f"{'='*60}\n")

    if video_path:
        cap = cv2.VideoCapture(str(video_path))
    else:
        cap = None
        print(f"[Annotate] Warning: source video not found — using saved crops only")

    total  = len(tracks)
    done   = 0
    skipped = 0

    for idx, (tid, row) in enumerate(tracks):
        # Skip if already labelled (unless re-running)
        if tid in results:
            print(f"  [{idx+1}/{total}] Track {tid:>3} — already labelled: "
                  f"{'PARTNER' if results[tid] else 'BYSTANDER'} (skipping)")
            done += 1
            continue

        # Load the best frame crop
        frame = None
        if cap:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame"]))
            ret, frame = cap.read()

        system_says = "PARTNER" if ever_partner.get((video_name, tid)) else "BYSTANDER"

        if frame is not None:
            crop = _crop_face(frame, int(row["x1"]), int(row["y1"]),
                              int(row["x2"]), int(row["y2"]))
        else:
            # Fall back to saved crop image
            stem = Path(video_name).stem
            saved = list(SAMPLES_DIR.glob(f"{stem}__track{tid}__*.png"))
            crop  = cv2.imread(str(saved[0])) if saved else None

        # Display the crop
        if crop is not None and crop.size > 0:
            display = cv2.resize(crop, (240, 240))
            # Add info banner
            banner = __import__('numpy').zeros((60, 240, 3), dtype='uint8')
            cv2.putText(banner, f"Track ID: {tid}  [{idx+1}/{total}]",
                        (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(banner, f"System: {system_says}",
                        (6, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0) if system_says == "PARTNER" else (0, 100, 255),
                        1, cv2.LINE_AA)
            display = __import__('numpy').vstack([banner, display])
            cv2.imshow("Annotate — p=Partner  b=Bystander  s=Skip  q=Quit", display)

        print(f"  [{idx+1}/{total}] Track {tid:>3}  |  System says: {system_says:<9}"
              f"  |  Your label?  [p/b/s] : ", end="", flush=True)

        # Wait for keypress from window OR terminal input
        label = None
        while label is None:
            # Check window key (30ms poll)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('p'), ord('P')):
                label = 1
            elif key in (ord('b'), ord('B')):
                label = 0
            elif key in (ord('s'), ord('S')):
                label = -1   # skip
            elif key in (ord('q'), ord('Q'), 27):  # ESC
                print("\n[Annotate] Quit — saving progress so far.")
                if cap:
                    cap.release()
                cv2.destroyAllWindows()
                return results

        if label == -1:
            print("skipped")
            skipped += 1
        else:
            results[tid] = label
            print("PARTNER" if label == 1 else "BYSTANDER")
            done += 1

    if cap:
        cap.release()
    cv2.destroyAllWindows()

    print(f"\n  Done — {done} labelled, {skipped} skipped out of {total} tracks.")
    return results


# =============================================================================
# Save labels to CSV
# =============================================================================

def _load_existing(gt_file):
    """Load existing labels from a GT CSV. Returns {track_id -> label}."""
    labels = {}
    if gt_file.exists():
        with open(gt_file, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    labels[int(row["track_id"])] = int(row["ground_truth"])
                except (ValueError, KeyError):
                    pass
    return labels


def _save_labels(video_name, category, labels):
    """Append / overwrite labels for this video into the correct GT CSV."""
    gt_file = INTERACTIONS_GT if category == "interaction" else BYSTANDER_GT
    gt_file.parent.mkdir(parents=True, exist_ok=True)

    # Load all existing rows (other videos stay untouched)
    all_rows = []
    if gt_file.exists():
        with open(gt_file, newline="") as f:
            for row in csv.DictReader(f):
                if row["video"] != video_name:
                    all_rows.append(row)

    # Add rows for this video
    for tid, lbl in sorted(labels.items()):
        all_rows.append({"video": video_name, "track_id": tid,
                         "ground_truth": lbl})

    with open(gt_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "track_id", "ground_truth"])
        w.writeheader()
        w.writerows(all_rows)

    print(f"\n[Annotate] Saved {len(labels)} labels → {gt_file}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Interactive annotation tool for ground-truth labelling"
    )
    parser.add_argument("--video",  type=str, default=None,
                        help="Annotate only this video (e.g. video1.mp4)")
    parser.add_argument("--sheet",  action="store_true",
                        help="Generate contact sheet(s) only — no annotation")
    parser.add_argument("--redo",   action="store_true",
                        help="Re-annotate even tracks that already have labels")
    args = parser.parse_args()

    if not LOG_PATH.exists():
        print("[Annotate] tracking_log.csv not found. Run main.py first.")
        return

    best_frame, ever_partner, track_cats = _load_log()

    # Which videos to process
    all_videos = sorted(set(vid for vid, _ in best_frame.keys()))
    if args.video:
        all_videos = [v for v in all_videos if v == args.video]

    if not all_videos:
        print("[Annotate] No matching videos found in the tracking log.")
        return

    for video_name in all_videos:
        print(f"\n{'='*60}")
        print(f" Video: {video_name}")

        # Always build contact sheet first
        sheet_path = build_contact_sheet(video_name, best_frame, ever_partner)
        if sheet_path:
            print(f" Contact sheet: {sheet_path}")
            print(" Open this image to see all track IDs and their face crops.")

        if args.sheet:
            continue

        # Determine category and load existing labels
        sample_key = (video_name, next(
            tid for (vid, tid) in best_frame if vid == video_name))
        category = track_cats.get(sample_key, "interaction")
        gt_file  = INTERACTIONS_GT if category == "interaction" else BYSTANDER_GT

        existing = {} if args.redo else _load_existing(gt_file)
        already  = sum(1 for (vid, tid) in best_frame
                       if vid == video_name and tid in existing)
        total    = sum(1 for (vid, _) in best_frame if vid == video_name)

        print(f" Already labelled: {already}/{total} tracks")

        if already == total and not args.redo:
            print(" All tracks labelled — use --redo to re-annotate.")
            continue

        labels = annotate_video(video_name, best_frame, ever_partner,
                                track_cats, existing)
        if labels:
            _save_labels(video_name, category, labels)

    print("\n[Annotate] All done. Run:  python eval/evaluate.py --accuracy")


if __name__ == "__main__":
    main()
