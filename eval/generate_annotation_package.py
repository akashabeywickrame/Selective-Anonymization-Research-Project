# =============================================================================
# eval/generate_annotation_package.py — Blind human annotation package
# =============================================================================
#
# Usage:
#   python eval/generate_annotation_package.py
#   python eval/generate_annotation_package.py --video video1.mp4 --frames 5
#
# WHAT IT DOES
# ------------
# 1. Picks frames from the ORIGINAL clean video (no debug overlay, no colours)
#    that contain 1+ visible faces.
#
# 2. Draws a numbered circle on each face (1, 2, 3...) with NO colour coding
#    and NO labels — annotators cannot tell what the model decided.
#
# 3. Saves each frame as:
#      eval/annotation_package/frames/frame_001.png
#      eval/annotation_package/frames/frame_002.png  ...
#
# 4. Generates a blank CSV template for each annotator:
#      eval/annotation_package/annotator_template.csv
#
#    Template columns:
#      frame_id, face_number, label
#    Label values: p = interaction partner,  b = bystander
#
# =============================================================================

import cv2
import csv
import argparse
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    LOGS_DIR, INTERACTION_INPUT_DIR, BYSTANDER_INPUT_DIR,
    FRAME_WIDTH, FRAME_HEIGHT,
)

LOG_PATH     = LOGS_DIR / "tracking_log.csv"
PKG_DIR      = Path(__file__).resolve().parent / "annotation_package"
FRAMES_DIR   = PKG_DIR / "frames"
MAP_FILE     = PKG_DIR / "frame_track_map.csv"
TEMPLATE_CSV = PKG_DIR / "annotator_template.csv"
INSTRUCTIONS = PKG_DIR / "INSTRUCTIONS.txt"

# Total annotation frames to produce across ALL videos
MAX_TOTAL_FRAMES = 10
# Max frames taken from each category (bystander / interaction) — ensures balance
N_PER_CATEGORY = 5
# Minimum faces in a frame to include it (1 = include single-partner frames)
MIN_FACES = 1
# Minimum face bounding-box side length in pixels (filters tiny false detections)
MIN_FACE_PX = 50
# Minimum detection confidence to include a face
MIN_CONF = 0.50
# Skip the first N frames (tracker is unstable at video start)
MIN_STABLE_FRAME = 30


# =============================================================================
# Load log — keyed by (video_name, category) to avoid filename collisions
# =============================================================================

def _load_log_by_frame(video_filter=None, category_filter=None):
    """
    Returns:
        by_video: (video_name, category) -> {frame_idx -> [row, ...]}
        ever_partner: (video_name, category, track_id) -> bool
    """
    by_video     = defaultdict(lambda: defaultdict(list))
    ever_partner = {}

    with open(LOG_PATH, newline="") as f:
        for row in csv.DictReader(f):
            vid = row["video"]
            cat = row["category"]
            if video_filter and vid != video_filter:
                continue
            if category_filter and cat != category_filter:
                continue
            key = (vid, cat)
            by_video[key][int(row["frame"])].append(row)
            ep_key = (vid, cat, int(row["face_id"]))
            if row["interacting"] == "1":
                ever_partner[ep_key] = True

    return by_video, ever_partner


def _find_video(video_name, category):
    """Find video file in the correct category directory first."""
    search_dirs = (
        [INTERACTION_INPUT_DIR, BYSTANDER_INPUT_DIR]
        if category == "interaction"
        else [BYSTANDER_INPUT_DIR, INTERACTION_INPUT_DIR]
    )
    for d in search_dirs:
        p = d / video_name
        if p.exists():
            return p
    return None


# =============================================================================
# Frame selection — spread evenly, require >= MIN_FACES tracked faces
# =============================================================================

def _filter_rows(rows):
    """Keep only faces that are large enough and confident enough to annotate."""
    good = []
    for r in rows:
        w = int(r["x2"]) - int(r["x1"])
        h = int(r["y2"]) - int(r["y1"])
        conf = float(r["confidence"]) if r["confidence"] else 0.0
        if w >= MIN_FACE_PX and h >= MIN_FACE_PX and conf >= MIN_CONF:
            good.append(r)
    return good


def _best_score(rows):
    """Score a set of face rows: prefer more faces, then larger minimum face."""
    sizes = [min(int(r["x2"]) - int(r["x1"]), int(r["y2"]) - int(r["y1"])) for r in rows]
    return (len(rows), min(sizes))


def _select_frames(frame_dict, n_frames=1):
    """
    Pick the single best frame from this video for annotation.
    'Best' = most qualifying faces visible; ties broken by smallest face size
    (larger = better visibility).
    Skips the first MIN_STABLE_FRAME frames where the tracker is unstable.
    """
    eligible = []
    for fi, rows in frame_dict.items():
        if fi < MIN_STABLE_FRAME:
            continue
        good = _filter_rows(rows)
        if len(good) >= MIN_FACES:
            eligible.append((fi, good))

    if not eligible:
        return []

    # Pick the frame with the most (and largest) qualifying faces
    best = max(eligible, key=lambda x: _best_score(x[1]))
    return [best]


# =============================================================================
# Draw numbered labels (no colour coding — blind annotation)
# =============================================================================

def _draw_numbered_frame(frame, rows):
    """
    Draw a numbered badge on each face (left-to-right order).
    Returns annotated frame + list of (face_number, track_id).
    """
    import numpy as np
    canvas  = frame.copy()
    overlay = canvas.copy()
    h, w    = canvas.shape[:2]

    sorted_rows = sorted(rows, key=lambda r: (int(r["x1"]) + int(r["x2"])) / 2)
    mapping = []

    for face_num, row in enumerate(sorted_rows, start=1):
        x1 = int(row["x1"]); y1 = int(row["y1"])
        x2 = int(row["x2"]); y2 = int(row["y2"])
        tid     = int(row["face_id"])
        face_w  = x2 - x1
        face_h  = y2 - y1

        # Thin white outline around the whole face
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # Badge centred on forehead (top 22% of face height)
        cx      = (x1 + x2) // 2
        cy      = y1 + max(int(face_h * 0.22), 14)
        badge_r = max(int(min(face_w, face_h) * 0.22), 14)

        cv2.circle(overlay, (cx, cy), badge_r, (0, 210, 255), cv2.FILLED)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
        overlay = canvas.copy()

        cv2.circle(canvas, (cx, cy), badge_r, (0, 0, 0), 2)

        label   = str(face_num)
        font    = cv2.FONT_HERSHEY_SIMPLEX
        scale   = max(badge_r / 20.0, 0.55)
        thick   = max(int(scale * 2), 1)
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        cv2.putText(canvas, label,
                    (cx - tw // 2, cy + th // 2),
                    font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
        cv2.putText(canvas, label,
                    (cx - tw // 2, cy + th // 2),
                    font, scale, (255, 255, 255), thick, cv2.LINE_AA)

        mapping.append((face_num, tid))

    # Bottom instruction banner
    banner_h = 38
    cv2.rectangle(canvas, (0, h - banner_h), (w, h), (20, 20, 20), cv2.FILLED)
    cv2.putText(canvas,
        "For each numbered face: Interaction Partner (p) or Bystander (b)?",
        (10, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (220, 220, 220), 1, cv2.LINE_AA)

    return canvas, mapping


# =============================================================================
# Main generator
# =============================================================================

def generate_package(video_filter=None):
    if not LOG_PATH.exists():
        print("[Package] tracking_log.csv not found. Run main.py first.")
        return

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    by_video, ever_partner = _load_log_by_frame(video_filter)
    video_keys = sorted(by_video.keys())  # list of (video_name, category)

    if not video_keys:
        print("[Package] No matching videos found in tracking log.")
        return

    print(f"[Package] Found {len(video_keys)} video(s) in log:")
    for vn, cat in video_keys:
        print(f"  {cat}/{vn}  ({len(by_video[(vn,cat)])} frames in log)")

    all_map_rows  = []
    template_rows = []
    frame_counter = 0
    cat_count     = {"bystander": 0, "interaction": 0}

    for (video_name, category) in video_keys:
        # Enforce per-category cap (5 bystander + 5 interaction = 10 total)
        if cat_count.get(category, 0) >= N_PER_CATEGORY:
            continue
        if frame_counter >= MAX_TOTAL_FRAMES:
            break
        video_path = _find_video(video_name, category)
        if not video_path:
            print(f"[Package] Source video not found: {category}/{video_name} — skipping")
            continue

        selected = _select_frames(by_video[(video_name, category)])
        if not selected:
            print(f"[Package] {category}/{video_name} — no frames with {MIN_FACES}+ faces, skipping")
            continue

        cat_count[category] = cat_count.get(category, 0) + 1
        cap = cv2.VideoCapture(str(video_path))
        print(f"\n[Package] {category}/{video_name}  (best frame selected)")

        for frame_idx, rows in selected:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            # Resize to match the pipeline's processing resolution so that
            # bbox coordinates from the tracking log align with the frame.
            if FRAME_WIDTH > 0 and FRAME_HEIGHT > 0:
                fh, fw = frame.shape[:2]
                if fw != FRAME_WIDTH or fh != FRAME_HEIGHT:
                    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            frame_counter += 1
            frame_id = f"frame_{frame_counter:03d}"

            annotated, mapping = _draw_numbered_frame(frame, rows)

            out_path = FRAMES_DIR / f"{frame_id}.png"
            cv2.imwrite(str(out_path), annotated)
            print(f"  {frame_id}  (video frame {frame_idx}, {len(mapping)} face(s))")

            for face_num, tid in mapping:
                ep_key   = (video_name, category, tid)
                sys_pred = 1 if ever_partner.get(ep_key) else 0
                all_map_rows.append({
                    "frame_id":          frame_id,
                    "face_number":       face_num,
                    "video":             video_name,
                    "category":          category,
                    "track_id":          tid,
                    "video_frame_idx":   frame_idx,
                    "system_prediction": sys_pred,
                })
                template_rows.append({
                    "frame_id":    frame_id,
                    "face_number": face_num,
                    "label":       "",
                })

        cap.release()

    if not all_map_rows:
        print("[Package] No frames generated.")
        return

    # Save internal mapping (KEEP PRIVATE from annotators)
    with open(MAP_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "frame_id", "face_number", "video", "category", "track_id",
            "video_frame_idx", "system_prediction"])
        w.writeheader()
        w.writerows(all_map_rows)

    # Save blank annotator template
    with open(TEMPLATE_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame_id", "face_number", "label"])
        w.writeheader()
        w.writerows(template_rows)

    # Instructions
    with open(INSTRUCTIONS, "w") as f:
        f.write("""ANNOTATION INSTRUCTIONS
=======================

You will be shown a series of video frames from a wearable camera.
Each frame has numbered faces (1, 2, 3...).

YOUR TASK
---------
For each numbered face, decide:

  p  =  Interaction Partner
         This person is actively talking TO the camera wearer.
         They are facing the camera and engaged in a conversation.

  b  =  Bystander
         This person is NOT interacting with the camera wearer.
         They may be walking past, standing nearby, or in the background.

HOW TO FILL IN THE CSV
----------------------
Open annotator_template.csv in Excel or any text editor.
Fill in the "label" column with either  p  or  b  for each row.

  frame_id  | face_number | label
  ----------|-------------|------
  frame_001 |      1      |  b
  frame_001 |      2      |  p
  frame_002 |      1      |  p
  ...

Save the file as:  annotations_YOUR_NAME.csv
and send it back.

NOTES
-----
- There is no right or wrong answer -- use your best judgement.
- If you genuinely cannot tell, leave the label blank.
- Each frame is independent -- do not worry about consistency across frames.
""")

    print(f"\n{'='*60}")
    print(f"[Package] Generated successfully!")
    print(f"  Frames      : {frame_counter}  ({len(template_rows)} faces total)")
    print(f"  Frames dir  : {FRAMES_DIR}")
    print(f"  Template    : {TEMPLATE_CSV}")
    print(f"  Instructions: {INSTRUCTIONS}")
    print(f"  Map (private): {MAP_FILE}")
    print()
    print("  SHARE WITH ANNOTATORS:")
    print("    → eval/annotation_package/frames/     (all frame images)")
    print("    → eval/annotation_package/annotator_template.csv")
    print("    → eval/annotation_package/INSTRUCTIONS.txt")
    print()
    print("  DO NOT SHARE:")
    print("    → eval/annotation_package/frame_track_map.csv  (has model decisions)")
    print(f"{'='*60}")


# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate blind annotation package for human annotators"
    )
    parser.add_argument("--video", type=str, default=None,
                        help="Process only this video filename (e.g. video1.mp4)")
    args = parser.parse_args()

    generate_package(video_filter=args.video)
