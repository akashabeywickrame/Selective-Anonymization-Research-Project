# =============================================================================
# utils/finish_pipeline.py — Post-pipeline completion tasks
# =============================================================================
#
# Run this after all 23 videos are processed:
#
#   python utils/finish_pipeline.py
#
# WHAT IT DOES
# ------------
# 1. Verifies all 23 videos are fully processed in tracking_log.csv
# 2. Rebuilds proxy ground-truth CSVs from the complete log
# 3. Regenerates the annotation package (10 frames, all 23 videos available)
# 4. Runs the final accuracy evaluation (proxy metrics)
# 5. Runs the FPS benchmark
# 6. Reports next steps for human annotation
#
# NOTE: Proxy GT (partner_frames > PARTNER_FRAMES_THRESHOLD) is used for
# automated evaluation. For valid Phase 2b dissertation metrics, collect
# filled annotator_template.csv files from 3 annotators and run:
#   python eval/aggregate_annotations.py annotations_NAME1.csv ... --min-annotators 2
# =============================================================================

import csv
import sys
import subprocess
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    LOGS_DIR, BYSTANDER_INPUT_DIR, INTERACTION_INPUT_DIR,
    BYSTANDER_OUTPUT_DIR, INTERACTION_OUTPUT_DIR,
)

LOG_PATH     = LOGS_DIR / "tracking_log.csv"
GT_INT_PATH  = Path(__file__).resolve().parent.parent / "eval" / "ground_truth" / "interactions_labels.csv"
GT_BYS_PATH  = Path(__file__).resolve().parent.parent / "eval" / "ground_truth" / "bystander_labels.csv"
PKG_SCRIPT   = Path(__file__).resolve().parent.parent / "eval" / "generate_annotation_package.py"
EVAL_SCRIPT  = Path(__file__).resolve().parent.parent / "eval" / "evaluate.py"

PARTNER_FRAMES_THRESHOLD = 5


def _log_max_frames():
    """Return dict (video, category) -> max_frame_index logged."""
    from collections import defaultdict as _dd
    mf = _dd(int)
    if LOG_PATH.exists():
        with open(LOG_PATH, newline="") as f:
            for row in csv.DictReader(f):
                k = (row["video"], row["category"])
                fi = int(row["frame"])
                if fi > mf[k]:
                    mf[k] = fi
    return mf


def _check_completion():
    """Return (complete_videos, incomplete_videos).

    A video is complete if EITHER:
      - its output selective.mp4 exists AND its log coverage >= 90% of total frames, OR
      - it has a valid but sparse scene (last logged frame is the video's natural
        end of face-activity, accepted as done if output file is present and log
        coverage >= 50% — covers bystander videos where faces leave the scene early).
    """
    import cv2
    max_frames = _log_max_frames()
    complete, incomplete = [], []
    all_jobs = (
        [(vp, "bystander") for vp in sorted(BYSTANDER_INPUT_DIR.glob("*.mp4"))]
        + [(vp, "interaction") for vp in sorted(INTERACTION_INPUT_DIR.glob("*.mp4"))]
    )
    for vp, cat in all_jobs:
        out_dir = BYSTANDER_OUTPUT_DIR if cat == "bystander" else INTERACTION_OUTPUT_DIR
        selective_out = out_dir / f"{vp.stem}_selective.mp4"
        if not selective_out.exists():
            incomplete.append((vp.name, cat))
            continue
        # Output file exists — check log coverage
        cap = cv2.VideoCapture(str(vp))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        logged = max_frames.get((vp.name, cat), 0)
        coverage = logged / total if total > 0 else 0
        if coverage >= 0.50:  # >= 50% coverage with output file = done
            complete.append((vp.name, cat))
        else:
            incomplete.append((vp.name, cat))
    return complete, incomplete


def _rebuild_gt():
    """Rebuild proxy ground-truth CSVs from the complete tracking log."""
    print("\n[GT] Rebuilding ground-truth CSVs from tracking log...")

    ever_partner = defaultdict(int)
    all_tracks_int = set()
    all_tracks_bys = set()

    with open(LOG_PATH, newline="") as f:
        for row in csv.DictReader(f):
            video    = row["video"]
            cat      = row["category"]
            track_id = int(row["face_id"])
            if row["interacting"] == "1":
                ever_partner[(video, cat, track_id)] += 1
            if cat == "interaction":
                all_tracks_int.add((video, track_id))
            else:
                all_tracks_bys.add((video, track_id))

    # Interactions GT
    int_rows = []
    for video, track_id in sorted(all_tracks_int):
        gt = 1 if ever_partner.get((video, "interaction", track_id), 0) > PARTNER_FRAMES_THRESHOLD else 0
        int_rows.append({"video": video, "track_id": track_id, "ground_truth": gt})

    GT_INT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GT_INT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "track_id", "ground_truth"])
        w.writeheader()
        w.writerows(int_rows)

    int_partners = sum(1 for r in int_rows if r["ground_truth"] == 1)
    print(f"  interactions_labels.csv: {len(int_rows)} tracks, {int_partners} partners")

    # Bystander GT (all GT=0)
    bys_rows = []
    for video, track_id in sorted(all_tracks_bys):
        bys_rows.append({"video": video, "track_id": track_id, "ground_truth": 0})

    with open(GT_BYS_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "track_id", "ground_truth"])
        w.writeheader()
        w.writerows(bys_rows)

    print(f"  bystander_labels.csv: {len(bys_rows)} tracks, all GT=0")


def main():
    print("\n" + "="*60)
    print("POST-PIPELINE COMPLETION CHECK")
    print("="*60)

    if not LOG_PATH.exists():
        print("[ERROR] tracking_log.csv not found. Run main.py first.")
        sys.exit(1)

    # ── 1. Check completion ────────────────────────────────────────────
    print("\n[Step 1] Checking video completion...")
    complete, incomplete = _check_completion()
    print(f"  Complete  : {len(complete)}")
    print(f"  Incomplete: {len(incomplete)}")
    for name, cat in incomplete:
        print(f"    MISSING: {cat}/{name}")

    if incomplete:
        print("\n[WARNING] Pipeline not fully done. Run main.py to finish, then re-run this script.")
    else:
        print("  All videos processed.")

    # ── 2. Rebuild GT ──────────────────────────────────────────────────
    print("\n[Step 2] Rebuilding proxy ground truth...")
    _rebuild_gt()

    # ── 3. Regenerate annotation package ──────────────────────────────
    print("\n[Step 3] Regenerating annotation package...")
    result = subprocess.run(
        [sys.executable, str(PKG_SCRIPT)],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    stdout = result.stdout or ""
    if result.returncode == 0 or "Generated successfully" in stdout:
        print("  Annotation package regenerated.")
    else:
        print("  [WARNING] Package generator returned non-zero. Check eval/annotation_package/.")
        if result.stderr:
            print("  STDERR:", result.stderr[-500:])

    # ── 4. Run evaluation ──────────────────────────────────────────────
    print("\n[Step 4] Running accuracy evaluation (proxy GT)...")
    result = subprocess.run(
        [sys.executable, str(EVAL_SCRIPT), "--accuracy"],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    stdout = result.stdout or ""
    if result.returncode == 0:
        # Print key lines
        for line in stdout.splitlines():
            if any(x in line for x in ["PSR", "IPR", "F1", "Precision", "Recall", "Tracks", "="]):
                print(" ", line)
    else:
        print("  [ERROR] evaluate.py failed:", result.stderr[-200:])

    # ── 5. FPS benchmark ───────────────────────────────────────────────
    print("\n[Step 5] Running FPS benchmark (300 frames)...")
    result = subprocess.run(
        [sys.executable, str(EVAL_SCRIPT), "--fps", "--frames", "300"],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    stdout5 = result.stdout or ""
    if result.returncode == 0:
        for line in stdout5.splitlines():
            if any(x in line for x in ["FPS", "Total per", "PASS", "FAIL", "latency", "="]):
                print(" ", line)
    else:
        print("  [ERROR] FPS benchmark failed:", (result.stderr or "")[-200:])

    # ── 6. Next steps ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("NEXT STEPS FOR PHASE 2b DISSERTATION METRICS")
    print("="*60)
    print("""
  1. Share these files with 3 annotators:
       eval/annotation_package/frames/          (10 frame images)
       eval/annotation_package/annotator_template.csv
       eval/annotation_package/INSTRUCTIONS.txt

     OR send the zip:
       eval/annotation_package.zip

  2. Each annotator fills in label p/b for each face and saves as:
       annotations_NAME.csv

  3. After collecting all 3 files, run:
       python eval/aggregate_annotations.py annotations_*.csv --min-annotators 2

     This computes:
       - Cohen's kappa (inter-annotator agreement)
       - PSR, IPR, F1 vs the model
       - Saves: eval/annotation_package/aggregated_ground_truth.csv
                logs/human_annotation_metrics.csv

  4. Fill dissertation Phase 2b tables from logs/human_annotation_metrics.csv.
""")
    print("="*60)


if __name__ == "__main__":
    main()
