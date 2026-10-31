# =============================================================================
# eval/evaluate.py — Accuracy & FPS evaluation for Phase 2a
# =============================================================================
#
# Usage:
#   python eval/evaluate.py              # accuracy + FPS benchmark
#   python eval/evaluate.py --accuracy   # accuracy only
#   python eval/evaluate.py --fps        # FPS benchmark only
#   python eval/evaluate.py --video video1.mp4 --fps
#
# ACCURACY EVALUATION
# -------------------
# Reads your ground-truth CSV files and compares against the system's decisions
# in tracking_log.csv. One prediction per track = majority vote across frames.
#
# Reports: Precision, Recall, F1, PSR, IPR, confusion matrix
#
# FPS BENCHMARK
# -------------
# Re-runs the full pipeline on a short clip (default 300 frames) and
# times each stage individually.
# =============================================================================

import csv
import time
import cv2
import argparse
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    LOGS_DIR, INTERACTION_INPUT_DIR, BYSTANDER_INPUT_DIR,
    FRAME_WIDTH, FRAME_HEIGHT,
)
from core.detector    import FaceDetector
from core.tracker     import FaceTracker
from core.classifier  import InteractionClassifier
from core.anonymizer  import SelectiveAnonymizer

GROUND_TRUTH_DIR = Path(__file__).resolve().parent / "ground_truth"
INTERACTIONS_GT  = GROUND_TRUTH_DIR / "interactions_labels.csv"
BYSTANDER_GT     = GROUND_TRUTH_DIR / "bystander_labels.csv"
LOG_PATH         = LOGS_DIR / "tracking_log.csv"


# =============================================================================
# Helpers
# =============================================================================

def _load_ground_truth():
    """
    Load ground truth from both CSV files.
    Returns dict: (category, video_name, track_id) -> int  (1=partner, 0=bystander)
    Keying by category prevents collision when the same filename exists in both
    bystander/ and interaction/ (e.g. video1.mp4 in both).
    """
    gt = {}
    category_map = {INTERACTIONS_GT: "interaction", BYSTANDER_GT: "bystander"}
    for gt_file, category in category_map.items():
        if not gt_file.exists():
            continue
        with open(gt_file, newline="") as f:
            for row in csv.DictReader(f):
                video = row["video"].strip()
                tid   = int(row["track_id"].strip())
                label = int(row["ground_truth"].strip())
                gt[(category, video, tid)] = label
    return gt


def _load_system_predictions():
    """
    Read tracking_log.csv and compute one prediction per (category, video, track_id).
    Keying by category prevents collision when the same filename exists in both
    bystander/ and interaction/ directories.

    Rule — matches INTERACT_LATCH=True behaviour:
      If the system EVER classified this track as a partner at any point,
      the final prediction = 1 (partner).
      Only bystander (0) if the system NEVER classified it as interacting.

    Returns dict: (category, video_name, track_id) -> int  (1=partner, 0=bystander)
    """
    ever_partner = defaultdict(bool)

    with open(LOG_PATH, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["category"], row["video"], int(row["face_id"]))
            if row["interacting"] == "1":
                ever_partner[key] = True

    all_keys = set()
    with open(LOG_PATH, newline="") as f:
        for row in csv.DictReader(f):
            all_keys.add((row["category"], row["video"], int(row["face_id"])))

    return {key: (1 if ever_partner[key] else 0) for key in all_keys}


def _confusion(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _metrics(cm):
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    psr = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # % of bystanders blurred
    ipr = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # % of partners preserved
    return {"precision": precision, "recall": recall,
            "f1": f1, "psr": psr, "ipr": ipr}


# =============================================================================
# Accuracy evaluation
# =============================================================================

def run_accuracy():
    print("\n" + "="*60)
    print("ACCURACY EVALUATION — Phase 2a (Proximity + Centrality)")
    print("="*60)

    gt   = _load_ground_truth()
    pred = _load_system_predictions()

    if not gt:
        print("\n[Evaluate] No ground truth found.")
        print("  Steps:")
        print("  1. Run:  python utils/extract_track_samples.py")
        print("  2. Open: eval/track_samples/  and review each face crop")
        print("  3. Fill: eval/ground_truth/interactions_labels.csv")
        print("           eval/ground_truth/bystander_labels.csv")
        print()
        print("  CSV format:")
        print("    video,track_id,ground_truth")
        print("    video1.mp4,0,0    <- 0=bystander  1=partner")
        return

    y_true, y_pred = [], []
    unmatched = 0

    for key, true_label in gt.items():
        if key in pred:
            y_true.append(true_label)
            y_pred.append(pred[key])
        else:
            unmatched += 1

    if not y_true:
        print("[Evaluate] Ground truth loaded but no matching tracks found in log.")
        return

    cm = _confusion(y_true, y_pred)
    m  = _metrics(cm)

    print(f"\n  Tracks evaluated : {len(y_true)}")
    if unmatched:
        print(f"  Unmatched        : {unmatched} (GT tracks not found in log)")

    print()
    print("  Confusion Matrix:")
    print(f"    True Positive  (partner  → preserved) : {cm['tp']:>4}")
    print(f"    True Negative  (bystander → blurred)  : {cm['tn']:>4}")
    print(f"    False Positive (bystander → preserved): {cm['fp']:>4}  ← privacy leak")
    print(f"    False Negative (partner  → blurred)   : {cm['fn']:>4}  ← utility loss")

    print()
    print("  Classification Metrics:")
    print(f"    Precision : {m['precision']*100:>6.1f}%")
    print(f"    Recall    : {m['recall']*100:>6.1f}%")
    print(f"    F1-Score  : {m['f1']*100:>6.1f}%")

    print()
    print("  Privacy & Utility:")
    print(f"    PSR (Privacy Success Rate)         : {m['psr']*100:>6.1f}%  (target >= 95%)")
    print(f"    IPR (Interaction Preservation Rate): {m['ipr']*100:>6.1f}%  (target >= 93%)")

    out_path = LOGS_DIR / "metrics_report.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["tracks_evaluated", len(y_true)])
        for k, v in cm.items():
            w.writerow([k, v])
        for k, v in m.items():
            w.writerow([k, f"{v*100:.2f}%"])
    print(f"\n  Results saved -> {out_path}")
    print("="*60)


# =============================================================================
# FPS benchmark
# =============================================================================

class _TimedExtractor:
    """Wraps LandmarkExtractor so landmark time is measured separately from
    the rest of classification (extract() is called inside classify())."""

    def __init__(self, inner):
        self.inner = inner
        self.seconds = 0.0
        self.calls = 0

    def extract(self, frame, bbox):
        t0 = time.perf_counter()
        try:
            return self.inner.extract(frame, bbox)
        finally:
            self.seconds += time.perf_counter() - t0
            self.calls += 1


def run_fps_benchmark(video_name="video1.mp4", n_frames=300, phase="2a"):
    print("\n" + "="*60)
    print(f"FPS BENCHMARK — Phase {phase} pipeline")
    print(f"Video: {video_name}  |  Frames: {n_frames}")
    print("="*60)

    video_path = INTERACTION_INPUT_DIR / video_name
    if not video_path.exists():
        video_path = BYSTANDER_INPUT_DIR / video_name
    if not video_path.exists():
        print(f"[Benchmark] ERROR: {video_name} not found in data/")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Benchmark] ERROR: cannot open {video_path}")
        return

    detector   = FaceDetector()
    w = FRAME_WIDTH  if FRAME_WIDTH  > 0 else int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = FRAME_HEIGHT if FRAME_HEIGHT > 0 else int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracker    = FaceTracker(frame_size=(w, h))
    classifier = InteractionClassifier(frame_size=(w, h))
    anonymizer = SelectiveAnonymizer()

    extractor = None
    if phase == "2b":
        from core.landmark_extractor import LandmarkExtractor
        extractor = _TimedExtractor(LandmarkExtractor())

    t_read = t_detect = t_track = t_classify = t_anon = 0.0
    frames_done = 0

    print("\n  Benchmarking — please wait...\n")

    for _ in range(n_frames):
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        if FRAME_WIDTH > 0 and FRAME_HEIGHT > 0:
            frame = cv2.resize(frame, (w, h))
        t_read += time.perf_counter() - t0

        t0 = time.perf_counter()
        detections = detector.detect(frame)
        t_detect += time.perf_counter() - t0

        t0 = time.perf_counter()
        tracks = tracker.update(detections, frame)
        t_track += time.perf_counter() - t0

        t0 = time.perf_counter()
        if extractor is not None:
            decided = classifier.classify(tracks, frame=frame, extractor=extractor)
        else:
            decided = classifier.classify(tracks)
        t_classify += time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = anonymizer.apply(frame, decided)
        t_anon += time.perf_counter() - t0

        frames_done += 1

    cap.release()

    # Split landmark time out of the classification stage
    t_landmark = extractor.seconds if extractor is not None else 0.0
    lm_calls   = extractor.calls   if extractor is not None else 0
    t_classify -= t_landmark
    if extractor is not None:
        extractor.inner.close()

    total = t_read + t_detect + t_track + t_classify + t_landmark + t_anon
    fps   = frames_done / total if total > 0 else 0

    def ms(t):
        return t / frames_done * 1000

    print(f"  Frames processed : {frames_done}")
    print()
    print("  Per-stage latency (avg ms / frame):")
    print(f"    Frame read & resize : {ms(t_read):>7.2f} ms")
    print(f"    Detection  (YOLO)   : {ms(t_detect):>7.2f} ms")
    print(f"    Tracking   (SORT)   : {ms(t_track):>7.2f} ms")
    print(f"    Classification      : {ms(t_classify):>7.2f} ms")
    if phase == "2b":
        per_face = t_landmark / lm_calls * 1000 if lm_calls else 0
        print(f"    Landmarks (MediaPipe): {ms(t_landmark):>6.2f} ms"
              f"  ({lm_calls / frames_done:.2f} faces/frame, {per_face:.2f} ms/face)")
    print(f"    Anonymization       : {ms(t_anon):>7.2f} ms")
    print(f"    {'─'*32}")
    print(f"    Total per frame     : {ms(total):>7.2f} ms")
    print()
    print(f"  Overall throughput   : {fps:.1f} FPS")
    print()
    if fps >= 20:
        print("  [PASS] Meets real-time target (>= 20 FPS)")
    else:
        print(f"  [FAIL] Below real-time target (>= 20 FPS)")
        print("         Consider: --frame-skip, lower resolution, or lighter detector")

    out_path = LOGS_DIR / f"fps_benchmark_phase{phase}.csv"
    with open(out_path, "w", newline="") as f:
        w_csv = csv.writer(f)
        w_csv.writerow(["stage", "avg_ms_per_frame"])
        w_csv.writerow(["frame_read",      f"{ms(t_read):.3f}"])
        w_csv.writerow(["detection_yolo",  f"{ms(t_detect):.3f}"])
        w_csv.writerow(["tracking_sort",   f"{ms(t_track):.3f}"])
        w_csv.writerow(["classification",  f"{ms(t_classify):.3f}"])
        if phase == "2b":
            w_csv.writerow(["landmarks_mediapipe", f"{ms(t_landmark):.3f}"])
            w_csv.writerow(["landmark_faces_per_frame", f"{lm_calls / frames_done:.3f}"])
            w_csv.writerow(["landmark_ms_per_face",
                            f"{(t_landmark / lm_calls * 1000) if lm_calls else 0:.3f}"])
        w_csv.writerow(["anonymization",   f"{ms(t_anon):.3f}"])
        w_csv.writerow(["total_per_frame", f"{ms(total):.3f}"])
        w_csv.writerow(["fps",             f"{fps:.2f}"])
    print(f"\n  Results saved -> {out_path}")
    print("="*60)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Phase 2a accuracy and FPS")
    parser.add_argument("--accuracy", action="store_true", help="Accuracy only")
    parser.add_argument("--fps",      action="store_true", help="FPS benchmark only")
    parser.add_argument("--video",    type=str, default="video1.mp4",
                        help="Video for FPS benchmark (default: video1.mp4)")
    parser.add_argument("--frames",   type=int, default=300,
                        help="Frames to benchmark (default: 300)")
    parser.add_argument("--phase",    choices=["2a", "2b"], default="2a",
                        help="2a = geometry only; 2b = adds MediaPipe landmarks")
    args = parser.parse_args()

    run_acc = args.accuracy or (not args.accuracy and not args.fps)
    run_fps = args.fps      or (not args.accuracy and not args.fps)

    if run_acc:
        run_accuracy()
    if run_fps:
        run_fps_benchmark(video_name=args.video, n_frames=args.frames,
                          phase=args.phase)

