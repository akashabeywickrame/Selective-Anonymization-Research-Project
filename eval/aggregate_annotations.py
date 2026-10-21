# =============================================================================
# eval/aggregate_annotations.py — Aggregate friend annotations + compare to model
# =============================================================================
#
# Usage (after friends send back their filled CSV files):
#
#   python eval/aggregate_annotations.py annotations_alice.csv annotations_bob.csv
#   python eval/aggregate_annotations.py annotations_*.csv --min-annotators 2
#
# WHAT IT DOES
# ------------
# 1. Reads each filled annotator CSV  (frame_id, face_number, label: p/b)
# 2. Merges responses — majority vote becomes the final ground truth
# 3. Computes Cohen's κ for each pair of annotators (inter-annotator agreement)
# 4. Maps face numbers back to (video, track_id) via frame_track_map.csv
# 5. Compares majority-vote GT against the model's ever_partner predictions
# 6. Reports: Precision, Recall, F1, PSR, IPR + confusion matrix
# 7. Saves:
#      eval/annotation_package/aggregated_ground_truth.csv  — final GT per track
#      eval/annotation_package/agreement_report.csv         — kappa per pair
#      logs/human_annotation_metrics.csv                    — model comparison
#
# =============================================================================

import csv
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import LOGS_DIR

# Windows consoles default to cp1252, which can't print the arrows / kappa below
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PKG_DIR  = Path(__file__).resolve().parent / "annotation_package"
MAP_FILE = PKG_DIR / "frame_track_map.csv"

AGG_GT_FILE = PKG_DIR / "aggregated_ground_truth.csv"
AGREE_FILE  = PKG_DIR / "agreement_report.csv"
METRICS_OUT = LOGS_DIR / "human_annotation_metrics.csv"


# =============================================================================
# Load annotator files
# =============================================================================

def _load_annotations(files):
    """
    Returns dict: annotator_name -> {(frame_id, face_number) -> label (1/0)}
    Skips blanks (unanswered rows).
    """
    data = {}
    for path in files:
        name = Path(path).stem.replace("annotations_", "")
        labels = {}
        # utf-8-sig: tolerate the BOM Excel adds when saving as "CSV UTF-8"
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                raw = row.get("label", "").strip().lower()
                if raw not in ("p", "b"):
                    continue   # blank or invalid — skip
                key = (row["frame_id"].strip(), int(row["face_number"].strip()))
                labels[key] = 1 if raw == "p" else 0
        data[name] = labels
        print(f"  [{name}]  {len(labels)} labels loaded")
    return data


# =============================================================================
# Load frame → track mapping
# =============================================================================

def _load_map():
    """
    Returns dict: (frame_id, face_number) -> {video, track_id, system_prediction}
    """
    mapping = {}
    with open(MAP_FILE, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["frame_id"].strip(), int(row["face_number"].strip()))
            mapping[key] = {
                "video":             row["video"].strip(),
                "track_id":          int(row["track_id"].strip()),
                "system_prediction": int(row["system_prediction"].strip()),
            }
    return mapping


# =============================================================================
# Majority vote
# =============================================================================

def _majority_vote(annotations, min_annotators=1):
    """
    For each (frame_id, face_number), take majority vote across all annotators.
    Ties (equal p/b) are resolved as bystander (0) — conservative for privacy.
    Only includes keys where >= min_annotators gave a label.

    Returns dict: (frame_id, face_number) -> 0 or 1
    """
    votes = defaultdict(list)
    for labels in annotations.values():
        for key, label in labels.items():
            votes[key].append(label)

    result = {}
    for key, lbls in votes.items():
        if len(lbls) < min_annotators:
            continue
        result[key] = 1 if lbls.count(1) > len(lbls) / 2 else 0

    return result


# =============================================================================
# Cohen's κ
# =============================================================================

def _cohens_kappa(labels_a, labels_b):
    """
    Compute Cohen's κ between two annotators over their shared keys.
    Returns (kappa, n_shared, pct_agreement).
    """
    shared = set(labels_a) & set(labels_b)
    n = len(shared)
    if n == 0:
        return None, 0, 0.0

    agree   = sum(1 for k in shared if labels_a[k] == labels_b[k])
    p_o     = agree / n

    # Expected agreement by chance
    a_pos = sum(labels_a[k] for k in shared) / n
    b_pos = sum(labels_b[k] for k in shared) / n
    p_e   = a_pos * b_pos + (1 - a_pos) * (1 - b_pos)

    if p_e == 1.0:
        return 1.0, n, p_o * 100

    kappa = (p_o - p_e) / (1 - p_e)
    return kappa, n, p_o * 100


# =============================================================================
# Metrics (same as evaluate.py)
# =============================================================================

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
    psr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ipr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"precision": precision, "recall": recall,
            "f1": f1, "psr": psr, "ipr": ipr}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate blind annotations and compare against model"
    )
    parser.add_argument("files", nargs="+",
                        help="Filled annotator CSV files (e.g. annotations_alice.csv)")
    parser.add_argument("--min-annotators", type=int, default=1,
                        help="Minimum annotators required to include a face (default: 1)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("BLIND ANNOTATION AGGREGATION")
    print("="*60)

    # ── Load ─────────────────────────────────────────────────────────
    print(f"\nLoading {len(args.files)} annotator file(s):")
    annotations = _load_annotations(args.files)

    if not MAP_FILE.exists():
        print(f"\n[Aggregate] ERROR: {MAP_FILE} not found.")
        print("  Run:  python eval/generate_annotation_package.py  first.")
        return

    mapping = _load_map()
    print(f"\nFrame→Track mapping: {len(mapping)} entries")

    # ── Inter-annotator agreement ─────────────────────────────────────
    names = list(annotations.keys())
    kappa_rows = []

    print(f"\n{'─'*60}")
    print("INTER-ANNOTATOR AGREEMENT (Cohen's κ)")
    print(f"{'─'*60}")

    if len(names) < 2:
        print("  (Need at least 2 annotators for κ)")
    else:
        for a, b in combinations(names, 2):
            k, n, pct = _cohens_kappa(annotations[a], annotations[b])
            if k is None:
                print(f"  {a} vs {b}: no shared keys")
                continue
            interp = ("almost perfect" if k >= 0.81 else
                      "substantial"    if k >= 0.61 else
                      "moderate"       if k >= 0.41 else
                      "fair"           if k >= 0.21 else
                      "slight")
            print(f"  {a} vs {b}:  κ = {k:.3f}  ({interp})  "
                  f"agreement = {pct:.1f}%  n = {n}")
            kappa_rows.append({"pair": f"{a} vs {b}", "kappa": f"{k:.3f}",
                                "agreement_pct": f"{pct:.1f}",
                                "n_shared": n, "interpretation": interp})

    # ── Majority vote ─────────────────────────────────────────────────
    majority_gt = _majority_vote(annotations, min_annotators=args.min_annotators)
    print(f"\n{'─'*60}")
    print(f"MAJORITY-VOTE GROUND TRUTH  ({len(majority_gt)} face labels)")
    print(f"{'─'*60}")
    n_partners   = sum(v for v in majority_gt.values())
    n_bystanders = len(majority_gt) - n_partners
    print(f"  Partners  (1): {n_partners}")
    print(f"  Bystanders(0): {n_bystanders}")

    # ── Map back to track IDs ─────────────────────────────────────────
    # Per (video, track_id) a face may appear in multiple frames.
    # "Ever labelled partner" → track GT = 1  (matches INTERACT_LATCH logic)
    track_ever_partner  = defaultdict(bool)
    track_seen          = set()
    track_sys_pred      = {}

    for key, label in majority_gt.items():
        if key not in mapping:
            continue
        info  = mapping[key]
        tkey  = (info["video"], info["track_id"])
        track_seen.add(tkey)
        track_sys_pred[tkey] = info["system_prediction"]
        if label == 1:
            track_ever_partner[tkey] = True

    track_gt   = {k: (1 if track_ever_partner[k] else 0) for k in track_seen}
    n_tracks   = len(track_gt)

    print(f"\n  → Mapped to {n_tracks} unique tracks")
    p_tracks = sum(track_gt.values())
    b_tracks = n_tracks - p_tracks
    print(f"     GT Partners  : {p_tracks}")
    print(f"     GT Bystanders: {b_tracks}")

    # Save aggregated GT
    AGG_GT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(AGG_GT_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "track_id", "ground_truth"])
        w.writeheader()
        for (vid, tid), label in sorted(track_gt.items()):
            w.writerow({"video": vid, "track_id": tid, "ground_truth": label})
    print(f"\n  GT saved → {AGG_GT_FILE}")

    # ── Model comparison ──────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("MODEL vs HUMAN GROUND TRUTH")
    print(f"{'─'*60}")

    y_true = [track_gt[k]        for k in track_seen if k in track_sys_pred]
    y_pred = [track_sys_pred[k]  for k in track_seen if k in track_sys_pred]

    if not y_true:
        print("  No matching tracks found between GT and model log.")
    else:
        cm = _confusion(y_true, y_pred)
        m  = _metrics(cm)

        print(f"\n  Tracks evaluated: {len(y_true)}")
        print()
        print("  Confusion Matrix:")
        print(f"    TP (partner  → preserved) : {cm['tp']:>4}")
        print(f"    TN (bystander → blurred)  : {cm['tn']:>4}")
        print(f"    FP (bystander → preserved): {cm['fp']:>4}  ← privacy leak")
        print(f"    FN (partner  → blurred)   : {cm['fn']:>4}  ← utility loss")
        print()
        print("  Classification Metrics:")
        print(f"    Precision : {m['precision']*100:>6.1f}%")
        print(f"    Recall    : {m['recall']*100:>6.1f}%")
        print(f"    F1-Score  : {m['f1']*100:>6.1f}%")
        print()
        print("  Privacy & Utility:")
        print(f"    PSR (Privacy Success Rate)         : {m['psr']*100:>6.1f}%")
        print(f"    IPR (Interaction Preservation Rate): {m['ipr']*100:>6.1f}%")

        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(METRICS_OUT, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            w.writerow(["annotators", len(annotations)])
            w.writerow(["tracks_evaluated", len(y_true)])
            for k, v in cm.items():
                w.writerow([k, v])
            for k, v in m.items():
                w.writerow([k, f"{v*100:.2f}%"])
        print(f"\n  Metrics saved → {METRICS_OUT}")

    # ── Save kappa report ─────────────────────────────────────────────
    if kappa_rows:
        with open(AGREE_FILE, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=kappa_rows[0].keys())
            w.writeheader()
            w.writerows(kappa_rows)
        print(f"  Agreement saved → {AGREE_FILE}")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
