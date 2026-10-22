# =============================================================================
# eval/aggregate_clip_annotations.py — Score the system against clip labels
# =============================================================================
#
# Usage (after annotators send back their files from eval/clip_package/):
#
#   python eval/aggregate_clip_annotations.py eval/clip_package/annotations_a.csv \
#          eval/clip_package/annotations_b.csv eval/clip_package/annotations_c.csv
#
# Each file has columns: clip_id, label  (p / b / blank)
#
# WHAT IT DOES
# ------------
# 1. Majority vote per clip (ties -> bystander, the privacy-conservative choice)
# 2. Pairwise Cohen's kappa and Fleiss' kappa across annotators
# 3. Compares the vote with the system's per-window decision in clip_map.csv
# 4. Reports PSR, IPR, Precision, Recall, F1 with 95% Wilson intervals,
#    overall and per clip category
# 5. Saves:
#      eval/clip_package/clip_ground_truth.csv
#      eval/clip_package/clip_agreement_report.csv
#      logs/clip_annotation_metrics.csv
# =============================================================================

import argparse
import csv
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import LOGS_DIR
from eval.aggregate_annotations import _cohens_kappa

# Windows consoles default to cp1252, which can't print kappa
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PKG_DIR     = Path(__file__).resolve().parent / "clip_package"
MAP_FILE    = PKG_DIR / "clip_map.csv"
GT_FILE     = PKG_DIR / "clip_ground_truth.csv"
AGREE_FILE  = PKG_DIR / "clip_agreement_report.csv"
METRICS_OUT = LOGS_DIR / "clip_annotation_metrics.csv"


def _load(files):
    data = {}
    for path in files:
        name = Path(path).stem.replace("annotations_", "")
        labels = {}
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                raw = (row.get("label") or "").strip().lower()
                if raw in ("p", "b"):
                    labels[row["clip_id"].strip()] = 1 if raw == "p" else 0
        data[name] = labels
        print(f"  [{name}]  {len(labels)} labels")
    return data


def _fleiss_kappa(annotations, keys):
    """Fleiss' kappa over items labelled by >= 2 annotators (variable raters)."""
    items = []
    for k in keys:
        votes = [a[k] for a in annotations.values() if k in a]
        if len(votes) >= 2:
            items.append((sum(votes), len(votes)))
    if not items:
        return None
    p_bar = sum((pos * (pos - 1) + (n - pos) * (n - pos - 1)) / (n * (n - 1))
                for pos, n in items) / len(items)
    total = sum(n for _, n in items)
    p1 = sum(pos for pos, _ in items) / total
    pe = p1 ** 2 + (1 - p1) ** 2
    return (p_bar - pe) / (1 - pe) if pe < 1 else 1.0


def _wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def _metrics(pairs):
    tp = sum(1 for g, p in pairs if g == 1 and p == 1)
    fn = sum(1 for g, p in pairs if g == 1 and p == 0)
    fp = sum(1 for g, p in pairs if g == 0 and p == 1)
    tn = sum(1 for g, p in pairs if g == 0 and p == 0)
    pct = lambda a, b: 100 * a / b if b else float("nan")
    prec, rec = pct(tp, tp + fp), pct(tp, tp + fn)
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    return {
        "n": len(pairs), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "psr": pct(tn, tn + fp), "psr_ci": _wilson(tn, tn + fp),
        "ipr": rec, "ipr_ci": _wilson(tp, tp + fn),
        "precision": prec, "recall": rec, "f1": f1,
    }


def _print_block(title, m):
    print(f"\n  {title}  (n = {m['n']})")
    print(f"    TP {m['tp']:>3}   FN {m['fn']:>3}   FP {m['fp']:>3}   TN {m['tn']:>3}")
    print(f"    PSR {m['psr']:6.2f}%  [{m['psr_ci'][0]:.1f}, {m['psr_ci'][1]:.1f}]")
    print(f"    IPR {m['ipr']:6.2f}%  [{m['ipr_ci'][0]:.1f}, {m['ipr_ci'][1]:.1f}]")
    print(f"    Precision {m['precision']:6.2f}%   F1 {m['f1']:6.2f}%")


def main():
    ap = argparse.ArgumentParser(description="Aggregate clip annotations and score the system")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--min-annotators", type=int, default=2)
    args = ap.parse_args()

    print("\n" + "=" * 60 + "\nCLIP ANNOTATION AGGREGATION\n" + "=" * 60)
    ann = _load(args.files)
    clip_map = {r["clip_id"]: r for r in csv.DictReader(open(MAP_FILE, newline=""))}
    keys = sorted(clip_map)

    # Agreement
    print("\nINTER-ANNOTATOR AGREEMENT")
    agree_rows = []
    for a, b in combinations(sorted(ann), 2):
        kappa, n, pct = _cohens_kappa(ann[a], ann[b])
        if kappa is None:
            continue
        agree_rows.append({"pair": f"{a} vs {b}", "kappa": round(kappa, 3),
                           "agreement_pct": round(pct, 1), "n_shared": n})
        print(f"  {a} vs {b}: kappa = {kappa:.3f}  agreement = {pct:.1f}%  n = {n}")
    fleiss = _fleiss_kappa(ann, keys) if len(ann) >= 2 else None
    if fleiss is not None:
        print(f"  Fleiss' kappa (all annotators): {fleiss:.3f}")

    # Majority vote
    gt, ties = {}, 0
    for k in keys:
        votes = [a[k] for a in ann.values() if k in a]
        if len(votes) < args.min_annotators:
            continue
        pos = sum(votes)
        ties += pos * 2 == len(votes)
        gt[k] = 1 if pos * 2 > len(votes) else 0
    print(f"\nMajority vote: {len(gt)} clips labelled ({ties} ties -> bystander), "
          f"{sum(gt.values())} partners, {len(gt) - sum(gt.values())} bystanders")

    with open(GT_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "category", "video", "track_id", "ground_truth", "system_prediction"])
        for k, g in gt.items():
            r = clip_map[k]
            w.writerow([k, r["category"], r["video"], r["track_id"], g, r["system_prediction"]])

    # Metrics
    pairs = {"all": [], "interaction": [], "bystander": []}
    for k, g in gt.items():
        p = int(clip_map[k]["system_prediction"])
        pairs["all"].append((g, p))
        pairs[clip_map[k]["category"]].append((g, p))
    results = {name: _metrics(v) for name, v in pairs.items() if v}

    print("\nMODEL vs HUMAN MAJORITY VOTE (per-clip decision)")
    for name, m in results.items():
        _print_block(name.upper(), m)

    with open(METRICS_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "metric", "value"])
        w.writerow(["all", "annotators", len(ann)])
        if fleiss is not None:
            w.writerow(["all", "fleiss_kappa", f"{fleiss:.3f}"])
        w.writerow(["all", "ties", ties])
        for name, m in results.items():
            for key in ("n", "tp", "fn", "fp", "tn"):
                w.writerow([name, key, m[key]])
            for key in ("psr", "ipr", "precision", "recall", "f1"):
                w.writerow([name, key, f"{m[key]:.2f}"])
            w.writerow([name, "psr_ci95", f"{m['psr_ci'][0]:.1f}-{m['psr_ci'][1]:.1f}"])
            w.writerow([name, "ipr_ci95", f"{m['ipr_ci'][0]:.1f}-{m['ipr_ci'][1]:.1f}"])
    if agree_rows:
        with open(AGREE_FILE, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(agree_rows[0]))
            w.writeheader()
            w.writerows(agree_rows)
    print(f"\n  Saved: {METRICS_OUT}\n         {GT_FILE}\n" + "=" * 60)


if __name__ == "__main__":
    main()
