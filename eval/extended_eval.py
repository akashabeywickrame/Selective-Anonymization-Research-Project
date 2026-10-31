# =============================================================================
# eval/extended_eval.py — Further analyses of the Set C video-clip evaluation
# =============================================================================
#
# Uses only existing logs and human labels (nothing is re-labelled):
#   1. Scores against each annotator on their own, and Cohen's kappa between
#      the system and each annotator (compare with inter-human kappa).
#   2. Leave-one-annotator-out majority vote: does any single annotator change
#      the ground truth or the scores?
#   3. Results by face size (median area ratio of the face during the clip).
#   4. Results per video.
#   5. Video-level (cluster) bootstrap intervals, since clips from the same
#      video are not independent.
#   6. Behaviour over the whole dataset (not only the labelled clips): share of
#      tracks of >= 2 s ever kept visible, and of face detections left visible.
#
# Each analysis is reported for the live system and for the offline second
# pass (whole-video decision with track joining and the group rule).
#
# Usage:  python eval/extended_eval.py   ->  logs/extended_eval.csv
# =============================================================================

import csv
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
import replay_classifier as R
from whole_video import decide
from config import LOGS_DIR

ANNOTATORS = ["A", "B", "C", "D", "E"]   # files are anonymised (letters only)
EXCLUDED = {"clip_067", "clip_071"}   # false detections (magazine cover, package)
rows = []


def out(analysis, config, s, **extra):
    rows.append({"analysis": analysis, "config": config, **extra,
                 **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items()}})
    print(f"{analysis:12} {config:34} TP {s['tp']:2} FN {s['fn']:2} FP {s['fp']:2} TN {s['tn']:2} "
          f"PSR {s['psr']:6.2f} IPR {s['ipr']:6.2f} F1 {s['f1']:6.2f} {extra if extra else ''}")


def score(labels, preds):
    """R.score, but tolerant of subsets with no partners or no bystanders."""
    tp = sum(1 for c in labels if labels[c] and preds[c]); fn = sum(1 for c in labels if labels[c] and not preds[c])
    fp = sum(1 for c in labels if not labels[c] and preds[c]); tn = sum(1 for c in labels if not labels[c] and not preds[c])
    pct = lambda a, b: 100 * a / b if b else float("nan")
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "psr": pct(tn, tn + fp), "ipr": pct(tp, tp + fn),
            "prec": pct(tp, tp + fp), "f1": pct(2 * tp, 2 * tp + fp + fn)}


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def load_annotator(name):
    lab = {}
    for r in csv.DictReader(open(R.CLIP_DIR / f"annotations_{name}.csv")):
        v = r["label"].strip().lower()
        if v in ("p", "b"):
            lab[r["clip_id"]] = int(v == "p")
    return lab


def majority(ann, names, ids):
    res = {}
    for c in ids:
        votes = [ann[n][c] for n in names if c in ann[n]]
        if votes:
            res[c] = int(sum(votes) * 2 > len(votes))   # a tie counts as bystander
    return res


def main():
    clips = {r["clip_id"]: r for r in csv.DictReader(open(R.CLIP_DIR / "clip_map.csv"))}
    gt = {r["clip_id"]: int(r["ground_truth"])
          for r in csv.DictReader(open(R.CLIP_DIR / "clip_ground_truth.csv"))}
    live = {c: clips[c]["system_prediction"] == "1" for c in gt}
    videos = sorted({(r["category"], r["video"]) for r in clips.values()})
    fps = {(r["category"], r["video"]): float(r["fps"]) for r in clips.values()}

    # replay log with boxes (for the second pass) and face areas
    log = R.load_log(set(videos))
    boxes = {}
    with open(LOGS_DIR / "tracking_log.csv", newline="") as f:
        for r in csv.DictReader(f):
            k = (r["category"], r["video"])
            if k in log:
                boxes[(k, int(r["frame"]), r["face_id"])] = [int(r[c]) for c in ("x1", "y1", "x2", "y2")]
    for v in videos:
        for fi, rs in log[v].items():
            for r in rs:
                r["box"] = boxes[(v, fi, r["tid"])]
    keep = {}
    for v in videos:
        dec = R.replay(log[v])
        keep[v] = decide(log[v], dec, fps[v], True, True)
    second = {c: keep[(clips[c]["category"], clips[c]["video"])].get(clips[c]["track_id"], False)
              for c in gt}
    systems = {"live": live, "second pass": second}

    for name, p in systems.items():
        out("overall", name, score(gt, p))

    # 1. each annotator on their own, and kappa system vs annotator
    ann = {n: load_annotator(n) for n in ANNOTATORS}
    letters = {n: n for n in ANNOTATORS}
    for n in ANNOTATORS:
        lab = {c: v for c, v in ann[n].items() if c in gt}
        for name, p in systems.items():
            ids = sorted(lab)
            k = kappa([lab[c] for c in ids], [int(p[c]) for c in ids])
            out("annotator", f"{name} vs {letters[n]}", score(lab, p), n=len(ids), kappa=round(k, 3))
    ids = sorted(gt)
    for name, p in systems.items():
        k = kappa([gt[c] for c in ids], [int(p[c]) for c in ids])
        print(f"kappa {name} vs majority: {k:.3f}")
        rows.append({"analysis": "kappa_majority", "config": name, "kappa": round(k, 3)})

    # 2. leave one annotator out
    for n in ANNOTATORS:
        rest = [m for m in ANNOTATORS if m != n]
        lab = majority(ann, rest, gt.keys())
        changed = sum(lab[c] != gt[c] for c in lab)
        for name, p in systems.items():
            out("leave_out", f"{name} without {letters[n]}", score(lab, p), changed=changed)

    # 3. by face size
    med = {}
    for c in gt:
        r = clips[c]; v = (r["category"], r["video"])
        a = [x["area"] for fi in range(int(r["start_frame"]), int(r["end_frame"]) + 1)
             for x in log[v].get(fi, []) if x["tid"] == r["track_id"]]
        med[c] = statistics.median(a) * 100
    bins = [(0, 0.6), (0.6, 1.2), (1.2, 2.5), (2.5, 100)]
    for lo, hi in bins:
        sub = {c: gt[c] for c in gt if lo <= med[c] < hi}
        for name, p in systems.items():
            out("size", f"{name} {lo}-{hi}%", score(sub, p), n=len(sub),
                partners=sum(sub.values()))

    # 4. per video
    for v in videos:
        sub = {c: gt[c] for c in gt if (clips[c]["category"], clips[c]["video"]) == v}
        if not sub:
            continue
        for name, p in systems.items():
            out("video", f"{name} {v[0]}/{v[1]}", score(sub, p), n=len(sub),
                partners=sum(sub.values()))

    # 5. video-level bootstrap
    by_v = defaultdict(list)
    for c in gt:
        by_v[(clips[c]["category"], clips[c]["video"])].append(c)
    vs = sorted(by_v)
    rng = random.Random(2026)
    for name, p in systems.items():
        res = {"psr": [], "ipr": [], "f1": []}
        for _ in range(5000):
            pick = [c for v in (rng.choice(vs) for _ in vs) for c in by_v[v]]
            tp = sum(1 for c in pick if gt[c] and p[c]); fn = sum(1 for c in pick if gt[c] and not p[c])
            fp = sum(1 for c in pick if not gt[c] and p[c]); tn = sum(1 for c in pick if not gt[c] and not p[c])
            if tp + fn == 0 or fp + tn == 0:
                continue
            res["psr"].append(100 * tn / (tn + fp)); res["ipr"].append(100 * tp / (tp + fn))
            res["f1"].append(200 * tp / (2 * tp + fp + fn))
        for m, xs in res.items():
            xs.sort()
            lo, hi = xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]
            print(f"bootstrap {name} {m}: {lo:.1f}-{hi:.1f} ({len(xs)} resamples)")
            rows.append({"analysis": "bootstrap", "config": f"{name} {m}", "lo": round(lo, 1), "hi": round(hi, 1)})

    # 6. whole dataset behaviour
    for v in videos:
        dec = R.replay(log[v])
        nfr = defaultdict(int); ever = defaultdict(bool); det = vis_live = vis_sec = 0
        seen = set()   # interaction/video3's log repeats some rows (interrupted run)
        for fi, rs in log[v].items():
            for r in rs:
                if (fi, r["tid"]) in seen:
                    continue
                seen.add((fi, r["tid"]))
                nfr[r["tid"]] += 1; det += 1
                if dec[(fi, r["tid"])]:
                    ever[r["tid"]] = True; vis_live += 1
                if keep[v][r["tid"]]:
                    vis_sec += 1
        long = [t for t in nfr if nfr[t] / fps[v] >= 2.0]
        row = {"analysis": "dataset", "config": f"{v[0]}/{v[1]}", "detections": det,
               "tracks_2s": len(long),
               "tracks_kept_live": sum(1 for t in long if ever[t]),
               "tracks_kept_second": sum(1 for t in long if keep[v][t]),
               "det_visible_live": round(100 * vis_live / det, 2) if det else 0,
               "det_visible_second": round(100 * vis_sec / det, 2) if det else 0}
        rows.append(row); print(row)

    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(LOGS_DIR / "extended_eval.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
