# =============================================================================
# eval/whole_video.py — One decision per person for the whole recorded video
# =============================================================================
#
# A partner is a partner for the whole video, not for one moment. For recorded
# video this script decides once per person instead of frame by frame:
#
#   1. Whole-track decision (retroactive latch): a track confirmed at any time
#      is kept visible for its whole time on screen.
#   2. Track stitching: SORT gives a person a new number after they have been
#      hidden for more than its 3 s memory. Without any face recognition, a new
#      track is joined to an old one if it starts within GAP_S seconds of the
#      old one ending, close to where the old one was last seen (centre
#      distance below one face width) and at a similar size. A confirmation on
#      any piece then covers the whole joined chain.
#   3. Group rule: a chain visible for >= 10 s and on screen with a confirmed
#      partner for >= 50% of its frames is kept visible.
#   4. Adaptive size threshold (faces turned towards the wearer may be smaller),
#      with its lower threshold chosen by leave-one-video-out cross-validation.
#
# Usage:  python eval/whole_video.py   ->  logs/whole_video.csv
# =============================================================================

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
import replay_classifier as R
from improvements import replay_adaptive
from config import LOGS_DIR

GAP_S, GROUP_S, COPRES = 5.0, 10.0, 0.5


def stitch(frames, fps):
    """Join track pieces of the same person by position and timing only.
    Returns {tid: chain_id}."""
    span = {}
    for fi in sorted(frames):
        for r in frames[fi]:
            t = r["tid"]
            if t not in span:
                span[t] = {"first": fi, "last": fi, "fbox": r["box"], "lbox": r["box"]}
            span[t]["last"], span[t]["lbox"] = fi, r["box"]
    chain = {t: t for t in span}

    def root(t):
        while chain[t] != t:
            t = chain[t]
        return t

    order = sorted(span, key=lambda t: span[t]["first"])
    ends = []   # tracks that have already ended, as candidates to continue
    for t in order:
        s = span[t]
        best, bestd = None, None
        for o in ends:
            so = span[o]
            gap = s["first"] - so["last"]
            if gap <= 0 or gap > GAP_S * fps:
                continue
            a, b = so["lbox"], s["fbox"]
            wa = a[2] - a[0]; wb = b[2] - b[0]
            if wa <= 0 or wb <= 0 or not (0.5 <= wb / wa <= 2.0):
                continue
            d = math.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2)
            if d < max(wa, wb) and (bestd is None or d < bestd):
                best, bestd = o, d
        if best is not None:
            chain[t] = root(best)
            ends.remove(best)
        ends.append(t)
    return {t: root(t) for t in span}


def decide(frames, dec, fps, use_stitch=True, use_group=True):
    """Whole-video decision per track: True = kept visible for its whole track."""
    ch = stitch(frames, fps) if use_stitch else {r["tid"]: r["tid"]
                                                  for rs in frames.values() for r in rs}
    on = defaultdict(bool); fr = defaultdict(set)
    for fi, rs in frames.items():
        for r in rs:
            c = ch[r["tid"]]
            fr[c].add(fi)
            if dec[(fi, r["tid"])]:
                on[c] = True
    keep = {c for c in fr if on[c]}
    if use_group:
        pf = set().union(*(fr[c] for c in keep)) if keep else set()
        for c in fr:
            if c not in keep and len(fr[c]) / fps >= GROUP_S and len(fr[c] & pf) / len(fr[c]) >= COPRES:
                keep.add(c)
    return {t: ch[t] in keep for t in ch}


def main():
    clips = {r["clip_id"]: r for r in csv.DictReader(open(R.CLIP_DIR / "clip_map.csv"))}
    labels = {r["clip_id"]: int(r["ground_truth"])
              for r in csv.DictReader(open(R.CLIP_DIR / "clip_ground_truth.csv"))}
    videos = sorted({(r["category"], r["video"]) for r in clips.values()})
    fps = {(r["category"], r["video"]): float(r["fps"]) for r in clips.values()}

    # boxes are needed for stitching; add them to the replay log
    log = R.load_log(set(videos))
    with open(LOGS_DIR / "tracking_log.csv", newline="") as f:
        boxes = {}
        for r in csv.DictReader(f):
            k = (r["category"], r["video"])
            if k in log:
                boxes[(k, int(r["frame"]), r["face_id"])] = [int(r[c]) for c in ("x1", "y1", "x2", "y2")]
    for v in videos:
        for fi, rs in log[v].items():
            for r in rs:
                r["box"] = boxes[(v, fi, r["tid"])]

    lows = [None, 0.004, 0.006, 0.008, 0.010]
    dec = {low: {v: replay_adaptive(log[v], area=0.012, low=low, need_talk=False) for v in videos}
           for low in lows}

    def preds(low, use_stitch, use_group):
        out = {}
        for v in videos:
            keep = decide(log[v], dec[low][v], fps[v], use_stitch, use_group)
            for cid, r in clips.items():
                if (r["category"], r["video"]) == v:
                    out[cid] = keep.get(r["track_id"], False)
        return out

    rows = []

    def report(name, p):
        s = R.score(labels, p)
        rows.append({"config": name, **s})
        print(f"{name:58} TP {s['tp']:2} FN {s['fn']:2} FP {s['fp']} TN {s['tn']:2} "
              f"PSR {s['psr']:6.2f} IPR {s['ipr']:6.2f} F1 {s['f1']:6.2f}")

    report("whole-track decision", preds(None, False, False))
    report("whole-track + stitching", preds(None, True, False))
    report("whole-track + group rule", preds(None, False, True))
    report("whole-track + stitching + group rule", preds(None, True, True))

    # adaptive threshold chosen by leave-one-video-out cross-validation
    by_video = defaultdict(list)
    for c in labels:
        by_video[(clips[c]["category"], clips[c]["video"])].append(c)
    cache = {low: preds(low, True, True) for low in lows}
    held, chosen = {}, []
    for v, ids in by_video.items():
        train = [c for c in labels if c not in ids]

        def bal(low):
            s = R.score({c: labels[c] for c in train}, cache[low])
            return round((s["psr"] + s["ipr"]) / 2, 6), -(low or 1)
        best = max(lows, key=bal)
        chosen.append(best)
        for c in ids:
            held[c] = cache[best][c]
    report("whole-track + stitching + group + adaptive (cross-validated)", held)
    print("lower threshold chosen per fold:", {str(l): chosen.count(l) for l in set(chosen)})
    for low in lows[1:]:
        report(f"   (in-sample) adaptive low={low}", cache[low])

    with open(LOGS_DIR / "whole_video.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "tp", "fn", "fp", "tn", "psr", "ipr", "prec", "f1"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
