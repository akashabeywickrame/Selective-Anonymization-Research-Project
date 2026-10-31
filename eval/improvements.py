# =============================================================================
# eval/improvements.py — Two improvements from the future work, tested fairly
# =============================================================================
#
# 1. Adaptive proximity threshold. Most missed partners were friends whose faces
#    were just below the 1.2% area threshold. Variants:
#      - a lower fixed threshold;
#      - an adaptive rule: a face below the threshold still counts as close if
#        it is above a lower threshold AND the landmarks show it facing the
#        wearer (|yaw| <= 20 deg), optionally also speaking.
#    Choosing the threshold on the same clips used to score it would overstate
#    the result, so the parameters are chosen by leave-one-video-out
#    cross-validation: for each of the 21 videos, the setting is picked on the
#    other 20 videos and applied to the held-out one. The pooled held-out
#    predictions give an honest estimate.
#
# 2. Delayed live output. The retroactive latch needs the whole recording. A
#    live system that holds frames back by D seconds can show a face as visible
#    if the track is confirmed at any time up to D seconds later. This simulates
#    D = 0 (live) ... infinity (offline).
#
# Usage:  python eval/improvements.py   ->  logs/improvements.csv
# =============================================================================

import csv
import sys
from collections import defaultdict, deque
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
import replay_classifier as R
from config import (LOGS_DIR, INTERACT_MAX_CENTER_DIST, INTERACT_ENTER_FRAMES,
                    INTERACT_BIG_AREA_RATIO, INTERACT_MAX_YAW, VVAD_WINDOW_FRAMES,
                    VVAD_MIN_MAR, VVAD_MIN_ACTIVE_RATIO)


def replay_adaptive(frames, area, low=None, yaw_front=20.0, need_talk=True):
    """The classifier with an optional landmark-gated lower area threshold."""
    states, out, last = {}, {}, None
    for fi in sorted(frames):
        rows = frames[fi]
        if last is not None and fi != last + 1:
            states = {}
        seen = set()
        for r in rows:
            seen.add(r["tid"])
            st = states.setdefault(r["tid"], {"on": False, "streak": 0,
                                              "mar": deque(maxlen=VVAD_WINDOW_FRAMES)})
            yaw, mar = r["yaw"], r["mar"]
            if yaw is not None and abs(yaw) > INTERACT_MAX_YAW:
                say = False
            else:
                if mar is not None:
                    st["mar"].append(mar)
                talking = False
                if len(st["mar"]) >= max(1, VVAD_WINDOW_FRAMES // 3):
                    act = sum(1 for m in st["mar"] if m >= VVAD_MIN_MAR) / len(st["mar"])
                    talking = act >= VVAD_MIN_ACTIVE_RATIO
                prox = r["area"] >= area
                if not prox and low is not None and r["area"] >= low and yaw is not None \
                        and abs(yaw) <= yaw_front and (talking or not need_talk):
                    prox = True
                geo = prox and r["cdist"] <= INTERACT_MAX_CENTER_DIST
                say = (talking and prox and r["cdist"] <= INTERACT_MAX_CENTER_DIST * 1.2) or geo
            st["streak"] = st["streak"] + 1 if say else 0
            if not st["on"] and (r["area"] >= INTERACT_BIG_AREA_RATIO or
                                 st["streak"] >= INTERACT_ENTER_FRAMES):
                st["on"] = True
            out[(fi, r["tid"])] = st["on"]
        for tid in list(states):
            if tid not in seen:
                del states[tid]
        last = fi
    return out


def clip_preds(dec, clips):
    preds = {}
    for cid, r in clips.items():
        d = dec[(r["category"], r["video"])]
        s, e, tid = int(r["start_frame"]), int(r["end_frame"]), r["track_id"]
        win = [d[(fi, tid)] for fi in range(s, e + 1) if (fi, tid) in d]
        preds[cid] = bool(win) and sum(win) / len(win) >= 0.5
    return preds


def main():
    clips = {r["clip_id"]: r for r in csv.DictReader(open(R.CLIP_DIR / "clip_map.csv"))}
    labels = {r["clip_id"]: int(r["ground_truth"])
              for r in csv.DictReader(open(R.CLIP_DIR / "clip_ground_truth.csv"))}
    videos = sorted({(r["category"], r["video"]) for r in clips.values()})
    log = R.load_log(set(videos))
    rows = []

    # ── 1. adaptive threshold, leave-one-video-out ────────────────────────────
    families = {
        "fixed threshold": [dict(area=a) for a in
                            (0.006, 0.008, 0.009, 0.010, 0.011, 0.012, 0.015)],
        "adaptive: facing and speaking": [dict(area=0.012, low=l, need_talk=True)
                                          for l in (0.004, 0.006, 0.008, 0.010)],
        "adaptive: facing": [dict(area=0.012, low=l, need_talk=False)
                             for l in (0.004, 0.006, 0.008, 0.010)],
    }
    cache = {}
    for fam, grid in families.items():
        for kw in grid:
            key = tuple(sorted(kw.items()))
            if key not in cache:
                dec = {v: replay_adaptive(log[v], **kw) for v in videos}
                cache[key] = clip_preds(dec, clips)

    def bal(preds, ids):
        s = R.score({c: labels[c] for c in ids}, preds)
        return (s["psr"] + s["ipr"]) / 2

    by_video = defaultdict(list)
    for c in labels:
        by_video[(clips[c]["category"], clips[c]["video"])].append(c)

    base = cache[(("area", 0.012),)]
    s0 = R.score(labels, base)
    rows.append({"experiment": "threshold_cv", "config": "current system (1.2%)", **s0})
    print(f"current system        PSR {s0['psr']:6.2f} IPR {s0['ipr']:6.2f} F1 {s0['f1']:6.2f}")
    for fam, grid in families.items():
        keys = [tuple(sorted(kw.items())) for kw in grid]
        held = {}
        chosen = []
        for v, ids in by_video.items():
            train = [c for c in labels if c not in ids]
            # balanced accuracy on the other 20 videos; ties go to the higher threshold
            best = max(keys, key=lambda k: (round(bal(cache[k], train), 6), dict(k).get("low", dict(k)["area"])))
            chosen.append(dict(best))
            for c in ids:
                held[c] = cache[best][c]
        s = R.score(labels, held)
        common = max(set(map(str, chosen)), key=lambda x: list(map(str, chosen)).count(x))
        rows.append({"experiment": "threshold_cv", "config": fam + " (cross-validated)", **s,
                     "note": f"most often chosen: {common}"})
        print(f"{fam:30} CV  PSR {s['psr']:6.2f} IPR {s['ipr']:6.2f} F1 {s['f1']:6.2f}  chosen {common}")
        for k in keys:
            s = R.score(labels, cache[k])
            rows.append({"experiment": "threshold_all", "config": f"{fam} {dict(k)}", **s})

    # ── 2. delayed live output ────────────────────────────────────────────────
    confirm = {}   # (video, tid) -> sorted frames where the track was confirmed
    for v in videos:
        for fi, rs in log[v].items():
            for r in rs:
                if r["logged"]:
                    confirm.setdefault((v, r["tid"]), []).append(fi)
    for k in confirm:
        confirm[k].sort()
    for d in (0, 1, 2, 3, 5, 10, 20, None):
        preds = {}
        for cid, r in clips.items():
            v = (r["category"], r["video"]); tid = r["track_id"]
            fps = float(r["fps"]); s, e = int(r["start_frame"]), int(r["end_frame"])
            first = confirm.get((v, tid), [])
            win = []
            for fi in range(s, e + 1):
                rr = [x for x in log[v].get(fi, []) if x["tid"] == tid]
                if not rr:
                    continue
                if rr[0]["logged"]:
                    win.append(True); continue
                horizon = float("inf") if d is None else fi + d * fps
                win.append(any(c <= horizon for c in first))
            preds[cid] = bool(win) and sum(win) / len(win) >= 0.5
        s = R.score(labels, preds)
        name = "offline (whole recording)" if d is None else f"{d} s delay"
        rows.append({"experiment": "delay", "config": name, **s})
        print(f"delay {name:26} PSR {s['psr']:6.2f} IPR {s['ipr']:6.2f} F1 {s['f1']:6.2f}")

    out = LOGS_DIR / "improvements.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment", "config", "tp", "fn", "fp", "tn",
                                          "psr", "ipr", "prec", "f1", "note"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    print("saved", out)


if __name__ == "__main__":
    main()
