# =============================================================================
# eval/replay_classifier.py — Re-run the classifier offline from logged cues
# =============================================================================
#
# Every face the pipeline saw is in logs/tracking_log.csv together with the cue
# values the classifier used (area ratio, centre distance, yaw, MAR). Detection
# and tracking do not depend on the classifier, so replaying the classifier on
# these logged cues gives exactly the decisions a full re-run would give, in
# seconds instead of hours.
#
# Used for:
#   - checking the replay reproduces the logged decisions (sanity check)
#   - ablation on the SAME labelled clips: geometry only / + yaw veto /
#     + speaking boost / full system
#   - sensitivity of the results to the area threshold
#
# Usage:
#   python eval/replay_classifier.py            # verify + ablation + sweep
# =============================================================================

import csv
import math
import sys
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import (
    LOGS_DIR, INTERACT_MIN_AREA_RATIO, INTERACT_MAX_CENTER_DIST,
    INTERACT_ENTER_FRAMES, INTERACT_BIG_AREA_RATIO, INTERACT_MAX_YAW,
    VVAD_WINDOW_FRAMES, VVAD_MIN_MAR, VVAD_MIN_ACTIVE_RATIO,
)

CLIP_DIR = ROOT / "eval" / "clip_package"
OUT_CSV = LOGS_DIR / "replay_ablation.csv"


def _f(v):
    return float(v) if v not in ("", None) else None


def load_log(videos):
    """(cat, video) -> {frame: [row, ...]} for the requested videos."""
    by = defaultdict(lambda: defaultdict(list))
    with open(LOGS_DIR / "tracking_log.csv", newline="") as f:
        for r in csv.DictReader(f):
            k = (r["category"], r["video"])
            if k in videos:
                by[k][int(r["frame"])].append({
                    "tid": r["face_id"], "area": float(r["area_ratio"]),
                    "cdist": float(r["center_dist"]), "yaw": _f(r["yaw"]),
                    "mar": _f(r["mar"]), "logged": r["interacting"] == "1"})
    return by


def replay(frames, use_yaw=True, use_vad=True, area=INTERACT_MIN_AREA_RATIO,
           big=INTERACT_BIG_AREA_RATIO):
    """Replays the classifier. Returns {(frame, tid): interacting}."""
    states, out, last_frame = {}, {}, None
    for fi in sorted(frames):
        rows = frames[fi]
        seen = {r["tid"] for r in rows}
        # classifier drops state of tracks absent from a processed frame
        if last_frame is not None and fi != last_frame + 1:
            states = {}
        for r in rows:
            st = states.setdefault(r["tid"], {"on": False, "streak": 0,
                                              "mar": deque(maxlen=VVAD_WINDOW_FRAMES)})
            yaw = r["yaw"] if use_yaw else None
            mar = r["mar"] if use_vad else None
            if yaw is not None and abs(yaw) > INTERACT_MAX_YAW:
                say = False
            else:
                prox = r["area"] >= area
                geo = prox and r["cdist"] <= INTERACT_MAX_CENTER_DIST
                if mar is not None:
                    st["mar"].append(mar)
                talking = False
                if len(st["mar"]) >= max(1, VVAD_WINDOW_FRAMES // 3):
                    act = sum(1 for m in st["mar"] if m >= VVAD_MIN_MAR) / len(st["mar"])
                    talking = act >= VVAD_MIN_ACTIVE_RATIO
                say = (talking and prox and r["cdist"] <= INTERACT_MAX_CENTER_DIST * 1.2) or geo
            st["streak"] = st["streak"] + 1 if say else 0
            if not st["on"] and (r["area"] >= big or st["streak"] >= INTERACT_ENTER_FRAMES):
                st["on"] = True
            out[(fi, r["tid"])] = st["on"]
        for tid in list(states):
            if tid not in seen:
                del states[tid]
        last_frame = fi
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def score(labels, preds):
    tp = sum(1 for c, g in labels.items() if g and preds[c])
    fn = sum(1 for c, g in labels.items() if g and not preds[c])
    fp = sum(1 for c, g in labels.items() if not g and preds[c])
    tn = sum(1 for c, g in labels.items() if not g and not preds[c])
    psr = 100 * tn / (tn + fp)
    ipr = 100 * tp / (tp + fn)
    prec = 100 * tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * ipr / (prec + ipr) if prec + ipr else 0.0
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, psr=psr, ipr=ipr, prec=prec, f1=f1)


def main():
    clips = {r["clip_id"]: r for r in csv.DictReader(open(CLIP_DIR / "clip_map.csv"))}
    labels = {r["clip_id"]: int(r["ground_truth"])
              for r in csv.DictReader(open(CLIP_DIR / "clip_ground_truth.csv"))}
    videos = {(r["category"], r["video"]) for r in clips.values()}
    log = load_log(videos)

    def clip_preds(**kw):
        dec = {v: replay(log[v], **kw) for v in videos}
        preds = {}
        for cid, r in clips.items():
            d = dec[(r["category"], r["video"])]
            s, e, tid = int(r["start_frame"]), int(r["end_frame"]), r["track_id"]
            win = [d[(fi, tid)] for fi in range(s, e + 1) if (fi, tid) in d]
            preds[cid] = bool(win) and sum(win) / len(win) >= 0.5
        return preds, dec

    # 1. sanity check against the logged decisions
    full, dec = clip_preds()
    mism = sum(1 for v in videos for fi, rows in log[v].items() for r in rows
               if dec[v][(fi, r["tid"])] != r["logged"])
    total = sum(len(rows) for v in videos for rows in log[v].values())
    same = sum(1 for c in clips if full[c] == (clips[c]["system_prediction"] == "1"))
    print(f"[Replay] {total - mism}/{total} logged decisions reproduced; "
          f"{same}/{len(clips)} clip decisions match")

    rows = []
    for name, kw in [("geometry only (Phase 2a)", dict(use_yaw=False, use_vad=False)),
                     ("geometry + yaw veto", dict(use_yaw=True, use_vad=False)),
                     ("geometry + speaking boost", dict(use_yaw=False, use_vad=True)),
                     ("full system (Phase 2b)", dict())]:
        p, _ = clip_preds(**kw)
        s = score(labels, p)
        rows.append({"experiment": "ablation", "config": name, **s})
        print(f"  {name:28} TP {s['tp']:2} FN {s['fn']:2} FP {s['fp']} TN {s['tn']:2} "
              f"PSR {s['psr']:6.2f} IPR {s['ipr']:6.2f} F1 {s['f1']:6.2f}")

    print("[Replay] area threshold sweep (full system)")
    for a in (0.004, 0.006, 0.008, 0.010, 0.012, 0.015, 0.020):
        p, _ = clip_preds(area=a)
        s = score(labels, p)
        rows.append({"experiment": "area_sweep", "config": f"{a:.3f}", **s})
        print(f"  area {a:.3f}  TP {s['tp']:2} FN {s['fn']:2} FP {s['fp']} TN {s['tn']:2} "
              f"PSR {s['psr']:6.2f} IPR {s['ipr']:6.2f} F1 {s['f1']:6.2f}")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows([{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}
                     for r in rows])
    print(f"[Replay] saved {OUT_CSV}")


if __name__ == "__main__":
    main()
