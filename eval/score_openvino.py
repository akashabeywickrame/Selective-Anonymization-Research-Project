# =============================================================================
# eval/score_openvino.py — Score the OpenVINO re-run against the Set C labels
# =============================================================================
#
# Track numbers differ between the original run and the OpenVINO re-run, so each
# labelled clip's person is matched to the track in the new log whose boxes
# overlap the original track most during the clip (mean IoU over the clip's
# frames, at least 0.3). The clip decision then uses the same rule as before:
# kept visible if unblurred in at least half of the frames the face was detected.
#
# Usage:  python eval/score_openvino.py
# =============================================================================

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
import replay_classifier as R
from benchmark_optimised import iou
from config import LOGS_DIR


def load_boxes(path, videos):
    out = defaultdict(lambda: defaultdict(list))
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            k = (r["category"], r["video"])
            if k in videos:
                out[k][int(r["frame"])].append(
                    (r["face_id"], [int(r[c]) for c in ("x1", "y1", "x2", "y2")],
                     r["interacting"] == "1"))
    return out


def main():
    clips = {r["clip_id"]: r for r in csv.DictReader(open(R.CLIP_DIR / "clip_map.csv"))}
    labels = {r["clip_id"]: int(r["ground_truth"])
              for r in csv.DictReader(open(R.CLIP_DIR / "clip_ground_truth.csv"))}
    videos = {(r["category"], r["video"]) for r in clips.values()}
    old = load_boxes(LOGS_DIR / "tracking_log.csv", videos)
    new = load_boxes(LOGS_DIR / "tracking_log_openvino.csv", videos)
    preds, unmatched, ious = {}, [], []
    for cid in labels:
        r = clips[cid]; v = (r["category"], r["video"]); tid = r["track_id"]
        s, e = int(r["start_frame"]), int(r["end_frame"])
        target = {}
        for fi in range(s, e + 1):
            for t, b, _ in old[v].get(fi, []):
                if t == tid:
                    target[fi] = b
        score = defaultdict(float)
        for fi, b in target.items():
            for t, nb, _ in new[v].get(fi, []):
                score[t] += iou(b, nb)
        best = max(score, key=score.get) if score else None
        m = score[best] / len(target) if best else 0.0
        ious.append(m)
        if best is None or m < 0.3:
            unmatched.append(cid)
            preds[cid] = False
            continue
        dec = [on for fi in range(s, e + 1) for t, _, on in new[v].get(fi, []) if t == best]
        preds[cid] = bool(dec) and sum(dec) / len(dec) >= 0.5
    sc = R.score(labels, preds)
    same = sum(1 for c in labels if preds[c] == (clips[c]["system_prediction"] == "1"))
    print(f"[OpenVINO] matched {len(labels) - len(unmatched)}/{len(labels)} clips "
          f"(mean IoU of matches {sum(ious) / len(ious):.2f}); unmatched: {unmatched}")
    print(f"[OpenVINO] same decision as original detector on {same}/{len(labels)} clips")
    print(f"[OpenVINO] TP {sc['tp']} FN {sc['fn']} FP {sc['fp']} TN {sc['tn']}  "
          f"PSR {sc['psr']:.2f} IPR {sc['ipr']:.2f} Prec {sc['prec']:.2f} F1 {sc['f1']:.2f}")


if __name__ == "__main__":
    main()
