# =============================================================================
# eval/benchmark_optimised.py — Speed of the pipeline with an optimised detector
# =============================================================================
#
# Face detection takes most of the processing time, so this script measures how
# much faster the same pipeline runs when only the detector is changed:
#
#   1. Runtime: the same YOLOv8n-face weights exported to ONNX Runtime and to
#      Intel OpenVINO, instead of PyTorch.
#   2. Input size: the detector run at 416 or 320 pixels instead of 640.
#   3. Detection interval: the detector run on every 2nd or 3rd frame, with the
#      previous decisions (and blur boxes) held on the frames in between.
#
# Speed alone is not enough: a faster detector that misses faces would leave
# bystanders unblurred. So for each variant the script also measures
#   - face recall: the share of faces found by the original PyTorch 640 detector
#     that the variant also finds (IoU >= 0.5), on frames from 8 videos, and
#   - for the interval variants, face coverage: the share of faces present on a
#     skipped frame that are still covered by a held box (IoU >= 0.3).
#
# Usage (laptop plugged in, other apps closed):
#   python eval/benchmark_optimised.py
# Output: logs/benchmark_optimised.csv
# =============================================================================

import csv
import statistics
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import core.detector as detmod
from config import (LOGS_DIR, INTERACTION_INPUT_DIR, BYSTANDER_INPUT_DIR,
                    FRAME_WIDTH as W, FRAME_HEIGHT as H)
from core.tracker import FaceTracker
from core.classifier import InteractionClassifier
from core.anonymizer import SelectiveAnonymizer
from ultralytics import YOLO

EXP = ROOT / "models" / "exported"
VARIANTS = [
    # name, weights, input size, detect every N frames
    ("PyTorch, 640",               ROOT / "models" / "yolov8n-face.pt",       640, 1),
    ("ONNX Runtime, 640",          EXP / "yolov8n-face_onnx_640.onnx",        640, 1),
    ("OpenVINO, 640",              EXP / "yolov8n-face_640_openvino_model",   640, 1),
    ("OpenVINO, 416",              EXP / "yolov8n-face_416_openvino_model",   416, 1),
    ("OpenVINO, 320",              EXP / "yolov8n-face_320_openvino_model",   320, 1),
    ("OpenVINO, 640, every 2nd",   EXP / "yolov8n-face_640_openvino_model",   640, 2),
    ("OpenVINO, 640, every 3rd",   EXP / "yolov8n-face_640_openvino_model",   640, 3),
    ("OpenVINO, 416, every 2nd",   EXP / "yolov8n-face_416_openvino_model",   416, 2),
]
N_FRAMES, WARMUP, REPEATS = 300, 30, 2
BENCH_VIDEO = INTERACTION_INPUT_DIR / "video1.mp4"


class _Detector:
    """The pipeline's face detector with its own input size, so several detectors
    with different sizes can be used side by side."""

    def __init__(self, weights, size):
        self.model = YOLO(str(weights), task="pose")
        self.size = size

    def detect(self, frame):
        detmod.INPUT_SIZE = self.size
        return detmod.FaceDetector.detect(self, frame)


def make_detector(weights, size):
    return _Detector(weights, size)


def load_frames(path, start, n):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def run_pipeline(weights, size, every, phase, extractor):
    """Times the full pipeline over N_FRAMES; returns ms per frame by stage."""
    det = make_detector(weights, size)
    cap = cv2.VideoCapture(str(BENCH_VIDEO))
    trk, cls, ano = FaceTracker(frame_size=(W, H)), InteractionClassifier(frame_size=(W, H)), SelectiveAnonymizer()
    t = dict(read=0.0, detect=0.0, track=0.0, classify=0.0, anon=0.0)
    held = []
    for i in range(WARMUP + N_FRAMES):
        s = time.perf_counter()
        ok, frame = cap.read()
        frame = cv2.resize(frame, (W, H))
        t_read = time.perf_counter() - s
        t_det = t_trk = t_cls = 0.0
        if i % every == 0:
            s = time.perf_counter(); dets = det.detect(frame); t_det = time.perf_counter() - s
            s = time.perf_counter(); tracks = trk.update(dets, frame); t_trk = time.perf_counter() - s
            s = time.perf_counter()
            held = (cls.classify(tracks, frame=frame, extractor=extractor) if phase == "2b"
                    else cls.classify(tracks))
            t_cls = time.perf_counter() - s
        s = time.perf_counter(); ano.apply(frame, held); t_ano = time.perf_counter() - s
        if i >= WARMUP:
            t["read"] += t_read; t["detect"] += t_det; t["track"] += t_trk
            t["classify"] += t_cls; t["anon"] += t_ano
    cap.release()
    return {k: 1000 * v / N_FRAMES for k, v in t.items()}


def recall_check(variants):
    """Face recall of each variant against PyTorch 640, on frames from 8 videos."""
    vids = ([INTERACTION_INPUT_DIR / f"video{n}.mp4" for n in (1, 4, 10, 13)] +
            [BYSTANDER_INPUT_DIR / f"video{n}.mp4" for n in (1, 3, 5, 7)])
    frames = []
    for v in vids:
        cap = cv2.VideoCapture(str(v)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        for k in range(25):
            frames += load_frames(v, int(n * (k + 0.5) / 25), 1)
    frames = [cv2.resize(f, (W, H)) for f in frames]
    ref = make_detector(*variants[0][1:3])
    refs = [[d["bbox"] for d in ref.detect(f)] for f in frames]
    out = {}
    for name, w, size, every in variants:
        if every != 1:
            continue
        d = make_detector(w, size)
        found = total = small_found = small_total = extra = 0
        for f, rb in zip(frames, refs):
            vb = [x["bbox"] for x in d.detect(f)]
            for r in rb:
                hit = any(iou(r, v) >= 0.5 for v in vb)
                total += 1; found += hit
                if (r[2] - r[0]) * (r[3] - r[1]) < 0.012 * W * H:
                    small_total += 1; small_found += hit
            extra += sum(1 for v in vb if not any(iou(r, v) >= 0.5 for r in rb))
        out[(w, size)] = dict(recall=100 * found / total, small_recall=100 * small_found / small_total,
                              extra=extra, ref_faces=total)
    return out, len(frames)


def coverage_check(weights, size, every):
    """Share of faces on skipped frames still covered by the held boxes."""
    d = make_detector(weights, size)
    ref = make_detector(VARIANTS[0][1], 640)
    cov = tot = 0
    for v in (BENCH_VIDEO, BYSTANDER_INPUT_DIR / "video5.mp4"):
        held = []
        for i, f in enumerate(load_frames(v, 0, 300)):
            f = cv2.resize(f, (W, H))
            if i % every == 0:
                held = [x["bbox"] for x in d.detect(f)]
                continue
            for r in ref.detect(f):
                tot += 1
                cov += any(iou(r["bbox"], h) >= 0.3 for h in held)
    return 100 * cov / tot


def main():
    from core.landmark_extractor import LandmarkExtractor
    ext = LandmarkExtractor()
    print("[Bench] checking face recall against PyTorch 640 ...")
    rec, nframes = recall_check(VARIANTS)
    rows = []
    for name, w, size, every in VARIANTS:
        res = {}
        for phase in ("2a", "2b"):
            runs = [run_pipeline(w, size, every, phase, ext if phase == "2b" else None)
                    for _ in range(REPEATS)]
            tot = [sum(r.values()) for r in runs]
            best = runs[tot.index(min(tot))]
            res[phase] = (best, 1000 / statistics.median(tot))
        r2a, fps2a = res["2a"]; r2b, fps2b = res["2b"]
        q = rec.get((w, size), {})
        cov = coverage_check(w, size, every) if every > 1 else None
        row = {"variant": name, "detect_ms": round(r2a["detect"], 1),
               "track_ms": round(r2a["track"], 2), "read_ms": round(r2a["read"], 1),
               "landmarks_ms": round(r2b["classify"] - r2a["classify"], 1),
               "fps_2a": round(fps2a, 1), "fps_2b": round(fps2b, 1),
               "face_recall": round(q["recall"], 1) if q else "",
               "small_face_recall": round(q["small_recall"], 1) if q else "",
               "extra_detections": q.get("extra", ""),
               "skipped_frame_coverage": round(cov, 1) if cov is not None else ""}
        rows.append(row)
        print("  ", row, flush=True)
        out = LOGS_DIR / "benchmark_optimised.csv"
        with open(out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader(); wr.writerows(rows)
    print(f"[Bench] {nframes} frames used for the recall check; saved {out}")


if __name__ == "__main__":
    main()
