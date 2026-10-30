# =============================================================================
# eval/rerun_openvino.py — Re-run the pipeline with the OpenVINO detector
# =============================================================================
#
# The labelled evaluation used the original PyTorch detector. This script runs
# the same pipeline (same tracker, cues and classifier) on all 21 evaluation
# videos with the faster OpenVINO export of the same detector, logging every
# decision to logs/tracking_log_openvino.csv. No videos are written, so it is
# much faster than main.py. eval/score_openvino.py then scores it against the
# human labels of the video-clip study.
#
# Usage:  python eval/rerun_openvino.py          (always starts a fresh log)
# =============================================================================

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from config import LOGS_DIR, INTERACTION_INPUT_DIR, BYSTANDER_INPUT_DIR
from core.tracker import FaceTracker
from core.classifier import InteractionClassifier
from core.landmark_extractor import LandmarkExtractor
from utils.video_loader import VideoLoader
from main import TrackingLogger
from benchmark_optimised import make_detector

WEIGHTS = ROOT / "models" / "exported" / "yolov8n-face_640_openvino_model"
LOG = LOGS_DIR / "tracking_log_openvino.csv"
JOBS = ([("bystander", BYSTANDER_INPUT_DIR / f"video{n}.mp4") for n in range(1, 8)] +
        [("interaction", INTERACTION_INPUT_DIR / f"video{n}.mp4") for n in range(1, 15)])


def main():
    det = make_detector(WEIGHTS, 640)
    ext = LandmarkExtractor()
    logger = TrackingLogger(LOG, append=False)   # fresh log: no duplicate rows
    for cat, path in JOBS:
        t0 = time.perf_counter()
        with VideoLoader(path) as loader:
            w, h = loader.frame_size
            trk, cls = FaceTracker(frame_size=(w, h)), InteractionClassifier(frame_size=(w, h))
            for fi, frame in enumerate(loader):
                tracks = trk.update(det.detect(frame), frame)
                logger.log(path.name, cat, fi, cls.classify(tracks, frame=frame, extractor=ext))
        logger._file.flush()
        print(f"[OpenVINO] {cat}/{path.name}: {fi + 1} frames in "
              f"{time.perf_counter() - t0:.0f} s", flush=True)
    print("[OpenVINO] done", flush=True)


if __name__ == "__main__":
    main()
