# =============================================================================
# eval/measure_power.py — Measure the pipeline's power draw on battery
# =============================================================================
#
# Windows reports the battery discharge rate (mW) through WMI
# (root\wmi BatteryStatus.DischargeRate). With the laptop UNPLUGGED and other
# apps closed, this script records:
#
#   1. idle      : the laptop doing nothing                   (baseline power)
#   2. phase2a   : the pipeline running, geometry cues only
#   3. phase2b   : the pipeline running, with landmark cues
#
# Pipeline power = mean(running) - mean(idle). Dividing by the measured frame
# rate gives the energy per processed frame (joules/frame).
#
# Usage (laptop on battery, everything else closed):
#   python eval/measure_power.py                 # 120 s per condition
#   python eval/measure_power.py --seconds 180
#
# Output: logs/power_measurement.csv
# =============================================================================

import argparse
import csv
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import LOGS_DIR

PS = ["powershell", "-NoProfile", "-Command"]
QUERY = ("$b = Get-CimInstance -Namespace root\\wmi -ClassName BatteryStatus; "
         "\"$($b.PowerOnline),$($b.DischargeRate)\"")


def read_battery():
    """Returns (on_ac_power, discharge_mW)."""
    out = subprocess.run(PS + [QUERY], capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stdout.strip()
    online, rate = out.split(",")
    return online.strip().lower() == "true", int(rate)


class Sampler(threading.Thread):
    """Samples the discharge rate about once a second until stopped."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            on_ac, mw = read_battery()
            if on_ac:
                print("\n[Power] Charger connected, measurement invalid. Unplug and rerun.")
                self.samples = []
                return
            self.samples.append(mw / 1000.0)
            time.sleep(0.5)

    def stop(self):
        self._stop.set()
        self.join()


def run_condition(name, seconds, workload=None):
    print(f"\n[Power] {name}: {seconds} s ...")
    sampler = Sampler()
    sampler.start()
    t0 = time.perf_counter()
    frames = 0
    if workload is None:
        time.sleep(seconds)
    else:
        while time.perf_counter() - t0 < seconds:
            frames += workload()
    elapsed = time.perf_counter() - t0
    sampler.stop()
    s = sampler.samples[3:]            # drop the first samples while the load settles
    if len(s) < 5:
        raise SystemExit("[Power] Not enough samples (charger connected?).")
    watts = statistics.mean(s)
    fps = frames / elapsed if frames else 0.0
    print(f"  mean draw {watts:.2f} W over {len(s)} samples"
          + (f", {fps:.2f} FPS" if frames else ""))
    return {"condition": name, "seconds": round(elapsed, 1), "samples": len(s),
            "mean_w": round(watts, 3), "sd_w": round(statistics.pstdev(s), 3),
            "fps": round(fps, 2)}


def make_workload(phase):
    """Runs the real pipeline over a video; returns a callable that processes a
    chunk of frames and returns how many it processed."""
    import cv2
    from config import INTERACTION_INPUT_DIR, FRAME_WIDTH, FRAME_HEIGHT
    from core.detector import FaceDetector
    from core.tracker import FaceTracker
    from core.classifier import InteractionClassifier
    from core.anonymizer import SelectiveAnonymizer

    video = INTERACTION_INPUT_DIR / "video1.mp4"
    w, h = FRAME_WIDTH, FRAME_HEIGHT
    det, ano = FaceDetector(), SelectiveAnonymizer()
    state = {"cap": cv2.VideoCapture(str(video)),
             "trk": FaceTracker(frame_size=(w, h)),
             "cls": InteractionClassifier(frame_size=(w, h))}
    ext = None
    if phase == "2b":
        from core.landmark_extractor import LandmarkExtractor
        ext = LandmarkExtractor()

    def chunk(n=30):
        done = 0
        for _ in range(n):
            ok, frame = state["cap"].read()
            if not ok:                         # loop the video
                state["cap"].set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame = cv2.resize(frame, (w, h))
            tracks = state["trk"].update(det.detect(frame), frame)
            decided = (state["cls"].classify(tracks, frame=frame, extractor=ext)
                       if ext else state["cls"].classify(tracks))
            ano.apply(frame, decided)
            done += 1
        return done
    return chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=120)
    args = ap.parse_args()

    on_ac, _ = read_battery()
    if on_ac:
        raise SystemExit("[Power] The laptop is on mains power. Unplug the charger, "
                         "close other apps, and run again.")

    print("[Power] Loading models (not measured) ...")
    w2a, w2b = make_workload("2a"), make_workload("2b")
    w2a(5); w2b(5)                              # warm-up

    rows = [run_condition("idle", args.seconds),
            run_condition("phase2a", args.seconds, w2a),
            run_condition("phase2b", args.seconds, w2b)]
    idle = rows[0]["mean_w"]
    for r in rows[1:]:
        r["pipeline_w"] = round(r["mean_w"] - idle, 3)
        r["joules_per_frame"] = round(r["pipeline_w"] / r["fps"], 4) if r["fps"] else ""

    out = LOGS_DIR / "power_measurement.csv"
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["condition", "seconds", "samples", "mean_w",
                                           "sd_w", "fps", "pipeline_w", "joules_per_frame"])
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n[Power] Saved {out}")
    for r in rows[1:]:
        print(f"  {r['condition']}: pipeline adds {r['pipeline_w']:.2f} W "
              f"at {r['fps']:.2f} FPS = {r['joules_per_frame']} J/frame")


if __name__ == "__main__":
    main()
