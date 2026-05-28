# Context-Aware Privacy for AR
### Real-Time Selective Anonymization via Egocentric Interaction Detection on Edge Devices

> Undergraduate Research Project — Department of Statistics and Computer Science  
> University of Peradeniya, Sri Lanka  
> **Akash Abeywickrame (S/20/302)**

---

## Overview

This project develops a **lightweight, context-aware privacy framework** that runs entirely on edge hardware (Raspberry Pi 5). Instead of blindly blurring every face in view, the system distinguishes between:

- **Interactors** — people actively engaging with the AR wearer (preserved, not anonymized)
- **Bystanders** — non-interacting people nearby (anonymized in real time)

The pipeline is designed for deployment on AR glasses where compute, memory, and power are severely constrained.

---

## The Problem

Existing privacy solutions for AR devices such as blanket face blurring (Google Maps, news footage) or LED indicators (Meta Ray-Ban glasses) are impractical for wearable use:

| Problem | Impact |
|---|---|
| Continuous detection is expensive | High power draw near the user's head |
| Indiscriminate blurring | Obscures socially relevant interactions |
| No context awareness | Cannot distinguish a conversation partner from a stranger |
| LED indicators invisible outdoors | No guarantee bystanders are notified |

---

## Proposed Solution

```
Egocentric video stream
        │
        ▼
┌─────────────────────┐
│  Pre-processing      │  Resize, normalize, frame rate control
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Face Detection      │  YOLOv8n-face — lightweight, crowd-accurate
│  + Tracking          │  ByteTrack — stable IDs, no embedding model needed
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Interaction Cues    │  Proximity · Head pose (yaw/pitch) · Mouth movement (V-VAD)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Decision Layer      │  Heuristic classifier → INTERACTOR or BYSTANDER
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
Preserve   Anonymize
identity   (Gaussian blur)
    │         │
    └────┬────┘
         ▼
  Privacy-protected output
```

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| Face Detection | **YOLOv8n-face** | Accurate in crowds, exportable to ONNX/TFLite |
| Face Tracking | **ByteTrack** | Motion-only — no embedding model, ultra lightweight |
| Inference | **PyTorch (dev) → ONNX (deploy)** | GPU on laptop, CPU on Raspberry Pi 5 |
| Anonymization | **Gaussian blur** | Near-zero compute cost |

> **Why not MediaPipe / BlazeFace?**  
> BlazeFace is optimized for 1–2 close-up faces. In crowd scenes with fast motion and occlusion it loses tracks and produces ID conflicts — confirmed during early testing on this dataset.

---

## Project Structure

```
AnonymizationAR/
│
├── main.py                        # Entry point — runs full pipeline
├── config.py                      # All settings (device, thresholds, paths)
├── requirements.txt
│
├── core/
│   ├── detector.py                # YOLOv8n-face detection
│   ├── tracker.py                 # ByteTrack stable face ID assignment
│   ├── interaction_cues.py        # Proximity, head pose, V-VAD  [Phase 2]
│   ├── classifier.py              # Interactor vs bystander logic  [Phase 3]
│   ├── anonymizer.py              # Gaussian blur bystander faces  [Phase 4]
│   └── visualizer.py             # Debug rendering (boxes + IDs)
│
├── utils/
│   ├── video_loader.py            # VideoCapture wrapper + resize
│   └── metrics.py                 # PSR, IPR, FPS, latency  [Phase 4]
│
├── models/
│   └── yolov8n-face.pt            # Auto-downloaded on first run (~6 MB)
│
├── data/
│   ├── bystander/                 # Input: bystander scenario videos
│   └── interaction/               # Input: interaction scenario videos
│
├── output/
│   ├── bystander/                 # video_tracked.mp4 + video_debug.mp4
│   └── interaction/
│
├── logs/
│   └── tracking_log.csv           # Per-frame: face_id, bbox, confidence
│
└── eval/
    ├── evaluate.py                # PSR + IPR evaluation script  [Phase 4]
    └── ground_truth/              # Manually labelled validation CSVs
```

---

## Installation

### Requirements
- Python 3.12+
- Windows / Linux
- NVIDIA GPU recommended for development (CPU also works)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/AnonymizationAR.git
cd AnonymizationAR

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### GPU Support (Optional but Recommended)

If you have an NVIDIA GPU, install CUDA-enabled PyTorch for full speed:

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA is available:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Usage

Place your `.mp4` videos in the appropriate data folder:

```
data/bystander/video1.mp4
data/interaction/video1.mp4
```

Then run:

```bash
# Process all videos
python main.py

# Process only bystander videos
python main.py --category bystander

# Process only interaction videos
python main.py --category interaction

# Process a single video
python main.py --video data/bystander/video1.mp4

# Show real-time tracking window while processing
python main.py --category bystander --preview
```

### Preview Window Controls

| Key | Action |
|---|---|
| `ESC` | Stop processing current video |
| `SPACE` | Pause / Resume |

---

## Output

Each processed video produces two files:

| File | Contents |
|---|---|
| `output/.../video_tracked.mp4` | Clean original frames (no annotations) |
| `output/.../video_debug.mp4` | Annotated frames — colored boxes + Face IDs |

A CSV log is saved at `logs/tracking_log.csv`:

```
video,       category,   frame, face_id, x1,  y1,  x2,  y2,  confidence
video1.mp4,  bystander,  0012,  1,       120, 45,  200, 135, 0.94
video1.mp4,  bystander,  0012,  2,       300, 60,  375, 145, 0.91
```

---

## Configuration

All tunable parameters are in `config.py`:

```python
DEVICE               = "cuda"   # "cuda" for GPU | "cpu" for Raspberry Pi 5
DETECTION_CONFIDENCE = 0.45     # lower = detect more faces (more false positives)
TRACK_MAX_AGE        = 30       # frames to keep a lost face track alive
FRAME_WIDTH          = 1280     # resize input frames (0 = no resize)
DEBUG_MODE           = True     # write annotated debug video
```

---

## Research Phases

| Phase | Description | Status |
|---|---|---|
| **Phase 1** | Face detection + stable tracking | ✅ Complete |
| **Phase 2** | Interaction cue extraction (proximity, head pose, V-VAD) | 🔄 In Progress |
| **Phase 3** | Interactor vs bystander classifier | ⏳ Pending |
| **Phase 4** | Selective anonymization + evaluation metrics (PSR, IPR) | ⏳ Pending |
| **Phase 5** | ONNX export + Raspberry Pi 5 deployment | ⏳ Pending |

---

## Evaluation Metrics

| Metric | Formula | Target |
|---|---|---|
| **Privacy Success Rate (PSR)** | Bystanders blurred / Total bystanders detected | > 95% |
| **Interaction Preservation Rate (IPR)** | Interactions preserved / Total interactions | > 95% |
| **Throughput** | Frames per second on Raspberry Pi 5 | > 30 FPS |
| **Inference Latency** | ms per frame | < 33ms |

---

## Hardware Targets

| Environment | Hardware | Notes |
|---|---|---|
| Development | Laptop + RTX 4050 6GB | CUDA inference |
| Final Benchmark | Raspberry Pi 5 (8GB) | CPU-only, ARM Cortex-A76 |
| Target Deployment | AR Glasses (Snapdragon XR2) | ONNX / TFLite |

The Raspberry Pi 5 is used as a **worst-case benchmark** — if the pipeline achieves real-time performance on a CPU-only ARM device, it is feasible for mass-market AR wearables.

---

## References

1. Corbett et al., *BystandAR: Protecting Bystander Visual Data in Augmented Reality Systems*, MobiSys 2023
2. Al Aaraj, *BystandARIA: Enabling AR Bystander Privacy using LEDs*, MobiHoc 2025
3. O'Hagan et al., *Privacy-enhancing technology and everyday augmented reality*, IMWUT 2023
4. Meta Reality Labs, *EgoBlur: Privacy-Preserving Head-Mounted Egocentric Vision*, arXiv 2023
5. Karakostas et al., *A real-time wearable AR system for egocentric vision on the edge*, Virtual Reality 2024

---

## License

This project is developed for academic research purposes at the University of Peradeniya, Sri Lanka.  
© 2026 Akash Abeywickrame. All rights reserved.
