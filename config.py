# =============================================================================
# config.py — Central configuration for AnonymizationAR
# =============================================================================

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / "data"
OUTPUT_DIR  = BASE_DIR / "output"
MODELS_DIR  = BASE_DIR / "models"
LOGS_DIR    = BASE_DIR / "logs"

BYSTANDER_INPUT_DIR   = DATA_DIR   / "bystander"
INTERACTION_INPUT_DIR = DATA_DIR   / "interaction"

BYSTANDER_OUTPUT_DIR   = OUTPUT_DIR / "bystander"
INTERACTION_OUTPUT_DIR = OUTPUT_DIR / "interaction"

# ── Device ─────────────────────────────────────────────────────────────────────
# "cuda" uses your RTX 4050 during development
# Switch to "cpu" when exporting for Raspberry Pi 5
DEVICE = "cpu"   # "cuda" | "cpu"

# ── Detection (YOLOv8n-face) ───────────────────────────────────────────────────
MODEL_PATH            = MODELS_DIR / "yolov8n-face.pt"
DETECTION_CONFIDENCE  = 0.45      # minimum confidence to accept a detection
DETECTION_IOU         = 0.45      # NMS IoU threshold
INPUT_SIZE            = 640       # YOLO input resolution (keep as power of 2)
MAX_FACES             = 20        # maximum faces to detect per frame

# ── Tracking (ByteTrack) ───────────────────────────────────────────────────────
TRACK_MAX_AGE         = 30        # frames to keep a lost track alive
TRACK_MIN_HITS        = 3         # detections before a track is confirmed
TRACK_IOU_THRESHOLD   = 0.3       # IoU for matching detections to tracks

# ── Video processing ───────────────────────────────────────────────────────────
FRAME_WIDTH   = 1280              # resize frames to this width (0 = no resize)
FRAME_HEIGHT  = 720               # resize frames to this height (0 = no resize)
TARGET_FPS    = 30                # target playback / output FPS

# ── Visualizer (debug mode) ────────────────────────────────────────────────────
DEBUG_MODE        = True          # draw boxes and IDs on debug output video
BOX_COLOR         = (0, 255, 0)   # BGR — green for tracked faces
BOX_THICKNESS     = 2
LABEL_COLOR       = (0, 255, 0)
LABEL_FONT_SCALE  = 0.6
LABEL_THICKNESS   = 2

# ── Output video codec ─────────────────────────────────────────────────────────
FOURCC = "mp4v"                   # codec for cv2.VideoWriter