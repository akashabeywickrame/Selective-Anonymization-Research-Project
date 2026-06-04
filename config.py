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

# =============================================================================
# Phase 2 — Interaction-aware decision layer
# =============================================================================
#
# These thresholds drive the heuristic that decides whether a tracked face is
# an INTERACTION PARTNER (preserve) or a BYSTANDER (anonymize).
#
# Phase 2a uses ONLY bounding-box geometry (proximity + centrality) — no
# landmarks required. Head-pose (yaw/pitch) and V-VAD (mouth movement) come in
# Phase 2b once MediaPipe Face Mesh is wired in.
# -----------------------------------------------------------------------------

# ── Proximity cue (face area relative to frame) ──────────────────────────────
# A conversation partner fills more of the frame than a distant bystander.
# area_ratio = (face_w * face_h) / (frame_w * frame_h)
# Calibrated from the cue distribution: per-track max area_ratio falls into two
# clean clusters — a "near" group (≈0.022–0.075, conversation partners) and a
# "far/tiny" group (≈0.0003–0.0026, distant bystanders). 0.012 separates them.
INTERACT_MIN_AREA_RATIO   = 0.012   # face must cover ≥ 1.2% of frame to count

# ── Centrality cue (distance from optical center) ────────────────────────────
# Egocentric "center bias": the wearer points their head at who they talk to.
# center_dist is normalised 0.0 (dead center) … 1.0 (frame corner).
INTERACT_MAX_CENTER_DIST  = 0.45    # face center must be within 45% of center

# ── Temporal smoothing (hysteresis to stop flicker) ──────────────────────────
# A face must satisfy the cues for several consecutive frames before it flips
# to "interacting".
INTERACT_ENTER_FRAMES     = 2       # consecutive interacting frames to confirm

# ── "Large face = instant partner" override ──────────────────────────────────
# A face this large fills the frame — it is unmistakably someone the wearer is
# face-to-face with. Such a face is confirmed as a partner IMMEDIATELY,
# regardless of centering or the enter-streak. Raised this from blurring a
# clearly-engaged partner who happened to drift off the optical center.
INTERACT_BIG_AREA_RATIO   = 0.040   # ≥ 4% of frame → instant partner

# ── Latch behaviour ──────────────────────────────────────────────────────────
# If True: once a track is confirmed as an interaction partner it STAYS a
# partner (never re-blurred) for the rest of the video. This removes flicker
# and stops a partner from being re-blurred every time they look away.
# If False: a partner reverts to bystander after INTERACT_EXIT_FRAMES of
# failing the cues (the old hysteresis behaviour).
INTERACT_LATCH            = True
INTERACT_EXIT_FRAMES      = 10      # (only used when INTERACT_LATCH = False)

# ── Anonymization ────────────────────────────────────────────────────────────
ANON_METHOD        = "blur"         # "blur" | "pixelate"
ANON_BLUR_KERNEL   = 51             # Gaussian kernel size (odd number)
ANON_PIXELATE_SIZE = 12             # mosaic block size (pixels) when pixelating
ANON_PADDING       = 0.10           # expand bbox by 10% so blur covers full head

# ── Decision-layer visualizer colours (BGR) ──────────────────────────────────
COLOR_INTERACTING  = (0, 200, 0)    # green box  — preserved
COLOR_BYSTANDER    = (0, 0, 255)    # red box    — anonymized