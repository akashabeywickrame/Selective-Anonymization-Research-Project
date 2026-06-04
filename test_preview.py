# =============================================================================
# test_preview.py — Standalone face tracking preview
# Just edit VIDEO_PATH and run: python test_preview.py
# =============================================================================

import cv2
from pathlib import Path
from ultralytics import YOLO

# ── EDIT THIS ─────────────────────────────────────────────────────────────────
VIDEO_PATH  = r"F:\ResearchProject\data\bystander\video1.mp4"
MODEL_PATH  = r"F:\ResearchProject\models\yolov8n-face.pt"
# ─────────────────────────────────────────────────────────────────────────────

# Color palette for different face IDs
COLORS = [
    (255, 56,  56),  (56, 255,  56),  (56,  56, 255),
    (255, 255, 56),  (56, 255, 255),  (255, 56, 255),
    (255, 165,  0),  (0,  255, 127),  (138, 43, 226),
    (255, 20,  147), (0,  191, 255),  (50, 205,  50),
]

def get_color(track_id):
    return COLORS[track_id % len(COLORS)]

# ── Load model ─────────────────────────────────────────────────────────────────
print("Loading model...")
model = YOLO(MODEL_PATH)
print("Model loaded. Opening video...")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Cannot open video: {VIDEO_PATH}")
    exit()

w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video: {w}x{h} @ {fps:.1f} FPS")
print("Window opening... Press ESC to quit, SPACE to pause")

# ── Simple IoU tracker (no external deps) ─────────────────────────────────────
tracks      = {}   # track_id -> {"bbox": [], "missing": int}
next_id     = 0
MAX_MISSING = 20   # frames before dropping a track

def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0

import time
frame_idx = 0
t_start   = time.time()

cv2.namedWindow("Face Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face Tracking", 1280, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster processing
    display = cv2.resize(frame, (1280, 720))
    dh, dw  = display.shape[:2]
    sx, sy  = dw / w, dh / h

    # ── Detect ────────────────────────────────────────────────────────────────
    results    = model.predict(display, conf=0.45, iou=0.45, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf            = float(box.conf[0])
            detections.append({"bbox": [x1,y1,x2,y2], "conf": conf})

    # ── Match to existing tracks via IoU ──────────────────────────────────────
    matched_tracks = set()
    matched_dets   = set()

    for tid, trk in tracks.items():
        best_iou, best_d = 0, -1
        for i, det in enumerate(detections):
            if i in matched_dets:
                continue
            s = iou(trk["bbox"], det["bbox"])
            if s > best_iou:
                best_iou, best_d = s, i
        if best_iou > 0.3 and best_d >= 0:
            tracks[tid]["bbox"]    = detections[best_d]["bbox"]
            tracks[tid]["conf"]    = detections[best_d]["conf"]
            tracks[tid]["missing"] = 0
            matched_tracks.add(tid)
            matched_dets.add(best_d)

    # ── New tracks for unmatched detections ───────────────────────────────────
    for i, det in enumerate(detections):
        if i not in matched_dets:
            tracks[next_id] = {
                "bbox"   : det["bbox"],
                "conf"   : det["conf"],
                "missing": 0,
            }
            next_id += 1

    # ── Age unmatched tracks, remove dead ones ────────────────────────────────
    dead = []
    for tid in tracks:
        if tid not in matched_tracks:
            tracks[tid]["missing"] += 1
            if tracks[tid]["missing"] > MAX_MISSING:
                dead.append(tid)
    for tid in dead:
        del tracks[tid]

    # ── Draw ──────────────────────────────────────────────────────────────────
    for tid, trk in tracks.items():
        if trk["missing"] > 0:
            continue                    # only draw active tracks
        x1, y1, x2, y2 = trk["bbox"]
        color           = get_color(tid)
        conf            = trk["conf"]

        # Bounding box
        cv2.rectangle(display, (x1,y1), (x2,y2), color, 2)

        # Label background + text
        label = f"Face {tid}  {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ly = max(y1, th + 8)
        cv2.rectangle(display, (x1, ly-th-8), (x1+tw+4, ly+2), color, cv2.FILLED)
        cv2.putText(display, label, (x1+2, ly-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)

    # ── HUD ───────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    cur_fps = frame_idx / elapsed if elapsed > 0 else 0
    hud = [
        f"Frame : {frame_idx}",
        f"FPS   : {cur_fps:.1f}",
        f"Faces : {len([t for t in tracks.values() if t['missing']==0])}",
        f"ESC=quit  SPACE=pause",
    ]
    cv2.rectangle(display, (0,0), (240, 115), (0,0,0), cv2.FILLED)
    for i, line in enumerate(hud):
        cv2.putText(display, line, (8, 25 + i*26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0,255,180), 2, cv2.LINE_AA)

    cv2.imshow("Face Tracking", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:       # ESC
        print("Stopped by user.")
        break
    elif key == 32:     # SPACE — pause
        print("Paused. Press any key to resume...")
        cv2.waitKey(0)

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
print(f"Done. Processed {frame_idx} frames.")