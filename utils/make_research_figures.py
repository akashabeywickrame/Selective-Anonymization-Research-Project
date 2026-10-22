# =============================================================================
# utils/make_research_figures.py — Generate all research output figures
# =============================================================================
# Produces:
#   output/figures/fig1_research_gap.png   — best frame showing blanket vs selective
#   output/figures/fig2_metrics_chart.png  — PSR/IPR bar chart (baseline vs proposed)
#   output/figures/fig3_pipeline_demo.png  — grid of 4 output types on one frame
# =============================================================================

import cv2
import numpy as np
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIGURES_DIR = Path(__file__).resolve().parent.parent / "output" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR  = Path(__file__).resolve().parent.parent / "output"

# ── which interaction compare video to scan ───────────────────────────────────
COMPARE_CANDIDATES = [
    OUTPUT_DIR / "interaction" / "video1_compare.mp4",
    OUTPUT_DIR / "interaction" / "video13_compare.mp4",
    OUTPUT_DIR / "interaction" / "video10_compare.mp4",
]


# =============================================================================
# 1. Best "research gap" frame
# =============================================================================

def _score_frame(frame):
    """
    Score how well this compare frame demonstrates the research gap.
    High score = left side (blanket) has a blurred central region
                 while right side (selective) has a sharp, large face there.

    Both halves sit side by side with a 2-px divider in the middle.
    """
    h, w = frame.shape[:2]
    mid  = w // 2

    left  = frame[:, :mid - 1]       # blanket blur half
    right = frame[:, mid + 1:]       # selective half

    cw = left.shape[1]

    # Central strip: middle 40% of width, middle 60% of height
    cx0 = int(cw * 0.30);  cx1 = int(cw * 0.70)
    cy0 = int(h  * 0.20);  cy1 = int(h  * 0.80)

    left_strip  = left [cy0:cy1, cx0:cx1].astype(np.float32)
    right_strip = right[cy0:cy1, cx0:cx1].astype(np.float32)

    # Variance measures sharpness: blurred region → low var, sharp face → high var
    left_var  = float(np.var(left_strip))
    right_var = float(np.var(right_strip))

    # We want: right sharp (high var) and left blurred (low var)
    gap = right_var - left_var
    return gap, left_var, right_var


def find_best_gap_frame(video_path: Path, sample_every: int = 15):
    """Scan the compare video and return (best_frame, best_frame_idx)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [!] Cannot open {video_path.name}")
        return None, -1

    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    best_score = -1e9
    best_frame = None
    best_idx   = 0
    frame_idx  = 0

    print(f"  Scanning {video_path.name} ({total} frames, every {sample_every})...")

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        score, lv, rv = _score_frame(frame)
        if score > best_score and rv > 200:
            best_score = score
            best_frame = frame.copy()
            best_idx   = frame_idx

        frame_idx += sample_every
        if frame_idx >= total:
            break

    cap.release()
    print(f"  Best gap score: {best_score:.1f}  (frame {best_idx})")
    return best_frame, best_idx


def make_fig1():
    print("\n[Fig 1] Research gap — best comparison frame")
    best      = None
    best_idx  = -1
    best_base = None
    for candidate in COMPARE_CANDIDATES:
        if candidate.exists():
            frame, idx = find_best_gap_frame(candidate, sample_every=10)
            if frame is not None:
                best      = frame
                best_idx  = idx
                best_base = candidate.parent / candidate.stem.replace("_compare", "")
                break

    if best is None:
        print("  No suitable compare video found.")
        return

    # Add cleaner annotation banner
    h, w = best.shape[:2]
    mid  = w // 2

    annotated = best.copy()

    # Thicker label bars at top
    bar_h = 48
    cv2.rectangle(annotated, (0, 0),       (mid - 1, bar_h), (20, 20, 20),  cv2.FILLED)
    cv2.rectangle(annotated, (mid + 1, 0), (w, bar_h),        (20, 20, 20), cv2.FILLED)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(annotated, "BASELINE: Blanket Blur",
                (12, 33), font, 0.85, (0, 140, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated, "PROPOSED: Selective Anonymization",
                (mid + 12, 33), font, 0.85, (50, 220, 50), 2, cv2.LINE_AA)

    # Bottom annotation
    bot_y = h - 42
    cv2.rectangle(annotated, (0, bot_y), (w, h), (20, 20, 20), cv2.FILLED)
    cv2.putText(annotated,
                "Bystanders blurred in BOTH  |  Interaction partner: BLURRED (left) vs PRESERVED (right)",
                (10, h - 14), font, 0.58, (200, 200, 200), 1, cv2.LINE_AA)

    out = FIGURES_DIR / "fig1_research_gap.png"
    cv2.imwrite(str(out), annotated)
    print(f"  Saved → {out}")


# =============================================================================
# 2. Metrics comparison bar chart  (matplotlib-free, pure OpenCV)
# =============================================================================

def _draw_bar(img, x, y_top, bar_w, bar_h, color, label_pct, label_name, h_total):
    """Draw one bar + labels."""
    y_bot = h_total - 80  # baseline y position
    cv2.rectangle(img, (x, y_top), (x + bar_w, y_bot), color, cv2.FILLED)
    cv2.rectangle(img, (x, y_top), (x + bar_w, y_bot), (60, 60, 60), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Percentage on top of bar
    (tw, th), _ = cv2.getTextSize(label_pct, font, 0.65, 2)
    cv2.putText(img, label_pct,
                (x + bar_w // 2 - tw // 2, y_top - 10),
                font, 0.65, (220, 220, 220), 2, cv2.LINE_AA)

    # Name below baseline
    (tw2, _), _ = cv2.getTextSize(label_name, font, 0.5, 1)
    cv2.putText(img, label_name,
                (x + bar_w // 2 - tw2 // 2, y_bot + 22),
                font, 0.5, (180, 180, 180), 1, cv2.LINE_AA)


def make_fig2():
    print("\n[Fig 2] Metrics comparison bar chart")

    W, H     = 960, 540
    img      = np.full((H, W, 3), 30, dtype=np.uint8)
    font     = cv2.FONT_HERSHEY_SIMPLEX
    baseline = H - 80   # y of the x-axis baseline
    chart_h  = baseline - 80   # usable height for bars

    # Data
    metrics = [
        # name,          psr,   ipr
        ("Blanket\nBlur", 100.0,  0.0),
        ("Phase 2a\n(Proposed)", 87.5, 84.6),
    ]

    # Colour scheme
    PSR_COLOR = (80,  160, 255)  # orange-ish
    IPR_COLOR = (80,  220, 100)  # green-ish

    bar_w     = 90
    group_gap = 160
    pair_gap  = 20
    start_x   = 130

    # Title
    cv2.putText(img, "Privacy vs. Utility: Proposed System vs. Baseline",
                (W // 2 - 310, 38), font, 0.82, (230, 230, 230), 2, cv2.LINE_AA)

    # Y-axis label ticks
    for pct in [0, 25, 50, 75, 100]:
        y = int(baseline - pct / 100 * chart_h)
        cv2.line(img, (80, y), (W - 40, y), (55, 55, 55), 1)
        cv2.putText(img, f"{pct}%", (35, y + 6), font, 0.42, (140, 140, 140), 1)

    # PSR / IPR legend
    cv2.rectangle(img, (W - 220, 55), (W - 200, 73), PSR_COLOR, cv2.FILLED)
    cv2.putText(img, "PSR (Privacy Success Rate)", (W - 195, 70),
                font, 0.42, (200, 200, 200), 1)
    cv2.rectangle(img, (W - 220, 82), (W - 200, 100), IPR_COLOR, cv2.FILLED)
    cv2.putText(img, "IPR (Interaction Preservation)", (W - 195, 97),
                font, 0.42, (200, 200, 200), 1)

    # Draw bars
    for i, (name, psr, ipr) in enumerate(metrics):
        gx = start_x + i * (2 * bar_w + pair_gap + group_gap)

        # PSR bar
        bar_psr_h = int(psr / 100 * chart_h)
        _draw_bar(img,
                  gx, baseline - bar_psr_h,
                  bar_w, bar_psr_h,
                  PSR_COLOR, f"{psr:.0f}%",
                  "", H)

        # IPR bar
        bar_ipr_h = int(ipr / 100 * chart_h)
        ipr_x = gx + bar_w + pair_gap
        _draw_bar(img,
                  ipr_x, baseline - max(bar_ipr_h, 4),
                  bar_w, max(bar_ipr_h, 4),
                  IPR_COLOR, f"{ipr:.0f}%" if ipr > 0 else "0%",
                  "", H)

        # Group label (centred under both bars)
        label_x = gx + bar_w + pair_gap // 2
        lines = name.split("\n")
        for j, line in enumerate(lines):
            (tw, _), _ = cv2.getTextSize(line, font, 0.54, 1)
            cv2.putText(img, line,
                        (label_x - tw // 2, baseline + 42 + j * 20),
                        font, 0.54, (210, 210, 210), 1, cv2.LINE_AA)

    # Axis line
    cv2.line(img, (90, baseline), (W - 40, baseline), (120, 120, 120), 2)

    # Key insight annotation
    cv2.putText(img,
                "Blanket blur: perfect privacy but ZERO utility (partners also blurred)",
                (90, H - 14), font, 0.46, (150, 150, 150), 1, cv2.LINE_AA)

    out = FIGURES_DIR / "fig2_metrics_chart.png"
    cv2.imwrite(str(out), img)
    print(f"  Saved → {out}")


# =============================================================================
# 3. Three-panel demo: same best frame from blanket / selective / debug
# =============================================================================

def make_fig3(best_base=None, best_idx=-1):
    print("\n[Fig 3] Three-panel pipeline demonstration (same frame as Fig 1)")

    # Fall back to searching if not provided
    if best_base is None or best_idx < 0:
        for bp in [OUTPUT_DIR / "interaction" / "video1",
                   OUTPUT_DIR / "interaction" / "video13",
                   OUTPUT_DIR / "interaction" / "video10"]:
            cmp = Path(str(bp) + "_compare.mp4")
            if cmp.exists():
                _, best_idx = find_best_gap_frame(cmp, sample_every=10)
                best_base = bp
                break

    if best_base is None:
        print("  No base found.")
        return

    label_map = {
        "blanket":   ("Baseline: Blanket Blur",      (0,  140, 255)),
        "selective": ("Proposed: Selective Blur",     (50, 220, 50)),
        "debug":     ("Debug: Classification View",  (180, 180, 0)),
    }

    panels    = {}
    for key in ["blanket", "selective", "debug"]:
        vpath = Path(str(best_base) + f"_{key}.mp4")
        if not vpath.exists():
            print(f"  [!] Missing: {vpath.name}")
            continue
        cap = cv2.VideoCapture(str(vpath))
        cap.set(cv2.CAP_PROP_POS_FRAMES, best_idx)
        ret, frame = cap.read()
        cap.release()
        if ret:
            panels[key] = frame

    if len(panels) < 2:
        print("  Not enough panels.")
        return

    h, w  = list(panels.values())[0].shape[:2]
    th    = 50
    row_imgs = []

    for key in ["blanket", "selective", "debug"]:
        if key not in panels:
            continue
        p = panels[key].copy()
        cv2.rectangle(p, (0, 0), (w, th), (15, 15, 15), cv2.FILLED)
        text, col = label_map[key]
        cv2.putText(p, text, (12, th - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, col, 2, cv2.LINE_AA)
        row_imgs.append(p)

    combined = np.hstack(row_imgs)

    out = FIGURES_DIR / "fig3_pipeline_demo.png"
    cv2.imwrite(str(out), combined)
    print(f"  Saved → {out}  ({combined.shape[1]}×{combined.shape[0]})")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Generating research figures...")
    print("=" * 60)
    make_fig1()
    make_fig2()
    # Re-use best_idx from fig1 scan — re-scan quickly
    best_base = None
    best_idx  = -1
    for candidate in COMPARE_CANDIDATES:
        if candidate.exists():
            _, best_idx = find_best_gap_frame(candidate, sample_every=10)
            best_base = candidate.parent / candidate.stem.replace("_compare", "")
            break
    make_fig3(best_base=best_base, best_idx=best_idx)
    print("\nAll done. Figures saved to output/figures/")
