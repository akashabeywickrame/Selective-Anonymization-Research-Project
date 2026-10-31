# =============================================================================
# utils/make_thesis_figures.py — Data figures for the revised thesis
# =============================================================================
# Writes vector PDFs into thesis_revised/figures/. Every number comes from the
# result files (logs/, eval/) or from the tables already in the thesis.
#
# Usage:  python utils/make_thesis_figures.py
# =============================================================================

import csv
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "thesis_revised" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT))

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddcd6"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "pdf.fonttype": 42,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=160)
    plt.close(fig)
    print("[Figures]", name)


# ── 1. Annotator agreement in the two blind studies ───────────────────────────
def fig_agreement():
    b = [float(r["kappa"]) for r in csv.DictReader(open(ROOT / "eval/annotation_package/agreement_report.csv"))]
    c = [float(r["kappa"]) for r in csv.DictReader(open(ROOT / "eval/clip_package/clip_agreement_report.csv"))]
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    bands = [(0.21, 0.40, "fair"), (0.41, 0.60, "moderate"), (0.61, 0.80, "substantial"),
             (0.81, 1.00, "almost perfect")]
    for i, (lo, hi, name) in enumerate(bands):
        ax.axvspan(lo, hi, color="#f2f1ed" if i % 2 else "#e9e8e2", lw=0)
        ax.text((lo + hi) / 2, 1.55, name, ha="center", va="bottom", fontsize=8, color=INK2)
    ax.scatter(b, [1] * len(b), s=46, color=BLUE, edgecolor="white", lw=1.2, zorder=3)
    ax.scatter(c, [0] * len(c), s=46, color=ORANGE, edgecolor="white", lw=1.2, zorder=3)
    ax.set_yticks([0, 1], ["Video clips (Set C),\n5 annotators", "Still frames (Set B),\n4 annotators"])
    ax.set_ylim(-0.6, 1.9)
    ax.set_xlim(0.21, 1.0)
    ax.set_xlabel("Cohen's $\\kappa$ for each pair of annotators")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_agreement")


# ── 2. Face size of kept partners, missed partners and bystanders ─────────────
def fig_face_size():
    per = json.load(open(ROOT / "logs/clip_face_facts.json"))
    groups = [("Partners kept visible", "TP", BLUE), ("Partners blurred", "FN", ORANGE),
              ("Bystanders", None, AQUA)]
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    for y, (name, g, col) in enumerate(groups):
        vals = [p["area"] for p in per if (p["group"] == g if g else p["group"] in ("TN", "FP"))]
        jit = [y + ((i * 37) % 11 - 5) * 0.03 for i in range(len(vals))]
        ax.scatter(vals, jit, s=22, color=col, alpha=0.85, edgecolor="white", lw=0.6, zorder=3)
        med = st.median(vals)
        ax.plot([med, med], [y - 0.28, y + 0.28], color=INK, lw=2, zorder=4)
        ax.text(med, y + 0.33, f"median {med:.2f}%", ha="center", fontsize=8, color=INK)
    ax.axvline(1.2, color=INK2, ls="--", lw=1)
    ax.text(1.25, 2.55, "area threshold 1.2%", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.3, 1, 3, 10], ["0.1%", "0.3%", "1%", "3%", "10%"])
    ax.set_yticks(range(3), [g[0] for g in groups])
    ax.set_ylim(-0.6, 2.8)
    ax.set_xlabel("Median share of the frame covered by the face during the clip (log scale)")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_face_size")


# ── 3. Privacy-utility trade-off: area threshold sweep and configurations ─────
def fig_tradeoff():
    rows = list(csv.DictReader(open(ROOT / "logs/replay_ablation.csv")))
    sw = [r for r in rows if r["experiment"] == "area_sweep"]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = [float(r["ipr"]) for r in sw]
    y = [float(r["psr"]) for r in sw]
    ax.plot(x, y, color=BLUE, lw=2, marker="o", ms=6, mec="white", mew=1, zorder=3,
            label="Live system, varying the area threshold")
    offs = {2.0: (0, 7), 1.5: (-10, -13), 1.2: (6, -13), 1.0: (0, 7),
            0.8: (14, -3), 0.6: (14, -3), 0.4: (14, -3)}
    for r, xi, yi in zip(sw, x, y):
        t = round(float(r["config"]) * 100, 1)
        ax.annotate(f"{t:.1f}%", (xi, yi), textcoords="offset points", xytext=offs[t],
                    ha="center", fontsize=7.5, color=INK2)
    ax.scatter([85.51], [90.91], s=70, marker="D", color=ORANGE, edgecolor="white", lw=1,
               zorder=4, label="Offline second pass (whole video, track joining)")
    ax.annotate("second pass", (85.51, 90.91), textcoords="offset points", xytext=(8, 4),
                fontsize=8, color=INK)
    ax.scatter([0], [100], s=70, marker="s", color=INK2, zorder=4, label="Blanket blurring")
    ax.annotate("blanket blurring", (0, 100), textcoords="offset points", xytext=(8, -4),
                fontsize=8, color=INK)
    ax.annotate("threshold used (1.2%)", (56.52, 96.97), textcoords="offset points",
                xytext=(-20, -34), fontsize=8, color=INK, ha="right",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xlabel("IPR: partners kept visible (%)")
    ax.set_ylabel("PSR: bystanders blurred (%)")
    ax.set_xlim(-4, 100)
    ax.set_ylim(74, 102)
    ax.legend(loc="lower left", fontsize=8)
    save(fig, "fig_tradeoff")


# ── 4. Time per pipeline stage ────────────────────────────────────────────────
def fig_latency():
    stages = [("Frame reading", 8.7), ("Face detection", 46.9), ("Tracking", 0.27),
              ("Landmarks (Phase 2b)", 14.7), ("Classification and blurring", 0.9)]
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    names = [s[0] for s in stages][::-1]
    vals = [s[1] for s in stages][::-1]
    ax.barh(names, vals, color=BLUE, height=0.55)
    for i, v in enumerate(vals):
        ax.text(v + 0.8, i, f"{v:g} ms", va="center", fontsize=8, color=INK)
    ax.set_xlabel("Mean time per frame on the laptop CPU (ms)")
    ax.set_xlim(0, 56)
    ax.grid(axis="y", visible=False)
    save(fig, "fig_latency")


# ── 5. Measured and projected throughput ──────────────────────────────────────
def fig_throughput():
    rows = [("Laptop, PyTorch (measured)", 17.6, 14.0),
            ("Laptop, OpenVINO (measured)", 31.1, 22.4),
            ("Laptop, OpenVINO, every 2nd frame (measured)", 49.3, 36.6),
            ("Raspberry Pi 5, PyTorch", 1.8, 1.7),
            ("Raspberry Pi 5, NCNN", 7.8, 5.6),
            ("Raspberry Pi 5, NCNN, every 2nd frame", 12.3, 9.4),
            ("Ray-Ban Meta / Display, CPU only", 6.1, 4.4),
            ("Ray-Ban Meta / Display, NPU (upper bound)", 18.1, 8.4),
            ("Ray-Ban Meta / Display, NPU, every 2nd frame", 20.4, 12.5),
            ("Snap Spectacles 5 (upper end of range)", 18.1, 8.4)][::-1]
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    h = 0.36
    for i, (name, a, b) in enumerate(rows):
        ax.barh(i + h / 2 + 0.01, a, height=h, color=BLUE, label="Phase 2a" if i == 0 else None)
        ax.barh(i - h / 2 - 0.01, b, height=h, color=ORANGE, label="Phase 2b" if i == 0 else None)
        ax.text(a + 0.4, i + h / 2, f"{a:.1f}", va="center", fontsize=7.5, color=INK)
        ax.text(b + 0.4, i - h / 2, f"{b:.1f}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel("Frames per second")
    ax.axvline(15, color=INK2, ls="--", lw=1)
    ax.text(15.4, len(rows) - 0.55, "15 FPS", fontsize=8, color=INK2)
    ax.set_xlim(0, 56)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "fig_throughput")


def facts():
    """Per-clip face size and group for Set C, cached for fig_face_size."""
    import replay_classifier as R
    clips = {r["clip_id"]: r for r in csv.DictReader(open(ROOT / "eval/clip_package/clip_map.csv"))}
    gt = {r["clip_id"]: int(r["ground_truth"])
          for r in csv.DictReader(open(ROOT / "eval/clip_package/clip_ground_truth.csv"))}
    log = R.load_log({(r["category"], r["video"]) for r in clips.values()})
    per = []
    for cid, g in gt.items():
        r = clips[cid]
        rows = log[(r["category"], r["video"])]
        s, e = int(r["start_frame"]), int(r["end_frame"])
        win = [x["area"] for fi in range(s, e + 1) for x in rows.get(fi, []) if x["tid"] == r["track_id"]]
        pred = r["system_prediction"] == "1"
        per.append({"clip": cid, "area": st.median(win) * 100,
                    "group": "TP" if g and pred else "FN" if g else "FP" if pred else "TN"})
    json.dump(per, open(ROOT / "logs/clip_face_facts.json", "w"))


if __name__ == "__main__":
    if not (ROOT / "logs/clip_face_facts.json").exists():
        facts()
    fig_agreement()
    fig_face_size()
    fig_tradeoff()
    fig_latency()
    fig_throughput()
