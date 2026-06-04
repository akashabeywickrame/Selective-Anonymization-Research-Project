# =============================================================================
# utils/make_playable.py — Convert pipeline outputs to H.264 (playable on Windows)
# =============================================================================
#
# Why this exists:
#   This machine's OpenCV build cannot encode H.264 (the OpenH264 DLL is
#   missing), so cv2.VideoWriter can only produce `mp4v` files — which the
#   default Windows players ("Movies & TV" / Media Player) refuse to play.
#
#   This script uses a self-contained ffmpeg (installed via `pip install
#   imageio-ffmpeg`, no system install) to:
#     1. Rebuild the side-by-side comparison from the already-rendered
#        selective + blanket videos (NO re-detection — fast).
#     2. Transcode every output .mp4 to H.264 + faststart so it plays anywhere.
#
# Usage:
#   python -m utils.make_playable                     # process all of output/
#   python -m utils.make_playable output/interaction  # one folder
# =============================================================================

import sys
import subprocess
from pathlib import Path

import cv2
import numpy as np
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


# =============================================================================
# Rebuild a playable comparison directly to H.264 via an ffmpeg pipe
# =============================================================================

def _banner(img, text, color):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 32), (0, 0, 0), cv2.FILLED)
    cv2.putText(out, text, (10, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return out


def rebuild_comparison(blanket_path: Path, selective_path: Path, out_path: Path):
    """Read blanket + selective frame-by-frame, stack them, pipe to H.264."""
    cb = cv2.VideoCapture(str(blanket_path))
    cs = cv2.VideoCapture(str(selective_path))
    if not (cb.isOpened() and cs.isOpened()):
        print(f"  [skip compare] could not open source videos")
        return

    fps = cb.get(cv2.CAP_PROP_FPS) or 30
    w   = int(cb.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cb.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = w * 2   # exactly two panes — even width, no fragile divider

    proc = subprocess.Popen(
        [FFMPEG, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{out_w}x{h}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         str(out_path)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    while True:
        rb, fb = cb.read()
        rs, fs = cs.read()
        if not (rb and rs):
            break
        left  = _banner(fb, "BASELINE: Blanket Blur", (0, 165, 255))
        right = _banner(fs, "PROPOSED: Selective",    (0, 255, 0))
        combo = cv2.hconcat([left, right])
        # 2px white divider drawn ON the seam (keeps width even)
        combo[:, w-1:w+1] = 255
        proc.stdin.write(combo.tobytes())

    cb.release(); cs.release()
    proc.stdin.close(); proc.wait()
    print(f"  [compare] rebuilt -> {out_path.name}")


# =============================================================================
# Transcode a single mp4v file to H.264 in place (writes *_h264.mp4)
# =============================================================================

def transcode_h264(src: Path) -> None:
    dst = src.with_name(src.stem + "_h264.mp4")
    proc = subprocess.run(
        [FFMPEG, "-y", "-i", str(src),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         str(dst)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if proc.returncode == 0:
        print(f"  [h264] {src.name} -> {dst.name}")
    else:
        print(f"  [h264] FAILED on {src.name}")


# =============================================================================
# Main
# =============================================================================

def process_folder(folder: Path):
    print(f"\n[make_playable] {folder}")

    # group files by video stem so we can rebuild each comparison
    stems = sorted({
        p.stem.rsplit("_", 1)[0]
        for p in folder.glob("*.mp4")
        if not p.stem.endswith("_h264")
    })

    for stem in stems:
        blanket   = folder / f"{stem}_blanket.mp4"
        selective = folder / f"{stem}_selective.mp4"
        compare   = folder / f"{stem}_compare_h264.mp4"
        if blanket.exists() and selective.exists():
            rebuild_comparison(blanket, selective, compare)

    # transcode the single-pane outputs (debug / selective / blanket)
    for p in folder.glob("*.mp4"):
        if p.stem.endswith("_h264") or "_compare" in p.stem:
            continue
        transcode_h264(p)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output")
    if not root.exists():
        print(f"[make_playable] not found: {root}")
        return

    if any(root.glob("*.mp4")):
        process_folder(root)
    else:
        for sub in sorted(root.iterdir()):
            if sub.is_dir() and any(sub.glob("*.mp4")):
                process_folder(sub)

    print("\n[make_playable] Done. Open the *_h264.mp4 files.")


if __name__ == "__main__":
    main()
