# =============================================================================
# utils/video_loader.py — VideoCapture wrapper with resize + metadata
# =============================================================================

import cv2
from pathlib import Path
from config import FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS


class VideoLoader:
    """
    Thin wrapper around cv2.VideoCapture.
    Handles open / read / release and optional frame resize.
    """

    def __init__(self, video_path: str | Path):
        self.video_path = Path(video_path)

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        # ── source metadata ────────────────────────────────────────────────────
        self.src_width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.src_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.src_fps    = self.cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # ── output dimensions after resize ─────────────────────────────────────
        self.out_width  = FRAME_WIDTH  if FRAME_WIDTH  > 0 else self.src_width
        self.out_height = FRAME_HEIGHT if FRAME_HEIGHT > 0 else self.src_height

        self._do_resize = (
            self.out_width  != self.src_width or
            self.out_height != self.src_height
        )

        print(
            f"[VideoLoader] {self.video_path.name} | "
            f"{self.src_width}x{self.src_height} → "
            f"{self.out_width}x{self.out_height} | "
            f"{self.src_fps:.1f} FPS | {self.total_frames} frames"
        )

    # ── iterator interface ─────────────────────────────────────────────────────
    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration
        if self._do_resize:
            frame = cv2.resize(frame, (self.out_width, self.out_height))
        return frame

    def __len__(self):
        return self.total_frames

    # ── context manager ────────────────────────────────────────────────────────
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

    # ── helpers ────────────────────────────────────────────────────────────────
    @property
    def frame_size(self) -> tuple[int, int]:
        """Returns (width, height) of output frames."""
        return (self.out_width, self.out_height)

    @property
    def fps(self) -> float:
        return self.src_fps