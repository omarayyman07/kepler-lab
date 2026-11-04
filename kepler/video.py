"""Video helpers.

Two things matter for measurement:

1. SAM 3.1 indexes frames by decode order, so frame N must map to the exact
   presentation time of the Nth decoded frame. Phones record variable frame
   rate, so we never assume N / fps; we read every frame's timestamp.
2. Muse Spark video understanding takes MP4 only, and SAM takes H.264/H.265
   MP4/MOV. We normalise everything to H.264 MP4 with faststart and keep the
   original timing (no frame dropping or duplication).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _need(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise RuntimeError(f"{tool} not found. Install ffmpeg (it ships ffprobe).")
    return path


@dataclass(frozen=True)
class ClipInfo:
    path: Path
    width: int
    height: int
    n_frames: int
    timestamps: np.ndarray  # seconds, one per decoded frame, starts at 0

    @property
    def fps(self) -> float:
        if self.n_frames < 2:
            return float("nan")
        return (self.n_frames - 1) / float(self.timestamps[-1] - self.timestamps[0])


def prepare_clip(
    src: str | Path,
    dst: str | Path,
    *,
    max_width: int = 1280,
    start: float | None = None,
    duration: float | None = None,
) -> Path:
    """Re-encode to H.264 MP4 (faststart), keep every frame and its timing."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [_need("ffmpeg"), "-y", "-loglevel", "error"]
    if start is not None:
        cmd += ["-ss", f"{start}"]
    cmd += ["-i", str(src)]
    if duration is not None:
        cmd += ["-t", f"{duration}"]
    cmd += [
        "-an",
        "-vf", f"scale='min({max_width},iw)':-2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-fps_mode", "passthrough",
        "-movflags", "+faststart",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    return dst


def probe(path: str | Path) -> ClipInfo:
    """Read size and the presentation timestamp of every frame."""
    path = Path(path)
    out = subprocess.run(
        [
            _need("ffprobe"), "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height:frame=best_effort_timestamp_time",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(out)
    stream = data["streams"][0]
    ts = np.array(
        [float(f["best_effort_timestamp_time"]) for f in data.get("frames", [])
         if f.get("best_effort_timestamp_time") not in (None, "N/A")],
        dtype=float,
    )
    if ts.size == 0:
        raise RuntimeError(f"No frame timestamps in {path}")
    ts = ts - ts[0]
    return ClipInfo(path, int(stream["width"]), int(stream["height"]), ts.size, ts)


def read_frame(path: str | Path, index: int) -> np.ndarray:
    """Return one BGR frame (used for calibration stills)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {index} of {path}")
    return frame
