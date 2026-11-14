"""Frame-accurate random access into a clip (OpenCV's own seeking is not exact for HEVC)."""
from __future__ import annotations
import subprocess, functools
import numpy as np, cv2


@functools.lru_cache(maxsize=16)
def _pts(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "packet=pts_time",
                          "-of", "csv=p=0", path], capture_output=True, text=True).stdout
    return np.sort(np.array([float(x.split(",")[0]) for x in out.split() if x.strip() and x[0].isdigit()]))


def grab(path, frame_index):
    """Return frame number `frame_index` (BGR) or None."""
    pts = _pts(path)
    if frame_index >= len(pts):
        return None
    t = max(pts[frame_index] - 0.001, 0)
    out = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.4f}", "-i", path, "-frames:v", "1", "-f", "image2pipe",
                          "-vcodec", "bmp", "-"], capture_output=True).stdout
    return cv2.imdecode(np.frombuffer(out, np.uint8), 1) if out else None
