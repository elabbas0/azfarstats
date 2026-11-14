"""Map XVR recording files to wall-clock time.

Files are named XVR_ch<N>_main_<start>_<end>.mp4 (local time, YYYYmmddHHMMSS). The first file of a
recording starts a few seconds *before* its nominal start (key-frame pre-roll) but ends exactly at
its nominal end, and continuation files start exactly where the previous one ended. So we anchor a
file at its start when it continues a previous slice (or begins on a clean hour) and at its end
otherwise, and count frames at the camera's true rate (verified against the burnt-in clock).
"""
from __future__ import annotations
import glob, os, re
from dataclasses import dataclass
from datetime import datetime
import cv2

FPS = 15.0
GRID_HZ = 5.0  # analysis rate
_RE = re.compile(r"XVR_(ch\d+)_main_(\d{14})_(\d{14})\.mp4$")


@dataclass
class Clip:
    path: str
    ch: str
    start: datetime
    end: datetime
    n_frames: int
    t0: float  # wall-clock time of frame 0, seconds since local midnight

    def frame_time(self, i):
        return self.t0 + i / FPS

    def grid_frames(self):
        """[(grid index k, frame index)] for every analysis tick covered by this clip."""
        import math
        k0 = math.ceil(self.t0 * GRID_HZ)
        out = []
        k = k0
        while True:
            i = round((k / GRID_HZ - self.t0) * FPS)
            if i >= self.n_frames:
                break
            if i >= 0:
                out.append((k, i))
            k += 1
        return out


def _secs(dt):
    return dt.hour * 3600 + dt.minute * 60 + dt.second


def scan(folder):
    clips = []
    for p in sorted(glob.glob(os.path.join(folder, "*.mp4"))):
        m = _RE.search(os.path.basename(p))
        if not m:
            continue
        ch, s, e = m.groups()
        s, e = datetime.strptime(s, "%Y%m%d%H%M%S"), datetime.strptime(e, "%Y%m%d%H%M%S")
        cap = cv2.VideoCapture(p)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        clips.append(Clip(p, ch, s, e, n, 0.0))
    ends = {(c.ch, c.end) for c in clips}
    for c in clips:
        continues = (c.ch, c.start) in ends or (c.start.minute == 0 and c.start.second == 0)
        c.t0 = _secs(c.start) if continues else _secs(c.end) - c.n_frames / FPS
    return sorted(clips, key=lambda c: (c.ch, c.t0))


def fmt(t):
    t = int(t)
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


if __name__ == "__main__":
    import sys
    for c in scan(sys.argv[1] if len(sys.argv) > 1 else "data/raw"):
        print(c.ch, os.path.basename(c.path), "frames", c.n_frames, "->", fmt(c.t0), "to", fmt(c.frame_time(c.n_frames - 1)))
