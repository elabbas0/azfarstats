"""Read the camera's burnt-in clock: the exact wall-clock time of every frame.

The recorder occasionally drops frames, so "frame number / 15" drifts from real time by up to a
couple of seconds within an hour. The burnt-in clock does not drift. We decode only the clock area,
find the frame where the seconds digit changes, and give every frame its clock time (interpolated
within the second). Anchor: a file that continues the previous one starts exactly on its name's
start time; the first file of a recording ends one second before its name's end time.

Usage: python -m azfar.osd --raw data/raw --out data/osd
Writes <clip>.npy: float64 clock time (seconds since midnight) for every frame.
"""
from __future__ import annotations
import argparse, os, subprocess
import numpy as np
from .timeline import scan, FPS

X0, Y0, W, H = 1250, 30, 630, 70     # clock area (full resolution)
SEC = slice(520, 630)                 # the two seconds digits inside it
THR = 150


def _secs(dt):
    return dt.hour * 3600 + dt.minute * 60 + dt.second


ONES = slice(575, 630)                # the last digit of the seconds


def change_frames(path, n):
    """Frames where the seconds change, and a picture of the last digit for every second."""
    cmd = ["ffmpeg", "-v", "error", "-threads", "2", "-i", path, "-fps_mode", "passthrough", "-vf",
           f"crop={W}:{H}:{X0}:{Y0},format=gray", "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    prev, i, diffs, ones = None, 0, [], []
    while True:
        buf = p.stdout.read(W * H)
        if len(buf) < W * H:
            break
        img = np.frombuffer(buf, np.uint8).reshape(H, W)
        a = img[:, SEC].astype(np.int16)
        diffs.append(int((np.abs(a - prev) > 70).sum()) if prev is not None else 0)
        ones.append(img[:, ONES].copy())
        prev, i = a, i + 1
    p.wait()
    d = np.array(diffs)
    c = np.flatnonzero(d > THR)
    keep = [0]
    for x in c:
        if x - keep[-1] > 3:   # dropped frames can make a second last only a few frames
            keep.append(int(x))
    starts = np.array(keep)
    ends = np.r_[starts[1:], i]
    # one picture per second: the frame in the middle of it
    pics = np.stack([ones[(s_ + e_ - 1) // 2] for s_, e_ in zip(starts, ends)]).astype(np.float32)
    return starts, pics, i


def read_digits(pics, anchor_digit, anchor_at_end):
    """Classify every second's last digit. Templates are learnt next to the anchor, where the digits
    are known to count up one by one."""
    n = len(pics)
    z = pics.reshape(n, -1)
    z = (z - z.mean(1, keepdims=True)) / (z.std(1, keepdims=True) + 1e-6)
    idx = np.arange(n - min(n, 200), n) if anchor_at_end else np.arange(min(n, 200))
    lab = (anchor_digit - (n - 1 - idx)) % 10 if anchor_at_end else (anchor_digit + idx) % 10
    T = np.stack([z[idx[lab == d]].mean(0) for d in range(10)])
    T = (T - T.mean(1, keepdims=True)) / (T.std(1, keepdims=True) + 1e-6)
    return np.argmax(z @ T.T, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/osd")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    clips = scan(a.raw)
    ends = {(c.ch, c.end) for c in clips}
    for c in clips:
        out = os.path.join(a.out, os.path.basename(c.path).replace(".mp4", ".npy"))
        if os.path.exists(out):
            continue
        starts, pics, n = change_frames(c.path, c.n_frames)
        continues = (c.ch, c.start) in ends or (c.start.minute == 0 and c.start.second == 0)
        if continues:
            anchor = _secs(c.start)
            dig = read_digits(pics, anchor % 10, anchor_at_end=False)
        else:
            # the first file of a recording ends exactly at its name's end time; its very last frames
            # may already show that second, otherwise the last complete second is one before it
            anchor = _secs(c.end) - (0 if n - starts[-1] < 8 else 1)
            dig = read_digits(pics, anchor % 10, anchor_at_end=True)
        # a single misread digit shows up as neighbours that are exactly two apart: repair it
        for _ in range(2):
            mid = np.r_[False, ((dig[2:] - dig[:-2]) % 10 == 2) & ((dig[1:-1] - dig[:-2]) % 10 != 1), False]
            dig = np.where(mid, (np.roll(dig, 1) + 1) % 10, dig)
        step = np.r_[0, (np.diff(dig) % 10)]
        step = np.where(step == 0, 1, step)       # a misread digit: assume the usual one second
        sec = np.cumsum(step)
        sec = sec - sec[0] + anchor if continues else sec - sec[-1] + anchor
        skips = int((step > 1).sum())
        # every frame: its second, plus the fraction of the way through it
        j = np.searchsorted(starts, np.arange(n), side="right") - 1
        e_ = np.r_[starts[1:], n]
        frac = (np.arange(n) - starts[j]) / np.maximum(e_[j] - starts[j], 1)
        t = sec[j] + np.clip(frac, 0, 0.999)
        cf = starts[1:]
        np.save(out, t)
        span = _secs(c.end) - _secs(c.start)
        drift = t - c.frame_time(np.arange(n))
        print(f"{os.path.basename(c.path)}: {len(cf)} second changes over {n} frames, {skips} skipped seconds (name span {span} s); "
              f"clock {t[0]:.1f}..{t[-1]:.1f}; frame-count timeline is off by {drift.min():+.2f}..{drift.max():+.2f} s", flush=True)


if __name__ == "__main__":
    main()
