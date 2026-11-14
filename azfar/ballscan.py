"""Stage 1c: motion-based ball candidates on EVERY frame (15 per second).

The object detector sees the ball only part of the time. A moving ball is, however, a small bright
blob that differs from the empty-pitch background, so for each frame we take the foreground
(adaptive background model), remove the players (their detection boxes), and keep blobs whose size
fits a ball at that spot on the pitch. These are noisy; azfar.ball picks the one consistent path.

Usage: python -m azfar.ballscan --raw data/raw --det data/det --out data/ballscan [--only <substring>]
Output per clip: f (frame index), xy (full-resolution pixel centre), size (px), bright, area.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np, cv2
from .camera import Camera
from .timeline import scan, FPS, GRID_HZ

S = 0.5          # working scale
TOP = 10         # blobs kept per frame


def expected_size_map(cam, w=960, h=540):
    """Expected ball diameter (working-scale px) if the ball lies on the ground at each pixel; 0 = sky."""
    xs, ys = np.meshgrid(np.arange(0, w, 8) / S, np.arange(0, h, 8) / S)
    px = np.stack([xs, ys], -1).reshape(-1, 2)
    g, ok = cam.to_ground(px, height=0.11)
    loc = cam._to_local(np.c_[g, np.zeros(len(g))])
    r = np.linalg.norm(loc - cam.C, axis=1)
    e = np.where(ok & (r < 70), cam.f * 0.22 / np.maximum(r, 1.0) * S, 0.0).reshape(ys.shape)
    return cv2.resize(e.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def colour(img, cx, cy, r):
    """Median saturation, median brightness and share of bright-yellow pixels in a small disc.
    The match ball is white/cream; boots that fool the detectors are mostly neon yellow."""
    r = max(2, int(r))
    pt = img[max(int(cy - r), 0):int(cy + r) + 1, max(int(cx - r), 0):int(cx + r) + 1]
    if pt.size == 0:
        return 0.0, 0.0, 0.0
    hsv = cv2.cvtColor(pt, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    yel = float(((hsv[:, 0] >= 18) & (hsv[:, 0] <= 45) & (hsv[:, 1] > 110)).mean())
    return float(np.median(hsv[:, 1])), float(np.median(hsv[:, 2])), yel


def blobs(small, fg, exp):
    n, lab, st, cen = cv2.connectedComponentsWithStats(fg, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        m = max(w, h)
        if a < 3 or m > 40 or m / max(min(w, h), 1) > 3.5 or a / (w * h) < 0.4:
            continue
        cx, cy = cen[i]
        e = exp[min(int(cy), exp.shape[0] - 1), min(int(cx), exp.shape[1] - 1)]
        ratio = m / e if e > 0 else 0.0
        if e > 0 and not (0.5 < ratio < 4.5):
            continue
        if e == 0 and not (2 <= m <= 16):
            continue
        patch = small[y:y + h, x:x + w]
        bright = float(patch.max(2).mean())
        # prefer ball-sized, bright blobs
        score = bright / 255.0 - 0.25 * abs(np.log(max(ratio, 0.3) / 1.3))
        sat, val, yel = colour(small, cx, cy, 0.35 * min(w, h) + 0.5)
        out.append((score, cx / S, cy / S, m / S, bright, a, sat, val, yel))
    out.sort(reverse=True)
    return out[:TOP]


def run_clip(clip, cam, det_path, out_path):
    d = np.load(det_path)
    per = (d["cls"] == 0) & (d["conf"] >= 0.3)
    pk, pb = d["k"][per], d["box"][per] * S
    order = np.argsort(pk, kind="stable"); pk, pb = pk[order], pb[order]
    exp = expected_size_map(cam)
    mog = cv2.createBackgroundSubtractorMOG2(history=900, varThreshold=36, detectShadows=False)
    cap = cv2.VideoCapture(clip.path)
    F, XY, SZ, BR, AR, COL = [], [], [], [], [], []
    db = np.flatnonzero(d["cls"] == 32)                       # detector ball hits: colour them too
    det_at = {}
    fm = dict(clip.grid_frames())
    for j in db:
        det_at.setdefault(fm.get(int(d["k"][j]), -1), []).append(j)
    DCOL = np.zeros((len(d["k"]), 3), np.float32)
    i, t0 = 0, time.time()
    while True:
        ok, im = cap.read()
        if not ok:
            break
        small = cv2.resize(im, (960, 540), interpolation=cv2.INTER_AREA)
        for j in det_at.get(i, ()):
            x1, y1, x2, y2 = d["box"][j]
            DCOL[j] = colour(im, (x1 + x2) / 2, (y1 + y2) / 2, 0.35 * min(x2 - x1, y2 - y1))
        fg = mog.apply(small, learningRate=0.004 if i > 300 else 0.03)
        if i >= 45:
            fg = (fg > 0).astype(np.uint8)
            k = int(round(clip.frame_time(i) * GRID_HZ))
            lo, hi = np.searchsorted(pk, [k, k + 1])
            if hi == lo:  # no boxes on this exact tick: use the previous one
                lo, hi = np.searchsorted(pk, [k - 1, k])
            for x1, y1, x2, y2 in pb[lo:hi]:
                gw = (x2 - x1) * 0.12 + 2
                fg[max(int(y1 - 3), 0):int(y2 + 3), max(int(x1 - gw), 0):int(x2 + gw)] = 0
            for s_, x, y, m, br, a, sat, val, yel in blobs(small, fg, exp):
                F.append(i); XY.append((x, y)); SZ.append(m); BR.append(br); AR.append(a); COL.append((sat, val, yel))
        i += 1
        if i % 9000 == 0:
            print(f"{os.path.basename(clip.path)} {i}/{clip.n_frames} {i / (time.time() - t0):.0f} fps", flush=True)
    cap.release()
    np.savez_compressed(out_path, f=np.array(F, np.int32), xy=np.array(XY, np.float32).reshape(-1, 2),
                        size=np.array(SZ, np.float32), bright=np.array(BR, np.float32), area=np.array(AR, np.int32),
                        col=np.array(COL, np.float32).reshape(-1, 3), det_col=DCOL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--det", default="data/det")
    ap.add_argument("--calib", default="config/calibration.json")
    ap.add_argument("--out", default="data/ballscan")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    cv2.setNumThreads(1)
    calib = json.load(open(a.calib))
    os.makedirs(a.out, exist_ok=True)
    for clip in scan(a.raw):
        name = os.path.basename(clip.path).replace(".mp4", ".npz")
        if a.only not in name or not os.path.exists(os.path.join(a.det, name)) or os.path.exists(os.path.join(a.out, name)):
            continue
        cam = Camera.from_dict(calib["cameras"][clip.ch])
        run_clip(clip, cam, os.path.join(a.det, name), os.path.join(a.out, name + ".tmp.npz"))
        os.replace(os.path.join(a.out, name + ".tmp.npz"), os.path.join(a.out, name))
        print("done", name, flush=True)


if __name__ == "__main__":
    main()
