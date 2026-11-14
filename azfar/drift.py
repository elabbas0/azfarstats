"""Track small changes in camera aim over a recording.

A corner camera can sag or get nudged, which shifts every projected position. For each camera and
each time window we build a player-free median image, re-locate the painted lines near where the
reference calibration expects them, and re-fit the camera rotation. The result is stored in the
calibration file under "aim": {ch: [{t0, t1, rvec, shift_px, n}]} and used by the fusion stage.

Usage: python -m azfar.drift --raw data/raw --calib config/calibration.json [--window 600]
"""
from __future__ import annotations
import argparse, json
import numpy as np, cv2
from scipy.optimize import least_squares
from .camera import Camera
from .calibrate import snap, line_def
from .timeline import scan, FPS, fmt


def local_dims(calib, ch):
    P = calib["pitch"]
    L, W = P["L"], P["W"]
    if calib["cameras"][ch].get("flip"):
        b = P["boxes"][1]
        return L, W, L - b["x1"], W - b["y1"], W - b["y0"]
    b = P["boxes"][0]
    return L, W, b["x1"], b["y0"], b["y1"]


def model_segments(L, W, d, b1, b2, reach=26.0):
    """Line pieces in the camera's own half, where the paint is clearly resolved."""
    r = min(reach, L)
    return {"T0": ((1.5, 0), (r, 0)), "G0": ((0, 1.5), (0, W)), "T1": ((0, W), (r * 0.7, W)),
            "BS1": ((0, b1), (d, b1)), "BS2": ((0, b2), (d, b2)), "BF": ((d, b1), (d, b2))}


def median_frame(clip, i0, i1, n=25):
    cap = cv2.VideoCapture(clip.path)
    frames = []
    for i in np.linspace(i0, i1 - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, im = cap.read()
        if ok:
            frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    cap.release()
    return np.median(np.stack(frames), 0).astype(np.uint8) if len(frames) >= 5 else None


def refit(cam0: Camera, dims, gray, rounds=3):
    """Re-fit rotation only. Returns (rvec, mean pixel shift of the lines, n points)."""
    L, W, d, b1, b2 = dims
    cam = Camera(cam0.f, cam0.k1, cam0.k2, cam0.cx, cam0.cy, cam0.rvec, cam0.C)
    segs = model_segments(L, W, d, b1, b2)
    n_pts = 0
    for it in range(rounds):
        names, pts = [], []
        for name, (a, b) in segs.items():
            t = np.linspace(0, 1, 40)[:, None]
            xy = np.asarray(a, float) * (1 - t) + np.asarray(b, float) * t
            px = cam.project(np.c_[xy, np.zeros(len(xy))])
            inb = (px[:, 0] > 20) & (px[:, 0] < 1900) & (px[:, 1] > 120) & (px[:, 1] < 1060)
            if inb.sum() < 4:
                continue
            q, ok = snap(gray, px[inb], radius=14 if it == 0 else 8, step=8, with_flags=True)
            if ok.sum() >= 4:
                names.append(name); pts.append(q[ok])
        n_pts = sum(len(p) for p in pts)
        if n_pts < 60:
            return None

        def fun(r):
            c = Camera(cam0.f, cam0.k1, cam0.k2, cam0.cx, cam0.cy, r, cam0.C)
            res = []
            for name, px in zip(names, pts):
                A, u = line_def(name, L, W, d, b1, b2)
                nrm = np.cross(np.asarray(u, float), np.asarray(A, float) - c.C)
                nrm /= np.linalg.norm(nrm)
                res.append(c.f * np.arcsin(np.clip(c.rays(px) @ nrm, -1, 1)))
            return np.concatenate(res)
        r = least_squares(fun, cam.rvec, loss="soft_l1", f_scale=2.0, max_nfev=60)
        cam = Camera(cam0.f, cam0.k1, cam0.k2, cam0.cx, cam0.cy, r.x, cam0.C)
    probe = np.array([[8, 5, 0], [15, 10, 0], [5, 15, 0], [20, 3, 0]], float)
    shift = float(np.linalg.norm(cam.project(probe) - Camera(cam0.f, cam0.k1, cam0.k2, cam0.cx, cam0.cy, cam0.rvec, cam0.C).project(probe), axis=1).mean())
    return cam.rvec.tolist(), shift, n_pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--calib", default="config/calibration.json")
    ap.add_argument("--window", type=float, default=600, help="seconds per aim estimate")
    a = ap.parse_args()
    calib = json.load(open(a.calib))
    aim = {}
    for clip in scan(a.raw):
        cam0 = Camera.from_dict({**calib["cameras"][clip.ch], "flip": None})
        dims = local_dims(calib, clip.ch)
        step = int(a.window * FPS)
        for i0 in range(0, clip.n_frames, step):
            i1 = min(i0 + step, clip.n_frames)
            if i1 - i0 < 60 * FPS and i0 > 0:
                continue
            g = median_frame(clip, i0, i1)
            res = refit(cam0, dims, g) if g is not None else None
            t0, t1 = clip.frame_time(i0), clip.frame_time(i1)
            if res is None:
                print(clip.ch, fmt(t0), "no fit"); continue
            aim.setdefault(clip.ch, []).append(dict(t0=t0, t1=t1, rvec=res[0], shift_px=round(res[1], 2), n=res[2]))
            print(clip.ch, fmt(t0), "-", fmt(t1), f"aim shift {res[1]:.1f}px from {res[2]} line points", flush=True)
    calib["aim"] = aim
    json.dump(calib, open(a.calib, "w"), indent=1)


if __name__ == "__main__":
    main()
