"""Joint calibration of both corner cameras from pitch-marking annotations.

Fits, with W fixed as the scale gauge: pitch length L, box depth d, box side offset b1,
and per camera: f, k1, k2, principal point, rotation, position.
Usage: python -m azfar.calibrate --bg data/bg --out config/calibration.json [--width 20]
"""
from __future__ import annotations
import argparse, json, itertools
import numpy as np, cv2
from scipy.optimize import least_squares
from .camera import Camera

CAMS = ("ch2", "ch3")


def snap(gray, poly, radius=7, step=6):
    """Sample a seed polyline and snap each sample to the brightest ridge across it."""
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    th = cv2.GaussianBlur(th, (0, 0), 1.2)
    poly = np.asarray(poly, float)
    out = []
    for a, b in zip(poly[:-1], poly[1:]):
        n = max(2, int(np.linalg.norm(b - a) / step))
        t = (b - a) / np.linalg.norm(b - a)
        nrm = np.array([-t[1], t[0]])
        for s in np.linspace(0, 1, n, endpoint=False):
            p = a + s * (b - a)
            offs = np.arange(-radius, radius + 1)
            q = p[None] + offs[:, None] * nrm[None]
            xi = np.clip(q[:, 0].round().astype(int), 0, gray.shape[1] - 1)
            yi = np.clip(q[:, 1].round().astype(int), 0, gray.shape[0] - 1)
            v = th[yi, xi]
            k = int(v.argmax())
            if v[k] > 12 and 0 < k < len(offs) - 1:
                out.append(q[k])
            else:
                out.append(p)  # keep the hand-placed point when no clear ridge
    return np.array(out)


def line_def(name, L, W, d, b1, b2):
    """(point on line, unit direction) on the ground plane, local frame."""
    return {"T0": ((0, 0, 0), (1, 0, 0)), "G0": ((0, 0, 0), (0, 1, 0)),
            "T1": ((0, W, 0), (1, 0, 0)), "GL": ((L, 0, 0), (0, 1, 0)),
            "BS1": ((0, b1, 0), (1, 0, 0)), "BS2": ((0, b2, 0), (1, 0, 0)),
            "BF": ((d, 0, 0), (0, 1, 0))}[name]


NP = 14  # per camera: box depth, box near side, box far side, f, k1, k2, cx, cy, rvec(3), C(3)


def unpack(p, i, W):
    q = p[1 + NP * i: 1 + NP * (i + 1)]
    cam = Camera(q[3], q[4], q[5], q[6], q[7], q[8:11], q[11:14])
    return cam, p[0], q[0], q[1], q[2]


def residuals(p, pts, W):
    """Angular distance (in ~pixels) between each annotated pixel's ray and the plane
    through the camera centre and its pitch line. Smooth everywhere."""
    res = []
    for i, ch in enumerate(CAMS):
        cam, L, d, b1, b2 = unpack(p, i, W)
        for name, px in pts[ch].items():
            A, u = line_def(name, L, W, d, b1, b2)
            n = np.cross(np.asarray(u, float), np.asarray(A, float) - cam.C)
            n /= np.linalg.norm(n)
            res.append(cam.f * np.arcsin(np.clip(cam.rays(px) @ n, -1, 1)))
    return np.concatenate(res)


def init_cam(f, h, yaw, pitch):
    # rotate R0 so the camera looks `yaw` rad towards +Y and `pitch` rad down
    Ry = cv2.Rodrigues(np.array([0., yaw, 0.]))[0]      # about camera y (down): yaw
    Rx = cv2.Rodrigues(np.array([-pitch, 0., 0.]))[0]   # about camera x: pitch down
    rvec = cv2.Rodrigues((Ry @ Rx).T)[0].ravel()
    return [f, 0.0, 0.0, 960, 540, *rvec, -0.5, -0.5, -h]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg", default="data/bg")
    ap.add_argument("--ann", default="config/calib_annotations.json")
    ap.add_argument("--out", default="config/calibration.json")
    ap.add_argument("--width", type=float, default=20.0, help="pitch width in metres (scale gauge)")
    a = ap.parse_args()
    ann = json.load(open(a.ann))
    W = a.width
    pts, imgs = {}, {}
    for ch in CAMS:
        imgs[ch] = cv2.imread(f"{a.bg}/{ch}.png")
        g = cv2.cvtColor(imgs[ch], cv2.COLOR_BGR2GRAY)
        pts[ch] = {k: snap(g, v) for k, v in ann[ch].items()}
    best = None
    box0 = [W * 0.25, W * 0.25, W * 0.75]
    lo = np.array([W * 1.2] + [W * 0.1, W * 0.05, W * 0.55, 300, -1, -1, 810, 390, -4, -4, -4, -4, -4, -12] * 2)
    hi = np.array([W * 3.0] + [W * 0.6, W * 0.45, W * 0.95, 2000, 1, 1, 1110, 690, 4, 4, 4, 2, 2, -1.5] * 2)

    def fit(p0, free):
        free = np.asarray(free)

        def fun(q):
            p = p0.copy(); p[free] = q
            return residuals(p, pts, W)
        r = least_squares(fun, p0[free], bounds=(lo[free], hi[free]), loss="soft_l1", f_scale=3.0, max_nfev=300)
        p = p0.copy(); p[free] = r.x
        return p, r.cost

    def sel(*cols):
        return [1 + NP * i + c for i in range(2) for c in cols]
    pose, dims, foc, k1, k2, cc = sel(*range(8, 14)), [0] + sel(0, 1, 2), sel(3), sel(4), sel(5), sel(6, 7)
    for f, h, yaw, pitch in itertools.product((700, 1000), (3.0, 5.0), (0.5, 0.8), (0.3, 0.6)):
        p = np.array([W * 2.0] + (box0 + init_cam(f, h, yaw, pitch)) * 2)
        try:
            p, c = fit(p, pose)
            p, c = fit(p, pose + foc + dims)
            p, c = fit(p, pose + foc + k1 + dims)
            p, c = fit(p, pose + foc + k1 + k2 + cc + dims)
        except Exception as e:  # noqa
            print("fit failed", e); continue
        if best is None or c < best[1] - 1e-6:
            print(f"init f={f} h={h} yaw={yaw} pitch={pitch}: cost={c:.1f}")
            best = (p, c)
    p = best[0]
    rr = residuals(p, pts, W)
    print(f"RMS residual ~{np.sqrt(np.mean(rr**2)):.2f} px over {len(rr)} samples")
    out = {"pitch": {"L": float(p[0]), "W": W, "boxes": []}, "cameras": {}}
    for i, ch in enumerate(CAMS):
        cam, L, d, b1, b2 = unpack(p, i, W)
        box = dict(x0=0.0, x1=float(d), y0=float(b1), y1=float(b2))
        if ch == "ch3":  # ch3 sits at the diagonally opposite corner
            cam.flip = [float(L), W]
            box = dict(x0=float(L), x1=float(L - d), y0=float(W - b2), y1=float(W - b1))
        out["pitch"]["boxes"].append(box)
        cd = cam.to_dict()
        out["cameras"][ch] = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in cd.items()}
        print(ch, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out["cameras"][ch].items()})
    print("pitch", out["pitch"])
    json.dump(out, open(a.out, "w"), indent=1)
    draw_overlays(out, imgs, pts, a.bg)


def pitch_lines(P):
    L, W = P["L"], P["W"]
    segs = [((0, 0), (L, 0)), ((L, 0), (L, W)), ((L, W), (0, W)), ((0, W), (0, 0)), ((L / 2, 0), (L / 2, W))]
    for b in P["boxes"]:
        segs += [((b["x0"], b["y0"]), (b["x1"], b["y0"])), ((b["x1"], b["y0"]), (b["x1"], b["y1"])),
                 ((b["x1"], b["y1"]), (b["x0"], b["y1"]))]
    return segs


def draw_pitch(img, cam, P, color=(0, 0, 255), grid=True):
    def poly(a, b, col, th=1):
        t = np.linspace(0, 1, 80)[:, None]
        xy = np.asarray(a) * (1 - t) + np.asarray(b) * t
        px = cam.project(np.c_[xy, np.zeros(len(xy))])
        cv2.polylines(img, [px.round().astype(np.int32)], False, col, th, cv2.LINE_AA)
    if grid:
        for x in np.arange(0, P["L"] + 0.1, 5):
            poly((x, 0), (x, P["W"]), (255, 255, 0))
        for y in np.arange(0, P["W"] + 0.1, 5):
            poly((0, y), (P["L"], y), (255, 255, 0))
    for a, b in pitch_lines(P):
        poly(a, b, color, 2)


def draw_overlays(calib, imgs, pts, outdir):
    for ch in CAMS:
        cam = Camera.from_dict(calib["cameras"][ch])
        im = imgs[ch].copy()
        draw_pitch(im, cam, calib["pitch"])
        for px in pts[ch].values():
            for q in px:
                cv2.circle(im, tuple(q.round().astype(int)), 3, (0, 255, 0), -1)
        cv2.imwrite(f"{outdir}/{ch}_calib.jpg", im)


if __name__ == "__main__":
    main()
