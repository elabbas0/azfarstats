"""Refine the calibration with the players themselves.

The same player seen by both cameras at the same instant must stand on the same spot. We collect
confident, upright, isolated detections that both cameras agree on, and re-fit all camera
parameters and the pitch length so that the pitch lines AND these foot points agree.

Usage: python -m azfar.refine --det data/det --raw data/raw   (rewrites config/calibration.json)
"""
from __future__ import annotations
import argparse, json
import numpy as np
from scipy.spatial import cKDTree
from .calibrate import load_points, pack, to_calib, solve, residuals, draw_overlays, CAMS
from .fuse import load_camera_dets, aim_cameras
from .timeline import scan


def collect_pairs(det_dir, raw_dir, calib, max_pairs=4000, seed=0):
    clips = scan(raw_dir)
    base = {k: v for k, v in calib.items() if k != "aim"}
    V = [load_camera_dets(det_dir, clips, aim_cameras(base, ch), ch, calib["pitch"]) for ch in CAMS]
    sel = []
    for v in V:
        b = v["box"]
        ok = (v["conf"] > 0.6) & ((b[:, 3] - b[:, 1]) / (b[:, 2] - b[:, 0]) > 1.8) & (v["h"] > 1.5) & (v["h"] < 2.1) & (v["rng"] < 34)
        # isolated within its own view
        t = np.c_[v["xy"], v["k"] * 100.0]
        d, _ = cKDTree(t).query(t, k=2)
        sel.append(ok & (d[:, 1] > 3.0))
    ta = np.c_[V[0]["xy"][sel[0]], V[0]["k"][sel[0]] * 100.0]
    tb = np.c_[V[1]["xy"][sel[1]], V[1]["k"][sel[1]] * 100.0]
    da, ia = cKDTree(tb).query(ta, distance_upper_bound=3.0)
    db, ib = cKDTree(ta).query(tb, distance_upper_bound=3.0)
    m = np.isfinite(da)
    m[m] &= ib[ia[m]] == np.flatnonzero(m)  # mutual nearest
    ja, jb = np.flatnonzero(sel[0])[m], np.flatnonzero(sel[1])[ia[m]]
    # balance over the pitch so the busy areas do not dominate
    xy = V[0]["xy"][ja]
    cell = (xy[:, 0] // 3).astype(int) * 100 + (xy[:, 1] // 3).astype(int)
    rng = np.random.default_rng(seed)
    keep = []
    cells = np.unique(cell)
    per = max(8, max_pairs // len(cells))
    for c in cells:
        idx = np.flatnonzero(cell == c)
        keep.extend(rng.choice(idx, min(per, len(idx)), replace=False))
    keep = np.array(keep)
    foot = lambda b: np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3]]
    w = 1.0 / np.sqrt(0.25 + (V[0]["rng"][ja[keep]] / 18) ** 4 + (V[1]["rng"][jb[keep]] / 18) ** 4)
    return foot(V[0]["box"][ja[keep]]), foot(V[1]["box"][jb[keep]]), w


def disagreement(p, W, pairs):
    r = residuals(p, {ch: {} for ch in CAMS}, W, pairs)
    d = (r / 6.0 / pairs[2].repeat(2)).reshape(-1, 2)
    return np.median(d, 0), np.median(np.linalg.norm(d, axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="data/det")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--bg", default="data/bg")
    ap.add_argument("--ann", default="config/calib_annotations.json")
    ap.add_argument("--calib", default="config/calibration.json")
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()
    calib = json.load(open(a.calib))
    W = calib["pitch"]["W"]
    pts, imgs = load_points(a.ann, a.bg, skip=("GL",))  # the far goal line is too coarse to trust
    for it in range(a.rounds):
        pairs = collect_pairs(a.det, a.raw, calib)
        p0 = pack(calib)
        b0, m0 = disagreement(p0, W, pairs)
        p, cost = solve(pts, W, pairs, p0)
        b1, m1 = disagreement(p, W, pairs)
        lines = residuals(p, pts, W)
        print(f"round {it}: {len(pairs[0])} pairs; cross-camera bias {b0.round(2)} -> {b1.round(2)} m, "
              f"median gap {m0:.2f} -> {m1:.2f} m; line RMS {np.sqrt(np.mean(lines**2)):.2f} px; L={p[0]:.2f}")
        calib = to_calib(p, W)
    json.dump(calib, open(a.calib, "w"), indent=1)
    draw_overlays(calib, imgs, pts, a.bg)
    for ch in CAMS:
        c = calib["cameras"][ch]
        print(ch, f"f={c['f']:.0f} k1={c['k1']:.3f} k2={c['k2']:.3f} c=({c['cx']:.0f},{c['cy']:.0f}) pos={np.round(c['C'], 2).tolist()}")


if __name__ == "__main__":
    main()
