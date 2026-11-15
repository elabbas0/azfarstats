"""Build a training set for a dedicated ball detector from this recording.

Positive labels: detector "ball" hits that pass every check (confidence, learnt recogniser, colour, not on a
fixed-clutter spot), plus hand-checked balls. For each chosen frame we cut a 640x640 tile from the frame
scaled by SCALE (the scale the detector will run at), with the ball at a random spot inside the tile, and
one or two ball-free tiles from the same frame (the ball is known to be elsewhere, so they are true
negatives: players, boots, the wall light, the curtain...).

Usage: python -m azfar.balldata --n 3000 --out data/balldet
"""
from __future__ import annotations
import argparse, json, os
from multiprocessing import Pool
import numpy as np, cv2
from .timeline import scan, GRID_HZ

SCALE = 0.8          # detector runs on 1536x864 frames
TILE = 640
HMS = lambda s: sum(int(p) * m for p, m in zip(s.split(":"), (3600, 60, 1)))


def static_spots(scan_npz, det_npz, sel, fm, lo, hi, clip, cell=4, sec=60):
    """Image cells that produce candidates in >= `sec` separate seconds (fixed clutter)."""
    s = dict(np.load(scan_npz))
    t = clip.frame_time(s["f"]); ok = (t >= lo) & (t < hi)
    xy = s["xy"][ok]; secs = t[ok].astype(int)
    k = (xy[:, 0] // cell).astype(np.int64) * 1000 + (xy[:, 1] // cell).astype(np.int64)
    u, n = np.unique(np.unique(k * 10**6 + secs) // 10**6, return_counts=True)
    return {int(a) for a, b in zip(u, n) if b >= sec}


def positives(clip, a, lo, hi):
    name = os.path.basename(clip.path)[:-4]
    d = dict(np.load(os.path.join(a.det, name + ".npz")))
    sel = np.flatnonzero(d["cls"] == 32)
    pd = np.load(os.path.join(a.patch, name + "_pdet.npy"))
    sc = dict(np.load(os.path.join(a.scan, name + ".npz")))
    col = sc["det_col"][sel] if "det_col" in sc else np.zeros((len(sel), 3))
    fm = dict(clip.grid_frames())
    static = static_spots(os.path.join(a.scan, name + ".npz"), None, sel, fm, lo, hi, clip)
    out = []
    for j, r in enumerate(sel):
        f = fm.get(int(d["k"][r]))
        if f is None or not (lo <= clip.frame_time(f) < hi):
            continue
        sat, val, yel = col[j]
        if d["conf"][r] < a.min_conf or pd[j] < 0.85 or sat > 88 or val < 110 or (yel > 0.22 and sat > 65):
            continue
        x1, y1, x2, y2 = d["box"][r]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        c4 = int(cx // 4) * 1000 + int(cy // 4)
        if any((c4 + dx * 1000 + dy) in static for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
            continue
        out.append((int(f), float(x1), float(y1), float(x2), float(y2)))
    if os.path.exists(a.labels):
        for h in json.load(open(a.labels)):
            if h["label"] == "1" and h["clip"] == name:
                s_ = max(h["size"], 8)
                out.append((int(h["frame"]), h["x"] - s_ / 2, h["y"] - s_ / 2, h["x"] + s_ / 2, h["y"] + s_ / 2))
    return out


def cut_clip(path, items, outdir, seed):
    """items: {frame: [box...]} -> write tiles + YOLO labels. One sequential pass over the clip."""
    cv2.setNumThreads(1)
    rng = np.random.default_rng(seed)
    cap = cv2.VideoCapture(path)
    base = os.path.basename(path)[:-4]
    n_pos = n_neg = 0
    i, last = 0, max(items)
    while i <= last:
        if i not in items:
            if not cap.grab():
                break
            i += 1; continue
        ok, im = cap.read()
        if not ok:
            break
        im = cv2.resize(im, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        H, W = im.shape[:2]
        boxes = np.array(items[i]) * SCALE
        bx, by = (boxes[0, 0] + boxes[0, 2]) / 2, (boxes[0, 1] + boxes[0, 3]) / 2
        # positive tile: ball at a random place inside it
        x0 = int(np.clip(bx - rng.uniform(40, TILE - 40), 0, W - TILE)); y0 = int(np.clip(by - rng.uniform(40, TILE - 40), 0, H - TILE))
        tiles = [(x0, y0, True)]
        # negative tiles: anywhere at least 40 px clear of the ball
        for _ in range(12):
            if len(tiles) >= 1 + NEG_PER_POS:
                break
            nx, ny = int(rng.uniform(0, W - TILE)), int(rng.uniform(0, H - TILE))
            if not (nx - 40 < bx < nx + TILE + 40 and ny - 40 < by < ny + TILE + 40):
                tiles.append((nx, ny, False))
        for t_, (tx, ty, pos) in enumerate(tiles):
            stem = f"{base}_{i:06d}_{t_}"
            cv2.imwrite(os.path.join(outdir, "images", stem + ".jpg"), im[ty:ty + TILE, tx:tx + TILE], [cv2.IMWRITE_JPEG_QUALITY, 92])
            lines = []
            for x1, y1, x2, y2 in boxes:
                x1, x2 = max(x1 - tx, 0), min(x2 - tx, TILE); y1, y2 = max(y1 - ty, 0), min(y2 - ty, TILE)
                if x2 - x1 > 2 and y2 - y1 > 2:
                    w_, h_ = max(x2 - x1, 6), max(y2 - y1, 6)
                    lines.append(f"0 {(x1 + x2) / 2 / TILE:.5f} {(y1 + y2) / 2 / TILE:.5f} {w_ / TILE:.5f} {h_ / TILE:.5f}")
            open(os.path.join(outdir, "labels", stem + ".txt"), "w").write("\n".join(lines))
            n_pos += bool(lines); n_neg += not lines
        i += 1
    cap.release()
    return n_pos, n_neg


NEG_PER_POS = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw"); ap.add_argument("--det", default="data/det")
    ap.add_argument("--scan", default="data/ballscan"); ap.add_argument("--patch", default="data/ballpatch")
    ap.add_argument("--labels", default="data/labels/labelled_s3.json")
    ap.add_argument("--session", default="config/session.json"); ap.add_argument("--out", default="data/balldet")
    ap.add_argument("--n", type=int, default=3000); ap.add_argument("--min-conf", type=float, default=0.45)
    ap.add_argument("--split-at", default="", help="frames after this wall-clock time go to validation (default: last 20%)")
    a = ap.parse_args()
    m = json.load(open(a.session))["matches"][0]
    lo, hi = HMS(m["start"]), HMS(m["end"])
    split = HMS(a.split_at) if a.split_at else lo + 0.8 * (hi - lo)
    clips = [c for c in scan(a.raw) if os.path.exists(os.path.join(a.patch, os.path.basename(c.path)[:-4] + "_pdet.npy"))]
    allpos = {c.path: positives(c, a, lo, hi) for c in clips}
    tot = sum(len(v) for v in allpos.values())
    print(f"{tot} trusted ball positions available", flush=True)
    rng = np.random.default_rng(0)
    jobs = []
    for split_name in ("train", "val"):
        os.makedirs(os.path.join(a.out, split_name, "images"), exist_ok=True)
        os.makedirs(os.path.join(a.out, split_name, "labels"), exist_ok=True)
    for c in clips:
        v = allpos[c.path]
        take = rng.choice(len(v), min(len(v), int(round(a.n * len(v) / max(tot, 1)))), replace=False)
        for split_name in ("train", "val"):
            items = {}
            for t in take:
                f, *box = v[t]
                in_val = c.frame_time(f) >= split
                if (split_name == "val") == in_val:
                    items.setdefault(f, []).append(box)
            if items:
                jobs.append((c.path, items, os.path.join(a.out, split_name), len(jobs)))
    with Pool(4) as p:
        res = p.starmap(cut_clip, jobs)
    print("tiles written (positive, negative):", [r for r in res])
    open(os.path.join(a.out, "data.yaml"), "w").write(
        f"path: {os.path.abspath(a.out)}\ntrain: train/images\nval: val/images\nnames:\n  0: ball\n")


if __name__ == "__main__":
    main()
