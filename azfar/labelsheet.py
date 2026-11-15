"""Make contact sheets of ball candidates for manual verification.

Samples candidates the tracker actually considers (recogniser score >= 0.3), weighted to the uncertain
ones, across the match window, and writes 8x8 sheets of zoomed crops (candidate in the centre, marked by
four ticks). Each sheet's candidate list is saved so labels can be attached later.

Usage: python -m azfar.labelsheet --n 512 --seed 3
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, cv2
from .timeline import scan, GRID_HZ

HMS = lambda s: sum(int(p) * m for p, m in zip(s.split(":"), (3600, 60, 1)))
TILE = 120


def pool(clip, scan_dir, det_dir, patch_dir, lo, hi):
    name = os.path.basename(clip.path).replace(".mp4", "")
    s = np.load(os.path.join(scan_dir, name + ".npz")); d = np.load(os.path.join(det_dir, name + ".npz"))
    pb = np.load(os.path.join(patch_dir, name + "_pblob.npy")); pd = np.load(os.path.join(patch_dir, name + "_pdet.npy"))
    sel = np.flatnonzero(d["cls"] == 32)
    fm = dict(clip.grid_frames())
    out = []
    t = clip.frame_time(s["f"])
    for r in np.flatnonzero((pb >= 0.3) & (t >= lo) & (t < hi)):
        out.append((name, "blob", int(r), int(s["f"][r]), float(s["xy"][r, 0]), float(s["xy"][r, 1]), float(s["size"][r]), float(pb[r])))
    for j, r in enumerate(sel):
        f = fm.get(int(d["k"][r]))
        if f is None or not (lo <= clip.frame_time(f) < hi) or (pd[j] < 0.1 and d["conf"][r] < 0.3):
            continue
        x1, y1, x2, y2 = d["box"][r]
        out.append((name, "det", int(j), int(f), float((x1 + x2) / 2), float((y1 + y2) / 2), float(min(x2 - x1, y2 - y1)), float(pd[j])))
    return out


def crops_for_clip(path, items):
    cv2.setNumThreads(1)
    want = {}
    for it in items:
        want.setdefault(it[3], []).append(it)
    cap = cv2.VideoCapture(path)
    out, i, last = [], 0, max(want)
    while i <= last:
        if i in want:
            ok, im = cap.read()
            if not ok:
                break
            for name, kind, row, f, x, y, sz, pp in want[i]:
                half = int(max(30, 4 * sz))
                x0, y0 = int(x) - half, int(y) - half
                crop = cv2.copyMakeBorder(im, half, half, half, half, cv2.BORDER_CONSTANT)[y0 + half:y0 + 3 * half, x0 + half:x0 + 3 * half]
                crop = cv2.resize(crop, (TILE, TILE), interpolation=cv2.INTER_CUBIC)
                c_, r_ = TILE // 2, int(max(9, sz / (2 * half) * TILE * 0.7))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    p0 = (c_ + dx * (r_ + 3), c_ + dy * (r_ + 3)); p1 = (c_ + dx * (r_ + 11), c_ + dy * (r_ + 11))
                    cv2.line(crop, p0, p1, (0, 0, 255), 2)
                out.append((crop, dict(clip=name, kind=kind, row=row, frame=f, x=x, y=y, size=sz, p=pp)))
        elif not cap.grab():
            break
        i += 1
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw"); ap.add_argument("--det", default="data/det")
    ap.add_argument("--scan", default="data/ballscan"); ap.add_argument("--patch", default="data/ballpatch")
    ap.add_argument("--session", default="config/session.json"); ap.add_argument("--out", default="data/labels")
    ap.add_argument("--n", type=int, default=512); ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    m = json.load(open(a.session))["matches"][0]
    lo, hi = HMS(m["start"]), HMS(m["end"])
    clips = {os.path.basename(c.path).replace(".mp4", ""): c for c in scan(a.raw)}
    P = []
    for c in clips.values():
        if os.path.exists(os.path.join(a.patch, os.path.basename(c.path).replace(".mp4", "_pblob.npy"))):
            P += pool(c, a.scan, a.det, a.patch, lo, hi)
    p = np.array([x[7] for x in P])
    wt = np.where(p < 0.97, 3.0, 1.0)                     # uncertain candidates are the informative ones
    rng = np.random.default_rng(a.seed)
    pick = rng.choice(len(P), a.n, replace=False, p=wt / wt.sum())
    pick = sorted(pick, key=lambda i: (P[i][0], P[i][3]))
    # one sequential pass per clip (seeking HEVC frame by frame is far too slow), clips in parallel
    from multiprocessing import Pool
    jobs = {}
    for i in pick:
        jobs.setdefault(P[i][0], []).append(P[i])
    with Pool(4) as pool_:
        res = pool_.starmap(crops_for_clip, [(clips[n].path, v) for n, v in jobs.items()])
    tiles, meta = [], []
    for part in res:
        for crop, md in part:
            tiles.append(crop); meta.append(md)
    order = rng.permutation(len(meta))                    # shuffle so neighbours are unrelated
    meta = [meta[i] for i in order]; tiles = [tiles[i] for i in order]
    os.makedirs(a.out, exist_ok=True)
    for s0 in range(0, len(meta), 64):
        sheet = np.zeros((8 * TILE, 8 * TILE, 3), np.uint8)
        for j, t in enumerate(tiles[s0:s0 + 64]):
            r, c = divmod(j, 8)
            t = t.copy(); cv2.putText(t, str(j + 1), (3, 14), 0, 0.45, (0, 255, 255), 1)
            sheet[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = t
        cv2.imwrite(os.path.join(a.out, f"sheet_s{a.seed}_{s0 // 64:02d}.jpg"), sheet)
    json.dump(meta, open(os.path.join(a.out, f"cands_s{a.seed}.json"), "w"))
    print(f"{len(P)} candidates in the pool; {len(meta)} sampled into {(len(meta) + 63) // 64} sheets")


if __name__ == "__main__":
    main()
