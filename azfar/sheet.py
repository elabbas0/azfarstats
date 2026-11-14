"""Contact sheet: one row of crops per identity, sampled across the recording, to eyeball identity purity
and to let a person put names to player numbers.

Usage: python -m azfar.sheet --out data/out/sheet.jpg [--per 10]
"""
from __future__ import annotations
import argparse
import numpy as np, cv2
from .timeline import scan, fmt, GRID_HZ
from .frames import grab


def pick(I, T, pid, per):
    m = np.flatnonzero(I["pid"] == pid)
    cand = []
    for c, key in enumerate(("i0", "i1")):
        idx = I[key][m]; ok = idx >= 0
        b = T[f"box{c}"][idx[ok]]
        h = b[:, 3] - b[:, 1]; asp = h / np.maximum(b[:, 2] - b[:, 0], 1)
        good = (h > 90) & (h < 420) & (asp > 1.9) & (asp < 3.8) & (T[f"conf{c}"][idx[ok]] > 0.7)
        for k, bb in zip(I["k"][m][ok][good], b[good]):
            cand.append((int(k), c, bb))
    if not cand:
        return []
    cand.sort(key=lambda x: x[0])
    ks = np.array([c[0] for c in cand])
    targets = np.linspace(ks[0], ks[-1], per)
    out, used = [], set()
    for t in targets:
        j = int(np.argmin(np.abs(ks - t)))
        if j not in used:
            used.add(j); out.append(cand[j])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="data/identities.npz")
    ap.add_argument("--tracks", default="data/tracks.npz")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/out/sheet.jpg")
    ap.add_argument("--per", type=int, default=10)
    a = ap.parse_args()
    I, T = np.load(a.ids), np.load(a.tracks)
    clips = scan(a.raw)
    chs = sorted({c.ch for c in clips})
    maps = [(c, dict(c.grid_frames())) for c in clips]
    cache = {}
    rows = []
    for pid in np.unique(I["pid"]):
        tiles = []
        for k, c, box in pick(I, T, pid, a.per):
            for clip, gm in maps:
                if clip.ch == chs[c] and k in gm:
                    key = (clip.path, gm[k])
                    if key not in cache:
                        cache[key] = grab(*key)
                    im = cache[key]
                    if im is None:
                        break
                    x1, y1, x2, y2 = box
                    pw, ph = (x2 - x1) * 0.2, (y2 - y1) * 0.08
                    crop = im[max(int(y1 - ph), 0):int(y2 + ph), max(int(x1 - pw), 0):int(x2 + pw)]
                    if crop.size:
                        crop = cv2.resize(crop, (90, 200))
                        cv2.putText(crop, fmt(k / GRID_HZ), (2, 196), 0, 0.38, (0, 255, 255), 1)
                        tiles.append(crop)
                    break
        while len(tiles) < a.per:
            tiles.append(np.zeros((200, 90, 3), np.uint8))
        label = np.zeros((200, 60, 3), np.uint8)
        cv2.putText(label, f"P{pid + 1}", (4, 110), 0, 0.8, (255, 255, 255), 2)
        rows.append(np.hstack([label] + tiles[:a.per]))
        cache.clear()
    cv2.imwrite(a.out, np.vstack(rows))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
