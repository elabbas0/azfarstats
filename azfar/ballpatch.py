"""Stage 1d: small image patches around every ball candidate, for the ball/not-ball recogniser.

For each clip, cuts a square patch (3x the candidate's size, so a ball fills the middle third) around
every motion blob found by azfar.ballscan and every detector ball hit, resized to 24x24.

Usage: python -m azfar.ballpatch --only <clip substring>
Writes data/ballpatch/<clip>_blob.npy (rows aligned with data/ballscan/<clip>.npz) and
       data/ballpatch/<clip>_det.npy  (rows aligned with the cls==32 rows of data/det/<clip>.npz).
"""
from __future__ import annotations
import argparse, os, time
import numpy as np, cv2
from .timeline import scan

P = 24


def cut(im, x, y, size):
    s = int(np.clip(3 * size, 12, 90))
    h = s // 2
    x0, y0 = int(round(x)) - h, int(round(y)) - h
    pad = cv2.copyMakeBorder(im[max(y0, 0):y0 + s, max(x0, 0):x0 + s], max(0, -y0), 0, max(0, -x0), 0, cv2.BORDER_REPLICATE) \
        if (x0 < 0 or y0 < 0) else im[y0:y0 + s, x0:x0 + s]
    if pad.shape[0] < 2 or pad.shape[1] < 2:
        return np.zeros((P, P, 3), np.uint8)
    return cv2.resize(pad, (P, P), interpolation=cv2.INTER_AREA)


def run(clip, scan_path, det_path, out_blob, out_det):
    s = np.load(scan_path); d = np.load(det_path)
    bf, bxy, bsz = s["f"], s["xy"], s["size"]
    sel = np.flatnonzero(d["cls"] == 32)
    fm = dict(clip.grid_frames())
    df = np.array([fm.get(int(k), -1) for k in d["k"][sel]])
    B = np.lib.format.open_memmap(out_blob + ".tmp.npy", "w+", np.uint8, (len(bf), P, P, 3))
    D = np.zeros((len(sel), P, P, 3), np.uint8)
    ob = np.argsort(bf, kind="stable"); od = np.argsort(df, kind="stable")
    bi = di = 0
    cap = cv2.VideoCapture(clip.path)
    i, t0 = 0, time.time()
    last = max(bf.max() if len(bf) else 0, df.max() if len(df) else 0)
    while i <= last:
        need = (bi < len(ob) and bf[ob[bi]] == i) or (di < len(od) and df[od[di]] == i)
        if not need:
            if not cap.grab():
                break
            i += 1; continue
        ok, im = cap.read()
        if not ok:
            break
        while bi < len(ob) and bf[ob[bi]] == i:
            r = ob[bi]; B[r] = cut(im, *bxy[r], bsz[r]); bi += 1
        while di < len(od) and df[od[di]] == i:
            r = od[di]; x1, y1, x2, y2 = d["box"][sel[r]]
            D[r] = cut(im, (x1 + x2) / 2, (y1 + y2) / 2, min(x2 - x1, y2 - y1)); di += 1
        i += 1
        if i % 9000 == 0:
            print(f"{os.path.basename(clip.path)} {i}/{clip.n_frames} {i / (time.time() - t0):.0f} fps", flush=True)
    cap.release()
    B.flush(); del B
    os.replace(out_blob + ".tmp.npy", out_blob)
    np.save(out_det, D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--det", default="data/det")
    ap.add_argument("--scan", default="data/ballscan")
    ap.add_argument("--out", default="data/ballpatch")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    cv2.setNumThreads(1)
    os.makedirs(a.out, exist_ok=True)
    for clip in scan(a.raw):
        name = os.path.basename(clip.path).replace(".mp4", "")
        sp, dp = os.path.join(a.scan, name + ".npz"), os.path.join(a.det, name + ".npz")
        ob, od = os.path.join(a.out, name + "_blob.npy"), os.path.join(a.out, name + "_det.npy")
        if a.only not in name or not (os.path.exists(sp) and os.path.exists(dp)) or os.path.exists(ob):
            continue
        run(clip, sp, dp, ob, od)
        print("done", name, flush=True)


if __name__ == "__main__":
    main()
