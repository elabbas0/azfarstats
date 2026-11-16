"""Training data for far-ball detection, labelled by the other camera.

When camera A sees the ball confidently on the ground close to itself (position known to ~0.3 m), the
same moment in camera B shows the ball far away, in B's far strip. We project A's position into B and snap
to the nearest small motion blob or weak detection in B; that is the label. The far strip is cropped and
enlarged FAR_SCALE times so a 7 px ball becomes ~11 px, and cut into 640x384 tiles.

Usage: python -m azfar.fardata --n 2500
"""
from __future__ import annotations
import argparse, json, os
from multiprocessing import Pool
import numpy as np, cv2
from .camera import Camera
from .timeline import scan

FAR_BOX = (110, 64, 1646, 320)    # x0, y0, x1, y1 (native px): where far balls appear in both cameras
FAR_SCALE = 1.5
TW, TH = 640, 384
SNAP_PX = 10
HMS = lambda s: sum(int(p) * m for p, m in zip(s.split(":"), (3600, 60, 1)))


def load_dets(clips, ch):
    out = {}
    for c in clips:
        if c.ch != ch:
            continue
        n = os.path.basename(c.path)[:-4]
        if not os.path.exists(f"data/balldet_out/{n}.npz"):
            continue
        d = dict(np.load(f"data/balldet_out/{n}.npz"))
        sp = f"data/ballscan/{n}.npz"
        s = dict(np.load(sp)) if os.path.exists(sp) else dict(f=np.zeros(0, int), xy=np.zeros((0, 2)), size=np.zeros(0), col=np.zeros((0, 3)))
        col = s.get("col", np.zeros((len(s["f"]), 3)))
        white = (col[:, 1] >= 120) & (col[:, 0] <= 75) & (col[:, 2] <= 0.15)      # ball-coloured blobs only
        s = dict(f=s["f"][white], xy=s["xy"][white], size=s["size"][white])
        b = d["box"]
        out[c.path] = dict(clip=c, f=d["f"], xy=np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], conf=d["conf"],
                           sz=np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]), bf=s["f"], bxy=s["xy"], bsz=s["size"])
    return out


def labels(calib, clips, A, B, lo, hi):
    camA, camB = Camera.from_dict(calib["cameras"][A]), Camera.from_dict(calib["cameras"][B])
    DA, DB = load_dets(clips, A), load_dets(clips, B)
    out = []
    for da in DA.values():
        c = da["clip"]; t = c.frame_time(da["f"])
        s = (da["conf"] >= 0.7) & (t >= lo) & (t < hi)
        g, ok = camA.to_ground(da["xy"][s], height=0.11)
        rA = np.linalg.norm(camA._to_local(np.c_[g, np.zeros(len(g))]) - camA.C, axis=1)
        keep = ok & (rA < 15)
        for tt, gg in zip(t[s][keep], g[keep]):
            px = camB.project(np.array([gg[0], gg[1], -0.11]))
            x0, y0, x1, y1 = FAR_BOX
            if not (x0 + 8 < px[0] < x1 - 8 and y0 + 8 < px[1] < y1 - 8):
                continue
            for db in DB.values():
                cb = db["clip"]; fB = int(round((tt - cb.t0) * 15))
                if not (0 <= fB < cb.n_frames):
                    continue
                cand = []
                m = db["f"] == fB
                cand += [(xy, sz) for xy, sz, cf_ in zip(db["xy"][m], db["sz"][m], db["conf"][m])]
                m = db["bf"] == fB
                cand += [(xy, sz) for xy, sz in zip(db["bxy"][m], db["bsz"][m])]
                if cand:
                    d = [np.linalg.norm(xy - px) for xy, _ in cand]
                    j = int(np.argmin(d))
                    if d[j] <= SNAP_PX:
                        out.append((cb.path, fB, float(cand[j][0][0]), float(cand[j][0][1]), float(max(cand[j][1], 6.0))))
                break
    return out


def cut(path, items, outdir, seed):
    cv2.setNumThreads(1)
    rng = np.random.default_rng(seed)
    want = {}
    for f, x, y, s in items:
        want.setdefault(f, []).append((x, y, s))
    cap = cv2.VideoCapture(path); base = os.path.basename(path)[:-4]
    x0, y0, x1, y1 = FAR_BOX
    i, n = 0, [0, 0]
    last = max(want)
    while i <= last:
        if i not in want:
            if not cap.grab():
                break
            i += 1; continue
        ok, im = cap.read()
        if not ok:
            break
        band = cv2.resize(im[y0:y1, x0:x1], None, fx=FAR_SCALE, fy=FAR_SCALE, interpolation=cv2.INTER_CUBIC)
        H, W = band.shape[:2]
        bx = [((x - x0) * FAR_SCALE, (y - y0) * FAR_SCALE, s * FAR_SCALE) for x, y, s in want[i]]
        cx, cy, _ = bx[0]
        tiles = [(int(np.clip(cx - rng.uniform(30, TW - 30), 0, W - TW)), 0, True)]
        for _ in range(10):
            tx = int(rng.uniform(0, W - TW))
            if not (tx - 30 < cx < tx + TW + 30):
                tiles.append((tx, 0, False)); break
        for k, (tx, ty, pos) in enumerate(tiles):
            stem = f"{base}_{i:06d}_{k}"
            cv2.imwrite(os.path.join(outdir, "images", stem + ".jpg"), band[ty:ty + TH, tx:tx + TW], [cv2.IMWRITE_JPEG_QUALITY, 93])
            lines = []
            for x, y, s in bx:
                if tx + 2 < x < tx + TW - 2 and ty + 2 < y < ty + TH - 2:
                    w = max(s, 8.0)
                    lines.append(f"0 {(x - tx) / TW:.5f} {(y - ty) / TH:.5f} {w / TW:.5f} {w / TH:.5f}")
            open(os.path.join(outdir, "labels", stem + ".txt"), "w").write("\n".join(lines))
            n[0 if lines else 1] += 1
        i += 1
    cap.release()
    return tuple(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw"); ap.add_argument("--calib", default="config/calibration.json")
    ap.add_argument("--session", default="config/session.json"); ap.add_argument("--out", default="data/fardet")
    ap.add_argument("--n", type=int, default=2500)
    a = ap.parse_args()
    calib = json.load(open(a.calib)); clips = scan(a.raw)
    m = json.load(open(a.session))["matches"][0]; lo, hi = HMS(m["start"]), HMS(m["end"])
    split = lo + 0.8 * (hi - lo)
    L = labels(calib, clips, "ch2", "ch3", lo, hi) + labels(calib, clips, "ch3", "ch2", lo, hi)
    print(f"{len(L)} far-ball labels from the other camera", flush=True)
    json.dump(L, open(os.path.join(os.path.dirname(a.out) or ".", "fardet_labels.json"), "w"))
    rng = np.random.default_rng(0)
    pick = rng.choice(len(L), min(a.n, len(L)), replace=False)
    jobs = {}
    clipt = {c.path: c for c in clips}
    for i in pick:
        p, f, x, y, s = L[i]
        sp = "val" if clipt[p].frame_time(f) >= split else "train"
        jobs.setdefault((p, sp), []).append((f, x, y, s))
    for sp in ("train", "val"):
        os.makedirs(os.path.join(a.out, sp, "images"), exist_ok=True); os.makedirs(os.path.join(a.out, sp, "labels"), exist_ok=True)
    with Pool(4) as pool:
        res = pool.starmap(cut, [(p, v, os.path.join(a.out, sp), k) for k, ((p, sp), v) in enumerate(jobs.items())])
    print("tiles (with ball, without):", res)
    open(os.path.join(a.out, "data.yaml"), "w").write(f"path: {os.path.abspath(a.out)}\ntrain: train/images\nval: val/images\nnames:\n  0: ball\n")


if __name__ == "__main__":
    main()
