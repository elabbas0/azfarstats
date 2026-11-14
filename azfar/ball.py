"""Stage 2b: find the single most plausible ball trajectory from both cameras.

Candidates are the detector's "sports ball" hits (kept down to very low confidence). We drop
fixed false alarms (lamps, signs), check the apparent size against the distance, triangulate when
both cameras see the ball, and then pick one path through the whole recording by dynamic
programming (high confidence, physically possible jumps).

Usage: python -m azfar.ball --det data/det --raw data/raw --out data/ball.npz
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from .camera import Camera
from .fuse import aim_cameras
from .timeline import scan, GRID_HZ

DT = 1.0 / GRID_HZ
BALL_R = 0.11
V_BALL = 28.0        # m/s, hardest plausible shot
MAX_GAP = 10         # ticks the ball may be unseen inside one path piece
TOP = 6              # candidates kept per tick
H_MAX = 4.0          # highest the ball is assumed to fly, m


def load(det_dir, clips, calib, ch):
    cams = aim_cameras(calib, ch)
    K, PX, SZ, CF, CI = [], [], [], [], []
    for c in clips:
        if c.ch != ch:
            continue
        p = os.path.join(det_dir, os.path.basename(c.path).replace(".mp4", ".npz"))
        if not os.path.exists(p):
            continue
        d = np.load(p)
        m = d["cls"] == 32
        b = d["box"][m]
        k = d["k"][m]
        K.append(k); PX.append(np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2])
        SZ.append(np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])); CF.append(d["conf"][m])
        t = k / GRID_HZ
        ci = np.zeros(len(k), int)
        for wi, (t0, t1, cam) in enumerate(cams):
            ci[t >= t0] = wi
        CI.append(ci)
    K, PX, SZ, CF, CI = (np.concatenate(x) for x in (K, PX, SZ, CF, CI))
    # fixed false alarms: the same image cell firing for minutes
    cell = (PX[:, 0] // 14).astype(int) * 1000 + (PX[:, 1] // 14).astype(int)
    u, inv, cnt = np.unique(cell, return_inverse=True, return_counts=True)
    keep = cnt[inv] < 400
    K, PX, SZ, CF, CI = K[keep], PX[keep], SZ[keep], CF[keep], CI[keep]
    # rays and on-ground guess, per aim window
    O = np.zeros((len(K), 3)); D = np.zeros((len(K), 3)); G = np.zeros((len(K), 2)); ok = np.zeros(len(K), bool)
    rng = np.zeros(len(K))
    for wi, (t0, t1, cam) in enumerate(cams):
        m = CI == wi
        if not m.any():
            continue
        d = cam.rays(PX[m])
        Cg = cam._to_global(cam.C[None])[0].copy(); Cg[2] = cam.C[2]
        if cam.flip is not None:
            d = d * np.array([-1, -1, 1])
        O[m] = Cg; D[m] = d
        g, o = cam.to_ground(PX[m], height=BALL_R)
        G[m] = g; ok[m] = o
        rng[m] = np.linalg.norm(np.c_[g, np.zeros(len(g))] - Cg, axis=1)
        f = cam.f
    expect = f * 2 * BALL_R / np.maximum(rng, 1.0)          # expected diameter in px if on the ground
    ratio = SZ / np.maximum(expect, 1e-3)
    good = ok & (ratio > 0.45) & (ratio < 6.0)              # bigger than expected = closer = airborne, allowed
    return dict(k=K[good], px=PX[good], conf=CF[good], O=O[good], D=D[good], g=G[good], rng=rng[good])


def triangulate(Oa, Da, Ob, Db):
    """Closest points between two rays. Returns midpoint and miss distance."""
    w = Oa - Ob
    a, b, c = Da @ Da, Da @ Db, Db @ Db
    d, e = Da @ w, Db @ w
    den = a * c - b * b
    if abs(den) < 1e-9:
        return None, 1e9
    s, t = (b * e - c * d) / den, (a * e - b * d) / den
    if s <= 0 or t <= 0:
        return None, 1e9
    pa, pb = Oa + s * Da, Ob + t * Db
    return (pa + pb) / 2, float(np.linalg.norm(pa - pb))


def candidates(A, B, pitch):
    """Per tick: list of (x, y, z, score, n_views)."""
    L, W = pitch["L"], pitch["W"]
    ks = np.union1d(A["k"], B["k"])
    ia = {k: np.flatnonzero(A["k"] == k) for k in []}
    out = {}
    sa, sb = np.argsort(A["k"], kind="stable"), np.argsort(B["k"], kind="stable")
    ba = np.searchsorted(A["k"][sa], [ks, ks + 1]); bb = np.searchsorted(B["k"][sb], [ks, ks + 1])
    for n, k in enumerate(ks):
        ia_, ib_ = sa[ba[0][n]:ba[1][n]], sb[bb[0][n]:bb[1][n]]
        c, used_a, used_b = [], set(), set()
        for i in ia_:
            for j in ib_:
                P, miss = triangulate(A["O"][i], A["D"][i], B["O"][j], B["D"][j])
                if P is not None and miss < 0.8 and -P[2] > -0.4 and -P[2] < 8:
                    c.append((P[0], P[1], max(-P[2], BALL_R), A["conf"][i] + B["conf"][j] + 0.3, 2))
                    used_a.add(i); used_b.add(j)
        for V, idx, used in ((A, ia_, used_a), (B, ib_, used_b)):
            for i in idx:
                if i in used:
                    continue
                # single view: the ball is somewhere on this sight line. XY(h) = A + Bv * h for height h.
                O_, D_ = V["O"][i], V["D"][i]
                A_ = O_[:2] - D_[:2] * O_[2] / D_[2]
                Bv = -D_[:2] / D_[2]
                c.append((V["g"][i, 0], V["g"][i, 1], np.nan, V["conf"][i] * (1.0 if V["rng"][i] < 24 else 0.6), 1, *A_, *Bv))
        c = [x + (x[0], x[1], 0.0, 0.0) if len(x) == 5 else x for x in c]   # triangulated: fixed point, no freedom

        def on_pitch(x):  # anywhere along the plausible part of the sight line counts
            return any(-3 < x[5] + x[7] * h < L + 3 and -3 < x[6] + x[8] * h < W + 3 for h in (BALL_R, 1.0, 2.0, H_MAX))
        c = [x for x in c if on_pitch(x)]
        c.sort(key=lambda x: -x[3])
        if c:
            out[int(k)] = np.array(c[:TOP])
    return out


def seg_dist(a, b, p):
    """Distance from point(s) p to segment(s) a-b (2-D, broadcasting)."""
    ab = b - a
    t = np.clip(((p - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9), 0, 1)
    return np.linalg.norm(a + ab * t[..., None] - p, axis=-1)


def solve_heights(k, nodes, lam=3.0, rounds=6, anchor=None):
    """Give every single-camera sighting a height so the ball's path over the ground is smooth.

    Each sighting can slide along its sight line: XY = A + Bv * h, h >= ball radius. Within a
    continuous piece of the path we minimise changes of ground velocity (robustly, so a real kick
    is still a kink) plus a mild pull to the ground, which is where the ball usually is."""
    from scipy.optimize import lsq_linear
    xy = nodes[:, :2].copy()
    h = np.where(nodes[:, 4] == 2, nodes[:, 2], BALL_R)
    A, Bv = nodes[:, 5:7], nodes[:, 7:9]
    free_all = nodes[:, 4] == 1
    cuts = np.flatnonzero(np.diff(k) > MAX_GAP) + 1
    for s, e in zip(np.r_[0, cuts], np.r_[cuts, len(k)]):
        n = e - s
        free = np.flatnonzero(free_all[s:e])
        if n < 3 or not len(free):
            continue
        col = -np.ones(n, int); col[free] = np.arange(len(free))
        t = k[s:e] * DT
        w = np.ones(n - 2)
        for _ in range(rounds):
            rows, rhs = [], []
            M = np.zeros(((n - 2) * 2 + len(free), len(free)))
            y = np.zeros((n - 2) * 2 + len(free))
            for i in range(1, n - 1):
                d0, d1 = t[i] - t[i - 1], t[i + 1] - t[i]
                for j, cf in ((i - 1, 1 / d0), (i, -1 / d0 - 1 / d1), (i + 1, 1 / d1)):   # velocity change at i
                    for ax in (0, 1):
                        r = (i - 1) * 2 + ax
                        y[r] -= w[i - 1] * cf * A[s + j, ax]
                        if col[j] >= 0:
                            M[r, col[j]] += w[i - 1] * cf * Bv[s + j, ax]
                        else:
                            y[r] -= w[i - 1] * cf * (xy[s + j, ax] - A[s + j, ax])
            base = (n - 2) * 2
            pw = lam / np.sqrt(np.maximum(h[s:e][free] - BALL_R, 0) + 0.25)   # sparse: on the ground unless needed
            if anchor is not None:
                pw = pw * np.where(anchor[s:e][free], 10.0, 1.0)            # at a player's feet: it is on the ground
            M[base + np.arange(len(free)), np.arange(len(free))] = pw
            y[base:] = pw * BALL_R
            sol = lsq_linear(M, y, bounds=(BALL_R, H_MAX)).x
            hh = h[s:e].copy(); hh[free] = sol
            h[s:e] = hh
            P = np.where(free_all[s:e, None], A[s:e] + Bv[s:e] * hh[:, None], xy[s:e])
            v = np.diff(P, axis=0) / np.diff(t)[:, None]
            dv = np.linalg.norm(np.diff(v, axis=0), axis=1)
            w = 1.0 / np.sqrt(1.0 + (dv / 6.0) ** 2)          # down-weight genuine kicks
        h[s:e] = hh
        xy[s:e] = P
    return xy, h


def best_path(cand):
    ks = sorted(cand)
    nodes, tick = [], []
    for k in ks:
        for row in cand[k]:
            nodes.append(row); tick.append(k)
    nodes, tick = np.array(nodes), np.array(tick)
    n = len(nodes)
    P0 = nodes[:, 5:7] + nodes[:, 7:9] * BALL_R      # sight line at ground level ...
    P1 = nodes[:, 5:7] + nodes[:, 7:9] * H_MAX       # ... and at the highest plausible flight
    score = np.full(n, -1e18); prev = np.full(n, -1)
    first = np.searchsorted(tick, ks)
    run_best, run_arg = -1e18, -1              # best node older than MAX_GAP (path may restart from it)
    older_ptr = 0
    RESTART = 1.2
    for a in range(n):
        k = tick[a]
        while older_ptr < n and tick[older_ptr] < k - MAX_GAP:
            if score[older_ptr] > run_best:
                run_best, run_arg = score[older_ptr], older_ptr
            older_ptr += 1
        own = 2.5 * nodes[a, 3] - 0.12       # reward confident hits, charge a little for each node
        best, arg = (run_best - RESTART, run_arg) if run_arg >= 0 else (0.0, -1)
        lo = np.searchsorted(tick, k - MAX_GAP); hi = np.searchsorted(tick, k)
        if hi > lo:
            g = (k - tick[lo:hi]) * DT
            dist = np.minimum(seg_dist(P0[a], P1[a], P0[lo:hi]), seg_dist(P0[lo:hi], P1[lo:hi], P0[a]))
            dist = np.minimum(dist, np.minimum(seg_dist(P0[a], P1[a], P1[lo:hi]), seg_dist(P0[lo:hi], P1[lo:hi], P1[a])))
            reach = 1.0 + V_BALL * g
            s = score[lo:hi] - 0.5 * dist / reach - 0.15 * (g / DT - 1)
            s = np.where(dist <= reach, s, -1e18)
            j = int(np.argmax(s))
            if s[j] > best:
                best, arg = s[j], lo + j
        score[a], prev[a] = own + best, arg
    path = []
    a = int(np.argmax(score))
    while a >= 0:
        path.append(a); a = prev[a]
    path = path[::-1]
    return tick[path], nodes[path]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="data/det")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--calib", default="config/calibration.json")
    ap.add_argument("--out", default="data/ball.npz")
    ap.add_argument("--tracks", default="data/tracks.npz", help="player tracks (optional): anchors the ball to the ground at players' feet")
    a = ap.parse_args()
    calib = json.load(open(a.calib))
    clips = scan(a.raw)
    chs = sorted(calib["cameras"])
    A, B = (load(a.det, clips, calib, ch) for ch in chs)
    cand = candidates(A, B, calib["pitch"])
    k, nodes = best_path(cand)
    anchor = None
    if os.path.exists(a.tracks):   # sightings whose on-the-ground position is at a player's feet
        from scipy.spatial import cKDTree
        T = np.load(a.tracks)
        d, _ = cKDTree(np.c_[T["xy"], T["k"] * 100.0]).query(np.c_[nodes[:, :2], k * 100.0], distance_upper_bound=1.3)
        anchor = np.isfinite(d)
        print(f"{anchor.mean() * 100:.0f}% of sightings are at a player's feet")
    xy, h = solve_heights(k, nodes, anchor=anchor)
    # whatever still darts out and straight back within one sample is a false sighting: drop it
    for _ in range(3):
        st = np.diff(xy, axis=0)
        near = (np.diff(k)[:-1] <= MAX_GAP) & (np.diff(k)[1:] <= MAX_GAP)
        la, lb = np.linalg.norm(st[:-1], axis=1), np.linalg.norm(st[1:], axis=1)
        spike = near & ((st[:-1] * st[1:]).sum(1) < -0.5 * la * lb) & (np.minimum(la, lb) > 2.0)
        if not spike.any():
            break
        keep = np.ones(len(k), bool); keep[1:-1] = ~spike
        k, nodes, xy, h = k[keep], nodes[keep], xy[keep], h[keep]
    moved = np.linalg.norm(xy - nodes[:, :2], axis=1)
    print(f"height solve: {(h > 0.5).mean() * 100:.0f}% of sightings airborne (>0.5 m); "
          f"{(moved > 1.0).mean() * 100:.0f}% moved by more than 1 m from their on-the-ground guess")
    np.savez_compressed(a.out, k=k, xy=xy, z=h, score=nodes[:, 3], views=nodes[:, 4].astype(int), xy_ground=nodes[:, :2])
    span = (max(cand) - min(cand) + 1)
    print(f"candidates on {len(cand)} of {span} ticks; ball path has {len(k)} points ({len(k) / span * 100:.0f}% of time), "
          f"{(nodes[:, 4] == 2).mean() * 100:.0f}% triangulated")


if __name__ == "__main__":
    main()
