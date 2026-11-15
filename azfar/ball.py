"""Stage 2b: find the single most plausible ball trajectory from both cameras, 15 times a second.

Two kinds of sightings are combined:
  * the object detector's "sports ball" hits (5 per second, precise, but it misses the ball often)
  * small moving blobs from azfar.ballscan (every frame, many false alarms)
Fixed false alarms (lamps, signs) are dropped, sightings from both cameras are triangulated when
their sight lines meet, and one path through the whole recording is chosen by dynamic programming
(confident sightings, physically possible moves). A single camera only gives a sight line, so the
height along it is then solved to make the path over the ground smooth (see solve_heights).

Usage: python -m azfar.ball --det data/det --scan data/ballscan --raw data/raw --out data/ball.npz
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, cv2
from .fuse import aim_cameras
from .timeline import scan, GRID_HZ, FPS

DT = 1.0 / FPS
STEP = int(round(FPS / GRID_HZ))   # frames per analysis tick
BALL_R = 0.11
V_BALL = 28.0        # m/s, hardest plausible shot
MAX_GAP = 30         # frames the ball may be unseen inside one path piece (2 s)
TOP = 8              # candidates kept per frame
H_MAX = 3.0          # highest the ball is assumed to fly, m
NODE_COST = 0.05     # small: faint but consistent sightings (ball in traffic, fast shots) must still win over skipping
W_MOTION = 0.12      # weight of a motion blob relative to detector confidence


def load(det_dir, scan_dir, clips, calib, ch, m_lo, m_hi):
    """All sightings of one camera as sight lines. Time index m = frames at 15 per second since midnight."""
    cams = aim_cameras(calib, ch)
    M, PX, SZ, CF, COL, EL, PR = [], [], [], [], [], [], []
    PK, PB = [], []
    for c in clips:
        if c.ch != ch:
            continue
        name = os.path.basename(c.path).replace(".mp4", ".npz")
        p = os.path.join(det_dir, name)
        if os.path.exists(p):
            d = np.load(p)
            sel = d["cls"] == 32
            b = d["box"][sel]
            pp = (d["cls"] == 0) & (d["conf"] >= 0.3)
            PK.append(d["k"][pp]); PB.append(d["box"][pp])
            M.append(d["k"][sel] * STEP); PX.append(np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2])
            SZ.append(np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])); CF.append(d["conf"][sel])
            ps = os.path.join(scan_dir, name) if scan_dir else ""
            dc = np.load(ps)["det_col"] if ps and os.path.exists(ps) and "det_col" in np.load(ps).files else None
            COL.append(dc[sel] if dc is not None else np.full((sel.sum(), 3), np.nan, np.float32))
            EL.append(np.ones(sel.sum(), np.float32))
            pp_ = os.path.join(os.path.dirname(scan_dir.rstrip("/")) or ".", "ballpatch", name.replace(".npz", "_pdet.npy")) if scan_dir else ""
            PR.append(np.load(pp_) if pp_ and os.path.exists(pp_) else np.full(sel.sum(), np.nan, np.float32))
        p = os.path.join(scan_dir, name) if scan_dir else ""
        if p and os.path.exists(p):
            d = np.load(p)
            COL.append(d["col"] if "col" in d.files else np.full((len(d["f"]), 3), np.nan, np.float32))
            EL.append(d["elong"] if "elong" in d.files else np.ones(len(d["f"]), np.float32))
            pp_ = os.path.join(os.path.dirname(scan_dir.rstrip("/")) or ".", "ballpatch", name.replace(".npz", "_pblob.npy"))
            PR.append(np.load(pp_) if os.path.exists(pp_) else np.full(len(d["f"]), np.nan, np.float32))
            M.append(np.round(c.t0 * FPS).astype(np.int64) + d["f"].astype(np.int64)); PX.append(d["xy"].astype(np.float64))
            SZ.append(d["size"]); CF.append(-(d["bright"] / 255.0))      # negative marks a motion blob (brightness kept)
    det_boxes = None
    if PK:
        pk, pb = np.concatenate(PK), np.concatenate(PB)
        o = np.argsort(pk, kind="stable")
        det_boxes = (pk[o], pb[o])
    M, PX, SZ, CF, COL, EL, PR = (np.concatenate(x) for x in (M, PX, SZ, CF, COL, EL, PR))
    keep = (M >= m_lo) & (M < m_hi)
    M, PX, SZ, CF, COL, EL, PR = M[keep], PX[keep], SZ[keep], CF[keep], COL[keep], EL[keep], PR[keep]
    mot = CF < 0
    n0 = mot.sum()
    drop = np.zeros(len(M), bool)
    # 0) colour: the match ball is white/cream. Neon boots, coloured socks and dark shoes are not.
    sat, val, yel = COL[:, 0], COL[:, 1], COL[:, 2]
    known = ~np.isnan(sat) & (val > 0)
    streak = EL > 1.8          # a fast ball blurs into the grass: more colour, less brightness
    not_ball = known & ((sat > np.where(streak, 115, 88)) | ((yel > 0.22) & (sat > 65)) | (val < np.where(streak, 95, 110)))
    scored = np.isfinite(PR)
    # where the learnt recogniser has scored a candidate, it decides (it has seen this ball and these boots)
    yellow = known & (yel > 0.22) & (sat > 65)    # neon boots fool everything else: always out
    not_ball = np.where(scored, np.where(CF < 0, PR < 0.6, (PR < 0.15) & (CF < 0.5)), not_ball) | yellow
    known = known | scored
    drop |= not_ball
    print(f"  {ch}: {not_ball[~mot].sum()} of {(~mot).sum()} detector hits and {not_ball[mot].sum()} of {mot.sum()} motion sightings rejected by colour")
    # 1) goal frames and nets: the net flutters and looks like a small moving object
    cam0 = cams[0][2]
    L, W = calib["pitch"]["L"], calib["pitch"]["W"]
    for gx0, gx1 in ((-1.3, 0.15), (L - 0.15, L + 1.3)):
        corners = np.array([(x, y, -z) for x in (gx0, gx1) for y in (W / 2 - 1.8, W / 2 + 1.8) for z in (0.0, 2.2)])
        px = cam0.project(corners)
        if not np.isfinite(px).all():
            continue
        hull = cv2.convexHull(px.astype(np.float32))
        inside = np.array([cv2.pointPolygonTest(hull, (float(x), float(y)), True) for x, y in PX[mot]]) > -8
        drop[np.flatnonzero(mot)[inside]] = True
    # 2) flicker: the same spot firing in a large share of frames (nets, bench, shadows, hair)
    cell = (PX[:, 0] // 16).astype(np.int64) * 100 + (PX[:, 1] // 16).astype(np.int64)
    for off in (0, 22):
        key = cell * 10**7 + (M + off) // 45
        u, inv, cnt = np.unique(key[mot], return_inverse=True, return_counts=True)
        drop[np.flatnonzero(mot)[cnt[inv] > 18]] = True
    # 3) moving boots and limbs just outside a player's box
    near = np.zeros(len(M), bool)
    if det_boxes is not None:
        pk, pb = det_boxes
        tk = np.round(M / STEP).astype(np.int64)
        for dk in (-1, 0, 1):
            lo, hi = np.searchsorted(pk, tk + dk), np.searchsorted(pk, tk + dk + 1)
            for i in np.flatnonzero(mot & ~drop & ~near & (hi > lo)):
                b = pb[lo[i]:hi[i]]
                w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
                x, y = PX[i]
                if ((x > b[:, 0] - 0.35 * w - 4) & (x < b[:, 2] + 0.35 * w + 4) & (y > b[:, 1]) & (y < b[:, 3] + 0.12 * h + 4)).any():
                    near[i] = True
    # next to a player a moving boot looks like a ball: there a motion sighting must also pass the colour
    # test and counts for less; without colour information it is not used at all
    print(f"  {ch}: {(near & ~drop).sum()} motion sightings next to players kept at reduced weight")
    drop |= near & ~known
    CF = np.where(near & ~scored, CF * 0.5, CF)
    M, PX, SZ, CF, PR = M[~drop], PX[~drop], SZ[~drop], CF[~drop], PR[~drop]
    # fixed false alarms: the same image cell firing again and again
    cell = (PX[:, 0] // 14).astype(int) * 1000 + (PX[:, 1] // 14).astype(int)
    u, inv, cnt = np.unique(cell, return_inverse=True, return_counts=True)
    keep = cnt[inv] < max(400, 0.012 * (m_hi - m_lo))
    M, PX, SZ, CF, PR = M[keep], PX[keep], SZ[keep], CF[keep], PR[keep]
    O = np.zeros((len(M), 3)); D = np.zeros((len(M), 3)); G = np.zeros((len(M), 2)); ok = np.zeros(len(M), bool)
    rng = np.zeros(len(M))
    t = M / FPS
    ci = np.zeros(len(M), int)
    for wi, (t0, t1, cam) in enumerate(cams):
        ci[t >= t0] = wi
    f = cams[0][2].f
    for wi, (t0, t1, cam) in enumerate(cams):
        m = ci == wi
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
    expect = f * 2 * BALL_R / np.maximum(rng, 1.0)
    ratio = SZ / np.maximum(expect, 1e-3)
    good = ok & (ratio > 0.45) & (ratio < 6.0)              # bigger than expected = closer = airborne, allowed
    # fixed false alarms off the pitch (wall prints, lamps, signs): the same pixel firing again and again,
    # whose ground point is outside the lines. (On the pitch a resting ball - the centre spot - may repeat.)
    L_, W_ = calib["pitch"]["L"], calib["pitch"]["W"]
    off = (G[:, 0] < -0.15) | (G[:, 0] > L_ + 0.15) | (G[:, 1] < -0.15) | (G[:, 1] > W_ + 0.15)
    cell = (PX[:, 0] // 12).astype(np.int64) * 1000 + (PX[:, 1] // 12).astype(np.int64)
    u, inv, cnt = np.unique(cell, return_inverse=True, return_counts=True)
    good &= ~(off & (cnt[inv] > 30))
    # a ball the detector finds INSIDE a player's outline, above the feet, is in their hands (keeper,
    # throw-in) or against their body: its height is roughly where it sits in that outline
    hfix = np.full(len(M), np.nan)
    if det_boxes is not None:
        pk, pb = det_boxes
        tk = np.round(M / STEP).astype(np.int64)
        lo_, hi_ = np.searchsorted(pk, tk), np.searchsorted(pk, tk + 1)
        sure = (CF >= 0.6) & (np.nan_to_num(PR, nan=0.0) >= 0.85)   # shirts, knees and boots in front of a body fool weaker hits
        for i in np.flatnonzero(sure & good & (hi_ > lo_)):
            b = pb[lo_[i]:hi_[i]]
            x, y = PX[i]
            bh = b[:, 3] - b[:, 1]
            inside = (x > b[:, 0]) & (x < b[:, 2]) & (y > b[:, 1] + 0.08 * bh) & (y < b[:, 3] - 0.18 * bh)
            if inside.any():
                j = int(np.argmax(inside))
                # on the ground it would sit at the player's feet; held up high, its ground reading lands
                # metres behind them. Only the second case is a held ball.
                cam_ = cams[ci[i]][2]
                foot, okf = cam_.to_ground(np.array([[(b[j, 0] + b[j, 2]) / 2, b[j, 3]]]))
                if okf[0] and np.linalg.norm(G[i] - foot[0]) > 1.5:
                    hfix[i] = float(np.clip(1.75 * (b[j, 3] - y) / bh[j], 0.3, 2.3))
    print(f"  {ch}: {np.isfinite(hfix).sum()} detector sightings of the ball held or against a body")
    w = np.where(CF >= 0, CF, W_MOTION * (0.5 + 0.5 * -CF) * np.exp(-0.5 * np.log(np.maximum(ratio, 0.2) / 1.2) ** 2))
    # recogniser score, where available: motion blobs it is sure of count almost like a detector hit
    sizefit = np.exp(-0.5 * np.log(np.maximum(ratio, 0.2) / 1.2) ** 2)
    w = np.where(np.isfinite(PR), np.where(CF >= 0, CF * (0.4 + 0.6 * np.nan_to_num(PR)), 0.25 * np.nan_to_num(PR) * sizefit), w)
    w = w * np.where(rng < 24, 1.0, 0.6)                    # trust a camera most in its own half
    return dict(m=M[good], w=w[good], det=(CF >= 0)[good], O=O[good], D=D[good], g=G[good], hfix=hfix[good])


def triangulate_all(Oa, Da, Ob, Db):
    """Closest approach of every sight line in a with every one in b. Returns midpoints [na, nb, 3], miss [na, nb]."""
    w = Oa[:, None] - Ob[None]
    b = Da @ Db.T
    d = (Da[:, None] * w).sum(-1); e = (Db[None] * w).sum(-1)
    den = 1.0 - b * b
    den = np.where(np.abs(den) < 1e-9, 1e-9, den)
    s = (b * e - d) / den; t = (e - b * d) / den
    pa = Oa[:, None] + s[..., None] * Da[:, None]; pb = Ob[None] + t[..., None] * Db[None]
    miss = np.linalg.norm(pa - pb, axis=-1)
    miss = np.where((s > 0) & (t > 0), miss, 1e9)
    return (pa + pb) / 2, miss


def candidates(A, B, pitch):
    """Per frame: rows (x, y, h, weight, views, Ax, Ay, Bx, By). A single-camera sighting can be anywhere
    on its sight line: XY(h) = (Ax, Ay) + (Bx, By) * h."""
    L, W = pitch["L"], pitch["W"]
    ms = np.union1d(A["m"], B["m"])
    sa, sb = np.argsort(A["m"], kind="stable"), np.argsort(B["m"], kind="stable")
    ba = np.searchsorted(A["m"][sa], [ms, ms + 1]); bb = np.searchsorted(B["m"][sb], [ms, ms + 1])
    out = {}
    for n, m in enumerate(ms):
        ia, ib = sa[ba[0][n]:ba[1][n]], sb[bb[0][n]:bb[1][n]]
        rows = []
        used_a, used_b = np.zeros(len(ia), bool), np.zeros(len(ib), bool)
        if len(ia) and len(ib):
            P, miss = triangulate_all(A["O"][ia], A["D"][ia], B["O"][ib], B["D"][ib])
            h = -P[..., 2]
            both_motion = (~A["det"][ia])[:, None] & (~B["det"][ib])[None]
            good = (miss < np.where(both_motion, 0.3, 0.6)) & (h > -0.3) & (h < np.where(both_motion, 3.0, 6.0)) & (P[..., 0] > -2) & (P[..., 0] < L + 2) & (P[..., 1] > -2) & (P[..., 1] < W + 2)
            for i, j in zip(*np.nonzero(good)):
                wa, wb = A["w"][ia[i]], B["w"][ib[j]]
                # two motion blobs agreeing is weak evidence; a detector hit on either side is strong
                bonus = 0.3 if (A["det"][ia[i]] or B["det"][ib[j]]) else 0.08
                rows.append((P[i, j, 0], P[i, j, 1], max(h[i, j], BALL_R), wa + wb + bonus, 2, P[i, j, 0], P[i, j, 1], 0.0, 0.0))
                used_a[i] = used_b[j] = True
        for V, idx, used in ((A, ia, used_a), (B, ib, used_b)):
            for i, u in zip(idx, used):
                if u and not V["det"][i]:
                    continue
                O_, D_ = V["O"][i], V["D"][i]
                A_ = O_[:2] - D_[:2] * O_[2] / D_[2]
                Bv = -D_[:2] / D_[2]
                p0, p1 = A_ + Bv * BALL_R, A_ + Bv * H_MAX
                if not ((-3 < p0[0] < L + 3 and -3 < p0[1] < W + 3) or (-3 < p1[0] < L + 3 and -3 < p1[1] < W + 3)):
                    continue
                if np.isfinite(V["hfix"][i]):   # held: fixed point at that height on the sight line
                    p = A_ + Bv * V["hfix"][i]
                    rows.append((p[0], p[1], V["hfix"][i], V["w"][i] + 0.2, 3, p[0], p[1], 0.0, 0.0))
                    continue
                rows.append((V["g"][i, 0], V["g"][i, 1], np.nan, V["w"][i], 1, *A_, *Bv))
        if rows:
            rows.sort(key=lambda r: -r[3])
            out[int(m)] = np.array(rows[:TOP])
    return out


def seg_dist(a, b, p):
    """Distance from point(s) p to segment(s) a-b (2-D, broadcasting)."""
    ab = b - a
    t = np.clip(((p - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9), 0, 1)
    return np.linalg.norm(a + ab * t[..., None] - p, axis=-1)


def best_path(cand):
    ms = sorted(cand)
    nodes = np.concatenate([cand[m] for m in ms])
    tick = np.concatenate([np.full(len(cand[m]), m) for m in ms])
    n = len(nodes)
    P0 = nodes[:, 5:7] + nodes[:, 7:9] * BALL_R      # sight line at ground level ...
    P1 = nodes[:, 5:7] + nodes[:, 7:9] * H_MAX       # ... and at the highest plausible flight
    score = np.full(n, -1e18); prev = np.full(n, -1)
    run_best, run_arg, older = -1e18, -1, 0          # best node older than MAX_GAP (the path may restart from it)
    RESTART = 1.5
    lo_all = np.searchsorted(tick, tick - MAX_GAP); hi_all = np.searchsorted(tick, tick)
    for a in range(n):
        m = tick[a]
        while older < n and tick[older] < m - MAX_GAP:
            if score[older] > run_best:
                run_best, run_arg = score[older], older
            older += 1
        own = 2.5 * nodes[a, 3] - NODE_COST      # reward confident sightings, charge a little for each point
        best, arg = (run_best - RESTART, run_arg) if run_arg >= 0 else (0.0, -1)
        lo, hi = lo_all[a], hi_all[a]
        if hi > lo:
            g = (m - tick[lo:hi]) * DT
            dist = np.minimum(seg_dist(P0[a], P1[a], P0[lo:hi]), seg_dist(P0[lo:hi], P1[lo:hi], P0[a]))
            dist = np.minimum(dist, np.minimum(seg_dist(P0[a], P1[a], P1[lo:hi]), seg_dist(P0[lo:hi], P1[lo:hi], P1[a])))
            reach = 0.3 + V_BALL * g
            gd = np.linalg.norm(P0[lo:hi] - P0[a], axis=1)          # same move if the ball stayed on the ground
            cost = 0.5 * dist + 0.5 * np.minimum(gd, dist + 3.0)    # leaving the ground must be earned
            # a ball keeps going: what matters is how far it strays from where its last velocity
            # would take it, not how fast it goes (otherwise hard shots lose to slow false sightings)
            pj = prev[lo:hi]
            has = pj >= 0
            pjc = np.maximum(pj, 0)
            gp = (tick[lo:hi] - tick[pjc]) * DT
            vj = (P0[lo:hi] - P0[pjc]) / np.maximum(gp, DT)[:, None]
            dev = np.linalg.norm(P0[lo:hi] + vj * g[:, None] - P0[a], axis=1)
            cost = np.where(has & (gp <= 6 * DT), np.minimum(dev, cost), cost)
            s = score[lo:hi] - 0.02 * cost / g - 0.02 * (g / DT - 1)  # slower hops and shorter gaps preferred
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


def solve_heights(k, nodes, lam=3.0, rounds=6, anchor=None, chunk=300):
    """Give every single-camera sighting a height so the ball's path over the ground is smooth.

    Each sighting can slide along its sight line: XY = A + Bv * h, h >= ball radius. Within a
    continuous piece of the path we minimise changes of ground velocity (robustly, so a real kick
    is still a kink) plus a pull to the ground - strong where the ball is at a player's feet."""
    from scipy.optimize import lsq_linear
    from scipy.sparse import csr_matrix
    xy = nodes[:, :2].copy()
    h = np.where(nodes[:, 4] >= 2, nodes[:, 2], BALL_R)
    A, Bv = nodes[:, 5:7], nodes[:, 7:9]
    free_all = nodes[:, 4] == 1
    cuts = np.flatnonzero(np.diff(k) > MAX_GAP) + 1
    pieces = []
    for s, e in zip(np.r_[0, cuts], np.r_[cuts, len(k)]):
        for c0 in range(s, e, chunk):
            pieces.append((max(c0 - 2, s), min(c0 + chunk, e)))   # two points of overlap keep chunks continuous
    for s, e in pieces:
        n = e - s
        free = np.flatnonzero(free_all[s:e])
        if n < 3 or not len(free):
            continue
        col = -np.ones(n, int); col[free] = np.arange(len(free))
        t = k[s:e] * DT
        d0, d1 = (t[1:-1] - t[:-2]), (t[2:] - t[1:-1])
        coef = np.stack([1 / d0, -1 / d0 - 1 / d1, 1 / d1], 1)        # velocity change at each inner point
        w = np.ones(n - 2)
        hh = h[s:e].copy()
        for _ in range(rounds):
            R, Cc, V = [], [], []
            y = np.zeros((n - 2) * 2 + len(free))
            for off in range(3):
                j = np.arange(n - 2) + off
                cf = w * coef[:, off]
                fixed = col[j] < 0
                for ax in (0, 1):
                    r = np.arange(n - 2) * 2 + ax
                    y[r] -= cf * np.where(fixed, xy[s + j, ax], A[s + j, ax])
                    R.append(r[~fixed]); Cc.append(col[j][~fixed]); V.append((cf * Bv[s + j, ax])[~fixed])
            base = (n - 2) * 2
            pw = lam / np.sqrt(np.maximum(hh[free] - BALL_R, 0) + 0.25)   # sparse: on the ground unless needed
            if anchor is not None:
                pw = pw * np.where(anchor[s:e][free], 10.0, 1.0)
            R.append(base + np.arange(len(free))); Cc.append(np.arange(len(free))); V.append(pw)
            y[base:] = pw * BALL_R
            M = csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(Cc))), shape=(len(y), len(free)))
            hh[free] = lsq_linear(M, y, bounds=(BALL_R, H_MAX), lsq_solver='lsmr', tol=1e-6).x
            P = np.where(free_all[s:e, None], A[s:e] + Bv[s:e] * hh[:, None], xy[s:e])
            v = np.diff(P, axis=0) / np.diff(t)[:, None]
            dv = np.linalg.norm(np.diff(v, axis=0), axis=1)
            w = 1.0 / np.sqrt(1.0 + (dv / 8.0) ** 2)          # down-weight genuine kicks
        h[s:e] = hh
        xy[s:e] = P
    return xy, h


def carry_with_players(K, XY, Z, tracks, max_gap=40):
    """Fill gaps where the ball is hidden at someone's feet: if a player is on the ball when it
    disappears, the ball goes with that player until it is seen again (blending into the next
    sighting over the last half second). Other gaps are left as gaps."""
    from scipy.spatial import cKDTree
    T = np.load(tracks)
    tk, txy = T["k"], T["xy"]
    o = np.argsort(tk, kind="stable"); tk, txy = tk[o], txy[o]
    lo_all = np.searchsorted(tk, np.arange(K.min(), K.max() + 2))
    def players(k):
        i = k - K.min()
        return txy[lo_all[i]:lo_all[i + 1]] if 0 <= i < len(lo_all) - 1 else np.zeros((0, 2))
    newK, newXY, newZ, carried = [K[:1]], [XY[:1]], [Z[:1]], [np.zeros(1, bool)]
    filled = 0
    for a in range(len(K) - 1):
        ka, kb = K[a], K[a + 1]
        if 1 < kb - ka <= max_gap:
            P = players(ka)
            d = np.linalg.norm(P - XY[a], axis=1) if len(P) else np.array([])
            if len(d) and d.min() < 1.5:
                pos = P[d.argmin()]
                ks, ps = [], []
                ok = True
                for k in range(ka + 1, kb):
                    Q = players(k)
                    if not len(Q):
                        ok = False; break
                    dd = np.linalg.norm(Q - pos, axis=1)
                    if dd.min() > 1.6:      # lost the player: give up rather than guess
                        ok = False; break
                    pos = Q[dd.argmin()]
                    ks.append(k); ps.append(pos.copy())
                if ok and ks and np.linalg.norm(ps[-1] - XY[a + 1]) < 3.0 + 0.5 * (kb - ka) * 0.2 * 8:
                    ps = np.array(ps)
                    nb = min(3, len(ks))                # blend into the next sighting
                    w = np.linspace(0, 1, nb + 2)[1:-1]
                    ps[-nb:] = ps[-nb:] * (1 - w[:, None]) + XY[a + 1] * w[:, None]
                    newK.append(np.array(ks)); newXY.append(ps); newZ.append(np.full(len(ks), BALL_R)); carried.append(np.ones(len(ks), bool))
                    filled += len(ks)
        newK.append(K[a + 1:a + 2]); newXY.append(XY[a + 1:a + 2]); newZ.append(Z[a + 1:a + 2]); carried.append(np.zeros(1, bool))
    print(f"carried with the player in possession on {filled} ticks ({filled / GRID_HZ / 60:.1f} min)")
    return np.concatenate(newK), np.concatenate(newXY), np.concatenate(newZ), np.concatenate(carried)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="data/det")
    ap.add_argument("--scan", default="data/ballscan", help="motion blobs from azfar.ballscan ('' to use the detector only)")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--calib", default="config/calibration.json")
    ap.add_argument("--session", default="config/session.json")
    ap.add_argument("--out", default="data/ball.npz")
    ap.add_argument("--tracks", default="data/tracks.npz", help="player tracks (optional): anchors the ball to the ground at players' feet")
    a = ap.parse_args()
    calib = json.load(open(a.calib))
    clips = scan(a.raw)
    hms = lambda s: sum(int(p) * m for p, m in zip(s.split(":"), (3600, 60, 1)))
    sess = json.load(open(a.session)) if os.path.exists(a.session) else {}
    mt = (sess.get("matches") or [{}])[0]
    m_lo, m_hi = int(hms(mt.get("start", "00:00:00")) * FPS), int(hms(mt.get("end", "23:59:59")) * FPS)
    chs = sorted(calib["cameras"])
    A, B = (load(a.det, a.scan, clips, calib, ch, m_lo, m_hi) for ch in chs)
    print(f"sightings: {chs[0]} {len(A['m'])} ({A['det'].sum()} from the detector), {chs[1]} {len(B['m'])} ({B['det'].sum()})")
    cand = candidates(A, B, calib["pitch"])
    m, nodes = best_path(cand)
    anchor = None
    if os.path.exists(a.tracks):   # sightings whose on-the-ground position is at a player's feet
        from scipy.spatial import cKDTree
        T = np.load(a.tracks)
        d, _ = cKDTree(np.c_[T["xy"], T["k"] * 2.0]).query(np.c_[nodes[:, :2], m / STEP * 2.0], distance_upper_bound=1.5)
        anchor = np.isfinite(d)
    xy, h = solve_heights(m, nodes, anchor=anchor)
    # whatever still darts out and straight back within one sample is a false sighting: drop it
    n0 = len(m)
    for _ in range(4):
        st = np.diff(xy, axis=0)
        near = (np.diff(m)[:-1] <= MAX_GAP) & (np.diff(m)[1:] <= MAX_GAP)
        la, lb = np.linalg.norm(st[:-1], axis=1), np.linalg.norm(st[1:], axis=1)
        back = np.linalg.norm(xy[2:] - xy[:-2], axis=1) < 0.6 * np.minimum(la, lb)   # returns to where it was:
        spike = near & back & ((st[:-1] * st[1:]).sum(1) < -0.5 * la * lb) & (np.minimum(la, lb) > 1.5)  # not a rebound
        if not spike.any():
            break
        keep = np.ones(len(m), bool); keep[1:-1] = ~spike
        m, nodes, xy, h = m[keep], nodes[keep], xy[keep], h[keep]
    # physics: no sighting may need more than V_BALL to reach both of its neighbours
    for _ in range(6):
        dm = np.diff(m) * DT
        sp = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.maximum(dm, DT)
        bad_in = np.r_[False, (sp > V_BALL) & (np.diff(m) <= MAX_GAP)]
        bad_out = np.r_[(sp > V_BALL) & (np.diff(m) <= MAX_GAP), False]
        drop = bad_in & bad_out
        if not drop.any():
            break
        m, nodes, xy, h = m[~drop], nodes[~drop], xy[~drop], h[~drop]
    # anything still too fast is a hand-over between two different objects: cut the path there
    sp = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.maximum(np.diff(m) * DT, DT)
    breaks = np.flatnonzero(sp > V_BALL) + 1
    # light smoothing (median of three) removes single-frame flicker without rounding off kicks
    from scipy.ndimage import median_filter
    cuts = np.union1d(np.flatnonzero(np.diff(m) > MAX_GAP) + 1, breaks)
    for s_, e_ in zip(np.r_[0, cuts], np.r_[cuts, len(m)]):
        if e_ - s_ >= 3:
            xy[s_:e_] = median_filter(xy[s_:e_], size=(3, 1), mode="nearest")
    print(f"physics check: {n0 - len(m)} sightings removed in total, path cut at {len(breaks)} impossible jumps")
    # resample to the 5-per-second analysis ticks for the stats stages (gaps over 2 s stay gaps)
    K, XY, Z = [], [], []
    brk = set(breaks.tolist())
    for s, e in zip(np.r_[0, cuts], np.r_[cuts, len(m)]):
        if e - s < 3:
            continue
        first = int(np.ceil(m[s] / STEP)) + (1 if s in brk else 0)   # after a cut, hide the ball for an instant
        kk = np.arange(first, int(np.floor(m[e - 1] / STEP)) + 1)
        if not len(kk):
            continue
        K.append(kk); Z.append(np.interp(kk * STEP, m[s:e], h[s:e]))
        XY.append(np.c_[np.interp(kk * STEP, m[s:e], xy[s:e, 0]), np.interp(kk * STEP, m[s:e], xy[s:e, 1])])
    K, XY, Z = np.concatenate(K), np.concatenate(XY), np.concatenate(Z)
    carried = np.zeros(len(K), bool)
    if os.path.exists(a.tracks):
        K, XY, Z, carried = carry_with_players(K, XY, Z, a.tracks)
    np.savez_compressed(a.out, k=K, xy=XY, z=Z, carried=carried, m15=m, xy15=xy, z15=h, views15=nodes[:, 4].astype(int), w15=nodes[:, 3])
    span = m_hi - m_lo if m_hi - m_lo < 10**7 else (max(cand) - min(cand) + 1)
    print(f"path: {len(m)} sightings on {len(cand)} frames with candidates ({n0 - len(m)} dropped as spikes); "
          f"{(nodes[:, 4] == 2).mean() * 100:.0f}% triangulated, {(h > 0.5).mean() * 100:.0f}% airborne")
    print(f"ball located on {len(K) / (span / STEP) * 100:.0f}% of analysis ticks in the match window")


if __name__ == "__main__":
    main()
