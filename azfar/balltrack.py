"""Conservative ball tracking on the dedicated detector's every-frame output.

Rules (per camera, in the image, 15 frames per second):
  * show nothing unless sure: a track is only shown once it has three sightings within
    CONFIRM_WINDOW frames that move consistently
  * once sure, stay locked: while a track lives, every other ball-like detection is ignored. It continues
    with the sighting nearest to where its own motion predicts the ball (fast shots follow that prediction);
    a sudden change of direction (a kick) is accepted only from a confident sighting close to the ball
  * lose it, stop showing it: after MAX_MISS frames without a sighting the track ends; gaps of up to
    FILL_GAP frames between two sightings of the same track are bridged in a straight line, nothing else
Both cameras' confirmed tracks are then put on the pitch: triangulated when both see it, otherwise from the
one camera (with the ball's apparent size telling how far along the line of sight it is). When the cameras
disagree, the one that has been locked on longer wins; the source does not flip back and forth.

Usage: python -m azfar.balltrack   (writes data/ball.npz)
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from .camera import Camera
from .fuse import aim_cameras
from .timeline import scan, FPS, GRID_HZ

STEP = int(round(FPS / GRID_HZ))
MIN_CONF = 0.10          # weak sightings are only ever used to continue a locked track, inside its gate
TRACK_CONF = 0.25
START_CONF = 0.5
CONFIRM_WINDOW = 6
MAX_MISS = 8           # frames (0.53 s)
FILL_GAP = 6           # frames (0.4 s)
KICK_CONF = 0.6
KICK_CONFIRM = 2       # sightings needed to confirm a change of direction
MAX_SPEED = 30.0       # m/s
BALL_D = 0.22
HMS = lambda s: sum(int(p) * m for p, m in zip(s.split(":"), (3600, 60, 1)))


def static_cells(F, XY, cell=4, sec=60):
    k = (XY[:, 0] // cell).astype(np.int64) * 1000 + (XY[:, 1] // cell).astype(np.int64)
    u, n = np.unique(np.unique(k * 10**7 + F // int(FPS)) // 10**7, return_counts=True)
    busy = set(u[n >= sec].tolist())
    near = np.zeros(len(F), bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            near |= np.isin(k + dx * 1000 + dy, list(busy))
    return near


def max_displacement(size1, size2, dt):
    """Max projected travel over dt frames, using ball diameter as the local pixel scale."""
    size = np.maximum(np.minimum(size1, size2), 3.0)
    return size * (MAX_SPEED / (BALL_D * FPS) * dt) + np.maximum(size, 4.0)


def acquire(F, XY, SZ, CF, first, start, stop):
    """Find a three-hit constant-velocity seed ending in this frame, within CONFIRM_WINDOW."""
    prior = np.flatnonzero(CF[first:start] >= TRACK_CONF) + first
    current = np.flatnonzero(CF[start:stop] >= TRACK_CONF) + start
    if (len(prior) < 2 or not len(current)
            or not np.any(CF[first:stop] >= START_CONF)):
        return None
    # Only a pair from distinct frames defines a velocity. Keep competing pairs
    # until a third sighting supports the projected motion.
    a, b = np.triu_indices(len(prior), 1)
    a, b = prior[a], prior[b]
    dt = F[b] - F[a]
    distance = np.linalg.norm(XY[b] - XY[a], axis=1)
    ok = ((dt > 0) & (distance <= max_displacement(SZ[a], SZ[b], dt))
          & (np.maximum(SZ[a], SZ[b]) <= 2 * np.minimum(SZ[a], SZ[b]) + 1))
    a, b, dt = a[ok], b[ok], dt[ok]
    if not len(a):
        return None
    velocity = (XY[b] - XY[a]) / dt[:, None]
    elapsed = F[start] - F[b]
    prediction = XY[b] + velocity * elapsed[:, None]
    # Allow for centroid noise and modest perspective/acceleration changes.
    gate = np.maximum(SZ[b], 4.0) + 0.25 * np.linalg.norm(velocity, axis=1) * elapsed
    best, best_score = None, -np.inf
    for c in current:
        error = np.linalg.norm(XY[c] - prediction, axis=1)
        travel = np.linalg.norm(XY[c] - XY[b], axis=1)
        ok = ((np.maximum(np.maximum(CF[a], CF[b]), CF[c]) >= START_CONF)
              & (error <= gate) & (travel <= max_displacement(SZ[b], SZ[c], elapsed))
              & (np.maximum(SZ[b], SZ[c]) <= 2 * np.minimum(SZ[b], SZ[c]) + 1))
        if ok.any():
            score = (CF[a] + CF[b] + CF[c]) / 3 - error / gate
            i = np.argmax(np.where(ok, score, -np.inf))
            if score[i] > best_score:
                best_score = score[i]
                best = [a[i], b[i], c]
    return best


def finish_track(track, tracks):
    # A provisional kick at timeout or EOF never becomes a published sighting.
    if track.get("probation", 0):
        for key in ("f", "xy", "sz", "cf"):
            del track[key][track["prob_start"]:]
    tracks.append(track)


def track_camera(F, XY, SZ, CF):
    """F: global frame index (15/s). Returns list of tracks: arrays (frame, x, y, size, conf, filled)."""
    if len(F) == 0:
        return []
    o = np.argsort(F, kind="stable")
    F, XY, SZ, CF = F[o].astype(np.int64), XY[o].astype(float), SZ[o], CF[o]
    frames, starts = np.unique(F, return_index=True)
    tracks, cur = [], None
    earliest = frames[0]
    for f, start, stop in zip(frames, starts, np.r_[starts[1:], len(F)]):
        if cur is not None:
            dt = f - cur["f"][-1]
            if dt > MAX_MISS or (cur.get("probation", 0) and dt > 3):
                finish_track(cur, tracks)
                earliest = cur["f"][-1] + 1
                cur = None
        if cur is None:
            first = np.searchsorted(F, max(earliest, f - CONFIRM_WINDOW))
            hits = acquire(F, XY, SZ, CF, first, start, stop)
            if hits is not None:
                a, _, c = hits
                cur = dict(v=(XY[c] - XY[a]) / (F[c] - F[a]), s=float(SZ[c]),
                           f=list(F[hits]), xy=list(XY[hits]), sz=list(SZ[hits]), cf=list(CF[hits]))
            continue
        xy, sz, cf = XY[start:stop], SZ[start:stop], CF[start:stop]
        p, v, s = cur["xy"][-1], cur["v"], cur["s"]
        dt = f - cur["f"][-1]
        dp = np.linalg.norm(xy - (p + v * dt), axis=1)
        dk = np.linalg.norm(xy - p, axis=1)
        gate = max(1.2 * s, 10.0) + 0.35 * np.linalg.norm(v) * dt + 0.4 * max(s, 3) / BALL_D * (dt - 1)
        reachable = dk <= max_displacement(s, sz, dt)
        follows = (cf >= MIN_CONF) & (dp <= gate) & reachable
        if follows.any():
            j = int(np.argmin(np.where(follows, dp / np.maximum(cf, 0.1), np.inf)))
            vn = (xy[j] - p) / dt
            cur["v"] = 0.6 * vn + 0.4 * v
            if cur.get("probation", 0):
                cur["probation"] -= 1
        else:
            same_size = np.maximum(sz, s) <= 2 * np.minimum(sz, s) + 1
            kick = (cf >= KICK_CONF) & reachable & same_size
            if cur.get("probation", 0) or not kick.any():
                continue
            j = int(np.argmin(np.where(kick, dk, np.inf)))
            cur["probation"] = KICK_CONFIRM
            cur["prob_start"] = len(cur["f"])
            cur["v"] = (xy[j] - p) / dt
        cur["s"] = 0.7 * s + 0.3 * float(sz[j])
        cur["f"].append(f)
        cur["xy"].append(xy[j])
        cur["sz"].append(float(sz[j]))
        cur["cf"].append(float(cf[j]))
    if cur is not None:
        finish_track(cur, tracks)
    out = []
    for t in tracks:
        f, xy = np.array(t["f"]), np.array(t["xy"])
        sz, cf = np.array(t["sz"]), np.array(t["cf"])
        # parked spare ball: a track that never moves for 8+ s is not the ball in play (the centre spot is
        # handled by the caller)
        if f[-1] - f[0] >= 8 * FPS and np.percentile(np.linalg.norm(xy - np.median(xy, 0), axis=1), 95) < max(np.median(sz), 6):
            out.append(("parked", f, xy, sz, cf)); continue
        # bridge short gaps, nothing longer
        ff, X, Y, S, C, FL = [], [], [], [], [], []
        for i in range(len(f)):
            if i and 1 < f[i] - f[i - 1] <= FILL_GAP:
                for g in range(f[i - 1] + 1, f[i]):
                    w = (g - f[i - 1]) / (f[i] - f[i - 1])
                    ff.append(g); X.append(xy[i - 1, 0] * (1 - w) + xy[i, 0] * w); Y.append(xy[i - 1, 1] * (1 - w) + xy[i, 1] * w)
                    S.append(sz[i]); C.append(0.0); FL.append(True)
            ff.append(f[i]); X.append(xy[i, 0]); Y.append(xy[i, 1]); S.append(sz[i]); C.append(cf[i]); FL.append(False)
        out.append(("ball", np.array(ff), np.c_[X, Y], np.array(S), np.array(C), np.array(FL)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--dets", default="data/balldet_out")
    ap.add_argument("--calib", default="config/calibration.json")
    ap.add_argument("--session", default="config/session.json")
    ap.add_argument("--out", default="data/ball.npz")
    a = ap.parse_args()
    with open(a.calib) as file:
        calib = json.load(file)
    pitch = calib["pitch"]
    with open(a.session) as file:
        m = json.load(file)["matches"][0]
    m_lo, m_hi = int(HMS(m["start"]) * FPS), int(HMS(m["end"]) * FPS)
    clips = scan(a.raw)
    per_cam = {}
    found_cache = False
    for ch in sorted(calib["cameras"]):
        cams = aim_cameras(calib, ch)
        F, XY, SZ, CF = [], [], [], []
        for c in clips:
            if c.ch != ch:
                continue
            p = os.path.join(a.dets, os.path.basename(c.path)[:-4] + ".npz")
            if not os.path.exists(p):
                continue
            found_cache = True
            with np.load(p) as d:
                CF.append(d["conf"])
                b = d["box"]
                F.append(np.round(c.t0 * FPS).astype(np.int64) + d["f"].astype(np.int64))
                XY.append((b[:, :2] + b[:, 2:]) / 2)
            sides = b[:, 2:] - b[:, :2]
            short, long = sides.min(axis=1), sides.max(axis=1)
            # The width across a motion streak estimates ball diameter. Its length
            # measures blur and would make a fast ground ball look airborne.
            SZ.append(np.where(long > 1.8 * short, short, long))
        if not F:
            per_cam[ch] = {}
            continue
        F, XY, SZ, CF = (np.concatenate(x) for x in (F, XY, SZ, CF))
        k = (F >= m_lo) & (F < m_hi) & (CF >= MIN_CONF)
        F, XY, SZ, CF = F[k], XY[k], SZ[k], CF[k]
        st = static_cells(F, XY)
        F, XY, SZ, CF = F[~st], XY[~st], SZ[~st], CF[~st]
        tr = track_camera(F, XY, SZ, CF)
        cam = cams[0][2]
        centre = cam.project(np.array([pitch["L"] / 2, pitch["W"] / 2, -0.11]))
        rows = []
        n_park = 0
        for t in tr:
            if t[0] == "parked":
                if np.linalg.norm(np.median(t[2], 0) - centre) > 40:   # not on the centre spot: a spare ball
                    n_park += 1; continue
                t = ("ball", t[1], t[2], t[3], t[4], np.zeros(len(t[1]), bool))
            _, f, xy, sz, cf, fl = t
            tid = len(rows)
            rows.append(dict(f=f, xy=xy, sz=sz, cf=cf, fl=fl, start=f[0]))
        print(f"{ch}: {len(F)} sightings (after fixed clutter), {len(rows)} confirmed tracks covering "
              f"{sum(len(r['f']) for r in rows) / ((m_hi - m_lo)) * 100:.0f}% of frames; {n_park} parked-ball tracks dropped")
        # rays per frame
        frame_pt = {}
        for r in rows:
            t = r["f"] / FPS
            for wi, (t0, t1, cm) in enumerate(cams):
                nxt = cams[wi + 1][0] if wi + 1 < len(cams) else 1e18
                s = (t >= t0) & (t < nxt)
                if not s.any():
                    continue
                d = cm.rays(r["xy"][s])
                Cg = cm._to_global(cm.C[None])[0].copy(); Cg[2] = cm.C[2]
                if cm.flip is not None:
                    d = d * np.array([-1, -1, 1])
                g, ok = cm.to_ground(r["xy"][s], height=0.11)
                from scipy.ndimage import median_filter
                szs = median_filter(r["sz"][s], size=7, mode="nearest")     # box size is noisy frame to frame
                r_obs = cm.f * BALL_D / np.maximum(szs, 3.0)
                for i, fi in enumerate(np.flatnonzero(s)):
                    frame_pt[int(r["f"][fi])] = (Cg, d[i], g[i], bool(ok[i]), r_obs[i], r["start"], bool(r["fl"][fi]))
        per_cam[ch] = frame_pt
    # put on the pitch
    if not found_cache:
        raise FileNotFoundError(f"No ball detection files in {a.dets} matched the clips in {a.raw}")
    chs = sorted(per_cam)
    allf = sorted(set(per_cam[chs[0]]) | set(per_cam[chs[1]]))
    L, W = pitch["L"], pitch["W"]
    out_f, out_xy, out_z, out_src, src_last = [], [], [], [], None
    for f in allf:
        A, B = per_cam[chs[0]].get(f), per_cam[chs[1]].get(f)
        pt = None
        if A and B:
            w_ = A[0] - B[0]; da, db = A[1], B[1]
            b_ = da @ db; d_ = da @ w_; e_ = db @ w_; den = 1 - b_ * b_
            if abs(den) > 1e-6:
                s_, t_ = (b_ * e_ - d_) / den, (e_ - b_ * d_) / den
                pa, pb = A[0] + s_ * da, B[0] + t_ * db
                if s_ > 0 and t_ > 0 and np.linalg.norm(pa - pb) < 0.8 and -0.3 < -(pa[2] + pb[2]) / 2 < 5:
                    P = (pa + pb) / 2; pt = (P[0], P[1], max(-P[2], 0.11)); src = "both"
        if pt is None:
            cand = []
            for ch, view in ((chs[0], A), (chs[1], B)):
                if view is None:
                    continue
                origin, _, g, ok, r_obs, start, _ = view
                if not ok or not (-1.5 < g[0] < L + 1.5 and -1.5 < g[1] < W + 1.5):
                    continue
                # A ball much larger than expected at its ground point may be airborne.
                ground_range = np.linalg.norm(np.r_[g, -0.11] - origin)
                if r_obs >= 0.6 * ground_range:
                    cand.append((ch, start, g))
            if not cand:
                continue
            src, _, g = min(cand, key=lambda v: (v[0] != src_last, v[1]))
            pt = (g[0], g[1], 0.11)
        if not (-1.5 < pt[0] < L + 1.5 and -1.5 < pt[1] < W + 1.5):
            continue
        src_last = src
        out_f.append(f); out_xy.append(pt[:2]); out_z.append(pt[2]); out_src.append(src)
    out_f, out_xy, out_z = np.array(out_f, dtype=np.int64), np.array(out_xy).reshape(-1, 2), np.array(out_z)
    # 5 per second for the stats and the replay: the frame on the tick, else one either side; no other filling
    ks, kxy, kz = [], [], []
    pos = {int(f): i for i, f in enumerate(out_f)}
    for k in range(int(np.ceil(m_lo / STEP)), int(m_hi // STEP)):
        i = pos.get(k * STEP)
        if i is not None:
            ks.append(k); kxy.append(out_xy[i]); kz.append(out_z[i]); continue
        a_, b_ = pos.get(k * STEP - 1), pos.get(k * STEP + 1)     # exact moment: midway between the frames either side
        if a_ is not None and b_ is not None:
            ks.append(k); kxy.append((out_xy[a_] + out_xy[b_]) / 2); kz.append((out_z[a_] + out_z[b_]) / 2)
    ks, kxy, kz = np.array(ks, dtype=np.int64), np.array(kxy).reshape(-1, 2), np.array(kz)
    np.savez_compressed(a.out, k=ks, xy=kxy, z=kz, m15=out_f, xy15=out_xy, z15=out_z,
                        views15=np.array([{'both': 2}.get(x, 1) for x in out_src], dtype=int), src15=np.array(out_src, dtype=str), w15=np.ones(len(out_f)))
    span = (m_hi - m_lo) / STEP
    st = np.diff(kxy, axis=0)[np.diff(ks) == 1]
    print(f"ball shown on {len(ks) / span * 100:.0f}% of the match; jumps > 4 m in 0.2 s: {int((np.linalg.norm(st, axis=1) > 4).sum())}")


if __name__ == "__main__":
    main()
