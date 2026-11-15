"""Possession-first ball tracking.

In small-sided football the ball is at someone's feet most of the time, and in between it travels from
one player to the next. Player tracks are far more reliable than ball sightings, so we decide, for every
0.2 s of the match, who has the ball (or that it is travelling), using only clear ball sightings as
evidence, with a single best explanation for the whole match (Viterbi):

  * owned by player j : the ball is at j's feet; staying with j costs nothing, sightings near j support it
  * travelling        : sightings away from every player support it
  * changes of state cost something, so one stray sighting (a boot, a sock) cannot flip possession,
    and the ball can only pass directly between two players who are next to each other.

The ball's position then follows the owner's feet, and while travelling it moves along straight lines
from where it left one player to where the next one received it, through reliable sightings on the way,
never faster than a hard shot.

Usage: python -m azfar.possession   (replaces data/ball.npz; keeps the sighting-based track as data/ball_sightings.npz)
"""
from __future__ import annotations
import argparse, json, os, shutil
import numpy as np
from .timeline import scan, GRID_HZ, FPS
from . import ball as B

DT = 1.0 / GRID_HZ
SIG_OWN = 1.3        # m: a sighting this close to a player's feet supports "player has it"
SIG_FREE = 2.0       # m: a sighting this far from everyone supports "travelling"
C_SWITCH = 3.0       # direct hand-over between two neighbouring players (tackle, shielding)
C_REL = 2.5          # player kicks / loses the ball
C_REC = 2.5          # player receives the ball
C_FREE = 0.05        # ball keeps travelling one more tick
NEAR_SWITCH = 1.5    # m: direct hand-over only between players this close
EPS = 0.05
V_MAX = 28.0         # m/s
W_WAY = 0.45        # sighting weight needed to bend a travelling ball's path
W_ANCHOR = 0.22      # sighting weight needed to count as evidence
HMS = lambda s: sum(int(p) * m for p, m in zip(s.split(":"), (3600, 60, 1)))


def evidence(calib, clips, det, scan_dir, m_lo, m_hi):
    """Clear ball sightings per 0.2 s tick: {k: array of (x, y, h, w, views)}."""
    chs = sorted(calib["cameras"])
    A, Bv = (B.load(det, scan_dir, clips, calib, ch, m_lo, m_hi) for ch in chs)
    cand = B.candidates(A, Bv, calib["pitch"])
    ev = {}
    for m, rows in cand.items():
        r = rows[rows[:, 3] >= W_ANCHOR]
        if len(r):
            k = int(round(m / B.STEP))
            ev.setdefault(k, []).append(r[:, :5])
    return {k: np.concatenate(v)[np.argsort(-np.concatenate(v)[:, 3])][:4] for k, v in ev.items()}


def players_by_tick(I, k0, k1):
    """Identified players (continuous, smoothed, short gaps bridged) present at each tick."""
    from .events import grid_tracks
    pids, Pg = grid_tracks(I, k0, k1, max_gap=10)
    out = []
    for i in range(k1 - k0):
        ok = ~np.isnan(Pg[:, i, 0])
        out.append((pids[ok].astype(np.int64), Pg[ok, i]))
    return out


def viterbi(P, E, k0):
    """P: per tick (tids, xy); E: evidence dict. Returns per tick the owner tid (-1 = travelling)."""
    n = len(P)
    back, states = [], []
    prev_ids = np.array([-1]); prev_xy = np.full((1, 2), np.nan); score = np.zeros(1)
    for i in range(n):
        tids, xy = P[i]
        ids = np.r_[-1, tids]; pos = np.vstack([np.full((1, 2), np.nan), xy]) if len(tids) else np.full((1, 2), np.nan)
        # emission
        em = np.zeros(len(ids))
        ev = E.get(k0 + i)
        if ev is not None:
            a, w = ev[:, :2], ev[:, 3]
            if len(tids):
                d = np.linalg.norm(a[:, None] - xy[None], axis=2)          # anchors x players
                own = (w[:, None] * np.exp(-d ** 2 / (2 * SIG_OWN ** 2))).sum(0)
                dnear = d.min(1)
            else:
                own = np.zeros(0); dnear = np.full(len(a), 99.0)
            free = (w * (1 - np.exp(-dnear ** 2 / (2 * SIG_FREE ** 2)))).sum()
            em = np.log(EPS + np.r_[free, own])
            em -= em.max()
        # transitions prev -> cur (costs)
        C = np.full((len(prev_ids), len(ids)), np.inf)
        C[0, 0] = C_FREE                                              # free -> free
        C[0, 1:] = C_REC                                              # free -> own
        if len(prev_ids) > 1:
            C[1:, 0] = C_REL                                          # own -> free
            if len(ids) > 1:
                same = prev_ids[1:, None] == ids[None, 1:]
                d = np.linalg.norm(prev_xy[1:, None] - pos[None, 1:], axis=2)
                gone = ~np.isin(prev_ids[1:], ids[1:])[:, None]       # tracklet ended: same person continues?
                Cw = np.where(d < NEAR_SWITCH, C_SWITCH, np.inf)
                Cw = np.where(gone & (d < 1.0), 0.3, Cw)
                C[1:, 1:] = np.where(same, 0.0, Cw)
        tot = score[:, None] + C
        arg = np.argmin(tot, 0)
        score = tot[arg, np.arange(len(ids))] - em
        score -= score.min()
        back.append(arg); states.append(ids)
        prev_ids, prev_xy = ids, pos
    out = np.empty(n, np.int64)
    j = int(np.argmin(score))
    for i in range(n - 1, -1, -1):
        out[i] = states[i][j]
        j = back[i][j]
    return out


def place(P, E, owner, k0):
    """Ball position per tick from the possession sequence."""
    n = len(P)
    xy = np.full((n, 2), np.nan); h = np.full(n, B.BALL_R)
    for i in range(n):
        if owner[i] >= 0:
            tids, pxy = P[i]
            f = pxy[tids == owner[i]][0]
            ev = E.get(k0 + i)
            if ev is not None:                       # a sighting at the feet refines the spot
                d = np.linalg.norm(ev[:, :2] - f, axis=1)
                j = int(np.argmin(d))
                xy[i] = ev[j, :2] if d[j] < 2.0 else f
            else:
                xy[i] = f
    # travelling stretches: straight lines between release, reliable sightings, reception
    i = 0
    while i < n:
        if owner[i] >= 0:
            i += 1; continue
        j = i
        while j < n and owner[j] < 0:
            j += 1
        pts = []
        if i > 0 and np.isfinite(xy[i - 1, 0]):
            pts.append((i - 1, xy[i - 1]))
        for t in range(i, j):
            ev = E.get(k0 + t)
            if ev is not None and (ev[0, 4] >= 2 or ev[0, 3] >= W_WAY):   # waypoints: only clear sightings
                pts.append((t, ev[0, :2]))
        if j < n and np.isfinite(xy[j, 0]):
            pts.append((j, xy[j]))
        # keep only waypoints reachable at ball speed from the previous kept one
        kept = []
        for t, p in pts:
            if not kept or np.linalg.norm(p - kept[-1][1]) <= 0.5 + V_MAX * (t - kept[-1][0]) * DT:
                kept.append((t, p))
        if len(kept) >= 2:
            ts = np.array([t for t, _ in kept]); ps = np.array([p for _, p in kept])
            tt = np.arange(max(i, ts[0]), min(j - 1, ts[-1]) + 1)
            xy[tt, 0] = np.interp(tt, ts, ps[:, 0]); xy[tt, 1] = np.interp(tt, ts, ps[:, 1])
        elif len(kept) == 1 and kept[0][0] >= i:
            xy[kept[0][0]] = kept[0][1]
        i = j
    return xy, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="data/det")
    ap.add_argument("--scan", default="data/ballscan")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--ids", default="data/identities.npz")
    ap.add_argument("--calib", default="config/calibration.json")
    ap.add_argument("--session", default="config/session.json")
    ap.add_argument("--out", default="data/ball.npz")
    a = ap.parse_args()
    calib = json.load(open(a.calib)); clips = scan(a.raw)
    mt = json.load(open(a.session))["matches"][0]
    t_lo, t_hi = HMS(mt["start"]), HMS(mt["end"])
    k0, k1 = int(t_lo * GRID_HZ), int(t_hi * GRID_HZ)
    E = evidence(calib, clips, a.det, a.scan, int(t_lo * FPS), int(t_hi * FPS))
    P = players_by_tick(np.load(a.ids), k0, k1)
    owner = viterbi(P, E, k0)
    xy, h = place(P, E, owner, k0)
    ok = np.isfinite(xy[:, 0])
    k = np.arange(k0, k1)[ok]
    if os.path.exists(a.out) and not os.path.exists("data/ball_sightings.npz"):
        shutil.copy(a.out, "data/ball_sightings.npz")
    own = owner[ok]
    np.savez_compressed(a.out, k=k, xy=xy[ok], z=h[ok], owner=own,
                        m15=k * B.STEP, xy15=xy[ok], z15=h[ok], views15=np.where(own >= 0, 4, 5), w15=np.ones(ok.sum()))
    sw = np.diff(owner) != 0
    print(f"evidence on {len(E)} of {k1 - k0} ticks; ball owned {np.mean(owner >= 0) * 100:.0f}% of the time, travelling "
          f"{np.mean(owner < 0) * 100:.0f}%; {sw.sum()} changes of state; ball placed on {ok.mean() * 100:.0f}% of ticks")
    st = np.diff(xy[ok], axis=0)[np.diff(k) == 1]
    print("jumps > 4 m in 0.2 s:", int((np.linalg.norm(st, axis=1) > 4).sum()))


if __name__ == "__main__":
    main()
