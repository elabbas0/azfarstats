"""Stage 5: ball events - touches, passes, shots, goals, assists, saves, dribbles.

Everything is derived from two things only: where each identified player is, and where the ball
is, five times a second. That makes these *estimates*: every event carries a time so it can be
checked in the replay.

Usage: python -m azfar.events --ids data/identities.npz --ball data/ball.npz --out data/out
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from .timeline import GRID_HZ, fmt

DT = 1.0 / GRID_HZ
CONTROL_R = 1.6        # m: ball this close to a player's feet is "theirs"
GOAL_W = 3.0           # goal width, m
SHOT_V = 7.5           # m/s: slower than this towards goal is a pass/roll, not a shot
BALL_GAP = 8           # ticks of missing ball bridged by interpolation


def grid_tracks(I, k0, k1, max_gap=15):
    """Player positions on a dense tick grid: array [n_players, n_ticks, 2] with nan where absent."""
    pids = np.unique(I["pid"])
    P = np.full((len(pids), k1 - k0, 2), np.nan)
    for n, pid in enumerate(pids):
        m = I["pid"] == pid
        k, xy = I["k"][m], I["xy"][m]
        o = np.argsort(k, kind="stable"); k, xy = k[o], xy[o]
        k, f = np.unique(k, return_index=True); xy = xy[f]
        cuts = np.flatnonzero(np.diff(k) > max_gap) + 1
        for s, e in zip(np.r_[0, cuts], np.r_[cuts, len(k)]):
            if e - s < 3:
                continue
            kk = np.arange(k[s], k[e - 1] + 1)
            x = gaussian_filter1d(np.interp(kk, k[s:e], xy[s:e, 0]), 1.5, mode="nearest")
            y = gaussian_filter1d(np.interp(kk, k[s:e], xy[s:e, 1]), 1.5, mode="nearest")
            P[n, kk - k0] = np.c_[x, y]
    return pids, P


def grid_ball(B, k0, k1):
    b = np.full((k1 - k0, 2), np.nan)
    k, xy = B["k"], B["xy"]
    cuts = np.flatnonzero(np.diff(k) > BALL_GAP) + 1
    for s, e in zip(np.r_[0, cuts], np.r_[cuts, len(k)]):
        if e - s < 2:
            continue
        kk = np.arange(k[s], k[e - 1] + 1)
        ok = (kk >= k0) & (kk < k1)
        x, y = np.interp(kk, k[s:e], xy[s:e, 0]), np.interp(kk, k[s:e], xy[s:e, 1])
        if len(kk) >= 3:  # median of three: removes single-sample flicker without rounding off kicks
            x, y = median_filter(x, 3, mode="nearest"), median_filter(y, 3, mode="nearest")
        b[kk[ok] - k0] = np.c_[x, y][ok]
    return b


def possession_spells(P, ball):
    """Runs of ticks where one player has the ball at their feet.
    Returns list of dicts: p (player index), s, e (tick offsets, inclusive)."""
    d = np.linalg.norm(P - ball[None], axis=2)              # [players, ticks]
    d = np.where(np.isnan(d), 1e9, d)
    near = d.argmin(0)
    dist = d.min(0)
    second = np.partition(d, 1, axis=0)[1] if d.shape[0] > 1 else np.full(d.shape[1], 1e9)
    owner = np.where((dist < CONTROL_R) & ~np.isnan(ball[:, 0]), near, -1)
    contested = (second < CONTROL_R) & (owner >= 0)
    spells, cur = [], None
    for t, o in enumerate(owner):
        if o >= 0:
            if cur is not None and cur["p"] == o and t - cur["e"] <= 4:
                cur["e"] = t
            else:
                if cur is not None:
                    spells.append(cur)
                cur = dict(p=int(o), s=t, e=t)
    if cur is not None:
        spells.append(cur)
    # a real touch lasts at least two ticks, or bends the ball's path
    v = np.gradient(ball, axis=0) / DT
    out = []
    for sp in spells:
        if sp["e"] - sp["s"] >= 1:
            out.append(sp); continue
        a, b = max(sp["s"] - 2, 0), min(sp["s"] + 2, len(ball) - 1)
        va, vb = v[a], v[b]
        if not (np.isnan(va).any() or np.isnan(vb).any()):
            cosang = va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-6)
            if cosang < 0.85 and max(np.linalg.norm(va), np.linalg.norm(vb)) > 2.0:
                out.append(sp)
    for sp in out:
        sp["contested"] = bool(contested[sp["s"]:sp["e"] + 1].mean() > 0.5)
    return out


def detect_kickoffs(P, ball, pitch, min_players=6):
    """Ticks where play restarts from the centre spot: ball resting on the spot, every player in
    one of two groups either side of the halfway line. Returns list of tick offsets."""
    L, W = pitch["L"], pitch["W"]
    centre = np.array([L / 2, W / 2])
    at = np.linalg.norm(ball - centre, axis=1) < 2.2
    speed = np.linalg.norm(np.gradient(ball, axis=0), axis=1) / DT
    still = at & (speed < 1.5)
    out, t = [], 0
    n = len(ball)
    while t < n:
        if still[t]:
            e = t
            while e + 1 < n and (still[e + 1] or (e + 2 < n and still[e + 2])):
                e += 1
            if e - t >= 5:  # a second on the spot
                x = P[:, t:e + 1, 0]
                present = ~np.isnan(x).all(1)
                mx = np.nanmedian(x[present], axis=1)
                far = np.abs(mx - L / 2) > 1.0     # everyone but the takers stands back in own half
                if present.sum() >= min_players and far.sum() >= present.sum() - 3 and (mx < L / 2).sum() >= 2 and (mx > L / 2).sum() >= 2:
                    if not out or e - out[-1] > 60 * GRID_HZ:   # one restart, however long the players dither
                        out.append(e)
            t = e + 1
        else:
            t += 1
    return out


def infer_teams(P, kickoffs, pitch, session_start=0):
    """Team per player from which half they stand in at kick-offs (teams line up in their own half).
    Returns (team array of 0/1/-1, side array: +1 if team 0 defends the low-x goal at each kick-off)."""
    L = pitch["L"]
    n = P.shape[0]
    snaps = []
    for t in kickoffs:
        x = np.nanmedian(P[:, max(t - 10, 0):t + 1, 0], axis=1)
        snaps.append(np.where(np.isnan(x), 0, np.where(x < L / 2, 1, -1)))  # +1 = low-x half
    if not snaps:
        return np.full(n, -1), np.array([])
    S = np.array(snaps, float)
    # sides may swap at half time: align every snapshot's sign to the first by agreement
    ref = S[0].copy()
    sign = np.ones(len(S))
    for it in range(3):
        for i in range(len(S)):
            sign[i] = 1 if (S[i] * ref).sum() >= 0 else -1
        ref = (S * sign[:, None]).sum(0)
    team = np.where(ref > 0, 0, np.where(ref < 0, 1, -1))
    return team, sign


def run(I, B, pitch, team_override=None):
    k0, k1 = int(I["k"].min()), int(I["k"].max()) + 1
    pids, P = grid_tracks(I, k0, k1)
    ball = grid_ball(B, k0, k1)
    L, W = pitch["L"], pitch["W"]
    kickoffs = detect_kickoffs(P, ball, pitch)
    team, sign = infer_teams(P, kickoffs, pitch)
    if team_override:
        for n, pid in enumerate(pids):
            if str(int(pid)) in team_override:
                team[n] = team_override[str(int(pid))]
    # which goal each team defends at each tick: +1 => team 0 defends x=0
    side = np.ones(k1 - k0)
    for t, s in zip(kickoffs, sign):
        side[t:] = s
    if len(kickoffs):
        side[:kickoffs[0]] = sign[0]
    spells = possession_spells(P, ball)
    bv = np.gradient(ball, axis=0) / DT
    ev = []
    T = lambda t: fmt((k0 + t) / GRID_HZ)

    def attack_goal_x(tm, t):
        """x of the goal that team tm attacks at tick t."""
        defends_low = (side[t] > 0) == (tm == 0)
        return L if defends_low else 0.0

    for i, sp in enumerate(spells):
        p, tm = sp["p"], team[sp["p"]]
        dur = (sp["e"] - sp["s"] + 1) * DT
        moved = float(np.linalg.norm(P[p, sp["e"]] - P[p, sp["s"]])) if not np.isnan(P[p, [sp["s"], sp["e"]]]).any() else 0.0
        ev.append(dict(type="touch", t=sp["s"], player=int(pids[p]), dur=round(dur, 1)))
        # --- dribble: carries the ball past a close opponent and keeps it
        if dur >= 1.2 and moved >= 3.0 and tm >= 0:
            opp = np.flatnonzero((team >= 0) & (team != tm))
            if len(opp):
                dd = np.linalg.norm(P[opp, sp["s"]:sp["e"] + 1] - P[p, sp["s"]:sp["e"] + 1][None], axis=2)
                close = np.nanmin(dd, axis=1) < 1.5
                nxt = spells[i + 1] if i + 1 < len(spells) else None
                lost = nxt is not None and team[nxt["p"]] != tm and nxt["s"] - sp["e"] <= 5
                if close.any():
                    ev.append(dict(type="dribble", t=sp["s"], player=int(pids[p]), won=not lost, metres=round(moved, 1)))
        if i + 1 >= len(spells):
            break
        nx = spells[i + 1]
        gap = (nx["s"] - sp["e"]) * DT
        # --- shot: ball leaves the foot fast, heading into the goal mouth
        shot = None
        if tm >= 0:
            a, b = sp["e"], min(sp["e"] + 6, len(ball) - 1)
            seg = bv[a:b + 1]
            sp_fast = np.linalg.norm(seg, axis=1)
            if np.isfinite(sp_fast).any():
                j = int(np.nanargmax(sp_fast)); v = seg[j]; pos = ball[a + j]
                gx = attack_goal_x(tm, sp["e"])
                if sp_fast[j] >= SHOT_V and abs(v[0]) > 1e-3 and (gx - pos[0]) * v[0] > 0 and abs(gx - pos[0]) < 22:
                    y_at = pos[1] + v[1] * (gx - pos[0]) / v[0]
                    off = abs(y_at - W / 2)
                    if off < GOAL_W / 2 + 2.5:
                        shot = dict(type="shot", t=sp["e"], player=int(pids[p]), speed_kmh=round(float(sp_fast[j]) * 3.6),
                                    dist_m=round(float(abs(gx - pos[0])), 1), on_target=bool(off < GOAL_W / 2 + 0.6), goal_x=gx)
                        ev.append(shot)
        if shot is None and nx["p"] != p and gap <= 5.0:
            travel = float(np.linalg.norm(ball[nx["s"]] - ball[sp["e"]]))
            same = tm >= 0 and team[nx["p"]] == tm
            if same and travel >= 2.0:
                ev.append(dict(type="pass", t=sp["e"], player=int(pids[p]), to=int(pids[nx["p"]]), metres=round(travel, 1)))
            elif tm >= 0 and team[nx["p"]] >= 0 and not same:
                kind = "interception" if travel >= 3.0 else "tackle"
                ev.append(dict(type=kind, t=nx["s"], player=int(pids[nx["p"]]), from_player=int(pids[p])))
                if travel >= 3.0:
                    ev.append(dict(type="pass_failed", t=sp["e"], player=int(pids[p])))
        if shot is not None:
            # save: on-target shot whose next touch is a defender standing in front of that goal
            q = nx["p"]
            if shot["on_target"] and team[q] >= 0 and team[q] != tm and gap <= 3.0 and abs(P[q, nx["s"], 0] - shot["goal_x"]) < 4.0:
                ev.append(dict(type="save", t=nx["s"], player=int(pids[q]), shot_by=int(pids[p])))
                shot["saved"] = True
    # --- goals: a kick-off means the previous attack ended in the net
    shots = [e for e in ev if e["type"] == "shot"]
    passes = [e for e in ev if e["type"] == "pass"]
    for n, t in enumerate(kickoffs):
        if n == 0 and t < 120 * GRID_HZ:
            continue  # the opening kick-off
        prev = [s for s in shots if t - 90 * GRID_HZ < s["t"] < t - 3 * GRID_HZ and not s.get("goal")]
        g = dict(type="goal", t=t, kickoff=True)
        if prev:
            s = prev[-1]
            s["goal"] = True; s.pop("saved", None)
            g.update(t=s["t"], player=s["player"])
            a = [p_ for p_ in passes if p_["to"] == s["player"] and 0 <= s["t"] - p_["t"] < 12 * GRID_HZ]
            if a:
                g["assist"] = a[-1]["player"]
                ev.append(dict(type="assist", t=a[-1]["t"], player=a[-1]["player"], to=s["player"]))
        ev.append(g)
    # --- probable goals: an on-target shot after which the ball disappears at the goal mouth
    # (the net hides it from both cameras) and nobody is seen saving it
    seen = ~np.isnan(ball[:, 0])
    last_goal_t = -10**9
    for s_ in shots:
        if s_.get("goal") or s_.get("saved") or not s_["on_target"]:
            continue
        a, b = s_["t"], min(s_["t"] + int(3 * GRID_HZ), len(ball) - 1)
        idx = np.flatnonzero(seen[a:b + 1]) + a
        if not len(idx):
            continue
        # last sighting before the first long blackout
        gaps = np.flatnonzero(np.diff(idx) >= 8)
        last = idx[gaps[0]] if len(gaps) else idx[-1]
        vanished = (not seen[last + 1:last + 9].any()) if last + 9 < len(ball) else False
        gx = s_["goal_x"]
        if vanished and abs(ball[last, 0] - gx) < 2.5 and abs(ball[last, 1] - W / 2) < GOAL_W / 2 + 0.7 and s_["t"] - last_goal_t > 20 * GRID_HZ:
            s_["goal"] = True; last_goal_t = s_["t"]
            g = dict(type="goal", t=s_["t"], player=s_["player"], probable=True)
            a_ = [p_ for p_ in passes if p_["to"] == s_["player"] and 0 <= s_["t"] - p_["t"] < 12 * GRID_HZ]
            if a_:
                g["assist"] = a_[-1]["player"]
                ev.append(dict(type="assist", t=a_[-1]["t"], player=a_[-1]["player"], to=s_["player"]))
            ev.append(g)
    ev = [e for e in ev if not (e["type"] == "save" and any(s.get("goal") and s["player"] == e["shot_by"] and abs(s["t"] - e["t"]) < 20 for s in shots))]
    ev.sort(key=lambda e: e["t"])
    for e in ev:
        e["k"] = k0 + e["t"]; e["time"] = T(e["t"]); e.pop("t"); e.pop("goal_x", None)
    return dict(events=ev, teams={int(p): int(t) for p, t in zip(pids, team)}, kickoffs=[T(t) for t in kickoffs],
                ball_seen=float((~np.isnan(ball[:, 0])).mean()))


def tally(res):
    keys = ("touch", "pass", "pass_failed", "shot", "goal", "assist", "save", "dribble", "interception", "tackle")
    tab = {}
    for e in res["events"]:
        if "player" not in e:
            continue
        row = tab.setdefault(e["player"], {k: 0 for k in keys} | {"shots_on_target": 0, "dribbles_won": 0})
        row[e["type"]] += 1
        if e["type"] == "shot" and e.get("on_target"):
            row["shots_on_target"] += 1
        if e["type"] == "dribble" and e.get("won"):
            row["dribbles_won"] += 1
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="data/identities.npz")
    ap.add_argument("--ball", default="data/ball.npz")
    ap.add_argument("--calib", default="config/calibration.json")
    ap.add_argument("--session", default="config/session.json")
    ap.add_argument("--out", default="data/out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    I, B = np.load(a.ids), np.load(a.ball)
    pitch = json.load(open(a.calib))["pitch"]
    sess = json.load(open(a.session)) if os.path.exists(a.session) else {}
    res = run(I, B, pitch, sess.get("teams"))
    res["tally"] = tally(res)
    json.dump(res, open(os.path.join(a.out, "events.json"), "w"))
    n = {}
    for e in res["events"]:
        n[e["type"]] = n.get(e["type"], 0) + 1
    print("ball located", f"{res['ball_seen'] * 100:.0f}% of the time; kick-offs at", res["kickoffs"])
    print("events:", n)
    print("teams:", res["teams"])


if __name__ == "__main__":
    main()
