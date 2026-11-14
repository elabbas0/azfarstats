"""Stage 4: per-player physical stats from identity trajectories.

Usage: python -m azfar.stats --ids data/identities.npz --out data/out
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .timeline import GRID_HZ, fmt
from .features import DIM_REGION, AB_BINS, L_BINS

DT = 1.0 / GRID_HZ
MAX_GAP = 15          # ticks (3 s) bridged by interpolation
SMOOTH = 2.0          # gaussian sigma in ticks (0.4 s)
V_CAP = 9.0           # m/s
V_MOVE = 0.5          # below this the player is "standing" (noise floor)
ZONES = (("walk", 0.5, 2.0), ("jog", 2.0, 4.0), ("run", 4.0, 5.5), ("sprint", 5.5, 99))
HEAT = (42, 20)       # heat-map cells along length, width


def segments(k, xy):
    """Split an identity's observations into continuous, gap-interpolated, smoothed segments.
    Yields (k_array, xy_array, speed_array)."""
    o = np.argsort(k, kind="stable")
    k, xy = k[o], xy[o]
    k, first = np.unique(k, return_index=True)
    xy = xy[first]
    cuts = np.flatnonzero(np.diff(k) > MAX_GAP) + 1
    for s, e in zip(np.r_[0, cuts], np.r_[cuts, len(k)]):
        if e - s < 5:
            continue
        kk = np.arange(k[s], k[e - 1] + 1)
        x = np.interp(kk, k[s:e], xy[s:e, 0]); y = np.interp(kk, k[s:e], xy[s:e, 1])
        x, y = gaussian_filter1d(x, SMOOTH, mode="nearest"), gaussian_filter1d(y, SMOOTH, mode="nearest")
        v = np.hypot(np.gradient(x), np.gradient(y)) / DT
        yield kk, np.c_[x, y], np.minimum(v, V_CAP)


def lab_name(lab):
    """Rough colour name for an OpenCV-scaled Lab triple in 0..1."""
    L, a, b = lab[0] * 100, lab[1] * 255 - 128, lab[2] * 255 - 128
    chroma = np.hypot(a, b)
    if chroma < 10:
        return "black" if L < 30 else "white" if L > 68 else "grey"
    hue = np.degrees(np.arctan2(b, a)) % 360
    for lim, name in ((20, "pink"), (55, "red"), (80, "orange"), (115, "yellow"), (190, "green"), (250, "cyan"), (310, "blue"), (345, "purple"), (361, "pink")):
        if hue < lim:
            return name


def lab_to_hex(lab):
    import cv2
    px = np.array([[[lab[0] * 255, lab[1] * 255, lab[2] * 255]]], np.uint8)
    b, g, r = cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0, 0]
    return f"#{r:02x}{g:02x}{b:02x}"


def player_stats(k, xy, pitch, windows):
    """windows: {name: (k_start, k_end)}; returns {window: stats dict} and the heat map over everything."""
    L, W = pitch["L"], pitch["W"]
    heat = np.zeros(HEAT)
    acc = {w: dict(seconds=0.0, distance_m=0.0, top=[], zone_m={z[0]: 0.0 for z in ZONES}, sprints=0, sx=0.0, sy=0.0)
           for w in windows}
    for kk, p, v in segments(k, xy):
        hx = np.clip((p[:, 0] / L * HEAT[0]).astype(int), 0, HEAT[0] - 1)
        hy = np.clip((p[:, 1] / W * HEAT[1]).astype(int), 0, HEAT[1] - 1)
        np.add.at(heat, (hx, hy), DT)
        for w, (ks, ke) in windows.items():
            m = (kk >= ks) & (kk < ke)
            if m.sum() < 2:
                continue
            a = acc[w]
            vv = v[m]
            a["seconds"] += m.sum() * DT
            a["distance_m"] += float((vv[vv > V_MOVE]).sum() * DT)
            a["top"].append(vv)
            a["sx"] += p[m, 0].sum() * DT; a["sy"] += p[m, 1].sum() * DT
            for name, lo, hi in ZONES:
                a["zone_m"][name] += float(vv[(vv >= lo) & (vv < hi)].sum() * DT)
            fast = np.r_[0, (vv >= 5.5).astype(int), 0]
            runs = np.flatnonzero(np.diff(fast))
            a["sprints"] += int(((runs[1::2] - runs[::2]) >= GRID_HZ * 0.8).sum())
    out = {}
    for w, a in acc.items():
        if a["seconds"] < 1:
            continue
        top = np.concatenate(a["top"])
        out[w] = dict(minutes=round(a["seconds"] / 60, 1), distance_m=round(a["distance_m"]),
                      m_per_min=round(a["distance_m"] / (a["seconds"] / 60), 1),
                      top_speed_kmh=round(float(np.percentile(top, 99.8)) * 3.6, 1),
                      sprints=a["sprints"], zone_m={z: round(m) for z, m in a["zone_m"].items()},
                      avg_x=round(a["sx"] / a["seconds"], 1), avg_y=round(a["sy"] / a["seconds"], 1))
    return out, heat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="data/identities.npz")
    ap.add_argument("--tracks", default="data/tracks.npz")
    ap.add_argument("--calib", default="config/calibration.json")
    ap.add_argument("--session", default="config/session.json")
    ap.add_argument("--out", default="data/out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    I = np.load(a.ids); T = np.load(a.tracks)
    pitch = json.load(open(a.calib))["pitch"]
    sess = json.load(open(a.session)) if os.path.exists(a.session) else {}
    hms = lambda s: sum(int(p) * m for p, m in zip(s.split(":"), (3600, 60, 1)))
    windows = {"all": (int(I["k"].min()), int(I["k"].max()) + 1)}
    for m in sess.get("matches", []):
        windows[m["name"]] = (int(hms(m["start"]) * GRID_HZ), int(hms(m["end"]) * GRID_HZ))
    feats, rngs = (T["feat0"].astype(np.float32), T["feat1"].astype(np.float32)), (T["rng0"], T["rng1"])
    players = []
    for pid in np.unique(I["pid"]):
        m = I["pid"] == pid
        st, heat = player_stats(I["k"][m], I["xy"][m], pitch, windows)
        if "all" not in st:
            continue
        good = m & (I["sig"] < 0.8) & ~np.isnan(I["h"])
        height = float(np.median(I["h"][good])) if good.sum() > 20 else float("nan")
        cols = {}
        o = 0
        for r, region in enumerate(("torso", "shorts", "legs")):
            labs = []
            for c, key in enumerate(("i0", "i1")):
                idx = I[key][m]; idx = idx[idx >= 0]
                idx = idx[rngs[c][idx] < 22]
                f = feats[c][idx, r * DIM_REGION + AB_BINS**2 + L_BINS:(r + 1) * DIM_REGION]
                labs.append(f[f.sum(1) > 0])
            labs = np.concatenate(labs)
            if len(labs):
                med = np.median(labs, 0)
                cols[region] = dict(name=lab_name(med), hex=lab_to_hex(med))
        players.append(dict(id=int(pid), first_seen=fmt(I["k"][m].min() / GRID_HZ), last_seen=fmt(I["k"][m].max() / GRID_HZ),
                            height_m=round(height, 2), colours=cols, stats=st, heat=np.round(heat, 1).tolist()))
    players.sort(key=lambda p: -p["stats"]["all"]["minutes"])
    json.dump(dict(pitch=pitch, windows={w: [fmt(a_ / GRID_HZ), fmt(b_ / GRID_HZ)] for w, (a_, b_) in windows.items()},
                   players=players), open(os.path.join(a.out, "players.json"), "w"))
    with open(os.path.join(a.out, "players.csv"), "w") as f:
        f.write("window,player,shirt,shorts,height_m,minutes,distance_m,m_per_min,top_speed_kmh,sprints,walk_m,jog_m,run_m,sprint_m,avg_x,avg_y\n")
        for w in windows:
            for p in players:
                s = p["stats"].get(w)
                if not s:
                    continue
                c = p["colours"]
                f.write(",".join(str(v) for v in (w, p["id"], c.get("torso", {}).get("name", ""), c.get("shorts", {}).get("name", ""),
                                                  p["height_m"], s["minutes"], s["distance_m"], s["m_per_min"], s["top_speed_kmh"], s["sprints"],
                                                  *s["zone_m"].values(), s["avg_x"], s["avg_y"])) + "\n")
    for p in players:
        s, c = p["stats"]["all"], p["colours"]
        print(f"P{p['id']:<3} {c.get('torso', {}).get('name', '?'):7} / {c.get('shorts', {}).get('name', '?'):7} h={p['height_m']:.2f} "
              f"{p['first_seen']}-{p['last_seen']} {s['minutes']:6.1f} min {s['distance_m']:6d} m {s['m_per_min']:6.1f} m/min top {s['top_speed_kmh']:.1f} km/h sprints {s['sprints']}")


if __name__ == "__main__":
    main()
