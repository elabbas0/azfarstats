"""Stage 2: fuse both cameras on the pitch plane and build short, pure tracklets.

Usage: python -m azfar.fuse --det data/det --raw data/raw --out data/tracks.npz
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from scipy.optimize import linear_sum_assignment
from .camera import Camera
from .timeline import scan, GRID_HZ
from .features import DIM

DT = 1.0 / GRID_HZ
VIEW_MARGIN = 1.5     # per-view tolerance before fusion (far-range foot points are noisy)
MARGIN = 0.5          # fused position must be within this of the lines: people off the pitch are not counted
MIN_CONF = 0.30
GATE = 2.2            # association gate (m) track <-> detection
MERGE_GATE = 2.5      # cross-camera merge gate (m) at mid range
MAX_MISS = 6          # ticks a track may coast
CROWD = 0.9           # two players closer than this (m) -> identities are ambiguous


def aim_cameras(calib, ch):
    """[(t0, t1, Camera)] following the camera's aim through the recording (see azfar.drift)."""
    base = calib["cameras"][ch]
    wins = calib.get("aim", {}).get(ch, [])
    if not wins:
        return [(-1e18, 1e18, Camera.from_dict(base))]
    out = [(w["t0"], w["t1"], Camera.from_dict({**base, "rvec": w["rvec"]})) for w in wins]
    out[0] = (-1e18,) + out[0][1:]
    out[-1] = (out[-1][0], 1e18, out[-1][2])
    return out


def load_camera_dets(det_dir, clips, cams, ch, pitch):
    """All person detections of one camera projected to the pitch. Returns dict of arrays."""
    ks, xy, hh, conf, feat, box, rng = [], [], [], [], [], [], []
    balls = []
    for c in clips:
        if c.ch != ch:
            continue
        p = os.path.join(det_dir, os.path.basename(c.path).replace(".mp4", ".npz"))
        if not os.path.exists(p):
            continue
        dd = np.load(p)
        tt = dd["k"] / GRID_HZ
        for wi, (t0, t1, cam) in enumerate(cams):
            nxt = cams[wi + 1][0] if wi + 1 < len(cams) else 1e18
            sel = (tt >= t0) & (tt < nxt)
            if not sel.any():
                continue
            d = {key: dd[key][sel] for key in ("k", "box", "conf", "cls", "feat")}
            b = d["box"]
            isb = d["cls"] == 32
            if isb.any():
                bb = b[isb]
                cen = np.c_[(bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2]
                g, ok = cam.to_ground(cen, height=0.11)
                balls.append(np.c_[d["k"][isb], g, d["conf"][isb], ok])
            m = (d["cls"] == 0) & (d["conf"] >= MIN_CONF)
            b = b[m]
            foot = np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3]]
            top = np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 1]]
            g, ok = cam.to_ground(foot)
            h = cam.height_at(top, g)
            # a box that ends above the feet (legs hidden behind someone) puts the foot point too
            # far away and reads as a very short person: place those from the head instead
            cut = ok & (h < 1.35)
            if cut.any():
                g2, ok2 = cam.to_ground(top[cut], height=1.65)
                g[cut] = np.where(ok2[:, None], g2, g[cut])
            aspect = (b[:, 3] - b[:, 1]) / np.maximum(b[:, 2] - b[:, 0], 1)
            h = np.where(cut | (aspect < 1.7) | (d["conf"][m] < 0.55), np.nan, h)  # height only from clean, upright views
            loc = cam._to_local(np.c_[g, np.zeros(len(g))])
            r = np.linalg.norm(loc - cam.C, axis=1)
            inside = ok & (g[:, 0] > -VIEW_MARGIN) & (g[:, 0] < pitch["L"] + VIEW_MARGIN) & (g[:, 1] > -VIEW_MARGIN) & (g[:, 1] < pitch["W"] + VIEW_MARGIN)
            inside &= ~(h > 2.4) & (b[:, 3] < 1075)  # not a giant; feet not cut by the frame
            ks.append(d["k"][m][inside]); xy.append(g[inside]); hh.append(h[inside]); conf.append(d["conf"][m][inside])
            feat.append(d["feat"][m][inside]); box.append(b[inside]); rng.append(r[inside])
    cat = lambda L, shape=(0,): np.concatenate(L) if L else np.zeros(shape)
    out = dict(k=cat(ks).astype(np.int64), xy=cat(xy, (0, 2)), h=cat(hh), conf=cat(conf), feat=cat(feat, (0, DIM)),
               box=cat(box, (0, 4)), rng=cat(rng))
    out["ball"] = np.concatenate(balls) if balls else np.zeros((0, 5))
    return out


def sigma(r):
    """Ground-position uncertainty (m) of a foot point seen from range r."""
    return 0.25 + (r / 18.0) ** 2


def merge_views(a, b):
    """a, b: lists of per-view observations for one tick (dicts with xy, rng, ...). Returns fused list."""
    if not len(a["xy"]):
        return [(None, j) for j in range(len(b["xy"]))]
    if not len(b["xy"]):
        return [(i, None) for i in range(len(a["xy"]))]
    d = np.linalg.norm(a["xy"][:, None] - b["xy"][None], axis=2)
    gate = MERGE_GATE * (0.6 + 0.4 * (sigma(a["rng"])[:, None] + sigma(b["rng"])[None]))
    cost = np.where(d < gate, d, 1e3)
    ri, ci = linear_sum_assignment(cost)
    pairs, ua, ub = [], set(range(len(a["xy"]))), set(range(len(b["xy"])))
    for i, j in zip(ri, ci):
        if cost[i, j] < 1e3:
            pairs.append((i, j)); ua.discard(i); ub.discard(j)
    return pairs + [(i, None) for i in ua] + [(None, j) for j in ub]


class Track:
    __slots__ = ("id", "x", "v", "miss", "n")

    def __init__(self, tid, xy):
        self.id, self.x, self.v, self.miss, self.n = tid, np.array(xy, float), np.zeros(2), 0, 1


def run(det_dir, raw_dir, calib_path, offset_ticks=0):
    calib = json.load(open(calib_path))
    pitch = calib["pitch"]
    clips = scan(raw_dir)
    chs = sorted(calib["cameras"])
    V = [load_camera_dets(det_dir, clips, aim_cameras(calib, ch), ch, pitch) for ch in chs]
    V[1]["k"] = V[1]["k"] + offset_ticks
    order = [np.argsort(v["k"], kind="stable") for v in V]
    for v, o in zip(V, order):
        for key in ("k", "xy", "h", "conf", "feat", "box", "rng"):
            v[key] = v[key][o]
    kmin = min(v["k"].min() for v in V if len(v["k"]))
    kmax = max(v["k"].max() for v in V if len(v["k"]))
    bounds = [np.searchsorted(v["k"], np.arange(kmin, kmax + 2)) for v in V]

    obs = dict(tid=[], k=[], xy=[], h=[], sig=[], i0=[], i1=[])
    tracks, next_id = [], 0
    for k in range(kmin, kmax + 1):
        views = []
        for v, bd in zip(V, bounds):
            s, e = bd[k - kmin], bd[k - kmin + 1]
            views.append(dict(xy=v["xy"][s:e], rng=v["rng"][s:e], h=v["h"][s:e], base=s))
        fused = []
        for i, j in merge_views(*views):
            if i is not None and j is not None:
                wa, wb = 1 / sigma(views[0]["rng"][i]) ** 2, 1 / sigma(views[1]["rng"][j]) ** 2
                xy = (views[0]["xy"][i] * wa + views[1]["xy"][j] * wb) / (wa + wb)
                h = np.nanmean([views[0]["h"][i], views[1]["h"][j]]) if not (np.isnan(views[0]["h"][i]) and np.isnan(views[1]["h"][j])) else np.nan
                fused.append((xy, h, (wa + wb) ** -0.5, views[0]["base"] + i, views[1]["base"] + j))
            elif i is not None:
                fused.append((views[0]["xy"][i], views[0]["h"][i], sigma(views[0]["rng"][i]), views[0]["base"] + i, -1))
            else:
                fused.append((views[1]["xy"][j], views[1]["h"][j], sigma(views[1]["rng"][j]), -1, views[1]["base"] + j))
        fused = [f for f in fused if -MARGIN < f[0][0] < pitch["L"] + MARGIN and -MARGIN < f[0][1] < pitch["W"] + MARGIN]
        # --- associate to tracks
        P = np.array([f[0] for f in fused]).reshape(-1, 2)
        if tracks and len(P):
            pred = np.array([t.x + t.v * DT for t in tracks])
            d = np.linalg.norm(pred[:, None] - P[None], axis=2)
            cost = np.where(d < GATE, d, 1e3)
            ri, ci = linear_sum_assignment(cost)
        else:
            ri, ci, cost = [], [], None
        used_t, used_d = set(), set()
        for i, j in zip(ri, ci):
            if cost[i, j] >= 1e3:
                continue
            t = tracks[i]
            newx = 0.6 * P[j] + 0.4 * (t.x + t.v * DT)
            t.v = 0.6 * t.v + 0.4 * (newx - t.x) / (DT * (1 + t.miss))
            t.x, t.miss, t.n = newx, 0, t.n + 1
            used_t.add(i); used_d.add(j)
            f = fused[j]
            obs["tid"].append(t.id); obs["k"].append(k); obs["xy"].append(newx.copy()); obs["h"].append(f[1])
            obs["sig"].append(f[2]); obs["i0"].append(f[3]); obs["i1"].append(f[4])
        for i, t in enumerate(tracks):
            if i not in used_t:
                t.miss += 1
                t.x = t.x + t.v * DT
                t.v *= 0.8
        tracks = [t for t in tracks if t.miss <= MAX_MISS]
        for j in range(len(P)):
            if j not in used_d:
                t = Track(next_id, P[j]); next_id += 1
                tracks.append(t)
                f = fused[j]
                obs["tid"].append(t.id); obs["k"].append(k); obs["xy"].append(P[j].copy()); obs["h"].append(f[1])
                obs["sig"].append(f[2]); obs["i0"].append(f[3]); obs["i1"].append(f[4])
        # --- crowding: when two live tracks come too close their identities may swap -> cut both
        live = [t for t in tracks if t.miss == 0]
        if len(live) > 1:
            X = np.array([t.x for t in live])
            D = np.linalg.norm(X[:, None] - X[None], axis=2) + np.eye(len(X)) * 1e3
            for i in np.where(D.min(1) < CROWD)[0]:
                if live[i].n > 3:  # restart under a new id; linking is decided later from appearance
                    live[i].id, live[i].n = next_id, 1
                    next_id += 1
    out = {k: np.array(v) for k, v in obs.items()}
    out["xy"] = out["xy"].reshape(-1, 2)
    return out, V, pitch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="data/det")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--calib", default="config/calibration.json")
    ap.add_argument("--out", default="data/tracks.npz")
    ap.add_argument("--offset-ticks", type=int, default=0, help="shift ch3 by this many ticks relative to ch2")
    a = ap.parse_args()
    obs, V, pitch = run(a.det, a.raw, a.calib, a.offset_ticks)
    np.savez_compressed(a.out, **obs,
                        feat0=V[0]["feat"], feat1=V[1]["feat"], box0=V[0]["box"], box1=V[1]["box"],
                        rng0=V[0]["rng"], rng1=V[1]["rng"], k0=V[0]["k"], k1=V[1]["k"],
                        ball0=V[0]["ball"], ball1=V[1]["ball"])
    tid, cnt = np.unique(obs["tid"], return_counts=True)
    print(f"{len(obs['k'])} observations, {len(tid)} tracklets, median length {np.median(cnt) * DT:.1f}s, "
          f">=10s: {(cnt >= 50).sum()}")


if __name__ == "__main__":
    main()
