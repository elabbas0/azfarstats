"""Find when the match itself starts and ends.

Before kick-off people warm up with several balls at once; during the match there is one. We count,
per minute, how often the detector sees two confident balls far apart, and how many people are on
the pitch. The match starts after the last warm-up minute near the beginning of the recording and
ends when the pitch empties. Writes the window into config/session.json (keeps names and teams).

Usage: python -m azfar.window [--write]
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from .camera import Camera
from .timeline import scan, GRID_HZ, fmt


def per_minute(det_dir, raw_dir, calib):
    multi, balls = {}, {}
    for c in scan(raw_dir):
        p = os.path.join(det_dir, os.path.basename(c.path).replace(".mp4", ".npz"))
        if not os.path.exists(p):
            continue
        cam = Camera.from_dict(calib["cameras"][c.ch])
        d = np.load(p)
        m = (d["cls"] == 32) & (d["conf"] > 0.35)
        b, k = d["box"][m], d["k"][m]
        g, ok = cam.to_ground(np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], height=0.11)
        for kk in np.unique(k):
            s = np.flatnonzero((k == kk) & ok)
            mn = int(kk // (60 * GRID_HZ))
            balls[mn] = balls.get(mn, 0) + 1
            if len(s) >= 2 and np.linalg.norm(g[s][:, None] - g[s][None], axis=2).max() > 6:
                multi[mn] = multi.get(mn, 0) + 1
    return multi, balls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="data/det")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--tracks", default="data/tracks.npz")
    ap.add_argument("--calib", default="config/calibration.json")
    ap.add_argument("--session", default="config/session.json")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    calib = json.load(open(a.calib))
    multi, balls = per_minute(a.det, a.raw, calib)
    T = np.load(a.tracks)
    u, c = np.unique(T["k"], return_counts=True)
    people = {}
    for mn in np.unique(u // (60 * GRID_HZ)):
        people[int(mn)] = float(np.median(c[u // (60 * GRID_HZ) == mn]))
    mins = sorted(set(people) | set(balls))
    busy = [mn for mn in mins if people.get(mn, 0) >= 8 and balls.get(mn, 0) >= 30]
    first, last = busy[0], busy[-1]
    warm = [mn for mn in mins if first <= mn <= first + 15 and multi.get(mn, 0) >= 10]
    start = (warm[-1] + 1) if warm else first
    end = last + 1
    # both cameras are needed: stop where the first one stops recording
    clips = scan(a.raw)
    cam_end = min(max(c.frame_time(c.n_frames - 1) for c in clips if c.ch == ch) for ch in {c.ch for c in clips})
    end = min(end, int(cam_end // 60))
    print(f"activity {fmt(first * 60)}-{fmt(end * 60)}; warm-up minutes (several balls): {[fmt(m * 60) for m in warm]}")
    print(f"match window: {fmt(start * 60)} - {fmt(end * 60)}")
    if a.write:
        sess = json.load(open(a.session)) if os.path.exists(a.session) else {}
        sess["matches"] = [{"name": "Match", "start": fmt(start * 60), "end": fmt(end * 60)}]
        json.dump(sess, open(a.session, "w"), indent=1)
        print("written to", a.session)


if __name__ == "__main__":
    main()
