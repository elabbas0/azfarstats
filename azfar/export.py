"""Stage 6: build the self-contained replay + stats page.

Usage: python -m azfar.export --out data/out/replay.html
"""
from __future__ import annotations
import argparse, base64, json, os
import numpy as np, cv2
from .timeline import scan, GRID_HZ, fmt
from .events import grid_tracks, grid_ball
from .frames import grab


def segs(a):
    """[ticks, 2] float array with nan gaps -> [[start, [x, y, ... in decimetres]]]."""
    ok = ~np.isnan(a[:, 0])
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    return [[int(s), np.round(a[s:e] * 10).astype(int).ravel().tolist()] for s, e in zip(edges[::2], edges[1::2])]


def thumbnail(I, T, pid, clips, height=104):
    """Best single view of a player: big, confident, alone in the frame."""
    m = np.flatnonzero(I["pid"] == pid)
    best = None
    for c, key in enumerate(("i0", "i1")):
        idx = I[key][m]
        ok = idx >= 0
        b = T[f"box{c}"][idx[ok]]
        h = b[:, 3] - b[:, 1]
        aspect = h / np.maximum(b[:, 2] - b[:, 0], 1)
        score = np.where((aspect > 2.0) & (aspect < 3.6) & (h < 420), h, 0)
        for j in np.argsort(-score)[:1]:
            if score[j] > 0 and (best is None or score[j] > best[0]):
                best = (score[j], c, int(I["k"][m][ok][j]), b[j])
    if best is None:
        return None
    _, c, k, box = best
    ch = sorted({cl.ch for cl in clips})[c]
    for cl in clips:
        if cl.ch != ch:
            continue
        fi = dict(cl.grid_frames()).get(k)
        if fi is None:
            continue
        im = grab(cl.path, fi)
        if im is None:
            return None
        x1, y1, x2, y2 = box
        pw, ph = (x2 - x1) * 0.15, (y2 - y1) * 0.06
        crop = im[max(int(y1 - ph), 0):int(y2 + ph), max(int(x1 - pw), 0):int(x2 + pw)]
        if crop.size == 0:
            return None
        crop = cv2.resize(crop, (height // 2, height), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="data/identities.npz")
    ap.add_argument("--tracks", default="data/tracks.npz")
    ap.add_argument("--ball", default="data/ball.npz")
    ap.add_argument("--stats", default="data/out/players.json")
    ap.add_argument("--events", default="data/out/events.json")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--session", default="config/session.json")
    ap.add_argument("--out", default="data/out/replay.html")
    ap.add_argument("--no-thumbs", action="store_true")
    ap.add_argument("--video", default="data/out/video", help="folder with split-view segments (see azfar.video)")
    a = ap.parse_args()
    I, T, B = np.load(a.ids), np.load(a.tracks), np.load(a.ball)
    stats, ev = json.load(open(a.stats)), json.load(open(a.events))
    sess = json.load(open(a.session)) if os.path.exists(a.session) else {}
    names = sess.get("names", {})
    clips = scan(a.raw)
    k0, k1 = int(I["k"].min()), int(I["k"].max()) + 1
    pids, P = grid_tracks(I, k0, k1)
    ball = grid_ball(B, k0, k1)
    hms = lambda s: sum(int(p) * m for p, m in zip(s.split(":"), (3600, 60, 1)))
    wins = {w: (hms(v[0]) * GRID_HZ, hms(v[1]) * GRID_HZ + GRID_HZ) for w, v in stats["windows"].items()}
    by = {p["id"]: p for p in stats["players"]}
    players = []
    for n, pid in enumerate(pids):
        pid = int(pid)
        if pid not in by:
            continue
        s = by[pid]
        st = {}
        for w, sw in s["stats"].items():
            evc = {}
            for e in ev["events"]:
                if e.get("player") == pid and wins[w][0] <= e["k"] < wins[w][1]:
                    evc[e["type"]] = evc.get(e["type"], 0) + 1
            st[w] = {k_: v for k_, v in sw.items() if k_ not in ("zone_m",)} | {"ev": evc}
        c = s["colours"]
        desc = f"{c.get('torso', {}).get('name', '?')} top, {c.get('shorts', {}).get('name', '?')} shorts"
        if s["height_m"] == s["height_m"]:
            desc += f", ~{s['height_m']:.2f} m"
        players.append(dict(id=pid, label=str(pid + 1), name=names.get(str(pid), f"Player {pid + 1}"), desc=desc,
                            team=ev["teams"].get(str(pid), ev["teams"].get(pid, -1)), hex=c.get("torso", {}).get("hex", "#888888"),
                            segs=segs(P[n]), stats=st, thumb=None if a.no_thumbs else thumbnail(I, T, pid, clips)))
    players.sort(key=lambda p: -p["stats"]["all"]["minutes"])
    day = clips[0].start.strftime("%d %b %Y")
    data = dict(title=f"{day}, {fmt(k0 / GRID_HZ)}–{fmt(k1 / GRID_HZ)} · {len(players)} players tracked · pitch {stats['pitch']['W']:.0f} × {stats['pitch']['L']:.1f} m",
                note=sess.get("note", "Positions come from two corner cameras, five samples a second. Distances and speeds are measured; "
                              "passes, shots, goals and the other ball events are estimates from ball and player positions - click any event to check it in the replay."),
                pitch=stats["pitch"], hz=GRID_HZ, n=k1 - k0, k0=k0, t0=k0 / GRID_HZ, windows=stats["windows"],
                players=players, ball=segs(ball), events=[e for e in ev["events"] if e["type"] != "touch"])
    seg_file = os.path.join(a.video, "segments.json")
    if os.path.exists(seg_file):
        have = [s for s in json.load(open(seg_file)) if os.path.exists(os.path.join(a.video, s["src"]))]
        if have:
            data["video"] = dict(base="video/", segments=have)
    tpl = open(os.path.join(os.path.dirname(__file__), "replay_template.html")).read()
    html = tpl.replace("/*DATA*/null/*END*/", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write(html)
    print(f"wrote {a.out} ({len(html) / 1e6:.1f} MB), {len(players)} players, {len(data['events'])} events")


if __name__ == "__main__":
    main()
