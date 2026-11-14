"""Stage 7: split-view video (both cameras side by side on the computed time alignment), cut into
small web-friendly segments so the replay page can play the whole match next to the top-down view.

Usage: python -m azfar.video --out data/out/video [--start 18:11:00 --end 20:03:50] [--seconds 300]
Writes seg_000.mp4 ... and segments.json ({src, start (s since midnight), dur}).
"""
from __future__ import annotations
import argparse, json, os, subprocess
from .timeline import scan, FPS
from .frames import _pts

HMS = lambda s: sum(int(p) * m for p, m in zip(s.split(":"), (3600, 60, 1)))


def locate(clips, ch, t):
    for c in clips:
        if c.ch == ch:
            i = round((t - c.t0) * FPS)
            if 0 <= i < c.n_frames:
                return c, i
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--session", default="config/session.json")
    ap.add_argument("--out", default="data/out/video")
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--seconds", type=int, default=300)
    ap.add_argument("--kbps", type=int, default=215)
    ap.add_argument("--only", type=int, default=-1, help="encode just this segment number")
    a = ap.parse_args()
    clips = scan(a.raw)
    chs = sorted({c.ch for c in clips})
    sess = json.load(open(a.session)) if os.path.exists(a.session) else {}
    m = (sess.get("matches") or [{}])[0]
    t0, t1 = HMS(a.start or m.get("start", "00:00:00")), HMS(a.end or m.get("end", "23:59:59"))
    # both cameras must have footage; never cross a file boundary inside a segment
    for ch in chs:
        cc = [c for c in clips if c.ch == ch]
        t0 = max(t0, min(c.t0 for c in cc) + 1); t1 = min(t1, max(c.frame_time(c.n_frames - 1) for c in cc) - 1)
    cuts = sorted({t0, t1} | {round(c.t0) for c in clips if t0 < c.t0 < t1})
    bounds = []
    for a_, b_ in zip(cuts[:-1], cuts[1:]):
        t = a_
        while t < b_:
            bounds.append((t, min(t + a.seconds, b_))); t += a.seconds
    os.makedirs(a.out, exist_ok=True)
    segs = []
    for n, (s, e) in enumerate(bounds):
        name = f"seg_{n:03d}.mp4"
        segs.append(dict(src=name, start=s, dur=e - s))
        path = os.path.join(a.out, name)
        if (a.only >= 0 and n != a.only) or os.path.exists(path):
            continue
        nfr = int((e - s) * FPS)
        cmd = ["ffmpeg", "-v", "error", "-y"]
        for ch in chs:
            c, i = locate(clips, ch, s)
            cmd += ["-ss", f"{max(_pts(c.path)[i] - 0.001, 0):.4f}", "-i", c.path]
        cmd += ["-filter_complex",
                f"[0:v]setpts=N/({FPS}*TB),scale=640:360[a];[1:v]setpts=N/({FPS}*TB),scale=640:360[b];[a][b]hstack=inputs=2[v]",
                "-map", "[v]", "-frames:v", str(nfr), "-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                "-b:v", f"{a.kbps}k", "-maxrate", f"{int(a.kbps * 1.25)}k", "-bufsize", f"{a.kbps * 4}k", "-g", "45",
                "-movflags", "+faststart", "-an", path + ".tmp.mp4"]
        subprocess.run(cmd, check=True)
        os.replace(path + ".tmp.mp4", path)
        print(name, f"{os.path.getsize(path) / 1e6:.1f} MB", flush=True)
    json.dump(segs, open(os.path.join(a.out, "segments.json"), "w"))
    print(len(segs), "segments")


if __name__ == "__main__":
    main()
