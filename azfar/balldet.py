"""Run the dedicated ball detector on every frame of the match window, both cameras.

Writes data/balldet_out/<clip>.npz with f (frame), box (xyxy, full-resolution px), conf.
Usage: python -m azfar.balldet --model data/models/balldet_openvino_model [--only <clip substring>]
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np, cv2
from .timeline import scan

SCALE = 0.8
HMS = lambda s: sum(int(p) * m for p, m in zip(s.split(":"), (3600, 60, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--model", default="data/models/balldet_openvino_model")
    ap.add_argument("--session", default="config/session.json")
    ap.add_argument("--out", default="data/balldet_out")
    ap.add_argument("--only", default="")
    ap.add_argument("--conf", type=float, default=0.1)
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(a.model, task="detect")
    m = json.load(open(a.session))["matches"][0]
    lo, hi = HMS(m["start"]), HMS(m["end"])
    os.makedirs(a.out, exist_ok=True)
    W, H = int(1920 * SCALE), int(1080 * SCALE)
    for clip in scan(a.raw):
        name = os.path.basename(clip.path)[:-4]
        out = os.path.join(a.out, name + ".npz")
        if a.only not in name or os.path.exists(out):
            continue
        f0 = max(0, int(np.ceil((lo - clip.t0) * 15))); f1 = min(clip.n_frames, int((hi - clip.t0) * 15) + 1)
        if f1 <= f0:
            continue
        cap = cv2.VideoCapture(clip.path)
        for _ in range(f0):
            cap.grab()
        F, B, C = [], [], []
        t0 = time.time()
        for i in range(f0, f1):
            ok, im = cap.read()
            if not ok:
                break
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
            r = model.predict(im, imgsz=(H // 32 * 32 + 32, W), conf=a.conf, iou=0.5, max_det=6, verbose=False)[0].boxes
            for b, c in zip(r.xyxy.cpu().numpy() / SCALE, r.conf.cpu().numpy()):
                F.append(i); B.append(b); C.append(c)
            if (i - f0) % 3000 == 0 and i > f0:
                print(f"{name} {i - f0}/{f1 - f0} frames, {(i - f0) / (time.time() - t0):.1f} fps", flush=True)
        cap.release()
        np.savez_compressed(out + ".tmp.npz", f=np.array(F, np.int32), box=np.array(B, np.float32).reshape(-1, 4), conf=np.array(C, np.float32))
        os.replace(out + ".tmp.npz", out)
        print("done", name, flush=True)


if __name__ == "__main__":
    main()
