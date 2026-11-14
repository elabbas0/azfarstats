"""Stage 1: run the person/ball detector over every clip of one camera at the analysis rate.

Usage: python -m azfar.detect --raw data/raw --out data/det --ch ch2 [--model data/models/yolo11s_openvino_model]
Writes one <clip>.npz per clip: k (grid tick), box (xyxy px), conf, cls (0 person, 32 ball), feat.
"""
from __future__ import annotations
import argparse, os, time
import numpy as np, cv2
from .timeline import scan
from .features import describe, DIM


def run_clip(model, clip, out_path, imgsz=(736, 1280)):
    want = dict((i, k) for k, i in clip.grid_frames())
    cap = cv2.VideoCapture(clip.path)
    K, B, C, CL, F = [], [], [], [], []
    i, t0, done = 0, time.time(), 0
    last = max(want)
    while i <= last:
        if not cap.grab():
            break
        if i in want:
            ok, im = cap.retrieve()
            if ok:
                r = model.predict(im, imgsz=imgsz, classes=[0, 32], conf=0.05, iou=0.6, verbose=False)[0].boxes
                box, conf, cls = r.xyxy.cpu().numpy(), r.conf.cpu().numpy(), r.cls.cpu().numpy().astype(np.int16)
                keep = (cls == 32) | (conf >= 0.15)
                for b, c, cl in zip(box[keep], conf[keep], cls[keep]):
                    K.append(want[i]); B.append(b); C.append(c); CL.append(cl)
                    F.append(describe(im, b) if cl == 0 else np.zeros(DIM, np.float32))
                done += 1
                if done % 2000 == 0:
                    print(f"{os.path.basename(clip.path)} {done}/{len(want)} {done / (time.time() - t0):.1f} it/s", flush=True)
        i += 1
    cap.release()
    np.savez_compressed(out_path, k=np.array(K, np.int64), box=np.array(B, np.float32).reshape(-1, 4),
                        conf=np.array(C, np.float32), cls=np.array(CL, np.int16),
                        feat=np.array(F, np.float16).reshape(-1, DIM))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/det")
    ap.add_argument("--ch", required=True)
    ap.add_argument("--model", default="data/models/yolo11s_openvino_model")
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(a.model, task="detect")
    os.makedirs(a.out, exist_ok=True)
    for clip in scan(a.raw):
        if clip.ch != a.ch:
            continue
        out = os.path.join(a.out, os.path.basename(clip.path).replace(".mp4", ".npz"))
        if os.path.exists(out):
            continue
        run_clip(model, clip, out + ".tmp.npz")
        os.replace(out + ".tmp.npz", out)
        print("done", out, flush=True)


if __name__ == "__main__":
    main()
