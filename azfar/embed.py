"""Stage 1b: appearance embeddings (OSNet re-identification features) for detected players.

Once per second, every confident, reasonably large player box is cropped and embedded. Output per
clip: idx (row numbers into the matching data/det/<clip>.npz) and emb (L2-normalised, float16).

Usage: python -m azfar.embed --raw data/raw --det data/det --out data/emb [--only <substring>]
"""
from __future__ import annotations
import argparse, os, time
import numpy as np, cv2, torch
from .timeline import scan, GRID_HZ
from . import osnet

EVERY = int(GRID_HZ)      # ticks between embedded frames (1 s)
MIN_H, MIN_CONF = 40, 0.5
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


def crop_batch(im, boxes):
    out = np.zeros((len(boxes), 3, 256, 128), np.float32)
    H, W = im.shape[:2]
    for n, (x1, y1, x2, y2) in enumerate(boxes):
        x1, y1, x2, y2 = max(int(x1), 0), max(int(y1), 0), min(int(np.ceil(x2)), W), min(int(np.ceil(y2)), H)
        c = cv2.resize(im[y1:y2, x1:x2], (128, 256), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
        out[n] = ((c - MEAN) / STD).transpose(2, 0, 1)
    return out


def run_clip(model, clip, det_path, out_path):
    d = np.load(det_path)
    box, k = d["box"], d["k"]
    sel = (d["cls"] == 0) & (d["conf"] >= MIN_CONF) & ((box[:, 3] - box[:, 1]) >= MIN_H) & (k % EVERY == 0)
    rows = np.flatnonzero(sel)
    frame_of = {kk: i for kk, i in clip.grid_frames()}
    by_frame = {}
    for r in rows:
        by_frame.setdefault(frame_of[int(k[r])], []).append(r)
    cap = cv2.VideoCapture(clip.path)
    idx, emb, pend_rows, pend = [], [], [], []
    last = max(by_frame) if by_frame else -1

    def flush():
        if pend:
            with torch.no_grad():
                e = model(torch.from_numpy(np.concatenate(pend))).numpy()
            e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-9
            emb.append(e.astype(np.float16)); idx.extend(pend_rows)
            pend.clear(); pend_rows.clear()
    i, t0, n = 0, time.time(), 0
    while i <= last:
        if not cap.grab():
            break
        if i in by_frame:
            ok, im = cap.retrieve()
            if ok:
                pend.append(crop_batch(im, box[by_frame[i]])); pend_rows.extend(by_frame[i])
                if sum(len(p) for p in pend) >= 96:
                    flush()
                n += 1
                if n % 500 == 0:
                    print(f"{os.path.basename(clip.path)} {n}/{len(by_frame)} frames {n / (time.time() - t0):.1f}/s", flush=True)
        i += 1
    flush()
    cap.release()
    np.savez_compressed(out_path, idx=np.array(idx, np.int64), emb=np.concatenate(emb) if emb else np.zeros((0, 512), np.float16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--det", default="data/det")
    ap.add_argument("--out", default="data/emb")
    ap.add_argument("--weights", default="data/models/osnet_x0_5_msmt17.pth")
    ap.add_argument("--width", default="x0_5")
    ap.add_argument("--only", default="")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cv2.setNumThreads(a.threads)
    model = osnet.load(a.weights, a.width)
    os.makedirs(a.out, exist_ok=True)
    for clip in scan(a.raw):
        name = os.path.basename(clip.path).replace(".mp4", ".npz")
        if a.only not in name or not os.path.exists(os.path.join(a.det, name)) or os.path.exists(os.path.join(a.out, name)):
            continue
        run_clip(model, clip, os.path.join(a.det, name), os.path.join(a.out, name + ".tmp.npz"))
        os.replace(os.path.join(a.out, name + ".tmp.npz"), os.path.join(a.out, name))
        print("done", name, flush=True)


if __name__ == "__main__":
    main()
