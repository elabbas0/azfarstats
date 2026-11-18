"""Run the dedicated ball detector on every frame of the match window, both cameras.

Writes data/balldet_out/<clip>.npz with f (frame), box (xyxy, full-resolution px), conf.
Usage: python -m azfar.balldet --model data/models/balldet_openvino_model [--only <clip substring>]
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np, cv2
from .ballmotion import BallMotion
from .timeline import scan

SCALE = 0.8
HMS = lambda s: sum(int(p) * m for p, m in zip(s.split(":"), (3600, 60, 1)))


def detect_frame(im, model, conf, motion):
    """Combine learned detections with weak motion evidence in original camera pixels."""
    h, w = im.shape[:2]
    width, height = int(1920 * SCALE), int(1080 * SCALE)
    small = cv2.resize(im, (width, height), interpolation=cv2.INTER_AREA)
    detections = model.predict(small, imgsz=(height // 32 * 32 + 32, width), conf=conf,
                               iou=0.5, max_det=6, verbose=False)[0].boxes
    boxes = detections.xyxy.cpu().numpy() * np.array([w / width, h / height] * 2, np.float32)
    scores = detections.conf.cpu().numpy()
    streaks, support = motion.detect(im)
    streaks, support = streaks[support >= conf], support[support >= conf]
    if len(boxes) and len(streaks):
        # Keep the stronger candidate when both describe the same ball. A long blurred box can have
        # low IoU with the compact diameter estimate at the centre of its streak.
        centres = (boxes[:, :2] + boxes[:, 2:]) / 2
        streak_centres = (streaks[:, :2] + streaks[:, 2:]) / 2
        distance = np.linalg.norm(streak_centres[:, None] - centres, axis=2)
        diameter = np.minimum((streaks[:, 2] - streaks[:, 0])[:, None],
                              np.min(boxes[:, 2:] - boxes[:, :2], axis=1))
        same_ball = distance <= diameter / 2
        keep_streak = ~np.any(same_ball & (scores >= support[:, None]), axis=1)
        keep_box = ~np.any(same_ball[keep_streak], axis=0)
        boxes, scores = boxes[keep_box], scores[keep_box]
        streaks, support = streaks[keep_streak], support[keep_streak]
    return np.concatenate((boxes, streaks)), np.concatenate((scores, support))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--model", default="data/models/balldet_openvino_model")
    ap.add_argument("--session", default="config/session.json")
    ap.add_argument("--out", default="data/balldet_out")
    ap.add_argument("--refresh", action="store_true", help="reprocess clips with existing detections")
    ap.add_argument("--only", default="")
    ap.add_argument("--conf", type=float, default=0.1)
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(a.model, task="detect")
    with open(a.session) as session:
        m = json.load(session)["matches"][0]
    lo, hi = HMS(m["start"]), HMS(m["end"])
    os.makedirs(a.out, exist_ok=True)
    for clip in scan(a.raw):
        name = os.path.basename(clip.path)[:-4]
        out = os.path.join(a.out, name + ".npz")
        if a.only not in name or (os.path.exists(out) and not a.refresh):
            continue
        f0 = max(0, int(np.ceil((lo - clip.t0) * 15))); f1 = min(clip.n_frames, int((hi - clip.t0) * 15) + 1)
        if f1 <= f0:
            continue
        motion = BallMotion()
        cap = cv2.VideoCapture(clip.path)
        for _ in range(f0):
            cap.grab()
        F, B, C = [], [], []
        t0 = time.time()
        for i in range(f0, f1):
            ok, im = cap.read()
            if not ok:
                cap.release()
                raise IOError(f"Could not decode frame {i} of {clip.path}")
            boxes, scores = detect_frame(im, model, a.conf, motion)
            for b, c in zip(boxes, scores):
                F.append(i); B.append(b); C.append(c)
            if (i - f0) % 3000 == 0 and i > f0:
                print(f"{name} {i - f0}/{f1 - f0} frames, {(i - f0) / (time.time() - t0):.1f} fps", flush=True)
        cap.release()
        np.savez_compressed(out + ".tmp.npz", f=np.array(F, np.int32), box=np.array(B, np.float32).reshape(-1, 4), conf=np.array(C, np.float32))
        os.replace(out + ".tmp.npz", out)
        print("done", name, flush=True)


if __name__ == "__main__":
    main()
