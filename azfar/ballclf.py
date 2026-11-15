"""Stage 1e: learn what the match ball looks like from this recording, then score every candidate.

Training labels come from the footage itself, inside the match window (one ball in play):
  ball      - confident detector hits whose colour is ball-like, and motion blobs on top of them
  not ball  - every other motion blob in those same frames (it cannot be the ball: the ball is elsewhere)
A small CNN on 24x24 patches (azfar.ballpatch) is trained on the earlier part of the match and checked
on the later part, then scores every blob and detector hit.

Usage: python -m azfar.ballclf
Writes data/models/ballclf.pt and data/ballpatch/<clip>_pblob.npy, <clip>_pdet.npy.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, torch, torch.nn as nn
from .timeline import scan, GRID_HZ, FPS

HMS = lambda s: sum(int(p) * m for p, m in zip(s.split(":"), (3600, 60, 1)))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        c = lambda i, o, s: nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.f = nn.Sequential(c(3, 24, 1), c(24, 32, 2), c(32, 48, 2), c(48, 64, 2), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 1))

    def forward(self, x):
        return self.f(x).squeeze(1)


def prep(p):
    return torch.from_numpy(p).permute(0, 3, 1, 2).float().div_(255.0).sub_(0.5)


def colour_ok(col):
    sat, val, yel = col[:, 0], col[:, 1], col[:, 2]
    return (val > 0) & (sat <= 88) & ~((yel > 0.22) & (sat > 65)) & (val >= 110)


def labels(clip, scan_dir, det_dir, patch_dir, lo, hi):
    name = os.path.basename(clip.path).replace(".mp4", "")
    s = np.load(os.path.join(scan_dir, name + ".npz")); d = np.load(os.path.join(det_dir, name + ".npz"))
    sel = np.flatnonzero(d["cls"] == 32)
    fm = dict(clip.grid_frames())
    dk = d["k"][sel]; df = np.array([fm.get(int(k), -1) for k in dk])
    conf = d["conf"][sel]; box = d["box"][sel]
    dcol = s["det_col"][sel] if "det_col" in s.files else np.zeros((len(sel), 3))
    in_win = (dk >= lo * GRID_HZ) & (dk < hi * GRID_HZ)
    pos_det = np.flatnonzero(in_win & (conf >= 0.55) & colour_ok(dcol))
    # frames with exactly one confident ball
    good_f = {}
    for r in pos_det:
        good_f.setdefault(int(df[r]), []).append(r)
    good_f = {f: rs[0] for f, rs in good_f.items() if len(rs) == 1 and f >= 0}
    bf, bxy, bsz = s["f"], s["xy"], s["size"]
    pos_blob, neg_blob = [], []
    for i in np.flatnonzero(np.isin(bf, list(good_f))):
        r = good_f[int(bf[i])]
        x1, y1, x2, y2 = box[r]
        dist = np.hypot(bxy[i, 0] - (x1 + x2) / 2, bxy[i, 1] - (y1 + y2) / 2)
        if dist < max(8.0, 0.6 * (x2 - x1)):
            pos_blob.append(i)
        elif dist > 80:
            neg_blob.append(i)
    t = lambda frames: clip.frame_time(np.asarray(frames))
    return dict(name=name, pos_det=pos_det, pos_blob=np.array(pos_blob, int), neg_blob=np.array(neg_blob, int),
                t_det=dk / GRID_HZ, t_blob=t(bf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--det", default="data/det")
    ap.add_argument("--scan", default="data/ballscan")
    ap.add_argument("--patch", default="data/ballpatch")
    ap.add_argument("--session", default="config/session.json")
    ap.add_argument("--model", default="data/models/ballclf.pt")
    ap.add_argument("--epochs", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(4); torch.manual_seed(0)
    m = json.load(open(a.session))["matches"][0]
    lo, hi = HMS(m["start"]), HMS(m["end"])
    split = lo + 0.75 * (hi - lo)
    clips = [c for c in scan(a.raw) if os.path.exists(os.path.join(a.patch, os.path.basename(c.path).replace(".mp4", "_blob.npy")))]
    Xtr, Ytr, Xva, Yva = [], [], [], []
    rng = np.random.default_rng(0)
    for c in clips:
        L = labels(c, a.scan, a.det, a.patch, lo, hi)
        PB = np.load(os.path.join(a.patch, L["name"] + "_blob.npy"), mmap_mode="r")
        PD = np.load(os.path.join(a.patch, L["name"] + "_det.npy"))
        neg = L["neg_blob"]
        if len(neg) > 6 * (len(L["pos_det"]) + len(L["pos_blob"])):
            neg = rng.choice(neg, 6 * (len(L["pos_det"]) + len(L["pos_blob"])), replace=False)
        for idx, src, tt, y in ((L["pos_det"], PD, L["t_det"], 1), (L["pos_blob"], PB, L["t_blob"], 1), (np.sort(neg), PB, L["t_blob"], 0)):
            if not len(idx):
                continue
            X = np.asarray(src[idx]); tr = tt[idx] < split
            Xtr.append(X[tr]); Ytr.append(np.full(tr.sum(), y)); Xva.append(X[~tr]); Yva.append(np.full((~tr).sum(), y))
        print(f"{L['name']}: {len(L['pos_det'])} ball (detector) + {len(L['pos_blob'])} ball (motion), {len(neg)} not-ball", flush=True)
    Xtr, Ytr, Xva, Yva = np.concatenate(Xtr), np.concatenate(Ytr), np.concatenate(Xva), np.concatenate(Yva)
    print(f"train {len(Ytr)} ({Ytr.mean() * 100:.0f}% ball), check {len(Yva)} ({Yva.mean() * 100:.0f}% ball)")
    net = Net(); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    pw = torch.tensor((1 - Ytr.mean()) / Ytr.mean())
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    for ep in range(a.epochs):
        net.train(); perm = rng.permutation(len(Ytr)); tot = 0
        for s0 in range(0, len(perm), 256):
            b = perm[s0:s0 + 256]
            x = prep(Xtr[b])
            if rng.random() < 0.5:
                x = x.flip(3)
            x = x * float(rng.uniform(0.8, 1.2)) + float(rng.uniform(-0.08, 0.08))
            loss = lossf(net(x), torch.from_numpy(Ytr[b]).float())
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        net.eval()
        with torch.no_grad():
            pv = torch.cat([torch.sigmoid(net(prep(Xva[i:i + 2048]))) for i in range(0, len(Yva), 2048)]).numpy()
        order = np.argsort(-pv); tp = np.cumsum(Yva[order]); auc = (np.argsort(np.argsort(pv))[Yva == 1].sum() - Yva.sum() * (Yva.sum() - 1) / 2) / (Yva.sum() * (len(Yva) - Yva.sum()))
        rec = ((pv > 0.5) & (Yva == 1)).sum() / Yva.sum(); prec = ((pv > 0.5) & (Yva == 1)).sum() / max((pv > 0.5).sum(), 1)
        print(f"epoch {ep}: loss {tot / len(Ytr):.3f}; later-match check: AUC {auc:.3f}, at 0.5 finds {rec * 100:.0f}% of balls, {prec * 100:.0f}% of its 'ball' calls are right", flush=True)
    torch.save(net.state_dict(), a.model)
    with torch.no_grad():
        for c in clips:
            name = os.path.basename(c.path).replace(".mp4", "")
            for kind in ("blob", "det"):
                X = np.load(os.path.join(a.patch, f"{name}_{kind}.npy"), mmap_mode="r")
                p = np.concatenate([torch.sigmoid(net(prep(np.asarray(X[i:i + 4096])))).numpy() for i in range(0, len(X), 4096)]) if len(X) else np.zeros(0)
                np.save(os.path.join(a.patch, f"{name}_p{kind}.npy"), p.astype(np.float32))
            print("scored", name, flush=True)


if __name__ == "__main__":
    main()
