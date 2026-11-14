"""Stage 3: turn tracklets into player identities.

1. summarise every tracklet (time span, end points, height, clothing colours per camera)
2. link tracklets across short gaps when the continuation is unambiguous
3. agglomerative clustering of the resulting chains on appearance + height, with
   cannot-link constraints (two chains that overlap in time, or that would need
   impossible running speed to be the same person, are different people)

Usage: python -m azfar.identify --tracks data/tracks.npz --out data/identities.npz
"""
from __future__ import annotations
import argparse, json
import numpy as np
from .timeline import GRID_HZ
from .features import DIM, DIM_REGION, AB_BINS, L_BINS

DT = 1.0 / GRID_HZ
VMAX = 8.5            # m/s, nobody on a 5-a-side pitch is faster
NH = AB_BINS**2
REGION_W = np.array([0.5, 0.3, 0.2])  # torso, shorts, legs


def app_dist(Fa, Fb):
    """Clothing distance between descriptor arrays (..., DIM). 0 = identical, ~1 = unrelated colours."""
    d = 0.0
    for r in range(3):
        o = r * DIM_REGION
        hab = np.abs(Fa[..., o:o + NH] - Fb[..., o:o + NH]).sum(-1) / 2
        hl = np.abs(Fa[..., o + NH:o + NH + L_BINS] - Fb[..., o + NH:o + NH + L_BINS]).sum(-1) / 2
        d = d + REGION_W[r] * (0.65 * hab + 0.35 * hl)
    return d


def summarise(T, min_len=5):
    """Per-tracklet summary arrays."""
    order = np.lexsort((T["k"], T["tid"]))
    tid, k, xy, h, sig, i0, i1 = (T[n][order] for n in ("tid", "k", "xy", "h", "sig", "i0", "i1"))
    cut = np.flatnonzero(np.diff(tid)) + 1
    starts, ends = np.r_[0, cut], np.r_[cut, len(tid)]
    S = dict(tid=[], ks=[], ke=[], ps=[], pe=[], vs=[], ve=[], n=[], h=[], hw=[], f0=[], f1=[], n0=[], n1=[], s=[], e=[])
    feats = (T["feat0"].astype(np.float32), T["feat1"].astype(np.float32))
    rngs = (T["rng0"], T["rng1"])
    boxes = (T["box0"], T["box1"])
    for s, e in zip(starts, ends):
        if e - s < min_len:
            continue
        m = min(5, e - s - 1)
        S["tid"].append(tid[s]); S["ks"].append(k[s]); S["ke"].append(k[e - 1]); S["n"].append(e - s)
        S["ps"].append(xy[s]); S["pe"].append(xy[e - 1])
        S["vs"].append((xy[s + m] - xy[s]) / ((k[s + m] - k[s]) * DT)); S["ve"].append((xy[e - 1] - xy[e - 1 - m]) / ((k[e - 1] - k[e - 1 - m]) * DT))
        good = sig[s:e] < 0.8
        S["h"].append(np.median(h[s:e][good]) if good.sum() >= 3 else np.nan); S["hw"].append(good.sum())
        for c, idx in enumerate((i0[s:e], i1[s:e])):
            idx = idx[idx >= 0]
            if len(idx):
                bh = boxes[c][idx, 3] - boxes[c][idx, 1]
                idx = idx[(rngs[c][idx] < 30) & (bh > 45) & (feats[c][idx].sum(1) > 1.5)]
            S[f"n{c}"].append(len(idx))
            S[f"f{c}"].append(feats[c][idx].mean(0) if len(idx) else np.zeros(DIM, np.float32))
        S["s"].append(s); S["e"].append(e)
    S = {k_: np.array(v) for k_, v in S.items()}
    return S, dict(tid=tid, k=k, xy=xy, h=h, sig=sig, i0=i0, i1=i1)


def pair_app(S, ia, ib, min_n=4):
    """Appearance distance between summaries ia and ib (arrays of indices); nan when not comparable."""
    num = np.zeros(len(ia)); den = np.zeros(len(ia))
    for c in (0, 1):
        ok = (S[f"n{c}"][ia] >= min_n) & (S[f"n{c}"][ib] >= min_n)
        d = app_dist(S[f"f{c}"][ia], S[f"f{c}"][ib])
        w = np.sqrt(np.minimum(S[f"n{c}"][ia], S[f"n{c}"][ib])) * ok
        num += w * d; den += w
    return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)


class Chains:
    """Union of tracklets believed to be one person."""

    def __init__(self, S):
        self.S = S
        self.members = [[i] for i in range(len(S["tid"]))]

    def link_short_gaps(self, max_gap=15, app_max=0.30, ratio=0.6):
        S = self.S
        order_s = np.argsort(S["ks"])
        ks_sorted = S["ks"][order_s]
        cand = []
        for a in range(len(S["tid"])):
            lo, hi = np.searchsorted(ks_sorted, [S["ke"][a] + 1, S["ke"][a] + max_gap + 1])
            bs = order_s[lo:hi]
            if not len(bs):
                continue
            gap = (S["ks"][bs] - S["ke"][a]) * DT
            pred = S["pe"][a] + S["ve"][a] * np.minimum(gap, 0.6)[:, None]
            dist = np.linalg.norm(S["ps"][bs] - pred, axis=1)
            reach = 1.0 + 4.5 * gap
            ok = dist < reach
            if not ok.any():
                continue
            app = pair_app(S, np.full(ok.sum(), a), bs[ok])
            cost = dist[ok] / reach[ok] + np.where(np.isnan(app), 0.25, app) / app_max
            for b, c, ap in zip(bs[ok], cost, app):
                if np.isnan(ap) or ap < app_max * 1.5:
                    cand.append((c, a, b))
        cand.sort()
        best_a, best_b = {}, {}
        for c, a, b in cand:  # best and second-best cost for each end
            best_a.setdefault(a, []).append(c); best_b.setdefault(b, []).append(c)
        nxt, prv = {}, {}
        for c, a, b in cand:
            if a in nxt or b in prv:
                continue
            ra, rb = best_a[a], best_b[b]
            # unambiguous: clearly better than the runner-up at both ends
            if (len(ra) > 1 and c > ratio * ra[1] and c != ra[0]) or (len(rb) > 1 and c > ratio * rb[1] and c != rb[0]):
                continue
            if (len(ra) > 1 and ra[0] > ratio * ra[1]) or (len(rb) > 1 and rb[0] > ratio * rb[1]):
                continue
            nxt[a] = b; prv[b] = a
        chains = []
        for i in range(len(S["tid"])):
            if i in prv:
                continue
            ch = [i]
            while ch[-1] in nxt:
                ch.append(nxt[ch[-1]])
            chains.append(ch)
        self.members = chains
        return len(nxt)


def chain_summary(S, members):
    """Aggregate tracklet summaries per chain."""
    n = len(members)
    C = dict(ks=np.zeros(n, int), ke=np.zeros(n, int), n=np.zeros(n, int), h=np.full(n, np.nan), hw=np.zeros(n),
             f0=np.zeros((n, DIM), np.float32), f1=np.zeros((n, DIM), np.float32), n0=np.zeros(n), n1=np.zeros(n))
    for j, m in enumerate(members):
        m = np.array(m)
        C["ks"][j], C["ke"][j], C["n"][j] = S["ks"][m].min(), S["ke"][m].max(), S["n"][m].sum()
        w = S["hw"][m] * ~np.isnan(S["h"][m])
        if w.sum() > 0:
            C["h"][j] = np.nansum(S["h"][m] * w) / w.sum(); C["hw"][j] = w.sum()
        for c in (0, 1):
            w = S[f"n{c}"][m].astype(float)
            C[f"n{c}"][j] = w.sum()
            if w.sum() > 0:
                C[f"f{c}"][j] = (S[f"f{c}"][m] * w[:, None]).sum(0) / w.sum()
    return C


def cluster(S, members, min_chain=25, stop=0.20, h_scale=0.12):
    """Constrained average-linkage clustering of chains. Returns label per chain (-1 = too short)."""
    C = chain_summary(S, members)
    big = np.flatnonzero((C["n"] >= min_chain) & ((C["n0"] >= 5) | (C["n1"] >= 5)))
    n = len(big)
    print(f"clustering {n} chains (of {len(members)})")
    # occupancy intervals per chain (list of tracklet spans) for the cannot-link test
    spans = [np.array([(S["ks"][i], S["ke"][i], *S["ps"][i], *S["pe"][i]) for i in members[j]]) for j in big]

    def conflict(sa, sb):
        # overlap in time, or unreachable between consecutive pieces
        for a in sa:
            ov = (sb[:, 0] <= a[1]) & (sb[:, 1] >= a[0])
            if ov.any():
                return True
            aft = sb[sb[:, 0] > a[1]]
            if len(aft):
                b = aft[np.argmin(aft[:, 0])]
                gap = (b[0] - a[1]) * DT
                if np.hypot(b[2] - a[4], b[3] - a[5]) > 2.0 + VMAX * gap:
                    return True
            bef = sb[sb[:, 1] < a[0]]
            if len(bef):
                b = bef[np.argmax(bef[:, 1])]
                gap = (a[0] - b[1]) * DT
                if np.hypot(a[2] - b[4], a[3] - b[5]) > 2.0 + VMAX * gap:
                    return True
        return False

    ii, jj = np.triu_indices(n, 1)
    D = np.full((n, n), np.inf, np.float32)
    num = np.zeros(len(ii)); den = np.zeros(len(ii))
    for c in (0, 1):
        ok = (C[f"n{c}"][big][ii] >= 5) & (C[f"n{c}"][big][jj] >= 5)
        d = app_dist(C[f"f{c}"][big][ii], C[f"f{c}"][big][jj])
        w = np.sqrt(np.minimum(C[f"n{c}"][big][ii], C[f"n{c}"][big][jj])) * ok
        num += w * d; den += w
    d = np.where(den > 0, num / np.maximum(den, 1e-9), np.inf)
    dh = np.abs(C["h"][big][ii] - C["h"][big][jj])
    d = d + np.where(np.isnan(dh), 0.0, np.minimum(dh, 0.3) / 0.3 * h_scale)
    D[ii, jj] = d; D[jj, ii] = d
    for a in range(n):  # constraints (vectorised on the cheap overlap test first)
        for b in range(a + 1, n):
            if np.isfinite(D[a, b]) and D[a, b] < stop * 2 and conflict(spans[a], spans[b]):
                D[a, b] = D[b, a] = np.inf
    # average linkage with constraint propagation
    label = np.arange(n)
    size = C["n"][big].astype(float).copy()
    alive = np.ones(n, bool)
    Dm = D.copy()
    np.fill_diagonal(Dm, np.inf)
    while True:
        flat = np.argmin(Dm)
        a, b = divmod(flat, n)
        if not np.isfinite(Dm[a, b]) or Dm[a, b] > stop:
            break
        wa, wb = size[a], size[b]
        new = (Dm[a] * wa + Dm[b] * wb) / (wa + wb)  # inf propagates = cannot-link
        Dm[a], Dm[:, a] = new, new
        Dm[a, a] = np.inf
        Dm[b], Dm[:, b] = np.inf, np.inf
        size[a] += size[b]; alive[b] = False
        label[label == b] = a
    out = np.full(len(members), -1)
    out[big] = label
    return out, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="data/tracks.npz")
    ap.add_argument("--out", default="data/identities.npz")
    ap.add_argument("--stop", type=float, default=0.20)
    ap.add_argument("--min-seconds", type=float, default=120, help="drop identities seen for less than this")
    a = ap.parse_args()
    T = dict(np.load(a.tracks))
    S, O = summarise(T)
    ch = Chains(S)
    nl = ch.link_short_gaps()
    print(f"{len(S['tid'])} tracklets -> {len(ch.members)} chains ({nl} links)")
    lab, C = cluster(S, ch.members, stop=a.stop)
    ids, cnt = np.unique(lab[lab >= 0], return_counts=True)
    secs = {i: sum(C["n"][j] for j in np.flatnonzero(lab == i)) * DT for i in ids}
    keep = sorted([i for i in ids if secs[i] >= a.min_seconds], key=lambda i: -secs[i])
    print(f"{len(ids)} clusters, {len(keep)} with >= {a.min_seconds:.0f}s; seconds: {[int(secs[i]) for i in keep]}")
    pid_of_obs = np.full(len(O["k"]), -1)
    for new, i in enumerate(keep):
        for j in np.flatnonzero(lab == i):
            for t in ch.members[j]:
                pid_of_obs[S["s"][t]:S["e"][t]] = new
    m = pid_of_obs >= 0
    np.savez_compressed(a.out, pid=pid_of_obs[m], k=O["k"][m], xy=O["xy"][m], h=O["h"][m], sig=O["sig"][m],
                        i0=O["i0"][m], i1=O["i1"][m])
    print(f"assigned {m.mean() * 100:.0f}% of observations to {len(keep)} identities")


if __name__ == "__main__":
    main()
