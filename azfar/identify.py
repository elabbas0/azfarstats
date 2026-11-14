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


def load_embeddings(T, emb_dir, raw_dir):
    """Attach re-identification embeddings to the per-camera detection tables of tracks.npz.
    Returns (E0, E1): float32 [n_det, 512] with zero rows where a detection was not embedded."""
    import os
    from .timeline import scan
    from .fuse import SRC_BASE
    clips = scan(raw_dir)
    out = []
    for c in (0, 1):
        src = T[f"src{c}"]
        E = np.zeros((len(src), 512), np.float32)
        order = np.argsort(src)
        for ci, clip in enumerate(clips):
            p = os.path.join(emb_dir, os.path.basename(clip.path).replace(".mp4", ".npz"))
            if not os.path.exists(p):
                continue
            d = np.load(p)
            key = ci * SRC_BASE + d["idx"]
            pos = np.searchsorted(src[order], key)
            pos = np.clip(pos, 0, len(src) - 1)
            hit = src[order][pos] == key
            E[order[pos[hit]]] = d["emb"][hit].astype(np.float32)
        # centre on this camera's average player: what is left is what tells people apart here
        has = np.abs(E).sum(1) > 0
        if has.sum() > 100:
            E[has] -= E[has].mean(0)
            E[has] /= np.maximum(np.linalg.norm(E[has], axis=1, keepdims=True), 1e-9)
        out.append(E)
    return out


def summarise(T, min_len=5, E=None):
    """Per-tracklet summary arrays."""
    order = np.lexsort((T["k"], T["tid"]))
    tid, k, xy, h, sig, i0, i1 = (T[n][order] for n in ("tid", "k", "xy", "h", "sig", "i0", "i1"))
    cut = np.flatnonzero(np.diff(tid)) + 1
    starts, ends = np.r_[0, cut], np.r_[cut, len(tid)]
    S = dict(tid=[], ks=[], ke=[], ps=[], pe=[], vs=[], ve=[], n=[], h=[], hw=[], f0=[], f1=[], n0=[], n1=[], s=[], e=[], emb=[], ne=[])
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
        good = (sig[s:e] < 0.8) & ~np.isnan(h[s:e])
        S["h"].append(np.median(h[s:e][good]) if good.sum() >= 3 else np.nan); S["hw"].append(good.sum())
        for c, idx in enumerate((i0[s:e], i1[s:e])):
            idx = idx[idx >= 0]
            if len(idx):
                bh = boxes[c][idx, 3] - boxes[c][idx, 1]
                idx = idx[(rngs[c][idx] < 30) & (bh > 45) & (feats[c][idx].sum(1) > 1.5)]
            S[f"n{c}"].append(len(idx))
            S[f"f{c}"].append(feats[c][idx].mean(0) if len(idx) else np.zeros(DIM, np.float32))
        S["s"].append(s); S["e"].append(e)
        acc, cnt = np.zeros(512, np.float32), 0
        if E is not None:
            for c, idx in enumerate((i0[s:e], i1[s:e])):
                idx = idx[idx >= 0]
                if len(idx):
                    v = E[c][idx]
                    has = np.abs(v).sum(1) > 0
                    acc += v[has].sum(0); cnt += int(has.sum())
        S["emb"].append(acc); S["ne"].append(cnt)   # emb is the SUM of unit vectors; normalise when comparing
    S = {k_: np.array(v) for k_, v in S.items()}
    return S, dict(tid=tid, k=k, xy=xy, h=h, sig=sig, i0=i0, i1=i1)


def emb_dist(A, B):
    """Cosine distance between (summed) embedding arrays."""
    na = np.linalg.norm(A, axis=-1); nb = np.linalg.norm(B, axis=-1)
    return 1.0 - (A * B).sum(-1) / np.maximum(na * nb, 1e-9)


EMB_SCALE = 1.0  # set by calibrate_scale(): maps embedding distance onto the colour-distance scale


def pair_app(S, ia, ib, min_n=4, chunk=100_000):
    """Appearance distance between summaries ia and ib (arrays of indices); nan when not comparable.
    Uses re-identification embeddings when both sides have them, clothing colours otherwise."""
    ia, ib = np.asarray(ia), np.asarray(ib)
    if len(ia) > chunk:  # keep memory flat for millions of pairs
        return np.concatenate([pair_app(S, ia[o:o + chunk], ib[o:o + chunk], min_n, chunk) for o in range(0, len(ia), chunk)])
    col = _pair_colour(S, ia, ib, min_n)
    if "ne" not in S:
        return col
    both = (S["ne"][ia] >= 1) & (S["ne"][ib] >= 1)
    e = emb_dist(S["emb"][ia], S["emb"][ib]) * EMB_SCALE
    return np.where(both, e, col)


def _pair_colour(S, ia, ib, min_n=4):
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
    """Aggregate tracklet summaries per chain (same keys as S, so pair_app works on chains too)."""
    n = len(members)
    C = dict(ks=np.zeros(n, int), ke=np.zeros(n, int), n=np.zeros(n, int), h=np.full(n, np.nan), hw=np.zeros(n),
             f0=np.zeros((n, DIM), np.float32), f1=np.zeros((n, DIM), np.float32), n0=np.zeros(n), n1=np.zeros(n),
             emb=np.zeros((n, 512), np.float32), ne=np.zeros(n, int), ps=np.zeros((n, 2)), pe=np.zeros((n, 2)))
    for j, m in enumerate(members):
        m = np.array(m)
        C["ks"][j], C["ke"][j], C["n"][j] = S["ks"][m].min(), S["ke"][m].max(), S["n"][m].sum()
        C["ps"][j], C["pe"][j] = S["ps"][m[np.argmin(S["ks"][m])]], S["pe"][m[np.argmax(S["ke"][m])]]
        w = S["hw"][m] * ~np.isnan(S["h"][m])
        if w.sum() > 0:
            C["h"][j] = np.nansum(S["h"][m] * w) / w.sum(); C["hw"][j] = w.sum()
        for c in (0, 1):
            w = S[f"n{c}"][m].astype(float)
            C[f"n{c}"][j] = w.sum()
            if w.sum() > 0:
                C[f"f{c}"][j] = (S[f"f{c}"][m] * w[:, None]).sum(0) / w.sum()
        C["emb"][j] = S["emb"][m].sum(0); C["ne"][j] = S["ne"][m].sum()
    return C


def reach_conflict(a_ke, a_pe, b_ks, b_ps):
    """True if going from the end of span a to the start of the later span b is physically impossible."""
    gap = (b_ks - a_ke) * DT
    return gap <= 0 or np.hypot(*(b_ps - a_pe)) > 2.0 + VMAX * gap


def cluster(C, stop, min_chain=25, h_scale=0.05):
    """Constrained average-linkage clustering of the longer chains. Returns label per chain (-1 = left out)."""
    big = np.flatnonzero((C["n"] >= min_chain) & ((C["ne"] >= 2) | (C["n0"] >= 5) | (C["n1"] >= 5)))
    n = len(big)
    ii, jj = np.triu_indices(n, 1)
    d = pair_app(C, big[ii], big[jj])
    dh = np.abs(C["h"][big][ii] - C["h"][big][jj])
    d = d + np.where(np.isnan(dh), 0.0, np.minimum(dh, 0.3) / 0.3 * h_scale)
    d = np.where(np.isnan(d), np.inf, d)
    # cannot-link: overlap in time, or unreachable
    ks, ke, ps, pe = C["ks"][big], C["ke"][big], C["ps"][big], C["pe"][big]
    overlap = (ks[ii] <= ke[jj]) & (ks[jj] <= ke[ii])
    first = np.where(ke[ii] < ks[jj], ii, jj); second = np.where(ke[ii] < ks[jj], jj, ii)
    gap = np.maximum((ks[second] - ke[first]) * DT, 0)
    unreachable = np.linalg.norm(ps[second] - pe[first], axis=1) > 2.0 + VMAX * gap
    d[overlap | unreachable] = np.inf
    D = np.full((n, n), np.inf, np.float32)
    D[ii, jj] = d; D[jj, ii] = d
    label = np.arange(n)
    size = C["n"][big].astype(float).copy()
    while True:
        flat = int(np.argmin(D))
        a, b = divmod(flat, n)
        if not np.isfinite(D[a, b]) or D[a, b] > stop:
            break
        wa, wb = size[a], size[b]
        new = (D[a] * wa + D[b] * wb) / (wa + wb)  # inf propagates = cannot-link
        D[a], D[:, a] = new, new
        D[a, a] = np.inf
        D[b], D[:, b] = np.inf, np.inf
        size[a] += size[b]
        label[label == b] = a
    out = np.full(len(C["n"]), -1)
    out[big] = label
    return out


def assign(C, labels, K, max_cost, rounds=3, cont_w=0.5):
    """Give every chain one of K identities (or -1).

    Identities start as the K biggest clusters. Each round rebuilds their appearance prototypes and
    re-assigns all chains, longest first, to the best identity that is free at that time and could
    physically have been there; continuity with what the identity did just before/after counts too."""
    import bisect
    ids, tot = [], []
    for l in np.unique(labels[labels >= 0]):
        ids.append(l); tot.append(C["n"][labels == l].sum())
    # two different people are on the pitch at the same time; two clusters that never coexist are
    # usually one person split in two, so only clusters that overlap every earlier seed become seeds
    k_lo, k_hi = int(C["ks"].min()), int(C["ke"].max()) + 1
    def occupancy(l):
        occ = np.zeros(k_hi - k_lo, bool)
        for j in np.flatnonzero(labels == l):
            occ[C["ks"][j] - k_lo:C["ke"][j] - k_lo + 1] = True
        return occ
    seeds, occs = [], []
    for i in np.argsort(tot)[::-1]:
        if len(seeds) >= K or tot[i] * DT < 60:
            break
        o = occupancy(ids[i])
        # compare with the overlap two independent players with this much coverage would have
        if all((o & p).sum() * len(o) >= 0.3 * o.sum() * p.sum() for p in occs):
            seeds.append(ids[i]); occs.append(o)
    lab = np.full(len(labels), -1)
    for new, l in enumerate(seeds):
        lab[labels == l] = new
    K = len(seeds)
    order = np.argsort(-C["n"], kind="stable")
    nC = len(C["n"])
    for rnd in range(rounds):
        # prototypes appended after the chains so pair_app can compare chain i with prototype nC + k
        X = {key: np.concatenate([C[key], np.zeros((K,) + C[key].shape[1:], C[key].dtype)]) for key in ("f0", "f1", "n0", "n1", "emb", "ne")}
        Hk = np.full(K, np.nan)
        for k_ in range(K):
            m = np.flatnonzero(lab == k_)
            if not len(m):
                continue
            X["emb"][nC + k_] = C["emb"][m].sum(0); X["ne"][nC + k_] = C["ne"][m].sum()
            for c in (0, 1):
                w = C[f"n{c}"][m]
                X[f"n{c}"][nC + k_] = w.sum()
                if w.sum() > 0:
                    X[f"f{c}"][nC + k_] = (C[f"f{c}"][m] * w[:, None]).sum(0) / w.sum()
            w = C["hw"][m] * ~np.isnan(C["h"][m])
            if w.sum() > 0:
                Hk[k_] = np.nansum(C["h"][m] * w) / w.sum()
        app = np.stack([pair_app(X, np.arange(nC), np.full(nC, nC + k_)) for k_ in range(K)], 1)  # [chains, K]
        dh = np.abs(C["h"][:, None] - Hk[None])
        app = app + np.where(np.isnan(dh), 0.0, np.minimum(dh, 0.3) / 0.3 * 0.05)
        new_lab = np.full(nC, -1)
        starts = [[] for _ in range(K)]   # per identity: sorted chain start ticks and the chain ids
        who = [[] for _ in range(K)]
        for j in order:
            best, arg = max_cost, -1
            for k_ in range(K):
                pos = bisect.bisect_left(starts[k_], C["ks"][j])
                prv = who[k_][pos - 1] if pos > 0 else None
                nxt = who[k_][pos] if pos < len(who[k_]) else None
                if prv is not None and (C["ke"][prv] >= C["ks"][j] or reach_conflict(C["ke"][prv], C["pe"][prv], C["ks"][j], C["ps"][j])):
                    continue
                if nxt is not None and (C["ks"][nxt] <= C["ke"][j] or reach_conflict(C["ke"][j], C["pe"][j], C["ks"][nxt], C["ps"][nxt])):
                    continue
                cont = 0.0
                for a_, b_ in ((prv, j), (j, nxt)):
                    if a_ is None or b_ is None:
                        continue
                    gap = (C["ks"][b_] - C["ke"][a_]) * DT
                    dist = np.hypot(*(C["ps"][b_] - C["pe"][a_]))
                    cont = max(cont, np.exp(-gap / 3.0) * max(0.0, 1 - dist / (1.5 + 3.0 * gap)))
                a = app[j, k_]
                if np.isnan(a):
                    if cont < 0.5:      # nothing to recognise them by: needs a clear hand-over
                        continue
                    a = max_cost * 0.9
                cost = a - cont_w * cont * max_cost
                if cost < best:
                    best, arg = cost, k_
            if arg >= 0:
                new_lab[j] = arg
                pos = bisect.bisect_left(starts[arg], C["ks"][j])
                starts[arg].insert(pos, C["ks"][j]); who[arg].insert(pos, j)
        lab = new_lab
        print(f"  assignment round {rnd}: {(lab >= 0).sum()} chains, {C['n'][lab >= 0].sum() / C['n'].sum() * 100:.0f}% of observations")
    return lab


def separability(S):
    """Same-person vs different-person appearance distances, to set thresholds from the data itself:
    same = two halves of one long tracklet chain; different = chains on the pitch at the same time."""
    big = np.flatnonzero((S["n"] >= 40) & (S["ne"] >= 2))
    ii, jj = np.triu_indices(len(big), 1)
    ov = (S["ks"][big][ii] <= S["ke"][big][jj]) & (S["ks"][big][jj] <= S["ke"][big][ii])
    d = pair_app(S, big[ii[ov]], big[jj[ov]])
    return d[~np.isnan(d)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="data/tracks.npz")
    ap.add_argument("--emb", default="data/emb")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/identities.npz")
    ap.add_argument("--players", type=int, default=0, help="number of people on the pitch (0 = estimate)")
    ap.add_argument("--stop", type=float, default=0.0, help="clustering threshold (0 = from data)")
    ap.add_argument("--start", default="", help="only identify within this wall-clock window (HH:MM:SS)")
    ap.add_argument("--end", default="")
    a = ap.parse_args()
    T = dict(np.load(a.tracks))
    if a.start and a.end:
        hms = lambda s: sum(int(p) * m for p, m in zip(s.split(":"), (3600, 60, 1)))
        w = (T["k"] >= hms(a.start) * GRID_HZ) & (T["k"] < hms(a.end) * GRID_HZ)
        for key in ("tid", "k", "xy", "h", "sig", "i0", "i1"):
            T[key] = T[key][w]
    E = load_embeddings(T, a.emb, a.raw)
    have = sum(int((np.abs(e).sum(1) > 0).sum()) for e in E)
    print(f"embeddings for {have} detections")
    S, O = summarise(T, E=E if have else None)
    # thresholds from the data: different people on the pitch at the same time set the scale
    diff = separability(S) if have else np.array([])
    if a.stop > 0:
        stop = a.stop
    elif len(diff) > 100:
        stop = float(np.percentile(diff, 3))
    else:
        stop = 0.20
    print(f"different-player distance percentiles 1/3/10/50: {np.percentile(diff, [1, 3, 10, 50]).round(3) if len(diff) else 'n/a'} -> merge threshold {stop:.3f}")
    ch = Chains(S)
    nl = ch.link_short_gaps(app_max=stop * 1.5)
    C = chain_summary(S, ch.members)
    print(f"{len(S['tid'])} tracklets -> {len(ch.members)} chains ({nl} links)")
    lab0 = cluster(C, stop)
    u, k = np.unique(T["k"], return_counts=True)
    K = a.players or int(np.percentile(k[(u >= T['k'].min()) & (u <= T['k'].max())], 75))
    print(f"people on the pitch: {K}; clusters before assignment: {len(np.unique(lab0[lab0 >= 0]))}")
    lab = assign(C, lab0, K, max_cost=stop * 2.0)
    pid_of_obs = np.full(len(O["k"]), -1)
    secs = [C["n"][lab == k_].sum() * DT for k_ in range(lab.max() + 1)]
    rank = np.argsort(secs)[::-1]
    for new, k_ in enumerate(rank):
        for j in np.flatnonzero(lab == k_):
            for t in ch.members[j]:
                pid_of_obs[S["s"][t]:S["e"][t]] = new
    m = pid_of_obs >= 0
    np.savez_compressed(a.out, pid=pid_of_obs[m], k=O["k"][m], xy=O["xy"][m], h=O["h"][m], sig=O["sig"][m],
                        i0=O["i0"][m], i1=O["i1"][m])
    print(f"assigned {m.sum() / len(T['k']) * 100:.0f}% of all observations to {len(rank)} identities; minutes each: {[round(secs[k_] / 60) for k_ in rank]}")


if __name__ == "__main__":
    main()
