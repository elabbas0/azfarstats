"""Combine the two ball tracks.

data/ball_sightings.npz - follows clear ball sightings (precise when it has one, but can hop to a boot)
data/ball_possession.npz - who has the ball / straight passes (never hops, but places the ball loosely)

Per 0.2 s: a sighting-track point is kept when it is strong (seen by both cameras, or a confident hit),
or when it agrees with the possession track; a weak single-camera point that strays from where possession
says the ball is, is dropped. Gaps are filled from the possession track.

Usage: python -m azfar.ballfuse   (writes data/ball.npz)
"""
from __future__ import annotations
import numpy as np
from .ball import STEP

AGREE = 2.5      # m
STRONG_W = 0.45


def main():
    S, P = np.load("data/ball_sightings.npz"), np.load("data/ball_possession.npz")
    k0, k1 = int(min(S["k"].min(), P["k"].min())), int(max(S["k"].max(), P["k"].max())) + 1
    n = k1 - k0
    sx = np.full((n, 2), np.nan); sz = np.zeros(n); px = np.full((n, 2), np.nan); own = np.full(n, -2)
    sx[S["k"] - k0] = S["xy"]; sz[S["k"] - k0] = S["z"]
    px[P["k"] - k0] = P["xy"]; own[P["k"] - k0] = P["owner"]
    # strength of the sighting track at each tick, from its 15 Hz points
    strong = np.zeros(n, bool)
    m = S["m15"]; kk = np.round(m / STEP).astype(int) - k0
    ok = (kk >= 0) & (kk < n)
    st = (S["views15"] >= 2) | (S["w15"] >= STRONG_W)
    np.logical_or.at(strong, kk[ok], st[ok])
    have_s, have_p = ~np.isnan(sx[:, 0]), ~np.isnan(px[:, 0])
    d = np.linalg.norm(sx - px, axis=1)
    use_s = have_s & (strong | ~have_p | (d <= AGREE))
    xy = np.where(use_s[:, None], sx, px)
    z = np.where(use_s, sz, 0.11)
    good = ~np.isnan(xy[:, 0])
    k = np.arange(k0, k1)[good]
    np.savez_compressed("data/ball.npz", k=k, xy=xy[good], z=z[good], owner=own[good],
                        m15=k * STEP, xy15=xy[good], z15=z[good], views15=np.where(use_s[good], 1, 4), w15=np.ones(good.sum()))
    print(f"sighting track used on {use_s.mean() * 100:.0f}% of ticks, possession on {(good & ~use_s).mean() * 100:.0f}%; "
          f"weak sightings dropped for straying: {(have_s & ~use_s).sum()} ticks")
    stp = np.diff(xy[good], axis=0)[np.diff(k) == 1]
    print("jumps > 4 m in 0.2 s:", int((np.linalg.norm(stp, axis=1) > 4).sum()), "| ball shown on", f"{good.mean() * 100:.0f}% of ticks")


if __name__ == "__main__":
    main()
