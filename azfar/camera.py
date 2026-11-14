"""Fisheye camera model for the corner-mounted XVR cameras.

World frame (per pitch): X along the pitch length, Y along the goal line,
Z pointing DOWN (so the ground is Z=0 and a point h metres above it is Z=-h).
Camera frame: OpenCV convention (x right, y down, z forward).
Projection: equidistant fisheye with radial polynomial,
    theta_d = theta * (1 + k1 theta^2 + k2 theta^4),  pix = f * theta_d * unit(xy) + c
"""
from __future__ import annotations
import json
import numpy as np
import cv2

R0 = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])  # camera looking along +X, level


class Camera:
    def __init__(self, f, k1, k2, cx, cy, rvec, C, flip=None):
        self.f, self.k1, self.k2, self.cx, self.cy = f, k1, k2, cx, cy
        self.rvec = np.asarray(rvec, float)
        self.C = np.asarray(C, float)
        self.R = cv2.Rodrigues(self.rvec)[0] @ R0  # world -> camera
        # optional (L, W): this camera's local frame is the global frame rotated 180 deg
        self.flip = flip

    # ---- serialisation
    def to_dict(self):
        return dict(f=self.f, k1=self.k1, k2=self.k2, cx=self.cx, cy=self.cy,
                    rvec=self.rvec.tolist(), C=self.C.tolist(), flip=self.flip)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    @classmethod
    def load(cls, path, name):
        return cls.from_dict(json.load(open(path))["cameras"][name])

    # ---- geometry
    def _to_local(self, P):
        if self.flip is None:
            return P
        P = np.array(P, float, copy=True)
        P[..., 0] = self.flip[0] - P[..., 0]
        P[..., 1] = self.flip[1] - P[..., 1]
        return P

    _to_global = _to_local  # the 180-degree flip is its own inverse

    def project(self, P):
        """World points (...,3) -> pixels (...,2)."""
        P = self._to_local(np.asarray(P, float))
        pc = (P - self.C) @ self.R.T
        r = np.hypot(pc[..., 0], pc[..., 1])
        th = np.arctan2(r, pc[..., 2])
        thd = th * (1 + self.k1 * th**2 + self.k2 * th**4)
        s = np.where(r > 1e-12, self.f * thd / np.maximum(r, 1e-12), 0.0)
        return np.stack([self.cx + s * pc[..., 0], self.cy + s * pc[..., 1]], -1)

    def rays(self, px):
        """Pixels (...,2) -> unit ray directions in the local world frame."""
        px = np.asarray(px, float)
        x = (px[..., 0] - self.cx) / self.f
        y = (px[..., 1] - self.cy) / self.f
        thd = np.hypot(x, y)
        th = thd.copy()
        for _ in range(8):  # Newton on th*(1+k1 th^2+k2 th^4) = thd
            g = th * (1 + self.k1 * th**2 + self.k2 * th**4) - thd
            dg = 1 + 3 * self.k1 * th**2 + 5 * self.k2 * th**4
            th = th - g / np.where(np.abs(dg) < 1e-6, 1e-6, dg)
        th = np.clip(th, 0, np.pi * 0.95)
        s = np.where(thd > 1e-12, np.sin(th) / np.maximum(thd, 1e-12), 0.0)
        d = np.stack([x * s, y * s, np.cos(th)], -1)
        return d @ self.R  # camera -> world

    def to_ground(self, px, height=0.0):
        """Pixels -> world XY on the horizontal plane `height` metres above ground.
        Returns (xy, ok); ok is False where the ray does not hit the plane."""
        d = self.rays(px)
        zt = -height
        dz = d[..., 2]
        ok = dz > 1e-3 if zt > self.C[2] else dz < -1e-3
        t = (zt - self.C[2]) / np.where(np.abs(dz) < 1e-6, 1e-6, dz)
        P = self.C + d * t[..., None]
        P = self._to_global(P)
        return P[..., :2], ok & (t > 0)

    def height_at(self, px_top, xy):
        """Height (m) of the point seen at px_top that lies vertically above ground point xy."""
        xy = self._to_local(np.concatenate([np.asarray(xy, float), np.zeros(np.shape(xy)[:-1] + (1,))], -1))[..., :2]
        d = self.rays(px_top)
        v = xy - self.C[:2]
        dh = d[..., :2]
        t = (v * dh).sum(-1) / np.maximum((dh * dh).sum(-1), 1e-9)
        return -(self.C[2] + t * d[..., 2])
