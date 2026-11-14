"""Clothing-colour descriptors for a detected person."""
import numpy as np, cv2

REGIONS = {"torso": (0.18, 0.48), "shorts": (0.50, 0.68), "legs": (0.72, 0.98)}
AB_BINS, L_BINS = 6, 4
DIM_REGION = AB_BINS * AB_BINS + L_BINS + 3  # ab histogram, L histogram, mean Lab
DIM = DIM_REGION * len(REGIONS)


def describe(img, box):
    """img: BGR frame; box: x1,y1,x2,y2. Returns float32[DIM]; zeros where a region has no usable pixels."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    h, w = y2 - y1, x2 - x1
    out = np.zeros(DIM, np.float32)
    if h < 12 or w < 4:
        return out
    xa, xb = x1 + int(0.25 * w), x2 - int(0.25 * w)
    for r, (f0, f1) in enumerate(REGIONS.values()):
        ya, yb = y1 + int(f0 * h), y1 + max(int(f1 * h), int(f0 * h) + 1)
        patch = img[max(ya, 0):yb, max(xa, 0):max(xb, xa + 1)]
        if patch.size == 0:
            continue
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        keep = ~((hsv[:, 0] > 32) & (hsv[:, 0] < 88) & (hsv[:, 1] > 45))  # drop pitch green
        if keep.sum() < 4:
            continue
        lab = lab[keep]
        a = np.clip(((lab[:, 1] - 128 + 48) / 96 * AB_BINS).astype(int), 0, AB_BINS - 1)
        b = np.clip(((lab[:, 2] - 128 + 48) / 96 * AB_BINS).astype(int), 0, AB_BINS - 1)
        l = np.clip((lab[:, 0] / 256 * L_BINS).astype(int), 0, L_BINS - 1)
        o = r * DIM_REGION
        out[o:o + AB_BINS**2] = np.bincount(a * AB_BINS + b, minlength=AB_BINS**2) / len(lab)
        out[o + AB_BINS**2:o + AB_BINS**2 + L_BINS] = np.bincount(l, minlength=L_BINS) / len(lab)
        out[o + AB_BINS**2 + L_BINS:o + DIM_REGION] = np.median(lab, 0) / 255.0
    return out
