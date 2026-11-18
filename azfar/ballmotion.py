"""Weak sightings of motion-blurred balls between detector hits on a fixed camera."""

import cv2
import numpy as np


class BallMotion:
    """Call once per consecutive BGR frame. Each clip needs its own history."""

    def __init__(self):
        self.previous = None
        self.background = None

    def detect(self, frame):
        h, w = frame.shape[:2]
        scale = min(960 / w, 540 / h, 1.0)
        small = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                           interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        empty = np.empty((0, 4), np.float32), np.empty(0, np.float32)
        if self.previous is None or self.previous.shape != gray.shape:
            self.previous = gray
            self.background = gray.astype(np.float32)
            return empty

        # Positive differences place the proposal at the arriving ball. Testing
        # the previous 3x3 neighbourhood also rejects small camera vibrations.
        change = cv2.subtract(gray, cv2.dilate(self.previous, None))
        contrast = cv2.subtract(gray, cv2.convertScaleAbs(self.background))
        self.previous = gray
        cv2.accumulateWeighted(gray, self.background, 0.02)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, 105), (179, 105, 255))
        mask &= cv2.inRange(change, 14, 255)
        mask &= cv2.inRange(contrast, 14, 255)
        if cv2.countNonZero(mask) > 0.1 * mask.size:
            return empty  # scene-wide lighting change

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not 3 <= area <= 600:
                continue
            (cx, cy), sides, _ = cv2.minAreaRect(contour)
            width, length = sorted(side + 1 for side in sides)
            if not (2 <= width <= 18 and 1.6 * width <= length <= min(64, 9 * width)):
                continue
            if area < 0.32 * width * length:
                continue
            # The smear length is exposure-dependent. Its minor axis estimates
            # the ball diameter used by the tracker for range and speed gates.
            x, y, bw, bh = cv2.boundingRect(contour)
            crop = np.s_[y:y + bh, x:x + bw]
            brightness_gain = cv2.mean(change[crop], mask=mask[crop])[0]
            score = 0.25 + 0.18 * min(brightness_gain / 120, 1)
            candidates.append((score, cx + 0.5, cy + 0.5, width))

        candidates.sort(reverse=True)
        sx, sy = w / small.shape[1], h / small.shape[0]
        boxes, scores = [], []
        for score, cx, cy, diameter in candidates[:8]:
            radius = diameter / 2
            boxes.append((max(0, cx - radius) * sx, max(0, cy - radius) * sy,
                          min(small.shape[1], cx + radius) * sx, min(small.shape[0], cy + radius) * sy))
            scores.append(score)
        return np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(scores, np.float32)
