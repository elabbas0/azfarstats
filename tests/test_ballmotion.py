import unittest

import cv2
import numpy as np

from azfar.ballmotion import BallMotion


class BallMotionTest(unittest.TestCase):
    def setUp(self):
        self.pitch = np.full((540, 960, 3), (55, 115, 65), np.uint8)
        cv2.line(self.pitch, (55, 0), (110, 539), (230, 230, 230), 4)
        self.motion = BallMotion()

    def streak(self, point, color=(180, 190, 185)):
        image = self.pitch.copy()
        cv2.ellipse(image, point, (12, 3), 40, 0, 360, color, -1)
        return image

    def test_fast_diagonal_streak_emits_only_the_current_ball(self):
        self.assertEqual(len(self.motion.detect(self.streak((200, 180)))[0]), 0)
        for xy in ((234, 202), (268, 224), (302, 246)):
            boxes, scores = self.motion.detect(self.streak(xy))
            self.assertEqual(len(boxes), 1)
            center = (boxes[0, :2] + boxes[0, 2:]) / 2
            np.testing.assert_allclose(center, xy, atol=1)
            size = boxes[0, 2:] - boxes[0, :2]
            self.assertTrue(np.all((size >= 5) & (size <= 9)))
            self.assertGreaterEqual(scores[0], 0.25)
            self.assertLess(scores[0], 0.45)

        boxes, _ = self.motion.detect(self.pitch)
        self.assertEqual(boxes.shape, (0, 4))

    def test_pitch_markings_neon_boot_and_revealed_line(self):
        self.motion.detect(self.pitch)
        covered = self.pitch.copy()
        cv2.rectangle(covered, (0, 110), (180, 135), (10, 10, 10), -1)
        for frame in (covered, self.pitch, self.streak((140, 190), (0, 255, 255))):
            self.assertEqual(len(self.motion.detect(frame)[0]), 0)
