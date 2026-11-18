import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from azfar.balldet import detect_frame


class BallDetectorTest(unittest.TestCase):
    def detect(self, learned_score, threshold=0.1):
        frame = np.zeros((720, 1280, 3), np.uint8)
        model = Mock()
        detections = Mock()
        # The model sees the frame resized to 1536 x 864.
        detections.xyxy.cpu.return_value.numpy.return_value = np.array([[120, 240, 144, 252]], np.float32)
        detections.conf.cpu.return_value.numpy.return_value = np.array([learned_score], np.float32)
        model.predict.return_value = [SimpleNamespace(boxes=detections)]
        motion = Mock()
        motion.detect.return_value = (
            np.array([[105, 200, 115, 210], [202, 200, 212, 210]], np.float32),
            np.array([0.4, 0.3], np.float32),
        )
        return detect_frame(frame, model, threshold, motion)

    def test_strong_detector_hit_and_separate_motion_candidate(self):
        boxes, scores = self.detect(0.8)
        np.testing.assert_allclose(boxes, [[100, 200, 120, 210], [202, 200, 212, 210]])
        np.testing.assert_allclose(scores, [0.8, 0.3])

    def test_streak_width_preserved_when_model_hit_is_weaker(self):
        boxes, scores = self.detect(0.2)
        np.testing.assert_allclose(boxes, [[105, 200, 115, 210], [202, 200, 212, 210]])
        np.testing.assert_allclose(scores, [0.4, 0.3])

    def test_confidence_threshold_also_applies_to_streaks(self):
        boxes, scores = self.detect(0.8, threshold=0.5)
        np.testing.assert_allclose(boxes, [[100, 200, 120, 210]])
        np.testing.assert_allclose(scores, [0.8])
