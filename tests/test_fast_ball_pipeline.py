import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from azfar.balldet import detect_frame
from azfar.ballmotion import BallMotion
from azfar.balltrack import track_camera


def fast_pass():
    """A 15 fps pass with an eleven-frame interval between detector hits."""
    model, result = Mock(), Mock()
    model.predict.return_value = [result]
    motion = BallMotion()
    truth = {}
    observations = []
    detector_hits = []
    # The ball starts slowly and travels 35 pixels per frame after the kick.
    for frame_index in range(38):
        image = np.full((1080, 1920, 3), (42, 95, 42), np.uint8)
        cv2.line(image, (0, 600), (1919, 600), (230, 230, 230), 4)
        # A second moving bright object continues after the ball leaves view.
        cv2.ellipse(image, (1750 - 23 * frame_index, 340), (12, 4), 0, 0, 360,
                    (208, 220, 228), -1)
        detector_boxes = []
        if 3 <= frame_index <= 24:
            x = 300 + 5 * (frame_index - 3) + 30 * max(0, frame_index - 5)
            y = 504 + frame_index - 3
            truth[frame_index] = (x, y)
            if frame_index in {3, 4, 5, 6, 7, 18, 24}:
                cv2.circle(image, (x, y), 5, (214, 231, 235), -1)
                detector_boxes.append((x - 5.5, y - 5.5, x + 5.5, y + 5.5))
                detector_hits.append((frame_index, (x, y), 11.0, 0.90))
            elif frame_index != 13:  # one frame of complete occlusion
                cv2.ellipse(image, (x, y), (13, 4), 2, 0, 360, (181, 196, 203), -1)
        # Supply the sparse YOLO outputs at its 1536x864 inference resolution.
        result.boxes.xyxy.cpu.return_value.numpy.return_value = (
            np.asarray(detector_boxes, np.float32).reshape(-1, 4) * 0.8)
        result.boxes.conf.cpu.return_value.numpy.return_value = np.full(
            len(detector_boxes), 0.90, np.float32)
        boxes, scores = detect_frame(image, model, 0.1, motion)
        for box, score in zip(boxes, scores):
            center = (box[:2] + box[2:]) / 2
            size = max(box[2:] - box[:2])
            observations.append((frame_index, center, size, score))
    return truth, detector_hits, observations


def track_inputs(hits):
    frames, centers, sizes, scores = zip(*hits)
    return tuple(np.asarray(col) for col in (frames, centers, sizes, scores))


class FastBallPipelineTest(unittest.TestCase):
    def test_blurred_pass_remains_localized_through_sparse_detector_hits(self):
        truth, _, observations = fast_pass()
        self.assertTrue(any(f > max(truth) and abs(xy[1] - 340) < 3
                            for f, xy, _, _ in observations))
        tracks = track_camera(*track_inputs(observations))
        self.assertEqual(len(tracks), 1)
        kind, frames, centers, _, _, filled = tracks[0]
        self.assertEqual(kind, "ball")
        np.testing.assert_array_equal(frames, list(truth))
        np.testing.assert_allclose(centers, [truth[f] for f in frames], atol=3)
        # Frame 13 should be bridged, while the other frames in the long
        # detector gap must have actual pixel observations.
        np.testing.assert_array_equal(frames[filled], [13])


if __name__ == "__main__":
    unittest.main()
