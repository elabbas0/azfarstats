import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from azfar import balltrack
from azfar.camera import Camera


class BallTrackFusionTest(unittest.TestCase):
    def track(self, sightings):
        """Run the normal detector-cache to pitch-track path on two calibrated cameras."""
        cameras = {
            ch: Camera(1000, 0, 0, 960, 540, [0, 0, 0], [-1, -1, -2], flip=flip)
            for ch, flip in (("ch2", None), ("ch3", [40, 20]))
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calib = root / "calibration.json"
            calib.write_text(json.dumps({
                "pitch": {"L": 40, "W": 20},
                "cameras": {ch: cam.to_dict() for ch, cam in cameras.items()},
            }))
            session = root / "session.json"
            session.write_text(json.dumps({"matches": [{"start": "00:00:00", "end": "00:00:01"}]}))
            clips = []
            for ch, rows in sightings.items():
                clips.append(SimpleNamespace(path=str(root / (ch + ".mp4")), t0=0, ch=ch))
                frames, boxes = [], []
                cam = cameras[ch]
                for f, point, scale in rows:
                    world = np.array([*point, -0.11])
                    px = cam.project(world)
                    size = cam.f * balltrack.BALL_D / np.linalg.norm(cam._to_local(world) - cam.C)
                    radius = size * scale / 2
                    boxes.append(np.r_[px - radius, px + radius])
                    frames.append(f)
                np.savez(root / (ch + ".npz"), f=frames, box=np.array(boxes).reshape(-1, 4),
                         conf=np.full(len(frames), 0.9))
            archive = root / "ball.npz"
            args = ["balltrack", "--dets", directory, "--calib", str(calib),
                    "--session", str(session), "--out", str(archive)]
            with patch.object(sys, "argv", args), patch.object(balltrack, "scan", return_value=clips):
                balltrack.main()
            with np.load(archive) as result:
                return dict(result)

    def test_other_view_used_when_older_lock_cannot_place_the_ball(self):
        for point, scale in (((10, 3), 3.0), ((45, 4), 1.0)):
            with self.subTest(point=point, scale=scale):
                result = self.track({
                    "ch2": [(f, point, scale) for f in range(6)],
                    "ch3": [(f, (20, 18), 1.0) for f in range(1, 6)],
                })
                np.testing.assert_array_equal(result["m15"], np.arange(1, 6))
                self.assertTrue(np.all(result["src15"] == "ch3"))
                np.testing.assert_allclose(result["xy15"], [[20, 18]] * 5)

    def test_agreeing_views_still_triangulated(self):
        hits = [(f, (20, 18), 1.0) for f in range(6)]
        result = self.track({"ch2": hits, "ch3": hits})
        np.testing.assert_array_equal(result["views15"], np.full(6, 2))
        np.testing.assert_allclose(result["xy15"], [[20, 18]] * 6)

    def test_single_camera_and_empty_detector_archives(self):
        hits = [(f, (20, 18), 1.0) for f in range(6)]
        for cameras in ({"ch3": hits}, {"ch2": [], "ch3": hits}):
            with self.subTest(cameras=list(cameras)):
                result = self.track(cameras)
                np.testing.assert_array_equal(result["m15"], np.arange(6))
                self.assertTrue(np.all(result["src15"] == "ch3"))
        result = self.track({"ch2": [], "ch3": []})
        self.assertEqual(result["xy15"].shape, (0, 2))
        self.assertEqual(result["xy"].shape, (0, 2))
        self.assertTrue(np.issubdtype(result["k"].dtype, np.integer))
