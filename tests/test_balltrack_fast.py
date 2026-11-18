import unittest

import numpy as np

from azfar.balltrack import track_camera


def tracks(rows, size=12):
    f, x, conf = zip(*rows)
    return track_camera(np.array(f), np.column_stack((x, np.zeros(len(x), dtype=int))),
                        np.full(len(f), size), np.array(conf))


class FastBallTest(unittest.TestCase):
    def test_acquisition_uses_motion_instead_of_the_strongest_first_hit(self):
        # At 15 fps this near-camera ball travels 80 px/frame. The strongest
        # first detection is a distractor, and frames 1 and 4 have no sighting.
        found = tracks([(0, 40, .65), (0, 800, .96), (2, 200, .35),
                        (3, 280, .4), (5, 440, .2), (6, 520, .8)])
        self.assertEqual(len(found), 1)
        _, frames, xy, _, _, filled = found[0]
        np.testing.assert_array_equal(frames, np.arange(7))
        np.testing.assert_array_equal(xy[:, 0], 40 + frames * 80)
        np.testing.assert_array_equal(frames[filled], [1, 4])

    def test_detector_anchor_can_arrive_after_motion_support(self):
        found = tracks([(0, 40, .3), (1, 100, .3), (2, 160, .65)])
        self.assertEqual(len(found), 1)
        np.testing.assert_array_equal(found[0][1], [0, 1, 2])

    def test_inconsistent_unanchored_and_impossible_paths_stay_hidden(self):
        for rows in (
            [(0, 100, .8), (1, 180, .8)],  # two hits
            [(0, 100, .42), (1, 180, .42), (2, 260, .42)],  # no detector anchor
            [(0, 100, .8), (1, 180, .8), (2, 102, .8)],  # sudden reversal
            [(0, 100, .8), (1, 260, .8), (2, 420, .8)],  # impossible speed at this size
            [(0, 100, .8), (4, 180, .8), (8, 260, .8)],  # sightings too far apart
        ):
            with self.subTest(rows=rows):
                self.assertEqual(tracks(rows), [])

    def test_unconfirmed_kicks_are_removed_even_at_eof(self):
        initial = [(i, 100, .75) for i in range(3)]
        for distractors in ([(3, 165, .85)], [(3, 165, .85), (4, 105, .9), (5, 190, .9)]):
            found = tracks(initial + distractors)
            self.assertEqual(len(found), 1)
            np.testing.assert_array_equal(found[0][1], [0, 1, 2])

    def test_consistent_fast_kick_survives(self):
        found = tracks([(i, 100, .75) for i in range(3)] +
                       [(3, 165, .9), (4, 230, .35), (6, 360, .3)])
        self.assertEqual(len(found), 1)
        np.testing.assert_array_equal(found[0][1], np.arange(7))
        np.testing.assert_array_equal(found[0][2][:, 0], [100, 100, 100, 165, 230, 295, 360])
