import unittest

import pandas as pd

from nusc_scene_agent.validation import (
    detect_cut_in_like,
    detect_map_supported_crossing_like,
    detect_map_supported_cut_in_like,
)


class ValidationHeuristicsTest(unittest.TestCase):
    def test_cut_in_detection(self) -> None:
        track = pd.DataFrame(
            [
                {"sample_idx": 1, "x_ego": 17.1, "y_ego": 3.3},
                {"sample_idx": 2, "x_ego": 12.9, "y_ego": 1.3},
                {"sample_idx": 3, "x_ego": 8.6, "y_ego": -0.7},
                {"sample_idx": 4, "x_ego": 4.4, "y_ego": -2.4},
                {"sample_idx": 5, "x_ego": 0.1, "y_ego": -4.2},
            ]
        )
        self.assertTrue(detect_cut_in_like(track))

    def test_left_cut_in_detection_with_direction(self) -> None:
        track = pd.DataFrame(
            [
                {"sample_idx": 1, "x_ego": 18.0, "y_ego": 12.0},
                {"sample_idx": 2, "x_ego": 12.0, "y_ego": 9.0},
                {"sample_idx": 3, "x_ego": 7.0, "y_ego": 6.0},
                {"sample_idx": 4, "x_ego": 2.0, "y_ego": 3.0},
                {"sample_idx": 5, "x_ego": -2.0, "y_ego": 2.0},
            ]
        )
        self.assertTrue(detect_cut_in_like(track, positions=["left"]))
        self.assertFalse(detect_cut_in_like(track, positions=["right"]))

    def test_map_supported_crossing_detection(self) -> None:
        track = pd.DataFrame(
            [
                {"sample_idx": 1, "x_ego": 12.7, "y_ego": -5.2},
                {"sample_idx": 2, "x_ego": 6.5, "y_ego": -5.7},
                {"sample_idx": 3, "x_ego": 0.2, "y_ego": -6.0},
                {"sample_idx": 4, "x_ego": -5.1, "y_ego": -6.2},
            ]
        )
        self.assertTrue(
            detect_map_supported_crossing_like(
                track,
                {"actor_on_crosswalk_any": True, "actor_on_walkway_any": False},
            )
        )

    def test_map_supported_cut_in_detection(self) -> None:
        track = pd.DataFrame(
            [
                {"sample_idx": 9, "x_ego": 16.8, "y_ego": -4.1},
                {"sample_idx": 10, "x_ego": 12.7, "y_ego": -4.1},
                {"sample_idx": 11, "x_ego": 8.5, "y_ego": -4.1},
                {"sample_idx": 12, "x_ego": 4.2, "y_ego": -4.0},
                {"sample_idx": 13, "x_ego": 0.0, "y_ego": -4.0},
                {"sample_idx": 14, "x_ego": -4.5, "y_ego": -3.8},
                {"sample_idx": 15, "x_ego": -9.1, "y_ego": -3.7},
            ]
        )
        self.assertTrue(
            detect_map_supported_cut_in_like(
                track,
                {"actor_uses_ego_lane_any": True, "shares_lane_at_anchor": True},
            )
        )


if __name__ == "__main__":
    unittest.main()
