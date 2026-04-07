import unittest
from unittest.mock import patch

import pandas as pd

from nusc_scene_agent.validation import (
    ValidationConfig,
    build_actor_grounding,
    detect_cut_in_like,
    detect_map_supported_crossing_like,
    detect_map_supported_cut_in_like,
    localize_event,
    validate_candidate,
)
from nusc_scene_agent.models import ParsedQuery, RetrievalCandidate


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

    def test_localize_event_for_crossing_track(self) -> None:
        track = pd.DataFrame(
            [
                {"sample_idx": 10, "t_sec": -1.0, "x_ego": 10.0, "y_ego": -4.5, "distance": 11.0, "ttc": 4.0, "rel_vx": -1.0, "speed": 1.0, "heading_delta": 0.0},
                {"sample_idx": 11, "t_sec": -0.5, "x_ego": 5.0, "y_ego": -2.0, "distance": 6.0, "ttc": 2.0, "rel_vx": -1.0, "speed": 1.0, "heading_delta": 0.0},
                {"sample_idx": 12, "t_sec": 0.0, "x_ego": 0.5, "y_ego": 0.2, "distance": 2.5, "ttc": 1.0, "rel_vx": -1.0, "speed": 1.0, "heading_delta": 0.0},
                {"sample_idx": 13, "t_sec": 0.5, "x_ego": -3.0, "y_ego": 2.0, "distance": 4.5, "ttc": 2.5, "rel_vx": -1.0, "speed": 1.0, "heading_delta": 0.0},
            ]
        )
        query = ParsedQuery(
            original_text="pedestrian crossing",
            normalized_text="pedestrian crossing",
            category_groups=["pedestrian"],
            positions=["front"],
            behaviors=["crossing"],
            near_distance_m=20.0,
            max_ttc_s=5.0,
        )
        candidate = RetrievalCandidate(
            ann_token="ann-a",
            sample_token="sample-a",
            scene_token="scene-a",
            scene_name="scene-0001",
            sample_idx=12,
            instance_token="instance-a",
            category_name="human.pedestrian.adult",
            category_group="pedestrian",
            location="singapore-queenstown",
            distance=2.5,
            ttc=1.0,
            x_ego=0.5,
            y_ego=0.2,
            speed=1.0,
            rel_vx=-1.0,
            rel_vy=0.0,
            heading_delta=0.0,
            retrieval_score=90.0,
        )
        event = localize_event(track, query, candidate)
        self.assertEqual(event["primary_behavior"], "crossing")
        self.assertEqual(event["peak_sample_idx"], 12)
        self.assertTrue(event["anchor_within_window"])

    def test_build_actor_grounding(self) -> None:
        track = pd.DataFrame([{"sample_idx": 7}, {"sample_idx": 8}, {"sample_idx": 9}])
        query = ParsedQuery(
            original_text="vehicle close in front",
            normalized_text="vehicle close in front",
            category_groups=["vehicle"],
            positions=["front"],
            behaviors=[],
            near_distance_m=12.0,
            max_ttc_s=5.0,
            risk_terms=["risky"],
        )
        candidate = RetrievalCandidate(
            ann_token="ann-b",
            sample_token="sample-b",
            scene_token="scene-b",
            scene_name="scene-0002",
            sample_idx=8,
            instance_token="instance-b",
            category_name="vehicle.car",
            category_group="vehicle",
            location="boston-seaport",
            distance=4.0,
            ttc=1.2,
            x_ego=1.0,
            y_ego=0.0,
            speed=0.0,
            rel_vx=-1.0,
            rel_vy=0.0,
            heading_delta=0.0,
            retrieval_score=88.0,
        )
        grounding = build_actor_grounding(track, candidate, query)
        self.assertEqual(grounding["role"], "primary_actor")
        self.assertEqual(grounding["track_start_sample_idx"], 7)
        self.assertEqual(grounding["track_end_sample_idx"], 9)
        self.assertEqual(grounding["grounded_positions"], ["front"])

    def test_validate_candidate_supports_map_and_event_ablation(self) -> None:
        query = ParsedQuery(
            original_text="pedestrian crossing",
            normalized_text="pedestrian crossing",
            category_groups=["pedestrian"],
            positions=["front"],
            behaviors=["crossing"],
            near_distance_m=20.0,
            max_ttc_s=5.0,
        )
        candidate = RetrievalCandidate(
            ann_token="ann-a",
            sample_token="sample-a",
            scene_token="scene-a",
            scene_name="scene-0001",
            sample_idx=12,
            instance_token="instance-a",
            category_name="human.pedestrian.adult",
            category_group="pedestrian",
            location="singapore-queenstown",
            distance=2.5,
            ttc=1.0,
            x_ego=0.5,
            y_ego=0.2,
            speed=1.0,
            rel_vx=-1.0,
            rel_vy=0.0,
            heading_delta=0.0,
            retrieval_score=90.0,
        )
        timeline = pd.DataFrame(
            [
                {"sample_idx": 10, "sample_token": "sample-x", "timestamp_us": 0, "x_ego": 10.0, "y_ego": -4.5, "distance": 11.0, "ttc": 4.0, "rel_vx": -1.0, "speed": 1.0, "heading_delta": 0.0, "ego_x": 0.0, "ego_y": 0.0, "ego_yaw": 0.0},
                {"sample_idx": 11, "sample_token": "sample-y", "timestamp_us": 500000, "x_ego": 5.0, "y_ego": -2.0, "distance": 6.0, "ttc": 2.0, "rel_vx": -1.0, "speed": 1.0, "heading_delta": 0.0, "ego_x": 0.0, "ego_y": 0.0, "ego_yaw": 0.0},
                {"sample_idx": 12, "sample_token": "sample-a", "timestamp_us": 1000000, "x_ego": 0.5, "y_ego": 0.2, "distance": 2.5, "ttc": 1.0, "rel_vx": -1.0, "speed": 1.0, "heading_delta": 0.0, "ego_x": 0.0, "ego_y": 0.0, "ego_yaw": 0.0},
                {"sample_idx": 13, "sample_token": "sample-b", "timestamp_us": 1500000, "x_ego": -3.0, "y_ego": 2.0, "distance": 4.5, "ttc": 2.5, "rel_vx": -1.0, "speed": 1.0, "heading_delta": 0.0, "ego_x": 0.0, "ego_y": 0.0, "ego_yaw": 0.0},
            ]
        )
        ego_window = pd.DataFrame(
            [
                {"sample_idx": 12, "ego_x": 0.0, "ego_y": 0.0, "ego_yaw": 0.0},
            ]
        )

        with patch(
            "nusc_scene_agent.validation._load_validation_window",
            return_value=(timeline, pd.DataFrame(), ego_window),
        ), patch("nusc_scene_agent.validation.build_case_map_context") as build_map_context:
            case = validate_candidate(
                None,
                query,
                candidate,
                validation_config=ValidationConfig(
                    name="no_map_no_event",
                    enable_map_context=False,
                    enable_event_localization=False,
                ),
            )

        build_map_context.assert_not_called()
        self.assertFalse(case.map_context["available"])
        self.assertEqual(case.event_localization, {})
        self.assertEqual(case.evidence["validation_profile"], "no_map_no_event")
        self.assertFalse(case.evidence["map_context_enabled"])
        self.assertFalse(case.evidence["event_localization_enabled"])


if __name__ == "__main__":
    unittest.main()
