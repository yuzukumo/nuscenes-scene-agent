import sqlite3
import unittest

import pandas as pd

from nusc_scene_agent.models import ParsedQuery
from nusc_scene_agent.retrieval import (
    RetrievalScoreConfig,
    _cut_in_temporal_bonus,
    _load_cut_in_temporal_features,
    _score_frame,
    _score_row,
)


def make_query(position: str) -> ParsedQuery:
    return ParsedQuery(
        original_text="cut in",
        normalized_text="cut in",
        category_groups=["vehicle"],
        positions=[position],
        behaviors=["cut_in"],
        near_distance_m=25.0,
        max_ttc_s=6.0,
    )


class RetrievalTemporalScoringTest(unittest.TestCase):
    def test_vectorized_score_matches_row_score(self) -> None:
        query = ParsedQuery(
            original_text="front cut in and crossing",
            normalized_text="front cut in and crossing",
            category_groups=["vehicle", "pedestrian"],
            positions=["left"],
            behaviors=["cut_in", "crossing", "oncoming", "stopped_lead"],
            near_distance_m=25.0,
            max_ttc_s=6.0,
        )
        frame = pd.DataFrame(
            [
                {
                    "distance": 6.0,
                    "ttc": 2.0,
                    "category_group": "vehicle",
                    "heading_delta": 3.0,
                    "rel_vx": -4.0,
                    "x_ego": 10.0,
                    "y_ego": 3.0,
                    "is_stationary": 1,
                    "num_lidar_pts": 5,
                    "num_radar_pts": 0,
                    "prev_max_y": 8.0,
                    "prev_min_y": 3.0,
                    "future_min_abs_y": 1.0,
                },
                {
                    "distance": 12.0,
                    "ttc": None,
                    "category_group": "pedestrian",
                    "heading_delta": 0.1,
                    "rel_vx": -1.0,
                    "x_ego": 5.0,
                    "y_ego": -4.0,
                    "is_stationary": 0,
                    "num_lidar_pts": 0,
                    "num_radar_pts": 2,
                    "prev_max_y": -2.0,
                    "prev_min_y": -6.0,
                    "future_min_abs_y": 3.5,
                },
            ]
        )

        vector_scores = _score_frame(frame, query).tolist()
        row_scores = [_score_row(row, query) for _, row in frame.iterrows()]

        for vector_score, row_score in zip(vector_scores, row_scores):
            self.assertAlmostEqual(vector_score, row_score, places=9)

    def test_equal_score_profile_changes_weighting_without_changing_shape(self) -> None:
        query = ParsedQuery(
            original_text="front vehicle stopped",
            normalized_text="front vehicle stopped",
            category_groups=["vehicle"],
            positions=["front"],
            behaviors=["stopped_lead"],
            near_distance_m=25.0,
            max_ttc_s=6.0,
        )
        frame = pd.DataFrame(
            [
                {
                    "distance": 6.0,
                    "ttc": 2.0,
                    "category_group": "vehicle",
                    "heading_delta": 0.0,
                    "rel_vx": -1.0,
                    "x_ego": 10.0,
                    "y_ego": 0.5,
                    "is_stationary": 1,
                    "num_lidar_pts": 5,
                    "num_radar_pts": 2,
                }
            ]
        )
        default_score = _score_frame(frame, query).iloc[0]
        equal_weights = RetrievalScoreConfig(profile_name="equal").resolved_weights()
        equal_score = _score_frame(frame, query, weights=equal_weights).iloc[0]

        self.assertGreater(default_score, equal_score)
        self.assertEqual(len(_score_frame(frame, query, weights=equal_weights)), len(frame))

    def test_retrieval_score_config_rejects_unknown_or_negative_weights(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown retrieval score profile"):
            RetrievalScoreConfig(profile_name="unknown").resolved_weights()
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            RetrievalScoreConfig(weights={"distance": -1.0}).resolved_weights()
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            RetrievalScoreConfig(weights={"distance": float("nan")}).resolved_weights()

    def test_retrieval_score_config_rejects_negative_candidate_scan_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "use 0 for a full scan"):
            RetrievalScoreConfig(candidate_scan_limit=-1).resolved_candidate_scan_limit()

        self.assertEqual(
            RetrievalScoreConfig(candidate_scan_limit=0).resolved_candidate_scan_limit(),
            0,
        )

    def test_left_cut_in_temporal_bonus_rewards_lateral_collapse(self) -> None:
        row = pd.Series(
            {
                "y_ego": 2.1,
                "prev_max_y": 9.0,
                "prev_min_y": 2.1,
                "future_min_abs_y": 0.8,
            }
        )
        bonus = _cut_in_temporal_bonus(row, make_query("left"))
        self.assertGreater(bonus, 5.0)

    def test_right_cut_in_temporal_bonus_respects_direction(self) -> None:
        left_like_row = pd.Series(
            {
                "y_ego": 2.1,
                "prev_max_y": 9.0,
                "prev_min_y": 2.1,
                "future_min_abs_y": 0.8,
            }
        )
        bonus = _cut_in_temporal_bonus(left_like_row, make_query("right"))
        self.assertEqual(bonus, 0.0)

    def test_load_cut_in_temporal_features_batches_large_ann_token_lists(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.execute(
            """
            CREATE TABLE agents (
                ann_token TEXT,
                scene_token TEXT,
                instance_token TEXT,
                sample_idx INTEGER,
                y_ego REAL
            )
            """
        )
        rows = [
            ("ann-{0}".format(idx), "scene-1", "inst-{0}".format(idx), 0, float(idx % 7))
            for idx in range(1005)
        ]
        conn.executemany("INSERT INTO agents VALUES (?, ?, ?, ?, ?)", rows)
        conn.commit()

        result = _load_cut_in_temporal_features(conn, [row[0] for row in rows])
        self.assertEqual(len(result), len(rows))
        self.assertIn("future_min_abs_y", result.columns)
        conn.close()


if __name__ == "__main__":
    unittest.main()
