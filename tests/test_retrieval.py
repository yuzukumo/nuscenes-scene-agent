import sqlite3
import unittest

import pandas as pd

from nusc_scene_agent.models import ParsedQuery
from nusc_scene_agent.retrieval import _cut_in_temporal_bonus, _load_cut_in_temporal_features


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
