import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from nusc_scene_agent.perception_benchmark import generate_perception_benchmark_from_scenario_config
from nusc_scene_agent.world_model_benchmark import (
    evaluate_world_model_predictions,
    generate_proxy_world_model_predictions,
    generate_world_model_benchmark_from_perception_benchmark,
)
from nusc_scene_agent.world_model_case_studies import render_world_model_case_studies


def _write_scenario_config(path: Path) -> None:
    path.write_text(
        """
queries:
  - id: crossing_canonical
    description: Pedestrian crossing ahead of ego
    top_k: 3
    expect_match: true
    benchmark_group: crossing_anchor
    variant_type: canonical
    tags: [scenario_mining, crossing]
    query:
      natural_language: Pedestrian crossing ahead of ego
      actors: [pedestrian]
      behaviors: [crossing]
      reference_case_keys: [s2:inst-1]
      reference_scene_names: [scene-0001]
      reference_instance_tokens: [inst-1]
      reference_event_sample_range: [0, 4]
      reference_peak_sample_idx: 2
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _build_test_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE samples (
            sample_token TEXT PRIMARY KEY,
            scene_token TEXT NOT NULL,
            scene_name TEXT NOT NULL,
            scene_description TEXT,
            sample_idx INTEGER NOT NULL,
            timestamp_us INTEGER NOT NULL,
            location TEXT NOT NULL,
            ego_x REAL NOT NULL,
            ego_y REAL NOT NULL,
            ego_yaw REAL NOT NULL
        );
        CREATE TABLE agents (
            ann_token TEXT PRIMARY KEY,
            sample_token TEXT NOT NULL,
            scene_token TEXT NOT NULL,
            scene_name TEXT NOT NULL,
            sample_idx INTEGER NOT NULL,
            instance_token TEXT NOT NULL,
            category_name TEXT NOT NULL,
            category_group TEXT NOT NULL,
            x_ego REAL NOT NULL,
            y_ego REAL NOT NULL,
            distance REAL NOT NULL,
            visibility INTEGER NOT NULL,
            ttc REAL,
            speed REAL NOT NULL,
            rel_vx REAL NOT NULL,
            rel_vy REAL NOT NULL,
            heading_delta REAL NOT NULL,
            num_lidar_pts INTEGER NOT NULL,
            num_radar_pts INTEGER NOT NULL
        );
        """
    )

    sample_rows = [
        ("s0", "scene-token-1", "scene-0001", "test scene", 0, 0, "boston-seaport", 100.0, 50.0, 0.0),
        ("s1", "scene-token-1", "scene-0001", "test scene", 1, 500000, "boston-seaport", 102.0, 50.5, 0.0),
        ("s2", "scene-token-1", "scene-0001", "test scene", 2, 1000000, "boston-seaport", 104.0, 51.0, 0.0),
        ("s3", "scene-token-1", "scene-0001", "test scene", 3, 1500000, "boston-seaport", 106.0, 51.5, 0.0),
        ("s4", "scene-token-1", "scene-0001", "test scene", 4, 2000000, "boston-seaport", 108.0, 52.0, 0.0),
    ]
    agent_rows = [
        ("ann-0", "s0", "scene-token-1", "scene-0001", 0, "inst-1", "human.pedestrian.adult", "pedestrian", 8.0, 1.2, 8.1, 4, 4.0, 1.3, -1.0, 0.0, 0.0, 8, 0),
        ("ann-1", "s1", "scene-token-1", "scene-0001", 1, "inst-1", "human.pedestrian.adult", "pedestrian", 6.0, 0.7, 6.0, 4, 3.0, 1.2, -1.0, 0.0, 0.0, 8, 0),
        ("ann-2", "s2", "scene-token-1", "scene-0001", 2, "inst-1", "human.pedestrian.adult", "pedestrian", 4.0, 0.1, 4.0, 3, 2.0, 1.1, -1.0, 0.0, 0.0, 8, 0),
        ("ann-3", "s3", "scene-token-1", "scene-0001", 3, "inst-1", "human.pedestrian.adult", "pedestrian", 2.0, -0.5, 2.1, 2, 1.2, 1.0, -1.0, 0.0, 0.0, 8, 0),
        ("ann-4", "s4", "scene-token-1", "scene-0001", 4, "inst-1", "human.pedestrian.adult", "pedestrian", 0.5, -1.0, 1.1, 2, 1.0, 1.0, -1.0, 0.0, 0.0, 8, 0),
    ]
    conn.execute("INSERT INTO metadata VALUES (?, ?)", ("dataroot", str(path.parent / "data")))
    conn.executemany("INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", sample_rows)
    conn.executemany("INSERT INTO agents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", agent_rows)
    conn.commit()
    conn.close()


class WorldModelCaseStudiesTest(unittest.TestCase):
    def test_render_world_model_case_studies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "scenario.yaml"
            db_path = root / "index.sqlite"
            perception_path = root / "perception.json"
            world_model_path = root / "world_model.json"
            _write_scenario_config(config_path)
            _build_test_db(db_path)
            generate_perception_benchmark_from_scenario_config(config_path, db_path, perception_path)
            generate_world_model_benchmark_from_perception_benchmark(perception_path, db_path, world_model_path)

            eval_dirs = []
            for profile in ["oracle_rollout", "kinematic_rollout"]:
                predictions_path = root / f"{profile}.json"
                output_dir = root / profile
                generate_proxy_world_model_predictions(world_model_path, predictions_path, profile)
                evaluate_world_model_predictions(world_model_path, predictions_path, output_dir, profile_name=profile)
                eval_dirs.append(output_dir)

            study_dir = root / "case_studies"
            metadata = render_world_model_case_studies(world_model_path, eval_dirs, study_dir, max_cases=1)

            self.assertEqual(metadata["case_count"], 1)
            self.assertTrue((study_dir / "world_model_case_studies.png").exists())
            self.assertTrue((study_dir / "world_model_case_studies.json").exists())
            self.assertTrue((study_dir / "world_model_case_studies.md").exists())
            payload = json.loads((study_dir / "world_model_case_studies.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["case_count"], 1)


if __name__ == "__main__":
    unittest.main()
