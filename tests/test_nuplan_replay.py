import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from nusc_scene_agent.nuplan_replay import (
    compare_nuplan_replay_evaluations,
    evaluate_nuplan_rollouts,
    generate_nuplan_proxy_rollouts,
    generate_nuplan_replay_benchmark,
    inspect_nuplan_dataset,
    render_nuplan_replay_case_studies,
)


def _token(value: int) -> bytes:
    return value.to_bytes(8, "big")


def _build_nuplan_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE log (
            token BLOB PRIMARY KEY,
            vehicle_name VARCHAR(64),
            date VARCHAR(64),
            timestamp INTEGER,
            logfile VARCHAR(64),
            location VARCHAR(64),
            map_version VARCHAR(64)
        );

        CREATE TABLE scene (
            token BLOB PRIMARY KEY,
            log_token BLOB NOT NULL,
            name TEXT,
            goal_ego_pose_token BLOB,
            roadblock_ids TEXT
        );

        CREATE TABLE ego_pose (
            token BLOB PRIMARY KEY,
            timestamp INTEGER,
            x FLOAT,
            y FLOAT,
            z FLOAT,
            qw FLOAT,
            qx FLOAT,
            qy FLOAT,
            qz FLOAT,
            vx FLOAT,
            vy FLOAT,
            vz FLOAT,
            acceleration_x FLOAT,
            acceleration_y FLOAT,
            acceleration_z FLOAT,
            angular_rate_x FLOAT,
            angular_rate_y FLOAT,
            angular_rate_z FLOAT,
            epsg INTEGER,
            log_token BLOB NOT NULL
        );

        CREATE TABLE lidar_pc (
            token BLOB PRIMARY KEY,
            next_token BLOB,
            prev_token BLOB,
            ego_pose_token BLOB NOT NULL,
            lidar_token BLOB NOT NULL,
            scene_token BLOB,
            filename VARCHAR(128),
            timestamp INTEGER
        );

        CREATE TABLE category (
            token BLOB PRIMARY KEY,
            name VARCHAR(64),
            description TEXT
        );

        CREATE TABLE track (
            token BLOB PRIMARY KEY,
            category_token BLOB NOT NULL,
            width FLOAT,
            length FLOAT,
            height FLOAT
        );

        CREATE TABLE lidar_box (
            token BLOB PRIMARY KEY,
            lidar_pc_token BLOB NOT NULL,
            track_token BLOB NOT NULL,
            next_token BLOB,
            prev_token BLOB,
            x FLOAT,
            y FLOAT,
            z FLOAT,
            width FLOAT,
            length FLOAT,
            height FLOAT,
            vx FLOAT,
            vy FLOAT,
            vz FLOAT,
            yaw FLOAT,
            confidence FLOAT
        );

        CREATE TABLE scenario_tag (
            token BLOB PRIMARY KEY,
            lidar_pc_token BLOB NOT NULL,
            type TEXT,
            agent_track_token BLOB
        );

        CREATE TABLE traffic_light_status (
            token BLOB PRIMARY KEY,
            lidar_pc_token BLOB NOT NULL,
            lane_connector_id INTEGER,
            status VARCHAR(8)
        );
        """
    )
    log_token = _token(1)
    scene_token = _token(2)
    category_token = _token(3)
    track_token = _token(4)
    conn.execute(
        "INSERT INTO log VALUES (?, ?, ?, ?, ?, ?, ?)",
        (log_token, "veh-test", "2021-01-01", 0, "test_log", "boston", "us-ma-boston"),
    )
    conn.execute(
        "INSERT INTO scene VALUES (?, ?, ?, ?, ?)",
        (scene_token, log_token, "scene-0001", None, ""),
    )
    conn.execute(
        "INSERT INTO category VALUES (?, ?, ?)",
        (category_token, "pedestrian", "test pedestrian"),
    )
    conn.execute(
        "INSERT INTO track VALUES (?, ?, ?, ?, ?)",
        (track_token, category_token, 0.8, 0.8, 1.8),
    )

    timestamps = [0, 500_000, 1_000_000, 1_500_000, 2_000_000]
    lidar_tokens = [_token(100 + idx) for idx in range(len(timestamps))]
    ego_tokens = [_token(200 + idx) for idx in range(len(timestamps))]
    for idx, timestamp_us in enumerate(timestamps):
        ego_x = float(idx)
        conn.execute(
            "INSERT INTO ego_pose VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ego_tokens[idx],
                timestamp_us,
                ego_x,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                2.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0,
                log_token,
            ),
        )
        conn.execute(
            "INSERT INTO lidar_pc VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                lidar_tokens[idx],
                lidar_tokens[idx + 1] if idx + 1 < len(lidar_tokens) else None,
                lidar_tokens[idx - 1] if idx else None,
                ego_tokens[idx],
                _token(300),
                scene_token,
                f"test/{idx}.pcd",
                timestamp_us,
            ),
        )
        conn.execute(
            "INSERT INTO lidar_box VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _token(400 + idx),
                lidar_tokens[idx],
                track_token,
                None,
                None,
                5.0,
                0.0,
                0.0,
                0.8,
                0.8,
                1.8,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ),
        )

    conn.execute(
        "INSERT INTO scenario_tag VALUES (?, ?, ?, ?)",
        (_token(500), lidar_tokens[2], "near_pedestrian_on_crosswalk", track_token),
    )
    conn.execute(
        "INSERT INTO traffic_light_status VALUES (?, ?, ?, ?)",
        (_token(600), lidar_tokens[2], 1, "red"),
    )
    conn.commit()
    conn.close()


class NuPlanReplayTest(unittest.TestCase):
    def test_inspect_nuplan_dataset_counts_cache_and_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            db_dir = root / "data/cache/mini"
            split_dir = root / "nuplan-v1.1/splits/mini"
            map_dir = root / "maps/us-ma-boston/1.0"
            db_dir.mkdir(parents=True)
            map_dir.mkdir(parents=True)
            (db_dir / "test.db").write_text("", encoding="utf-8")
            split_dir.parent.mkdir(parents=True)
            split_dir.symlink_to("../../data/cache/mini")
            (map_dir / "map.json").write_text("{}", encoding="utf-8")

            inventory = inspect_nuplan_dataset(root)

            self.assertEqual(inventory["cache_counts"]["mini"], 1)
            self.assertEqual(inventory["split_counts"]["mini"], 1)
            self.assertEqual(inventory["map_json_count"], 1)

    def test_generate_replay_benchmark_and_evaluate_logged_ego(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            split_dir = root / "split"
            split_dir.mkdir()
            db_path = split_dir / "mini.db"
            benchmark_path = root / "nuplan_replay.json"
            predictions_path = root / "logged_ego_rollouts.json"
            stopped_predictions_path = root / "stopped_rollouts.json"
            eval_dir = root / "eval"
            stopped_eval_dir = root / "stopped_eval"
            comparison_dir = root / "comparison"
            case_studies_dir = root / "case_studies"
            _build_nuplan_db(db_path)

            metadata = generate_nuplan_replay_benchmark(
                split_dir=split_dir,
                output_path=benchmark_path,
                max_dbs=1,
                max_cases=1,
                max_cases_per_db=1,
                history_s=0.5,
                future_s=1.0,
                frame_hz=2.0,
            )
            self.assertEqual(metadata["schema"], "nuplan_replay_benchmark_v2")
            self.assertEqual(metadata["case_count"], 1)
            self.assertIn("taxonomy", metadata)
            self.assertIn("vru_interaction", metadata["taxonomy"])
            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
            case = benchmark["cases"][0]
            self.assertEqual(case["scenario_family"], "vru_interaction")
            self.assertEqual(case["difficulty_label"], "hard")
            self.assertEqual(case["risk_facets"]["actor_category"], "pedestrian")
            self.assertEqual(case["risk_facets"]["distance_band"], "collision_proxy")
            self.assertIn("comfort_targets", case)
            self.assertEqual(case["history_frame_count"], 1)
            self.assertEqual(case["future_frame_count"], 3)
            self.assertLess(case["risk_targets"]["min_distance_m"], 4.0)

            rollout_metadata = generate_nuplan_proxy_rollouts(
                benchmark_path=benchmark_path,
                output_path=predictions_path,
                profile_name="logged_ego",
            )
            self.assertEqual(rollout_metadata["prediction_count"], 1)

            summary = evaluate_nuplan_rollouts(
                benchmark_path=benchmark_path,
                predictions_path=predictions_path,
                output_dir=eval_dir,
            )
            self.assertEqual(summary["overview"]["case_count"], 1)
            self.assertEqual(summary["overview"]["full_horizon_count"], 1)
            self.assertEqual(summary["overview"]["mean_ego_ade_m"], 0.0)
            self.assertEqual(summary["overview"]["mean_risk_fidelity_score"], 1.0)
            self.assertEqual(summary["overview"]["mean_min_ttc_error_s"], 0.0)
            self.assertEqual(summary["overview"]["mean_red_light_context_recall"], 1.0)
            self.assertTrue((eval_dir / "nuplan_replay_metrics.json").exists())
            self.assertTrue((eval_dir / "nuplan_replay_case_metrics.csv").exists())
            self.assertTrue((eval_dir / "nuplan_replay_metrics_summary.md").exists())

            generate_nuplan_proxy_rollouts(
                benchmark_path=benchmark_path,
                output_path=stopped_predictions_path,
                profile_name="stopped",
            )
            evaluate_nuplan_rollouts(
                benchmark_path=benchmark_path,
                predictions_path=stopped_predictions_path,
                output_dir=stopped_eval_dir,
            )

            comparison = compare_nuplan_replay_evaluations(
                evaluation_dirs=[eval_dir, stopped_eval_dir],
                output_dir=comparison_dir,
            )
            self.assertEqual(comparison["overview"]["profile_count"], 2)
            self.assertEqual(comparison["overview"]["case_count"], 1)
            self.assertTrue((comparison_dir / "nuplan_replay_comparison.json").exists())
            self.assertTrue((comparison_dir / "nuplan_replay_leaderboard.csv").exists())
            self.assertTrue((comparison_dir / "nuplan_replay_comparison_summary.md").exists())

            case_study_metadata = render_nuplan_replay_case_studies(
                benchmark_path=benchmark_path,
                evaluation_dirs=[eval_dir, stopped_eval_dir],
                output_dir=case_studies_dir,
                max_cases=1,
            )
            self.assertEqual(case_study_metadata["case_count"], 1)
            self.assertTrue((case_studies_dir / "nuplan_replay_case_studies.png").exists())
            self.assertTrue((case_studies_dir / "nuplan_replay_case_studies.json").exists())
            self.assertTrue((case_studies_dir / "nuplan_replay_case_studies.md").exists())
            self.assertTrue((case_studies_dir / "nuplan_replay_case_studies.html").exists())


if __name__ == "__main__":
    unittest.main()
