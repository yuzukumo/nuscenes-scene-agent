import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from nusc_scene_agent.perception_benchmark import (
    adapt_and_evaluate_nuscenes_predictions,
    adapt_filter_and_evaluate_nuscenes_predictions,
    adapt_nuscenes_predictions,
    compare_perception_evaluations,
    evaluate_perception_predictions,
    filter_perception_benchmark_by_predictions,
    generate_perception_benchmark_from_scenario_config,
    generate_proxy_perception_predictions,
    run_proxy_perception_study,
)


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
  - id: crossing_paraphrase
    description: Person moving through the ego lane
    top_k: 3
    expect_match: true
    benchmark_group: crossing_anchor
    variant_type: paraphrase
    tags: [scenario_mining, crossing, paraphrase]
    query:
      natural_language: Person moving through the ego lane
      actors: [person]
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
    conn.executemany(
        "INSERT INTO metadata VALUES (?, ?)",
        [
            ("schema", "nusc_scene_agent_index"),
            ("schema_version", "1"),
            ("build_complete", "true"),
            ("dataroot", str(path.parent / "data")),
        ],
    )
    conn.executemany("INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", sample_rows)
    conn.executemany("INSERT INTO agents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", agent_rows)
    conn.commit()
    conn.close()


class PerceptionBenchmarkTest(unittest.TestCase):
    def _create_benchmark(self, root: Path) -> Path:
        config_path = root / "scenario.yaml"
        db_path = root / "index.sqlite"
        benchmark_path = root / "perception_benchmark.json"
        _write_scenario_config(config_path)
        _build_test_db(db_path)
        generate_perception_benchmark_from_scenario_config(config_path, db_path, benchmark_path)
        return benchmark_path

    def test_generate_perception_benchmark_from_scenario_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "scenario.yaml"
            db_path = root / "index.sqlite"
            output_path = root / "perception_benchmark.json"
            _write_scenario_config(config_path)
            _build_test_db(db_path)

            metadata = generate_perception_benchmark_from_scenario_config(config_path, db_path, output_path)
            self.assertEqual(metadata["case_count"], 1)

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata"]["case_count"], 1)
            case = payload["cases"][0]
            self.assertEqual(case["benchmark_group"], "crossing_anchor")
            self.assertEqual(case["instance_token"], "inst-1")
            self.assertEqual(case["anchor_sample_token"], "s2")
            self.assertEqual(case["anchor_sample_idx"], 2)
            self.assertEqual(case["frame_count"], 5)
            self.assertEqual(case["primary_behavior"], "crossing")
            self.assertEqual(case["category_group"], "pedestrian")
            self.assertEqual(case["source_query_ids"], ["crossing_canonical", "crossing_paraphrase"])
            self.assertEqual(len(case["source_query_texts"]), 2)
            self.assertEqual(case["risk_facets"]["distance_band"], "critical_range")
            self.assertEqual(case["risk_facets"]["ttc_band"], "urgent_ttc")
            self.assertEqual(case["risk_facets"]["map_relation"], "crosswalk_like")

    def test_evaluate_perception_predictions_with_perfect_tracking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            benchmark_path = self._create_benchmark(root)
            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
            case = benchmark["cases"][0]
            predictions_path = root / "perfect_predictions.json"
            output_dir = root / "perfect_eval"
            predictions_path.write_text(
                json.dumps(
                    {
                        "metadata": {"profile_name": "perfect_tracking"},
                        "predictions": [
                            {
                                "sample_token": frame["sample_token"],
                                "sample_idx": frame["sample_idx"],
                                "track_id": "track-perfect",
                                "category_group": "person",
                                "x_ego": frame["x_ego"],
                                "y_ego": frame["y_ego"],
                                "score": 0.99,
                            }
                            for frame in case["frames"]
                        ],
                    }
                ),
                encoding="utf-8",
            )

            summary = evaluate_perception_predictions(benchmark_path, predictions_path, output_dir)

            self.assertEqual(summary["profile_name"], "perfect_tracking")
            self.assertEqual(summary["overview"]["case_count"], 1)
            self.assertEqual(summary["overview"]["anchor_recall_count"], 1)
            self.assertEqual(summary["overview"]["full_track_count"], 1)
            self.assertEqual(summary["overview"]["perfect_case_count"], 1)
            self.assertEqual(summary["overview"]["mean_event_recall"], 1.0)
            self.assertEqual(summary["overview"]["mean_contiguous_coverage"], 1.0)
            self.assertEqual(summary["overview"]["mean_center_error_m"], 0.0)
            self.assertEqual(summary["case_metrics"][0]["failure_tags"], [])
            self.assertIn("distance_band", summary["case_metrics"][0])
            self.assertIn("risk_breakdowns", summary)
            self.assertTrue((output_dir / "perception_metrics.json").exists())
            self.assertTrue((output_dir / "perception_metrics_summary.md").exists())
            self.assertTrue((output_dir / "perception_metrics_summary.html").exists())
            self.assertTrue((output_dir / "perception_case_metrics.csv").exists())

    def test_adapt_official_nuscenes_predictions_and_evaluate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            db_path = root / "index.sqlite"
            benchmark_path = self._create_benchmark(root)
            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
            case = benchmark["cases"][0]
            raw_tracking_path = root / "tracking_results.json"

            sample_global_centers = {
                frame["sample_token"]: {
                    "translation": [
                        100.0 + 2.0 * idx + float(frame["x_ego"]),
                        50.0 + 0.5 * idx + float(frame["y_ego"]),
                        0.0,
                    ]
                }
                for idx, frame in enumerate(case["frames"])
            }
            raw_tracking_path.write_text(
                json.dumps(
                    {
                        "meta": {"use_camera": True},
                        "results": {
                            sample_token: [
                                {
                                    "sample_token": sample_token,
                                    "translation": payload["translation"],
                                    "size": [0.7, 0.7, 1.7],
                                    "rotation": [1.0, 0.0, 0.0, 0.0],
                                    "velocity": [0.0, 0.0],
                                    "tracking_id": "ped-track-1",
                                    "tracking_name": "pedestrian",
                                    "tracking_score": 0.98,
                                }
                            ]
                            for sample_token, payload in sample_global_centers.items()
                        },
                    }
                ),
                encoding="utf-8",
            )

            adapted_tracking_path = root / "adapted_tracking.json"
            metadata = adapt_nuscenes_predictions(
                benchmark_path=benchmark_path,
                db_path=db_path,
                input_path=raw_tracking_path,
                output_path=adapted_tracking_path,
                task_type="tracking",
            )
            self.assertEqual(metadata["prediction_count"], 5)
            tracking_summary = evaluate_perception_predictions(
                benchmark_path=benchmark_path,
                predictions_path=adapted_tracking_path,
                output_dir=root / "adapted_tracking_eval",
                profile_name="official_tracking",
            )
            self.assertEqual(tracking_summary["overview"]["full_track_count"], 1)
            adapted_tracking = json.loads(adapted_tracking_path.read_text(encoding="utf-8"))
            self.assertEqual({row["track_id"] for row in adapted_tracking["predictions"]}, {"ped-track-1"})

            raw_detection_path = root / "detection_results.json"
            raw_detection_path.write_text(
                json.dumps(
                    {
                        "meta": {"use_camera": True},
                        "results": {
                            sample_token: [
                                {
                                    "sample_token": sample_token,
                                    "translation": payload["translation"],
                                    "size": [0.7, 0.7, 1.7],
                                    "rotation": [1.0, 0.0, 0.0, 0.0],
                                    "velocity": [0.0, 0.0],
                                    "detection_name": "pedestrian",
                                    "detection_score": 0.96,
                                }
                            ]
                            for sample_token, payload in sample_global_centers.items()
                        },
                    }
                ),
                encoding="utf-8",
            )
            adapted_detection_eval = adapt_and_evaluate_nuscenes_predictions(
                benchmark_path=benchmark_path,
                db_path=db_path,
                input_path=raw_detection_path,
                output_dir=root / "adapted_detection_eval",
                task_type="detection",
                profile_name="official_detection",
            )
            self.assertEqual(adapted_detection_eval["overview"]["full_track_count"], 1)
            self.assertTrue((root / "adapted_detection_eval" / "adapted_predictions.json").exists())
            self.assertTrue((root / "adapted_detection_eval" / "perception_metrics_summary.md").exists())

    def test_filter_perception_benchmark_by_predictions_and_evaluate_covered_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            db_path = root / "index.sqlite"
            benchmark_path = self._create_benchmark(root)
            raw_tracking_path = root / "tracking_results.json"
            raw_tracking_path.write_text(
                json.dumps(
                    {
                        "meta": {"use_camera": True},
                        "results": {
                            "s2": [
                                {
                                    "sample_token": "s2",
                                    "translation": [108.0, 51.1, 0.0],
                                    "size": [0.7, 0.7, 1.7],
                                    "rotation": [1.0, 0.0, 0.0, 0.0],
                                    "velocity": [0.0, 0.0],
                                    "tracking_id": "ped-track-1",
                                    "tracking_name": "pedestrian",
                                    "tracking_score": 0.98,
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            adapted_path = root / "adapted_tracking.json"
            adapt_nuscenes_predictions(
                benchmark_path=benchmark_path,
                db_path=db_path,
                input_path=raw_tracking_path,
                output_path=adapted_path,
                task_type="tracking",
            )
            filtered_benchmark_path = root / "filtered_benchmark.json"
            filter_metadata = filter_perception_benchmark_by_predictions(
                benchmark_path=benchmark_path,
                predictions_path=adapted_path,
                output_path=filtered_benchmark_path,
                coverage_mode="anchor",
            )
            self.assertEqual(filter_metadata["filtered_case_count"], 1)
            covered_eval = adapt_filter_and_evaluate_nuscenes_predictions(
                benchmark_path=benchmark_path,
                db_path=db_path,
                input_path=raw_tracking_path,
                output_dir=root / "covered_eval",
                task_type="tracking",
                profile_name="covered_tracking",
                coverage_mode="anchor",
            )
            self.assertEqual(covered_eval["filter"]["filtered_case_count"], 1)
            self.assertEqual(covered_eval["overview"]["case_count"], 1)
            self.assertTrue((root / "covered_eval" / "filtered_benchmark.json").exists())

    def test_compare_perception_evaluations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for idx, (name, recall) in enumerate([("profile_a", 1.0), ("profile_b", 0.5)]):
                eval_dir = root / name
                eval_dir.mkdir(parents=True, exist_ok=True)
                payload = {
                    "profile_name": name,
                    "overview": {
                        "case_count": 2,
                        "anchor_recall_count": 2,
                        "anchor_recall_rate": 1.0,
                        "full_track_count": 2 if idx == 0 else 1,
                        "full_track_rate": 1.0 if idx == 0 else 0.5,
                        "mean_event_recall": recall,
                        "mean_contiguous_coverage": recall,
                        "mean_center_error_m": 0.1 + idx,
                    },
                    "behavior_breakdown": [
                        {
                            "behavior": "crossing",
                            "case_count": 2,
                            "anchor_recall_count": 2,
                            "anchor_recall_rate": 1.0,
                            "full_track_count": 2 if idx == 0 else 1,
                            "full_track_rate": 1.0 if idx == 0 else 0.5,
                            "mean_event_recall": recall,
                            "mean_center_error_m": 0.1 + idx,
                            "top_failure_modes": [],
                            "top_failure_summary": "none",
                        }
                    ],
                    "risk_breakdowns": {
                        "distance_band": [
                            {
                                "distance_band": "critical_range",
                                "case_count": 2,
                                "anchor_recall_count": 2,
                                "anchor_recall_rate": 1.0,
                                "full_track_count": 2 if idx == 0 else 1,
                                "full_track_rate": 1.0 if idx == 0 else 0.5,
                                "mean_event_recall": recall,
                                "mean_center_error_m": 0.1 + idx,
                                "top_failure_modes": [],
                                "top_failure_summary": "none",
                            }
                        ]
                    },
                    "case_metrics": [],
                }
                (eval_dir / "perception_metrics.json").write_text(json.dumps(payload), encoding="utf-8")

            output_dir = root / "comparison"
            metadata = compare_perception_evaluations(
                evaluation_dirs=[root / "profile_a", root / "profile_b"],
                output_dir=output_dir,
            )
            self.assertEqual(metadata["profile_count"], 2)
            self.assertTrue((output_dir / "perception_comparison.json").exists())
            self.assertTrue((output_dir / "perception_comparison_summary.md").exists())
            comparison = json.loads((output_dir / "perception_comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(comparison["profiles"][0]["name"], "profile_a")
            self.assertIn("distance_band", comparison["risk_matrices"])

    def test_proxy_profiles_and_comparison_exports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            benchmark_path = self._create_benchmark(root)

            sparse_predictions_path = root / "crossing_sparse.json"
            generate_proxy_perception_predictions(
                benchmark_path=benchmark_path,
                output_path=sparse_predictions_path,
                profile_name="crossing_sparse_track",
            )
            sparse_summary = evaluate_perception_predictions(
                benchmark_path=benchmark_path,
                predictions_path=sparse_predictions_path,
                output_dir=root / "crossing_sparse_eval",
            )
            self.assertEqual(sparse_summary["overview"]["full_track_count"], 0)
            self.assertLess(sparse_summary["overview"]["mean_event_recall"], 1.0)
            self.assertTrue(sparse_summary["overview"]["anchor_recall_count"] >= 1)

            delayed_predictions_path = root / "delayed_track.json"
            generate_proxy_perception_predictions(
                benchmark_path=benchmark_path,
                output_path=delayed_predictions_path,
                profile_name="delayed_track",
            )
            delayed_summary = evaluate_perception_predictions(
                benchmark_path=benchmark_path,
                predictions_path=delayed_predictions_path,
                output_dir=root / "delayed_track_eval",
            )
            self.assertGreater(delayed_summary["overview"]["mean_first_match_lag_frames"], 0.0)

            study_output = root / "proxy_study"
            metadata = run_proxy_perception_study(benchmark_path=benchmark_path, output_dir=study_output)
            self.assertEqual(metadata["profile_count"], 3)
            self.assertEqual(metadata["case_count"], 1)
            self.assertTrue((study_output / "perception_comparison.json").exists())
            self.assertTrue((study_output / "perception_comparison_summary.md").exists())
            self.assertTrue((study_output / "perception_comparison_summary.html").exists())
            self.assertTrue((study_output / "perception_leaderboard.csv").exists())

            comparison = json.loads((study_output / "perception_comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(comparison["profiles"][0]["name"], "oracle_tracking")
            self.assertIn("risk_matrices", comparison)
            self.assertIn("distance_band", comparison["risk_matrices"])


if __name__ == "__main__":
    unittest.main()
