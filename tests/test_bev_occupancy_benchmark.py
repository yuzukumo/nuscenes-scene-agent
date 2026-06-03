import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from nusc_scene_agent.bev_occupancy_benchmark import (
    adapt_perception_predictions_to_bev_occupancy,
    evaluate_bev_occupancy_predictions,
    generate_bev_occupancy_benchmark_from_perception_benchmark,
    generate_proxy_bev_occupancy_predictions,
    run_proxy_bev_occupancy_study,
)
from nusc_scene_agent.experiment_config import run_experiment_config
from nusc_scene_agent.perception_benchmark import generate_perception_benchmark_from_scenario_config
from test_perception_benchmark import _build_test_db, _write_scenario_config


def _add_context_agents(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    rows = [
        (
            f"ctx-{idx}",
            f"s{idx}",
            "scene-token-1",
            "scene-0001",
            idx,
            "inst-2",
            "vehicle.car",
            "vehicle",
            10.0 + idx,
            3.0,
            11.0 + idx,
            4,
            None,
            0.0,
            0.0,
            0.0,
            0.0,
            8,
            0,
        )
        for idx in range(5)
    ]
    conn.executemany("INSERT INTO agents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()


class BevOccupancyBenchmarkTest(unittest.TestCase):
    def _create_perception_benchmark(self, root: Path) -> tuple[Path, Path]:
        config_path = root / "scenario.yaml"
        db_path = root / "index.sqlite"
        perception_path = root / "perception.json"
        _write_scenario_config(config_path)
        _build_test_db(db_path)
        _add_context_agents(db_path)
        generate_perception_benchmark_from_scenario_config(config_path, db_path, perception_path)
        return perception_path, db_path

    def test_generate_and_evaluate_bev_occupancy_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            perception_path, db_path = self._create_perception_benchmark(root)
            benchmark_path = root / "bev_occupancy.json"
            metadata = generate_bev_occupancy_benchmark_from_perception_benchmark(perception_path, db_path, benchmark_path)

            self.assertEqual(metadata["case_count"], 1)
            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
            case = benchmark["cases"][0]
            self.assertEqual(benchmark["metadata"]["schema"], "risk_bev_occupancy_benchmark_v1")
            self.assertGreater(case["frames"][0]["context_actor_count"], 0)
            self.assertTrue(case["frames"][0]["occupied_cells"])

            oracle_predictions = root / "oracle.json"
            risk_only_predictions = root / "risk_only.json"
            generate_proxy_bev_occupancy_predictions(benchmark_path, oracle_predictions, "oracle_occupancy")
            generate_proxy_bev_occupancy_predictions(benchmark_path, risk_only_predictions, "risk_actor_only")

            oracle_summary = evaluate_bev_occupancy_predictions(benchmark_path, oracle_predictions, root / "oracle_eval")
            risk_only_summary = evaluate_bev_occupancy_predictions(
                benchmark_path,
                risk_only_predictions,
                root / "risk_only_eval",
            )

            self.assertEqual(oracle_summary["overview"]["mean_occupancy_iou"], 1.0)
            self.assertEqual(oracle_summary["overview"]["mean_primary_actor_recall"], 1.0)
            self.assertLess(risk_only_summary["overview"]["mean_context_recall"], 1.0)
            self.assertTrue((root / "oracle_eval" / "bev_occupancy_metrics_summary.md").exists())
            self.assertTrue((root / "oracle_eval" / "bev_occupancy_case_metrics.csv").exists())

    def test_adapt_perception_predictions_to_bev_occupancy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            perception_path, db_path = self._create_perception_benchmark(root)
            benchmark_path = root / "bev_occupancy.json"
            generate_bev_occupancy_benchmark_from_perception_benchmark(perception_path, db_path, benchmark_path)
            perception = json.loads(perception_path.read_text(encoding="utf-8"))
            case = perception["cases"][0]
            perception_predictions_path = root / "perception_predictions.json"
            perception_predictions_path.write_text(
                json.dumps(
                    {
                        "metadata": {"profile_name": "primary_only_detector"},
                        "predictions": [
                            {
                                "sample_token": frame["sample_token"],
                                "sample_idx": frame["sample_idx"],
                                "track_id": "primary",
                                "category_group": "pedestrian",
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

            adapted_path = root / "adapted_bev.json"
            metadata = adapt_perception_predictions_to_bev_occupancy(
                benchmark_path,
                perception_predictions_path,
                adapted_path,
            )
            self.assertEqual(metadata["profile_name"], "primary_only_detector")
            summary = evaluate_bev_occupancy_predictions(benchmark_path, adapted_path, root / "adapted_eval")
            self.assertEqual(summary["overview"]["mean_primary_actor_recall"], 1.0)
            self.assertLess(summary["overview"]["mean_context_recall"], 1.0)

    def test_proxy_study_and_experiment_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            perception_path, db_path = self._create_perception_benchmark(root)
            benchmark_path = root / "bev_occupancy.json"
            generate_bev_occupancy_benchmark_from_perception_benchmark(perception_path, db_path, benchmark_path)

            study = run_proxy_bev_occupancy_study(benchmark_path, root / "study")
            self.assertEqual(study["profile_count"], 3)
            self.assertTrue((root / "study" / "bev_occupancy_leaderboard.csv").exists())

            config_path = root / "config.yaml"
            output_dir = root / "configured_study"
            config_path.write_text(
                "\n".join(
                    [
                        "experiment:",
                        "  id: bev_occupancy_test",
                        "  type: bev_occupancy_study",
                        f"  output: {output_dir}",
                        f"  result_path: {output_dir / 'result.json'}",
                        "bev_occupancy:",
                        f"  perception_benchmark: {perception_path}",
                        f"  db: {db_path}",
                        f"  benchmark_output: {root / 'configured_benchmark.json'}",
                        f"  output: {output_dir}",
                    ]
                ),
                encoding="utf-8",
            )
            result = run_experiment_config(config_path)
            self.assertEqual(result["experiment_type"], "bev_occupancy_study")
            self.assertEqual(result["result"]["study"]["profile_count"], 3)


if __name__ == "__main__":
    unittest.main()
