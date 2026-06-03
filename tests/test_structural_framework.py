import json
import tempfile
import unittest
from pathlib import Path

from nusc_scene_agent.artifact_manifest import build_artifact_entry, write_artifact_manifest
from nusc_scene_agent.benchmark_registry import build_default_benchmark_registry, write_benchmark_registry
from nusc_scene_agent.dataset_backends import inspect_dataset_backends
from nusc_scene_agent.experiment_config import run_experiment_config
from nusc_scene_agent.unified_schema import (
    load_unified_case_source,
    unified_cases_from_nuplan_benchmark,
    unified_cases_from_nuscenes_case_library,
    write_unified_case_collection,
)
from test_perception_benchmark import _build_test_db


class StructuralFrameworkTest(unittest.TestCase):
    def test_unified_cases_from_nuplan_benchmark(self) -> None:
        benchmark = {
            "cases": [
                {
                    "case_id": "nuplan_case",
                    "source_db": "mini.db",
                    "log_name": "log",
                    "scene_name": "scene",
                    "scenario_tag": "near_pedestrian_on_crosswalk",
                    "scenario_family": "vru_interaction",
                    "difficulty_label": "hard",
                    "location": "boston",
                    "anchor_timestamp_us": 1000,
                    "anchor_lidar_pc_token": "abc",
                    "anchor_track_token": "track",
                    "category_name": "pedestrian",
                    "map_version": "map",
                    "history_frame_count": 1,
                    "future_frame_count": 1,
                    "anchor_frame": {
                        "primary_actor": {
                            "category_name": "pedestrian",
                            "x": 1.0,
                            "y": 2.0,
                            "distance_m": 3.0,
                        }
                    },
                    "frames": [
                        {
                            "timestamp_us": 0,
                            "dt_from_anchor_s": -0.5,
                            "ego": {"x": 0.0, "y": 0.0, "yaw": 0.0, "vx": 1.0, "vy": 0.0},
                        }
                    ],
                    "future_frames": [
                        {
                            "timestamp_us": 1000,
                            "dt_from_anchor_s": 0.0,
                            "ego": {"x": 1.0, "y": 0.0, "yaw": 0.0, "vx": 1.0, "vy": 0.0},
                        }
                    ],
                    "risk_targets": {"min_distance_m": 3.0},
                    "comfort_targets": {"max_acceleration_mps2": 0.0},
                    "risk_facets": {"distance_band": "critical_range"},
                }
            ]
        }

        cases = unified_cases_from_nuplan_benchmark(benchmark)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].dataset, "nuplan")
        self.assertEqual(cases[0].benchmark_layer, "replay_regression")
        self.assertEqual(cases[0].scenario_family, "vru_interaction")
        self.assertEqual(cases[0].actors[0]["role"], "primary_risk_actor")
        self.assertEqual(len(cases[0].ego_history), 1)
        self.assertEqual(len(cases[0].ego_future), 1)

    def test_unified_cases_from_nuscenes_case_library(self) -> None:
        entries = [
            {
                "case_key": "sample:instance",
                "scene_name": "scene-0001",
                "scene_token": "scene-token",
                "sample_idx": 7,
                "sample_token": "sample",
                "instance_token": "instance",
                "category_name": "vehicle.car",
                "category_group": "vehicle",
                "location": "singapore",
                "passed": True,
                "validation_score": 91.0,
                "retrieval_score": 3.0,
                "min_distance_m": 2.0,
                "min_ttc_s": 1.0,
                "source_query_tags": ["vehicle", "lane_blocking"],
                "all_behaviors": ["stopped_lead"],
                "matched_behaviors": ["stopped_lead"],
            }
        ]

        cases = unified_cases_from_nuscenes_case_library(entries)

        self.assertEqual(cases[0].dataset, "nuscenes")
        self.assertEqual(cases[0].scenario_family, "blocked_path_interaction")
        self.assertEqual(cases[0].difficulty_label, "high_confidence")

    def test_write_and_load_unified_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "case_library.json"
            output = root / "unified.json"
            source.write_text(
                json.dumps(
                    [
                        {
                            "case_key": "sample:instance",
                            "sample_idx": 1,
                            "passed": False,
                            "validation_score": 10.0,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            cases = load_unified_case_source(source, "nuscenes_case_library")
            payload = write_unified_case_collection(cases, output, metadata={"source_type": "test"})

            self.assertTrue(output.exists())
            self.assertEqual(payload["schema"], "unified_risk_case_collection_v1")
            self.assertEqual(payload["cases"][0]["schema"], "unified_risk_case_v1")

    def test_dataset_backend_inventory_counts_local_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            nuscenes_root = root / "nuscenes"
            mini = nuscenes_root / "v1.0-mini"
            mini.mkdir(parents=True)
            (mini / "scene.json").write_text("[{}]", encoding="utf-8")
            (mini / "sample.json").write_text("[{}, {}]", encoding="utf-8")
            nuplan_root = root / "nuplan"
            (nuplan_root / "data/cache/mini").mkdir(parents=True)
            (nuplan_root / "data/cache/mini/test.db").write_text("", encoding="utf-8")

            inventory = inspect_dataset_backends(
                nuscenes_root=nuscenes_root,
                nuplan_root=nuplan_root,
                index_root=root / "index",
            )

            self.assertEqual(inventory["schema"], "dataset_backend_inventory_v1")
            self.assertEqual(inventory["backends"]["nuscenes"]["versions"]["v1.0-mini"]["counts"]["scene"], 1)
            self.assertEqual(inventory["backends"]["nuplan"]["cache_counts"]["mini"], 1)

    def test_benchmark_registry_and_artifact_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output = root / "registry.json"
            registry = write_benchmark_registry(output, build_default_benchmark_registry())
            artifact = root / "metrics.json"
            artifact.write_text("{}", encoding="utf-8")
            manifest = write_artifact_manifest(
                root,
                [build_artifact_entry(artifact, role="evaluation", kind="metrics", output_root=root)],
                metadata={"layer": "test"},
            )

            self.assertTrue(output.exists())
            self.assertIn("nuplan_replay_regression", registry["layers"])
            self.assertEqual(manifest["overview"]["existing_artifact_count"], 1)

    def test_registry_export_experiment_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = root / "registry.yaml"
            config.write_text(
                "\n".join(
                    [
                        "experiment:",
                        "  id: registry_test",
                        "  type: registry_export",
                        f"  output: {root / 'out'}",
                        f"  result_path: {root / 'result.json'}",
                    ]
                ),
                encoding="utf-8",
            )

            result = run_experiment_config(config)

            self.assertEqual(result["schema"], "experiment_result_v1")
            self.assertTrue((root / "out/benchmark_registry.json").exists())
            self.assertTrue((root / "out/dataset_backends.json").exists())

    def test_risk_benchmark_suite_experiment_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            db_path = root / "index.sqlite"
            case_library_path = root / "case_library.json"
            scenario_output = root / "scenario.yaml"
            perception_output = root / "perception.json"
            world_model_output = root / "world_model.json"
            bev_output = root / "bev_occupancy.json"
            output_dir = root / "suite"
            _build_test_db(db_path)
            case_library_path.write_text(
                json.dumps(
                    [
                        {
                            "case_key": "s2:inst-1",
                            "scene_name": "scene-0001",
                            "instance_token": "inst-1",
                            "category_name": "human.pedestrian.adult",
                            "category_group": "pedestrian",
                            "validation_score": 96.0,
                            "passed": True,
                            "min_distance_m": 1.1,
                            "min_ttc_s": 1.0,
                            "event_start_sample_idx": 0,
                            "event_end_sample_idx": 4,
                            "event_peak_sample_idx": 2,
                            "source_query_ids": ["crossing_close"],
                            "source_queries": ["pedestrian crossing close in front of ego"],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            config = root / "risk_suite.yaml"
            config.write_text(
                "\n".join(
                    [
                        "experiment:",
                        "  id: risk_suite_test",
                        "  type: risk_benchmark_suite",
                        f"  output: {output_dir}",
                        f"  result_path: {output_dir / 'result.json'}",
                        "risk_benchmark_suite:",
                        f"  case_library: {case_library_path}",
                        f"  db: {db_path}",
                        "  max_cases: 1",
                        f"  scenario_output: {scenario_output}",
                        f"  perception_output: {perception_output}",
                        f"  world_model_output: {world_model_output}",
                        f"  bev_occupancy_output: {bev_output}",
                        "  run_proxy_studies: true",
                    ]
                ),
                encoding="utf-8",
            )

            result = run_experiment_config(config)

            self.assertEqual(result["experiment_type"], "risk_benchmark_suite")
            self.assertEqual(result["result"]["scenario_mining"]["anchor_case_count"], 1)
            self.assertEqual(result["result"]["perception"]["case_count"], 1)
            self.assertEqual(result["result"]["world_model"]["case_count"], 1)
            self.assertEqual(result["result"]["bev_occupancy"]["case_count"], 1)
            self.assertEqual(result["result"]["proxy_studies"]["perception"]["case_count"], 1)
            self.assertEqual(result["result"]["proxy_studies"]["world_model"]["case_count"], 1)
            self.assertEqual(result["result"]["proxy_studies"]["bev_occupancy"]["case_count"], 1)
            self.assertTrue(scenario_output.exists())
            self.assertTrue(perception_output.exists())
            self.assertTrue(world_model_output.exists())
            self.assertTrue(bev_output.exists())
            self.assertTrue((output_dir / "result.json").exists())


if __name__ == "__main__":
    unittest.main()
