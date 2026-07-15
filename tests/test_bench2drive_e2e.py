import gzip
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nusc_scene_agent.bench2drive_e2e import (
    DEFAULT_BENCH2DRIVE_CAMERAS,
    VisionE2EModelConfig,
    VisionE2ELossConfig,
    _DistributedShardSamplerNoPadding,
    _build_vision_e2e_model,
    _bootstrap_prediction_metrics,
    compare_vision_e2e_prediction_sets,
    _planner_losses,
    _trajectory_calibration_comparison,
    build_bench2drive_vision_manifest,
    diagnose_vision_e2e_predictions,
    inspect_bench2drive_dataset,
)
from nusc_scene_agent.bench2drive_closed_loop import (
    ClosedLoopControlConfig,
    compare_bench2drive_closed_loop_reports,
)
from nusc_scene_agent.carla_closed_loop import (
    PurePursuitConfig,
    _discover_carla_maps,
    build_carla_launch_command,
    format_carla_launch_command,
)
from nusc_scene_agent.carla_vision_closed_loop import (
    CarlaVisionClosedLoopConfig,
    CarlaScenarioRuntime,
    _condition_ego_route_traffic_lights,
    _apply_scenario_safety_override,
    _build_carla_route_features,
    _carla_control_from_prediction,
    _crosswalk_target_distance,
    _group_crosswalk_locations,
    _normalize_scenario_specs,
    _pedestrian_yield_rollout_complete,
    _prediction_path_stats,
    _stabilize_straight_model_steer,
    _summarize_carla_vision_control_attribution,
    _update_scenario_runtime,
)
from nusc_scene_agent.carla_semantic_demo_mining import mine_carla_semantic_demos
from nusc_scene_agent.carla_video_audit import audit_carla_vision_rollouts
from nusc_scene_agent.experiment_config import run_experiment_config


class Bench2DriveE2ETest(unittest.TestCase):
    def test_prediction_comparison_uses_clip_level_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            baseline_path = root / "baseline.jsonl"
            candidate_path = root / "candidate.jsonl"
            baseline_rows = []
            candidate_rows = []
            for clip_idx in range(2):
                for frame_idx in range(2):
                    case_id = f"clip_{clip_idx}:{frame_idx}"
                    base = {
                        "case_id": case_id,
                        "clip_name": f"clip_{clip_idx}",
                        "scenario_family": "Test",
                        "target_future_waypoints_ego": [[0.0, -1.0], [0.0, -2.0]],
                        "predicted_future_waypoints_ego": [[0.0, -1.0], [0.0, -2.0]],
                        "target_should_brake": False,
                        "predicted_should_brake": False,
                    }
                    candidate = dict(base)
                    candidate["predicted_future_waypoints_ego"] = [[0.0, -0.5], [0.0, -1.5]]
                    baseline_rows.append(base)
                    candidate_rows.append(candidate)
            baseline_path.write_text("\n".join(json.dumps(row) for row in baseline_rows) + "\n", encoding="utf-8")
            candidate_path.write_text("\n".join(json.dumps(row) for row in candidate_rows) + "\n", encoding="utf-8")

            report = compare_vision_e2e_prediction_sets(
                baseline_path,
                candidate_path,
                root / "comparison",
                bootstrap_replicates=16,
            )

            metrics = {row["metric"]: row for row in report["metrics"]}
            self.assertEqual(report["case_alignment"]["paired_case_count"], 4)
            self.assertEqual(metrics["ade_m"]["oriented_improvement"], -0.5)
            self.assertEqual(metrics["brake_f1"]["baseline_mean"], 0.0)
            self.assertTrue((root / "comparison" / "prediction_comparison.png").exists())

    def test_build_manifest_materializes_camera_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "dataset"
            dataset_root.mkdir()
            archive_path = dataset_root / "Accident_Town01_Route1_Weather1.tar.gz"
            _build_bench2drive_archive(archive_path)

            inventory = inspect_bench2drive_dataset(dataset_root, sample_archives=1)
            self.assertEqual(inventory["archive_count"], 1)
            self.assertEqual(inventory["sample_archives"][0]["annotation_frame_count"], 3)

            manifest_path = root / "manifest.jsonl"
            cache_root = root / "cache"
            metadata = build_bench2drive_vision_manifest(
                dataset_root=dataset_root,
                output_path=manifest_path,
                frame_stride=1,
                future_steps=2,
                future_frame_stride=1,
                train_fraction=1.0,
                cache_root=cache_root,
            )

            self.assertEqual(metadata["row_count"], 1)
            row = json.loads(manifest_path.read_text(encoding="utf-8").strip())
            self.assertEqual(row["split"], "train")
            self.assertEqual(len(row["future_waypoints_ego"]), 2)
            self.assertEqual(set(row["camera_cache_paths"]), set(DEFAULT_BENCH2DRIVE_CAMERAS))
            for path in row["camera_cache_paths"].values():
                self.assertTrue(Path(path).exists())

    def test_distributed_validation_sampler_does_not_pad(self) -> None:
        shards = [
            list(_DistributedShardSamplerNoPadding(dataset_length=11, rank=rank, world_size=4))
            for rank in range(4)
        ]

        self.assertEqual(shards[0], [0, 4, 8])
        self.assertEqual(shards[1], [1, 5, 9])
        self.assertEqual(shards[2], [2, 6, 10])
        self.assertEqual(shards[3], [3, 7])
        flattened = [idx for shard in shards for idx in shard]
        self.assertEqual(sorted(flattened), list(range(11)))
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_prediction_diagnostics_identify_underreach(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            predictions_path = root / "predictions.jsonl"
            predictions = [
                {
                    "case_id": "case_a",
                    "scenario_family": "FollowLeadingVehicle",
                    "predicted_future_waypoints_ego": [[0.0, -0.2], [0.0, -0.3], [0.0, -0.4]],
                    "target_future_waypoints_ego": [[0.0, -2.0], [0.0, -4.0], [0.0, -6.0]],
                    "predicted_control": [0.0, 0.1, 0.0],
                    "target_control": [0.0, 0.5, 0.0],
                    "predicted_brake_probability": 0.05,
                    "target_should_brake": False,
                },
                {
                    "case_id": "case_b",
                    "scenario_family": "Accident",
                    "predicted_future_waypoints_ego": [[0.0, -0.1], [0.0, -0.2], [0.0, -0.3]],
                    "target_future_waypoints_ego": [[0.0, -1.5], [0.0, -3.0], [0.0, -4.5]],
                    "predicted_control": [0.0, 0.1, 0.0],
                    "target_control": [0.0, 0.0, 1.0],
                    "predicted_brake_probability": 0.05,
                    "target_should_brake": True,
                },
            ]
            predictions_path.write_text(
                "\n".join(json.dumps(row) for row in predictions) + "\n",
                encoding="utf-8",
            )

            report = diagnose_vision_e2e_predictions(
                predictions_path=predictions_path,
                output_dir=root / "diagnostics",
                brake_threshold=0.5,
            )

            self.assertEqual(report["sample_count"], 2)
            self.assertIn("predicted_horizon_underreach", report["readiness"]["findings"])
            self.assertIn("weak_brake_event_detection", report["readiness"]["findings"])
            self.assertTrue((root / "diagnostics" / "planner_diagnostics_report.json").exists())
            self.assertTrue((root / "diagnostics" / "planner_diagnostics_samples.csv").exists())

    def test_prediction_diagnostics_cli_defaults(self) -> None:
        from nusc_scene_agent.cli import _build_parser

        args = _build_parser().parse_args(["diagnose-bench2drive-vision-planner"])

        self.assertEqual(args.output, "outputs/bench2drive_vision_e2e_final/diagnostics")
        self.assertEqual(args.brake_threshold, 0.5)

    def test_closed_loop_cli_accepts_test_split(self) -> None:
        from nusc_scene_agent.cli import _build_parser

        args = _build_parser().parse_args(["run-bench2drive-vision-closed-loop", "--split", "test"])

        self.assertEqual(args.split, "test")

    def test_train_cli_exposes_lateral_aware_loss_options(self) -> None:
        from nusc_scene_agent.cli import _build_parser

        args = _build_parser().parse_args(
            [
                "train-bench2drive-vision-planner",
                "--lateral-loss-weight",
                "3.0",
                "--architecture",
                "trajectory_transformer",
                "--trajectory-modes",
                "4",
                "--trajectory-selection",
                "expected",
                "--trajectory-temperature",
                "0.5",
                "--turn-sample-weight",
                "4.0",
                "--turn-lateral-threshold-m",
                "1.5",
                "--mode-classification-weight",
                "0.2",
                "--selected-waypoint-loss-weight",
                "0.75",
                "--displacement-loss-weight",
                "0.2",
                "--endpoint-loss-weight",
                "0.1",
                "--path-length-loss-weight",
                "0.05",
                "--selection-metric",
                "lateral_aware",
            ]
        )

        self.assertEqual(args.architecture, "trajectory_transformer")
        self.assertEqual(args.trajectory_modes, 4)
        self.assertEqual(args.trajectory_selection, "expected")
        self.assertEqual(args.trajectory_temperature, 0.5)
        self.assertEqual(args.lateral_loss_weight, 3.0)
        self.assertEqual(args.turn_sample_weight, 4.0)
        self.assertEqual(args.turn_lateral_threshold_m, 1.5)
        self.assertEqual(args.mode_classification_weight, 0.2)
        self.assertEqual(args.selected_waypoint_loss_weight, 0.75)
        self.assertEqual(args.displacement_loss_weight, 0.2)
        self.assertEqual(args.endpoint_loss_weight, 0.1)
        self.assertEqual(args.path_length_loss_weight, 0.05)
        self.assertEqual(args.selection_metric, "lateral_aware")

    def test_lateral_weighted_loss_upweights_turn_samples(self) -> None:
        import torch

        prediction = {
            "future": torch.zeros((2, 4), dtype=torch.float32),
            "control": torch.zeros((2, 3), dtype=torch.float32),
            "brake_logits": torch.zeros((2, 1), dtype=torch.float32),
        }
        future = torch.tensor(
            [
                [0.0, -1.0, 0.0, -2.0],
                [3.0, -1.0, 4.0, -2.0],
            ],
            dtype=torch.float32,
        )
        control = torch.zeros((2, 3), dtype=torch.float32)
        brake = torch.zeros((2, 1), dtype=torch.float32)

        baseline = _planner_losses(torch, prediction, future, control, brake)
        weighted = _planner_losses(
            torch,
            prediction,
            future,
            control,
            brake,
            loss_config=VisionE2ELossConfig(
                lateral_loss_weight=3.0,
                turn_sample_weight=4.0,
                turn_lateral_threshold_m=2.0,
            ),
        )

        self.assertGreater(float(weighted["waypoint_loss"]), float(baseline["waypoint_loss"]))
        self.assertAlmostEqual(float(weighted["turn_sample_rate"]), 0.5)
        self.assertGreater(float(weighted["turn_lateral_mae_m"]), 0.0)

    def test_trajectory_transformer_outputs_multimodal_predictions(self) -> None:
        import torch

        config = VisionE2EModelConfig(
            model_size="tiny",
            architecture="trajectory_transformer",
            image_size=32,
            future_steps=5,
            trajectory_modes=3,
            camera_count=2,
            camera_pooling="transformer",
        )
        model = _build_vision_e2e_model(config)
        prediction = model(
            torch.zeros((2, 2, 3, 32, 32), dtype=torch.float32),
            torch.zeros((2, 8), dtype=torch.float32),
        )

        self.assertEqual(tuple(prediction["future"].shape), (2, 10))
        self.assertEqual(tuple(prediction["future_modes"].shape), (2, 3, 10))
        self.assertEqual(tuple(prediction["mode_logits"].shape), (2, 3))
        self.assertEqual(tuple(prediction["control"].shape), (2, 3))
        self.assertEqual(tuple(prediction["brake_logits"].shape), (2, 1))

    def test_multimodal_loss_reports_oracle_metrics(self) -> None:
        import torch

        prediction = {
            "future": torch.tensor([[0.0, -1.0, 0.0, -2.0]], dtype=torch.float32),
            "future_modes": torch.tensor(
                [
                    [
                        [10.0, -1.0, 10.0, -2.0],
                        [0.0, -1.0, 0.0, -2.0],
                    ]
                ],
                dtype=torch.float32,
            ),
            "mode_logits": torch.tensor([[0.0, 2.0]], dtype=torch.float32),
            "control": torch.zeros((1, 3), dtype=torch.float32),
            "brake_logits": torch.zeros((1, 1), dtype=torch.float32),
        }
        future = torch.tensor([[0.0, -1.0, 0.0, -2.0]], dtype=torch.float32)
        control = torch.zeros((1, 3), dtype=torch.float32)
        brake = torch.zeros((1, 1), dtype=torch.float32)

        losses = _planner_losses(torch, prediction, future, control, brake)

        self.assertAlmostEqual(float(losses["oracle_ade_m"]), 0.0, places=5)
        self.assertAlmostEqual(float(losses["oracle_fde_m"]), 0.0, places=5)
        self.assertGreaterEqual(float(losses["mode_loss"]), 0.0)

    def test_selected_waypoint_loss_is_reported_for_multimodal_output(self) -> None:
        import torch

        prediction = {
            "future": torch.zeros((1, 4), dtype=torch.float32),
            "future_modes": torch.zeros((1, 2, 4), dtype=torch.float32),
            "mode_logits": torch.zeros((1, 2), dtype=torch.float32),
            "control": torch.zeros((1, 3), dtype=torch.float32),
            "brake_logits": torch.zeros((1, 1), dtype=torch.float32),
        }
        future = torch.tensor([[1.0, -1.0, 1.0, -2.0]], dtype=torch.float32)
        control = torch.zeros((1, 3), dtype=torch.float32)
        brake = torch.zeros((1, 1), dtype=torch.float32)

        losses = _planner_losses(torch, prediction, future, control, brake)

        self.assertIn("selected_waypoint_loss", losses)
        self.assertGreater(float(losses["selected_waypoint_loss"]), 0.0)

    def test_dynamic_trajectory_losses_are_reported(self) -> None:
        import torch

        prediction = {
            "future": torch.tensor([[0.0, -0.5, 0.0, -1.0]], dtype=torch.float32),
            "control": torch.zeros((1, 3), dtype=torch.float32),
            "brake_logits": torch.zeros((1, 1), dtype=torch.float32),
        }
        future = torch.tensor([[0.0, -1.0, 0.0, -2.0]], dtype=torch.float32)
        losses = _planner_losses(
            torch,
            prediction,
            future,
            torch.zeros((1, 3), dtype=torch.float32),
            torch.zeros((1, 1), dtype=torch.float32),
            loss_config=VisionE2ELossConfig(
                displacement_weight=0.2,
                endpoint_weight=0.1,
                path_length_weight=0.05,
            ),
        )

        self.assertGreater(float(losses["displacement_loss"]), 0.0)
        self.assertGreater(float(losses["endpoint_loss"]), 0.0)
        self.assertGreater(float(losses["path_length_loss"]), 0.0)
        self.assertLess(float(losses["path_length_ratio"]), 1.0)

    def test_mode_calibrator_is_applied_inside_trajectory_model(self) -> None:
        import torch

        future_steps = 2
        mode_count = 2
        feature_count = future_steps * 2 + mode_count * future_steps * 2 + mode_count + 3 + 1
        config = VisionE2EModelConfig(
            model_size="tiny",
            architecture="trajectory_transformer",
            image_size=32,
            future_steps=future_steps,
            trajectory_modes=mode_count,
            camera_count=2,
            camera_pooling="transformer",
            trajectory_mode_calibrator={
                "enabled": True,
                "feature_mean": [0.0] * feature_count,
                "feature_scale": [1.0] * feature_count,
                "coef": [[0.0] * feature_count for _ in range(mode_count)],
                "intercept": [5.0, -5.0],
            },
        )
        model = _build_vision_e2e_model(config)
        prediction = model(
            torch.zeros((1, 2, 3, 32, 32), dtype=torch.float32),
            torch.zeros((1, 8), dtype=torch.float32),
        )

        probabilities = prediction["calibrated_mode_probabilities"]
        self.assertEqual(tuple(probabilities.shape), (1, mode_count))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=5)
        self.assertGreater(float(probabilities[0, 0]), 0.99)

    def test_trajectory_calibration_comparison_uses_paired_clip_bootstrap(self) -> None:
        rows = []
        for clip_name in ["clip_a", "clip_b"]:
            rows.append(
                {
                    "case_id": f"{clip_name}:0",
                    "uncalibrated_future_waypoints_ego": [[0.0, -0.5]],
                    "predicted_future_waypoints_ego": [[0.0, -0.9]],
                    "target_future_waypoints_ego": [[0.0, -1.0]],
                }
            )

        report = _trajectory_calibration_comparison(rows, seed=7, replicates=16)

        self.assertTrue(report["enabled"])
        self.assertEqual(report["cluster_count"], 2)
        self.assertLess(report["metrics"]["ade_m"]["delta"], 0.0)

    def test_prediction_bootstrap_resamples_clips(self) -> None:
        rows = []
        for clip_name in ["clip_a", "clip_b"]:
            for frame_idx in range(2):
                rows.append(
                    {
                        "case_id": f"{clip_name}:{frame_idx}",
                        "predicted_future_waypoints_ego": [[0.0, -1.0]],
                        "target_future_waypoints_ego": [[0.0, -1.0]],
                        "target_should_brake": False,
                        "predicted_should_brake": False,
                    }
                )
        uncertainty = _bootstrap_prediction_metrics(rows, seed=7, replicates=8)
        self.assertEqual(uncertainty["method"], "cluster-level percentile bootstrap")
        self.assertEqual(uncertainty["cluster_count"], 2)


class Bench2DriveClosedLoopConfigTest(unittest.TestCase):
    def test_closed_loop_comparison_uses_paired_case_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            baseline_path = root / "baseline.json"
            candidate_path = root / "candidate.json"
            baseline_cases = [
                {
                    "case_id": f"case_{idx}",
                    "scenario_family": "CutIn",
                    "closed_loop_ade_m": 10.0 + idx,
                    "closed_loop_fde_m": 20.0 + idx,
                    "mean_lateral_error_m": 2.0,
                    "route_completion": 0.5,
                    "closed_loop_score": 0.1,
                }
                for idx in range(4)
            ]
            candidate_cases = [
                {
                    **row,
                    "closed_loop_ade_m": row["closed_loop_ade_m"] - 2.0,
                    "closed_loop_fde_m": row["closed_loop_fde_m"] - 3.0,
                    "mean_lateral_error_m": 1.5,
                    "route_completion": 0.6,
                    "closed_loop_score": 0.2,
                }
                for row in baseline_cases
            ]
            baseline_path.write_text(
                json.dumps({"schema": "bench2drive_vision_closed_loop_v1", "cases": baseline_cases}),
                encoding="utf-8",
            )
            candidate_path.write_text(
                json.dumps({"schema": "bench2drive_vision_closed_loop_v1", "cases": candidate_cases}),
                encoding="utf-8",
            )

            report = compare_bench2drive_closed_loop_reports(
                baseline_path,
                candidate_path,
                root / "comparison",
                bootstrap_replicates=32,
            )

            metrics = {row["metric"]: row for row in report["metrics"]}
            self.assertEqual(report["case_alignment"]["paired_case_count"], 4)
            self.assertTrue(report["case_alignment"]["identical_case_sets"])
            self.assertEqual(metrics["closed_loop_ade_m"]["oriented_improvement"], 2.0)
            self.assertEqual(metrics["route_completion"]["oriented_improvement"], 0.1)
            self.assertTrue((root / "comparison" / "closed_loop_comparison.json").exists())
            self.assertTrue((root / "comparison" / "closed_loop_comparison.png").exists())

    def test_closed_loop_control_defaults_use_validated_calibration(self) -> None:
        config = ClosedLoopControlConfig()

        self.assertEqual(config.horizon_s, 10.0)
        self.assertEqual(config.target_speed_mps, 5.5)
        self.assertEqual(config.brake_probability_threshold, 0.85)
        self.assertEqual(config.lookahead_m, 9.0)
        self.assertEqual(config.speed_kp, 0.45)

    def test_closed_loop_cli_exposes_control_calibration(self) -> None:
        from nusc_scene_agent.cli import _build_parser

        args = _build_parser().parse_args(["run-bench2drive-vision-closed-loop"])

        self.assertEqual(args.max_cases, 64)
        self.assertEqual(args.max_frames_per_clip, 20)
        self.assertFalse(args.render_case_media)
        self.assertEqual(args.horizon_s, 10.0)
        self.assertEqual(args.target_speed_mps, 5.5)
        self.assertEqual(args.brake_threshold, 0.85)
        self.assertEqual(args.lookahead_m, 9.0)
        self.assertEqual(args.speed_kp, 0.45)

    def test_closed_loop_yaml_uses_validated_calibration(self) -> None:
        import yaml

        config_path = Path(__file__).resolve().parents[1] / "configs" / "bench2drive_vision_closed_loop.yaml"
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        closed_loop = payload["bench2drive_vision_closed_loop"]

        self.assertEqual(closed_loop["max_cases"], 64)
        self.assertEqual(closed_loop["max_frames_per_clip"], 20)
        self.assertFalse(closed_loop["render_case_media"])
        self.assertEqual(closed_loop["target_speed_mps"], 5.5)
        self.assertEqual(closed_loop["brake_threshold"], 0.85)
        self.assertEqual(closed_loop["lookahead_m"], 9.0)
        self.assertEqual(closed_loop["speed_kp"], 0.45)

    def test_carla_vision_cli_exposes_rpc_timeout(self) -> None:
        from nusc_scene_agent.cli import _build_parser

        args = _build_parser().parse_args(["run-carla-vision-closed-loop"])

        self.assertEqual(args.rpc_timeout_s, 30.0)

    def test_carla_video_audit_reports_missing_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            states_path = root / "states.csv"
            states_path.write_text(
                "\n".join(
                    [
                        "step,speed_mps,lateral_error_m,ego_control_mode,safety_brake,scenario_phase",
                        "0,1.0,0.1,e2e_waypoint_control,0.0,ambient_traffic",
                    ]
                ),
                encoding="utf-8",
            )
            report_path = root / "carla_vision_batch_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "schema": "carla_vision_closed_loop_batch_v1",
                        "scenarios": [
                            {
                                "name": "demo",
                                "metrics": {"collision_count": 0, "route_completion": 0.5},
                                "control_attribution": {
                                    "direct_model_control_ratio": 1.0,
                                    "traffic_manager_vehicle_count": 2,
                                    "scripted_vehicle_count": 0,
                                },
                                "media": {
                                    "video_path": str(root / "missing.mp4"),
                                    "states_csv": str(states_path),
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            audit = audit_carla_vision_rollouts(
                report_path=report_path,
                output_path=root / "audit.json",
                min_frames=1,
                min_resolution_width=1,
                min_resolution_height=1,
            )

            self.assertEqual(audit["status"], "fail")
            self.assertIn("demo:missing_video", audit["failures"])
            self.assertTrue((root / "audit.json").exists())
            self.assertTrue((root / "audit.md").exists())

    def test_carla_video_audit_cli_defaults(self) -> None:
        from nusc_scene_agent.cli import _build_parser

        args = _build_parser().parse_args(["audit-carla-vision-rollout"])

        self.assertEqual(args.report, "outputs/carla_semantic_demo_final/carla_semantic_demo_report.json")
        self.assertEqual(args.min_resolution_width, 1920)
        self.assertEqual(args.max_scripted_vehicles, 0)
        self.assertEqual(args.max_mean_lateral_error_m, 2.5)
        self.assertEqual(args.max_nearest_actor_distance_m, 60.0)
        self.assertEqual(args.nearby_actor_distance_m, 30.0)
        self.assertEqual(args.min_nearby_actor_ratio, 0.30)
        self.assertFalse(args.require_semantic_match)

    def test_carla_video_audit_falls_back_to_scenario_distance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            states_path = root / "states.csv"
            states_path.write_text(
                "\n".join(
                    [
                        "step,speed_mps,lateral_error_m,ego_control_mode,safety_brake,scenario_phase,scenario_actor_distance_m",
                        "0,1.0,0.1,e2e_waypoint_control,0.0,ambient_traffic,12.0",
                        "1,1.0,0.1,e2e_waypoint_control,0.0,ambient_traffic,24.0",
                        "2,1.0,0.1,e2e_waypoint_control,0.0,ambient_traffic,48.0",
                    ]
                ),
                encoding="utf-8",
            )
            report_path = root / "report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "schema": "carla_vision_closed_loop_batch_v1",
                        "scenarios": [
                            {
                                "name": "demo",
                                "metrics": {
                                    "collision_count": 0,
                                    "route_completion": 0.5,
                                    "mean_lateral_error_m": 0.1,
                                },
                                "control_attribution": {
                                    "direct_model_control_ratio": 1.0,
                                    "traffic_manager_vehicle_count": 2,
                                    "scripted_vehicle_count": 0,
                                },
                                "media": {
                                    "video_path": str(root / "missing.mp4"),
                                    "states_csv": str(states_path),
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            audit = audit_carla_vision_rollouts(
                report_path=report_path,
                output_path=root / "audit.json",
                min_frames=1,
                min_resolution_width=1,
                min_resolution_height=1,
                min_nearby_actor_ratio=0.5,
            )
            visibility = audit["scenarios"][0]["traffic_visibility"]

            self.assertEqual(visibility["nearby_actor_ratio"], 2 / 3)
            self.assertEqual(visibility["min_actor_distance_m"], 12.0)
            self.assertNotIn("demo:insufficient_nearby_natural_traffic", audit["failures"])

    def test_carla_video_audit_can_require_dense_scene_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            states_path = root / "states.csv"
            states_path.write_text(
                "\n".join(
                    [
                        "step,speed_mps,lateral_error_m,ego_control_mode,safety_brake,scenario_phase,"
                        "natural_traffic_nearest_distance_m,natural_traffic_front_actor_count,"
                        "natural_traffic_same_lane_front_actor_count,natural_traffic_adjacent_actor_count",
                        "0,1.0,0.1,e2e_waypoint_control,0.0,natural_dense_traffic,12.0,0,0,1",
                        "1,1.0,0.1,e2e_waypoint_control,0.0,natural_dense_traffic,18.0,0,0,1",
                        "2,1.0,0.1,e2e_waypoint_control,0.0,natural_dense_traffic,22.0,0,0,1",
                    ]
                ),
                encoding="utf-8",
            )
            report_path = root / "report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "schema": "carla_vision_closed_loop_batch_v1",
                        "scenarios": [
                            {
                                "name": "dense_traffic_follow",
                                "type": "dense_follow_overtake",
                                "metrics": {
                                    "collision_count": 0,
                                    "route_completion": 0.5,
                                    "mean_lateral_error_m": 0.1,
                                },
                                "control_attribution": {
                                    "direct_model_control_ratio": 1.0,
                                    "traffic_manager_vehicle_count": 2,
                                    "scripted_vehicle_count": 0,
                                },
                                "media": {
                                    "video_path": str(root / "missing.mp4"),
                                    "states_csv": str(states_path),
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            audit = audit_carla_vision_rollouts(
                report_path=report_path,
                output_path=root / "audit.json",
                min_frames=1,
                min_resolution_width=1,
                min_resolution_height=1,
                require_semantic_match=True,
            )
            scenario = audit["scenarios"][0]

            self.assertEqual(scenario["semantic_evidence"]["target"], "dense_follow")
            self.assertFalse(scenario["semantic_evidence"]["passed"])
            self.assertIn(
                "dense_traffic_follow:semantic_dense_follow_missing_front_vehicle_context",
                audit["failures"],
            )

    def test_carla_video_audit_requires_complete_right_turn_pedestrian_yield(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            states_path = root / "states.csv"
            states_path.write_text(
                "\n".join(
                    [
                        "step,speed_mps,lateral_error_m,ego_control_mode,safety_brake,scenario_phase,command,"
                        "route_progress_m,walker_route_progress_m,walker_route_lateral_m,walker_crossing_completion",
                        "0,1.0,0.1,e2e_waypoint_control,0.0,pedestrian_crossing,2,0.0,10.0,-4.0,0.10",
                        "1,0.0,0.1,e2e_waypoint_control,1.0,pedestrian_crossing,2,0.0,10.0,0.0,0.60",
                        "2,0.8,0.1,e2e_waypoint_control,0.0,pedestrian_cleared,2,0.2,10.0,4.2,1.00",
                        "3,1.2,0.1,e2e_waypoint_control,0.0,pedestrian_cleared,2,0.8,10.0,4.2,1.00",
                    ]
                ),
                encoding="utf-8",
            )
            report_path = root / "report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "schema": "carla_vision_closed_loop_batch_v1",
                        "scenarios": [
                            {
                                "name": "right_turn_pedestrian_yield",
                                "type": "pedestrian_crossing",
                                "metrics": {
                                    "collision_count": 0,
                                    "route_completion": 0.5,
                                    "mean_lateral_error_m": 0.1,
                                },
                                "control_attribution": {
                                    "direct_model_control_ratio": 1.0,
                                    "traffic_manager_vehicle_count": 1,
                                    "scripted_vehicle_count": 0,
                                    "crosswalk_walker_count": 1,
                                    "safety_override_ratio": 0.33,
                                },
                                "media": {
                                    "video_path": str(root / "missing.mp4"),
                                    "states_csv": str(states_path),
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            audit = audit_carla_vision_rollouts(
                report_path=report_path,
                output_path=root / "audit.json",
                min_frames=1,
                min_resolution_width=1,
                min_resolution_height=1,
                require_semantic_match=True,
            )
            scenario = audit["scenarios"][0]

            self.assertEqual(scenario["semantic_evidence"]["target"], "pedestrian_yield")
            self.assertTrue(scenario["semantic_evidence"]["passed"])
            self.assertNotIn(
                "right_turn_pedestrian_yield:semantic_pedestrian_yield_pedestrian_did_not_complete_crossing",
                audit["failures"],
            )

    def test_carla_video_audit_rejects_large_lateral_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            states_path = root / "states.csv"
            states_path.write_text(
                "\n".join(
                    [
                        "step,speed_mps,lateral_error_m,ego_control_mode,safety_brake,scenario_phase",
                        "0,1.0,8.0,e2e_waypoint_control,0.0,ambient_traffic",
                    ]
                ),
                encoding="utf-8",
            )
            report_path = root / "report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "schema": "carla_vision_closed_loop_batch_v1",
                        "scenarios": [
                            {
                                "name": "demo",
                                "metrics": {
                                    "collision_count": 0,
                                    "route_completion": 0.5,
                                    "mean_lateral_error_m": 8.0,
                                },
                                "control_attribution": {
                                    "direct_model_control_ratio": 1.0,
                                    "traffic_manager_vehicle_count": 2,
                                    "scripted_vehicle_count": 0,
                                },
                                "media": {
                                    "video_path": str(root / "missing.mp4"),
                                    "states_csv": str(states_path),
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            audit = audit_carla_vision_rollouts(
                report_path=report_path,
                output_path=root / "audit.json",
                min_frames=1,
                min_resolution_width=1,
                min_resolution_height=1,
            )

            self.assertIn("demo:mean_lateral_error_above_threshold", audit["failures"])

    def test_carla_video_audit_rejects_ego_leaving_driving_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            states_path = root / "states.csv"
            states_path.write_text(
                "\n".join(
                    [
                        "step,speed_mps,lateral_error_m,ego_control_mode,safety_brake,scenario_phase,ego_on_driving_lane",
                        "0,1.0,0.1,e2e_waypoint_control,0.0,route_following,1.0",
                        "1,1.0,0.1,e2e_waypoint_control,0.0,route_following,0.0",
                    ]
                ),
                encoding="utf-8",
            )
            report_path = root / "report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "schema": "carla_vision_closed_loop_batch_v1",
                        "scenarios": [
                            {
                                "name": "demo",
                                "metrics": {
                                    "collision_count": 0,
                                    "route_completion": 0.5,
                                    "mean_lateral_error_m": 0.1,
                                },
                                "control_attribution": {
                                    "direct_model_control_ratio": 1.0,
                                    "traffic_manager_vehicle_count": 2,
                                    "scripted_vehicle_count": 0,
                                },
                                "media": {
                                    "video_path": str(root / "missing.mp4"),
                                    "states_csv": str(states_path),
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            audit = audit_carla_vision_rollouts(
                report_path=report_path,
                output_path=root / "audit.json",
                min_frames=1,
                min_resolution_width=1,
                min_resolution_height=1,
            )

            self.assertIn("demo:ego_left_driving_lane", audit["failures"])


class CarlaClosedLoopUtilityTest(unittest.TestCase):
    def test_discover_carla_maps_filters_component_maps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            map_root = Path(tmp_dir)
            for name in [
                "Town01.umap",
                "Town01_Opt.umap",
                "OpenDriveMap.umap",
                "EmptyMap.umap",
                "T01_Buildings.umap",
                "Town01_BuiltData.umap",
            ]:
                (map_root / name).write_text("", encoding="utf-8")

            maps = _discover_carla_maps(map_root)

            self.assertEqual(maps, ["EmptyMap", "OpenDriveMap", "Town01", "Town01_Opt"])

    def test_launch_command_formatting(self) -> None:
        command = build_carla_launch_command(Path("/opt/carla"), port=2300, fps=10)
        text = format_carla_launch_command(command, cuda_visible_devices="4")

        self.assertIn("CUDA_VISIBLE_DEVICES=4", text)
        self.assertIn("-carla-rpc-port=2300", text)
        self.assertIn("-fps=10", text)

    def test_carla_vision_scenario_params_are_preserved(self) -> None:
        specs = _normalize_scenario_specs(
            [
                {
                    "name": "demo",
                    "scenario_type": "dense_follow_overtake",
                    "spawn_index": 3,
                    "target_speed_mps": 5.0,
                    "parameters": {"ambient_vehicle_count": 24},
                    "ambient_lane_change_percentage": 35.0,
                }
            ]
        )

        self.assertEqual(specs[0].scenario_type, "dense_follow_overtake")
        self.assertEqual(specs[0].spawn_index, 3)
        self.assertEqual(specs[0].parameters["ambient_vehicle_count"], 24)
        self.assertEqual(specs[0].parameters["ambient_lane_change_percentage"], 35.0)

    def test_crosswalk_locations_are_grouped_by_closed_contour(self) -> None:
        groups = _group_crosswalk_locations(
            [
                _Loc(0.0, 0.0),
                _Loc(1.0, 0.0),
                _Loc(1.0, 1.0),
                _Loc(0.0, 0.0),
                _Loc(4.0, 0.0),
                _Loc(5.0, 0.0),
                _Loc(5.0, 1.0),
                _Loc(4.0, 0.0),
            ]
        )

        self.assertEqual([len(group) for group in groups], [4, 4])

    def test_crosswalk_target_distance_uses_full_crosswalk_span(self) -> None:
        distance = _crosswalk_target_distance(
            [_Loc(0.0, 0.0), _Loc(12.0, 0.0), _Loc(12.0, 2.0), _Loc(0.0, 2.0)],
            x=6.0,
            y=1.0,
            direction=(1.0, 0.0),
        )

        self.assertGreaterEqual(distance, 13.0)

    def test_crossing_pedestrian_keeps_moving_after_trigger_latches(self) -> None:
        walker = _WalkerActor(x=10.0, y=-5.0)
        ego = _EgoActor(x=11.0, y=0.0, yaw=0.0)
        runtime = CarlaScenarioRuntime(
            name="right_turn_pedestrian_yield",
            scenario_type="pedestrian_crossing",
            actors=[walker],
            metadata={
                "trigger_progress_m": 10.0,
                "event_end_progress_m": 40.0,
                "target_speed_mps": 1.2,
                "direction_x": 0.0,
                "direction_y": 1.0,
                "start_x": 10.0,
                "start_y": -5.0,
                "crosswalk_target_distance_m": 12.0,
            },
        )
        route = [{"x": 0.0, "y": 0.0}, {"x": 50.0, "y": 0.0}]

        _update_scenario_runtime(
            carla=_CarlaModule(),
            runtime=runtime,
            step_idx=0,
            ego=ego,
            route_points=route,
        )
        ego.transform.location.x = 9.0
        _update_scenario_runtime(
            carla=_CarlaModule(),
            runtime=runtime,
            step_idx=1,
            ego=ego,
            route_points=route,
        )

        self.assertTrue(runtime.metadata["pedestrian_started"])
        self.assertGreater(walker.controls[0].speed, 0.0)
        self.assertGreater(walker.controls[1].speed, 0.0)

    def test_time_based_crossing_pedestrian_starts_without_ego_progress_trigger(self) -> None:
        walker = _WalkerActor(x=10.0, y=-5.0)
        ego = _EgoActor(x=0.0, y=0.0, yaw=0.0)
        runtime = CarlaScenarioRuntime(
            name="right_turn_pedestrian_yield",
            scenario_type="pedestrian_crossing",
            actors=[walker],
            metadata={
                "trigger_progress_m": 100.0,
                "event_end_progress_m": 140.0,
                "target_speed_mps": 1.2,
                "direction_x": 0.0,
                "direction_y": 1.0,
                "start_x": 10.0,
                "start_y": -5.0,
                "crosswalk_target_distance_m": 12.0,
                "pedestrian_trigger_mode": "time",
                "pedestrian_trigger_time_s": 0.0,
                "fps": 10,
            },
        )

        _update_scenario_runtime(
            carla=_CarlaModule(),
            runtime=runtime,
            step_idx=0,
            ego=ego,
            route_points=[{"x": 0.0, "y": 0.0}, {"x": 50.0, "y": 0.0}],
        )

        self.assertTrue(runtime.metadata["pedestrian_started"])
        self.assertGreater(walker.controls[-1].speed, 0.0)

    def test_time_based_crossing_pedestrian_delay_is_relative_to_initial_trigger_tick(self) -> None:
        walker = _WalkerActor(x=10.0, y=-5.0)
        ego = _EgoActor(x=0.0, y=0.0, yaw=0.0)
        runtime = CarlaScenarioRuntime(
            name="right_turn_pedestrian_yield",
            scenario_type="pedestrian_crossing",
            actors=[walker],
            metadata={
                "active_event": "route_following",
                "pedestrian_trigger_mode": "time",
                "pedestrian_trigger_time_s": 0.0,
                "fps": 10,
                "actor_specs": [
                    {
                        "actor_index": 0,
                        "role": "crossing_pedestrian",
                        "kind": "walker",
                        "target_speed_mps": 1.2,
                        "direction_x": 0.0,
                        "direction_y": 1.0,
                        "start_x": 10.0,
                        "start_y": -5.0,
                        "crosswalk_target_distance_m": 12.0,
                        "start_delay_s": 0.3,
                    }
                ],
            },
        )
        route = [{"x": 0.0, "y": 0.0}, {"x": 50.0, "y": 0.0}]

        for step in range(3):
            _update_scenario_runtime(
                carla=_CarlaModule(),
                runtime=runtime,
                step_idx=step,
                ego=ego,
                route_points=route,
            )
        self.assertEqual(walker.controls[-1].speed, 0.0)

        _update_scenario_runtime(
            carla=_CarlaModule(),
            runtime=runtime,
            step_idx=3,
            ego=ego,
            route_points=route,
        )

        self.assertGreater(walker.controls[-1].speed, 0.0)

    def test_time_based_crossing_pedestrian_second_wave_uses_delayed_start(self) -> None:
        first_wave = _WalkerActor(x=10.0, y=-5.0)
        second_wave = _WalkerActor(x=10.0, y=-5.5)
        ego = _EgoActor(x=0.0, y=0.0, yaw=0.0)
        runtime = CarlaScenarioRuntime(
            name="right_turn_pedestrian_yield",
            scenario_type="pedestrian_crossing",
            actors=[first_wave, second_wave],
            metadata={
                "active_event": "route_following",
                "pedestrian_trigger_mode": "time",
                "pedestrian_trigger_time_s": 0.0,
                "fps": 10,
                "actor_specs": [
                    {
                        "actor_index": 0,
                        "role": "crossing_pedestrian",
                        "kind": "walker",
                        "target_speed_mps": 1.2,
                        "direction_x": 0.0,
                        "direction_y": 1.0,
                        "start_x": 10.0,
                        "start_y": -5.0,
                        "crosswalk_target_distance_m": 12.0,
                        "start_delay_s": 0.0,
                    },
                    {
                        "actor_index": 1,
                        "role": "crossing_pedestrian",
                        "kind": "walker",
                        "target_speed_mps": 1.2,
                        "direction_x": 0.0,
                        "direction_y": 1.0,
                        "start_x": 10.0,
                        "start_y": -5.5,
                        "crosswalk_target_distance_m": 12.0,
                        "start_delay_s": 3.0,
                    },
                ],
            },
        )
        route = [{"x": 0.0, "y": 0.0}, {"x": 50.0, "y": 0.0}]

        _update_scenario_runtime(
            carla=_CarlaModule(),
            runtime=runtime,
            step_idx=0,
            ego=ego,
            route_points=route,
        )
        self.assertGreater(first_wave.controls[-1].speed, 0.0)
        self.assertEqual(second_wave.controls[-1].speed, 0.0)

        for step in range(1, 30):
            _update_scenario_runtime(
                carla=_CarlaModule(),
                runtime=runtime,
                step_idx=step,
                ego=ego,
                route_points=route,
            )
        self.assertEqual(second_wave.controls[-1].speed, 0.0)

        _update_scenario_runtime(
            carla=_CarlaModule(),
            runtime=runtime,
            step_idx=30,
            ego=ego,
            route_points=route,
        )
        self.assertGreater(second_wave.controls[-1].speed, 0.0)

    def test_carla_route_features_use_bench2drive_coordinate_convention(self) -> None:
        route_features, command = _build_carla_route_features(
            transform=_Transform(x=0.0, y=0.0, yaw=0.0),
            speed_mps=4.0,
            route_points=[
                {"x": 0.0, "y": 0.0, "command": 4.0},
                {"x": 5.0, "y": 0.0, "command": 2.0},
                {"x": 50.0, "y": 10.0, "command": 2.0},
            ],
        )

        self.assertEqual(command, 2.0)
        self.assertGreater(route_features[0], 0.0)
        self.assertLess(route_features[1], 0.0)
        self.assertAlmostEqual(route_features[4], 0.2)

    def test_carla_route_features_delays_intersection_turn_command_until_nearby(self) -> None:
        route_points = [
            {"x": 0.0, "y": 0.0, "command": 4.0},
            {"x": 3.0, "y": 0.0, "command": 6.0},
            {"x": 8.0, "y": 0.0, "command": 6.0},
            {"x": 16.0, "y": 0.0, "command": 2.0},
            {"x": 30.0, "y": 6.0, "command": 2.0},
        ]

        _, far_command = _build_carla_route_features(
            transform=_Transform(x=0.0, y=0.0, yaw=0.0),
            speed_mps=4.0,
            route_points=route_points,
        )
        _, near_command = _build_carla_route_features(
            transform=_Transform(x=8.0, y=0.0, yaw=0.0),
            speed_mps=4.0,
            route_points=route_points,
        )

        self.assertEqual(far_command, 4.0)
        self.assertEqual(near_command, 2.0)

    def test_active_crossing_pedestrian_triggers_safety_brake(self) -> None:
        ego = _EgoActor(x=10.0, y=0.0, yaw=0.0)
        walker = _WalkerActor(x=15.0, y=1.2)
        runtime = CarlaScenarioRuntime(
            name="right_turn_pedestrian_yield",
            scenario_type="pedestrian_crossing",
            actors=[walker],
            metadata={
                "active_event": "pedestrian_crossing",
                "pedestrian_completed": False,
                "actor_specs": [{"actor_index": 0, "role": "crossing_pedestrian", "kind": "walker"}],
            },
        )

        control, details = _apply_scenario_safety_override(
            carla=_CarlaModule(),
            ego=ego,
            route_points=[{"x": 0.0, "y": 0.0}, {"x": 60.0, "y": 0.0}],
            runtime=runtime,
            control={"steer": 0.0, "throttle": 0.4, "brake": 0.0},
        )

        self.assertEqual(control["throttle"], 0.0)
        self.assertGreater(control["brake"], 0.0)
        self.assertGreater(details["safety_brake"], 0.0)

    def test_active_crossing_pedestrian_far_ahead_does_not_trigger_safety_brake(self) -> None:
        ego = _EgoActor(x=10.0, y=0.0, yaw=0.0)
        walker = _WalkerActor(x=30.0, y=1.2)
        runtime = CarlaScenarioRuntime(
            name="right_turn_pedestrian_yield",
            scenario_type="pedestrian_crossing",
            actors=[walker],
            metadata={
                "active_event": "pedestrian_crossing",
                "pedestrian_completed": False,
                "pedestrian_started": True,
                "actor_specs": [
                    {
                        "actor_index": 0,
                        "role": "crossing_pedestrian",
                        "kind": "walker",
                        "start_x": 30.0,
                        "start_y": -5.0,
                        "direction_x": 0.0,
                        "direction_y": 1.0,
                        "crosswalk_target_distance_m": 12.0,
                    }
                ],
            },
        )

        control, details = _apply_scenario_safety_override(
            carla=_CarlaModule(),
            ego=ego,
            route_points=[{"x": 0.0, "y": 0.0}, {"x": 60.0, "y": 0.0}],
            runtime=runtime,
            control={"steer": 0.0, "throttle": 0.4, "brake": 0.0},
        )

        self.assertEqual(control["throttle"], 0.4)
        self.assertAlmostEqual(control["brake"], 0.0)
        self.assertAlmostEqual(details["safety_brake"], 0.0)

    def test_active_crossing_pedestrian_remains_protected_after_route_projection_passes(self) -> None:
        ego = _EgoActor(x=20.0, y=0.0, yaw=0.0)
        walker = _WalkerActor(x=14.0, y=1.5)
        runtime = CarlaScenarioRuntime(
            name="right_turn_pedestrian_yield",
            scenario_type="pedestrian_crossing",
            actors=[walker],
            metadata={
                "active_event": "pedestrian_crossing",
                "pedestrian_completed": False,
                "actor_specs": [
                    {
                        "actor_index": 0,
                        "role": "crossing_pedestrian",
                        "kind": "walker",
                        "start_x": 14.0,
                        "start_y": -3.0,
                        "direction_x": 0.0,
                        "direction_y": 1.0,
                        "crosswalk_target_distance_m": 12.0,
                    }
                ],
            },
        )

        control, details = _apply_scenario_safety_override(
            carla=_CarlaModule(),
            ego=ego,
            route_points=[{"x": 0.0, "y": 0.0}, {"x": 60.0, "y": 0.0}],
            runtime=runtime,
            control={"steer": 0.0, "throttle": 0.4, "brake": 0.0},
        )

        self.assertEqual(control["throttle"], 0.0)
        self.assertGreater(control["brake"], 0.0)
        self.assertGreater(details["safety_brake"], 0.0)

    def test_pedestrian_safety_brake_releases_after_walker_clears_ego_path(self) -> None:
        ego = _EgoActor(x=10.0, y=0.0, yaw=0.0)
        walker = _WalkerActor(x=20.0, y=4.2)
        runtime = CarlaScenarioRuntime(
            name="right_turn_pedestrian_yield",
            scenario_type="pedestrian_crossing",
            actors=[walker],
            metadata={
                "active_event": "pedestrian_crossing",
                "pedestrian_completed": False,
                "pedestrian_started": True,
                "actor_specs": [
                    {
                        "actor_index": 0,
                        "role": "crossing_pedestrian",
                        "kind": "walker",
                        "start_x": 20.0,
                        "start_y": -5.0,
                        "direction_x": 0.0,
                        "direction_y": 1.0,
                        "crosswalk_target_distance_m": 12.0,
                    }
                ],
            },
        )

        control, details = _apply_scenario_safety_override(
            carla=_CarlaModule(),
            ego=ego,
            route_points=[{"x": 0.0, "y": 0.0}, {"x": 60.0, "y": 0.0}],
            runtime=runtime,
            control={"steer": 0.0, "throttle": 0.4, "brake": 0.0},
        )

        self.assertEqual(control["throttle"], 0.4)
        self.assertAlmostEqual(control["brake"], 0.0)
        self.assertAlmostEqual(details["safety_brake"], 0.0)

    def test_emergency_only_ambient_safety_does_not_brake_for_normal_following_gap(self) -> None:
        ego = _EgoActor(x=10.0, y=0.0, yaw=0.0)
        ego.velocity = _Loc(4.0, 0.0, 0.0)
        vehicle = _WalkerActor(x=22.0, y=0.0)
        runtime = CarlaScenarioRuntime(
            name="dense_traffic",
            scenario_type="pedestrian_crossing",
            actors=[vehicle],
            metadata={
                "ambient_safety_mode": "emergency_only",
                "actor_specs": [
                    {
                        "actor_index": 0,
                        "role": "ambient_autopilot_vehicle",
                        "kind": "autopilot_vehicle",
                    }
                ],
            },
        )

        control, details = _apply_scenario_safety_override(
            carla=_CarlaModule(),
            ego=ego,
            route_points=[{"x": 0.0, "y": 0.0}, {"x": 60.0, "y": 0.0}],
            runtime=runtime,
            control={"steer": 0.0, "throttle": 0.4, "brake": 0.0},
        )

        self.assertAlmostEqual(control["throttle"], 0.4)
        self.assertAlmostEqual(control["brake"], 0.0)
        self.assertAlmostEqual(details["safety_brake"], 0.0)

    def test_emergency_only_ambient_safety_brakes_for_imminent_gap(self) -> None:
        ego = _EgoActor(x=10.0, y=0.0, yaw=0.0)
        vehicle = _WalkerActor(x=12.4, y=0.0)
        runtime = CarlaScenarioRuntime(
            name="dense_traffic",
            scenario_type="pedestrian_crossing",
            actors=[vehicle],
            metadata={
                "ambient_safety_mode": "emergency_only",
                "actor_specs": [
                    {
                        "actor_index": 0,
                        "role": "ambient_autopilot_vehicle",
                        "kind": "autopilot_vehicle",
                    }
                ],
            },
        )

        control, details = _apply_scenario_safety_override(
            carla=_CarlaModule(),
            ego=ego,
            route_points=[{"x": 0.0, "y": 0.0}, {"x": 60.0, "y": 0.0}],
            runtime=runtime,
            control={"steer": 0.0, "throttle": 0.4, "brake": 0.0},
        )

        self.assertEqual(control["throttle"], 0.0)
        self.assertGreater(control["brake"], 0.0)
        self.assertGreater(details["safety_brake"], 0.0)

    def test_traffic_light_conditioning_sets_ego_light_green_and_records_metadata(self) -> None:
        ego_light = _TrafficLight(10)
        conflict_light = _TrafficLight(11)
        ego_light.group = [ego_light, conflict_light]
        conflict_light.group = [ego_light, conflict_light]
        ego = _EgoActorWithTrafficLight(x=0.0, y=0.0, yaw=0.0, traffic_light=ego_light)
        runtime = CarlaScenarioRuntime(name="demo", scenario_type="pedestrian_crossing", actors=[], metadata={})

        details = _condition_ego_route_traffic_lights(
            carla=_CarlaTrafficModule(),
            world=_TrafficWorld([]),
            ego=ego,
            route_points=[],
            runtime=runtime,
            enabled=True,
        )

        self.assertEqual(details["selected_light_id"], 10)
        self.assertEqual(details["selected_light_ids"], [10])
        self.assertEqual(details["group_size"], 2)
        self.assertEqual(ego_light.state, "Green")
        self.assertEqual(conflict_light.state, "Red")
        self.assertTrue(ego_light.frozen)
        self.assertTrue(runtime.metadata["traffic_light_conditioning"]["enabled"])

        ego.traffic_light = None
        details = _condition_ego_route_traffic_lights(
            carla=_CarlaTrafficModule(),
            world=_TrafficWorld([]),
            ego=ego,
            route_points=[],
            runtime=runtime,
            enabled=True,
        )
        self.assertEqual(details["selected_light_id"], -1)
        self.assertEqual(details["selected_light_ids"], [10])

    def test_carla_vision_ego_control_uses_direct_model_output(self) -> None:
        control, details = _carla_control_from_prediction(
            state={"x": 0.0, "y": 0.0, "yaw": 0.0, "speed_mps": 3.0},
            transform=_Transform(x=0.0, y=0.0, yaw=0.0),
            speed_mps=3.0,
            pred_waypoints=[[0.0, -2.0], [0.0, -4.0]],
            pred_control=[0.25, 0.40, 0.10],
            brake_probability=0.10,
            config=CarlaVisionClosedLoopConfig(),
            controller_config=PurePursuitConfig(lookahead_m=5.0),
        )

        self.assertEqual(details["ego_control_mode"], "e2e_waypoint_control")
        self.assertGreater(control["throttle"], 0.0)
        self.assertAlmostEqual(control["brake"], 0.0)

    def test_carla_vision_safety_brake_can_override_direct_control(self) -> None:
        control, details = _carla_control_from_prediction(
            state={"x": 0.0, "y": 0.0, "yaw": 0.0, "speed_mps": 3.0},
            transform=_Transform(x=0.0, y=0.0, yaw=0.0),
            speed_mps=3.0,
            pred_waypoints=[[0.0, -2.0], [0.0, -4.0]],
            pred_control=[0.25, 0.40, 0.10],
            brake_probability=0.95,
            config=CarlaVisionClosedLoopConfig(brake_probability_threshold=0.80),
            controller_config=PurePursuitConfig(lookahead_m=5.0),
        )

        self.assertEqual(details["ego_control_mode"], "e2e_waypoint_control")
        self.assertAlmostEqual(control["throttle"], 0.0)
        self.assertGreater(control["brake"], 0.0)

    def test_carla_vision_low_speed_launch_assist_overcomes_stiction(self) -> None:
        control, details = _carla_control_from_prediction(
            state={"x": 0.0, "y": 0.0, "yaw": 0.0, "speed_mps": 0.0},
            transform=_Transform(x=0.0, y=0.0, yaw=0.0),
            speed_mps=0.0,
            pred_waypoints=[[0.0, -2.0], [0.0, -4.0]],
            pred_control=[0.0, 0.05, 0.0],
            brake_probability=0.05,
            config=CarlaVisionClosedLoopConfig(target_speed_mps=5.0),
            controller_config=PurePursuitConfig(lookahead_m=5.0),
        )

        self.assertEqual(details["ego_control_mode"], "e2e_waypoint_control")
        self.assertGreaterEqual(control["throttle"], 0.36)
        self.assertAlmostEqual(control["brake"], 0.0)

    def test_carla_vision_config_exposes_pure_e2e_ablation_switches(self) -> None:
        config = CarlaVisionClosedLoopConfig(
            enable_scenario_safety_override=False,
            enable_lane_departure_guard=False,
        )

        self.assertFalse(config.enable_scenario_safety_override)
        self.assertFalse(config.enable_lane_departure_guard)

    def test_prediction_path_stats_use_bench2drive_waypoint_convention(self) -> None:
        stats = _prediction_path_stats([[0.0, -2.0], [1.0, -4.0], [2.0, -6.0]])

        self.assertEqual(stats["count"], 3.0)
        self.assertGreater(stats["path_length_m"], 6.0)
        self.assertAlmostEqual(stats["final_forward_m"], 6.0)
        self.assertAlmostEqual(stats["final_right_m"], 2.0)
        self.assertGreater(stats["mean_abs_right_m"], 0.0)

    def test_straight_model_steer_stabilizer_deadbands_small_bias(self) -> None:
        stabilized = _stabilize_straight_model_steer(
            steer=0.018,
            pred_waypoints=[[0.0, -2.0], [0.01, -4.0], [0.0, -6.0]],
        )
        preserved = _stabilize_straight_model_steer(
            steer=0.018,
            pred_waypoints=[[0.0, -2.0], [0.8, -4.0], [1.4, -6.0]],
        )

        self.assertEqual(stabilized, 0.0)
        self.assertAlmostEqual(preserved, 0.018)

    def test_carla_vision_control_attribution_separates_model_and_autopilot(self) -> None:
        attribution = _summarize_carla_vision_control_attribution(
            [
                {
                    "ego_control_mode": "e2e_waypoint_control",
                    "safety_brake": 0.0,
                    "behavior_override": "",
                    "navigation_route_offset_m": 0.0,
                }
            ],
            CarlaScenarioRuntime(
                name="test",
                scenario_type="free_drive",
                actors=[],
                metadata={"actor_specs": [{"role": "ambient_autopilot_vehicle", "kind": "autopilot_vehicle"}]},
            ),
        )

        self.assertFalse(attribution["ego_uses_carla_autopilot"])
        self.assertFalse(attribution["ego_uses_map_route_tracking"])
        self.assertTrue(attribution["ego_uses_model_waypoints"])
        self.assertTrue(attribution["ego_without_safety_override"])
        self.assertEqual(attribution["traffic_manager_vehicle_count"], 1)

    def test_pedestrian_yield_rollout_completion_requires_clearance_and_resume(self) -> None:
        states = []
        for idx in range(28):
            states.append(
                {
                    "scenario_type": "pedestrian_crossing",
                    "scenario_phase": "pedestrian_crossing",
                    "walker_crossing_completion": 1.0 if idx >= 8 else idx / 8.0,
                    "speed_mps": 0.0 if idx < 12 else 1.1,
                    "route_progress_m": 4.0 + max(0, idx - 12) * 1.25,
                    "walker_route_progress_m": 6.0,
                    "command": 2.0 if idx < 24 else 4.0,
                }
            )

        self.assertFalse(_pedestrian_yield_rollout_complete(states, fps=10))

        states[-1]["scenario_phase"] = "pedestrian_cleared"
        self.assertTrue(_pedestrian_yield_rollout_complete(states, fps=10))

    def test_experiment_config_supports_carla_vision_closed_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_dir = root / "carla"
            config_path = root / "carla.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "experiment:",
                        "  id: carla_vision_closed_loop_test",
                        "  type: carla_vision_closed_loop",
                        f"  output: {output_dir}",
                        f"  result_path: {output_dir / 'result.json'}",
                        "carla_vision_closed_loop:",
                        f"  output: {output_dir}",
                        "  carla_root: external/carla/latest",
                        "  checkpoint: outputs/bench2drive_vision_e2e_final/vision_e2e_planner_best.pt",
                        "  horizon_s: 1.0",
                        "  rpc_timeout_s: 12.0",
                        "  render_gif: false",
                    ]
                ),
                encoding="utf-8",
            )

            with patch(
                "nusc_scene_agent.experiment_config.run_carla_vision_closed_loop",
                return_value={
                    "schema": "carla_vision_closed_loop_v1",
                    "metrics": {"frame_count": 10, "collision_count": 0},
                    "media": {"video_path": str(output_dir / "demo.mp4")},
                    "route_length_m": 42.0,
                },
            ) as mocked:
                result = run_experiment_config(config_path)

            self.assertEqual(result["experiment_type"], "carla_vision_closed_loop")
            self.assertEqual(result["result"]["schema"], "carla_vision_closed_loop_v1")
            self.assertEqual(result["result"]["metrics"]["collision_count"], 0)
            mocked.assert_called_once()
            self.assertEqual(mocked.call_args.kwargs["rpc_timeout_s"], 12.0)
            self.assertTrue((output_dir / "result.json").exists())

    def test_experiment_config_preserves_carla_vision_batch_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_dir = root / "carla_batch"
            config_path = root / "carla_batch.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "experiment:",
                        "  id: carla_vision_batch_test",
                        "  type: carla_vision_closed_loop",
                        f"  output: {output_dir}",
                        f"  result_path: {output_dir / 'result.json'}",
                        "carla_vision_closed_loop:",
                        f"  output: {output_dir}",
                        "  scenarios:",
                        "    - name: one",
                        "      scenario_type: free_drive",
                    ]
                ),
                encoding="utf-8",
            )

            with patch(
                "nusc_scene_agent.experiment_config.run_carla_vision_closed_loop",
                return_value={
                    "schema": "carla_vision_closed_loop_batch_v1",
                    "scenario_count": 1,
                    "aggregate": {
                        "mean_route_completion": 0.25,
                        "mean_driving_score": 0.20,
                        "total_collisions": 1,
                    },
                    "scenarios": [{"name": "one"}],
                },
            ):
                result = run_experiment_config(config_path)

            self.assertEqual(result["result"]["schema"], "carla_vision_closed_loop_batch_v1")
            self.assertEqual(result["result"]["scenario_count"], 1)
            self.assertEqual(result["result"]["metrics"]["mean_route_completion"], 0.25)
            self.assertEqual(result["result"]["metrics"]["total_collisions"], 1)

    def test_carla_semantic_demo_mining_promotes_passing_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_dir = root / "semantic_demo"
            trials_dir = root / "trials"
            stale_attempt = trials_dir / "dense_traffic_follow" / "attempt_01_dense_traffic_follow"
            stale_attempt.mkdir(parents=True, exist_ok=True)
            (stale_attempt / "stale.txt").write_text("old", encoding="utf-8")

            def fake_run(**kwargs):
                attempt_dir = Path(kwargs["output_dir"])
                self.assertFalse((attempt_dir / "stale.txt").exists())
                attempt_dir.mkdir(parents=True, exist_ok=True)
                report = {
                    "schema": "carla_vision_closed_loop_v1",
                    "output_dir": str(attempt_dir),
                    "town": "Town10HD_Opt",
                    "scenario": {"name": "dense_traffic_follow", "type": "dense_follow_overtake", "actor_count": 2},
                    "route_length_m": 80.0,
                    "metrics": {
                        "frame_count": 120,
                        "route_completion": 0.4,
                        "driving_score": 0.4,
                        "collision_count": 0,
                    },
                    "control_attribution": {
                        "direct_model_control_ratio": 1.0,
                        "traffic_manager_vehicle_count": 2,
                        "scripted_vehicle_count": 0,
                    },
                    "media": {"video_path": str(attempt_dir / "carla_vision_closed_loop.mp4")},
                }
                (attempt_dir / "carla_vision_closed_loop_report.json").write_text(
                    json.dumps(report),
                    encoding="utf-8",
                )
                return report

            pass_audit = {
                "status": "pass",
                "scenario_count": 1,
                "failure_count": 0,
                "warning_count": 0,
                "failures": [],
                "warnings": [],
                "summary": {"passed_scenarios": 1, "failed_scenarios": 0},
            }

            with patch("nusc_scene_agent.carla_semantic_demo_mining.run_carla_vision_closed_loop", side_effect=fake_run), patch(
                "nusc_scene_agent.carla_semantic_demo_mining.audit_carla_vision_rollouts",
                return_value=pass_audit,
            ):
                report = mine_carla_semantic_demos(
                    output_dir=output_dir,
                    trials_output_dir=trials_dir,
                    max_attempts_per_target=1,
                    targets=[
                        {
                            "target_id": "dense_traffic_follow",
                            "attempts": [
                                {
                                    "name": "dense_traffic_follow",
                                    "scenario_type": "dense_follow_overtake",
                                    "spawn_index": 1,
                                    "destination_index": 2,
                                }
                            ],
                        }
                    ],
                )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["passed_target_count"], 1)
            self.assertTrue((output_dir / "dense_traffic_follow/carla_vision_closed_loop_report.json").exists())
            self.assertTrue((output_dir / "carla_semantic_demo_report.json").exists())
            self.assertTrue((output_dir / "carla_semantic_demo_mining_report.json").exists())

    def test_experiment_config_supports_carla_semantic_demo_mining(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_dir = root / "semantic_demo"
            config_path = root / "carla_semantic.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "experiment:",
                        "  id: carla_semantic_demo_test",
                        "  type: carla_semantic_demo_mining",
                        f"  output: {output_dir}",
                        f"  result_path: {output_dir / 'result.json'}",
                        "carla_semantic_demo:",
                        f"  output: {output_dir}",
                        f"  trials_output: {root / 'trials'}",
                        "  port_start: 2100",
                        "  max_attempts_per_target: 1",
                    ]
                ),
                encoding="utf-8",
            )

            with patch(
                "nusc_scene_agent.experiment_config.mine_carla_semantic_demos",
                return_value={
                    "schema": "carla_semantic_demo_mining_v1",
                    "status": "pass",
                    "output_dir": str(output_dir),
                    "passed_target_count": 1,
                    "target_count": 1,
                },
            ) as mocked:
                result = run_experiment_config(config_path)

            self.assertEqual(result["experiment_type"], "carla_semantic_demo_mining")
            self.assertEqual(result["result"]["schema"], "carla_semantic_demo_mining_v1")
            mocked.assert_called_once()
            self.assertEqual(mocked.call_args.kwargs["port_start"], 2100)
            self.assertTrue((output_dir / "result.json").exists())

    def test_experiment_config_preserves_zero_carla_semantic_demo_traffic_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_dir = root / "semantic_demo"
            config_path = root / "carla_semantic.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "experiment:",
                        "  id: carla_semantic_demo_test",
                        "  type: carla_semantic_demo_mining",
                        f"  output: {output_dir}",
                        "carla_semantic_demo:",
                        f"  output: {output_dir}",
                        "  min_traffic_manager_vehicles: 0",
                    ]
                ),
                encoding="utf-8",
            )

            with patch(
                "nusc_scene_agent.experiment_config.mine_carla_semantic_demos",
                return_value={
                    "schema": "carla_semantic_demo_mining_v1",
                    "status": "pass",
                    "output_dir": str(output_dir),
                    "passed_target_count": 1,
                    "target_count": 1,
                },
            ) as mocked:
                run_experiment_config(config_path)

            self.assertEqual(mocked.call_args.kwargs["min_traffic_manager_vehicles"], 0)


def _build_bench2drive_archive(path: Path) -> None:
    clip = path.name.replace(".tar.gz", "")
    image_bytes = _tiny_jpeg_bytes()
    with tarfile.open(path, "w:gz") as tar:
        for frame_id in range(3):
            anno = {
                "x": float(frame_id),
                "y": 0.0,
                "theta": 0.0,
                "speed": 1.0,
                "steer": 0.1,
                "throttle": 0.2,
                "brake": 0.0,
                "should_brake": frame_id == 2,
                "x_target": 10.0,
                "y_target": 0.0,
                "x_command_near": 5.0,
                "y_command_near": 0.0,
                "next_command": 4,
                "command_near": 4,
                "command_far": 4,
                "bounding_boxes": [],
                "weather": {},
            }
            _add_bytes(
                tar,
                f"./{clip}/anno/{frame_id:05d}.json.gz",
                gzip.compress(json.dumps(anno).encode("utf-8")),
            )
            for camera in DEFAULT_BENCH2DRIVE_CAMERAS:
                _add_bytes(tar, f"./{clip}/camera/{camera}/{frame_id:05d}.jpg", image_bytes)


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _tiny_jpeg_bytes() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color=(128, 64, 32)).save(buffer, format="JPEG")
    return buffer.getvalue()


class _Loc:
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z

    def distance(self, other: object) -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2) ** 0.5


class _Rot:
    def __init__(self, yaw: float) -> None:
        self.yaw = yaw


class _Transform:
    def __init__(self, x: float, y: float, yaw: float) -> None:
        self.location = _Loc(x, y)
        self.rotation = _Rot(yaw)


class _EgoActor:
    def __init__(self, x: float, y: float, yaw: float) -> None:
        self.transform = _Transform(x, y, yaw)
        self.velocity = _Loc(0.0, 0.0, 0.0)

    def get_transform(self) -> _Transform:
        return self.transform

    def get_location(self) -> _Loc:
        return self.transform.location

    def get_velocity(self) -> _Loc:
        return self.velocity


class _EgoActorWithTrafficLight(_EgoActor):
    def __init__(self, x: float, y: float, yaw: float, traffic_light: object) -> None:
        super().__init__(x=x, y=y, yaw=yaw)
        self.traffic_light = traffic_light

    def get_traffic_light(self) -> object:
        return self.traffic_light


class _WalkerActor:
    def __init__(self, x: float, y: float) -> None:
        self.location = _Loc(x, y)
        self.velocity = _Loc(0.0, 0.0, 0.0)
        self.controls = []

    def get_location(self) -> _Loc:
        return self.location

    def get_velocity(self) -> _Loc:
        return self.velocity

    def apply_control(self, control: object) -> None:
        self.controls.append(control)


class _CarlaModule:
    class Vector3D:
        def __init__(self, x: float, y: float, z: float) -> None:
            self.x = x
            self.y = y
            self.z = z

    class WalkerControl:
        def __init__(self, direction: object, speed: float) -> None:
            self.direction = direction
            self.speed = speed


class _CarlaTrafficModule(_CarlaModule):
    class TrafficLightState:
        Green = "Green"
        Red = "Red"


class _TrafficLight:
    def __init__(self, actor_id: int, x: float = 0.0, y: float = 0.0) -> None:
        self.id = actor_id
        self.location = _Loc(x, y)
        self.group = [self]
        self.state = ""
        self.frozen = False

    def get_group_traffic_lights(self) -> list:
        return list(self.group)

    def set_state(self, state: object) -> None:
        self.state = state

    def set_green_time(self, value: float) -> None:
        self.green_time = value

    def set_red_time(self, value: float) -> None:
        self.red_time = value

    def set_yellow_time(self, value: float) -> None:
        self.yellow_time = value

    def freeze(self, enabled: bool) -> None:
        self.frozen = bool(enabled)

    def get_location(self) -> _Loc:
        return self.location


class _ActorList(list):
    def filter(self, pattern: str) -> list:
        return list(self)


class _TrafficWorld:
    def __init__(self, actors: list) -> None:
        self.actors = _ActorList(actors)

    def get_actors(self) -> _ActorList:
        return self.actors


if __name__ == "__main__":
    unittest.main()
