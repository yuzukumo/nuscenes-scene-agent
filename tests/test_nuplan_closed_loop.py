import json
import tempfile
import unittest
from pathlib import Path

from nusc_scene_agent.experiment_config import run_experiment_config
from nusc_scene_agent.nuplan_closed_loop import run_nuplan_closed_loop_study
from test_nuplan_replay import _build_nuplan_db


class NuPlanClosedLoopTest(unittest.TestCase):
    def test_run_closed_loop_study_generates_metrics_and_case_studies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            split_dir = root / "split"
            split_dir.mkdir()
            _build_nuplan_db(split_dir / "mini.db")
            output_dir = root / "closed_loop"

            manifest = run_nuplan_closed_loop_study(
                split_dir=split_dir,
                output_dir=output_dir,
                max_dbs=1,
                max_cases=1,
                max_cases_per_db=1,
                history_s=0.5,
                future_s=1.0,
                frame_hz=2.0,
                profiles=["logged_ego_oracle", "history_kinematic", "idm_like_following"],
            )

            self.assertEqual(manifest["schema"], "nuplan_closed_loop_study_v1")
            self.assertEqual(manifest["benchmark"]["metadata"]["case_count"], 1)
            self.assertEqual(manifest["comparison"]["overview"]["profile_count"], 3)
            self.assertTrue((output_dir / "nuplan_closed_loop_benchmark.json").exists())
            self.assertTrue((output_dir / "comparison/closed_loop_leaderboard.csv").exists())
            self.assertTrue((output_dir / "case_studies/nuplan_closed_loop_case_studies.png").exists())
            self.assertTrue((output_dir / "artifact_manifest.json").exists())

            oracle = json.loads(
                (output_dir / "logged_ego_oracle_closed_loop/closed_loop_metrics.json").read_text(encoding="utf-8")
            )
            history = json.loads(
                (output_dir / "history_kinematic_closed_loop/closed_loop_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(oracle["overview"]["full_horizon_count"], 1)
            self.assertAlmostEqual(oracle["overview"]["mean_ego_ade_m"], 0.0)
            self.assertAlmostEqual(history["overview"]["mean_ego_ade_m"], 0.0)
            self.assertGreaterEqual(history["overview"]["mean_closed_loop_score"], 0.9)

    def test_experiment_config_supports_closed_loop_study(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            split_dir = root / "split"
            split_dir.mkdir()
            _build_nuplan_db(split_dir / "mini.db")
            output_dir = root / "outputs"
            config_path = root / "closed_loop.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "experiment:",
                        "  id: closed_loop_test",
                        "  type: nuplan_closed_loop_study",
                        f"  output: {output_dir}",
                        f"  result_path: {output_dir / 'result.json'}",
                        "nuplan_closed_loop:",
                        f"  split_dir: {split_dir}",
                        f"  output: {output_dir}",
                        "  max_dbs: 1",
                        "  max_cases: 1",
                        "  max_cases_per_db: 1",
                        "  history_s: 0.5",
                        "  future_s: 1.0",
                        "  frame_hz: 2.0",
                        "  profiles:",
                        "    - logged_ego_oracle",
                        "    - history_kinematic",
                    ]
                ),
                encoding="utf-8",
            )

            result = run_experiment_config(config_path)

            self.assertEqual(result["experiment_type"], "nuplan_closed_loop_study")
            self.assertEqual(result["result"]["comparison"]["overview"]["profile_count"], 2)
            self.assertTrue((output_dir / "result.json").exists())


if __name__ == "__main__":
    unittest.main()
