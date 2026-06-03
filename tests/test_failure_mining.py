import json
import tempfile
import unittest
from pathlib import Path

from nusc_scene_agent.failure_mining import run_failure_mining


class FailureMiningTest(unittest.TestCase):
    def test_nuplan_metric_failures_keep_scenario_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            metric_path = root / "constant_velocity_evaluation" / "nuplan_replay_metrics.json"
            metric_path.parent.mkdir(parents=True)
            metric_path.write_text(
                json.dumps(
                    {
                        "scenario_family_breakdown": {},
                        "case_metrics": [
                            {
                                "case_id": "nuplan_case_001",
                                "scenario_family": "high_speed_interaction",
                                "scenario_tag": "near_high_speed_vehicle",
                                "difficulty_label": "hard",
                                "location": "sg-one-north",
                                "failure_tags": ["risk_distance_error"],
                                "risk_fidelity_score": 0.5,
                                "predicted_min_distance_m": 2.0,
                                "logged_min_distance_m": 8.0,
                                "min_distance_error_m": 6.0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = run_failure_mining(
                sources=[metric_path],
                output_dir=root / "failure_mining",
                top_k=4,
                min_count=1,
            )

            payload = json.loads(Path(result["report_json"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["record_count"], 1)
            self.assertEqual(payload["clusters"][0]["source_type"], "nuplan_replay")
            self.assertEqual(payload["clusters"][0]["context"], "high_speed_interaction")
            self.assertEqual(payload["clusters"][0]["actor_or_scenario"], "near_high_speed_vehicle")
            self.assertNotIn("model_metrics | risk_distance_error | unspecified", Path(result["report_md"]).read_text())


if __name__ == "__main__":
    unittest.main()
