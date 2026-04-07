import json
import tempfile
import unittest
from pathlib import Path

from nusc_scene_agent.benchmark_schema import load_benchmark_config
from nusc_scene_agent.scenario_mining_benchmark import generate_scenario_mining_benchmark_from_case_library


class ScenarioMiningBenchmarkGenerationTest(unittest.TestCase):
    def test_generate_scenario_mining_benchmark_from_case_library(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            case_library_path = root / "case_library.json"
            case_library_path.write_text(
                json.dumps(
                    [
                        {
                            "case_key": "sample-a:instance-a",
                            "scene_name": "scene-0001",
                            "instance_token": "instance-a",
                            "category_name": "vehicle.car",
                            "category_group": "vehicle",
                            "validation_score": 96.0,
                            "passed": True,
                            "min_distance_m": 4.0,
                            "min_ttc_s": 1.0,
                            "event_start_sample_idx": 20,
                            "event_end_sample_idx": 24,
                            "event_peak_sample_idx": 22,
                            "source_query_ids": ["oncoming_vehicle"],
                            "source_queries": ["oncoming vehicle close to ego"],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            output_path = root / "scenario.yaml"

            metadata = generate_scenario_mining_benchmark_from_case_library(case_library_path, output_path, max_cases=1)
            self.assertEqual(metadata["anchor_case_count"], 1)
            self.assertEqual(metadata["query_count"], 2)

            specs = load_benchmark_config(output_path)
            self.assertEqual(len(specs), 2)
            self.assertTrue(all(spec.expect_match is True for spec in specs))
            self.assertTrue(all(spec.reference_scene_names == ["scene-0001"] for spec in specs))
            self.assertTrue(all(spec.reference_instance_tokens == ["instance-a"] for spec in specs))


if __name__ == "__main__":
    unittest.main()
