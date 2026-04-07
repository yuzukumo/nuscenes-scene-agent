import json
import tempfile
import unittest
from pathlib import Path

from nusc_scene_agent.benchmark_schema import load_benchmark_config
from nusc_scene_agent.counterfactual_benchmark import generate_counterfactual_benchmark_from_case_library


class CounterfactualBenchmarkGenerationTest(unittest.TestCase):
    def test_generate_counterfactual_benchmark_from_case_library(self) -> None:
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
                            "category_name": "human.pedestrian.adult",
                            "category_group": "pedestrian",
                            "validation_score": 94.0,
                            "passed": True,
                            "min_distance_m": 3.5,
                            "min_ttc_s": 1.1,
                            "event_start_sample_idx": 11,
                            "event_end_sample_idx": 15,
                            "event_peak_sample_idx": 13,
                            "source_query_ids": ["pedestrian_crossing_front"],
                            "source_queries": ["pedestrian crossing in front of ego lane"],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            output_path = root / "generated.yaml"

            metadata = generate_counterfactual_benchmark_from_case_library(case_library_path, output_path, max_cases=1)
            self.assertEqual(metadata["anchor_case_count"], 1)
            self.assertEqual(metadata["query_count"], 4)

            specs = load_benchmark_config(output_path)
            self.assertEqual(len(specs), 4)
            self.assertTrue(all(spec.reference_case_keys == ["sample-a:instance-a"] for spec in specs))
            self.assertTrue(all(spec.reference_scene_names == ["scene-0001"] for spec in specs))
            self.assertTrue(all(spec.reference_instance_tokens == ["instance-a"] for spec in specs))
            self.assertTrue(all(spec.reference_event_sample_range == [11, 15] for spec in specs))
            self.assertEqual(sum(1 for spec in specs if spec.expect_match is True), 2)
            self.assertEqual(sum(1 for spec in specs if spec.expect_match is False), 2)


if __name__ == "__main__":
    unittest.main()
