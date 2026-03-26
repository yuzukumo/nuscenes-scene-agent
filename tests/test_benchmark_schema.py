import tempfile
import unittest
from pathlib import Path

from nusc_scene_agent.benchmark_schema import apply_benchmark_spec, load_benchmark_config
from nusc_scene_agent.query_parser import parse_query


class BenchmarkSchemaTest(unittest.TestCase):
    def test_load_and_apply_structured_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "queries.yaml"
            config_path.write_text(
                """
queries:
  - id: crossing_case
    description: pedestrian crossing in front of ego lane
    top_k: 2
    candidate_pool: 8
    tags: [crossing, front]
    query:
      natural_language: pedestrian crossing in front of ego lane
      actors: [pedestrian]
      positions: [front]
      behaviors: [crossing]
      risk_terms: [risky]
      map_constraints:
        prefer_crosswalk: true
      thresholds:
        near_distance_m: 18
        max_ttc_s: 4
""",
                encoding="utf-8",
            )

            specs = load_benchmark_config(config_path)
            self.assertEqual(len(specs), 1)
            spec = specs[0]
            self.assertEqual(spec.id, "crossing_case")
            self.assertEqual(spec.candidate_pool, 8)
            self.assertIn("pedestrian", spec.actors)
            self.assertTrue(spec.apply_query_overrides)

            parsed = parse_query(spec.natural_language)
            merged = apply_benchmark_spec(parsed, spec)
            self.assertIn("pedestrian", merged.category_groups)
            self.assertIn("front", merged.positions)
            self.assertIn("crossing", merged.behaviors)
            self.assertEqual(merged.near_distance_m, 18.0)
            self.assertEqual(merged.max_ttc_s, 4.0)
            self.assertIn("map:prefer_crosswalk", merged.specific_keywords)

    def test_apply_benchmark_spec_can_skip_query_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "queries.yaml"
            config_path.write_text(
                """
queries:
  - id: language_stress_case
    description: someone steps into ego lane directly ahead
    apply_query_overrides: false
    query:
      natural_language: someone steps into ego lane directly ahead
      actors: [pedestrian]
      positions: [front]
      behaviors: [crossing]
      risk_terms: [risky]
""",
                encoding="utf-8",
            )

            spec = load_benchmark_config(config_path)[0]
            self.assertFalse(spec.apply_query_overrides)

            parsed = parse_query(spec.natural_language)
            merged = apply_benchmark_spec(parsed, spec)
            self.assertEqual(merged.category_groups, parsed.category_groups)
            self.assertEqual(merged.positions, parsed.positions)
            self.assertEqual(merged.behaviors, parsed.behaviors)


if __name__ == "__main__":
    unittest.main()
