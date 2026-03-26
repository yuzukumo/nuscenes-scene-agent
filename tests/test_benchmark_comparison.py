import json
import tempfile
import unittest
from pathlib import Path

from nusc_scene_agent.benchmark_comparison import (
    build_benchmark_comparison,
    default_benchmark_profiles,
    write_benchmark_comparison,
)


class BenchmarkComparisonTest(unittest.TestCase):
    def test_default_profiles_include_three_modes_when_llm_is_available(self) -> None:
        profiles = default_benchmark_profiles(include_llm=True)
        self.assertEqual([profile["name"] for profile in profiles], ["rule_only", "llm_planner", "hybrid_agent"])

    def test_default_profiles_fall_back_to_rule_only_without_llm(self) -> None:
        profiles = default_benchmark_profiles(include_llm=False)
        self.assertEqual([profile["name"] for profile in profiles], ["rule_only"])

    def test_build_benchmark_comparison_merges_profile_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs = []
            for name, label, pass_at_1_count, mean_score, top_taxonomy in [
                ("rule_only", "Rule-Only", 1, 80.0, "behavior_mismatch"),
                ("hybrid_agent", "Hybrid-Agent", 2, 90.0, "multi_query_overlap"),
            ]:
                profile_dir = root / name
                profile_dir.mkdir(parents=True, exist_ok=True)
                metrics = {
                    "overview": {
                        "query_count": 2,
                        "pass_at_1_count": pass_at_1_count,
                        "pass_at_1_rate": pass_at_1_count / 2.0,
                        "pass_at_k_count": pass_at_1_count,
                        "pass_at_k_rate": pass_at_1_count / 2.0,
                        "mean_best_validation_score": mean_score,
                        "unique_case_count": 3,
                    },
                    "query_metrics": [
                        {
                            "id": "q1",
                            "description": "desc 1",
                            "actors": ["vehicle"],
                            "behaviors": ["cut_in"],
                            "pass_at_1": True,
                            "pass_at_k": True,
                            "best_validation_score": mean_score,
                            "selected_count": 1,
                            "passed_count": 1,
                        },
                        {
                            "id": "q2",
                            "description": "desc 2",
                            "actors": ["pedestrian"],
                            "behaviors": ["crossing"],
                            "pass_at_1": bool(pass_at_1_count == 2),
                            "pass_at_k": bool(pass_at_1_count == 2),
                            "best_validation_score": mean_score - 10.0,
                            "selected_count": 1,
                            "passed_count": int(pass_at_1_count == 2),
                        },
                    ],
                }
                taxonomy = {
                    "overview": {"hard_case_count": 2, "failed_count": 1},
                    "label_distribution": [{"name": top_taxonomy, "count": 1}],
                }
                (profile_dir / "benchmark_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
                (profile_dir / "hard_case_taxonomy.json").write_text(json.dumps(taxonomy), encoding="utf-8")
                runs.append(
                    {
                        "name": name,
                        "label": label,
                        "query_mode": "rule" if name == "rule_only" else "hybrid",
                        "rerank_mode": "none" if name == "rule_only" else "llm",
                        "output_dir": str(profile_dir),
                    }
                )

            comparison = build_benchmark_comparison(runs)
            self.assertEqual(comparison["overview"]["profile_count"], 2)
            self.assertEqual(comparison["overview"]["query_count"], 2)
            self.assertEqual(comparison["profiles"][1]["top_taxonomy_label"], "multi_query_overlap")
            self.assertEqual(comparison["query_comparison"][0]["best_profile"], "hybrid_agent")
            self.assertEqual(comparison["deltas_vs_baseline"][0]["profile"], "hybrid_agent")

            write_benchmark_comparison(comparison, root)
            self.assertTrue((root / "benchmark_profile_comparison.json").exists())
            self.assertTrue((root / "benchmark_profile_comparison_summary.md").exists())


if __name__ == "__main__":
    unittest.main()
