import json
import tempfile
import unittest
from pathlib import Path

from nusc_scene_agent.gallery import build_benchmark_gallery, build_comparison_browser


def write_minimal_benchmark_output(root: Path, name: str, score: float, scene_match: bool) -> Path:
    output_dir = root / name
    output_dir.mkdir(parents=True, exist_ok=True)
    query_dir = output_dir / "a_pedestrian_crosses_in_front_of_ego"
    case_dir = query_dir / "rank_01_scene_0001_sample_003"
    case_dir.mkdir(parents=True, exist_ok=True)

    (query_dir / "summary.md").write_text("# Summary\n", encoding="utf-8")
    (query_dir / "summary.html").write_text("<html></html>\n", encoding="utf-8")
    (case_dir / "case.md").write_text("# Case\n", encoding="utf-8")
    (case_dir / "evidence.png").write_text("png", encoding="utf-8")

    case_payload = {
        "query": {
            "original_text": "a pedestrian crosses in front of ego",
        },
        "candidate": {
            "scene_name": "scene-0001" if scene_match else "scene-0999",
            "sample_idx": 3,
            "category_name": "human.pedestrian.adult",
        },
        "validation_score": score,
        "passed": True,
        "evidence": {
            "min_distance_m": 5.5,
            "min_ttc_s": 0.8,
        },
        "notes": ["note"],
    }
    (case_dir / "case.json").write_text(json.dumps(case_payload), encoding="utf-8")

    benchmark_summary = [
        {
            "id": "q1",
            "description": "crossing canonical",
            "query_dir": str(query_dir),
            "candidate_count": 12,
            "selected_count": 3,
            "tags": ["scenario_mining", "crossing"],
            "behaviors": ["crossing"],
            "actors": ["pedestrian"],
        }
    ]
    (output_dir / "benchmark_summary.json").write_text(json.dumps(benchmark_summary), encoding="utf-8")

    benchmark_metrics = {
        "query_metrics": [
            {
                "id": "q1",
                "pass_at_1": True,
                "scene_objective_at_1": scene_match,
                "actor_objective_at_1": scene_match,
                "reference_objective_at_1": scene_match,
                "event_iou": 1.0 if scene_match else 0.2,
                "peak_error": 0 if scene_match else 5,
            }
        ]
    }
    (output_dir / "benchmark_metrics.json").write_text(json.dumps(benchmark_metrics), encoding="utf-8")
    return output_dir


class GalleryTest(unittest.TestCase):
    def test_build_benchmark_gallery_writes_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            benchmark_dir = write_minimal_benchmark_output(root, "benchmark_a", score=91.0, scene_match=True)

            metadata = build_benchmark_gallery(benchmark_dir, title="Test Gallery")

            self.assertEqual(metadata["mode"], "benchmark")
            self.assertTrue((benchmark_dir / "query_gallery.html").exists())
            self.assertTrue((benchmark_dir / "query_gallery.json").exists())

            payload = json.loads((benchmark_dir / "query_gallery.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["query_count"], 1)
            self.assertEqual(payload["cards"][0]["query_id"], "q1")
            self.assertEqual(payload["cards"][0]["scene_name"], "scene-0001")

    def test_build_comparison_browser_writes_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            rule_dir = write_minimal_benchmark_output(root, "rule_output", score=91.0, scene_match=True)
            llm_dir = write_minimal_benchmark_output(root, "llm_output", score=88.0, scene_match=False)

            comparison_dir = root / "comparison"
            comparison_dir.mkdir(parents=True, exist_ok=True)
            comparison_payload = {
                "profiles": [
                    {
                        "name": "rule_only",
                        "label": "Rule-Only",
                        "output_dir": str(rule_dir),
                    },
                    {
                        "name": "llm_planner",
                        "label": "LLM-Planner",
                        "output_dir": str(llm_dir),
                    },
                ],
                "leaderboard": [
                    {
                        "label": "Rule-Only",
                    }
                ],
                "query_comparison": [
                    {
                        "id": "q1",
                        "description": "crossing canonical",
                        "actors": ["pedestrian"],
                        "behaviors": ["crossing"],
                        "best_profile": "rule_only",
                        "signal_divergence": True,
                        "score_span": 3.0,
                        "profiles": {
                            "rule_only": {
                                "best_validation_score": 91.0,
                                "scene_objective_at_1": True,
                                "actor_objective_at_1": True,
                                "reference_objective_at_1": True,
                                "event_iou": 1.0,
                                "peak_error": 0,
                            },
                            "llm_planner": {
                                "best_validation_score": 88.0,
                                "scene_objective_at_1": False,
                                "actor_objective_at_1": False,
                                "reference_objective_at_1": False,
                                "event_iou": 0.2,
                                "peak_error": 5,
                            },
                        },
                    }
                ],
            }
            (comparison_dir / "benchmark_profile_comparison.json").write_text(
                json.dumps(comparison_payload),
                encoding="utf-8",
            )

            metadata = build_comparison_browser(comparison_dir, title="Comparison Browser")

            self.assertEqual(metadata["mode"], "comparison")
            self.assertTrue((comparison_dir / "comparison_browser.html").exists())
            self.assertTrue((comparison_dir / "comparison_browser.json").exists())
            self.assertTrue((rule_dir / "query_gallery.html").exists())
            self.assertTrue((llm_dir / "query_gallery.html").exists())

            payload = json.loads((comparison_dir / "comparison_browser.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["query_count"], 1)
            self.assertEqual(payload["cards"][0]["best_profile"], "rule_only")
            self.assertEqual(len(payload["cards"][0]["profiles"]), 2)


if __name__ == "__main__":
    unittest.main()
