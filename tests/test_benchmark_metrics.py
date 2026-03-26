import tempfile
import unittest
from pathlib import Path

from nusc_scene_agent.benchmark_metrics import build_benchmark_metrics, write_benchmark_metrics
from nusc_scene_agent.benchmark_schema import BenchmarkQuerySpec
from nusc_scene_agent.case_library import build_case_library
from nusc_scene_agent.models import ParsedQuery, RetrievalCandidate, ValidatedCase


def make_case(
    query_text: str,
    category_group: str,
    category_name: str,
    location: str,
    sample_token: str,
    instance_token: str,
    score: float,
    passed: bool,
) -> ValidatedCase:
    query = ParsedQuery(
        original_text=query_text,
        normalized_text=query_text,
        category_groups=[category_group],
        positions=["front"],
        behaviors=["crossing"] if category_group == "pedestrian" else ["oncoming"],
        near_distance_m=20.0,
        max_ttc_s=5.0,
    )
    candidate = RetrievalCandidate(
        ann_token="ann-" + instance_token,
        sample_token=sample_token,
        scene_token="scene-" + sample_token,
        scene_name="scene-" + sample_token[-1],
        sample_idx=7,
        instance_token=instance_token,
        category_name=category_name,
        category_group=category_group,
        location=location,
        distance=4.0,
        ttc=1.2,
        x_ego=3.0,
        y_ego=0.5,
        speed=0.0,
        rel_vx=-2.0,
        rel_vy=0.0,
        heading_delta=0.0,
        retrieval_score=score,
    )
    behavior_name = query.behaviors[0]
    return ValidatedCase(
        query=query,
        candidate=candidate,
        validation_score=score,
        passed=passed,
        behavior_matches={behavior_name: passed},
        evidence={"min_distance_m": 4.0, "min_ttc_s": 1.2},
        notes=["note"],
        timeline=None,
        context_agents=None,
        ego_window=None,
        map_context={"available": True},
        map_geometries={},
        figure_path="/tmp/{0}.png".format(instance_token),
        report_dir="/tmp/{0}".format(instance_token),
    )


class BenchmarkMetricsTest(unittest.TestCase):
    def test_build_benchmark_metrics_summarizes_queries_and_locations(self) -> None:
        passed_case = make_case(
            query_text="pedestrian crossing",
            category_group="pedestrian",
            category_name="human.pedestrian.adult",
            location="singapore-queenstown",
            sample_token="sample-a",
            instance_token="instance-a",
            score=91.0,
            passed=True,
        )
        failed_case = make_case(
            query_text="oncoming vehicle",
            category_group="vehicle",
            category_name="vehicle.car",
            location="boston-seaport",
            sample_token="sample-b",
            instance_token="instance-b",
            score=73.0,
            passed=False,
        )

        results = [
            {
                "id": "ped_crossing",
                "query": passed_case.query,
                "query_spec": BenchmarkQuerySpec(
                    id="ped_crossing",
                    description="pedestrian crossing",
                    natural_language="pedestrian crossing",
                    tags=["vru"],
                    actors=["pedestrian"],
                    behaviors=["crossing"],
                ),
                "selected_cases": [passed_case],
            },
            {
                "id": "oncoming_vehicle",
                "query": failed_case.query,
                "query_spec": BenchmarkQuerySpec(
                    id="oncoming_vehicle",
                    description="oncoming vehicle",
                    natural_language="oncoming vehicle",
                    tags=["vehicle"],
                    actors=["vehicle"],
                    behaviors=["oncoming"],
                ),
                "selected_cases": [failed_case],
            },
        ]
        case_library_entries = build_case_library(results)

        metrics = build_benchmark_metrics(results, case_library_entries)
        self.assertEqual(metrics["overview"]["query_count"], 2)
        self.assertEqual(metrics["overview"]["pass_at_1_count"], 1)
        self.assertEqual(metrics["overview"]["pass_at_k_count"], 1)
        self.assertEqual(metrics["overview"]["unique_locations"], 2)
        self.assertEqual(len(metrics["behavior_coverage"]), 2)
        self.assertEqual(metrics["location_distribution"][0]["case_count"], 1)

    def test_write_benchmark_metrics_emits_summary_files(self) -> None:
        metrics = {
            "overview": {
                "query_count": 1,
                "selected_case_count": 2,
                "passed_selected_case_count": 1,
                "unique_case_count": 2,
                "unique_passed_case_count": 1,
                "pass_at_1_count": 1,
                "pass_at_1_rate": 1.0,
                "pass_at_k_count": 1,
                "pass_at_k_rate": 1.0,
                "unique_locations": 1,
                "mean_best_validation_score": 90.0,
                "median_best_validation_score": 90.0,
            },
            "query_metrics": [
                {
                    "id": "q1",
                    "description": "desc",
                    "query_text": "text",
                    "tags": ["tag"],
                    "actors": ["vehicle"],
                    "behaviors": ["oncoming"],
                    "selected_count": 2,
                    "passed_count": 1,
                    "pass_at_1": True,
                    "pass_at_k": True,
                    "best_validation_score": 90.0,
                    "best_scene_name": "scene-1",
                    "best_sample_idx": 3,
                    "locations": ["boston-seaport"],
                }
            ],
            "behavior_coverage": [
                {
                    "name": "oncoming",
                    "query_count": 1,
                    "pass_at_1_count": 1,
                    "pass_at_k_count": 1,
                    "pass_at_1_rate": 1.0,
                    "pass_at_k_rate": 1.0,
                    "avg_best_validation_score": 90.0,
                }
            ],
            "actor_coverage": [
                {
                    "name": "vehicle",
                    "query_count": 1,
                    "pass_at_1_count": 1,
                    "pass_at_k_count": 1,
                    "pass_at_1_rate": 1.0,
                    "pass_at_k_rate": 1.0,
                    "avg_best_validation_score": 90.0,
                }
            ],
            "tag_coverage": [],
            "location_distribution": [
                {
                    "location": "boston-seaport",
                    "case_count": 2,
                    "passed_case_count": 1,
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_benchmark_metrics(metrics, root)
            self.assertTrue((root / "benchmark_metrics.json").exists())
            self.assertTrue((root / "benchmark_metrics_summary.md").exists())
            self.assertTrue((root / "benchmark_metrics_summary.html").exists())


if __name__ == "__main__":
    unittest.main()
