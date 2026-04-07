import tempfile
import unittest
from pathlib import Path

from nusc_scene_agent.benchmark_exports import (
    build_counterfactual_group_summary,
    build_hard_case_taxonomy,
    build_hard_cases,
    build_query_splits,
    build_scenario_group_summary,
    write_benchmark_exports,
)
from nusc_scene_agent.benchmark_schema import BenchmarkQuerySpec
from nusc_scene_agent.case_library import build_case_library
from nusc_scene_agent.models import ParsedQuery, RetrievalCandidate, ValidatedCase


def make_case(
    query_text: str,
    behavior: str,
    actor: str,
    sample_token: str,
    instance_token: str,
    score: float,
    passed: bool,
) -> ValidatedCase:
    query = ParsedQuery(
        original_text=query_text,
        normalized_text=query_text,
        category_groups=[actor],
        positions=["front"],
        behaviors=[behavior],
        near_distance_m=20.0,
        max_ttc_s=5.0,
    )
    candidate = RetrievalCandidate(
        ann_token="ann-" + instance_token,
        sample_token=sample_token,
        scene_token="scene-" + sample_token,
        scene_name="scene-" + sample_token[-1],
        sample_idx=9,
        instance_token=instance_token,
        category_name="vehicle.car" if actor == "vehicle" else "human.pedestrian.adult",
        category_group=actor,
        location="boston-seaport",
        distance=4.0,
        ttc=1.0,
        x_ego=3.0,
        y_ego=1.0,
        speed=0.0,
        rel_vx=-2.0,
        rel_vy=0.0,
        heading_delta=0.0,
        retrieval_score=score,
    )
    return ValidatedCase(
        query=query,
        candidate=candidate,
        validation_score=score,
        passed=passed,
        behavior_matches={behavior: passed},
        evidence={"min_distance_m": 4.0, "min_ttc_s": 1.0},
        notes=["note"],
        timeline=None,
        context_agents=None,
        ego_window=None,
        map_context={"available": True},
        figure_path="/tmp/{0}.png".format(instance_token),
        report_dir="/tmp/{0}".format(instance_token),
    )


class BenchmarkExportsTest(unittest.TestCase):
    def test_build_query_splits_groups_queries(self) -> None:
        passed_case = make_case("left cut in", "cut_in", "vehicle", "sample-a", "inst-a", 88.0, True)
        failed_case = make_case("ped crossing", "crossing", "pedestrian", "sample-b", "inst-b", 72.0, False)
        results = [
            {
                "id": "left_cut",
                "query": passed_case.query,
                "query_spec": BenchmarkQuerySpec(
                    id="left_cut",
                    description="left cut",
                    natural_language="left cut",
                    tags=["merge"],
                    actors=["vehicle"],
                    positions=["left"],
                    behaviors=["cut_in"],
                ),
                "selected_cases": [passed_case],
            },
            {
                "id": "ped_cross",
                "query": failed_case.query,
                "query_spec": BenchmarkQuerySpec(
                    id="ped_cross",
                    description="ped crossing",
                    natural_language="ped crossing",
                    tags=["vru"],
                    actors=["pedestrian"],
                    positions=["front"],
                    behaviors=["crossing"],
                ),
                "selected_cases": [failed_case],
            },
        ]

        splits = build_query_splits(results)
        self.assertEqual(splits["overview"]["query_count"], 2)
        self.assertEqual(splits["overview"]["gap_query_ids"], ["ped_cross"])
        self.assertIn("left_cut", splits["by_position"]["left"])
        self.assertIn("ped_cross", splits["by_behavior"]["crossing"])

    def test_build_hard_cases_includes_failed_and_borderline_entries(self) -> None:
        good_case = make_case("left cut in", "cut_in", "vehicle", "sample-a", "inst-a", 88.0, True)
        borderline_case = make_case("near vehicle", "cut_in", "vehicle", "sample-b", "inst-b", 82.0, True)
        failed_case = make_case("ped crossing", "crossing", "pedestrian", "sample-c", "inst-c", 72.0, False)

        results = [
            {"id": "q1", "query": good_case.query, "selected_cases": [good_case]},
            {"id": "q2", "query": borderline_case.query, "selected_cases": [borderline_case]},
            {"id": "q3", "query": failed_case.query, "selected_cases": [failed_case]},
        ]
        case_library_entries = build_case_library(results)
        hard_cases = build_hard_cases(case_library_entries, borderline_score=85.0)
        labels = [entry["difficulty_label"] for entry in hard_cases]
        self.assertIn("failed", labels)
        self.assertIn("borderline", labels)
        self.assertTrue(all("taxonomy_label" in entry for entry in hard_cases))

    def test_build_hard_case_taxonomy_groups_entries(self) -> None:
        good_case = make_case("left cut in", "cut_in", "vehicle", "sample-a", "inst-a", 88.0, True)
        borderline_case = make_case("near vehicle", "cut_in", "vehicle", "sample-b", "inst-b", 82.0, True)
        failed_case = make_case("ped crossing", "crossing", "pedestrian", "sample-c", "inst-c", 72.0, False)

        results = [
            {"id": "q1", "query": good_case.query, "selected_cases": [good_case]},
            {"id": "q2", "query": borderline_case.query, "selected_cases": [borderline_case]},
            {"id": "q3", "query": failed_case.query, "selected_cases": [failed_case]},
        ]
        case_library_entries = build_case_library(results)
        hard_cases = build_hard_cases(case_library_entries, borderline_score=85.0)
        taxonomy = build_hard_case_taxonomy(hard_cases)

        self.assertEqual(taxonomy["overview"]["hard_case_count"], 2)
        self.assertEqual(len(taxonomy["group_distribution"]) >= 1, True)
        self.assertEqual(len(taxonomy["label_distribution"]) >= 1, True)

    def test_build_counterfactual_group_summary(self) -> None:
        case = make_case("ped crossing", "crossing", "pedestrian", "sample-a", "inst-a", 88.0, True)
        results = [
            {
                "id": "positive_query",
                "query": case.query,
                "query_spec": BenchmarkQuerySpec(
                    id="positive_query",
                    description="positive",
                    natural_language="ped crossing",
                    actors=["pedestrian"],
                    behaviors=["crossing"],
                    reference_case_keys=["sample-a:inst-a"],
                    expect_match=True,
                    benchmark_group="anchor_a",
                    variant_type="positive_canonical",
                ),
                "selected_cases": [case],
            }
        ]
        summary = build_counterfactual_group_summary(results)
        self.assertEqual(summary["overview"]["group_count"], 1)
        self.assertEqual(summary["overview"]["success_at_1_count"], 1)

    def test_build_scenario_group_summary(self) -> None:
        case = make_case("ped crossing", "crossing", "pedestrian", "sample-a", "inst-a", 88.0, True)
        case.event_localization = {
            "primary_behavior": "crossing",
            "start_sample_idx": 6,
            "end_sample_idx": 8,
            "peak_sample_idx": 7,
            "duration_s": 1.0,
        }
        results = [
            {
                "id": "scenario_query",
                "query": case.query,
                "query_spec": BenchmarkQuerySpec(
                    id="scenario_query",
                    description="scenario",
                    natural_language="ped crossing",
                    tags=["scenario_mining"],
                    actors=["pedestrian"],
                    behaviors=["crossing"],
                    reference_case_keys=["sample-a:inst-a"],
                    reference_scene_names=["scene-a"],
                    reference_instance_tokens=["inst-a"],
                    reference_event_sample_range=[6, 8],
                    reference_peak_sample_idx=7,
                    expect_match=True,
                    benchmark_group="scenario_a",
                    variant_type="scenario_canonical",
                ),
                "selected_cases": [case],
            }
        ]
        summary = build_scenario_group_summary(results)
        self.assertEqual(summary["overview"]["group_count"], 1)
        self.assertEqual(summary["overview"]["scene_success_at_1_count"], 1)
        self.assertEqual(summary["overview"]["reference_success_at_1_count"], 1)
        self.assertAlmostEqual(summary["overview"]["mean_event_iou"], 1.0)

    def test_write_benchmark_exports_emits_expected_files(self) -> None:
        case = make_case("left cut in", "cut_in", "vehicle", "sample-a", "inst-a", 82.0, False)
        case.event_localization = {
            "primary_behavior": "cut_in",
            "start_sample_idx": 6,
            "end_sample_idx": 8,
            "peak_sample_idx": 7,
            "duration_s": 1.0,
        }
        results = [
            {
                "id": "left_cut",
                "query": case.query,
                "query_spec": BenchmarkQuerySpec(
                    id="left_cut",
                    description="left cut",
                    natural_language="left cut",
                    tags=["merge"],
                    actors=["vehicle"],
                    positions=["left"],
                    behaviors=["cut_in"],
                    reference_scene_names=["scene-a"],
                    reference_instance_tokens=["inst-a"],
                    reference_event_sample_range=[6, 8],
                    reference_peak_sample_idx=7,
                    reference_case_keys=["sample-a:inst-a"],
                    expect_match=True,
                    benchmark_group="anchor_a",
                    variant_type="scenario_canonical",
                ),
                "selected_cases": [case],
            }
        ]
        case_library_entries = build_case_library(results)

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_benchmark_exports(results, case_library_entries, root)
            self.assertTrue((root / "benchmark_splits.json").exists())
            self.assertTrue((root / "benchmark_splits_summary.md").exists())
            self.assertTrue((root / "hard_cases.json").exists())
            self.assertTrue((root / "hard_cases.csv").exists())
            self.assertTrue((root / "hard_cases_summary.md").exists())
            self.assertTrue((root / "hard_case_taxonomy.json").exists())
            self.assertTrue((root / "hard_case_taxonomy_summary.md").exists())
            self.assertTrue((root / "counterfactual_group_summary.json").exists())
            self.assertTrue((root / "counterfactual_group_summary.md").exists())
            self.assertTrue((root / "scenario_group_summary.json").exists())
            self.assertTrue((root / "scenario_group_summary.md").exists())


if __name__ == "__main__":
    unittest.main()
