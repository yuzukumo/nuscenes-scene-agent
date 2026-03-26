import unittest

from nusc_scene_agent.case_library import build_case_library
from nusc_scene_agent.models import ParsedQuery, RetrievalCandidate, ValidatedCase


def make_case(
    query_text: str,
    sample_token: str,
    instance_token: str,
    score: float,
    passed: bool,
    figure_suffix: str,
) -> ValidatedCase:
    query = ParsedQuery(
        original_text=query_text,
        normalized_text=query_text,
        category_groups=[],
        positions=[],
        behaviors=["crossing"],
        near_distance_m=25.0,
        max_ttc_s=6.0,
        specific_keywords=["planner:rule", "planner:hybrid_selected:rule"],
    )
    candidate = RetrievalCandidate(
        ann_token="ann-" + instance_token + figure_suffix,
        sample_token=sample_token,
        scene_token="scene-token",
        scene_name="scene-0001",
        sample_idx=12,
        instance_token=instance_token,
        category_name="human.pedestrian.adult",
        category_group="pedestrian",
        location="singapore-queenstown",
        distance=5.0,
        ttc=1.0,
        x_ego=1.0,
        y_ego=0.0,
        speed=0.0,
        rel_vx=-1.0,
        rel_vy=0.0,
        heading_delta=0.0,
        retrieval_score=score,
    )
    return ValidatedCase(
        query=query,
        candidate=candidate,
        validation_score=score,
        passed=passed,
        behavior_matches={"crossing": passed},
        evidence={"min_distance_m": 5.0, "min_ttc_s": 1.0},
        notes=["note-" + figure_suffix],
        timeline=None,
        context_agents=None,
        ego_window=None,
        map_context={"available": True, "actor_on_crosswalk_any": True},
        map_geometries={},
        figure_path="/tmp/figure-" + figure_suffix + ".png",
        report_dir="/tmp/report-" + figure_suffix,
    )


class CaseLibraryTest(unittest.TestCase):
    def test_build_case_library_merges_same_case_across_queries(self) -> None:
        query_a_case = make_case("query a", "sample-a", "instance-a", 92.0, True, "a")
        query_b_case = make_case("query b", "sample-a", "instance-a", 88.0, True, "b")
        other_case = make_case("query c", "sample-b", "instance-b", 70.0, False, "c")

        entries = build_case_library(
            [
                {"id": "q_a", "query": query_a_case.query, "selected_cases": [query_a_case]},
                {"id": "q_b", "query": query_b_case.query, "selected_cases": [query_b_case]},
                {"id": "q_c", "query": other_case.query, "selected_cases": [other_case]},
            ]
        )

        self.assertEqual(len(entries), 2)
        merged = entries[0]
        self.assertEqual(merged["case_key"], "sample-a:instance-a")
        self.assertEqual(merged["source_query_ids"], ["q_a", "q_b"])
        self.assertEqual(merged["validation_score"], 92.0)
        self.assertEqual(merged["selected_hypothesis"], "rule")


if __name__ == "__main__":
    unittest.main()
