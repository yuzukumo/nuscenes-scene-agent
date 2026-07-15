import unittest

from nusc_scene_agent.models import ParsedQuery, RetrievalCandidate, ValidatedCase
from nusc_scene_agent.pipeline import _build_agent_trace, _hypothesis_priority, _select_best_hypothesis, _select_diverse_cases
from nusc_scene_agent.retrieval import RetrievalScoreConfig


def make_case(sample_token: str, score: float, ann_token: str) -> ValidatedCase:
    query = ParsedQuery(
        original_text="query",
        normalized_text="query",
        category_groups=[],
        positions=[],
        behaviors=[],
        near_distance_m=25.0,
        max_ttc_s=6.0,
    )
    candidate = RetrievalCandidate(
        ann_token=ann_token,
        sample_token=sample_token,
        scene_token="scene",
        scene_name="scene-0001",
        sample_idx=0,
        instance_token=ann_token,
        category_name="vehicle.car",
        category_group="vehicle",
        location="boston-seaport",
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
        passed=True,
        behavior_matches={},
        evidence={},
        notes=[],
        timeline=None,
        context_agents=None,
        ego_window=None,
    )


class PipelineSelectionTest(unittest.TestCase):
    def test_select_diverse_cases_prefers_unique_samples(self) -> None:
        cases = [
            make_case("sample_a", 95.0, "a1"),
            make_case("sample_a", 94.0, "a2"),
            make_case("sample_b", 93.0, "b1"),
            make_case("sample_c", 92.0, "c1"),
        ]
        selected = _select_diverse_cases(cases, top_k=3)
        self.assertEqual([case.candidate.sample_token for case in selected], ["sample_a", "sample_b", "sample_c"])

    def test_select_best_hypothesis_prefers_passing_high_score_query(self) -> None:
        rule_query = ParsedQuery(
            original_text="query",
            normalized_text="query",
            category_groups=["vehicle"],
            positions=[],
            behaviors=[],
            near_distance_m=25.0,
            max_ttc_s=6.0,
            risk_terms=["risky"],
            specific_keywords=["planner:rule"],
        )
        llm_query = ParsedQuery(
            original_text="query",
            normalized_text="query",
            category_groups=["vehicle"],
            positions=["left"],
            behaviors=["cut_in"],
            near_distance_m=18.0,
            max_ttc_s=4.0,
            risk_terms=["risky"],
            specific_keywords=["planner:llm_only"],
        )
        rule_case = make_case("sample_rule", 94.0, "rule_case")
        llm_case = make_case("sample_llm", 82.0, "llm_case")

        chosen = _select_best_hypothesis(
            [
                {"query": rule_query, "priority": _hypothesis_priority(rule_query, [rule_case])},
                {"query": llm_query, "priority": _hypothesis_priority(llm_query, [llm_case])},
            ]
        )
        self.assertEqual(chosen["query"].specific_keywords, ["planner:rule"])

    def test_agent_trace_records_retrieval_score_profile(self) -> None:
        query = ParsedQuery(
            original_text="query",
            normalized_text="query",
            category_groups=["vehicle"],
            positions=["front"],
            behaviors=["stopped_lead"],
            near_distance_m=25.0,
            max_ttc_s=6.0,
            specific_keywords=["planner:rule"],
        )
        trace = _build_agent_trace(
            "rule",
            [{"name": "rule", "query": query, "candidates": [], "validated": []}],
            query,
            RetrievalScoreConfig(profile_name="equal"),
        )

        self.assertEqual(trace["retrieval_score_profile"], "equal")
        self.assertEqual(trace["retrieval_score_weights"]["distance"], 1.0)


if __name__ == "__main__":
    unittest.main()
