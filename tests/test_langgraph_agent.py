import unittest
from unittest.mock import patch

from nusc_scene_agent.langgraph_agent import _build_framework_trace, run_langgraph_query_pipeline
from nusc_scene_agent.models import ParsedQuery, RetrievalCandidate, ValidatedCase


def make_case(sample_token: str, score: float, ann_token: str, passed: bool = True) -> ValidatedCase:
    query = ParsedQuery(
        original_text="query",
        normalized_text="query",
        category_groups=["vehicle"],
        positions=["front"],
        behaviors=[],
        near_distance_m=25.0,
        max_ttc_s=6.0,
        risk_terms=[],
        specific_keywords=["planner:rule"],
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
        passed=passed,
        behavior_matches={},
        evidence={},
        notes=[],
        timeline=None,
        context_agents=None,
        ego_window=None,
    )


class LangGraphAgentTest(unittest.TestCase):
    def test_build_framework_trace_summarizes_hypotheses(self) -> None:
        query = ParsedQuery(
            original_text="query",
            normalized_text="query",
            category_groups=["vehicle"],
            positions=["front"],
            behaviors=[],
            near_distance_m=25.0,
            max_ttc_s=6.0,
            risk_terms=[],
            specific_keywords=["planner:rule"],
        )
        trace = _build_framework_trace(
            "hybrid",
            [
                {
                    "name": "rule",
                    "candidates": [object(), object()],
                    "validated": [make_case("sample-a", 91.0, "a1"), make_case("sample-b", 82.0, "b1", passed=False)],
                }
            ],
            query,
        )
        self.assertEqual(trace["framework"], "langgraph")
        self.assertEqual(trace["selected_hypothesis"], "rule")
        self.assertEqual(trace["nodes"][0]["hypothesis_count"], 1)
        self.assertEqual(trace["nodes"][1]["candidate_counts"]["rule"], 2)

    def test_run_langgraph_query_pipeline_surfaces_missing_dependency(self) -> None:
        with patch(
            "nusc_scene_agent.langgraph_agent._build_langgraph_app",
            side_effect=RuntimeError("LangGraph is not installed."),
        ):
            with self.assertRaisesRegex(RuntimeError, "LangGraph is not installed."):
                run_langgraph_query_pipeline(
                    db_path="artifacts/index/v1.0-mini.sqlite",
                    query_text="vehicle close in front",
                    output_root="outputs/langgraph_query",
                )


if __name__ == "__main__":
    unittest.main()
