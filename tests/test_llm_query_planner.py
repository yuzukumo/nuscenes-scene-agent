import unittest
from unittest.mock import patch

from nusc_scene_agent.llm_client import LLMConfig
from nusc_scene_agent.llm_query_planner import merge_queries, plan_query_with_llm, resolve_hybrid_queries, resolve_query
from nusc_scene_agent.models import ParsedQuery
from nusc_scene_agent.query_parser import parse_query


class LLMQueryPlannerTest(unittest.TestCase):
    def test_plan_query_with_llm_normalizes_aliases(self) -> None:
        config = LLMConfig(base_url="https://example.com", api_key="key", model="model")
        payload = {
            "category_groups": ["car", "person"],
            "positions": ["ahead", "left"],
            "behaviors": ["merge"],
            "risk_terms": ["dangerous"],
            "near_distance_m": 14,
            "max_ttc_s": 4,
            "specific_keywords": ["night", "rain"],
        }
        with patch("nusc_scene_agent.llm_query_planner.responses_json", return_value=payload):
            planned = plan_query_with_llm("car merges from left", config)

        self.assertEqual(planned.category_groups, ["vehicle", "pedestrian"])
        self.assertEqual(planned.positions, ["front", "left"])
        self.assertEqual(planned.behaviors, ["cut_in"])
        self.assertEqual(planned.risk_terms, ["risky"])
        self.assertEqual(planned.near_distance_m, 14.0)
        self.assertIn("planner:llm", planned.specific_keywords)

    def test_merge_queries_preserves_rule_and_llm_signals(self) -> None:
        rule_query = parse_query("pedestrian crossing in front of ego lane")
        llm_query = ParsedQuery(
            original_text=rule_query.original_text,
            normalized_text=rule_query.normalized_text,
            category_groups=["pedestrian"],
            positions=["front", "left"],
            behaviors=["crossing"],
            near_distance_m=18.0,
            max_ttc_s=4.0,
            risk_terms=["risky"],
            specific_keywords=["planner:llm"],
        )
        merged = merge_queries(rule_query, llm_query)
        self.assertIn("pedestrian", merged.category_groups)
        self.assertIn("front", merged.positions)
        self.assertNotIn("left", merged.positions)
        self.assertEqual(merged.near_distance_m, 25.0)
        self.assertIn("planner:hybrid", merged.specific_keywords)

    def test_merge_queries_avoids_conflicting_behavior_union(self) -> None:
        rule_query = parse_query("a bike rider cuts across the path in front of the car")
        llm_query = ParsedQuery(
            original_text=rule_query.original_text,
            normalized_text=rule_query.normalized_text,
            category_groups=["bicycle"],
            positions=["front"],
            behaviors=["cut_in"],
            near_distance_m=18.0,
            max_ttc_s=4.0,
            risk_terms=["risky"],
            specific_keywords=["planner:llm"],
        )
        merged = merge_queries(rule_query, llm_query)
        self.assertEqual(merged.category_groups, ["bicycle"])
        self.assertEqual(merged.behaviors, ["crossing"])
        self.assertEqual(merged.positions, ["front"])
        self.assertEqual(merged.risk_terms, [])

    def test_resolve_query_uses_mocked_llm_for_hybrid_mode(self) -> None:
        llm_query = ParsedQuery(
            original_text="vehicle cuts in from right side",
            normalized_text="vehicle cuts in from right side",
            category_groups=["vehicle"],
            positions=["right"],
            behaviors=["cut_in"],
            near_distance_m=20.0,
            max_ttc_s=5.0,
            risk_terms=[],
            specific_keywords=["planner:llm"],
        )
        with patch("nusc_scene_agent.llm_query_planner.plan_query_with_llm", return_value=llm_query):
            merged = resolve_query(
                "vehicle cuts in from right side",
                mode="hybrid",
                config=LLMConfig(base_url="https://example.com", api_key="key", model="model"),
            )
        self.assertIn("vehicle", merged.category_groups)
        self.assertIn("right", merged.positions)
        self.assertIn("cut_in", merged.behaviors)

    def test_resolve_hybrid_queries_dedupes_equivalent_hypotheses(self) -> None:
        llm_query = ParsedQuery(
            original_text="vehicle cuts in from right side",
            normalized_text="vehicle cuts in from right side",
            category_groups=["vehicle"],
            positions=["right"],
            behaviors=["cut_in"],
            near_distance_m=20.0,
            max_ttc_s=5.0,
            risk_terms=[],
            specific_keywords=["planner:llm"],
        )
        with patch("nusc_scene_agent.llm_query_planner.plan_query_with_llm", return_value=llm_query):
            queries = resolve_hybrid_queries(
                "vehicle cuts in from right side",
                config=LLMConfig(base_url="https://example.com", api_key="key", model="model"),
            )
        self.assertGreaterEqual(len(queries), 2)
        self.assertLessEqual(len(queries), 3)

    def test_resolve_query_falls_back_when_llm_returns_empty_structure(self) -> None:
        empty_llm_query = ParsedQuery(
            original_text="pedestrian crossing in front of ego lane",
            normalized_text="pedestrian crossing in front of ego lane",
            category_groups=[],
            positions=[],
            behaviors=[],
            near_distance_m=25.0,
            max_ttc_s=6.0,
            risk_terms=[],
            specific_keywords=["planner:llm"],
        )
        with patch("nusc_scene_agent.llm_query_planner.plan_query_with_llm", return_value=empty_llm_query):
            resolved = resolve_query(
                "pedestrian crossing in front of ego lane",
                mode="llm",
                config=LLMConfig(base_url="https://example.com", api_key="key", model="model"),
            )
        self.assertIn("pedestrian", resolved.category_groups)
        self.assertIn("crossing", resolved.behaviors)
        self.assertIn("planner:llm_fallback", resolved.specific_keywords)

    def test_resolve_query_falls_back_when_llm_raises(self) -> None:
        with patch("nusc_scene_agent.llm_query_planner.plan_query_with_llm", side_effect=RuntimeError("boom")):
            resolved = resolve_query(
                "pedestrian crossing in front of ego lane",
                mode="hybrid",
                config=LLMConfig(base_url="https://example.com", api_key="key", model="model"),
            )
        self.assertIn("pedestrian", resolved.category_groups)
        self.assertIn("planner:llm_error", resolved.specific_keywords)


if __name__ == "__main__":
    unittest.main()
