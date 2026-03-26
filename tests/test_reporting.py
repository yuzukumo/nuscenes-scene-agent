import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nusc_scene_agent.models import ParsedQuery, RetrievalCandidate, ValidatedCase
from nusc_scene_agent.reporting import write_query_report


def make_case() -> ValidatedCase:
    query = ParsedQuery(
        original_text="vehicle cuts in from right side",
        normalized_text="vehicle cuts in from right side",
        category_groups=["vehicle"],
        positions=["right"],
        behaviors=["cut_in"],
        near_distance_m=25.0,
        max_ttc_s=6.0,
        risk_terms=["risky"],
        specific_keywords=["planner:hybrid_selected:rule", "planner:rule"],
    )
    candidate = RetrievalCandidate(
        ann_token="ann-a",
        sample_token="sample-a",
        scene_token="scene-a",
        scene_name="scene-0001",
        sample_idx=5,
        instance_token="instance-a",
        category_name="vehicle.car",
        category_group="vehicle",
        location="boston-seaport",
        distance=4.0,
        ttc=1.0,
        x_ego=3.0,
        y_ego=-1.0,
        speed=0.0,
        rel_vx=-2.0,
        rel_vy=0.0,
        heading_delta=0.0,
        retrieval_score=91.0,
    )
    return ValidatedCase(
        query=query,
        candidate=candidate,
        validation_score=91.0,
        passed=True,
        behavior_matches={"cut_in": True},
        evidence={"min_distance_m": 4.0, "min_ttc_s": 1.0},
        notes=["note"],
        timeline=None,
        context_agents=None,
        ego_window=None,
        map_context={"available": True},
        map_geometries={},
    )


class ReportingTest(unittest.TestCase):
    def test_write_query_report_emits_agent_trace_artifacts(self) -> None:
        case = make_case()
        agent_trace = {
            "mode": "hybrid",
            "selected_hypothesis": "rule",
            "selection_policy": "policy",
            "hypotheses": [
                {
                    "name": "rule",
                    "selected": True,
                    "query": case.query.to_dict(),
                    "candidate_count": 4,
                    "passed_count": 1,
                    "best_validation_score": 91.0,
                },
                {
                    "name": "llm",
                    "selected": False,
                    "query": case.query.to_dict(),
                    "candidate_count": 4,
                    "passed_count": 0,
                    "best_validation_score": 70.0,
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with patch("nusc_scene_agent.reporting.render_case_figure", side_effect=lambda case, path: path):
                write_query_report(case.query, [case], root, agent_trace=agent_trace)

            self.assertTrue((root / "query_trace.json").exists())
            trace = json.loads((root / "query_trace.json").read_text())
            self.assertEqual(trace["selected_hypothesis"], "rule")

            summary_text = (root / "summary.md").read_text()
            self.assertIn("## Agent Trace", summary_text)
            self.assertIn("Selected hypothesis: rule", summary_text)

            case_json = json.loads(next(root.glob("rank_*/case.json")).read_text())
            self.assertIn("agent_trace", case_json)
            self.assertEqual(case_json["agent_trace"]["selected_hypothesis"], "rule")


if __name__ == "__main__":
    unittest.main()
