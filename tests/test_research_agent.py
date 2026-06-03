import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nusc_scene_agent.llm_client import LLMConfig
from nusc_scene_agent.research_agent import (
    _analyze_with_llm_node,
    _artifact_digest,
    _write_report_node,
    run_research_agent,
)


class ResearchAgentTest(unittest.TestCase):
    def test_run_research_agent_requires_llm_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires an LLMConfig"):
            run_research_agent(llm_config=None)

    def test_artifact_digest_reads_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "summary.md"
            path.write_text("# Summary\n\n- Mean IoU: 0.5\n", encoding="utf-8")

            digest = _artifact_digest(path)

            self.assertTrue(digest["exists"])
            self.assertIn("content_excerpt", digest)
            self.assertIn("Mean IoU", digest["content_excerpt"])

    def test_llm_analysis_node_normalizes_response(self) -> None:
        llm_config = LLMConfig(base_url="http://127.0.0.1:11434", model="gemma4:latest")
        payload = {
            "project_positioning": "Benchmark agent.",
            "completed_capabilities": [{"capability": "BEV occupancy", "evidence": "leaderboard"}],
            "industry_gaps": [{"gap": "planner-in-loop", "evidence": "replay only", "priority": "high"}],
            "next_actions": [
                {
                    "action": "Add model-in-the-loop evaluation",
                    "rationale": "Connect benchmark to perception outputs.",
                    "expected_artifact": "failure leaderboard",
                    "risk": "requires external predictions",
                }
            ],
            "agent_workflow": [{"node": "analyze_with_llm", "purpose": "rank gaps"}],
            "benchmark_update_queries": ["vehicle cuts in under occlusion"],
            "claims_to_avoid": ["full simulator"],
        }
        state = {
            "context": {"artifacts": []},
            "llm_config": llm_config,
        }

        with patch("nusc_scene_agent.research_agent.llm_json", return_value=payload):
            result = _analyze_with_llm_node(state)

        self.assertEqual(result["analysis"]["schema"], "research_agent_report_v1")
        self.assertEqual(result["analysis"]["industry_gaps"][0]["priority"], "high")

    def test_write_report_node_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            state = {
                "output_dir": output_dir,
                "analysis": {
                    "schema": "research_agent_report_v1",
                    "project_positioning": "Benchmark agent.",
                    "completed_capabilities": [{"capability": "BEV occupancy", "evidence": "local benchmark"}],
                    "industry_gaps": [],
                    "next_actions": [],
                    "agent_workflow": [],
                    "benchmark_update_queries": [],
                    "claims_to_avoid": [],
                },
            }

            result = _write_report_node(state)

            self.assertTrue((output_dir / "research_agent_report.json").exists())
            self.assertTrue((output_dir / "research_agent_report.md").exists())
            written = json.loads((output_dir / "research_agent_report.json").read_text(encoding="utf-8"))
            self.assertEqual(written["schema"], "research_agent_report_v1")
            self.assertEqual(result["report_path"], output_dir / "research_agent_report.md")


if __name__ == "__main__":
    unittest.main()
