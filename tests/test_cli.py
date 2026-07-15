import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nusc_scene_agent.cli import _rate_string, _retrieval_score_sweep_row, _threshold_scale_overrides, _with_baseline_deltas
from nusc_scene_agent.llm_client import LLMConfig, inspect_ollama_model, llm_json, verify_ollama_model


class RetrievalScoreSweepCliTest(unittest.TestCase):
    def test_threshold_scale_overrides_only_change_metric_thresholds(self) -> None:
        overrides = _threshold_scale_overrides(1.15)

        self.assertAlmostEqual(overrides["crossing_lateral_span_m"], 2.3)
        self.assertAlmostEqual(overrides["cut_in_closing_dx_m"], -0.575)
        self.assertNotIn("cut_in_closing_min_consecutive_frames", overrides)

    def test_rate_string_marks_non_applicable_reference_metrics(self) -> None:
        self.assertEqual(_rate_string(0, 0), "n/a")
        self.assertEqual(_rate_string(3, 4), "3/4 (75.0%)")

    def test_threshold_sweep_cli_has_deterministic_defaults(self) -> None:
        from nusc_scene_agent.cli import _build_parser

        args = _build_parser().parse_args(["benchmark-threshold-sweep"])

        self.assertEqual(args.scale, [])
        self.assertEqual(args.query_mode, "rule")
        self.assertEqual(args.rerank_mode, "none")

    def test_retrieval_score_sweep_row_reads_benchmark_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "benchmark_metrics.json").write_text(
                json.dumps(
                    {
                        "overview": {
                            "query_count": 4,
                            "pass_at_1_count": 3,
                            "pass_at_1_rate": 0.75,
                            "pass_at_k_count": 4,
                            "pass_at_k_rate": 1.0,
                            "mean_best_validation_score": 87.5,
                            "unique_case_count": 6,
                            "unique_passed_case_count": 5,
                        },
                        "reference_metrics": {
                            "query_count": 2,
                            "objective_at_1_count": 1,
                            "objective_at_1_rate": 0.5,
                            "objective_at_k_count": 2,
                            "objective_at_k_rate": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            row = _retrieval_score_sweep_row("default", output)

        self.assertEqual(row["profile"], "default")
        self.assertEqual(row["anchor_pass_at_1"], "3/4 (75.0%)")
        self.assertEqual(row["reference_objective_at_1"], "1/2 (50.0%)")
        self.assertEqual(row["validation_acceptance_at_1"], "3/4 (75.0%)")
        self.assertEqual(row["mean_best_validation_quality_score"], 87.5)
        self.assertEqual(row["mean_best_validation_score"], 87.5)

    def test_with_baseline_deltas_uses_first_profile_as_reference(self) -> None:
        rows = _with_baseline_deltas(
            [
                {"profile": "default", "anchor_pass_at_1_rate": 0.75, "mean_best_validation_score": 80.0},
                {"profile": "equal", "anchor_pass_at_1_rate": 0.5, "mean_best_validation_score": 81.5},
            ]
        )
        self.assertEqual(rows[0]["delta_anchor_pass_at_1_rate"], 0.0)
        self.assertEqual(rows[1]["delta_anchor_pass_at_1_rate"], -0.25)
        self.assertEqual(rows[1]["delta_mean_best_validation_score"], 1.5)


class OllamaModelMetadataTest(unittest.TestCase):
    def test_inspect_ollama_model_records_digest_when_available(self) -> None:
        with patch("nusc_scene_agent.llm_client._post_json") as post_json, patch(
            "nusc_scene_agent.llm_client._get_json"
        ) as get_json:
            post_json.return_value = {"modelfile": "FROM gemma4", "details": {"family": "gemma"}}
            get_json.return_value = {"models": [{"name": "gemma4:latest", "digest": "sha256:test"}]}

            metadata = inspect_ollama_model(
                LLMConfig(base_url="http://127.0.0.1:11434/api/chat", model="gemma4:latest")
            )

        self.assertEqual(metadata["base_url"], "http://127.0.0.1:11434")
        self.assertEqual(metadata["digest"], "sha256:test")
        self.assertEqual(metadata["model_identifier"], "sha256:test")
        self.assertFalse(metadata["reproducible"])
        self.assertEqual(metadata["tag"]["name"], "gemma4:latest")

    def test_verify_requires_expected_digest_when_requested(self) -> None:
        config = LLMConfig(
            base_url="http://127.0.0.1:11434",
            model="gemma4:latest",
            require_digest=True,
        )
        with patch(
            "nusc_scene_agent.llm_client.inspect_ollama_model",
            return_value={"digest": "sha256:actual", "digest_matches": True},
        ):
            with self.assertRaisesRegex(RuntimeError, "stable Ollama digest is required"):
                verify_ollama_model(config)

    def test_verify_requires_digest_before_contacting_ollama(self) -> None:
        config = LLMConfig(
            base_url="http://127.0.0.1:11434",
            model="gemma4:latest",
            require_digest=True,
        )
        with patch("nusc_scene_agent.llm_client.inspect_ollama_model") as inspect:
            with self.assertRaisesRegex(RuntimeError, "stable Ollama digest is required"):
                verify_ollama_model(config)
        inspect.assert_not_called()

    def test_verify_ollama_model_rejects_digest_mismatch(self) -> None:
        config = LLMConfig(
            base_url="http://127.0.0.1:11434",
            model="gemma4:latest",
            digest="sha256:expected",
        )
        with patch(
            "nusc_scene_agent.llm_client.inspect_ollama_model",
            return_value={"digest": "sha256:actual", "digest_matches": False},
        ):
            with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
                verify_ollama_model(config)

    def test_llm_json_verifies_configured_digest(self) -> None:
        config = LLMConfig(
            base_url="http://127.0.0.1:11434",
            model="gemma4:latest",
            digest="sha256:test",
        )
        with patch("nusc_scene_agent.llm_client.verify_ollama_model") as verify, patch(
            "nusc_scene_agent.llm_client._post_json",
            return_value={"message": {"content": "{}"}},
        ):
            self.assertEqual(llm_json(config, "", "test"), {})
        verify.assert_called_once_with(config)

    def test_llm_json_reuses_verified_model_identity(self) -> None:
        config = LLMConfig(
            base_url="http://127.0.0.1:11434",
            model="gemma4:latest",
            digest="sha256:test",
            resolved_digest="sha256:test",
            digest_verified=True,
        )
        with patch("nusc_scene_agent.llm_client.verify_ollama_model") as verify, patch(
            "nusc_scene_agent.llm_client._post_json",
            return_value={"message": {"content": "{}"}},
        ):
            self.assertEqual(llm_json(config, "", "test"), {})
        verify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
