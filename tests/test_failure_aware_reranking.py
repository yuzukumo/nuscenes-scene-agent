import unittest
from types import SimpleNamespace

from nusc_scene_agent.failure_aware_reranking import _build_overview, _ranking_metrics


class FailureAwareRerankingTest(unittest.TestCase):
    def test_ranking_quality_is_gate_aware_and_keeps_ungated_maximum(self) -> None:
        rejected_high = SimpleNamespace(
            passed=False,
            validation_quality_score=99.0,
            candidate=SimpleNamespace(scene_name="rejected", category_group="vehicle", distance=4.0),
        )
        accepted_lower = SimpleNamespace(
            passed=True,
            validation_quality_score=80.0,
            candidate=SimpleNamespace(scene_name="accepted", category_group="vehicle", distance=5.0),
        )

        metrics = _ranking_metrics([rejected_high, accepted_lower])

        self.assertEqual(metrics["best_validation_quality_score"], 80.0)
        self.assertEqual(metrics["max_validation_quality_score"], 99.0)
        self.assertEqual(metrics["best_scene"], "accepted")
        self.assertEqual(metrics["max_quality_scene"], "rejected")

    def test_overview_distinguishes_candidate_generation_from_final_ranking(self) -> None:
        rows = [
            {
                "rule": {"pass_at_1": True, "pass_at_k": True},
                "learned": {"pass_at_1": False, "pass_at_k": True},
                "top1_score_delta": -5.0,
                "best_score_delta": 2.0,
            },
            {
                "rule": {"pass_at_1": False, "pass_at_k": False},
                "learned": {"pass_at_1": False, "pass_at_k": True},
                "top1_score_delta": -1.0,
                "best_score_delta": 3.0,
            },
        ]

        overview = _build_overview(rows)

        self.assertFalse(overview["final_ranker_selected"])
        self.assertTrue(overview["candidate_generator_selected"])
        self.assertEqual(overview["selection_policy"], "validation_gated_candidate_generation")


if __name__ == "__main__":
    unittest.main()
