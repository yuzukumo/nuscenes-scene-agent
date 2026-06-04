import unittest

from nusc_scene_agent.failure_aware_reranking import _build_overview


class FailureAwareRerankingTest(unittest.TestCase):
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
