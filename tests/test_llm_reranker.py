import unittest
from unittest.mock import patch

from nusc_scene_agent.llm_client import LLMConfig
from nusc_scene_agent.llm_reranker import rerank_candidates_with_llm
from nusc_scene_agent.models import ParsedQuery, RetrievalCandidate


def make_candidate(idx: int) -> RetrievalCandidate:
    return RetrievalCandidate(
        ann_token="ann-{0}".format(idx),
        sample_token="sample-{0}".format(idx),
        scene_token="scene",
        scene_name="scene-{0}".format(idx),
        sample_idx=idx,
        instance_token="instance-{0}".format(idx),
        category_name="vehicle.car",
        category_group="vehicle",
        location="boston-seaport",
        distance=5.0 + idx,
        ttc=1.0 + idx,
        x_ego=idx,
        y_ego=0.0,
        speed=0.0,
        rel_vx=-1.0,
        rel_vy=0.0,
        heading_delta=0.0,
        retrieval_score=10.0 - idx,
    )


class LLMRerankerTest(unittest.TestCase):
    def test_rerank_candidates_with_llm_uses_returned_order(self) -> None:
        query = ParsedQuery(
            original_text="oncoming vehicle close to ego",
            normalized_text="oncoming vehicle close to ego",
            category_groups=["vehicle"],
            positions=["front"],
            behaviors=["oncoming"],
            near_distance_m=20.0,
            max_ttc_s=5.0,
        )
        candidates = [make_candidate(1), make_candidate(2), make_candidate(3)]
        with patch("nusc_scene_agent.llm_reranker.responses_json", return_value={"ranking": [3, 1]}):
            reranked = rerank_candidates_with_llm(
                query,
                candidates,
                LLMConfig(base_url="https://example.com", api_key="key", model="model"),
            )
        self.assertEqual([item.ann_token for item in reranked], ["ann-3", "ann-1", "ann-2"])


if __name__ == "__main__":
    unittest.main()
