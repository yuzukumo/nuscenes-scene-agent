import json
import math
import tempfile
import unittest
from pathlib import Path

from nusc_scene_agent.models import ParsedQuery, RetrievalCandidate
from nusc_scene_agent.multimodal_retrieval import (
    rerank_candidates_with_multimodal_model,
    score_candidates_with_multimodal_model,
    write_multimodal_retrieval_report,
)


def _candidate(
    ann_token: str,
    category_group: str,
    x_ego: float,
    y_ego: float,
    distance: float,
    ttc: float,
    retrieval_score: float,
    speed: float = 1.0,
    rel_vy: float = 0.0,
    heading_delta: float = 0.0,
    num_lidar_pts: int = 8,
    num_radar_pts: int = 2,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        ann_token=ann_token,
        sample_token=f"sample-{ann_token}",
        scene_token="scene-token",
        scene_name="scene-0001",
        sample_idx=1,
        instance_token=f"instance-{ann_token}",
        category_name=f"human.{category_group}" if category_group == "pedestrian" else f"vehicle.{category_group}",
        category_group=category_group,
        location="singapore-onenorth",
        distance=distance,
        ttc=ttc,
        x_ego=x_ego,
        y_ego=y_ego,
        speed=speed,
        rel_vx=-1.0,
        rel_vy=rel_vy,
        heading_delta=heading_delta,
        retrieval_score=retrieval_score,
        num_lidar_pts=num_lidar_pts,
        num_radar_pts=num_radar_pts,
    )


class MultimodalRetrievalTest(unittest.TestCase):
    def test_structural_multimodal_reranker_prefers_semantic_bev_motion_match(self) -> None:
        query = ParsedQuery(
            original_text="pedestrian crossing close in front of ego",
            normalized_text="pedestrian crossing close in front of ego",
            category_groups=["pedestrian"],
            positions=["front"],
            behaviors=["crossing"],
            near_distance_m=18.0,
            max_ttc_s=4.0,
            risk_terms=["risky", "urgent"],
        )
        good = _candidate(
            "good",
            "pedestrian",
            x_ego=8.0,
            y_ego=2.4,
            distance=8.4,
            ttc=2.0,
            retrieval_score=1.0,
            speed=1.2,
            rel_vy=2.0,
        )
        weak = _candidate(
            "weak",
            "vehicle",
            x_ego=-25.0,
            y_ego=0.0,
            distance=25.0,
            ttc=math.inf,
            retrieval_score=10.0,
            speed=8.0,
            num_lidar_pts=1,
            num_radar_pts=0,
        )

        scores = score_candidates_with_multimodal_model(query, [weak, good])
        reranked = rerank_candidates_with_multimodal_model(query, [weak, good])

        self.assertEqual(scores[0].ann_token, "good")
        self.assertEqual(reranked[0].ann_token, "good")
        self.assertGreater(scores[0].modality_scores["bev_geometry"], scores[1].modality_scores["bev_geometry"])

    def test_write_multimodal_retrieval_report_outputs_json_csv_markdown(self) -> None:
        query = ParsedQuery(
            original_text="vehicle close ahead",
            normalized_text="vehicle close ahead",
            category_groups=["vehicle"],
            positions=["front"],
            behaviors=["stopped_lead"],
            near_distance_m=20.0,
            max_ttc_s=5.0,
            risk_terms=["risky"],
        )
        scores = score_candidates_with_multimodal_model(
            query,
            [_candidate("vehicle", "vehicle", x_ego=5.0, y_ego=0.0, distance=5.0, ttc=3.0, retrieval_score=1.0)],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = write_multimodal_retrieval_report(query, scores, Path(tmp_dir))

            self.assertTrue(Path(result["json"]).exists())
            self.assertTrue(Path(result["csv"]).exists())
            self.assertTrue(Path(result["markdown"]).exists())
            payload = json.loads(Path(result["json"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "multimodal_scene_retrieval_report_v1")
            self.assertEqual(payload["candidate_count"], 1)


if __name__ == "__main__":
    unittest.main()
