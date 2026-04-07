import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nusc_scene_agent.case_library_enrichment import _reconstruct_query, enrich_case_library
from nusc_scene_agent.models import ParsedQuery, RetrievalCandidate, ValidatedCase


def make_validated_case() -> ValidatedCase:
    query = ParsedQuery(
        original_text="pedestrian crossing in front of ego lane",
        normalized_text="pedestrian crossing in front of ego lane",
        category_groups=["pedestrian"],
        positions=["front"],
        behaviors=["crossing"],
        near_distance_m=18.0,
        max_ttc_s=4.0,
    )
    candidate = RetrievalCandidate(
        ann_token="ann-a",
        sample_token="sample-a",
        scene_token="scene-token",
        scene_name="scene-0001",
        sample_idx=12,
        instance_token="instance-a",
        category_name="human.pedestrian.adult",
        category_group="pedestrian",
        location="singapore-queenstown",
        distance=4.0,
        ttc=1.2,
        x_ego=1.0,
        y_ego=0.0,
        speed=0.0,
        rel_vx=-1.0,
        rel_vy=0.0,
        heading_delta=0.0,
        retrieval_score=88.0,
    )
    return ValidatedCase(
        query=query,
        candidate=candidate,
        validation_score=88.0,
        passed=True,
        behavior_matches={"crossing": True},
        evidence={"min_distance_m": 4.0},
        notes=[],
        timeline=None,
        context_agents=None,
        ego_window=None,
        actor_grounding={"track_start_sample_idx": 10, "track_end_sample_idx": 14, "track_frame_count": 5},
        event_localization={"primary_behavior": "crossing", "start_sample_idx": 11, "end_sample_idx": 13, "peak_sample_idx": 12, "duration_s": 1.0},
    )


class CaseLibraryEnrichmentTest(unittest.TestCase):
    def test_reconstruct_query_uses_case_library_fields(self) -> None:
        query = _reconstruct_query(
            {
                "source_queries": ["pedestrian crossing in front of ego lane"],
                "category_group": "pedestrian",
                "all_behaviors": ["crossing"],
            }
        )
        self.assertIn("pedestrian", query.category_groups)
        self.assertIn("crossing", query.behaviors)

    def test_enrich_case_library_writes_event_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            case_library_path = root / "case_library.json"
            db_path = root / "dummy.sqlite"
            output_path = root / "case_library_enriched.json"

            case_library_path.write_text(
                json.dumps(
                    [
                        {
                            "case_key": "sample-a:instance-a",
                            "sample_token": "sample-a",
                            "instance_token": "instance-a",
                            "source_queries": ["pedestrian crossing in front of ego lane"],
                            "category_group": "pedestrian",
                            "all_behaviors": ["crossing"],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            db_path.write_text("", encoding="utf-8")

            with patch("nusc_scene_agent.case_library_enrichment._load_candidate", return_value=make_validated_case().candidate):
                with patch("nusc_scene_agent.case_library_enrichment.validate_candidate", return_value=make_validated_case()):
                    metadata = enrich_case_library(case_library_path, db_path, output_path)

            self.assertEqual(metadata["enriched_case_count"], 1)
            payload = json.loads(output_path.read_text())
            self.assertEqual(payload[0]["event_peak_sample_idx"], 12)
            self.assertEqual(payload[0]["actor_track_frame_count"], 5)


if __name__ == "__main__":
    unittest.main()
