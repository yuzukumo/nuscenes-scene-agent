import json
import tempfile
import unittest
from pathlib import Path

from nusc_scene_agent.human_audit import evaluate_human_audit_set, generate_human_audit_set


def make_case(case_id: int, behavior: str, passed: bool = True) -> dict:
    return {
        "case_key": "sample-{0}:instance-{0}".format(case_id),
        "scene_name": "scene-{0:04d}".format(case_id),
        "scene_token": "scene-token-{0}".format(case_id),
        "sample_idx": case_id,
        "sample_token": "sample-{0}".format(case_id),
        "instance_token": "instance-{0}".format(case_id),
        "category_name": "vehicle.car" if behavior != "crossing" else "human.pedestrian.adult",
        "category_group": "vehicle" if behavior != "crossing" else "pedestrian",
        "location": "boston-seaport",
        "passed": passed,
        "validation_score": 92.0 if passed else 65.0,
        "retrieval_score": 11.5,
        "min_distance_m": 3.0,
        "min_ttc_s": 1.2,
        "source_query_ids": ["query-{0}".format(case_id)],
        "source_queries": ["find {0}".format(behavior)],
        "source_query_tags": ["scenario_mining"],
        "matched_behaviors": [behavior] if passed else [],
        "all_behaviors": [behavior],
        "event_start_sample_idx": 10,
        "event_end_sample_idx": 14,
        "event_peak_sample_idx": 12,
        "figure_path": "/tmp/case-{0}.png".format(case_id),
        "report_dir": "/tmp/case-{0}".format(case_id),
        "notes": ["example"],
    }


class HumanAuditTest(unittest.TestCase):
    def test_generate_human_audit_set_writes_annotation_templates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_library = root / "case_library.json"
            case_library.write_text(
                json.dumps(
                    [
                        make_case(1, "crossing"),
                        make_case(2, "cut_in"),
                        make_case(3, "stopped_lead", passed=False),
                    ]
                ),
                encoding="utf-8",
            )

            manifest = generate_human_audit_set(case_library, root / "audit", sample_size=2, seed=1)

            jsonl_path = Path(manifest["jsonl"])
            csv_path = Path(manifest["csv"])
            self.assertTrue(jsonl_path.exists())
            self.assertTrue(csv_path.exists())
            self.assertTrue((root / "audit" / "README.md").exists())
            self.assertTrue((root / "audit" / "review_queue.md").exists())
            self.assertIn(
                "system_event_start_sample_idx",
                csv_path.read_text(encoding="utf-8").splitlines()[0],
            )
            rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(manifest["requested_sample_size"], 2)
            self.assertEqual(manifest["sample_size"], 2)
            self.assertEqual(rows[0]["schema"], "human_audit_item_v1")
            self.assertIn(rows[0]["behavior"], {"crossing", "cut_in", "stopped_lead"})
            self.assertIsNone(rows[0]["labels"]["semantic_match"])
            self.assertFalse(Path(rows[0]["evidence"]["figure_path"]).is_absolute())

            written_manifest = json.loads(
                (root / "audit" / "human_audit_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(written_manifest["path_base"], "audit_directory")
            self.assertFalse(Path(written_manifest["case_library"]).is_absolute())
            self.assertEqual(written_manifest["output_dir"], ".")
            self.assertEqual(written_manifest["requested_sample_size"], 2)
            self.assertEqual(written_manifest["sample_size"], 2)

    def test_generate_human_audit_set_records_requested_and_available_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_library = root / "case_library.json"
            case_library.write_text(
                json.dumps([make_case(1, "crossing")]),
                encoding="utf-8",
            )

            generate_human_audit_set(case_library, root / "audit", sample_size=100, seed=1)
            written_manifest = json.loads(
                (root / "audit" / "human_audit_manifest.json").read_text(encoding="utf-8")
            )

            self.assertEqual(written_manifest["requested_sample_size"], 100)
            self.assertEqual(written_manifest["sample_size"], 1)
            self.assertEqual(written_manifest["audit_item_count"], 1)
            self.assertTrue(written_manifest["sample_size_limited_by_available_cases"])

    def test_evaluate_human_audit_set_computes_rates_and_event_iou(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotation_path = root / "human_audit_items.jsonl"
            items = []
            for idx, semantic_match in [(1, True), (2, False)]:
                item = {
                    "schema": "human_audit_item_v1",
                    "audit_id": "audit_{0:04d}".format(idx),
                    "case_key": "case-{0}".format(idx),
                    "behavior": "crossing",
                    "system": {
                        "event_start_sample_idx": 10,
                        "event_end_sample_idx": 14,
                        "event_peak_sample_idx": 12,
                    },
                    "labels": {
                        "semantic_match": semantic_match,
                        "primary_actor_correct": semantic_match,
                        "behavior_correct": semantic_match,
                        "event_window_correct": semantic_match,
                        "event_start_sample_idx": 12,
                        "event_end_sample_idx": 16,
                        "event_peak_sample_idx": 13,
                        "confidence": 0.8,
                        "notes": "reviewed",
                    },
                }
                items.append(item)
            annotation_path.write_text("\n".join(json.dumps(item) for item in items) + "\n", encoding="utf-8")

            metrics = evaluate_human_audit_set(annotation_path, root / "eval")

            self.assertEqual(metrics["overview"]["item_count"], 2)
            self.assertEqual(metrics["overview"]["labeled_count"], 2)
            self.assertEqual(metrics["overview"]["semantic_match_rate"], 0.5)
            self.assertEqual(metrics["overview"]["mean_event_iou"], round(3 / 7, 4))
            self.assertEqual(len(metrics["false_positive_cases"]), 1)
            self.assertTrue((root / "eval" / "human_audit_metrics.md").exists())


if __name__ == "__main__":
    unittest.main()
