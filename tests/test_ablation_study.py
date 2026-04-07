import json
import tempfile
import unittest
from pathlib import Path

from nusc_scene_agent.ablation_study import default_ablation_profiles, write_ablation_manifest


class AblationStudyTest(unittest.TestCase):
    def test_default_ablation_profiles_for_hybrid_include_expected_variants(self) -> None:
        profiles = default_ablation_profiles(base_query_mode="hybrid", base_rerank_mode="llm")
        self.assertEqual(
            [profile["name"] for profile in profiles],
            ["full_system", "no_rerank", "no_map_context", "no_event_localization"],
        )
        self.assertEqual(profiles[0]["validation_config"].name, "full_system")
        self.assertFalse(profiles[2]["validation_config"].enable_map_context)
        self.assertFalse(profiles[3]["validation_config"].enable_event_localization)

    def test_default_ablation_profiles_skip_no_rerank_when_base_has_no_rerank(self) -> None:
        profiles = default_ablation_profiles(base_query_mode="rule", base_rerank_mode="none")
        self.assertEqual(
            [profile["name"] for profile in profiles],
            ["full_system", "no_map_context", "no_event_localization"],
        )

    def test_write_ablation_manifest(self) -> None:
        profiles = default_ablation_profiles(base_query_mode="hybrid", base_rerank_mode="llm")
        profiles[0]["output_dir"] = "/tmp/full_system"
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_ablation_manifest(profiles, root)
            manifest_path = root / "ablation_manifest.json"
            self.assertTrue(manifest_path.exists())
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload[0]["name"], "full_system")
            self.assertEqual(payload[0]["validation_config"]["name"], "full_system")


if __name__ == "__main__":
    unittest.main()
