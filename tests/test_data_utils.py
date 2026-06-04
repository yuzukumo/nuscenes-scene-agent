import tempfile
import unittest
from pathlib import Path

from nusc_scene_agent.data_utils import discover_archive_inventory, normalize_map_layout, resolve_archives


class DataUtilsTest(unittest.TestCase):
    def test_discover_archive_inventory_marks_trainval_not_ready_without_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "archives" / "trainval").mkdir(parents=True)
            (root / "archives" / "maps").mkdir(parents=True)
            (root / "archives" / "trainval" / "v1.0-trainval01_blobs.tgz").write_text("blob")
            (root / "archives" / "maps" / "nuScenes-map-expansion-v1.3.zip").write_text("map")

            inventory = discover_archive_inventory(root).to_dict()
            self.assertEqual(inventory["trainval_blob_count"], 1)
            self.assertFalse(inventory["trainval_ready"])

    def test_normalize_map_layout_creates_expected_expansion_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            expansion = root / "expansion"
            expansion.mkdir(parents=True)
            (expansion / "singapore-queenstown.json").write_text("{}")

            normalize_map_layout(root)

            self.assertTrue((root / "maps" / "expansion").exists())
            self.assertTrue((root / "maps" / "expansion" / "singapore-queenstown.json").exists())

    def test_resolve_archives_from_archives_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            mini = root / "archives" / "mini"
            maps = root / "archives" / "maps"
            mini.mkdir(parents=True)
            maps.mkdir(parents=True)
            (mini / "v1.0-mini.tgz").write_text("mini")
            (maps / "nuScenes-map-expansion-v1.3.zip").write_text("map")

            resolved = resolve_archives(root)
            self.assertEqual(
                resolved,
                [
                    (mini / "v1.0-mini.tgz").resolve(),
                    (maps / "nuScenes-map-expansion-v1.3.zip").resolve(),
                ],
            )

    def test_resolve_explicit_archive_relative_to_archives_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            mini = root / "archives" / "mini"
            mini.mkdir(parents=True)
            archive = mini / "v1.0-mini.tgz"
            archive.write_text("mini")

            resolved = resolve_archives(root, ["mini/v1.0-mini.tgz"])
            self.assertEqual(resolved, [archive.resolve()])

    def test_resolve_trainval_profile_prefers_metadata_and_maps_for_smoke_setup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            trainval = root / "archives" / "trainval"
            maps = root / "archives" / "maps"
            trainval.mkdir(parents=True)
            maps.mkdir(parents=True)
            meta = trainval / "v1.0-trainval_meta.tgz"
            blobs = trainval / "v1.0-trainval01_blobs.tgz"
            map_zip = maps / "nuScenes-map-expansion-v1.3.zip"
            meta.write_text("meta")
            blobs.write_text("blob")
            map_zip.write_text("map")

            resolved = resolve_archives(root, profile="trainval")
            self.assertEqual(resolved, [meta.resolve(), map_zip.resolve()])

    def test_resolve_trainval_full_profile_includes_blobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            trainval = root / "archives" / "trainval"
            maps = root / "archives" / "maps"
            trainval.mkdir(parents=True)
            maps.mkdir(parents=True)
            meta = trainval / "v1.0-trainval_meta.tgz"
            blobs = trainval / "v1.0-trainval01_blobs.tgz"
            map_zip = maps / "nuScenes-map-expansion-v1.3.zip"
            meta.write_text("meta")
            blobs.write_text("blob")
            map_zip.write_text("map")

            resolved = resolve_archives(root, profile="trainval-full")
            self.assertEqual(resolved, [meta.resolve(), blobs.resolve(), map_zip.resolve()])


if __name__ == "__main__":
    unittest.main()
