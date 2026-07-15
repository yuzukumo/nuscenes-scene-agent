import sqlite3
import tempfile
import unittest
from pathlib import Path

from nusc_scene_agent.indexing import migrate_index_schema, simplify_category, validate_index_schema


class IndexingCategoryTest(unittest.TestCase):
    def test_simplify_category_keeps_bicycle_actor_but_not_bicycle_rack(self) -> None:
        self.assertEqual(simplify_category("vehicle.bicycle"), "bicycle")
        self.assertEqual(simplify_category("static_object.bicycle_rack"), "static_object")

    def test_simplify_category_maps_vehicle_families(self) -> None:
        self.assertEqual(simplify_category("human.pedestrian.adult"), "pedestrian")
        self.assertEqual(simplify_category("vehicle.motorcycle"), "motorcycle")
        self.assertEqual(simplify_category("vehicle.bus.rigid"), "bus")
        self.assertEqual(simplify_category("vehicle.trailer"), "truck")
        self.assertEqual(simplify_category("vehicle.car"), "vehicle")


class IndexSchemaTest(unittest.TestCase):
    def test_validate_index_schema_rejects_unversioned_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "index.sqlite"
            conn = sqlite3.connect(str(db_path))
            conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            with self.assertRaisesRegex(RuntimeError, "Unsupported SQLite index schema"):
                validate_index_schema(conn)
            conn.close()

    def test_migrate_index_schema_stamps_compatible_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "index.sqlite"
            conn = sqlite3.connect(str(db_path))
            conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            sample_columns = ", ".join(f"{name} TEXT" for name in sorted({
                "sample_token", "scene_token", "scene_name", "scene_description", "sample_idx",
                "timestamp_us", "location", "ego_x", "ego_y", "ego_yaw",
            }))
            agent_columns = ", ".join(f"{name} TEXT" for name in sorted({
                "ann_token", "sample_token", "scene_token", "scene_name", "sample_idx", "instance_token",
                "category_name", "category_group", "x_ego", "y_ego", "z_ego", "distance", "visibility",
                "ttc", "speed", "rel_vx", "rel_vy", "heading_delta", "num_lidar_pts", "num_radar_pts",
                "width", "length", "height", "is_front", "is_rear", "is_left", "is_right", "is_stationary",
            }))
            conn.execute(f"CREATE TABLE samples ({sample_columns})")
            conn.execute(f"CREATE TABLE agents ({agent_columns})")
            conn.commit()
            conn.close()

            metadata = migrate_index_schema(db_path)

            self.assertEqual(metadata["schema"], "nusc_scene_agent_index")
            self.assertEqual(metadata["schema_version"], "1")
            self.assertEqual(metadata["build_complete"], "true")


if __name__ == "__main__":
    unittest.main()
