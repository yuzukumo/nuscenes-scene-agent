import unittest

from nusc_scene_agent.indexing import simplify_category


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


if __name__ == "__main__":
    unittest.main()
