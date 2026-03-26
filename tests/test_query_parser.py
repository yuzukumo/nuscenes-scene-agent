import unittest

from nusc_scene_agent.query_parser import parse_query


class QueryParserTest(unittest.TestCase):
    def test_parse_english_crossing_query(self) -> None:
        query = parse_query("pedestrian crossing in front of ego lane 10 m")
        self.assertIn("pedestrian", query.category_groups)
        self.assertIn("front", query.positions)
        self.assertIn("crossing", query.behaviors)
        self.assertEqual(query.near_distance_m, 10.0)

    def test_parse_english_cut_in_query(self) -> None:
        query = parse_query("vehicle cuts in from right side")
        self.assertIn("vehicle", query.category_groups)
        self.assertIn("right", query.positions)
        self.assertIn("cut_in", query.behaviors)

    def test_parse_language_stress_synonyms(self) -> None:
        query = parse_query("a vehicle is running very close alongside on the driver's side")
        self.assertIn("vehicle", query.category_groups)
        self.assertIn("left", query.positions)
        self.assertIn("risky", query.risk_terms)

    def test_parse_cyclist_query_does_not_treat_ego_car_as_actor(self) -> None:
        query = parse_query("a bike rider cuts across the path in front of the car")
        self.assertIn("bicycle", query.category_groups)
        self.assertNotIn("vehicle", query.category_groups)
        self.assertIn("crossing", query.behaviors)


if __name__ == "__main__":
    unittest.main()
