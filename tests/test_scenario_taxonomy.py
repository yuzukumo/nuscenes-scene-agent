import unittest

from nusc_scene_agent.scenario_taxonomy import (
    build_taxonomy_coverage,
    load_scenario_taxonomy,
    map_scenario_labels,
    taxonomy_label_index,
)


class ScenarioTaxonomyTest(unittest.TestCase):
    def test_backend_match_labels_are_unambiguous(self) -> None:
        taxonomy = load_scenario_taxonomy()
        for backend in ["nuscenes", "nuplan", "bench2drive", "carla"]:
            index = taxonomy_label_index(taxonomy, backend)
            self.assertFalse({label: values for label, values in index.items() if len(values) > 1})

    def test_maps_native_bench2drive_family(self) -> None:
        mapping = map_scenario_labels("bench2drive", ["PedestrianCrossing"])
        self.assertEqual(mapping["family_ids"], ["vru_crossing"])
        self.assertEqual(mapping["unmapped_labels"], [])

    def test_coverage_reports_unmapped_labels(self) -> None:
        coverage = build_taxonomy_coverage(
            "bench2drive",
            {"PedestrianCrossing": 8, "UnknownScenario": 2},
        )
        self.assertEqual(coverage["mapped_sample_rate"], 0.8)
        self.assertEqual(coverage["canonical_family_counts"]["vru_crossing"], 8)
        self.assertEqual(coverage["unmapped_label_counts"], {"UnknownScenario": 2})


if __name__ == "__main__":
    unittest.main()
