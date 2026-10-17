import unittest

from src.evaluation.parameter_modes import (
    CLASS_ORDER,
    DATASET_LABEL_GROUPS,
    direction_checks,
    run_simulation_experiment,
)


class ParameterModesTests(unittest.TestCase):
    def test_simulation_establishes_the_expected_shift_directions(self):
        result = run_simulation_experiment(seed=417, repeats=18, n_points=1600)

        self.assertTrue(all(direction_checks(result["parameter_shifts"]).values()))
        self.assertEqual(int(result["confusion_matrix"].sum()), 3 * 18)
        self.assertEqual(tuple(result["recalls"]), CLASS_ORDER)
        self.assertTrue(all(value is not None for value in result["recalls"].values()))

    def test_dataset_label_mapping_includes_all_seven_shading_codes(self):
        self.assertEqual(len(DATASET_LABEL_GROUPS["lareunion"]), 7)
        self.assertEqual(set(DATASET_LABEL_GROUPS["lareunion"].values()), {"shading"})
        self.assertEqual(
            DATASET_LABEL_GROUPS["ding"],
            {
                "spike": "hardware",
                "inverter fault": "hardware",
                "low value": "hardware",
                "cloudy": "weather",
                "snowy": "weather",
                "shading": "shading",
            },
        )


if __name__ == "__main__":
    unittest.main()
