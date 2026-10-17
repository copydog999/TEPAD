import unittest

from src.evaluation.cross_dataset import run_dataset


class PaperResultTests(unittest.TestCase):
    EXPECTED = {
        "ding": (0.908, 0.991, 0.985, 0.997),
        "ds226": (1.000, 0.941, 0.889, 1.000),
        "lareunion": (0.536, 0.708, 0.548, 1.000),
    }

    def test_main_metrics_match_paper(self):
        for dataset, expected in self.EXPECTED.items():
            with self.subTest(dataset=dataset):
                result = run_dataset(dataset)
                actual = (
                    result["stage1_auc"],
                    result["cascade_f1"],
                    result["cascade_precision"],
                    result["cascade_recall"],
                )
                for value, target in zip(actual, expected):
                    self.assertAlmostEqual(value, target, delta=0.001)


if __name__ == "__main__":
    unittest.main()
