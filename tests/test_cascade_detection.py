import unittest

import numpy as np

from src.detection.cascade import run_two_level
from src.preprocess.daily_split import DailyCurve


def _curves():
    rng = np.random.default_rng(19)
    curves = []
    for index in range(32):
        irradiance = np.clip(
            np.linspace(0.05, 0.95, 80) + rng.normal(0, 0.025, 80), 0, 1
        )
        power = np.clip(0.82 * irradiance + rng.normal(0, 0.012, 80), 0, 1)
        abnormal = index >= 24
        if abnormal:
            power[30:45] *= 0.45
        curves.append(DailyCurve(
            dataset="synthetic",
            site="site",
            date=str(index),
            label=int(abnormal),
            cls="abnormal" if abnormal else "normal",
            g=irradiance,
            p=power,
            pref=0.82 * irradiance,
            n_obs=len(irradiance),
        ))
    return curves


class CascadeDetectionTests(unittest.TestCase):
    def test_cascade_is_or_of_parameter_and_residual_decisions(self):
        result = run_two_level(_curves(), has_label=True)

        self.assertEqual(result.d.shape, (32,))
        self.assertEqual(result.s1.shape, (32,))
        self.assertEqual(result.s2.shape, (32,))
        np.testing.assert_array_equal(result.casc, result.s1 | result.s2)
        self.assertIn("r_min", result.feats)
        self.assertIn("tau1_rmin", result.taus)

    def test_fit_mask_excludes_test_curves_from_parameter_reference_fit(self):
        curves = _curves()
        fit_mask = np.arange(len(curves)) < 24
        first = run_two_level(curves, has_label=True, fit_mask=fit_mask)

        changed_curves = list(curves)
        changed = changed_curves[25]
        changed_curves[25] = DailyCurve(
            dataset=changed.dataset, site=changed.site, date=changed.date,
            label=changed.label, cls=changed.cls, g=changed.g,
            p=np.zeros_like(changed.p), pref=changed.pref, n_obs=changed.n_obs,
        )
        second = run_two_level(changed_curves, has_label=True, fit_mask=fit_mask)

        self.assertEqual(first.tau1, second.tau1)
        np.testing.assert_allclose(first.d[:24], second.d[:24])

    def test_invalid_reference_mask_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "normal reference curve"):
            run_two_level(_curves(), has_label=True,
                          fit_mask=np.zeros(32, dtype=bool))


if __name__ == "__main__":
    unittest.main()
