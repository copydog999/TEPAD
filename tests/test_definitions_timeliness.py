import unittest

import numpy as np

from src.simulation.definitions import (
    ExternalReferenceDetector,
    ControlCache,
    inject_smooth_shading,
    shading_profile,
)
from src.simulation.model import TrueParameters, estimate_theta, simulate_normal_day
from src.simulation.definitions_experiment import _build_calibration
from src.simulation.timeliness import _inject, run_timeliness_experiment


class ShadingProfileTests(unittest.TestCase):
    def _check_symmetric_trapezoid(self, profile, factor, ramp):
        """Check the geometric properties of the shading profile: mirror symmetry, bounded
        slope, a single valley, and endpoints returning to 1."""
        self.assertAlmostEqual(float(profile[0]), 1.0)
        self.assertAlmostEqual(float(profile[-1]), 1.0)
        self.assertAlmostEqual(float(np.min(profile)), factor)
        # The falling and recovery segments have equal slope magnitude (the old implementation
        # wrote the recovery as ramp-1 steps, which made it steeper than the onset).
        np.testing.assert_allclose(profile, profile[::-1], atol=1e-12)
        slope = (1.0 - factor) / ramp
        steps = np.diff(profile)
        self.assertLessEqual(float(np.max(np.abs(steps))), slope + 1e-12)
        # Single valley: non-decreasing prefix followed by a non-increasing tail, no rebound.
        rise = int(np.argmax(steps > 0.0)) if np.any(steps > 0.0) else len(steps)
        self.assertTrue(np.all(steps[:rise] <= 1e-12))
        self.assertTrue(np.all(steps[rise:] >= -1e-12))

    def test_profile_is_symmetric_and_bounded(self):
        profile = shading_profile(30, 0.5, ramp=5)
        self.assertEqual(len(profile), 30)
        self._check_symmetric_trapezoid(profile, 0.5, 5)
        self.assertTrue(np.all((profile > 0.0) & (profile <= 1.0)))

    def test_profile_stays_symmetric_for_very_short_events(self):
        """The ramp shrinks automatically when it does not fit, but the profile must still be
        symmetric and return to 1."""
        profile = shading_profile(3, 0.4, ramp=5)
        self.assertEqual(len(profile), 3)
        # ramp shrinks to duration // 2 = 1, so the per-step slope is (1 - factor) / 1.
        self._check_symmetric_trapezoid(profile, 0.4, 1)

    def test_definition_one_leaves_irradiance_untouched(self):
        g = np.linspace(0.2, 0.9, 200)
        p = 0.8 * g
        g_pv, p_pv = inject_smooth_shading(g, p, 50, 30, 0.5, sensor_shaded=False)
        g_both, p_both = inject_smooth_shading(g, p, 50, 30, 0.5, sensor_shaded=True)

        np.testing.assert_allclose(g_pv, g)
        self.assertLess(float(p_pv[60:80].mean()), float(p[60:80].mean()))
        self.assertLess(float(g_both[60:80].mean()), float(g[60:80].mean()))
        self.assertLess(float(p_both[60:80].mean()), float(p[60:80].mean()))

    def test_sensor_shaded_definition_keeps_endogenous_ratio(self):
        """Definition #2 preserves P/mu(G), so the endogenous residual stays small."""
        theta = TrueParameters()
        day = simulate_normal_day(11, 720, theta)
        g = np.clip(day.irradiance / 1000.0, 0.0, 1.0)
        eta = 0.82
        g_both, p_both = inject_smooth_shading(g, day.power, 300, 30, 0.5,
                                               sensor_shaded=True)
        window = slice(310, 325)
        endogenous = np.abs(p_both[window] - eta * g_both[window])
        raw = np.abs(p_both[window] - eta * g[window])
        self.assertLess(float(endogenous.mean()), float(raw.mean()))


class ExternalReferenceDetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.theta = TrueParameters()
        cls.calibration = _build_calibration(cls.theta)

    def test_external_channel_can_be_disabled(self):
        day = simulate_normal_day(7, 720, self.theta)
        g = np.clip(day.irradiance / 1000.0, 0.0, 1.0)
        g_inj, p_inj = inject_smooth_shading(g, day.power, 300, 30, 0.5,
                                             sensor_shaded=True)
        scores_full, alarms_full = [], []
        scores_endo, alarms_endo = [], []
        full = ExternalReferenceDetector(self.calibration, use_external=True)
        endo = ExternalReferenceDetector(self.calibration, use_external=False)
        for index in range(len(p_inj)):
            score, alarm, _ = full.step(float(g_inj[index]), float(p_inj[index]),
                                        float(g[index]))
            scores_full.append(score)
            alarms_full.append(alarm)
            score, alarm, _ = endo.step(float(g_inj[index]), float(p_inj[index]),
                                        float(g[index]))
            scores_endo.append(score)
            alarms_endo.append(alarm)
        self.assertGreaterEqual(max(scores_full), max(scores_endo))
        self.assertGreaterEqual(int(np.sum(alarms_full)), int(np.sum(alarms_endo)))

    def test_control_cache_reuses_alarms_per_channel_mode(self):
        cache = ControlCache(self.calibration, self.theta)
        first = cache.alarms(3, 200, 30, use_external=True)
        second = cache.alarms(3, 200, 30, use_external=True)
        other = cache.alarms(3, 200, 30, use_external=False)
        self.assertIs(first, second)
        self.assertEqual(len(first), len(other))


class EstimatorFastPathTests(unittest.TestCase):
    def test_fast_estimator_matches_expected_values(self):
        """The optimized estimator must keep the documented moment/OLS semantics."""
        theta = TrueParameters()
        rng = np.random.default_rng(2)
        g = rng.beta(theta.alpha, theta.beta, size=20000)
        mu = np.minimum(theta.eta * g, 1.0)
        p = mu + rng.normal(0.0, theta.sigma0 * mu + theta.sigma_min)
        estimate = estimate_theta(g, p)
        self.assertLess(abs(estimate[2] - theta.eta), 0.02)
        self.assertLess(abs(estimate[3] - theta.sigma0), 0.01)
        self.assertLess(abs(estimate[4] - theta.sigma_min), 0.01)

    def test_estimator_returns_nan_for_short_windows(self):
        self.assertTrue(np.all(np.isnan(estimate_theta(np.arange(3.0), np.arange(3.0)))))


class TimelinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.theta = TrueParameters()
        cls.calibration = _build_calibration(cls.theta)

    def test_injections_are_smooth_and_type_specific(self):
        day = simulate_normal_day(21, 720, self.theta)
        g = np.clip(day.irradiance / 1000.0, 0.0, 1.0)
        for kind in ("spike", "shading", "cloudy", "low_value"):
            g_inj, p_inj, duration = _inject(kind, day, 300, "medium", self.theta)
            self.assertEqual(len(g_inj), len(g))
            self.assertEqual(len(p_inj), len(p_inj))
            self.assertGreaterEqual(duration, 1)
            if kind == "cloudy":
                self.assertFalse(np.allclose(g_inj, g))
            if kind == "low_value":
                np.testing.assert_allclose(g_inj, g)

    def test_timeliness_reports_dtm_and_mtd(self):
        result = run_timeliness_experiment(self.calibration, self.theta, repeats=6)
        self.assertEqual(len(result["summary"]), 4)
        for entry in result["summary"]:
            self.assertGreaterEqual(entry["detection_rate"], 0.0)
            self.assertLessEqual(entry["detection_rate"], 1.0)
            for tau in result["thresholds"]:
                value = entry["dtm_{}".format(tau)]
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)
        # A transient spike must be detected no later than any slow temporal fault.
        by_type = {e["anomaly_type"]: e for e in result["summary"]}
        if by_type["spike"]["mtd_ticks"] is not None and \
                by_type["low_value"]["mtd_ticks"] is not None:
            self.assertLessEqual(by_type["spike"]["mtd_ticks"],
                                 by_type["low_value"]["mtd_ticks"] + 1e-9)


if __name__ == "__main__":
    unittest.main()
