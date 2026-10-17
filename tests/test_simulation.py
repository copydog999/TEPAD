import unittest

import numpy as np

from src.simulation.model import (
    TrueParameters,
    estimate_theta,
    generate_stationary_samples,
    inject_mixed,
    inject_shading,
    inject_spike,
    pvwatts_power,
    simulate_normal_day,
)


class SimulationModelTests(unittest.TestCase):
    def test_fault_injections_touch_exact_samples(self):
        base = np.ones(100, dtype=float)
        spike = inject_spike(base, 20, 0.25)
        np.testing.assert_allclose(np.flatnonzero(spike != base), [20])
        self.assertAlmostEqual(spike[20], 1.25)

        shade = inject_shading(base, 30, 10, 0.5)
        np.testing.assert_allclose(np.flatnonzero(shade != base), np.arange(30, 40))
        np.testing.assert_allclose(shade[30:40], 0.5)

        mixed = inject_mixed(base, 10, 0.2, 40, 5, 0.7)
        self.assertAlmostEqual(mixed[10], 1.2)
        np.testing.assert_allclose(mixed[40:45], 0.7)

    def test_simulator_is_reproducible_and_one_minute_ticks(self):
        first = simulate_normal_day(42, n_ticks=720)
        second = simulate_normal_day(42, n_ticks=720)
        np.testing.assert_array_equal(first.ticks, np.arange(720))
        np.testing.assert_allclose(first.irradiance, second.irradiance)
        np.testing.assert_allclose(first.power, second.power)
        self.assertEqual(first.theta_true.alpha, 8.0)
        self.assertTrue(np.all(first.irradiance >= 0))

    def test_pvwatts_power_is_bounded_and_monotone_at_stc(self):
        irradiance = np.asarray([0.0, 250.0, 500.0, 1000.0])
        power = pvwatts_power(irradiance, np.full(4, 25.0), eta=0.82)
        self.assertTrue(np.all(np.diff(power) >= 0))
        self.assertTrue(np.all((power >= 0) & (power <= 1)))

    def test_stationary_model_estimator_tracks_known_parameters(self):
        theta = TrueParameters()
        g, p = generate_stationary_samples(123, 30000, theta)
        estimate = estimate_theta(g, p)
        self.assertLess(abs(estimate[0] - theta.alpha), 0.35)
        self.assertLess(abs(estimate[1] - theta.beta), 0.35)
        self.assertLess(abs(estimate[2] - theta.eta), 0.01)
        self.assertLess(abs(estimate[3] - theta.sigma0), 0.015)
        self.assertLess(abs(estimate[4] - theta.sigma_min), 0.01)


if __name__ == "__main__":
    unittest.main()
