"""Minute-tick PV simulation, exact fault injection, and dual-window detector."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .. import config
from ..model.mahalanobis import MahalanobisReference

#: ``sqrt(pi/2)``: converts a mean absolute deviation into a Gaussian standard deviation; precomputed to avoid recomputation per call.
_HALF_PI_SQRT = float(np.sqrt(np.pi / 2.0))


@dataclass(frozen=True)
class TrueParameters:
    alpha: float = 8.0
    beta: float = 2.0
    eta: float = 0.82
    sigma0: float = 0.025
    sigma_min: float = 0.008


@dataclass
class SimulatedDay:
    ticks: np.ndarray
    clear_sky_irradiance: np.ndarray
    irradiance: np.ndarray
    temperature_c: np.ndarray
    power_normal: np.ndarray
    power: np.ndarray
    sigma: np.ndarray
    clear_sky_factor: np.ndarray
    theta_true: TrueParameters


def pvwatts_power(irradiance_wm2: np.ndarray, temperature_c: np.ndarray,
                  eta: float, gamma: float = -0.004) -> np.ndarray:
    """PVWatts-style normalized AC power (kW/kWdc), with temperature derating."""
    irradiance = np.asarray(irradiance_wm2, dtype=float)
    temperature = np.asarray(temperature_c, dtype=float)
    cell_temperature = temperature + ((45.0 - 20.0) / 800.0) * irradiance
    pdc = eta * (irradiance / 1000.0) * (1.0 + gamma * (cell_temperature - 25.0))
    return np.clip(0.96 * pdc, 0.0, 1.0)


def simulate_normal_day(seed: int, n_ticks: int = 720,
                        theta: TrueParameters = TrueParameters(),
                        noise_scale: float = 1.0,
                        temperature_base: float = 25.0,
                        noise_sigma: Optional[float] = None) -> SimulatedDay:
    """Generate a 12-hour clear-sky envelope with Beta-distributed cloud opacity.

    Parameters
    ----------
    noise_sigma : float, optional
        If given, replaces the model's own heteroscedastic
        :math:`\\sigma(g)=\\sigma_0\\mu(g)+\\sigma_{\\min}` with a **constant**
        measurement-noise standard deviation. This is used for the noise-level sweep
        (E3), so that the meaning of ``sigma`` is uniquely determined and does not
        vary with the operating point.
    """
    if n_ticks < 100:
        raise ValueError("n_ticks must be at least 100")
    if noise_scale < 0:
        raise ValueError("noise_scale must be non-negative")
    if noise_sigma is not None and noise_sigma < 0:
        raise ValueError("noise_sigma must be non-negative")
    rng = np.random.default_rng(seed)
    ticks = np.arange(n_ticks, dtype=int)
    phase = np.linspace(0.0, np.pi, n_ticks)
    clear_sky = 1000.0 * np.sin(phase)

    cloud = rng.beta(theta.alpha, theta.beta, size=n_ticks)
    # A weak AR component models the temporal persistence of cloud passages while
    # preserving a strictly positive and bounded irradiance attenuation factor.
    persistent = np.empty(n_ticks, dtype=float)
    persistent[0] = cloud[0]
    for i in range(1, n_ticks):
        persistent[i] = np.clip(0.75 * persistent[i - 1] + 0.25 * cloud[i], 0.0, 1.0)
    irradiance = clear_sky * persistent
    temperature = temperature_base + 2.0 * np.sin(phase - 0.4)
    p_mean = pvwatts_power(irradiance, temperature, theta.eta)
    g_norm = np.clip(irradiance / 1000.0, 0.0, 1.0)
    if noise_sigma is not None:
        p_sigma = np.full(n_ticks, float(noise_sigma))
    else:
        p_sigma = noise_scale * (theta.sigma0 * p_mean + theta.sigma_min)
    noise = rng.normal(0.0, p_sigma, size=n_ticks)
    power = np.clip(p_mean + noise, 0.0, 1.0)
    return SimulatedDay(
        ticks=ticks,
        clear_sky_irradiance=clear_sky,
        irradiance=irradiance,
        temperature_c=temperature,
        power_normal=p_mean,
        power=power,
        sigma=np.maximum(p_sigma, 1e-9),
        clear_sky_factor=persistent,
        theta_true=theta,
    )


def generate_stationary_samples(seed: int, n_samples: int,
                                theta: TrueParameters,
                                noise_sigma: Optional[float] = None
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate stationary samples from the exact TEPAD Beta/conditional-Normal model."""
    rng = np.random.default_rng(seed)
    g = rng.beta(theta.alpha, theta.beta, size=n_samples)
    mu = np.minimum(theta.eta * g, 1.0)
    if noise_sigma is None:
        sigma = theta.sigma0 * mu + theta.sigma_min
    else:
        sigma = np.full(n_samples, float(noise_sigma))
    p = mu + rng.normal(0.0, sigma, size=n_samples)
    return g, p


def estimate_theta(g: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Estimate [alpha,beta,eta,sigma0,sigma_min] with moment/OLS corrections.

    All sums are accumulated with BLAS-level reductions and the clamps are plain
    scalar operations, so the streaming detector can call this once per tick
    without the NumPy per-call overhead dominating the runtime.
    """
    g = np.asarray(g, dtype=float)
    p = np.asarray(p, dtype=float)
    if not (np.isfinite(g).all() and np.isfinite(p).all()):
        valid = np.isfinite(g) & np.isfinite(p)
        g, p = g[valid], p[valid]
    n = len(g)
    if n < 4:
        return np.full(5, np.nan)

    sum_g = float(g.sum())
    sum_p = float(p.sum())
    sum_gg = float(g @ g)
    sum_gp = float(g @ p)
    mean_g = sum_g / n
    mean_p = sum_p / n
    mean_g = 1e-5 if mean_g < 1e-5 else (1.0 - 1e-5 if mean_g > 1.0 - 1e-5 else mean_g)
    var_g = (sum_gg - n * mean_g * mean_g) / max(n - 1, 1)
    concentration = mean_g * (1.0 - mean_g) / max(var_g, 1e-8) - 1.0
    concentration = 1e-4 if concentration < 1e-4 else (1e5 if concentration > 1e5 else concentration)
    alpha = mean_g * concentration
    beta = (1.0 - mean_g) * concentration
    alpha = 1e-4 if alpha < 1e-4 else (1e5 if alpha > 1e5 else alpha)
    beta = 1e-4 if beta < 1e-4 else (1e5 if beta > 1e5 else beta)

    var_centered_g = sum_gg - n * mean_g * mean_g
    eta = (sum_gp - n * mean_g * mean_p) / var_centered_g if var_centered_g > 1e-12 else 0.0
    eta = 0.0 if eta < 0.0 else (2.0 if eta > 2.0 else eta)
    mu = np.minimum(eta * g, 1.0)
    abs_residual = np.abs(p - mu) * _HALF_PI_SQRT

    sum_mu = float(mu.sum())
    sum_ar = float(abs_residual.sum())
    mean_mu = sum_mu / n
    mean_ar = sum_ar / n
    var_mu = float(mu @ mu) - n * mean_mu * mean_mu
    if var_mu > 1e-12:
        sigma0 = (float(mu @ abs_residual) - n * mean_mu * mean_ar) / var_mu
        sigma_min = mean_ar - sigma0 * mean_mu
    else:
        sigma0 = 0.0
        sigma_min = mean_ar
    sigma0 = 0.0 if sigma0 < 0.0 else (2.0 if sigma0 > 2.0 else sigma0)
    sigma_min = 1e-6 if sigma_min < 1e-6 else (2.0 if sigma_min > 2.0 else sigma_min)
    return np.asarray([alpha, beta, eta, sigma0, sigma_min], dtype=float)


class CausalParameterEstimator:
    """L-point parameter estimation with the specified alpha=0.95 EWMA."""

    def __init__(self, window: int, alpha: float = config.EWMA_ALPHA_CALIBRATION):
        if window < 4 or not 0.0 < alpha <= 1.0:
            raise ValueError("window must be >=4 and EWMA alpha must be in (0,1]")
        self.window = int(window)
        self.alpha = float(alpha)
        self.g = []
        self.p = []
        self.theta: Optional[np.ndarray] = None

    def step(self, g: float, p: float) -> Optional[np.ndarray]:
        self.g.append(float(g))
        self.p.append(float(p))
        if len(self.g) > self.window:
            del self.g[0]
            del self.p[0]
        if len(self.g) < self.window:
            return None
        estimate = estimate_theta(np.asarray(self.g), np.asarray(self.p))
        if not np.all(np.isfinite(estimate)):
            return None
        if self.theta is None:
            self.theta = estimate
        else:
            self.theta = self.alpha * estimate + (1.0 - self.alpha) * self.theta
        return self.theta.copy()


def inject_spike(power: np.ndarray, tick: int, amplitude: float) -> np.ndarray:
    """Copy power and add one exact spike sample."""
    if not 0 <= tick < len(power):
        raise ValueError("spike tick outside the simulated stream")
    result = np.asarray(power, dtype=float).copy()
    result[tick] += float(amplitude)
    return result


def inject_shading(power: np.ndarray, start: int, duration: int,
                   factor: float) -> np.ndarray:
    """Copy power and attenuate the exact half-open interval [start,start+duration)."""
    if duration < 1 or start < 0 or start + duration > len(power):
        raise ValueError("invalid shading interval")
    if not 0.0 < factor < 1.0:
        raise ValueError("shading factor must be in (0,1)")
    result = np.asarray(power, dtype=float).copy()
    result[start:start + duration] *= factor
    return result


def inject_mixed(power: np.ndarray, spike_tick: int, spike_amplitude: float,
                 shade_start: int, shade_duration: int,
                 shade_factor: float) -> np.ndarray:
    """Inject a single spike followed by a shading interval."""
    if spike_tick >= shade_start:
        raise ValueError("mixed injection requires spike_tick < shade_start")
    return inject_shading(
        inject_spike(power, spike_tick, spike_amplitude),
        shade_start, shade_duration, shade_factor,
    )


def _window_features(g: np.ndarray, p: np.ndarray, length: int) -> Optional[np.ndarray]:
    if len(g) < length:
        return None
    params = estimate_theta(g[-length:], p[-length:])
    if not np.all(np.isfinite(params)):
        return None
    features = params.copy()
    features[[0, 1, 3, 4]] = np.log(np.maximum(features[[0, 1, 3, 4]], 1e-8))
    return features


def _features_from_params(params: np.ndarray) -> np.ndarray:
    features = np.asarray(params, dtype=float).copy()
    if not np.all(np.isfinite(features)):
        return features
    features[[0, 1, 3, 4]] = np.log(np.maximum(features[[0, 1, 3, 4]], 1e-8))
    return features


class DualWindowDetector:
    """Short/long TEPAD Mahalanobis detector with a point-residual transient channel."""

    def __init__(self, short_reference: MahalanobisReference,
                 long_reference: MahalanobisReference, eta_reference: float,
                 residual_sigma: float, residual_threshold: float,
                 tau_short: float, tau_long: float):
        self.short_reference = short_reference
        self.long_reference = long_reference
        self.eta_reference = float(eta_reference)
        self.residual_sigma = max(float(residual_sigma), 1e-8)
        self.residual_threshold = float(residual_threshold)
        self.tau_short = float(tau_short)
        self.tau_long = float(tau_long)
        self.g10 = []
        self.p10 = []
        self.g60 = []
        self.p60 = []
        self.est10 = CausalParameterEstimator(10, config.EWMA_ALPHA_DETECTOR)
        self.est60 = CausalParameterEstimator(60, config.EWMA_ALPHA_DETECTOR)

    def step(self, g: float, p: float) -> Tuple[float, bool, Dict[str, float]]:
        g, p = float(g), float(p)
        sigma_point = max(self.residual_sigma, 0.025 * max(g, 0.1))
        point_score = abs(p - self.eta_reference * g) / sigma_point
        self.g10.append(g)
        self.p10.append(p)
        self.g60.append(g)
        self.p60.append(p)
        if len(self.g10) > 10:
            del self.g10[0]
            del self.p10[0]
        if len(self.g60) > 60:
            del self.g60[0]
            del self.p60[0]
        theta10 = self.est10.step(g, p)
        theta60 = self.est60.step(g, p)
        short_score = 0.0
        long_score = 0.0
        if theta10 is not None:
            short_score = self.short_reference.score(_features_from_params(theta10))
        if theta60 is not None:
            long_score = self.long_reference.score(_features_from_params(theta60))
        alarm = (
            point_score > self.residual_threshold
            or short_score > self.tau_short
            or long_score > self.tau_long
        )
        score = max(
            point_score / max(self.residual_threshold, 1e-12),
            short_score / max(self.tau_short, 1e-12),
            long_score / max(self.tau_long, 1e-12),
        )
        return score, bool(alarm), {
            "point_z": float(point_score),
            "short_mahalanobis": float(short_score),
            "long_mahalanobis": float(long_score),
        }


def calibrate_dual_window(normal_days) -> Dict[str, object]:
    """Fit empirical 95% thresholds exclusively on simulated normal days."""
    features10 = []
    features60 = []
    etas = []
    residuals = []
    for day in normal_days:
        g = np.clip(day.irradiance / 1000.0, 0.0, 1.0)
        p = day.power
        # Calibration feature extraction is causal and identical to the test step.
        estimator10 = CausalParameterEstimator(10, config.EWMA_ALPHA_CALIBRATION)
        estimator60 = CausalParameterEstimator(60, config.EWMA_ALPHA_CALIBRATION)
        for i in range(len(g)):
            theta10 = estimator10.step(g[i], p[i])
            theta60 = estimator60.step(g[i], p[i])
            if theta10 is not None:
                features10.append(_features_from_params(theta10))
            if theta60 is not None:
                features60.append(_features_from_params(theta60))
        theta = estimate_theta(g, p)
        etas.append(theta[2])
    x10 = np.vstack(features10)
    x60 = np.vstack(features60)
    eta_reference = float(np.median(etas))
    for day in normal_days:
        g = np.clip(day.irradiance / 1000.0, 0.0, 1.0)
        valid = g > 0.1
        residuals.extend((day.power[valid] - eta_reference * g[valid]).tolist())
    residual_sigma = max(float(np.std(residuals, ddof=1)), 1e-8)
    point_scores = []
    short_scores = []
    long_scores = []
    ref10 = MahalanobisReference.fit(x10)
    ref60 = MahalanobisReference.fit(x60)
    for day in normal_days:
        g = np.clip(day.irradiance / 1000.0, 0.0, 1.0)
        valid = g > 0.1
        z_sigma = np.maximum(residual_sigma, 0.025 * np.maximum(g[valid], 0.1))
        day_point_score = np.zeros(len(g), dtype=float)
        day_point_score[valid] = (
            np.abs(day.power[valid] - eta_reference * g[valid]) / z_sigma
        )
        point_scores.extend(day_point_score.tolist())
        estimator10 = CausalParameterEstimator(10, config.EWMA_ALPHA_CALIBRATION)
        estimator60 = CausalParameterEstimator(60, config.EWMA_ALPHA_CALIBRATION)
        for i in range(len(g)):
            theta10 = estimator10.step(g[i], day.power[i])
            theta60 = estimator60.step(g[i], day.power[i])
            short_scores.append(
                ref10.score(_features_from_params(theta10)) if theta10 is not None else 0.0
            )
            long_scores.append(
                ref60.score(_features_from_params(theta60)) if theta60 is not None else 0.0
            )

    def block_maxima(values):
        values = np.asarray(values, dtype=float)
        n_blocks = len(values) // 60
        if n_blocks == 0:
            raise ValueError("At least one 60-tick calibration block is required")
        return np.asarray([
            np.max(values[i * 60:(i + 1) * 60]) for i in range(n_blocks)
        ])

    # Set thresholds on 60-tick normal block maxima, controlling the false-alarm
    # probability over the full response horizon rather than at each individual tick.
    threshold_point = float(np.percentile(block_maxima(point_scores), 95.0))
    threshold_short = float(np.percentile(block_maxima(short_scores), 95.0))
    threshold_long = float(np.percentile(block_maxima(long_scores), 95.0))
    return {
        "short_reference": ref10,
        "long_reference": ref60,
        "eta_reference": eta_reference,
        "residual_sigma": residual_sigma,
        "residual_threshold": threshold_point,
        "tau_short": threshold_short,
        "tau_long": threshold_long,
        "n_short_reference": int(len(x10)),
        "n_long_reference": int(len(x60)),
        "threshold_calibration": "95th percentile of non-overlapping normal 60-tick block maxima",
    }


def make_dual_window_detector(calibration: Dict[str, object]) -> DualWindowDetector:
    return DualWindowDetector(
        calibration["short_reference"],
        calibration["long_reference"],
        calibration["eta_reference"],
        calibration["residual_sigma"],
        calibration["residual_threshold"],
        calibration["tau_short"],
        calibration["tau_long"],
    )


def run_detector(detector: DualWindowDetector, g: np.ndarray, p: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    scores = np.zeros(len(g), dtype=float)
    alarms = np.zeros(len(g), dtype=bool)
    components = {
        "point_z": np.zeros(len(g), dtype=float),
        "short_mahalanobis": np.zeros(len(g), dtype=float),
        "long_mahalanobis": np.zeros(len(g), dtype=float),
    }
    for index, (irradiance, power) in enumerate(zip(g, p)):
        scores[index], alarms[index], current = detector.step(irradiance, power)
        for key in components:
            components[key][index] = current[key]
    return scores, alarms, components
