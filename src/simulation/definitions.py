"""E3: refined simulation injection -- the two shading definitions and the external reference channel.

Two shading definitions
-----------------------
* **Definition #1 (PV shaded only)**: the irradiance sensor is healthy (``G`` unchanged) and
  only the PV output drops. The endogenous probabilistic model can already detect the
  decoupling of ``P`` from ``\\mu(G)``; this is the **easy** case.
* **Definition #2 (PV and sensor shaded together)**: ``G`` and ``P`` drop together, so they
  still satisfy ``P \\approx \\mu(G)`` and the endogenous model cannot tell; an
  **irradiance/power external reference** :math:`P^{\\mathrm{ref}}` unaffected by the shading
  is required. This is the **hard** case.

Injection profile
-----------------
The intensity is injected as a **smooth ramp** (over ``ramp`` ticks), not a step drop;
durations are 10/30/60 ticks; time of day is morning / noon / afternoon; every
combination is repeated ``N=100`` times.

External reference
------------------
:math:`G^{\\mathrm{ref}}` comes from **a healthy twin sensor / clear-sky model** and is
independent of the shaded ``G``. In simulation it is the pre-injection true irradiance
series, representing an independent, unshaded reference source.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import config
from ..model.mahalanobis import MahalanobisReference
from .model import (
    CausalParameterEstimator,
    DualWindowDetector,
    TrueParameters,
    calibrate_dual_window,
    estimate_theta,
    run_detector,
    simulate_normal_day,
)

DAY_TICKS = config.SIMULATION_DAY_TICKS
SEED = config.SIMULATION_SEED

#: Internal keys and display names of the shading definitions
SHADING_DEFINITIONS: Tuple[Tuple[str, str], ...] = (
    ("pv_only", "Definition #1: PV only (G unchanged)"),
    ("pv_and_sensor", "Definition #2: PV + sensor (G and P drop)"),
)

#: Intensity levels: shading factor (PV output multiplier); larger means milder
INTENSITY_LEVELS: Tuple[Tuple[str, float], ...] = (
    ("low", 0.70), ("medium", 0.50), ("high", 0.30),
)

#: Duration levels (ticks)
DURATION_LEVELS: Tuple[int, ...] = (10, 30, 60)

#: Time-of-day levels (fraction of the day)
TIME_OF_DAY_LEVELS: Tuple[Tuple[str, float], ...] = (
    ("morning", 0.25), ("noon", 0.50), ("afternoon", 0.75),
)

#: Noise sweep (constant measurement-noise standard deviation on the normalized power)
NOISE_LEVELS: Tuple[float, ...] = (0.01, 0.05, 0.1, 0.2)

#: Smooth ramp length (ticks)
RAMP_TICKS = 5

#: Alarm-search tail (ticks): response window still allowed after the interval or fault ends
RESPONSE_TAIL_TICKS = 30


def shading_profile(duration: int, factor: float, ramp: int = RAMP_TICKS) -> np.ndarray:
    """Generate a smooth shading multiplier profile: ``1 → factor → 1``.

    Parameters
    ----------
    duration : int
        Number of ticks the event lasts.
    factor : float
        Deepest multiplier (``0 < factor < 1``).
    ramp : int
        One-sided transition length; automatically shrunk to at most ``duration // 2``.

    Notes
    -----
    The trapezoidal profile (or triangular for very short events) is mirror-symmetric:
    the descent and the recovery use the same slope magnitude
    ``delta = (1 - factor) / ramp``, so ``profile == profile[::-1]`` and the recovery is
    never steeper than the onset. Short events (``2 * ramp >= duration``) have no plateau
    and degenerate to a triangle.
    """
    if duration < 1:
        raise ValueError("duration must be >= 1")
    if not 0.0 < factor < 1.0:
        raise ValueError("shading factor must be in (0, 1)")
    ramp = int(max(1, min(ramp, max(1, duration // 2))))
    shape = np.ones(int(duration), dtype=float)
    if 2 * ramp >= duration:
        positions = np.array([0.0, (duration - 1) / 2.0, float(duration - 1)])
        shape = np.interp(np.arange(duration, dtype=float), positions,
                          [1.0, float(factor), 1.0])
        return shape
    descent = np.linspace(1.0, float(factor), ramp, endpoint=False)
    shape[:ramp] = descent
    shape[ramp:duration - ramp] = float(factor)
    # The recovery must mirror the descent. Writing ``linspace(factor, 1.0, ramp, endpoint=True)``
    # would give only ``ramp - 1`` steps (slope magnitude ``(1-factor)/(ramp-1)``), making the
    # recovery steeper than the onset; this breaks the symmetry and biases the timeliness
    # statistics (detection delays on the descent/recovery directions) toward the injection shape.
    shape[duration - ramp:] = descent[::-1]
    return shape


def inject_smooth_shading(g: np.ndarray, p: np.ndarray, start: int, duration: int,
                          factor: float, *, sensor_shaded: bool,
                          ramp: int = RAMP_TICKS) -> Tuple[np.ndarray, np.ndarray]:
    """Inject shading smoothly according to one of the two definitions, returning ``(g_injected, p_injected)``.

    ``sensor_shaded=False`` corresponds to definition #1 (only ``P`` drops); ``True`` to
    definition #2 (``G`` and ``P`` drop together).
    """
    if start < 0 or start + duration > len(p):
        raise ValueError("shading interval outside the simulated stream")
    profile = shading_profile(duration, factor, ramp)
    g_out = np.asarray(g, dtype=float).copy()
    p_out = np.asarray(p, dtype=float).copy()
    p_out[start:start + duration] *= profile
    if sensor_shaded:
        g_out[start:start + duration] *= profile
    return g_out, p_out


class ExternalReferenceDetector:
    """Streaming detector combining the endogenous dual-window Mahalanobis channels with an external-reference residual channel.

    Three channels:
      * ``point``: endogenous pointwise residual ``|p - eta*g| / sigma``;
      * ``short``/``long``: endogenous sliding-window parameter Mahalanobis (reusing
        :func:`src.simulation.model.calibrate_dual_window`);
      * ``external``: external-reference residual ``|p - eta*g_ref| / sigma_ref``,
        which provides separability only when the G sensor itself is shaded (definition #2).
    """

    def __init__(self, calibration: Dict[str, object], use_external: bool = True):
        self.inner = DualWindowDetector(
            calibration["short_reference"], calibration["long_reference"],
            calibration["eta_reference"], calibration["residual_sigma"],
            calibration["residual_threshold"], calibration["tau_short"],
            calibration["tau_long"],
        )
        self.eta_reference = float(calibration["eta_reference"])
        self.external_sigma = float(calibration["external_sigma"])
        self.external_threshold = float(calibration["external_threshold"])
        self.use_external = bool(use_external)

    def step(self, g: float, p: float, g_ref: float) -> Tuple[float, bool, Dict[str, float]]:
        inner_score, inner_alarm, components = self.inner.step(g, p)
        sigma = max(self.external_sigma, 0.025 * max(float(g_ref), 0.1))
        external_z = abs(float(p) - self.eta_reference * float(g_ref)) / sigma
        if self.use_external:
            alarm = bool(inner_alarm or external_z > self.external_threshold)
            score = max(inner_score, external_z / max(self.external_threshold, 1e-12))
        else:
            alarm = bool(inner_alarm)
            score = inner_score
        components = dict(components, external_z=float(external_z))
        return score, alarm, components


def _block_maxima(values: np.ndarray, block: int = 60) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n_blocks = len(values) // block
    if n_blocks == 0:
        return values.copy()
    return np.asarray([np.max(values[i * block:(i + 1) * block])
                       for i in range(n_blocks)])


def calibrate_external_reference(normal_days, calibration: Dict[str, object],
                                 quantile: float = 95.0) -> Dict[str, object]:
    """Calibrate the external-reference residual threshold on normal days (empirical quantile of 60-tick block maxima)."""
    eta_reference = float(calibration["eta_reference"])
    residuals: List[np.ndarray] = []
    z_scores: List[np.ndarray] = []
    for day in normal_days:
        g = np.clip(day.irradiance / 1000.0, 0.0, 1.0)
        valid = g > 0.1
        if not np.any(valid):
            continue
        residuals.append(day.power[valid] - eta_reference * g[valid])
    sigma = float(np.std(np.concatenate(residuals), ddof=1)) if residuals else 1e-8
    sigma = max(sigma, 1e-8)
    for day in normal_days:
        g = np.clip(day.irradiance / 1000.0, 0.0, 1.0)
        z = np.abs(day.power - eta_reference * g) / np.maximum(
            sigma, 0.025 * np.maximum(g, 0.1))
        z_scores.append(z)
    threshold = float(np.percentile(_block_maxima(np.concatenate(z_scores)), quantile))
    return dict(calibration, external_sigma=sigma, external_threshold=threshold,
                external_calibration=("{}th percentile of normal 60-tick block "
                                      "maxima of |p - eta*g_ref| / sigma".format(quantile)))


def run_detector_with_reference(detector: ExternalReferenceDetector, g: np.ndarray,
                                p: np.ndarray, g_ref: np.ndarray
                                ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Advance the detector causally tick by tick, returning ``(scores, alarms, components)``."""
    n = len(p)
    scores = np.zeros(n, dtype=float)
    alarms = np.zeros(n, dtype=bool)
    keys = ("point_z", "short_mahalanobis", "long_mahalanobis", "external_z")
    components = {key: np.zeros(n, dtype=float) for key in keys}
    for index in range(n):
        scores[index], alarms[index], current = detector.step(
            float(g[index]), float(p[index]), float(g_ref[index]))
        for key in keys:
            components[key][index] = float(current.get(key, 0.0))
    return scores, alarms, components


def _first_detection(alarms: np.ndarray, start: int, stop: int) -> Optional[int]:
    """First alarm tick within the event window (including the ``RESPONSE_TAIL_TICKS`` tail)."""
    stop = min(max(stop, start + 1), len(alarms))
    tail = min(stop + RESPONSE_TAIL_TICKS, len(alarms))
    selected = np.flatnonzero(alarms[start:tail])
    return int(start + selected[0]) if len(selected) else None


def _control_false_alarm(alarms: np.ndarray, start: int, stop: int) -> bool:
    stop = min(max(stop, start + 1), len(alarms))
    tail = min(stop + RESPONSE_TAIL_TICKS, len(alarms))
    return bool(np.any(alarms[start:tail]))


class ControlCache:
    """Cache the alarm series of clean days over the same window.

    The control (no-injection) detection does not depend on the fault definition or
    intensity, only on ``(seed, start, duration)``, so it can be shared by many cells
    within one sweep and the detector need not be re-run. The endogenous channel and
    the external-reference channel are cached separately.
    """

    def __init__(self, calibration: Dict[str, object], theta: TrueParameters,
                 noise_sigma: Optional[float] = None):
        self.calibration = calibration
        self.theta = theta
        self.noise_sigma = noise_sigma
        self._days: Dict[int, object] = {}
        self._alarms: Dict[Tuple[int, int, int, bool], np.ndarray] = {}

    def day(self, seed: int):
        if seed not in self._days:
            self._days[seed] = simulate_normal_day(
                seed, DAY_TICKS, self.theta, noise_sigma=self.noise_sigma)
        return self._days[seed]

    def alarms(self, seed: int, start: int, duration: int,
               use_external: bool = True) -> np.ndarray:
        key = (int(seed), int(start), int(duration), bool(use_external))
        if key not in self._alarms:
            day = self.day(seed)
            g = np.clip(day.irradiance / 1000.0, 0.0, 1.0)
            detector = ExternalReferenceDetector(self.calibration,
                                                 use_external=use_external)
            _, alarms, _ = run_detector_with_reference(detector, g, day.power, g)
            self._alarms[key] = alarms
        return self._alarms[key]


def run_definition_experiment(calibration: Dict[str, object], theta: TrueParameters,
                              *, repeats: int = 100, noise_sigma: Optional[float] = None,
                              ramp: int = RAMP_TICKS,
                              time_levels: Optional[Sequence[Tuple[str, float]]] = None,
                              duration_levels: Optional[Sequence[int]] = None,
                              intensity_levels: Optional[Sequence[Tuple[str, float]]] = None,
                              use_external: bool = True,
                              cache: Optional[ControlCache] = None,
                              logger=None) -> Dict[str, object]:
    """Run the shading-injection experiment for definition #1 / definition #2 (intensity × duration × time of day).

    Returns
    -------
    dict
        ``cells`` (detection rate and delay of each definition×intensity×duration×time-of-day cell),
        ``by_definition``, ``by_intensity``, ``by_duration``, ``by_time_of_day``
        and ``trials`` (per-trial records).
    """
    log = logger
    rng = np.random.default_rng(SEED)
    seeds = rng.integers(0, 2 ** 31 - 1, size=int(repeats))
    cache = cache or ControlCache(calibration, theta, noise_sigma=noise_sigma)
    time_levels = list(time_levels or TIME_OF_DAY_LEVELS)
    duration_levels = list(duration_levels or DURATION_LEVELS)
    intensity_levels = list(intensity_levels or INTENSITY_LEVELS)
    cells: List[Dict[str, object]] = []
    trials: List[Dict[str, object]] = []

    for definition, definition_label in SHADING_DEFINITIONS:
        sensor_shaded = definition == "pv_and_sensor"
        for intensity_name, factor in intensity_levels:
            for duration in duration_levels:
                for tod_name, tod_fraction in time_levels:
                    detected: List[bool] = []
                    delays: List[float] = []
                    control_alarms: List[bool] = []
                    for repeat, seed in enumerate(seeds):
                        day = cache.day(int(seed))
                        g_true = np.clip(day.irradiance / 1000.0, 0.0, 1.0)
                        start = int(round(tod_fraction * (DAY_TICKS - 1)))
                        start = int(np.clip(start, 0, DAY_TICKS - duration - 1))
                        g_inj, p_inj = inject_smooth_shading(
                            g_true, day.power, start, duration, factor,
                            sensor_shaded=sensor_shaded, ramp=ramp)
                        detector = ExternalReferenceDetector(
                            calibration, use_external=use_external)
                        _, alarms, _ = run_detector_with_reference(
                            detector, g_inj, p_inj, g_true)
                        hit = _first_detection(alarms, start, start + duration)
                        detected.append(hit is not None)
                        if hit is not None:
                            delays.append(float(hit - start))
                        control_alarms.append(_control_false_alarm(
                            cache.alarms(int(seed), start, duration,
                                         use_external=use_external), start,
                            start + duration))
                        trials.append({
                            "definition": definition, "intensity": intensity_name,
                            "factor": factor, "duration_ticks": duration,
                            "time_of_day": tod_name, "repeat": repeat,
                            "seed": int(seed), "inject_tick": start,
                            "detected": bool(hit is not None),
                            "detect_tick": hit,
                            "delay_ticks": None if hit is None else int(hit - start),
                            "noise_sigma": noise_sigma,
                            "use_external": bool(use_external),
                        })
                    delays_array = np.asarray(delays, dtype=float)
                    cells.append({
                        "definition": definition, "definition_label": definition_label,
                        "intensity": intensity_name, "factor": float(factor),
                        "duration_ticks": int(duration), "time_of_day": tod_name,
                        "tod_fraction": float(tod_fraction),
                        "n_trials": int(repeats),
                        "detection_rate": float(np.mean(detected)) if detected else 0.0,
                        "control_false_alarm_rate": float(np.mean(control_alarms))
                        if control_alarms else 0.0,
                        "mtd_ticks": float(delays_array.mean()) if delays_array.size else None,
                        "delay_median_ticks": float(np.median(delays_array))
                        if delays_array.size else None,
                        "delay_p90_ticks": float(np.percentile(delays_array, 90))
                        if delays_array.size else None,
                        "noise_sigma": noise_sigma,
                        "use_external": bool(use_external),
                    })
                    if log:
                        log.info("E3 %s | %s | %d ticks | %s: detection %.3f, MTD %s",
                                 definition, intensity_name, duration, tod_name,
                                 cells[-1]["detection_rate"],
                                 None if delays_array.size == 0
                                 else round(float(delays_array.mean()), 2))

    def _aggregate(key: str) -> List[Dict[str, object]]:
        groups: Dict[object, List[Dict[str, object]]] = {}
        for cell in cells:
            groups.setdefault(cell[key], []).append(cell)
        out = []
        for value, entries in groups.items():
            out.append({
                key: value,
                "detection_rate": float(np.mean([e["detection_rate"] for e in entries])),
                "control_false_alarm_rate": float(np.mean(
                    [e["control_false_alarm_rate"] for e in entries])),
                "mtd_ticks": float(np.mean([e["mtd_ticks"] for e in entries
                                            if e["mtd_ticks"] is not None]))
                if any(e["mtd_ticks"] is not None for e in entries) else None,
                "n_cells": len(entries),
            })
        return out

    return {
        "definitions": [name for name, _ in SHADING_DEFINITIONS],
        "intensity_levels": {name: factor for name, factor in INTENSITY_LEVELS},
        "duration_levels": list(DURATION_LEVELS),
        "time_of_day_levels": {name: fraction for name, fraction in TIME_OF_DAY_LEVELS},
        "ramp_ticks": int(ramp),
        "repeats": int(repeats),
        "noise_sigma": noise_sigma,
        "cells": cells,
        "by_definition": _aggregate("definition"),
        "by_intensity": _aggregate("intensity"),
        "by_duration": _aggregate("duration_ticks"),
        "by_time_of_day": _aggregate("time_of_day"),
        "trials": trials,
    }


def run_noise_sweep(calibration: Dict[str, object], theta: TrueParameters, *,
                    repeats: int = 100, logger=None) -> Dict[str, object]:
    """Sweep the measurement noise ``sigma`` for both definitions and observe the detection rate and MTD.

    To bound the computational cost, the noise sweep is fixed at **noon**, **medium
    duration (30 ticks)** and **low intensity** (shading factor 0.70, i.e. the hardest
    setting): each definition is repeated ``repeats`` times at
    ``sigma ∈ {0.01, 0.05, 0.1, 0.2}``. Low intensity is chosen so that noise becomes the
    limiting factor; otherwise the detection rate would saturate at 1 for every noise level.
    """
    rows: List[Dict[str, object]] = []
    middle_time = (TIME_OF_DAY_LEVELS[1],)
    middle_duration = (DURATION_LEVELS[1],)
    low_intensity = (INTENSITY_LEVELS[0],)
    for noise_sigma in NOISE_LEVELS:
        cache = ControlCache(calibration, theta, noise_sigma=float(noise_sigma))
        result = run_definition_experiment(
            calibration, theta, repeats=repeats, noise_sigma=float(noise_sigma),
            time_levels=middle_time, duration_levels=middle_duration,
            intensity_levels=low_intensity, cache=cache, logger=None)
        for entry in result["by_definition"]:
            rows.append({
                "noise_sigma": float(noise_sigma),
                "definition": entry["definition"],
                "detection_rate": entry["detection_rate"],
                "mtd_ticks": entry["mtd_ticks"],
                "control_false_alarm_rate": entry["control_false_alarm_rate"],
            })
        if logger:
            logger.info("E3 noise sigma=%.2f: %s", noise_sigma,
                        {r["definition"]: round(r["detection_rate"], 3)
                         for r in rows[-2:]})
    return {"noise_levels": list(NOISE_LEVELS), "rows": rows, "repeats": int(repeats),
            "cells": "noon time-of-day, low intensity (factor 0.70), medium duration "
                     "(30 ticks)"}
