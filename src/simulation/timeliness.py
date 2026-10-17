"""E5: timeliness evaluation -- detection-delay statistics of two anomaly timing profiles in simulation.

* **MTD** (Mean Time to Detect): mean number of ticks from anomaly onset to the first alarm.
* **DTM(τ)** (Detection-within-Tolerance): fraction of anomalies detected within a delay of τ ticks.

Four anomaly types
------------------
============  ==========================================================
``spike``     single-tick transient spike (point-level channel)
``shading``   smooth shading: ``P`` drops while ``G`` stays unchanged (endogenously detectable, definition #1)
``cloudy``    cloud passage: ``G`` and ``P`` drift together, small amplitude, medium duration
``low_value`` persistently low efficiency: ``G`` normal but ``P`` below ``μ(G)`` for a long time
============  ==========================================================

Each anomaly is injected ``N=100`` times per (intensity × time-of-day) cell; the detector
advances causally tick by tick and the first alarm time is recorded. Detection delay is
always measured from the **injection start tick**.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import config
from .definitions import (
    ControlCache,
    ExternalReferenceDetector,
    RESPONSE_TAIL_TICKS,
    _first_detection,
    _control_false_alarm,
    inject_smooth_shading,
    run_detector_with_reference,
)
from .model import TrueParameters, inject_spike, simulate_normal_day

DAY_TICKS = config.SIMULATION_DAY_TICKS
SEED = config.SIMULATION_SEED

#: Delay-threshold grid (ticks)
DTM_THRESHOLDS: Tuple[int, ...] = (1, 5, 10, 20, 50)

#: Anomaly types (this order is the table row order)
ANOMALY_TYPES: Tuple[str, ...] = ("spike", "shading", "cloudy", "low_value")

#: Intensity levels (low / medium / high)
INTENSITY_LEVELS: Tuple[str, ...] = ("low", "medium", "high")

#: Time-of-day levels (morning / noon / afternoon)
TIME_OF_DAY_LEVELS: Tuple[Tuple[str, float], ...] = (
    ("morning", 0.25), ("noon", 0.50), ("afternoon", 0.75),
)

#: Duration of each anomaly type (ticks)
DURATIONS: Dict[str, int] = {"spike": 1, "shading": 30, "cloudy": 20, "low_value": 120}

#: Parameters of each anomaly type at low / medium / high intensity (larger multiplier means stronger)
SPIKE_SIGMA = {"low": 2.0, "medium": 3.5, "high": 5.0}
SHADING_FACTOR = {"low": 0.70, "medium": 0.50, "high": 0.30}
CLOUDY_FACTOR = {"low": 0.80, "medium": 0.60, "high": 0.40}
LOW_VALUE_FACTOR = {"low": 0.90, "medium": 0.75, "high": 0.55}


def inject_cloudy(g: np.ndarray, p: np.ndarray, start: int, duration: int,
                  factor: float) -> Tuple[np.ndarray, np.ndarray]:
    """Cloud passage: ``G`` and ``P`` ramp up/down together smoothly (irradiance and power move in the same direction)."""
    g_out, p_out = inject_smooth_shading(g, p, start, duration, factor,
                                         sensor_shaded=True)
    return g_out, p_out


def inject_low_value(g: np.ndarray, p: np.ndarray, start: int, duration: int,
                     factor: float) -> Tuple[np.ndarray, np.ndarray]:
    """Persistently low efficiency: ``G`` unchanged, ``P`` attenuated by ``factor`` for a long time."""
    g_out, p_out = inject_smooth_shading(g, p, start, duration, factor,
                                         sensor_shaded=False)
    return g_out, p_out


def _inject(kind: str, day, start: int, level: str, theta: TrueParameters
            ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Inject according to the anomaly type, returning ``(g, p, duration)``."""
    g_true = np.clip(day.irradiance / 1000.0, 0.0, 1.0)
    if kind == "spike":
        tick = int(np.clip(start, 0, DAY_TICKS - 1))
        amplitude = SPIKE_SIGMA[level] * day.sigma[tick]
        return g_true, inject_spike(day.power, tick, amplitude), 1
    if kind == "shading":
        duration = DURATIONS["shading"]
        start = int(np.clip(start, 0, DAY_TICKS - duration - 1))
        g_out, p_out = inject_smooth_shading(g_true, day.power, start, duration,
                                             SHADING_FACTOR[level],
                                             sensor_shaded=False)
        return g_out, p_out, duration
    if kind == "cloudy":
        duration = DURATIONS["cloudy"]
        start = int(np.clip(start, 0, DAY_TICKS - duration - 1))
        g_out, p_out = inject_cloudy(g_true, day.power, start, duration,
                                     CLOUDY_FACTOR[level])
        return g_out, p_out, duration
    if kind == "low_value":
        duration = DURATIONS["low_value"]
        start = int(np.clip(start, 0, DAY_TICKS - duration - 1))
        g_out, p_out = inject_low_value(g_true, day.power, start, duration,
                                        LOW_VALUE_FACTOR[level])
        return g_out, p_out, duration
    raise ValueError("unknown anomaly type: {}".format(kind))


def run_timeliness_experiment(calibration: Dict[str, object], theta: TrueParameters,
                              *, repeats: int = 100, logger=None) -> Dict[str, object]:
    """Run the timeliness experiment over the four anomaly types, returning per-trial records and per-cell statistics."""
    rng = np.random.default_rng(SEED + 7717)
    seeds = rng.integers(0, 2 ** 31 - 1, size=int(repeats))
    cache = ControlCache(calibration, theta)
    trials: List[Dict[str, object]] = []
    cells: List[Dict[str, object]] = []

    for kind in ANOMALY_TYPES:
        for level in INTENSITY_LEVELS:
            for tod_name, tod_fraction in TIME_OF_DAY_LEVELS:
                detected: List[bool] = []
                delays: List[float] = []
                control_alarms: List[bool] = []
                for repeat, seed in enumerate(seeds):
                    day = cache.day(int(seed))
                    g_true = np.clip(day.irradiance / 1000.0, 0.0, 1.0)
                    anchor = int(round(tod_fraction * (DAY_TICKS - 1)))
                    g_inj, p_inj, duration = _inject(kind, day, anchor, level, theta)
                    start = int(np.clip(anchor, 0, DAY_TICKS - duration - 1))
                    detector = ExternalReferenceDetector(calibration)
                    _, alarms, _ = run_detector_with_reference(
                        detector, g_inj, p_inj, g_true)
                    hit = _first_detection(alarms, start, start + duration)
                    detected.append(hit is not None)
                    if hit is not None:
                        delays.append(float(hit - start))
                    control_alarms.append(_control_false_alarm(
                        cache.alarms(int(seed), start, duration), start,
                        start + duration))
                    trials.append({
                        "anomaly_type": kind, "intensity": level,
                        "time_of_day": tod_name, "repeat": repeat,
                        "seed": int(seed), "inject_tick": start,
                        "duration_ticks": int(duration),
                        "detected": bool(hit is not None),
                        "delay_ticks": None if hit is None else int(hit - start),
                    })
                delays_array = np.asarray(delays, dtype=float)
                cell = {
                    "anomaly_type": kind, "intensity": level,
                    "time_of_day": tod_name, "duration_ticks": int(duration),
                    "n_trials": int(repeats),
                    "detection_rate": float(np.mean(detected)) if detected else 0.0,
                    "mtd_ticks": float(delays_array.mean())
                    if delays_array.size else None,
                    "delay_sd_ticks": float(delays_array.std(ddof=1))
                    if delays_array.size > 1 else None,
                    "delay_median_ticks": float(np.median(delays_array))
                    if delays_array.size else None,
                    "delay_p90_ticks": float(np.percentile(delays_array, 90))
                    if delays_array.size else None,
                    "control_false_alarm_rate": float(np.mean(control_alarms))
                    if control_alarms else 0.0,
                }
                for tau in DTM_THRESHOLDS:
                    cell["dtm_{}".format(tau)] = (
                        float(np.mean(delays_array <= tau)) if delays_array.size else 0.0
                    )
                cells.append(cell)
                if logger:
                    logger.info("E5 %s/%s/%s: detection %.3f, MTD %s, DTM(1)=%.3f",
                                kind, level, tod_name, cell["detection_rate"],
                                None if cell["mtd_ticks"] is None
                                else round(cell["mtd_ticks"], 2),
                                cell["dtm_1"])

    summaries: List[Dict[str, object]] = []
    for kind in ANOMALY_TYPES:
        entries = [c for c in cells if c["anomaly_type"] == kind]
        entry_trials = [t for t in trials if t["anomaly_type"] == kind]
        delays = np.asarray([t["delay_ticks"] for t in entry_trials
                             if t["delay_ticks"] is not None], dtype=float)
        summary = {
            "anomaly_type": kind,
            "duration_ticks": DURATIONS[kind],
            "n_trials": len(entry_trials),
            "detection_rate": float(np.mean([t["detected"] for t in entry_trials])),
            "mtd_ticks": float(delays.mean()) if delays.size else None,
            "delay_sd_ticks": float(delays.std(ddof=1)) if delays.size > 1 else None,
            "delay_median_ticks": float(np.median(delays)) if delays.size else None,
            "delay_p90_ticks": float(np.percentile(delays, 90)) if delays.size else None,
            "control_false_alarm_rate": float(np.mean(
                [c["control_false_alarm_rate"] for c in entries])),
        }
        for tau in DTM_THRESHOLDS:
            summary["dtm_{}".format(tau)] = (
                float(np.mean(delays <= tau)) if delays.size else 0.0
            )
        summaries.append(summary)

    return {
        "thresholds": list(DTM_THRESHOLDS),
        "anomaly_types": list(ANOMALY_TYPES),
        "intensity_levels": list(INTENSITY_LEVELS),
        "time_of_day_levels": {name: fraction
                               for name, fraction in TIME_OF_DAY_LEVELS},
        "durations": dict(DURATIONS),
        "repeats": int(repeats),
        "cells": cells,
        "summary": summaries,
        "dtm_curve": {
            str(tau): [s["dtm_{}".format(tau)] for s in summaries]
            for tau in DTM_THRESHOLDS
        },
        "trials": trials,
    }
