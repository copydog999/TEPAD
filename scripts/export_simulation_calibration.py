# -*- coding: utf-8 -*-
"""Export the calibration parameters of the simulation dual-window detector so that the
``sim_ui/`` MFC host application can load them.

The calibration procedure is identical to ``src/simulation/experiment.py::run_simulation``:
generate ``SIMULATION_CALIBRATION_DAYS`` normal simulated days (``simulate_normal_day``),
call ``calibrate_dual_window`` to fit the short/long-window Mahalanobis references, the
point-residual baseline and the three-channel 95% quantile thresholds, then serialize
everything to plain JSON (directly parseable on the C++ side).

Usage
-----
    python scripts/export_simulation_calibration.py
    python scripts/export_simulation_calibration.py --days 24 --out sim_ui/calibration/dual_window_calibration.json

The serialized fields correspond exactly to what ``DualWindowDetector`` /
``MahalanobisReference`` need for inference: the EWMA coefficient is
``config.EWMA_ALPHA_DETECTOR`` (0.1), matching the coefficient used when the detector is
constructed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.simulation.model import (  # noqa: E402
    TrueParameters,
    calibrate_dual_window,
    make_dual_window_detector,
    run_detector,
    simulate_normal_day,
)

#: Calibration file location read by default by the MFC host application
DEFAULT_OUT = config.PROJECT_ROOT / "sim_ui" / "calibration" / "dual_window_calibration.json"

FORMAT_VERSION = "tepad-dual-window-calibration-v1"


def _vector(arr: np.ndarray) -> List[float]:
    return [float(x) for x in np.asarray(arr, dtype=float).reshape(-1)]


def _matrix(mat: np.ndarray) -> List[List[float]]:
    return [[float(x) for x in row] for row in np.atleast_2d(np.asarray(mat, dtype=float))]


def _reference_to_dict(ref) -> Dict[str, object]:
    """Serialize the inference fields of MahalanobisReference (center / scale / standardized center / inverse covariance)."""
    return {
        "dimension": int(len(ref.center)),
        "center": _vector(ref.center),
        "scale": _vector(ref.scale),
        "standardized_center": _vector(ref.standardized_center),
        "inverse_covariance": _matrix(ref.inverse_covariance),
        "threshold": float(ref.threshold),
        "n_reference": int(ref.n_reference),
    }


def build_calibration(days: int, seed: int, ticks: int) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Fit the dual-window detector calibration following the simulation experiment workflow,
    returning (JSON structure, raw calibration dictionary)."""
    theta = TrueParameters()
    print("[1/2] generating {} normal calibration days (seed={}, {} ticks/day)...".format(days, seed, ticks))
    normal_days = [
        simulate_normal_day(seed + i, ticks, theta) for i in range(days)
    ]
    print("[2/2] calibrate_dual_window fitting the Mahalanobis references and 95% quantile thresholds...")
    calibration = calibrate_dual_window(normal_days)

    payload = {
        "format": FORMAT_VERSION,
        "detector": {
            "short_window": 10,
            "long_window": 60,
            # The CausalParameterEstimator of DualWindowDetector uses EWMA_ALPHA_DETECTOR
            "ewma_alpha": float(config.EWMA_ALPHA_DETECTOR),
            "z_clip": 12.0,
            "sigma_floor_rate": 0.025,
            "g_floor": 0.1,
        },
        "theta_true": {
            "alpha": theta.alpha, "beta": theta.beta, "eta": theta.eta,
            "sigma0": theta.sigma0, "sigma_min": theta.sigma_min,
        },
        "eta_reference": float(calibration["eta_reference"]),
        "residual_sigma": float(calibration["residual_sigma"]),
        "residual_threshold": float(calibration["residual_threshold"]),
        "tau_short": float(calibration["tau_short"]),
        "tau_long": float(calibration["tau_long"]),
        "threshold_calibration": str(calibration["threshold_calibration"]),
        "n_calibration_days": int(days),
        "calibration_seed": int(seed),
        "ticks_per_day": int(ticks),
        "short_reference": _reference_to_dict(calibration["short_reference"]),
        "long_reference": _reference_to_dict(calibration["long_reference"]),
    }
    return payload, calibration


def write_golden(csv_path: Path, calibration: Dict[str, object], seed: int,
                 ticks: int) -> None:
    """Run a fresh simulated day with the same exported calibration and write the per-tick
    expected values for C++ self-test comparison."""
    day = simulate_normal_day(seed, ticks, TrueParameters())
    g = np.clip(day.irradiance / 1000.0, 0.0, 1.0)
    detector = make_dual_window_detector(calibration)
    scores, alarms, components = run_detector(detector, g, day.power)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="ascii", newline="\n") as fh:
        fh.write("tick,G,P,point_z,short_mahalanobis,long_mahalanobis,score,alarm\n")
        for i in range(len(g)):
            # Emit floats in their shortest round-trippable form (repr) to keep full 64-bit
            # precision, so the C++ side does not accumulate bias from text truncation
            # (EWMA amplifies errors).
            values = [float(g[i]), float(day.power[i]), float(components["point_z"][i]),
                      float(components["short_mahalanobis"][i]),
                      float(components["long_mahalanobis"][i]), float(scores[i])]
            fh.write("{},{},{},{},{},{},{},{}\n".format(
                i, values[0], values[1], values[2], values[3], values[4], values[5],
                int(alarms[i])))
    print("golden reference data: {} ({} rows)".format(csv_path, len(g)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Export the simulation dual-window detector calibration JSON")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output JSON path")
    ap.add_argument("--days", type=int, default=config.SIMULATION_CALIBRATION_DAYS,
                    help="number of calibration days (default config.SIMULATION_CALIBRATION_DAYS)")
    ap.add_argument("--seed", type=int, default=config.SIMULATION_SEED,
                    help="calibration seed (default config.SIMULATION_SEED)")
    ap.add_argument("--ticks", type=int, default=config.SIMULATION_DAY_TICKS,
                    help="ticks per day (default config.SIMULATION_DAY_TICKS)")
    ap.add_argument("--golden", default=None,
                    help="additionally write a per-tick expected-value CSV (for the sim_ui/selftest numeric comparison)")
    args = ap.parse_args()

    payload, calibration = build_calibration(args.days, args.seed, args.ticks)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nCalibration summary")
    print("  eta_reference      = {:.6f}".format(payload["eta_reference"]))
    print("  residual_sigma     = {:.6f}".format(payload["residual_sigma"]))
    print("  residual_threshold = {:.6f}".format(payload["residual_threshold"]))
    print("  tau_short (L=10)   = {:.6f}".format(payload["tau_short"]))
    print("  tau_long  (L=60)   = {:.6f}".format(payload["tau_long"]))
    print("  short-window reference samples = {}".format(payload["short_reference"]["n_reference"]))
    print("  long-window reference samples  = {}".format(payload["long_reference"]["n_reference"]))
    print("written: {}".format(out))

    if args.golden:
        # Run a new simulated day with an offset seed directly from the same un-serialized
        # calibration dictionary, so the thresholds are strictly identical to the JSON and
        # refitting is avoided.
        write_golden(Path(args.golden), calibration, args.seed + 700000, args.ticks)


if __name__ == "__main__":
    main()
