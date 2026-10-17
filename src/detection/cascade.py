"""Cascade the probability-model detector with external-reference residual checks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..model.param_estimation import flatten_window_params
from ..model.reference import resolve_reference
from ..model.residual import residual_corr, residual_features, residual_tau
from ..preprocess.daily_split import DailyCurve
from .stage1 import mahalanobis_distance, stage1_flags
from .stage2 import stage2_flags


@dataclass
class DetectionResult:
    """Scores, stage decisions, and thresholds for a set of daily curves."""

    y: np.ndarray
    d: np.ndarray
    s1: np.ndarray
    s2: np.ndarray
    casc: np.ndarray
    feats: Dict[str, np.ndarray]
    taus: Dict[str, float]
    tau1: float
    pref_source: str
    n_curves: int
    has_label: bool


def run_two_level(curves: List[DailyCurve], has_label: bool,
                  fit_mask: Optional[np.ndarray] = None) -> DetectionResult:
    """Apply the window-parameter detector, residual rules, and OR cascade.

    ``fit_mask`` restricts fitting and threshold estimation to training curves.
    Labels are used only to identify normal reference curves when ``has_label``
    is true.
    """
    n = len(curves)
    if n == 0:
        raise ValueError("At least one curve is required")
    y = np.array([curve.label for curve in curves], dtype=int)
    normal = (y == 0) if has_label else np.ones(n, dtype=bool)
    if fit_mask is not None:
        fit_mask = np.asarray(fit_mask, dtype=bool)
        if fit_mask.shape != (n,):
            raise ValueError("fit_mask must have one entry per curve")
        normal &= fit_mask
    if not np.any(normal):
        raise ValueError("At least one normal reference curve is required")

    parameter_features = np.stack(
        [flatten_window_params(curve.g, curve.p) for curve in curves]
    )
    distances = mahalanobis_distance(parameter_features, normal)
    parameter_flags, tau1 = stage1_flags(distances, normal)

    irradiance = np.concatenate([curve.g for curve in curves])
    power = np.concatenate([curve.p for curve in curves])
    eta = float(np.polyfit(irradiance, power, 1)[0]) if np.std(irradiance) > 1e-12 else 0.0

    residuals: List[np.ndarray] = []
    correlations: List[float] = []
    sources: List[str] = []
    for curve in curves:
        reference, source = resolve_reference(
            curve.g, eta, curve.pref,
            source_hint=curve.extra.get("pref_source_hint"),
        )
        sources.append(source)
        residuals.append(curve.p - reference)
        correlations.append(residual_corr(curve.p, reference))

    reference_residuals = [residuals[i] for i in range(n) if normal[i]]
    tau_residual = residual_tau(reference_residuals or residuals)
    features: Dict[str, np.ndarray] = {
        name: np.zeros(n, dtype=float)
        for name in ("r_min", "delta_r_min", "f_neg", "T_neg", "corr")
    }
    for i, residual in enumerate(residuals):
        values = residual_features(residual, tau_residual)
        for name in ("r_min", "delta_r_min", "f_neg", "T_neg"):
            features[name][i] = values[name]
    features["corr"] = np.asarray(correlations, dtype=float)

    residual_flags, thresholds = stage2_flags(features, normal)
    cascade_flags = parameter_flags | residual_flags
    pref_source = max(set(sources), key=sources.count) if sources else "physical"
    return DetectionResult(
        y=y,
        d=distances,
        s1=parameter_flags,
        s2=residual_flags,
        casc=cascade_flags,
        feats=features,
        taus=thresholds,
        tau1=tau1,
        pref_source=pref_source,
        n_curves=n,
        has_label=has_label,
    )


if __name__ == "__main__":  # pragma: no cover - minimal smoke test
    from ..preprocess.daily_split import DailyCurve as DC

    rng = np.random.default_rng(5)
    curves = []
    for i in range(60):
        irradiance = np.clip(rng.beta(2, 3, 40), 0, 1)
        abnormal = i >= 50
        power = np.clip(0.85 * irradiance + rng.normal(0, 0.02, 40), 0, 1)
        if abnormal:
            power[20:23] *= 0.1
        curves.append(DC(
            "demo", "s", str(i), int(abnormal),
            "abnormal" if abnormal else "normal",
            irradiance, power, 0.85 * irradiance, 40,
        ))
    result = run_two_level(curves, has_label=True)
    print("cascade recall on last 10:", round(float(result.casc[-10:].mean()), 3))
