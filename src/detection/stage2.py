"""Legacy residual-threshold diagnostic; not used for the final joint decision.

Criterion
---------
.. math::
    J = (r_{\\min}<\\tau_1)\\;|\\;(\\Delta r_{\\min}<\\tau_2)\\;|\\;
        (f_{\\mathrm{neg}}>\\tau_3)\\;|\\;(\\mathrm{corr}<\\tau_5)

Thresholds :math:`\\tau_1,\\tau_2,\\tau_5` are the ``STAGE2_Q_LOW`` quantile of the
normal set, and :math:`\\tau_3` is its ``STAGE2_Q_HIGH`` quantile.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ..config import STAGE2_Q_HIGH, STAGE2_Q_LOW

FEATURE_NAMES = ["r_min", "delta_r_min", "f_neg", "T_neg", "corr"]


def feature_matrix(feats: Dict[str, np.ndarray]) -> np.ndarray:
    """Stack the feature dictionary into a matrix (column order = :data:`FEATURE_NAMES`)."""
    return np.column_stack([np.asarray(feats[k], dtype=float) for k in FEATURE_NAMES])


def stage2_flags(feats: Dict[str, np.ndarray],
                 normal_mask: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute the stage 2 criterion J.

    Parameters
    ----------
    feats : dict
        Must contain at least ``r_min``, ``delta_r_min``, ``f_neg``, and ``corr``.
    normal_mask : np.ndarray
        Mask of the normal reference set.

    Returns
    -------
    (flags, taus) : Tuple[np.ndarray, Dict[str, float]]
        Boolean flags and the thresholds used.
    """
    r_min = np.asarray(feats["r_min"], dtype=float)
    d_rmin = np.asarray(feats["delta_r_min"], dtype=float)
    f_neg = np.asarray(feats["f_neg"], dtype=float)
    corr = np.asarray(feats["corr"], dtype=float)

    tau1 = float(np.nanpercentile(r_min[normal_mask], STAGE2_Q_LOW))
    tau2 = float(np.nanpercentile(d_rmin[normal_mask], STAGE2_Q_LOW))
    tau3 = float(np.nanpercentile(f_neg[normal_mask], STAGE2_Q_HIGH))
    tau5 = float(np.nanpercentile(corr[normal_mask], STAGE2_Q_LOW))

    flags = (r_min < tau1) | (d_rmin < tau2) | (f_neg > tau3) | (corr < tau5)
    return flags, {"tau1_rmin": tau1, "tau2_drmin": tau2, "tau3_fneg": tau3, "tau5_corr": tau5}


if __name__ == "__main__":  # pragma: no cover - minimal example
    rng = np.random.default_rng(4)
    n = 300
    feats = {"r_min": rng.normal(0, .01, n), "delta_r_min": rng.normal(0, .01, n),
             "f_neg": rng.random(n) * .05, "corr": 0.99 - rng.random(n) * .01}
    feats["r_min"][-10:] -= 0.5
    nm = np.r_[np.ones(n - 10, bool), np.zeros(10, bool)]
    fl, taus = stage2_flags(feats, nm)
    print("flags on last 10:", int(fl[-10:].sum()), "| taus:", {k: round(v, 4) for k, v in taus.items()})
