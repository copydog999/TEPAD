"""Residual (stage 2) features: local dip shape and correlation."""
from __future__ import annotations

from typing import Dict

import numpy as np

from ..config import RESID_TAU_K


def longest_negative_run(flags: np.ndarray) -> float:
    """Length of the longest consecutive run of ``True``."""
    best = cur = 0
    for v in np.asarray(flags, dtype=bool):
        cur = cur + 1 if v else 0
        best = b = max(best, cur)
    return float(best)


def residual_tau(residuals_normal: np.ndarray, k: float = RESID_TAU_K) -> float:
    """Residual depth threshold :math:`\\tau = k\\cdot\\sigma(\\text{normal residuals})`."""
    arr = np.concatenate([np.asarray(r, dtype=float) for r in residuals_normal])
    return float(k * np.std(arr))


def residual_features(res: np.ndarray, tau: float) -> Dict[str, float]:
    """Compute the stage 2 shape features from a residual series.

    Parameters
    ----------
    res : np.ndarray
        Residual :math:`r_t=p_t-p^{\\mathrm{ref}}_t`.
    tau : float
        Depth threshold (usually ``2 * std(normal residuals)``).

    Returns
    -------
    dict
        ``r_min`` (deepest dip), ``delta_r_min`` (steepest drop),
        ``f_neg`` (fraction of deep negatives), ``T_neg`` (longest deep-negative run).
    """
    res = np.asarray(res, dtype=float)
    neg = res < -tau
    return {
        "r_min": float(np.nanmin(res)),
        "delta_r_min": float(np.nanmin(np.diff(res))) if res.size > 1 else float("nan"),
        "f_neg": float(np.mean(neg)),
        "T_neg": longest_negative_run(neg),
    }


def residual_corr(p: np.ndarray, p_ref: np.ndarray) -> float:
    """Correlation coefficient between measured and reference power (returns 0 when the variance degenerates)."""
    p = np.asarray(p, dtype=float)
    p_ref = np.asarray(p_ref, dtype=float)
    if np.std(p) > 1e-9 and np.std(p_ref) > 1e-9:
        return float(np.corrcoef(p, p_ref)[0, 1])
    return 0.0


if __name__ == "__main__":  # pragma: no cover - minimal example
    rng = np.random.default_rng(2)
    r = rng.normal(0, 0.01, 100)
    r[40:44] -= 0.3
    tau = residual_tau([rng.normal(0, 0.01, 500)])
    print("tau =", round(tau, 5), "features =", residual_features(r, tau))
