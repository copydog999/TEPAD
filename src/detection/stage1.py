"""Probability-model parameter distance against the normal reference.

Uses the **normal** curves to estimate :math:`(\\mu_0,\\Sigma_0)` and computes

.. math::
    d = \\sqrt{(x-\\mu_0)^{\\top}\\Sigma_0^{-1}(x-\\mu_0)}

This is the first level of :func:`src.detection.cascade.run_two_level`; its
distance is also reported as the TEPAD ranking score.
"""
from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np

from ..config import STAGE1_QUANTILE


def mahalanobis_distance(x: np.ndarray, normal_mask: np.ndarray) -> np.ndarray:
    """Compute the Mahalanobis distance relative to the normal reference distribution.

    Parameters
    ----------
    x : np.ndarray, shape (n_samples, n_features)
        Feature matrix (``nan`` entries are filled with the normal median).
    normal_mask : np.ndarray, shape (n_samples,)
        Marks which samples belong to the normal reference set.

    Returns
    -------
    np.ndarray, shape (n_samples,)
    """
    x = np.asarray(x, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(x, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    xf = np.where(np.isfinite(x), x, med)
    mu = xf[normal_mask].mean(axis=0)
    cov = np.cov(xf[normal_mask], rowvar=False, ddof=1) + 1e-6 * np.eye(xf.shape[1])
    diff = xf - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, np.linalg.pinv(cov, rcond=1e-9), diff))


def stage1_flags(d: np.ndarray, normal_mask: np.ndarray,
                 quantile: float = STAGE1_QUANTILE) -> Tuple[np.ndarray, float]:
    """Flag stage 1 anomalies using an empirical quantile threshold from the normal set.

    Returns
    -------
    (flags, tau) : Tuple[np.ndarray, float]
        Boolean flags and the threshold used.
    """
    tau = float(np.percentile(d[normal_mask], quantile))
    return d > tau, tau


if __name__ == "__main__":  # pragma: no cover - minimal example
    rng = np.random.default_rng(3)
    x = np.vstack([rng.normal(0, 1, size=(200, 5)), rng.normal(3, 1, size=(20, 5))])
    nm = np.r_[np.ones(200, bool), np.zeros(20, bool)]
    dd = mahalanobis_distance(x, nm)
    print("mean d (normal/abnormal):", round(dd[nm].mean(), 2), round(dd[~nm].mean(), 2))
