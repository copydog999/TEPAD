"""The two input representations for the comparison experiments.

* :func:`curve_features` -- a hand-crafted statistical feature vector per daily curve (used by the classical baselines);
* :func:`curve_sequence` -- a fixed-length resampled ``(g, p)`` two-channel sequence (used by the deep baselines).

Both use **only the measured irradiance and measured power**, never ``modeled_power``.
"""
from __future__ import annotations

import warnings
from typing import List, Sequence

import numpy as np
from scipy import stats

from ..config import BASELINE_FEATURE_BINS
from ..preprocess.daily_split import DailyCurve

_STAT_FEATURES = [
    "log_n_obs", "g_mean", "g_std", "g_max", "p_mean", "p_std", "p_max", "p_min",
    "p_skew", "p_kurt", "corr_gp", "lin_slope", "lin_intercept", "lin_rmse",
    "auc_p", "peak_pos", "g_at_peak", "ratio_pg", "dn_deriv_max", "low_frac",
]


def feature_names(n_bins: int = BASELINE_FEATURE_BINS) -> List[str]:
    """Statistical feature column names (length = ``20 + n_bins``)."""
    return list(_STAT_FEATURES) + ["p_bin_%02d" % i for i in range(n_bins)]


def resample(x: np.ndarray, n: int) -> np.ndarray:
    """Linearly interpolate a variable-length series onto ``n`` equally spaced samples (endpoints aligned)."""
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        return np.zeros(n, dtype=float)
    if x.size == 1:
        return np.repeat(float(x[0]), n)
    return np.interp(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, x.size), x)


def _finite(v: float, default: float = 0.0) -> float:
    """Collapse ``nan`` / ``inf`` into a finite value."""
    v = float(v)
    return v if np.isfinite(v) else default


def _moment(fn, x: np.ndarray) -> float:
    """Compute skewness/kurtosis; near-constant series trigger scipy precision warnings, silenced here."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(fn(x))


def curve_features(curve: DailyCurve, n_bins: int = BASELINE_FEATURE_BINS) -> np.ndarray:
    """Statistical feature vector of one daily curve (shape ``(20 + n_bins,)``)."""
    g = np.asarray(curve.g, dtype=float)
    p = np.asarray(curve.p, dtype=float)
    n = int(len(g))
    if n == 0:
        return np.zeros(len(feature_names(n_bins)), dtype=float)

    slope = intercept = lin_rmse = 0.0
    if n >= 3 and np.std(g) > 1e-9:
        slope, intercept = (float(v) for v in np.polyfit(g, p, 1))
        lin_rmse = float(np.sqrt(np.mean((p - (slope * g + intercept)) ** 2)))
    corr = float(np.corrcoef(g, p)[0, 1]) if (np.std(g) > 1e-9 and np.std(p) > 1e-9) else 0.0
    diff = np.diff(p)
    span = float(max(n - 1, 1))

    feats = [
        np.log1p(n),
        g.mean(), g.std(), g.max(),
        p.mean(), p.std(), p.max(), p.min(),
        _finite(_moment(stats.skew, p)), _finite(_moment(stats.kurtosis, p)),
        _finite(corr), slope, intercept, lin_rmse,
        float(np.trapz(p, dx=1.0 / span)),
        float(np.argmax(p)) / span,
        float(g[int(np.argmax(p))]),
        float(p.mean() / (g.mean() + 1e-6)),
        float(-np.min(diff)) if diff.size else 0.0,
        float(np.mean(p < 0.15)),
    ]
    feats += list(resample(p, n_bins))
    return np.nan_to_num(np.asarray(feats, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def feature_matrix(curves: Sequence[DailyCurve],
                   n_bins: int = BASELINE_FEATURE_BINS) -> np.ndarray:
    """Turn a list of curves into an ``(n_curves, n_features)`` feature matrix."""
    if not curves:
        return np.zeros((0, len(feature_names(n_bins))), dtype=float)
    return np.vstack([curve_features(c, n_bins) for c in curves])


def curve_sequence(curve: DailyCurve, length: int, n_ch: int = 2) -> np.ndarray:
    """Fixed-length two-channel sequence of one daily curve, ``(2, length)``: channel 0 = ``g``, channel 1 = ``p``."""
    seq = np.vstack([resample(curve.g, length), resample(curve.p, length)])
    return seq[:n_ch]


def sequence_matrix(curves: Sequence[DailyCurve], length: int) -> np.ndarray:
    """Turn a list of curves into an ``(n_curves, 2, length)`` sequence tensor."""
    if not curves:
        return np.zeros((0, 2, length), dtype=float)
    return np.stack([curve_sequence(c, length) for c in curves])


if __name__ == "__main__":  # pragma: no cover - minimal example
    from .data import load_ding_split
    tr, te = load_ding_split()
    print("feature dim :", feature_matrix(tr[:5]).shape)
    print("sequence dim:", sequence_matrix(tr[:5], 96).shape)
