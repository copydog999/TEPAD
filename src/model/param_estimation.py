"""Parameter estimation (two-stage): sliding-window 5 parameters + residual dynamics (AR(1)/AR(2))."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..config import AR_MIN_POINTS, N_WINDOWS, STEP, WINDOW
from .probabilistic import estimate_5params, mu_of_g


def window_starts(n: int, window: int = WINDOW, step: int = STEP,
                  n_win: int = N_WINDOWS) -> List[int]:
    """Compute the sliding-window start indices.

    Rules (matching the experiments):
      * ``n >= window``: ``start = 0, step, 2*step, ...``;
      * ``n <  window``: the window degenerates to the whole curve (``start=0, window=n``);
      * when there are more starts than ``n_win``, ``n_win`` of them are taken at equal spacing.

    Returns
    -------
    List[int]
        Window start indices (length <= n_win).
    """
    if n >= window:
        starts = list(range(0, n - window + 1, step))
    else:
        starts = [0]
    if len(starts) > n_win:
        idx = np.linspace(0, len(starts) - 1, n_win).astype(int)
        starts = [starts[i] for i in idx]
    return starts


def flatten_window_params(g: np.ndarray, p: np.ndarray, window: int = WINDOW,
                          step: int = STEP, n_win: int = N_WINDOWS) -> np.ndarray:
    """Flatten the sliding-window 5 parameters of one daily curve into a fixed-length vector ``(5 * n_win,)``.

    When there are fewer than ``n_win`` windows, the remainder is padded with ``nan``
    (filled with the median later).
    """
    g = np.asarray(g, dtype=float)
    p = np.asarray(p, dtype=float)
    n = len(g)
    w = window if n >= window else n
    starts = window_starts(n, window, step, n_win)
    win = [estimate_5params(g[s:s + w], p[s:s + w]) for s in starts]
    win += [np.full(5, np.nan)] * (n_win - len(win))
    return np.concatenate(win)


def local_residual(g: np.ndarray, p: np.ndarray, window: int = WINDOW,
                   step: int = STEP) -> np.ndarray:
    """Compute the pointwise residual :math:`r_i=p_i-\\min(\\eta_i g_i,1)` with a **local** :math:`\\eta_i`.

    The local efficiency is estimated per sliding window and assigned to every point
    in the window by forward fill.
    """
    g = np.asarray(g, dtype=float)
    p = np.asarray(p, dtype=float)
    n = len(g)
    w = window if n >= window else n
    starts = window_starts(n, window, step, n_win=max(n, 1))
    etas = []
    for s in starts:
        gg = g[s:s + w]
        etas.append(float(np.polyfit(gg, p[s:s + w], 1)[0]) if np.std(gg) >= 1e-12 else 0.0)
    etas = np.asarray(etas, dtype=float)
    idx = np.searchsorted(np.asarray(starts), np.arange(n), side="right") - 1
    idx = np.clip(idx, 0, len(starts) - 1)
    return p - np.minimum(etas[idx] * g, 1.0)


def ar1(r: np.ndarray) -> Tuple[float, float]:
    """AR(1) estimate of :math:`r_t=\\phi r_{t-1}+\\epsilon_t`.

    Returns
    -------
    (phi, sigma_eps) : Tuple[float, float]
        Autoregressive coefficient and residual standard deviation; returns
        ``(nan, nan)`` when there are too few samples or the fit degenerates.
    """
    r = np.asarray(r, dtype=float)
    if len(r) < AR_MIN_POINTS or np.std(r[:-1]) < 1e-12:
        return float("nan"), float("nan")
    x, y = r[:-1], r[1:]
    phi = float(np.dot(x - x.mean(), y - y.mean()) / np.sum((x - x.mean()) ** 2))
    eps = y - phi * x
    return phi, float(np.sqrt(np.mean(eps ** 2)))


def ar2(r: np.ndarray) -> Tuple[float, float, float]:
    """AR(2) estimate of :math:`r_t=\\phi_1 r_{t-1}+\\phi_2 r_{t-2}+\\epsilon_t` (OLS).

    Returns
    -------
    (phi1, phi2, sigma_eps) : Tuple[float, float, float]
    """
    r = np.asarray(r, dtype=float)
    if len(r) < AR_MIN_POINTS + 2:
        return float("nan"), float("nan"), float("nan")
    y = r[2:]
    x1, x2 = r[1:-1], r[:-2]
    xmat = np.column_stack([x1, x2])
    try:
        coef, *_ = np.linalg.lstsq(xmat, y, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan"), float("nan")
    eps = y - xmat @ coef
    return float(coef[0]), float(coef[1]), float(np.sqrt(np.mean(eps ** 2)))


if __name__ == "__main__":  # pragma: no cover - minimal example
    rng = np.random.default_rng(1)
    gg = np.clip(rng.beta(2.0, 3.0, 120), 0, 1)
    pp = np.clip(0.85 * gg + rng.normal(0, 0.02, 120), 0, 1)
    print("starts      =", window_starts(len(gg))[:5], "...")
    print("flat shape  =", flatten_window_params(gg, pp).shape)
    print("ar1         =", ar1(local_residual(gg, pp)))
