"""Endogenous probabilistic model (the physical model of TEPAD stage 1).

.. math::

    G      &\\sim \\mathrm{Beta}(\\alpha,\\beta)               \\\\
    P\\,|\\,G{=}g &\\sim \\mathcal N\\big(\\mu(g),\\ \\sigma^2(g)\\big) \\\\
    \\mu(g) &= \\min(\\eta g,\\ 1)                              \\\\
    \\sigma(g) &= \\sigma_0\\,\\mu(g) + \\sigma_{\\min}

where :math:`\\eta` is the conversion efficiency (G→P slope), and :math:`\\sigma_0`, :math:`\\sigma_{\\min}`
describe the linear growth of the residual standard deviation with the operating point
and the minimum noise floor.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from ..config import VARIANCE_FLOOR


def mu_of_g(g: np.ndarray, eta: float) -> np.ndarray:
    """Theoretical power :math:`\\mu(g)=\\min(\\eta g,1)`.

    Parameters
    ----------
    g : np.ndarray
        Normalized irradiance.
    eta : float
        Conversion efficiency (slope).
    """
    return np.minimum(eta * np.asarray(g, dtype=float), 1.0)


def sigma_of_mu(mu: np.ndarray, sigma0: float, sigma_min: float) -> np.ndarray:
    """Residual standard deviation :math:`\\sigma(\\mu)=\\sigma_0\\mu+\\sigma_{\\min}`."""
    return sigma0 * np.asarray(mu, dtype=float) + sigma_min


def beta_moments(g: np.ndarray, floor: float = VARIANCE_FLOOR) -> Tuple[float, float]:
    """Estimate the Beta parameters :math:`(\\alpha,\\beta)` by the method of moments, applying the variance floor ``floor``.

    Returns
    -------
    (alpha, beta) : Tuple[float, float]
        Returns ``(nan, nan)`` when estimation fails (mean out of range or variance too large).
    """
    g = np.asarray(g, dtype=float)
    m_g = float(np.mean(g))
    v_g = max(float(np.var(g)), floor)
    if m_g <= 0.0 or m_g >= 1.0:
        return float("nan"), float("nan")
    c = m_g * (1.0 - m_g) / v_g - 1.0
    if c <= 0.0:
        return float("nan"), float("nan")
    return m_g * c, (1.0 - m_g) * c


def estimate_5params(g: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Estimate the 5 parameters ``[alpha, beta, eta, sigma0, sigma_min]`` of a single window.

    Estimation order (matching the paper):
      1. ``alpha, beta``: Beta method of moments + variance floor;
      2. ``eta``: OLS slope (P on G);
      3. ``sigma0, sigma_min``: OLS of ``|r|`` on :math:`\\mu`, where :math:`r=p-\\mu(g)`.

    Parameters
    ----------
    g, p : np.ndarray
        Normalized irradiance and normalized power within the same window.

    Returns
    -------
    np.ndarray, shape (5,)
        ``[alpha, beta, eta, sigma0, sigma_min]``; entries that cannot be estimated are ``nan``.
    """
    g = np.asarray(g, dtype=float)
    p = np.asarray(p, dtype=float)
    a, b = beta_moments(g)
    eta = float(np.polyfit(g, p, 1)[0]) if np.std(g) >= 1e-12 else 0.0
    mu = mu_of_g(g, eta)
    r = p - mu
    if np.std(mu) >= 1e-9 and len(mu) >= 4:
        s0, smin = np.polyfit(mu, np.abs(r), 1)
        s0, smin = float(s0), float(smin)
    else:
        s0, smin = float("nan"), float("nan")
    return np.array([a, b, eta, s0, smin], dtype=float)


if __name__ == "__main__":  # pragma: no cover - minimal example
    rng = np.random.default_rng(0)
    gg = np.clip(rng.beta(2.0, 3.0, size=200), 0, 1)
    pp = np.clip(0.85 * gg + rng.normal(0, 0.02, size=200), 0, 1)
    print("alpha,beta =", beta_moments(gg))
    print("5 params   =", estimate_5params(gg, pp))
