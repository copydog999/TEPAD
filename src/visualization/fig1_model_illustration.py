# -*- coding: utf-8 -*-
"""Figure 1: illustration of the endogenous probabilistic model (mu(g), sigma(g) and the scatter)."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from ..config import SEED
from .style import COLORS, legend_below, new_fig, save


def build(results: List[dict], out_dir: Path) -> List[Path]:
    """Generate ``fig1_model_illustration.{pdf,png}``."""
    rng = np.random.default_rng(SEED)
    g = np.linspace(0.02, 1.0, 200)
    eta, s0, smin = 0.85, 0.03, 0.005
    mu = np.minimum(eta * g, 1.0)
    sig = s0 * mu + smin

    fig, ax = new_fig(double=False, height=2.6)
    gs = rng.beta(2.0, 2.5, 400)
    ps = np.minimum(eta * gs, 1.0) + rng.normal(0, s0 * np.minimum(eta * gs, 1.0) + smin)
    ax.scatter(gs, ps, s=4, color=COLORS[0], alpha=0.35, label="samples")
    ax.plot(g, mu, color=COLORS[3], lw=1.6, label=r"$\mu(g)=\min(\eta g,1)$")
    ax.fill_between(g, mu - 2 * sig, mu + 2 * sig, color=COLORS[3], alpha=0.18,
                    label=r"$\mu \pm 2\sigma(g)$")
    ax.set_xlabel("Normalised irradiance $g$")
    ax.set_ylabel("Normalised power $p$")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.05, 1.05)
    legend_below(fig, ax, ncol=3, y=-0.08)
    fig.subplots_adjust(bottom=0.22, top=0.92, left=0.15, right=0.98)
    return save(fig, out_dir, "fig1_model_illustration")


if __name__ == "__main__":  # pragma: no cover
    print("fig1: model illustration")
