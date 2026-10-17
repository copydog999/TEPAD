# -*- coding: utf-8 -*-
"""Figure 2: block diagram of the two-stage TEPAD pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .style import COLORS, new_fig, save


def _box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                linewidth=1.0, edgecolor=color, facecolor=color, alpha=0.14))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5)


def _arrow(ax, p1, p2, color="#444444"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=9,
                                 linewidth=1.0, color=color))


def build(results: List[dict], out_dir: Path) -> List[Path]:
    """Generate ``fig2_pipeline.{pdf,png}``."""
    fig, ax = new_fig(double=True, height=2.2)
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)

    _box(ax, 0.1, 1.1, 1.5, 0.8, "Raw PV data\n$(g_t, p_t)$", COLORS[0])
    _box(ax, 2.0, 1.6, 1.8, 0.8, "Stage 1\nendogenous\nMahalanobis $d$", COLORS[1])
    _box(ax, 2.0, 0.3, 1.8, 0.8, "Stage 2\nexogenous residual\n$J$", COLORS[2])
    _box(ax, 4.4, 1.1, 1.5, 0.8, "Cascade\n($s_1 \\cup s_2$)", COLORS[3])
    _box(ax, 6.4, 1.1, 1.6, 0.8, "Detection\noutput", COLORS[4])
    _box(ax, 8.3, 1.1, 1.5, 0.8, "Evaluation\nP / R / F1 / AUC", COLORS[5])

    _arrow(ax, (1.6, 1.5), (2.0, 1.9))
    _arrow(ax, (1.6, 1.3), (2.0, 0.8))
    _arrow(ax, (3.8, 2.0), (4.4, 1.6))
    _arrow(ax, (3.8, 0.7), (4.4, 1.2))
    _arrow(ax, (5.9, 1.5), (6.4, 1.5))
    _arrow(ax, (8.0, 1.5), (8.3, 1.5))
    ax.set_title("TEPAD pipeline: endogenous Stage 1 gated by an exogenous Stage 2", fontsize=8)
    return save(fig, out_dir, "fig2_pipeline")


if __name__ == "__main__":  # pragma: no cover
    print("fig2: pipeline diagram")
