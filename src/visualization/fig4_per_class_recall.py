# -*- coding: utf-8 -*-
"""Figure 4: feature ablation and union-decision recall for each label group of Dataset A."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from matplotlib.patches import Patch

from .style import BEST_COLOR, COLORS, by_name, new_fig, save

ORDER = ["spike", "inverter_fault", "snowy", "cloudy", "low_value", "shading"]


def build(results: List[dict], out_dir: Path) -> List[Path]:
    """Generate ``fig4_per_class_recall.{pdf,png}``."""
    r = by_name(results)["ding"]
    levels = r.get("per_class_levels", {})
    labels, s1, casc = [], [], []
    for c in ORDER:
        if c in levels:
            labels.append(c.replace("_", "\n"))
            s1.append(levels[c].get("stage1", 0.0))
            casc.append(levels[c].get("cascade", 0.0))
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = new_fig(double=True, height=2.8)
    baseline = max(0.0, min(0.65, float(np.nanmin(s1 + casc)) - 0.05))
    best = np.maximum(s1, casc)
    ax.bar(x - w / 2, np.asarray(s1) - baseline, w, bottom=baseline,
           label="Probability-model only",
           color=[BEST_COLOR if value == best[i] and s1[i] > casc[i] else COLORS[0]
                  for i, value in enumerate(s1)])
    ax.bar(x + w / 2, np.asarray(casc) - baseline, w, bottom=baseline,
           label="Cascade",
           color=[BEST_COLOR if value == best[i] and casc[i] >= s1[i] else COLORS[1]
                  for i, value in enumerate(casc)])
    for xi, v in zip(x - w / 2, s1):
        ax.text(xi, v + 0.012, "%.2f" % v, ha="center", fontsize=6)
    for xi, v in zip(x + w / 2, casc):
        ax.text(xi, v + 0.012, "%.2f" % v, ha="center", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(baseline, 1.08)
    ax.set_ylabel("Recall")
    handles = [
        Patch(facecolor=COLORS[0], label="Probability-model only"),
        Patch(facecolor=COLORS[1], label="Cascade"),
        Patch(facecolor=BEST_COLOR, label="Best recall per class"),
    ]
    fig.legend(handles, [handle.get_label() for handle in handles],
               frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.01), fontsize=6)
    ax.set_title("Dataset A: per-label recall (Stage 1 vs cascade)")
    fig.subplots_adjust(bottom=0.22, top=0.90, left=0.08, right=0.99)
    return save(fig, out_dir, "fig4_per_class_recall")


if __name__ == "__main__":  # pragma: no cover
    print("fig4: per-class recall")
