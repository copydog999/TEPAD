# -*- coding: utf-8 -*-
"""Visualisation entry point: generates all paper tables and figures in one place.

The manifest contains only artefacts that appear in the paper and maps one-to-one to ``paper.tex``:

* Tables 1--6: main experiment, cross-dataset, per-class recall, feature-group ablation, residual-feature separability, baseline comparison;
* Figures 1--7: model illustration, two-stage pipeline, score distribution, per-class recall, cross-dataset, residual features, baseline comparison.
"""
from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import List

from .style import apply_style, by_name  # noqa: F401

TABLE_MODULES = ["table1_main_ding", "table2_cross_dataset", "table3_per_class_recall",
                 "table4_ablation", "table5_stage2_features", "table6_model_comparison"]
FIGURE_MODULES = ["fig1_model_illustration", "fig2_pipeline", "fig3_score_distribution",
                  "fig4_per_class_recall", "fig5_cross_dataset", "fig6_stage2_features",
                  "fig7_model_comparison"]


def make_all(results: List[dict], tables_dir: Path, figures_dir: Path,
             logger: logging.Logger = None) -> List[Path]:
    """Generate all paper tables and figures, returning the written paths."""
    apply_style()
    created: List[Path] = []
    for name in TABLE_MODULES + FIGURE_MODULES:
        module = importlib.import_module("{}.{}".format(__name__, name))
        out = Path(tables_dir) if name.startswith("table") else Path(figures_dir)
        try:
            created += list(module.build(results, out))
        except Exception as exc:  # pragma: no cover - one failing artefact must not abort the rest
            if logger:
                logger.warning("%s failed: %s", name, exc)
            else:
                raise
    if logger:
        logger.info("generated %d files", len(created))
    return created


def make_cached_figures(results_dir: Path) -> List[Path]:
    """Re-render only the figures, reusing ``results/logs/metrics.json`` without rerunning any experiment."""
    results_dir = Path(results_dir)
    logs_dir = results_dir / "logs"
    figures_dir = results_dir / "figures"
    with (logs_dir / "metrics.json").open(encoding="utf-8") as stream:
        metrics = json.load(stream)
    results = [dict(value, dataset=name) for name, value in metrics.items()]
    apply_style()
    created: List[Path] = []
    for name in FIGURE_MODULES:
        module = importlib.import_module("{}.{}".format(__name__, name))
        created.extend(module.build(results, figures_dir))

    simulation_path = results_dir / "simulation" / "logs" / "simulation_results.json"
    if simulation_path.exists():
        import pandas as pd
        from ..simulation.plots import make_simulation_figures
        simulation_dir = simulation_path.parents[1]
        payload = json.loads(simulation_path.read_text(encoding="utf-8"))
        names = ("bias_variance", "window", "noise", "contamination", "convergence")
        frames = {
            name: pd.read_csv(simulation_dir / "tables" / ("simulation_stability_" + name + ".csv"))
            for name in names
        }
        make_simulation_figures(payload, frames, simulation_dir / "figures")
        created.extend((simulation_dir / "figures").glob("sim_*.pdf"))
        created.extend((simulation_dir / "figures").glob("sim_*.png"))
    return created


if __name__ == "__main__":  # pragma: no cover - minimal example
    print("tables :", TABLE_MODULES)
    print("figures:", FIGURE_MODULES)
