# -*- coding: utf-8 -*-
"""Table 2: cross-dataset validation (Ding / DS226 / La Réunion)."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .style import booktabs, by_name, write_tex_csv

ORDER = ["ding", "ds226", "lareunion"]


def _f(v, fmt="%.3f"):
    return fmt % v if isinstance(v, (int, float)) and np.isfinite(v) else "--"


def build(results: List[dict], out_dir: Path) -> List[Path]:
    """Generate ``table2_cross_dataset.{tex,csv}``."""
    d = by_name(results)
    header = ["Dataset", "Curves", "Stage-1 AUC", "Precision", "Recall", "F1"]
    rows, csv_rows = [], []
    for name in ORDER:
        if name not in d:
            continue
        r = d[name]
        rows.append([name, str(r["n_curves"]), _f(r["stage1_auc"]),
                     _f(r.get("cascade_precision", np.nan)),
                     _f(r.get("cascade_recall", np.nan)),
                     _f(r.get("cascade_f1", np.nan))])
        csv_rows.append(dict(dataset=name, n_curves=r["n_curves"],
                             stage1_auc=r["stage1_auc"],
                             precision=r.get("cascade_precision"),
                             recall=r.get("cascade_recall"),
                             f1=r.get("cascade_f1"),
                             anomaly_ratio=r["anomaly_ratio"],
                             pref_source=r["pref_source"]))
    tex = booktabs(
        header, rows,
        caption="Cross-dataset validation with the same TEPAD cascade. Precision, recall, and "
                "F1 refer to the cascade decision; the AUC column is the Stage-1 "
                "Mahalanobis ranking score.",
        label="tab:cross_dataset", colspec="lrrrrr",
        note="Ding uses the built-in \\texttt{modeled\\_power}; La R\\'eunion uses a healthy "
             "twin inverter; DS226 falls back to the physical reference "
             "$\\min(\\eta g,1)$.")
    return write_tex_csv(out_dir / "table2_cross_dataset", tex, csv_rows)


if __name__ == "__main__":  # pragma: no cover
    print("table2: cross-dataset")
