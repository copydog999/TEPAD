# -*- coding: utf-8 -*-
"""Table 1: main Ding experiment -- Stage 1 (endogenous) and cascade results."""
from __future__ import annotations

from pathlib import Path
from typing import List

from .style import booktabs, by_name, write_tex_csv


def build(results: List[dict], out_dir: Path) -> List[Path]:
    """Generate ``table1_main_ding.{tex,csv}``."""
    r = by_name(results)["ding"]
    best = max(r["stage1_f1"], r["cascade_f1"])

    def b(v):  # bold the best value
        return "\\textbf{%.3f}" % v if abs(v - best) < 1e-9 else "%.3f" % v

    header = ["Method", "AUC", "Precision", "Recall", "F1"]
    rows = [
        ["Stage 1 (endogenous Mahalanobis)",
         "%.3f" % r["stage1_auc"], "%.3f" % r["stage1_precision"],
         "%.3f" % r["stage1_recall"], b(r["stage1_f1"])],
        ["Cascade (Stage 1 $\\cup$ Stage 2)",
         "--", "%.3f" % r["cascade_precision"],
         "%.3f" % r["cascade_recall"], b(r["cascade_f1"])],
    ]
    tex = booktabs(
        header, rows,
        caption="Main experiment on the Ding dataset (%d daily curves, 7 classes). "
                "Stage 1 scores window-parameter features; the cascade adds the "
                "external-reference residual rules. Thresholds are normal empirical "
                "quantiles, so the boolean cascade has no native ranking score."
                % r["n_curves"],
        label="tab:main_ding", colspec="lrrrr")
    csv_rows = [
        dict(method="stage1", auc=r["stage1_auc"], precision=r["stage1_precision"],
             recall=r["stage1_recall"], f1=r["stage1_f1"]),
        dict(method="cascade", auc="", precision=r["cascade_precision"],
             recall=r["cascade_recall"], f1=r["cascade_f1"]),
    ]
    return write_tex_csv(out_dir / "table1_main_ding", tex, csv_rows)


if __name__ == "__main__":  # pragma: no cover
    print("table1: main Ding comparison")
