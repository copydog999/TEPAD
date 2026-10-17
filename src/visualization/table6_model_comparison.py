# -*- coding: utf-8 -*-
"""Table 6: head-to-head model comparison on the Ding dataset (TEPAD vs 9 baselines)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .style import booktabs, load_baselines, write_tex_csv


def _fmt(v: Optional[float], nd: int = 3) -> str:
    """Format a number; ``None`` / ``nan`` are shown as ``--``."""
    if v is None:
        return "--"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "--"
    return "--" if f != f else ("%.{}f".format(nd) % f)


def build(source, out_dir: Path, payload: Optional[dict] = None) -> List[Path]:
    """Generate ``table6_model_comparison.{tex,csv}``.

    When ``payload`` is empty it is read from ``results/logs/baselines.json``; if that
    file does not exist (the comparison experiments have not been run), skip and
    return an empty list.
    """
    data = load_baselines(source, out_dir, payload)
    if not data:
        return []
    records = data["models"]
    ok = [r for r in records if "error" not in r]

    best_auc = max((r["auc"] for r in ok if r.get("auc") is not None), default=None)
    best_f1 = max((r["f1"] for r in ok), default=None)

    def bold(v, best, nd=3):
        s = _fmt(v, nd)
        return "\\textbf{%s}" % s if (v is not None and best is not None
                                      and abs(float(v) - float(best)) < 1e-9) else s

    header = ["Method", "AUC", "Precision", "Recall", "F1", "FPR"]
    rows = []
    for r in records:
        if "error" in r:
            continue
        name = r["name"] + ("$^{\\dagger}$" if r.get("supervised") else "")
        rows.append([name, bold(r.get("auc"), best_auc), _fmt(r["precision"]),
                     _fmt(r["recall"]), bold(r["f1"], best_f1), _fmt(r["fpr"], 3)])

    caption = ("Horizontal comparison of anomaly-detection models on the Ding dataset "
               "(%d training-normal daily curves for fitting, %d test curves for evaluation, "
               "%d normal / %d anomalous). Baseline threshold = 95th percentile of the "
               "reference-normal scores. $^{\\dagger}$supervised (uses training labels); all "
               "other methods are one-class and use only the measured irradiance and power "
               "channels (no modeled power)."
               % (data.get("n_ref", 0), data.get("n_eval", 0), data.get("n_eval_normal", 0),
                  data.get("n_eval_abnormal", 0)))
    tex = booktabs(header, rows, caption=caption, label="tab:model_comparison",
                   colspec="lrrrrr",
                   note="The cascade produces a boolean decision without a native continuous score, so its "
                                       "AUC is computed on the monotone normalized margin "
                                       "$\\max(d/\\tau_1, m_2/q_{95}(m_2^{\\mathrm{ref}}))$, where $m_2$ is the "
                                       "largest directed standard score of the residual features; the precision, "
                                       "recall, F1 and FPR columns keep the original thresholds and describe the "
                                       "same decision. TEPAD uses its own thresholds (Stage 1: 95\\% quantile; "
                                       "Stage 2: 1\\%/99\\% quantiles); all other models use the 95\\% quantile rule.")
    csv_rows = [{k: r.get(k) for k in ("name", "family", "supervised", "auc", "precision",
                                       "recall", "f1", "fpr", "threshold", "f1_best",
                                       "seconds", "error")} for r in records]
    return write_tex_csv(out_dir / "table6_model_comparison", tex, csv_rows)


if __name__ == "__main__":  # pragma: no cover
    print("table6: model comparison on Ding")
