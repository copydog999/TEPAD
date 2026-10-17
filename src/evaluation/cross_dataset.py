"""Cross-dataset execution and summarization.

``run_dataset`` chains the full "load → preprocess → two-level detection →
cascade → metrics" flow, and ``summarize`` assembles the results of all datasets
into the paper comparison table.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ..detection.cascade import DetectionResult, run_two_level
from ..io import get_loader
from ..preprocess.daily_split import build_curves
from .metrics import (anomaly_ratio, auc_score, binary_metrics, curve_counts,
                      false_positive_rate)
from .per_class import per_class_recall, per_class_table


def summarize_result(name: str, det: DetectionResult, curves: list) -> Dict:
    """Compress the two-level cascade detection result into a serializable metric dictionary."""
    out: Dict = {"dataset": name, "n_curves": det.n_curves,
                 "pref_source": det.pref_source, "has_label": det.has_label}
    out.update(curve_counts(det.y))
    out["stage1_auc"] = auc_score(det.y, det.d)
    out["stage1_tau"] = det.tau1
    out["stage2_taus"] = det.taus
    out["anomaly_ratio"] = anomaly_ratio(det.casc)
    out["stage1_ratio"] = anomaly_ratio(det.s1)
    out["stage2_candidate_ratio"] = anomaly_ratio(det.s2)

    if det.has_label:
        out.update({"cascade_" + k: v for k, v in binary_metrics(det.y, det.casc).items()})
        out.update({"stage1_" + k: v for k, v in binary_metrics(det.y, det.s1).items()})
        out.update({"stage2_" + k: v for k, v in binary_metrics(det.y, det.s2).items()})
        out["cascade_fpr"] = false_positive_rate(det.y, det.casc)
        out["stage1_fpr"] = false_positive_rate(det.y, det.s1)
        out["stage2_feat_auc"] = {k: auc_score(det.y, det.feats[k]) for k in det.feats}
        out["per_class"] = per_class_recall(curves, det.casc)
        out["per_class_levels"] = per_class_table(curves, det.s1, det.s2, det.casc)
    return out


def run_dataset(name: str) -> Dict:
    """Run a full dataset and return the result dictionary.

    Parameters
    ----------
    name : str
        ``ding`` / ``ds226`` / ``lareunion``.

    Returns
    -------
    dict
        Metric dictionary, additionally containing ``curves`` and ``det`` (used for plotting).
    """
    df, spec = get_loader(name)()
    has_label = spec.get("label_col") is not None
    curves = build_curves(df, name, **spec)
    det = run_two_level(curves, has_label)
    summary = summarize_result(name, det, curves)
    summary["curves"] = curves
    summary["det"] = det
    return summary


def summarize(results: List[Dict]) -> pd.DataFrame:
    """Assemble the results of several datasets into a comparison table (DataFrame)."""
    rows = []
    for r in results:
        rows.append({
            "dataset": r["dataset"],
            "n_curves": r["n_curves"],
            "n_normal": r["n_normal"],
            "n_abnormal": r["n_abnormal"],
            "stage1_auc": r["stage1_auc"],
            "cascade_precision": r.get("cascade_precision", float("nan")),
            "cascade_recall": r.get("cascade_recall", float("nan")),
            "cascade_f1": r.get("cascade_f1", float("nan")),
            "cascade_fpr": r.get("cascade_fpr", float("nan")),
            "anomaly_ratio": r["anomaly_ratio"],
            "pref_source": r["pref_source"],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":  # pragma: no cover - minimal example
    res = run_dataset("ds226")
    print({k: v for k, v in res.items() if not isinstance(v, (list, dict))})
