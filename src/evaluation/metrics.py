"""Evaluation metrics: AUC, Precision/Recall/F1, and anomaly ratio."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score


def auc_score(y_true: np.ndarray, score: np.ndarray) -> float:
    """Binary AUC (labels are binarized first: 0 = normal, otherwise abnormal; returns ``nan`` for a single class)."""
    y_bin = (np.asarray(y_true) != 0).astype(int)
    if len(np.unique(y_bin)) < 2:
        return float("nan")
    return float(roc_auc_score(y_bin, np.asarray(score, dtype=float)))


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Binary P / R / F1 / Accuracy (labels are binarized first; division-by-zero safe)."""
    y_true = (np.asarray(y_true) != 0).astype(int)
    y_pred = np.asarray(y_pred, dtype=bool).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / len(y_true) if len(y_true) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """False-positive rate on normal samples (labels are binarized first)."""
    y_true = (np.asarray(y_true) != 0).astype(int)
    y_pred = np.asarray(y_pred, dtype=bool).astype(int)
    neg = y_true == 0
    return float(np.mean(y_pred[neg])) if neg.any() else 0.0


def anomaly_ratio(flags: np.ndarray) -> float:
    """Anomaly ratio (the reporting measure for unlabeled datasets)."""
    return float(np.mean(np.asarray(flags, dtype=bool)))


def curve_counts(y_true: np.ndarray) -> Dict[str, int]:
    """Counts of normal/abnormal curves."""
    y_true = np.asarray(y_true, dtype=int)
    return {"n_normal": int(np.sum(y_true == 0)), "n_abnormal": int(np.sum(y_true != 0)),
            "n_total": int(len(y_true))}


if __name__ == "__main__":  # pragma: no cover - minimal example
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 1, 1, 1])
    print(binary_metrics(y, p))
    print("auc:", auc_score(y, np.array([.1, .2, .8, .9])))
