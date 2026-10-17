"""Comparison experiment runner: run all head-to-head models on the Ding dataset and persist the results.

Usage
-----
    python src/pipeline/run_baselines.py --output results/
    python src/pipeline/run_baselines.py --models IForest,Autoencoder
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import (BASELINE_EPOCHS, BASELINE_SEQ_LEN, BASELINE_THRESHOLD_QUANTILE,
                      RESULTS_DIR, SEED, STAGE1_QUANTILE, set_seed)
from ..detection.cascade import run_two_level
from ..evaluation.metrics import auc_score, binary_metrics, false_positive_rate
from . import classical, features
from .data import load_ding_split, reference_curves

LOGGER = logging.getLogger("tepad")

#: Comparison models (this order is the table row order)
MODELS: List[str] = [
    "TEPAD", "DBSCAN", "IForest", "One-Class SVM", "LGBM (supervised)",
    "Autoencoder", "AE-LSTM-GA", "GRU-Bayesian", "AnoGAN",
    "AnomalyTransformer", "Deep-SVDD",
]

#: Model -> (family, whether labels are used)
MODEL_INFO: Dict[str, tuple] = {
    "TEPAD": ("Unsupervised (ours)", False),
    "DBSCAN": ("Clustering", False),
    "IForest": ("Ensemble / tree", False),
    "One-Class SVM": ("Kernel", False),
    "LGBM (supervised)": ("Boosting", True),
    "Autoencoder": ("Reconstruction / MLP", False),
    "AE-LSTM-GA": ("Reconstruction / LSTM + GA", False),
    "GRU-Bayesian": ("Reconstruction / GRU + MC dropout", False),
    "AnoGAN": ("Generative / adversarial", False),
    "AnomalyTransformer": ("Reconstruction / transformer", False),
    "Deep-SVDD": ("Deep one-class", False),
}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def best_f1(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Best F1 attainable over all percentile thresholds (oracle, reference upper bound only)."""
    y = (np.asarray(y_true) != 0).astype(int)
    order = np.argsort(-np.asarray(scores, dtype=float))
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(int(y.sum()), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(np.max(f1)) if f1.size else 0.0


def evaluate(y_true: np.ndarray, scores_eval: np.ndarray, *,
             scores_ref: Optional[np.ndarray] = None, flags: Optional[np.ndarray] = None,
             auc: Optional[float] = None,
             threshold: Optional[float] = None) -> Dict[str, float]:
    """Uniform evaluation.

    The default threshold is the ``BASELINE_THRESHOLD_QUANTILE`` empirical quantile of
    the reference normal scores; if ``flags`` are passed directly (e.g. the TEPAD
    multi-criterion output), their decisions are used as-is.
    """
    scores_eval = np.asarray(scores_eval, dtype=float)
    if flags is None:
        if threshold is None:
            if scores_ref is None:
                raise ValueError("one of scores_ref or threshold is required")
            threshold = float(np.percentile(np.asarray(scores_ref, dtype=float),
                                            BASELINE_THRESHOLD_QUANTILE))
        pred = scores_eval > float(threshold)
    else:
        pred = np.asarray(flags, dtype=bool)
    m = binary_metrics(y_true, pred)
    return {
        "auc": float(auc_score(y_true, scores_eval)) if auc is None else auc,
        "threshold": None if threshold is None else float(threshold),
        "precision": float(m["precision"]), "recall": float(m["recall"]),
        "f1": float(m["f1"]), "accuracy": float(m["accuracy"]),
        "fpr": float(false_positive_rate(y_true, pred)),
        "tp": int(m["tp"]), "fp": int(m["fp"]), "fn": int(m["fn"]), "tn": int(m["tn"]),
        "f1_best": best_f1(y_true, scores_eval),
    }


def _sanitize(obj):
    """Turn ``nan`` into ``None`` so the written JSON is strictly valid."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, float) and obj != obj:
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    return obj


# --------------------------------------------------------------------------- #
# TEPAD reference fit
# --------------------------------------------------------------------------- #

def _normalized_margin(d: np.ndarray, feats: Dict[str, np.ndarray], n_ref: int,
                       tau1: float) -> np.ndarray:
    """Create a continuous ranking score aligned with the two hard decisions."""
    d = np.asarray(d, dtype=float)
    reference_mask = np.arange(len(d)) < n_ref
    margins = [d / max(float(tau1), 1e-12)]
    tails = {
        "r_min": "lower",
        "delta_r_min": "lower",
        "f_neg": "upper",
        "corr": "lower",
    }
    residual_margins = []
    for name, tail in tails.items():
        values = np.asarray(feats[name], dtype=float)
        reference_values = values[reference_mask]
        center = float(np.nanmean(reference_values))
        scale = float(np.nanstd(reference_values))
        scale = scale if scale > 1e-12 else 1.0
        z_score = (values - center) / scale
        residual_margins.append(z_score if tail == "upper" else -z_score)
    residual_score = np.max(np.vstack(residual_margins), axis=0)
    reference_quantile = float(
        np.nanpercentile(residual_score[reference_mask], STAGE1_QUANTILE)
    )
    if reference_quantile <= 1e-12:
        reference_quantile = 1.0
    margins.append(residual_score / reference_quantile)
    return np.maximum.reduce(margins)


def tepad_detection(ref: Sequence, test: Sequence):
    """Fit the cascade on training-normal curves and score the test curves."""
    curves = list(ref) + list(test)
    fit_mask = np.array([True] * len(ref) + [False] * len(test), dtype=bool)
    det = run_two_level(curves, has_label=True, fit_mask=fit_mask)
    n = len(ref)
    cascade_score = _normalized_margin(det.d, det.feats, n, det.tau1)
    return det.d[n:], det.s1[n:], det.casc[n:], float(det.tau1), cascade_score[n:]


# --------------------------------------------------------------------------- #
# Main flow
# --------------------------------------------------------------------------- #
def run_baselines(dataset: str = "ding", output: Path = RESULTS_DIR,
                  models: Optional[Sequence[str]] = None, seq_len: int = BASELINE_SEQ_LEN,
                  epochs: int = BASELINE_EPOCHS, cache_dir: Optional[Path] = None,
                  use_cache: bool = True,
                  logger: Optional[logging.Logger] = None) -> Dict:
    """Run a set of comparison models, write ``results/logs/baselines.json`` and build Table 6 / Figure 7.

    Parameters
    ----------
    dataset : str
        Only ``ding`` is supported for now.
    output : Path
        Result directory (``tables/``, ``figures/``, ``logs/``).
    models : Sequence[str], optional
        Subset of models to run; defaults to all.
    seq_len : int
        Resampling length of the daily curves.
    epochs : int
        Number of training epochs for the deep baselines.
    cache_dir : Path, optional
        Weight/score cache directory; defaults to ``<output>/models``.
    use_cache : bool
        Whether to reuse the cache (``False`` forces retraining and rescoring).
    logger : logging.Logger, optional
    """
    log = logger or LOGGER
    set_seed(SEED)
    if dataset != "ding":
        raise NotImplementedError("the comparison experiments currently cover the Ding dataset only")
    cache_dir = Path(cache_dir) if cache_dir is not None else Path(output) / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)

    train, test = load_ding_split()
    ref = reference_curves(train)
    y_eval = np.array([c.label for c in test], dtype=int)
    log.info("comparison experiments | reference normal=%d | test=%d (normal=%d, abnormal=%d)",
             len(ref), len(test), int((y_eval == 0).sum()), int((y_eval != 0).sum()))

    X_ref = features.feature_matrix(ref)
    X_eval = features.feature_matrix(test)
    X_train = features.feature_matrix(train)
    y_train = np.array([c.label for c in train], dtype=int)
    S_ref = features.sequence_matrix(ref, seq_len)
    S_eval = features.sequence_matrix(test, seq_len)

    # import the deep models lazily, so the other baselines still run without torch
    from . import deep

    selected = list(models) if models else list(MODELS)
    unknown = [m for m in selected if m not in MODEL_INFO]
    if unknown:
        raise KeyError("unknown model: %s (available: %s)" % (unknown, MODELS))

    d_eval = stage1_flags = cascade_flags = cascade_score = None
    stage1_threshold = None
    if "TEPAD" in selected:
        d_eval, stage1_flags, cascade_flags, stage1_threshold, cascade_score = (
            tepad_detection(ref, test)
        )

    kw = dict(cache_dir=cache_dir, use_cache=use_cache)
    records: List[Dict] = []
    failures = []
    for name in selected:
        t0 = time.time()
        log.info("--> %s", name)
        try:
            if name == "TEPAD":
                res = evaluate(y_eval, cascade_score, flags=cascade_flags)
            elif name == "DBSCAN":
                s_ref, s_ev = classical.dbscan_scores(X_ref, X_eval)
                res = evaluate(y_eval, s_ev, scores_ref=s_ref)
            elif name == "IForest":
                s_ref, s_ev = classical.iforest_scores(X_ref, X_eval)
                res = evaluate(y_eval, s_ev, scores_ref=s_ref)
            elif name == "One-Class SVM":
                s_ref, s_ev = classical.ocsvm_scores(X_ref, X_eval)
                res = evaluate(y_eval, s_ev, scores_ref=s_ref)
            elif name == "LGBM (supervised)":
                s_ref, s_ev = classical.lgbm_scores(X_train, y_train, X_eval, X_ref=X_ref)
                res = evaluate(y_eval, s_ev, scores_ref=s_ref)
            elif name == "Autoencoder":
                s_ref, s_ev = deep.autoencoder_scores(S_ref, S_eval, epochs=epochs, **kw)
                res = evaluate(y_eval, s_ev, scores_ref=s_ref)
            elif name == "AE-LSTM-GA":
                s_ref, s_ev = deep.aelstm_ga_scores(S_ref, S_eval, epochs=epochs, **kw)
                res = evaluate(y_eval, s_ev, scores_ref=s_ref)
            elif name == "GRU-Bayesian":
                s_ref, s_ev = deep.gru_bayesian_scores(S_ref, S_eval, epochs=epochs, **kw)
                res = evaluate(y_eval, s_ev, scores_ref=s_ref)
            elif name == "AnoGAN":
                s_ref, s_ev = deep.anogan_scores(S_ref, S_eval, **kw)
                res = evaluate(y_eval, s_ev, scores_ref=s_ref)
            elif name == "AnomalyTransformer":
                s_ref, s_ev = deep.anomaly_transformer_scores(S_ref, S_eval, epochs=epochs, **kw)
                res = evaluate(y_eval, s_ev, scores_ref=s_ref)
            elif name == "Deep-SVDD":
                s_ref, s_ev = deep.deep_svdd_scores(S_ref, S_eval, epochs=epochs, **kw)
                res = evaluate(y_eval, s_ev, scores_ref=s_ref)
            else:  # pragma: no cover - guarded by the unknown check above
                raise KeyError(name)
        except Exception as exc:  # pragma: no cover - record the failure but keep the overall run going
            log.exception("model %s failed: %s", name, exc)
            records.append({"name": name, "family": MODEL_INFO[name][0],
                            "supervised": MODEL_INFO[name][1], "error": str(exc)})
            failures.append("{}: {}".format(name, exc))
            continue
        family, supervised = MODEL_INFO[name]
        res.update({"name": name, "family": family, "supervised": bool(supervised),
                    "seconds": round(time.time() - t0, 2)})
        records.append(res)
        auc_v = res["auc"]
        log.info("    AUC=%s  P=%.3f  R=%.3f  F1=%.3f  FPR=%.3f  (%.1fs)",
                 "n/a" if auc_v is None or auc_v != auc_v else "%.4f" % auc_v,
                 res["precision"], res["recall"], res["f1"], res["fpr"], res["seconds"])

    payload = _sanitize({
        "dataset": dataset,
        "protocol": ("fit = Ding train-split normal daily curves; "
                     "eval = Ding test-split daily curves"),
        "inputs": ("measured irradiance, measured power, and available external reference; "
                   "window-parameter Mahalanobis scores are cascaded with residual rules"),
        "threshold": ("95th percentile of reference-normal parameter distances; residual "
                      "features use the configured empirical quantiles"),
        "cache_dir": str(cache_dir),
        "cache_used": bool(use_cache),
        "seq_len": int(seq_len),
        "n_features": int(X_ref.shape[1]),
        "epochs": int(epochs),
        "n_ref": len(ref), "n_eval": len(test),
        "n_eval_normal": int((y_eval == 0).sum()),
        "n_eval_abnormal": int((y_eval != 0).sum()),
        "models": records,
    })

    out_dir = Path(output)
    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    metrics_path = logs / "baselines.json"
    if metrics_path.exists():
        try:
            previous = json.loads(metrics_path.read_text(encoding="utf-8")).get("models", [])
        except (OSError, ValueError):
            previous = []
        dropped = [m["name"] for m in previous if m.get("name") not in selected]
        if dropped:
            log.warning("this is a subset run; baselines.json and Table 6 / Figure 7 will be overwritten, "
                        "and the results of these models will be dropped from the artefacts: %s", dropped)
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    log.info("baselines.json written (%d models)", len(records))

    from ..visualization import fig7_model_comparison, table6_model_comparison
    table6_model_comparison.build(payload["models"], out_dir / "tables", payload)
    fig7_model_comparison.build(payload["models"], out_dir / "figures", payload)
    log.info("table6 & fig7 written")
    if failures:
        raise RuntimeError("Baseline models failed: {}".format("; ".join(failures)))
    return payload


if __name__ == "__main__":  # pragma: no cover - minimal example
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    run_baselines(models=["IForest", "One-Class SVM", "Autoencoder"])
