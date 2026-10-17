"""Classical (sklearn / LightGBM) comparison baselines.

All functions follow the uniform interface ``f(X_ref, X_eval) -> (scores_ref, scores_eval)``,
where a larger score means more anomalous.

* :func:`dbscan_scores` -- density clustering, anomaly score = distance to the nearest core point (noise points are naturally the farthest);
* :func:`iforest_scores` -- isolation forest, anomaly score = ``-score_samples``;
* :func:`ocsvm_scores` -- one-class SVM (RBF), anomaly score = ``-decision_function``;
* :func:`lgbm_scores` -- LightGBM **supervised** classification (the only label-using baseline), outputs the anomaly probability.

These baselines all take an ``(n, n_features)`` statistical feature matrix and are
**fitted only on the reference normals**, using no labels (except `lgbm_scores`,
which is the supervised reference in the paper).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from ..config import SEED

__all__ = ["dbscan_scores", "iforest_scores", "ocsvm_scores", "lgbm_scores"]


def _standardize(ref: np.ndarray, ev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Standardize with the reference mean/std (fit on ``ref`` only to avoid information leakage)."""
    sc = StandardScaler().fit(ref)
    return sc.transform(ref), sc.transform(ev)


def _knee_eps(x: np.ndarray, k: int) -> float:
    """Estimate DBSCAN ``eps`` from the "elbow" of the ``k``-th nearest-neighbour distances."""
    nn = NearestNeighbors(n_neighbors=int(min(k, len(x)))).fit(x)
    dist, _ = nn.kneighbors(x)
    kd = np.sort(dist[:, -1])
    if kd.size < 3 or kd[-1] <= kd[0]:
        return float(np.percentile(kd, 90)) if kd.size else 1.0
    xaxis = np.linspace(0.0, 1.0, kd.size)
    yaxis = (kd - kd[0]) / (kd[-1] - kd[0])
    return float(kd[int(np.argmax(yaxis - xaxis))])


def dbscan_scores(X_ref: np.ndarray, X_eval: np.ndarray,
                  min_samples: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """DBSCAN baseline: anomaly score = Euclidean distance to the nearest core sample.

    ``eps`` comes from the elbow of the ``min_samples``-th nearest-neighbour distances
    of the reference set; if that ``eps`` yields no core points (everything is labelled
    noise), ``eps`` is doubled until core points appear.
    """
    Xr, Xe = _standardize(np.asarray(X_ref, float), np.asarray(X_eval, float))
    eps = _knee_eps(Xr, min_samples)
    if not np.isfinite(eps) or eps <= 0:
        eps = float(np.percentile(np.linalg.norm(Xr - Xr.mean(0), axis=1), 50) or 1.0)
    core = np.empty((0, Xr.shape[1]))
    for _ in range(8):
        model = DBSCAN(eps=float(eps), min_samples=int(min(min_samples, len(Xr)))).fit(Xr)
        core = np.asarray(model.components_, dtype=float)   # the core sample points themselves (not indices)
        if len(core) > 0:
            break
        eps *= 2.0
    if len(core) == 0:
        core = Xr

    def _dist(q: np.ndarray) -> np.ndarray:
        if len(q) == 0:
            return np.zeros(0)
        nn = NearestNeighbors(n_neighbors=1).fit(core)
        d, _ = nn.kneighbors(q)
        return d.ravel()

    return _dist(Xr), _dist(Xe)


def iforest_scores(X_ref: np.ndarray, X_eval: np.ndarray,
                   n_estimators: int = 300, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """Isolation-forest baseline: anomaly score = ``-score_samples`` (larger means more anomalous)."""
    Xr, Xe = _standardize(np.asarray(X_ref, float), np.asarray(X_eval, float))
    model = IsolationForest(n_estimators=int(n_estimators), random_state=seed,
                            contamination="auto").fit(Xr)
    return -model.score_samples(Xr), -model.score_samples(Xe)


def ocsvm_scores(X_ref: np.ndarray, X_eval: np.ndarray,
                 nu: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """One-class SVM baseline (RBF kernel): anomaly score = ``-decision_function``."""
    Xr, Xe = _standardize(np.asarray(X_ref, float), np.asarray(X_eval, float))
    model = OneClassSVM(kernel="rbf", gamma="scale", nu=float(nu)).fit(Xr)
    return (-model.decision_function(Xr).ravel(), -model.decision_function(Xe).ravel())


def lgbm_scores(X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray,
                n_estimators: int = 400, seed: int = SEED,
                X_ref: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """LightGBM baseline (**supervised**): binary classification on the labelled training set, outputs the anomaly probability.

    ``X_ref`` (the normal training set) is only used to fill the reference scores of the uniform return interface.
    """
    import lightgbm as lgb

    X_train = np.asarray(X_train, float)
    y_train = (np.asarray(y_train) != 0).astype(int)
    model = lgb.LGBMClassifier(n_estimators=int(n_estimators), learning_rate=0.05,
                               num_leaves=31, subsample=0.9, subsample_freq=1,
                               colsample_bytree=0.9, random_state=seed, n_jobs=-1,
                               verbose=-1)
    model.fit(X_train, y_train)
    scores_eval = model.predict_proba(np.asarray(X_eval, float))[:, 1]
    scores_ref = model.predict_proba(np.asarray(X_ref, float))[:, 1] if X_ref is not None \
        else np.zeros(0)
    return scores_ref, scores_eval


if __name__ == "__main__":  # pragma: no cover - minimal example
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (200, 6))
    b = rng.normal(2.5, 1, (40, 6))
    for fn in (dbscan_scores, iforest_scores, ocsvm_scores, knn_scores, lof_scores):
        s_ref, s_ev = fn(a, b)
        print(fn.__name__, "ref/ev mean:", round(float(s_ref.mean()), 3),
              round(float(s_ev.mean()), 3))
    seq_a = rng.normal(0, 1, (60, 2, 48)) + np.linspace(0, 1, 48)
    seq_b = seq_a[:10] + 0.8
    for fn in (sigma3_scores, arima_scores):
        s_ref, s_ev = fn(seq_a, seq_b)
        print(fn.__name__, "ref/ev mean:", round(float(s_ref.mean()), 3),
              round(float(s_ev.mean()), 3))
