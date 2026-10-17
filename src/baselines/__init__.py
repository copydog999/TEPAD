"""Comparison experiments: head-to-head model comparison on the Ding dataset
(classical + deep anomaly detection baselines).

Protocol
--------
* **Fitting set**: normal daily curves in the official Ding ``train`` split
  (``datasets/ding/raw/train/*_train.csv``);
* **Evaluation set**: all daily curves in the official Ding ``test`` split
  (normal + 6 anomaly classes).

Apart from ``LGBM (supervised)``, which uses all training labels for supervised
classification, the remaining baselines are **one-class / unsupervised** methods:
they are fitted only on the normal reference curves and score the test curves
(higher score means more anomalous), with the threshold taken at the 95%
empirical quantile of the reference normal scores (consistent with the first stage).

All baselines use **only the measured irradiance ``g`` and the measured power ``p``**,
not the ``modeled_power`` field shipped with the dataset (that field matches the
measured power point by point for normal samples, so using it would reduce the task
to trivial discrimination). They are therefore strictly comparable with the first
(endogenous) stage of TEPAD; the TEPAD cascade additionally uses an external
reference and is reported as the full method.
"""
from __future__ import annotations

import os as _os
import warnings as _warnings


def _quiet_loky_cpu_probe() -> None:
    """Pre-fill the joblib/loky physical-core cache.

    Some Windows environments lack ``wmic``, which makes joblib dump an unrelated
    exception traceback to stderr while probing physical cores (and, since the
    probe failed, nothing is cached, so it repeats). Here the core count is set
    explicitly and the cache pre-filled to keep the logs clean.
    """
    try:
        _os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(_os.cpu_count() or 1))
        from joblib.externals.loky.backend import context as _loky
        if getattr(_loky, "physical_cores_cache", None) is None:
            _loky.physical_cores_cache = int(_os.environ["LOKY_MAX_CPU_COUNT"])
    except Exception:  # pragma: no cover - fallback for environment differences
        pass


_quiet_loky_cpu_probe()
_warnings.filterwarnings("ignore", category=UserWarning,
                         module=r"joblib\.externals\.loky\.backend\.context")

from .runner import MODELS, run_baselines  # noqa: E402

__all__ = ["MODELS", "run_baselines"]
