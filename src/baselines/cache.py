"""Model-weight / score cache for the comparison experiments.

Every run of a deep baseline retrains and rescores (all 9 baselines take about
6 minutes on this machine, of which the GA search of AE-LSTM-GA is ~3.7 min and
the AnoGAN training + latent optimisation ~2 min). The artefacts are therefore
cached under ``results/models/``:

* ``<slug>.pt``  -- weights saved via ``torch.save`` (may hold several sub-models)
  plus hyper-parameter metadata;
* ``<slug>.npz`` -- reference / evaluation scores; on a hit, **training and
  scoring are skipped entirely**.

``slug`` = ``<model name>-<hyper-parameter hash><data hash>``, so any change to
the hyper-parameters, the fitting set or the evaluation set invalidates the entry
automatically and stale results are never reused; pass ``--no-cache`` to force
retraining.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = ["fingerprint", "data_digest", "make_slug", "load_scores", "save_scores",
           "save_checkpoint", "load_checkpoint"]


def fingerprint(obj: Any) -> str:
    """Compute a stable hash of a config dict (keys sorted, so equal configs hash equally)."""
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def data_digest(*arrays: np.ndarray) -> str:
    """Hash the input data so the cache is invalidated automatically when data changes."""
    h = hashlib.sha1()
    for a in arrays:
        arr = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
        h.update(str(arr.shape).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()[:12]


def make_slug(name: str, params: Dict, digest: str) -> str:
    """Build the file name (filesystem-safe characters only)."""
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in name)
    return "{}-{}{}".format(safe, fingerprint(params), digest)


def _stem(cache_dir: Path, slug: str) -> Path:
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / slug


def load_scores(cache_dir: Optional[Path], slug: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load the cached ``(scores_ref, scores_eval)``; return ``None`` if missing or corrupted."""
    if cache_dir is None:
        return None
    path = _stem(cache_dir, slug).with_suffix(".npz")
    if not path.exists():
        return None
    try:
        with np.load(path) as z:
            return np.asarray(z["ref"], dtype=float), np.asarray(z["eval"], dtype=float)
    except Exception:  # pragma: no cover - treat a corrupted cache as a miss
        return None


def save_scores(cache_dir: Optional[Path], slug: str, scores_ref: np.ndarray,
                scores_eval: np.ndarray, params: Optional[Dict] = None) -> Optional[Path]:
    """Cache the scores (plus a side-car ``.json`` with the hyper-parameters for manual inspection)."""
    if cache_dir is None:
        return None
    path = _stem(cache_dir, slug).with_suffix(".npz")
    np.savez(path, ref=np.asarray(scores_ref, dtype=float),
             eval=np.asarray(scores_eval, dtype=float))
    if params is not None:
        path.with_suffix(".json").write_text(
            json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_checkpoint(cache_dir: Optional[Path], slug: str, state_dicts: Dict[str, dict],
                    meta: Optional[Dict] = None) -> Optional[Path]:
    """Save the model weights (supports cases such as AnoGAN with several sub-models)."""
    if cache_dir is None:
        return None
    import torch

    path = _stem(cache_dir, slug).with_suffix(".pt")
    torch.save({"state_dicts": dict(state_dicts), "meta": meta or {}}, path)
    return path


def load_checkpoint(cache_dir: Optional[Path], slug: str) -> Optional[Dict]:
    """Load a weight checkpoint; return ``None`` if missing or corrupted."""
    if cache_dir is None:
        return None
    path = _stem(cache_dir, slug).with_suffix(".pt")
    if not path.exists():
        return None
    try:
        import torch

        return torch.load(path, map_location="cpu")
    except Exception:  # pragma: no cover - fall back to retraining on version mismatch
        return None
