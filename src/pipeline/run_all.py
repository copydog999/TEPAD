# -*- coding: utf-8 -*-
"""One-click run of all TEPAD experiments, tables and figures.

Usage
-----
    python src/pipeline/run_all.py --datasets all   --output results/
    python src/pipeline/run_all.py --datasets ding  --output results/
    python src/pipeline/run_all.py --datasets cross --output results/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.evaluation.cross_dataset import run_dataset, summarize  # noqa: E402

LOGGER = logging.getLogger("tepad")


def _setup_logging(log_path: Path) -> None:
    """Log to both the console and a log file."""
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    LOGGER.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    LOGGER.addHandler(sh)


def _jsonable(v):
    """Convert numpy types / nested structures into JSON-serializable objects."""
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def resolve_datasets(which: str) -> List[str]:
    """Resolve the value of ``--datasets`` into a list of dataset names."""
    if which == "all":
        return list(config.DATASETS_ALL)
    if which == "cross":
        return list(config.DATASETS_CROSS)
    if which in config.DATASETS_ALL:
        return [which]
    return [w for w in which.split(",") if w]


def _save_processed(r: Dict) -> None:
    """Write the daily-curve summary of a dataset to ``datasets/<name>/processed/``."""
    d = config.dataset_processed_dir(r["dataset"])
    d.mkdir(parents=True, exist_ok=True)
    rows = [{
        "site": c.site, "date": c.date, "label": c.label, "cls": c.cls,
        "n_obs": c.n_obs, "stage1_distance": float(score),
        "stage1_flag": bool(stage1_flag),
        "stage2_flag": bool(stage2_flag),
        "cascade_flag": bool(cascade_flag),
    } for c, score, stage1_flag, stage2_flag, cascade_flag in zip(
        r["curves"], r["det"].d, r["det"].s1,
        r["det"].s2, r["det"].casc
    )]
    import pandas as pd
    pd.DataFrame(rows).to_csv(d / "daily_curves.csv", index=False)


def run(names: List[str], output: Path, baselines: bool = False,
        use_cache: bool = True, cache_dir: Path = None) -> List[Dict]:
    """Run a group of datasets and persist metrics.json plus the processed daily curves.

    Parameters
    ----------
    names : List[str]
        List of dataset names.
    output : Path
        Result output directory.
    baselines : bool
        Whether to additionally run the Ding comparison experiment (table 6 / figure 7 /
        ``baselines.json``).
    use_cache : bool
        Whether the comparison experiment reuses cached weights and scores.
    cache_dir : Path, optional
        Cache directory for the comparison experiment, defaults to ``<output>/models``.

    Returns
    -------
    List[dict]
        Result dictionary per dataset (including ``curves`` / ``det``, reused for plotting).
    """
    config.set_seed()
    tables = output / "tables"
    figs = output / "figures"
    logs = output / "logs"
    for d in (tables, figs, logs):
        d.mkdir(parents=True, exist_ok=True)
    _setup_logging(logs / "run_all.log")

    LOGGER.info("TEPAD pipeline start | datasets=%s | output=%s", names, output)
    results: List[Dict] = []
    for name in names:
        LOGGER.info("==> running dataset: %s", name)
        r = run_dataset(name)
        _save_processed(r)
        results.append(r)
        LOGGER.info("    curves=%d normal=%d abnormal=%d | stage1_AUC=%.4f | anomaly_ratio=%.4f",
                    r["n_curves"], r["n_normal"], r["n_abnormal"], r["stage1_auc"],
                    r["anomaly_ratio"])
        if r["has_label"]:
            LOGGER.info("    cascade P=%.3f R=%.3f F1=%.3f FPR=%.3f",
                        r["cascade_precision"], r["cascade_recall"], r["cascade_f1"],
                        r["cascade_fpr"])

    metrics = {}
    for r in results:
        metrics[r["dataset"]] = _jsonable({k: v for k, v in r.items()
                                           if k not in ("curves", "det")})
    with open(logs / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    LOGGER.info("metrics.json written (%d datasets)", len(metrics))

    # tables + figures
    from src.visualization import make_all
    make_all(results, tables, figs, LOGGER)
    LOGGER.info("tables & figures written")

    if baselines:
        from src.baselines.runner import run_baselines
        LOGGER.info("==> running model-comparison baselines (Ding)")
        run_baselines(dataset="ding", output=output, use_cache=use_cache,
                      cache_dir=cache_dir, logger=LOGGER)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="TEPAD one-click pipeline")
    ap.add_argument("--datasets", default="all",
                    help="all | cross | ding | ds226 | lareunion")
    ap.add_argument("--output", default="results/", help="output directory")
    ap.add_argument("--baselines", action="store_true",
                    help="additionally run the Ding cross-model comparison (table 6 / figure 7)")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore cached weights in the comparison experiment and retrain")
    ap.add_argument("--cache-dir", default="",
                    help="cache directory for the comparison experiment (default <output>/models)")
    args = ap.parse_args()

    config.ensure_dirs()
    out = Path(args.output)
    if not out.is_absolute():
        out = config.PROJECT_ROOT / out
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir is not None and not cache_dir.is_absolute():
        cache_dir = config.PROJECT_ROOT / cache_dir
    run(resolve_datasets(args.datasets), out, baselines=args.baselines,
        use_cache=not args.no_cache, cache_dir=cache_dir)


if __name__ == "__main__":
    main()
