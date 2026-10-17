"""Global configuration.

Centralized management of **all paths** and **all hyperparameters**; hard-coded
paths/thresholds are forbidden in any other module.
Environment variables may override them: ``TEPAD_DATASETS``, ``TEPAD_RESULTS``.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATASETS_DIR: Path = Path(os.environ.get("TEPAD_DATASETS", str(PROJECT_ROOT / "datasets")))
RESULTS_DIR: Path = Path(os.environ.get("TEPAD_RESULTS", str(PROJECT_ROOT / "results")))
TABLES_DIR: Path = RESULTS_DIR / "tables"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
LOGS_DIR: Path = RESULTS_DIR / "logs"
MODELS_DIR: Path = RESULTS_DIR / "models"   # model weights/score cache for baseline experiments

# Paper asset directory (research_paper is not versioned; figures/tables/data live here)
PAPER_DIR: Path = PROJECT_ROOT / "research_paper"
PAPER_FIGURES_DIR: Path = PAPER_DIR / "figures"
PAPER_TABLES_DIR: Path = PAPER_DIR / "tables"
PAPER_DATA_DIR: Path = PAPER_DIR / "data"

# --------------------------------------------------------------------------- #
# Probabilistic model / parameter estimation hyperparameters
# --------------------------------------------------------------------------- #
VARIANCE_FLOOR: float = 1e-6      # variance floor for the alpha/beta method-of-moments estimate
WINDOW: int = 30                  # sliding window length
STEP: int = 10                    # sliding window step
N_WINDOWS: int = 10               # exactly 10 windows per daily curve (padded with NaN)
AR_MIN_POINTS: int = 4            # minimum number of points required for AR(1) estimation

# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
DAYLIGHT_QUANTILE: float = 0.05   # daylight filter: G > 5% quantile
MIN_DAY_POINTS: int = 10          # minimum number of valid points per daily curve
MAX_ROWS_PER_DAY: int = 864       # over-long group rejection threshold (Ding preprocessing)

# --------------------------------------------------------------------------- #
# Detection thresholds
# --------------------------------------------------------------------------- #
STAGE1_QUANTILE: float = 95.0     # stage 1: 95% empirical quantile of the normal d
STAGE2_Q_LOW: float = 1.0         # stage 2 lower-tail feature quantile (r_min / delta_r_min / corr)
STAGE2_Q_HIGH: float = 99.0       # stage 2 upper-tail feature quantile (f_neg)
RESID_TAU_K: float = 2.0          # residual depth threshold τ = K * std(normal residuals)

# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
DING_LABELS: List[str] = ["normal", "spike", "inverter_fault", "snowy",
                          "cloudy", "low_value", "shading"]
DING_LABEL_MAP: Dict[str, int] = {name: i for i, name in enumerate(DING_LABELS)}

#: La Réunion analysis window, in UTC.
#:
#: The Zenodo record publishes a longer irradiance series (to 2023-04-14) than the
#: inverter series used as the reference. The reported results were obtained on the
#: overlap that ends where the irradiance export used at the time ended, which gives
#: 247 daily curves. Pinning the window keeps that overlap reproducible when a longer
#: dt1 is downloaded.
LAREUNION_TIME_WINDOW: Tuple[str, str] = (
    "2021-10-28 12:37:45+00:00",
    "2022-09-16 01:02:22+00:00",
)

# Main experiment + cross-dataset (TECNALIA excluded: all 5 modules in the released subset are known faults, with no healthy reference)
DATASETS_ALL: List[str] = ["ding", "ds226", "lareunion"]
DATASETS_CROSS: List[str] = ["ds226", "lareunion"]

# --------------------------------------------------------------------------- #
# Baseline experiments (baselines)
# --------------------------------------------------------------------------- #
BASELINE_SEQ_LEN: int = 96         # daily curve resampling length (deep-learning baseline input)
BASELINE_THRESHOLD_QUANTILE: float = STAGE1_QUANTILE  # baseline threshold: 95% quantile of the normal reference
BASELINE_FEATURE_BINS: int = 8     # number of curve bins in the statistical features
BASELINE_EPOCHS: int = 400         # deep-learning baseline training epochs
BASELINE_PATIENCE: int = 25        # early-stopping patience
BASELINE_BATCH: int = 32           # batch size
BASELINE_LR: float = 1e-3          # learning rate
BASELINE_HIDDEN: int = 128         # MLP autoencoder hidden width
BASELINE_LATENT: int = 32          # latent dimension
BASELINE_GRU_HIDDEN: int = 64      # GRU/LSTM hidden width
BASELINE_GRU_DROPOUT: float = 0.2  # MC-dropout probability
BASELINE_MC_PASSES: int = 20       # number of Monte Carlo forward passes for GRU-Bayesian
BASELINE_MAE_BETA: float = 0.5     # uncertainty weight (score = mean + beta * std)
BASELINE_ANOGAN_Z: int = 16        # AnoGAN latent dimension
BASELINE_ANOGAN_EPOCHS: int = 600  # AnoGAN generator/discriminator training epochs
BASELINE_ANOGAN_STEPS: int = 400   # AnoGAN per-sample latent optimization steps
BASELINE_ANOGAN_LAMBDA: float = 0.1  # AnoGAN discriminator loss weight
BASELINE_AT_DMODEL: int = 32       # AnomalyTransformer hidden dimension
BASELINE_AT_HEADS: int = 4         # AnomalyTransformer attention heads
BASELINE_AT_LAYERS: int = 2        # AnomalyTransformer layers
BASELINE_AT_K: float = 3.0         # AnomalyTransformer association-discrepancy loss weight
BASELINE_AT_TEMPERATURE: float = 1.0  # softmax temperature for AnomalyTransformer anomaly scores
BASELINE_GA_POP: int = 4           # AE-LSTM-GA population size
BASELINE_GA_GENS: int = 3          # AE-LSTM-GA generations
BASELINE_GA_EPOCHS: int = 120      # AE-LSTM-GA individual evaluation epochs

# --------------------------------------------------------------------------- #
# Streaming real-time detection
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Simulated fault injection
# --------------------------------------------------------------------------- #
SIMULATION_TICK_MINUTES: int = 1
SIMULATION_DAY_TICKS: int = 720
SIMULATION_REPEATS: int = 100
SIMULATION_CALIBRATION_DAYS: int = 24
SIMULATION_SEED: int = 601006
#: Window-parameter smoothing factor for calibration (EWMA over the per-window estimates of healthy days).
EWMA_ALPHA_CALIBRATION: float = 0.95
#: Per-tick parameter smoothing factor of the causal dual-window detector.
EWMA_ALPHA_DETECTOR: float = 0.1
SIMULATION_SPIKE_SIGMAS = (2.0, 3.5, 5.0)
SIMULATION_SHADING_FACTORS = (0.70, 0.50, 0.30)
SIMULATION_SHADING_DURATIONS = (10, 60)
SIMULATION_ANCHOR_FRACTIONS = (0.20, 0.40, 0.60)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED: int = 42

# --------------------------------------------------------------------------- #
# Plots / tables
# --------------------------------------------------------------------------- #
FIG_SINGLE_COL: float = 3.5       # single-column figure width (inches)
FIG_DOUBLE_COL: float = 7.0       # double-column figure width (inches)
FIG_DPI: int = 300                # PNG resolution
#: Whether the figure text-overlap check is a hard failure condition. Kept on
#: for the paper runs (N=100); set TEPAD_FIG_STRICT_OVERLAP=0 in development or
#: smoke runs so that overlaps only warn instead of aborting.
FIG_STRICT_OVERLAP: bool = (
    os.environ.get("TEPAD_FIG_STRICT_OVERLAP", "1").lower() not in ("0", "false", "no")
)
FONT_FAMILY: str = "Times New Roman"
FONT_SIZE: float = 9.0            # uniform font size for axes/legends
TITLE_SIZE: float = 10.0


def set_seed(seed: int = SEED) -> None:
    """Fix all random seeds so that results are reproducible.

    Parameters
    ----------
    seed : int
        Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dirs() -> None:
    """Create all output directories if they do not exist."""
    for d in (DATASETS_DIR, RESULTS_DIR, TABLES_DIR, FIGURES_DIR, LOGS_DIR, MODELS_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)


def ensure_paper_dirs() -> None:
    """Create the paper asset directories (``research_paper/{figures,tables,data}``)."""
    for d in (PAPER_FIGURES_DIR, PAPER_TABLES_DIR, PAPER_DATA_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)


def dataset_raw_dir(name: str) -> Path:
    """Return the ``raw/`` directory of a dataset.

    Parameters
    ----------
    name : str
        Dataset name, e.g. ``"ding"``.
    """
    return DATASETS_DIR / name / "raw"


def dataset_processed_dir(name: str) -> Path:
    """Return the ``processed/`` directory of a dataset."""
    return DATASETS_DIR / name / "processed"


if __name__ == "__main__":  # pragma: no cover - minimal example
    set_seed()
    ensure_dirs()
    print("PROJECT_ROOT :", PROJECT_ROOT)
    print("DATASETS_DIR :", DATASETS_DIR)
    print("RESULTS_DIR  :", RESULTS_DIR)
    print("SEED         :", SEED)
