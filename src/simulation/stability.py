"""Monte Carlo bias/variance experiments for the five TEPAD parameters."""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from .model import TrueParameters, estimate_theta, generate_stationary_samples


PARAMETER_NAMES = ("alpha", "beta", "eta", "sigma0", "sigma_min")
WINDOW_LENGTHS = (10, 20, 30, 50, 100)
NOISE_LEVELS = (0.01, 0.05, 0.1, 0.2)
CONTAMINATION_SIGMAS = (0, 1, 3, 5)
SAMPLE_SIZES = (10, 50, 100, 500, 1000)
REPEATS = 100


def _summary_row(group: str, condition: object, estimates: np.ndarray,
                 truth: np.ndarray, parameter: str) -> Dict[str, object]:
    errors = np.asarray(estimates, dtype=float) - float(truth)
    bias = float(np.mean(errors))
    variance = float(np.var(errors, ddof=1))
    return {
        "analysis": group,
        "condition": condition,
        "parameter": parameter,
        "truth": float(truth),
        "mean_estimate": float(np.mean(estimates)),
        "bias": bias,
        "std_estimate": float(np.std(estimates, ddof=1)),
        "variance": variance,
        "mse": float(np.mean(errors ** 2)),
        "n_repeats": int(len(estimates)),
    }


def _estimate_replicates(seeds: Sequence[int], n: int, theta: TrueParameters,
                         noise_sigma: float = None,
                         contamination_sigma: float = 0.0) -> np.ndarray:
    estimates = []
    for seed in seeds:
        g, p = generate_stationary_samples(seed, n, theta, noise_sigma)
        if contamination_sigma:
            rng = np.random.default_rng(seed + 800000)
            index = int(rng.integers(0, n))
            mu = min(theta.eta * g[index], 1.0)
            sigma = (
                float(noise_sigma) if noise_sigma is not None
                else theta.sigma0 * mu + theta.sigma_min
            )
            p[index] += contamination_sigma * sigma
        estimates.append(estimate_theta(g, p))
    return np.vstack(estimates)


def run_stability(repeats: int = REPEATS, seed: int = 24061006
                  ) -> Dict[str, pd.DataFrame]:
    """Run all estimator stability experiments and return tidy result frames."""
    if repeats < 2:
        raise ValueError("repeats must be at least 2 to estimate variance")
    theta = TrueParameters()
    truth = np.asarray([
        theta.alpha, theta.beta, theta.eta, theta.sigma0, theta.sigma_min,
    ])
    rng = np.random.default_rng(seed)
    rows: Dict[str, List[Dict[str, object]]] = {
        "bias_variance": [],
        "window": [],
        "noise": [],
        "contamination": [],
        "convergence": [],
    }
    max_sample = max(SAMPLE_SIZES)

    baseline_seeds = rng.integers(0, 2 ** 31 - 1, size=repeats)
    estimates = _estimate_replicates(baseline_seeds, 30, theta)
    for j, name in enumerate(PARAMETER_NAMES):
        rows["bias_variance"].append(
            _summary_row("baseline_L30", "L=30", estimates[:, j], truth[j], name)
        )

    for length in WINDOW_LENGTHS:
        seeds = rng.integers(0, 2 ** 31 - 1, size=repeats)
        estimates = _estimate_replicates(seeds, length, theta)
        for j, name in enumerate(PARAMETER_NAMES):
            rows["window"].append(
                _summary_row("window_length", length, estimates[:, j], truth[j], name)
            )

    for noise in NOISE_LEVELS:
        noisy_truth = TrueParameters(
            theta.alpha, theta.beta, theta.eta, 0.0, noise,
        )
        seeds = rng.integers(0, 2 ** 31 - 1, size=repeats)
        estimates = _estimate_replicates(
            seeds, 100, noisy_truth, noise_sigma=noise,
        )
        for j, name in enumerate(PARAMETER_NAMES):
            rows["noise"].append(
                _summary_row("noise_sigma", noise, estimates[:, j],
                             np.asarray([noisy_truth.alpha, noisy_truth.beta,
                                         noisy_truth.eta, noisy_truth.sigma0,
                                         noisy_truth.sigma_min])[j], name)
            )

    for contamination in CONTAMINATION_SIGMAS:
        seeds = rng.integers(0, 2 ** 31 - 1, size=repeats)
        estimates = _estimate_replicates(
            seeds, 100, theta, contamination_sigma=float(contamination),
        )
        for j, name in enumerate(PARAMETER_NAMES):
            rows["contamination"].append(
                _summary_row("spike_contamination", contamination, estimates[:, j],
                             truth[j], name)
            )

    seeds = rng.integers(0, 2 ** 31 - 1, size=repeats)
    for n in SAMPLE_SIZES:
        errors = []
        for seed_value in seeds:
            seed_value = int(seed_value)
            g, p = generate_stationary_samples(seed_value, max_sample, theta)
            estimate = estimate_theta(g[:n], p[:n])
            errors.append(estimate[2] - theta.eta)
        estimates_eta = np.asarray(errors, dtype=float)
        rows["convergence"].append({
            "analysis": "sample_convergence",
            "condition": n,
            "parameter": "eta",
            "truth": theta.eta,
            "mean_estimate": float(theta.eta + np.mean(estimates_eta)),
            "bias": float(np.mean(estimates_eta)),
            "std_estimate": float(np.std(estimates_eta, ddof=1)),
            "variance": float(np.var(estimates_eta, ddof=1)),
            "mse": float(np.mean(estimates_eta ** 2)),
            "n_repeats": int(repeats),
        })

    return {key: pd.DataFrame(value) for key, value in rows.items()}


def write_stability_tables(frames: Dict[str, pd.DataFrame], output_dir) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(output_dir / "simulation_stability_{}.csv".format(name),
                     index=False)

