"""Simulation-derived, three-way PV anomaly classification by parameter shifts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .. import config
from ..io import get_loader
from ..preprocess.daily_split import DailyCurve, build_curves
from ..simulation.model import TrueParameters, estimate_theta, generate_stationary_samples

CLASS_ORDER: Tuple[str, ...] = ("hardware", "weather", "shading")
PARAMETER_NAMES: Tuple[str, ...] = ("alpha", "beta", "eta", "sigma0", "r_min")
PARAMETER_LABELS_ZH: Mapping[str, str] = {
    "alpha": "α",
    "beta": "β",
    "eta": "η",
    "sigma0": "σ₀",
    "r_min": "r_min",
}
_NORMAL = "normal"
_CLASS_INDEX = {name: index for index, name in enumerate(CLASS_ORDER)}

DING_LABEL_GROUPS: Mapping[str, str] = {
    "spike": "hardware",
    "inverter fault": "hardware",
    "low value": "hardware",
    "cloudy": "weather",
    "snowy": "weather",
    "shading": "shading",
}
DS226_LABEL_GROUPS: Mapping[str, str] = {
    "degradation": "hardware",
    "open circuit": "hardware",
    "short circuit": "hardware",
    "shading": "shading",
}
LAREUNION_LABEL_GROUPS: Mapping[str, str] = {
    "1": "shading",
    "2.1": "shading",
    "2.2": "shading",
    "2.3": "shading",
    "3.1": "shading",
    "3.2": "shading",
    "4": "shading",
}
DATASET_LABEL_GROUPS: Mapping[str, Mapping[str, str]] = {
    "ding": DING_LABEL_GROUPS,
    "ds226": DS226_LABEL_GROUPS,
    "lareunion": LAREUNION_LABEL_GROUPS,
}


@dataclass(frozen=True)
class Signature:
    """Five estimated parameters; ``r_min`` is the minimum reference residual."""

    alpha: float
    beta: float
    eta: float
    sigma0: float
    r_min: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [self.alpha, self.beta, self.eta, self.sigma0, self.r_min],
            dtype=float,
        )


@dataclass(frozen=True)
class SimulationClassifier:
    """Normal-centered simulation prototypes and scales, fitted without datasets."""

    center: np.ndarray
    scale: np.ndarray
    prototypes: Mapping[str, np.ndarray]


def _signature(g: np.ndarray, p: np.ndarray, eta_reference: float) -> Signature:
    params = estimate_theta(g, p)
    if not np.isfinite(params).all():
        raise ValueError("parameter estimation returned a non-finite value")
    residual = np.asarray(p, dtype=float) - float(eta_reference) * np.asarray(g, dtype=float)
    return Signature(*(float(value) for value in params[:4]),
                     r_min=float(np.min(residual)))


def _draw_signature(seed: int, kind: str, n_points: int,
                    normal_eta: float) -> Signature:
    normal = TrueParameters()
    if kind == "weather":
        theta = TrueParameters(alpha=4.0, beta=5.0, eta=normal.eta,
                               sigma0=normal.sigma0, sigma_min=normal.sigma_min)
        g, p = generate_stationary_samples(seed, n_points, theta)
    elif kind == "hardware":
        theta = TrueParameters(alpha=normal.alpha, beta=normal.beta, eta=0.52,
                               sigma0=normal.sigma0, sigma_min=normal.sigma_min)
        g, p = generate_stationary_samples(seed, n_points, theta)
        rng = np.random.default_rng(seed + 19)
        spike = int(rng.integers(n_points))
        p[spike] += 8.0 * (theta.sigma0 * theta.eta * g[spike] + theta.sigma_min)
    elif kind == "shading":
        g, p = generate_stationary_samples(seed, n_points, normal)
        rng = np.random.default_rng(seed + 23)
        duration = max(1, int(round(0.08 * n_points)))
        start = int(rng.integers(0, n_points - duration + 1))
        p[start:start + duration] *= 0.15
    elif kind == _NORMAL:
        g, p = generate_stationary_samples(seed, n_points, normal)
    else:
        raise ValueError("unknown simulation class: {!r}".format(kind))
    return _signature(g, p, normal_eta)


def _as_matrix(signatures: Sequence[Signature]) -> np.ndarray:
    matrix = np.vstack([signature.as_array() for signature in signatures])
    if not np.isfinite(matrix).all():
        raise ValueError("parameter signatures must all be finite")
    return matrix


def fit_simulation_classifier(seed: int = 20261011, repeats: int = 100,
                              n_points: int = 2400) -> SimulationClassifier:
    """Fit robust normal scales and anomaly prototypes on simulated training samples."""
    if repeats < 2 or n_points < 100:
        raise ValueError("simulation requires at least 2 repeats and 100 points")
    normal_eta = TrueParameters().eta
    samples: Dict[str, List[Signature]] = {name: [] for name in (_NORMAL,) + CLASS_ORDER}
    for kind_index, kind in enumerate(samples):
        for repeat in range(repeats):
            sample_seed = seed + kind_index * 100_000 + repeat
            samples[kind].append(
                _draw_signature(sample_seed, kind, n_points, normal_eta)
            )

    normal_matrix = _as_matrix(samples[_NORMAL])
    center = np.median(normal_matrix, axis=0)
    mad_scale = 1.4826 * np.median(np.abs(normal_matrix - center), axis=0)
    std_scale = np.std(normal_matrix, axis=0, ddof=1)
    scale = np.where(mad_scale > 1e-8, mad_scale, std_scale)
    scale = np.maximum(scale, np.asarray([1e-4, 1e-4, 1e-4, 1e-5, 1e-5]))

    prototypes: Dict[str, np.ndarray] = {}
    for kind in CLASS_ORDER:
        class_matrix = _as_matrix(samples[kind])
        prototypes[kind] = np.median((class_matrix - center) / scale, axis=0)
    return SimulationClassifier(center=center, scale=scale, prototypes=prototypes)


def predict_signatures(signatures: Sequence[Signature],
                       classifier: SimulationClassifier,
                       normal_center: Optional[np.ndarray] = None
                       ) -> List[str]:
    """Assign the nearest simulated class prototype in normal-scaled coordinates."""
    matrix = _as_matrix(signatures)
    center = classifier.center if normal_center is None else np.asarray(normal_center)
    if center.shape != classifier.center.shape or not np.isfinite(center).all():
        raise ValueError("normal calibration center has the wrong shape or non-finite values")
    features = (matrix - center) / classifier.scale
    prototype_matrix = np.vstack([classifier.prototypes[name] for name in CLASS_ORDER])
    distances = np.sum((features[:, None, :] - prototype_matrix[None, :, :]) ** 2, axis=2)
    return [CLASS_ORDER[index] for index in np.argmin(distances, axis=1)]


def run_simulation_experiment(seed: int = 20261011, repeats: int = 100,
                              n_points: int = 2400
                              ) -> Dict[str, object]:
    """Fit on simulated samples, then report shifts and held-out three-way metrics."""
    train_seed = seed
    test_seed = seed + 10_000_000
    classifier = fit_simulation_classifier(train_seed, repeats, n_points)
    normal_eta = TrueParameters().eta
    test: Dict[str, List[Signature]] = {name: [] for name in (_NORMAL,) + CLASS_ORDER}
    for kind_index, kind in enumerate(test):
        for repeat in range(repeats):
            sample_seed = test_seed + kind_index * 100_000 + repeat
            test[kind].append(_draw_signature(sample_seed, kind, n_points, normal_eta))

    normal_center = np.median(_as_matrix(test[_NORMAL]), axis=0)
    shifts: Dict[str, Dict[str, float]] = {}
    y_true: List[str] = []
    y_pred: List[str] = []
    for kind in CLASS_ORDER:
        matrix = _as_matrix(test[kind])
        delta = matrix - normal_center
        shifts[kind] = {
            name: float(np.median(delta[:, index]))
            for index, name in enumerate(PARAMETER_NAMES)
        }
        y_true.extend([kind] * repeats)
        y_pred.extend(predict_signatures(test[kind], classifier, normal_center))
    confusion = confusion_matrix(y_true, y_pred)
    recalls = _recalls(confusion)
    return {
        "seed": seed,
        "repeats_per_class": repeats,
        "points_per_curve": n_points,
        "parameter_shifts": shifts,
        "normal_scale": classifier.scale,
        "confusion_matrix": confusion,
        "recalls": recalls,
        "normal_center": normal_center,
        "classifier": classifier,
    }


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str]) -> np.ndarray:
    if len(y_true) != len(y_pred):
        raise ValueError("true and predicted labels must have the same length")
    counts = np.zeros((len(CLASS_ORDER), len(CLASS_ORDER)), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        if actual not in _CLASS_INDEX or predicted not in _CLASS_INDEX:
            raise ValueError("confusion matrix received an unknown class")
        counts[_CLASS_INDEX[actual], _CLASS_INDEX[predicted]] += 1
    return counts


def _recalls(matrix: np.ndarray) -> Dict[str, Optional[float]]:
    return {
        name: (float(matrix[index, index] / matrix[index].sum())
               if matrix[index].sum() else None)
        for index, name in enumerate(CLASS_ORDER)
    }


def _canonical_label(label: str) -> str:
    text = str(label).strip().lower()
    try:
        number = float(text)
    except ValueError:
        return " ".join(text.replace("_", " ").split())
    return "{:g}".format(number)


def _curve_signature(curve: DailyCurve, eta_reference: float) -> Signature:
    if curve.pref is not None:
        pref = np.asarray(curve.pref, dtype=float)
        if pref.shape != curve.p.shape or not np.isfinite(pref).all():
            raise ValueError("invalid external power reference for {!r}".format(curve.date))
        params = estimate_theta(curve.g, curve.p)
        if not np.isfinite(params).all():
            raise ValueError("parameter estimation failed for {!r}".format(curve.date))
        residual = curve.p - pref
        return Signature(*(float(value) for value in params[:4]),
                         r_min=float(np.min(residual)))
    return _signature(curve.g, curve.p, eta_reference)


def run_dataset_experiment(classifier: SimulationClassifier,
                           names: Sequence[str] = ("ding", "ds226", "lareunion")
                           ) -> List[Dict[str, object]]:
    """Classify labeled dataset curves; labels are applied only after prediction."""
    results: List[Dict[str, object]] = []
    for dataset_name in names:
        if dataset_name not in DATASET_LABEL_GROUPS:
            raise ValueError("no validation label map is defined for {!r}".format(dataset_name))
        frame, spec = get_loader(dataset_name)()
        curves = build_curves(frame, dataset_name, **spec)
        if not curves:
            raise ValueError("no valid daily curves found for {!r}".format(dataset_name))

        signatures = [
            _curve_signature(curve, float(TrueParameters().eta))
            for curve in curves
        ]
        label_groups = DATASET_LABEL_GROUPS[dataset_name]
        # DS226 has no measured reference power. Estimate its healthy conversion
        # reference without labels, then recompute those residual depths.
        if dataset_name == "ds226":
            eta_reference = float(np.median([
                signature.eta for signature in signatures
            ]))
            signatures = [
                _signature(curve.g, curve.p, eta_reference) if curve.pref is None
                else signature
                for curve, signature in zip(curves, signatures)
            ]

        dataset_center = np.median(_as_matrix(signatures), axis=0)
        predicted = predict_signatures(signatures, classifier, dataset_center)
        actual: List[str] = []
        validated_predictions: List[str] = []
        label_counts: Dict[str, int] = {}
        unmapped: Dict[str, int] = {}
        for curve, prediction in zip(curves, predicted):
            canonical = _canonical_label(curve.cls)
            if curve.label == 0:
                continue
            group = label_groups.get(canonical)
            label_counts[curve.cls] = label_counts.get(curve.cls, 0) + 1
            if group is None:
                unmapped[curve.cls] = unmapped.get(curve.cls, 0) + 1
                continue
            actual.append(group)
            validated_predictions.append(prediction)
        if unmapped:
            raise ValueError(
                "unmapped abnormal labels in {!r}: {}".format(dataset_name, unmapped)
            )

        matrix = confusion_matrix(actual, validated_predictions)
        distribution_rows = []
        for curve, signature in zip(curves, signatures):
            canonical = _canonical_label(curve.cls)
            mapped_group = label_groups.get(canonical)
            distribution_rows.append({
                "dataset": dataset_name,
                "label": curve.cls,
                "class": "normal" if curve.label == 0 else (mapped_group or "unmapped"),
                "alpha": signature.alpha,
                "beta": signature.beta,
                "eta": signature.eta,
                "sigma0": signature.sigma0,
                "r_min": signature.r_min,
            })
        results.append({
            "dataset": dataset_name,
            "n_curves": len(curves),
            "n_normal": int(sum(curve.label == 0 for curve in curves)),
            "label_counts": label_counts,
            "class_counts": {
                name: int(sum(1 for label in actual if label == name))
                for name in CLASS_ORDER
            },
            "confusion_matrix": matrix,
            "recalls": _recalls(matrix),
            "parameter_distributions": distribution_rows,
        })
    return results


def direction_checks(shifts: Mapping[str, Mapping[str, float]]) -> Dict[str, bool]:
    """Check the simulated efficiency, irradiance-shape, and negative-depth changes."""
    hardware = shifts["hardware"]
    weather = shifts["weather"]
    shading = shifts["shading"]
    return {
        "hardware_eta_decreases": hardware["eta"] < 0.0,
        "weather_alpha_changes": abs(weather["alpha"]) > 1e-3,
        "weather_beta_changes": abs(weather["beta"]) > 1e-3,
        "shading_r_min_decreases": shading["r_min"] < 0.0,
    }
