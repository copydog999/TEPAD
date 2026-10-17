"""Online Mahalanobis reference for windowed parameter features.

The cascade detector fits its reference in batch form
(:func:`src.detection.stage1.mahalanobis_distance`). The causal detectors used
by the simulation and desktop demos need the same model as an object that can
be fitted once and then queried tick by tick, which is what
:class:`MahalanobisReference` provides.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import config


@dataclass
class MahalanobisReference:
    """Fitted center, scale, inverse covariance, and empirical threshold."""

    center: np.ndarray
    scale: np.ndarray
    standardized_center: np.ndarray
    inverse_covariance: np.ndarray
    threshold: float
    n_reference: int

    @classmethod
    def fit(cls, features: np.ndarray) -> "MahalanobisReference":
        """Fit the reference on normal feature rows; non-finite rows are dropped."""
        x = np.asarray(features, dtype=float)
        x = x[np.all(np.isfinite(x), axis=1)]
        dimension = x.shape[1] if x.ndim == 2 else 0
        if len(x) < max(20, dimension + 2):
            raise ValueError(
                "Not enough normal reference windows: {} for {} features".format(
                    len(x), dimension
                )
            )
        center = np.median(x, axis=0)
        scale = np.std(x, axis=0, ddof=1)
        scale = np.where(scale > 1e-8, scale, 1.0)
        z = np.clip((x - center) / scale, -12.0, 12.0)
        z_center = np.mean(z, axis=0)
        covariance = np.atleast_2d(np.cov(z, rowvar=False, ddof=1))
        # Shrinkage regularizes highly correlated parameter and trend features.
        covariance = 0.95 * covariance + 0.05 * np.eye(dimension)
        covariance += 1e-6 * np.eye(dimension)
        inverse = np.linalg.pinv(covariance, rcond=1e-9)
        distances = cls._distances(z, z_center, inverse)
        threshold = float(np.percentile(distances, config.STAGE1_QUANTILE))
        return cls(center, scale, z_center, inverse, threshold, len(x))

    @staticmethod
    def _distances(z: np.ndarray, center: np.ndarray, inverse: np.ndarray) -> np.ndarray:
        diff = np.asarray(z, dtype=float) - center
        quadratic = np.einsum("ij,jk,ik->i", diff, inverse, diff)
        return np.sqrt(np.maximum(quadratic, 0.0))

    def score(self, feature: np.ndarray) -> float:
        """Distance of one feature row from the fitted normal reference."""
        z = np.clip((np.asarray(feature) - self.center) / self.scale, -12.0, 12.0)
        return float(self._distances(
            z.reshape(1, -1), self.standardized_center, self.inverse_covariance
        )[0])


if __name__ == "__main__":  # pragma: no cover - minimal smoke test
    rng = np.random.default_rng(7)
    reference = MahalanobisReference.fit(rng.normal(0, 1, size=(200, 5)))
    print("threshold:", round(reference.threshold, 3),
          "| outlier score:", round(reference.score(np.full(5, 6.0)), 3))
