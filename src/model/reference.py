"""Source handling for the external power reference :math:`P^{\\mathrm{ref}}`.

Priority (TEPAD stage 2)
------------------------
1. **Theoretical/modelled power shipped with the dataset** (e.g. Ding's ``modeled_power``);
2. **Healthy twin inverter** (e.g. dt3 ``Pg`` in La Réunion);
3. **Physical reference** :math:`P^{\\mathrm{ref}}=\\min(\\eta g,1)` (fallback when no external reference exists).
"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from .probabilistic import mu_of_g

ReferenceSource = Literal["dataset", "twin_inverter", "physical"]


def reference_from_dataset(pref: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Return the theoretical power field if the dataset provides a finite one; otherwise return ``None``."""
    if pref is None:
        return None
    arr = np.asarray(pref, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).all():
        return None
    return arr


def reference_physical(g: np.ndarray, eta: float) -> np.ndarray:
    """Physical reference :math:`\\min(\\eta g,1)` (fallback when no external reference exists)."""
    return mu_of_g(g, eta)


def resolve_reference(g: np.ndarray, eta: float,
                      pref: Optional[np.ndarray] = None,
                      source_hint: Optional[str] = None) -> "tuple[np.ndarray, ReferenceSource]":
    """Resolve the reference power series according to the priority order.

    Parameters
    ----------
    g : np.ndarray
        Normalized irradiance.
    eta : float
        Global conversion efficiency (used for the physical reference).
    pref : np.ndarray, optional
        External reference provided by the dataset (modelled power or healthy twin inverter).
    source_hint : str, optional
        ``"dataset"`` or ``"twin_inverter"``, used to label the source.

    Returns
    -------
    (pref, source) : tuple
        Reference power series and source label.
    """
    ref = reference_from_dataset(pref)
    if ref is not None:
        src: ReferenceSource = "twin_inverter" if source_hint == "twin_inverter" else "dataset"
        return ref, src
    return reference_physical(g, eta), "physical"


if __name__ == "__main__":  # pragma: no cover - minimal example
    gg = np.linspace(0, 1, 20)
    print(resolve_reference(gg, 0.9)[1])
    print(resolve_reference(gg, 0.9, pref=0.9 * gg)[1])
    print(resolve_reference(gg, 0.9, pref=0.9 * gg, source_hint="twin_inverter")[1])
