"""``src.model``: endogenous probabilistic model, parameter estimation, residuals, and external reference.

Submodules (imported explicitly on demand):

* :mod:`src.model.probabilistic`    —— $\mu(g)$, $\sigma(g)$, 5-parameter estimation
* :mod:`src.model.param_estimation` —— sliding-window flattening, local residuals, AR
* :mod:`src.model.residual`         —— residual thresholds and shape features
* :mod:`src.model.reference`        —— $P^{\\mathrm{ref}}$ source priority
* :mod:`src.model.mahalanobis`      —— reusable online Mahalanobis reference model
"""
