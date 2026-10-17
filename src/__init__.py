"""TEPAD: a two-level endogenous-exogenous probabilistic anomaly detection framework.

Module layout
-------------
- ``src.io``            : dataset loaders
- ``src.preprocess``    : daily splitting / normalization / filtering
- ``src.model``         : probabilistic model, parameter estimation, residuals, external reference
- ``src.detection``     : stage 1 / stage 2 / cascade
- ``src.evaluation``    : metrics, per-class recall, cross-dataset summary
- ``src.pipeline``      : one-command pipeline
- ``src.visualization`` : paper figure and table generation
"""

__version__ = "1.0.0"
