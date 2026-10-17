"""TEPAD detection: probability-model stage, residual stage, and their cascade.

* :mod:`src.detection.stage1` scores window-parameter features with a
  Mahalanobis distance against the normal reference.
* :mod:`src.detection.stage2` applies the external-reference residual rules.
* :mod:`src.detection.cascade` combines both with :func:`run_two_level`.
"""
