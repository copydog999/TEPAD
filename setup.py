"""Installable package configuration for TEPAD.

Install with ``pip install -e .`` (run from the project root).
"""
from setuptools import find_packages, setup

setup(
    name="tepad",
    version="1.0.0",
    description="TEPAD: Two-level Endogenous-Exogenous Probabilistic Anomaly Detection for PV systems",
    author="TEPAD contributors",
    packages=find_packages(include=["src", "src.*"]),
    package_dir={"": "."},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21,<1.25",
        "pandas>=1.3,<2.1",
        "scikit-learn>=1.0,<1.4",
        "scipy>=1.7,<1.11",
        "matplotlib>=3.5,<3.8",
        "lightgbm>=4.0,<4.6",
        "torch>=1.13",
    ],
)
