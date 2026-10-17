// DualWindowDetector.h : C++ port of src/simulation/model.py::DualWindowDetector.
//
// Three-channel OR alarm:
//   * point residual z = |P - eta_ref * G| / max(residual_sigma, 0.025 * max(G, 0.1))
//   * short-window Mahalanobis (L=10, causal parameter estimation with EWMA coefficient 0.1)
//   * long-window Mahalanobis (L=60, same as above)
// Aligned line by line with estimate_theta / CausalParameterEstimator / _features_from_params /
// DualWindowDetector.step in model.py, so results can be compared with Python.
#pragma once

#include <deque>

#include "Calibration.h"

namespace tepad {

/// The 5 physical quantities used for parameter estimation: [alpha, beta, eta, sigma0, sigma_min].
struct ParameterVector {
    double value[kFeatureDim] = {0.0};
};

/// Detection result for a single tick (corresponds to the triple returned by model.py step()).
struct DetectionResult {
    double score = 0.0;            ///< maximum of the three normalized channels
    bool alarm = false;            ///< OR of the three channels
    double pointZ = 0.0;           ///< point_z
    double shortMahalanobis = 0.0; ///< short_mahalanobis
    double longMahalanobis = 0.0;  ///< long_mahalanobis
    bool shortReady = false;       ///< whether the short window is full (false during warm-up)
    bool longReady = false;        ///< whether the long window is full
    ParameterVector thetaShort;    ///< current EWMA-smoothed short-window parameters
    ParameterVector thetaLong;     ///< current EWMA-smoothed long-window parameters
};

/// Moment/OLS estimate from the (g, p) series inside the window; returns false when samples are insufficient or non-finite.
bool EstimateTheta(const double* g, const double* p, int n, double out[kFeatureDim]);

/// Take the log of alpha/beta/sigma0/sigma_min (eta keeps its value), corresponding to _features_from_params.
void FeaturesFromParams(const double params[kFeatureDim], double features[kFeatureDim]);

/// Fixed-length sliding window with EWMA smoothing, the causal parameter estimator corresponding to CausalParameterEstimator.
class CausalParameterEstimator {
public:
    CausalParameterEstimator(int window, double alpha);

    /// Feed one frame; returns false while the window is not full or the estimate is invalid (theta is left unchanged).
    bool Step(double g, double p, double theta[kFeatureDim]);

    bool ready() const { return hasTheta_; }
    const double* theta() const { return theta_; }
    int filled() const { return static_cast<int>(g_.size()); }
    int window() const { return window_; }

private:
    int window_;
    double alpha_;
    std::deque<double> g_;
    std::deque<double> p_;
    bool hasTheta_;
    double theta_[kFeatureDim];
};

/// Dual-window plus point-residual three-channel detector.
class DualWindowDetector {
public:
    explicit DualWindowDetector(const Calibration& calibration);

    DetectionResult Step(double g, double p);
    void Reset();

    const Calibration& calibration() const { return calibration_; }
    bool Usable() const { return calibration_.Usable(); }

private:
    Calibration calibration_;
    CausalParameterEstimator shortEstimator_;
    CausalParameterEstimator longEstimator_;
};

}  // namespace tepad
