// DualWindowDetector.cpp : detector implementation aligned line by line with Python.
#include "DualWindowDetector.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace tepad {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kSqrtPiOverTwo = 1.25331413731550025519;  // sqrt(pi/2)

double Clip(double value, double low, double high) {
    if (std::isnan(value)) return value;
    if (value < low) return low;
    if (value > high) return high;
    return value;
}

double Mean(const double* x, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += x[i];
    return sum / static_cast<double>(n);
}

/// Sample variance (ddof=1), corresponding to np.var(x, ddof=1).
double VarianceDDOF1(const double* x, int n, double mean) {
    if (n < 2) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        const double d = x[i] - mean;
        sum += d * d;
    }
    return sum / static_cast<double>(n - 1);
}

double MaxOrFloor(double value, double floorValue) {
    return value > floorValue ? value : floorValue;
}

}  // namespace

bool EstimateTheta(const double* gIn, const double* pIn, int n, double out[kFeatureDim]) {
    // Drop non-finite samples, corresponding to valid = np.isfinite(g) & np.isfinite(p).
    std::vector<double> g;
    std::vector<double> p;
    g.reserve(n);
    p.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (std::isfinite(gIn[i]) && std::isfinite(pIn[i])) {
            g.push_back(gIn[i]);
            p.push_back(pIn[i]);
        }
    }
    const int count = static_cast<int>(g.size());
    if (count < 4) {
        for (int i = 0; i < kFeatureDim; ++i) out[i] = std::numeric_limits<double>::quiet_NaN();
        return false;
    }

    const double rawMeanG = Mean(g.data(), count);
    const double varG = VarianceDDOF1(g.data(), count, rawMeanG);
    const double meanG = Clip(rawMeanG, 1e-5, 1.0 - 1e-5);
    const double concentration =
        Clip(meanG * (1.0 - meanG) / MaxOrFloor(varG, 1e-8) - 1.0, 1e-4, 1e5);
    const double alpha = Clip(meanG * concentration, 1e-4, 1e5);
    const double beta = Clip((1.0 - meanG) * concentration, 1e-4, 1e5);

    // centered_g uses the unclipped rawMeanG (consistent with np.mean(g)).
    const double meanP = Mean(p.data(), count);
    double sumGG = 0.0;
    double sumGP = 0.0;
    for (int i = 0; i < count; ++i) {
        const double centeredG = g[i] - rawMeanG;
        sumGG += centeredG * centeredG;
        sumGP += centeredG * (p[i] - meanP);
    }
    double eta = 0.0;
    if (sumGG > 1e-12) eta = sumGP / sumGG;
    eta = Clip(eta, 0.0, 2.0);

    std::vector<double> mu(static_cast<std::size_t>(count));
    std::vector<double> absResidual(static_cast<std::size_t>(count));
    double sumMu = 0.0;
    double sumAbs = 0.0;
    for (int i = 0; i < count; ++i) {
        mu[i] = eta * g[i] < 1.0 ? eta * g[i] : 1.0;  // np.minimum(eta * g, 1.0)
        absResidual[i] = std::fabs(p[i] - mu[i]) * kSqrtPiOverTwo;
        sumMu += mu[i];
        sumAbs += absResidual[i];
    }
    const double meanMu = sumMu / static_cast<double>(count);
    const double meanAbs = sumAbs / static_cast<double>(count);

    double varMu = 0.0;
    double cross = 0.0;
    for (int i = 0; i < count; ++i) {
        const double centeredMu = mu[i] - meanMu;
        varMu += centeredMu * centeredMu;
        cross += centeredMu * (absResidual[i] - meanAbs);
    }

    double sigma0 = 0.0;
    double sigmaMin = 0.0;
    if (varMu > 1e-12) {
        sigma0 = cross / varMu;
        sigmaMin = meanAbs - sigma0 * meanMu;
    } else {
        sigma0 = 0.0;
        sigmaMin = meanAbs;
    }
    sigma0 = Clip(sigma0, 0.0, 2.0);
    sigmaMin = Clip(sigmaMin, 1e-6, 2.0);

    out[0] = alpha;
    out[1] = beta;
    out[2] = eta;
    out[3] = sigma0;
    out[4] = sigmaMin;
    for (int i = 0; i < kFeatureDim; ++i) {
        if (!std::isfinite(out[i])) return false;
    }
    return true;
}

void FeaturesFromParams(const double params[kFeatureDim], double features[kFeatureDim]) {
    std::memcpy(features, params, sizeof(double) * kFeatureDim);
    // Take the log of alpha / beta / sigma0 / sigma_min; eta keeps its value (model.py indices [0,1,3,4]).
    const int logIndices[4] = {0, 1, 3, 4};
    for (int k = 0; k < 4; ++k) {
        const int i = logIndices[k];
        const double base = MaxOrFloor(features[i], 1e-8);
        features[i] = std::log(base);
    }
}

CausalParameterEstimator::CausalParameterEstimator(int window, double alpha)
    : window_(window < 4 ? 4 : window), alpha_(alpha), hasTheta_(false) {
    for (int i = 0; i < kFeatureDim; ++i) theta_[i] = 0.0;
}

bool CausalParameterEstimator::Step(double g, double p, double theta[kFeatureDim]) {
    g_.push_back(g);
    p_.push_back(p);
    while (static_cast<int>(g_.size()) > window_) {
        g_.pop_front();
        p_.pop_front();
    }
    if (static_cast<int>(g_.size()) < window_) {
        if (theta != nullptr && hasTheta_) std::memcpy(theta, theta_, sizeof(double) * kFeatureDim);
        return false;
    }
    double estimate[kFeatureDim];
    // std::deque does not guarantee contiguous memory, so copy into a temporary vector before calling the estimator.
    std::vector<double> gWindow(g_.begin(), g_.end());
    std::vector<double> pWindow(p_.begin(), p_.end());
    if (!EstimateTheta(gWindow.data(), pWindow.data(), window_, estimate)) {
        // Same as Python: when the estimate is invalid neither the EWMA nor theta is updated.
        if (theta != nullptr && hasTheta_) std::memcpy(theta, theta_, sizeof(double) * kFeatureDim);
        return false;
    }
    if (!hasTheta_) {
        std::memcpy(theta_, estimate, sizeof(double) * kFeatureDim);
        hasTheta_ = true;
    } else {
        for (int i = 0; i < kFeatureDim; ++i) {
            theta_[i] = alpha_ * estimate[i] + (1.0 - alpha_) * theta_[i];
        }
    }
    if (theta != nullptr) std::memcpy(theta, theta_, sizeof(double) * kFeatureDim);
    return true;
}

DualWindowDetector::DualWindowDetector(const Calibration& calibration)
    : calibration_(calibration),
      // The causal estimation inside the detector always uses STREAM_EWMA_ALPHA (ewma_alpha=0.1 in the calibration file),
      // exactly matching the construction of model.py::DualWindowDetector.
      shortEstimator_(calibration.shortWindow, calibration.ewmaAlpha),
      longEstimator_(calibration.longWindow, calibration.ewmaAlpha) {}

void DualWindowDetector::Reset() {
    shortEstimator_ = CausalParameterEstimator(calibration_.shortWindow, calibration_.ewmaAlpha);
    longEstimator_ = CausalParameterEstimator(calibration_.longWindow, calibration_.ewmaAlpha);
}

DetectionResult DualWindowDetector::Step(double g, double p) {
    DetectionResult result;

    // ---- Channel 1: point residual z ---------------------------------------
    const double sigmaPoint = MaxOrFloor(
        calibration_.residualSigma,
        calibration_.sigmaFloorRate * (g > calibration_.gFloor ? g : calibration_.gFloor));
    result.pointZ = std::fabs(p - calibration_.etaReference * g) / sigmaPoint;

    // ---- Channels 2/3: short- and long-window Mahalanobis ------------------
    double thetaShort[kFeatureDim];
    double thetaLong[kFeatureDim];
    const bool shortReady = shortEstimator_.Step(g, p, thetaShort);
    const bool longReady = longEstimator_.Step(g, p, thetaLong);
    result.shortReady = shortReady;
    result.longReady = longReady;
    if (shortReady) {
        std::memcpy(result.thetaShort.value, thetaShort, sizeof(double) * kFeatureDim);
        double features[kFeatureDim];
        FeaturesFromParams(thetaShort, features);
        result.shortMahalanobis = calibration_.shortReference.Score(features);
    }
    if (longReady) {
        std::memcpy(result.thetaLong.value, thetaLong, sizeof(double) * kFeatureDim);
        double features[kFeatureDim];
        FeaturesFromParams(thetaLong, features);
        result.longMahalanobis = calibration_.longReference.Score(features);
    }

    // ---- Three-channel OR alarm + normalized score -------------------------
    result.alarm = result.pointZ > calibration_.residualThreshold ||
                   result.shortMahalanobis > calibration_.tauShort ||
                   result.longMahalanobis > calibration_.tauLong;
    const double normalizedPoint = result.pointZ / MaxOrFloor(calibration_.residualThreshold, 1e-12);
    const double normalizedShort = result.shortMahalanobis / MaxOrFloor(calibration_.tauShort, 1e-12);
    const double normalizedLong = result.longMahalanobis / MaxOrFloor(calibration_.tauLong, 1e-12);
    result.score = normalizedPoint;
    if (normalizedShort > result.score) result.score = normalizedShort;
    if (normalizedLong > result.score) result.score = normalizedLong;
    return result;
}

}  // namespace tepad
