// Calibration.h : in-memory structure for the dual-window detector calibration and JSON loading.
//
// The fields map one-to-one to the output of scripts/export_simulation_calibration.py; the numeric semantics
// are copied verbatim from src/streaming/detectors.py::MahalanobisReference and
// src/simulation/model.py::calibrate_dual_window.
#pragma once

#include <string>

#include "Json.h"

namespace tepad {

/// Parameter feature dimension: the 5-dimensional vector of [alpha, beta, eta, sigma0, sigma_min] after the log transform.
constexpr int kFeatureDim = 5;

/// Mahalanobis reference: robust standardization (center/scale) + clipped mean/inverse covariance.
struct MahalanobisReference {
    double center[kFeatureDim] = {0.0};
    double scale[kFeatureDim] = {1.0};
    double standardizedCenter[kFeatureDim] = {0.0};
    double inverseCovariance[kFeatureDim][kFeatureDim] = {0.0};
    double threshold = 0.0;
    int nReference = 0;

    /// score = sqrt(d' * Sigma^-1 * d), d = clip((f-center)/scale, -12, 12) - zcenter.
    double Score(const double features[kFeatureDim]) const;
};

/// All constants required for DualWindowDetector inference.
struct Calibration {
    std::string format;
    std::string source;
    int shortWindow = 10;
    int longWindow = 60;
    double ewmaAlpha = 0.1;
    double zClip = 12.0;
    double sigmaFloorRate = 0.025;
    double gFloor = 0.1;

    double etaReference = 0.0;
    double residualSigma = 0.0;
    double residualThreshold = 0.0;
    double tauShort = 0.0;
    double tauLong = 0.0;

    MahalanobisReference shortReference;
    MahalanobisReference longReference;

    int nCalibrationDays = 0;
    long long calibrationSeed = 0;
    int ticksPerDay = 0;

    bool valid = false;
    std::string summary;  // one-line summary for display in the UI

    /// Whether all three channel thresholds are ready (loaded successfully and thresholds non-zero).
    bool Usable() const {
        return valid && residualThreshold > 0.0 && tauShort > 0.0 && tauLong > 0.0 &&
               residualSigma > 0.0;
    }
};

/// Read file -> parse -> fill Calibration; on failure error carries the reason.
bool LoadCalibrationFile(const std::string& utf8Path, Calibration* out, std::string* error);

/// Load from already-parsed JSON text (used by the self-test and unit tests).
bool LoadCalibrationFromText(const std::string& jsonText, Calibration* out, std::string* error);

/// Write the reference object of the JSON into MahalanobisReference; reports an error on dimension mismatch.
bool FillReference(const Json& node, MahalanobisReference* out, const char* name,
                   std::string* error);

/// Read the whole file into a string (UTF-8 text or binary).
bool ReadTextFile(const std::string& utf8Path, std::string* out, std::string* error);

}  // namespace tepad
