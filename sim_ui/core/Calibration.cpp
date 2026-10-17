// Calibration.cpp : Mahalanobis scoring and calibration JSON loading implementation.
#include "Calibration.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>

#ifdef _WIN32
#include <windows.h>  // MultiByteToWideChar: UTF-8 path -> wide-character path
#endif

namespace tepad {

namespace {

std::string FormatNumber(double value, int precision = 6) {
    std::ostringstream oss;
    oss.precision(precision);
    oss << value;
    return oss.str();
}

bool CopyVector(const std::vector<double>& src, double* dst, const char* name,
                std::string* error) {
    if (src.size() != static_cast<std::size_t>(kFeatureDim)) {
        if (error != nullptr) {
            *error = std::string(name) + " requires " + std::to_string(kFeatureDim) +
                     " dimensions, got " + std::to_string(src.size()) + " dimensions";
        }
        return false;
    }
    for (int i = 0; i < kFeatureDim; ++i) dst[i] = src[i];
    return true;
}

}  // namespace

double MahalanobisReference::Score(const double features[kFeatureDim]) const {
    // Mirrors MahalanobisReference.score: first robustly standardize and clip to [-zClip, zClip],
    // then subtract the standardized center and finally compute sqrt(d' * inv(Cov) * d).
    double diff[kFeatureDim];
    for (int i = 0; i < kFeatureDim; ++i) {
        // The scale stored in the calibration file was already zero-guarded on the Python side (where(s>1e-8, s, 1)),
        // so divide by scale directly here and stay step-by-step identical to MahalanobisReference.score.
        double z = (features[i] - center[i]) / scale[i];
        if (z < -12.0) z = -12.0;
        if (z > 12.0) z = 12.0;
        diff[i] = z - standardizedCenter[i];
    }
    double quadratic = 0.0;
    for (int i = 0; i < kFeatureDim; ++i) {
        double rowSum = 0.0;
        for (int j = 0; j < kFeatureDim; ++j) {
            rowSum += inverseCovariance[i][j] * diff[j];
        }
        quadratic += diff[i] * rowSum;
    }
    if (!(quadratic > 0.0)) quadratic = 0.0;  // NaN takes this branch too, matching np.maximum
    return std::sqrt(quadratic);
}

bool FillReference(const Json& node, MahalanobisReference* out, const char* name,
                   std::string* error) {
    if (!node.isObject()) {
        if (error != nullptr) *error = std::string(name) + ": missing object node";
        return false;
    }
    if (!CopyVector(node["center"].asDoubleVector(), out->center,
                    (std::string(name) + " center").c_str(), error)) return false;
    if (!CopyVector(node["scale"].asDoubleVector(), out->scale,
                    (std::string(name) + " scale").c_str(), error)) return false;
    if (!CopyVector(node["standardized_center"].asDoubleVector(), out->standardizedCenter,
                    (std::string(name) + " standardized_center").c_str(), error)) return false;

    const std::vector<std::vector<double>> matrix = node["inverse_covariance"].asDoubleMatrix();
    if (matrix.size() != static_cast<std::size_t>(kFeatureDim)) {
        if (error != nullptr) {
            *error = std::string(name) + " inverse_covariance requires " +
                     std::to_string(kFeatureDim) + " rows, got " + std::to_string(matrix.size());
        }
        return false;
    }
    for (int i = 0; i < kFeatureDim; ++i) {
        if (matrix[i].size() != static_cast<std::size_t>(kFeatureDim)) {
            if (error != nullptr) {
                *error = std::string(name) + " inverse_covariance row " + std::to_string(i) +
                         " has wrong width (expected " + std::to_string(kFeatureDim) + ")";
            }
            return false;
        }
        for (int j = 0; j < kFeatureDim; ++j) out->inverseCovariance[i][j] = matrix[i][j];
    }
    out->threshold = node["threshold"].asDouble();
    out->nReference = node["n_reference"].asInt();
    return true;
}

bool ReadTextFile(const std::string& utf8Path, std::string* out, std::string* error) {
    if (out == nullptr) return false;
    out->clear();
#ifdef _WIN32
    wchar_t widePath[MAX_PATH * 4];
    const int converted = MultiByteToWideChar(CP_UTF8, 0, utf8Path.c_str(),
                                              static_cast<int>(utf8Path.size()),
                                              widePath, MAX_PATH * 4 - 1);
    if (converted <= 0) {
        if (error != nullptr) *error = "path is not a valid UTF-8 string";
        return false;
    }
    widePath[converted] = L'\0';
    FILE* handle = nullptr;
    if (_wfopen_s(&handle, widePath, L"rb") != 0 || handle == nullptr) {
        if (error != nullptr) *error = "cannot open file: " + utf8Path;
        return false;
    }
    char chunk[65536];
    std::size_t read = 0;
    while ((read = std::fread(chunk, 1, sizeof(chunk), handle)) > 0) {
        out->append(chunk, read);
    }
    std::fclose(handle);
#else
    std::ifstream stream(utf8Path.c_str(), std::ios::binary);
    if (!stream) {
        if (error != nullptr) *error = "cannot open file: " + utf8Path;
        return false;
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    out->assign(buffer.str());
#endif
    if (out->empty() && error != nullptr) *error = "file is empty: " + utf8Path;
    return !out->empty();
}

bool LoadCalibrationFromText(const std::string& jsonText, Calibration* out, std::string* error) {
    if (out == nullptr) return false;
    Json root;
    std::string parseError;
    if (!Json::Parse(jsonText, &root, &parseError)) {
        if (error != nullptr) *error = "JSON parse failed: " + parseError;
        return false;
    }
    if (!root.isObject()) {
        if (error != nullptr) *error = "JSON root node must be an object";
        return false;
    }

    Calibration parsed;
    parsed.format = root["format"].asString();
    parsed.source = root["source"].asString();

    const Json& detector = root["detector"];
    parsed.shortWindow = detector["short_window"].asInt(10);
    parsed.longWindow = detector["long_window"].asInt(60);
    parsed.ewmaAlpha = detector["ewma_alpha"].asDouble(0.1);
    parsed.zClip = detector["z_clip"].asDouble(12.0);
    parsed.sigmaFloorRate = detector["sigma_floor_rate"].asDouble(0.025);
    parsed.gFloor = detector["g_floor"].asDouble(0.1);

    parsed.etaReference = root["eta_reference"].asDouble();
    parsed.residualSigma = root["residual_sigma"].asDouble();
    parsed.residualThreshold = root["residual_threshold"].asDouble();
    parsed.tauShort = root["tau_short"].asDouble();
    parsed.tauLong = root["tau_long"].asDouble();
    parsed.nCalibrationDays = root["n_calibration_days"].asInt();
    parsed.calibrationSeed = static_cast<long long>(root["calibration_seed"].asDouble());
    parsed.ticksPerDay = root["ticks_per_day"].asInt();

    if (!FillReference(root["short_reference"], &parsed.shortReference, "short_reference", error)) {
        return false;
    }
    if (!FillReference(root["long_reference"], &parsed.longReference, "long_reference", error)) {
        return false;
    }
    if (parsed.ewmaAlpha <= 0.0 || parsed.ewmaAlpha > 1.0) {
        if (error != nullptr) *error = "ewma_alpha must lie in (0,1]";
        return false;
    }
    if (parsed.residualSigma <= 0.0 || parsed.residualThreshold <= 0.0 ||
        parsed.tauShort <= 0.0 || parsed.tauLong <= 0.0) {
        if (error != nullptr) *error = "calibration thresholds incomplete (non-positive values); re-run the export script";
        return false;
    }

    parsed.summary = "eta=" + FormatNumber(parsed.etaReference) +
                     "  z_sigma=" + FormatNumber(parsed.residualSigma) +
                     "  tau_z=" + FormatNumber(parsed.residualThreshold) +
                     "  tau_10=" + FormatNumber(parsed.tauShort) +
                     "  tau_60=" + FormatNumber(parsed.tauLong) +
                     "  (L=" + std::to_string(parsed.shortWindow) + "/L=" +
                     std::to_string(parsed.longWindow) + ", EWMA=" +
                     FormatNumber(parsed.ewmaAlpha, 2) + ", " +
                     std::to_string(parsed.nCalibrationDays) + "-day calibration)";
    parsed.valid = true;
    *out = parsed;
    return true;
}

bool LoadCalibrationFile(const std::string& utf8Path, Calibration* out, std::string* error) {
    std::string text;
    if (!ReadTextFile(utf8Path, &text, error)) return false;
    if (!LoadCalibrationFromText(text, out, error)) return false;
    return true;
}

}  // namespace tepad
