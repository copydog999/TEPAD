// selftest.cpp : numeric self-test that does not depend on MFC; verifies the core port matches Python.
//
// Usage: selftest.exe [calibration JSON] [golden CSV]
// Defaults to ..\calibration\dual_window_calibration.json and golden\golden_frames.csv.
// The golden CSV is produced by scripts/export_simulation_calibration.py --golden.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../core/Calibration.h"
#include "../core/DualWindowDetector.h"
#include "../core/FrameStream.h"

namespace {

struct GoldenRow {
    long long tick;
    double g;
    double p;
    double pointZ;
    double shortMahalanobis;
    double longMahalanobis;
    double score;
    int alarm;
};

bool LoadGolden(const std::string& path, std::vector<GoldenRow>* rows, std::string* error) {
    std::ifstream stream(path.c_str());
    if (!stream) {
        *error = "cannot open golden CSV: " + path;
        return false;
    }
    std::string line;
    while (std::getline(stream, line)) {
        if (!line.empty() && line[line.size() - 1] == '\r') line.erase(line.size() - 1);
        if (line.empty()) continue;
        std::vector<std::string> fields;
        std::size_t start = 0;
        while (true) {
            const std::size_t comma = line.find(',', start);
            fields.push_back(line.substr(start, comma == std::string::npos
                                                       ? std::string::npos
                                                       : comma - start));
            if (comma == std::string::npos) break;
            start = comma + 1;
        }
        if (fields.size() != 8) continue;
        // The first column of the header line (tick,G,P,...) is not a number, so skip it.
        const char firstChar = fields[0].empty() ? '\0' : fields[0][0];
        if (firstChar != '-' && (firstChar < '0' || firstChar > '9')) continue;
        GoldenRow row;
        row.tick = std::strtoll(fields[0].c_str(), nullptr, 10);
        row.g = std::strtod(fields[1].c_str(), nullptr);
        row.p = std::strtod(fields[2].c_str(), nullptr);
        row.pointZ = std::strtod(fields[3].c_str(), nullptr);
        row.shortMahalanobis = std::strtod(fields[4].c_str(), nullptr);
        row.longMahalanobis = std::strtod(fields[5].c_str(), nullptr);
        row.score = std::strtod(fields[6].c_str(), nullptr);
        row.alarm = std::atoi(fields[7].c_str());
        rows->push_back(row);
    }
    if (rows->empty()) {
        *error = "golden CSV contains no data rows: " + path;
        return false;
    }
    return true;
}

bool CloseEnough(double actual, double expected, double tolerance) {
    if (std::isnan(actual) || std::isnan(expected)) return actual == expected;
    const double scale = std::max(1.0, std::fabs(expected));
    return std::fabs(actual - expected) <= tolerance * scale;
}

int Fail(const std::string& message) {
    std::cerr << "[FAIL] " << message << std::endl;
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string calibrationPath =
        argc > 1 ? argv[1] : "../calibration/dual_window_calibration.json";
    const std::string goldenPath =
        argc > 2 ? argv[2] : "golden/golden_frames.csv";

    tepad::Calibration calibration;
    std::string error;
    if (!tepad::LoadCalibrationFile(calibrationPath, &calibration, &error)) {
        return Fail("failed to load calibration: " + error + " (path: " + calibrationPath + ")");
    }
    std::cout << "[ok] calibration loaded: " << calibrationPath << std::endl;
    std::cout << "     " << calibration.summary << std::endl;

    std::vector<GoldenRow> golden;
    if (!LoadGolden(goldenPath, &golden, &error)) {
        return Fail(error + " (path: " + goldenPath + ")");
    }
    std::cout << "[ok] golden rows: " << golden.size() << std::endl;

    // ---- 1) Compare tick by tick with Python DualWindowDetector.step --------
    tepad::DualWindowDetector detector(calibration);
    double maxDeviation = 0.0;
    for (std::size_t i = 0; i < golden.size(); ++i) {
        const GoldenRow& row = golden[i];
        const tepad::DetectionResult result = detector.Step(row.g, row.p);
        const bool alarmMatched = result.alarm == (row.alarm != 0);
        const double deviation = std::max(
            std::fabs(result.pointZ - row.pointZ),
            std::max(std::fabs(result.shortMahalanobis - row.shortMahalanobis),
                     std::max(std::fabs(result.longMahalanobis - row.longMahalanobis),
                              std::fabs(result.score - row.score))));
        if (deviation > maxDeviation) maxDeviation = deviation;
        if (!alarmMatched || deviation > 1e-6) {
            std::ostringstream oss;
            oss.precision(9);
            oss << "tick " << row.tick << " mismatch: dev=" << deviation << " alarm cpp="
                << (result.alarm ? 1 : 0) << " py=" << row.alarm << " | cpp(z,ms,ml,score)="
                << result.pointZ << "," << result.shortMahalanobis << ","
                << result.longMahalanobis << "," << result.score << " py="
                << row.pointZ << "," << row.shortMahalanobis << "," << row.longMahalanobis
                << "," << row.score;
            return Fail(oss.str());
        }
    }
    std::cout << "[ok] tick-by-tick comparison passed, max deviation = " << maxDeviation << std::endl;

    // ---- 2) Alarm semantics: a normal day must not trigger; injected spike/shading must ----
    int normalAlarms = 0;
    for (std::size_t i = 0; i < golden.size(); ++i) {
        if (golden[i].alarm != 0) ++normalAlarms;
    }
    std::cout << "[info] normal-day alarm ticks = " << normalAlarms << "/" << golden.size()
              << std::endl;

    {
        tepad::DualWindowDetector injected(calibration);
        bool alarmed = false;
        const std::size_t spikeAt = golden.size() / 2;
        for (std::size_t i = 0; i < golden.size(); ++i) {
            double p = golden[i].p;
            if (i == spikeAt) p += 8.0 * calibration.residualSigma;
            const tepad::DetectionResult result = injected.Step(golden[i].g, p);
            if (result.alarm) alarmed = true;
        }
        if (!alarmed) return Fail("spike injection triggered no alarm; the detection logic may be wrong");
        std::cout << "[ok] spike injection (+" << 8 << "sigma @" << spikeAt << ") triggered an alarm" << std::endl;
    }

    {
        tepad::DualWindowDetector shaded(calibration);
        bool alarmed = false;
        const std::size_t start = golden.size() / 2;
        const std::size_t stop = std::min(golden.size(), start + 60);
        for (std::size_t i = 0; i < golden.size(); ++i) {
            double p = golden[i].p;
            if (i >= start && i < stop) p *= 0.30;  // corresponds to inject_shading(factor=0.30)
            const tepad::DetectionResult result = shaded.Step(golden[i].g, p);
            if (result.alarm) alarmed = true;
        }
        if (!alarmed) return Fail("shading injection (0.30 for 60 tick) triggered no alarm");
        std::cout << "[ok] shading injection (factor=0.30, 60 tick) triggered an alarm" << std::endl;
    }

    // ---- 3) Frame parsing: standard frame / header / empty line / CRLF / dirty data ----
    {
        tepad::LineSplitter splitter;
        const char* chunk = "0,0.100000,0.080000\n1,0.200000,0.160000\r\ntick,G,P\n\n";
        const std::vector<std::string> lines = splitter.Feed(chunk, std::strlen(chunk));
        // Lines end with \n: 4 lines result (the trailing \n only terminates the empty line, it does not add another).
        if (lines.size() != 4) {
            std::ostringstream oss;
            oss << "unexpected line count: expected 4, got " << lines.size();
            return Fail(oss.str());
        }
        const tepad::Frame first = tepad::ParseFrameLine(lines[0]);
        const tepad::Frame second = tepad::ParseFrameLine(lines[1]);
        const tepad::Frame header = tepad::ParseFrameLine(lines[2]);
        const tepad::Frame empty = tepad::ParseFrameLine(lines[3]);
        const tepad::Frame junk = tepad::ParseFrameLine("abc,def,ghi");
        if (!first.valid || first.tick != 0 || !CloseEnough(first.g, 0.1, 1e-9)) {
            return Fail("frame parsing: line 1 mismatch");
        }
        if (!second.valid || second.tick != 1) return Fail("frame parsing: line 2 mismatch");
        if (header.valid) return Fail("frame parsing: header line must be skipped");
        if (empty.valid) return Fail("frame parsing: empty line must be skipped");
        if (junk.valid) return Fail("frame parsing: dirty data must be rejected");
        // Split packet: a line without a terminator must be buffered until the \n arrives.
        const std::string head = "2,0.300000,";
        const std::string tail = "0.240000\n";
        if (!splitter.Feed(head.data(), head.size()).empty()) {
            return Fail("frame parsing: an incomplete line must not be emitted early");
        }
        const std::vector<std::string> joined = splitter.Feed(tail.data(), tail.size());
        if (joined.size() != 1) {
            std::ostringstream oss;
            oss << "frame parsing: joining across packets must yield 1 line, got " << joined.size();
            return Fail(oss.str());
        }
        const tepad::Frame third = tepad::ParseFrameLine(joined[0]);
        if (!third.valid || third.tick != 2 || !CloseEnough(third.p, 0.24, 1e-9)) {
            return Fail("frame parsing: the joined line failed to parse");
        }
        std::cout << "[ok] frame parsing (header/empty/CRLF/dirty data/split line) passed" << std::endl;
    }

    // ---- 4) Warm-up semantics: the short window becomes ready before the long one ----
    {
        tepad::DualWindowDetector warmup(calibration);
        tepad::DetectionResult result;
        for (int i = 0; i < calibration.shortWindow - 1; ++i) {
            result = warmup.Step(0.5, 0.4);
        }
        if (result.shortReady) return Fail("short window must not be ready during warm-up");
        result = warmup.Step(0.5, 0.4);
        if (!result.shortReady) return Fail("short window must be ready once filled");
        if (result.longReady) return Fail("long window must not be ready before it is filled");
        for (int i = 0; i < calibration.longWindow - calibration.shortWindow; ++i) {
            result = warmup.Step(0.5, 0.4);
        }
        if (!result.longReady) return Fail("long window must be ready once filled");
        std::cout << "[ok] dual-window warm-up timing (L=10 before L=60) passed" << std::endl;
    }

    std::cout << "\nall self-tests passed." << std::endl;
    return 0;
}
