// FrameStream.cpp : line splitting and frame parsing implementation.
#include "FrameStream.h"

#include <cctype>
#include <cstdlib>
#include <sstream>

namespace tepad {

namespace {

std::string Trim(const std::string& text) {
    std::size_t begin = 0;
    std::size_t end = text.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(text[begin]))) ++begin;
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) --end;
    return text.substr(begin, end - begin);
}

bool IsHeaderLine(const std::string& line) {
    std::string lower;
    lower.reserve(line.size());
    for (std::size_t i = 0; i < line.size(); ++i) {
        lower.push_back(static_cast<char>(
            std::tolower(static_cast<unsigned char>(line[i]))));
    }
    return lower == "tick,g,p" || lower == "tick,g,p,point_z,short_mahalanobis,"
                                "long_mahalanobis,score,alarm";
}

}  // namespace

LineSplitter::LineSplitter(std::size_t maxLineLength) : maxLineLength_(maxLineLength) {}

std::vector<std::string> LineSplitter::Feed(const char* data, std::size_t length) {
    std::vector<std::string> lines;
    for (std::size_t i = 0; i < length; ++i) {
        const char c = data[i];
        if (c == '\n') {
            // Strip the trailing \r for CRLF compatibility.
            if (!buffer_.empty() && buffer_[buffer_.size() - 1] == '\r') buffer_.erase(buffer_.size() - 1);
            lines.push_back(buffer_);
            buffer_.clear();
            continue;
        }
        buffer_.push_back(c);
        if (buffer_.size() >= maxLineLength_) {
            // Overlong data is treated as dirty: drop it and return it as one line (the parser yields an invalid frame).
            lines.push_back(buffer_);
            buffer_.clear();
        }
    }
    return lines;
}

void LineSplitter::Reset() { buffer_.clear(); }

Frame ParseFrameLine(const std::string& line) {
    Frame frame;
    frame.raw = line;
    const std::string trimmed = Trim(line);
    if (trimmed.empty() || IsHeaderLine(trimmed)) return frame;  // valid=false, silently skipped

    std::vector<std::string> fields;
    std::size_t start = 0;
    while (true) {
        const std::size_t comma = trimmed.find(',', start);
        fields.push_back(trimmed.substr(start, comma == std::string::npos ? std::string::npos
                                                                          : comma - start));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    if (fields.size() < 3) return frame;

    char* endTick = nullptr;
    char* endG = nullptr;
    char* endP = nullptr;
    const std::string f0 = Trim(fields[0]);
    const std::string f1 = Trim(fields[1]);
    const std::string f2 = Trim(fields[2]);
    const long long tick = std::strtoll(f0.c_str(), &endTick, 10);
    const double g = std::strtod(f1.c_str(), &endG);
    const double p = std::strtod(f2.c_str(), &endP);
    const bool wellFormed = endTick == f0.c_str() + f0.size() &&
                            endG == f1.c_str() + f1.size() &&
                            endP == f2.c_str() + f2.size() &&
                            !f0.empty() && !f1.empty() && !f2.empty();
    if (!wellFormed) return frame;

    frame.tick = tick;
    frame.g = g;
    frame.p = p;
    frame.valid = true;
    return frame;
}

std::string FormatFixed(double value, int precision) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(precision);
    oss << value;
    return oss.str();
}

}  // namespace tepad
