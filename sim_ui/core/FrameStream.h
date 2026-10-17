// FrameStream.h : serial/network byte stream -> CSV lines -> tick,G,P data frames.
//
// Protocol (same as scripts/serial_bridge.py): one frame per line, "tick,G,P", terminated by \n;
// a tick,G,P header line and empty lines are allowed and skipped during parsing.
#pragma once

#include <string>
#include <vector>

namespace tepad {

/// One measurement frame.
struct Frame {
    long long tick = -1;
    double g = 0.0;
    double p = 0.0;
    bool valid = false;
    std::string raw;  ///< raw line text, used for logging
};

/// Split an arbitrarily fragmented byte stream into complete lines (handles \r\n and \n, and limits line length to avoid dirty-data buildup).
class LineSplitter {
public:
    explicit LineSplitter(std::size_t maxLineLength = 4096);

    /// Append received bytes and return the complete lines gathered this time (without the line terminator).
    std::vector<std::string> Feed(const char* data, std::size_t length);

    /// Clear the partial-line buffer when the connection drops.
    void Reset();

private:
    std::string buffer_;
    std::size_t maxLineLength_;
};

/// Parse one "tick,G,P" line; a header/empty line returns valid=false and is not treated as an error.
Frame ParseFrameLine(const std::string& line);

/// Format a double into a fixed-length string for UI/log display.
std::string FormatFixed(double value, int precision = 4);

}  // namespace tepad
