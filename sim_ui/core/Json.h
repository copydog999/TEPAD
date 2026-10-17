// Json.h : minimal JSON reader needed to parse the calibration JSON (no MFC / Windows API dependencies).
//
// Only the read direction is implemented: objects, arrays, strings, numbers, booleans, null. Key order does not matter
// (objects are stored in a std::map), which is enough to parse the dual_window_calibration.json produced by
// scripts/export_simulation_calibration.py.
#pragma once

#include <map>
#include <string>
#include <vector>

namespace tepad {

enum class JsonKind { Null, Bool, Number, String, Array, Object };

class Json {
public:
    Json();

    /// Parse a whole text; returns false and sets error on failure (including the error line and column).
    static bool Parse(const std::string& text, Json* out, std::string* error);

    JsonKind kind() const { return kind_; }
    bool isNull() const { return kind_ == JsonKind::Null; }
    bool isObject() const { return kind_ == JsonKind::Object; }
    bool isArray() const { return kind_ == JsonKind::Array; }
    bool isNumber() const { return kind_ == JsonKind::Number; }

    /// Number of object/array elements, 0 for other types.
    std::size_t size() const;

    bool contains(const std::string& key) const;
    /// Object lookup by key; a missing key returns a static Null instance, so chained access is safe.
    const Json& operator[](const std::string& key) const;
    /// Array element by index; out-of-range returns a Null instance.
    const Json& operator[](std::size_t index) const;

    double asDouble(double fallback = 0.0) const;
    int asInt(int fallback = 0) const;
    const std::string& asString() const;

    /// Array -> one-dimensional double sequence; returns empty for non-arrays.
    std::vector<double> asDoubleVector() const;
    /// Array of arrays -> two-dimensional double matrix; ragged rows are truncated to the first row width.
    std::vector<std::vector<double>> asDoubleMatrix() const;

private:
    JsonKind kind_;
    bool bool_;
    double num_;
    std::string str_;
    std::vector<Json> array_;
    std::map<std::string, Json> object_;

    friend class JsonParser;
};

}  // namespace tepad
