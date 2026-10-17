// Json.cpp : minimal JSON reader implementation.
#include "Json.h"

#include <cstdlib>

namespace tepad {

namespace {

const Json& NullJson() {
    static const Json kNull;
    return kNull;
}

std::string PositionText(std::size_t offset, const std::string& text) {
    std::size_t line = 1, column = 1;
    for (std::size_t i = 0; i < offset && i < text.size(); ++i) {
        if (text[i] == '\n') {
            ++line;
            column = 1;
        } else {
            ++column;
        }
    }
    return "line " + std::to_string(line) + ", column " + std::to_string(column);
}

}  // namespace

Json::Json() : kind_(JsonKind::Null), bool_(false), num_(0.0) {}

std::size_t Json::size() const {
    if (kind_ == JsonKind::Object) return object_.size();
    if (kind_ == JsonKind::Array) return array_.size();
    return 0;
}

bool Json::contains(const std::string& key) const {
    return kind_ == JsonKind::Object && object_.find(key) != object_.end();
}

const Json& Json::operator[](const std::string& key) const {
    if (kind_ != JsonKind::Object) return NullJson();
    std::map<std::string, Json>::const_iterator it = object_.find(key);
    return it == object_.end() ? NullJson() : it->second;
}

const Json& Json::operator[](std::size_t index) const {
    if (kind_ != JsonKind::Array || index >= array_.size()) return NullJson();
    return array_[index];
}

double Json::asDouble(double fallback) const {
    if (kind_ == JsonKind::Number) return num_;
    if (kind_ == JsonKind::Bool) return bool_ ? 1.0 : 0.0;
    return fallback;
}

int Json::asInt(int fallback) const {
    if (kind_ == JsonKind::Number) return static_cast<int>(num_);
    if (kind_ == JsonKind::Bool) return bool_ ? 1 : 0;
    return fallback;
}

const std::string& Json::asString() const {
    return kind_ == JsonKind::String ? str_ : NullJson().str_;
}

std::vector<double> Json::asDoubleVector() const {
    std::vector<double> out;
    if (kind_ != JsonKind::Array) return out;
    out.reserve(array_.size());
    for (std::size_t i = 0; i < array_.size(); ++i) out.push_back(array_[i].asDouble());
    return out;
}

std::vector<std::vector<double>> Json::asDoubleMatrix() const {
    std::vector<std::vector<double>> out;
    if (kind_ != JsonKind::Array) return out;
    for (std::size_t i = 0; i < array_.size(); ++i) {
        out.push_back(array_[i].asDoubleVector());
    }
    return out;
}

/// Recursive-descent parser: only the syntax subset needed by the calibration file; invalid input fails immediately.
class JsonParser {
public:
    JsonParser(const std::string& text, std::string* error)
        : text_(text), pos_(0), error_(error) {}

    bool ParseRoot(Json* out) {
        SkipWhitespace();
        if (!ParseValue(out)) return false;
        SkipWhitespace();
        if (pos_ != text_.size()) return Fail("trailing characters after JSON value");
        return true;
    }

private:
    bool Fail(const std::string& message) {
        if (error_ != nullptr) {
            *error_ = message + " (" + PositionText(pos_, text_) + ")";
        }
        return false;
    }

    void SkipWhitespace() {
        while (pos_ < text_.size()) {
            const char c = text_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++pos_;
            } else {
                break;
            }
        }
    }

    bool Peek(char* c) {
        if (pos_ >= text_.size()) {
            if (error_ != nullptr) *error_ = "unexpected end of JSON input";
            return false;
        }
        *c = text_[pos_];
        return true;
    }

    bool ParseValue(Json* out) {
        char c;
        if (!Peek(&c)) return false;
        switch (c) {
        case '{':
            return ParseObject(out);
        case '[':
            return ParseArray(out);
        case '"':
            out->kind_ = JsonKind::String;
            return ParseString(&out->str_);
        case 't':
        case 'f':
            return ParseLiteral(out);
        case 'n':
            return ParseLiteral(out);
        default:
            return ParseNumber(out);
        }
    }

    bool ParseObject(Json* out) {
        out->kind_ = JsonKind::Object;
        ++pos_;  // {
        SkipWhitespace();
        char c;
        if (!Peek(&c)) return false;
        if (c == '}') {
            ++pos_;
            return true;
        }
        while (true) {
            SkipWhitespace();
            if (!Peek(&c)) return false;
            if (c != '"') return Fail("expected string key in object");
            std::string key;
            if (!ParseString(&key)) return false;
            SkipWhitespace();
            if (!Peek(&c) || c != ':') return Fail("expected ':' after object key");
            ++pos_;
            SkipWhitespace();
            Json value;
            if (!ParseValue(&value)) return false;
            out->object_[key] = value;
            SkipWhitespace();
            if (!Peek(&c)) return false;
            if (c == ',') {
                ++pos_;
                continue;
            }
            if (c == '}') {
                ++pos_;
                return true;
            }
            return Fail("expected ',' or '}' in object");
        }
    }

    bool ParseArray(Json* out) {
        out->kind_ = JsonKind::Array;
        ++pos_;  // [
        SkipWhitespace();
        char c;
        if (!Peek(&c)) return false;
        if (c == ']') {
            ++pos_;
            return true;
        }
        while (true) {
            SkipWhitespace();
            Json value;
            if (!ParseValue(&value)) return false;
            out->array_.push_back(value);
            SkipWhitespace();
            if (!Peek(&c)) return false;
            if (c == ',') {
                ++pos_;
                continue;
            }
            if (c == ']') {
                ++pos_;
                return true;
            }
            return Fail("expected ',' or ']' in array");
        }
    }

    bool ParseString(std::string* out) {
        ++pos_;  // opening quote
        out->clear();
        while (true) {
            if (pos_ >= text_.size()) return Fail("unterminated string");
            const char c = text_[pos_++];
            if (c == '"') return true;
            if (c != '\\') {
                out->push_back(c);
                continue;
            }
            if (pos_ >= text_.size()) return Fail("unterminated escape");
            const char e = text_[pos_++];
            switch (e) {
            case '"': out->push_back('"'); break;
            case '\\': out->push_back('\\'); break;
            case '/': out->push_back('/'); break;
            case 'b': out->push_back('\b'); break;
            case 'f': out->push_back('\f'); break;
            case 'n': out->push_back('\n'); break;
            case 'r': out->push_back('\r'); break;
            case 't': out->push_back('\t'); break;
            case 'u': {
                // The calibration file is ASCII only; \uXXXX is handled as "skip 4 characters when it cannot be restored",
                // so unknown field escapes can never make the whole parse fail for the UI.
                if (pos_ + 4 > text_.size()) return Fail("truncated \\u escape");
                const std::string hex = text_.substr(pos_, 4);
                pos_ += 4;
                const long code = std::strtol(hex.c_str(), nullptr, 16);
                if (code > 0 && code < 0x80) {
                    out->push_back(static_cast<char>(code));
                } else {
                    out->push_back('?');
                }
                break;
            }
            default:
                return Fail("unknown escape character");
            }
        }
    }

    bool ParseLiteral(Json* out) {
        static const struct { const char* text; JsonKind kind; bool value; } kLiterals[] = {
            {"true", JsonKind::Bool, true},
            {"false", JsonKind::Bool, false},
            {"null", JsonKind::Null, false},
        };
        for (std::size_t i = 0; i < 3; ++i) {
            const std::size_t length = std::char_traits<char>::length(kLiterals[i].text);
            if (pos_ + length <= text_.size() &&
                text_.compare(pos_, length, kLiterals[i].text) == 0) {
                pos_ += length;
                out->kind_ = kLiterals[i].kind;
                out->bool_ = kLiterals[i].value;
                out->num_ = 0.0;
                return true;
            }
        }
        return Fail("invalid literal");
    }

    bool ParseNumber(Json* out) {
        const std::size_t start = pos_;
        if (pos_ < text_.size() && (text_[pos_] == '-' || text_[pos_] == '+')) ++pos_;
        bool digits = false;
        while (pos_ < text_.size()) {
            const char c = text_[pos_];
            if (c >= '0' && c <= '9') {
                digits = true;
                ++pos_;
            } else if (c == '.' || c == 'e' || c == 'E' || c == '-' || c == '+') {
                ++pos_;
            } else {
                break;
            }
        }
        if (!digits) return Fail("invalid number");
        // Use strtod rather than std::stod: avoids exception overhead; out-of-range values become inf and are validated upstream.
        const std::string token = text_.substr(start, pos_ - start);
        char* end = nullptr;
        const double value = std::strtod(token.c_str(), &end);
        if (end != token.c_str() + token.size()) return Fail("malformed number token");
        out->kind_ = JsonKind::Number;
        out->num_ = value;
        return true;
    }

    const std::string& text_;
    std::size_t pos_;
    std::string* error_;
};

bool Json::Parse(const std::string& text, Json* out, std::string* error) {
    if (out == nullptr) return false;
    if (error != nullptr) error->clear();
    JsonParser parser(text, error);
    *out = Json();
    return parser.ParseRoot(out);
}

}  // namespace tepad
