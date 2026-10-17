// stablercwa/json.hpp —— 最小的 JSON 读写（只用 C++17 标准库）
//
//   namespace json = stablercwa::json;
//   json::Value v = json::parse(R"({"M": 20, "n": [1.96, 0.0]})");     // 语法有错时抛 json::ParseError
//   int m = 0;
//   bool is_int = v.at("M").to_int(m);                                 // m = 20
//   double re = v.at("n").at(0).as_double();                           // 1.96
//
//   json::Value out = json::Value::object();                           // 对象保持键的插入顺序
//   out.set("ok", true);
//   out.set("DEr", json::Value::array()).push_back(0.25);
//   std::string text = json::dump(out);                                // {"ok": true, "DEr": [0.25]}
//
// 输出很大时不必先建一棵 Value 树，可以用 Writer 边算边写（rcwa_json.hpp 的应答就是这样写出来的）：
//   json::Writer w;
//   w.begin_object();  w.key("R");  w.begin_array();  w.value(0.5);  w.end_array();  w.end_object();
//   std::string text = w.take();                                       // {"R": [0.5]}
//
// 读（parse）：
//   · 语法按 RFC 8259：null、true、false、数、字符串、数组、对象；值之间只允许空格、制表符、换行、回车。
//     不接受注释、单引号、末尾多余的逗号、NaN、Infinity、十六进制数、前导零；整段文字只能有一个值，后面还有别的字符算错。
//     开头的 UTF-8 字节序标记（BOM）跳过不计。
//   · 数一律按 double 读（正确舍入）。超出 double 范围的数（如 1e999）算错，不会悄悄变成无穷大。
//     写成整数形式（没有小数点和指数）、绝对值小于 2^53 的数记为“整数”，再写出去时仍是整数形式；
//     更大的整数 double 未必能精确表示，只当作普通的浮点数。
//   · 字符串按 UTF-8 原样保存。\uXXXX 转义换成对应的 UTF-8，一前一后两个转义组成的代理对（表示 U+10000 以上的字符）合成一个码点；
//     落单的代理、不合法的 UTF-8 字节、没有转义的控制字符（小于 0x20）都算错。
//   · 同一个对象里出现重复的键算错（各种程序对重复键的取法不一，这里干脆不接受）。
//   · 数组、对象的嵌套层数有上限（默认 128 层），免得恶意输入把栈压垮。
//   · 出错时抛 ParseError（派生自 std::invalid_argument）。what() 里带出错处的字节偏移（从 0 起）和行、列
//     （从 1 起，列按字符数）；也可以用 offset()、line()、column()、reason() 分别取。
//
// 写（dump / Writer）：
//   · 输出是确定的：同样的值总给出同样的文字，对象的各项按插入顺序。
//   · 数取“读回来还是同一个 double”的最短写法：先试 15、16 位有效数字，不行就用 17 位（17 位总是够的）。
//     整数写成 20；浮点数总带小数点或指数（1.0、2.5e-07），读回来仍分得清。NaN、±Inf 没有对应的 JSON 写法，写成 null。
//   · 字符串里的引号、反斜杠、控制字符按规定转义，其余 UTF-8 原样写出；不合法的字节换成 U+FFFD，输出总是合法的 UTF-8。
//   · indent < 0（默认）：整个值写在一行，逗号、冒号后面各空一格。
//     indent >= 0：对象、以及里面含有对象的数组，每项占一行，逐层缩进 indent 个空格；
//     其余的数组（一串数、[实部, 虚部] 之类）仍写在一行。
//
// 数与文字之间的换算交给标准库，用哪一套由功能测试宏 __cpp_lib_to_chars 决定：
//   · 标准库有 double 的 std::from_chars / std::to_chars 时（MSVC、GCC 11 起的 libstdc++）用它们：正确舍入，与 locale 无关。
//   · 没有时（如 Apple 的 libc++）用 C 库的 strtod / snprintf。它们受 C locale 影响（有的 locale 用逗号当小数点），
//     这里已经处理过，读写的结果同样与 locale 无关；只是不要在别的线程调用 setlocale 的同时读写。
// 不一律用 strtod，是因为 Windows 上两种 C 运行库的 strtod 都有舍入出错的输入：MinGW-w64 的会把一些紧挨在 2 的幂
// 下面的数进位成 2 的幂（于是这样的 double 写出去读不回来）；微软运行库的只看前 768 位有效数字，个别恰在两个
// double 正中间的数也舍入错。两套函数写出的文字相同，所以输出不随平台而变。
// 超过 768 个字符的数先化成不超过 768 位有效数字再换算，结果仍是正确舍入的，用时与长度成正比
// （做法和理由见 Parser::convert_reduced）。
#pragma once

#include <algorithm>
#if __has_include(<charconv>)
#include <charconv>      // 有 double 的 from_chars / to_chars 时，它会定义 __cpp_lib_to_chars
#endif
#include <clocale>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace stablercwa {
namespace json {

enum class Type { Null, Bool, Number, String, Array, Object };

inline constexpr int kDefaultMaxDepth = 128;                    // parse() 默认允许的嵌套层数
inline constexpr double kMaxSafeInteger = 9007199254740991.0;   // 2^53 − 1：绝对值不超过它的整数 double 都能精确表示
// 直接交给库函数换算的数最多有这么多位有效数字；字符数超过它的数先化简（见 Parser::convert_reduced）
inline constexpr std::size_t kMaxDigits = 768;

namespace detail {
class Parser;
[[noreturn]] inline void type_error(const char* what) { throw std::invalid_argument(std::string("json: ") + what); }
}  // namespace detail

// 一个 JSON 值。数组的元素、对象的各项都按顺序放在 items_ 里；对象另有一串与之对应的键 keys_。
class Value {
public:
    Value() = default;                                           // null
    Value(std::nullptr_t) {}
    Value(bool b) : type_(Type::Bool), bool_(b) {}
    Value(int x) : type_(Type::Number), integer_(true), number_(x) {}
    // 绝对值达到 2^53 的整数 double 未必能精确表示，这时只当作普通的浮点数。
    Value(long long x)
        : type_(Type::Number), integer_(x >= -9007199254740991LL && x <= 9007199254740991LL), number_(static_cast<double>(x)) {}
    Value(double x) : type_(Type::Number), number_(x) {}
    Value(const char* s) : type_(Type::String), string_(s) {}
    Value(std::string s) : type_(Type::String), string_(std::move(s)) {}
    template <class T>
    Value(const T*) = delete;                                    // 别的指针不许悄悄变成 bool

    static Value array() {
        Value v;
        v.type_ = Type::Array;
        return v;
    }
    static Value object() {
        Value v;
        v.type_ = Type::Object;
        return v;
    }

    Type type() const noexcept { return type_; }
    bool is_null() const noexcept { return type_ == Type::Null; }
    bool is_bool() const noexcept { return type_ == Type::Bool; }
    bool is_number() const noexcept { return type_ == Type::Number; }
    bool is_string() const noexcept { return type_ == Type::String; }
    bool is_array() const noexcept { return type_ == Type::Array; }
    bool is_object() const noexcept { return type_ == Type::Object; }
    // 是数，而且是按整数给出的（构造时给的是整数，或者解析时写成不带小数点和指数的形式）。
    bool is_integer() const noexcept { return type_ == Type::Number && integer_; }

    // 取值；类型不对时抛 std::invalid_argument。
    bool as_bool() const {
        if (type_ != Type::Bool) detail::type_error("value is not a boolean");
        return bool_;
    }
    double as_double() const {
        if (type_ != Type::Number) detail::type_error("value is not a number");
        return number_;
    }
    const std::string& as_string() const {
        if (type_ != Type::String) detail::type_error("value is not a string");
        return string_;
    }
    // 是数、数值是整数（20、20.0、2e1 都算）、又落在 int 的范围内时，写进 out 并返回 true。
    bool to_int(int& out) const noexcept {
        if (type_ != Type::Number) return false;
        if (!(number_ >= -2147483648.0 && number_ <= 2147483647.0) || number_ != std::floor(number_)) return false;
        out = static_cast<int>(number_);
        return true;
    }

    // 数组的元素个数 / 对象的项数；其余类型是 0。
    std::size_t size() const noexcept { return items_.size(); }
    // 数组的第 i 个元素，或对象的第 i 项的值（按插入顺序）；越界抛 std::invalid_argument。
    const Value& at(std::size_t i) const {
        if (i >= items_.size()) detail::type_error("index out of range");
        return items_[i];
    }
    Value& at(std::size_t i) {
        if (i >= items_.size()) detail::type_error("index out of range");
        return items_[i];
    }
    // 对象第 i 项的键。
    const std::string& key(std::size_t i) const {
        if (i >= keys_.size()) detail::type_error("index out of range");
        return keys_[i];
    }
    // 对象里键为 name 的值；没有这个键、或者本身不是对象时返回空指针。
    const Value* find(const std::string& name) const noexcept {
        for (std::size_t i = 0; i < keys_.size(); ++i)
            if (keys_[i] == name) return &items_[i];
        return nullptr;
    }
    Value* find(const std::string& name) noexcept {
        for (std::size_t i = 0; i < keys_.size(); ++i)
            if (keys_[i] == name) return &items_[i];
        return nullptr;
    }
    bool contains(const std::string& name) const noexcept { return find(name) != nullptr; }
    // 同 find，但没有这个键时抛 std::invalid_argument。
    const Value& at(const std::string& name) const {
        const Value* v = find(name);
        if (!v) detail::type_error("no such key in the object");
        return *v;
    }

    // 在数组末尾添一个元素，返回它的引用。null 值先变成空数组；其余类型抛 std::invalid_argument。
    Value& push_back(Value v) {
        if (type_ == Type::Null) type_ = Type::Array;
        if (type_ != Type::Array) detail::type_error("push_back needs an array");
        items_.push_back(std::move(v));
        return items_.back();
    }
    // 设置对象的一项，返回它的引用：键已经有了就换掉值（位置不变），没有就添在末尾。null 值先变成空对象。
    Value& set(const std::string& name, Value v) {
        if (type_ == Type::Null) type_ = Type::Object;
        if (type_ != Type::Object) detail::type_error("set needs an object");
        if (Value* old = find(name)) {
            *old = std::move(v);
            return *old;
        }
        keys_.push_back(name);
        items_.push_back(std::move(v));
        return items_.back();
    }

private:
    friend class detail::Parser;

    Type type_ = Type::Null;
    bool bool_ = false;
    bool integer_ = false;
    double number_ = 0.0;
    std::string string_;
    std::vector<Value> items_;
    std::vector<std::string> keys_;
};

// 语法错误。what() 形如 "json: unexpected character '}' at offset 17 (line 2, column 5)"。
class ParseError : public std::invalid_argument {
public:
    ParseError(const std::string& why, std::size_t at, std::size_t row, std::size_t col)
        : std::invalid_argument("json: " + why + " at offset " + std::to_string(at) + " (line " + std::to_string(row) +
                                ", column " + std::to_string(col) + ")"),
          reason_size_(why.size()), offset_(at), line_(row), column_(col) {}
    std::string reason() const { return std::string(what() + 6, reason_size_); }   // 不带位置的那一段说明
    std::size_t offset() const noexcept { return offset_; }                        // 字节偏移，从 0 起
    std::size_t line() const noexcept { return line_; }                            // 行号，从 1 起
    std::size_t column() const noexcept { return column_; }                        // 列号（按字符数），从 1 起

private:
    std::size_t reason_size_, offset_, line_, column_;
};

namespace detail {

// ------------------------------------------------------------------ UTF-8

// s[0] 是一个多字节字符的首字节（>= 0x80）时，返回这个字符的字节数（2 ~ 4）；不是合法的 UTF-8 时返回 0。
// 过长的编码、代理区 U+D800 ~ U+DFFF、超过 U+10FFFF 的码点都算不合法。avail 是从 s 起还剩多少字节。
inline std::size_t utf8_length(const char* s, std::size_t avail) noexcept {
    const auto octet = [&](std::size_t k) { return static_cast<unsigned char>(s[k]); };
    const auto tail = [&](std::size_t k) { return (octet(k) & 0xC0) == 0x80; };
    const unsigned char b0 = octet(0);
    if (b0 >= 0xC2 && b0 <= 0xDF) return avail >= 2 && tail(1) ? 2 : 0;
    if (b0 >= 0xE0 && b0 <= 0xEF) {
        if (avail < 3 || !tail(1) || !tail(2)) return 0;
        if (b0 == 0xE0 && octet(1) < 0xA0) return 0;                  // 过长的编码
        if (b0 == 0xED && octet(1) > 0x9F) return 0;                  // 代理区
        return 3;
    }
    if (b0 >= 0xF0 && b0 <= 0xF4) {
        if (avail < 4 || !tail(1) || !tail(2) || !tail(3)) return 0;
        if (b0 == 0xF0 && octet(1) < 0x90) return 0;                  // 过长的编码
        if (b0 == 0xF4 && octet(1) > 0x8F) return 0;                  // 超过 U+10FFFF
        return 4;
    }
    return 0;
}

// 把码点 cp（不在代理区，不超过 U+10FFFF）按 UTF-8 接到 out 后面。
inline void append_utf8(std::string& out, unsigned long cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

// ------------------------------------------------------------------ 数与文字

// 把 [text, text + size) 这个写法合乎 JSON 的数换算成最接近的 double（正确舍入）。
// 返回 false 表示库函数没能给出结果，调用者改走 Parser::convert_reduced。
#ifdef __cpp_lib_to_chars
// std::from_chars：数值超出 double 的范围时它只报告“超出范围”，这时 x 里是什么标准没有规定
// （有的库不动它，有的给零或无穷大），所以也算没有结果。
inline bool text_to_double(const char* text, std::size_t size, double& x) {
    const std::from_chars_result r = std::from_chars(text, text + size, x);
    return r.ec == std::errc() && r.ptr == text + size;
}
#else
// strtod：要以 '\0' 结尾的字符串，所以先抄一份（短的抄在栈上）。它认的小数点随 C locale 而变（有的 C 库则始终只认 '.'），
// 所以先照原样交给它；它在半路停下时，把 '.' 换成当前 locale 的小数点再试一次。超出范围时它给的是无穷大、零或
// 次正规数，正是要的结果。
inline bool text_to_double(const char* text, std::size_t size, double& x) {
    char local[64];
    std::string copy;
    const char* z = local;
    if (size < sizeof local) {
        std::memcpy(local, text, size);
        local[size] = '\0';
    } else {
        copy.assign(text, size);
        z = copy.c_str();
    }
    char* stop = nullptr;
    x = std::strtod(z, &stop);
    if (stop != z && *stop == '\0') return true;
    const std::lconv* conv = std::localeconv();
    const char* point = conv && conv->decimal_point ? conv->decimal_point : ".";
    std::string swapped;
    for (std::size_t k = 0; k < size; ++k) {
        if (text[k] == '.') swapped += point;
        else swapped.push_back(text[k]);
    }
    x = std::strtod(swapped.c_str(), &stop);
    return stop != swapped.c_str() && *stop == '\0';
}
#endif

// 把 x 按 printf 的 "%.{digits}g" 的样子写进 buf（以 '\0' 结尾），返回写了多少个字符；写不下时返回 0。
// std::to_chars 的这种写法按标准与 "C" locale 下的 printf 相同；snprintf 用的是当前 locale 的小数点，由调用者整理。
inline int format_general(char* buf, std::size_t size, double x, int digits) {
#ifdef __cpp_lib_to_chars
    const std::to_chars_result r = std::to_chars(buf, buf + size - 1, x, std::chars_format::general, digits);
    if (r.ec != std::errc()) return 0;
    *r.ptr = '\0';
    return static_cast<int>(r.ptr - buf);
#else
    const int len = std::snprintf(buf, size, "%.*g", digits, x);
    return len > 0 && len < static_cast<int>(size) ? len : 0;
#endif
}

// ------------------------------------------------------------------ 写：数与字符串

// 整数的十进制写法。
inline void append_integer(std::string& out, long long x) {
    // 先取绝对值再逐位取余；无符号数的取负是模 2^64 的，最小的负数也不会溢出
    unsigned long long u = x < 0 ? 0ULL - static_cast<unsigned long long>(x) : static_cast<unsigned long long>(x);
    char buf[24];
    int len = 0;
    do {
        buf[len++] = static_cast<char>('0' + static_cast<int>(u % 10));
        u /= 10;
    } while (u != 0);
    if (x < 0) out.push_back('-');
    while (len > 0) out.push_back(buf[--len]);
}

// 浮点数的写法：读回来还是同一个 double 的最短十进制数；NaN、±Inf 写成 null。
inline void append_double(std::string& out, double x) {
    if (!std::isfinite(x)) {
        out += "null";
        return;
    }
    char text[64] = "";                                              // 整理好的写法
    for (int digits = 15; digits <= 17; ++digits) {
        char buf[48];
        const int len = format_general(buf, sizeof buf, x, digits);
        if (len == 0) {                                              // 不会发生；真发生了也不写出半截的数
            out += "null";
            return;
        }
        // 整理成与 locale、与库无关的固定样子：小数点一律是 '.'（snprintf 用的是 locale 的小数点，可能是逗号，
        // 甚至是多字节的字符）；指数写成 e+NN / e-NN，至少两位（有的 C 库写三位）；既没有小数点也没有指数的补上 ".0"。
        int n = 0, k = 0;
        bool marked = false;                                         // 已经有小数点或指数
        for (; k < len && buf[k] != 'e' && buf[k] != 'E'; ++k) {
            const char c = buf[k];
            if ((c >= '0' && c <= '9') || c == '-') {
                text[n++] = c;
            } else if (!marked) {
                text[n++] = '.';
                marked = true;
            }
        }
        if (k < len) {                                               // 指数部分
            ++k;
            text[n++] = 'e';
            text[n++] = k < len && buf[k] == '-' ? '-' : '+';
            if (k < len && (buf[k] == '-' || buf[k] == '+')) ++k;
            while (len - k > 2 && buf[k] == '0') ++k;
            if (len - k < 2) text[n++] = '0';
            for (; k < len; ++k) text[n++] = buf[k];
        } else if (!marked) {
            text[n++] = '.';
            text[n++] = '0';
        }
        text[n] = '\0';
        double back = 0.0;
        if (digits == 17 || (text_to_double(text, static_cast<std::size_t>(n), back) && back == x)) break;
    }
    out += text;
}

// 带引号、转义好的字符串。
inline void append_string(std::string& out, const char* s, std::size_t size) {
    static const char hex[] = "0123456789abcdef";
    out.push_back('"');
    std::size_t i = 0;
    while (i < size) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (c == '"') {
            out += "\\\"";
            ++i;
        } else if (c == '\\') {
            out += "\\\\";
            ++i;
        } else if (c < 0x20) {
            switch (c) {
                case '\b': out += "\\b"; break;
                case '\f': out += "\\f"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                default:
                    out += "\\u00";
                    out.push_back(hex[c >> 4]);
                    out.push_back(hex[c & 15]);
            }
            ++i;
        } else if (c < 0x80) {
            out.push_back(s[i]);
            ++i;
        } else {
            const std::size_t len = utf8_length(s + i, size - i);
            if (len == 0) {
                out += "\xEF\xBF\xBD";                               // 不合法的字节：换成 U+FFFD
                ++i;
            } else {
                out.append(s + i, len);
                i += len;
            }
        }
    }
    out.push_back('"');
}

}  // namespace detail

// ------------------------------------------------------------------ Writer

// 边算边写的 JSON 输出。调用的次序就是输出的次序：
//     begin_object()，然后交替 key(...) 与一个值（value(...)、null()，或者再开一个数组 / 对象），最后 end_object()；
//     begin_array()，然后依次写各个元素，最后 end_array()。
// one_line 为真的数组 / 对象，连同它里面的所有内容，在缩进模式下也写在一行。
// 次序不对（对象里没写键就写值、括号不配对、最外层写了两个值）时抛 std::logic_error。
class Writer {
public:
    explicit Writer(int indent = -1) : indent_((std::min)(indent, 16)) {}

    void begin_object(bool one_line = false) { begin('{', true, one_line); }
    void end_object() { end('}', true); }
    void begin_array(bool one_line = false) { begin('[', false, one_line); }
    void end_array() { end(']', false); }

    void key(const std::string& name) { write_key(name.data(), name.size()); }
    void key(const char* name) { write_key(name, std::strlen(name)); }

    void null() {
        before_value();
        out_ += "null";
    }
    void value(bool b) {
        before_value();
        out_ += b ? "true" : "false";
    }
    void value(int x) { value(static_cast<long long>(x)); }
    void value(long long x) {
        before_value();
        detail::append_integer(out_, x);
    }
    void value(double x) {                                       // NaN、±Inf 写成 null
        before_value();
        detail::append_double(out_, x);
    }
    void value(const std::string& s) {
        before_value();
        detail::append_string(out_, s.data(), s.size());
    }
    void value(const char* s) {
        before_value();
        detail::append_string(out_, s, std::strlen(s));
    }
    template <class T>
    void value(const T*) = delete;                               // 别的指针不许悄悄变成 bool

    bool complete() const noexcept { return done_ && stack_.empty(); }   // 已经写完一个完整的值
    const std::string& str() const noexcept { return out_; }
    // 取走写好的文字，Writer 回到刚构造时的状态。
    std::string take() {
        std::string text = std::move(out_);
        out_.clear();
        stack_.clear();
        have_key_ = false;
        done_ = false;
        return text;
    }

private:
    struct Frame {
        bool object;      // 对象（否则是数组）
        bool one_line;    // 写在一行
        bool empty;       // 还没有写过任何一项
    };

    [[noreturn]] static void misuse(const char* what) { throw std::logic_error(std::string("json::Writer: ") + what); }

    void newline(std::size_t depth) {
        out_.push_back('\n');
        out_.append(depth * static_cast<std::size_t>(indent_), ' ');
    }
    // 数组的元素、对象的键之前：与上一项之间的逗号，以及空格或换行缩进
    void separate() {
        Frame& f = stack_.back();
        if (!f.empty) out_.push_back(',');
        if (!f.one_line) newline(stack_.size());
        else if (!f.empty) out_.push_back(' ');
        f.empty = false;
    }
    void before_value() {
        if (stack_.empty()) {
            if (done_) misuse("more than one top-level value");
            done_ = true;
        } else if (stack_.back().object) {
            if (!have_key_) misuse("a value inside an object needs a key first");
            have_key_ = false;
        } else {
            separate();
        }
    }
    void write_key(const char* name, std::size_t size) {
        if (stack_.empty() || !stack_.back().object || have_key_) misuse("a key is only allowed inside an object, one per value");
        separate();
        detail::append_string(out_, name, size);
        out_ += ": ";
        have_key_ = true;
    }
    void begin(char bracket, bool object, bool one_line) {
        before_value();
        const bool flat = indent_ < 0 || one_line || (!stack_.empty() && stack_.back().one_line);
        stack_.push_back(Frame{object, flat, true});
        out_.push_back(bracket);
    }
    void end(char bracket, bool object) {
        if (stack_.empty() || stack_.back().object != object || have_key_) misuse("mismatched end of an array or object");
        const Frame f = stack_.back();
        stack_.pop_back();
        if (!f.one_line && !f.empty) newline(stack_.size());
        out_.push_back(bracket);
    }

    int indent_;
    std::string out_;
    std::vector<Frame> stack_;
    bool have_key_ = false;   // 对象里刚写了键，正等着它的值
    bool done_ = false;       // 最外层的值已经开始写了
};

namespace detail {

// v（或它里面的任何一层）是不是含有对象
inline bool has_object(const Value& v) {
    if (v.is_object()) return true;
    if (!v.is_array()) return false;
    for (std::size_t i = 0; i < v.size(); ++i)
        if (has_object(v.at(i))) return true;
    return false;
}

// 把 v 交给 Writer。flat：外层已经决定写在一行，里面不必再判断。
inline void write_value(Writer& w, const Value& v, bool flat) {
    switch (v.type()) {
        case Type::Null: w.null(); break;
        case Type::Bool: w.value(v.as_bool()); break;
        case Type::Number:
            if (v.is_integer()) w.value(static_cast<long long>(v.as_double()));
            else w.value(v.as_double());
            break;
        case Type::String: w.value(v.as_string()); break;
        case Type::Array: {
            const bool one_line = flat || !has_object(v);
            w.begin_array(one_line);
            for (std::size_t i = 0; i < v.size(); ++i) write_value(w, v.at(i), one_line);
            w.end_array();
            break;
        }
        case Type::Object:
            w.begin_object(flat);
            for (std::size_t i = 0; i < v.size(); ++i) {
                w.key(v.key(i));
                write_value(w, v.at(i), flat);
            }
            w.end_object();
            break;
    }
}

// ------------------------------------------------------------------ 读

// 递归下降的解析器。data 不必以 '\0' 结尾，里面可以有 '\0'（它不是合法的 JSON 字符，会报错）。
class Parser {
public:
    Parser(const char* data, std::size_t size, int max_depth) : p_(data), n_(size), max_depth_(max_depth) {}

    Value run() {
        if (n_ >= 3 && std::memcmp(p_, "\xEF\xBB\xBF", 3) == 0) i_ = start_ = 3;   // 跳过 BOM
        Value v = value(0);
        skip_space();
        if (i_ < n_) fail("unexpected " + show(p_[i_]) + " after the JSON value", i_);
        return v;
    }

private:
    [[noreturn]] void fail(const std::string& reason, std::size_t at) const {
        std::size_t line = 1, column = 1;
        for (std::size_t k = start_; k < at && k < n_; ++k) {
            const unsigned char c = static_cast<unsigned char>(p_[k]);
            if (c == '\n') {
                ++line;
                column = 1;
            } else if ((c & 0xC0) != 0x80) {     // UTF-8 的后续字节不另算一列
                ++column;
            }
        }
        throw ParseError(reason, at, line, column);
    }

    static std::string show(char ch) {
        const unsigned char c = static_cast<unsigned char>(ch);
        if (c >= 0x20 && c < 0x7F) return std::string("character '") + ch + "'";
        static const char hex[] = "0123456789ABCDEF";
        std::string s = "byte 0x";
        s.push_back(hex[c >> 4]);
        s.push_back(hex[c & 15]);
        return s;
    }
    static bool is_digit(char c) noexcept { return c >= '0' && c <= '9'; }

    void skip_space() noexcept {
        while (i_ < n_ && (p_[i_] == ' ' || p_[i_] == '\t' || p_[i_] == '\n' || p_[i_] == '\r')) ++i_;
    }

    Value value(int depth) {
        skip_space();
        if (i_ >= n_) fail("unexpected end of input", i_);
        const char c = p_[i_];
        if (c == '{') return object(depth);
        if (c == '[') return array(depth);
        if (c == '"') {
            std::string s;
            string(s);
            return Value(std::move(s));
        }
        if (c == 't') {
            literal("true", 4);
            return Value(true);
        }
        if (c == 'f') {
            literal("false", 5);
            return Value(false);
        }
        if (c == 'n') {
            literal("null", 4);
            return Value();
        }
        if (c != '-' && !is_digit(c)) fail("unexpected " + show(c), i_);
        return number();
    }

    void literal(const char* word, std::size_t size) {
        if (n_ - i_ < size || std::memcmp(p_ + i_, word, size) != 0)
            fail("invalid literal (expected true, false or null)", i_);
        i_ += size;
    }

    // 数：-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?。先自己核对语法，再把这一段文字交给库函数换算。
    Value number() {
        const std::size_t start = i_;
        bool integer_form = true;
        if (p_[i_] == '-') ++i_;
        if (i_ >= n_ || !is_digit(p_[i_])) fail("a digit must follow the minus sign", i_);
        if (p_[i_] == '0') {
            ++i_;
            if (i_ < n_ && is_digit(p_[i_])) fail("leading zeros are not allowed in a number", start);
        } else {
            while (i_ < n_ && is_digit(p_[i_])) ++i_;
        }
        if (i_ < n_ && p_[i_] == '.') {
            integer_form = false;
            ++i_;
            if (i_ >= n_ || !is_digit(p_[i_])) fail("a digit must follow the decimal point", i_);
            while (i_ < n_ && is_digit(p_[i_])) ++i_;
        }
        if (i_ < n_ && (p_[i_] == 'e' || p_[i_] == 'E')) {
            integer_form = false;
            ++i_;
            if (i_ < n_ && (p_[i_] == '+' || p_[i_] == '-')) ++i_;
            if (i_ >= n_ || !is_digit(p_[i_])) fail("a digit must follow the exponent mark", i_);
            while (i_ < n_ && is_digit(p_[i_])) ++i_;
        }

        // 换算。通常直接交给库函数；很长的数，以及库函数没能给出结果的数（多半是超出了 double 的范围），先化简。
        const std::size_t size = i_ - start;
        double x = 0.0;
        if (size > kMaxDigits || !text_to_double(p_ + start, size, x)) x = convert_reduced(start, i_);
        if (!std::isfinite(x)) fail("number is outside the range of double", start);
        // 整数形式、绝对值小于 2^53（所以与写出来的十进制数严格相等）、又不是 −0 的记为整数
        if (integer_form && std::abs(x) <= kMaxSafeInteger && !(x == 0.0 && std::signbit(x)))
            return Value(static_cast<long long>(x));
        return Value(x);
    }

    // 库函数不能直接换算的数走这里：字符数超过 kMaxDigits 的，以及库函数说超出 double 范围的。
    // 做法：把数拆成“有效数字 × 10^scale”，有效数字只留前 768 位，写成不带小数点的样子再交给库函数；
    // 超出范围时按数量级给出无穷大或零。[start, end) 是已经核对过语法的数。
    //
    // 为什么只留 768 位还能正确舍入：一个数舍入成哪个 double，只看它落在哪两个分界点之间；分界点是相邻两个 double
    // 的中点（上溢、下溢的界也是这种形式）。写成十进制，任何 double 不超过 767 位有效数字，任何分界点不超过 768 位，
    // 所以分界点在第 768 位之后全是零，被舍掉的尾巴不会让数越过分界点。只有一种情形要分清：截断后恰好落在分界点上，
    // 而原数比它略大（尾巴不全是零）。分界点的第 768 位只能是 0 或 5（不足 768 位的补零；恰有 768 位的是 5 的奇数倍），
    // 所以尾巴不全是零、第 768 位又是 0 或 5 时，给这一位加 1：变成 1 或 6，不进位，结果不在分界点上，与原数之间也
    // 没有隔着分界点。其余情形截断后本来就不在分界点上，照原样。
    // 位数不能留得更多：有的标准库只看前 768 位有效数字，后面的悄悄丢掉。数字串短了，也避开了有的 C 库在很长的
    // 数字串上花平方量级时间的问题。
    double convert_reduced(std::size_t start, std::size_t end) const {
        std::string digits;                  // 保留下来的有效数字（去掉了前导零）
        digits.reserve(kMaxDigits + 32);
        bool sticky = false;                 // 舍掉的数字里有非零的
        long long scale = 0;                 // 数值 = digits（当作整数）× 10^scale
        std::size_t k = start;
        const bool negative = p_[k] == '-';
        if (negative) ++k;
        bool fraction = false;               // 已经过了小数点
        for (; k < end && p_[k] != 'e' && p_[k] != 'E'; ++k) {
            const char c = p_[k];
            if (c == '.') {
                fraction = true;
            } else if (digits.empty() && c == '0') {         // 前导零：不存，小数部分的仍要计入位置
                if (fraction) --scale;
            } else if (digits.size() < kMaxDigits) {
                digits.push_back(c);
                if (fraction) --scale;
            } else {                                         // 存满之后的数字：整数部分的每一位把数值放大十倍
                if (c != '0') sticky = true;
                if (!fraction) ++scale;
            }
        }
        long long exponent = 0;
        if (k < end) {                       // 指数的位数可以任意多：累加到足够大就不再加，结果反正是溢出或下溢
            ++k;
            const bool minus = p_[k] == '-';
            if (p_[k] == '+' || p_[k] == '-') ++k;
            for (; k < end; ++k)
                if (exponent < 100000000) exponent = exponent * 10 + (p_[k] - '0');
            if (minus) exponent = -exponent;
        }
        if (digits.empty()) return negative ? -0.0 : 0.0;
        if (sticky && (digits.back() == '0' || digits.back() == '5')) ++digits.back();
        scale += exponent;
        // 数值在 10^(top−1) 与 10^top 之间。超出 double 的范围时，top > 0 的是上溢，其余是下溢
        const long long top = static_cast<long long>(digits.size()) + scale;
        double x = top > 0 ? HUGE_VAL : 0.0;
        if (top > -400 && top < 400) {       // 远在范围之外的不必再问库函数
            digits += 'e';
            digits += std::to_string(scale);
            double converted = 0.0;
            if (text_to_double(digits.data(), digits.size(), converted)) x = converted;
        }
        return negative ? -x : x;
    }

    // \uXXXX 里的四位十六进制数。escape 是这个转义开头的反斜杠的位置（报错用）。
    unsigned long hex4(std::size_t escape) {
        if (n_ - i_ < 4) fail("\\u must be followed by four hexadecimal digits", escape);
        unsigned long v = 0;
        for (int k = 0; k < 4; ++k) {
            const char c = p_[i_ + static_cast<std::size_t>(k)];
            unsigned long d = 0;
            if (c >= '0' && c <= '9') d = static_cast<unsigned long>(c - '0');
            else if (c >= 'a' && c <= 'f') d = static_cast<unsigned long>(c - 'a') + 10;
            else if (c >= 'A' && c <= 'F') d = static_cast<unsigned long>(c - 'A') + 10;
            else fail("\\u must be followed by four hexadecimal digits", escape);
            v = v * 16 + d;
        }
        i_ += 4;
        return v;
    }

    // 字符串。进来时 p_[i_] 是开头的引号；解码后的内容接到 out 后面。
    void string(std::string& out) {
        const std::size_t opened = i_++;
        for (;;) {
            // 一段不需要处理的普通 ASCII 字符整段照抄
            const std::size_t plain = i_;
            while (i_ < n_) {
                const unsigned char c = static_cast<unsigned char>(p_[i_]);
                if (c < 0x20 || c >= 0x80 || c == '"' || c == '\\') break;
                ++i_;
            }
            out.append(p_ + plain, i_ - plain);
            if (i_ >= n_) fail("unterminated string", opened);
            const unsigned char c = static_cast<unsigned char>(p_[i_]);
            if (c == '"') {
                ++i_;
                return;
            }
            if (c < 0x20) fail("control character in a string (it must be escaped)", i_);
            if (c >= 0x80) {
                const std::size_t len = utf8_length(p_ + i_, n_ - i_);
                if (len == 0) fail("invalid UTF-8 in a string", i_);
                out.append(p_ + i_, len);
                i_ += len;
                continue;
            }
            // 反斜杠转义
            const std::size_t escape = i_++;
            if (i_ >= n_) fail("unterminated string", opened);
            const char e = p_[i_++];
            if (e == '"' || e == '\\' || e == '/') {
                out.push_back(e);
            } else if (e == 'b') {
                out.push_back('\b');
            } else if (e == 'f') {
                out.push_back('\f');
            } else if (e == 'n') {
                out.push_back('\n');
            } else if (e == 'r') {
                out.push_back('\r');
            } else if (e == 't') {
                out.push_back('\t');
            } else if (e == 'u') {
                unsigned long cp = hex4(escape);
                if (cp >= 0xD800 && cp <= 0xDBFF) {                  // 高位代理：后面必须紧跟一个 \u 低位代理
                    if (n_ - i_ < 2 || p_[i_] != '\\' || p_[i_ + 1] != 'u')
                        fail("unpaired surrogate in a \\u escape", escape);
                    i_ += 2;
                    const unsigned long low = hex4(escape);
                    if (low < 0xDC00 || low > 0xDFFF) fail("unpaired surrogate in a \\u escape", escape);
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                    fail("unpaired surrogate in a \\u escape", escape);
                }
                append_utf8(out, cp);
            } else {
                fail("invalid escape sequence in a string", escape);
            }
        }
    }

    Value array(int depth) {
        const std::size_t opened = i_;
        if (depth >= max_depth_) fail("arrays and objects are nested too deeply", opened);
        ++i_;
        Value out = Value::array();
        skip_space();
        if (i_ < n_ && p_[i_] == ']') {
            ++i_;
            return out;
        }
        for (;;) {
            out.items_.push_back(value(depth + 1));
            skip_space();
            if (i_ >= n_) fail("unexpected end of input inside an array", i_);
            if (p_[i_] == ']') {
                ++i_;
                return out;
            }
            if (p_[i_] != ',') fail("expected ',' or ']' in an array, found " + show(p_[i_]), i_);
            ++i_;
        }
    }

    Value object(int depth) {
        const std::size_t opened = i_;
        if (depth >= max_depth_) fail("arrays and objects are nested too deeply", opened);
        ++i_;
        Value out = Value::object();
        std::vector<std::size_t> key_at;                 // 各个键在文字里的位置（查到重复的键时报错用）
        skip_space();
        if (i_ < n_ && p_[i_] == '}') {
            ++i_;
            return out;
        }
        for (;;) {
            skip_space();
            if (i_ >= n_) fail("unexpected end of input inside an object", i_);
            if (p_[i_] != '"') fail("expected a string key in an object, found " + show(p_[i_]), i_);
            key_at.push_back(i_);
            std::string name;
            string(name);
            skip_space();
            if (i_ >= n_) fail("unexpected end of input inside an object", i_);
            if (p_[i_] != ':') fail("expected ':' after an object key, found " + show(p_[i_]), i_);
            ++i_;
            Value v = value(depth + 1);
            out.keys_.push_back(std::move(name));
            out.items_.push_back(std::move(v));
            skip_space();
            if (i_ >= n_) fail("unexpected end of input inside an object", i_);
            if (p_[i_] == '}') {
                ++i_;
                break;
            }
            if (p_[i_] != ',') fail("expected ',' or '}' in an object, found " + show(p_[i_]), i_);
            ++i_;
        }
        check_duplicate_keys(out.keys_, key_at);
        return out;
    }

    // 重复的键：报第一个“前面已经出现过”的键的位置。项数少时两两比较；多时先排序，免得项数很多的对象耗时成平方增长。
    void check_duplicate_keys(const std::vector<std::string>& keys, const std::vector<std::size_t>& key_at) const {
        const std::size_t count = keys.size();
        std::size_t repeated = count;                    // 重复出现的键里位置最靠前的一个
        if (count <= 16) {
            for (std::size_t b = 1; b < count && repeated == count; ++b)
                for (std::size_t a = 0; a < b; ++a)
                    if (keys[a] == keys[b]) {
                        repeated = b;
                        break;
                    }
        } else {
            std::vector<std::size_t> order(count);
            for (std::size_t k = 0; k < count; ++k) order[k] = k;
            std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
                const int c = keys[a].compare(keys[b]);
                return c != 0 ? c < 0 : a < b;
            });
            for (std::size_t k = 1; k < count; ++k)
                if (keys[order[k]] == keys[order[k - 1]]) repeated = (std::min)(repeated, order[k]);
        }
        if (repeated < count) fail("duplicate key in an object", key_at[repeated]);
    }

    const char* p_;
    std::size_t n_;
    std::size_t i_ = 0;
    std::size_t start_ = 0;       // 跳过 BOM 之后正文开始的位置
    int max_depth_;
};

}  // namespace detail

// 解析一段 JSON 文字。max_depth 是数组、对象允许嵌套的层数。出错抛 ParseError。
inline Value parse(const std::string& text, int max_depth = kDefaultMaxDepth) {
    return detail::Parser(text.data(), text.size(), max_depth).run();
}
// 同上，文字由指针和长度给出（不必以 '\0' 结尾）。这里 max_depth 没有默认值：
// 否则 parse("[1]", 2) 这样的调用分不清第二个参数是长度还是层数。
inline Value parse(const char* data, std::size_t size, int max_depth) {
    return detail::Parser(data, size, max_depth).run();
}

// 把一个值写成 JSON 文字。indent 的含义见文件头。
inline std::string dump(const Value& v, int indent = -1) {
    Writer w(indent);
    detail::write_value(w, v, indent < 0);
    return w.take();
}

}  // namespace json
}  // namespace stablercwa
