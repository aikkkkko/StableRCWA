// tests/test_rcwa_json.cpp —— json.hpp（JSON 读写）与 rcwa_json.hpp（一维 RCWA 的 JSON 入口）的测试
//
//   c++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -I include tests/test_rcwa_json.cpp -o build/test_rcwa_json
//   c++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -I include tests/test_rcwa_json.cpp -o build/test_rcwa_json_san
//
// 只依赖 C++17 标准库。内容：
//   · JSON 的读：各种值、数、字符串的转义与 UTF-8、空白、嵌套的上限；每一种语法错误报的位置；
//   · JSON 的写：数读回来逐位相同、转义、缩进的样子、键的次序、与 locale 无关；
//   · 请求的检查：每一种写错的请求都得到 ok = false 和预期的 where；
//   · solve_json 的应答与直接调用 rcwa::solve / rcwa::sweep 逐位相同：单点、多点扫描（1 个与 4 个线程）、TE 与 TM、
//     吸收层、膜系、各个选项；数值失败的点写成 null；几个线程同时调用也一样；
//   · 圆锥入射：应答里的复 Jones 矩阵等与直接调用 rcwa::solve_conical / rcwa::sweep_conical 逐位相同；
//     min_abs_lambda（没有做本征分解的层时写成 null）与 degenerate_mode_warning 在通常的点、纯膜系、奇点附近和奇点上的取值。
//     这一部分只在 rcwa_json.hpp 编入了 rcwa1d_conical.hpp 时才有；没有编入时改为检查 conical 请求得到明确的报错。
// “逐位相同”的比法：把应答重新解析成数，与直接调用得到的 double 按内存里的 8 个字节比较。
#include <clocale>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "stablercwa/json.hpp"
#include "stablercwa/rcwa_json.hpp"

using namespace stablercwa;
namespace rcwa = stablercwa::rcwa;
using rcwa::Polarization;

// ---------------------------------------------------------------------------------------------
// 输出与计数（写法同 tests/test_rcwa1d.cpp：文字一律经 say() 输出，宏里不出现中文）
#if defined(__GNUC__) && !defined(__MINGW32__)
[[gnu::format(printf, 1, 2)]]
#endif
static void say(const char* format, ...) {
    std::va_list args;
    va_start(args, format);
    std::vprintf(format, args);
    va_end(args);
}

static int g_pass = 0, g_total = 0;

static void check(bool ok, const char* file, int line, const char* text) {
    ++g_total;
    if (ok) ++g_pass;
    else say("  失败 %s:%d  %s\n", file, line, text);
}
static void check_throws(bool caught, const char* file, int line, const char* text, const char* type) {
    ++g_total;
    if (caught) ++g_pass;
    else say("  失败 %s:%d  %s 没有抛出 %s\n", file, line, text, type);
}

#define CHECK(cond) check(static_cast<bool>(cond), __FILE__, __LINE__, #cond)

#define CHECK_THROWS(expr, exception_type)                                        \
    do {                                                                          \
        bool caught_ = false;                                                     \
        try { (void)(expr); } catch (const exception_type&) { caught_ = true; }   \
        check_throws(caught_, __FILE__, __LINE__, #expr, #exception_type);        \
    } while (0)

// ---------------------------------------------------------------------------------------------
// 比较工具

static bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; }

// 应答里的一个数 v 与直接调用得到的 x 是否逐位相同；x 是 NaN / Inf 时应答里应当是 null。
static bool is_number(const json::Value& v, double x) {
    if (!std::isfinite(x)) return v.is_null();
    return v.is_number() && same_bits(v.as_double(), x);
}
static bool is_int(const json::Value& v, int x) {
    int got = 0;
    return v.is_integer() && v.to_int(got) && got == x;
}
static bool is_complex(const json::Value& v, cplx z) {             // [实部, 虚部]
    return v.is_array() && v.size() == 2 && is_number(v.at(0), z.real()) && is_number(v.at(1), z.imag());
}
static bool are_numbers(const json::Value& v, const RealVector& x) {
    if (!v.is_array() || v.size() != x.size()) return false;
    for (std::size_t j = 0; j < x.size(); ++j)
        if (!is_number(v.at(j), x[j])) return false;
    return true;
}
static bool are_complex(const json::Value& v, const Vector& z) {
    if (!v.is_array() || v.size() != z.size()) return false;
    for (std::size_t j = 0; j < z.size(); ++j)
        if (!is_complex(v.at(j), z[j])) return false;
    return true;
}
// 对象的键恰好是 keys，次序也相同
static bool has_keys(const json::Value& v, std::initializer_list<const char*> keys) {
    if (!v.is_object() || v.size() != keys.size()) return false;
    std::size_t i = 0;
    for (const char* k : keys)
        if (v.key(i++) != k) return false;
    return true;
}
static bool contains(const std::string& text, const char* part) { return text.find(part) != std::string::npos; }

// 应答开头的几项：ok、version、mode、M、orders、threads，以及 results 的个数
static bool head_matches(const json::Value& reply, const char* mode, int M, int threads, std::size_t count) {
    if (!has_keys(reply, {"ok", "version", "mode", "M", "orders", "threads", "results"})) return false;
    if (!reply.at("ok").is_bool() || !reply.at("ok").as_bool() || !is_int(reply.at("version"), 1)) return false;
    if (!reply.at("mode").is_string() || reply.at("mode").as_string() != mode) return false;
    if (!is_int(reply.at("M"), M) || !is_int(reply.at("threads"), threads)) return false;
    const json::Value orders = reply.at("orders");
    if (!orders.is_array() || orders.size() != static_cast<std::size_t>(2 * M + 1)) return false;
    for (int m = -M; m <= M; ++m)
        if (!is_int(orders.at(static_cast<std::size_t>(m + M)), m)) return false;
    return reply.at("results").is_array() && reply.at("results").size() == count;
}

// 每个结果末尾的几项：与直接调用的同名成员相同；用时是非负的数（两次求解的用时自然不同，不比较数值）
template <class AnyResult>
static bool bookkeeping_matches(const json::Value& item, const AnyResult& r) {
    if (!is_int(item.at("eig_info"), r.eig_info) || !is_int(item.at("n_eig"), r.n_eig) ||
        !is_int(item.at("n_dense_solve"), r.n_dense_solve))
        return false;
    if (!is_number(item.at("max_eig_residual"), r.max_eig_residual) ||
        !is_number(item.at("max_solve_backward_error"), r.max_solve_backward_error))
        return false;
    const json::Value timing = item.at("timing");
    if (!has_keys(timing, {"total", "eig", "linear", "product"})) return false;
    for (std::size_t i = 0; i < timing.size(); ++i)
        if (!timing.at(i).is_number() || !(timing.at(i).as_double() >= 0.0)) return false;
    return timing.at("total").as_double() > 0.0;
}

// planar 应答里的一个结果与直接调用的结果逐位相同
static bool planar_item_matches(const json::Value& item, const rcwa::Point& p, const rcwa::Result& r) {
    if (!has_keys(item, {"wavelength", "theta_deg", "R", "T", "DEr", "DEt", "R_total", "T_total", "eig_info", "n_eig",
                         "n_dense_solve", "max_eig_residual", "max_solve_backward_error", "timing"}))
        return false;
    return is_number(item.at("wavelength"), p.wavelength) && is_number(item.at("theta_deg"), p.theta_deg) &&
           are_complex(item.at("R"), r.R) && are_complex(item.at("T"), r.T) && are_numbers(item.at("DEr"), r.DEr) &&
           are_numbers(item.at("DEt"), r.DEt) && is_number(item.at("R_total"), r.reflectance()) &&
           is_number(item.at("T_total"), r.transmittance()) && bookkeeping_matches(item, r);
}

// ---------------------------------------------------------------------------------------------
// 测试用的结构（长度单位：微米）

// rcwa_json.hpp 文件头的例子：一个光栅层加一层高折射率膜
static rcwa::Stack example_stack() {
    rcwa::Stack s;
    s.nI = 1.0;
    s.nII = 1.46;
    s.period = 0.574;
    s.layers = {rcwa::Layer::grating(0.38, {0.0, 0.1722, 0.574}, {1.46, 1.0}), rcwa::Layer::uniform(0.115, 1.96)};
    return s;
}
// 两个不同的光栅层夹一层吸收膜；齿有吸收，入射介质 n = 1.2，出射介质的折射率由参数给出
static rcwa::Stack lossy_stack(cplx nII) {
    rcwa::Stack s;
    s.nI = 1.2;
    s.nII = nII;
    s.period = 0.9;
    s.layers = {rcwa::Layer::grating(0.23, {0.0, 0.21, 0.47, 0.9}, {1.7, 1.0, cplx(2.1, 0.05)}),
                rcwa::Layer::uniform(0.19, cplx(1.4, 0.02)),
                rcwa::Layer::grating(0.31, {-0.12, 0.17, 0.78}, {1.0, cplx(1.9, 0.3)})};
    return s;
}
// 光栅层下面 20 层介质膜：3L · H · (2L · H)^9
static rcwa::Stack film_stack_grating() {
    rcwa::Stack s;
    s.nI = 1.0;
    s.nII = 1.46;
    s.period = 0.574;
    s.layers.push_back(rcwa::Layer::grating(0.38, {0.0, 0.574 * 0.3, 0.574}, {1.46, 1.0}));
    s.layers.push_back(rcwa::Layer::uniform(0.46438, 1.46));
    s.layers.push_back(rcwa::Layer::uniform(0.115306, 1.96));
    for (int k = 0; k < 9; ++k) {
        s.layers.push_back(rcwa::Layer::uniform(0.30958, 1.46));
        s.layers.push_back(rcwa::Layer::uniform(0.115306, 1.96));
    }
    return s;
}
// 纯膜系（没有光栅层），其中一层有吸收
static rcwa::Stack pure_films() {
    rcwa::Stack s;
    s.nI = 1.0;
    s.nII = 1.52;
    s.period = 1.0;
    s.layers = {rcwa::Layer::uniform(0.1, 2.3), rcwa::Layer::uniform(0.2, 1.38), rcwa::Layer::uniform(0.05, cplx(1.7, 0.3))};
    return s;
}
// 侧壁倾斜的齿切成 slices 片：多个光栅层，各要一次本征分解
static rcwa::Stack trapezoid_stack(int slices) {
    const double base = 0.574, depth = 0.38;
    rcwa::Stack s;
    s.nI = 1.0;
    s.nII = 1.46;
    s.period = base;
    for (int k = 0; k < slices; ++k) {
        const double w = base * (0.2 + 0.25 * (k + 0.5) / slices), x0 = 0.5 * (base - w);
        s.layers.push_back(rcwa::Layer::grating(depth / slices, {0.0, x0, x0 + w, base}, {1.0, 1.46, 1.0}));
    }
    s.layers.push_back(rcwa::Layer::uniform(0.115306, 1.96));
    return s;
}
// 银光栅（光栅层本身强吸收），衬底也是银
static rcwa::Stack silver_stack() {
    const cplx silver{0.079095, 5.42875};
    rcwa::Stack s;
    s.nI = 1.0;
    s.nII = silver;
    s.period = 0.6;
    s.layers = {rcwa::Layer::grating(0.10, {-0.1, 0.15, 0.5}, {silver, 1.0})};
    return s;
}
// 出射介质里 ±2 级恰好掠射、上面又紧挨着光栅层（λ = Λ、正入射时）：求解按数值失败处理，结果全部是 NaN
static rcwa::Stack rayleigh_stack() {
    const double p = 0.5747, f = 0.30;
    rcwa::Stack s;
    s.nI = 1.0;
    s.nII = 2.0;
    s.period = p;
    s.layers = {rcwa::Layer::grating(0.38, {-f * p / 2, f * p / 2, p - f * p / 2}, {1.45, 1.0})};
    return s;
}

// ---------------------------------------------------------------------------------------------
// 由 C++ 的结构拼出请求。数都经 json::dump 写出，读回来与原来的 double 逐位相同。

// 请求的几种写法（可以相加）
static const int kPairs = 1;          // 折射率一律写成 [实部, 虚部]（否则实折射率写成一个数）
static const int kTopLevel = 2;       // 单个入射点直接写在最外层（否则写 points 数组）
static const int kNoOptions = 4;      // 不写 options
static const int kIndented = 8;       // 请求本身分行缩进

static json::Value index_json(cplx n, bool pairs) {
    if (!pairs && n.imag() == 0.0) return json::Value(n.real());
    json::Value pair = json::Value::array();
    pair.push_back(n.real());
    pair.push_back(n.imag());
    return pair;
}
static json::Value stack_json(const rcwa::Stack& s, bool pairs) {
    json::Value layers = json::Value::array();
    for (const rcwa::Layer& layer : s.layers) {
        json::Value item = json::Value::object();
        item.set("d", layer.d);
        if (layer.homogeneous) {
            item.set("n", index_json(layer.n, pairs));
        } else {
            json::Value edges = json::Value::array(), values = json::Value::array();
            for (double x : layer.edges) edges.push_back(x);
            for (const cplx& n : layer.n_values) values.push_back(index_json(n, pairs));
            item.set("edges", std::move(edges));
            item.set("n_values", std::move(values));
        }
        layers.push_back(std::move(item));
    }
    json::Value v = json::Value::object();
    v.set("nI", index_json(s.nI, pairs));
    v.set("nII", index_json(s.nII, pairs));
    v.set("period", s.period);
    v.set("layers", std::move(layers));
    return v;
}
// threads <= 0 时不写 threads
static json::Value options_json(const rcwa::Options& o, int threads) {
    json::Value v = json::Value::object();
    v.set("eig", o.eig == EigMethod::LR ? "lr" : "qr");
    v.set("linear", o.linear == rcwa::LinearSolver::LUPartial ? "lu"
                    : o.linear == rcwa::LinearSolver::LUNoPivot ? "lu_nopivot" : "qr");
    v.set("homogeneous_fast_path", o.homogeneous_fast_path);
    v.set("parallel_layers", o.parallel_layers);
    v.set("diagnostics", o.diagnostics);
    if (threads > 0) v.set("threads", threads);
    return v;
}
static std::string planar_request(const rcwa::Stack& s, int M, Polarization pol, const std::vector<rcwa::Point>& points,
                                  const rcwa::Options& opt, int threads, int style) {
    json::Value v = json::Value::object();
    v.set("mode", "planar");
    v.set("stack", stack_json(s, (style & kPairs) != 0));
    v.set("M", M);
    v.set("polarization", pol == Polarization::TE ? "TE" : "TM");
    if ((style & kTopLevel) != 0 && points.size() == 1) {
        v.set("wavelength", points[0].wavelength);
        v.set("theta_deg", points[0].theta_deg);
    } else {
        json::Value list = json::Value::array();
        for (const rcwa::Point& p : points) {
            json::Value item = json::Value::object();
            item.set("wavelength", p.wavelength);
            item.set("theta_deg", p.theta_deg);
            list.push_back(std::move(item));
        }
        v.set("points", std::move(list));
    }
    if ((style & kNoOptions) == 0) v.set("options", options_json(opt, threads));
    return json::dump(v, (style & kIndented) != 0 ? 2 : -1);
}

// 同一个问题走两条路：经 solve_json，以及直接调用 solve（单点）或 sweep（多点）；两边的数应当逐位相同。
// threads > 0 时请求里写 threads，直接调用那一边先 set_num_threads(threads)。
static bool planar_agrees(const rcwa::Stack& s, int M, Polarization pol, const std::vector<rcwa::Point>& points,
                          const rcwa::Options& opt = {}, int threads = 0, int style = 0) {
    const rcwa::JsonReply reply = rcwa::solve_json_reply(planar_request(s, M, pol, points, opt, threads, style));
    const int before = num_threads();
    if (threads > 0) set_num_threads(threads);
    const int used = num_threads();
    std::vector<rcwa::Result> direct;
    if (points.size() == 1) direct.push_back(rcwa::solve(s, points[0].wavelength, points[0].theta_deg, M, pol, opt));
    else direct = rcwa::sweep(s, points, M, pol, opt);
    set_num_threads(before);
    if (!reply.ok) {
        say("    应答：%s\n", reply.text.c_str());
        return false;
    }
    const json::Value v = json::parse(reply.text);
    if (!head_matches(v, "planar", M, used, points.size())) return false;
    for (std::size_t k = 0; k < points.size(); ++k)
        if (!planar_item_matches(v.at("results").at(k), points[k], direct[k])) return false;
    return true;
}

// ---------------------------------------------------------------------------------------------
// JSON 的读

// (2m + 1)·5^1075 的十进制数字。最小的那个二进制指数上，相邻的两个 double m·2^-1074 与 (m + 1)·2^-1074 的中点是
// (2m + 1)·2^-1075 = (2m + 1)·5^1075 × 10^-1075，所以这串数字后面接上 "e-1075" 就恰好是那个中点。
static std::string midpoint_digits(std::uint64_t m) {
    std::string digits = std::to_string(2 * m + 1);            // 高位在前
    for (int k = 0; k < 1075; ++k) {                           // 逐次乘 5（竖式，从低位算起）
        int carry = 0;
        for (std::size_t i = digits.size(); i-- > 0;) {
            const int v = (digits[i] - '0') * 5 + carry;
            digits[i] = static_cast<char>('0' + v % 10);
            carry = v / 10;
        }
        if (carry != 0) digits.insert(digits.begin(), static_cast<char>('0' + carry));
    }
    return digits;
}

// text 应当解析失败，而且报的位置是 offset、说明里含有 part
static bool fails_at(const std::string& text, std::size_t offset, const char* part = "") {
    try {
        json::parse(text);
    } catch (const json::ParseError& e) {
        const bool ok = e.offset() == offset && contains(e.reason(), part);
        if (!ok) say("    实际：%s\n", e.what());
        return ok;
    }
    say("    实际：没有报错\n");
    return false;
}

static void test_json_values() {
    say("[JSON：值的类型与取值]\n");
    const json::Value none;
    CHECK(none.is_null() && none.type() == json::Type::Null && none.size() == 0);
    CHECK(json::Value(nullptr).is_null());
    CHECK(json::Value(true).is_bool() && json::Value(true).as_bool() && !json::Value(false).as_bool());
    CHECK(json::Value(3).is_number() && json::Value(3).is_integer() && json::Value(3).as_double() == 3.0);
    CHECK(json::Value(2.5).is_number() && !json::Value(2.5).is_integer());
    CHECK(json::Value(3.0).is_number() && !json::Value(3.0).is_integer());          // 按浮点数给的 3.0 不算“整数”
    CHECK(json::Value("abc").is_string() && json::Value(std::string("abc")).as_string() == "abc");
    CHECK(json::Value::array().is_array() && json::Value::object().is_object());
    // to_int：数值是整数、又在 int 范围内
    int k = -1;
    CHECK(json::Value(20).to_int(k) && k == 20);
    CHECK(json::Value(-7.0).to_int(k) && k == -7);
    CHECK(json::Value(2147483647.0).to_int(k) && k == 2147483647);
    CHECK(json::Value(-2147483648.0).to_int(k) && k == -2147483647 - 1);
    k = 99;
    CHECK(!json::Value(2.5).to_int(k) && !json::Value(2147483648.0).to_int(k) && !json::Value(-2147483649.0).to_int(k));
    CHECK(!json::Value(std::numeric_limits<double>::quiet_NaN()).to_int(k) && !json::Value("7").to_int(k) && k == 99);
    // 达到 2^53 的整数不再记为“整数”
    CHECK(json::Value(9007199254740991LL).is_integer() && json::Value(-9007199254740991LL).is_integer() &&
          !json::Value(9007199254740992LL).is_integer() && !json::Value(-9007199254740993LL).is_integer());

    // 数组与对象
    json::Value a;
    a.push_back(1);
    a.push_back("x");
    CHECK(a.is_array() && a.size() == 2 && a.at(0).as_double() == 1.0 && a.at(1).as_string() == "x");
    json::Value o;
    o.set("b", 1);
    o.set("a", 2);
    o.set("c", 3);
    o.set("a", "again");                                       // 已有的键：换掉值，位置不变
    CHECK(o.is_object() && o.size() == 3 && o.key(0) == "b" && o.key(1) == "a" && o.key(2) == "c");
    CHECK(o.at("a").as_string() == "again" && o.find("c") != nullptr && o.find("zz") == nullptr && o.contains("b"));
    CHECK(o.at(2).as_double() == 3.0 && a.find("x") == nullptr);
    json::Value copy = o;                                      // 拷贝是深拷贝
    copy.set("b", 10);
    CHECK(o.at("b").as_double() == 1.0 && copy.at("b").as_double() == 10.0);
    // 类型不对、越界、没有这个键
    CHECK_THROWS(json::Value(1).as_bool(), std::invalid_argument);
    CHECK_THROWS(json::Value("1").as_double(), std::invalid_argument);
    CHECK_THROWS(json::Value(1).as_string(), std::invalid_argument);
    CHECK_THROWS(a.at(2), std::invalid_argument);
    CHECK_THROWS(o.at("zz"), std::invalid_argument);
    CHECK_THROWS(o.key(3), std::invalid_argument);
    CHECK_THROWS(a.set("k", 1), std::invalid_argument);
    CHECK_THROWS(o.push_back(1), std::invalid_argument);
}

static void test_json_parse() {
    say("[JSON：读]\n");
    CHECK(json::parse("null").is_null() && json::parse("true").as_bool() && !json::parse("false").as_bool());
    CHECK(json::parse(" \t\r\n 12 \n").as_double() == 12.0);

    // 数
    {
        const json::Value zero = json::parse("0"), minus_zero = json::parse("-0");
        CHECK(zero.is_integer() && zero.as_double() == 0.0 && !std::signbit(zero.as_double()));
        CHECK(!minus_zero.is_integer() && minus_zero.as_double() == 0.0 && std::signbit(minus_zero.as_double()));
        CHECK(json::parse("-17").is_integer() && json::parse("-17").as_double() == -17.0);
        CHECK(!json::parse("3.25").is_integer() && json::parse("3.25").as_double() == 3.25);
        CHECK(!json::parse("20.0").is_integer() && json::parse("20.0").as_double() == 20.0);
        CHECK(json::parse("1e3").as_double() == 1000.0 && json::parse("1E+3").as_double() == 1000.0 &&
              !json::parse("1e3").is_integer());
        CHECK(same_bits(json::parse("1.5e-3").as_double(), 1.5e-3) && same_bits(json::parse("-2.5E-7").as_double(), -2.5e-7));
        CHECK(same_bits(json::parse("0.1").as_double(), 0.1) && same_bits(json::parse("1.053").as_double(), 1.053));
        CHECK(same_bits(json::parse("0.30000000000000004").as_double(), 0.1 + 0.2));
        CHECK(same_bits(json::parse("1.7976931348623157e308").as_double(), (std::numeric_limits<double>::max)()));
        CHECK(same_bits(json::parse("5e-324").as_double(), std::numeric_limits<double>::denorm_min()));
        CHECK(json::parse("1e-400").as_double() == 0.0);                    // 下溢成零不算错
        CHECK(json::parse("9007199254740991").is_integer() && json::parse("-9007199254740991").as_double() == -9007199254740991.0);
        CHECK(!json::parse("9007199254740992").is_integer() && json::parse("9007199254740992").as_double() == 9007199254740992.0);
        CHECK(!json::parse("9007199254740993").is_integer() && json::parse("9007199254740993").as_double() == 9007199254740992.0);
        CHECK(!json::parse("18014398509481984").is_integer() && json::parse("18014398509481984").as_double() == 18014398509481984.0);
        CHECK(json::parse("123456789012").is_integer() && json::parse("123456789012").as_double() == 123456789012.0);
        // 很长的数字串也按最接近的 double 读
        CHECK(same_bits(json::parse("0.1000000000000000055511151231257827021181583404541015625").as_double(), 0.1));
        CHECK(same_bits(json::parse("3.14159265358979323846264338327950288419716939937510582097494459").as_double(), kPi));
        const std::string many_zeros = "0." + std::string(400, '0') + "1e401";
        CHECK(same_bits(json::parse(many_zeros).as_double(), 1.0));
        // 紧挨在 2 的幂下面的数（尾数全是 1），不能进位成 2 的幂
        CHECK(same_bits(json::parse("9.5466761359362636e-153").as_double(), std::nextafter(std::ldexp(1.0, -505), 0.0)) &&
              same_bits(json::parse("9.546676135936264e-153").as_double(), std::nextafter(std::ldexp(1.0, -505), 0.0)));
        CHECK(same_bits(json::parse("1.868775676978051e+196").as_double(), std::nextafter(std::ldexp(1.0, 652), 0.0)));
        // 恰在两个 double 正中间的数向偶数舍入：6973021663144072 × 2^-8 与下一个 double 的中点
        CHECK(same_bits(json::parse("27238365871656.533203125").as_double(), std::ldexp(6973021663144072.0, -8)) &&
              same_bits(json::parse("2.7238365871656533203125e13").as_double(), std::ldexp(6973021663144072.0, -8)));
        // double 范围的两头：到了最大的数与 2^1024 的中点就算超出范围；不到最小的正数的一半（2^-1075）舍入成零
        CHECK(same_bits(json::parse("1.7976931348623158e308").as_double(), (std::numeric_limits<double>::max)()) &&
              fails_at("1.7976931348623159e308", 0, "range of double") && fails_at("-1.7976931348623159e308", 0, "range of double"));
        CHECK(same_bits(json::parse("2.4703282292062328e-324").as_double(), std::numeric_limits<double>::denorm_min()) &&
              same_bits(json::parse("2.4703282292062327e-324").as_double(), 0.0) &&
              same_bits(json::parse("-2.4703282292062327e-324").as_double(), -0.0) && same_bits(json::parse("-1e-400").as_double(), -0.0));
        // 指数的位数、前导零再多也一样（这几个数不到 768 个字符，直接交给库函数换算）
        CHECK(fails_at("1e99999999999999999999", 0, "range of double") && fails_at("[-1e+99999999999999999999]", 1, "range of double") &&
              same_bits(json::parse("1e-99999999999999999999").as_double(), 0.0) &&
              same_bits(json::parse("0e99999999999999999999").as_double(), 0.0));
        CHECK(same_bits(json::parse("0." + std::string(700, '0') + "1e701").as_double(), 1.0) &&
              same_bits(json::parse("1e+" + std::string(700, '0') + "5").as_double(), 1e5) &&
              same_bits(json::parse("15e-" + std::string(700, '0') + "1").as_double(), 1.5));
    }
    // 超过 768 个字符的数先化成不超过 768 位有效数字再换算（见 json.hpp 的 Parser::convert_reduced），结果仍然正确舍入
    {
        const std::string zeros(1100, '0'), nines(1100, '9');
        CHECK(same_bits(json::parse("0." + std::string(2000, '0') + "1e2001").as_double(), 1.0));
        CHECK(same_bits(json::parse("1" + std::string(2000, '0') + "e-2000").as_double(), 1.0));
        CHECK(same_bits(json::parse("1.053" + zeros).as_double(), 1.053) && same_bits(json::parse("-66.36" + zeros + "e0").as_double(), -66.36));
        // 1 + 2^-53 恰好是 1 与下一个 double 的中点：向偶数舍入得 1；后面隔得再远，只要多出一点就进位
        const std::string mid = "1.00000000000000011102230246251565404236316680908203125";
        CHECK(same_bits(json::parse(mid).as_double(), 1.0) && same_bits(json::parse(mid + zeros).as_double(), 1.0));
        CHECK(same_bits(json::parse(mid + zeros + "1").as_double(), std::nextafter(1.0, 2.0)));
        CHECK(same_bits(json::parse("1.00000000000000011102230246251565404236316680908203124" + nines).as_double(), 1.0));
        // 2^53 + 1 是 2^53 与 2^53 + 2 的中点
        CHECK(same_bits(json::parse("9007199254740993." + zeros).as_double(), 9007199254740992.0));
        CHECK(same_bits(json::parse("9007199254740993." + zeros + "1").as_double(), 9007199254740994.0));
        CHECK(same_bits(json::parse("9007199254740993" + zeros + "e-1100").as_double(), 9007199254740992.0));
        CHECK(same_bits(json::parse("9007199254740993" + zeros + "1e-1101").as_double(), 9007199254740994.0));
        CHECK(same_bits(json::parse("9007199254740992" + nines + "e-1100").as_double(), 9007199254740992.0));
        CHECK(same_bits(json::parse("-9007199254740994." + nines).as_double(), -9007199254740994.0));   // 差一点不到下一个中点
        CHECK(same_bits(json::parse("9007199254740995." + zeros).as_double(), 9007199254740996.0));     // 中点：向偶数舍入是进位
        // 零、下溢、溢出，以及位数很多的指数
        const json::Value minus_zero = json::parse("-0." + zeros + "e+5");
        CHECK(minus_zero.as_double() == 0.0 && std::signbit(minus_zero.as_double()) && !minus_zero.is_integer());
        CHECK(json::parse("1e-" + nines).as_double() == 0.0 && json::parse("0e" + nines).as_double() == 0.0);
        CHECK(fails_at("1" + zeros, 0, "range of double") && fails_at("[1e" + nines + "]", 1, "range of double"));
        CHECK(same_bits(json::parse("1e+" + zeros + "5").as_double(), 1e5));
        CHECK(same_bits(json::parse("17976931348623157" + zeros + "e-808").as_double(), (std::numeric_limits<double>::max)()));
        CHECK(same_bits(json::parse("4.9406564584124654" + zeros + "e-324").as_double(), std::numeric_limits<double>::denorm_min()));
        CHECK(fails_at("17976931348623159" + zeros + "e-808", 0, "range of double") &&
              same_bits(json::parse("-2.47032822920623271" + nines + "e-324").as_double(), -0.0) &&       // 2^-1075 = 2.4703282292062327208…e-324
              same_bits(json::parse("2.47032822920623273" + zeros + "e-324").as_double(), std::numeric_limits<double>::denorm_min()));
        // 768 个字符（直接换算与先化简的交界）两边：中点后面隔着一串零再多出一点，仍要进位；全是零则不进位
        bool same = true;
        for (std::size_t total = 760; total <= 780; ++total) {
            const std::string padding(total - mid.size() - 1, '0');
            same = same && same_bits(json::parse(mid + padding + "1").as_double(), std::nextafter(1.0, 2.0)) &&
                   same_bits(json::parse(mid + padding + "0").as_double(), 1.0);
        }
        CHECK(same);
    }
    // 有效数字最多的中点恰有 768 位（m 接近 2^53 时），正是 convert_reduced 保留的位数；这一位之后才多出一点的数、
    // 这一位少 1 而后面全是 9 的数，都要与恰在中点上的数分得清
    {
        const std::string zeros(1100, '0'), nines(1100, '9');
        const std::uint64_t m = (std::uint64_t(1) << 53) - 2;                              // 偶数：恰在中点时舍入到它自己
        const double lower = std::ldexp(static_cast<double>(m), -1074), upper = std::ldexp(static_cast<double>(m + 1), -1074);
        const std::string digits = midpoint_digits(m), less = digits.substr(0, 767) + "4";
        CHECK(digits.size() == 768 && digits.back() == '5' && midpoint_digits(0).size() == 752);
        CHECK(same_bits(json::parse(digits + "e-1075").as_double(), lower) && same_bits(json::parse(digits + zeros + "e-2175").as_double(), lower));
        CHECK(same_bits(json::parse(digits + "1e-1076").as_double(), upper) && same_bits(json::parse(digits + zeros + "1e-2176").as_double(), upper));
        CHECK(same_bits(json::parse(less + "9e-1076").as_double(), lower) && same_bits(json::parse(less + nines + "e-2175").as_double(), lower));
        // 同一个中点写成小数（小数点后先有 307 个零），带负号
        const std::string plain = "-0." + std::string(307, '0') + digits;
        CHECK(same_bits(json::parse(plain).as_double(), -lower) && same_bits(json::parse(plain + zeros + "3").as_double(), -upper));
        // 下一个中点（m + 1 是奇数）：恰在中点时进位到 m + 2 = 2^53，即 2^-1021；略低于中点则不进位
        const std::string odd = midpoint_digits(m + 1);
        CHECK(odd.size() == 768 && same_bits(json::parse(odd + "e-1075").as_double(), std::ldexp(1.0, -1021)) &&
              same_bits(json::parse(odd.substr(0, 767) + "4" + nines + "e-2175").as_double(), upper) &&
              same_bits(json::parse(odd + zeros + "7e-2176").as_double(), std::ldexp(1.0, -1021)));
        // 最小的中点 2^-1075（752 位）：恰在中点时舍入成零，多出一点就是最小的正数
        const std::string least = midpoint_digits(0);
        CHECK(same_bits(json::parse(least + "e-1075").as_double(), 0.0) && same_bits(json::parse(least + zeros + "e-2175").as_double(), 0.0) &&
              same_bits(json::parse(least + zeros + "1e-2176").as_double(), std::numeric_limits<double>::denorm_min()));
    }

    // 字符串：转义、\u（含代理对）、原样的 UTF-8
    {
        CHECK(json::parse("\"\"").as_string().empty() && json::parse("\"abc\"").as_string() == "abc");
        CHECK(json::parse("\"\\\"\\\\\\/\\b\\f\\n\\r\\t\"").as_string() == "\"\\/\b\f\n\r\t");
        const std::string escaped = "\"\\u0041\\u00e9\\u4E2d\\ud83d\\uDE00\"";      // A é 中 U+1F600
        CHECK(json::parse(escaped).as_string() == "A\xC3\xA9\xE4\xB8\xAD\xF0\x9F\x98\x80");
        const std::string raw = "\"A\xC3\xA9\xE4\xB8\xAD\xF0\x9F\x98\x80\"";        // 同样的字符直接写成 UTF-8
        CHECK(json::parse(raw).as_string() == "A\xC3\xA9\xE4\xB8\xAD\xF0\x9F\x98\x80");
        CHECK(json::parse("\"\\u0000\"").as_string() == std::string(1, '\0'));
        CHECK(json::parse("\"\\uFFFF\\u007f\\u0080\\u07ff\\u0800\"").as_string() ==
              "\xEF\xBF\xBF\x7F\xC2\x80\xDF\xBF\xE0\xA0\x80");
        CHECK(json::parse("\"\\udbff\\udfff\"").as_string() == "\xF4\x8F\xBF\xBF");   // U+10FFFF
        CHECK(json::parse("\"\x7F\"").as_string() == "\x7F");                        // 0x7F 不必转义
        const std::string chinese = "{\"\xE5\x90\x8D\xE5\xAD\x97\": \"\xE8\x86\x9C\xE5\xB1\x82\"}";   // 键、值都是汉字
        const json::Value v = json::parse(chinese);
        CHECK(v.size() == 1 && v.key(0) == "\xE5\x90\x8D\xE5\xAD\x97" && v.at(0).as_string() == "\xE8\x86\x9C\xE5\xB1\x82");
    }

    // 数组、对象、空白、键的次序
    {
        const json::Value v = json::parse("\n{ \"b\" : [ 1 , 2.5 , [ ] , { } ] ,\t\"a\":{\"x\":null,\"y\":[true,false]}\r\n,\"\":\"empty key\"}  ");
        CHECK(v.is_object() && v.size() == 3 && v.key(0) == "b" && v.key(1) == "a" && v.key(2).empty());
        CHECK(v.at("b").size() == 4 && v.at("b").at(1).as_double() == 2.5 && v.at("b").at(2).is_array() &&
              v.at("b").at(2).size() == 0 && v.at("b").at(3).is_object() && v.at("b").at(3).size() == 0);
        CHECK(v.at("a").at("x").is_null() && v.at("a").at("y").at(0).as_bool() && v.at("").as_string() == "empty key");
        CHECK(json::parse("[]").size() == 0 && json::parse("{}").is_object() && json::parse("[[[]]]").at(0).at(0).is_array());
        // 开头的 BOM 跳过
        CHECK(json::parse("\xEF\xBB\xBF[1]").at(0).as_double() == 1.0);
        // 指针加长度的接口：不要求以 '\0' 结尾，也不会读到给定的长度之外
        const char buffer[] = {'[', '1', '2', ']', '3', '4'};
        CHECK(json::parse(buffer, 4, json::kDefaultMaxDepth).at(0).as_double() == 12.0);
        CHECK(json::parse(buffer + 1, 2, json::kDefaultMaxDepth).as_double() == 12.0);
    }

    // 嵌套层数的上限
    {
        const auto nested = [](int depth) { return std::string(static_cast<std::size_t>(depth), '[') + std::string(static_cast<std::size_t>(depth), ']'); };
        CHECK(json::parse(nested(json::kDefaultMaxDepth)).is_array());
        CHECK(fails_at(nested(json::kDefaultMaxDepth + 1), static_cast<std::size_t>(json::kDefaultMaxDepth), "nested too deeply"));
        CHECK(json::parse("[[1]]", 2).is_array() && fails_at(std::string(1000000, '['), static_cast<std::size_t>(json::kDefaultMaxDepth)));
        CHECK_THROWS(json::parse("[[1]]", 1), json::ParseError);
        CHECK_THROWS(json::parse("[]", 0), json::ParseError);
        CHECK(json::parse("1", 0).as_double() == 1.0);
        std::string mixed;                                     // 对象与数组交替嵌套
        for (int k = 0; k < 64; ++k) mixed += "{\"k\":[";
        mixed += "0";
        for (int k = 0; k < 64; ++k) mixed += "]}";
        CHECK(json::parse(mixed).is_object());
        CHECK_THROWS(json::parse("[" + mixed + "]"), json::ParseError);
    }

    // 项数很多的对象：重复的键仍然查得出（走排序的那条路）
    {
        std::string big = "{";
        for (int k = 0; k < 2000; ++k) big += (k ? ",\"k" : "\"k") + std::to_string(k) + "\":" + std::to_string(k);
        const json::Value v = json::parse(big + "}");
        CHECK(v.size() == 2000 && v.at("k1999").as_double() == 1999.0 && v.key(7) == "k7");
        CHECK(fails_at(big + ",\"k1234\":0}", big.size() + 1, "duplicate key"));
    }
}

static void test_json_syntax_errors() {
    say("[JSON：语法错误与报错的位置]\n");
    struct Case {
        std::string text;
        std::size_t offset;
        const char* part;      // 说明里应当出现的一段
    };
    const std::string nul(1, '\0');
    const Case cases[] = {
        // 没有内容、内容不完整
        {"", 0, "end of input"},
        {"   ", 3, "end of input"},
        {"{", 1, "end of input"},
        {"[", 1, "end of input"},
        {"[1,", 3, "end of input"},
        {"[1", 2, "end of input"},
        {"{\"a\"", 4, "end of input"},
        {"{\"a\":", 5, "end of input"},
        {"{\"a\":1", 6, "end of input"},
        {"{\"a\":1,", 7, "end of input"},
        {"\xEF\xBB\xBF", 3, "end of input"},
        // 逗号、冒号、括号
        {"[1,]", 3, "unexpected character ']'"},
        {"[,1]", 1, "unexpected character ','"},
        {"[1 2]", 3, "expected ',' or ']'"},
        {"[\"a\":1]", 4, "expected ',' or ']'"},
        {"[1}", 2, "expected ',' or ']'"},
        {"{\"a\":1,}", 7, "expected a string key"},
        {"{\"a\" 1}", 5, "expected ':'"},
        {"{a:1}", 1, "expected a string key"},
        {"{1:2}", 1, "expected a string key"},
        {"{\"a\":1 \"b\":2}", 7, "expected ',' or '}'"},
        {"{\"a\":1,,\"b\":2}", 7, "expected a string key"},
        {"{\"a\":}", 5, "unexpected character '}'"},
        {"{\"a\":1]", 6, "expected ',' or '}'"},
        {"]", 0, "unexpected character ']'"},
        // 一段文字只能有一个值
        {"1 2", 2, "after the JSON value"},
        {"{} x", 3, "after the JSON value"},
        {"[1]]", 3, "after the JSON value"},
        {"null\n,", 5, "after the JSON value"},
        {"1.2.3", 3, "after the JSON value"},
        {"0x10", 1, "after the JSON value"},
        {"truefalse", 4, "after the JSON value"},
        {"\"a\" \"b\"", 4, "after the JSON value"},
        // 数
        {"01", 0, "leading zeros"},
        {"-01", 0, "leading zeros"},
        {"[1, 007]", 4, "leading zeros"},
        {"-", 1, "minus sign"},
        {"-x", 1, "minus sign"},
        {"--1", 1, "minus sign"},
        {"1.", 2, "decimal point"},
        {"1.e3", 2, "decimal point"},
        {"[1.]", 3, "decimal point"},
        {".5", 0, "unexpected character '.'"},
        {"+1", 0, "unexpected character '+'"},
        {"1e", 2, "exponent"},
        {"1e+", 3, "exponent"},
        {"1e-x", 3, "exponent"},
        {"[1e5.5]", 4, "expected ',' or ']'"},
        {"1e999", 0, "range of double"},
        {"-1e999", 0, "range of double"},
        {"[0, 2e308]", 4, "range of double"},
        // 字面量
        {"tru", 0, "invalid literal"},
        {"nul", 0, "invalid literal"},
        {"falsy", 0, "invalid literal"},
        {"[nil]", 1, "invalid literal"},
        {"True", 0, "unexpected character 'T'"},
        {"NaN", 0, "unexpected character 'N'"},
        {"Infinity", 0, "unexpected character 'I'"},
        {"-Infinity", 1, "minus sign"},
        {"undefined", 0, "unexpected character 'u'"},
        // 不属于 JSON 的写法
        {"'a'", 0, "unexpected character '''"},
        {"[1, // c\n2]", 4, "unexpected character '/'"},
        {"[1,2,/**/]", 5, "unexpected character '/'"},
        {"{\"a\":1;}", 6, "expected ',' or '}'"},
        {nul, 0, "unexpected byte 0x00"},
        {"[1," + nul + "]", 3, "unexpected byte 0x00"},
        {"\xEF\xBB\xBF[1,]", 6, "unexpected character ']'"},
        {"\xC2\xA0" "1", 0, "unexpected byte 0xC2"},           // 不换行空格不是 JSON 的空白
        // 字符串
        {"\"abc", 0, "unterminated string"},
        {"[\"abc]", 1, "unterminated string"},
        {"\"a\\", 0, "unterminated string"},
        {"\"a\\\"", 0, "unterminated string"},
        {"\"\\x\"", 1, "invalid escape"},
        {"\"ab\\'\"", 3, "invalid escape"},
        {"\"\\u12G4\"", 1, "four hexadecimal digits"},
        {"\"\\u12\"", 1, "four hexadecimal digits"},
        {"\"\\u", 1, "four hexadecimal digits"},
        {"\"\\uD800\"", 1, "unpaired surrogate"},
        {"\"\\uDC00\"", 1, "unpaired surrogate"},
        {"\"\\uD800\\u0041\"", 1, "unpaired surrogate"},
        {"\"\\uD800\\uD800\"", 1, "unpaired surrogate"},
        {"\"\\uD83D\\n\"", 1, "unpaired surrogate"},
        {"\"x\\ud83d\\ude0\"", 2, "four hexadecimal digits"},
        {"\"a\nb\"", 2, "control character"},
        {"\"a\tb\"", 2, "control character"},
        {"\"a" + nul + "\"", 2, "control character"},
        {"\"\x1F\"", 1, "control character"},
        // 不合法的 UTF-8
        {"\"\xFF\"", 1, "invalid UTF-8"},
        {"\"\x80\"", 1, "invalid UTF-8"},
        {"\"\xC0\x80\"", 1, "invalid UTF-8"},                  // 过长的编码
        {"\"\xC1\xBF\"", 1, "invalid UTF-8"},
        {"\"\xE0\x80\x80\"", 1, "invalid UTF-8"},
        {"\"\xF0\x80\x80\x80\"", 1, "invalid UTF-8"},
        {"\"\xE4\xB8\"", 1, "invalid UTF-8"},                  // 截断的三字节字符
        {"\"\xE4\xB8", 1, "invalid UTF-8"},
        {"\"\xC3\"", 1, "invalid UTF-8"},
        {"\"\xED\xA0\x80\"", 1, "invalid UTF-8"},              // 代理区直接编成 UTF-8
        {"\"\xF4\x90\x80\x80\"", 1, "invalid UTF-8"},          // 超过 U+10FFFF
        {"\"\xF5\x80\x80\x80\"", 1, "invalid UTF-8"},
        {"\"a\xE4\xB8\xAD\xFF\"", 5, "invalid UTF-8"},
        {"{\"\xFF\": 1}", 2, "invalid UTF-8"},                 // 键里的也查
        // 重复的键
        {"{\"a\":1,\"a\":2}", 7, "duplicate key"},
        {"{\"a\":1,\"b\":2,\"c\":{\"x\":1,\"y\":2,\"x\":3},\"a\":4}", 30, "duplicate key"},
        {"{\"a\":1,\"b\":2,\"\\u0061\":3}", 13, "duplicate key"},  // 转义之后相同的键也算重复
    };
    int index = 0;
    for (const Case& c : cases) {
        const std::string label = "syntax case #" + std::to_string(index++);
        check(fails_at(c.text, c.offset, c.part), __FILE__, __LINE__, label.c_str());
    }

    // 行、列与 what() 的样子；出错的类型可以当作 std::invalid_argument 接住
    {
        bool caught = false;
        try {
            json::parse("{\n  \"a\": 1,\n  \"b\": tru\n}");
        } catch (const json::ParseError& e) {
            caught = e.offset() == 19 && e.line() == 3 && e.column() == 8 &&
                     std::string(e.what()) == "json: invalid literal (expected true, false or null) at offset 19 (line 3, column 8)" &&
                     e.reason() == "invalid literal (expected true, false or null)";
        }
        CHECK(caught);
        CHECK_THROWS(json::parse("[1,]"), std::invalid_argument);
        // 列按字符数：两个汉字各占 3 个字节，只算两列
        const std::string wide = "[\"\xE4\xB8\xAD\xE6\x96\x87\", x]";
        caught = false;
        try {
            json::parse(wide);
        } catch (const json::ParseError& e) {
            caught = e.offset() == 11 && e.line() == 1 && e.column() == 8;
        }
        CHECK(caught);
        // BOM 不计入列
        caught = false;
        try {
            json::parse("\xEF\xBB\xBF[1,\r\n x]");
        } catch (const json::ParseError& e) {
            caught = e.offset() == 9 && e.line() == 2 && e.column() == 2;
        }
        CHECK(caught);
    }
}

// ---------------------------------------------------------------------------------------------
// JSON 的写

// 固定种子的伪随机数（xorshift64），不依赖随机设备
static std::uint64_t next_random(std::uint64_t& state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

// x 写出去再读回来是否逐位相同
static bool round_trips(double x) {
    const std::string text = json::dump(json::Value(x));
    const json::Value back = json::parse(text);
    return back.is_number() && same_bits(back.as_double(), x);
}

static void test_json_dump() {
    say("[JSON：写]\n");
    // 标量
    CHECK(json::dump(json::Value()) == "null" && json::dump(json::Value(true)) == "true" && json::dump(json::Value(false)) == "false");
    CHECK(json::dump(json::Value(0)) == "0" && json::dump(json::Value(-17)) == "-17" && json::dump(json::Value(2147483647)) == "2147483647");
    CHECK(json::dump(json::Value(9007199254740991LL)) == "9007199254740991" && json::dump(json::Value(-9007199254740991LL)) == "-9007199254740991");
    CHECK(json::dump(json::Value(9007199254740992LL)) == "9007199254740992.0" &&
          json::dump(json::Value(-9007199254740993LL)) == "-9007199254740992.0");   // 达到 2^53：当作浮点数
    CHECK(json::dump(json::Value(0.0)) == "0.0" && json::dump(json::Value(-0.0)) == "-0.0" && json::dump(json::Value(1.0)) == "1.0");
    CHECK(json::dump(json::Value(-20.0)) == "-20.0" && json::dump(json::Value(123456789.125)) == "123456789.125");
    // 最短的、读回来不变的写法
    CHECK(json::dump(json::Value(0.1)) == "0.1" && json::dump(json::Value(1.053)) == "1.053" && json::dump(json::Value(66.36)) == "66.36");
    CHECK(json::dump(json::Value(0.1 + 0.2)) == "0.30000000000000004" && json::dump(json::Value(1.0 / 3.0)) == "0.3333333333333333");
    CHECK(json::dump(json::Value(2.0 / 3.0)) == "0.6666666666666666" && json::dump(json::Value(kPi)) == "3.141592653589793");
    CHECK(json::dump(json::Value(1e21)) == "1e+21" && json::dump(json::Value(1e-7)) == "1e-07" && json::dump(json::Value(-2.5e-7)) == "-2.5e-07");
    CHECK(json::dump(json::Value(1e100)) == "1e+100" && json::dump(json::Value(1.5e300)) == "1.5e+300" && json::dump(json::Value(1e15)) == "1e+15");
    CHECK(json::dump(json::Value(123456789012345680.0)) == "1.2345678901234568e+17");
    CHECK(json::dump(json::Value((std::numeric_limits<double>::max)())) == "1.7976931348623157e+308");
    CHECK(json::dump(json::Value(std::numeric_limits<double>::epsilon())) == "2.220446049250313e-16");
    // 紧挨在 2 的幂下面的数（尾数全是 1）：16 位就读得回来，不必写 17 位
    CHECK(json::dump(json::Value(std::nextafter(std::ldexp(1.0, 652), 0.0))) == "1.868775676978051e+196" &&
          json::dump(json::Value(std::nextafter(std::ldexp(1.0, -505), 0.0))) == "9.546676135936264e-153");
    // NaN、Inf 写成 null
    CHECK(json::dump(json::Value(std::numeric_limits<double>::quiet_NaN())) == "null");
    CHECK(json::dump(json::Value(std::numeric_limits<double>::infinity())) == "null" &&
          json::dump(json::Value(-std::numeric_limits<double>::infinity())) == "null");

    // 数的往返：逐位相同
    {
        bool same = true;
        const double fixed[] = {0.0, -0.0, 1.0, -1.0, 0.1, 0.2, 0.3, 1.053, 66.36, 1e-300, 1e300, 5e-324, 2.2250738585072014e-308,
                                2.2250738585072009e-308, 1.7976931348623157e308, 4.35, 0.000001, 1e22, 1e23, 9007199254740993.0,
                                0.1 + 0.2, 1.0 / 3.0, 100.0, 1e15, 1e16, 1e17, 123456.789e-20, 8.41e21, 5e-7};
        for (double x : fixed) same = same && round_trips(x) && round_trips(-x);
        CHECK(same);
        same = true;
        for (int e = -1074; e <= 1023; ++e) {                  // 2 的各次幂及其相邻的数
            const double p = std::ldexp(1.0, e);
            same = same && round_trips(p) && round_trips(std::nextafter(p, 0.0)) && round_trips(std::nextafter(p, 2.0 * p));
        }
        CHECK(same);
        same = true;
        std::uint64_t state = 0x9E3779B97F4A7C15ull;
        int tried = 0;
        for (int k = 0; k < 20000; ++k) {                      // 任意的位模式（各个数量级都有）
            const std::uint64_t bits = next_random(state);
            double x = 0.0;
            std::memcpy(&x, &bits, sizeof x);
            if (!std::isfinite(x)) continue;
            ++tried;
            same = same && round_trips(x);
        }
        CHECK(same && tried > 19000);
        same = true;
        for (int k = 0; k < 20000; ++k) {                      // 小数位数不多的数：最短的写法也要读得回来
            const double x = static_cast<double>(static_cast<std::int64_t>(next_random(state) % 2000001) - 1000000) / 1000.0;
            same = same && round_trips(x) && json::dump(json::Value(x)).size() <= 9;
            const double y = static_cast<double>(next_random(state) % 1000000) * 1e-9;
            same = same && round_trips(y);
        }
        CHECK(same);
    }

    // 字符串的转义
    {
        const std::string tricky = std::string("a\"b\\c/d\b\f\n\r\t") + '\x01' + '\x1F' + '\x7F' + std::string(1, '\0') + "z";
        const std::string expected = "\"a\\\"b\\\\c/d\\b\\f\\n\\r\\t\\u0001\\u001f\x7F\\u0000z\"";
        CHECK(json::dump(json::Value(tricky)) == expected);
        CHECK(json::parse(json::dump(json::Value(tricky))).as_string() == tricky);
        const std::string utf8 = "A\xC3\xA9\xE4\xB8\xAD\xF0\x9F\x98\x80";               // UTF-8 原样写出
        CHECK(json::dump(json::Value(utf8)) == "\"" + utf8 + "\"");
        // 不合法的字节换成 U+FFFD，输出仍是合法的 UTF-8（读得回来）
        const std::string broken = std::string("a\xFF") + "b\xE4\xB8" + "c\xED\xA0\x80" + "d\xC0\x80";
        const std::string fixed = json::dump(json::Value(broken));
        CHECK(fixed == "\"a\xEF\xBF\xBD" "b\xEF\xBF\xBD\xEF\xBF\xBD" "c\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD" "d\xEF\xBF\xBD\xEF\xBF\xBD\"");
        CHECK(json::parse(fixed).is_string());
        // 键同样转义
        json::Value o = json::Value::object();
        o.set("k\n\"", 1);
        CHECK(json::dump(o) == "{\"k\\n\\\"\": 1}");
    }

    // 一行的写法、缩进的写法；键按插入顺序；写出去再读回来是同一个值
    {
        const std::string text = "{\"z\": 1, \"a\": [1, 2.5, \"x\", null, true], \"m\": {\"k\": [], \"j\": {}}, "
                                 "\"list\": [{\"p\": [1.0, 0.0]}, {\"q\": [[1, 2], [3, 4]]}, []], \"e\": \"\"}";
        const json::Value v = json::parse(text);
        CHECK(json::dump(v) == text);
        const std::string indented = "{\n"
                                     "  \"z\": 1,\n"
                                     "  \"a\": [1, 2.5, \"x\", null, true],\n"
                                     "  \"m\": {\n"
                                     "    \"k\": [],\n"
                                     "    \"j\": {}\n"
                                     "  },\n"
                                     "  \"list\": [\n"
                                     "    {\n"
                                     "      \"p\": [1.0, 0.0]\n"
                                     "    },\n"
                                     "    {\n"
                                     "      \"q\": [[1, 2], [3, 4]]\n"
                                     "    },\n"
                                     "    []\n"
                                     "  ],\n"
                                     "  \"e\": \"\"\n"
                                     "}";
        CHECK(json::dump(v, 2) == indented);
        CHECK(json::dump(json::parse(indented)) == text && json::dump(json::parse(json::dump(v, 0))) == text);
        CHECK(json::dump(v, 4) != indented && json::dump(json::parse(json::dump(v, 4)), 2) == indented);
        CHECK(json::dump(v) == json::dump(v) && json::dump(json::Value(v)) == text);       // 确定的输出；拷贝不改变内容
        CHECK(json::dump(json::Value::array()) == "[]" && json::dump(json::Value::object(), 2) == "{}" &&
              json::dump(json::parse("[1, [2, [3]]]"), 2) == "[1, [2, [3]]]");
        // 整数形式、浮点形式各自保持
        CHECK(json::dump(json::parse("[20, 20.0, 2e1, -0, 0, 1E2]")) == "[20, 20.0, 20.0, -0.0, 0, 100.0]");
    }

    // Writer：边算边写；次序不对时抛 std::logic_error
    {
        json::Writer w;
        w.begin_object();
        w.key("R");
        w.begin_array();
        w.value(0.5);
        w.value(std::numeric_limits<double>::quiet_NaN());
        w.value(3);
        w.value(-4LL);
        w.value(true);
        w.value("s");
        w.value(std::string("t"));
        w.null();
        w.end_array();
        w.key(std::string("empty"));
        w.begin_object();
        w.end_object();
        w.end_object();
        CHECK(w.complete() && w.str() == "{\"R\": [0.5, null, 3, -4, true, \"s\", \"t\", null], \"empty\": {}}");
        CHECK(w.take() == "{\"R\": [0.5, null, 3, -4, true, \"s\", \"t\", null], \"empty\": {}}" && w.str().empty() && !w.complete());
        w.value(1);                                            // take() 之后可以接着写下一个值
        CHECK(w.take() == "1");
        w.begin_array();                                       // 64 位整数的两端
        w.value((std::numeric_limits<long long>::min)());
        w.value((std::numeric_limits<long long>::max)());
        w.end_array();
        CHECK(w.take() == "[-9223372036854775808, 9223372036854775807]");

        json::Writer pretty(2);
        pretty.begin_object();
        pretty.key("a");
        pretty.begin_array(true);                              // 指定写在一行
        pretty.begin_object();                                 // 一行的数组里面的对象也跟着在一行
        pretty.key("x");
        pretty.value(1);
        pretty.end_object();
        pretty.end_array();
        pretty.key("b");
        pretty.begin_array();
        pretty.value(1);
        pretty.value(2);
        pretty.end_array();
        pretty.end_object();
        CHECK(pretty.take() == "{\n  \"a\": [{\"x\": 1}],\n  \"b\": [\n    1,\n    2\n  ]\n}");

        json::Writer bad;
        bad.begin_object();
        CHECK_THROWS(bad.value(1), std::logic_error);          // 对象里没写键就写值
        CHECK_THROWS(bad.begin_array(), std::logic_error);
        CHECK_THROWS(bad.end_array(), std::logic_error);       // 括号不配对
        bad.key("k");
        CHECK_THROWS(bad.key("again"), std::logic_error);      // 一个值只能有一个键
        CHECK_THROWS(bad.end_object(), std::logic_error);      // 键后面还没有值
        bad.value(1);
        bad.end_object();
        CHECK_THROWS(bad.value(2), std::logic_error);          // 最外层只能有一个值
        CHECK_THROWS(bad.end_object(), std::logic_error);
        json::Writer in_array;
        in_array.begin_array();
        CHECK_THROWS(in_array.key("k"), std::logic_error);     // 数组里不能写键
        CHECK_THROWS(json::Writer().key("k"), std::logic_error);
    }

    // 与 C locale 无关：小数点是逗号的 locale 下，读写的结果不变
    {
        bool switched = false, same = true;
        for (const char* name : {"de_DE.UTF-8", "de_DE", "fr_FR.UTF-8", "German_Germany.1252", "de-DE"}) {
            if (std::setlocale(LC_NUMERIC, name) == nullptr) continue;
            char probe[16];
            std::snprintf(probe, sizeof probe, "%.1f", 1.5);
            if (probe[1] == '.' && std::strtod("1.5", nullptr) == 1.5) continue;   // 这个 locale 的小数点仍是 '.'，试下一个
            switched = true;
            same = json::dump(json::Value(0.5)) == "0.5" && json::dump(json::Value(-1.053e-9)) == "-1.053e-09" &&
                   json::dump(json::Value(12.0)) == "12.0" && same_bits(json::parse("0.5").as_double(), 0.5) &&
                   same_bits(json::parse("[1.053, -2.5e-7]").at(1).as_double(), -2.5e-7) && round_trips(0.1 + 0.2) &&
                   round_trips(1.0 / 3.0) && fails_at("0,5", 1, "after the JSON value") &&
                   same_bits(json::parse("0." + std::string(80, '0') + "1").as_double(), 1e-81);
            break;
        }
        std::setlocale(LC_NUMERIC, "C");
        if (!switched) say("  （本机没有小数点为逗号的 locale，这一项只在默认 locale 下检查）\n");
        CHECK(same && json::dump(json::Value(0.5)) == "0.5" && same_bits(json::parse("0.5").as_double(), 0.5));
    }
}

// ---------------------------------------------------------------------------------------------
// 请求的检查

// 请求由几段拼成；每个出错的例子只改其中一段。
struct Parts {
    std::string mode = "\"mode\": \"planar\"";
    std::string stack = "\"stack\": {\"nI\": 1.0, \"nII\": 1.46, \"period\": 0.574, \"layers\": "
                        "[{\"d\": 0.38, \"edges\": [0.0, 0.1722, 0.574], \"n_values\": [1.46, 1.0]}, {\"d\": 0.115, \"n\": [1.96, 0.0]}]}";
    std::string order = "\"M\": 3";
    std::string polarization = "\"polarization\": \"TE\"";
    std::string points = "\"points\": [{\"wavelength\": 1.053, \"theta_deg\": 66.36}, {\"wavelength\": 1.06, \"theta_deg\": 60.0}]";
    std::string input;
    std::string options = "\"options\": {\"eig\": \"qr\", \"threads\": 1}";
    std::string extra;

    std::string text() const {
        std::string out;
        for (const std::string* part : {&mode, &stack, &order, &polarization, &points, &input, &options, &extra}) {
            if (part->empty()) continue;
            out += out.empty() ? "{" : ", ";
            out += *part;
        }
        return out.empty() ? "{}" : out + "}";
    }
};

static bool g_bad_edit = false;        // 测试自己写错了（要替换的片段不是恰好出现一次）

// 把 text 里恰好出现一次的 from 换成 to；from 为空时整段换成 to
static std::string edited(const std::string& text, const std::string& from, const std::string& to) {
    if (from.empty()) return to;
    const std::size_t at = text.find(from);
    if (at == std::string::npos || text.find(from, at + 1) != std::string::npos) {
        g_bad_edit = true;
        say("    测试写错了：片段 %s 不是恰好出现一次\n", from.c_str());
        return text;
    }
    return text.substr(0, at) + to + text.substr(at + from.size());
}

// 请求应当被拒绝：ok = false，应答恰好是 ok、error、where 三项，where 是预期的路径，error 里含有 part
static bool rejected(const std::string& request, const std::string& where, const char* part = "") {
    const rcwa::JsonReply reply = rcwa::solve_json_reply(request);
    bool ok = !reply.ok;
    try {
        const json::Value v = json::parse(reply.text);
        ok = ok && has_keys(v, {"ok", "error", "where"}) && v.at("ok").is_bool() && !v.at("ok").as_bool() &&
             v.at("error").is_string() && !v.at("error").as_string().empty() && v.at("where").is_string() &&
             v.at("where").as_string() == where && contains(v.at("error").as_string(), part);
    } catch (const std::exception&) {
        ok = false;
    }
    if (!ok) say("    预期 where = \"%s\"，实际应答：%.300s\n", where.c_str(), reply.text.c_str());
    return ok;
}

enum class Part { Mode, Stack, Order, Polarization, Points, Input, Options, Extra };

struct BadRequest {
    Part part;
    std::string from, to;      // 把这一段里的 from 换成 to（from 为空：整段换成 to）
    std::string where;
};

static void run_bad_requests(const Parts& base, const std::vector<BadRequest>& cases, const char* file, int line) {
    int index = 0;
    for (const BadRequest& c : cases) {
        Parts p = base;
        std::string* target = nullptr;
        switch (c.part) {
            case Part::Mode: target = &p.mode; break;
            case Part::Stack: target = &p.stack; break;
            case Part::Order: target = &p.order; break;
            case Part::Polarization: target = &p.polarization; break;
            case Part::Points: target = &p.points; break;
            case Part::Input: target = &p.input; break;
            case Part::Options: target = &p.options; break;
            case Part::Extra: target = &p.extra; break;
        }
        *target = edited(*target, c.from, c.to);
        const std::string label = "bad request #" + std::to_string(index++) + " (" + c.where + ")";
        check(rejected(p.text(), c.where), file, line, label.c_str());
    }
}

static void test_request_errors() {
    say("[请求的检查：ok = false 与 where]\n");
    const Parts base;
    {
        const rcwa::JsonReply good = rcwa::solve_json_reply(base.text());   // 没改动的请求本身是好的
        CHECK(good.ok && contains(good.text, "\"ok\": true"));
    }
    const std::string layers = "[{\"d\": 0.38, \"edges\": [0.0, 0.1722, 0.574], \"n_values\": [1.46, 1.0]}, {\"d\": 0.115, \"n\": [1.96, 0.0]}]";
    const std::vector<BadRequest> cases = {
        // mode
        {Part::Mode, "", "", "mode"},
        {Part::Mode, "\"planar\"", "3", "mode"},
        {Part::Mode, "\"planar\"", "null", "mode"},
        {Part::Mode, "\"planar\"", "\"spherical\"", "mode"},
        {Part::Mode, "\"planar\"", "\"Planar\"", "mode"},
        // 最外层不认识的键；planar 模式不用的键
        {Part::Extra, "", "\"polarisation\": \"TE\"", "polarisation"},
        {Part::Extra, "", "\"version\": 1", "version"},
        {Part::Extra, "", "\"phi_deg\": 10", "phi_deg"},
        {Part::Extra, "", "\"input\": {\"psi_deg\": 90}", "input"},
        // stack
        {Part::Stack, "", "", "stack"},
        {Part::Stack, "", "\"stack\": 5", "stack"},
        {Part::Stack, "", "\"stack\": [1, 2]", "stack"},
        {Part::Stack, "\"period\": 0.574", "\"period\": 0.574, \"nIII\": 1", "stack.nIII"},
        {Part::Stack, "\"nI\": 1.0, ", "", "stack.nI"},
        {Part::Stack, "\"nI\": 1.0", "\"nI\": \"air\"", "stack.nI"},
        {Part::Stack, "\"nI\": 1.0", "\"nI\": [1.0, 0.0, 0.0]", "stack.nI"},
        {Part::Stack, "\"nI\": 1.0", "\"nI\": [1.0]", "stack.nI"},
        {Part::Stack, "\"nI\": 1.0", "\"nI\": [1.0, \"0\"]", "stack.nI"},
        {Part::Stack, "\"nI\": 1.0", "\"nI\": [1.0, 0.1]", "stack.nI"},       // 入射介质不能有吸收
        {Part::Stack, "\"nI\": 1.0", "\"nI\": 0", "stack.nI"},
        {Part::Stack, "\"nI\": 1.0", "\"nI\": -1.0", "stack.nI"},
        {Part::Stack, "\"nII\": 1.46, ", "", "stack.nII"},
        {Part::Stack, "\"nII\": 1.46", "\"nII\": [1.5, -0.1]", "stack.nII"},   // 增益介质
        {Part::Stack, "\"nII\": 1.46", "\"nII\": 0.0", "stack.nII"},
        {Part::Stack, "\"nII\": 1.46", "\"nII\": true", "stack.nII"},
        {Part::Stack, "\"period\": 0.574, ", "", "stack.period"},
        {Part::Stack, "\"period\": 0.574", "\"period\": 0", "stack.period"},
        {Part::Stack, "\"period\": 0.574", "\"period\": -0.574", "stack.period"},
        {Part::Stack, "\"period\": 0.574", "\"period\": \"0.574\"", "stack.period"},
        {Part::Stack, "\"period\": 0.574", "\"period\": 0.6", "stack.layers[0].edges"},   // 边界不再跨一个周期
        {Part::Stack, ", \"layers\": " + layers, "", "stack.layers"},
        {Part::Stack, layers, "{}", "stack.layers"},
        {Part::Stack, layers, "[3]", "stack.layers[0]"},
        {Part::Stack, layers, "[{\"d\": 0.1, \"n\": 1.5}, null]", "stack.layers[1]"},
        // 层
        {Part::Stack, "\"d\": 0.115", "\"d\": 0.115, \"thickness\": 1", "stack.layers[1].thickness"},
        {Part::Stack, "\"d\": 0.38, ", "", "stack.layers[0].d"},
        {Part::Stack, "\"d\": 0.38", "\"d\": -0.38", "stack.layers[0].d"},
        {Part::Stack, "\"d\": 0.115", "\"d\": \"thin\"", "stack.layers[1].d"},
        {Part::Stack, "\"n\": [1.96, 0.0]", "\"n\": [1.96, 0.0], \"edges\": [0.0, 0.574]", "stack.layers[1]"},
        {Part::Stack, "\"n\": [1.96, 0.0]", "\"n\": [1.96, 0.0], \"n_values\": [1.0]", "stack.layers[1]"},
        {Part::Stack, ", \"n\": [1.96, 0.0]", "", "stack.layers[1]"},
        {Part::Stack, ", \"n_values\": [1.46, 1.0]", "", "stack.layers[0].n_values"},
        {Part::Stack, "\"edges\": [0.0, 0.1722, 0.574], ", "", "stack.layers[0].edges"},
        {Part::Stack, "\"edges\": [0.0, 0.1722, 0.574]", "\"edges\": 0.574", "stack.layers[0].edges"},
        {Part::Stack, "\"n_values\": [1.46, 1.0]", "\"n_values\": {}", "stack.layers[0].n_values"},
        {Part::Stack, "[0.0, 0.1722, 0.574]", "[0.0, \"a\", 0.574]", "stack.layers[0].edges[1]"},
        {Part::Stack, "[1.46, 1.0]", "[1.46, [1.0]]", "stack.layers[0].n_values[1]"},
        {Part::Stack, "[1.46, 1.0]", "[0.0, 1.0]", "stack.layers[0].n_values[0]"},
        {Part::Stack, "[1.46, 1.0]", "[1.46, [1.0, -0.2]]", "stack.layers[0].n_values[1]"},
        {Part::Stack, "[1.46, 1.0]", "[1.46, 1.0, 2.0]", "stack.layers[0].n_values"},      // 段数与边界数对不上
        {Part::Stack, "[1.46, 1.0]", "[]", "stack.layers[0].n_values"},
        {Part::Stack, "[0.0, 0.1722, 0.574]", "[0.0, 0.574]", "stack.layers[0].n_values"},
        {Part::Stack, "[0.0, 0.1722, 0.574]", "[0.0, 0.6, 0.574]", "stack.layers[0].edges[2]"},   // 不递增
        {Part::Stack, "[0.0, 0.1722, 0.574]", "[0.0, 0.0, 0.574]", "stack.layers[0].edges[1]"},
        {Part::Stack, "[0.0, 0.1722, 0.574]", "[0.0, 0.1722, 0.6]", "stack.layers[0].edges"},     // 首尾不是一个周期
        {Part::Stack, "\"n\": [1.96, 0.0]", "\"n\": [1.96]", "stack.layers[1].n"},
        {Part::Stack, "\"n\": [1.96, 0.0]", "\"n\": [0.0, 0.0]", "stack.layers[1].n"},
        {Part::Stack, "\"n\": [1.96, 0.0]", "\"n\": \"H\"", "stack.layers[1].n"},
        // M
        {Part::Order, "", "", "M"},
        {Part::Order, "3", "2.5", "M"},
        {Part::Order, "3", "-1", "M"},
        {Part::Order, "3", "\"3\"", "M"},
        {Part::Order, "3", "[3]", "M"},
        {Part::Order, "3", "1e12", "M"},
        {Part::Order, "3", "536870912", "M"},                                  // 比 rcwa1d.hpp 允许的上限大 1
        {Part::Order, "3", "8192", "M"},                                       // JSON 入口自己的上限：planar 是 8191
        // polarization
        {Part::Polarization, "", "", "polarization"},
        {Part::Polarization, "\"TE\"", "\"te\"", "polarization"},
        {Part::Polarization, "\"TE\"", "\"TEM\"", "polarization"},
        {Part::Polarization, "\"TE\"", "1", "polarization"},
        // 入射点
        {Part::Points, "", "", "points"},
        {Part::Points, "", "\"points\": {}", "points"},
        {Part::Points, "", "\"points\": []", "points"},
        {Part::Points, "", "\"points\": [1.053]", "points[0]"},
        {Part::Points, "{\"wavelength\": 1.06, \"theta_deg\": 60.0}", "7", "points[1]"},
        {Part::Points, "\"theta_deg\": 60.0", "\"theta_deg\": 60.0, \"lambda\": 1", "points[1].lambda"},
        {Part::Points, "\"theta_deg\": 60.0", "\"theta_deg\": 60.0, \"phi_deg\": 5", "points[1].phi_deg"},
        {Part::Points, "\"wavelength\": 1.053, ", "", "points[0].wavelength"},
        {Part::Points, "\"wavelength\": 1.06", "\"wavelength\": 0", "points[1].wavelength"},
        {Part::Points, "\"wavelength\": 1.06", "\"wavelength\": -1.06", "points[1].wavelength"},
        {Part::Points, "\"wavelength\": 1.06", "\"wavelength\": \"1.06\"", "points[1].wavelength"},
        {Part::Points, ", \"theta_deg\": 60.0", "", "points[1].theta_deg"},
        {Part::Points, "\"theta_deg\": 60.0", "\"theta_deg\": 90", "points[1].theta_deg"},
        {Part::Points, "\"theta_deg\": 60.0", "\"theta_deg\": -90.0", "points[1].theta_deg"},
        {Part::Points, "\"theta_deg\": 60.0", "\"theta_deg\": null", "points[1].theta_deg"},
        {Part::Points, "", "\"wavelength\": 1.053", "theta_deg"},              // 单点写在最外层
        {Part::Points, "", "\"theta_deg\": 10", "wavelength"},
        {Part::Points, "", "\"wavelength\": -1, \"theta_deg\": 10", "wavelength"},
        {Part::Points, "", "\"wavelength\": 1, \"theta_deg\": 95", "theta_deg"},
        {Part::Points, "", "\"wavelength\": [1], \"theta_deg\": 10", "wavelength"},
        {Part::Extra, "", "\"wavelength\": 1.0", "points"},                    // 两种写法不能都写
        {Part::Extra, "", "\"theta_deg\": 1.0", "points"},
        // options
        {Part::Options, "", "\"options\": []", "options"},
        {Part::Options, "", "\"options\": null", "options"},
        {Part::Options, "\"threads\": 1", "\"thread\": 1", "options.thread"},
        {Part::Options, "\"qr\"", "\"QR\"", "options.eig"},
        {Part::Options, "\"qr\"", "1", "options.eig"},
        {Part::Options, "\"threads\": 1", "\"linear\": \"cholesky\"", "options.linear"},
        {Part::Options, "\"threads\": 1", "\"linear\": 2", "options.linear"},
        {Part::Options, "\"threads\": 1", "\"homogeneous_fast_path\": 1", "options.homogeneous_fast_path"},
        {Part::Options, "\"threads\": 1", "\"parallel_layers\": \"yes\"", "options.parallel_layers"},
        {Part::Options, "\"threads\": 1", "\"diagnostics\": null", "options.diagnostics"},
        {Part::Options, "\"threads\": 1", "\"threads\": 0", "options.threads"},
        {Part::Options, "\"threads\": 1", "\"threads\": -2", "options.threads"},
        {Part::Options, "\"threads\": 1", "\"threads\": 1.5", "options.threads"},
        {Part::Options, "\"threads\": 1", "\"threads\": 257", "options.threads"},
        {Part::Options, "\"threads\": 1", "\"threads\": \"4\"", "options.threads"},
    };
    run_bad_requests(base, cases, __FILE__, __LINE__);

    // 请求不是合法的 JSON，或者最外层不是对象：where 是空的，error 里有位置
    CHECK(rejected("", "", "JSON"));
    CHECK(rejected("[1, 2]", ""));
    CHECK(rejected("\"planar\"", ""));
    const std::string second_line = "第 2 行";
    CHECK(rejected("{\"mode\": \"planar\",\n \"M\": }", "", second_line.c_str()));
    CHECK(rejected("{\"mode\": \"planar\", \"mode\": \"planar\"}", "", "duplicate key"));
    CHECK(rejected(std::string(100000, '['), "", "nested too deeply"));                      // 很深的嵌套不会把栈压垮
    CHECK(rejected(std::string("{\"mode\": \"planar\"") + '\0' + "}", ""));
    CHECK(rejected("{\"mode\": \"planar\", \"M\": 1e999}", "", "range of double"));
    CHECK(rejected("{}", "mode"));
    // 很长的键只显示开头一段，而且不把汉字切成两半（应答仍是合法的 UTF-8，rejected() 里解析得了）
    {
        Parts p = base;
        p.extra = "\"" + std::string(100, 'k') + "\": 1";
        CHECK(rejected(p.text(), std::string(48, 'k') + "..."));
        std::string hanzi;
        for (int k = 0; k < 40; ++k) hanzi += "\xE9\x94\xAE";                                // 40 个“键”字，120 个字节
        p.extra = "\"" + hanzi + "\": 1";
        CHECK(rejected(p.text(), hanzi.substr(0, 48) + "..."));
        p.extra = "\"x" + hanzi + "\": 1";                                                   // 第 48 个字节落在一个汉字中间
        CHECK(rejected(p.text(), "x" + hanzi.substr(0, 45) + "..."));
    }
    // 报错的说明是给人看的：挑几条看内容
    CHECK(rejected(edited(base.text(), "\"M\": 3", "\"M\": 3, \"Polarization\": \"TE\""), "Polarization", "polarization"));
    CHECK(rejected(edited(base.text(), "\"TE\"", "\"TEM\""), "polarization", "\"TE\""));
    CHECK(rejected(edited(base.text(), "[1.46, 1.0]", "[1.46, 1.0, 2.0]"), "stack.layers[0].n_values", "3"));
    CHECK(!g_bad_edit);
    CHECK(num_threads() == 1);                                 // 被拒绝的请求不会留下改过的线程数
}

// ---------------------------------------------------------------------------------------------
// planar：与直接调用逐位相同

static void test_planar_single_points() {
    say("[planar：单点，与 rcwa::solve 逐位相同]\n");
    // rcwa_json.hpp 文件头的例子，原样写成文字（M 换成 20）；另一边用同样的字面量直接调用
    {
        const std::string request = R"({
            "mode": "planar",
            "stack": {"nI": 1.0, "nII": 1.46, "period": 0.574,
                      "layers": [{"d": 0.38, "edges": [0.0, 0.1722, 0.574], "n_values": [1.46, [1.0, 0.0]]},
                                 {"d": 0.115, "n": [1.96, 0.0]}]},
            "M": 20,
            "polarization": "TE",
            "points": [{"wavelength": 1.053, "theta_deg": 66.36}],
            "options": {"eig": "qr", "linear": "lu", "homogeneous_fast_path": true, "parallel_layers": false,
                        "diagnostics": false, "threads": 1}
        })";
        const rcwa::Result direct = rcwa::solve(example_stack(), 1.053, 66.36, 20, Polarization::TE);
        const std::string text = rcwa::solve_json(request);
        const json::Value v = json::parse(text);
        CHECK(head_matches(v, "planar", 20, 1, 1));
        CHECK(planar_item_matches(v.at("results").at(0), rcwa::Point{1.053, 66.36}, direct));
        CHECK(std::abs(direct.reflectance() + direct.transmittance() - 1.0) < 1e-12);       // 这个例子无吸收：能量守恒
        CHECK(text.find('\n') == std::string::npos && text.find("null") == std::string::npos);
        // 缩进的应答：数还是那些数，只是分了行
        const rcwa::JsonReply pretty = rcwa::solve_json_reply(request, 2);
        const json::Value pv = json::parse(pretty.text);
        CHECK(pretty.ok && contains(pretty.text, "\n  \"results\": [\n    {\n      \"wavelength\": 1.053,\n      \"theta_deg\": 66.36,\n"));
        CHECK(head_matches(pv, "planar", 20, 1, 1) && planar_item_matches(pv.at("results").at(0), rcwa::Point{1.053, 66.36}, direct));
        // 同一个请求换成 TM
        const std::string tm = edited(request, "\"TE\"", "\"TM\"");
        const json::Value tv = json::parse(rcwa::solve_json(tm));
        CHECK(planar_item_matches(tv.at("results").at(0), rcwa::Point{1.053, 66.36},
                                  rcwa::solve(example_stack(), 1.053, 66.36, 20, Polarization::TM)));
        CHECK(!planar_item_matches(tv.at("results").at(0), rcwa::Point{1.053, 66.36}, direct));   // TE 与 TM 的结果确实不同
    }

    // 各种结构 × TE / TM × 请求的各种写法
    const rcwa::Stack example = example_stack(), lossy = lossy_stack(cplx(1.5, 0.03)), films = film_stack_grating();
    const rcwa::Stack pure = pure_films(), silver = silver_stack(), trapezoid = trapezoid_stack(5);
    rcwa::Stack bare;                                          // 没有任何层：只有一个界面
    bare.nI = 1.0;
    bare.nII = cplx(1.5, 0.2);
    for (Polarization pol : {Polarization::TE, Polarization::TM}) {
        CHECK(planar_agrees(example, 12, pol, {{1.053, 66.36}}));
        CHECK(planar_agrees(example, 12, pol, {{1.053, 66.36}}, {}, 0, kTopLevel | kNoOptions));
        CHECK(planar_agrees(example, 0, pol, {{1.053, 20.0}}, {}, 0, kPairs));               // M = 0：只有零级
        CHECK(planar_agrees(lossy, 12, pol, {{0.83, 27.0}}));                                // 吸收的齿、吸收膜、吸收的出射介质
        CHECK(planar_agrees(lossy, 9, pol, {{0.83, -41.5}}, {}, 0, kPairs | kIndented));     // 负的入射角
        CHECK(planar_agrees(films, 16, pol, {{1.053, 66.36}}));                              // 光栅层 + 20 层膜
        CHECK(planar_agrees(films, 10, pol, {{1.053, 66.36}}, {}, 0, kTopLevel | kPairs | kIndented));
        CHECK(planar_agrees(pure, 3, pol, {{0.633, 50.0}}));                                 // 纯膜系
        CHECK(planar_agrees(silver, 14, pol, {{0.65, 30.0}}));                               // 强吸收的光栅层和衬底
        CHECK(planar_agrees(trapezoid, 8, pol, {{1.053, 66.36}}));                           // 多个光栅层
        CHECK(planar_agrees(bare, 2, pol, {{0.8, 35.0}}));
    }

    // 各个选项原样传到求解器
    {
        rcwa::Options lr, qr_solve, no_pivot, dense, diag, layers;
        lr.eig = EigMethod::LR;
        qr_solve.linear = rcwa::LinearSolver::QRHouseholder;
        no_pivot.linear = rcwa::LinearSolver::LUNoPivot;
        dense.homogeneous_fast_path = false;
        diag.diagnostics = true;
        layers.parallel_layers = true;
        for (Polarization pol : {Polarization::TE, Polarization::TM}) {
            CHECK(planar_agrees(films, 8, pol, {{1.053, 66.36}}, lr));
            CHECK(planar_agrees(lossy, 8, pol, {{0.83, 27.0}}, qr_solve));
            CHECK(planar_agrees(lossy, 8, pol, {{0.83, 27.0}}, no_pivot));
            CHECK(planar_agrees(films, 6, pol, {{1.053, 66.36}}, dense));
            CHECK(planar_agrees(lossy, 8, pol, {{0.83, 27.0}}, diag));
            CHECK(planar_agrees(trapezoid, 8, pol, {{1.053, 66.36}}, layers, 4));
        }
        // 确认这些选项真的起了作用（否则上面的比较说明不了什么）：换一个选项，数就变了，计数也跟着变
        const auto reply_for = [&](const rcwa::Options& opt) {
            return json::parse(rcwa::solve_json(planar_request(films, 8, Polarization::TE, {{1.053, 66.36}}, opt, 0, 0)));
        };
        const rcwa::Result base = rcwa::solve(films, 1.053, 66.36, 8, Polarization::TE);
        const rcwa::Point point{1.053, 66.36};
        CHECK(planar_item_matches(reply_for(rcwa::Options()).at("results").at(0), point, base));
        CHECK(!planar_item_matches(reply_for(lr).at("results").at(0), point, base));
        CHECK(!planar_item_matches(reply_for(qr_solve).at("results").at(0), point, base));
        CHECK(!planar_item_matches(reply_for(no_pivot).at("results").at(0), point, base));
        CHECK(is_int(reply_for(rcwa::Options()).at("results").at(0).at("n_eig"), 1) &&
              is_int(reply_for(dense).at("results").at(0).at("n_eig"), 21));
        const json::Value with_diag = reply_for(diag).at("results").at(0);
        CHECK(with_diag.at("max_eig_residual").as_double() > 0.0 && with_diag.at("max_eig_residual").as_double() < 1e-12 &&
              with_diag.at("max_solve_backward_error").as_double() > 0.0);
        const json::Value without_diag = reply_for(rcwa::Options()).at("results").at(0);
        CHECK(without_diag.at("max_eig_residual").as_double() == 0.0 && json::dump(without_diag.at("max_eig_residual")) == "0.0");
    }
}

static void test_planar_sweeps_and_threads() {
    say("[planar：多点扫描与线程数，与 rcwa::sweep 逐位相同]\n");
    const rcwa::Stack films = film_stack_grating(), lossy = lossy_stack(cplx(1.5, 0.03));
    std::vector<rcwa::Point> points;
    for (int k = 0; k < 7; ++k) points.push_back({1.03 + 0.008 * k, 60.0 + 1.5 * k});
    const int M = 10;

    for (int threads : {1, 4}) {
        for (Polarization pol : {Polarization::TE, Polarization::TM}) {
            CHECK(planar_agrees(films, M, pol, points, {}, threads));
            CHECK(planar_agrees(lossy, 8, pol, points, {}, threads, kPairs));
        }
        rcwa::Options opt;                                     // 选项原样传给每个点
        opt.homogeneous_fast_path = false;
        opt.linear = rcwa::LinearSolver::QRHouseholder;
        opt.diagnostics = true;
        CHECK(planar_agrees(films, 5, Polarization::TM, points, opt, threads));
    }
    // 扫描里的每个点也与单独求解逐位相同，次序与请求相同；1 个线程与 4 个线程的应答除了用时完全相同
    {
        const json::Value one = json::parse(rcwa::solve_json(planar_request(films, M, Polarization::TE, points, {}, 1, 0)));
        const json::Value four = json::parse(rcwa::solve_json(planar_request(films, M, Polarization::TE, points, {}, 4, 0)));
        bool same = one.at("results").size() == points.size() && four.at("results").size() == points.size();
        for (std::size_t k = 0; same && k < points.size(); ++k) {
            const rcwa::Result single = rcwa::solve(films, points[k].wavelength, points[k].theta_deg, M, Polarization::TE);
            same = planar_item_matches(one.at("results").at(k), points[k], single) &&
                   planar_item_matches(four.at("results").at(k), points[k], single);
        }
        CHECK(same);
        CHECK(is_int(one.at("threads"), 1) && is_int(four.at("threads"), 4));
        CHECK(!planar_item_matches(one.at("results").at(0), points[0],
                                   rcwa::solve(films, points[1].wavelength, points[1].theta_deg, M, Polarization::TE)));
    }
    // 只有一个点的 points 数组、单线程与多线程（这时并行用在单次求解的内部）
    CHECK(planar_agrees(films, 30, Polarization::TE, {{1.053, 66.36}}, {}, 4));
    CHECK(planar_agrees(films, 30, Polarization::TM, {{1.053, 66.36}}, {}, 1));

    // 线程数：请求里写了 threads 只在这次调用期间生效，结束后改回原值；不写则沿用当前的设定
    {
        set_num_threads(3);
        const json::Value two = json::parse(rcwa::solve_json(planar_request(films, 4, Polarization::TE, points, {}, 2, 0)));
        CHECK(is_int(two.at("threads"), 2) && num_threads() == 3);
        const json::Value inherited = json::parse(rcwa::solve_json(planar_request(films, 4, Polarization::TE, points, {}, 0, 0)));
        CHECK(is_int(inherited.at("threads"), 3) && num_threads() == 3);
        const json::Value none = json::parse(rcwa::solve_json(planar_request(films, 4, Polarization::TE, points, {}, 0, kNoOptions)));
        CHECK(is_int(none.at("threads"), 3) && num_threads() == 3);
        CHECK(rejected(planar_request(films, -1, Polarization::TE, points, {}, 2, 0), "M") && num_threads() == 3);
        set_num_threads(1);
        CHECK(planar_agrees(films, 4, Polarization::TE, points, {}, 256));     // 允许的最大线程数
        CHECK(num_threads() == 1);
    }
}

static void test_planar_numerical_failure() {
    say("[planar：数值失败写成 null，ok 仍为 true]\n");
    const rcwa::Stack bad = rayleigh_stack();
    const double wavelength = bad.period;
    const int M = 8;
    const rcwa::Result direct = rcwa::solve(bad, wavelength, 0.0, M, Polarization::TE);
    CHECK(std::isnan(direct.R[0].real()) && std::isnan(direct.DEt[static_cast<std::size_t>(M)]) && std::isnan(direct.reflectance()));
    const rcwa::JsonReply reply = rcwa::solve_json_reply(planar_request(bad, M, Polarization::TE, {{wavelength, 0.0}}, {}, 0, 0));
    CHECK(reply.ok);
    const json::Value v = json::parse(reply.text);
    const json::Value item = v.at("results").at(0);
    CHECK(head_matches(v, "planar", M, 1, 1) && planar_item_matches(item, rcwa::Point{wavelength, 0.0}, direct));
    CHECK(item.at("R_total").is_null() && item.at("T_total").is_null() && item.at("DEr").at(0).is_null() &&
          item.at("R").at(3).at(0).is_null() && item.at("R").at(3).at(1).is_null() && item.at("T").at(16).at(1).is_null());
    CHECK(is_int(item.at("eig_info"), 0) && is_int(item.at("n_eig"), 1) && item.at("wavelength").is_number());
    CHECK(planar_agrees(bad, M, Polarization::TM, {{wavelength, 0.0}}));
    // 扫描里只有这一个点失败，其余的点照常
    const std::vector<rcwa::Point> points = {{0.6, 10.0}, {wavelength, 0.0}, {0.61, 0.0}};
    for (int threads : {1, 4}) CHECK(planar_agrees(bad, M, Polarization::TE, points, {}, threads));
    const json::Value swept = json::parse(rcwa::solve_json(planar_request(bad, M, Polarization::TE, points, {}, 0, 0)));
    CHECK(swept.at("ok").as_bool() && swept.at("results").at(0).at("R_total").is_number() &&
          swept.at("results").at(1).at("R_total").is_null() && swept.at("results").at(2).at("R_total").is_number());
}

// 几个线程同时调用 solve_json（请求里不写 threads，线程池不动）：各自的应答仍与直接调用逐位相同
static void test_reentrancy() {
    say("[solve_json 可以被多个线程同时调用]\n");
    struct Job {
        rcwa::Stack stack;
        int M;
        Polarization pol;
        rcwa::Point point;
    };
    const std::vector<Job> jobs = {
        {example_stack(), 9, Polarization::TE, {1.053, 66.36}},  {lossy_stack(cplx(1.5, 0.03)), 7, Polarization::TM, {0.83, 27.0}},
        {film_stack_grating(), 8, Polarization::TM, {1.06, 60.0}}, {pure_films(), 3, Polarization::TE, {0.633, 50.0}},
        {trapezoid_stack(4), 6, Polarization::TE, {1.053, 66.36}}, {rayleigh_stack(), 8, Polarization::TE, {0.5747, 0.0}},
    };
    std::vector<rcwa::Result> expect;
    for (const Job& j : jobs) expect.push_back(rcwa::solve(j.stack, j.point.wavelength, j.point.theta_deg, j.M, j.pol));
    CHECK(std::isnan(expect.back().DEr[0]));                    // 最后一个是数值失败的例子，也混在里面
    const int workers = 4;
    std::vector<std::string> replies(jobs.size() * 4);
    std::vector<std::thread> pool;
    for (int t = 0; t < workers; ++t)
        pool.emplace_back([&, t] {
            for (std::size_t i = static_cast<std::size_t>(t); i < replies.size(); i += static_cast<std::size_t>(workers)) {
                const Job& j = jobs[(i + i / jobs.size()) % jobs.size()];
                replies[i] = rcwa::solve_json(planar_request(j.stack, j.M, j.pol, {j.point}, {}, 0, i % 2 == 0 ? kNoOptions : kPairs));
            }
        });
    for (std::thread& th : pool) th.join();
    bool same = true;
    for (std::size_t i = 0; i < replies.size(); ++i) {
        const std::size_t k = (i + i / jobs.size()) % jobs.size();
        const json::Value v = json::parse(replies[i]);
        same = same && head_matches(v, "planar", jobs[k].M, 1, 1) && planar_item_matches(v.at("results").at(0), jobs[k].point, expect[k]);
    }
    CHECK(same);
    CHECK(num_threads() == 1);
}

// ---------------------------------------------------------------------------------------------
// conical

#if STABLERCWA_RCWA_JSON_CONICAL

// 入射偏振的三种给法
struct InputSpec {
    int kind = 0;                      // 0：不写 input；1：psi_deg（和 delta_deg）；2：jones
    double psi_deg = 0.0, delta_deg = 0.0;
    bool write_delta = true;           // kind = 1 时是否写出 delta_deg
    cplx Es, Ep;
};

static std::string conical_request(const rcwa::Stack& s, int M, const std::vector<rcwa::ConicalPoint>& points,
                                   const rcwa::Options& opt, int threads, int style, const InputSpec& input) {
    json::Value v = json::Value::object();
    v.set("mode", "conical");
    v.set("stack", stack_json(s, (style & kPairs) != 0));
    v.set("M", M);
    if ((style & kTopLevel) != 0 && points.size() == 1) {
        v.set("wavelength", points[0].wavelength);
        v.set("theta_deg", points[0].theta_deg);
        v.set("phi_deg", points[0].phi_deg);
    } else {
        json::Value list = json::Value::array();
        for (const rcwa::ConicalPoint& p : points) {
            json::Value item = json::Value::object();
            item.set("wavelength", p.wavelength);
            item.set("theta_deg", p.theta_deg);
            if (p.phi_deg != 0.0 || (style & kPairs) != 0) item.set("phi_deg", p.phi_deg);   // φ = 0 可以不写
            list.push_back(std::move(item));
        }
        v.set("points", std::move(list));
    }
    if (input.kind == 1) {
        json::Value in = json::Value::object();
        in.set("psi_deg", input.psi_deg);
        if (input.write_delta) in.set("delta_deg", input.delta_deg);
        v.set("input", std::move(in));
    } else if (input.kind == 2) {
        json::Value pair = json::Value::array();
        pair.push_back(index_json(input.Es, (style & kPairs) != 0));
        pair.push_back(index_json(input.Ep, true));
        json::Value in = json::Value::object();
        in.set("jones", std::move(pair));
        v.set("input", std::move(in));
    }
    if ((style & kNoOptions) == 0) v.set("options", options_json(opt, threads));
    return json::dump(v, (style & kIndented) != 0 ? 2 : -1);
}

static bool is_jones(const json::Value& v, const rcwa::Jones& j) {              // [ss, sp, ps, pp]
    return v.is_array() && v.size() == 4 && is_complex(v.at(0), j.ss) && is_complex(v.at(1), j.sp) &&
           is_complex(v.at(2), j.ps) && is_complex(v.at(3), j.pp);
}
static bool are_jones(const json::Value& v, const std::vector<rcwa::Jones>& j) {
    if (!v.is_array() || v.size() != j.size()) return false;
    for (std::size_t k = 0; k < j.size(); ++k)
        if (!is_jones(v.at(k), j[k])) return false;
    return true;
}
static bool are_channels(const json::Value& v, const std::vector<rcwa::JonesEfficiency>& e) {
    if (!v.is_array() || v.size() != e.size()) return false;
    for (std::size_t k = 0; k < e.size(); ++k) {
        const json::Value c = v.at(k);
        if (!c.is_array() || c.size() != 4 || !is_number(c.at(0), e[k].ss) || !is_number(c.at(1), e[k].sp) ||
            !is_number(c.at(2), e[k].ps) || !is_number(c.at(3), e[k].pp))
            return false;
    }
    return true;
}

// conical 应答里的一个结果与直接调用的结果逐位相同
static bool conical_item_matches(const json::Value& item, const rcwa::ConicalPoint& p, const rcwa::ConicalResult& r,
                                 const InputSpec& input) {
    const bool keys = input.kind == 0
        ? has_keys(item, {"wavelength", "theta_deg", "phi_deg", "kx", "ky", "phi_m_deg", "kzI", "kzII", "flux_r", "flux_t",
                          "Jr", "Jt", "DEr_channels", "DEt_channels", "flux_unitarity", "eig_info", "n_eig", "n_dense_solve",
                          "max_eig_residual", "max_solve_backward_error", "timing", "n_schur", "min_abs_lambda",
                          "degenerate_mode_warning"})
        : has_keys(item, {"wavelength", "theta_deg", "phi_deg", "kx", "ky", "phi_m_deg", "kzI", "kzII", "flux_r", "flux_t",
                          "Jr", "Jt", "DEr_channels", "DEt_channels", "flux_unitarity", "polarized", "eig_info", "n_eig",
                          "n_dense_solve", "max_eig_residual", "max_solve_backward_error", "timing", "n_schur", "min_abs_lambda",
                          "degenerate_mode_warning"});
    if (!keys) return false;
    // 圆锥入射特有的两项：min_abs_lambda 是 +∞（没有做本征分解的层）时应答里是 null；警告是 true / false
    const json::Value warning = item.at("degenerate_mode_warning");
    if (!is_int(item.at("n_schur"), r.n_schur) || !is_number(item.at("min_abs_lambda"), r.min_abs_lambda) || !warning.is_bool() ||
        warning.as_bool() != r.degenerate_mode_warning)
        return false;
    if (!is_number(item.at("wavelength"), p.wavelength) || !is_number(item.at("theta_deg"), p.theta_deg) ||
        !is_number(item.at("phi_deg"), p.phi_deg))
        return false;
    if (!are_numbers(item.at("kx"), r.kx) || !is_number(item.at("ky"), r.ky) || !are_numbers(item.at("phi_m_deg"), r.phi_m_deg) ||
        !are_complex(item.at("kzI"), r.kzI) || !are_complex(item.at("kzII"), r.kzII) ||
        !are_numbers(item.at("flux_r"), r.flux_r) || !are_numbers(item.at("flux_t"), r.flux_t))
        return false;
    if (!are_jones(item.at("Jr"), r.Jr) || !are_jones(item.at("Jt"), r.Jt) || !are_channels(item.at("DEr_channels"), r.DEr) ||
        !are_channels(item.at("DEt_channels"), r.DEt))
        return false;
    const std::array<cplx, 4> unitarity = r.flux_unitarity();
    const json::Value u = item.at("flux_unitarity");
    if (!u.is_array() || u.size() != 4) return false;
    for (std::size_t k = 0; k < 4; ++k)
        if (!is_complex(u.at(k), unitarity[k])) return false;
    if (input.kind != 0) {
        const rcwa::ConicalPolarized c = input.kind == 1
            ? (input.write_delta ? r.for_angles(input.psi_deg, input.delta_deg) : r.for_angles(input.psi_deg))
            : r.for_input(input.Es, input.Ep);
        const json::Value z = item.at("polarized");
        if (!has_keys(z, {"jones_in", "Rs", "Rp", "Ts", "Tp", "DEr", "DEt", "DEr_s", "DEr_p", "DEt_s", "DEt_p", "R_total", "T_total"}))
            return false;
        const json::Value in = z.at("jones_in");
        if (!in.is_array() || in.size() != 2 || !is_complex(in.at(0), c.Es) || !is_complex(in.at(1), c.Ep)) return false;
        if (!are_complex(z.at("Rs"), c.Rs) || !are_complex(z.at("Rp"), c.Rp) || !are_complex(z.at("Ts"), c.Ts) ||
            !are_complex(z.at("Tp"), c.Tp))
            return false;
        if (!are_numbers(z.at("DEr"), c.DEr) || !are_numbers(z.at("DEt"), c.DEt) || !are_numbers(z.at("DEr_s"), c.DEr_s) ||
            !are_numbers(z.at("DEr_p"), c.DEr_p) || !are_numbers(z.at("DEt_s"), c.DEt_s) || !are_numbers(z.at("DEt_p"), c.DEt_p))
            return false;
        if (!is_number(z.at("R_total"), c.reflectance()) || !is_number(z.at("T_total"), c.transmittance())) return false;
    }
    return bookkeeping_matches(item, r);
}

// 同一个问题经 solve_json，以及直接调用 solve_conical（单点）或 sweep_conical（多点）；两边的数应当逐位相同。
static bool conical_agrees(const rcwa::Stack& s, int M, const std::vector<rcwa::ConicalPoint>& points,
                           const InputSpec& input = {}, const rcwa::Options& opt = {}, int threads = 0, int style = 0) {
    const rcwa::JsonReply reply = rcwa::solve_json_reply(conical_request(s, M, points, opt, threads, style, input));
    const int before = num_threads();
    if (threads > 0) set_num_threads(threads);
    const int used = num_threads();
    std::vector<rcwa::ConicalResult> direct;
    if (points.size() == 1)
        direct.push_back(rcwa::solve_conical(s, points[0].wavelength, points[0].theta_deg, points[0].phi_deg, M, opt));
    else
        direct = rcwa::sweep_conical(s, points, M, opt);
    set_num_threads(before);
    if (!reply.ok) {
        say("    应答：%s\n", reply.text.c_str());
        return false;
    }
    const json::Value v = json::parse(reply.text);
    if (!head_matches(v, "conical", M, used, points.size())) return false;
    for (std::size_t k = 0; k < points.size(); ++k)
        if (!conical_item_matches(v.at("results").at(k), points[k], direct[k], input)) return false;
    return true;
}

static InputSpec angles(double psi_deg, double delta_deg, bool write_delta = true) {
    InputSpec in;
    in.kind = 1;
    in.psi_deg = psi_deg;
    in.delta_deg = delta_deg;
    in.write_delta = write_delta;
    return in;
}
static InputSpec jones(cplx Es, cplx Ep) {
    InputSpec in;
    in.kind = 2;
    in.Es = Es;
    in.Ep = Ep;
    return in;
}

static void test_conical() {
    say("[conical：与 rcwa::solve_conical / sweep_conical 逐位相同]\n");
    // 原样写成文字的请求；另一边用同样的字面量直接调用
    {
        const std::string request = R"({
            "mode": "conical",
            "stack": {"nI": 1.0, "nII": 1.46, "period": 0.574,
                      "layers": [{"d": 0.38, "edges": [0.0, 0.1722, 0.574], "n_values": [1.46, 1.0]},
                                 {"d": 0.115, "n": [1.96, 0.0]}]},
            "M": 12,
            "wavelength": 1.053, "theta_deg": 66.36, "phi_deg": 17.0,
            "input": {"psi_deg": 45.0, "delta_deg": 90.0}
        })";
        const rcwa::ConicalResult direct = rcwa::solve_conical(example_stack(), 1.053, 66.36, 17.0, 12);
        const std::string text = rcwa::solve_json(request);
        const json::Value v = json::parse(text);
        CHECK(head_matches(v, "conical", 12, 1, 1));
        CHECK(conical_item_matches(v.at("results").at(0), rcwa::ConicalPoint{1.053, 66.36, 17.0}, direct, angles(45.0, 90.0)));
        const rcwa::ConicalPolarized c = direct.for_angles(45.0, 90.0);
        CHECK(std::abs(c.reflectance() + c.transmittance() - 1.0) < 1e-11);                  // 无吸收：能量守恒
        // 不写 input：没有 polarized，其余的数不变
        const std::string plain = edited(request, ",\n            \"input\": {\"psi_deg\": 45.0, \"delta_deg\": 90.0}", "");
        const json::Value pv = json::parse(rcwa::solve_json(plain));
        CHECK(!pv.at("results").at(0).contains("polarized") &&
              conical_item_matches(pv.at("results").at(0), rcwa::ConicalPoint{1.053, 66.36, 17.0}, direct, InputSpec()));
        // 入射偏振换成 Jones 矢量（内部归一化）：圆偏振的两种写法给出同样的 Jones 矢量方向
        const std::string by_jones = edited(request, "{\"psi_deg\": 45.0, \"delta_deg\": 90.0}", "{\"jones\": [[0.0, 3.0], 3]}");
        const json::Value jv = json::parse(rcwa::solve_json(by_jones));
        CHECK(conical_item_matches(jv.at("results").at(0), rcwa::ConicalPoint{1.053, 66.36, 17.0}, direct, jones(cplx(0.0, 3.0), 3.0)));
        const json::Value in = jv.at("results").at(0).at("polarized").at("jones_in");
        CHECK(std::abs(in.at(0).at(1).as_double() - std::sqrt(0.5)) < 1e-15 && std::abs(in.at(1).at(0).as_double() - std::sqrt(0.5)) < 1e-15);
        // 缩进的应答
        const rcwa::JsonReply pretty = rcwa::solve_json_reply(request, 2);
        CHECK(pretty.ok && contains(pretty.text, "\n      \"polarized\": {\n        \"jones_in\": [["));
        CHECK(conical_item_matches(json::parse(pretty.text).at("results").at(0), rcwa::ConicalPoint{1.053, 66.36, 17.0}, direct,
                                   angles(45.0, 90.0)));
    }

    const rcwa::Stack example = example_stack(), lossy = lossy_stack(1.5), films = film_stack_grating();
    const rcwa::Stack pure = pure_films(), trapezoid = trapezoid_stack(3);
    // 各种结构 × 入射偏振的各种给法 × 请求的各种写法
    CHECK(conical_agrees(example, 10, {{1.053, 66.36, 17.0}}));
    CHECK(conical_agrees(example, 10, {{1.053, 66.36, 17.0}}, angles(90.0, 0.0), {}, 0, kTopLevel | kNoOptions));
    CHECK(conical_agrees(example, 10, {{1.053, 0.0, 30.0}}, angles(0.0, 0.0, false)));       // 正入射；delta_deg 不写
    CHECK(conical_agrees(example, 0, {{1.053, 40.0, 90.0}}, angles(33.0, -120.0), {}, 0, kPairs));
    CHECK(conical_agrees(lossy, 10, {{0.83, 27.0, 63.0}}, jones(cplx(0.3, -0.4), cplx(1.2, 0.5))));   // 吸收的齿和膜
    CHECK(conical_agrees(lossy, 8, {{0.83, -35.0, -40.0}}, jones(1.0, 0.0), {}, 0, kPairs | kIndented));
    CHECK(conical_agrees(films, 10, {{1.053, 66.36, 5.0}}, angles(45.0, 90.0)));             // 光栅层 + 20 层膜
    CHECK(conical_agrees(pure, 3, {{0.633, 50.0, 33.0}}, angles(60.0, 10.0)));               // 纯膜系
    CHECK(conical_agrees(trapezoid, 6, {{1.053, 66.36, 278.8733853924696}}, angles(20.0, 0.0)));
    CHECK(conical_agrees(example, 8, {{1.053, 66.36, 0.0}}, angles(90.0, 0.0)));             // φ = 0：请求里不写 phi_deg
    // 选项
    {
        rcwa::Options lr, qr_solve, dense, diag, layers;
        lr.eig = EigMethod::LR;
        qr_solve.linear = rcwa::LinearSolver::QRHouseholder;
        dense.homogeneous_fast_path = false;
        diag.diagnostics = true;
        layers.parallel_layers = true;
        CHECK(conical_agrees(films, 6, {{1.053, 66.36, 17.0}}, angles(45.0, 0.0), lr));
        CHECK(conical_agrees(lossy, 6, {{0.83, 27.0, 63.0}}, InputSpec(), qr_solve));
        CHECK(conical_agrees(example, 5, {{1.053, 66.36, 17.0}}, angles(45.0, 0.0), dense));
        CHECK(conical_agrees(lossy, 6, {{0.83, 27.0, 63.0}}, angles(45.0, 0.0), diag));
        CHECK(conical_agrees(trapezoid, 6, {{1.053, 66.36, 17.0}}, angles(45.0, 0.0), layers, 4));
        const auto item_for = [&](const rcwa::Options& opt) {
            return json::parse(rcwa::solve_json(conical_request(films, 6, {{1.053, 66.36, 17.0}}, opt, 0, 0, InputSpec()))).at("results").at(0);
        };
        const rcwa::ConicalResult base = rcwa::solve_conical(films, 1.053, 66.36, 17.0, 6);
        const rcwa::ConicalPoint point{1.053, 66.36, 17.0};
        CHECK(conical_item_matches(item_for(rcwa::Options()), point, base, InputSpec()));
        CHECK(!conical_item_matches(item_for(lr), point, base, InputSpec()) && !conical_item_matches(item_for(qr_solve), point, base, InputSpec()));
        CHECK(is_int(item_for(rcwa::Options()).at("n_eig"), base.n_eig) && !is_int(item_for(dense).at("n_eig"), base.n_eig));
        CHECK(item_for(diag).at("max_eig_residual").as_double() > 0.0 && item_for(rcwa::Options()).at("max_eig_residual").as_double() == 0.0);
    }
    // 多点扫描，1 个与 4 个线程
    {
        std::vector<rcwa::ConicalPoint> points;
        for (int k = 0; k < 6; ++k) points.push_back({1.03 + 0.008 * k, 60.0 + 1.5 * k, -20.0 + 11.0 * k});
        points.push_back({1.053, 66.36, 0.0});
        for (int threads : {1, 4}) {
            CHECK(conical_agrees(films, 6, points, InputSpec(), {}, threads));
            CHECK(conical_agrees(lossy, 6, points, angles(30.0, 45.0), {}, threads, kPairs));
            CHECK(conical_agrees(example, 6, points, jones(cplx(0.0, 1.0), 1.0), {}, threads));
        }
        const json::Value four = json::parse(rcwa::solve_json(conical_request(films, 6, points, {}, 4, 0, angles(30.0, 45.0))));
        bool same = four.at("results").size() == points.size() && is_int(four.at("threads"), 4);
        for (std::size_t k = 0; same && k < points.size(); ++k)
            same = conical_item_matches(four.at("results").at(k), points[k],
                                        rcwa::solve_conical(films, points[k].wavelength, points[k].theta_deg, points[k].phi_deg, 6),
                                        angles(30.0, 45.0));
        CHECK(same);
        CHECK(num_threads() == 1);
    }
    // 数值失败：Jones 矩阵与效率写成 null，只由入射条件决定的量照常给出
    {
        const rcwa::Stack bad = rayleigh_stack();
        const rcwa::ConicalResult direct = rcwa::solve_conical(bad, bad.period, 0.0, 0.0, 8);
        CHECK(std::isnan(direct.Jr[0].ss.real()) && std::isnan(direct.DEt[8].pp) && std::isfinite(direct.kx[0]));
        const rcwa::JsonReply reply = rcwa::solve_json_reply(conical_request(bad, 8, {{bad.period, 0.0, 0.0}}, {}, 0, 0, angles(45.0, 0.0)));
        CHECK(reply.ok);
        const json::Value item = json::parse(reply.text).at("results").at(0);
        CHECK(conical_item_matches(item, rcwa::ConicalPoint{bad.period, 0.0, 0.0}, direct, angles(45.0, 0.0)));
        CHECK(item.at("Jr").at(0).at(0).at(0).is_null() && item.at("DEt_channels").at(8).at(3).is_null() &&
              item.at("flux_unitarity").at(0).at(0).is_null() && item.at("polarized").at("R_total").is_null() &&
              item.at("kx").at(0).is_number() && item.at("kzII").at(8).at(0).is_number() && item.at("polarized").at("jones_in").at(1).at(0).is_number());
        CHECK(conical_agrees(bad, 8, {{0.6, 10.0, 20.0}, {bad.period, 0.0, 0.0}, {0.61, 0.0, 45.0}}, angles(45.0, 0.0), {}, 4));
    }
}

// 第一层的 A = Kx² − ⟦ε⟧ 的负本征值个数。A 只通过 kx_0 = nI·sin θ·cos φ 依赖入射方向：取 sin θ' = sin θ·cos φ 的
// 平面内入射，rcwa::layer_matrix 给出的 TE 层矩阵就是它。无吸收时本征值都是实数，每有一个过零，个数就变 1。
static int negative_eigenvalues(const rcwa::Stack& s, double wavelength, double theta, double phi, int M) {
    const double effective = std::asin(std::sin(theta * kPi / 180.0) * std::cos(phi * kPi / 180.0)) * 180.0 / kPi;
    const Matrix a = rcwa::layer_matrix(s.layers[0], s.period, wavelength, s.nI, effective, M, Polarization::TE);
    EigOptions values_only;
    values_only.vectors = false;
    int count = 0;
    for (const cplx& v : eig_qr(a, values_only).values) count += v.real() < 0.0 ? 1 : 0;
    return count;
}

// 圆锥入射特有的两项（含义见 rcwa1d_conical.hpp 开头关于奇点的说明）：
//   min_abs_lambda           做了本征分解的各层全部本征值的模的最小值；没有这样的层时是 +∞，应答里写成 null
//   degenerate_mode_warning  ky ≠ 0 且 η = min_abs_lambda / |ky|^1.5 < 1e-2：约化模式退化，相关层改用 Schur 传播
//   n_schur                 使用完整 Schur 传播的层数
static void test_conical_degenerate_modes() {
    say("[conical：min_abs_lambda 与 degenerate_mode_warning]\n");
    const double inf = std::numeric_limits<double>::infinity();
    const rcwa::Stack s = example_stack();                     // 一个无吸收的光栅层，下面一层膜
    const double theta = 40.0, phi = 25.0;
    const int M = 8;
    const auto direct_at = [&](double wavelength) { return rcwa::solve_conical(s, wavelength, theta, phi, M); };
    const auto reply_at = [&](double wavelength, const InputSpec& input) {
        return rcwa::solve_json(conical_request(s, M, {{wavelength, theta, phi}}, {}, 0, 0, input));
    };
    const auto finite_jones = [](const json::Value& item) {     // 应答里的 Jones 矩阵没有 null
        for (const char* key : {"Jr", "Jt"})
            for (std::size_t j = 0; j < item.at(key).size(); ++j)
                for (std::size_t c = 0; c < 4; ++c)
                    if (!item.at(key).at(j).at(c).at(0).is_number() || !item.at(key).at(j).at(c).at(1).is_number()) return false;
        return true;
    };

    // (1) 通常的点：一个正数，不置警告；这两项排在每个结果的最后
    {
        const rcwa::ConicalResult direct = direct_at(1.053);
        const std::string text = reply_at(1.053, InputSpec());
        const json::Value item = json::parse(text).at("results").at(0);
        CHECK(direct.min_abs_lambda > 1e-2 && direct.min_abs_lambda < 1e3 && !direct.degenerate_mode_warning);
        CHECK(conical_item_matches(item, rcwa::ConicalPoint{1.053, theta, phi}, direct, InputSpec()));
        CHECK(item.at("min_abs_lambda").is_number() && same_bits(item.at("min_abs_lambda").as_double(), direct.min_abs_lambda));
        CHECK(item.at("degenerate_mode_warning").is_bool() && !item.at("degenerate_mode_warning").as_bool());
        CHECK(item.key(item.size() - 2) == "min_abs_lambda" && item.key(item.size() - 1) == "degenerate_mode_warning");
        CHECK(contains(text, ", \"degenerate_mode_warning\": false}]}") && !contains(text, "null"));
    }
    // (2) 没有做本征分解的层（纯膜系）：+∞ 写成 null，不置警告；应答里只有这一处 null
    {
        const rcwa::Stack pure = pure_films();
        const rcwa::ConicalResult direct = rcwa::solve_conical(pure, 0.633, 50.0, 33.0, 3);
        const std::string text = rcwa::solve_json(conical_request(pure, 3, {{0.633, 50.0, 33.0}}, {}, 0, 0, angles(60.0, 10.0)));
        const json::Value item = json::parse(text).at("results").at(0);
        CHECK(direct.min_abs_lambda == inf && !direct.degenerate_mode_warning && direct.n_eig == 0);
        CHECK(item.at("min_abs_lambda").is_null() && item.at("degenerate_mode_warning").is_bool() &&
              !item.at("degenerate_mode_warning").as_bool());
        CHECK(contains(text, "\"min_abs_lambda\": null, \"degenerate_mode_warning\": false}]}"));
        CHECK(text.find("null") == text.rfind("null") && finite_jones(item));
        CHECK(conical_item_matches(item, rcwa::ConicalPoint{0.633, 50.0, 33.0}, direct, angles(60.0, 10.0)));
        rcwa::Options dense;                                    // 均匀层也走本征分解：这时有数
        dense.homogeneous_fast_path = false;
        const json::Value dense_item =
            json::parse(rcwa::solve_json(conical_request(pure, 3, {{0.633, 50.0, 33.0}}, dense, 0, 0, InputSpec()))).at("results").at(0);
        CHECK(dense_item.at("min_abs_lambda").is_number() && dense_item.at("min_abs_lambda").as_double() > 0.0 &&
              !dense_item.at("degenerate_mode_warning").as_bool());
    }
    // (3) 奇点：对波长二分，找到光栅层的 A 有一个本征值过零的波长（做法同 tests/test_rcwa1d_conical.cpp）
    const auto count = [&](double wavelength) { return negative_eigenvalues(s, wavelength, theta, phi, M); };
    CHECK(count(0.95) - count(1.00) == 1);
    double lo = 0.95, hi = 1.00;
    const int at_lo = count(lo);
    for (int it = 0; it < 200; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (mid == lo || mid == hi) break;
        if (count(mid) == at_lo) lo = mid;
        else hi = mid;
    }
    const double t0 = 0.5 * (lo + hi);
    // 恰好落在上面：Schur 传播恢复有限的 Jones 矩阵和效率，保留退化警告及回退层数。
    {
        const rcwa::ConicalResult direct = direct_at(t0);
        const rcwa::JsonReply reply = rcwa::solve_json_reply(conical_request(s, M, {{t0, theta, phi}}, {}, 0, 0, angles(45.0, 0.0)));
        CHECK(reply.ok);
        const json::Value item = json::parse(reply.text).at("results").at(0);
        CHECK(std::isfinite(direct.Jr[0].ss.real()) && direct.degenerate_mode_warning && direct.min_abs_lambda < 1e-12 &&
              direct.eig_info == 0 && direct.n_schur == 1);
        CHECK(conical_item_matches(item, rcwa::ConicalPoint{t0, theta, phi}, direct, angles(45.0, 0.0)));
        CHECK(finite_jones(item) && item.at("polarized").at("R_total").is_number() && is_int(item.at("n_schur"), 1) &&
              item.at("degenerate_mode_warning").as_bool() && item.at("min_abs_lambda").is_number() &&
              item.at("min_abs_lambda").as_double() < 1e-12 && is_int(item.at("eig_info"), 0));
    }
    // min|λ| 随相对偏离线性变化：量出斜率之后直接按 η 取点（η = 1 对应的相对偏离是 unit）
    const rcwa::ConicalResult above = direct_at(t0 * (1.0 + 1e-6)), below = direct_at(t0 * (1.0 - 1e-6));
    const double ky15 = std::abs(above.ky) * std::sqrt(std::abs(above.ky));
    const double slope = 0.5 * (above.min_abs_lambda + below.min_abs_lambda) / 1e-6;
    CHECK(above.ky > 0.2 && slope > 1.0 && slope < 10.0);
    const double unit = ky15 / slope;
    say("  奇点波长 %.12f，ky = %.4f：Schur 回退窗口是相对偏离 %.1e 以内\n", t0, above.ky, 1e-2 * unit);
    for (double side : {1.0, -1.0}) {
        struct Case {
            double eta;
            bool warned, failed;
        };
        for (const Case& c : {Case{3e-2, false, false}, Case{5e-3, true, false}, Case{3e-4, true, false}, Case{3e-5, true, false}}) {
            const double wavelength = t0 * (1.0 + side * c.eta * unit);
            const rcwa::ConicalResult direct = direct_at(wavelength);
            const json::Value item = json::parse(reply_at(wavelength, angles(45.0, 90.0))).at("results").at(0);
            CHECK(direct.degenerate_mode_warning == c.warned && std::isnan(direct.Jr[0].ss.real()) == c.failed &&
                  direct.n_schur == (c.warned ? 1 : 0));
            CHECK(std::abs(direct.min_abs_lambda / (c.eta * ky15) - 1.0) < 0.05);
            CHECK(conical_item_matches(item, rcwa::ConicalPoint{wavelength, theta, phi}, direct, angles(45.0, 90.0)));
            CHECK(item.at("degenerate_mode_warning").as_bool() == c.warned && finite_jones(item) == !c.failed &&
                  item.at("polarized").at("T_total").is_null() == c.failed);
        }
    }
    // 扫描：窗口外、窗口里、奇点上、纯粹的平面内入射（φ = 0，ky = 0：没有这个问题）混在一起，1 个与 4 个线程
    {
        const std::vector<rcwa::ConicalPoint> points = {{t0 * (1.0 + 0.5 * unit), theta, phi},  {t0 * (1.0 + 5e-3 * unit), theta, phi},
                                                        {t0, theta, phi},                       {t0 * (1.0 - 2e-3 * unit), theta, phi},
                                                        {1.053, 66.36, 0.0},                    {1.2, theta, phi}};
        for (int threads : {1, 4}) CHECK(conical_agrees(s, M, points, angles(30.0, 45.0), {}, threads));
        const json::Value results = json::parse(rcwa::solve_json(conical_request(s, M, points, {}, 0, 0, InputSpec()))).at("results");
        const bool expected[6] = {false, true, true, true, false, false};
        bool flags = results.size() == 6;
        for (std::size_t k = 0; flags && k < 6; ++k) flags = results.at(k).at("degenerate_mode_warning").as_bool() == expected[k];
        CHECK(flags);
        CHECK(finite_jones(results.at(2)) && finite_jones(results.at(1)) && finite_jones(results.at(3)) &&
              is_int(results.at(2).at("n_schur"), 1));
        CHECK(num_threads() == 1);
    }
}

static void test_conical_request_errors() {
    say("[conical：请求的检查]\n");
    Parts base;
    base.mode = "\"mode\": \"conical\"";
    base.polarization.clear();
    base.points = "\"points\": [{\"wavelength\": 1.053, \"theta_deg\": 66.36, \"phi_deg\": 17.0}, {\"wavelength\": 1.06, \"theta_deg\": 60.0}]";
    base.input = "\"input\": {\"psi_deg\": 45.0, \"delta_deg\": 90.0}";
    CHECK(rcwa::solve_json_reply(base.text()).ok);
    const std::vector<BadRequest> cases = {
        {Part::Extra, "", "\"polarization\": \"TE\"", "polarization"},        // conical 模式不用 polarization
        {Part::Extra, "", "\"psi_deg\": 45", "psi_deg"},
        {Part::Stack, "\"nII\": 1.46", "\"nII\": [1.5, 0.1]", "stack.nII"},   // 出射介质必须是实折射率
        {Part::Stack, "\"nI\": 1.0", "\"nI\": [1.0, 0.1]", "stack.nI"},
        {Part::Order, "3", "-3", "M"},
        {Part::Order, "3", "4096", "M"},                                       // conical 的上限是 4095
        {Part::Points, "\"phi_deg\": 17.0", "\"phi_deg\": \"17\"", "points[0].phi_deg"},
        {Part::Points, "\"theta_deg\": 60.0", "\"theta_deg\": 60.0, \"psi\": 1", "points[1].psi"},
        {Part::Points, "\"theta_deg\": 60.0", "\"theta_deg\": 90.0", "points[1].theta_deg"},
        {Part::Points, "", "\"wavelength\": 1.053, \"theta_deg\": 30, \"phi_deg\": []", "phi_deg"},
        {Part::Points, "", "\"phi_deg\": 10", "wavelength"},
        {Part::Points, "", "", "points"},
        {Part::Extra, "", "\"phi_deg\": 10", "points"},                        // points 与最外层的写法不能都写
        {Part::Input, "", "\"input\": 5", "input"},
        {Part::Input, "", "\"input\": []", "input"},
        {Part::Input, "", "\"input\": {}", "input.psi_deg"},
        {Part::Input, "", "\"input\": {\"delta_deg\": 5}", "input.psi_deg"},
        {Part::Input, "", "\"input\": {\"psi\": 45}", "input.psi"},
        {Part::Input, "45.0", "\"s\"", "input.psi_deg"},
        {Part::Input, "90.0", "[90]", "input.delta_deg"},
        {Part::Input, "", "\"input\": {\"psi_deg\": 45, \"jones\": [1, 0]}", "input"},
        {Part::Input, "", "\"input\": {\"delta_deg\": 45, \"jones\": [1, 0]}", "input"},
        {Part::Input, "", "\"input\": {\"jones\": 1}", "input.jones"},
        {Part::Input, "", "\"input\": {\"jones\": [1]}", "input.jones"},
        {Part::Input, "", "\"input\": {\"jones\": [1, 0, 0]}", "input.jones"},
        {Part::Input, "", "\"input\": {\"jones\": [0, [0, 0]]}", "input.jones"},   // 零矢量
        {Part::Input, "", "\"input\": {\"jones\": [1, \"p\"]}", "input.jones[1]"},
        {Part::Input, "", "\"input\": {\"jones\": [[1, 2, 3], 0]}", "input.jones[0]"},
        {Part::Options, "\"qr\"", "\"qz\"", "options.eig"},
    };
    run_bad_requests(base, cases, __FILE__, __LINE__);
    CHECK(!g_bad_edit && num_threads() == 1);
}

#else   // 没有编入圆锥入射

static void test_conical() {
    say("[conical：没有编入 rcwa1d_conical.hpp，请求得到明确的报错]\n");
    Parts p;
    p.mode = "\"mode\": \"conical\"";
    p.polarization.clear();
    CHECK(rejected(p.text(), "mode", "rcwa1d_conical.hpp"));
    p.points = "\"wavelength\": 1.053, \"theta_deg\": 66.36, \"phi_deg\": 17.0";
    p.input = "\"input\": {\"psi_deg\": 45.0}";
    CHECK(rejected(p.text(), "mode", "rcwa1d_conical.hpp"));
    CHECK(rcwa::solve_json_reply(Parts().text()).ok);          // planar 不受影响
}

static void test_conical_degenerate_modes() {}
static void test_conical_request_errors() {}

#endif

// 跑一组测试。中途抛出了意料之外的异常时记为一项失败，后面的各组照常进行。
static void run(void (*test)(), const char* name) {
    try {
        test();
    } catch (const std::exception& e) {
        check(false, __FILE__, __LINE__, name);
        say("    %s 中途抛出了异常：%s\n", name, e.what());
    }
}

int main() {
    run(test_json_values, "test_json_values");
    run(test_json_parse, "test_json_parse");
    run(test_json_syntax_errors, "test_json_syntax_errors");
    run(test_json_dump, "test_json_dump");
    run(test_request_errors, "test_request_errors");
    run(test_planar_single_points, "test_planar_single_points");
    run(test_planar_sweeps_and_threads, "test_planar_sweeps_and_threads");
    run(test_planar_numerical_failure, "test_planar_numerical_failure");
    run(test_reentrancy, "test_reentrancy");
    run(test_conical, "test_conical");
    run(test_conical_degenerate_modes, "test_conical_degenerate_modes");
    run(test_conical_request_errors, "test_conical_request_errors");
    set_num_threads(1);
    say("通过 %d / 共 %d\n", g_pass, g_total);
    return g_pass == g_total ? 0 : 1;
}
