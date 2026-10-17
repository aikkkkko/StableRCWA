// tests/test_rcwa2d_json.cpp —— 二维 RCWA 的 JSON 协议（rcwa_json.hpp 的 mode = "periodic2d"，rcwa_fields_json.hpp 的二维场采样）的测试
//
//   c++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -I include -I tests -I examples tests/test_rcwa2d_json.cpp -o build/test_rcwa2d_json
//   c++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -I include -I tests -I examples tests/test_rcwa2d_json.cpp -o build/test_rcwa2d_json_san
//
// 协议规格：gui/validation/field_reference_2d/PROTOCOL_2D.md；数值合同：CONTRACT_2D.md。只依赖 C++17 标准库与仓库里的头文件。内容：
//   · 请求的检查：二维请求的每一种写错（未知键、缺键、类型错、N 超限、几何非法、nI 吸收、linear、input 缺失、period / polarization 混入……）
//     都得到 ok = false 和预期的 where、说明；nII 吸收、My = 0 等合法写法；N 上限的两侧；
//   · 远场应答：键的次序、orders（harmonic_index_2d 的次序）、每个数组读回来与直接调用 solve2d 逐位相同（各选项、两种入射写法、多个点）；
//     对冻结参照（tests/rcwa2d_reference.inc）的成功算例，经 JSON 入口得到的全部数组 ≤ 冻结容差；
//     Rayleigh 失败例与 LR 数值失败例：ok 仍为 true，status、message 与直接调用相同，数组为 null；
//   · 线程数：options.threads 为 1 与 4 的应答逐位相同；parallel_layers；线程数事后复原；几个线程同时调用；
//   · 场协议：输出目录里的 NPY（魔数、头、dtype、shape、字节数）读回来与直接 field_slice / phase_slice 逐位相同，xz 与 yz 各测
//     （at 不为 0、nt 与 z 个数的各种组合、z 的顺序与重复值）；清单逐项与规格相符；掩膜规则（独立重算）；相对入射载波的相位（独立重算）；
//     失败例（Rayleigh、数值失败、非有限采样）返回 ok:false 且目录未发布、没有临时目录残留；output_dir 已存在被拒绝；request_id 规则；
//     采样的各种错误（一维的 nx、y 在二维里是未知键，反之亦然）。
// “逐位相同”的比法：把应答重新解析成数，与直接调用得到的 double 按内存里的 8 个字节比较。
#include <algorithm>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "stablercwa/json.hpp"
#include "stablercwa/rcwa_fields_json.hpp"
#include "stablercwa/rcwa_json.hpp"

using namespace stablercwa;
namespace rcwa = stablercwa::rcwa;
namespace fs = std::filesystem;

#include "rcwa2d_reference.inc"

// ---------------------------------------------------------------------------------------------
// 输出与计数（写法同 tests/test_rcwa_json.cpp：文字一律经 say() 输出，宏里不出现中文）
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
#define CHECK(cond) check(static_cast<bool>(cond), __FILE__, __LINE__, #cond)

// ---------------------------------------------------------------------------------------------
// 比较工具

static bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; }

// 应答里的一个数 v 与直接调用得到的 x 是否逐位相同；x 是 NaN / Inf 时应答里应当是 null。
static bool is_number(const json::Value& v, double x) {
    if (!std::isfinite(x)) return v.is_null();
    return v.is_number() && same_bits(v.as_double(), x);
}
static bool is_int(const json::Value& v, long long x) {
    int got = 0;
    return v.is_integer() && v.to_int(got) && static_cast<long long>(got) == x;
}
static bool is_text(const json::Value& v, const char* x) { return v.is_string() && v.as_string() == x; }
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
static bool are_flags(const json::Value& v, const std::vector<char>& flags) {
    if (!v.is_array() || v.size() != flags.size()) return false;
    for (std::size_t j = 0; j < flags.size(); ++j)
        if (!is_int(v.at(j), flags[j] != 0 ? 1 : 0)) return false;
    return true;
}
// n 个 null（复数是 [null, null]）
static bool are_nulls(const json::Value& v, std::size_t n, bool complex_values) {
    if (!v.is_array() || v.size() != n) return false;
    for (std::size_t j = 0; j < n; ++j) {
        const json::Value& item = v.at(j);
        if (!complex_values) {
            if (!item.is_null()) return false;
        } else if (!item.is_array() || item.size() != 2 || !item.at(0).is_null() || !item.at(1).is_null()) {
            return false;
        }
    }
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
static bool contains(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }
static bool finite_nonneg(const json::Value& v) { return v.is_number() && std::isfinite(v.as_double()) && v.as_double() >= 0.0; }

// 两棵 JSON 树，逐位相同（数按 8 个字节比较）；skip 里的键（"timing"、"threads"……）整项不比
static bool same_json(const json::Value& a, const json::Value& b, std::initializer_list<const char*> skip = {}) {
    if (a.type() != b.type()) return false;
    switch (a.type()) {
    case json::Type::Null: return true;
    case json::Type::Bool: return a.as_bool() == b.as_bool();
    case json::Type::Number: return same_bits(a.as_double(), b.as_double());
    case json::Type::String: return a.as_string() == b.as_string();
    case json::Type::Array:
        if (a.size() != b.size()) return false;
        for (std::size_t i = 0; i < a.size(); ++i)
            if (!same_json(a.at(i), b.at(i), skip)) return false;
        return true;
    case json::Type::Object:
        if (a.size() != b.size()) return false;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (a.key(i) != b.key(i)) return false;
            bool skipped = false;
            for (const char* name : skip) skipped = skipped || a.key(i) == name;
            if (!skipped && !same_json(a.at(i), b.at(i), skip)) return false;
        }
        return true;
    }
    return false;
}

static json::Value parse_or_null(const std::string& text) {
    try {
        return json::parse(text);
    } catch (const std::exception& e) {
        say("    应答不是合法的 JSON：%s\n", e.what());
        return json::Value();
    }
}

// 非 const 地取对象里的某一项（json::Value 的 at(键) 只有 const 版本）；没有这个键时测试直接失败
static json::Value& ref(json::Value& v, const char* key) {
    json::Value* found = v.find(key);
    if (!found) {
        check(false, __FILE__, __LINE__, "ref: 对象里没有这个键");
        say("    没有键 %s\n", key);
        static json::Value dummy;
        dummy = json::Value();
        return dummy;
    }
    return *found;
}
// 对象去掉一项后的副本
static json::Value dropped(const json::Value& object, const char* key) {
    json::Value out = json::Value::object();
    for (std::size_t i = 0; i < object.size(); ++i)
        if (object.key(i) != key) out.set(object.key(i), object.at(i));
    return out;
}

// ---------------------------------------------------------------------------------------------
// 请求的构造：从 Stack2D 等得到 JSON 树（数用 json::dump 的最短写法，读回来是同一个 double）

static json::Value index_value(cplx n) {
    json::Value v = json::Value::array();
    v.push_back(n.real());
    v.push_back(n.imag());
    return v;
}
static json::Value numbers_value(const std::vector<double>& x) {
    json::Value v = json::Value::array();
    for (double e : x) v.push_back(e);
    return v;
}
static json::Value stack_value(const rcwa::Stack2D& s) {
    json::Value layers = json::Value::array();
    for (const rcwa::Layer2D& layer : s.layers) {
        json::Value item = json::Value::object();
        item.set("d", layer.d);
        if (layer.homogeneous) {
            item.set("n", index_value(layer.n));
        } else {
            item.set("y_edges", numbers_value(layer.y_edges));
            json::Value rows = json::Value::array();
            for (const rcwa::XProfile2D& p : layer.x_profiles) {
                json::Value row = json::Value::object();
                row.set("edges", numbers_value(p.edges));
                json::Value values = json::Value::array();
                for (const cplx& n : p.n_values) values.push_back(index_value(n));
                row.set("n_values", std::move(values));
                rows.push_back(std::move(row));
            }
            item.set("x_profiles", std::move(rows));
        }
        layers.push_back(std::move(item));
    }
    json::Value v = json::Value::object();
    v.set("nI", s.nI);
    v.set("nII", index_value(s.nII));
    v.set("Lx", s.Lx);
    v.set("Ly", s.Ly);
    v.set("layers", std::move(layers));
    return v;
}

// 一个二维的求解问题：JSON 的写法与直接调用的参数放在一起，保证两边是同一个问题
struct Case {
    rcwa::Stack2D stack;
    double wavelength = 1.053, theta_deg = 42.0, phi_deg = -25.0;
    int M = 2, My = 1;
    bool jones_form = false;                 // 入射写法：false 用 psi/delta，true 用 jones
    double psi_deg = 90.0, delta_deg = 0.0;
    rcwa::Jones2D jones{cplx{1.0, 0.0}, cplx{0.0, 0.0}};   // jones_form = true 时的入射 Jones 矢量
    rcwa::Options2D options;
    int threads = 0;                         // > 0 时请求里写 threads
    bool write_options = true;               // false：请求里不写 options
    bool top_level_point = false;            // true：wavelength、theta_deg、phi_deg 写在最外层（只一个点时）
    std::vector<std::array<double, 3>> extra_points;   // 另外的点（波长、θ、φ）；非空时用 points 数组
    rcwa::Jones2D effective_jones() const {
        return jones_form ? jones : rcwa::jones_from_angles_2d(psi_deg, delta_deg);
    }
    // 请求里不写 options 时求解器取默认值
    rcwa::Options2D effective_options() const { return write_options ? options : rcwa::Options2D{}; }
};

static json::Value options_value(const Case& c) {
    json::Value v = json::Value::object();
    v.set("eig", c.options.eig == EigMethod::LR ? "lr" : "qr");
    v.set("factorization", c.options.factorization == rcwa::Factorization2D::Li ? "li" : "laurent");
    v.set("homogeneous_fast_path", c.options.homogeneous_fast_path);
    v.set("parallel_layers", c.options.parallel_layers);
    v.set("diagnostics", c.options.diagnostics);
    if (c.threads > 0) v.set("threads", c.threads);
    return v;
}
static json::Value input_value(const Case& c) {
    json::Value v = json::Value::object();
    if (c.jones_form) {
        json::Value j = json::Value::array();
        j.push_back(index_value(c.jones[0]));
        j.push_back(index_value(c.jones[1]));
        v.set("jones", std::move(j));
    } else {
        v.set("psi_deg", c.psi_deg);
        v.set("delta_deg", c.delta_deg);
    }
    return v;
}
static json::Value point_value(double wavelength, double theta_deg, double phi_deg) {
    json::Value p = json::Value::object();
    p.set("wavelength", wavelength);
    p.set("theta_deg", theta_deg);
    p.set("phi_deg", phi_deg);
    return p;
}
static json::Value solve_value(const Case& c) {
    json::Value v = json::Value::object();
    v.set("mode", "periodic2d");
    v.set("stack", stack_value(c.stack));
    v.set("M", c.M);
    v.set("My", c.My);
    if (c.top_level_point && c.extra_points.empty()) {
        v.set("wavelength", c.wavelength);
        v.set("theta_deg", c.theta_deg);
        v.set("phi_deg", c.phi_deg);
    } else {
        json::Value points = json::Value::array();
        points.push_back(point_value(c.wavelength, c.theta_deg, c.phi_deg));
        for (const auto& p : c.extra_points) points.push_back(point_value(p[0], p[1], p[2]));
        v.set("points", std::move(points));
    }
    v.set("input", input_value(c));
    if (c.write_options) v.set("options", options_value(c));
    return v;
}
static std::string far_request(const Case& c, int indent = -1) { return json::dump(solve_value(c), indent); }

static rcwa::Result2D direct_solve(const Case& c, double wavelength, double theta_deg, double phi_deg) {
    return rcwa::solve2d(c.stack, wavelength, theta_deg, phi_deg, c.M, c.My, c.effective_jones(), c.effective_options());
}
static rcwa::Result2D direct_solve(const Case& c) { return direct_solve(c, c.wavelength, c.theta_deg, c.phi_deg); }

// ---------------------------------------------------------------------------------------------
// 几何（算例）

static rcwa::XProfile2D profile(std::vector<double> edges, std::vector<cplx> values) {
    return rcwa::XProfile2D{std::move(edges), std::move(values)};
}

// 请求文字里的基准结构，与 base_stack() 是同一个（用于直接改文字的错误请求）：Lx = 0.8、Ly = 0.6，两个 y 分段的结构层 + 一层膜
static const char* kBaseText = R"({"mode": "periodic2d",
 "stack": {"nI": 1.0, "nII": 1.46, "Lx": 0.8, "Ly": 0.6,
           "layers": [{"d": 0.3, "y_edges": [0.0, 0.25, 0.6],
                       "x_profiles": [{"edges": [0.0, 0.3, 0.8], "n_values": [1.0, 1.8]},
                                      {"edges": [0.0, 0.8], "n_values": [1.5]}]},
                      {"d": 0.115, "n": [1.96, 0.0]}]},
 "M": 2, "My": 1,
 "points": [{"wavelength": 1.053, "theta_deg": 42.0, "phi_deg": -25.0}],
 "input": {"psi_deg": 90.0, "delta_deg": 0.0},
 "options": {"eig": "qr", "factorization": "li", "homogeneous_fast_path": true,
             "parallel_layers": false, "diagnostics": false, "threads": 1}})";

static rcwa::Stack2D base_stack() {
    rcwa::Stack2D s;
    s.nI = 1.0;
    s.nII = 1.46;
    s.Lx = 0.8;
    s.Ly = 0.6;
    s.layers.push_back(rcwa::Layer2D::patterned(0.3, {0.0, 0.25, 0.6},
                                                {profile({0.0, 0.3, 0.8}, {1.0, 1.8}), profile({0.0, 0.8}, {1.5})}));
    s.layers.push_back(rcwa::Layer2D::uniform(0.115, cplx{1.96, 0.0}));
    return s;
}
static Case base_case() {
    Case c;
    c.stack = base_stack();
    return c;
}

// 吸收的结构层 + 零厚度层 + 吸收的出射介质、Lx ≠ Ly
static rcwa::Stack2D mixed_stack() {
    rcwa::Stack2D s;
    s.nI = 1.0;
    s.nII = cplx{1.62, 0.05};
    s.Lx = 0.7;
    s.Ly = 0.9;
    s.layers.push_back(rcwa::Layer2D::uniform(0.0, cplx{2.0, 0.0}));
    s.layers.push_back(rcwa::Layer2D::patterned(
        0.25, {-0.1, 0.35, 0.8},
        {profile({0.05, 0.4, 0.75}, {cplx{1.0, 0.0}, cplx{3.2, 0.12}}), profile({0.05, 0.3, 0.75}, {cplx{2.2, 0.0}, cplx{1.0, 0.0}})}));
    s.layers.push_back(rcwa::Layer2D::uniform(0.0, cplx{1.8, 0.0}));
    s.layers.push_back(rcwa::Layer2D::uniform(0.12, cplx{1.7, 0.03}));
    return s;
}

// 一半的 nI = 1.5 → nII = 1.0 的单界面（全内反射：透射区只有倏逝波）
static rcwa::Stack2D tir_stack() {
    rcwa::Stack2D s;
    s.nI = 1.5;
    s.nII = 1.0;
    s.Lx = 0.8;
    s.Ly = 0.8;
    return s;
}

// ---------------------------------------------------------------------------------------------
// 请求文字的编辑（g_bad_edit：要换的那段文字必须在基准文字里恰好出现一次，免得编辑悄悄没生效）

static bool g_bad_edit = false;
static std::string edit(const std::string& text, const std::string& from, const std::string& to) {
    const std::size_t first = text.find(from);
    if (first == std::string::npos || text.find(from, first + 1) != std::string::npos) {
        g_bad_edit = true;
        say("  编辑失败：%s 在基准请求里出现 %s\n", from.c_str(), first == std::string::npos ? "0 次" : "多次");
        return text;
    }
    return text.substr(0, first) + to + text.substr(first + from.size());
}

// 请求有错：ok = false、where 恰好是 where、error 含 part（part 为空时不查）；应答只有 ok、error、where 三个键。
static bool rejected(const std::string& request, const char* where, const char* part = "") {
    const rcwa::JsonReply reply = rcwa::solve_json_reply(request);
    const json::Value v = parse_or_null(reply.text);
    const bool shape = !reply.ok && has_keys(v, {"ok", "error", "where"}) && v.at("ok").is_bool() && !v.at("ok").as_bool() &&
                       v.at("error").is_string() && v.at("where").is_string();
    const bool good = shape && v.at("where").as_string() == where && contains(v.at("error").as_string(), part);
    if (!good) {
        say("    请求被拒绝的方式不符（期望 where = %s，说明含“%s”），应答：%.300s\n", where, part, reply.text.c_str());
    }
    return good;
}
static bool accepted(const std::string& request) {
    const rcwa::JsonReply reply = rcwa::solve_json_reply(request);
    if (!reply.ok) say("    本应接受的请求被拒绝：%.300s\n", reply.text.c_str());
    return reply.ok;
}

// ---------------------------------------------------------------------------------------------
// 远场应答

static const char* const kResultKeys[] = {"wavelength", "theta_deg", "phi_deg", "status", "message", "Kx", "Ky", "kzI", "kzII", "Rs", "Rp", "Ts", "Tp",
                                          "Ex_r", "Ey_r", "Ez_r", "Ex_t", "Ey_t", "Ez_t", "DEr", "DEt", "prop_r", "prop_t", "eig_info", "n_eig",
                                          "n_dense_solve", "max_eig_residual", "max_solve_backward_error", "timing"};

static bool result_keys_in_order(const json::Value& r) {
    constexpr std::size_t n = sizeof kResultKeys / sizeof kResultKeys[0];
    if (!r.is_object() || r.size() != n) return false;
    for (std::size_t i = 0; i < n; ++i)
        if (r.key(i) != kResultKeys[i]) return false;
    return true;
}

// 应答头部：ok、version、mode、M、My、orders（[m, n] 对，次序 = harmonic_index_2d）、threads、results
static bool head_matches(const json::Value& v, int M, int My, int threads, std::size_t npoints, bool with_field_data = false) {
    const bool keys = with_field_data
                          ? has_keys(v, {"ok", "version", "mode", "M", "My", "orders", "threads", "results", "protocol", "field_data"})
                          : has_keys(v, {"ok", "version", "mode", "M", "My", "orders", "threads", "results"});
    if (!keys) return false;
    if (!v.at("ok").is_bool() || !v.at("ok").as_bool() || !is_int(v.at("version"), 1) || !is_text(v.at("mode"), "periodic2d")) return false;
    if (!is_int(v.at("M"), M) || !is_int(v.at("My"), My) || !is_int(v.at("threads"), threads)) return false;
    const json::Value& orders = v.at("orders");
    const std::size_t N = static_cast<std::size_t>(2 * M + 1) * static_cast<std::size_t>(2 * My + 1);
    if (!orders.is_array() || orders.size() != N) return false;
    for (int m = -M; m <= M; ++m)
        for (int n = -My; n <= My; ++n) {
            const json::Value& pair = orders.at(static_cast<std::size_t>(rcwa::harmonic_index_2d(m, n, M, My)));
            if (!pair.is_array() || pair.size() != 2 || !is_int(pair.at(0), m) || !is_int(pair.at(1), n)) return false;
        }
    return v.at("results").is_array() && v.at("results").size() == npoints;
}

// 应答里的一个结果与直接调用的 Result2D 逐位相同（成功），或失败的样子（status、message 与直接调用相同，数组为 null，prop_* 照常）
static bool result_matches(const json::Value& r, double wavelength, double theta_deg, double phi_deg, const rcwa::Result2D& d,
                           bool diagnostics_on = false) {
    if (!result_keys_in_order(r)) return false;
    if (!is_number(r.at("wavelength"), wavelength) || !is_number(r.at("theta_deg"), theta_deg) || !is_number(r.at("phi_deg"), phi_deg)) return false;
    const char* status = d.status == rcwa::Status2D::Ok ? "ok" : d.status == rcwa::Status2D::RayleighAnomaly ? "rayleigh_anomaly" : "numerical_failure";
    if (!is_text(r.at("status"), status) || !r.at("message").is_string() || r.at("message").as_string() != d.message) return false;
    const std::size_t N = static_cast<std::size_t>(d.N);
    bool good = true;
    if (d.status == rcwa::Status2D::Ok) {
        good = good && are_numbers(r.at("Kx"), d.Kx) && are_numbers(r.at("Ky"), d.Ky);
        good = good && are_complex(r.at("kzI"), d.kzI) && are_complex(r.at("kzII"), d.kzII);
        good = good && are_complex(r.at("Rs"), d.Rs) && are_complex(r.at("Rp"), d.Rp) && are_complex(r.at("Ts"), d.Ts) && are_complex(r.at("Tp"), d.Tp);
        good = good && are_complex(r.at("Ex_r"), d.Ex_r) && are_complex(r.at("Ey_r"), d.Ey_r) && are_complex(r.at("Ez_r"), d.Ez_r);
        good = good && are_complex(r.at("Ex_t"), d.Ex_t) && are_complex(r.at("Ey_t"), d.Ey_t) && are_complex(r.at("Ez_t"), d.Ez_t);
        good = good && are_numbers(r.at("DEr"), d.DEr) && are_numbers(r.at("DEt"), d.DEt);
    } else {
        good = good && are_nulls(r.at("Kx"), N, false) && are_nulls(r.at("Ky"), N, false);
        good = good && are_nulls(r.at("kzI"), N, true) && are_nulls(r.at("kzII"), N, true);
        for (const char* key : {"Rs", "Rp", "Ts", "Tp", "Ex_r", "Ey_r", "Ez_r", "Ex_t", "Ey_t", "Ez_t"}) good = good && are_nulls(r.at(key), N, true);
        good = good && are_nulls(r.at("DEr"), N, false) && are_nulls(r.at("DEt"), N, false);
    }
    good = good && are_flags(r.at("prop_r"), d.prop_r) && are_flags(r.at("prop_t"), d.prop_t);
    good = good && is_int(r.at("eig_info"), d.eig_info) && is_int(r.at("n_eig"), d.n_eig) && is_int(r.at("n_dense_solve"), d.n_dense_solve);
    good = good && is_number(r.at("max_eig_residual"), d.max_eig_residual) && is_number(r.at("max_solve_backward_error"), 0.0);
    if (!diagnostics_on) good = good && same_bits(r.at("max_eig_residual").as_double(), 0.0);
    const json::Value& timing = r.at("timing");
    good = good && has_keys(timing, {"total", "operators", "eig", "linear", "product"});
    if (good)
        for (std::size_t i = 0; i < timing.size(); ++i) good = good && finite_nonneg(timing.at(i));
    return good;
}

// 经 JSON 入口求解（可以有多个点），与直接调用逐位比较
static bool far_agrees(const Case& c, bool diagnostics_on = false, int expected_threads = 1) {
    const rcwa::JsonReply reply = rcwa::solve_json_reply(far_request(c));
    if (!reply.ok) {
        say("    应答：%.300s\n", reply.text.c_str());
        return false;
    }
    const json::Value v = parse_or_null(reply.text);
    const std::size_t npoints = 1 + c.extra_points.size();
    if (!head_matches(v, c.M, c.My, expected_threads, npoints)) return false;
    if (!result_matches(v.at("results").at(0), c.wavelength, c.theta_deg, c.phi_deg, direct_solve(c), diagnostics_on)) return false;
    for (std::size_t k = 0; k < c.extra_points.size(); ++k) {
        const auto& p = c.extra_points[k];
        if (!result_matches(v.at("results").at(k + 1), p[0], p[1], p[2], direct_solve(c, p[0], p[1], p[2]), diagnostics_on)) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------
// 请求的检查

static void test_request_errors() {
    say("[二维请求的检查：每一种写错都得到 ok = false 与预期的 where]\n");
    const std::string base = kBaseText;
    CHECK(accepted(base));
    CHECK(!g_bad_edit);

    // 最外层、JSON 本身
    CHECK(rejected("[]", "", "对象"));
    CHECK(rejected("5", ""));
    CHECK(rejected("{", "", "第 1 行"));
    CHECK(rejected("", ""));
    CHECK(rejected("{}", "mode"));
    CHECK(rejected(edit(base, "\"mode\": \"periodic2d\"", "\"mode\": 3"), "mode"));
    CHECK(rejected(edit(base, "\"mode\": \"periodic2d\"", "\"mode\": \"periodic3d\""), "mode", "periodic2d"));
    CHECK(rejected(edit(base, "\"mode\": \"periodic2d\"", "\"mode\": \"Periodic2D\""), "mode", "periodic2d"));

    // 未知键、一维的写法混进来
    CHECK(rejected(edit(base, "\"M\": 2,", "\"M\": 2, \"foo\": 1,"), "foo", "不认识的键"));
    CHECK(rejected(edit(base, "\"M\": 2,", "\"M\": 2, \"nx\": 4,"), "nx", "不认识的键"));
    CHECK(rejected(edit(base, "\"M\": 2,", "\"M\": 2, \"period\": 0.8,"), "period", "Lx"));
    CHECK(rejected(edit(base, "\"M\": 2,", "\"M\": 2, \"polarization\": \"TE\","), "polarization", "input"));
    CHECK(rejected(edit(base, "\"Lx\": 0.8,", "\"Lx\": 0.8, \"period\": 0.8,"), "stack.period", "Lx"));
    CHECK(rejected(edit(base, "\"Lx\": 0.8,", "\"Lx\": 0.8, \"Lz\": 0.8,"), "stack.Lz", "不认识的键"));
    CHECK(rejected(edit(base, "\"eig\": \"qr\",", "\"eig\": \"qr\", \"linear\": \"lu\","), "options.linear", "二维不适用"));
    CHECK(rejected(edit(base, "\"eig\": \"qr\",", "\"eig\": \"qr\", \"foo\": 1,"), "options.foo", "不认识的键"));
    CHECK(rejected(edit(base, "\"theta_deg\": 42.0, \"phi_deg\"", "\"theta_deg\": 42.0, \"foo\": 1, \"phi_deg\""), "points[0].foo", "不认识的键"));
    CHECK(rejected(edit(base, "\"psi_deg\": 90.0,", "\"psi_deg\": 90.0, \"foo\": 1,"), "input.foo", "不认识的键"));
    CHECK(rejected(edit(base, "\"d\": 0.115,", "\"d\": 0.115, \"foo\": 1,"), "stack.layers[1].foo", "不认识的键"));
    CHECK(rejected(edit(base, "{\"edges\": [0.0, 0.8], \"n_values\": [1.5]}", "{\"edges\": [0.0, 0.8], \"n_values\": [1.5], \"foo\": 2}"),
                   "stack.layers[0].x_profiles[1].foo", "不认识的键"));
    // 把一维写法里的 periodic2d 之外的键写在二维里：y（一维场采样的键）也是未知键
    CHECK(rejected(edit(base, "\"M\": 2,", "\"M\": 2, \"y\": 0,"), "y", "不认识的键"));

    // stack
    CHECK(rejected(edit(base, "\"stack\": {\"nI\": 1.0, \"nII\": 1.46, \"Lx\": 0.8, \"Ly\": 0.6,", "\"stack\": {\"nII\": 1.46, \"Lx\": 0.8, \"Ly\": 0.6,"), "stack.nI", "缺少"));
    CHECK(rejected(edit(base, "\"nII\": 1.46, ", ""), "stack.nII", "缺少"));
    CHECK(rejected(edit(base, "\"Lx\": 0.8,", ""), "stack.Lx", "缺少"));
    CHECK(rejected(edit(base, "\"Ly\": 0.6,", ""), "stack.Ly", "缺少"));
    CHECK(rejected(edit(base, "\"nI\": 1.0,", "\"nI\": 1.0, \"nI\": 1.0,"), "", "不是合法的 JSON"));           // 重复的键是语法错误
    {
        json::Value tree = solve_value(base_case());
        CHECK(accepted(json::dump(tree)));
        json::Value t = tree;
        t.set("stack", 5);
        CHECK(rejected(json::dump(t), "stack", "对象"));
        t = tree;
        ref(t, "stack").set("layers", 3);
        CHECK(rejected(json::dump(t), "stack.layers", "数组"));
        t = tree;
        t.set("stack", dropped(*tree.find("stack"), "layers"));
        CHECK(rejected(json::dump(t), "stack.layers", "缺少"));
        // 结构层只有 y_edges 或只有 x_profiles
        t = tree;
        ref(ref(t, "stack"), "layers").at(0) = dropped(tree.find("stack")->find("layers")->at(0), "x_profiles");
        CHECK(rejected(json::dump(t), "stack.layers[0].x_profiles", "x_profiles"));
        t = tree;
        ref(ref(t, "stack"), "layers").at(0) = dropped(tree.find("stack")->find("layers")->at(0), "y_edges");
        CHECK(rejected(json::dump(t), "stack.layers[0].y_edges", "y_edges"));
        t = tree;
        ref(ref(t, "stack"), "layers").at(0) = dropped(dropped(tree.find("stack")->find("layers")->at(0), "y_edges"), "x_profiles");
        CHECK(rejected(json::dump(t), "stack.layers[0]", "缺少折射率"));
        // 层不是对象、x_profiles 的第一项不是对象
        t = tree;
        ref(ref(t, "stack"), "layers").at(0) = json::Value(7);
        CHECK(rejected(json::dump(t), "stack.layers[0]", "对象"));
        t = tree;
        ref(ref(ref(ref(t, "stack"), "layers").at(0), "x_profiles").at(0), "edges") = json::Value(0.8);
        CHECK(rejected(json::dump(t), "stack.layers[0].x_profiles[0].edges", "数组"));
        t = tree;
        ref(ref(ref(t, "stack"), "layers").at(0), "x_profiles").at(0) = json::Value(3);
        CHECK(rejected(json::dump(t), "stack.layers[0].x_profiles[0]", "对象"));
        t = tree;
        ref(ref(t, "stack"), "layers").at(0).set("x_profiles", json::Value::array());
        CHECK(rejected(json::dump(t), "stack.layers[0].x_profiles", "段数"));
    }
    CHECK(rejected(edit(base, "\"nI\": 1.0", "\"nI\": [1.0, 0.1]"), "stack.nI", "无吸收"));     // 入射介质有吸收：非法
    CHECK(rejected(edit(base, "\"nI\": 1.0", "\"nI\": -1.0"), "stack.nI"));
    CHECK(rejected(edit(base, "\"nI\": 1.0", "\"nI\": 0"), "stack.nI"));
    CHECK(rejected(edit(base, "\"nI\": 1.0", "\"nI\": \"x\""), "stack.nI", "实数"));
    CHECK(accepted(edit(base, "\"nI\": 1.0", "\"nI\": [1.0, 0.0]")));                         // 虚部为 0 的写法合法
    CHECK(accepted(edit(base, "\"nII\": 1.46", "\"nII\": [1.46, 0.2]")));                      // 吸收出射介质合法
    CHECK(rejected(edit(base, "\"nII\": 1.46", "\"nII\": [1.46, -0.2]"), "stack.nII"));
    CHECK(rejected(edit(base, "\"nII\": 1.46", "\"nII\": [-1.46, 0.2]"), "stack.nII"));
    CHECK(rejected(edit(base, "\"nII\": 1.46", "\"nII\": 0"), "stack.nII"));
    CHECK(rejected(edit(base, "\"nII\": 1.46", "\"nII\": [1, 2, 3]"), "stack.nII", "[实部, 虚部]"));
    CHECK(rejected(edit(base, "\"nII\": 1.46", "\"nII\": null"), "stack.nII"));
    CHECK(rejected(edit(base, "\"Lx\": 0.8", "\"Lx\": 0"), "stack.Lx", "正数"));
    CHECK(rejected(edit(base, "\"Lx\": 0.8", "\"Lx\": -0.8"), "stack.Lx", "正数"));
    CHECK(rejected(edit(base, "\"Lx\": 0.8", "\"Lx\": \"0.8\""), "stack.Lx", "数"));
    CHECK(rejected(edit(base, "\"Ly\": 0.6", "\"Ly\": 0"), "stack.Ly", "正数"));
    CHECK(rejected(edit(base, "\"Ly\": 0.6", "\"Ly\": [0.6]"), "stack.Ly"));

    // 层
    CHECK(rejected(edit(base, "{\"d\": 0.115, \"n\": [1.96, 0.0]}", "7"), "stack.layers[1]", "对象"));
    CHECK(rejected(edit(base, "{\"d\": 0.115, \"n\": [1.96, 0.0]}", "{\"n\": [1.96, 0.0]}"), "stack.layers[1].d", "缺少"));
    CHECK(rejected(edit(base, "\"d\": 0.115", "\"d\": -0.1"), "stack.layers[1].d", "负"));
    CHECK(rejected(edit(base, "\"d\": 0.115", "\"d\": \"x\""), "stack.layers[1].d"));
    CHECK(accepted(edit(base, "\"d\": 0.115", "\"d\": 0")));                                   // 零厚度层合法
    CHECK(rejected(edit(base, "{\"d\": 0.115, \"n\": [1.96, 0.0]}", "{\"d\": 0.115}"), "stack.layers[1]", "缺少折射率"));
    CHECK(rejected(edit(base, "\"n\": [1.96, 0.0]", "\"n\": 0"), "stack.layers[1].n"));
    CHECK(rejected(edit(base, "\"n\": [1.96, 0.0]", "\"n\": [1.96, -0.5]"), "stack.layers[1].n"));
    CHECK(rejected(edit(base, "\"n\": [1.96, 0.0]", "\"n\": [-1.96, 0.0]"), "stack.layers[1].n"));
    CHECK(rejected(edit(base, "\"n\": [1.96, 0.0]", "\"n\": \"glass\""), "stack.layers[1].n"));
    CHECK(rejected(edit(base, "\"n\": [1.96, 0.0]", "\"n\": [1.96, 0.0], \"y_edges\": [0.0, 0.6]"), "stack.layers[1]", "不能混"));
    CHECK(rejected(edit(base, "\"n\": [1.96, 0.0]", "\"n\": [1.96, 0.0], \"x_profiles\": []"), "stack.layers[1]", "不能混"));
    CHECK(rejected(edit(base, "\"d\": 0.3, \"y_edges\": [0.0, 0.25, 0.6],", "\"d\": 0.3, \"n\": 1.5, \"y_edges\": [0.0, 0.25, 0.6],"), "stack.layers[0]", "不能混"));
    CHECK(rejected(edit(base, "\"d\": 0.3, \"y_edges\": [0.0, 0.25, 0.6],", "\"d\": 0.3,"), "stack.layers[0].y_edges", "y_edges"));
    CHECK(rejected(edit(base, "\"y_edges\": [0.0, 0.25, 0.6]", "\"y_edges\": 0.6"), "stack.layers[0].y_edges", "数组"));
    CHECK(rejected(edit(base, "\"y_edges\": [0.0, 0.25, 0.6]", "\"y_edges\": [0.0, \"a\", 0.6]"), "stack.layers[0].y_edges[1]", "数"));
    CHECK(rejected(edit(base, "\"y_edges\": [0.0, 0.25, 0.6]", "\"y_edges\": [0.0, 0.7, 0.6]"), "stack.layers[0].y_edges[2]", "递增"));
    CHECK(rejected(edit(base, "\"y_edges\": [0.0, 0.25, 0.6]", "\"y_edges\": [0.0, 0.25, 0.25]"), "stack.layers[0].y_edges[2]", "递增"));
    CHECK(rejected(edit(base, "\"y_edges\": [0.0, 0.25, 0.6]", "\"y_edges\": [0.0, 0.25, 0.5]"), "stack.layers[0].y_edges", "一个周期"));
    CHECK(rejected(edit(base, "\"y_edges\": [0.0, 0.25, 0.6]", "\"y_edges\": [0.0, 0.25, 0.8]"), "stack.layers[0].y_edges", "Ly"));
    CHECK(rejected(edit(base, "\"y_edges\": [0.0, 0.25, 0.6]", "\"y_edges\": [0.0, 0.6]"), "stack.layers[0].x_profiles", "段数"));
    CHECK(rejected(edit(base, "\"y_edges\": [0.0, 0.25, 0.6]", "\"y_edges\": [0.0, 0.2, 0.4, 0.6]"), "stack.layers[0].x_profiles", "段数"));
    CHECK(accepted(edit(base, "\"y_edges\": [0.0, 0.25, 0.6]", "\"y_edges\": [-0.3, 0.2, 0.3]")));    // 起点可以不是 0
    CHECK(rejected(edit(base, "{\"edges\": [0.0, 0.8], \"n_values\": [1.5]}", "5"), "stack.layers[0].x_profiles[1]", "对象"));
    CHECK(rejected(edit(base, "{\"edges\": [0.0, 0.8], \"n_values\": [1.5]}", "{\"n_values\": [1.5]}"), "stack.layers[0].x_profiles[1].edges", "缺少"));
    CHECK(rejected(edit(base, "{\"edges\": [0.0, 0.8], \"n_values\": [1.5]}", "{\"edges\": [0.0, 0.8]}"), "stack.layers[0].x_profiles[1].n_values", "缺少"));
    CHECK(rejected(edit(base, "\"edges\": [0.0, 0.3, 0.8]", "\"edges\": 0.8"), "stack.layers[0].x_profiles[0].edges", "数组"));
    CHECK(rejected(edit(base, "\"edges\": [0.0, 0.3, 0.8]", "\"edges\": [0.0, null, 0.8]"), "stack.layers[0].x_profiles[0].edges[1]", "数"));
    CHECK(rejected(edit(base, "\"edges\": [0.0, 0.3, 0.8]", "\"edges\": [0.0, 0.9, 0.8]"), "stack.layers[0].x_profiles[0].edges[2]", "递增"));
    CHECK(rejected(edit(base, "\"edges\": [0.0, 0.3, 0.8]", "\"edges\": [0.0, 0.3, 0.7]"), "stack.layers[0].x_profiles[0].edges", "一个周期"));
    CHECK(rejected(edit(base, "\"edges\": [0.0, 0.3, 0.8]", "\"edges\": [0.0, 0.3, 0.9]"), "stack.layers[0].x_profiles[0].edges", "Lx"));
    CHECK(rejected(edit(base, "\"edges\": [0.0, 0.3, 0.8]", "\"edges\": [0.0, 0.8]"), "stack.layers[0].x_profiles[0].n_values", "段数"));
    CHECK(rejected(edit(base, "\"n_values\": [1.0, 1.8]", "\"n_values\": [1.0]"), "stack.layers[0].x_profiles[0].n_values", "段数"));
    CHECK(rejected(edit(base, "\"n_values\": [1.0, 1.8]", "\"n_values\": []"), "stack.layers[0].x_profiles[0].n_values", "段数"));
    CHECK(rejected(edit(base, "\"n_values\": [1.0, 1.8]", "\"n_values\": 1.8"), "stack.layers[0].x_profiles[0].n_values", "数组"));
    CHECK(rejected(edit(base, "\"n_values\": [1.0, 1.8]", "\"n_values\": [1.0, -1.8]"), "stack.layers[0].x_profiles[0].n_values[1]"));
    CHECK(rejected(edit(base, "\"n_values\": [1.0, 1.8]", "\"n_values\": [0, 1.8]"), "stack.layers[0].x_profiles[0].n_values[0]"));
    CHECK(rejected(edit(base, "\"n_values\": [1.0, 1.8]", "\"n_values\": [1.0, [1.8, -0.1]]"), "stack.layers[0].x_profiles[0].n_values[1]"));
    CHECK(rejected(edit(base, "\"n_values\": [1.0, 1.8]", "\"n_values\": [1.0, \"x\"]"), "stack.layers[0].x_profiles[0].n_values[1]"));
    CHECK(accepted(edit(base, "\"n_values\": [1.0, 1.8]", "\"n_values\": [1.0, [1.8, 0.2]]")));         // 吸收的结构层合法
    CHECK(accepted(edit(base, "\"edges\": [0.0, 0.3, 0.8]", "\"edges\": [-0.2, 0.3, 0.6]")));          // 起点可以不是 0

    // M、My、N
    CHECK(rejected(edit(base, "\"M\": 2, ", ""), "M", "缺少"));
    CHECK(rejected(edit(base, "\"My\": 1,", ""), "My", "缺少"));
    CHECK(rejected(edit(base, "\"M\": 2, \"My\": 1,", "\"M\": 2,"), "My", "缺少"));
    CHECK(rejected(edit(base, "\"M\": 2,", "\"M\": -1,"), "M", "负"));
    CHECK(rejected(edit(base, "\"M\": 2,", "\"M\": 1.5,"), "M", "整数"));
    CHECK(rejected(edit(base, "\"M\": 2,", "\"M\": \"2\","), "M", "整数"));
    CHECK(rejected(edit(base, "\"M\": 2,", "\"M\": null,"), "M", "整数"));
    CHECK(rejected(edit(base, "\"M\": 2,", "\"M\": 100000,"), "M", "1023"));
    CHECK(rejected(edit(base, "\"M\": 2,", "\"M\": 1024,"), "M", "1023"));                      // 2M+1 = 2049 > 2048
    CHECK(rejected(edit(base, "\"M\": 2,", "\"M\": 2147483647,"), "M", "1023"));
    CHECK(rejected(edit(base, "\"M\": 2,", "\"M\": 4294967296,"), "M", "整数"));
    CHECK(rejected(edit(base, "\"My\": 1,", "\"My\": -1,"), "My", "负"));
    CHECK(rejected(edit(base, "\"My\": 1,", "\"My\": 0.5,"), "My", "整数"));
    CHECK(rejected(edit(base, "\"My\": 1,", "\"My\": 5000,"), "My", "1023"));
    CHECK(rejected(edit(base, "\"My\": 1,", "\"My\": 1e30,"), "My", "整数"));
    CHECK(rejected(edit(edit(base, "\"M\": 2,", "\"M\": 30,"), "\"My\": 1,", "\"My\": 30,"), "My", "2048"));      // N = 61² = 3721
    CHECK(rejected(edit(edit(base, "\"M\": 2,", "\"M\": 22,"), "\"My\": 1,", "\"My\": 23,"), "My", "2048"));      // N = 45·47 = 2115
    CHECK(rejected(edit(edit(base, "\"M\": 2,", "\"M\": 1,"), "\"My\": 1,", "\"My\": 341,"), "My", "2049"));       // N = 3·683 = 2049（刚好超过一个）
    CHECK(rejected(edit(edit(base, "\"M\": 2,", "\"M\": 12,"), "\"My\": 1,", "\"My\": 41,"), "My", "2075"));       // N = 25·83
    // 上限两侧：N = 2047 合法（没有结构层时求解几乎不花时间）；上面的 N ≥ 2049 都被拒绝
    {
        Case c = base_case();
        c.stack.layers.clear();
        c.M = 1023;
        c.My = 0;                                              // N = 2047 · 1
        const rcwa::JsonReply wide = rcwa::solve_json_reply(far_request(c));
        CHECK(wide.ok);
        CHECK(wide.ok && head_matches(parse_or_null(wide.text), 1023, 0, 1, 1));
        c.M = 11;
        c.My = 44;                                             // N = 23·89 = 2047，纵横各有谐波
        c.stack.layers.push_back(rcwa::Layer2D::uniform(0.115, cplx{1.96, 0.0}));
        const rcwa::JsonReply both = rcwa::solve_json_reply(far_request(c));
        CHECK(both.ok);
        CHECK(both.ok && head_matches(parse_or_null(both.text), 11, 44, 1, 1));
        c.M = 1024;
        c.My = 0;
        CHECK(rejected(far_request(c), "M", "1023"));
        c.M = 0;
        c.My = 1024;
        CHECK(rejected(far_request(c), "My", "1023"));
        c.M = 22;
        c.My = 22;                                             // N = 45² = 2025 合法（也只是检查，不求解：用没有层的结构）
        c.stack.layers.clear();
        CHECK(accepted(far_request(c)));
    }
    CHECK(accepted(edit(base, "\"My\": 1,", "\"My\": 0,")));                                      // My = 0 合法
    CHECK(accepted(edit(base, "\"M\": 2,", "\"M\": 0,")));                                        // M = 0 合法

    // 入射点
    const std::string point = "\"points\": [{\"wavelength\": 1.053, \"theta_deg\": 42.0, \"phi_deg\": -25.0}]";
    CHECK(accepted(edit(base, point, "\"wavelength\": 1.053, \"theta_deg\": 42.0")));              // 单点写在最外层，phi_deg 不写取 0
    CHECK(accepted(edit(base, point, "\"wavelength\": 1.053, \"theta_deg\": 42.0, \"phi_deg\": 10.0")));
    CHECK(rejected(edit(base, point, "\"theta_deg\": 42.0"), "wavelength", "缺少"));
    CHECK(rejected(edit(base, point, "\"wavelength\": 1.053"), "theta_deg", "缺少"));
    CHECK(rejected(edit(base, point + ",", ""), "points", "缺少入射点"));
    CHECK(rejected(edit(base, point, point + ", \"wavelength\": 1.0, \"theta_deg\": 0"), "points", "一种写法"));
    CHECK(rejected(edit(base, point, "\"points\": []"), "points", "至少"));
    CHECK(rejected(edit(base, point, "\"points\": {}"), "points", "数组"));
    CHECK(rejected(edit(base, point, "\"points\": [3]"), "points[0]", "对象"));
    CHECK(rejected(edit(base, point, "\"points\": [{\"theta_deg\": 42.0}]"), "points[0].wavelength", "缺少"));
    CHECK(rejected(edit(base, point, "\"points\": [{\"wavelength\": 1.0, \"theta_deg\": 0.0}, {\"wavelength\": -1.0, \"theta_deg\": 0.0}]"), "points[1].wavelength", "正数"));
    CHECK(rejected(edit(base, point, "\"points\": [{\"wavelength\": 0, \"theta_deg\": 0.0}]"), "points[0].wavelength", "正数"));
    CHECK(rejected(edit(base, point, "\"points\": [{\"wavelength\": 1.0, \"theta_deg\": 90.0}]"), "points[0].theta_deg", "90"));
    CHECK(rejected(edit(base, point, "\"points\": [{\"wavelength\": 1.0, \"theta_deg\": -91.0}]"), "points[0].theta_deg", "90"));
    CHECK(rejected(edit(base, point, "\"points\": [{\"wavelength\": 1.0, \"theta_deg\": \"x\"}]"), "points[0].theta_deg"));
    CHECK(rejected(edit(base, point, "\"points\": [{\"wavelength\": 1.0, \"theta_deg\": 0.0, \"phi_deg\": \"x\"}]"), "points[0].phi_deg", "数"));
    CHECK(rejected(edit(base, point, "\"wavelength\": 1.0, \"theta_deg\": 0.0, \"phi_deg\": null"), "phi_deg", "数"));
    CHECK(rejected(edit(base, point, "\"wavelength\": 0.0, \"theta_deg\": 0.0"), "wavelength", "正数"));
    CHECK(accepted(edit(base, point, "\"points\": [{\"wavelength\": 1.0, \"theta_deg\": 89.0, \"phi_deg\": 720.0}]")));   // φ 任意
    CHECK(accepted(edit(base, point, "\"points\": [{\"wavelength\": 1.0, \"theta_deg\": -89.0}]")));

    // input：二维必须写
    const std::string input = "\"input\": {\"psi_deg\": 90.0, \"delta_deg\": 0.0},";
    CHECK(rejected(edit(base, input, ""), "input", "缺少"));
    CHECK(rejected(edit(base, input, "\"input\": 3,"), "input", "对象"));
    CHECK(rejected(edit(base, input, "\"input\": {},"), "input.psi_deg", "缺少"));
    CHECK(rejected(edit(base, input, "\"input\": {\"delta_deg\": 30.0},"), "input.psi_deg", "缺少"));
    CHECK(rejected(edit(base, input, "\"input\": {\"psi_deg\": \"x\"},"), "input.psi_deg", "数"));
    CHECK(rejected(edit(base, input, "\"input\": {\"psi_deg\": 45.0, \"delta_deg\": null},"), "input.delta_deg", "数"));
    CHECK(rejected(edit(base, input, "\"input\": {\"psi_deg\": 45.0, \"jones\": [1, 0]},"), "input", "一种写法"));
    CHECK(rejected(edit(base, input, "\"input\": {\"jones\": [1]},"), "input.jones", "[Es, Ep]"));
    CHECK(rejected(edit(base, input, "\"input\": {\"jones\": [1, 0, 0]},"), "input.jones"));
    CHECK(rejected(edit(base, input, "\"input\": {\"jones\": [0, [0, 0]]},"), "input.jones", "零"));
    CHECK(rejected(edit(base, input, "\"input\": {\"jones\": [1, \"p\"]},"), "input.jones[1]"));
    CHECK(rejected(edit(base, input, "\"input\": {\"jones\": [[1, 2, 3], 0]},"), "input.jones[0]"));
    CHECK(accepted(edit(base, input, "\"input\": {\"jones\": [1, 0]},")));
    CHECK(accepted(edit(base, input, "\"input\": {\"jones\": [[0.3, 0.1], [0.2, -0.5]]},")));
    CHECK(accepted(edit(base, input, "\"input\": {\"psi_deg\": 30.0},")));                         // delta_deg 不写取 0

    // options
    const std::string options = "\"options\": {\"eig\": \"qr\", \"factorization\": \"li\", \"homogeneous_fast_path\": true,\n             \"parallel_layers\": false, \"diagnostics\": false, \"threads\": 1}";
    CHECK(rejected(edit(base, options, "\"options\": 5"), "options", "对象"));
    CHECK(accepted(edit(base, options, "\"options\": {}")));
    CHECK(rejected(edit(base, "\"eig\": \"qr\"", "\"eig\": \"qz\""), "options.eig", "qr"));
    CHECK(rejected(edit(base, "\"eig\": \"qr\"", "\"eig\": 3"), "options.eig", "字符串"));
    CHECK(rejected(edit(base, "\"factorization\": \"li\"", "\"factorization\": \"Li\""), "options.factorization", "laurent"));
    CHECK(rejected(edit(base, "\"factorization\": \"li\"", "\"factorization\": null"), "options.factorization"));
    CHECK(rejected(edit(base, "\"homogeneous_fast_path\": true", "\"homogeneous_fast_path\": 1"), "options.homogeneous_fast_path", "true"));
    CHECK(rejected(edit(base, "\"parallel_layers\": false", "\"parallel_layers\": \"no\""), "options.parallel_layers", "true"));
    CHECK(rejected(edit(base, "\"diagnostics\": false", "\"diagnostics\": null"), "options.diagnostics", "true"));
    CHECK(rejected(edit(base, "\"threads\": 1", "\"threads\": 0"), "options.threads", "256"));
    CHECK(rejected(edit(base, "\"threads\": 1", "\"threads\": 257"), "options.threads", "256"));
    CHECK(rejected(edit(base, "\"threads\": 1", "\"threads\": 2.5"), "options.threads", "整数"));
    CHECK(rejected(edit(base, "\"threads\": 1", "\"threads\": \"2\""), "options.threads", "整数"));
    CHECK(accepted(edit(base, "\"eig\": \"qr\"", "\"eig\": \"lr\"")));                              // lr 被接受（二维上常因残差检查返回数值失败）
    CHECK(accepted(edit(base, "\"factorization\": \"li\"", "\"factorization\": \"laurent\"")));

    // 一维请求里写二维的键，仍是一维的未知键错误；二维的 input、Lx 在一维里同样不认识
    {
        const std::string planar =
            R"({"mode": "planar", "stack": {"nI": 1.0, "nII": 1.46, "period": 0.574, "layers": []}, "M": 2, "polarization": "TE", "wavelength": 1.0, "theta_deg": 10.0})";
        CHECK(accepted(planar));
        CHECK(rejected(edit(planar, "\"M\": 2,", "\"M\": 2, \"My\": 1,"), "My", "不认识的键"));
        CHECK(rejected(edit(planar, "\"period\": 0.574,", "\"period\": 0.574, \"Lx\": 0.574,"), "stack.Lx", "不认识的键"));
        CHECK(rejected(edit(planar, "\"M\": 2,", "\"M\": 2, \"phi_deg\": 1.0,"), "phi_deg", "conical"));
        const std::string conical = edit(edit(planar, "\"planar\"", "\"conical\""), "\"polarization\": \"TE\", ", "");
        CHECK(accepted(conical));
        CHECK(rejected(edit(conical, "\"M\": 2,", "\"M\": 2, \"My\": 1,"), "My", "不认识的键"));
        CHECK(rejected(edit(conical, "\"M\": 2,", "\"M\": 2, \"factorization\": \"li\","), "factorization", "不认识的键"));
        CHECK(rejected(edit(conical, "\"M\": 2,", "\"M\": 2, \"options\": {\"factorization\": \"li\"},"), "options.factorization", "不认识的键"));
        CHECK(rejected(edit(planar, "\"M\": 2,", "\"M\": 2, \"options\": {\"factorization\": \"li\"},"), "options.factorization", "不认识的键"));
        CHECK(rejected(edit(planar, "\"mode\": \"planar\"", "\"mode\": \"periodic1d\""), "mode", "periodic2d"));
    }
    CHECK(!g_bad_edit);
    CHECK(num_threads() == 1);
}

static void test_far_field_basic() {
    say("[远场应答：与直接调用 solve2d 逐位相同]\n");
    {
        const Case c = base_case();
        const rcwa::JsonReply reply = rcwa::solve_json_reply(far_request(c));
        CHECK(reply.ok);
        const json::Value v = parse_or_null(reply.text);
        CHECK(head_matches(v, 2, 1, 1, 1));
        CHECK(v.at("results").size() == 1 && result_matches(v.at("results").at(0), c.wavelength, c.theta_deg, c.phi_deg, direct_solve(c)));
        const rcwa::Result2D d = direct_solve(c);
        CHECK(d.status == rcwa::Status2D::Ok && d.N == 15 && d.reflectance() + d.transmittance() > 0.99);
        // 应答里写的 orders 恰是 Result2D 的 (m, n)
        bool orders_ok = true;
        for (int i = 0; i < d.N; ++i)
            orders_ok = orders_ok && is_int(v.at("orders").at(static_cast<std::size_t>(i)).at(0), d.m[static_cast<std::size_t>(i)]) &&
                        is_int(v.at("orders").at(static_cast<std::size_t>(i)).at(1), d.n[static_cast<std::size_t>(i)]);
        CHECK(orders_ok);
        // 把文字原样写进文本再读：与请求写成一行 / 缩进无关
        const rcwa::JsonReply pretty = rcwa::solve_json_reply(far_request(c), 2);
        CHECK(pretty.ok && same_json(parse_or_null(pretty.text), v, {"timing"}));
        CHECK(contains(pretty.text, "\n"));
    }
    {   // 经 solve_json（只给一个字符串的入口）也一样，应答是一行、以 { 开头、没有换行
        const std::string text = rcwa::solve_json(far_request(base_case()));
        CHECK(!text.empty() && text.front() == '{' && text.find('\n') == std::string::npos);
    }
    {   // 文字请求（kBaseText，Lx ≠ Ly 的基准）与 base_case 是同一个问题
        const rcwa::JsonReply a = rcwa::solve_json_reply(kBaseText), b = rcwa::solve_json_reply(far_request(base_case()));
        CHECK(a.ok && b.ok && same_json(parse_or_null(a.text), parse_or_null(b.text), {"timing"}));
    }
    {   // 两种入射写法、单点写在最外层
        Case c = base_case();
        c.top_level_point = true;
        CHECK(far_agrees(c));
        c.jones_form = true;
        c.jones = {cplx{0.3, 0.1}, cplx{0.2, -0.5}};
        CHECK(far_agrees(c));
        c.jones = {cplx{0.0, 0.0}, cplx{1.0, 0.0}};
        c.top_level_point = false;
        CHECK(far_agrees(c));
        c.jones_form = false;
        c.psi_deg = 37.0;
        c.delta_deg = 71.0;
        CHECK(far_agrees(c));
        c.theta_deg = 0.0;
        c.phi_deg = 0.0;                       // θ = 0：零级的 kt = 0 分支
        CHECK(far_agrees(c));
    }
    {   // 选项：每一项单独与组合
        std::vector<Case> cases;
        Case c = base_case();
        cases.push_back(c);
        c.options.eig = EigMethod::LR;
        cases.push_back(c);
        c = base_case();
        c.options.factorization = rcwa::Factorization2D::Laurent;
        cases.push_back(c);
        c = base_case();
        c.options.homogeneous_fast_path = false;
        cases.push_back(c);
        c = base_case();
        c.options.parallel_layers = true;
        cases.push_back(c);
        c = base_case();
        c.options.diagnostics = true;
        cases.push_back(c);
        c.write_options = false;               // 没有 options 键：取默认值（diagnostics = false）
        cases.push_back(c);
        c = base_case();
        c.stack = mixed_stack();
        c.M = 3;
        c.My = 2;
        cases.push_back(c);
        c.options.diagnostics = true;
        c.options.homogeneous_fast_path = false;
        c.options.factorization = rcwa::Factorization2D::Laurent;
        cases.push_back(c);
        for (std::size_t i = 0; i < cases.size(); ++i) {
            const bool diag = cases[i].effective_options().diagnostics;
            if (!far_agrees(cases[i], diag)) say("    选项算例 %zu 不符\n", i);
            CHECK(far_agrees(cases[i], diag));
        }
        // diagnostics = true 时 max_eig_residual 是求解器给出的正数；false 时 0
        Case d = base_case();
        d.options.diagnostics = true;
        const json::Value v = parse_or_null(rcwa::solve_json_reply(far_request(d)).text);
        CHECK(v.at("results").at(0).at("max_eig_residual").as_double() > 0.0);
        CHECK(v.at("results").at(0).at("max_solve_backward_error").as_double() == 0.0);
        CHECK(direct_solve(d).max_eig_residual > 0.0);
    }
    {   // 多个点：次序同请求，每个点与直接调用相同；1 与 4 个线程
        Case c = base_case();
        c.extra_points = {{0.9, 0.0, 0.0}, {1.2, -30.0, 17.0}, {0.75, 45.0, 90.0}, {1.3, 10.0, -180.0}};
        CHECK(far_agrees(c));
        c.threads = 4;
        const int before = num_threads();
        CHECK(far_agrees(c, false, 4));
        CHECK(num_threads() == before);
        c.stack = mixed_stack();
        c.M = 2;
        c.My = 2;
        CHECK(far_agrees(c, false, 4));
        CHECK(num_threads() == before);
    }
    {   // 没有层、只有均匀层、M = 0 / My = 0、吸收出射介质
        Case c = base_case();
        c.stack.layers.clear();
        CHECK(far_agrees(c));
        c.stack.layers.push_back(rcwa::Layer2D::uniform(0.2, cplx{1.7, 0.1}));
        c.stack.nII = cplx{3.4, 0.1};
        CHECK(far_agrees(c));
        c.M = 0;
        c.My = 0;
        CHECK(far_agrees(c));
        c.M = 3;
        c.My = 0;
        c.stack = base_stack();
        CHECK(far_agrees(c));
        c.M = 0;
        c.My = 3;
        CHECK(far_agrees(c));
    }
    CHECK(!g_bad_edit);
}

// 把冻结参照的算例 c 写成请求（经 rcwa2d_ref_stack(id) 得到结构）
static Case case_from_reference(const Rcwa2dRefCase& r) {
    Case c;
    c.stack = rcwa2d_ref_stack(r.id);
    c.wavelength = r.wavelength;
    c.theta_deg = r.theta_deg;
    c.phi_deg = r.phi_deg;
    c.M = r.M1;
    c.My = r.M2;
    c.jones_form = true;
    c.jones = {cplx{r.jones[0], r.jones[1]}, cplx{r.jones[2], r.jones[3]}};
    c.options.factorization = r.laurent ? rcwa::Factorization2D::Laurent : rcwa::Factorization2D::Li;
    return c;
}

// 逐复数 |a − b| ≤ atol + rtol·|b|；记录最大差
struct Tolerance {
    double maxdiff = 0.0;
    bool ok = true;
    void add(const json::Value& got, const double want[2], double atol, double rtol) {
        if (!got.is_array() || got.size() != 2 || !got.at(0).is_number() || !got.at(1).is_number()) {
            ok = false;
            return;
        }
        const cplx a{got.at(0).as_double(), got.at(1).as_double()}, b{want[0], want[1]};
        const double diff = std::abs(a - b);
        if (!(diff <= atol + rtol * std::abs(b))) ok = false;
        if (diff > maxdiff) maxdiff = diff;
    }
    void add(const json::Value& got, double want, double atol, double rtol) {
        if (!got.is_number()) {
            ok = false;
            return;
        }
        const double diff = std::abs(got.as_double() - want);
        if (!(diff <= atol + rtol * std::abs(want))) ok = false;
        if (diff > maxdiff) maxdiff = diff;
    }
};

static void test_far_field_frozen() {
    say("[远场应答：冻结参照（Python 求解器的输出）的成功算例经 JSON 入口，≤ 冻结容差]\n");
    double worst = 0.0;
    int count = 0;
    for (int i = 0; i < kRcwa2dRefCaseCount; ++i) {
        const Rcwa2dRefCase& r = kRcwa2dRefCases[i];
        if (r.status != 0) continue;
        const Case c = case_from_reference(r);
        const rcwa::JsonReply reply = rcwa::solve_json_reply(far_request(c));
        CHECK(reply.ok);
        const json::Value v = parse_or_null(reply.text);
        CHECK(head_matches(v, r.M1, r.M2, 1, 1));
        if (!reply.ok || !v.is_object()) continue;
        const json::Value& res = v.at("results").at(0);
        CHECK(is_text(res.at("status"), "ok") && res.at("message").as_string().empty());
        Tolerance tol;
        bool structure = res.at("Kx").size() == static_cast<std::size_t>(r.count);
        for (int k = 0; k < r.count; ++k) {
            const Rcwa2dRefOrder& o = r.orders[k];
            const std::size_t j = static_cast<std::size_t>(k);
            const json::Value& pair = v.at("orders").at(j);
            structure = structure && is_int(pair.at(0), o.m) && is_int(pair.at(1), o.n) && rcwa::harmonic_index_2d(o.m, o.n, r.M1, r.M2) == k;
            tol.add(res.at("Kx").at(j), o.Kx, r.far_atol, r.far_rtol);
            tol.add(res.at("Ky").at(j), o.Ky, r.far_atol, r.far_rtol);
            tol.add(res.at("kzI").at(j), o.kzI, r.far_atol, r.far_rtol);
            tol.add(res.at("kzII").at(j), o.kzII, r.far_atol, r.far_rtol);
            tol.add(res.at("Ex_r").at(j), o.Ex_r, r.far_atol, r.far_rtol);
            tol.add(res.at("Ey_r").at(j), o.Ey_r, r.far_atol, r.far_rtol);
            tol.add(res.at("Ez_r").at(j), o.Ez_r, r.far_atol, r.far_rtol);
            tol.add(res.at("Ex_t").at(j), o.Ex_t, r.far_atol, r.far_rtol);
            tol.add(res.at("Ey_t").at(j), o.Ey_t, r.far_atol, r.far_rtol);
            tol.add(res.at("Ez_t").at(j), o.Ez_t, r.far_atol, r.far_rtol);
            tol.add(res.at("Rs").at(j), o.Rs, r.far_atol, r.far_rtol);
            tol.add(res.at("Rp").at(j), o.Rp, r.far_atol, r.far_rtol);
            tol.add(res.at("Ts").at(j), o.Ts, r.far_atol, r.far_rtol);
            tol.add(res.at("Tp").at(j), o.Tp, r.far_atol, r.far_rtol);
            tol.add(res.at("DEr").at(j), o.DEr, r.far_atol, r.far_rtol);
            tol.add(res.at("DEt").at(j), o.DEt, r.far_atol, r.far_rtol);
            structure = structure && is_int(res.at("prop_r").at(j), o.prop_r != 0 ? 1 : 0) && is_int(res.at("prop_t").at(j), o.prop_t != 0 ? 1 : 0);
        }
        CHECK(structure);
        CHECK(tol.ok);
        // 同时与直接调用逐位相同
        CHECK(result_matches(res, c.wavelength, c.theta_deg, c.phi_deg, direct_solve(c)));
        if (!tol.ok) say("    算例 %d %s 超出冻结容差（最大差 %.2e）\n", r.id, r.name, tol.maxdiff);
        worst = std::max(worst, tol.maxdiff);
        ++count;
    }
    say("  %d 个成功算例，经 JSON 入口与冻结参照的最大差 %.2e（容差 1e-9 + 1e-9·|b|）\n", count, worst);
    CHECK(count == 21);
}

static void test_far_field_failures() {
    say("[求解失败：ok 仍为 true，status 与 message 与直接调用相同，数组为 null]\n");
    int rayleigh = 0;
    for (int i = 0; i < kRcwa2dRefCaseCount; ++i) {
        const Rcwa2dRefCase& r = kRcwa2dRefCases[i];
        if (r.status == 0) continue;
        const Case c = case_from_reference(r);
        const rcwa::JsonReply reply = rcwa::solve_json_reply(far_request(c));
        CHECK(reply.ok);
        const json::Value v = parse_or_null(reply.text);
        CHECK(head_matches(v, r.M1, r.M2, 1, 1));
        if (!reply.ok || !v.is_object()) continue;
        const json::Value& res = v.at("results").at(0);
        const rcwa::Result2D d = direct_solve(c);
        CHECK(d.status == rcwa::Status2D::RayleighAnomaly);
        CHECK(result_matches(res, c.wavelength, c.theta_deg, c.phi_deg, d));
        CHECK(is_text(res.at("status"), "rayleigh_anomaly"));
        const std::string message = res.at("message").is_string() ? res.at("message").as_string() : std::string();
        CHECK(contains(message, r.failure_region));
        bool all_orders = true;
        for (int k = 0; k < r.failure_count; ++k) {
            char label[48];
            std::snprintf(label, sizeof label, "(%d, %d)", r.failure_orders[k][0], r.failure_orders[k][1]);
            all_orders = all_orders && contains(message, label);
        }
        CHECK(all_orders);
        // 所有数值数组都是 null，prop 照常，其余记账项照常给出
        CHECK(are_nulls(res.at("Rs"), static_cast<std::size_t>(d.N), true) && are_nulls(res.at("DEr"), static_cast<std::size_t>(d.N), false));
        CHECK(res.at("prop_r").size() == static_cast<std::size_t>(d.N) && res.at("prop_t").size() == static_cast<std::size_t>(d.N));
        say("  %2d %-24s status = %s，区域“%s”，(m, n) 共 %d 个都在 message 里\n", r.id, r.name,
            res.at("status").as_string().c_str(), r.failure_region, r.failure_count);
        ++rayleigh;
    }
    CHECK(rayleigh == 3);

    // LR 本征分解在二维上常因本征残差检查返回数值失败（见 rcwa2d_types.hpp）：同样 ok = true、status = numerical_failure
    int numerical = 0, solved = 0;
    for (int i = 0; i < kRcwa2dRefCaseCount; ++i) {
        const Rcwa2dRefCase& r = kRcwa2dRefCases[i];
        if (r.status != 0) continue;
        Case c = case_from_reference(r);
        c.options.eig = EigMethod::LR;
        const rcwa::JsonReply reply = rcwa::solve_json_reply(far_request(c));
        CHECK(reply.ok);
        const json::Value v = parse_or_null(reply.text);
        if (!reply.ok || !v.is_object()) continue;
        const rcwa::Result2D d = direct_solve(c);
        CHECK(result_matches(v.at("results").at(0), c.wavelength, c.theta_deg, c.phi_deg, d));
        if (d.status == rcwa::Status2D::NumericalFailure) {
            ++numerical;
            CHECK(is_text(v.at("results").at(0).at("status"), "numerical_failure"));
            CHECK(contains(v.at("results").at(0).at("message").as_string(), "数值失败"));
        } else {
            ++solved;
        }
    }
    say("  LR：%d 个算例得到 numerical_failure（应答与直接调用相同），%d 个求解成功\n", numerical, solved);
    CHECK(numerical > 0);

    // 多个点里只有一个失败：其余照常
    Case c = case_from_reference(kRcwa2dRefCases[kRcwa2dRefCaseCount - 3]);   // 编号 23：入射区 Rayleigh
    c.extra_points = {{1.1, 20.0, 5.0}};
    const json::Value v = parse_or_null(rcwa::solve_json_reply(far_request(c)).text);
    CHECK(head_matches(v, c.M, c.My, 1, 2));
    CHECK(is_text(v.at("results").at(0).at("status"), "rayleigh_anomaly"));
    CHECK(is_text(v.at("results").at(1).at("status"), "ok"));
    CHECK(result_matches(v.at("results").at(1), 1.1, 20.0, 5.0, direct_solve(c, 1.1, 20.0, 5.0)));
}

static void test_threads_and_reentrancy() {
    say("[线程数、parallel_layers、同时调用]\n");
    const int before = num_threads();
    for (int variant = 0; variant < 3; ++variant) {
        Case c = base_case();
        if (variant == 1) c.stack = mixed_stack();
        if (variant == 2) {
            c.stack = mixed_stack();
            c.options.factorization = rcwa::Factorization2D::Laurent;
            c.extra_points = {{0.9, 10.0, 3.0}, {1.1, 55.0, -80.0}};
        }
        c.M = 3;
        c.My = 2;
        c.threads = 1;
        const json::Value one = parse_or_null(rcwa::solve_json_reply(far_request(c)).text);
        c.threads = 4;
        const json::Value four = parse_or_null(rcwa::solve_json_reply(far_request(c)).text);
        CHECK(is_int(one.at("threads"), 1) && is_int(four.at("threads"), 4));
        CHECK(same_json(one, four, {"timing", "threads"}));
        c.options.parallel_layers = true;
        const json::Value par = parse_or_null(rcwa::solve_json_reply(far_request(c)).text);
        CHECK(same_json(one, par, {"timing", "threads"}));
        c.threads = 1;
        const json::Value par1 = parse_or_null(rcwa::solve_json_reply(far_request(c)).text);
        CHECK(same_json(one, par1, {"timing", "threads"}));
        CHECK(num_threads() == before);
    }
    {   // 不写 threads：沿用当前设定；写了就用，结束后复原
        set_num_threads(2);
        Case c = base_case();
        c.write_options = false;
        CHECK(is_int(parse_or_null(rcwa::solve_json_reply(far_request(c)).text).at("threads"), 2));
        c.write_options = true;
        c.threads = 3;
        CHECK(is_int(parse_or_null(rcwa::solve_json_reply(far_request(c)).text).at("threads"), 3));
        CHECK(num_threads() == 2);
        // 请求出错时也复原
        CHECK(!rcwa::solve_json_reply(edit(kBaseText, "\"threads\": 1", "\"threads\": 4, \"foo\": 1")).ok);
        CHECK(num_threads() == 2);
        set_num_threads(before);
    }
    {   // 几个线程同时调用 solve_json：每个应答与串行时相同
        std::vector<Case> jobs;
        Case c = base_case();
        jobs.push_back(c);
        c.stack = mixed_stack();
        c.theta_deg = 30.0;
        jobs.push_back(c);
        c.stack = base_stack();
        c.options.factorization = rcwa::Factorization2D::Laurent;
        c.jones_form = true;
        c.jones = {cplx{0.6, 0.2}, cplx{-0.3, 0.7}};
        jobs.push_back(c);
        c = case_from_reference(kRcwa2dRefCases[kRcwa2dRefCaseCount - 3]);        // Rayleigh 失败的也混在里面
        jobs.push_back(c);
        c = base_case();
        c.M = 3;
        c.My = 3;
        jobs.push_back(c);
        std::vector<std::string> replies(jobs.size() * 3);
        std::vector<std::thread> pool;
        const int workers = 4;
        for (int t = 0; t < workers; ++t)
            pool.emplace_back([&, t] {
                for (std::size_t i = static_cast<std::size_t>(t); i < replies.size(); i += static_cast<std::size_t>(workers))
                    replies[i] = rcwa::solve_json(far_request(jobs[i % jobs.size()]));
            });
        for (std::thread& th : pool) th.join();
        bool same = true;
        for (std::size_t i = 0; i < replies.size(); ++i) {
            const Case& job = jobs[i % jobs.size()];
            const json::Value v = parse_or_null(replies[i]);
            same = same && head_matches(v, job.M, job.My, 1, 1) &&
                   result_matches(v.at("results").at(0), job.wavelength, job.theta_deg, job.phi_deg, direct_solve(job));
        }
        CHECK(same);
        CHECK(num_threads() == before);
    }
}

// ---------------------------------------------------------------------------------------------
// NPY 读取器（最小）：校验魔数、头、dtype、shape、字节数

struct Npy {
    bool valid = false;
    std::string problem;
    std::string dtype;
    std::vector<std::size_t> shape;
    std::vector<unsigned char> bytes;     // 数据部分（不含头）
    std::size_t count() const {
        std::size_t n = 1;
        for (std::size_t d : shape) n *= d;
        return n;
    }
    double real(std::size_t i) const { return value(i * 8); }
    double cplx_re(std::size_t i) const { return value(i * 16); }
    double cplx_im(std::size_t i) const { return value(i * 16 + 8); }
    unsigned char byte(std::size_t i) const { return bytes[i]; }
private:
    double value(std::size_t offset) const {
        double x = 0.0;
        std::memcpy(&x, bytes.data() + offset, sizeof x);      // 测试平台是小端、IEEE 754
        return x;
    }
};

static Npy read_npy(const fs::path& path) {
    Npy out;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        out.problem = "打不开";
        return out;
    }
    const std::vector<unsigned char> all((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (all.size() < 10) {
        out.problem = "太短";
        return out;
    }
    const unsigned char magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    if (std::memcmp(all.data(), magic, 6) != 0) {
        out.problem = "魔数不对";
        return out;
    }
    if (all[6] != 1 || all[7] != 0) {
        out.problem = "版本不是 1.0";
        return out;
    }
    const std::size_t hlen = static_cast<std::size_t>(all[8]) | (static_cast<std::size_t>(all[9]) << 8);
    if (10 + hlen > all.size() || (10 + hlen) % 64 != 0) {
        out.problem = "头长度不对或没有按 64 字节对齐";
        return out;
    }
    const std::string header(reinterpret_cast<const char*>(all.data()) + 10, hlen);
    if (header.empty() || header.back() != '\n' || header.front() != '{') {
        out.problem = "头不是以 { 开头、换行结尾的字典";
        return out;
    }
    const std::string d1 = "{'descr': '", d2 = "', 'fortran_order': False, 'shape': (";
    if (header.compare(0, d1.size(), d1) != 0) {
        out.problem = "头里没有 descr";
        return out;
    }
    const std::size_t q = header.find('\'', d1.size());
    if (q == std::string::npos || header.compare(q, d2.size(), d2) != 0) {
        out.problem = "头里没有 fortran_order: False 与 shape";
        return out;
    }
    out.dtype = header.substr(d1.size(), q - d1.size());
    std::size_t pos = q + d2.size();
    const std::size_t close = header.find(')', pos);
    if (close == std::string::npos) {
        out.problem = "shape 没有闭合";
        return out;
    }
    // shape：用 ", " 分隔的非负整数，末尾可以多一个逗号
    while (pos < close) {
        std::size_t end = pos;
        while (end < close && header[end] >= '0' && header[end] <= '9') ++end;
        if (end == pos) {
            out.problem = "shape 里有不是数字的东西";
            return out;
        }
        out.shape.push_back(static_cast<std::size_t>(std::strtoull(header.substr(pos, end - pos).c_str(), nullptr, 10)));
        pos = end;
        if (pos < close && header[pos] == ',') ++pos;
        while (pos < close && header[pos] == ' ') ++pos;
    }
    const std::string rest = header.substr(close);
    if (rest.compare(0, 4, "), }") != 0) {
        out.problem = "shape 之后不是 '), }'";
        return out;
    }
    for (std::size_t i = 4; i + 1 < rest.size(); ++i)
        if (rest[i] != ' ') {
            out.problem = "头的补白不是空格";
            return out;
        }
    std::size_t item = 0;
    if (out.dtype == "<f8") item = 8;
    else if (out.dtype == "<c16") item = 16;
    else if (out.dtype == "|u1") item = 1;
    else {
        out.problem = "dtype 不认识：" + out.dtype;
        return out;
    }
    if (all.size() - 10 - hlen != out.count() * item) {
        out.problem = "数据的字节数与 shape 不符";
        return out;
    }
    out.bytes.assign(all.begin() + static_cast<std::ptrdiff_t>(10 + hlen), all.end());
    out.valid = true;
    return out;
}

// 一个 npy 文件是否有给定的 dtype、shape
static bool npy_is(const Npy& a, const char* dtype, const std::vector<std::size_t>& shape) {
    if (!a.valid) say("    NPY 无效：%s\n", a.problem.c_str());
    return a.valid && a.dtype == dtype && a.shape == shape;
}

// ---------------------------------------------------------------------------------------------
// 场协议

// 一次场请求
struct FieldCase {
    Case solve;
    RealVector z;
    rcwa::SlicePlane plane = rcwa::SlicePlane::XZ;
    int nt = 8;
    bool write_at = true;
    double at = 0.0;
    std::string id = "token-1";
};

static fs::path g_root;     // 本次测试的临时根目录
static int g_counter = 0;
static fs::path fresh_dir() { return g_root / ("out" + std::to_string(++g_counter)); }

static json::Value fields_value(const FieldCase& f, const fs::path& out) {
    json::Value v = json::Value::object();
    v.set("solve", solve_value(f.solve));
    json::Value sampling = json::Value::object();
    sampling.set("z", numbers_value(f.z));
    sampling.set("plane", rcwa::slice_plane_name(f.plane));
    sampling.set("nt", f.nt);
    if (f.write_at) sampling.set("at", f.at);
    v.set("sampling", std::move(sampling));
    v.set("output_dir", out.u8string());
    v.set("request_id", f.id);
    return v;
}

static bool dir_exists(const fs::path& p) {
    std::error_code ec;
    return fs::exists(p, ec);
}
// 父目录里有没有 .tmp- 开头的残留（DirectoryOutput 的临时目录）
static bool temporary_leftovers(const fs::path& parent) {
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(parent, ec))
        if (entry.path().filename().string().find(".tmp-") != std::string::npos) return true;
    return false;
}
static std::vector<std::string> file_names(const fs::path& dir) {
    std::vector<std::string> names;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(dir, ec)) names.push_back(entry.path().filename().string());
    std::sort(names.begin(), names.end());
    return names;
}

// 切面的坐标：第 j 点 t = (j/nt)·L（与求解器同一个表达式）
static RealVector expected_t(int nt, double period) {
    RealVector t(static_cast<std::size_t>(nt));
    for (int j = 0; j < nt; ++j) t[static_cast<std::size_t>(j)] = (static_cast<double>(j) / nt) * period;
    return t;
}

// 清单逐项与规格相符（见 PROTOCOL_2D.md 第 3 节）
static bool manifest_matches(const json::Value& reply, const FieldCase& f, const fs::path& out, const rcwa::FieldMetadata& meta,
                             std::size_t nz, std::size_t nt) {
    bool good = head_matches(reply, f.solve.M, f.solve.My, f.solve.threads > 0 ? f.solve.threads : 1, 1, true);
    if (!good) return false;
    good = good && is_text(reply.at("protocol"), "stablercwa-fields");
    const json::Value& fd = reply.at("field_data");
    good = good && has_keys(fd, {"version", "producer", "request_id", "directory", "layout", "shape", "components", "slice", "harmonics", "phase",
                                 "normalization", "files", "layers", "timing"});
    if (!good) return false;
    good = good && is_int(fd.at("version"), 1) && is_text(fd.at("producer"), "stablercwa.cpp.rcwa_fields") && is_text(fd.at("request_id"), f.id.c_str());
    good = good && fd.at("directory").is_string() && fd.at("directory").as_string() == fs::absolute(out).lexically_normal().u8string();
    good = good && is_text(fd.at("layout"), "C");
    good = good && fd.at("shape").is_array() && fd.at("shape").size() == 2 && is_int(fd.at("shape").at(0), static_cast<long long>(nz)) &&
           is_int(fd.at("shape").at(1), static_cast<long long>(nt));
    const char* const names[] = {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"};
    good = good && fd.at("components").is_array() && fd.at("components").size() == 6;
    for (std::size_t i = 0; good && i < 6; ++i) good = good && is_text(fd.at("components").at(i), names[i]);
    // slice：plane、at、period（沿切面方向的周期：xz 取 Lx，yz 取 Ly）
    const json::Value& slice = fd.at("slice");
    good = good && has_keys(slice, {"plane", "at", "period"}) && is_text(slice.at("plane"), rcwa::slice_plane_name(f.plane)) &&
           is_number(slice.at("at"), f.write_at ? f.at : 0.0) &&
           is_number(slice.at("period"), f.plane == rcwa::SlicePlane::XZ ? f.solve.stack.Lx : f.solve.stack.Ly);
    const json::Value& harmonics = fd.at("harmonics");
    good = good && has_keys(harmonics, {"M", "My", "N"}) && is_int(harmonics.at("M"), f.solve.M) && is_int(harmonics.at("My"), f.solve.My) &&
           is_int(harmonics.at("N"), (2 * f.solve.M + 1) * (2 * f.solve.My + 1));
    const json::Value& phase = fd.at("phase");
    good = good && has_keys(phase, {"references", "floor", "unwrap", "unit", "mask_true"}) && phase.at("references").is_array() &&
           phase.at("references").size() == 2 && is_text(phase.at("references").at(0), "absolute") && is_text(phase.at("references").at(1), "incident") &&
           is_number(phase.at("floor"), 1e-6) && is_text(phase.at("unwrap"), "none") && is_text(phase.at("unit"), "rad") &&
           is_text(phase.at("mask_true"), "undefined");
    // normalization：E = |Jones|，H = nI·|Jones|（原样取求解器的 hypot）
    const rcwa::Jones2D jones = f.solve.effective_jones();
    const double jones_norm = std::hypot(std::abs(jones[0]), std::abs(jones[1]));
    const json::Value& norm = fd.at("normalization");
    good = good && has_keys(norm, {"E", "H", "H_convention"}) && is_number(norm.at("E"), jones_norm) &&
           is_number(norm.at("H"), f.solve.stack.nI * jones_norm) && is_text(norm.at("H_convention"), "Z0 H_physical");
    good = good && is_number(norm.at("E"), meta.normalization.electric_incident_amplitude) &&
           is_number(norm.at("H"), meta.normalization.magnetic_incident_amplitude);
    if (!f.solve.jones_form) good = good && std::abs(norm.at("E").as_double() - 1.0) <= 4e-16;     // psi/delta 写法：模为 1
    // files：七个文件，键的次序、name、dtype、shape
    const json::Value& files = fd.at("files");
    good = good && has_keys(files, {"fields", "phase", "phase_mask", "t", "x", "y", "z"});
    struct Spec { const char* key; const char* name; const char* dtype; std::vector<std::size_t> shape; };
    const std::vector<Spec> specs = {{"fields", "fields.npy", "<c16", {6, nz, nt}},          {"phase", "phase.npy", "<f8", {2, 6, nz, nt}},
                                     {"phase_mask", "phase_mask.npy", "|u1", {2, 6, nz, nt}}, {"t", "t.npy", "<f8", {nt}},
                                     {"x", "x.npy", "<f8", {nt}},                              {"y", "y.npy", "<f8", {nt}},
                                     {"z", "z.npy", "<f8", {nz}}};
    for (const Spec& s : specs) {
        if (!good) break;
        const json::Value& entry = files.at(s.key);
        good = good && has_keys(entry, {"name", "dtype", "shape"}) && is_text(entry.at("name"), s.name) && is_text(entry.at("dtype"), s.dtype) &&
               entry.at("shape").is_array() && entry.at("shape").size() == s.shape.size();
        for (std::size_t i = 0; good && i < s.shape.size(); ++i) good = good && is_int(entry.at("shape").at(i), static_cast<long long>(s.shape[i]));
    }
    // layers：每层都有（含零厚层），effective_index = original_index，finite_zero_modes 0，schur false
    const json::Value& layers = fd.at("layers");
    good = good && layers.is_array() && layers.size() == f.solve.stack.layers.size();
    double z = 0.0;
    for (std::size_t j = 0; good && j < layers.size(); ++j) {
        const json::Value& layer = layers.at(j);
        const double top = z;
        z += f.solve.stack.layers[j].d;
        good = good && has_keys(layer, {"original_index", "effective_index", "z_top", "z_bottom", "finite_zero_modes", "schur"}) &&
               is_int(layer.at("original_index"), static_cast<long long>(j)) && is_int(layer.at("effective_index"), static_cast<long long>(j)) &&
               is_number(layer.at("z_top"), top) && is_number(layer.at("z_bottom"), z) && is_int(layer.at("finite_zero_modes"), 0) &&
               layer.at("schur").is_bool() && !layer.at("schur").as_bool();
    }
    const json::Value& timing = fd.at("timing");
    good = good && has_keys(timing, {"solve", "sample", "phase", "write"});
    for (std::size_t i = 0; good && i < timing.size(); ++i) good = good && finite_nonneg(timing.at(i));
    return good;
}

struct FieldCheck {
    bool reply_ok = false;
    bool manifest = false;
    bool files = false;             // 七个 NPY 与直接 field_slice / phase_slice 逐位相同
    bool masks = false;             // 掩膜规则（独立重算）与相位（独立重算）
    bool listing = false;           // 目录里恰有这七个文件，没有临时目录残留
    std::size_t masked = 0, total = 0;
    bool all() const { return reply_ok && manifest && files && masks && listing; }
};

// 运行一次场请求并全面检查
static FieldCheck run_field_case(const FieldCase& f, const fs::path& out, const char* label, int indent = -1) {
    FieldCheck result;
    const std::string request = json::dump(fields_value(f, out), indent);
    const rcwa::JsonReply reply = rcwa::solve_fields_json_reply(request, indent);
    result.reply_ok = reply.ok;
    if (!reply.ok) {
        say("    [%s] 应答：%.400s\n", label, reply.text.c_str());
        return result;
    }
    const json::Value v = parse_or_null(reply.text);
    // 直接调用
    const rcwa::FieldResult2D run = rcwa::solve2d_with_fields(f.solve.stack, f.solve.wavelength, f.solve.theta_deg, f.solve.phi_deg, f.solve.M, f.solve.My,
                                                              f.solve.effective_jones(), f.solve.options);
    const double at = f.write_at ? f.at : 0.0;
    const rcwa::FieldGridSlice grid = rcwa::field_slice(run, f.z, f.plane, f.nt, at);
    const std::size_t nt = grid.nt(), nz = grid.nz();
    const rcwa::Result2D plain = direct_solve(f.solve);
    result.manifest = v.is_object() && manifest_matches(v, f, out, grid.metadata, nz, nt) &&
                      result_matches(v.at("results").at(0), f.solve.wavelength, f.solve.theta_deg, f.solve.phi_deg, plain, f.solve.options.diagnostics);
    if (!result.manifest) say("    [%s] 清单或远场结果不符\n", label);

    // 目录：恰有七个文件，没有临时目录残留
    const std::vector<std::string> want = {"fields.npy", "phase.npy", "phase_mask.npy", "t.npy", "x.npy", "y.npy", "z.npy"};
    std::vector<std::string> sorted = want;
    std::sort(sorted.begin(), sorted.end());
    result.listing = dir_exists(out) && file_names(out) == sorted && !temporary_leftovers(out.parent_path());
    if (!result.listing) say("    [%s] 输出目录里的文件不对或有临时目录残留\n", label);

    const Npy fields = read_npy(out / "fields.npy"), phase = read_npy(out / "phase.npy"), mask = read_npy(out / "phase_mask.npy");
    const Npy t = read_npy(out / "t.npy"), x = read_npy(out / "x.npy"), y = read_npy(out / "y.npy"), z = read_npy(out / "z.npy");
    const bool shapes = npy_is(fields, "<c16", {6, nz, nt}) && npy_is(phase, "<f8", {2, 6, nz, nt}) && npy_is(mask, "|u1", {2, 6, nz, nt}) &&
                        npy_is(t, "<f8", {nt}) && npy_is(x, "<f8", {nt}) && npy_is(y, "<f8", {nt}) && npy_is(z, "<f8", {nz});
    if (!shapes) {
        say("    [%s] NPY 的 dtype 或 shape 不符\n", label);
        return result;
    }
    // 坐标
    const double period = f.plane == rcwa::SlicePlane::XZ ? f.solve.stack.Lx : f.solve.stack.Ly;
    const RealVector tt = expected_t(f.nt, period);
    bool coordinates = nt == static_cast<std::size_t>(f.nt) && nz == f.z.size();
    for (std::size_t j = 0; j < nt; ++j) {
        coordinates = coordinates && same_bits(t.real(j), tt[j]) && same_bits(t.real(j), grid.t[j]);
        coordinates = coordinates && same_bits(x.real(j), f.plane == rcwa::SlicePlane::XZ ? tt[j] : at);       // xz：x = t；yz：x 全是 at
        coordinates = coordinates && same_bits(y.real(j), f.plane == rcwa::SlicePlane::XZ ? at : tt[j]);       // xz：y 全是 at；yz：y = t
        coordinates = coordinates && same_bits(x.real(j), grid.x[j]) && same_bits(y.real(j), grid.y[j]);
    }
    for (std::size_t k = 0; k < nz; ++k) coordinates = coordinates && same_bits(z.real(k), f.z[k]);          // z 保留输入顺序与重复值
    // fields.npy 与直接 field_slice 逐位相同
    bool same_fields = true;
    for (std::size_t c = 0; c < 6; ++c) {
        const Matrix& component = grid.component(static_cast<rcwa::FieldComponent>(c));
        for (std::size_t j = 0; j < nz * nt; ++j) {
            const cplx value = component.data()[j];
            const std::size_t index = c * nz * nt + j;
            same_fields = same_fields && same_bits(fields.cplx_re(index), value.real()) && same_bits(fields.cplx_im(index), value.imag());
        }
    }
    // phase.npy、phase_mask.npy 与直接 phase_slice 逐位相同（无定义处相位 NaN、掩膜 1）
    bool same_phase = true;
    std::size_t masked = 0, total = 0;
    for (std::size_t ref = 0; ref < 2; ++ref)
        for (std::size_t c = 0; c < 6; ++c) {
            rcwa::PhaseOptions options;
            options.reference = ref == 0 ? rcwa::PhaseReference::Absolute : rcwa::PhaseReference::Incident;
            options.phase_floor = 1e-6;
            options.unwrap = rcwa::PhaseUnwrap::None;
            const rcwa::PhaseGrid pg = rcwa::phase_slice(run, f.z, f.plane, f.nt, at, static_cast<rcwa::FieldComponent>(c), options);
            same_phase = same_phase && pg.phase.size() == nz * nt && pg.mask.size() == nz * nt;
            for (std::size_t j = 0; same_phase && j < nz * nt; ++j) {
                const std::size_t index = (ref * 6 + c) * nz * nt + j;
                const bool m = pg.mask[j];
                same_phase = same_phase && mask.byte(index) == (m ? 1 : 0);
                if (m) same_phase = same_phase && std::isnan(phase.real(index));
                else same_phase = same_phase && same_bits(phase.real(index), pg.phase[j]);
                if (ref == 0) {
                    ++total;
                    if (m) ++masked;
                }
            }
        }
    result.files = coordinates && same_fields && same_phase;
    if (!coordinates) say("    [%s] 坐标 NPY 不符\n", label);
    if (!same_fields) say("    [%s] fields.npy 与 field_slice 不是逐位相同\n", label);
    if (!same_phase) say("    [%s] phase / phase_mask 与 phase_slice 不是逐位相同\n", label);
    result.masked = masked;
    result.total = total;

    // 掩膜规则与相位：只用 fields.npy 里的数独立重算
    //   掩膜 = 非有限，或 幅值 <= 1e-6 × 该分量的有限幅值峰值；相对入射载波时再加“参考非有限或为零”（这里不会发生）；
    //   绝对相位 = arg(场)；入射载波 = exp(i(−kx0·x − ky0·y + kz0·z))，(kx0, ky0, kz0) 取 (0, 0) 级，x、y 取每个采样点的实际值。
    bool rule = true;
    const rcwa::Result2D& r = plain;
    const std::size_t i0 = static_cast<std::size_t>(r.i0);
    const double kx0 = r.k0 * r.Kx[i0], ky0 = r.k0 * r.Ky[i0], kz0 = (r.k0 * r.kzI[i0]).real();
    double worst_incident = 0.0;
    for (std::size_t c = 0; c < 6; ++c) {
        double peak = 0.0;
        for (std::size_t j = 0; j < nz * nt; ++j) {
            const std::size_t index = c * nz * nt + j;
            const double amplitude = std::abs(cplx{fields.cplx_re(index), fields.cplx_im(index)});
            if (std::isfinite(amplitude)) peak = std::max(peak, amplitude);
        }
        for (std::size_t iz = 0; iz < nz; ++iz)
            for (std::size_t it = 0; it < nt; ++it) {
                const std::size_t j = iz * nt + it;
                const cplx value{fields.cplx_re(c * nz * nt + j), fields.cplx_im(c * nz * nt + j)};
                const bool undefined = !std::isfinite(value.real()) || !std::isfinite(value.imag()) || std::abs(value) <= 1e-6 * peak;
                const std::size_t absolute = (0 * 6 + c) * nz * nt + j, incident = (1 * 6 + c) * nz * nt + j;
                rule = rule && mask.byte(absolute) == (undefined ? 1 : 0) && mask.byte(incident) == (undefined ? 1 : 0);
                if (undefined) {
                    rule = rule && std::isnan(phase.real(absolute)) && std::isnan(phase.real(incident));
                    continue;
                }
                rule = rule && same_bits(phase.real(absolute), std::arg(value));
                const double argument = -kx0 * x.real(it) - ky0 * y.real(it) + kz0 * z.real(iz);
                const cplx carrier = std::exp(cplx{0.0, 1.0} * argument);
                const double expected = std::arg(value * std::conj(carrier));
                double diff = std::abs(phase.real(incident) - expected);
                diff = std::min(diff, std::abs(diff - 2.0 * kPi));
                worst_incident = std::max(worst_incident, diff);
                rule = rule && diff <= 1e-9;
            }
    }
    result.masks = rule;
    if (!rule) say("    [%s] 掩膜规则或相位的独立重算不符（相对入射相位最大差 %.2e）\n", label, worst_incident);
    return result;
}

static FieldCase base_field_case() {
    FieldCase f;
    f.solve = base_case();
    f.solve.M = 3;
    f.solve.My = 3;
    f.z = {-0.25, 0.0, 0.1, 0.3, 0.3, 0.2, 0.415, 0.5, 0.0};     // 区域 I、z = 0、层内、层界面（重复）、乱序、膜层底、区域 II、重复的 0
    f.nt = 8;
    f.at = 0.4;
    return f;
}

static void test_field_slices() {
    say("[场协议：NPY 与直接 field_slice / phase_slice 逐位相同；清单、掩膜、相位]\n");
    g_root = fs::temp_directory_path() / ("rcwa2d_json_test_" + std::to_string(static_cast<long long>(std::chrono::steady_clock::now().time_since_epoch().count())));
    fs::create_directories(g_root);

    struct Config {
        const char* label;
        FieldCase f;
    };
    std::vector<Config> configs;
    {
        FieldCase f = base_field_case();
        configs.push_back({"xz at=0.4 nt=8", f});
        f.plane = rcwa::SlicePlane::YZ;
        configs.push_back({"yz at=0.4 nt=8", f});
        f.at = -0.3;
        f.nt = 5;
        configs.push_back({"yz at=-0.3 nt=5", f});
        f.plane = rcwa::SlicePlane::XZ;
        f.at = 1.7;                                  // 超出一个周期的 at：周期延拓
        f.nt = 16;
        configs.push_back({"xz at=1.7 nt=16", f});
        f.write_at = false;                          // at 不写：取 0
        f.nt = 1;
        configs.push_back({"xz at omitted nt=1", f});
        f.plane = rcwa::SlicePlane::YZ;
        f.nt = 2;
        f.z = {0.1};
        configs.push_back({"yz nt=2 nz=1", f});
        f.z = {0.415, 0.415, 0.415, -0.01, -0.01};   // 全是重复值、区域 I 紧靠界面
        f.nt = 3;
        f.write_at = true;
        f.at = 0.0;
        configs.push_back({"yz nz=5 duplicates", f});
        f.z = {0.0};
        f.nt = 7;
        f.plane = rcwa::SlicePlane::XZ;
        configs.push_back({"xz nz=1 z=0", f});
        f.z = {0.7, 0.9, 0.8, 0.45, 0.1};            // 全在区域 II / 层内，乱序
        f.nt = 12;
        configs.push_back({"xz unsorted", f});
    }
    {   // 另一个结构：零厚层 + 吸收层 + 吸收出射介质、Lx ≠ Ly、椭圆偏振的 Jones、Laurent、快路径关
        FieldCase f;
        f.solve.stack = mixed_stack();
        f.solve.M = 2;
        f.solve.My = 2;
        f.solve.theta_deg = 33.0;
        f.solve.phi_deg = 71.0;
        f.solve.wavelength = 0.83;
        f.solve.jones_form = true;
        f.solve.jones = {cplx{0.3, 0.1}, cplx{0.2, -0.5}};
        f.solve.options.factorization = rcwa::Factorization2D::Laurent;
        f.solve.options.homogeneous_fast_path = false;
        f.z = {-0.1, 0.0, 0.05, 0.2, 0.25, 0.3, 0.37, 0.6};
        f.nt = 9;
        f.at = 0.25;
        f.plane = rcwa::SlicePlane::YZ;
        configs.push_back({"mixed yz", f});
        f.plane = rcwa::SlicePlane::XZ;
        f.at = 0.35;
        configs.push_back({"mixed xz", f});
        f.solve.jones_form = false;
        f.solve.psi_deg = 30.0;
        f.solve.delta_deg = 50.0;
        f.solve.options = rcwa::Options2D{};
        f.solve.options.diagnostics = true;
        f.solve.options.parallel_layers = true;
        configs.push_back({"mixed psi-delta diagnostics parallel", f});
    }
    {   // θ = 0、没有层：Ex、Ez 等分量恒为零（整个分量被掩膜）
        FieldCase f;
        f.solve.stack = rcwa::Stack2D{};
        f.solve.stack.nI = 1.0;
        f.solve.stack.nII = 1.5;
        f.solve.stack.Lx = 0.8;
        f.solve.stack.Ly = 0.8;
        f.solve.theta_deg = 0.0;
        f.solve.phi_deg = 0.0;
        f.solve.M = 1;
        f.solve.My = 1;
        f.z = {-0.3, 0.0, 0.4};
        f.nt = 6;
        f.at = 0.1;
        configs.push_back({"no layers normal incidence", f});
    }
    {   // 全内反射：区域 II 只有倏逝波，深处的场相对峰值低于 1e-6，部分点被掩膜
        FieldCase f;
        f.solve.stack = tir_stack();
        f.solve.theta_deg = 60.0;
        f.solve.phi_deg = 20.0;
        f.solve.M = 0;                               // 只有零级：透射区只有倏逝波
        f.solve.My = 0;
        f.solve.psi_deg = 45.0;
        f.solve.delta_deg = 20.0;
        f.z = {-0.2, 0.0, 0.5, 1.5, 3.0};
        f.nt = 5;
        f.at = 0.2;
        configs.push_back({"total internal reflection (partial masks)", f});
        f.plane = rcwa::SlicePlane::YZ;
        configs.push_back({"total internal reflection yz", f});
    }

    bool any_partial_mask = false, any_mask = false;
    for (const Config& cfg : configs) {
        const fs::path out = fresh_dir();
        const FieldCheck r = run_field_case(cfg.f, out, cfg.label);
        CHECK(r.reply_ok);
        CHECK(r.manifest);
        CHECK(r.files);
        CHECK(r.masks);
        CHECK(r.listing);
        say("  %-44s nz=%2zu nt=%2d  掩膜 %zu / %zu\n", cfg.label, cfg.f.z.size(), cfg.f.nt, r.masked, r.total);
        if (r.masked > 0 && r.masked < r.total) any_partial_mask = true;
        if (r.masked > 0) any_mask = true;
    }
    CHECK(any_partial_mask);       // 掩膜规则在“部分点被掩膜”的算例里被真正检验过
    CHECK(any_mask);

    // 缩进的应答写法与一行写法的内容相同
    {
        FieldCase f = base_field_case();
        const fs::path a = fresh_dir(), b = fresh_dir();
        const rcwa::JsonReply one = rcwa::solve_fields_json_reply(json::dump(fields_value(f, a)), -1);
        const rcwa::JsonReply two = rcwa::solve_fields_json_reply(json::dump(fields_value(f, b), 2), 2);
        CHECK(one.ok && two.ok && contains(two.text, "\n"));
        json::Value va = parse_or_null(one.text), vb = parse_or_null(two.text);
        ref(va, "field_data").set("directory", "");
        ref(vb, "field_data").set("directory", "");
        CHECK(same_json(va, vb, {"timing"}));
    }
}

// 线程数：fields 的全部输出与线程数无关；parallel_layers 同
static void test_field_threads() {
    say("[场协议：options.threads 为 1 与 4、parallel_layers：应答与每个 NPY 文件逐位相同]\n");
    FieldCase f = base_field_case();
    f.solve.stack = mixed_stack();
    f.solve.M = 3;
    f.solve.My = 2;
    f.z = {-0.1, 0.0, 0.1, 0.25, 0.37, 0.5};
    f.plane = rcwa::SlicePlane::YZ;
    f.solve.threads = 1;
    const fs::path a = fresh_dir();
    const FieldCheck ra = run_field_case(f, a, "threads=1");
    f.solve.threads = 4;
    const fs::path b = fresh_dir();
    const FieldCheck rb = run_field_case(f, b, "threads=4");
    f.solve.options.parallel_layers = true;
    const fs::path c = fresh_dir();
    const FieldCheck rc = run_field_case(f, c, "threads=4 parallel_layers");
    CHECK(ra.all() && rb.all() && rc.all());
    for (const char* name : {"fields.npy", "phase.npy", "phase_mask.npy", "t.npy", "x.npy", "y.npy", "z.npy"}) {
        std::ifstream fa(a / name, std::ios::binary), fb(b / name, std::ios::binary), fc(c / name, std::ios::binary);
        const std::string sa((std::istreambuf_iterator<char>(fa)), std::istreambuf_iterator<char>());
        const std::string sb((std::istreambuf_iterator<char>(fb)), std::istreambuf_iterator<char>());
        const std::string sc((std::istreambuf_iterator<char>(fc)), std::istreambuf_iterator<char>());
        CHECK(!sa.empty() && sa == sb && sa == sc);
    }
    CHECK(num_threads() == 1);
}

// ---------------------------------------------------------------------------------------------
// 场协议的失败与规则

static std::string field_request_text(const FieldCase& f, const fs::path& out) { return json::dump(fields_value(f, out)); }

static bool field_rejected(const std::string& request, const char* where, const char* part = "") {
    const rcwa::JsonReply reply = rcwa::solve_fields_json_reply(request);
    const json::Value v = parse_or_null(reply.text);
    const bool shape = !reply.ok && has_keys(v, {"ok", "error", "where"}) && v.at("ok").is_bool() && !v.at("ok").as_bool() &&
                       v.at("error").is_string() && v.at("where").is_string();
    const bool good = shape && v.at("where").as_string() == where && contains(v.at("error").as_string(), part);
    if (!good) say("    场请求被拒绝的方式不符（期望 where = %s，说明含“%s”），应答：%.300s\n", where, part, reply.text.c_str());
    return good;
}

// 失败的请求：ok:false，输出目录没有发布，父目录里没有临时目录残留
static bool field_failed_clean(const FieldCase& f, const char* where, const char* part) {
    const fs::path out = fresh_dir();
    const bool rejected_properly = field_rejected(field_request_text(f, out), where, part);
    const bool clean = !dir_exists(out) && !temporary_leftovers(out.parent_path());
    if (!clean) say("    失败的请求留下了目录：%s\n", out.u8string().c_str());
    return rejected_properly && clean;
}

static void test_field_failures_and_rules() {
    say("[场协议：失败即错误且不发布；output_dir、request_id、sampling 的规则]\n");
    // 求解失败：Rayleigh（入射区、出射区、第 j 层）
    for (int i = 0; i < kRcwa2dRefCaseCount; ++i) {
        const Rcwa2dRefCase& r = kRcwa2dRefCases[i];
        if (r.status == 0) continue;
        FieldCase f;
        f.solve = case_from_reference(r);
        f.z = {0.1, 0.2};
        f.nt = 4;
        f.at = 0.1;
        CHECK(field_failed_clean(f, "fields", r.failure_region));
        f.plane = rcwa::SlicePlane::YZ;
        CHECK(field_failed_clean(f, "fields", "Rayleigh"));
        // 失败例的错误信息也含 (m, n)
        const fs::path out = fresh_dir();
        const rcwa::JsonReply reply = rcwa::solve_fields_json_reply(field_request_text(f, out));
        const json::Value v = parse_or_null(reply.text);
        bool all = !reply.ok && v.is_object();
        for (int k = 0; all && k < r.failure_count; ++k) {
            char label[48];
            std::snprintf(label, sizeof label, "(%d, %d)", r.failure_orders[k][0], r.failure_orders[k][1]);
            all = all && contains(v.at("error").as_string(), label);
        }
        CHECK(all);
    }
    // 数值失败（LR 本征分解的残差检查）：找一个失败的算例
    {
        int found = 0;
        for (int i = 0; i < kRcwa2dRefCaseCount && found == 0; ++i) {
            const Rcwa2dRefCase& r = kRcwa2dRefCases[i];
            if (r.status != 0) continue;
            FieldCase f;
            f.solve = case_from_reference(r);
            f.solve.options.eig = EigMethod::LR;
            if (direct_solve(f.solve).status != rcwa::Status2D::NumericalFailure) continue;
            f.z = {0.1, 0.2};
            f.nt = 4;
            CHECK(field_failed_clean(f, "fields", "数值失败"));
            ++found;
        }
        CHECK(found == 1);
    }
    // 采样产生非有限值：z 或 at 大到溢出
    {
        FieldCase f = base_field_case();
        f.z = {0.1, 1e308};
        CHECK(field_failed_clean(f, "fields", ""));
        f.z = {-1e308};
        CHECK(field_failed_clean(f, "fields", ""));
        f = base_field_case();
        f.at = 1e308;
        CHECK(field_failed_clean(f, "fields", ""));
        f.plane = rcwa::SlicePlane::YZ;
        CHECK(field_failed_clean(f, "fields", ""));
    }

    // output_dir：已存在被拒绝（目录、文件），不动里面的东西；父目录不存在；空；NUL；根目录
    {
        FieldCase f = base_field_case();
        const fs::path existing = fresh_dir();
        fs::create_directories(existing);
        { std::ofstream(existing / "sentinel.txt") << "keep"; }
        CHECK(field_rejected(field_request_text(f, existing), "fields", "已存在"));
        CHECK(file_names(existing) == std::vector<std::string>{"sentinel.txt"});
        const fs::path file = fresh_dir();
        { std::ofstream(file) << "a file"; }
        CHECK(field_rejected(field_request_text(f, file), "fields", "已存在"));
        CHECK(field_rejected(field_request_text(f, g_root / "no_such_parent" / "sub"), "fields", "父目录"));
        CHECK(!dir_exists(g_root / "no_such_parent"));
        CHECK(field_rejected(field_request_text(f, g_root), "fields", "已存在"));
        json::Value v = fields_value(f, g_root / "x");
        v.set("output_dir", "");
        CHECK(field_rejected(json::dump(v), "output_dir", "不能为空"));
        v.set("output_dir", std::string("a\0b", 3));
        CHECK(field_rejected(json::dump(v), "output_dir", "NUL"));
        v.set("output_dir", "/");
        CHECK(field_rejected(json::dump(v), "output_dir", "子目录"));
        v.set("output_dir", 3);
        CHECK(field_rejected(json::dump(v), "output_dir", "字符串"));
        // 相对路径、带 .. 的路径：发布到规范化后的绝对路径，清单里写的就是它
        const fs::path nested = g_root / "a" / ".." / "b_dir";
        v.set("output_dir", nested.u8string());
        const rcwa::JsonReply reply = rcwa::solve_fields_json_reply(json::dump(v));
        CHECK(reply.ok);
        const json::Value rv = parse_or_null(reply.text);
        CHECK(rv.is_object() && rv.at("field_data").at("directory").as_string() == (g_root / "b_dir").u8string());
        CHECK(dir_exists(g_root / "b_dir") && file_names(g_root / "b_dir").size() == 7);
    }
    // 同一个请求重复提交：第二次被拒绝，第一次的结果原封不动
    {
        FieldCase f = base_field_case();
        const fs::path out = fresh_dir();
        const rcwa::JsonReply first = rcwa::solve_fields_json_reply(field_request_text(f, out));
        CHECK(first.ok);
        std::ifstream before(out / "fields.npy", std::ios::binary);
        const std::string content((std::istreambuf_iterator<char>(before)), std::istreambuf_iterator<char>());
        CHECK(field_rejected(field_request_text(f, out), "fields", "已存在"));
        std::ifstream after(out / "fields.npy", std::ios::binary);
        CHECK(content == std::string((std::istreambuf_iterator<char>(after)), std::istreambuf_iterator<char>()) && !temporary_leftovers(out.parent_path()));
    }

    // request_id：1 到 128 字节
    {
        FieldCase f = base_field_case();
        f.id = "";
        CHECK(field_failed_clean(f, "request_id", "128"));
        f.id = std::string(129, 'a');
        CHECK(field_failed_clean(f, "request_id", "128"));
        f.id = std::string(128, 'a');
        const fs::path out = fresh_dir();
        const rcwa::JsonReply ok128 = rcwa::solve_fields_json_reply(field_request_text(f, out));
        CHECK(ok128.ok && is_text(parse_or_null(ok128.text).at("field_data").at("request_id"), f.id.c_str()));
        f.id = "令牌-é-1";                                      // 多字节字符：按字节数计，这个只有 12 字节
        const fs::path out2 = fresh_dir();
        const rcwa::JsonReply uni = rcwa::solve_fields_json_reply(field_request_text(f, out2));
        CHECK(uni.ok && is_text(parse_or_null(uni.text).at("field_data").at("request_id"), f.id.c_str()));
        f.id.clear();
        for (int i = 0; i < 43; ++i) f.id += "令";                  // 43 个字 = 129 字节
        CHECK(f.id.size() == 129);
        CHECK(field_failed_clean(f, "request_id", "128"));
        json::Value v = fields_value(base_field_case(), fresh_dir());
        v.set("request_id", 5);
        CHECK(field_rejected(json::dump(v), "request_id", "字符串"));
    }

    // sampling 的各种错误；solve、output_dir 等顶层键
    {
        const FieldCase f = base_field_case();
        const fs::path out = fresh_dir();
        const json::Value good = fields_value(f, out);
        const auto with_sampling = [&](const char* key, json::Value value) {
            json::Value v = good;
            ref(v, "sampling").set(key, std::move(value));
            return json::dump(v);
        };
        const auto without_sampling = [&](const char* key) {
            json::Value v = json::Value::object(), s = json::Value::object();
            for (std::size_t i = 0; i < good.at("sampling").size(); ++i)
                if (good.at("sampling").key(i) != key) s.set(good.at("sampling").key(i), good.at("sampling").at(i));
            for (std::size_t i = 0; i < good.size(); ++i) v.set(good.key(i), good.key(i) == "sampling" ? s : good.at(i));
            return json::dump(v);
        };
        CHECK(field_rejected(with_sampling("nx", json::Value(8)), "sampling.nx", "不认识的键"));       // 一维的键
        CHECK(field_rejected(with_sampling("y", json::Value(0.0)), "sampling.y", "不认识的键"));
        CHECK(field_rejected(with_sampling("foo", json::Value(1)), "sampling.foo", "不认识的键"));
        CHECK(field_rejected(without_sampling("plane"), "sampling.plane", "缺少"));
        CHECK(field_rejected(without_sampling("nt"), "sampling.nt", "缺少"));
        CHECK(field_rejected(without_sampling("z"), "sampling.z", "缺少"));
        for (const char* bad : {"XZ", "xy", "zx", "", " xz", "xz "}) CHECK(field_rejected(with_sampling("plane", json::Value(bad)), "sampling.plane", "xz"));
        CHECK(field_rejected(with_sampling("plane", json::Value(1)), "sampling.plane", "字符串"));
        CHECK(field_rejected(with_sampling("plane", json::Value()), "sampling.plane", "字符串"));
        for (double bad : {0.0, -1.0, 2.5, 1e12}) CHECK(field_rejected(with_sampling("nt", json::Value(bad)), "sampling.nt", ""));
        CHECK(field_rejected(with_sampling("nt", json::Value(0)), "sampling.nt", "正整数"));
        CHECK(field_rejected(with_sampling("nt", json::Value("8")), "sampling.nt", "整数"));
        CHECK(field_rejected(with_sampling("nt", json::Value()), "sampling.nt", "整数"));
        CHECK(field_rejected(with_sampling("at", json::Value("x")), "sampling.at", "数"));
        CHECK(field_rejected(with_sampling("at", json::Value()), "sampling.at", "数"));
        CHECK(field_rejected(with_sampling("at", json::Value::array()), "sampling.at", "数"));
        CHECK(field_rejected(with_sampling("z", json::Value::array()), "sampling", "为空"));
        CHECK(field_rejected(with_sampling("z", json::Value(0.1)), "sampling.z", "数组"));
        {
            json::Value z = json::Value::array();
            z.push_back(0.1);
            z.push_back("a");
            CHECK(field_rejected(with_sampling("z", z), "sampling.z[1]", "数"));
            json::Value z2 = json::Value::array();
            z2.push_back(json::Value());
            CHECK(field_rejected(with_sampling("z", z2), "sampling.z[0]", "数"));
        }
        {   // nt·len(z) 的上限 4000000
            json::Value z = json::Value::array();
            for (int i = 0; i < 1000; ++i) z.push_back(0.1);
            json::Value v = good;
            ref(v, "sampling").set("z", z);
            ref(v, "sampling").set("nt", 4001);
            CHECK(field_rejected(json::dump(v), "sampling", "4000000"));
            ref(v, "sampling").set("nt", 4000001);
            ref(v, "sampling").set("z", numbers_value({0.1}));
            CHECK(field_rejected(json::dump(v), "sampling", "4000000"));
        }
        // 一维的 sampling 写法（nx、y）在二维里是未知键，反过来二维的写法在一维里也是
        {
            json::Value v = good;
            json::Value s = json::Value::object();
            s.set("z", numbers_value({0.1}));
            s.set("nx", 4);
            v.set("sampling", s);
            CHECK(field_rejected(json::dump(v), "sampling.nx", "不认识的键"));
        }
        // 顶层
        for (const char* key : {"solve", "sampling", "output_dir", "request_id"}) {
            json::Value v = json::Value::object();
            for (std::size_t i = 0; i < good.size(); ++i)
                if (good.key(i) != key) v.set(good.key(i), good.at(i));
            CHECK(field_rejected(json::dump(v), key, "缺少"));
        }
        {
            json::Value v = good;
            v.set("foo", 1);
            CHECK(field_rejected(json::dump(v), "foo", "不认识的键"));
        }
        {
            json::Value v = good;
            v.set("sampling", 3);
            CHECK(field_rejected(json::dump(v), "sampling", "对象"));
        }
        // solve：每次只接受一个入射点；二维必须有 input；solve 本身的错误照远场的规则
        {
            FieldCase two = f;
            two.solve.extra_points = {{1.1, 20.0, 5.0}};
            CHECK(field_rejected(field_request_text(two, fresh_dir()), "solve.points", "一个入射点"));
            json::Value v = good;
            json::Value solve = json::Value::object();
            for (std::size_t i = 0; i < good.at("solve").size(); ++i)
                if (good.at("solve").key(i) != "input") solve.set(good.at("solve").key(i), good.at("solve").at(i));
            v.set("solve", solve);
            CHECK(field_rejected(json::dump(v), "input", "缺少"));
            json::Value bad = good;
            ref(bad, "solve").set("My", -1);
            CHECK(field_rejected(json::dump(bad), "My", "负"));
            bad = good;
            ref(ref(bad, "solve"), "options").set("linear", "lu");
            CHECK(field_rejected(json::dump(bad), "options.linear", "二维不适用"));
            bad = good;
            ref(bad, "solve").set("M", 1024);
            CHECK(field_rejected(json::dump(bad), "M", "1023"));
        }
        CHECK(!dir_exists(out));
    }
    CHECK(num_threads() == 1);
    CHECK(!temporary_leftovers(g_root));
}

// 一维的场请求照常：二维的 sampling 键（plane、nt、at）在一维里是未知键
static void test_field_1d_unaffected() {
    say("[场协议：一维不受影响，二维的 sampling 键在一维里是未知键]\n");
    const char* const planar =
        R"({"solve": {"mode": "planar", "stack": {"nI": 1.0, "nII": 1.46, "period": 0.574, "layers": [{"d": 0.38, "edges": [0.0, 0.1722, 0.574], "n_values": [1.46, 1.0]}]},
 "M": 4, "polarization": "TE", "wavelength": 1.053, "theta_deg": 40.0}, "sampling": {"z": [0.0, 0.2], "nx": 4}, "output_dir": "OUT", "request_id": "r1"})";
    const auto with_dir = [&](const std::string& text, const fs::path& dir) { return edit(text, "OUT", dir.generic_u8string()); };
    {
        const fs::path out = fresh_dir();
        const rcwa::JsonReply reply = rcwa::solve_fields_json_reply(with_dir(planar, out));
        CHECK(reply.ok);
        const json::Value v = parse_or_null(reply.text);
        CHECK(v.is_object() && is_text(v.at("mode"), "planar") && !v.contains("My") && !v.at("field_data").contains("slice"));
        CHECK(v.is_object() && v.at("field_data").contains("y") && file_names(out).size() == 5);
    }
    CHECK(field_rejected(with_dir(edit(planar, "\"nx\": 4", "\"nx\": 4, \"y\": 0.3"), fresh_dir()), "sampling.y", "必须是 0"));       // planar 的 y 只能是 0
    CHECK(field_rejected(with_dir(edit(planar, "\"nx\": 4", "\"nx\": 4, \"plane\": \"xz\""), fresh_dir()), "sampling.plane", "不认识的键"));
    CHECK(field_rejected(with_dir(edit(planar, "\"nx\": 4", "\"nx\": 4, \"nt\": 4"), fresh_dir()), "sampling.nt", "不认识的键"));
    CHECK(field_rejected(with_dir(edit(planar, "\"nx\": 4", "\"nx\": 4, \"at\": 0.0"), fresh_dir()), "sampling.at", "不认识的键"));
    CHECK(field_rejected(with_dir(edit(planar, "\"nx\": 4", "\"plane\": \"xz\", \"nt\": 4"), fresh_dir()), "sampling.plane", "不认识的键"));
    CHECK(!g_bad_edit);
}

static void run(void (*test)(), const char* name) {
    try {
        test();
    } catch (const std::exception& e) {
        check(false, __FILE__, __LINE__, name);
        say("    %s 中途抛出了异常：%s\n", name, e.what());
    }
}

int main() {
    std::setlocale(LC_ALL, "C");
    const auto start = std::chrono::steady_clock::now();
    run(test_request_errors, "test_request_errors");
    run(test_far_field_basic, "test_far_field_basic");
    run(test_far_field_frozen, "test_far_field_frozen");
    run(test_far_field_failures, "test_far_field_failures");
    run(test_threads_and_reentrancy, "test_threads_and_reentrancy");
    run(test_field_slices, "test_field_slices");
    run(test_field_threads, "test_field_threads");
    run(test_field_failures_and_rules, "test_field_failures_and_rules");
    run(test_field_1d_unaffected, "test_field_1d_unaffected");
    set_num_threads(1);
    check(!g_bad_edit, __FILE__, __LINE__, "g_bad_edit");
    std::error_code ignored;
    if (!g_root.empty()) fs::remove_all(g_root, ignored);
    say("用时 %.1f 秒\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    say("通过 %d / 共 %d\n", g_pass, g_total);
    return g_pass == g_total ? 0 : 1;
}
