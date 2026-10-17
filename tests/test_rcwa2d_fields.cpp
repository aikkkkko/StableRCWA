// tests/test_rcwa2d_fields.cpp —— rcwa2d_fields.hpp（二维近场：场上下文、任意 z 的谐波列、x-z / y-z 切面、相位）的测试
//
//   c++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -I include -I tests tests/test_rcwa2d_fields.cpp -o build/test_rcwa2d_fields
//   c++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -I include -I tests tests/test_rcwa2d_fields.cpp -o build/test_rcwa2d_fields_san
//   build/test_rcwa2d_fields                 默认：全部检验，约半分钟
//   build/test_rcwa2d_fields --big           另加 N = 625 的算例（对冻结 NPZ 的切面与相位；约十几秒）
//   build/test_rcwa2d_fields --bench         另加 N = 289 与 N = 625 的 field_at / field_slice 用时与内存（可加 --threads=8）
//
// 全部检验自包含（不依赖 Python 在运行时）。参照与独立判据：
//   a) 谐波列对冻结的 Python 参照（tests/rcwa2d_reference.inc 的 Rcwa2dRefDepth，21 个成功算例，每例 2–3 个 z），逐复数 |a−b| ≤ atol + rtol|b|；
//   b) 切面与相位对冻结的 NPZ（tests/rcwa2d_fields_fixture.inc：xz、yz 切面的复场、绝对相位、相对入射相位、掩膜），相位用 CONTRACT.md 第 8 节的弦差判据；
//      另有 M1 = 60 的 Python 谐波列（Python 在 z ≈ −2 µm 处 0×∞ = NaN，这里在 z = −3 µm 处全部有限）；
//   c) 独立判据：层界面两侧切向 E、H（Ex、Ey、Hx、Hy、Hz）连续与 Eps·Ez 连续（法向 D）；Maxwell 方程的 z 分量与沿 z 的导数关系（有限差分）；
//      均匀多层的 Airy 解析场（本文件自己的传输线方程，不用 Python 输出）；y 不变 / x 不变结构对一维 C++ 的 solve_with_fields（TE、TM）与
//      solve_conical_with_fields（圆锥；x 不变按 x_1D = y、y_1D = −x、φ_1D = φ − 90° 映射）；无吸收时 Poynting 通量与 z 无关并等于 nI cosθ ΣDE；
//      吸收时通量不增；x↔y 互换结构只比模；
//   d) 语义：上下文拥有数据（Stack 先销毁再采样）、深拷贝、移动后 Empty、数值失败不可采样、无层栈 z = 0 取出射侧、零厚层、z 的顺序与重复、
//      边界 z 与 ±1e-12 带的归属、空 zs、批量分块、非法参数；
//   e) capture == nullptr 与 solve2d_with_fields 的 Result2D 逐位相同。
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

#include "stablercwa/rcwa1d_conical_fields.hpp"
#include "stablercwa/rcwa1d_fields.hpp"
#include "stablercwa/rcwa2d_fields.hpp"

using namespace stablercwa;
namespace rcwa = stablercwa::rcwa;

#include "rcwa2d_reference.inc"
#include "rcwa2d_fields_fixture.inc"

// ---------------------------------------------------------------------------------------------
// 输出与计数（做法同 tests/test_rcwa1d.cpp：文字一律经 say()）
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
static void check_le(double value, double bound, const char* file, int line, const char* text) {
    ++g_total;
    if (value <= bound) ++g_pass;
    else say("  失败 %s:%d  %s = %.3e > %.3e\n", file, line, text, value, bound);
}
static void check_throws(bool caught, const char* file, int line, const char* text, const char* type) {
    ++g_total;
    if (caught) ++g_pass;
    else say("  失败 %s:%d  %s 没有抛出 %s\n", file, line, text, type);
}
#define CHECK(cond) check(static_cast<bool>(cond), __FILE__, __LINE__, #cond)
#define CHECK_LE(value, bound)                              \
    do {                                                    \
        const double v_ = (value), b_ = (bound);            \
        check_le(v_, b_, __FILE__, __LINE__, #value);       \
    } while (0)
#define CHECK_THROWS(expr, exception_type)                                       \
    do {                                                                         \
        bool caught_ = false;                                                    \
        try { (void)(expr); } catch (const exception_type&) { caught_ = true; }  \
        check_throws(caught_, __FILE__, __LINE__, #expr, #exception_type);       \
    } while (0)
// std::logic_error 但不是 std::invalid_argument（上下文不可用）；std::invalid_argument 是 logic_error 的子类，要单独区分
#define CHECK_THROWS_UNAVAILABLE(expr)                                           \
    do {                                                                         \
        bool caught_ = false;                                                    \
        try { (void)(expr); } catch (const std::invalid_argument&) {} catch (const std::logic_error&) { caught_ = true; } \
        check_throws(caught_, __FILE__, __LINE__, #expr, "std::logic_error（非 invalid_argument）"); \
    } while (0)

static const double kNaN = std::numeric_limits<double>::quiet_NaN();

static bool g_big = false, g_bench = false;
static int g_threads = 1;

// ---------------------------------------------------------------------------------------------
// 比较工具
static bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; }
static bool same_bits(cplx a, cplx b) { return same_bits(a.real(), b.real()) && same_bits(a.imag(), b.imag()); }
static bool same_matrix(const Matrix& a, const Matrix& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    const std::size_t n = static_cast<std::size_t>(a.rows()) * static_cast<std::size_t>(a.cols());
    for (std::size_t k = 0; k < n; ++k)
        if (!same_bits(a.data()[k], b.data()[k])) return false;
    return true;
}
static bool same_harmonics(const rcwa::HarmonicFields& a, const rcwa::HarmonicFields& b) {
    if (a.orders != b.orders || a.z.size() != b.z.size()) return false;
    for (std::size_t k = 0; k < a.z.size(); ++k)
        if (!same_bits(a.z[k], b.z[k])) return false;
    for (std::size_t c = 0; c < 6; ++c)
        if (!same_matrix(a.components[c], b.components[c])) return false;
    return true;
}
// a 的第 ia 列与 b 的第 ib 列（六个分量）逐位相同
static bool same_column(const rcwa::HarmonicFields& a, std::size_t ia, const rcwa::HarmonicFields& b, std::size_t ib) {
    for (std::size_t c = 0; c < 6; ++c) {
        if (a.components[c].rows() != b.components[c].rows()) return false;
        for (int j = 0; j < a.components[c].rows(); ++j)
            if (!same_bits(a.components[c](j, static_cast<int>(ia)), b.components[c](j, static_cast<int>(ib)))) return false;
    }
    return true;
}
static bool same_vector(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t k = 0; k < a.size(); ++k)
        if (!same_bits(a[k], b[k])) return false;
    return true;
}
static bool same_vector(const RealVector& a, const RealVector& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t k = 0; k < a.size(); ++k)
        if (!same_bits(a[k], b[k])) return false;
    return true;
}
// Result2D 的全部数值与状态逐位相同（timing 除外）
static bool same_result(const rcwa::Result2D& a, const rcwa::Result2D& b) {
    return a.status == b.status && a.message == b.message && a.M1 == b.M1 && a.M2 == b.M2 && a.N == b.N && a.i0 == b.i0 && a.m == b.m &&
           a.n == b.n && same_bits(a.k0, b.k0) && same_bits(a.theta, b.theta) && same_bits(a.phi, b.phi) && same_vector(a.Kx, b.Kx) &&
           same_vector(a.Ky, b.Ky) && same_vector(a.kzI, b.kzI) && same_vector(a.kzII, b.kzII) && same_vector(a.Ex_r, b.Ex_r) &&
           same_vector(a.Ey_r, b.Ey_r) && same_vector(a.Ez_r, b.Ez_r) && same_vector(a.Ex_t, b.Ex_t) && same_vector(a.Ey_t, b.Ey_t) &&
           same_vector(a.Ez_t, b.Ez_t) && same_vector(a.Rs, b.Rs) && same_vector(a.Rp, b.Rp) && same_vector(a.Ts, b.Ts) &&
           same_vector(a.Tp, b.Tp) && same_vector(a.DEr, b.DEr) && same_vector(a.DEt, b.DEt) && a.prop_r == b.prop_r && a.prop_t == b.prop_t &&
           a.eig_info == b.eig_info && a.n_eig == b.n_eig && a.n_dense_solve == b.n_dense_solve && same_bits(a.max_eig_residual, b.max_eig_residual);
}

// 逐复数比较：max |a − b|，以及 |a − b| − (atol + rtol|b|) 的最大值（≤ 0 即通过）
struct Cmp {
    double maxdiff = 0.0, excess = -1e300;
    long n = 0;
    void add(cplx a, cplx ref, double atol, double rtol) {
        const double d = std::abs(a - ref);
        ++n;
        if (!(d <= maxdiff)) maxdiff = std::isnan(d) ? std::numeric_limits<double>::infinity() : d;
        const double ex = std::isnan(d) ? 1e300 : d - (atol + rtol * std::abs(ref));
        if (ex > excess) excess = ex;
    }
    void add(cplx a, const double b[2], double atol, double rtol) { add(a, cplx{b[0], b[1]}, atol, rtol); }
    bool ok() const { return excess <= 0.0; }
};

static double max_abs_harmonic(const rcwa::HarmonicFields& h, rcwa::FieldComponent c, std::size_t iz) {
    double best = 0.0;
    for (int j = 0; j < h.component(c).rows(); ++j) best = (std::max)(best, std::abs(h.component(c)(j, static_cast<int>(iz))));
    return best;
}

// ---------------------------------------------------------------------------------------------
// 求解问题与几何
struct Problem {
    rcwa::Stack2D stack;
    double wavelength = 1.053, theta = 42.0, phi = -25.0;
    int M1 = 2, M2 = 2;
    rcwa::Jones2D jones{cplx{1.0, 0.0}, cplx{0.0, 0.0}};
    rcwa::Options2D options;
};
static rcwa::FieldResult2D solve(const Problem& p) {
    return rcwa::solve2d_with_fields(p.stack, p.wavelength, p.theta, p.phi, p.M1, p.M2, p.jones, p.options);
}
static rcwa::Result2D solve_plain(const Problem& p) {
    return rcwa::solve2d(p.stack, p.wavelength, p.theta, p.phi, p.M1, p.M2, p.jones, p.options, nullptr);
}

static rcwa::XProfile2D profile(std::vector<double> edges, std::vector<cplx> n) { return rcwa::XProfile2D{std::move(edges), std::move(n)}; }

// 矩形柱：x ∈ [x0, x1]、y ∈ [y0, y1] 内是 n_in，其余 n_out（0 < x0 < x1 < Lx，0 < y0 < y1 < Ly）
static rcwa::Layer2D pillar_layer(double d, double Lx, double Ly, double x0, double x1, double y0, double y1, cplx n_in, cplx n_out) {
    const rcwa::XProfile2D bar = profile({0.0, Lx}, {n_out});
    return rcwa::Layer2D::patterned(d, {0.0, y0, y1, Ly}, {bar, profile({0.0, x0, x1, Lx}, {n_out, n_in, n_out}), bar});
}

// 多层几何 A：结构层 + 两层均匀膜，无吸收（连续性、通量、Maxwell 关系的主要对象）
static Problem problem_a() {
    Problem p;
    p.stack.nI = 1.0;
    p.stack.nII = 1.46;
    p.stack.Lx = 0.7;
    p.stack.Ly = 0.55;
    p.stack.layers.push_back(pillar_layer(0.3, 0.7, 0.55, 0.15, 0.45, 0.12, 0.38, 1.8, 1.0));
    p.stack.layers.push_back(rcwa::Layer2D::uniform(0.115, 1.96));
    p.stack.layers.push_back(rcwa::Layer2D::uniform(0.31, 1.46));
    p.theta = 40.0;
    p.phi = 18.0;
    p.M1 = 3;
    p.M2 = 3;
    p.jones = rcwa::jones_from_angles_2d(55.0, 35.0);
    return p;
}
// 几何 B：吸收（吸收柱、吸收膜、吸收基底）
static Problem problem_b() {
    Problem p;
    p.stack.nI = 1.2;
    p.stack.nII = cplx{1.5, 0.03};
    p.stack.Lx = 0.62;
    p.stack.Ly = 0.93;
    p.stack.layers.push_back(rcwa::Layer2D::uniform(0.1, cplx{1.65, 0.12}));
    p.stack.layers.push_back(pillar_layer(0.28, 0.62, 0.93, 0.2, 0.46, 0.3, 0.7, cplx{1.8, 0.13}, 1.0));
    p.stack.layers.push_back(rcwa::Layer2D::uniform(0.2, 1.9));
    p.wavelength = 0.83;
    p.theta = 27.0;
    p.phi = -61.0;
    p.M1 = 2;
    p.M2 = 3;
    p.jones = rcwa::jones_from_angles_2d(33.0, 50.0);
    return p;
}
// 几何 C：厚层——吸收结构层 5 µm、无吸收均匀层 3 µm（k0·d 达 30 以上）
static Problem problem_c() {
    Problem p;
    p.stack.nI = 1.0;
    p.stack.nII = 1.5;
    p.stack.Lx = 0.8;
    p.stack.Ly = 0.8;
    p.stack.layers.push_back(pillar_layer(5.0, 0.8, 0.8, 0.2, 0.55, 0.25, 0.6, cplx{1.7, 0.2}, cplx{1.0, 0.05}));
    p.stack.layers.push_back(rcwa::Layer2D::uniform(3.0, 1.46));
    p.stack.layers.push_back(pillar_layer(0.4, 0.8, 0.8, 0.1, 0.4, 0.3, 0.5, 2.1, 1.0));
    p.theta = 33.0;
    p.phi = 40.0;
    p.M1 = 2;
    p.M2 = 2;
    p.jones = rcwa::jones_from_angles_2d(70.0, 20.0);
    return p;
}
// 层界面位置（含 0 与 z_bot；零厚层给出重复值）
static RealVector interfaces_of(const rcwa::Stack2D& stack) {
    RealVector z{0.0};
    double acc = 0.0;
    for (const auto& layer : stack.layers) {
        acc += layer.d;
        z.push_back(acc);
    }
    return z;
}

// ---------------------------------------------------------------------------------------------
// a) 谐波列对冻结的 Python 参照
static rcwa::FieldResult2D solve_ref_case(const Rcwa2dRefCase& c, const rcwa::Options2D& base) {
    rcwa::Options2D options = base;
    options.factorization = c.laurent ? rcwa::Factorization2D::Laurent : rcwa::Factorization2D::Li;
    const rcwa::Jones2D jones = {cplx{c.jones[0], c.jones[1]}, cplx{c.jones[2], c.jones[3]}};
    return rcwa::solve2d_with_fields(rcwa2d_ref_stack(c.id), c.wavelength, c.theta_deg, c.phi_deg, c.M1, c.M2, jones, options);
}

static double compare_ref_harmonics(const Rcwa2dRefCase& c, const rcwa::FieldResult2D& run, bool verbose) {
    Cmp cmp;
    RealVector zs;
    for (int k = 0; k < c.depth_count; ++k) zs.push_back(c.depths[k].z);
    const rcwa::HarmonicFields h = rcwa::field_at(run, zs);
    CHECK(h.nz() == zs.size() && h.norders() == static_cast<std::size_t>(c.count));
    CHECK(h.z == zs);
    for (int k = 0; k < c.depth_count; ++k) {
        // 单个 z 的入口与批量的对应列逐位相同
        const rcwa::HarmonicFields single = rcwa::field_at(run, zs[static_cast<std::size_t>(k)]);
        CHECK(single.nz() == 1 && same_column(single, 0, h, static_cast<std::size_t>(k)));
        for (int i = 0; i < c.count; ++i)
            for (int comp = 0; comp < 6; ++comp)
                cmp.add(h.components[static_cast<std::size_t>(comp)](i, k), c.depths[k].columns[i].c[comp], c.near_atol, c.near_rtol);
    }
    CHECK(cmp.ok());
    if (verbose || !cmp.ok())
        say("  %2d %-16s N=%3d  %d 个 z  最大差 %.2e（%ld 个复数）%s\n", c.id, c.name, c.count, c.depth_count, cmp.maxdiff, cmp.n,
            cmp.ok() ? "" : "  超出容差");
    return cmp.maxdiff;
}

static void test_frozen_harmonics() {
    say("[谐波列对冻结的 Python 参照：%d 个算例中的成功算例]\n", kRcwa2dRefCaseCount);
    struct Variant { const char* name; rcwa::Options2D options; bool verbose; };
    std::vector<Variant> variants;
    variants.push_back({"默认选项（QR 本征分解、均匀层快路径、串行）", rcwa::Options2D{}, true});
    {
        rcwa::Options2D o;
        o.homogeneous_fast_path = false;
        variants.push_back({"homogeneous_fast_path = false（均匀层按稠密层：稠密 W、V、Ei 的路径）", o, false});
    }
    {
        rcwa::Options2D o;
        o.parallel_layers = true;
        variants.push_back({"parallel_layers = true", o, false});
    }
    for (const Variant& v : variants) {
        say(" %s\n", v.name);
        double worst = 0.0;
        int count = 0;
        for (int i = 0; i < kRcwa2dRefCaseCount; ++i) {
            const Rcwa2dRefCase& c = kRcwa2dRefCases[i];
            if (c.status != 0) continue;
            const rcwa::FieldResult2D run = solve_ref_case(c, v.options);
            CHECK(run.result.status == rcwa::Status2D::Ok && run.fields.available() && run.fields.status() == rcwa::FieldContextStatus::Ready);
            if (!run.fields.available()) continue;
            worst = (std::max)(worst, compare_ref_harmonics(c, run, v.verbose));
            ++count;
        }
        say("  %d 个算例，最大差 %.2e\n", count, worst);
    }
}

// ---------------------------------------------------------------------------------------------
// b) 切面与相位对冻结的 NPZ
struct PhaseStats {
    long compared = 0, mask_differences = 0;
    double worst_ratio = 0.0;   // 弦差 / 容许量
    double worst_chord = 0.0;   // 弦差本身
};

static void compare_phase(const rcwa::PhaseGrid& grid, const double* frozen_phase, const char* frozen_mask, const char* comparison,
                          const Rcwa2dFieldsSlice& ref, std::size_t comp, std::size_t nz, std::size_t nt, double atol, double rtol,
                          PhaseStats& stats) {
    CHECK(grid.nz == nz && grid.nx == nt);
    if (grid.nz != nz || grid.nx != nt) return;
    for (std::size_t iz = 0; iz < nz; ++iz)
        for (std::size_t it = 0; it < nt; ++it) {
            const std::size_t k = (comp * nz + iz) * nt + it;
            const bool frozen_masked = frozen_mask[k] == '1';
            if (grid.mask[iz * nt + it] != frozen_masked) ++stats.mask_differences;
            if (comparison[k] == '1') continue;
            // 比较点：冻结相位有定义，C++ 的掩膜必须也是“有定义”
            CHECK(!grid.mask[iz * nt + it]);
            if (grid.mask[iz * nt + it]) continue;
            const double ar = ref.fields[2 * k], ai = ref.fields[2 * k + 1];
            const double amp = std::hypot(ar, ai);
            const double chord = std::abs(std::polar(1.0, grid.phase[iz * nt + it]) - std::polar(1.0, frozen_phase[k]));
            const double tolerance = 2.0 * (atol + rtol * amp) / amp + 5e-14;
            ++stats.compared;
            stats.worst_ratio = (std::max)(stats.worst_ratio, chord / tolerance);
            stats.worst_chord = (std::max)(stats.worst_chord, chord);
            CHECK(chord <= tolerance);
        }
}

static void check_slice_against_fixture(const Rcwa2dFieldsCase& fc, const rcwa::FieldResult2D& run, rcwa::SlicePlane plane, double period,
                                        double& worst_field, PhaseStats& stats_abs, PhaseStats& stats_inc) {
    const Rcwa2dFieldsSlice& ref = plane == rcwa::SlicePlane::XZ ? fc.xz : fc.yz;
    const std::size_t nz = static_cast<std::size_t>(fc.nz), nt = static_cast<std::size_t>(fc.nt);
    RealVector zs(fc.z, fc.z + fc.nz);
    const rcwa::FieldGridSlice slice = rcwa::field_slice(run, zs, plane, fc.nt, ref.fixed);
    CHECK(slice.plane == plane && slice.nt() == nt && slice.nz() == nz && slice.slice_at == ref.fixed && slice.z == zs);
    for (std::size_t it = 0; it < nt; ++it) {                       // 坐标：t_j = j L / nt（NPZ 是 arange(nt)·L/nt）
        CHECK(std::abs(slice.t[it] - static_cast<double>(it) * period / static_cast<double>(fc.nt)) <= 2e-16 * period);
        CHECK(slice.x[it] == (plane == rcwa::SlicePlane::XZ ? slice.t[it] : ref.fixed));
        CHECK(slice.y[it] == (plane == rcwa::SlicePlane::XZ ? ref.fixed : slice.t[it]));
    }
    Cmp cmp;
    for (std::size_t comp = 0; comp < 6; ++comp)
        for (std::size_t iz = 0; iz < nz; ++iz)
            for (std::size_t it = 0; it < nt; ++it) {
                const std::size_t k = (comp * nz + iz) * nt + it;
                cmp.add(slice.at(static_cast<rcwa::FieldComponent>(comp), iz, it), cplx{ref.fields[2 * k], ref.fields[2 * k + 1]}, fc.near_atol,
                        fc.near_rtol);
            }
    CHECK(cmp.ok());
    worst_field = (std::max)(worst_field, cmp.maxdiff);
    // 相位：C++ 用自己的复场、自己的掩膜与载波算出来，与冻结的 numpy 相位比较（phase_comparison_mask 为真的点跳过）
    for (std::size_t comp = 0; comp < 6; ++comp) {
        rcwa::PhaseOptions absolute, incident;
        incident.reference = rcwa::PhaseReference::Incident;
        const auto component = static_cast<rcwa::FieldComponent>(comp);
        const rcwa::PhaseGrid pa = rcwa::phase_data(slice, component, absolute), pi = rcwa::phase_data(slice, component, incident);
        CHECK(pa.plane == plane && pi.plane == plane);
        // phase_slice 便捷入口与“先 field_slice 再 phase_data”逐位相同
        const rcwa::PhaseGrid pc = rcwa::phase_slice(run, zs, plane, fc.nt, ref.fixed, component, incident);
        CHECK(pc.phase.size() == pi.phase.size() && pc.mask == pi.mask);
        bool same = pc.phase.size() == pi.phase.size();
        for (std::size_t k = 0; same && k < pc.phase.size(); ++k) same = same_bits(pc.phase[k], pi.phase[k]);
        CHECK(same);
        compare_phase(pa, ref.phase_absolute, ref.phase_mask, ref.comparison_mask, ref, comp, nz, nt, fc.near_atol, fc.near_rtol, stats_abs);
        compare_phase(pi, ref.phase_incident, ref.phase_mask, ref.comparison_mask, ref, comp, nz, nt, fc.near_atol, fc.near_rtol, stats_inc);
    }
}

static void test_frozen_slices() {
    say("[切面与相位对冻结的 NPZ（xz、yz 复场；绝对相位与相对入射相位；掩膜）]\n");
    for (int i = 0; i < kRcwa2dFieldsCaseCount; ++i) {
        const Rcwa2dFieldsCase& fc = kRcwa2dFieldsCases[i];
        const bool big = fc.id == 22;
        if (big && !g_big) continue;
        Rcwa2dRefCase c{};
        rcwa::Stack2D stack;
        rcwa::Jones2D jones{cplx{1.0, 0.0}, cplx{0.0, 0.0}};
        double wavelength, theta, phi;
        int M1, M2;
        rcwa::Options2D options;
        if (big) {        // 案例 22：案例 9 的几何，M1 = M2 = 12（不在 .inc 的参照表里）
            c = rcwa2d_ref_case(9);
            stack = rcwa2d_ref_stack(9);
            wavelength = c.wavelength; theta = c.theta_deg; phi = c.phi_deg; M1 = 12; M2 = 12;
        } else {
            c = rcwa2d_ref_case(fc.id);
            stack = rcwa2d_ref_stack(fc.id);
            jones = {cplx{c.jones[0], c.jones[1]}, cplx{c.jones[2], c.jones[3]}};
            wavelength = c.wavelength; theta = c.theta_deg; phi = c.phi_deg; M1 = c.M1; M2 = c.M2;
            options.factorization = c.laurent ? rcwa::Factorization2D::Laurent : rcwa::Factorization2D::Li;
        }
        const rcwa::FieldResult2D run = rcwa::solve2d_with_fields(stack, wavelength, theta, phi, M1, M2, jones, options);
        CHECK(run.fields.available());
        if (!run.fields.available()) continue;
        double worst_field = 0.0;
        PhaseStats stats_abs, stats_inc;
        check_slice_against_fixture(fc, run, rcwa::SlicePlane::XZ, stack.Lx, worst_field, stats_abs, stats_inc);
        check_slice_against_fixture(fc, run, rcwa::SlicePlane::YZ, stack.Ly, worst_field, stats_abs, stats_inc);
        CHECK(stats_abs.mask_differences == 0 && stats_inc.mask_differences == 0);
        say("  %2d %-14s N=%3d  nz=%2d nt=%2d  复场最大差 %.2e；相位比较点 绝对 %ld / 相对入射 %ld，弦差最大 %.1e / %.1e（占容许量 %.1e / %.1e），掩膜不一致 %ld\n",
            fc.id, fc.name, run.result.N, fc.nz, fc.nt, worst_field, stats_abs.compared, stats_inc.compared, stats_abs.worst_chord, stats_inc.worst_chord,
            stats_abs.worst_ratio, stats_inc.worst_ratio, stats_abs.mask_differences + stats_inc.mask_differences);
    }
}

// M1 = 60 的 y 不变光栅：区域 I 的入射项只对 (0,0) 求增长指数。Python 的 field_at 在 z ≈ −2 µm 处 0×∞ = NaN。
static void test_large_order_region_I() {
    say("[M1 = 60, M2 = 0（N = 121）：区域 I 深处没有 0×∞]\n");
    Problem p;
    p.stack = rcwa2d_ref_stack(3);
    p.wavelength = 1.053;
    p.theta = 66.36;
    p.phi = 0.0;
    p.M1 = 60;
    p.M2 = 0;
    const rcwa::FieldResult2D run = solve(p);
    CHECK(run.fields.available() && run.result.N == 121);
    const RealVector zs{-3.0, -2.0, -1.5, -0.9, -0.25, 0.19, 0.6, 5.0, -50.0};
    const rcwa::HarmonicFields h = rcwa::field_at(run, zs);
    bool finite = true;
    for (const Matrix& m : h.components) finite = finite && m.all_finite();
    CHECK(finite);
    // 与 Python 一致（Python 在这些深度还没有溢出）
    const RealVector python_z(kRcwa2dFieldsLargeMZ, kRcwa2dFieldsLargeMZ + 3);
    const rcwa::HarmonicFields hp = rcwa::field_at(run, python_z);
    Cmp cmp;
    for (std::size_t k = 0; k < 3; ++k)
        for (std::size_t comp = 0; comp < 6; ++comp)
            for (int j = 0; j < 121; ++j) {
                const std::size_t at = ((k * 6 + comp) * 121 + static_cast<std::size_t>(j)) * 2;
                cmp.add(hp.components[comp](j, static_cast<int>(k)), cplx{kRcwa2dFieldsLargeMH[at], kRcwa2dFieldsLargeMH[at + 1]}, 1e-8, 1e-8);
            }
    CHECK(cmp.ok());
    say("  对 Python（z = −0.9、−0.25 在区域 I，0.19 在层内）：最大差 %.2e（容差 1e-8；%ld 个复数）\n", cmp.maxdiff, cmp.n);
    // 区域 I 的横向电场 = 入射（只有零阶）+ 反射：用远场数组独立写出（Ex_r、Ey_r 是各级反射电场的 Ex、Ey 分量），包括 z = −3、−50 的深处
    double worst = 0.0;
    const rcwa::Result2D& r = run.result;
    const Vector& kzI = run.fields.metadata().kzI;
    for (std::size_t k = 0; k < zs.size(); ++k) {
        if (zs[k] >= 0.0) continue;
        for (int j = 0; j < r.N; ++j) {
            const std::size_t u = static_cast<std::size_t>(j);
            const cplx up = std::exp(-kI * kzI[u] * zs[k]);                                // 反射波 e^{−i kz z}（z < 0 时衰减或纯相位）
            cplx ex = r.Ex_r[u] * up, ey = r.Ey_r[u] * up;
            if (j == r.i0) {            // 入射零阶：s 偏振 φ = 0 → 入射电场只有 Ey = 1，e^{+i kz z}
                ey += std::exp(kI * kzI[u] * zs[k]);
            }
            worst = (std::max)({worst, std::abs(h.component(rcwa::FieldComponent::Ex)(j, static_cast<int>(k)) - ex),
                                std::abs(h.component(rcwa::FieldComponent::Ey)(j, static_cast<int>(k)) - ey)});
        }
    }
    CHECK_LE(worst, 1e-11);
    say("  z = −3、−2、−1.5、−0.9、−0.25、−50 全部有限；与远场数组独立写出的反射波叠加（含 z = −50）最大差 %.2e\n", worst);
}

// ---------------------------------------------------------------------------------------------
// c1) 界面连续、Maxwell 关系
// 界面两侧（z_i ∓ 1e-9）的 Ex、Ey、Hx、Hy、Hz 谐波列相对差（相对该分量上侧最大谐波），以及 Eps·Ez 的相对差（法向 D）
struct ContinuityResult { double tangential = 0.0, normal_d = 0.0; int interfaces = 0; };

static Matrix eps_of_medium(const Problem& p, int medium) {   // 介质 −1：入射区；0…L−1：层；L：出射区
    const int N = (2 * p.M1 + 1) * (2 * p.M2 + 1);
    const int L = static_cast<int>(p.stack.layers.size());
    if (medium == -1) return Matrix::identity(N) * cplx{p.stack.nI * p.stack.nI, 0.0};
    if (medium == L) return Matrix::identity(N) * (p.stack.nII * p.stack.nII);
    return rcwa::detail2d::layer_operators(p.stack.layers[static_cast<std::size_t>(medium)], p.stack.Lx, p.stack.Ly, p.M1, p.M2,
                                           p.options.factorization).eps;
}
// z 所在的介质（本文件自己写的一份归属规则：外部优先，其次第一个满足 z ≤ z_top + d + 1e-12 的层；无层时 z = 0 取出射区）
static int medium_at(const rcwa::Stack2D& stack, double z) {
    const int L = static_cast<int>(stack.layers.size());
    double top = 0.0;
    for (const auto& layer : stack.layers) top += layer.d;
    if (z < 0.0) return -1;
    if (z > top) return L;
    double z_top = 0.0;
    for (int j = 0; j < L; ++j) {
        if (z <= z_top + stack.layers[static_cast<std::size_t>(j)].d + 1e-12) return j;
        z_top += stack.layers[static_cast<std::size_t>(j)].d;
    }
    return L;
}

static ContinuityResult interface_continuity(const Problem& p, const rcwa::FieldResult2D& run, double step = 1e-9) {
    ContinuityResult out;
    const RealVector zi = interfaces_of(p.stack);
    for (std::size_t i = 0; i < zi.size(); ++i) {
        const rcwa::HarmonicFields h = rcwa::field_at(run, RealVector{zi[i] - step, zi[i] + step});
        for (const rcwa::FieldComponent c : {rcwa::FieldComponent::Ex, rcwa::FieldComponent::Ey, rcwa::FieldComponent::Hx,
                                             rcwa::FieldComponent::Hy, rcwa::FieldComponent::Hz}) {
            const double scale = (std::max)(max_abs_harmonic(h, c, 0), 1e-30);
            double diff = 0.0;
            for (int j = 0; j < h.component(c).rows(); ++j) diff = (std::max)(diff, std::abs(h.component(c)(j, 0) - h.component(c)(j, 1)));
            out.tangential = (std::max)(out.tangential, diff / scale);
        }
        // 法向 D：上侧介质（i − 1，i = 0 时为入射区）在 z_i − step 的 Eps·Ez 与下侧介质在 z_i + step 的 Eps·Ez
        const Matrix up_eps = eps_of_medium(p, medium_at(p.stack, zi[i] - step));
        const Matrix down_eps = eps_of_medium(p, medium_at(p.stack, zi[i] + step));
        const Vector ez_up = h.component(rcwa::FieldComponent::Ez).column(0), ez_down = h.component(rcwa::FieldComponent::Ez).column(1);
        const Vector d_up = up_eps * ez_up, d_down = down_eps * ez_down;
        double scale = 1e-30, diff = 0.0;
        for (std::size_t j = 0; j < d_up.size(); ++j) {
            scale = (std::max)(scale, std::abs(d_up[j]));
            diff = (std::max)(diff, std::abs(d_up[j] - d_down[j]));
        }
        out.normal_d = (std::max)(out.normal_d, diff / scale);
        ++out.interfaces;
    }
    return out;
}

// 单个 z 上的代数关系：Hz = Ky·Ex − Kx·Ey（Faraday 的 z 分量），Eps·Ez = Kx·Hy − Ky·Hx（Ampère 的 z 分量）。
// eps = 该点所在介质的 Eps（N×N）。返回最大相对差。
static double maxwell_z_relations(const rcwa::FieldResult2D& run, const Matrix& eps, double z, double* faraday = nullptr, double* ampere = nullptr) {
    const rcwa::HarmonicFields h = rcwa::field_at(run, z);
    const rcwa::Result2D& r = run.result;
    const int N = r.N;
    Vector hz_expected(static_cast<std::size_t>(N)), d_expected(static_cast<std::size_t>(N)), ez = h.component(rcwa::FieldComponent::Ez).column(0);
    double hz_scale = 1e-30, d_scale = 1e-30, hz_diff = 0.0, d_diff = 0.0;
    for (int j = 0; j < N; ++j) {
        const std::size_t k = static_cast<std::size_t>(j);
        const cplx Ex = h.component(rcwa::FieldComponent::Ex)(j, 0), Ey = h.component(rcwa::FieldComponent::Ey)(j, 0);
        const cplx Hx = h.component(rcwa::FieldComponent::Hx)(j, 0), Hy = h.component(rcwa::FieldComponent::Hy)(j, 0);
        hz_expected[k] = r.Ky[k] * Ex - r.Kx[k] * Ey;
        d_expected[k] = r.Kx[k] * Hy - r.Ky[k] * Hx;
        hz_scale = (std::max)(hz_scale, std::abs(hz_expected[k]));
        d_scale = (std::max)(d_scale, std::abs(d_expected[k]));
        hz_diff = (std::max)(hz_diff, std::abs(h.component(rcwa::FieldComponent::Hz)(j, 0) - hz_expected[k]));
    }
    const Vector d = eps * ez;
    for (int j = 0; j < N; ++j) d_diff = (std::max)(d_diff, std::abs(d[static_cast<std::size_t>(j)] - d_expected[static_cast<std::size_t>(j)]));
    if (faraday) *faraday = (std::max)(*faraday, hz_diff / hz_scale);
    if (ampere) *ampere = (std::max)(*ampere, d_diff / d_scale);
    return (std::max)(hz_diff / hz_scale, d_diff / d_scale);
}

// 沿 z 的 Faraday 方程（四阶中心差分）：∂z Ex = i k0 (Hy − Kx Ez)，∂z Ey = −i k0 (Hx + Ky Ez)。z 离界面至少 2h。
// scalar_eps != nullptr（均匀介质）时再检验 Ampère 的横向方程：∂z Hy = i k0 (ε Ex − Ky Hz)，∂z Hx = −i k0 (ε Ey + Kx Hz)。
static double maxwell_z_derivatives(const rcwa::FieldResult2D& run, double z, const cplx* scalar_eps, double h_step = 1e-3) {
    const rcwa::HarmonicFields h = rcwa::field_at(run, RealVector{z - 2 * h_step, z - h_step, z, z + h_step, z + 2 * h_step});
    const rcwa::Result2D& r = run.result;
    const double k0 = r.k0;
    double worst = 0.0;
    const auto derivative = [&](rcwa::FieldComponent c, int j) {
        const auto& m = h.component(c);
        return (-m(j, 4) + 8.0 * m(j, 3) - 8.0 * m(j, 1) + m(j, 0)) / (12.0 * h_step);
    };
    double scale_e = 1e-30, scale_h = 1e-30;
    for (int j = 0; j < r.N; ++j) {
        scale_e = (std::max)({scale_e, std::abs(h.component(rcwa::FieldComponent::Ex)(j, 2)), std::abs(h.component(rcwa::FieldComponent::Ey)(j, 2))});
        scale_h = (std::max)({scale_h, std::abs(h.component(rcwa::FieldComponent::Hx)(j, 2)), std::abs(h.component(rcwa::FieldComponent::Hy)(j, 2))});
    }
    const double scale = (std::max)(scale_e, scale_h) * k0;
    for (int j = 0; j < r.N; ++j) {
        const std::size_t k = static_cast<std::size_t>(j);
        const cplx Ex = h.component(rcwa::FieldComponent::Ex)(j, 2), Ey = h.component(rcwa::FieldComponent::Ey)(j, 2);
        const cplx Ez = h.component(rcwa::FieldComponent::Ez)(j, 2), Hx = h.component(rcwa::FieldComponent::Hx)(j, 2);
        const cplx Hy = h.component(rcwa::FieldComponent::Hy)(j, 2), Hz = h.component(rcwa::FieldComponent::Hz)(j, 2);
        worst = (std::max)(worst, std::abs(derivative(rcwa::FieldComponent::Ex, j) - kI * k0 * (Hy - r.Kx[k] * Ez)) / scale);
        worst = (std::max)(worst, std::abs(derivative(rcwa::FieldComponent::Ey, j) + kI * k0 * (Hx + r.Ky[k] * Ez)) / scale);
        if (scalar_eps) {
            worst = (std::max)(worst, std::abs(derivative(rcwa::FieldComponent::Hy, j) - kI * k0 * (*scalar_eps * Ex - r.Ky[k] * Hz)) / scale);
            worst = (std::max)(worst, std::abs(derivative(rcwa::FieldComponent::Hx, j) + kI * k0 * (*scalar_eps * Ey + r.Kx[k] * Hz)) / scale);
        }
    }
    return worst;
}

static void test_continuity_and_maxwell() {
    say("[层界面连续与 Maxwell 关系（独立判据）]\n");
    struct Named { const char* name; Problem problem; };
    std::vector<Named> problems{{"A 结构层 + 两层膜（无吸收，椭圆偏振）", problem_a()}, {"B 吸收柱 + 吸收膜 + 吸收基底，Lx≠Ly", problem_b()},
                                {"C 厚层（吸收结构层 5 µm、均匀层 3 µm）", problem_c()}};
    {   // A 的变体：均匀层按稠密层处理；Laurent
        Problem q = problem_a();
        q.options.homogeneous_fast_path = false;
        problems.push_back({"A（homogeneous_fast_path = false）", q});
        Problem l = problem_a();
        l.options.factorization = rcwa::Factorization2D::Laurent;
        problems.push_back({"A（Laurent 因式分解）", l});
        Problem z = problem_a();                 // 零厚层夹在中间
        z.stack.layers.insert(z.stack.layers.begin() + 2, rcwa::Layer2D::uniform(0.0, 2.3));
        problems.push_back({"A 加一层零厚层", z});
    }
    for (const Named& n : problems) {
        const rcwa::FieldResult2D run = solve(n.problem);
        CHECK(run.fields.available());
        if (!run.fields.available()) continue;
        const ContinuityResult cr = interface_continuity(n.problem, run);
        CHECK_LE(cr.tangential, 1e-7);
        CHECK_LE(cr.normal_d, 1e-7);
        // 代数关系：每个区域和每一层里取几个 z
        double faraday = 0.0, ampere = 0.0, derivative = 0.0;
        const RealVector zi = interfaces_of(n.problem.stack);
        const int L = static_cast<int>(n.problem.stack.layers.size());
        std::vector<std::pair<int, double>> points{{-1, -0.4}, {-1, -0.013}, {L, zi.back() + 0.02}, {L, zi.back() + 0.5}};
        for (int j = 0; j < L; ++j) {
            const double d = n.problem.stack.layers[static_cast<std::size_t>(j)].d;
            if (d <= 0.0) continue;
            points.push_back({j, zi[static_cast<std::size_t>(j)] + 0.17 * d});
            points.push_back({j, zi[static_cast<std::size_t>(j)] + 0.83 * d});
        }
        for (const auto& pt : points) {
            maxwell_z_relations(run, eps_of_medium(n.problem, pt.first), pt.second, &faraday, &ampere);
            // 沿 z 的导数：点离所在介质的边界至少 2h（层太薄的跳过）
            const double lo = pt.first < 0 ? -1e300 : zi[static_cast<std::size_t>(pt.first)], hi = pt.first == L || pt.first < 0
                                                                                              ? (pt.first < 0 ? 0.0 : 1e300)
                                                                                              : zi[static_cast<std::size_t>(pt.first) + 1];
            if (pt.second - 2e-3 <= lo || pt.second + 2e-3 >= hi) continue;
            cplx eps{};
            const cplx* scalar = nullptr;
            if (pt.first < 0) { eps = n.problem.stack.nI * n.problem.stack.nI; scalar = &eps; }
            else if (pt.first == L) { eps = n.problem.stack.nII * n.problem.stack.nII; scalar = &eps; }
            else if (n.problem.stack.layers[static_cast<std::size_t>(pt.first)].homogeneous) {
                eps = n.problem.stack.layers[static_cast<std::size_t>(pt.first)].n * n.problem.stack.layers[static_cast<std::size_t>(pt.first)].n;
                scalar = &eps;
            }
            derivative = (std::max)(derivative, maxwell_z_derivatives(run, pt.second, scalar));
        }
        CHECK_LE(faraday, 1e-12);
        CHECK_LE(ampere, 1e-9);
        CHECK_LE(derivative, 1e-7);
        say("  %-44s 界面 %d 个：切向 %.2e，Eps·Ez %.2e；Hz = Ky·Ex − Kx·Ey 相对差 %.2e，Eps·Ez = Kx·Hy − Ky·Hx %.2e；∂z 关系（有限差分）%.2e\n", n.name,
            cr.interfaces, cr.tangential, cr.normal_d, faraday, ampere, derivative);
    }
}

// ---------------------------------------------------------------------------------------------
// c2) 均匀多层的 Airy 解析场（本文件自己的传输线方程）
static Vector solve_linear(std::vector<Vector> A, Vector b) {
    const std::size_t n = b.size();
    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot = col;
        for (std::size_t r = col + 1; r < n; ++r)
            if (std::abs(A[r][col]) > std::abs(A[pivot][col])) pivot = r;
        std::swap(A[col], A[pivot]);
        std::swap(b[col], b[pivot]);
        for (std::size_t r = col + 1; r < n; ++r) {
            const cplx f = A[r][col] / A[col][col];
            for (std::size_t c = col; c < n; ++c) A[r][c] -= f * A[col][c];
            b[r] -= f * b[col];
        }
    }
    Vector x(n);
    for (std::size_t i = n; i-- > 0;) {
        cplx s = b[i];
        for (std::size_t c = i + 1; c < n; ++c) s -= A[i][c] * x[c];
        x[i] = s / A[i][i];
    }
    return x;
}

// 均匀分层介质：入射 nI（实）、各层 (d, n)、出射 nII；零阶 (0,0) 的六个分量。
// 每个界面写 S（TE：E⊥，TM：Z0 H⊥）与 U（i·P·(下行 − 上行)）连续，振幅锚定层顶（下行波）与层底（上行波）、区域 II 锚定 z_bot，
// 所以所有指数都 ≤ 1，厚吸收层也稳定（不同于冻结 Python 测试里 b 锚在层顶的写法）。
struct AiryModel {
    double nI = 1.0, k0 = 1.0, kt = 0.0, sin_phi = 0.0, cos_phi = 1.0;
    cplx nII = 1.0;
    std::vector<std::pair<double, cplx>> layers;
    rcwa::Jones2D jones{cplx{1.0, 0.0}, cplx{0.0, 0.0}};
    std::vector<cplx> kz, index;             // 介质 0 = 入射区，1…L = 各层，L+1 = 出射区；kz 以 k0 为单位
    std::vector<double> top;                 // 各层顶面（长 L + 1，最后是 z_bot）
    Vector te, tm;                           // 线性方程组的解 [r, a1, b1, …, t]

    static cplx kz_of(cplx n, double kt_) {
        cplx k = std::sqrt(n * n - kt_ * kt_);
        if (k.imag() < 0.0) k = -k;
        return k;
    }
    AiryModel(double nI_, cplx nII_, std::vector<std::pair<double, cplx>> layers_, double wavelength, double theta_deg, double phi_deg,
              rcwa::Jones2D jones_)
        : nI(nI_), k0(2.0 * kPi / wavelength), nII(nII_), layers(std::move(layers_)), jones(jones_) {
        kt = nI * std::sin(theta_deg * kPi / 180.0);
        sin_phi = std::sin(phi_deg * kPi / 180.0);
        cos_phi = std::cos(phi_deg * kPi / 180.0);
        index.push_back(nI);
        for (const auto& l : layers) index.push_back(l.second);
        index.push_back(nII);
        for (cplx n : index) kz.push_back(kz_of(n, kt));
        top.push_back(0.0);
        for (const auto& l : layers) top.push_back(top.back() + l.first);
        te = solve(false);
        tm = solve(true);
    }
    cplx admittance(std::size_t medium, bool TM) const { return kz[medium] / (TM ? index[medium] * index[medium] : cplx{1.0, 0.0}); }
    cplx phase(std::size_t medium, double length) const { return std::exp(kI * k0 * kz[medium] * length); }

    Vector solve(bool TM) const {
        const std::size_t L = layers.size(), n = 2 * L + 2;
        std::vector<Vector> A(n, Vector(n));
        Vector rhs(n);
        for (std::size_t i = 0; i <= L; ++i) {
            const std::size_t rS = 2 * i, rU = 2 * i + 1;
            if (i == 0) {                      // 上侧：S = 1 + r，U = i·P0·(1 − r)
                const cplx p0 = admittance(0, TM);
                A[rS][0] += 1.0; rhs[rS] -= 1.0;
                A[rU][0] += -kI * p0; rhs[rU] -= kI * p0;
            } else {                           // 上侧：第 i − 1 层的层底（介质 i）：S = g·a + b，U = i·P·(g·a − b)
                const std::size_t ia = 1 + 2 * (i - 1);
                const cplx g = phase(i, layers[i - 1].first), p = admittance(i, TM);
                A[rS][ia] += g; A[rS][ia + 1] += 1.0;
                A[rU][ia] += kI * p * g; A[rU][ia + 1] += -kI * p;
            }
            if (i == L) {                      // 下侧：S = t，U = i·P_II·t
                A[rS][2 * L + 1] -= 1.0;
                A[rU][2 * L + 1] -= kI * admittance(L + 1, TM);
            } else {                           // 下侧：第 i 层的层顶（介质 i + 1）：S = a + g·b，U = i·P·(a − g·b)
                const std::size_t ib = 1 + 2 * i;
                const cplx g = phase(i + 1, layers[i].first), p = admittance(i + 1, TM);
                A[rS][ib] -= 1.0; A[rS][ib + 1] -= g;
                A[rU][ib] -= kI * p; A[rU][ib + 1] += kI * p * g;
            }
        }
        return solve_linear(A, rhs);
    }
    // 标量解在 z 处的 (S, U) 与该点的折射率
    void scalar(bool TM, const Vector& sol, double z, cplx& S, cplx& U, cplx& n) const {
        const std::size_t L = layers.size();
        if (z < 0.0) {
            const cplx f = std::exp(kI * k0 * kz[0] * z), b = sol[0] * std::exp(-kI * k0 * kz[0] * z), p = admittance(0, TM);
            S = f + b; U = kI * p * (f - b); n = nI;
        } else if (z > top[L]) {
            const cplx s = sol[2 * L + 1] * phase(L + 1, z - top[L]), p = admittance(L + 1, TM);
            S = s; U = kI * p * s; n = nII;
        } else {
            std::size_t j = 0;
            while (j + 1 < L && z > top[j + 1]) ++j;
            const double zl = z - top[j], d = layers[j].first;
            const cplx a = sol[1 + 2 * j] * phase(j + 1, zl), b = sol[2 + 2 * j] * phase(j + 1, d - zl), p = admittance(j + 1, TM);
            S = a + b; U = kI * p * (a - b); n = layers[j].second;
        }
    }
    std::array<cplx, 6> fields(double z) const {
        cplx S, U, nz, Sm, Um;
        scalar(false, te, z, S, U, nz);
        scalar(true, tm, z, Sm, Um, nz);
        const std::array<cplx, 6> TE{cplx{}, S, cplx{}, kI * U, cplx{}, -kt * S};
        const std::array<cplx, 6> TM{-kI * nI * Um, cplx{}, nI * kt * Sm / (nz * nz), cplx{}, nI * Sm, cplx{}};
        const double s = sin_phi, c = cos_phi;
        const std::array<cplx, 6> basis_s{-s * TE[1], c * TE[1], TE[2], c * TE[3], s * TE[3], TE[5]};
        const std::array<cplx, 6> basis_p{c * TM[0], s * TM[0], TM[2], -s * TM[4], c * TM[4], TM[5]};
        std::array<cplx, 6> out;
        for (std::size_t k = 0; k < 6; ++k) out[k] = jones[0] * basis_s[k] + jones[1] * basis_p[k];
        return out;
    }
};

struct AiryCase {
    const char* name;
    Problem problem;
    RealVector zs;
};

static void test_airy() {
    say("[均匀多层的 Airy 解析场（零阶六分量；其余级次恒为零）]\n");
    std::vector<AiryCase> cases;
    auto uniform_problem = [](double nI, cplx nII, std::vector<std::pair<double, cplx>> layers, double lambda, double theta, double phi,
                              rcwa::Jones2D jones) {
        Problem p;
        p.stack.nI = nI;
        p.stack.nII = nII;
        p.stack.Lx = 0.8;
        p.stack.Ly = 0.65;
        for (const auto& l : layers) p.stack.layers.push_back(rcwa::Layer2D::uniform(l.first, l.second));
        p.wavelength = lambda;
        p.theta = theta;
        p.phi = phi;
        p.M1 = 2;
        p.M2 = 2;
        p.jones = jones;
        return p;
    };
    cases.push_back({"单界面（无层），z < 0 与 z > 0",
                     uniform_problem(1.0, 1.5, {}, 1.053, 42.0, -25.0, rcwa::jones_from_angles_2d(40.0, 0.0)), {-2.0, -0.43, -1e-6, 1e-6, 0.3, 1.7}});
    cases.push_back({"无吸收膜 n = 1.8，d = 0.3",
                     uniform_problem(1.0, 1.5, {{0.3, 1.8}}, 1.053, 42.0, -25.0, rcwa::jones_from_angles_2d(40.0, 0.0)),
                     {-0.43, -1e-6, 0.05, 0.15, 0.299999, 0.3 + 1e-6, 0.9}});
    cases.push_back({"吸收膜 n = 1.65 + 0.12i、吸收基底 1.38 + 0.05i，椭圆偏振",
                     uniform_problem(1.0, cplx{1.38, 0.05}, {{0.27, cplx{1.65, 0.12}}}, 1.0, 27.0, 33.0,
                                     {cplx{0.7198463103929542, 0.6040227735550536}, cplx{0.3420201433256688, 0.0}}),
                     {-1.0, -0.07, 1e-6, 0.081, 0.189, 0.269999, 0.270001, 0.36, 0.68, 3.0}});
    cases.push_back({"全反射：nI = 1.5、θ = 50°，出射区倏逝；两层膜",
                     uniform_problem(1.5, 1.0, {{0.2, 1.9}, {0.35, 1.3}}, 0.9, 50.0, 70.0, rcwa::jones_from_angles_2d(25.0, 80.0)),
                     {-0.6, -0.001, 0.0 + 1e-7, 0.1, 0.199999, 0.200001, 0.4, 0.549999, 0.550001, 0.9, 2.5}});
    cases.push_back({"厚层：吸收 15 µm（n = 1.8 + 0.3i）、无吸收 12 µm、薄膜、吸收基底",
                     uniform_problem(1.0, cplx{1.5, 0.01}, {{0.2, 1.46}, {15.0, cplx{1.8, 0.3}}, {12.0, 1.46}, {0.1, 2.2}}, 1.053, 33.0, 40.0,
                                     rcwa::jones_from_angles_2d(70.0, 20.0)),
                     {-3.0, 0.1, 0.2 + 1e-6, 5.0, 15.19, 15.2 + 1e-6, 20.0, 27.1, 27.2 - 1e-6, 27.3 + 1e-6, 27.4, 31.0}});
    cases.push_back({"θ = 0、φ = 30°（kt = 0）",
                     uniform_problem(1.0, 1.5, {{0.25, 2.0}, {0.1, 1.2}}, 1.053, 0.0, 30.0, rcwa::jones_from_angles_2d(60.0, 10.0)),
                     {-0.5, 0.1, 0.3, 0.7}});
    for (int variant = 0; variant < 2; ++variant) {       // variant 1：均匀层按稠密层处理（本征分解路径），容差放宽
        double worst = 0.0, worst_others = 0.0;
        for (AiryCase& ac : cases) {
            if (variant == 1 && std::string(ac.name).find("厚层") != std::string::npos) continue;     // 稠密路径下 12 µm 层的本征解不再解析
            Problem p = ac.problem;
            p.options.homogeneous_fast_path = variant == 0;
            const rcwa::FieldResult2D run = solve(p);
            CHECK(run.fields.available());
            if (!run.fields.available()) continue;
            std::vector<std::pair<double, cplx>> layers;
            for (const auto& l : p.stack.layers) layers.push_back({l.d, l.n});
            const AiryModel model(p.stack.nI, p.stack.nII, layers, p.wavelength, p.theta, p.phi, p.jones);
            const rcwa::HarmonicFields h = rcwa::field_at(run, ac.zs);
            Cmp cmp;
            double others = 0.0;
            for (std::size_t k = 0; k < ac.zs.size(); ++k) {
                const auto expected = model.fields(ac.zs[k]);
                for (std::size_t comp = 0; comp < 6; ++comp) {
                    cmp.add(h.components[comp](run.result.i0, static_cast<int>(k)), expected[comp], variant == 0 ? 1e-12 : 1e-9, variant == 0 ? 1e-12 : 1e-9);
                    for (int j = 0; j < run.result.N; ++j)
                        if (j != run.result.i0) others = (std::max)(others, std::abs(h.components[comp](j, static_cast<int>(k))));
                }
            }
            CHECK(cmp.ok());
            CHECK_LE(others, variant == 0 ? 1e-14 : 1e-10);
            worst = (std::max)(worst, cmp.maxdiff);
            worst_others = (std::max)(worst_others, others);
            if (variant == 0) say("  %-62s %2zu 个 z  零阶最大差 %.2e  其余级次最大幅值 %.2e\n", ac.name, ac.zs.size(), cmp.maxdiff, others);
        }
        say(" %s：最大差 %.2e，其余级次最大幅值 %.2e\n", variant == 0 ? "快路径" : "homogeneous_fast_path = false", worst, worst_others);
    }
}

// ---------------------------------------------------------------------------------------------
// c6) 外部区域里每个谐波都是一列平面波：E ⊥ k，H = (k/k0) × E（独立于 V、Ei 的写法；只用 Result2D 的 Ex、Ey 远场振幅与 Maxwell 的平面波关系）
// 谐波 (m, n) 的横向因子是 exp[−i k0 (Kx x + Ky y)]，所以 k/k0 = (−Kx, −Ky, s·kz)，下行波 s = +1（exp(+i k0 kz z)），上行波 s = −1。
// 平面波：Ez = s(Kx Ex + Ky Ey)/kz（E·k = 0），Z0H = (k/k0) × E：Hx = −Ky Ez − s kz Ey，Hy = s kz Ex + Kx Ez，Hz = Ky Ex − Kx Ey。
// 对倏逝级（kz = +i|kz|）和吸收出射区（复 kz）同样成立（k·E = 0 对复 kz 也成立）。
// 注意：Result2D 的 Ez_r、Ez_t（从 rcwa2d.py 的 efficiency 照搬）与这里的 Ez 符号相反（它们用 −sign·(Kx Ex + Ky Ey)/kz），只进效率的 |Ez|²，不影响效率。
static std::array<cplx, 6> plane_wave_fields(cplx Ex, cplx Ey, cplx kz, double s, double Kx, double Ky) {
    const cplx Ez = s * (Kx * Ex + Ky * Ey) / kz;
    return {Ex, Ey, Ez, -Ky * Ez - s * kz * Ey, s * kz * Ex + Kx * Ez, Ky * Ex - Kx * Ey};
}

static void test_exterior_plane_waves() {
    say("[外部区域的谐波是平面波：E ⊥ k、H = (k/k0) × E（用 Result2D 的远场 Ex、Ey 独立写出六分量）]\n");
    struct Named { const char* name; Problem problem; };
    std::vector<Named> problems{{"A", problem_a()}, {"B（吸收出射区、nI = 1.2）", problem_b()}, {"C（厚层）", problem_c()}};
    {
        Problem tir = problem_a();       // 全反射：出射区大量倏逝级
        tir.stack.nI = 1.5;
        tir.stack.nII = 1.0;
        tir.theta = 50.0;
        problems.push_back({"A 全反射（nI = 1.5 → 1.0，θ = 50°）", tir});
    }
    for (const Named& n : problems) {
        const Problem& p = n.problem;
        const rcwa::FieldResult2D run = solve(p);
        CHECK(run.fields.available());
        if (!run.fields.available()) continue;
        const rcwa::Result2D& r = run.result;
        const double k0 = r.k0, z_bot = run.fields.metadata().z_bottom;
        const double th = p.theta * kPi / 180.0, ph = p.phi * kPi / 180.0;
        const cplx inc_x = -p.jones[0] * std::sin(ph) + p.jones[1] * std::cos(th) * std::cos(ph);
        const cplx inc_y = p.jones[0] * std::cos(ph) + p.jones[1] * std::cos(th) * std::sin(ph);
        const RealVector zs{-0.3, -0.01, z_bot + 0.01, z_bot + 0.3, z_bot + 1.1};
        const rcwa::HarmonicFields h = rcwa::field_at(run, zs);
        double worst = 0.0, scale = 0.0;
        for (std::size_t k = 0; k < zs.size(); ++k)
            for (int j = 0; j < r.N; ++j) {
                const std::size_t u = static_cast<std::size_t>(j);
                std::array<cplx, 6> expected{};
                if (zs[k] < 0.0) {
                    const std::array<cplx, 6> up = plane_wave_fields(r.Ex_r[u], r.Ey_r[u], r.kzI[u], -1.0, r.Kx[u], r.Ky[u]);
                    const cplx wave = std::exp(-kI * k0 * r.kzI[u] * zs[k]);
                    for (std::size_t c = 0; c < 6; ++c) expected[c] = up[c] * wave;
                    if (j == r.i0) {
                        const std::array<cplx, 6> down = plane_wave_fields(inc_x, inc_y, r.kzI[u], +1.0, r.Kx[u], r.Ky[u]);
                        const cplx incident = std::exp(kI * k0 * r.kzI[u] * zs[k]);
                        for (std::size_t c = 0; c < 6; ++c) expected[c] += down[c] * incident;
                    }
                } else {
                    const std::array<cplx, 6> down = plane_wave_fields(r.Ex_t[u], r.Ey_t[u], r.kzII[u], +1.0, r.Kx[u], r.Ky[u]);
                    const cplx wave = std::exp(kI * k0 * r.kzII[u] * (zs[k] - z_bot));
                    for (std::size_t c = 0; c < 6; ++c) expected[c] = down[c] * wave;
                }
                for (std::size_t c = 0; c < 6; ++c) {
                    worst = (std::max)(worst, std::abs(h.components[c](j, static_cast<int>(k)) - expected[c]));
                    scale = (std::max)(scale, std::abs(expected[c]));
                }
            }
        CHECK_LE(worst, 1e-12);
        // 观察：Result2D 的 Ez_t 是近场 Ez（紧贴 z_bot 外侧）的相反数，|Ez| 相同
        const rcwa::HarmonicFields at_bottom = rcwa::field_at(run, z_bot + 1e-13);
        const rcwa::HarmonicFields at_top = rcwa::field_at(run, -1e-13);
        double sign_opposite = 0.0, magnitude = 0.0;
        for (int j = 0; j < r.N; ++j) {
            const std::size_t u = static_cast<std::size_t>(j);
            if (r.prop_t[u]) {
                const cplx ez = at_bottom.component(rcwa::FieldComponent::Ez)(j, 0), ez_t = r.Ez_t[u];
                sign_opposite = (std::max)(sign_opposite, std::abs(ez + ez_t));
                magnitude = (std::max)(magnitude, std::abs(std::abs(ez) - std::abs(ez_t)));
            }
            if (r.prop_r[u] && j != r.i0) {      // 零阶在区域 I 里还有入射波，不比
                const cplx ez = at_top.component(rcwa::FieldComponent::Ez)(j, 0), ez_r = r.Ez_r[u];
                sign_opposite = (std::max)(sign_opposite, std::abs(ez + ez_r));
                magnitude = (std::max)(magnitude, std::abs(std::abs(ez) - std::abs(ez_r)));
            }
        }
        CHECK_LE(magnitude, 1e-12);
        say("  %-36s 区域 I（%d 级）与区域 II 各 %zu 个 z：对平面波公式最大差 %.2e（最大场 %.2f）；|Ez| 对 Result2D::Ez_t 差 %.1e，Ez + Ez_t 差 %.1e（符号相反）\n", n.name, r.N,
            zs.size(), worst, scale, magnitude, sign_opposite);
    }
}

// ---------------------------------------------------------------------------------------------
// c3) y 不变 / x 不变结构对一维 C++
static rcwa::Stack one_d_stack(double nI, cplx nII, double period, const std::vector<rcwa::Layer>& layers) {
    rcwa::Stack s;
    s.nI = nI;
    s.nII = nII;
    s.period = period;
    s.layers = layers;
    return s;
}

// 一个 y 不变（axis = 'x'：沿 x 周期）或 x 不变（axis = 'y'）的多层结构：第一层光栅（带起点偏移的边界）、均匀膜、第二层光栅
struct Invariant {
    Problem problem;
    rcwa::Stack stack1d;
};
static Invariant make_invariant(char axis, int M, double nI, cplx nII, bool absorbing) {
    const double P = 0.574, other = 0.8;
    const std::vector<double> e1{-0.0861, 0.0861, 0.4879}, e2{0.05, 0.31, 0.574 + 0.05};
    const cplx hi1 = absorbing ? cplx{1.46, 0.02} : cplx{1.46, 0.0}, hi2 = absorbing ? cplx{2.2, 0.15} : cplx{2.2, 0.0};
    Invariant out;
    Problem& p = out.problem;
    p.stack.nI = nI;
    p.stack.nII = nII;
    p.wavelength = 1.053;
    p.theta = 36.0;
    if (axis == 'x') {          // y 不变：沿 x 周期 P
        p.stack.Lx = P;
        p.stack.Ly = other;
        p.stack.layers.push_back(rcwa::Layer2D::patterned(0.38, {0.0, other}, {profile(e1, {hi1, 1.0})}));
        p.stack.layers.push_back(rcwa::Layer2D::uniform(0.12, 1.96));
        p.stack.layers.push_back(rcwa::Layer2D::patterned(0.25, {0.0, other}, {profile(e2, {hi2, 1.0})}));
        p.M1 = M;
        p.M2 = 0;
    } else {                    // x 不变：沿 y 周期 P，每个 y 分段在 x 方向是均匀的
        p.stack.Lx = other;
        p.stack.Ly = P;
        const auto bars = [&](const std::vector<double>& e, cplx a, cplx b) {
            std::vector<rcwa::XProfile2D> rows{profile({0.0, other}, {a}), profile({0.0, other}, {b})};
            return rcwa::Layer2D::patterned(0.0, e, rows);
        };
        rcwa::Layer2D g1 = bars(e1, hi1, 1.0), g2 = bars(e2, hi2, 1.0);
        g1.d = 0.38;
        g2.d = 0.25;
        p.stack.layers.push_back(g1);
        p.stack.layers.push_back(rcwa::Layer2D::uniform(0.12, 1.96));
        p.stack.layers.push_back(g2);
        p.M1 = 0;
        p.M2 = M;
    }
    out.stack1d = one_d_stack(nI, nII, P,
                              {rcwa::Layer::grating(0.38, e1, {hi1, 1.0}), rcwa::Layer::uniform(0.12, 1.96), rcwa::Layer::grating(0.25, e2, {hi2, 1.0})});
    return out;
}

// 二维的 (Ex, Ey, Ez, Hx, Hy, Hz) = 一维坐标下的分量按轴映射
static std::array<cplx, 6> map_to_2d(const std::array<cplx, 6>& f, char axis) {
    if (axis == 'x') return f;
    return {-f[1], f[0], f[2], -f[4], f[3], f[5]};      // x_1D = y、y_1D = −x：V_x = −V_1y，V_y = V_1x
}

static void test_one_d_limits() {
    say("[y 不变 / x 不变结构对一维 C++（solve_with_fields TE/TM；solve_conical_with_fields）]\n");
    const RealVector zs{-0.6, -0.003, 0.0, 0.1, 0.379, 0.4, 0.45, 0.505, 0.56, 0.7, 0.755, 1.0, 2.0};
    for (const char axis : {'x', 'y'}) {
        for (int variant = 0; variant < 4; ++variant) {
            // 0：无吸收，nI = 1，TE/TM（φ = 0 或 90°）；1：无吸收，nI = 1.2，椭圆偏振圆锥；2：吸收，圆锥；3：吸收，nI = 1.2、nII = 1.5
            // （圆锥的一维求解器要求两侧半无限介质的折射率为实数，吸收只放在层里）
            const int M = 6;
            const double nI = (variant == 1 || variant == 3) ? 1.2 : 1.0;
            const cplx nII = variant == 3 ? cplx{1.5, 0.0} : cplx{1.46, 0.0};
            Invariant inv = make_invariant(axis, M, nI, nII, variant >= 2);
            const double phi2 = variant == 0 ? (axis == 'x' ? 0.0 : 90.0) : (variant == 1 ? 33.0 : variant == 2 ? -48.0 : 77.0);
            inv.problem.phi = phi2;
            const double phi1 = axis == 'x' ? phi2 : phi2 - 90.0;
            const std::size_t K = 2 * static_cast<std::size_t>(M) + 1;
            auto jones_list = std::vector<rcwa::Jones2D>{rcwa::jones_from_angles_2d(90.0), rcwa::jones_from_angles_2d(0.0), rcwa::jones_from_angles_2d(55.0, 35.0)};
            if (variant == 0) jones_list.resize(2);     // 纯 s / p（对一维 TE / TM）
            else jones_list.erase(jones_list.begin(), jones_list.begin() + 2);
            for (const rcwa::Jones2D& jones : jones_list) {
                inv.problem.jones = jones;
                const rcwa::FieldResult2D run = solve(inv.problem);
                CHECK(run.fields.available());
                if (!run.fields.available()) continue;
                const rcwa::HarmonicFields h2 = rcwa::field_at(run, zs);
                CHECK(h2.norders() == K);
                double worst = 0.0;
                std::string label;
                {   // 圆锥（任意偏振）
                    const rcwa::ConicalFieldResult c1 = rcwa::solve_conical_with_fields(inv.stack1d, inv.problem.wavelength, inv.problem.theta, phi1, M);
                    CHECK(c1.fields.available());
                    if (c1.fields.available()) {
                        const rcwa::HarmonicFields h1 = rcwa::field_at(c1, zs, {jones[0], jones[1]});
                        for (std::size_t k = 0; k < zs.size(); ++k)
                            for (std::size_t j = 0; j < K; ++j) {
                                std::array<cplx, 6> f1;
                                for (std::size_t c = 0; c < 6; ++c) f1[c] = h1.components[c](static_cast<int>(j), static_cast<int>(k));
                                const auto m = map_to_2d(f1, axis);
                                for (std::size_t c = 0; c < 6; ++c)
                                    worst = (std::max)(worst, std::abs(h2.components[c](static_cast<int>(j), static_cast<int>(k)) - m[c]));
                            }
                        label = "圆锥";
                    }
                }
                CHECK_LE(worst, 1e-10);
                double worst_scalar = -1.0;
                if (variant == 0) {   // 纯 s / p 且 φ 让入射面垂直 / 平行于栅线：对一维 TE / TM
                    const bool te = jones[0] == cplx{1.0, 0.0};
                    const rcwa::PlanarFieldResult p1 = rcwa::solve_with_fields(inv.stack1d, inv.problem.wavelength, inv.problem.theta, M,
                                                                              te ? rcwa::Polarization::TE : rcwa::Polarization::TM);
                    CHECK(p1.fields.available());
                    if (p1.fields.available()) {
                        const rcwa::HarmonicFields h1 = rcwa::field_at(p1, zs);
                        // TM：一维入射 Z0·Hy = 1（|E| = 1/nI），二维单位电场 = nI 倍
                        const double scale = te ? 1.0 : nI;
                        worst_scalar = 0.0;
                        for (std::size_t k = 0; k < zs.size(); ++k)
                            for (std::size_t j = 0; j < K; ++j) {
                                std::array<cplx, 6> f1;
                                for (std::size_t c = 0; c < 6; ++c) f1[c] = scale * h1.components[c](static_cast<int>(j), static_cast<int>(k));
                                const auto m = map_to_2d(f1, axis);
                                for (std::size_t c = 0; c < 6; ++c)
                                    worst_scalar = (std::max)(worst_scalar, std::abs(h2.components[c](static_cast<int>(j), static_cast<int>(k)) - m[c]));
                            }
                    }
                    CHECK_LE(worst_scalar, 1e-10);
                }
                say("  %c 不变  变体 %d  Jones (%+.2f%+.2fi, %+.2f%+.2fi)  对%s 最大差 %.2e", axis, variant, jones[0].real(), jones[0].imag(), jones[1].real(),
                    jones[1].imag(), label.c_str(), worst);
                if (worst_scalar >= 0.0) say("；对一维 %s %.2e", jones[0] == cplx{1.0, 0.0} ? "TE" : "TM（×nI）", worst_scalar);
                say("\n");
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// c4) 能流
static double poynting_z(const rcwa::HarmonicFields& h, std::size_t iz) {
    double s = 0.0;
    for (int j = 0; j < h.component(rcwa::FieldComponent::Ex).rows(); ++j)
        s += (h.component(rcwa::FieldComponent::Ex)(j, static_cast<int>(iz)) * std::conj(h.component(rcwa::FieldComponent::Hy)(j, static_cast<int>(iz))) -
              h.component(rcwa::FieldComponent::Ey)(j, static_cast<int>(iz)) * std::conj(h.component(rcwa::FieldComponent::Hx)(j, static_cast<int>(iz)))).real();
    return s;
}

static void test_poynting_flux() {
    say("[Poynting 通量]\n");
    for (int variant = 0; variant < 2; ++variant) {
        Problem p = problem_a();
        if (variant == 1) p = problem_a(), p.options.homogeneous_fast_path = false;
        p.jones = {cplx{0.6, 0.0}, cplx{0.0, 0.8}};      // 单位范数
        const rcwa::FieldResult2D run = solve(p);
        CHECK(run.fields.available());
        if (!run.fields.available()) continue;
        const double inc = p.stack.nI * std::cos(p.theta * kPi / 180.0);
        const double reflect = run.result.reflectance(), transmit = run.result.transmittance();
        CHECK_LE(std::abs(reflect + transmit - 1.0), 1e-9);
        RealVector zs{-1.2, -0.31, -0.02, -1e-9, 1e-9, 0.05, 0.15, 0.29, 0.3 - 1e-9, 0.3 + 1e-9, 0.35, 0.40, 0.415 - 1e-9, 0.415 + 1e-9, 0.5, 0.6, 0.725 - 1e-9,
                      0.725 + 1e-9, 0.8, 1.9};
        const rcwa::HarmonicFields h = rcwa::field_at(run, zs);
        double worst = 0.0, worst_t = 0.0, worst_r = 0.0;
        for (std::size_t k = 0; k < zs.size(); ++k) {
            const double s = poynting_z(h, k);
            worst = (std::max)(worst, std::abs(s - inc * transmit));
            if (zs[k] < 0.0) worst_r = (std::max)(worst_r, std::abs(s - inc * (1.0 - reflect)));
            if (zs[k] > 0.725) worst_t = (std::max)(worst_t, std::abs(s - inc * transmit));
        }
        CHECK_LE(worst, 1e-9);
        CHECK_LE(worst_r, 1e-9);
        CHECK_LE(worst_t, 1e-9);
        say("  无吸收 A%s：Sz 与 z 无关并等于 nI cosθ ΣDEt，%zu 个 z（区域 I、层内、界面两侧、区域 II）最大差 %.2e；区域 I 对 nI cosθ(1−ΣDEr) %.2e；区域 II %.2e\n",
            variant ? "（稠密路径）" : "", zs.size(), worst, worst_r, worst_t);
    }
    // 吸收：通量不随 z 增加（容许 1e-9 的截断噪声）；入射 Sz 的区域 I 部分（z < 0）与 nI cosθ(1 − ΣDEr) 一致
    Problem p = problem_b();
    p.jones = rcwa::jones_from_angles_2d(33.0, 50.0);
    const rcwa::FieldResult2D run = solve(p);
    CHECK(run.fields.available());
    RealVector zs;
    for (double z = -0.9; z < 1.5; z += 0.0371) zs.push_back(z);
    const rcwa::HarmonicFields h = rcwa::field_at(run, zs);
    double previous = poynting_z(h, 0), worst_increase = 0.0;
    for (std::size_t k = 1; k < zs.size(); ++k) {
        const double s = poynting_z(h, k);
        worst_increase = (std::max)(worst_increase, s - previous);
        previous = s;
    }
    CHECK_LE(worst_increase, 1e-9);
    const double inc = p.stack.nI * std::cos(p.theta * kPi / 180.0);
    const double s_first = poynting_z(h, 0), s_last = poynting_z(h, zs.size() - 1);
    CHECK(s_first > s_last && s_last > 0.0);
    CHECK_LE(std::abs(s_first - inc * (1.0 - run.result.reflectance())), 1e-9);
    say("  吸收 B：Sz 从 %.6f 单调降到 %.6f（最大回升 %.2e）；区域 I 的 Sz 与 nI cosθ(1−ΣDEr) 差 %.2e\n", s_first, s_last, worst_increase,
        std::abs(s_first - inc * (1.0 - run.result.reflectance())));
}

// ---------------------------------------------------------------------------------------------
// c5) x↔y 互换（只比模）
// 对角镜面 (x, y) → (y, x)：入射 φ → 90° − φ，Jones (Es, Ep) → (−Es, Ep)（s 单位矢量在镜像下变号，p 不变）；
// 场的谐波列 (m, n) ↔ (n, m)，Ex ↔ Ey、Hx ↔ Hy（H 是赝矢量，镜像下再差一个整体符号，所以只比模），Ez、Hz 不动。
// 前两个几何关于对角线自对称（方格子、柱 [a, b]²），后两个是 Lx ≠ Ly、M1 ≠ M2 的矩形柱，把整个结构转置（Lx↔Ly、x↔y 分段互换）。
static double xy_exchange_difference(const Problem& p, const Problem& q, double& scale) {
    const rcwa::FieldResult2D a = solve(p), b = solve(q);
    CHECK(a.fields.available() && b.fields.available());
    if (!a.fields.available() || !b.fields.available()) return 1e300;
    const RealVector zs{-0.4, -0.01, 0.07, 0.29, 0.35, 0.42, 0.5, 0.6, 1.4};
    const rcwa::HarmonicFields ha = rcwa::field_at(a, zs), hb = rcwa::field_at(b, zs);
    double worst = 0.0;
    scale = 0.0;
    const std::size_t partner[6] = {1, 0, 2, 4, 3, 5};      // 镜像下 Ex↔Ey、Hx↔Hy，Ez、Hz 不动
    for (std::size_t k = 0; k < zs.size(); ++k)
        for (int m = -p.M1; m <= p.M1; ++m)
            for (int n = -p.M2; n <= p.M2; ++n) {
                const int ja = rcwa::harmonic_index_2d(m, n, p.M1, p.M2), jb = rcwa::harmonic_index_2d(n, m, q.M1, q.M2);
                for (std::size_t c = 0; c < 6; ++c) {
                    const double ma = std::abs(ha.components[c](ja, static_cast<int>(k))), mb = std::abs(hb.components[partner[c]](jb, static_cast<int>(k)));
                    worst = (std::max)(worst, std::abs(ma - mb));
                    scale = (std::max)(scale, ma);
                }
            }
    return worst;
}

static void test_xy_exchange() {
    say("[x↔y 互换（只比模）]\n");
    for (int variant = 0; variant < 4; ++variant) {
        Problem p, q;
        p.stack.nI = q.stack.nI = 1.1;
        p.stack.nII = q.stack.nII = variant == 1 ? cplx{1.5, 0.02} : cplx{1.5, 0.0};
        const cplx n_pillar = variant == 1 ? cplx{1.8, 0.1} : cplx{1.8, 0.0};
        if (variant < 2) {
            const double L = 0.8;
            p.stack.Lx = p.stack.Ly = q.stack.Lx = q.stack.Ly = L;
            p.stack.layers.push_back(pillar_layer(0.3, L, L, 0.15, 0.45, 0.15, 0.45, n_pillar, 1.0));
            p.stack.layers.push_back(rcwa::Layer2D::uniform(0.12, 1.9));
            p.stack.layers.push_back(pillar_layer(0.2, L, L, 0.3, 0.6, 0.3, 0.6, 2.2, 1.2));
            q.stack.layers = p.stack.layers;       // 关于对角线自对称：镜像后是同一个结构
        } else {
            const double Lx = 0.8, Ly = 0.6;
            p.stack.Lx = Lx; p.stack.Ly = Ly;
            q.stack.Lx = Ly; q.stack.Ly = Lx;
            p.stack.layers.push_back(pillar_layer(0.3, Lx, Ly, 0.15, 0.45, 0.1, 0.35, n_pillar, 1.0));
            p.stack.layers.push_back(rcwa::Layer2D::uniform(0.12, 1.9));
            p.stack.layers.push_back(pillar_layer(0.2, Lx, Ly, 0.4, 0.7, 0.2, 0.5, 2.2, 1.2));
            q.stack.layers.push_back(pillar_layer(0.3, Ly, Lx, 0.1, 0.35, 0.15, 0.45, n_pillar, 1.0));
            q.stack.layers.push_back(rcwa::Layer2D::uniform(0.12, 1.9));
            q.stack.layers.push_back(pillar_layer(0.2, Ly, Lx, 0.2, 0.5, 0.4, 0.7, 2.2, 1.2));
            p.options.factorization = q.options.factorization = variant == 3 ? rcwa::Factorization2D::Laurent : rcwa::Factorization2D::Li;
        }
        p.theta = q.theta = 37.0;
        p.phi = 23.0;
        q.phi = 90.0 - p.phi;
        p.M1 = 3;
        p.M2 = variant < 2 ? 3 : 2;
        q.M1 = p.M2;
        q.M2 = p.M1;
        p.jones = {cplx{0.4, 0.3}, cplx{-0.5, 0.2}};
        q.jones = {-p.jones[0], p.jones[1]};
        double scale = 0.0;
        const double worst = xy_exchange_difference(p, q, scale);
        CHECK_LE(worst, 1e-9);
        static const char* const names[4] = {"方格子、对角对称几何", "方格子、对角对称几何（吸收）", "矩形格子 0.8×0.6（Li），整体转置", "矩形格子 0.8×0.6（Laurent），整体转置"};
        say("  %-36s：|场| 最大差 %.2e（最大 |场| %.2f）\n", names[variant], worst, scale);
    }
}

// ---------------------------------------------------------------------------------------------
// 切面：与直接求和一致、坐标与布局
static void test_slice_brute_force() {
    say("[切面对直接求和（独立于 reconstruct_field_slice 的写法）]\n");
    Problem p = problem_a();
    const rcwa::FieldResult2D run = solve(p);
    CHECK(run.fields.available());
    const RealVector zs{-0.3, 0.05, 0.05, 0.32, 0.5, 0.9};
    const int nt = 7;
    double worst = 0.0;
    for (const rcwa::SlicePlane plane : {rcwa::SlicePlane::XZ, rcwa::SlicePlane::YZ}) {
        const double at = plane == rcwa::SlicePlane::XZ ? 0.21 : -0.13;
        const rcwa::FieldGridSlice g = rcwa::field_slice(run, zs, plane, nt, at);
        const rcwa::HarmonicFields h = rcwa::field_at(run, zs);
        CHECK(g.plane == plane && g.nt() == static_cast<std::size_t>(nt) && g.nz() == zs.size() && g.z == zs);
        CHECK(g.metadata.kind == rcwa::FieldSolveKind::Periodic2D && g.metadata.orders_m.size() == static_cast<std::size_t>(run.result.N));
        const double period = plane == rcwa::SlicePlane::XZ ? p.stack.Lx : p.stack.Ly;
        for (int it = 0; it < nt; ++it) {
            const double t = it * period / nt;
            const double x = plane == rcwa::SlicePlane::XZ ? t : at, y = plane == rcwa::SlicePlane::XZ ? at : t;
            CHECK(std::abs(g.t[static_cast<std::size_t>(it)] - t) <= 1e-15 && std::abs(g.x[static_cast<std::size_t>(it)] - x) <= 1e-15 &&
                  std::abs(g.y[static_cast<std::size_t>(it)] - y) <= 1e-15);
            for (std::size_t iz = 0; iz < zs.size(); ++iz)
                for (std::size_t c = 0; c < 6; ++c) {
                    cplx sum{};
                    for (int j = 0; j < run.result.N; ++j) {
                        const std::size_t k = static_cast<std::size_t>(j);
                        sum += h.components[c](j, static_cast<int>(iz)) * std::exp(-kI * run.result.k0 * (run.result.Kx[k] * x + run.result.Ky[k] * y));
                    }
                    worst = (std::max)(worst, std::abs(sum - g.at(static_cast<rcwa::FieldComponent>(c), iz, static_cast<std::size_t>(it))));
                }
        }
    }
    CHECK_LE(worst, 1e-12);
    say("  xz（y = 0.21）与 yz（x = −0.13），nt = %d，%zu 个 z：最大差 %.2e\n", nt, zs.size(), worst);
}

// ---------------------------------------------------------------------------------------------
// d) 语义
static void test_context_ownership() {
    say("[上下文：拥有数据、深拷贝、移动、失败]\n");
    const RealVector zs{-0.2, 0.0, 0.1, 0.33, 0.4, 0.9};
    rcwa::HarmonicFields before;
    rcwa::FieldContext2D ctx;
    CHECK(ctx.status() == rcwa::FieldContextStatus::Empty && !ctx.available());
    CHECK_THROWS_UNAVAILABLE(ctx.require_available());
    CHECK_THROWS_UNAVAILABLE(rcwa::field_at(ctx, zs));
    CHECK_THROWS_UNAVAILABLE(rcwa::field_slice(ctx, zs, rcwa::SlicePlane::XZ, 8, 0.1));
    std::size_t bytes = 0;
    {
        auto stack = std::make_unique<rcwa::Stack2D>(problem_a().stack);
        Problem p = problem_a();
        rcwa::FieldResult2D run = rcwa::solve2d_with_fields(*stack, p.wavelength, p.theta, p.phi, p.M1, p.M2, p.jones, p.options);
        CHECK(run.fields.available() && run.fields.status() == rcwa::FieldContextStatus::Ready);
        before = rcwa::field_at(run.fields, zs);
        bytes = run.fields.memory_bytes();
        ctx = std::move(run.fields);
        CHECK(run.fields.status() == rcwa::FieldContextStatus::Empty);
        // 把 Stack 里的东西全部改写再销毁：上下文不能读它
        for (rcwa::Layer2D& layer : stack->layers) {
            layer.d = 123.0;
            layer.n = cplx{9.0, 9.0};
            for (double& e : layer.y_edges) e = -1e9;
            for (rcwa::XProfile2D& pr : layer.x_profiles) {
                for (double& e : pr.edges) e = 1e9;
                for (cplx& v : pr.n_values) v = cplx{kNaN, kNaN};
            }
        }
        stack->nI = 5.0;
        stack->nII = 3.0;
        stack->Lx = 17.0;
        stack->layers.clear();
        stack.reset();
    }
    CHECK(ctx.available() && bytes > 0 && ctx.memory_bytes() == bytes);
    CHECK(same_harmonics(before, rcwa::field_at(ctx, zs)));       // Stack 销毁后采样不变
    // 元数据是独立的快照
    const rcwa::FieldMetadata& meta = ctx.metadata();
    CHECK(meta.kind == rcwa::FieldSolveKind::Periodic2D && meta.layers.size() == 3 && meta.period == 0.7 && meta.period_y == 0.55);
    CHECK(std::abs(meta.z_bottom - (0.3 + 0.115 + 0.31)) <= 1e-15 && meta.layers[1].z_top == 0.3 && meta.layers[2].z_bottom == meta.z_bottom);
    // 深拷贝
    rcwa::FieldContext2D copy = ctx;
    CHECK(copy.available() && copy.memory_bytes() == bytes && same_harmonics(before, rcwa::field_at(copy, zs)));
    ctx = rcwa::FieldContext2D{};
    CHECK(ctx.status() == rcwa::FieldContextStatus::Empty && ctx.metadata().orders.empty());
    CHECK(same_harmonics(before, rcwa::field_at(copy, zs)));
    rcwa::FieldContext2D assigned;
    assigned = copy;
    copy = rcwa::FieldContext2D{};
    CHECK(same_harmonics(before, rcwa::field_at(assigned, zs)));
    // 移动后源对象为 Empty，元数据与数据一并重置
    rcwa::FieldContext2D moved(std::move(assigned));
    CHECK(assigned.status() == rcwa::FieldContextStatus::Empty && !assigned.available());
    CHECK(assigned.metadata().orders.empty() && assigned.metadata().layers.empty() && assigned.metadata().kxm.empty() && assigned.memory_bytes() == 0);
    CHECK(assigned.metadata().period == 0.0 && assigned.metadata().period_y == 0.0 && assigned.metadata().k0 == 0.0 && assigned.metadata().z_bottom == 0.0 &&
          assigned.metadata().truncation_order == 0 && assigned.metadata().wavelength == 0.0 && assigned.metadata().kind == rcwa::FieldSolveKind::PlanarTE);
    CHECK_THROWS_UNAVAILABLE(assigned.require_available());
    CHECK_THROWS_UNAVAILABLE(rcwa::field_at(assigned, 0.1));
    CHECK(moved.available() && same_harmonics(before, rcwa::field_at(moved, zs)));
    rcwa::FieldContext2D target = moved;                 // 移动赋值到已有数据的对象
    rcwa::FieldContext2D source = moved;
    target = std::move(source);
    CHECK(source.status() == rcwa::FieldContextStatus::Empty && target.available() && same_harmonics(before, rcwa::field_at(target, zs)));
    CHECK(moved.available());                            // 没被动过
    rcwa::FieldContext2D& alias = target;
    target = std::move(alias);                           // 自移动赋值不破坏对象
    CHECK(target.available() && same_harmonics(before, rcwa::field_at(target, zs)));
    // 移动后的对象可以重新被赋值使用
    source = moved;
    CHECK(source.available() && same_harmonics(before, rcwa::field_at(source, zs)));
    // 结果对象（FieldResult2D）的移动：result 与 fields 一起走
    rcwa::FieldResult2D run = solve(problem_a());
    rcwa::FieldResult2D run2(std::move(run));
    CHECK(run.fields.status() == rcwa::FieldContextStatus::Empty && run2.fields.available() && run2.result.status == rcwa::Status2D::Ok);
}

static void test_failure_unavailable() {
    say("[数值失败：上下文不可用，不返回貌似有效的零场]\n");
    struct Failing { const char* name; Problem problem; rcwa::Status2D status; };
    std::vector<Failing> failing;
    {
        Problem p = problem_a();
        p.options.custom_eig = [](const Matrix& a) {
            EigResult r;
            r.info = 5;
            r.values.assign(static_cast<std::size_t>(a.rows()), cplx{});
            return r;
        };
        failing.push_back({"本征迭代未收敛（custom_eig 报告 info = 5）", p, rcwa::Status2D::NumericalFailure});
    }
    {
        Problem p = problem_a();
        p.options.custom_eig = [](const Matrix& a) {
            EigResult r = eig(a);
            r.values[0] = cplx{kNaN, 0.0};
            return r;
        };
        failing.push_back({"本征值含 NaN", p, rcwa::Status2D::NumericalFailure});
    }
    {
        const Rcwa2dRefCase& c = rcwa2d_ref_case(23);
        Problem p;
        p.stack = rcwa2d_ref_stack(23);
        p.wavelength = c.wavelength; p.theta = c.theta_deg; p.phi = c.phi_deg; p.M1 = c.M1; p.M2 = c.M2;
        p.jones = {cplx{c.jones[0], c.jones[1]}, cplx{c.jones[2], c.jones[3]}};
        failing.push_back({"Rayleigh 异常（入射区 kz = 0）", p, rcwa::Status2D::RayleighAnomaly});
    }
    {
        Problem p;
        p.stack = rcwa2d_ref_stack(24);
        const Rcwa2dRefCase& c = rcwa2d_ref_case(24);
        p.wavelength = c.wavelength; p.theta = c.theta_deg; p.phi = c.phi_deg; p.M1 = c.M1; p.M2 = c.M2;
        p.jones = {cplx{c.jones[0], c.jones[1]}, cplx{c.jones[2], c.jones[3]}};
        failing.push_back({"Rayleigh 异常（出射区 kz = 0）", p, rcwa::Status2D::RayleighAnomaly});
    }
    {
        Problem p;
        p.stack = rcwa2d_ref_stack(25);
        const Rcwa2dRefCase& c = rcwa2d_ref_case(25);
        p.wavelength = c.wavelength; p.theta = c.theta_deg; p.phi = c.phi_deg; p.M1 = c.M1; p.M2 = c.M2;
        p.jones = {cplx{c.jones[0], c.jones[1]}, cplx{c.jones[2], c.jones[3]}};
        failing.push_back({"Rayleigh 异常（第 2 层 kz = 0）", p, rcwa::Status2D::RayleighAnomaly});
    }
    for (const Failing& f : failing) {
        const rcwa::FieldResult2D run = solve(f.problem);
        CHECK(run.result.status == f.status);
        CHECK(run.fields.status() == rcwa::FieldContextStatus::Unavailable && !run.fields.available());
        CHECK_THROWS_UNAVAILABLE(run.fields.require_available());
        CHECK_THROWS_UNAVAILABLE(rcwa::field_at(run, 0.1));
        CHECK_THROWS_UNAVAILABLE(rcwa::field_at(run, RealVector{-0.1, 0.2}));
        CHECK_THROWS_UNAVAILABLE(rcwa::field_at(run.fields, RealVector{}));
        CHECK_THROWS_UNAVAILABLE(rcwa::field_slice(run, RealVector{0.1}, rcwa::SlicePlane::XZ, 8, 0.1));
        CHECK_THROWS_UNAVAILABLE(rcwa::field_slice(run, RealVector{0.1}, rcwa::SlicePlane::YZ, 8, 0.1));
        CHECK_THROWS_UNAVAILABLE(rcwa::phase_slice(run, RealVector{0.1}, rcwa::SlicePlane::YZ, 8, 0.1, rcwa::FieldComponent::Ez));
        // 参数非法先于不可用检查（纯参数检查不依赖上下文）
        CHECK_THROWS(rcwa::field_slice(run, RealVector{0.1}, rcwa::SlicePlane::XZ, 0, 0.1), std::invalid_argument);
        // 元数据仍给出输入决定的波矢，Result2D 与不带 capture 的 solve2d 逐位相同（NaN 填充）
        CHECK(run.fields.metadata().kxm.size() == static_cast<std::size_t>(run.result.N) && run.fields.metadata().kzI.size() == run.fields.metadata().kxm.size());
        CHECK(same_result(run.result, solve_plain(f.problem)));
        bool nan_filled = !run.result.Rs.empty();
        for (cplx v : run.result.Rs) nan_filled = nan_filled && std::isnan(v.real());
        CHECK(nan_filled);
        say("  %-44s status = %d，上下文 Unavailable，field_at / field_slice / phase_slice 抛 logic_error\n", f.name, static_cast<int>(run.result.status));
    }
    // 参数非法：与 solve2d 一样抛 invalid_argument（不返回任何对象）
    Problem bad = problem_a();
    bad.M1 = -1;
    CHECK_THROWS(solve(bad), std::invalid_argument);
    bad = problem_a();
    bad.theta = 95.0;
    CHECK_THROWS(solve(bad), std::invalid_argument);
}

static void test_ownership_and_boundaries() {
    say("[区域与层的归属：边界 z、±1e-12 带、零厚层、无层栈]\n");
    // 三层均匀膜 n = 1.6、2.4、1.3：(0,0) 阶的 D_z = Kx·Hy − Ky·Hx（跨界面不变）；Ez·n² 与它比较
    Problem p;
    p.stack.nI = 1.0;
    p.stack.nII = 1.9;
    p.stack.Lx = p.stack.Ly = 0.9;
    p.stack.layers = {rcwa::Layer2D::uniform(0.3, 1.6), rcwa::Layer2D::uniform(0.2, 2.4), rcwa::Layer2D::uniform(0.25, 1.3)};
    p.theta = 40.0;
    p.phi = 20.0;
    p.M1 = p.M2 = 1;
    p.jones = {cplx{0.6, 0.0}, cplx{0.0, 0.8}};
    const rcwa::FieldResult2D run = solve(p);
    CHECK(run.fields.available());
    const std::vector<double> n2{1.0, 1.6 * 1.6, 2.4 * 2.4, 1.3 * 1.3, 1.9 * 1.9};        // 区域 I、三层、区域 II
    const RealVector zi = interfaces_of(p.stack);                                              // 0、0.3、0.5、0.75
    // 同一点的 D = Kx·Hy − Ky·Hx（H 切向连续，D 不依赖介质）；Ez·n² 只有取到该点实际所在介质的 n² 才与它相等
    const auto owner_at = [&](double z) {
        const rcwa::HarmonicFields h = rcwa::field_at(run, z);
        const std::size_t k = static_cast<std::size_t>(run.result.i0);
        const cplx d = run.result.Kx[k] * h.component(rcwa::FieldComponent::Hy)(run.result.i0, 0) - run.result.Ky[k] * h.component(rcwa::FieldComponent::Hx)(run.result.i0, 0);
        const cplx ez = h.component(rcwa::FieldComponent::Ez)(run.result.i0, 0);
        int owner = -1;
        double best = 1e300;
        for (std::size_t j = 0; j < n2.size(); ++j) {
            const double diff = std::abs(ez * n2[j] - d);
            if (diff < best) { best = diff; owner = static_cast<int>(j); }
        }
        return best <= 1e-9 * std::abs(d) ? owner : -1;
    };
    const double zb = zi.back();
    struct Probe { double z; int owner; const char* what; };
    const Probe probes[] = {
        {-1e-300, 0, "z = −1e-300 → 区域 I"},
        {-1e-15, 0, "z = −1e-15 → 区域 I"},
        {0.0, 1, "z = 0 → 第一层"},
        {-0.0, 1, "z = −0.0 → 第一层（−0.0 不 < 0）"},
        {1e-15, 1, "z = 1e-15 → 第一层"},
        {zi[1] - 1e-13, 1, "第一界面 −1e-13 → 上层"},
        {zi[1], 1, "第一界面 → 上层"},
        {zi[1] + 5e-13, 1, "第一界面 +5e-13 → 上层（≤ 1e-12 带）"},
        {zi[1] + 2e-12, 2, "第一界面 +2e-12 → 下层"},
        {zi[1] + 1e-9, 2, "第一界面 +1e-9 → 下层"},
        {zi[2] - 1e-9, 2, "第二界面 −1e-9 → 上层"},
        {zi[2], 2, "第二界面 → 上层"},
        {zi[2] + 9e-13, 2, "第二界面 +9e-13 → 上层"},
        {zi[2] + 3e-12, 3, "第二界面 +3e-12 → 下层"},
        {zb - 1e-13, 3, "z_bot −1e-13 → 最后一层"},
        {zb, 3, "z = z_bot → 最后一层"},
        {zb + 1e-13, 4, "z_bot +1e-13 → 区域 II"},
        {zb + 1e-12, 4, "z_bot +1e-12 → 区域 II"},
        {zb + 1e-9, 4, "z_bot +1e-9 → 区域 II"},
        {zb + 3.0, 4, "z_bot +3 → 区域 II"},
    };
    int wrong = 0;
    for (const Probe& pr : probes) {
        const int got = owner_at(pr.z);
        CHECK(got == pr.owner);
        if (got != pr.owner) { ++wrong; say("    归属错误：%s（得到 %d，应为 %d）\n", pr.what, got, pr.owner); }
    }
    say("  %zu 个边界探针（用 Ez·n² 与 Kx·Hy − Ky·Hx 相符识别介质），归属错误 %d 个\n", sizeof probes / sizeof probes[0], wrong);

    // 无层栈：z = 0 取出射侧（Python 此处抛 StopIteration）；与 z = 1e-300（区域 II，指数恰为 1）逐位相同
    {
        Problem q;
        q.stack.nI = 1.0;
        q.stack.nII = 1.5;
        q.stack.Lx = q.stack.Ly = 0.8;
        q.M1 = q.M2 = 1;
        q.jones = rcwa::jones_from_angles_2d(40.0);
        const rcwa::FieldResult2D empty = solve(q);
        CHECK(empty.fields.available() && empty.fields.metadata().layers.empty() && empty.fields.metadata().z_bottom == 0.0);
        const rcwa::HarmonicFields h0 = rcwa::field_at(empty, 0.0), h1 = rcwa::field_at(empty, 1e-300), hm = rcwa::field_at(empty, -0.0);
        CHECK(same_column(h0, 0, hm, 0));                  // z = 0 与 z = −0.0 逐位相同
        for (std::size_t c = 0; c < 6; ++c)                 // z = 1e-300 的指数是 1 − O(1e-300)：数值上相同
            for (int j = 0; j < h0.component(rcwa::FieldComponent::Ex).rows(); ++j)
                CHECK(std::abs(h0.components[c](j, 0) - h1.components[c](j, 0)) <= 1e-15 * (1.0 + std::abs(h0.components[c](j, 0))));
        const rcwa::HarmonicFields hneg = rcwa::field_at(empty, -1e-300);
        // 区域 II 的 Ez 用 1/nII²，区域 I 用 1/nI²：两侧的 Ez 差 nII² / nI² 倍
        const std::size_t i0 = static_cast<std::size_t>(empty.result.i0);
        const cplx ez_out = h0.component(rcwa::FieldComponent::Ez)(static_cast<int>(i0), 0), ez_in = hneg.component(rcwa::FieldComponent::Ez)(static_cast<int>(i0), 0);
        CHECK(std::abs(ez_in / ez_out - cplx{1.5 * 1.5, 0.0}) <= 1e-12);
        // 单界面的区域 II 场就是透射波：横向电场 = T
        CHECK(std::abs(h0.component(rcwa::FieldComponent::Ex)(static_cast<int>(i0), 0) - empty.result.Ex_t[i0]) <= 1e-14);
        CHECK(std::abs(h0.component(rcwa::FieldComponent::Ey)(static_cast<int>(i0), 0) - empty.result.Ey_t[i0]) <= 1e-14);
        say("  无层栈：z = 0 与 z = −0.0 取出射侧（与 z = 1e-300 的区域 II 值相同）\n");
    }
    // 零厚层：Python 规则 z ≤ z_top + d + 1e-12，第一个满足的层。排在上一层之后的零厚层永远不会被选中；第一层若是零厚层，z ∈ [0, 1e-12] 取它
    {
        Problem q;
        q.stack.nI = 1.0;
        q.stack.nII = 1.9;
        q.stack.Lx = q.stack.Ly = 0.9;
        q.stack.layers = {rcwa::Layer2D::uniform(0.3, 1.6), rcwa::Layer2D::uniform(0.0, 2.8), rcwa::Layer2D::uniform(0.0, 3.1), rcwa::Layer2D::uniform(0.2, 1.3)};
        q.theta = 40.0;
        q.phi = 20.0;
        q.M1 = q.M2 = 1;
        q.jones = {cplx{0.6, 0.0}, cplx{0.0, 0.8}};
        const rcwa::FieldResult2D zr = solve(q);
        CHECK(zr.fields.available() && zr.fields.metadata().layers.size() == 4 && zr.fields.metadata().layers[1].thickness == 0.0);
        // 零厚层照常保留（二维不过滤）：有模式、不传播，下标不变；z 位置与上一层的层底相同
        CHECK(zr.fields.metadata().layers[1].has_modes && !zr.fields.metadata().layers[1].propagates && zr.fields.metadata().layers[1].effective_index == 1);
        CHECK(zr.fields.metadata().layers[1].z_top == 0.3 && zr.fields.metadata().layers[1].z_bottom == 0.3 && zr.fields.metadata().layers[3].z_top == 0.3);
        CHECK(zr.fields.metadata().layers[0].propagates && zr.fields.metadata().layers[3].propagates);
        const std::vector<double> m2{1.0, 1.6 * 1.6, 2.8 * 2.8, 3.1 * 3.1, 1.3 * 1.3, 1.9 * 1.9};
        const auto owner = [&](double z) {
            const rcwa::HarmonicFields h = rcwa::field_at(zr, z);
            const std::size_t k = static_cast<std::size_t>(zr.result.i0);
            const cplx d = zr.result.Kx[k] * h.component(rcwa::FieldComponent::Hy)(zr.result.i0, 0) - zr.result.Ky[k] * h.component(rcwa::FieldComponent::Hx)(zr.result.i0, 0);
            const cplx ez = h.component(rcwa::FieldComponent::Ez)(zr.result.i0, 0);
            int best = -1;
            double bd = 1e300;
            for (std::size_t j = 0; j < m2.size(); ++j) {
                const double diff = std::abs(ez * m2[j] - d);
                if (diff < bd) { bd = diff; best = static_cast<int>(j); }
            }
            return bd <= 1e-9 * std::abs(d) ? best : -1;
        };
        // 介质编号：0 区域 I，1 第一层 n=1.6，2、3 两个零厚层，4 第 4 层 n=1.3，5 区域 II
        CHECK(owner(0.15) == 1);
        CHECK(owner(0.3) == 1);                    // 界面（上层，零厚层不被选中）
        CHECK(owner(0.3 + 5e-13) == 1);
        CHECK(owner(0.3 + 3e-12) == 4);            // 越过 1e-12 带：直接到第 4 层，两个零厚层都被跳过
        CHECK(owner(0.4) == 4);
        CHECK(owner(0.5) == 4);                    // z_bot
        CHECK(owner(0.5 + 1e-13) == 5);
        // 零厚层两侧：z = 0.3 ∓ 1e-9 的切向场连续（界面集合里有重复的 0.3）
        const ContinuityResult cr = interface_continuity(q, zr);
        CHECK_LE(cr.tangential, 1e-7);
        CHECK_LE(cr.normal_d, 1e-7);
        say("  零厚层夹在中间：只被第一个满足条件的层取走，零厚层永不被选中；界面连续 %.2e\n", cr.tangential);
        // 第一层是零厚层：z ∈ [0, 1e-12] 取零厚层（Python 的规则，一维“全部为零厚才取出射侧”的规则不同）
        Problem lead = q;
        lead.stack.layers = {rcwa::Layer2D::uniform(0.0, 2.8), rcwa::Layer2D::uniform(0.3, 1.6), rcwa::Layer2D::uniform(0.2, 1.3)};
        const rcwa::FieldResult2D lr = solve(lead);
        CHECK(lr.fields.available());
        const std::vector<double> l2{1.0, 2.8 * 2.8, 1.6 * 1.6, 1.3 * 1.3, 1.9 * 1.9};
        const auto lead_owner = [&](double z) {
            const rcwa::HarmonicFields h = rcwa::field_at(lr, z);
            const std::size_t k = static_cast<std::size_t>(lr.result.i0);
            const cplx d = lr.result.Kx[k] * h.component(rcwa::FieldComponent::Hy)(lr.result.i0, 0) - lr.result.Ky[k] * h.component(rcwa::FieldComponent::Hx)(lr.result.i0, 0);
            const cplx ez = h.component(rcwa::FieldComponent::Ez)(lr.result.i0, 0);
            int best = -1;
            double bd = 1e300;
            for (std::size_t j = 0; j < l2.size(); ++j) {
                const double diff = std::abs(ez * l2[j] - d);
                if (diff < bd) { bd = diff; best = static_cast<int>(j); }
            }
            return bd <= 1e-9 * std::abs(d) ? best : -1;
        };
        CHECK(lead_owner(-1e-9) == 0);
        CHECK(lead_owner(0.0) == 1);               // 第一层是零厚层：z = 0 取它（Python 规则）
        CHECK(lead_owner(1e-12) == 1);
        CHECK(lead_owner(2e-12) == 2);
        CHECK(lead_owner(0.1) == 2);
        say("  第一层为零厚层：z ∈ [0, 1e-12] 取它（Python 规则），其后取下一层\n");
    }
}

static void test_depth_semantics() {
    say("[z 的顺序、重复、空输入、批量分块与非法参数]\n");
    Problem p = problem_a();
    p.M1 = p.M2 = 1;
    const rcwa::FieldResult2D run = solve(p);
    const RealVector zs{0.5, -0.2, 0.1, 0.1, 0.9, 0.35, -0.2, 0.0, 0.1, 0.3, 0.725, 0.2};
    const rcwa::HarmonicFields h = rcwa::field_at(run, zs);
    CHECK(h.z == zs && h.nz() == zs.size() && h.norders() == 9 && h.orders.size() == 9);
    // 每个 z 单独采样的结果与批量里的对应列逐位相同；重复值逐位相同
    bool single_same = true;
    for (std::size_t k = 0; k < zs.size(); ++k) single_same = single_same && same_column(rcwa::field_at(run, zs[k]), 0, h, k);
    CHECK(single_same);
    CHECK(same_column(h, 2, h, 3) && same_column(h, 2, h, 8) && same_column(h, 1, h, 6));
    // 排列：打乱顺序后各列对应相同
    RealVector perm(zs.rbegin(), zs.rend());
    const rcwa::HarmonicFields hp = rcwa::field_at(run, perm);
    bool perm_same = true;
    for (std::size_t k = 0; k < zs.size(); ++k) perm_same = perm_same && same_column(hp, zs.size() - 1 - k, h, k);
    CHECK(perm_same);
    // 对 FieldResult2D 与上下文的重载一致
    CHECK(same_harmonics(rcwa::field_at(run, zs), rcwa::field_at(run.fields, zs)));
    // 元数据与 orders
    CHECK(h.metadata.kind == rcwa::FieldSolveKind::Periodic2D && h.metadata.orders_m.size() == 9 && h.metadata.orders_n.size() == 9);
    for (int m = -1; m <= 1; ++m)
        for (int n = -1; n <= 1; ++n) {
            const std::size_t j = static_cast<std::size_t>(rcwa::harmonic_index_2d(m, n, 1, 1));
            CHECK(h.orders[j] == static_cast<int>(j) && h.metadata.orders_m[j] == m && h.metadata.orders_n[j] == n);
            CHECK(run.result.m[j] == m && run.result.n[j] == n);
        }
    // field_slice 的默认参数：nt = 256，at = 0
    {
        const rcwa::FieldGridSlice expl = rcwa::field_slice(run, RealVector{0.1}, rcwa::SlicePlane::YZ, 256, 0.0);
        const rcwa::FieldGridSlice dflt_result = rcwa::field_slice(run, RealVector{0.1}, rcwa::SlicePlane::YZ);          // FieldResult2D 的重载
        const rcwa::FieldGridSlice dflt_context = rcwa::field_slice(run.fields, RealVector{0.1}, rcwa::SlicePlane::YZ);  // 上下文的重载
        CHECK(dflt_result.nt() == 256 && dflt_result.slice_at == 0.0 && same_matrix(dflt_result.components[2], expl.components[2]));
        CHECK(dflt_context.nt() == 256 && dflt_context.slice_at == 0.0 && same_matrix(dflt_context.components[2], expl.components[2]));
    }
    // 由谐波列重建：与 field_slice(context, zs, …) 逐位相同；参数检查同样先于计算
    {
        const RealVector zq{-0.2, 0.1, 0.5};
        const rcwa::HarmonicFields hq = rcwa::field_at(run, zq);
        for (const rcwa::SlicePlane plane : {rcwa::SlicePlane::XZ, rcwa::SlicePlane::YZ}) {
            const rcwa::FieldGridSlice direct = rcwa::field_slice(run, zq, plane, 17, 0.21), shared = rcwa::field_slice(hq, plane, 17, 0.21);
            bool same = shared.plane == plane && shared.z == zq && shared.slice_at == 0.21;
            for (std::size_t c = 0; c < 6; ++c) same = same && same_matrix(direct.components[c], shared.components[c]);
            CHECK(same);
        }
        CHECK_THROWS(rcwa::field_slice(hq, rcwa::SlicePlane::XZ, 0, 0.1), std::invalid_argument);
        CHECK_THROWS(rcwa::field_slice(hq, rcwa::SlicePlane::YZ, 8, kNaN), std::invalid_argument);
        CHECK_THROWS(rcwa::field_slice(hq, static_cast<rcwa::SlicePlane>(3), 8, 0.1), std::invalid_argument);
        CHECK(rcwa::field_slice(hq, rcwa::SlicePlane::XZ).nt() == 256);
    }
    // 空 zs
    const rcwa::HarmonicFields he = rcwa::field_at(run, RealVector{});
    CHECK(he.nz() == 0 && he.norders() == 9);
    for (const Matrix& m : he.components) CHECK(m.rows() == 9 && m.cols() == 0);
    const rcwa::FieldGridSlice se = rcwa::field_slice(run, RealVector{}, rcwa::SlicePlane::YZ, 11, 0.2);
    CHECK(se.nz() == 0 && se.nt() == 11 && se.plane == rcwa::SlicePlane::YZ);
    for (const Matrix& m : se.components) CHECK(m.rows() == 11 && m.cols() == 0);
    // 层内距离夹到 [0, d]：1e-12 带内、仍属于上一层的 z，场就是界面上的值（不向层外外推；Python 不夹，会差 ~1e-11）
    {
        const RealVector zi = interfaces_of(p.stack);
        double worst = 0.0;
        for (std::size_t i = 1; i + 1 < zi.size(); ++i) {
            const rcwa::HarmonicFields hb = rcwa::field_at(run, RealVector{zi[i], zi[i] + 9e-13, zi[i] + 1e-12});
            for (std::size_t c = 0; c < 6; ++c)
                for (int j = 0; j < hb.components[c].rows(); ++j) {
                    const double scale = 1.0 + std::abs(hb.components[c](j, 0));
                    worst = (std::max)({worst, std::abs(hb.components[c](j, 1) - hb.components[c](j, 0)) / scale, std::abs(hb.components[c](j, 2) - hb.components[c](j, 0)) / scale});
                }
        }
        CHECK_LE(worst, 1e-14);
        say("  1e-12 带内属于上一层的 z：场与界面上的值相差 %.1e（夹到 [0, d]，不外推）\n", worst);
    }
    // 批量分块（同一层或区域里每 256 个 z 一批）：区域 I 700 个、区域 II 300 个、第一层 520 个、第三层 90 个，打乱顺序；
    // 每一列都与单独采样的结果逐位相同（任何一批丢列、错位都会被发现）
    RealVector many;
    for (int k = 0; k < 700; ++k) many.push_back(-0.5 + 0.499 * (k + 0.5) / 700.0);
    for (int k = 0; k < 300; ++k) many.push_back(0.725 + 0.01 + 1.0 * (k + 0.5) / 300.0);
    for (int k = 0; k < 520; ++k) many.push_back(0.3 * (k + 0.5) / 520.0);
    for (int k = 0; k < 90; ++k) many.push_back(0.415 + 0.31 * (k + 0.5) / 90.0);
    {
        RealVector shuffled(many.size());
        for (std::size_t k = 0; k < many.size(); ++k) shuffled[(k * 37) % many.size()] = many[k];      // 1610 = 2·5·7·23 与 37 互素
        many.swap(shuffled);
    }
    const rcwa::HarmonicFields hm = rcwa::field_at(run, many);
    CHECK(hm.nz() == 1610 && hm.z == many);
    bool chunk_same = true;
    for (std::size_t k = 0; k < many.size(); ++k) chunk_same = chunk_same && same_column(rcwa::field_at(run, many[k]), 0, hm, k);
    CHECK(chunk_same);
    // 非法参数
    const double inf = std::numeric_limits<double>::infinity();
    CHECK_THROWS(rcwa::field_at(run, kNaN), std::invalid_argument);
    CHECK_THROWS(rcwa::field_at(run, inf), std::invalid_argument);
    CHECK_THROWS(rcwa::field_at(run, -inf), std::invalid_argument);
    CHECK_THROWS(rcwa::field_at(run, RealVector{0.1, kNaN, 0.2}), std::invalid_argument);
    CHECK_THROWS(rcwa::field_slice(run, RealVector{0.1, inf}, rcwa::SlicePlane::XZ, 8, 0.1), std::invalid_argument);
    CHECK_THROWS(rcwa::field_slice(run, RealVector{0.1}, rcwa::SlicePlane::XZ, 0, 0.1), std::invalid_argument);
    CHECK_THROWS(rcwa::field_slice(run, RealVector{0.1}, rcwa::SlicePlane::YZ, -3, 0.1), std::invalid_argument);
    CHECK_THROWS(rcwa::field_slice(run, RealVector{0.1}, rcwa::SlicePlane::XZ, 8, kNaN), std::invalid_argument);
    CHECK_THROWS(rcwa::field_slice(run, RealVector{0.1}, rcwa::SlicePlane::YZ, 8, inf), std::invalid_argument);
    CHECK_THROWS(rcwa::field_slice(run, RealVector{0.1}, static_cast<rcwa::SlicePlane>(2), 8, 0.1), std::invalid_argument);
    CHECK_THROWS(rcwa::field_slice(run, RealVector{0.1}, static_cast<rcwa::SlicePlane>(-1), 8, 0.1), std::invalid_argument);
    CHECK_THROWS(rcwa::phase_slice(run, RealVector{0.1}, rcwa::SlicePlane::XZ, 8, 0.1, std::string("E2")), std::invalid_argument);
    CHECK_THROWS(rcwa::phase_slice(run, RealVector{0.1}, rcwa::SlicePlane::XZ, 8, 0.1, std::string("Ez"), [] { rcwa::PhaseOptions o; o.phase_floor = -1.0; return o; }()),
                 std::invalid_argument);
    // |z| 大到 k0·z 溢出：不返回貌似有效的场
    CHECK_THROWS(rcwa::field_at(run, 1e308), std::runtime_error);
    CHECK_THROWS(rcwa::field_at(run, -1e308), std::runtime_error);
    // 很深的 z 是有限的（指数衰减到 0，传播级只有相位）
    const rcwa::HarmonicFields deep = rcwa::field_at(run, RealVector{-1e4, 1e4});
    for (const Matrix& m : deep.components) CHECK(m.all_finite());
    // 相位便捷入口：字符串与枚举两种写法一致；非法分量名抛 invalid_argument
    const rcwa::PhaseGrid a = rcwa::phase_slice(run, zs, rcwa::SlicePlane::XZ, 9, 0.2, rcwa::FieldComponent::Hy);
    const rcwa::PhaseGrid b = rcwa::phase_slice(run, zs, rcwa::SlicePlane::XZ, 9, 0.2, std::string("hy"));
    CHECK(a.nx == 9 && a.nz == zs.size() && a.plane == rcwa::SlicePlane::XZ && a.component == "Hy" && a.mask == b.mask);
    bool same = a.phase.size() == b.phase.size();
    for (std::size_t k = 0; same && k < a.phase.size(); ++k) same = same_bits(a.phase[k], b.phase[k]);
    CHECK(same);
}

// 边角栈：只有零阶（N = 1）、全部为零厚层
static void test_edge_stacks() {
    say("[边角栈：N = 1（M1 = M2 = 0）、全部为零厚层]\n");
    {   // N = 1：结构层退化成有效均匀介质，各检验照常成立
        Problem p = problem_a();
        p.M1 = p.M2 = 0;
        const rcwa::FieldResult2D run = solve(p);
        CHECK(run.fields.available() && run.result.N == 1);
        if (run.fields.available()) {
            const ContinuityResult cr = interface_continuity(p, run);
            CHECK_LE(cr.tangential, 1e-7);
            CHECK_LE(cr.normal_d, 1e-7);
            const rcwa::HarmonicFields h = rcwa::field_at(run, RealVector{-0.3, 0.1, 0.5, 1.2});
            CHECK(h.norders() == 1 && h.components[0].rows() == 1 && h.components[0].cols() == 4);
            const double inc = p.stack.nI * std::cos(p.theta * kPi / 180.0);
            for (std::size_t k = 0; k < 4; ++k) CHECK_LE(std::abs(poynting_z(h, k) - inc * run.result.transmittance()), 1e-12);
            const rcwa::FieldGridSlice g = rcwa::field_slice(run, RealVector{0.1, 0.5}, rcwa::SlicePlane::YZ, 5, 0.3);
            CHECK(g.nt() == 5 && g.nz() == 2 && g.metadata.orders_m == std::vector<int>{0});
            say("  N = 1：界面连续 %.2e，Sz 与 z 无关\n", cr.tangential);
        }
    }
    {   // 全部为零厚层（Python 的规则：z = 0 取第一层；z > 0 取出射区）：与出射区、层的 D 一致
        Problem p;
        p.stack.nI = 1.0;
        p.stack.nII = 1.9;
        p.stack.Lx = p.stack.Ly = 0.9;
        p.stack.layers = {rcwa::Layer2D::uniform(0.0, 2.8), rcwa::Layer2D::uniform(0.0, 1.6)};
        p.theta = 40.0;
        p.phi = 20.0;
        p.M1 = p.M2 = 1;
        p.jones = {cplx{0.6, 0.0}, cplx{0.0, 0.8}};
        const rcwa::FieldResult2D run = solve(p);
        CHECK(run.fields.available() && run.fields.metadata().z_bottom == 0.0);
        const std::vector<double> m2{1.0, 2.8 * 2.8, 1.6 * 1.6, 1.9 * 1.9};
        const auto owner = [&](double z) {
            const rcwa::HarmonicFields h = rcwa::field_at(run, z);
            const std::size_t k = static_cast<std::size_t>(run.result.i0);
            const cplx d = run.result.Kx[k] * h.component(rcwa::FieldComponent::Hy)(run.result.i0, 0) - run.result.Ky[k] * h.component(rcwa::FieldComponent::Hx)(run.result.i0, 0);
            const cplx ez = h.component(rcwa::FieldComponent::Ez)(run.result.i0, 0);
            int best = -1;
            double bd = 1e300;
            for (std::size_t j = 0; j < m2.size(); ++j) {
                const double diff = std::abs(ez * m2[j] - d);
                if (diff < bd) { bd = diff; best = static_cast<int>(j); }
            }
            return bd <= 1e-9 * std::abs(d) ? best : -1;
        };
        CHECK(owner(-1e-9) == 0);
        CHECK(owner(0.0) == 1);          // Python 的规则：第一个满足 z ≤ z_top + d + 1e-12 的层（一维“全部为零厚时取出射侧”不同）
        CHECK(owner(-0.0) == 1);
        CHECK(owner(5e-13) == 3);        // z > z_bot = 0 先判为区域 II（外部优先）：1e-12 的带只在 z ≤ z_bot 的层循环里起作用
        CHECK(owner(0.3) == 3);
        say("  全部零厚层：z = 0 取第一层（Python 的规则），任何 z > 0 取出射区\n");
    }
}

// 元数据（波矢、归一化、层几何）
static void test_metadata() {
    say("[元数据]\n");
    Problem p = problem_b();
    p.jones = {cplx{3.0, 0.0}, cplx{0.0, 4.0}};      // 范数 5，不归一
    const rcwa::FieldResult2D run = solve(p);
    CHECK(run.fields.available());
    const rcwa::FieldMetadata& m = run.fields.metadata();
    const rcwa::Result2D& r = run.result;
    CHECK(m.kind == rcwa::FieldSolveKind::Periodic2D && m.incident_basis_columns == 1);
    CHECK(m.wavelength == p.wavelength && m.theta_deg == p.theta && m.phi_deg == p.phi);
    CHECK(m.k0 == r.k0 && std::abs(m.k0 - 2.0 * kPi / p.wavelength) <= 1e-15);
    CHECK(m.period == p.stack.Lx && m.period_y == p.stack.Ly && m.truncation_order == p.M1 && m.truncation_order_y == p.M2);
    CHECK((m.nI == cplx{p.stack.nI, 0.0} && m.nII == p.stack.nII));
    CHECK(std::abs(m.normalization.electric_incident_amplitude - 5.0) <= 1e-15 && std::abs(m.normalization.magnetic_incident_amplitude - 5.0 * p.stack.nI) <= 1e-14);
    CHECK(m.orders.size() == static_cast<std::size_t>(r.N) && m.kxm.size() == m.orders.size() && m.kyn.size() == m.orders.size() &&
          m.kzI.size() == m.orders.size() && m.kzII.size() == m.orders.size());
    double err = 0.0;
    for (std::size_t j = 0; j < m.orders.size(); ++j) {
        CHECK(m.orders[j] == static_cast<int>(j) && m.orders_m[j] == r.m[j] && m.orders_n[j] == r.n[j]);
        const double kx = r.k0 * r.Kx[j], ky = r.k0 * r.Ky[j];       // 先各自舍入成 double（否则编译器可能把乘法融进减法）
        const cplx kzI = r.k0 * r.kzI[j], kzII = r.k0 * r.kzII[j];
        err = (std::max)({err, std::abs(m.kxm[j] - kx), std::abs(m.kyn[j] - ky), std::abs(m.kzI[j] - kzI), std::abs(m.kzII[j] - kzII)});
        CHECK(m.kxm[j].imag() == 0.0);
    }
    CHECK(err == 0.0 && m.ky == m.kyn[static_cast<std::size_t>(r.i0)]);
    CHECK(m.layers.size() == 3 && std::abs(m.z_bottom - (0.1 + 0.28 + 0.2)) <= 1e-15);
    const bool homog[3] = {true, false, true};
    for (std::size_t j = 0; j < 3; ++j) {
        const rcwa::FieldLayerInfo& l = m.layers[j];
        CHECK(l.original_index == j && l.effective_index == static_cast<int>(j) && l.homogeneous == homog[j] && l.has_modes && l.propagates);
        CHECK(l.analytic_modes == homog[j] && l.mode_count == 2 * r.N && l.propagation == rcwa::FieldPropagation::Exponential);
        CHECK(std::abs(l.thickness - p.stack.layers[j].d) == 0.0);
    }
    CHECK(m.layers[1].z_top == m.layers[0].z_bottom && m.layers[2].z_top == m.layers[1].z_bottom);
    // 入射电场与磁场的范数：不归一 Jones 时入射 |E| = 5，|Z0 H| = 5 nI（直接从区域 I 远处零阶的 Ey 与 Hx 读不出范数，改读入射矢量范数）
    // 范数 = 谐波总和：z → −∞ 方向取入射项，用一个单界面栈验证
    Problem q;
    q.stack.nI = 1.3;
    q.stack.nII = 1.7;
    q.stack.Lx = q.stack.Ly = 0.9;
    q.M1 = q.M2 = 0;
    q.theta = 31.0;
    q.phi = 57.0;
    q.jones = {cplx{3.0, 1.0}, cplx{-1.0, 2.0}};
    const rcwa::FieldResult2D one = solve(q);
    CHECK(one.fields.available());
    // 无反射的界面是不存在的；但 z = −0.1 处的总场 = 入射 + 反射，入射范数从 inc 反推：|Ex_inc|² + |Ey_inc|² + |Ez_inc|² = |Jones|²
    const double jn = std::hypot(std::abs(q.jones[0]), std::abs(q.jones[1]));
    CHECK(std::abs(one.fields.metadata().normalization.electric_incident_amplitude - jn) <= 1e-15 &&
          std::abs(one.fields.metadata().normalization.magnetic_incident_amplitude - 1.3 * jn) <= 1e-14);
    const double th = q.theta * kPi / 180.0, ph = q.phi * kPi / 180.0;
    const cplx ex = -q.jones[0] * std::sin(ph) + q.jones[1] * std::cos(th) * std::cos(ph), ey = q.jones[0] * std::cos(ph) + q.jones[1] * std::cos(th) * std::sin(ph);
    const cplx ez = -q.jones[1] * std::sin(th);
    CHECK(std::abs(std::sqrt(std::norm(ex) + std::norm(ey) + std::norm(ez)) - jn) <= 1e-14);
    // Jones 不归一：Jones 乘 2（二进制精确）则全部场逐位翻倍，归一化范数翻倍；乘 i 则全部场乘 i（仅差舍入）
    {
        Problem a = problem_a();
        a.jones = {cplx{0.6, 0.0}, cplx{0.0, 0.8}};
        Problem b = a, c = a;
        b.jones = {2.0 * a.jones[0], 2.0 * a.jones[1]};
        c.jones = {kI * a.jones[0], kI * a.jones[1]};
        const rcwa::FieldResult2D ra = solve(a), rb = solve(b), rc = solve(c);
        const RealVector zs{-0.3, 0.1, 0.35, 0.9};
        const rcwa::HarmonicFields ha = rcwa::field_at(ra, zs), hb = rcwa::field_at(rb, zs), hc = rcwa::field_at(rc, zs);
        bool doubled = true;
        double rotated = 0.0;
        for (std::size_t comp = 0; comp < 6; ++comp)
            for (int j = 0; j < ha.components[comp].rows(); ++j)
                for (int k = 0; k < ha.components[comp].cols(); ++k) {
                    doubled = doubled && same_bits(hb.components[comp](j, k), 2.0 * ha.components[comp](j, k));
                    rotated = (std::max)(rotated, std::abs(hc.components[comp](j, k) - kI * ha.components[comp](j, k)));
                }
        CHECK(doubled);
        CHECK_LE(rotated, 1e-14);
        CHECK(rb.fields.metadata().normalization.electric_incident_amplitude == 2.0 * ra.fields.metadata().normalization.electric_incident_amplitude);
        CHECK(std::abs(rc.fields.metadata().normalization.electric_incident_amplitude - ra.fields.metadata().normalization.electric_incident_amplitude) <= 1e-16);
        // 零 Jones：合法输入，零场（不当作失败）
        Problem z = a;
        z.jones = {cplx{0.0, 0.0}, cplx{0.0, 0.0}};
        const rcwa::FieldResult2D rz = solve(z);
        CHECK(rz.fields.available() && rz.fields.metadata().normalization.electric_incident_amplitude == 0.0);
        const rcwa::HarmonicFields hz = rcwa::field_at(rz, zs);
        bool zero = true;
        for (const Matrix& m : hz.components) zero = zero && max_abs(m) == 0.0;
        CHECK(zero);
    }
    say("  kxm、kyn、kzI、kzII = k0·(Kx, Ky, kzI, kzII)（逐位），层几何与模式标志，归一化 {|Jones|, nI|Jones|}；场对 Jones 线性（×2 逐位翻倍）\n");
}

// 线程数：采样（批量矩阵乘法）与切面重建的结果与线程数无关（逐位相同）
static void test_thread_independence() {
    say("[线程数无关：1 线程与 4 线程的采样逐位相同]\n");
    Problem p;
    p.stack = rcwa2d_ref_stack(9);
    p.M1 = p.M2 = 4;
    p.jones = {cplx{1.0, 0.0}, cplx{0.0, 0.0}};
    const int saved = g_threads;
    set_num_threads(1);
    const rcwa::FieldResult2D run = solve(p);
    RealVector zs;
    for (int k = 0; k < 300; ++k) zs.push_back(-0.2 + 0.7 * (k + 0.5) / 300.0);
    const rcwa::HarmonicFields h1 = rcwa::field_at(run, zs);
    const rcwa::FieldGridSlice s1 = rcwa::field_slice(run, zs, rcwa::SlicePlane::XZ, 64, 0.2);
    set_num_threads(4);
    const rcwa::HarmonicFields h4 = rcwa::field_at(run, zs);
    const rcwa::FieldGridSlice s4 = rcwa::field_slice(run, zs, rcwa::SlicePlane::XZ, 64, 0.2);
    const rcwa::FieldResult2D run4 = solve(p);
    set_num_threads(saved);
    CHECK(same_harmonics(h1, h4));
    bool same = true;
    for (std::size_t c = 0; c < 6; ++c) same = same && same_matrix(s1.components[c], s4.components[c]);
    CHECK(same);
    CHECK(same_result(run.result, run4.result) && same_harmonics(h1, rcwa::field_at(run4, zs)));
    say("  300 个 z（N = 81）与 64 点切面：field_at、field_slice 与整个 solve2d_with_fields 在 1、4 线程下逐位相同\n");
}

// e) capture == nullptr 与 solve2d_with_fields 的 Result2D 逐位相同
static void test_result_identical_without_capture() {
    say("[capture == nullptr 与 solve2d_with_fields 的 Result2D 逐位相同]\n");
    int count = 0;
    std::vector<Problem> problems{problem_a(), problem_b(), problem_c()};
    {
        Problem fast_off = problem_a();
        fast_off.options.homogeneous_fast_path = false;
        problems.push_back(fast_off);
        Problem par = problem_a();
        par.options.parallel_layers = true;
        par.options.diagnostics = true;
        problems.push_back(par);
        Problem lau = problem_b();
        lau.options.factorization = rcwa::Factorization2D::Laurent;
        problems.push_back(lau);
        Problem bare;
        bare.stack.nII = 1.5;
        bare.stack.Lx = bare.stack.Ly = 0.8;
        problems.push_back(bare);
    }
    for (const Problem& p : problems) {
        const rcwa::Result2D plain = solve_plain(p);
        const rcwa::FieldResult2D with = solve(p);
        CHECK(plain.status == rcwa::Status2D::Ok);
        CHECK(same_result(plain, with.result));
        // 并行与串行的 capture 结果也相同
        ++count;
    }
    say("  %d 个问题（含稠密路径、并行层、Laurent、无层）逐位相同\n", count);
}

// ---------------------------------------------------------------------------------------------
// 性能（--bench）
static double peak_rss_mb() {
#if defined(__unix__) || defined(__APPLE__)
    struct rusage u;
    if (getrusage(RUSAGE_SELF, &u) != 0) return -1.0;
#if defined(__APPLE__)
    return static_cast<double>(u.ru_maxrss) / (1024.0 * 1024.0);
#else
    return static_cast<double>(u.ru_maxrss) / 1024.0;
#endif
#else
    return -1.0;
#endif
}
static double seconds_since(std::chrono::steady_clock::time_point t0) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }

static void bench_case(int M) {
    Problem p;
    p.stack = rcwa2d_ref_stack(9);          // 0.8×0.8 元胞里 0.3×0.3 的柱，n = 1.8，厚 0.3，基底 1.5
    p.M1 = p.M2 = M;
    p.theta = 42.0;
    p.phi = -25.0;
    p.jones = {cplx{1.0, 0.0}, cplx{0.0, 0.0}};
    const int N = (2 * M + 1) * (2 * M + 1);
    const auto t1 = std::chrono::steady_clock::now();
    const rcwa::FieldResult2D run = solve(p);
    const double t_capture = seconds_since(t1);
    const double rss_solve = peak_rss_mb();
    const auto t0 = std::chrono::steady_clock::now();
    const rcwa::Result2D plain = rcwa::solve2d(p.stack, p.wavelength, p.theta, p.phi, p.M1, p.M2, p.jones, p.options, nullptr);
    const double t_plain = seconds_since(t0);
    CHECK(run.fields.available() && same_result(plain, run.result));
    say("  N = %d（M = %d）：solve2d_with_fields %.2f s（solve2d 不带 capture %.2f s）；上下文 %.1f MB；求解后峰值 RSS %.0f MB\n", N, M, t_capture,
        t_plain, static_cast<double>(run.fields.memory_bytes()) / 1048576.0, rss_solve);
    // 200 个 z 全部落在结构层内（最坏情形：每个 z 都要过 W、V、Ei 的批量乘法）、跨区域各占一部分、以及 nt = 300 的切面
    RealVector in_layer, mixed;
    for (int k = 0; k < 200; ++k) {
        in_layer.push_back(0.3 * (k + 0.5) / 200.0);
        mixed.push_back(-0.5 + 1.4 * (k + 0.5) / 200.0);
    }
    const auto time_field_at = [&](const char* label, const RealVector& zs) {
        const auto ta = std::chrono::steady_clock::now();
        const rcwa::HarmonicFields h = rcwa::field_at(run, zs);
        const double t = seconds_since(ta);
        say("    field_at(%s，%zu 个 z)：%.3f s（输出 %.1f MB）\n", label, zs.size(), t,
            static_cast<double>(6 * static_cast<std::size_t>(N) * zs.size() * sizeof(cplx)) / 1048576.0);
        CHECK(h.nz() == zs.size());
    };
    time_field_at("全部在结构层内", in_layer);
    time_field_at("-0.5…0.9 跨区域", mixed);
    {
        const auto ta = std::chrono::steady_clock::now();
        const rcwa::HarmonicFields h = rcwa::field_at(run, 0.15);
        say("    field_at(单个 z = 0.15)：%.4f s\n", seconds_since(ta));
        CHECK(h.nz() == 1);
    }
    for (const rcwa::SlicePlane plane : {rcwa::SlicePlane::XZ, rcwa::SlicePlane::YZ}) {
        const auto ta = std::chrono::steady_clock::now();
        const rcwa::FieldGridSlice g = rcwa::field_slice(run, in_layer, plane, 300, 0.37);
        const double t = seconds_since(ta);
        const double rss2 = peak_rss_mb();
        say("    field_slice(%s，200 个 z 全在结构层内，nt = 300)：%.3f s（输出 %.1f MB）；采样后峰值 RSS %.0f MB\n", rcwa::slice_plane_name(plane), t,
            static_cast<double>(6 * 300 * 200 * sizeof(cplx)) / 1048576.0, rss2);
        CHECK(g.nt() == 300 && g.nz() == 200);
    }
    {   // x-z 与 y-z 两个面共用一次采样
        const auto ta = std::chrono::steady_clock::now();
        const rcwa::HarmonicFields h = rcwa::field_at(run, in_layer);
        const double t_sample = seconds_since(ta);
        const auto tb = std::chrono::steady_clock::now();
        const rcwa::FieldGridSlice gx = rcwa::field_slice(h, rcwa::SlicePlane::XZ, 300, 0.37), gy = rcwa::field_slice(h, rcwa::SlicePlane::YZ, 300, 0.46);
        say("    x-z 与 y-z 共用一次采样：field_at %.3f s + 两次重建（nt = 300）%.3f s\n", t_sample, seconds_since(tb));
        CHECK(gx.nt() == 300 && gy.nt() == 300);
    }
    {
        const auto ta = std::chrono::steady_clock::now();
        const rcwa::FieldGridSlice g = rcwa::field_slice(run, mixed, rcwa::SlicePlane::XZ, 300, 0.37);
        say("    field_slice(xz，200 个 z 跨区域，nt = 300)：%.3f s\n", seconds_since(ta));
        CHECK(g.nz() == 200);
        const auto tb = std::chrono::steady_clock::now();
        const rcwa::PhaseGrid ph = rcwa::phase_data(g, rcwa::FieldComponent::Ez, [] { rcwa::PhaseOptions o; o.reference = rcwa::PhaseReference::Incident; return o; }());
        say("    phase_data(Ez，incident)：%.3f s\n", seconds_since(tb));
        CHECK(ph.nx == 300);
    }
}

static void bench() {
    say("[性能与内存：n = 0.8×0.8 µm 元胞、0.3×0.3 µm 的柱，线程数 %d]\n", g_threads);
    bench_case(8);
    bench_case(12);
}

// 每组检验各自捕获异常：一组里抛出未预期的异常只记一次失败，不中断其余检验
static void run_test(const char* name, void (*test)()) {
    try {
        test();
    } catch (const std::exception& e) {
        ++g_total;
        say("  失败 %s 抛出了未预期的异常：%s\n", name, e.what());
    } catch (...) {
        ++g_total;
        say("  失败 %s 抛出了未预期的异常\n", name);
    }
    say("\n");
}
#define RUN_TEST(name) run_test(#name, name)

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--big") g_big = true;
        else if (arg == "--bench") g_bench = true;
        else if (arg.rfind("--threads=", 0) == 0) g_threads = (std::max)(1, std::atoi(arg.c_str() + 10));
        else {
            say("未知参数 %s\n用法：test_rcwa2d_fields [--big] [--bench] [--threads=N]\n", arg.c_str());
            return 2;
        }
    }
    set_num_threads(g_threads);
    say("二维近场：rcwa2d_fields.hpp（线程数 %d%s%s）\n\n", g_threads, g_big ? "，--big" : "", g_bench ? "，--bench（只做性能测量）" : "");
    if (g_bench) {            // 只做性能测量：干净的进程里峰值 RSS 才有意义
        RUN_TEST(bench);
        say("通过 %d / 共 %d\n", g_pass, g_total);
        return g_pass == g_total ? 0 : 1;
    }
    RUN_TEST(test_frozen_harmonics);
    RUN_TEST(test_frozen_slices);
    RUN_TEST(test_large_order_region_I);
    RUN_TEST(test_continuity_and_maxwell);
    RUN_TEST(test_airy);
    RUN_TEST(test_exterior_plane_waves);
    RUN_TEST(test_one_d_limits);
    RUN_TEST(test_poynting_flux);
    RUN_TEST(test_xy_exchange);
    RUN_TEST(test_slice_brute_force);
    RUN_TEST(test_context_ownership);
    RUN_TEST(test_failure_unavailable);
    RUN_TEST(test_ownership_and_boundaries);
    RUN_TEST(test_depth_semantics);
    RUN_TEST(test_edge_stacks);
    RUN_TEST(test_metadata);
    RUN_TEST(test_thread_independence);
    RUN_TEST(test_result_identical_without_capture);
    say("通过 %d / 共 %d\n", g_pass, g_total);
    return g_pass == g_total ? 0 : 1;
}
