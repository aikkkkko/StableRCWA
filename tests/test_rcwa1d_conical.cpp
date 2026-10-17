// tests/test_rcwa1d_conical.cpp —— rcwa1d_conical.hpp（一维光栅的圆锥入射）的测试
//
//   c++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -I include tests/test_rcwa1d_conical.cpp -o build/test_rcwa1d_conical
//   c++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -I include tests/test_rcwa1d_conical.cpp -o build/test_rcwa1d_conical_san
//
// 这里管的是接口与实现上的事：各个量的定义和尺寸、参数检查、数值失败的报告方式、选项互换（快速路径与稠密路径、
// 线性求解方法、QR 与 LR 迭代、custom_eig）、结果与线程数逐位无关、扫描等于逐点求解、与参照表逐级对拍，
// 以及圆锥入射特有的奇点（光栅层的 A 出现零本征值）上的警告与失败。
// 参照表在 tests/conical_reference.inc 里：十个算例各级的复 Jones 矩阵，由独立的 Python 求解器给出。
// 以闭式解和 TE/TM 结果为参照的物理测试在 tests/test_rcwa1d_conical_physics.cpp。
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

#include "stablercwa/eig_qr.hpp"
#include "stablercwa/rcwa1d_conical.hpp"

using namespace stablercwa;
namespace rcwa = stablercwa::rcwa;

#include "conical_reference.inc"

// ---------------------------------------------------------------------------------------------
// 输出与计数。文字一律经 say() 输出（用法同 std::printf）：带中文的格式串不直接交给 printf 系列函数，
// 缘由见 tests/test_rcwa1d.cpp 里的说明。
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

// 断言的计数与报错都在下面几个普通函数里；宏只负责取文件名、行号和表达式的文字，宏里不出现中文。
static void check(bool ok, const char* file, int line, const char* text) {
    ++g_total;
    if (ok) ++g_pass;
    else say("  失败 %s:%d  %s\n", file, line, text);
}
// value <= bound；失败时打印实际数值。NaN 也算失败。
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

#define CHECK_LE(value, bound)                                                    \
    do {                                                                          \
        const double v_ = (value), b_ = (bound);                                  \
        check_le(v_, b_, __FILE__, __LINE__, #value);                             \
    } while (0)

#define CHECK_THROWS(expr, exception_type)                                        \
    do {                                                                          \
        bool caught_ = false;                                                     \
        try { (void)(expr); } catch (const exception_type&) { caught_ = true; }   \
        check_throws(caught_, __FILE__, __LINE__, #expr, #exception_type);        \
    } while (0)

// ---------------------------------------------------------------------------------------------
// 测试用的结构（长度单位微米）。参照算例的结构由 conical_reference.inc 里的 conical_ref_stack(编号) 给出：
//   1 单层无吸收光栅；2 单层吸收光栅；3 光栅 + 20 层膜；4 两个光栅层夹一层吸收膜（入射介质 n = 1.2）；10 纯膜系。

// 侧壁倾斜的梯形齿切成 slices 个光栅层，下面两层膜
static rcwa::Stack trapezoid_stack(int slices) {
    rcwa::Stack s;
    s.nI = 1.0;
    s.nII = 1.46;
    s.period = 0.574;
    for (int k = 0; k < slices; ++k) {
        const double w = s.period * (0.2 + 0.25 * (k + 0.5) / slices);
        const double x0 = 0.5 * (s.period - w);
        s.layers.push_back(rcwa::Layer::grating(0.38 / slices, {0.0, x0, x0 + w, s.period}, {1.0, 1.46, 1.0}));
    }
    s.layers.push_back(rcwa::Layer::uniform(0.46438, 1.46));
    s.layers.push_back(rcwa::Layer::uniform(0.115306, 1.96));
    return s;
}

// 四脊大周期光栅（周期 6.3，四条宽度、折射率各不相同的脊），下面两层膜：级次很密，用来把阶数取大。
// 与 test_rcwa1d.cpp 里 ridges_on_mirror() 的光栅层相同，只取它的前两层膜；用在 λ = 0.94，θ = 22°。
static const double kRidgesWavelength = 0.94, kRidgesTheta = 22.0;
static rcwa::Stack ridges_stack() {
    rcwa::Stack s;
    s.nI = 1.0;
    s.nII = 1.50;
    s.period = 6.3;
    s.layers.push_back(rcwa::Layer::grating(0.32, {0.0, 0.20, 1.05, 1.60, 2.05, 2.90, 4.25, 5.10, 5.45, 6.30},
                                            {1.0, 1.52, 1.0, 2.10, 1.0, 1.75, 1.0, 1.38, 1.0}));
    s.layers.push_back(rcwa::Layer::uniform(0.105, 2.25));
    s.layers.push_back(rcwa::Layer::uniform(0.160, 1.48));
    return s;
}

// λ = Λ = 1 的结构：正入射时第 ±m 级的 kx = ∓m，在折射率恰好是整数 m 的介质里这两级恰好掠射
static rcwa::Stack unit_stack(std::vector<rcwa::Layer> layers, cplx nII) {
    rcwa::Stack s;
    s.nI = 1.0;
    s.nII = nII;
    s.period = 1.0;
    s.layers = std::move(layers);
    return s;
}
static rcwa::Layer unit_grating() { return rcwa::Layer::grating(0.3, {0.0, 0.3, 1.0}, {1.5, 1.0}); }

// ---------------------------------------------------------------------------------------------
// 比较用的小工具

static std::array<cplx, 4> entries(const rcwa::Jones& j) { return {j.ss, j.sp, j.ps, j.pp}; }
static std::array<double, 4> entries(const rcwa::JonesEfficiency& e) { return {e.ss, e.sp, e.ps, e.pp}; }

// 取较大者；value 是 NaN 时结果就是 NaN（之后的 CHECK_LE 必然失败）
static double worse(double current, double value) {
    if (std::isnan(current) || std::isnan(value)) return std::numeric_limits<double>::quiet_NaN();
    return value > current ? value : current;
}

// 两个结果各级 Jones 矩阵元之差的最大模
static double jones_gap(const rcwa::ConicalResult& a, const rcwa::ConicalResult& b) {
    if (a.Jr.size() != b.Jr.size() || a.Jt.size() != b.Jt.size()) return std::numeric_limits<double>::infinity();
    double gap = 0.0;
    for (std::size_t j = 0; j < a.Jr.size(); ++j) {
        const std::array<cplx, 4> ar = entries(a.Jr[j]), br = entries(b.Jr[j]), at = entries(a.Jt[j]), bt = entries(b.Jt[j]);
        for (int k = 0; k < 4; ++k) gap = worse(worse(gap, std::abs(ar[k] - br[k])), std::abs(at[k] - bt[k]));
    }
    return gap;
}

template <class T>
static bool same_bytes(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}
// 两个结果逐位相同（计时与计数除外）
static bool same_bits(const rcwa::ConicalResult& a, const rcwa::ConicalResult& b) {
    return a.orders == b.orders && same_bytes(a.kx, b.kx) && std::memcmp(&a.ky, &b.ky, sizeof(double)) == 0 &&
           same_bytes(a.phi_m_deg, b.phi_m_deg) && same_bytes(a.kzI, b.kzI) && same_bytes(a.kzII, b.kzII) &&
           same_bytes(a.flux_r, b.flux_r) && same_bytes(a.flux_t, b.flux_t) && same_bytes(a.Jr, b.Jr) &&
           same_bytes(a.Jt, b.Jt) && same_bytes(a.DEr, b.DEr) && same_bytes(a.DEt, b.DEt) &&
           std::memcmp(&a.min_abs_lambda, &b.min_abs_lambda, sizeof(double)) == 0 &&
           a.degenerate_mode_warning == b.degenerate_mode_warning;
}

// Jones 矩阵和效率是不是全部为 NaN（数值失败时的约定）
static bool all_nan(const rcwa::ConicalResult& r) {
    if (r.Jr.empty() || r.Jr.size() != r.Jt.size() || r.DEr.size() != r.Jr.size() || r.DEt.size() != r.Jr.size()) return false;
    for (std::size_t j = 0; j < r.Jr.size(); ++j) {
        for (const rcwa::Jones* a : {&r.Jr[j], &r.Jt[j]})
            for (const cplx& v : entries(*a))
                if (!std::isnan(v.real()) || !std::isnan(v.imag())) return false;
        for (const rcwa::JonesEfficiency* e : {&r.DEr[j], &r.DEt[j]})
            for (double v : entries(*e))
                if (!std::isnan(v)) return false;
    }
    return true;
}
// 全部有限
static bool all_finite(const rcwa::ConicalResult& r) {
    for (std::size_t j = 0; j < r.Jr.size(); ++j) {
        for (const rcwa::Jones* a : {&r.Jr[j], &r.Jt[j]})
            for (const cplx& v : entries(*a))
                if (!std::isfinite(v.real()) || !std::isfinite(v.imag())) return false;
        for (const rcwa::JonesEfficiency* e : {&r.DEr[j], &r.DEt[j]})
            for (double v : entries(*e))
                if (!std::isfinite(v)) return false;
    }
    return !r.Jr.empty();
}

// 通量幺正性矩阵偏离单位矩阵的程度（无吸收结构应为零）
static double unitarity_defect(const rcwa::ConicalResult& r) {
    const std::array<cplx, 4> u = r.flux_unitarity();
    return worse(worse(std::abs(u[0] - 1.0), std::abs(u[3] - 1.0)), worse(std::abs(u[1]), std::abs(u[2])));
}

// 与参照表逐级比较：各矩阵元 |ΔJ| / max(1, |J|) 的最大值。propagating_only 为真时只看传播级
// （反射看 flux_r > 0，透射看 flux_t > 0）。
static double table_error(const rcwa::ConicalResult& r, const ConicalRefCase& c, bool propagating_only) {
    if (static_cast<int>(r.Jr.size()) != c.count || static_cast<int>(r.Jt.size()) != c.count)
        return std::numeric_limits<double>::infinity();
    double err = 0.0;
    for (int i = 0; i < c.count; ++i) {
        const ConicalRefOrder& o = c.orders[i];
        const int j = o.m + c.M;
        if (j < 0 || j >= c.count) return std::numeric_limits<double>::infinity();
        const std::array<cplx, 4> jr = entries(r.Jr[j]), jt = entries(r.Jt[j]);
        for (int k = 0; k < 4; ++k) {
            const cplx ref_r(o.jr[2 * k], o.jr[2 * k + 1]), ref_t(o.jt[2 * k], o.jt[2 * k + 1]);
            if (!propagating_only || r.flux_r[j] > 0.0)
                err = worse(err, std::abs(jr[k] - ref_r) / (std::max)(1.0, std::abs(ref_r)));
            if (!propagating_only || r.flux_t[j] > 0.0)
                err = worse(err, std::abs(jt[k] - ref_t) / (std::max)(1.0, std::abs(ref_t)));
        }
    }
    return err;
}

static const ConicalRefCase& ref_case(int id) {
    for (const ConicalRefCase& c : kConicalRefCases)
        if (c.id == id) return c;
    throw std::invalid_argument("unknown reference case");
}
// 按参照算例的结构和入射条件求解
static rcwa::ConicalResult solve_ref(int id, const rcwa::Options& opt = {}) {
    const ConicalRefCase& c = ref_case(id);
    return rcwa::solve_conical(conical_ref_stack(id), c.wavelength, c.theta_deg, c.phi_deg, c.M, opt);
}

// ---------------------------------------------------------------------------------------------
// 与参照表对拍：十个算例，−M … +M 全部级次的复 Jones 矩阵。容差 |ΔJ| ≤ 1e-9·max(1, |J|)，倏逝级也按它核对。
static void test_reference_tables() {
    say("[参照表：各级复 Jones 矩阵逐级比较]\n");
    CHECK(sizeof(kConicalRefCases) / sizeof(kConicalRefCases[0]) == 10);
    double worst = 0.0, worst_variants = 0.0;
    for (const ConicalRefCase& c : kConicalRefCases) {
        // 表本身的格式：−M … +M 依次排列
        bool ordered = c.count == 2 * c.M + 1;
        for (int i = 0; ordered && i < c.count; ++i) ordered = c.orders[i].m == i - c.M;
        CHECK(ordered);
        if (!ordered) continue;

        const rcwa::Stack stack = conical_ref_stack(c.id);
        const rcwa::ConicalResult r = rcwa::solve_conical(stack, c.wavelength, c.theta_deg, c.phi_deg, c.M);
        const double prop = table_error(r, c, true), all = table_error(r, c, false);
        CHECK(r.eig_info == 0);
        CHECK_LE(prop, 1e-9);
        CHECK_LE(all, 1e-9);
        worst = worse(worst, all);
        int propagating_r = 0, propagating_t = 0;
        for (int j = 0; j < c.count; ++j) {
            if (r.flux_r[j] > 0.0) ++propagating_r;
            if (r.flux_t[j] > 0.0) ++propagating_t;
        }
        say("  算例 %2d  θ = %7.2f°  φ = %7.2f°  M = %2d  传播级 %d + %d 个  偏差 %.1e（传播级）%.1e（全部级次）\n", c.id,
            c.theta_deg, c.phi_deg, c.M, propagating_r, propagating_t, prop, all);

        // 换一种选项，结果仍与参照一致：稠密路径、QR 分解解方程、LR 迭代
        rcwa::Options dense, qr, lr, dense_lr;
        dense.homogeneous_fast_path = false;
        qr.linear = rcwa::LinearSolver::QRHouseholder;
        lr.eig = EigMethod::LR;
        dense_lr.homogeneous_fast_path = false;
        dense_lr.eig = EigMethod::LR;
        dense_lr.linear = rcwa::LinearSolver::QRHouseholder;
        for (const rcwa::Options* opt : {&dense, &qr, &lr, &dense_lr}) {
            const double err = table_error(rcwa::solve_conical(stack, c.wavelength, c.theta_deg, c.phi_deg, c.M, *opt), c, false);
            CHECK_LE(err, 1e-9);
            worst_variants = worse(worst_variants, err);
        }
    }
    say("  十个算例的最大偏差 %.1e；换用稠密路径 / QR 解方程 / LR 迭代时 %.1e\n", worst, worst_variants);
    CHECK_THROWS(conical_ref_stack(0), std::invalid_argument);
    CHECK_THROWS(conical_ref_stack(11), std::invalid_argument);
}

// ---------------------------------------------------------------------------------------------
// 结果里各个量的定义
static void test_result_fields() {
    say("[结果里各个量的定义]\n");
    const ConicalRefCase& c = ref_case(4);                     // 入射介质 n = 1.2，两个光栅层夹一层吸收膜
    const rcwa::Stack stack = conical_ref_stack(4);
    const int M = c.M, N = 2 * M + 1;
    const rcwa::ConicalResult r = solve_ref(4);
    const std::size_t count = static_cast<std::size_t>(N);
    CHECK(r.orders.size() == count && r.orders.front() == -M && r.orders.back() == M && r.orders[M] == 0);
    CHECK(r.kx.size() == count && r.phi_m_deg.size() == count && r.kzI.size() == count && r.kzII.size() == count);
    CHECK(r.flux_r.size() == count && r.flux_t.size() == count && r.Jr.size() == count && r.Jt.size() == count &&
          r.DEr.size() == count && r.DEt.size() == count);

    const double theta = c.theta_deg * kPi / 180.0, phi = c.phi_deg * kPi / 180.0;
    const double nI = stack.nI.real(), nII = stack.nII.real();
    double kx_err = 0.0, angle_err = 0.0, kz_err = 0.0, flux_err = 0.0, de_err = 0.0;
    bool branch_ok = true, evanescent_zero = true;
    CHECK_LE(std::abs(r.ky - nI * std::sin(theta) * std::sin(phi)), 1e-15);
    for (int j = 0; j < N; ++j) {
        const int m = j - M;
        const double kx = nI * std::sin(theta) * std::cos(phi) - m * c.wavelength / stack.period;
        kx_err = worse(kx_err, std::abs(r.kx[j] - kx));
        angle_err = worse(angle_err, std::abs(r.phi_m_deg[j] - std::atan2(r.ky, r.kx[j]) * 180.0 / kPi));
        // kz² = n² − kx² − ky²，Im ≥ 0，Im = 0 时 Re ≥ 0；折射率是实数，所以实部、虚部总有一个严格为零
        for (int side = 0; side < 2; ++side) {
            const cplx kz = side == 0 ? r.kzI[j] : r.kzII[j];
            const double n = side == 0 ? nI : nII;
            kz_err = worse(kz_err, std::abs(kz * kz - (n * n - r.kx[j] * r.kx[j] - r.ky * r.ky)));
            branch_ok = branch_ok && kz.imag() >= 0.0 && kz.real() >= 0.0 && (kz.real() == 0.0 || kz.imag() == 0.0);
        }
        const double fr = r.kzI[j].real() / (nI * std::cos(theta)), ft = r.kzII[j].real() / (nI * std::cos(theta));
        flux_err = worse(flux_err, worse(std::abs(r.flux_r[j] - fr), std::abs(r.flux_t[j] - ft)));
        if (r.kzI[j].imag() > 0.0) evanescent_zero = evanescent_zero && r.flux_r[j] == 0.0 && r.DEr[j].ss == 0.0 && r.DEr[j].pp == 0.0;
        if (r.kzII[j].imag() > 0.0) evanescent_zero = evanescent_zero && r.flux_t[j] == 0.0 && r.DEt[j].sp == 0.0 && r.DEt[j].ps == 0.0;
        // 通道效率 = 能流因子 × |Jones 矩阵元|²
        const std::array<cplx, 4> jr = entries(r.Jr[j]), jt = entries(r.Jt[j]);
        const std::array<double, 4> er = entries(r.DEr[j]), et = entries(r.DEt[j]);
        for (int k = 0; k < 4; ++k)
            de_err = worse(de_err, worse(std::abs(er[k] - r.flux_r[j] * std::norm(jr[k])), std::abs(et[k] - r.flux_t[j] * std::norm(jt[k]))));
    }
    CHECK_LE(kx_err, 1e-15);
    CHECK_LE(angle_err, 1e-12);
    CHECK_LE(kz_err, 1e-13);
    CHECK_LE(flux_err, 1e-15);
    CHECK_LE(de_err, 1e-15);
    CHECK(branch_ok);
    CHECK(evanescent_zero);
    CHECK_LE(std::abs(r.flux_r[M] - 1.0), 1e-15);              // 入射级：kz = nI·cos θ
    CHECK(r.flux_r[0] == 0.0 && r.flux_t[0] == 0.0 && r.flux_r[2 * M] == 0.0);      // 最外面的级次是倏逝的
    CHECK(r.n_eig == 4 && r.eig_info == 0 && r.max_eig_residual == 0.0 && r.max_solve_backward_error == 0.0);

    // 同一个方向的两种写法：(θ, φ) 与 (−θ, φ + 180°) 的 kx、ky 完全相同，结果也相同；φ 加 360° 不变
    {
        const rcwa::ConicalResult flipped = rcwa::solve_conical(stack, c.wavelength, -c.theta_deg, c.phi_deg + 180.0, M);
        const rcwa::ConicalResult turned = rcwa::solve_conical(stack, c.wavelength, c.theta_deg, c.phi_deg - 360.0, M);
        CHECK_LE(std::abs(flipped.ky - r.ky) + std::abs(flipped.kx[0] - r.kx[0]), 1e-15);
        CHECK_LE(jones_gap(r, flipped), 1e-11);
        CHECK_LE(jones_gap(r, turned), 1e-11);
    }
    // 沿 z 轴传播的级次（正入射的零级）取入射的方位角；其余各级的衍射面就是 xz 平面（φ_m 是 0° 或 180°）
    {
        const rcwa::ConicalResult normal = rcwa::solve_conical(conical_ref_stack(1), 1.053, 0.0, 37.0, 5);
        bool in_plane = true;
        for (int j = 0; j <= 10; ++j)
            if (j != 5) in_plane = in_plane && (normal.phi_m_deg[j] == 0.0 || normal.phi_m_deg[j] == 180.0);
        CHECK(normal.phi_m_deg[5] == 37.0 && in_plane && normal.ky == 0.0 && normal.kx[5] == 0.0);
        CHECK(normal.kx[4] == -normal.kx[6] && normal.kx[6] < 0.0 && normal.phi_m_deg[6] == 180.0);
        CHECK_LE(unitarity_defect(normal), 1e-12);
    }
    // 默认构造
    {
        const rcwa::ConicalPoint p;
        CHECK(p.wavelength == 1.0 && p.theta_deg == 0.0 && p.phi_deg == 0.0);
        const rcwa::JonesEfficiency e;
        CHECK(e.ss == 0.0 && e.sp == 0.0 && e.ps == 0.0 && e.pp == 0.0);
        const rcwa::Jones j{};
        CHECK(j.ss == cplx{} && j.sp == cplx{} && j.ps == cplx{} && j.pp == cplx{});
        const rcwa::ConicalResult empty;
        CHECK(empty.orders.empty() && empty.Jr.empty() && empty.ky == 0.0 && empty.n_eig == 0 && empty.eig_info == 0);
        const std::array<cplx, 4> u = empty.flux_unitarity();
        CHECK(u[0] == cplx{} && u[1] == cplx{} && u[2] == cplx{} && u[3] == cplx{});
    }
}

// ---------------------------------------------------------------------------------------------
// for_input / for_angles / flux_unitarity
static void test_polarization() {
    say("[指定入射偏振：for_input、for_angles、flux_unitarity]\n");
    const rcwa::ConicalResult r = solve_ref(1);               // 无吸收
    const int M = ref_case(1).M, N = 2 * M + 1;
    const std::size_t count = static_cast<std::size_t>(N);

    // 任意 Jones 矢量：先归一化，再逐级做 2×2 的乘法
    const cplx es(0.3, -1.1), ep(-0.7, 0.4);
    const double norm = std::sqrt(std::norm(es) + std::norm(ep));
    const rcwa::ConicalPolarized c = r.for_input(es, ep);
    CHECK_LE(std::abs(c.Es - es / norm) + std::abs(c.Ep - ep / norm), 1e-15);
    CHECK_LE(std::abs(std::norm(c.Es) + std::norm(c.Ep) - 1.0), 1e-15);
    CHECK(c.Rs.size() == count && c.Rp.size() == count && c.Ts.size() == count && c.Tp.size() == count);
    CHECK(c.DEr.size() == count && c.DEt.size() == count && c.DEr_s.size() == count && c.DEr_p.size() == count &&
          c.DEt_s.size() == count && c.DEt_p.size() == count);
    double amp_err = 0.0, de_err = 0.0, sum_r = 0.0, sum_t = 0.0;
    for (int j = 0; j < N; ++j) {
        amp_err = worse(amp_err, std::abs(c.Rs[j] - (r.Jr[j].ss * c.Es + r.Jr[j].sp * c.Ep)));
        amp_err = worse(amp_err, std::abs(c.Rp[j] - (r.Jr[j].ps * c.Es + r.Jr[j].pp * c.Ep)));
        amp_err = worse(amp_err, std::abs(c.Ts[j] - (r.Jt[j].ss * c.Es + r.Jt[j].sp * c.Ep)));
        amp_err = worse(amp_err, std::abs(c.Tp[j] - (r.Jt[j].ps * c.Es + r.Jt[j].pp * c.Ep)));
        de_err = worse(de_err, std::abs(c.DEr_s[j] - r.flux_r[j] * std::norm(c.Rs[j])));
        de_err = worse(de_err, std::abs(c.DEr_p[j] - r.flux_r[j] * std::norm(c.Rp[j])));
        de_err = worse(de_err, std::abs(c.DEt_s[j] - r.flux_t[j] * std::norm(c.Ts[j])));
        de_err = worse(de_err, std::abs(c.DEt_p[j] - r.flux_t[j] * std::norm(c.Tp[j])));
        de_err = worse(de_err, std::abs(c.DEr[j] - (c.DEr_s[j] + c.DEr_p[j])));
        de_err = worse(de_err, std::abs(c.DEt[j] - (c.DEt_s[j] + c.DEt_p[j])));
        sum_r += c.DEr[j];
        sum_t += c.DEt[j];
    }
    CHECK_LE(amp_err, 1e-15);
    CHECK_LE(de_err, 1e-15);
    CHECK_LE(std::abs(c.reflectance() - sum_r) + std::abs(c.transmittance() - sum_t), 1e-15);
    CHECK_LE(std::abs(c.reflectance() + c.transmittance() - 1.0), 1e-11);      // 无吸收：任何入射偏振都守恒

    // 幅值、整体相位不影响效率
    {
        const rcwa::ConicalPolarized scaled = r.for_input(es * cplx(-3.0, 4.0), ep * cplx(-3.0, 4.0));
        double gap = 0.0;
        for (int j = 0; j < N; ++j) gap = worse(gap, worse(std::abs(scaled.DEr[j] - c.DEr[j]), std::abs(scaled.DEt[j] - c.DEt[j])));
        CHECK_LE(gap, 1e-14);
        CHECK_LE(std::abs(scaled.Es - c.Es * cplx(-0.6, 0.8)) + std::abs(scaled.Ep - c.Ep * cplx(-0.6, 0.8)), 1e-15);
        const rcwa::ConicalPolarized simple = r.for_input(3.0, cplx(0.0, 4.0));
        CHECK_LE(std::abs(simple.Es - 0.6) + std::abs(simple.Ep - cplx(0.0, 0.8)), 1e-16);
        const rcwa::ConicalPolarized tiny = r.for_input(1e-200, cplx(0.0, 1e-200)), huge = r.for_input(1e200, cplx(0.0, 1e200));
        CHECK_LE(std::abs(tiny.Es - std::sqrt(0.5)) + std::abs(huge.Ep - cplx(0.0, std::sqrt(0.5))), 1e-15);
    }
    // 角度的写法：[Es, Ep] = [sin ψ·exp(iδ), cos ψ]
    {
        const rcwa::ConicalPolarized s = r.for_angles(90.0), p = r.for_angles(0.0), d = r.for_angles(45.0), circ = r.for_angles(45.0, 90.0);
        const rcwa::ConicalPolarized e = r.for_angles(30.0, -60.0);
        CHECK_LE(std::abs(s.Es - 1.0) + std::abs(s.Ep), 1e-15);
        CHECK(p.Es == cplx{} && p.Ep == cplx(1.0, 0.0));
        CHECK_LE(std::abs(d.Es - std::sqrt(0.5)) + std::abs(d.Ep - std::sqrt(0.5)), 1e-15);
        CHECK_LE(std::abs(circ.Es - cplx(0.0, std::sqrt(0.5))) + std::abs(circ.Ep - std::sqrt(0.5)), 1e-15);
        CHECK_LE(std::abs(e.Es - 0.5 * cplx(0.5, -std::sqrt(0.75))) + std::abs(e.Ep - std::sqrt(0.75)), 1e-15);
        // s、p 入射时各级效率就是对应的两个通道之和
        double gap = 0.0;
        for (int j = 0; j < N; ++j) {
            gap = worse(gap, std::abs(s.DEr[j] - (r.DEr[j].ss + r.DEr[j].ps)) + std::abs(s.DEt[j] - (r.DEt[j].ss + r.DEt[j].ps)));
            gap = worse(gap, std::abs(p.DEr[j] - (r.DEr[j].sp + r.DEr[j].pp)) + std::abs(p.DEt[j] - (r.DEt[j].sp + r.DEt[j].pp)));
            gap = worse(gap, std::abs(s.DEr_s[j] - r.DEr[j].ss) + std::abs(s.DEr_p[j] - r.DEr[j].ps));
            gap = worse(gap, std::abs(p.DEt_s[j] - r.DEt[j].sp) + std::abs(p.DEt_p[j] - r.DEt[j].pp));
        }
        CHECK_LE(gap, 1e-15);
    }
    // 零矢量、非有限的分量
    CHECK_THROWS(r.for_input(0.0, 0.0), std::invalid_argument);
    CHECK_THROWS(r.for_input(std::numeric_limits<double>::quiet_NaN(), 1.0), std::invalid_argument);
    CHECK_THROWS(r.for_input(1.0, cplx(0.0, std::numeric_limits<double>::infinity())), std::invalid_argument);
    CHECK_THROWS(r.for_angles(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);

    // 通量幺正性矩阵 U = Σ (flux_r·Jr†Jr + flux_t·Jt†Jt)：厄米；对任何入射 v 有 R + T = v†·U·v；无吸收时是单位矩阵
    for (int id : {1, 2}) {
        const rcwa::ConicalResult q = solve_ref(id);
        const std::array<cplx, 4> u = q.flux_unitarity();
        CHECK_LE(std::abs(u[0].imag()) + std::abs(u[3].imag()) + std::abs(u[1] - std::conj(u[2])), 1e-15);
        double quad_err = 0.0;
        for (int k = 0; k < 6; ++k) {
            const rcwa::ConicalPolarized v = q.for_angles(15.0 + 31.0 * k, 47.0 * k);
            const cplx quad = std::conj(v.Es) * (u[0] * v.Es + u[1] * v.Ep) + std::conj(v.Ep) * (u[2] * v.Es + u[3] * v.Ep);
            quad_err = worse(quad_err, std::abs(quad - (v.reflectance() + v.transmittance())));
        }
        CHECK_LE(quad_err, 1e-14);
        if (id == 1) {
            CHECK_LE(unitarity_defect(q), 1e-11);
        } else {                                               // 有吸收：两种入射都有损耗
            CHECK(u[0].real() < 0.99 && u[3].real() < 0.99 && u[0].real() > 0.0 && u[3].real() > 0.0);
            CHECK(std::norm(u[1]) < u[0].real() * u[3].real());   // U 正定
        }
    }
    say("  无吸收光栅的 |U − I| = %.1e\n", unitarity_defect(r));
}

// ---------------------------------------------------------------------------------------------
static void test_arguments() {
    say("[参数检查]\n");
    const rcwa::Stack s = conical_ref_stack(1);
    const double nan = std::numeric_limits<double>::quiet_NaN(), inf = std::numeric_limits<double>::infinity();
    const auto run = [](const rcwa::Stack& st, double wavelength = 1.053, double theta = 30.0, double phi = 20.0, int M = 4) {
        return rcwa::solve_conical(st, wavelength, theta, phi, M);
    };
    CHECK(run(s).eig_info == 0 && all_finite(run(s)));
    CHECK(all_finite(run(s, 1.053, 30.0, 20.0, 0)) && run(s, 1.053, 30.0, 20.0, 0).Jr.size() == 1);   // 只保留零级
    CHECK_THROWS(run(s, 1.053, 30.0, 20.0, -1), std::invalid_argument);
    CHECK_THROWS(run(s, 0.0), std::invalid_argument);
    CHECK_THROWS(run(s, -1.0), std::invalid_argument);
    CHECK_THROWS(run(s, nan), std::invalid_argument);
    CHECK_THROWS(run(s, 1.053, 90.0), std::invalid_argument);
    CHECK_THROWS(run(s, 1.053, -90.0), std::invalid_argument);
    CHECK_THROWS(run(s, 1.053, inf), std::invalid_argument);
    CHECK_THROWS(run(s, 1.053, nan), std::invalid_argument);
    CHECK_THROWS(run(s, 1.053, 30.0, nan), std::invalid_argument);
    CHECK_THROWS(run(s, 1.053, 30.0, inf), std::invalid_argument);
    CHECK_THROWS(run(s, 1.053, 30.0, -inf), std::invalid_argument);
    CHECK(all_finite(run(s, 1.053, -89.0, -725.0)));          // 负的入射角、任意大小的方位角都是合法的
    {
        rcwa::Stack t = s;
        t.period = 0.0;
        CHECK_THROWS(run(t), std::invalid_argument);
        t = s;
        t.nI = cplx(1.0, 0.01);                              // 入射介质必须无吸收
        CHECK_THROWS(run(t), std::invalid_argument);
        t = s;
        t.nI = 0.0;
        CHECK_THROWS(run(t), std::invalid_argument);
        t = s;
        t.nII = cplx(1.5, 0.1);                              // 出射介质也必须是实折射率（与 rcwa::solve 不同）
        CHECK_THROWS(run(t), std::invalid_argument);
        CHECK(rcwa::solve(t, 1.053, 30.0, 4).eig_info == 0);
        t.nII = cplx(1.5, 1e-300);
        CHECK_THROWS(run(t), std::invalid_argument);
        t.nII = cplx(1.5, -0.1);
        CHECK_THROWS(run(t), std::invalid_argument);
        t.nII = 0.0;
        CHECK_THROWS(run(t), std::invalid_argument);
        t.nII = nan;
        CHECK_THROWS(run(t), std::invalid_argument);
        t = s;
        t.layers[0].d = -0.1;
        CHECK_THROWS(run(t), std::invalid_argument);
        t = s;
        t.layers[0].n_values.pop_back();                     // 段数与边界数对不上
        CHECK_THROWS(run(t), std::invalid_argument);
        t = s;
        t.layers[0].edges.back() += 0.01;                    // 边界没有正好跨一个周期
        CHECK_THROWS(run(t), std::invalid_argument);
        t = s;
        std::swap(t.layers[0].edges[0], t.layers[0].edges[1]);   // 边界不递增
        CHECK_THROWS(run(t), std::invalid_argument);
        t = s;
        t.layers[0].n_values[0] = cplx(nan, 0.0);
        CHECK_THROWS(run(t), std::invalid_argument);
        t = s;
        t.layers[0].n_values[0] = cplx(1.5, -0.2);           // 增益介质不支持
        CHECK_THROWS(run(t), std::invalid_argument);
        t = s;
        t.layers.push_back(rcwa::Layer::uniform(0.1, 0.0));  // 折射率为零
        CHECK_THROWS(run(t), std::invalid_argument);
        t = s;
        t.layers.push_back(rcwa::Layer::uniform(0.1, cplx(0.2, 3.0)));   // 层里的吸收（金属）是合法的
        CHECK(all_finite(run(t)));
    }
    {
        rcwa::Options opt;
        opt.linear = static_cast<rcwa::LinearSolver>(7);
        CHECK_THROWS(rcwa::solve_conical(s, 1.053, 30.0, 20.0, 4, opt), std::invalid_argument);
        opt = rcwa::Options();
        opt.eig = static_cast<EigMethod>(7);
        CHECK_THROWS(rcwa::solve_conical(s, 1.053, 30.0, 20.0, 4, opt), std::invalid_argument);
    }
    // 参数非法时扫描也抛异常（从并行区里原样传出来），之后线程池照常可用
    {
        std::vector<rcwa::ConicalPoint> points(6, rcwa::ConicalPoint{1.053, 30.0, 20.0});
        points[4].phi_deg = nan;
        set_num_threads(3);
        CHECK_THROWS(rcwa::sweep_conical(s, points, 4), std::invalid_argument);
        points[4].phi_deg = 21.0;
        const std::vector<rcwa::ConicalResult> swept = rcwa::sweep_conical(s, points, 4);
        set_num_threads(1);
        CHECK(swept.size() == 6 && same_bits(swept[0], run(s)) && same_bits(swept[4], run(s, 1.053, 30.0, 21.0)));
    }
}

// ---------------------------------------------------------------------------------------------
static void test_failures() {
    say("[数值失败的报告方式]\n");
    const rcwa::Stack s = conical_ref_stack(1);
    const int M = 6;
    const rcwa::ConicalResult good = rcwa::solve_conical(s, 1.053, 66.36, 17.0, M);

    // 本征迭代未收敛：不抛异常，Jones 矩阵和效率全部是 NaN，eig_info 给出 info；只由入射条件决定的量照常给出
    {
        rcwa::Options opt;
        opt.custom_eig = [](const Matrix& a) {
            EigOptions one;
            one.max_iter = 1;
            return eig_qr(a, one);
        };
        const rcwa::ConicalResult r = rcwa::solve_conical(s, 1.053, 66.36, 17.0, M, opt);
        CHECK(r.eig_info > 0 && all_nan(r) && r.n_eig == 2);
        CHECK(static_cast<int>(r.orders.size()) == 2 * M + 1 && r.timing.total > 0.0);
        CHECK(same_bytes(r.kx, good.kx) && r.ky == good.ky && same_bytes(r.kzI, good.kzI) && same_bytes(r.kzII, good.kzII) &&
              same_bytes(r.flux_r, good.flux_r) && same_bytes(r.flux_t, good.flux_t) && same_bytes(r.phi_m_deg, good.phi_m_deg));
        opt.diagnostics = true;
        CHECK(std::isnan(rcwa::solve_conical(s, 1.053, 66.36, 17.0, M, opt).max_eig_residual));
        // 失败的结果再指定入射偏振：振幅、效率、合计仍然全部是 NaN
        const rcwa::ConicalPolarized c = r.for_angles(30.0, 10.0);
        bool nan_everywhere = c.DEr.size() == r.orders.size();
        for (std::size_t j = 0; nan_everywhere && j < c.DEr.size(); ++j)
            nan_everywhere = std::isnan(c.DEr[j]) && std::isnan(c.DEt[j]) && std::isnan(c.Rs[j].real()) && std::isnan(c.Tp[j].imag());
        CHECK(nan_everywhere && std::isnan(c.reflectance()) && std::isnan(c.transmittance()));
        CHECK(std::isnan(r.flux_unitarity()[0].real()));
    }
    // 只有第二次分解（B·C）没收敛：同样是失败，eig_info 取这一次的 info
    {
        int calls = 0;
        rcwa::Options opt;
        opt.custom_eig = [&calls](const Matrix& a) {
            EigResult r = eig_qr(a);
            if (++calls == 2) {
                r.info = 3;
                r.vectors = Matrix();
            }
            return r;
        };
        const rcwa::ConicalResult r = rcwa::solve_conical(s, 1.053, 66.36, 17.0, M, opt);
        CHECK(r.eig_info == 3 && all_nan(r) && r.n_eig == 2 && calls == 2);
    }
    // 本征向量矩阵奇异（全零）：三种线性求解方法都报告失败
    for (rcwa::LinearSolver lin : {rcwa::LinearSolver::LUPartial, rcwa::LinearSolver::LUNoPivot, rcwa::LinearSolver::QRHouseholder}) {
        rcwa::Options opt;
        opt.linear = lin;
        opt.custom_eig = [](const Matrix& a) {
            EigResult r = eig_qr(a);
            r.vectors = Matrix(a.rows(), a.cols());
            return r;
        };
        const rcwa::ConicalResult r = rcwa::solve_conical(s, 1.053, 66.36, 17.0, M, opt);
        CHECK(all_nan(r) && r.eig_info == 0);
    }
    // 本征求解器返回的尺寸不对：这是调用方的错误
    {
        rcwa::Options opt;
        opt.custom_eig = [](const Matrix& a) { return eig_qr(Matrix::identity(a.rows() - 1)); };
        CHECK_THROWS(rcwa::solve_conical(s, 1.053, 66.36, 17.0, M, opt), std::invalid_argument);
        opt.custom_eig = [](const Matrix& a) {               // 本征值多一个
            EigResult r = eig_qr(a);
            r.values.push_back(1.0);
            return r;
        };
        CHECK_THROWS(rcwa::solve_conical(s, 1.053, 66.36, 17.0, M, opt), std::invalid_argument);
        opt.custom_eig = [](const Matrix& a) {               // 本征向量矩阵多一行
            EigResult r = eig_qr(a);
            Matrix tall(a.rows() + 1, a.cols());
            for (int j = 0; j < a.cols(); ++j)
                for (int i = 0; i < a.rows(); ++i) tall(i, j) = r.vectors(i, j);
            r.vectors = tall;
            return r;
        };
        CHECK_THROWS(rcwa::solve_conical(s, 1.053, 66.36, 17.0, M, opt), std::invalid_argument);
    }
    // custom_eig 自己抛的异常不会被当成数值失败吞掉
    for (bool parallel : {false, true}) {
        rcwa::Options bad;
        bad.parallel_layers = parallel;
        bad.custom_eig = [](const Matrix& a) -> EigResult {
            if (a.rows() > 0) throw std::runtime_error("backend lost");
            return EigResult();
        };
        set_num_threads(parallel ? 3 : 1);
        CHECK_THROWS(rcwa::solve_conical(s, 1.053, 66.36, 17.0, M, bad), std::runtime_error);
        set_num_threads(1);
    }
    // 齿的折射率取极端值：层矩阵里出现 Inf，按失败处理
    {
        const rcwa::Stack t = unit_stack({rcwa::Layer::grating(0.3, {0.0, 0.3, 1.0}, {1e200, 1.0})}, 1.5);
        CHECK(all_nan(rcwa::solve_conical(t, 0.9, 20.0, 25.0, 3)));
    }
    // 外侧介质或膜的折射率取极端值：结果要么全部是 NaN，要么全部有限而且通量幺正（这几个结构都没有吸收），
    // 不留下“一部分是 NaN/Inf”的结果
    {
        bool consistent = true;
        int failed = 0, solved = 0;
        for (double extreme : {1e300, 1e160, 1e-160, 1e-300})
            for (const rcwa::Stack& t : {unit_stack({unit_grating()}, extreme), unit_stack({}, extreme),
                                         unit_stack({rcwa::Layer::uniform(0.2, 1.7)}, extreme),
                                         unit_stack({rcwa::Layer::uniform(0.2, extreme)}, 1.5)})
                for (int m : {0, 3})
                    for (double phi : {0.0, 25.0}) {
                        const rcwa::ConicalResult r = rcwa::solve_conical(t, 0.9, 20.0, phi, m);
                        const bool nan = all_nan(r), fine = all_finite(r) && unitarity_defect(r) < 1e-9;
                        consistent = consistent && (nan || fine) && static_cast<int>(r.Jr.size()) == 2 * m + 1;
                        if (nan) ++failed;
                        else ++solved;
                    }
        CHECK(consistent);
        CHECK(failed > 0 && solved > 0);
    }
}

// ---------------------------------------------------------------------------------------------
// 恰好落在 Rayleigh 异常点上（某一级在某个均匀介质里 kz 精确为零）时怎么处理，约定见 rcwa1d_conical.hpp 开头。
// 这里都取 λ = Λ = 1、θ = 0：ky = 0，第 ±m 级的 kx = ∓m。方位角只决定零级的 s、p 朝哪个方向。
static void test_rayleigh_anomalies() {
    say("[恰好落在 Rayleigh 异常点上]\n");
    const int M = 6;
    for (double phi : {0.0, 30.0}) {
        // (1) 入射介质里 ±1 级恰好掠射：不妨碍求解，这两级的通量为零，通量幺正性照旧；快速路径与稠密路径一致
        {
            const rcwa::Stack s = unit_stack({unit_grating()}, 1.5);
            const rcwa::ConicalResult r = rcwa::solve_conical(s, 1.0, 0.0, phi, M);
            CHECK(all_finite(r) && r.kzI[M + 1] == cplx{} && r.kzI[M - 1] == cplx{});
            CHECK(r.flux_r[M + 1] == 0.0 && r.flux_r[M - 1] == 0.0 && r.DEr[M + 1].ss == 0.0 && r.DEr[M - 1].pp == 0.0);
            CHECK_LE(unitarity_defect(r), 1e-12);
            rcwa::Options dense;
            dense.homogeneous_fast_path = false;
            CHECK_LE(jones_gap(r, rcwa::solve_conical(s, 1.0, 0.0, phi, M, dense)), 1e-12);
            // 膜紧挨着入射介质时也一样
            const rcwa::Stack f = unit_stack({rcwa::Layer::uniform(0.1, 1.7), unit_grating()}, 1.5);
            const rcwa::ConicalResult rf = rcwa::solve_conical(f, 1.0, 0.0, phi, M);
            CHECK(all_finite(rf));
            CHECK_LE(unitarity_defect(rf), 1e-12);
            // 波长挪开一点点：结果连续（异常点上是平方根型的尖点，1e-12 的偏离对应 1e-6 的变化）
            CHECK_LE(jones_gap(r, rcwa::solve_conical(s, 1.0 + 1e-12, 0.0, phi, M)), 1e-4);
            CHECK_LE(jones_gap(r, rcwa::solve_conical(s, 1.0 - 1e-12, 0.0, phi, M)), 1e-4);
        }
        // (2) 出射介质里 ±2 级恰好掠射（n = 2）：上面紧挨着均匀层、走快速路径时照常求解；
        //     紧挨光栅层，或者 homogeneous_fast_path = false，界面上的方程组奇异，按失败处理
        {
            const rcwa::Stack direct = unit_stack({unit_grating()}, 2.0);
            const rcwa::Stack buffered = unit_stack({unit_grating(), rcwa::Layer::uniform(0.2, 1.7)}, 2.0);
            const rcwa::ConicalResult a = rcwa::solve_conical(direct, 1.0, 0.0, phi, M);
            const rcwa::ConicalResult b = rcwa::solve_conical(buffered, 1.0, 0.0, phi, M);
            CHECK(all_nan(a) && a.eig_info == 0 && a.n_eig == 2);
            CHECK(all_finite(b) && b.kzII[M + 2] == cplx{} && b.flux_t[M + 2] == 0.0 && b.DEt[M - 2].pp == 0.0);
            CHECK_LE(unitarity_defect(b), 1e-12);
            rcwa::Options dense;
            dense.homogeneous_fast_path = false;
            CHECK(all_nan(rcwa::solve_conical(buffered, 1.0, 0.0, phi, M, dense)));
            CHECK_LE(unitarity_defect(rcwa::solve_conical(direct, 1.0 + 1e-9, 0.0, phi, M)), 1e-9);
        }
        // (3) 均匀层里恰好掠射、结构里又有光栅层：报告失败。M 取小、不含那一级时没有问题；波长挪开一点点也照常求解
        {
            const rcwa::Stack above = unit_stack({rcwa::Layer::uniform(0.2, 2.0), unit_grating()}, 1.5);
            const rcwa::Stack below = unit_stack({unit_grating(), rcwa::Layer::uniform(0.2, 2.0)}, 1.5);
            const rcwa::Stack buried = unit_stack({unit_grating(), rcwa::Layer::uniform(0.1, 1.7), rcwa::Layer::uniform(0.2, 2.0), rcwa::Layer::uniform(0.1, 1.7)}, 1.5);
            for (const rcwa::Stack* s : {&above, &below, &buried}) {
                const rcwa::ConicalResult r = rcwa::solve_conical(*s, 1.0, 0.0, phi, M);
                CHECK(all_nan(r) && r.eig_info == 0);
                CHECK_LE(unitarity_defect(rcwa::solve_conical(*s, 1.0, 0.0, phi, 1)), 1e-12);
                CHECK_LE(unitarity_defect(rcwa::solve_conical(*s, 1.0 + 1e-9, 0.0, phi, M)), 1e-9);
                CHECK_LE(unitarity_defect(rcwa::solve_conical(*s, 1.0 - 1e-9, 0.0, phi, M)), 1e-9);
            }
        }
        // (4) 纯膜系不产生衍射：零级之外的各级没有被激发，振幅是零，它们恰好掠射不影响结果——与只保留零级（M = 0）相同。
        //     几种情形：膜里掠射；膜与入射介质相同、界面两侧同时掠射；膜与出射介质相同；只有出射介质里掠射
        {
            const rcwa::Stack films[] = {
                unit_stack({rcwa::Layer::uniform(0.2, 2.0)}, 1.5),
                unit_stack({rcwa::Layer::uniform(0.2, 1.0)}, 1.5),
                unit_stack({rcwa::Layer::uniform(0.2, 2.0)}, 2.0),
                unit_stack({rcwa::Layer::uniform(0.2, 1.7)}, 2.0),
                unit_stack({}, 1.0),
            };
            for (const rcwa::Stack& f : films) {
                const int m = 4;
                const rcwa::ConicalResult r = rcwa::solve_conical(f, 1.0, 0.0, phi, m), r0 = rcwa::solve_conical(f, 1.0, 0.0, phi, 0);
                CHECK(all_finite(r) && r.eig_info == 0 && r.n_eig == 0 && r.n_dense_solve == 0);
                bool others_zero = static_cast<int>(r.Jr.size()) == 2 * m + 1;
                for (int j = 0; others_zero && j <= 2 * m; ++j) {
                    if (j == m) continue;
                    for (const cplx& v : entries(r.Jr[j])) others_zero = others_zero && v == cplx{};
                    for (const cplx& v : entries(r.Jt[j])) others_zero = others_zero && v == cplx{};
                    for (double v : entries(r.DEr[j])) others_zero = others_zero && v == 0.0;
                    for (double v : entries(r.DEt[j])) others_zero = others_zero && v == 0.0;
                }
                CHECK(others_zero);
                double gap = 0.0;
                for (int k = 0; others_zero && k < 4; ++k)
                    gap = worse(gap, std::abs(entries(r.Jr[m])[k] - entries(r0.Jr[0])[k]) + std::abs(entries(r.Jt[m])[k] - entries(r0.Jt[0])[k]));
                CHECK_LE(others_zero ? gap : 1.0, 1e-15);
                CHECK_LE(unitarity_defect(r), 1e-13);
            }
        }
    }
    // (5) 扫描里有一个点落在异常点上（出射介质恰好掠射，上面紧挨光栅层）：只有那个点是 NaN，其余各点不受影响
    {
        const rcwa::Stack s = unit_stack({unit_grating()}, 2.0);
        const std::vector<rcwa::ConicalPoint> points{{1.01, 0.0, 10.0}, {1.0, 0.0, 10.0}, {0.93, 12.0, 40.0}, {1.0, 1e-3, 10.0}};
        set_num_threads(3);
        const std::vector<rcwa::ConicalResult> swept = rcwa::sweep_conical(s, points, M);
        set_num_threads(1);
        bool ok = swept.size() == points.size() && all_nan(swept[1]) && swept[1].eig_info == 0 && swept[1].n_eig == 2;
        for (std::size_t i = 0; ok && i < points.size(); ++i)
            if (i != 1)
                ok = same_bits(swept[i], rcwa::solve_conical(s, points[i].wavelength, points[i].theta_deg, points[i].phi_deg, M)) &&
                     unitarity_defect(swept[i]) < 1e-9;
        CHECK(ok);
    }
}

// ---------------------------------------------------------------------------------------------
// 选项互换：结果应当一致
static void test_option_equivalence() {
    say("[选项互换：快速路径与稠密路径、线性求解方法、QR 与 LR 迭代]\n");
    struct Case {
        int id;                       // 参照算例的结构
        double wavelength, theta, phi;
        int M;
    };
    // 参照表之外的入射条件：平面内入射、φ = 90°、负角度、正入射（均匀层的层矩阵有成对的重本征值）、
    // Littrow 型的入射（kx_0 = −kx_1）
    const double littrow = std::asin(0.5 * 1.053 / 0.574) * 180.0 / kPi;
    const Case cases[] = {
        {3, 1.053, 66.36, 0.0, 8},  {3, 1.053, 50.0, 90.0, 8},  {3, 1.053, -35.0, 200.0, 8}, {3, 1.053, 0.0, 0.0, 8},
        {3, 1.053, 0.0, 51.0, 8},   {3, 1.053, littrow, 0.0, 8}, {4, 0.83, 27.0, -15.0, 7},   {4, 0.83, 0.0, 120.0, 7},
        {8, 1.053, 20.0, 33.0, 9},  {10, 0.633, 50.0, 33.0, 3},  {10, 0.633, 0.0, 77.0, 4},   {2, 1.053, 42.0, 5.0, 8},
    };
    double gap_dense = 0.0, gap_linear = 0.0, gap_lr = 0.0;
    for (const Case& c : cases) {
        const rcwa::Stack stack = conical_ref_stack(c.id);
        const int layers = static_cast<int>(stack.layers.size());
        int gratings = 0;
        for (const rcwa::Layer& layer : stack.layers) gratings += layer.homogeneous ? 0 : 1;
        const rcwa::ConicalResult base = rcwa::solve_conical(stack, c.wavelength, c.theta, c.phi, c.M);
        CHECK(base.eig_info == 0 && base.n_eig == 2 * gratings);

        // 稠密路径：每一层两次本征分解；稠密求解的次数：每层组装 2 次，L+1 个界面各 3 次，
        // 最上面的块之外的 2L 个块叠加时各 1 次，入射的两列 1 次，共 7L + 4 次
        rcwa::Options dense;
        dense.homogeneous_fast_path = false;
        const rcwa::ConicalResult rd = rcwa::solve_conical(stack, c.wavelength, c.theta, c.phi, c.M, dense);
        CHECK(rd.n_eig == 2 * layers && rd.n_dense_solve == 7 * layers + 4);
        CHECK(base.n_dense_solve < rd.n_dense_solve);
        CHECK_LE(jones_gap(base, rd), 1e-10);
        gap_dense = worse(gap_dense, jones_gap(base, rd));

        // 三种线性求解方法里选主元的 LU 与 Householder QR 都是稳定的：结果一致，但末位上不同（确实换了算法）
        rcwa::Options qr;
        qr.linear = rcwa::LinearSolver::QRHouseholder;
        const rcwa::ConicalResult rq = rcwa::solve_conical(stack, c.wavelength, c.theta, c.phi, c.M, qr);
        CHECK_LE(jones_gap(base, rq), 1e-10);
        CHECK(gratings == 0 || !same_bits(base, rq));
        gap_linear = worse(gap_linear, jones_gap(base, rq));
        qr.homogeneous_fast_path = false;
        CHECK_LE(jones_gap(base, rcwa::solve_conical(stack, c.wavelength, c.theta, c.phi, c.M, qr)), 1e-10);

        // LR 迭代（阶数低时与 QR 迭代同样准）
        rcwa::Options lr;
        lr.eig = EigMethod::LR;
        const rcwa::ConicalResult rl = rcwa::solve_conical(stack, c.wavelength, c.theta, c.phi, c.M, lr);
        CHECK(rl.eig_info == 0);
        CHECK_LE(jones_gap(base, rl), 1e-9);
        CHECK(gratings == 0 || !same_bits(base, rl));
        gap_lr = worse(gap_lr, jones_gap(base, rl));
    }
    say("  各级 Jones 矩阵元的最大差别：快速路径开/关 %.1e；LU/QR 求解 %.1e；LR/QR 迭代 %.1e\n", gap_dense, gap_linear, gap_lr);

    // 快速路径下稠密求解的次数。单个光栅层：组装 2 次；上界面 W、V 各解一次、T+ 求逆一次、入射的两列一次；下界面 T+ 求逆一次。
    // 下面还有膜系时，把下界面叠到膜系上再多一次。纯膜系一次也没有。
    CHECK(solve_ref(1).n_dense_solve == 7 && solve_ref(3).n_dense_solve == 8 && solve_ref(4).n_dense_solve == 15 &&
          solve_ref(10).n_dense_solve == 0);

    // 不选主元的 LU 只用于演示不稳定性：后向误差如实变大（圆锥入射的模式矩阵里有成块的零，没有行交换时很容易坏）
    {
        rcwa::Options pivot, none;
        pivot.diagnostics = true;
        none.diagnostics = true;
        none.linear = rcwa::LinearSolver::LUNoPivot;
        const rcwa::ConicalResult a = solve_ref(3, pivot), b = solve_ref(3, none);
        CHECK(a.max_solve_backward_error < 1e-13 && b.max_solve_backward_error > 100.0 * a.max_solve_backward_error);
        CHECK(all_nan(b) || all_finite(b));
        say("  稠密求解的后向误差：选主元 %.1e，不选主元 %.1e\n", a.max_solve_backward_error, b.max_solve_backward_error);
    }
}

// ---------------------------------------------------------------------------------------------
// 各段折射率全都相同的“光栅层”就是均匀层；折射率差很小的光栅层连续地趋于它
static void test_uniform_gratings() {
    say("[各段折射率相同的光栅层、弱对比的光栅层]\n");
    const auto make = [](rcwa::Layer middle) {
        rcwa::Stack s;
        s.nI = 1.0;
        s.nII = 1.4;
        s.period = 0.8;
        s.layers = {rcwa::Layer::grating(0.2, {0.0, 0.25, 0.8}, {2.0, 1.0}), std::move(middle), rcwa::Layer::uniform(0.1, 1.9)};
        return s;
    };
    const cplx n(1.6, 0.02);
    const rcwa::Stack uniform = make(rcwa::Layer::uniform(0.27, n));
    const rcwa::Stack same = make(rcwa::Layer::grating(0.27, {0.1, 0.3, 0.55, 0.9}, {n, n, n}));
    struct Angle {
        double theta, phi;
    };
    double worst_dense = 0.0;
    for (const Angle& a : {Angle{35.0, 19.0}, Angle{0.0, 0.0}, Angle{0.0, 64.0}, Angle{20.0, 0.0}, Angle{-48.0, 90.0}}) {
        const rcwa::ConicalResult u = rcwa::solve_conical(uniform, 1.053, a.theta, a.phi, 6);
        // 快速路径：与写成均匀层逐位相同，只有真正的光栅层做本征分解
        const rcwa::ConicalResult g = rcwa::solve_conical(same, 1.053, a.theta, a.phi, 6);
        CHECK(same_bits(u, g) && g.n_eig == 2 && u.n_eig == 2);
        // 稠密路径：它与均匀层一样走本征分解（两次分解的谱重合），结果仍然一致
        rcwa::Options dense;
        dense.homogeneous_fast_path = false;
        const rcwa::ConicalResult gd = rcwa::solve_conical(same, 1.053, a.theta, a.phi, 6, dense);
        const rcwa::ConicalResult ud = rcwa::solve_conical(uniform, 1.053, a.theta, a.phi, 6, dense);
        CHECK(gd.n_eig == 6 && ud.n_eig == 6);
        CHECK_LE(jones_gap(u, gd), 1e-10);
        CHECK_LE(jones_gap(u, ud), 1e-10);
        worst_dense = worse(worst_dense, worse(jones_gap(u, gd), jones_gap(u, ud)));
        dense.eig = EigMethod::LR;
        CHECK_LE(jones_gap(u, rcwa::solve_conical(same, 1.053, a.theta, a.phi, 6, dense)), 1e-9);
        // 弱对比：与均匀层的差别正比于折射率差
        double previous = 0.0;
        for (double dn : {1e-3, 1e-6, 1e-9}) {
            const rcwa::Stack weak = make(rcwa::Layer::grating(0.27, {0.1, 0.3, 0.55, 0.9}, {n + dn, n, n + 0.5 * dn}));
            const rcwa::ConicalResult w = rcwa::solve_conical(weak, 1.053, a.theta, a.phi, 6);
            const double gap = jones_gap(u, w);
            CHECK(w.n_eig == 4 && gap > 0.0);
            CHECK_LE(gap, 30.0 * dn);
            if (previous > 0.0) CHECK_LE(std::abs(gap / previous - 1e-3), 2e-5);     // 线性：差别也缩小 1000 倍
            previous = gap;
        }
    }
    say("  同折射率的光栅层：快速路径下与均匀层逐位相同；稠密路径下相差 %.1e\n", worst_dense);
    // 纯膜系里夹着这样的层：整个结构仍是纯膜系，没有任何稠密运算
    {
        rcwa::Stack films = conical_ref_stack(10), mixed = films;
        mixed.layers[1] = rcwa::Layer::grating(films.layers[1].d, {0.0, 0.4, 1.0}, {films.layers[1].n, films.layers[1].n});
        const rcwa::ConicalResult a = rcwa::solve_conical(films, 0.633, 50.0, 33.0, 3), b = rcwa::solve_conical(mixed, 0.633, 50.0, 33.0, 3);
        CHECK(same_bits(a, b) && b.n_eig == 0 && b.n_dense_solve == 0);
    }
}

// ---------------------------------------------------------------------------------------------
static void test_sweep_and_threads() {
    say("[扫描与线程数]\n");
    const rcwa::Stack stack = conical_ref_stack(3);
    std::vector<rcwa::ConicalPoint> points;
    for (int k = 0; k < 7; ++k) points.push_back({1.03 + 0.008 * k, 60.0 + 1.5 * k, -20.0 + 13.0 * k});
    const int M = 8;

    set_num_threads(1);
    std::vector<rcwa::ConicalResult> serial;
    for (const rcwa::ConicalPoint& p : points) serial.push_back(rcwa::solve_conical(stack, p.wavelength, p.theta_deg, p.phi_deg, M));
    CHECK(!same_bits(serial[0], serial[1]));                  // 各点确实不同
    for (int threads : {1, 2, 4}) {
        set_num_threads(threads);
        const std::vector<rcwa::ConicalResult> swept = rcwa::sweep_conical(stack, points, M);
        bool same = swept.size() == points.size();
        for (std::size_t i = 0; same && i < points.size(); ++i)
            same = same_bits(swept[i], serial[i]) && swept[i].n_eig == 2 && swept[i].eig_info == 0 &&
                   swept[i].n_dense_solve == serial[i].n_dense_solve;
        CHECK(same);
    }
    set_num_threads(4);
    CHECK(rcwa::sweep_conical(stack, {}, M).empty());
    // 选项原样传给每个点
    {
        rcwa::Options opt;
        opt.homogeneous_fast_path = false;
        opt.linear = rcwa::LinearSolver::QRHouseholder;
        opt.eig = EigMethod::LR;
        const std::vector<rcwa::ConicalResult> swept = rcwa::sweep_conical(stack, points, 5, opt);
        set_num_threads(1);
        bool same = swept.size() == points.size();
        for (std::size_t i = 0; same && i < points.size(); ++i)
            same = swept[i].n_eig == 42 &&
                   same_bits(swept[i], rcwa::solve_conical(stack, points[i].wavelength, points[i].theta_deg, points[i].phi_deg, 5, opt));
        CHECK(same);
    }

    // 单次求解内部的并行（矩阵乘法、LU 的列并行）也不改变任何一位：阶数取大一些让并行分支真正用上
    {
        const rcwa::Stack wide = ridges_stack();
        for (bool fast : {true, false})
            for (rcwa::LinearSolver lin : {rcwa::LinearSolver::LUPartial, rcwa::LinearSolver::QRHouseholder}) {
                rcwa::Options opt;
                opt.homogeneous_fast_path = fast;
                opt.linear = lin;
                const int m = fast ? 36 : 24;                   // 模式数 146、98
                set_num_threads(1);
                const rcwa::ConicalResult one = rcwa::solve_conical(wide, kRidgesWavelength, kRidgesTheta, 5.0, m, opt);
                CHECK_LE(unitarity_defect(one), 1e-9);
                for (int threads : {2, 4}) {
                    set_num_threads(threads);
                    CHECK(same_bits(one, rcwa::solve_conical(wide, kRidgesWavelength, kRidgesTheta, 5.0, m, opt)));
                    opt.parallel_layers = true;
                    CHECK(same_bits(one, rcwa::solve_conical(wide, kRidgesWavelength, kRidgesTheta, 5.0, m, opt)));
                    opt.parallel_layers = false;
                }
            }
    }
    set_num_threads(1);
}

// ---------------------------------------------------------------------------------------------
static void test_parallel_layers_and_custom_eig() {
    say("[各层各支的本征分解并行、自定义本征求解器]\n");
    const rcwa::Stack stack = trapezoid_stack(5);             // 5 个光栅层 + 2 层膜
    const int M = 8, N = 2 * M + 1;
    const double wavelength = 1.053, theta = 66.36, phi = 12.0;
    set_num_threads(1);
    const rcwa::ConicalResult base = rcwa::solve_conical(stack, wavelength, theta, phi, M);
    CHECK(base.n_eig == 10 && base.eig_info == 0);
    CHECK_LE(unitarity_defect(base), 1e-10);

    rcwa::Options par;
    par.parallel_layers = true;
    CHECK(same_bits(base, rcwa::solve_conical(stack, wavelength, theta, phi, M, par)));      // 线程数为 1：等同串行
    for (int threads : {2, 4}) {
        set_num_threads(threads);
        const rcwa::ConicalResult r = rcwa::solve_conical(stack, wavelength, theta, phi, M, par);
        CHECK(same_bits(base, r) && r.n_eig == 10 && r.n_dense_solve == base.n_dense_solve);
    }
    // 均匀层也做本征分解时共 7 层、14 次分解
    par.homogeneous_fast_path = false;
    set_num_threads(1);
    const rcwa::ConicalResult dense_serial = rcwa::solve_conical(stack, wavelength, theta, phi, M, par);
    set_num_threads(4);
    const rcwa::ConicalResult dense_parallel = rcwa::solve_conical(stack, wavelength, theta, phi, M, par);
    CHECK(same_bits(dense_serial, dense_parallel) && dense_parallel.n_eig == 14);
    CHECK_LE(jones_gap(base, dense_parallel), 1e-10);
    set_num_threads(1);

    // custom_eig 里调用 eig_qr：与默认结果逐位相同，每个光栅层调用两次，矩阵都是 N 阶；它取代 options.eig
    int calls = 0;
    bool sizes_ok = true;
    rcwa::Options custom;
    custom.custom_eig = [&](const Matrix& a) {
        ++calls;
        sizes_ok = sizes_ok && a.rows() == N && a.cols() == N;
        return eig_qr(a);
    };
    const rcwa::ConicalResult rc = rcwa::solve_conical(stack, wavelength, theta, phi, M, custom);
    CHECK(same_bits(base, rc) && calls == 10 && rc.n_eig == 10 && sizes_ok);
    custom.eig = EigMethod::LR;
    calls = 0;
    CHECK(same_bits(base, rcwa::solve_conical(stack, wavelength, theta, phi, M, custom)) && calls == 10);
    rcwa::Options lr;
    lr.eig = EigMethod::LR;
    const rcwa::ConicalResult rl = rcwa::solve_conical(stack, wavelength, theta, phi, M, lr);
    CHECK(!same_bits(base, rl));                              // 确认 LR 的结果在末位上确实不同
    CHECK_LE(jones_gap(base, rl), 1e-9);
    // 快速路径下纯膜系不调用它
    calls = 0;
    CHECK(solve_ref(10, custom).n_eig == 0 && calls == 0);

    // 本征对的排列顺序、本征向量的归一和相位不影响物理结果
    rcwa::Options shuffled;
    shuffled.custom_eig = [](const Matrix& a) {
        EigResult r = eig_qr(a);
        const int n = a.rows();
        EigResult out = r;
        for (int j = 0; j < n; ++j) {
            const int src = n - 1 - j;
            const cplx factor = (0.3 + 0.1 * j) * std::exp(cplx(0.0, 0.13 + j));
            out.values[j] = r.values[src];
            for (int i = 0; i < n; ++i) out.vectors(i, j) = r.vectors(i, src) * factor;
        }
        return out;
    };
    CHECK_LE(jones_gap(base, rcwa::solve_conical(stack, wavelength, theta, phi, M, shuffled)), 1e-10);

    // 可被多个线程同时调用的 custom_eig 与 parallel_layers、sweep_conical 同用
    rcwa::Options both;
    both.parallel_layers = true;
    both.custom_eig = [](const Matrix& a) { return eig_qr(a); };
    set_num_threads(4);
    CHECK(same_bits(base, rcwa::solve_conical(stack, wavelength, theta, phi, M, both)));
    const std::vector<rcwa::ConicalResult> swept =
        rcwa::sweep_conical(stack, {{wavelength, theta, phi}, {1.06, 64.0, 12.0}, {wavelength, theta, phi}, {wavelength, theta, -12.0}}, M, both);
    CHECK(swept.size() == 4 && same_bits(base, swept[0]) && same_bits(base, swept[2]) && !same_bits(base, swept[1]) &&
          !same_bits(base, swept[3]));
    set_num_threads(1);

    // 交给本征求解器的两个矩阵。平面内入射（φ = 0）时第一个就是 rcwa::layer_matrix 的 TE 层矩阵 Kx² − ⟦ε⟧；
    // 第二个 B·C 与 TM 层矩阵 C·B 相似，本征值相同（无吸收光栅的本征值都是实数，排序之后逐个比较）。
    // 方位角不为零时两个矩阵的 kx 换成 nI·sin θ·cos φ − m·λ/Λ，ky 不进入矩阵：θ' 满足 sin θ' = sin θ·cos φ 时矩阵相同。
    {
        const rcwa::Stack single = conical_ref_stack(1);
        std::vector<Matrix> seen;
        rcwa::Options spy;
        spy.custom_eig = [&seen](const Matrix& a) {
            seen.push_back(a);
            return eig_qr(a);
        };
        const auto sorted_real = [](const Matrix& a) {
            const EigResult e = eig_qr(a);
            std::vector<double> v;
            for (const cplx& z : e.values) v.push_back(z.real());
            std::sort(v.begin(), v.end());
            return v;
        };
        for (double azimuth : {0.0, 40.0}) {
            seen.clear();
            rcwa::solve_conical(single, wavelength, theta, azimuth, M, spy);
            const double effective = std::asin(std::sin(theta * kPi / 180.0) * std::cos(azimuth * kPi / 180.0)) * 180.0 / kPi;
            const Matrix te = rcwa::layer_matrix(single.layers[0], single.period, wavelength, single.nI, effective, M, rcwa::Polarization::TE);
            const Matrix tm = rcwa::layer_matrix(single.layers[0], single.period, wavelength, single.nI, effective, M, rcwa::Polarization::TM);
            CHECK(seen.size() == 2 && seen[0].rows() == N && seen[1].rows() == N);
            if (seen.size() != 2) continue;
            CHECK_LE(max_abs(seen[0] - te), 1e-13 * max_abs(te));
            const std::vector<double> mine = sorted_real(seen[1]), reference = sorted_real(tm);
            double gap = 0.0;
            for (std::size_t j = 0; j < mine.size() && j < reference.size(); ++j) gap = worse(gap, std::abs(mine[j] - reference[j]));
            CHECK_LE(gap, 1e-9 * max_abs(tm));
        }
    }
}

// ---------------------------------------------------------------------------------------------
static void test_bookkeeping() {
    say("[计数、计时、诊断量]\n");
    for (int id : {3, 4}) {
        const rcwa::ConicalResult r = solve_ref(id);
        CHECK(r.max_eig_residual == 0.0 && r.max_solve_backward_error == 0.0);       // 没开诊断
        const rcwa::Timing& t = r.timing;                      // 三项互不重叠，都包含在 total 里
        CHECK(t.total > 0.0 && t.eig > 0.0 && t.linear > 0.0 && t.product > 0.0);
        CHECK(t.eig + t.linear + t.product <= t.total * (1.0 + 1e-9));

        rcwa::Options diag;
        diag.diagnostics = true;
        const rcwa::ConicalResult d = solve_ref(id, diag);
        CHECK(same_bits(r, d) && d.n_eig == r.n_eig && d.n_dense_solve == r.n_dense_solve);   // 诊断不改变结果
        CHECK(d.max_eig_residual > 0.0 && d.max_eig_residual < 1e-13);
        CHECK(d.max_solve_backward_error > 0.0 && d.max_solve_backward_error < 1e-13);

        diag.eig = EigMethod::LR;                             // LR 迭代的残差明显大于 QR 迭代
        diag.homogeneous_fast_path = false;
        const rcwa::ConicalResult dl = solve_ref(id, diag);
        CHECK(dl.n_eig == 2 * static_cast<int>(conical_ref_stack(id).layers.size()));
        CHECK(dl.max_eig_residual > d.max_eig_residual && dl.max_eig_residual < 1e-8);
        if (id == 3)
            say("  本征对残差：QR %.1e，LR（全部 21 层）%.1e；稠密求解的后向误差 %.1e\n", d.max_eig_residual,
                dl.max_eig_residual, d.max_solve_backward_error);
    }
    // 残差两支都算：只把第二次分解（B·C）的本征向量弄坏，max_eig_residual 如实变大
    {
        int calls = 0;
        rcwa::Options opt;
        opt.diagnostics = true;
        opt.custom_eig = [&calls](const Matrix& a) {
            EigResult r = eig_qr(a);
            if (++calls == 2) r.vectors(0, 0) += 1e-3;
            return r;
        };
        const rcwa::ConicalResult r = solve_ref(1, opt);
        CHECK(calls == 2 && r.max_eig_residual > 1e-6 && r.max_eig_residual < 1e-2);
    }
    // 纯膜系全是逐元素运算：三项计时严格为零，没有本征分解，没有稠密求解；选项不起作用（稠密路径除外）
    {
        const rcwa::ConicalResult f = solve_ref(10);
        CHECK(f.timing.eig == 0.0 && f.timing.linear == 0.0 && f.timing.product == 0.0 && f.timing.total > 0.0);
        CHECK(f.n_eig == 0 && f.n_dense_solve == 0);
        rcwa::Options other;
        other.eig = EigMethod::LR;
        other.linear = rcwa::LinearSolver::QRHouseholder;
        other.parallel_layers = true;
        other.diagnostics = true;
        const rcwa::ConicalResult g = solve_ref(10, other);
        CHECK(same_bits(f, g) && g.max_eig_residual == 0.0 && g.max_solve_backward_error == 0.0);
        // 只有零级被激发
        bool only_zero_order = true;
        const int M = ref_case(10).M;
        for (int j = 0; j <= 2 * M; ++j)
            if (j != M)
                for (const cplx& v : entries(f.Jr[j])) only_zero_order = only_zero_order && v == cplx{};
        CHECK(only_zero_order && f.Jr[M].ss != cplx{} && f.Jr[M].pp != cplx{});
    }
    // 几次本征分解都没收敛时，eig_info 取（自上而下、每层先 A 后 B·C）第一个失败的 info，与完成的先后无关
    for (bool parallel : {false, true}) {
        const rcwa::Stack s = trapezoid_stack(4);            // 4 个光栅层，齿宽各不相同
        std::vector<Matrix> order;                            // 先串行记下各次分解的矩阵
        rcwa::Options spy;
        spy.custom_eig = [&order](const Matrix& a) {
            order.push_back(a);
            return eig_qr(a);
        };
        rcwa::solve_conical(s, 1.053, 66.36, 12.0, 5, spy);
        CHECK(order.size() == 8);
        if (order.size() != 8) continue;
        rcwa::Options opt;
        opt.parallel_layers = parallel;
        opt.custom_eig = [&order](const Matrix& a) {
            EigResult r = eig_qr(a);
            const int code[8] = {0, 0, 0, 5, 2, 0, 9, 4};     // 第 3、4、6、7 次分解（0 起）报告失败
            for (int k = 0; k < 8; ++k)
                if (code[k] != 0 && std::memcmp(a.data(), order[k].data(), sizeof(cplx) * a.rows() * a.cols()) == 0) {
                    r.info = code[k];
                    r.vectors = Matrix();
                }
            return r;
        };
        set_num_threads(parallel ? 4 : 1);
        const rcwa::ConicalResult r = rcwa::solve_conical(s, 1.053, 66.36, 12.0, 5, opt);
        set_num_threads(1);
        CHECK(r.eig_info == 5 && all_nan(r) && r.n_eig == 8);
    }
    // parallel_layers、sweep_conical 确实把工作分给了线程池里别的线程：用 custom_eig 记下是哪些线程在调用它
    {
        std::mutex guard;
        std::set<std::thread::id> callers;
        rcwa::Options opt;
        opt.custom_eig = [&](const Matrix& a) {
            {
                std::lock_guard<std::mutex> lock(guard);
                callers.insert(std::this_thread::get_id());
            }
            return eig_qr(a);
        };
        const rcwa::Stack s = trapezoid_stack(6);            // 6 个光栅层，12 次分解
        const std::vector<rcwa::ConicalPoint> points(8, rcwa::ConicalPoint{1.053, 66.36, 12.0});
        set_num_threads(4);
        rcwa::solve_conical(s, 1.053, 66.36, 12.0, 6, opt);
        CHECK(callers.size() == 1 && *callers.begin() == std::this_thread::get_id());    // 没开 parallel_layers：都在调用线程上
        callers.clear();
        opt.parallel_layers = true;
        const rcwa::ConicalResult r = rcwa::solve_conical(s, 1.053, 66.36, 12.0, 6, opt);
        CHECK(callers.size() >= 2 && callers.size() <= 4);
        CHECK(r.timing.eig > 0.0 && r.timing.eig + r.timing.linear + r.timing.product <= r.timing.total * (1.0 + 1e-9));
        // 只有一个光栅层时两支也分给两个线程
        callers.clear();
        solve_ref(1, opt);
        CHECK(callers.size() == 2);
        callers.clear();
        opt.parallel_layers = false;
        const std::vector<rcwa::ConicalResult> swept = rcwa::sweep_conical(s, points, 6, opt);
        CHECK(callers.size() >= 2 && callers.size() <= 4);
        bool timing_ok = swept.size() == points.size();
        for (const rcwa::ConicalResult& p : swept)
            timing_ok = timing_ok && p.timing.total > 0.0 && p.timing.eig + p.timing.linear + p.timing.product <= p.timing.total * (1.0 + 1e-9);
        CHECK(timing_ok);
        set_num_threads(1);                                   // 线程数为 1：全部在调用线程上
        callers.clear();
        opt.parallel_layers = true;
        rcwa::sweep_conical(s, points, 6, opt);
        CHECK(callers.size() == 1 && *callers.begin() == std::this_thread::get_id());
    }
}

// ---------------------------------------------------------------------------------------------
// 平面内入射（φ = 0）与 rcwa::solve 的对应关系只在这里核对接线是否正确（符号 σ_m、因子 nI/nII），
// 系统的物理核对在 test_rcwa1d_conical_physics.cpp。
static void test_in_plane_limit() {
    say("[平面内入射：两支严格不耦合]\n");
    for (int id : {1, 4}) {
        const ConicalRefCase& c = ref_case(id);
        const rcwa::Stack stack = conical_ref_stack(id);
        const int M = c.M;
        for (double theta : {c.theta_deg, -c.theta_deg, 0.0}) {
            const rcwa::ConicalResult r = rcwa::solve_conical(stack, c.wavelength, theta, 0.0, M);
            // 交叉偏振的矩阵元严格为零（不是 1e-17：ky = 0 时两组模式之间没有任何运算把它们连起来）
            bool decoupled = r.ky == 0.0;
            for (int j = 0; j <= 2 * M; ++j)
                decoupled = decoupled && r.Jr[j].sp == cplx{} && r.Jr[j].ps == cplx{} && r.Jt[j].sp == cplx{} && r.Jt[j].ps == cplx{};
            CHECK(decoupled);
            // s 对 TE、p 对 TM：第 m 级多一个符号 σ_m·σ_0（σ = kx 的符号，零取正），p 的透射再乘 nI/nII
            const rcwa::Result te = rcwa::solve(stack, c.wavelength, theta, M, rcwa::Polarization::TE);
            const rcwa::Result tm = rcwa::solve(stack, c.wavelength, theta, M, rcwa::Polarization::TM);
            const double sigma0 = r.kx[M] < 0.0 ? -1.0 : 1.0;
            const double ratio = stack.nI.real() / stack.nII.real();
            double gap = 0.0;
            for (int j = 0; j <= 2 * M; ++j) {
                const double sign = (r.kx[j] < 0.0 ? -1.0 : 1.0) * sigma0;
                gap = worse(gap, std::abs(r.Jr[j].ss - sign * te.R[j]) + std::abs(r.Jt[j].ss - sign * te.T[j]));
                gap = worse(gap, std::abs(r.Jr[j].pp - sign * tm.R[j]) + std::abs(r.Jt[j].pp - sign * ratio * tm.T[j]));
            }
            CHECK_LE(gap, 1e-10);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// solve_conical() 可重入（没有全局可变状态）：用户自己的线程并发调用、在 custom_eig 回调里再调用它，
// 结果都与单独调用逐位相同。
static void test_reentrancy() {
    say("[可重入：用户线程并发调用、回调里嵌套调用]\n");
    struct Job {
        rcwa::Stack stack;
        double wavelength, theta, phi;
        int M;
        rcwa::Options opt;
    };
    std::vector<Job> jobs;
    {
        rcwa::Options plain, dense, lr, diag, par;
        dense.homogeneous_fast_path = false;
        dense.linear = rcwa::LinearSolver::QRHouseholder;
        lr.eig = EigMethod::LR;
        diag.diagnostics = true;
        par.parallel_layers = true;
        jobs.push_back({conical_ref_stack(3), 1.053, 66.36, 5.0, 8, plain});
        jobs.push_back({conical_ref_stack(4), 0.83, 27.0, 63.0, 7, dense});
        jobs.push_back({trapezoid_stack(4), 1.053, 66.36, -30.0, 6, par});
        jobs.push_back({conical_ref_stack(2), 1.053, -35.0, 20.0, 8, lr});
        jobs.push_back({conical_ref_stack(8), 1.053, 66.36, 12.0, 9, diag});
        jobs.push_back({ridges_stack(), kRidgesWavelength, kRidgesTheta, 5.0, 26, par});   // 模式数 106：内部的乘法也去用线程池
        jobs.push_back({conical_ref_stack(10), 0.633, 50.0, 33.0, 3, plain});
    }
    set_num_threads(1);
    std::vector<rcwa::ConicalResult> expect;
    for (const Job& j : jobs) expect.push_back(rcwa::solve_conical(j.stack, j.wavelength, j.theta, j.phi, j.M, j.opt));
    for (int pool : {1, 3}) {                                   // 线程池的线程数；用户线程另有 4 个
        set_num_threads(pool);
        std::vector<rcwa::ConicalResult> got(3 * jobs.size());
        std::vector<std::thread> users;
        for (std::size_t t = 0; t < 4; ++t)
            users.emplace_back([&, t] {
                for (std::size_t i = t; i < got.size(); i += 4) {
                    const Job& j = jobs[i % jobs.size()];
                    got[i] = rcwa::solve_conical(j.stack, j.wavelength, j.theta, j.phi, j.M, j.opt);
                }
            });
        for (std::thread& t : users) t.join();
        bool same = true;
        for (std::size_t i = 0; i < got.size(); ++i) same = same && same_bits(got[i], expect[i % jobs.size()]);
        CHECK(same);
    }

    // 回调里再完整地解另一个问题（嵌套调用），圆锥的和平面内的各一个
    {
        set_num_threads(1);
        const rcwa::Stack outer = trapezoid_stack(4), inner = conical_ref_stack(4);
        const rcwa::ConicalResult outer_expect = rcwa::solve_conical(outer, 1.053, 66.36, 12.0, 6);
        const rcwa::ConicalResult inner_expect = rcwa::solve_conical(inner, 0.83, 27.0, 63.0, 4);
        const rcwa::Result plane_expect = rcwa::solve(inner, 0.83, 27.0, 4, rcwa::Polarization::TM);
        std::mutex guard;
        int inner_calls = 0, inner_same = 0;
        rcwa::Options nested;
        nested.parallel_layers = true;
        nested.custom_eig = [&](const Matrix& a) {
            const rcwa::ConicalResult r = rcwa::solve_conical(inner, 0.83, 27.0, 63.0, 4);
            const rcwa::Result p = rcwa::solve(inner, 0.83, 27.0, 4, rcwa::Polarization::TM);
            {
                std::lock_guard<std::mutex> lock(guard);
                ++inner_calls;
                if (same_bits(r, inner_expect) && same_bytes(p.R, plane_expect.R) && same_bytes(p.T, plane_expect.T)) ++inner_same;
            }
            return eig_qr(a);
        };
        for (int pool : {1, 4}) {
            set_num_threads(pool);
            CHECK(same_bits(outer_expect, rcwa::solve_conical(outer, 1.053, 66.36, 12.0, 6, nested)));
        }
        set_num_threads(1);
        CHECK(inner_calls == 16 && inner_same == 16);
    }
}

// ---------------------------------------------------------------------------------------------
// 圆锥入射特有的模式构造退化（说明见 rcwa1d_conical.hpp 开头）：某个做本征分解的层里 A = Kx² − ⟦ε⟧ 的
// 本征值过零，而 ky ≠ 0。判据用的量是 η = min_abs_lambda / |ky|^1.5：η < 1e-2 置 degenerate_mode_warning，
// 同时仅该层切换为完整 Schur 传播，η 再小也保持有限且准确的结果。

// 第一层的 A 的负本征值个数。A 只通过 kx_0 = nI·sin θ·cos φ 依赖入射方向：取 sin θ' = sin θ·cos φ 的平面内入射，
// rcwa::layer_matrix 给出的 TE 层矩阵就是它。无吸收时 A 是厄米的，本征值都是实数，每有一个过零，个数就变 1。
static int negative_eigenvalues(const rcwa::Stack& s, double wavelength, double theta, double phi, int M) {
    const double effective = std::asin(std::sin(theta * kPi / 180.0) * std::cos(phi * kPi / 180.0)) * 180.0 / kPi;
    const Matrix a = rcwa::layer_matrix(s.layers[0], s.period, wavelength, s.nI, effective, M, rcwa::Polarization::TE);
    EigOptions values_only;
    values_only.vectors = false;
    int count = 0;
    for (const cplx& v : eig_qr(a, values_only).values) count += v.real() < 0.0 ? 1 : 0;
    return count;
}

// 取整数值的函数 count 在 [lo, hi] 里跳变一次：二分到区间不能再缩，返回跳变的位置
template <class Count>
static double bisect_step(Count&& count, double lo, double hi) {
    const int at_lo = count(lo);
    for (int it = 0; it < 200; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (mid == lo || mid == hi) break;
        if (count(mid) == at_lo) lo = mid;
        else hi = mid;
    }
    return 0.5 * (lo + hi);
}

// 各级 Jones 矩阵的全部矩阵元排成一列
static std::vector<cplx> jones_entries(const rcwa::ConicalResult& r) {
    std::vector<cplx> v;
    for (std::size_t j = 0; j < r.Jr.size(); ++j)
        for (const rcwa::Jones* a : {&r.Jr[j], &r.Jt[j]})
            for (const cplx& z : entries(*a)) v.push_back(z);
    return v;
}
// 两列数之差的最大模，按 max(1, |参照值|) 归一
static double scaled_gap(const std::vector<cplx>& value, const std::vector<cplx>& reference) {
    if (value.size() != reference.size()) return std::numeric_limits<double>::infinity();
    double gap = 0.0;
    for (std::size_t k = 0; k < value.size(); ++k)
        gap = worse(gap, std::abs(value[k] - reference[k]) / (std::max)(1.0, std::abs(reference[k])));
    return gap;
}

static void test_degenerate_modes() {
    say("[圆锥入射特有的奇点：光栅层 A 的本征值过零]\n");
    const double inf = std::numeric_limits<double>::infinity();
    const rcwa::Stack s = conical_ref_stack(1);                // 单层无吸收光栅
    const double theta = 40.0, phi = 25.0;
    const int M = 8;
    const auto solve_at = [&](double wavelength, const rcwa::Options& opt = {}) {
        return rcwa::solve_conical(s, wavelength, theta, phi, M, opt);
    };

    // (1) 对波长二分，找到 A 的一个本征值过零的波长。恰好落在上面：该层切换为 Schur 传播，警告置上，
    //     min_abs_lambda 与只由入射条件决定的量照常给出
    const auto count = [&](double wavelength) { return negative_eigenvalues(s, wavelength, theta, phi, M); };
    CHECK(count(0.9) - count(1.05) == 1);
    const double t0 = bisect_step(count, 0.9, 1.05);
    const rcwa::ConicalResult at = solve_at(t0);
    const double ky15 = std::abs(at.ky) * std::sqrt(std::abs(at.ky));
    CHECK(all_finite(at) && at.eig_info == 0 && at.n_eig == 3 && at.n_schur == 1 && at.degenerate_mode_warning);
    CHECK(at.min_abs_lambda < 1e-12 && at.ky > 0.2 && static_cast<int>(at.kx.size()) == 2 * M + 1);
    CHECK_LE(std::abs(at.flux_r[M] - 1.0), 1e-15);

    // A mixed stack switches only the degenerate layer; custom callbacks still see
    // the original N dimensional problems, and parallel assembly is deterministic.
    {
        rcwa::Stack mixed=s;
        mixed.layers.push_back(rcwa::Layer::uniform(.2,1.7));
        mixed.layers.push_back(rcwa::Layer::grating(.1,{0.,.15,mixed.period},{1.8,1.1}));
        rcwa::Options parallel; parallel.parallel_layers=true;
        const auto serial=rcwa::solve_conical(mixed,t0,theta,phi,M);
        const auto par=rcwa::solve_conical(mixed,t0,theta,phi,M,parallel);
        CHECK(all_finite(serial) && serial.n_schur==1 && serial.n_eig==5);
        CHECK(same_bits(serial,par));
        CHECK_LE(unitarity_defect(serial),1e-10);
        rcwa::Options callback;
        int calls=0;
        callback.custom_eig=[&](const Matrix& a) {
            CHECK(a.rows()==2*M+1); ++calls; return eig_qr(a);
        };
        CHECK(same_bits(serial,rcwa::solve_conical(mixed,t0,theta,phi,M,callback)) && calls==4);
        rcwa::Stack thick=s; thick.layers[0].d=20.;
        const auto r=rcwa::solve_conical(thick,t0,theta,phi,M);
        CHECK(all_finite(r) && r.n_schur==1);
        CHECK_LE(unitarity_defect(r),1e-9);
    }

    // min|λ| 随相对偏离 δ 线性变化。记下斜率，之后直接按 η 取点：η = 1 对应的相对偏离是 unit，eta 的正负表示在哪一侧。
    // （量斜率的这两个点已经切换为 Schur 传播。）
    const double probe = 1e-6;
    const rcwa::ConicalResult above = solve_at(t0 * (1.0 + probe)), below = solve_at(t0 * (1.0 - probe));
    CHECK(all_finite(above) && all_finite(below) && above.n_schur == 1 && below.n_schur == 1);
    CHECK_LE(std::abs(above.min_abs_lambda / below.min_abs_lambda - 1.0), 1e-3);
    const double slope = 0.5 * (above.min_abs_lambda + below.min_abs_lambda) / probe;
    CHECK(slope > 1.0 && slope < 10.0);
    const double unit = ky15 / slope;
    const auto at_eta = [&](double eta, const rcwa::Options& opt = {}) { return solve_at(t0 * (1.0 + eta * unit), opt); };
    say("  奇点波长 %.12f，ky = %.4f，d(min|λ|)/dδ = %.3f：Schur 切换窗口是相对偏离 %.1e 以内，深处取点尺度 %.1e\n", t0, at.ky,
        slope, 1e-2 * unit, 1e-4 * unit);

    // (2) Schur 切换的范围，两侧都看
    for (double side : {1.0, -1.0}) {
        for (double eta : {3e-2, 1.2e-2}) {                     // 窗口之外：不置警告，结果有限而且通量幺正
            const rcwa::ConicalResult r = at_eta(side * eta);
            CHECK(!r.degenerate_mode_warning && all_finite(r) && r.eig_info == 0);
            CHECK_LE(std::abs(r.min_abs_lambda / (eta * ky15) - 1.0), 0.05);
            CHECK_LE(unitarity_defect(r), 1e-10);
        }
        for (double eta : {8e-3, 1e-3, 1.5e-4}) {               // 警告的窗口里：置警告，结果仍然有限
            const rcwa::ConicalResult r = at_eta(side * eta);
            CHECK(r.degenerate_mode_warning && all_finite(r) && r.eig_info == 0);
            CHECK_LE(std::abs(r.min_abs_lambda / (eta * ky15) - 1.0), 0.05);
        }
        for (double eta : {7e-5, 1e-6, 1e-9}) {                 // 再近：仍使用 Schur 传播
            const rcwa::ConicalResult r = at_eta(side * eta);
            CHECK(r.degenerate_mode_warning && all_finite(r) && r.eig_info == 0 && r.n_eig == 3 && r.n_schur == 1);
        }
    }

    // (3) min_abs_lambda 就是交给本征求解器的两个矩阵（A 与 B·C）全部本征值的模的最小值；与 diagnostics 无关
    {
        std::vector<Matrix> seen;
        rcwa::Options spy;
        spy.custom_eig = [&seen](const Matrix& a) {
            seen.push_back(a);
            return eig_qr(a);
        };
        const rcwa::ConicalResult r = at_eta(5e-3, spy);
        double smallest = inf;
        for (const Matrix& a : seen)
            for (const cplx& v : eig_qr(a).values) smallest = (std::min)(smallest, std::abs(v));
        CHECK(seen.size() == 2 && r.min_abs_lambda == smallest && r.degenerate_mode_warning);
        rcwa::Options diag;
        diag.diagnostics = true;
        CHECK(same_bits(r, at_eta(5e-3)) && same_bits(r, at_eta(5e-3, diag)));
    }
    // 齿带一点吸收时本征值是复数，过不了零：min_abs_lambda 取的是模（在无吸收的奇点波长上，实部几乎是零、虚部不是）。
    // κ = 1e-3 时 η ≈ 3.6e-3，置警告而结果有限；κ = 1e-2 时 η ≈ 3.6e-2，不置警告
    for (double kappa : {1e-3, 1e-2}) {
        rcwa::Stack lossy = s;
        lossy.layers[0].n_values[0] = cplx(1.46, kappa);
        std::vector<Matrix> seen;
        rcwa::Options spy;
        spy.custom_eig = [&seen](const Matrix& a) {
            seen.push_back(a);
            return eig_qr(a);
        };
        const rcwa::ConicalResult r = rcwa::solve_conical(lossy, t0, theta, phi, M, spy);
        double smallest = inf, smallest_real = inf;
        for (const Matrix& a : seen)
            for (const cplx& v : eig_qr(a).values) {
                smallest = (std::min)(smallest, std::abs(v));
                smallest_real = (std::min)(smallest_real, std::abs(v.real()));
            }
        CHECK(all_finite(r) && r.min_abs_lambda == smallest && smallest > 100.0 * smallest_real);
        CHECK_LE(std::abs(r.min_abs_lambda / ky15 / (3.65 * kappa) - 1.0), 0.05);
        CHECK(r.degenerate_mode_warning == (kappa < 5e-3));
    }
    // 没有做本征分解的层：+∞，不置警告。纯膜系走稠密路径时每层都分解，最小值是 |kx_m² − n²| 里最小的
    {
        const ConicalRefCase& c = ref_case(10);
        const rcwa::Stack films = conical_ref_stack(10);
        const rcwa::ConicalResult fast = solve_ref(10);
        CHECK(fast.min_abs_lambda == inf && !fast.degenerate_mode_warning);
        rcwa::Options dense;
        dense.homogeneous_fast_path = false;
        const rcwa::ConicalResult d = solve_ref(10, dense);
        double smallest = inf;
        for (const rcwa::Layer& layer : films.layers)
            for (int j = 0; j <= 2 * c.M; ++j) smallest = (std::min)(smallest, std::abs(d.kx[j] * d.kx[j] - layer.n * layer.n));
        CHECK(!d.degenerate_mode_warning && smallest > 0.1);
        CHECK_LE(std::abs(d.min_abs_lambda - smallest), 1e-12);
        CHECK(rcwa::ConicalResult().min_abs_lambda == inf && !rcwa::ConicalResult().degenerate_mode_warning);
    }
    // 十个参照算例都离奇点很远
    for (const ConicalRefCase& c : kConicalRefCases) {
        const rcwa::ConicalResult r = solve_ref(c.id);
        const bool decomposed = r.n_eig > 0;
        CHECK(!r.degenerate_mode_warning && (decomposed ? r.min_abs_lambda > 1e-2 && r.min_abs_lambda < 1e3 : r.min_abs_lambda == inf));
    }
    // 本征迭代没有收敛的那几次不参加统计
    {
        int calls = 0;
        rcwa::Options opt;
        opt.custom_eig = [&calls](const Matrix& a) {
            EigResult r = eig_qr(a);
            if (++calls == 1) {
                r.info = 3;
                r.vectors = Matrix();
                for (cplx& v : r.values) v = 0.0;
            }
            return r;
        };
        const rcwa::ConicalResult r = at_eta(3e-2, opt);      // 第一次分解（A）报告失败：只统计 B·C 的本征值
        CHECK(r.eig_info == 3 && all_nan(r) && !r.degenerate_mode_warning);
        CHECK_LE(std::abs(r.min_abs_lambda / (3e-2 * ky15) - 1.0), 0.2);
        opt.custom_eig = [](const Matrix& a) {
            EigOptions one;
            one.max_iter = 1;
            return eig_qr(a, one);
        };
        const rcwa::ConicalResult none = at_eta(3e-2, opt);   // 两次都没收敛
        CHECK(none.eig_info > 0 && all_nan(none) && none.min_abs_lambda == inf && !none.degenerate_mode_warning);
    }

    // (4) ky = 0 时没有这个问题。取 sin θ' = sin θ·cos φ 的平面内入射，A 相同、奇点波长相同：警告不起作用，
    //     离截止点同样近的地方照常求解；圆锥入射在同样的偏离上切换为 Schur 传播（上面量斜率的那两个点）。
    {
        const double effective = std::asin(std::sin(theta * kPi / 180.0) * std::cos(phi * kPi / 180.0)) * 180.0 / kPi;
        for (double delta : {probe, -probe}) {
            const rcwa::ConicalResult r = rcwa::solve_conical(s, t0 * (1.0 + delta), effective, 0.0, M);
            CHECK(r.ky == 0.0 && !r.degenerate_mode_warning && all_finite(r));
            CHECK_LE(std::abs(r.min_abs_lambda / above.min_abs_lambda - 1.0), 1e-3);
            CHECK_LE(unitarity_defect(r), 1e-10);
        }
        for (double delta : {1e-9, -1e-9, 1e-11, -1e-11}) {     // 再近几个量级也一样（这里 q = sqrt(λ) 只有 1e-5 上下）
            const rcwa::ConicalResult r = rcwa::solve_conical(s, t0 * (1.0 + delta), effective, 0.0, M);
            CHECK(!r.degenerate_mode_warning && all_finite(r) && r.min_abs_lambda < 1e-8);
            CHECK_LE(unitarity_defect(r), 1e-9);
        }
        const rcwa::ConicalResult on = rcwa::solve_conical(s, t0, effective, 0.0, M);
        CHECK(!on.degenerate_mode_warning && on.min_abs_lambda < 1e-12 && (all_nan(on) || unitarity_defect(on) < 1e-6));
    }

    // (5) 靠近奇点时保持精度、与两侧的衔接。物理的解在奇点上是光滑的：用窗口外侧两边各三个点
    //     （η = ±1.5e-2、±3e-2、±4.5e-2，都不置警告）定出一个 5 次插值多项式 P，它在窗口里面的值就是“从两侧外推”的值。
    {
        const double step = 1.5e-2;
        const int nodes[6] = {-3, -2, -1, 1, 2, 3};
        std::vector<std::vector<cplx>> values;
        bool clean = true;
        for (int k : nodes) {
            const rcwa::ConicalResult r = at_eta(k * step);
            clean = clean && !r.degenerate_mode_warning && all_finite(r) && unitarity_defect(r) < 1e-10;
            values.push_back(jones_entries(r));
        }
        CHECK(clean);
        const auto predict = [&](double eta, int first, int last) {      // 只用 nodes[first … last] 插值
            std::vector<cplx> p(values[0].size());
            for (int i = first; i <= last; ++i) {
                double weight = 1.0;
                for (int j = first; j <= last; ++j)
                    if (j != i) weight *= (eta - nodes[j] * step) / ((nodes[i] - nodes[j]) * step);
                for (std::size_t e = 0; e < p.size(); ++e) p[e] += weight * values[static_cast<std::size_t>(i)][e];
            }
            return p;
        };
        // 两侧衔接得上：用里面四个点与用全部六个点预言奇点上的值，结果一致（两侧之间若有跳变或尖点，这个差是 O(1) 的）
        const double joint = scaled_gap(predict(0.0, 1, 4), predict(0.0, 0, 5));
        CHECK_LE(joint, 1e-7);

        // 窗口内逐点与仅使用普通模式的外推值比较，检查全部 Jones 分量而不仅是能流。
        const double etas[] = {5e-3, 2.5e-3, 1.2e-3, 6e-4, 3e-4, 1.5e-4};
        double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0, err_far = 0.0, err_near = 0.0, unit_near = 0.0;
        int points = 0;
        bool bounded = true, flagged = true;
        for (double eta : etas)
            for (double side : {1.0, -1.0}) {
                const rcwa::ConicalResult r = at_eta(side * eta);
                const double err = scaled_gap(jones_entries(r), predict(side * eta, 0, 5));
                flagged = flagged && r.degenerate_mode_warning && all_finite(r);
                bounded = bounded && err <= 1.5 * kEps / (eta * eta * eta) + 1e-10;
                if (eta == etas[1]) err_far = worse(err_far, err);
                if (eta == etas[5]) {
                    err_near = worse(err_near, err);
                    unit_near = worse(unit_near, unitarity_defect(r));
                }
                if (eta <= etas[1]) {                           // 拟合只用误差明显高于外推值自身精度的点
                    const double x = std::log10(eta), y = std::log10((std::max)(err, 1e-300));
                    sx += x;
                    sy += y;
                    sxx += x * x;
                    sxy += x * y;
                    ++points;
                }
            }
        const double exponent = (points * sxy - sx * sy) / (points * sxx - sx * sx);
        CHECK(flagged);
        CHECK(bounded);
        CHECK_LE(err_near, 1e-7);
        CHECK_LE(scaled_gap(jones_entries(at), predict(0., 0, 5)), 1e-7);
        CHECK_LE(err_far, 1e-7);              // 深入退化窗口也保持精度
        CHECK_LE(unit_near, 1e-10);
        say("  与两侧外推值的偏差：η = 2.5e-3 时 %.1e，η = 1.5e-4 时 %.1e（通量幺正性偏差 %.1e），拟合的幂次 %.2f；两侧衔接 %.1e\n",
            err_far, err_near, unit_near, exponent, joint);
    }

    // (6) 换成对角度二分：固定波长，对 θ（φ 不变）、对 φ（θ 不变）各找一个奇点。扫描里退化点也返回有限结果
    {
        const double wavelength = 1.0;
        const auto by_theta = [&](double angle) { return negative_eigenvalues(s, wavelength, angle, 25.0, M); };
        const auto by_phi = [&](double angle) { return negative_eigenvalues(s, wavelength, 50.0, angle, M); };
        CHECK(by_theta(45.0) - by_theta(40.0) == 1 && by_phi(30.0) - by_phi(40.0) == 1);
        const double theta0 = bisect_step(by_theta, 40.0, 45.0), phi0 = bisect_step(by_phi, 30.0, 40.0);
        // 角度偏开 0.5° 就在警告的窗口之外（η ≈ 7e-2、2.5e-2）；偏开 0.002° 在 Schur 切换窗口里面（η ≈ 3e-4）
        const std::vector<rcwa::ConicalPoint> points{
            {wavelength, theta0 - 0.5, 25.0}, {wavelength, theta0, 25.0}, {wavelength, theta0 + 0.5, 25.0},
            {wavelength, 50.0, phi0 - 0.5},   {wavelength, 50.0, phi0},   {wavelength, 50.0, phi0 + 0.5},
            {wavelength, theta0 + 2e-3, 25.0},
        };
        set_num_threads(3);
        const std::vector<rcwa::ConicalResult> swept = rcwa::sweep_conical(s, points, M);
        set_num_threads(1);
        bool ok = swept.size() == points.size();
        for (std::size_t i = 0; ok && i < points.size(); ++i) {
            const rcwa::ConicalResult& r = swept[i];
            ok = same_bits(r, rcwa::solve_conical(s, points[i].wavelength, points[i].theta_deg, points[i].phi_deg, M));
            if (i == 1 || i == 4) ok = ok && all_finite(r) && r.n_schur == 1 && r.degenerate_mode_warning && r.eig_info == 0 && r.min_abs_lambda < 1e-12;
            else if (i == 6) ok = ok && all_finite(r) && r.degenerate_mode_warning;
            else ok = ok && all_finite(r) && !r.degenerate_mode_warning && unitarity_defect(r) < 1e-10;
        }
        CHECK(ok);
        say("  对角度二分：λ = 1、φ = 25° 时奇点在 θ = %.9f°；λ = 1、θ = 50° 时在 φ = %.9f°\n", theta0, phi0);
    }

    // (7) 均匀层交给本征分解（homogeneous_fast_path = false）时，奇点是某一级的 kx_m² = n²。光栅层下面加一层 n = 1.7 的膜，
    //     取第 +1 级 kx = −1.7 的波长：稠密路径在这一点上切换 Schur，快速路径（解析模式）不受影响；
    //     偏离之后以快速路径为准，两条路径保持一致。
    {
        rcwa::Stack film = s;
        film.layers.push_back(rcwa::Layer::uniform(0.2, 1.7));
        const double w = (std::sin(theta * kPi / 180.0) * std::cos(phi * kPi / 180.0) + 1.7) * film.period;
        rcwa::Options dense;
        dense.homogeneous_fast_path = false;
        const auto pair_at = [&](double delta, double& eta) {
            const rcwa::ConicalResult f = rcwa::solve_conical(film, w * (1.0 + delta), theta, phi, M);
            const rcwa::ConicalResult d = rcwa::solve_conical(film, w * (1.0 + delta), theta, phi, M, dense);
            eta = d.min_abs_lambda / ky15;
            CHECK(all_finite(f) && !f.degenerate_mode_warning && f.min_abs_lambda > 0.1 && f.n_eig == 2 && d.n_eig == 4 + d.n_schur);
            CHECK_LE(unitarity_defect(f), 1e-10);
            return std::make_pair(f, d);
        };
        double eta = 0.0;
        const auto on = pair_at(0.0, eta);
        CHECK(all_finite(on.second) && on.second.n_schur == 1 && on.second.degenerate_mode_warning && on.second.eig_info == 0 && eta < 1e-9);
        double gap_far = 0.0, gap_near = 0.0;
        for (double side : {1.0, -1.0}) {
            const auto outside = pair_at(side * 1e-3, eta);       // η ≈ 5e-2
            CHECK(eta > 1e-2 && !outside.second.degenerate_mode_warning);
            CHECK_LE(jones_gap(outside.first, outside.second), 1e-10);
            for (double factor : {0.8, 1.0, 1.25}) {              // 每个距离上取相邻的三个点，比较的是其中最大的差
                const auto inside = pair_at(side * factor * 1e-4, eta);        // η ≈ 5e-3
                CHECK(eta < 1e-2 && eta > 1e-3 && inside.second.degenerate_mode_warning && all_finite(inside.second));
                gap_far = worse(gap_far, jones_gap(inside.first, inside.second));
                const auto closer = pair_at(side * factor * 1e-5, eta);        // η ≈ 5e-4
                CHECK(eta < 1e-3 && eta > 1e-4 && closer.second.degenerate_mode_warning && all_finite(closer.second));
                CHECK_LE(jones_gap(closer.first, closer.second), 1.5 * kEps / (eta * eta * eta));
                gap_near = worse(gap_near, jones_gap(closer.first, closer.second));
            }
            const auto failed = pair_at(side * 1e-6, eta);        // η ≈ 5e-5
            CHECK(eta < 1e-4 && all_finite(failed.second) && failed.second.n_schur == 1 && failed.second.degenerate_mode_warning);
        }
        CHECK_LE(gap_near, 1e-10);
        CHECK_LE(gap_far, 1e-10);
        CHECK_LE(jones_gap(on.first, on.second), 1e-10);     // 两条独立模式构造保持一致
        say("  均匀层走本征分解：稠密路径与快速路径相差 %.1e（η ≈ 5e-3）、%.1e（η ≈ 5e-4）\n", gap_far, gap_near);
    }
}

// 接近掠入射：零级的 kz 取 nI·cos θ（不从 nI² − kx² − ky² 开方，那样在 90° 附近丢光有效数字），
// 所以 flux_r 的零级是 1，裸界面的反射率与 Fresnel 公式一致，通量幺正性保持。
static void test_grazing_incidence() {
    say("[接近掠入射的裸界面]\n");
    rcwa::Stack bare;
    bare.nI = 1.0;
    bare.nII = 1.5;
    bare.period = 1.0;
    double worst = 0.0, worst_unit = 0.0;
    for (double theta_deg : {89.9, 89.9999, 89.999999, 89.99999999}) {
        const double c1 = std::sin((90.0 - theta_deg) * kPi / 180.0);
        const double k2 = std::sqrt((1.5 - 1.0) * (1.5 + 1.0) + c1 * c1);
        const double rs = (c1 - k2) / (c1 + k2);
        const double rp = (1.5 * 1.5 * c1 - k2) / (1.5 * 1.5 * c1 + k2);
        for (double phi_deg : {0.0, 30.0, -100.0}) {
            const rcwa::ConicalResult r = rcwa::solve_conical(bare, 0.9, theta_deg, phi_deg, 2);
            CHECK_LE(std::abs(r.flux_r[2] - 1.0), 1e-15);
            const double err = (std::max)(std::abs(r.DEr[2].ss - rs * rs), std::abs(r.DEr[2].pp - rp * rp));
            const std::array<cplx, 4> u = r.flux_unitarity();
            const double unit = (std::max)((std::max)(std::abs(u[0] - 1.0), std::abs(u[3] - 1.0)), (std::max)(std::abs(u[1]), std::abs(u[2])));
            worst = (std::max)(worst, err);
            worst_unit = (std::max)(worst_unit, unit);
            CHECK_LE(err, 1e-11);
            CHECK_LE(unit, 1e-11);
            CHECK_LE(std::abs(r.DEr[2].sp) + std::abs(r.DEr[2].ps), 1e-20);
        }
    }
    say("  θ = 89.9° … 89.99999999°：反射率与 Fresnel 公式最多相差 %.1e，通量幺正性最多偏离 %.1e\n", worst, worst_unit);
}

// 厚度为零的层在物理上不存在：不参加求解，不做本征分解，也不会把它自己的奇点、Rayleigh 点带进来。
static void test_zero_thickness_layers() {
    say("[厚度为零的层]\n");
    rcwa::Stack films;
    films.nI = 1.0;
    films.nII = 1.5;
    films.period = 0.8;
    films.layers.push_back(rcwa::Layer::uniform(0.1, 1.96));
    films.layers.push_back(rcwa::Layer::uniform(0.2, 1.3));
    rcwa::Stack with_ghosts = films;
    with_ghosts.layers.insert(with_ghosts.layers.begin() + 1, rcwa::Layer::grating(0.0, {0.0, 0.3, 0.8}, {1.8, 1.0}));
    with_ghosts.layers.push_back(rcwa::Layer::uniform(0.0, cplx(2.5, 0.4)));
    for (bool fast : {true, false}) {
        rcwa::Options opt;
        opt.homogeneous_fast_path = fast;
        double worst = 0.0;
        for (double wavelength : {0.9, 0.980484673141, 1.053}) {          // 中间那个波长在零厚度光栅层自己的奇点附近
            const rcwa::ConicalResult a = rcwa::solve_conical(films, wavelength, 40.0, 25.0, 8, opt);
            const rcwa::ConicalResult b = rcwa::solve_conical(with_ghosts, wavelength, 40.0, 25.0, 8, opt);
            CHECK(a.n_eig == b.n_eig && a.n_dense_solve == b.n_dense_solve);
            CHECK(a.degenerate_mode_warning == b.degenerate_mode_warning);
            for (std::size_t j = 0; j < a.Jr.size(); ++j) {
                worst = (std::max)(worst, std::abs(a.Jr[j].ss - b.Jr[j].ss) + std::abs(a.Jr[j].sp - b.Jr[j].sp) +
                                              std::abs(a.Jr[j].ps - b.Jr[j].ps) + std::abs(a.Jr[j].pp - b.Jr[j].pp) +
                                              std::abs(a.Jt[j].ss - b.Jt[j].ss) + std::abs(a.Jt[j].sp - b.Jt[j].sp) +
                                              std::abs(a.Jt[j].ps - b.Jt[j].ps) + std::abs(a.Jt[j].pp - b.Jt[j].pp));
            }
        }
        CHECK(worst == 0.0);                                             // 走的是同一条计算路径，逐位相同
        say("  %s：插入两个零厚度的层前后，Jones 矩阵相差 %.1e\n", fast ? "快速路径" : "稠密路径", worst);
    }
}

int main() {
    test_reference_tables();
    test_result_fields();
    test_polarization();
    test_arguments();
    test_failures();
    test_rayleigh_anomalies();
    test_option_equivalence();
    test_uniform_gratings();
    test_sweep_and_threads();
    test_parallel_layers_and_custom_eig();
    test_bookkeeping();
    test_in_plane_limit();
    test_reentrancy();
    test_degenerate_modes();
    test_grazing_incidence();
    test_zero_thickness_layers();
    say("通过 %d / 共 %d\n", g_pass, g_total);
    return g_pass == g_total ? 0 : 1;
}
