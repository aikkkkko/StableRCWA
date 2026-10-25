// tests/test_rcwa_dense_hooks.cpp —— Options / Options2D 的 custom_multiply、custom_solve 钩子
//
//   c++ -std=c++17 -O2 -I include tests/test_rcwa_dense_hooks.cpp -o build/test_rcwa_dense_hooks
//
// 钩子换成库自己的运算（a * b、stablercwa::solve）时，结果必须与不设钩子逐位相同，钩子也确实被调用了：
//   1  一维 TE/TM：几种结构、快速路径开与关、diagnostics；sweep 在 4 个线程下
//   2  圆锥入射：一般方位角与 ky 接近零（退化层改用 Schur 基）
//   3  二维：tests/rcwa2d_reference.inc 的全部算例，串行与 parallel_layers（钩子在线程池里同时被调用）
//   4  近场：一维 TE/TM、圆锥、二维切面
//   5  约定：linear 不是 LUPartial 时不用 custom_solve；钩子抛 std::runtime_error 是数值失败；返回的大小不对时抛
//      std::invalid_argument
// 退出状态：全部通过 0，有失败 1。
#include <array>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "stablercwa/stablercwa.hpp"
#include "stablercwa/rcwa1d_fields.hpp"
#include "stablercwa/rcwa1d_conical_fields.hpp"
#include "stablercwa/rcwa2d_fields.hpp"

using namespace stablercwa;
namespace rcwa = stablercwa::rcwa;
#include "rcwa2d_reference.inc"

// 文字一律经 say() 输出（用法同 std::printf）：MSVC 检查 printf 的格式串时按系统代码页读，含中文的格式串会被误报。
#if defined(__GNUC__) && !defined(__MINGW32__)
[[gnu::format(printf, 1, 2)]]
#endif
static void say(const char* format, ...) {
    std::va_list args;
    va_start(args, format);
    std::vprintf(format, args);
    va_end(args);
}

// 结构：与 examples/structures.hpp 的 single_period、trapezoid、ridges_on_mirror 相同（测试不依赖 examples/）
namespace examples {
struct Mdg {
    double wavelength = 1.053, theta_deg = 66.36, period = 0.574, duty = 0.3, depth = 0.38;
    cplx n_tooth{1.46, 0.0};
    double n_air = 1.0, n_sub = 1.46, n_low = 1.46, n_high = 1.96;
    double d_high = 0.115306, d_2low = 0.30958, d_3low = 0.46438;
};
static void add_film_stack(std::vector<rcwa::Layer>& layers, const Mdg& c) {
    layers.push_back(rcwa::Layer::uniform(c.d_3low, c.n_low));
    layers.push_back(rcwa::Layer::uniform(c.d_high, c.n_high));
    for (int k = 0; k < 9; ++k) {
        layers.push_back(rcwa::Layer::uniform(c.d_2low, c.n_low));
        layers.push_back(rcwa::Layer::uniform(c.d_high, c.n_high));
    }
}
static rcwa::Stack single_period(const Mdg& c = {}) {
    rcwa::Stack stack;
    stack.nI = c.n_air;
    stack.nII = c.n_sub;
    stack.period = c.period;
    stack.layers.push_back(rcwa::Layer::grating(c.depth, {0.0, c.period * c.duty, c.period}, {c.n_tooth, cplx(c.n_air)}));
    add_film_stack(stack.layers, c);
    return stack;
}
static rcwa::Stack trapezoid(int slices, const Mdg& c = {}) {
    rcwa::Stack stack;
    stack.nI = c.n_air;
    stack.nII = c.n_sub;
    stack.period = c.period;
    for (int s = 0; s < slices; ++s) {
        const double f = 0.2 + (0.45 - 0.2) * (s + 0.5) / slices;
        const double w = c.period * f, x0 = 0.5 * (c.period - w);
        stack.layers.push_back(
            rcwa::Layer::grating(c.depth / slices, {0.0, x0, x0 + w, c.period}, {cplx(c.n_air), c.n_tooth, cplx(c.n_air)}));
    }
    add_film_stack(stack.layers, c);
    return stack;
}
struct Ridges {
    double wavelength = 0.94, theta_deg = 22.0;
};
static rcwa::Stack ridges_on_mirror() {
    rcwa::Stack stack;
    stack.nI = 1.0;
    stack.nII = 1.50;
    stack.period = 6.3;
    stack.layers.push_back(rcwa::Layer::grating(0.32, {0.0, 0.20, 1.05, 1.60, 2.05, 2.90, 4.25, 5.10, 5.45, 6.30},
                                                {1.0, 1.52, 1.0, 2.10, 1.0, 1.75, 1.0, 1.38, 1.0}));
    for (int k = 0; k < 20; ++k)
        stack.layers.push_back(k % 2 == 0 ? rcwa::Layer::uniform(0.105, 2.25) : rcwa::Layer::uniform(0.160, 1.48));
    return stack;
}
}  // namespace examples

static int g_total = 0, g_failed = 0;
static void check(bool ok, const char* text, int line) {
    ++g_total;
    if (!ok) {
        ++g_failed;
        say("  失败 第 %d 行：%s\n", line, text);
    }
}
#define CHECK(x) check(static_cast<bool>(x), #x, __LINE__)

// ------------------------------------------------------------------ 逐位比较
static std::size_t g_compared = 0;
static bool same(double a, double b) {
    ++g_compared;
    return std::memcmp(&a, &b, sizeof a) == 0;
}
static bool same(cplx a, cplx b) { return same(a.real(), b.real()) && same(a.imag(), b.imag()); }
static bool same(const rcwa::Jones& a, const rcwa::Jones& b) { return same(a.ss, b.ss) && same(a.sp, b.sp) && same(a.ps, b.ps) && same(a.pp, b.pp); }
static bool same(const rcwa::JonesEfficiency& a, const rcwa::JonesEfficiency& b) {
    return same(a.ss, b.ss) && same(a.sp, b.sp) && same(a.ps, b.ps) && same(a.pp, b.pp);
}
template <class T>
static bool same(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) return false;
    bool ok = true;
    for (std::size_t i = 0; i < a.size(); ++i) ok = same(a[i], b[i]) && ok;
    return ok;
}
static bool same(const Matrix& a, const Matrix& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    bool ok = true;
    for (int j = 0; j < a.cols(); ++j)
        for (int i = 0; i < a.rows(); ++i) ok = same(a(i, j), b(i, j)) && ok;
    return ok;
}
static bool same(const std::array<Matrix, 6>& a, const std::array<Matrix, 6>& b) {
    bool ok = true;
    for (int c = 0; c < 6; ++c) ok = same(a[c], b[c]) && ok;
    return ok;
}

static bool same(const rcwa::Result& a, const rcwa::Result& b) {
    return same(a.R, b.R) && same(a.T, b.T) && same(a.DEr, b.DEr) && same(a.DEt, b.DEt) &&
           same(a.max_solve_backward_error, b.max_solve_backward_error) && same(a.max_eig_residual, b.max_eig_residual) &&
           a.n_dense_solve == b.n_dense_solve && a.n_eig == b.n_eig && a.eig_info == b.eig_info;
}
static bool same(const rcwa::ConicalResult& a, const rcwa::ConicalResult& b) {
    return same(a.Jr, b.Jr) && same(a.Jt, b.Jt) && same(a.DEr, b.DEr) && same(a.DEt, b.DEt) &&
           same(a.max_solve_backward_error, b.max_solve_backward_error) && a.n_dense_solve == b.n_dense_solve &&
           a.n_schur == b.n_schur && a.eig_info == b.eig_info;
}
static bool same(const rcwa::Result2D& a, const rcwa::Result2D& b) {
    return a.status == b.status && a.message == b.message && same(a.Rs, b.Rs) && same(a.Rp, b.Rp) && same(a.Ts, b.Ts) &&
           same(a.Tp, b.Tp) && same(a.Ex_r, b.Ex_r) && same(a.Ey_r, b.Ey_r) && same(a.Ez_r, b.Ez_r) && same(a.Ex_t, b.Ex_t) &&
           same(a.Ey_t, b.Ey_t) && same(a.Ez_t, b.Ez_t) && same(a.DEr, b.DEr) && same(a.DEt, b.DEt) &&
           a.n_dense_solve == b.n_dense_solve && a.eig_info == b.eig_info;
}

// ------------------------------------------------------------------ 钩子：库自己的运算，外加计数
struct Calls {
    std::atomic<int> multiply{0}, solve{0};
};

// 一维、圆锥：内置的乘法是 rcwa::detail::multiply（与 a * b 算法相同，但余下那一列的加法次序不同，末位会差），
// 内置的列主元求解就是 stablercwa::solve(a, b)（右端是一列也按矩阵解）。
template <class O>
static O hooked_1d(O o, Calls& calls) {
    o.custom_multiply = [&calls](const Matrix& a, const Matrix& b) {
        ++calls.multiply;
        rcwa::detail::Work plain;                         // 不带钩子的 Work：走内置的内核
        return rcwa::detail::multiply(a, b, plain);
    };
    o.custom_solve = [&calls](const Matrix& a, const Matrix& b) {
        ++calls.solve;
        return stablercwa::solve(a, b);
    };
    return o;
}

// 二维：内置的乘法就是 a * b。右端是一个向量时，内置的做法是按向量解（solve_dense_vector），钩子收到的是一列的矩阵，
// 这里照内置的按向量解。
static rcwa::Options2D hooked_2d(rcwa::Options2D o, Calls& calls) {
    o.custom_multiply = [&calls](const Matrix& a, const Matrix& b) {
        ++calls.multiply;
        return a * b;
    };
    o.custom_solve = [&calls](const Matrix& a, const Matrix& b) {
        ++calls.solve;
        if (b.cols() != 1) return stablercwa::solve(a, b);
        const Vector x = stablercwa::solve(a, b.column(0));
        Matrix m(static_cast<int>(x.size()), 1);
        for (std::size_t i = 0; i < x.size(); ++i) m(static_cast<int>(i), 0) = x[i];
        return m;
    };
    return o;
}

// ------------------------------------------------------------------ 1：一维
static void test_planar() {
    say("[1] 一维 TE/TM\n");
    const examples::Mdg mdg;
    examples::Mdg lossy;
    lossy.n_tooth = cplx(1.46, 0.08);
    const examples::Ridges ridges;
    struct Case {
        const char* name;
        rcwa::Stack stack;
        double wavelength, theta;
        int M;
    };
    const Case cases[] = {
        {"单周期光栅 + 20 层膜", examples::single_period(), mdg.wavelength, mdg.theta_deg, 40},
        {"8 层梯形齿 + 20 层膜", examples::trapezoid(8), mdg.wavelength, mdg.theta_deg, 30},
        {"有吸收的齿", examples::single_period(lossy), mdg.wavelength, mdg.theta_deg, 30},
        {"四脊大周期 + 20 层膜", examples::ridges_on_mirror(), ridges.wavelength, ridges.theta_deg, 60},
    };
    int runs = 0;
    for (const Case& c : cases)
        for (rcwa::Polarization pol : {rcwa::Polarization::TE, rcwa::Polarization::TM})
            for (bool fast : {true, false}) {
                rcwa::Options base;
                base.homogeneous_fast_path = fast;
                base.diagnostics = true;
                Calls calls;
                const rcwa::Result a = rcwa::solve(c.stack, c.wavelength, c.theta, c.M, pol, base);
                const rcwa::Result b = rcwa::solve(c.stack, c.wavelength, c.theta, c.M, pol, hooked_1d(base, calls));
                CHECK(a.eig_info == 0 && same(a, b));
                CHECK(calls.multiply.load() > 0 && calls.solve.load() == b.n_dense_solve);
                ++runs;
            }
    // sweep：4 个线程同时调用钩子
    std::vector<rcwa::Point> points;
    for (int k = 0; k < 8; ++k) points.push_back({mdg.wavelength, 40.0 + 4.0 * k});
    Calls calls;
    const rcwa::Options base;
    const std::vector<rcwa::Result> serial = rcwa::sweep(examples::trapezoid(8), points, 25, rcwa::Polarization::TM, base);
    set_num_threads(4);
    const std::vector<rcwa::Result> swept = rcwa::sweep(examples::trapezoid(8), points, 25, rcwa::Polarization::TM, hooked_1d(base, calls));
    set_num_threads(1);
    bool all_same = serial.size() == swept.size();
    for (std::size_t k = 0; all_same && k < serial.size(); ++k) all_same = same(serial[k], swept[k]);
    CHECK(all_same && calls.solve.load() > 0);
    say("  %d 次求解与 8 点 sweep（4 个线程）：与不设钩子逐位相同\n", runs);
}

// ------------------------------------------------------------------ 2：圆锥入射
static void test_conical() {
    say("[2] 圆锥入射\n");
    const examples::Mdg mdg;
    int runs = 0;
    for (double phi : {25.0, 1e-4})
        for (bool fast : {true, false}) {
            rcwa::Options base;
            base.homogeneous_fast_path = fast;
            base.diagnostics = true;
            Calls calls;
            const rcwa::ConicalResult a = rcwa::solve_conical(examples::single_period(), mdg.wavelength, mdg.theta_deg, phi, 20, base);
            const rcwa::ConicalResult b =
                rcwa::solve_conical(examples::single_period(), mdg.wavelength, mdg.theta_deg, phi, 20, hooked_1d(base, calls));
            CHECK(a.eig_info == 0 && same(a, b));
            CHECK(calls.multiply.load() > 0 && calls.solve.load() == b.n_dense_solve);
            ++runs;
        }
    say("  %d 次求解（含 ky 接近零的 Schur 路径）：与不设钩子逐位相同\n", runs);
}

// ------------------------------------------------------------------ 3：二维
static rcwa::Result2D run_2d(const Rcwa2dRefCase& c, const rcwa::Options2D& o) {
    rcwa::Options2D options = o;
    options.factorization = c.laurent ? rcwa::Factorization2D::Laurent : rcwa::Factorization2D::Li;
    const rcwa::Jones2D jones = {cplx{c.jones[0], c.jones[1]}, cplx{c.jones[2], c.jones[3]}};
    return rcwa::solve2d(rcwa2d_ref_stack(c.id), c.wavelength, c.theta_deg, c.phi_deg, c.M1, c.M2, jones, options);
}

static void test_2d() {
    say("[3] 二维：%d 个参照算例\n", kRcwa2dRefCaseCount);
    int runs = 0, rayleigh = 0, solves = 0, multiplies = 0;
    for (int i = 0; i < kRcwa2dRefCaseCount; ++i) {
        const Rcwa2dRefCase& c = kRcwa2dRefCases[i];
        for (bool parallel : {false, true}) {
            rcwa::Options2D base;
            base.parallel_layers = parallel;
            Calls calls;
            if (parallel) set_num_threads(4);
            const rcwa::Result2D a = run_2d(c, base);
            const rcwa::Result2D b = run_2d(c, hooked_2d(base, calls));
            if (parallel) set_num_threads(1);
            CHECK(same(a, b));
            if (a.status == rcwa::Status2D::Ok) {
                CHECK(calls.solve.load() == b.n_dense_solve);  // 只有均匀层的算例（界面、单层膜）一次也不解
                solves += calls.solve.load();
                multiplies += calls.multiply.load();
            } else {
                ++rayleigh;
            }
            ++runs;
        }
    }
    CHECK(solves > 0 && multiplies > 0);
    say("  %d 次求解（串行与 parallel_layers；其中 %d 次是照样报告的失败例）：与不设钩子逐位相同；钩子共解 %d 次、乘 %d 次\n", runs,
        rayleigh, solves, multiplies);
}

// ------------------------------------------------------------------ 4：近场
static void test_fields() {
    say("[4] 近场\n");
    const examples::Mdg mdg;
    const rcwa::Stack stack = examples::single_period();
    const RealVector zs{-0.3, 0.0, 0.19, 0.38, 0.9, 3.0};
    for (rcwa::Polarization pol : {rcwa::Polarization::TE, rcwa::Polarization::TM}) {
        Calls calls;
        const auto a = rcwa::solve_with_fields(stack, mdg.wavelength, mdg.theta_deg, 30, pol);
        const auto b = rcwa::solve_with_fields(stack, mdg.wavelength, mdg.theta_deg, 30, pol, hooked_1d(rcwa::Options{}, calls));
        CHECK(same(rcwa::field_map(a, zs, 48).components, rcwa::field_map(b, zs, 48).components));
        CHECK(calls.solve.load() > 0);
    }
    {
        Calls calls;
        const auto a = rcwa::solve_conical_with_fields(stack, mdg.wavelength, mdg.theta_deg, 25.0, 20);
        const auto b = rcwa::solve_conical_with_fields(stack, mdg.wavelength, mdg.theta_deg, 25.0, 20, hooked_1d(rcwa::Options{}, calls));
        const rcwa::JonesVector jones{{cplx(0.6, 0.0), cplx(0.0, 0.8)}};
        CHECK(same(rcwa::field_map(a, zs, 48, 0.0, jones).components, rcwa::field_map(b, zs, 48, 0.0, jones).components));
        CHECK(calls.solve.load() > 0);
    }
    int slices = 0;
    for (int i = 0; i < kRcwa2dRefCaseCount && slices < 4; ++i) {
        const Rcwa2dRefCase& c = kRcwa2dRefCases[i];
        if (c.status != 0 || c.count < 9) continue;
        const rcwa::Stack2D stack2 = rcwa2d_ref_stack(c.id);
        bool patterned = false;
        double depth = 0.0;
        for (const auto& layer : stack2.layers) {
            patterned = patterned || !layer.homogeneous;
            depth += layer.d;
        }
        if (!patterned) continue;
        rcwa::Options2D base;
        base.factorization = c.laurent ? rcwa::Factorization2D::Laurent : rcwa::Factorization2D::Li;
        Calls calls;
        const rcwa::Jones2D jones = {cplx{c.jones[0], c.jones[1]}, cplx{c.jones[2], c.jones[3]}};
        const auto a = rcwa::solve2d_with_fields(stack2, c.wavelength, c.theta_deg, c.phi_deg, c.M1, c.M2, jones, base);
        const auto b = rcwa::solve2d_with_fields(stack2, c.wavelength, c.theta_deg, c.phi_deg, c.M1, c.M2, jones, hooked_2d(base, calls));
        const RealVector z2{-0.2, 0.0, 0.5 * depth, depth, depth + 0.3};
        for (rcwa::SlicePlane plane : {rcwa::SlicePlane::XZ, rcwa::SlicePlane::YZ})
            CHECK(same(rcwa::field_slice(a.fields, z2, plane, 32, 0.1).components, rcwa::field_slice(b.fields, z2, plane, 32, 0.1).components));
        CHECK(calls.solve.load() > 0);
        slices += 2;
    }
    CHECK(slices == 4);
    say("  一维 TE/TM、圆锥、二维 %d 个切面：与不设钩子逐位相同\n", slices);
}

// ------------------------------------------------------------------ 5：约定
static void test_rules() {
    say("[5] 约定\n");
    const examples::Mdg mdg;
    const rcwa::Stack stack = examples::single_period();
    // linear 不是 LUPartial：不用 custom_solve（乘法照样用 custom_multiply），结果与不设钩子逐位相同
    for (rcwa::LinearSolver method : {rcwa::LinearSolver::LUNoPivot, rcwa::LinearSolver::QRHouseholder}) {
        rcwa::Options base;
        base.linear = method;
        Calls calls;
        const rcwa::Result a = rcwa::solve(stack, mdg.wavelength, mdg.theta_deg, 30, rcwa::Polarization::TM, base);
        const rcwa::Result b = rcwa::solve(stack, mdg.wavelength, mdg.theta_deg, 30, rcwa::Polarization::TM, hooked_1d(base, calls));
        CHECK(same(a, b) && calls.solve.load() == 0 && calls.multiply.load() > 0);
    }
    // 只设一个钩子也可以
    {
        rcwa::Options only;
        only.custom_solve = [](const Matrix& a, const Matrix& b) { return stablercwa::solve(a, b); };
        CHECK(same(rcwa::solve(stack, mdg.wavelength, mdg.theta_deg, 30, rcwa::Polarization::TE),
                   rcwa::solve(stack, mdg.wavelength, mdg.theta_deg, 30, rcwa::Polarization::TE, only)));
    }
    // 钩子抛 std::runtime_error：数值失败（效率是 NaN、二维的 status 是 NumericalFailure），不是异常
    {
        rcwa::Options bad;
        bad.custom_solve = [](const Matrix&, const Matrix&) -> Matrix { throw std::runtime_error("singular"); };
        const rcwa::Result r = rcwa::solve(stack, mdg.wavelength, mdg.theta_deg, 30, rcwa::Polarization::TE, bad);
        CHECK(!r.DEr.empty() && std::isnan(r.DEr[0]) && std::isnan(r.reflectance()));
        const rcwa::ConicalResult rc = rcwa::solve_conical(stack, mdg.wavelength, mdg.theta_deg, 25.0, 20, bad);
        CHECK(!rc.DEr.empty() && std::isnan(rc.DEr[0].ss));
        const Rcwa2dRefCase& c = kRcwa2dRefCases[8];                 // pillar_sq
        rcwa::Options2D bad2;
        bad2.custom_solve = bad.custom_solve;
        CHECK(run_2d(c, bad2).status == rcwa::Status2D::NumericalFailure);
    }
    // 返回的大小不对：std::invalid_argument
    {
        const auto wrong = [](const Matrix&, const Matrix&) { return Matrix(1, 1); };
        bool thrown = false;
        try {
            rcwa::Options o;
            o.custom_multiply = wrong;
            (void)rcwa::solve(stack, mdg.wavelength, mdg.theta_deg, 30, rcwa::Polarization::TE, o);
        } catch (const std::invalid_argument&) {
            thrown = true;
        }
        CHECK(thrown);
        thrown = false;
        try {
            rcwa::Options o;
            o.custom_solve = wrong;
            (void)rcwa::solve_conical(stack, mdg.wavelength, mdg.theta_deg, 25.0, 20, o);
        } catch (const std::invalid_argument&) {
            thrown = true;
        }
        CHECK(thrown);
        thrown = false;
        try {
            rcwa::Options2D o;
            o.custom_multiply = wrong;
            (void)run_2d(kRcwa2dRefCases[8], o);
        } catch (const std::invalid_argument&) {
            thrown = true;
        }
        CHECK(thrown);
    }
    say("  LUNoPivot、QRHouseholder 不用 custom_solve；钩子抛 runtime_error 是数值失败；大小不对时抛 invalid_argument\n");
}

int main() {
    test_planar();
    test_conical();
    test_2d();
    test_fields();
    test_rules();
    say("逐位比较了 %llu 个数\n", static_cast<unsigned long long>(g_compared));
    say("%d 项检查，%d 项失败\n", g_total, g_failed);
    return g_failed == 0 ? 0 : 1;
}
