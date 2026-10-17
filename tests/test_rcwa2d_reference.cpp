// tests/test_rcwa2d_reference.cpp —— 二维 RCWA（rcwa2d.hpp 的 solve2d）对冻结 Python 参照的验收
//
//   c++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -I include tests/test_rcwa2d_reference.cpp -o build/test_rcwa2d_reference
//
// 参照是 tests/rcwa2d_reference.inc（由 gui/validation/generate_field_reference_2d.py 从权威 Python 求解器 gui/2drcwa/rcwa2d.py
// 生成；源 NPZ 的 SHA-256 写在 .inc 的头部）。比较都是逐复数 |a − b| ≤ atol + rtol·|b|，不是只比模平方；
// 容差由冻结数据自带（N ≤ 289 取 1e-9）。参照里没有 W、V（简并子空间里本征基不唯一），只比较可观测量。
// 被验证的是 solve2d：装配 Fourier 算子（rcwa2d_operators.hpp）再求解（rcwa2d.hpp）。
//   · 24 个算例的远场：逐级的 Kx、Ky、kzI、kzII、Ex_r…Ez_t、Rs、Rp、Ts、Tp、DEr、DEt、prop；
//   · 有算子参照的结构层：layer_operators 的 Eps、Cx、Cy（逐元素，相对该矩阵最大模，1e-12）；
//   · Rayleigh 失败例：status = RayleighAnomaly，message 含区域与全部 (m, n)，数组为 NaN；
//   · 同一批算例再用不同选项算一遍并与同一份冻结值比：快路径关、并行层；
//   · LR 本征分解：要么与参照一致（≤ 1e-9），要么报告数值失败（本征残差检查），绝不静默给出错误的结果；
//     它在二维（2N 阶非对称矩阵、本征值成簇）上不可靠，实测 N=169 的方柱差到 1e-2，默认的 QR 不受影响；
//   · 近场（谐波列）、切面与相位不在这里，见 rcwa2d 近场的测试。
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "stablercwa/rcwa2d.hpp"
#include "stablercwa/rcwa2d_operators.hpp"

using namespace stablercwa;
namespace rcwa = stablercwa::rcwa;

#include "rcwa2d_reference.inc"

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
static bool g_quiet = false;     // 仅供参考的变体里不打印“失败”

static void check(bool ok, const char* file, int line, const char* text) {
    ++g_total;
    if (ok) ++g_pass;
    else if (!g_quiet) say("  失败 %s:%d  %s\n", file, line, text);
}
#define CHECK(cond) check(static_cast<bool>(cond), __FILE__, __LINE__, #cond)

// 逐复数比较：返回 max |a − b|，并把 |a − b| − (atol + rtol|b|) 的最大值记在 excess（≤ 0 即通过）。
struct Cmp {
    double maxdiff = 0.0;
    double excess = -1e300;
    long n = 0;
    void add(cplx a, const double b[2], double atol, double rtol) {
        const cplx ref{b[0], b[1]};
        const double d = std::abs(a - ref);
        ++n;
        if (!(d <= maxdiff)) maxdiff = std::isnan(d) ? std::numeric_limits<double>::infinity() : d;
        const double ex = std::isnan(d) ? 1e300 : d - (atol + rtol * std::abs(ref));
        if (ex > excess) excess = ex;
    }
    void add(double a, double b, double atol, double rtol) {
        const double bb[2] = {b, 0.0};
        add(cplx{a, 0.0}, bb, atol, rtol);
    }
    bool ok() const { return excess <= 0.0; }
};

struct Variant {
    const char* name;
    rcwa::Options2D options;
};

static rcwa::Result2D run_case(const Rcwa2dRefCase& c, const rcwa::Options2D& base) {
    rcwa::Options2D options = base;
    options.factorization = c.laurent ? rcwa::Factorization2D::Laurent : rcwa::Factorization2D::Li;
    const rcwa::Stack2D stack = rcwa2d_ref_stack(c.id);
    const rcwa::Jones2D jones = {cplx{c.jones[0], c.jones[1]}, cplx{c.jones[2], c.jones[3]}};
    return rcwa::solve2d(stack, c.wavelength, c.theta_deg, c.phi_deg, c.M1, c.M2, jones, options);
}

// 一个成功算例：与冻结的逐级数组比较。tol_scale 放宽 LR 之类精度较低的选项。
static double compare_far(const Rcwa2dRefCase& c, const rcwa::Result2D& r, double tol_scale, bool verbose) {
    Cmp cmp;
    const double atol = c.far_atol * tol_scale, rtol = c.far_rtol * tol_scale;
    CHECK(r.status == rcwa::Status2D::Ok);
    CHECK(r.N == c.count && static_cast<int>(r.m.size()) == c.count);
    if (r.status != rcwa::Status2D::Ok || r.N != c.count) return 1e300;
    for (int i = 0; i < c.count; ++i) {
        const Rcwa2dRefOrder& o = c.orders[i];
        const std::size_t k = static_cast<std::size_t>(i);
        CHECK(r.m[k] == o.m && r.n[k] == o.n);
        CHECK(rcwa::harmonic_index_2d(o.m, o.n, c.M1, c.M2) == i);
        cmp.add(r.Kx[k], o.Kx, atol, rtol);
        cmp.add(r.Ky[k], o.Ky, atol, rtol);
        cmp.add(r.kzI[k], o.kzI, atol, rtol);
        cmp.add(r.kzII[k], o.kzII, atol, rtol);
        cmp.add(r.Ex_r[k], o.Ex_r, atol, rtol); cmp.add(r.Ey_r[k], o.Ey_r, atol, rtol); cmp.add(r.Ez_r[k], o.Ez_r, atol, rtol);
        cmp.add(r.Ex_t[k], o.Ex_t, atol, rtol); cmp.add(r.Ey_t[k], o.Ey_t, atol, rtol); cmp.add(r.Ez_t[k], o.Ez_t, atol, rtol);
        cmp.add(r.Rs[k], o.Rs, atol, rtol); cmp.add(r.Rp[k], o.Rp, atol, rtol);
        cmp.add(r.Ts[k], o.Ts, atol, rtol); cmp.add(r.Tp[k], o.Tp, atol, rtol);
        cmp.add(r.DEr[k], o.DEr, atol, rtol);
        cmp.add(r.DEt[k], o.DEt, atol, rtol);
        CHECK((r.prop_r[k] != 0) == (o.prop_r != 0));
        CHECK((r.prop_t[k] != 0) == (o.prop_t != 0));
    }
    CHECK(cmp.ok());
    if (verbose || !cmp.ok())
        say("  %2d %-26s N=%3d  max |a-b| = %.2e  (%ld 个复数)%s\n", c.id, c.name, c.count, cmp.maxdiff, cmp.n, cmp.ok() ? "" : "  超出 1e-9 容差");
    return cmp.maxdiff;
}

// 失败例：状态、区域、(m, n) 列表。
static void check_failure(const Rcwa2dRefCase& c, const rcwa::Result2D& r) {
    CHECK(r.status == rcwa::Status2D::RayleighAnomaly);
    CHECK(r.message.find(c.failure_region) != std::string::npos);
    bool all = true;
    for (int i = 0; i < c.failure_count; ++i) {
        char label[48];
        std::snprintf(label, sizeof label, "(%d, %d)", c.failure_orders[i][0], c.failure_orders[i][1]);
        if (r.message.find(label) == std::string::npos) all = false;
    }
    CHECK(all);
    bool nan_filled = true;
    for (std::size_t i = 0; i < r.Rs.size(); ++i)
        if (!std::isnan(r.Rs[i].real()) || !std::isnan(r.DEr[i])) nan_filled = false;
    CHECK(nan_filled && !r.Rs.empty());
    say("  %2d %-26s 状态 = Rayleigh，区域“%s”，(m, n) 共 %d 个，message 全部含有%s\n", c.id, c.name, c.failure_region, c.failure_count,
        (r.status == rcwa::Status2D::RayleighAnomaly && all) ? "" : "  <-- 不符");
}

static void test_operators() {
    say("[Fourier 算子对冻结的 Eps、Cx、Cy]\n");
    double worst = 0.0;
    int nmat = 0;
    for (int i = 0; i < kRcwa2dRefCaseCount; ++i) {
        const Rcwa2dRefCase& c = kRcwa2dRefCases[i];
        if (c.status != 0 || c.operator_count == 0) continue;
        const rcwa::Stack2D stack = rcwa2d_ref_stack(c.id);
        for (int k = 0; k < c.operator_count; ++k) {
            const Rcwa2dRefOperators& o = c.operators[k];
            const rcwa::detail2d::LayerOperators2D ops = rcwa::detail2d::layer_operators(
                stack.layers[static_cast<std::size_t>(o.layer)], stack.Lx, stack.Ly, c.M1, c.M2,
                c.laurent ? rcwa::Factorization2D::Laurent : rcwa::Factorization2D::Li);
            CHECK(ops.eps.rows() == o.N && ops.cx.rows() == o.N && ops.cy.rows() == o.N);
            const Matrix* got[3] = {&ops.eps, &ops.cx, &ops.cy};
            const double* ref[3] = {o.eps, o.cx, o.cy};
            for (int t = 0; t < 3; ++t) {
                double peak = 0.0, diff = 0.0;
                for (int a = 0; a < o.N; ++a)
                    for (int b = 0; b < o.N; ++b) {
                        const cplx r{ref[t][(a * o.N + b) * 2], ref[t][(a * o.N + b) * 2 + 1]};
                        peak = std::max(peak, std::abs(r));
                        diff = std::max(diff, std::abs((*got[t])(a, b) - r));
                    }
                const double rel = diff / (peak > 0 ? peak : 1.0);
                worst = std::max(worst, rel);
                ++nmat;
                CHECK(rel <= 1e-12);
            }
        }
    }
    say("  %d 个矩阵，最大相对误差 %.2e（容差 1e-12）\n", nmat, worst);
}

int main() {
    say("二维 RCWA：solve2d 对冻结 Python 参照（%d 个算例）\n", kRcwa2dRefCaseCount);

    say("\n[远场，默认选项（QR 本征分解、快路径、串行）]\n");
    double worst_default = 0.0;
    for (int i = 0; i < kRcwa2dRefCaseCount; ++i) {
        const Rcwa2dRefCase& c = kRcwa2dRefCases[i];
        const rcwa::Result2D r = run_case(c, rcwa::Options2D{});
        if (c.status != 0) check_failure(c, r);
        else worst_default = std::max(worst_default, compare_far(c, r, 1.0, true));
    }
    say("  成功算例中的最大差 %.2e\n", worst_default);

    test_operators();

    const Variant variants[] = {
        {"homogeneous_fast_path = false（均匀层按稠密层处理）", [] { rcwa::Options2D o; o.homogeneous_fast_path = false; return o; }()},
        {"parallel_layers = true", [] { rcwa::Options2D o; o.parallel_layers = true; return o; }()},
        {"LR 本征分解（二维上不可靠：必须报告失败，不得静默出错）", [] { rcwa::Options2D o; o.eig = EigMethod::LR; return o; }()},
    };
    for (const Variant& v : variants) {
        say("\n[选项变体：%s]\n", v.name);
        const bool lr = v.options.eig == EigMethod::LR;
        double worst = 0.0;
        int count = 0, refused = 0, wrong_silently = 0;
        for (int i = 0; i < kRcwa2dRefCaseCount; ++i) {
            const Rcwa2dRefCase& c = kRcwa2dRefCases[i];
            const rcwa::Result2D r = run_case(c, v.options);
            if (c.status != 0) {
                CHECK(r.status == rcwa::Status2D::RayleighAnomaly);
                continue;
            }
            ++count;
            if (!lr) {
                worst = std::max(worst, compare_far(c, r, 1.0, false));
                continue;
            }
            // LR：要么与冻结参照一致（≤ 1e-9），要么报告数值失败——绝不静默给出错误的结果
            if (r.status == rcwa::Status2D::NumericalFailure) {
                ++refused;
                CHECK(r.message.find("LR") != std::string::npos);
                bool nan_filled = !r.Rs.empty() && std::isnan(r.Rs[0].real());
                CHECK(nan_filled);
                continue;
            }
            const int saved_pass = g_pass, saved_total = g_total;
            g_quiet = true;
            const double d = compare_far(c, r, 1.0, false);
            g_quiet = false;
            const bool good = (g_total - saved_total) == (g_pass - saved_pass);
            g_pass = saved_pass; g_total = saved_total;
            CHECK(good);
            if (!good) ++wrong_silently;
            worst = std::max(worst, d);
        }
        if (lr) say("  %d 个成功算例：%d 个报告数值失败（残差检查拒绝），其余与参照一致到 %.2e；静默出错 %d 个\n", count, refused, worst, wrong_silently);
        else say("  %d 个算例，最大差 %.2e\n", count, worst);
    }

    say("\n通过 %d / 共 %d\n", g_pass, g_total);
    return g_pass == g_total ? 0 : 1;
}
