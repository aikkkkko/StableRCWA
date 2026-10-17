// tests/test_rcwa2d.cpp —— rcwa2d.hpp（二维交叉光栅 RCWA 的求解核心）的测试
//
//   c++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -I include tests/test_rcwa2d.cpp -o build/test_rcwa2d
//   c++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -I include tests/test_rcwa2d.cpp -o build/test_rcwa2d_san
//   build/test_rcwa2d                        默认：全部正确性测试加一个很小的计时算例（总共几秒）
//   build/test_rcwa2d perf M films above capture threads
//                                            计时：M = M1 = M2（N = (2M+1)²），films = 膜层数，above = 1 表示膜堆在结构层之上
//                                            （默认 0：膜堆在结构层之下），capture = 1 时同时保留近场快照，threads = 线程数（默认 1）
//
// 这里测的是求解核心 solve2d_with_operators（吃现成的 Eps、Cx、Cy）；算子装配有自己的测试。参考值有两类：
//   · tests/rcwa2d_core_fixture.inc：gui/2drcwa/rcwa2d.py（权威参照）对 17 个小算例（N ≤ 49）给出的算子、远场输出与若干 z 处的
//     六个场分量（field_at），远场逐复数比较 |a − b| ≤ 1e-9 + 1e-9·|b|（合同第 6 节），最大差在输出里打印；
//   · 不依赖 Python 的独立检验：均匀多层对 Airy 递推式；y 不变 / x 不变的结构（算子在这里按 Kronecker 形式自己组装）
//     对一维 C++ 的 rcwa::solve 与 rcwa::solve_conical；能量守恒；2×2 超胞等价；x↔y 互换；
//     快路径与稠密路径、parallel_layers、线程数、custom_eig、capture 与否各自互相一致；capture 里的界面连续性。
// 此外覆盖参数检查（invalid_argument）、Rayleigh 异常与数值失败的状态。
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

#include "stablercwa/rcwa1d.hpp"
#include "stablercwa/rcwa1d_conical.hpp"
#include "stablercwa/rcwa2d.hpp"

using namespace stablercwa;
namespace rcwa = stablercwa::rcwa;
namespace d2 = rcwa::detail2d;
using d2::LayerOperators2D;

#include "rcwa2d_core_fixture.inc"

// ---------------------------------------------------------------------------------------------
// 输出与计数。文字一律经 say() 输出（用法同 std::printf），缘由见 tests/test_rcwa1d.cpp 开头的说明。
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

static const double kNaN = std::numeric_limits<double>::quiet_NaN();

// ---------------------------------------------------------------------------------------------
// 夹具 → 求解输入

static Matrix matrix_from_rows(const double* ri, int n) {
    Matrix m(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            const std::size_t k = 2 * (static_cast<std::size_t>(i) * n + j);
            m(i, j) = cplx{ri[k], ri[k + 1]};
        }
    return m;
}

struct Problem {
    rcwa::Stack2D stack;
    std::vector<LayerOperators2D> ops;
    double wavelength = 1, theta = 0, phi = 0;
    int M1 = 0, M2 = 0;
    rcwa::Jones2D jones{1.0, 0.0};
};

static Problem problem_from_case(const CoreRefCase& c) {
    Problem p;
    p.stack.nI = c.nI;
    p.stack.nII = cplx{c.nII_re, c.nII_im};
    p.stack.Lx = c.Lx;
    p.stack.Ly = c.Ly;
    p.wavelength = c.wavelength;
    p.theta = c.theta_deg;
    p.phi = c.phi_deg;
    p.M1 = c.M1;
    p.M2 = c.M2;
    p.jones = {cplx{c.jones[0], c.jones[1]}, cplx{c.jones[2], c.jones[3]}};
    for (int j = 0; j < c.layer_count; ++j) {
        const CoreRefLayer& L = c.layers[j];
        LayerOperators2D op;
        if (L.homogeneous) {
            p.stack.layers.push_back(rcwa::Layer2D::uniform(L.d, cplx{L.n_re, L.n_im}, L.name));
        } else {
            std::vector<double> ye(L.y_edges, L.y_edges + L.K + 1);
            std::vector<rcwa::XProfile2D> rows;
            const double* xe = L.x_edges;
            const double* nv = L.n_values;
            for (int k = 0; k < L.K; ++k) {
                rcwa::XProfile2D row;
                row.edges.assign(xe, xe + L.x_count[k] + 1);
                for (int s = 0; s < L.x_count[k]; ++s) row.n_values.push_back(cplx{nv[2 * s], nv[2 * s + 1]});
                xe += L.x_count[k] + 1;
                nv += 2 * L.x_count[k];
                rows.push_back(std::move(row));
            }
            p.stack.layers.push_back(rcwa::Layer2D::patterned(L.d, std::move(ye), std::move(rows), L.name));
            op.eps = matrix_from_rows(L.eps, c.N);
            op.cx = matrix_from_rows(L.cx, c.N);
            op.cy = matrix_from_rows(L.cy, c.N);
        }
        p.ops.push_back(std::move(op));
    }
    return p;
}

static rcwa::Result2D run(const Problem& p, const rcwa::Options2D& o = {}, d2::FieldCapture2D* cap = nullptr) {
    return rcwa::solve2d_with_operators(p.stack, p.wavelength, p.theta, p.phi, p.M1, p.M2, p.jones, p.ops, o, cap);
}
// 同一个求解，稠密界面改用 S 块（rcwa2d.py 的 s_interface2/_rud2/_step2 逐段对应的路径），只用来交叉检验
static rcwa::Result2D run_s_blocks(const Problem& p, const rcwa::Options2D& o = {}, d2::FieldCapture2D* cap = nullptr) {
    return d2::solve_impl(p.stack, p.wavelength, p.theta, p.phi, p.M1, p.M2, p.jones, p.ops, o, cap, true);
}

// ---------------------------------------------------------------------------------------------
// 比较工具

// 复数组与参照（实部、虚部交替）逐元素比较：|a − b| ≤ 1e-9 + 1e-9·|b|；最大绝对差累计在 worst 里
struct Cmp {
    double worst = 0.0;     // 最大 |a − b|
    double worst_rel = 0.0; // 最大 |a − b| / (1e-9 + 1e-9|b|)：≤ 1 即满足合同的容差
};
static void cmp_complex(const Vector& got, const double* ref, std::size_t n, Cmp& c) {
    for (std::size_t i = 0; i < n; ++i) {
        const cplx b{ref[2 * i], ref[2 * i + 1]};
        const double e = std::abs(got[i] - b);
        const double r = e / (1e-9 + 1e-9 * std::abs(b));
        if (!(e <= c.worst)) c.worst = e;                  // NaN 也会进来
        if (!(r <= c.worst_rel)) c.worst_rel = r;
    }
}
static void cmp_real(const RealVector& got, const double* ref, std::size_t n, Cmp& c) {
    for (std::size_t i = 0; i < n; ++i) {
        const double e = std::abs(got[i] - ref[i]);
        const double r = e / (1e-9 + 1e-9 * std::abs(ref[i]));
        if (!(e <= c.worst)) c.worst = e;
        if (!(r <= c.worst_rel)) c.worst_rel = r;
    }
}

static Cmp compare_with_case(const rcwa::Result2D& r, const CoreRefCase& c) {
    Cmp m;
    const std::size_t N = static_cast<std::size_t>(c.N);
    cmp_real(r.Kx, c.Kx, N, m);
    cmp_real(r.Ky, c.Ky, N, m);
    cmp_complex(r.kzI, c.kzI, N, m);
    cmp_complex(r.kzII, c.kzII, N, m);
    cmp_complex(r.Ex_r, c.Ex_r, N, m);
    cmp_complex(r.Ey_r, c.Ey_r, N, m);
    cmp_complex(r.Ez_r, c.Ez_r, N, m);
    cmp_complex(r.Ex_t, c.Ex_t, N, m);
    cmp_complex(r.Ey_t, c.Ey_t, N, m);
    cmp_complex(r.Ez_t, c.Ez_t, N, m);
    cmp_complex(r.Rs, c.Rs, N, m);
    cmp_complex(r.Rp, c.Rp, N, m);
    cmp_complex(r.Ts, c.Ts, N, m);
    cmp_complex(r.Tp, c.Tp, N, m);
    cmp_real(r.DEr, c.DEr, N, m);
    cmp_real(r.DEt, c.DEt, N, m);
    return m;
}

// 两个结果逐位相同（振幅、效率、kz、状态）
static bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; }
static bool same_bits(const cplx& a, const cplx& b) { return same_bits(a.real(), b.real()) && same_bits(a.imag(), b.imag()); }
template <class V>
static bool same_vec(const V& a, const V& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (!same_bits(a[i], b[i])) return false;
    return true;
}
static bool same_result(const rcwa::Result2D& a, const rcwa::Result2D& b) {
    return a.status == b.status && same_vec(a.Ex_r, b.Ex_r) && same_vec(a.Ey_r, b.Ey_r) && same_vec(a.Ez_r, b.Ez_r) &&
           same_vec(a.Ex_t, b.Ex_t) && same_vec(a.Ey_t, b.Ey_t) && same_vec(a.Ez_t, b.Ez_t) && same_vec(a.Rs, b.Rs) &&
           same_vec(a.Rp, b.Rp) && same_vec(a.Ts, b.Ts) && same_vec(a.Tp, b.Tp) && same_vec(a.DEr, b.DEr) &&
           same_vec(a.DEt, b.DEt) && same_vec(a.kzI, b.kzI) && same_vec(a.kzII, b.kzII);
}

// 两个结果的最大差（振幅与效率）
static double result_distance(const rcwa::Result2D& a, const rcwa::Result2D& b) {
    double d = 0.0;
    for (const auto& pr : {std::make_pair(&a.Rs, &b.Rs), std::make_pair(&a.Rp, &b.Rp), std::make_pair(&a.Ts, &b.Ts),
                           std::make_pair(&a.Tp, &b.Tp), std::make_pair(&a.Ex_r, &b.Ex_r), std::make_pair(&a.Ey_t, &b.Ey_t)})
        for (std::size_t i = 0; i < pr.first->size(); ++i) {
            const double e = std::abs((*pr.first)[i] - (*pr.second)[i]);
            if (!(e <= d)) d = e;
        }
    for (std::size_t i = 0; i < a.DEr.size(); ++i) {
        d = std::max(d, std::abs(a.DEr[i] - b.DEr[i]));
        d = std::max(d, std::abs(a.DEt[i] - b.DEt[i]));
    }
    return d;
}

static bool all_nan(const Vector& v) {
    for (const cplx& z : v)
        if (!std::isnan(z.real()) || !std::isnan(z.imag())) return false;
    return !v.empty();
}
static bool all_nan(const RealVector& v) {
    for (double x : v)
        if (!std::isnan(x)) return false;
    return !v.empty();
}
static bool result_is_nan(const rcwa::Result2D& r) {
    return all_nan(r.Ex_r) && all_nan(r.Ey_r) && all_nan(r.Ez_r) && all_nan(r.Ex_t) && all_nan(r.Ey_t) && all_nan(r.Ez_t) &&
           all_nan(r.Rs) && all_nan(r.Rp) && all_nan(r.Ts) && all_nan(r.Tp) && all_nan(r.DEr) && all_nan(r.DEt);
}

static double total_energy(const rcwa::Result2D& r) { return r.reflectance() + r.transmittance(); }

// ---------------------------------------------------------------------------------------------
// 1. 夹具：Python（rcwa2d.py）逐复数对拍

static void test_fixture() {
    say("[1] 与 rcwa2d.py 逐复数对拍（%d 个算例；容差 1e-9 + 1e-9|b|）\n", kCoreRefCaseCount);
    double worst_all = 0.0, worst_fast_off = 0.0, worst_sblocks = 0.0;
    for (int ci = 0; ci < kCoreRefCaseCount; ++ci) {
        const CoreRefCase& c = kCoreRefCases[ci];
        const Problem p = problem_from_case(c);
        if (c.status == 2) {
            const rcwa::Result2D r = run(p);
            CHECK(r.status == rcwa::Status2D::RayleighAnomaly);
            CHECK(r.message.find(c.rayleigh_where) != std::string::npos);
            CHECK(r.message.find(c.rayleigh_mn) != std::string::npos);
            CHECK(result_is_nan(r));
            CHECK(r.N == c.N && static_cast<int>(r.Kx.size()) == c.N);
            say("  算例 %2d %-20s Rayleigh：%s %s\n", c.id, c.name, c.rayleigh_where, c.rayleigh_mn);
            // 快路径关闭时检查同样适用
            rcwa::Options2D o;
            o.homogeneous_fast_path = false;
            CHECK(run(p, o).status == rcwa::Status2D::RayleighAnomaly);
            CHECK(run_s_blocks(p).status == rcwa::Status2D::RayleighAnomaly);
            continue;
        }
        const rcwa::Result2D r = run(p);
        CHECK(r.status == rcwa::Status2D::Ok);
        const Cmp m = compare_with_case(r, c);
        worst_all = std::max(worst_all, m.worst);
        say("  算例 %2d %-20s N=%2d  最大 |a−b| = %.2e  容差用掉 %.2e\n", c.id, c.name, c.N, m.worst, m.worst_rel);
        CHECK(m.worst_rel <= 1.0);
        CHECK_LE(m.worst, 1e-10);   // 合同：> 1e-10 要查原因
        // 同一算例用稠密路径（均匀层按结构层处理）；N ≤ 49，开销很小
        rcwa::Options2D o;
        o.homogeneous_fast_path = false;
        const rcwa::Result2D r2 = run(p, o);
        const Cmp m2 = compare_with_case(r2, c);
        worst_fast_off = std::max(worst_fast_off, m2.worst);
        CHECK(r2.status == rcwa::Status2D::Ok);
        CHECK(m2.worst_rel <= 1.0);
        // 同一算例，稠密界面用 S 块（与 Python 的 _rud2/_step2 逐段对应），快路径开与关
        const Cmp m3 = compare_with_case(run_s_blocks(p), c);
        const Cmp m4 = compare_with_case(run_s_blocks(p, o), c);
        worst_sblocks = std::max({worst_sblocks, m3.worst, m4.worst});
        CHECK(m3.worst_rel <= 1.0 && m4.worst_rel <= 1.0);
    }
    say("  汇总：最大 |a−b|：快路径 %.2e，稠密路径（均匀层也做本征分解）%.2e，S 块路径 %.2e\n", worst_all, worst_fast_off, worst_sblocks);
}

// ---------------------------------------------------------------------------------------------
// 2. 均匀多层对 Airy 递推式（不依赖 rcwa2d.py，也不依赖一维求解器）

struct AiryRef {
    cplx rs, rp, ts, tp;
};
static cplx kz_ref(cplx n, double s) {
    cplx kz = std::sqrt(n * n - s * s);
    if (kz.imag() < 0.0) kz = -kz;
    return kz;
}
// 各层 (厚度, 折射率)；入射介质 nI（实），出射介质 nII。振幅取各界面处的参考（入射波在 z=0，透射波在最底面），
// s 与 p 都是电场振幅（p̂ × ŝ = k̂，反射 p 在法向入射时与 s 反号）。
static AiryRef airy_ref(double nI, const std::vector<std::pair<double, cplx>>& layers, cplx nII, double lambda, double theta_deg) {
    const double s = nI * std::sin(theta_deg * (kPi / 180.0)), k0 = 2.0 * kPi / lambda;
    std::vector<cplx> n{cplx{nI, 0.0}};
    std::vector<double> d{0.0};
    for (const auto& l : layers) {
        n.push_back(l.second);
        d.push_back(l.first);
    }
    n.push_back(nII);
    d.push_back(0.0);
    const std::size_t L = layers.size();
    std::vector<cplx> kz(n.size());
    for (std::size_t j = 0; j < n.size(); ++j) kz[j] = kz_ref(n[j], s);
    AiryRef out;
    for (int pol = 0; pol < 2; ++pol) {
        std::vector<cplx> r(L + 1), t(L + 1);
        for (std::size_t j = 0; j <= L; ++j) {
            if (pol == 0) {
                r[j] = (kz[j] - kz[j + 1]) / (kz[j] + kz[j + 1]);
                t[j] = 2.0 * kz[j] / (kz[j] + kz[j + 1]);
            } else {
                const cplx den = n[j + 1] * n[j + 1] * kz[j] + n[j] * n[j] * kz[j + 1];
                r[j] = (n[j + 1] * n[j + 1] * kz[j] - n[j] * n[j] * kz[j + 1]) / den;
                t[j] = 2.0 * n[j] * n[j + 1] * kz[j] / den;
            }
        }
        cplx gamma = r[L], tt = t[L], den_all = 1.0;
        for (std::size_t j = L; j-- > 0;) {
            const cplx ph = std::exp(cplx{0.0, 2.0 * k0 * d[j + 1]} * kz[j + 1]);
            const cplx den = 1.0 + r[j] * gamma * ph;
            gamma = (r[j] + gamma * ph) / den;
            den_all *= den;
            tt *= t[j] * std::exp(cplx{0.0, k0 * d[j + 1]} * kz[j + 1]);
        }
        (pol == 0 ? out.rs : out.rp) = gamma;
        (pol == 0 ? out.ts : out.tp) = tt / den_all;
    }
    return out;
}

static void test_airy() {
    say("[2] 均匀多层对 Airy 递推式\n");
    struct Config {
        double nI;
        std::vector<std::pair<double, cplx>> layers;
        cplx nII;
        double lambda, theta, phi;
    };
    const std::vector<Config> configs = {
        {1.0, {{0.23, 2.1}}, 1.46, 0.6, 42.0, -25.0},                                                       // 单层 Airy
        {1.0, {{0.35, cplx{1.65, 0.12}}}, 1.46, 0.6, 42.0, 70.0},                                           // 吸收单层
        {1.3, {{0.1, 2.3}, {0.17, 1.38}, {0.31, cplx{3.5, 0.05}}, {0.2, 1.9}}, cplx{1.5, 0.02}, 0.8, 27.0, 123.0},   // 多层，吸收出射介质
        {1.0, {{0.4, 1.7}, {0.0, 2.2}, {0.25, 1.2}}, 1.0, 0.55, 0.0, 33.0},                                // 正入射，含零厚层
        {1.5, {{2.0, 1.0}}, 1.5, 0.6, 55.0, 15.0},                                                           // 全内反射区：空气隙中是倏逝波
    };
    double worst = 0.0;
    int ci = 0;
    for (const Config& c : configs) {
        ++ci;
        for (int fast = 1; fast >= 0; --fast) {
            for (int pol = 0; pol < 2; ++pol) {
                rcwa::Stack2D st;
                st.nI = c.nI;
                st.nII = c.nII;
                st.Lx = 0.9;
                st.Ly = 0.7;
                for (const auto& l : c.layers) st.layers.push_back(rcwa::Layer2D::uniform(l.first, l.second));
                std::vector<LayerOperators2D> ops(st.layers.size());
                rcwa::Options2D o;
                o.homogeneous_fast_path = fast != 0;
                const rcwa::Jones2D jones = pol == 0 ? rcwa::Jones2D{cplx{1.0, 0.0}, cplx{0.0, 0.0}}
                                                      : rcwa::Jones2D{cplx{0.0, 0.0}, cplx{0.8, -0.6}};
                const rcwa::Result2D r = rcwa::solve2d_with_operators(st, c.lambda, c.theta, c.phi, 1, 1, jones, ops, o);
                CHECK(r.status == rcwa::Status2D::Ok);
                const AiryRef a = airy_ref(c.nI, c.layers, c.nII, c.lambda, c.theta);
                const std::size_t i0 = static_cast<std::size_t>(r.i0);
                const cplx Es = jones[0], Ep = jones[1];
                double e = 0.0;
                if (pol == 0) {
                    e = std::max(std::abs(r.Rs[i0] - Es * a.rs), std::abs(r.Ts[i0] - Es * a.ts));
                    e = std::max(e, std::max(std::abs(r.Rp[i0]), std::abs(r.Tp[i0])));          // s 入射不产生 p
                } else {
                    e = std::max(std::abs(r.Rp[i0] - Ep * a.rp), std::abs(r.Tp[i0] - Ep * a.tp));
                    e = std::max(e, std::max(std::abs(r.Rs[i0]), std::abs(r.Ts[i0])));
                }
                // 均匀结构不产生衍射：非零级振幅为零
                for (std::size_t i = 0; i < r.Rs.size(); ++i) {
                    if (i == i0) continue;
                    e = std::max(e, std::max(std::abs(r.Rs[i]), std::abs(r.Tp[i])));
                    e = std::max(e, std::abs(r.DEr[i]) + std::abs(r.DEt[i]));
                }
                // 反射效率 = |入射矢量|²·|r|²（入射介质无吸收，反射级 kz = nI·cosθ）
                const double R = (pol == 0 ? std::norm(Es) : std::norm(Ep)) * std::norm(pol == 0 ? a.rs : a.rp);
                e = std::max(e, std::abs(r.DEr[i0] - R));
                worst = std::max(worst, e);
                CHECK_LE(e, 1e-11);
                if (std::isfinite(c.nII.imag()) && c.nII.imag() == 0.0 && std::all_of(c.layers.begin(), c.layers.end(),
                        [](const std::pair<double, cplx>& l) { return l.second.imag() == 0.0; }))
                    CHECK_LE(std::abs(total_energy(r) - 1.0), 1e-12);   // 无吸收：R + T = 1（入射 Jones 矢量已归一）
            }
        }
        say("  配置 %d：对 Airy 的最大差 %.2e（累计）\n", ci, worst);
    }
}

// ---------------------------------------------------------------------------------------------
// 3. 一维极限：y 不变 / x 不变的结构，算子按 Kronecker 形式在这里自己组装

// 分段常数剖面 (edges, values) 的 Fourier 系数 c_h，h = −hmax…hmax（下标 h + hmax），周期 period
static Vector fourier_coeffs(const std::vector<double>& edges, const Vector& values, double period, int hmax) {
    Vector c(static_cast<std::size_t>(2 * hmax + 1));
    for (int h = -hmax; h <= hmax; ++h) {
        cplx s{};
        for (std::size_t k = 0; k < values.size(); ++k) {
            if (h == 0) {
                s += values[k] * (edges[k + 1] - edges[k]) / period;
            } else {
                const double w = -2.0 * kPi * h / period;          // exp(−i·2π·h·x/Λ) = exp(i·w·x)
                const cplx a = std::exp(cplx{0.0, w * edges[k + 1]}), b = std::exp(cplx{0.0, w * edges[k]});
                s += values[k] * (a - b) / cplx{0.0, -2.0 * kPi * h};
            }
        }
        c[static_cast<std::size_t>(h + hmax)] = s;
    }
    return c;
}
static Matrix toeplitz_from(const Vector& c, int M) {
    Matrix t(2 * M + 1, 2 * M + 1);
    for (int i = 0; i < 2 * M + 1; ++i)
        for (int j = 0; j < 2 * M + 1; ++j) t(i, j) = c[static_cast<std::size_t>(i - j + 2 * M)];
    return t;
}
// 一维剖面沿 x（栅线沿 y）或沿 y（栅线沿 x）的 Li 算子，整个 N×N 是 Kronecker 形式：
//   沿 x：Eps = Cy = δ_nn'·⟦ε⟧[m,m']，Cx = δ_nn'·⟦1/ε⟧⁻¹[m,m']；沿 y：把 m、n 的角色互换（Eps = Cx，Cy 用逆规则）
static LayerOperators2D kron_ops(const std::vector<double>& edges, const Vector& n_values, double period, int M1, int M2, bool along_x) {
    Vector eps(n_values.size()), inv(n_values.size());
    for (std::size_t k = 0; k < n_values.size(); ++k) {
        eps[k] = n_values[k] * n_values[k];
        inv[k] = 1.0 / eps[k];
    }
    const int M = along_x ? M1 : M2;
    const Matrix te = toeplitz_from(fourier_coeffs(edges, eps, period, 2 * M), M);
    const Matrix tinv = inverse(toeplitz_from(fourier_coeffs(edges, inv, period, 2 * M), M));
    const int n1 = 2 * M1 + 1, n2 = 2 * M2 + 1, N = n1 * n2;
    LayerOperators2D op;
    op.eps = Matrix(N, N);
    op.cx = Matrix(N, N);
    op.cy = Matrix(N, N);
    for (int m = 0; m < n1; ++m)
        for (int n = 0; n < n2; ++n)
            for (int mp = 0; mp < n1; ++mp)
                for (int np = 0; np < n2; ++np) {
                    const int row = m * n2 + n, col = mp * n2 + np;
                    if (along_x) {
                        if (n != np) continue;
                        op.eps(row, col) = te(m, mp);
                        op.cy(row, col) = te(m, mp);
                        op.cx(row, col) = tinv(m, mp);
                    } else {
                        if (m != mp) continue;
                        op.eps(row, col) = te(n, np);
                        op.cx(row, col) = te(n, np);
                        op.cy(row, col) = tinv(n, np);
                    }
                }
    return op;
}

struct Grating1D {
    std::vector<double> edges;
    Vector n_values;
    double d;
};

// 一维结构：Grating1D 的栅层 + 两个均匀膜（栅层在中间），入射 nI、出射 nII
static rcwa::Stack stack_1d(const Grating1D& g, double period, double nI, cplx nII, bool films_above, bool films_below) {
    rcwa::Stack s;
    s.nI = nI;
    s.nII = nII;
    s.period = period;
    if (films_above) s.layers.push_back(rcwa::Layer::uniform(0.12, 1.7));
    s.layers.push_back(rcwa::Layer::grating(g.d, g.edges, g.n_values));
    if (films_below) {
        s.layers.push_back(rcwa::Layer::uniform(0.2, cplx{2.0, 0.03}));
        s.layers.push_back(rcwa::Layer::uniform(0.09, 1.4));
    }
    return s;
}
static Problem problem_from_1d(const Grating1D& g, double period, double nI, cplx nII, bool films_above, bool films_below,
                                bool along_x, int M1, int M2, double lambda, double theta, double phi) {
    Problem p;
    p.stack.nI = nI;
    p.stack.nII = nII;
    const double other = 0.7;
    p.stack.Lx = along_x ? period : other;
    p.stack.Ly = along_x ? other : period;
    p.wavelength = lambda;
    p.theta = theta;
    p.phi = phi;
    p.M1 = M1;
    p.M2 = M2;
    auto add_uniform = [&](double d, cplx n) {
        p.stack.layers.push_back(rcwa::Layer2D::uniform(d, n));
        p.ops.emplace_back();
    };
    if (films_above) add_uniform(0.12, 1.7);
    {
        rcwa::XProfile2D row;
        row.edges = g.edges;
        row.n_values = g.n_values;
        std::vector<rcwa::XProfile2D> rows{row};
        std::vector<double> ye{0.0, along_x ? other : period};
        if (!along_x) {   // 沿 y 变化：x 方向是均匀的，剖面放在 y 分段里
            rows.clear();
            for (std::size_t k = 0; k < g.n_values.size(); ++k) {
                rcwa::XProfile2D r;
                r.edges = {0.0, other};
                r.n_values = {g.n_values[k]};
                rows.push_back(r);
            }
            ye = g.edges;
        }
        p.stack.layers.push_back(rcwa::Layer2D::patterned(g.d, ye, rows));
        p.ops.push_back(kron_ops(g.edges, g.n_values, period, M1, M2, along_x));
    }
    if (films_below) {
        add_uniform(0.2, cplx{2.0, 0.03});
        add_uniform(0.09, 1.4);
    }
    return p;
}

static void test_one_d_limits() {
    say("[3] 一维极限：y 不变 / x 不变的结构对 rcwa::solve 与 rcwa::solve_conical\n");
    const double period = 0.574;
    const Grating1D grating{{0.0, 0.172, 0.574}, {1.46, 1.0}, 0.38};
    const Grating1D grating3{{0.0, 0.1, 0.35, 0.574}, {1.5, 2.2, 1.0}, 0.31};
    double worst_te = 0.0, worst_tm = 0.0, worst_con = 0.0, worst_rot = 0.0, worst_zero = 0.0;
    int idx = 0;
    for (const Grating1D* g : {&grating, &grating3}) {
        for (int films = 0; films < 4; ++films) {
            const bool above = films & 1, below = films & 2;
            ++idx;
            const int M = 6;
            const double nI = 1.0;
            const cplx nII{1.46, 0.0};
            const rcwa::Stack s1 = stack_1d(*g, period, nI, nII, above, below);
            // --- φ = 0：TE ↔ s（Ey 振幅逐复数比较），TM ↔ p（效率比较）；M2 > 0 时多余的 n 级不被激发
            for (int M2 : {0, 2}) {
                const double theta = 37.0 + 5.0 * idx, lambda = 0.9;
                {
                    const Problem p = problem_from_1d(*g, period, nI, nII, above, below, true, M, M2, lambda, theta, 0.0);
                    Problem ps = p;
                    ps.jones = {cplx{1.0, 0.0}, cplx{0.0, 0.0}};
                    const rcwa::Result2D r = run(ps);
                    const rcwa::Result te = rcwa::solve(s1, lambda, theta, M, rcwa::Polarization::TE);
                    CHECK(r.status == rcwa::Status2D::Ok);
                    for (int m = -M; m <= M; ++m) {
                        const std::size_t i = static_cast<std::size_t>(rcwa::harmonic_index_2d(m, 0, M, M2));
                        worst_te = std::max(worst_te, std::abs(r.Ey_r[i] - te.R[static_cast<std::size_t>(m + M)]));
                        worst_te = std::max(worst_te, std::abs(r.Ey_t[i] - te.T[static_cast<std::size_t>(m + M)]));
                        worst_te = std::max(worst_te, std::abs(r.DEr[i] - te.DEr[static_cast<std::size_t>(m + M)]));
                        worst_te = std::max(worst_te, std::abs(r.DEt[i] - te.DEt[static_cast<std::size_t>(m + M)]));
                    }
                    for (int m = -M; m <= M; ++m)
                        for (int n = -M2; n <= M2; ++n)
                            if (n != 0) worst_zero = std::max(worst_zero, std::abs(r.Rs[static_cast<std::size_t>(rcwa::harmonic_index_2d(m, n, M, M2))]));
                    ps.jones = {cplx{0.0, 0.0}, cplx{1.0, 0.0}};
                    const rcwa::Result2D rp = run(ps);
                    const rcwa::Result tm = rcwa::solve(s1, lambda, theta, M, rcwa::Polarization::TM);
                    for (int m = -M; m <= M; ++m) {
                        const std::size_t i = static_cast<std::size_t>(rcwa::harmonic_index_2d(m, 0, M, M2));
                        worst_tm = std::max(worst_tm, std::abs(rp.DEr[i] - tm.DEr[static_cast<std::size_t>(m + M)]));
                        worst_tm = std::max(worst_tm, std::abs(rp.DEt[i] - tm.DEt[static_cast<std::size_t>(m + M)]));
                    }
                }
                // --- φ ≠ 0：圆锥入射，s/p 复振幅逐级逐复数比较（单位 Jones 矢量）
                for (double phi : {23.0, -140.0}) {
                    const rcwa::Jones2D jones{cplx{0.6, 0.3}, cplx{0.2, -0.7}};
                    const double nrm = std::sqrt(std::norm(jones[0]) + std::norm(jones[1]));
                    Problem p = problem_from_1d(*g, period, nI, nII, above, below, true, M, M2, lambda, theta, phi);
                    p.jones = {jones[0] / nrm, jones[1] / nrm};
                    const rcwa::Result2D r = run(p);
                    const rcwa::ConicalResult con = rcwa::solve_conical(s1, lambda, theta, phi, M);
                    const rcwa::ConicalPolarized pol = con.for_input(p.jones[0], p.jones[1]);
                    CHECK(r.status == rcwa::Status2D::Ok);
                    for (int m = -M; m <= M; ++m) {
                        const std::size_t i = static_cast<std::size_t>(rcwa::harmonic_index_2d(m, 0, M, M2));
                        const std::size_t k = static_cast<std::size_t>(m + M);
                        for (double e : {std::abs(r.Rs[i] - pol.Rs[k]), std::abs(r.Rp[i] - pol.Rp[k]), std::abs(r.Ts[i] - pol.Ts[k]),
                                         std::abs(r.Tp[i] - pol.Tp[k]), std::abs(r.DEr[i] - pol.DEr[k]), std::abs(r.DEt[i] - pol.DEt[k])})
                            worst_con = std::max(worst_con, e);
                    }
                }
            }
            // --- x 不变（沿 y 变化）：旋转 90° 后就是 φ' = φ − 90° 的一维圆锥问题，(0, n) 级 ↔ m' = n
            for (double phi : {35.0, 90.0, -170.0}) {
                const double theta = 41.0, lambda = 0.9;
                for (int M1 : {0, 2}) {
                    Problem p = problem_from_1d(*g, period, nI, nII, above, below, false, M1, M, lambda, theta, phi);
                    p.jones = {cplx{0.8, 0.0}, cplx{0.0, 0.6}};
                    const rcwa::Result2D r = run(p);
                    const rcwa::ConicalResult con = rcwa::solve_conical(s1, lambda, theta, phi - 90.0, M);
                    const rcwa::ConicalPolarized pol = con.for_input(p.jones[0], p.jones[1]);
                    CHECK(r.status == rcwa::Status2D::Ok);
                    for (int n = -M; n <= M; ++n) {
                        const std::size_t i = static_cast<std::size_t>(rcwa::harmonic_index_2d(0, n, M1, M));
                        const std::size_t k = static_cast<std::size_t>(n + M);
                        for (double e : {std::abs(r.Rs[i] - pol.Rs[k]), std::abs(r.Rp[i] - pol.Rp[k]), std::abs(r.Ts[i] - pol.Ts[k]),
                                         std::abs(r.Tp[i] - pol.Tp[k]), std::abs(r.DEr[i] - pol.DEr[k]), std::abs(r.DEt[i] - pol.DEt[k])})
                            worst_rot = std::max(worst_rot, e);
                    }
                }
            }
        }
    }
    say("  TE（Ey 振幅与效率）最大差 %.2e；TM（效率）%.2e；圆锥 y 不变（s/p 振幅与效率）%.2e；x 不变（旋转 90°）%.2e；\n"
        "  未激发的 n≠0 级振幅最大 %.2e\n", worst_te, worst_tm, worst_con, worst_rot, worst_zero);
    CHECK_LE(worst_te, 1e-9);
    CHECK_LE(worst_tm, 1e-9);
    CHECK_LE(worst_con, 1e-9);
    CHECK_LE(worst_rot, 1e-9);
    CHECK_LE(worst_zero, 1e-10);
}

// x↔y 互换：反射对称面 x = y。把 y 不变的结构换成 x 不变（同一剖面）、φ → 90° − φ，纯 s 或纯 p 入射时各级 (m, n) ↔ (n, m) 的
// 振幅模与效率不变（互换是镜像，s 振幅会差一个符号，所以只比模）。
static void test_xy_swap() {
    say("[3b] x↔y 互换（(m, n) ↔ (n, m)，φ → 90° − φ）\n");
    const double period = 0.574;
    const Grating1D g{{0.0, 0.1, 0.35, 0.574}, {1.5, 2.2, 1.0}, 0.31};
    double worst = 0.0;
    for (int pol = 0; pol < 2; ++pol)
        for (double phi : {25.0, -64.0}) {
            const int M = 3;
            const double theta = 39.0, lambda = 0.8;
            Problem a = problem_from_1d(g, period, 1.0, 1.46, true, false, true, M, M, lambda, theta, phi);
            Problem b = problem_from_1d(g, period, 1.0, 1.46, true, false, false, M, M, lambda, theta, 90.0 - phi);
            a.jones = b.jones = pol == 0 ? rcwa::Jones2D{cplx{1.0, 0.0}, cplx{0.0, 0.0}} : rcwa::Jones2D{cplx{0.0, 0.0}, cplx{1.0, 0.0}};
            const rcwa::Result2D ra = run(a), rb = run(b);
            CHECK(ra.status == rcwa::Status2D::Ok && rb.status == rcwa::Status2D::Ok);
            for (int m = -M; m <= M; ++m)
                for (int n = -M; n <= M; ++n) {
                    const std::size_t i = static_cast<std::size_t>(rcwa::harmonic_index_2d(m, n, M, M));
                    const std::size_t j = static_cast<std::size_t>(rcwa::harmonic_index_2d(n, m, M, M));
                    worst = std::max({worst, std::abs(ra.DEr[i] - rb.DEr[j]), std::abs(ra.DEt[i] - rb.DEt[j]),
                                      std::abs(std::abs(ra.Rs[i]) - std::abs(rb.Rs[j])), std::abs(std::abs(ra.Rp[i]) - std::abs(rb.Rp[j])),
                                      std::abs(std::abs(ra.Ts[i]) - std::abs(rb.Ts[j])), std::abs(std::abs(ra.Tp[i]) - std::abs(rb.Tp[j]))});
                }
        }
    say("  最大差 %.2e\n", worst);
    CHECK_LE(worst, 1e-9);
}

// ---------------------------------------------------------------------------------------------
// 4. 能量守恒与 2×2 超胞

static void test_energy_and_supercell() {
    say("[4] 能量守恒（无吸收算例）与 2×2 超胞等价\n");
    double worst = 0.0;
    for (int ci = 0; ci < kCoreRefCaseCount; ++ci) {
        const CoreRefCase& c = kCoreRefCases[ci];
        if (c.status != 0) continue;
        bool lossless = c.nII_im == 0.0;
        for (int j = 0; j < c.layer_count; ++j) {
            const CoreRefLayer& L = c.layers[j];
            if (L.homogeneous) lossless = lossless && L.n_im == 0.0;
            else {
                int total = 0;
                for (int k = 0; k < L.K; ++k) total += L.x_count[k];
                for (int s = 0; s < total; ++s) lossless = lossless && L.n_values[2 * s + 1] == 0.0;
            }
        }
        if (!lossless) continue;
        const rcwa::Result2D r = run(problem_from_case(c));
        const double e = std::abs(total_energy(r) - 1.0) / std::max(1e-300, std::norm(cplx{c.jones[0], c.jones[1]}) + c.jones[2] * c.jones[2]);
        worst = std::max(worst, e);
        CHECK_LE(e, 1e-9);
    }
    say("  无吸收算例最大 |ΣDE − 1|（按入射强度归一）= %.2e\n", worst);

    // 超胞：cell_small（M = 1×1）与 supercell_small（Lx、Ly 加倍，M = 2×2）。
    // 超胞的 (2m, 2n) 级就是单元的 (m, n) 级，奇数级不被激发。算子装配在 Python 夹具里（Cx、Cy 的 Li 规则在超胞里也成立）。
    const CoreRefCase *uc = nullptr, *sc = nullptr;
    for (int ci = 0; ci < kCoreRefCaseCount; ++ci) {
        if (std::strcmp(kCoreRefCases[ci].name, "cell_small") == 0) uc = &kCoreRefCases[ci];
        if (std::strcmp(kCoreRefCases[ci].name, "supercell_small") == 0) sc = &kCoreRefCases[ci];
    }
    CHECK(uc != nullptr && sc != nullptr);
    if (uc && sc) {
        const rcwa::Result2D a = run(problem_from_case(*uc)), b = run(problem_from_case(*sc));
        double e = 0.0, odd = 0.0;
        for (int m = -uc->M1; m <= uc->M1; ++m)
            for (int n = -uc->M2; n <= uc->M2; ++n) {
                const std::size_t i = static_cast<std::size_t>(rcwa::harmonic_index_2d(m, n, uc->M1, uc->M2));
                const std::size_t j = static_cast<std::size_t>(rcwa::harmonic_index_2d(2 * m, 2 * n, sc->M1, sc->M2));
                e = std::max({e, std::abs(a.Rs[i] - b.Rs[j]), std::abs(a.Rp[i] - b.Rp[j]), std::abs(a.Ts[i] - b.Ts[j]),
                              std::abs(a.Tp[i] - b.Tp[j]), std::abs(a.DEr[i] - b.DEr[j]), std::abs(a.DEt[i] - b.DEt[j]),
                              std::abs(a.Kx[i] - b.Kx[j]), std::abs(a.Ky[i] - b.Ky[j])});
            }
        for (int m = -sc->M1; m <= sc->M1; ++m)
            for (int n = -sc->M2; n <= sc->M2; ++n) {
                if (m % 2 == 0 && n % 2 == 0) continue;
                const std::size_t j = static_cast<std::size_t>(rcwa::harmonic_index_2d(m, n, sc->M1, sc->M2));
                odd = std::max({odd, std::abs(b.Rs[j]), std::abs(b.Rp[j]), std::abs(b.Ts[j]), std::abs(b.Tp[j])});
            }
        say("  超胞（M=2×2，N=25）与单元（M=1×1，N=9）：偶数级最大差 %.2e，奇数级振幅最大 %.2e；总效率差 %.2e\n", e, odd,
            std::abs(total_energy(a) - total_energy(b)));
        CHECK_LE(e, 1e-10);
        CHECK_LE(odd, 1e-10);
    }
}

// ---------------------------------------------------------------------------------------------
// 5. 选项互相一致：快路径、parallel_layers、线程数、custom_eig、capture

static const CoreRefCase& find_case(const char* name) {
    for (int ci = 0; ci < kCoreRefCaseCount; ++ci)
        if (std::strcmp(kCoreRefCases[ci].name, name) == 0) return kCoreRefCases[ci];
    std::fprintf(stderr, "no such case %s\n", name);
    std::exit(2);
}

// 把夹具算例改成别的层序：膜堆（3 层）放到结构层之上或之下
static Problem with_films(const CoreRefCase& c, bool above, bool below) {
    Problem p = problem_from_case(c);
    std::vector<rcwa::Layer2D> layers;
    std::vector<LayerOperators2D> ops;
    if (above) {
        layers.push_back(rcwa::Layer2D::uniform(0.11, 1.6));
        ops.emplace_back();
        layers.push_back(rcwa::Layer2D::uniform(0.07, cplx{2.2, 0.01}));
        ops.emplace_back();
    }
    for (std::size_t j = 0; j < p.stack.layers.size(); ++j) {
        layers.push_back(p.stack.layers[j]);
        ops.push_back(p.ops[j]);
    }
    if (below) {
        layers.push_back(rcwa::Layer2D::uniform(0.13, 1.9));
        ops.emplace_back();
        layers.push_back(rcwa::Layer2D::uniform(0.21, cplx{1.3, 0.0}));
        ops.emplace_back();
        layers.push_back(rcwa::Layer2D::uniform(0.05, 2.4));
        ops.emplace_back();
    }
    p.stack.layers = std::move(layers);
    p.ops = std::move(ops);
    return p;
}

static void test_option_equivalence() {
    say("[5] 选项互相一致\n");
    // 快路径 vs 稠密路径：均匀层在结构层之上、之下、两侧；也含纯膜系
    double worst_fast = 0.0;
    for (const char* name : {"pillar_square", "pillar_eccentric", "pillar_lossy", "pillar_elliptical"}) {
        for (int pos = 0; pos < 4; ++pos) {
            const Problem p = with_films(find_case(name), pos & 1, pos & 2);
            rcwa::Options2D slow;
            slow.homogeneous_fast_path = false;
            const rcwa::Result2D a = run(p), b = run(p, slow);
            CHECK(a.status == rcwa::Status2D::Ok && b.status == rcwa::Status2D::Ok);
            const double e = result_distance(a, b);
            worst_fast = std::max(worst_fast, e);
            CHECK_LE(e, 1e-10);
            CHECK(a.n_eig == 1);
            CHECK(b.n_eig == static_cast<int>(p.stack.layers.size()));
        }
    }
    {
        const Problem p = with_films(find_case("film"), true, true);   // 纯膜系
        rcwa::Options2D slow;
        slow.homogeneous_fast_path = false;
        const double e = result_distance(run(p), run(p, slow));
        worst_fast = std::max(worst_fast, e);
        CHECK_LE(e, 1e-10);
    }
    say("  快路径与稠密路径（均匀层也做本征分解）的最大差 %.2e\n", worst_fast);

    // 直接递推的界面（默认）与 S 块路径：层序、快路径开关、结构层 + 膜堆 + 结构层
    double worst_direct = 0.0;
    for (const char* name : {"pillar_square", "pillar_lossy", "pillar_eccentric", "struct_film_struct", "multi_segments"}) {
        for (int pos = 0; pos < 4; ++pos)
            for (int fast = 1; fast >= 0; --fast) {
                const Problem p = with_films(find_case(name), pos & 1, pos & 2);
                rcwa::Options2D o;
                o.homogeneous_fast_path = fast != 0;
                const double e = result_distance(run(p, o), run_s_blocks(p, o));
                worst_direct = std::max(worst_direct, e);
                CHECK_LE(e, 1e-10);
            }
    }
    say("  直接递推的界面与 S 块路径的最大差 %.2e\n", worst_direct);

    // 强反射的出射介质（金属衬底，r 接近 −1；(I + r) 近奇异也不影响直接递推）：直接递推、S 块、稠密路径互相一致
    {
        double worst_metal = 0.0;
        for (const cplx nII : {cplx{0.05, 4.0}, cplx{0.01, 50.0}, cplx{1000.0, 0.0}})
            for (int pos = 0; pos < 4; ++pos) {
                Problem p = with_films(find_case("pillar_lossy"), pos & 1, pos & 2);
                p.stack.nII = nII;
                const rcwa::Result2D a = run(p), b = run_s_blocks(p);
                rcwa::Options2D slow;
                slow.homogeneous_fast_path = false;
                const rcwa::Result2D c = run(p, slow);
                CHECK(a.status == rcwa::Status2D::Ok && b.status == rcwa::Status2D::Ok && c.status == rcwa::Status2D::Ok);
                const double e = std::max(result_distance(a, b), result_distance(a, c));
                worst_metal = std::max(worst_metal, e);
                CHECK_LE(e, 1e-9);
            }
        say("  强反射出射介质（0.05+4i、0.01+50i、n=1000）：三条路径最大差 %.2e\n", worst_metal);
    }

    // parallel_layers 与线程数：结果逐位相同
    {
        const Problem p = problem_from_case(find_case("struct_film_struct"));
        const rcwa::Result2D base = run(p);
        rcwa::Options2D par;
        par.parallel_layers = true;
        bool same = same_result(base, run(p, par));
        set_num_threads(4);
        same = same && same_result(base, run(p, par)) && same_result(base, run(p));
        rcwa::Options2D slow;
        slow.homogeneous_fast_path = false;
        const rcwa::Result2D slow1 = run(p, slow);
        slow.parallel_layers = true;
        same = same && same_result(slow1, run(p, slow));
        set_num_threads(1);
        same = same && same_result(base, run(p, par));
        CHECK(same);
        say("  parallel_layers 与线程数（1、4）：结果逐位相同 = %s\n", same ? "是" : "否");
    }

    // 可重入：几个用户线程同时求解（每个自己的 Problem、选项），结果与串行逐位相同
    {
        std::vector<Problem> problems;
        for (const char* name : {"pillar_square", "struct_film_struct", "pillar_eccentric", "pillar_lossy"})
            problems.push_back(with_films(find_case(name), true, false));
        std::vector<rcwa::Result2D> serial, concurrent(problems.size());
        for (const Problem& p : problems) serial.push_back(run(p));
        std::vector<std::thread> pool;
        for (std::size_t i = 0; i < problems.size(); ++i)
            pool.emplace_back([&, i] {
                rcwa::Options2D o;
                o.parallel_layers = i % 2 == 0;
                concurrent[i] = run(problems[i], o);
            });
        for (std::thread& t : pool) t.join();
        bool same = true;
        for (std::size_t i = 0; i < problems.size(); ++i) same = same && same_result(serial[i], concurrent[i]);
        CHECK(same);
        say("  4 个线程同时求解：与串行逐位相同 = %s\n", same ? "是" : "否");
    }

    // custom_eig 钩子：转发到 eig()，调用次数 = 做了本征分解的层数，结果逐位相同
    {
        const Problem p = with_films(find_case("struct_film_struct"), true, false);
        int calls = 0;
        rcwa::Options2D o;
        o.custom_eig = [&](const Matrix& a) {
            ++calls;
            return eig(a, EigMethod::QR);
        };
        const rcwa::Result2D a = run(p), b = run(p, o);
        CHECK(calls == 2);
        CHECK(b.n_eig == 2);
        CHECK(same_result(a, b));
        calls = 0;
        o.homogeneous_fast_path = false;
        const rcwa::Result2D c = run(p, o);
        CHECK(calls == static_cast<int>(p.stack.layers.size()));
        // LR 迭代的结果与 QR 一致（同一个本征问题）
        rcwa::Options2D lr;
        lr.eig = EigMethod::LR;
        const double e = result_distance(a, run(p, lr));
        CHECK_LE(e, 1e-8);          // LR 是非酉的消元变换，精度比 QR 低（实测 1.6e-10）
        // diagnostics：本征残差
        rcwa::Options2D dg;
        dg.diagnostics = true;
        const rcwa::Result2D d = run(p, dg);
        CHECK(d.max_eig_residual > 0.0 && d.max_eig_residual < 1e-12);
        CHECK(same_result(a, d));
        say("  custom_eig 被调用 %d 次（结构层数 2）；LR 与 QR 差 %.2e；本征残差 %.2e\n", b.n_eig, e, d.max_eig_residual);
    }
}

// ---------------------------------------------------------------------------------------------
// 6. capture：与不带 capture 逐位相同；界面连续性

// 层内深度 zl 处的 E、h（2N）
static void layer_fields(const d2::LayerCapture2D& L, double k0, double zl, Vector& E, Vector& H) {
    const std::size_t n = L.q.size();
    Vector cp(n), cm(n);
    for (std::size_t i = 0; i < n; ++i) {
        cp[i] = std::exp(-k0 * L.q[i] * zl) * L.a_plus[i];
        cm[i] = std::exp(-k0 * L.q[i] * (L.d - zl)) * L.a_minus[i];
    }
    Vector sum(n), dif(n);
    for (std::size_t i = 0; i < n; ++i) {
        sum[i] = cp[i] + cm[i];
        dif[i] = -cp[i] + cm[i];
    }
    if (L.homogeneous) {
        E = sum;
        H = d2::b2_apply(L.Vbd, dif);
    } else {
        E = L.W * sum;
        H = L.V * dif;
    }
}
static double rel_diff(const Vector& a, const Vector& b) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        num += std::norm(a[i] - b[i]);
        den += std::norm(a[i]) + std::norm(b[i]);
    }
    return std::sqrt(num / std::max(den, 1e-300));
}

// 由 capture 重建 z 处的谐波列（与 rcwa2d.py 的 field_at 同一公式；区域 I 里入射项只对被激发的零阶求指数）
struct Fields6 {
    Vector Ex, Ey, Ez, Hx, Hy, Hz;
};
static Fields6 fields_at(const d2::FieldCapture2D& cap, double z) {
    const std::size_t N = static_cast<std::size_t>(cap.N), n = 2 * N;
    const double k0 = cap.k0;
    Vector F(n), G(n);
    cplx Ei_scalar{};
    const Matrix* Ei = nullptr;
    if (z < 0.0) {
        Vector dif(n);
        for (std::size_t i = 0; i < n; ++i) {
            const cplx re = cap.R[i] * std::exp(k0 * cap.region_I.q[i] * z);
            const cplx in = cap.inc[i] == cplx{} ? cplx{} : cap.inc[i] * std::exp(-k0 * cap.region_I.q[i] * z);
            F[i] = in + re;
            dif[i] = -in + re;
        }
        G = d2::b2_apply(cap.region_I.V, dif);
        Ei_scalar = 1.0 / (cap.nI * cap.nI);
    } else if (z > cap.z_bot) {
        Vector t(n), mt(n);
        for (std::size_t i = 0; i < n; ++i) {
            t[i] = cap.T[i] * std::exp(-k0 * cap.region_II.q[i] * (z - cap.z_bot));
            mt[i] = -t[i];
        }
        F = t;
        G = d2::b2_apply(cap.region_II.V, mt);
        Ei_scalar = 1.0 / (cap.nII * cap.nII);
    } else {
        std::size_t j = 0;
        while (j + 1 < cap.layers.size() && !(z <= cap.layers[j].z_top + cap.layers[j].d + 1e-12)) ++j;
        const d2::LayerCapture2D& L = cap.layers[j];
        const double zl = z - L.z_top;
        Vector sum(n), dif(n);
        for (std::size_t i = 0; i < n; ++i) {
            const cplx cp = std::exp(-k0 * L.q[i] * zl) * L.a_plus[i], cm = std::exp(-k0 * L.q[i] * (L.d - zl)) * L.a_minus[i];
            sum[i] = cp + cm;
            dif[i] = -cp + cm;
        }
        if (L.homogeneous) {
            F = sum;
            G = d2::b2_apply(L.Vbd, dif);
            Ei_scalar = 1.0 / L.eps;
        } else {
            F = L.W * sum;
            G = L.V * dif;
            Ei = &L.Ei;
        }
    }
    Fields6 out;
    out.Ex.assign(F.begin(), F.begin() + static_cast<std::ptrdiff_t>(N));
    out.Ey.assign(F.begin() + static_cast<std::ptrdiff_t>(N), F.end());
    Vector hx(G.begin(), G.begin() + static_cast<std::ptrdiff_t>(N)), hy(G.begin() + static_cast<std::ptrdiff_t>(N), G.end());
    Vector t(N);
    out.Hx.resize(N);
    out.Hy.resize(N);
    out.Hz.resize(N);
    for (std::size_t i = 0; i < N; ++i) {
        t[i] = cap.Ky[i] * hx[i] - cap.Kx[i] * hy[i];
        out.Hx[i] = cplx{0.0, -1.0} * hx[i];
        out.Hy[i] = cplx{0.0, -1.0} * hy[i];
        out.Hz[i] = cplx{0.0, -1.0} * (cplx{0.0, 1.0} * (cap.Ky[i] * out.Ex[i] - cap.Kx[i] * out.Ey[i]));
    }
    out.Ez.resize(N);
    if (Ei) {
        const Vector u = *Ei * t;
        for (std::size_t i = 0; i < N; ++i) out.Ez[i] = cplx{0.0, 1.0} * u[i];
    } else {
        for (std::size_t i = 0; i < N; ++i) out.Ez[i] = cplx{0.0, 1.0} * (Ei_scalar * t[i]);
    }
    return out;
}

// capture 重建的场与 rcwa2d.py 的 field_at 逐复数比较
static Cmp compare_fields_with_case(const d2::FieldCapture2D& cap, const CoreRefCase& c) {
    Cmp m;
    const std::size_t N = static_cast<std::size_t>(c.N);
    for (int k = 0; k < c.field_count; ++k) {
        const Fields6 f = fields_at(cap, c.field_z[k]);
        const double* d = c.field_data + 2 * 6 * N * static_cast<std::size_t>(k);
        const Vector* comp[6] = {&f.Ex, &f.Ey, &f.Ez, &f.Hx, &f.Hy, &f.Hz};
        for (int q = 0; q < 6; ++q) cmp_complex(*comp[q], d + 2 * N * static_cast<std::size_t>(q), N, m);
    }
    return m;
}

static double check_capture_continuity(const d2::FieldCapture2D& cap) {
    double worst = 0.0;
    const std::size_t n = cap.inc.size();
    // 区域 I 在 z = 0：E = inc + R，h = V_I·(−inc + R)
    Vector E(n), H, E2, H2;
    for (std::size_t i = 0; i < n; ++i) E[i] = cap.inc[i] + cap.R[i];
    Vector dif(n);
    for (std::size_t i = 0; i < n; ++i) dif[i] = -cap.inc[i] + cap.R[i];
    H = d2::b2_apply(cap.region_I.V, dif);
    Vector Eprev = E, Hprev = H;
    for (std::size_t j = 0; j < cap.layers.size(); ++j) {
        layer_fields(cap.layers[j], cap.k0, 0.0, E2, H2);
        worst = std::max({worst, rel_diff(Eprev, E2), rel_diff(Hprev, H2)});
        layer_fields(cap.layers[j], cap.k0, cap.layers[j].d, Eprev, Hprev);
    }
    // 区域 II 在 z = z_bot：E = T，h = −V_II·T
    Vector mt(n);
    for (std::size_t i = 0; i < n; ++i) mt[i] = -cap.T[i];
    worst = std::max({worst, rel_diff(Eprev, cap.T), rel_diff(Hprev, d2::b2_apply(cap.region_II.V, mt))});
    return worst;
}

static void test_capture() {
    say("[6] capture：与不带 capture 逐位相同；界面连续性\n");
    double worst = 0.0;
    bool same = true;
    int count = 0;
    for (const char* name : {"pillar_square", "pillar_lossy", "struct_film_struct", "pillar_eccentric", "film", "interface"}) {
        for (int pos = 0; pos < 4; ++pos) {
            const Problem p = with_films(find_case(name), pos & 1, pos & 2);
            for (int fast = 1; fast >= 0; --fast) {
                rcwa::Options2D o;
                o.homogeneous_fast_path = fast != 0;
                d2::FieldCapture2D cap;
                const rcwa::Result2D a = run(p), b = run(p, o, &cap);
                const rcwa::Result2D a2 = run(p, o);
                same = same && same_result(a2, b) && (fast == 0 || same_result(a, b));
                CHECK(cap.available);
                CHECK(cap.layers.size() == p.stack.layers.size());
                CHECK(cap.N == b.N && cap.k0 == b.k0);
                CHECK(same_vec(cap.R, [&] {
                    Vector v(b.Ex_r);
                    v.insert(v.end(), b.Ey_r.begin(), b.Ey_r.end());
                    return v;
                }()));
                double z = 0.0;
                bool geometry = true;
                for (std::size_t j = 0; j < cap.layers.size(); ++j) {
                    geometry = geometry && cap.layers[j].z_top == z && cap.layers[j].d == p.stack.layers[j].d;
                    geometry = geometry && (cap.layers[j].homogeneous == (fast != 0 && p.stack.layers[j].homogeneous));
                    z += p.stack.layers[j].d;
                }
                CHECK(geometry && cap.z_bot == z);
                const double e = check_capture_continuity(cap);
                worst = std::max(worst, e);
                CHECK_LE(e, 1e-8);
                ++count;
            }
        }
    }
    CHECK(same);
    say("  %d 个组合：与不带 capture 的结果逐位相同 = %s；界面连续性最大相对差 %.2e（容差 1e-8）\n", count, same ? "是" : "否", worst);

    // capture 重建的六个场分量（含 Ez = i·Ei·(Ky·hx − Kx·hy)）对 rcwa2d.py 的 field_at：z 取入射区、各层内部、出射区
    {
        double worst_f = 0.0, worst_f_slow = 0.0, worst_f_s = 0.0;
        int points = 0;
        for (int ci = 0; ci < kCoreRefCaseCount; ++ci) {
            const CoreRefCase& c = kCoreRefCases[ci];
            if (c.status != 0) continue;
            const Problem p = problem_from_case(c);
            d2::FieldCapture2D cap, cap_slow, cap_s;
            run(p, {}, &cap);
            rcwa::Options2D slow;
            slow.homogeneous_fast_path = false;
            run(p, slow, &cap_slow);
            run_s_blocks(p, {}, &cap_s);
            const Cmp a = compare_fields_with_case(cap, c), b = compare_fields_with_case(cap_slow, c), s = compare_fields_with_case(cap_s, c);
            worst_f = std::max(worst_f, a.worst);
            worst_f_slow = std::max(worst_f_slow, b.worst);
            worst_f_s = std::max(worst_f_s, s.worst);
            points += c.field_count;
            CHECK(a.worst_rel <= 1.0 && b.worst_rel <= 1.0 && s.worst_rel <= 1.0);
        }
        say("  capture 重建的场（%d 个 z 点，六个分量）对 rcwa2d.py 的 field_at：最大 |a−b| = %.2e（稠密路径 %.2e，S 块路径 %.2e）\n", points,
            worst_f, worst_f_slow, worst_f_s);
        CHECK_LE(worst_f, 1e-10);
    }

    // 厚吸收结构层：a⁻ 不能由 exp(+k0·q·d) 反推。把吸收方柱加厚到 8 µm，再接一层 3 µm 的均匀吸收膜
    {
        Problem p = problem_from_case(find_case("pillar_lossy"));
        p.stack.layers[0].d = 8.0;
        p.stack.layers.push_back(rcwa::Layer2D::uniform(3.0, cplx{1.7, 0.4}));
        p.ops.emplace_back();
        d2::FieldCapture2D cap;
        const rcwa::Result2D r = run(p, {}, &cap);
        CHECK(r.status == rcwa::Status2D::Ok && cap.available);
        const double e = check_capture_continuity(cap);
        say("  厚吸收层（8 µm 吸收方柱 + 3 µm 吸收膜）界面连续性 %.2e，|R+T| = %.4f\n", e, total_energy(r));
        CHECK_LE(e, 1e-8);
        bool finite = true;
        for (const auto& L : cap.layers) finite = finite && d2::finite_vector(L.a_plus) && d2::finite_vector(L.a_minus);
        CHECK(finite);
    }

    // capture 与 parallel_layers、线程数同用：快照逐位相同
    {
        const Problem p = with_films(find_case("struct_film_struct"), true, true);
        d2::FieldCapture2D serial, parallel;
        const rcwa::Result2D a = run(p, {}, &serial);
        rcwa::Options2D par;
        par.parallel_layers = true;
        set_num_threads(4);
        const rcwa::Result2D b = run(p, par, &parallel);
        set_num_threads(1);
        bool same_cap = serial.available && parallel.available && same_result(a, b) && same_vec(serial.R, parallel.R) &&
                        same_vec(serial.T, parallel.T) && serial.layers.size() == parallel.layers.size();
        for (std::size_t j = 0; same_cap && j < serial.layers.size(); ++j)
            same_cap = same_vec(serial.layers[j].a_plus, parallel.layers[j].a_plus) &&
                       same_vec(serial.layers[j].a_minus, parallel.layers[j].a_minus) && same_vec(serial.layers[j].q, parallel.layers[j].q) &&
                       serial.layers[j].W.rows() == parallel.layers[j].W.rows() &&
                       (serial.layers[j].W.rows() == 0 ||
                        std::memcmp(serial.layers[j].W.data(), parallel.layers[j].W.data(),
                                    sizeof(cplx) * static_cast<std::size_t>(serial.layers[j].W.rows()) * serial.layers[j].W.cols()) == 0);
        CHECK(same_cap);
    }

    // S 块路径同样能给出满足界面连续性的快照
    {
        const Problem p = with_films(find_case("struct_film_struct"), true, true);
        d2::FieldCapture2D cap;
        const rcwa::Result2D r = run_s_blocks(p, {}, &cap);
        CHECK(r.status == rcwa::Status2D::Ok && cap.available);
        CHECK_LE(check_capture_continuity(cap), 1e-8);
        CHECK(same_result(r, run_s_blocks(p)));
    }

    // 区域模式与 Ei 的尺寸；失败时快照复位
    {
        const Problem p = problem_from_case(find_case("pillar_square"));
        d2::FieldCapture2D cap;
        run(p, {}, &cap);
        const std::size_t n = static_cast<std::size_t>(2 * cap.N);
        CHECK(cap.region_I.q.size() == n && cap.region_II.q.size() == n && cap.region_I.V.size() == static_cast<std::size_t>(cap.N));
        CHECK(cap.layers.size() == 1 && cap.layers[0].q.size() == n && cap.layers[0].W.rows() == 2 * cap.N &&
              cap.layers[0].V.cols() == 2 * cap.N && cap.layers[0].Ei.rows() == cap.N && cap.layers[0].Ei.cols() == cap.N);
        CHECK(cap.layers[0].a_plus.size() == n && cap.layers[0].a_minus.size() == n);
        // Ei 确实是 Eps 的逆
        const Matrix prod = p.ops[0].eps * cap.layers[0].Ei;
        double e = 0.0;
        for (int i = 0; i < prod.rows(); ++i)
            for (int j = 0; j < prod.cols(); ++j) e = std::max(e, std::abs(prod(i, j) - (i == j ? 1.0 : 0.0)));
        CHECK_LE(e, 1e-12);
        rcwa::Options2D bad;
        bad.custom_eig = [](const Matrix& a) {
            EigResult r;
            r.info = 2;
            r.values.assign(static_cast<std::size_t>(a.rows()), cplx{});
            return r;
        };
        d2::FieldCapture2D cap2;
        const rcwa::Result2D r = run(p, bad, &cap2);
        CHECK(r.status == rcwa::Status2D::NumericalFailure && !cap2.available && cap2.layers.empty());
        const Problem ray = problem_from_case(find_case("rayleigh_incident"));
        const rcwa::Result2D r2 = run(ray, {}, &cap2);
        CHECK(r2.status == rcwa::Status2D::RayleighAnomaly && !cap2.available);
    }
}

// ---------------------------------------------------------------------------------------------
// 7. 参数检查、Rayleigh 与数值失败

static void test_arguments() {
    say("[7] 参数检查（std::invalid_argument）\n");
    const CoreRefCase& c = find_case("pillar_square");
    const Problem base = problem_from_case(c);
    const auto bad = [&](auto mutate) {
        Problem p = base;
        mutate(p);
        return p;
    };
    CHECK(run(base).status == rcwa::Status2D::Ok);
    CHECK_THROWS(run(bad([](Problem& p) { p.M1 = -1; })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.M2 = -1; })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.M1 = 2000000000; })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.wavelength = 0.0; })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.wavelength = -1.0; })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.wavelength = kNaN; })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.wavelength = HUGE_VAL; })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.theta = kNaN; })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.theta = 90.0; })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.theta = -90.0; })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.theta = 135.0; })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.phi = kNaN; })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.phi = HUGE_VAL; })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.stack.Lx = 0.0; })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.stack.Ly = -0.8; })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.stack.Lx = kNaN; })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.stack.nI = 0.0; })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.stack.nI = -1.0; })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.stack.nI = kNaN; })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.stack.nII = cplx{kNaN, 0.0}; })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.stack.nII = cplx{0.0, 0.0}; })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.stack.nII = cplx{1.5, -0.1}; })), std::invalid_argument);   // 增益
    CHECK_THROWS(run(bad([](Problem& p) { p.stack.nII = cplx{-1.5, 0.0}; })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.jones = {cplx{kNaN, 0.0}, cplx{0.0, 0.0}}; })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.jones = {cplx{1.0, 0.0}, cplx{0.0, HUGE_VAL}}; })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.stack.layers[0].d = -0.1; })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.stack.layers[0].d = kNaN; })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.stack.layers.push_back(rcwa::Layer2D::uniform(0.1, cplx{kNaN, 0.0})); p.ops.emplace_back(); })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.stack.layers.push_back(rcwa::Layer2D::uniform(0.1, cplx{0.0, 0.0})); p.ops.emplace_back(); })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.stack.layers.push_back(rcwa::Layer2D::uniform(0.1, cplx{1.5, -0.2})); p.ops.emplace_back(); })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.stack.layers.push_back(rcwa::Layer2D::uniform(-0.1, 1.5)); p.ops.emplace_back(); })), std::invalid_argument);
    // 结构层的几何
    CHECK_THROWS(run(bad([](Problem& p) { p.stack.layers[0].y_edges[1] = p.stack.layers[0].y_edges[2]; })), std::invalid_argument);   // 不递增
    CHECK_THROWS(run(bad([](Problem& p) { p.stack.layers[0].y_edges.back() += 0.1; })), std::invalid_argument);                       // 跨度不是一个周期
    CHECK_THROWS(run(bad([](Problem& p) { p.stack.layers[0].y_edges[0] = kNaN; })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.stack.layers[0].y_edges.pop_back(); })), std::invalid_argument);                          // K+1 条边界
    CHECK_THROWS(run(bad([](Problem& p) { p.stack.layers[0].x_profiles.clear(); })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.stack.layers[0].x_profiles[1].edges[1] = p.stack.layers[0].x_profiles[1].edges[0]; })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.stack.layers[0].x_profiles[1].edges.back() += 0.2; })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.stack.layers[0].x_profiles[1].n_values.pop_back(); })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.stack.layers[0].x_profiles[1].n_values[1] = cplx{kNaN, 0.0}; })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.stack.layers[0].x_profiles[1].n_values[1] = cplx{1.8, -0.1}; })), std::invalid_argument);
    // 算子的个数与尺寸
    CHECK_THROWS(run(bad([](Problem& p) { p.ops.clear(); })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.ops.emplace_back(); })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.ops[0].eps = Matrix(3, 3); })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.ops[0].cx = Matrix(p.ops[0].cx.rows(), p.ops[0].cx.cols() - 1); })), std::invalid_argument);
    CHECK_THROWS(run(bad([](Problem& p) { p.ops[0].cy = Matrix(); })), std::invalid_argument);
    // 选项
    {
        rcwa::Options2D o;
        o.eig = static_cast<EigMethod>(7);
        CHECK_THROWS(run(base, o), std::invalid_argument);
        rcwa::Options2D f;
        f.factorization = static_cast<rcwa::Factorization2D>(5);
        CHECK_THROWS(run(base, f), std::invalid_argument);
        rcwa::Options2D w;
        w.custom_eig = [](const Matrix&) {
            EigResult r;
            r.values.assign(3, cplx{});
            r.vectors = Matrix(3, 3);
            return r;
        };
        CHECK_THROWS(run(base, w), std::invalid_argument);        // 本征求解器返回的尺寸不对
        w.parallel_layers = true;
        CHECK_THROWS(run(base, w), std::invalid_argument);        // 异常能穿过并行区
    }
    // 均匀层的 ops 项被忽略：给垃圾也不影响
    {
        Problem p = with_films(c, true, true);
        const rcwa::Result2D ref = run(p);
        for (std::size_t j = 0; j < p.ops.size(); ++j)
            if (p.stack.layers[j].homogeneous) p.ops[j].eps = Matrix(1, 1, cplx{kNaN, kNaN});
        CHECK(same_result(ref, run(p)));
    }
    // 零厚层、M = 0（N = 1）可以求解
    {
        Problem p = base;
        p.M1 = p.M2 = 0;
        p.ops[0].eps = Matrix(1, 1, cplx{1.0, 0.0});
        p.ops[0].cx = p.ops[0].cy = p.ops[0].eps;
        const rcwa::Result2D r = run(p);
        CHECK(r.status == rcwa::Status2D::Ok && r.N == 1 && r.i0 == 0);
    }
}

static void test_failures() {
    say("[8] Rayleigh 异常与数值失败的状态\n");
    // Rayleigh：出射区
    {
        rcwa::Stack2D st;
        st.nI = 1.0;
        st.nII = 1.0;
        st.Lx = st.Ly = 1.0;
        st.layers.push_back(rcwa::Layer2D::uniform(0.3, 1.5));
        std::vector<LayerOperators2D> ops(1);
        const rcwa::Result2D r = rcwa::solve2d_with_operators(st, 1.0, 0.0, 0.0, 1, 1, rcwa::jones_from_angles_2d(90.0), ops);
        CHECK(r.status == rcwa::Status2D::RayleighAnomaly);
        CHECK(r.message.find("入射区") != std::string::npos && r.message.find("出射区") != std::string::npos);
        CHECK(r.message.find("(1, 0)") != std::string::npos && r.message.find("(0, -1)") != std::string::npos);
        CHECK(result_is_nan(r));
        CHECK(r.kzI[static_cast<std::size_t>(rcwa::harmonic_index_2d(1, 0, 1, 1))] == cplx{});
        CHECK(!r.prop_r[static_cast<std::size_t>(rcwa::harmonic_index_2d(1, 0, 1, 1))]);
        CHECK(r.Kx.size() == 9 && r.m.size() == 9 && r.n.size() == 9);
        // 稍微改一下角度就恢复正常
        const rcwa::Result2D ok = rcwa::solve2d_with_operators(st, 1.0, 0.01, 0.0, 1, 1, rcwa::jones_from_angles_2d(90.0), ops);
        CHECK(ok.status == rcwa::Status2D::Ok);
        CHECK(std::abs(total_energy(ok) - 1.0) < 1e-9);
    }
    // 只有出射区：nII = 1，nI = 1.5
    {
        rcwa::Stack2D st;
        st.nI = 1.5;
        st.nII = cplx{1.0, 0.0};
        st.Lx = st.Ly = 1.0;
        std::vector<LayerOperators2D> ops;
        const rcwa::Result2D r = rcwa::solve2d_with_operators(st, 1.0, 0.0, 0.0, 1, 0, rcwa::jones_from_angles_2d(90.0), ops);
        CHECK(r.status == rcwa::Status2D::RayleighAnomaly);
        CHECK(r.message.find("出射区") != std::string::npos && r.message.find("入射区") == std::string::npos);
        CHECK(r.message.find("(1, 0)") != std::string::npos && r.message.find("(-1, 0)") != std::string::npos);
    }
    const Problem p = problem_from_case(find_case("pillar_square"));
    // 本征迭代未收敛：custom_eig 报告 info
    {
        rcwa::Options2D o;
        o.custom_eig = [](const Matrix& a) {
            EigResult r;
            r.info = 5;
            r.values.assign(static_cast<std::size_t>(a.rows()), cplx{});
            return r;
        };
        const rcwa::Result2D r = run(p, o);
        CHECK(r.status == rcwa::Status2D::NumericalFailure && r.eig_info == 5 && result_is_nan(r));
        CHECK(!r.message.empty());
        o.diagnostics = true;
        o.parallel_layers = true;
        const rcwa::Result2D r2 = run(p, o);
        CHECK(r2.status == rcwa::Status2D::NumericalFailure && r2.eig_info == 5 && std::isnan(r2.max_eig_residual));
        // 真实的迭代次数不足：EigOptions 不能从这里传，改用返回 NaN 的本征值（q 为 NaN → V 含 NaN）
        rcwa::Options2D nan;
        nan.custom_eig = [](const Matrix& a) {
            EigResult r = eig(a);
            r.values[0] = cplx{kNaN, 0.0};
            return r;
        };
        const rcwa::Result2D r3 = run(p, nan);
        CHECK(r3.status == rcwa::Status2D::NumericalFailure && result_is_nan(r3));
    }
    // 算子里有 NaN；Eps 奇异（全零）
    {
        Problem q = p;
        q.ops[0].cx(0, 0) = cplx{kNaN, 0.0};
        const rcwa::Result2D r = run(q);
        CHECK(r.status == rcwa::Status2D::NumericalFailure && result_is_nan(r));
        Problem z = p;
        z.ops[0].eps = Matrix(z.ops[0].eps.rows(), z.ops[0].eps.cols());
        const rcwa::Result2D rz = run(z);
        CHECK(rz.status == rcwa::Status2D::NumericalFailure && result_is_nan(rz));
        say("  Eps 奇异：%s\n", rz.message.c_str());
    }
    // 结果的簿记
    {
        const rcwa::Result2D r = run(p);
        CHECK(r.status == rcwa::Status2D::Ok && r.message.empty());
        CHECK(r.N == 49 && r.M1 == 3 && r.M2 == 3 && r.i0 == 24);
        CHECK(r.m.size() == 49 && r.m[0] == -3 && r.n[0] == -3 && r.m[48] == 3 && r.n[48] == 3 && r.m[24] == 0 && r.n[24] == 0);
        CHECK(r.n_eig == 1 && r.n_dense_solve > 0 && r.eig_info == 0);
        CHECK(r.timing.total > 0.0 && r.timing.eig > 0.0 && r.timing.eig <= r.timing.total);
        CHECK(r.timing.linear >= 0.0 && r.timing.product > 0.0);
        CHECK(r.prop_r.size() == 49 && r.prop_t.size() == 49);
        for (std::size_t i = 0; i < 49; ++i) {
            CHECK((r.DEr[i] > 0.0) == (r.prop_r[i] != 0) || r.DEr[i] == 0.0);
            if (!r.prop_r[i]) CHECK(r.DEr[i] == 0.0 && r.Ez_r[i] == cplx{});
        }
        CHECK(std::abs(r.theta - 42.0 * kPi / 180.0) < 1e-15 && std::abs(r.k0 - 2.0 * kPi / 0.6) < 1e-12);
    }
}

// ---------------------------------------------------------------------------------------------
// 8. 计时（只测用时，不嵌参考值）

static double peak_rss_mb() {
#if defined(__unix__) || defined(__APPLE__)
    struct rusage u;
    if (getrusage(RUSAGE_SELF, &u) != 0) return -1.0;
#if defined(__APPLE__)
    return static_cast<double>(u.ru_maxrss) / (1024.0 * 1024.0);     // macOS：字节
#else
    return static_cast<double>(u.ru_maxrss) / 1024.0;                // Linux：KB
#endif
#else
    return -1.0;
#endif
}

// 良态的“方柱型”算子，N = (2M+1)²：ε = 1 + (n²−1)·f(x)·g(y)（f、g 是示性函数）的 Laurent 阵是两个 Toeplitz 阵的 Kronecker 积加单位阵；
// Cx、Cy 取 Eps 加一个与 (m−m', n−n') 有关的小扰动（只为让两个矩阵不同于 Eps，不是物理的因式分解；这里只测用时）。
static LayerOperators2D synthetic_ops(int M, double fill, double n_hi, unsigned seed) {
    const int n1 = 2 * M + 1, N = n1 * n1;
    // f 的 Fourier 系数（周期 1）：位于 [0.5 − fill/2, 0.5 + fill/2) 的示性函数
    Vector c(static_cast<std::size_t>(4 * M + 1));
    for (int h = -2 * M; h <= 2 * M; ++h) {
        if (h == 0) {
            c[static_cast<std::size_t>(2 * M)] = fill;
        } else {
            const double w = 2.0 * kPi * h;
            c[static_cast<std::size_t>(h + 2 * M)] =
                (std::exp(cplx{0.0, -w * (0.5 + fill / 2)}) - std::exp(cplx{0.0, -w * (0.5 - fill / 2)})) / cplx{0.0, -w};
        }
    }
    LayerOperators2D op;
    op.eps = Matrix(N, N);
    op.cx = Matrix(N, N);
    op.cy = Matrix(N, N);
    const double eps_hi = n_hi * n_hi - 1.0;
    unsigned state = seed;
    auto rnd = [&]() {
        state = state * 1664525u + 1013904223u;
        return static_cast<double>(state >> 8) / 16777216.0 - 0.5;
    };
    for (int m = 0; m < n1; ++m)
        for (int n = 0; n < n1; ++n)
            for (int mp = 0; mp < n1; ++mp)
                for (int np = 0; np < n1; ++np) {
                    const int row = m * n1 + n, col = mp * n1 + np;
                    cplx v = eps_hi * c[static_cast<std::size_t>(m - mp + 2 * M)] * c[static_cast<std::size_t>(n - np + 2 * M)];
                    if (row == col) v += 1.0;
                    op.eps(row, col) = v;
                    op.cx(row, col) = v;
                    op.cy(row, col) = v;
                }
    // 小扰动：Hermite 型（保持无吸收），幅度 1e-2，使 Cx、Cy 互不相同
    for (int j = 0; j < N; ++j)
        for (int i = 0; i <= j; ++i) {
            const cplx dx{0.01 * rnd(), i == j ? 0.0 : 0.01 * rnd()};
            const cplx dy{0.01 * rnd(), i == j ? 0.0 : 0.01 * rnd()};
            op.cx(i, j) += dx;
            op.cy(i, j) += dy;
            if (i != j) {
                op.cx(j, i) += std::conj(dx);
                op.cy(j, i) += std::conj(dy);
            }
        }
    return op;
}

static void run_perf(int M, int films, bool above, bool with_capture, bool quiet = false) {
    const int N = (2 * M + 1) * (2 * M + 1);
    rcwa::Stack2D st;
    st.nI = 1.0;
    st.nII = 1.46;
    st.Lx = st.Ly = 0.8;
    std::vector<LayerOperators2D> ops;
    const auto add_films = [&]() {
        for (int k = 0; k < films; ++k) {
            st.layers.push_back(rcwa::Layer2D::uniform(0.08 + 0.004 * k, k % 2 ? 1.45 : 2.1));
            ops.emplace_back();
        }
    };
    if (above) add_films();
    {
        std::vector<rcwa::XProfile2D> rows{{{0.0, 0.8}, {1.0}}};
        st.layers.push_back(rcwa::Layer2D::patterned(0.3, {0.0, 0.8}, rows));
        ops.push_back(synthetic_ops(M, 0.4, 1.8, 7));
    }
    if (!above) add_films();
    d2::FieldCapture2D cap;
    const rcwa::Result2D r = rcwa::solve2d_with_operators(st, 0.6, 30.0, 20.0, M, M, rcwa::jones_from_angles_2d(90.0), ops, {},
                                                           with_capture ? &cap : nullptr);
    const double other = r.timing.total - r.timing.eig - r.timing.linear - r.timing.product;
    if (!quiet)
        say("  N=%d (2N=%d)，1 个结构层 + %d 层均匀膜（膜堆在结构层%s），%s：status=%d，R+T=%.9f\n", N, 2 * N, films, above ? "之上" : "之下",
            with_capture ? "带 capture" : "不带 capture", static_cast<int>(r.status), total_energy(r));
    say("    总用时 %.3f s：本征分解 %.3f s（%.1f%%）、稠密线性求解 %.3f s、稠密乘法 %.3f s、其余 %.3f s；稠密求解 %d 次；峰值内存 %.0f MB\n", r.timing.total,
        r.timing.eig, 100.0 * r.timing.eig / r.timing.total, r.timing.linear, r.timing.product, other, r.n_dense_solve, peak_rss_mb());
    CHECK(r.status == rcwa::Status2D::Ok);
}

// ---------------------------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "perf") == 0) {
        const int M = argc > 2 ? std::atoi(argv[2]) : 8;
        const int films = argc > 3 ? std::atoi(argv[3]) : 20;
        const bool above = argc > 4 && std::atoi(argv[4]) != 0;
        const bool capture = argc > 5 && std::atoi(argv[5]) != 0;
        if (argc > 6) set_num_threads(std::atoi(argv[6]));
        run_perf(M, films, above, capture);
        return g_pass == g_total ? 0 : 1;
    }
    test_fixture();
    test_airy();
    test_one_d_limits();
    test_xy_swap();
    test_energy_and_supercell();
    test_option_equivalence();
    test_capture();
    test_arguments();
    test_failures();
    say("[9] 计时（小算例；大算例用 perf 参数）\n");
    run_perf(3, 20, false, false);
    run_perf(3, 20, true, false);
    say("通过 %d / 共 %d\n", g_pass, g_total);
    return g_pass == g_total ? 0 : 1;
}
