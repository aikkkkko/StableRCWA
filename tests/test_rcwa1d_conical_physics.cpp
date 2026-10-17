// tests/test_rcwa1d_conical_physics.cpp —— rcwa1d_conical.hpp（一维光栅的圆锥入射）的物理测试
//
//   c++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -I include tests/test_rcwa1d_conical_physics.cpp -o build/test_rcwa1d_conical_physics
//   c++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -I include tests/test_rcwa1d_conical_physics.cpp -o build/test_rcwa1d_conical_physics_san
//
// 只依赖 C++17 标准库。这里没有一个参照值出自 rcwa1d_conical.hpp 自己，也没有抄来的数表，参照只有三类：
//   · 闭式解：膜系的 s、p 振幅（本文件里另写的特征矩阵法，先与 Fresnel 公式、Airy 公式互相核对）；
//   · 精确成立的关系：通量幺正性、互易性、镜像与平移、超胞等价、正入射时基的旋转，以及由解析性给出的幂律；
//   · 已有的 TE/TM 求解器 rcwa::solve（φ = 0、180° 与正入射）。
//
// 约定（与 rcwa1d_conical.hpp 相同）：时间因子 exp(−iωt)，z 由入射侧指向出射侧；第 m 级随 x、y 的变化是
// exp[−i·k0·(kx_m·x + ky·y)]，kx_m = nI·sinθ·cosφ − m·λ/Λ，ky = nI·sinθ·sinφ，kt = sqrt(kx_m² + ky²)；
// 衍射面方位角 φ_m = atan2(ky, kx_m)（kx_m = ky = 0 时取入射的 φ），c = cos φ_m，s = sin φ_m；
//     ŝ = (−s, c, 0)，   t̂ = (c, s, 0)，   Es = E·ŝ，   Et = E·t̂，
//     下行波（入射、透射）Et = +(kz/n)·Ep，上行波（反射）Et = −(kz/n)·Ep，即 p̂ = ±(kz/n)·t̂ + (kt/n)·ẑ。
// 这样一束波实际的波矢是 k0·(−kx_m, −ky, ±kz)，p̂ × ŝ = k̂；由 Z0·H = n·k̂ × E，p 偏振的磁场是 Z0·H = n·Ep·ŝ，
// 入射、反射、透射都是这一个式子。下面各组用到的换算全部由这几条推出，推导写在各组开头。
// Jones 矩阵的下标 ab 表示“输出 a ← 输入 b”；第 m 级在各数组里的下标是 m + M。
//
// 容差的来由写在用到的地方。大致三档：闭式算术对闭式算术 1e-12；同一个截断问题的两种算法（或两种等价写法）
// 之间只差舍入，按各自线性求解与本征分解的条件数放大，复振幅取 1e-9·max(1, |J|)、效率取 1e-10；
// 由解析性给出的幂律，容差取下一阶项的量级。
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdarg>
#include <cstdio>
#include <exception>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "stablercwa/rcwa1d_conical.hpp"

using namespace stablercwa;
namespace rcwa = stablercwa::rcwa;
using rcwa::Polarization;

// ---------------------------------------------------------------------------------------------
// 输出与计数
//
// 文字一律经 say() 输出（用法同 std::printf），不把带中文的格式串直接交给 printf 系列函数：
// MSVC 检查 printf 的格式串时按系统代码页（中文 Windows 上是 GBK）而不是按 /utf-8 去读它，UTF-8 的汉字
// 会被读错——报 C4819；汉字后面紧跟的 % 还可能被当成半个汉字吞掉，于是误报参数个数、类型不符
// （C4474、C4477）。经 say() 转一道手，MSVC 不再解析格式串；GCC、Clang 仍照 printf 的规则核对参数
// （MinGW 除外：那里 printf 这个原型按微软 C 库的旧规则检查）。
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

// 断言的计数与报错都在下面两个普通函数里；宏只负责取文件名、行号和表达式的文字，宏里不出现中文。
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

#define CHECK(cond) check(static_cast<bool>(cond), __FILE__, __LINE__, #cond)

#define CHECK_LE(value, bound)                                                    \
    do {                                                                          \
        const double v_ = (value), b_ = (bound);                                  \
        check_le(v_, b_, __FILE__, __LINE__, #value);                             \
    } while (0)

// ---------------------------------------------------------------------------------------------
// 小工具

static double rad(double degrees) { return degrees * kPi / 180.0; }
static double deg(double radians) { return radians * 180.0 / kPi; }

// 取较大者；出现过 NaN 之后一直是 NaN（交给 CHECK_LE 时算失败）。
static double grow(double current, double value) {
    if (std::isnan(current) || std::isnan(value)) return std::numeric_limits<double>::quiet_NaN();
    return value > current ? value : current;
}
static double grow(double current, std::initializer_list<double> values) {
    for (double v : values) current = grow(current, v);
    return current;
}

static std::array<cplx, 4> elements(const rcwa::Jones& j) { return {j.ss, j.sp, j.ps, j.pp}; }
static std::array<double, 4> elements(const rcwa::JonesEfficiency& e) { return {e.ss, e.sp, e.ps, e.pp}; }

// 四个元素里模最大的
static double magnitude(const rcwa::Jones& j) {
    double worst = 0.0;
    for (const cplx& v : elements(j)) worst = grow(worst, std::abs(v));
    return worst;
}
// 两个 Jones 矩阵逐元素之差的最大模
static double gap(const rcwa::Jones& a, const rcwa::Jones& b) {
    const std::array<cplx, 4> x = elements(a), y = elements(b);
    double worst = 0.0;
    for (int k = 0; k < 4; ++k) worst = grow(worst, std::abs(x[k] - y[k]));
    return worst;
}
// 相对 max(1, |参照|) 的差：倏逝的高级次振幅可以远大于 1
static double scaled_gap(const rcwa::Jones& value, const rcwa::Jones& reference) {
    return gap(value, reference) / (std::max)(1.0, magnitude(reference));
}
static double gap(const rcwa::JonesEfficiency& a, const rcwa::JonesEfficiency& b) {
    const std::array<double, 4> x = elements(a), y = elements(b);
    double worst = 0.0;
    for (int k = 0; k < 4; ++k) worst = grow(worst, std::abs(x[k] - y[k]));
    return worst;
}
// 转置：sp 与 ps 互换
static rcwa::Jones transposed(const rcwa::Jones& j) { return rcwa::Jones{j.ss, j.ps, j.sp, j.pp}; }
static rcwa::JonesEfficiency transposed(const rcwa::JonesEfficiency& e) {
    rcwa::JonesEfficiency out = e;
    std::swap(out.sp, out.ps);
    return out;
}
// diag(−1, 1)·J·diag(−1, 1)：s 分量变号（输入、输出各一次），即两个交叉元变号
static rcwa::Jones s_flipped(const rcwa::Jones& j) { return rcwa::Jones{j.ss, -j.sp, -j.ps, j.pp}; }
static rcwa::Jones times(const rcwa::Jones& j, cplx factor) {
    return rcwa::Jones{j.ss * factor, j.sp * factor, j.ps * factor, j.pp * factor};
}

static bool all_finite(const rcwa::ConicalResult& r) {
    bool ok = true;
    for (std::size_t j = 0; j < r.Jr.size(); ++j) {
        ok = ok && std::isfinite(magnitude(r.Jr[j])) && std::isfinite(magnitude(r.Jt[j]));
        for (double v : elements(r.DEr[j])) ok = ok && std::isfinite(v);
        for (double v : elements(r.DEt[j])) ok = ok && std::isfinite(v);
    }
    return ok && !r.Jr.empty();
}

// 调用 solve_conical 并核对各数组的长度。长度不对时记一次失败，再把各数组改成长度正确、内容是 NaN 的，
// 这样后面的比较不会越界，并且全部判为失败。
static rcwa::ConicalResult conical(const rcwa::Stack& stack, double wavelength, double theta_deg, double phi_deg, int M,
                                   const rcwa::Options& options = {}) {
    rcwa::ConicalResult r = rcwa::solve_conical(stack, wavelength, theta_deg, phi_deg, M, options);
    const std::size_t n = static_cast<std::size_t>(2 * M + 1);
    const bool ok = r.orders.size() == n && r.kx.size() == n && r.phi_m_deg.size() == n && r.kzI.size() == n &&
                    r.kzII.size() == n && r.flux_r.size() == n && r.flux_t.size() == n && r.Jr.size() == n && r.Jt.size() == n &&
                    r.DEr.size() == n && r.DEt.size() == n;
    if (!ok) {
        check(false, __FILE__, __LINE__, "solve_conical: every array must hold 2M+1 orders");
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const cplx z{nan, nan};
        r.orders.assign(n, 0);
        r.kx.assign(n, nan);
        r.phi_m_deg.assign(n, nan);
        r.kzI.assign(n, z);
        r.kzII.assign(n, z);
        r.flux_r.assign(n, nan);
        r.flux_t.assign(n, nan);
        r.Jr.assign(n, rcwa::Jones{z, z, z, z});
        r.Jt.assign(n, rcwa::Jones{z, z, z, z});
        r.DEr.assign(n, rcwa::JonesEfficiency{nan, nan, nan, nan});
        r.DEt.assign(n, rcwa::JonesEfficiency{nan, nan, nan, nan});
    }
    return r;
}

// 指定入射偏振之后的结果，同样核对长度（不对时记一次失败，改成长度正确、内容是 NaN 的）
static rcwa::ConicalPolarized sized(rcwa::ConicalPolarized p, std::size_t n) {
    const bool ok = p.Rs.size() == n && p.Rp.size() == n && p.Ts.size() == n && p.Tp.size() == n && p.DEr.size() == n &&
                    p.DEt.size() == n && p.DEr_s.size() == n && p.DEr_p.size() == n && p.DEt_s.size() == n && p.DEt_p.size() == n;
    if (!ok) {
        check(false, __FILE__, __LINE__, "for_input / for_angles: every array must hold 2M+1 orders");
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const cplx z{nan, nan};
        for (Vector* v : {&p.Rs, &p.Rp, &p.Ts, &p.Tp}) v->assign(n, z);
        for (RealVector* v : {&p.DEr, &p.DEt, &p.DEr_s, &p.DEr_p, &p.DEt_s, &p.DEt_p}) v->assign(n, nan);
    }
    return p;
}
static rcwa::ConicalPolarized polarized(const rcwa::ConicalResult& r, cplx Es, cplx Ep) {
    return sized(r.for_input(Es, Ep), r.Jr.size());
}
static rcwa::ConicalPolarized polarized_by_angles(const rcwa::ConicalResult& r, double psi_deg, double delta_deg = 0.0) {
    return sized(r.for_angles(psi_deg, delta_deg), r.Jr.size());
}

// ---------------------------------------------------------------------------------------------
// 按定义另算的各级波矢、方位角与通量因子（不取自求解结果）

// sqrt 取 Im ≥ 0 的一支，Im = 0 时取 Re ≥ 0。实数单独处理：负实数的平方根是严格的纯虚数。
static cplx kz_branch(cplx v) {
    if (v.imag() == 0.0) return v.real() >= 0.0 ? cplx{std::sqrt(v.real()), 0.0} : cplx{0.0, std::sqrt(-v.real())};
    cplx kz = std::sqrt(v);
    if (kz.imag() < 0.0) kz = -kz;
    return kz;
}

struct Kinematics {
    RealVector kx;             // kx_m
    double ky = 0.0;
    RealVector c, s;           // cos φ_m、sin φ_m
    RealVector kzI2, kzII2;    // nI² − kx_m² − ky²、nII² − kx_m² − ky²（正：传播；负：倏逝）
    Vector kzI, kzII;
    RealVector flux_r, flux_t; // Re(kz) / (nI·cosθ)
};
static Kinematics kinematics(const rcwa::Stack& stack, double wavelength, double theta_deg, double phi_deg, int M) {
    const int n = 2 * M + 1;
    const double nI = stack.nI.real(), nII = stack.nII.real();
    const double sin_theta = std::sin(rad(theta_deg)), cos_theta = std::cos(rad(theta_deg));
    Kinematics k;
    k.ky = nI * sin_theta * std::sin(rad(phi_deg));
    for (int j = 0; j < n; ++j) {
        const double kx = nI * sin_theta * std::cos(rad(phi_deg)) - (j - M) * wavelength / stack.period;
        const double kt = std::hypot(kx, k.ky);
        k.kx.push_back(kx);
        k.c.push_back(kt == 0.0 ? std::cos(rad(phi_deg)) : kx / kt);
        k.s.push_back(kt == 0.0 ? std::sin(rad(phi_deg)) : k.ky / kt);
        k.kzI2.push_back(nI * nI - kx * kx - k.ky * k.ky);
        k.kzII2.push_back(nII * nII - kx * kx - k.ky * k.ky);
        k.kzI.push_back(kz_branch(k.kzI2.back()));
        k.kzII.push_back(kz_branch(k.kzII2.back()));
        k.flux_r.push_back(k.kzI.back().real() / (nI * cos_theta));
        k.flux_t.push_back(k.kzII.back().real() / (nI * cos_theta));
    }
    return k;
}

// Σ_m (flux_r·Jr†·Jr + flux_t·Jt†·Jt) − I 的最大模。通量因子用上面另算的，不用结果里的。
static double unitarity_defect(const rcwa::ConicalResult& r, const Kinematics& k) {
    cplx u[4] = {};
    for (std::size_t j = 0; j < r.Jr.size(); ++j) {
        const rcwa::Jones* both[2] = {&r.Jr[j], &r.Jt[j]};
        const double flux[2] = {k.flux_r[j], k.flux_t[j]};
        for (int side = 0; side < 2; ++side) {
            if (flux[side] == 0.0) continue;            // 倏逝级不带走能量（它的振幅再大也不该算进来）
            const rcwa::Jones& a = *both[side];
            u[0] += flux[side] * (std::conj(a.ss) * a.ss + std::conj(a.ps) * a.ps);
            u[1] += flux[side] * (std::conj(a.ss) * a.sp + std::conj(a.ps) * a.pp);
            u[2] += flux[side] * (std::conj(a.sp) * a.ss + std::conj(a.pp) * a.ps);
            u[3] += flux[side] * (std::conj(a.sp) * a.sp + std::conj(a.pp) * a.pp);
        }
    }
    return grow(0.0, {std::abs(u[0] - 1.0), std::abs(u[1]), std::abs(u[2]), std::abs(u[3] - 1.0)});
}

// ---------------------------------------------------------------------------------------------
// 测试用的结构（长度单位：微米）

static rcwa::Stack make_stack(double nI, double nII, double period, std::vector<rcwa::Layer> layers) {
    rcwa::Stack s;
    s.nI = nI;
    s.nII = nII;
    s.period = period;
    s.layers = std::move(layers);
    return s;
}
// 无吸收的单层矩形光栅
static rcwa::Stack binary_grating() {
    return make_stack(1.0, 1.46, 0.574, {rcwa::Layer::grating(0.38, {0.0, 0.1722, 0.574}, {1.46, 1.0})});
}
// 齿有吸收的单层光栅
static rcwa::Stack lossy_grating() {
    return make_stack(1.0, 1.5, 0.8, {rcwa::Layer::grating(0.3, {0.0, 0.3, 0.8}, {cplx(1.8, 0.13), 1.0})});
}
// 两个不同的光栅层夹一层吸收膜，入射介质 n = 1.2
static rcwa::Stack double_grating() {
    return make_stack(1.2, 1.5, 0.9,
                      {rcwa::Layer::grating(0.25, {0.0, 0.4, 0.9}, {2.0, 1.2}), rcwa::Layer::uniform(0.12, cplx(1.7, 0.05)),
                       rcwa::Layer::grating(0.2, {0.1, 0.35, 0.7, 1.0}, {1.5, cplx(2.2, 0.02), 1.5})});
}
// 三段、左右不对称、无吸收的光栅，下面一层膜
static rcwa::Stack skew_grating() {
    return make_stack(1.0, 1.5, 1.1,
                      {rcwa::Layer::grating(0.31, {0.0, 0.2, 0.55, 1.1}, {1.9, 1.3, 1.0}), rcwa::Layer::uniform(0.17, 1.7)});
}

struct Film {
    double d;
    cplx n;
};
// 3L·H·(2L·H)^9：20 层无吸收的介质膜
static std::vector<Film> dielectric_mirror() {
    const double n_low = 1.46, n_high = 1.96, d_high = 0.115306, d_2low = 0.30958, d_3low = 0.46438;
    std::vector<Film> films{{d_3low, n_low}, {d_high, n_high}};
    for (int k = 0; k < 9; ++k) {
        films.push_back({d_2low, n_low});
        films.push_back({d_high, n_high});
    }
    return films;
}
static rcwa::Stack film_stack(double nI, const std::vector<Film>& films, double nII, double period) {
    std::vector<rcwa::Layer> layers;
    for (const Film& f : films) layers.push_back(rcwa::Layer::uniform(f.d, f.n));
    return make_stack(nI, nII, period, layers);
}
// 无吸收的矩形光栅 + 20 层介质膜
static rcwa::Stack coated_grating() {
    rcwa::Stack s = binary_grating();
    for (const Film& f : dielectric_mirror()) s.layers.push_back(rcwa::Layer::uniform(f.d, f.n));
    return s;
}

// 所有光栅层的边界整体平移 dx
static rcwa::Stack shifted(rcwa::Stack s, double dx) {
    for (rcwa::Layer& layer : s.layers)
        for (double& e : layer.edges) e += dx;
    return s;
}
// 左右翻转：x → −x
static rcwa::Stack mirrored(rcwa::Stack s) {
    for (rcwa::Layer& layer : s.layers) {
        if (layer.homogeneous) continue;
        std::vector<double> e(layer.edges.rbegin(), layer.edges.rend());
        for (double& x : e) x = -x;
        layer.edges = e;
        std::reverse(layer.n_values.begin(), layer.n_values.end());
    }
    return s;
}
// 上下翻转：z → −z（层序颠倒，入射、出射介质互换）
static rcwa::Stack flipped(rcwa::Stack s) {
    std::reverse(s.layers.begin(), s.layers.end());
    std::swap(s.nI, s.nII);
    return s;
}
// 把周期重复 3 次，写成一个周期为 3Λ 的结构
static rcwa::Stack tripled(rcwa::Stack s) {
    for (rcwa::Layer& layer : s.layers) {
        if (layer.homogeneous) continue;
        const std::vector<double> edges = layer.edges;
        const std::vector<cplx> values = layer.n_values;
        for (int copy = 1; copy < 3; ++copy) {
            for (std::size_t k = 1; k < edges.size(); ++k) layer.edges.push_back(edges[k] + copy * s.period);
            layer.n_values.insert(layer.n_values.end(), values.begin(), values.end());
        }
    }
    s.period *= 3.0;
    return s;
}

// ---------------------------------------------------------------------------------------------
// 膜系的闭式解：特征矩阵法。
//
// 均匀膜系绕 z 轴旋转对称，入射面取在哪个方位角都一样：面内波数 kt = nI·sinθ，各层 kz = sqrt(n² − kt²)，
// s、p 互不耦合。每种偏振取一对在界面上连续的切向场 (U, V)：层内 U = A·e^{i·kz·z'} + B·e^{−i·kz·z'}，
// V = η·(A·e^{i·kz·z'} − B·e^{−i·kz·z'})（z' = k0·z），于是厚 d 的一层把层底的 (U, V) 变成层顶的：
//     [U; V]顶 = [cos δ, −i·sin δ/η; −i·η·sin δ, cos δ]·[U; V]底，   δ = k0·kz·d。
//   s：U = Es，η = kz；   p：U = Z0·H·ŝ（切向磁场），η = kz/n²。
// 入射侧 U = 1 + r、V = η1·(1 − r)，出射侧 U = t、V = ηL·t。记各层矩阵之积为 [m11, m12; m21, m22]，
//     r = [(m11 + m12·ηL)·η1 − (m21 + m22·ηL)] / [(m11 + m12·ηL)·η1 + (m21 + m22·ηL)]，   t = 2·η1 / [同一分母]。
// p 偏振的 r、t 是切向磁场之比；按文件头 Z0·H = n·Ep·ŝ（入射、反射、透射同一个 ŝ）换成电场：
//     rp = r，   tp = (nI/nII)·t。
struct FilmReference {
    cplx rs, ts, rp, tp;
};
static FilmReference film_reference(double nI, const std::vector<Film>& films, double nII, double wavelength, double theta_deg) {
    const double k0 = 2.0 * kPi / wavelength, kt = nI * std::sin(rad(theta_deg));
    FilmReference out{};
    for (int pol = 0; pol < 2; ++pol) {
        const auto kz = [&](cplx n) { return kz_branch(n * n - kt * kt); };
        const auto eta = [&](cplx n) { return pol == 0 ? kz(n) : kz(n) / (n * n); };
        cplx m11 = 1.0, m12 = 0.0, m21 = 0.0, m22 = 1.0;
        for (const Film& f : films) {
            const cplx delta = k0 * kz(f.n) * f.d, e = eta(f.n);
            const cplx cs = std::cos(delta), sn = std::sin(delta);
            const cplx a11 = cs, a12 = -kI * sn / e, a21 = -kI * e * sn, a22 = cs;
            const cplx b11 = m11 * a11 + m12 * a21, b12 = m11 * a12 + m12 * a22;
            const cplx b21 = m21 * a11 + m22 * a21, b22 = m21 * a12 + m22 * a22;
            m11 = b11;
            m12 = b12;
            m21 = b21;
            m22 = b22;
        }
        const cplx e1 = eta(nI), eL = eta(nII);
        const cplx a = (m11 + m12 * eL) * e1, b = m21 + m22 * eL;
        const cplx r = (a - b) / (a + b), t = 2.0 * e1 / (a + b);
        if (pol == 0) {
            out.rs = r;
            out.ts = t;
        } else {
            out.rp = r;
            out.tp = t * nI / nII;
        }
    }
    return out;
}

// 先核对这个参照本身：与教科书写法的 Fresnel 公式、Airy 公式一致，无吸收时能量守恒，Brewster 角处 rp = 0。
static void test_film_reference() {
    say("[参照] 特征矩阵法对 Fresnel 公式、Airy 公式\n");
    // 单个界面：用折射角写的 Fresnel 公式（p 的符号与文件头的 p̂ 一致，正入射时 rp = −rs）
    double worst_fresnel = 0.0;
    for (double theta : {0.0, 23.0, 56.0, 80.0})
        for (double n2 : {1.5, 2.4}) {
            const double n1 = 1.2, c1 = std::cos(rad(theta)), c2 = std::sqrt(1.0 - std::pow(n1 * std::sin(rad(theta)) / n2, 2));
            const FilmReference f = film_reference(n1, {}, n2, 0.9, theta);
            worst_fresnel = grow(worst_fresnel, {std::abs(f.rs - (n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2)),
                                                 std::abs(f.ts - 2.0 * n1 * c1 / (n1 * c1 + n2 * c2)),
                                                 std::abs(f.rp - (n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2)),
                                                 std::abs(f.tp - 2.0 * n1 * c1 / (n2 * c1 + n1 * c2))});
        }
    CHECK_LE(worst_fresnel, 1e-14);
    CHECK_LE(std::abs(film_reference(1.0, {}, 1.5, 0.9, deg(std::atan(1.5))).rp), 1e-15);      // Brewster 角

    // 单层膜：Airy 公式（界面系数 r_ab = (η_a − η_b)/(η_a + η_b)，t_ab = 2·η_a/(η_a + η_b)）
    double worst_airy = 0.0;
    struct Slab {
        double n0;
        cplx n1;
        double n2, d, theta;
    };
    for (const Slab& c : {Slab{1.0, 1.72, 1.45, 0.62, 41.0}, Slab{1.0, cplx(1.72, 0.05), 1.45, 0.62, 41.0},
                          Slab{1.33, cplx(0.2, 3.1), 1.5, 0.03, 60.0}, Slab{1.5, 1.0, 1.5, 0.2, 50.0}}) {
        const double wavelength = 1.053, k0 = 2.0 * kPi / wavelength, kt = c.n0 * std::sin(rad(c.theta));
        const FilmReference f = film_reference(c.n0, {{c.d, c.n1}}, c.n2, wavelength, c.theta);
        for (int pol = 0; pol < 2; ++pol) {
            const auto kz = [&](cplx n) { return kz_branch(n * n - kt * kt); };
            const auto eta = [&](cplx n) { return pol == 0 ? kz(n) : kz(n) / (n * n); };
            const cplx e0 = eta(c.n0), e1 = eta(c.n1), e2 = eta(c.n2);
            const cplx r01 = (e0 - e1) / (e0 + e1), r12 = (e1 - e2) / (e1 + e2);
            const cplx t01 = 2.0 * e0 / (e0 + e1), t12 = 2.0 * e1 / (e1 + e2);
            const cplx phase = std::exp(kI * k0 * kz(c.n1) * c.d), den = 1.0 + r01 * r12 * phase * phase;
            const cplx r = (r01 + r12 * phase * phase) / den, t = t01 * t12 * phase / den;
            worst_airy = grow(worst_airy, pol == 0 ? std::abs(f.rs - r) : std::abs(f.rp - r));
            worst_airy = grow(worst_airy, pol == 0 ? std::abs(f.ts - t) : std::abs(f.tp - t * c.n0 / c.n2));
        }
    }
    CHECK_LE(worst_airy, 1e-13);

    // 无吸收的 20 层膜：|r|² + Re(kzII)/kzI·|t|² = 1（s、p 用电场振幅写出来是同一个式子）
    double worst_energy = 0.0;
    for (double theta : {0.0, 30.0, 66.36}) {
        const double nI = 1.0, nII = 1.46, kt = nI * std::sin(rad(theta));
        const FilmReference f = film_reference(nI, dielectric_mirror(), nII, 1.053, theta);
        const double flux = kz_branch(nII * nII - kt * kt).real() / (nI * std::cos(rad(theta)));
        worst_energy = grow(worst_energy, {std::abs(std::norm(f.rs) + flux * std::norm(f.ts) - 1.0),
                                           std::abs(std::norm(f.rp) + flux * std::norm(f.tp) - 1.0)});
    }
    CHECK_LE(worst_energy, 1e-13);
    say("  最大偏差：Fresnel %.1e，Airy %.1e，20 层膜的能量守恒 %.1e\n", worst_fresnel, worst_airy, worst_energy);
}

// ---------------------------------------------------------------------------------------------
// [0] 结果里的运动学量按定义核对：kx_m、ky、kz、φ_m、通量因子，以及 效率 = 通量因子·|J|²。
// 都是几步浮点运算，容差 1e-12（kz 是开方，离截止不近时相对误差在 1e-15 量级）。
static void test_kinematics() {
    say("[0] 各级波矢、方位角、通量因子：按定义另算\n");
    const rcwa::Stack s = double_grating();           // 入射介质 n = 1.2
    const double wavelength = 0.83;
    const int M = 6;
    struct Angle {
        double theta, phi;
    };
    double worst_k = 0.0, worst_angle = 0.0, worst_flux = 0.0, worst_de = 0.0;
    for (const Angle& a : {Angle{27.0, 63.0}, Angle{-27.0, 63.0}, Angle{40.0, 0.0}, Angle{40.0, 180.0}, Angle{0.0, 37.0},
                           Angle{0.0, -160.0}, Angle{66.36, -90.0}, Angle{12.0, 300.0}}) {
        const rcwa::ConicalResult r = conical(s, wavelength, a.theta, a.phi, M);
        const Kinematics k = kinematics(s, wavelength, a.theta, a.phi, M);
        bool orders_ok = true, branch_ok = true;
        worst_k = grow(worst_k, std::abs(r.ky - k.ky));
        for (int j = 0; j <= 2 * M; ++j) {
            orders_ok = orders_ok && r.orders[j] == j - M;
            worst_k = grow(worst_k, {std::abs(r.kx[j] - k.kx[j]), std::abs(r.kzI[j] - k.kzI[j]), std::abs(r.kzII[j] - k.kzII[j])});
            for (const cplx& kz : {r.kzI[j], r.kzII[j]}) branch_ok = branch_ok && kz.imag() >= 0.0 && (kz.imag() > 0.0 || kz.real() >= 0.0);
            worst_flux = grow(worst_flux, {std::abs(r.flux_r[j] - k.flux_r[j]), std::abs(r.flux_t[j] - k.flux_t[j])});
            const double angle = rad(r.phi_m_deg[j]);
            worst_angle = grow(worst_angle, {std::abs(std::cos(angle) - k.c[j]), std::abs(std::sin(angle) - k.s[j])});
            const std::array<cplx, 4> jr = elements(r.Jr[j]), jt = elements(r.Jt[j]);
            const std::array<double, 4> er = elements(r.DEr[j]), et = elements(r.DEt[j]);
            for (int e = 0; e < 4; ++e)
                worst_de = grow(worst_de, {std::abs(er[e] - k.flux_r[j] * std::norm(jr[e])), std::abs(et[e] - k.flux_t[j] * std::norm(jt[e]))});
        }
        CHECK(orders_ok);
        CHECK(branch_ok);
    }
    CHECK_LE(worst_k, 1e-12);
    CHECK_LE(worst_angle, 1e-12);
    CHECK_LE(worst_flux, 1e-12);
    CHECK_LE(worst_de, 1e-12);
    say("  最大偏差：波矢 %.1e，方位角的 cos/sin %.1e，通量因子 %.1e，效率 − 通量·|J|² %.1e\n", worst_k, worst_angle, worst_flux, worst_de);
}

// ---------------------------------------------------------------------------------------------
// [1] φ = 0 与 φ = 180°：ky = 0，问题退化为平面内入射，可以直接与 rcwa::solve 的 TE、TM 结果逐级比较。
//
// φ = 180° 时 kx_m = −nI·sinθ − m·λ/Λ，就是平面内入射角取 −θ。ky = 0 时 φ_m 是 0 或 π，c = σ_m = ±1（kx_m 的符号），
// s = 0，于是 ŝ = σ_m·ŷ：
//   · TE 的主场是 Ey：Es = σ_m·Ey。入射 Es = 1 即 Ey = σ_0，所以
//         Jr.ss = σ_m·σ_0·R_TE，   Jt.ss = σ_m·σ_0·T_TE；
//   · TM 的主场是 Hy：由 Z0·H = n·Ep·ŝ 得 Z0·Hy = n·σ_m·Ep。入射 Ep = 1 即 Z0·Hy = nI·σ_0，所以
//         Jr.pp = σ_m·σ_0·R_TM，   Jt.pp = σ_m·σ_0·(nI/nII)·T_TM；
//   · 交叉项为零；效率 DEr.ss、DEt.ss、DEr.pp、DEt.pp 分别等于 TE、TM 的 DEr、DEt（σ 和 nI/nII 在效率里消掉）。
// 两边解的是同一个截断问题（p 一支的本征问题一个用电场、一个用磁场写出，本征值相同），差别只有舍入：
// 复振幅容差 1e-9·max(1, |参照|)，效率 1e-10。交叉项在 φ = 0 时结构上是零，φ = 180° 时 sin 180° 的舍入
// 使 ky ≈ 1e-16，交叉项也只有这个量级，容差 1e-12。
static void test_planar_limit() {
    say("[1] phi = 0、180 度：对 TE/TM 求解器\n");
    struct Case {
        const char* label;
        rcwa::Stack stack;
        double wavelength, theta;
        int M;
    };
    const Case cases[] = {
        {"无吸收光栅", binary_grating(), 1.053, 66.36, 12},
        {"无吸收光栅，负入射角", binary_grating(), 1.053, -35.0, 12},
        {"吸收光栅", lossy_grating(), 1.053, 42.0, 14},
        {"双光栅夹吸收膜，nI = 1.2", double_grating(), 0.83, 27.0, 12},
        {"不对称光栅 + 膜", skew_grating(), 0.7, -20.0, 10},
        {"光栅 + 20 层膜", coated_grating(), 1.053, 66.36, 10},
    };
    for (const Case& c : cases)
        for (double phi : {0.0, 180.0}) {
            const double plane_theta = phi == 0.0 ? c.theta : -c.theta;
            const int M = c.M;
            const rcwa::Result te = rcwa::solve(c.stack, c.wavelength, plane_theta, M, Polarization::TE);
            const rcwa::Result tm = rcwa::solve(c.stack, c.wavelength, plane_theta, M, Polarization::TM);
            const rcwa::ConicalResult co = conical(c.stack, c.wavelength, c.theta, phi, M);
            const Kinematics k = kinematics(c.stack, c.wavelength, c.theta, phi, M);
            const double ratio = c.stack.nI.real() / c.stack.nII.real();
            const rcwa::ConicalPolarized s_in = polarized_by_angles(co, 90.0), p_in = polarized_by_angles(co, 0.0);
            double amplitude = 0.0, cross = 0.0, efficiency = 0.0, by_angles = 0.0;
            for (int j = 0; j <= 2 * M; ++j) {
                const double sigma = (k.c[j] < 0.0 ? -1.0 : 1.0) * (k.c[M] < 0.0 ? -1.0 : 1.0);
                const rcwa::Jones &jr = co.Jr[j], &jt = co.Jt[j];
                amplitude = grow(amplitude, {std::abs(jr.ss - sigma * te.R[j]) / (std::max)(1.0, std::abs(te.R[j])),
                                             std::abs(jt.ss - sigma * te.T[j]) / (std::max)(1.0, std::abs(te.T[j])),
                                             std::abs(jr.pp - sigma * tm.R[j]) / (std::max)(1.0, std::abs(tm.R[j])),
                                             std::abs(jt.pp - sigma * ratio * tm.T[j]) / (std::max)(1.0, std::abs(tm.T[j]))});
                cross = grow(cross, {std::abs(jr.sp), std::abs(jr.ps), std::abs(jt.sp), std::abs(jt.ps)});
                efficiency = grow(efficiency, {std::abs(co.DEr[j].ss - te.DEr[j]), std::abs(co.DEt[j].ss - te.DEt[j]),
                                               std::abs(co.DEr[j].pp - tm.DEr[j]), std::abs(co.DEt[j].pp - tm.DEt[j]),
                                               co.DEr[j].sp, co.DEr[j].ps, co.DEt[j].sp, co.DEt[j].ps});
                by_angles = grow(by_angles, {std::abs(s_in.DEr[j] - te.DEr[j]), std::abs(s_in.DEt[j] - te.DEt[j]),
                                             std::abs(p_in.DEr[j] - tm.DEr[j]), std::abs(p_in.DEt[j] - tm.DEt[j])});
            }
            CHECK_LE(amplitude, 1e-9);
            CHECK_LE(cross, 1e-12);
            CHECK_LE(efficiency, 1e-10);
            CHECK_LE(by_angles, 1e-10);       // ψ = 90° 是 s，ψ = 0° 是 p
            say("  %-26s phi = %3.0f  M = %2d  振幅 %.1e  交叉项 %.1e  效率 %.1e\n", c.label, phi, M, amplitude, cross, efficiency);
        }
}

// ---------------------------------------------------------------------------------------------
// [2] 全均匀的多层膜，ky ≠ 0：零级的 s、p 复振幅对特征矩阵法，交叉项与其余各级为零。
// 这是 ky ≠ 0 时唯一有闭式真值的情形。逐级次的 (s, p) 基正好是膜系的本征偏振，所以 Jones 矩阵是对角的：
//     Jr.ss = rs，  Jt.ss = ts，  Jr.pp = rp，  Jt.pp = tp        （film_reference 的四个数）。
// 默认的快速路径下两边都是闭式的标量运算，容差 1e-12（20 层膜的舍入累积也在这之内），其余各级 1e-14；
// homogeneous_fast_path = false 时每层都按一般层做本征分解（两支的谱完全重合），容差 1e-9。
// 恰好正入射、或入射面恰好沿栅线（φ = ±90°）时 kx_m = −kx_{−m}，均匀层的层矩阵里 ±m 两级的本征值相等。默认的本征求解器
// 对这样成对重合的本征值给出两个线性无关的本征向量（见 rcwa1d.hpp 里关于恰好正入射的说明），所以稠密路径在这两种入射下
// 照原样核对；另外再把重合错开算一遍（前者换成 θ = 1e-3°、后者的 φ 挪开 2°，解析解按同样的 θ 计算，它与 φ 无关），
// 两种情形用同一个容差。
static void test_film_stacks() {
    say("[2] 全均匀多层膜，ky != 0：对特征矩阵法\n");
    struct Case {
        const char* label;
        double nI, nII;
        std::vector<Film> films;
        double period, wavelength, theta, phi;
        bool lossless;
    };
    const std::vector<Film> three{{0.1, 2.3}, {0.2, 1.38}, {0.05, cplx(1.7, 0.3)}};
    const Case cases[] = {
        {"三层膜（一层吸收）", 1.0, 1.52, three, 1.0, 0.633, 50.0, 33.0, false},
        {"同上，负入射角", 1.0, 1.52, three, 1.0, 0.633, -50.0, 33.0, false},
        {"同上，phi = 200", 1.0, 1.52, three, 1.0, 0.633, 20.0, 200.0, false},
        {"同上，phi = -90（入射面沿栅线）", 1.0, 1.52, three, 1.0, 0.633, 75.0, -90.0, false},
        {"同上，正入射", 1.0, 1.52, three, 1.0, 0.633, 0.0, 37.0, false},
        {"入射介质 n = 1.33", 1.33, 2.1, {{0.21, 1.9}, {0.33, cplx(1.38, 0.01)}}, 0.7, 0.8, 40.0, 123.0, false},
        {"单层无吸收膜", 1.0, 1.45, {{0.62, 1.72}}, 0.8, 1.053, 41.0, 23.0, true},
        {"没有膜（单个界面）", 1.0, 1.5, {}, 0.8, 0.9, 56.0, 77.0, true},
        {"20 层介质膜", 1.0, 1.46, dielectric_mirror(), 0.574, 1.053, 66.36, 5.0, true},
        {"受抑全反射（空气隙）", 1.5, 1.5, {{0.2, 1.0}}, 0.9, 0.633, 50.0, 40.0, true},
        {"出射介质里全反射", 1.5, 1.0, {{0.15, 2.0}}, 0.9, 0.633, 60.0, -25.0, true},
    };
    const int M = 3;
    double worst[2] = {0.0, 0.0}, worst_others[2] = {0.0, 0.0};
    for (const Case& c : cases) {
        const rcwa::Stack s = film_stack(c.nI, c.films, c.nII, c.period);
        // 三种算法：快速路径；稠密路径；稠密路径并把成对重合的本征值错开（只有恰好正入射、φ = ±90° 的两例有这一种）
        for (int variant = 0; variant < 3; ++variant) {
            const bool fast = variant == 0, nudged = variant == 2;
            rcwa::Options opt;
            opt.homogeneous_fast_path = fast;
            const double theta = (nudged && c.theta == 0.0) ? 1e-3 : c.theta;
            const double phi = (nudged && std::abs(std::cos(rad(c.phi))) < 1e-6) ? c.phi + 2.0 : c.phi;
            if (nudged && theta == c.theta && phi == c.phi) continue;
            const FilmReference ref = film_reference(c.nI, c.films, c.nII, c.wavelength, theta);
            const rcwa::Jones jr_ref{ref.rs, 0.0, 0.0, ref.rp}, jt_ref{ref.ts, 0.0, 0.0, ref.tp};
            const rcwa::ConicalResult r = conical(s, c.wavelength, theta, phi, M, opt);
            const Kinematics k = kinematics(s, c.wavelength, theta, phi, M);
            const double tol = fast ? 1e-12 : 1e-9, tol_others = fast ? 1e-14 : 1e-9;
            const double zero_order = grow(gap(r.Jr[M], jr_ref), gap(r.Jt[M], jt_ref));
            double others = 0.0;
            for (int j = 0; j <= 2 * M; ++j)
                if (j != M) others = grow(others, {magnitude(r.Jr[j]), magnitude(r.Jt[j])});
            CHECK_LE(zero_order, tol);
            CHECK_LE(others, tol_others);
            const double efficiency =
                grow(0.0, {std::abs(r.DEr[M].ss - std::norm(ref.rs)), std::abs(r.DEr[M].pp - std::norm(ref.rp)),
                           std::abs(r.DEt[M].ss - k.flux_t[M] * std::norm(ref.ts)), std::abs(r.DEt[M].pp - k.flux_t[M] * std::norm(ref.tp))});
            CHECK_LE(efficiency, tol);
            if (c.lossless) CHECK_LE(unitarity_defect(r, k), tol);
            // 圆偏振入射 [Es, Ep] = [i, 1]/√2（ψ = 45°，δ = 90°）：两个分量各走各的系数
            {
                const rcwa::ConicalPolarized p = polarized_by_angles(r, 45.0, 90.0);
                const double h = std::sqrt(0.5);
                CHECK_LE(grow(0.0, {std::abs(p.Es - kI * h), std::abs(p.Ep - h)}), 1e-15);
                CHECK_LE(grow(0.0, {std::abs(p.Rs[M] - kI * h * ref.rs), std::abs(p.Rp[M] - h * ref.rp),
                                    std::abs(p.Ts[M] - kI * h * ref.ts), std::abs(p.Tp[M] - h * ref.tp),
                                    std::abs(p.DEr[M] - 0.5 * (std::norm(ref.rs) + std::norm(ref.rp))),
                                    std::abs(p.DEt[M] - 0.5 * k.flux_t[M] * (std::norm(ref.ts) + std::norm(ref.tp)))}),
                         tol);
            }
            worst[fast ? 0 : 1] = grow(worst[fast ? 0 : 1], zero_order);
            worst_others[fast ? 0 : 1] = grow(worst_others[fast ? 0 : 1], others);
        }
    }
    say("  零级的最大偏差：快速路径 %.1e，稠密路径 %.1e；其余各级的最大振幅：%.1e，%.1e\n", worst[0], worst[1], worst_others[0],
        worst_others[1]);
}

// ---------------------------------------------------------------------------------------------
// [3] θ = 0 时任意 φ：ky = 0，物理上只有一个正入射问题，φ 只决定入射的 (s, p) 基朝哪里摆，
// 结果可以由 θ = 0 的 TE、TM 解闭式给出。记 c = cos φ、s = sin φ。
//
// 入射（下行，kz = nI）：ŝ_0 = (−s, c)，t̂_0 = (c, s)，Et = Ep，所以 Ex = −s·Es + c·Ep，Ey = c·Es + s·Ep。
// 直角分量各自的响应（ky = 0 时 x、y 互不耦合）：
//   · Ey：就是 TE 的主场，第 m 级 Ey = R_TE·Ey入（反射）、T_TE·Ey入（透射）；
//   · Ex：由 TM 的主场 Hy 换算。ky = 0 的平面波 Ex = ±(kz/n²)·Z0·Hy（下行取正、上行取负），于是
//         Ex(反射, m) = −(kzI_m/nI)·R_TM·Ex入，     Ex(透射, m) = (nI·kzII_m/nII²)·T_TM·Ex入。
// 再投影到第 m 级自己的基上（Es = −s_m·Ex + c_m·Ey，Et = c_m·Ex + s_m·Ey，反射 Ep = −Et·nI/kzI_m，透射 Ep = Et·nII/kzII_m）：
//   · 零级（kx = ky = 0，φ_0 取入射的 φ，kz = n）。记 rx = −R_TM、ry = R_TE、tx = (nI/nII)·T_TM、ty = T_TE：
//         Jr = [ s²·rx + c²·ry      s·c·(ry − rx) ]        Jt = [ s²·tx + c²·ty     s·c·(ty − tx) ]
//              [ s·c·(rx − ry)     −(c²·rx + s²·ry) ]           [ s·c·(ty − tx)     c²·tx + s²·ty ]
//   · m ≠ 0（kx_m = −m·λ/Λ，φ_m 是 0 或 π，σ_m = −sign(m)）：
//         Jr = σ_m·[ c·R_TE    s·R_TE ]                    Jt = σ_m·[ c·T_TE               s·T_TE             ]
//                  [ −s·R_TM   c·R_TM ]                             [ −s·(nI/nII)·T_TM     c·(nI/nII)·T_TM    ]
// 同一个截断问题的两种解法，容差同第 [1] 组。另外从效率上看：入射电场沿 x（[Es, Ep] = [−s, c]）时各级效率就是
// TM 的，沿 y（[c, s]）时就是 TE 的，与 φ 无关。
static void test_normal_incidence() {
    say("[3] theta = 0、任意 phi：基的旋转，对正入射的 TE/TM 结果\n");
    struct Case {
        const char* label;
        rcwa::Stack stack;
        double wavelength;
        int M;
    };
    const Case cases[] = {
        {"无吸收光栅", binary_grating(), 1.053, 8},
        {"双光栅夹吸收膜，nI = 1.2", double_grating(), 0.83, 10},
        {"不对称光栅 + 膜", skew_grating(), 0.7, 8},
    };
    for (const Case& cs : cases) {
        const int M = cs.M;
        const rcwa::Result te = rcwa::solve(cs.stack, cs.wavelength, 0.0, M, Polarization::TE);
        const rcwa::Result tm = rcwa::solve(cs.stack, cs.wavelength, 0.0, M, Polarization::TM);
        const double ratio = cs.stack.nI.real() / cs.stack.nII.real();
        double worst = 0.0, worst_eff = 0.0;
        for (double phi : {0.0, 37.0, 90.0, 123.0, 180.0, -37.0, 200.0, 270.0}) {
            const rcwa::ConicalResult co = conical(cs.stack, cs.wavelength, 0.0, phi, M);
            const double c = std::cos(rad(phi)), s = std::sin(rad(phi));
            double amplitude = 0.0;
            for (int j = 0; j <= 2 * M; ++j) {
                rcwa::Jones jr, jt;
                if (j == M) {
                    const cplx rx = -tm.R[j], ry = te.R[j], tx = ratio * tm.T[j], ty = te.T[j];
                    jr = rcwa::Jones{s * s * rx + c * c * ry, s * c * (ry - rx), s * c * (rx - ry), -(c * c * rx + s * s * ry)};
                    jt = rcwa::Jones{s * s * tx + c * c * ty, s * c * (ty - tx), s * c * (ty - tx), c * c * tx + s * s * ty};
                } else {
                    const double sigma = j < M ? 1.0 : -1.0;
                    jr = rcwa::Jones{sigma * c * te.R[j], sigma * s * te.R[j], -sigma * s * tm.R[j], sigma * c * tm.R[j]};
                    jt = rcwa::Jones{sigma * c * te.T[j], sigma * s * te.T[j], -sigma * s * ratio * tm.T[j], sigma * c * ratio * tm.T[j]};
                }
                amplitude = grow(amplitude, {scaled_gap(co.Jr[j], jr), scaled_gap(co.Jt[j], jt)});
            }
            CHECK_LE(amplitude, 1e-9);
            worst = grow(worst, amplitude);
            // 入射电场沿 x、沿 y
            const rcwa::ConicalPolarized along_x = polarized(co, -s, c), along_y = polarized(co, c, s);
            double efficiency = 0.0;
            for (int j = 0; j <= 2 * M; ++j)
                efficiency = grow(efficiency, {std::abs(along_x.DEr[j] - tm.DEr[j]), std::abs(along_x.DEt[j] - tm.DEt[j]),
                                               std::abs(along_y.DEr[j] - te.DEr[j]), std::abs(along_y.DEt[j] - te.DEt[j])});
            CHECK_LE(efficiency, 1e-10);
            worst_eff = grow(worst_eff, efficiency);
        }
        say("  %-26s M = %2d  8 个 phi 的最大偏差：振幅 %.1e，效率 %.1e\n", cs.label, M, worst, worst_eff);
    }
}

// ---------------------------------------------------------------------------------------------
// [4] 无吸收结构的 2×2 通量幺正性：
//     Σ_m (flux_r·Jr†·Jr + flux_t·Jt†·Jt) = I。
// 对角元是 s 入射、p 入射各自的能量守恒，非对角元说的是两种入射产生的出射场按通量正交，合起来等价于“任意入射
// 偏振下 R + T = 1”，比单独一条能量守恒多三个实约束。折射率全为实数时 ⟦ε⟧、⟦1/ε⟧ 是厄米矩阵，截断之后的方程组
// 仍然保持能流，所以它在任何截断阶数下都成立到舍入误差：容差 1e-10（rcwa1d 的能量守恒用的也是这个量级）。
static void test_flux_unitarity() {
    say("[4] 无吸收结构的 2x2 通量幺正性\n");
    struct Case {
        const char* label;
        rcwa::Stack stack;
        double wavelength;
        int M;
    };
    const Case cases[] = {
        {"无吸收光栅", binary_grating(), 1.053, 10},
        {"不对称光栅 + 膜", skew_grating(), 0.7, 9},
        {"光栅 + 20 层膜", coated_grating(), 1.053, 10},
        {"强对比的多段光栅",
         make_stack(1.0, 1.0, 1.0,
                    {rcwa::Layer::grating(0.5, {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
                                          {1.0, 5.0, 1.0, 5.0, 5.0, 1.0, 5.0, 1.0, 5.0, 5.0})}),
         0.9, 12},
    };
    struct Angle {
        double theta, phi;
    };
    const Angle angles[] = {{40.0, 0.0}, {40.0, 1.0}, {40.0, 5.0}, {40.0, 30.0}, {40.0, 90.0}, {66.36, 17.0},
                            {-50.0, 137.0}, {20.0, -60.0}, {0.0, 45.0}, {75.0, 213.0}};
    struct Input {
        double psi, delta;
    };
    const Input inputs[] = {{90.0, 0.0}, {0.0, 0.0}, {45.0, 0.0}, {45.0, 90.0}, {30.0, -70.0}};
    for (const Case& c : cases) {
        double worst = 0.0, worst_member = 0.0, worst_energy = 0.0, worst_split = 0.0;
        for (const Angle& a : angles) {
            const rcwa::ConicalResult r = conical(c.stack, c.wavelength, a.theta, a.phi, c.M);
            const Kinematics k = kinematics(c.stack, c.wavelength, a.theta, a.phi, c.M);
            worst = grow(worst, unitarity_defect(r, k));
            // 结果自带的 flux_unitarity() 应当给出同一个矩阵
            const std::array<cplx, 4> u = r.flux_unitarity();
            worst_member = grow(worst_member, {std::abs(u[0] - 1.0), std::abs(u[1]), std::abs(u[2]), std::abs(u[3] - 1.0)});
            // 任意入射偏振下 R + T = 1，按输出偏振分开的效率加起来等于总效率
            for (const Input& in : inputs) {
                const rcwa::ConicalPolarized p = polarized_by_angles(r, in.psi, in.delta);
                worst_energy = grow(worst_energy, std::abs(p.reflectance() + p.transmittance() - 1.0));
                for (std::size_t j = 0; j < r.Jr.size(); ++j)
                    worst_split = grow(worst_split, {std::abs(p.DEr_s[j] + p.DEr_p[j] - p.DEr[j]), std::abs(p.DEt_s[j] + p.DEt_p[j] - p.DEt[j])});
            }
        }
        CHECK_LE(worst, 1e-10);
        CHECK_LE(worst_member, 1e-10);
        CHECK_LE(worst_energy, 1e-10);
        CHECK_LE(worst_split, 1e-14);
        say("  %-20s M = %2d  10 组角度：|U - I| %.1e（flux_unitarity() %.1e），5 种入射偏振的 |R + T - 1| %.1e\n", c.label, c.M, worst,
            worst_member, worst_energy);
    }

    // 有吸收时这个矩阵 U 仍是厄米的，两个本征值都落在 (0, 1) 之内：任何入射偏振都有一部分能量被吸收
    for (const rcwa::Stack& s : {lossy_grating(), double_grating()}) {
        const rcwa::ConicalResult r = conical(s, 0.9, 35.0, 28.0, 8);
        const std::array<cplx, 4> u = r.flux_unitarity();
        const double a = u[0].real(), d = u[3].real(), root = std::sqrt(0.25 * (a - d) * (a - d) + std::norm(u[1]));
        CHECK_LE(grow(0.0, {std::abs(u[0].imag()), std::abs(u[3].imag()), std::abs(u[1] - std::conj(u[2]))}), 1e-13);
        CHECK(0.5 * (a + d) + root < 1.0 - 1e-3 && 0.5 * (a + d) - root > 0.0);
    }
}

// ---------------------------------------------------------------------------------------------
// [5] 互易性。
//
// 第一次以 (θ, φ) 入射，看第 m 级反射；第二次把入射方向取为这一级的反方向：新的 (kx_0, ky) = (−kx_m, −ky)，即
// nI·sinθ' = kt_m，φ' = atan2(−ky, −kx_m)。第二次的第 m 级正好是第一次入射波的反方向（kx'_m = −kx_0）。
// Lorentz 互易定理 ∮(E1×H2 − E2×H1)·dS = 0 里只有“一次的入射波配另一次的同一条光路上的出射波”留下来，
// 一对相向的平面波贡献 −(2n/Z0)·(k̂·ẑ)·(E_a·E_b)。把四束波按文件头的 ŝ、p̂ 写开（反方向的那束 ŝ 反号、p̂ 相同）：
//     反射：  kz_m·Jr1[m] = kz_0·diag(1, −1)·Jr2[m]^T·diag(1, −1)        （ss ↔ ss，pp ↔ pp，sp ↔ −ps）
// kz_0 = nI·cosθ、kz_m = nI·cosθ' 是两次入射各自的 kz。透射同理，第二次要从出射介质一侧入射，这里把结构上下翻转
//（层序颠倒、nI 与 nII 互换）再用同一个求解器，nII·sinθ' = kt_m；z 的镜像使 p̂ 反号一次，结果没有负号：
//     透射：  kzII_m·Jt1[m] = kzI_0·Jt2[m]^T。
// 换成效率就是“通道 ab 的效率 = 反过来走时通道 ba 的效率”，与基的符号约定无关；有吸收时照样成立。
//
// m = 0 时两次求解保留的是同一组谐波（第二次的各级正好是第一次各级取反），截断之后的方程组互为转置，
// 等式成立到舍入误差：效率 1e-10，振幅 1e-9。m ≠ 0 时第二次保留的谐波相对第一次平移了 m 级，
// 两个截断问题不同，等式只在 M → ∞ 时成立：偏差不应超过两次求解各自的截断误差。截断误差用 M 与 2M 的结果之差估计，
// 按 1/M 这样最慢的收敛留出 2 倍（那时 |f(M) − f(∞)| = 2·|f(M) − f(2M)|），并且 M 加倍后偏差应当减小。
struct Direction {
    double theta, phi;      // 度
};
static Direction reversed(double kx_m, double ky, double n) {
    return {deg(std::asin(std::hypot(kx_m, ky) / n)), deg(std::atan2(-ky, -kx_m))};
}
// kz_out·J1 对 kz_in·(J2 的转置，sign = −1 时交叉元再变号) 的差
static double reciprocity_gap(const rcwa::Jones& j1, double kz_out, const rcwa::Jones& j2, double kz_in, double sign) {
    const rcwa::Jones t = transposed(j2);
    return gap(times(j1, kz_out), rcwa::Jones{kz_in * t.ss, sign * kz_in * t.sp, sign * kz_in * t.ps, kz_in * t.pp});
}

static void test_reciprocity() {
    say("[5] 互易性\n");
    struct Case {
        const char* label;
        rcwa::Stack stack;
        double wavelength, theta, phi;
        int M;
    };
    // (a) 零级：反射与透射，效率式与振幅式
    const Case cases[] = {
        {"无吸收光栅", binary_grating(), 1.053, 30.0, 25.0, 8},
        {"吸收光栅", lossy_grating(), 1.053, 42.0, -40.0, 10},
        {"双光栅夹吸收膜，nI = 1.2", double_grating(), 0.83, 27.0, 63.0, 8},
        {"不对称光栅 + 膜", skew_grating(), 0.7, 35.0, 28.0, 8},
        {"光栅 + 20 层膜", coated_grating(), 1.053, 66.36, 17.0, 8},
    };
    for (const Case& c : cases) {
        const int M = c.M;
        const double nI = c.stack.nI.real(), nII = c.stack.nII.real();
        const rcwa::ConicalResult a = conical(c.stack, c.wavelength, c.theta, c.phi, M);
        const Kinematics k = kinematics(c.stack, c.wavelength, c.theta, c.phi, M);
        // 反射：同一个结构，入射方向取零级反射的反方向（θ 不变，φ 转 180°）
        const Direction back = reversed(k.kx[M], k.ky, nI);
        const rcwa::ConicalResult b = conical(c.stack, c.wavelength, back.theta, back.phi, M);
        const double r_eff = gap(a.DEr[M], transposed(b.DEr[M]));
        const double r_amp = reciprocity_gap(a.Jr[M], k.kzI[M].real(), b.Jr[M], k.kzI[M].real(), -1.0);
        CHECK_LE(r_eff, 1e-10);
        CHECK_LE(r_amp, 1e-9);
        // 透射：结构上下翻转，从原来的出射介质一侧沿零级透射的反方向入射
        const Direction up = reversed(k.kx[M], k.ky, nII);
        const rcwa::ConicalResult t = conical(flipped(c.stack), c.wavelength, up.theta, up.phi, M);
        const double t_eff = gap(a.DEt[M], transposed(t.DEt[M]));
        const double t_amp = reciprocity_gap(a.Jt[M], k.kzII[M].real(), t.Jt[M], k.kzI[M].real(), 1.0);
        CHECK_LE(t_eff, 1e-10);
        CHECK_LE(t_amp, 1e-9);
        // 这一条不是空话：交叉项确实不小
        CHECK(magnitude(rcwa::Jones{0.0, a.Jr[M].sp, a.Jr[M].ps, 0.0}) > 1e-3 && magnitude(rcwa::Jones{0.0, a.Jt[M].sp, a.Jt[M].ps, 0.0}) > 1e-3);
        say("  %-26s 零级反射：效率 %.1e，振幅 %.1e；零级透射：效率 %.1e，振幅 %.1e\n", c.label, r_eff, r_amp, t_eff, t_amp);
    }
    // 左右不对称的光栅，单独一次求解里 sp 与 ps 两个通道的效率并不相等，上面“sp ↔ ps”的互换因此是实质性的
    {
        const rcwa::ConicalResult a = conical(skew_grating(), 0.7, 35.0, 28.0, 8);
        CHECK(std::abs(a.DEr[8].sp - a.DEr[8].ps) > 1e-4 || std::abs(a.DEt[8].sp - a.DEt[8].ps) > 1e-4);
    }

    // (b) 非零的传播级：偏差不超过截断误差，并随 M 减小
    struct Higher {
        const char* label;
        rcwa::Stack stack;
        double wavelength, theta, phi;
        int m;
    };
    const Higher higher[] = {
        {"无吸收光栅", binary_grating(), 1.053, 66.36, 17.0, 1},
        {"吸收光栅", lossy_grating(), 1.053, 42.0, -40.0, 1},
        {"不对称光栅 + 膜", skew_grating(), 0.7, 35.0, 28.0, 1},
        {"不对称光栅 + 膜", skew_grating(), 0.7, 35.0, 28.0, 2},
    };
    for (const Higher& c : higher) {
        const int M = 8, m = c.m;
        const double nI = c.stack.nI.real();
        const Kinematics k = kinematics(c.stack, c.wavelength, c.theta, c.phi, M);
        CHECK(k.kzI2[m + M] > 1e-3);                                  // 这一级在入射介质里是传播的
        const Direction back = reversed(k.kx[m + M], k.ky, nI);
        const double kz_0 = k.kzI[M].real(), kz_m = k.kzI[m + M].real();
        const rcwa::ConicalResult a1 = conical(c.stack, c.wavelength, c.theta, c.phi, M);
        const rcwa::ConicalResult a2 = conical(c.stack, c.wavelength, c.theta, c.phi, 2 * M);
        const rcwa::ConicalResult b1 = conical(c.stack, c.wavelength, back.theta, back.phi, M);
        const rcwa::ConicalResult b2 = conical(c.stack, c.wavelength, back.theta, back.phi, 2 * M);
        const double eff1 = gap(a1.DEr[m + M], transposed(b1.DEr[m + M]));
        const double eff2 = gap(a2.DEr[m + 2 * M], transposed(b2.DEr[m + 2 * M]));
        const double amp1 = reciprocity_gap(a1.Jr[m + M], kz_m, b1.Jr[m + M], kz_0, -1.0);
        const double amp2 = reciprocity_gap(a2.Jr[m + 2 * M], kz_m, b2.Jr[m + 2 * M], kz_0, -1.0);
        const double eff_truncation = gap(a1.DEr[m + M], a2.DEr[m + 2 * M]) + gap(b1.DEr[m + M], b2.DEr[m + 2 * M]);
        const double amp_truncation = kz_m * gap(a1.Jr[m + M], a2.Jr[m + 2 * M]) + kz_0 * gap(b1.Jr[m + M], b2.Jr[m + 2 * M]);
        CHECK_LE(eff1, 2.0 * eff_truncation + 1e-10);
        CHECK_LE(amp1, 2.0 * amp_truncation + 1e-10);
        CHECK_LE(eff2, eff1);
        CHECK_LE(amp2, amp1);
        CHECK(eff_truncation < 1e-3);                                 // 截断误差本身是小量，上面的界才有意义
        say("  %-18s 第 %+d 级反射：效率的偏差 %.1e (M = 8)、%.1e (M = 16)，截断误差约 %.1e\n", c.label, m, eff1, eff2, eff_truncation);
    }
}

// ---------------------------------------------------------------------------------------------
// [6] 镜像对称与几种等价的写法。都是截断之后仍然精确成立的关系（保留的谐波是同一组），两边只差舍入：
// 复振幅 1e-9·max(1, |J|)，效率 1e-10。
//
//   · φ → −φ（y → −y 的镜像，结构不变）：kx_m 不变、ky 反号，c 不变、s 反号。镜像把 (Ex, Ey) 变成 (Ex, −Ey)，
//     于是 Es → −Es、Et 不变：J(−φ) = diag(−1, 1)·J(φ)·diag(−1, 1)。同偏振的 ss、pp 是 φ 的偶函数，
//     交叉的 sp、ps 是奇函数，四个通道的效率都是偶函数；
//   · (θ, φ + 360°)、(−θ, φ + 180°) 与 (θ, φ) 是同一个入射方向（kx_0、ky 都不变），基也相同：J 完全相同
//     （后一种写法 θ = 0 时除外：那时零级的基取决于 φ 本身，φ 加 180° 把入射的 ŝ、t̂ 都反了号，所以正入射的一例不比它）；
//   · x → −x 的镜像：结构左右翻转，φ → 180° − φ，第 m 级与第 −m 级互换，同样是 s 分量反号：
//         J'[−m] = diag(−1, 1)·J[m]·diag(−1, 1)；
//   · 结构沿 x 平移 x0：J'[m] = J[m]·exp(−i·2π·m·x0/Λ)（入射波在原点的相位不变）。
static void test_mirror_symmetries() {
    say("[6] 镜像对称与等价的写法\n");
    struct Case {
        const char* label;
        rcwa::Stack stack;
        double wavelength, theta, phi;
        int M;
    };
    const Case cases[] = {
        {"无吸收光栅", binary_grating(), 1.053, 40.0, 7.0, 9},
        {"吸收光栅", lossy_grating(), 1.053, 42.0, -40.0, 9},
        {"双光栅夹吸收膜，nI = 1.2", double_grating(), 0.83, 27.0, 63.0, 9},
        {"不对称光栅 + 膜", skew_grating(), 0.7, 35.0, 118.0, 9},
        {"不对称光栅，正入射", skew_grating(), 0.7, 0.0, 28.0, 9},
    };
    for (const Case& c : cases) {
        const int M = c.M;
        const rcwa::ConicalResult a = conical(c.stack, c.wavelength, c.theta, c.phi, M);
        const rcwa::ConicalResult y_mirror = conical(c.stack, c.wavelength, c.theta, -c.phi, M);
        const rcwa::ConicalResult turn = conical(c.stack, c.wavelength, c.theta, c.phi + 360.0, M);
        const rcwa::ConicalResult same = c.theta != 0.0 ? conical(c.stack, c.wavelength, -c.theta, c.phi + 180.0, M) : turn;
        const rcwa::ConicalResult x_mirror = conical(mirrored(c.stack), c.wavelength, c.theta, 180.0 - c.phi, M);
        const double x0 = 0.37 * c.stack.period;
        const rcwa::ConicalResult moved = conical(shifted(c.stack, x0), c.wavelength, c.theta, c.phi, M);
        double even = 0.0, even_eff = 0.0, direction = 0.0, mirror = 0.0, shift = 0.0, cross = 0.0;
        for (int j = 0; j <= 2 * M; ++j) {
            const int m = j - M;
            even = grow(even, {scaled_gap(y_mirror.Jr[j], s_flipped(a.Jr[j])), scaled_gap(y_mirror.Jt[j], s_flipped(a.Jt[j]))});
            even_eff = grow(even_eff, {gap(y_mirror.DEr[j], a.DEr[j]), gap(y_mirror.DEt[j], a.DEt[j])});
            direction = grow(direction, {scaled_gap(same.Jr[j], a.Jr[j]), scaled_gap(same.Jt[j], a.Jt[j]),
                                         scaled_gap(turn.Jr[j], a.Jr[j]), scaled_gap(turn.Jt[j], a.Jt[j])});
            mirror = grow(mirror, {scaled_gap(x_mirror.Jr[2 * M - j], s_flipped(a.Jr[j])), scaled_gap(x_mirror.Jt[2 * M - j], s_flipped(a.Jt[j]))});
            const cplx phase = std::exp(cplx(0.0, -2.0 * kPi * m * x0 / c.stack.period));
            shift = grow(shift, {scaled_gap(moved.Jr[j], times(a.Jr[j], phase)), scaled_gap(moved.Jt[j], times(a.Jt[j], phase))});
            cross = grow(cross, {std::abs(a.Jr[j].sp), std::abs(a.Jr[j].ps)});
        }
        CHECK_LE(even, 1e-9);
        CHECK_LE(even_eff, 1e-10);
        CHECK_LE(direction, 1e-9);
        CHECK_LE(mirror, 1e-9);
        CHECK_LE(shift, 1e-9);
        CHECK(cross > 1e-3);          // 交叉振幅本身不小，“奇函数”才有内容
        say("  %-26s phi -> -phi %.1e（效率 %.1e），同一方向 %.1e，x 镜像 %.1e，平移 %.1e\n", c.label, even, even_eff, direction, mirror,
            shift);
    }
}

// ---------------------------------------------------------------------------------------------
// [7] 超胞等价性（φ ≠ 0）：把周期重复 3 次、写成周期 3Λ 的结构，它的第 3m 级就是单胞的第 m 级
//（kx 相同：3m·λ/(3Λ) = m·λ/Λ），其余各级没有被激发。
// 超胞剖面的 Fourier 系数只在 3 的倍数上不为零，卷积矩阵（连同它们的逆）按级次模 3 分成互不耦合的三块，
// 含零级的那一块与单胞取 M 时的方程组逐元素相同（超胞取 M' = 3M、3M + 1、3M + 2 都是）。所以截断之后仍然精确成立，
// 只差舍入：对应级 1e-9·max(1, |J|)；其余各级由 Fourier 系数里 1e-17 量级的舍入激发，容差 1e-10。
static void test_supercell() {
    say("[7] 超胞等价性（phi != 0）：周期重复 3 次\n");
    struct Case {
        const char* label;
        rcwa::Stack stack;
        double wavelength, theta, phi;
        int M;
    };
    const Case cases[] = {
        {"无吸收光栅", binary_grating(), 1.053, 32.0, 14.0, 6},
        {"不对称光栅 + 膜", skew_grating(), 0.7, 35.0, 28.0, 5},
        {"双光栅夹吸收膜，nI = 1.2", double_grating(), 0.83, 27.0, 63.0, 5},
    };
    for (const Case& c : cases) {
        const int M = c.M;
        const rcwa::ConicalResult unit = conical(c.stack, c.wavelength, c.theta, c.phi, M);
        const rcwa::Stack cell3 = tripled(c.stack);
        for (int M3 : {3 * M, 3 * M + 1}) {
            const rcwa::ConicalResult r = conical(cell3, c.wavelength, c.theta, c.phi, M3);
            double matched = 0.0, others = 0.0, kx = 0.0;
            for (int j = 0; j <= 2 * M3; ++j) {
                const int m3 = j - M3;
                if (m3 % 3 == 0 && std::abs(m3 / 3) <= M) {
                    const int i = m3 / 3 + M;
                    matched = grow(matched, {scaled_gap(r.Jr[j], unit.Jr[i]), scaled_gap(r.Jt[j], unit.Jt[i])});
                    kx = grow(kx, std::abs(r.kx[j] - unit.kx[i]));
                } else {
                    others = grow(others, {magnitude(r.Jr[j]), magnitude(r.Jt[j])});
                }
            }
            CHECK_LE(matched, 1e-9);
            CHECK_LE(others, 1e-10);
            CHECK_LE(kx, 1e-12);
            say("  %-26s M = %d，超胞 M = %2d：对应级 %.1e，其余各级的最大振幅 %.1e\n", c.label, M, M3, matched, others);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// [8] 小方位角的连续性。
// 由第 [6] 组，交叉振幅 Jsp、Jps 是 φ 的奇函数，同偏振的 Jss、Jpp 是偶函数。解对 ky 解析；逐级次的基
// c = kx_m/kt、s = ky/kt 在 |ky| < |kx_m| 之内解析，所以展开的小参数是 ε = max_m |ky/kx_m|（各级衍射面转过的最大角度）：
//     交叉振幅 = a·φ·(1 + O(ε²))，   交叉偏振效率 = |a|²·φ²·(1 + O(ε²))，   同偏振振幅 − (φ = 0 的值) = b·φ²·(1 + O(ε²))。
// φ 取 0.05°、0.1°、0.2°、0.4°。相邻两个 φ 之间，效率之比是 4·(1 + K·ε²)，交叉振幅之比是 2·(1 + K·ε²)，同偏振的偏离
// 之比是 4·(1 + K·ε²)，ε 取较大的那个 φ 处的值，K 是量级为 1 的系数：容差取 10·ε²。指数若不是 2，效率之比会是 2 或 8。
// 第二个结构里 +1 级的 kx 只有 −0.063，它的衍射面随 φ 转得很快（ε 约为 φ 的 9 倍），用来确认小参数确实是 ε 而不是 φ。
// 再往下到 1e-4°、1e-6°：交叉振幅仍按同一个斜率线性地趋于零——既没有淹没在舍入里，也没有被某个 1/ky 放大。
// 容差是 10·ε²（ε 取 0.05° 处的值）加上舍入的份额：交叉项的绝对精度按第 [1] 组取 1e-12，除以振幅本身。
// 到 1e-9° 交叉振幅只剩 1e-11 量级，只要求它不超过线性外推；这几个极小的 φ 下同偏振的振幅都回到 TE/TM 的结果。
static void test_small_azimuth() {
    say("[8] 小方位角：交叉偏振效率 ~ phi^2\n");
    struct Case {
        const char* label;
        rcwa::Stack stack;
        double wavelength, theta;
        int M;
    };
    const Case cases[] = {
        {"无吸收光栅", binary_grating(), 1.053, 40.0, 10},
        {"不对称光栅 + 膜", skew_grating(), 0.7, 35.0, 9},
    };
    const double phis[4] = {0.05, 0.1, 0.2, 0.4};
    for (const Case& c : cases) {
        const int M = c.M;
        const rcwa::ConicalResult base = conical(c.stack, c.wavelength, c.theta, 0.0, M);
        std::vector<rcwa::ConicalResult> r;
        double cross_eff[4] = {}, cross_amp[4] = {}, eps2[4] = {};
        for (int i = 0; i < 4; ++i) {
            r.push_back(conical(c.stack, c.wavelength, c.theta, phis[i], M));
            const Kinematics k = kinematics(c.stack, c.wavelength, c.theta, phis[i], M);
            for (int j = 0; j <= 2 * M; ++j) {
                cross_eff[i] += r[i].DEr[j].ps + r[i].DEt[j].ps + r[i].DEr[j].sp + r[i].DEt[j].sp;
                cross_amp[i] = grow(cross_amp[i], {std::abs(r[i].Jr[j].sp), std::abs(r[i].Jr[j].ps), std::abs(r[i].Jt[j].sp), std::abs(r[i].Jt[j].ps)});
                eps2[i] = (std::max)(eps2[i], (k.ky / k.kx[j]) * (k.ky / k.kx[j]));
            }
        }
        CHECK(cross_eff[0] > 1e-9);           // 交叉偏振确实存在
        CHECK(eps2[3] < 1e-2);                // 小参数确实小
        double worst_k = 0.0;                 // 各项偏差除以 ε²，即上面说的系数 K
        for (int i = 0; i + 1 < 4; ++i) {
            const double tol = 10.0 * eps2[i + 1];
            const double ratio = std::abs(cross_eff[i + 1] / cross_eff[i] / 4.0 - 1.0);
            double linear = 0.0, even = 0.0, even_scale = 0.0;
            for (int j = 0; j <= 2 * M; ++j) {
                const rcwa::Jones &lo_r = r[i].Jr[j], &hi_r = r[i + 1].Jr[j], &lo_t = r[i].Jt[j], &hi_t = r[i + 1].Jt[j];
                const rcwa::Jones &b_r = base.Jr[j], &b_t = base.Jt[j];
                // 交叉振幅：J(2φ) = 2·J(φ)（复数，相位也要对上）
                linear = grow(linear, {std::abs(hi_r.sp - 2.0 * lo_r.sp), std::abs(hi_r.ps - 2.0 * lo_r.ps),
                                       std::abs(hi_t.sp - 2.0 * lo_t.sp), std::abs(hi_t.ps - 2.0 * lo_t.ps)});
                // 同偏振振幅相对 φ = 0 的偏离：Δ(2φ) = 4·Δ(φ)
                even = grow(even, {std::abs((hi_r.ss - b_r.ss) - 4.0 * (lo_r.ss - b_r.ss)), std::abs((hi_r.pp - b_r.pp) - 4.0 * (lo_r.pp - b_r.pp)),
                                   std::abs((hi_t.ss - b_t.ss) - 4.0 * (lo_t.ss - b_t.ss)), std::abs((hi_t.pp - b_t.pp) - 4.0 * (lo_t.pp - b_t.pp))});
                even_scale = grow(even_scale, {std::abs(hi_r.ss - b_r.ss), std::abs(hi_r.pp - b_r.pp), std::abs(hi_t.ss - b_t.ss), std::abs(hi_t.pp - b_t.pp)});
            }
            CHECK_LE(ratio, tol);
            CHECK_LE(linear / cross_amp[i + 1], tol);
            CHECK_LE(even / even_scale, tol);
            worst_k = grow(worst_k, {ratio / eps2[i + 1], linear / cross_amp[i + 1] / eps2[i + 1], even / even_scale / eps2[i + 1]});
        }
        // 对数坐标下最小二乘拟合的指数
        double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
        for (int i = 0; i < 4; ++i) {
            const double x = std::log(phis[i]), y = std::log(cross_eff[i]);
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
        const double exponent = (4.0 * sxy - sx * sy) / (4.0 * sxx - sx * sx);
        CHECK_LE(std::abs(exponent - 2.0), 10.0 * eps2[3]);

        // 极小的 φ
        const rcwa::Result te = rcwa::solve(c.stack, c.wavelength, c.theta, M, Polarization::TE);
        const rcwa::Result tm = rcwa::solve(c.stack, c.wavelength, c.theta, M, Polarization::TM);
        const Kinematics k = kinematics(c.stack, c.wavelength, c.theta, 0.0, M);
        const double slope = cross_amp[0] / rad(phis[0]), ratio = c.stack.nI.real() / c.stack.nII.real();
        double worst_slope = 0.0, worst_co = 0.0;
        for (double tiny : {1e-4, 1e-6, 1e-9}) {
            const rcwa::ConicalResult t = conical(c.stack, c.wavelength, c.theta, tiny, M);
            double amp = 0.0;
            for (int j = 0; j <= 2 * M; ++j) {
                amp = grow(amp, {std::abs(t.Jr[j].sp), std::abs(t.Jr[j].ps), std::abs(t.Jt[j].sp), std::abs(t.Jt[j].ps)});
                const double sigma = (k.c[j] < 0.0 ? -1.0 : 1.0) * (k.c[M] < 0.0 ? -1.0 : 1.0);
                worst_co = grow(worst_co, {std::abs(t.Jr[j].ss - sigma * te.R[j]) / (std::max)(1.0, std::abs(te.R[j])),
                                           std::abs(t.Jt[j].ss - sigma * te.T[j]) / (std::max)(1.0, std::abs(te.T[j])),
                                           std::abs(t.Jr[j].pp - sigma * tm.R[j]) / (std::max)(1.0, std::abs(tm.R[j])),
                                           std::abs(t.Jt[j].pp - sigma * ratio * tm.T[j]) / (std::max)(1.0, std::abs(tm.T[j]))});
            }
            const double expected = slope * rad(tiny);
            if (tiny > 1e-8) {
                CHECK_LE(std::abs(amp / expected - 1.0), 10.0 * eps2[0] + 1e-12 / expected);
                worst_slope = grow(worst_slope, std::abs(amp / expected - 1.0));
            } else {
                CHECK_LE(amp, 1.01 * expected + 1e-12);
            }
        }
        CHECK_LE(worst_co, 1e-9);
        say("  %-18s eps^2 = %.1e (0.4 度)；各比值的偏差 / eps^2 最大 %.2f，拟合指数 %.6f；\n", c.label, eps2[3], worst_k, exponent);
        say("  %-18s phi = 1e-4、1e-6 度时 |J交叉|/phi 的相对变化 %.1e，极小 phi 下同偏振对 TE/TM %.1e\n", "", worst_slope, worst_co);
    }
}

// ---------------------------------------------------------------------------------------------
// [9] 穿越 Rayleigh 截止：扫 θ 或 φ，让某一级在某个介质里的 kz 过零（传播 ↔ 倏逝）。
//
// 截止点两侧各取一点（相距 2δ，δ = 1e-4°、1e-6°、1e-8°，最近时 |kz| ≈ 2e-5；恰好 kz = 0 的那一点按约定可以
// 报告失败，这里不取）。要求：结果全部有限；无吸收时通量幺正性保持；没有跳变。“没有跳变”的定量说法：
//   · 截止发生在入射或出射介质里：解是这一级 kz 的解析函数，kz 在截止的一侧是实数 ε，另一侧是 i·ε，ε ∝ sqrt(δ)，
//     两侧各通道效率之差 = |f'·ε·(1 − i)|·(1 + O(ε)) ∝ sqrt(δ)（Wood 异常处的尖点：连续，斜率无穷大）。
//     δ 缩小 100 倍，差缩小 10 倍；ε ≤ 2e-3，O(ε) 的修正在 1% 以下，容差 10%。若某一级选错了分支，差不会趋于零；
//   · 截止发生在夹在中间的均匀膜里：膜的传递矩阵只含 cos(kz·d) 与 sin(kz·d)/kz，是 kz² 的解析函数，
//     所以这里根本没有尖点，两侧之差 ∝ δ。δ 缩小 100 倍，差至少缩小到 1.1/100（另加 1e-9 的舍入余量）。
// 通量幺正性的容差 1e-9：kz 很小时精度随 |kz| 缓慢下降（rcwa1d.hpp 文件头：|kz| ≈ 1e-7 时能量守恒仍到 1e-11），
// 圆锥入射多一步除以 kz 的 p 分量换算，再留两个量级。
// 截止角是闭式的：(nI·sinθ·cosφ − g)² + (nI·sinθ·sinφ)² = n²，g = m·λ/Λ，
//     扫 θ：nI·sinθ = g·cosφ ± sqrt(n² − g²·sin²φ)；     扫 φ：cosφ = (s² + g² − n²)/(2·s·g)，s = nI·sinθ。
static void test_rayleigh_crossing() {
    say("[9] 穿越 Rayleigh 截止\n");
    const double period = 1.3, wavelength = 1.0, nI = 1.0, nII = 1.45, n_film = 1.8;
    const int M = 8;
    const rcwa::Layer grating = rcwa::Layer::grating(0.22, {0.0, 0.37 * period, period}, {1.5, 1.0});
    const rcwa::Stack with_film = make_stack(nI, nII, period, {grating, rcwa::Layer::uniform(0.15, n_film)});
    const rcwa::Stack bare = make_stack(nI, nII, period, {grating});
    const auto theta_cut = [&](int m, double n, double phi, double sign) {
        const double g = m * wavelength / period, c = std::cos(rad(phi)), s = std::sin(rad(phi));
        return deg(std::asin((g * c + sign * std::sqrt(n * n - g * g * s * s)) / nI));
    };
    const auto phi_cut = [&](int m, double n, double theta) {
        const double g = m * wavelength / period, s = nI * std::sin(rad(theta));
        return deg(std::acos((s * s + g * g - n * n) / (2.0 * s * g)));
    };
    struct Crossing {
        const char* label;
        const rcwa::Stack* stack;
        bool scan_theta;        // true：固定 φ 扫 θ；false：固定 θ 扫 φ
        double fixed, critical;
        int m;
        int medium;             // 1：入射介质；2：出射介质；0：中间的均匀膜
        double n;
    };
    const Crossing crossings[] = {
        {"扫 theta，+1 级在入射介质里", &with_film, true, 21.0, theta_cut(1, nI, 21.0, -1.0), 1, 1, nI},
        {"扫 theta，+2 级在出射介质里", &with_film, true, 21.0, theta_cut(2, nII, 21.0, -1.0), 2, 2, nII},
        {"同上，光栅直接压在衬底上", &bare, true, 21.0, theta_cut(2, nII, 21.0, -1.0), 2, 2, nII},
        {"扫 phi，-1 级在入射介质里", &with_film, false, 20.0, phi_cut(-1, nI, 20.0), -1, 1, nI},
        {"扫 theta，+3 级在均匀膜里", &with_film, true, 21.0, theta_cut(3, n_film, 21.0, -1.0), 3, 0, n_film},
    };
    const double deltas[3] = {1e-4, 1e-6, 1e-8};
    for (const Crossing& c : crossings) {
        const int j = c.m + M;
        double jump[3] = {}, unitarity = 0.0;
        bool finite = true, straddles = true, flux_ok = true;
        for (int i = 0; i < 3; ++i) {
            rcwa::ConicalResult side[2];
            double kz2[2] = {};
            for (int w = 0; w < 2; ++w) {
                const double x = c.critical + (w == 0 ? -deltas[i] : deltas[i]);
                const double theta = c.scan_theta ? x : c.fixed, phi = c.scan_theta ? c.fixed : x;
                side[w] = conical(*c.stack, wavelength, theta, phi, M);
                const Kinematics k = kinematics(*c.stack, wavelength, theta, phi, M);
                kz2[w] = c.n * c.n - k.kx[j] * k.kx[j] - k.ky * k.ky;
                finite = finite && all_finite(side[w]);
                unitarity = grow(unitarity, unitarity_defect(side[w], k));
                // 这一级在倏逝的一侧不带走能量，在传播的一侧通量因子是正的
                if (c.medium != 0) {
                    const double flux = c.medium == 1 ? side[w].flux_r[j] : side[w].flux_t[j];
                    const rcwa::JonesEfficiency& de = c.medium == 1 ? side[w].DEr[j] : side[w].DEt[j];
                    flux_ok = flux_ok && (kz2[w] > 0.0 ? flux > 0.0 : (std::abs(flux) <= 1e-12 && gap(de, rcwa::JonesEfficiency{}) <= 1e-12));
                }
            }
            straddles = straddles && kz2[0] * kz2[1] < 0.0 && std::abs(kz2[0] + kz2[1]) < 1e-3 * std::abs(kz2[0] - kz2[1]);
            for (int o = 0; o <= 2 * M; ++o)
                jump[i] = grow(jump[i], {gap(side[0].DEr[o], side[1].DEr[o]), gap(side[0].DEt[o], side[1].DEt[o])});
        }
        CHECK(straddles);               // 两个点确实对称地分居截止的两侧（核对的是测试自己的截止角公式）
        CHECK(finite);
        CHECK(flux_ok);
        CHECK_LE(unitarity, 1e-9);
        for (int i = 1; i < 3; ++i) {
            if (c.medium != 0) CHECK_LE(std::abs(jump[i] / (0.1 * jump[i - 1]) - 1.0), 0.1);
            else CHECK_LE(jump[i], 1.1 * 0.01 * jump[i - 1] + 1e-9);
        }
        say("  %-28s 截止点 %.6f 度：两侧效率之差 %.2e、%.2e、%.2e（delta = 1e-4、1e-6、1e-8 度），|U - I| %.1e\n", c.label, c.critical,
            jump[0], jump[1], jump[2], unitarity);
    }
}

// ---------------------------------------------------------------------------------------------
// [10] 弱对比极限：齿与槽的折射率差 Δn 取 1e-1、1e-2、1e-3、1e-6、0，结果连续地趋于均匀膜系的解析解。
//
// 光栅层的齿（占空比 f）折射率 n_g + Δn、槽 n_g，下面一层均匀膜。对 Δε = (n_g + Δn)² − n_g² 作微扰：
//   · 零级的 Jones 矩阵 = “把光栅层换成 ε_eff = f·(n_g + Δn)² + (1 − f)·n_g² 的均匀层”的解析解 + O(Δε²)。
//     一阶项只通过 ε 的周期平均值（Fourier 系数 c_0）进来，已经含在 ε_eff 里；s、p 的等效介电常数（平均值与
//     调和平均值的倒数）彼此也只差 O(Δε²)；交叉项、经由其他级次再回到零级的贡献都从二阶开始；
//   · 其余各级的振幅 ∝ Δε（一阶 Born 近似，正比于 ε 的 Fourier 系数 c_m）。
// 所以零级对“ε_eff 膜系”的偏差 dev(Δn) = K·Δε²·(1 + O(Δε))，其余各级的最大振幅 = K1·Δε·(1 + O(Δε))：
//   · Δn = 1e-3 与 1e-2 之间按 Δε² 缩放，O(Δε) 的修正约 3%，容差 10%；Δn = 1e-1（Δε = 0.33，高阶项不小）只要求
//     dev 落在 K·Δε² 的 0.5 ~ 2 倍之内；
//   · Δn = 1e-6：K·Δε² ≈ 1e-12，要求 dev ≤ 1e-9——这时两支本征问题的谱只差 1e-6，是最容易丢精度的地方，
//     1e-9 是整个模块对 Jones 矩阵元的精度目标；其余各级的振幅仍然严格按 Δε 缩放（与 Δn = 1e-3 的比例差不超过 5%）；
//   · Δn = 0（各段折射率相同的“光栅层”，两支的谱完全重合）：与均匀膜系的解析解相差不超过 1e-9，其余各级不超过 1e-9；
//   · 每一个 Δn 下通量幺正性都保持（1e-10）。
static void test_weak_contrast() {
    say("[10] 弱对比极限：趋于均匀膜系的解析解\n");
    const double nI = 1.0, nII = 1.5, period = 0.9, wavelength = 0.8, theta = 38.0, phi = 29.0;
    const double n_groove = 1.6, fill = 0.4, depth = 0.27;
    const Film below{0.21, 1.9};
    const int M = 8;
    struct Outcome {
        double d_eps, dev, others, unitarity;
    };
    const auto solve_case = [&](double dn, const rcwa::Layer& top) {
        const rcwa::Stack s = make_stack(nI, nII, period, {top, rcwa::Layer::uniform(below.d, below.n)});
        const rcwa::ConicalResult r = conical(s, wavelength, theta, phi, M);
        const double eps_eff = fill * (n_groove + dn) * (n_groove + dn) + (1.0 - fill) * n_groove * n_groove;
        const FilmReference ref = film_reference(nI, {{depth, std::sqrt(eps_eff)}, below}, nII, wavelength, theta);
        Outcome o{};
        o.d_eps = (n_groove + dn) * (n_groove + dn) - n_groove * n_groove;
        o.dev = grow(gap(r.Jr[M], rcwa::Jones{ref.rs, 0.0, 0.0, ref.rp}), gap(r.Jt[M], rcwa::Jones{ref.ts, 0.0, 0.0, ref.tp}));
        for (int j = 0; j <= 2 * M; ++j)
            if (j != M) o.others = grow(o.others, {magnitude(r.Jr[j]), magnitude(r.Jt[j])});
        o.unitarity = unitarity_defect(r, kinematics(s, wavelength, theta, phi, M));
        return o;
    };
    const auto ridge = [&](double dn) {
        return rcwa::Layer::grating(depth, {0.0, fill * period, period}, {n_groove + dn, n_groove});
    };
    const Outcome o1 = solve_case(1e-1, ridge(1e-1)), o2 = solve_case(1e-2, ridge(1e-2)), o3 = solve_case(1e-3, ridge(1e-3));
    const Outcome o6 = solve_case(1e-6, ridge(1e-6));
    const Outcome o0 = solve_case(0.0, ridge(0.0));
    const Outcome one_segment = solve_case(0.0, rcwa::Layer::grating(depth, {0.1, 0.1 + period}, {n_groove}));     // 只有一段的“光栅层”
    for (const Outcome* o : {&o1, &o2, &o3, &o6, &o0, &one_segment}) CHECK_LE(o->unitarity, 1e-10);

    const double k2 = o3.dev / (o3.d_eps * o3.d_eps), k1 = o3.others / o3.d_eps;
    CHECK(o3.dev > 1e-9 && o3.others > 1e-6);                                   // 二阶、一阶的效应在 Δn = 1e-3 时都远在舍入之上
    CHECK_LE(std::abs(o2.dev / (o2.d_eps * o2.d_eps) / k2 - 1.0), 0.1);         // dev ∝ Δε²
    CHECK(o1.dev > 0.5 * k2 * o1.d_eps * o1.d_eps && o1.dev < 2.0 * k2 * o1.d_eps * o1.d_eps);
    CHECK_LE(o6.dev, 1e-9);
    CHECK_LE(std::abs(o6.others / o6.d_eps / k1 - 1.0), 0.05);                  // 其余各级 ∝ Δε，一直到 Δn = 1e-6
    CHECK_LE(std::abs(o2.others / o2.d_eps / k1 - 1.0), 0.05);
    CHECK_LE(o0.dev, 1e-9);
    CHECK_LE(o0.others, 1e-9);
    CHECK_LE(one_segment.dev, 1e-9);
    CHECK_LE(one_segment.others, 1e-9);
    say("  dn = 1e-1、1e-2、1e-3、1e-6、0：零级对等效膜系的偏差 %.2e、%.2e、%.2e、%.2e、%.2e\n", o1.dev, o2.dev, o3.dev, o6.dev, o0.dev);
    say("                                  其余各级的最大振幅   %.2e、%.2e、%.2e、%.2e、%.2e\n", o1.others, o2.others, o3.others,
        o6.others, o0.others);
    say("  dev/d_eps^2 = %.4e、%.4e、%.4e；只有一段的光栅层：偏差 %.1e\n", o1.dev / (o1.d_eps * o1.d_eps),
        o2.dev / (o2.d_eps * o2.d_eps), k2, one_segment.dev);
}

// ---------------------------------------------------------------------------------------------
// 一组测试里若抛出了异常（接口本不该抛），记一次失败，其余各组照常进行。
static void run(void (*test)()) {
    try {
        test();
    } catch (const std::exception& e) {
        check(false, __FILE__, __LINE__, e.what());
    }
}

int main() {
    run(test_film_reference);
    run(test_kinematics);
    run(test_planar_limit);
    run(test_film_stacks);
    run(test_normal_incidence);
    run(test_flux_unitarity);
    run(test_reciprocity);
    run(test_mirror_symmetries);
    run(test_supercell);
    run(test_small_azimuth);
    run(test_rayleigh_crossing);
    run(test_weak_contrast);
    say("通过 %d / 共 %d\n", g_pass, g_total);
    return g_pass == g_total ? 0 : 1;
}
