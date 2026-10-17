// stablercwa/rcwa2d_operators.hpp —— 二维（交叉光栅）RCWA 的 Fourier 算子：⟦ε⟧、Cx、Cy（Li 因式分解）
//
//   namespace d2 = stablercwa::rcwa::detail2d;
//   d2::LayerOperators2D op = d2::layer_operators(layer, Lx, Ly, M1, M2, rcwa::Factorization2D::Li);
//   op.eps、op.cx、op.cy 是 N×N 复矩阵，N = (2M1+1)(2M2+1)，行列下标 = harmonic_index_2d(m, n, M1, M2)（m 在外、n 在内）。
//
// 数值权威：gui/2drcwa/rcwa2d.py 第 39–195 行（fourier_1d、toeplitz_1d、_seg_fourier、Layer2D.operators、_eps_at、
// _assemble_xfirst、_assemble_yfirst），本文件逐步照它移植（接口与约定见 gui/validation/field_reference_2d/CONTRACT_2D.md）。
// 这些函数放在 stablercwa::rcwa::detail2d（内部），各个中间步骤都单独成函数，便于单独测试。
//
// 三个算子（以 Python 的记号，⟦·⟧ 是 Fourier 系数排成的矩阵）：
//   Eps = ⟦ε⟧                  纯 Laurent 规则的介电张量卷积阵，作用在 Ez 一侧；
//   Cx  = ⟦⟦1/ε⟧_x⁻¹⟧_y         作用在 Ex 上：x 方向（法向）用逆规则，沿 y 再做 Laurent；
//   Cy  = ⟦⟦1/ε⟧_y⁻¹⟧_x         作用在 Ey 上：y 方向（法向）用逆规则，沿 x 再做 Laurent。
// 即 Li 1997 的嵌套因式分解。Factorization2D::Laurent 时 Cx = Cy = Eps（收敛差，仅作对照）。
// 一维极限（只有一个 y 分段）里 Cx 的 x 逆规则就是一维 TM 的 ⟦1/ε⟧⁻¹。
//
// 计算流程（照 Layer2D.operators）：
//   · 每个 y 分段 k 内沿 x 的两个 (2M1+1) 阶 Toeplitz 阵：⟦ε⟧_k 与 ⟦1/ε⟧_k（系数 fourier_1d 解析积分，无 FFT）；
//   · 沿 y 的分段权重 wseg = seg_fourier(y_edges, 2M2, Ly)，Eps_h[h] = Σ_k wseg[k][h]·⟦ε⟧_k，Cx_h[h] = Σ_k wseg[k][h]·⟦1/ε⟧_k⁻¹，
//     h = n − n′ + 2M2 = 0 … 4M2 是 y 方向的 Fourier 指标差；
//   · Cy：把所有 y 分段的 x 边界取模 Lx、合并去重成 x 切点集合（merge_x_cuts），对每个 x 小段取中点，用 eps_at 取出
//     每个 y 分段在该处的 ε（x 剖面的周期回绕），沿 y 做一维逆规则 ⟦1/ε_col⟧⁻¹（(2M2+1) 阶），
//     再用 x 小段的权重 seg_fourier(x_cuts, 2M1, Lx) 沿 x 做 Laurent 求和，得到 Cy_h[h]，h = m − m′ + 2M1 = 0 … 4M1；
//   · 装配成 N×N：assemble_xfirst（Eps、Cx：块 h 是 (2M1+1) 阶的 x 算符，块下标 n−n′）、
//     assemble_yfirst（Cy：块 h 是 (2M2+1) 阶的 y 算符，块下标 m−m′）。
//
// 与 Python 的两处有意差别（结果在舍入误差内相同）：
//   · Laurent 模式不计算逆规则那几组矩阵（Python 算完又丢掉）；
//   · 与 Python 一样不对 ⟦ε⟧ 做任何清零：y 不变的层里 wseg 的 n ≠ 0 分量不是精确的零，而是 1e-17 ~ 1e-16 的舍入误差，原样保留。
//
// 出错：参数非法抛 std::invalid_argument，信息是中文、带层名（Layer2D::name）。包括：M 为负、周期非正或非有限、
// y 边界不递增或不恰好覆盖一个 y 周期（相对容差 1e-9）、x 剖面的边界不递增或不恰好覆盖一个 x 周期、n_values 个数不对、
// x_profiles 与 y 分段数不一致、折射率非有限或 n² 溢出；结构层里 ε = n² 为零（折射率为零或 n² 下溢）或 1/ε 溢出（1/ε 无定义，Python 里是 inf/NaN）。
// 逆规则要求的 Toeplitz 阵奇异（出现精确零主元），或结果里出现 NaN/Inf：抛 std::runtime_error（求解层应按“数值失败”处理）。
//
// 内存：层内的 Fourier 块只有 (4M2+1)(2M1+1)² 与 (4M1+1)(2M2+1)² 个复数（M1 = M2 = 12 时各约 0.5 MB）；三个输出矩阵各 16·N² 字节
// （N = 625 时 6.25 MB，共 18.75 MB）。三组块依次计算、装配、释放，峰值只多出一组块。
// parallel = true 时按 y 分段、x 切点分段、Fourier 指标 h 与输出列用 parallel_for 分给线程池（线程数由 set_num_threads 设定）；
// 每个输出元素的累加次序固定，结果与线程数、parallel 取值逐位相同。
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "lu.hpp"
#include "matrix.hpp"
#include "parallel.hpp"
#include "rcwa2d_types.hpp"

namespace stablercwa {
namespace rcwa {
namespace detail2d {

struct LayerOperators2D {
    Matrix eps, cx, cy;   // N×N，N = (2M1+1)(2M2+1)，下标 = harmonic_index_2d；均匀层 = n²·I 三份
};

// ------------------------------------------------------------------ 一维 Fourier 工具（rcwa2d.py:42–69）

// 第 h 个谐波在各边界处的相位 exp(−i·2π·h·x_k/period)。参数的计算次序与 NumPy 的 exp(−2j·π·outer(h, edges)/period) 相同。
inline Vector edge_phases(const std::vector<double>& edges, int h, double period) {
    Vector phase(edges.size());
    for (std::size_t k = 0; k < edges.size(); ++k) {
        const double arg = (-2.0 * kPi * (static_cast<double>(h) * edges[k])) / period;
        phase[k] = cplx{std::cos(arg), std::sin(arg)};
    }
    return phase;
}

// 分段常数剖面的 Fourier 系数 c[h + hmax]（h = −hmax … hmax），f(x) = Σ c_h exp(i·2π·h·x/period)。
//   c_0 = Σ v_j·(x_{j+1} − x_j)/period；  c_h = Σ_j v_j·[exp(−i2πh x_{j+1}/P) − exp(−i2πh x_j/P)] / (−i2πh)。
// 对应 rcwa2d.py:42 fourier_1d。edges 个数必须比 values 多一个。
inline Vector fourier_1d(const std::vector<double>& edges, const Vector& values, double period, int hmax) {
    if (hmax < 0) throw std::invalid_argument("fourier_1d：hmax 不能为负");
    if (edges.size() != values.size() + 1) throw std::invalid_argument("fourier_1d：边界个数必须比取值个数多一个");
    Vector c(static_cast<std::size_t>(2 * hmax + 1));
    cplx mean{};
    for (std::size_t j = 0; j < values.size(); ++j) mean += values[j] * (edges[j + 1] - edges[j]);
    c[static_cast<std::size_t>(hmax)] = mean / period;
    for (int h = -hmax; h <= hmax; ++h) {
        if (h == 0) continue;
        const Vector phase = edge_phases(edges, h, period);
        cplx sum{};
        for (std::size_t j = 0; j < values.size(); ++j) sum += values[j] * (phase[j + 1] - phase[j]);
        const double w = -2.0 * kPi * static_cast<double>(h);   // 分母是纯虚数 i·w：(a + i·b)/(i·w) = (b − i·a)/w
        c[static_cast<std::size_t>(h + hmax)] = cplx{sum.imag() / w, -sum.real() / w};
    }
    return c;
}

// 由 c[h + 2M]（h = −2M … 2M，共 4M+1 个）生成 (2M+1) 阶 Toeplitz 阵，(i, j) 元 = c[i − j]（下标 i、j 从 0 起，对应谐波 i−M、j−M）。
// 对应 rcwa2d.py:55 toeplitz_1d。
inline Matrix toeplitz_1d(const Vector& c, int M) {
    if (M < 0) throw std::invalid_argument("toeplitz_1d：M 不能为负");
    if (c.size() != static_cast<std::size_t>(4 * M + 1)) throw std::invalid_argument("toeplitz_1d：系数个数必须是 4M+1");
    const int n = 2 * M + 1;
    Matrix t(n, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) t(i, j) = c[static_cast<std::size_t>(i - j + 2 * M)];
    return t;
}

// 每个分段对第 h 个 Fourier 分量的贡献系数，K×(2·hmax+1) 的矩阵：第 (k, h + hmax) 元是
//   h = 0：(x_{k+1} − x_k)/period；  h ≠ 0：[exp(−i2πh x_{k+1}/P) − exp(−i2πh x_k/P)] / (−i2πh)。
// 对应 rcwa2d.py:61 _seg_fourier。
inline Matrix seg_fourier(const std::vector<double>& edges, int hmax, double period) {
    if (hmax < 0) throw std::invalid_argument("seg_fourier：hmax 不能为负");
    if (edges.size() < 2) throw std::invalid_argument("seg_fourier：至少要有两个边界");
    const int K = static_cast<int>(edges.size()) - 1;
    Matrix out(K, 2 * hmax + 1);
    for (int k = 0; k < K; ++k) out(k, hmax) = (edges[k + 1] - edges[k]) / period;
    for (int h = -hmax; h <= hmax; ++h) {
        if (h == 0) continue;
        const Vector phase = edge_phases(edges, h, period);
        const double w = -2.0 * kPi * static_cast<double>(h);
        for (int k = 0; k < K; ++k) {
            const cplx d = phase[k + 1] - phase[k];
            out(k, h + hmax) = cplx{d.imag() / w, -d.real() / w};
        }
    }
    return out;
}

// ------------------------------------------------------------------ x 剖面的取值与 x 切点（rcwa2d.py:144–152、169–174）

// 与 Python / NumPy 的浮点 % 相同：结果与除数同号，落在 [0, b)（舍入后可能恰好等于 b）。fmod 的结果为零时返回 +0（b > 0）。
inline double floor_mod(double a, double b) {
    double mod = std::fmod(a, b);
    if (mod != 0.0) {
        if ((b < 0.0) != (mod < 0.0)) mod += b;
    } else {
        mod = std::copysign(0.0, b);
    }
    return mod;
}

// 排序并去掉相等的元素（np.unique）。
inline void sort_unique(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
}

// x 剖面在 x 处的 ε = n²，对应 rcwa2d.py:169 _eps_at：剖面按周期 Lx 回绕（起点可以不是 0，也可以是负数）：
//     xx = xe[0] + (x − xe[0]) mod Lx；  k = searchsorted(xe, xx, side="right") − 1，夹到 [0, 段数 − 1]。
inline cplx eps_at(const XProfile2D& profile, double x, double Lx) {
    const std::vector<double>& xe = profile.edges;
    if (xe.empty() || profile.n_values.empty()) throw std::invalid_argument("eps_at：剖面为空");
    const double xx = xe[0] + floor_mod(x - xe[0], Lx);
    const std::ptrdiff_t k = (std::upper_bound(xe.begin(), xe.end(), xx) - xe.begin()) - 1;
    const std::ptrdiff_t last = static_cast<std::ptrdiff_t>(profile.n_values.size()) - 1;
    const cplx n = profile.n_values[static_cast<std::size_t>((std::min)((std::max)(k, std::ptrdiff_t{0}), last))];
    return n * n;
}

// 所有 x 剖面的边界合并成的 x 切点集合（rcwa2d.py:144–152，逐步照搬）：
//   1) 各剖面的边界取模 Lx，再加上 0 与 Lx，排序去重；
//   2) round 到 12 位小数（NumPy 的 round：乘 1e12、四舍五入到偶数、再除 1e12），再次排序去重；
//   3) 只保留落在 [−1e-12, Lx + 1e-12] 里的点（Python 是筛掉区间外的点，不是夹回）；
//   4) 首点大于 1e-12 则在前面补 0，末点小于 Lx − 1e-12 则在后面补 Lx。
inline std::vector<double> merge_x_cuts(const std::vector<XProfile2D>& profiles, double Lx) {
    std::vector<double> cuts;
    for (const XProfile2D& p : profiles)
        for (double e : p.edges) cuts.push_back(floor_mod(e, Lx));
    cuts.push_back(0.0);
    cuts.push_back(Lx);
    sort_unique(cuts);
    for (double& c : cuts) c = std::rint(c * 1e12) / 1e12;
    sort_unique(cuts);
    cuts.erase(std::remove_if(cuts.begin(), cuts.end(), [&](double c) { return !(c >= -1e-12 && c <= Lx + 1e-12); }),
               cuts.end());
    if (cuts.empty() || cuts.front() > 1e-12) cuts.insert(cuts.begin(), 0.0);
    if (cuts.back() < Lx - 1e-12) cuts.push_back(Lx);
    return cuts;
}

// ------------------------------------------------------------------ 装配（rcwa2d.py:177–188）

namespace impl {

template <class Body>
inline void for_each_index(bool parallel, int count, Body&& body) {
    if (parallel && count > 1) parallel_for(0, count, body);
    else
        for (int i = 0; i < count; ++i) body(i);
}

inline void check_blocks(const std::vector<Matrix>& blocks, int count, int size, const char* who) {
    if (blocks.size() != static_cast<std::size_t>(count))
        throw std::invalid_argument(std::string(who) + "：块的个数不对");
    for (const Matrix& b : blocks)
        if (b.rows() != size || b.cols() != size) throw std::invalid_argument(std::string(who) + "：块的阶数不对");
}

}  // namespace impl

// blocks[h]（h = n − n′ + 2M2 = 0 … 4M2）是 (2M1+1) 阶的 x 算符 → N×N：
//   全阵 (行 (m, n)、列 (m′, n′)) = blocks[n − n′ + 2M2](m + M1, m′ + M1)。
inline Matrix assemble_xfirst(const std::vector<Matrix>& blocks, int M1, int M2, bool parallel = false) {
    if (M1 < 0 || M2 < 0) throw std::invalid_argument("assemble_xfirst：M1、M2 不能为负");
    const int n1 = 2 * M1 + 1, n2 = 2 * M2 + 1, N = n1 * n2;
    impl::check_blocks(blocks, 4 * M2 + 1, n1, "assemble_xfirst");
    Matrix out(N, N);
    const auto body = [&](int lo, int hi) {
        for (int col = lo; col < hi; ++col) {
            const int mp = col / n2, np = col % n2;          // m′ + M1、n′ + M2
            cplx* dst = out.col(col);
            for (int nn = 0; nn < n2; ++nn) {                 // nn = n + M2
                const cplx* src = blocks[static_cast<std::size_t>(nn - np + 2 * M2)].col(mp);
                for (int mm = 0; mm < n1; ++mm) dst[mm * n2 + nn] = src[mm];
            }
        }
    };
    if (parallel) parallel_ranges(0, N, body, 64);
    else body(0, N);
    return out;
}

// blocks[h]（h = m − m′ + 2M1 = 0 … 4M1）是 (2M2+1) 阶的 y 算符 → N×N：
//   全阵 (行 (m, n)、列 (m′, n′)) = blocks[m − m′ + 2M1](n + M2, n′ + M2)。
inline Matrix assemble_yfirst(const std::vector<Matrix>& blocks, int M1, int M2, bool parallel = false) {
    if (M1 < 0 || M2 < 0) throw std::invalid_argument("assemble_yfirst：M1、M2 不能为负");
    const int n1 = 2 * M1 + 1, n2 = 2 * M2 + 1, N = n1 * n2;
    impl::check_blocks(blocks, 4 * M1 + 1, n2, "assemble_yfirst");
    Matrix out(N, N);
    const auto body = [&](int lo, int hi) {
        for (int col = lo; col < hi; ++col) {
            const int mp = col / n2, np = col % n2;
            cplx* dst = out.col(col);
            for (int mm = 0; mm < n1; ++mm) {
                const cplx* src = blocks[static_cast<std::size_t>(mm - mp + 2 * M1)].col(np);
                for (int nn = 0; nn < n2; ++nn) dst[mm * n2 + nn] = src[nn];
            }
        }
    };
    if (parallel) parallel_ranges(0, N, body, 64);
    else body(0, N);
    return out;
}

// ------------------------------------------------------------------ 输入校验

namespace impl {

inline std::string layer_label(const Layer2D& layer) {
    return layer.name.empty() ? std::string("层（未命名）") : "层 \"" + layer.name + "\"";
}

[[noreturn]] inline void bad_layer(const Layer2D& layer, const std::string& what) {
    throw std::invalid_argument("rcwa2d：" + layer_label(layer) + "：" + what);
}

inline bool finite_index(const cplx& n) { return std::isfinite(n.real()) && std::isfinite(n.imag()); }

inline void check_geometry(double Lx, double Ly, int M1, int M2) {
    if (!(std::isfinite(Lx) && Lx > 0.0) || !(std::isfinite(Ly) && Ly > 0.0))
        throw std::invalid_argument("rcwa2d：周期 Lx、Ly 必须是有限的正数");
    if (M1 < 0 || M2 < 0) throw std::invalid_argument("rcwa2d：截断阶数 M1、M2 不能为负");
    if ((2LL * M1 + 1) * (2LL * M2 + 1) > static_cast<long long>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("rcwa2d：谐波总数 (2M1+1)(2M2+1) 超出 int 范围");
}

// 边界个数 ≥ 2、全部有限、严格递增、首尾相差一个周期（相对容差 1e-9）。what 是出错信息里的名字。
inline void check_edges(const Layer2D& layer, const std::vector<double>& e, double period, const std::string& what) {
    if (e.size() < 2) bad_layer(layer, what + "的边界至少要有两个");
    for (std::size_t j = 0; j < e.size(); ++j) {
        if (!std::isfinite(e[j])) bad_layer(layer, what + "的边界含非有限值");
        if (j > 0 && !(e[j] > e[j - 1])) bad_layer(layer, what + "的边界必须严格递增");
    }
    if (std::abs((e.back() - e.front()) - period) > 1e-9 * period)
        bad_layer(layer, what + "的边界必须恰好覆盖一个周期");
}

// 结构层的全部几何与折射率。
inline void check_patterned(const Layer2D& layer, double Lx, double Ly) {
    check_edges(layer, layer.y_edges, Ly, "y 分段");
    const std::size_t K = layer.y_edges.size() - 1;
    if (layer.x_profiles.size() != K)
        bad_layer(layer, "x_profiles 的个数 " + std::to_string(layer.x_profiles.size()) + " 与 y 分段数 " +
                             std::to_string(K) + " 不一致");
    for (std::size_t k = 0; k < K; ++k) {
        const XProfile2D& p = layer.x_profiles[k];
        const std::string who = "第 " + std::to_string(k) + " 个 y 分段的 x 剖面";
        check_edges(layer, p.edges, Lx, who);
        if (p.n_values.size() + 1 != p.edges.size())
            bad_layer(layer, who + "的 n_values 个数 " + std::to_string(p.n_values.size()) + " 与边界数 " +
                                 std::to_string(p.edges.size()) + " 不匹配（应比边界少一个）");
        for (const cplx& n : p.n_values) {
            if (!finite_index(n)) bad_layer(layer, who + "的折射率不是有限值");
            const cplx eps = n * n;
            if (!finite_index(eps)) bad_layer(layer, who + "的 ε = n² 溢出");
            if (eps == cplx{}) bad_layer(layer, who + "的 ε = n² 为零（折射率为零或 n² 下溢，1/ε 无定义）");
            if (!finite_index(cplx{1.0, 0.0} / eps)) bad_layer(layer, who + "的 1/ε 溢出");
        }
    }
}

inline Matrix checked_inverse(const Matrix& a, const Layer2D& layer, const std::string& what) {
    try {
        return stablercwa::inverse(a);
    } catch (const std::runtime_error&) {
        throw std::runtime_error("rcwa2d：" + layer_label(layer) + "：" + what + "奇异，无法求逆");
    }
}

inline Vector squares(const std::vector<cplx>& n) {
    Vector e(n.size());
    for (std::size_t j = 0; j < n.size(); ++j) e[j] = n[j] * n[j];
    return e;
}

inline Vector reciprocals(const Vector& e) {
    Vector r(e.size());
    for (std::size_t j = 0; j < e.size(); ++j) r[j] = cplx{1.0, 0.0} / e[j];
    return r;
}

// out[h] = Σ_k w(k, h)·mats[k]（h = 0 … w.cols()−1；k 从小到大逐个累加，与 Python 的次序相同）。mats 里每个都是 size 阶方阵。
inline std::vector<Matrix> weighted_sum(const Matrix& w, const std::vector<Matrix>& mats, int size, bool parallel) {
    const int K = static_cast<int>(mats.size()), H = w.cols();
    if (w.rows() != K) throw std::invalid_argument("weighted_sum：权重行数与矩阵个数不符");
    std::vector<Matrix> out(static_cast<std::size_t>(H));
    const std::size_t total = static_cast<std::size_t>(size) * static_cast<std::size_t>(size);
    for_each_index(parallel, H, [&](int h) {
        Matrix acc(size, size);
        cplx* a = acc.data();
        for (int k = 0; k < K; ++k) {
            const cplx wk = w(k, h);
            const cplx* m = mats[static_cast<std::size_t>(k)].data();
            for (std::size_t t = 0; t < total; ++t) a[t] += wk * m[t];
        }
        out[static_cast<std::size_t>(h)] = std::move(acc);
    });
    return out;
}

}  // namespace impl

// ------------------------------------------------------------------ 层算子（rcwa2d.py:109 Layer2D.operators）

// Eps、Cx、Cy（N×N）。均匀层返回 n²·I 三份。结构层按上面文件头的流程计算。
// Lx、Ly 是 x、y 周期；M1、M2 是截断（−M1…M1）×（−M2…M2）；parallel = true 时用 parallel_for 分担各段的求逆与累加。
inline LayerOperators2D layer_operators(const Layer2D& layer, double Lx, double Ly, int M1, int M2,
                                        Factorization2D factorization, bool parallel = false) {
    impl::check_geometry(Lx, Ly, M1, M2);
    const int n1 = 2 * M1 + 1, n2 = 2 * M2 + 1, N = n1 * n2;
    LayerOperators2D out;

    if (layer.homogeneous) {
        if (!impl::finite_index(layer.n)) impl::bad_layer(layer, "均匀层的折射率不是有限值");
        const cplx eps = layer.n * layer.n;
        if (!impl::finite_index(eps)) impl::bad_layer(layer, "均匀层的 ε = n² 溢出");
        out.eps = Matrix(N, N);
        for (int i = 0; i < N; ++i) out.eps(i, i) = eps;
        out.cx = out.eps;
        out.cy = out.eps;
        return out;
    }

    impl::check_patterned(layer, Lx, Ly);
    const int K = static_cast<int>(layer.y_edges.size()) - 1;
    const std::vector<XProfile2D>& xps = layer.x_profiles;

    // 沿 y 的分段权重，K×(4M2+1)。
    const Matrix wseg = seg_fourier(layer.y_edges, 2 * M2, Ly);

    // --- Eps = ⟦ε⟧：每个 y 分段的 ⟦ε⟧_k（Toeplitz），沿 y 加权 ---
    {
        std::vector<Matrix> lau(static_cast<std::size_t>(K));
        impl::for_each_index(parallel, K, [&](int k) {
            const XProfile2D& xp = xps[static_cast<std::size_t>(k)];
            lau[static_cast<std::size_t>(k)] =
                toeplitz_1d(fourier_1d(xp.edges, impl::squares(xp.n_values), Lx, 2 * M1), M1);
        });
        std::vector<Matrix> eps_h = impl::weighted_sum(wseg, lau, n1, parallel);
        std::vector<Matrix>().swap(lau);
        out.eps = assemble_xfirst(eps_h, M1, M2, parallel);
    }

    if (factorization == Factorization2D::Laurent) {
        out.cx = out.eps;
        out.cy = out.eps;
    } else {
        // --- Cx：每个 y 分段的 ⟦1/ε⟧_k⁻¹（x 逆规则），沿 y 加权，装配 ---
        {
            std::vector<Matrix> invx(static_cast<std::size_t>(K));
            impl::for_each_index(parallel, K, [&](int k) {
                const XProfile2D& xp = xps[static_cast<std::size_t>(k)];
                const Vector ci = fourier_1d(xp.edges, impl::reciprocals(impl::squares(xp.n_values)), Lx, 2 * M1);
                invx[static_cast<std::size_t>(k)] = impl::checked_inverse(
                    toeplitz_1d(ci, M1), layer, "第 " + std::to_string(k) + " 个 y 分段的 x 逆规则 Toeplitz 矩阵 ⟦1/ε⟧");
            });
            std::vector<Matrix> cx_h = impl::weighted_sum(wseg, invx, n1, parallel);
            std::vector<Matrix>().swap(invx);
            out.cx = assemble_xfirst(cx_h, M1, M2, parallel);
        }

        // --- Cy：x 切点集合上每个小段的 y 方向逆规则，沿 x 加权，装配 ---
        {
            const std::vector<double> xcuts = merge_x_cuts(xps, Lx);
            const int Kx = static_cast<int>(xcuts.size()) - 1;
            const Matrix wsegx = seg_fourier(xcuts, 2 * M1, Lx);   // Kx×(4M1+1)
            std::vector<Matrix> invy(static_cast<std::size_t>(Kx));
            impl::for_each_index(parallel, Kx, [&](int j) {
                const double xm = 0.5 * (xcuts[static_cast<std::size_t>(j)] + xcuts[static_cast<std::size_t>(j) + 1]);
                Vector inv_col(static_cast<std::size_t>(K));
                for (int k = 0; k < K; ++k)
                    inv_col[static_cast<std::size_t>(k)] =
                        cplx{1.0, 0.0} / eps_at(xps[static_cast<std::size_t>(k)], xm, Lx);
                const Vector cy = fourier_1d(layer.y_edges, inv_col, Ly, 2 * M2);
                invy[static_cast<std::size_t>(j)] = impl::checked_inverse(
                    toeplitz_1d(cy, M2), layer, "第 " + std::to_string(j) + " 个 x 切点分段的 y 逆规则 Toeplitz 矩阵 ⟦1/ε⟧");
            });
            std::vector<Matrix> cy_h = impl::weighted_sum(wsegx, invy, n2, parallel);
            std::vector<Matrix>().swap(invy);
            out.cy = assemble_yfirst(cy_h, M1, M2, parallel);
        }
    }

    if (!out.eps.all_finite() || !out.cx.all_finite() || !out.cy.all_finite())
        throw std::runtime_error("rcwa2d：" + impl::layer_label(layer) + "：算子含有 NaN 或 Inf");
    return out;
}

}  // namespace detail2d
}  // namespace rcwa
}  // namespace stablercwa
