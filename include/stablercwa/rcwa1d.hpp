// stablercwa/rcwa1d.hpp —— 一维周期结构的严格耦合波分析（RCWA）：TE/TM、复折射率、S 矩阵递推
//
//   namespace rcwa = stablercwa::rcwa;
//   rcwa::Stack stack;                                   // 自上而下：入射介质 nI | 各层 | 出射介质 nII
//   stack.nI = 1.0;  stack.nII = 1.46;  stack.period = 0.574;
//   stack.layers.push_back(rcwa::Layer::grating(0.38, {0.0, 0.172, 0.574}, {1.46, 1.0}));   // 齿 | 空气
//   stack.layers.push_back(rcwa::Layer::uniform(0.115, 1.96));
//   rcwa::Result r = rcwa::solve(stack, 1.053, 66.36, 20, rcwa::Polarization::TE);
//   r.DEr[20 + m]、r.DEt[20 + m] 是第 m 级的反射、透射效率（m = −M … M，这里 M = 20）
//
// 约定（所有长度用同一单位；时间因子 exp(−iωt)；z 轴由入射侧指向出射侧）：
//   · 第 m 级的横向波矢 kx_m = nI·sin θ − m·λ/Λ，以 k0 = 2π/λ 为单位（下面的 q、kz 也是）；
//   · 主场 S（TE 是 Ey，TM 是 Hy）按各级谐波展开，第 m 级随 x 的变化是 exp(−i·k0·kx_m·x)：θ > 0 时入射波的
//     横向波矢指向 −x，第 m 级的横向波矢比入射波多 +m·2π/Λ。齿形左右不对称的光栅要留意这个方向
//     （把结构左右翻转、同时把 θ 换成 −θ，第 m 级与第 −m 级互换）。厚度为 d 的层内，非零根模式满足
//         S(z) = W·[exp(−k0·q·z)·a⁺ + exp(−k0·q·(d − z))·a⁻]，
//     a⁺ 向下传、以层顶为参考，a⁻ 向上传、以层底为参考，两个指数的模都不超过 1，所以厚层、强吸收层都不会溢出；
//   · q² 是层矩阵 A 的本征值，W 的各列是对应的本征向量：
//         TE：A = Kx² − ⟦ε⟧                          V = W·q
//         TM：A = ⟦1/ε⟧^{-1}·(Kx·⟦ε⟧^{-1}·Kx − I)      V = ⟦1/ε⟧·W·q        （逆规则）
//     有限层的数值精确零根 q_j = 0 使用辅助界面基 q_basis,j = 1 代入 V，物理 q_j 仍为零；
//     该模式的主场必须线性恢复：S_j(z) = W_j·[(a_j+b_j)−k0·z·(a_j−b_j)]，a_j、b_j 均以层顶为参考。
//     层内散射块为 Tdd = Tuu = 2/(2+h)、Rdu = Rud = h/(2+h)，h = k0·d；不会给 q 添加小量。
//     ⟦f⟧ 是 f(x) 的 Fourier 系数排成的卷积矩阵，第 (m, p) 元为 c_{m−p}；Kx = diag(kx_m)；
//   · q 取 Re q ≥ 0 的一支（倏逝模沿传播方向衰减）；Re q ≈ 0 的传播模取 Im q ≤ 0，即 exp(+i·kz·z)；
//   · 吸收介质的折射率写成 n + i·k，k > 0，均匀层、光栅的齿和槽、出射介质都可以是吸收的。
//     入射介质必须无吸收，否则衍射效率无从归一。
//
// 算法（与常见的 S 矩阵 RCWA 相同）：
//   1. 逐层求模式 (W, V, q)。均匀层有解析解：W = I，V = diag(p)，q = −i·sqrt(n² − kx²)（TE：p = q；TM：p = q/n²）；
//      有限层精确零根的 p 同样使用上述辅助基；光栅层要对 A 做一次稠密本征分解，这是最花时间的一步；
//   2. 相邻两层的界面给出一个 S 矩阵块，层内的传播给出一个对角的块；
//   3. 自下而上递推“从每个块的上方看下去的总反射”，再自上而下把入射波一路推到出射介质；
//   4. 把各级复振幅换算成衍射效率。
// 只算远场（各级复振幅与效率），不做层内场的重建。
//
// 数值失败不抛异常：某一层的本征迭代没有收敛、某个线性方程组奇异（出现零主元）、或者计算中途出现了 NaN/Inf 时，
// 返回的 R、T、DEr、DEt 全部是 NaN，eig_info 给出本征求解的 info。参数非法抛 std::invalid_argument。
//
// 恰好落在 Rayleigh 异常点上（某一级在某个均匀介质里恰好掠射，kz 精确为零）时：
//   · 发生在入射介质里：不妨碍求解；
//   · 发生在有限层里：零根子空间的本征基完整时使用上述线性场分支，支持多个零根与零厚度层；
//     一般亏损本征基仍不在此分支的支持范围内，线性方程组奇异时按数值失败处理；
//   · 发生在出射介质里：它上面紧挨着均匀层、且走默认的快速路径时照常求解，其余情形（紧挨光栅层，或
//     homogeneous_fast_path = false）界面上的方程组奇异，按数值失败处理。辅助基不改变半无限介质的辐射条件。
// 精确零根分支仅匹配数值 q == 0。homogeneous_fast_path = false 时均匀层的 q 来自本征分解，带着舍入误差，
// 一般不会精确为零；这些近零根仍走原指数模式路径，结果的精度可能下降。
// kz 只是很小而不为零时照常求解，精度随 |kz| 减小缓慢下降（|kz| ≈ 1e-7 时无吸收结构的能量守恒实测仍到 1e-11）。
// 把波长或入射角挪动一点点就能避开异常点。
#pragma once

#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "eig.hpp"
#include "lu.hpp"
#include "matrix.hpp"
#include "parallel.hpp"
#include "qr.hpp"

namespace stablercwa {
namespace rcwa {

enum class Polarization { TE, TM };   // TE：电场平行于栅线（主场 Ey）；TM：磁场平行于栅线（主场 Hy）

// 求解稠密线性方程组的方法。求解过程中每一处稠密求解都用它。
enum class LinearSolver {
    LUPartial,       // 按列选主元的 LU 分解（默认）
    LUNoPivot,       // 不选主元的 LU 分解（只用于演示不稳定性）
    QRHouseholder,   // Householder QR 分解
};

struct Layer {
    double d = 0.0;                    // 层厚
    cplx n = 1.0;                      // 均匀层折射率；吸收介质写成 n + i·k，k > 0
    std::vector<double> edges;         // 光栅层：一个周期内的分段边界（递增，首尾相差一个周期，起点可以不是 0）
    std::vector<cplx> n_values;        // 光栅层：各段折射率（可为复数），个数比边界少 1
    bool homogeneous = true;

    static Layer uniform(double thickness, cplx index) {
        Layer layer;
        layer.d = thickness;
        layer.n = index;
        return layer;
    }
    static Layer grating(double thickness, std::vector<double> boundaries, std::vector<cplx> indices) {
        Layer layer;
        layer.d = thickness;
        layer.edges = std::move(boundaries);
        layer.n_values = std::move(indices);
        layer.homogeneous = false;
        return layer;
    }
};

// 自上而下：入射介质 nI、各层 layers、出射介质 nII；period 是横向周期 Λ。
struct Stack {
    cplx nI = 1.0;
    std::vector<Layer> layers;
    cplx nII = 1.0;
    double period = 1.0;
};

struct Options {
    EigMethod eig = EigMethod::QR;                    // 层矩阵的本征分解用哪种迭代
    LinearSolver linear = LinearSolver::LUPartial;
    bool homogeneous_fast_path = true;   // 均匀层用解析模式和对角 S 块；false 时把均匀层当作一般层走稠密路径
    bool diagnostics = false;            // 是否额外计算残差类诊断量（会多花时间）
    bool parallel_layers = false;        // 各个需要本征分解的层用 parallel_for 分给线程池（这样的层至少 2 个才有意义）
    // 若设置，层矩阵的本征分解改由它完成（例如 GPU 后端）；返回值的约定与 eig() 相同（本征向量按 2-范数归一）。
    // 与 parallel_layers 或 sweep() 同用时，它必须能被多个线程同时调用。
    std::function<EigResult(const Matrix&)> custom_eig;
};

struct Point { double wavelength = 1.0; double theta_deg = 0.0; };   // 扫描里的一个点

// 一次求解的用时（秒）。eig：本征分解；linear：稠密线性求解（分解 + 回代）；product：稠密矩阵乘法。
// 其余的时间（组装矩阵、对角运算、诊断量）只计入 total。
struct Timing { double total = 0, eig = 0, linear = 0, product = 0; };

struct Result {
    std::vector<int> orders;           // -M … +M；第 m 级在各数组里的下标是 m + M
    Vector R, T;                       // 各级复振幅（主场的反射、透射振幅，入射波振幅为 1）
    RealVector DEr, DEt;               // 各级反射 / 透射效率
    double reflectance() const {
        double sum = 0.0;
        for (double v : DEr) sum += v;
        return sum;
    }
    double transmittance() const {
        double sum = 0.0;
        for (double v : DEt) sum += v;
        return sum;
    }
    double max_eig_residual = 0;       // diagnostics=true 时填写：各层本征对相对残差（eig_residual）的最大值
    double max_solve_backward_error = 0;   // diagnostics=true 时填写：各次稠密求解后向误差的最大值
    int eig_info = 0;                  // 本征求解的 info（多层取第一个非零值）
    int n_eig = 0, n_dense_solve = 0;  // 做了多少次稠密本征分解、稠密线性求解
    Timing timing;
};

namespace detail {

using Clock = std::chrono::steady_clock;
inline double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

// 数值失败（方程组奇异、出现 NaN/Inf、本征迭代未收敛）时在内部抛出，solve() 接住后把结果填成 NaN。
struct NumericalFailure {};

// 一次求解过程中的设置、计时与计数。每次 solve() 自己建一个，没有全局状态，所以 solve() 可重入。
struct Work {
    LinearSolver linear = LinearSolver::LUPartial;
    bool diagnostics = false;
    Timing timing;
    int n_solve = 0;
    double max_backward_error = 0.0;
};

// ------------------------------------------------------------------ 参数检查

inline bool is_finite(cplx v) { return std::isfinite(v.real()) && std::isfinite(v.imag()); }

// 折射率须有限、非零、Re n ≥ 0、Im n ≥ 0（无增益）。
inline void check_index(cplx n) {
    if (!is_finite(n) || std::abs(n) == 0.0 || n.real() < 0.0 || n.imag() < 0.0)
        throw std::invalid_argument("rcwa: refractive index must be finite, nonzero, with Re n >= 0 and Im n >= 0");
}

inline void check_layer(const Layer& layer, double period) {
    if (!std::isfinite(layer.d) || layer.d < 0.0)
        throw std::invalid_argument("rcwa: layer thickness must be finite and >= 0");
    if (layer.homogeneous) {
        check_index(layer.n);
        return;
    }
    const std::vector<double>& e = layer.edges;
    if (layer.n_values.empty() || e.size() != layer.n_values.size() + 1)
        throw std::invalid_argument("rcwa: a grating layer needs K indices and K+1 edges");
    for (std::size_t j = 0; j < e.size(); ++j)
        if (!std::isfinite(e[j]) || (j > 0 && e[j] <= e[j - 1]))
            throw std::invalid_argument("rcwa: grating edges must be finite and strictly increasing");
    if (std::abs((e.back() - e.front()) - period) > 1e-10 * period)
        throw std::invalid_argument("rcwa: grating edges must span exactly one period");
    for (const cplx& n : layer.n_values) check_index(n);
}

inline void check_setup(double period, double wavelength, cplx nI, double theta_deg, int M, Polarization pol) {
    if (M < 0 || M > ((std::numeric_limits<int>::max)() - 1) / 4)   // 多一层括号：见 parallel.hpp 开头的说明
        throw std::invalid_argument("rcwa: invalid truncation order M");
    if (pol != Polarization::TE && pol != Polarization::TM)
        throw std::invalid_argument("rcwa: polarization must be TE or TM");
    if (!std::isfinite(wavelength) || wavelength <= 0.0 || !std::isfinite(period) || period <= 0.0)
        throw std::invalid_argument("rcwa: wavelength and period must be finite and positive");
    if (!std::isfinite(theta_deg) || std::abs(theta_deg) >= 90.0)
        throw std::invalid_argument("rcwa: incidence angle must lie strictly between -90 and 90 degrees");
    check_index(nI);
    if (nI.imag() != 0.0 || nI.real() <= 0.0)
        throw std::invalid_argument("rcwa: the incident medium must have a real positive index");
}

// 取较大者；任何一个是 NaN 时结果就是 NaN（诊断量里出现过 NaN 不能被后面的正常值盖掉）。
inline double max_keep_nan(double current, double value) {
    if (std::isnan(current) || std::isnan(value)) return std::numeric_limits<double>::quiet_NaN();
    return value > current ? value : current;
}

// ------------------------------------------------------------------ 计时的稠密运算

// 可选近场快照使用与层矩阵组装同一次求解的分解；普通 solve 不构造或保留它。
struct LinearFactors {
    LinearSolver method = LinearSolver::LUPartial;
    LU lu;
    QR qr;
};

// 解 A X = B，方法由 work.linear 指定。方程组奇异（零主元）时抛 NumericalFailure。
inline Matrix linear_solve(const Matrix& a, const Matrix& b, Work& work, LinearFactors* retained = nullptr) {
    ++work.n_solve;
    const auto t0 = Clock::now();
    Matrix x;
    try {
        if (retained) {
            retained->method = work.linear;
            if (work.linear == LinearSolver::QRHouseholder) {
                retained->qr = qr_factor(a);
                x = qr_solve(retained->qr, b);
            } else {
                const Pivoting pivot = work.linear == LinearSolver::LUPartial ? Pivoting::Partial : Pivoting::None;
                // solve() 本来也使用不跟踪增长因子的这份分解，保留时不改变运算或诊断。
                retained->lu = stablercwa::detail::dense::lu_factor_impl<false>(a, pivot);
                x = lu_solve(retained->lu, b);
            }
        } else switch (work.linear) {
            case LinearSolver::LUPartial: x = stablercwa::solve(a, b, Pivoting::Partial); break;
            case LinearSolver::LUNoPivot: x = stablercwa::solve(a, b, Pivoting::None); break;
            case LinearSolver::QRHouseholder: x = solve_qr(a, b); break;
        }
    } catch (const std::runtime_error&) {
        work.timing.linear += seconds_since(t0);
        throw NumericalFailure{};
    }
    work.timing.linear += seconds_since(t0);
    if (work.diagnostics) work.max_backward_error = max_keep_nan(work.max_backward_error, solve_backward_error(a, x, b));
    return x;
}

// c[i] += a[i]·t。直接写实部、虚部的算式：std::complex 的乘法带有 NaN 修正分支，循环无法向量化。
// （std::complex<double> 的数组可以当作实部、虚部交替的 double 数组访问，这是标准保证的布局。）
inline void add_column(cplx* c_, const cplx* a_, cplx t, int len) noexcept {
    double* c = reinterpret_cast<double*>(c_);
    const double* a = reinterpret_cast<const double*>(a_);
    const double tr = t.real(), ti = t.imag();
    for (int i = 0; i < len; ++i) {
        const int r = 2 * i, s = r + 1;
        const double xr = c[r] + a[r] * tr - a[s] * ti;
        const double xi = c[s] + a[r] * ti + a[s] * tr;
        c[r] = xr;
        c[s] = xi;
    }
}

// 依次做 c += a0·t[0]、c += a1·t[1]、c += a2·t[2]、c += a3·t[3]，但 c 只读写一遍（省一大半的内存访问）。
inline void add_columns4(cplx* c_, const cplx* a0_, const cplx* a1_, const cplx* a2_, const cplx* a3_, const cplx* t,
                         int len) noexcept {
    double* c = reinterpret_cast<double*>(c_);
    const double* a0 = reinterpret_cast<const double*>(a0_);
    const double* a1 = reinterpret_cast<const double*>(a1_);
    const double* a2 = reinterpret_cast<const double*>(a2_);
    const double* a3 = reinterpret_cast<const double*>(a3_);
    const double t0r = t[0].real(), t0i = t[0].imag(), t1r = t[1].real(), t1i = t[1].imag();
    const double t2r = t[2].real(), t2i = t[2].imag(), t3r = t[3].real(), t3i = t[3].imag();
    for (int i = 0; i < len; ++i) {
        const int r = 2 * i, s = r + 1;
        double xr = c[r], xi = c[s];
        xr = xr + a0[r] * t0r - a0[s] * t0i;
        xi = xi + a0[r] * t0i + a0[s] * t0r;
        xr = xr + a1[r] * t1r - a1[s] * t1i;
        xi = xi + a1[r] * t1i + a1[s] * t1r;
        xr = xr + a2[r] * t2r - a2[s] * t2i;
        xi = xi + a2[r] * t2i + a2[s] * t2r;
        xr = xr + a3[r] * t3r - a3[s] * t3i;
        xi = xi + a3[r] * t3i + a3[s] * t3r;
        c[r] = xr;
        c[s] = xi;
    }
}

// C = A·B（计时）。稠密乘法是求解里除本征分解之外最花时间的运算，所以单独写了内核：
// C 的每一列是 A 的各列的线性组合，四列一组累加；B 里整组为零的系数直接跳过（B 是对角阵时只剩 O(n²)）。
// 列与列互不相干，矩阵够大时按列并行；每个元素上各项相加的次序是固定的，结果与线程数无关。
inline Matrix multiply(const Matrix& a, const Matrix& b, Work& work) {
    const auto t0 = Clock::now();
    const int m = a.rows(), n = b.cols(), p = a.cols();
    Matrix c(m, n);
    const auto block = [&](int lo, int hi) {
        for (int j = lo; j < hi; ++j) {
            cplx* cj = c.col(j);
            const cplx* bj = b.col(j);
            int k = 0;
            for (; k + 4 <= p; k += 4) {
                const cplx t[4] = {bj[k], bj[k + 1], bj[k + 2], bj[k + 3]};
                if (t[0] == cplx{} && t[1] == cplx{} && t[2] == cplx{} && t[3] == cplx{}) continue;
                add_columns4(cj, a.col(k), a.col(k + 1), a.col(k + 2), a.col(k + 3), t, m);
            }
            for (; k < p; ++k)
                if (bj[k] != cplx{}) add_column(cj, a.col(k), bj[k], m);
        }
    };
    if (static_cast<long long>(m) * n * p < 200000) block(0, n);
    else parallel_ranges(0, n, block, 4);
    work.timing.product += seconds_since(t0);
    return c;
}

// ------------------------------------------------------------------ Fourier 系数与层矩阵

// 分段常数函数 f(x) 的 Fourier 系数 c_h（h = −hmax … hmax，存在下标 h + hmax 处），f(x) = Σ c_h·exp(i·2πh·x/Λ)。
// 每一段上对指数函数精确积分：c_0 是周期平均值，h ≠ 0 时
//     c_h = Σ_j v_j·[exp(−i·2πh·x_{j+1}/Λ) − exp(−i·2πh·x_j/Λ)] / (−i·2πh)。
// 只有一段的剖面（均匀层）代进这个公式，h ≠ 0 的系数是两个几乎相等的数相减，得到的不是严格的零，
// 而是 1e-17 ~ 1e-16 量级的舍入误差。这里不做任何清零，原样返回（见 solve() 关于 homogeneous_fast_path 的说明）。
inline Vector fourier_coefficients(const std::vector<double>& edges, const Vector& values, double period, int hmax) {
    Vector c(static_cast<std::size_t>(2 * hmax + 1));
    const std::size_t segments = values.size();
    Vector phase(edges.size());
    for (int h = -hmax; h <= hmax; ++h) {
        cplx sum{};
        if (h == 0) {
            for (std::size_t j = 0; j < segments; ++j) sum += values[j] * (edges[j + 1] - edges[j]) / period;
        } else {
            for (std::size_t k = 0; k < edges.size(); ++k) {
                const double arg = -2.0 * kPi * h * edges[k] / period;
                phase[k] = cplx{std::cos(arg), std::sin(arg)};
            }
            const double w = -2.0 * kPi * h;                 // 分母是纯虚数 i·w：(a + i·b)/(i·w) = (b − i·a)/w
            for (std::size_t j = 0; j < segments; ++j) {
                const cplx t = values[j] * (phase[j + 1] - phase[j]);
                sum += cplx{t.imag() / w, -t.real() / w};
            }
        }
        c[static_cast<std::size_t>(h + hmax)] = sum;
    }
    return c;
}

// 卷积矩阵 ⟦f⟧：(2M+1) 阶，第 (m, p) 元是 c_{m−p}。两个保留级次相减可到 ±2M，所以系数要算到 2M。
inline Matrix convolution_matrix(const std::vector<double>& edges, const Vector& values, double period, int M) {
    const Vector c = fourier_coefficients(edges, values, period, 2 * M);
    const int n = 2 * M + 1;
    Matrix m(n, n);
    for (int col = 0; col < n; ++col)
        for (int row = 0; row < n; ++row) m(row, col) = c[static_cast<std::size_t>(row - col + 2 * M)];
    return m;
}

// 各级横向波矢 kx_m = nI·sin θ − m·λ/Λ（m = −M … M）。入射介质无吸收，所以都是实数。
inline Vector transverse_wavevectors(double nI, double theta_deg, double wavelength, double period, int M) {
    const double s = nI * std::sin(theta_deg * kPi / 180.0);
    Vector kx(static_cast<std::size_t>(2 * M + 1));
    for (int j = 0; j <= 2 * M; ++j) kx[j] = s - (j - M) * wavelength / period;
    return kx;
}

struct LayerOperator {
    Matrix a;          // 层矩阵 A，本征值是 q²
    Matrix inv_eps;    // ⟦1/ε⟧，只有 TM 用到（由 W 求 V）
};

struct PlanarFieldMaterial {
    bool analytic = false;           // 均匀快速路径：ε 为标量
    cplx epsilon{};
    Matrix eps;                      // TE 的材料算符；TM 保留下面同一次 Eps 求解的分解
    LinearFactors eps_factors;
    bool factorized = false;
};

// 组装一层的层矩阵。均匀层在这里被看成只有一段的光栅层（edges = {0, Λ}），与光栅层用同一套公式，
// 所以它的 A 是“对角矩阵 + Fourier 系数里的舍入误差”。
inline LayerOperator assemble_layer(const Layer& layer, double period, const Vector& kx, int M, Polarization pol,
                                    Work& work, PlanarFieldMaterial* retained = nullptr) {
    const std::vector<double> one_edge{0.0, period};
    const std::vector<double>& edges = layer.homogeneous ? one_edge : layer.edges;
    Vector eps, inv;                                    // 各段的 ε = n² 与 1/ε
    if (layer.homogeneous) {
        eps.push_back(layer.n * layer.n);
        inv.push_back(1.0 / (layer.n * layer.n));
    } else {
        for (const cplx& n : layer.n_values) {
            eps.push_back(n * n);
            inv.push_back(1.0 / (n * n));
        }
    }
    const int size = 2 * M + 1;
    LayerOperator out;
    Matrix e = convolution_matrix(edges, eps, period, M);
    if (pol == Polarization::TE) {
        if (retained) retained->eps = e;
        // A = Kx² − ⟦ε⟧
        cplx* p = e.data();
        const std::size_t total = static_cast<std::size_t>(size) * size;
        for (std::size_t k = 0; k < total; ++k) p[k] = -p[k];
        for (int j = 0; j < size; ++j) e(j, j) += kx[j] * kx[j];
        out.a = std::move(e);
    } else {
        // A = ⟦1/ε⟧^{-1}·(Kx·⟦ε⟧^{-1}·Kx − I)。两处都是解方程，不显式求逆矩阵。
        out.inv_eps = convolution_matrix(edges, inv, period, M);
        Matrix b = scale_rows(linear_solve(e, Matrix::diagonal(kx), work,
                                         retained ? &retained->eps_factors : nullptr), kx);
        if (retained) retained->factorized = true;
        for (int j = 0; j < size; ++j) b(j, j) -= 1.0;
        out.a = linear_solve(out.inv_eps, b, work);
    }
    return out;
}

// ------------------------------------------------------------------ 模式

struct Modes {
    bool analytic = false;   // true：W = I、V = diag(p)，不存矩阵
    bool finite_zero_basis = false; // 有限层的精确零根使用辅助界面基；外部辐射模式不启用
    Vector q;                // 纵向传播常数（以 k0 为单位）
    Vector p;                // analytic 时 V 的对角元
    Matrix W, V;             // 一般模式：W 把模式振幅变成主场 S 的谐波，V 给出配对的切向场
};

// 内部快照是值对象。模式按区域排列（上介质、各层、下介质）；振幅按有限层排列。
// a_plus 锚定层顶，a_minus 锚定层底，包括有限零根的辅助基振幅。
struct PlanarFieldCapture {
    bool available = false;
    std::vector<Modes> modes;
    std::vector<PlanarFieldMaterial> materials;
    std::vector<Vector> a_plus, a_minus;
    Vector kx;                      // 同次求解使用的 k0 归一横向波矢
    Vector reflected, transmitted;
};

inline bool finite_vector(const Vector& v) {
    for (cplx z : v) if (!is_finite(z)) return false;
    return true;
}

inline bool valid_capture(const PlanarFieldCapture& fields) {
    for (const Modes& m : fields.modes)
        if (!finite_vector(m.q) || !finite_vector(m.p) || !m.W.all_finite() || !m.V.all_finite()) return false;
    for (const PlanarFieldMaterial& m : fields.materials) {
        if (m.analytic && !is_finite(m.epsilon)) return false;
        if (!m.eps.all_finite() || !m.eps_factors.lu.lu.all_finite() || !m.eps_factors.qr.qr.all_finite() ||
            !finite_vector(m.eps_factors.qr.tau)) return false;
    }
    for (const auto* amplitudes : {&fields.a_plus, &fields.a_minus})
        for (const Vector& a : *amplitudes) if (!finite_vector(a)) return false;
    return finite_vector(fields.reflected) && finite_vector(fields.transmitted);
}

inline void clear_unexcited(Vector& a, int M) {
    for (int j = 0; j < static_cast<int>(a.size()); ++j) if (j != M) a[j] = cplx{};
}

// 平方根取哪一支：Re q ≥ 0；Re q ≈ 0（传播模）时取 Im q ≤ 0。
inline cplx branch(cplx q) {
    if (q.real() < 0.0) q = -q;
    if (q.real() < 1e-10 && q.imag() > 0.0) q = -q;
    return q;
}

// 折射率为 n 的均匀介质的解析模式：kz = sqrt(n² − kx²)，传播模取实部为正，倏逝模、吸收介质取虚部为正，q = −i·kz。
// as_dense 为真时把同样的 W = I、V = diag(p) 写成稠密矩阵（homogeneous_fast_path = false 时入射、出射介质这样给出）。
inline Modes analytic_modes(cplx n, const Vector& kx, Polarization pol, bool as_dense, bool finite_zero_basis = false) {
    const std::size_t size = kx.size();

    Modes m;
    m.finite_zero_basis = finite_zero_basis;
    m.q.resize(size);
    m.p.resize(size);
    for (std::size_t j = 0; j < size; ++j) {
        cplx kz = std::sqrt(n * n - kx[j] * kx[j]);
        if (kz.imag() < 0.0) kz = -kz;
        if (std::abs(kz.imag()) < 1e-12 && kz.real() < 0.0) kz = -kz;
        m.q[j] = cplx{kz.imag(), -kz.real()};
        const cplx basis_q = finite_zero_basis && m.q[j] == cplx{} ? cplx(1.0) : m.q[j];
        m.p[j] = pol == Polarization::TE ? basis_q : basis_q / (n * n);
    }
    if (as_dense) {
        m.W = Matrix::identity(static_cast<int>(size));
        m.V = Matrix::diagonal(m.p);
    } else {
        m.analytic = true;
    }
    return m;
}

// ------------------------------------------------------------------ S 矩阵块

// n 阶算符的三种存法：全零、对角（只存对角元）、稠密。对角算符乘稠密矩阵只是按行或按列缩放，不是矩阵乘法。
struct Op {
    enum class Kind { Zero, Diag, Dense };
    Kind kind = Kind::Zero;
    Vector d;     // Kind::Diag 时的对角元
    Matrix m;     // Kind::Dense 时的矩阵
};
inline Op diag_op(Vector v) {
    Op o;
    o.kind = Op::Kind::Diag;
    o.d = std::move(v);
    return o;
}
inline Op dense_op(Matrix a) {
    Op o;
    o.kind = Op::Kind::Dense;
    o.m = std::move(a);
    return o;
}

// a·b
inline Op op_mul(const Op& a, const Op& b, Work& work) {
    using K = Op::Kind;
    if (a.kind == K::Zero || b.kind == K::Zero) return Op{};
    if (a.kind == K::Diag && b.kind == K::Diag) {
        Vector v(a.d.size());
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = a.d[i] * b.d[i];
        return diag_op(std::move(v));
    }
    if (a.kind == K::Diag) return dense_op(scale_rows(b.m, a.d));
    if (b.kind == K::Diag) return dense_op(scale_columns(a.m, b.d));
    return dense_op(multiply(a.m, b.m, work));
}

// a + b
inline Op op_add(Op a, const Op& b) {
    using K = Op::Kind;
    if (b.kind == K::Zero) return a;
    if (a.kind == K::Zero) return b;
    if (a.kind == K::Diag && b.kind == K::Diag) {
        for (std::size_t i = 0; i < a.d.size(); ++i) a.d[i] += b.d[i];
        return a;
    }
    if (a.kind == K::Dense && b.kind == K::Dense) {
        a.m += b.m;
        return a;
    }
    // 一个对角、一个稠密：把对角元加到稠密矩阵的对角线上
    if (a.kind == K::Dense) {
        for (int i = 0; i < a.m.rows(); ++i) a.m(i, i) += b.d[i];
        return a;
    }
    Op out = b;
    for (int i = 0; i < out.m.rows(); ++i) out.m(i, i) += a.d[i];
    return out;
}

// a·x
inline Vector op_apply(const Op& a, const Vector& x) {
    using K = Op::Kind;
    if (a.kind == K::Dense) return a.m * x;
    Vector y(x.size());
    if (a.kind == K::Diag)
        for (std::size_t i = 0; i < y.size(); ++i) y[i] = a.d[i] * x[i];
    return y;
}

inline Matrix op_to_dense(const Op& a, int n) {
    using K = Op::Kind;
    if (a.kind == K::Dense) return a.m;
    if (a.kind == K::Diag) return Matrix::diagonal(a.d);
    return Matrix(n, n);
}

// I − m
inline Matrix identity_minus(Matrix m) {
    cplx* p = m.data();
    const std::size_t total = static_cast<std::size_t>(m.rows()) * m.cols();
    for (std::size_t k = 0; k < total; ++k) p[k] = -p[k];
    for (int i = 0; i < m.rows(); ++i) m(i, i) += 1.0;
    return m;
}

// 一个 S 矩阵块：用进入两端的波表示离开两端的波，
//     下行输出 = Tdd·下行输入 + Rdu·上行输入，     上行输出 = Rud·下行输入 + Tuu·上行输入。
struct SBlock {
    Op Tdd, Rdu, Rud, Tuu;
};

// 上层 (W1, V1) 与下层 (W2, V2) 之间的界面。两侧切向场连续：
//     W1·(a1⁺ + a1⁻) = W2·(a2⁺ + a2⁻)，   V1·(a1⁺ − a1⁻) = V2·(a2⁺ − a2⁻)。
// 记 Mm = W2^{-1}·W1、Nn = V2^{-1}·V1、T± = (Mm ± Nn)/2，则
//     Tdd = T+ − T−·T+^{-1}·T−，   Rdu = T−·T+^{-1}，   Rud = −T+^{-1}·T−，   Tuu = T+^{-1}。

// 两侧都是均匀层（解析模式）：各级互不耦合，每一级就是一个 Fresnel 界面，四个块都是对角的。
// 写成除以 p1 + p2 的形式，某一侧 p = 0（该级恰好掠射）时也是有限的。
inline SBlock diagonal_interface(const Modes& up, const Modes& low) {
    const std::size_t size = up.p.size();
    Vector tdd(size), rdu(size), rud(size), tuu(size);
    for (std::size_t j = 0; j < size; ++j) {
        const cplx p1 = up.p[j], p2 = low.p[j];
        const cplx inv = 1.0 / (p1 + p2);
        tdd[j] = 2.0 * p1 * inv;
        rdu[j] = (p2 - p1) * inv;
        rud[j] = (p1 - p2) * inv;
        tuu[j] = 2.0 * p2 * inv;
    }
    SBlock s;
    s.Tdd = diag_op(std::move(tdd));
    s.Rdu = diag_op(std::move(rdu));
    s.Rud = diag_op(std::move(rud));
    s.Tuu = diag_op(std::move(tuu));
    return s;
}

// 至少有一侧是一般模式时的匹配矩阵 T+（tp）与 T−（tm）。
inline void interface_matching(const Modes& up, const Modes& low, Work& work, Matrix& tp, Matrix& tm) {
    const int n = static_cast<int>(up.q.size());
    Matrix mm, nn;
    if (low.analytic) {
        // 下侧是均匀层：W2 = I、V2 对角，两次稠密求解变成“原样照抄”和按行除以 p2。
        mm = up.W;
        Vector inv_p(low.p.size());
        for (std::size_t j = 0; j < inv_p.size(); ++j) inv_p[j] = 1.0 / low.p[j];
        nn = scale_rows(up.V, inv_p);
    } else if (up.analytic) {
        // 上侧是均匀层：右端项是单位阵和对角阵，方程组还是要解（系数矩阵是下层的 W、V）。
        mm = linear_solve(low.W, Matrix::identity(n), work);
        nn = linear_solve(low.V, Matrix::diagonal(up.p), work);
    } else {
        mm = linear_solve(low.W, up.W, work);
        nn = linear_solve(low.V, up.V, work);
    }
    tp = mm;
    tp += nn;
    tm = std::move(mm);
    tm -= nn;
    cplx* a = tp.data();
    cplx* b = tm.data();
    const std::size_t total = static_cast<std::size_t>(n) * n;
    for (std::size_t k = 0; k < total; ++k) {
        a[k] = 0.5 * a[k];
        b[k] = 0.5 * b[k];
    }
}

inline SBlock interface_block(const Modes& up, const Modes& low, Work& work) {
    if (up.analytic && low.analytic) return diagonal_interface(up, low);
    const int n = static_cast<int>(up.q.size());
    Matrix tp, tm;
    interface_matching(up, low, work, tp, tm);
    Matrix tpi = linear_solve(tp, Matrix::identity(n), work);
    Matrix rdu = multiply(tm, tpi, work);
    tp -= multiply(rdu, tm, work);        // tp 现在是 Tdd
    Matrix rud = multiply(tpi, tm, work);
    cplx* a = rud.data();
    const std::size_t total = static_cast<std::size_t>(n) * n;
    for (std::size_t k = 0; k < total; ++k) a[k] = -a[k];
    SBlock s;
    s.Tdd = dense_op(std::move(tp));
    s.Rdu = dense_op(std::move(rdu));
    s.Rud = dense_op(std::move(rud));
    s.Tuu = dense_op(std::move(tpi));
    return s;
}

// 最上面的界面块。入射波只有第 0 级（下标 M），Tdd 和 Rud 只用得到这一列，不必把整个矩阵乘出来：
//     Tdd·e = T+·e − Rdu·(T−·e)，     Rud·e = −T+^{-1}·(T−·e)。
struct TopBlock {
    Op Rdu, Tuu;
    Vector tdd, rud;      // Tdd、Rud 的第 M 列
};
inline TopBlock top_interface_block(const Modes& up, const Modes& low, int M, Work& work) {
    const int n = static_cast<int>(up.q.size());
    TopBlock t;
    if (up.analytic && low.analytic) {
        SBlock s = diagonal_interface(up, low);
        t.tdd.assign(static_cast<std::size_t>(n), cplx{});
        t.rud.assign(static_cast<std::size_t>(n), cplx{});
        t.tdd[M] = s.Tdd.d[M];
        t.rud[M] = s.Rud.d[M];
        t.Rdu = std::move(s.Rdu);
        t.Tuu = std::move(s.Tuu);
        return t;
    }
    Matrix tp, tm;
    interface_matching(up, low, work, tp, tm);
    Matrix tpi = linear_solve(tp, Matrix::identity(n), work);
    Matrix rdu = multiply(tm, tpi, work);
    const Vector tm_e = tm.column(M);
    t.tdd = tp.column(M);
    const Vector back = rdu * tm_e;
    for (int j = 0; j < n; ++j) t.tdd[j] -= back[j];
    t.rud = tpi * tm_e;
    for (cplx& v : t.rud) v = -v;
    t.Rdu = dense_op(std::move(rdu));
    t.Tuu = dense_op(std::move(tpi));
    return t;
}

// 普通模式乘 exp(−k0·q·d)。有限层的精确零根另用非奇异辅助基 q_basis=1：
// 零根的主场为 W_j·[(a_j+b_j)−k0*z·(a_j−b_j)]，必须线性恢复，
// 不能把辅助振幅当作 exp(±k0*z) 的物理波。默认行为仍供普通/圆锥模式使用。
inline SBlock propagation_block(const Vector& q, double d, double k0, bool as_dense,
                                bool finite_zero_basis = false) {
    Vector x(q.size()), reflection(q.size());
    bool has_zero = false;
    for (std::size_t j = 0; j < x.size(); ++j) {
        if (finite_zero_basis && q[j] == cplx{}) {
            const double h = k0 * d;
            if (!std::isfinite(h)) throw NumericalFailure{};
            // 1+h/2 避免有限 h 的 2+h 溢出；大厚度时仍保留有限的散射块。
            const double denominator = 1.0 + 0.5 * h;
            x[j] = 1.0 / denominator;
            reflection[j] = (0.5 * h) / denominator;
            has_zero = true;
        } else {
            x[j] = std::exp(-k0 * d * q[j]);
        }
    }
    SBlock s;
    if (as_dense) {
        const int n = static_cast<int>(q.size());
        s.Tdd = dense_op(Matrix::diagonal(x));
        s.Tuu = s.Tdd;
        s.Rdu = dense_op(has_zero ? Matrix::diagonal(reflection) : Matrix(n, n));
        s.Rud = s.Rdu;
    } else {
        s.Tdd = diag_op(x);
        s.Tuu = diag_op(std::move(x));
        if (has_zero) {
            s.Rdu = diag_op(reflection);
            s.Rud = diag_op(std::move(reflection));
        }
    }
    return s;
}

// 把一个块叠到下方结构之上。下方结构的总反射为 r（它上表面处 a⁻ = r·a⁺），则
//     g  = (I − Rdu·r)^{-1}·Tdd       块上方的下行波 → 块下方的下行波（已计入块与下方之间的多次往返）
//     r' = Rud + Tuu·r·g              从块的上方看下去的总反射
// 各个算符是全零、对角还是稠密由 Op 自己分辨：全是对角时逐元素算；对角乘稠密是缩放；
// 只有 Rdu·r 是稠密矩阵时才需要解一次稠密方程组。
struct Seam {
    Op g, r;
};
inline Seam stack_on(SBlock&& s, const Op& r, int n, Work& work) {
    using K = Op::Kind;
    Seam out;
    Op k = op_mul(s.Rdu, r, work);
    if (k.kind == K::Zero) {
        out.g = std::move(s.Tdd);
    } else if (k.kind == K::Diag) {
        for (cplx& v : k.d) v = 1.0 / (1.0 - v);
        out.g = op_mul(k, s.Tdd, work);
    } else {
        out.g = dense_op(linear_solve(identity_minus(std::move(k.m)), op_to_dense(s.Tdd, n), work));
    }
    out.r = op_add(std::move(s.Rud), op_mul(s.Tuu, op_mul(r, out.g, work), work));
    return out;
}

inline void fill_nan(Result& result, int n) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    result.R.assign(static_cast<std::size_t>(n), cplx{nan, nan});
    result.T.assign(static_cast<std::size_t>(n), cplx{nan, nan});
    result.DEr.assign(static_cast<std::size_t>(n), nan);
    result.DEt.assign(static_cast<std::size_t>(n), nan);
}

}  // namespace detail

// 主入口。wavelength 是真空波长，theta_deg 是入射角（度），M 是单侧截断阶数（保留 −M … M 共 2M+1 级）。
//
// options.homogeneous_fast_path = true（默认）：
//   均匀层直接用解析模式，不做本征分解；对角的 S 块（层内传播、均匀层之间的界面）只存对角元，
//   一段连续的均匀膜系整个是逐元素运算；光栅层与均匀层之间的界面，下侧是均匀层时不需要对 W、V 解方程。
//   最下面一个光栅层以下的膜系不产生稠密运算；它以上的膜，自下而上递推到那里时总反射已经是稠密矩阵，
//   每个膜界面要多做一次稠密求解和一次稠密乘法。
// options.homogeneous_fast_path = false：
//   不做上面任何一项特殊处理。每个均匀层都与光栅层走逐行相同的代码：把它写成只有一段的光栅层，
//   用通用的分段常数公式算 Fourier 系数（非零级的系数因此是 1e-17 ~ 1e-16 的舍入误差，而不是严格的零），
//   组装层矩阵（对角矩阵 + 舍入误差），交给 options.eig 指定的本征求解器，之后所有界面块、传播块都按稠密矩阵
//   参加递推。入射、出射介质是半无限的，仍用解析模式，只是同样写成稠密矩阵。许多 RCWA 程序就是这样统一处理
//   所有层的；这种模式既是衡量快速路径省了多少时间的基准，也用来考察本征求解器在“近对角矩阵”上的表现。
//   有两级的 kx² 相等时，均匀层的层矩阵里这两级的本征值相等——恰好正入射（θ = 0，kx_m = −kx_{−m}）和 Littrow 入射
//   （nI·sin θ = m·λ/(2Λ)）都是这样。迭代之后这一对本征值往往只在 1e-31 ~ 1e-35 的舍入上有差别；eig_qr / eig_lr 的
//   回代把这样小的差当作零处理（见 eig_qr.hpp 里回代一节的说明），给出两个线性无关的本征向量。实测 300 个随机结构
//   （1 ~ 5 层，至少一层均匀膜，M = 4 ~ 12）在恰好正入射和恰好 Littrow 入射时，这种模式与快速路径的各级效率
//   相差不超过 1e-12。换用 custom_eig 时，自己的本征求解器也要能处理这种情形：重合的一对本征值如果得到两个几乎
//   平行的本征向量，这一层的 W 实际上是奇异的，而本征对残差和线性求解的后向误差仍然是 1e-16 量级，solve() 察觉不到。
//   （默认的快速路径下，各段折射率全都相同的“光栅层”是同一种情形：它实际上是均匀层，却要做本征分解。）
//
// options.eig / custom_eig：每个需要本征分解的层都用它；custom_eig 若设置则取代 options.eig。
//   solve() 不检查 custom_eig 返回的本征对的残差，所以精度较低的本征求解器（例如单精度的 GPU 后端）可以直接接入，
//   换回双精度只需要改回调里面的一处；结果的精度随本征求解器而定，打开 diagnostics 后 max_eig_residual 如实反映。
//   参考量级（四脊大周期光栅加 20 层膜，即 examples::ridges_on_mirror，TE/TM，61 ~ 601 阶；本征分解用
//   gpu/cuda_backend.hpp 的单精度模式 Precision::Single，全程 float 运算）：本征对残差 7e-7 ~ 2e-6，
//   各级效率与双精度结果相差 2e-7 ~ 5e-5，|R + T − 1| 最大 4e-5。
// options.linear：所有稠密线性求解用的方法。
// options.parallel_layers：各层的本征分解互不相干，打开后用 parallel_for 分给线程池，结果与串行逐位相同。
//
// 本函数可重入，可以被多个线程同时调用；结果与 set_num_threads 设定的线程数无关（逐位相同）。
namespace detail {
inline Result solve_impl(const Stack& stack, double wavelength, double theta_deg, int M,
                         Polarization pol, const Options& options, PlanarFieldCapture* fields) {
    namespace dt = detail;
    const auto t_start = dt::Clock::now();

    // ---- 0. 检查输入
    dt::check_setup(stack.period, wavelength, stack.nI, theta_deg, M, pol);
    dt::check_index(stack.nII);
    for (const Layer& layer : stack.layers) dt::check_layer(layer, stack.period);
    if (options.linear != LinearSolver::LUPartial && options.linear != LinearSolver::LUNoPivot &&
        options.linear != LinearSolver::QRHouseholder)
        throw std::invalid_argument("rcwa: unknown linear solver");
    if (options.eig != EigMethod::QR && options.eig != EigMethod::LR)
        throw std::invalid_argument("rcwa: unknown eigen-solver");

    const int n = 2 * M + 1;
    const int layer_count = static_cast<int>(stack.layers.size());
    const double k0 = 2.0 * kPi / wavelength;
    const bool fast = options.homogeneous_fast_path;

    Result result;
    result.orders.resize(static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j) result.orders[j] = j - M;
    const Vector kx = dt::transverse_wavevectors(stack.nI.real(), theta_deg, wavelength, stack.period, M);

    dt::Work work;
    work.linear = options.linear;
    work.diagnostics = options.diagnostics;
    if (fields) {
        *fields = PlanarFieldCapture{};
        fields->materials.resize(static_cast<std::size_t>(layer_count));
        fields->a_plus.resize(static_cast<std::size_t>(layer_count));
        fields->a_minus.resize(static_cast<std::size_t>(layer_count));
    }

    try {
        // ---- 1. 各区域的模式。区域 0 是入射介质，区域 j+1 是第 j 层，最后一个区域是出射介质。
        std::vector<dt::Modes> modes(static_cast<std::size_t>(layer_count) + 2);
        modes.front() = dt::analytic_modes(stack.nI, kx, pol, !fast);
        modes.back() = dt::analytic_modes(stack.nII, kx, pol, !fast);
        std::vector<int> eig_layers;                    // 需要做本征分解的层
        // 纯膜系（快速路径下没有任何层要做本征分解）：所有的块都是对角的，各级互不耦合，只有入射级（下标 M）被激发。
        bool pure_films = fast;
        for (const Layer& layer : stack.layers) pure_films = pure_films && layer.homogeneous;
        for (int j = 0; j < layer_count; ++j) {
            const Layer& layer = stack.layers[j];
            if (fast && layer.homogeneous) {
                modes[j + 1] = dt::analytic_modes(layer.n, kx, pol, false, true);
                if (fields) {
                    fields->materials[j].analytic = true;
                    fields->materials[j].epsilon = layer.n * layer.n;
                }
            } else {
                eig_layers.push_back(j);
            }
        }
        const int eig_count = static_cast<int>(eig_layers.size());

        // 1a. 组装层矩阵
        std::vector<dt::LayerOperator> ops(static_cast<std::size_t>(eig_count));
        for (int i = 0; i < eig_count; ++i) {
            ops[i] = dt::assemble_layer(stack.layers[eig_layers[i]], stack.period, kx, M, pol, work,
                                       fields ? &fields->materials[eig_layers[i]] : nullptr);
            if (!ops[i].a.all_finite()) throw dt::NumericalFailure{};
        }

        // 1b. 本征分解。各层互不相干，可以并行；计时取这一段的墙上时间。
        std::vector<EigResult> eigs(static_cast<std::size_t>(eig_count));
        const auto decompose = [&](int i) {
            eigs[i] = options.custom_eig ? options.custom_eig(ops[i].a) : stablercwa::eig(ops[i].a, options.eig);
            if (!options.diagnostics) ops[i].a = Matrix();      // 层矩阵用完就释放（诊断还要用它算残差）
        };
        if (eig_count > 0) {                            // 没有要分解的层（纯膜系）时 eig 用时严格为零
            const auto t0 = dt::Clock::now();
            if (options.parallel_layers) parallel_for(0, eig_count, decompose);
            else for (int i = 0; i < eig_count; ++i) decompose(i);
            work.timing.eig += dt::seconds_since(t0);
        }
        result.n_eig = eig_count;
        for (int i = 0; i < eig_count; ++i) {
            const EigResult& e = eigs[i];
            if (e.info != 0) {
                if (result.eig_info == 0) result.eig_info = e.info;
            } else if (static_cast<int>(e.values.size()) != n || e.vectors.rows() != n || e.vectors.cols() != n) {
                throw std::invalid_argument("rcwa: the eigen-solver returned a result of the wrong size");
            }
            if (options.diagnostics) {      // 没收敛的层没有本征向量，残差记为 NaN
                const double res = e.info == 0 ? eig_residual(ops[i].a, e) : std::numeric_limits<double>::quiet_NaN();
                result.max_eig_residual = dt::max_keep_nan(result.max_eig_residual, res);
            }
        }
        if (result.eig_info != 0) throw dt::NumericalFailure{};

        // 1c. q = sqrt(本征值)，W = 本征向量，V = W·q（TE）或 ⟦1/ε⟧·W·q（TM）
        for (int i = 0; i < eig_count; ++i) {
            dt::Modes& m = modes[eig_layers[i] + 1];
            m.q.resize(static_cast<std::size_t>(n));
            for (int j = 0; j < n; ++j) m.q[j] = dt::branch(std::sqrt(eigs[i].values[j]));
            m.W = std::move(eigs[i].vectors);
            m.finite_zero_basis = true;
            Vector basis_q = m.q;
            for (cplx& q : basis_q) if (q == cplx{}) q = 1.0;
            if (pol == Polarization::TE) m.V = scale_columns(m.W, basis_q);
            else m.V = scale_columns(dt::multiply(ops[i].inv_eps, m.W, work), basis_q);
            eigs[i] = EigResult();
            ops[i] = dt::LayerOperator();
        }

        // ---- 2. 自下而上叠加各块。块 2j 是区域 j 与 j+1 之间的界面，块 2j+1 是第 j 层内的传播。
        // 块在用到时才构造，叠完只留下 g（自上而下推振幅时要用）和当前的总反射 r。
        const int block_count = 2 * layer_count + 1;
        std::vector<dt::Op> g(static_cast<std::size_t>(block_count));
        std::vector<dt::Op> r_below;                      // 仅近场路径保留各层底面以下的总反射
        if (fields) r_below.resize(static_cast<std::size_t>(layer_count));
        dt::Op r;                                        // 出射介质里没有向上传的波：最底下的总反射为零
        if (!fast) r = dt::dense_op(Matrix(n, n));
        for (int b = block_count - 1; b >= 1; --b) {
            const int j = b / 2;
            dt::SBlock block;
            if (b % 2 == 0) {
                block = dt::interface_block(modes[j], modes[j + 1], work);
                if (!fields) {
                    modes[j + 1].W = Matrix();           // 效率-only 路径仍立即释放模式
                    modes[j + 1].V = Matrix();
                }
            } else {
                if (fields) r_below[j] = r;
                block = dt::propagation_block(modes[j + 1].q, stack.layers[j].d, k0, !fast, modes[j + 1].finite_zero_basis);
            }
            dt::Seam seam = dt::stack_on(std::move(block), r, n, work);
            g[b] = std::move(seam.g);
            r = std::move(seam.r);
        }

        // ---- 3. 最上面的界面块，只算入射级次（下标 M）对应的那一列：
        //     down = (I − Rdu·r)^{-1}·Tdd·e，    R = Rud·e + Tuu·r·down，
        // 然后自上而下逐块乘 g，得到出射介质里的下行振幅 T。
        {
            using K = dt::Op::Kind;
            dt::TopBlock top = dt::top_interface_block(modes[0], modes[1], M, work);
            dt::Op k = dt::op_mul(top.Rdu, r, work);
            Vector down = std::move(top.tdd);
            if (k.kind == K::Diag) {
                for (int j = 0; j < n; ++j) down[j] = down[j] / (1.0 - k.d[j]);
            } else if (k.kind == K::Dense) {
                Matrix rhs(n, 1);
                for (int j = 0; j < n; ++j) rhs(j, 0) = down[j];
                down = dt::linear_solve(dt::identity_minus(std::move(k.m)), rhs, work).column(0);
            }
            result.R = std::move(top.rud);
            const Vector back = dt::op_apply(top.Tuu, dt::op_apply(r, down));
            for (int j = 0; j < n; ++j) result.R[j] += back[j];
            for (int b = 1; b < block_count; ++b) {
                const int j = b / 2;
                if (fields && b % 2 != 0) {
                    fields->a_plus[j] = down;
                    if (pure_films) dt::clear_unexcited(fields->a_plus[j], M);
                }
                down = dt::op_apply(g[b], down);
                if (fields && b % 2 != 0) {
                    // 已通过该层散射块到达层底，再应用底面以下的反射；不逆传播增长指数。
                    Vector bottom = down;
                    if (pure_films) dt::clear_unexcited(bottom, M);
                    fields->a_minus[j] = dt::op_apply(r_below[j], bottom);
                    if (pure_films) dt::clear_unexcited(fields->a_minus[j], M);
                }
            }
            result.T = std::move(down);
        }
        // 纯膜系不产生衍射：入射级之外的各级没有被激发，振幅严格为零。它们各自的（标量）递推里若出现了 0/0 或 0·Inf
        // （这一级恰好在某一层里掠射，或者在界面两侧同时掠射），并不是求解失败，这里把这些非有限的值还原成零。
        if (pure_films)
            for (int j = 0; j < n; ++j) {
                if (j == M) continue;
                if (!dt::is_finite(result.R[j])) result.R[j] = cplx{};
                if (!dt::is_finite(result.T[j])) result.T[j] = cplx{};
            }
        // 中途出现过 NaN/Inf（例如出射介质里恰好有一级 kz = 0，上面又紧挨着光栅层）：整个结果按失败处理
        for (int j = 0; j < n; ++j)
            if (!dt::is_finite(result.R[j]) || !dt::is_finite(result.T[j])) throw dt::NumericalFailure{};

        // ---- 4. 衍射效率：各级沿 z 方向的能流与入射能流之比。倏逝级的 kz 是纯虚数，效率为零
        //      （有的标准库对负实数开方会留下 1e-17 量级的实部，这时倏逝级的效率是 1e-19 量级而不是严格的零）。
        const Vector& qI = modes.front().q;
        const Vector& qII = modes.back().q;
        // 入射能流用入射级自己的 kz（= sqrt(nI² − kx0²)，与场里用的是同一个数），不用 nI·cos θ：两者在掠入射时
        // 相差约 ε/cos²θ（θ = 89.9999° 时 1e-5），用后者会让 R + T 偏离 1 这么多。
        const cplx den{-k0 * qI[M].imag(), k0 * qI[M].real()};
        result.DEr.resize(static_cast<std::size_t>(n));
        result.DEt.resize(static_cast<std::size_t>(n));
        for (int j = 0; j < n; ++j) {
            const cplx kIz{-k0 * qI[j].imag(), k0 * qI[j].real()};         // kz = i·k0·q
            const cplx kIIz{-k0 * qII[j].imag(), k0 * qII[j].real()};
            result.DEr[j] = std::norm(result.R[j]) * (kIz / den).real();
            const cplx factor = pol == Polarization::TE
                                    ? kIIz / den
                                    : (kIIz / (stack.nII * stack.nII)) / (den / (stack.nI * stack.nI));
            result.DEt[j] = std::norm(result.T[j]) * factor.real();
            // 振幅有限、效率却溢出或成了 NaN（折射率、波长取了极端的数值）：同样按失败处理，不留下“一部分是 Inf/NaN”的结果
            if (!std::isfinite(result.DEr[j]) || !std::isfinite(result.DEt[j])) throw dt::NumericalFailure{};
        }
        if (fields) {
            fields->modes = std::move(modes);
            fields->kx = kx;
            fields->reflected = result.R;
            fields->transmitted = result.T;
            fields->available = dt::valid_capture(*fields);
            if (!fields->available) *fields = PlanarFieldCapture{};
        }
    } catch (const dt::NumericalFailure&) {
        dt::fill_nan(result, n);
        if (fields) *fields = PlanarFieldCapture{};
    }

    result.n_dense_solve = work.n_solve;
    result.max_solve_backward_error = work.max_backward_error;
    result.timing = work.timing;
    result.timing.total = dt::seconds_since(t_start);
    return result;
}
}  // namespace detail

inline Result solve(const Stack& stack, double wavelength, double theta_deg, int M,
                    Polarization pol = Polarization::TE, const Options& options = {}) {
    return detail::solve_impl(stack, wavelength, theta_deg, M, pol, options, nullptr);
}

// 扫描：对每个点独立调用 solve()，用 parallel_for 把各点分给线程池（set_num_threads 设定线程数）。
// 线程池不嵌套，所以每次求解内部自动串行；结果的顺序与 points 相同，且与线程数无关（逐位相同）。
inline std::vector<Result> sweep(const Stack& stack, const std::vector<Point>& points, int M,
                                 Polarization pol = Polarization::TE, const Options& options = {}) {
    std::vector<Result> results(points.size());
    parallel_for(0, static_cast<int>(points.size()), [&](int i) {
        results[i] = solve(stack, points[i].wavelength, points[i].theta_deg, M, pol, options);
    });
    return results;
}

// 光栅层的层矩阵（TE：Kx² − ⟦ε⟧；TM：⟦1/ε⟧^{-1}(Kx ⟦ε⟧^{-1} Kx − I)），供演示和测试直接取用。
// 它就是 solve() 交给本征求解器的那个矩阵（TM 里的两次求解这里固定用选主元的 LU，即 solve() 的默认设置）。
// 传入均匀层时，按“只有一段的光栅层”用同一套通用公式构造，得到“对角矩阵 + 舍入误差”，
// 与 solve() 在 homogeneous_fast_path = false 时用的矩阵相同。
// TM 时 ⟦ε⟧ 或 ⟦1/ε⟧ 奇异则抛 std::runtime_error。
inline Matrix layer_matrix(const Layer& layer, double period, double wavelength, cplx nI, double theta_deg,
                           int M, Polarization pol) {
    namespace dt = detail;
    dt::check_setup(period, wavelength, nI, theta_deg, M, pol);
    dt::check_layer(layer, period);
    const Vector kx = dt::transverse_wavevectors(nI.real(), theta_deg, wavelength, period, M);
    dt::Work work;
    try {
        return dt::assemble_layer(layer, period, kx, M, pol, work).a;
    } catch (const dt::NumericalFailure&) {
        throw std::runtime_error("rcwa::layer_matrix: singular permittivity matrix");
    }
}

}  // namespace rcwa
}  // namespace stablercwa
