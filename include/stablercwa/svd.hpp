// stablercwa/svd.hpp —— 奇异值（单边 Jacobi）、2-范数、2-范数条件数
//
//   RealVector s = singular_values(A);   // 降序
//   double kappa = cond2(A);             // σ_max / σ_min
//
// 用途是量“累积变换有多病态”，条件数可达 1e14，所以小奇异值也要尽量算准。
// 通过 A^H A 的本征值来算会把条件数平方，小奇异值全部淹没在舍入误差里，不能用。这里的做法：
//   1) 把各行按大小排序，再做带列主元的 Householder QR：A·P = Q·R。奇异值只与 R 有关；
//   2) 对 X = R^H 的各列做单边 Jacobi（Hestenes）：每次取两列，用一个 2×2 的酉旋转把它们变成正交，
//      一遍一遍地扫，直到任意两列都正交到舍入误差的水平。这时各列的 2-范数就是奇异值。
// Jacobi 旋转只在两列之间做酉组合，量级悬殊的列不会互相污染；预先做带主元的 QR，
// 则让 X = (条件数不大的三角阵) × (对角阵)，正是单边 Jacobi 能保证相对精度的形式。
//
// 精度：
//   · 行、列分级的矩阵 D1·B·D2（D1、D2 对角，B 条件数不大）：每个奇异值都有接近机器精度的相对精度，
//     哪怕 σ_min/σ_max 小到 1e-30。不选主元的消元得到的 L、U 因子，以及消元型变换累积出来的矩阵，
//     病态往往主要来自这种行、列量级的悬殊；
//   · 一般矩阵：小奇异值的相对误差约为 eps × “把行、列量级扣除之后剩下的条件数”。完全没有分级结构时
//     它就是 cond2·eps（与常规 SVD 算法相同）：1e14 时还有两位有效数字，超过 1e16 只能读作“非常大”。
// 代价：O(n³)，Jacobi 一般扫 5～10 遍；n = 400 约需 0.1～0.6 秒（单线程），不并行。
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "matrix.hpp"
#include "qr.hpp"       // 复用 Householder 反射与列运算内核

namespace stablercwa {

namespace detail {
namespace dense {

inline constexpr int kJacobiMaxSweeps = 60;
inline constexpr double kJacobiFloor = 1e-290;   // 相对最大元素小于它的列按零处理

// 把各行按“行内最大模”从大到小重排。配合列主元，Householder QR 对每一行的相对误差都小
// （Cox–Higham 的结论），行分级的矩阵才不会在这一步丢掉小奇异值。
// 模用 std::abs 而不是 |z|²：平方在 1e-154 以下就下溢成 0，极小的那些行会排不出先后。
inline void sort_rows_by_size(Matrix& a) {
    const int m = a.rows(), n = a.cols();
    RealVector size(static_cast<std::size_t>(m), 0.0);
    for (int j = 0; j < n; ++j) {
        const cplx* c = a.col(j);
        for (int i = 0; i < m; ++i) size[i] = (std::max)(size[i], std::abs(c[i]));
    }
    std::vector<int> order(static_cast<std::size_t>(m));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int p, int q) { return size[p] > size[q]; });
    Vector tmp(static_cast<std::size_t>(m));
    for (int j = 0; j < n; ++j) {
        cplx* c = a.col(j);
        for (int i = 0; i < m; ++i) tmp[i] = c[order[i]];
        std::copy(tmp.begin(), tmp.end(), c);
    }
}

// 带列主元的 Householder QR（Businger–Golub），只要 R：返回时 a 的上三角就是 R。
// 每一步把“剩余部分范数最大”的列换到前面。范数每步重新计算，不用递推公式，省掉递推失准时的补救。
inline void qr_colpivot_r(Matrix& a) {
    const int m = a.rows(), n = a.cols();
    for (int k = 0; k < n; ++k) {
        int best = k;
        double best_norm = -1.0;
        for (int j = k; j < n; ++j) {
            const double v = nrm2(a.col(j) + k, m - k);
            if (v > best_norm) { best_norm = v; best = j; }
        }
        if (best != k) std::swap_ranges(a.col(k), a.col(k) + m, a.col(best));
        cplx* v = a.col(k) + k;
        const cplx tau = make_householder(v[0], v + 1, m - k - 1);
        for (int j = k + 1; j < n; ++j) apply_reflector(v, std::conj(tau), a.col(j) + k, m - k);
    }
}

// (x, y) ← (c·x − conj(sp)·y, sp·x + c·y)，c 为实数、|sp|² + c² = 1：两列之间的酉旋转。
inline void rotate_columns(cplx* x_, cplx* y_, double c, cplx sp, int len) noexcept {
    double* x = as_real(x_);
    double* y = as_real(y_);
    const double pr = sp.real(), pi = sp.imag();
    for (int i = 0; i < len; ++i) {
        const int r = 2 * i, s = r + 1;
        const double xr = x[r], xi = x[s], yr = y[r], yi = y[s];
        x[r] = c * xr - pr * yr - pi * yi;
        x[s] = c * xi - pr * yi + pi * yr;
        y[r] = c * yr + pr * xr - pi * xi;
        y[s] = c * yi + pr * xi + pi * xr;
    }
}

// 对 x 的各列做单边 Jacobi，直到两两正交；返回扫描遍数，未收敛时返回 -1。
// 约定 x 已经缩放到最大元素为 1 的量级。列的顺序用下标数组 idx 表示：每一遍里把范数大的列排在前面
// （de Rijk 的做法，收敛更快），只换下标不搬数据。
inline int jacobi_orthogonalize(Matrix& x, std::vector<int>& idx) {
    const int n = x.cols(), len = x.rows();
    // “已经正交”的门槛。算出来的余弦本身带有几个 eps 的舍入噪声，门槛再低就会在噪声上空转。
    const double tol = (std::max)(4.0, std::sqrt(static_cast<double>(n))) * kEps;
    RealVector nrm(static_cast<std::size_t>(n));        // 各列当前的 2-范数（按 idx 的位置存放）
    Vector scaled;                                      // 两列都极小时用的缩放副本
    for (int sweep = 1; sweep <= kJacobiMaxSweeps; ++sweep) {
        for (int p = 0; p < n; ++p) {                   // 每遍重新算范数，不让递推误差累积
            cplx* xp = x.col(idx[p]);
            nrm[p] = nrm2(xp, len);
            if (nrm[p] < kJacobiFloor) {                // 小到这个程度已无精度可言，按零列处理
                std::fill(xp, xp + len, cplx{});
                nrm[p] = 0.0;
            }
        }
        bool rotated = false;
        for (int p = 0; p + 1 < n; ++p) {
            int big = p;
            for (int q = p + 1; q < n; ++q)
                if (nrm[q] > nrm[big]) big = q;
            if (big != p) {
                std::swap(idx[p], idx[big]);
                std::swap(nrm[p], nrm[big]);
            }
            cplx* xp = x.col(idx[p]);
            for (int q = p + 1; q < n; ++q) {
                const double a = nrm[p], b = nrm[q];
                if (a == 0.0 || b == 0.0) continue;                  // 零列与任何列都正交
                cplx* xq = x.col(idx[q]);
                // g = x_p^H x_q，cosine = |g| / (a·b) 是两列夹角的余弦。
                cplx g;
                double cosine;
                if (a * b >= 1e-250) {
                    g = dot_conj(xp, xq, len);
                    cosine = (std::abs(g) / a) / b;
                } else {                                             // 乘积会下溢：先把两列各自放大到 1 的量级
                    scaled.resize(2 * static_cast<std::size_t>(len));
                    const int ea = std::ilogb(a), eb = std::ilogb(b);
                    for (int i = 0; i < len; ++i) {
                        scaled[i] = cplx{std::ldexp(xp[i].real(), -ea), std::ldexp(xp[i].imag(), -ea)};
                        scaled[len + i] = cplx{std::ldexp(xq[i].real(), -eb), std::ldexp(xq[i].imag(), -eb)};
                    }
                    g = dot_conj(scaled.data(), scaled.data() + len, len);
                    cosine = (std::abs(g) / std::ldexp(a, -ea)) / std::ldexp(b, -eb);
                }
                if (!(cosine > tol)) continue;
                // 把 2×2 的 Gram 矩阵 [[a², g], [conj(g), b²]] 对角化。tan 取模较小的那个根（转角不超过 45°）：
                //   ζ = (b² − a²) / (2|g|)，t = sign(ζ) / (|ζ| + sqrt(1 + ζ²))。两列量级悬殊时 |ζ| 极大，t ≈ 1/(2ζ)。
                const double r = b / a - a / b;                      // = 2·ζ·cosine
                double t;
                if (std::abs(r) > 1e100) {
                    t = cosine / r;
                } else {
                    const double zeta = r / (2.0 * cosine);
                    t = std::copysign(1.0, zeta) / (std::abs(zeta) + std::hypot(1.0, zeta));
                }
                if (t == 0.0) continue;
                const double c = 1.0 / std::hypot(1.0, t);
                const double s = c * t;
                const double gabs = std::abs(g);
                rotate_columns(xp, xq, c, cplx{s * (g.real() / gabs), s * (g.imag() / gabs)}, len);
                rotated = true;
                // 旋转后的范数：a'² = a² − t|g|，b'² = b² + t|g|。相消严重时直接重算。
                const double fa = 1.0 - (t * (b / a)) * cosine, fb = 1.0 + (t * (a / b)) * cosine;
                nrm[p] = fa > 1e-4 ? a * std::sqrt(fa) : nrm2(xp, len);
                nrm[q] = fb > 1e-4 ? b * std::sqrt(fb) : nrm2(xq, len);
            }
        }
        if (!rotated) return sweep;
    }
    return -1;
}

}  // namespace dense
}  // namespace detail

// 全部奇异值，降序排列，个数为 min(rows, cols)。
// 输入含 NaN/Inf 时返回全 NaN；Jacobi 迭代在 60 遍内不收敛（实际不会发生）时抛 std::runtime_error。
// 比 σ_max 小 290 个数量级以上的奇异值已经超出 double 能可靠运算的范围，按 0 给出。
inline RealVector singular_values(Matrix a) {
    using namespace detail::dense;
    if (a.rows() < a.cols()) a = adjoint(a);          // A 与 A^H 的奇异值相同；下面总假定 rows >= cols
    const int m = a.rows(), n = a.cols();
    RealVector sigma(static_cast<std::size_t>(n), 0.0);
    if (n == 0) return sigma;
    if (!a.all_finite()) {
        std::fill(sigma.begin(), sigma.end(), std::numeric_limits<double>::quiet_NaN());
        return sigma;
    }
    const double amax = max_abs(a);
    if (amax == 0.0) return sigma;
    // 整体乘 2 的幂，使最大元素落在 [1, 2)：之后求平方和不会上溢，结果最后再乘回去（都是精确运算）。
    const int shift = std::ilogb(amax);
    if (shift != 0) {
        double* p = as_real(a.data());
        const std::size_t count = 2 * static_cast<std::size_t>(m) * n;
        for (std::size_t i = 0; i < count; ++i) p[i] = std::ldexp(p[i], -shift);
    }

    sort_rows_by_size(a);
    qr_colpivot_r(a);
    Matrix x(n, n);                                   // X = R^H（下三角）
    for (int j = 0; j < n; ++j)
        for (int i = j; i < n; ++i) x(i, j) = std::conj(a(j, i));
    std::vector<int> idx(static_cast<std::size_t>(n));
    std::iota(idx.begin(), idx.end(), 0);
    if (jacobi_orthogonalize(x, idx) < 0)
        throw std::runtime_error("singular_values: Jacobi sweeps did not converge");

    for (int j = 0; j < n; ++j) sigma[j] = std::ldexp(nrm2(x.col(j), n), shift);
    std::sort(sigma.begin(), sigma.end(), [](double p, double q) { return p > q; });
    return sigma;
}

// 2-范数（最大奇异值）。空矩阵返回 0。
inline double norm_2(const Matrix& a) {
    const RealVector s = singular_values(a);
    return s.empty() ? 0.0 : s.front();
}

// 2-范数条件数 σ_max / σ_min；σ_min 为 0（包括零矩阵）时返回 +inf。矩阵不必是方阵。
// 注意：只有 σ_min 精确为 0（或小到 σ_max 的 1e-290 倍以下、被按 0 给出）才得到 +inf；
// 一般的秩亏矩阵受舍入影响，会得到 1e16 量级或更大的有限值。
inline double cond2(const Matrix& a) {
    if (a.empty()) throw std::invalid_argument("cond2: empty matrix");
    const RealVector s = singular_values(a);
    if (std::isnan(s.back())) return std::numeric_limits<double>::quiet_NaN();
    if (s.back() == 0.0) return std::numeric_limits<double>::infinity();
    return s.front() / s.back();
}

}  // namespace stablercwa
