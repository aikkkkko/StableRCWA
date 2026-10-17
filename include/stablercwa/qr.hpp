// stablercwa/qr.hpp —— Householder QR 分解、解线性方程组与最小二乘
//
//   QR f = qr_factor(A);            // A = Q·R，A 为 m×n（m ≥ n）
//   Matrix X = qr_solve(f, B);      // 方阵：解 A X = B；m > n：使 ||A X − B|| 最小的 X
//   Matrix Q = qr_q(f), R = qr_r(f);
//
// 约定与 LAPACK 的 xGEQRF 相同：Q = H_0 H_1 … H_{n-1}，H_k = I − tau_k·v_k·v_k^H，
// v_k 的前 k 个分量为 0、第 k 个分量为 1（不存），其余分量存在 qr 第 k 列的对角线以下；
// tau_k 是复数，R 的对角元是实数。每个 H_k 都是酉矩阵，所以整个过程不放大误差。
//
// 与 lu.hpp 一样按面板推进：先把一个面板的几列分解完，再让右边每一列依次乘上这几个反射。
// 每一列受到的运算及顺序与逐列的教科书算法完全相同，线程数不影响结果。
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

#include "lu.hpp"       // 复用其中的列运算内核与上三角回代
#include "matrix.hpp"

namespace stablercwa {

struct QR {
    Matrix qr;    // R 在上三角；第 k 个 Householder 向量在第 k 列对角线以下（v_k 首元为 1 不存）
    Vector tau;   // 各反射的系数 tau_k
};

namespace detail {
namespace dense {

inline constexpr int kQrBlock = 32;

// x^H y = Σ conj(x_i)·y_i。分成 4 路各自累加再合并：求和顺序在源码里写死，
// 既打断了加法的依赖链，又不依赖编译器的浮点重排，结果可复现。
// 每一项先单独算出来再累加（两条语句），理由同下面的 nrm2。
inline cplx dot_conj(const cplx* x_, const cplx* y_, int len) noexcept {
    const double* x = as_real(x_);
    const double* y = as_real(y_);
    double sr[4] = {0.0, 0.0, 0.0, 0.0}, si[4] = {0.0, 0.0, 0.0, 0.0};
    int i = 0;
    for (; i + 4 <= len; i += 4) {
        for (int u = 0; u < 4; ++u) {
            const int r = 2 * (i + u), s = r + 1;
            const double pr = x[r] * y[r] + x[s] * y[s];
            const double pi = x[r] * y[s] - x[s] * y[r];
            sr[u] += pr;
            si[u] += pi;
        }
    }
    for (int u = 0; i < len; ++i, ++u) {
        const int r = 2 * i, s = r + 1;
        const double pr = x[r] * y[r] + x[s] * y[s];
        const double pi = x[r] * y[s] - x[s] * y[r];
        sr[u] += pr;
        si[u] += pi;
    }
    return {(sr[0] + sr[1]) + (sr[2] + sr[3]), (si[0] + si[1]) + (si[2] + si[3])};
}

// ||x||_2。量级正常时直接求平方和；平方和可能上溢或下溢时改用带缩放的求和。
inline double nrm2(const cplx* x_, int len) noexcept {
    const double* x = as_real(x_);
    // 平方与累加分成两条语句写，免得被并成一条乘加指令：在累加循环里并不并由编译器逐处决定，
    // 并了与没并差一次舍入，结果就会随优化级别变化。
    double s[4] = {0.0, 0.0, 0.0, 0.0};
    int i = 0;
    for (; i + 2 <= len; i += 2) {
        for (int u = 0; u < 4; ++u) {
            const double sq = x[2 * i + u] * x[2 * i + u];
            s[u] += sq;
        }
    }
    for (int u = 0; u < 2 && i < len; ++u) {
        const double sq = x[2 * i + u] * x[2 * i + u];
        s[u] += sq;
    }
    const double sum = (s[0] + s[1]) + (s[2] + s[3]);
    if (sum >= 1e-280 && sum <= 1e280) return std::sqrt(sum);
    double scale = 0.0, ssq = 1.0;
    for (int k = 0; k < 2 * len; ++k) {
        const double v = std::abs(x[k]);
        if (v == 0.0) continue;
        if (!(v <= scale)) {                 // v 更大（或是 NaN，让它传出去）
            ssq = 1.0 + ssq * (scale / v) * (scale / v);
            scale = v;
        } else {
            ssq += (v / scale) * (v / scale);
        }
    }
    return scale * std::sqrt(ssq);
}

// 构造 Householder 反射 H = I − tau·v·v^H（v 首元为 1），使 H^H·(alpha; x) = (beta; 0)，beta 为实数。
// 返回 tau；alpha 被改写成 beta，x 被改写成 v 的其余分量。做法同 LAPACK 的 ZLARFG：
// beta 取与 Re(alpha) 相反的符号，这样 alpha − beta 不会相消；向量极小时先放大再算。
inline cplx make_householder(cplx& alpha, cplx* x, int len) noexcept {
    double xnorm = nrm2(x, len);
    double ar = alpha.real(), ai = alpha.imag();
    if (xnorm == 0.0 && ai == 0.0) return cplx{};        // 已经是 (实数; 0)，H = I
    // 三个数的模用两次二元 hypot：有的标准库的三元 hypot 是直接平方求和，极小的数会下溢成 0。
    double beta = -std::copysign(std::hypot(std::hypot(ar, ai), xnorm), ar);
    const double tiny = (std::numeric_limits<double>::min)() / kEps;  // 低于它时 1/(alpha−beta) 会丢精度
    const double big = 1.0 / tiny;
    int rescaled = 0;
    if (std::abs(beta) < tiny) {
        double* p = as_real(x);
        do {
            ++rescaled;
            for (int k = 0; k < 2 * len; ++k) p[k] *= big;
            beta *= big;
            ar *= big;
            ai *= big;
        } while (std::abs(beta) < tiny && rescaled < 20);
        xnorm = nrm2(x, len);
        beta = -std::copysign(std::hypot(std::hypot(ar, ai), xnorm), ar);
    }
    const cplx tau{(beta - ar) / beta, -ai / beta};
    const cplx scale = cdiv(cplx{1.0, 0.0}, cplx{ar - beta, ai});
    for (int i = 0; i < len; ++i) x[i] = cmul(x[i], scale);
    for (int k = 0; k < rescaled; ++k) beta *= tiny;
    alpha = cplx{beta, 0.0};
    return tau;
}

// c ← (I − tau·v·v^H)·c，v[0] 视为 1，len 是 v 和 c 的长度。传 conj(tau_k) 就是乘 H_k^H。
inline void apply_reflector(const cplx* v, cplx tau, cplx* c, int len) noexcept {
    if (tau == cplx{}) return;
    const cplx w = cmul(tau, c[0] + dot_conj(v + 1, c + 1, len - 1));
    c[0] -= w;
    sub_col1<false>(c + 1, v + 1, w, len - 1);
}

}  // namespace dense
}  // namespace detail

// A = Q·R，要求 rows >= cols。
inline QR qr_factor(Matrix a) {
    using namespace detail::dense;
    const int m = a.rows(), n = a.cols();
    if (m < n) throw std::invalid_argument("qr_factor: requires rows >= cols");
    QR f;
    f.tau.assign(static_cast<std::size_t>(n), cplx{});
    for (int ks = 0; ks < n; ks += kQrBlock) {
        const int ke = (std::min)(ks + kQrBlock, n);
        // 1) 面板：第 j 列先乘上本面板里已有的反射，再对它的对角线以下部分造新的反射。
        for (int j = ks; j < ke; ++j) {
            cplx* c = a.col(j);
            for (int k = ks; k < j; ++k) apply_reflector(a.col(k) + k, std::conj(f.tau[k]), c + k, m - k);
            f.tau[j] = make_householder(c[j], c + j + 1, m - j - 1);
        }
        // 2) 右边每一列依次乘 H_ks^H … H_{ke-1}^H。列与列互不相干，可以并行。
        for_columns(ke, n, 2LL * (ke - ks) * (m - ks), [&](int lo, int hi) {
            for (int j = lo; j < hi; ++j) {
                cplx* c = a.col(j);
                for (int k = ks; k < ke; ++k)
                    apply_reflector(a.col(k) + k, std::conj(f.tau[k]), c + k, m - k);
            }
        });
    }
    f.qr = std::move(a);
    return f;
}

// 显式的 Q（rows × cols，各列标准正交）：把 H_0 … H_{n-1} 依次作用到单位阵的前 n 列上。
inline Matrix qr_q(const QR& f) {
    using namespace detail::dense;
    const int m = f.qr.rows(), n = f.qr.cols();
    if (static_cast<int>(f.tau.size()) != n || m < n) throw std::invalid_argument("qr_q: malformed QR");
    Matrix q(m, n);
    for (int j = 0; j < n; ++j) q(j, j) = 1.0;
    // Q e_j = H_0 … H_j e_j（k > j 的反射不改变 e_j），所以从后往前乘，且第 j 列只需要 k <= j 的反射。
    for (int ke = n; ke > 0; ke -= (std::min)(ke, kQrBlock)) {
        const int ks = ke - (std::min)(ke, kQrBlock);
        for_columns(ks, n, 2LL * (ke - ks) * (m - ks), [&](int lo, int hi) {
            for (int j = lo; j < hi; ++j) {
                cplx* c = q.col(j);
                for (int k = (std::min)(ke - 1, j); k >= ks; --k)
                    apply_reflector(f.qr.col(k) + k, f.tau[k], c + k, m - k);
            }
        });
    }
    return q;
}

// R（cols × cols 上三角）。
inline Matrix qr_r(const QR& f) {
    const int n = f.qr.cols();
    if (f.qr.rows() < n) throw std::invalid_argument("qr_r: malformed QR");
    Matrix r(n, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i <= j; ++i) r(i, j) = f.qr(i, j);
    return r;
}

// 方阵时解 A X = B；rows > cols 时给出最小二乘解。做法：X = R^{-1}·(Q^H B 的前 n 行)。
// R 的对角线上有精确的零（A 列不满秩）时抛 std::runtime_error。
inline Matrix qr_solve(const QR& f, Matrix b) {
    using namespace detail::dense;
    const int m = f.qr.rows(), n = f.qr.cols();
    if (static_cast<int>(f.tau.size()) != n || m < n) throw std::invalid_argument("qr_solve: malformed QR");
    if (b.rows() != m) throw std::invalid_argument("qr_solve: right-hand side has wrong number of rows");
    for (int k = 0; k < n; ++k)
        if (f.qr(k, k) == cplx{}) throw std::runtime_error("qr_solve: R is singular (zero diagonal)");
    const int nrhs = b.cols();
    // 1) B ← Q^H B = H_{n-1}^H … H_0^H B
    for (int ks = 0; ks < n; ks += kQrBlock) {
        const int ke = (std::min)(ks + kQrBlock, n);
        for_columns(0, nrhs, 2LL * (ke - ks) * (m - ks), [&](int lo, int hi) {
            for (int j = lo; j < hi; ++j) {
                cplx* c = b.col(j);
                for (int k = ks; k < ke; ++k)
                    apply_reflector(f.qr.col(k) + k, std::conj(f.tau[k]), c + k, m - k);
            }
        });
    }
    // 2) 回代 R X = (Q^H B)(0:n, :)
    for (int ke = n; ke > 0; ke -= (std::min)(ke, kLuBlock)) {
        const int ks = ke - (std::min)(ke, kLuBlock);
        for_columns(0, nrhs, static_cast<long long>(ke - ks) * ke, [&](int lo, int hi) {
            for (int j = lo; j < hi; ++j) back_substitute_column(f.qr, ks, ke, b.col(j));
        });
    }
    if (m == n) return b;
    Matrix x(n, nrhs);
    for (int j = 0; j < nrhs; ++j) std::copy(b.col(j), b.col(j) + n, x.col(j));
    return x;
}

inline Matrix solve_qr(const Matrix& a, const Matrix& b) {
    if (b.rows() != a.rows()) throw std::invalid_argument("solve_qr: right-hand side has wrong number of rows");
    return qr_solve(qr_factor(a), b);
}

}  // namespace stablercwa
