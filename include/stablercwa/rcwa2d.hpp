// stablercwa/rcwa2d.hpp —— 二维（交叉光栅）RCWA 求解核心：模式、S 矩阵级联、各级 s/p 复振幅与效率
//
//   namespace rcwa = stablercwa::rcwa;
//   rcwa::Stack2D stack;  stack.nI = 1.0;  stack.nII = 1.46;  stack.Lx = 0.8;  stack.Ly = 0.8;
//   stack.layers.push_back(rcwa::Layer2D::patterned(0.3, y_edges, x_profiles));        // 结构层
//   stack.layers.push_back(rcwa::Layer2D::uniform(0.1, 1.9));                           // 均匀膜
//   rcwa::Result2D r = rcwa::solve2d(stack, 1.053, 42.0, -25.0, 3, 3, rcwa::jones_from_angles_2d(90.0));
//   // solve2d 装配各结构层的 Fourier 算子（rcwa2d_operators.hpp），再调用 solve2d_with_operators；已有算子时可直接调用后者：
//   //   std::vector<rcwa::detail2d::LayerOperators2D> ops(stack.layers.size());   // 结构层的 Eps、Cx、Cy（N×N），均匀层的项被忽略
//   //   rcwa::Result2D r2 = rcwa::solve2d_with_operators(stack, 1.053, 42.0, -25.0, 3, 3, rcwa::jones_from_angles_2d(90.0), ops);
//   r.DEr[rcwa::harmonic_index_2d(m, n, 3, 3)] 是 (m, n) 级的反射效率；r.Rs、r.Rp、r.Ts、r.Tp 是各级 s/p 复振幅
//
// 数值约定的权威是 gui/2drcwa/rcwa2d.py（第 195–528 行），合同是 gui/validation/field_reference_2d/CONTRACT_2D.md；
// 谐波下标、Kx、Ky、时间因子见 rcwa2d_types.hpp 文件头。本文件逐段移植 rcwa2d.py 的模式求解与 S 矩阵级联：
//   · 均匀介质（入射区、出射区、均匀层）：W = I，q = [q1; q1]（q1 = branch(−i·kz)），V 是每个级次一个 2×2 块（Qb/q1）；
//     kz 精确为 0 时抛 Rayleigh 异常（见下）；
//   · 结构层：Ei = Eps⁻¹，P、Q 按 rcwa2d.py:403–404 装配，对 P·Q（2N 阶，复，非 Hermite）做一次本征分解，
//     q = branch(sqrt(w2))，V = (Q·W)/q；
//   · 界面 S 块 Tp = ½(W2⁻¹W1 + V2⁻¹V1)、Tm = ½(W2⁻¹W1 − V2⁻¹V1)，Tdd = Tp − Tm·Tp⁻¹·Tm、Rdu = Tm·Tp⁻¹、
//     Rud = −Tp⁻¹·Tm、Tuu = Tp⁻¹；传播块 X = exp(−k0·q·d)，Rdu = Rud = 0；
//   · 自下而上只递推 Rud 一条链（rcwa2d.py 的 _rud2），再自上而下推进入射波（_step2）；star2 没有移植。
//
// 2×2 分块对角快路径（性能的关键）：均匀层的 W、V、传播块、均匀–均匀界面块都存成 (2,2,N)（BlockDiag2：四个长 N 的向量），
// 乘法与求逆退化成 N 维逐元素运算，一段连续的均匀膜整个几乎免费。总反射 Rud（自下而上的那条链）与 S 块因此有三种存法：
//   · 全分块对角：块与下方的 Rud 都是分块对角（均匀–均匀界面、传播块；_rud2/_step2 的第一支）；
//   · 分块对角 × 稠密：下方的 Rud 已经稠密（结构层之上的膜界面与传播块；第二支）。传播块（Rdu = 0）左右乘分块对角只是
//     O(n²) 的行列缩放；
//   · 全稠密：与结构层相邻的界面（第三支），下方的 Rud 若仍是分块对角，则用行列缩放代替稠密乘法。
// 与结构层相邻的稠密界面（以及结构层之上的膜界面）默认不造 S 块，而是直接求“界面连同下方”的总反射
// r' = A₊⁻¹·A₋（A± = (I + r)·Nn ± (I − r)·Mm，见 direct_dense 的注释）：精确算术里与 S 块的
// Rud + Tuu·r·(I − Rdu·r)⁻¹·Tdd 相同，但只要一次 LU，不要 Tpi、Rdu、Tdd、Rud 四个 n×n 乘积；最上面的界面块只解一个向量方程
// K·d = 2e，不求 W⁻¹、V⁻¹。S 块路径（rcwa2d.py 的 s_interface2 / s_interface2_to_homog / _rud2 / _step2 逐段对应）仍然保留，
// 用 detail2d::solve_impl(…, s_blocks = true) 调用，测试里与直接递推、与 Python 互相核对。
// 每个稠密界面自下而上时只分解一次，留给自上而下推进用的是 r' 与 Tm、Nn（或 S 块的 (I − Rdu·r)⁻¹·Tdd），不重算；
// 不要 capture 时，每层的模式矩阵 W、V 在它上方的界面造好后就释放。结构层之下的膜层不占额外内存（分块对角块只有 O(N)）；
// 结构层之上的每个膜界面却要为自上而下的推进各留一份 n×n 的复矩阵（N=625 时约 21 MB/层，N=289 时约 5.5 MB/层，20 层膜约 +330 MB），
// 与 rcwa2d.py 的 rud 列表同样随膜层数增长。
//
// homogeneous_fast_path = false：均匀层按结构层处理——用 Eps = Cx = Cy = n²·I 同样装配 P、Q、做本征分解、走稠密界面；
// 入射区、出射区仍用解析模式（W = I，V 分块对角），与 Python 里“均匀层作为一般层”的路径逐段对应。用来核对快路径，
// 也是衡量快路径省了多少时间的基准。均匀介质的 Rayleigh 检查不依赖这个选项。
//
// 失败（不抛异常，结果数组填 NaN，Kx、Ky、kzI、kzII、prop_r、prop_t 照常给出）：
//   · 均匀介质（入射区、出射区、均匀层）里某级 kz 精确为 0：status = RayleighAnomaly，message 写区域和 (m, n) 列表
//     （与 rcwa2d.py 的 RayleighAnomalyError 一致；与一维 C++ 允许入射区掠射不同，这是有意的）；
//     结构层里近零的 q 不特殊处理，不加任何小量；
//   · 本征迭代未收敛、线性方程组奇异（出现零主元）、中途出现 NaN/Inf：status = NumericalFailure，eig_info 记录第一个非零 info。
// 参数非法（M 为负、非有限、层厚为负、边界不递增、折射率非有限、入射介质非实或有吸收、θ 非有限或 |θ| ≥ 90°、算子尺寸不对…）
// 抛 std::invalid_argument。
//
// 近场用的 FieldCapture2D：capture != nullptr 时在同一次求解里保留各区域的模式与各层的 a⁺、a⁻（见该结构的注释）；
// capture == nullptr 时不多占内存，结果与有 capture 时逐位相同。a⁻ 取“该面以下的反射链作用在层底下行上”，
// 不用 exp(+k0·q·d) 反推，所以厚吸收层、倏逝层也稳定。
//
// parallel_layers = true：每个结构层的整个模式求解（Ei、P·Q、本征分解、V）用 parallel_for 分给线程池，结果与串行逐位相同；
// 此时 timing 里的 eig/linear/product 是各层 CPU 时间之和，不是墙上时间；custom_eig、custom_multiply、custom_solve
// 必须能被多个线程同时调用。
// solve2d_with_operators 的 timing.operators 恒为 0（算子由调用者装配）；solve2d 填写算子装配用时，并计入 timing.total。
// 本函数可重入，没有全局状态，结果与 set_num_threads 设定的线程数无关（逐位相同）。
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "eig.hpp"
#include "lu.hpp"
#include "matrix.hpp"
#include "parallel.hpp"
#include "rcwa2d_operators.hpp"
#include "rcwa2d_types.hpp"

namespace stablercwa {
namespace rcwa {
namespace detail2d {

// ------------------------------------------------------------------ 基本工具

using Clock = std::chrono::steady_clock;
inline double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

// 数值失败（方程组奇异、出现 NaN/Inf、本征迭代未收敛）时在内部抛出，solve2d_with_operators 接住后把结果填成 NaN。
struct NumericalFailure2D {
    std::string reason;
};
// 均匀介质里 kz 精确为 0
struct RayleighFailure2D {
    std::string message;
};

// 一次求解（或一个结构层）的计时与计数；parallel_layers 时每层各用一个，最后相加。
struct Work2D {
    Timing2D timing;
    int n_solve = 0;
    // Options2D::custom_multiply、custom_solve（指向调用方的 Options2D；空指针或空的函数表示用内置的运算）
    const std::function<Matrix(const Matrix&, const Matrix&)>* multiply_hook = nullptr;
    const std::function<Matrix(const Matrix&, const Matrix&)>* solve_hook = nullptr;
    void add(const Work2D& o) {
        timing.eig += o.timing.eig;
        timing.linear += o.timing.linear;
        timing.product += o.timing.product;
        n_solve += o.n_solve;
    }
};

inline bool is_finite(cplx v) { return std::isfinite(v.real()) && std::isfinite(v.imag()); }
inline bool finite_vector(const Vector& v) {
    for (const cplx& z : v)
        if (!is_finite(z)) return false;
    return true;
}
inline double max_keep_nan(double current, double value) {
    if (std::isnan(current) || std::isnan(value)) return std::numeric_limits<double>::quiet_NaN();
    return value > current ? value : current;
}

// ------------------------------------------------------------------ 2×2 分块对角矩阵 (2,2,N)

// [[a11, a12], [a21, a22]]，四块都是 N 维对角阵，只存对角元。
struct BlockDiag2 {
    Vector a11, a12, a21, a22;
    BlockDiag2() = default;
    explicit BlockDiag2(std::size_t n) : a11(n), a12(n), a21(n), a22(n) {}
    std::size_t size() const { return a11.size(); }
};

inline BlockDiag2 b2_eye(std::size_t n) {
    BlockDiag2 e(n);
    for (std::size_t i = 0; i < n; ++i) e.a11[i] = e.a22[i] = 1.0;
    return e;
}
inline BlockDiag2 b2_mul(const BlockDiag2& A, const BlockDiag2& B) {
    const std::size_t n = A.size();
    BlockDiag2 C(n);
    for (std::size_t i = 0; i < n; ++i) {
        C.a11[i] = A.a11[i] * B.a11[i] + A.a12[i] * B.a21[i];
        C.a12[i] = A.a11[i] * B.a12[i] + A.a12[i] * B.a22[i];
        C.a21[i] = A.a21[i] * B.a11[i] + A.a22[i] * B.a21[i];
        C.a22[i] = A.a21[i] * B.a12[i] + A.a22[i] * B.a22[i];
    }
    return C;
}
inline BlockDiag2 b2_inv(const BlockDiag2& A) {
    const std::size_t n = A.size();
    BlockDiag2 C(n);
    for (std::size_t i = 0; i < n; ++i) {
        const cplx det = A.a11[i] * A.a22[i] - A.a12[i] * A.a21[i];
        C.a11[i] = A.a22[i] / det;
        C.a12[i] = -A.a12[i] / det;
        C.a21[i] = -A.a21[i] / det;
        C.a22[i] = A.a11[i] / det;
    }
    return C;
}
inline BlockDiag2 b2_add(const BlockDiag2& A, const BlockDiag2& B) {
    const std::size_t n = A.size();
    BlockDiag2 C(n);
    for (std::size_t i = 0; i < n; ++i) {
        C.a11[i] = A.a11[i] + B.a11[i];
        C.a12[i] = A.a12[i] + B.a12[i];
        C.a21[i] = A.a21[i] + B.a21[i];
        C.a22[i] = A.a22[i] + B.a22[i];
    }
    return C;
}
inline BlockDiag2 b2_sub(const BlockDiag2& A, const BlockDiag2& B) {
    const std::size_t n = A.size();
    BlockDiag2 C(n);
    for (std::size_t i = 0; i < n; ++i) {
        C.a11[i] = A.a11[i] - B.a11[i];
        C.a12[i] = A.a12[i] - B.a12[i];
        C.a21[i] = A.a21[i] - B.a21[i];
        C.a22[i] = A.a22[i] - B.a22[i];
    }
    return C;
}
inline BlockDiag2 b2_scale(const BlockDiag2& A, double s) {
    const std::size_t n = A.size();
    BlockDiag2 C(n);
    for (std::size_t i = 0; i < n; ++i) {
        C.a11[i] = s * A.a11[i];
        C.a12[i] = s * A.a12[i];
        C.a21[i] = s * A.a21[i];
        C.a22[i] = s * A.a22[i];
    }
    return C;
}
inline BlockDiag2 b2_neg(BlockDiag2 A) {
    for (std::size_t i = 0; i < A.size(); ++i) {
        A.a11[i] = -A.a11[i];
        A.a12[i] = -A.a12[i];
        A.a21[i] = -A.a21[i];
        A.a22[i] = -A.a22[i];
    }
    return A;
}
// I − A
inline BlockDiag2 b2_eye_minus(const BlockDiag2& A) {
    BlockDiag2 C = b2_neg(A);
    for (std::size_t i = 0; i < A.size(); ++i) {
        C.a11[i] += 1.0;
        C.a22[i] += 1.0;
    }
    return C;
}
// 对应 np.any(A)：有任何一个元素非零（NaN 也算非零）
inline bool b2_any(const BlockDiag2& A) {
    for (std::size_t i = 0; i < A.size(); ++i)
        if (A.a11[i] != cplx{} || A.a12[i] != cplx{} || A.a21[i] != cplx{} || A.a22[i] != cplx{}) return true;
    return false;
}
// A 的稠密形式（2N×2N）
inline Matrix b2_dense(const BlockDiag2& A) {
    const int N = static_cast<int>(A.size());
    Matrix m(2 * N, 2 * N);
    for (int i = 0; i < N; ++i) {
        m(i, i) = A.a11[i];
        m(i, N + i) = A.a12[i];
        m(N + i, i) = A.a21[i];
        m(N + i, N + i) = A.a22[i];
    }
    return m;
}
// 把分块对角阵加到稠密矩阵上
inline void b2_add_to(Matrix& m, const BlockDiag2& A) {
    const int N = static_cast<int>(A.size());
    for (int i = 0; i < N; ++i) {
        m(i, i) += A.a11[i];
        m(i, N + i) += A.a12[i];
        m(N + i, i) += A.a21[i];
        m(N + i, N + i) += A.a22[i];
    }
}
// A·v，v 长 2N
inline Vector b2_apply(const BlockDiag2& A, const Vector& v) {
    const std::size_t N = A.size();
    Vector out(2 * N);
    for (std::size_t i = 0; i < N; ++i) {
        out[i] = A.a11[i] * v[i] + A.a12[i] * v[N + i];
        out[N + i] = A.a21[i] * v[i] + A.a22[i] * v[N + i];
    }
    return out;
}
// bdense(A)·M，O(n²)：每一列做分块对角左乘
inline Matrix b2_apply_rows(const BlockDiag2& A, const Matrix& M) {
    const int N = static_cast<int>(A.size());
    Matrix out(2 * N, M.cols());
    for (int j = 0; j < M.cols(); ++j) {
        const cplx* x = M.col(j);
        cplx* y = out.col(j);
        for (int i = 0; i < N; ++i) {
            y[i] = A.a11[i] * x[i] + A.a12[i] * x[N + i];
            y[N + i] = A.a21[i] * x[i] + A.a22[i] * x[N + i];
        }
    }
    return out;
}
// M·bdense(B)，O(n²)：第 j 列（j < N）= M(:,j)·b11[j] + M(:,N+j)·b21[j]，第 N+j 列 = M(:,j)·b12[j] + M(:,N+j)·b22[j]
inline Matrix b2_rmul(const Matrix& M, const BlockDiag2& B) {
    const int N = static_cast<int>(B.size());
    Matrix out(M.rows(), 2 * N);
    const int rows = M.rows();
    for (int j = 0; j < N; ++j) {
        const cplx* x1 = M.col(j);
        const cplx* x2 = M.col(N + j);
        cplx* y1 = out.col(j);
        cplx* y2 = out.col(N + j);
        const cplx b11 = B.a11[j], b12 = B.a12[j], b21 = B.a21[j], b22 = B.a22[j];
        for (int i = 0; i < rows; ++i) {
            y1[i] = x1[i] * b11 + x2[i] * b21;
            y2[i] = x1[i] * b12 + x2[i] * b22;
        }
    }
    return out;
}

// I − M
inline Matrix identity_minus(Matrix m) {
    cplx* p = m.data();
    const std::size_t total = static_cast<std::size_t>(m.rows()) * m.cols();
    for (std::size_t k = 0; k < total; ++k) p[k] = -p[k];
    for (int i = 0; i < m.rows(); ++i) m(i, i) += 1.0;
    return m;
}

// ------------------------------------------------------------------ 计时的稠密运算

inline bool hooked(const std::function<Matrix(const Matrix&, const Matrix&)>* hook) { return hook && *hook; }

// C = A·B（计时）；设了 custom_multiply 时交给它。
inline Matrix mul(const Matrix& a, const Matrix& b, Work2D& work) {
    const auto t0 = Clock::now();
    const bool use_hook = hooked(work.multiply_hook);
    Matrix c = use_hook ? (*work.multiply_hook)(a, b) : a * b;
    work.timing.product += seconds_since(t0);
    if (use_hook && (c.rows() != a.rows() || c.cols() != b.cols()))
        throw std::invalid_argument("rcwa2d: custom_multiply returned a matrix of the wrong size");
    return c;
}

// 解 A X = B（选主元 LU；设了 custom_solve 时交给它）。方程组奇异（零主元，或钩子抛 std::runtime_error）时抛 NumericalFailure2D。
inline Matrix solve_dense(Matrix a, Matrix b, Work2D& work) {
    ++work.n_solve;
    const auto t0 = Clock::now();
    const bool use_hook = hooked(work.solve_hook);
    const int rows = a.cols(), cols = b.cols();
    Matrix x;
    try {
        if (use_hook) {
            x = (*work.solve_hook)(a, b);
        } else {
            const LU f = ::stablercwa::detail::dense::lu_factor_impl<false>(std::move(a), Pivoting::Partial);
            x = lu_solve(f, std::move(b));
        }
    } catch (const std::runtime_error&) {
        work.timing.linear += seconds_since(t0);
        throw NumericalFailure2D{"线性方程组奇异（LU 出现零主元）"};
    }
    work.timing.linear += seconds_since(t0);
    if (use_hook && (x.rows() != rows || x.cols() != cols))
        throw std::invalid_argument("rcwa2d: custom_solve returned a matrix of the wrong size");
    return x;
}
// 右端是一个向量；设了 custom_solve 时把它当作一列的矩阵交给钩子。
inline Vector solve_dense_vector(Matrix a, Vector b, Work2D& work) {
    if (hooked(work.solve_hook)) {
        Matrix column(static_cast<int>(b.size()), 1);
        std::copy(b.begin(), b.end(), column.col(0));
        return solve_dense(std::move(a), std::move(column), work).column(0);
    }
    ++work.n_solve;
    const auto t0 = Clock::now();
    Vector x;
    try {
        const LU f = ::stablercwa::detail::dense::lu_factor_impl<false>(std::move(a), Pivoting::Partial);
        x = lu_solve(f, std::move(b));
    } catch (const std::runtime_error&) {
        work.timing.linear += seconds_since(t0);
        throw NumericalFailure2D{"线性方程组奇异（LU 出现零主元）"};
    }
    work.timing.linear += seconds_since(t0);
    return x;
}
inline Matrix inverse_dense(Matrix a, Work2D& work) {
    const int n = a.rows();
    return solve_dense(std::move(a), Matrix::identity(n), work);
}

// ------------------------------------------------------------------ kz、分支、模式

// 平方根取哪一支：先把 Re q < 0 的整体取负；再把 Re q < 1e-10 且 Im q > 0 的取负（rcwa2d.py:199，顺序与阈值不能改）
inline cplx branch(cplx q) {
    if (q.real() < 0.0) q = -q;
    if (q.real() < 1e-10 && q.imag() > 0.0) q = -q;
    return q;
}

// kz/k0 = sqrt(n² − Kx² − Ky²)：先对复数开平方，再把 Im < 0 的翻成 −kz（rcwa2d.py:205）。
// 传播级是实正，倏逝级是 +i|kz|，吸收介质保留复数。
inline cplx kz_over_k0(cplx n, double Kx, double Ky) {
    const cplx s = n * n - Kx * Kx - Ky * Ky;
    cplx kz = std::sqrt(s);
    if (kz.imag() < 0.0) kz = -kz;
    return kz;
}

// 一个区域（入射区、某一层、出射区）的模式。层内：
//     [Ex;Ey] = W·[a⁺·exp(−k0·q·(z − z_top)) + a⁻·exp(−k0·q·(z_bot − z))]，
//     [hx;hy] = V·[−a⁺·exp(…) + a⁻·exp(…)]
// homog = true：W = I，V 是 Vb（(2,2,N) 分块对角），不存矩阵；否则 W、V 是稠密的 2N×2N。
struct Modes2D {
    bool homog = false;
    Vector q;                // 长 2N
    BlockDiag2 Vb;           // homog
    Matrix W, V;             // 非 homog
};

inline Modes2D homogeneous_modes(cplx n, const RealVector& Kx, const RealVector& Ky, const Vector& kz) {
    const std::size_t N = Kx.size();
    Modes2D m;
    m.homog = true;
    m.q.resize(2 * N);
    m.Vb = BlockDiag2(N);
    const cplx n2 = n * n;
    for (std::size_t i = 0; i < N; ++i) {
        const cplx q1 = branch(cplx{kz[i].imag(), -kz[i].real()});         // −i·kz
        m.q[i] = q1;
        m.q[N + i] = q1;
        // Qb = [[Kx·Ky, n² − Kx²], [Ky² − n², −Ky·Kx]]，V = Qb / q1
        m.Vb.a11[i] = cplx{Kx[i] * Ky[i], 0.0} / q1;
        m.Vb.a12[i] = (n2 - Kx[i] * Kx[i]) / q1;
        m.Vb.a21[i] = (Ky[i] * Ky[i] - n2) / q1;
        m.Vb.a22[i] = cplx{-Ky[i] * Kx[i], 0.0} / q1;
    }
    return m;
}

// 结构层（以及 homogeneous_fast_path = false 时的均匀层）的模式。eps、cx、cy 是 N×N 的 Fourier 算子。
// P、Q 按 rcwa2d.py:403–404 装配（Kx、Ky 是对角阵，用行列缩放）：
//     P = [[Kx·Ei·Ky,      I − Kx·Ei·Kx],          Q = [[Kx·Ky,        Cy − Kx²],
//          [Ky·Ei·Ky − I, −Ky·Ei·Kx   ]],               [Ky² − Cx,   −Ky·Kx ]]
// 本征分解只对 H = P·Q 做一次（2N 阶）。H 里只有四个 N×N 的稠密乘积（Q 的对角块是缩放），V = Q·W/q 用两个 N×N×2N 的乘积。
// 返回前 out.W、out.V、out.q 已就绪；本征迭代未收敛时只设 eig_info（调用者据此判失败）。
inline void structured_modes(const Matrix& eps, const Matrix& cx, const Matrix& cy, const RealVector& Kx, const RealVector& Ky,
                             const Options2D& options, Modes2D& out, Matrix* Ei_keep, Work2D& work, int& eig_info,
                             double& residual) {
    const int N = static_cast<int>(Kx.size()), n = 2 * N;
    if (!eps.all_finite() || !cx.all_finite() || !cy.all_finite())
        throw NumericalFailure2D{"Fourier 算子含 NaN/Inf"};

    Matrix Ei = inverse_dense(eps, work);
    if (!Ei.all_finite()) throw NumericalFailure2D{"Eps 的逆含 NaN/Inf"};

    // P 的两个“列块”：pc1 = [P11; P21]（P 的前 N 列），pc2 = [P12; P22]（后 N 列），各 2N×N
    Matrix pc1(n, N), pc2(n, N);
    for (int j = 0; j < N; ++j) {
        cplx* c1 = pc1.col(j);
        cplx* c2 = pc2.col(j);
        const cplx* e = Ei.col(j);
        for (int i = 0; i < N; ++i) {
            const cplx kxe = e[i] * Kx[i];
            const cplx kye = e[i] * Ky[i];
            const double delta = i == j ? 1.0 : 0.0;
            c1[i] = kxe * Ky[j];                      // P11 = Kx·Ei·Ky
            c2[i] = delta - kxe * Kx[j];              // P12 = I − Kx·Ei·Kx
            c1[N + i] = kye * Ky[j] - delta;          // P21 = Ky·Ei·Ky − I
            c2[N + i] = -(kye * Kx[j]);               // P22 = −Ky·Ei·Kx
        }
    }
    if (Ei_keep) *Ei_keep = std::move(Ei);
    else Ei = Matrix();

    // Q12 = Cy − Kx²，Q21 = Ky² − Cx（稠密）；Q11 = Kx·Ky、Q22 = −Ky·Kx 是对角块
    Matrix q12 = cy, q21(N, N);
    for (int j = 0; j < N; ++j) {
        const cplx* xj = cx.col(j);
        cplx* y = q21.col(j);
        for (int i = 0; i < N; ++i) y[i] = -xj[i];
    }
    for (int i = 0; i < N; ++i) {
        q12(i, i) -= Kx[i] * Kx[i];
        q21(i, i) += Ky[i] * Ky[i];
    }

    // H = P·Q = [ pc1·Q11 + pc2·Q21 | pc1·Q12 + pc2·Q22 ]
    Matrix H(n, n);
    {
        Matrix t1 = mul(pc2, q21, work);
        Matrix t2 = mul(pc1, q12, work);
        for (int j = 0; j < N; ++j) {
            const double d1 = Kx[j] * Ky[j], d2 = -(Ky[j] * Kx[j]);
            cplx* h1 = H.col(j);
            const cplx *a1 = t1.col(j), *p1 = pc1.col(j);
            cplx* h2 = H.col(N + j);
            const cplx *a2 = t2.col(j), *p2 = pc2.col(j);
            for (int i = 0; i < n; ++i) {
                h1[i] = a1[i] + p1[i] * d1;
                h2[i] = a2[i] + p2[i] * d2;
            }
        }
    }
    pc1 = Matrix();
    pc2 = Matrix();

    EigResult e;
    {
        const auto t0 = Clock::now();
        e = options.custom_eig ? options.custom_eig(H) : stablercwa::eig(H, options.eig);
        work.timing.eig += seconds_since(t0);
    }
    if (e.info != 0) {
        eig_info = e.info;
        residual = std::numeric_limits<double>::quiet_NaN();
        return;
    }
    if (static_cast<int>(e.values.size()) != n || e.vectors.rows() != n || e.vectors.cols() != n)
        throw std::invalid_argument("rcwa2d: the eigen-solver returned a result of the wrong size");
    if (options.eig == EigMethod::LR && !options.custom_eig) {
        // LR（非酉消元）在二维的 2N 阶非对称矩阵上不可靠：本征值成簇时残差可以到 1e-2 以上而 info 仍是 0，
        // 不检查就会静默给出错误的结果（实测：N=169 的方柱差到 1e-2，随机算例里最大差 2e8）。
        // 所以对 LR 始终算一次本征残差，超过合同精度（1e-9）就报数值失败；默认的 QR 不需要（实测残差 ≤ 7.5e-16）。
        const double lr_residual = eig_residual(H, e);
        if (!(lr_residual <= 1e-9)) {
            char buf[160];
            std::snprintf(buf, sizeof buf, "LR 本征分解的相对残差 %.2e 超过 1e-9（二维请用 EigMethod::QR）", lr_residual);
            throw NumericalFailure2D{buf};
        }
        residual = lr_residual;
    } else if (options.diagnostics) residual = eig_residual(H, e);
    H = Matrix();

    out.homog = false;
    out.q.resize(static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j) out.q[static_cast<std::size_t>(j)] = branch(std::sqrt(e.values[static_cast<std::size_t>(j)]));
    out.W = std::move(e.vectors);
    e = EigResult();

    // V = (Q·W)/q：上半 = Q11·Wt + Q12·Wb，下半 = Q21·Wt + Q22·Wb
    Matrix wt(N, n), wb(N, n);
    for (int j = 0; j < n; ++j) {
        const cplx* w = out.W.col(j);
        cplx* a = wt.col(j);
        cplx* b = wb.col(j);
        for (int i = 0; i < N; ++i) {
            a[i] = w[i];
            b[i] = w[N + i];
        }
    }
    Matrix top = mul(q12, wb, work);
    Matrix bot = mul(q21, wt, work);
    q12 = Matrix();
    q21 = Matrix();
    out.V = Matrix(n, n);
    for (int j = 0; j < n; ++j) {
        const cplx qj = out.q[static_cast<std::size_t>(j)];
        const cplx *t = top.col(j), *b = bot.col(j), *a = wt.col(j), *c = wb.col(j);
        cplx* v = out.V.col(j);
        for (int i = 0; i < N; ++i) {
            const cplx up = t[i] + a[i] * (Kx[i] * Ky[i]);
            const cplx dn = b[i] + c[i] * (-(Ky[i] * Kx[i]));
            v[i] = up / qj;
            v[N + i] = dn / qj;
        }
    }
    if (!out.W.all_finite() || !out.V.all_finite()) throw NumericalFailure2D{"模式矩阵 W、V 含 NaN/Inf（某个 q 为零或本征向量退化）"};
}

// ------------------------------------------------------------------ S 矩阵块

// 一个 S 矩阵块：下行输出 = Tdd·下行输入 + Rdu·上行输入，上行输出 = Rud·下行输入 + Tuu·上行输入。
// bd = true 时四块都是 (2,2,N) 分块对角；否则是稠密 2N×2N。rdu_zero：Rdu = Rud = 0（传播块）。
struct SBlock2 {
    bool bd = false;
    bool rdu_zero = false;
    BlockDiag2 tdd, rdu, rud, tuu;
    Matrix Tdd, Rdu, Rud, Tuu;
};

// 由 Mm = W2⁻¹·W1、Nn = V2⁻¹·V1（稠密）拼出稠密块（rcwa2d.py 的 _s_dense）
inline SBlock2 s_dense(Matrix mm, const Matrix& nn, Work2D& work) {
    const std::size_t total = static_cast<std::size_t>(mm.rows()) * mm.cols();
    Matrix tp(mm.rows(), mm.cols());
    {
        cplx* a = tp.data();
        cplx* b = mm.data();
        const cplx* c = nn.data();
        for (std::size_t k = 0; k < total; ++k) {
            const cplx s = b[k] + c[k], d = b[k] - c[k];
            a[k] = 0.5 * s;
            b[k] = 0.5 * d;       // mm 就地变成 Tm
        }
    }
    const Matrix& tm = mm;
    Matrix tpi = inverse_dense(tp, work);
    SBlock2 s;
    s.bd = false;
    s.Rdu = mul(tm, tpi, work);
    s.Tdd = tp;
    s.Tdd -= mul(s.Rdu, tm, work);
    s.Rud = mul(tpi, tm, work);
    {
        cplx* a = s.Rud.data();
        for (std::size_t k = 0; k < total; ++k) a[k] = -a[k];
    }
    s.Tuu = std::move(tpi);
    return s;
}

// 两侧都是均匀介质（W = I）的界面（rcwa2d.py 的 s_interface2(bd=True)）：Mm = I，Nn = V2⁻¹·V1，全部 O(N)
inline SBlock2 interface_bd(const Modes2D& up, const Modes2D& low) {
    const std::size_t N = up.Vb.size();
    const BlockDiag2 nn = b2_mul(b2_inv(low.Vb), up.Vb);
    const BlockDiag2 mm = b2_eye(N);
    const BlockDiag2 tp = b2_scale(b2_add(mm, nn), 0.5), tm = b2_scale(b2_sub(mm, nn), 0.5);
    const BlockDiag2 tpi = b2_inv(tp);
    SBlock2 s;
    s.bd = true;
    s.tdd = b2_sub(tp, b2_mul(b2_mul(tm, tpi), tm));
    s.rdu = b2_mul(tm, tpi);
    s.rud = b2_neg(b2_mul(tpi, tm));
    s.tuu = tpi;
    return s;
}

// 匹配矩阵 Mm = W2⁻¹·W1、Nn = V2⁻¹·V1，至少有一侧是结构层（稠密模式）。三种情形与 Python 一一对应：
//   下侧是均匀介质（s_interface2_to_homog：W2 = I，V2 分块对角）：Mm = W1，Nn = V2⁻¹·V1 是 O(n²) 的行缩放；
//   上侧是均匀介质：W1 = I、V1 = bdense(Vb)，两个方程组要解（系数矩阵是下层的 W、V）；
//   两侧都是结构层：Mm = solve(W2, W1)，Nn = solve(V2, V1)。
inline void interface_matching(const Modes2D& up, const Modes2D& low, Work2D& work, Matrix& mm, Matrix& nn) {
    if (low.homog) {
        mm = up.W;
        nn = b2_apply_rows(b2_inv(low.Vb), up.V);
    } else if (up.homog) {
        mm = inverse_dense(low.W, work);
        nn = solve_dense(low.V, b2_dense(up.Vb), work);
    } else {
        mm = solve_dense(low.W, up.W, work);
        nn = solve_dense(low.V, up.V, work);
    }
}

inline SBlock2 interface_block(const Modes2D& up, const Modes2D& low, Work2D& work) {
    if (up.homog && low.homog) return interface_bd(up, low);
    Matrix mm, nn;
    interface_matching(up, low, work, mm, nn);
    return s_dense(std::move(mm), nn, work);
}

// 层内传播块 X = exp(−k0·q·d)（|X| ≤ 1），Rdu = Rud = 0，分块对角
inline SBlock2 propagation_block(const Vector& q, double d, double k0) {
    const std::size_t N = q.size() / 2;
    SBlock2 s;
    s.bd = true;
    s.rdu_zero = true;
    s.tdd = BlockDiag2(N);
    for (std::size_t i = 0; i < N; ++i) {
        s.tdd.a11[i] = std::exp(-k0 * q[i] * d);
        s.tdd.a22[i] = std::exp(-k0 * q[N + i] * d);
    }
    s.tuu = s.tdd;
    s.rdu = BlockDiag2(N);
    s.rud = BlockDiag2(N);
    return s;
}

// ------------------------------------------------------------------ 自下而上的总反射与自上而下的推进

// 块下方的总反射 r（从该处向下看，上行 = r·下行）：零、分块对角或稠密
struct Rstate {
    enum class Kind { Zero, Bd, Dense };
    Kind kind = Kind::Zero;
    BlockDiag2 bd;
    Matrix m;
};

inline Vector rstate_apply(const Rstate& r, const Vector& v) {
    if (r.kind == Rstate::Kind::Zero) return Vector(v.size());
    if (r.kind == Rstate::Kind::Bd) return b2_apply(r.bd, v);
    return r.m * v;
}

// 自上而下推进一个块要用的数据：
//   BdMat：down' = bd·down，bd 是 (I − Rdu·r)⁻¹·Tdd（或没有耦合时的 Tdd），分块对角；
//   Dense：down' = X·down，X = (I − Rdu·r)⁻¹·Tdd（S 块路径）；
//   Iface：直接递推的界面（见 direct_interface）：down' = Tm·(I + r')·down + Nn·down，r' 是这个界面连同下方的总反射，
//          Tm = ½(Mm − Nn)；iface_bd 为真时 Tm、Nn 是分块对角（Mm = I），否则稠密。
struct StepData {
    enum class Kind { None, BdMat, Dense, Iface };
    Kind kind = Kind::None;
    BlockDiag2 bd;
    Matrix X;
    bool iface_bd = false;
    BlockDiag2 tm_bd, nn_bd;
    Matrix tm, nn, rp;
};

inline Vector step_apply(const StepData& s, const Vector& v) {
    if (s.kind == StepData::Kind::BdMat) return b2_apply(s.bd, v);
    if (s.kind == StepData::Kind::Dense) return s.X * v;
    Vector u = s.rp * v;
    for (std::size_t i = 0; i < u.size(); ++i) u[i] += v[i];                 // (I + r')·v
    if (s.iface_bd) {
        Vector out = b2_apply(s.tm_bd, u);
        const Vector w = b2_apply(s.nn_bd, v);
        for (std::size_t i = 0; i < out.size(); ++i) out[i] += w[i];
        return out;
    }
    Vector out = s.tm * u;
    const Vector w = s.nn * v;
    for (std::size_t i = 0; i < out.size(); ++i) out[i] += w[i];
    return out;
}

// 把一个块叠到下方结构（总反射 r）之上：r' = Rud + Tuu·r·(I − Rdu·r)⁻¹·Tdd，同时给出自上而下推进的数据。
// 路径与 rcwa2d.py 的 _rud2/_step2 一一对应（见文件头）。
inline Rstate stack_on(SBlock2&& blk, const Rstate& r, StepData& step, Work2D& work) {
    using K = Rstate::Kind;
    Rstate out;
    if (blk.bd) {
        const std::size_t N = blk.tdd.size();
        if (r.kind != K::Dense) {
            // 第一支：全分块对角
            const BlockDiag2 rr = r.kind == K::Zero ? BlockDiag2(N) : r.bd;
            BlockDiag2 gt;
            if (blk.rdu_zero) gt = blk.tdd;
            else gt = b2_mul(b2_inv(b2_eye_minus(b2_mul(blk.rdu, rr))), blk.tdd);
            out.kind = K::Bd;
            out.bd = r.kind == K::Zero ? blk.rud : b2_add(blk.rud, b2_mul(b2_mul(blk.tuu, rr), gt));
            step.kind = StepData::Kind::BdMat;
            step.bd = std::move(gt);
            return out;
        }
        // 第二支：分块对角块 + 稠密 Rud
        out.kind = K::Dense;
        if (blk.rdu_zero) {
            out.m = b2_rmul(b2_apply_rows(blk.tuu, r.m), blk.tdd);
            b2_add_to(out.m, blk.rud);
            step.kind = StepData::Kind::BdMat;
            step.bd = std::move(blk.tdd);
            return out;
        }
        Matrix mat = identity_minus(b2_apply_rows(blk.rdu, r.m));      // I − Rdu·r
        Matrix X = solve_dense(mat, b2_dense(blk.tdd), work);          // (I − Rdu·r)⁻¹·Tdd
        mat = Matrix();
        out.m = b2_apply_rows(blk.tuu, mul(r.m, X, work));
        b2_add_to(out.m, blk.rud);
        step.kind = StepData::Kind::Dense;
        step.X = std::move(X);
        return out;
    }
    // 第三支：稠密块
    out.kind = K::Dense;
    if (r.kind == K::Zero) {
        out.m = std::move(blk.Rud);
        step.kind = StepData::Kind::Dense;
        step.X = std::move(blk.Tdd);
        return out;
    }
    Matrix mat = identity_minus(r.kind == K::Bd ? b2_rmul(blk.Rdu, r.bd) : mul(blk.Rdu, r.m, work));   // I − Rdu·r
    blk.Rdu = Matrix();
    Matrix X = solve_dense(mat, std::move(blk.Tdd), work);
    mat = Matrix();
    Matrix y = r.kind == K::Bd ? b2_apply_rows(r.bd, X) : mul(r.m, X, work);
    out.m = std::move(blk.Rud);
    out.m += mul(blk.Tuu, y, work);
    step.kind = StepData::Kind::Dense;
    step.X = std::move(X);
    return out;
}

// ---- 直接递推的界面（默认）：不造 S 块，直接给出“界面连同下方”的总反射 r'
//
// 界面上方（W1、V1、a⁺、a⁻ = r'·a⁺）与下方（W2、V2、下行 c、上行 r·c）的切向场连续：
//     W1·(I + r')·a⁺ = W2·(I + r)·c，    V1·(I − r')·a⁺ = V2·(I − r)·c。
// 记 Mm = W2⁻¹·W1、Nn = V2⁻¹·V1（与 S 块相同），则 (I + r)·c = Mm·(I + r')·a⁺、(I − r)·c = Nn·(I − r')·a⁺。
// 两式左乘 (I − r)、(I + r)（它们可交换）并相减，得
//     r' = A₊⁻¹·A₋，    A± = (I + r)·Nn ± (I − r)·Mm；
// 两式相加（(I + r) + (I − r) = 2I，不需要 (I ± r) 的逆，r = ±1 的全反射也没有奇点）得
//     c = Tm·(I + r')·a⁺ + Nn·a⁺，    Tm = ½(Mm − Nn)。
// 它与 S 块的 r' = Rud + Tuu·r·(I − Rdu·r)⁻¹·Tdd、c = (I − Rdu·r)⁻¹·Tdd·a⁺ 在精确算术里相同（r = 0 时就是 Rud = −Tp⁻¹·Tm），
// 但只要一次 LU 加一次 n 个右端的回代，不要 Tpi、Rdu、Tdd、Rud 四个 n×n 乘积；下方 r 是分块对角时 A± 只是行列缩放。
// 稠密 Rud 在膜堆之上要递推多次，所以这是 N 很大时省时间的地方；S 块路径保留作交叉检验（solve_impl 的 s_blocks 参数）。

// Mm、Nn 都是稠密的（至少一侧是结构层）
inline Rstate direct_dense(Matrix mm, Matrix nn, const Rstate& r, StepData& step, Work2D& work) {
    using K = Rstate::Kind;
    Matrix pn, qm;                                         // (I + r)·Nn、(I − r)·Mm
    if (r.kind == K::Zero) {
        pn = nn;
        qm = mm;
    } else if (r.kind == K::Bd) {
        pn = b2_apply_rows(b2_add(b2_eye(r.bd.size()), r.bd), nn);
        qm = b2_apply_rows(b2_eye_minus(r.bd), mm);
    } else {
        pn = mul(r.m, nn, work);
        pn += nn;                                          // Nn + r·Nn
        qm = mul(r.m, mm, work);
        {
            cplx* a = qm.data();
            const cplx* b = mm.data();
            const std::size_t total = static_cast<std::size_t>(qm.rows()) * qm.cols();
            for (std::size_t k = 0; k < total; ++k) a[k] = b[k] - a[k];        // Mm − r·Mm
        }
    }
    Matrix ap = pn;
    ap += qm;                                              // A₊
    pn -= qm;                                              // A₋
    qm = Matrix();
    Rstate out;
    out.kind = K::Dense;
    out.m = solve_dense(std::move(ap), std::move(pn), work);
    mm -= nn;
    mm *= 0.5;                                             // Tm
    step.kind = StepData::Kind::Iface;
    step.iface_bd = false;
    step.tm = std::move(mm);
    step.nn = std::move(nn);
    step.rp = out.m;
    return out;
}

// 两侧都是均匀介质（Mm = I，Nn 分块对角），下方的 r 已稠密（结构层之上的膜界面）
inline Rstate direct_bd_over_dense(const BlockDiag2& nn, const Rstate& r, StepData& step, Work2D& work) {
    const BlockDiag2 one = b2_eye(nn.size());
    const BlockDiag2 np1 = b2_add(nn, one), nm1 = b2_sub(nn, one);
    Matrix ap = b2_rmul(r.m, nm1);                         // r·(Nn − I)
    b2_add_to(ap, np1);                                    // A₊ = (Nn + I) + r·(Nn − I)
    Matrix am = b2_rmul(r.m, np1);
    b2_add_to(am, nm1);                                    // A₋ = (Nn − I) + r·(Nn + I)
    Rstate out;
    out.kind = Rstate::Kind::Dense;
    out.m = solve_dense(std::move(ap), std::move(am), work);
    step.kind = StepData::Kind::Iface;
    step.iface_bd = true;
    step.tm_bd = b2_scale(nm1, -0.5);                      // Tm = ½(I − Nn)
    step.nn_bd = nn;
    step.rp = out.m;
    return out;
}

// 把界面 (up | low) 叠到下方结构（总反射 r）之上，给出新的总反射与自上而下推进的数据。
//   两侧都是分块对角：O(N)（r 稠密时用 direct_bd_over_dense）；其余用 direct_dense（s_blocks = true 时改用 S 块，与 Python 逐段对应）。
inline Rstate stack_on_interface(const Modes2D& up, const Modes2D& low, const Rstate& r, StepData& step, Work2D& work,
                                 bool s_blocks) {
    if (up.homog && low.homog) {
        if (r.kind != Rstate::Kind::Dense || s_blocks) return stack_on(interface_bd(up, low), r, step, work);
        return direct_bd_over_dense(b2_mul(b2_inv(low.Vb), up.Vb), r, step, work);
    }
    if (s_blocks) return stack_on(interface_block(up, low, work), r, step, work);
    Matrix mm, nn;
    interface_matching(up, low, work, mm, nn);
    return direct_dense(std::move(mm), std::move(nn), r, step, work);
}

// 最上面的界面块（入射区 up 与区域 1 low 之间）。入射波只有 (0,0) 级的 Ex、Ey 两个分量，Tdd、Rud 只用得到这一个向量，
// 不必把整个矩阵乘出来：Tdd·e = Tp·e − Rdu·(Tm·e)，Rud·e = −Tp⁻¹·(Tm·e)。
// down1 = (I − Rdu·r)⁻¹·Tdd·e（区域 1 顶部的下行振幅），R = Rud·e + Tuu·r·down1。
inline void top_block(const Modes2D& up, const Modes2D& low, const Rstate& r, const Vector& inc, Work2D& work, Vector& down1,
                      Vector& R, bool s_blocks) {
    using K = Rstate::Kind;
    if (up.homog && low.homog) {
        const SBlock2 blk = interface_bd(up, low);
        if (r.kind != K::Dense) {
            const BlockDiag2 rr = r.kind == K::Zero ? BlockDiag2(blk.tdd.size()) : r.bd;
            const BlockDiag2 g = b2_inv(b2_eye_minus(b2_mul(blk.rdu, rr)));
            down1 = b2_apply(b2_mul(g, blk.tdd), inc);
        } else {
            const Matrix mat = identity_minus(b2_apply_rows(blk.rdu, r.m));
            down1 = solve_dense_vector(mat, b2_apply(blk.tdd, inc), work);
        }
        R = b2_apply(blk.rud, inc);
        const Vector back = b2_apply(blk.tuu, rstate_apply(r, down1));
        for (std::size_t i = 0; i < R.size(); ++i) R[i] += back[i];
        return;
    }
    if (!s_blocks) {
        // 直接解：入射区 e + R = W·(I + r)·d，V_I·(e − R) = V·(I − r)·d，相加得 K·d = 2e，K = W·(I + r) + V_I⁻¹·V·(I − r)；
        // R = W·(I + r)·d − e。不要 W⁻¹、V⁻¹，只有两个 n×n 乘积（r 分块对角时是缩放）和一次 LU。
        Matrix wp, vq;                                     // W·(I + r)、V·(I − r)
        if (r.kind == K::Zero) {
            wp = low.W;
            vq = low.V;
        } else if (r.kind == K::Bd) {
            wp = b2_rmul(low.W, b2_add(b2_eye(r.bd.size()), r.bd));
            vq = b2_rmul(low.V, b2_eye_minus(r.bd));
        } else {
            wp = mul(low.W, r.m, work);
            wp += low.W;
            vq = mul(low.V, r.m, work);
            {
                cplx* a = vq.data();
                const cplx* b = low.V.data();
                const std::size_t total = static_cast<std::size_t>(vq.rows()) * vq.cols();
                for (std::size_t k = 0; k < total; ++k) a[k] = b[k] - a[k];
            }
        }
        Matrix kmat = b2_apply_rows(b2_inv(up.Vb), vq);
        vq = Matrix();
        kmat += wp;
        Vector rhs(inc);
        for (cplx& v : rhs) v *= 2.0;
        down1 = solve_dense_vector(std::move(kmat), std::move(rhs), work);
        R = wp * down1;
        for (std::size_t i = 0; i < R.size(); ++i) R[i] -= inc[i];
        return;
    }
    Matrix mm, nn;
    interface_matching(up, low, work, mm, nn);
    const std::size_t total = static_cast<std::size_t>(mm.rows()) * mm.cols();
    Matrix tp(mm.rows(), mm.cols());
    {
        cplx* a = tp.data();
        cplx* b = mm.data();
        const cplx* c = nn.data();
        for (std::size_t k = 0; k < total; ++k) {
            const cplx s = b[k] + c[k], d = b[k] - c[k];
            a[k] = 0.5 * s;
            b[k] = 0.5 * d;
        }
    }
    nn = Matrix();
    const Matrix& tm = mm;
    const Matrix tpi = inverse_dense(tp, work);
    const Matrix rdu = mul(tm, tpi, work);
    const Vector tm_e = tm * inc;
    Vector tdd_e = tp * inc;
    {
        const Vector back = rdu * tm_e;
        for (std::size_t i = 0; i < tdd_e.size(); ++i) tdd_e[i] -= back[i];
    }
    Vector rud_e = tpi * tm_e;
    for (cplx& v : rud_e) v = -v;
    if (r.kind == K::Zero) {
        down1 = std::move(tdd_e);
    } else {
        const Matrix mat = identity_minus(r.kind == K::Bd ? b2_rmul(rdu, r.bd) : mul(rdu, r.m, work));
        down1 = solve_dense_vector(mat, std::move(tdd_e), work);
    }
    R = std::move(rud_e);
    const Vector back = tpi * rstate_apply(r, down1);
    for (std::size_t i = 0; i < R.size(); ++i) R[i] += back[i];
}

// ------------------------------------------------------------------ 近场用的快照

// 一个区域（入射区或出射区）的模式：q（2N）、V（2×2 分块对角）、kz/k0（N）。W = I。
struct RegionCapture2D {
    Vector q;
    BlockDiag2 V;
    Vector kz;
};

// 一层的快照。层内场（rcwa2d.py 第 557 行起的约定，zl = z − z_top）：
//     [Ex;Ey] = W·(exp(−k0·q·zl)·a_plus + exp(−k0·q·(d − zl))·a_minus)，  [hx;hy] = V·(−exp(…)·a_plus + exp(…)·a_minus)
//     Ez = i·Ei·(Ky·hx − Kx·hy)（Ei = Eps⁻¹；均匀层 Ei = 1/n²），hz = i·(Ky·Ex − Kx·Ey)
// homogeneous = true：快路径表示——W = I（不存），V 是分块对角（Vbd），Eps 用标量 eps = n²；
// 否则（结构层，或 homogeneous_fast_path = false 时的均匀层）：W、V 是稠密的 2N×2N，Eps 的求解表示是 Ei = Eps⁻¹（N×N 稠密逆）。
struct LayerCapture2D {
    double z_top = 0.0, d = 0.0;
    bool homogeneous = false;
    Vector q;                  // 长 2N
    Matrix W, V;               // 结构层
    BlockDiag2 Vbd;            // 均匀层（快路径）
    Matrix Ei;                 // 结构层：Eps⁻¹（N×N）
    cplx eps = 1.0;            // 均匀层：n²
    cplx n = 1.0;              // 均匀层折射率（快路径时才有意义）
    Vector a_plus;             // 层顶下行振幅，长 2N
    Vector a_minus;            // 层底上行振幅 = 该面以下的反射链作用在层底下行上，长 2N
};

struct FieldCapture2D {
    bool available = false;    // 求解成功且全部数据有限时为 true；失败时整个结构复位成空
    int N = 0, M1 = 0, M2 = 0;
    double k0 = 0.0, theta = 0.0, phi = 0.0;
    double nI = 1.0;
    cplx nII = 1.0;
    double z_bot = 0.0;        // 各层厚度之和（结构上表面 z = 0）
    RealVector Kx, Ky;         // 长 N，以 k0 为单位
    RegionCapture2D region_I, region_II;
    Vector inc, R, T;          // 入射矢量（长 2N：前 N 个 Ex、后 N 个 Ey）、各级反射、透射（电场的 Ex、Ey 分量）
    std::vector<LayerCapture2D> layers;
};

// ------------------------------------------------------------------ 输入检查

inline void check_index_2d(cplx n, const char* what) {
    if (!is_finite(n) || std::abs(n) == 0.0 || n.real() < 0.0 || n.imag() < 0.0)
        throw std::invalid_argument(std::string("rcwa2d: ") + what + " must be finite, nonzero, with Re n >= 0 and Im n >= 0");
}

inline void check_edge_list(const std::vector<double>& e, double period, const char* what) {
    for (std::size_t j = 0; j < e.size(); ++j)
        if (!std::isfinite(e[j]) || (j > 0 && e[j] <= e[j - 1]))
            throw std::invalid_argument(std::string("rcwa2d: ") + what + " must be finite and strictly increasing");
    if (std::abs((e.back() - e.front()) - period) > 1e-9 * period)     // 与 rcwa2d_operators.hpp 的相对容差一致
        throw std::invalid_argument(std::string("rcwa2d: ") + what + " must span exactly one period");
}

inline void check_layer_core(const Layer2D& layer, double Lx, double Ly) {
    if (!std::isfinite(layer.d) || layer.d < 0.0) throw std::invalid_argument("rcwa2d: layer thickness must be finite and >= 0");
    if (layer.homogeneous) {
        check_index_2d(layer.n, "layer index");
        return;
    }
    const std::size_t K = layer.x_profiles.size();
    if (K == 0 || layer.y_edges.size() != K + 1)
        throw std::invalid_argument("rcwa2d: a patterned layer needs K x-profiles and K+1 y-edges");
    check_edge_list(layer.y_edges, Ly, "y-edges");
    for (const XProfile2D& p : layer.x_profiles) {
        if (p.n_values.empty() || p.edges.size() != p.n_values.size() + 1)
            throw std::invalid_argument("rcwa2d: an x-profile needs K indices and K+1 edges");
        check_edge_list(p.edges, Lx, "x-edges");
        for (const cplx& n : p.n_values) check_index_2d(n, "profile index");
    }
}

inline void check_core_inputs(const Stack2D& stack, double wavelength, double theta_deg, double phi_deg, int M1, int M2,
                         const Jones2D& jones, const std::vector<LayerOperators2D>& ops, const Options2D& options) {
    const long long limit = ((std::numeric_limits<int>::max)() - 1) / 4;
    if (M1 < 0 || M2 < 0 || static_cast<long long>(M1) > limit || static_cast<long long>(M2) > limit ||
        (2LL * M1 + 1) * (2LL * M2 + 1) > limit)
        throw std::invalid_argument("rcwa2d: invalid truncation order M1, M2");
    if (!std::isfinite(wavelength) || wavelength <= 0.0 || !std::isfinite(stack.Lx) || stack.Lx <= 0.0 ||
        !std::isfinite(stack.Ly) || stack.Ly <= 0.0)
        throw std::invalid_argument("rcwa2d: wavelength and periods must be finite and positive");
    if (!std::isfinite(theta_deg) || std::abs(theta_deg) >= 90.0)
        throw std::invalid_argument("rcwa2d: incidence angle must lie strictly between -90 and 90 degrees");
    if (!std::isfinite(phi_deg)) throw std::invalid_argument("rcwa2d: azimuth angle must be finite");
    if (!std::isfinite(stack.nI) || stack.nI <= 0.0)
        throw std::invalid_argument("rcwa2d: the incident medium must have a real positive index");
    check_index_2d(stack.nII, "exit-medium index");
    if (!is_finite(jones[0]) || !is_finite(jones[1])) throw std::invalid_argument("rcwa2d: the Jones vector must be finite");
    if (options.eig != EigMethod::QR && options.eig != EigMethod::LR) throw std::invalid_argument("rcwa2d: unknown eigen-solver");
    if (options.factorization != Factorization2D::Li && options.factorization != Factorization2D::Laurent)
        throw std::invalid_argument("rcwa2d: unknown factorization");
    if (ops.size() != stack.layers.size())
        throw std::invalid_argument("rcwa2d: ops must have one entry per layer (entries of uniform layers are ignored)");
    const int N = (2 * M1 + 1) * (2 * M2 + 1);
    for (std::size_t j = 0; j < stack.layers.size(); ++j) {
        const Layer2D& layer = stack.layers[j];
        check_layer_core(layer, stack.Lx, stack.Ly);
        if (layer.homogeneous) continue;
        for (const Matrix* m : {&ops[j].eps, &ops[j].cx, &ops[j].cy})
            if (m->rows() != N || m->cols() != N)
                throw std::invalid_argument("rcwa2d: the operators of a patterned layer must be N x N, N = (2*M1+1)*(2*M2+1)");
    }
}

// ------------------------------------------------------------------ 输出的小工具

inline std::string format_index(cplx n) {
    char buf[96];
    if (n.imag() == 0.0) std::snprintf(buf, sizeof buf, "%.10g", n.real());
    else std::snprintf(buf, sizeof buf, "(%.10g%+.10gj)", n.real(), n.imag());
    return buf;
}

// 某均匀区域里 kz 精确为 0 的级次；有则把一段说明追加到 message
inline void report_rayleigh(std::string& message, const std::string& where, cplx n, const Vector& kz, const std::vector<int>& m,
                            const std::vector<int>& nn) {
    std::string labels;
    for (std::size_t i = 0; i < kz.size(); ++i) {
        if (kz[i] != cplx{}) continue;
        if (!labels.empty()) labels += ", ";
        labels += "(" + std::to_string(m[i]) + ", " + std::to_string(nn[i]) + ")";
    }
    if (labels.empty()) return;
    if (!message.empty()) message += "；";
    message += where + "（n=" + format_index(n) + "）的 (m, n) = " + labels + " 级恰好沿界面掠射（kz=0，Rayleigh 异常点），无法求解";
}

// 求解的实现。s_blocks = false（默认）：与结构层相邻的稠密界面用直接递推（direct_interface）；true：用 S 块
// （rcwa2d.py 的 s_interface2 / _rud2 / _step2 逐段对应），只用来交叉检验，数值上两者相同。
inline Result2D solve_impl(const Stack2D& stack, double wavelength, double theta_deg, double phi_deg, int M1, int M2,
                           const Jones2D& jones, const std::vector<LayerOperators2D>& ops, const Options2D& options,
                           FieldCapture2D* capture, bool s_blocks) {
    namespace d2 = ::stablercwa::rcwa::detail2d;
    const auto t_start = d2::Clock::now();
    d2::check_core_inputs(stack, wavelength, theta_deg, phi_deg, M1, M2, jones, ops, options);

    const int n1 = 2 * M1 + 1, n2 = 2 * M2 + 1;
    const int N = n1 * n2, n = 2 * N;
    const std::size_t Nz = static_cast<std::size_t>(N);
    const int L = static_cast<int>(stack.layers.size());
    const double nI = stack.nI;
    const cplx nII = stack.nII;
    const double k0 = 2.0 * kPi / wavelength;
    const double th = theta_deg * (kPi / 180.0), ph = phi_deg * (kPi / 180.0);
    const double cos_th = std::cos(th), sin_th = std::sin(th), cos_ph = std::cos(ph), sin_ph = std::sin(ph);
    const bool fast = options.homogeneous_fast_path;

    Result2D result;
    result.M1 = M1;
    result.M2 = M2;
    result.N = N;
    result.i0 = M1 * n2 + M2;
    result.k0 = k0;
    result.theta = th;
    result.phi = ph;
    result.m.resize(Nz);
    result.n.resize(Nz);
    result.Kx.resize(Nz);
    result.Ky.resize(Nz);
    result.kzI.resize(Nz);
    result.kzII.resize(Nz);
    for (int a = 0; a < n1; ++a)
        for (int b = 0; b < n2; ++b) {
            const std::size_t i = static_cast<std::size_t>(a * n2 + b);
            const int m = a - M1, nn = b - M2;
            result.m[i] = m;
            result.n[i] = nn;
            result.Kx[i] = nI * sin_th * cos_ph - m * wavelength / stack.Lx;
            result.Ky[i] = nI * sin_th * sin_ph - nn * wavelength / stack.Ly;
            result.kzI[i] = d2::kz_over_k0(nI, result.Kx[i], result.Ky[i]);
            result.kzII[i] = d2::kz_over_k0(nII, result.Kx[i], result.Ky[i]);
        }
    result.prop_r.assign(Nz, 0);
    result.prop_t.assign(Nz, 0);
    for (std::size_t i = 0; i < Nz; ++i) {
        result.prop_r[i] = result.kzI[i].real() > 1e-12;
        result.prop_t[i] = result.kzII[i].real() > 1e-12;
    }
    const RealVector& Kx = result.Kx;
    const RealVector& Ky = result.Ky;

    d2::Work2D work;
    work.multiply_hook = &options.custom_multiply;
    work.solve_hook = &options.custom_solve;
    if (capture) *capture = d2::FieldCapture2D{};

    try {
        // ---- 1. Rayleigh 检查：均匀介质（入射区、出射区、均匀层）里 kz 精确为 0 的级次
        {
            std::string message;
            d2::report_rayleigh(message, "入射区", nI, result.kzI, result.m, result.n);
            d2::report_rayleigh(message, "出射区", nII, result.kzII, result.m, result.n);
            for (int j = 0; j < L; ++j) {
                const Layer2D& layer = stack.layers[static_cast<std::size_t>(j)];
                if (!layer.homogeneous) continue;
                Vector kz(Nz);
                for (std::size_t i = 0; i < Nz; ++i) kz[i] = d2::kz_over_k0(layer.n, Kx[i], Ky[i]);
                d2::report_rayleigh(message, "第" + std::to_string(j + 1) + "层" + (layer.name.empty() ? "" : "「" + layer.name + "」"),
                                    layer.n, kz, result.m, result.n);
            }
            if (!message.empty())
                throw d2::RayleighFailure2D{message + "。请把入射角、方位角或周期略作改动（例如入射角改动 0.01°）后重算。"};
        }

        // ---- 2. 各区域的模式。区域 0 是入射区，区域 j+1 是第 j 层，最后一个区域是出射区。
        std::vector<d2::Modes2D> modes(static_cast<std::size_t>(L) + 2);
        modes.front() = d2::homogeneous_modes(nI, Kx, Ky, result.kzI);
        modes.back() = d2::homogeneous_modes(nII, Kx, Ky, result.kzII);
        std::vector<int> solve_layers;                         // 需要做本征分解的层
        std::vector<d2::LayerCapture2D> layer_capture;
        if (capture) layer_capture.resize(static_cast<std::size_t>(L));
        for (int j = 0; j < L; ++j) {
            const Layer2D& layer = stack.layers[static_cast<std::size_t>(j)];
            if (fast && layer.homogeneous) {
                Vector kz(Nz);
                for (std::size_t i = 0; i < Nz; ++i) kz[i] = d2::kz_over_k0(layer.n, Kx[i], Ky[i]);
                modes[static_cast<std::size_t>(j) + 1] = d2::homogeneous_modes(layer.n, Kx, Ky, kz);
                if (capture) {
                    d2::LayerCapture2D& lc = layer_capture[static_cast<std::size_t>(j)];
                    lc.homogeneous = true;
                    lc.eps = layer.n * layer.n;
                    lc.n = layer.n;
                }
            } else {
                solve_layers.push_back(j);
            }
        }
        const int count = static_cast<int>(solve_layers.size());
        d2::Work2D layer_work;                           // 每层一个（parallel_layers 时各在一个线程里用），最后合计
        layer_work.multiply_hook = work.multiply_hook;
        layer_work.solve_hook = work.solve_hook;
        std::vector<d2::Work2D> works(static_cast<std::size_t>(count), layer_work);
        std::vector<int> infos(static_cast<std::size_t>(count), 0);
        RealVector residuals(static_cast<std::size_t>(count), 0.0);
        std::vector<Matrix> Eis(capture ? static_cast<std::size_t>(count) : 0);
        const auto decompose = [&](int i) {
            const int j = solve_layers[static_cast<std::size_t>(i)];
            const Layer2D& layer = stack.layers[static_cast<std::size_t>(j)];
            Matrix* keep = capture ? &Eis[static_cast<std::size_t>(i)] : nullptr;
            if (layer.homogeneous) {     // fast = false：均匀层按稠密层处理，Eps = Cx = Cy = n²·I
                Matrix eps(N, N);
                const cplx n2c = layer.n * layer.n;
                for (int k = 0; k < N; ++k) eps(k, k) = n2c;
                d2::structured_modes(eps, eps, eps, Kx, Ky, options, modes[static_cast<std::size_t>(j) + 1], keep,
                                     works[static_cast<std::size_t>(i)], infos[static_cast<std::size_t>(i)],
                                     residuals[static_cast<std::size_t>(i)]);
            } else {
                const d2::LayerOperators2D& op = ops[static_cast<std::size_t>(j)];
                d2::structured_modes(op.eps, op.cx, op.cy, Kx, Ky, options, modes[static_cast<std::size_t>(j) + 1], keep,
                                     works[static_cast<std::size_t>(i)], infos[static_cast<std::size_t>(i)],
                                     residuals[static_cast<std::size_t>(i)]);
            }
        };
        result.n_eig = count;
        if (options.parallel_layers) parallel_for(0, count, decompose);
        else for (int i = 0; i < count; ++i) decompose(i);
        for (int i = 0; i < count; ++i) {
            work.add(works[static_cast<std::size_t>(i)]);
            if (infos[static_cast<std::size_t>(i)] != 0 && result.eig_info == 0) result.eig_info = infos[static_cast<std::size_t>(i)];
            if (options.diagnostics) result.max_eig_residual = d2::max_keep_nan(result.max_eig_residual, residuals[static_cast<std::size_t>(i)]);
        }
        if (result.eig_info != 0) throw d2::NumericalFailure2D{"本征迭代未收敛（eig_info = " + std::to_string(result.eig_info) + "）"};
        if (capture)
            for (int i = 0; i < count; ++i) layer_capture[static_cast<std::size_t>(solve_layers[static_cast<std::size_t>(i)])].Ei =
                std::move(Eis[static_cast<std::size_t>(i)]);

        // ---- 3. 入射矢量
        const cplx Es = jones[0], Ep = jones[1];
        Vector inc(static_cast<std::size_t>(n));
        inc[static_cast<std::size_t>(result.i0)] = -Es * sin_ph + Ep * cos_th * cos_ph;           // Ex
        inc[static_cast<std::size_t>(N + result.i0)] = Es * cos_ph + Ep * cos_th * sin_ph;        // Ey

        // ---- 4. 自下而上叠加各块。块 2j 是区域 j 与 j+1 之间的界面，块 2j+1 是第 j 层内的传播；共 2L+1 块。
        // 块在用到时才构造；叠完只留下推进用的数据（steps）和当前的总反射 r。最上面的块 0 另外处理（只算入射级）。
        const int block_count = 2 * L + 1;
        std::vector<d2::StepData> steps(static_cast<std::size_t>(block_count));
        std::vector<d2::Rstate> r_below;                 // 仅近场路径保留各层底面以下的总反射
        if (capture) r_below.resize(static_cast<std::size_t>(L));
        d2::Rstate r;                                    // 出射区里没有向上传的波：最底下的总反射为零
        for (int b = block_count - 1; b >= 1; --b) {
            const int j = b / 2;
            if (b % 2 == 0) {
                r = d2::stack_on_interface(modes[static_cast<std::size_t>(j)], modes[static_cast<std::size_t>(j) + 1], r,
                                           steps[static_cast<std::size_t>(b)], work, s_blocks);
                if (!capture) {                          // 第 j 层的上界面已经造好：它的 W、V 不再有用
                    modes[static_cast<std::size_t>(j) + 1].W = Matrix();
                    modes[static_cast<std::size_t>(j) + 1].V = Matrix();
                }
            } else {
                if (capture) r_below[static_cast<std::size_t>(j)] = r;
                d2::SBlock2 blk =
                    d2::propagation_block(modes[static_cast<std::size_t>(j) + 1].q, stack.layers[static_cast<std::size_t>(j)].d, k0);
                r = d2::stack_on(std::move(blk), r, steps[static_cast<std::size_t>(b)], work);
            }
        }

        // ---- 5. 最上面的块与自上而下推进
        Vector down, R;
        d2::top_block(modes[0], modes[1], r, inc, work, down, R, s_blocks);
        if (!capture && L > 0) {
            modes[1].W = Matrix();
            modes[1].V = Matrix();
        }
        r = d2::Rstate{};
        Vector T;
        for (int b = 1; b < block_count; ++b) {
            const int j = b / 2;
            const bool propagation = b % 2 != 0;
            if (capture && propagation) layer_capture[static_cast<std::size_t>(j)].a_plus = down;
            down = d2::step_apply(steps[static_cast<std::size_t>(b)], down);
            steps[static_cast<std::size_t>(b)] = d2::StepData{};
            if (capture && propagation)
                layer_capture[static_cast<std::size_t>(j)].a_minus = d2::rstate_apply(r_below[static_cast<std::size_t>(j)], down);
        }
        T = std::move(down);
        for (std::size_t i = 0; i < R.size(); ++i)
            if (!d2::is_finite(R[i]) || !d2::is_finite(T[i])) throw d2::NumericalFailure2D{"振幅中出现 NaN/Inf"};

        // ---- 6. 效率与 s/p 复振幅
        const auto efficiency = [&](const Vector& F, const Vector& kz, double sign, Vector& Ex, Vector& Ey, Vector& Ez,
                                    RealVector& de) {
            Ex.assign(F.begin(), F.begin() + N);
            Ey.assign(F.begin() + N, F.end());
            Ez.assign(Nz, cplx{});
            de.assign(Nz, 0.0);
            for (std::size_t i = 0; i < Nz; ++i) {
                if (!(kz[i].real() > 1e-12)) continue;
                Ez[i] = -sign * (Kx[i] * Ex[i] + Ky[i] * Ey[i]) / kz[i];
                const double E2 = std::norm(Ex[i]) + std::norm(Ey[i]) + std::norm(Ez[i]);
                de[i] = kz[i].real() * E2 / (nI * cos_th);
            }
        };
        const auto spol = [&](const Vector& Ex, const Vector& Ey, const Vector& kz, cplx n_med, double sign, Vector& s_, Vector& p_) {
            s_.resize(Nz);
            p_.resize(Nz);
            for (std::size_t i = 0; i < Nz; ++i) {
                const double kt = std::hypot(Kx[i], Ky[i]);
                const cplx cth = std::abs(kz[i]) > 1e-12 ? kz[i] / n_med : cplx{1.0, 0.0};
                if (kt > 1e-12) {
                    s_[i] = (-Ky[i] * Ex[i] + Kx[i] * Ey[i]) / kt;
                    p_[i] = sign * (Kx[i] * Ex[i] + Ky[i] * Ey[i]) / (kt * cth);
                } else {
                    s_[i] = -sin_ph * Ex[i] + cos_ph * Ey[i];
                    p_[i] = sign * (cos_ph * Ex[i] + sin_ph * Ey[i]) / cth;
                }
            }
        };
        efficiency(R, result.kzI, -1.0, result.Ex_r, result.Ey_r, result.Ez_r, result.DEr);
        efficiency(T, result.kzII, +1.0, result.Ex_t, result.Ey_t, result.Ez_t, result.DEt);
        spol(result.Ex_r, result.Ey_r, result.kzI, cplx{nI, 0.0}, -1.0, result.Rs, result.Rp);
        spol(result.Ex_t, result.Ey_t, result.kzII, nII, +1.0, result.Ts, result.Tp);
        for (std::size_t i = 0; i < Nz; ++i)
            if (!std::isfinite(result.DEr[i]) || !std::isfinite(result.DEt[i]) || !d2::is_finite(result.Rs[i]) ||
                !d2::is_finite(result.Rp[i]) || !d2::is_finite(result.Ts[i]) || !d2::is_finite(result.Tp[i]))
                throw d2::NumericalFailure2D{"效率或 s/p 振幅中出现 NaN/Inf"};

        // ---- 7. 近场快照
        if (capture) {
            d2::FieldCapture2D& cap = *capture;
            cap.N = N;
            cap.M1 = M1;
            cap.M2 = M2;
            cap.k0 = k0;
            cap.theta = th;
            cap.phi = ph;
            cap.nI = nI;
            cap.nII = nII;
            cap.Kx = Kx;
            cap.Ky = Ky;
            cap.region_I.q = std::move(modes.front().q);
            cap.region_I.V = std::move(modes.front().Vb);
            cap.region_I.kz = result.kzI;
            cap.region_II.q = std::move(modes.back().q);
            cap.region_II.V = std::move(modes.back().Vb);
            cap.region_II.kz = result.kzII;
            cap.inc = std::move(inc);
            cap.R = std::move(R);
            cap.T = std::move(T);
            double z = 0.0;
            for (int j = 0; j < L; ++j) {
                d2::LayerCapture2D& lc = layer_capture[static_cast<std::size_t>(j)];
                d2::Modes2D& md = modes[static_cast<std::size_t>(j) + 1];
                lc.z_top = z;
                lc.d = stack.layers[static_cast<std::size_t>(j)].d;
                z += lc.d;
                lc.q = std::move(md.q);
                if (md.homog) lc.Vbd = std::move(md.Vb);
                else {
                    lc.W = std::move(md.W);
                    lc.V = std::move(md.V);
                }
            }
            cap.z_bot = z;
            cap.layers = std::move(layer_capture);
            bool ok = d2::finite_vector(cap.inc) && d2::finite_vector(cap.R) && d2::finite_vector(cap.T);
            for (const d2::LayerCapture2D& lc : cap.layers)
                ok = ok && d2::finite_vector(lc.a_plus) && d2::finite_vector(lc.a_minus) && d2::finite_vector(lc.q) &&
                     lc.W.all_finite() && lc.V.all_finite() && lc.Ei.all_finite();
            cap.available = ok;
            if (!ok) *capture = d2::FieldCapture2D{};
        }
    } catch (const d2::NumericalFailure2D& failure) {
        result.status = Status2D::NumericalFailure;
        result.message = "数值失败：" + failure.reason;
        if (capture) *capture = d2::FieldCapture2D{};
    } catch (const d2::RayleighFailure2D& failure) {
        result.status = Status2D::RayleighAnomaly;
        result.message = failure.message;
        if (capture) *capture = d2::FieldCapture2D{};
    }
    if (result.status != Status2D::Ok) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const cplx cnan{nan, nan};
        for (Vector* v : {&result.Ex_r, &result.Ey_r, &result.Ez_r, &result.Ex_t, &result.Ey_t, &result.Ez_t, &result.Rs,
                          &result.Rp, &result.Ts, &result.Tp})
            v->assign(Nz, cnan);
        result.DEr.assign(Nz, nan);
        result.DEt.assign(Nz, nan);
    }

    result.n_dense_solve = work.n_solve;
    result.timing.eig = work.timing.eig;
    result.timing.linear = work.timing.linear;
    result.timing.product = work.timing.product;
    result.timing.total = d2::seconds_since(t_start);
    return result;
}

}  // namespace detail2d

// 吃现成算子的求解：stack 的结构层第 j 个对应 ops[j]（Eps、Cx、Cy，N×N，行列下标 = harmonic_index_2d）；
// 均匀层的 ops 项被忽略（可以留空）。wavelength 是真空波长，theta_deg、phi_deg 是入射角、方位角（度），
// (M1, M2) 是两个方向的截断（保留 m = −M1…M1、n = −M2…M2），jones = [Es, Ep]（不归一）。
// options.factorization 在这里不起作用（算子已经装好）；由调用算子装配的 solve2d 使用。
// capture != nullptr 时同一次求解里保留近场要用的数据（见 detail2d::FieldCapture2D）。
inline Result2D solve2d_with_operators(const Stack2D& stack, double wavelength, double theta_deg, double phi_deg, int M1, int M2,
                                       const Jones2D& jones, const std::vector<detail2d::LayerOperators2D>& ops,
                                       const Options2D& options = {}, detail2d::FieldCapture2D* capture = nullptr) {
    return detail2d::solve_impl(stack, wavelength, theta_deg, phi_deg, M1, M2, jones, ops, options, capture, false);
}


// 完整求解：先装配各结构层的 Fourier 算子（Eps、Cx、Cy，用 options.factorization；均匀层不装配），再调用 solve2d_with_operators。
// 装配时算子矩阵奇异（⟦1/ε⟧ 的 Toeplitz 阵出现零主元）或含 NaN/Inf，按数值失败处理（status = NumericalFailure，结果填 NaN，
// message 里带层名）；几何或参数非法照常抛 std::invalid_argument。options.parallel_layers = true 时各结构层的装配也分给线程池。
// timing.operators 是装配用时（并行时是各层之和），计入 timing.total。capture 的含义同 solve2d_with_operators。
inline Result2D solve2d(const Stack2D& stack, double wavelength, double theta_deg, double phi_deg, int M1, int M2,
                        const Jones2D& jones, const Options2D& options = {}, detail2d::FieldCapture2D* capture = nullptr) {
    namespace d2 = detail2d;
    const auto t_start = d2::Clock::now();
    if (M1 < 0 || M2 < 0) throw std::invalid_argument("rcwa2d: 截断阶数 M1、M2 不能为负");
    const int L = static_cast<int>(stack.layers.size());
    std::vector<d2::LayerOperators2D> ops(static_cast<std::size_t>(L));
    std::vector<std::exception_ptr> invalid(static_cast<std::size_t>(L));    // 几何/参数非法：装配结束后原样重抛
    std::vector<std::string> singular(static_cast<std::size_t>(L));          // 非空 = 该层算子数值失败
    std::vector<double> spent(static_cast<std::size_t>(L), 0.0);
    int patterned = 0;
    for (const Layer2D& layer : stack.layers)
        if (!layer.homogeneous) ++patterned;
    auto assemble = [&](int j) {
        const Layer2D& layer = stack.layers[static_cast<std::size_t>(j)];
        if (layer.homogeneous) return;
        const auto t0 = d2::Clock::now();
        try {
            ops[static_cast<std::size_t>(j)] =
                d2::layer_operators(layer, stack.Lx, stack.Ly, M1, M2, options.factorization, false);
        } catch (const std::invalid_argument&) {
            invalid[static_cast<std::size_t>(j)] = std::current_exception();
        } catch (const std::runtime_error& e) {
            singular[static_cast<std::size_t>(j)] = e.what();
        }
        spent[static_cast<std::size_t>(j)] = d2::seconds_since(t0);
    };
    if (options.parallel_layers && patterned >= 2) parallel_for(0, L, assemble);
    else
        for (int j = 0; j < L; ++j) assemble(j);
    for (int j = 0; j < L; ++j)
        if (invalid[static_cast<std::size_t>(j)]) std::rethrow_exception(invalid[static_cast<std::size_t>(j)]);

    std::string failure;
    const int N = (2 * M1 + 1) * (2 * M2 + 1);
    for (int j = 0; j < L; ++j)
        if (!singular[static_cast<std::size_t>(j)].empty()) {
            if (failure.empty()) failure = singular[static_cast<std::size_t>(j)];
            // 占位：尺寸正确、含 NaN 的算子，使核心按“算子含 NaN”走数值失败路径，同时照常校验其余参数、给出 Kx 等数组
            d2::LayerOperators2D& op = ops[static_cast<std::size_t>(j)];
            op.eps = Matrix(N, N); op.cx = Matrix(N, N); op.cy = Matrix(N, N);
            const double nan = std::numeric_limits<double>::quiet_NaN();
            op.eps(0, 0) = cplx{nan, nan};
        }
    double assembled = 0.0;
    for (double v : spent) assembled += v;

    Result2D result = d2::solve_impl(stack, wavelength, theta_deg, phi_deg, M1, M2, jones, ops, options, capture, false);
    if (!failure.empty() && result.status == Status2D::NumericalFailure) result.message = "数值失败：" + failure;
    result.timing.operators = assembled;
    result.timing.total = d2::seconds_since(t_start);
    return result;
}

}  // namespace rcwa
}  // namespace stablercwa
