// stablercwa/eig.hpp —— 本征分解的统一入口，以及两个衡量结果好坏的量
//
//   EigResult r = eig(A);                          // 默认用 QR 迭代
//   EigResult r = eig(A, EigMethod::LR);           // 只换迭代方法，其余完全相同
//   double res  = eig_residual(A, r);              // 最差的那一对 (λ, v) 满足 A v = λ v 到什么程度
//   double orth = orthogonality_defect(r.vectors); // 本征向量离“两两正交”有多远
//
// 两种方法的接口、选项、返回值约定一致（见 eig_types.hpp），差别只在所用变换的种类：
//   EigMethod::QR —— 酉变换（Householder 反射 + 旋转），见 eig_qr.hpp；
//   EigMethod::LR —— 带行交换的初等消元，见 eig_lr.hpp。
#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>

#include "eig_lr.hpp"
#include "eig_qr.hpp"
#include "eig_types.hpp"
#include "matrix.hpp"

namespace stablercwa {

// 一般复矩阵的本征值与右本征向量。method 选迭代方法，options 原样转交给 eig_qr / eig_lr。
inline EigResult eig(const Matrix& a, EigMethod method = EigMethod::QR, const EigOptions& options = {}) {
    switch (method) {
        case EigMethod::QR: return eig_qr(a, options);
        case EigMethod::LR: return eig_lr(a, options);
    }
    throw std::invalid_argument("eig: unknown method");
}

// 本征对的相对残差，取最差的一对：
//     max_j ||A v_j − λ_j v_j||_2 / (||A||_F · ||v_j||_2)
// 后向稳定的算法给出机器精度的小倍数（与本征值本身好不好算无关）；变换的条件数很大时它会被放大。
// r 里没有本征向量时（迭代未收敛，或调用时 options.vectors = false）无从计算，返回 NaN。
// 零向量不是合法的本征向量，记为 +inf。a 不是方阵、r 的尺寸与 a 对不上时抛 std::invalid_argument。
inline double eig_residual(const Matrix& a, const EigResult& r) {
    if (!a.square()) throw std::invalid_argument("eig_residual: matrix must be square");
    const int n = a.rows();
    if (static_cast<int>(r.values.size()) != n) throw std::invalid_argument("eig_residual: wrong number of eigenvalues");
    if (n == 0) return 0.0;
    if (r.vectors.empty()) return std::numeric_limits<double>::quiet_NaN();
    if (r.vectors.rows() != n || r.vectors.cols() != n)
        throw std::invalid_argument("eig_residual: eigenvector matrix has wrong shape");

    const double inf = std::numeric_limits<double>::infinity();
    const double anorm = norm_fro(a);
    const Matrix av = a * r.vectors;
    Vector v(static_cast<std::size_t>(n)), defect(static_cast<std::size_t>(n));
    double worst = 0.0;
    for (int j = 0; j < n; ++j) {
        const cplx lambda = r.values[j];
        const cplx* vj = r.vectors.col(j);
        const cplx* avj = av.col(j);
        for (int i = 0; i < n; ++i) {
            v[i] = vj[i];
            defect[i] = avj[i] - lambda * vj[i];
        }
        const double vnorm = norm2(v), dnorm = norm2(defect);
        double value;
        if (vnorm == 0.0) value = inf;
        else if (anorm == 0.0) value = dnorm == 0.0 ? 0.0 : inf;      // 零矩阵：本征值只能是 0
        else value = (dnorm / anorm) / vnorm;                         // 分两次除，免得 anorm·vnorm 上溢或下溢
        if (std::isnan(value)) return value;                          // 本征对里有 NaN：残差也是 NaN
        if (value > worst) worst = value;
    }
    return worst;
}

// ||V^H V − I||_F：各列是否标准正交。厄米（或正规）矩阵的本征向量本应两两正交，
// 酉变换算出来的结果保持这一点（机器精度量级），消元变换则不保证。
// v 没有列（例如迭代未收敛、没有本征向量）时返回 NaN。
inline double orthogonality_defect(const Matrix& v) {
    if (v.cols() == 0) return std::numeric_limits<double>::quiet_NaN();
    Matrix g = adjoint(v) * v;
    for (int j = 0; j < g.cols(); ++j) g(j, j) -= 1.0;
    return norm_fro(g);
}

}  // namespace stablercwa
