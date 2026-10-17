// stablercwa/matrix.hpp —— 复数稠密矩阵（按列存放）与基本运算
//
// 整个模块只有这一种矩阵类型：元素为 std::complex<double>，下标从 0 开始，
// A(i, j) 是第 i 行第 j 列；内存按列连续存放，data()[i + j*rows()] 就是 A(i, j)。
// 实矩阵、厄米矩阵、有吸收（复折射率）时的一般复矩阵都用它表示。
// （代码里 (std::min)(a, b) 这种多一层括号的写法，是为了不被 <windows.h> 的 min / max 宏展开，见 parallel.hpp。）
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "parallel.hpp"

namespace stablercwa {

using cplx = std::complex<double>;
using Vector = std::vector<cplx>;
using RealVector = std::vector<double>;

inline constexpr double kPi = 3.141592653589793238462643383279502884;
inline constexpr double kEps = std::numeric_limits<double>::epsilon();   // 2.22e-16
inline const cplx kI{0.0, 1.0};

// |Re| + |Im|：EISPACK 里用它代替模，省一次开方；与 |z| 相差不超过 sqrt(2) 倍。
inline double abs1(const cplx& z) { return std::abs(z.real()) + std::abs(z.imag()); }

class Matrix {
public:
    Matrix() = default;
    Matrix(int rows, int cols) : rows_(rows), cols_(cols), a_(checked(rows, cols)) {}
    Matrix(int rows, int cols, cplx fill) : rows_(rows), cols_(cols), a_(checked(rows, cols), fill) {}
    // 按行书写的初始化：Matrix{{1, 2}, {3, 4}}
    Matrix(std::initializer_list<std::initializer_list<cplx>> rows) {
        rows_ = static_cast<int>(rows.size());
        cols_ = rows_ ? static_cast<int>(rows.begin()->size()) : 0;
        a_.assign(static_cast<std::size_t>(rows_) * cols_, cplx{});
        int i = 0;
        for (const auto& row : rows) {
            if (static_cast<int>(row.size()) != cols_) throw std::invalid_argument("Matrix: ragged initializer");
            int j = 0;
            for (const auto& v : row) (*this)(i, j++) = v;
            ++i;
        }
    }

    static Matrix identity(int n) {
        Matrix m(n, n);
        for (int i = 0; i < n; ++i) m(i, i) = 1.0;
        return m;
    }
    static Matrix diagonal(const Vector& d) {
        const int n = static_cast<int>(d.size());
        Matrix m(n, n);
        for (int i = 0; i < n; ++i) m(i, i) = d[i];
        return m;
    }

    int rows() const noexcept { return rows_; }
    int cols() const noexcept { return cols_; }
    bool empty() const noexcept { return a_.empty(); }
    bool square() const noexcept { return rows_ == cols_; }

    cplx& operator()(int i, int j) noexcept { return a_[static_cast<std::size_t>(j) * rows_ + i]; }
    const cplx& operator()(int i, int j) const noexcept { return a_[static_cast<std::size_t>(j) * rows_ + i]; }
    cplx* data() noexcept { return a_.data(); }
    const cplx* data() const noexcept { return a_.data(); }
    cplx* col(int j) noexcept { return a_.data() + static_cast<std::size_t>(j) * rows_; }
    const cplx* col(int j) const noexcept { return a_.data() + static_cast<std::size_t>(j) * rows_; }

    Vector column(int j) const { return Vector(col(j), col(j) + rows_); }
    Vector diag() const {
        Vector d((std::min)(rows_, cols_));
        for (int i = 0; i < static_cast<int>(d.size()); ++i) d[i] = (*this)(i, i);
        return d;
    }
    bool all_finite() const {
        for (const auto& v : a_)
            if (!std::isfinite(v.real()) || !std::isfinite(v.imag())) return false;
        return true;
    }

    Matrix& operator+=(const Matrix& b) {
        same_shape(b);
        for (std::size_t k = 0; k < a_.size(); ++k) a_[k] += b.a_[k];
        return *this;
    }
    Matrix& operator-=(const Matrix& b) {
        same_shape(b);
        for (std::size_t k = 0; k < a_.size(); ++k) a_[k] -= b.a_[k];
        return *this;
    }
    Matrix& operator*=(cplx s) {
        for (auto& v : a_) v *= s;
        return *this;
    }

private:
    static std::size_t checked(int rows, int cols) {
        if (rows < 0 || cols < 0) throw std::invalid_argument("Matrix: negative size");
        return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    }
    void same_shape(const Matrix& b) const {
        if (rows_ != b.rows_ || cols_ != b.cols_) throw std::invalid_argument("Matrix: shape mismatch");
    }
    int rows_ = 0, cols_ = 0;
    std::vector<cplx> a_;
};

inline Matrix operator+(Matrix a, const Matrix& b) { return a += b; }
inline Matrix operator-(Matrix a, const Matrix& b) { return a -= b; }
inline Matrix operator*(Matrix a, cplx s) { return a *= s; }
inline Matrix operator*(cplx s, Matrix a) { return a *= s; }
inline Matrix operator-(Matrix a) { return a *= cplx{-1.0, 0.0}; }

namespace detail {

// c ← c + t·a（长度 len 的复数列）。实部、虚部分开写：std::complex 的乘法带有对 NaN/Inf 的特殊处理，
// 会挡住编译器把循环向量化；写成四个实数乘法之后同一个循环快 2~3 倍。
// （标准保证 std::complex<double> 的数组可以当作实部、虚部交错的 double 数组来读写。）
inline void add_column(cplx* c_, const cplx* a_, cplx t, int len) noexcept {
    double* c = reinterpret_cast<double*>(c_);
    const double* a = reinterpret_cast<const double*>(a_);
    const double tr = t.real(), ti = t.imag();
    for (int i = 0; i < len; ++i) {
        const int r = 2 * i, s = r + 1;
        const double ar = a[r], ai = a[s];
        c[r] += ar * tr - ai * ti;
        c[s] += ar * ti + ai * tr;
    }
}

// c ← c + t0·a0 + t1·a1 + t2·a2 + t3·a3。四列一起加，c 的每个元素只读写一次。各项相加的次序固定。
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

}  // namespace detail

// C = A * B。C 的每一列是 A 的各列的线性组合，四列一组累加，内层循环连续访问内存；
// B 里整组为零的系数直接跳过（B 是对角矩阵时只剩 O(n²) 的工作量）。
// 列与列互不相干，矩阵够大时按列并行；每个元素上各项相加的次序固定，所以结果与线程数无关。
inline Matrix operator*(const Matrix& a, const Matrix& b) {
    if (a.cols() != b.rows()) throw std::invalid_argument("Matrix product: shape mismatch");
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
                detail::add_columns4(cj, a.col(k), a.col(k + 1), a.col(k + 2), a.col(k + 3), t, m);
            }
            for (; k < p; ++k)
                if (bj[k] != cplx{}) detail::add_column(cj, a.col(k), bj[k], m);
        }
    };
    const long long work = static_cast<long long>(m) * n * p;
    if (work < 200000) block(0, n);
    else parallel_ranges(0, n, block, 4);
    return c;
}

// y = A * x
inline Vector operator*(const Matrix& a, const Vector& x) {
    if (a.cols() != static_cast<int>(x.size())) throw std::invalid_argument("Matrix*Vector: shape mismatch");
    Vector y(a.rows());
    for (int k = 0; k < a.cols(); ++k) {
        const cplx t = x[k];
        const cplx* ak = a.col(k);
        for (int i = 0; i < a.rows(); ++i) y[i] += ak[i] * t;
    }
    return y;
}

// A * diag(d)：把第 j 列乘以 d[j]。
inline Matrix scale_columns(Matrix a, const Vector& d) {
    if (a.cols() != static_cast<int>(d.size())) throw std::invalid_argument("scale_columns: size mismatch");
    for (int j = 0; j < a.cols(); ++j) {
        cplx* aj = a.col(j);
        for (int i = 0; i < a.rows(); ++i) aj[i] *= d[j];
    }
    return a;
}
// diag(d) * A：把第 i 行乘以 d[i]。
inline Matrix scale_rows(Matrix a, const Vector& d) {
    if (a.rows() != static_cast<int>(d.size())) throw std::invalid_argument("scale_rows: size mismatch");
    for (int j = 0; j < a.cols(); ++j) {
        cplx* aj = a.col(j);
        for (int i = 0; i < a.rows(); ++i) aj[i] *= d[i];
    }
    return a;
}

inline Matrix transpose(const Matrix& a) {
    Matrix t(a.cols(), a.rows());
    for (int j = 0; j < a.cols(); ++j)
        for (int i = 0; i < a.rows(); ++i) t(j, i) = a(i, j);
    return t;
}
// 共轭转置 A^H
inline Matrix adjoint(const Matrix& a) {
    Matrix t(a.cols(), a.rows());
    for (int j = 0; j < a.cols(); ++j)
        for (int i = 0; i < a.rows(); ++i) t(j, i) = std::conj(a(i, j));
    return t;
}

inline double norm_fro(const Matrix& a) {
    double scale = 0.0, ssq = 1.0;          // 防上溢/下溢的缩放求和
    const cplx* p = a.data();
    const std::size_t n = static_cast<std::size_t>(a.rows()) * a.cols();
    for (std::size_t k = 0; k < n; ++k) {
        for (double v : {std::abs(p[k].real()), std::abs(p[k].imag())}) {
            if (v == 0.0) continue;
            if (scale < v) {
                ssq = 1.0 + ssq * (scale / v) * (scale / v);
                scale = v;
            } else {
                ssq += (v / scale) * (v / scale);
            }
        }
    }
    return scale * std::sqrt(ssq);
}
// 1-范数：各列绝对值之和的最大值。
inline double norm_1(const Matrix& a) {
    double best = 0.0;
    for (int j = 0; j < a.cols(); ++j) {
        double s = 0.0;
        for (int i = 0; i < a.rows(); ++i) s += std::abs(a(i, j));
        best = (std::max)(best, s);
    }
    return best;
}
inline double norm_inf(const Matrix& a) {
    RealVector s(a.rows(), 0.0);
    for (int j = 0; j < a.cols(); ++j)
        for (int i = 0; i < a.rows(); ++i) s[i] += std::abs(a(i, j));
    return a.rows() ? *std::max_element(s.begin(), s.end()) : 0.0;
}
inline double max_abs(const Matrix& a) {
    double best = 0.0;
    const cplx* p = a.data();
    const std::size_t n = static_cast<std::size_t>(a.rows()) * a.cols();
    for (std::size_t k = 0; k < n; ++k) best = (std::max)(best, std::abs(p[k]));
    return best;
}
inline double norm2(const Vector& x) {
    double scale = 0.0, ssq = 1.0;
    for (const auto& z : x) {
        for (double v : {std::abs(z.real()), std::abs(z.imag())}) {
            if (v == 0.0) continue;
            if (scale < v) {
                ssq = 1.0 + ssq * (scale / v) * (scale / v);
                scale = v;
            } else {
                ssq += (v / scale) * (v / scale);
            }
        }
    }
    return scale * std::sqrt(ssq);
}

// A 是否厄米（A = A^H），容差相对 max|a_ij|。
inline bool is_hermitian(const Matrix& a, double tol = 0.0) {
    if (!a.square()) return false;
    const double lim = tol * max_abs(a);
    for (int j = 0; j < a.cols(); ++j)
        for (int i = 0; i <= j; ++i)
            if (std::abs(a(i, j) - std::conj(a(j, i))) > lim) return false;
    return true;
}

}  // namespace stablercwa
