// tests/test_eig_qr.cpp —— eig_qr()（CORTH + COMQR2）的测试
//
//   c++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -I include tests/test_eig_qr.cpp -o build/test_eig_qr
//
// 只依赖 C++17 标准库。测试矩阵由下面的 Rng（splitmix64）生成，种子固定，不同平台上得到同样的矩阵。
// 主要度量：
//   残差      max_j ||A v_j − λ_j v_j||_2 / (||A||_F ||v_j||_2)
//   迹        |Σ λ_j − tr A| / ||A||_F
//   本征值差  与已知本征值最近邻配对后的最大距离
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "stablercwa/eig_qr.hpp"
#include "stablercwa/svd.hpp"

using namespace stablercwa;

// ---------------------------------------------------------------------------------------------
// 输出与计数用的断言
//
// 文字一律经 say() 输出（用法同 std::printf），不把带中文的格式串直接交给 printf 系列函数：
// MSVC 检查 printf 的格式串时按系统代码页（中文 Windows 上是 GBK）而不是按 /utf-8 去读它，UTF-8 的汉字
// 会被读错——报 C4819；汉字后面紧跟的 % 还可能被当成半个汉字吞掉，于是误报参数个数、类型不符
// （C4474、C4477）。经 say() 转一道手，MSVC 不再解析格式串；GCC、Clang 仍照 printf 的规则核对参数
// （MinGW 除外：那里 printf 这个原型按微软 C 库的旧规则检查，不认 %zu）。
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
// value <= bound；失败时把实际数值打印出来。NaN 也算失败。
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
// 固定种子的随机数与测试矩阵
struct Rng {
    std::uint64_t s;
    explicit Rng(std::uint64_t seed) : s(seed) {}
    std::uint64_t next() {                           // splitmix64
        std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    double uniform() { return static_cast<double>(next() >> 11) * (2.0 / 9007199254740992.0) - 1.0; }   // (-1, 1)
    cplx cuniform() {
        const double re = uniform();
        return cplx(re, uniform());
    }
};

static Matrix random_complex(int n, std::uint64_t seed) {
    Rng rng(seed);
    Matrix a(n, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) a(i, j) = rng.cuniform();
    return a;
}

static Matrix random_real(int n, std::uint64_t seed) {
    Rng rng(seed);
    Matrix a(n, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) a(i, j) = rng.uniform();
    return a;
}

// 随机酉矩阵：n 个随机 Householder 反射的乘积。
static Matrix random_unitary(int n, std::uint64_t seed) {
    Rng rng(seed);
    Matrix q = Matrix::identity(n);
    Vector v(n), qv(n);
    for (int k = 0; k < n; ++k) {
        double vv = 0.0;
        for (int i = 0; i < n; ++i) {
            v[i] = rng.cuniform();
            vv += std::norm(v[i]);
        }
        qv = q * v;                                  // Q ← Q (I − 2 v v^H / v^H v)
        for (int j = 0; j < n; ++j) {
            const cplx f = 2.0 * std::conj(v[j]) / vv;
            for (int i = 0; i < n; ++i) q(i, j) -= qv[i] * f;
        }
    }
    return q;
}

// 取厄米部分 (M + M^H)/2，结果在浮点意义下严格厄米。
static Matrix hermitian_part(const Matrix& m) {
    const int n = m.rows();
    Matrix a(n, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) a(i, j) = 0.5 * (m(i, j) + std::conj(m(j, i)));
    return a;
}

// 首一多项式 Π (x − r_k) 的友矩阵：次对角线为 1，最后一列是 −c_0 … −c_{n−1}。
static Matrix companion(const Vector& roots) {
    const int n = static_cast<int>(roots.size());
    Vector c(n + 1, cplx{});                         // c[k] 是 x^k 的系数
    c[0] = 1.0;
    for (int k = 0; k < n; ++k) {
        for (int i = k + 1; i >= 1; --i) c[i] = c[i - 1] - roots[k] * c[i];
        c[0] = -roots[k] * c[0];
    }
    Matrix a(n, n);
    for (int i = 1; i < n; ++i) a(i, i - 1) = 1.0;
    for (int i = 0; i < n; ++i) a(i, n - 1) = -c[i];
    return a;
}

// RCWA 的 TE 层矩阵（以 k0² 为单位）：A = diag(kx_m²) − Toeplitz(ε_h)。
// 二元光栅：一个周期内占空比 fill 的部分折射率为 n_ridge，其余为 n_groove；ε_h 是 ε(x) 的 Fourier 系数，
// kx_m = sinθ + m·λ/Λ，m = −M … M，n = 2M + 1。折射率为实数时 A 厄米；有吸收（复折射率）时是一般复矩阵。
static Matrix rcwa_te_matrix(int n, cplx n_ridge, cplx n_groove, double fill, double lambda_over_period, double sin_theta) {
    const int M = n / 2;
    const cplx e1 = n_ridge * n_ridge, e2 = n_groove * n_groove;
    Matrix a(n, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            const int h = i - j;
            const cplx eps_h = h == 0 ? fill * e1 + (1.0 - fill) * e2 : (e1 - e2) * (std::sin(kPi * h * fill) / (kPi * h));
            a(i, j) = -eps_h;
        }
    for (int i = 0; i < n; ++i) {
        const double kx = sin_theta + (i - M) * lambda_over_period;
        a(i, i) += kx * kx;
    }
    return a;
}

// ---------------------------------------------------------------------------------------------
// 度量
static double eig_residual(const Matrix& a, const EigResult& r) {
    const int n = a.rows();
    if (r.vectors.rows() != n || r.vectors.cols() != n) return std::numeric_limits<double>::infinity();
    const double na = norm_fro(a);
    const Matrix av = a * r.vectors;
    double worst = 0.0;
    for (int j = 0; j < n; ++j) {
        Vector d(n);
        for (int i = 0; i < n; ++i) d[i] = av(i, j) - r.values[j] * r.vectors(i, j);
        const double res = norm2(d) / (na * norm2(r.vectors.column(j)));
        if (!(res <= worst)) worst = res;            // NaN 也会被留下
    }
    return worst;
}

static double trace_defect(const Matrix& a, const Vector& values) {
    cplx sum{}, tr{};
    for (int i = 0; i < a.rows(); ++i) {
        sum += values[i];
        tr += a(i, i);
    }
    return std::abs(sum - tr) / norm_fro(a);
}

// 各列 2-范数与 1 的最大偏差
static double unit_norm_defect(const Matrix& v) {
    double worst = 0.0;
    for (int j = 0; j < v.cols(); ++j) worst = std::max(worst, std::abs(norm2(v.column(j)) - 1.0));
    return worst;
}

// ||V^H V − I||_F
static double orthogonality_defect(const Matrix& v) {
    Matrix g = adjoint(v) * v;
    for (int i = 0; i < g.rows(); ++i) g(i, i) -= 1.0;
    return norm_fro(g);
}

// 两组数最近邻配对（贪心）后的最大距离
static double match_distance(const Vector& got, const Vector& expected) {
    if (got.size() != expected.size()) return std::numeric_limits<double>::infinity();
    std::vector<char> used(expected.size(), 0);
    double worst = 0.0;
    for (const cplx& g : got) {
        int best = -1;
        double bd = std::numeric_limits<double>::infinity();
        for (std::size_t k = 0; k < expected.size(); ++k) {
            if (used[k]) continue;
            const double d = std::abs(g - expected[k]);
            if (d < bd) {
                bd = d;
                best = static_cast<int>(k);
            }
        }
        if (best < 0) return std::numeric_limits<double>::infinity();   // 出现了 NaN
        used[best] = 1;
        worst = std::max(worst, bd);
    }
    return worst;
}

static double max_abs(const Vector& v) {
    double m = 0.0;
    for (const cplx& z : v) m = std::max(m, std::abs(z));
    return m;
}

static bool all_finite(const Vector& v) {
    for (const cplx& z : v)
        if (!std::isfinite(z.real()) || !std::isfinite(z.imag())) return false;
    return true;
}

// ||Z^H A Z − T||_F / ||A||_F
static double schur_defect(const Matrix& a, const Matrix& z, const Matrix& t) { return norm_fro(adjoint(z) * a * z - t) / norm_fro(a); }

static bool strictly_lower_is_zero(const Matrix& t, int below = 1) {
    for (int j = 0; j < t.cols(); ++j)
        for (int i = j + below; i < t.rows(); ++i)
            if (t(i, j) != cplx{}) return false;
    return true;
}

// 第 first_row 行及其以下是否为严格的上三角（对角线以下全为零）
static bool lower_rows_triangular(const Matrix& t, int first_row) {
    for (int j = 0; j < t.cols(); ++j)
        for (int i = std::max(j + 1, first_row); i < t.rows(); ++i)
            if (t(i, j) != cplx{}) return false;
    return true;
}

// 每个元素乘 2^k：只改指数，不引入舍入
static cplx times_pow2(cplx z, int k) { return cplx(std::ldexp(z.real(), k), std::ldexp(z.imag(), k)); }
static Matrix times_pow2(const Matrix& a, int k) {
    Matrix b(a.rows(), a.cols());
    for (int j = 0; j < a.cols(); ++j)
        for (int i = 0; i < a.rows(); ++i) b(i, j) = times_pow2(a(i, j), k);
    return b;
}

static bool same_bits(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t k = 0; k < a.size(); ++k)
        if (a[k].real() != b[k].real() || a[k].imag() != b[k].imag()) return false;
    return true;
}
static bool same_bits(const Matrix& a, const Matrix& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    for (int j = 0; j < a.cols(); ++j)
        for (int i = 0; i < a.rows(); ++i)
            if (a(i, j).real() != b(i, j).real() || a(i, j).imag() != b(i, j).imag()) return false;
    return true;
}

// 常用的一组检查：收敛、有限、残差、迹、向量归一。返回残差供打印。
static double check_decomposition(const Matrix& a, const EigResult& r, double residual_tol = 1e-13) {
    const int n = a.rows();
    CHECK(r.info == 0);
    CHECK(static_cast<int>(r.values.size()) == n);
    CHECK(r.vectors.rows() == n && r.vectors.cols() == n);
    CHECK(all_finite(r.values) && r.vectors.all_finite());
    const double res = eig_residual(a, r);
    CHECK_LE(res, residual_tol);
    CHECK_LE(trace_defect(a, r.values), 1e-13);
    CHECK_LE(unit_norm_defect(r.vectors), 1e-14);
    return res;
}

// ---------------------------------------------------------------------------------------------
static void test_small() {
    say("[n = 0, 1, 2, 3]\n");
    {
        const EigResult r = eig_qr(Matrix());
        CHECK(r.info == 0 && r.values.empty() && r.vectors.empty() && r.sweeps == 0);
    }
    {
        const Matrix a{{cplx(3.0, -2.0)}};
        const EigResult r = eig_qr(a);
        CHECK(r.info == 0 && r.sweeps == 0);
        CHECK(r.values.size() == 1 && r.values[0] == cplx(3.0, -2.0));
        CHECK(r.vectors.rows() == 1 && std::abs(std::abs(r.vectors(0, 0)) - 1.0) <= 1e-15);
    }
    {
        const Matrix a{{2.0, 1.0}, {1.0, 2.0}};                         // 本征值 1、3
        const EigResult r = eig_qr(a);
        check_decomposition(a, r);
        CHECK_LE(match_distance(r.values, Vector{1.0, 3.0}), 1e-14);
    }
    {
        const Matrix a{{0.0, 1.0}, {-1.0, 0.0}};                        // 本征值 ±i
        const EigResult r = eig_qr(a);
        check_decomposition(a, r);
        CHECK_LE(match_distance(r.values, Vector{kI, -kI}), 1e-14);
    }
    {
        const Matrix a{{cplx(1.0, 1.0), 2.0}, {3.0, cplx(4.0, -1.0)}};  // 特征方程 λ² − 5λ + (−1 + 3i) = 0
        const EigResult r = eig_qr(a);
        check_decomposition(a, r);
        const cplx disc = std::sqrt(cplx(29.0, -12.0));
        CHECK_LE(match_distance(r.values, Vector{(5.0 + disc) / 2.0, (5.0 - disc) / 2.0}), 1e-14);
    }
    {
        const Matrix a{{2.0, 0.0, 0.0}, {0.0, 3.0, 4.0}, {0.0, 4.0, 9.0}};   // 本征值 2、1、11
        const EigResult r = eig_qr(a);
        check_decomposition(a, r);
        CHECK_LE(match_distance(r.values, Vector{1.0, 2.0, 11.0}), 1e-13);
    }
    {
        const Matrix a{{0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}};   // 循环置换：三个单位根
        const EigResult r = eig_qr(a);
        check_decomposition(a, r);
        const cplx w1 = std::polar(1.0, 2.0 * kPi / 3.0);
        CHECK_LE(match_distance(r.values, Vector{1.0, w1, std::conj(w1)}), 1e-14);
    }
    {
        const Matrix a = random_complex(3, 11);
        const EigResult r = eig_qr(a);
        check_decomposition(a, r);
    }
}

static void test_diagonal_and_triangular() {
    say("[对角与三角矩阵：不需要迭代]\n");
    {
        const Vector d{cplx(1.0, 2.0), -3.0, cplx(0.0, 0.5), 4.0, cplx(-2.0, -2.0), 0.0};
        const Matrix a = Matrix::diagonal(d);
        const EigResult r = eig_qr(a);
        check_decomposition(a, r);
        CHECK(r.sweeps == 0);
        CHECK(same_bits(r.values, d));               // 本征值就是对角元，顺序不变
        bool unit_columns = true;                    // 本征向量就是坐标轴
        for (int j = 0; j < 6; ++j)
            for (int i = 0; i < 6; ++i) unit_columns = unit_columns && std::abs(std::abs(r.vectors(i, j)) - (i == j ? 1.0 : 0.0)) <= 1e-15;
        CHECK(unit_columns);
    }
    {
        const int n = 8;
        Matrix a = random_complex(n, 21);
        for (int j = 0; j < n; ++j) {
            for (int i = j + 1; i < n; ++i) a(i, j) = 0.0;
            a(j, j) = cplx(j - 3.5, 0.25 * j);        // 互不相同的对角元
        }
        const EigResult r = eig_qr(a);
        check_decomposition(a, r);
        CHECK(r.sweeps == 0);
        CHECK(same_bits(r.values, a.diag()));
    }
    {
        const int n = 8;                             // 下三角：本征值仍是对角元，但需要迭代
        Matrix a = random_complex(n, 22);
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < j; ++i) a(i, j) = 0.0;
            a(j, j) = cplx(j - 3.5, 0.25 * j);
        }
        const EigResult r = eig_qr(a);
        check_decomposition(a, r);
        CHECK(r.sweeps > 0);
        CHECK_LE(match_distance(r.values, a.diag()), 1e-13);
    }
    {
        const Matrix a(5, 5);                        // 零矩阵
        const EigResult r = eig_qr(a);
        CHECK(r.info == 0 && r.sweeps == 0);
        CHECK(same_bits(r.values, Vector(5, cplx{})));
        CHECK(r.vectors.all_finite());
        CHECK_LE(unit_norm_defect(r.vectors), 1e-15);
        CHECK_LE(orthogonality_defect(r.vectors), 1e-15);
    }
    {
        const Matrix a = Matrix::identity(7);
        const EigResult r = eig_qr(a);
        check_decomposition(a, r);
        CHECK(r.sweeps == 0);
        CHECK(same_bits(r.values, Vector(7, cplx(1.0, 0.0))));
    }
}

static void test_companion() {
    say("[友矩阵：已知根的多项式]\n");
    {
        const Vector roots{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        const Matrix a = companion(roots);
        const EigResult r = eig_qr(a);
        check_decomposition(a, r);
        const double err = match_distance(r.values, roots);
        say("  实根 1…6              误差 %.2e  迭代 %d 次\n", err, r.sweeps);
        CHECK_LE(err, 1e-10);                        // 这个多项式的根本身是病态的（条件数约 1e4~1e5）
    }
    {
        const Vector roots{cplx(1.0, 2.0), cplx(-0.5, 0.3), cplx(2.0, -1.0), cplx(-1.5, -1.5), cplx(0.0, 0.2)};
        const Matrix a = companion(roots);
        const EigResult r = eig_qr(a);
        check_decomposition(a, r);
        const double err = match_distance(r.values, roots);
        say("  5 个复根               误差 %.2e  迭代 %d 次\n", err, r.sweeps);
        CHECK_LE(err, 1e-13);
    }
    for (int n : {8, 16}) {
        // x^n − 1：友矩阵是循环置换（酉矩阵），普通位移恒为 0，QR 步原地不动，要靠第 10 次的例外位移才能启动。
        Vector roots(n);
        for (int k = 0; k < n; ++k) roots[k] = std::polar(1.0, 2.0 * kPi * k / n);
        Matrix a(n, n);
        for (int i = 1; i < n; ++i) a(i, i - 1) = 1.0;
        a(0, n - 1) = 1.0;
        EigOptions keep;
        keep.keep_transform = true;
        const EigResult r = eig_qr(a, keep);
        check_decomposition(a, r);
        const double err = match_distance(r.values, roots);
        say("  x^%d − 1 的 %d 个单位根   误差 %.2e  迭代 %d 次\n", n, n, err, r.sweeps);
        CHECK_LE(err, 1e-13);
        CHECK(r.sweeps > 10);                        // 确实用到了例外位移
        CHECK_LE(schur_defect(a, r.Z, r.T), 1e-13);

        // 把上限设成 10，就在例外位移之前停下：必须如实报告未收敛。
        EigOptions limited;
        limited.max_iter = 10;
        limited.keep_transform = true;
        const EigResult f = eig_qr(a, limited);
        CHECK(f.info == n);                          // 第 n 个本征值（1 起）没收敛
        CHECK(f.sweeps == 10);
        CHECK(f.vectors.empty());
        CHECK(static_cast<int>(f.values.size()) == n && all_finite(f.values));
        CHECK_LE(orthogonality_defect(f.Z), 1e-13);
        CHECK_LE(schur_defect(a, f.Z, f.T), 1e-13);  // 变换关系在未收敛时仍然成立
        CHECK(strictly_lower_is_zero(f.T, 2));       // T 仍是 Hessenberg 形
    }
}

static void test_random_general() {
    say("[随机一般复矩阵]\n");
    for (int n : {5, 20, 60, 150}) {
        const Matrix a = random_complex(n, 100 + n);
        const Matrix copy = a;
        const EigResult r = eig_qr(a);
        const double res = check_decomposition(a, r);
        CHECK(same_bits(a, copy));                   // 输入不被修改
        CHECK(r.Z.empty() && r.T.empty());           // 没要求就不返回 Z、T
        say("  n = %3d  残差 %.2e  迹 %.2e  迭代 %d 次\n", n, res, trace_defect(a, r.values), r.sweeps);
    }
}

static void test_hermitian() {
    say("[厄米不定矩阵]\n");
    for (int n : {10, 40, 100}) {
        // A = Q D Q^H，D 的对角元正负都有、间隔均匀
        Vector d(n);
        for (int i = 0; i < n; ++i) d[i] = -0.7 * n + 2.0 * i + 0.3;
        const Matrix q = random_unitary(n, 300 + n);
        const Matrix a = hermitian_part(q * Matrix::diagonal(d) * adjoint(q));
        CHECK(is_hermitian(a));
        const EigResult r = eig_qr(a);
        const double res = check_decomposition(a, r);
        const double scale = max_abs(r.values);
        double imag = 0.0;
        for (const cplx& z : r.values) imag = std::max(imag, std::abs(z.imag()));
        const double err = match_distance(r.values, d);
        const double orth = orthogonality_defect(r.vectors);
        say("  n = %3d  残差 %.2e  max|Im λ|/max|λ| %.2e  本征值误差/max|λ| %.2e  ||V^H V − I||_F %.2e\n", n, res, imag / scale,
            err / scale, orth);
        CHECK_LE(imag / scale, 1e-12);
        CHECK_LE(err / scale, 1e-12);
        CHECK_LE(orth, 1e-11);
    }
    {
        // 随机厄米矩阵加一条跨越正负的对角线（本征值事先不知道）
        const int n = 60;
        Matrix a = hermitian_part(random_complex(n, 361));
        for (int i = 0; i < n; ++i) a(i, i) += -double(n) + 4.0 * i;
        const EigResult r = eig_qr(a);
        check_decomposition(a, r);
        double imag = 0.0;
        int negative = 0;
        for (const cplx& z : r.values) {
            imag = std::max(imag, std::abs(z.imag()));
            if (z.real() < 0.0) ++negative;
        }
        CHECK_LE(imag / max_abs(r.values), 1e-12);
        CHECK(negative > 0 && negative < n);         // 确实不定
        CHECK_LE(orthogonality_defect(r.vectors), 1e-11);
    }
}

static void test_real_nonsymmetric() {
    say("[实的非对称矩阵：本征值成共轭对]\n");
    for (int n : {9, 30, 64}) {
        const Matrix a = random_real(n, 400 + n);
        const EigResult r = eig_qr(a);
        const double res = check_decomposition(a, r);
        Vector conjugated(r.values);
        int complex_count = 0;
        for (cplx& z : conjugated) {
            if (std::abs(z.imag()) > 1e-10) ++complex_count;
            z = std::conj(z);
        }
        const double pair = match_distance(r.values, conjugated) / max_abs(r.values);
        say("  n = %2d  残差 %.2e  共轭配对误差 %.2e  其中复本征值 %d 个\n", n, res, pair, complex_count);
        CHECK_LE(pair, 1e-12);
        CHECK(complex_count > 0 && complex_count % 2 == 0);
    }
}

static void test_repeated_and_defective() {
    say("[重本征值与亏损矩阵]\n");
    {
        // 可对角化的重本征值：A = Q diag(1,1,1,2,3,3) Q^H
        const Vector d{1.0, 1.0, 1.0, 2.0, 3.0, 3.0};
        const Matrix q = random_unitary(6, 501);
        const Matrix a = hermitian_part(q * Matrix::diagonal(d) * adjoint(q));
        const EigResult r = eig_qr(a);
        const double res = check_decomposition(a, r);
        const double err = match_distance(r.values, d);
        say("  半单重本征值            残差 %.2e  本征值误差 %.2e\n", res, err);
        CHECK_LE(err, 1e-13);
    }
    {
        // Jordan 块直接输入（上三角）：回代时分母恰为零，要走 eps·norm 的替换，不能出 NaN。
        Matrix a(5, 5);
        a(0, 0) = a(1, 1) = a(2, 2) = 2.0;
        a(0, 1) = a(1, 2) = 1.0;
        a(3, 3) = 5.0;
        a(4, 4) = -1.0;
        const EigResult r = eig_qr(a);
        const double res = check_decomposition(a, r);
        CHECK(r.sweeps == 0);
        CHECK(same_bits(r.values, a.diag()));
        say("  J_3(2) ⊕ diag(5, −1)   残差 %.2e\n", res);
    }
    for (int variant = 0; variant < 2; ++variant) {
        // 2×2 Jordan 块经相似变换藏起来：本征值只能准到 sqrt(eps) 量级，但残差仍应很小，且不能出 NaN。
        const int n = 5;
        Matrix j = Matrix::diagonal(Vector{2.0, 2.0, 5.0, -1.0, cplx(0.0, 0.5)});
        j(0, 1) = 1.0;
        Matrix a;
        if (variant == 0) {
            const Matrix q = random_unitary(n, 502);
            a = q * j * adjoint(q);
        } else {
            // 一般的（非酉）相似变换 S J S^{-1}：取 S = L U，L、U 是单位下、上三角矩阵，逆可以精确写出。
            Matrix l = Matrix::identity(n), u = Matrix::identity(n), li = Matrix::identity(n), ui = Matrix::identity(n);
            l(1, 0) = 0.5;  li(1, 0) = -0.5;
            l(4, 3) = -0.25; li(4, 3) = 0.25;
            u(0, 2) = cplx(0.5, 0.5); ui(0, 2) = cplx(-0.5, -0.5);
            u(3, 4) = 2.0;  ui(3, 4) = -2.0;
            a = (l * u) * j * (ui * li);
        }
        const EigResult r = eig_qr(a);
        const double res = check_decomposition(a, r);
        const double err = match_distance(r.values, j.diag());
        say("  隐藏的 2×2 Jordan 块 %s  残差 %.2e  本征值误差 %.2e\n", variant == 0 ? "(酉相似)  " : "(非酉相似)", res, err);
        CHECK_LE(err, 1e-6);
    }
    for (double delta : {5e-324, 1e-310}) {
        // 本征值之差不为零、但小到是次正规数：回代里的商（约 1/δ）会上溢，必须先把向量缩小，不能出 Inf/NaN。
        const Matrix a{{0.0, 1.0}, {0.0, delta}};
        const EigResult r = eig_qr(a);
        CHECK(r.info == 0 && r.sweeps == 0);
        CHECK(r.vectors.all_finite());
        CHECK_LE(unit_norm_defect(r.vectors), 1e-14);
        // λ = δ 的本征向量是 (1, δ)
        CHECK_LE(std::abs(std::abs(r.vectors(0, 1)) - 1.0), 1e-14);
        CHECK_LE(std::abs(r.vectors(1, 1)), 1e-300);

        const int n = 6;                             // 同样的事放进 6 阶上三角矩阵：对角元 0, δ, 2δ, …
        Matrix t = random_complex(n, 503);
        for (int j = 0; j < n; ++j) {
            for (int i = j + 1; i < n; ++i) t(i, j) = 0.0;
            t(j, j) = j * delta;
        }
        const EigResult rt = eig_qr(t);
        CHECK(rt.info == 0 && rt.vectors.all_finite());
        CHECK_LE(unit_norm_defect(rt.vectors), 1e-14);
        const double res = eig_residual(t, rt);
        say("  本征值间隔 %.0e（次正规数）      残差 %.2e\n", delta, res);
        CHECK_LE(res, 1e-13);
    }
    {
        // 三重亏损本征值（复对称矩阵的三阶例外点）：A³ = 0，本征值全为 0。
        // 平常的位移先原地打转，靠例外位移才启动，之后只能线性收敛。每个本征值 30 次的上限在这里处于边缘：
        // 不够时必须如实报告（info > 0、不给向量、数值有限），放宽上限后一定收敛。
        const Matrix a{{0.0, -1.0, 0.0}, {-1.0, 0.0, kI}, {0.0, kI, 0.0}};
        EigOptions opt;
        opt.max_iter = 30;
        const EigResult r = eig_qr(a, opt);
        CHECK(r.info >= 0 && r.info <= 3 && all_finite(r.values));
        if (r.info == 0) CHECK_LE(eig_residual(a, r), 1e-13);
        else CHECK(r.vectors.empty());
        opt.max_iter = 100;
        const EigResult ok = eig_qr(a, opt);
        const double res = check_decomposition(a, ok);
        say("  三重亏损本征值 (A³ = 0)        上限 30：info = %d；上限 100：迭代 %d 次，残差 %.2e，max|λ| %.2e\n", r.info, ok.sweeps, res,
            max_abs(ok.values));
        CHECK_LE(max_abs(ok.values), 1e-4);          // 三重本征值只能准到 eps^(1/3) 量级
    }
}

// 均匀层在正入射时的层矩阵，按“只有一段的光栅”用通用公式构造：对角元 kx_m² − n² 里 +m 与 −m 成对相等；
// 非对角元 −c_h 本该是零，c_h = n²·[exp(−i·2πh) − 1] / (−i·2πh) 在浮点运算里留下 1e-17 ~ 1e-16 的舍入误差。
static Matrix uniform_layer_matrix(int n, double index, double lambda_over_period) {
    const int M = n / 2;
    Matrix a(n, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            const int h = i - j;
            cplx eps_h = index * index;
            if (h != 0) {
                const double arg = -2.0 * kPi * h;
                const cplx t = (index * index) * (cplx{std::cos(arg), std::sin(arg)} - cplx{1.0, 0.0});
                const double w = -2.0 * kPi * h;                  // 除以 i·w
                eps_h = cplx{t.imag() / w, -t.real() / w};
            }
            a(i, j) = -eps_h;
        }
    for (int i = 0; i < n; ++i) {
        const double kx = (i - M) * lambda_over_period;
        a(i, i) += kx * kx;
    }
    return a;
}

// 两个本征值在舍入误差以内重合。迭代之后这样一对本征值之差可以是 1e-30 之类非零但毫无意义的数；
// 回代时若原样拿它去除，这一对本征向量会几乎平行（本征向量矩阵的条件数 1e14 以上），而残差仍然很小，
// 看不出来。这里要求本征向量矩阵保持良态。
static void test_nearly_equal_eigenvalues() {
    say("[舍入误差以内重合的本征值]\n");
    double worst = 0.0, worst_res = 0.0;
    int cases = 0;
    for (int n = 5; n <= 81; n += 4) {
        for (double index : {1.46, 1.96, 2.0}) {
            for (double ratio : {0.0706, 0.61, 1.0}) {
                const Matrix a = uniform_layer_matrix(n, index, ratio);
                const EigResult r = eig_qr(a);
                CHECK(r.info == 0);
                const double res = eig_residual(a, r);
                const double cond = cond2(r.vectors);
                worst = (std::max)(worst, cond);
                worst_res = (std::max)(worst_res, res);
                CHECK_LE(res, 1e-13);
                CHECK_LE(cond, 1e4);
                ++cases;
            }
        }
    }
    say("  均匀层的层矩阵（正入射，n = 5 … 81，共 %d 个）：残差最大 %.2e，本征向量矩阵的条件数最大 %.2e\n", cases, worst_res, worst);
}

static void test_nonnormal() {
    say("[非正规矩阵]\n");
    {
        // A = Q (D + 50 N) Q^H，N 严格上三角：偏离正规性很大，本征值已知。
        const int n = 40;
        Rng rng(601);
        Matrix t(n, n);
        Vector d(n);
        for (int j = 0; j < n; ++j) {
            d[j] = cplx(j - 0.5 * n, 0.5 * ((j * 7) % 11) - 2.5);
            t(j, j) = d[j];
            for (int i = 0; i < j; ++i) t(i, j) = 50.0 * rng.cuniform();
        }
        const Matrix q = random_unitary(n, 602);
        const Matrix a = q * t * adjoint(q);
        EigOptions keep;
        keep.keep_transform = true;
        const EigResult r = eig_qr(a, keep);
        const double res = check_decomposition(a, r);
        const double zt = schur_defect(a, r.Z, r.T);
        say("  Q (D + 50N) Q^H  n = %d  残差 %.2e  ||Z^H A Z − T||/||A|| %.2e\n", n, res, zt);
        CHECK_LE(zt, 1e-13);
        CHECK_LE(orthogonality_defect(r.Z), 1e-13);
    }
    {
        // Grcar 矩阵：次对角线 −1，对角线和上面三条对角线为 1。实矩阵，本征值对扰动极其敏感。
        const int n = 60;
        Matrix a(n, n);
        for (int i = 0; i < n; ++i) {
            if (i > 0) a(i, i - 1) = -1.0;
            for (int j = i; j < std::min(n, i + 4); ++j) a(i, j) = 1.0;
        }
        const EigResult r = eig_qr(a);
        const double res = check_decomposition(a, r);
        say("  Grcar  n = %d              残差 %.2e  迭代 %d 次\n", n, res, r.sweeps);
    }
}

static void test_rcwa_like() {
    say("[RCWA 层矩阵 diag(kx²) − Toeplitz(ε_h)]\n");
    const double fill = 0.4, lambda_over_period = 0.61, sin_theta = 0.2;
    for (int n : {41, 101, 201}) {
        {
            // 无吸收：ε 为实数，矩阵厄米，本征值应为实数、本征向量应正交
            const Matrix a = rcwa_te_matrix(n, 1.5, 1.0, fill, lambda_over_period, sin_theta);
            CHECK(is_hermitian(a));
            const EigResult r = eig_qr(a);
            const double res = check_decomposition(a, r);
            double imag = 0.0;
            for (const cplx& z : r.values) imag = std::max(imag, std::abs(z.imag()));
            const double orth = orthogonality_defect(r.vectors);
            say("  n = %3d  无吸收 1.5 / 1.0          残差 %.2e  max|Im λ|/max|λ| %.2e  ||V^H V − I||_F %.2e  迭代 %d 次\n", n, res,
                imag / max_abs(r.values), orth, r.sweeps);
            CHECK_LE(imag / max_abs(r.values), 1e-12);
            CHECK_LE(orth, 1e-11);
        }
        for (const cplx index : {cplx(1.8, 0.13), cplx(0.08, 5.43)}) {
            // 有吸收：复折射率使矩阵成为一般复矩阵（复对称但不厄米）
            const Matrix a = rcwa_te_matrix(n, index, 1.0, fill, lambda_over_period, sin_theta);
            CHECK(!is_hermitian(a, 1e-8));
            const EigResult r = eig_qr(a);
            const double res = check_decomposition(a, r);
            // Im ε >= 0（被动介质）时 ⟦ε⟧ 的数值域在上半平面，所以 A = Kx² − ⟦ε⟧ 的本征值虚部不会为正
            double max_imag = -1e300;
            for (const cplx& z : r.values) max_imag = std::max(max_imag, z.imag());
            say("  n = %3d  吸收 %.2f+%.2fi / 1.0     残差 %.2e  迹 %.2e  max Im λ %+.2e  迭代 %d 次\n", n, index.real(), index.imag(),
                res, trace_defect(a, r.values), max_imag, r.sweeps);
            CHECK_LE(max_imag, 1e-10 * norm_fro(a));
            // 只求本征值得到同样的结果
            EigOptions values_only;
            values_only.vectors = false;
            const EigResult r0 = eig_qr(a, values_only);
            CHECK(r0.info == 0 && r0.vectors.empty());
            CHECK_LE(match_distance(r0.values, r.values) / max_abs(r.values), 1e-13);
        }
    }
    {
        // 更大的截断：n = 321（M = 160）的吸收光栅层
        const int n = 321;
        const cplx index(1.8, 0.13);
        const Matrix a = rcwa_te_matrix(n, index, 1.0, fill, lambda_over_period, sin_theta);
        EigOptions keep;
        keep.keep_transform = true;
        const EigResult r = eig_qr(a, keep);
        const double res = check_decomposition(a, r);
        double max_imag = -1e300;
        for (const cplx& z : r.values) max_imag = std::max(max_imag, z.imag());
        const double zt = schur_defect(a, r.Z, r.T);
        say("  n = %3d  吸收 %.2f+%.2fi / 1.0     残差 %.2e  ||Z^H A Z − T||/||A|| %.2e  max Im λ %+.2e  迭代 %d 次\n", n, index.real(), index.imag(), res,
            zt, max_imag, r.sweeps);
        CHECK_LE(max_imag, 1e-10 * norm_fro(a));
        CHECK_LE(zt, 1e-13);
        CHECK_LE(orthogonality_defect(r.Z), 1e-12);
    }
    {
        // 接近 Littrow 条件（2 sinθ ≈ λ/Λ）：级次 m 与 −1−m 的 kx² 几乎相等，本征值成对靠近。
        // 残差不受影响；但一般矩阵的算法不知道矩阵厄米，靠得近的本征向量之间的正交性只能到
        // eps·||A|| / 间隔 的量级，所以这里的正交性界放宽到 1e-10。
        const int n = 201;
        const Matrix a = rcwa_te_matrix(n, 1.5, 1.0, fill, lambda_over_period, 0.3);
        const EigResult r = eig_qr(a);
        const double res = check_decomposition(a, r);
        double imag = 0.0;
        for (const cplx& z : r.values) imag = std::max(imag, std::abs(z.imag()));
        const double orth = orthogonality_defect(r.vectors);
        say("  n = %3d  无吸收，近 Littrow        残差 %.2e  max|Im λ|/max|λ| %.2e  ||V^H V − I||_F %.2e\n", n, res,
            imag / max_abs(r.values), orth);
        CHECK_LE(imag / max_abs(r.values), 1e-12);
        CHECK_LE(orth, 1e-10);
    }
}

static void test_schur_identities() {
    say("[Z、T 的恒等式]\n");
    for (int n : {6, 30, 80}) {
        const Matrix a = random_complex(n, 700 + n);
        EigOptions keep;
        keep.keep_transform = true;
        const EigResult r = eig_qr(a, keep);
        check_decomposition(a, r);
        CHECK(r.Z.rows() == n && r.Z.cols() == n && r.T.rows() == n && r.T.cols() == n);
        const double unit = orthogonality_defect(r.Z);
        const double zt = schur_defect(a, r.Z, r.T);
        say("  n = %2d  ||Z^H Z − I||_F %.2e  ||Z^H A Z − T||_F/||A||_F %.2e\n", n, unit, zt);
        CHECK_LE(unit, 1e-13);
        CHECK_LE(zt, 1e-13);
        CHECK(strictly_lower_is_zero(r.T));          // T 是严格的上三角
        CHECK(same_bits(r.T.diag(), r.values));      // 对角线就是本征值

        // 不要本征向量、只要 Schur 形
        keep.vectors = false;
        const EigResult s = eig_qr(a, keep);
        CHECK(s.info == 0 && s.vectors.empty());
        CHECK(same_bits(s.values, r.values) && same_bits(s.Z, r.Z) && same_bits(s.T, r.T));
    }
}

static void test_observer() {
    say("[观察者回调]\n");
    const int n = 12;
    const Matrix a = random_complex(n, 801);
    const double na = norm_fro(a);

    struct Log {
        int calls = 0, hessenberg = 0, iteration = 0, done = 0;
        bool order_ok = true, sweep_ok = true, active_ok = true, shape_ok = true;
        double worst_similarity = 0.0, worst_unitary = 0.0;
        int last_active = 0;
        Matrix final_h;
        cplx final_shift;
    } log;
    log.last_active = n - 1;

    EigOptions opt;
    opt.keep_transform = true;
    opt.observer = [&](const EigProgress& p) {
        const std::string stage = p.stage;
        if (stage == "hessenberg") {
            ++log.hessenberg;
            log.order_ok = log.order_ok && log.calls == 0;
            log.sweep_ok = log.sweep_ok && p.sweep == 0 && p.shift == cplx{} && p.active == n - 1;
            // 干净的 Hessenberg 形；复矩阵的次对角元被换成了它的模，所以是非负实数
            log.shape_ok = log.shape_ok && strictly_lower_is_zero(p.H, 2);
            for (int i = 1; i < n; ++i) log.shape_ok = log.shape_ok && p.H(i, i - 1).imag() == 0.0 && p.H(i, i - 1).real() >= 0.0;
        } else if (stage == "iteration") {
            ++log.iteration;
            log.order_ok = log.order_ok && log.hessenberg == 1 && log.done == 0;
            log.sweep_ok = log.sweep_ok && p.sweep == log.iteration;
            log.active_ok = log.active_ok && p.active >= 0 && p.active <= log.last_active;
            log.shape_ok = log.shape_ok && strictly_lower_is_zero(p.H, 2);
            log.last_active = p.active;
        } else if (stage == "done") {
            ++log.done;
            log.order_ok = log.order_ok && log.hessenberg == 1;
            log.sweep_ok = log.sweep_ok && p.sweep == log.iteration;
            log.active_ok = log.active_ok && p.active == -1;
            log.shape_ok = log.shape_ok && strictly_lower_is_zero(p.H);
            log.final_h = p.H;
            log.final_shift = p.shift;
        } else {
            log.order_ok = false;
        }
        ++log.calls;
        // 迭代全程次对角元都是实数（旋转只用到它的实部）
        for (int i = 1; i < n; ++i) log.shape_ok = log.shape_ok && p.H(i, i - 1).imag() == 0.0;
        // 每次回调：Z 是酉矩阵，Z^H A Z = H + shift·I
        Matrix shifted = p.H;
        for (int i = 0; i < n; ++i) shifted(i, i) += p.shift;
        log.worst_similarity = std::max(log.worst_similarity, norm_fro(adjoint(p.Z) * a * p.Z - shifted) / na);
        log.worst_unitary = std::max(log.worst_unitary, orthogonality_defect(p.Z));
    };
    const EigResult r = eig_qr(a, opt);
    check_decomposition(a, r);
    say("  n = %d  回调 %d 次（迭代 %d 次）  max ||Z^H A Z − (H + tI)||/||A|| %.2e  max ||Z^H Z − I|| %.2e\n", n, log.calls, r.sweeps,
        log.worst_similarity, log.worst_unitary);
    CHECK(log.hessenberg == 1 && log.done == 1);
    CHECK(log.iteration == r.sweeps);
    CHECK(log.calls == r.sweeps + 2);
    CHECK(log.order_ok);
    CHECK(log.sweep_ok);
    CHECK(log.active_ok);
    CHECK(log.shape_ok);
    CHECK_LE(log.worst_similarity, 1e-13);
    CHECK_LE(log.worst_unitary, 1e-13);
    // "done" 时看到的 H + shift·I 就是返回的 T；回调结束后对角元恢复成本征值本身
    Matrix final_t = log.final_h;
    for (int i = 0; i < n; ++i) final_t(i, i) += log.final_shift;
    CHECK_LE(norm_fro(final_t - r.T) / na, 1e-14);
    CHECK(same_bits(r.T.diag(), r.values));

    // 有没有观察者，结果逐位相同
    EigOptions plain;
    plain.keep_transform = true;
    const EigResult r_plain = eig_qr(a, plain);
    CHECK(same_bits(r_plain.values, r.values) && same_bits(r_plain.vectors, r.vectors));
    CHECK(same_bits(r_plain.Z, r.Z) && same_bits(r_plain.T, r.T));
    CHECK(r_plain.sweeps == r.sweeps);

    // 只求本征值时也可以挂观察者（此时会为它形成 Z）
    int calls = 0;
    bool z_ok = true;
    EigOptions values_only;
    values_only.vectors = false;
    values_only.observer = [&](const EigProgress& p) {
        ++calls;
        z_ok = z_ok && p.Z.rows() == n && p.H.rows() == n;
    };
    const EigResult r0 = eig_qr(a, values_only);
    CHECK(r0.info == 0 && r0.vectors.empty() && r0.Z.empty() && r0.T.empty());
    CHECK(calls == r0.sweeps + 2 && z_ok);
    CHECK(same_bits(r0.values, r.values));

    // 未收敛时没有 "done" 回调
    int done_calls = 0, all_calls = 0;
    EigOptions limited;
    limited.max_iter = 1;
    limited.observer = [&](const EigProgress& p) {
        ++all_calls;
        if (std::string(p.stage) == "done") ++done_calls;
    };
    const EigResult f = eig_qr(a, limited);
    CHECK(f.info > 0 && done_calls == 0 && all_calls == f.sweeps + 1);

    // 实矩阵：次对角元本来就是实数，保持原样（可以为负）；相似关系照样成立
    const Matrix b = random_real(9, 802);
    const double nb = norm_fro(b);
    bool real_sub = true;
    int negative = 0;
    double worst = 0.0;
    EigOptions real_opt;
    real_opt.observer = [&](const EigProgress& p) {
        for (int i = 1; i < 9; ++i) {
            real_sub = real_sub && p.H(i, i - 1).imag() == 0.0;
            if (std::string(p.stage) == "hessenberg" && p.H(i, i - 1).real() < 0.0) ++negative;
        }
        Matrix shifted = p.H;
        for (int i = 0; i < 9; ++i) shifted(i, i) += p.shift;
        worst = std::max(worst, norm_fro(adjoint(p.Z) * b * p.Z - shifted) / nb);
    };
    const EigResult rb = eig_qr(b, real_opt);
    check_decomposition(b, rb);
    say("  实矩阵 n = 9：Hessenberg 形里负的次对角元 %d 个，max ||Z^H A Z − (H + tI)||/||A|| %.2e\n", negative, worst);
    CHECK(real_sub);
    CHECK_LE(worst, 1e-13);
}

static void test_values_only() {
    say("[vectors = false：只求本征值]\n");
    for (int n : {2, 5, 33, 60, 150}) {
        const Matrix a = random_complex(n, 900 + n);
        const EigResult full = eig_qr(a);
        EigOptions opt;
        opt.vectors = false;
        const EigResult vals = eig_qr(a, opt);
        CHECK(vals.info == 0 && full.info == 0);
        CHECK(vals.vectors.empty() && vals.Z.empty() && vals.T.empty());
        CHECK(vals.sweeps == full.sweeps);
        const double diff = match_distance(vals.values, full.values) / max_abs(full.values);
        CHECK_LE(diff, 1e-13);
        CHECK_LE(trace_defect(a, vals.values), 1e-13);
        if (n == 150) say("  n = %d  与求向量时的本征值最大相对差 %.2e\n", n, diff);
    }
}

static void test_not_converged() {
    say("[未收敛时的报告]\n");
    const int n = 20;
    const Matrix a = random_complex(n, 1001);
    const EigResult reference = eig_qr(a);
    CHECK(reference.info == 0);
    int failures = 0, partial = 0;
    for (int max_iter = 1; max_iter <= 6; ++max_iter) {
        EigOptions opt;
        opt.max_iter = max_iter;
        opt.keep_transform = true;
        const EigResult r = eig_qr(a, opt);
        CHECK(r.info >= 0 && r.info <= n);
        CHECK(static_cast<int>(r.values.size()) == n && all_finite(r.values));
        if (r.info == 0) {
            CHECK(same_bits(r.values, reference.values));
            continue;
        }
        ++failures;
        if (r.info < n) ++partial;
        CHECK(r.vectors.empty());
        // 下标 >= info 的本征值已收敛，而且与不设上限时走的是同一条路，逐位相同
        bool tail_ok = true;
        for (int i = r.info; i < n; ++i) tail_ok = tail_ok && r.values[i] == reference.values[i];
        CHECK(tail_ok);
        CHECK_LE(orthogonality_defect(r.Z), 1e-13);
        CHECK_LE(schur_defect(a, r.Z, r.T), 1e-13);
        CHECK(same_bits(r.T.diag(), r.values));
        CHECK(strictly_lower_is_zero(r.T, 2) && lower_rows_triangular(r.T, r.info));
    }
    say("  max_iter = 1…6：%d 次报告未收敛，其中 %d 次已求出一部分本征值\n", failures, partial);
    CHECK(failures > 0);                             // max_iter = 1 时一定不够

    // 部分收敛的例子：A = P ⊕ B。迭代从右下角开始，B（4×4）的本征值先求出来；
    // 左上角的 P 是 8 阶循环置换，不用例外位移就原地不动，max_iter = 10 时停在第 8 个本征值上。
    const int m = 8, k = 4;
    const Matrix b = random_complex(k, 1002);
    Matrix c(m + k, m + k);
    for (int i = 1; i < m; ++i) c(i, i - 1) = 1.0;
    c(0, m - 1) = 1.0;
    for (int j = 0; j < k; ++j)
        for (int i = 0; i < k; ++i) c(m + i, m + j) = b(i, j);
    EigOptions limited;
    limited.max_iter = 10;
    limited.keep_transform = true;
    const EigResult r = eig_qr(c, limited);
    CHECK(r.info == m);                              // 下标 >= 8 的本征值有效
    CHECK(r.vectors.empty());
    const EigResult rb = eig_qr(b);
    CHECK(rb.info == 0);
    const Vector tail(r.values.begin() + m, r.values.end());
    const double err = match_distance(tail, rb.values) / max_abs(rb.values);
    say("  P_8 ⊕ B_4，max_iter = 10：info = %d，已收敛的 4 个本征值误差 %.2e\n", r.info, err);
    CHECK_LE(err, 1e-13);
    CHECK_LE(schur_defect(c, r.Z, r.T), 1e-13);
    CHECK(lower_rows_triangular(r.T, r.info));       // 已收敛的 4 行是严格的上三角，可忽略的次对角元已清零
    CHECK(same_bits(r.T.diag(), r.values));
    const EigResult ok = eig_qr(c);                  // 默认上限下能全部收敛
    CHECK(ok.info == 0);
    CHECK_LE(eig_residual(c, ok), 1e-13);
}

static void test_scaling_and_eps() {
    say("[极端量级与 eps 选项]\n");
    const int n = 12;
    const Matrix a = random_complex(n, 1101);
    const EigResult base = eig_qr(a);
    check_decomposition(a, base);
    for (double factor : {1e145, 1e-145}) {          // 平方会上溢/下溢的量级
        const Matrix b = a * cplx(factor, 0.0);
        const EigResult r = eig_qr(b);
        const double res = check_decomposition(b, r);
        Vector scaled(base.values);
        for (cplx& z : scaled) z *= factor;
        const double diff = match_distance(r.values, scaled) / max_abs(scaled);
        say("  A × %.0e  残差 %.2e  本征值相对差 %.2e\n", factor, res, diff);
        CHECK_LE(diff, 1e-12);
    }
    {
        EigOptions loose;
        loose.eps = 1e-8;                            // 放宽收敛阈值：迭代更少，精度相应降低
        const EigResult r = eig_qr(a, loose);
        CHECK(r.info == 0 && r.sweeps <= base.sweeps);
        CHECK_LE(eig_residual(a, r), 1e-7);
        CHECK_LE(match_distance(r.values, base.values) / max_abs(base.values), 1e-6);
    }
    {
        EigOptions strict;
        strict.eps = 0.0;                            // 只认精确的零：要么收敛、要么报告未收敛，不能死循环或出 NaN
        const EigResult r = eig_qr(a, strict);
        CHECK(r.info >= 0 && r.info <= n);
        CHECK(all_finite(r.values));
        if (r.info == 0) CHECK_LE(eig_residual(a, r), 1e-13);
    }
    for (int m : {25, 40}) {
        // eps = 0 时要一直迭代到次对角元下溢成零，途中被消去的元素是次正规数。
        // 由次正规数定出的旋转和相位因子也必须是酉的，否则 Z^H A Z = T 会坏到 1e-3 而不报错。
        const Matrix b = random_complex(m, 5975 + m);
        EigOptions strict;
        strict.eps = 0.0;
        strict.keep_transform = true;
        const EigResult r = eig_qr(b, strict);
        CHECK(r.info >= 0 && r.info <= m && all_finite(r.values));
        const double unit = orthogonality_defect(r.Z), zt = schur_defect(b, r.Z, r.T);
        say("  eps = 0  n = %d  info = %d  迭代 %d 次  ||Z^H Z − I||_F %.2e  ||Z^H A Z − T||/||A|| %.2e\n", m, r.info, r.sweeps, unit, zt);
        CHECK_LE(unit, 1e-13);
        CHECK_LE(zt, 1e-13);
        if (r.info == 0) CHECK_LE(eig_residual(b, r), 1e-13);
    }
}

static void test_extreme_magnitude() {
    say("[极端量级：整体缩放到 1e±300]\n");
    for (int n : {12, 40}) {
        const Matrix a = random_complex(n, 1400 + n);
        const double na = norm_fro(a);
        EigOptions keep;
        keep.keep_transform = true;
        const EigResult base = eig_qr(a, keep);
        check_decomposition(a, base);

        // (1) 乘 2 的幂：输入只是指数变了，而算法内部的缩放也都是 2 的幂，所以迭代过程应当一模一样。
        double worst_value = 0.0, worst_vector = 0.0;
        for (int k : {-900, -600, 600, 900}) {
            const EigResult r = eig_qr(times_pow2(a, k), keep);
            CHECK(r.info == 0 && r.sweeps == base.sweeps);
            if (r.info != 0) continue;
            Vector back(r.values);
            for (cplx& z : back) z = times_pow2(z, -k);
            worst_value = std::max(worst_value, match_distance(back, base.values) / max_abs(base.values));
            worst_vector = std::max(worst_vector, std::max(max_abs(r.vectors - base.vectors), max_abs(r.Z - base.Z)));
        }
        say("  n = %2d  A × 2^±600、2^±900：本征值相对差 %.2e，向量与 Z 的最大差 %.2e\n", n, worst_value, worst_vector);
        CHECK_LE(worst_value, 1e-14);
        CHECK_LE(worst_vector, 1e-12);

        // (2) 乘 10 的幂。度量一律换算回原来的量级再算，免得度量自己上溢或下溢。
        for (double factor : {1e150, 1e-150, 1e154, 1e-160, 1e250, 1e-250, 1e300, 1e-300}) {
            const Matrix b = a * cplx(factor, 0.0);
            const EigResult r = eig_qr(b, keep);
            CHECK(r.info == 0);
            CHECK(all_finite(r.values) && r.vectors.all_finite() && r.Z.all_finite() && r.T.all_finite());
            if (r.info != 0) continue;
            EigResult back;
            back.values = r.values;
            for (cplx& z : back.values) z /= factor;
            back.vectors = r.vectors;
            const double res = eig_residual(a, back);
            const double unit = orthogonality_defect(r.Z);
            const double zt = norm_fro(adjoint(r.Z) * a * r.Z - r.T * cplx(1.0 / factor, 0.0)) / na;
            const double diff = match_distance(back.values, base.values) / max_abs(base.values);
            if (n == 40) say("  n = %2d  A × %-6.0e  残差 %.2e  ||Z^H Z − I||_F %.2e  ||Z^H A Z − T||/||A|| %.2e  本征值相对差 %.2e\n", n, factor, res, unit, zt, diff);
            CHECK_LE(res, 1e-13);
            CHECK_LE(unit, 1e-13);
            CHECK_LE(zt, 1e-13);
            CHECK_LE(diff, 1e-12);
            EigOptions values_only;
            values_only.vectors = false;
            const EigResult r0 = eig_qr(b, values_only);
            CHECK(r0.info == 0);
            CHECK_LE(match_distance(r0.values, r.values) / max_abs(r.values), 1e-13);
        }
    }

    say("[极端量级：同一个矩阵里量级悬殊的块]\n");
    {
        // A = [s1·A1  C; 0  s2·A2]：两个对角块精确解耦。右下角的块先算完，它的本征值和本征向量应当准到它自己的量级；
        // 左上角的块要等右下角算完才轮到，期间累计位移一直从整条对角线上减（EISPACK 的做法），
        // 所以它比右下角小得多时，本征值只能准到 eps·||A|| 的绝对精度。
        const int n1 = 7, n2 = 6, n = n1 + n2;
        const Matrix a1 = random_complex(n1, 1501), a2 = random_complex(n2, 1502), c = random_complex(n, 1503);
        const EigResult r1 = eig_qr(a1), r2 = eig_qr(a2);
        CHECK(r1.info == 0 && r2.info == 0);
        const double scales[3][2] = {{1.0, 1e-200}, {1e200, 1.0}, {1e-150, 1e150}};
        for (const auto& s : scales) {
            const double s1 = s[0], s2 = s[1], sc = std::max(s1, s2);
            Matrix a(n, n);
            for (int j = 0; j < n1; ++j)
                for (int i = 0; i < n1; ++i) a(i, j) = a1(i, j) * s1;
            for (int j = 0; j < n2; ++j) {
                for (int i = 0; i < n2; ++i) a(n1 + i, n1 + j) = a2(i, j) * s2;
                for (int i = 0; i < n1; ++i) a(i, n1 + j) = c(i, j) * sc;
            }
            EigOptions keep;
            keep.keep_transform = true;
            const EigResult r = eig_qr(a, keep);
            CHECK(r.info == 0);
            CHECK(all_finite(r.values) && r.vectors.all_finite() && r.Z.all_finite());
            if (r.info != 0 || !r.vectors.all_finite()) continue;
            CHECK_LE(orthogonality_defect(r.Z), 1e-13);
            CHECK_LE(unit_norm_defect(r.vectors), 1e-14);
            CHECK_LE(eig_residual(a, r), 1e-13);
            // 本征值：前 n1 个属于 A1、后 n2 个属于 A2，各按自己的量级比较
            Vector head(r.values.begin(), r.values.begin() + n1), tail(r.values.begin() + n1, r.values.end());
            for (cplx& z : head) z /= s1;
            for (cplx& z : tail) z /= s2;
            const double e1 = match_distance(head, r1.values) / max_abs(r1.values), e2 = match_distance(tail, r2.values) / max_abs(r2.values);
            // 右下块 A2 的本征向量：v 的后 n2 个分量应当是 A2 的本征向量（按 A2 自己的量级衡量残差）
            double local = 0.0;
            for (int j = n1; j < n; ++j) {
                Vector v2(n2);
                for (int i = 0; i < n2; ++i) v2[i] = r.vectors(n1 + i, j);
                const Vector av = a2 * v2;
                Vector d(n2);
                for (int i = 0; i < n2; ++i) d[i] = av[i] - (r.values[j] / s2) * v2[i];
                const double res = norm2(d) / (norm_fro(a2) * norm2(v2));
                if (!(res <= local)) local = res;
            }
            say("  块的量级 %.0e / %.0e：本征值误差（各按自己的量级）%.2e / %.2e，右下块本征向量的局部残差 %.2e%s\n", s1, s2, e1, e2, local,
                s1 >= s2 ? "" : "  （左上的小块只保证绝对精度）");
            if (s1 >= s2) CHECK_LE(e1, 1e-12);
            else CHECK_LE(e1 * s1 * max_abs(r1.values), 1e-13 * norm_fro(a));
            CHECK_LE(e2, 1e-12);
            CHECK_LE(local, 1e-11);
        }
    }

    say("[有限但会在内部上溢的输入：必须结束并如实报告]\n");
    for (int n : {3, 10}) {
        const Matrix a(n, n, cplx(std::numeric_limits<double>::max(), 0.0));
        const EigResult r = eig_qr(a);
        CHECK(r.info >= 0 && r.info <= n && static_cast<int>(r.values.size()) == n);
        CHECK(r.sweeps <= 30 * n);                   // 迭代次数有上限，不会挂死
        CHECK(r.info > 0 || (all_finite(r.values) && r.vectors.all_finite()));
        say("  n = %2d，元素全为 DBL_MAX：info = %d，迭代 %d 次\n", n, r.info, r.sweeps);
    }
}

// 元素取自 {0, ±1, ±i, 2, 1+i, 1/2} 的小矩阵：到处是精确的零、重本征值、亏损和幂零的情形，最容易碰到边界分支。
static void test_small_integer_matrices() {
    say("[小整数矩阵批量检查]\n");
    static const cplx alphabet[] = {0.0, 0.0, 0.0, 1.0, -1.0, cplx(0.0, 1.0), cplx(0.0, -1.0), 2.0, cplx(1.0, 1.0), 0.5};
    const int count = 400;
    int bad = 0, slow = 0;
    double worst_res = 0.0, worst_zt = 0.0, worst_unit = 0.0;
    for (int c = 0; c < count; ++c) {
        Rng rng(77000 + c);
        const int n = 1 + static_cast<int>(rng.next() % 9);
        const int flavour = static_cast<int>(rng.next() % 5);
        const int letters = 3 + static_cast<int>(rng.next() % 8);        // 用字母表的前几个：越少，零越多
        Matrix a(n, n);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                cplx z = alphabet[rng.next() % letters];
                if (flavour == 1 && i > j + 1) z = 0.0;                  // 已经是 Hessenberg 形
                if (flavour == 2) z = z.real();                          // 实矩阵
                a(i, j) = z;
            }
        if (flavour == 3)                                                // 复对称（有吸收时 RCWA 层矩阵的类型）
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < j; ++i) a(j, i) = a(i, j);
        if (flavour == 4)                                                // 厄米
            for (int j = 0; j < n; ++j) {
                a(j, j) = a(j, j).real();
                for (int i = 0; i < j; ++i) a(j, i) = std::conj(a(i, j));
            }
        EigOptions keep;
        keep.keep_transform = true;
        EigResult r = eig_qr(a, keep);
        if (r.info != 0) {                           // 高重数的亏损本征值偶尔超过每个本征值 30 次的上限
            ++slow;
            keep.max_iter = 100;
            r = eig_qr(a, keep);
        }
        bool ok = r.info == 0 && all_finite(r.values) && r.vectors.all_finite() && r.Z.all_finite() && r.T.all_finite();
        ok = ok && strictly_lower_is_zero(r.T) && same_bits(r.T.diag(), r.values);
        if (ok) {
            const double unit = orthogonality_defect(r.Z);
            const double res = norm_fro(a) > 0.0 ? eig_residual(a, r) : 0.0;
            const double zt = norm_fro(a) > 0.0 ? schur_defect(a, r.Z, r.T) : norm_fro(r.T);
            worst_unit = std::max(worst_unit, unit);
            worst_res = std::max(worst_res, res);
            worst_zt = std::max(worst_zt, zt);
            ok = unit <= 1e-13 && res <= 1e-13 && zt <= 1e-13 && unit_norm_defect(r.vectors) <= 1e-14;
        }
        // 把上限压到 1～3 次：未收敛时 Z、T 的关系仍然成立，已收敛的行是上三角
        EigOptions limited;
        limited.keep_transform = true;
        limited.max_iter = 1 + c % 3;
        const EigResult f = eig_qr(a, limited);
        ok = ok && f.info >= 0 && f.info <= n && all_finite(f.values) && same_bits(f.T.diag(), f.values);
        ok = ok && strictly_lower_is_zero(f.T, 2) && lower_rows_triangular(f.T, f.info);
        if (f.info > 0) ok = ok && f.vectors.empty() && orthogonality_defect(f.Z) <= 1e-13 && (norm_fro(a) == 0.0 || schur_defect(a, f.Z, f.T) <= 1e-13);
        if (!ok) {
            ++bad;
            say("  失败：第 %d 个矩阵（n = %d，类型 %d）\n", c, n, flavour);
        }
    }
    say("  %d 个矩阵（n <= 9）：最大残差 %.2e，最大 ||Z^H A Z − T||/||A|| %.2e，最大 ||Z^H Z − I||_F %.2e；%d 个要放宽迭代上限\n", count, worst_res, worst_zt,
        worst_unit, slow);
    CHECK(bad == 0);
    CHECK(slow <= count / 50);
}

static void test_invalid_arguments() {
    say("[非法参数]\n");
    const auto throws = [](const Matrix& a, const EigOptions& opt) {
        try {
            eig_qr(a, opt);
        } catch (const std::invalid_argument&) {
            return true;
        } catch (...) {
            return false;
        }
        return false;
    };
    const Matrix ok = random_complex(4, 1201);
    CHECK(!throws(ok, EigOptions{}));
    CHECK(throws(Matrix(3, 4), EigOptions{}));       // 非方阵
    Matrix bad = ok;
    bad(2, 1) = cplx(std::numeric_limits<double>::quiet_NaN(), 0.0);
    CHECK(throws(bad, EigOptions{}));
    bad(2, 1) = cplx(0.0, std::numeric_limits<double>::infinity());
    CHECK(throws(bad, EigOptions{}));
    EigOptions opt;
    opt.eps = -1e-16;
    CHECK(throws(ok, opt));
    opt.eps = std::numeric_limits<double>::quiet_NaN();
    CHECK(throws(ok, opt));
    opt = EigOptions{};
    opt.max_iter = 0;
    CHECK(throws(ok, opt));
}

static void test_reentrant() {
    say("[可重入：多个线程同时调用]\n");
    const Matrix a = random_complex(40, 1301), b = random_complex(50, 1302);
    const EigResult ra = eig_qr(a), rb = eig_qr(b);
    EigResult ta, tb;
    std::thread first([&] { ta = eig_qr(a); });
    std::thread second([&] { tb = eig_qr(b); });
    first.join();
    second.join();
    CHECK(same_bits(ta.values, ra.values) && same_bits(ta.vectors, ra.vectors));
    CHECK(same_bits(tb.values, rb.values) && same_bits(tb.vectors, rb.vectors));

    // 8 个线程各自反复分解不同大小的矩阵（一半带观察者、一半只求本征值），结果必须与串行时逐位相同
    const int threads = 8, per_thread = 4;
    std::vector<Matrix> mats;
    std::vector<EigOptions> opts(threads * per_thread);
    for (int k = 0; k < threads * per_thread; ++k) {
        mats.push_back(random_complex(6 + (k * 11) % 45, 1310 + k));
        opts[k].keep_transform = k % 2 == 0;
        opts[k].vectors = k % 3 != 0;
    }
    std::vector<EigResult> serial(mats.size()), parallel(mats.size());
    for (std::size_t k = 0; k < mats.size(); ++k) serial[k] = eig_qr(mats[k], opts[k]);
    std::vector<int> callbacks(mats.size(), 0);
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([&, t] {
            for (int rep = 0; rep < 2; ++rep)
                for (int k = t; k < threads * per_thread; k += threads) {
                    EigOptions o = opts[k];
                    int* counter = &callbacks[k];
                    if (k % 4 == 1) o.observer = [counter](const EigProgress&) { ++*counter; };
                    parallel[k] = eig_qr(mats[k], o);
                }
        });
    for (std::thread& th : pool) th.join();
    int different = 0;
    for (std::size_t k = 0; k < mats.size(); ++k) {
        const EigResult &s = serial[k], &p = parallel[k];
        if (!(s.info == p.info && s.sweeps == p.sweeps && same_bits(s.values, p.values) && same_bits(s.vectors, p.vectors) && same_bits(s.Z, p.Z) &&
              same_bits(s.T, p.T)))
            ++different;
        if (k % 4 == 1 && callbacks[k] != 2 * (p.sweeps + 2)) ++different;       // 两轮，每轮 sweeps + 2 次回调
    }
    say("  %d 个线程 × %d 个矩阵 × 2 轮：与串行结果不同的 %d 个\n", threads, per_thread, different);
    CHECK(different == 0);
}

int main() {
    test_small();
    test_diagonal_and_triangular();
    test_companion();
    test_random_general();
    test_hermitian();
    test_real_nonsymmetric();
    test_repeated_and_defective();
    test_nearly_equal_eigenvalues();
    test_nonnormal();
    test_rcwa_like();
    test_schur_identities();
    test_observer();
    test_values_only();
    test_not_converged();
    test_scaling_and_eps();
    test_extreme_magnitude();
    test_small_integer_matrices();
    test_invalid_arguments();
    test_reentrant();
    say("通过 %d / 共 %d\n", g_pass, g_total);
    return g_pass == g_total ? 0 : 1;
}
