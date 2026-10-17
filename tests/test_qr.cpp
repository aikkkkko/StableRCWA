// tests/test_qr.cpp —— qr.hpp（Householder QR、解方程、最小二乘）的测试
//
//   c++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -I include tests/test_qr.cpp -o build/test_qr
//
// 只依赖 C++17 标准库；测试矩阵由固定种子的 Rng 生成。主要度量：
//   正交性     ||Q^H Q − I||_F / sqrt(n)
//   重构误差   ||Q R − A||_F / ||A||_F
//   后向误差   ||A X − B||_F / (||A||_F ||X||_F + ||B||_F)（方阵）
//   法方程残差 ||A^H (A X − B)||_F / (||A||_F (||A||_F ||X||_F + ||B||_F))（最小二乘）
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include "stablercwa/lu.hpp"
#include "stablercwa/qr.hpp"

using namespace stablercwa;

// ---------------------------------------------------------------------------------------------
// 输出与计数
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

// 断言的计数与报错都在下面几个普通函数里；宏只负责取文件名、行号和表达式的文字，宏里不出现中文。
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

// ---------------------------------------------------------------------------------------------
struct Rng {   // splitmix64
    std::uint64_t s;
    explicit Rng(std::uint64_t seed) : s(seed) {}
    std::uint64_t next() {
        std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    double uniform() { return static_cast<double>(next() >> 11) * (2.0 / 9007199254740992.0) - 1.0; }
    cplx cuniform() {
        const double re = uniform();
        return cplx(re, uniform());
    }
};

static Matrix random_matrix(int m, int n, std::uint64_t seed) {
    Rng rng(seed);
    Matrix a(m, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) a(i, j) = rng.cuniform();
    return a;
}

static const double kTol = 50.0 * kEps;

static bool same_bits(const Matrix& a, const Matrix& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    const std::size_t bytes = sizeof(cplx) * static_cast<std::size_t>(a.rows()) * a.cols();
    return bytes == 0 || std::memcmp(a.data(), b.data(), bytes) == 0;
}
static double orthogonality(const Matrix& q) {
    if (q.cols() == 0) return 0.0;
    return norm_fro(adjoint(q) * q - Matrix::identity(q.cols())) / std::sqrt(static_cast<double>(q.cols()));
}
static double reconstruction(const Matrix& a, const QR& f) {
    return norm_fro(qr_q(f) * qr_r(f) - a) / norm_fro(a);
}
static double normal_residual(const Matrix& a, const Matrix& x, const Matrix& b) {
    return norm_fro(adjoint(a) * (a * x - b)) / (norm_fro(a) * (norm_fro(a) * norm_fro(x) + norm_fro(b)));
}
static bool is_upper_with_real_diagonal(const Matrix& r) {
    for (int j = 0; j < r.cols(); ++j) {
        if (r(j, j).imag() != 0.0) return false;
        for (int i = j + 1; i < r.rows(); ++i)
            if (r(i, j) != cplx{}) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------
static void test_basic() {
    say("[基本情形]\n");
    {   // 空矩阵、0 列
        const QR f = qr_factor(Matrix());
        CHECK(f.qr.empty() && f.tau.empty());
        const QR g = qr_factor(Matrix(3, 0));
        CHECK(g.tau.empty() && qr_q(g).rows() == 3 && qr_q(g).cols() == 0 && qr_r(g).rows() == 0);
        const Matrix x = qr_solve(g, Matrix(3, 2));               // 没有未知数：解是 0 行 2 列
        CHECK(x.rows() == 0 && x.cols() == 2);
        CHECK(qr_solve(f, Matrix(0, 3)).cols() == 3);
        CHECK(qr_solve(qr_factor(Matrix::identity(4)), Matrix(4, 0)).rows() == 4);   // 没有右端
    }
    {   // 单位阵：不需要任何反射
        const Matrix eye = Matrix::identity(4);
        const QR f = qr_factor(eye);
        CHECK(same_bits(f.qr, eye) && same_bits(qr_q(f), eye) && same_bits(qr_r(f), eye));
        for (const cplx& t : f.tau) CHECK(t == cplx{});
    }
    {   // 对角线为正实数的上三角阵：已经是 R，原样返回
        Matrix up = random_matrix(6, 6, 1);
        for (int j = 0; j < 6; ++j)
            for (int i = j; i < 6; ++i) up(i, j) = (i == j) ? cplx(2.0 + j, 0.0) : cplx{};
        const QR f = qr_factor(up);
        CHECK(same_bits(qr_r(f), up) && same_bits(qr_q(f), Matrix::identity(6)));
    }
    {   // 复对角阵：|R(k,k)| = |d_k|，R 的对角元是实数
        Rng rng(2);
        Vector d(5);
        for (auto& v : d) v = rng.cuniform();
        const Matrix a = Matrix::diagonal(d);
        const QR f = qr_factor(a);
        const Matrix r = qr_r(f);
        CHECK(is_upper_with_real_diagonal(r));
        for (int k = 0; k < 5; ++k) CHECK_LE(std::abs(std::abs(r(k, k)) - std::abs(d[k])), 4 * kEps);
        CHECK_LE(reconstruction(a, f), kTol);
    }
    {   // 1×1 与单列
        const Matrix a{{cplx(3.0, -4.0)}};
        const QR f = qr_factor(a);
        CHECK_LE(std::abs(qr_r(f)(0, 0) - cplx(-5.0, 0.0)), 8 * kEps);      // beta = −sign(Re alpha)·|alpha|
        CHECK_LE(std::abs(qr_solve(f, Matrix{{cplx(5.0, 0.0)}})(0, 0) - cplx(0.6, 0.8)), 8 * kEps);
        const Matrix col{{3.0}, {0.0}, {4.0}};
        const QR g = qr_factor(col);
        CHECK_LE(std::abs(qr_r(g)(0, 0) - cplx(-5.0, 0.0)), 8 * kEps);
        CHECK_LE(orthogonality(qr_q(g)), kTol);
    }
    for (const auto& shape : {std::pair<int, int>{6, 4}, std::pair<int, int>{40, 35}}) {
        // 存储约定：用 (v_k, tau_k) 显式拼出 H_0 H_1 … H_{n-1}，应与 qr_q 一致。40×35 跨过一个面板的边界
        const int m = shape.first, n = shape.second;
        const Matrix a = random_matrix(m, n, 3 + m);
        const QR f = qr_factor(a);
        Matrix q = Matrix::identity(m);
        for (int k = 0; k < n; ++k) {
            Matrix v(m, 1);
            v(k, 0) = 1.0;
            for (int i = k + 1; i < m; ++i) v(i, 0) = f.qr(i, k);
            const Matrix h = Matrix::identity(m) - f.tau[k] * (v * adjoint(v));
            CHECK_LE(orthogonality(h), kTol);                             // 每个反射都是酉矩阵
            CHECK_LE(std::abs(f.tau[k] - cplx(1.0, 0.0)), 1.0 + 8 * kEps);   // LAPACK 约定下 |tau − 1| <= 1
            q = q * h;
        }
        Matrix thin(m, n);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) thin(i, j) = q(i, j);
        CHECK_LE(norm_fro(thin - qr_q(f)), kTol);
        Matrix r_padded(m, n);                                            // Q^H A = (R; 0)
        const Matrix r = qr_r(f);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i <= j; ++i) r_padded(i, j) = r(i, j);
        CHECK_LE(norm_fro(adjoint(q) * a - r_padded) / norm_fro(a), kTol);
    }
}

static void test_random() {
    say("[随机复矩阵]\n");
    const int shapes[][2] = {{1, 1}, {2, 2}, {3, 3}, {10, 10}, {50, 50}, {200, 200},
                             {5, 3}, {40, 17}, {65, 33}, {300, 120}, {200, 1}};
    for (const auto& sh : shapes) {
        const int m = sh[0], n = sh[1];
        const Matrix a = random_matrix(m, n, 100 + 7 * m + n), b = random_matrix(m, 3, 200 + 7 * m + n);
        const QR f = qr_factor(a);
        const Matrix q = qr_q(f), r = qr_r(f);
        CHECK(q.rows() == m && q.cols() == n && r.rows() == n && r.cols() == n);
        CHECK(is_upper_with_real_diagonal(r));
        const double orth = orthogonality(q), rec = norm_fro(q * r - a) / norm_fro(a);
        CHECK_LE(orth, kTol);
        CHECK_LE(rec, kTol);
        const Matrix x = qr_solve(f, b);
        CHECK(x.rows() == n && x.cols() == 3);
        CHECK(same_bits(x, solve_qr(a, b)));
        double metric;
        if (m == n) {
            metric = solve_backward_error(a, x, b);
            CHECK_LE(metric, kTol);
            CHECK_LE(norm_fro(x - solve(a, b)) / norm_fro(x), 1e-9);       // 与 LU 的解一致
        } else {
            metric = normal_residual(a, x, b);
            CHECK_LE(metric, kTol);
            // 右端恰好在列空间里时，最小二乘解就是原来的系数
            const Matrix x0 = random_matrix(n, 2, 300 + m);
            CHECK_LE(norm_fro(qr_solve(f, a * x0) - x0) / norm_fro(x0), 1e-11);
        }
        say("  %3d×%-3d  正交性 %.2e  重构 %.2e  %s %.2e\n", m, n, orth, rec,
            m == n ? "后向误差  " : "法方程残差", metric);
    }
}

static void test_hard_cases() {
    say("[病态矩阵与小主元：QR 不需要选主元也稳定]\n");
    for (int n : {6, 10, 13}) {
        Matrix h(n, n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) h(i, j) = cplx(1.0, 0.0) / cplx(i + j + 1.0, 0.5 * (i - j));
        const Matrix b = random_matrix(n, 2, 400 + n);
        const QR f = qr_factor(h);
        const double berr = solve_backward_error(h, qr_solve(f, b), b);
        CHECK_LE(berr, kTol);
        CHECK_LE(orthogonality(qr_q(f)), kTol);
        CHECK_LE(reconstruction(h, f), kTol);
        say("  Cauchy 型 n=%2d  后向误差 %.2e\n", n, berr);
    }
    {   // LU 不选主元会失败的 2×2
        const Matrix a{{1e-20, 1.0}, {1.0, 1.0}};
        const Matrix b{{1.0}, {2.0}};
        const Matrix x = solve_qr(a, b);
        CHECK_LE(std::abs(x(0, 0) - 1.0), 8 * kEps);
        CHECK_LE(std::abs(x(1, 0) - 1.0), 8 * kEps);
    }
    for (int n : {10, 50}) {   // 左上角很小的随机矩阵
        Matrix a = random_matrix(n, n, 500 + n);
        a(0, 0) = cplx(1e-13, 0.0);
        const Matrix b = random_matrix(n, 2, 600 + n);
        CHECK_LE(solve_backward_error(a, solve_qr(a, b), b), kTol);
    }
    for (int e : {700, -700, -997}) {   // 量级极端的矩阵：范数要用带缩放的求和，极小时反射要先放大再算
        const Matrix a0 = random_matrix(12, 9, 700);
        Matrix a = a0;
        for (int j = 0; j < 9; ++j)
            for (int i = 0; i < 12; ++i) a(i, j) = cplx(std::ldexp(a0(i, j).real(), e), std::ldexp(a0(i, j).imag(), e));
        const QR f = qr_factor(a);
        const Matrix q = qr_q(f);
        Matrix r = qr_r(f);                              // 把 R 缩回 1 的量级再比较，避免范数本身上溢/下溢
        for (int j = 0; j < 9; ++j)
            for (int i = 0; i <= j; ++i) r(i, j) = cplx(std::ldexp(r(i, j).real(), -e), std::ldexp(r(i, j).imag(), -e));
        CHECK(q.all_finite() && r.all_finite());
        CHECK_LE(orthogonality(q), kTol);
        CHECK_LE(norm_fro(q * r - a0) / norm_fro(a0), kTol);
    }
}

static void test_phase_conventions() {
    say("[复输入的相位约定：R 的对角元总是实数]\n");
    {   // 1×1 纯虚数：beta = −|alpha|（Re alpha = +0 时取负号），tau = 1 + i，都是精确值
        const QR f = qr_factor(Matrix{{cplx(0.0, 2.0)}});
        CHECK(f.qr(0, 0) == cplx(-2.0, 0.0) && f.tau[0] == cplx(1.0, 1.0));
        CHECK_LE(std::abs(qr_solve(f, Matrix{{cplx(4.0, 0.0)}})(0, 0) - cplx(0.0, -2.0)), 4 * kEps);
        CHECK_LE(std::abs(qr_q(f)(0, 0) - cplx(0.0, -1.0)), 4 * kEps);   // Q·R = (−i)·(−2) = 2i
    }
    {   // 1×1 负实数：已经是实数，不做反射（H = I），R 保持为负
        const QR f = qr_factor(Matrix{{cplx(-3.0, 0.0)}});
        CHECK(f.qr(0, 0) == cplx(-3.0, 0.0) && f.tau[0] == cplx{});
    }
    {   // 纯虚的列 (3i, 4i)^T：R = ±5（实数），Q·R 还原出原来的列
        const Matrix a{{cplx(0.0, 3.0)}, {cplx(0.0, 4.0)}};
        const QR f = qr_factor(a);
        CHECK(f.qr(0, 0).imag() == 0.0);
        CHECK_LE(std::abs(std::abs(f.qr(0, 0).real()) - 5.0), 8 * kEps);
        CHECK_LE(norm_fro(qr_q(f) * qr_r(f) - a), 16 * kEps);
    }
    {   // 上三角、对角元为复数：每个反射只是给一行乘上一个模为 1 的相位，所以 |R| 与 |A| 逐元素相等
        Matrix a = random_matrix(7, 7, 31);
        for (int j = 0; j < 7; ++j)
            for (int i = j + 1; i < 7; ++i) a(i, j) = cplx{};
        const QR f = qr_factor(a);
        const Matrix r = qr_r(f);
        CHECK(is_upper_with_real_diagonal(r));
        double worst = 0.0;
        for (int j = 0; j < 7; ++j)
            for (int i = 0; i <= j; ++i) worst = std::max(worst, std::abs(std::abs(r(i, j)) - std::abs(a(i, j))));
        CHECK_LE(worst, 8 * kEps);
        CHECK_LE(reconstruction(a, f), kTol);
    }
    for (int m : {6, 9}) {   // 混合：纯虚的列、全零的列、对角位置为零的列、负实数列、实部为 −0 的列
        const int n = 6;
        Matrix a = random_matrix(m, n, 32 + m);
        for (int i = 0; i < m; ++i) {
            a(i, 1) = cplx(0.0, a(i, 1).imag());
            a(i, 2) = cplx{};
            a(i, 4) = cplx(-std::abs(a(i, 4).real()), 0.0);
            a(i, 5) = cplx(-0.0, a(i, 5).imag());
        }
        a(3, 3) = cplx{};
        const QR f = qr_factor(a);
        const Matrix q = qr_q(f), r = qr_r(f);
        CHECK(f.qr.all_finite() && is_upper_with_real_diagonal(r));
        CHECK(f.tau[2] == cplx{} && r(2, 2) == cplx{});            // 零列：不做反射，R 的这个对角元精确为零
        CHECK_LE(orthogonality(q), kTol);
        CHECK_LE(norm_fro(q * r - a) / norm_fro(a), kTol);
        CHECK_LE(norm_fro(adjoint(q) * a - r) / norm_fro(a), kTol);   // Q^H A = R：反射的共轭关系没有弄反
        CHECK_THROWS(qr_solve(f, Matrix(m, 1)), std::runtime_error);  // R 有零对角元，求解时报奇异
    }
}

static void test_panel_boundaries_and_least_squares() {
    say("[面板边界两侧的尺寸；最小二乘]\n");
    double worst_orth = 0.0, worst_rec = 0.0, worst_solve = 0.0;
    for (int n : {31, 32, 33, 63, 64, 65}) {
        for (int extra : {0, 3}) {
            const int m = n + extra;
            const Matrix a = random_matrix(m, n, 900 + 3 * n + extra);
            const QR f = qr_factor(a);
            const Matrix q = qr_q(f), r = qr_r(f);
            const double orth = orthogonality(q), rec = norm_fro(q * r - a) / norm_fro(a);
            CHECK(is_upper_with_real_diagonal(r));
            CHECK_LE(orth, kTol);
            CHECK_LE(rec, kTol);
            for (int nrhs : {1, n + 2}) {
                const Matrix b = random_matrix(m, nrhs, 950 + n + nrhs);
                const Matrix x = qr_solve(f, b);
                CHECK(x.rows() == n && x.cols() == nrhs);
                const double metric = extra == 0 ? solve_backward_error(a, x, b) : normal_residual(a, x, b);
                CHECK_LE(metric, kTol);
                worst_solve = std::max(worst_solve, metric);
                // 与显式的 X = R^{-1}·(Q^H B) 对照（Q、R 取自 qr_q / qr_r，回代逐列、不分块）
                Matrix y = adjoint(q) * b;
                for (int j = 0; j < nrhs; ++j)
                    for (int k = n - 1; k >= 0; --k) {
                        y(k, j) /= r(k, k);
                        for (int i = 0; i < k; ++i) y(i, j) -= r(i, k) * y(k, j);
                    }
                CHECK_LE(norm_fro(x - y) / norm_fro(y), 1e-10);
            }
            worst_orth = std::max(worst_orth, orth);
            worst_rec = std::max(worst_rec, rec);
        }
    }
    say("  n = 31…65：正交性 %.2e  重构 %.2e  后向误差/法方程残差 %.2e\n", worst_orth, worst_rec, worst_solve);
    {   // 手算的最小二乘：用直线 y = c0 + c1·t 拟合 (0,1)、(1,3)、(2,5)、(3,8)，解是 c = (0.8, 2.3)
        const Matrix a{{1.0, 0.0}, {1.0, 1.0}, {1.0, 2.0}, {1.0, 3.0}};
        const Matrix b{{1.0}, {3.0}, {5.0}, {8.0}};
        const Matrix x = solve_qr(a, b);
        CHECK(x.rows() == 2 && x.cols() == 1);
        CHECK_LE(std::abs(x(0, 0) - 0.8), 1e-14);
        CHECK_LE(std::abs(x(1, 0) - 2.3), 1e-14);
    }
    {   // 右端与系数矩阵是同一个对象
        const Matrix a = random_matrix(20, 20, 990);
        CHECK_LE(max_abs(solve_qr(a, a) - Matrix::identity(20)), 1e-12);
    }
    {   // 元素落在次正规数范围（2^-1040）：1/(alpha − beta) 会溢出，反射必须先放大再算。
        // 这个量级下每个元素只剩三十几位有效数字，所以重构只能要求到 1e-8；Q 的正交性不受影响。
        const Matrix a0 = random_matrix(12, 9, 700);
        Matrix a = a0;
        for (int j = 0; j < 9; ++j)
            for (int i = 0; i < 12; ++i) a(i, j) = cplx(std::ldexp(a0(i, j).real(), -1040), std::ldexp(a0(i, j).imag(), -1040));
        const QR f = qr_factor(a);
        const Matrix q = qr_q(f);
        Matrix r = qr_r(f);
        for (int j = 0; j < 9; ++j)
            for (int i = 0; i <= j; ++i) r(i, j) = cplx(std::ldexp(r(i, j).real(), 1040), std::ldexp(r(i, j).imag(), 1040));
        CHECK(q.all_finite() && r.all_finite());
        CHECK_LE(orthogonality(q), kTol);
        CHECK_LE(norm_fro(q * r - a0) / norm_fro(a0), 1e-8);
    }
}

static void test_singular_and_arguments() {
    say("[奇异与非法参数]\n");
    {   // 零矩阵：分解本身没问题，求解时报奇异
        const QR f = qr_factor(Matrix(4, 3));
        CHECK(f.qr.all_finite());
        CHECK_LE(orthogonality(qr_q(f)), kTol);
        CHECK_THROWS(qr_solve(f, Matrix(4, 1)), std::runtime_error);
    }
    {   // 最后一列为零
        Matrix a = random_matrix(5, 5, 8);
        for (int i = 0; i < 5; ++i) a(i, 4) = 0.0;
        CHECK_THROWS(solve_qr(a, Matrix(5, 1)), std::runtime_error);
    }
    CHECK_THROWS(qr_factor(Matrix(3, 4)), std::invalid_argument);
    CHECK_THROWS(solve_qr(Matrix(3, 4), Matrix(3, 1)), std::invalid_argument);
    const Matrix a = random_matrix(5, 3, 9);
    CHECK_THROWS(qr_solve(qr_factor(a), Matrix(4, 1)), std::invalid_argument);
    CHECK_THROWS(solve_qr(a, Matrix(3, 1)), std::invalid_argument);
    QR broken = qr_factor(a);
    broken.tau.pop_back();
    CHECK_THROWS(qr_q(broken), std::invalid_argument);
    CHECK_THROWS(qr_solve(broken, Matrix(5, 1)), std::invalid_argument);
}

static void test_threads() {
    say("[线程数不影响结果]\n");
    const int m = 320, n = 260;
    const Matrix a = random_matrix(m, n, 21), b = random_matrix(m, 150, 22);
    set_num_threads(1);
    const QR f1 = qr_factor(a), f1b = qr_factor(a);
    const Matrix q1 = qr_q(f1), x1 = qr_solve(f1, b);
    CHECK(same_bits(f1.qr, f1b.qr) && f1.tau == f1b.tau);
    for (int threads : {2, 3, 4, 8}) {
        set_num_threads(threads);
        const QR f = qr_factor(a);
        CHECK(same_bits(f.qr, f1.qr) && f.tau == f1.tau);
        CHECK(same_bits(qr_q(f), q1));
        CHECK(same_bits(qr_solve(f, b), x1));
    }
    set_num_threads(1);
    CHECK_LE(orthogonality(q1), kTol);
    CHECK_LE(norm_fro(q1 * qr_r(f1) - a) / norm_fro(a), kTol);
    CHECK_LE(normal_residual(a, x1, b), kTol);
    say("  %d×%d：1、2、3、4、8 线程的分解、Q、最小二乘解逐位相同；正交性 %.2e\n", m, n, orthogonality(q1));
}

int main() {
    test_basic();
    test_random();
    test_hard_cases();
    test_phase_conventions();
    test_panel_boundaries_and_least_squares();
    test_singular_and_arguments();
    test_threads();
    say("通过 %d / 共 %d\n", g_pass, g_total);
    return g_pass == g_total ? 0 : 1;
}
