// tests/test_lu.cpp —— lu.hpp（LU 分解、解方程、求逆）的测试
//
//   c++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -I include tests/test_lu.cpp -o build/test_lu
//
// 只依赖 C++17 标准库；测试矩阵由固定种子的 Rng 生成。主要度量：
//   后向误差   ||A X − B||_F / (||A||_F ||X||_F + ||B||_F)，稳定的解法应在几个 eps 之内，与条件数无关
//   重构误差   ||P·A − L·U||_F / ||A||_F
// 另外用一份照教科书逐步消元写成的参考实现，核对行交换、增长因子和最大乘子。
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "stablercwa/lu.hpp"

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
// 固定种子的随机数（splitmix64）
struct Rng {
    std::uint64_t s;
    explicit Rng(std::uint64_t seed) : s(seed) {}
    std::uint64_t next() {
        std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    double uniform() { return static_cast<double>(next() >> 11) * (2.0 / 9007199254740992.0) - 1.0; }   // [-1, 1)
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

static const double kTol = 50.0 * kEps;   // 后向误差的容差

static bool same_bits(const Matrix& a, const Matrix& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    const std::size_t bytes = sizeof(cplx) * static_cast<std::size_t>(a.rows()) * a.cols();
    return bytes == 0 || std::memcmp(a.data(), b.data(), bytes) == 0;
}
static bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; }

static Matrix unit_lower(const Matrix& lu) {
    Matrix l = Matrix::identity(lu.rows());
    for (int j = 0; j < lu.cols(); ++j)
        for (int i = j + 1; i < lu.rows(); ++i) l(i, j) = lu(i, j);
    return l;
}
static Matrix upper(const Matrix& lu) {
    Matrix u(lu.rows(), lu.cols());
    for (int j = 0; j < lu.cols(); ++j)
        for (int i = 0; i <= j; ++i) u(i, j) = lu(i, j);
    return u;
}
// P·A：按 perm 依次交换行
static Matrix permute_rows(Matrix a, const std::vector<int>& perm) {
    for (int k = 0; k < static_cast<int>(perm.size()); ++k)
        if (perm[k] != k)
            for (int j = 0; j < a.cols(); ++j) std::swap(a(k, j), a(perm[k], j));
    return a;
}
static double reconstruction_error(const Matrix& a, const LU& f) {
    return norm_fro(permute_rows(a, f.perm) - unit_lower(f.lu) * upper(f.lu)) / norm_fro(a);
}

// 教科书式的逐步消元（不分块、不并行、用 std::complex 的运算符），带同样定义的增长因子和最大乘子。
static LU reference_lu(Matrix a, Pivoting pivoting) {
    const int n = a.rows();
    LU f;
    f.perm.resize(n);
    const double amax = max_abs(a);
    double seen = amax;
    for (int k = 0; k < n; ++k) {
        int p = k;
        if (pivoting == Pivoting::Partial)
            for (int i = k + 1; i < n; ++i)
                if (std::abs(a(i, k)) > std::abs(a(p, k))) p = i;
        f.perm[k] = p;
        if (p != k)
            for (int j = 0; j < n; ++j) std::swap(a(k, j), a(p, j));
        if (a(k, k) == cplx{}) {
            f.singular = true;
            for (int i = k + 1; i < n; ++i) a(i, k) = cplx{};
            continue;
        }
        for (int i = k + 1; i < n; ++i) {
            a(i, k) /= a(k, k);
            f.max_multiplier = std::max(f.max_multiplier, std::abs(a(i, k)));
        }
        for (int j = k + 1; j < n; ++j)
            for (int i = k + 1; i < n; ++i) {
                a(i, j) -= a(i, k) * a(k, j);
                seen = std::max(seen, std::abs(a(i, j)));
            }
    }
    f.growth = amax > 0.0 ? seen / amax : 1.0;
    f.lu = std::move(a);
    return f;
}

// ---------------------------------------------------------------------------------------------
static void test_basic() {
    say("[基本情形]\n");
    {   // 空矩阵
        const LU f = lu_factor(Matrix());
        CHECK(f.lu.empty() && f.perm.empty() && !f.singular && f.growth == 1.0 && f.max_multiplier == 0.0);
        CHECK(lu_solve(f, Matrix()).empty());
        const Matrix x = lu_solve(f, Matrix(0, 3));               // 0 行 3 列的右端：形状原样返回
        CHECK(x.rows() == 0 && x.cols() == 3);
        CHECK(lu_solve(f, Vector()).empty() && inverse(Matrix()).empty() && solve(Matrix(), Matrix(0, 2)).cols() == 2);
    }
    {   // 1×1 零矩阵：两种模式都报奇异
        for (Pivoting piv : {Pivoting::Partial, Pivoting::None}) {
            const LU f = lu_factor(Matrix{{cplx(0.0, 0.0)}}, piv);
            CHECK(f.singular && f.growth == 1.0 && f.max_multiplier == 0.0 && f.perm[0] == 0);
        }
    }
    {   // 1×1
        const Matrix a{{cplx(0.0, 2.0)}};
        const LU f = lu_factor(a);
        CHECK(!f.singular && f.perm.size() == 1 && f.perm[0] == 0 && f.growth == 1.0);
        const Vector x = solve(a, Vector{cplx(4.0, 0.0)});
        CHECK_LE(std::abs(x[0] - cplx(0.0, -2.0)), 4 * kEps);
    }
    {   // 单位阵：分解就是它自己，解就是右端
        const Matrix eye = Matrix::identity(5), b = random_matrix(5, 2, 1);
        const LU f = lu_factor(eye);
        CHECK(same_bits(f.lu, eye) && !f.singular && f.growth == 1.0 && f.max_multiplier == 0.0);
        for (int k = 0; k < 5; ++k) CHECK(f.perm[k] == k);
        CHECK(same_bits(lu_solve(f, b), b));
    }
    {   // 对角阵
        Rng rng(2);
        Vector d(6), b(6);
        for (int i = 0; i < 6; ++i) { d[i] = rng.cuniform() + cplx(2.0, 0.0); b[i] = rng.cuniform(); }
        const Vector x = solve(Matrix::diagonal(d), b);
        double worst = 0.0;
        for (int i = 0; i < 6; ++i) worst = std::max(worst, std::abs(x[i] - b[i] / d[i]) / std::abs(x[i]));
        CHECK_LE(worst, 4 * kEps);
        CHECK(lu_factor(Matrix::diagonal(d)).max_multiplier == 0.0);
    }
    {   // 手算的 2×2：第 1 步要换行。P A = L U，L = [[1,0],[0.5,1]]，U = [[4,3],[0,-0.5]]
        const Matrix a{{2.0, 1.0}, {4.0, 3.0}};
        const LU f = lu_factor(a);
        CHECK(f.perm[0] == 1 && f.perm[1] == 1);
        CHECK(f.lu(0, 0) == cplx(4.0) && f.lu(0, 1) == cplx(3.0) && f.lu(1, 0) == cplx(0.5) && f.lu(1, 1) == cplx(-0.5));
        CHECK(f.max_multiplier == 0.5 && f.growth == 1.0 && !f.singular);
        const LU g = lu_factor(a, Pivoting::None);   // 不换行：L = [[1,0],[2,1]]，U = [[2,1],[0,1]]
        CHECK(g.perm[0] == 0 && g.perm[1] == 1);
        CHECK(g.lu(0, 0) == cplx(2.0) && g.lu(0, 1) == cplx(1.0) && g.lu(1, 0) == cplx(2.0) && g.lu(1, 1) == cplx(1.0));
        CHECK(g.max_multiplier == 2.0);
    }
    {   // 主元并列时取下标最小的那个（与 LAPACK 相同）：这里 |1| = |−1| = |i|，都不换行
        CHECK(lu_factor(Matrix{{1.0, 2.0}, {-1.0, 3.0}}).perm[0] == 0);
        CHECK(lu_factor(Matrix{{cplx(0.0, 1.0), 2.0}, {1.0, 3.0}}).perm[0] == 0);
    }
    {   // 右端与系数矩阵是同一个对象：A X = A 的解是单位阵；用分解自己的 lu 当右端也没问题
        const Matrix a = random_matrix(37, 37, 6);
        CHECK_LE(max_abs(solve(a, a) - Matrix::identity(37)), 1e-12);
        const LU f = lu_factor(a);
        CHECK_LE(solve_backward_error(a, lu_solve(f, f.lu), f.lu), kTol);
    }
    {   // 上三角、下三角
        Matrix up = random_matrix(8, 8, 3), lo = random_matrix(8, 8, 4);
        for (int j = 0; j < 8; ++j)
            for (int i = 0; i < 8; ++i) {
                if (i > j) up(i, j) = 0.0;
                if (i < j) lo(i, j) = 0.0;
                if (i == j) { up(i, j) += cplx(3.0, 0.0); lo(i, j) += cplx(3.0, 0.0); }
            }
        const Matrix b = random_matrix(8, 3, 5);
        const LU fu = lu_factor(up);
        CHECK(same_bits(fu.lu, up) && fu.max_multiplier == 0.0 && fu.growth == 1.0);   // 上三角不需要任何消元
        CHECK_LE(solve_backward_error(up, lu_solve(fu, b), b), kTol);
        CHECK_LE(solve_backward_error(lo, solve(lo, b), b), kTol);
        CHECK_LE(reconstruction_error(lo, lu_factor(lo)), kTol);
    }
}

static void test_random() {
    say("[随机复矩阵]\n");
    for (int n : {1, 2, 3, 10, 50, 200}) {
        const Matrix a = random_matrix(n, n, 100 + n), b = random_matrix(n, 3, 200 + n);
        const LU f = lu_factor(a);
        CHECK(!f.singular);
        CHECK_LE(f.max_multiplier, 1.0 + 8 * kEps);              // 选了主元，乘子的模不超过 1
        CHECK(f.growth >= 1.0 && f.growth <= n);                  // 随机矩阵上增长很温和
        CHECK_LE(reconstruction_error(a, f), kTol);
        const Matrix x = lu_solve(f, b);
        const double berr = solve_backward_error(a, x, b);
        CHECK_LE(berr, kTol);
        // 单个右端向量的重载：与矩阵右端的对应列逐位相同
        const Vector x0 = lu_solve(f, b.column(0));
        CHECK(std::memcmp(x0.data(), x.col(0), sizeof(cplx) * n) == 0);
        CHECK(same_bits(solve(a, b), x));                         // solve() 内部走“不统计增长因子”的一支，解逐位相同
        CHECK(same_bits(solve(a, b, Pivoting::None), lu_solve(lu_factor(a, Pivoting::None), b)));
        // 求逆
        const Matrix inv = inverse(a), eye = Matrix::identity(n);
        CHECK_LE(solve_backward_error(a, inv, eye), kTol);
        const double left = max_abs(inv * a - eye);
        CHECK_LE(left, 1e-10);
        say("  n=%3d  后向误差 %.2e  重构 %.2e  growth %.2f  max|乘子| %.6f  max|A^{-1}A − I| %.2e\n", n, berr,
            reconstruction_error(a, f), f.growth, f.max_multiplier, left);
    }
}

static void test_ill_conditioned() {
    say("[病态矩阵：后向误差仍然是 eps 量级]\n");
    for (int n : {6, 10, 13}) {
        Matrix h(n, n), c(n, n);   // Hilbert 矩阵，以及一个复的 Cauchy 型矩阵
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                h(i, j) = 1.0 / (i + j + 1);
                c(i, j) = cplx(1.0, 0.0) / cplx(i + j + 1.0, 0.5);
            }
        const Vector ones(static_cast<std::size_t>(n), cplx(1.0, 0.0));
        for (const Matrix* m : {&h, &c}) {
            Matrix b(n, 1);
            const Vector rhs = *m * ones;                // 真解是全 1
            for (int i = 0; i < n; ++i) b(i, 0) = rhs[i];
            const Matrix x = solve(*m, b);
            const double berr = solve_backward_error(*m, x, b);
            double ferr = 0.0;
            for (int i = 0; i < n; ++i) ferr = std::max(ferr, std::abs(x(i, 0) - 1.0));
            CHECK_LE(berr, kTol);
            say("  %s n=%2d  后向误差 %.2e  前向误差 %.2e\n", m == &h ? "Hilbert" : "Cauchy ", n, berr, ferr);
            if (n == 6) CHECK_LE(ferr, 1e-8);            // 条件数约 1e7，前向误差约 cond·eps
        }
    }
}

static void test_pivoting() {
    say("[是否选主元的差别]\n");
    {   // 经典的 2×2：真解约为 (1, 1)
        const Matrix a{{1e-20, 1.0}, {1.0, 1.0}};
        const Matrix b{{1.0}, {2.0}};
        const LU fp = lu_factor(a), fn = lu_factor(a, Pivoting::None);
        const Matrix xp = lu_solve(fp, b), xn = lu_solve(fn, b);
        CHECK_LE(std::abs(xp(0, 0) - 1.0), 4 * kEps);
        CHECK_LE(std::abs(xp(1, 0) - 1.0), 4 * kEps);
        CHECK_LE(solve_backward_error(a, xp, b), kTol);
        CHECK(fp.max_multiplier <= 1.0 && fp.growth <= 2.0);
        CHECK(std::abs(xn(0, 0) - 1.0) > 0.5);           // 不选主元：x_1 全错
        CHECK(solve_backward_error(a, xn, b) > 0.1);
        CHECK(fn.max_multiplier == 1e20 && fn.growth >= 0.99e20 && !fn.singular);
        say("  2×2：选主元 x=(%.16g, %.16g)，不选主元 x=(%.3g, %.3g)，growth %.1e，max|乘子| %.1e\n", xp(0, 0).real(),
            xp(1, 0).real(), xn(0, 0).real(), xn(1, 0).real(), fn.growth, fn.max_multiplier);
    }
    for (int n : {10, 50}) {   // 随机矩阵，把左上角换成很小的数
        Matrix a = random_matrix(n, n, 300 + n);
        a(0, 0) = cplx(1e-13, 0.0);
        const Matrix b = random_matrix(n, 2, 400 + n);
        const LU fp = lu_factor(a), fn = lu_factor(a, Pivoting::None);
        const double ep = solve_backward_error(a, lu_solve(fp, b), b);
        const double en = solve_backward_error(a, lu_solve(fn, b), b);
        CHECK_LE(ep, kTol);
        CHECK(en > 1e-9 && en > 1e4 * ep);               // 不选主元时误差放大了许多个量级
        CHECK(fn.growth > 1e10 && fn.max_multiplier > 1e10);
        CHECK(fp.growth < n && fp.max_multiplier <= 1.0 + 8 * kEps);
        for (int k = 0; k < n; ++k) CHECK(fn.perm[k] == k);
        say("  n=%2d 左上角 1e-13：后向误差 选主元 %.2e / 不选 %.2e；growth %.2f / %.2e\n", n, ep, en, fp.growth,
            fn.growth);
    }
    {   // 对角元为零的置换矩阵：选主元毫无困难，不选主元第一步就遇到零主元
        const Matrix a{{0.0, 1.0}, {1.0, 0.0}};
        const Matrix b{{3.0}, {5.0}};
        const LU fp = lu_factor(a);
        CHECK(!fp.singular);
        const Matrix x = lu_solve(fp, b);
        CHECK(x(0, 0) == cplx(5.0) && x(1, 0) == cplx(3.0));
        const LU fn = lu_factor(a, Pivoting::None);
        CHECK(fn.singular && fn.lu.all_finite());
        CHECK_THROWS(solve(a, b, Pivoting::None), std::runtime_error);
        CHECK_THROWS(lu_solve(fn, b), std::runtime_error);
    }
}

static void test_singular() {
    say("[奇异矩阵]\n");
    {   // 零矩阵
        const LU f = lu_factor(Matrix(3, 3));
        CHECK(f.singular && f.growth == 1.0 && f.lu.all_finite());
    }
    {   // 秩 1：第二步的主元精确为零
        const Matrix a{{1.0, 2.0}, {2.0, 4.0}};
        const LU f = lu_factor(a);
        CHECK(f.singular && f.lu.all_finite());
        CHECK_THROWS(solve(a, Matrix{{1.0}, {1.0}}), std::runtime_error);
        CHECK_THROWS(solve(a, Vector{1.0, 1.0}), std::runtime_error);
        CHECK_THROWS(inverse(a), std::runtime_error);
        CHECK_THROWS(lu_solve(f, Vector{1.0, 1.0}), std::runtime_error);
    }
    {   // 有一整列为零
        Matrix a = random_matrix(40, 40, 7);
        for (int i = 0; i < 40; ++i) a(i, 17) = 0.0;
        const LU fp = lu_factor(a), fn = lu_factor(a, Pivoting::None);
        CHECK(fp.singular && fp.lu.all_finite());
        CHECK(fn.singular && fn.lu.all_finite());
    }
    {   // 不奇异的矩阵不应被标成奇异
        CHECK(!lu_factor(random_matrix(40, 40, 8)).singular);
        CHECK(!lu_factor(random_matrix(40, 40, 8), Pivoting::None).singular);
    }
}

static void test_arguments() {
    say("[非法参数]\n");
    const Matrix a = random_matrix(4, 4, 9);
    CHECK_THROWS(lu_factor(Matrix(3, 4)), std::invalid_argument);
    CHECK_THROWS(solve(Matrix(3, 4), Matrix(3, 1)), std::invalid_argument);
    CHECK_THROWS(solve(a, Matrix(5, 1)), std::invalid_argument);
    CHECK_THROWS(solve(a, Vector(3)), std::invalid_argument);
    CHECK_THROWS(lu_solve(lu_factor(a), Matrix(5, 2)), std::invalid_argument);
    CHECK_THROWS(lu_solve(lu_factor(a), Vector(3)), std::invalid_argument);
    CHECK_THROWS(inverse(Matrix(2, 3)), std::invalid_argument);
    CHECK_THROWS(solve_backward_error(a, Matrix(3, 1), Matrix(4, 1)), std::invalid_argument);
    CHECK(solve_backward_error(Matrix(2, 2), Matrix(2, 1), Matrix(2, 1)) == 0.0);
    {   // 公式本身：A = I（||A||_F = √2），X = (1, 0)^T。B = 0 时是 1/√2；B = (1, 1)^T 时残差 1、分母 √2 + √2
        const Matrix eye = Matrix::identity(2), x{{1.0}, {0.0}};
        CHECK_LE(std::abs(solve_backward_error(eye, x, Matrix(2, 1)) - std::sqrt(0.5)), 4 * kEps);
        CHECK_LE(std::abs(solve_backward_error(eye, x, Matrix{{1.0}, {1.0}}) - 0.5 * std::sqrt(0.5)), 4 * kEps);
        CHECK(solve_backward_error(Matrix(2, 2), Matrix(2, 1), Matrix{{1.0}, {0.0}}) == 1.0);   // A = 0、X = 0：残差就是 B
    }
    // 输入含 NaN：不抛异常，growth 报 NaN
    Matrix bad = a;
    bad(2, 1) = cplx(std::numeric_limits<double>::quiet_NaN(), 0.0);
    CHECK(std::isnan(lu_factor(bad).growth));
}

static void test_growth_and_reference() {
    say("[增长因子、最大乘子；与逐步消元的参考实现对照]\n");
    {   // Wilkinson 的最坏例子：对角线 1、下三角 −1、最后一列 1；列主元下增长因子恰为 2^(n-1)
        const int n = 10;
        Matrix w(n, n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) w(i, j) = (i == j || j == n - 1) ? 1.0 : (i > j ? -1.0 : 0.0);
        const LU f = lu_factor(w);
        CHECK(f.growth == 512.0 && f.max_multiplier == 1.0 && !f.singular);
        CHECK(f.lu(n - 1, n - 1) == cplx(512.0));
    }
    {   // growth 要看消元途中的所有中间值，不只是最后的 U。这个例子不换行时：
        //   第 1 步后右下角变成 10.5（比输入的最大元素 10 还大），第 2 步又被消成 0.5；U 的最大元素只有 9。
        const Matrix a{{1.0, 0.0, -9.0}, {0.0, 1.0, 1.0}, {1.0, 10.0, 1.5}};
        const LU f = lu_factor(a, Pivoting::None), r = reference_lu(a, Pivoting::None);
        CHECK(f.lu(2, 2) == cplx(0.5) && max_abs(upper(f.lu)) == 9.0);
        CHECK(f.growth == 10.5 / 10.0 && r.growth == 10.5 / 10.0);
        CHECK(f.max_multiplier == 10.0);
    }
    // n 取在面板边界（32、64）两侧
    for (int n : {7, 31, 32, 33, 63, 64, 65, 70, 96, 97, 130}) {
        for (Pivoting piv : {Pivoting::Partial, Pivoting::None}) {
            Matrix a = random_matrix(n, n, 500 + n);
            if (piv == Pivoting::None)                    // 不选主元时加重对角线，免得增长失控、比较失去意义
                for (int i = 0; i < n; ++i) a(i, i) += cplx(0.3 * n, 0.0);
            const LU f = lu_factor(a, piv), r = reference_lu(a, piv);
            CHECK(f.perm == r.perm);
            CHECK(f.singular == r.singular);
            CHECK_LE(std::abs(f.growth - r.growth) / r.growth, 1e-12);
            CHECK_LE(std::abs(f.max_multiplier - r.max_multiplier) / r.max_multiplier, 1e-12);
            CHECK_LE(norm_fro(f.lu - r.lu) / norm_fro(r.lu), 1e-12);
        }
    }
    {   // 不选主元、对角线偏小：增长明显，两种实现给出的 growth 仍应一致
        Matrix a = random_matrix(24, 24, 11);
        for (int i = 0; i < 24; ++i) a(i, i) *= 0.05;
        const LU f = lu_factor(a, Pivoting::None), r = reference_lu(a, Pivoting::None);
        CHECK(f.growth > 10.0);
        CHECK_LE(std::abs(f.growth - r.growth) / r.growth, 1e-8);
        CHECK_LE(std::abs(f.max_multiplier - r.max_multiplier) / r.max_multiplier, 1e-8);
        say("  不选主元、对角线偏小（n=24）：growth %.3e（参考 %.3e），max|乘子| %.3e\n", f.growth, r.growth,
            f.max_multiplier);
    }
    {   // 量级极端的矩阵：乘以 2 的幂只是把 U 整体缩放，L、行交换、growth 都不变
        const Matrix a = random_matrix(20, 20, 12);
        const LU f = lu_factor(a);
        for (int e : {700, -700}) {
            Matrix s = a;
            for (int j = 0; j < 20; ++j)
                for (int i = 0; i < 20; ++i) s(i, j) = cplx(std::ldexp(a(i, j).real(), e), std::ldexp(a(i, j).imag(), e));
            const LU g = lu_factor(s);
            Matrix expect = f.lu;
            for (int j = 0; j < 20; ++j)
                for (int i = 0; i <= j; ++i)
                    expect(i, j) = cplx(std::ldexp(f.lu(i, j).real(), e), std::ldexp(f.lu(i, j).imag(), e));
            CHECK(g.perm == f.perm && same_bits(g.lu, expect));
            CHECK(g.growth == f.growth && g.max_multiplier == f.max_multiplier && !g.singular);
            const Matrix b = random_matrix(20, 2, 13);
            CHECK_LE(solve_backward_error(s, lu_solve(g, b), b), kTol);
            CHECK(same_bits(solve(s, b), lu_solve(g, b)));           // solve() 内部不统计增长因子，解应当一样
        }
    }
}

// 用分解结果 (lu, perm) 逐列、逐步地做前代和回代（std::complex 运算符，不分块）：
// 既是 lu_solve 的参考实现，也把 perm 的约定（第 k 步交换第 k 行与第 perm[k] 行，按 k 从小到大依次作用）钉住。
static Matrix reference_solve(const LU& f, Matrix b) {
    const int n = f.lu.rows();
    for (int j = 0; j < b.cols(); ++j) {
        for (int k = 0; k < n; ++k)
            if (f.perm[k] != k) std::swap(b(k, j), b(f.perm[k], j));
        for (int k = 0; k < n; ++k)
            for (int i = k + 1; i < n; ++i) b(i, j) -= f.lu(i, k) * b(k, j);
        for (int k = n - 1; k >= 0; --k) {
            b(k, j) /= f.lu(k, k);
            for (int i = 0; i < k; ++i) b(i, j) -= f.lu(i, k) * b(k, j);
        }
    }
    return b;
}

static void test_panel_boundaries() {
    say("[面板边界两侧的 n、各种右端列数]\n");
    double worst = 0.0, worst_ref = 0.0;
    for (int n : {4, 5, 31, 32, 33, 35, 36, 37, 63, 64, 65, 68, 69}) {
        for (Pivoting piv : {Pivoting::Partial, Pivoting::None}) {
            Matrix a = random_matrix(n, n, 2000 + n);
            if (piv == Pivoting::None)
                for (int i = 0; i < n; ++i) a(i, i) += cplx(0.3 * n + 1.0, 0.0);
            const LU f = lu_factor(a, piv);
            CHECK_LE(reconstruction_error(a, f), kTol);
            for (int nrhs : {0, 1, 5, n + 3}) {
                const Matrix b = random_matrix(n, nrhs, 3000 + n + nrhs);
                const Matrix x = lu_solve(f, b);
                CHECK(x.rows() == n && x.cols() == nrhs);
                if (nrhs == 0) continue;
                const double berr = solve_backward_error(a, x, b);
                const Matrix y = reference_solve(f, b);
                const double dref = norm_fro(x - y) / norm_fro(y);
                CHECK_LE(berr, kTol);
                CHECK_LE(dref, 1e-11);
                worst = std::max(worst, berr);
                worst_ref = std::max(worst_ref, dref);
                const Vector xlast = lu_solve(f, b.column(nrhs - 1));   // 多个右端一起解与单独解一列，逐位相同
                CHECK(std::memcmp(xlast.data(), x.col(nrhs - 1), sizeof(cplx) * n) == 0);
                CHECK(same_bits(solve(a, b, piv), x));
            }
        }
    }
    say("  最大后向误差 %.2e；与逐列、不分块的参考回代的最大相对差 %.2e\n", worst, worst_ref);
}

// lu_solve；分解被（误）标成奇异时返回空矩阵，让后面的 CHECK 报失败，而不是让异常中断整个测试。
static Matrix try_lu_solve(const LU& f, const Matrix& b) {
    try {
        return lu_solve(f, b);
    } catch (const std::runtime_error&) {
        return Matrix();
    }
}

static void test_zero_pivot_and_extremes() {
    say("[零主元的处理、极端的主元与乘子、量级跨度极大的矩阵]\n");
    {   // 不选主元、对角线上有零而下方非零：这一步跳过，乘子记为 0，其余各步照常
        Matrix t(6, 6);   // 下三角：对角线 2（t(2,2) = 0 除外），对角线以下全是 1
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j <= i; ++j) t(i, j) = (i == j) ? (i == 2 ? 0.0 : 2.0) : 1.0;
        const LU f = lu_factor(t, Pivoting::None);
        CHECK(f.singular && f.lu.all_finite());
        CHECK(f.lu(3, 2) == cplx{} && f.lu(4, 2) == cplx{} && f.lu(5, 2) == cplx{});      // 第 2 步的乘子
        CHECK(f.lu(1, 0) == cplx(0.5) && f.lu(5, 4) == cplx(0.5) && f.lu(5, 5) == cplx(2.0));   // 其余各步不受影响
        CHECK(f.max_multiplier == 0.5 && f.growth == 1.0);
        for (int k = 0; k < 6; ++k) CHECK(f.perm[k] == k);
        CHECK(lu_factor(t).singular);                             // 行列式为零，选主元也一样是奇异的
    }
    {   // 零主元出现在最后一步、出现在一个面板的最后一列
        Matrix a = Matrix::identity(33);
        a(32, 32) = 0.0;
        CHECK(lu_factor(a).singular && lu_factor(a, Pivoting::None).singular);
        a(32, 32) = 1.0;
        a(31, 31) = 0.0;
        CHECK(lu_factor(a).singular && lu_factor(a, Pivoting::None).singular);
        a(31, 31) = 1.0;
        CHECK(!lu_factor(a).singular && !lu_factor(a, Pivoting::None).singular);
    }
    {   // 主元是次正规数：1/主元 溢出，乘子要逐个做除法。这一列是 (2^-1070, 2^-1071)，乘子恰为 0.5
        const Matrix a{{std::ldexp(1.0, -1070), 1.0}, {std::ldexp(1.0, -1071), 1.0}};
        for (Pivoting piv : {Pivoting::Partial, Pivoting::None}) {
            const LU f = lu_factor(a, piv);
            CHECK(!f.singular && f.lu.all_finite() && f.perm[0] == 0);
            CHECK(f.lu(1, 0) == cplx(0.5) && f.lu(1, 1) == cplx(0.5));
            CHECK(f.max_multiplier == 0.5 && f.growth == 1.0);
        }
        const Matrix x = solve(a, Matrix{{1.0}, {1.0}});          // 两个方程相减：x_0 = 0，x_1 = 1
        CHECK(x(0, 0) == cplx{} && x(1, 0) == cplx(1.0));
    }
    {   // 所有元素都是次正规数（且有复数）：主元 i·2^-1070，乘子 2^-1071 / (i·2^-1070) = −0.5i，
        // U(1,1) = 2^-1070 − (−0.5i)·2^-1072 = 2^-1070·(1 + i/8)，全是精确运算；growth = |1 + i/8| = sqrt(65/64)
        const double h = std::ldexp(1.0, -1070);
        const Matrix a{{cplx(0.0, h), cplx(h / 4, 0.0)}, {cplx(h / 2, 0.0), cplx(h, 0.0)}};
        const LU f = lu_factor(a);
        CHECK(!f.singular && f.perm[0] == 0 && f.lu(1, 0) == cplx(0.0, -0.5) && f.lu(1, 1) == cplx(h, h / 8));
        CHECK(f.growth == std::sqrt(65.0 / 64.0) && f.max_multiplier == 0.5);
    }
    {   // 乘子极大（平方会上溢）与极小（平方会下溢）：max_multiplier 仍要报对
        const LU big = lu_factor(Matrix{{1e-200, 1.0}, {1.0, 1.0}}, Pivoting::None);
        CHECK(!big.singular);
        CHECK_LE(std::abs(big.max_multiplier / 1e200 - 1.0), 4 * kEps);
        CHECK(big.growth >= 1e120);                               // 真值 1e200；按文档，超过约 1e120 可以记成 +inf
        for (Pivoting piv : {Pivoting::Partial, Pivoting::None}) {
            const LU tiny = lu_factor(Matrix{{1.0, 0.0}, {1e-200, 1.0}}, piv);
            CHECK(tiny.max_multiplier == 1e-200 && tiny.growth == 1.0 && !tiny.singular);
        }
    }
    {   // 元素量级跨越 400 个数量级的对角阵：既不奇异，也不该丢掉小的那个对角元
        const Matrix a{{1e200, 0.0}, {0.0, 1e-200}};
        for (Pivoting piv : {Pivoting::Partial, Pivoting::None}) {
            const LU f = lu_factor(a, piv);
            CHECK(!f.singular && f.growth == 1.0 && f.max_multiplier == 0.0);
            CHECK(f.lu(0, 0) == cplx(1e200) && f.lu(1, 1) == cplx(1e-200));
            const Matrix x = try_lu_solve(f, Matrix{{1e200}, {1e-200}});
            CHECK(x.rows() == 2 && x(0, 0) == cplx(1.0) && x(1, 0) == cplx(1.0));
        }
    }
    {   // 大元素 1e160 旁边的 3e-160：U(1,1) = 3e-160 − (1/1e160)·1，统计增长因子不应改变它的任何一位
        const Matrix a{{1e160, 1.0}, {1.0, 3e-160}};
        const double l = 1.0 / 1e160, u11 = 3e-160 - l;
        const LU f = lu_factor(a);
        CHECK(f.lu(1, 0) == cplx(l) && f.lu(1, 1) == cplx(u11) && !f.singular);
        CHECK_LE(std::abs(u11 / 2e-160 - 1.0), 1e-15);
        CHECK(f.growth == 1.0 && f.max_multiplier == l);
        const Matrix b{{1.0}, {2.0}};
        CHECK(same_bits(solve(a, b), try_lu_solve(f, b)));
    }
    // 行、列各乘上跨度很大的 2 的幂（D1·B·D2，元素从 2^-960 到 2^960）：
    //   · 量级这样悬殊时 |z|² 会上溢/下溢，growth 与 max_multiplier 仍应与参考实现一致；
    //   · lu_factor（统计增长因子）与 solve()（不统计）必须给出逐位相同的分解。
    for (int trial = 0; trial < 6; ++trial) {
        const int n = 20 + 9 * trial;                             // 20 … 65，跨过面板边界
        Rng rng(4000 + trial);
        Matrix a = random_matrix(n, n, 4100 + trial);
        std::vector<int> er(n), ec(n);
        for (int i = 0; i < n; ++i) {
            a(i, i) += cplx(4.0, 0.0);
            er[i] = static_cast<int>(rng.next() % 961) - 480;
            ec[i] = static_cast<int>(rng.next() % 961) - 480;
        }
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                a(i, j) = cplx(std::ldexp(a(i, j).real(), er[i] + ec[j]), std::ldexp(a(i, j).imag(), er[i] + ec[j]));
        const Matrix b = random_matrix(n, 2, 4200 + trial);
        for (Pivoting piv : {Pivoting::Partial, Pivoting::None}) {
            const LU f = lu_factor(a, piv), r = reference_lu(a, piv);
            CHECK(!f.singular && f.lu.all_finite() && f.perm == r.perm);
            CHECK_LE(std::abs(f.growth - r.growth) / r.growth, 1e-8);
            CHECK_LE(std::abs(f.max_multiplier - r.max_multiplier) / r.max_multiplier, 1e-8);
            CHECK(same_bits(solve(a, b, piv), try_lu_solve(f, b)));
            // 重构：把 P·A − L·U 的各元素按 2^(行指数 + 列指数) 缩回 1 的量级再量（直接求范数会上溢）
            Matrix d = permute_rows(a, f.perm) - unit_lower(f.lu) * upper(f.lu);
            std::vector<int> erp = er;
            for (int k = 0; k < n; ++k) std::swap(erp[k], erp[f.perm[k]]);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    d(i, j) = cplx(std::ldexp(d(i, j).real(), -erp[i] - ec[j]), std::ldexp(d(i, j).imag(), -erp[i] - ec[j]));
            CHECK_LE(max_abs(d), 1e-11 * std::max(1.0, r.growth));
        }
    }
}

static void test_threads() {
    say("[线程数不影响结果]\n");
    const int n = 300;
    const Matrix a = random_matrix(n, n, 21), b = random_matrix(n, n, 22), b7 = random_matrix(n, 7, 23);
    set_num_threads(1);
    const LU f1 = lu_factor(a), f1b = lu_factor(a);
    const Matrix x1 = lu_solve(f1, b), x1b = lu_solve(f1b, b), x7 = lu_solve(f1, b7);
    const LU n1 = lu_factor(a, Pivoting::None);
    CHECK(same_bits(f1.lu, f1b.lu) && f1.perm == f1b.perm && same_bits(x1, x1b));   // 单线程逐位可复现
    CHECK(same_bits(f1.growth, f1b.growth) && same_bits(f1.max_multiplier, f1b.max_multiplier));
    for (int threads : {2, 3, 4, 8}) {
        set_num_threads(threads);
        const LU f = lu_factor(a);
        const Matrix x = lu_solve(f, b);
        CHECK(same_bits(lu_solve(f, b7), x7) && same_bits(solve(a, b7), x7));
        const LU nt = lu_factor(a, Pivoting::None);
        CHECK(same_bits(f.lu, f1.lu) && f.perm == f1.perm);
        CHECK(same_bits(f.growth, f1.growth) && same_bits(f.max_multiplier, f1.max_multiplier));
        CHECK(same_bits(x, x1));
        CHECK(same_bits(nt.lu, n1.lu) && same_bits(nt.growth, n1.growth) && same_bits(nt.max_multiplier, n1.max_multiplier));
        CHECK(same_bits(inverse(a), lu_solve(f1, Matrix::identity(n))));
    }
    set_num_threads(1);
    CHECK_LE(solve_backward_error(a, x1, b), kTol);
    CHECK_LE(reconstruction_error(a, f1), kTol);
    say("  n=%d：1、2、3、4、8 线程的 LU、增长因子、解逐位相同；后向误差 %.2e\n", n, solve_backward_error(a, x1, b));
}

int main() {
    test_basic();
    test_random();
    test_ill_conditioned();
    test_pivoting();
    test_singular();
    test_arguments();
    test_growth_and_reference();
    test_panel_boundaries();
    test_zero_pivot_and_extremes();
    test_threads();
    say("通过 %d / 共 %d\n", g_pass, g_total);
    return g_pass == g_total ? 0 : 1;
}
