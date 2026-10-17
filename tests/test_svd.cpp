// tests/test_svd.cpp —— svd.hpp（单边 Jacobi 奇异值、norm_2、cond2）的测试
//
//   c++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -I include tests/test_svd.cpp -o build/test_svd
//
// 只依赖 C++17 标准库；测试矩阵由固定种子的 Rng 生成。重点是小奇异值的“相对”精度：
//   · 列分级 U·diag(s)、行分级 diag(s)·U：奇异值就是 s，逐个比较相对误差；
//   · 行列同时分级 D1·U·D2：它的逆 D2^{-1}·U^H·D1^{-1} 可以逐元素直接写出来，
//     而 σ_i(A)·σ_{n-1-i}(A^{-1}) = 1。只保证绝对精度 eps·σ_max 的算法过不了这一关；
//   · 一个元素都是“小整数 × 2 的幂”的分级矩阵，参考值是用 80 位十进制运算另行算出的。
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
#include "stablercwa/qr.hpp"
#include "stablercwa/svd.hpp"

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
// m×n（m >= n）、各列标准正交的矩阵：随机矩阵做 Householder QR 取 Q。
static Matrix random_isometry(int m, int n, std::uint64_t seed) { return qr_q(qr_factor(random_matrix(m, n, seed))); }
static Matrix random_unitary(int n, std::uint64_t seed) { return random_isometry(n, n, seed); }

// 从 1 等比下降到 smallest 的 n 个数
static RealVector geometric(int n, double smallest) {
    RealVector s(static_cast<std::size_t>(n), 1.0);
    for (int i = 1; i < n; ++i) s[i] = std::pow(smallest, static_cast<double>(i) / (n - 1));
    return s;
}
static Vector as_complex(const RealVector& s) { return Vector(s.begin(), s.end()); }
static RealVector reciprocal(const RealVector& s) {
    RealVector r(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) r[i] = 1.0 / s[i];
    return r;
}

static double max_relative_error(const RealVector& got, const RealVector& expect) {
    if (got.size() != expect.size()) return std::numeric_limits<double>::infinity();
    double worst = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) worst = std::max(worst, std::abs(got[i] - expect[i]) / expect[i]);
    return worst;
}
static double max_absolute_error(const RealVector& got, const RealVector& expect) {
    if (got.size() != expect.size()) return std::numeric_limits<double>::infinity();
    double worst = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) worst = std::max(worst, std::abs(got[i] - expect[i]));
    return worst;
}
static bool is_descending(const RealVector& s) {
    for (std::size_t i = 1; i < s.size(); ++i)
        if (!(s[i - 1] >= s[i])) return false;
    return true;
}
static bool same_bits(const RealVector& a, const RealVector& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), sizeof(double) * a.size()) == 0);
}
static const double kInf = std::numeric_limits<double>::infinity();

// ---------------------------------------------------------------------------------------------
static void test_basic() {
    say("[基本情形]\n");
    CHECK(singular_values(Matrix()).empty());
    CHECK(singular_values(Matrix(3, 0)).empty() && singular_values(Matrix(0, 3)).empty());
    CHECK(norm_2(Matrix()) == 0.0);
    CHECK_THROWS(cond2(Matrix()), std::invalid_argument);
    {   // 1×1
        const RealVector s = singular_values(Matrix{{cplx(3.0, -4.0)}});
        CHECK(s.size() == 1);
        CHECK_LE(std::abs(s[0] - 5.0), 4 * kEps);
        CHECK(cond2(Matrix{{cplx(3.0, -4.0)}}) == 1.0);
    }
    {   // 单位阵
        const RealVector s = singular_values(Matrix::identity(7));
        for (double v : s) CHECK(v == 1.0);
        CHECK(cond2(Matrix::identity(7)) == 1.0 && norm_2(Matrix::identity(7)) == 1.0);
    }
    {   // 复对角阵：奇异值是对角元的模，降序
        const Vector d = {cplx(0.0, 2.0), cplx(-7.0, 0.0), cplx(3.0, 4.0), cplx(1e-9, 0.0), cplx(0.5, -0.5)};
        const RealVector s = singular_values(Matrix::diagonal(d));
        const RealVector expect = {7.0, 5.0, 2.0, std::sqrt(0.5), 1e-9};
        CHECK(is_descending(s));
        CHECK_LE(max_relative_error(s, expect), 4 * kEps);
        CHECK_LE(std::abs(cond2(Matrix::diagonal(d)) / 7e9 - 1.0), 8 * kEps);
    }
    {   // 手算的 2×2：[[3,0],[4,5]] 的奇异值是 3√5 和 √5
        const Matrix a{{3.0, 0.0}, {4.0, 5.0}};
        const RealVector s = singular_values(a);
        CHECK_LE(max_relative_error(s, {3.0 * std::sqrt(5.0), std::sqrt(5.0)}), 8 * kEps);
        CHECK_LE(std::abs(cond2(a) - 3.0), 16 * kEps);
        CHECK_LE(std::abs(norm_2(a) - 3.0 * std::sqrt(5.0)), 16 * kEps);
    }
    {   // 秩 1：u·v^H 只有一个非零奇异值 |u|·|v|
        const Matrix u = random_matrix(6, 1, 1), v = random_matrix(4, 1, 2);
        const double expect = norm_fro(u) * norm_fro(v);
        CHECK_LE(std::abs(norm_2(u * adjoint(v)) - expect) / expect, 16 * kEps);
        CHECK_LE(std::abs(norm_2(v * adjoint(u)) - expect) / expect, 16 * kEps);
    }
}

static void test_known_singular_values() {
    say("[已知奇异值：A = U·diag(s)·V^H，U、V 由 Householder QR 生成]\n");
    for (int n : {1, 2, 3, 10, 50, 120}) {
        const Matrix u = random_unitary(n, 100 + n), v = random_unitary(n, 200 + n);
        // 条件数不大：每个奇异值都应有相对精度
        RealVector s(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) s[i] = 10.0 - 9.0 * i / std::max(1, n - 1);
        const RealVector got = singular_values(scale_columns(u, as_complex(s)) * adjoint(v));
        const double rel = max_relative_error(got, s);
        CHECK(is_descending(got));
        CHECK_LE(rel, 1e-13);
        // 条件数 1e14、没有分级结构：构造 A 时的舍入已经是 eps·σ_max，只能要求绝对精度
        const RealVector wide = geometric(n, 1e-14);
        const double abs_err = max_absolute_error(singular_values(scale_columns(u, as_complex(wide)) * adjoint(v)), wide);
        CHECK_LE(abs_err, 1e-13);
        say("  n=%3d  s∈[1,10] 最大相对误差 %.2e；s∈[1e-14,1] 最大绝对误差 %.2e\n", n, rel, abs_err);
    }
    {   // 长方阵：高瘦与矮胖（后者在内部转成共轭转置来算）
        const int m = 40, n = 12;
        const Matrix u = random_isometry(m, n, 301), v = random_unitary(n, 302);
        RealVector s(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) s[i] = 5.0 / (1.0 + i);
        const Matrix a = scale_columns(u, as_complex(s)) * adjoint(v);
        const RealVector tall = singular_values(a), fat = singular_values(adjoint(a));
        CHECK(tall.size() == 12 && fat.size() == 12);
        CHECK_LE(max_relative_error(tall, s), 1e-13);
        CHECK(same_bits(tall, fat));
        CHECK_LE(std::abs(cond2(a) - 12.0) / 12.0, 1e-13);
    }
    for (int n : {3, 10, 50, 120}) {   // 随机矩阵：Σσ² = ||A||_F²，σ_max 与 Frobenius 范数的关系
        const Matrix a = random_matrix(n + 3, n, 400 + n);
        const RealVector s = singular_values(a);
        double sum = 0.0;
        for (double v : s) sum += v * v;
        const double fro = norm_fro(a);
        CHECK(is_descending(s));
        CHECK_LE(std::abs(std::sqrt(sum) - fro) / fro, 1e-13);
        CHECK(s.front() <= fro * (1 + 1e-14) && s.front() >= fro / std::sqrt(static_cast<double>(n)) * (1 - 1e-14));
        CHECK(norm_2(a) == s.front() && cond2(a) == s.front() / s.back());
    }
    {   // cond2(A) = ||A||_2 · ||A^{-1}||_2
        const Matrix a = random_matrix(30, 30, 500);
        const double c = cond2(a), c2 = norm_2(a) * norm_2(inverse(a));
        CHECK_LE(std::abs(c - c2) / c, 1e-11);
    }
}

static void test_repeated_and_ill_conditioned() {
    say("[重奇异值；无分级结构的病态矩阵（cond 到 1e15）]\n");
    for (int n : {2, 3, 7, 16, 33}) {
        const Matrix u = random_unitary(n, 1100 + n), v = random_unitary(n, 1200 + n), tall = random_isometry(n + 5, n, 1300 + n);
        double worst_abs = 0.0, worst_rel = 0.0;
        for (int kind = 0; kind < 5; ++kind) {
            RealVector s(static_cast<std::size_t>(n));
            for (int i = 0; i < n; ++i) {
                switch (kind) {
                    case 0: s[i] = 1.0; break;                           // 全部相等
                    case 1: s[i] = i < (n + 1) / 2 ? 2.0 : 1.0; break;   // 两簇
                    case 2: s[i] = i < (n + 1) / 2 ? 1.0 : 1e-8; break;  // 两簇，相差悬殊
                    case 3: s[i] = i == 0 ? 1.0 : 1e-12; break;          // 一个大的，其余全等
                    default: s[i] = std::ldexp(1.0, -(i / 3)); break;    // 三个一组
                }
            }
            // 一般的 U·Σ·V^H：构造时的舍入就是 eps·σ_max，只能要求绝对精度
            worst_abs = std::max(worst_abs, max_absolute_error(singular_values(scale_columns(u, as_complex(s)) * adjoint(v)), s));
            worst_abs = std::max(worst_abs, max_absolute_error(singular_values(scale_columns(tall, as_complex(s)) * adjoint(v)), s));
            // U·Σ（列缩放）与 Σ·U（行缩放）：重奇异值也应有相对精度
            worst_rel = std::max(worst_rel, max_relative_error(singular_values(scale_columns(u, as_complex(s))), s));
            worst_rel = std::max(worst_rel, max_relative_error(singular_values(scale_rows(u, as_complex(s))), s));
            worst_rel = std::max(worst_rel, max_relative_error(singular_values(Matrix::diagonal(as_complex(s))), s));
        }
        CHECK_LE(worst_abs, 1e-13);
        CHECK_LE(worst_rel, 1e-13);
        say("  n=%2d  重奇异值：U·Σ·V^H 最大绝对误差 %.2e；U·Σ、Σ·U、Σ 最大相对误差 %.2e\n", n, worst_abs, worst_rel);
    }
    for (int n : {10, 40}) {   // cond2 本身：真值 1e12 时应准到 1e-2 以内；1e15 时小奇异值只剩一位有效数字，数量级要对
        const Matrix u = random_unitary(n, 1400 + n), v = random_unitary(n, 1500 + n);
        const double c12 = cond2(scale_columns(u, as_complex(geometric(n, 1e-12))) * adjoint(v));
        const double c15 = cond2(scale_columns(u, as_complex(geometric(n, 1e-15))) * adjoint(v));
        CHECK_LE(std::abs(c12 / 1e12 - 1.0), 1e-2);
        CHECK(c15 > 3e14 && c15 < 3e15);
        say("  n=%2d  无分级结构：cond2 真值 1e12 → %.6e，真值 1e15 → %.3e\n", n, c12, c15);
    }
}

static void test_unitary() {
    say("[酉矩阵：cond2 = 1]\n");
    for (int n : {1, 2, 5, 40, 150}) {
        const Matrix u = random_unitary(n, 600 + n);
        const RealVector s = singular_values(u);
        const double c = cond2(u);
        double worst = 0.0;
        for (double v : s) worst = std::max(worst, std::abs(v - 1.0));
        CHECK_LE(worst, 1e-13);
        CHECK_LE(std::abs(c - 1.0), 1e-13);
        CHECK_LE(std::abs(norm_2(u) - 1.0), 1e-13);
        say("  n=%3d  cond2 − 1 = %.2e\n", n, c - 1.0);
    }
}

static void test_graded() {
    say("[分级矩阵：小奇异值的相对精度]\n");
    for (int n : {4, 20, 60}) {
        for (double smallest : {1e-14, 1e-30}) {
            const Matrix u = random_unitary(n, 700 + n);
            const RealVector s = geometric(n, smallest);
            const double col = max_relative_error(singular_values(scale_columns(u, as_complex(s))), s);
            const double row = max_relative_error(singular_values(scale_rows(u, as_complex(s))), s);
            CHECK_LE(col, 1e-13);
            CHECK_LE(row, 1e-13);
            const double c = cond2(scale_columns(u, as_complex(s)));
            CHECK_LE(std::abs(c * smallest - 1.0), 1e-13);
            say("  n=%2d  σ∈[%.0e,1]：列分级 U·D 相对误差 %.2e，行分级 D·U 相对误差 %.2e，cond2 = %.6e\n", n, smallest,
                col, row, c);
        }
    }
    for (int n : {6, 20, 60}) {
        for (double g : {1e-7, 1e-10}) {   // D1、D2 各自从 1 降到 g，A 的奇异值跨越约 g² 的范围
            const Matrix u = random_unitary(n, 800 + n);
            const RealVector d = geometric(n, g), di = reciprocal(d);
            const Matrix a = scale_rows(scale_columns(u, as_complex(d)), as_complex(d));
            const Matrix a_inv = scale_rows(scale_columns(adjoint(u), as_complex(di)), as_complex(di));
            const RealVector s = singular_values(a), si = singular_values(a_inv);
            double defect = 0.0, log_sum = 0.0, log_expect = 0.0;
            for (int i = 0; i < n; ++i) {
                defect = std::max(defect, std::abs(s[i] * si[n - 1 - i] - 1.0));
                log_sum += std::log(s[i]);
                log_expect += 2.0 * std::log(d[i]);
            }
            CHECK(is_descending(s) && s.back() > 0.0);
            CHECK_LE(defect, 1e-12);
            CHECK_LE(std::abs(log_sum - log_expect), 1e-11);      // Πσ_i = |det A| = Π d_i²
            // 把行、列的顺序打乱（分级不再“从大到小排好”），奇异值不应改变
            Matrix shuffled(n, n);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) shuffled((i * 7 + 3) % n, (n - 1 - j + 5) % n) = a(i, j);
            const double moved = max_relative_error(singular_values(shuffled), s);
            CHECK_LE(moved, 1e-12);
            say("  n=%2d  D1·U·D2（各降到 %.0e，cond2 = %.2e）：max|σ_i(A)·σ_{n-1-i}(A^{-1}) − 1| = %.2e，打乱后相对变化 %.2e\n",
                n, g, s.front() / s.back(), defect, moved);
        }
    }
    {   // 写死的分级矩阵：A(i,j) = (cr + i·ci)·2^(−12i−9j)，cr = (7i+3j)%5+1，ci = (i+2j)%3−1。
        // 参考值由 80 位十进制的 Jacobi 迭代算出（计算脚本不在本仓库里）。
        const auto build = [](int m, int n) {
            Matrix a(m, n);
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    a(i, j) = cplx(std::ldexp((7 * i + 3 * j) % 5 + 1.0, -12 * i - 9 * j),
                                   std::ldexp((i + 2 * j) % 3 - 1.0, -12 * i - 9 * j));
            return a;
        };
        const RealVector ref5 = {1.41423667974454063302e+0, 4.38320139932015042251e-6, 1.55673318879907657303e-12,
                                 5.79828816490351362211e-19, 2.75342017154279268212e-25};
        const RealVector ref74 = {1.41423667974454063302e+0, 4.38320139932015034621e-6, 1.55673318878802438018e-12,
                                  5.79828811677679473682e-19};
        const Matrix a5 = build(5, 5), a74 = build(7, 4);
        const double e5 = max_relative_error(singular_values(a5), ref5);
        const double e74 = max_relative_error(singular_values(a74), ref74);
        const double e47 = max_relative_error(singular_values(transpose(a74)), ref74);
        Matrix flipped(5, 5);                              // 行、列都倒序：小的在前
        for (int j = 0; j < 5; ++j)
            for (int i = 0; i < 5; ++i) flipped(4 - i, 4 - j) = a5(i, j);
        const double ef = max_relative_error(singular_values(flipped), ref5);
        CHECK_LE(e5, 1e-14);
        CHECK_LE(e74, 1e-14);
        CHECK_LE(e47, 1e-14);
        CHECK_LE(ef, 1e-14);
        CHECK_LE(std::abs(cond2(a5) / (ref5.front() / ref5.back()) - 1.0), 1e-14);
        say("  写死的 5×5（cond2 = 5.1e24）相对误差 %.2e，倒序后 %.2e；7×4 相对误差 %.2e，转置后 %.2e\n", e5, ef, e74, e47);
    }
}

static void test_singular_and_scaling() {
    say("[奇异矩阵、极端量级、非有限输入]\n");
    // σ_min 精确为 0 的情形：cond2 = +inf
    CHECK(cond2(Matrix(4, 4)) == kInf);
    CHECK(cond2(Matrix(5, 2)) == kInf && norm_2(Matrix(5, 2)) == 0.0);
    for (int n : {3, 12, 40}) {
        Matrix zero_col = random_matrix(n, n, 900 + n), zero_row = zero_col;
        for (int i = 0; i < n; ++i) { zero_col(i, n / 2) = 0.0; zero_row(n / 3, i) = 0.0; }
        CHECK(cond2(zero_col) == kInf);
        CHECK(cond2(zero_row) == kInf);
        CHECK(singular_values(zero_col).back() == 0.0 && singular_values(zero_col)[n - 2] > 1e-3);
    }
    {   // 一般的秩亏矩阵：受舍入影响，σ_min 是 eps·σ_max 量级或恰为 0，cond2 至少是 1e15 量级
        Matrix dup = random_matrix(15, 15, 950);
        for (int i = 0; i < 15; ++i) dup(i, 9) = dup(i, 2);                // 两列相同
        CHECK(cond2(dup) > 1e15);
        CHECK(cond2(Matrix(8, 8, cplx(1.0, 0.0))) > 1e15);               // 全 1 矩阵，秩 1
        const Matrix low = random_matrix(20, 3, 951) * random_matrix(3, 20, 952);   // 秩 3
        const RealVector s = singular_values(low);
        CHECK(s[2] > 1e-2 && s[3] < 1e-13 * s[0]);
        CHECK(cond2(low) > 1e15);
        CHECK(cond2(Matrix{{1.0, 2.0}, {2.0, 4.0}}) > 1e15);
    }
    {   // 乘以 2 的幂：奇异值精确地乘以同一个数，不会上溢或下溢
        const Matrix a = random_matrix(9, 6, 960);
        const RealVector s = singular_values(a);
        for (int e : {600, -600, 1000, -1000}) {
            Matrix b = a;
            for (int j = 0; j < 6; ++j)
                for (int i = 0; i < 9; ++i) b(i, j) = cplx(std::ldexp(a(i, j).real(), e), std::ldexp(a(i, j).imag(), e));
            RealVector expect = s;
            for (double& v : expect) v = std::ldexp(v, e);
            CHECK(same_bits(singular_values(b), expect));
            CHECK(cond2(b) == cond2(a));
        }
    }
    {   // 奇异值跨越 200 个数量级（平方会下溢的范围）仍然收敛并保持相对精度
        const int n = 12;
        const Matrix u = random_unitary(n, 970);
        const RealVector s = geometric(n, 1e-200);
        CHECK_LE(max_relative_error(singular_values(scale_columns(u, as_complex(s))), s), 1e-13);
        CHECK_LE(max_relative_error(singular_values(scale_rows(u, as_complex(s))), s), 1e-13);
    }
    for (double smallest : {1e-200, 1e-250}) {
        // 同样的跨度，但行（或列）不是按从大到小排好的：内部给行排序时，1e-154 以下的行也必须排得出先后
        const int n = 12;
        const Matrix u = random_unitary(n, 975);
        const RealVector s = geometric(n, smallest);
        const Matrix rows = scale_rows(u, as_complex(s)), cols = scale_columns(u, as_complex(s));
        Matrix reversed_rows(n, n), shuffled(n, n), reversed_cols(n, n);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                reversed_rows(n - 1 - i, j) = rows(i, j);
                shuffled((i * 5 + 1) % n, (j * 7 + 2) % n) = rows(i, j);
                reversed_cols(i, n - 1 - j) = cols(i, j);
            }
        CHECK_LE(max_relative_error(singular_values(reversed_rows), s), 1e-13);
        CHECK_LE(max_relative_error(singular_values(shuffled), s), 1e-13);
        CHECK_LE(max_relative_error(singular_values(reversed_cols), s), 1e-13);
        CHECK_LE(max_relative_error(singular_values(adjoint(reversed_rows)), s), 1e-13);
    }
    {   // 比 σ_max 小 290 个数量级以上的奇异值按 0 给出（见 svd.hpp 的说明），cond2 随之为 +inf；1e-280 还照常算
        const Matrix below{{1.0, 0.0}, {0.0, 1e-300}}, above{{1.0, 0.0}, {0.0, 1e-280}};
        CHECK(singular_values(below)[1] == 0.0 && cond2(below) == kInf);
        CHECK_LE(std::abs(singular_values(above)[1] / 1e-280 - 1.0), 4 * kEps);
        CHECK_LE(std::abs(cond2(above) / 1e280 - 1.0), 8 * kEps);
    }
    {   // 同时有一整行和一整列为零；秩 1 且分级
        Matrix a = random_matrix(9, 9, 985);
        for (int i = 0; i < 9; ++i) { a(i, 4) = 0.0; a(6, i) = 0.0; }
        const RealVector s = singular_values(a);
        CHECK(s[8] == 0.0 && s[6] > 1e-3 && cond2(a) == kInf);
        Matrix r1(6, 5);
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 5; ++j) r1(i, j) = std::ldexp(1.0, -40 * i - 30 * j);
        const RealVector t = singular_values(r1);
        CHECK(std::abs(t[0] - 1.0) <= 4 * kEps && t[1] <= 1e-15 && cond2(r1) > 1e15);   // σ_1 = |u|·|v| = 1 + O(2^-60)
    }
    {   // NaN / Inf：返回 NaN，不抛异常
        Matrix bad = random_matrix(4, 4, 980);
        bad(1, 2) = cplx(0.0, std::numeric_limits<double>::quiet_NaN());
        const RealVector s = singular_values(bad);
        CHECK(s.size() == 4 && std::isnan(s[0]) && std::isnan(s[3]));
        CHECK(std::isnan(cond2(bad)) && std::isnan(norm_2(bad)));
        bad(1, 2) = cplx(kInf, 0.0);
        CHECK(std::isnan(cond2(bad)));
    }
}

int main() {
    test_basic();
    test_known_singular_values();
    test_repeated_and_ill_conditioned();
    test_unitary();
    test_graded();
    test_singular_and_scaling();
    say("通过 %d / 共 %d\n", g_pass, g_total);
    return g_pass == g_total ? 0 : 1;
}
