// tests/test_eig_lr.cpp —— eig_lr（EISPACK COMHES + COMLR2）的测试
//
// 残差的容差按“忠实的 COMLR2 实际能做到的精度”来定：LR 迭代用的是非酉的消元变换，
// 精度随阶数明显变差（n = 40 时已比 QR 迭代差 5 个量级以上），所以各阶的容差不同，
// 取的是 300 个随机种子上实测最大残差的 3～5 倍（统计程序不在本仓库里）。
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "stablercwa/eig_lr.hpp"

using namespace stablercwa;

// ------------------------------------------------------------ 输出与计数 ----

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

static int g_total = 0, g_failed = 0;

// 断言的计数与报错在这个普通函数里；宏只负责取文件名、行号和表达式的文字，宏里不出现中文。
static void check(bool ok, const char* file, int line, const char* text) {
    ++g_total;
    if (!ok) {
        ++g_failed;
        say("  失败 %s:%d  %s\n", file, line, text);
    }
}
#define CHECK(cond) check(static_cast<bool>(cond), __FILE__, __LINE__, #cond)

// ---------------------------------------------------------------- 工具 ----

// 固定种子的伪随机数（splitmix64），不依赖标准库的分布实现，各平台结果相同。
struct Rng {
    std::uint64_t s;
    explicit Rng(std::uint64_t seed) : s(seed) {}
    std::uint64_t next() {
        std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    double uniform() { return static_cast<double>(next() >> 11) * (2.0 / 9007199254740992.0) - 1.0; }   // [-1, 1)
    cplx cuniform() {
        const double re = uniform();
        return {re, uniform()};
    }
};

static Matrix random_matrix(int n, std::uint64_t seed) {
    Rng rng(seed);
    Matrix a(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) a(i, j) = rng.cuniform();
    return a;
}

static Matrix random_hermitian(int n, std::uint64_t seed) {
    const Matrix g = random_matrix(n, seed);
    Matrix a(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) a(i, j) = 0.5 * (g(i, j) + std::conj(g(j, i)));
    return a;
}

// 首一多项式 Π (x − r_k) 的友矩阵（第一行是系数取负，次对角线全 1），本征值就是各个根。
static Matrix companion(const Vector& roots) {
    const int n = static_cast<int>(roots.size());
    Vector c(n + 1, cplx{});          // c[k] 是 x^k 的系数
    c[0] = 1.0;
    for (int k = 0; k < n; ++k) {
        for (int i = k + 1; i >= 1; --i) c[i] = c[i - 1] - roots[k] * c[i];
        c[0] = -roots[k] * c[0];
    }
    Matrix a(n, n);
    for (int j = 0; j < n; ++j) a(0, j) = -c[n - 1 - j];
    for (int i = 1; i < n; ++i) a(i, i - 1) = 1.0;
    return a;
}

// 二元光栅层的 TE 层矩阵 A = diag(kx²) − Toeplitz(ε_h)，n = 2M+1；ε 为复数时是一般复矩阵（吸收介质）。
static Matrix rcwa_layer(int n, cplx eps_a, cplx eps_b, double fill, double lam_over_period) {
    const int M = (n - 1) / 2;
    Matrix a(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const int h = i - j;
            const cplx e = h == 0 ? fill * eps_a + (1.0 - fill) * eps_b
                                  : (eps_a - eps_b) * (std::sin(kPi * h * fill) / (kPi * h));
            a(i, j) = -e;
        }
        const double kx = std::sin(10.0 * kPi / 180.0) + (i - M) * lam_over_period;
        a(i, i) += kx * kx;
    }
    return a;
}

// max_j ||A v_j − λ_j v_j||_2 / (||A||_F ||v_j||_2)
static double residual(const Matrix& a, const EigResult& r) {
    const int n = a.rows();
    if (r.vectors.rows() != n || r.vectors.cols() != n) return 1e300;   // 没有本征向量（未收敛）：按失败计
    const double na = norm_fro(a);
    double worst = 0.0;
    for (int j = 0; j < n; ++j) {
        const Vector v = r.vectors.column(j);
        Vector av = a * v;
        for (int i = 0; i < n; ++i) av[i] -= r.values[j] * v[i];
        worst = std::max(worst, norm2(av) / (na * norm2(v)));
    }
    return worst;
}

// ||A Z − Z T||_F / (||A||_F ||Z||_F)
static double transform_defect(const Matrix& a, const Matrix& z, const Matrix& t) {
    return norm_fro(a * z - z * t) / (norm_fro(a) * norm_fro(z));
}

// 各列 2-范数与 1 的最大偏差
static double unit_norm_error(const Matrix& v) {
    double worst = 0.0;
    for (int j = 0; j < v.cols(); ++j) worst = std::max(worst, std::abs(norm2(v.column(j)) - 1.0));
    return worst;
}

// 把 expected 里的每个值配给 values 里最近的未用值，返回最大配对距离
static double match_error(const Vector& values, const Vector& expected) {
    if (values.size() != expected.size()) return 1e300;
    std::vector<bool> used(values.size(), false);
    double worst = 0.0;
    for (const cplx& e : expected) {
        int best = -1;
        for (std::size_t k = 0; k < values.size(); ++k)
            if (!used[k] && (best < 0 || std::abs(values[k] - e) < std::abs(values[best] - e))) best = static_cast<int>(k);
        used[best] = true;
        worst = std::max(worst, std::abs(values[best] - e));
    }
    return worst;
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

static bool upper_hessenberg(const Matrix& h) {
    for (int j = 0; j < h.cols(); ++j)
        for (int i = j + 2; i < h.rows(); ++i)
            if (h(i, j) != cplx{}) return false;
    return true;
}
static bool upper_triangular(const Matrix& t) {
    for (int j = 0; j < t.cols(); ++j)
        for (int i = j + 1; i < t.rows(); ++i)
            if (t(i, j) != cplx{}) return false;
    return true;
}
// 单位下三角（不做任何行交换的消元累积出来的 Z 一定是这种形状）
static bool unit_lower_triangular(const Matrix& z) {
    for (int j = 0; j < z.cols(); ++j) {
        if (z(j, j) != cplx{1.0, 0.0}) return false;
        for (int i = 0; i < j; ++i)
            if (z(i, j) != cplx{}) return false;
    }
    return true;
}

template <class F>
static bool throws_invalid(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

// ---------------------------------------------------------------- 测试 ----

static void test_small() {
    say("[小矩阵 n = 0, 1, 2, 3]\n");
    {
        const EigResult r = eig_lr(Matrix());
        CHECK(r.values.empty() && r.info == 0 && r.sweeps == 0);
    }
    {
        const EigResult r = eig_lr(Matrix(1, 1, cplx{3.0, -2.0}));
        CHECK(r.info == 0 && r.sweeps == 0);
        CHECK(r.values.size() == 1 && r.values[0] == cplx(3.0, -2.0));
        CHECK(r.vectors.rows() == 1 && r.vectors.cols() == 1 && r.vectors(0, 0) == cplx(1.0, 0.0));
    }
    {
        const Matrix a{{2.0, 1.0}, {1.0, 2.0}};
        const EigResult r = eig_lr(a);
        CHECK(r.info == 0);
        CHECK(match_error(r.values, {1.0, 3.0}) < 1e-14);
        CHECK(residual(a, r) < 1e-14);
    }
    {
        const Matrix a{{0.0, 1.0}, {-1.0, 0.0}};                   // 实矩阵、一对共轭的纯虚本征值
        const EigResult r = eig_lr(a);
        CHECK(r.info == 0);
        CHECK(match_error(r.values, {kI, -kI}) < 1e-14);
        CHECK(residual(a, r) < 1e-14);
    }
    {
        const Matrix a{{cplx{1.0, 1.0}, 2.0}, {3.0, cplx{4.0, -1.0}}};
        const EigResult r = eig_lr(a);
        CHECK(r.info == 0);
        CHECK(std::abs(r.values[0] + r.values[1] - cplx(5.0, 0.0)) < 1e-14);     // 迹
        CHECK(std::abs(r.values[0] * r.values[1] - cplx(-1.0, 3.0)) < 1e-13);    // 行列式
        CHECK(residual(a, r) < 1e-14);
        CHECK(unit_norm_error(r.vectors) < 1e-14);
    }
    {
        const Matrix a{{2.0, 0.0, 0.0}, {0.0, 3.0, 4.0}, {0.0, 4.0, 9.0}};
        const EigResult r = eig_lr(a);
        CHECK(r.info == 0);
        CHECK(match_error(r.values, {1.0, 2.0, 11.0}) < 1e-13);
        CHECK(residual(a, r) < 1e-14);
    }
    {
        const Matrix a{{cplx{1.0, 2.0}, cplx{0.0, 1.0}, 3.0}, {cplx{2.0, -1.0}, 0.5, cplx{1.0, 1.0}}, {4.0, cplx{0.0, -2.0}, cplx{-1.0, 0.5}}};
        const EigResult r = eig_lr(a);
        CHECK(r.info == 0);
        CHECK(std::abs(r.values[0] + r.values[1] + r.values[2] - cplx(0.5, 2.5)) < 1e-13);
        CHECK(residual(a, r) < 1e-14);
        CHECK(unit_norm_error(r.vectors) < 1e-14);
    }
}

static void test_diagonal_triangular() {
    say("[对角与三角矩阵]\n");
    {
        const Vector d{cplx{1.0, 2.0}, -3.0, cplx{0.0, 0.5}, 4.0, cplx{-2.0, -2.0}};
        const EigResult r = eig_lr(Matrix::diagonal(d));
        CHECK(r.info == 0 && r.sweeps == 0);
        CHECK(same_bits(r.values, d));                              // 不迭代，对角元原样给出
        CHECK(same_bits(r.vectors, Matrix::identity(5)));
    }
    {
        Matrix a = random_matrix(6, 101);
        for (int j = 0; j < 6; ++j)
            for (int i = j + 1; i < 6; ++i) a(i, j) = 0.0;
        const EigResult r = eig_lr(a);
        CHECK(r.info == 0 && r.sweeps == 0);
        CHECK(same_bits(r.values, a.diag()));
        CHECK(residual(a, r) < 1e-14);
        CHECK(unit_norm_error(r.vectors) < 1e-14);
    }
    {
        Matrix a = random_matrix(6, 102);
        for (int j = 0; j < 6; ++j)
            for (int i = 0; i < j; ++i) a(i, j) = 0.0;
        const EigResult r = eig_lr(a);
        CHECK(r.info == 0);
        CHECK(match_error(r.values, a.diag()) < 1e-13);
        CHECK(residual(a, r) < 1e-13);
    }
}

static void test_companion() {
    say("[友矩阵：根已知]\n");
    {
        const Vector roots{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        const Matrix a = companion(roots);
        const EigResult r = eig_lr(a);
        const double err = match_error(r.values, roots);
        say("  根 1…6：最大误差 %.2e，残差 %.2e\n", err, residual(a, r));
        CHECK(r.info == 0);
        CHECK(err < 1e-11);                 // 友矩阵的本征值本身就病态，达不到机器精度
        CHECK(residual(a, r) < 1e-14);
    }
    {
        const Vector roots{cplx{1.0, 1.0}, cplx{1.0, -1.0}, -2.0, cplx{0.0, 0.5}, 3.0};
        const Matrix a = companion(roots);
        const EigResult r = eig_lr(a);
        const double err = match_error(r.values, roots);
        say("  复根 5 个：最大误差 %.2e，残差 %.2e\n", err, residual(a, r));
        CHECK(r.info == 0);
        CHECK(err < 1e-13);
        CHECK(residual(a, r) < 1e-14);
    }
    {
        // x^5 − 1：友矩阵是循环置换阵，本征值是 5 个单位根。对角元全为零，LR 分解每一行都必须交换；
        // 而且带交换的 LR 步把它原样变回自己，前 10 次迭代毫无进展，要靠第 10 次的例外位移才能脱身。
        // （阶数再高一些时，第一个本征值在 30 次内收敛不了，这是 COMLR2 本来的表现。）
        const int n = 5;
        Vector roots(n);
        for (int k = 0; k < n; ++k) roots[k] = std::polar(1.0, 2.0 * kPi * k / n);
        Matrix a(n, n);
        a(0, n - 1) = 1.0;
        for (int i = 1; i < n; ++i) a(i, i - 1) = 1.0;
        const EigResult r = eig_lr(a);
        const double err = match_error(r.values, roots);
        say("  x^5 − 1：最大误差 %.2e，残差 %.2e，迭代 %d 次\n", err, residual(a, r), r.sweeps);
        CHECK(r.info == 0);
        CHECK(r.sweeps > 10);
        CHECK(err < 1e-13);
        CHECK(residual(a, r) < 1e-13);

        // 同一个矩阵乘以 i：次对角元成了纯虚数，例外位移的实部是 0，全靠虚部
        // （si = |Im h(en,en-1)| + |Im h(en-1,en-2)|）；漏掉虚部的话位移为零，永远不收敛。
        const Matrix b = kI * a;
        for (cplx& root : roots) root *= kI;
        const EigResult rb = eig_lr(b);
        const double errb = rb.info == 0 ? match_error(rb.values, roots) : 1e300;
        say("  i·(x^5 − 1 的友矩阵)：info = %d，最大误差 %.2e，迭代 %d 次\n", rb.info, errb, rb.sweeps);
        CHECK(rb.info == 0);
        CHECK(rb.sweeps > 10);
        CHECK(errb < 1e-13);
    }
}

static void test_random_general() {
    say("[一般复矩阵 n = 5, 10, 20, 40]\n");
    const int sizes[] = {5, 10, 20, 40};
    const double tol[] = {2e-13, 5e-13, 5e-12, 1e-9};
    for (int c = 0; c < 4; ++c) {
        const int n = sizes[c];
        double worst = 0.0;
        for (int seed = 1; seed <= 6; ++seed) {
            const Matrix a = random_matrix(n, 1000 * n + seed);
            const EigResult r = eig_lr(a);
            CHECK(r.info == 0);
            if (r.info != 0) continue;
            const double res = residual(a, r);
            worst = std::max(worst, res);
            CHECK(res < tol[c]);
            CHECK(unit_norm_error(r.vectors) < 1e-14);
            cplx trace{}, sum{};
            for (int i = 0; i < n; ++i) {
                trace += a(i, i);
                sum += r.values[i];
            }
            CHECK(std::abs(trace - sum) < 1e-12 * n);
        }
        say("  n = %2d：6 个矩阵的最大残差 %.2e（容差 %.0e）\n", n, worst, tol[c]);
    }
}

static void test_hermitian() {
    say("[厄米矩阵 n = 4, 8, 16]\n");
    const int sizes[] = {4, 8, 16};
    const double tol[] = {5e-13, 3e-12, 3e-11};
    for (int c = 0; c < 3; ++c) {
        const int n = sizes[c];
        double worst = 0.0, worst_imag = 0.0;
        for (int seed = 1; seed <= 6; ++seed) {
            const Matrix a = random_hermitian(n, 1000 * n + seed);
            const EigResult r = eig_lr(a);
            CHECK(r.info == 0);
            if (r.info != 0) continue;
            const double res = residual(a, r);
            worst = std::max(worst, res);
            CHECK(res < tol[c]);
            for (const cplx& v : r.values) worst_imag = std::max(worst_imag, std::abs(v.imag()));
        }
        CHECK(worst_imag < 1e-12 * n);          // 本征值应为实数
        say("  n = %2d：最大残差 %.2e（容差 %.0e），本征值虚部最大 %.1e\n", n, worst, tol[c], worst_imag);
    }
}

static void test_absorbing_layer() {
    say("[吸收介质的 RCWA 层矩阵：一般复矩阵]\n");
    {
        const cplx index{3.9, 0.02};                                 // 弱吸收的高折射率材料
        const Matrix a = rcwa_layer(21, index * index, 1.0, 0.5, 0.45);
        CHECK(!is_hermitian(a, 1e-12));
        const EigResult r = eig_lr(a);
        const double res = residual(a, r);
        say("  n = 21，n = 3.9 + 0.02i：残差 %.2e\n", res);
        CHECK(r.info == 0);
        CHECK(res < 2e-12);
        double max_imag = 0.0;
        for (const cplx& v : r.values) max_imag = std::max(max_imag, std::abs(v.imag()));
        CHECK(max_imag > 1e-3);                                       // 有吸收：本征值是复数
    }
    {
        const cplx index{0.2, 3.3};                                  // 金属
        const Matrix a = rcwa_layer(41, index * index, 1.0, 0.3, 0.7);
        const EigResult r = eig_lr(a);
        const double res = residual(a, r);
        say("  n = 41，n = 0.2 + 3.3i：残差 %.2e\n", res);
        CHECK(r.info == 0);
        CHECK(res < 2e-11);
    }
}

static void test_interchanges() {
    say("[必须交换行的矩阵]\n");
    {
        // 第 0 列对角线以下是 (0, 8, 3)：化 Hessenberg 形的第一步必须把第 2 行换上来作主元。
        const Matrix a{{1.0, 2.0, 3.0, 4.0}, {0.0, 5.0, 6.0, 7.0}, {8.0, 9.0, 1.0, 2.0}, {3.0, 4.0, 5.0, 6.0}};
        Matrix z0, h0;
        EigOptions opt;
        opt.observer = [&](const EigProgress& p) {
            if (std::string(p.stage) == "hessenberg") {
                z0 = p.Z;
                h0 = p.H;
            }
        };
        const EigResult r = eig_lr(a, opt);
        CHECK(r.info == 0);
        CHECK(z0.rows() == 4 && !unit_lower_triangular(z0));         // 发生过行交换
        CHECK(upper_hessenberg(h0));
        CHECK(transform_defect(a, z0, h0) < 1e-15);
        double max_entry = 0.0;                                       // Z 的元素是乘子、0 或 1；实矩阵时选主元保证乘子的绝对值 <= 1
        for (int j = 0; j < 4; ++j)
            for (int i = 0; i < 4; ++i) max_entry = std::max(max_entry, abs1(z0(i, j)));
        CHECK(max_entry <= 1.0);
        CHECK(residual(a, r) < 1e-14);
    }
    {
        const Matrix a{{0.0, 1.0}, {1.0, 0.0}};
        const EigResult r = eig_lr(a);
        CHECK(r.info == 0);
        CHECK(match_error(r.values, {1.0, -1.0}) < 1e-15);
    }
    {
        // 循环置换阵已经是 Hessenberg 形，化简阶段不动它（Z = I）；之后 Z 只在 LR 步里变。
        // 不交换行的 LR 步只会给出单位下三角的 Z，所以 Z 不是单位下三角就说明 LR 步里交换过。
        const int n = 5;
        Matrix a(n, n);
        a(0, n - 1) = 1.0;
        for (int i = 1; i < n; ++i) a(i, i - 1) = 1.0;
        EigOptions opt;
        opt.keep_transform = true;
        const EigResult r = eig_lr(a, opt);
        CHECK(r.info == 0);
        CHECK(!unit_lower_triangular(r.Z));
        CHECK(transform_defect(a, r.Z, r.T) < 1e-14);
        CHECK(residual(a, r) < 1e-13);
    }
}

static void test_repeated() {
    say("[重本征值]\n");
    {
        const EigResult r = eig_lr(Matrix::identity(4));
        CHECK(r.info == 0);
        CHECK(same_bits(r.values, Vector(4, cplx{1.0, 0.0})));
        CHECK(same_bits(r.vectors, Matrix::identity(4)));
    }
    {
        const EigResult r = eig_lr(Matrix(3, 3));                    // 零矩阵：不回代，直接给出 Z 的各列
        CHECK(r.info == 0);
        CHECK(same_bits(r.values, Vector(3, cplx{})));
        CHECK(same_bits(r.vectors, Matrix::identity(3)));
    }
    {
        // 上双对角的 Jordan 块：只有一个本征向量，回代里每一步分母都是零，换成 eps·norm 后不应出现 NaN/Inf
        const int n = 5;
        Matrix a(n, n);
        for (int i = 0; i < n; ++i) {
            a(i, i) = 2.0;
            if (i + 1 < n) a(i, i + 1) = 1.0;
        }
        const EigResult r = eig_lr(a);
        CHECK(r.info == 0);
        CHECK(same_bits(r.values, Vector(n, cplx{2.0, 0.0})));
        CHECK(r.vectors.all_finite());
        CHECK(unit_norm_error(r.vectors) < 1e-14);
        CHECK(residual(a, r) < 1e-14);
    }
    {
        // 下双对角的 Jordan 块：要真的迭代。亏损本征值对扰动的敏感度是 eps^(1/5) ≈ 7e-4
        const int n = 5;
        Matrix a(n, n);
        for (int i = 0; i < n; ++i) {
            a(i, i) = 2.0;
            if (i > 0) a(i, i - 1) = 1.0;
        }
        const EigResult r = eig_lr(a);
        CHECK(r.info == 0);
        CHECK(r.vectors.all_finite());
        CHECK(match_error(r.values, Vector(n, cplx{2.0, 0.0})) < 1e-2);
        CHECK(residual(a, r) < 1e-13);
    }
    {
        // 可对角化、但本征值有重复：A = S diag(1,1,2,2,2,3) S^{-1}，S 取单位下三角（逆也是整数阵）
        const int n = 6;
        const double d[] = {1.0, 1.0, 2.0, 2.0, 2.0, 3.0};
        Matrix s = Matrix::identity(n), sinv = Matrix::identity(n);
        for (int i = 1; i < n; ++i) {
            s(i, i - 1) = 1.0;                                        // S = I + N
            for (int j = 0; j < i; ++j) sinv(i, j) = ((i - j) % 2 == 0) ? 1.0 : -1.0;   // S^{-1} = I − N + N² − …
        }
        Vector dv(n);
        for (int i = 0; i < n; ++i) dv[i] = d[i];
        const Matrix a = scale_columns(s, dv) * sinv;
        const EigResult r = eig_lr(a);
        CHECK(r.info == 0);
        CHECK(r.vectors.all_finite());
        CHECK(match_error(r.values, dv) < 1e-12);
        CHECK(residual(a, r) < 1e-13);
        CHECK(unit_norm_error(r.vectors) < 1e-14);
    }
}

static void test_observer() {
    say("[观察者回调]\n");
    const int n = 12;
    const Matrix a = random_matrix(n, 777);
    const EigResult plain = eig_lr(a);

    for (int with_vectors = 1; with_vectors >= 0; --with_vectors) {
        int hess = 0, iter = 0, done = 0, calls = 0;
        bool order_ok = true, shape_ok = true, hess_ok = false, done_ok = false;
        double worst = 0.0;
        Vector done_diag;
        EigOptions opt;
        opt.vectors = with_vectors != 0;
        opt.observer = [&](const EigProgress& p) {
            const std::string stage = p.stage;
            ++calls;
            if (stage == "hessenberg") {
                ++hess;
                order_ok = order_ok && calls == 1 && p.sweep == 0 && p.active == n - 1;
                hess_ok = upper_hessenberg(p.H) && p.shift == cplx{};
            } else if (stage == "iteration") {
                ++iter;
                order_ok = order_ok && hess == 1 && done == 0 && p.sweep == iter && p.active >= 1 && p.active < n;
            } else if (stage == "done") {
                ++done;
                order_ok = order_ok && p.active == -1 && p.sweep == iter;
                done_ok = upper_triangular(p.H);
                for (int i = 0; i < n; ++i) done_diag.push_back(p.H(i, i) + p.shift);
            } else {
                order_ok = false;
            }
            shape_ok = shape_ok && p.H.rows() == n && p.H.cols() == n && p.Z.rows() == n && p.Z.cols() == n;
            if (p.Z.rows() != n) return;
            Matrix m = p.H;                                           // Z^{-1} A Z = H + shift·I
            for (int i = 0; i < n; ++i) m(i, i) += p.shift;
            worst = std::max(worst, transform_defect(a, p.Z, m));
        };
        const EigResult r = eig_lr(a, opt);
        say("  vectors=%d：hessenberg %d 次，iteration %d 次，done %d 次；快照 ||AZ − Z(H+tI)|| 相对值最大 %.1e\n",
            with_vectors, hess, iter, done, worst);
        CHECK(r.info == 0);
        CHECK(hess == 1 && done == 1);
        CHECK(iter == r.sweeps && r.sweeps > 0);
        CHECK(order_ok && shape_ok && hess_ok && done_ok);
        CHECK(worst < 1e-13);
        CHECK(match_error(done_diag, r.values) < 1e-13);
        CHECK(same_bits(r.values, plain.values));                     // 回调不改变计算
        if (with_vectors) CHECK(same_bits(r.vectors, plain.vectors));
        else CHECK(r.vectors.empty());
        CHECK(r.sweeps == plain.sweeps);
    }
}

static void test_keep_transform() {
    say("[keep_transform：Z^{-1} A Z = T]\n");
    const int sizes[] = {3, 10, 20, 40};
    const double tol[] = {1e-14, 3e-14, 1e-13, 1e-12};
    for (int c = 0; c < 4; ++c) {
        const int n = sizes[c];
        const Matrix a = random_matrix(n, 500 + n);
        EigOptions opt;
        opt.keep_transform = true;
        const EigResult r = eig_lr(a, opt);
        CHECK(r.info == 0);
        CHECK(r.Z.rows() == n && r.Z.cols() == n && r.T.rows() == n && r.T.cols() == n);
        CHECK(upper_triangular(r.T));
        CHECK(same_bits(r.T.diag(), r.values));
        const double defect = transform_defect(a, r.Z, r.T);
        const double non_unitary = norm_fro(adjoint(r.Z) * r.Z - Matrix::identity(n));
        say("  n = %2d：||AZ − ZT||/(||A|| ||Z||) = %.1e，||Z^H Z − I|| = %.1e\n", n, defect, non_unitary);
        CHECK(defect < tol[c]);
        CHECK(non_unitary > 0.1);                                     // Z 不是酉矩阵
        CHECK(same_bits(r.values, eig_lr(a).values));
        CHECK(same_bits(r.vectors, eig_lr(a).vectors));

        opt.vectors = false;                                          // 只要 Z、T，不要本征向量
        const EigResult r2 = eig_lr(a, opt);
        CHECK(r2.vectors.empty());
        CHECK(same_bits(r2.Z, r.Z) && same_bits(r2.T, r.T) && same_bits(r2.values, r.values));
    }
    {
        const EigResult r = eig_lr(random_matrix(8, 31));             // 默认不保留
        CHECK(r.Z.empty() && r.T.empty());
    }
}

static void test_values_only() {
    say("[vectors = false：本征值逐位相同]\n");
    std::vector<Matrix> mats;
    mats.push_back(random_matrix(20, 61));
    mats.push_back(random_matrix(40, 62));
    mats.push_back(random_hermitian(16, 63));
    mats.push_back(rcwa_layer(21, cplx{3.9, 0.02} * cplx{3.9, 0.02}, 1.0, 0.5, 0.45));
    mats.push_back(rcwa_layer(41, 2.25, 1.0, 0.4, 0.6));              // 大 kx 处近乎解耦，会出现中间的可忽略次对角元
    mats.push_back(companion({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));
    for (const Matrix& a : mats) {
        const EigResult full = eig_lr(a);
        EigOptions opt;
        opt.vectors = false;
        const EigResult r = eig_lr(a, opt);
        CHECK(r.info == 0 && full.info == 0);
        CHECK(r.vectors.empty() && r.Z.empty() && r.T.empty());
        CHECK(r.sweeps == full.sweeps);
        CHECK(same_bits(r.values, full.values));
    }
}

static void test_not_converged() {
    say("[max_iter 很小：info > 0]\n");
    const int n = 12;
    const Matrix a = random_matrix(n, 909);
    const EigResult ref = eig_lr(a);
    CHECK(ref.info == 0);
    {
        EigOptions opt;
        opt.max_iter = 1;
        const EigResult r = eig_lr(a, opt);
        say("  max_iter = 1：info = %d，迭代 %d 次\n", r.info, r.sweeps);
        CHECK(r.info > 0 && r.info <= n);
        CHECK(r.vectors.empty());
        CHECK(r.values.size() == static_cast<std::size_t>(n));
    }
    bool partial = false;                                             // 至少有一个上限让它停在半路
    for (int limit : {3, 5, 6, 7}) {
        // 失败之前走的路与不设上限时完全相同，所以已收敛的本征值（下标 >= info）应逐位相同
        EigOptions opt;
        opt.max_iter = limit;
        opt.keep_transform = true;
        int done = 0;
        opt.observer = [&](const EigProgress& p) { done += std::string(p.stage) == "done"; };
        const EigResult r = eig_lr(a, opt);
        say("  max_iter = %d：info = %d，迭代 %d 次\n", limit, r.info, r.sweeps);
        bool tail_same = true;
        for (int i = r.info; i < n; ++i) tail_same = tail_same && r.values[i] == ref.values[i];
        CHECK(tail_same);
        CHECK(r.Z.rows() == n && r.T.rows() == n);
        CHECK(transform_defect(a, r.Z, r.T) < 1e-13);                 // 失败时 Z、T 仍是相似变换
        CHECK(upper_hessenberg(r.T));
        partial = partial || (r.info > 0 && r.info < n);
        if (r.info > 0) {
            CHECK(r.vectors.empty());
            CHECK(done == 0);
            bool tri = true;                                          // 已收敛的那几行已经是三角形
            for (int i = r.info; i < n; ++i) tri = tri && std::abs(r.T(i, i - 1)) <= 1e-14 * norm_fro(a);
            CHECK(tri);
        } else {
            CHECK(done == 1);
        }
    }
    CHECK(partial);
    {
        // 循环置换阵：前 10 次迭代原地不动（全是 0/1 的精确运算），上限设为 10 必然失败在最后一个本征值上
        const int m = 6;
        Matrix c(m, m);
        c(0, m - 1) = 1.0;
        for (int i = 1; i < m; ++i) c(i, i - 1) = 1.0;
        EigOptions opt;
        opt.max_iter = 10;
        opt.keep_transform = true;
        const EigResult r = eig_lr(c, opt);
        CHECK(r.info == m && r.sweeps == 10);
        CHECK(r.vectors.empty());
        CHECK(same_bits(r.T, c));                                     // 矩阵原样未变
        CHECK(transform_defect(c, r.Z, r.T) == 0.0);                  // Z 是置换阵
        opt.max_iter = 60;
        const EigResult ok = eig_lr(c, opt);
        CHECK(ok.info == 0 && ok.sweeps > 10);
        CHECK(residual(c, ok) < 1e-13);
    }
}

static void test_options_and_errors() {
    say("[参数]\n");
    const Matrix a = random_matrix(10, 2024);
    {
        EigOptions loose;
        loose.eps = 1e-8;                                             // 放宽收敛阈值：迭代次数不增，精度相应变差
        const EigResult r = eig_lr(a, loose), ref = eig_lr(a);
        CHECK(r.info == 0);
        CHECK(r.sweeps <= ref.sweeps);
        CHECK(match_error(r.values, ref.values) < 1e-6);
        CHECK(residual(a, r) < 1e-6);
    }
    CHECK(throws_invalid([&] { eig_lr(Matrix(3, 4)); }));
    CHECK(throws_invalid([&] {
        EigOptions o;
        o.max_iter = 0;
        eig_lr(a, o);
    }));
    CHECK(throws_invalid([&] {
        EigOptions o;
        o.eps = -1.0;
        eig_lr(a, o);
    }));
    CHECK(throws_invalid([&] {
        Matrix b = a;
        b(2, 3) = cplx{std::nan(""), 0.0};
        eig_lr(b);
    }));
    {
        const Matrix copy = a;
        eig_lr(a);
        CHECK(same_bits(a, copy));                                    // 输入不被修改
    }
}

static void test_threads() {
    say("[线程数不影响结果]\n");
    const Matrix a = random_matrix(80, 4321);
    const EigResult r1 = eig_lr(a);
    set_num_threads(4);
    const EigResult r4 = eig_lr(a);
    set_num_threads(1);
    CHECK(r1.info == 0 && r4.info == 0);
    CHECK(same_bits(r1.values, r4.values));
    CHECK(same_bits(r1.vectors, r4.vectors));
    CHECK(unit_norm_error(r1.vectors) < 1e-14);
}


// 选主元比较的是 |Re| + |Im|（EISPACK 的写法），不是模；两者给出不同结论的矩阵可以把这一点钉死。
// 用模来比会少一些“模大于 1 的乘子”，等于悄悄给 LR 方法加了稳定化，所以这里专门检查。
static void test_pivot_rule() {
    say("[选主元规则：比较 |Re| + |Im|]\n");
    const cplx mult{0.75, -0.75};                                     // 1.5 / (1 + i)，模约 1.06
    {
        // 化 Hessenberg 形，第 0 列对角线以下是 (1+i, 1.5)：|Re|+|Im| 为 2 和 1.5（不交换），模为 1.41 和 1.5（会交换）。
        const Matrix a{{1.0, 2.0, 3.0}, {cplx{1.0, 1.0}, 4.0, 5.0}, {1.5, 6.0, 7.0}};
        Matrix z0, h0;
        EigOptions opt;
        opt.observer = [&](const EigProgress& p) {
            if (std::string(p.stage) == "hessenberg") {
                z0 = p.Z;
                h0 = p.H;
            }
        };
        const EigResult r = eig_lr(a, opt);
        CHECK(r.info == 0);
        CHECK(z0.rows() == 3 && unit_lower_triangular(z0));           // 没有交换
        CHECK(z0.rows() == 3 && std::abs(z0(2, 1) - mult) < 1e-15);
        CHECK(z0.rows() == 3 && std::abs(z0(2, 1)) > 1.0);            // 乘子的模可以超过 1（上限是 √2）
        CHECK(upper_hessenberg(h0) && transform_defect(a, z0, h0) < 1e-15);
        CHECK(residual(a, r) < 1e-14);
    }
    {
        // 上下两个元素对调：|Re|+|Im| 为 1.5 和 2（交换），模为 1.5 和 1.41（不会交换）。
        const Matrix a{{1.0, 2.0, 3.0}, {1.5, 4.0, 5.0}, {cplx{1.0, 1.0}, 6.0, 7.0}};
        Matrix z0, h0;
        EigOptions opt;
        opt.observer = [&](const EigProgress& p) {
            if (std::string(p.stage) == "hessenberg") {
                z0 = p.Z;
                h0 = p.H;
            }
        };
        const EigResult r = eig_lr(a, opt);
        CHECK(r.info == 0);
        CHECK(z0.rows() == 3 && !unit_lower_triangular(z0));          // 交换了第 1、2 行
        CHECK(z0.rows() == 3 && std::abs(z0(1, 1) - mult) < 1e-15 && z0(1, 2) == cplx(1.0, 0.0) &&
              z0(2, 1) == cplx(1.0, 0.0) && z0(2, 2) == cplx{});
        CHECK(upper_hessenberg(h0) && transform_defect(a, z0, h0) < 1e-15);
        CHECK(residual(a, r) < 1e-14);
    }
    // LR 步。下面两个 2×2 矩阵都有本征值 0 和另一个远离 h(1,1) 的本征值，第一次迭代的位移就是 0（差一个舍入），
    // 于是分解时比较的正是 h(0,0) 与 h(1,0)。Z 在这一步之后的形状说明有没有交换。
    {
        // h(0,0) = 1+i，h(1,0) = 1.6：|Re|+|Im| 为 2 和 1.6（不交换），模为 1.41 和 1.6（会交换）。
        const Matrix a{{cplx{1.0, 1.0}, cplx{0.0625, 0.0625}}, {1.6, 0.1}};
        Matrix z1;
        EigOptions opt;
        opt.observer = [&](const EigProgress& p) {
            if (std::string(p.stage) == "iteration" && p.sweep == 1) z1 = p.Z;
        };
        const EigResult r = eig_lr(a, opt);
        CHECK(r.info == 0);
        CHECK(z1.rows() == 2 && unit_lower_triangular(z1));
        CHECK(z1.rows() == 2 && std::abs(z1(1, 0) - cplx(0.8, -0.8)) < 1e-14);   // 乘子 1.6 / (1+i)，模约 1.13
        CHECK(match_error(r.values, {0.0, cplx{1.1, 1.0}}) < 1e-15);
        CHECK(residual(a, r) < 1e-15);
    }
    {
        // h(0,0) = 1.5，h(1,0) = 1+i：|Re|+|Im| 为 1.5 和 2（交换），模为 1.5 和 1.41（不会交换）。
        const Matrix a{{1.5, cplx{0.075, -0.075}}, {cplx{1.0, 1.0}, 0.1}};
        Matrix z1;
        EigOptions opt;
        opt.observer = [&](const EigProgress& p) {
            if (std::string(p.stage) == "iteration" && p.sweep == 1) z1 = p.Z;
        };
        const EigResult r = eig_lr(a, opt);
        CHECK(r.info == 0);
        CHECK(z1.rows() == 2 && std::abs(z1(0, 0) - mult) < 1e-14 && z1(0, 1) == cplx(1.0, 0.0) &&
              z1(1, 0) == cplx(1.0, 0.0) && z1(1, 1) == cplx{});      // 交换过：Z = [[乘子, 1], [1, 0]]
        CHECK(match_error(r.values, {0.0, 1.6}) < 1e-15);
        CHECK(residual(a, r) < 1e-15);
    }
}

// 另外三处容易被“改良”掉的 EISPACK 细节，各用一个结论确定的小矩阵钉住。
static void test_eispack_details() {
    say("[收敛判据、分解起始行、回代的溢出控制]\n");
    {
        // 收敛判据比较的也是 |Re| + |Im|：次对角元 7.5e-16，对角元 ±(1+i)。
        // eps·(|Re|+|Im| 之和) = 8.9e-16（可忽略，不迭代）；eps·(模之和) = 6.3e-16（按模比就得迭代）。
        const Matrix a{{cplx{1.0, 1.0}, 0.5}, {7.5e-16, cplx{-1.0, -1.0}}};
        const EigResult r = eig_lr(a);
        CHECK(r.info == 0 && r.sweeps == 0);
        CHECK(same_bits(r.values, a.diag()));
    }
    {
        // 两个相邻的小次对角元：h(1,0) = h(2,1) = 1e-9 单独都不可忽略，乘积却可忽略，
        // 所以第一次迭代应当从第 1 行开始分解，第 0 行、第 0 列（H 与 Z 的）原样不动。
        const Matrix a{{1.0, 0.3, 0.2}, {1e-9, 2.0, 0.4}, {0.0, 1e-9, 3.0}};
        Matrix h0, h1, z1;
        cplx shift1{};
        EigOptions opt;
        opt.observer = [&](const EigProgress& p) {
            if (std::string(p.stage) == "hessenberg") h0 = p.H;
            if (std::string(p.stage) == "iteration" && p.sweep == 1) {
                h1 = p.H;
                z1 = p.Z;
                shift1 = p.shift;
            }
        };
        const EigResult r = eig_lr(a, opt);
        CHECK(r.info == 0 && r.sweeps >= 1);
        CHECK(same_bits(h0, a));                                      // 已经是 Hessenberg 形，化简阶段不动它
        CHECK(z1.rows() == 3 && z1(0, 0) == cplx(1.0, 0.0) && z1(1, 0) == cplx{} && z1(2, 0) == cplx{});
        CHECK(h1.rows() == 3 && h1(1, 0) == cplx(1e-9, 0.0) && h1(0, 0) == cplx(1.0, 0.0) - shift1);
        CHECK(match_error(r.values, {1.0, 2.0, 3.0}) < 1e-8);         // 本征值离对角元约 1e-9（耦合的一阶修正）
        CHECK(residual(a, r) < 1e-14);
    }
    {
        // 回代的溢出控制：上三角阵，对角元相差 1e-120，回代每一步都要除以这么小的数。
        // 分量一超过约 1e8 就把已求出的部分整体缩小，否则三步之后就上溢。正确的结果是各列都几乎等于 e_0。
        const int n = 4;
        Matrix a(n, n);
        for (int i = 0; i < n; ++i) {
            a(i, i) = i * 1e-120;
            if (i + 1 < n) a(i, i + 1) = 1.0;
        }
        const EigResult r = eig_lr(a);
        CHECK(r.info == 0 && r.sweeps == 0);
        CHECK(same_bits(r.values, a.diag()));
        CHECK(r.vectors.all_finite());
        CHECK(unit_norm_error(r.vectors) < 1e-14);
        bool along_e0 = r.vectors.rows() == n;
        for (int j = 0; along_e0 && j < n; ++j) along_e0 = std::abs(std::abs(r.vectors(0, j)) - 1.0) < 1e-14;
        CHECK(along_e0);
        CHECK(residual(a, r) < 1e-14);
    }
}

// 第 0 个本征值为零：COMLR2 的原文在第 0 个本征向量上多乘了一个标量，这里直接取 Z 的第 0 列，
// 所以即便那个标量是零（或很小）也能得到归一的向量。
static void test_zero_first_eigenvalue() {
    say("[第 0 个本征值为零]\n");
    const Matrix a{{0.0, 0.0, 0.0}, {0.0, 3.0, 4.0}, {0.0, 4.0, 9.0}};
    const EigResult r = eig_lr(a);
    CHECK(r.info == 0);
    CHECK(std::abs(r.values[0]) < 1e-14);
    CHECK(match_error(r.values, {0.0, 1.0, 11.0}) < 1e-13);
    CHECK(unit_norm_error(r.vectors) < 1e-14);
    CHECK(residual(a, r) < 1e-14);
}

// 输入有限、但运算中会上溢的矩阵：不做缩放（与 EISPACK 相同），位移公式里的乘积溢出成 NaN，
// 之后永远不满足收敛判据。要求：一定在 max_iter 次内停下，用 info > 0 报告，不给本征向量。
static void test_extreme_scaling() {
    say("[极端量级：一定终止，失败时有标志]\n");
    const Matrix base = random_matrix(6, 4248);
    const EigResult ref = eig_lr(base);
    CHECK(ref.info == 0);
    for (double scale : {1e-200, 1e150}) {                            // 乘积不溢出的范围内，结果与量级无关
        Matrix a = base;
        a *= cplx{scale, 0.0};
        const EigResult r = eig_lr(a);
        CHECK(r.info == 0);
        CHECK(r.info == 0 && residual(a, r) < 1e-13);
        CHECK(r.info == 0 && r.vectors.all_finite());
    }
    for (double scale : {1e200, 1e300}) {
        Matrix a = base;
        a *= cplx{scale, 0.0};
        CHECK(a.all_finite());
        const EigResult r = eig_lr(a);
        say("  元素量级 %.0e：info = %d，迭代 %d 次\n", scale, r.info, r.sweeps);
        CHECK(r.info > 0 && r.info <= 6);
        CHECK(r.vectors.empty());
        CHECK(r.sweeps <= 30 * 6);
    }
}

// 可重入：eig_lr 没有共享的可变状态。几个线程同时调用、在 parallel_for 里嵌套调用（n >= 64 时
// 回代那一段自己也会用 parallel_ranges），结果都应与单线程逐位相同。
static void test_reentrant() {
    say("[多线程同时调用、嵌套在 parallel_for 里调用]\n");
    const int count = 6;
    std::vector<Matrix> mats;
    for (int k = 0; k < count; ++k) mats.push_back(random_matrix(k % 2 == 0 ? 70 : 30, 8800 + k));
    std::vector<EigResult> ref(count);
    for (int k = 0; k < count; ++k) ref[k] = eig_lr(mats[k]);

    const auto same_as_ref = [&](const std::vector<EigResult>& out) {
        bool ok = true;
        for (int k = 0; k < count; ++k)
            ok = ok && out[k].info == 0 && out[k].sweeps == ref[k].sweeps && same_bits(out[k].values, ref[k].values) &&
                 same_bits(out[k].vectors, ref[k].vectors);
        return ok;
    };
    for (int pool : {1, 3}) {
        set_num_threads(pool);
        {
            std::vector<EigResult> out(count);
            std::vector<std::thread> workers;
            for (int k = 0; k < count; ++k) workers.emplace_back([&, k] { out[k] = eig_lr(mats[k]); });
            for (std::thread& w : workers) w.join();
            CHECK(same_as_ref(out));
        }
        {
            std::vector<EigResult> out(count);
            parallel_for(0, count, [&](int k) { out[k] = eig_lr(mats[k]); });
            CHECK(same_as_ref(out));
        }
    }
    set_num_threads(1);
}

int main() {
    test_small();
    test_diagonal_triangular();
    test_companion();
    test_random_general();
    test_hermitian();
    test_absorbing_layer();
    test_interchanges();
    test_repeated();
    test_observer();
    test_keep_transform();
    test_values_only();
    test_not_converged();
    test_options_and_errors();
    test_threads();
    test_pivot_rule();
    test_eispack_details();
    test_zero_first_eigenvalue();
    test_extreme_scaling();
    test_reentrant();
    say("通过 %d / 共 %d\n", g_total - g_failed, g_total);
    return g_failed == 0 ? 0 : 1;
}
