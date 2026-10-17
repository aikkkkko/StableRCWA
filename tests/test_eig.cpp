// tests/test_eig.cpp —— eig.hpp 的测试：统一入口 eig()、本征对残差 eig_residual()、正交性 orthogonality_defect()
//
//   c++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -I include tests/test_eig.cpp -o build/test_eig
//   c++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -I include tests/test_eig.cpp -o build/test_eig_san
//
// 只依赖 C++17 标准库；测试矩阵由固定种子的伪随机数生成。
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "stablercwa/eig.hpp"

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

// 断言的计数与报错都在下面几个普通函数里；宏只负责取文件名、行号和表达式的文字，宏里不出现中文
// （带 # 运算符的宏里若有中文的字符串，MSVC 会在每一处展开上报 C4819）。
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
struct Rng {   // splitmix64：不依赖标准库的分布实现，各平台结果相同
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

static Matrix random_matrix(int rows, int cols, std::uint64_t seed) {
    Rng rng(seed);
    Matrix a(rows, cols);
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i) a(i, j) = rng.cuniform();
    return a;
}

static Matrix random_hermitian(int n, std::uint64_t seed) {
    const Matrix g = random_matrix(n, n, seed);
    Matrix a(n, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) a(i, j) = 0.5 * (g(i, j) + std::conj(g(j, i)));
    return a;
}

static bool same_bits(const Matrix& a, const Matrix& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    const std::size_t bytes = static_cast<std::size_t>(a.rows()) * a.cols() * sizeof(cplx);
    return bytes == 0 || std::memcmp(a.data(), b.data(), bytes) == 0;
}
static bool same_bits(const Vector& a, const Vector& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(cplx)) == 0);
}

// 按定义直接算的残差（不带任何缩放技巧），用来核对 eig_residual 的公式。
static double residual_by_definition(const Matrix& a, const Vector& values, const Matrix& vectors) {
    const int n = a.rows();
    double fro = 0.0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) fro += std::norm(a(i, j));
    fro = std::sqrt(fro);
    double worst = 0.0;
    for (int j = 0; j < n; ++j) {
        double defect = 0.0, vnorm = 0.0;
        for (int i = 0; i < n; ++i) {
            cplx s = -values[j] * vectors(i, j);
            for (int k = 0; k < n; ++k) s += a(i, k) * vectors(k, j);
            defect += std::norm(s);
            vnorm += std::norm(vectors(i, j));
        }
        worst = std::max(worst, std::sqrt(defect) / (fro * std::sqrt(vnorm)));
    }
    return worst;
}

// ---------------------------------------------------------------------------------------------
static void test_dispatch() {
    say("[eig() 分派]\n");
    for (int n : {1, 2, 7, 24}) {
        const Matrix a = random_matrix(n, n, 100 + n);
        const Matrix copy = a;
        const EigResult qr = eig_qr(a), lr = eig_lr(a);
        const EigResult d0 = eig(a);                         // 默认方法是 QR
        const EigResult d1 = eig(a, EigMethod::QR);
        const EigResult d2 = eig(a, EigMethod::LR);
        CHECK(same_bits(d0.values, qr.values) && same_bits(d0.vectors, qr.vectors));
        CHECK(same_bits(d1.values, qr.values) && same_bits(d1.vectors, qr.vectors));
        CHECK(same_bits(d2.values, lr.values) && same_bits(d2.vectors, lr.vectors));
        CHECK(d0.sweeps == qr.sweeps && d0.info == qr.info && d2.sweeps == lr.sweeps && d2.info == lr.info);
        CHECK(same_bits(a, copy));                           // 输入不被修改
    }

    // 选项原样转交
    const Matrix a = random_matrix(12, 12, 7);
    for (EigMethod method : {EigMethod::QR, EigMethod::LR}) {
        const EigResult full = eig(a, method);
        EigOptions values_only;
        values_only.vectors = false;
        const EigResult v = eig(a, method, values_only);
        CHECK(v.vectors.empty() && v.Z.empty() && v.T.empty());
        CHECK(v.values.size() == 12 && v.info == 0);

        EigOptions keep;
        keep.keep_transform = true;
        const EigResult k = eig(a, method, keep);
        CHECK(k.Z.rows() == 12 && k.Z.cols() == 12 && k.T.rows() == 12 && k.T.cols() == 12);
        CHECK(same_bits(k.values, full.values) && same_bits(k.vectors, full.vectors));
        CHECK_LE(norm_fro(a * k.Z - k.Z * k.T) / norm_fro(a), method == EigMethod::QR ? 1e-13 : 1e-10);   // A Z = Z T

        int calls = 0, done = 0;
        EigOptions observed;
        observed.observer = [&](const EigProgress& p) {
            ++calls;
            if (std::string(p.stage) == "done") ++done;
        };
        const EigResult o = eig(a, method, observed);
        CHECK(calls == o.sweeps + 2 && done == 1);           // 化 Hessenberg 形 1 次 + 每次迭代 1 次 + 结束 1 次
        CHECK(same_bits(o.values, full.values));

        EigOptions one;                                      // 迭代上限太小：通过 info 报告，不抛异常
        one.max_iter = 1;
        const EigResult bad = eig(a, method, one);
        CHECK(bad.info > 0 && bad.vectors.empty());
    }

    CHECK_THROWS(eig(Matrix(2, 3)), std::invalid_argument);
    CHECK_THROWS(eig(Matrix(2, 3), EigMethod::LR), std::invalid_argument);
    CHECK_THROWS(eig(a, static_cast<EigMethod>(99)), std::invalid_argument);
    const EigResult empty = eig(Matrix());
    CHECK(empty.values.empty() && empty.info == 0);
}

static void test_residual() {
    say("[eig_residual]\n");
    const double nan = std::numeric_limits<double>::quiet_NaN();
    (void)nan;

    // 手算：A = [[2, 1], [0, 3]]，λ = 2、3，v = (1, 0)、(1, 1)：精确的本征对，残差为 0
    {
        const Matrix a{{2.0, 1.0}, {0.0, 3.0}};
        EigResult r;
        r.values = {2.0, 3.0};
        r.vectors = Matrix{{1.0, 1.0}, {0.0, 1.0}};
        CHECK(eig_residual(a, r) == 0.0);
        // 把第二个本征值改成 3 + δ：A v − λ v = −δ v，残差 = δ / ||A||_F = δ / sqrt(14)
        r.values[1] = cplx(3.0, 0.25);
        CHECK_LE(std::abs(eig_residual(a, r) - 0.25 / std::sqrt(14.0)), 1e-16);
        // 本征向量整体乘一个复数，残差不变
        const double before = eig_residual(a, r);
        for (int i = 0; i < 2; ++i) r.vectors(i, 1) *= cplx(7.3, -2.0);
        CHECK_LE(std::abs(eig_residual(a, r) - before), 1e-16);
    }

    // 与按定义直接计算的结果一致；酉变换算出的本征对残差是机器精度的小倍数
    for (int n : {3, 10, 40}) {
        const Matrix a = random_matrix(n, n, 300 + n);
        const EigResult qr = eig(a, EigMethod::QR), lr = eig(a, EigMethod::LR);
        const double rq = eig_residual(a, qr), rl = eig_residual(a, lr);
        CHECK_LE(std::abs(rq - residual_by_definition(a, qr.values, qr.vectors)), 1e-15);
        CHECK_LE(std::abs(rl - residual_by_definition(a, lr.values, lr.vectors)), 1e-3 * rl + 1e-15);
        CHECK(rq > 0.0);
        CHECK_LE(rq, 1e-14);
        CHECK_LE(rl, 1e-9);
        // 人为把一个本征值挪动 1e-3·||A||：残差应当正好是这个量级
        EigResult shifted = qr;
        shifted.values[n / 2] += 1e-3 * norm_fro(a);
        const double rs = eig_residual(a, shifted);
        CHECK(rs > 0.99e-3 && rs < 1.01e-3);
    }

    // 残差与矩阵的整体缩放无关，极大、极小的量级也不溢出
    {
        const Matrix a = random_matrix(8, 8, 55);
        const EigResult r = eig(a);
        const double base = eig_residual(a, r);
        for (double scale : {1e-200, 1e200}) {
            Matrix b = a;
            b *= cplx(scale, 0.0);
            EigResult rs = r;
            for (cplx& v : rs.values) v *= scale;
            const double scaled = eig_residual(b, rs);
            CHECK(std::isfinite(scaled));
            CHECK_LE(std::abs(scaled - base), 1e-15);
        }
    }

    // 对角矩阵、零矩阵：精确
    {
        const Matrix d = Matrix::diagonal({cplx(1.0, 2.0), -3.0, 0.5});
        CHECK(eig_residual(d, eig(d)) == 0.0);
        const Matrix z(4, 4);
        const EigResult rz = eig(z);
        CHECK(eig_residual(z, rz) == 0.0);
        EigResult wrong = rz;                    // 零矩阵配一个非零“本征值”
        wrong.values[0] = 1.0;
        CHECK(std::isinf(eig_residual(z, wrong)));
    }

    // 没有本征向量 → NaN；零向量 → +inf；NaN 能传出来
    {
        const Matrix a = random_matrix(6, 6, 9);
        EigOptions values_only;
        values_only.vectors = false;
        CHECK(std::isnan(eig_residual(a, eig(a, EigMethod::QR, values_only))));
        EigOptions one;
        one.max_iter = 1;
        const EigResult unconverged = eig(a, EigMethod::LR, one);
        CHECK(unconverged.info > 0 && std::isnan(eig_residual(a, unconverged)));

        EigResult r = eig(a);
        for (int i = 0; i < 6; ++i) r.vectors(i, 2) = 0.0;
        CHECK(std::isinf(eig_residual(a, r)));
        r = eig(a);
        r.vectors(1, 4) = std::numeric_limits<double>::quiet_NaN();
        CHECK(std::isnan(eig_residual(a, r)));
    }

    // 参数检查
    {
        const Matrix a = random_matrix(4, 4, 3);
        const EigResult r = eig(a);
        CHECK_THROWS(eig_residual(Matrix(3, 4), r), std::invalid_argument);
        CHECK_THROWS(eig_residual(random_matrix(5, 5, 4), r), std::invalid_argument);     // 本征值个数不对
        EigResult bad = r;
        bad.vectors = Matrix(4, 3);
        CHECK_THROWS(eig_residual(a, bad), std::invalid_argument);                        // 本征向量矩阵形状不对
        bad = r;
        bad.values.pop_back();                                                            // 本征向量齐全，本征值少一个
        CHECK_THROWS(eig_residual(a, bad), std::invalid_argument);
        bad = r;
        bad.values.push_back(1.0);                                                        // 本征值多一个
        CHECK_THROWS(eig_residual(a, bad), std::invalid_argument);
        CHECK(eig_residual(Matrix(), eig(Matrix())) == 0.0);
    }
}

static void test_orthogonality() {
    say("[orthogonality_defect]\n");
    CHECK(orthogonality_defect(Matrix::identity(5)) == 0.0);

    // 手算：V = [[1, 1], [0, 1]]，V^H V − I = [[0, 1], [1, 1]]，F-范数 = sqrt(3)
    CHECK_LE(std::abs(orthogonality_defect(Matrix{{1.0, 1.0}, {0.0, 1.0}}) - std::sqrt(3.0)), 1e-15);
    // 复数：V = [[1, i], [0, 1]]，V^H V − I = [[0, i], [−i, 1]]，同样是 sqrt(3)
    CHECK_LE(std::abs(orthogonality_defect(Matrix{{1.0, cplx(0.0, 1.0)}, {0.0, 1.0}}) - std::sqrt(3.0)), 1e-15);
    // 各列长度为 2 的正交向量：V^H V = 4 I，偏差 = 3·sqrt(n)
    {
        Matrix v = Matrix::identity(4);
        v *= cplx(0.0, 2.0);
        CHECK_LE(std::abs(orthogonality_defect(v) - 6.0), 1e-14);
    }
    // 长方形：3×2 的两列标准正交
    {
        const double h = std::sqrt(0.5);
        const Matrix v{{h, h}, {cplx(0.0, h), cplx(0.0, -h)}, {0.0, 0.0}};
        CHECK_LE(orthogonality_defect(v), 1e-15);
    }

    // 厄米矩阵：QR 迭代（酉变换）给出的本征向量两两正交到机器精度；LR 迭代的结果也是本征向量，但正交性差得多
    for (int n : {6, 30}) {
        const Matrix h = random_hermitian(n, 700 + n);
        const EigResult qr = eig(h, EigMethod::QR), lr = eig(h, EigMethod::LR);
        const double dq = orthogonality_defect(qr.vectors), dl = orthogonality_defect(lr.vectors);
        CHECK_LE(dq, 1e-13);
        CHECK(std::isfinite(dl) && dl >= 0.0);
        CHECK_LE(dl, 1e-6);
        CHECK_LE(eig_residual(h, qr), 1e-14);
        say("  n = %2d 的厄米矩阵：||V^H V − I||_F  QR %.1e   LR %.1e\n", n, dq, dl);
    }
    // 非正规矩阵的本征向量本来就不正交
    {
        const Matrix a{{1.0, 5.0}, {0.0, 2.0}};
        CHECK(orthogonality_defect(eig(a).vectors) > 0.5);
    }

    // 没有列（例如迭代未收敛、没有本征向量）→ NaN；NaN 能传出来
    CHECK(std::isnan(orthogonality_defect(Matrix())));
    {
        Matrix v = Matrix::identity(3);
        v(1, 2) = std::numeric_limits<double>::quiet_NaN();
        CHECK(std::isnan(orthogonality_defect(v)));
    }
}

int main() {
    test_dispatch();
    test_residual();
    test_orthogonality();
    say("通过 %d / 共 %d\n", g_pass, g_total);
    return g_pass == g_total ? 0 : 1;
}
