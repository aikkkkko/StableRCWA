// stablercwa/lu.hpp —— LU 分解（高斯消元）、解线性方程组、求逆
//
//   LU f = lu_factor(A);            // P·A = L·U，默认按列选主元；同时给出增长因子和最大乘子
//   Matrix X = lu_solve(f, B);      // 解 A X = B，可以一次带多个右端
//   Matrix X = solve(A, B);         // 分解 + 求解；A 奇异时抛 std::runtime_error
//   Matrix Ai = inverse(A);
//
// 算法是右视（right-looking）的列主元高斯消元，存储约定与 LAPACK 的 xGETRF 相同：
// L 的乘子放在严格下三角，U 放在上三角，perm 记录每一步交换的行。
//
// 为了少扫内存，消元按“面板”推进：每次取 kLuBlock 列，先把这几列分解完，再让右边每一列
// 一口气补做这 kLuBlock 步消元。对任何一个元素来说，它经历的运算及先后顺序与教科书上
// 一步一步消元完全相同，分块和并行只改变“先处理哪一列”，所以线程数不影响结果的任何一位。
// 多个右端的前代、回代也按同样的办法分块，并对右端列并行。
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "matrix.hpp"

namespace stablercwa {

enum class Pivoting {
    Partial,   // 按列选主元：每一步把本列中模最大的元素换到对角线上
    None,      // 不选主元：对角元是什么就用什么（只用于演示不稳定性）
};

struct LU {
    Matrix lu;                     // 严格下三角存 L 的乘子（单位对角不存），上三角存 U
    std::vector<int> perm;         // perm[k]：第 k 步与第 k 行交换的行号（LAPACK ipiv 约定，0 起）
    bool singular = false;         // 出现了精确为零的主元
    double growth = 1.0;           // 元素增长因子：消元过程中出现过的 max|a_ij^(k)| 除以 max|a_ij|
    double max_multiplier = 0.0;   // 最大乘子的模
};

namespace detail {
namespace dense {   // lu / qr / svd 共用的内部工具；多嵌一层名字空间，免得与其他头文件里的同名函数冲突

inline constexpr int kLuBlock = 32;               // 面板宽度：32 列乘子约 0.5 MB（n=1000），留在二级缓存里
inline constexpr long long kParallelWork = 200000;  // 一个并行区至少要有这么多次“元素更新”才值得开线程
inline constexpr long long kChunkWork = 50000;      // 每个线程至少分到的工作量

// std::complex<double> 数组可以当作“实部、虚部交替”的 double 数组访问（标准保证的布局）。
// 内层循环都直接写实部/虚部的算式：std::complex 的 operator* 带有 NaN 修正分支，会妨碍向量化。
inline double* as_real(cplx* p) noexcept { return reinterpret_cast<double*>(p); }
inline const double* as_real(const cplx* p) noexcept { return reinterpret_cast<const double*>(p); }

// 非负 double 的二进制表示按无符号整数比较，与按数值比较的顺序一致（NaN 排在最大）。
// 用它来求 max|z|²：整数取最大的循环可以向量化，严格 IEEE 语义下的浮点 max 不行。
inline std::uint64_t to_bits(double v) noexcept {
    std::uint64_t u;
    std::memcpy(&u, &v, sizeof u);
    return u;
}
inline double from_bits(std::uint64_t u) noexcept {
    double v;
    std::memcpy(&v, &u, sizeof v);
    return v;
}
inline std::uint64_t sq_bits(double re, double im) noexcept { return to_bits(re * re + im * im); }

inline std::uint64_t max_sq_bits(const cplx* x, std::size_t len) noexcept {
    const double* p = as_real(x);
    std::uint64_t m = 0;
    for (std::size_t i = 0; i < len; ++i) {
        const std::uint64_t q = sq_bits(p[2 * i], p[2 * i + 1]);
        m = q > m ? q : m;
    }
    return m;
}

// a·b，不带 NaN 修正。
inline cplx cmul(const cplx& a, const cplx& b) noexcept {
    return {a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real()};
}
// a / b，Smith 公式：先用 b 中较大的分量归一，避免 |b|² 上溢或下溢。
inline cplx cdiv(const cplx& a, const cplx& b) noexcept {
    const double br = b.real(), bi = b.imag();
    if (std::abs(br) >= std::abs(bi)) {
        const double r = bi / br, d = br + bi * r;
        return {(a.real() + a.imag() * r) / d, (a.imag() - a.real() * r) / d};
    }
    const double r = br / bi, d = bi + br * r;
    return {(a.real() * r + a.imag()) / d, (a.imag() * r - a.real()) / d};
}

// 消元内核的模板参数 Track 决定要不要顺带统计“被改写的元素的 max|z|²”（增长因子要用）：
//   0（或 false）  不统计；
//   1（或 true）   统计 |z|²；
//   2              统计 |sc·z|²，sc 是 2 的幂。给量级极端的矩阵用：直接平方会上溢或下溢，就先乘 sc 再平方。
//                  缩放只发生在统计量里，矩阵元素本身一位都不动，所以三种方式算出的分解逐位相同。
template <int Track>
inline void track_max([[maybe_unused]] std::uint64_t& m, [[maybe_unused]] double re, [[maybe_unused]] double im,
                      [[maybe_unused]] double sc) noexcept {
    if constexpr (Track == 1) {
        const std::uint64_t q = sq_bits(re, im);
        m = q > m ? q : m;
    } else if constexpr (Track == 2) {
        const std::uint64_t q = sq_bits(re * sc, im * sc);
        m = q > m ? q : m;
    }
}

// c[i] -= l[i]·t，i = 0 … len-1。返回更新后 max|c[i]|² 的位模式（按 Track 的方式统计；不统计时返回 0）。
template <int Track>
inline std::uint64_t sub_col1(cplx* c_, const cplx* l_, cplx t, int len, double sc = 1.0) noexcept {
    double* c = as_real(c_);
    const double* l = as_real(l_);
    const double tr = t.real(), ti = t.imag();
    std::uint64_t m = 0;
    for (int i = 0; i < len; ++i) {
        const int r = 2 * i, s = r + 1;
        const double xr = c[r] - l[r] * tr + l[s] * ti;
        const double xi = c[s] - l[r] * ti - l[s] * tr;
        track_max<Track>(m, xr, xi, sc);
        c[r] = xr;
        c[s] = xi;
    }
    return m;
}

// 依次做 c -= l0·t[0]、c -= l1·t[1]、c -= l2·t[2]、c -= l3·t[3]，但这一列只读写一遍。
// 每个元素上四次减法的先后顺序与分四次调用 sub_col1 相同。统计时每一步之后的值都参与取最大。
template <int Track>
inline std::uint64_t sub_cols4(cplx* c_, const cplx* l0_, const cplx* l1_, const cplx* l2_, const cplx* l3_,
                               const cplx* t, int len, double sc = 1.0) noexcept {
    double* c = as_real(c_);
    const double* l0 = as_real(l0_);
    const double* l1 = as_real(l1_);
    const double* l2 = as_real(l2_);
    const double* l3 = as_real(l3_);
    const double t0r = t[0].real(), t0i = t[0].imag(), t1r = t[1].real(), t1i = t[1].imag();
    const double t2r = t[2].real(), t2i = t[2].imag(), t3r = t[3].real(), t3i = t[3].imag();
    std::uint64_t m = 0;
    for (int i = 0; i < len; ++i) {
        const int r = 2 * i, s = r + 1;
        double xr = c[r], xi = c[s];
        xr = xr - l0[r] * t0r + l0[s] * t0i;
        xi = xi - l0[r] * t0i - l0[s] * t0r;
        track_max<Track>(m, xr, xi, sc);
        xr = xr - l1[r] * t1r + l1[s] * t1i;
        xi = xi - l1[r] * t1i - l1[s] * t1r;
        track_max<Track>(m, xr, xi, sc);
        xr = xr - l2[r] * t2r + l2[s] * t2i;
        xi = xi - l2[r] * t2i - l2[s] * t2r;
        track_max<Track>(m, xr, xi, sc);
        xr = xr - l3[r] * t3r + l3[s] * t3i;
        xi = xi - l3[r] * t3i - l3[s] * t3r;
        track_max<Track>(m, xr, xi, sc);
        c[r] = xr;
        c[s] = xi;
    }
    return m;
}

// 对一列 c 补做第 ks … ke-1 步消元：c[i] -= l(i,k)·c[k]（i > k），k 从小到大；l 是 lu 的第 k 列。
// 四步一组：先在 4 行的小三角里逐步定出这四步各自用到的 c[k]，再对下方所有行一次减掉四个乘子列的倍数。
// 返回被改写过的元素的 max|·|²（位模式，按 Track 的方式统计）；不统计时恒为 0。前代解 L y = b 用的也是它。
template <int Track>
inline std::uint64_t eliminate_column(const Matrix& lu, int ks, int ke, cplx* c, double sc = 1.0) noexcept {
    const int n = lu.rows();
    std::uint64_t m = 0;
    int k = ks;
    for (; k + 4 <= ke; k += 4) {
        const cplx* l[4] = {lu.col(k), lu.col(k + 1), lu.col(k + 2), lu.col(k + 3)};
        for (int p = 0; p < 3; ++p) {
            const cplx t = c[k + p];
            const std::uint64_t q = sub_col1<Track>(c + k + p + 1, l[p] + k + p + 1, t, 3 - p, sc);
            m = q > m ? q : m;
        }
        const cplx t[4] = {c[k], c[k + 1], c[k + 2], c[k + 3]};
        if (t[0] == cplx{} && t[1] == cplx{} && t[2] == cplx{} && t[3] == cplx{}) continue;   // 这四步都不改变这一列
        const std::uint64_t q =
            sub_cols4<Track>(c + k + 4, l[0] + k + 4, l[1] + k + 4, l[2] + k + 4, l[3] + k + 4, t, n - k - 4, sc);
        m = q > m ? q : m;
    }
    for (; k < ke; ++k) {
        const cplx t = c[k];
        if (t == cplx{}) continue;
        const std::uint64_t q = sub_col1<Track>(c + k + 1, lu.col(k) + k + 1, t, n - k - 1, sc);
        m = q > m ? q : m;
    }
    return m;
}

// 回代解 U x = y 的一段：依次处理第 ke-1 … ks 列。x[k] /= u(k,k)，再 x[i] -= u(i,k)·x[k]（i < k）。
// u 只用到上三角（含对角）；同样四列一组。QR 分解解方程时的回代也用它。
inline void back_substitute_column(const Matrix& u, int ks, int ke, cplx* x) noexcept {
    int k = ke;
    for (; k - 4 >= ks; k -= 4) {
        const cplx* uc[4] = {u.col(k - 1), u.col(k - 2), u.col(k - 3), u.col(k - 4)};
        for (int p = 0; p < 4; ++p) {
            const int row = k - 1 - p;
            x[row] = cdiv(x[row], uc[p][row]);
            sub_col1<false>(x + k - 4, uc[p] + k - 4, x[row], row - (k - 4));
        }
        const cplx t[4] = {x[k - 1], x[k - 2], x[k - 3], x[k - 4]};
        if (t[0] == cplx{} && t[1] == cplx{} && t[2] == cplx{} && t[3] == cplx{}) continue;
        sub_cols4<false>(x, uc[0], uc[1], uc[2], uc[3], t, k - 4);
    }
    for (; k > ks; --k) {
        const cplx* uk = u.col(k - 1);
        x[k - 1] = cdiv(x[k - 1], uk[k - 1]);
        if (x[k - 1] == cplx{}) continue;
        sub_col1<false>(x, uk, x[k - 1], k - 1);
    }
}

// 对第 begin … end-1 列调用 body(lo, hi)。每列的工作量约为 work_per_column 次元素更新；
// 总量太小就不并行。各列互不相干，所以怎么分段都不影响结果。
template <class Body>
inline void for_columns(int begin, int end, long long work_per_column, Body&& body) {
    const int count = end - begin;
    if (count <= 0) return;
    work_per_column = (std::max)(1LL, work_per_column);
    if (count < 2 || work_per_column * count < kParallelWork) {
        body(begin, end);
        return;
    }
    const int min_chunk = static_cast<int>(std::min<long long>(count, kChunkWork / work_per_column + 1));
    parallel_ranges(begin, end, body, min_chunk);
}

// 就地解 A X = B：b 指向 n × nrhs 的按列存放的右端，返回时被解覆盖。
inline void lu_solve_inplace(const LU& f, cplx* b, int nrhs) {
    const int n = f.lu.rows();
    const auto column = [&](int j) { return b + static_cast<std::size_t>(j) * n; };
    // 1) 把分解时的行交换作用到右端：P·B
    for (int j = 0; j < nrhs; ++j) {
        cplx* x = column(j);
        for (int k = 0; k < n; ++k)
            if (f.perm[k] != k) std::swap(x[k], x[f.perm[k]]);
    }
    // 2) 前代 L Y = P·B。块在外层、右端列在内层：一个块的乘子列留在缓存里给所有右端用。
    for (int ks = 0; ks < n; ks += kLuBlock) {
        const int ke = (std::min)(ks + kLuBlock, n);
        for_columns(0, nrhs, static_cast<long long>(ke - ks) * (n - ks), [&](int lo, int hi) {
            for (int j = lo; j < hi; ++j) eliminate_column<false>(f.lu, ks, ke, column(j));
        });
    }
    // 3) 回代 U X = Y，自最后一块向前。
    for (int ke = n; ke > 0; ke -= (std::min)(ke, kLuBlock)) {
        const int ks = ke - (std::min)(ke, kLuBlock);
        for_columns(0, nrhs, static_cast<long long>(ke - ks) * ke, [&](int lo, int hi) {
            for (int j = lo; j < hi; ++j) back_substitute_column(f.lu, ks, ke, column(j));
        });
    }
}

inline void check_lu_for_solve(const LU& f, int rhs_rows) {
    if (!f.lu.square() || static_cast<int>(f.perm.size()) != f.lu.rows())
        throw std::invalid_argument("lu_solve: malformed LU");
    if (rhs_rows != f.lu.rows()) throw std::invalid_argument("lu_solve: right-hand side has wrong number of rows");
    if (f.singular) throw std::runtime_error("lu_solve: matrix is singular (zero pivot)");
}

// lu_factor 的实现。Track=false 时不统计 growth 和 max_multiplier（两者留在默认值），
// 分解本身逐位相同、省掉约三成时间；solve() 和 inverse() 用不到这两个量，内部走这一支。
template <bool Track>
inline LU lu_factor_impl(Matrix a, Pivoting pivoting) {
    if (!a.square()) throw std::invalid_argument("lu_factor: matrix must be square");
    const int n = a.rows();
    LU f;
    f.perm.resize(static_cast<std::size_t>(n));
    for (int k = 0; k < n; ++k) f.perm[k] = k;
    if (n == 0) return f;
    const std::size_t total = static_cast<std::size_t>(n) * n;

    // 输入的 max|a_ij|²。增长因子是用 |z|² 跟踪的，量级太大或太小时平方会上溢/下溢，
    // 那就在统计时先把元素乘一个 2 的幂 sc 再平方（内核的 Track = 2）。只缩放统计量，不缩放矩阵：
    // 把矩阵整体缩小会把量级悬殊的小元素压成次正规数甚至 0，分解就不对了。
    double q0 = 0.0;
    double sc = 1.0;
    bool scaled_stats = false;
    bool finite_input = true;
    if constexpr (Track) {
        q0 = from_bits(max_sq_bits(a.data(), total));
        if (!(q0 >= 1e-60 && q0 <= 1e60)) {
            finite_input = a.all_finite();
            const double amax = finite_input ? max_abs(a) : 0.0;
            q0 = 0.0;
            if (amax > 0.0) {
                scaled_stats = true;
                sc = std::ldexp(1.0, -std::clamp(std::ilogb(amax), -1000, 1000));   // 夹住指数，sc 本身才不会溢出
                const double* p = as_real(a.data());
                std::uint64_t m = 0;
                for (std::size_t i = 0; i < total; ++i) track_max<2>(m, p[2 * i], p[2 * i + 1], sc);
                q0 = from_bits(m);
            }
        }
    }

    // col_max[j]：第 j 列在消元中出现过的 max|a_ij|²（位模式）。一列只由处理它的那个线程写。
    std::vector<std::uint64_t> col_max(Track ? static_cast<std::size_t>(n) : 0, 0);
    std::uint64_t mult_max = 0;     // max|乘子|²（位模式）
    double mult_max_slow = 0.0;     // 平方上溢/下溢的列改用 std::abs 统计
    const bool partial = pivoting == Pivoting::Partial;
    const int* perm = f.perm.data();

    // 让第 j 列补上面板 [ks, ke) 的行交换和消元，并记录这一列出现过的最大元素。
    const auto update_column = [&](int j, int ks, int ke) {
        cplx* c = a.col(j);
        if (partial)
            for (int k = ks; k < ke; ++k)
                if (perm[k] != k) std::swap(c[k], c[perm[k]]);
        if constexpr (Track) {
            const std::uint64_t q =
                scaled_stats ? eliminate_column<2>(a, ks, ke, c, sc) : eliminate_column<1>(a, ks, ke, c);
            if (q > col_max[j]) col_max[j] = q;
        } else {
            eliminate_column<0>(a, ks, ke, c);
        }
    };

    for (int ks = 0; ks < n; ks += kLuBlock) {
        const int ke = (std::min)(ks + kLuBlock, n);

        // 1) 分解面板 [ks, ke)。第 j 列先补上本面板前 j-ks 步的消元，然后选主元、算乘子。
        for (int j = ks; j < ke; ++j) {
            update_column(j, ks, j);
            cplx* c = a.col(j);
            if (partial) {
                int p = j;
                double best = std::abs(c[j]);
                for (int i = j + 1; i < n; ++i) {
                    const double v = std::abs(c[i]);
                    if (v > best) { best = v; p = i; }
                }
                f.perm[j] = p;
                if (p != j)   // 交换本列和面板内已完成的列；面板外的列稍后再补
                    for (int k = ks; k <= j; ++k) std::swap(a(j, k), a(p, k));
            }
            const cplx pivot = c[j];
            if (pivot == cplx{}) {
                f.singular = true;
                for (int i = j + 1; i < n; ++i) c[i] = cplx{};
                continue;
            }
            // 乘子 l(i,j) = a(i,j) / pivot。先求 1/pivot 再相乘；pivot 小到倒数溢出时逐个做除法。
            const cplx inv = cdiv(cplx{1.0, 0.0}, pivot);
            if (std::isfinite(inv.real()) && std::isfinite(inv.imag())) {
                for (int i = j + 1; i < n; ++i) c[i] = cmul(c[i], inv);
            } else {
                for (int i = j + 1; i < n; ++i) c[i] = cdiv(c[i], pivot);
            }
            if constexpr (Track) {
                const std::uint64_t qb = max_sq_bits(c + j + 1, static_cast<std::size_t>(n - j - 1));
                const double q = from_bits(qb);
                if (q >= 1e-280 && q <= 1e280) {
                    if (qb > mult_max) mult_max = qb;
                } else {
                    for (int i = j + 1; i < n; ++i) {
                        const double v = std::abs(c[i]);
                        if (!(v <= mult_max_slow)) mult_max_slow = v;   // 这种写法让 NaN 也能传出来
                    }
                }
            }
        }

        // 2) 把本面板的行交换补到左边已经完成的列上（那里存的是 L 的乘子）。
        if (partial)
            for (int j = 0; j < ks; ++j) {
                cplx* c = a.col(j);
                for (int k = ks; k < ke; ++k)
                    if (perm[k] != k) std::swap(c[k], c[perm[k]]);
            }

        // 3) 右边每一列补做这 ke-ks 步消元。列与列互不相干，可以并行。
        for_columns(ke, n, static_cast<long long>(ke - ks) * (n - ks), [&](int lo, int hi) {
            for (int j = lo; j < hi; ++j) update_column(j, ks, ke);
        });
    }

    if constexpr (Track) {
        std::uint64_t qmax = to_bits(q0);
        for (const std::uint64_t q : col_max) qmax = q > qmax ? q : qmax;
        if (!finite_input) {
            f.growth = std::numeric_limits<double>::quiet_NaN();
        } else if (q0 > 0.0) {
            f.growth = std::sqrt(from_bits(qmax)) / std::sqrt(q0);
            if (std::isnan(f.growth)) f.growth = std::numeric_limits<double>::infinity();   // 中途溢出
        }
        f.max_multiplier = std::sqrt(from_bits(mult_max));
        if (!(mult_max_slow <= f.max_multiplier)) f.max_multiplier = mult_max_slow;
    }
    f.lu = std::move(a);
    return f;
}

}  // namespace dense
}  // namespace detail

// P·A = L·U。a 必须是方阵。
//  · Pivoting::Partial：第 k 步在第 k 列的对角线及以下找模最大的元素（std::abs）换上来，乘子的模都不超过 1。
//  · Pivoting::None：不做任何行交换；对角元很小时乘子和后续元素会被放得很大，growth 与 max_multiplier 会如实反映。
// 主元精确为零时不做除法：singular 置为 true，这一步跳过（乘子记为 0），继续往下做，结果只表示“分解失败”。
// growth 把输入矩阵本身也算作第 0 步，所以总不小于 1；它以 |z|² 的形式跟踪，超过约 1e120 时可能记为 +inf。
// 输入里有 NaN/Inf 时不抛异常，growth 为 NaN。
inline LU lu_factor(Matrix a, Pivoting pivoting = Pivoting::Partial) {
    using namespace detail::dense;
    return lu_factor_impl<true>(std::move(a), pivoting);
}

// 用已有的分解解 A X = B（B 可以有多列）。分解标记为奇异时抛 std::runtime_error。
inline Matrix lu_solve(const LU& f, Matrix b) {
    using namespace detail::dense;
    check_lu_for_solve(f, b.rows());
    lu_solve_inplace(f, b.data(), b.cols());
    return b;
}
inline Vector lu_solve(const LU& f, Vector b) {
    using namespace detail::dense;
    check_lu_for_solve(f, static_cast<int>(b.size()));
    lu_solve_inplace(f, b.data(), 1);
    return b;
}

// 分解并求解 A X = B；A 奇异（出现零主元）时抛 std::runtime_error。
inline Matrix solve(const Matrix& a, const Matrix& b, Pivoting pivoting = Pivoting::Partial) {
    using namespace detail::dense;
    if (!a.square()) throw std::invalid_argument("solve: matrix must be square");
    if (b.rows() != a.rows()) throw std::invalid_argument("solve: right-hand side has wrong number of rows");
    const LU f = lu_factor_impl<false>(a, pivoting);
    if (f.singular) throw std::runtime_error("solve: matrix is singular (zero pivot)");
    return lu_solve(f, b);
}
inline Vector solve(const Matrix& a, const Vector& b, Pivoting pivoting = Pivoting::Partial) {
    using namespace detail::dense;
    if (!a.square()) throw std::invalid_argument("solve: matrix must be square");
    if (static_cast<int>(b.size()) != a.rows())
        throw std::invalid_argument("solve: right-hand side has wrong number of rows");
    const LU f = lu_factor_impl<false>(a, pivoting);
    if (f.singular) throw std::runtime_error("solve: matrix is singular (zero pivot)");
    return lu_solve(f, b);
}

// A^{-1}：解 A X = I。A 奇异时抛 std::runtime_error。
inline Matrix inverse(const Matrix& a) {
    if (!a.square()) throw std::invalid_argument("inverse: matrix must be square");
    return solve(a, Matrix::identity(a.rows()));
}

// 后向误差 ||A X − B||_F / (||A||_F ||X||_F + ||B||_F)。稳定的解法给出 eps 量级的值，与 A 的条件数无关。
inline double solve_backward_error(const Matrix& a, const Matrix& x, const Matrix& b) {
    if (a.cols() != x.rows() || a.rows() != b.rows() || x.cols() != b.cols())
        throw std::invalid_argument("solve_backward_error: shape mismatch");
    const double denom = norm_fro(a) * norm_fro(x) + norm_fro(b);
    const double resid = norm_fro(a * x - b);
    if (denom == 0.0) return resid == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
    return resid / denom;
}

}  // namespace stablercwa
