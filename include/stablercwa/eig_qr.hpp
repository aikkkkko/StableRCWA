// stablercwa/eig_qr.hpp —— 一般复矩阵的本征分解：酉变换化 Hessenberg 形 + 带位移的 QR 迭代
//
//   #include "stablercwa/eig_qr.hpp"
//   stablercwa::EigResult r = stablercwa::eig_qr(A);      // A 是任意 n×n 复矩阵
//   r.values[j]、r.vectors 的第 j 列：A v_j = λ_j v_j，||v_j||_2 = 1
//
// 算法与 EISPACK 的 CORTH + COMQR2 相同（不做平衡，low = 0、igh = n-1）：
//   1. CORTH：n-2 个 Householder 反射 P_m 做相似变换，把 A 化成上 Hessenberg 形 H；
//   2. 把这些反射累乘成酉矩阵 Z，并用对角酉相似把 H 的次对角元变成实数；
//   3. COMQR2：对未收敛的前 en+1 阶反复做带位移的 QR 步
//        H − s·I = Q R，  H ← R Q   （Q 由 Givens 类旋转组成，同时 Z ← Z Q），
//      次对角元 |h(l,l-1)| <= eps·(|h(l-1,l-1)| + |h(l,l)|) 时视为零，逐个收缩出本征值；
//   4. 迭代结束得到 Z^H A Z = T（上三角，即 Schur 形），回代求 T 的本征向量，再乘回 Z。
// 全程只用酉变换，所以 Z 的条件数恒为 1，舍入误差不被放大；这是它与 eig_lr（消元变换）唯一的差别。
//
// 位移公式、收敛判据、第 10/20 次迭代的例外位移、回代里的溢出控制与零分母替换都照 EISPACK。
// 与 EISPACK 的写法不同的有两处：
//   · 循环的组织方式。Matrix 按列存放，所以 QR 步改成逐列推进（每个元素经历的运算及其次序不变），
//     Householder 变换里的内积分四路累加，回代改成按列累加。这些只影响舍入的末位。
//   · 极端量级的保护。凡是要用到元素量级平方的地方都先乘 2 的幂缩放（见下面“极端量级的处理”一段），
//     正常量级下结果逐位不变。元素的绝对值在 1e-300 … 1e300 之间时（一个矩阵里大小悬殊也可以）
//     不会因为上溢、下溢而失准；更大时内部求和可能上溢，更小（次正规数）时误差只能小到次正规数的间隔。
//
// 精度是后向误差意义上的：Z^H A Z = T、A v = λ v 的偏差是 eps·||A|| 的小倍数。本征值本身准到什么程度
// 还取决于它的条件数；比 ||A|| 小很多个量级的本征值不保证相对精度（累计位移是从整条对角线上减的）。
//
// 多核并行：n >= 192 时，化 Hessenberg 形、累乘 Z、QR 迭代里链以外的更新、回代都分成块，交给 parallel.hpp 的线程池
// （set_num_threads 设的线程数）；n 更小时走原来的串行写法。n >= 192 时不论线程数是几、是否已经在别的并行区里
// （parallel_for 的循环体、RCWA 的层间并行……），走的都是分块版；线程数为 1 或已在并行区里时，各个并行区在调用线程上依次执行。
// 分块版里每个元素经历的运算及其次序与原来的串行写法一字不差，分块、分线程、推迟只改变“谁、何时算”，没有跨线程的归约。
// 保证的范围：编译器不把 a*b+c 融合成 fma 时（GCC、Clang 加 -ffp-contract=off；MSVC 默认的 /fp:precise），
// 任何线程数下的结果都与单线程的结果、与并行化之前的串行实现逐位相同（验收：tests/test_eig_qr_parallel.cpp）。
// 唯一的例外：结果本身是 NaN 时（元素量级超出上面声明的范围，例如全为 DBL_MAX，
// info > 0），NaN 的符号位与载荷不在保证之内（x86 上新旧两份代码给出的 NaN 符号位可以不同，与并行无关）。
// 开着乘加融合时（GCC 在带 FMA 的目标上默认如此，例如 x86 加 -march=native、aarch64；MSVC 加 /fp:contract；
// Clang 加 -ffp-contract=fast，以及 Clang 14 起的默认值 -ffp-contract=on 在 arm64 上——Apple 芯片上不加选项的构建就是）没有验证：融合与否由编译器按内联之后的上下文决定，同一个算式在串行写法里、在分块版里、
// 在不同的分块范围上可以融合得不一样，结果的末位就可能与旧版、与别的线程数的结果不同。要逐位可重复请加 -ffp-contract=off。
// 调用者不需要改任何代码。做法见下面“并行的组织方式”和各个 *_par 函数的说明。
//
// 本文件只用 C++17 标准库，可重入（没有全局状态）。
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "eig_types.hpp"
#include "parallel.hpp"

namespace stablercwa {
namespace detail {
namespace eigqr {

// 把复数数组当作 (re, im) 交错的 double 数组读写（标准保证 std::complex<double> 的数组可以这样访问）。
// 内层循环手写实部、虚部，是为了避开 std::complex 乘法里的 NaN 检查，编译器才能把循环向量化。
inline double* reals(cplx* p) { return reinterpret_cast<double*>(p); }
inline const double* reals(const cplx* p) { return reinterpret_cast<const double*>(p); }

// Σ conj(u_k)·a_k。分四路累加，打断加法的依赖链。
inline cplx dot_conj(const cplx* cu, const cplx* ca, int len) {
    const double* u = reals(cu);
    const double* a = reals(ca);
    double sr[4] = {0.0, 0.0, 0.0, 0.0}, si[4] = {0.0, 0.0, 0.0, 0.0};
    int k = 0;
    for (; k + 4 <= len; k += 4) {
        for (int q = 0; q < 4; ++q) {
            const double ur = u[2 * (k + q)], ui = u[2 * (k + q) + 1];
            const double ar = a[2 * (k + q)], ai = a[2 * (k + q) + 1];
            sr[q] += ur * ar + ui * ai;
            si[q] += ur * ai - ui * ar;
        }
    }
    for (; k < len; ++k) {
        const double ur = u[2 * k], ui = u[2 * k + 1], ar = a[2 * k], ai = a[2 * k + 1];
        sr[0] += ur * ar + ui * ai;
        si[0] += ur * ai - ui * ar;
    }
    return {(sr[0] + sr[1]) + (sr[2] + sr[3]), (si[0] + si[1]) + (si[2] + si[3])};
}

// y ← y + t·x
inline void axpy(cplx* cy, cplx t, const cplx* cx, int len) {
    double* y = reals(cy);
    const double* x = reals(cx);
    const double tr = t.real(), ti = t.imag();
    for (int k = 0; k < 2 * len; k += 2) {
        const double xr = x[k], xi = x[k + 1];
        y[k] += tr * xr - ti * xi;
        y[k + 1] += tr * xi + ti * xr;
    }
}

// x ← t·x
inline void scale(cplx* cx, cplx t, int len) {
    double* x = reals(cx);
    const double tr = t.real(), ti = t.imag();
    for (int k = 0; k < 2 * len; k += 2) {
        const double xr = x[k], xi = x[k + 1];
        x[k] = tr * xr - ti * xi;
        x[k + 1] = tr * xi + ti * xr;
    }
}

// ---------------------------------------------------------------------------------------------
// 并行的组织方式。n >= kParallelMinN 时 eig_qr 走“分块版”（下面各个 *_par 函数与 ParallelQr），n 更小时走原样保留的
// 串行写法。选哪一版只看 n（以及 full），与线程数无关：线程数为 1、或者调用线程已经在别的并行区里时，分块版照样走，
// 只是各个并行区在调用线程上依次执行；不另写“按线程数选用的串行副本”。
// 内核（axpy、dot_conj、rotate_rows、rotate_columns …）与串行写法共用。
//
// 分块版只改变“谁、何时算哪个元素”：每个矩阵元素经历的运算及其次序与原来的串行写法相同，没有跨线程的归约。
// 所以在编译器不把 a*b+c 融合成 fma 的前提下（GCC、Clang 加 -ffp-contract=off；MSVC 默认的 /fp:precise），
// 结果与线程数无关，并且与旧版逐位相同。
// 开着乘加融合时这两条都没有验证，也不靠结构保证，原因有两处：
//   · 融合与否取决于内联之后的上下文。分块版与串行版是不同的函数；同一个循环体“在调用线程上直接做”和“分给线程池做”
//     是两个调用点，可以被编译成两份融合方式不同的机器码。
//   · 按行切的工作（化 Hessenberg 的 work、H 的行右乘、Z 的波前）交给内核的行范围随线程数变，
//     而向量化的内核对一段数据的主体与尾部可以融合得不一样。
// 要逐位可重复（与旧版、与别的线程数相同）请加 -ffp-contract=off。
//
// 下面两个宏只供测试用（tests/test_eig_qr_parallel.cpp 与验收用的对照程序），正常使用不要定义：
//   STABLERCWA_EIGQR_FORCE_PARALLEL  把按工作量定的门槛降到最低，矩阵不大时也真的开出每一种并行区；
//   STABLERCWA_EIGQR_PARALLEL_STRESS 另外把 n_min、批大小、小块行数降到极小，几十阶的矩阵也走满分块版。
// 两者都不改变任何元素的运算，所以（同样在不融合乘加的前提下）结果不变。
#ifdef STABLERCWA_EIGQR_PARALLEL_STRESS
#define STABLERCWA_EIGQR_FORCE_PARALLEL 1
#endif
#ifdef STABLERCWA_EIGQR_FORCE_PARALLEL
// 只供测试：把各个按工作量定的门槛降到最低，使并行区（QR 迭代里的 R 区、S 区，化 Hessenberg 的每一步，Z 的批处理……）
// 在矩阵不大时也真的开出来；n_min 与每个元素的运算都不变。
constexpr long long kHessStepWork = 1;
constexpr long long kSegmentWork = 1;
constexpr long long kQueueWork = 0;
constexpr long long kRowsWork = 0;
#else
constexpr long long kHessStepWork = 400000;   // 化 Hessenberg 的一步：工作量（复乘加）达到它才并行
// 并行区里每个段至少要有多少工作量（“元素-旋转”次数）才值得开：线程池派发与汇合要 3–15 微秒，
// 机器上有别的进程时还会等最慢醒来的线程，所以一段至少要有约 0.1 毫秒的工作。
constexpr long long kSegmentWork = 100000;
// QR 迭代的两个并行区（见下面 ParallelQr）：R 区里队列的工作量（元素-旋转）达到 kQueueWork 才开，否则链单独做、队列继续攒；
// S 区（H 的行右乘）的工作量达到 kRowsWork 才并行。开了就用全部线程：试过按“每段工作量”限制段数，没有稳定的收益
// （这些区很短，用时由派发延迟和最慢的一段决定）。
constexpr long long kQueueWork = 150000;
constexpr long long kRowsWork = 60000;
#endif
#ifdef STABLERCWA_EIGQR_PARALLEL_STRESS
// 只供测试：再把 n_min、批大小、小块行数都降到极小，让几十阶的矩阵也走满分块版的每一条路径。
constexpr int kParallelMinN = 4;
constexpr int kZBatch = 3;
constexpr int kWaveRows = 5;
constexpr double kChainCost = 1.0;
#else
constexpr int kParallelMinN = 192;        // n 小于它整个走串行写法
constexpr int kZBatch = 16;               // Z 的右乘攒够这么多个 sweep 为一批，按波前施加
constexpr int kWaveRows = 64;             // 波前的小块行数
constexpr double kChainCost = 1.25;       // 链一个元素-旋转的用时与队列里一个元素-旋转之比（链与别的线程同时跑，访存更挤）
#endif

// 调用线程能用几个线程：1 表示所有并行区都在调用线程上依次执行（线程池只有一个线程，或者已经在别的并行区里）。
// 线程数只在入口读一次；之后线程池的线程数再变，也只是若干段落到同一个线程上依次执行，不影响结果。
inline int available_parts() { return in_parallel_region() ? 1 : num_threads(); }

// 把第 0 … segments-1 段分给线程池，body(p) 处理第 p 段。各段之间没有等待关系，
// 所以线程池给的线程比段数少（线程数被别的线程改小了）时只是变慢，不会死锁。
template <class Body>
inline void run_segments(int segments, Body&& body) {
    parallel_ranges(0, segments, [&](int lo, int hi) { for (int p = lo; p < hi; ++p) body(p); }, 1);
}

// 把 [0, count) 按权重切成 segments 段：第 p 段是 [cuts[p], cuts[p+1])，前 p 段的权重之和约占总权重的 fraction(p)
// （fraction 单调增，fraction(0) = 0）。
template <class Weight, class Fraction>
inline void cuts_by_weight(int count, int segments, Weight&& weight, Fraction&& fraction, std::vector<int>& cuts) {
    cuts.assign(segments + 1, count);
    cuts[0] = 0;
    double total = 0.0;
    for (int i = 0; i < count; ++i) total += weight(i);
    double acc = 0.0;
    int p = 1;
    for (int i = 0; i < count && p < segments; ++i) {
        acc += weight(i);
        while (p < segments && acc >= total * fraction(p)) cuts[p++] = i + 1;
    }
}

// 各段的权重之和尽量相等。
template <class Weight>
inline void balanced_cuts(int count, int segments, Weight&& weight, std::vector<int>& cuts) {
    cuts_by_weight(count, segments, weight, [&](int p) { return static_cast<double>(p) / segments; }, cuts);
}

// ---------------------------------------------------------------------------------------------
// 极端量级的处理。算法里有几处要用到矩阵元素量级的平方（反射的 u^H u、旋转的模、位移公式），
// 元素在 1e±154 之外时平方会上溢或下溢；元素是次正规数时除法得不到单位模的结果。
// 办法都一样：先乘一个 2 的幂把量级拉到 1 附近，算完再乘回去。乘 2 的幂只改指数、不引入舍入，
// 所以正常量级下的结果一位都不变，极端量级下的结果与“同一个矩阵整体缩放到正常量级”时相同。

// 量级是否在平方既不上溢、下溢，也不落进次正规数的范围内。
inline bool safe_magnitude(double m) { return m > 1e-140 && m < 1e140; }

// m > 0 写成 f·2^e、f ∈ [0.5, 1) 时的 e；乘以 2^-e 就把 m 拉到 1 附近。m 为零或非有限数时返回 0（不缩放）。
inline int binary_exponent(double m) {
    int e = 0;
    if (m > 0.0 && std::isfinite(m)) std::frexp(m, &e);
    return e;
}

// z·2^e
inline cplx scale2(cplx z, int e) { return {std::ldexp(z.real(), e), std::ldexp(z.imag(), e)}; }

// 返回单位模的 z/|z|，并把 |z| 写进 modulus（z ≠ 0）。每步迭代只调用一次，所以不分情况，总是先缩放。
inline cplx unit_phase(cplx z, double& modulus) {
    const int e = binary_exponent((std::max)(std::abs(z.real()), std::abs(z.imag())));
    z = scale2(z, -e);
    const double r = std::abs(z);
    modulus = std::ldexp(r, e);
    return z / r;
}

// ---------------------------------------------------------------------------------------------
// CORTH：Householder 反射化上 Hessenberg 形。
//
// 第 m 步（m = 1 … n-2）取第 m-1 列对角线以下的部分 x = A(m:n-1, m-1)，构造
//     P_m = I − u u^H / h，   u = x/scale，再把 u_m 加长 ||u||（沿 u_m 的方向），  h = u^H u / 2，
// 使 P_m x 只剩第 m 个分量，然后 A ← P_m A P_m。
// 返回时 a 的上 Hessenberg 部分就是 H；次对角线以下和 ort 里留着各个反射向量（scale·u），供累乘 Z 用。
inline void reduce_to_hessenberg(Matrix& a, Vector& ort) {
    const int n = a.rows();
    Vector work(n);                                   // 右乘时存 A·u / h
    for (int m = 1; m < n - 1; ++m) {
        cplx* x = a.col(m - 1);
        ort[m] = 0.0;
        double scale_x = 0.0;                         // 先缩放，防止平方时溢出
        for (int i = m; i < n; ++i) scale_x += abs1(x[i]);
        if (scale_x == 0.0) continue;                 // 这一列已经是 Hessenberg 形
        double h = 0.0;
        for (int i = n - 1; i >= m; --i) {
            ort[i] = x[i] / scale_x;
            h += std::norm(ort[i]);
        }
        double g = std::sqrt(h);                      // ||u||
        const double f = std::abs(ort[m]);
        if (f == 0.0) {
            ort[m] = g;
            x[m] = scale_x;
        } else {
            h += f * g;
            g /= f;
            ort[m] *= 1.0 + g;
        }
        const cplx* u = ort.data();
        const int len = n - m;
        // 左乘 P_m：只动第 m 行以下、第 m 列以右（第 m-1 列的结果是已知的，最后直接写）。
        // 每列左乘完，趁它还在缓存里，顺手把它乘 u_j 累加进 work = (P_m A)(:, m:) u，给右乘用。
        std::fill(work.begin(), work.end(), cplx{});
        for (int j = n - 1; j >= m; --j) {
            cplx* aj = a.col(j);
            axpy(aj + m, -dot_conj(u + m, aj + m, len) / h, u + m, len);
            axpy(work.data(), u[j], aj, n);
        }
        // 右乘 P_m：A(:, m:) ← A(:, m:) − (work / h) u^H
        for (int i = 0; i < n; ++i) work[i] /= h;
        for (int j = m; j < n; ++j) axpy(a.col(j), -std::conj(u[j]), work.data(), n);

        ort[m] *= scale_x;
        x[m] *= -g;                                   // 新的次对角元 h(m, m-1)
    }
}

// reduce_to_hessenberg 的并行版：每个元素的运算及其次序与原版相同。
//
// 第 m 步里 dot_conj 是沿一整列的四路累加，work[i] 是沿一整行跨列的累加（j 从 n-1 到 m），都不能拆开，
// 所以每步至少要“按列”和“按行”各做一次；而同一个元素若被三个不同的线程依次写、读、写，缓存行就在核之间
// 来回搬，比串行还慢。这里把第 m 步的右乘推迟到第 m+1 步，与第 m+1 步的左乘在同一列上连着做
// （列在 L1 里连用两次），一步只剩两个并行区：按列（补上一步的右乘 + 本步的左乘）、按行（work）。
// 每个元素的运算及其次序不变：第 m+1 步的反射由第 m 列定出，所以第 m 列先单独补上第 m 步的右乘；
// 其余各列在做第 m+1 步的左乘之前补。
inline void reduce_to_hessenberg_par(Matrix& a, Vector& ort, int parts) {
    const int n = a.rows();
    Vector work(n);                                   // 这一步的 A·u / h
    Vector prev_u(n);                                 // 上一步的 u
    bool pending = false;                             // 上一步的右乘（用 prev_u、work）还没做
    for (int m = 1; m < n - 1; ++m) {
        cplx* x = a.col(m - 1);
        if (pending) axpy(x, -std::conj(prev_u[m - 1]), work.data(), n);
        ort[m] = 0.0;
        double scale_x = 0.0;                         // 先缩放，防止平方时溢出
        for (int i = m; i < n; ++i) scale_x += abs1(x[i]);
        const bool reflect = scale_x != 0.0;          // 否则这一列已经是 Hessenberg 形，没有反射
        double h = 0.0, g = 0.0;
        if (reflect) {
            for (int i = n - 1; i >= m; --i) {
                ort[i] = x[i] / scale_x;
                h += std::norm(ort[i]);
            }
            g = std::sqrt(h);                         // ||u||
            const double f = std::abs(ort[m]);
            if (f == 0.0) {
                ort[m] = g;
                x[m] = scale_x;
            } else {
                h += f * g;
                g /= f;
                ort[m] *= 1.0 + g;
            }
        }
        const cplx* u = ort.data();
        const int len = n - m;
        const bool had_pending = pending;
        // 第 lo … hi-1 列：补上一步的右乘，再做本步的左乘（只动第 m 行以下）
        const auto columns = [&](int lo, int hi) {
            for (int j = hi - 1; j >= lo; --j) {
                cplx* aj = a.col(j);
                if (had_pending) axpy(aj, -std::conj(prev_u[j]), work.data(), n);
                if (reflect) axpy(aj + m, -dot_conj(u + m, aj + m, len) / h, u + m, len);
            }
        };
        const bool par = parts > 1 && 2LL * len * len + 2LL * n * len >= kHessStepWork;
        if (par) parallel_ranges(m, n, columns, 8);
        else columns(m, n);
        if (!reflect) {
            pending = false;
            continue;
        }
        // work = (P_m A)(:, m:) u / h：第 lo … hi-1 行，各列按 j 从大到小累加（与原版的次序一致）
        const auto rows = [&](int lo, int hi) {
            cplx* wk = work.data() + lo;
            std::fill(wk, wk + (hi - lo), cplx{});
            for (int j = n - 1; j >= m; --j) axpy(wk, u[j], a.col(j) + lo, hi - lo);
            for (int i = lo; i < hi; ++i) work[i] /= h;
        };
        if (par) parallel_ranges(0, n, rows, 64);
        else rows(0, n);
        std::copy(u + m, u + n, prev_u.begin() + m);
        pending = true;

        ort[m] *= scale_x;
        x[m] *= -g;                                   // 新的次对角元 h(m, m-1)
    }
    if (pending)                                      // 最后一步（m = n-2）的右乘
        for (int j = n - 2; j < n; ++j) axpy(a.col(j), -std::conj(prev_u[j]), work.data(), n);
}

// COMQR2 的第一段：Z = P_1 P_2 … P_{n-2}，从最后一个反射往回乘，每次只动右下角的块。
// ort 被当作工作数组改写。
//
// P_i = I − v v^H / h 里 v v^H 和 h 都是矩阵元素量级的平方，元素在 1e±154 之外时会上溢或下溢，
// 而且不报错：Z 悄悄地不再是酉矩阵。P_i 对 v 的整体缩放不变，所以先把 v 和 h(i,i-1) 乘同一个 2 的幂，
// 使 v 的最大分量 v_i 落在 1 附近（见上面“极端量级的处理”：正常量级下结果逐位不变）。
inline Matrix accumulate_reflectors(const Matrix& a, Vector& ort) {
    const int n = a.rows();
    Matrix z = Matrix::identity(n);
    for (int i = n - 2; i >= 1; --i) {
        if (ort[i] == cplx{} || a(i, i - 1) == cplx{}) continue;
        const int e = binary_exponent(abs1(ort[i]));
        const cplx sub = scale2(a(i, i - 1), -e);
        ort[i] = scale2(ort[i], -e);
        for (int k = i + 1; k < n; ++k) ort[k] = scale2(a(k, i - 1), -e);
        // 等于 −h（h 是 CORTH 里的 u^H u / 2，按现在的缩放），由存下来的量重新算出
        const double minus_h = sub.real() * ort[i].real() + sub.imag() * ort[i].imag();
        const cplx* v = ort.data() + i;
        const int len = n - i;
        for (int j = i; j < n; ++j) {
            cplx* zj = z.col(j) + i;
            axpy(zj, dot_conj(v, zj, len) / minus_h, v, len);
        }
    }
    return z;
}

// accumulate_reflectors 的分块版：Z 的各列互不相干（第 j 列只受 i <= j 的反射影响，且各列都是反射 i 从大到小
// 依次作用），所以改成列块在外层并行、反射在中间、块内的列在内层；每一列经历的反射及其次序不变。
// 每个列块自己把缩放后的反射向量算出来（与 accumulate_reflectors 用同一个 scale2，所以数值相同）；−h 先算好存起来。
// parts = 1 时就是一个列块，在调用线程上执行。
inline Matrix accumulate_reflectors_par(const Matrix& a, const Vector& ort, int parts) {
    const int n = a.rows();
    Matrix z = Matrix::identity(n);
    if (n < 3) return z;
    std::vector<int> exponent(n, 0);                  // 第 i 个反射的缩放指数
    std::vector<double> minus_h(n, 0.0);
    std::vector<char> used(n, 0);                     // 与 accumulate_reflectors 一样，退化的反射跳过
    for (int i = n - 2; i >= 1; --i) {
        if (ort[i] == cplx{} || a(i, i - 1) == cplx{}) continue;
        used[i] = 1;
        const int e = binary_exponent(abs1(ort[i]));
        const cplx sub = scale2(a(i, i - 1), -e);
        const cplx v0 = scale2(ort[i], -e);
        exponent[i] = e;
        minus_h[i] = sub.real() * v0.real() + sub.imag() * v0.imag();
    }
    const auto columns = [&](int lo, int hi) {        // 第 lo … hi-1 列
        Vector v(n);
        for (int i = (std::min)(hi - 1, n - 2); i >= 1; --i) {
            if (!used[i]) continue;
            const int e = exponent[i];
            v[0] = scale2(ort[i], -e);
            for (int k = i + 1; k < n; ++k) v[k - i] = scale2(a(k, i - 1), -e);
            const int len = n - i;
            for (int j = (std::max)(lo, i); j < hi; ++j) {
                cplx* zj = z.col(j) + i;
                axpy(zj, dot_conj(v.data(), zj, len) / minus_h[i], v.data(), len);
            }
        }
    };
    if (parts <= 1) {
        columns(0, n);
    } else {
        std::vector<int> cuts;                        // 第 j 列的工作量 ≈ Σ_{i<=j} (n-i)
        balanced_cuts(n, parts, [&](int j) { return static_cast<double>(j) * (2.0 * n - j); }, cuts);
        run_segments(parts, [&](int p) { columns(cuts[p], cuts[p + 1]); });
    }
    return z;
}

// 把次对角线以下清零（那里原先存着反射向量），之后 h 是干净的 Hessenberg 矩阵。
inline void clear_below_subdiagonal(Matrix& h) {
    const int n = h.rows();
    for (int j = 0; j + 2 < n; ++j) std::fill(h.col(j) + j + 2, h.col(j) + n, cplx{});
}

// 用对角酉矩阵 D 做相似变换 H ← D^H H D、Z ← Z D，使次对角元全为实数
// （原来是复数的变成它的模；原来就是实数的不动，可以为负）。之后 QR 迭代里的旋转只需要一个实的正弦。
inline void make_subdiagonal_real(Matrix& h, Matrix* z) {
    const int n = h.rows();
    for (int i = 1; i < n; ++i) {
        const cplx sub = h(i, i - 1);
        if (sub.imag() == 0.0) continue;
        double norm = 0.0;
        const cplx y = unit_phase(sub, norm);         // 单位模
        h(i, i - 1) = norm;
        for (int j = i; j < n; ++j) h(i, j) = std::conj(y) * h(i, j);      // 第 i 行乘 conj(y)
        scale(h.col(i), y, (std::min)(i + 1, n - 1) + 1);                  // 第 i 列乘 y
        if (z) scale(z->col(i), y, n);
    }
}

// 一个平面旋转：作用在相邻两行上的酉矩阵 G = [conj(c)  s; −s  c]，c 为复数、s 为实数，|c|² + s² = 1。
struct Rotation {
    double cr, ci, s;
};

// 由一列里的 (h(j,j), h(j+1,j)) 定出消去 h(j+1,j) 的旋转 G_{j+1}，并把这两个元素写成 (r, 0)，
// r = sqrt(|h(j,j)|² + h(j+1,j)²)。次对角元 h(j+1,j) 是实数，且在活动块内不为零，所以 r > 0。
// 极端量级下先缩放再平方（EISPACK 用嵌套两次的 pythag 达到同样目的）。c、s 也由缩放后的量算出：
// 这样即使三个数都是次正规数，|c|² + s² 仍等于 1，旋转仍是酉的。
inline Rotation make_rotation(cplx* col, int j) {
    double xr = col[j].real(), xi = col[j].imag(), sub = col[j + 1].real();
    const double m = (std::max)({std::abs(xr), std::abs(xi), std::abs(sub)});
    int e = 0;
    if (!safe_magnitude(m)) {
        e = binary_exponent(m);
        xr = std::ldexp(xr, -e);
        xi = std::ldexp(xi, -e);
        sub = std::ldexp(sub, -e);
    }
    const double norm = std::sqrt(xr * xr + xi * xi + sub * sub);
    col[j] = e ? std::ldexp(norm, e) : norm;
    col[j + 1] = 0.0;
    return Rotation{xr / norm, xi / norm, sub / norm};
}

// 把旋转 G_first … G_last 依次左乘到一列上（G_i 作用在第 i-1、i 行）。
// 第 i 行旋转后的值是下一个旋转的输入，沿列是一条依赖链，不能向量化。
inline void rotate_rows(cplx* col, const Rotation* rot, int first, int last) {
    if (first > last) return;
    double* p = reals(col);
    double yr = p[2 * first - 2], yi = p[2 * first - 1];
    for (int i = first; i <= last; ++i) {
        const Rotation g = rot[i];
        const double zr = p[2 * i], zi = p[2 * i + 1];
        p[2 * i - 2] = g.cr * yr + g.ci * yi + g.s * zr;
        p[2 * i - 1] = g.cr * yi - g.ci * yr + g.s * zi;
        const double tr = g.cr * zr - g.ci * zi - g.s * yr;
        const double ti = g.cr * zi + g.ci * zr - g.s * yi;
        yr = tr;
        yi = ti;
    }
    p[2 * last] = yr;
    p[2 * last + 1] = yi;
}

// 同上，两列一起做：两条依赖链互不相干，处理器可以交替执行，比一列一列做快。
inline void rotate_rows(cplx* col_a, cplx* col_b, const Rotation* rot, int first, int last) {
    if (first > last) return;
    double* p = reals(col_a);
    double* q = reals(col_b);
    double ar = p[2 * first - 2], ai = p[2 * first - 1];
    double br = q[2 * first - 2], bi = q[2 * first - 1];
    for (int i = first; i <= last; ++i) {
        const Rotation g = rot[i];
        const double zr = p[2 * i], zi = p[2 * i + 1];
        const double vr = q[2 * i], vi = q[2 * i + 1];
        p[2 * i - 2] = g.cr * ar + g.ci * ai + g.s * zr;
        p[2 * i - 1] = g.cr * ai - g.ci * ar + g.s * zi;
        q[2 * i - 2] = g.cr * br + g.ci * bi + g.s * vr;
        q[2 * i - 1] = g.cr * bi - g.ci * br + g.s * vi;
        const double tr = g.cr * zr - g.ci * zi - g.s * ar;
        const double ti = g.cr * zi + g.ci * zr - g.s * ai;
        const double ur = g.cr * vr - g.ci * vi - g.s * br;
        const double ui = g.cr * vi + g.ci * vr - g.s * bi;
        ar = tr;
        ai = ti;
        br = ur;
        bi = ui;
    }
    p[2 * last] = ar;
    p[2 * last + 1] = ai;
    q[2 * last] = br;
    q[2 * last + 1] = bi;
}

// 右乘 G^H：两列 (a, b) ← (c·a + s·b, conj(c)·b − s·a)。
inline void rotate_columns(cplx* ca, cplx* cb, int len, const Rotation& g) {
    double* a = reals(ca);
    double* b = reals(cb);
    const double cr = g.cr, ci = g.ci, s = g.s;
    for (int k = 0; k < 2 * len; k += 2) {
        const double yr = a[k], yi = a[k + 1], zr = b[k], zi = b[k + 1];
        a[k] = cr * yr - ci * yi + s * zr;
        a[k + 1] = cr * yi + ci * yr + s * zi;
        b[k] = cr * zr + ci * zi - s * yr;
        b[k + 1] = cr * zi - ci * zr - s * yi;
    }
}

// 位移：右下角 2×2 块 [a  b; c  d]（c 是实的次对角元）的两个本征值里离 d 近的那个，
//     s = d − x / (y + zz)，   x = b·c，   y = (a − d)/2，   zz = ±sqrt(y² + x)，
// zz 的符号取使 |y + zz| 较大的那个，避免相消。b = 0 时就是 d。
// x 和 y² 是矩阵元素量级的平方，元素在 1e±154 之外会上溢（位移变成 NaN，迭代不收敛）或下溢。
// 所以先把 b、c、y 乘同一个 2 的幂，使其中最大的落在 1 附近，算出修正量后再乘回去。
inline cplx wilkinson_shift(cplx a, cplx b, double c, cplx d) {
    const cplx y = (a - d) / 2.0;
    const int e = binary_exponent((std::max)({abs1(b), std::abs(c), abs1(y)}));
    const cplx x = scale2(b, -e) * std::ldexp(c, -e);
    if (x == cplx{}) return d;
    const cplx ys = scale2(y, -e);
    cplx zz = std::sqrt(ys * ys + x);
    if (ys.real() * zz.real() + ys.imag() * zz.imag() < 0.0) zz = -zz;
    return d - scale2(x / (ys + zz), e);
}

// 一个 QR 步的前半：活动块 [l, en] 内各列的左乘 Q^H = G_en … G_{l+1}，并定出旋转 G_{l+1} … G_en
// （写进 rot[l+1 … en]）。旋转 G_i 作用在第 i-1、i 两行上，消去 h(i, i-1)。
// EISPACK 逐对行地做；这里换成逐列推进——第 j 列先补上已经定下来的旋转 G_{l+1} … G_j，
// 这时 h(j,j)、h(j+1,j) 就绪，再定出 G_{j+1}。每个元素经历的运算和次序不变，但访问内存是连续的。
// 这是整个 QR 步里唯一严格串行的部分：G_{j+1} 依赖第 j 列做完前面全部旋转之后的值。
inline void qr_chain(Matrix& h, int l, int en, std::vector<Rotation>& rot) {
    const Rotation* g = rot.data();
    int j = l;
    for (; j < en; j += 2) {                          // 活动块内的列，两列一组
        cplx* a = h.col(j);
        cplx* b = h.col(j + 1);
        rotate_rows(a, b, g, l + 1, j);
        rot[j + 1] = make_rotation(a, j);
        rotate_rows(b, g, j + 1, j + 1);              // 第 j+1 列比第 j 列多一个旋转
        if (j + 1 < en) rot[j + 2] = make_rotation(b, j + 1);
    }
    if (j == en) rotate_rows(h.col(en), g, l + 1, en);
}

// 第 c0 … c1-1 列（都在活动块右边）补做旋转 G_{l+1} … G_en，两列一组，落单的一列单独做。
// 各列互不相干；单独做与两列一起做，对同一列是同样的算式、同样的次序。
inline void rotate_rows_columns(Matrix& h, const Rotation* rot, int l, int en, int c0, int c1) {
    int j = c0;
    for (; j + 1 < c1; j += 2) rotate_rows(h.col(j), h.col(j + 1), rot, l + 1, en);
    if (j < c1) rotate_rows(h.col(j), rot, l + 1, en);
}

// 对活动块 [l, en] 做一步 QR：H（对角元已减去位移）= Q R，H ← R Q，Z ← Z Q。
//
// full = true 时旋转作用到整行、整列（要得到完整的三角矩阵 T 才能回代求向量，COMQR2）；
// full = false 时只更新活动块本身，够求本征值（COMQR）。这一步之后活动块内的数值两种情况下完全相同；
// 活动块以外的元素 full = false 时不再更新，日后活动块的范围变动时会带着旧值进来，
// 所以两条路径最终的本征值可以差到舍入误差的量级（多数矩阵逐位相同）。
inline void qr_step(Matrix& h, Matrix* z, int l, int en, bool full, std::vector<Rotation>& rot) {
    const int n = h.rows();
    const int last_col = full ? n - 1 : en;
    const int first_row = full ? 0 : l;

    // (1) 左乘 Q^H：活动块内的列（链），再是活动块右边的列：全部旋转都已定好，照做即可。
    qr_chain(h, l, en, rot);
    rotate_rows_columns(h, rot.data(), l, en, en + 1, last_col + 1);

    // (2) 再用一个相位因子把 r(en, en) 转成实数，这样 R Q 的次对角元仍是实的。
    cplx phase{1.0, 0.0};
    const bool has_phase = h(en, en).imag() != 0.0;
    if (has_phase) {
        double norm = 0.0;
        phase = unit_phase(h(en, en), norm);
        h(en, en) = norm;
        for (int j = en + 1; j <= last_col; ++j) h(en, j) = std::conj(phase) * h(en, j);
    }

    // (3) 右乘 Q = G_{l+1}^H … G_en^H：G_j^H 作用在第 j-1、j 两列上，H 只需动到第 j 行。
    for (int j = l + 1; j <= en; ++j) {
        rotate_columns(h.col(j - 1) + first_row, h.col(j) + first_row, j - first_row + 1, rot[j]);
        if (z) rotate_columns(z->col(j - 1), z->col(j), n, rot[j]);
    }
    if (has_phase) {
        scale(h.col(en) + first_row, phase, en - first_row + 1);
        if (z) scale(z->col(en), phase, n);
    }
}

// ---------------------------------------------------------------------------------------------
// 并行版的 QR 迭代（只用于 full = true）。
//
// 一个 sweep 里真正串行的只有链（qr_chain，占单线程总用时的 8%）；其余都能拆开：
//   · 活动块右边的各列补做左乘：列与列互不相干，只依赖本 sweep 的旋转；而且这些列以后不会再回到活动块里
//     （列号 > en），所以可以推迟，与之后的链同时做；
//   · H 的第 0 … en 行的右乘 Q：行与行互不相干；下一个 sweep 的链要读它，所以必须在下一个链之前做完；
//   · Z 的右乘：行与行互不相干，而且与 H 无关，所以可以先把旋转记下来，攒够一批再按行块施加。
// 每个元素经历的运算及其次序都不变（见上面“并行的组织方式”）。
//
// 每个 sweep 用两个并行区：
//   R 区：第 0 段（调用线程）做本 sweep 的链与相位，其余各段做之前攒下的“活动块右边各列”和 Z 的行；
//   S 区：全部线程做 H 的第 0 … en 行的右乘。
// R 区里两边碰的元素互不相交：链只碰第 l … en 列，而队列里右边各列的列号 > 它们各自的 en >= 本 sweep 的 en；
// Z 与 H 无关。S 区在 R 区之后、下一个链之前，所以活动块的右乘照串行版“全部左乘之后、下一个 sweep 之前”的次序完成。

// 一个 sweep 的旋转 G_{l+1} … G_en 与末尾的相位因子，记下来供之后施加。
struct SweepRecord {
    int l, en;
    bool has_phase;
    cplx phase;
    std::size_t offset;                               // G_{l+1+k} 是 SweepLog::rots[offset + k]
};

// 一批 sweep 的记录，按先后次序排列。
struct SweepLog {
    std::vector<SweepRecord> sweeps;
    std::vector<Rotation> rots;
    long long work = 0;                               // Σ (en − l)：Z 的一行要做的旋转总数

    bool empty() const { return sweeps.empty(); }
    int size() const { return static_cast<int>(sweeps.size()); }
    const Rotation* rotations(const SweepRecord& s) const { return rots.data() + s.offset; }
    void add(int l, int en, bool has_phase, cplx phase, const Rotation* rot) {
        sweeps.push_back({l, en, has_phase, phase, rots.size()});
        rots.insert(rots.end(), rot + l + 1, rot + en + 1);
        work += en - l;
    }
    void clear() {
        sweeps.clear();
        rots.clear();
        work = 0;
    }
};

// 记下的各个 sweep 对第 c0 … c1-1 列补做左乘与相位；只动列号 > en 的列。c0 必须是偶数：列按 (2m, 2m+1) 两两成组，
// 一组的两列都在活动块右边时用两列版内核，只有第二列在时用单列版——用哪个内核只由列号和 en 定，与怎么切段无关。
// 列在外层、sweep 在内层，这样每一组列读进缓存后可以连用若干次。各列互不相干；单列版与两列版对同一列是同样的算式、同样的次序。
inline void apply_log_to_columns(Matrix& h, const SweepLog& log, int c0, int c1) {
    int c = c0;
    for (; c + 1 < c1; c += 2)
        for (const SweepRecord& s : log.sweeps) {
            const Rotation* g = log.rotations(s);
            const int last = s.en - s.l - 1;
            if (c > s.en) {
                rotate_rows(h.col(c) + (s.l + 1), h.col(c + 1) + (s.l + 1), g, 0, last);
                if (s.has_phase) {
                    const cplx conj_phase = std::conj(s.phase);
                    h(s.en, c) = conj_phase * h(s.en, c);
                    h(s.en, c + 1) = conj_phase * h(s.en, c + 1);
                }
            } else if (c + 1 > s.en) {                // 只有第二列在活动块右边
                rotate_rows(h.col(c + 1) + (s.l + 1), g, 0, last);
                if (s.has_phase) h(s.en, c + 1) = std::conj(s.phase) * h(s.en, c + 1);
            }
        }
    if (c < c1)
        for (const SweepRecord& s : log.sweeps) {
            if (c <= s.en) continue;
            rotate_rows(h.col(c) + (s.l + 1), log.rotations(s), 0, s.en - s.l - 1);
            if (s.has_phase) h(s.en, c) = std::conj(s.phase) * h(s.en, c);
        }
}

// 记下的各个 sweep 对 Z 的第 [r0, r1) 行补做右乘，按“波前”的次序：第 s 个 sweep 的旋转 G_j 排在 τ = j + 2s 时刻，
// 相位紧跟在 G_en 之后。依赖检查：旋转 (s, j) 碰第 j-1、j 列，它之前必须做完 (s, j-1)（τ 小 1）和 (s-1, j+1)
// （τ 小 1）；同一 τ 上不同 s 的旋转碰的列互不相交，所以每个元素的运算次序与逐个 sweep 对整列做时相同。
// 好处是这几行的“热”列只有约 2·(批内 sweep 数) 列，留在 L1 里；Z 每一批只从内存读写一遍。
inline void apply_wave_to_rows(Matrix& z, const SweepLog& log, int r0, int r1) {
    const int len = r1 - r0;
    if (len <= 0) return;
    const int count = log.size();
    int tau_min = (std::numeric_limits<int>::max)(), tau_max = -1;
    for (int s = 0; s < count; ++s) {
        tau_min = (std::min)(tau_min, log.sweeps[s].l + 1 + 2 * s);
        tau_max = (std::max)(tau_max, log.sweeps[s].en + 2 * s);
    }
    for (int tau = tau_min; tau <= tau_max; ++tau)
        for (int s = 0; s < count; ++s) {
            const SweepRecord& sw = log.sweeps[s];
            const int j = tau - 2 * s;
            if (j < sw.l + 1 || j > sw.en) continue;
            rotate_columns(z.col(j - 1) + r0, z.col(j) + r0, len, log.rotations(sw)[j - (sw.l + 1)]);
            if (j == sw.en && sw.has_phase) scale(z.col(sw.en) + r0, sw.phase, len);
        }
}

// 同上，每 kWaveRows 行一个小块，一块一块地做。
inline void apply_wave_to_row_blocks(Matrix& z, const SweepLog& log, int r0, int r1) {
    for (int r = r0; r < r1; r += kWaveRows) apply_wave_to_rows(z, log, r, (std::min)(r1, r + kWaveRows));
}

// H 的第 [r0, r1) 行（r1 <= en+1）右乘 G_{l+1}^H … G_en^H，再乘相位。第 i 行受旋转 j（j >= i）的影响。
inline void rotate_h_rows(Matrix& h, const Rotation* rot, int l, int en, bool has_phase, cplx phase, int r0, int r1) {
    if (r0 >= r1) return;
    for (int j = (std::max)(l + 1, r0); j <= en; ++j)
        rotate_columns(h.col(j - 1) + r0, h.col(j) + r0, (std::min)(r1, j + 1) - r0, rot[j]);
    if (has_phase) scale(h.col(en) + r0, phase, r1 - r0);
}

class ParallelQr {
public:
    // rot 是 n 个旋转的工作区（与串行版的 qr_step 共用）。
    ParallelQr(Matrix& h, Matrix& z, std::vector<Rotation>& rot, int parts) : h_(h), z_(z), rot_(rot), parts_(parts) {}

    // 对活动块 [l, en] 做一步 QR（full = true 的 qr_step）；右边各列与 Z 的更新可能推迟。
    void sweep(int l, int en) {
        const int n = h_.rows();
        cplx phase{1.0, 0.0};
        bool has_phase = false;
        const auto chain = [&] {
            qr_chain(h_, l, en, rot_);
            has_phase = h_(en, en).imag() != 0.0;
            if (has_phase) {
                double norm = 0.0;
                phase = unit_phase(h_(en, en), norm);
                h_(en, en) = norm;
            }
        };

        // R 区。Z 的一批分摊到随后的约 kZBatch 个 R 区里做，每个 R 区做其中的一段行，使“链”与“队列”的工作量相当。
        const int z_rows = wave_active() ? (std::min)(n - wave_next_, (n + kZBatch - 1) / kZBatch) : 0;
        const long long queue_work = cols_work_ + static_cast<long long>(z_rows) * wave_.work;
        if (queue_work >= kQueueWork) {
            // 队列的工作分给 parts_ 段（parts_ = 1 时全归第 0 段，在链之后依次做）。第 0 段先要做链，所以只分到做完链之后
            // “多出来的”那一份（链比平均每段的工作量还长时一份也没有），其余各段均分剩下的。
            // 这里只是估计，估错了只影响各段谁先做完，不影响结果。
            const double chain_cost = kChainCost * (static_cast<double>(en - l) * (en - l) / 2.0);
            const double queue_cost = static_cast<double>(queue_work);
            const double share0 = (std::min)(queue_cost, (std::max)(0.0, (chain_cost + queue_cost) / parts_ - chain_cost));
            const double f0 = queue_work > 0 ? share0 / queue_cost : 0.0;
            fractions_.assign(parts_ + 1, 1.0);
            fractions_[0] = 0.0;
            for (int p = 1; p < parts_; ++p) fractions_[p] = f0 + (1.0 - f0) * (p - 1) / (parts_ - 1);
            if (!cols_.empty()) column_cuts();
            const int z0 = wave_next_;
            run_segments(parts_, [&](int p) {
                if (p == 0) chain();
                if (!cols_.empty()) apply_log_to_columns(h_, cols_, col_cuts_[p], col_cuts_[p + 1]);
                apply_wave_to_row_blocks(z_, wave_, z0 + static_cast<int>(z_rows * fractions_[p]), z0 + static_cast<int>(z_rows * fractions_[p + 1]));
            });
            wave_next_ += z_rows;
            clear_cols();
        } else {
            chain();
        }

        // S 区：H 的第 0 … en 行的右乘。第 i 行的工作量：i < l 时 en-l 个旋转；l <= i <= en 时 en - max(i, l+1) + 1 个。
        const long long rows_work = static_cast<long long>(l) * (en - l) + static_cast<long long>(en - l) * (en - l + 3) / 2;
        if (parts_ > 1 && rows_work >= kRowsWork) {
            balanced_cuts(en + 1, parts_, [&](int i) { return static_cast<double>(i < l ? en - l : en - (std::max)(i, l + 1) + 1); }, row_cuts_);
            run_segments(parts_, [&](int p) { rotate_h_rows(h_, rot_.data(), l, en, has_phase, phase, row_cuts_[p], row_cuts_[p + 1]); });
        } else {
            rotate_h_rows(h_, rot_.data(), l, en, has_phase, phase, 0, en + 1);
        }

        // 入队：右边各列的左乘进 cols_，Z 的右乘进 fill_
        if (en < n - 1) {
            cols_.add(l, en, has_phase, phase, rot_.data());
            cols_work_ += static_cast<long long>(n - 1 - en) * (en - l);
            cols_min_en_ = (std::min)(cols_min_en_, en);
        }
        fill_.add(l, en, has_phase, phase, rot_.data());
        if (fill_.size() >= kZBatch) {
            finish_wave();                            // 新的一批开始之前，上一批必须做完（每一行按批次顺序经历全部旋转）
            std::swap(wave_, fill_);
            fill_.clear();
            wave_next_ = 0;
        }
    }

    // 把推迟的工作全部做完（要用到 H 的右边各列或 Z 之前调用）。
    void flush() {
        finish_wave();
        wave_.clear();
        if (!fill_.empty()) {
            std::swap(wave_, fill_);
            fill_.clear();
            wave_next_ = 0;
            finish_wave();
            wave_.clear();
        }
        if (!cols_.empty()) {
            const int segments = segments_for(cols_work_);
            fractions_.resize(segments + 1);
            for (int p = 0; p <= segments; ++p) fractions_[p] = static_cast<double>(p) / segments;
            column_cuts();
            run_segments(segments, [&](int p) { apply_log_to_columns(h_, cols_, col_cuts_[p], col_cuts_[p + 1]); });
            clear_cols();
        }
    }

private:
    bool wave_active() const { return !wave_.empty() && wave_next_ < h_.rows(); }

    void clear_cols() {
        cols_.clear();
        cols_work_ = 0;
        cols_min_en_ = (std::numeric_limits<int>::max)();
    }

    int segments_for(long long work) const {
        return static_cast<int>((std::min<long long>)(parts_, (std::max<long long>)(1, work / kSegmentWork)));
    }

    // wave_ 剩下的行（wave_next_ … n-1）用全部线程做完。
    void finish_wave() {
        if (!wave_active()) return;
        const int n = h_.rows();
        const int r0 = wave_next_;
        const int segments = segments_for(static_cast<long long>(n - r0) * wave_.work);
        run_segments(segments, [&](int p) {
            apply_wave_to_row_blocks(z_, wave_, r0 + static_cast<int>(static_cast<long long>(n - r0) * p / segments),
                                     r0 + static_cast<int>(static_cast<long long>(n - r0) * (p + 1) / segments));
        });
        wave_next_ = n;
    }

    // 把 cols_ 里待做的列按工作量切成 fractions_.size()-1 段：第 p 段是 [col_cuts_[p], col_cuts_[p+1])，
    // 前 p 段的工作量约占 fractions_[p]。第 c 列的工作量 = Σ_{en_s < c} (en_s − l_s)。
    void column_cuts() {
        const int n = h_.rows();
        const int c_begin = (std::min)(cols_min_en_ + 1, n);
        col_weight_.assign(n - c_begin + 1, 0.0);
        for (const SweepRecord& s : cols_.sweeps) col_weight_[s.en + 1 - c_begin] += s.en - s.l;
        for (int c = 1; c < n - c_begin; ++c) col_weight_[c] += col_weight_[c - 1];
        const int segments = static_cast<int>(fractions_.size()) - 1;
        cuts_by_weight(n - c_begin, segments, [&](int i) { return col_weight_[i]; }, [&](int p) { return fractions_[p]; }, col_cuts_);
        for (int p = 0; p <= segments; ++p) col_cuts_[p] += c_begin;
        for (int p = 0; p < segments; ++p) col_cuts_[p] &= ~1;        // 段的起点取偶数，不拆开 (2m, 2m+1) 一组
    }

    Matrix& h_;
    Matrix& z_;
    std::vector<Rotation>& rot_;
    int parts_;
    SweepLog cols_;                                   // 活动块右边各列待补的左乘
    long long cols_work_ = 0;                         // 它们的工作量 Σ (n-1-en)(en-l)
    int cols_min_en_ = (std::numeric_limits<int>::max)();
    SweepLog fill_;                                   // Z：正在攒的一批
    SweepLog wave_;                                   // Z：攒满了、正在分摊到各个 R 区里施加的一批
    int wave_next_ = 0;                               // wave_ 已经做完的行数
    std::vector<int> row_cuts_, col_cuts_;
    std::vector<double> col_weight_, fractions_;
};

// ---------------------------------------------------------------------------------------------
// 回代求上三角矩阵 T 的本征向量并乘回 Z：V 的第 j 列 = Z·x_j，其中 (T − λ_j I) x_j = 0、x_j(j) = 1，
// 最后把每列归一到 2-范数为 1。
//
// 逐行的公式（EISPACK）是 x(i) = [ Σ_{k>i} T(i,k) x(k) ] / (λ_j − λ_i)，i = j-1 … 0；
// 这里把求和改成按列累加：x(i) 一算出来就把 T(0:i-1, i)·x(i) 加到上面各行的累加量里，内存访问连续。
// |x(i)| 过大时把整个向量缩小，防止溢出；分母 λ_j − λ_i 恰好为零时换成 eps·norm（这两条照 EISPACK）。
// 另加一条 EISPACK 没有的保护：分母不为零、但模小于 eps·max(|λ_i|, |λ_j|) 时，换成这个值。
// 两个本征值在舍入误差以内重合时，算出来的差可以是 1e-30 这样非零但毫无意义的数，原样去除会把两个
// 本征向量算得几乎平行；替换之后得到的是这个二维不变子空间里两个线性无关的向量。阈值按这一对本征值
// 自身的大小取（LAPACK 的做法），所以量级悬殊的矩阵里小块的本征向量不受影响。
// 这是 eig_qr 与 COMQR2 在数值上唯一的出入；eig_lr 的回代加了同一条保护，两种迭代的回代仍然逐行相同。
// 另外，分母不为零但小到相除会上溢时，也先把向量缩小，免得出现 Inf/NaN。
// 各列互不相关。每 kBlock 列一组同时做，T 和 Z 的每一列读进缓存后可以连用 kBlock 次；
// 每一列自己的运算次序与逐列做完全一样。组与组互不相干，所以并行版把组分给不同的线程。

constexpr int kBlock = 4;                             // 回代时同时做的列数

// 过小分母的替换量：rel = max(eps, 机器精度)，两个本征值相对相差不到它就算在舍入误差以内重合；
// tiny = rel·norm，norm 是 T 上三角部分的最大 |Re| + |Im|。T 为零矩阵（或 eps·norm 下溢）时 tiny 取最小的正规数，
// 这样 0/tiny = 0，不会出现 0/0。
inline void schur_tolerances(const Matrix& t, double eps, double& rel, double& tiny) {
    const int n = t.rows();
    double norm = 0.0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i <= j; ++i) norm = (std::max)(norm, abs1(t(i, j)));
    rel = (std::max)(eps, kEps);
    tiny = rel * norm;
    if (tiny == 0.0) tiny = (std::numeric_limits<double>::min)();
}

// 回代的第 g0 … g1-1 组（第 g 组是第 g·kBlock … 列），结果写进 v 的对应各列。T、w、Z 只读，
// 每次调用自带一份工作区，所以各组可以分给不同的线程。
inline void schur_vector_groups(const Matrix& t, const Vector& w, const Matrix& z, double rel, double tiny, Matrix& v, int g0, int g1) {
    const int n = t.rows();
    constexpr double kQuotientMax = 1e300;            // 商的 |Re| + |Im| 不超过它就不会上溢
    constexpr double kShrink = 0x1p-512;              // 2^-512 ≈ 7.5e-155；2 的幂，乘它不引入舍入
    Matrix xs(n, kBlock);        // 一组里各列的 x_j：下标 > i 的部分是已求出的解，<= i 的部分是累加量
    for (int g = g0; g < g1; ++g) {
        const int j0 = g * kBlock;
        const int j1 = (std::min)(n, j0 + kBlock);
        for (int j = j0; j < j1; ++j) {
            cplx* x = xs.col(j - j0);
            std::copy(t.col(j), t.col(j) + j, x);
            x[j] = 1.0;
        }
        // 回代
        for (int i = j1 - 2; i >= 0; --i) {
            const cplx* ti = t.col(i);
            for (int j = (std::max)(j0, i + 1); j < j1; ++j) {
                cplx* x = xs.col(j - j0);
                cplx y = w[j] - w[i];
                if (y == cplx{}) {
                    y = tiny;
                } else {
                    const double floor_value = rel * (std::max)(abs1(w[j]), abs1(w[i]));    // （不叫 small：<windows.h> 把它定义成了宏）
                    if (abs1(y) < floor_value) y = floor_value;
                }
                // 分母不为零但小到商会上溢（只有它是次正规数量级时才可能）：先把整个向量缩小。
                // 本征向量只差一个常数因子，缩小不改变结果；不处理的话这里得到 Inf，随后变成 NaN。
                for (int pass = 0; pass < 3 && abs1(x[i]) > abs1(y) * kQuotientMax; ++pass)
                    for (int k = 0; k <= j; ++k) x[k] *= kShrink;
                x[i] /= y;
                const double tr = abs1(x[i]);
                if (tr != 0.0 && tr + 1.0 / tr <= tr)             // 溢出控制
                    for (int k = 0; k <= j; ++k) x[k] /= tr;
                axpy(x, x[i], ti, i);
            }
        }
        // 乘回 Z：v_j = Σ_{k<=j} Z(:, k) x_j(k)
        for (int k = 0; k < j1; ++k) {
            const cplx* zk = z.col(k);
            for (int j = (std::max)(j0, k); j < j1; ++j) axpy(v.col(j), xs(k, j - j0), zk, n);
        }
        // 归一：先除以最大分量再求平方和，避免溢出
        for (int j = j0; j < j1; ++j) {
            cplx* vj = v.col(j);
            double big = 0.0;
            for (int i = 0; i < n; ++i) big = (std::max)(big, (std::max)(std::abs(vj[i].real()), std::abs(vj[i].imag())));
            if (big == 0.0 || !std::isfinite(big)) continue;
            double ssq = 0.0;
            for (int i = 0; i < n; ++i) ssq += std::norm(vj[i] / big);
            const double inv = 1.0 / std::sqrt(ssq);
            for (int i = 0; i < n; ++i) vj[i] = (vj[i] / big) * inv;
        }
    }
}

// 求全部本征向量（串行）。
inline Matrix eigenvectors_from_schur(const Matrix& t, const Vector& w, const Matrix& z, double eps) {
    const int n = t.rows();
    double rel, tiny;
    schur_tolerances(t, eps, rel, tiny);
    Matrix v(n, n);
    schur_vector_groups(t, w, z, rel, tiny, v, 0, (n + kBlock - 1) / kBlock);
    return v;
}

// 同上，各组分给线程池的 parts 个线程；每一组的计算与 eigenvectors_from_schur 里的相同（同一个 schur_vector_groups）。
// 第 g 组的工作量 ≈ 回代 j²/2 + 乘回 Z 的 j·n，随 g 增大；先按工作量切成 4·parts 段，再交错分给各线程，
// 免得最重的几段落在同一个线程上。
inline Matrix eigenvectors_from_schur_par(const Matrix& t, const Vector& w, const Matrix& z, double eps, int parts) {
    const int n = t.rows();
    const int groups = (n + kBlock - 1) / kBlock;
    double rel, tiny;
    schur_tolerances(t, eps, rel, tiny);
    Matrix v(n, n);
    const int segments = 4 * parts;
    std::vector<int> cuts;
    balanced_cuts(groups, segments, [&](int g) {
        const double j0 = static_cast<double>(g) * kBlock + 0.5 * kBlock;
        return j0 * j0 / 2 + j0 * n + n;
    }, cuts);
    run_segments(parts, [&](int th) {
        for (int p = th; p < segments; p += parts) schur_vector_groups(t, w, z, rel, tiny, v, cuts[p], cuts[p + 1]);
    });
    return v;
}

}  // namespace eigqr
}  // namespace detail

// 一般复矩阵的本征值与右本征向量（EISPACK CORTH + COMQR2）。a 不被修改。
//
//   options.vectors         是否求本征向量（否则只算本征值，迭代只更新活动块，快得多）
//   options.keep_transform  为真时返回酉矩阵 Z 与上三角矩阵 T，Z^H A Z = T
//   options.eps             次对角元可忽略的相对阈值；回代里过小分母的替换量也由它定：max(eps, 机器精度)·norm
//   options.max_iter        每个本征值的迭代次数上限；已迭代 10、20 次仍未收敛时各用一次例外位移
//   options.observer        见 eig_types.hpp；每次回调时 Z^H A Z = H + shift·I 对整个矩阵成立。
//                           回调总在调用 eig_qr 的线程上发生，不在 eig_qr 自己开的并行区里（回调里可以再调用 parallel_for）；
//                           若 eig_qr 本身是在别的并行区里调用的（parallel_for 的循环体、parallel_layers、sweep），回调仍在那个
//                           并行区之内，回调里不能调用 set_num_threads（parallel.hpp 的约定：并行区内改线程数会死锁）；
//                           分块版每次回调之前先把推迟的更新做完，所以回调时看到的 H、Z 与串行版相同
//                           （逐位相同的前提见文件开头“保证的范围”；有观察者时会慢一些）
//
// 返回值的 info = 0 表示全部收敛。info = k > 0 表示第 k 个本征值（1 起）没有收敛：此时
// values 里下标 >= k 的是已收敛的本征值，下标 < k 的只是当前对角元（近似值）；不给本征向量；
// keep_transform 时 Z、T 仍满足 Z^H A Z = T，T 的前 k 阶还是 Hessenberg 形、其余各行已是上三角；
// 也不会有 "done" 回调。
// 上限是按每个本征值计的（默认 30 次）。高重数的亏损本征值（例如三阶例外点）收敛很慢，
// 偶尔要三十多次；遇到 info > 0 时把 max_iter 调大重算即可。
//
// 非方阵、含 NaN/Inf 的矩阵、eps < 0、max_iter < 1 抛 std::invalid_argument。
inline EigResult eig_qr(const Matrix& a, const EigOptions& options = {}) {
    namespace q = detail::eigqr;
    if (!a.square()) throw std::invalid_argument("eig_qr: matrix must be square");
    if (!(options.eps >= 0.0) || !std::isfinite(options.eps)) throw std::invalid_argument("eig_qr: eps must be finite and >= 0");
    if (options.max_iter < 1) throw std::invalid_argument("eig_qr: max_iter must be >= 1");
    if (!a.all_finite()) throw std::invalid_argument("eig_qr: matrix has non-finite entries");

    const int n = a.rows();
    const double eps = options.eps;
    const bool observed = static_cast<bool>(options.observer);
    // 需要完整的 T 和 Z 的三种情况；否则只在活动块里迭代，也不形成 Z。
    const bool full = options.vectors || options.keep_transform || observed;
    const bool blocked = n >= q::kParallelMinN;      // 走分块版还是原样保留的串行写法：只看 n，与线程数无关
    const int parts = blocked ? q::available_parts() : 1;   // 分块版里各个并行区最多分给几个线程（1：全在调用线程上做）

    EigResult result;
    Vector& w = result.values;
    w.assign(n, cplx{});

    // 1. 化 Hessenberg 形，累乘 Z，次对角元化实。
    Matrix h = a;
    Matrix z;
    {
        Vector ort(n);
        if (blocked) q::reduce_to_hessenberg_par(h, ort, parts);
        else q::reduce_to_hessenberg(h, ort);
        if (full) z = blocked ? q::accumulate_reflectors_par(h, ort, parts) : q::accumulate_reflectors(h, ort);
    }
    q::clear_below_subdiagonal(h);
    q::make_subdiagonal_real(h, full ? &z : nullptr);

    cplx t{};                    // 累计位移：活动部分存的是 H − t·I
    int en = n - 1;              // 正在求第 en 个本征值；en 以下的行已收敛
    int its = 0;                 // 当前本征值已迭代的次数

    // 分块版（full 时）：每个 sweep 的离链工作由 qr 对象分给线程，其中右边各列与 Z 的更新推迟；
    // 凡是要用到 Z（或整个 H）的地方，先调用 flush 补齐。
    std::vector<q::Rotation> rot(n);
    const bool blocked_qr = blocked && full;
    q::ParallelQr qr(h, z, rot, parts);
    const auto flush_z = [&] { if (blocked_qr) qr.flush(); };

    // 回调观察者。已收敛的对角元平时存的是加回位移后的 λ_i，
    // 回调期间临时换成 λ_i − t，让 Z^H A Z = H + t·I 对整个矩阵成立。
    const auto notify = [&](const char* stage, int active) {
        flush_z();
        for (int i = active + 1; i < n; ++i) h(i, i) = w[i] - t;
        options.observer(EigProgress{stage, result.sweeps, active, t, h, z});
        for (int i = active + 1; i < n; ++i) h(i, i) = w[i];
    };
    if (observed) notify("hessenberg", en);

    // 2. 带位移的 QR 迭代，从右下角开始逐个收缩。
    while (en >= 0) {
        // 自下而上找第一个可忽略的次对角元 h(l, l-1)；找不到则 l = 0。活动块是 [l, en]。
        int l = en;
        for (; l > 0; --l)
            if (abs1(h(l, l - 1)) <= eps * (abs1(h(l - 1, l - 1)) + abs1(h(l, l)))) break;

        if (l == en) {                               // 1×1 块：收敛出一个本征值
            w[en] = h(en, en) + t;
            h(en, en) = w[en];
            --en;
            its = 0;
            continue;
        }
        if (its == options.max_iter) {               // 未收敛
            result.info = en + 1;
            break;
        }

        // 位移 s：右下角 2×2 块的两个本征值里离 h(en,en) 近的那个；
        // 第 10、20 次换成例外位移，打破可能出现的循环。
        cplx s;
        if (its == 10 || its == 20) {
            s = std::abs(h(en, en - 1).real()) + (en >= 2 ? std::abs(h(en - 1, en - 2).real()) : 0.0);
        } else {
            s = q::wilkinson_shift(h(en - 1, en - 1), h(en - 1, en), h(en, en - 1).real(), h(en, en));
        }
        for (int i = 0; i <= en; ++i) h(i, i) -= s;
        t += s;
        ++its;
        ++result.sweeps;

        if (blocked_qr) qr.sweep(l, en);
        else q::qr_step(h, full ? &z : nullptr, l, en, full, rot);
        if (observed) notify("iteration", en);
    }
    flush_z();                                       // 迭代结束（含未收敛）：Z 要补齐

    if (result.info != 0) {
        // 未收敛的部分：把位移加回对角元，给出当前的近似值。
        for (int i = 0; i <= en; ++i) {
            h(i, i) += t;
            w[i] = h(i, i);
        }
        // 已收敛的部分：和全部收敛时一样，把可忽略的次对角元清零，这几行就是严格的上三角。
        for (int j = en; j + 1 < n; ++j) h(j + 1, j) = 0.0;
        if (options.keep_transform) {
            result.Z = std::move(z);
            result.T = std::move(h);
        }
        return result;
    }

    // 3. 已收敛：次对角线上剩下的都是可忽略的量，清零后 h 就是上三角的 T。
    for (int j = 0; j + 1 < n; ++j) h(j + 1, j) = 0.0;
    if (observed) notify("done", -1);

    if (options.vectors)
        result.vectors = blocked ? q::eigenvectors_from_schur_par(h, w, z, eps, parts) : q::eigenvectors_from_schur(h, w, z, eps);
    if (options.keep_transform) {
        result.Z = std::move(z);
        result.T = std::move(h);
    }
    return result;
}

}  // namespace stablercwa
