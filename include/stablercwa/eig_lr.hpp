// stablercwa/eig_lr.hpp —— 一般复矩阵的本征分解：消元变换化 Hessenberg 形 + 带位移的 LR 迭代
//
//   #include "stablercwa/eig_lr.hpp"
//   stablercwa::EigResult r = stablercwa::eig_lr(A);      // A 是任意 n×n 复矩阵
//   r.values[j]、r.vectors 的第 j 列：A v_j = λ_j v_j，||v_j||_2 = 1
//
// 算法与 EISPACK 的 COMHES + COMLR2 相同（不做平衡，low = 0、igh = n-1）：
//   1. COMHES：n-2 步带行、列交换的高斯消元（初等相似变换），把 A 化成上 Hessenberg 形 H；
//   2. 由留下的乘子和交换记录拼出累积变换 Z（此时 A Z = Z H）；
//   3. COMLR2：对未收敛的前 en+1 阶反复做带位移的 LR 步
//        H − s·I = L R，  H ← R L   （L 由相邻两行之间的消元组成，消元前按需交换这两行；同时 Z ← Z L），
//      次对角元 |h(l,l-1)| <= eps·(|h(l-1,l-1)| + |h(l,l)|) 时视为零，逐个收缩出本征值；
//   4. 迭代结束得到 Z^{-1} A Z = T（上三角），回代求 T 的本征向量，再乘回 Z。
// 全程只用初等消元：每个乘子的模不超过 √2，但变换不是酉的，许多步累乘起来的 Z 可以非常病态，
// 舍入误差随之被放大。这是它与 eig_qr（酉变换）唯一的差别，也是 LR 方法不稳定的来源；
// 本文件不添加任何额外的稳定化措施。
//
// 选主元规则、位移公式、“两个相邻小次对角元”的判断、第 10/20 次迭代的例外位移、回代里的溢出控制
// 都照 EISPACK；只是循环改成按列访问内存（各元素经历的运算及其次序不变）。
// 这一点是可以检验的：不让编译器把 a*b+c 合并成一条指令时，H、Z、T、本征值、迭代次数和归一之前的
// 本征向量都与 EISPACK 原文按行循环的写法逐位相同（第 0 个本征向量除外，见 eigenvectors_from_triangular
// 的说明）。改动内层循环时请保持每个元素的运算次序。
// “可忽略”的判据写成显式的 eps 形式，迭代次数按每个本征值单独计数，零分母换成 eps·norm，
// 这三处用的是 EISPACK 较早版本（带 MACHEP 参数）的写法，便于把 eps、max_iter 做成选项。
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <utility>
#include <vector>

#include "eig_types.hpp"

namespace stablercwa {
namespace detail {
namespace eiglr {

// 把复数数组当作 (re, im) 交错的 double 数组读写（标准保证 std::complex<double> 的数组可以这样访问）。
// 内层循环手写实部、虚部，是为了避开 std::complex 乘法里的 NaN 检查，运算次序也与 EISPACK 的写法一致。
inline double* reals(cplx* p) { return reinterpret_cast<double*>(p); }
inline const double* reals(const cplx* p) { return reinterpret_cast<const double*>(p); }

// EISPACK 的 CDIV：a / b，先用 |Re b| + |Im b| 缩放，避免中间量上溢。
inline cplx cdiv(const cplx& a, const cplx& b) {
    const double s = std::abs(b.real()) + std::abs(b.imag());
    const double ars = a.real() / s, ais = a.imag() / s;
    const double brs = b.real() / s, bis = b.imag() / s;
    const double d = brs * brs + bis * bis;
    return {(ars * brs + ais * bis) / d, (ais * brs - ars * bis) / d};
}

// EISPACK 的 CSROOT：复数平方根，取实部 >= 0 的一支（虚部与被开方数的虚部同号）。
inline cplx csroot(double xr, double xi) {
    double s = std::sqrt(0.5 * (std::hypot(xr, xi) + std::abs(xr)));
    double yr = 0.0, yi = 0.0;
    if (xr >= 0.0) yr = s;
    if (xi < 0.0) s = -s;
    if (xr <= 0.0) yi = s;
    if (xr < 0.0) yr = 0.5 * (xi / yi);
    if (xr > 0.0) yi = 0.5 * (xi / yr);
    return {yr, yi};
}

// y ← y + t·x
inline void axpy(cplx* cy, const cplx& t, const cplx* cx, int len) {
    double* y = reals(cy);
    const double* x = reals(cx);
    const double tr = t.real(), ti = t.imag();
    for (int k = 0; k < 2 * len; k += 2) {
        const double xr = x[k], xi = x[k + 1];
        y[k] = y[k] + tr * xr - ti * xi;
        y[k + 1] = y[k + 1] + tr * xi + ti * xr;
    }
}

// y ← y + x·t。数学上与 axpy 相同，只是虚部两项的相加次序不同：这是 COMLR2 最后一段
// （本征向量乘回 Z）的写法，照它写，舍入也与 EISPACK 的逐行公式一致。
inline void axpy_back(cplx* cy, const cplx* cx, const cplx& t, int len) {
    double* y = reals(cy);
    const double* x = reals(cx);
    const double tr = t.real(), ti = t.imag();
    for (int k = 0; k < 2 * len; k += 2) {
        const double xr = x[k], xi = x[k + 1];
        y[k] = y[k] + xr * tr - xi * ti;
        y[k + 1] = y[k + 1] + xr * ti + xi * tr;
    }
}

// 先交换两列再做 a ← a + t·b，合成一趟：(a, b) ← (b + t·a, a)
inline void swap_axpy(cplx* ca, cplx* cb, const cplx& t, int len) {
    double* a = reals(ca);
    double* b = reals(cb);
    const double tr = t.real(), ti = t.imag();
    for (int k = 0; k < 2 * len; k += 2) {
        const double ar = a[k], ai = a[k + 1];
        a[k] = b[k] + tr * ar - ti * ai;
        a[k + 1] = b[k + 1] + tr * ai + ti * ar;
        b[k] = ar;
        b[k + 1] = ai;
    }
}

// y(k) ← y(k) − m(k)·s：同一列里的各行各减去“本行乘子 × 主元行的元素 s”
inline void sub_scaled(cplx* cy, const cplx* cm, const cplx& s, int len) {
    double* y = reals(cy);
    const double* m = reals(cm);
    const double sr = s.real(), si = s.imag();
    for (int k = 0; k < 2 * len; k += 2) {
        const double mr = m[k], mi = m[k + 1];
        y[k] = y[k] - mr * sr + mi * si;
        y[k + 1] = y[k + 1] - mr * si - mi * sr;
    }
}

// ---------------------------------------------------------------------------------------------
// COMHES：带行、列交换的高斯消元化上 Hessenberg 形。
//
// 第 m 步（m = 1 … n-2）处理第 m-1 列：
//   · 在 a(m:n-1, m-1) 里找 |Re| + |Im| 最大的元素作主元 x，把它所在的行、列与第 m 行、列交换
//     （行列同时交换才是相似变换），交换的行号记在 perm[m]；
//   · 乘子 y_i = a(i, m-1) / x（i > m，模不超过 √2）留在被消去的位置上；
//   · 相似变换 A ← L^{-1} A L，L = I + Σ_i y_i e_i e_m^T：
//       左乘 L^{-1}：第 i 行减去 y_i × 第 m 行；右乘 L：第 m 列加上 Σ_i y_i × 第 i 列。
// 返回时 a 的上 Hessenberg 部分就是 H，次对角线以下是各步的乘子。
//
// EISPACK 对每个 i 先做“行 i 的消元”再做“列 m 的累加”。这里把行消元拆到各列上去做：
// 对第 i 列，先补上第 m+1 … i 行的消元，然后累加到第 m 列，最后补上第 i+1 … n-1 行的消元。
// 每个元素经历的运算和次序与 EISPACK 相同，但内层循环都沿着列走。
inline void reduce_to_hessenberg(Matrix& a, std::vector<int>& perm) {
    const int n = a.rows();
    for (int m = 1; m < n - 1; ++m) {
        cplx* y = a.col(m - 1);                       // 这一列对角线以下先是待消去的元素，随后存乘子
        cplx x{};
        int p = m;
        for (int j = m; j < n; ++j) {
            if (abs1(y[j]) > abs1(x)) {
                x = y[j];
                p = j;
            }
        }
        perm[m] = p;
        if (p != m) {
            for (int j = m - 1; j < n; ++j) std::swap(a(p, j), a(m, j));
            std::swap_ranges(a.col(p), a.col(p) + n, a.col(m));
        }
        if (x == cplx{}) continue;                    // 这一列已经是 Hessenberg 形

        for (int i = m + 1; i < n; ++i)
            if (y[i] != cplx{}) y[i] = cdiv(y[i], x);

        cplx* am = a.col(m);
        for (int i = m + 1; i < n; ++i) {
            cplx* ai = a.col(i);
            const cplx pivot_row = ai[m];             // a(m, i)，这一步里不变
            sub_scaled(ai + m + 1, y + m + 1, pivot_row, i - m);            // 第 m+1 … i 行的消元
            if (y[i] != cplx{}) {
                const cplx yi = y[i];
                const cplx amm = am[m];
                am[i] = cplx(am[i].real() - yi.real() * amm.real() + yi.imag() * amm.imag(),
                             am[i].imag() - yi.real() * amm.imag() - yi.imag() * amm.real());
                axpy(am, yi, ai, n);                                        // 第 m 列 += y_i × 第 i 列
            }
            sub_scaled(ai + i + 1, y + i + 1, pivot_row, n - 1 - i);        // 第 i+1 … n-1 行的消元
        }
    }
}

// COMLR2 的第一段：由 COMHES 留下的乘子和交换记录拼出累积变换 Z，使 A Z = Z H。
// 从最后一步往回做：先把第 i 步的乘子放进第 i 列，再补上第 i 步的行交换。
inline Matrix accumulate_eliminations(const Matrix& a, const std::vector<int>& perm) {
    const int n = a.rows();
    Matrix z = Matrix::identity(n);
    for (int i = n - 2; i >= 1; --i) {
        for (int k = i + 1; k < n; ++k) z(k, i) = a(k, i - 1);
        const int j = perm[i];
        if (j == i) continue;
        for (int k = i; k < n; ++k) {
            z(i, k) = z(j, k);
            z(j, k) = 0.0;
        }
        z(j, i) = 1.0;
    }
    return z;
}

// 把次对角线以下清零（那里原先存着乘子），之后 h 是干净的 Hessenberg 矩阵。
inline void clear_below_subdiagonal(Matrix& h) {
    const int n = h.rows();
    for (int j = 0; j + 2 < n; ++j) std::fill(h.col(j) + j + 2, h.col(j) + n, cplx{});
}

// 一步 LR 留下的记录：第 i 行消元用的乘子，以及消元前第 i-1、i 两行是否交换过。
struct Elimination {
    cplx multiplier;
    bool swapped = false;
};

// 对第 m … en 行做一步 LR：H（对角元已减去位移）= L R，H ← R L，Z ← Z L。
//
// full = true 时行变换一直做到最后一列（要得到完整的三角矩阵 T 才能回代求向量，COMLR2）；
// full = false 时只做到第 en 列，够求本征值，前 en+1 阶主子阵里的数值与 full = true 时完全相同。
inline void lr_step(Matrix& h, Matrix* z, int m, int en, bool full, std::vector<Elimination>& elim) {
    const int n = h.rows();
    const int last_col = full ? n - 1 : en;

    // (1) 分解 H = L R：对 i = m+1 … en，比较 h(i-1,i-1) 与 h(i,i-1)，把 |Re|+|Im| 大的换到上面作主元，
    //     再用乘子 zz（模不超过 √2）消去下面那个。EISPACK 逐对行地做；这里换成逐列推进——
    //     对第 j 列先补上已经定下来的第 m+1 … j 行的变换，这时 h(j,j)、h(j+1,j) 就绪，再定出第 j+1 行的变换。
    //     每个元素经历的运算和次序不变，但访问内存是连续的。
    for (int j = m; j <= last_col; ++j) {
        double* p = reals(h.col(j));
        const int last = (std::min)(j, en);
        double ur = p[2 * m], ui = p[2 * m + 1];      // 上一行经过变换后的值，沿列往下传
        for (int i = m + 1; i <= last; ++i) {
            double vr = p[2 * i], vi = p[2 * i + 1];
            if (elim[i].swapped) {
                std::swap(ur, vr);
                std::swap(ui, vi);
            }
            const double zr = elim[i].multiplier.real(), zi = elim[i].multiplier.imag();
            p[2 * i - 2] = ur;
            p[2 * i - 1] = ui;
            const double tr = vr - zr * ur + zi * ui;
            const double ti = vi - zr * ui - zi * ur;
            ur = tr;
            ui = ti;
        }
        p[2 * last] = ur;
        p[2 * last + 1] = ui;
        if (j < en) {
            const cplx x{ur, ui};                     // h(j, j)
            const cplx y{p[2 * j + 2], p[2 * j + 3]}; // h(j+1, j)
            Elimination& e = elim[j + 1];
            if (abs1(x) >= abs1(y)) {
                e.multiplier = cdiv(y, x);
                e.swapped = false;
            } else {
                p[2 * j] = y.real();                  // 交换两行：大的做主元
                p[2 * j + 1] = y.imag();
                e.multiplier = cdiv(x, y);
                e.swapped = true;
            }
            p[2 * j + 2] = 0.0;                       // 被消去；乘子另存在 elim 里
            p[2 * j + 3] = 0.0;
        }
    }

    // (2) 乘回 R L：第 j-1、j 两列先按同样的记录交换，再把 乘子 × 第 j 列 加到第 j-1 列；H 只需动到第 j 行。
    //     同样的列变换累积到 Z 上。
    for (int j = m + 1; j <= en; ++j) {
        const Elimination& e = elim[j];
        if (e.swapped) {
            swap_axpy(h.col(j - 1), h.col(j), e.multiplier, j + 1);
            if (z) swap_axpy(z->col(j - 1), z->col(j), e.multiplier, n);
        } else {
            axpy(h.col(j - 1), e.multiplier, h.col(j), j + 1);
            if (z) axpy(z->col(j - 1), e.multiplier, z->col(j), n);
        }
    }
}

// 回代求上三角矩阵 T 的本征向量并乘回 Z：V 的第 j 列 = Z·x_j，其中 (T − λ_j I) x_j = 0、x_j(j) = 1，
// 最后把每列归一到 2-范数为 1。
//
// 逐行的公式（EISPACK）是 x(i) = [ Σ_{k=i+1…j} T(i,k) x(k) ] / (λ_j − λ_i)，i = j-1 … 0。
// 求和要沿 T 的行取数，所以先转置一份（tt 的第 i 列就是 T 的第 i 行），求和次序不变而访问内存连续。
// 分母为零（重本征值）时换成 eps·norm；|x(i)| 过大时把已求出的部分整体缩小，防止溢出。
// T 是零矩阵时（A = 0）不回代，直接返回 Z 的各列。第 0 列的 x_0 = e_0，所以就是 Z 的第 0 列
// （EISPACK 在这一列上多乘了一个标量，归一之后只差一个相位）。
// 各列互不相关，可以按列并行，结果与线程数无关。
inline Matrix eigenvectors_from_triangular(const Matrix& t, const Vector& w, const Matrix& z, double eps) {
    const int n = t.rows();
    double norm = 0.0;                                // 上三角部分的最大 |Re| + |Im|
    for (int j = 0; j < n; ++j)
        for (int i = 0; i <= j; ++i) norm = (std::max)(norm, abs1(t(i, j)));
    const bool trivial = (n <= 1 || norm == 0.0);
    const double rel = (std::max)(eps, kEps);
    const double tiny = rel * norm;
    const Matrix tt = trivial ? Matrix() : transpose(t);

    Matrix v(n, n);
    const auto block = [&](int lo, int hi) {
        Vector x(n);
        for (int j = lo; j < hi; ++j) {
            cplx* vj = v.col(j);
            if (trivial || j == 0) {
                std::copy(z.col(j), z.col(j) + n, vj);            // T 的第一个本征向量是 e_0
            } else {
                x[j] = 1.0;
                for (int i = j - 1; i >= 0; --i) {
                    const double* row = reals(tt.col(i));         // row[2k], row[2k+1] = T(i, k)
                    const double* xs = reals(x.data());
                    double zr = 0.0, zi = 0.0;
                    for (int k = i + 1; k <= j; ++k) {
                        zr = zr + row[2 * k] * xs[2 * k] - row[2 * k + 1] * xs[2 * k + 1];
                        zi = zi + row[2 * k] * xs[2 * k + 1] + row[2 * k + 1] * xs[2 * k];
                    }
                    cplx y{w[j].real() - w[i].real(), w[j].imag() - w[i].imag()};
                    if (y == cplx{}) {
                        y = tiny;
                    } else {                                      // 舍入误差以内重合的一对本征值，见 eig_qr.hpp 同一处的说明
                        const double small = rel * (std::max)(abs1(w[j]), abs1(w[i]));
                        if (abs1(y) < small) y = small;
                    }
                    x[i] = cdiv(cplx{zr, zi}, y);
                    const double tr = abs1(x[i]);
                    if (tr != 0.0 && !(tr + 1.0 / tr > tr))       // 溢出控制：1/tr 相对 tr 已可忽略
                        for (int k = i; k <= j; ++k) x[k] = cplx(x[k].real() / tr, x[k].imag() / tr);
                }
                for (int k = 0; k <= j; ++k) axpy_back(vj, z.col(k), x[k], n);
            }
            // 归一：先除以最大分量再求平方和，防止平方时溢出
            double big = 0.0;
            for (int i = 0; i < n; ++i) big = (std::max)(big, (std::max)(std::abs(vj[i].real()), std::abs(vj[i].imag())));
            if (big == 0.0 || !std::isfinite(big)) continue;
            double ssq = 0.0;
            for (int i = 0; i < n; ++i) {
                const double re = vj[i].real() / big, im = vj[i].imag() / big;
                ssq += re * re + im * im;
            }
            const double len = std::sqrt(ssq);
            if (!(len > 0.0) || !std::isfinite(len)) continue;
            for (int i = 0; i < n; ++i) vj[i] = cplx(vj[i].real() / big / len, vj[i].imag() / big / len);
        }
    };
    if (n < 64) block(0, n);
    else parallel_ranges(0, n, block, 8);
    return v;
}

}  // namespace eiglr
}  // namespace detail

// 一般复矩阵的本征值与右本征向量（EISPACK COMHES + COMLR2）。a 不被修改。
//
//   options.vectors         是否求本征向量（否则只算本征值：不形成 Z，行变换只做到活动块的右边界）
//   options.keep_transform  为真时返回累积变换 Z 与上三角矩阵 T，Z^{-1} A Z = T。Z 不是酉矩阵，
//                           它的条件数越大，T 的本征值、回代出的本征向量里被放大的舍入误差就越多
//   options.eps             次对角元可忽略的相对阈值；也用作回代里零分母的替换量 eps·norm
//   options.max_iter        每个本征值的迭代次数上限；已迭代 10、20 次仍未收敛时各用一次例外位移
//   options.observer        见 eig_types.hpp；每次回调时 Z^{-1} A Z = H + shift·I 对整个矩阵成立
//
// vectors、keep_transform、observer 怎样设置都不影响本征值和迭代次数（逐位相同）。
//
// 返回值的 info = 0 表示全部收敛。info = k > 0 表示第 k 个本征值（1 起）没有收敛：此时
// values 里下标 >= k 的是已收敛的本征值，下标 < k 的只是当前对角元（近似值）；不给本征向量；
// keep_transform 时 Z、T 仍满足 Z^{-1} A Z = T，只是 T 的前 k 阶还是 Hessenberg 形；也不会有 "done" 回调。
//
// 非方阵、含 NaN/Inf 的矩阵、eps < 0、max_iter < 1 抛 std::invalid_argument。
//
// 与 EISPACK 一样不做缩放，所以对元素的量级有要求：位移公式里要算两个元素的乘积，元素的模超过
// 约 1e150 时乘积上溢成 NaN，此后收敛判据永远不成立，结果是 info > 0（迭代照常在 max_iter 次内停下）。
// 同一个矩阵里元素量级相差 300 个数量级以上、或元素贴近双精度上限时，回代里的商也会上溢，
// 这时 info = 0 而本征向量里有 NaN/Inf。遇到这类矩阵请先整体乘一个常数再调用。
inline EigResult eig_lr(const Matrix& a, const EigOptions& options = {}) {
    namespace lr = detail::eiglr;
    if (!a.square()) throw std::invalid_argument("eig_lr: matrix must be square");
    if (!(options.eps >= 0.0) || !std::isfinite(options.eps)) throw std::invalid_argument("eig_lr: eps must be finite and >= 0");
    if (options.max_iter < 1) throw std::invalid_argument("eig_lr: max_iter must be >= 1");
    if (!a.all_finite()) throw std::invalid_argument("eig_lr: matrix has non-finite entries");

    const int n = a.rows();
    const double eps = options.eps;
    const bool observed = static_cast<bool>(options.observer);
    // 需要完整的 T 和 Z 的三种情况；否则不形成 Z，行变换也只做到第 en 列。
    const bool full = options.vectors || options.keep_transform || observed;

    EigResult result;
    Vector& w = result.values;
    w.assign(n, cplx{});

    // 1. 化 Hessenberg 形，拼出累积变换 Z。
    Matrix h = a;
    Matrix z;
    {
        std::vector<int> perm(n, 0);
        lr::reduce_to_hessenberg(h, perm);
        if (full) z = lr::accumulate_eliminations(h, perm);
    }
    lr::clear_below_subdiagonal(h);

    cplx t{};                    // 累计位移：活动部分存的是 H − t·I
    int en = n - 1;              // 正在求第 en 个本征值；en 以下的行已收敛
    int its = 0;                 // 当前本征值已迭代的次数

    // 回调观察者。已收敛的对角元平时存的是加回位移后的 λ_i，
    // 回调期间临时换成 λ_i − t，让 Z^{-1} A Z = H + t·I 对整个矩阵成立。
    const auto notify = [&](const char* stage, int active) {
        for (int i = active + 1; i < n; ++i) h(i, i) = w[i] - t;
        options.observer(EigProgress{stage, result.sweeps, active, t, h, z});
        for (int i = active + 1; i < n; ++i) h(i, i) = w[i];
    };
    if (observed) notify("hessenberg", en);

    // 2. 带位移的 LR 迭代，从右下角开始逐个收缩。
    std::vector<lr::Elimination> elim(n);
    while (en >= 0) {
        // 自下而上找第一个可忽略的次对角元 h(l, l-1)；找不到则 l = 0。活动块是 [l, en]。
        int l = en;
        for (; l > 0; --l)
            if (abs1(h(l, l - 1)) <= eps * (abs1(h(l - 1, l - 1)) + abs1(h(l, l)))) break;

        if (l == en) {                               // 1×1 块：收敛出一个本征值
            h(en, en) += t;
            w[en] = h(en, en);
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
        double sr, si;
        if (its == 10 || its == 20) {
            // en = 1 时没有 h(en-1, en-2) 这一项（EISPACK 在这里读到数组之外），取 0。
            const cplx sub1 = h(en, en - 1);
            const cplx sub2 = en >= 2 ? h(en - 1, en - 2) : cplx{};
            sr = std::abs(sub1.real()) + std::abs(sub2.real());
            si = std::abs(sub1.imag()) + std::abs(sub2.imag());
        } else {
            sr = h(en, en).real();
            si = h(en, en).imag();
            const cplx u = h(en - 1, en), v = h(en, en - 1);
            const double xr = u.real() * v.real() - u.imag() * v.imag();
            const double xi = u.real() * v.imag() + u.imag() * v.real();
            if (xr != 0.0 || xi != 0.0) {
                const double yr = (h(en - 1, en - 1).real() - sr) / 2.0;
                const double yi = (h(en - 1, en - 1).imag() - si) / 2.0;
                cplx zz = lr::csroot(yr * yr - yi * yi + xr, 2.0 * yr * yi + xi);
                if (yr * zz.real() + yi * zz.imag() < 0.0) zz = -zz;   // 取模较大的 y + zz，避免相消
                const cplx d = lr::cdiv(cplx{xr, xi}, cplx{yr + zz.real(), yi + zz.imag()});
                sr -= d.real();
                si -= d.imag();
            }
        }
        const cplx s{sr, si};
        for (int i = 0; i <= en; ++i) h(i, i) -= s;
        t += s;
        ++its;
        ++result.sweeps;

        // 找两个相邻的小次对角元：若 h(m, m-1) 与 h(m+1, m) 的乘积相对可忽略，
        // 这一步从第 m 行开始分解即可（m 取尽量靠下的位置，找不到则 m = l）。
        double xr = abs1(h(en - 1, en - 1));
        double yr = abs1(h(en, en - 1));
        double zzr = abs1(h(en, en));
        int m = en - 1;
        for (; m > l; --m) {
            const double yi = yr;
            yr = abs1(h(m, m - 1));
            const double xi = zzr;
            zzr = xr;
            xr = abs1(h(m - 1, m - 1));
            if (yr <= eps * zzr / yi * (zzr + xr + xi)) break;
        }

        lr::lr_step(h, full ? &z : nullptr, m, en, full, elim);
        if (observed) notify("iteration", en);
    }

    if (result.info != 0) {
        // 未收敛的部分：把位移加回对角元，给出当前的近似值。
        for (int i = 0; i <= en; ++i) {
            h(i, i) += t;
            w[i] = h(i, i);
        }
        if (options.keep_transform) {
            result.Z = std::move(z);
            result.T = std::move(h);
        }
        return result;
    }

    // 3. 已收敛：次对角线上剩下的都是可忽略的量，清零后 h 就是上三角的 T。
    for (int j = 0; j + 1 < n; ++j) h(j + 1, j) = 0.0;
    if (observed) notify("done", -1);

    if (options.vectors) result.vectors = lr::eigenvectors_from_triangular(h, w, z, eps);
    if (options.keep_transform) {
        result.Z = std::move(z);
        result.T = std::move(h);
    }
    return result;
}

}  // namespace stablercwa
