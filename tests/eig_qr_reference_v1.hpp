// tests/eig_qr_reference_v1.hpp —— eig_qr 的冻结副本（并行化之前的内部开发版本中的 eig_qr.hpp，即现在的 include/stablercwa/eig_qr.hpp 的前身，除命名空间外一个字不改）。
// 只供测试用：名字空间 stablercwa::ref_v1，使它能和 include/ 里正在被并行化的 stablercwa::eig_qr 同一个翻译单元、同一个编译器、
// 同一组选项下逐位比较。**不要修改，不要放进 include/**；eig_qr 的并行化以与它逐位相同为硬性验收。
// 下面是原文件（命名空间之外的改动只有两处：#include 路径、`detail::eigqr` 的限定名）。

// stablesolver/eig_qr.hpp —— 一般复矩阵的本征分解：酉变换化 Hessenberg 形 + 带位移的 QR 迭代
//
//   #include "stablesolver/eig_qr.hpp"
//   stablesolver::EigResult r = stablesolver::eig_qr(A);      // A 是任意 n×n 复矩阵
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
// 本文件只用 C++17 标准库，单线程，可重入（没有全局状态）。
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "stablercwa/eig_types.hpp"

namespace stablercwa {
namespace ref_v1 {
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

    // (1) 左乘 Q^H = G_en … G_{l+1}：旋转 G_i 作用在第 i-1、i 两行上，消去 h(i, i-1)。
    //     EISPACK 逐对行地做；这里换成逐列推进——第 j 列先补上已经定下来的旋转 G_{l+1} … G_j，
    //     这时 h(j,j)、h(j+1,j) 就绪，再定出 G_{j+1}。每个元素经历的运算和次序不变，但访问内存是连续的。
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
    //     活动块右边的列：全部旋转都已定好，照做即可。
    for (j = en + 1; j < last_col; j += 2) rotate_rows(h.col(j), h.col(j + 1), g, l + 1, en);
    if (j == last_col) rotate_rows(h.col(j), g, l + 1, en);

    // (2) 再用一个相位因子把 r(en, en) 转成实数，这样 R Q 的次对角元仍是实的。
    cplx phase{1.0, 0.0};
    const bool has_phase = h(en, en).imag() != 0.0;
    if (has_phase) {
        double norm = 0.0;
        phase = unit_phase(h(en, en), norm);
        h(en, en) = norm;
        for (j = en + 1; j <= last_col; ++j) h(en, j) = std::conj(phase) * h(en, j);
    }

    // (3) 右乘 Q = G_{l+1}^H … G_en^H：G_j^H 作用在第 j-1、j 两列上，H 只需动到第 j 行。
    for (j = l + 1; j <= en; ++j) {
        rotate_columns(h.col(j - 1) + first_row, h.col(j) + first_row, j - first_row + 1, rot[j]);
        if (z) rotate_columns(z->col(j - 1), z->col(j), n, rot[j]);
    }
    if (has_phase) {
        scale(h.col(en) + first_row, phase, en - first_row + 1);
        if (z) scale(z->col(en), phase, n);
    }
}

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
// 每一列自己的运算次序与逐列做完全一样。
inline Matrix eigenvectors_from_schur(const Matrix& t, const Vector& w, const Matrix& z, double eps) {
    const int n = t.rows();
    double norm = 0.0;                                // 上三角部分的最大 |Re| + |Im|
    for (int j = 0; j < n; ++j)
        for (int i = 0; i <= j; ++i) norm = (std::max)(norm, abs1(t(i, j)));
    // 过小分母的替换量。T 为零矩阵（或 eps·norm 下溢）时取最小的正规数，这样 0/tiny = 0，不会出现 0/0。
    const double rel = (std::max)(eps, kEps);         // 两个本征值相对相差不到它，就算在舍入误差以内重合
    double tiny = rel * norm;
    if (tiny == 0.0) tiny = (std::numeric_limits<double>::min)();

    constexpr int kBlock = 4;
    constexpr double kQuotientMax = 1e300;            // 商的 |Re| + |Im| 不超过它就不会上溢
    constexpr double kShrink = 0x1p-512;              // 2^-512 ≈ 7.5e-155；2 的幂，乘它不引入舍入
    Matrix v(n, n);
    Matrix xs(n, kBlock);        // 一组里各列的 x_j：下标 > i 的部分是已求出的解，<= i 的部分是累加量
    for (int j0 = 0; j0 < n; j0 += kBlock) {
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
                    const double small = rel * (std::max)(abs1(w[j]), abs1(w[i]));
                    if (abs1(y) < small) y = small;
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
//   options.observer        见 eig_types.hpp；每次回调时 Z^H A Z = H + shift·I 对整个矩阵成立
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
    namespace q = ref_v1::detail::eigqr;
    if (!a.square()) throw std::invalid_argument("eig_qr: matrix must be square");
    if (!(options.eps >= 0.0) || !std::isfinite(options.eps)) throw std::invalid_argument("eig_qr: eps must be finite and >= 0");
    if (options.max_iter < 1) throw std::invalid_argument("eig_qr: max_iter must be >= 1");
    if (!a.all_finite()) throw std::invalid_argument("eig_qr: matrix has non-finite entries");

    const int n = a.rows();
    const double eps = options.eps;
    const bool observed = static_cast<bool>(options.observer);
    // 需要完整的 T 和 Z 的三种情况；否则只在活动块里迭代，也不形成 Z。
    const bool full = options.vectors || options.keep_transform || observed;

    EigResult result;
    Vector& w = result.values;
    w.assign(n, cplx{});

    // 1. 化 Hessenberg 形，累乘 Z，次对角元化实。
    Matrix h = a;
    Matrix z;
    {
        Vector ort(n);
        q::reduce_to_hessenberg(h, ort);
        if (full) z = q::accumulate_reflectors(h, ort);
    }
    q::clear_below_subdiagonal(h);
    q::make_subdiagonal_real(h, full ? &z : nullptr);

    cplx t{};                    // 累计位移：活动部分存的是 H − t·I
    int en = n - 1;              // 正在求第 en 个本征值；en 以下的行已收敛
    int its = 0;                 // 当前本征值已迭代的次数

    // 回调观察者。已收敛的对角元平时存的是加回位移后的 λ_i，
    // 回调期间临时换成 λ_i − t，让 Z^H A Z = H + t·I 对整个矩阵成立。
    const auto notify = [&](const char* stage, int active) {
        for (int i = active + 1; i < n; ++i) h(i, i) = w[i] - t;
        options.observer(EigProgress{stage, result.sweeps, active, t, h, z});
        for (int i = active + 1; i < n; ++i) h(i, i) = w[i];
    };
    if (observed) notify("hessenberg", en);

    // 2. 带位移的 QR 迭代，从右下角开始逐个收缩。
    std::vector<q::Rotation> rot(n);
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

        q::qr_step(h, full ? &z : nullptr, l, en, full, rot);
        if (observed) notify("iteration", en);
    }

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

    if (options.vectors) result.vectors = q::eigenvectors_from_schur(h, w, z, eps);
    if (options.keep_transform) {
        result.Z = std::move(z);
        result.T = std::move(h);
    }
    return result;
}

}  // namespace ref_v1
}  // namespace stablercwa
