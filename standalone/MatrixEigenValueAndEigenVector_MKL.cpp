// MatrixEigenValueAndEigenVector_MKL.cpp —— 一般实矩阵、复矩阵的本征值与右本征向量，交给 Intel oneMKL 的 LAPACK（zgeev / dgeev）
//
// 直接加进 DLL 项目、替换原来的同名函数即可。本文件对外只定义下面两个函数，其余都在内部名字空间里（内部链接）：
//
//   int MatrixEigenValueAndEigenVector(double** A1, std::complex<double>* EigenValue, double** EigenVector, int mn);
//   int MatrixEigenValueAndEigenVector(std::complex<double>** A1, std::complex<double>* EigenValue,
//                                      std::complex<double>** EigenVector, int mn);
//
// 本文件自成一体：单独加进项目、链接 oneMKL 即可，不需要 StableRCWA 的其他文件。用法、需要的 MKL 版本和
// 一键下载运行库的脚本（install_mkl.bat）见同目录的 README.md。
//
// 约定（两个函数相同）
//   · mn 是阶数 n。A1、EigenVector 各是 n 个行指针、每行 n 个元素；EigenValue 有 n 个元素。内存都由调用方分配。
//   · A1[i][j] 是第 i 行第 j 列。A1 不被修改。输出可以与 A1 共用存储（A1 先整个读进来，算完才写输出）。
//   · 第 k 个本征向量放在第 k 列：EigenVector[i][k] 是它的第 i 个分量。A v_k = λ_k v_k，||v_k||_2 = 1。
//   · 本征值按实部从大到小排列，实部相同时按虚部从大到小（实矩阵版本另有一条，见下）；本征向量的列跟着同样排列。
//   · 每个本征向量乘了一个单位模的因子，使原来模最大的分量（并列时取行号小的）成为正实数
//     （这个因子的模只在舍入意义下为 1，所以别的分量的模偶尔会比它大 1 ulp）。
//
// 实矩阵版本：EigenVector 是 double**，复本征值的向量按 LAPACK dgeev 的格式打包——
//   · λ_k 是实数（EigenValue[k] 的虚部恰为 0）：第 k 列就是实的本征向量，模最大的分量为正；
//   · λ_k、λ_{k+1} 是一对共轭复数（虚部为正的在前，两者严格共轭，总是相邻）：
//         第 k 列 = x，第 k+1 列 = y，    λ_k 的本征向量是 x + i·y，λ_{k+1} 的是 x − i·y，||x + i·y||_2 = 1；
//   · 排序以“一个实本征值”或“一对共轭本征值”为单位：实部大的在前，实部相同时虚部（取正的那个）大的在前，
//     所以一对共轭本征值不会被别的本征值隔开。这一条代替上面的“实部相同时按虚部从大到小”：
//     实部相同的几个本征值会排成 (+b, −b, 0) 或 (+b, −b, +c, −c)（b > c）这样。
//
// 返回值
//    0  成功
//   -1  参数不合法：mn < 1，或某个指针为空（此时不碰任何输出）
//   -2  A1 里有 NaN 或 Inf
//   -3  内存不足（或其他意外的异常）
//   -4  算出的本征值或本征向量里有 NaN/Inf（只在元素量级极端、超出算法能处理的范围时出现）
//   k>0 迭代未收敛：LAPACK 的 QR 迭代没有收敛（k 是 LAPACK 返回的 info）
// 返回 -2、-3、-4、k>0 时 EigenValue、EigenVector 全部填成 NaN。
//
// 算法：本征分解交给 MKL 的 LAPACK：矩阵的元素虚部全为 0 时（实数版总是如此）用 dgeev，否则用 zgeev。两者都是
// 分块的 Householder 化 Hessenberg 形、多位移 QR 加提前收缩、分块回代，在 MKL 里多线程执行。之后的排序、相位规范、
// 实矩阵的打包由本文件自己完成。返回 k > 0 表示 LAPACK 的 QR 迭代没有收敛；意外的错误返回 −3。
// 与 StableRCWA 的 eig_qr（只依赖标准库的 QR 迭代）相比：
//   · 快：用时见同目录 README.md。
//   · 结果与 eig_qr 只在舍入误差以内一致，不逐位相同。MKL 默认在不同线程数、不同处理器上的结果也不逐位相同；要可重复，
//     设环境变量 MKL_CBWR=AVX2（或在程序第一次调用 MKL 之前调用 mkl_cbwr_set），并把 MATRIX_EIG_THREADS
//     设成固定的 n（默认的 0 在有别的调用同时在算时改用 1 个线程，线程数随之变化）。
//   · 实矩阵走 dgeev，算的是实 Schur 形：实本征值的虚部严格为 0，共轭对严格共轭。下面那段配对的代码照样走一遍，
//     这时它只是把 dgeev 已经给好的结构认出来；把实矩阵当复矩阵求解时的两条局限（三重以上亏损实根的残差、
//     同一实本征值上两个亏损块，见下）在这里没有。
//   · MKL 的版本要求：2025.0.1 或更新。实测（AMD Ryzen 9 5950X，Windows）2024.2.2 与 2025.0.0 的 dgeev 有毛病：
//     不要左本征向量（JOBVL = 'N'）时仍往 VL 里写 n² 个数，还写出它自己报的工作区的末尾；单线程时 LAPACKE_dgeev 还会
//     卡死。2025.0.1、2025.1.0、2025.2.0、2025.3.1、2026.2.0 都没有（对应 2025.0.1 的发布说明“修正了 AMD 硬件、
//     Windows 上若干 BLAS 函数的问题”）。zgeev 在这些版本上都没有问题。所以本文件在第一次调用时读 MKL 的版本字符串，
//     构建日期早于 2025.0.1（20241031）或读不出来时，实矩阵也改走 zgeev（当作复矩阵求解再整理成实的形式，
//     这时有下面说的两条局限）。
//
// 下面这段说的是 dgeev 不可用、实矩阵走 zgeev 的情形：实矩阵当作复矩阵求解，然后整理成实的形式。复数算法算出的实本征值带着舍入量级的虚部，共轭对的两半也只是
// 近似共轭，所以要先配对：d(i, j) = |λ_i − conj(λ_j)|（对称，d(i, i) = 2|Im λ_i|），每个 λ_i 找 d 最小的 j；
// 最近的是自己就是实本征值，两个互为最近的配成一对。配对本身不用人为设的阈值。配好以后：
//   · 实本征值取实部，向量取（转相位之后的）实部再归一；
//   · 一对取虚部为正的那一半 λ 与它的向量 v，输出 λ、conj(λ) 与 Re v、Im v；
//   · 但重复的实本征值被算成虚部极小的一对、两半的向量又都几乎是实的时候，x ± i·y 两列几乎线性相关，
//     改成两个实本征值、各用各的向量（条件见 split_pair：两半都几乎是实向量，且拆开后的向量组比成对时更好）。
// 取实部丢掉的量是 |Im λ|·|Im v|（两个小量之积），所以一般矩阵、重复的实本征值、接近亏损的二重实根，残差都保持在
// 机器精度。重复的实本征值可能被标成虚部极小的共轭对，反过来也一样；LAPACK dgeev 在这种情形下同样会这样标。
// 局限（只在实矩阵走 zgeev 时；复数版本没有这两条，实矩阵也可以转成复矩阵调用复数版本）：
//   · 三重以上的亏损实根在复数算法里散成 eps^(1/3) 量级的一簇，不再近似共轭对称，配对只能就近，
//     取实之后残差可能到 1e-12 量级（(x−1)³(x+2) 的友矩阵实测 9e-13）；实数双位移 QR（LAPACK dgeev）这时仍是机器精度。
//   · 同一个实本征值上有两个以上亏损块时（例如 S·(J2(1)⊕J2(1)⊕[−1])·S⁻¹），向量组本来就病态到 1e8 左右；
//     实数版约千分之二的情形会变成数值上奇异（cond 约 1e16），复数版不会。残差仍在机器精度。
//
// 多线程：每次调用最多用几个线程由宏 MATRIX_EIG_THREADS 定，函数接口不变：
//   0（默认）= 用 MKL 自己的线程数（默认等于物理核数，可用环境变量 MKL_NUM_THREADS 改）；但同时还有别的调用正在算时，
//   这次调用只用 1 个线程——宿主程序已经在几个矩阵之间并行时，每个调用再各开一组线程只会互相抢核。
//   n >= 1 = 每次调用最多 n 个线程，不管别的调用。
//   只改调用线程自己的设定（MKL_Set_Num_Threads_Local），调用结束就恢复，不影响宿主程序别处对 MKL 的使用。
//   全局可变状态只有一个计数器（正在进行的调用数，用于上面的规则）。可重入：调用方可以在多个线程里同时调用。
//
// 编译与链接：
//   · 需要 Intel oneMKL 2025.0.1 或更新（oneAPI 的一部分，免费，可随程序再分发）。本文件自己声明用到的四个 MKL 函数
//     （zgeev_64_、dgeev_64_、MKL_Set_Num_Threads_Local、MKL_Get_Version_String），不需要 MKL 的头文件，只要链接
//     mkl_rt.lib（“单动态库”接口；2026 版起对应的 DLL 叫 mkl_rt.3.dll）。
//     导入库是同目录的 mkl/lib/mkl_rt.lib；运行库用同目录的 install_mkl.bat 下载到 mkl/bin（oneMKL 2026.2.0），见 README.md。
//     调的是 LAPACK 的 Fortran 接口（与 NumPy 相同），并且自己先问好工作区的大小。
//     运行时要能找到 mkl_rt.*.dll 与它按需加载的 mkl_core、mkl_intel_thread、mkl_avx2 等 DLL 和 libiomp5md.dll：
//     随 DLL 一起发布，或者放在 PATH 里。
//   · 调的是 64 位整数（ILP64）的 zgeev_64_、dgeev_64_，所以阶数不受 32 位整数的限制（LP64 的 zgeev_ 只能处理
//     2³¹ − 1 个元素以内的数组，即 46340 阶以内）。这两个名字在 2024.2.2 到 2026.2.0 的 mkl_rt 里都有，
//     与环境变量 MKL_INTERFACE_LAYER 的设定无关（实测设成 LP64、ILP64 结果都对）。
//   · 不要在这个文件之前包含 mkl.h（例如放在预编译头里）：它对同名函数的声明用的是 MKL_Complex16，与这里的声明冲突。
//   · AMD 处理器上同样能用（这里的用时就是在 AMD Ryzen 9 5950X 上测的）。

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#ifndef MATRIX_EIG_THREADS
#define MATRIX_EIG_THREADS 0      // 每次调用最多用几个线程：0 = MKL 自己的线程数（有别的调用在算时用 1 个），n = 最多 n 个
#endif

#if defined(_WIN32)
#define MATRIX_EIG_EXPORT __declspec(dllexport)
#else
#define MATRIX_EIG_EXPORT
#endif

// 用到的四个 MKL 函数：LAPACK 的 Fortran 接口 zgeev、dgeev 的 64 位整数版本 zgeev_64_、dgeev_64_（mkl_lapack.h 里的
// 声明；参数都按地址传，整数是 MKL_INT64，即 long long；MKL_Complex16 与 std::complex<double> 的内存布局相同；MKL 不要求
// 在 C 里传字符参数的长度），以及 mkl_service.h 里的线程设定与版本字符串。
extern "C" {
void zgeev_64_(const char* jobvl, const char* jobvr, const long long* n, std::complex<double>* a, const long long* lda,
               std::complex<double>* w, std::complex<double>* vl, const long long* ldvl, std::complex<double>* vr,
               const long long* ldvr, std::complex<double>* work, const long long* lwork, double* rwork, long long* info);
void dgeev_64_(const char* jobvl, const char* jobvr, const long long* n, double* a, const long long* lda, double* wr, double* wi,
               double* vl, const long long* ldvl, double* vr, const long long* ldvr, double* work, const long long* lwork,
               long long* info);
int MKL_Set_Num_Threads_Local(int nt);
void MKL_Get_Version_String(char* buffer, int length);
}

namespace matrix_eig {
namespace {

typedef std::complex<double> cplx;
typedef std::vector<cplx> Vector;

const double kEps = std::numeric_limits<double>::epsilon();   // 2.22e-16

// 复数稠密矩阵，按列存放：(i, j) 是第 i 行第 j 列，存在第 i + j·rows 个元素。
class Matrix {
public:
    Matrix() : rows_(0), cols_(0) {}
    Matrix(int rows, int cols)
        : rows_(rows), cols_(cols), a_(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols)) {}

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    cplx& operator()(int i, int j) { return a_[static_cast<std::size_t>(j) * rows_ + i]; }
    const cplx& operator()(int i, int j) const { return a_[static_cast<std::size_t>(j) * rows_ + i]; }
    cplx* col(int j) { return a_.data() + static_cast<std::size_t>(j) * rows_; }
    const cplx* col(int j) const { return a_.data() + static_cast<std::size_t>(j) * rows_; }

private:
    int rows_, cols_;
    std::vector<cplx> a_;
};

// =============================================================================================
// MKL：zgeev_64_ / dgeev_64_，以及每次调用的线程数
// =============================================================================================
namespace mkl {

// 正在进行的调用数。线程数是“自动”（MATRIX_EIG_THREADS = 0）时，同时还有别的调用在算就只用 1 个线程。
std::atomic<int> g_calls_in_flight(0);

// 一次调用的生存期：构造时登记，并按需要设好调用线程自己的 MKL 线程数；析构时恢复、注销。
class CallScope {
public:
    CallScope() : other_calls_(g_calls_in_flight.fetch_add(1, std::memory_order_relaxed)), previous_(0), changed_(false) {
        const int threads = MATRIX_EIG_THREADS >= 1 ? MATRIX_EIG_THREADS : (other_calls_ > 0 ? 1 : 0);
        if (threads > 0) {
            previous_ = MKL_Set_Num_Threads_Local(threads);
            changed_ = true;
        }
    }
    ~CallScope() {
        if (changed_) MKL_Set_Num_Threads_Local(previous_);   // previous_ = 0 表示恢复成全局设定
        g_calls_in_flight.fetch_sub(1, std::memory_order_relaxed);
    }
    CallScope(const CallScope&) = delete;
    CallScope& operator=(const CallScope&) = delete;

private:
    int other_calls_;
    int previous_;
    bool changed_;
};

// LAPACK 的 info：0 成功；> 0 是 QR 迭代没有收敛（照原样返回）；< 0 是参数错误，按意外处理。
inline int check_info(long long info) {
    if (info < 0) throw std::runtime_error("?geev: illegal argument");
    return static_cast<int>(info);                    // info > 0 时不超过 n，放得进 int
}

// 先问一次工作区要多大（lwork = -1），再分配、正式计算。
inline int zgeev(int order, cplx* a, cplx* w, cplx* vr) {
    const char no = 'N', yes = 'V';
    const long long n = order, one = 1, query = -1;
    cplx vl_dummy, size;
    std::vector<double> rwork(2 * static_cast<std::size_t>(n));
    long long info = 0;
    zgeev_64_(&no, &yes, &n, a, &n, w, &vl_dummy, &one, vr, &n, &size, &query, rwork.data(), &info);
    if (info != 0) return check_info(info);
    const long long lwork = (std::max)(1LL, static_cast<long long>(size.real()));
    std::vector<cplx> work(static_cast<std::size_t>(lwork));
    zgeev_64_(&no, &yes, &n, a, &n, w, &vl_dummy, &one, vr, &n, work.data(), &lwork, rwork.data(), &info);
    return check_info(info);
}

// 本征值 w 与本征向量 v（第 j 列对应 w[j]，2-范数为 1）。返回 0 表示成功；k > 0 表示 QR 迭代没有收敛。
inline int dgeev(int order, double* a, double* wr, double* wi, double* vr) {
    const char no = 'N', yes = 'V';
    const long long n = order, one = 1, query = -1;
    double vl_dummy = 0.0, size = 0.0;
    long long info = 0;
    dgeev_64_(&no, &yes, &n, a, &n, wr, wi, &vl_dummy, &one, vr, &n, &size, &query, &info);
    if (info != 0) return check_info(info);
    const long long lwork = (std::max)(1LL, static_cast<long long>(size));
    std::vector<double> work(static_cast<std::size_t>(lwork));
    dgeev_64_(&no, &yes, &n, a, &n, wr, wi, &vl_dummy, &one, vr, &n, work.data(), &lwork, &info);
    return check_info(info);
}

// 这个 MKL 的 dgeev 能不能用：版本字符串里的构建日期不早于 2025.0.1 的 20241031（见文件开头）。只读一次。
inline bool dgeev_usable() {
    static const bool usable = [] {
        char version[256] = {0};
        MKL_Get_Version_String(version, static_cast<int>(sizeof version) - 1);
        const char* build = std::strstr(version, "Build ");
        return build != nullptr && std::strtol(build + 6, nullptr, 10) >= 20241031L;
    }();
    return usable;
}

// 本征值 w 与本征向量 v（第 j 列对应 w[j]，2-范数为 1）。返回 0 表示成功；k > 0 表示 QR 迭代没有收敛。
inline int eig(const Matrix& a, Vector& w, Matrix& v) {
    const CallScope call;
    const int n = a.rows();
    const std::size_t nn = static_cast<std::size_t>(n) * n;
    bool real = true;
    for (int j = 0; j < n && real; ++j)
        for (int i = 0; i < n; ++i)
            if (a(i, j).imag() != 0.0) {
                real = false;
                break;
            }
    w.assign(n, cplx());
    v = Matrix(n, n);
    if (!real || !dgeev_usable()) {
        Matrix work = a;                              // zgeev 会改写输入
        return zgeev(n, work.col(0), w.data(), v.col(0));
    }
    // 实矩阵：dgeev。实本征值的 wi = 0；一对共轭复本征值相邻、虚部为正的在前，第 j、j+1 列是 x、y，
    // λ_j 的向量是 x + i·y、λ_{j+1} 的是 x − i·y。这里写成复数形式，交给下面共用的那段代码。
    std::vector<double> ar(nn), vr(nn), wr(n), wi(n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) ar[static_cast<std::size_t>(j) * n + i] = a(i, j).real();
    const int info = dgeev(n, ar.data(), wr.data(), wi.data(), vr.data());
    if (info != 0) return info;
    for (int j = 0; j < n;) {
        const double* x = vr.data() + static_cast<std::size_t>(j) * n;
        if (wi[j] == 0.0 || j + 1 == n) {
            w[j] = cplx(wr[j], 0.0);
            for (int i = 0; i < n; ++i) v(i, j) = x[i];
            j += 1;
        } else {
            const double* y = x + n;
            w[j] = cplx(wr[j], wi[j]);
            w[j + 1] = cplx(wr[j + 1], wi[j + 1]);
            for (int i = 0; i < n; ++i) {
                v(i, j) = cplx(x[i], y[i]);
                v(i, j + 1) = cplx(x[i], -y[i]);
            }
            j += 2;
        }
    }
    return 0;
}

}  // namespace mkl

// =============================================================================================
// 两个对外函数共用的部分
// =============================================================================================

inline int eig_core(const Matrix& a, Vector& w, Matrix& v) { return mkl::eig(a, w, v); }

inline bool finite(const cplx& z) { return std::isfinite(z.real()) && std::isfinite(z.imag()); }

inline bool all_finite(const Vector& w, const Matrix& v) {
    for (std::size_t i = 0; i < w.size(); ++i)
        if (!finite(w[i])) return false;
    for (int j = 0; j < v.cols(); ++j)
        for (int i = 0; i < v.rows(); ++i)
            if (!finite(v(i, j))) return false;
    return true;
}

// 输出的次序：实部大的在前，实部相同时虚部大的在前。
inline bool sorts_before(const cplx& a, const cplx& b) {
    if (a.real() != b.real()) return a.real() > b.real();
    return a.imag() > b.imag();
}

// x（n 个分量）乘一个单位模的数，使模最大的分量（并列时取下标小的）成为正实数。
inline void make_largest_real(cplx* x, int n) {
    int k = 0;
    double big = -1.0;
    for (int i = 0; i < n; ++i) {
        const double m = std::abs(x[i]);
        if (m > big) {
            big = m;
            k = i;
        }
    }
    if (!(big > 0.0) || !std::isfinite(big)) return;
    const double cr = x[k].real() / big, ci = -x[k].imag() / big;   // conj(x_k) / |x_k|
    for (int i = 0; i < n; ++i) {
        const double xr = x[i].real(), xi = x[i].imag();
        x[i] = cplx(cr * xr - ci * xi, cr * xi + ci * xr);
    }
    x[k] = cplx(big, 0.0);
}

// v（2-范数约为 1）的虚部相对实部是否可以忽略：||Im v||_2 <= sqrt(eps)·||Re v||_2。
inline bool nearly_real(const cplx* v, int n) {
    double re = 0.0, im = 0.0;
    for (int i = 0; i < n; ++i) {
        re += v[i].real() * v[i].real();
        im += v[i].imag() * v[i].imag();
    }
    return im <= kEps * re;
}

// 一对 (λ_p, λ_q) 是否改成两个实本征值、各用各的向量。vp 已转过相位；vq 是 λ_q 的向量（只读，这里转一份副本）。
// 两个条件都成立才拆：
//   1. v_p、v_q 转相位之后都几乎是实向量。这样拆出的两个实本征值的残差 |Im λ|·||Im v|| / ||Re v|| 都在 eps·||A||
//      量级：由 A·y = Im λ·x + Re λ·y 得 |Im λ| <= ||A − Re λ·I||·||y|| / ||x||。
//   2. 拆开后的两个实向量比原来的 x ± i·y 更接近线性无关：Re v_p 与 Re v_q 的夹角 θ 满足 2 sin(θ/2) > ||y⊥|| / ||x||，
//      y⊥ 是 y 去掉 x 方向的分量。接近亏损的二重实根被算成一对时 v_q ≈ conj(v_p)，两个实部几乎相同，
//      这一条不成立，保留成对（向量组本来就病态到 1/sqrt(eps) 左右，拆开会变成完全奇异）。
//      夹角用 min(||x̂_p − x̂_q||, ||x̂_p + x̂_q||) = 2 sin(θ/2) 算：用内积求 cos 时，θ < 1e-8 的部分全被相消掉。
inline bool split_pair(const cplx* vp, const cplx* vq_in, int n) {
    if (!nearly_real(vp, n)) return false;
    std::vector<cplx> vq(vq_in, vq_in + n);
    make_largest_real(vq.data(), n);
    if (!nearly_real(vq.data(), n)) return false;
    double xx = 0.0, xy = 0.0, qq = 0.0;            // 两个向量的模最大分量都是正实数，所以 xx、qq 不为 0
    for (int i = 0; i < n; ++i) {
        xx += vp[i].real() * vp[i].real();
        xy += vp[i].real() * vp[i].imag();
        qq += vq[i].real() * vq[i].real();
    }
    const double c = xy / xx, sp = 1.0 / std::sqrt(xx), sq = 1.0 / std::sqrt(qq);
    double yperp = 0.0, minus = 0.0, plus = 0.0;
    for (int i = 0; i < n; ++i) {
        const double y = vp[i].imag() - c * vp[i].real();
        yperp += y * y;
        const double a = vp[i].real() * sp, b = vq[i].real() * sq;
        minus += (a - b) * (a - b);
        plus += (a + b) * (a + b);
    }
    return (std::min)(minus, plus) > yperp / xx;
}

// 实本征值的向量：复数算法给出的 v 与实向量只差一个相位。转成模最大的分量为正实数之后取实部，再归一到 2-范数为 1。
inline void real_vector(cplx* v, int n, double* x) {
    make_largest_real(v, n);
    double big = 0.0;
    for (int i = 0; i < n; ++i) {
        x[i] = v[i].real();
        big = (std::max)(big, std::abs(x[i]));
    }
    if (!(big > 0.0) || !std::isfinite(big)) return;
    double ssq = 0.0;
    for (int i = 0; i < n; ++i) {
        const double r = x[i] / big;
        ssq += r * r;
    }
    const double inv = 1.0 / std::sqrt(ssq);
    for (int i = 0; i < n; ++i) x[i] = (x[i] / big) * inv;
    // 取实部、归一之后模最大的分量可能换了一个（原来几个分量的模在舍入以内相等时），按最后的结果再定一次符号
    int k = 0;
    for (int i = 1; i < n; ++i)
        if (std::abs(x[i]) > std::abs(x[k])) k = i;
    if (x[k] < 0.0)
        for (int i = 0; i < n; ++i) x[i] = -x[i];
}

// 实矩阵的 n 个本征值（复数算法算出、带舍入误差）配成“实本征值”和“共轭对”：
// partner[i] = i 是实本征值，partner[i] = j ≠ i 表示与 j 成一对。距离 d(i, j) = |λ_i − conj(λ_j)|。
// 每一轮在还没配上的本征值里给每个 i 找 d 最小的 j（自己优先，严格更近才换），互为最近的配上；
// 一轮下来一组也没配上（只会因为距离恰好相等）时，把全局最近的那一组配上。每轮至少配上一组，所以一定结束。
inline std::vector<int> pair_conjugates(const Vector& w) {
    const int n = static_cast<int>(w.size());
    std::vector<int> partner(n, -1), nearest(n, -1), open(n), rest;
    for (int i = 0; i < n; ++i) open[i] = i;
    while (!open.empty()) {
        int closest = open.front();
        double closest_d = std::numeric_limits<double>::infinity();
        for (std::size_t p = 0; p < open.size(); ++p) {
            const int i = open[p];
            int best = i;
            double best_d = 2.0 * std::abs(w[i].imag());
            for (std::size_t q = 0; q < open.size(); ++q) {
                const int j = open[q];
                if (j == i) continue;
                const double d = std::hypot(w[i].real() - w[j].real(), w[i].imag() + w[j].imag());
                if (d < best_d) {
                    best_d = d;
                    best = j;
                }
            }
            nearest[i] = best;
            if (best_d < closest_d) {
                closest_d = best_d;
                closest = i;
            }
        }
        bool matched = false;
        for (std::size_t p = 0; p < open.size(); ++p) {
            const int i = open[p], j = nearest[i];
            if (j == i || nearest[j] == i) {
                partner[i] = j;
                matched = true;
            }
        }
        if (!matched) {
            const int j = nearest[closest];
            partner[closest] = j;
            partner[j] = closest;
        }
        rest.clear();
        for (std::size_t p = 0; p < open.size(); ++p)
            if (partner[open[p]] < 0) rest.push_back(open[p]);
        open.swap(rest);
    }
    return partner;
}

// 输出排序时的一个单位：一个实本征值，或一对共轭本征值（lambda 是虚部为正的那个）。
struct Unit {
    cplx lambda;
    int column;      // 对应的复本征向量在 v 里的列号（一对时是 lambda 那一半的）
    bool pair;
};

inline Unit make_unit(const cplx& lambda, int column, bool pair) {
    Unit u;
    u.lambda = lambda;
    u.column = column;
    u.pair = pair;
    return u;
}

inline bool unit_before(const Unit& a, const Unit& b) { return sorts_before(a.lambda, b.lambda); }
inline bool value_before(const std::pair<cplx, int>& a, const std::pair<cplx, int>& b) { return sorts_before(a.first, b.first); }

// 实矩阵：求解、配对、排序，算好之后才写进 EigenValue / EigenVector（打包格式）。
inline int solve_real(double** A1, cplx* EigenValue, double** EigenVector, int n) {
    Matrix a(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            if (!std::isfinite(A1[i][j])) return -2;
            a(i, j) = A1[i][j];
        }
    Vector w;
    Matrix v;
    const int info = eig_core(a, w, v);
    if (info != 0) return info;
    if (!all_finite(w, v)) return -4;

    const std::vector<int> partner = pair_conjugates(w);
    std::vector<Unit> units;
    for (int i = 0; i < n; ++i) {
        const int j = partner[i];
        if (j == i) {
            units.push_back(make_unit(cplx(w[i].real(), 0.0), i, false));
            continue;
        }
        if (j < i) continue;                          // 一对只在较小的下标处处理一次
        // 配上的两个本征值虚部总是一正一负：互为最近要求 d(i, j) 比 2|Im λ_i|、2|Im λ_j| 都小（全局最近那一步里
        // 至多与其中一个相等），而 d(i, j) >= |Im λ_i + Im λ_j|，同号或有一个为 0 时做不到。
        // 用虚部为正的那一半 λ_p 与它的向量 v_p，输出 λ_p、conj(λ_p) 与 Re v_p、Im v_p
        // （不取两半的平均：v_p 是 λ_p 的向量，平均值的残差可能大得多）。总取虚部为正的一半，重复的共轭对
        // （例如 ±3i 各两重）才会各用各的向量；若按别的规则挑，两对可能挑到同一个向量的两份，向量组就退化了。
        const int p = w[i].imag() > 0.0 ? i : j, q = i + j - p;
        cplx* vp = v.col(p);
        make_largest_real(vp, n);
        if (w[p].imag() > 0.0 && !split_pair(vp, v.col(q), n)) {
            units.push_back(make_unit(w[p], p, true));
        } else {
            // 重复的实本征值被算成虚部极小的一对、而两半的向量都几乎是实的：x ± i·y 这两列几乎线性相关，
            // 改成两个实本征值、各用各的向量（条件见 split_pair）。
            units.push_back(make_unit(cplx(w[p].real(), 0.0), p, false));
            units.push_back(make_unit(cplx(w[q].real(), 0.0), q, false));
        }
    }
    std::stable_sort(units.begin(), units.end(), unit_before);

    std::vector<cplx> values(n);
    std::vector<double> vectors(static_cast<std::size_t>(n) * n);   // 按列：第 k 列从 vectors[k·n] 开始
    int k = 0;
    for (std::size_t u = 0; u < units.size(); ++u) {
        cplx* vu = v.col(units[u].column);
        double* out = vectors.data() + static_cast<std::size_t>(k) * n;
        if (!units[u].pair) {
            values[k] = units[u].lambda;
            real_vector(vu, n, out);
            k += 1;
        } else {
            values[k] = units[u].lambda;                // 向量在配对时已经转过相位
            values[k + 1] = std::conj(units[u].lambda);
            for (int i = 0; i < n; ++i) {
                out[i] = vu[i].real();
                out[n + i] = vu[i].imag();
            }
            k += 2;
        }
    }
    for (int i = 0; i < n; ++i)
        if (!finite(values[i])) return -4;
    for (std::size_t i = 0; i < vectors.size(); ++i)
        if (!std::isfinite(vectors[i])) return -4;

    for (int j = 0; j < n; ++j) {
        EigenValue[j] = values[j];
        const double* col = vectors.data() + static_cast<std::size_t>(j) * n;
        for (int i = 0; i < n; ++i) EigenVector[i][j] = col[i];
    }
    return 0;
}

// 复矩阵：求解、规范相位、排序，算好之后才写输出。
inline int solve_complex(cplx** A1, cplx* EigenValue, cplx** EigenVector, int n) {
    Matrix a(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            if (!finite(A1[i][j])) return -2;
            a(i, j) = A1[i][j];
        }
    Vector w;
    Matrix v;
    const int info = eig_core(a, w, v);
    if (info != 0) return info;
    if (!all_finite(w, v)) return -4;

    std::vector<std::pair<cplx, int> > order(n);
    for (int j = 0; j < n; ++j) {
        make_largest_real(v.col(j), n);
        order[j] = std::make_pair(w[j], j);
    }
    std::stable_sort(order.begin(), order.end(), value_before);

    for (int k = 0; k < n; ++k) {
        EigenValue[k] = order[k].first;
        const cplx* col = v.col(order[k].second);
        for (int i = 0; i < n; ++i) EigenVector[i][k] = col[i];
    }
    return 0;
}

inline void fill_nan(cplx* values, double** vectors, int n) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (int i = 0; i < n; ++i) values[i] = cplx(nan, nan);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) vectors[i][j] = nan;
}

inline void fill_nan(cplx* values, cplx** vectors, int n) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (int i = 0; i < n; ++i) values[i] = cplx(nan, nan);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) vectors[i][j] = cplx(nan, nan);
}

template <class T>
inline bool pointers_ok(T** A1, cplx* EigenValue, T** EigenVector, int mn) {
    if (mn < 1 || !A1 || !EigenValue || !EigenVector) return false;
    for (int i = 0; i < mn; ++i)
        if (!A1[i] || !EigenVector[i]) return false;
    return true;
}

}  // namespace
}  // namespace matrix_eig

// 求实数矩阵的特征值和特征矢量
extern "C++" MATRIX_EIG_EXPORT int MatrixEigenValueAndEigenVector(double** A1,
    std::complex<double>* EigenValue, double** EigenVector,
    int mn) {
    if (!matrix_eig::pointers_ok(A1, EigenValue, EigenVector, mn)) return -1;
    int status;
    try {
        status = matrix_eig::solve_real(A1, EigenValue, EigenVector, mn);
    } catch (...) {
        status = -3;
    }
    if (status != 0) matrix_eig::fill_nan(EigenValue, EigenVector, mn);
    return status;
}

// 求复数矩阵的特征值和特征矢量
extern "C++" MATRIX_EIG_EXPORT int MatrixEigenValueAndEigenVector(std::complex<double>** A1,
    std::complex<double>* EigenValue, std::complex<double>** EigenVector,
    int mn) {
    if (!matrix_eig::pointers_ok(A1, EigenValue, EigenVector, mn)) return -1;
    int status;
    try {
        status = matrix_eig::solve_complex(A1, EigenValue, EigenVector, mn);
    } catch (...) {
        status = -3;
    }
    if (status != 0) matrix_eig::fill_nan(EigenValue, EigenVector, mn);
    return status;
}
