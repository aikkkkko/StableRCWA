// stablercwa/mkl/mkl_backend.hpp —— 可选的 oneMKL 后端：层矩阵的本征分解（zgeev / dgeev）、稠密乘法（zgemm）与
// 稠密线性求解（zgetrf + zgetrs）交给 Intel oneMKL
//
// RCWA 的三个求解器（一维 rcwa::solve、圆锥入射 rcwa::solve_conical、二维 rcwa::solve2d）都有 custom_eig、
// custom_multiply、custom_solve 三个钩子，设了它们，层矩阵的本征分解、稠密乘法、稠密线性求解就交给它们做。
// 本文件提供用 oneMKL 的实现：
//
//   #include "stablercwa/mkl/mkl_backend.hpp"
//
//   rcwa::Options o;                                     // 二维的 rcwa::Options2D 同样
//   o.custom_eig = stablercwa::mkl::eig_hook();          // 之后照常调用 rcwa::solve(..., o)
//
//   stablercwa::mkl::Settings s;                         // 各层都无吸收（折射率全是实数）时可以再打开这一项，
//   s.real_path = true;                                  // 对称结构的层矩阵就走更快的 dgeev（见下面“实矩阵走 dgeev”）
//   o.custom_eig = stablercwa::mkl::eig_hook(s);
//
//   stablercwa::mkl::use_mkl(o, s);                      // 三个钩子一次设好（本征分解、乘法、线性求解都交给 MKL）
//
//   stablercwa::EigResult r = stablercwa::mkl::eig(A);   // 也可以单独用：一般复矩阵的本征值与右本征向量
//   Matrix C = stablercwa::mkl::multiply(A, B);          // C = A·B
//   Matrix X = stablercwa::mkl::solve(A, B);             // 解 A·X = B（按列选主元的 LU）
//
// JSON 入口（rcwa_json.hpp）在编译时定义了 STABLERCWA_WITH_MKL 时接受 "options": {"backend": "mkl"}，做的就是 use_mkl，
// real_path 按请求里各层的折射率决定。
//
// 几点约定：
//   · 这是可选的。stablercwa.hpp 不包含本文件；只有包含它的程序要链接 oneMKL 的单动态库 mkl_rt（Windows：
//     standalone/mkl/lib/mkl_rt.lib，运行库用 standalone/install_mkl.bat 下载，见 standalone/README.md），
//     其余代码仍然只依赖 C++17 标准库。CMake 选项 STABLERCWA_WITH_MKL 编译本文件的测试与测速程序。
//   · 不需要 MKL 的头文件：用到的七个函数（zgeev_64_、dgeev_64_、zgemm_64_、zgetrf_64_、zgetrs_64_、
//     MKL_Set_Num_Threads_Local、MKL_Get_Version_String）在下面自己声明，导入库 standalone/mkl/lib/mkl_rt.lib 要有它们。不要在本文件之前包含 mkl.h（它对同名函数的声明用的是 MKL_Complex16，两者冲突）。
//     调的是 64 位整数（ILP64）的版本，阶数不受 32 位整数的限制，与环境变量 MKL_INTERFACE_LAYER 无关。
//   · 实矩阵走 dgeev（要调用方打开 Settings::real_path，默认关闭）：打开后，每个元的虚部都不超过 kRealTolerance =
//     512·2^-52（约 1.1e-13）乘以它所在行、列的量级（行、列里各元 max(|实部|, |虚部|) 的最大值）的几何平均时
//     （见 nearly_real），丢掉虚部、按实矩阵求解，比 zgeev 快 2.2 到 2.5 倍；否则仍走 zgeev。
//     对 RCWA 来说，结构无吸收、并且相对原点或单元中心对称（ε(x) = ε(−x)；二维是 ε(x, y) = ε(−x, −y)）时，层矩阵在数学上
//     是实的，算出来的虚部只是组装时的舍入误差；这时丢掉虚部不会让矩阵离精确值更远（精确值是实的，取实部只会更近）。
//     按行、列的量级比，而不是按整个矩阵的范数比：高阶时对角元 kx² 很大，整个矩阵的范数会把吸收带来的虚部“淹没”
//     （TE、k = 1e-9 的吸收在 M = 150 时只占范数的 5e-14，丢掉它，效率差 2e-9）。实测（单位 2^-52）：对称、无吸收的结构
//     一维 TE 不到 2，TM 随阶数、折射率差增大（n = 4：M = 800 时 281，M = 1600 时 431）；二维不到 10。超过阈值的照样走
//     zgeev，只是慢一些：例如 n = 5、M = 1600 的 TM（652），ε = −20 的无吸收金属 M ≥ 800 时（1636 以上）。
//     为什么要调用方来开：弱到 k ≈ 1e-13 的吸收（TE 到 4e-14 左右）在层矩阵里与舍入误差分不开，光看矩阵认不出来，
//     而在高 Q 共振附近（例如对称保护的连续谱束缚态），这么弱的吸收对效率的影响远大于舍入误差：Q 很高的对称光栅，
//     k = 1e-13 的 TM 吸收掉 4.5e-6，当成无吸收以后效率差 3.5e-6。所以只在调用方确认各层折射率都是实数时才打开；
//     JSON 入口按请求里的折射率自动决定。打开后，吸收到 k = 1e-12 以上（≥ 931）或不对称（1e10 以上）的层矩阵仍走 zgeev。
//   · oneMKL 的版本：请用 2025.0.1 或更新。2024.2.2、2025.0.0 在 AMD 处理器、Windows 上的 dgeev 有毛病（不要左本征
//     向量时仍往 VL 里写，还写出工作区的末尾，见 standalone/README.md），所以第一次调用时读 MKL 的版本字符串，
//     构建日期早于 2025.0.1 的 20241031 或读不出来时，实矩阵也走 zgeev（与 standalone/ 的 MKL 版同一个判断；
//     2024.2.2 的构建日期是 20240823，2025.0.0 是 20241009）。
//     zgeev 在 2024.2.2 到 2026.2.0 上都测过没有问题。
//   · 返回值的约定与 eig() 相同：values；vectors 的各列是右本征向量，2-范数为 1；info = 0 表示收敛，
//     info = k > 0 表示 LAPACK 的 QR 迭代没有收敛（下标 >= k 的本征值有效，不给本征向量），与 eig_qr 相同。
//     不填 Z、T、sweeps。本征值的排列次序与 eig_qr 不同；RCWA 求解器不依赖次序，也不依赖本征向量的相位。
//     走 dgeev 时实本征值的虚部严格为 0，一对共轭本征值相邻（虚部为正的在前），向量严格共轭。
//   · 结果与 eig_qr 只在舍入误差以内一致，不逐位相同。eig_qr 在任何线程数下都逐位相同，MKL 不保证：线程数、处理器
//     （AVX2 或 AVX-512）不同时末位可能不同。要可重复，设环境变量 MKL_CBWR=AVX2，并把线程数固定（见下一条）。
//     zgeev、dgeev 先对矩阵做平衡（对角相似变换）再迭代，对元素量级悬殊的矩阵，本征值往往比不做平衡的 eig_qr 更准。
//   · 线程：每次调用设一次调用线程自己的 MKL 线程数（MKL_Set_Num_Threads_Local），调用结束就恢复，不影响程序别处
//     对 MKL 的使用；所以环境变量 MKL_NUM_THREADS 对这里不起作用。用几个：
//       在 StableRCWA 的并行区里（parallel_layers、sweep 等，in_parallel_region() 为真）：1 个，免得线程池的每个线程
//         再各开一组 MKL 线程，互相抢核；
//       否则 Settings::threads = n >= 1 时用 n 个；
//       否则（默认的 0）用 num_threads() 个，即 set_num_threads() 的设定，与 StableRCWA 其余部分（线性求解、
//         矩阵乘法、eig_qr）用的一样多（没设过就是 1）；JSON 入口里就是 options.threads。同时还有别的调用在算时只用 1 个。
//     规则在 detail::call_threads 里。不用 MKL 自己的默认值（物理核数）：在 16 核的 Ryzen 9 5950X 上，zgeev 从 8 个线程加到
//     16 个，3200 阶时只再快 12 %，1000 阶以下差不多或更慢；dgeev 过了 8 个线程不再变快，900 阶以下 1 个线程就与 8 个差不多
//     （实测见 docs/user-guide.zh-CN.md 的“oneMKL 后端”一节）。MKL 的 OpenMP 线程在每次多线程调用之后默认还会空转约
//     200 毫秒（环境变量 KMP_BLOCKTIME）；num_threads() 接近核数时可以设 KMP_BLOCKTIME=0，免得它们与线程池抢核（没有实测）。
//     可以被多个线程同时调用（custom_eig 与 parallel_layers、sweep 同用时要求如此）。
//   · 乘法（multiply）：zgemm。内置的乘法跳过 B 里整组为零的系数（每列四个一组；B 是对角阵时只要 O(n²)），zgemm 不跳，
//     所以 B 里非零的组不超过 kSparseGroups（1/8）时交回内置的 a * b（matrix.hpp）。结果与内置的只在舍入误差以内一致。
//   · 线性求解（solve）：zgetrf（按列选主元的 LU）加 zgetrs，与内置 LU 是同一种方法。只在出现精确为零的主元时
//     （zgetrf 的 info > 0）抛 std::runtime_error，RCWA 求解器当作数值失败，与内置 LU 的约定相同；分块的次序与内置的
//     不同，结构上奇异的矩阵在两边是否都恰好得到零主元，要靠测试核对（tests/mkl/test_rcwa_mkl.cpp 的 K 组）。
//   · 出错：本征分解遇到非方阵、含 NaN/Inf 的矩阵抛 std::invalid_argument（同 eig_qr）；乘法、求解的矩阵大小不合抛
//     std::invalid_argument；LAPACK 报参数错误时抛 stablercwa::mkl::Error（派生自 std::logic_error，是调用错误，
//     RCWA 求解器不把它当作数值失败）；本征向量不是有限值时抛 std::overflow_error
//     （同 eig_qr，RCWA 求解器把它当作数值失败）；内存不足抛 std::bad_alloc。
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../eig_types.hpp"
#include "../matrix.hpp"
#include "../parallel.hpp"

// 用到的七个 oneMKL 函数：BLAS、LAPACK 的 Fortran 接口 zgeev、dgeev、zgemm、zgetrf、zgetrs 的 64 位整数版本（mkl_lapack.h、
// mkl_blas.h 里的声明；参数都按地址传，整数是 MKL_INT64，即 long long；MKL_Complex16 与 std::complex<double> 的内存布局相同；
// MKL 不要求在 C 里传字符参数的长度），以及 mkl_service.h 里的线程设定与版本字符串。zgeev、dgeev 和后两个与 standalone/
// 里 MKL 版的声明一字不差，同一个程序里两者并存也没问题。
extern "C" {
void zgeev_64_(const char* jobvl, const char* jobvr, const long long* n, std::complex<double>* a, const long long* lda,
               std::complex<double>* w, std::complex<double>* vl, const long long* ldvl, std::complex<double>* vr,
               const long long* ldvr, std::complex<double>* work, const long long* lwork, double* rwork, long long* info);
void dgeev_64_(const char* jobvl, const char* jobvr, const long long* n, double* a, const long long* lda, double* wr, double* wi,
               double* vl, const long long* ldvl, double* vr, const long long* ldvr, double* work, const long long* lwork,
               long long* info);
void zgemm_64_(const char* transa, const char* transb, const long long* m, const long long* n, const long long* k,
               const std::complex<double>* alpha, const std::complex<double>* a, const long long* lda,
               const std::complex<double>* b, const long long* ldb, const std::complex<double>* beta, std::complex<double>* c,
               const long long* ldc);
void zgetrf_64_(const long long* m, const long long* n, std::complex<double>* a, const long long* lda, long long* ipiv,
                long long* info);
void zgetrs_64_(const char* trans, const long long* n, const long long* nrhs, const std::complex<double>* a, const long long* lda,
                const long long* ipiv, std::complex<double>* b, const long long* ldb, long long* info);
int MKL_Set_Num_Threads_Local(int nt);
void MKL_Get_Version_String(char* buffer, int length);
}

namespace stablercwa {
namespace mkl {

// 打开 Settings::real_path 时，每个元的虚部与它所在行、列量级之比都不超过它就按实矩阵求解（dgeev），见文件开头与 nearly_real。
inline constexpr double kRealTolerance = 512.0 * 2.220446049250313e-16;

struct Settings {
    // 并行区外每次调用用几个 MKL 线程：0（默认）= num_threads()（set_num_threads() 的设定），但同时还有别的调用在算时
    // 只用 1 个；n >= 1 = n 个。并行区里总是 1 个。
    int threads = 0;
    // true：调用方确认各层折射率都是实数（没有吸收），虚部在舍入误差以内的矩阵（见 nearly_real）按实矩阵用 dgeev。
    // false（默认）：一律用 zgeev。为什么默认关闭见文件开头“实矩阵走 dgeev”。
    bool real_path = false;
};

// LAPACK 报参数错误之类的调用错误（例如链接了接口不符的 mkl_rt）。派生自 std::logic_error：RCWA 求解器只把
// std::runtime_error 当作“方程组奇异”的数值失败，这类错误要原样传出去，不能变成一组 NaN 效率。
class Error : public std::logic_error {
public:
    explicit Error(const std::string& what) : std::logic_error("stablercwa::mkl: " + what) {}
};

// MKL 的版本字符串，例如 "Intel(R) oneAPI Math Kernel Library Version 2026.2-Product Build 20260915 for ..."。
inline std::string version_string() {
    char buffer[256] = {0};
    MKL_Get_Version_String(buffer, static_cast<int>(sizeof buffer) - 1);
    return buffer;
}

// 这个 MKL 的 dgeev 能不能用：版本字符串里的构建日期不早于 2025.0.1 的 20241031（见文件开头）。只读一次。
inline bool dgeev_usable() {
    static const bool usable = [] {
        const std::string version = version_string();
        const std::size_t build = version.find("Build ");
        return build != std::string::npos && std::strtol(version.c_str() + build + 6, nullptr, 10) >= 20241031L;
    }();
    return usable;
}

// 是否每个元都满足 |Im a_ij| <= kRealTolerance·sqrt(r_i·c_j)，r_i、c_j 是第 i 行、第 j 列各元 max(|实部|, |虚部|) 的最大值。
// 虚部全为 0 时当然是。元素必须都是有限值；先各取平方根再相乘，元素再大也不会溢出。
inline bool nearly_real(const Matrix& a) {
    const int rows = a.rows(), cols = a.cols();
    std::vector<double> r(static_cast<std::size_t>(rows), 0.0), c(static_cast<std::size_t>(cols), 0.0);
    bool imaginary = false;
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i) {
            const cplx z = a(i, j);
            const double size = (std::max)(std::abs(z.real()), std::abs(z.imag()));
            r[i] = (std::max)(r[i], size);
            c[j] = (std::max)(c[j], size);
            imaginary = imaginary || z.imag() != 0.0;
        }
    if (!imaginary) return true;
    for (double& x : r) x = std::sqrt(x);
    for (double& x : c) x = std::sqrt(x);
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i)
            if (std::abs(a(i, j).imag()) > kRealTolerance * r[i] * c[j]) return false;
    return true;
}

namespace detail {

// 正在进行的调用数（Settings::threads = 0 时用于“有别的调用在算就只用 1 个线程”）。内联函数里的静态变量在整个程序里只有一份。
inline std::atomic<int>& calls_in_flight() {
    static std::atomic<int> count(0);
    return count;
}

// 一次调用给 MKL 几个线程（规则见文件开头）。setting 是 Settings::threads，others 是同时在算的别的调用数。
inline int call_threads(int setting, int others) {
    if (in_parallel_region()) return 1;
    if (setting >= 1) return setting;
    return others > 0 ? 1 : (std::max)(1, num_threads());
}

// 一次调用的生存期：构造时登记，并设好调用线程自己的 MKL 线程数；析构时恢复、注销。
class CallScope {
public:
    explicit CallScope(int threads_setting)
        : previous_(MKL_Set_Num_Threads_Local(
              call_threads(threads_setting, calls_in_flight().fetch_add(1, std::memory_order_relaxed)))) {}
    ~CallScope() {
        MKL_Set_Num_Threads_Local(previous_);   // previous_ = 0 表示恢复成全局设定
        calls_in_flight().fetch_sub(1, std::memory_order_relaxed);
    }
    CallScope(const CallScope&) = delete;
    CallScope& operator=(const CallScope&) = delete;

private:
    int previous_;
};

inline void check_info(const char* routine, long long info) {
    if (info < 0) throw Error(std::string(routine) + ": illegal argument #" + std::to_string(-info));
}

// zgeev：values 与 v（第 j 列对应 values[j]）。work 被改写。返回 LAPACK 的 info（>= 0）。
inline long long zgeev(int n, Matrix& work, Vector& values, Matrix& v) {
    const char no = 'N', yes = 'V';
    const long long order = n, one = 1, query = -1;
    cplx vl_dummy, size;
    std::vector<double> rwork(2 * static_cast<std::size_t>(n));
    long long info = 0;
    // 先问一次工作区要多大（lwork = -1），再分配、正式计算。
    zgeev_64_(&no, &yes, &order, work.data(), &order, values.data(), &vl_dummy, &one, v.data(), &order, &size, &query,
              rwork.data(), &info);
    check_info("zgeev", info);
    const long long lwork = (std::max)(1LL, static_cast<long long>(size.real()));
    std::vector<cplx> buffer(static_cast<std::size_t>(lwork));
    zgeev_64_(&no, &yes, &order, work.data(), &order, values.data(), &vl_dummy, &one, v.data(), &order, buffer.data(), &lwork,
              rwork.data(), &info);
    check_info("zgeev", info);
    return info;
}

// dgeev，输入取 a 的实部：values 与 v 写成复数形式。实本征值的 wi = 0；一对共轭复本征值相邻、虚部为正的在前，
// dgeev 的第 j、j+1 列是 x、y，λ_j 的向量是 x + i·y、λ_{j+1} 的是 x − i·y。返回 LAPACK 的 info（>= 0）。
inline long long dgeev(int n, const Matrix& a, Vector& values, Matrix& v) {
    const std::size_t nn = static_cast<std::size_t>(n) * n;
    std::vector<double> ar(nn), vr(nn), wr(static_cast<std::size_t>(n)), wi(static_cast<std::size_t>(n));
    const cplx* p = a.data();
    for (std::size_t k = 0; k < nn; ++k) ar[k] = p[k].real();
    const char no = 'N', yes = 'V';
    const long long order = n, one = 1, query = -1;
    double vl_dummy = 0.0, size = 0.0;
    long long info = 0;
    dgeev_64_(&no, &yes, &order, ar.data(), &order, wr.data(), wi.data(), &vl_dummy, &one, vr.data(), &order, &size, &query,
              &info);
    check_info("dgeev", info);
    const long long lwork = (std::max)(1LL, static_cast<long long>(size));
    std::vector<double> buffer(static_cast<std::size_t>(lwork));
    dgeev_64_(&no, &yes, &order, ar.data(), &order, wr.data(), wi.data(), &vl_dummy, &one, vr.data(), &order, buffer.data(),
              &lwork, &info);
    check_info("dgeev", info);
    for (int j = 0; j < n; ++j) values[j] = cplx(wr[j], wi[j]);
    if (info > 0) return info;                         // 没有全部收敛：不用本征向量
    for (int j = 0; j < n;) {
        const double* x = vr.data() + static_cast<std::size_t>(j) * n;
        if (wi[j] == 0.0) {
            for (int i = 0; i < n; ++i) v(i, j) = x[i];
            j += 1;
        } else {
            if (j + 1 == n || wi[j + 1] != -wi[j])
                throw Error("dgeev: complex eigenvalues are not in conjugate pairs");
            const double* y = x + n;
            for (int i = 0; i < n; ++i) {
                v(i, j) = cplx(x[i], y[i]);
                v(i, j + 1) = cplx(x[i], -y[i]);
            }
            j += 2;
        }
    }
    return 0;
}

}  // namespace detail

// 一般复矩阵的本征值与右本征向量（zgeev；虚部在舍入误差以内时 dgeev）。a 不被修改。返回值的约定见文件开头。
inline EigResult eig(const Matrix& a, const Settings& settings = {}) {
    if (!a.square()) throw std::invalid_argument("mkl::eig: matrix must be square");
    if (!a.all_finite()) throw std::invalid_argument("mkl::eig: matrix has non-finite entries");
    const int n = a.rows();
    EigResult r;
    r.values.assign(static_cast<std::size_t>(n), cplx{});
    if (n == 0) return r;

    const bool real = settings.real_path && nearly_real(a) && dgeev_usable();
    const detail::CallScope call(settings.threads);
    Matrix v(n, n);
    long long info = 0;
    if (real) {
        info = detail::dgeev(n, a, r.values, v);
    } else {
        Matrix work = a;                              // zgeev 改写输入
        info = detail::zgeev(n, work, r.values, v);
    }
    if (info > 0) {                                   // 没有全部收敛：不给本征向量（与 eig_qr 相同）
        r.info = static_cast<int>(info);
        return r;
    }
    // zgeev、dgeev 已把每个本征向量归一到 2-范数为 1（模最大的分量为实数）。
    if (!v.all_finite()) throw std::overflow_error("mkl::eig: eigenvectors are not finite (matrix entries out of range)");
    r.vectors = std::move(v);
    return r;
}

// 直接赋给 rcwa::Options::custom_eig 或 rcwa::Options2D::custom_eig。
inline std::function<EigResult(const Matrix&)> eig_hook(const Settings& settings = {}) {
    return [settings](const Matrix& a) { return eig(a, settings); };
}

// B 里非零的组（每列四个系数一组，与内置的乘法相同）不超过这个比例时，multiply 交回内置的 a * b（见文件开头）。
inline constexpr double kSparseGroups = 0.125;

namespace detail {

// B 是否稀疏到应当交回内置的乘法：数每列里不全为零的四个一组。
inline bool sparse_for_gemm(const Matrix& b) {
    const int p = b.rows(), n = b.cols();
    long long nonzero = 0;
    for (int j = 0; j < n; ++j) {
        const cplx* c = b.col(j);
        for (int k = 0; k < p; k += 4) {
            const int end = (std::min)(k + 4, p);
            for (int i = k; i < end; ++i)
                if (c[i] != cplx{}) {
                    ++nonzero;
                    break;
                }
        }
    }
    const double groups = static_cast<double>((p + 3) / 4) * n;
    return static_cast<double>(nonzero) <= kSparseGroups * groups;
}

}  // namespace detail

// C = A·B（zgemm；B 稀疏时交回内置的 a * b）。A.cols() != B.rows() 时抛 std::invalid_argument。
inline Matrix multiply(const Matrix& a, const Matrix& b, const Settings& settings = {}) {
    if (a.cols() != b.rows()) throw std::invalid_argument("mkl::multiply: shape mismatch");
    const int m = a.rows(), n = b.cols(), p = a.cols();
    if (m == 0 || n == 0 || p == 0 || detail::sparse_for_gemm(b)) return a * b;
    Matrix c(m, n);
    const char no = 'N';
    const long long rows = m, cols = n, inner = p;
    const cplx one(1.0, 0.0), zero(0.0, 0.0);
    const detail::CallScope call(settings.threads);
    zgemm_64_(&no, &no, &rows, &cols, &inner, &one, a.data(), &rows, b.data(), &inner, &zero, c.data(), &rows);
    return c;
}

// 解 A·X = B（zgetrf + zgetrs，按列选主元的 LU）。A 不是方阵、B 的行数不对时抛 std::invalid_argument；
// 出现精确为零的主元时抛 std::runtime_error（与 stablercwa::solve 相同）。
inline Matrix solve(const Matrix& a, const Matrix& b, const Settings& settings = {}) {
    if (!a.square()) throw std::invalid_argument("mkl::solve: matrix must be square");
    if (b.rows() != a.rows()) throw std::invalid_argument("mkl::solve: right-hand side has wrong number of rows");
    const int n = a.rows();
    Matrix x = b;
    if (n == 0 || b.cols() == 0) return x;
    Matrix lu = a;                                    // zgetrf 改写输入
    std::vector<long long> pivots(static_cast<std::size_t>(n));
    const long long order = n, columns = b.cols();
    long long info = 0;
    const detail::CallScope call(settings.threads);
    zgetrf_64_(&order, &order, lu.data(), &order, pivots.data(), &info);
    detail::check_info("zgetrf", info);
    if (info > 0) throw std::runtime_error("mkl::solve: matrix is singular (zero pivot)");
    const char no = 'N';
    zgetrs_64_(&no, &order, &columns, lu.data(), &order, pivots.data(), x.data(), &order, &info);
    detail::check_info("zgetrs", info);
    return x;
}

// 直接赋给 Options / Options2D 的 custom_multiply、custom_solve。
inline std::function<Matrix(const Matrix&, const Matrix&)> multiply_hook(const Settings& settings = {}) {
    return [settings](const Matrix& a, const Matrix& b) { return multiply(a, b, settings); };
}
inline std::function<Matrix(const Matrix&, const Matrix&)> solve_hook(const Settings& settings = {}) {
    return [settings](const Matrix& a, const Matrix& b) { return solve(a, b, settings); };
}

// 三个钩子一次设好：本征分解、稠密乘法、稠密线性求解都交给 MKL。O 是 rcwa::Options 或 rcwa::Options2D
// （一维的 linear 不是默认的 LUPartial 时，求解器不用 custom_solve）。
template <class O>
inline void use_mkl(O& options, const Settings& settings = {}) {
    options.custom_eig = eig_hook(settings);
    options.custom_multiply = multiply_hook(settings);
    options.custom_solve = solve_hook(settings);
}

}  // namespace mkl
}  // namespace stablercwa
