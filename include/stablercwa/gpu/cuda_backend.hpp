// stablercwa/gpu/cuda_backend.hpp —— 可选的 CUDA 后端：把三个稠密核心交给 NVIDIA 显卡
//
// RCWA 的时间几乎全花在三件事上：一般复矩阵的本征分解、解线性方程组、矩阵乘法。
// 本文件把它们原样转交给 CUDA Toolkit 自带的 cuSOLVER / cuBLAS，接口仍然是 Matrix 进、Matrix 出：
//
//   #include "stablercwa/gpu/cuda_backend.hpp"
//   namespace gpu = stablercwa::gpu;
//
//   gpu::Context ctx;                                    // 建一次，反复用；默认双精度、0 号显卡
//   stablercwa::EigResult r = gpu::eig(ctx, A);          // cusolverDnXgeev：本征值 + 右本征向量
//   stablercwa::Matrix    X = gpu::solve(ctx, A, B);     // cusolverDnXgetrf + cusolverDnXgetrs：解 A X = B
//   stablercwa::Matrix    C = gpu::multiply(ctx, A, B);  // cublas?gemm：C = A B
//
//   gpu::Timing t;                                       // 想知道时间花在哪：最后一个参数传 &t
//   r = gpu::eig(ctx, A, true, &t);                      // t.h2d / t.compute / t.d2h / t.total
//
// 精度开关放在 Context 上，调用处一个字都不用改：
//
//   gpu::Context ctx(gpu::Precision::Single);            // 或者随时 ctx.set_precision(gpu::Precision::Single)
//
// Single 模式下，输入先在主机上从 complex<double> 压成 complex<float>，上传，用同一批例程的 32 位
// 数据类型计算，下载后再放宽成 complex<double>。结果的类型不变，但只有单精度的准确度
// （本征对残差约 1e-6，解方程的后向误差 1e-7～1e-6），用途是估计“双精度吞吐不受限的显卡大概能多快”，
// 不能当作正式计算的结果。
// 两种精度走的是同一份模板代码（detail::Scalar<Real> 只负责挑数据类型和 gemm 例程）。
//
// 几点约定：
//   · 这是整个模块里唯一用到标准库以外东西的文件。不包含它，其余头文件照旧只依赖 C++17 标准库。
//   · 需要 CUDA Toolkit 12.6 或更新的版本（cusolverDnXgeev 从那一版开始提供）。这里没有自己写的 CUDA
//     内核，所以用普通的 C++ 编译器即可，不需要 nvcc：编译时加上 Toolkit 的 include 目录，
//     链接 cusolver、cublas、cudart 三个库。
//   · Matrix 里的 std::complex<double> 按列连续存放，每个元素实部在前、虚部在后，正是 cuSOLVER / cuBLAS
//     要的排法，所以双精度时数据不做任何转换，直接整块拷进显存。
//   · eig() 与 eig_qr() 的约定一致：本征向量按 2-范数归一；info = 0 表示全部收敛，info = k > 0 表示
//     QR 迭代没有全部收敛（此时不给本征向量，下标 >= k 的本征值有效）。本征值的排列次序与 eig_qr 不同。
//     cusolverDnXgeev 不是纯显卡算法：除了显存里的工作区，它还要一块约 2.2 N² 个复数的主机内存工作区，
//     调用期间主机上始终有一个线程在忙；显卡的功耗只在每次调用开头的一小段明显升高。
//     所以它的用时由显卡和处理器共同决定，Timing::compute 里也不全是显卡的时间。
//   · solve() 用按列选主元的 LU；矩阵奇异（出现精确为零的主元）时抛 gpu::Error，与 CPU 版 solve() 的做法相同。
//   · 形状不符抛 std::invalid_argument；找不到显卡、显存不够、CUDA 例程报错一律抛 gpu::Error
//     （std::runtime_error 的子类），信息里写明是哪一步、要多少显存、还剩多少。不会静默地给出错误结果。
//   · Timing 的每一段在读表之前都先 cudaStreamSynchronize，所以 compute 是算完的时间，不是“提交完”的时间。
//     total 是整个调用的墙钟时间，除三段之外还包含显存分配、工作区查询、精度转换和主机端的归一。
//   · Context 持有句柄和流，不可拷贝；同一个 Context 不要同时在两个线程里用。
//     建立 Context 和每个例程的第一次调用要加载、初始化库，是一次性的开销：库文件已在系统缓存里时
//     合计约 0.2–0.3 秒，开机后头一次运行要从磁盘读入几百兆字节的库，会更久。
//   · 显存不够时 cudaMalloc 报错，这里把它变成异常。但 Windows 上较新的驱动默认会在显存用完后
//     悄悄改用系统内存（不报错，只是变得很慢），所以大矩阵请自己对照 memory_free() 留出余量：
//     eig 约需 3 N² 个复数的显存，multiply 约 3 N²，solve 约 N² + N·(右端列数)。
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "../eig_types.hpp"
#include "../matrix.hpp"

namespace stablercwa {
namespace gpu {

// 在显卡上用哪种精度计算。接口两边始终是 complex<double>，Single 只影响显卡上的那一段。
enum class Precision { Double, Single };

// 一次调用的时间分解（秒）和内存用量（字节）。
struct Timing {
    double convert = 0.0;   // 主机上的精度转换（Double 模式不做转换，这一项只剩两次读表的间隔）
    double h2d = 0.0;       // 主机内存 → 显存
    double compute = 0.0;   // cuSOLVER / cuBLAS 例程本身，已等流排空
    double d2h = 0.0;       // 显存 → 主机内存
    double total = 0.0;     // 整个调用的墙钟时间：以上四项 + 显存分配、工作区查询、主机端归一
    std::size_t device_bytes = 0;      // 这次调用申请的显存总量
    std::size_t host_work_bytes = 0;   // cuSOLVER 要求的主机端工作区
};

// CUDA 一侧的任何失败。
class Error : public std::runtime_error {
public:
    explicit Error(const std::string& what) : std::runtime_error("stablercwa::gpu: " + what) {}
};

namespace detail {

// 数据是整块字节拷进显存的，所以只要求大小相同（实部在前、虚部在后是两边共同的约定）。
// 新版 Toolkit 给 cuDoubleComplex 加了更严的对齐要求，这里不涉及：主机上从不把 Matrix 的数组当作它来访问，
// 显存由 cudaMalloc 分配，对齐足够。
static_assert(sizeof(std::complex<double>) == sizeof(cuDoubleComplex), "std::complex<double> must match cuDoubleComplex in size");
static_assert(sizeof(std::complex<float>) == sizeof(cuComplex), "std::complex<float> must match cuComplex in size");

using Clock = std::chrono::steady_clock;
inline double seconds(Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<double>(to - from).count();
}

inline const char* status_name(cusolverStatus_t s) {
    switch (s) {
        case CUSOLVER_STATUS_SUCCESS: return "success";
        case CUSOLVER_STATUS_NOT_INITIALIZED: return "library not initialized";
        case CUSOLVER_STATUS_ALLOC_FAILED: return "allocation failed";
        case CUSOLVER_STATUS_INVALID_VALUE: return "invalid value";
        case CUSOLVER_STATUS_ARCH_MISMATCH: return "architecture mismatch";
        case CUSOLVER_STATUS_EXECUTION_FAILED: return "execution failed";
        case CUSOLVER_STATUS_INTERNAL_ERROR: return "internal error";
        case CUSOLVER_STATUS_NOT_SUPPORTED: return "not supported";
        default: return "unknown status";
    }
}

inline void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw Error(std::string(what) + " failed: " + cudaGetErrorString(e));
}
inline void check(cusolverStatus_t s, const char* what) {
    if (s != CUSOLVER_STATUS_SUCCESS)
        throw Error(std::string(what) + " failed: cuSOLVER status " + std::to_string(static_cast<int>(s)) + " (" +
                    status_name(s) + ")");
}
inline void check(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw Error(std::string(what) + " failed: cuBLAS status " + std::to_string(static_cast<int>(s)) + " (" +
                    cublasGetStatusString(s) + ")");
}

inline std::string megabytes(std::size_t bytes) {
    return std::to_string((bytes + (std::size_t{1} << 19)) >> 20) + " MiB";
}

// 一块显存，析构时释放。申请不到时报告想要多少、显卡还剩多少。
class DeviceBuffer {
public:
    DeviceBuffer(std::size_t bytes, const char* what, std::size_t* tally = nullptr) : bytes_(std::max<std::size_t>(bytes, 16)) {
        const cudaError_t e = cudaMalloc(&p_, bytes_);
        if (e != cudaSuccess) {
            p_ = nullptr;
            std::size_t free_b = 0, total_b = 0;
            const bool known = cudaMemGetInfo(&free_b, &total_b) == cudaSuccess;
            std::string msg = std::string("out of GPU memory allocating ") + megabytes(bytes_) + " for " + what;
            if (known) msg += " (free " + megabytes(free_b) + " of " + megabytes(total_b) + ")";
            throw Error(msg + ": " + cudaGetErrorString(e));
        }
        if (tally) *tally += bytes_;
    }
    ~DeviceBuffer() {
        if (p_) cudaFree(p_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void* get() const noexcept { return p_; }
    template <class T>
    T* as() const noexcept { return static_cast<T*>(p_); }

private:
    void* p_ = nullptr;
    std::size_t bytes_ = 0;
};

// 精度相关的东西只有这些：标量类型、cuSOLVER 的数据类型编号、gemm 例程。其余代码两种精度共用。
template <class Real>
struct Scalar;

template <>
struct Scalar<double> {
    using type = std::complex<double>;
    static constexpr cudaDataType cuda_type = CUDA_C_64F;
    static cublasStatus_t gemm(cublasHandle_t h, int m, int n, int k, const void* a, const void* b, void* c) {
        const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0);
        return cublasZgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, static_cast<const cuDoubleComplex*>(a), m,
                           static_cast<const cuDoubleComplex*>(b), k, &zero, static_cast<cuDoubleComplex*>(c), m);
    }
};

template <>
struct Scalar<float> {
    using type = std::complex<float>;
    static constexpr cudaDataType cuda_type = CUDA_C_32F;
    static cublasStatus_t gemm(cublasHandle_t h, int m, int n, int k, const void* a, const void* b, void* c) {
        const cuComplex one = make_cuComplex(1.0f, 0.0f), zero = make_cuComplex(0.0f, 0.0f);
        return cublasCgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, static_cast<const cuComplex*>(a), m,
                           static_cast<const cuComplex*>(b), k, &zero, static_cast<cuComplex*>(c), m);
    }
};

// 准备上传的主机数据。双精度时就是原数组本身，不拷贝；单精度时是一份压成 float 的副本。
template <class Real>
class HostInput {
public:
    HostInput(const cplx* src, std::size_t count) : count_(count) { convert(src, static_cast<Real*>(nullptr)); }
    const void* data() const noexcept { return p_; }
    std::size_t bytes() const noexcept { return count_ * sizeof(std::complex<Real>); }

private:
    void convert(const cplx* src, double*) { p_ = src; }
    void convert(const cplx* src, float*) {
        narrow_.reset(new std::complex<float>[count_]);
        const double* s = reinterpret_cast<const double*>(src);
        float* d = reinterpret_cast<float*>(narrow_.get());
        for (std::size_t k = 0; k < 2 * count_; ++k) d[k] = static_cast<float>(s[k]);
        p_ = narrow_.get();
    }
    std::size_t count_;
    const void* p_ = nullptr;
    std::unique_ptr<std::complex<float>[]> narrow_;
};

// 接收下载数据的主机缓冲。双精度时直接写进结果数组；单精度时先落在 float 缓冲里，finish() 再放宽。
template <class Real>
class HostOutput {
public:
    HostOutput(cplx* dst, std::size_t count) : dst_(dst), count_(count) { prepare(static_cast<Real*>(nullptr)); }
    void* data() noexcept { return p_; }
    std::size_t bytes() const noexcept { return count_ * sizeof(std::complex<Real>); }
    void finish() { widen(static_cast<Real*>(nullptr)); }

private:
    void prepare(double*) { p_ = dst_; }
    void prepare(float*) {
        narrow_.reset(new std::complex<float>[count_]);
        p_ = narrow_.get();
    }
    void widen(double*) {}
    void widen(float*) {
        const float* s = reinterpret_cast<const float*>(narrow_.get());
        double* d = reinterpret_cast<double*>(dst_);
        for (std::size_t k = 0; k < 2 * count_; ++k) d[k] = static_cast<double>(s[k]);
    }
    cplx* dst_;
    std::size_t count_;
    void* p_ = nullptr;
    std::unique_ptr<std::complex<float>[]> narrow_;
};

inline std::string dotted(int major, int minor, int patch = -1) {
    std::string s = std::to_string(major) + "." + std::to_string(minor);
    if (patch >= 0) s += "." + std::to_string(patch);
    return s;
}

}  // namespace detail

// 一块显卡上的句柄和流。创建要花一点时间（建立 CUDA 上下文、加载库），所以建一次、反复用。
class Context {
public:
    explicit Context(Precision precision = Precision::Double, int device = 0) : precision_(precision), device_(device) {
        int count = 0;
        const cudaError_t e = cudaGetDeviceCount(&count);
        if (e != cudaSuccess) throw Error(std::string("no usable CUDA device: ") + cudaGetErrorString(e));
        if (count <= 0) throw Error("no CUDA device found");
        if (device < 0 || device >= count)
            throw std::invalid_argument("gpu::Context: device index " + std::to_string(device) + " out of range (found " +
                                        std::to_string(count) + " device(s))");
        try {
            detail::check(cudaSetDevice(device_), "cudaSetDevice");
            cudaDeviceProp prop;
            detail::check(cudaGetDeviceProperties(&prop, device_), "cudaGetDeviceProperties");
            name_ = prop.name;
            std::size_t free_b = 0;
            detail::check(cudaMemGetInfo(&free_b, &total_), "cudaMemGetInfo");
            detail::check(cudaStreamCreate(&stream_), "cudaStreamCreate");
            detail::check(cusolverDnCreate(&solver_), "cusolverDnCreate");
            detail::check(cusolverDnCreateParams(&params_), "cusolverDnCreateParams");
            detail::check(cusolverDnSetStream(solver_, stream_), "cusolverDnSetStream");
            detail::check(cublasCreate(&blas_), "cublasCreate");
            detail::check(cublasSetStream(blas_, stream_), "cublasSetStream");
        } catch (...) {
            release();
            throw;
        }
    }
    ~Context() { release(); }
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    Precision precision() const noexcept { return precision_; }
    void set_precision(Precision p) noexcept { precision_ = p; }

    const std::string& device_name() const noexcept { return name_; }
    std::size_t memory_total() const noexcept { return total_; }   // 显存总量（字节）
    std::size_t memory_free() const {                              // 此刻的空闲显存（字节）
        activate();
        std::size_t free_b = 0, total_b = 0;
        detail::check(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
        return free_b;
    }
    // 运行时实际加载到的库版本，例如 "CUDA runtime 13.1, cuSOLVER 12.0.9, cuBLAS 13.2.2"。
    std::string library_versions() const {
        int rt = 0, so_major = 0, so_minor = 0, so_patch = 0, bl = 0;
        detail::check(cudaRuntimeGetVersion(&rt), "cudaRuntimeGetVersion");
        detail::check(cusolverGetProperty(MAJOR_VERSION, &so_major), "cusolverGetProperty");
        detail::check(cusolverGetProperty(MINOR_VERSION, &so_minor), "cusolverGetProperty");
        detail::check(cusolverGetProperty(PATCH_LEVEL, &so_patch), "cusolverGetProperty");
        detail::check(cublasGetVersion(blas_, &bl), "cublasGetVersion");
        return "CUDA runtime " + detail::dotted(rt / 1000, (rt % 1000) / 10) + ", cuSOLVER " +
               detail::dotted(so_major, so_minor, so_patch) + ", cuBLAS " +
               detail::dotted(bl / 10000, (bl / 100) % 100, bl % 100);
    }

    // 等这个 Context 的流上已提交的工作全部做完。
    void synchronize() const { detail::check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize"); }
    // 把本显卡设为当前线程的 CUDA 设备（“当前设备”是每个线程各自的状态）。每个例程开头调用一次。
    void activate() const { detail::check(cudaSetDevice(device_), "cudaSetDevice"); }

    cusolverDnHandle_t solver() const noexcept { return solver_; }
    cusolverDnParams_t params() const noexcept { return params_; }
    cublasHandle_t blas() const noexcept { return blas_; }
    cudaStream_t stream() const noexcept { return stream_; }

private:
    void release() noexcept {
        if (blas_) cublasDestroy(blas_);
        if (params_) cusolverDnDestroyParams(params_);
        if (solver_) cusolverDnDestroy(solver_);
        if (stream_) cudaStreamDestroy(stream_);
        blas_ = nullptr;
        params_ = nullptr;
        solver_ = nullptr;
        stream_ = nullptr;
    }

    Precision precision_;
    int device_;
    std::string name_;
    std::size_t total_ = 0;
    cudaStream_t stream_ = nullptr;
    cusolverDnHandle_t solver_ = nullptr;
    cusolverDnParams_t params_ = nullptr;
    cublasHandle_t blas_ = nullptr;
};

namespace detail {

inline void upload(const Context& ctx, void* device, const void* host, std::size_t bytes) {
    check(cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, ctx.stream()), "copy host -> device");
}
inline void download(const Context& ctx, void* host, const void* device, std::size_t bytes) {
    check(cudaMemcpyAsync(host, device, bytes, cudaMemcpyDeviceToHost, ctx.stream()), "copy device -> host");
}
inline std::size_t count_of(int rows, int cols) { return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols); }

// 各列按 2-范数归一（在双精度下做）。先除以最大分量再求平方和，避免溢出；全零或含非有限数的列不动。
inline void normalize_columns(Matrix& v) {
    const int n = v.rows();
    for (int j = 0; j < v.cols(); ++j) {
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

template <class Real>
EigResult eig_impl(Context& ctx, const Matrix& a, bool vectors, Timing& tm) {
    using T = typename Scalar<Real>::type;
    const cudaDataType type = Scalar<Real>::cuda_type;
    const int n = a.rows();
    const std::size_t nn = count_of(n, n);
    const auto t_begin = Clock::now();
    ctx.activate();

    EigResult r;
    r.values.assign(n, cplx{});
    if (vectors) r.vectors = Matrix(n, n);

    DeviceBuffer d_a(nn * sizeof(T), "eig: matrix", &tm.device_bytes);
    DeviceBuffer d_w(n * sizeof(T), "eig: eigenvalues", &tm.device_bytes);
    DeviceBuffer d_v(vectors ? nn * sizeof(T) : 0, "eig: eigenvectors", &tm.device_bytes);
    DeviceBuffer d_info(sizeof(int), "eig: info", &tm.device_bytes);

    // 不求左本征向量；右本征向量按需要。两块工作区的大小先问例程。
    const cusolverEigMode_t jobvl = CUSOLVER_EIG_MODE_NOVECTOR;
    const cusolverEigMode_t jobvr = vectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
    void* const vr = vectors ? d_v.get() : nullptr;
    const std::int64_t ldvr = vectors ? n : 1;
    std::size_t device_work = 0, host_work = 0;
    check(cusolverDnXgeev_bufferSize(ctx.solver(), ctx.params(), jobvl, jobvr, n, type, d_a.get(), n, type, d_w.get(), type,
                                     nullptr, 1, type, vr, ldvr, type, &device_work, &host_work),
          "cusolverDnXgeev_bufferSize");
    DeviceBuffer d_work(device_work, "eig: workspace", &tm.device_bytes);
    const std::unique_ptr<unsigned char[]> h_work(new unsigned char[std::max<std::size_t>(host_work, 16)]);
    tm.host_work_bytes = host_work;

    ctx.synchronize();
    const auto t0 = Clock::now();
    HostInput<Real> h_a(a.data(), nn);
    HostOutput<Real> h_w(r.values.data(), static_cast<std::size_t>(n));
    HostOutput<Real> h_v(vectors ? r.vectors.data() : nullptr, vectors ? nn : 0);
    const auto t1 = Clock::now();
    upload(ctx, d_a.get(), h_a.data(), h_a.bytes());
    ctx.synchronize();
    const auto t2 = Clock::now();
    check(cusolverDnXgeev(ctx.solver(), ctx.params(), jobvl, jobvr, n, type, d_a.get(), n, type, d_w.get(), type, nullptr, 1,
                          type, vr, ldvr, type, d_work.get(), device_work, h_work.get(), host_work, d_info.as<int>()),
          "cusolverDnXgeev");
    ctx.synchronize();
    const auto t3 = Clock::now();
    int info = 0;
    download(ctx, &info, d_info.get(), sizeof(int));
    download(ctx, h_w.data(), d_w.get(), h_w.bytes());
    if (vectors) download(ctx, h_v.data(), d_v.get(), h_v.bytes());
    ctx.synchronize();
    const auto t4 = Clock::now();
    h_w.finish();
    h_v.finish();
    const auto t5 = Clock::now();

    if (info < 0) throw Error("cusolverDnXgeev: illegal argument #" + std::to_string(-info));
    r.info = info;
    if (info > 0) r.vectors = Matrix();               // 没有全部收敛：不给本征向量
    else if (vectors) normalize_columns(r.vectors);

    tm.convert = seconds(t0, t1) + seconds(t4, t5);
    tm.h2d = seconds(t1, t2);
    tm.compute = seconds(t2, t3);
    tm.d2h = seconds(t3, t4);
    tm.total = seconds(t_begin, Clock::now());
    return r;
}

template <class Real>
Matrix solve_impl(Context& ctx, const Matrix& a, const Matrix& b, Timing& tm) {
    using T = typename Scalar<Real>::type;
    const cudaDataType type = Scalar<Real>::cuda_type;
    const int n = a.rows(), nrhs = b.cols();
    const std::size_t nn = count_of(n, n), nb = count_of(n, nrhs);
    const auto t_begin = Clock::now();
    ctx.activate();

    Matrix x(n, nrhs);
    DeviceBuffer d_a(nn * sizeof(T), "solve: matrix", &tm.device_bytes);
    DeviceBuffer d_b(nb * sizeof(T), "solve: right-hand sides", &tm.device_bytes);
    DeviceBuffer d_piv(static_cast<std::size_t>(n) * sizeof(std::int64_t), "solve: pivots", &tm.device_bytes);
    DeviceBuffer d_info(2 * sizeof(int), "solve: info", &tm.device_bytes);   // [0] 分解，[1] 回代
    std::size_t device_work = 0, host_work = 0;
    check(cusolverDnXgetrf_bufferSize(ctx.solver(), ctx.params(), n, n, type, d_a.get(), n, type, &device_work, &host_work),
          "cusolverDnXgetrf_bufferSize");
    DeviceBuffer d_work(device_work, "solve: workspace", &tm.device_bytes);
    const std::unique_ptr<unsigned char[]> h_work(new unsigned char[std::max<std::size_t>(host_work, 16)]);
    tm.host_work_bytes = host_work;

    ctx.synchronize();
    const auto t0 = Clock::now();
    HostInput<Real> h_a(a.data(), nn), h_b(b.data(), nb);
    HostOutput<Real> h_x(x.data(), nb);
    const auto t1 = Clock::now();
    upload(ctx, d_a.get(), h_a.data(), h_a.bytes());
    upload(ctx, d_b.get(), h_b.data(), h_b.bytes());
    ctx.synchronize();
    const auto t2 = Clock::now();
    // P A = L U（按列选主元），随后对全部右端做前代和回代；解覆盖在右端所在的显存上。
    check(cusolverDnXgetrf(ctx.solver(), ctx.params(), n, n, type, d_a.get(), n, d_piv.as<std::int64_t>(), type, d_work.get(),
                           device_work, h_work.get(), host_work, d_info.as<int>()),
          "cusolverDnXgetrf");
    check(cusolverDnXgetrs(ctx.solver(), ctx.params(), CUBLAS_OP_N, n, nrhs, type, d_a.get(), n, d_piv.as<std::int64_t>(), type,
                           d_b.get(), n, d_info.as<int>() + 1),
          "cusolverDnXgetrs");
    ctx.synchronize();
    const auto t3 = Clock::now();
    int info[2] = {0, 0};
    download(ctx, info, d_info.get(), sizeof(info));
    download(ctx, h_x.data(), d_b.get(), h_x.bytes());
    ctx.synchronize();
    const auto t4 = Clock::now();
    h_x.finish();
    const auto t5 = Clock::now();

    if (info[0] < 0) throw Error("cusolverDnXgetrf: illegal argument #" + std::to_string(-info[0]));
    if (info[0] > 0) throw Error("solve: matrix is singular (zero pivot at step " + std::to_string(info[0]) + ")");
    if (info[1] != 0) throw Error("cusolverDnXgetrs: illegal argument #" + std::to_string(-info[1]));

    tm.convert = seconds(t0, t1) + seconds(t4, t5);
    tm.h2d = seconds(t1, t2);
    tm.compute = seconds(t2, t3);
    tm.d2h = seconds(t3, t4);
    tm.total = seconds(t_begin, Clock::now());
    return x;
}

template <class Real>
Matrix multiply_impl(Context& ctx, const Matrix& a, const Matrix& b, Timing& tm) {
    using T = typename Scalar<Real>::type;
    const int m = a.rows(), p = a.cols(), n = b.cols();
    const std::size_t na = count_of(m, p), nb = count_of(p, n), nc = count_of(m, n);
    const auto t_begin = Clock::now();
    ctx.activate();

    Matrix c(m, n);
    DeviceBuffer d_a(na * sizeof(T), "multiply: left factor", &tm.device_bytes);
    DeviceBuffer d_b(nb * sizeof(T), "multiply: right factor", &tm.device_bytes);
    DeviceBuffer d_c(nc * sizeof(T), "multiply: product", &tm.device_bytes);

    ctx.synchronize();
    const auto t0 = Clock::now();
    HostInput<Real> h_a(a.data(), na), h_b(b.data(), nb);
    HostOutput<Real> h_c(c.data(), nc);
    const auto t1 = Clock::now();
    upload(ctx, d_a.get(), h_a.data(), h_a.bytes());
    upload(ctx, d_b.get(), h_b.data(), h_b.bytes());
    ctx.synchronize();
    const auto t2 = Clock::now();
    check(Scalar<Real>::gemm(ctx.blas(), m, n, p, d_a.get(), d_b.get(), d_c.get()), "cublas gemm");
    ctx.synchronize();
    const auto t3 = Clock::now();
    download(ctx, h_c.data(), d_c.get(), h_c.bytes());
    ctx.synchronize();
    const auto t4 = Clock::now();
    h_c.finish();
    const auto t5 = Clock::now();

    tm.convert = seconds(t0, t1) + seconds(t4, t5);
    tm.h2d = seconds(t1, t2);
    tm.compute = seconds(t2, t3);
    tm.d2h = seconds(t3, t4);
    tm.total = seconds(t_begin, Clock::now());
    return c;
}

}  // namespace detail

// 一般复矩阵的本征值与右本征向量（cusolverDnXgeev）。a 不被修改；精度由 ctx.precision() 决定。
// vectors = false 时只求本征值。返回值的含义与 eig_qr() 相同，只是不填 Z、T、sweeps。
inline EigResult eig(Context& ctx, const Matrix& a, bool vectors = true, Timing* timing = nullptr) {
    if (!a.square()) throw std::invalid_argument("gpu::eig: matrix must be square");
    if (!a.all_finite()) throw std::invalid_argument("gpu::eig: matrix has non-finite entries");
    Timing tm;
    EigResult r;
    if (a.rows() > 0) {
        if (ctx.precision() == Precision::Double) r = detail::eig_impl<double>(ctx, a, vectors, tm);
        else r = detail::eig_impl<float>(ctx, a, vectors, tm);
    }
    if (timing) *timing = tm;
    return r;
}

// 解 A X = B（cusolverDnXgetrf + cusolverDnXgetrs，按列选主元的 LU）。A 是 n×n，B 是 n×任意列。
inline Matrix solve(Context& ctx, const Matrix& a, const Matrix& b, Timing* timing = nullptr) {
    if (!a.square()) throw std::invalid_argument("gpu::solve: matrix must be square");
    if (a.rows() != b.rows()) throw std::invalid_argument("gpu::solve: shape mismatch");
    Timing tm;
    Matrix x;
    if (a.rows() == 0 || b.cols() == 0) x = Matrix(b.rows(), b.cols());
    else if (ctx.precision() == Precision::Double) x = detail::solve_impl<double>(ctx, a, b, tm);
    else x = detail::solve_impl<float>(ctx, a, b, tm);
    if (timing) *timing = tm;
    return x;
}

// C = A B（cublasZgemm；单精度时 cublasCgemm）。
inline Matrix multiply(Context& ctx, const Matrix& a, const Matrix& b, Timing* timing = nullptr) {
    if (a.cols() != b.rows()) throw std::invalid_argument("gpu::multiply: shape mismatch");
    Timing tm;
    Matrix c;
    if (a.rows() == 0 || a.cols() == 0 || b.cols() == 0) c = Matrix(a.rows(), b.cols());
    else if (ctx.precision() == Precision::Double) c = detail::multiply_impl<double>(ctx, a, b, tm);
    else c = detail::multiply_impl<float>(ctx, a, b, tm);
    if (timing) *timing = tm;
    return c;
}

}  // namespace gpu
}  // namespace stablercwa
