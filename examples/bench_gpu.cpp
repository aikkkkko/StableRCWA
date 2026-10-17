// examples/bench_gpu.cpp —— 同一份输入，CPU（StableRCWA）与 GPU（cuSOLVER / cuBLAS）各算一遍，计时并校验
//
// 测三个运算，对应 RCWA 里最花时间的三件事：
//   eig   一般复矩阵的本征分解（含本征向量）。CPU：eig_qr；GPU：gpu::eig
//   lu    解 A X = B，B 是 N×N（界面匹配时就是这样的方程组）。CPU：solve()；GPU：gpu::solve
//   gemm  N×N 复矩阵相乘。CPU：operator*；GPU：gpu::multiply
// GPU 一侧双精度、单精度各测一遍。单精度只用来估计“双精度吞吐不受限的显卡能到什么水平”，
// 它的残差在 1e-6 量级，不能代替双精度的结果。
//
// 用法（构建见 examples/build_gpu_win.bat）：
//   bench_gpu [--sizes 201,401,801,1201,1601,2401] [--threads 16] [--ops eig,lu,gemm]
//             [--cpu-eig-max N] [--no-gpu] [--detail 明细.json]
//
// 标准输出是一个 JSON 对象，时间单位是秒，没有测的位置填 null；进度和每一步的明细写到标准错误。
//   cpu1 / cpu16        CPU 用 1 个线程 / --threads 个线程（键名固定写 16）。eig_qr 本身是单线程的，只有 cpu1。
//   gpu / gpu_total     双精度：只算显卡上的例程 / 整个调用（含显存分配、上传、下载、主机端收尾）
//   gpu_single / gpu_single_total   单精度的同样两项；total 还包含主机上 double ⇄ float 的转换
//   res_*               本征对残差 max_j ||A v_j − λ_j v_j|| / (||A||_F ||v_j||)
//   berr_*              后向误差 ||A X − B||_F / (||A||_F ||X||_F + ||B||_F)
//   init_sec            一次性开销：建立 Context，加上三个例程在两种精度下各自第一次调用（小矩阵）的时间
// 小矩阵重复几次取最小值；大矩阵的 CPU 本征分解很慢，只跑一次。
// lu 的 CPU 时间按 solve() 计：它与 gpu::solve 做的是同一件事（按列选主元的 LU 分解 + 全部右端的回代），
// 也是 CPU 上最快的走法。分两步的 lu_factor + lu_solve 结果逐位相同，但 lu_factor 还要统计元素增长因子，
// 分解那一段慢一些；它的时间另外写在标准错误和 --detail 文件里（lu_cpu_factor_solve1 / 16），不进主 JSON。
//
// 没有 NVIDIA 显卡的机器可以加 -DBENCH_NO_CUDA 编译，只得到 CPU 那几列。
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "stablercwa/eig_qr.hpp"
#include "stablercwa/lu.hpp"
#include "stablercwa/matrix.hpp"
#include "stablercwa/parallel.hpp"
#ifndef BENCH_NO_CUDA
#include "stablercwa/gpu/cuda_backend.hpp"
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#endif

namespace ss = stablercwa;
using ss::cplx;
using ss::Matrix;

namespace {

const double kNaN = std::numeric_limits<double>::quiet_NaN();   // “没有测”

using Clock = std::chrono::steady_clock;
double since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

// ---------------------------------------------------------------- 测试矩阵

// RCWA 型的层矩阵（TE 偏振、吸收光栅）：A = diag(kx_m²) − ⟦ε⟧。
//   kx_m = sin θ − m·λ/Λ，m = −M … M（N = 2M+1），θ = 30°，λ/Λ = 0.07；
//   ⟦ε⟧(i, j) = ε_{i−j} 是介电常数的傅里叶系数排成的 Toeplitz 矩阵。一个周期内占空比 f = 0.3 的部分
//   是吸收介质 ε_r = (1.46 + 0.05i)²，其余是 ε_g = 1：
//       ε_0 = f ε_r + (1 − f) ε_g，   ε_h = (ε_r − ε_g) sin(π h f) / (π h)   (h ≠ 0)。
// ε_r 是复数，所以 A 是一般复矩阵（复对称，但不是厄米矩阵）。
Matrix rcwa_layer_matrix(int n) {
    const double theta = 30.0 * ss::kPi / 180.0, ratio = 0.07, fill = 0.3;
    const cplx eps_r = cplx(1.46, 0.05) * cplx(1.46, 0.05), eps_g = 1.0;
    ss::Vector eps(n);
    eps[0] = fill * eps_r + (1.0 - fill) * eps_g;
    for (int h = 1; h < n; ++h) eps[h] = (eps_r - eps_g) * (std::sin(ss::kPi * h * fill) / (ss::kPi * h));
    Matrix a(n, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) a(i, j) = -eps[std::abs(i - j)];
    for (int i = 0; i < n; ++i) {
        const double kx = std::sin(theta) - (i - n / 2) * ratio;
        a(i, i) += kx * kx;
    }
    return a;
}

// 实部、虚部各自在 [−1, 1) 上均匀分布的稠密矩阵。只用 mt19937_64 的原始输出（标准规定了它的每一位），
// 不经过标准库的分布类，所以不同编译器生成的矩阵完全相同。
Matrix random_matrix(int rows, int cols, std::uint64_t seed) {
    std::mt19937_64 gen(seed);
    const auto uniform = [&] { return static_cast<double>(gen() >> 11) * (2.0 / 9007199254740992.0) - 1.0; };
    Matrix m(rows, cols);
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i) {
            const double re = uniform(), im = uniform();
            m(i, j) = cplx(re, im);
        }
    return m;
}

// ---------------------------------------------------------------- 校验量

// 本征对残差 max_j ||A v_j − λ_j v_j||_2 / (||A||_F ||v_j||_2)。
double eig_residual(const Matrix& a, const ss::EigResult& r) {
    const int n = a.rows();
    if (r.vectors.rows() != n || r.vectors.cols() != n) return kNaN;
    const Matrix av = a * r.vectors;
    const double na = ss::norm_fro(a);
    double worst = 0.0;
    for (int j = 0; j < n; ++j) {
        const cplx* v = r.vectors.col(j);
        const cplx* w = av.col(j);
        double rr = 0.0, vv = 0.0;
        for (int i = 0; i < n; ++i) {
            rr += std::norm(w[i] - r.values[j] * v[i]);
            vv += std::norm(v[i]);
        }
        const double q = std::sqrt(rr) / (na * std::sqrt(vv));
        if (!(q <= worst)) worst = q;                 // 写成这样，NaN 也会被带出来
    }
    return worst;
}

// ||X − Y||_F / ||Y||_F
double relative_difference(const Matrix& x, const Matrix& y) {
    const double ny = ss::norm_fro(y);
    return ny == 0.0 ? kNaN : ss::norm_fro(x - y) / ny;
}

// 两组本征值的差别：每个 a_j 到 b 中最近一个的距离，取最大，再除以 max|b|。两边的排列次序可以不同。
[[maybe_unused]] double spectrum_distance(const ss::Vector& a, const ss::Vector& b) {
    if (a.size() != b.size() || a.empty()) return kNaN;
    double scale = 0.0, worst = 0.0;
    for (const cplx& z : b) scale = std::max(scale, std::abs(z));
    for (const cplx& z : a) {
        double best = std::numeric_limits<double>::infinity();
        for (const cplx& y : b) best = std::min(best, std::norm(z - y));
        worst = std::max(worst, std::sqrt(best));
    }
    return scale == 0.0 ? kNaN : worst / scale;
}

// ---------------------------------------------------------------- 机器信息

std::string trim(const std::string& s) {
    const std::size_t b = s.find_first_not_of(' ');
    if (b == std::string::npos) return "";
    return s.substr(b, s.find_last_not_of(' ') - b + 1);
}

// 处理器型号：CPUID 0x80000002–4 给出的 48 个字符。
std::string cpu_name() {
    unsigned regs[12] = {0};
    bool ok = false;
#if defined(_MSC_VER)
    int r[4];
    __cpuid(r, static_cast<int>(0x80000000u));
    if (static_cast<unsigned>(r[0]) >= 0x80000004u) {
        for (int k = 0; k < 3; ++k) {
            __cpuid(r, static_cast<int>(0x80000002u + k));
            for (int q = 0; q < 4; ++q) regs[4 * k + q] = static_cast<unsigned>(r[q]);
        }
        ok = true;
    }
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    if (__get_cpuid_max(0x80000000u, nullptr) >= 0x80000004u) {
        for (unsigned k = 0; k < 3; ++k)
            __get_cpuid(0x80000002u + k, &regs[4 * k], &regs[4 * k + 1], &regs[4 * k + 2], &regs[4 * k + 3]);
        ok = true;
    }
#endif
    std::string name = "unknown CPU";
    if (ok) {
        char text[49];
        std::memcpy(text, regs, 48);
        text[48] = '\0';
        name = trim(text);
    }
    return name + " (" + std::to_string(ss::hardware_threads()) + " hardware threads)";
}

#define BENCH_STR2(x) #x
#define BENCH_STR(x) BENCH_STR2(x)

// 编译器与优化选项。选项由构建脚本用 -DBENCH_OPT=-O2 之类的宏传进来。
std::string compiler_name() {
    std::string s;
#if defined(_MSC_VER) && !defined(__clang__)
    s = "MSVC " + std::to_string(_MSC_VER / 100) + "." + std::to_string(_MSC_VER % 100) + " (cl " +
        std::to_string(_MSC_FULL_VER) + ")";
#elif defined(__clang__)
    s = std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
    s = std::string("g++ ") + __VERSION__;
#if defined(__MINGW64__)
    s += " (MinGW-w64)";
#endif
#else
    s = "unknown compiler";
#endif
#ifdef BENCH_OPT
    s += std::string(" ") + BENCH_STR(BENCH_OPT);
#endif
    return s;
}

// ---------------------------------------------------------------- JSON 输出

std::string json_string(const std::string& s) {
    std::string out = "\"";
    for (const char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        if (static_cast<unsigned char>(c) >= 0x20) out += c;
    }
    return out + "\"";
}
std::string json_number(double v) {
    if (!std::isfinite(v)) return "null";
    char buf[40];
    std::snprintf(buf, sizeof buf, "%.6g", v);
    return buf;
}
std::string json_array(const std::vector<double>& v) {
    std::string s = "[";
    for (std::size_t k = 0; k < v.size(); ++k) s += (k ? "," : "") + json_number(v[k]);
    return s + "]";
}
std::string json_array(const std::vector<int>& v) {
    std::string s = "[";
    for (std::size_t k = 0; k < v.size(); ++k) s += (k ? "," : "") + std::to_string(v[k]);
    return s + "]";
}

// ---------------------------------------------------------------- 命令行

struct Args {
    std::vector<int> sizes{201, 401, 801, 1201, 1601, 2401};
    int threads = 16;
    bool eig = true, lu = true, gemm = true;
    int cpu_eig_max = std::numeric_limits<int>::max();   // N 超过它就不测 CPU 的本征分解
    bool gpu = true;
    std::string detail;                                  // 非空时把每一步的时间分解另存成一个 JSON 文件
};

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream in(s);
    for (std::string item; std::getline(in, item, ',');)
        if (!item.empty()) out.push_back(item);
    return out;
}

Args parse(int argc, char** argv) {
    Args a;
    for (int k = 1; k < argc; ++k) {
        const std::string opt = argv[k];
        const auto value = [&]() -> std::string {
            if (k + 1 >= argc) throw std::invalid_argument("missing value after " + opt);
            return argv[++k];
        };
        if (opt == "--sizes") {
            a.sizes.clear();
            for (const auto& s : split(value())) a.sizes.push_back(std::stoi(s));
        } else if (opt == "--threads") {
            a.threads = std::stoi(value());
        } else if (opt == "--ops") {
            a.eig = a.lu = a.gemm = false;
            for (const auto& s : split(value())) {
                if (s == "eig") a.eig = true;
                else if (s == "lu") a.lu = true;
                else if (s == "gemm") a.gemm = true;
                else throw std::invalid_argument("unknown operation: " + s);
            }
        } else if (opt == "--cpu-eig-max") {
            a.cpu_eig_max = std::stoi(value());
        } else if (opt == "--no-gpu") {
            a.gpu = false;
        } else if (opt == "--detail") {
            a.detail = value();
        } else {
            throw std::invalid_argument("unknown option: " + opt);
        }
    }
    for (const int n : a.sizes)
        if (n < 1) throw std::invalid_argument("sizes must be positive");
    if (a.sizes.empty() || a.threads < 1) throw std::invalid_argument("need at least one size and one thread");
    return a;
}

// ---------------------------------------------------------------- 计时

// 重复次数：小矩阵多跑几次取最小值，大矩阵跑一两次。显卡上的 lu 和 gemm 很快，多重复几次，
// 好把桌面程序偶尔占用显卡造成的抖动压下去。
[[maybe_unused]] int reps_gpu(int n) { return n <= 401 ? 5 : n <= 1201 ? 3 : 2; }
[[maybe_unused]] int reps_gpu_fast(int n) { return n <= 1201 ? 9 : 5; }
int reps_cpu(int n) { return n <= 401 ? 5 : n <= 1201 ? 2 : 1; }
int reps_cpu_eig(int n) { return n <= 401 ? 3 : 1; }

template <class F>
double best_of(int reps, F&& run) {
    double best = std::numeric_limits<double>::infinity();
    for (int k = 0; k < reps; ++k) {
        const auto t0 = Clock::now();
        run();
        best = std::min(best, since(t0));
    }
    return best;
}

// 一个运算在一种 GPU 精度下的测量结果。
struct GpuRun {
    double compute = kNaN, total = kNaN;       // 各自取多次中的最小值
    double first = kNaN;                       // 第一次调用的计算时间（看这个规模下有没有额外的首次开销）
    double convert = kNaN, h2d = kNaN, d2h = kNaN;   // 取自 total 最小的那一次
    double device_mb = kNaN, host_work_mb = kNaN;
};

#ifndef BENCH_NO_CUDA
// 重复调用 call(Timing&)，记录最小的计算时间、最小的整体时间和后者对应的时间分解。
template <class F>
GpuRun time_gpu(int reps, F&& call) {
    GpuRun g;
    g.compute = g.total = std::numeric_limits<double>::infinity();
    for (int k = 0; k < reps; ++k) {
        ss::gpu::Timing t;
        const auto t0 = Clock::now();
        call(t);
        const double wall = since(t0);          // 调用者实际等待的时间
        g.compute = std::min(g.compute, t.compute);
        if (k == 0) g.first = t.compute;
        if (wall < g.total) {
            g.total = wall;
            g.convert = t.convert;
            g.h2d = t.h2d;
            g.d2h = t.d2h;
        }
        g.device_mb = static_cast<double>(t.device_bytes) / 1048576.0;
        g.host_work_mb = static_cast<double>(t.host_work_bytes) / 1048576.0;
    }
    return g;
}
#endif

[[maybe_unused]] void log_gpu(const char* what, const GpuRun& g) {
    std::fprintf(stderr,
                 "    %-16s compute %9.4f s (first call %.4f) | total %9.4f s (convert %.4f, h2d %.4f, d2h %.4f) | "
                 "device %.0f MiB, host work %.1f MiB\n",
                 what, g.compute, g.first, g.total, g.convert, g.h2d, g.d2h, g.device_mb, g.host_work_mb);
}

// 每个运算、每个 N 一格的全部数字。
struct Series {
    std::vector<double> cpu1, cpu16, gpu, gpu_total, gpu_single, gpu_single_total;
    std::vector<double> acc_cpu, acc_gpu, acc_gpu_single;   // 残差或后向误差
    std::vector<GpuRun> d, s;                               // 明细：双精度、单精度
    std::vector<double> cross_d, cross_s;                   // GPU 结果与 CPU 结果的相对差别
    void push(double c1, double c16, const GpuRun& gd, const GpuRun& gs, double ac, double ad, double as_, double xd, double xs) {
        cpu1.push_back(c1);
        cpu16.push_back(c16);
        gpu.push_back(gd.compute);
        gpu_total.push_back(gd.total);
        gpu_single.push_back(gs.compute);
        gpu_single_total.push_back(gs.total);
        acc_cpu.push_back(ac);
        acc_gpu.push_back(ad);
        acc_gpu_single.push_back(as_);
        d.push_back(gd);
        s.push_back(gs);
        cross_d.push_back(xd);
        cross_s.push_back(xs);
    }
};

std::string detail_runs(const std::vector<GpuRun>& runs) {
    std::vector<double> convert, h2d, compute, d2h, total, device, host;
    for (const GpuRun& g : runs) {
        convert.push_back(g.convert);
        h2d.push_back(g.h2d);
        compute.push_back(g.compute);
        d2h.push_back(g.d2h);
        total.push_back(g.total);
        device.push_back(g.device_mb);
        host.push_back(g.host_work_mb);
    }
    return "{\"convert\":" + json_array(convert) + ",\"h2d\":" + json_array(h2d) + ",\"compute\":" + json_array(compute) +
           ",\"d2h\":" + json_array(d2h) + ",\"total\":" + json_array(total) + ",\"device_mib\":" + json_array(device) +
           ",\"host_work_mib\":" + json_array(host) + "}";
}
std::string detail_series(const Series& x) {
    return "{\"double\":" + detail_runs(x.d) + ",\"single\":" + detail_runs(x.s) + ",\"gpu_vs_cpu_double\":" +
           json_array(x.cross_d) + ",\"gpu_vs_cpu_single\":" + json_array(x.cross_s) + "}";
}

int run(const Args& args) {
    const int T = args.threads;
    int failures = 0;
    const auto expect = [&](bool ok, const char* what, int n, double value) {
        if (ok) return;
        ++failures;
        std::fprintf(stderr, "    CHECK FAILED: %s at N=%d (%.3e)\n", what, n, value);
    };

    std::string gpu_name = "none", cuda_name = "none";
    double init_sec = kNaN;
#ifndef BENCH_NO_CUDA
    namespace gpu = ss::gpu;
    std::unique_ptr<gpu::Context> ctx;
    if (args.gpu) {
        // 一次性开销：建立上下文和句柄，再让三个例程在两种精度下各跑一次小矩阵（库在第一次调用时才真正初始化）。
        const auto t0 = Clock::now();
        ctx.reset(new gpu::Context());
        const double t_context = since(t0);
        const Matrix a = rcwa_layer_matrix(65), g = random_matrix(65, 65, 7);
        double first[2][3];
        int p = 0;
        for (const gpu::Precision prec : {gpu::Precision::Double, gpu::Precision::Single}) {
            ctx->set_precision(prec);
            auto t1 = Clock::now();
            (void)gpu::eig(*ctx, a);
            first[p][0] = since(t1);
            t1 = Clock::now();
            (void)gpu::solve(*ctx, g, a);
            first[p][1] = since(t1);
            t1 = Clock::now();
            (void)gpu::multiply(*ctx, a, g);
            first[p][2] = since(t1);
            ++p;
        }
        init_sec = since(t0);
        gpu_name = ctx->device_name() + " (" + std::to_string(ctx->memory_total() >> 20) + " MiB)";
        cuda_name = ctx->library_versions();
        std::fprintf(stderr, "GPU: %s | %s | free memory %.0f MiB\n", gpu_name.c_str(), cuda_name.c_str(),
                     static_cast<double>(ctx->memory_free()) / 1048576.0);
        std::fprintf(stderr,
                     "one-off initialisation %.3f s = context %.3f s + first calls at N=65: double eig %.3f, solve %.3f, gemm %.3f; "
                     "single eig %.3f, solve %.3f, gemm %.3f\n",
                     init_sec, t_context, first[0][0], first[0][1], first[0][2], first[1][0], first[1][1], first[1][2]);
    }
    const bool use_gpu = static_cast<bool>(ctx);
#endif
    const std::string cpu = cpu_name(), compiler = compiler_name();
    std::fprintf(stderr, "CPU: %s | compiler: %s | CPU threads for the parallel runs: %d\n", cpu.c_str(), compiler.c_str(), T);

    Series eig, lu, gemm;
    std::vector<double> eig_values_d, eig_values_s;   // GPU 与 CPU 本征值集合的差别
    std::vector<double> lu_two_step1, lu_two_step16;  // 同一个方程组分两步（lu_factor + lu_solve）求解的 CPU 时间

    for (const int n : args.sizes) {
        std::fprintf(stderr, "N = %d\n", n);
        const Matrix a = rcwa_layer_matrix(n);          // 层矩阵：eig 的输入，lu 的右端，gemm 的左因子
        const Matrix g = random_matrix(n, n, 20261008u + static_cast<std::uint64_t>(n));   // lu 的系数矩阵，gemm 的右因子

        // ---- eig ----
        if (args.eig) {
            double c1 = kNaN, rc = kNaN, rd = kNaN, rs = kNaN, vd = kNaN, vs = kNaN;
            GpuRun gd, gs;
            ss::EigResult rcpu;
            if (n <= args.cpu_eig_max) {
                ss::set_num_threads(1);
                c1 = best_of(reps_cpu_eig(n), [&] { rcpu = ss::eig_qr(a); });
                ss::set_num_threads(T);                 // 校验用的乘法可以并行
                rc = eig_residual(a, rcpu);
                std::fprintf(stderr, "  eig  cpu 1 thread  %9.4f s | residual %.2e | info %d\n", c1, rc, rcpu.info);
                expect(rcpu.info == 0 && rc < 1e-12, "CPU eig residual", n, rc);
            }
#ifndef BENCH_NO_CUDA
            if (use_gpu) {
                ss::set_num_threads(T);
                ss::EigResult r;
                ctx->set_precision(gpu::Precision::Double);
                gd = time_gpu(reps_gpu(n), [&](gpu::Timing& t) { r = gpu::eig(*ctx, a, true, &t); });
                rd = eig_residual(a, r);
                if (!rcpu.values.empty()) vd = spectrum_distance(r.values, rcpu.values);
                log_gpu("eig  gpu double", gd);
                std::fprintf(stderr, "    residual %.2e | info %d | eigenvalues vs cpu %.2e\n", rd, r.info, vd);
                expect(r.info == 0 && rd < 1e-12, "GPU double eig residual", n, rd);

                ctx->set_precision(gpu::Precision::Single);
                gs = time_gpu(reps_gpu(n), [&](gpu::Timing& t) { r = gpu::eig(*ctx, a, true, &t); });
                rs = eig_residual(a, r);
                if (!rcpu.values.empty()) vs = spectrum_distance(r.values, rcpu.values);
                log_gpu("eig  gpu single", gs);
                std::fprintf(stderr, "    residual %.2e | info %d | eigenvalues vs cpu %.2e\n", rs, r.info, vs);
                expect(r.info == 0 && rs < 1e-3, "GPU single eig residual", n, rs);
            }
#endif
            eig.push(c1, kNaN, gd, gs, rc, rd, rs, kNaN, kNaN);
            eig_values_d.push_back(vd);
            eig_values_s.push_back(vs);
        }

        // ---- lu ----
        if (args.lu) {
            double c1, c16, bc, bd = kNaN, bs = kNaN, xd = kNaN, xs = kNaN;
            GpuRun gd, gs;
            Matrix x_cpu;
            ss::set_num_threads(1);
            c1 = best_of(reps_cpu(n), [&] { x_cpu = ss::solve(g, a); });
            ss::set_num_threads(T);
            c16 = best_of(reps_cpu(n) + 1, [&] { x_cpu = ss::solve(g, a); });
            bc = ss::solve_backward_error(g, x_cpu, a);
            std::fprintf(stderr, "  lu   cpu 1 thread  %9.4f s | %d threads %9.4f s | backward error %.2e\n", c1, T, c16, bc);
            expect(bc < 1e-12, "CPU lu backward error", n, bc);
            {   // 同一件事分两步做：lu_factor（多统计元素增长因子）+ lu_solve
                Matrix x_two;
                const auto two_step = [&] {
                    const ss::LU f = ss::lu_factor(g);
                    x_two = ss::lu_solve(f, a);
                };
                ss::set_num_threads(1);
                const double s1 = best_of(reps_cpu(n), two_step);
                ss::set_num_threads(T);
                const double s16 = best_of(reps_cpu(n) + 1, two_step);
                std::fprintf(stderr, "       lu_factor + lu_solve: %9.4f s | %d threads %9.4f s | differs from solve() by %.1e\n", s1, T, s16,
                             relative_difference(x_two, x_cpu));
                lu_two_step1.push_back(s1);
                lu_two_step16.push_back(s16);
            }
#ifndef BENCH_NO_CUDA
            if (use_gpu) {
                Matrix x;
                ctx->set_precision(gpu::Precision::Double);
                gd = time_gpu(reps_gpu_fast(n), [&](gpu::Timing& t) { x = gpu::solve(*ctx, g, a, &t); });
                bd = ss::solve_backward_error(g, x, a);
                xd = relative_difference(x, x_cpu);
                log_gpu("lu   gpu double", gd);
                std::fprintf(stderr, "    backward error %.2e | solution vs cpu %.2e\n", bd, xd);
                expect(bd < 1e-12, "GPU double lu backward error", n, bd);

                ctx->set_precision(gpu::Precision::Single);
                gs = time_gpu(reps_gpu_fast(n), [&](gpu::Timing& t) { x = gpu::solve(*ctx, g, a, &t); });
                bs = ss::solve_backward_error(g, x, a);
                xs = relative_difference(x, x_cpu);
                log_gpu("lu   gpu single", gs);
                std::fprintf(stderr, "    backward error %.2e | solution vs cpu %.2e\n", bs, xs);
                expect(bs < 1e-3, "GPU single lu backward error", n, bs);
            }
#endif
            lu.push(c1, c16, gd, gs, bc, bd, bs, xd, xs);
        }

        // ---- gemm ----
        if (args.gemm) {
            double c1, c16, xd = kNaN, xs = kNaN;
            GpuRun gd, gs;
            Matrix c_cpu;
            ss::set_num_threads(1);
            c1 = best_of(reps_cpu(n), [&] { c_cpu = a * g; });
            ss::set_num_threads(T);
            c16 = best_of(reps_cpu(n) + 1, [&] { c_cpu = a * g; });
            std::fprintf(stderr, "  gemm cpu 1 thread  %9.4f s | %d threads %9.4f s\n", c1, T, c16);
#ifndef BENCH_NO_CUDA
            if (use_gpu) {
                Matrix c;
                ctx->set_precision(gpu::Precision::Double);
                gd = time_gpu(reps_gpu_fast(n), [&](gpu::Timing& t) { c = gpu::multiply(*ctx, a, g, &t); });
                xd = relative_difference(c, c_cpu);
                log_gpu("gemm gpu double", gd);
                std::fprintf(stderr, "    product vs cpu %.2e\n", xd);
                expect(xd < 1e-12, "GPU double gemm difference", n, xd);

                ctx->set_precision(gpu::Precision::Single);
                gs = time_gpu(reps_gpu_fast(n), [&](gpu::Timing& t) { c = gpu::multiply(*ctx, a, g, &t); });
                xs = relative_difference(c, c_cpu);
                log_gpu("gemm gpu single", gs);
                std::fprintf(stderr, "    product vs cpu %.2e\n", xs);
                expect(xs < 1e-3, "GPU single gemm difference", n, xs);
            }
#endif
            gemm.push(c1, c16, gd, gs, kNaN, kNaN, kNaN, xd, xs);
        }
    }
    ss::set_num_threads(1);

    // 没有测的运算也照样输出，各项全是 null，保持 JSON 的形状不变。
    const auto pad = [&](Series& x) {
        const GpuRun none;
        while (x.cpu1.size() < args.sizes.size()) x.push(kNaN, kNaN, none, none, kNaN, kNaN, kNaN, kNaN, kNaN);
    };
    pad(eig);
    pad(lu);
    pad(gemm);
    eig_values_d.resize(args.sizes.size(), kNaN);
    eig_values_s.resize(args.sizes.size(), kNaN);
    lu_two_step1.resize(args.sizes.size(), kNaN);
    lu_two_step16.resize(args.sizes.size(), kNaN);

    std::string out = "{\"machine\":{\"cpu\":" + json_string(cpu) + ",\"gpu\":" + json_string(gpu_name) + ",\"cuda\":" +
                      json_string(cuda_name) + ",\"compiler\":" + json_string(compiler) + "},\"init_sec\":" +
                      json_number(init_sec) + ",\"N\":" + json_array(args.sizes);
    out += ",\"eig\":{\"cpu1\":" + json_array(eig.cpu1) + ",\"gpu\":" + json_array(eig.gpu) + ",\"gpu_total\":" +
           json_array(eig.gpu_total) + ",\"gpu_single\":" + json_array(eig.gpu_single) + ",\"gpu_single_total\":" +
           json_array(eig.gpu_single_total) + ",\"res_cpu\":" + json_array(eig.acc_cpu) + ",\"res_gpu\":" +
           json_array(eig.acc_gpu) + ",\"res_gpu_single\":" + json_array(eig.acc_gpu_single) + "}";
    out += ",\"lu\":{\"cpu1\":" + json_array(lu.cpu1) + ",\"cpu16\":" + json_array(lu.cpu16) + ",\"gpu\":" + json_array(lu.gpu) +
           ",\"gpu_total\":" + json_array(lu.gpu_total) + ",\"gpu_single\":" + json_array(lu.gpu_single) +
           ",\"gpu_single_total\":" + json_array(lu.gpu_single_total) + ",\"berr_cpu\":" + json_array(lu.acc_cpu) +
           ",\"berr_gpu\":" + json_array(lu.acc_gpu) + ",\"berr_gpu_single\":" + json_array(lu.acc_gpu_single) + "}";
    out += ",\"gemm\":{\"cpu1\":" + json_array(gemm.cpu1) + ",\"cpu16\":" + json_array(gemm.cpu16) + ",\"gpu\":" +
           json_array(gemm.gpu) + ",\"gpu_total\":" + json_array(gemm.gpu_total) + ",\"gpu_single\":" +
           json_array(gemm.gpu_single) + ",\"gpu_single_total\":" + json_array(gemm.gpu_single_total) + "}}";
    std::printf("%s\n", out.c_str());

    if (!args.detail.empty()) {
        std::ofstream f(args.detail);
        f << "{\"N\":" << json_array(args.sizes) << ",\"threads\":" << T << ",\"eig\":" << detail_series(eig)
          << ",\"eig_values_vs_cpu_double\":" << json_array(eig_values_d) << ",\"eig_values_vs_cpu_single\":"
          << json_array(eig_values_s) << ",\"lu\":" << detail_series(lu) << ",\"lu_cpu_factor_solve1\":" << json_array(lu_two_step1)
          << ",\"lu_cpu_factor_solve16\":" << json_array(lu_two_step16) << ",\"gemm\":" << detail_series(gemm) << "}\n";
        if (!f) {
            std::fprintf(stderr, "could not write %s\n", args.detail.c_str());
            ++failures;
        }
    }
    if (failures) std::fprintf(stderr, "%d check(s) FAILED\n", failures);
    else std::fprintf(stderr, "all checks passed\n");
    return failures ? 2 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stderr, nullptr, _IONBF, 0);   // 进度即时可见，即使标准错误被重定向到文件
    try {
        return run(parse(argc, argv));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "bench_gpu: %s\n", e.what());
        return 1;
    }
}
