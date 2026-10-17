// tests/test_eig_qr_parallel.cpp —— eig_qr 并行化的验收：与冻结的旧版逐位比较、与线程数无关
//
//   c++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -I include -I tests tests/test_eig_qr_parallel.cpp -o build/test_eig_qr_parallel
//   ./build/test_eig_qr_parallel [--big] [--only T1,T3] [--dump DIR] [--nmin 192] [--threads 1,2,8] [--timeout 900]
//
// 判据只有一条：新的 stablercwa::eig_qr 在任何线程数下，返回的 values、vectors、Z、T、sweeps、info 的每一个字节，
// 都与冻结副本 stablercwa::ref_v1::eig_qr（tests/eig_qr_reference_v1.hpp，并行化之前的串行实现）相同；
// 带观察者时，每一次回调看到的 H、Z 也与旧版回调时逐位相同。比较用 memcmp，所以 +0 与 −0 也算不同；
// 唯一的例外是“元素全为 DBL_MAX、结果是一片 NaN”那一族：NaN 的符号位与载荷不比（见 g_nan_loose 的说明），其余一概逐字节。
// 测试只通过公共入口 eig_qr(a, options)、EigOptions、EigResult，以及 detail::eigqr 里保留下来的三个函数
// reduce_to_hessenberg、accumulate_reflectors、eigenvectors_from_schur 使用被测代码。
//
// 测试编号：
//   T1 逐位对照  T2 线程数不变性  T3 嵌套  T4 并发用户线程  T5 小矩阵与阈值  T6 观察者  T7 vectors / keep_transform 组合
//   T8 非收敛  T9 近亏损与精确简并  T10 极端量级与非法输入  T11 三个保留函数单独比较
// T2 不是单独一轮：它在 T1、T7、T8、T9、T10 的每一次比较里顺带做（各线程数的结果与第一个线程数的结果比），单独计数。
//
// 命令行：--filter S 只跑名字含 S 的矩阵；--suite-timeout S 每个套件的看门狗；--require-parallel 见 T12
//         --big      加上 2N = 578 … 1250 的矩阵和 n = 333 … 512 的随机矩阵（慢，几分钟）
//         --only L   只跑 L 里列出的测试（T1,T3 …；T2 随 T1 一起跑）
//         --dump D   另外读 D 目录里真实二维 P·Q 矩阵的转储（<name>.A.bin；--big 时多读 2N ≥ 882 的）
//         --nmin K   并行版的阈值 n_min（默认 192）。只用来挑测试尺寸（n_min−1、n_min、n_min+1 …），判据对所有尺寸一视同仁
//         --threads  线程数列表，默认 1,2,3,4,7,8（大矩阵上自动抽稀，见 threads_for）
//         --timeout  看门狗秒数：超过它还没跑完就打印当前测试名并以非零状态退出（用来把死锁变成失败）
//         --coverage 不跑测试，列出目录里 n >= 100 的矩阵的迭代窗口统计（l > 0 的 sweep、窗口向上扩张次数）
//         --times    T1 等套件里逐个矩阵打印用时（调试用时间预算用）
// 默认运行（不加 --big）：被测的 eig_qr 若是串行实现，M1 Pro 上约 20–25 s（旧版和新版共算约 3.5 个“2N=1250”的完整分解，结束时有统计；
// 机器上有别的任务时按负载放大）；并行实现落地后新版那一半会变快。--big 另加几分钟。
// 退出状态：全部通过 0；有失败 1；看门狗超时 3；命令行错误 2。
#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <typeinfo>
#include <vector>

#include "stablercwa/eig.hpp"
#include "stablercwa/eig_qr.hpp"
#include "stablercwa/lu.hpp"
#include "stablercwa/parallel.hpp"
#include "stablercwa/rcwa1d.hpp"
#include "stablercwa/svd.hpp"

#include "eig_qr_reference_v1.hpp"

// 逐位比较的前提：新旧两份代码里同一个表达式被编译成同样的浮点运算。乘加收缩（FMA contraction）若在内联之后由后端决定，
// 两份副本、或并行版的分块内核与原版，会收缩得不一样，与算法对错无关（实测，被测代码见括号）：
//   MinGW g++ 13.1 -O3 -march=native（-std=c++17 与 gnu++17 都一样：GCC 对 C++ 不因 ISO 方言关掉收缩）：串行旧版自己 T11 失败 804 次；
//   同上，原型：T1 失败 4229 次；MSVC /O2 /arch:AVX2 /fp:contract，原型：T1 失败 94 次；Apple clang -O3 -ffp-contract=fast，原型：T1 98、T6 19 次。
//   加 -ffp-contract=off（GCC/Clang）或不用 /fp:contract（MSVC）后全部通过。
// 所以能认出来的收缩方式直接拒绝编译；确实加了 -ffp-contract=off 的，定义 EIGQR_TEST_FP_CONTRACT_OFF 放行。
// （Clang 的 -ffp-contract=fast 没有宏可认：请在构建脚本里给本测试固定 -ffp-contract=off。）
#if !defined(EIGQR_TEST_FP_CONTRACT_OFF) &&     ((defined(__GNUC__) && !defined(__clang__) && defined(__FP_FAST_FMA)) || defined(_M_FP_CONTRACT) || defined(__FAST_MATH__))
#error "test_eig_qr_parallel: FMA contraction is enabled (GCC with an FMA target, MSVC /fp:contract or -ffast-math); add -ffp-contract=off -DEIGQR_TEST_FP_CONTRACT_OFF"
#endif

#if defined(_WIN32)   // 进程 CPU 时间：Windows 的 std::clock() 是墙钟（CPU 列会恒等于墙钟），改用 GetProcessTimes；不 include <windows.h>
extern "C" __declspec(dllimport) void* __stdcall GetCurrentProcess();
extern "C" __declspec(dllimport) int __stdcall GetProcessTimes(void*, unsigned long long*, unsigned long long*, unsigned long long*, unsigned long long*);
#endif

using namespace stablercwa;

// ---------------------------------------------------------------------------------------------
// 输出（同 test_eig_qr.cpp：文字一律经 say()，免得 MSVC 按系统代码页去读带中文的格式串）
#if defined(__GNUC__) && !defined(__MINGW32__)
[[gnu::format(printf, 1, 2)]]
#endif
static void say(const char* format, ...) {
    std::va_list args;
    va_start(args, format);
    std::vprintf(format, args);
    va_end(args);
    std::fflush(stdout);
}

#if defined(__GNUC__) && !defined(__MINGW32__)
[[gnu::format(printf, 1, 2)]]
#endif
static std::string fmt(const char* format, ...) {
    char buf[1024];
    std::va_list args;
    va_start(args, format);
    std::vsnprintf(buf, sizeof buf, format, args);
    va_end(args);
    return buf;
}

// ---------------------------------------------------------------------------------------------
// 命令行与计数
struct Config {
    bool big = false;
    int nmin = 192;
    std::vector<int> threads{1, 2, 3, 4, 7, 8};
    std::set<std::string> only;
    std::string dump;
    double timeout = 900.0;
    bool times = false;
    bool coverage = false;
    std::string filter;            // 只跑名字里含这个子串的矩阵（复现单个失败用）
    double suite_timeout = 300.0;  // 每个套件（T1、T3 …）各自的看门狗秒数
    bool require_parallel = false; // T12 确认不了“并行路径真的执行了”时算失败（并行实现落地后应当默认打开）
};
static Config g_cfg;

struct Tally {
    std::string name;
    long long matrices = 0;      // 不同的矩阵（或输入）个数
    long long runs = 0;          // 被测代码的调用次数
    long long comparisons = 0;   // 逐位比较的次数（一次比较含 values、vectors、Z、T、sweeps、info 或回调序列）
    long long bytes = 0;         // 比较过的字节数（回调里 n > kExactN 的 H、Z 是 128 位散列，字节数照实累计）
    long long checks = 0;        // 其它断言
    long long failures = 0;
    double seconds = 0, cpu = 0;   // 墙钟、进程 CPU 时间（各线程之和）
};
static Tally g_t2;                       // T2：跨套件累计
static Tally* g_cur = nullptr;           // 当前测试；失败记在它名下
static std::mutex g_fail_mutex;
static int g_printed = 0;
static std::atomic<const char*> g_stage{"启动"};
static std::atomic<double> g_stage_start{0.0};
static const int kMaxPrinted = 14;

static void fail_in(Tally& t, const std::string& msg) {
    std::lock_guard<std::mutex> lock(g_fail_mutex);
    ++t.failures;
    if (g_printed < kMaxPrinted) say("  失败 [%s] %s\n", t.name.c_str(), msg.c_str());
    else if (g_printed == kMaxPrinted) say("  （其余失败不再逐条打印，只计数）\n");
    ++g_printed;
}
static void fail(const std::string& msg) { fail_in(g_cur ? *g_cur : g_t2, msg); }
static void check(bool ok, const std::string& what) {
    if (g_cur) ++g_cur->checks;
    if (!ok) fail(what);
}

// ---------------------------------------------------------------------------------------------
// 固定种子的随机数与测试矩阵（splitmix64，同 test_eig_qr.cpp：不同平台得到同样的矩阵）
struct Rng {
    std::uint64_t s;
    explicit Rng(std::uint64_t seed) : s(seed) {}
    std::uint64_t next() {
        std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    double uniform() { return static_cast<double>(next() >> 11) * (2.0 / 9007199254740992.0) - 1.0; }   // (-1, 1)
    cplx cuniform() {
        const double re = uniform();
        return cplx(re, uniform());
    }
};

static Matrix random_complex(int n, std::uint64_t seed) {
    Rng rng(seed);
    Matrix a(n, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) a(i, j) = rng.cuniform();
    return a;
}
static Matrix random_real(int n, std::uint64_t seed) {
    Rng rng(seed);
    Matrix a(n, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) a(i, j) = rng.uniform();
    return a;
}
static Matrix random_unitary(int n, std::uint64_t seed) {       // n 个随机 Householder 反射的乘积
    Rng rng(seed);
    Matrix q = Matrix::identity(n);
    Vector v(n), qv(n);
    for (int k = 0; k < n; ++k) {
        double vv = 0.0;
        for (int i = 0; i < n; ++i) {
            v[i] = rng.cuniform();
            vv += std::norm(v[i]);
        }
        qv = q * v;
        for (int j = 0; j < n; ++j) {
            const cplx f = 2.0 * std::conj(v[j]) / vv;
            for (int i = 0; i < n; ++i) q(i, j) -= qv[i] * f;
        }
    }
    return q;
}
static Matrix hermitian_part(const Matrix& m) {
    const int n = m.rows();
    Matrix a(n, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) a(i, j) = 0.5 * (m(i, j) + std::conj(m(j, i)));
    return a;
}
static Matrix hermitian_indefinite(int n, std::uint64_t seed) {
    Matrix a = hermitian_part(random_complex(n, seed));
    for (int i = 0; i < n; ++i) a(i, i) += -double(n) + 4.0 * i;
    return a;
}
// 块上三角：cuts 是各块的起点（首个为 0），对角块之下的整块清零。迭代会早早分裂，活动块的上沿 l > 0。
static Matrix block_upper(int n, const std::vector<int>& cuts, std::uint64_t seed) {
    Matrix a = random_complex(n, seed);
    for (std::size_t b = 1; b < cuts.size(); ++b)
        for (int j = 0; j < cuts[b]; ++j)
            for (int i = cuts[b]; i < n; ++i) a(i, j) = 0.0;
    return a;
}
static Matrix cyclic_permutation(int n) {                       // 酉矩阵，普通位移恒为 0，要靠例外位移启动
    Matrix a(n, n);
    for (int i = 1; i < n; ++i) a(i, i - 1) = 1.0;
    if (n > 0) a(0, n - 1) = 1.0;
    return a;
}
static Matrix companion(const Vector& roots) {
    const int n = static_cast<int>(roots.size());
    Vector c(n + 1, cplx{});
    c[0] = 1.0;
    for (int k = 0; k < n; ++k) {
        for (int i = k + 1; i >= 1; --i) c[i] = c[i - 1] - roots[k] * c[i];
        c[0] = -roots[k] * c[0];
    }
    Matrix a(n, n);
    for (int i = 1; i < n; ++i) a(i, i - 1) = 1.0;
    for (int i = 0; i < n; ++i) a(i, n - 1) = -c[i];
    return a;
}
static Matrix companion_ring(int n, std::uint64_t seed) {       // 根在单位圆附近
    Rng rng(seed);
    Vector roots(n);
    for (int k = 0; k < n; ++k) roots[k] = std::polar(0.7 + 0.3 * std::abs(rng.uniform()), 2.0 * kPi * (k + 0.3 * rng.uniform()) / n);
    return companion(roots);
}
// RCWA TE 层矩阵 diag(kx²) − Toeplitz(ε_h)，n = 2M+1
static Matrix rcwa_te_matrix(int n, cplx n_ridge, cplx n_groove, double fill, double lambda_over_period, double sin_theta) {
    const int M = n / 2;
    const cplx e1 = n_ridge * n_ridge, e2 = n_groove * n_groove;
    Matrix a(n, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            const int h = i - j;
            const cplx eps_h = h == 0 ? fill * e1 + (1.0 - fill) * e2 : (e1 - e2) * (std::sin(kPi * h * fill) / (kPi * h));
            a(i, j) = -eps_h;
        }
    for (int i = 0; i < n; ++i) {
        const double kx = sin_theta + (i - M) * lambda_over_period;
        a(i, i) += kx * kx;
    }
    return a;
}
// 均匀层的层矩阵：对角线 + 1e-17 量级的舍入误差，本征值成对（精确或几乎）重合（同 test_eig_qr.cpp）
static Matrix uniform_layer_matrix(int n, double index, double lambda_over_period) {
    const int M = n / 2;
    Matrix a(n, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            const int h = i - j;
            cplx eps_h = index * index;
            if (h != 0) {
                const double arg = -2.0 * kPi * h;
                const cplx t = (index * index) * (cplx{std::cos(arg), std::sin(arg)} - cplx{1.0, 0.0});
                const double w = -2.0 * kPi * h;
                eps_h = cplx{t.imag() / w, -t.real() / w};
            }
            a(i, j) = -eps_h;
        }
    for (int i = 0; i < n; ++i) {
        const double kx = (i - M) * lambda_over_period;
        a(i, i) += kx * kx;
    }
    return a;
}

// 二维光栅层的 P·Q（Li 的逆规则，非磁性，周期 1×1，矩形柱），N = (2M+1)² 个级次，返回 2N×2N。
//   P = [Kx E⁻¹ Ky,  I − Kx E⁻¹ Kx;  Ky E⁻¹ Ky − I,  −Ky E⁻¹ Kx]      Q = [Kx Ky,  E − Kx Kx;  Ky Ky − E,  −Ky Kx]
// variant 0：普通光栅；1：uni0——E = ε·I 精确，谱完全简并；2：uni——用 fx = fy = 1 的 sinc 公式，非对角元只剩 1e-17 量级的舍入误差。
// θ = 0、fx = fy 时有 C4 对称，本征值成四重（接近）简并，这是真实 P·Q 的特点。
static Matrix pq2d(int M, cplx n_in, cplx n_out, double fx, double fy, double lambda_over_period, double theta_deg, double phi_deg, int variant) {
    const int w = 2 * M + 1, N = w * w;
    const double th = theta_deg * kPi / 180.0, ph = phi_deg * kPi / 180.0;
    std::vector<int> pp(N), qq(N);
    Vector kx(N), ky(N);
    for (int q = 0; q < w; ++q)
        for (int p = 0; p < w; ++p) {
            const int a = p + w * q;
            pp[a] = p - M;
            qq[a] = q - M;
            kx[a] = std::sin(th) * std::cos(ph) + pp[a] * lambda_over_period;
            ky[a] = std::sin(th) * std::sin(ph) + qq[a] * lambda_over_period;
        }
    const cplx e_in = n_in * n_in, e_out = n_out * n_out;
    const auto coef = [](int h, double f) { return h == 0 ? f : std::sin(kPi * h * f) / (kPi * h); };
    Matrix E(N, N);
    for (int b = 0; b < N; ++b)
        for (int a = 0; a < N; ++a) {
            if (variant == 1) {
                E(a, b) = a == b ? e_in : cplx{};
                continue;
            }
            const int dp = pp[a] - pp[b], dq = qq[a] - qq[b];
            cplx v = (e_in - e_out) * (coef(dp, fx) * coef(dq, fy));
            if (dp == 0 && dq == 0) v += e_out;
            E(a, b) = v;
        }
    const Matrix Ei = inverse(E);
    const auto kek = [&](const Vector& l, const Vector& r) { return scale_rows(scale_columns(Ei, r), l); };
    const auto put = [](Matrix& big, int r0, int c0, const Matrix& blk) {
        for (int j = 0; j < blk.cols(); ++j)
            for (int i = 0; i < blk.rows(); ++i) big(r0 + i, c0 + j) = blk(i, j);
    };
    const Matrix I = Matrix::identity(N);
    Vector kxkx(N), kyky(N), kxky(N);
    for (int a = 0; a < N; ++a) {
        kxkx[a] = kx[a] * kx[a];
        kyky[a] = ky[a] * ky[a];
        kxky[a] = kx[a] * ky[a];
    }
    Matrix P(2 * N, 2 * N), Q(2 * N, 2 * N);
    put(P, 0, 0, kek(kx, ky));
    put(P, 0, N, I - kek(kx, kx));
    put(P, N, 0, kek(ky, ky) - I);
    put(P, N, N, -kek(ky, kx));
    put(Q, 0, 0, Matrix::diagonal(kxky));
    put(Q, 0, N, E - Matrix::diagonal(kxkx));
    put(Q, N, 0, Matrix::diagonal(kyky) - E);
    put(Q, N, N, -Matrix::diagonal(kxky));
    return P * Q;
}

// 前 k 列在对角线以下全为零（上三角），其余列稠密：化 Hessenberg 的前 k 步整列为零，走“已经是 Hessenberg 形”的分支
static Matrix tri_then_dense(int n, int k, std::uint64_t seed) {
    Matrix a = random_complex(n, seed);
    for (int j = 0; j < k && j < n; ++j)
        for (int i = j + 1; i < n; ++i) a(i, j) = 0.0;
    return a;
}
// 前 k 列在次对角线以下全为零（Hessenberg 形），其余列稠密：反射向量只有一个非零分量
static Matrix hess_then_dense(int n, int k, std::uint64_t seed) {
    Matrix a = random_complex(n, seed);
    for (int j = 0; j < k && j < n; ++j)
        for (int i = j + 2; i < n; ++i) a(i, j) = 0.0;
    return a;
}
// 三个对角块：稠密 k1 阶、上三角 k2 阶、稠密（其余），块下方全为零。化 Hessenberg 时中间块的各步整列为零（被跳过），
// 前后两块的步不跳过：测“跳过一步之后，上一步推迟的右乘怎样处理”。
static Matrix dense_tri_dense(int n, int k1, int k2, std::uint64_t seed) {
    Matrix a = random_complex(n, seed);
    for (int j = 0; j < k1; ++j)
        for (int i = k1; i < n; ++i) a(i, j) = 0.0;
    for (int j = k1; j < k1 + k2; ++j)
        for (int i = j + 1; i < n; ++i) a(i, j) = 0.0;
    return a;
}
// k 个稠密块交错排列（只有 i ≡ j mod k 的元素非零）：反射向量只落在一个剩余类的行上，零结构在整个约化过程里精确保持，
// 所以 x[m] 恰为零的列（f = 0 的分支）与整列为零的步（每个块的最后一列之后）都出现，而且跳过的步夹在反射向量支撑很广的步之间。
static Matrix interleaved_blocks(int n, int k, std::uint64_t seed) {
    Matrix a = random_complex(n, seed);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            if (i % k != j % k) a(i, j) = 0.0;
    return a;
}
static Matrix antidiagonal(int n, double noise, std::uint64_t seed) {   // 交换矩阵：第 m−1 列的反射向量里 x[m] 恰为零（f = 0 的分支）
    Rng rng(seed);
    Matrix a(n, n);
    for (int i = 0; i < n; ++i) a(i, n - 1 - i) = 1.0;
    if (noise != 0.0)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) a(i, j) += noise * rng.cuniform();
    return a;
}
static Matrix arrowhead(int n, std::uint64_t seed) {
    Rng rng(seed);
    Matrix a(n, n);
    for (int i = 0; i < n; ++i) {
        a(i, i) = cplx(i - 0.5 * n, 0.25 * (i % 7));
        a(0, i) += rng.cuniform();
        a(i, 0) += rng.cuniform();
    }
    return a;
}
static Matrix tridiagonal_symmetric(int n, std::uint64_t seed) {         // 复对称三对角
    Rng rng(seed);
    Matrix a(n, n);
    for (int i = 0; i < n; ++i) {
        a(i, i) = cplx(2.0 * rng.uniform(), 0.1 * rng.uniform());
        if (i + 1 < n) a(i, i + 1) = a(i + 1, i) = rng.cuniform();
    }
    return a;
}
static Matrix triangular(int n, bool upper, std::uint64_t seed) {
    Matrix a = random_complex(n, seed);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            if (upper ? i > j : i < j) a(i, j) = 0.0;
    for (int j = 0; j < n; ++j) a(j, j) = cplx(j - 0.5 * n, 0.25 * j);   // 互不相同的对角元
    return a;
}
static Matrix rank_one(int n, std::uint64_t seed) {
    Rng rng(seed);
    Vector u(n), v(n);
    for (int i = 0; i < n; ++i) {
        u[i] = rng.cuniform();
        v[i] = rng.cuniform();
    }
    Matrix a(n, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) a(i, j) = u[i] * std::conj(v[j]);
    return a;
}
static Matrix nilpotent_shift(int n) {
    Matrix a(n, n);
    for (int i = 1; i < n; ++i) a(i, i - 1) = 1.0;
    return a;
}
static Matrix jordan_block(int n, cplx lambda) {
    Matrix a(n, n);
    for (int i = 0; i < n; ++i) {
        a(i, i) = lambda;
        if (i + 1 < n) a(i, i + 1) = 1.0;
    }
    return a;
}
static Matrix scaled_by_pow2(const Matrix& a, int k) {
    Matrix b(a.rows(), a.cols());
    for (int j = 0; j < a.cols(); ++j)
        for (int i = 0; i < a.rows(); ++i) b(i, j) = cplx(std::ldexp(a(i, j).real(), k), std::ldexp(a(i, j).imag(), k));
    return b;
}
static Matrix scaled_by(const Matrix& a, double f) { return a * cplx(f, 0.0); }
static Matrix direct_sum(const std::vector<Matrix>& blocks) {
    int n = 0;
    for (const Matrix& b : blocks) n += b.rows();
    Matrix a(n, n);
    int o = 0;
    for (const Matrix& b : blocks) {
        for (int j = 0; j < b.cols(); ++j)
            for (int i = 0; i < b.rows(); ++i) a(o + i, o + j) = b(i, j);
        o += b.rows();
    }
    return a;
}
static Matrix repeat_block(const Matrix& b, int k) { return direct_sum(std::vector<Matrix>(k, b)); }
// Q diag(d) Q^H（Q 随机酉）
static Matrix unitary_similar(const Matrix& d, std::uint64_t seed) {
    const Matrix q = random_unitary(d.rows(), seed);
    return q * d * adjoint(q);
}
static Matrix read_dump(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::int32_t hdr[2];
    f.read(reinterpret_cast<char*>(hdr), sizeof hdr);
    if (!f || hdr[0] < 0 || hdr[1] < 0) throw std::runtime_error("bad header in " + path);
    Matrix m(hdr[0], hdr[1]);
    f.read(reinterpret_cast<char*>(m.data()), static_cast<std::streamsize>(sizeof(cplx)) * hdr[0] * hdr[1]);
    if (!f) throw std::runtime_error("short read in " + path);
    return m;
}

// ---------------------------------------------------------------------------------------------
// 逐字节比较
static const std::size_t kSame = static_cast<std::size_t>(-1);
// 只有“元素全是 DBL_MAX、结果是一片 NaN”的那一族把两个 NaN 当作相同：NaN 的符号位与载荷取决于编译器怎样排乘加的操作数、
// 以及一个元素落在向量化循环体还是标量尾部，微小的、与数值无关的改动（甚至另一处代码的内联）都会改变它，没有要求逐位相同的意义。
// 其余各族里出现 NaN 一律算不同。
static bool g_nan_loose = false;
static std::size_t first_diff_byte(const void* a, const void* b, std::size_t n) {
    if (n == 0 || std::memcmp(a, b, n) == 0) return kSame;
    const unsigned char* p = static_cast<const unsigned char*>(a);
    const unsigned char* q = static_cast<const unsigned char*>(b);
    if (g_nan_loose && n % sizeof(double) == 0) {
        for (std::size_t k = 0; k < n; k += sizeof(double)) {
            if (std::memcmp(p + k, q + k, sizeof(double)) == 0) continue;
            double x, y;
            std::memcpy(&x, p + k, sizeof x);
            std::memcpy(&y, q + k, sizeof y);
            if (std::isnan(x) && std::isnan(y)) continue;
            for (std::size_t j = 0; j < sizeof(double); ++j)
                if (p[k + j] != q[k + j]) return k + j;
        }
        return kSame;
    }
    for (std::size_t k = 0; k < n; ++k)
        if (p[k] != q[k]) return k;
    return kSame;
}
static unsigned long long bits_of(double x) {
    std::uint64_t u;
    std::memcpy(&u, &x, sizeof u);
    return u;
}
// 两段复数数组第一个不同字节的说明。rows > 0 时按矩阵 (行,列) 报位置，否则按下标。
static std::string diff_text(const char* what, const cplx* x, const cplx* y, std::size_t byte, int rows, const char* la, const char* lb) {
    const std::size_t e = byte / sizeof(cplx), within = byte % sizeof(cplx);
    const double dx = within < 8 ? x[e].real() : x[e].imag(), dy = within < 8 ? y[e].real() : y[e].imag();
    const std::string pos = rows > 0 ? fmt("(%d,%d)", static_cast<int>(e % rows), static_cast<int>(e / rows)) : fmt("[%d]", static_cast<int>(e));
    return fmt("%s%s 的第 %d 字节（%s）不同：%s = %.17g (0x%016llx)，%s = %.17g (0x%016llx)", what, pos.c_str(), static_cast<int>(within),
               within < 8 ? "实部" : "虚部", la, dx, bits_of(dx), lb, dy, bits_of(dy));
}
static std::string cmp_vec(const char* what, const Vector& x, const Vector& y, const char* la, const char* lb, long long& bytes) {
    if (x.size() != y.size()) return fmt("%s 的长度不同：%s = %d，%s = %d", what, la, static_cast<int>(x.size()), lb, static_cast<int>(y.size()));
    const std::size_t n = x.size() * sizeof(cplx);
    bytes += static_cast<long long>(n);
    const std::size_t d = first_diff_byte(x.data(), y.data(), n);
    return d == kSame ? std::string() : diff_text(what, x.data(), y.data(), d, 0, la, lb);
}
static std::string cmp_mat(const char* what, const Matrix& x, const Matrix& y, const char* la, const char* lb, long long& bytes) {
    if (x.rows() != y.rows() || x.cols() != y.cols())
        return fmt("%s 的尺寸不同：%s = %d×%d，%s = %d×%d", what, la, x.rows(), x.cols(), lb, y.rows(), y.cols());
    const std::size_t n = static_cast<std::size_t>(x.rows()) * x.cols() * sizeof(cplx);
    bytes += static_cast<long long>(n);
    const std::size_t d = first_diff_byte(x.data(), y.data(), n);
    return d == kSame ? std::string() : diff_text(what, x.data(), y.data(), d, x.rows(), la, lb);
}
// values、vectors、Z、T、sweeps、info
static std::string cmp_result(const EigResult& x, const EigResult& y, const char* la, const char* lb, long long& bytes) {
    bytes += 2 * static_cast<long long>(sizeof(int));
    if (x.info != y.info) return fmt("info 不同：%s = %d，%s = %d", la, x.info, lb, y.info);
    if (x.sweeps != y.sweeps) return fmt("sweeps 不同：%s = %d，%s = %d", la, x.sweeps, lb, y.sweeps);
    std::string s = cmp_vec("values", x.values, y.values, la, lb, bytes);
    if (s.empty()) s = cmp_mat("vectors", x.vectors, y.vectors, la, lb, bytes);
    if (s.empty()) s = cmp_mat("Z", x.Z, y.Z, la, lb, bytes);
    if (s.empty()) s = cmp_mat("T", x.T, y.T, la, lb, bytes);
    return s;
}
static void hash_bytes(const void* p, std::size_t n, std::uint64_t& h1, std::uint64_t& h2) {
    const unsigned char* b = static_cast<const unsigned char*>(p);
    std::uint64_t a0 = 0x9e3779b97f4a7c15ULL ^ n, a1 = 0xc2b2ae3d27d4eb4fULL + n, a2 = 0x165667b19e3779f9ULL, a3 = 0x27d4eb2f165667c5ULL;
    std::size_t k = 0;
    for (; k + 32 <= n; k += 32) {
        std::uint64_t w[4];
        std::memcpy(w, b + k, 32);
        a0 = (a0 ^ w[0]) * 0xff51afd7ed558ccdULL;
        a0 ^= a0 >> 32;
        a1 = (a1 ^ w[1]) * 0xc4ceb9fe1a85ec53ULL;
        a1 ^= a1 >> 29;
        a2 = (a2 ^ w[2]) * 0x9e3779b185ebca87ULL;
        a2 ^= a2 >> 31;
        a3 = (a3 ^ w[3]) * 0xc2b2ae3d27d4eb4fULL;
        a3 ^= a3 >> 33;
    }
    for (; k < n; ++k) a0 = (a0 ^ b[k]) * 0x100000001b3ULL;
    h1 = (a0 * 0x9e3779b97f4a7c15ULL) ^ (a2 + (a1 << 17));
    h2 = (a1 * 0xbf58476d1ce4e5b9ULL) ^ (a3 + (a0 >> 7)) ^ (a2 << 11);
}

// ---------------------------------------------------------------------------------------------
// 选项
struct OptSpec {
    const char* name;
    bool vectors, keep, observer;
    double eps;
    int max_iter;
};
enum OptId { kDefault, kKeep, kKeepNoVec, kNoVec, kIter1, kIter2, kIter3NoVec, kObserver, kObserverNoVec, kEps0, kEps8, kOptCount };
static const OptSpec kOptions[kOptCount] = {
    {"default", true, false, false, kEps, 30},
    {"keep", true, true, false, kEps, 30},
    {"keep,novec", false, true, false, kEps, 30},
    {"novec", false, false, false, kEps, 30},
    {"iter1", true, true, false, kEps, 1},
    {"iter2", true, false, false, kEps, 2},
    {"iter3,novec", false, true, false, kEps, 3},
    {"observer", true, false, true, kEps, 30},
    {"observer,novec", false, false, true, kEps, 30},
    {"eps0", true, true, false, 0.0, 30},
    {"eps1e-8", true, true, false, 1e-8, 30},
};
static std::vector<OptId> option_set(int tier) {         // 0 全部；1 缩减；2 最少
    if (tier == 0) return {kDefault, kKeep, kKeepNoVec, kNoVec, kIter1, kIter2, kIter3NoVec, kObserver, kObserverNoVec, kEps0, kEps8};
    if (tier == 1) return {kDefault, kKeepNoVec, kNoVec, kIter2, kObserver, kEps8};
    return {kDefault, kKeepNoVec, kIter2};
}
static EigOptions make_options(const OptSpec& s, long long* calls) {
    EigOptions o;
    o.vectors = s.vectors;
    o.keep_transform = s.keep;
    o.eps = s.eps;
    o.max_iter = s.max_iter;
    if (s.observer) o.observer = [calls](const EigProgress&) { ++*calls; };
    return o;
}
// 工作量记账：每次调用记 n³（只求本征值的按 0.65 计），折成“2N = 1250 的完整分解”的个数，用来在负载很高的机器上估计空闲时的用时
static std::atomic<long long> g_work_ref{0}, g_work_new{0};
static void account(std::atomic<long long>& w, const Matrix& a, const EigOptions& o) {
    const double n = a.rows();
    w += static_cast<long long>(n * n * n * ((o.vectors || o.keep_transform || o.observer) ? 1.0 : 0.65));
}
static EigResult run_ref(const Matrix& a, const EigOptions& o) {
    account(g_work_ref, a, o);
    return ref_v1::eig_qr(a, o);
}
static EigResult run_new(const Matrix& a, const EigOptions& o) {
    account(g_work_new, a, o);
    return stablercwa::eig_qr(a, o);
}

// ---------------------------------------------------------------------------------------------
// 矩阵目录
struct MatCase {
    std::string name;     // 族 + 参数
    std::string family;
    int n = 0;
    Matrix a;
    int tier = 1;         // 选项集合：0 全部、1 缩减、2 最少
    bool primary = false; // 主要的并行区矩阵：所有线程数都跑
};

// 小矩阵（运行便宜）与“主要”矩阵用全部线程数；其余抽稀成 {3, 8}（奇数线程、2 的幂各一个；n <= 130 时加上 1）
static std::vector<int> threads_for(const MatCase& mc) {
    if (mc.primary || mc.n <= 70) return g_cfg.threads;
    std::vector<int> out;
    std::size_t h = 0;
    for (char ch : mc.name) h = h * 131 + static_cast<unsigned char>(ch);
    static const int kExtra[3] = {2, 4, 7};
    const int extra = kExtra[h % 3];                         // 2、4、7 轮流：每个矩阵多一个线程数，代价 +50%
    for (int t : g_cfg.threads)
        if (t == 3 || t == 8 || t == extra) out.push_back(t);
    if (mc.n <= 130) out.insert(out.begin(), 1);
    return out.empty() ? g_cfg.threads : out;
}

static void add(std::vector<MatCase>& v, const std::string& family, const std::string& tag, Matrix a, int tier, bool primary = false) {
    MatCase mc;
    mc.family = family;
    mc.n = a.rows();
    mc.name = family + (tag.empty() ? "" : " " + tag) + " n=" + std::to_string(mc.n);
    mc.a = std::move(a);
    mc.tier = tier;
    mc.primary = primary;
    v.push_back(std::move(mc));
}

static std::vector<MatCase> catalogue() {
    const int nm = g_cfg.nmin;
    const bool big = g_cfg.big;
    std::vector<MatCase> v;
    const std::uint64_t seed = 1000;

    // 选项集合（tier）：0 全部 11 种；1 缩减 6 种；2 只有 default、keep+novec、iter2。n <= 70 的矩阵便宜，选项与线程数都用全的；
    // 更大的抽稀（threads_for），阈值两侧与阈值以上的主要矩阵（primary）用全部线程数。完整覆盖靠 T5–T10 另外安排的矩阵与 --big。
    // 随机复矩阵：小尺寸全部、阈值两侧、阈值以上
    for (int n : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 63, 64, 65}) add(v, "rand_c", "", random_complex(n, seed + n), 0);
    for (int n : {129, 150}) add(v, "rand_c", "", random_complex(n, seed + n), 2);
    add(v, "rand_c", "", random_complex(nm - 1, seed + nm - 1), 2);
    add(v, "rand_c", "", random_complex(nm, seed + nm), 1, true);
    add(v, "rand_c", "", random_complex(nm + 1, seed + nm + 1), 2);
    // 并行化的按工作量阈值：化 Hessenberg 每步 4e5（n >= 317 才有并行步）、S 区 6e4（窗口 >= 346）。n <= 257 的矩阵永远进不了
    // 这两种并行区（实测：原型在默认运行里 hess_PAR = 0、S_region_PAR = 0），所以默认运行至少要有一个 n = 400 的稠密矩阵和一个带窗口扩张的
    if (!big) add(v, "rand_c", "", random_complex(400, seed + 400), 2);
    add(v, "block", "3块 400", block_upper(400, {0, 150, 300}, 4007), 2);
    if (big)
        for (int n : {177, 200, 257, 300, 333, 400, 512}) add(v, "rand_c", "", random_complex(n, seed + n), 2);

    // 随机实矩阵：次对角元已经是实数，make_subdiagonal_real 整个跳过
    for (int n : {2, 3, 5, 9, 33, 64}) add(v, "rand_r", "", random_real(n, 2000 + n), 0);
    add(v, "rand_r", "", random_real(100, 2100), 2);
    add(v, "rand_r", "", random_real(nm + 1, 2000 + nm + 1), 2);
    if (big)
        for (int n : {240, 301}) add(v, "rand_r", "", random_real(n, 2000 + n), 2);

    // 厄米（不定）
    for (int n : {6, 24, 64}) add(v, "herm", "", hermitian_indefinite(n, 3000 + n), 0);
    add(v, "herm", "", hermitian_indefinite(100, 3100), 2);
    add(v, "herm", "", hermitian_indefinite(nm, 3000 + nm), 2);
    if (big)
        for (int n : {150, 250}) add(v, "herm", "", hermitian_indefinite(n, 3000 + n), 2);

    // 块上三角：l > 0 早出现，活动块的上沿不是 0；三块、四块的窗口向上扩张的机会更多
    add(v, "block", "2块 12/5", block_upper(12, {0, 5}, 4001), 0);
    add(v, "block", "2块 40/17", block_upper(40, {0, 17}, 4002), 0);
    add(v, "block", "3块 100", block_upper(100, {0, 30, 70}, 4003), 2);
    if (big) add(v, "block", "2块", block_upper(nm + 1, {0, nm / 2}, 4004), 2);
    add(v, "block", "3块", block_upper(nm + 8, {0, 50, 120}, 4005), 2);
    if (big) add(v, "block", "4块", block_upper(280, {0, 50, 130, 200}, 4006), 2);

    // 循环置换：普通位移恒为 0，第 10、20 次迭代的例外位移才使它启动
    for (int n : {3, 8, 16, 64}) add(v, "cyclic", "", cyclic_permutation(n), 0);
    add(v, "cyclic", "", cyclic_permutation(nm), 2);
    if (big) add(v, "cyclic", "", cyclic_permutation(nm + 8), 2);

    // 友矩阵
    add(v, "companion", "根 1..6", companion(Vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}), 0);
    for (int n : {12, 40}) add(v, "companion", "圆环", companion_ring(n, 5000 + n), 0);
    add(v, "companion", "圆环", companion_ring(100, 5100), 2);
    if (big) add(v, "companion", "圆环", companion_ring(nm + 1, 5101), 2);

    // 化 Hessenberg 里的特殊分支（整列为零、f = 0、反射向量只有一个分量）
    for (int n : {20, 64}) {
        add(v, "tri_dense", "", tri_then_dense(n, n / 3, 6000 + n), n == 20 ? 0 : 2);
        add(v, "hess_dense", "", hess_then_dense(n, n / 2, 6100 + n), n == 20 ? 0 : 2);
        add(v, "antidiag", "", antidiagonal(n, 0.0, 0), n == 20 ? 0 : 2);
        add(v, "antidiag", "+噪声", antidiagonal(n, 1e-3, 6200 + n), 0);
    }
    add(v, "dense_tri_dense", "", dense_tri_dense(20, 6, 7, 6601), 0);
    add(v, "dense_tri_dense", "", dense_tri_dense(64, 20, 24, 6602), 0);
    add(v, "dense_tri_dense", "", dense_tri_dense(nm + 1, 60, 70, 6603), 2);
    add(v, "interleaved", "k=4", interleaved_blocks(20, 4, 6611), 0);
    add(v, "interleaved", "k=3", interleaved_blocks(64, 3, 6612), 0);
    add(v, "interleaved", "k=4", interleaved_blocks(nm + 1, 4, 6613), 2);
    add(v, "arrowhead", "", arrowhead(40, 6301), 0);
    add(v, "tridiag", "", tridiagonal_symmetric(100, 6401), 2);
    add(v, "tri_dense", "", tri_then_dense(nm + 1, (nm + 1) / 3, 6501), 2);
    add(v, "hess_dense", "", hess_then_dense(nm, nm / 2, 6502), 2);
    if (big) {
        add(v, "antidiag", "+噪声", antidiagonal(nm + 1, 1e-3, 6504), 2);
        add(v, "arrowhead", "", arrowhead(nm + 1, 6302), 2);
    }
    add(v, "tridiag", "", tridiagonal_symmetric(nm + 8, 6402), 2);
    if (big) add(v, "tri_dense", "", tri_then_dense(257, 80, 6503), 2);

    // 平凡与退化
    for (int n : {1, 2, 10, 64}) add(v, "zero", "", Matrix(n, n), 0);
    add(v, "identity", "", Matrix::identity(5), 0);
    add(v, "identity", "", Matrix::identity(nm + 1), 2);
    add(v, "zero", "", Matrix(nm + 1, nm + 1), 2);
    add(v, "upper_tri", "", triangular(30, true, 7030), 0);
    add(v, "lower_tri", "", triangular(30, false, 7130), 0);
    if (big) add(v, "lower_tri", "", triangular(nm + 1, false, 7201), 2);
    add(v, "upper_tri", "", triangular(nm + 8, true, 7200), 2);
    add(v, "rank1", "", rank_one(50, 7301), 0);
    add(v, "rank1", "", rank_one(nm + 8, 7302), 2);
    add(v, "nilpotent", "", nilpotent_shift(100), 2);

    // 一维层矩阵（TE，含吸收与金属）
    add(v, "layer1d", "无吸收", rcwa_te_matrix(41, 1.5, 1.0, 0.4, 0.61, 0.2), 0);
    add(v, "layer1d", "吸收", rcwa_te_matrix(41, cplx(1.8, 0.13), 1.0, 0.4, 0.61, 0.2), 0);
    add(v, "layer1d", "金属", rcwa_te_matrix(41, cplx(0.08, 5.43), 1.0, 0.4, 0.61, 0.2), 0);
    add(v, "layer1d", "吸收", rcwa_te_matrix(101, cplx(1.8, 0.13), 1.0, 0.4, 0.61, 0.2), 2);
    add(v, "layer1d", "金属", rcwa_te_matrix(101, cplx(0.08, 5.43), 1.0, 0.4, 0.61, 0.2), 2);
    if (big) {                                                          // 默认运行里一维层矩阵由 T3 的 rcwa::solve 覆盖
        add(v, "layer1d", "吸收", rcwa_te_matrix(nm + 9, cplx(1.8, 0.13), 1.0, 0.4, 0.61, 0.2), 2);
        add(v, "layer1d", "无吸收 近Littrow", rcwa_te_matrix(201, 1.5, 1.0, 0.4, 0.61, 0.3), 2);
        add(v, "layer1d", "金属", rcwa_te_matrix(257, cplx(0.08, 5.43), 1.0, 0.4, 0.61, 0.2), 2);
        add(v, "layer1d", "吸收", rcwa_te_matrix(321, cplx(1.8, 0.13), 1.0, 0.4, 0.61, 0.2), 2);
    }

    // 二维 P·Q（2N = 18, 50, 98, 162, 242, 338 …）：真实结构的合成版（Li 逆规则），C4 对称、吸收、斜入射、均匀（uni0 / uni）
    add(v, "pq2d", "sq M=1", pq2d(1, 3.5, 1.0, 0.5, 0.5, 0.61, 0, 0, 0), 0);
    add(v, "pq2d", "sq M=2", pq2d(2, 3.5, 1.0, 0.5, 0.5, 0.61, 0, 0, 0), 0);
    add(v, "pq2d", "abs M=2", pq2d(2, cplx(3.5, 0.2), 1.0, 0.5, 0.5, 0.61, 0, 0, 0), 0);
    add(v, "pq2d", "off M=3", pq2d(3, 3.5, 1.0, 0.4, 0.6, 0.61, 25.0, 30.0, 0), 2);
    add(v, "pq2d", "uni0 M=3", pq2d(3, 1.7, 1.7, 1.0, 1.0, 0.61, 0, 0, 1), 2);
    add(v, "pq2d", "uni M=3", pq2d(3, 1.7, 1.7, 1.0, 1.0, 0.61, 0, 0, 2), 2);
    add(v, "pq2d", "sq M=4", pq2d(4, 3.5, 1.0, 0.5, 0.5, 0.61, 0, 0, 0), 2);
    add(v, "pq2d", "sq M=5", pq2d(5, 3.5, 1.0, 0.5, 0.5, 0.61, 0, 0, 0), 2);
    if (big) {
        add(v, "pq2d", "off M=5", pq2d(5, cplx(3.5, 0.2), 1.0, 0.4, 0.6, 0.61, 25.0, 30.0, 0), 2);
        add(v, "pq2d", "uni M=5", pq2d(5, 1.7, 1.7, 1.0, 1.0, 0.61, 0, 0, 2), 2);
        add(v, "pq2d", "sq M=6", pq2d(6, 3.5, 1.0, 0.5, 0.5, 0.61, 0, 0, 0), 2);
        add(v, "pq2d", "sq M=8", pq2d(8, 3.5, 1.0, 0.5, 0.5, 0.61, 0, 0, 0), 2);
        add(v, "pq2d", "abs M=10", pq2d(10, cplx(3.5, 0.2), 1.0, 0.5, 0.5, 0.61, 0, 0, 0), 2);
        add(v, "pq2d", "sq M=12", pq2d(12, 3.5, 1.0, 0.5, 0.5, 0.61, 0, 0, 0), 2);
    }

    // 真实二维 P·Q 矩阵的转储（可选）
    if (!g_cfg.dump.empty()) {
        std::vector<std::string> names{"sq_M04", "sq_M06", "c4_M08", "off_M08", "abs_M08", "uni_M08", "uni0_M08", "sq_M08"};
        if (g_cfg.big) names.insert(names.end(), {"sq_M10", "sq_M12", "c4_M12", "abs_M12", "off_M12"});
        for (const std::string& name : names) {
            try {
                MatCase mc;
                mc.a = read_dump(g_cfg.dump + "/" + name + ".A.bin");
                mc.n = mc.a.rows();
                mc.family = "dump";
                mc.name = "dump " + name + " n=" + std::to_string(mc.n);
                mc.tier = 2;
                v.push_back(std::move(mc));
            } catch (const std::exception& e) {
                say("  （跳过转储 %s：%s）\n", name.c_str(), e.what());
            }
        }
    }
    return v;
}

// ---------------------------------------------------------------------------------------------
// 套件：一批矩阵 × 一组选项 × 一组线程数。每个 (矩阵, 选项) 先跑一次旧版，再在每个线程数下跑新版，
// 与旧版比较（T1 类）；各线程数的结果之间也互相比（T2）。
static std::string case_context(const std::string& name, const char* opt, int threads) {
    return fmt("%s 选项=%s 线程=%d：", name.c_str(), opt, threads);
}

static double now_seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void run_case(const MatCase& mc, const std::vector<OptId>& opts, const std::vector<int>& threads, Tally& t) {
    if (!g_cfg.filter.empty() && mc.name.find(g_cfg.filter) == std::string::npos) return;
    const double t_case = now_seconds();
    g_nan_loose = mc.family == "dbl_max";
    ++t.matrices;
    const Matrix input_copy = mc.a;
    for (OptId id : opts) {
        const OptSpec& os = kOptions[id];
        long long ref_calls = 0;
        EigResult ref;
        try {
            ref = run_ref(mc.a, make_options(os, &ref_calls));
        } catch (const std::exception& e) {
            fail_in(t, case_context(mc.name, os.name, 0) + "旧版意外抛出异常：" + e.what());
            continue;
        }
        ++t.runs;
        EigResult base;
        long long base_calls = 0;
        int base_threads = -1;
        for (int th : threads) {
            set_num_threads(th);
            long long calls = 0;
            EigResult got;
            try {
                got = run_new(mc.a, make_options(os, &calls));
            } catch (const std::exception& e) {
                fail_in(t, case_context(mc.name, os.name, th) + "新版意外抛出异常：" + e.what());
                continue;
            }
            ++t.runs;
            ++t.comparisons;
            std::string why = cmp_result(ref, got, "旧版", "新版", t.bytes);
            if (why.empty() && calls != ref_calls) why = fmt("观察者回调次数不同：旧版 %lld，新版 %lld", ref_calls, calls);
            if (!why.empty()) fail_in(t, case_context(mc.name, os.name, th) + why);
            if (base_threads < 0) {
                base = std::move(got);
                base_calls = calls;
                base_threads = th;
            } else {
                ++g_t2.comparisons;
                std::string why2 = cmp_result(base, got, fmt("线程%d", base_threads).c_str(), fmt("线程%d", th).c_str(), g_t2.bytes);
                if (why2.empty() && calls != base_calls) why2 = fmt("观察者回调次数不同：线程%d %lld，线程%d %lld", base_threads, base_calls, th, calls);
                if (!why2.empty()) fail_in(g_t2, case_context(mc.name, os.name, th) + why2);
            }
        }
    }
    set_num_threads(1);
    g_nan_loose = false;
    ++t.comparisons;
    const std::string why = cmp_mat("输入矩阵", input_copy, mc.a, "调用前", "调用后", t.bytes);
    if (!why.empty()) fail_in(t, mc.name + "：eig_qr 改动了输入矩阵：" + why);
    if (g_cfg.times) say("    %-40s %6.2f s（选项 %d 个，线程 %d 个）\n", mc.name.c_str(), now_seconds() - t_case, static_cast<int>(opts.size()), static_cast<int>(threads.size()));
}

static double g_cpu_start = 0;
static double cpu_seconds() {
#if defined(_WIN32)
    unsigned long long c = 0, e = 0, k = 0, u = 0;
    if (GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u)) return static_cast<double>(k + u) * 1e-7;
#endif
    return static_cast<double>(std::clock()) / CLOCKS_PER_SEC;
}
static void begin(Tally& t, const char* title) {
    g_stage_start = now_seconds();
    g_cpu_start = cpu_seconds();
    g_cur = &t;
    g_printed = 0;
    g_stage = title;
    say("[%s]\n", title);
}
static void finish(Tally& t, double t0, const char* extra = "") {
    t.seconds = now_seconds() - t0;
    t.cpu = cpu_seconds() - g_cpu_start;
    say("  矩阵/输入 %lld 个，新版运行 %lld 次，逐位比较 %lld 次（%.1f MB），其它断言 %lld 条，失败 %lld，用时 %.1f s（CPU %.1f s）%s\n", t.matrices, t.runs, t.comparisons,
        static_cast<double>(t.bytes) / 1e6, t.checks, t.failures, t.seconds, t.cpu, extra);
    g_cur = nullptr;
}

// ---------------------------------------------------------------------------------------------
// 一个套件的入口：对目录里满足 pick 的每个矩阵运行
static std::vector<MatCase> g_catalogue;

static void test_t1(Tally& t) {
    begin(t, "T1 逐位对照（新 eig_qr = 旧版；T2：各线程数之间也相同）");
    const double t0 = now_seconds();
    for (const MatCase& mc : g_catalogue) {
        std::vector<OptId> opts = option_set(mc.tier);
        if (mc.n > 130)                                             // eps = 0 要迭代到次对角元下溢，大矩阵上太慢
            opts.erase(std::remove(opts.begin(), opts.end(), kEps0), opts.end());
        run_case(mc, opts, threads_for(mc), t);
    }
    // test_eig_qr.cpp 里的 400 个小整数矩阵（n <= 9，到处是精确的零、重本征值、亏损）
    static const cplx alphabet[] = {0.0, 0.0, 0.0, 1.0, -1.0, cplx(0.0, 1.0), cplx(0.0, -1.0), 2.0, cplx(1.0, 1.0), 0.5};
    long long count = 0;
    for (int c = 0; c < 400; ++c) {
        Rng rng(77000 + c);
        const int n = 1 + static_cast<int>(rng.next() % 9);
        const int flavour = static_cast<int>(rng.next() % 5);
        const int letters = 3 + static_cast<int>(rng.next() % 8);
        Matrix a(n, n);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                cplx z = alphabet[rng.next() % letters];
                if (flavour == 1 && i > j + 1) z = 0.0;
                if (flavour == 2) z = z.real();
                a(i, j) = z;
            }
        if (flavour == 3)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < j; ++i) a(j, i) = a(i, j);
        if (flavour == 4)
            for (int j = 0; j < n; ++j) {
                a(j, j) = a(j, j).real();
                for (int i = 0; i < j; ++i) a(j, i) = std::conj(a(i, j));
            }
        MatCase mc;
        mc.family = "int400";
        mc.n = n;
        mc.name = "int400 #" + std::to_string(c) + " n=" + std::to_string(n);
        mc.a = std::move(a);
        run_case(mc, {kKeep, kKeepNoVec, c % 2 ? kIter1 : kIter2, kObserver}, {1, 8}, t);
        ++count;
    }
    finish(t, t0, fmt("（其中小整数矩阵 %lld 个）", count).c_str());
}

// ---------------------------------------------------------------------------------------------
// T3 嵌套
static std::string rcwa_result_diff(const rcwa::Result& x, const rcwa::Result& y, long long& bytes) {
    std::string s = cmp_vec("R", x.R, y.R, "a", "b", bytes);
    if (s.empty()) s = cmp_vec("T", x.T, y.T, "a", "b", bytes);
    if (s.empty() && x.DEr.size() != y.DEr.size()) s = "DEr 长度不同";
    if (s.empty() && first_diff_byte(x.DEr.data(), y.DEr.data(), x.DEr.size() * sizeof(double)) != kSame) s = "DEr 不同";
    if (s.empty() && x.DEt.size() != y.DEt.size()) s = "DEt 长度不同";
    if (s.empty() && first_diff_byte(x.DEt.data(), y.DEt.data(), x.DEt.size() * sizeof(double)) != kSame) s = "DEt 不同";
    bytes += static_cast<long long>((x.DEr.size() + x.DEt.size()) * sizeof(double));
    return s;
}
static rcwa::Stack grating_stack() {
    rcwa::Stack s;
    s.nI = 1.0;
    s.nII = 1.45;
    s.period = 1.0;
    s.layers.push_back(rcwa::Layer::grating(0.30, {0.0, 0.35, 1.0}, {cplx(3.2, 0.05), cplx(1.0, 0.0)}));
    s.layers.push_back(rcwa::Layer::uniform(0.12, cplx(1.9, 0.0)));
    s.layers.push_back(rcwa::Layer::grating(0.25, {0.0, 0.5, 1.0}, {cplx(2.1, 0.0), cplx(1.5, 0.0)}));
    return s;
}

static void test_t3(Tally& t) {
    begin(t, "T3 嵌套（parallel_for 里、观察者里、rcwa 的 custom_eig 里）");
    const double t0 = now_seconds();
    const int nm = g_cfg.nmin;
    // (a) parallel_for / parallel_ranges 的循环体里调用：并行区内部，必须自动串行且与旧版逐位相同
    std::vector<Matrix> mats{random_complex(nm + 1, 31), random_complex(200, 32), random_complex(100, 34),
                             random_complex(30, 35), hermitian_indefinite(nm, 36), block_upper(nm + 8, {0, 90}, 37)};
    const OptId kinds[3] = {kKeep, kDefault, kKeepNoVec};
    std::vector<EigResult> refs(12);
    for (int i = 0; i < 12; ++i) refs[i] = run_ref(mats[i % mats.size()], make_options(kOptions[kinds[i % 3]], nullptr));
    t.matrices += static_cast<long long>(mats.size());
    for (int th : {4, 8}) {
        set_num_threads(th);
        std::vector<EigResult> out(12);
        if (th == 4)
            parallel_for(0, 12, [&](int i) { out[i] = run_new(mats[i % mats.size()], make_options(kOptions[kinds[i % 3]], nullptr)); });
        else
            parallel_ranges(0, 12, [&](int lo, int hi) {
                for (int i = lo; i < hi; ++i) out[i] = run_new(mats[i % mats.size()], make_options(kOptions[kinds[i % 3]], nullptr));
            }, 2);
        t.runs += 12;
        for (int i = 0; i < 12; ++i) {
            ++t.comparisons;
            const std::string why = cmp_result(refs[i], out[i], "旧版", "新版", t.bytes);
            if (!why.empty()) fail_in(t, fmt("%s 内 线程=%d 第 %d 个任务：", th == 4 ? "parallel_for" : "parallel_ranges", th, i) + why);
        }
        // 嵌套之后线程池仍然可用，且并行版（若有）照常工作
        ++t.comparisons;
        const EigResult after = run_new(mats[0], make_options(kOptions[kKeep], nullptr));
        const std::string why = cmp_result(refs[0], after, "旧版", "新版", t.bytes);
        if (!why.empty()) fail_in(t, fmt("嵌套调用之后 线程=%d：", th) + why);
        ++t.runs;
    }
    if (g_cfg.times) say("    T3(a) %.2f s\n", now_seconds() - t0);
    // (b) 观察者回调里再调用 eig_qr：回调在调用线程上、并行区之外，内层调用可以用线程池（也可以串行）；不能死锁，结果不变
    {
        const Matrix outer = random_complex(nm + 1, 41), inner = random_complex(nm + 8, 42);
        const EigResult inner_ref = run_ref(inner, make_options(kOptions[kKeep], nullptr)), outer_ref = run_ref(outer, make_options(kOptions[kKeep], nullptr));
        t.matrices += 2;
        for (int th : {1, 4, 8}) {
            set_num_threads(th);
            int inner_runs = 0;
            std::string inner_why;
            EigOptions o = make_options(kOptions[kKeep], nullptr);
            o.observer = [&](const EigProgress& p) {
                if (std::string(p.stage) == "iteration" && p.sweep != 60) return;     // 只在三次回调里做（hessenberg、第 60 次迭代、done）
                const EigResult r = run_new(inner, make_options(kOptions[kKeep], nullptr));
                ++inner_runs;
                ++t.comparisons;
                const std::string w = cmp_result(inner_ref, r, "旧版", "新版", t.bytes);
                if (!w.empty() && inner_why.empty()) inner_why = w;
            };
            const EigResult r = run_new(outer, o);
            t.runs += 1 + inner_runs;
            ++t.comparisons;
            std::string why = cmp_result(outer_ref, r, "旧版", "新版", t.bytes);
            if (!why.empty()) fail_in(t, fmt("观察者里再调 eig_qr 线程=%d 外层：", th) + why);
            if (!inner_why.empty()) fail_in(t, fmt("观察者里再调 eig_qr 线程=%d 内层：", th) + inner_why);
            check(inner_runs >= 3, fmt("观察者里再调 eig_qr 线程=%d：内层只调用了 %d 次", th, inner_runs));
        }
    }
    if (g_cfg.times) say("    T3(b) 累计 %.2f s\n", now_seconds() - t0);
    // (c) rcwa::solve 的 custom_eig / 默认本征求解 / parallel_layers / sweep：结果与用旧版做本征分解时逐位相同
    {
        const int M = (nm + 1) / 2, n = 2 * M + 1;                   // 层矩阵 n = 2M+1 >= n_min（n_min 为偶数时 = n_min + 1）：走并行版
        const rcwa::Stack stack = grating_stack();
        rcwa::Options ref_opt;
        std::atomic<int> ref_calls{0};
        ref_opt.custom_eig = [&ref_calls](const Matrix& a) {
            ++ref_calls;
            return run_ref(a, EigOptions{});
        };
        set_num_threads(1);
        const rcwa::Result base = rcwa::solve(stack, 1.053, 40.0, M, rcwa::Polarization::TE, ref_opt);
        check(ref_calls == 2 && base.n_eig == 2, fmt("rcwa 基准：本征分解 %d 次（应为 2 次），层矩阵 n = %d", static_cast<int>(ref_calls), n));
        ++t.matrices;
        for (int th : {1, 4, 8}) {
            set_num_threads(th);
            for (int mode = 0; mode < 4; ++mode) {     // 0 默认本征求解；1 custom_eig；2 custom_eig + parallel_layers；3 默认 + parallel_layers
                if ((th == 1 && mode != 1) || (th == 8 && mode != 2) || (th == 4 && (mode == 0 || mode == 3))) continue;
                rcwa::Options o;
                std::atomic<int> calls{0};
                if (mode == 1 || mode == 2) o.custom_eig = [&calls](const Matrix& a) { ++calls; return run_new(a, EigOptions{}); };
                o.parallel_layers = mode >= 2;
                const rcwa::Result r = rcwa::solve(stack, 1.053, 40.0, M, rcwa::Polarization::TE, o);
                ++t.runs;
                ++t.comparisons;
                const std::string why = rcwa_result_diff(base, r, t.bytes);
                if (!why.empty()) fail_in(t, fmt("rcwa::solve 线程=%d 模式=%d：", th, mode) + why);
                if (mode == 1 || mode == 2) check(calls == 2, fmt("rcwa::solve 线程=%d 模式=%d：custom_eig 被调用 %d 次（应为 2 次）", th, mode, static_cast<int>(calls)));
            }
            // sweep：各点在 parallel_for 里求解，里面的本征分解是嵌套调用
            if (th == 1) continue;
            rcwa::Options o;
            o.custom_eig = [](const Matrix& a) { return run_new(a, EigOptions{}); };
            const std::vector<rcwa::Result> swept = rcwa::sweep(stack, {{1.053, 40.0}, {1.053, 40.0}}, M, rcwa::Polarization::TE, o);
            t.runs += 2;
            ++t.comparisons;
            std::string why = swept.size() == 2 ? rcwa_result_diff(base, swept[0], t.bytes) : std::string("sweep 结果个数不对");
            if (why.empty()) why = rcwa_result_diff(base, swept[1], t.bytes);
            if (!why.empty()) fail_in(t, fmt("rcwa::sweep 线程=%d：", th) + why);
        }
    }
    set_num_threads(1);
    finish(t, t0);
}

// ---------------------------------------------------------------------------------------------
// T4 并发的用户线程
static void test_t4(Tally& t) {
    begin(t, "T4 并发用户线程（8 个 std::thread 同时调用；另有一个线程反复改线程数）");
    const double t0 = now_seconds();
    const int nm = g_cfg.nmin;
    struct Job {
        Matrix a;
        OptId opt;
        EigResult ref;
        long long ref_calls = 0;
        std::string name;
    };
    std::vector<Job> jobs(8);
    const int sizes[8] = {nm + 1, 200, 130, 64, 257, nm, 100, nm + 40};
    const OptId opts[8] = {kDefault, kKeep, kKeepNoVec, kNoVec, kObserver, kIter2, kDefault, kKeep};
    for (int k = 0; k < 8; ++k) {
        jobs[k].a = k == 3 ? random_real(sizes[k], 400 + k) : random_complex(sizes[k], 400 + k);
        jobs[k].opt = opts[k];
        jobs[k].name = fmt("任务%d n=%d 选项=%s", k, sizes[k], kOptions[opts[k]].name);
        jobs[k].ref = run_ref(jobs[k].a, make_options(kOptions[opts[k]], &jobs[k].ref_calls));
    }
    t.matrices += 8;
    // (a) 池 4 线程，8 个用户线程同时调用（它们在线程池上排队轮流用，或各自串行）
    const auto hammer = [&](int nthreads_user, int rounds, bool churn) {
        std::vector<std::string> errors(nthreads_user);
        std::vector<long long> bytes(nthreads_user, 0), cmps(nthreads_user, 0);
        std::atomic<int> ready{0};
        std::atomic<bool> stop{false};
        std::vector<std::thread> users;
        for (int u = 0; u < nthreads_user; ++u)
            users.emplace_back([&, u] {
                ++ready;
                while (ready.load() < nthreads_user) std::this_thread::yield();
                for (int r = 0; r < rounds; ++r) {
                    const Job& j = jobs[(u + r * 3) % 8];
                    long long calls = 0;
                    const EigResult got = run_new(j.a, make_options(kOptions[j.opt], &calls));
                    ++cmps[u];
                    std::string why = cmp_result(j.ref, got, "旧版", "新版", bytes[u]);
                    if (why.empty() && calls != j.ref_calls) why = fmt("回调次数不同：旧版 %lld，新版 %lld", j.ref_calls, calls);
                    if (!why.empty() && errors[u].empty()) errors[u] = j.name + "：" + why;
                }
            });
        std::thread changer;
        if (churn)
            changer = std::thread([&] {
                int k = 0;
                while (!stop.load()) {
                    set_num_threads(1 + (k++ % 5));
                    // 不用 sleep_for：MinGW（winpthreads）把 300 µs 当 0（实测），这个线程就不停地 resize、占着 run_mutex_，
                    // 并行版的每个并行区都要和它抢锁（原型在 MinGW 上 T4 用了 497 s）；MSVC 则至少睡 15 ms。改成按 steady_clock 让出
                    const double until = now_seconds() + 0.0005;
                    while (!stop.load() && now_seconds() < until) std::this_thread::yield();
                }
            });
        for (std::thread& th : users) th.join();
        stop = true;
        if (changer.joinable()) changer.join();
        for (int u = 0; u < nthreads_user; ++u) {
            t.runs += rounds;
            t.comparisons += cmps[u];
            t.bytes += bytes[u];
            if (!errors[u].empty()) fail_in(t, fmt("%s用户线程 %d：", churn ? "线程数不停变化 " : "", u) + errors[u]);
        }
    };
    set_num_threads(4);
    hammer(8, 3, false);
    // (b) 池线程数在 1…5 之间不停地变，两个用户线程反复求解（照 test_parallel.cpp 第 7 项）
    hammer(2, 6, true);
    set_num_threads(1);
    finish(t, t0);
}

// ---------------------------------------------------------------------------------------------
// T5 小矩阵与阈值
static void test_t5(Tally& t) {
    begin(t, "T5 小矩阵与阈值（n = 0 … n_min+8，set_num_threads(8)；阈值附近另有 2、3 线程）");
    const double t0 = now_seconds();
    const int nm = g_cfg.nmin;
    std::vector<int> sizes;
    for (int n = 0; n <= nm + 8; ++n)
        if (n <= 60 || (n >= nm - 2 && n <= nm + 2) || n == nm + 8 || n % 13 == 0) sizes.push_back(n);
    for (int n : sizes) {
        const bool near_threshold = n >= nm - 1 && n <= nm + 1;
        MatCase mc;
        mc.family = "T5";
        mc.n = n;
        mc.name = "T5 rand_c n=" + std::to_string(n);
        mc.a = random_complex(n, 9000 + n);
        std::vector<OptId> opts{kDefault, kKeepNoVec};
        if (n <= 12) opts = {kDefault, kKeep, kKeepNoVec, kNoVec, kObserver, kIter2};
        if (near_threshold) opts = {kDefault, kObserver};
        run_case(mc, opts, near_threshold ? std::vector<int>{2, 3, 8} : std::vector<int>{8}, t);
        if (n >= 2 && (n <= 60 ? n % 3 == 0 : n == nm - 1 || n == nm || n == nm + 1)) {          // 实矩阵也取一些
            MatCase mr;
            mr.family = "T5";
            mr.n = n;
            mr.name = "T5 rand_r n=" + std::to_string(n);
            mr.a = random_real(n, 9500 + n);
            run_case(mr, {kKeep}, {8}, t);
        }
    }
    finish(t, t0);
}

// ---------------------------------------------------------------------------------------------
// T6 观察者
struct CallRec {
    std::string stage;
    int sweep = 0, active = 0;
    cplx shift;
    int hr = 0, hc = 0, zr = 0, zc = 0;
    std::uint64_t hh1 = 0, hh2 = 0, zh1 = 0, zh2 = 0;
    std::vector<cplx> hexact, zexact;
    bool on_caller = true;
};
struct ObsLog {
    std::vector<CallRec> calls;
    std::thread::id caller;
    double worst_similarity = 0, worst_unitary = 0;
    long long checked = 0, bad_shape = 0, region_probes = 0, region_bad = 0;
    bool order_ok = true;
    int last_active = INT_MAX, iterations = 0;
    // 窗口统计（只用来说明测试覆盖了什么）：下一个 sweep 的窗口 [l, en]；l > 0 的个数；同一个 en 上 l 变小（窗口向上扩张）的次数
    int prev_en = -2, prev_l = -2;
    long long sweeps_seen = 0, l_positive = 0, expansions = 0;
};
static long long g_cov_sweeps = 0, g_cov_lpos = 0, g_cov_expand = 0;   // n >= n_min 的矩阵，旧版那一遍里的统计
static const int kExactN = 48;

// 在回调里调用 parallel_for：若回调确实在并行区之外，应有至少两个线程参与（嵌套的话 parallel.hpp 会让它串行）
static std::atomic<bool> g_region_probe_failed{false};     // 探测失败一次之后不再探测（失败时每次要白等 1 秒）
static bool parallel_region_is_usable() {
    std::mutex m;
    std::set<std::thread::id> ids;
    const double start = now_seconds();
    parallel_for(0, 8, [&](int) {
        {
            std::lock_guard<std::mutex> lock(m);
            ids.insert(std::this_thread::get_id());
        }
        while (now_seconds() - start < 5.0) {
            {
                std::lock_guard<std::mutex> lock(m);
                if (ids.size() >= 2) break;
            }
            std::this_thread::yield();
        }
    });
    return ids.size() >= 2;
}

// math_checks：每次回调（n 大时抽查）验证 Z^H A Z = H + tI；新版的回调与旧版逐位相同，所以大矩阵上只在旧版那一遍里做
static std::function<void(const EigProgress&)> logging_observer(ObsLog& log, const Matrix& a, bool probe_region, bool math_checks) {
    const int n = a.rows();
    const double na = norm_fro(a);
    return [&log, &a, n, na, probe_region, math_checks](const EigProgress& p) {
        CallRec r;
        r.stage = p.stage;
        r.sweep = p.sweep;
        r.active = p.active;
        r.shift = p.shift;
        r.hr = p.H.rows();
        r.hc = p.H.cols();
        r.zr = p.Z.rows();
        r.zc = p.Z.cols();
        r.on_caller = std::this_thread::get_id() == log.caller;
        hash_bytes(p.H.data(), static_cast<std::size_t>(p.H.rows()) * p.H.cols() * sizeof(cplx), r.hh1, r.hh2);
        hash_bytes(p.Z.data(), static_cast<std::size_t>(p.Z.rows()) * p.Z.cols() * sizeof(cplx), r.zh1, r.zh2);
        if (n <= kExactN) {
            r.hexact.assign(p.H.data(), p.H.data() + static_cast<std::size_t>(p.H.rows()) * p.H.cols());
            r.zexact.assign(p.Z.data(), p.Z.data() + static_cast<std::size_t>(p.Z.rows()) * p.Z.cols());
        }
        const std::size_t idx = log.calls.size();
        const bool is_iter = r.stage == "iteration";
        if (r.stage != "done" && p.H.rows() == n && n > 0) {               // 下一个 sweep 的窗口：与 eig_qr 的主循环同样地找 l、必要时收缩 en
            int en = p.active, l = en;
            while (en >= 0) {
                for (l = en; l > 0; --l)
                    if (abs1(p.H(l, l - 1)) <= kEps * (abs1(p.H(l - 1, l - 1)) + abs1(p.H(l, l)))) break;
                if (l != en) break;
                --en;
            }
            if (en >= 1) {
                ++log.sweeps_seen;
                if (l > 0) ++log.l_positive;
                if (en == log.prev_en && l < log.prev_l) ++log.expansions;
                log.prev_en = en;
                log.prev_l = l;
            }
        }
        // 回调顺序：hessenberg 一次，iteration 若干次，converged 时 done 一次
        if (r.stage == "hessenberg") log.order_ok = log.order_ok && idx == 0 && p.sweep == 0 && p.active == n - 1;
        else if (is_iter) {
            ++log.iterations;
            log.order_ok = log.order_ok && idx >= 1 && p.sweep == log.iterations && p.active >= -1 && p.active <= log.last_active;
            log.last_active = p.active;
        } else if (r.stage == "done") log.order_ok = log.order_ok && p.active == -1;
        else log.order_ok = false;
        // 形状：H 在第二条次对角线以下全为零；次对角元是实数
        if (p.H.rows() == n && p.Z.rows() == n && n > 0) {
            bool shape = true;
            for (int j = 0; j < n && shape; ++j)
                for (int i = j + 2; i < n; ++i)
                    if (p.H(i, j) != cplx{}) {
                        shape = false;
                        break;
                    }
            for (int i = 1; i < n; ++i) shape = shape && p.H(i, i - 1).imag() == 0.0;
            if (!shape) ++log.bad_shape;
        }
        // Z^H A Z = H + t·I（容差）：n 小时每次都查，否则抽查
        if (math_checks && n > 0 && na > 0.0 && p.H.rows() == n && p.Z.rows() == n && (n <= 40 || idx < 2 || idx % 41 == 0 || !is_iter)) {
            Matrix shifted = p.H;
            for (int i = 0; i < n; ++i) shifted(i, i) += p.shift;
            log.worst_similarity = (std::max)(log.worst_similarity, norm_fro(adjoint(p.Z) * a * p.Z - shifted) / na);
            Matrix g = adjoint(p.Z) * p.Z;
            for (int i = 0; i < n; ++i) g(i, i) -= 1.0;
            log.worst_unitary = (std::max)(log.worst_unitary, norm_fro(g));
            ++log.checked;
        }
        if (probe_region && num_threads() > 1 && !g_region_probe_failed.load() && (idx == 0 || r.stage == "done" || idx == 7)) {
            ++log.region_probes;
            if (!parallel_region_is_usable()) {
                ++log.region_bad;
                g_region_probe_failed = true;
            }
        }
        log.calls.push_back(std::move(r));
    };
}

static std::string cmp_logs(const ObsLog& x, const ObsLog& y, long long& bytes) {
    if (x.calls.size() != y.calls.size()) return fmt("回调次数不同：旧版 %d，新版 %d", static_cast<int>(x.calls.size()), static_cast<int>(y.calls.size()));
    for (std::size_t k = 0; k < x.calls.size(); ++k) {
        const CallRec &a = x.calls[k], &b = y.calls[k];
        const std::string at = fmt("第 %d 次回调（%s，sweep %d）", static_cast<int>(k), a.stage.c_str(), a.sweep);
        if (a.stage != b.stage || a.sweep != b.sweep || a.active != b.active) return at + fmt("：阶段/sweep/active 不同（新版 %s, %d, %d）", b.stage.c_str(), b.sweep, b.active);
        if (std::memcmp(&a.shift, &b.shift, sizeof(cplx)) != 0) return at + "：累计位移 t 不同";
        if (a.hr != b.hr || a.hc != b.hc || a.zr != b.zr || a.zc != b.zc) return at + "：H 或 Z 的尺寸不同";
        const long long hb = static_cast<long long>(a.hr) * a.hc * sizeof(cplx), zb = static_cast<long long>(a.zr) * a.zc * sizeof(cplx);
        bytes += hb + zb;
        if (!a.hexact.empty() || !b.hexact.empty()) {
            const std::size_t dh = first_diff_byte(a.hexact.data(), b.hexact.data(), a.hexact.size() * sizeof(cplx));
            if (dh != kSame) return at + "：H 的" + diff_text("", a.hexact.data(), b.hexact.data(), dh, a.hr, "旧版", "新版");
            const std::size_t dz = first_diff_byte(a.zexact.data(), b.zexact.data(), a.zexact.size() * sizeof(cplx));
            if (dz != kSame) return at + "：Z 的" + diff_text("", a.zexact.data(), b.zexact.data(), dz, a.zr, "旧版", "新版");
        } else {
            if (a.hh1 != b.hh1 || a.hh2 != b.hh2) return at + "：H 的散列不同（回调时 H 与旧版不一致）";
            if (a.zh1 != b.zh1 || a.zh2 != b.zh2) return at + "：Z 的散列不同（回调时 Z 与旧版不一致；observer 前没有 flush？）";
        }
    }
    return {};
}

static void observer_case(const MatCase& mc, const OptSpec& os, const std::vector<int>& threads, Tally& t) {
    ObsLog ref_log;
    ref_log.caller = std::this_thread::get_id();
    EigOptions ro = make_options(os, nullptr);
    ro.observer = logging_observer(ref_log, mc.a, false, true);
    const EigResult ref = run_ref(mc.a, ro);
    ++t.runs;
    const int n = mc.n;
    const std::string ctx0 = mc.name + " 选项=" + os.name;
    check(ref_log.order_ok, ctx0 + "：旧版的回调顺序/sweep/active 不合要求（测试本身有问题）");
    if (n >= g_cfg.nmin) {
        g_cov_sweeps += ref_log.sweeps_seen;
        g_cov_lpos += ref_log.l_positive;
        g_cov_expand += ref_log.expansions;
    }
    const long long expect = static_cast<long long>(ref.sweeps) + (ref.info == 0 ? 2 : 1);
    check(static_cast<long long>(ref_log.calls.size()) == expect, ctx0 + fmt("：旧版回调 %d 次，应为 sweeps + %d = %lld", static_cast<int>(ref_log.calls.size()), ref.info == 0 ? 2 : 1, expect));
    const double tol_n = (std::max)(1e-13, 1e-15 * n);                 // n = 360 时旧版自己就超过 1e-13（实测）：容差随 n 放宽
    check(ref_log.worst_similarity <= tol_n && ref_log.worst_unitary <= tol_n, ctx0 + fmt("旧版自己的回调就不满足 Z^H A Z = H + tI（%.2e / %.2e，测试本身有问题）", ref_log.worst_similarity, ref_log.worst_unitary));
    for (const int th : threads) {
        set_num_threads(th);
        ObsLog log;
        log.caller = std::this_thread::get_id();
        EigOptions o = make_options(os, nullptr);
        o.observer = logging_observer(log, mc.a, n >= 40, n <= 60);
        const EigResult got = run_new(mc.a, o);
        ++t.runs;
        const std::string ctx = ctx0 + fmt(" 线程=%d：", th);
        ++t.comparisons;
        std::string why = cmp_result(ref, got, "旧版", "新版", t.bytes);
        if (!why.empty()) fail_in(t, ctx + why);
        ++t.comparisons;
        why = cmp_logs(ref_log, log, t.bytes);
        if (!why.empty()) fail_in(t, ctx + why);
        bool all_on_caller = true;
        for (const CallRec& r : log.calls) all_on_caller = all_on_caller && r.on_caller;
        check(all_on_caller, ctx + "回调不在调用线程上");
        check(static_cast<long long>(log.calls.size()) == expect, ctx + fmt("回调 %d 次，应为 %lld 次（sweeps + %d）", static_cast<int>(log.calls.size()), expect, ref.info == 0 ? 2 : 1));
        check(log.order_ok && log.bad_shape == 0, ctx + fmt("回调的顺序或 H 的形状不合要求（形状不合 %lld 次）", log.bad_shape));
        check(log.worst_similarity <= 1e-13, ctx + fmt("max ||Z^H A Z − (H + tI)||/||A|| = %.3e > 1e-13", log.worst_similarity));
        check(log.worst_unitary <= 1e-13, ctx + fmt("max ||Z^H Z − I||_F = %.3e > 1e-13", log.worst_unitary));
        check(log.region_bad == 0, ctx + fmt("回调里调用 parallel_for 没有多个线程参与（%lld 次探测中 %lld 次）：回调被放进了并行区？", log.region_probes, log.region_bad));
    }
    set_num_threads(1);
}

static void test_t6(Tally& t) {
    begin(t, "T6 观察者（线程 id、回调次数、每次回调的 H 与 Z、Z^H A Z = H + tI）");
    const double t0 = now_seconds();
    const int nm = g_cfg.nmin;
    std::vector<MatCase> list;
    for (int n : {1, 2, 3, 5, 12, 40}) add(list, "rand_c", "", random_complex(n, 600 + n), 0);
    add(list, "rand_r", "", random_real(9, 609), 0);
    add(list, "block", "", block_upper(30, {0, 10, 20}, 631), 0);
    add(list, "cyclic", "", cyclic_permutation(16), 0);
    add(list, "pq2d", "uni0 M=1", pq2d(1, 1.7, 1.7, 1.0, 1.0, 0.61, 0, 0, 1), 0);
    add(list, "rand_c", "", random_complex(100, 700), 1);
    add(list, "herm", "", hermitian_indefinite(120, 701), 1);
    add(list, "rand_c", "", random_complex(nm, 703), 1);
    add(list, "block", "", block_upper(nm + 20, {0, 60, 140}, 705), 1);
    add(list, "rank1", "", rank_one(nm + 8, 7302), 1);                      // 窗口向上扩张的矩阵
    add(list, "rand_c", "", random_complex(360, 707), 2);                    // 窗口 >= 346：S 区（H 行右乘）真正并行的 sweep 里回调
    if (g_cfg.big) {
        add(list, "cyclic", "", cyclic_permutation(nm + 4), 1);
        add(list, "pq2d", "sq M=5", pq2d(5, 3.5, 1.0, 0.5, 0.5, 0.61, 0, 0, 0), 1);
        add(list, "rand_c", "", random_complex(300, 706), 1);
    }
    {   // P_8 ⊕ B_4：迭代停在第 8 个本征值上（没有 "done" 回调）
        Matrix c(12, 12);
        const Matrix b = random_complex(4, 1002);
        for (int i = 1; i < 8; ++i) c(i, i - 1) = 1.0;
        c(0, 7) = 1.0;
        for (int j = 0; j < 4; ++j)
            for (int i = 0; i < 4; ++i) c(8 + i, 8 + j) = b(i, j);
        add(list, "P8+B4", "", c, 0);
    }
    const OptSpec obs_default = kOptions[kObserver], obs_novec = kOptions[kObserverNoVec];
    const OptSpec obs_keep{"observer,keep", true, true, true, kEps, 30}, obs_iter{"observer,iter3", true, true, true, kEps, 3};
    for (const MatCase& mc : list) {
        ++t.matrices;
        std::vector<int> threads = g_cfg.threads;
        if (mc.n > 60 && mc.n < 130) threads = {1, 3, 8};
        if (mc.n >= 130) threads = mc.n == nm ? std::vector<int>{2, 3, 7, 8} : std::vector<int>{3, 8};   // 阈值处的矩阵：7 线程也要看回调
        observer_case(mc, obs_default, threads, t);
        if (mc.n == nm) observer_case(mc, obs_novec, threads, t);            // observer 且 vectors=keep=false：full 只因 observer 而成立
        if (mc.n <= 60) {
            observer_case(mc, obs_novec, threads, t);
            observer_case(mc, obs_keep, threads, t);
            observer_case(mc, obs_iter, threads, t);
        }
    }
    // 回调里抛异常：必须原样传出来，回调次数照旧，之后线程池照常可用
    {
        const Matrix a = random_complex(nm + 1, 720);
        const EigResult ref = run_ref(a, make_options(kOptions[kKeep], nullptr));
        ++t.matrices;
        for (int th : {1, 8}) {
            set_num_threads(th);
            for (int stop_at : {0, 7, 150}) {
                int seen_ref = 0, seen_new = 0;
                std::string msg_ref = "-", msg_new = "-";
                for (int which = 0; which < 2; ++which) {
                    int& seen = which ? seen_new : seen_ref;
                    std::string& msg = which ? msg_new : msg_ref;
                    EigOptions o = make_options(kOptions[kKeep], nullptr);
                    o.observer = [&seen, stop_at](const EigProgress&) {
                        if (seen++ == stop_at) throw std::runtime_error("观察者要求停止");
                    };
                    try {
                        if (which) run_new(a, o);
                        else run_ref(a, o);
                        msg = "没有抛出";
                    } catch (const std::runtime_error& e) {
                        msg = e.what();
                    } catch (...) {
                        msg = "抛出了别的类型";
                    }
                }
                ++t.runs;
                check(msg_ref == msg_new && seen_ref == seen_new, fmt("回调里抛异常 线程=%d 第 %d 次回调：旧版 [%s] 回调 %d 次，新版 [%s] 回调 %d 次", th, stop_at, msg_ref.c_str(), seen_ref, msg_new.c_str(), seen_new));
                ++t.comparisons;
                const EigResult after = run_new(a, make_options(kOptions[kKeep], nullptr));
                ++t.runs;
                const std::string why = cmp_result(ref, after, "旧版", "新版", t.bytes);
                if (!why.empty()) fail_in(t, fmt("回调抛异常之后 线程=%d：", th) + why);
                std::atomic<int> sum{0};
                parallel_for(0, 100, [&](int i) { sum += i; });
                check(sum == 4950, "回调抛异常之后 parallel_for 不再工作");
            }
        }
        set_num_threads(1);
    }
    say("  覆盖：n >= n_min 的矩阵里，旧版的 %lld 个 sweep 中 l > 0 的 %lld 个，窗口向上扩张（同一个 en 上 l 变小）%lld 次\n", g_cov_sweeps, g_cov_lpos, g_cov_expand);
    finish(t, t0, "（n > 48 的回调里 H、Z 用 128 位散列比较）");
}

// ---------------------------------------------------------------------------------------------
// T7 vectors = false / keep_transform 的组合
static void test_t7(Tally& t) {
    begin(t, "T7 vectors / keep_transform 四种组合");
    const double t0 = now_seconds();
    const int nm = g_cfg.nmin;
    std::vector<MatCase> list;
    for (int n : {1, 2, 5, 17, 64}) add(list, "rand_c", "", random_complex(n, 800 + n), 0);
    add(list, "rand_c", "", random_complex(nm, 810), 0, true);
    add(list, "block", "", block_upper(nm + 20, {0, 80}, 813), 1);
    const OptSpec combos[4] = {{"vec=0,keep=0", false, false, false, kEps, 30},
                               {"vec=0,keep=1", false, true, false, kEps, 30},
                               {"vec=1,keep=0", true, false, false, kEps, 30},
                               {"vec=1,keep=1", true, true, false, kEps, 30}};
    for (const MatCase& mc : list) {
        ++t.matrices;
        const int n = mc.n;
        for (const OptSpec& os : combos) {
            const std::vector<int> th = mc.n >= 130 ? std::vector<int>{3, 8} : g_cfg.threads;
            const EigResult ref = run_ref(mc.a, make_options(os, nullptr));
            ++t.runs;
            for (int threads : th) {
                set_num_threads(threads);
                const EigResult got = run_new(mc.a, make_options(os, nullptr));
                ++t.runs;
                ++t.comparisons;
                const std::string ctx = case_context(mc.name, os.name, threads);
                const std::string why = cmp_result(ref, got, "旧版", "新版", t.bytes);
                if (!why.empty()) fail_in(t, ctx + why);
                // 该有的有、不该有的没有
                const bool want_v = os.vectors && got.info == 0, want_zt = os.keep;
                check(got.vectors.empty() == !want_v || n == 0, ctx + "vectors 的有无不对");
                check(got.Z.empty() == !want_zt || n == 0, ctx + "Z 的有无不对");
                check(got.T.empty() == !want_zt || n == 0, ctx + "T 的有无不对");
                check(static_cast<int>(got.values.size()) == n, ctx + "values 的长度不对");
                if (want_v && n > 0) check(got.vectors.rows() == n && got.vectors.cols() == n, ctx + "vectors 不是 n×n");
                const bool shapes_ok = got.Z.rows() == n && got.Z.cols() == n && got.T.rows() == n && got.T.cols() == n && static_cast<int>(got.values.size()) == n;
                if (want_zt && n > 0) {
                    check(shapes_ok, ctx + "Z、T 不是 n×n");
                    if (got.info == 0 && shapes_ok) {
                        bool diag_ok = true;
                        for (int i = 0; i < n; ++i) diag_ok = diag_ok && std::memcmp(&got.T(i, i), &got.values[i], sizeof(cplx)) == 0;
                        check(diag_ok, ctx + "T 的对角线不等于 values");
                        bool lower = true;
                        for (int j = 0; j < n; ++j)
                            for (int i = j + 1; i < n; ++i) lower = lower && got.T(i, j) == cplx{};
                        check(lower, ctx + "T 不是上三角");
                        if (n <= 140) {
                            check(orthogonality_defect(got.Z) <= 1e-13, ctx + "Z 不是酉矩阵");
                            check(norm_fro(adjoint(got.Z) * mc.a * got.Z - got.T) / (std::max)(norm_fro(mc.a), 1e-300) <= 1e-13, ctx + "Z^H A Z ≠ T");
                        }
                    }
                }
            }
        }
    }
    set_num_threads(1);
    finish(t, t0);
}

// ---------------------------------------------------------------------------------------------
// T8 非收敛
static void test_t8(Tally& t) {
    begin(t, "T8 非收敛（max_iter = 1 … 6；P_8 ⊕ B_4 等）");
    const double t0 = now_seconds();
    const int nm = g_cfg.nmin;
    std::vector<MatCase> list;
    add(list, "rand_c", "", random_complex(20, 1001), 0);
    add(list, "rand_c", "", random_complex(60, 1060), 0);
    add(list, "rand_c", "", random_complex(nm + 8, 1061), 0);
    add(list, "block", "", block_upper(nm + 1, {0, 70}, 1063), 1);
    add(list, "cyclic", "", cyclic_permutation(16), 0);
    add(list, "cyclic", "", cyclic_permutation(nm + 4), 1);
    add(list, "companion", "圆环", companion_ring(80, 1064), 0);
    add(list, "pq2d", "uni M=3", pq2d(3, 1.7, 1.7, 1.0, 1.0, 0.61, 0, 0, 2), 0);
    if (g_cfg.big) {
        add(list, "rand_c", "", random_complex(257, 1062), 1);
        add(list, "pq2d", "sq M=5", pq2d(5, 3.5, 1.0, 0.5, 0.5, 0.61, 0, 0, 0), 1);
    }
    {
        Matrix c(12, 12);
        const Matrix b = random_complex(4, 1002);
        for (int i = 1; i < 8; ++i) c(i, i - 1) = 1.0;
        c(0, 7) = 1.0;
        for (int j = 0; j < 4; ++j)
            for (int i = 0; i < 4; ++i) c(8 + i, 8 + j) = b(i, j);
        add(list, "P8+B4", "", c, 0);
    }
    {   // 更大的 P ⊕ B：循环置换在左上，随机块在右下；大块先收敛、停在 P 上
        const int m = nm + 8, k = 16;
        Matrix c(m + k, m + k);
        const Matrix b = random_complex(k, 1065);
        for (int i = 1; i < m; ++i) c(i, i - 1) = 1.0;
        c(0, m - 1) = 1.0;
        for (int j = 0; j < k; ++j)
            for (int i = 0; i < k; ++i) c(m + i, m + j) = b(i, j);
        add(list, "P+B", "", c, 1);
    }
    long long unconverged = 0, partial = 0;
    for (const MatCase& mc : list) {
        ++t.matrices;
        const bool large = mc.n >= 130;                                  // 大矩阵上只取 max_iter = 1、3、6
        for (int max_iter = 1; max_iter <= 6; ++max_iter) {
            if (large && max_iter != 1 && max_iter != 3 && max_iter != 6) continue;
            const OptSpec specs[3] = {{"iter,keep", true, true, false, kEps, max_iter},
                                      {"iter,novec", false, false, false, kEps, max_iter},
                                      {"iter,observer", true, false, true, kEps, max_iter}};
            const EigResult ref = run_ref(mc.a, make_options(specs[0], nullptr));
            ++t.runs;
            if (ref.info > 0) {
                ++unconverged;
                if (ref.info < mc.n) ++partial;
            }
            for (int si = 0; si < 3; ++si) {
                if (large && si == 1 && max_iter != 3) continue;
                const OptSpec os = specs[si];
                long long ref_calls = 0;
                const EigResult r2 = si == 0 ? ref : run_ref(mc.a, make_options(os, &ref_calls));
                if (si != 0) ++t.runs;
                for (int th : threads_for(mc)) {
                    set_num_threads(th);
                    long long calls = 0;
                    const EigResult got = run_new(mc.a, make_options(os, &calls));
                    ++t.runs;
                    ++t.comparisons;
                    const std::string ctx = mc.name + fmt(" max_iter=%d 选项=%s 线程=%d：", max_iter, os.name, th);
                    std::string why = cmp_result(r2, got, "旧版", "新版", t.bytes);
                    if (why.empty() && calls != ref_calls) why = fmt("回调次数不同：旧版 %lld，新版 %lld", ref_calls, calls);
                    if (!why.empty()) fail_in(t, ctx + why);
                    const bool shapes_ok = got.Z.rows() == mc.n && got.Z.cols() == mc.n && got.T.rows() == mc.n && got.T.cols() == mc.n && static_cast<int>(got.values.size()) == mc.n;
                    if (si == 0 && got.info > 0) {                              // 未收敛时的结构（独立于旧版）
                        check(got.vectors.empty(), ctx + "未收敛却给了本征向量");
                        check(shapes_ok, ctx + "keep_transform 下未收敛，Z、T 缺失");
                        if (shapes_ok && mc.n <= 140) {
                            check(orthogonality_defect(got.Z) <= 1e-13, ctx + "未收敛时 Z 不是酉矩阵");
                            check(norm_fro(adjoint(got.Z) * mc.a * got.Z - got.T) / norm_fro(mc.a) <= 1e-13, ctx + "未收敛时 Z^H A Z ≠ T");
                        }
                        bool tri = shapes_ok;
                        for (int j = 0; j < mc.n && tri; ++j)
                            for (int i = (std::max)(j + 1, got.info); i < mc.n; ++i)
                                if (got.T(i, j) != cplx{}) {
                                    tri = false;
                                    break;
                                }
                        check(tri, ctx + "已收敛的行不是严格上三角");
                        bool diag = shapes_ok;
                        for (int i = 0; i < mc.n && diag; ++i) diag = std::memcmp(&got.T(i, i), &got.values[i], sizeof(cplx)) == 0;
                        check(diag, ctx + "T 的对角线不等于 values");
                    }
                }
            }
        }
    }
    set_num_threads(1);
    finish(t, t0, fmt("（旧版有 %lld 次未收敛，其中 %lld 次已求出一部分本征值）", unconverged, partial).c_str());
}

// ---------------------------------------------------------------------------------------------
// T9 近亏损与精确简并
static void test_t9(Tally& t) {
    begin(t, "T9 近亏损 / 精确简并（均匀层、Jordan 块、三阶例外点、全部本征值重合）");
    const double t0 = now_seconds();
    const int nm = g_cfg.nmin;
    // (a) 均匀层的层矩阵：test_eig_qr.cpp 的 180 个（n = 5 … 81），加上本征向量矩阵的条件数界
    {
        long long cond_checks = 0;
        double worst_cond = 0;
        for (int n = 5; n <= 81; n += 4)
            for (double index : {1.46, 1.96, 2.0})
                for (double ratio : {0.0706, 0.61, 1.0}) {
                    MatCase mc;
                    mc.family = "uniform_layer";
                    mc.n = n;
                    mc.name = fmt("uniform_layer n=%d index=%.2f ratio=%.4f", n, index, ratio);
                    mc.a = uniform_layer_matrix(n, index, ratio);
                    run_case(mc, n <= 33 ? std::vector<OptId>{kDefault, kKeep} : std::vector<OptId>{kDefault}, {1, 4, 8}, t);
                    if (n <= 9) {                                           // 条件数界与旧版的判据相同（Jacobi SVD 较慢，只查较小的 n）
                        set_num_threads(8);
                        const EigResult r = run_new(mc.a, EigOptions{});
                        set_num_threads(1);
                        const double cond = cond2(r.vectors);
                        worst_cond = (std::max)(worst_cond, cond);
                        check(r.info == 0 && cond <= 1e4, mc.name + fmt("：本征向量矩阵条件数 %.3e > 1e4", cond));
                        ++cond_checks;
                    }
                }
        say("  均匀层 n = 5…81 共 180 个，其中 %lld 个查了条件数，最大 %.2e\n", cond_checks, worst_cond);
    }
    std::vector<MatCase> list;
    for (int n : {nm, nm + 1, 201, 257}) {
        if (!g_cfg.big && n != nm + 1) continue;
        for (double index : {1.46, 2.0})
            for (double ratio : {0.0706, 1.0})
                if (g_cfg.big || (index == 2.0 && ratio == 1.0))
                    add(list, "uniform_layer", fmt("index=%.2f ratio=%.4f", index, ratio), uniform_layer_matrix(n, index, ratio), 2);
    }
    // (b) Jordan 块：直接给上三角（回代里分母恰为零），以及经酉相似藏起来的（所有本征值在舍入误差内重合）
    for (int n : {5, 64}) add(list, "jordan", "J_n(2)", jordan_block(n, 2.0), 0);
    add(list, "jordan", "J_n(2)", jordan_block(nm + 1, 2.0), 2, true);
    add(list, "jordan", "J_n(2) 上三角", jordan_block(257, cplx(0.5, -1.0)), 2);
    for (int n : {5, 20, 64}) add(list, "jordan_hidden", "Q J Q^H", unitary_similar(jordan_block(n, 2.0), 9100 + n), 0);
    add(list, "jordan_hidden", "Q J Q^H", unitary_similar(jordan_block(nm + 1, 2.0), 9200), 2);
    {
        Matrix j = Matrix::diagonal(Vector{2.0, 2.0, 5.0, -1.0, cplx(0.0, 0.5)});
        j(0, 1) = 1.0;
        add(list, "jordan2x2", "Q (J_2 ⊕ …) Q^H", unitary_similar(j, 9300), 0);
    }
    // (c) 三阶例外点（复对称矩阵，A³ = 0），单个与直和
    {
        const Matrix ep{{0.0, -1.0, 0.0}, {-1.0, 0.0, kI}, {0.0, kI, 0.0}};
        add(list, "exceptional", "3阶", ep, 0);
        add(list, "exceptional", "3阶×5", repeat_block(ep, 5), 0);
        add(list, "exceptional", "3阶×65", repeat_block(ep, (nm + 3) / 3), 2, true);
        if (g_cfg.big) add(list, "exceptional", "3阶 + 随机", direct_sum({repeat_block(ep, 30), random_complex((std::max)(nm + 8 - 90, 20), 9400)}), 2);
    }
    // (d) 全部本征值重合：c·I、c·I + 极小扰动、酉相似的高重数厄米、循环置换的直和（谱重复）
    add(list, "scalar", "c·I", Matrix::identity(nm + 8) * cplx(0.3, -1.2), 2);
    {
        Matrix a = Matrix::identity(100) * cplx(1.0, 0.0) + hermitian_part(random_complex(100, 9500)) * cplx(1e-11, 0.0);
        add(list, "scalar", "I + 1e-11·H", a, 2);
        if (g_cfg.big) {
            Matrix b = Matrix::identity(nm + 8) * cplx(1.0, 0.0) + hermitian_part(random_complex(nm + 8, 9501)) * cplx(1e-12, 0.0);
            add(list, "scalar", "I + 1e-12·H", b, 2);
        }
    }
    {
        Vector d(64), e(nm + 8);
        for (int i = 0; i < 64; ++i) d[i] = i < 20 ? 1.0 : i < 41 ? 2.0 : 3.0;
        for (int i = 0; i < nm + 8; ++i) e[i] = i < 70 ? -1.0 : i < 130 ? 0.5 : 2.0;
        add(list, "herm_multi", "重数 20/21/23", hermitian_part(unitary_similar(Matrix::diagonal(d), 9600)), 0);
        if (g_cfg.big) add(list, "herm_multi", "重数 70/60/…", hermitian_part(unitary_similar(Matrix::diagonal(e), 9601)), 2);
    }
    add(list, "perm_sum", "P_8×k", repeat_block(cyclic_permutation(8), nm / 8), 2);
    add(list, "perm_sum", "P_8×k", repeat_block(cyclic_permutation(8), 5), 0);
    // (e) 二维：uni0（E = εI 精确）与 uni（sinc 公式的舍入误差）
    add(list, "pq2d", "uni0 M=2", pq2d(2, 1.7, 1.7, 1.0, 1.0, 0.61, 0, 0, 1), 0);
    add(list, "pq2d", "uni M=2", pq2d(2, 1.7, 1.7, 1.0, 1.0, 0.61, 0, 0, 2), 0);
    add(list, "pq2d", "uni0 M=5", pq2d(5, 1.7, 1.7, 1.0, 1.0, 0.61, 0, 0, 1), 2);
    if (g_cfg.big) {
        add(list, "pq2d", "uni M=5", pq2d(5, 1.7, 1.7, 1.0, 1.0, 0.61, 0, 0, 2), 2);
        add(list, "pq2d", "uni0 斜入射 M=5", pq2d(5, 1.7, 1.7, 1.0, 1.0, 0.61, 25.0, 30.0, 1), 2);
    }
    for (const MatCase& mc : list) {
        std::vector<OptId> opts = option_set(mc.tier);
        opts.erase(std::remove(opts.begin(), opts.end(), kEps0), opts.end());
        run_case(mc, opts, threads_for(mc), t);
        // 例外点要的迭代次数多：放宽上限再各跑一次，覆盖“终于收敛”的路径
        if (mc.family == "exceptional" || mc.family == "jordan_hidden" || mc.family == "perm_sum") {   // 例外点收敛得慢
            const OptSpec wide{"max_iter=100,keep", true, true, false, kEps, 100};
            const EigResult ref = run_ref(mc.a, make_options(wide, nullptr));
            ++t.runs;
            for (int th : threads_for(mc)) {
                set_num_threads(th);
                const EigResult got = run_new(mc.a, make_options(wide, nullptr));
                ++t.runs;
                ++t.comparisons;
                const std::string why = cmp_result(ref, got, "旧版", "新版", t.bytes);
                if (!why.empty()) fail_in(t, case_context(mc.name, wide.name, th) + why);
            }
            set_num_threads(1);
        }
    }
    finish(t, t0);
}

// ---------------------------------------------------------------------------------------------
// T10 极端量级与非法输入
struct Thrown {
    bool threw = false;
    std::string type, what;
};
static Thrown catch_run(bool use_ref, const Matrix& a, const EigOptions& o) {
    Thrown r;
    try {
        if (use_ref) run_ref(a, o);
        else run_new(a, o);
    } catch (const std::exception& e) {
        r.threw = true;
        r.type = typeid(e).name();
        r.what = e.what();
    } catch (...) {
        r.threw = true;
        r.type = "non-std";
    }
    return r;
}

static void test_t10(Tally& t) {
    begin(t, "T10 极端量级与非法输入");
    const double t0 = now_seconds();
    const int nm = g_cfg.nmin;
    // (a) 极端量级：乘 2^±900、1e±300（及其它）；量级悬殊的块；全部次正规数；DBL_MAX
    std::vector<MatCase> list;
    for (int n : {12, 40, nm + 1}) {
        const Matrix base = random_complex(n, 10000 + n);
        const bool big = n > 130;
        for (int k : {-900, 900}) add(list, "scaled", fmt("×2^%d", k), scaled_by_pow2(base, k), big ? 2 : 1);
        for (double f : {1e300, 1e-300, 1e150, 1e-160}) {
            if (big && (f == 1e150 || f == 1e-160 || (f == 1e300 && !g_cfg.big)) ) continue;
            add(list, "scaled", fmt("×%.0e", f), scaled_by(base, f), big ? 2 : 1);
        }
    }
    add(list, "scaled", "×2^-1000 (次正规数边缘)", scaled_by_pow2(random_complex(60, 10100), -1000), 1);
    add(list, "scaled", "×2^-1060 (次正规数)", scaled_by_pow2(random_complex(60, 10101), -1060), 1);
    add(list, "scaled", "pq2d ×1e-250", scaled_by(pq2d(2, 3.5, 1.0, 0.5, 0.5, 0.61, 0, 0, 0), 1e-250), 1);
    {   // 同一个矩阵里量级悬殊的块：A = [s1·A1  C; 0  s2·A2]
        const int n1 = nm / 2, n2 = nm - nm / 2 + 4;
        const Matrix a1 = random_complex(n1, 10201), a2 = random_complex(n2, 10202), c = random_complex(n1 + n2, 10203);
        const double scales[3][2] = {{1.0, 1e-200}, {1e200, 1.0}, {1e-150, 1e150}};
        for (const auto& s : scales) {
            if (!g_cfg.big && s[0] == 1e200) continue;
            const int n = n1 + n2;
            Matrix a(n, n);
            const double sc = (std::max)(s[0], s[1]);
            for (int j = 0; j < n1; ++j)
                for (int i = 0; i < n1; ++i) a(i, j) = a1(i, j) * s[0];
            for (int j = 0; j < n2; ++j) {
                for (int i = 0; i < n2; ++i) a(n1 + i, n1 + j) = a2(i, j) * s[1];
                for (int i = 0; i < n1; ++i) a(i, n1 + j) = c(i, j) * sc;
            }
            add(list, "mixed_scales", fmt("%.0e / %.0e", s[0], s[1]), a, 2);
        }
    }
    for (int n : {3, 10}) add(list, "dbl_max", "全为 DBL_MAX", Matrix(n, n, cplx((std::numeric_limits<double>::max)(), 0.0)), 0);
    add(list, "dbl_max", "全为 DBL_MAX", Matrix(nm + 8, nm + 8, cplx((std::numeric_limits<double>::max)(), 0.0)), 2);
    add(list, "subnormal", "全是次正规数", scaled_by_pow2(random_complex(50, 10301), -1070), 1);
    add(list, "subnormal", "对角 j·5e-324", [] {
        Matrix a = random_complex(40, 10302);
        for (int j = 0; j < 40; ++j) {
            for (int i = j + 1; i < 40; ++i) a(i, j) = 0.0;
            a(j, j) = j * 5e-324;
        }
        return a;
    }(), 0);
    if (g_cfg.big)
        add(list, "one_huge", "一个 1e300 的元素", [&] {
            Matrix a = random_complex(nm + 8, 10303);
            a(5, 7) = 1e300;
            return a;
        }(), 2);
    for (const MatCase& mc : list) {
        std::vector<OptId> opts = mc.tier == 2 ? std::vector<OptId>{kDefault, kKeepNoVec} : std::vector<OptId>{kDefault, kKeep, kKeepNoVec, kIter2, kObserver};
        run_case(mc, opts, mc.n < 130 ? std::vector<int>{1, 3, 8} : std::vector<int>{3, 8}, t);
    }
    // (b) 非法输入：新版与旧版抛同类型、同消息的异常；校验在并行区之前完成，之后线程池照常可用
    {
        const double nan = std::numeric_limits<double>::quiet_NaN(), inf = std::numeric_limits<double>::infinity();
        struct Bad {
            std::string name;
            Matrix a;
            EigOptions o;
        };
        std::vector<Bad> bads;
        const auto with = [&](const Matrix& a, double eps, int max_iter) {
            EigOptions o;
            o.eps = eps;
            o.max_iter = max_iter;
            return Bad{"", a, o};
        };
        const auto push = [&](const std::string& name, Bad b) {
            b.name = name;
            bads.push_back(std::move(b));
        };
        for (int n : {1, 5, nm + 1, 300}) {
            if (n == 300 && !g_cfg.big) continue;
            Matrix ok = random_complex(n, 11000 + n);
            for (int where = 0; where < 3; ++where) {
                const int i = where == 0 ? 0 : where == 1 ? n - 1 : n / 2, j = where == 2 ? n - 1 : i;
                for (int kind = 0; kind < 4; ++kind) {
                    if (kind == 1 || kind == 2) continue;
                    Matrix a = ok;
                    const cplx bad = kind == 0 ? cplx(nan, 0.0) : kind == 1 ? cplx(0.0, nan) : kind == 2 ? cplx(inf, 0.0) : cplx(0.0, -inf);
                    a(i, j) = bad;
                    push(fmt("n=%d 元素(%d,%d)=%s", n, i, j, kind == 0 ? "NaN+0i" : kind == 1 ? "0+NaNi" : kind == 2 ? "Inf" : "-Inf i"), with(a, kEps, 30));
                }
            }
            push(fmt("n=%d eps<0", n), with(ok, -1e-16, 30));
            push(fmt("n=%d eps=NaN", n), with(ok, nan, 30));
            push(fmt("n=%d eps=Inf", n), with(ok, inf, 30));
            push(fmt("n=%d max_iter=0", n), with(ok, kEps, 0));
            push(fmt("n=%d max_iter=-5", n), with(ok, kEps, -5));
            push(fmt("n=%d max_iter=INT_MIN", n), with(ok, kEps, INT_MIN));
            Matrix nanm = ok;
            nanm(0, 0) = nan;
            push(fmt("n=%d NaN 且 eps<0", n), with(nanm, -1.0, 30));
            push(fmt("n=%d NaN 且 max_iter=0", n), with(nanm, kEps, 0));
            push(fmt("n=%d eps<0 且 max_iter=0", n), with(ok, -1.0, 0));
        }
        for (auto rc : std::vector<std::pair<int, int>>{{3, 4}, {4, 3}, {0, 5}, {5, 0}, {nm + 1, nm + 2}, {300, 299}, {1, 2}}) {
            push(fmt("非方阵 %d×%d", rc.first, rc.second), with(Matrix(rc.first, rc.second, cplx(1.0, 0.0)), kEps, 30));
            push(fmt("非方阵 %d×%d 且 eps<0", rc.first, rc.second), with(Matrix(rc.first, rc.second, cplx(1.0, 0.0)), -1.0, 30));
            Matrix nanm(rc.first, rc.second, cplx(1.0, 0.0));
            if (rc.first > 0 && rc.second > 0) nanm(0, 0) = nan;
            push(fmt("非方阵 %d×%d 且含 NaN 且 max_iter=0", rc.first, rc.second), with(nanm, kEps, 0));
        }
        const Matrix sane = random_complex(nm + 1, 11999);
        const EigResult sane_ref = run_ref(sane, make_options(kOptions[kKeep], nullptr));
        for (const Bad& b : bads) {
            ++t.matrices;
            for (int th : {1, 8}) {
                set_num_threads(th);
                const Thrown r = catch_run(true, b.a, b.o), g = catch_run(false, b.a, b.o);
                t.runs += 2;
                ++t.comparisons;
                const std::string ctx = fmt("非法输入 [%s] 线程=%d：", b.name.c_str(), th);
                if (r.threw != g.threw || r.type != g.type || r.what != g.what)
                    fail_in(t, ctx + fmt("异常不同：旧版 %s[%s]「%s」，新版 %s[%s]「%s」", r.threw ? "抛出" : "没抛", r.type.c_str(), r.what.c_str(), g.threw ? "抛出" : "没抛", g.type.c_str(), g.what.c_str()));
                check(r.threw && r.type == typeid(std::invalid_argument).name(), ctx + "旧版没有抛 std::invalid_argument（测试本身有问题）");
            }
        }
        // 抛过异常之后，池里不应残留任何状态
        for (int th : {4, 8}) {
            set_num_threads(th);
            ++t.comparisons;
            const EigResult after = run_new(sane, make_options(kOptions[kKeep], nullptr));
            ++t.runs;
            const std::string why = cmp_result(sane_ref, after, "旧版", "新版", t.bytes);
            if (!why.empty()) fail_in(t, fmt("非法输入之后 线程=%d：", th) + why);
        }
        // 合法但特别的选项：eps = −0、eps 很大、max_iter 很大、n = 0
        std::vector<MatCase> odd;
        add(odd, "rand_c", "", random_complex(nm + 1, 11500), 2);
        add(odd, "rand_c", "", random_complex(30, 11501), 2);
        add(odd, "empty", "", Matrix(), 2);
        for (const MatCase& mc : odd) {
            ++t.matrices;
            for (const OptSpec& os : {OptSpec{"eps=-0", true, true, false, -0.0, 30}, OptSpec{"eps=1e300", true, true, false, 1e300, 30},
                                      OptSpec{"max_iter=INT_MAX", true, true, false, kEps, INT_MAX}, OptSpec{"eps=denorm", false, true, false, 5e-324, 30}}) {
                const EigResult ref = run_ref(mc.a, make_options(os, nullptr));
                ++t.runs;
                for (int th : {1, 3, 8}) {
                    set_num_threads(th);
                    const EigResult got = run_new(mc.a, make_options(os, nullptr));
                    ++t.runs;
                    ++t.comparisons;
                    const std::string why = cmp_result(ref, got, "旧版", "新版", t.bytes);
                    if (!why.empty()) fail_in(t, case_context(mc.name, os.name, th) + why);
                }
            }
        }
    }
    set_num_threads(1);
    finish(t, t0);
}

// ---------------------------------------------------------------------------------------------
// T11 三个保留的函数单独与旧版比较
namespace nq = stablercwa::detail::eigqr;
namespace rq = stablercwa::ref_v1::detail::eigqr;

static void test_t11(Tally& t) {
    begin(t, "T11 reduce_to_hessenberg / accumulate_reflectors / eigenvectors_from_schur 单独与旧版比较");
    const double t0 = now_seconds();
    const int nm = g_cfg.nmin;
    std::vector<int> sizes{1, 2, 3, 4, 5, 7, 8, 9, 16, 31, 32, 33, 63, 64, 65, 100, 128, 200};
    sizes.push_back(nm - 1);
    sizes.push_back(nm);
    sizes.push_back(nm + 1);
    if (g_cfg.big)
        for (int n : {257, 300, 400, 512}) sizes.push_back(n);
    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
    // 小矩阵用全部线程数，大的抽稀成 {1, 3, 8}
    const auto threads_of = [](int n) {
        if (n <= 70) return g_cfg.threads;
        std::vector<int> out;
        for (int th : g_cfg.threads)
            if (th == 3 || th == 8) out.push_back(th);
        return out.empty() ? g_cfg.threads : out;
    };

    // 输入矩阵族
    const auto make_family = [&](int fam, int n) -> Matrix {
        const std::uint64_t s = 12000 + 31 * n + fam;
        switch (fam) {
            case 0: return random_complex(n, s);
            case 1: return random_real(n, s);
            case 2: return hermitian_indefinite(n, s);
            case 3: return block_upper(n, {0, n / 3, 2 * n / 3}, s);
            case 4: return tri_then_dense(n, n / 3, s);
            case 5: return hess_then_dense(n, n / 2, s);
            case 6: return antidiagonal(n, 0.0, s);
            case 7: return antidiagonal(n, 1e-3, s);
            case 8: return arrowhead(n, s);
            case 9: return Matrix(n, n);
            case 10: return rank_one(n, s);
            case 14: return dense_tri_dense(n, n / 3, n / 3, s);
            case 15: return interleaved_blocks(n, 4, s);
            case 11: return scaled_by_pow2(random_complex(n, s), 700);
            case 12: return scaled_by_pow2(random_complex(n, s), -700);
            default: return nilpotent_shift(n);
        }
    };
    const char* fam_names[16] = {"rand_c", "rand_r", "herm", "block", "tri_dense", "hess_dense", "antidiag", "antidiag+noise", "arrowhead", "zero", "rank1", "×2^700", "×2^-700", "nilpotent", "dense_tri_dense", "interleaved"};
    long long reduce_runs = 0, accum_runs = 0, back_runs = 0;
    for (int n : sizes)
        for (int fam = 0; fam < 16; ++fam) {
            if (n > 70 && fam != 0 && fam != 3 && fam != 4 && fam != 5 && fam != 7 && fam != 11 && fam != 14 && fam != 15) continue;      // 大尺寸只取几个族，控制用时
            const Matrix a = make_family(fam, n);
            const std::string name = fmt("%s n=%d", fam_names[fam], n);
            ++t.matrices;
            // reduce_to_hessenberg：新旧各从同一份拷贝出发，比较结果矩阵（Hessenberg 部分与存放反射向量的下三角）与 ort
            Matrix h_ref = a;
            Vector ort_ref(n);
            rq::reduce_to_hessenberg(h_ref, ort_ref);
            Matrix z_ref;
            {
                Vector ort_work = ort_ref;
                z_ref = rq::accumulate_reflectors(h_ref, ort_work);
            }
            for (int th : threads_of(n)) {
                set_num_threads(th);
                Matrix h = a;
                Vector ort(n);
                nq::reduce_to_hessenberg(h, ort);
                ++reduce_runs;
                t.runs += 1;
                ++t.comparisons;
                std::string why = cmp_mat("reduce 后的矩阵", h_ref, h, "旧版", "新版", t.bytes);
                if (why.empty()) why = cmp_vec("reduce 后的 ort", ort_ref, ort, "旧版", "新版", t.bytes);
                if (!why.empty()) fail_in(t, fmt("reduce_to_hessenberg %s 线程=%d：", name.c_str(), th) + why);
                // accumulate_reflectors：输入用旧版 reduce 的结果（const 矩阵与 ort 的拷贝）
                const Matrix h_in = h_ref;
                Vector ort_work = ort_ref;
                const Matrix z = nq::accumulate_reflectors(h_in, ort_work);
                ++accum_runs;
                t.runs += 1;
                ++t.comparisons;
                why = cmp_mat("accumulate 的 Z", z_ref, z, "旧版", "新版", t.bytes);
                if (why.empty()) why = cmp_mat("accumulate 不应改动的输入矩阵", h_ref, h_in, "调用前", "调用后", t.bytes);
                if (!why.empty()) fail_in(t, fmt("accumulate_reflectors %s 线程=%d：", name.c_str(), th) + why);
            }
        }
    // eigenvectors_from_schur：输入 (T, w, Z, eps)。一类来自旧版 eig_qr 的 keep_transform 结果，一类是人为构造的（逼出各个分支）
    struct SchurIn {
        std::string name;
        Matrix t;
        Vector w;
        Matrix z;
        double eps;
    };
    std::vector<SchurIn> ins;
    for (int n : sizes) {
        if (n > 260 || (n > 70 && n != 100 && n != 128 && n != nm && n != 200)) continue;
        for (int variant = 0; variant < 3; ++variant) {
            const Matrix a = variant == 0 ? random_complex(n, 13000 + n) : variant == 1 ? pq2d(n < 40 ? 2 : (n < 150 ? 3 : 5), 1.7, 1.7, 1.0, 1.0, 0.61, 0, 0, 2) : block_upper(n, {0, n / 2}, 13100 + n);
            if (variant == 1 && n != 33 && n != 100 && n != 200) continue;                // pq2d 的尺寸由 M 定，只取三个
            EigOptions o;
            o.keep_transform = true;
            o.vectors = false;
            if (variant == 2) o.eps = 1e-8;
            const EigResult r = run_ref(a, o);
            ins.push_back({fmt("来自 eig_qr %s n=%d", variant == 0 ? "rand_c" : variant == 1 ? "pq2d uni" : "block eps=1e-8", a.rows()), r.T, r.values, r.Z, o.eps});
        }
        Rng rng(14000 + n);
        for (int kind = 0; kind < 7; ++kind) {
            Matrix tm(n, n);
            Vector w(n);
            double tscale = 1.0;
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < j; ++i) tm(i, j) = kind == 6 ? cplx{} : rng.cuniform();
                const double u = rng.uniform();
                switch (kind) {
                    case 0: w[j] = cplx(2.0 * rng.uniform(), 2.0 * rng.uniform()); break;                 // 互异
                    case 1: w[j] = cplx(1.0 + j * 1e-17, 0.0); break;                                    // 相差在 eps 以内：替换成 rel·max|w|
                    case 2: w[j] = cplx(1.0 + 1e-9 * (j / 7), 0.5 * (j % 3)); break;                     // 成簇、一部分恰好相同（分母为零）
                    case 3: w[j] = cplx(j * 5e-324, 0.0); break;                                         // 次正规数的差：商会上溢，要先缩小向量
                    case 4: w[j] = cplx(1.0 + 1e-3 * j, 0.0); tscale = 1e12; break;                      // 大的 T：溢出控制
                    case 5: w[j] = cplx(5.0, 0.0) + u * 1e-14; break;                                    // 几乎重合
                    default: w[j] = cplx(j, -j); break;                                                  // T 的严格上三角为零
                }
                tm(j, j) = w[j];
            }
            if (tscale != 1.0)
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < j; ++i) tm(i, j) *= tscale;
            Matrix zm = kind == 5 ? Matrix::identity(n) : random_complex(n, 14100 + n + kind);
            ins.push_back({fmt("构造 kind=%d n=%d", kind, n), tm, w, zm, kind == 2 ? 1e-8 : kEps});
        }
    }
    for (const SchurIn& s : ins) {
        ++t.matrices;
        const Matrix v_ref = rq::eigenvectors_from_schur(s.t, s.w, s.z, s.eps);
        for (int th : threads_of(s.t.rows())) {
            set_num_threads(th);
            const Matrix v = nq::eigenvectors_from_schur(s.t, s.w, s.z, s.eps);
            ++back_runs;
            ++t.runs;
            ++t.comparisons;
            const std::string why = cmp_mat("本征向量矩阵", v_ref, v, "旧版", "新版", t.bytes);
            if (!why.empty()) fail_in(t, fmt("eigenvectors_from_schur %s 线程=%d：", s.name.c_str(), th) + why);
        }
    }
    set_num_threads(1);
    finish(t, t0, fmt("（reduce %lld 次、accumulate %lld 次、eigenvectors_from_schur %lld 次）", reduce_runs, accum_runs, back_runs).c_str());
}

// ---------------------------------------------------------------------------------------------
// T12 并行路径确实被执行（防“并行实现没有接上、或阈值太高，所有比较平凡地通过”）。
// 需要被测实现配合：parallel.hpp 提供 parallel_regions_started()（真正分给多个线程的并行区个数）并定义
// STABLERCWA_PARALLEL_REGION_COUNT。没有时只能看 CPU/墙钟（只是提示：负载高时不可靠）。
static void test_t12(Tally& t) {
    begin(t, "T12 并行路径确实被执行（n >= n_min 且线程数 > 1 时开了并行区；n < n_min 或 1 线程时没有）");
    const double t0 = now_seconds();
    const int nm = g_cfg.nmin;
    const Matrix big_a = random_complex(nm + 8, 12001), small_a = random_complex(nm - 1, 12002);
    t.matrices += 2;
#if defined(STABLERCWA_PARALLEL_REGION_COUNT)
    const auto regions = [&](const Matrix& a, int th, const EigOptions& o) {
        set_num_threads(th);
        const long long r0 = parallel_regions_started();
        run_new(a, o);
        ++t.runs;
        return parallel_regions_started() - r0;
    };
    for (OptId id : {kDefault, kKeepNoVec, kNoVec, kObserver}) {
        long long calls = 0;
        const EigOptions o = make_options(kOptions[id], &calls);
        const long long on = regions(big_a, 4, o), off1 = regions(big_a, 1, o), off_small = regions(small_a, 8, o);
        check(on > 0 || id == kNoVec, fmt("选项=%s：n = %d、4 线程，没有开任何并行区（并行版没有接上？）", kOptions[id].name, nm + 8));
        check(off1 == 0, fmt("选项=%s：1 线程却开了 %lld 个并行区", kOptions[id].name, off1));
        check(off_small == 0, fmt("选项=%s：n = %d < n_min，8 线程，却开了 %lld 个并行区（阈值与 --nmin 不一致？）", kOptions[id].name, nm - 1, off_small));
        say("  选项 %-14s n=%d 4 线程：并行区 %lld 个\n", kOptions[id].name, nm + 8, on);
    }
#else
    {
        const Matrix a = random_complex(400, 12003);
        EigOptions o = make_options(kOptions[kKeep], nullptr);
        set_num_threads((std::min)(8, (std::max)(2, hardware_threads())));
        const double w0 = now_seconds(), c0 = cpu_seconds();
        run_new(a, o);
        ++t.runs;
        const double ratio = (cpu_seconds() - c0) / (std::max)(1e-9, now_seconds() - w0);
        say("  注意：被测实现没有提供并行区计数（STABLERCWA_PARALLEL_REGION_COUNT），无法确认并行路径被执行；n=400、%d 线程的 CPU/墙钟 = %.2f\n",
            num_threads(), ratio);
        if (g_cfg.require_parallel) check(ratio > 1.3, fmt("--require-parallel：CPU/墙钟 = %.2f，看起来没有并行", ratio));
    }
#endif
    (void)small_a;
    set_num_threads(1);
    finish(t, t0);
}

// ---------------------------------------------------------------------------------------------
// --coverage：目录里每个 n >= 100 的矩阵，旧版 eig_qr（默认选项）的迭代窗口统计：sweep 数、l > 0 的 sweep 数、窗口向上扩张次数。
// 只是帮着挑测试矩阵用的（窗口向上扩张是并行化里最容易出错的情形），不是测试。
struct WindowStats {
    long long sweeps = 0, l_positive = 0, expansions = 0, sweeps_wide = 0;
};
static WindowStats window_stats(const Matrix& a) {
    WindowStats st;
    const int n = a.rows();
    int prev_en = -2, prev_l = -2;
    EigOptions o;
    o.observer = [&](const EigProgress& p) {
        if (std::string(p.stage) == "done") return;
        int en = p.active, l = en;
        while (en >= 0) {
            for (l = en; l > 0; --l)
                if (abs1(p.H(l, l - 1)) <= kEps * (abs1(p.H(l - 1, l - 1)) + abs1(p.H(l, l)))) break;
            if (l != en) break;
            --en;
        }
        if (en < 1) return;
        ++st.sweeps;
        if (l > 0) ++st.l_positive;
        if (en - l + 1 > n / 2) ++st.sweeps_wide;
        if (en == prev_en && l < prev_l) ++st.expansions;
        prev_en = en;
        prev_l = l;
    };
    run_ref(a, o);
    return st;
}
static void coverage_report() {
    say("[窗口统计：n >= 100 的矩阵，旧版 eig_qr 默认选项]\n  %-34s %5s %7s %9s %10s %6s\n", "矩阵", "n", "sweeps", "l>0", "窗口>n/2", "向上扩张");
    for (const MatCase& mc : g_catalogue) {
        if (mc.n < 100) continue;
        const WindowStats st = window_stats(mc.a);
        say("  %-34s %5d %7lld %9lld %10lld %6lld\n", mc.name.c_str(), mc.n, st.sweeps, st.l_positive, st.sweeps_wide, st.expansions);
    }
}

// ---------------------------------------------------------------------------------------------
// 命令行与入口
static std::vector<int> parse_ints(const std::string& s) {
    std::vector<int> out;
    std::size_t k = 0;
    while (k < s.size()) {
        std::size_t e = s.find(',', k);
        if (e == std::string::npos) e = s.size();
        const int v = std::atoi(s.substr(k, e - k).c_str());
        if (v >= 1) out.push_back(v);
        k = e + 1;
    }
    return out;
}
static bool selected(const char* id) {
    if (g_cfg.only.empty()) return true;
    if (g_cfg.only.count(id)) return true;
    return std::string(id) == "T1" && g_cfg.only.count("T2") > 0;
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--big") g_cfg.big = true;
        else if (a == "--only") {
            const std::string list = next();
            std::size_t k = 0;
            while (k < list.size()) {
                std::size_t e = list.find(',', k);
                if (e == std::string::npos) e = list.size();
                g_cfg.only.insert(list.substr(k, e - k));
                k = e + 1;
            }
        } else if (a == "--dump") g_cfg.dump = next();
        else if (a == "--nmin") g_cfg.nmin = (std::max)(8, std::atoi(next().c_str()));
        else if (a == "--threads") {
            g_cfg.threads = parse_ints(next());
            if (g_cfg.threads.empty()) g_cfg.threads = {1, 2, 3, 4, 7, 8};
        } else if (a == "--timeout") g_cfg.timeout = std::atof(next().c_str());
        else if (a == "--times") g_cfg.times = true;
        else if (a == "--coverage") g_cfg.coverage = true;
        else if (a == "--filter") g_cfg.filter = next();
        else if (a == "--suite-timeout") g_cfg.suite_timeout = std::atof(next().c_str());
        else if (a == "--require-parallel") g_cfg.require_parallel = true;
        else {
            say("未知选项 %s\n用法：test_eig_qr_parallel [--big] [--only T1,T3] [--dump DIR] [--nmin K] [--threads 1,2,8] [--timeout SEC]\n", a.c_str());
            return 2;
        }
    }

    // 看门狗：把死锁变成失败
    std::mutex wd_mutex;
    std::condition_variable wd_cv;
    bool wd_done = false;
    const double timeout = g_cfg.timeout;
    const double wd_start = now_seconds();
    std::thread watchdog([&] {   // 总时限 --timeout，每个套件另有 --suite-timeout：一次死锁不必等满 15 分钟
        std::unique_lock<std::mutex> lock(wd_mutex);
        while (!wd_cv.wait_for(lock, std::chrono::seconds(1), [&] { return wd_done; })) {
            const double now = now_seconds(), st = g_stage_start.load();
            if (now - wd_start > timeout || (st > 0.0 && now - st > g_cfg.suite_timeout)) {
                say("\n超时：当前在 [%s]（本套件 %.0f 秒，总计 %.0f 秒），疑似死锁或严重变慢，强制退出\n", g_stage.load(), now - st, now - wd_start);   // 经 say()：MSVC 不去按代码页解析中文格式串
                std::_Exit(3);
            }
        }
    });

    say("test_eig_qr_parallel：n_min = %d，线程数 {", g_cfg.nmin);
    for (std::size_t k = 0; k < g_cfg.threads.size(); ++k) say("%s%d", k ? "," : "", g_cfg.threads[k]);
    say("}，硬件线程 %d%s%s\n", hardware_threads(), g_cfg.big ? "，--big" : "", g_cfg.dump.empty() ? "" : "，含真实转储");
    set_num_threads(1);                                      // 构造矩阵时（含 P·Q 的矩阵乘）不用线程池
    g_stage = "构造矩阵";
    g_catalogue = catalogue();
    {
        int largest = 0;
        for (const MatCase& mc : g_catalogue) largest = (std::max)(largest, mc.n);
        say("矩阵目录：%d 个（最大 n = %d）\n", static_cast<int>(g_catalogue.size()), largest);
    }

    const auto stop_watchdog = [&] {
        {
            std::lock_guard<std::mutex> lock(wd_mutex);
            wd_done = true;
        }
        wd_cv.notify_all();
        watchdog.join();
    };
    if (g_cfg.coverage) {
        coverage_report();
        stop_watchdog();
        return 0;
    }
    g_t2.name = "T2";
    Tally t1, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12;
    t1.name = "T1";
    t3.name = "T3";
    t4.name = "T4";
    t5.name = "T5";
    t6.name = "T6";
    t7.name = "T7";
    t8.name = "T8";
    t9.name = "T9";
    t10.name = "T10";
    t11.name = "T11";
    t12.name = "T12";
    const double t_start = now_seconds();
    std::vector<Tally*> ran;
    const auto run = [&](const char* id, Tally& t, void (*fn)(Tally&)) {
        if (!selected(id)) return;
        fn(t);
        ran.push_back(&t);
    };
    run("T1", t1, test_t1);
    run("T3", t3, test_t3);
    run("T4", t4, test_t4);
    run("T5", t5, test_t5);
    run("T6", t6, test_t6);
    run("T7", t7, test_t7);
    run("T8", t8, test_t8);
    run("T9", t9, test_t9);
    run("T10", t10, test_t10);
    run("T11", t11, test_t11);
    run("T12", t12, test_t12);

    long long failures = g_t2.failures;
    say("\n%-5s %8s %9s %10s %12s %8s %6s %8s %8s\n", "测试", "矩阵/输入", "运行", "比较次数", "字节", "其它断言", "失败", "墙钟(s)", "CPU(s)");
    double cpu_total = 0;
    for (const Tally* t : ran) {
        say("%-5s %8lld %9lld %10lld %12lld %8lld %6lld %8.1f %8.1f\n", t->name.c_str(), t->matrices, t->runs, t->comparisons, t->bytes, t->checks, t->failures, t->seconds, t->cpu);
        failures += t->failures;
        cpu_total += t->cpu;
    }
    if (g_t2.comparisons > 0)
        say("%-5s %8s %9s %10lld %12lld %8s %6lld %8s   （线程数之间的比较，含在上面各测试的运行里）\n", "T2", "-", "-", g_t2.comparisons, g_t2.bytes, "-", g_t2.failures, "-");
    say("总用时 %.1f s（CPU %.1f s）；%s\n", now_seconds() - t_start, cpu_total, failures == 0 ? "全部通过" : "有失败");
    {
        const double unit = 1250.0 * 1250.0 * 1250.0;
        say("工作量：旧版 %.2f、新版 %.2f 个“2N=1250 完整分解”（单线程各约 9 s，M1 Pro 空闲时）；新版若是串行实现，空闲机器上总计约 %.0f s\n",
            static_cast<double>(g_work_ref.load()) / unit, static_cast<double>(g_work_new.load()) / unit,
            9.3 * static_cast<double>(g_work_ref.load() + g_work_new.load()) / unit);
    }
    stop_watchdog();
    return failures == 0 ? 0 : 1;
}
