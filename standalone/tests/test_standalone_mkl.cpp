// standalone/tests/test_standalone_mkl.cpp —— MatrixEigenValueAndEigenVector_MKL.cpp 的验收测试
//
// 要有 oneMKL 2025.0.1 或更新（Windows：先运行 standalone/install_mkl.bat）。在仓库根目录：
//   cmake -S . -B build -DSTABLERCWA_BUILD_STANDALONE_MKL=ON
//   cmake --build build --config Release --target test_standalone_mkl
//   build\Release\test_standalone_mkl.exe      （或 cd build，再 ctest -C Release -R test_standalone_mkl）
//
// 检查的几组（MKL 的结果与 StableRCWA 的 eig_qr 不逐位相同，所以只查残差、排序、相位和打包格式）：
//   K  内核：每个随机矩阵都收敛，每个本征对的残差在 64·n·eps 以内
//   C  复数版：按 (Re 降, Im 降) 排序；模最大的分量是正实数，2-范数为 1，残差小
//   R  实数版：实本征值的虚部恰为 0；复本征值成对、相邻、严格共轭、虚部为正的在前；以单位计的次序单调；
//      由打包格式还原的向量 2-范数为 1、模最大的分量是正实数、残差小；与复数版的本征值一一对应；
//      还原出的 n 个复向量线性无关：cond(V_实数版) <= 1e3·cond(V_复数版)
//   E  返回码与 NaN 填充、输出与 A1 共用存储、多个线程同时调用
// 本文件是从开发仓库里同时测几种实现（QR、LR、MKL 等）的测试程序裁出来的，只保留 MKL 的部分；个别注释和常量
// （kIsQR、kBitwiseVsLibrary 等）还提到别的实现，在这里它们都取“否”。
// 退出状态：全部通过 0，有失败 1。
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <complex>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "stablercwa/eig_lr.hpp"
#include "stablercwa/eig_qr.hpp"
#include "stablercwa/lu.hpp"
#include "stablercwa/svd.hpp"
#include "../MatrixEigenValueAndEigenVector_MKL.cpp"

namespace ss = stablercwa;

// 文字一律经 say() 输出（用法同 std::printf）。不把带中文的格式串直接交给 printf：MSVC 检查 printf 的格式串时按系统
// 代码页（中文 Windows 上是 GBK）读它，会误报 C4819、C4477、C4313（与 tests/test_eig_qr.cpp 的做法相同）。
#if defined(__GNUC__) && !defined(__MINGW32__)
[[gnu::format(printf, 1, 2)]]
#endif
static void say(const char* format, ...) {
    std::va_list args;
    va_start(args, format);
    std::vprintf(format, args);
    va_end(args);
}
namespace me = matrix_eig;
using cplx = std::complex<double>;

// 被测文件用的迭代，以及库里对应的实现。
//   kIsQR：只对 QR 版成立的检查（多线程、近实共轭对样例走到拆分分支、二重亏损根的实根个数）；
//   kBitwiseVsLibrary：内核与库里对应的实现逐位相同（QR、LR）；
//   kBackwardStable：残差按绝对容差查（QR、MKL、GPU；LR 不是后向稳定的，只要求实数版不比复数版差太多）。
static const char* const kMethod = "MKL";
static const bool kIsQR = false;
static const bool kBitwiseVsLibrary = false;
static const bool kBackwardStable = true;
static const int kKernelParallelMinN = std::numeric_limits<int>::max();   // 线程由 MKL 自己管
static int kernel_eig(const me::Matrix& a, me::Vector& w, me::Matrix& v, int) { return me::mkl::eig(a, w, v); }
static ss::EigResult library_eig(const ss::Matrix& a, const ss::EigOptions& o) { return ss::eig_qr(a, o); }

static int default_threads_for_print() { return 1; }

static int g_checks = 0, g_failures = 0;
static double g_cond_ratio = 0.0;   // 实数版与复数版向量组条件数之比的最大值（复数版 cond <= 1e12 的矩阵）

// 科学计数法（std::to_string 用的是 %f，小数会显示成 0.000000）
static std::string sci(double x) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.2e", x);
    return buf;
}

static void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        say("  失败：%s\n", what.c_str());
    }
}

// n 个行指针的二维数组，元素连续存放
template <class T>
struct Rows {
    std::vector<T> data;
    std::vector<T*> rows;
    explicit Rows(int n, T fill = T()) : data(static_cast<std::size_t>(n) * n, fill), rows(n) {
        for (int i = 0; i < n; ++i) rows[i] = data.data() + static_cast<std::size_t>(i) * n;
    }
    T** ptr() { return rows.data(); }
};

static bool same_bytes(const void* a, const void* b, std::size_t bytes) { return std::memcmp(a, b, bytes) == 0; }

static ss::Matrix random_complex(int n, std::mt19937_64& g) {
    std::normal_distribution<double> N(0.0, 1.0);
    ss::Matrix a(n, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) a(i, j) = cplx(N(g), N(g));
    return a;
}

static ss::Matrix random_real(int n, std::mt19937_64& g) {
    std::normal_distribution<double> N(0.0, 1.0);
    ss::Matrix a(n, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) a(i, j) = N(g);
    return a;
}

// 相对残差 ||A v − λ v|| / (||A||_F ||v||)；A = 0 时用 1 代替 ||A||_F。
// A、λ 先一起乘一个 2 的幂（使最大元素落在 1 附近），元素是 1e±200 时平方也不会上溢、下溢。
static double residual(const ss::Matrix& a, cplx lambda, const std::vector<cplx>& v) {
    const int n = a.rows();
    double big = 0.0, vbig = 0.0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) big = (std::max)(big, std::abs(a(i, j)));
    for (int i = 0; i < n; ++i) vbig = (std::max)(vbig, std::abs(v[i]));
    int e = 0, f = 0;
    if (big > 0.0) std::frexp(big, &e);
    if (vbig > 0.0) std::frexp(vbig, &f);
    const auto s2 = [](cplx z, int k) { return cplx(std::ldexp(z.real(), k), std::ldexp(z.imag(), k)); };
    double anorm = 0.0, dn = 0.0, vn = 0.0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) anorm += std::norm(s2(a(i, j), -e));
    anorm = anorm == 0.0 ? 1.0 : std::sqrt(anorm);
    const cplx ls = s2(lambda, -e);
    for (int i = 0; i < n; ++i) {
        cplx s = 0.0;
        for (int k = 0; k < n; ++k) s += s2(a(i, k), -e) * s2(v[k], -f);
        dn += std::norm(s - ls * s2(v[i], -f));
        vn += std::norm(s2(v[i], -f));
    }
    return std::sqrt(dn) / anorm / std::sqrt(vn);
}

// 2-范数为 1；有一个虚部恰为 0、实部为正的分量，它的模与最大的模相差不超过几个 ulp
// （相位因子的模只在舍入意义下为 1，几个分量的模在舍入以内相等时，别的分量可能比它大一点）。
// 实向量（虚部全为 0）要求更严：模最大的分量（并列取下标小的）本身为正。
static bool normalized(const std::vector<cplx>& v) {
    double ssq = 0.0, big = 0.0;
    bool real = true;
    int k = 0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        ssq += std::norm(v[i]);
        if (std::abs(v[i]) > big) {
            big = std::abs(v[i]);
            k = static_cast<int>(i);
        }
        real = real && v[i].imag() == 0.0;
    }
    if (!(std::abs(std::sqrt(ssq) - 1.0) < 1e-13)) return false;
    if (real) return v[k].real() > 0.0;
    for (const cplx& z : v)
        if (z.imag() == 0.0 && z.real() > 0.0 && z.real() >= big * (1.0 - 8.0 * std::numeric_limits<double>::epsilon())) return true;
    return false;
}

// ---------------------------------------------------------------------------------------------
// K：内核逐位
static void test_kernels(const std::string& name, const ss::Matrix& A) {
    const int n = A.rows();
    me::Matrix a(n, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) a(i, j) = A(i, j);
    const std::size_t nn = static_cast<std::size_t>(n) * n;

    if (!kBitwiseVsLibrary) {                         // MKL、GPU：只查收敛与每个本征对的残差
        me::Vector w;
        me::Matrix v;
        const int info = kernel_eig(a, w, v, 1);
        check(info == 0, name + " " + kMethod + " 收敛（info " + std::to_string(info) + "）");
        if (info != 0) return;
        const double eps = std::numeric_limits<double>::epsilon();
        double worst = 0.0;
        for (int k = 0; k < n; ++k) worst = (std::max)(worst, residual(A, w[k], std::vector<cplx>(v.col(k), v.col(k) + n)));
        check(worst <= 64.0 * (std::max)(n, 1) * eps, name + " " + kMethod + " 内核残差 " + sci(worst));
        return;
    }
    // 独立文件的内核：达到并行门槛时在几种线程数下各算一遍（每一遍都要与库逐字节相同，也就互相相同）
    std::vector<int> mine(1, 1);
    if (n >= kKernelParallelMinN) mine = {1, 2, 3, 4, 7, 8};
    std::vector<int> infos(mine.size());
    std::vector<me::Vector> ws(mine.size());
    std::vector<me::Matrix> vs(mine.size());
    for (std::size_t k = 0; k < mine.size(); ++k) infos[k] = kernel_eig(a, ws[k], vs[k], mine[k]);
    // 与库比两次：库的迭代上限设成与独立文件相同时 info 与结果都逐位相同；库用默认上限（30）且收敛时结果也逐位相同。
    ss::EigOptions same_limit;
    same_limit.max_iter = 100;   // 只在与库逐位比较的分支里用（MKL 版不走）
    ss::EigOptions library_default;
    const std::string method = kMethod;
    for (int threads : {1, 4}) {                      // 库的 eig_qr 在 n >= 192 时按线程数分块；eig_lr 的结果与线程数无关
        ss::set_num_threads(threads);
        for (const ss::EigOptions* o : {&same_limit, &library_default}) {
            const ss::EigResult r = library_eig(A, *o);
            for (std::size_t k = 0; k < mine.size(); ++k) {
                const std::string tag = std::string(o == &same_limit ? "同上限" : "库默认上限") + "，库线程 " + std::to_string(threads) +
                                        "，独立文件线程 " + std::to_string(mine[k]);
                if (o == &same_limit) check(infos[k] == r.info, name + " " + method + " info（" + tag + "）");
                if (infos[k] == 0 && r.info == 0) {
                    check(same_bytes(ws[k].data(), r.values.data(), n * sizeof(cplx)), name + " " + method + " 本征值逐位（" + tag + "）");
                    check(same_bytes(vs[k].col(0), r.vectors.data(), nn * sizeof(cplx)), name + " " + method + " 本征向量逐位（" + tag + "）");
                }
            }
        }
    }
    ss::set_num_threads(1);
}

// ---------------------------------------------------------------------------------------------
// C：复数版
struct ComplexOut {
    int status;
    std::vector<cplx> values;
    Rows<cplx> vectors;
    explicit ComplexOut(int n) : status(0), values(n), vectors(n) {}
};

static ComplexOut call_complex(const ss::Matrix& A) {
    const int n = A.rows();
    Rows<cplx> a(n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) a.rows[i][j] = A(i, j);
    ComplexOut out(n);
    out.status = MatrixEigenValueAndEigenVector(a.ptr(), out.values.data(), out.vectors.ptr(), n);
    return out;
}

static double test_complex(const std::string& name, const ss::Matrix& A, double tol) {
    const int n = A.rows();
    ComplexOut out = call_complex(A);
    check(out.status == 0, name + " 复数版返回 0（实际 " + std::to_string(out.status) + "）");
    if (out.status != 0) return 0.0;

    // 期望：库的结果，向量经同样的相位规范，按同样的规则稳定排序。只有逐位比较时才要（QR、LR 版）；
    // MKL、GPU 版不调库，库在某个矩阵上不收敛（不给本征向量）也不影响它们的检查。
    if (kBitwiseVsLibrary) {
        ss::EigOptions same_limit;
        same_limit.max_iter = 100;   // 只在与库逐位比较的分支里用（MKL 版不走）
        ss::EigResult r = library_eig(A, same_limit);
        check(r.info == 0, name + " 库在同样的迭代上限下收敛（info " + std::to_string(r.info) + "）");
        if (r.info != 0) return 0.0;
        std::vector<std::pair<cplx, int> > order(n);
        for (int j = 0; j < n; ++j) {
            me::make_largest_real(r.vectors.col(j), n);
            order[j] = std::make_pair(r.values[j], j);
        }
        std::stable_sort(order.begin(), order.end(), me::value_before);
        bool values_same = true, vectors_same = true;
        for (int k = 0; k < n; ++k) {
            if (!same_bytes(&out.values[k], &order[k].first, sizeof(cplx))) values_same = false;
            for (int i = 0; i < n; ++i)
                if (!same_bytes(&out.vectors.rows[i][k], &r.vectors(i, order[k].second), sizeof(cplx))) vectors_same = false;
        }
        check(values_same, name + " 复数版本征值 = 库的结果排序后（逐位）");
        check(vectors_same, name + " 复数版本征向量 = 库的结果规范相位、排序后（逐位）");
    }

    double worst = 0.0;
    bool sorted = true, norm_ok = true;
    for (int k = 0; k < n; ++k) {
        if (k + 1 < n && me::sorts_before(out.values[k + 1], out.values[k])) sorted = false;
        std::vector<cplx> v(n);
        for (int i = 0; i < n; ++i) v[i] = out.vectors.rows[i][k];
        if (!normalized(v)) norm_ok = false;
        worst = (std::max)(worst, residual(A, out.values[k], v));
    }
    check(sorted, name + " 复数版按 (Re 降, Im 降) 排序");
    check(norm_ok, name + " 复数版向量 2-范数为 1、最大分量为正实数");
    if (kBackwardStable) check(worst <= tol, name + " 复数版残差 " + sci(worst) + " <= " + sci(tol));
    return worst;
}

// ---------------------------------------------------------------------------------------------
// R：实数版
struct RealOut {
    int status;
    std::vector<cplx> values;
    Rows<double> vectors;
    explicit RealOut(int n) : status(0), values(n), vectors(n) {}
};

static RealOut call_real(const ss::Matrix& A) {
    const int n = A.rows();
    Rows<double> a(n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) a.rows[i][j] = A(i, j).real();
    RealOut out(n);
    out.status = MatrixEigenValueAndEigenVector(a.ptr(), out.values.data(), out.vectors.ptr(), n);
    return out;
}

// tol：残差上限；match：与复数版本征值的最大距离（相对 ||A||_F）
// check_cond = false 时不比较向量组条件数（只用于已知的局限，见独立文件开头）
static double test_real(const std::string& name, const ss::Matrix& A, double tol, double match, int* n_real = nullptr,
                        bool check_cond = true) {
    const int n = A.rows();
    RealOut out = call_real(A);
    check(out.status == 0, name + " 实数版返回 0（实际 " + std::to_string(out.status) + "）");
    if (out.status != 0) return 0.0;

    double worst = 0.0;
    std::vector<std::pair<cplx, double> > unit_res;   // 每个单位（实本征值，或共轭对里虚部为正的那个）的本征值与残差
    bool structure = true, sorted = true, norm_ok = true;
    int reals = 0;
    cplx prev_key;
    bool has_prev = false;
    for (int k = 0; k < n;) {
        const cplx lambda = out.values[k];
        std::vector<cplx> v(n);
        int width;
        if (lambda.imag() == 0.0) {
            if (std::signbit(lambda.imag())) structure = false;      // 不要 −0
            for (int i = 0; i < n; ++i) v[i] = out.vectors.rows[i][k];
            width = 1;
            ++reals;
        } else {
            if (!(lambda.imag() > 0.0) || k + 1 >= n || out.values[k + 1].real() != lambda.real() ||
                out.values[k + 1].imag() != -lambda.imag()) {
                structure = false;
                break;
            }
            for (int i = 0; i < n; ++i) v[i] = cplx(out.vectors.rows[i][k], out.vectors.rows[i][k + 1]);
            width = 2;
        }
        if (has_prev && me::sorts_before(lambda, prev_key)) sorted = false;
        prev_key = lambda;
        has_prev = true;
        if (!normalized(v)) norm_ok = false;
        const double r = residual(A, lambda, v);       // 共轭的那一个残差相同（A 是实矩阵）
        worst = (std::max)(worst, r);
        unit_res.push_back(std::make_pair(lambda, r));
        k += width;
    }
    check(structure, name + " 实数版：实本征值虚部为 +0，复本征值成对相邻、严格共轭、正虚部在前");
    check(sorted, name + " 实数版以单位计按 (Re 降, Im 降) 排序");
    check(norm_ok, name + " 实数版向量 2-范数为 1、最大分量为正实数");
    // QR 版按绝对容差查残差。LR 版不是后向稳定的，复数版本身的残差就可以很大，所以逐个单位地比：
    // 实数版每个本征对的残差不超过 max(tol, 4 × 离它最近的复数版本征对的残差)——取实部、配对不该让任何一对的残差明显变大。
    ComplexOut c = call_complex(A);
    if (kBackwardStable) {
        check(worst <= tol, name + " 实数版残差 " + sci(worst) + " <= " + sci(tol));
    } else if (c.status == 0) {
        std::vector<double> res_c(n);
        for (int k = 0; k < n; ++k) {
            std::vector<cplx> vk(n);
            for (int i = 0; i < n; ++i) vk[i] = c.vectors.rows[i][k];
            res_c[k] = residual(A, c.values[k], vk);
        }
        double worst_excess = 0.0;                    // max(残差 / 上限)，> 1 即不合格
        for (const auto& u : unit_res) {
            int nearest = 0;
            for (int j = 1; j < n; ++j)
                if (std::abs(u.first - c.values[j]) < std::abs(u.first - c.values[nearest])) nearest = j;
            worst_excess = (std::max)(worst_excess, u.second / (std::max)(tol, 4.0 * res_c[nearest]));
        }
        check(worst_excess <= 1.0, name + " 实数版每个本征对的残差 <= max(tol, 4 × 最近的复数版本征对残差)（最差为上限的 " +
                                       sci(worst_excess) + " 倍）");
    }
    if (n_real) *n_real = reals;

    // 与复数版对同一矩阵的结果一一对应（贪心取最近的未用者）；还原出的向量组与复数版的一样好（条件数）
    if (c.status == 0 && structure) {
        ss::Matrix vr(n, n), vc(n, n);
        for (int k = 0; k < n;) {
            if (out.values[k].imag() == 0.0) {
                for (int i = 0; i < n; ++i) vr(i, k) = out.vectors.rows[i][k];
                k += 1;
            } else {
                for (int i = 0; i < n; ++i) {
                    vr(i, k) = cplx(out.vectors.rows[i][k], out.vectors.rows[i][k + 1]);
                    vr(i, k + 1) = cplx(out.vectors.rows[i][k], -out.vectors.rows[i][k + 1]);
                }
                k += 2;
            }
        }
        for (int k = 0; k < n; ++k)
            for (int i = 0; i < n; ++i) vc(i, k) = c.vectors.rows[i][k];
        const double cond_r = ss::cond2(vr), cond_c = ss::cond2(vc);
        if (check_cond && cond_c <= 1e12) g_cond_ratio = (std::max)(g_cond_ratio, cond_r / cond_c);
        check(!check_cond || cond_c > 1e12 || cond_r <= 1e3 * cond_c,
              name + " 实数版向量组 cond " + sci(cond_r) + " <= 1e3 × 复数版 " + sci(cond_c));
    }
    if (c.status == 0) {
        double anorm = ss::norm_fro(A);
        if (anorm == 0.0) anorm = 1.0;
        std::vector<bool> used(n, false);
        double worst_gap = 0.0;
        for (int k = 0; k < n; ++k) {
            int best = -1;
            double bd = std::numeric_limits<double>::infinity();
            for (int j = 0; j < n; ++j)
                if (!used[j] && std::abs(out.values[k] - c.values[j]) < bd) {
                    bd = std::abs(out.values[k] - c.values[j]);
                    best = j;
                }
            used[best] = true;
            worst_gap = (std::max)(worst_gap, bd / anorm);
        }
        check(worst_gap <= match, name + " 实数版本征值与复数版相差 " + sci(worst_gap) + " <= " + sci(match));
    }
    return worst;
}

// ---------------------------------------------------------------------------------------------
// E：返回码、共用存储、并发
static void test_errors() {
    const int n = 3;
    const double sentinel = 12345.0;
    Rows<double> a(n, 1.0), v(n, sentinel);
    std::vector<cplx> w(n, cplx(sentinel, sentinel));
    check(MatrixEigenValueAndEigenVector(a.ptr(), w.data(), v.ptr(), 0) == -1, "mn = 0 返回 -1");
    check(MatrixEigenValueAndEigenVector(static_cast<double**>(nullptr), w.data(), v.ptr(), n) == -1, "A1 为空返回 -1");
    check(MatrixEigenValueAndEigenVector(a.ptr(), nullptr, v.ptr(), n) == -1, "EigenValue 为空返回 -1");
    double* saved = v.rows[1];
    v.rows[1] = nullptr;
    check(MatrixEigenValueAndEigenVector(a.ptr(), w.data(), v.ptr(), n) == -1, "EigenVector 某行为空返回 -1");
    v.rows[1] = saved;
    bool untouched = true;
    for (double x : v.data) untouched = untouched && x == sentinel;
    for (cplx x : w) untouched = untouched && x == cplx(sentinel, sentinel);
    check(untouched, "返回 -1 时不碰输出");

    a.rows[2][1] = std::numeric_limits<double>::quiet_NaN();
    check(MatrixEigenValueAndEigenVector(a.ptr(), w.data(), v.ptr(), n) == -2, "实数版输入含 NaN 返回 -2");
    bool all_nan = true;
    for (double x : v.data) all_nan = all_nan && std::isnan(x);
    for (cplx x : w) all_nan = all_nan && std::isnan(x.real()) && std::isnan(x.imag());
    check(all_nan, "返回 -2 时输出全为 NaN");
    a.rows[2][1] = std::numeric_limits<double>::infinity();
    check(MatrixEigenValueAndEigenVector(a.ptr(), w.data(), v.ptr(), n) == -2, "实数版输入含 Inf 返回 -2");

    Rows<cplx> ca(n, cplx(1.0, 0.5)), cv(n);
    ca.rows[0][2] = cplx(0.0, std::numeric_limits<double>::infinity());
    check(MatrixEigenValueAndEigenVector(ca.ptr(), w.data(), cv.ptr(), n) == -2, "复数版输入含 Inf 返回 -2");
    check(MatrixEigenValueAndEigenVector(ca.ptr(), w.data(), cv.ptr(), -5) == -1, "复数版 mn < 0 返回 -1");
}

static void test_aliasing(const ss::Matrix& A) {
    const int n = A.rows();
    {
        RealOut ref = call_real(A);
        Rows<double> a(n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) a.rows[i][j] = A(i, j).real();
        std::vector<cplx> w(n);
        const int status = MatrixEigenValueAndEigenVector(a.ptr(), w.data(), a.ptr(), n);   // 输出写回 A1
        check(status == 0 && same_bytes(w.data(), ref.values.data(), n * sizeof(cplx)) &&
                  same_bytes(a.data.data(), ref.vectors.data.data(), a.data.size() * sizeof(double)),
              "实数版输出与 A1 共用存储，结果不变");
    }
    {
        ComplexOut ref = call_complex(A);
        Rows<cplx> a(n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) a.rows[i][j] = A(i, j);
        std::vector<cplx> w(n);
        const int status = MatrixEigenValueAndEigenVector(a.ptr(), w.data(), a.ptr(), n);
        check(status == 0 && same_bytes(w.data(), ref.values.data(), n * sizeof(cplx)) &&
                  same_bytes(a.data.data(), ref.vectors.data.data(), a.data.size() * sizeof(cplx)),
              "复数版输出与 A1 共用存储，结果不变");
    }
}

static void test_concurrency() {
    std::mt19937_64 g(99);
    std::vector<ss::Matrix> mats;
    for (int t = 0; t < 4; ++t) mats.push_back(t % 2 ? random_real(40 + 7 * t, g) : random_complex(40 + 7 * t, g));
    std::vector<ComplexOut> ref_c;
    std::vector<RealOut> ref_r;
    for (const auto& m : mats) {
        ref_c.push_back(call_complex(m));
        ref_r.push_back(call_real(m));
    }
    std::vector<int> ok(mats.size(), 1);
    std::vector<std::thread> threads;
    for (std::size_t t = 0; t < mats.size(); ++t)
        threads.emplace_back([&, t] {
            const std::size_t n = static_cast<std::size_t>(mats[t].rows());
            for (int rep = 0; rep < 10; ++rep) {
                ComplexOut c = call_complex(mats[t]);
                RealOut r = call_real(mats[t]);
                if (!kBitwiseVsLibrary) {                 // MKL、GPU：只查返回值、复数版的残差、实数版的本征值与参考相近
                    const double eps = std::numeric_limits<double>::epsilon();
                    bool good = c.status == 0 && r.status == 0;
                    for (std::size_t k = 0; good && k < n; ++k) {
                        std::vector<cplx> v(n);
                        for (std::size_t i = 0; i < n; ++i) v[i] = c.vectors.rows[i][k];
                        good = residual(mats[t], c.values[k], v) <= 64.0 * n * eps &&
                               std::abs(r.values[k] - ref_r[t].values[k]) <= 1e-10 * (1.0 + std::abs(ref_r[t].values[k]));
                    }
                    if (!good) ok[t] = 0;
                    continue;
                }
                if (!same_bytes(c.values.data(), ref_c[t].values.data(), n * sizeof(cplx)) ||
                    !same_bytes(c.vectors.data.data(), ref_c[t].vectors.data.data(), n * n * sizeof(cplx)) ||
                    !same_bytes(r.values.data(), ref_r[t].values.data(), n * sizeof(cplx)) ||
                    !same_bytes(r.vectors.data.data(), ref_r[t].vectors.data.data(), n * n * sizeof(double)))
                    ok[t] = 0;
            }
        });
    for (auto& th : threads) th.join();
    bool all = true;
    for (int x : ok) all = all && x;
    check(all, kBitwiseVsLibrary ? "4 个线程同时调用，结果与单独调用逐位相同" : std::string("4 个线程同时调用，结果都正常（") + kMethod + " 不保证逐位相同，只比残差）");
}

// S·(2I₃ ⊕ [[.5,3],[-3,.5]] ⊕ -1)·S⁻¹ 里的 8 个矩阵（2 万个随机 S 中挑出，按行，%.17g 精确往返）：三重实本征值 2 的
// 一部分被算成虚部极小的共轭对，而那一半的向量 v_p 几乎是实向量。不拆成两个实本征值时 x ± i·y 两列几乎线性相关，
// λ = 2 的本征空间条件数到 1e16。是否出现取决于舍入，所以在乘加融合的编译下可能一个也不出现（那时这几项只是普通算例）。
static const double kNearRealPairCases[][36] = {
    {3.7648825817067348, -2.3294995497565361, -1.3027110710796936, 4.8461080905531455, 2.0617395018233013, 6.2377768574711725,
     1.2821480303934092, 2.6802328153229062, -0.27936810087083236, 1.6838782742367371, 1.1511538121893605, 2.676657850403712,
     1.4256074047617413, 1.3293943692221986, 1.6466471947662775, 0.4682252385138751, 2.0041883343745122, 0.78055070111266867,
     -0.89127438032183048, 2.3337378434351419, 0.99331865397673968, -1.2957956032771565, -1.250207024520396, -3.9685987069758739,
     -0.72279719474966564, 0.80864437543810941, 0.74595744710004219, -0.6785600184752324, 0.17277683987114439, -0.26918474049378532,
     -1.11566872021888, -3.1060742470980975, -0.76761650977985563, -0.95073260781281821, 0.57017597103981599, -0.96874382838990625},
    {2.8371909146116558, 0.54823859856528789, 0.8103501658959904, -0.74162066767495505, 1.9292052562399205, 1.5677803848693217,
     2.1206859795207142, 0.33864414301524709, -3.0274081552084633, 0.75384561894971946, 3.1086666672946204, 7.4273256128795353,
     0.22451005700871932, 0.8117576064514842, 0.80762930680407241, 1.2897599118492928, 0.45869585431630316, 0.37196253935025014,
     -3.2379971009649626, -1.0874863806160036, 4.7720227319162847, -1.0919182914415235, -5.7627211844828725, -8.9661313530492528,
     6.083926532623118, 0.69208765094335878, -8.0906353961702902, 4.414037260499911, 12.594136845327274, 17.502308604043854,
     -4.113668207208856, -0.30292350742694929, 3.8326987961864387, -1.5597041558955289, -7.4061404797212873, -9.4856829183167264},
    {-1.2231758664924577, 1.5841916231789783, -1.4465833814871649, -0.75121655902145101, 1.441974894445853, -4.6972666851422957,
     -1.8947848509648721, 3.2282419163364673, -0.56659647063188157, -1.0385847857260055, 1.2038783448001711, -3.5022196793408269,
     -1.2551641770898945, -0.13434115100545624, 2.2078779371627517, 1.4480685081931757, 1.4827499089059495, 1.4868439773230722,
     2.8266521223915428, -1.4771659851413805, 2.1924741558247787, 2.9913305107687989, -0.13666575944234727, 5.3143417773396759,
     -0.73374026778572909, -0.00027117601153428539, -1.4601206432505514, 0.43295649084232524, 0.93362427688761873, -0.92976158091623651,
     0.46014374187281015, 0.68684126736008511, -1.5348233374372673, -2.1324981280429314, -2.3094100347858277, -2.1378987746631779},
    {3.5287284825350929, -2.122317538427402, -2.7587570271277118, 6.3552008006193983, -4.104601683086897, -5.5226106685188725,
     2.8387899748777925, -1.3957012040538281, -4.2976964872933978, 3.2285463836445283, -1.2693022580099018, -3.5320280240762632,
     -6.5830497075963725, 6.5991510352343905, 9.1747705968346356, -7.7569259645948829, 3.312573876673099, 8.0086939008231184,
     -1.693241012769451, 1.6893646115890713, 1.7870279210454592, -0.96043081505810113, 1.5764018031817719, 2.7957891505005623,
     -1.5918546310425994, 2.0937706024937035, 2.857753292288173, -0.99754097870553804, 2.0778407461690591, 1.416610522438889,
     5.3789234397699648, -6.2538579057025885, -7.6511295436917113, 8.4546334015992421, -4.1800355895358043, -6.4252078064268554},
    {2.6002441482734304, 0.34350800762722478, -0.43470960084568899, 0.64345276575877186, 0.83962242608156223, 0.97163878408506232,
     -2.0826390024079098, 1.9218693285195003, 1.3337385732836762, -0.2845686072624985, 2.5422488627530333, -1.1768164963798031,
     1.4743164177023538, -0.23983428883529792, -4.5533422545708788, 3.7913911227051935, 4.8188512180796348, 3.753769579229842,
     -3.0729937702318244, -1.6075442086518597, -2.5113856674554365, 2.3921110181258229, 3.1622375997801022, -1.7579571153226388,
     4.2423734137121958, 0.86733606299620314, -2.3857391677799522, 1.4973458721944426, -0.34006297764458593, 3.5187891127383635,
     2.0218890583371136, -0.14887798133230129, -3.3257440559946017, 1.3833288083635316, -0.62274513438684942, 3.9791807372967107},
    {1.5758757028092041, -0.40357696723006325, -0.21467288559298853, -5.3927748801979885, -0.52777499585479826, 1.0860668484138565,
     -0.94376567728343486, 0.78623342480794156, -0.0811502005989993, -4.738079335751638, -1.7041257766709486, 0.63436046854019756,
     -0.33717450651355718, -0.22166398672727597, 1.4057561976140109, 0.82957654868331143, 0.85198863921859691, -1.4129949423079959,
     1.2422799759745968, 0.46421152351714967, 2.9458019091322987, -0.70933006208401594, -4.8897568580391413, 6.1913640951356239,
     -0.58475451454984162, -0.97264869690181444, 0.086383693883415366, 5.6121977840857991, 1.0929526031135499, -2.1841481081149867,
     -0.032378058921146063, -0.43661339310422792, 0.71010270169555612, 3.5582393499933094, -1.5224669745654811, 1.8485121337393107},
    {0.84247891184075363, -2.0461726536219631, -2.1431445490907493, 0.25979817361181012, 1.1895550700084085, 1.1085538148529535,
     0.52301894027821216, 0.09277877537239107, 3.8623775620718934, -3.2488762762944092, -3.7475938402401416, -5.6170307152514285,
     -0.68839594089241674, 1.6448080825855653, 0.71404898483939394, 1.8180567785775688, -0.88655464902174985, -0.75756499612912209,
     0.36529657321510922, 0.77064271817331553, 0.33227114446194628, 2.1676464498105554, 0.12560807366932611, 0.36517824938761129,
     -0.28784558641602664, -0.51906196897528523, -0.19114891928141781, -0.11743231858934461, 1.7339504284761382, -0.49200732223333266,
     -0.37611141645286689, 2.6894815449738121, -0.59899543204629524, 1.977296955263619, -1.6657694456987457, 0.44909644966076773},
    {0.73723383375294183, 0.45938537416734448, 0.46337988074481151, -0.016603545869387976, 1.2379976984768954, 0.41701611139878508,
     1.1868358467110516, 0.44067571584806986, -0.49731394623589253, 0.48510501274989837, -0.85066471084391893, -0.21335076536893596,
     -7.2556943707711357, -8.147387736268108, -1.7877374735544396, -4.6943656287783933, -0.11976020596909781, 0.49311705815651447,
     0.60380469893610011, 4.563154734551345, 1.7898407163482923, 2.730503693734172, 1.1340456316816954, 0.12132009129702684,
     -7.9681429086360458, -10.440209497937913, -4.8477256471112931, -5.4743366056965357, 1.2245780053913176, 0.40428891908319414,
     -3.2154905339160078, 2.7708393580293071, 1.0185810924222531, -1.0954414114773665, 2.2733332381115741, 2.6547462248279379},
};

// ---------------------------------------------------------------------------------------------
// 构造的矩阵

// 实正交矩阵（Gram–Schmidt）
static ss::Matrix random_orthogonal(int n, std::mt19937_64& g) {
    ss::Matrix q = random_real(n, g);
    for (int j = 0; j < n; ++j) {
        for (int k = 0; k < j; ++k) {
            double s = 0.0;
            for (int i = 0; i < n; ++i) s += q(i, k).real() * q(i, j).real();
            for (int i = 0; i < n; ++i) q(i, j) -= s * q(i, k);
        }
        double nr = 0.0;
        for (int i = 0; i < n; ++i) nr += std::norm(q(i, j));
        nr = std::sqrt(nr);
        for (int i = 0; i < n; ++i) q(i, j) /= nr;
    }
    return q;
}

static ss::Matrix real_part(const ss::Matrix& a) {
    ss::Matrix r(a.rows(), a.cols());
    for (int j = 0; j < a.cols(); ++j)
        for (int i = 0; i < a.rows(); ++i) r(i, j) = a(i, j).real();
    return r;
}



int main() {
    const double eps = std::numeric_limits<double>::epsilon();
    std::mt19937_64 g(20261015);

    say("被测：%s 版，导出函数默认线程数 %d\nK 内核与库逐位（库的线程数 1/4）\n", kMethod, kIsQR ? default_threads_for_print() : 1);
    for (int n : {1, 2, 3, 4, 5, 8, 16, 33, 64, 100, 191, 192, 193, 250}) test_kernels("复随机 n=" + std::to_string(n), random_complex(n, g));
    for (int n : {1, 2, 3, 7, 30, 120, 200}) test_kernels("实随机 n=" + std::to_string(n), random_real(n, g));

    say("C 复数版\n");
    for (int n : {1, 2, 3, 4, 5, 8, 16, 33, 64, 100, 191, 192, 250}) {
        const ss::Matrix A = random_complex(n, g);
        test_complex("复随机 n=" + std::to_string(n), A, 64.0 * n * eps);
        test_kernels("复随机(C) n=" + std::to_string(n), A);
    }

    say("R 实数版（随机）\n");
    double worst_real = 0.0;
    for (int n : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 31, 32, 60, 100, 150, 200}) {
        for (int rep = 0; rep < (n <= 10 ? 20 : 2); ++rep) {
            const ss::Matrix A = random_real(n, g);
            const std::string name = "实随机 n=" + std::to_string(n) + " #" + std::to_string(rep);
            worst_real = (std::max)(worst_real, test_real(name, A, 64.0 * n * eps, 1e-12));
            test_complex(name, A, 64.0 * n * eps);
        }
    }
    say("  实随机矩阵实数版的最大残差 %.2e\n", worst_real);

    say("R 实数版（构造的矩阵）\n");
    struct Case {
        std::string name;
        ss::Matrix A;
        double tol, match;
        int expect_real;   // 期望的实本征值个数，-1 不检查
        bool lr_unsupported = false;   // LR 版按文件开头的说明不支持（不做缩放）：只查返回 k > 0 或 -4
    };
    std::vector<Case> cases;
    cases.push_back({"零矩阵 4", ss::Matrix(4, 4), 1e-15, 1e-15, 4});
    cases.push_back({"单位阵 5", ss::Matrix::identity(5), 1e-15, 1e-15, 5});
    {
        ss::Matrix d(5, 5);
        const double diag[] = {3, -1, 3, 2, -1};
        for (int i = 0; i < 5; ++i) d(i, i) = diag[i];
        cases.push_back({"对角 (3,-1,3,2,-1)", d, 1e-15, 1e-15, 5});
    }
    {
        ss::Matrix u = random_real(6, g);
        for (int j = 0; j < 6; ++j)
            for (int i = j + 1; i < 6; ++i) u(i, j) = 0.0;
        cases.push_back({"上三角 6", u, 1e-14, 1e-15, 6});
    }
    {
        ss::Matrix j4(4, 4);
        for (int i = 0; i < 4; ++i) j4(i, i) = 2.0;
        for (int i = 0; i + 1 < 4; ++i) j4(i, i + 1) = 1.0;
        cases.push_back({"Jordan 块 J4(2)", j4, 1e-15, 1e-15, 4});
    }
    cases.push_back({"友矩阵 (x-1)^2(x-2)", ss::Matrix{{4.0, -5.0, 2.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}}, 1e-7, 1e-7, kIsQR ? 3 : -1});   // 二重亏损根标成实根还是一对都对；只记下 QR 版现在的结果
    {
        ss::Matrix d(5, 5);
        d(0, 0) = 1; d(0, 1) = 2; d(1, 0) = -2; d(1, 1) = 1;
        d(2, 2) = 1; d(2, 3) = 3; d(3, 2) = -3; d(3, 3) = 1;
        d(4, 4) = 1;
        cases.push_back({"块对角 1±2i,1±3i,1（未打乱）", d, 1e-15, 1e-15, 1});
        const ss::Matrix q = random_orthogonal(5, g);
        cases.push_back({"1±2i,1±3i,1（正交相似打乱）", real_part(q * d * ss::adjoint(q)), 1e-14, 1e-14, 1});
    }
    {
        ss::Matrix s = random_real(10, g);
        cases.push_back({"对称 10", real_part(s + ss::transpose(s)), 64.0 * 10 * eps, 1e-12, 10});
        ss::Matrix k = random_real(7, g);
        cases.push_back({"反对称 7", real_part(k - ss::transpose(k)), 64.0 * 7 * eps, 1e-12, 1});
        ss::Matrix k8 = random_real(8, g);
        cases.push_back({"I + 反对称 8（实部全为 1）", real_part(ss::Matrix::identity(8) + k8 - ss::transpose(k8)), 64.0 * 8 * eps, 1e-12, 0});
    }
    {
        ss::Matrix p(6, 6);
        for (int i = 0; i < 6; ++i) p((i + 1) % 6, i) = 1.0;
        cases.push_back({"循环置换 6（六次单位根）", p, 64.0 * 6 * eps, 1e-12, 2});
    }
    // 亏损的实重根：复数算法把 k 重根散成 eps^(1/k) 量级的一簇。三重根取实之后残差到 1e-12 量级（实测 9e-13），
    // 容差给 1e-11；本征值与复数版只能对到这一簇的大小（eps^(1/3) ≈ 6e-6，eps^(1/4) ≈ 1.2e-4）。
    cases.push_back({"友矩阵 (x-1)^3(x+2)", ss::Matrix{{1.0, 3.0, -5.0, 2.0}, {1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}}, 1e-11, 1e-4, -1});
    {
        ss::Matrix c(5, 5);                           // (x-1)^4 (x+1) = x^5 - 3x^4 + 2x^3 + 2x^2 - 3x + 1
        const double coef[] = {3.0, -2.0, -2.0, 3.0, -1.0};
        for (int j = 0; j < 5; ++j) c(0, j) = coef[j];
        for (int i = 1; i < 5; ++i) c(i, i - 1) = 1.0;
        cases.push_back({"友矩阵 (x-1)^4(x+1)", c, 1e-11, 1e-3, -1});
    }
    for (int k : {2, 3, 4}) {                         // Jordan 块 J_k(1) ⊕ diag(-1, 0.5)，再做一次实正交相似
        const int n = k + 2;
        ss::Matrix j(n, n);
        for (int i = 0; i < k; ++i) j(i, i) = 1.0;
        for (int i = 0; i + 1 < k; ++i) j(i, i + 1) = 1.0;
        j(k, k) = -1.0;
        j(k + 1, k + 1) = 0.5;
        const ss::Matrix q = random_orthogonal(n, g);
        const double tol = k == 2 ? 64.0 * n * eps : 1e-11;   // 二重亏损根：残差应保持机器精度
        cases.push_back({"Q·(J" + std::to_string(k) + "(1)⊕diag(-1,0.5))·Q^T", real_part(q * j * ss::adjoint(q)), tol, 1e-3, -1});
    }
    {
        ss::Matrix d(6, 6);                           // 两两相差 1e-12 的实本征值，正交相似打乱
        const double diag[] = {1.0, 1.0 + 1e-12, 1.0 - 1e-12, 2.0, 2.0 + 1e-12, -3.0};
        for (int i = 0; i < 6; ++i) d(i, i) = diag[i];
        const ss::Matrix q = random_orthogonal(6, g);
        cases.push_back({"相差 1e-12 的实本征值簇", real_part(q * d * ss::adjoint(q)), 64.0 * 6 * eps, 1e-12, -1});
    }
    {
        ss::Matrix r = random_real(12, g);
        cases.push_back({"随机 12 × 1e-200", real_part(r * cplx(1e-200)), 64.0 * 12 * eps, 1e-12, -1, true});
        cases.push_back({"随机 12 × 1e+200", real_part(r * cplx(1e200)), 64.0 * 12 * eps, 1e-12, -1, true});
    }
    {
        // 只在快要上溢时才起作用的两处保护（eig_qr.hpp 的 kBoundMax、kTinyLead）：元素约 1e300 的上三角（回代），
        // 一列里 1e-10 与 1e300（化 Hessenberg）。各自再放进 196、195 阶的左上角，走分块写法。
        const ss::Matrix t{{0.0, 1e300, 1e300, 0.0}, {0.0, 1e292, 0.0, 1e300}, {0.0, 0.0, 1e292, 1e300}, {0.0, 0.0, 0.0, 2e292}};
        const ss::Matrix h{{0.0, 0.0, 0.0}, {1e-10, 1.0, 0.0}, {1e300, 0.0, 2.0}};
        const auto embed = [](const ss::Matrix& b, int n) {
            ss::Matrix a(n, n);
            for (int j = 0; j < b.cols(); ++j)
                for (int i = 0; i < b.rows(); ++i) a(i, j) = b(i, j);
            for (int k = b.rows(); k < n; ++k) a(k, k) = 1.0 + k;
            return a;
        };
        cases.push_back({"回代：元素约 1e300 的上三角", t, 64.0 * 4 * eps, 1e-12, 4, true});
        cases.push_back({"化 Hessenberg：一列里 1e-10 与 1e300", h, 64.0 * 3 * eps, 1e-12, 3});   // LR 版用初等消元，没有 g/f，照常能算
        cases.push_back({"同上的上三角放进 196 阶", embed(t, 196), 64.0 * 196 * eps, 1e-12, 196, true});
        cases.push_back({"同上的 3 阶放进 195 阶", embed(h, 195), 64.0 * 195 * eps, 1e-12, 195});
    }
    {
        // 化 Hessenberg 的同一个反例，u_m 改成复数（走 unit_phase 的复相位），只能交给复数版
        const ss::Matrix h3{{0.0, 0.0, 0.0}, {cplx(1e-10, -3e-10), 1.0, 0.0}, {cplx(-4e299, 7e299), 0.0, 2.0}};
        const double res_c = test_complex("化 Hessenberg：复数的 1e-10 与 1e300", h3, 64.0 * 3 * eps);
        test_kernels("化 Hessenberg：复数的 1e-10 与 1e300", h3);
        say("  %-34s 复数版残差 %.2e\n", "化 Hessenberg：复数的 1e-10 与 1e300", res_c);
    }
    for (const Case& c : cases) {
        if (!kBackwardStable && c.lr_unsupported) {
            const RealOut r = call_real(c.A);
            const ComplexOut co = call_complex(c.A);
            check((r.status > 0 || r.status == -4) && (co.status > 0 || co.status == -4),
                  c.name + " LR 版返回 k > 0 或 -4（实际 " + std::to_string(r.status) + "、" + std::to_string(co.status) + "）");
            say("  %-34s LR 版返回 %d / %d（不做缩放，见文件开头）\n", c.name.c_str(), r.status, co.status);
            continue;
        }
        int reals = -1;
        const double res = test_real(c.name, c.A, c.tol, c.match, &reals);
        if (c.expect_real >= 0) check(reals == c.expect_real, c.name + " 实本征值个数 " + std::to_string(reals) + "，期望 " + std::to_string(c.expect_real));
        const double res_c = test_complex(c.name, c.A, (std::max)(c.tol, 64.0 * c.A.rows() * eps));
        test_kernels(c.name, c.A);
        say("  %-34s 实数版残差 %.2e，复数版残差 %.2e，实本征值 %d 个\n", c.name.c_str(), res, res_c, reals);
    }
    {   // 把同实部的情形单独看一眼次序
        RealOut out = call_real(cases[7].A);
        say("  %s 的输出次序：", cases[7].name.c_str());
        for (cplx z : out.values) say(" %.17g%+.3gi", z.real(), z.imag());
        say("\n");
    }

    say("R 实数版（重复的本征值）\n");
    {
        // 整数反对称矩阵，本征值 ±3i 各两重：曾经两对挑到同一个向量的两份（向量组奇异）
        const ss::Matrix A{{0, 1, -2, -2}, {-1, 0, 2, -2}, {2, -2, 0, -1}, {2, 2, 1, 0}};
        test_real("整数反对称 4（±3i 各两重）", A, 64.0 * 4 * eps, 1e-12);
        ss::Matrix d(5, 5);                           // Q·(R⊕R⊕3)·Q^T，R = [[1,2],[-2,1]]：1±2i 各两重
        d(0, 0) = 1; d(0, 1) = 2; d(1, 0) = -2; d(1, 1) = 1;
        d(2, 2) = 1; d(2, 3) = 2; d(3, 2) = -2; d(3, 3) = 1;
        d(4, 4) = 3;
        for (int rep = 0; rep < 2000; ++rep) {
            const ss::Matrix q = random_orthogonal(5, g);
            test_real("Q(R⊕R⊕3)Q^T #" + std::to_string(rep), real_part(q * d * ss::adjoint(q)), 64.0 * 5 * eps, 1e-12);
        }
        ss::Matrix e(6, 6);                           // S·(2I₃ ⊕ [[.5,3],[-3,.5]] ⊕ -1)·S⁻¹，S 是高斯随机矩阵：三重的实本征值 2
        e(0, 0) = 2; e(1, 1) = 2; e(2, 2) = 2;
        e(3, 3) = 0.5; e(3, 4) = 3; e(4, 3) = -3; e(4, 4) = 0.5;
        e(5, 5) = -1;
        for (int rep = 0; rep < 2000; ++rep) {
            const ss::Matrix s = random_real(6, g);
            const ss::Matrix a = real_part(s * e * ss::inverse(s));
            test_real("S(2I₃⊕…)S⁻¹ #" + std::to_string(rep), a, 1e-13, 1e-10);
        }
        for (std::size_t c = 0; c < sizeof(kNearRealPairCases) / sizeof(kNearRealPairCases[0]); ++c) {
            ss::Matrix a(6, 6);
            for (int i = 0; i < 6; ++i)
                for (int j = 0; j < 6; ++j) a(i, j) = kNearRealPairCases[c][6 * i + j];
            test_real("近实共轭对样例 " + std::to_string(c), a, 1e-13, 1e-10);
        }
        // 这几个样例确实走到了“拆成两个实本征值”的分支（至少一个；在不融合乘加的编译下 clang 8 个、MSVC 6 个）
        int fired = 0;
        for (std::size_t c = 0; c < sizeof(kNearRealPairCases) / sizeof(kNearRealPairCases[0]); ++c) {
            me::Matrix a(6, 6);
            for (int i = 0; i < 6; ++i)
                for (int j = 0; j < 6; ++j) a(i, j) = kNearRealPairCases[c][6 * i + j];
            me::Vector w;
            me::Matrix v;
            if (kernel_eig(a, w, v, 1) != 0) continue;
            const std::vector<int> partner = me::pair_conjugates(w);
            for (int i = 0; i < 6; ++i) {
                const int j = partner[i];
                if (j <= i || std::abs(w[i] - 2.0) > 1e-6) continue;
                const int pp = w[i].imag() > 0.0 ? i : j, qq = i + j - pp;
                me::make_largest_real(v.col(pp), 6);
                if (me::split_pair(v.col(pp), v.col(qq), 6)) ++fired;
            }
        }
        // 样例是用 QR 挑出来的；LR 的舍入不同，不一定走到这个分支
        if (kIsQR) check(fired >= 1, "嵌入的近实共轭对样例里有走到拆分分支的（实际 " + std::to_string(fired) + " 个）");
    }
    {
        // 接近亏损的二重实根被算成一对时 v_q ≈ conj(v_p)：不能拆（拆开两个实向量相同，向量组奇异）
        test_real("友矩阵 (x+3)^2(x^2-1)", ss::Matrix{{-6.0, -8.0, 6.0, 9.0}, {1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}},
                  64.0 * 4 * eps, 1e-6);
        int count = 0;
        for (int ra = -3; ra <= 3; ++ra)                  // (x−a)²(x−b)(x−c) 的友矩阵，a、b、c 是互不相同的小整数
            for (int rb = -3; rb <= 3; ++rb)
                for (int rc = rb + 1; rc <= 3; ++rc) {
                    if (rb == ra || rc == ra) continue;
                    double poly[5] = {1.0, 0.0, 0.0, 0.0, 0.0};   // 首一多项式的系数，次数从高到低
                    int deg = 0;
                    const double roots[] = {double(ra), double(ra), double(rb), double(rc)};
                    for (double r : roots) {
                        for (int k = deg + 1; k >= 1; --k) poly[k] -= r * poly[k - 1];
                        ++deg;
                    }
                    ss::Matrix c(4, 4);
                    for (int j = 0; j < 4; ++j) c(0, j) = -poly[j + 1];
                    for (int i = 1; i < 4; ++i) c(i, i - 1) = 1.0;
                    test_real("友矩阵 (x-" + std::to_string(ra) + ")^2(x-" + std::to_string(rb) + ")(x-" + std::to_string(rc) + ")",
                              c, 64.0 * 4 * eps, 1e-6);
                    ++count;
                }
        say("  (x-a)^2(x-b)(x-c) 友矩阵 %d 个\n", count);
        ss::Matrix j2(4, 4);                              // S·(J2(1) ⊕ diag(-1, 0.5))·S⁻¹
        j2(0, 0) = 1; j2(0, 1) = 1; j2(1, 1) = 1; j2(2, 2) = -1; j2(3, 3) = 0.5;
        for (int rep = 0; rep < 1000; ++rep) {
            const ss::Matrix s = random_real(4, g);
            test_real("S(J2⊕diag(-1,.5))S⁻¹ #" + std::to_string(rep), real_part(s * j2 * ss::inverse(s)), 64.0 * 4 * eps, 1e-6);
        }
        // 同一个本征值上两个亏损块：已知约千分之二的情形实数版向量组数值上奇异（独立文件开头的局限），这里只查残差、结构
        ss::Matrix jj(5, 5);
        jj(0, 0) = 1; jj(0, 1) = 1; jj(1, 1) = 1; jj(2, 2) = 1; jj(2, 3) = 1; jj(3, 3) = 1; jj(4, 4) = -1;
        for (int rep = 0; rep < 500; ++rep) {
            const ss::Matrix s = random_real(5, g);
            test_real("S(J2⊕J2⊕-1)S⁻¹ #" + std::to_string(rep), real_part(s * jj * ss::inverse(s)), 64.0 * 5 * eps, 1e-6, nullptr, false);
        }
    }
    say("  实数版与复数版向量组条件数之比的最大值 %.2e\n", g_cond_ratio);

    say("E 返回码、共用存储、并发\n");
    test_errors();
    test_aliasing(random_real(9, g));
    test_concurrency();

    say("%d 项检查，%d 项失败\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
