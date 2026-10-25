// tests/mkl/test_rcwa_mkl.cpp —— oneMKL 后端（stablercwa/mkl/mkl_backend.hpp）的验收：本征分解本身，以及经 custom_eig
// 接进一维、圆锥、二维求解器和近场重建之后的结果
//
// 要有 oneMKL 2025.0.1 或更新（Windows：先运行 standalone/install_mkl.bat）。在仓库根目录：
//   cmake -S . -B build -DSTABLERCWA_WITH_MKL=ON
//   cmake --build build --config Release --target test_rcwa_mkl
//   build\Release\test_rcwa_mkl.exe
//
// MKL 的结果与 eig_qr 只在舍入误差以内一致，所以这里比的都是数值差，不是逐位：
//   A  后端的本征分解：随机矩阵（复的走 zgeev；实的打开 real_path 走 dgeev）的本征对残差、归一、与 eig_qr 的本征值；dgeev 与 zgeev
//      两条路；实矩阵的判断（nearly_real）；空矩阵与非法输入；线程池里、多个宿主线程同时调用；线程数的规则（call_threads）；
//      调用后恢复调用线程的 MKL 线程数
//   A2 后端的乘法（zgemm，稀疏的 B 交回内置）与线性求解（zgetrf + zgetrs）：与内置的比、后向误差、零主元、空矩阵、
//      大小不合、在线程池里与宿主线程里同时调用
//   B  一维 TE/TM：几种结构、M 到 150，各级效率与默认（QR）的差、无吸收时的能量守恒、本征残差；README 表里的高阶算例
//   C  一维的重合本征值：均匀层也做本征分解（homogeneous_fast_path = false）、恰好正入射与恰好 Littrow 入射的随机结构，
//      与快速路径比；rcwa1d.hpp 写明 custom_eig 要能处理这种情形。打开 real_path：均匀层的层矩阵是实的，走 dgeev
//   D  圆锥入射：几种结构与方位角，含 ky 接近零、退化层改用 Schur 基的情形
//   E  二维：tests/rcwa2d_reference.inc 的 24 个冻结 Python 参照，容差用参照自带的；并行层的变体。打开 real_path：
//      居中对称、无吸收的算例走 dgeev，有吸收、不对称的由判据挑出来走 zgeev
//   F  近场：一维、圆锥、二维的场分布与默认（QR）比
//   G  parallel_layers、sweep 与 MKL 后端同用
//   H  对称、无吸收的结构，打开 real_path（层矩阵在舍入误差以内是实的，走 dgeev）：一维 TE/TM、圆锥、近场，与 QR、
//      与默认设定（一律 zgeev）比；有吸收（k = 1e-12 也算）或不对称时仍走 zgeev；k = 1e-15 的吸收与舍入误差分不开，走 dgeev。
//      高 Q 共振附近 k = 1e-13 的弱吸收：默认设定照样算对；打开 real_path 时会被当成无吸收（所以 real_path 默认关闭）
//   J  JSON 入口的 "backend": "mkl"：planar、conical、periodic2d 的应答与 "builtin" 比；三个钩子都设上；按折射率决定
//      real_path（layers_lossless）；与 eig: lr、linear: lu_nopivot / qr 同用、"eig": "mkl" 都是请求错误
//   K  奇异与 Rayleigh 异常点，三个钩子都交给 MKL：本征向量全为零、出射介质恰好掠射又紧挨光栅层（界面方程组奇异）
//      都要报数值失败，入射介质一侧掠射、有限膜恰好截止要照常求解、能量守恒，扫描里只有异常点那一个是 NaN，极端折射率
//      要么全部失败要么守恒——与内置的 LU 一致
// B ~ G 跑两遍：先只把本征分解交给 MKL（默认设定，一律 zgeev），再用 use_mkl 把乘法、线性求解也交给它。
// 退出状态：全部通过 0，有失败 1。
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "stablercwa/stablercwa.hpp"
#include "stablercwa/rcwa1d_fields.hpp"
#include "stablercwa/rcwa1d_conical_fields.hpp"
#include "stablercwa/rcwa2d_fields.hpp"
#include "stablercwa/mkl/mkl_backend.hpp"
#include "stablercwa/rcwa_json.hpp"
#include "structures.hpp"

#if !STABLERCWA_WITH_MKL
#error "test_rcwa_mkl needs STABLERCWA_WITH_MKL=1 (the CMake target stablercwa_mkl defines it)"
#endif

using namespace stablercwa;
namespace rcwa = stablercwa::rcwa;
#include "rcwa2d_reference.inc"

// ---------------------------------------------------------------------------------------------
// 文字一律经 say() 输出（用法同 std::printf）：MSVC 检查 printf 的格式串时按系统代码页读，含中文的格式串会被误报。
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

static int g_pass = 0, g_total = 0;

static void check(bool ok, const char* file, int line, const char* text) {
    ++g_total;
    if (ok) ++g_pass;
    else say("  失败 %s:%d  %s\n", file, line, text);
}
// value <= bound；失败时把实际数值打印出来。NaN 也算失败。
static void check_le(double value, double bound, const char* file, int line, const char* text) {
    ++g_total;
    if (value <= bound) ++g_pass;
    else say("  失败 %s:%d  %s：%.3e > %.3e\n", file, line, text, value, bound);
}
#define CHECK(cond) check(static_cast<bool>(cond), __FILE__, __LINE__, #cond)
#define CHECK_LE(value, bound) check_le((value), (bound), __FILE__, __LINE__, #value)

static const double kMachEps = std::numeric_limits<double>::epsilon();

// ---------------------------------------------------------------------------------------------
// 度量
static double gap(const RealVector& a, const RealVector& b) {
    if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
    double worst = 0.0;
    for (std::size_t j = 0; j < a.size(); ++j) {
        const double e = std::abs(a[j] - b[j]);
        if (!(e <= worst)) worst = e;                 // NaN 也会被留下
    }
    return worst;
}
static double gap(const rcwa::Result& a, const rcwa::Result& b) { return std::max(gap(a.DEr, b.DEr), gap(a.DEt, b.DEt)); }

static double gap(const std::vector<rcwa::JonesEfficiency>& a, const std::vector<rcwa::JonesEfficiency>& b) {
    if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
    double worst = 0.0;
    for (std::size_t j = 0; j < a.size(); ++j)
        for (double e : {a[j].ss - b[j].ss, a[j].sp - b[j].sp, a[j].ps - b[j].ps, a[j].pp - b[j].pp})
            if (!(std::abs(e) <= worst)) worst = std::abs(e);
    return worst;
}
static double gap(const rcwa::ConicalResult& a, const rcwa::ConicalResult& b) { return std::max(gap(a.DEr, b.DEr), gap(a.DEt, b.DEt)); }

// 六个场分量的最大差，相对整幅场的最大模
static double field_gap(const std::array<Matrix, 6>& a, const std::array<Matrix, 6>& b) {
    double worst = 0.0, scale = 0.0;
    for (int c = 0; c < 6; ++c) {
        if (a[c].rows() != b[c].rows() || a[c].cols() != b[c].cols()) return std::numeric_limits<double>::infinity();
        for (int j = 0; j < a[c].cols(); ++j)
            for (int i = 0; i < a[c].rows(); ++i) {
                const double d = std::abs(a[c](i, j) - b[c](i, j));
                if (!(d <= worst)) worst = d;
                scale = std::max(scale, std::abs(a[c](i, j)));
            }
    }
    return scale > 0.0 ? worst / scale : worst;
}

// 两组本征值逐个配对（贪心找最近、不重复使用）后的最大距离
static double match_distance(const Vector& got, const Vector& expected) {
    if (got.size() != expected.size()) return std::numeric_limits<double>::infinity();
    std::vector<char> used(expected.size(), 0);
    double worst = 0.0;
    for (const cplx& z : got) {
        std::size_t best = 0;
        double d = std::numeric_limits<double>::infinity();
        for (std::size_t k = 0; k < expected.size(); ++k)
            if (!used[k] && std::abs(z - expected[k]) < d) {
                d = std::abs(z - expected[k]);
                best = k;
            }
        used[best] = 1;
        if (!(d <= worst)) worst = d;
    }
    return worst;
}

static Matrix random_matrix(int n, std::mt19937_64& g, bool real) {
    std::normal_distribution<double> N(0.0, 1.0);
    Matrix a(n, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) a(i, j) = cplx(N(g), real ? 0.0 : N(g));
    return a;
}

// B ~ G 的第二遍（g_full）：乘法、线性求解也交给 MKL。
static bool g_full = false;
template <class O>
static void add_dense(O& o) {
    if (!g_full) return;
    o.custom_multiply = mkl::multiply_hook();
    o.custom_solve = mkl::solve_hook();
}

static rcwa::Options mkl_options() {
    rcwa::Options o;
    o.custom_eig = mkl::eig_hook();
    add_dense(o);
    return o;
}

// 打开 real_path 的设定（调用方确认各层无吸收）
static mkl::Settings real_settings() {
    mkl::Settings s;
    s.real_path = true;
    return s;
}

// 记下层矩阵有多少个、其中多少个按实矩阵走了 dgeev（可以被几个线程同时调用）
struct PathCount {
    std::atomic<int> real{0}, total{0};
};
static std::function<EigResult(const Matrix&)> counting_hook(PathCount& count, const mkl::Settings& settings = {}) {
    return [&count, settings](const Matrix& a) {
        ++count.total;
        if (settings.real_path && mkl::nearly_real(a) && mkl::dgeev_usable()) ++count.real;
        return mkl::eig(a, settings);
    };
}

// ---------------------------------------------------------------------------------------------
// A：后端本身
static void test_backend() {
    say("[A] 后端：%s\n", mkl::version_string().c_str());
    std::mt19937_64 g(2026102501);
    double worst_res = 0.0, worst_norm = 0.0, worst_vs_qr = 0.0;
    for (int n : {1, 2, 3, 8, 50, 300})
        for (bool real : {false, true}) {
            const Matrix a = random_matrix(n, g, real);
            mkl::Settings settings;
            settings.real_path = real;
            const EigResult r = mkl::eig(a, settings);
            CHECK(r.info == 0 && r.vectors.rows() == n && r.vectors.cols() == n);
            if (r.info != 0) continue;
            const double res = eig_residual(a, r);
            CHECK_LE(res, 64.0 * n * kMachEps);
            worst_res = std::max(worst_res, res);
            for (int j = 0; j < n; ++j) {
                double s = 0.0;
                for (int i = 0; i < n; ++i) s += std::norm(r.vectors(i, j));
                worst_norm = std::max(worst_norm, std::abs(std::sqrt(s) - 1.0));
            }
            const EigResult q = eig_qr(a);
            double scale = 0.0;
            for (const cplx& z : q.values) scale = std::max(scale, std::abs(z));
            const double d = match_distance(r.values, q.values) / scale;
            CHECK_LE(d, 1e-10);
            worst_vs_qr = std::max(worst_vs_qr, d);
        }
    CHECK_LE(worst_norm, 1e-14);
    say("  随机矩阵（n = 1 … 300，复与实）：残差最大 %.2e，|‖v‖ − 1| 最大 %.2e，与 eig_qr 的本征值差最大 %.2e（相对 max|λ|）\n", worst_res,
        worst_norm, worst_vs_qr);

    CHECK(mkl::eig(Matrix()).values.empty());
    bool threw = false;
    try {
        (void)mkl::eig(Matrix(2, 3));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    Matrix bad(3, 3);
    bad(1, 2) = std::numeric_limits<double>::quiet_NaN();
    try {
        (void)mkl::eig(bad);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    for (int threads : {1, 4}) {
        mkl::Settings s;
        s.threads = threads;
        const Matrix a = random_matrix(200, g, false);
        const EigResult r = mkl::eig(a, s);
        CHECK(r.info == 0);
        CHECK_LE(eig_residual(a, r), 64.0 * 200 * kMachEps);
    }

    // 线程池里同时调用（每次调用只给 MKL 1 个线程），以及几个宿主线程同时调用
    std::vector<Matrix> mats;
    for (int k = 0; k < 8; ++k) mats.push_back(random_matrix(120, g, k % 2 == 1));
    std::vector<double> res(mats.size(), 1.0);
    std::vector<char> inside(mats.size(), 0);
    set_num_threads(4);
    parallel_for(0, static_cast<int>(mats.size()), [&](int k) {
        inside[k] = in_parallel_region() ? 1 : 0;
        const EigResult r = mkl::eig(mats[k]);
        res[k] = r.info == 0 ? eig_residual(mats[k], r) : 1.0;
    });
    set_num_threads(1);
    bool all_inside = true;
    for (std::size_t k = 0; k < mats.size(); ++k) {
        CHECK_LE(res[k], 64.0 * 120 * kMachEps);
        all_inside = all_inside && inside[k] != 0;
    }
    CHECK(all_inside);
    std::vector<double> host(4, 1.0);
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t)
        threads.emplace_back([&, t] {
            double worst = 0.0;
            for (int k = 0; k < 3; ++k) {
                const Matrix& a = mats[static_cast<std::size_t>(2 * t + k % 2)];
                const EigResult r = mkl::eig(a);
                worst = std::max(worst, r.info == 0 ? eig_residual(a, r) : 1.0);
            }
            host[static_cast<std::size_t>(t)] = worst;
        });
    for (auto& th : threads) th.join();
    for (double h : host) CHECK_LE(h, 64.0 * 120 * kMachEps);
    say("  线程池 4 个线程、宿主 4 个线程同时调用：结果都正常\n");

    // 线程数的规则：并行区里 1 个；否则 Settings::threads，或 num_threads()（有别的调用在算时 1 个）
    set_num_threads(4);
    CHECK(mkl::detail::call_threads(0, 0) == 4 && mkl::detail::call_threads(0, 2) == 1 && mkl::detail::call_threads(3, 2) == 3);
    std::vector<int> inner(8, 0);
    parallel_for(0, 8, [&](int k) { inner[static_cast<std::size_t>(k)] = mkl::detail::call_threads(k % 2 == 0 ? 6 : 0, 0); });
    set_num_threads(1);
    bool all_one = true;
    for (int t : inner) all_one = all_one && t == 1;
    CHECK(all_one);
    CHECK(mkl::detail::call_threads(0, 0) == 1 && mkl::detail::call_threads(5, 0) == 5);

    // 每次调用结束都恢复调用线程自己原来的 MKL 线程数（MKL_Set_Num_Threads_Local 返回原来的值）
    MKL_Set_Num_Threads_Local(3);
    (void)mkl::eig(random_matrix(40, g, false));
    (void)mkl::eig(random_matrix(40, g, true));
    CHECK(MKL_Set_Num_Threads_Local(0) == 3);

    // dgeev 与 zgeev 两条路：实矩阵的结果的结构（实本征值的向量是实的，共轭对相邻、严格共轭），两条路的本征值之差
    CHECK(mkl::dgeev_usable());
    CHECK(!mkl::Settings{}.real_path);                               // 默认关闭
    double worst_paths = 0.0;
    bool structure_ok = true;
    for (int n : {2, 3, 7, 64, 250}) {
        const Matrix a = random_matrix(n, g, true);
        const EigResult r = mkl::eig(a, real_settings()), z = mkl::eig(a);
        CHECK(r.info == 0 && z.info == 0);
        if (r.info != 0 || z.info != 0) continue;
        CHECK_LE(eig_residual(a, z), 64.0 * n * kMachEps);
        for (int j = 0; j < n; ++j) {
            const cplx lam = r.values[static_cast<std::size_t>(j)];
            if (lam.imag() == 0.0) {
                for (int i = 0; i < n; ++i) structure_ok = structure_ok && r.vectors(i, j).imag() == 0.0;
            } else if (lam.imag() > 0.0) {
                structure_ok = structure_ok && j + 1 < n && r.values[static_cast<std::size_t>(j) + 1] == std::conj(lam);
                for (int i = 0; structure_ok && i < n; ++i) structure_ok = r.vectors(i, j + 1) == std::conj(r.vectors(i, j));
            }
        }
        double scale = 0.0;
        for (const cplx& v : z.values) scale = std::max(scale, std::abs(v));
        worst_paths = std::max(worst_paths, match_distance(r.values, z.values) / scale);
    }
    CHECK(structure_ok);
    CHECK_LE(worst_paths, 1e-10);

    // 实矩阵的判断：每个元的虚部不超过所在行、列量级的 1.1e-13 倍
    {
        const Matrix a = random_matrix(50, g, true);
        Matrix noisy = a, tilted = a, huge = a;
        for (int j = 0; j < 50; ++j)
            for (int i = 0; i < 50; ++i) {
                noisy(i, j) += cplx(0.0, 1e-15 * a(i, j).real());    // 虚部是实部的 1e-15：在阈值以内
                tilted(i, j) += cplx(0.0, 1e-12 * a(i, j).real());   // 1e-12：模最大的元超过阈值
                huge(i, j) = 1e300 * noisy(i, j);                     // 缩放不改变判断，也不溢出
            }
        CHECK(mkl::nearly_real(a) && mkl::nearly_real(Matrix(4, 4)) && mkl::nearly_real(noisy) && mkl::nearly_real(huge));
        CHECK(!mkl::nearly_real(tilted) && !mkl::nearly_real(random_matrix(50, g, false)));
        const EigResult r = mkl::eig(noisy, real_settings());        // 求的是 Re(A)，对 A 本身的残差仍在舍入误差以内
        CHECK(r.info == 0);
        if (r.info == 0) CHECK_LE(eig_residual(noisy, r), 64.0 * 50 * kMachEps);
    }
    say("  实矩阵走 dgeev：与 zgeev 的本征值差最大 %.2e（相对 max|λ|），实本征值、共轭对的结构%s\n", worst_paths,
        structure_ok ? "正确" : "有误");
}

// ---------------------------------------------------------------------------------------------
// A2：后端的乘法与线性求解
static Matrix random_rect(int m, int n, std::mt19937_64& g) {
    std::normal_distribution<double> N(0.0, 1.0);
    Matrix a(m, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) a(i, j) = cplx(N(g), N(g));
    return a;
}
static double max_diff(const Matrix& a, const Matrix& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return std::numeric_limits<double>::infinity();
    double v = 0.0;
    for (int j = 0; j < a.cols(); ++j)
        for (int i = 0; i < a.rows(); ++i) {
            const double d = std::abs(a(i, j) - b(i, j));
            if (!(d <= v)) v = d;
        }
    return v;
}
static bool same_bits(const Matrix& a, const Matrix& b) {
    return a.rows() == b.rows() && a.cols() == b.cols() &&
           (a.rows() * a.cols() == 0 ||
            std::memcmp(a.data(), b.data(), sizeof(cplx) * static_cast<std::size_t>(a.rows()) * a.cols()) == 0);
}
template <class F>
static bool throws_invalid(F f) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

static void test_dense_backend() {
    say("[A2] 后端的乘法（zgemm）与线性求解（zgetrf + zgetrs）\n");
    std::mt19937_64 g(2026102502);
    double worst_mul = 0.0, worst_solve = 0.0, worst_backward = 0.0;
    for (int m : {1, 3, 17, 120})
        for (int p : {1, 5, 64, 130})
            for (int n : {1, 9, 70}) {
                const Matrix a = random_rect(m, p, g), b = random_rect(p, n, g);
                const double d = max_diff(mkl::multiply(a, b), a * b) / (max_abs(a) * max_abs(b) * p);
                CHECK_LE(d, 8.0 * kMachEps);
                worst_mul = std::max(worst_mul, d);
            }
    // 稀疏的 B（对角阵、只有一列非零）交回内置的 a * b：逐位相同
    {
        const Matrix a = random_rect(90, 90, g);
        Matrix diag(90, 90), one_column(90, 90);
        for (int k = 0; k < 90; ++k) diag(k, k) = cplx(1.0 + k, -0.5 * k);
        for (int k = 0; k < 90; ++k) one_column(k, 3) = cplx(0.1 * k, 1.0);
        CHECK(mkl::detail::sparse_for_gemm(diag) && mkl::detail::sparse_for_gemm(one_column));
        CHECK(!mkl::detail::sparse_for_gemm(random_rect(90, 90, g)));
        CHECK(same_bits(mkl::multiply(a, diag), a * diag) && same_bits(mkl::multiply(a, one_column), a * one_column));
    }
    // 空矩阵与大小不合
    CHECK(mkl::multiply(Matrix(0, 3), Matrix(3, 4)).rows() == 0 && mkl::multiply(Matrix(0, 3), Matrix(3, 4)).cols() == 4);
    {
        const Matrix z = mkl::multiply(Matrix(2, 0), Matrix(0, 3));
        CHECK(z.rows() == 2 && z.cols() == 3 && max_abs(z) == 0.0);
    }
    CHECK(throws_invalid([] { (void)mkl::multiply(Matrix(2, 3), Matrix(2, 3)); }));
    // 线性求解：与 stablercwa::solve 比，后向误差
    for (int n : {1, 2, 7, 64, 300})
        for (int k : {1, 5, n}) {
            const Matrix a = random_rect(n, n, g), b = random_rect(n, k, g);
            const Matrix x = mkl::solve(a, b), ref = stablercwa::solve(a, b);
            const double back = solve_backward_error(a, x, b);
            CHECK_LE(back, 64.0 * n * kMachEps);
            worst_backward = std::max(worst_backward, back);
            const double d = max_diff(x, ref) / max_abs(ref);
            CHECK_LE(d, 1e-9);
            worst_solve = std::max(worst_solve, d);
        }
    // 精确为零的主元：抛 std::runtime_error（求解器当作数值失败）
    for (int n : {1, 6, 200}) {
        Matrix a = random_rect(n, n, g);
        for (int i = 0; i < n; ++i) a(i, n / 2) = cplx{};                 // 有一列全为零
        bool singular = false;
        try {
            (void)mkl::solve(a, random_rect(n, 2, g));
        } catch (const std::runtime_error&) {
            singular = true;
        }
        CHECK(singular);
    }
    CHECK(throws_invalid([] { (void)mkl::solve(Matrix(2, 3), Matrix(2, 1)); }));
    CHECK(throws_invalid([] { (void)mkl::solve(Matrix::identity(3), Matrix(2, 1)); }));
    CHECK(mkl::solve(Matrix(0, 0), Matrix(0, 4)).cols() == 4);
    // 线程池里、几个宿主线程里同时调用
    {
        std::vector<Matrix> as, bs;
        for (int k = 0; k < 8; ++k) {
            as.push_back(random_rect(150, 150, g));
            bs.push_back(random_rect(150, 40, g));
        }
        std::vector<double> err(8, 1.0);
        set_num_threads(4);
        parallel_for(0, 8, [&](int k) {
            const Matrix x = mkl::solve(as[k], bs[k]);
            err[k] = std::max(solve_backward_error(as[k], x, bs[k]),
                              max_diff(mkl::multiply(as[k], x), bs[k]) / (max_abs(as[k]) * max_abs(x) * 150));
        });
        set_num_threads(1);
        std::vector<std::thread> threads;
        std::vector<double> host(4, 1.0);
        for (int t = 0; t < 4; ++t)
            threads.emplace_back([&, t] {
                const Matrix x = mkl::solve(as[t], bs[t]);
                host[static_cast<std::size_t>(t)] = solve_backward_error(as[t], x, bs[t]);
            });
        for (auto& th : threads) th.join();
        for (double e : err) CHECK_LE(e, 64.0 * 150 * kMachEps);
        for (double e : host) CHECK_LE(e, 64.0 * 150 * kMachEps);
    }
    // use_mkl 一次设好三个钩子
    {
        rcwa::Options o;
        rcwa::Options2D o2;
        mkl::use_mkl(o);
        mkl::use_mkl(o2, real_settings());
        CHECK(o.custom_eig && o.custom_multiply && o.custom_solve && o2.custom_eig && o2.custom_multiply && o2.custom_solve);
    }
    say("  乘法与内置的差最大 %.2e（相对 max|A|·max|B|·k）；求解与内置的差最大 %.2e，后向误差最大 %.2e；零主元报失败；稀疏的 B 交回内置\n",
        worst_mul, worst_solve, worst_backward);
}

// ---------------------------------------------------------------------------------------------
// B：一维 TE/TM 与默认（QR）比
static void test_planar() {
    say("[B] 一维 TE/TM：各级效率与 QR 的差\n");
    struct Case {
        const char* name;
        rcwa::Stack stack;
        double wavelength, theta;
        std::vector<int> orders;
        bool lossless;
    };
    examples::Mdg lossy;
    lossy.n_tooth = cplx(1.46, 0.08);
    const examples::Ridges ridges;
    const examples::Mdg mdg;
    const Case cases[] = {
        {"单周期光栅 + 20 层膜", examples::single_period(), mdg.wavelength, mdg.theta_deg, {20, 60, 150}, true},
        {"16 层梯形齿 + 20 层膜", examples::trapezoid(16), mdg.wavelength, mdg.theta_deg, {40}, true},
        {"有吸收的齿", examples::single_period(lossy), mdg.wavelength, mdg.theta_deg, {40}, false},
        {"四脊大周期 + 20 层膜", examples::ridges_on_mirror(), ridges.wavelength, ridges.theta_deg, {100}, true},
    };
    double worst_gap = 0.0, worst_energy = 0.0, worst_res = 0.0;
    for (const Case& c : cases)
        for (int M : c.orders)
            for (rcwa::Polarization pol : {rcwa::Polarization::TE, rcwa::Polarization::TM}) {
                const rcwa::Result base = rcwa::solve(c.stack, c.wavelength, c.theta, M, pol);
                rcwa::Options o = mkl_options();
                o.diagnostics = true;
                const rcwa::Result r = rcwa::solve(c.stack, c.wavelength, c.theta, M, pol, o);
                CHECK(r.eig_info == 0 && base.eig_info == 0);
                const double d = gap(base, r);
                CHECK_LE(d, 1e-9);
                worst_gap = std::max(worst_gap, d);
                CHECK_LE(r.max_eig_residual, 1e-13);
                worst_res = std::max(worst_res, r.max_eig_residual);
                if (c.lossless) {
                    const double e = std::abs(r.reflectance() + r.transmittance() - 1.0);
                    CHECK_LE(e, 1e-10);
                    worst_energy = std::max(worst_energy, e);
                }
                if (std::strcmp(c.name, "单周期光栅 + 20 层膜") == 0 && M == 150 && pol == rcwa::Polarization::TE) {
                    // README 那张表的 QR 一栏：第 +1 级（Littrow 级）反射效率 98.586942 %
                    const double q = 100.0 * base.DEr[M + 1], m = 100.0 * r.DEr[M + 1];
                    CHECK_LE(std::abs(q - 98.586942), 5e-7);
                    CHECK_LE(std::abs(q - m), 1e-7);                 // 百分数：效率差 1e-9
                    say("  README 的高阶算例（M = 150，TE）：Littrow 级效率 QR %.6f %%，MKL %.6f %%\n", q, m);
                }
            }
    say("  与 QR 的效率差最大 %.2e；无吸收时 |R + T − 1| 最大 %.2e；MKL 的本征残差最大 %.2e\n", worst_gap, worst_energy, worst_res);
}

// ---------------------------------------------------------------------------------------------
// C：重合本征值（均匀层也做本征分解，恰好正入射与恰好 Littrow 入射）
static void test_degenerate() {
    say("[C] 一维：均匀层也做本征分解，恰好正入射 / Littrow 入射的随机结构\n");
    std::mt19937_64 g(20261025);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    double worst_qr = 0.0, worst_mkl = 0.0;
    int cases = 0, mkl_over = 0, qr_over = 0;
    PathCount count;                                     // 结构都无吸收：打开 real_path，均匀层（严格的实矩阵）走 dgeev
    for (int k = 0; k < 60; ++k) {
        rcwa::Stack s;
        s.nI = 1.0;
        s.nII = 1.4 + 0.4 * U(g);
        s.period = 0.5 + U(g);
        const int layers = 1 + static_cast<int>(4 * U(g));
        const int film_at = static_cast<int>(layers * U(g));
        for (int l = 0; l < layers; ++l) {
            const double d = 0.05 + 0.4 * U(g);
            if (l == film_at || U(g) < 0.3) {
                s.layers.push_back(rcwa::Layer::uniform(d, 1.3 + U(g)));
            } else {
                const double a = s.period * (0.1 + 0.3 * U(g)), b = a + s.period * (0.2 + 0.4 * U(g));
                s.layers.push_back(rcwa::Layer::grating(d, {0.0, a, b, s.period}, {1.0, 1.3 + U(g), 1.0}));
            }
        }
        const double wavelength = 0.6 + 0.6 * U(g);
        const int M = 4 + k % 9;
        std::vector<double> angles{0.0};
        const double littrow = wavelength / (2.0 * s.period * s.nI.real());
        if (littrow < 0.95) angles.push_back(std::asin(littrow) * 180.0 / 3.14159265358979323846);
        for (double theta : angles)
            for (rcwa::Polarization pol : {rcwa::Polarization::TE, rcwa::Polarization::TM}) {
                const rcwa::Result fast = rcwa::solve(s, wavelength, theta, M, pol);
                rcwa::Options dense;
                dense.homogeneous_fast_path = false;
                const rcwa::Result dq = rcwa::solve(s, wavelength, theta, M, pol, dense);
                dense.custom_eig = counting_hook(count, real_settings());
                add_dense(dense);
                const rcwa::Result dm = rcwa::solve(s, wavelength, theta, M, pol, dense);
                const double gq = gap(fast, dq), gm = gap(fast, dm);
                // MKL 不应比 QR 差：要么在 1e-9 以内，要么不超过 QR 在同一算例上的差的 10 倍
                CHECK_LE(gm, std::max(1e-9, 10.0 * gq));
                worst_qr = std::max(worst_qr, gq);
                worst_mkl = std::max(worst_mkl, gm);
                qr_over += gq > 1e-9;
                mkl_over += gm > 1e-9;
                ++cases;
            }
    }
    CHECK(count.real.load() > 0);
    say("  60 个结构、%d 次求解：稠密路径与快速路径的效率差最大 QR %.2e、MKL %.2e；超过 1e-9 的 QR %d 个、MKL %d 个；"
        "%d 个层矩阵里 %d 个走了 dgeev\n", cases, worst_qr, worst_mkl, qr_over, mkl_over, count.total.load(), count.real.load());
}

// ---------------------------------------------------------------------------------------------
// D：圆锥入射
static void test_conical() {
    say("[D] 圆锥入射：四个通道的效率与 QR 的差\n");
    const examples::Mdg mdg;
    struct Case {
        const char* name;
        rcwa::Stack stack;
        double phi;
        int M;
    };
    const Case cases[] = {
        {"单周期光栅，φ = 25°", examples::single_period(), 25.0, 20},
        {"单周期光栅，φ = 25°", examples::single_period(), 25.0, 60},
        {"8 层梯形齿，φ = 40°", examples::trapezoid(8), 40.0, 30},
        {"单周期光栅，φ = 1e-4°（ky 接近零）", examples::single_period(), 1e-4, 20},
    };
    double worst = 0.0;
    for (const Case& c : cases) {
        const rcwa::ConicalResult base = rcwa::solve_conical(c.stack, mdg.wavelength, mdg.theta_deg, c.phi, c.M);
        const rcwa::ConicalResult r = rcwa::solve_conical(c.stack, mdg.wavelength, mdg.theta_deg, c.phi, c.M, mkl_options());
        CHECK(r.eig_info == 0 && base.eig_info == 0);
        const double d = gap(base, r);
        CHECK_LE(d, 1e-9);
        worst = std::max(worst, d);
        say("  %-36s M = %3d：差 %.2e\n", c.name, c.M, d);
    }
    say("  最大差 %.2e\n", worst);
}

// ---------------------------------------------------------------------------------------------
// E：二维，与冻结的 Python 参照比
struct Cmp {
    double maxdiff = 0.0, excess = -1e300;
    void add(cplx a, const double b[2], double atol, double rtol) {
        const cplx ref{b[0], b[1]};
        const double d = std::abs(a - ref);
        if (!(d <= maxdiff)) maxdiff = std::isnan(d) ? std::numeric_limits<double>::infinity() : d;
        const double ex = std::isnan(d) ? 1e300 : d - (atol + rtol * std::abs(ref));
        if (ex > excess) excess = ex;
    }
    void add(double a, double b, double atol, double rtol) {
        const double bb[2] = {b, 0.0};
        add(cplx{a, 0.0}, bb, atol, rtol);
    }
    bool ok() const { return excess <= 0.0; }
};

static rcwa::Result2D run_case(const Rcwa2dRefCase& c, const rcwa::Options2D& base) {
    rcwa::Options2D options = base;
    options.factorization = c.laurent ? rcwa::Factorization2D::Laurent : rcwa::Factorization2D::Li;
    const rcwa::Jones2D jones = {cplx{c.jones[0], c.jones[1]}, cplx{c.jones[2], c.jones[3]}};
    return rcwa::solve2d(rcwa2d_ref_stack(c.id), c.wavelength, c.theta_deg, c.phi_deg, c.M1, c.M2, jones, options);
}

static double compare_far(const Rcwa2dRefCase& c, const rcwa::Result2D& r) {
    CHECK(r.status == rcwa::Status2D::Ok && r.N == c.count);
    if (r.status != rcwa::Status2D::Ok || r.N != c.count) return std::numeric_limits<double>::infinity();
    Cmp cmp;
    for (int i = 0; i < c.count; ++i) {
        const Rcwa2dRefOrder& o = c.orders[i];
        const std::size_t k = static_cast<std::size_t>(i);
        const double atol = c.far_atol, rtol = c.far_rtol;
        cmp.add(r.Ex_r[k], o.Ex_r, atol, rtol); cmp.add(r.Ey_r[k], o.Ey_r, atol, rtol); cmp.add(r.Ez_r[k], o.Ez_r, atol, rtol);
        cmp.add(r.Ex_t[k], o.Ex_t, atol, rtol); cmp.add(r.Ey_t[k], o.Ey_t, atol, rtol); cmp.add(r.Ez_t[k], o.Ez_t, atol, rtol);
        cmp.add(r.Rs[k], o.Rs, atol, rtol); cmp.add(r.Rp[k], o.Rp, atol, rtol);
        cmp.add(r.Ts[k], o.Ts, atol, rtol); cmp.add(r.Tp[k], o.Tp, atol, rtol);
        cmp.add(r.DEr[k], o.DEr, atol, rtol);
        cmp.add(r.DEt[k], o.DEt, atol, rtol);
    }
    CHECK(cmp.ok());
    if (!cmp.ok()) say("  %2d %-26s 超出参照自带的容差：max |a − b| = %.2e\n", c.id, c.name, cmp.maxdiff);
    return cmp.maxdiff;
}

static void test_2d() {
    say("[E] 二维：%d 个冻结的 Python 参照\n", kRcwa2dRefCaseCount);
    PathCount count;                                     // 只数串行的一遍
    rcwa::Options2D o;
    o.custom_eig = counting_hook(count, real_settings());
    add_dense(o);
    rcwa::Options2D par = o;
    par.custom_eig = mkl::eig_hook(real_settings());
    par.parallel_layers = true;
    double worst = 0.0;
    int ok = 0, rayleigh = 0;
    for (int i = 0; i < kRcwa2dRefCaseCount; ++i) {
        const Rcwa2dRefCase& c = kRcwa2dRefCases[i];
        const rcwa::Result2D r = run_case(c, o);
        if (c.status == 1) {
            CHECK(r.status == rcwa::Status2D::RayleighAnomaly);
            ++rayleigh;
            continue;
        }
        worst = std::max(worst, compare_far(c, r));
        set_num_threads(4);
        worst = std::max(worst, compare_far(c, run_case(c, par)));
        set_num_threads(1);
        ++ok;
    }
    say("  %d 个成功算例（串行与并行层各一遍）全部在参照自带的容差内，max |a − b| = %.2e；%d 个 Rayleigh 失败例照样报告失败\n", ok, worst,
        rayleigh);
    CHECK(count.real.load() > 0 && count.real.load() < count.total.load());   // 居中对称的算例走 dgeev，其余走 zgeev
    say("  打开 real_path：%d 个层矩阵里 %d 个按实矩阵走了 dgeev（单元居中对称、无吸收的结构）\n", count.total.load(),
        count.real.load());
}

// ---------------------------------------------------------------------------------------------
// F：近场
static void test_fields() {
    say("[F] 近场：场分布与 QR 的差（相对整幅场的最大模）\n");
    const examples::Mdg mdg;
    const rcwa::Stack stack = examples::single_period();
    const RealVector zs{-0.3, 0.0, 0.19, 0.38, 0.9, 3.0, 6.0};
    for (rcwa::Polarization pol : {rcwa::Polarization::TE, rcwa::Polarization::TM}) {
        const auto a = rcwa::solve_with_fields(stack, mdg.wavelength, mdg.theta_deg, 30, pol);
        const auto b = rcwa::solve_with_fields(stack, mdg.wavelength, mdg.theta_deg, 30, pol, mkl_options());
        const double d = field_gap(rcwa::field_map(a, zs, 64).components, rcwa::field_map(b, zs, 64).components);
        CHECK_LE(d, 1e-9);
        say("  一维 %s，M = 30：%.2e\n", pol == rcwa::Polarization::TE ? "TE" : "TM", d);
    }
    {
        const auto a = rcwa::solve_conical_with_fields(stack, mdg.wavelength, mdg.theta_deg, 25.0, 20);
        const auto b = rcwa::solve_conical_with_fields(stack, mdg.wavelength, mdg.theta_deg, 25.0, 20, mkl_options());
        const rcwa::JonesVector jones{{cplx(0.6, 0.0), cplx(0.0, 0.8)}};
        const double d = field_gap(rcwa::field_map(a, zs, 64, 0.0, jones).components, rcwa::field_map(b, zs, 64, 0.0, jones).components);
        CHECK_LE(d, 1e-9);
        say("  圆锥入射 φ = 25°，M = 20：%.2e\n", d);
    }
    int done = 0;
    for (int i = 0; i < kRcwa2dRefCaseCount && done < 3; ++i) {
        const Rcwa2dRefCase& c = kRcwa2dRefCases[i];
        if (c.status != 0 || c.count < 9) continue;
        const rcwa::Stack2D stack2 = rcwa2d_ref_stack(c.id);
        bool patterned = false;
        double depth = 0.0;
        for (const auto& layer : stack2.layers) {
            patterned = patterned || !layer.homogeneous;
            depth += layer.d;
        }
        if (!patterned) continue;
        rcwa::Options2D qr, mk;
        qr.factorization = mk.factorization = c.laurent ? rcwa::Factorization2D::Laurent : rcwa::Factorization2D::Li;
        mk.custom_eig = mkl::eig_hook();
        add_dense(mk);
        const rcwa::Jones2D jones = {cplx{c.jones[0], c.jones[1]}, cplx{c.jones[2], c.jones[3]}};
        const auto a = rcwa::solve2d_with_fields(stack2, c.wavelength, c.theta_deg, c.phi_deg, c.M1, c.M2, jones, qr);
        const auto b = rcwa::solve2d_with_fields(stack2, c.wavelength, c.theta_deg, c.phi_deg, c.M1, c.M2, jones, mk);
        const RealVector z2{-0.2, 0.0, 0.5 * depth, depth, depth + 0.3};
        for (rcwa::SlicePlane plane : {rcwa::SlicePlane::XZ, rcwa::SlicePlane::YZ}) {
            const double d = field_gap(rcwa::field_slice(a.fields, z2, plane, 48, 0.1).components,
                                       rcwa::field_slice(b.fields, z2, plane, 48, 0.1).components);
            CHECK_LE(d, 1e-9);
            say("  二维 %-26s %s 切面：%.2e\n", c.name, rcwa::slice_plane_name(plane), d);
        }
        ++done;
    }
    CHECK(done == 3);
}

// ---------------------------------------------------------------------------------------------
// G：parallel_layers、sweep 与 MKL 后端同用
static void test_parallel() {
    say("[G] parallel_layers 与 sweep\n");
    const examples::Mdg mdg;
    const rcwa::Stack stack = examples::trapezoid(8);
    const rcwa::Options o = mkl_options();
    const rcwa::Result serial = rcwa::solve(stack, mdg.wavelength, mdg.theta_deg, 30, rcwa::Polarization::TE, o);
    rcwa::Options par = o;
    par.parallel_layers = true;
    set_num_threads(4);
    const rcwa::Result parallel = rcwa::solve(stack, mdg.wavelength, mdg.theta_deg, 30, rcwa::Polarization::TE, par);
    const double d_layers = gap(serial, parallel);
    CHECK_LE(d_layers, 1e-12);
    std::vector<rcwa::Point> points;
    for (int k = 0; k < 6; ++k) points.push_back({mdg.wavelength, 50.0 + 4.0 * k});
    const std::vector<rcwa::Result> swept = rcwa::sweep(stack, points, 30, rcwa::Polarization::TM, o);
    set_num_threads(1);
    double d_sweep = 0.0;
    for (std::size_t k = 0; k < points.size(); ++k)
        d_sweep = std::max(d_sweep, gap(swept[k], rcwa::solve(stack, points[k].wavelength, points[k].theta_deg, 30, rcwa::Polarization::TM, o)));
    CHECK_LE(d_sweep, 1e-12);
    say("  parallel_layers（4 个线程）与串行的差 %.2e；sweep（4 个线程）与逐点求解的差 %.2e\n", d_layers, d_sweep);
}

// ---------------------------------------------------------------------------------------------
// H：对称、无吸收的结构：层矩阵在舍入误差以内是实的，走 dgeev
static rcwa::Stack symmetric_stack(const std::vector<double>& edges, const std::vector<cplx>& values) {
    rcwa::Stack s;
    s.nI = 1.0;
    s.nII = 1.46;
    s.period = 0.574;
    s.layers.push_back(rcwa::Layer::grating(0.38, edges, values));
    s.layers.push_back(rcwa::Layer::uniform(0.115, 1.96));
    s.layers.push_back(rcwa::Layer::grating(0.2, edges, values));
    return s;
}

static void test_symmetric() {
    say("[H] 对称、无吸收的结构，打开 real_path：层矩阵在舍入误差以内是实的，走 dgeev；与 QR、与默认设定（zgeev）比\n");
    const double L = 0.574;
    struct Case {
        const char* name;
        std::vector<double> edges;
        std::vector<cplx> values;
        bool real;                                   // 层矩阵应当被当作实的
    };
    const Case cases[] = {
        {"齿在单元中心", {0.0, 0.2, L - 0.2, L}, {1.0, 1.46, 1.0}, true},
        {"齿在原点", {-0.0861, 0.0861, L - 0.0861}, {1.46, 1.0}, true},
        {"高折射率的齿（n = 3.5）", {0.0, 0.13, L - 0.13, L}, {1.0, 3.5, 1.0}, true},
        {"三个齿、相对原点对称", {-0.25, -0.15, -0.05, 0.05, 0.15, 0.25, L - 0.25}, {1.46, 1.0, 1.96, 1.0, 1.46, 1.0}, true},
        {"齿在单元中心，有吸收（k = 0.01）", {0.0, 0.2, L - 0.2, L}, {1.0, cplx(1.46, 0.01), 1.0}, false},
        {"齿在单元中心，吸收只有 k = 1e-9", {0.0, 0.2, L - 0.2, L}, {1.0, cplx(1.46, 1e-9), 1.0}, false},
        {"齿在单元中心，吸收只有 k = 1e-12", {0.0, 0.2, L - 0.2, L}, {1.0, cplx(1.46, 1e-12), 1.0}, false},
        {"齿在单元中心，k = 1e-15（与舍入误差分不开）", {0.0, 0.2, L - 0.2, L}, {1.0, cplx(1.46, 1e-15), 1.0}, true},
        {"不对称的齿", {0.0, 0.1722, L}, {1.46, 1.0}, false},
    };
    double worst_qr = 0.0, worst_paths = 0.0, worst_energy = 0.0, worst_res = 0.0, worst_conical = 0.0;
    for (const Case& c : cases) {
        const rcwa::Stack s = symmetric_stack(c.edges, c.values);
        bool lossless = true;
        for (const cplx& v : c.values) lossless = lossless && v.imag() == 0.0;
        bool path_ok = true;
        for (int M : {20, 60, 150})
            for (double theta : {0.0, 20.0})
                for (rcwa::Polarization pol : {rcwa::Polarization::TE, rcwa::Polarization::TM}) {
                    const rcwa::Result base = rcwa::solve(s, 1.053, theta, M, pol);
                    PathCount count, zcount;
                    rcwa::Options o;
                    o.custom_eig = counting_hook(count, real_settings());
                    o.diagnostics = true;
                    const rcwa::Result r = rcwa::solve(s, 1.053, theta, M, pol, o);
                    rcwa::Options z;                 // 默认设定：一律 zgeev
                    z.custom_eig = counting_hook(zcount);
                    const rcwa::Result rz = rcwa::solve(s, 1.053, theta, M, pol, z);
                    CHECK(base.eig_info == 0 && r.eig_info == 0 && rz.eig_info == 0);
                    path_ok = path_ok && count.total.load() > 0 && count.real.load() == (c.real ? count.total.load() : 0) &&
                              zcount.real.load() == 0;
                    const double dq = gap(base, r), dz = gap(rz, r);
                    CHECK_LE(dq, 1e-9);
                    CHECK_LE(dz, 1e-9);
                    CHECK_LE(r.max_eig_residual, 1e-13);   // 对原来的（带着舍入虚部的）层矩阵的残差
                    worst_qr = std::max(worst_qr, dq);
                    worst_paths = std::max(worst_paths, dz);
                    worst_res = std::max(worst_res, r.max_eig_residual);
                    if (lossless) {
                        const double e = std::abs(r.reflectance() + r.transmittance() - 1.0);
                        CHECK_LE(e, 1e-10);
                        worst_energy = std::max(worst_energy, e);
                    }
                }
        for (double phi : {25.0, 1e-4}) {            // 圆锥入射：两次 N 阶本征分解（A 与 B·C）都应走同一条路
            const rcwa::ConicalResult base = rcwa::solve_conical(s, 1.053, 20.0, phi, 30);
            PathCount count;
            rcwa::Options o;
            o.custom_eig = counting_hook(count, real_settings());
            const rcwa::ConicalResult r = rcwa::solve_conical(s, 1.053, 20.0, phi, 30, o);
            CHECK(base.eig_info == 0 && r.eig_info == 0);
            path_ok = path_ok && count.total.load() > 0 && count.real.load() == (c.real ? count.total.load() : 0);
            const double d = gap(base, r);
            CHECK_LE(d, 1e-9);
            worst_conical = std::max(worst_conical, d);
        }
        CHECK(path_ok);
        say("  %-36s %s\n", c.name, path_ok ? (c.real ? "走 dgeev" : "走 zgeev") : "走的路与预期不符");
    }
    say("  一维 TE/TM（M 到 150，正入射与 20°）：与 QR 的效率差最大 %.2e，与默认设定的差最大 %.2e；无吸收时 |R + T − 1| 最大 %.2e；\n"
        "  本征残差（对带着舍入虚部的原矩阵）最大 %.2e。圆锥入射（φ = 25°、1e-4°）与 QR 的差最大 %.2e\n",
        worst_qr, worst_paths, worst_energy, worst_res, worst_conical);

    // 高 Q 共振（对称保护的连续谱束缚态附近，θ = 0.005°）上 k = 1e-13 的弱吸收：吸收掉的约 4.5e-6 远大于舍入误差。
    // 层矩阵的虚部却与舍入误差分不开：默认设定（zgeev）照样算对；打开 real_path 时会被当成无吸收，所以 real_path 默认关闭。
    {
        const double Lb = 0.6, k = 1e-13;
        rcwa::Stack b;
        b.nI = 1.0;
        b.nII = 1.0;
        b.period = Lb;
        b.layers.push_back(rcwa::Layer::grating(0.25, {0.0, 0.15, Lb - 0.15, Lb}, {cplx(2.0, k), cplx(3.0, k), cplx(2.0, k)}));
        const double wavelength = 1.025510284, theta = 0.005;
        const rcwa::Result q = rcwa::solve(b, wavelength, theta, 15, rcwa::Polarization::TM);
        PathCount dcount, rcount;
        rcwa::Options od, orr;
        od.custom_eig = counting_hook(dcount);
        orr.custom_eig = counting_hook(rcount, real_settings());
        const rcwa::Result m = rcwa::solve(b, wavelength, theta, 15, rcwa::Polarization::TM, od);
        const rcwa::Result mr = rcwa::solve(b, wavelength, theta, 15, rcwa::Polarization::TM, orr);
        const double aq = 1.0 - q.reflectance() - q.transmittance(), am = 1.0 - m.reflectance() - m.transmittance();
        const double ar = 1.0 - mr.reflectance() - mr.transmittance();
        CHECK(q.eig_info == 0 && m.eig_info == 0 && dcount.real.load() == 0);
        // 共振把舍入误差也放大了：QR 与 zgeev 之间本来就差 1e-8 ~ 1e-7（随 LAPACK 的实现而变），吸收量要对得上
        CHECK_LE(gap(q, m), 1e-6);
        CHECK_LE(std::abs(am - aq), 0.05 * aq);
        CHECK(rcount.real.load() == rcount.total.load());          // 记下局限：打开 real_path 时这一层被当成实矩阵
        say("  高 Q 共振上 k = 1e-13 的吸收：QR 吸收 %.2e；默认设定吸收 %.2e、效率与 QR 差 %.2e；打开 real_path 时这一层%s，"
            "吸收 %.2e、效率差 %.2e\n", aq, am, gap(q, m), rcount.real.load() == rcount.total.load() ? "被当成实矩阵" : "走 zgeev", ar,
            gap(q, mr));
    }

    rcwa::Options om;                                // 近场：打开 real_path，对称结构走 dgeev
    om.custom_eig = mkl::eig_hook(real_settings());
    const rcwa::Stack s = symmetric_stack(cases[1].edges, cases[1].values);
    const RealVector zs{-0.3, 0.0, 0.19, 0.38, 0.6, 0.8, 2.0};
    for (rcwa::Polarization pol : {rcwa::Polarization::TE, rcwa::Polarization::TM}) {
        const auto a = rcwa::solve_with_fields(s, 1.053, 20.0, 30, pol);
        const auto b = rcwa::solve_with_fields(s, 1.053, 20.0, 30, pol, om);
        const double d = field_gap(rcwa::field_map(a, zs, 64).components, rcwa::field_map(b, zs, 64).components);
        CHECK_LE(d, 1e-9);
        say("  近场（齿在原点，%s，M = 30）与 QR 的差 %.2e\n", pol == rcwa::Polarization::TE ? "TE" : "TM", d);
    }
}

// ---------------------------------------------------------------------------------------------
// J：JSON 入口的 "backend": "mkl"
// 两段应答的结构（键、次序、类型、长度）要相同，数逐个比；timing 不比
static void json_gap(const json::Value& a, const json::Value& b, double& worst, bool& same) {
    if (a.type() != b.type() || a.size() != b.size()) {
        same = false;
        return;
    }
    if (a.is_number()) {
        const double d = std::abs(a.as_double() - b.as_double());
        if (!(d <= worst)) worst = d;
        return;
    }
    if (a.is_string()) same = same && a.as_string() == b.as_string();
    if (a.is_bool()) same = same && a.as_bool() == b.as_bool();
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a.is_object()) {
            if (a.key(i) != b.key(i)) {
                same = false;
                continue;
            }
            if (a.key(i) == "timing") continue;
        }
        json_gap(a.at(i), b.at(i), worst, same);
    }
}

static std::string with_backend(std::string request, const char* backend) {
    const std::size_t at = request.find("BACKEND");
    request.replace(at, 7, backend);
    return request;
}

// 请求有错：ok 为 false，where 是给定的字段，error 里有给定的一段
static bool rejected(const std::string& request, const char* where, const char* part = "") {
    const rcwa::JsonReply r = rcwa::solve_json_reply(request);
    if (r.ok) return false;
    const json::Value v = json::parse(r.text);
    return v.at("where").as_string() == where && v.at("error").as_string().find(part) != std::string::npos;
}

static void compare_json(const char* label, const std::string& request) {
    const rcwa::JsonReply q = rcwa::solve_json_reply(with_backend(request, "builtin"));
    const rcwa::JsonReply m = rcwa::solve_json_reply(with_backend(request, "mkl"));
    CHECK(q.ok && m.ok);
    if (!q.ok || !m.ok) {
        say("  %s：%s\n", label, (q.ok ? m.text : q.text).substr(0, 300).c_str());
        return;
    }
    double worst = 0.0;
    bool same = true;
    json_gap(json::parse(q.text), json::parse(m.text), worst, same);
    CHECK(same);
    CHECK_LE(worst, 1e-9);
    // 三个钩子都设上了
    const rcwa::json_detail::Request rq = rcwa::json_detail::read_request(json::parse(with_backend(request, "mkl")));
    const bool hooks = rq.periodic2d ? rq.options2d.custom_eig && rq.options2d.custom_multiply && rq.options2d.custom_solve
                                     : rq.options.custom_eig && rq.options.custom_multiply && rq.options.custom_solve;
    CHECK(hooks);
    say("  %-28s 应答的结构与 \"builtin\" 相同，数值最大差 %.2e%s\n", label, worst, hooks ? "" : "（钩子没有设全）");
}

static void test_json() {
    say("[J] JSON 入口的 \"backend\": \"mkl\"\n");
    const std::string planar = R"({"mode": "planar",
        "stack": {"nI": 1.0, "nII": 1.46, "period": 0.574,
                  "layers": [{"d": 0.38, "edges": [0.0, 0.1722, 0.574], "n_values": [1.46, 1.0]},
                             {"d": 0.2, "edges": [-0.0861, 0.0861, 0.4879], "n_values": [1.46, 1.0]},
                             {"d": 0.115, "n": [1.96, 0.0]}]},
        "M": 40, "polarization": "TM", POINTS,
        "options": {"backend": "BACKEND", "threads": 4, "diagnostics": true}})";
    const auto points = [](std::string request, const char* text) {
        const std::size_t at = request.find("POINTS");
        request.replace(at, 6, text);
        return request;
    };
    compare_json("planar，一个点", points(planar, R"("wavelength": 1.053, "theta_deg": 66.36)"));
    compare_json("planar，三个点（sweep）",
                 points(planar, R"("points": [{"wavelength": 1.053, "theta_deg": 66.36}, {"wavelength": 1.06, "theta_deg": 60.0},
                                              {"wavelength": 0.9, "theta_deg": 10.0}])"));
    compare_json("conical", R"({"mode": "conical",
        "stack": {"nI": 1.0, "nII": 1.46, "period": 0.574,
                  "layers": [{"d": 0.38, "edges": [0.0, 0.1722, 0.574], "n_values": [1.46, 1.0]},
                             {"d": 0.115, "n": [1.96, 0.0]}]},
        "M": 30, "points": [{"wavelength": 1.053, "theta_deg": 66.36, "phi_deg": 17.0},
                            {"wavelength": 1.053, "theta_deg": 30.0, "phi_deg": 0.0001}],
        "input": {"psi_deg": 45.0, "delta_deg": 90.0},
        "options": {"backend": "BACKEND", "eig": "qr", "linear": "lu", "threads": 4, "parallel_layers": true}})");
    const std::string periodic2d = R"({"mode": "periodic2d",
        "stack": {"nI": 1.0, "nII": 1.46, "Lx": 0.8, "Ly": 0.8,
                  "layers": [{"d": 0.3, "y_edges": [0.0, 0.25, 0.55, 0.8],
                              "x_profiles": [{"edges": [0.0, 0.8], "n_values": [1.0]},
                                             {"edges": [0.0, 0.25, 0.55, 0.8], "n_values": [1.0, 1.8, 1.0]},
                                             {"edges": [0.0, 0.8], "n_values": [1.0]}]},
                             {"d": 0.2, "y_edges": [0.0, 0.3, 0.8],
                              "x_profiles": [{"edges": [0.0, 0.5, 0.8], "n_values": [1.0, [2.0, 0.05]]},
                                             {"edges": [0.0, 0.8], "n_values": [1.0]}]},
                             {"d": 0.115, "n": [1.96, 0.0]}]},
        "M": 5, "My": 4,
        "points": [{"wavelength": 1.053, "theta_deg": 42.0, "phi_deg": -25.0}, {"wavelength": 0.95, "theta_deg": 0.0, "phi_deg": 0.0}],
        "input": {"psi_deg": 30.0, "delta_deg": 0.0},
        "options": {"backend": "BACKEND", "factorization": "li", "threads": 4}})";
    compare_json("periodic2d（居中方柱 + 有吸收的偏心块）", periodic2d);
    std::string lossless2d = periodic2d;                 // 去掉吸收：各层无吸收，real_path 打开，居中方柱那层走 dgeev
    lossless2d.replace(lossless2d.find("[2.0, 0.05]"), 11, "2.0");
    compare_json("periodic2d（无吸收）", lossless2d);
    CHECK(num_threads() == 1);                       // solve_json 调用结束时把线程数改回原值

    // 按折射率决定 real_path：各层的折射率都是实数才打开（出射介质不进层矩阵，不算）
    rcwa::Stack ls = symmetric_stack({0.0, 0.2, 0.374, 0.574}, {1.0, 1.46, 1.0});
    CHECK(rcwa::json_detail::layers_lossless(ls));
    ls.nII = cplx(1.46, 0.3);
    CHECK(rcwa::json_detail::layers_lossless(ls));
    ls.layers[1] = rcwa::Layer::uniform(0.1, cplx(1.96, 1e-15));
    CHECK(!rcwa::json_detail::layers_lossless(ls));
    ls.layers[1] = rcwa::Layer::uniform(0.1, 1.96);
    ls.layers[2].n_values[1] = cplx(1.46, 1e-14);
    CHECK(!rcwa::json_detail::layers_lossless(ls));
    rcwa::Stack2D l2 = rcwa2d_ref_stack(9);              // pillar_sq：无吸收
    CHECK(rcwa::json_detail::layers_lossless(l2));
    l2.layers[0].x_profiles[1].n_values[1] = cplx(1.8, 1e-14);
    CHECK(!rcwa::json_detail::layers_lossless(l2));

    // 请求错误：backend 写错；"mkl" 与内置迭代、内置解法的选项同用；MKL 不再用 eig 选
    const std::string one = points(planar, R"("wavelength": 1.053, "theta_deg": 66.36)");
    const auto options = [&](const std::string& request, const char* text) {
        std::string out = request;
        const std::size_t at = out.find("\"options\"");
        out.replace(at, out.find('}', at) + 1 - at, text);
        return out;
    };
    CHECK(rejected(with_backend(one, "MKL"), "options.backend", "builtin"));
    CHECK(rejected(options(one, R"("options": {"backend": "mkl", "eig": "lr"})"), "options.eig", "backend"));
    CHECK(rejected(options(one, R"("options": {"eig": "lr", "backend": "mkl"})"), "options.eig", "backend"));
    CHECK(rejected(options(one, R"("options": {"backend": "mkl", "linear": "qr"})"), "options.linear", "backend"));
    CHECK(rejected(options(one, R"("options": {"backend": "mkl", "linear": "lu_nopivot"})"), "options.linear", "backend"));
    CHECK(rejected(options(one, R"("options": {"eig": "mkl"})"), "options.eig", "qr"));
    CHECK(rejected(options(periodic2d, R"("options": {"backend": "mkl", "eig": "lr"})"), "options.eig", "backend"));
    CHECK(rcwa::solve_json_reply(options(one, R"("options": {"backend": "mkl", "eig": "qr", "linear": "lu"})")).ok);
    CHECK(rcwa::solve_json_reply(options(one, R"("options": {"backend": "builtin", "eig": "lr", "linear": "qr"})")).ok);
}

// ---------------------------------------------------------------------------------------------
// K：奇异与 Rayleigh 异常点，三个钩子都交给 MKL（与 tests/test_rcwa1d.cpp 里内置求解的同名检查对应）
static bool all_nan(const rcwa::Result& r) {
    if (r.R.empty() || r.R.size() != r.T.size() || r.DEr.size() != r.R.size() || r.DEt.size() != r.R.size()) return false;
    for (std::size_t j = 0; j < r.R.size(); ++j)
        if (!std::isnan(r.R[j].real()) || !std::isnan(r.T[j].real()) || !std::isnan(r.DEr[j]) || !std::isnan(r.DEt[j]))
            return false;
    return true;
}
static bool all_finite(const rcwa::Result& r) {
    bool ok = !r.R.empty();
    for (std::size_t j = 0; ok && j < r.R.size(); ++j)
        ok = std::isfinite(r.R[j].real()) && std::isfinite(r.R[j].imag()) && std::isfinite(r.T[j].real()) &&
             std::isfinite(r.T[j].imag()) && std::isfinite(r.DEr[j]) && std::isfinite(r.DEt[j]);
    return ok;
}
static double energy_defect(const rcwa::Result& r) { return std::abs(r.reflectance() + r.transmittance() - 1.0); }

static void test_singular() {
    say("[K] 奇异与 Rayleigh 异常点：三个钩子都交给 MKL\n");
    rcwa::Options full;
    mkl::use_mkl(full);
    // (1) 本征向量全为零：W 奇异，按数值失败处理
    {
        rcwa::Options o = full;
        o.custom_eig = [](const Matrix& a) {
            EigResult r = mkl::eig(a);
            r.vectors = Matrix(a.rows(), a.cols());
            return r;
        };
        const rcwa::Result r = rcwa::solve(examples::single_period(), 1.053, 66.36, 20, rcwa::Polarization::TE, o);
        CHECK(all_nan(r) && r.eig_info == 0);
    }
    // (2) λ = Λ、正入射的光栅：入射介质一侧 ±1 级恰好掠射，照常求解；出射介质（n = 2）一侧 ±2 级恰好掠射、又紧挨光栅层，
    //     界面方程组奇异，按失败处理
    const double p = 0.5747, f = 0.30;
    rcwa::Stack g;
    g.nI = 1.0;
    g.nII = 1.45;
    g.period = p;
    g.layers.push_back(rcwa::Layer::grating(0.38, {-f * p / 2, f * p / 2, p - f * p / 2}, {cplx(1.45), cplx(1.0)}));
    int failures_match = 0, failures_total = 0;
    for (rcwa::Polarization pol : {rcwa::Polarization::TE, rcwa::Polarization::TM}) {
        const rcwa::Result r = rcwa::solve(g, g.period, 0.0, 8, pol, full);
        CHECK(all_finite(r));
        CHECK_LE(energy_defect(r), 1e-10);
        CHECK(r.DEr[8 + 1] == 0.0 && r.DEr[8 - 1] == 0.0);
        rcwa::Stack bad = g;
        bad.nII = 2.0;
        const bool builtin_failed = all_nan(rcwa::solve(bad, bad.period, 0.0, 8, pol));
        const rcwa::Result rb = rcwa::solve(bad, bad.period, 0.0, 8, pol, full);
        ++failures_total;
        if (all_nan(rb) == builtin_failed) ++failures_match;
        CHECK(all_nan(rb) == builtin_failed);
        CHECK(all_nan(rb) || all_finite(rb));                         // 不能是一部分 NaN、一部分有限
    }
    CHECK(all_nan(rcwa::solve([&] {
                                  rcwa::Stack b = g;
                                  b.nII = 2.0;
                                  return b;
                              }(),
                              p, 0.0, 8, rcwa::Polarization::TE, full)));
    // (3) 有限均匀层恰好截止（精确零根，线性场分支），与光栅耦合：照常求解、能量守恒，与内置的一致
    const rcwa::Layer grating = rcwa::Layer::grating(0.3, {0.0, 0.3, 1.0}, {1.5, 1.0});
    const auto make = [](std::vector<rcwa::Layer> layers, cplx nII) {
        rcwa::Stack s;
        s.nI = 1.0;
        s.nII = nII;
        s.period = 1.0;
        s.layers = std::move(layers);
        return s;
    };
    const rcwa::Stack cutoff_cases[] = {
        make({rcwa::Layer::uniform(0.2, 2.0), grating}, 1.5),
        make({grating, rcwa::Layer::uniform(0.2, 2.0)}, 1.5),
        make({grating, rcwa::Layer::uniform(0.1, 1.7), rcwa::Layer::uniform(0.2, 2.0), rcwa::Layer::uniform(0.1, 1.7)}, 1.5),
        make({rcwa::Layer::uniform(0.0, 2.0), grating}, 1.5),
    };
    double worst_cutoff = 0.0;
    for (const rcwa::Stack& s : cutoff_cases)
        for (rcwa::Polarization pol : {rcwa::Polarization::TE, rcwa::Polarization::TM}) {
            const rcwa::Result r = rcwa::solve(s, 1.0, 0.0, 6, pol, full), q = rcwa::solve(s, 1.0, 0.0, 6, pol);
            CHECK(r.eig_info == 0 && all_finite(r));
            CHECK_LE(energy_defect(r), 2e-11);
            const double d = gap(q, r);
            CHECK_LE(d, 1e-9);
            worst_cutoff = std::max(worst_cutoff, d);
        }
    // (4) 扫描里有一个点落在异常点上（出射介质恰好掠射，上面紧挨光栅层）：只有那个点是 NaN
    {
        const rcwa::Stack s = make({grating}, 2.0);
        const std::vector<rcwa::Point> points{{1.01, 0.0}, {1.0, 0.0}, {0.93, 12.0}, {1.0, 1e-3}};
        set_num_threads(3);
        const std::vector<rcwa::Result> swept = rcwa::sweep(s, points, 6, rcwa::Polarization::TM, full);
        set_num_threads(1);
        bool ok = swept.size() == points.size() && all_nan(swept[1]);
        for (std::size_t i = 0; ok && i < points.size(); ++i)
            if (i != 1) ok = all_finite(swept[i]) && energy_defect(swept[i]) < 1e-10;
        CHECK(ok);
    }
    // (5) 极端的折射率：要么全部失败，要么全部有限、能量守恒
    int extreme_failed = 0, extreme_total = 0;
    for (rcwa::Polarization pol : {rcwa::Polarization::TE, rcwa::Polarization::TM})
        for (double extreme : {1e300, 1e160, 1e-160, 1e-300})
            for (const rcwa::Stack& s : {make({grating}, extreme), make({}, extreme), make({rcwa::Layer::uniform(0.2, 1.7)}, extreme),
                                         make({rcwa::Layer::uniform(0.2, extreme)}, 1.5)})
                for (int m : {0, 3}) {
                    const rcwa::Result r = rcwa::solve(s, 0.9, 20.0, m, pol, full);
                    CHECK(all_nan(r) || (all_finite(r) && energy_defect(r) < 1e-9));
                    extreme_failed += all_nan(r);
                    ++extreme_total;
                }
    say("  全零本征向量、出射介质掠射（%d/%d 与内置的一样报失败）都报数值失败；入射侧掠射、截止膜照常求解（与内置的差最大 %.2e）；\n"
        "  扫描里只有异常点是 NaN；极端折射率 %d 个里 %d 个报失败，其余守恒\n",
        failures_match, failures_total, worst_cutoff, extreme_total, extreme_failed);
}

int main() {
    test_backend();
    test_dense_backend();
    for (bool full : {false, true}) {
        g_full = full;
        say(full ? "== B ~ G 第二遍：本征分解、乘法、线性求解都交给 MKL（use_mkl）==\n" : "== B ~ G 第一遍：只有本征分解交给 MKL ==\n");
        test_planar();
        test_degenerate();
        test_conical();
        test_2d();
        test_fields();
        test_parallel();
    }
    g_full = false;
    test_singular();
    test_symmetric();
    test_json();
    say("%d 项检查，%d 项失败\n", g_total, g_total - g_pass);
    return g_pass == g_total ? 0 : 1;
}
