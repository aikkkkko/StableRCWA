// examples/bench_accel.cpp —— CPU 基准：用时组成、均匀层、三个层次的并行（JSON 写到标准输出，进度写到标准错误）
//
//   c++ -std=c++17 -O2 -I include -I examples examples/bench_accel.cpp -o bench_accel
//   ./bench_accel --threads "1,2,4,6,8" --machine "Apple M1 Pro" > bench_mac.json
//   ./bench_accel --quick > quick.json               # 小 M 的冒烟测试，几秒钟
//   ./bench_accel --only layers,kernel > part.json   # 只重测某几项（输出里只有这几项，再加 machine）
//
// 全部是 TE 偏振、QR 本征迭代、选主元的 LU。测四件事：
//   breakdown   一次求解的时间花在哪里：本征分解 / 稠密线性求解 / 稠密矩阵乘法
//   films       均匀层解析捷径省了多少：四脊大周期光栅层下面垫 F 层介质膜，捷径开、关各算一遍（counts 是 F = 20 时的次数）
//   sweep / layers / kernel
//               三个层次的多线程：扫描点之间、各层的本征分解之间、单个稠密内核（矩阵乘法、LU）内部
//   total       同一件活（64 个波长）的三种做法：全部走稠密路径、打开解析捷径、再加上多线程扫描
//
// 计时方法：steady_clock；每一项先热身一次，再重复若干次取最小值，重复次数记在 JSON 的 reps 里。
// 多线程的各项按“每一轮把所有线程数各测一次”的次序重复：机器上偶然的干扰只会坏掉某一轮里的某一个数，
// 取最小值时就被别的轮次盖掉了。每一轮的原始读数都写在标准错误里，可以回头查。
// 这几项除了最小值（sec）还给出各轮的中位数（sec_median）：两者差得多，说明读数不稳，
// 例如操作系统有时把两个线程排到同一个物理核心的两个超线程上，有时又不这样。
//
// 选项：--threads "1,2,4,8"   多线程各项要测的线程数（默认 1、2、4 … 直到硬件线程数）
//       --quick               用很小的 M 走一遍流程（JSON 里带 "quick":true，数字没有参考价值）
//       --machine "名称"      写进 JSON 的处理器名称
//       --only a,b,...        只测列出的项：breakdown films sweep layers kernel total（films 连带 counts）
//       --reps k              多线程各项重复的轮数（默认 5）
//
// 进度信息只用 ASCII 字符：这个程序常常隔着远程终端在 Windows 上跑，那里的控制台不是 UTF-8。
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <string>
#include <vector>

#include "stablercwa/stablercwa.hpp"
#include "structures.hpp"

using namespace stablercwa;
namespace rcwa = stablercwa::rcwa;

namespace {

using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

const double kInf = std::numeric_limits<double>::infinity();

// ------------------------------------------------------------------ 极简 JSON 输出

std::string num(double x, int digits = 6) {
    if (!std::isfinite(x)) return "null";
    char buf[48];
    std::snprintf(buf, sizeof buf, "%.*g", digits, x);
    return buf;
}
std::string arr(const std::vector<double>& v, int digits = 6) {
    std::string s = "[";
    for (std::size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + num(v[i], digits);
    return s + "]";
}
std::string arr(const std::vector<int>& v) {
    std::string s = "[";
    for (std::size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
    return s + "]";
}
// JSON 字符串字面量。名字不叫 quoted：实参是 std::string 时，依赖实参的名字查找会把标准库的 std::quoted
// （<iomanip>，有的标准库会被别的头文件间接带进来，C++20 模式下尤其如此）也找出来，并抢走这个调用。
std::string json_string(const std::string& text) {
    std::string s = "\"";
    for (char ch : text) {
        if (ch == '"' || ch == '\\') s += '\\';
        if (static_cast<unsigned char>(ch) >= 0x20) s += ch;
    }
    return s + "\"";
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
#if defined(__apple_build_version__)
    s = std::string("Apple clang ") + __clang_version__;
#else
    s = std::string("clang ") + __clang_version__;
#endif
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

// ------------------------------------------------------------------ 设置

struct Config {
    std::vector<int> threads;                    // 多线程各项要测的线程数
    std::string machine = "unknown CPU";
    std::string only;                            // 空：全部都测
    bool quick = false;
    int rounds = 5;                              // 多线程各项重复的轮数
    int breakdown_m[2] = {100, 200};
    int films_m = 100;
    std::vector<int> film_counts{0, 5, 10, 20, 40};
    int counts_films = 20;                       // counts 一项用几层膜
    int sweep_points = 64, sweep_m = 60;
    int layer_slices = 16, layer_m = 60;
    int kernel_m = 200;
    int total_points = 64, total_m = 100;
    double dense_budget = 300.0;                 // total 里全稠密的那一遍超过这么多秒就只测 4 个点再外推
};

bool wanted(const Config& cfg, const std::string& name) {
    if (cfg.only.empty()) return true;
    return ("," + cfg.only + ",").find("," + name + ",") != std::string::npos;
}

std::vector<int> parse_list(const std::string& text) {
    std::vector<int> out;
    std::string item;
    for (std::size_t i = 0; i <= text.size(); ++i) {
        if (i < text.size() && text[i] != ',' && text[i] != ' ') {
            item += text[i];
            continue;
        }
        if (!item.empty()) {
            const int v = std::atoi(item.c_str());
            if (v >= 1) out.push_back(v);
            item.clear();
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

// 默认的线程数序列：1、2、4 … 再加上硬件线程数本身。
std::vector<int> default_threads() {
    std::vector<int> out;
    const int hw = hardware_threads();
    for (int t = 1; t < hw; t *= 2) out.push_back(t);
    out.push_back(hw);
    return out;
}

// ------------------------------------------------------------------ 计时

int g_failures = 0;      // 结果检查没通过的次数（非零时程序最后返回 1）

void check(bool ok, const char* what, double value) {
    if (ok) return;
    ++g_failures;
    std::fprintf(stderr, "    CHECK FAILED: %s (%.3e)\n", what, value);
}

// 无吸收的结构应当有 R + T = 1。返回 |R + T − 1|，结果里有 NaN 时返回 NaN。
double energy_error(const rcwa::Result& r) { return std::abs(r.reflectance() + r.transmittance() - 1.0); }

// 重复调用 job()（它返回这一次的用时，秒），取最小值。先热身一次；之后至少测 min_reps 次，
// 累计用时不到 budget 秒就继续测，至多 max_reps 次。
struct Best {
    double sec = kInf;
    int reps = 0;
};
template <class Job>
Best best_of(Job&& job, int min_reps, int max_reps, double budget) {
    job();
    Best b;
    double spent = 0.0;
    while (b.reps < min_reps || (b.reps < max_reps && spent < budget)) {
        const double t = job();
        b.sec = std::min(b.sec, t);
        spent += t;
        ++b.reps;
    }
    return b;
}

// 一次计时得到的数：总的墙上时间，以及求解器自己统计的三个分项。
struct Shot {
    double sec = kInf, eig = 0, linear = 0, product = 0;
};

double median_of(std::vector<double> v) {
    if (v.empty()) return kInf;
    std::sort(v.begin(), v.end());
    const std::size_t h = v.size() / 2;
    return v.size() % 2 ? v[h] : 0.5 * (v[h - 1] + v[h]);
}

// 多线程加速比的测法：在最大的线程数下热身一次（把所有核心叫醒），然后重复 rounds 轮，
// 每一轮把各个线程数各测一次；每个线程数留下总用时最小的那一次，另外记下各轮总用时的中位数。
struct Scaled {
    std::vector<Shot> best;          // 每个线程数里总用时最小的那一次
    std::vector<double> median;      // 每个线程数各轮总用时的中位数
};
template <class Job>
Scaled scaling(const char* tag, const std::vector<int>& threads, int rounds, Job&& job) {
    Scaled out;
    out.best.resize(threads.size());
    std::vector<std::vector<double>> all(threads.size());
    set_num_threads(threads.back());
    job();
    for (int round = 0; round < rounds; ++round) {
        std::fprintf(stderr, "[%s] round %d:", tag, round + 1);
        for (std::size_t k = 0; k < threads.size(); ++k) {
            set_num_threads(threads[k]);
            const Shot s = job();
            if (s.sec < out.best[k].sec) out.best[k] = s;
            all[k].push_back(s.sec);
            std::fprintf(stderr, "  %dt %.4f", threads[k], s.sec);
        }
        std::fprintf(stderr, "\n");
    }
    set_num_threads(1);
    for (const std::vector<double>& v : all) out.median.push_back(median_of(v));
    std::fprintf(stderr, "[%s] best    :", tag);
    for (std::size_t k = 0; k < threads.size(); ++k)
        std::fprintf(stderr, "  %dt %.4f (x%.2f)", threads[k], out.best[k].sec, out.best[0].sec / out.best[k].sec);
    std::fprintf(stderr, "\n[%s] median  :", tag);
    for (std::size_t k = 0; k < threads.size(); ++k)
        std::fprintf(stderr, "  %dt %.4f (x%.2f)", threads[k], out.median[k], out.median[0] / out.median[k]);
    std::fprintf(stderr, "\n");
    return out;
}

std::vector<double> seconds_of(const std::vector<Shot>& shots) {
    std::vector<double> v;
    for (const Shot& s : shots) v.push_back(s.sec);
    return v;
}

// ------------------------------------------------------------------ 结构

// 大周期结构：四脊光栅层 + 介质膜（examples::ridges_on_films，膜自上而下 H · L · H · L …，F = 20 时就是
// examples::ridges_on_mirror）。“各层之间”一项另用单周期的梯形齿结构，见 run_layers。

// 扫描用的波长点：中心波长的 ±1.5 % 之内等间隔（0.94 微米时是 0.9259 … 0.9541），入射角不变。
std::vector<rcwa::Point> wavelength_points(int count, double center, double theta_deg) {
    std::vector<rcwa::Point> pts(static_cast<std::size_t>(count));
    const double lo = 0.985 * center, hi = 1.015 * center;
    for (int i = 0; i < count; ++i) {
        pts[static_cast<std::size_t>(i)].wavelength = count > 1 ? lo + (hi - lo) * i / (count - 1) : center;
        pts[static_cast<std::size_t>(i)].theta_deg = theta_deg;
    }
    return pts;
}

// 一组结果里 |R + T − 1| 的最大值（有 NaN 时返回 NaN）。
double worst_energy_error(const std::vector<rcwa::Result>& results) {
    double worst = 0.0;
    for (const rcwa::Result& r : results) {
        const double e = energy_error(r);
        if (!(e <= worst)) worst = e;                    // 这样写 NaN 也会留下来
        if (std::isnan(e)) return e;
    }
    return worst;
}

// ------------------------------------------------------------------ 1. 时间花在哪里

std::string run_breakdown(const Config& cfg) {
    set_num_threads(1);
    const examples::Ridges wide;
    const rcwa::Stack stack = examples::ridges_on_mirror(wide);
    std::string cases;
    for (int m : cfg.breakdown_m) {
        rcwa::Result best;
        best.timing.total = kInf;
        const Best b = best_of(
            [&] {
                rcwa::Result r = rcwa::solve(stack, wide.wavelength, wide.theta_deg, m, rcwa::Polarization::TE);
                const double t = r.timing.total;
                if (t < best.timing.total) best = std::move(r);
                return t;
            },
            3, 9, 3.0);
        const rcwa::Timing& t = best.timing;
        check(energy_error(best) < 1e-8, "breakdown: R + T != 1", energy_error(best));
        std::fprintf(stderr,
                     "[breakdown] M=%d N=%d  total %.4f s = eig %.4f + linear %.4f + product %.4f + other %.4f | "
                     "n_eig %d n_solve %d | reps %d\n",
                     m, 2 * m + 1, t.total, t.eig, t.linear, t.product, t.total - t.eig - t.linear - t.product,
                     best.n_eig, best.n_dense_solve, b.reps);
        if (!cases.empty()) cases += ",";
        cases += "{\"M\":" + std::to_string(m) + ",\"N\":" + std::to_string(2 * m + 1) + ",\"fast\":{\"total\":" +
                 num(t.total) + ",\"eig\":" + num(t.eig) + ",\"linear\":" + num(t.linear) + ",\"product\":" +
                 num(t.product) + ",\"n_eig\":" + std::to_string(best.n_eig) + ",\"n_solve\":" +
                 std::to_string(best.n_dense_solve) + "},\"reps\":" + std::to_string(b.reps) + "}";
    }
    return "\"breakdown\":{\"cases\":[" + cases + "]}";
}

// ------------------------------------------------------------------ 2. 均匀层解析捷径

// 两组效率之间最大的差（逐级比较反射、透射效率）。任何一边有 NaN 时返回 NaN。
double efficiency_difference(const rcwa::Result& a, const rcwa::Result& b) {
    double worst = 0.0;
    for (std::size_t j = 0; j < a.DEr.size(); ++j) {
        const double dr = std::abs(a.DEr[j] - b.DEr[j]), dt = std::abs(a.DEt[j] - b.DEt[j]);
        if (std::isnan(dr) || std::isnan(dt)) return std::numeric_limits<double>::quiet_NaN();
        worst = std::max(worst, std::max(dr, dt));
    }
    return worst;
}

std::string run_films(const Config& cfg) {
    set_num_threads(1);
    const examples::Ridges wide;
    const int m = cfg.films_m;
    rcwa::Options fast_opt, dense_opt;
    dense_opt.homogeneous_fast_path = false;
    std::vector<double> fast_sec, dense_sec;
    std::vector<int> fast_reps, dense_reps;
    double max_diff = 0.0;
    std::string counts;
    for (int films : cfg.film_counts) {
        const rcwa::Stack stack = examples::ridges_on_films(wide, films);
        rcwa::Result rf, rd;
        const auto timed = [&](const rcwa::Options& opt, rcwa::Result& keep) {
            const auto t0 = Clock::now();
            keep = rcwa::solve(stack, wide.wavelength, wide.theta_deg, m, rcwa::Polarization::TE, opt);
            return seconds_since(t0);
        };
        const Best bf = best_of([&] { return timed(fast_opt, rf); }, 3, 9, 1.5);
        const Best bd = best_of([&] { return timed(dense_opt, rd); }, 2, 5, 4.0);
        const double diff = efficiency_difference(rf, rd);
        if (!(diff <= max_diff)) max_diff = diff;
        check(energy_error(rf) < 1e-8, "films: fast path R + T != 1", energy_error(rf));
        check(energy_error(rd) < 1e-8, "films: dense path R + T != 1", energy_error(rd));
        check(diff < 1e-8, "films: fast and dense paths disagree", diff);
        fast_sec.push_back(bf.sec);
        dense_sec.push_back(bd.sec);
        fast_reps.push_back(bf.reps);
        dense_reps.push_back(bd.reps);
        std::fprintf(stderr,
                     "[films] M=%d F=%2d  fast %.4f s (n_eig %d n_solve %d, reps %d) | dense %.4f s (n_eig %d n_solve %d, "
                     "reps %d) | x%.1f | max diff %.2e\n",
                     m, films, bf.sec, rf.n_eig, rf.n_dense_solve, bf.reps, bd.sec, rd.n_eig, rd.n_dense_solve, bd.reps,
                     bd.sec / bf.sec, diff);
        if (films == cfg.counts_films)
            counts = "\"counts\":{\"films\":" + std::to_string(films) + ",\"fast\":{\"n_eig\":" + std::to_string(rf.n_eig) +
                     ",\"n_solve\":" + std::to_string(rf.n_dense_solve) + "},\"dense\":{\"n_eig\":" +
                     std::to_string(rd.n_eig) + ",\"n_solve\":" + std::to_string(rd.n_dense_solve) + "}}";
    }
    std::string out = "\"films\":{\"M\":" + std::to_string(m) + ",\"count\":" + arr(cfg.film_counts) + ",\"fast\":" +
                      arr(fast_sec) + ",\"dense\":" + arr(dense_sec) + ",\"max_diff\":" + num(max_diff, 3) +
                      ",\"reps\":{\"fast\":" + arr(fast_reps) + ",\"dense\":" + arr(dense_reps) + "}}";
    if (!counts.empty()) out += ",\n " + counts;
    return out;
}

// ------------------------------------------------------------------ 3. 三个层次的多线程

// 扫描点之间：rcwa::sweep 把各个波长点分给线程池，每个点内部是串行的。
std::string run_sweep(const Config& cfg) {
    const examples::Ridges wide;
    const rcwa::Stack stack = examples::ridges_on_mirror(wide);
    const std::vector<rcwa::Point> pts = wavelength_points(cfg.sweep_points, wide.wavelength, wide.theta_deg);
    std::vector<rcwa::Result> reference;                 // 第一次得到的结果；之后每一次都应与它逐位相同
    double worst_diff = 0.0;
    const Scaled scaled = scaling("sweep", cfg.threads, cfg.rounds, [&] {
        const auto t0 = Clock::now();
        std::vector<rcwa::Result> results = rcwa::sweep(stack, pts, cfg.sweep_m);
        Shot s;
        s.sec = seconds_since(t0);
        if (reference.empty()) reference = std::move(results);
        else
            for (std::size_t i = 0; i < results.size(); ++i) {
                const double d = efficiency_difference(results[i], reference[i]);
                if (!(d <= worst_diff)) worst_diff = d;
            }
        return s;
    });
    const double e = worst_energy_error(reference);
    check(e < 1e-8, "sweep: R + T != 1", e);
    check(worst_diff == 0.0, "sweep: results depend on the thread count", worst_diff);
    return "\"sweep\":{\"points\":" + std::to_string(cfg.sweep_points) + ",\"M\":" + std::to_string(cfg.sweep_m) +
           ",\"threads\":" + arr(cfg.threads) + ",\"sec\":" + arr(seconds_of(scaled.best)) + ",\"sec_median\":" +
           arr(scaled.median) + ",\"reps\":" + std::to_string(cfg.rounds) + "}";
}

// 各层之间：梯形齿切成若干片，每片一个光栅层，parallel_layers 把这些层的本征分解分给线程池。
// 一次求解很短，所以连着求解 inner 遍当作一次计时，再除以 inner。
std::string run_layers(const Config& cfg) {
    const examples::Mdg mdg;
    const rcwa::Stack stack = examples::trapezoid(cfg.layer_slices, mdg);
    rcwa::Options opt;
    opt.parallel_layers = true;
    const auto once = [&] { return rcwa::solve(stack, mdg.wavelength, mdg.theta_deg, cfg.layer_m, rcwa::Polarization::TE, opt); };

    set_num_threads(1);
    rcwa::Result reference = once();
    reference = once();
    const double single = reference.timing.total;
    const int inner = std::max(1, std::min(50, static_cast<int>(std::ceil(0.25 / std::max(single, 1e-6)))));
    check(energy_error(reference) < 1e-8, "layers: R + T != 1", energy_error(reference));
    std::fprintf(stderr, "[layers] slices=%d M=%d  one solve %.4f s (eig %.4f, n_eig %d, n_solve %d) -> %d solve(s) per shot\n",
                 cfg.layer_slices, cfg.layer_m, single, reference.timing.eig, reference.n_eig, reference.n_dense_solve, inner);

    double worst_diff = 0.0;
    const Scaled scaled = scaling("layers", cfg.threads, cfg.rounds, [&] {
        Shot s;
        s.sec = 0.0;
        const auto t0 = Clock::now();
        for (int k = 0; k < inner; ++k) {
            const rcwa::Result r = once();
            s.eig += r.timing.eig;
            const double d = efficiency_difference(r, reference);
            if (!(d <= worst_diff)) worst_diff = d;
        }
        s.sec = seconds_since(t0) / inner;
        s.eig /= inner;
        return s;
    });
    check(worst_diff == 0.0, "layers: results depend on the thread count", worst_diff);
    const std::vector<Shot>& best = scaled.best;
    std::vector<double> eig_sec;
    for (const Shot& s : best) eig_sec.push_back(s.eig);
    std::fprintf(stderr, "[layers] eig part:");
    for (std::size_t k = 0; k < best.size(); ++k)
        std::fprintf(stderr, "  %dt %.4f (x%.2f)", cfg.threads[k], best[k].eig, best[0].eig / best[k].eig);
    std::fprintf(stderr, "\n");
    return "\"layers\":{\"slices\":" + std::to_string(cfg.layer_slices) + ",\"M\":" + std::to_string(cfg.layer_m) +
           ",\"threads\":" + arr(cfg.threads) + ",\"sec\":" + arr(seconds_of(best)) + ",\"sec_median\":" +
           arr(scaled.median) + ",\"eig_sec\":" + arr(eig_sec) + ",\"reps\":" + std::to_string(cfg.rounds) +
           ",\"solves_per_rep\":" + std::to_string(inner) + "}";
}

// 内核内部：一次大的求解，parallel_layers 关着，只有矩阵乘法和 LU 的内核会用到多个线程。
std::string run_kernel(const Config& cfg) {
    const examples::Ridges wide;
    const rcwa::Stack stack = examples::ridges_on_mirror(wide);
    rcwa::Options opt;
    opt.parallel_layers = false;
    rcwa::Result reference;
    double worst_diff = 0.0;
    const Scaled scaled = scaling("kernel", cfg.threads, cfg.rounds, [&] {
        const auto t0 = Clock::now();
        rcwa::Result r = rcwa::solve(stack, wide.wavelength, wide.theta_deg, cfg.kernel_m, rcwa::Polarization::TE, opt);
        Shot s;
        s.sec = seconds_since(t0);
        s.eig = r.timing.eig;
        s.linear = r.timing.linear;
        s.product = r.timing.product;
        if (reference.DEr.empty()) reference = std::move(r);
        else {
            const double d = efficiency_difference(r, reference);
            if (!(d <= worst_diff)) worst_diff = d;
        }
        return s;
    });
    check(energy_error(reference) < 1e-8, "kernel: R + T != 1", energy_error(reference));
    check(worst_diff == 0.0, "kernel: results depend on the thread count", worst_diff);
    const std::vector<Shot>& best = scaled.best;
    std::vector<double> eig_sec, linear_sec, product_sec;
    for (const Shot& s : best) {
        eig_sec.push_back(s.eig);
        linear_sec.push_back(s.linear);
        product_sec.push_back(s.product);
    }
    std::fprintf(stderr, "[kernel] parts  :");
    for (std::size_t k = 0; k < best.size(); ++k)
        std::fprintf(stderr, "  %dt eig %.4f lin %.4f prod %.4f", cfg.threads[k], best[k].eig, best[k].linear, best[k].product);
    std::fprintf(stderr, "\n");
    return "\"kernel\":{\"M\":" + std::to_string(cfg.kernel_m) + ",\"threads\":" + arr(cfg.threads) + ",\"sec\":" +
           arr(seconds_of(best)) + ",\"sec_median\":" + arr(scaled.median) + ",\"eig_sec\":" + arr(eig_sec) +
           ",\"linear_sec\":" + arr(linear_sec) + ",\"product_sec\":" + arr(product_sec) + ",\"reps\":" +
           std::to_string(cfg.rounds) + "}";
}

// ------------------------------------------------------------------ 4. 合在一起

std::string run_total(const Config& cfg) {
    const examples::Ridges wide;
    const rcwa::Stack stack = examples::ridges_on_mirror(wide);
    const std::vector<rcwa::Point> pts = wavelength_points(cfg.total_points, wide.wavelength, wide.theta_deg);
    const int m = cfg.total_m;
    const int count = cfg.total_points;

    // (a) 全部走稠密路径，单线程，逐点循环。先算一个点（顺带热身）估计总用时；太长就只测均匀分布的 4 个点再按点数放大。
    set_num_threads(1);
    rcwa::Options dense_opt;
    dense_opt.homogeneous_fast_path = false;
    const auto dense_point = [&](int i, double& worst) {
        const rcwa::Point& p = pts[static_cast<std::size_t>(i)];
        const auto t0 = Clock::now();
        const rcwa::Result r = rcwa::solve(stack, p.wavelength, p.theta_deg, m, rcwa::Polarization::TE, dense_opt);
        const double t = seconds_since(t0);
        const double e = energy_error(r);
        if (!(e <= worst)) worst = e;
        return t;
    };
    double dense_worst = 0.0;
    const double first = dense_point(0, dense_worst);
    const bool extrapolated = count > 4 && first * count > cfg.dense_budget;
    std::fprintf(stderr, "[total] dense path, 1 thread: first point %.3f s -> %s\n", first,
                 extrapolated ? "time 4 points and scale" : "time every point");
    double dense_sec = 0.0;
    int dense_points = 0;
    if (extrapolated) {
        for (int k = 0; k < 4; ++k) {
            const int i = k * (count - 1) / 3;
            const double t = dense_point(i, dense_worst);
            std::fprintf(stderr, "[total]   point %d: %.3f s\n", i, t);
            dense_sec += t;
            ++dense_points;
        }
        dense_sec *= static_cast<double>(count) / dense_points;
    } else {
        const auto t0 = Clock::now();
        for (int i = 0; i < count; ++i) dense_point(i, dense_worst);
        dense_sec = seconds_since(t0);
        dense_points = count;
    }
    check(dense_worst < 1e-8, "total: dense path R + T != 1", dense_worst);
    std::fprintf(stderr, "[total] (a) dense path, 1 thread: %.2f s (%d point(s) timed)\n", dense_sec, dense_points);

    // (b) 打开解析捷径，单线程
    rcwa::Options fast_opt;
    std::vector<rcwa::Result> reference;
    const auto fast_sweep = [&] {
        const auto t0 = Clock::now();
        std::vector<rcwa::Result> results = rcwa::sweep(stack, pts, m, rcwa::Polarization::TE, fast_opt);
        const double t = seconds_since(t0);
        if (reference.empty()) reference = std::move(results);
        return t;
    };
    const Best fast1 = best_of(fast_sweep, 2, 5, 20.0);
    const double fast_worst = worst_energy_error(reference);
    check(fast_worst < 1e-8, "total: fast path R + T != 1", fast_worst);
    std::fprintf(stderr, "[total] (b) fast path, 1 thread: %.3f s (reps %d)\n", fast1.sec, fast1.reps);

    // (c) 解析捷径 + 多线程扫描：把列表里大于 1 的线程数和硬件线程数都试一遍，留下最快的。
    std::vector<int> tried;
    for (int t : cfg.threads)
        if (t > 1) tried.push_back(t);
    tried.push_back(hardware_threads());
    std::sort(tried.begin(), tried.end());
    tried.erase(std::unique(tried.begin(), tried.end()), tried.end());
    const int rounds = std::max(2, std::min(cfg.rounds, 3));
    const Scaled scaled = scaling("total", tried, rounds, [&] {
        Shot s;
        s.sec = fast_sweep();
        return s;
    });
    const std::vector<Shot>& shots = scaled.best;
    std::size_t best = 0;
    for (std::size_t k = 1; k < shots.size(); ++k)
        if (shots[k].sec < shots[best].sec) best = k;
    std::fprintf(stderr, "[total] (c) fast path + sweep: %.3f s with %d threads | (a)/(b) x%.1f  (b)/(c) x%.1f  (a)/(c) x%.0f\n",
                 shots[best].sec, tried[best], dense_sec / fast1.sec, fast1.sec / shots[best].sec, dense_sec / shots[best].sec);

    return "\"total\":{\"points\":" + std::to_string(count) + ",\"M\":" + std::to_string(m) + ",\"dense_1t\":" +
           num(dense_sec) + ",\"fast_1t\":" + num(fast1.sec) + ",\"fast_sweep\":" + num(shots[best].sec) +
           ",\"sweep_threads\":" + std::to_string(tried[best]) + ",\"extrapolated\":" + (extrapolated ? "true" : "false") +
           ",\"dense_points_timed\":" + std::to_string(dense_points) + ",\"reps\":{\"dense_1t\":1,\"fast_1t\":" +
           std::to_string(fast1.reps) + ",\"fast_sweep\":" + std::to_string(rounds) + "},\"sweep_tried\":{\"threads\":" +
           arr(tried) + ",\"sec\":" + arr(seconds_of(shots)) + ",\"sec_median\":" + arr(scaled.median) + "}}";
}

}  // namespace

int main(int argc, char** argv) {
    // 进度要随写随见：有的运行库在标准错误被重定向到文件时会攒到程序结束才写出去。
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--threads") cfg.threads = parse_list(next());
        else if (a == "--machine") cfg.machine = next();
        else if (a == "--only") cfg.only = next();
        else if (a == "--reps") cfg.rounds = std::max(1, std::atoi(next().c_str()));
        else if (a == "--quick") cfg.quick = true;
        else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (cfg.threads.empty()) cfg.threads = default_threads();
    if (cfg.threads.front() != 1) cfg.threads.insert(cfg.threads.begin(), 1);     // 加速比要以 1 个线程为基准
    if (cfg.quick) {
        cfg.breakdown_m[0] = 20;
        cfg.breakdown_m[1] = 40;
        cfg.films_m = 20;
        cfg.sweep_points = 16;
        cfg.sweep_m = 20;
        cfg.layer_m = 20;
        cfg.kernel_m = 40;
        cfg.total_points = 8;
        cfg.total_m = 20;
        cfg.rounds = std::min(cfg.rounds, 2);
    }

    const auto t_start = Clock::now();
    const std::string compiler = compiler_name();
    std::fprintf(stderr, "bench_accel: %s | %s | %d hardware threads | threads tested:", cfg.machine.c_str(), compiler.c_str(),
                 hardware_threads());
    for (int t : cfg.threads) std::fprintf(stderr, " %d", t);
    std::fprintf(stderr, "%s\n", cfg.quick ? " | QUICK MODE" : "");

    std::string out = "{\"machine\":{\"cpu\":" + json_string(cfg.machine) + ",\"compiler\":" + json_string(compiler) +
                      ",\"hardware_threads\":" + std::to_string(hardware_threads()) + "}";
    if (cfg.quick) out += ",\n \"quick\":true";
    const auto section = [&](const char* name, std::string (*run)(const Config&)) {
        if (!wanted(cfg, name)) return;
        const auto t0 = Clock::now();
        out += ",\n " + run(cfg);
        std::fprintf(stderr, "[%s] done in %.1f s\n", name, seconds_since(t0));
    };
    try {
        section("breakdown", run_breakdown);
        section("films", run_films);
        section("sweep", run_sweep);
        section("layers", run_layers);
        section("kernel", run_kernel);
        section("total", run_total);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "bench_accel: %s\n", e.what());
        return 1;
    }
    out += "}\n";
    std::fputs(out.c_str(), stdout);
    std::fflush(stdout);
    std::fprintf(stderr, "finished in %.1f s, %s\n", seconds_since(t_start),
                 g_failures ? "some checks FAILED" : "all checks passed");
    return g_failures ? 1 : 0;
}
