// examples/bench_mkl_rcwa.cpp —— RCWA 求解的用时：全用内置的运算，只把本征分解交给 oneMKL，还是本征分解、稠密乘法、
// 稠密线性求解都交给它（mkl::use_mkl）
//
//   bench_mkl_rcwa [--threads 8] [--orders 100,200,400] [--orders2d 4,6,8,10] [--repeats 2] [--sweep 1,2,4,8,16]
//
// 四个算例都无吸收，层矩阵两个是复的、两个是实的（相对原点或单元中心对称，打开 real_path 时 mkl::eig 走 dgeev）：
//   一维 复：examples::ridges_on_mirror（四脊大周期光栅 + 20 层膜，TE），层矩阵 2M+1 阶；
//   一维 实：同样的周期、膜系，光栅换成相对 x = 0 对称的五脊（TE）；
//   二维 实：tests/rcwa2d_reference.inc 里的方柱 pillar_sq（柱子在单元中心），M1 = M2 = M，P·Q 是 2(2M+1)² 阶；
//   二维 复：同一文件里的偏心方柱 pillar_ecc。
// 每种设置取几次里最快的一次，单位秒。都先 set_num_threads(--threads)；MKL 线程数取默认的 num_threads()，打开 real_path
// （结构都无吸收），实矩阵走 dgeev：
//   内置：eig_qr、内置的乘法与 LU；
//   本征 MKL：只设 custom_eig = mkl::eig_hook()，乘法与线性求解仍是内置的；
//   全部 MKL：mkl::use_mkl，三个钩子都交给 MKL。
// 表里的“本征”“求解”“乘法”是 timing.eig、timing.linear、timing.product，“总计”是整次求解；“效率差”是全部 MKL 与内置的
// 各级效率的最大差。
// --sweep 给出各算例第一个层矩阵的本征分解在不同 MKL 线程数下的用时（只算本征分解，不解 RCWA）。
// 要有 oneMKL 2025.0.1 或更新，用 CMake 选项 STABLERCWA_WITH_MKL 编译（见 CMakeLists.txt）。
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "stablercwa/stablercwa.hpp"
#include "stablercwa/mkl/mkl_backend.hpp"
#include "say.hpp"
#include "structures.hpp"

using namespace stablercwa;
namespace rcwa = stablercwa::rcwa;
#include "rcwa2d_reference.inc"

using examples::say;

namespace {

struct Run {
    double total = 1e300, eig = 0, linear = 0, product = 0;
    RealVector DEr, DEt;
};

// 三个钩子（空的就用内置的运算）
struct Hooks {
    std::function<EigResult(const Matrix&)> eig;
    std::function<Matrix(const Matrix&, const Matrix&)> multiply, solve;
};
template <class O>
O with_hooks(O o, const Hooks& h) {
    o.custom_eig = h.eig;
    o.custom_multiply = h.multiply;
    o.custom_solve = h.solve;
    return o;
}

double gap(const Run& a, const Run& b) {
    double worst = 0.0;
    for (std::size_t j = 0; j < a.DEr.size() && j < b.DEr.size(); ++j)
        worst = (std::max)(worst, (std::max)(std::abs(a.DEr[j] - b.DEr[j]), std::abs(a.DEt[j] - b.DEt[j])));
    return worst;
}

std::vector<int> parse_list(const std::string& text) {
    std::vector<int> out;
    std::size_t pos = 0;
    while (pos < text.size()) {
        const std::size_t comma = text.find(',', pos);
        out.push_back(std::atoi(text.substr(pos, comma - pos).c_str()));
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return out;
}

template <class R>
Run to_run(const R& r) {
    Run run;
    run.total = r.timing.total;
    run.eig = r.timing.eig;
    run.linear = r.timing.linear;
    run.product = r.timing.product;
    run.DEr = r.DEr;
    run.DEt = r.DEt;
    return run;
}

// 一个算例：给定钩子就解一次；layer_matrix() 拿到第一个需要本征分解的层矩阵。
struct Case {
    std::string label;
    int order = 0;                                       // 层矩阵的阶数
    bool real = false;                                   // 层矩阵是否（在舍入误差以内）是实的
    std::function<Run(const Hooks&)> solve;
};

Matrix layer_matrix(const Case& c) {
    Matrix captured;
    Hooks h;
    h.eig = [&](const Matrix& a) {
        if (captured.rows() == 0) captured = a;
        EigResult r;                                     // 只要矩阵：报告未收敛，求解器就此按数值失败返回
        r.values.assign(static_cast<std::size_t>(a.rows()), cplx{});
        r.info = 1;
        return r;
    };
    c.solve(h);
    return captured;
}

template <class F>
double best_seconds(int repeats, F f) {
    double best = 1e300;
    for (int k = 0; k < repeats; ++k) {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        best = (std::min)(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    return best;
}

Run best_of(int repeats, const Case& c, const Hooks& hooks) {
    Run best;
    for (int k = 0; k < repeats; ++k) {
        Run r = c.solve(hooks);
        if (r.total < best.total) best = r;
    }
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<int> orders{100, 200, 400}, orders2d{4, 6, 8, 10}, sweep;
    int threads = 8, repeats = 2;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--threads") threads = std::atoi(v.c_str());
        else if (a == "--orders") orders = parse_list(v);
        else if (a == "--orders2d") orders2d = parse_list(v);
        else if (a == "--repeats") repeats = (std::max)(1, std::atoi(v.c_str()));
        else if (a == "--sweep") sweep = parse_list(v);
    }
    set_num_threads(threads);   // 三种设置都一样：线性求解、矩阵乘法也按这个线程数并行；MKL 的默认线程数也是它

    // ---------------------------------------------------------------- 算例
    const examples::Ridges wide;
    const rcwa::Stack ridges = examples::ridges_on_mirror(wide);
    examples::Ridges sym_r = wide;                       // 相对 x = 0 对称的五脊（首末相差一个周期 6.3）
    sym_r.edges = {-2.9, -2.05, -1.6, -1.05, -0.2, 0.2, 1.05, 1.6, 2.05, 2.9, 3.4};
    sym_r.values = {1.0, 2.10, 1.0, 1.52, 1.75, 1.52, 1.0, 2.10, 1.0, 1.0};
    const rcwa::Stack ridges_sym = examples::ridges_on_mirror(sym_r);

    std::vector<Case> cases;
    for (int pass = 0; pass < 2; ++pass)
        for (int m : orders) {
            const rcwa::Stack* stack = pass == 0 ? &ridges : &ridges_sym;
            Case c;
            c.label = std::string(pass == 0 ? "一维 复" : "一维 实") + " M=" + std::to_string(m);
            c.order = 2 * m + 1;
            c.real = pass == 1;
            c.solve = [stack, &wide, m](const Hooks& hooks) {
                const rcwa::Options o = with_hooks(rcwa::Options{}, hooks);
                return to_run(rcwa::solve(*stack, wide.wavelength, wide.theta_deg, m, rcwa::Polarization::TE, o));
            };
            cases.push_back(c);
        }
    for (const char* name : {"pillar_sq", "pillar_ecc"}) {
        const Rcwa2dRefCase* ref = nullptr;
        for (int i = 0; i < kRcwa2dRefCaseCount; ++i)
            if (std::string(kRcwa2dRefCases[i].name) == name) ref = &kRcwa2dRefCases[i];
        if (!ref) continue;
        for (int m : orders2d) {
            Case c;
            c.real = std::string(name) == "pillar_sq";
            c.label = std::string(c.real ? "二维 实" : "二维 复") + " M=" + std::to_string(m);
            c.order = 2 * (2 * m + 1) * (2 * m + 1);
            c.solve = [ref, m](const Hooks& hooks) {
                const rcwa::Stack2D stack = rcwa2d_ref_stack(ref->id);
                const rcwa::Jones2D jones = {cplx{ref->jones[0], ref->jones[1]}, cplx{ref->jones[2], ref->jones[3]}};
                const rcwa::Options2D o = with_hooks(rcwa::Options2D{}, hooks);
                return to_run(rcwa::solve2d(stack, ref->wavelength, ref->theta_deg, ref->phi_deg, m, m, jones, o));
            };
            cases.push_back(c);
        }
    }

    say("%s\n", mkl::version_string().c_str());
    say("StableRCWA %d 个线程；MKL %d 个线程，real_path = true（实矩阵走 dgeev%s）。单位秒，取 %d 次里最快的一次。\n\n", threads, threads,
        mkl::dgeev_usable() ? "" : "——这个 MKL 版本的 dgeev 不可用，实际都走 zgeev", repeats);

    // ---------------------------------------------------------------- 整次求解
    say("%-14s %6s | %8s %8s %8s %8s | %8s %6s | %8s %8s %8s %8s %6s | %8s\n", "算例", "阶数", "内置总计", "本征", "求解", "乘法",
        "本征MKL", "加速", "全部MKL", "本征", "求解", "乘法", "加速", "效率差");
    mkl::Settings lossless;                              // 四个算例都无吸收
    lossless.real_path = true;
    Hooks eig_only, all;
    eig_only.eig = mkl::eig_hook(lossless);
    all.eig = eig_only.eig;
    all.multiply = mkl::multiply_hook(lossless);
    all.solve = mkl::solve_hook(lossless);
    for (const Case& c : cases) {
        const Matrix a = layer_matrix(c);
        const bool real = a.rows() > 0 && mkl::nearly_real(a);
        const Run qr = best_of(repeats, c, {});
        const Run me = best_of(repeats, c, eig_only);
        const Run ma = best_of(repeats, c, all);
        say("%-14s %6d | %8.3f %8.3f %8.3f %8.3f | %8.3f %5.2fx | %8.3f %8.3f %8.3f %8.3f %5.2fx | %8.1e%s\n", c.label.c_str(), c.order,
            qr.total, qr.eig, qr.linear, qr.product, me.total, qr.total / me.total, ma.total, ma.eig, ma.linear, ma.product,
            qr.total / ma.total, gap(qr, ma), real == c.real ? "" : "  （层矩阵的虚实与预期不符）");
    }

    // ---------------------------------------------------------------- MKL 线程数
    if (!sweep.empty()) {
        say("\n本征分解的用时与 MKL 线程数（各算例第一个层矩阵；实矩阵另列 zgeev，即当作复矩阵）\n");
        std::string head = "算例            阶数 路径  ";
        for (int t : sweep) head += "  " + std::to_string(t) + " 线程";
        say("%s   最快\n", head.c_str());
        for (const Case& c : cases) {
            const Matrix a = layer_matrix(c);
            if (a.rows() == 0) continue;
            for (int pass = 0; pass < (mkl::nearly_real(a) ? 2 : 1); ++pass) {
                mkl::Settings s;
                s.real_path = pass == 0;
                std::string line;
                double best = 1e300;
                int best_t = 0;
                for (int t : sweep) {
                    s.threads = t;
                    const double sec = best_seconds(repeats, [&] { mkl::eig(a, s); });
                    char buf[32];
                    std::snprintf(buf, sizeof buf, " %8.3f", sec);
                    line += buf;
                    if (sec < best) best = sec, best_t = t;
                }
                const char* path = pass == 0 && mkl::nearly_real(a) && mkl::dgeev_usable() ? "dgeev" : "zgeev";
                say("%-14s %6d %s%s   %d\n", c.label.c_str(), a.rows(), path, line.c_str(), best_t);
            }
        }
    }
    return 0;
}
