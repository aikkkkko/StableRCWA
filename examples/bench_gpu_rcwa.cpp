// examples/bench_gpu_rcwa.cpp —— 整个 RCWA 求解里，把本征分解交给显卡能快多少
//
// 对四脊大周期光栅加 20 层膜（examples::ridges_on_mirror）在几个截断阶数下各算四遍。前两遍全部在处理器上算
// （本征分解用 eig_qr），线程数分别是 1 和 --threads，对应 JSON 里的 cpu1、cpuN。后两遍把本征分解交给显卡，
// 双精度、单精度各一遍（gpu、gpu_single）；其余步骤（界面匹配、S 矩阵合并）仍在处理器上用 --threads 个线程。
// 结果以 JSON 写到标准输出，进度写到标准错误。
//
//   构建见 examples\build_gpu_win.bat 里 g++ 那一行，把源文件换成本文件并加上 -I examples 即可。
//   bench_gpu_rcwa.exe --orders 100,200,400,600 --threads 16
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "stablercwa/stablercwa.hpp"
#include "stablercwa/gpu/cuda_backend.hpp"
#include "say.hpp"
#include "structures.hpp"

using namespace stablercwa;
using examples::say_err;      // 用法同 std::fprintf(stderr, ...)，格式串里可以有中文（缘由见 say.hpp）

namespace {

struct Run {
    double total = 0, eig = 0, linear = 0, product = 0, energy = 0, eff = 0;
};

Run solve_once(const rcwa::Stack& stack, const examples::Ridges& wide, int m, const rcwa::Options& opt, int repeats) {
    Run best;
    best.total = 1e300;
    for (int k = 0; k < repeats; ++k) {
        const rcwa::Result r = rcwa::solve(stack, wide.wavelength, wide.theta_deg, m, rcwa::Polarization::TE, opt);
        if (r.timing.total < best.total) {
            best.total = r.timing.total;
            best.eig = r.timing.eig;
            best.linear = r.timing.linear;
            best.product = r.timing.product;
        }
        best.energy = std::abs(r.reflectance() + r.transmittance() - 1.0);
        best.eff = 100.0 * r.DEr[m + wide.order];
    }
    return best;
}

std::string run_json(const Run& r) {
    char buf[256];
    std::snprintf(buf, sizeof buf, "{\"total\":%.6g,\"eig\":%.6g,\"linear\":%.6g,\"product\":%.6g,\"energy\":%.3g,\"eff\":%.10g}",
                  r.total, r.eig, r.linear, r.product, r.energy, r.eff);
    return buf;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<int> orders{100, 200, 400};
    int threads = 16;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--orders" && i + 1 < argc) {
            orders.clear();
            std::string list = argv[++i];
            std::size_t pos = 0;
            while (pos < list.size()) {
                const std::size_t comma = list.find(',', pos);
                orders.push_back(std::atoi(list.substr(pos, comma - pos).c_str()));
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
        } else if (a == "--threads" && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        }
    }

    const examples::Ridges wide;
    const rcwa::Stack stack = examples::ridges_on_mirror(wide);
    for (int m : orders)
        if (m < std::abs(wide.order)) {
            say_err("--orders 里的每个截断阶数至少要取到 %d，才包含第 %d 级\n", std::abs(wide.order), wide.order);
            return 2;
        }
    gpu::Context ctx;
    std::fprintf(stderr, "GPU: %s\n", ctx.device_name().c_str());
    {   // 预热：第一次调用要加载并初始化库
        const Matrix warm = rcwa::layer_matrix(stack.layers.front(), stack.period, wide.wavelength, stack.nI, wide.theta_deg, 20,
                                               rcwa::Polarization::TE);
        ctx.set_precision(gpu::Precision::Double);
        (void)gpu::eig(ctx, warm);
        ctx.set_precision(gpu::Precision::Single);
        (void)gpu::eig(ctx, warm);
    }

    std::string out = "{\"threads\":" + std::to_string(threads) + ",\"cases\":[";
    for (std::size_t k = 0; k < orders.size(); ++k) {
        const int m = orders[k];
        const int repeats = m <= 200 ? 3 : 1;
        rcwa::Options cpu;

        set_num_threads(1);
        const Run cpu1 = solve_once(stack, wide, m, cpu, repeats);
        set_num_threads(threads);
        const Run cpun = solve_once(stack, wide, m, cpu, repeats);

        rcwa::Options on_gpu;
        on_gpu.custom_eig = [&](const Matrix& a) { return gpu::eig(ctx, a); };
        ctx.set_precision(gpu::Precision::Double);
        const Run g64 = solve_once(stack, wide, m, on_gpu, repeats);
        ctx.set_precision(gpu::Precision::Single);
        const Run g32 = solve_once(stack, wide, m, on_gpu, repeats);

        say_err("M=%d N=%d  CPU 1 线程 %.3f s（本征 %.3f）  CPU %d 线程 %.3f s  GPU 双精度 %.3f s（本征 %.3f）  GPU 单精度 %.3f s"
                "  |R+T-1|：%.1e / %.1e / %.1e\n",
                m, 2 * m + 1, cpu1.total, cpu1.eig, threads, cpun.total, g64.total, g64.eig, g32.total, cpu1.energy, g64.energy,
                g32.energy);
        out += std::string(k ? "," : "") + "{\"M\":" + std::to_string(m) + ",\"N\":" + std::to_string(2 * m + 1) + ",\"cpu1\":" +
               run_json(cpu1) + ",\"cpuN\":" + run_json(cpun) + ",\"gpu\":" + run_json(g64) + ",\"gpu_single\":" + run_json(g32) + "}";
    }
    out += "]}";
    std::puts(out.c_str());
    return 0;
}
