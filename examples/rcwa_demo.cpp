// examples/rcwa_demo.cpp —— 一维 RCWA 的命令行演示：换本征算法、换结构、看能量是否守恒
//
//   c++ -std=c++17 -O2 -I include -I examples examples/rcwa_demo.cpp -o rcwa_demo
//   ./rcwa_demo                                  # 默认：单周期光栅 + 20 层膜，M = 100，QR 迭代
//   ./rcwa_demo --eig lr --M 150                 # 光栅层交给 LR 迭代：单周期光栅的 R + T 明显偏离 100%
//   ./rcwa_demo --case wide --eig lr --M 120     # 四脊大周期光栅 + 20 层膜，光栅层交给 LR：R + T 仍接近 100%
//   ./rcwa_demo --case wide --eig lr --M 120 --all-layers   # 同一结构 21 层全部交给 LR 迭代：R + T 远离 100%
//   ./rcwa_demo --case absorb --M 60             # 齿有吸收（复折射率）
//   ./rcwa_demo --M 20 --pol TM
//
// 选项：--case single|absorb|trapezoid|wide   --M <截断阶数>   --pol TE|TM   --eig qr|lr
//       --all-layers（均匀层也当作一般层，走本征迭代）   --threads <线程数>
#include <cmath>
#include <cstdlib>
#include <string>

#include "stablercwa/stablercwa.hpp"
#include "say.hpp"
#include "structures.hpp"

using namespace stablercwa;
using examples::say;          // 用法同 std::printf，格式串里可以有中文（缘由见 say.hpp）
using examples::say_err;      // 同上，写到标准错误

int main(int argc, char** argv) {
    std::string which = "single", pol_name = "TE", eig_name = "qr";
    int m = 100, threads = 1;
    bool all_layers = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--case") which = next();
        else if (a == "--M") m = std::atoi(next().c_str());
        else if (a == "--pol") pol_name = next();
        else if (a == "--eig") eig_name = next();
        else if (a == "--threads") threads = std::atoi(next().c_str());
        else if (a == "--all-layers") all_layers = true;
        else {
            say_err("不认识的选项 %s\n", a.c_str());
            say_err("选项：--case single|absorb|trapezoid|wide  --M <截断阶数>  --pol TE|TM  --eig qr|lr  --all-layers  --threads <线程数>\n");
            return 2;
        }
    }
    // 取值写错时直接报错，免得悄悄按默认值算、表头却照抄输入
    if (which != "single" && which != "absorb" && which != "trapezoid" && which != "wide") {
        say_err("--case 只能是 single、absorb、trapezoid、wide\n");
        return 2;
    }
    if (pol_name != "TE" && pol_name != "TM") {
        say_err("--pol 只能是 TE 或 TM\n");
        return 2;
    }
    if (eig_name != "qr" && eig_name != "lr") {
        say_err("--eig 只能是 qr 或 lr\n");
        return 2;
    }

    examples::Mdg mdg;                                // single、absorb、trapezoid：单周期光栅
    if (which == "absorb") mdg.n_tooth = cplx(1.46, 0.05);
    const examples::Ridges ridges;                    // wide：四脊大周期光栅
    rcwa::Stack stack;
    double wavelength = mdg.wavelength, theta = mdg.theta_deg;
    int order = 1;                                    // 关心的反射级次
    if (which == "wide") {
        stack = examples::ridges_on_mirror(ridges);
        wavelength = ridges.wavelength;
        theta = ridges.theta_deg;
        order = ridges.order;
    } else {
        stack = which == "trapezoid" ? examples::trapezoid(16, mdg) : examples::single_period(mdg);
    }
    if (m < std::abs(order)) {
        say_err("M 至少要取到 %d，才包含第 %d 级\n", std::abs(order), order);
        return 2;
    }

    rcwa::Options opt;
    opt.eig = eig_name == "lr" ? EigMethod::LR : EigMethod::QR;
    opt.homogeneous_fast_path = !all_layers;
    opt.parallel_layers = threads > 1;
    opt.diagnostics = true;
    set_num_threads(threads);

    const rcwa::Polarization pol = pol_name == "TM" ? rcwa::Polarization::TM : rcwa::Polarization::TE;
    const rcwa::Result r = rcwa::solve(stack, wavelength, theta, m, pol, opt);

    const double total = r.reflectance() + r.transmittance();
    say("结构 %s，%s，M = %d（N = %d），本征分解用 %s 迭代%s\n", which.c_str(), pol_name.c_str(), m, 2 * m + 1,
        eig_name == "lr" ? "LR" : "QR", all_layers ? "，均匀层也走本征迭代" : "");
    say("  第 %d 级反射效率  %.6f %%\n", order, 100.0 * r.DEr[m + order]);
    say("  R = %.6f %%   T = %.6f %%   R + T = %.6f %%\n", 100.0 * r.reflectance(), 100.0 * r.transmittance(),
        100.0 * total);
    say("  本征对残差（最差的一层） %.2e   线性求解后向误差 %.2e   本征求解 info = %d\n", r.max_eig_residual,
        r.max_solve_backward_error, r.eig_info);
    say("  用时 %.3f s：本征分解 %.3f，线性求解 %.3f，矩阵乘法 %.3f；稠密本征分解 %d 次，稠密求解 %d 次\n",
        r.timing.total, r.timing.eig, r.timing.linear, r.timing.product, r.n_eig, r.n_dense_solve);
    return 0;
}
