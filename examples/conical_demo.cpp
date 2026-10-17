// examples/conical_demo.cpp —— 圆锥入射的命令行演示：各传播级四个偏振通道的效率、指定入射偏振的合计、通量幺正性
//
//   c++ -std=c++17 -O2 -I include -I examples examples/conical_demo.cpp -o conical_demo
//   ./conical_demo                                      # 默认：单周期光栅 + 20 层膜，θ = 66.36°，φ = 10°，M = 20，s 偏振入射
//   ./conical_demo --phi 0                              # 平面内入射：交叉偏振的通道严格为零
//   ./conical_demo --psi 45 --delta 90                  # 圆偏振入射
//   ./conical_demo --case trapezoid --M 40 --threads 2  # 16 个光栅层的本征分解分给 2 个线程
//   ./conical_demo --case absorb --theta 30 --phi 40    # 齿有吸收：R + T < 100%
//   ./conical_demo --theta 46.83                        # 紧挨着圆锥入射特有的奇点（θ ≈ 46.8265° 处光栅层的 A 有零本征值）：末行给出警告
//
// 选项：--case single|absorb|trapezoid|film   --M <截断阶数>   --theta <入射角，度>   --phi <方位角，度>
//       --psi <度> --delta <度>（入射偏振 [Es, Ep] = [sin ψ·exp(iδ), cos ψ]；ψ = 90 是 s，ψ = 0 是 p）
//       --eig qr|lr   --all-layers（均匀层也当作一般层，走本征分解）   --threads <线程数>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

#include "stablercwa/rcwa1d_conical.hpp"
#include "say.hpp"
#include "structures.hpp"

using namespace stablercwa;
using examples::say;          // 用法同 std::printf，格式串里可以有中文（缘由见 say.hpp）
using examples::say_err;      // 同上，写到标准错误

// 把整个字符串读成一个有限的实数；读不成（空串、夹着别的字符、溢出）返回 false
static bool parse_number(const std::string& text, double& value) {
    if (text.empty()) return false;
    char* end = nullptr;
    value = std::strtod(text.c_str(), &end);
    return end == text.c_str() + text.size() && std::isfinite(value);
}
// 同上，并且要求是不超过 limit 的整数
static bool parse_integer(const std::string& text, int limit, int& value) {
    double x = 0.0;
    if (!parse_number(text, x) || x != std::floor(x) || std::abs(x) > limit) return false;
    value = static_cast<int>(x);
    return true;
}

static void print_usage() {
    say_err("选项：--case single|absorb|trapezoid|film  --M <截断阶数>  --theta <度>  --phi <度>  --psi <度>  --delta <度>\n");
    say_err("      --eig qr|lr  --all-layers  --threads <线程数>\n");
}

// 打印一侧（反射或透射）各传播级的四个通道效率，以及指定入射偏振时这一级的效率
static void print_side(const char* title, const rcwa::ConicalResult& r, const std::vector<rcwa::JonesEfficiency>& channel,
                       const RealVector& flux, const RealVector& chosen) {
    int propagating = 0;
    for (double f : flux) propagating += f > 0.0 ? 1 : 0;
    say("%s（传播级 %d 个），效率 %%：\n", title, propagating);
    say("    级次  衍射面方位角      s<-s         s<-p         p<-s         p<-p       本次入射\n");
    for (std::size_t j = 0; j < flux.size(); ++j) {
        if (!(flux[j] > 0.0)) continue;
        const rcwa::JonesEfficiency& e = channel[j];
        say("   %5d   %9.3f   %11.6f  %11.6f  %11.6f  %11.6f  %11.6f\n", r.orders[j], r.phi_m_deg[j], 100.0 * e.ss,
            100.0 * e.sp, 100.0 * e.ps, 100.0 * e.pp, 100.0 * chosen[j]);
    }
}

int main(int argc, char** argv) {
    std::string which = "single", eig_name = "qr";
    examples::Mdg mdg;
    int m = 20, threads = 1;
    double theta = mdg.theta_deg, phi = 10.0, psi = 90.0, delta = 0.0;
    bool all_layers = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        bool ok = true;
        const bool whole = a == "--M" || a == "--threads";     // 这两个选项要的是整数
        if (a == "--case") which = next();
        else if (a == "--M") ok = parse_integer(next(), 100000, m);
        else if (a == "--theta") ok = parse_number(next(), theta);
        else if (a == "--phi") ok = parse_number(next(), phi);
        else if (a == "--psi") ok = parse_number(next(), psi);
        else if (a == "--delta") ok = parse_number(next(), delta);
        else if (a == "--eig") eig_name = next();
        else if (a == "--threads") ok = parse_integer(next(), 4096, threads);
        else if (a == "--all-layers") all_layers = true;
        else {
            say_err("不认识的选项 %s\n", a.c_str());
            print_usage();
            return 2;
        }
        if (!ok) {
            say_err("%s 后面要跟一个%s\n", a.c_str(), whole ? "整数" : "数");
            print_usage();
            return 2;
        }
    }
    // 取值写错时直接报错，免得悄悄按默认值算、表头却照抄输入
    if (which != "single" && which != "absorb" && which != "trapezoid" && which != "film") {
        say_err("--case 只能是 single、absorb、trapezoid、film\n");
        return 2;
    }
    if (eig_name != "qr" && eig_name != "lr") {
        say_err("--eig 只能是 qr 或 lr\n");
        return 2;
    }
    if (m < 0) {
        say_err("--M 不能是负数\n");
        return 2;
    }
    if (threads < 1) {
        say_err("--threads 至少是 1\n");
        return 2;
    }
    if (!(std::abs(theta) < 90.0)) {
        say_err("--theta 要严格在 -90 与 90 之间\n");
        return 2;
    }

    if (which == "absorb") mdg.n_tooth = cplx(1.46, 0.05);
    rcwa::Stack stack;
    if (which == "trapezoid") stack = examples::trapezoid(16, mdg);
    else stack = examples::single_period(mdg);             // single、absorb、film 都从单周期结构出发
    if (which == "film") stack.layers.erase(stack.layers.begin());   // 去掉光栅层，只剩 20 层膜

    rcwa::Options opt;
    opt.eig = eig_name == "lr" ? EigMethod::LR : EigMethod::QR;
    opt.homogeneous_fast_path = !all_layers;
    opt.parallel_layers = threads > 1;
    opt.diagnostics = true;
    set_num_threads(threads);

    const rcwa::ConicalResult r = rcwa::solve_conical(stack, mdg.wavelength, theta, phi, m, opt);
    const rcwa::ConicalPolarized c = r.for_angles(psi, delta);

    say("结构 %s，λ = %.4f，θ = %.4f°，φ = %.4f°，M = %d（N = %d，模式数 %d），本征分解用 %s 迭代%s\n", which.c_str(),
        mdg.wavelength, theta, phi, m, 2 * m + 1, 2 * (2 * m + 1), eig_name == "lr" ? "LR" : "QR",
        all_layers ? "，均匀层也走本征分解" : "");
    say("入射偏振 ψ = %.4f°，δ = %.4f°：Es = %+.6f %+.6fi，Ep = %+.6f %+.6fi\n", psi, delta, c.Es.real(), c.Es.imag(),
        c.Ep.real(), c.Ep.imag());
    if (r.eig_info != 0 || std::isnan(c.reflectance() + c.transmittance())) {
        say("求解失败：本征求解 info = %d（info = 0 时是方程组奇异或恰好落在 Rayleigh 异常点上，把角度或波长挪动一点再试）\n",
            r.eig_info);
        if (r.degenerate_mode_warning)
            say("  落在圆锥入射特有的奇点上：光栅层的 A 有接近零的本征值，min|λ| = %.2e（ky = %.4f）\n", r.min_abs_lambda, r.ky);
        return 1;
    }
    print_side("反射", r, r.DEr, r.flux_r, c.DEr);
    print_side("透射", r, r.DEt, r.flux_t, c.DEt);

    const double total = c.reflectance() + c.transmittance();
    say("本次入射的合计：R = %.6f %%   T = %.6f %%   R + T = %.6f %%\n", 100.0 * c.reflectance(), 100.0 * c.transmittance(),
        100.0 * total);
    // 通量幺正性矩阵 U = Σ (flux_r·Jr†Jr + flux_t·Jt†Jt)：无吸收时是单位矩阵。对角元是 s、p 入射各自的 R + T，
    // 非对角元为零说明两种入射的散射场在能流上正交——它比能量守恒多管三个实数。
    const std::array<cplx, 4> u = r.flux_unitarity();
    const double defect = (std::max)((std::max)(std::abs(u[0] - 1.0), std::abs(u[3] - 1.0)), (std::max)(std::abs(u[1]), std::abs(u[2])));
    say("通量幺正性：s 入射 R + T = %.8f %%，p 入射 R + T = %.8f %%，非对角元 |U_sp| = %.2e；与单位矩阵的最大偏差 %.2e\n",
        100.0 * u[0].real(), 100.0 * u[3].real(), std::abs(u[1]), defect);
    say("  本征对残差（最差的一次） %.2e   线性求解后向误差 %.2e   本征求解 info = %d\n", r.max_eig_residual,
        r.max_solve_backward_error, r.eig_info);
    say("  用时 %.3f s：本征分解 %.3f，线性求解 %.3f，矩阵乘法 %.3f；稠密本征分解 %d 次，稠密求解 %d 次\n",
        r.timing.total, r.timing.eig, r.timing.linear, r.timing.product, r.n_eig, r.n_dense_solve);
    // 圆锥入射特有的奇点：光栅层的 A = Kx² − ⟦ε⟧ 有接近零的本征值而 ky ≠ 0（说明见 rcwa1d_conical.hpp 开头）
    if (r.n_eig > 0)
        say("  各层本征值模的最小值 min|λ| = %.3e%s\n", r.min_abs_lambda,
            r.degenerate_mode_warning ? "：约化模式接近退化，已启用完整 Schur 传播" : "");
    if (r.n_schur > 0) say("  使用完整 Schur 传播的层数：%d\n", r.n_schur);
    return 0;
}
