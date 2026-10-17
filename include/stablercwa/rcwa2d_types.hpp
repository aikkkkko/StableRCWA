// stablercwa/rcwa2d_types.hpp —— 二维（交叉光栅）RCWA 的公共类型：几何、选项、结果
//
// 约定与数值依据：gui/2drcwa/rcwa2d.py（权威参照）与 gui/validation/field_reference_2d/CONTRACT_2D.md。
// 本头文件只放类型和下标约定，不含算法；算子在 rcwa2d_operators.hpp，求解在 rcwa2d.hpp。
// 所有新公共名字都带 2D 后缀，使它能与 rcwa1d.hpp、rcwa1d_conical.hpp 在同一个翻译单元里共存。
//
// 几何：矩形格子（x 周期 Lx、y 周期 Ly）；每个结构层由 y 分段构成，每个 y 分段内沿 x 分段常数（Layer2D::patterned）；
// 均匀层只有一个折射率。z 由入射侧指向出射侧，结构上表面 z = 0；时间因子 exp(−iωt)。
// 谐波 (m, n) 的归一化横向波矢（以 k0 为单位）：
//     Kx = nI sinθ cosφ − m λ/Lx，   Ky = nI sinθ sinφ − n λ/Ly；     场的横向因子 exp[−i k0 (Kx x + Ky y)]。
// 谐波排成一列的次序与 Python 的 Lattice.harmonics 相同：m 在外层、n 在内层（C 序），
//     下标 index(m, n) = (m + M1)·(2·M2 + 1) + (n + M2)，m = −M1…M1，n = −M2…M2，N = (2M1+1)(2M2+1)。
#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "eig_types.hpp"
#include "matrix.hpp"

namespace stablercwa {
namespace rcwa {

// 因式分解规则：Li = Li 1997 的嵌套因式分解（Cx、Cy 用逆规则，生产默认）；Laurent = 纯 Laurent 规则（收敛差，仅作对照）。
enum class Factorization2D { Li, Laurent };

// 一次求解的结局。数值失败不抛异常：结果数组填 NaN，status 与 message 说明原因；参数非法抛 std::invalid_argument。
enum class Status2D {
    Ok,
    NumericalFailure,    // 本征迭代未收敛、线性方程组奇异、或中途出现 NaN/Inf
    RayleighAnomaly,     // 某个均匀介质（入射区、出射区或均匀层）里有衍射级 kz 恰为 0；message 列出区域和 (m, n)
};

// 沿 x 的一维分段常数剖面：edges 递增，首尾恰好相差一个 x 周期（起点可以不是 0），n_values 比 edges 少一个，可为复数。
struct XProfile2D {
    std::vector<double> edges;
    std::vector<cplx> n_values;
};

struct Layer2D {
    double d = 0.0;                          // 层厚
    bool homogeneous = true;
    cplx n = 1.0;                            // 均匀层折射率；吸收介质写成 n + i·k，k > 0
    std::vector<double> y_edges;             // 结构层：K+1 个 y 分段边界，首尾相差一个 y 周期
    std::vector<XProfile2D> x_profiles;      // 结构层：K 个 x 剖面，第 k 个对应 [y_edges[k], y_edges[k+1])
    std::string name;                        // 只用于出错信息

    static Layer2D uniform(double thickness, cplx index, std::string label = {}) {
        Layer2D layer;
        layer.d = thickness;
        layer.n = index;
        layer.name = std::move(label);
        return layer;
    }
    static Layer2D patterned(double thickness, std::vector<double> y_boundaries, std::vector<XProfile2D> rows,
                             std::string label = {}) {
        Layer2D layer;
        layer.d = thickness;
        layer.homogeneous = false;
        layer.y_edges = std::move(y_boundaries);
        layer.x_profiles = std::move(rows);
        layer.name = std::move(label);
        return layer;
    }
};

// 自上而下：入射介质 nI | 各层 | 出射介质 nII。入射介质必须无吸收（效率归一用 nI cosθ）。
struct Stack2D {
    double nI = 1.0;
    cplx nII = 1.0;
    std::vector<Layer2D> layers;
    double Lx = 1.0, Ly = 1.0;
};

struct Options2D {
    // 2N 阶 P·Q 的本征分解方法。用 QR（默认）：对冻结参照的 21 个成功算例最大差 3e-13。
    // LR 在二维上不可靠（2N 阶非对称矩阵、本征值成簇时 N=169 的方柱差到 1e-2，info 仍是 0）：solve2d 对 LR 始终检查
    // 本征残差，超过 1e-9 就返回 NumericalFailure，不会静默给出错误结果；小矩阵上 LR 通过检查时可用，但没有理由选它。
    EigMethod eig = EigMethod::QR;
    Factorization2D factorization = Factorization2D::Li;
    bool homogeneous_fast_path = true;                 // 均匀层的 W、V、传播块与均匀–均匀界面用 2×2 分块对角（每块 N 维对角阵）快路径
    bool diagnostics = false;
    bool parallel_layers = false;                      // 结构层的算子装配与本征分解用 parallel_for 分给线程池
    std::function<EigResult(const Matrix&)> custom_eig;   // 若设置则代替 eig()，约定同一维的 Options::custom_eig
};

struct Timing2D { double total = 0, operators = 0, eig = 0, linear = 0, product = 0; };

// 入射偏振的 Jones 矢量 [Es, Ep]（不在 solve2d 里归一，与 Python 一致）：s 垂直于入射面，p 在入射面内。
using Jones2D = std::array<cplx, 2>;

// Python 的 psi/delta 入口：Es = sin ψ · exp(iδ)，Ep = cos ψ。ψ=90° 为 s（浮点 cos90° 不一定精确为零）。
inline Jones2D jones_from_angles_2d(double psi_deg = 90.0, double delta_deg = 0.0) {
    const double psi = psi_deg * kPi / 180.0, delta = delta_deg * kPi / 180.0;
    return {std::sin(psi) * std::exp(kI * delta), cplx{std::cos(psi), 0.0}};
}

// 谐波排成一列的下标（与 Python 一致，见文件头）。
inline int harmonic_index_2d(int m, int n, int M1, int M2) { return (m + M1) * (2 * M2 + 1) + (n + M2); }

struct Result2D {
    Status2D status = Status2D::Ok;
    std::string message;                   // status != Ok 时的原因（中文，含区域与 (m, n)）
    int M1 = 0, M2 = 0, N = 0, i0 = 0;     // i0 是 (0,0) 级的下标
    std::vector<int> m, n;                 // 长 N：各列对应的 (m, n)
    double k0 = 0.0, theta = 0.0, phi = 0.0;   // k0 = 2π/λ；θ、φ 为弧度
    RealVector Kx, Ky;                     // 长 N，以 k0 为单位
    Vector kzI, kzII;                      // 长 N，kz/k0：传播级取实正，倏逝级取 +i|kz|（见 rcwa2d.py 的 _kz_over_k0）
    Vector Ex_r, Ey_r, Ez_r, Ex_t, Ey_t, Ez_t;   // 各级反射 / 透射电场的三个分量（入射电场矢量范数 = 入射 Jones 的范数）
    Vector Rs, Rp, Ts, Tp;                 // 各级 s、p 复振幅（电场振幅，不含能流因子）
    RealVector DEr, DEt;                   // 各级效率；非传播级为 0
    std::vector<char> prop_r, prop_t;      // 非零 = 传播级（Re kz > 1e-12）
    double reflectance() const { double s = 0; for (double v : DEr) s += v; return s; }
    double transmittance() const { double s = 0; for (double v : DEt) s += v; return s; }
    int eig_info = 0;                      // 本征求解的 info（多层取第一个非零值）
    int n_eig = 0, n_dense_solve = 0;
    double max_eig_residual = 0;           // diagnostics=true 时填写
    Timing2D timing;
};

}  // namespace rcwa
}  // namespace stablercwa
