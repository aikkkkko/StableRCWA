// stablercwa/rcwa1d_conical.hpp —— 一维周期结构的圆锥入射：各级的 2×2 Jones 矩阵与衍射效率
//
//   namespace rcwa = stablercwa::rcwa;
//   rcwa::Stack stack;                                   // 结构的写法与 rcwa1d.hpp 完全相同
//   stack.nI = 1.0;  stack.nII = 1.46;  stack.period = 0.574;
//   stack.layers.push_back(rcwa::Layer::grating(0.38, {0.0, 0.172, 0.574}, {1.46, 1.0}));
//   rcwa::ConicalResult r = rcwa::solve_conical(stack, 1.053, 66.36, 17.0, 20);   // θ = 66.36°，φ = 17°，M = 20
//   r.Jr[20 + m]、r.Jt[20 + m] 是第 m 级反射、透射的 Jones 矩阵，r.DEr[20 + m].sp 是“p 入、s 出”的反射效率
//   rcwa::ConicalPolarized c = r.for_angles(45.0, 90.0); // 指定入射偏振（这里是圆偏振）之后的各级振幅与效率
//
// 约定。rcwa1d.hpp 文件头的约定全部沿用（时间因子 exp(−iωt)，z 轴由入射侧指向出射侧，长度用同一单位，
// 波矢以 k0 = 2π/λ 为单位），另外增加：
//   · 方位角 φ：入射面与 x 轴（光栅矢量方向）的夹角。φ = 0 就是 rcwa1d.hpp 的平面内入射；
//   · 第 m 级的横向波矢 kx_m = nI·sin θ·cos φ − m·λ/Λ，ky = nI·sin θ·sin φ（各级相同），
//     场随 x、y 的变化是 exp[−i·k0·(kx_m·x + ky·y)]；
//   · 各均匀介质里 kz_m = sqrt(n² − kx_m² − ky²)，取 Im ≥ 0 的一支，Im ≈ 0 时取 Re ≥ 0；
//   · 第 m 级的衍射面（波矢与 z 轴所在的平面）方位角 φ_m = atan2(ky, kx_m)；沿 z 轴传播的级次（|(kx_m, ky)| ≤ 1e-14：
//     正入射的零级，或者恰好沿法线衍射的某一级，这时算出来的 kx_m 只是舍入残差，可正可负）取入射的 φ。
//     记 c = cos φ_m、s = sin φ_m，逐级次的切向基是
//         Es = −s·Ex + c·Ey（垂直于衍射面），    Et = c·Ex + s·Ey（衍射面与 xy 平面的交线方向）；
//   · 物理的 s 分量就是 Es；p 分量是电场在衍射面内、垂直于波矢的分量，它与切向分量的关系是
//         下行波（入射、透射）Et = +(kz_m/n)·Ep，    上行波（反射）Et = −(kz_m/n)·Ep，
//     这样每一束波都有 p̂ × ŝ = k̂；
//   · Jones 矩阵的行是输出 (s, p)，列是输入 (s, p)；入射波是零级、单位电场幅值。
//     Jr[m].ab、Jt[m].ab 是输入 b 偏振时第 m 级反射、透射电场的 a 分量；
//   · 入射偏振用角度写成 [Es, Ep] = [sin ψ·exp(iδ), cos ψ]：ψ = 90° 是 s，ψ = 0° 是 p；
//   · 效率：flux_r[m] = Re(kzI_m)/(nI·cos θ)，flux_t[m] = Re(kzII_m)/(nI·cos θ)（倏逝级为 0），
//     通道效率 DEr[m].ab = flux_r[m]·|Jr[m].ab|²；给定入射偏振之后 DEr[m] = flux_r[m]·(|Rs|² + |Rp|²)；
//   · 入射介质、出射介质都必须是实折射率（否则 p 分量与效率无从定义），各层可以有吸收。θ 不能是 ±90°。
// φ = 0 时 φ_m 只取 0 或 π，Es = σ_m·Ey，σ_m = cos φ_m（以 r.phi_m_deg[m] 为准）：kx_m > 0 时 +1，kx_m < 0 时 −1，
// 上面说的沿 z 轴传播的级次按 kx_m = 0 算，取 +1。与 rcwa::solve 的 TE 振幅相比，
// 第 m 级的 Jss 多一个因子 σ_m·σ_0；p 与 TM（主场 Hy）之间除了同样的符号，透射还差一个因子 nI/nII。
//
// 算法：
//   1. 光栅层做两次 N = 2M+1 阶的本征分解：
//          A = Kx² − ⟦ε⟧ 的本征对 (λ1, W1)，          B·C 的本征对 (λ2, W2)，
//          B = Kx·⟦ε⟧⁻¹·Kx − I，  C = ⟦1/ε⟧⁻¹（逆规则），
//      ky 只把本征值平移：q1 = sqrt(λ1 + ky²)，q2 = sqrt(λ2 + ky²)，分支同 rcwa1d.hpp。
//      直角切向基下的 2N 个模式（行是 [Ex; Ey] 与配对的磁场 [hx; hy]，列先是第一组、后是第二组）不需要任何矩阵求逆：
//          Wxy = [ 0        W2·λ2             ]        Vxy = [ −W1·λ1·q1        0        ]
//                [ W1·q1²   ky·⟦ε⟧⁻¹·Kx·C·W2 ]              [ −ky·Kx·W1·q1    −C·W2·q2  ]
//      再逐级次旋转到 (s, t) 基（电场 [Es; Et]，磁场 [−ht; hs]），每一列按 [W; V] 的 2-范数归一；
//   2. 均匀介质在 (s, t) 基下各级、两支都互不耦合，模式矩阵是对角的。这里取
//          s 支：w = 1，v = q；       t 支：w = kz/n，v = −i·n；       q = −i·kz（两支相同），
//      t 支的振幅因此就是物理的 p 分量（上行波差一个负号），而且某一级恰好掠射（kz = 0）时 w、v 仍然有限；
//   3. 界面块、传播块、自下而上的总反射递推、自上而下推入射波，都与 rcwa1d.hpp 相同，只是阶数是 2N，
//      右端是 s、p 两列入射，一次递推同时求出；
//   4. 外侧两个介质里的振幅直接就是 Jones 矩阵的各元。
// 只算远场，不做层内场的重建。
//
// 数值失败不抛异常：本征迭代没有收敛、某个线性方程组奇异、中途出现了 NaN/Inf、或者矩阵函数不能可靠计算时，Jr、Jt、DEr、DEt 全部是 NaN，eig_info 给出本征求解的 info；kx、ky、kzI、kzII、flux_r、flux_t 这些只由
// 入射条件决定的量照常填写。参数非法抛 std::invalid_argument。
//
// 恰好落在 Rayleigh 异常点上（某一级在某个均匀介质里恰好掠射，kz 精确为零）时，处理原则同 rcwa1d.hpp：
//   · 发生在入射介质里：不妨碍求解，这一级的通量为零；
//   · 发生在均匀层里：这一级的上行、下行两个解重合，按数值失败处理（结果全部是 NaN）。纯膜系例外：
//     没有被激发的级次振幅本来就是零；
//   · 发生在出射介质里：它上面紧挨着均匀层、且走默认的快速路径时照常求解，其余情形按数值失败处理。
// ky ≠ 0 时，约化的 A 或 B·C 接近零本征值会使两组模式的构造退化。
// 每层独立检查 η = min|λ| / |ky|^1.5；η < 1e-2 时仅该层切换到完整 2N 阶
// 算符的 Schur 基，传播 exp(−k0·d·G) 保留上三角耦合（包括 Jordan 块）。
// min_abs_lambda 与 degenerate_mode_warning 保留接近退化点的诊断含义，警告不表示精度失保。
// 附加 Schur 分解使用内置 QR，每层另计一次 n_eig，并计入 n_schur；custom_eig
// 仍负责原两次 N 阶分解。实际 q=0、矩阵函数或线性求解失败仍返回 NaN。
// ky=0 时保留原模式算法；默认快速路径的均匀层仍使用解析模式。
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rcwa1d.hpp"

namespace stablercwa {
namespace rcwa {

struct Jones {                      // 一级的 2×2 Jones 矩阵：ab = 输出 a ← 输入 b
    cplx ss, sp, ps, pp;
};
struct JonesEfficiency {            // 四个通道的效率：flux·|J_ab|²
    double ss = 0, sp = 0, ps = 0, pp = 0;
};

struct ConicalPolarized {           // 指定入射偏振之后的各级振幅与效率
    cplx Es, Ep;                    // 归一化之后的入射 Jones 矢量
    Vector Rs, Rp, Ts, Tp;          // 各级反射 / 透射电场的 s、p 分量
    RealVector DEr, DEt;            // 各级效率（两个输出偏振之和）
    RealVector DEr_s, DEr_p, DEt_s, DEt_p;   // 按输出偏振分开
    double reflectance() const {    // Σ DEr
        double sum = 0.0;
        for (double v : DEr) sum += v;
        return sum;
    }
    double transmittance() const {  // Σ DEt
        double sum = 0.0;
        for (double v : DEt) sum += v;
        return sum;
    }
};

struct ConicalResult {
    std::vector<int> orders;        // −M … +M；第 m 级的下标是 m + M
    RealVector kx;                  // kx_m / k0
    double ky = 0;                  // ky / k0
    RealVector phi_m_deg;           // 各级衍射面的方位角，度
    Vector kzI, kzII;               // 入射、出射介质里各级的 kz / k0
    RealVector flux_r, flux_t;      // 各级沿 z 的能流与入射能流之比（单位电场幅值时）；倏逝级为 0
    std::vector<Jones> Jr, Jt;      // 各级反射、透射的 Jones 矩阵
    std::vector<JonesEfficiency> DEr, DEt;   // 各级四个通道的效率

    // 任意入射 Jones 矢量 [Es, Ep]；内部归一化到单位幅值，零矢量（或含非有限数）抛 std::invalid_argument。
    ConicalPolarized for_input(cplx Es, cplx Ep) const {
        const double norm = std::hypot(std::abs(Es), std::abs(Ep));
        if (!(norm > 0.0) || !std::isfinite(norm))
            throw std::invalid_argument("rcwa: the input Jones vector must be finite and nonzero");
        ConicalPolarized out;
        out.Es = Es / norm;
        out.Ep = Ep / norm;
        const std::size_t size = Jr.size();
        out.Rs.resize(size);
        out.Rp.resize(size);
        out.Ts.resize(size);
        out.Tp.resize(size);
        out.DEr.resize(size);
        out.DEt.resize(size);
        out.DEr_s.resize(size);
        out.DEr_p.resize(size);
        out.DEt_s.resize(size);
        out.DEt_p.resize(size);
        for (std::size_t j = 0; j < size; ++j) {
            out.Rs[j] = Jr[j].ss * out.Es + Jr[j].sp * out.Ep;
            out.Rp[j] = Jr[j].ps * out.Es + Jr[j].pp * out.Ep;
            out.Ts[j] = Jt[j].ss * out.Es + Jt[j].sp * out.Ep;
            out.Tp[j] = Jt[j].ps * out.Es + Jt[j].pp * out.Ep;
            out.DEr_s[j] = flux_r[j] * std::norm(out.Rs[j]);
            out.DEr_p[j] = flux_r[j] * std::norm(out.Rp[j]);
            out.DEt_s[j] = flux_t[j] * std::norm(out.Ts[j]);
            out.DEt_p[j] = flux_t[j] * std::norm(out.Tp[j]);
            out.DEr[j] = out.DEr_s[j] + out.DEr_p[j];
            out.DEt[j] = out.DEt_s[j] + out.DEt_p[j];
        }
        return out;
    }
    // 用角度给出入射偏振：[Es, Ep] = [sin ψ·exp(iδ), cos ψ]。ψ = 90° 是 s，ψ = 0° 是 p；δ 是 s 相对 p 的相位。
    ConicalPolarized for_angles(double psi_deg, double delta_deg = 0.0) const {
        const double psi = psi_deg * kPi / 180.0, delta = delta_deg * kPi / 180.0;
        return for_input(std::sin(psi) * cplx{std::cos(delta), std::sin(delta)}, cplx{std::cos(psi), 0.0});
    }
    // Σ_m (flux_r·Jr†Jr + flux_t·Jt†Jt)，按 [0][0]、[0][1]、[1][0]、[1][1] 存放（下标 0 是 s，1 是 p）。
    // 无吸收结构应为单位矩阵：对角元是 s、p 入射各自的能量守恒，非对角元为零说明两种入射的散射场在能流上正交。
    std::array<cplx, 4> flux_unitarity() const {
        std::array<cplx, 4> u{};
        for (std::size_t j = 0; j < Jr.size(); ++j) {
            const Jones* both[2] = {&Jr[j], &Jt[j]};
            const double flux[2] = {flux_r[j], flux_t[j]};
            for (int k = 0; k < 2; ++k) {
                const Jones& a = *both[k];
                u[0] += flux[k] * (std::norm(a.ss) + std::norm(a.ps));
                u[1] += flux[k] * (std::conj(a.ss) * a.sp + std::conj(a.ps) * a.pp);
                u[2] += flux[k] * (std::conj(a.sp) * a.ss + std::conj(a.pp) * a.ps);
                u[3] += flux[k] * (std::norm(a.sp) + std::norm(a.pp));
            }
        }
        return u;
    }

    double max_eig_residual = 0;            // 以下各项含义同 rcwa::Result
    double max_solve_backward_error = 0;
    int eig_info = 0;
    int n_eig = 0, n_dense_solve = 0;
    int n_schur = 0;                // 切换到完整 2N 阶 Schur 传播的层数；每层另计一次 n_eig
    Timing timing;

    // 圆锥入射特有的两项（见文件头关于奇点的说明）。只要做过本征分解就填写，与 diagnostics 无关，求解失败时也给出：
    // 做了本征分解的各层、两支全部本征值（加 ky² 之前的 λ1、λ2）的模的最小值；没有这样的层时是 +∞。
    double min_abs_lambda = std::numeric_limits<double>::infinity();
    // ky ≠ 0 且 min_abs_lambda < 1e-2·|ky|^1.5：触发该层的 Schur 传播。
    bool degenerate_mode_warning = false;
};

struct ConicalPoint { double wavelength = 1.0; double theta_deg = 0.0; double phi_deg = 0.0; };   // 扫描里的一个点

namespace detail {
namespace conical {   // 圆锥入射专用的内部工具；与维数无关的部分（Work、Op、SBlock、stack_on 等）直接用上一层的

// η = min|λ| / |ky|^1.5 小于此阈值时，仅该层切换为完整 Schur 传播。
inline constexpr double kDegenerateWarn = 1e-2;

// 折射率为 n 的均匀介质里某一级的 kz = sqrt(n² − kx² − ky²)（以 k0 为单位）：取 Im ≥ 0 的一支，Im ≈ 0 时取 Re ≥ 0。
// 折射率是实数时根号下也是实数，分正负两种情形开方：传播级的虚部、倏逝级的实部都严格为零，恰好掠射时严格为零。
inline cplx longitudinal(cplx n, double kx, double ky) {
    const cplx z = n * n - kx * kx - ky * ky;
    if (z.imag() == 0.0)
        return z.real() >= 0.0 ? cplx{std::sqrt(z.real()), 0.0} : cplx{0.0, std::sqrt(-z.real())};
    cplx kz = std::sqrt(z);
    if (kz.imag() < 0.0) kz = -kz;
    if (std::abs(kz.imag()) < 1e-12 && kz.real() < 0.0) kz = -kz;
    return kz;
}

// 一个区域（入射介质、某一层、出射介质）的 2N 个模式。层内的切向场（各 2N 个谐波：前 N 个是 s 分量，后 N 个是 t 分量）
// G 非空时 W 是 Schur 基，下面的逐模式指数应换成完整矩阵 exp(−k0·z·G)。
//     电场 = W·[exp(−k0·q·z)·a⁺ + exp(−k0·q·(d − z))·a⁻]，     磁场 = V·[−exp(−k0·q·z)·a⁺ + exp(−k0·q·(d − z))·a⁻]。
struct Region {
    bool analytic = false;   // true：W = diag(w)、V = diag(v)，不存矩阵
    Vector q;                // 纵向传播常数（以 k0 为单位），长度 2N
    Vector w, v;             // analytic 时 W、V 的对角元：前 N 个是 s 支，后 N 个是 t 支
    Matrix W, V;             // 一般模式
    Matrix G;                // 非空时是 Schur 基中的上三角平方根，传播必须保留耦合
};

struct FieldMaterial {
    bool analytic = false;
    cplx epsilon{};          // 均匀快速路径用标量；一般层接收组装时已有的两个算符
    Matrix inv_eps, c;
};

struct FieldCapture {
    bool available = false;
    std::vector<Region> modes;                 // 上介质、有效层、下介质
    std::vector<FieldMaterial> materials;       // 只对应有效层，零厚层不新增本征分解
    std::vector<Matrix> a_plus, a_minus;        // 每层 2N×2，列次序为 s、p 入射
    RealVector kx, cs, sn;                     // 复用同次求解的归一波矢和逐级旋转，含法线级延拓
    double ky = 0.0;
    Matrix reflected, transmitted;             // 外部模式的两列响应，尚未应用反射 p 的 Jones 负号
};

inline bool valid_capture(const FieldCapture& fields) {
    for (const Region& m : fields.modes)
        if (!finite_vector(m.q) || !finite_vector(m.w) || !finite_vector(m.v) ||
            !m.W.all_finite() || !m.V.all_finite() || !m.G.all_finite()) return false;
    for (const FieldMaterial& m : fields.materials)
        if ((m.analytic && !is_finite(m.epsilon)) || !m.inv_eps.all_finite() || !m.c.all_finite()) return false;
    for (const auto* amplitudes : {&fields.a_plus, &fields.a_minus})
        for (const Matrix& a : *amplitudes) if (!a.all_finite()) return false;
    return fields.reflected.all_finite() && fields.transmitted.all_finite();
}

inline void clear_unexcited(Matrix& a, int orders, int M) {
    for (int c = 0; c < a.cols(); ++c)
        for (int j = 0; j < a.rows(); ++j) if (j % orders != M) a(j, c) = cplx{};
}

// 均匀介质的解析模式：s 支 (w, v) = (1, q)，t 支 (w, v) = (kz/n, −i·n)，q = −i·kz。
// 与“t 支也取 w = 1、v = −n²/q”相比只是把 t 支的每一列乘了 kz/n：振幅变成物理的 p 分量，kz = 0 时也不出现无穷大。
// as_dense 为真时把同样的对角阵写成稠密矩阵（homogeneous_fast_path = false 时入射、出射介质这样给出）。
inline cplx longitudinal0(cplx n, double nI, double cosine) {
    const cplx z = (n - nI) * (n + nI) + (nI * cosine) * (nI * cosine);
    if (z.imag() == 0.0)
        return z.real() >= 0.0 ? cplx{std::sqrt(z.real()), 0.0} : cplx{0.0, std::sqrt(-z.real())};
    cplx kz = std::sqrt(z);
    if (kz.imag() < 0.0) kz = -kz;
    if (std::abs(kz.imag()) < 1e-12 && kz.real() < 0.0) kz = -kz;
    return kz;
}
inline Region analytic_region(cplx n, const RealVector& kx, double ky, bool as_dense, double nI, double cosine) {
    const std::size_t orders = kx.size();
    Region r;
    r.q.resize(2 * orders);
    r.w.resize(2 * orders);
    r.v.resize(2 * orders);
    for (std::size_t j = 0; j < orders; ++j) {
        const cplx kz = 2 * j + 1 == orders ? longitudinal0(n, nI, cosine) : longitudinal(n, kx[j], ky);
        const cplx q{kz.imag(), -kz.real()};
        r.q[j] = q;
        r.q[orders + j] = q;
        r.w[j] = 1.0;
        r.v[j] = q;
        r.w[orders + j] = kz / n;
        r.v[orders + j] = cplx{n.imag(), -n.real()};
    }
    if (as_dense) {
        r.W = Matrix::diagonal(r.w);
        r.V = Matrix::diagonal(r.v);
    } else {
        r.analytic = true;
    }
    return r;
}

// 这一层是不是均匀介质：均匀层，或者各段折射率全都相同（数值上完全相等，不留容差）的光栅层。
inline bool is_uniform(const Layer& layer) {
    if (layer.homogeneous) return true;
    for (const cplx& n : layer.n_values)
        if (n != layer.n_values.front()) return false;
    return true;
}

// 需要做本征分解的层用到的几个 N 阶矩阵
struct Operators {
    Matrix a;          // A = Kx² − ⟦ε⟧，本征值 λ1
    Matrix bc;         // B·C，本征值 λ2
    Matrix inv_eps;    // ⟦ε⟧⁻¹
    Matrix c;          // C = ⟦1/ε⟧⁻¹
};

// 组装一层的 A 与 B·C。均匀层在这里被看成只有一段的光栅层（edges = {0, Λ}），与光栅层用同一套公式。
inline Operators assemble(const Layer& layer, double period, const RealVector& kx, int M, Work& work) {
    const std::vector<double> one_edge{0.0, period};
    const std::vector<double>& edges = layer.homogeneous ? one_edge : layer.edges;
    Vector eps, inv;                                    // 各段的 ε = n² 与 1/ε
    if (layer.homogeneous) {
        eps.push_back(layer.n * layer.n);
        inv.push_back(1.0 / (layer.n * layer.n));
    } else {
        for (const cplx& n : layer.n_values) {
            eps.push_back(n * n);
            inv.push_back(1.0 / (n * n));
        }
    }
    const int size = 2 * M + 1;
    Operators out;
    Matrix e = convolution_matrix(edges, eps, period, M);
    out.inv_eps = linear_solve(e, Matrix::identity(size), work);
    out.c = linear_solve(convolution_matrix(edges, inv, period, M), Matrix::identity(size), work);
    // B = Kx·⟦ε⟧⁻¹·Kx − I：Kx 是对角阵，左右各乘一次只是按行、按列缩放
    Matrix b = out.inv_eps;
    for (int j = 0; j < size; ++j) {
        cplx* bj = b.col(j);
        for (int i = 0; i < size; ++i) bj[i] = (kx[i] * bj[i]) * kx[j];
        bj[j] -= 1.0;
    }
    out.bc = multiply(b, out.c, work);
    // A = Kx² − ⟦ε⟧
    cplx* p = e.data();
    const std::size_t total = static_cast<std::size_t>(size) * size;
    for (std::size_t k = 0; k < total; ++k) p[k] = -p[k];
    for (int j = 0; j < size; ++j) e(j, j) += cplx{kx[j] * kx[j], 0.0};
    out.a = std::move(e);
    return out;
}

// 由两次本征分解拼出一层的 2N 个模式（公式见文件头），旋转到逐级次的 (s, t) 基，每一列按 [W; V] 的 2-范数归一。
// te、tm 是 A、B·C 的本征分解；cs、sn 是各级衍射面方位角的余弦、正弦。
// 某一列整个是零（这个模式恰好在截止点上，q = 0）时模式基不完备，按数值失败处理。
inline Region general_region(const EigResult& te, const EigResult& tm, const Operators& op, const RealVector& kx,
                             double ky, const RealVector& cs, const RealVector& sn, Work& work) {
    const int orders = static_cast<int>(kx.size());
    const int n = 2 * orders;
    Region r;
    r.q.resize(static_cast<std::size_t>(n));
    for (int j = 0; j < orders; ++j) {
        r.q[j] = branch(std::sqrt(te.values[j] + ky * ky));
        r.q[orders + j] = branch(std::sqrt(tm.values[j] + ky * ky));
    }
    const Matrix cw = multiply(op.c, tm.vectors, work);          // C·W2
    Matrix coupling;                                             // ky·⟦ε⟧⁻¹·Kx·C·W2；ky = 0 时两组互不耦合，不必算
    if (ky != 0.0) {
        Matrix scaled = cw;
        for (int j = 0; j < orders; ++j) {
            cplx* sj = scaled.col(j);
            for (int m = 0; m < orders; ++m) sj[m] = (ky * kx[m]) * sj[m];
        }
        coupling = multiply(op.inv_eps, scaled, work);
    }
    r.W = Matrix(n, n);
    r.V = Matrix(n, n);
    const auto normalise = [&](int column) {
        cplx* wc = r.W.col(column);
        cplx* vc = r.V.col(column);
        double sum = 0.0;
        for (int i = 0; i < n; ++i) sum += std::norm(wc[i]) + std::norm(vc[i]);
        const double scale = std::sqrt(sum);
        if (!(scale > 0.0) || !std::isfinite(scale)) throw NumericalFailure{};
        for (int i = 0; i < n; ++i) {
            wc[i] = wc[i] / scale;
            vc[i] = vc[i] / scale;
        }
    };
    for (int j = 0; j < orders; ++j) {
        // 第一组：Ex = 0，Ey = W1·q1²，hx = −W1·λ1·q1，hy = −ky·Kx·W1·q1
        {
            const cplx q1 = r.q[j];
            const cplx fy = q1 * q1, gx = -(te.values[j] * q1), gy = -ky * q1;
            const cplx* w1 = te.vectors.col(j);
            cplx* wc = r.W.col(j);
            cplx* vc = r.V.col(j);
            for (int m = 0; m < orders; ++m) {
                const cplx ey = w1[m] * fy, hx = w1[m] * gx, hy = (kx[m] * w1[m]) * gy;
                wc[m] = cs[m] * ey;
                wc[orders + m] = sn[m] * ey;
                vc[m] = -(cs[m] * hx) - sn[m] * hy;
                vc[orders + m] = cs[m] * hy - sn[m] * hx;
            }
            normalise(j);
        }
        // 第二组：Ex = W2·λ2，Ey = ky·⟦ε⟧⁻¹·Kx·C·W2，hx = 0，hy = −C·W2·q2
        {
            const cplx q2 = r.q[orders + j], lambda2 = tm.values[j];
            const cplx* w2 = tm.vectors.col(j);
            const cplx* cwj = cw.col(j);
            const cplx* yj = ky != 0.0 ? coupling.col(j) : nullptr;
            cplx* wc = r.W.col(orders + j);
            cplx* vc = r.V.col(orders + j);
            for (int m = 0; m < orders; ++m) {
                const cplx ex = w2[m] * lambda2, ey = yj ? yj[m] : cplx{}, hy = -(cwj[m] * q2);
                wc[m] = cs[m] * ey - sn[m] * ex;
                wc[orders + m] = cs[m] * ex + sn[m] * ey;
                vc[m] = -(sn[m] * hy);
                vc[orders + m] = cs[m] * hy;
            }
            normalise(orders + j);
        }
    }
    return r;
}

// Upper triangular matrix functions retain Jordan coupling. Fail rather than regularize
// a zero square root, an unrepresentable scale, or a nonfinite intermediate.
inline Matrix triangular_root(const Matrix& t) {
    const int n = t.rows();
    if (t.cols() != n || !t.all_finite()) throw NumericalFailure{};
    Matrix g(n,n);
    for (int i=0;i<n;++i) {
        g(i,i)=branch(std::sqrt(t(i,i)));
        if (g(i,i)==cplx{} || !is_finite(g(i,i))) throw NumericalFailure{};
    }
    for (int gap=1;gap<n;++gap) for (int i=0;i+gap<n;++i) {
        const int j=i+gap;
        cplx v=t(i,j);
        for (int k=i+1;k<j;++k) v-=g(i,k)*g(k,j);
        const cplx den=g(i,i)+g(j,j);
        if (den==cplx{}) throw NumericalFailure{};
        g(i,j)=v/den;
    }
    if (!g.all_finite()) throw NumericalFailure{};
    const Matrix residual=g*g-t;
    const double scale=norm_1(t), error=norm_1(residual);
    if (!residual.all_finite() || !std::isfinite(scale) || !std::isfinite(error) || error > 1e-10*(std::max)(scale, std::numeric_limits<double>::min()))
        throw NumericalFailure{};
    return g;
}

inline Matrix triangular_exp(const Matrix& a) {
    const int n=a.rows();
    if (a.cols()!=n || !a.all_finite()) throw NumericalFailure{};
    const double size=norm_1(a);
    if (!std::isfinite(size)) throw NumericalFailure{};
    const int scaling=size>.25 ? (std::max)(0,static_cast<int>(std::ceil(std::log2(size)+2.))) : 0;
    // Avoid more than 1020 squarings or an underflowed scaling factor.
    if (scaling>1020) throw NumericalFailure{};
    const Matrix b=a*cplx(std::ldexp(1.,-scaling));
    Matrix e=Matrix::identity(n), term=e;
    bool converged=n==0;
    for (int k=1;k<=40 && !converged;++k) {
        term=(term*b)*cplx(1./k);
        e+=term;
        if (!e.all_finite() || !term.all_finite()) throw NumericalFailure{};
        converged=norm_1(term)<=std::numeric_limits<double>::epsilon()*(std::max)(1.,norm_1(e));
    }
    if (!converged) throw NumericalFailure{};
    for (int i=0;i<n;++i) e(i,i)=std::exp(a(i,i)*std::ldexp(1.,-scaling));
    for (int k=0;k<scaling;++k) {
        e=e*e;
        for (int i=0;i<n;++i) e(i,i)=std::exp(a(i,i)*std::ldexp(1.,k+1-scaling));
        if (!e.all_finite()) throw NumericalFailure{};
    }
    return e;
}

inline Region schur_region(const Operators& op, const RealVector& kx, double ky,
                           const RealVector& cs, const RealVector& sn, Work& work, int& info) {
    const int N=static_cast<int>(kx.size()), n=2*N;
    Matrix p(n,n), q(n,n), re(n,n), rv(n,n);
    for (int j=0;j<N;++j) for (int i=0;i<N;++i) {
        const cplx id=i==j ? 1. : 0.;
        p(i,j)=ky*kx[i]*op.inv_eps(i,j);
        p(i,N+j)=id-kx[i]*op.inv_eps(i,j)*kx[j];
        p(N+i,j)=ky*ky*op.inv_eps(i,j)-id;
        p(N+i,N+j)=-ky*op.inv_eps(i,j)*kx[j];
        q(i,j)=ky*kx[i]*id; q(i,N+j)=-op.a(i,j);
        q(N+i,j)=ky*ky*id-op.c(i,j); q(N+i,N+j)=-ky*kx[i]*id;
    }
    for (int i=0;i<N;++i) {
        re(i,i)=-sn[i]; re(i,N+i)=re(N+i,i)=cs[i]; re(N+i,N+i)=sn[i];
        rv(i,i)=-cs[i]; rv(i,N+i)=rv(N+i,i)=-sn[i]; rv(N+i,N+i)=cs[i];
    }
    p=multiply(multiply(re,p,work),rv,work);
    q=multiply(multiply(rv,q,work),re,work);
    Matrix h=multiply(p,q,work);
    if (!h.all_finite()) throw NumericalFailure{};
    EigOptions options; options.vectors=false; options.keep_transform=true; options.max_iter=100;
    const auto t0=Clock::now();
    EigResult e=eig_qr(h,options);
    work.timing.eig+=seconds_since(t0);
    if (e.info!=0) { info=e.info; throw NumericalFailure{}; }
    Region r; r.G=triangular_root(e.T); r.q=r.G.diag(); r.W=std::move(e.Z);
    r.V=multiply(q,r.W,work);
    // Right triangular solve V G = Q Z, without forming an inverse.
    for (int i=0;i<n;++i) for (int j=0;j<n;++j) {
        cplx v=r.V(i,j);
        for (int k=0;k<j;++k) v-=r.V(i,k)*r.G(k,j);
        r.V(i,j)=v/r.G(j,j);
    }
    if (!r.V.all_finite()) throw NumericalFailure{};
    return r;
}

// 两侧都是解析模式的界面：各级、两支都互不耦合，四个块是对角的。由切向场连续
//     w1·(a1⁺ + a1⁻) = w2·(a2⁺ + a2⁻)，   v1·(a1⁺ − a1⁻) = v2·(a2⁺ − a2⁻)
// 得到（都写成除以 w1·v2 + v1·w2 的形式，某一侧恰好掠射时也是有限的）
//     Tdd = 2·w1·v1/D，   Rdu = (w1·v2 − v1·w2)/D，   Rud = −Rdu，   Tuu = 2·w2·v2/D，   D = w1·v2 + v1·w2。
// s 支就是 TE 的 Fresnel 界面（w = 1，v = q），t 支是 TM 的。
inline SBlock diagonal_interface(const Region& up, const Region& low) {
    const std::size_t size = up.q.size();
    Vector tdd(size), rdu(size), rud(size), tuu(size);
    for (std::size_t j = 0; j < size; ++j) {
        const cplx w1 = up.w[j], v1 = up.v[j], w2 = low.w[j], v2 = low.v[j];
        const cplx a = w1 * v2, b = v1 * w2;
        const cplx inv = 1.0 / (a + b);
        tdd[j] = 2.0 * (w1 * v1) * inv;
        rdu[j] = (a - b) * inv;
        rud[j] = (b - a) * inv;
        tuu[j] = 2.0 * (w2 * v2) * inv;
    }
    SBlock s;
    s.Tdd = diag_op(std::move(tdd));
    s.Rdu = diag_op(std::move(rdu));
    s.Rud = diag_op(std::move(rud));
    s.Tuu = diag_op(std::move(tuu));
    return s;
}

// 至少有一侧是一般模式时的匹配矩阵 T± = (W2⁻¹·W1 ± V2⁻¹·V1)/2（含义同 rcwa1d.hpp 的界面公式）。
inline void interface_matching(const Region& up, const Region& low, Work& work, Matrix& tp, Matrix& tm) {
    const int n = static_cast<int>(up.q.size());
    Matrix mm, nn;
    if (low.analytic) {
        // 下侧是均匀介质：W2、V2 对角，两次稠密求解变成按行除以 w2、v2。
        // 其中有零（下侧恰好有一级掠射）时界面上的方程组奇异。
        Vector inv_w(low.w.size()), inv_v(low.v.size());
        for (std::size_t j = 0; j < inv_w.size(); ++j) {
            if (low.w[j] == cplx{} || low.v[j] == cplx{}) throw NumericalFailure{};
            inv_w[j] = 1.0 / low.w[j];
            inv_v[j] = 1.0 / low.v[j];
        }
        mm = scale_rows(up.W, inv_w);
        nn = scale_rows(up.V, inv_v);
    } else if (up.analytic) {
        // 上侧是均匀介质：右端项是对角阵，方程组还是要解（系数矩阵是下层的 W、V）。
        mm = linear_solve(low.W, Matrix::diagonal(up.w), work);
        nn = linear_solve(low.V, Matrix::diagonal(up.v), work);
    } else {
        mm = linear_solve(low.W, up.W, work);
        nn = linear_solve(low.V, up.V, work);
    }
    tp = mm;
    tp += nn;
    tm = std::move(mm);
    tm -= nn;
    cplx* a = tp.data();
    cplx* b = tm.data();
    const std::size_t total = static_cast<std::size_t>(n) * n;
    for (std::size_t k = 0; k < total; ++k) {
        a[k] = 0.5 * a[k];
        b[k] = 0.5 * b[k];
    }
}

// 界面块：Tdd = T+ − T−·T+⁻¹·T−，Rdu = T−·T+⁻¹，Rud = −T+⁻¹·T−，Tuu = T+⁻¹。
inline SBlock interface_block(const Region& up, const Region& low, Work& work) {
    if (up.analytic && low.analytic) return diagonal_interface(up, low);
    const int n = static_cast<int>(up.q.size());
    Matrix tp, tm;
    interface_matching(up, low, work, tp, tm);
    Matrix tpi = linear_solve(tp, Matrix::identity(n), work);
    Matrix rdu = multiply(tm, tpi, work);
    tp -= multiply(rdu, tm, work);        // tp 现在是 Tdd
    Matrix rud = multiply(tpi, tm, work);
    cplx* a = rud.data();
    const std::size_t total = static_cast<std::size_t>(n) * n;
    for (std::size_t k = 0; k < total; ++k) a[k] = -a[k];
    SBlock s;
    s.Tdd = dense_op(std::move(tp));
    s.Rdu = dense_op(std::move(rdu));
    s.Rud = dense_op(std::move(rud));
    s.Tuu = dense_op(std::move(tpi));
    return s;
}

// 最上面的界面块。入射波只有零级的 s、p 两个分量（模式下标 M 与 N+M，振幅都是 1），Tdd 和 Rud 只用得到这两列。
struct TopBlock {
    Op Rdu, Tuu;
    Matrix tdd, rud;      // Tdd、Rud 的第 M 列（s 入射）与第 N+M 列（p 入射），2N × 2
};
inline TopBlock top_interface_block(const Region& up, const Region& low, int M, Work& work) {
    const int n = static_cast<int>(up.q.size());
    const int column[2] = {M, n / 2 + M};
    TopBlock t;
    t.tdd = Matrix(n, 2);
    t.rud = Matrix(n, 2);
    if (up.analytic && low.analytic) {
        SBlock s = diagonal_interface(up, low);
        for (int k = 0; k < 2; ++k) {
            t.tdd(column[k], k) = s.Tdd.d[column[k]];
            t.rud(column[k], k) = s.Rud.d[column[k]];
        }
        t.Rdu = std::move(s.Rdu);
        t.Tuu = std::move(s.Tuu);
        return t;
    }
    Matrix tp, tm;
    interface_matching(up, low, work, tp, tm);
    Matrix tpi = linear_solve(tp, Matrix::identity(n), work);
    Matrix rdu = multiply(tm, tpi, work);
    Matrix tm_e(n, 2);
    for (int k = 0; k < 2; ++k)
        for (int i = 0; i < n; ++i) {
            tm_e(i, k) = tm(i, column[k]);
            t.tdd(i, k) = tp(i, column[k]);
        }
    t.tdd -= rdu * tm_e;                  // Tdd·e = T+·e − Rdu·(T−·e)
    t.rud = tpi * tm_e;                   // Rud·e = −T+⁻¹·(T−·e)
    t.rud *= cplx{-1.0, 0.0};
    t.Rdu = dense_op(std::move(rdu));
    t.Tuu = dense_op(std::move(tpi));
    return t;
}

// a·x，x 是两列右端（2N × 2）
inline Matrix apply(const Op& a, const Matrix& x) {
    using K = Op::Kind;
    if (a.kind == K::Dense) return a.m * x;
    Matrix y(x.rows(), x.cols());
    if (a.kind == K::Diag)
        for (int k = 0; k < x.cols(); ++k)
            for (int i = 0; i < x.rows(); ++i) y(i, k) = a.d[i] * x(i, k);
    return y;
}

// 一级的四个矩阵元是否都有限
inline bool all_finite(const Jones& j) {
    return is_finite(j.ss) && is_finite(j.sp) && is_finite(j.ps) && is_finite(j.pp);
}

inline void fill_nan(ConicalResult& result, int orders) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const cplx z{nan, nan};
    result.Jr.assign(static_cast<std::size_t>(orders), Jones{z, z, z, z});
    result.Jt.assign(static_cast<std::size_t>(orders), Jones{z, z, z, z});
    JonesEfficiency e;
    e.ss = e.sp = e.ps = e.pp = nan;
    result.DEr.assign(static_cast<std::size_t>(orders), e);
    result.DEt.assign(static_cast<std::size_t>(orders), e);
}

}  // namespace conical
}  // namespace detail

// 圆锥入射的主入口。wavelength 是真空波长，theta_deg 是入射角、phi_deg 是方位角（度），M 是单侧截断阶数
// （保留 −M … M 共 N = 2M+1 级，模式数是 2N）。
//
// options 就是 rcwa::solve 用的那个，各字段的含义不变：
//   eig / custom_eig：每个需要本征分解的层做两次 N 阶本征分解（先 A，后 B·C），都用它；custom_eig 若设置则取代 eig。
//   linear：所有稠密线性求解用的方法。不选主元的 LU 只用于演示：这里的模式矩阵里有成块的零，没有行交换时
//     结果常常是坏的，打开 diagnostics 后 max_solve_backward_error 如实反映。
//   homogeneous_fast_path = true（默认）：均匀层用解析模式。在逐级次的 (s, t) 基下它的 W、V 都是对角的，膜与膜的界面、
//     层内传播只存长度 2N 的对角元。最下面一个光栅层以下的膜系整个是逐元素运算；它以上的膜，自下而上递推到那里时
//     总反射已经是稠密矩阵，每个膜界面要多解一次 2N 阶的稠密方程组、多做一次稠密乘法（层内传播仍只是缩放）。
//     厚度为零的层不参加求解（n_eig、n_dense_solve 也不计它），homogeneous_fast_path = false 时同样如此。
//   homogeneous_fast_path = false：每个均匀层都当作只有一段的光栅层，与光栅层走逐行相同的稠密代码
//     （两次本征分解、稠密的界面与递推）；入射、出射介质仍用解析模式，只是同样写成稠密矩阵。
//     用来与快速路径互相核对，也是衡量快速路径省了多少时间的基准。
//   parallel_layers：各层、各支的本征分解互不相干，打开后用 parallel_for 分给线程池，结果与串行逐位相同。
//   diagnostics：填写 max_eig_residual（两支的本征对残差都算）与 max_solve_backward_error。
// 各段折射率全都相同的“光栅层”实际上是均匀层：快速路径下它也用解析模式（这一点与 rcwa::solve 不同，缘由见文件头
// 关于奇点的说明），结果与折射率差趋于零的弱对比光栅连续衔接；homogeneous_fast_path = false 时它与均匀层一样走本征分解。
//
// 本函数可重入，可以被多个线程同时调用；结果与 set_num_threads 设定的线程数无关（逐位相同）。
// 不要在并行区内部（custom_eig 回调、parallel_for / sweep_conical 的循环体里）调用 set_num_threads，那样会死锁。
namespace detail {
inline ConicalResult solve_conical_impl(const Stack& stack, double wavelength, double theta_deg, double phi_deg, int M,
                                       const Options& options, conical::FieldCapture* fields) {
    namespace dt = detail;
    namespace dc = detail::conical;
    const auto t_start = dt::Clock::now();

    // ---- 0. 检查输入
    dt::check_setup(stack.period, wavelength, stack.nI, theta_deg, M, Polarization::TE);
    if (!std::isfinite(phi_deg)) throw std::invalid_argument("rcwa: azimuth angle must be finite");
    dt::check_index(stack.nII);
    if (stack.nII.imag() != 0.0)
        throw std::invalid_argument("rcwa: conical incidence needs a real index in the exit medium");
    for (const Layer& layer : stack.layers) dt::check_layer(layer, stack.period);
    if (options.linear != LinearSolver::LUPartial && options.linear != LinearSolver::LUNoPivot &&
        options.linear != LinearSolver::QRHouseholder)
        throw std::invalid_argument("rcwa: unknown linear solver");
    if (options.eig != EigMethod::QR && options.eig != EigMethod::LR)
        throw std::invalid_argument("rcwa: unknown eigen-solver");

    const int orders = 2 * M + 1;                       // N：保留的级次数
    const int n = 2 * orders;                           // 模式数：s、t 两支
    // 厚度为零的层在物理上不存在：它上下两个界面合起来就是相邻两个区域之间的界面（传播块是单位阵），
    // 所以不参加求解，也就不会把它自己的 Rayleigh 点、奇点带进来。下面的 layers 只含厚度大于零的层。
    std::vector<const Layer*> layers;
    for (const Layer& layer : stack.layers)
        if (layer.d > 0.0) layers.push_back(&layer);
    const int layer_count = static_cast<int>(layers.size());
    const double k0 = 2.0 * kPi / wavelength;
    const double theta = theta_deg * kPi / 180.0, phi = phi_deg * kPi / 180.0;
    const double nI = stack.nI.real();
    const bool fast = options.homogeneous_fast_path;

    // 各级的横向波矢、衍射面方位角，外侧两个介质里的 kz 与能流因子：只由入射条件决定，求解失败时也照常给出
    ConicalResult result;
    const std::size_t count = static_cast<std::size_t>(orders);
    result.orders.resize(count);
    result.kx.resize(count);
    result.phi_m_deg.resize(count);
    result.kzI.resize(count);
    result.kzII.resize(count);
    result.flux_r.resize(count);
    result.flux_t.resize(count);
    result.Jr.resize(count);
    result.Jt.resize(count);
    result.DEr.resize(count);
    result.DEt.resize(count);
    const double sine = nI * std::sin(theta);
    const double ky = sine * std::sin(phi);
    const double flux_unit = nI * std::cos(theta);
    result.ky = ky;
    RealVector cs(count), sn(count);                     // cos φ_m、sin φ_m
    for (int j = 0; j < orders; ++j) {
        const double kxj = sine * std::cos(phi) - (j - M) * wavelength / stack.period;
        const double kt = std::hypot(kxj, ky);
        result.orders[j] = j - M;
        result.kx[j] = kxj;
        if (kt <= 1e-14) {                               // 沿 z 轴传播的级次（正入射的零级）：取入射的方位角
            cs[j] = std::cos(phi);
            sn[j] = std::sin(phi);
            result.phi_m_deg[j] = phi_deg;
        } else {
            cs[j] = kxj / kt;
            sn[j] = ky / kt;
            result.phi_m_deg[j] = std::atan2(ky, kxj) * 180.0 / kPi;
        }
        result.kzI[j] = j == M ? dc::longitudinal0(stack.nI, nI, std::cos(theta)) : dc::longitudinal(stack.nI, kxj, ky);
        result.kzII[j] = j == M ? dc::longitudinal0(stack.nII, nI, std::cos(theta)) : dc::longitudinal(stack.nII, kxj, ky);
        result.flux_r[j] = result.kzI[j].real() / flux_unit;
        result.flux_t[j] = result.kzII[j].real() / flux_unit;
    }
    const RealVector& kx = result.kx;

    dt::Work work;
    work.linear = options.linear;
    work.diagnostics = options.diagnostics;
    if (fields) {
        *fields = dc::FieldCapture{};
        fields->materials.resize(static_cast<std::size_t>(layer_count));
        fields->a_plus.resize(static_cast<std::size_t>(layer_count));
        fields->a_minus.resize(static_cast<std::size_t>(layer_count));
    }

    try {
        // ---- 1. 各区域的模式。区域 0 是入射介质，区域 j+1 是第 j 层，最后一个区域是出射介质。
        std::vector<dc::Region> modes(static_cast<std::size_t>(layer_count) + 2);
        modes.front() = dc::analytic_region(stack.nI, kx, ky, !fast, nI, std::cos(theta));
        modes.back() = dc::analytic_region(stack.nII, kx, ky, !fast, nI, std::cos(theta));
        std::vector<int> eig_layers;                    // 需要做本征分解的层
        // 纯膜系（快速路径下没有任何层要做本征分解）：所有的块都是对角的，只有零级的 s、t 两个模式（下标 M、N+M）被激发。
        bool pure_films = fast;
        for (const Layer* layer : layers) pure_films = pure_films && dc::is_uniform(*layer);
        for (int j = 0; j < layer_count; ++j) {
            const Layer& layer = *layers[j];
            if (fast && dc::is_uniform(layer)) {
                modes[j + 1] = dc::analytic_region(layer.homogeneous ? layer.n : layer.n_values.front(), kx, ky, false, nI, std::cos(theta));
                if (fields) {
                    const cplx index = layer.homogeneous ? layer.n : layer.n_values.front();
                    fields->materials[j].analytic = true;
                    fields->materials[j].epsilon = index * index;
                }
                // 均匀层里某一级恰好掠射（q 精确为零）：这一级的上行、下行两个解重合，直接判为失败（缘由见 rcwa::solve）。
                // 纯膜系里没有被激发的级次例外：它们的振幅本来就是零，见第 3 步末尾。
                for (int k = 0; k < orders; ++k)
                    if (modes[j + 1].q[k] == cplx{} && (!pure_films || k == M)) throw dt::NumericalFailure{};
            } else {
                eig_layers.push_back(j);
            }
        }
        const int eig_count = static_cast<int>(eig_layers.size());

        // 1a. 组装层矩阵
        std::vector<dc::Operators> ops(static_cast<std::size_t>(eig_count));
        for (int i = 0; i < eig_count; ++i) {
            ops[i] = dc::assemble(*layers[eig_layers[i]], stack.period, kx, M, work);
            if (!ops[i].a.all_finite() || !ops[i].bc.all_finite()) throw dt::NumericalFailure{};
        }

        // 1b. 本征分解：每层两次（下标 2i 是 A，2i+1 是 B·C），互不相干，可以并行；计时取这一段的墙上时间。
        std::vector<EigResult> eigs(2 * static_cast<std::size_t>(eig_count));
        const auto decompose = [&](int t) {
            Matrix& a = t % 2 == 0 ? ops[t / 2].a : ops[t / 2].bc;
            eigs[t] = options.custom_eig ? options.custom_eig(a) : stablercwa::eig(a, options.eig);
            if (!options.diagnostics && t % 2 != 0) a = Matrix();             // 用完就释放（诊断还要用它算残差）
        };
        if (eig_count > 0) {                            // 没有要分解的层（纯膜系）时 eig 用时严格为零
            const auto t0 = dt::Clock::now();
            if (options.parallel_layers) parallel_for(0, 2 * eig_count, decompose);
            else for (int t = 0; t < 2 * eig_count; ++t) decompose(t);
            work.timing.eig += dt::seconds_since(t0);
        }
        result.n_eig = 2 * eig_count;
        for (int t = 0; t < 2 * eig_count; ++t) {
            const EigResult& e = eigs[t];
            if (e.info != 0) {
                if (result.eig_info == 0) result.eig_info = e.info;
            } else if (static_cast<int>(e.values.size()) != orders || e.vectors.rows() != orders ||
                       e.vectors.cols() != orders) {
                throw std::invalid_argument("rcwa: the eigen-solver returned a result of the wrong size");
            }
            if (options.diagnostics) {      // 没收敛的没有本征向量，残差记为 NaN
                const Matrix& a = t % 2 == 0 ? ops[t / 2].a : ops[t / 2].bc;
                const double res = e.info == 0 ? eig_residual(a, e) : std::numeric_limits<double>::quiet_NaN();
                result.max_eig_residual = dt::max_keep_nan(result.max_eig_residual, res);
            }
        }
        // 各层约化本征值的最小模用于诊断；真正的传播切换按下面每层的局部最小值决定。
        for (int t = 0; t < 2 * eig_count; ++t) {
            if (eigs[t].info != 0) continue;
            for (const cplx& value : eigs[t].values) {
                const double size = std::abs(value);
                if (size < result.min_abs_lambda) result.min_abs_lambda = size;
            }
        }
        const double ky_scale = std::abs(ky) * std::sqrt(std::abs(ky));                 // |ky|^1.5
        if (ky != 0.0 && result.min_abs_lambda < dc::kDegenerateWarn * ky_scale) result.degenerate_mode_warning = true;
        if (result.eig_info != 0) throw dt::NumericalFailure{};


        // 1c. 拼出 2N 个模式
        for (int i = 0; i < eig_count; ++i) {
            double local_min = std::numeric_limits<double>::infinity();
            for (int t : {2*i, 2*i+1})
                for (cplx value : eigs[t].values) local_min = (std::min)(local_min, std::abs(value));
            if (ky != 0.0 && local_min < dc::kDegenerateWarn * ky_scale) {
                ++result.n_schur;
                ++result.n_eig;
                modes[eig_layers[i] + 1] = dc::schur_region(ops[i], kx, ky, cs, sn, work, result.eig_info);
            } else {
                modes[eig_layers[i] + 1] = dc::general_region(eigs[2 * i], eigs[2 * i + 1], ops[i], kx, ky, cs, sn, work);
            }
            if (fields) {
                fields->materials[eig_layers[i]].inv_eps = std::move(ops[i].inv_eps);
                fields->materials[eig_layers[i]].c = std::move(ops[i].c);
            }
            eigs[2 * i] = EigResult();
            eigs[2 * i + 1] = EigResult();
            ops[i] = dc::Operators();
        }

        // ---- 2. 自下而上叠加各块。块 2j 是区域 j 与 j+1 之间的界面，块 2j+1 是第 j 层内的传播。
        const int block_count = 2 * layer_count + 1;
        std::vector<dt::Op> g(static_cast<std::size_t>(block_count));
        std::vector<dt::Op> r_below;
        if (fields) r_below.resize(static_cast<std::size_t>(layer_count));
        dt::Op r;                                        // 出射介质里没有向上传的波：最底下的总反射为零
        if (!fast) r = dt::dense_op(Matrix(n, n));
        for (int b = block_count - 1; b >= 1; --b) {
            const int j = b / 2;
            dt::SBlock block;
            if (b % 2 == 0) {
                block = dc::interface_block(modes[j], modes[j + 1], work);
                if (!fields) {
                    modes[j + 1].W = Matrix();           // 效率-only 路径仍立即释放模式
                    modes[j + 1].V = Matrix();
                }
            } else {
                if (fields) r_below[j] = r;
                if (modes[j + 1].G.rows() != 0) {
                    Matrix x = dc::triangular_exp(modes[j + 1].G * cplx(-k0 * layers[j]->d));
                    block.Tdd = dt::dense_op(x);
                    block.Tuu = dt::dense_op(std::move(x));
                } else block = dt::propagation_block(modes[j + 1].q, layers[j]->d, k0, !fast);
            }
            dt::Seam seam = dt::stack_on(std::move(block), r, n, work);
            g[b] = std::move(seam.g);
            r = std::move(seam.r);
        }

        // ---- 3. 最上面的界面块，只算零级 s、p 入射对应的两列：
        //     down = (I − Rdu·r)^{-1}·Tdd·e，    up = Rud·e + Tuu·r·down，
        // 然后自上而下逐块乘 g，得到出射介质里的下行振幅。
        Matrix up, down;
        {
            using K = dt::Op::Kind;
            dc::TopBlock top = dc::top_interface_block(modes[0], modes[1], M, work);
            dt::Op k = dt::op_mul(top.Rdu, r, work);
            down = std::move(top.tdd);
            if (k.kind == K::Diag) {
                for (int c = 0; c < 2; ++c)
                    for (int j = 0; j < n; ++j) down(j, c) = down(j, c) / (1.0 - k.d[j]);
            } else if (k.kind == K::Dense) {
                down = dt::linear_solve(dt::identity_minus(std::move(k.m)), down, work);
            }
            up = std::move(top.rud);
            up += dc::apply(top.Tuu, dc::apply(r, down));
            for (int b = 1; b < block_count; ++b) {
                const int j = b / 2;
                if (fields && b % 2 != 0) {
                    fields->a_plus[j] = down;
                    if (pure_films) dc::clear_unexcited(fields->a_plus[j], orders, M);
                }
                down = dc::apply(g[b], down);
                if (fields && b % 2 != 0) {
                    Matrix bottom = down;
                    if (pure_films) dc::clear_unexcited(bottom, orders, M);
                    fields->a_minus[j] = dc::apply(r_below[j], bottom);
                    if (pure_films) dc::clear_unexcited(fields->a_minus[j], orders, M);
                }
            }
        }
        // 外侧介质里 s 支的振幅是 Es，t 支的振幅对下行波是 Ep、对上行波是 −Ep。
        for (int j = 0; j < orders; ++j) {
            result.Jr[j] = Jones{up(j, 0), up(j, 1), -up(orders + j, 0), -up(orders + j, 1)};
            result.Jt[j] = Jones{down(j, 0), down(j, 1), down(orders + j, 0), down(orders + j, 1)};
        }
        // 纯膜系不产生衍射：零级之外的各级没有被激发，振幅严格为零。它们各自的（标量）递推里若出现了 0/0 或 0·Inf
        // （这一级恰好在某一层里掠射，或者在界面两侧同时掠射），并不是求解失败，这里把这些非有限的值还原成零。
        if (pure_films)
            for (int j = 0; j < orders; ++j) {
                if (j == M) continue;
                for (Jones* a : {&result.Jr[j], &result.Jt[j]})
                    for (cplx* v : {&a->ss, &a->sp, &a->ps, &a->pp})
                        if (!dt::is_finite(*v)) *v = cplx{};
            }
        // 中途出现过 NaN/Inf：整个结果按失败处理
        for (int j = 0; j < orders; ++j)
            if (!dc::all_finite(result.Jr[j]) || !dc::all_finite(result.Jt[j])) throw dt::NumericalFailure{};

        // ---- 4. 各通道的衍射效率
        for (int j = 0; j < orders; ++j) {
            const Jones& a = result.Jr[j];
            const Jones& b = result.Jt[j];
            const double fr = result.flux_r[j], ft = result.flux_t[j];
            JonesEfficiency& er = result.DEr[j];
            JonesEfficiency& et = result.DEt[j];
            er.ss = fr * std::norm(a.ss);
            er.sp = fr * std::norm(a.sp);
            er.ps = fr * std::norm(a.ps);
            er.pp = fr * std::norm(a.pp);
            et.ss = ft * std::norm(b.ss);
            et.sp = ft * std::norm(b.sp);
            et.ps = ft * std::norm(b.ps);
            et.pp = ft * std::norm(b.pp);
            // 振幅有限、效率却溢出或成了 NaN（折射率、波长取了极端的数值）：同样按失败处理
            for (double v : {er.ss, er.sp, er.ps, er.pp, et.ss, et.sp, et.ps, et.pp})
                if (!std::isfinite(v)) throw dt::NumericalFailure{};
        }
        if (fields) {
            fields->modes = std::move(modes);
            fields->kx = kx;
            fields->ky = ky;
            fields->cs = std::move(cs);
            fields->sn = std::move(sn);
            fields->reflected = std::move(up);
            fields->transmitted = std::move(down);
            if (pure_films) {
                dc::clear_unexcited(fields->reflected, orders, M);
                dc::clear_unexcited(fields->transmitted, orders, M);
            }
            fields->available = dc::valid_capture(*fields);
            if (!fields->available) *fields = dc::FieldCapture{};
        }
    } catch (const dt::NumericalFailure&) {
        dc::fill_nan(result, orders);
        if (fields) *fields = dc::FieldCapture{};
    }

    result.n_dense_solve = work.n_solve;
    result.max_solve_backward_error = work.max_backward_error;
    result.timing = work.timing;
    result.timing.total = dt::seconds_since(t_start);
    return result;
}
}  // namespace detail

inline ConicalResult solve_conical(const Stack& stack, double wavelength, double theta_deg, double phi_deg, int M,
                                   const Options& options = {}) {
    return detail::solve_conical_impl(stack, wavelength, theta_deg, phi_deg, M, options, nullptr);
}

// 扫描：对每个点独立调用 solve_conical()，用 parallel_for 把各点分给线程池（set_num_threads 设定线程数）。
// 线程池不嵌套，所以每次求解内部自动串行；结果的顺序与 points 相同，且与线程数无关（逐位相同）。
inline std::vector<ConicalResult> sweep_conical(const Stack& stack, const std::vector<ConicalPoint>& points, int M,
                                                const Options& options = {}) {
    std::vector<ConicalResult> results(points.size());
    parallel_for(0, static_cast<int>(points.size()), [&](int i) {
        results[i] = solve_conical(stack, points[i].wavelength, points[i].theta_deg, points[i].phi_deg, M, options);
    });
    return results;
}

}  // namespace rcwa
}  // namespace stablercwa
