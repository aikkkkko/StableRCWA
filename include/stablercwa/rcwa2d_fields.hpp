// stablercwa/rcwa2d_fields.hpp —— 二维（交叉光栅）RCWA 的近场：拥有数据的场上下文、任意 z 的六分量谐波列、x-z / y-z 切面与相位
//
//   namespace rcwa = stablercwa::rcwa;
//   rcwa::FieldResult2D run = rcwa::solve2d_with_fields(stack, 1.053, 42.0, -25.0, 4, 4, rcwa::jones_from_angles_2d(90.0));
//   run.result            // 与 solve2d 逐位相同的 Result2D（各级振幅、效率、status）
//   run.fields            // 同一次求解里保留的数据；Stack 之后销毁、修改都不影响它
//   rcwa::HarmonicFields h = rcwa::field_at(run, 0.15);                        // z = 0.15 处六个分量的谐波列（[N, 1]）
//   rcwa::HarmonicFields hs = rcwa::field_at(run, RealVector{-0.2, 0.1, 0.1}); // 一组 z：保留输入顺序与重复值
//   rcwa::FieldGridSlice xz = rcwa::field_slice(run, zs, rcwa::SlicePlane::XZ, 300, 0.37);   // 固定 y = 0.37 的 x-z 面，300 个 x 点
//   rcwa::PhaseGrid ph = rcwa::phase_slice(run, zs, rcwa::SlicePlane::YZ, 300, 0.46, rcwa::FieldComponent::Ez);   // y-z 面上的相位
//
// 数值约定的权威是 gui/2drcwa/rcwa2d.py 的 field_at（第 572–607 行）与 field_slice（第 645–663 行）；合同是
// gui/validation/field_reference_2d/CONTRACT_2D.md，近场的区域归属、六分量与归一化沿用一维合同 gui/validation/field_reference/CONTRACT.md
// 第 2、3、6、7 节。谐波下标、Kx、Ky、时间因子见 rcwa2d_types.hpp 文件头；容器、切面与相位见 rcwa_fields.hpp、rcwa_phase.hpp。
// 本文件不重复它们，只做三件事：把同一次求解保留下来的数据（detail2d::FieldCapture2D，rcwa2d.hpp）装进拥有数据的上下文；
// 按层、按区域批量传播得到任意 z 处的谐波列；把谐波列交给 detail::reconstruct_field_slice 重建切面（每个分量一次矩阵乘法）。
//
// 场的公式（z 向下，结构上表面 z = 0，z_bot = 各层厚度之和，k0 = 2π/λ，q 是模式的衰减常数，exp(−k0·q·Δz) 沿传播方向不增长）：
//   层内（zl = z − z_top）  [Ex;Ey] = W·(a⁺·exp(−k0·q·zl) + a⁻·exp(−k0·q·(d − zl))），  [hx;hy] = V·(−a⁺·exp(…) + a⁻·exp(…))；
//     a⁺ 锚定层顶（向下传播）、a⁻ 锚定层底（向上传播），两个指数的模都 ≤ 1；不能用 exp(+k0·q·d) 反推层底振幅。
//   区域 I（z < 0）         F = inc·exp(−k0·q·z) + R·exp(+k0·q·z)，  G = V_I·(−inc·exp(−k0·q·z) + R·exp(+k0·q·z))，Ei = 1/nI²
//   区域 II（z > z_bot）    F = T·exp(−k0·q·(z − z_bot))，           G = V_II·(−F)，                                  Ei = 1/nII²
//   六分量   Ex、Ey = F 的前后 N 个，hx、hy = G 的前后 N 个，
//            hz = i·(Ky·Ex − Kx·Ey)，  Ez = i·Ei·(Ky·hx − Kx·hy)，  (Hx, Hy, Hz) = −i·(hx, hy, hz)；
//   Ei = Eps⁻¹（结构层，N×N 稠密逆，上下文里已有）；均匀层、区域 I、II 为标量 1/n²。Ez 走 Laurent 阵 Eps 的逆（沿竖直侧壁它是切向、连续的）；
//   用错算符时衍射效率照样收敛、场却是错的，只能靠界面连续性与冻结参照把关（见 tests/test_rcwa2d_fields.cpp）。
// E 对入射 |E| = 1 归一（Jones 向量不在求解里归一：入射电场矢量范数 = Jones 的二范数）；H 存的是 Z0·H，入射 |Z0 H| = nI·|Jones|，
// 即 metadata().normalization = {|Jones|, nI·|Jones|}（原始复场不缩放）。
//
// 区域归属（与 rcwa2d.py:581–594 同一规则，外部优先）：z < 0 为区域 I；z > z_bot 为区域 II（只多 1e-13 也属于区域 II）；其余从第一层开始取
// 第一个满足 z ≤ z_top + d + 1e-12 的层。后果：内界面上、或其下方不超过 1e-12 的 z 属于上一层；z = 0 属于第一层；z = z_bot 属于第一个满足条件的层（即最后一个非零厚的层；层厚为 [0.3, 0, 0] 时是第 0 层）；
// 排在前一层之后的零厚层永远不会被选中（它的阈值与前一层的相同）。**与 Python / 一维不同或 Python 没有的地方**：
//   · 没有任何层的栈：Python 在 z = 0 抛 StopIteration；这里 z = 0 取出射侧（区域 II，与一维合同第 6 节第 1 条一致）；
//   · 第一层是零厚层时，z ∈ [0, 1e-12] 按 Python 的规则取这个零厚层（它的 Ez 用它自己的 Ei）；全部层厚为零（z_bot = 0）时只有 z = 0 取第一层，
//     z > 0 已是区域 II。一维 C++ 在“全部层厚为零”时 z = 0 取出射侧，这里不照那条走（二维沿用 Python 的规则）。
// 层内距离夹到 [0, d]：Python 不夹，选中某层的 z 至多比 z_top + d 多 1e-12，夹过之后传播指数永远不增长，数值差 ≲ 3e-11（审查在 |q| 达 267 的极端几何下实测相对差 ≤ 2.2e-11，只在 z ∈ (z_top+d, z_top+d+1e-12] 起作用）。
//
// 外部区域的每个谐波是一列平面波：E ⊥ k，Z0·H = (k/k0) × E，k/k0 = (−Kx, −Ky, ±kz)（横向因子 exp[−i k0 (Kx x + Ky y)]，下行 +kz、上行 −kz）。
// 所以透射级 Ez = +(Kx·Ex + Ky·Ey)/kz，反射级 Ez = −(Kx·Ex + Ky·Ey)/kz。**Result2D::Ez_r / Ez_t**（从 rcwa2d.py 的 efficiency 照搬，取
// −sign·(Kx·Ex + Ky·Ey)/kz）与这里的 Ez 符号相反、模相同；它们只进效率的 |Ez|²，不影响任何效率，但不能当作近场 Ez 的相位参照用。
//
// 避开 0×∞：区域 I 的入射项只对被激发的零阶 (0, 0) 的两个分量（Ex、Ey）求 exp(−k0·q·z)——z < 0 时它是增长指数，对未激发的高阶倏逝波会溢出，
// 再乘 0 得 NaN（Python 的 field_at 在 M1 = 60 时从 |z| ≳ 1.07 µm 起就是 NaN：z = −1.5 µm 已经 NaN，−1.0 µm 仍有限）。反射项 exp(+k0·q·z)（z < 0）、区域 II 与层内的指数都是衰减的。
// 结果里任何一个复数非有限时抛 std::runtime_error（例如 |z| 大到 k0·z 溢出），不返回貌似有效的场。
//
// 失败与生命周期：数值失败（Status2D 不是 Ok、或保留的数据含非有限值、或元数据非有限）时上下文为 Unavailable，require_available() 抛 std::logic_error，
// field_at / field_slice 同样抛；参数非法（nt < 1、z 非有限、at 非有限、plane 不是 XZ / YZ）抛 std::invalid_argument。
// 上下文拥有全部数据，不持有 Stack 的指针：拷贝是深拷贝，移动后源对象为 Empty（元数据与数据一并重置）。
// 采样是 const 的、可重入的；矩阵乘法按 stablercwa::set_num_threads 的设定并行，结果与线程数无关（逐位相同）。
//
// 性能：同一层（或同一区域）的全部 z 一起做：先把 N 个谐波的指数 O(2N·nz) 地算出来，再每层各一次 W·(…)、V·(…)、Ei·(…) 的批量矩阵乘法
// （2N×2N × 2N×nz、2N×2N × 2N×nz、N×N × N×nz），不对每个 z 重新求逆或分解。均匀层（快路径）的 W = I、V 是分块对角，只剩 O(N·nz)。
// 为了限制临时内存，同一层的 z 按每 256 个一批处理（批与批之间互不相干，不改变数值）。切面重建每个分量一次 [nt×N]·[N×nz] 的矩阵乘法。
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rcwa2d.hpp"
#include "rcwa_fields.hpp"
#include "rcwa_phase.hpp"

namespace stablercwa {
namespace rcwa {

namespace detail2d { struct FieldAccess2D; }   // 非稳定的实现 / 测试访问点
struct FieldResult2D;

class FieldContext2D {
public:
    FieldContext2D() = default;
    // 拷贝是深拷贝；移动后源对象为 Empty，元数据与保留的数据也重置。没有 Stack 指针或共享可变缓存。
    FieldContext2D(const FieldContext2D&) = default;
    FieldContext2D& operator=(const FieldContext2D&) = default;
    FieldContext2D(FieldContext2D&& other) noexcept { *this = std::move(other); }
    FieldContext2D& operator=(FieldContext2D&& other) noexcept {
        if (this != &other) {
            metadata_ = std::move(other.metadata_);
            capture_ = std::move(other.capture_);
            status_ = std::exchange(other.status_, FieldContextStatus::Empty);
            other.metadata_ = FieldMetadata{};
            other.capture_ = detail2d::FieldCapture2D{};
        }
        return *this;
    }
    FieldContextStatus status() const noexcept { return status_; }
    bool available() const noexcept { return status_ == FieldContextStatus::Ready; }
    // 波矢、层几何、归一化等；Unavailable 时也给出输入决定的部分（kxm、kyn、kzI、kzII、层位置），层的模式标志为默认值。
    const FieldMetadata& metadata() const noexcept { return metadata_; }
    void require_available() const {
        if (!available()) throw std::logic_error("rcwa2d: the field context is unavailable");
    }
    // 上下文拥有的数据的字节数（近似：各数组与矩阵的元素个数 × 元素大小，不含 std::vector 的富余容量与对象本身）。
    std::size_t memory_bytes() const noexcept;

private:
    FieldMetadata metadata_;
    detail2d::FieldCapture2D capture_;
    FieldContextStatus status_ = FieldContextStatus::Empty;
    friend struct detail2d::FieldAccess2D;
};

struct FieldResult2D {
    Result2D result;
    FieldContext2D fields;
};

namespace detail2d {

constexpr int kFieldRegionI = -1, kFieldRegionII = -2;   // 区域标记；其余非负值是层下标
constexpr int kFieldChunk2D = 256;                       // 同一层里每批处理的 z 个数

// z 属于哪个区域（见文件头：外部优先，其次第一个满足 z ≤ z_top + d + 1e-12 的层；无层时 z = 0 取出射侧）。z 必须有限。
inline int field_region_2d(const FieldCapture2D& cap, double z) {
    if (z < 0.0) return kFieldRegionI;
    if (z > cap.z_bot) return kFieldRegionII;
    for (std::size_t j = 0; j < cap.layers.size(); ++j)
        if (z <= cap.layers[j].z_top + cap.layers[j].d + 1e-12) return static_cast<int>(j);
    return kFieldRegionII;    // 只剩 z == z_bot == 0 且没有层
}

inline std::size_t matrix_bytes(const Matrix& m) {
    return static_cast<std::size_t>(m.rows()) * static_cast<std::size_t>(m.cols()) * sizeof(cplx);
}

// field_slice 的纯参数检查（不依赖上下文，先于任何计算）。
inline void check_slice_arguments(SlicePlane plane, int nt, double at) {
    if (plane != SlicePlane::XZ && plane != SlicePlane::YZ) throw std::invalid_argument("rcwa2d: invalid slice plane");
    if (nt <= 0) throw std::invalid_argument("rcwa2d: slice nt must be positive");
    if (!std::isfinite(at)) throw std::invalid_argument("rcwa2d: slice coordinate must be finite");
}

struct FieldAccess2D {
    static FieldCapture2D& capture(FieldContext2D& c) { return c.capture_; }
    static const FieldCapture2D& capture(const FieldContext2D& c) { return c.capture_; }
    static FieldMetadata& metadata(FieldContext2D& c) { return c.metadata_; }
    static const FieldMetadata& metadata(const FieldContext2D& c) { return c.metadata_; }

    // 一次求解并装配上下文（solve2d_with_fields 的实现）。
    static void build(FieldResult2D& out, const Stack2D& stack, double wavelength, double theta_deg, double phi_deg, int M1, int M2,
                      const Jones2D& jones, const Options2D& options) {
        FieldContext2D& context = out.fields;
        out.result = solve2d(stack, wavelength, theta_deg, phi_deg, M1, M2, jones, options, &context.capture_);
        const Result2D& r = out.result;
        context.status_ = context.capture_.available ? FieldContextStatus::Ready : FieldContextStatus::Unavailable;
        FieldMetadata& meta = context.metadata_;
        meta = FieldMetadata{};
        meta.wavelength = wavelength;
        meta.k0 = r.k0;
        meta.theta_deg = theta_deg;
        meta.phi_deg = phi_deg;
        meta.incident_basis_columns = 1;            // 一次求解对应一个入射偏振态（Jones 在求解时给定）
        // 入射电场矢量范数 = Jones 二范数（s、p 单位矢量正交）；|Z0 H_inc| = nI·|E_inc|。Jones 不归一，也不检查非零。
        const double jones_norm = std::hypot(std::abs(jones[0]), std::abs(jones[1]));
        meta.normalization.electric_incident_amplitude = jones_norm;
        meta.normalization.magnetic_incident_amplitude = stack.nI * jones_norm;
        detail::copy_field_geometry_2d(meta, stack, false);       // 二维保留全部输入层，零厚层也留着（effective_index = 原下标）
        detail::fill_harmonic_orders_2d(meta, M1, M2);
        const std::size_t N = r.Kx.size();
        meta.kxm.resize(N);
        meta.kyn.resize(N);
        meta.kzI.resize(N);
        meta.kzII.resize(N);
        for (std::size_t j = 0; j < N; ++j) {
            meta.kxm[j] = cplx{meta.k0 * r.Kx[j], 0.0};
            meta.kyn[j] = meta.k0 * r.Ky[j];
            meta.kzI[j] = meta.k0 * r.kzI[j];
            meta.kzII[j] = meta.k0 * r.kzII[j];
        }
        if (N > 0) meta.ky = meta.kyn[static_cast<std::size_t>(r.i0)];     // 只作显示用：二维的重建与相位逐列用 kyn
        if (!detail::finite_field_metadata(meta)) {
            context.status_ = FieldContextStatus::Unavailable;
            context.capture_ = FieldCapture2D{};
        }
        if (context.status_ == FieldContextStatus::Ready)
            for (std::size_t j = 0; j < meta.layers.size(); ++j) {
                FieldLayerInfo& layer = meta.layers[j];
                const LayerCapture2D& lc = context.capture_.layers[j];
                layer.has_modes = true;
                layer.propagates = layer.thickness > 0.0;
                layer.analytic_modes = lc.homogeneous;
                layer.mode_count = 2 * static_cast<int>(N);
                layer.finite_zero_mode_count = 0;
                layer.propagation = FieldPropagation::Exponential;
            }
    }
};

// 把一批 z（列 cols[begin, begin+count)）的 F = [Ex;Ey]（2N×count）、G = [hx;hy]（2N×count）换成六个分量写进 out。
// Ei != nullptr：Ez = i·Ei·(Ky·hx − Kx·hy)，Ei 是 N×N 稠密阵（一次批量乘法）；否则 Ei 是标量 inv_eps。
inline void emit_field_chunk(HarmonicFields& out, const RealVector& Kx, const RealVector& Ky, const std::vector<int>& cols,
                             std::size_t begin, int count, const Matrix& F, const Matrix& G, const Matrix* Ei, cplx inv_eps) {
    const int N = static_cast<int>(Kx.size());
    Matrix T(N, count);        // Ky·hx − Kx·hy
    for (int col = 0; col < count; ++col) {
        const int iz = cols[begin + static_cast<std::size_t>(col)];
        const cplx* f = F.col(col);
        const cplx* g = G.col(col);
        cplx* t = T.col(col);
        for (int j = 0; j < N; ++j) {
            const cplx Ex = f[j], Ey = f[N + j], hx = g[j], hy = g[N + j];
            const cplx hz = kI * (Ky[static_cast<std::size_t>(j)] * Ex - Kx[static_cast<std::size_t>(j)] * Ey);
            out.components[0](j, iz) = Ex;
            out.components[1](j, iz) = Ey;
            out.components[3](j, iz) = -kI * hx;
            out.components[4](j, iz) = -kI * hy;
            out.components[5](j, iz) = -kI * hz;
            t[j] = Ky[static_cast<std::size_t>(j)] * hx - Kx[static_cast<std::size_t>(j)] * hy;
        }
    }
    if (Ei) {
        const Matrix E = (*Ei) * T;
        for (int col = 0; col < count; ++col) {
            const int iz = cols[begin + static_cast<std::size_t>(col)];
            for (int j = 0; j < N; ++j) out.components[2](j, iz) = kI * E(j, col);
        }
    } else {
        for (int col = 0; col < count; ++col) {
            const int iz = cols[begin + static_cast<std::size_t>(col)];
            const cplx* t = T.col(col);
            for (int j = 0; j < N; ++j) out.components[2](j, iz) = kI * (inv_eps * t[j]);
        }
    }
}

// 层内一批 z：p = a⁺·exp(−k0·q·zl)、m = a⁻·exp(−k0·q·(d − zl))（零振幅不求指数），F = W·(p + m)、G = V·(−p + m)。
inline void layer_chunk_fields(const FieldCapture2D& cap, const LayerCapture2D& lc, const RealVector& zs, const std::vector<int>& cols,
                               std::size_t begin, int count, Matrix& F, Matrix& G) {
    const int n2 = 2 * cap.N;
    Matrix S(n2, count), D(n2, count);
    for (int col = 0; col < count; ++col) {
        const double z = zs[static_cast<std::size_t>(cols[begin + static_cast<std::size_t>(col)])];
        const double from_top = (std::max)(0.0, (std::min)(lc.d, z - lc.z_top)), from_bottom = lc.d - from_top;
        cplx* s = S.col(col);
        cplx* dd = D.col(col);
        for (int j = 0; j < n2; ++j) {
            const cplx a = lc.a_plus[static_cast<std::size_t>(j)], b = lc.a_minus[static_cast<std::size_t>(j)], q = lc.q[static_cast<std::size_t>(j)];
            const cplx p = a == cplx{} ? cplx{} : a * std::exp(-cap.k0 * from_top * q);
            const cplx m = b == cplx{} ? cplx{} : b * std::exp(-cap.k0 * from_bottom * q);
            s[j] = p + m;
            dd[j] = m - p;
        }
    }
    if (lc.homogeneous) {          // 快路径：W = I，V 是 2×2 分块对角
        F = std::move(S);
        G = b2_apply_rows(lc.Vbd, D);
    } else {
        F = lc.W * S;
        G = lc.V * D;
    }
}

// 区域 I 一批 z（都 < 0）。入射项只对零阶 (0,0) 的 Ex、Ey 求增长指数 exp(−k0·q·z)；反射项 exp(+k0·q·z) 衰减。
inline void region_I_chunk_fields(const FieldCapture2D& cap, const RealVector& zs, const std::vector<int>& cols, std::size_t begin,
                                  int count, Matrix& F, Matrix& G) {
    const int N = cap.N, n2 = 2 * N, i0 = cap.M1 * (2 * cap.M2 + 1) + cap.M2;
    const std::size_t i0x = static_cast<std::size_t>(i0), i0y = static_cast<std::size_t>(N + i0);
    F = Matrix(n2, count);
    G = Matrix(n2, count);
    Vector v(static_cast<std::size_t>(n2));
    for (int col = 0; col < count; ++col) {
        const double z = zs[static_cast<std::size_t>(cols[begin + static_cast<std::size_t>(col)])];
        cplx* f = F.col(col);
        cplx* g = G.col(col);
        for (int j = 0; j < n2; ++j) {
            const cplx R = cap.R[static_cast<std::size_t>(j)];
            f[j] = R == cplx{} ? cplx{} : R * std::exp(cap.k0 * z * cap.region_I.q[static_cast<std::size_t>(j)]);
            v[static_cast<std::size_t>(j)] = f[j];
        }
        const cplx inc_x = cap.inc[i0x] == cplx{} ? cplx{} : cap.inc[i0x] * std::exp(-cap.k0 * z * cap.region_I.q[i0x]);
        const cplx inc_y = cap.inc[i0y] == cplx{} ? cplx{} : cap.inc[i0y] * std::exp(-cap.k0 * z * cap.region_I.q[i0y]);
        f[i0] += inc_x;
        f[N + i0] += inc_y;
        v[i0x] -= inc_x;
        v[i0y] -= inc_y;
        const BlockDiag2& V = cap.region_I.V;
        for (int j = 0; j < N; ++j) {
            const std::size_t k = static_cast<std::size_t>(j), l = static_cast<std::size_t>(N + j);
            g[j] = V.a11[k] * v[k] + V.a12[k] * v[l];
            g[N + j] = V.a21[k] * v[k] + V.a22[k] * v[l];
        }
    }
}

// 区域 II 一批 z（都 > z_bot）：F = T·exp(−k0·q·(z − z_bot))，G = V·(−F)。
inline void region_II_chunk_fields(const FieldCapture2D& cap, const RealVector& zs, const std::vector<int>& cols, std::size_t begin,
                                   int count, Matrix& F, Matrix& G) {
    const int N = cap.N, n2 = 2 * N;
    F = Matrix(n2, count);
    G = Matrix(n2, count);
    Vector v(static_cast<std::size_t>(n2));
    for (int col = 0; col < count; ++col) {
        const double z = zs[static_cast<std::size_t>(cols[begin + static_cast<std::size_t>(col)])];
        cplx* f = F.col(col);
        cplx* g = G.col(col);
        for (int j = 0; j < n2; ++j) {
            const cplx T = cap.T[static_cast<std::size_t>(j)];
            f[j] = T == cplx{} ? cplx{} : T * std::exp(-cap.k0 * (z - cap.z_bot) * cap.region_II.q[static_cast<std::size_t>(j)]);
            v[static_cast<std::size_t>(j)] = -f[j];
        }
        const BlockDiag2& V = cap.region_II.V;
        for (int j = 0; j < N; ++j) {
            const std::size_t k = static_cast<std::size_t>(j), l = static_cast<std::size_t>(N + j);
            g[j] = V.a11[k] * v[k] + V.a12[k] * v[l];
            g[N + j] = V.a21[k] * v[k] + V.a22[k] * v[l];
        }
    }
}

// 任意一组 z 处的六分量谐波列（field_at 的实现）。
inline HarmonicFields sample_field_2d(const FieldContext2D& context, const RealVector& zs) {
    context.require_available();
    const FieldCapture2D& cap = FieldAccess2D::capture(context);
    const FieldMetadata& meta = FieldAccess2D::metadata(context);
    detail::checked_field_size(zs.size());
    for (double z : zs)
        if (!std::isfinite(z)) throw std::invalid_argument("rcwa2d: field depths must be finite");
    HarmonicFields out = detail::make_harmonic_fields(meta, zs);

    // 分组：0 = 区域 I，1 = 区域 II，2 + j = 第 j 层；组内保留输入的相对次序。
    std::vector<std::vector<int>> groups(2 + cap.layers.size());
    for (int iz = 0; iz < static_cast<int>(zs.size()); ++iz) {
        const int region = field_region_2d(cap, zs[static_cast<std::size_t>(iz)]);
        const std::size_t slot = region == kFieldRegionI ? 0 : region == kFieldRegionII ? 1 : 2 + static_cast<std::size_t>(region);
        groups[slot].push_back(iz);
    }
    for (std::size_t slot = 0; slot < groups.size(); ++slot) {
        const std::vector<int>& cols = groups[slot];
        for (std::size_t begin = 0; begin < cols.size(); begin += static_cast<std::size_t>(kFieldChunk2D)) {
            const int count = static_cast<int>((std::min)(cols.size() - begin, static_cast<std::size_t>(kFieldChunk2D)));
            Matrix F, G;
            const Matrix* Ei = nullptr;
            cplx inv_eps{};
            if (slot == 0) {
                region_I_chunk_fields(cap, zs, cols, begin, count, F, G);
                inv_eps = cplx{1.0 / (cap.nI * cap.nI), 0.0};
            } else if (slot == 1) {
                region_II_chunk_fields(cap, zs, cols, begin, count, F, G);
                inv_eps = cplx{1.0, 0.0} / (cap.nII * cap.nII);
            } else {
                const LayerCapture2D& lc = cap.layers[slot - 2];
                layer_chunk_fields(cap, lc, zs, cols, begin, count, F, G);
                if (lc.homogeneous) inv_eps = cplx{1.0, 0.0} / lc.eps;
                else Ei = &lc.Ei;
            }
            emit_field_chunk(out, cap.Kx, cap.Ky, cols, begin, count, F, G, Ei, inv_eps);
        }
    }
    for (const Matrix& component : out.components)
        if (!component.all_finite()) throw std::runtime_error("rcwa2d: field propagation produced nonfinite values");
    return out;
}

}  // namespace detail2d

inline std::size_t FieldContext2D::memory_bytes() const noexcept {
    namespace d2 = detail2d;
    std::size_t bytes = 0;
    const auto vec = [&](const Vector& v) { bytes += v.size() * sizeof(cplx); };
    const auto bd = [&](const d2::BlockDiag2& b) { for (const Vector* v : {&b.a11, &b.a12, &b.a21, &b.a22}) vec(*v); };
    vec(capture_.region_I.q); bd(capture_.region_I.V); vec(capture_.region_I.kz);
    vec(capture_.region_II.q); bd(capture_.region_II.V); vec(capture_.region_II.kz);
    vec(capture_.inc); vec(capture_.R); vec(capture_.T);
    bytes += (capture_.Kx.size() + capture_.Ky.size()) * sizeof(double);
    for (const d2::LayerCapture2D& lc : capture_.layers) {
        vec(lc.q); vec(lc.a_plus); vec(lc.a_minus); bd(lc.Vbd);
        bytes += d2::matrix_bytes(lc.W) + d2::matrix_bytes(lc.V) + d2::matrix_bytes(lc.Ei);
    }
    bytes += (metadata_.kxm.size() + metadata_.kzI.size() + metadata_.kzII.size()) * sizeof(cplx) + metadata_.kyn.size() * sizeof(double) +
             (metadata_.orders.size() + metadata_.orders_m.size() + metadata_.orders_n.size()) * sizeof(int);
    return bytes;
}

// 一次求解同时得到 Result2D 和场上下文（capture 与 solve2d 同一次本征分解；Result2D 与 solve2d(…, capture = nullptr) 逐位相同，timing 除外）。
// 参数非法抛 std::invalid_argument（同 solve2d）。数值失败（NumericalFailure、RayleighAnomaly）沿用 Result2D 的 NaN 语义，上下文为 Unavailable。
// jones 不归一（与 solve2d 一致，零 Jones 给出零场与 normalization = {0, 0}，不当作失败）。
inline FieldResult2D solve2d_with_fields(const Stack2D& stack, double wavelength, double theta_deg, double phi_deg, int M1, int M2,
                                         const Jones2D& jones, const Options2D& options = {}) {
    FieldResult2D out;
    detail2d::FieldAccess2D::build(out, stack, wavelength, theta_deg, phi_deg, M1, M2, jones, options);
    return out;
}

// 任意 z 处的六分量谐波列：六个 Matrix [N, nz]（行 = 谐波列下标 harmonic_index_2d，列 = 输入的第几个 z）；
// metadata.orders 是列下标，orders_m / orders_n 给 (m, n)。z 保留输入顺序与重复值；空 zs 给出 [N, 0] 矩阵。
// 上下文不可用抛 std::logic_error；z 非有限抛 std::invalid_argument；结果含非有限数抛 std::runtime_error。
inline HarmonicFields field_at(const FieldContext2D& context, const RealVector& zs) { return detail2d::sample_field_2d(context, zs); }
inline HarmonicFields field_at(const FieldContext2D& context, double z) { return field_at(context, RealVector{z}); }
inline HarmonicFields field_at(const FieldResult2D& result, const RealVector& zs) { return field_at(result.fields, zs); }
inline HarmonicFields field_at(const FieldResult2D& result, double z) { return field_at(result.fields, z); }

// x-z（plane = XZ：固定 y = at，沿 x，周期 Lx）或 y-z（plane = YZ：固定 x = at，沿 y，周期 Ly）切面上的六复场：
// t_j = j·L/nt（不重复周期终点），每个分量 Matrix [nt, nz]，at(c, iz, it)；谐波列经 detail::reconstruct_field_slice 每个分量一次矩阵乘法重建。
// 原始 E/H 不缩放（H 是 Z0·H），入射范数在 metadata.normalization。nt < 1、at 非有限、plane 不是 XZ / YZ 抛 std::invalid_argument（先于任何计算）；
// 上下文不可用抛 std::logic_error。
inline FieldGridSlice field_slice(const FieldContext2D& context, const RealVector& zs, SlicePlane plane, int nt = 256, double at = 0.0) {
    detail2d::check_slice_arguments(plane, nt, at);
    return detail::reconstruct_field_slice(field_at(context, zs), plane, nt, at);
}
inline FieldGridSlice field_slice(const FieldResult2D& result, const RealVector& zs, SlicePlane plane, int nt = 256, double at = 0.0) {
    return field_slice(result.fields, zs, plane, nt, at);
}
// 已有谐波列（field_at 的结果）重建切面：同一组 z 要做 x-z 和 y-z 两个面（或同一面的几个 at）时只采样一次。
// 采样（批量矩阵乘法）是大头——N = 625、200 个 z 全部落在结构层里时采样约占 80% 以上（实测 82%–88%；z 分散在区域 I、层内、区域 II 时约 58%，全在区域 I 时只有 6%–8%）——重建每个分量只是一次 [nt×N]·[N×nz] 乘法。
// 与 field_slice(context, zs, …) 逐位相同。harmonics 必须来自本头文件的 field_at（二维元数据）；参数检查同上。
inline FieldGridSlice field_slice(const HarmonicFields& harmonics, SlicePlane plane, int nt = 256, double at = 0.0) {
    detail2d::check_slice_arguments(plane, nt, at);
    return detail::reconstruct_field_slice(harmonics, plane, nt, at);
}

// 便捷入口：切面采样后直接交给 rcwa_phase.hpp 的 phase_data（规则完全相同：掩膜、absolute / incident / custom 参考、展开）。
// 要同时用场和相位、或对同一切面算多个分量，先调 field_slice 再多次调 phase_data，不要重复采样；
// 同一组 z 的两个面（x-z 与 y-z）先 field_at 一次，再对两个 plane 各调 field_slice(harmonics, plane, nt, at)。
inline PhaseGrid phase_slice(const FieldContext2D& context, const RealVector& zs, SlicePlane plane, int nt, double at,
                             FieldComponent component, const PhaseOptions& options = {}) {
    return phase_data(field_slice(context, zs, plane, nt, at), component, options);
}
inline PhaseGrid phase_slice(const FieldContext2D& context, const RealVector& zs, SlicePlane plane, int nt, double at,
                             const std::string& component, const PhaseOptions& options = {}) {
    return phase_data(field_slice(context, zs, plane, nt, at), component, options);
}
inline PhaseGrid phase_slice(const FieldResult2D& result, const RealVector& zs, SlicePlane plane, int nt, double at,
                             FieldComponent component, const PhaseOptions& options = {}) {
    return phase_slice(result.fields, zs, plane, nt, at, component, options);
}
inline PhaseGrid phase_slice(const FieldResult2D& result, const RealVector& zs, SlicePlane plane, int nt, double at,
                             const std::string& component, const PhaseOptions& options = {}) {
    return phase_slice(result.fields, zs, plane, nt, at, component, options);
}

}  // namespace rcwa
}  // namespace stablercwa
