// stablercwa/rcwa_fields.hpp —— 六复场、采样网格与入射归一化的公共约定
//
// 二维（交叉光栅）泛化只做增量：FieldGridXZ、HarmonicFields、FieldMetadata 原有字段的含义与数组序全部保持；
// 新增的切面网格 FieldGridSlice（x-z：固定 y；y-z：固定 x）、二维元数据字段和 Periodic2D 标记放在旁边，
// 二维专用字段在一维结果里保持默认值。设计与数组序见各结构体前的注释；合同见
// gui/validation/field_reference/CONTRACT.md（一维）和 gui/validation/field_reference_2d/CONTRACT_2D.md（二维）。
#pragma once

#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "matrix.hpp"

namespace stablercwa {
namespace rcwa {

enum class FieldComponent { Ex, Ey, Ez, Hx, Hy, Hz };  // 复分量次序；H 存 Z0 H_physical
enum class FieldContextStatus { Empty, Ready, Unavailable };
enum class FieldSolveKind { PlanarTE, PlanarTM, Conical, Periodic2D };  // Periodic2D：二维交叉光栅，谐波是二维指标 (m, n)
enum class FieldPropagation { None, Exponential, FiniteZero, Schur };
// 切面：XZ = 固定 y 的 x-z 面（沿 x 变化，周期 Lx）；YZ = 固定 x 的 y-z 面（沿 y 变化，周期 Ly）。
enum class SlicePlane { XZ, YZ };
using JonesVector = std::array<cplx, 2>;  // [Es, Ep]；电场范数为 1，保留整体复相位

inline std::size_t field_component_index(FieldComponent component) {
    const auto index = static_cast<std::size_t>(component);
    if (index >= 6) throw std::invalid_argument("rcwa: invalid field component");
    return index;
}
inline const char* field_component_name(FieldComponent component) {
    static constexpr const char* names[] = {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"};
    return names[field_component_index(component)];
}

// 切面名字与 Python NPZ 的 slice_plane 字符串一致："xz" / "yz"。
inline const char* slice_plane_name(SlicePlane plane) {
    if (plane == SlicePlane::XZ) return "xz";
    if (plane == SlicePlane::YZ) return "yz";
    throw std::invalid_argument("rcwa: invalid slice plane");
}
inline SlicePlane parse_slice_plane(const std::string& name) {
    std::string label;
    for (char c : name) if (c != ' ' && c != '\t' && c != '\r' && c != '\n') label.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    if (label == "xz") return SlicePlane::XZ;
    if (label == "yz") return SlicePlane::YZ;
    throw std::invalid_argument("rcwa: slice plane must be \"xz\" or \"yz\"");
}

struct FieldNormalization {
    double electric_incident_amplitude = 1.0;
    double magnetic_incident_amplitude = 1.0;
    // H 始终是 Z0 H_physical；上面两项是完整入射矢量的范数，不自动缩放缓存振幅。
};

struct FieldLayerInfo {
    std::size_t original_index = 0;
    int effective_index = -1;           // 圆锥中被过滤的零厚层为 -1；平面沿用全部输入层
    double z_top = 0.0, z_bottom = 0.0, thickness = 0.0;
    bool homogeneous = true;           // 输入几何的写法；相同折射率的分段层也可能有解析模式
    cplx uniform_index = 1.0;
    RealVector edges;
    Vector n_values;                    // 几何/材料快照，独立于调用者 Stack
    bool has_modes = false;
    bool propagates = false;
    bool analytic_modes = false;
    int mode_count = 0;
    int finite_zero_mode_count = 0;
    FieldPropagation propagation = FieldPropagation::None;
};

struct FieldMetadata {
    FieldSolveKind kind = FieldSolveKind::PlanarTE;
    double wavelength = 0.0, period = 0.0, k0 = 0.0;
    double theta_deg = 0.0, phi_deg = 0.0, z_bottom = 0.0;
    cplx nI = 1.0, nII = 1.0;
    int truncation_order = 0;
    int incident_basis_columns = 0;      // 平面 1；圆锥 2，列次序为单位完整电场的 s、p 入射
    std::vector<int> orders;
    Vector kxm, kzI, kzII;              // 物理波数（不是 k0 归一值）；横向因子 exp[-i(kxm*x+ky*y)]
    double ky = 0.0;
    FieldNormalization normalization;
    std::vector<FieldLayerInfo> layers;  // 原始输入层次序，包括被过滤的层

    // —— 以下字段只在二维（kind == Periodic2D）里填写；一维保持默认值，上面各字段的含义不变 ——
    // 二维时上面几个字段这样用：
    //   period = Lx（x 周期），truncation_order = M1；
    //   orders 仍是“每个谐波一项”：取该谐波的列下标 j = (m+M1)(2M2+1)+(n+M2)，即 rcwa2d_types.hpp 的 harmonic_index_2d
    //     （m 在外层、n 在内层）；真正的 (m, n) 在 orders_m / orders_n，零级 (0,0) 靠它们找，不是 orders 里的 0；
    //   kxm[j] = k0·Kx(m) 是该列的物理 kx（nI 为实数，虚部为 0），kyn[j] = k0·Ky(n) 是该列的物理 ky（实数）；
    //   kzI / kzII 逐列，与一维同义；
    //   ky 取入射零级的 ky（= kyn[零级的列]），只作显示用——二维的重建与相位一律用逐列的 kyn，不用 ky。
    // 谐波的横向因子 exp[−i(kxm[j]·x + kyn[j]·y)]，与一维的 exp[−i(kxm·x + ky·y)] 一致。
    double period_y = 0.0;               // y 周期 Ly
    int truncation_order_y = 0;          // M2（x 方向的 M1 在 truncation_order）
    std::vector<int> orders_m, orders_n; // 长 N：第 j 列的 (m, n)
    RealVector kyn;                      // 长 N：第 j 列的物理 ky
};

// 每个 Matrix 为 [order, depth]（按列）；六分量次序为 Ex,Ey,Ez,Hx,Hy,Hz。
// 单点 field_at 的 z 长度为 1；空深度输入给出 [orders,0] 矩阵。
struct HarmonicFields {
    std::vector<int> orders;
    RealVector z;
    FieldMetadata metadata;
    std::array<Matrix, 6> components;
    Matrix& component(FieldComponent c) { return components[field_component_index(c)]; }
    const Matrix& component(FieldComponent c) const { return components[field_component_index(c)]; }
    std::size_t nz() const noexcept { return z.size(); }
    std::size_t norders() const noexcept { return orders.size(); }
    cplx& at(FieldComponent c, std::size_t iz, std::size_t order_index) {
        if (iz >= nz() || order_index >= norders()) throw std::out_of_range("rcwa: harmonic field index");
        return component(c)(static_cast<int>(order_index), static_cast<int>(iz));
    }
    const cplx& at(FieldComponent c, std::size_t iz, std::size_t order_index) const {
        if (iz >= nz() || order_index >= norders()) throw std::out_of_range("rcwa: harmonic field index");
        return component(c)(static_cast<int>(order_index), static_cast<int>(iz));
    }
};

// 每个 Matrix 为 [x, z]（按列），data()[iz*nx+ix] 等价 Python [nz,nx] C 顺序。
// x_j=j*period/nx，不重复周期终点；z 保留输入顺序和重复值，y 是固定标量。
// 原始 E/H 不缩放；metadata.normalization 给出 GUI/强度显示所需的入射范数。
struct FieldGridXZ {
    RealVector x, z;
    double y = 0.0;
    FieldMetadata metadata;
    std::array<Matrix, 6> components;
    Matrix& component(FieldComponent c) { return components[field_component_index(c)]; }
    const Matrix& component(FieldComponent c) const { return components[field_component_index(c)]; }
    std::size_t nx() const noexcept { return x.size(); }
    std::size_t nz() const noexcept { return z.size(); }
    cplx& at(FieldComponent c, std::size_t iz, std::size_t ix) {
        if (iz >= nz() || ix >= nx()) throw std::out_of_range("rcwa: field grid index");
        return component(c)(static_cast<int>(ix), static_cast<int>(iz));
    }
    const cplx& at(FieldComponent c, std::size_t iz, std::size_t ix) const {
        if (iz >= nz() || ix >= nx()) throw std::out_of_range("rcwa: field grid index");
        return component(c)(static_cast<int>(ix), static_cast<int>(iz));
    }
};

// 沿一条竖直切面采样的六复场（二维求解器的 x-z / y-z 剖面；也能装一维的 x-z 网格，见 to_field_slice）。
// 与 FieldGridXZ 的布局一致，只是把水平轴写成“沿切面的变化坐标 t”：
//   plane = XZ：固定 y = slice_at，t 沿 x，周期 Lx = metadata.period；
//   plane = YZ：固定 x = slice_at，t 沿 y，周期 Ly = metadata.period_y。
//   t_j = j·L/nt（j = 0…nt-1，不重复周期终点），z 保留输入顺序和重复值。
//   x[j]、y[j] 是第 j 个采样点的实际物理坐标（XZ：x = t、y = slice_at；YZ：x = slice_at、y = t），长度都是 nt；
//   相位的 carrier 逐点取这两个值，所以不必关心切面是哪一种。
// 每个分量 Matrix 为 [nt, nz]（按列），data()[iz*nt + it] 等价 Python 二维 NPZ 的 (nz, npts) C 顺序；
// 公开访问器 at(component, iz, it)：iz 在前（深度），it 在后（沿切面），与 FieldGridXZ::at(c, iz, ix) 同序。
// 原始 E/H 不缩放（H 是 Z0 H_physical）；入射范数在 metadata.normalization。
struct FieldGridSlice {
    SlicePlane plane = SlicePlane::XZ;
    RealVector t, z;
    RealVector x, y;
    double slice_at = 0.0;
    FieldMetadata metadata;
    std::array<Matrix, 6> components;
    Matrix& component(FieldComponent c) { return components[field_component_index(c)]; }
    const Matrix& component(FieldComponent c) const { return components[field_component_index(c)]; }
    std::size_t nt() const noexcept { return t.size(); }
    std::size_t nz() const noexcept { return z.size(); }
    cplx& at(FieldComponent c, std::size_t iz, std::size_t it) {
        if (iz >= nz() || it >= nt()) throw std::out_of_range("rcwa: field slice index");
        return component(c)(static_cast<int>(it), static_cast<int>(iz));
    }
    const cplx& at(FieldComponent c, std::size_t iz, std::size_t it) const {
        if (iz >= nz() || it >= nt()) throw std::out_of_range("rcwa: field slice index");
        return component(c)(static_cast<int>(it), static_cast<int>(iz));
    }
};

// 切面方向的周期：XZ 取 metadata.period（Lx），YZ 取 metadata.period_y（Ly）；必须有限且为正。
inline double slice_period(const FieldMetadata& metadata, SlicePlane plane) {
    if (plane != SlicePlane::XZ && plane != SlicePlane::YZ) throw std::invalid_argument("rcwa: invalid slice plane");
    const double period = plane == SlicePlane::XZ ? metadata.period : metadata.period_y;
    if (!std::isfinite(period) || !(period > 0.0))
        throw std::invalid_argument(plane == SlicePlane::XZ ? "rcwa: x-z slice needs a positive finite x period"
                                                            : "rcwa: y-z slice needs a positive finite y period");
    return period;
}

// 切面采样点坐标。t_j = (j/nt)·period（与 FieldGridXZ 的 x 同一个表达式，XZ 面逐位相同），
// 另一个横向坐标全取 at。nt ≤ 0、period 不是有限正数、at 非有限时抛 invalid_argument。
struct SliceCoordinates { RealVector t, x, y; };
inline SliceCoordinates slice_coordinates(SlicePlane plane, int nt, double period, double at) {
    if (plane != SlicePlane::XZ && plane != SlicePlane::YZ) throw std::invalid_argument("rcwa: invalid slice plane");
    if (nt <= 0) throw std::invalid_argument("rcwa: slice nt must be positive");
    if (!std::isfinite(at)) throw std::invalid_argument("rcwa: slice coordinate must be finite");
    if (!std::isfinite(period) || !(period > 0.0)) throw std::invalid_argument("rcwa: slice period must be positive and finite");
    SliceCoordinates out;
    out.t.resize(static_cast<std::size_t>(nt));
    for (int it = 0; it < nt; ++it) out.t[it] = (static_cast<double>(it) / nt) * period;
    if (plane == SlicePlane::XZ) { out.x = out.t; out.y.assign(out.t.size(), at); }
    else { out.x.assign(out.t.size(), at); out.y = out.t; }
    return out;
}

// 一维的 x-z 网格换成切面网格（XZ，slice_at = y）；坐标与全部分量原样复制，所以对它做相位与对原网格做相位逐位相同。
inline FieldGridSlice to_field_slice(const FieldGridXZ& grid) {
    FieldGridSlice out;
    out.plane = SlicePlane::XZ;
    out.t = grid.x; out.x = grid.x; out.z = grid.z;
    out.y.assign(grid.x.size(), grid.y);
    out.slice_at = grid.y;
    out.metadata = grid.metadata;
    out.components = grid.components;
    return out;
}

inline JonesVector normalized_jones(JonesVector jones) {
    double scale = 0.0;
    for (cplx v : jones) {
        if (!std::isfinite(v.real()) || !std::isfinite(v.imag()))
            throw std::invalid_argument("rcwa: Jones input must be finite and nonzero");
        scale = (std::max)(scale, (std::max)(std::abs(v.real()), std::abs(v.imag())));
    }
    if (scale == 0.0) throw std::invalid_argument("rcwa: Jones input must be finite and nonzero");
    for (auto& v : jones) v /= scale;
    const double norm = std::sqrt(std::norm(jones[0]) + std::norm(jones[1]));
    for (auto& v : jones) v /= norm;
    return jones;
}

inline JonesVector jones_from_angles(double psi_deg = 90.0, double delta_deg = 0.0) {
    if (!std::isfinite(psi_deg) || !std::isfinite(delta_deg))
        throw std::invalid_argument("rcwa: Jones angles must be finite");
    const double psi = psi_deg * kPi / 180.0, delta = delta_deg * kPi / 180.0;
    return normalized_jones({std::sin(psi) * std::exp(kI * delta), std::cos(psi)});
}

namespace detail {
inline int checked_field_size(std::size_t n) {
    if (n > static_cast<std::size_t>((std::numeric_limits<int>::max)()))
        throw std::invalid_argument("rcwa: field grid exceeds Matrix dimensions");
    return static_cast<int>(n);
}

struct FieldDepthGroup {
    int region = -1;  // -1: 入射侧，-2: 出射侧，其余为有效层下标
    const FieldLayerInfo* layer = nullptr;
    std::vector<int> columns;
};

// 外部严格判断优先；有限层界面属于上层（1e-12 容差）。无正厚有限层时 z=0 定义为出射侧。
inline std::vector<FieldDepthGroup> field_depth_groups(const FieldMetadata& meta, const RealVector& zs) {
    checked_field_size(zs.size());
    std::vector<FieldDepthGroup> groups(2 + meta.layers.size());
    groups[0].region = -1; groups[1].region = -2;
    bool has_finite_layer = false;
    for (std::size_t j = 0; j < meta.layers.size(); ++j) {
        const auto& layer = meta.layers[j];
        groups[2 + j].region = layer.effective_index;
        groups[2 + j].layer = &layer;
        has_finite_layer = has_finite_layer || (layer.has_modes && layer.thickness > 0.0);
    }
    for (int iz = 0; iz < static_cast<int>(zs.size()); ++iz) {
        const double z = zs[iz];
        if (!std::isfinite(z)) throw std::invalid_argument("rcwa: field depths must be finite");
        std::size_t group = z < 0.0 ? 0 : 1;
        if (z >= 0.0 && z <= meta.z_bottom && has_finite_layer) {
            bool found = false;
            for (std::size_t j = 0; j < meta.layers.size(); ++j) {
                const auto& layer = meta.layers[j];
                if (layer.has_modes && z <= layer.z_bottom + 1e-12) { group = 2 + j; found = true; break; }
            }
            if (!found) throw std::logic_error("rcwa: no retained layer at field depth");
        }
        groups[group].columns.push_back(iz);
    }
    return groups;
}

inline HarmonicFields make_harmonic_fields(const FieldMetadata& meta, const RealVector& zs) {
    HarmonicFields out;
    out.orders = meta.orders; out.z = zs; out.metadata = meta;
    const int n = checked_field_size(meta.orders.size()), nz = checked_field_size(zs.size());
    for (auto& component : out.components) component = Matrix(n, nz);
    return out;
}

inline void validate_field_grid(int nx, double y) {
    if (nx <= 0) throw std::invalid_argument("rcwa: field nx must be positive");
    if (!std::isfinite(y)) throw std::invalid_argument("rcwa: field y must be finite");
}

inline bool is_periodic_2d(const FieldMetadata& meta) noexcept { return meta.kind == FieldSolveKind::Periodic2D; }

// 二维谐波表：按 harmonic_index_2d 的次序（m 外层、n 内层）填 orders / orders_m / orders_n / truncation_order(_y)。
// 不动波矢（kxm、kyn、kzI、kzII 由求解器按 Kx、Ky 填）。M1、M2 为负或谐波总数超出 Matrix 维数时抛 invalid_argument。
inline void fill_harmonic_orders_2d(FieldMetadata& meta, int M1, int M2) {
    if (M1 < 0 || M2 < 0) throw std::invalid_argument("rcwa: 2D truncation orders must be nonnegative");
    const long long total = (2LL * M1 + 1) * (2LL * M2 + 1);
    if (total > static_cast<long long>((std::numeric_limits<int>::max)()))
        throw std::invalid_argument("rcwa: 2D harmonic count exceeds Matrix dimensions");
    meta.truncation_order = M1; meta.truncation_order_y = M2;
    meta.orders.clear(); meta.orders_m.clear(); meta.orders_n.clear();
    meta.orders.reserve(static_cast<std::size_t>(total));
    for (int m = -M1; m <= M1; ++m)
        for (int n = -M2; n <= M2; ++n) {
            meta.orders.push_back(static_cast<int>(meta.orders.size()));
            meta.orders_m.push_back(m); meta.orders_n.push_back(n);
        }
}

// 二维元数据的数组长度必须一致（每个谐波一项）；一维元数据不检查（沿用原行为）。
inline void check_field_metadata_2d(const FieldMetadata& meta) {
    const std::size_t n = meta.orders.size();
    if (meta.orders_m.size() != n || meta.orders_n.size() != n || meta.kxm.size() != n || meta.kyn.size() != n)
        throw std::invalid_argument("rcwa: 2D field metadata arrays must have one entry per harmonic");
}

// 第 j 个谐波的物理 ky：一维是常数 ky，二维是逐列的 kyn[j]（调用前二维元数据须已过 check_field_metadata_2d）。
inline double harmonic_ky(const FieldMetadata& meta, std::size_t j) {
    return is_periodic_2d(meta) ? meta.kyn[j] : meta.ky;
}

// 横向 carrier 只构建一次，所有分量/深度用矩阵乘法重建，不逐像素调用采样器。
inline FieldGridXZ reconstruct_field_grid(HarmonicFields harmonics, int nx, double y) {
    validate_field_grid(nx, y);
    if (is_periodic_2d(harmonics.metadata)) check_field_metadata_2d(harmonics.metadata);
    FieldGridXZ out;
    out.x.resize(nx); out.z = std::move(harmonics.z); out.y = y;
    out.metadata = std::move(harmonics.metadata);
    const int n = checked_field_size(harmonics.orders.size());
    for (int ix = 0; ix < nx; ++ix) out.x[ix] = (static_cast<double>(ix) / nx) * out.metadata.period;
    if (out.z.empty()) {
        for (auto& component : out.components) component = Matrix(nx, 0);
        return out;
    }
    Matrix carrier(nx, n);
    for (int ix = 0; ix < nx; ++ix) {
        for (int order = 0; order < n; ++order)
            carrier(ix, order) = std::exp(-kI * (out.metadata.kxm[order] * out.x[ix] + harmonic_ky(out.metadata, order) * y));
    }
    if (!carrier.all_finite()) throw std::runtime_error("rcwa: field carrier produced nonfinite values");
    for (std::size_t c = 0; c < 6; ++c) {
        out.components[c] = carrier * harmonics.components[c];
        if (!out.components[c].all_finite()) throw std::runtime_error("rcwa: field reconstruction produced nonfinite values");
    }
    return out;
}

// 同一组谐波在切面上重建：与 reconstruct_field_grid 相同的做法（carrier 一次，六分量各一次矩阵乘法），
// 只是 carrier 的 (x, y) 逐点取切面坐标，ky 逐列取（二维）或常数（一维）。
//   XZ 面：t = x，周期 metadata.period，y = at；一维谐波在 XZ 面的结果与 reconstruct_field_grid 逐位相同。
//   YZ 面：t = y，周期 metadata.period_y，x = at（一维元数据没有 y 周期，此时抛 invalid_argument）。
// 参数非法（nt ≤ 0、at 非有限、周期不是有限正数、谐波/波矢数组长度不一致）抛 invalid_argument；
// carrier 或结果出现非有限数抛 runtime_error（与 reconstruct_field_grid 一致）。
inline FieldGridSlice reconstruct_field_slice(HarmonicFields harmonics, SlicePlane plane, int nt, double at) {
    if (is_periodic_2d(harmonics.metadata)) check_field_metadata_2d(harmonics.metadata);
    const int n = checked_field_size(harmonics.orders.size()), nz = checked_field_size(harmonics.z.size());
    if (harmonics.metadata.kxm.size() < harmonics.orders.size())
        throw std::invalid_argument("rcwa: harmonic wavevectors do not match the harmonic orders");
    for (const auto& component : harmonics.components)
        if (component.rows() != n || component.cols() != nz)
            throw std::invalid_argument("rcwa: harmonic components do not match orders and depths");
    auto coordinates = slice_coordinates(plane, nt, slice_period(harmonics.metadata, plane), at);
    FieldGridSlice out;
    out.plane = plane; out.slice_at = at;
    out.t = std::move(coordinates.t); out.x = std::move(coordinates.x); out.y = std::move(coordinates.y);
    out.z = std::move(harmonics.z);
    out.metadata = std::move(harmonics.metadata);
    if (out.z.empty()) {
        for (auto& component : out.components) component = Matrix(nt, 0);
        return out;
    }
    Matrix carrier(nt, n);
    for (int it = 0; it < nt; ++it) {
        for (int order = 0; order < n; ++order)
            carrier(it, order) = std::exp(-kI * (out.metadata.kxm[order] * out.x[it] + harmonic_ky(out.metadata, order) * out.y[it]));
    }
    if (!carrier.all_finite()) throw std::runtime_error("rcwa: field carrier produced nonfinite values");
    for (std::size_t c = 0; c < 6; ++c) {
        out.components[c] = carrier * harmonics.components[c];
        if (!out.components[c].all_finite()) throw std::runtime_error("rcwa: field reconstruction produced nonfinite values");
    }
    return out;
}

inline bool finite_field_metadata(const FieldMetadata& metadata) {
    if (!std::isfinite(metadata.k0) || !std::isfinite(metadata.z_bottom) || !std::isfinite(metadata.ky) ||
        !std::isfinite(metadata.normalization.electric_incident_amplitude) ||
        !std::isfinite(metadata.normalization.magnetic_incident_amplitude)) return false;
    for (const auto* values : {&metadata.kxm, &metadata.kzI, &metadata.kzII})
        for (cplx value : *values)
            if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) return false;
    // 二维专用字段：一维里是默认值（period_y = 0、kyn 为空），总是有限，不改变一维结果。
    if (!std::isfinite(metadata.period_y)) return false;
    for (double value : metadata.kyn) if (!std::isfinite(value)) return false;
    return true;
}

// 模板只访问 Stack/Layer 的几何字段，让本头文件独立于任何求解器头文件。
template<class StackType>
inline void copy_field_geometry(FieldMetadata& metadata, const StackType& stack, bool skip_zero) {
    metadata.period = stack.period;
    metadata.nI = stack.nI;
    metadata.nII = stack.nII;
    metadata.layers.reserve(stack.layers.size());
    double z = 0.0;
    int effective = 0;
    for (std::size_t j = 0; j < stack.layers.size(); ++j) {
        const auto& layer = stack.layers[j];
        FieldLayerInfo info;
        info.original_index = j;
        info.effective_index = skip_zero && layer.d == 0.0 ? -1 : effective++;
        info.z_top = z;
        info.thickness = layer.d;
        z += layer.d;
        info.z_bottom = z;
        info.homogeneous = layer.homogeneous;
        info.uniform_index = layer.n;
        info.edges = layer.edges;
        info.n_values = layer.n_values;
        metadata.layers.push_back(std::move(info));
    }
    metadata.z_bottom = z;
}

// 二维版本：只访问 Stack2D/Layer2D 的 Lx、Ly、nI、nII、layers[].d / homogeneous / n（同样不依赖求解器头文件）。
// period = Lx、period_y = Ly；层的 z 位置、厚度、均匀标志与 copy_field_geometry 同义，
// 但不复制分段几何（edges / n_values 留空——二维层的 y 分段 × x 剖面不是一维 edges 能表达的）。
// skip_zero 同一维：true 时零厚层的 effective_index 为 -1，其余层按有效层顺序编号。
// 前置条件：metadata 是新建或已重置的（与一维的 copy_field_geometry 相同，这里在 layers 后面追加，不先清空；对同一个 metadata 调用两次
// 层数会翻倍）。StackType 是鸭子类型：只要有 Lx、Ly、nI、nII 与 layers[].d / homogeneous / n 即可（包括 rcwa::Stack2D）；
// 它不检查入射介质无吸收，那是求解器的责任。
template<class StackType>
inline void copy_field_geometry_2d(FieldMetadata& metadata, const StackType& stack, bool skip_zero) {
    metadata.kind = FieldSolveKind::Periodic2D;
    metadata.period = stack.Lx;
    metadata.period_y = stack.Ly;
    metadata.nI = stack.nI;
    metadata.nII = stack.nII;
    metadata.layers.reserve(stack.layers.size());
    double z = 0.0;
    int effective = 0;
    for (std::size_t j = 0; j < stack.layers.size(); ++j) {
        const auto& layer = stack.layers[j];
        FieldLayerInfo info;
        info.original_index = j;
        info.effective_index = skip_zero && layer.d == 0.0 ? -1 : effective++;
        info.z_top = z;
        info.thickness = layer.d;
        z += layer.d;
        info.z_bottom = z;
        info.homogeneous = layer.homogeneous;
        info.uniform_index = layer.n;
        metadata.layers.push_back(std::move(info));
    }
    metadata.z_bottom = z;
}
}  // namespace detail

}  // namespace rcwa
}  // namespace stablercwa
