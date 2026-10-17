// stablercwa/rcwa_phase.hpp —— x-z / y-z 复场相位、参考 carrier、节点 mask 与分段展开
// 仅依赖公共场容器和标准库，不调用任何求解器。所有数组为 [z,横向] C 顺序（横向 = x 或沿切面的 t）。
//
// 两类输入网格，同一套规则：
//   FieldGridXZ    一维求解器的 x-z 网格（固定 y），保持原有接口与数值；
//   FieldGridSlice 切面网格：二维求解器的 x-z（固定 y）或 y-z（固定 x）剖面，见 rcwa_fields.hpp。
// 掩膜（非有限；|F| <= floor·本分量有限幅值峰值；参考非有限或为零）、全零场全部掩膜、
// 展开沿水平轴再沿 z 轴、不跨掩膜节点，对两类网格完全相同。
// 空间 carrier 是 exp(i[-Re(kx0)·x - Re(ky0)·y + Re(kzI0)·z])：(kx0, ky0, kzI0) 是入射零级的物理波矢，
// x、y、z 取每个采样点的实际值——切面网格逐点取 x[it]、y[it]，所以 y-z 面和 x-z 面写法一样，不必分情况。
// 一维：ky0 = metadata.ky；二维（kind == Periodic2D）：零级是 (m,n) = (0,0) 那一列，ky0 = kyn[该列]。
#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <limits>
#include <string>

#include "rcwa_fields.hpp"

namespace stablercwa {
namespace rcwa {

enum class PhaseReference { Absolute, Incident, Custom };
// 展开方向：X（= Horizontal = T）是水平轴——FieldGridXZ 上是 x，切面网格上是沿切面的 t（xz 时 x，yz 时 y）；
// Z（= Vertical）是深度。Both 先水平后 z（不是全局二维展开）。
enum class PhaseUnwrap { None, X, Z, Both, Horizontal = X, Vertical = Z, T = X };
enum class FieldVectorFamily { E, H };

struct PhaseOptions {
    PhaseReference reference = PhaseReference::Absolute;
    double phase_floor = 1e-6;
    PhaseUnwrap unwrap = PhaseUnwrap::None;
    // Custom 时按 Python [nz,nx] 的广播规则：Matrix [x,z] 两轴各为目标长度或 1。
    // 因此 [1,1] 是标量，[nx,1] 是横向行，[1,nz] 是纵向列，[nx,nz] 是完整网格。
    // 切面网格同理，水平轴长度是 nt（Matrix [nt,nz]）。
    Matrix custom_reference;
};

// 水平轴的点数仍叫 nx（保持一维用法）；对切面网格它就是 nt（也可以读 nt()），plane 说明水平轴是 x 还是 y。
// phase / amplitude / mask / field / reference 都是 [nz, nx] C 顺序（水平轴最快）。
struct PhaseGrid {
    std::size_t nx = 0, nz = 0;
    RealVector phase, amplitude;  // rad；无定义的 phase 为 NaN
    std::vector<bool> mask;       // true = 无定义/被掩膜
    Vector field, reference;      // 原始复标量场；absolute 的 reference 为空
    std::string component, reference_name;
    double phase_floor = 1e-6;
    bool unwrapped = false;
    SlicePlane plane = SlicePlane::XZ;  // 水平轴的含义：XZ → x，YZ → y（一维网格恒为 XZ）
    std::size_t nt() const noexcept { return nx; }
    std::size_t index(std::size_t iz, std::size_t ix) const {
        if (iz >= nz || ix >= nx) throw std::out_of_range("rcwa: phase grid index");
        return iz * nx + ix;
    }
    double at(std::size_t iz, std::size_t ix) const { return phase[index(iz, ix)]; }
    bool masked(std::size_t iz, std::size_t ix) const { return mask[index(iz, ix)]; }
};

namespace detail {
inline bool finite_phase_complex(cplx z) { return std::isfinite(z.real()) && std::isfinite(z.imag()); }

inline void check_phase_shape(const FieldGridXZ& grid, const Matrix& component) {
    if (component.rows() != checked_field_size(grid.nx()) || component.cols() != checked_field_size(grid.nz()))
        throw std::invalid_argument("rcwa: phase component does not match x-z coordinates");
}
inline void check_phase_shape(const FieldGridSlice& grid, const Matrix& component) {
    if (grid.plane != SlicePlane::XZ && grid.plane != SlicePlane::YZ)
        throw std::invalid_argument("rcwa: invalid slice plane");
    if (grid.x.size() != grid.t.size() || grid.y.size() != grid.t.size())
        throw std::invalid_argument("rcwa: slice coordinates t, x and y must have the same length");
    if (component.rows() != checked_field_size(grid.nt()) || component.cols() != checked_field_size(grid.nz()))
        throw std::invalid_argument("rcwa: phase component does not match slice coordinates");
}

// 水平轴点数、网格名字（只用于错误信息）和切面方向的传递：模板化的相位核心对两类网格共用。
inline std::size_t phase_columns(const FieldGridXZ& grid) { return grid.nx(); }
inline std::size_t phase_columns(const FieldGridSlice& grid) { return grid.nt(); }
inline const char* phase_grid_name(const FieldGridXZ&) { return "x-z"; }
inline const char* phase_grid_name(const FieldGridSlice& grid) { return grid.plane == SlicePlane::YZ ? "y-z" : "x-z"; }
inline void note_phase_plane(PhaseGrid&, const FieldGridXZ&) {}
inline void note_phase_plane(PhaseGrid& out, const FieldGridSlice& grid) { out.plane = grid.plane; }

inline std::string phase_label(std::string label) {
    const auto first = label.find_first_not_of(" \t\r\n"), last = label.find_last_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    label = label.substr(first, last - first + 1);
    for (auto& c : label) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return label;
}

// 与 np.unwrap 的默认 period=2*pi/discont=pi 一致。mask 节点终止有效连续段。
inline void unwrap_phase_line(RealVector& values, const std::vector<bool>& mask,
                              std::size_t start, std::size_t stride, std::size_t length) {
    bool active = false;
    double previous = 0.0, correction = 0.0;
    for (std::size_t j = 0, index = start; j < length; ++j, index += stride) {
        if (mask[index]) { active = false; correction = 0.0; continue; }
        const double original = values[index];
        if (active) {
            const double delta = original - previous;
            double wrapped = std::fmod(delta + kPi, 2.0 * kPi);
            if (wrapped < 0.0) wrapped += 2.0 * kPi;
            wrapped -= kPi;
            if (wrapped == -kPi && delta > 0.0) wrapped = kPi;
            if (std::abs(delta) >= kPi) correction += wrapped - delta;
            values[index] = original + correction;
        }
        previous = original; active = true;
    }
}

// 入射零级的物理波矢实部 (kx0, ky0, kzI0)。
//   一维：orders 里必须恰有一个 0，kx0 = kxm[零级]，ky0 = metadata.ky，kzI0 = kzI[零级]；
//   二维（kind == Periodic2D）：orders_m/orders_n 里必须恰有一个 (0,0)，kx0 = kxm[该列]，ky0 = kyn[该列]，kzI0 = kzI[该列]。
// 找不到唯一零级、数组长度不够或波矢非有限抛 invalid_argument。
struct IncidentWavevector { double kx = 0.0, ky = 0.0, kz = 0.0; };
inline IncidentWavevector incident_wavevector(const FieldMetadata& meta) {
    const char* const unavailable = "rcwa: incident reference wavevector is unavailable";
    if (is_periodic_2d(meta)) {
        if (meta.orders_m.size() != meta.orders_n.size())
            throw std::invalid_argument("rcwa: 2D field metadata arrays must have one entry per harmonic");
        std::size_t count = 0, i0 = 0;
        for (std::size_t j = 0; j < meta.orders_m.size(); ++j)
            if (meta.orders_m[j] == 0 && meta.orders_n[j] == 0) { ++count; i0 = j; }
        if (count != 1) throw std::invalid_argument("rcwa: incident reference requires a unique zero order");
        if (i0 >= meta.kxm.size() || i0 >= meta.kyn.size() || i0 >= meta.kzI.size() ||
            !finite_phase_complex(meta.kxm[i0]) || !finite_phase_complex(meta.kzI[i0]) || !std::isfinite(meta.kyn[i0]))
            throw std::invalid_argument(unavailable);
        return {meta.kxm[i0].real(), meta.kyn[i0], meta.kzI[i0].real()};
    }
    const auto zero = std::find(meta.orders.begin(), meta.orders.end(), 0);
    if (zero == meta.orders.end() || std::count(meta.orders.begin(), meta.orders.end(), 0) != 1)
        throw std::invalid_argument("rcwa: incident reference requires a unique zero order");
    const auto i0 = static_cast<std::size_t>(zero - meta.orders.begin());
    if (i0 >= meta.kxm.size() || i0 >= meta.kzI.size() ||
        !finite_phase_complex(meta.kxm[i0]) || !finite_phase_complex(meta.kzI[i0]) ||
        !std::isfinite(meta.ky))
        throw std::invalid_argument(unavailable);
    return {meta.kxm[i0].real(), meta.ky, meta.kzI[i0].real()};
}

// 与两个公共入口 complex_component 共用的取值：把六分量之一、或 projection^H @ field 摊平成 [nz, 水平轴] C 顺序。
template<class Grid>
inline Vector component_values(const Grid& grid, FieldComponent component) {
    const auto& values = grid.component(component);
    check_phase_shape(grid, values);
    Vector out(phase_columns(grid) * grid.nz());
    for (std::size_t j = 0; j < out.size(); ++j) out[j] = values.data()[j];
    return out;
}

template<class Grid>
inline Vector projected_values(const Grid& grid, FieldVectorFamily family, const std::array<cplx, 3>& projection) {
    bool nonzero = false;
    for (cplx v : projection) {
        if (!finite_phase_complex(v)) throw std::invalid_argument("rcwa: projection must be finite and nonzero");
        nonzero = nonzero || v != cplx{};
    }
    if (!nonzero) throw std::invalid_argument("rcwa: projection must be finite and nonzero");
    const std::size_t offset = family == FieldVectorFamily::E ? 0 : family == FieldVectorFamily::H ? 3 : 6;
    if (offset == 6) throw std::invalid_argument("rcwa: unsupported field vector family");
    Vector out(phase_columns(grid) * grid.nz());
    for (std::size_t c = 0; c < 3; ++c) {
        check_phase_shape(grid, grid.components[offset + c]);
        for (std::size_t j = 0; j < out.size(); ++j) out[j] += std::conj(projection[c]) * grid.components[offset + c].data()[j];
    }
    return out;
}
}  // namespace detail

inline Vector complex_component(const FieldGridXZ& grid, FieldComponent component) {
    return detail::component_values(grid, component);
}
inline Vector complex_component(const FieldGridSlice& grid, FieldComponent component) {
    return detail::component_values(grid, component);
}

// projection^H @ field，不把分析器向量归一化；只接受有限非零的三复数。
inline Vector complex_component(const FieldGridXZ& grid, FieldVectorFamily family,
                                 const std::array<cplx, 3>& projection) {
    return detail::projected_values(grid, family, projection);
}
inline Vector complex_component(const FieldGridSlice& grid, FieldVectorFamily family,
                                 const std::array<cplx, 3>& projection) {
    return detail::projected_values(grid, family, projection);
}

// 单位模空间 carrier；只用物理波矢的实部，不扣偏振符号、Jones 整体相位或参考幅值。
inline Vector incident_reference(const FieldGridXZ& grid) {
    const auto k = detail::incident_wavevector(grid.metadata);
    if (!std::isfinite(grid.y)) throw std::invalid_argument("rcwa: incident reference wavevector is unavailable");
    Vector out(grid.nx() * grid.nz());
    for (std::size_t iz = 0; iz < grid.nz(); ++iz) for (std::size_t ix = 0; ix < grid.nx(); ++ix) {
        const double argument = -k.kx * grid.x[ix] - k.ky * grid.y + k.kz * grid.z[iz];
        if (!std::isfinite(argument)) throw std::invalid_argument("rcwa: incident reference coordinates must be finite and representable");
        out[iz * grid.nx() + ix] = std::exp(kI * argument);
    }
    return out;
}

// 切面网格的 carrier：(x, y) 逐点取 grid.x[it]、grid.y[it]（xz 面里 y 恒为 slice_at，yz 面里 x 恒为 slice_at）。
// 二维下 ky0 非零，所以 yz 面里 -ky0·y 随 t 变化，不能当常数丢掉。
inline Vector incident_reference(const FieldGridSlice& grid) {
    const auto k = detail::incident_wavevector(grid.metadata);
    const std::size_t nt = grid.nt(), nz = grid.nz();
    if (grid.plane != SlicePlane::XZ && grid.plane != SlicePlane::YZ) throw std::invalid_argument("rcwa: invalid slice plane");
    if (grid.x.size() != nt || grid.y.size() != nt)
        throw std::invalid_argument("rcwa: slice coordinates t, x and y must have the same length");
    Vector out(nt * nz);
    for (std::size_t iz = 0; iz < nz; ++iz) for (std::size_t it = 0; it < nt; ++it) {
        const double argument = -k.kx * grid.x[it] - k.ky * grid.y[it] + k.kz * grid.z[iz];
        if (!std::isfinite(argument)) throw std::invalid_argument("rcwa: incident reference coordinates must be finite and representable");
        out[iz * nt + it] = std::exp(kI * argument);
    }
    return out;
}

namespace detail {
// 掩膜 / 相位 / 展开的共同核心：Grid 是 FieldGridXZ 或 FieldGridSlice（水平轴点数 phase_columns）。
// FieldGridXZ 的运算次序与原实现完全相同。
template<class Grid>
inline PhaseGrid scalar_phase_data(const Grid& grid, Vector field, const std::string& label,
                                   const PhaseOptions& options) {
    if (!std::isfinite(options.phase_floor) || options.phase_floor < 0.0)
        throw std::invalid_argument("rcwa: phase_floor must be finite and nonnegative");
    if (options.unwrap != PhaseUnwrap::None && options.unwrap != PhaseUnwrap::X &&
        options.unwrap != PhaseUnwrap::Z && options.unwrap != PhaseUnwrap::Both)
        throw std::invalid_argument("rcwa: unsupported phase unwrap axis");
    PhaseGrid out;
    out.nx = phase_columns(grid); out.nz = grid.nz(); out.field = std::move(field);
    out.component = label; out.phase_floor = options.phase_floor;
    out.unwrapped = options.unwrap != PhaseUnwrap::None;
    note_phase_plane(out, grid);
    const std::size_t size = out.field.size();
    switch (options.reference) {
    case PhaseReference::Absolute: out.reference_name = "absolute"; break;
    case PhaseReference::Incident: out.reference_name = "incident"; out.reference = incident_reference(grid); break;
    case PhaseReference::Custom: {
        out.reference_name = "custom";
        const auto& ref = options.custom_reference;
        if ((ref.rows() != 1 && ref.rows() != checked_field_size(out.nx)) ||
            (ref.cols() != 1 && ref.cols() != checked_field_size(out.nz)))
            throw std::invalid_argument(std::string("rcwa: custom phase reference cannot broadcast to the ") + phase_grid_name(grid) + " grid");
        out.reference.resize(size);
        for (std::size_t iz = 0; iz < out.nz; ++iz) for (std::size_t ix = 0; ix < out.nx; ++ix)
            out.reference[iz * out.nx + ix] = ref(ref.rows() == 1 ? 0 : static_cast<int>(ix), ref.cols() == 1 ? 0 : static_cast<int>(iz));
        break;
    }
    default: throw std::invalid_argument("rcwa: unsupported phase reference");
    }
    out.amplitude.resize(size); out.phase.resize(size); out.mask.resize(size);
    double peak = 0.0;
    for (std::size_t j = 0; j < size; ++j) {
        out.amplitude[j] = std::abs(out.field[j]);
        if (std::isfinite(out.amplitude[j])) peak = (std::max)(peak, out.amplitude[j]);
    }
    const bool has_reference = options.reference != PhaseReference::Absolute;
    for (std::size_t j = 0; j < size; ++j) {
        out.mask[j] = !finite_phase_complex(out.field[j]) || out.amplitude[j] <= options.phase_floor * peak;
        if (has_reference) out.mask[j] = out.mask[j] || !finite_phase_complex(out.reference[j]) || std::abs(out.reference[j]) == 0.0;
        out.phase[j] = std::arg(has_reference ? out.field[j] * std::conj(out.reference[j]) : out.field[j]);
    }
    if (options.unwrap == PhaseUnwrap::X || options.unwrap == PhaseUnwrap::Both)
        for (std::size_t iz = 0; iz < out.nz; ++iz) unwrap_phase_line(out.phase, out.mask, iz * out.nx, 1, out.nx);
    if (options.unwrap == PhaseUnwrap::Z || options.unwrap == PhaseUnwrap::Both)
        for (std::size_t ix = 0; ix < out.nx; ++ix) unwrap_phase_line(out.phase, out.mask, ix, out.nx, out.nz);
    for (std::size_t j = 0; j < size; ++j) if (out.mask[j]) out.phase[j] = std::numeric_limits<double>::quiet_NaN();
    return out;
}

template<class Grid>
inline PhaseGrid named_phase_data(const Grid& grid, const std::string& component, const PhaseOptions& options) {
    const std::string label = phase_label(component);
    if (label == "E2" || label == "H2") throw std::invalid_argument("rcwa: real intensity has no complex phase");
    for (std::size_t c = 0; c < 6; ++c)
        if (label == phase_label(field_component_name(static_cast<FieldComponent>(c))))
            return scalar_phase_data(grid, complex_component(grid, static_cast<FieldComponent>(c)),
                                     field_component_name(static_cast<FieldComponent>(c)), options);
    throw std::invalid_argument("rcwa: phase requires Ex/Ey/Ez/Hx/Hy/Hz or an E/H projection");
}
}  // namespace detail

inline PhaseGrid phase_data(const FieldGridXZ& grid, FieldComponent component, const PhaseOptions& options = {}) {
    return detail::scalar_phase_data(grid, complex_component(grid, component), field_component_name(component), options);
}
inline PhaseGrid phase_data(const FieldGridXZ& grid, FieldVectorFamily family, const std::array<cplx, 3>& projection,
                            const PhaseOptions& options = {}) {
    return detail::scalar_phase_data(grid, complex_component(grid, family, projection),
                                     family == FieldVectorFamily::E ? "E_projection" : "H_projection", options);
}
inline PhaseGrid phase_data(const FieldGridXZ& grid, const std::string& component, const PhaseOptions& options = {}) {
    return detail::named_phase_data(grid, component, options);
}

// 切面网格（x-z 或 y-z）的相位入口：与上面三个同名同义，规则完全相同。
inline PhaseGrid phase_data(const FieldGridSlice& grid, FieldComponent component, const PhaseOptions& options = {}) {
    return detail::scalar_phase_data(grid, complex_component(grid, component), field_component_name(component), options);
}
inline PhaseGrid phase_data(const FieldGridSlice& grid, FieldVectorFamily family, const std::array<cplx, 3>& projection,
                            const PhaseOptions& options = {}) {
    return detail::scalar_phase_data(grid, complex_component(grid, family, projection),
                                     family == FieldVectorFamily::E ? "E_projection" : "H_projection", options);
}
inline PhaseGrid phase_data(const FieldGridSlice& grid, const std::string& component, const PhaseOptions& options = {}) {
    return detail::named_phase_data(grid, component, options);
}

}  // namespace rcwa
}  // namespace stablercwa
