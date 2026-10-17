// stablercwa/rcwa1d_fields.hpp —— 平面 TE/TM 求解与拥有数据的近场上下文
#pragma once

#include <stdexcept>
#include <utility>

#include "rcwa1d.hpp"
#include "rcwa_fields.hpp"

namespace stablercwa {
namespace rcwa {

namespace detail { struct PlanarFieldAccess; struct PlanarFieldSampler; }  // 非稳定的实现/测试访问点
struct PlanarFieldResult;

class PlanarFieldContext {
public:
    PlanarFieldContext() = default;
    // 拷贝是深拷贝；移动后源对象为 Empty，元数据也重置。没有 Stack 指针或共享可变缓存。
    PlanarFieldContext(const PlanarFieldContext&) = default;
    PlanarFieldContext& operator=(const PlanarFieldContext&) = default;
    PlanarFieldContext(PlanarFieldContext&& other) noexcept { *this = std::move(other); }
    PlanarFieldContext& operator=(PlanarFieldContext&& other) noexcept {
        if (this != &other) {
            metadata_ = std::move(other.metadata_);
            capture_ = std::move(other.capture_);
            status_ = std::exchange(other.status_, FieldContextStatus::Empty);
            other.metadata_ = FieldMetadata{};
            other.capture_ = detail::PlanarFieldCapture{};
        }
        return *this;
    }
    FieldContextStatus status() const noexcept { return status_; }
    bool available() const noexcept { return status_ == FieldContextStatus::Ready; }
    const FieldMetadata& metadata() const noexcept { return metadata_; }
    void require_available() const {
        if (!available()) throw std::logic_error("rcwa: planar field context is unavailable");
    }

private:
    FieldMetadata metadata_;
    detail::PlanarFieldCapture capture_;
    FieldContextStatus status_ = FieldContextStatus::Empty;
    friend struct detail::PlanarFieldAccess;
    friend struct detail::PlanarFieldSampler;
    friend PlanarFieldResult solve_with_fields(const Stack&, double, double, int, Polarization, const Options&);
};

struct PlanarFieldResult {
    Result result;
    PlanarFieldContext fields;
};

// 一次本征求解同时得到原 Result 和场快照。数值失败沿用 Result 的 NaN 语义；上下文标为 Unavailable。
inline PlanarFieldResult solve_with_fields(const Stack& stack, double wavelength, double theta_deg, int M,
                                           Polarization pol = Polarization::TE, const Options& options = {}) {
    PlanarFieldResult out;
    auto& context = out.fields;
    out.result = detail::solve_impl(stack, wavelength, theta_deg, M, pol, options, &context.capture_);
    context.status_ = context.capture_.available ? FieldContextStatus::Ready : FieldContextStatus::Unavailable;
    auto& meta = context.metadata_;
    meta.kind = pol == Polarization::TE ? FieldSolveKind::PlanarTE : FieldSolveKind::PlanarTM;
    meta.wavelength = wavelength;
    meta.k0 = 2.0 * kPi / wavelength;
    meta.theta_deg = theta_deg;
    meta.truncation_order = M;
    meta.incident_basis_columns = 1;
    meta.orders = out.result.orders;
    meta.normalization.electric_incident_amplitude = pol == Polarization::TE ? 1.0 : 1.0 / stack.nI.real();
    meta.normalization.magnetic_incident_amplitude = pol == Polarization::TE ? stack.nI.real() : 1.0;
    detail::copy_field_geometry(meta, stack, false);
    const Vector kx = detail::transverse_wavevectors(stack.nI.real(), theta_deg, wavelength, stack.period, M);
    meta.kxm.resize(kx.size());
    meta.kzI.resize(kx.size());
    meta.kzII.resize(kx.size());
    // 成功时直接读同次求解保留的外部模式。失败时也给出输入决定的波矢元数据。
    detail::Modes failed_up, failed_low;
    if (!context.available()) {
        failed_up = detail::analytic_modes(stack.nI, kx, pol, false);
        failed_low = detail::analytic_modes(stack.nII, kx, pol, false);
    }
    const auto& up = context.available() ? context.capture_.modes.front() : failed_up;
    const auto& low = context.available() ? context.capture_.modes.back() : failed_low;
    for (std::size_t j = 0; j < kx.size(); ++j) {
        meta.kxm[j] = meta.k0 * kx[j];
        meta.kzI[j] = cplx(-meta.k0 * up.q[j].imag(), meta.k0 * up.q[j].real());
        meta.kzII[j] = cplx(-meta.k0 * low.q[j].imag(), meta.k0 * low.q[j].real());
    }
    if (!detail::finite_field_metadata(meta)) {
        context.status_ = FieldContextStatus::Unavailable;
        context.capture_ = detail::PlanarFieldCapture{};
    }
    if (context.available()) for (std::size_t j = 0; j < meta.layers.size(); ++j) {
        const auto& mode = context.capture_.modes[j + 1];
        auto& layer = meta.layers[j];
        layer.has_modes = true;
        layer.propagates = layer.thickness > 0.0;
        layer.analytic_modes = mode.analytic;
        layer.mode_count = static_cast<int>(mode.q.size());
        if (mode.finite_zero_basis) for (cplx q : mode.q) if (q == cplx{}) ++layer.finite_zero_mode_count;
        layer.propagation = layer.finite_zero_mode_count ? FieldPropagation::FiniteZero : FieldPropagation::Exponential;
    }
    return out;
}

namespace detail {
struct PlanarFieldSampler {
    static HarmonicFields sample(const PlanarFieldContext& context, const RealVector& zs) {
        context.require_available();
        const auto& meta = context.metadata_;
        const auto& state = context.capture_;
        const auto groups = field_depth_groups(meta, zs);
        auto out = make_harmonic_fields(meta, zs);
        const int n = checked_field_size(meta.orders.size()), i0 = meta.truncation_order;
        for (const auto& group : groups) {
            if (group.columns.empty()) continue;
            const int count = checked_field_size(group.columns.size()), region = group.region;
            Matrix S(n, count), U(n, count);
            if (group.layer) {
                const auto& mode = state.modes[region + 1];
                const double d = group.layer->thickness;
                for (int col = 0; col < count; ++col) {
                    const double z = (std::max)(0.0, (std::min)(d, zs[group.columns[col]] - group.layer->z_top));
                    for (int j = 0; j < n; ++j) {
                        const cplx a = state.a_plus[region][j], b = state.a_minus[region][j];
                        if (mode.finite_zero_basis && mode.q[j] == cplx{}) {
                            const double denominator = 1.0 + 0.5 * meta.k0 * d;
                            // 先形成距离系数，避免 (1+h)-h 丢失界面常量项及大振幅中间乘积溢出。
                            const double from_bottom = (1.0 + meta.k0 * (d - z)) / denominator;
                            const double from_top = (1.0 + meta.k0 * z) / denominator;
                            S(j, col) = from_bottom * a + from_top * b;
                            U(j, col) = -a / denominator + b / denominator;
                        } else {
                            // 距离从各自锚点沿非增长方向取值；零振幅不求指数，避免 0*Inf。
                            const cplx p = a == cplx{} ? cplx{} : a * std::exp(-meta.k0 * z * mode.q[j]);
                            const cplx m = b == cplx{} ? cplx{} : b * std::exp(-meta.k0 * (d - z) * mode.q[j]);
                            S(j, col) = p + m; U(j, col) = -p + m;
                        }
                    }
                }
                if (mode.analytic) U = scale_rows(std::move(U), mode.p);
                else { S = mode.W * S; U = mode.V * U; }
            } else {
                const auto& mode = region == -1 ? state.modes.front() : state.modes.back();
                for (int col = 0; col < count; ++col) {
                    const double z = zs[group.columns[col]];
                    const cplx inc = region == -1 ? std::exp(kI * meta.kzI[i0] * z) : cplx{};
                    for (int j = 0; j < n; ++j) {
                        const cplx p = j == i0 ? inc : cplx{};
                        const cplx r = region == -1 && state.reflected[j] != cplx{} ?
                            state.reflected[j] * std::exp(-kI * meta.kzI[j] * z) : cplx{};
                        const cplx t = region == -2 && state.transmitted[j] != cplx{} ?
                            state.transmitted[j] * std::exp(kI * meta.kzII[j] * (z - meta.z_bottom)) : cplx{};
                        S(j, col) = p + r + t; U(j, col) = mode.p[j] * (-p + r - t);
                    }
                }
            }
            if (meta.kind == FieldSolveKind::PlanarTE) {
                for (int col = 0; col < count; ++col) for (int j = 0; j < n; ++j) {
                    const int iz = group.columns[col];
                    out.components[1](j, iz) = S(j, col);
                    out.components[3](j, iz) = kI * U(j, col);
                    out.components[5](j, iz) = -state.kx[j] * S(j, col);
                }
            } else {
                Matrix Ez = scale_rows(S, state.kx);
                if (group.layer) {
                    const auto& material = state.materials[region];
                    if (material.analytic) Ez *= 1.0 / material.epsilon;
                    else {
                        if (!material.factorized) throw std::logic_error("rcwa: missing retained Eps factors");
                        Ez = material.eps_factors.method == LinearSolver::QRHouseholder ?
                            qr_solve(material.eps_factors.qr, std::move(Ez)) :
                            lu_solve(material.eps_factors.lu, std::move(Ez));
                    }
                } else Ez *= 1.0 / (region == -1 ? meta.nI * meta.nI : meta.nII * meta.nII);
                for (int col = 0; col < count; ++col) for (int j = 0; j < n; ++j) {
                    const int iz = group.columns[col];
                    out.components[0](j, iz) = -kI * U(j, col);
                    out.components[2](j, iz) = Ez(j, col);
                    out.components[4](j, iz) = S(j, col);
                }
            }
        }
        for (const auto& component : out.components)
            if (!component.all_finite()) throw std::runtime_error("rcwa: planar field propagation produced nonfinite values");
        return out;
    }
};
}  // namespace detail

// 只读同次求解的快照，不重新求模式或材料分解。多深度一次按区域/层分组。
inline HarmonicFields field_at(const PlanarFieldContext& context, const RealVector& zs) {
    return detail::PlanarFieldSampler::sample(context, zs);
}
inline HarmonicFields field_at(const PlanarFieldContext& context, double z) { return field_at(context, RealVector{z}); }
inline HarmonicFields field_at(const PlanarFieldResult& result, const RealVector& zs) { return field_at(result.fields, zs); }
inline HarmonicFields field_at(const PlanarFieldResult& result, double z) { return field_at(result.fields, z); }

inline FieldGridXZ field_map(const PlanarFieldContext& context, const RealVector& zs, int nx = 256, double y = 0.0) {
    detail::validate_field_grid(nx, y);
    return detail::reconstruct_field_grid(field_at(context, zs), nx, y);
}
inline FieldGridXZ field_map(const PlanarFieldResult& result, const RealVector& zs, int nx = 256, double y = 0.0) {
    return field_map(result.fields, zs, nx, y);
}

}  // namespace rcwa
}  // namespace stablercwa
