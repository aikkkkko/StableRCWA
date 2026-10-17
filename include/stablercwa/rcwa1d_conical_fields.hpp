// stablercwa/rcwa1d_conical_fields.hpp —— 圆锥求解与两列入射基的近场上下文
#pragma once

#include <stdexcept>
#include <utility>

#include "rcwa1d_conical.hpp"
#include "rcwa_fields.hpp"

namespace stablercwa {
namespace rcwa {

namespace detail { struct ConicalFieldAccess; struct ConicalFieldSampler; }
struct ConicalFieldResult;

class ConicalFieldContext {
public:
    ConicalFieldContext() = default;
    ConicalFieldContext(const ConicalFieldContext&) = default;
    ConicalFieldContext& operator=(const ConicalFieldContext&) = default;
    ConicalFieldContext(ConicalFieldContext&& other) noexcept { *this = std::move(other); }
    ConicalFieldContext& operator=(ConicalFieldContext&& other) noexcept {
        if (this != &other) {
            metadata_ = std::move(other.metadata_);
            capture_ = std::move(other.capture_);
            status_ = std::exchange(other.status_, FieldContextStatus::Empty);
            other.metadata_ = FieldMetadata{};
            other.capture_ = detail::conical::FieldCapture{};
        }
        return *this;
    }
    FieldContextStatus status() const noexcept { return status_; }
    bool available() const noexcept { return status_ == FieldContextStatus::Ready; }
    const FieldMetadata& metadata() const noexcept { return metadata_; }
    void require_available() const {
        if (!available()) throw std::logic_error("rcwa: conical field context is unavailable");
    }

private:
    FieldMetadata metadata_;
    detail::conical::FieldCapture capture_;
    FieldContextStatus status_ = FieldContextStatus::Empty;
    friend struct detail::ConicalFieldAccess;
    friend struct detail::ConicalFieldSampler;
    friend ConicalFieldResult solve_conical_with_fields(const Stack&, double, double, double, int, const Options&);
};

struct ConicalFieldResult {
    ConicalResult result;
    ConicalFieldContext fields;
};

// 缓存列次序 s、p，与单位完整入射电场对应。后选 Jones 时无需再次本征求解。
inline ConicalFieldResult solve_conical_with_fields(const Stack& stack, double wavelength, double theta_deg,
                                                   double phi_deg, int M, const Options& options = {}) {
    ConicalFieldResult out;
    auto& context = out.fields;
    out.result = detail::solve_conical_impl(stack, wavelength, theta_deg, phi_deg, M, options, &context.capture_);
    context.status_ = context.capture_.available ? FieldContextStatus::Ready : FieldContextStatus::Unavailable;
    auto& meta = context.metadata_;
    meta.kind = FieldSolveKind::Conical;
    meta.wavelength = wavelength;
    meta.k0 = 2.0 * kPi / wavelength;
    meta.theta_deg = theta_deg;
    meta.phi_deg = phi_deg;
    meta.truncation_order = M;
    meta.incident_basis_columns = 2;
    meta.orders = out.result.orders;
    meta.normalization.electric_incident_amplitude = 1.0;
    meta.normalization.magnetic_incident_amplitude = stack.nI.real();
    detail::copy_field_geometry(meta, stack, true);
    meta.ky = meta.k0 * out.result.ky;
    meta.kxm.resize(out.result.kx.size());
    meta.kzI.resize(out.result.kzI.size());
    meta.kzII.resize(out.result.kzII.size());
    for (std::size_t j = 0; j < meta.kxm.size(); ++j) {
        meta.kxm[j] = meta.k0 * out.result.kx[j];
        meta.kzI[j] = meta.k0 * out.result.kzI[j];
        meta.kzII[j] = meta.k0 * out.result.kzII[j];
    }
    if (!detail::finite_field_metadata(meta)) {
        context.status_ = FieldContextStatus::Unavailable;
        context.capture_ = detail::conical::FieldCapture{};
    }
    if (context.available()) for (auto& layer : meta.layers) {
        if (layer.effective_index < 0) continue;
        const auto& mode = context.capture_.modes[layer.effective_index + 1];
        layer.has_modes = true;
        layer.propagates = true;
        layer.analytic_modes = mode.analytic;
        layer.mode_count = static_cast<int>(mode.q.size());
        layer.propagation = mode.G.empty() ? FieldPropagation::Exponential : FieldPropagation::Schur;
    }
    return out;
}

namespace detail {
inline Vector field_jones_response(const Matrix& responses, const JonesVector& jones) {
    Vector out(responses.rows());
    for (int j = 0; j < responses.rows(); ++j) out[j] = responses(j, 0) * jones[0] + responses(j, 1) * jones[1];
    return out;
}

// 同一锚点的非负距离排序推进；重复深度复用向量，等距深度复用一个步进矩阵。
// 工作内存是输出 O(N*nz) 与固定上限的 O(N²) 步进缓存，不为每个深度缓存完整指数。
inline Matrix field_schur_action(const Matrix& generator, const Vector& anchor,
                                 const RealVector& distances, double k0) {
    const int n = checked_field_size(anchor.size()), count = checked_field_size(distances.size());
    Matrix out(n, count);
    if (std::all_of(anchor.begin(), anchor.end(), [](cplx a) { return a == cplx{}; })) return out;
    bool diagonal = true;
    for (int j = 0; j < n; ++j) for (int i = 0; i < j; ++i) diagonal = diagonal && generator(i, j) == cplx{};
    // 独立模不用步进：直接锚点指数也只有 O(N*nz)，并保留超大相位的相同求值顺序。
    if (diagonal) {
        for (int col = 0; col < count; ++col) for (int j = 0; j < n; ++j)
            if (anchor[j] != cplx{}) out(j, col) = anchor[j] * std::exp(-k0 * distances[col] * generator(j, j));
        return out;
    }
    // 绝对传播相位已大到其舍入误差影响步进/直接一致性时，改为每点从锚点传播。
    // 这只限制采样算法，不修改生成元、根或坐标；任意深度仍只保留一个临时指数。
    const double scale = k0 * norm_1(generator);
    for (double distance : distances) if (scale * distance * kEps > 1e-10) {
        for (int col = 0; col < count; ++col) {
            const Vector value = conical::triangular_exp(generator * cplx(-k0 * distances[col])) * anchor;
            for (int j = 0; j < n; ++j) out(j, col) = value[j];
        }
        return out;
    }
    std::vector<int> order(count);
    for (int j = 0; j < count; ++j) order[j] = j;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return distances[a] < distances[b]; });
    Vector value = anchor;
    struct Step { double distance = -1.0; Matrix propagation; };
    std::array<Step, 8> steps;
    std::size_t next_step = 0;
    double previous = 0.0;
    for (int col : order) {
        const double distance = distances[col], delta = distance - previous;
        if (delta > 0.0) {
            Step* step = nullptr;
            // 只复用位级相同的距离，不能用随绝对 z 增长的容差吞掉物理显著的微小步差。
            for (auto& candidate : steps) if (candidate.distance == delta) { step = &candidate; break; }
            if (!step) {
                step = &steps[next_step++ % steps.size()];
                step->propagation = conical::triangular_exp(generator * cplx(-k0 * delta));
                step->distance = delta;
            }
            value = step->propagation * value;
        }
        for (int j = 0; j < n; ++j) out(j, col) = value[j];
        previous = distance;
    }
    return out;
}

struct ConicalFieldSampler {
    static HarmonicFields sample(const ConicalFieldContext& context, const RealVector& zs, JonesVector jones) {
        context.require_available();
        jones = normalized_jones(jones);
        const auto& meta = context.metadata_;
        const auto& state = context.capture_;
        const auto groups = field_depth_groups(meta, zs);
        auto out = make_harmonic_fields(meta, zs);
        const int n = checked_field_size(meta.orders.size()), i0 = meta.truncation_order;
        for (const auto& group : groups) {
            if (group.columns.empty()) continue;
            const int count = checked_field_size(group.columns.size()), region = group.region;
            Matrix F(2 * n, count), G(2 * n, count);
            if (group.layer) {
                const auto& mode = state.modes[region + 1];
                const double d = group.layer->thickness;
                const Vector a = field_jones_response(state.a_plus[region], jones);
                const Vector b = field_jones_response(state.a_minus[region], jones);
                RealVector from_top(count), from_bottom(count);
                for (int col = 0; col < count; ++col) {
                    from_top[col] = (std::max)(0.0, (std::min)(d, zs[group.columns[col]] - group.layer->z_top));
                    from_bottom[col] = d - from_top[col];
                }
                Matrix p(2 * n, count), m(2 * n, count);
                if (!mode.G.empty()) {
                    try {
                        p = field_schur_action(mode.G, a, from_top, meta.k0);
                        m = field_schur_action(mode.G, b, from_bottom, meta.k0);
                    } catch (const NumericalFailure&) { throw std::runtime_error("rcwa: Schur field propagation failed"); }
                } else {
                    for (int col = 0; col < count; ++col) for (int j = 0; j < 2 * n; ++j) {
                        p(j, col) = a[j] == cplx{} ? cplx{} : a[j] * std::exp(-meta.k0 * from_top[col] * mode.q[j]);
                        m(j, col) = b[j] == cplx{} ? cplx{} : b[j] * std::exp(-meta.k0 * from_bottom[col] * mode.q[j]);
                    }
                }
                F = p + m; G = -p + m;
                if (mode.analytic) { F = scale_rows(std::move(F), mode.w); G = scale_rows(std::move(G), mode.v); }
                else { F = mode.W * F; G = mode.V * G; }
            } else {
                const auto& mode = region == -1 ? state.modes.front() : state.modes.back();
                const Vector response = field_jones_response(region == -1 ? state.reflected : state.transmitted, jones);
                for (int col = 0; col < count; ++col) {
                    const double z = zs[group.columns[col]];
                    const cplx inc = region == -1 ? std::exp(kI * meta.kzI[i0] * z) : cplx{};
                    for (int j = 0; j < 2 * n; ++j) {
                        const int order = j % n;
                        const cplx a = region == -1 && order == i0 ? (j < n ? jones[0] : jones[1]) * inc : cplx{};
                        const cplx b = region == -1 && response[j] != cplx{} ? response[j] * std::exp(-kI * meta.kzI[order] * z) : cplx{};
                        const cplx t = region == -2 && response[j] != cplx{} ? response[j] * std::exp(kI * meta.kzII[order] * (z - meta.z_bottom)) : cplx{};
                        F(j, col) = mode.w[j] * (a + b + t);
                        G(j, col) = mode.v[j] * (-a + b - t);
                    }
                }
            }
            Matrix Ex(n, count), Ey(n, count), Hx(n, count), Hy(n, count), Hz(n, count), Ez(n, count);
            for (int col = 0; col < count; ++col) for (int j = 0; j < n; ++j) {
                const double cs = state.cs[j], sn = state.sn[j];
                Ex(j, col) = -sn * F(j, col) + cs * F(n + j, col);
                Ey(j, col) = cs * F(j, col) + sn * F(n + j, col);
                // G 是物理 [-ht;hs]，不是一阶指数测试辅助状态中的反号 G。
                const cplx hx = -cs * G(j, col) - sn * G(n + j, col);
                const cplx hy = -sn * G(j, col) + cs * G(n + j, col);
                Hx(j, col) = -kI * hx; Hy(j, col) = -kI * hy;
                Hz(j, col) = state.ky * Ex(j, col) - state.kx[j] * Ey(j, col);
                Ez(j, col) = kI * (state.ky * hx - state.kx[j] * hy);
            }
            if (group.layer && !state.materials[region].analytic) Ez = state.materials[region].inv_eps * Ez;
            else {
                const cplx eps = group.layer ? state.materials[region].epsilon :
                    region == -1 ? meta.nI * meta.nI : meta.nII * meta.nII;
                Ez *= 1.0 / eps;
            }
            const std::array<const Matrix*, 6> values{{&Ex, &Ey, &Ez, &Hx, &Hy, &Hz}};
            for (std::size_t c = 0; c < 6; ++c) for (int col = 0; col < count; ++col)
                for (int j = 0; j < n; ++j) out.components[c](j, group.columns[col]) = (*values[c])(j, col);
        }
        for (const auto& component : out.components)
            if (!component.all_finite()) throw std::runtime_error("rcwa: conical field propagation produced nonfinite values");
        return out;
    }
};
}  // namespace detail

inline HarmonicFields field_at(const ConicalFieldContext& context, const RealVector& zs, JonesVector jones = {{1.0, 0.0}}) {
    return detail::ConicalFieldSampler::sample(context, zs, jones);
}
inline HarmonicFields field_at(const ConicalFieldContext& context, double z, JonesVector jones = {{1.0, 0.0}}) {
    return field_at(context, RealVector{z}, jones);
}
inline HarmonicFields field_at(const ConicalFieldResult& result, const RealVector& zs, JonesVector jones = {{1.0, 0.0}}) {
    return field_at(result.fields, zs, jones);
}
inline HarmonicFields field_at(const ConicalFieldResult& result, double z, JonesVector jones = {{1.0, 0.0}}) {
    return field_at(result.fields, z, jones);
}
inline FieldGridXZ field_map(const ConicalFieldContext& context, const RealVector& zs, int nx = 256,
                             double y = 0.0, JonesVector jones = {{1.0, 0.0}}) {
    detail::validate_field_grid(nx, y);
    return detail::reconstruct_field_grid(field_at(context, zs, jones), nx, y);
}
inline FieldGridXZ field_map(const ConicalFieldResult& result, const RealVector& zs, int nx = 256,
                             double y = 0.0, JonesVector jones = {{1.0, 0.0}}) {
    return field_map(result.fields, zs, nx, y, jones);
}

}  // namespace rcwa
}  // namespace stablercwa
