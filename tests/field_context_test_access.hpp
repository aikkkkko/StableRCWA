// 仅用于验证快照是否足以重建场，不安装或发布为求解器 API。
#pragma once

#include <array>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "stablercwa/rcwa1d_fields.hpp"
#include "stablercwa/rcwa1d_conical_fields.hpp"

namespace stablercwa { namespace rcwa { namespace detail {
struct PlanarFieldAccess {
    static const PlanarFieldCapture& get(const PlanarFieldContext& c) { c.require_available(); return c.capture_; }
};
struct ConicalFieldAccess {
    static const conical::FieldCapture& get(const ConicalFieldContext& c) { c.require_available(); return c.capture_; }
};
}}}

namespace field_context_test {
using namespace stablercwa;
namespace rc = stablercwa::rcwa;
namespace dt = rc::detail;
namespace dc = rc::detail::conical;
using Harmonics = std::array<Vector, 6>;

inline int layer_at(const rc::FieldMetadata& meta, double z) {
    if (z < 0.0) return -1;
    if (z > meta.z_bottom) return -2;
    for (const auto& layer : meta.layers)
        if (layer.has_modes && z <= layer.z_bottom + 1e-12) return layer.effective_index;
    throw std::invalid_argument("test reconstruction: no layer at this depth");
}

inline Vector epsilon_solve(const dt::PlanarFieldMaterial& material, const Vector& rhs) {
    if (material.analytic) {
        Vector out = rhs;
        for (auto& v : out) v /= material.epsilon;
        return out;
    }
    if (!material.factorized) throw std::logic_error("test reconstruction: no Eps solve factors");
    if (material.eps_factors.method != rc::LinearSolver::QRHouseholder)
        return lu_solve(material.eps_factors.lu, rhs);
    Matrix b(static_cast<int>(rhs.size()), 1);
    for (int j = 0; j < b.rows(); ++j) b(j, 0) = rhs[j];
    return qr_solve(material.eps_factors.qr, b).column(0);
}

inline std::pair<Vector, Vector> planar_layer(const rc::PlanarFieldContext& context, int layer, double z_local) {
    const auto& state = dt::PlanarFieldAccess::get(context);
    const auto& meta = context.metadata();
    const auto& mode = state.modes[layer + 1];
    const double d = meta.layers[layer].thickness;
    const double z = (std::max)(0.0, (std::min)(d, z_local));
    const int n = static_cast<int>(mode.q.size());
    Vector sum(n), difference(n);
    for (int j = 0; j < n; ++j) {
        const cplx a = state.a_plus[layer][j], b = state.a_minus[layer][j];
        if (mode.finite_zero_basis && mode.q[j] == cplx{}) {
            const double h = meta.k0 * d, xi = meta.k0 * z;
            const double factor = 1.0 / (1.0 + 0.5 * h);
            sum[j] = factor * ((1.0 + h - xi) * a + (1.0 + xi) * b);
            difference[j] = factor * (-a + b);
        } else {
            const cplx p = std::exp(-meta.k0 * z * mode.q[j]) * a;
            const cplx m = std::exp(-meta.k0 * (d - z) * mode.q[j]) * b;
            sum[j] = p + m;
            difference[j] = -p + m;
        }
    }
    if (!mode.analytic) return {mode.W * sum, mode.V * difference};
    for (int j = 0; j < n; ++j) difference[j] *= mode.p[j];
    return {std::move(sum), std::move(difference)};
}

inline Harmonics planar(const rc::PlanarFieldContext& context, double z) {
    const auto& state = dt::PlanarFieldAccess::get(context);
    const auto& meta = context.metadata();
    const int n = static_cast<int>(meta.orders.size()), M = meta.truncation_order;
    const int region = layer_at(meta, z);
    Vector S(n), U(n);
    if (region >= 0) {
        auto values = planar_layer(context, region, z - meta.layers[region].z_top);
        S = std::move(values.first); U = std::move(values.second);
    } else {
        const auto& mode = region == -1 ? state.modes.front() : state.modes.back();
        for (int j = 0; j < n; ++j) {
            const cplx p = region == -1 && j == M ? std::exp(kI * meta.kzI[M] * z) : cplx{};
            const cplx m = region == -1 ? state.reflected[j] * std::exp(-kI * meta.kzI[j] * z) : cplx{};
            const cplx transmitted = region == -2 ? state.transmitted[j] * std::exp(kI * meta.kzII[j] * (z - meta.z_bottom)) : cplx{};
            S[j] = p + m + transmitted;
            U[j] = mode.p[j] * (-p + m - transmitted);
        }
    }
    Harmonics out;
    for (auto& v : out) v.resize(n);
    if (meta.kind == rc::FieldSolveKind::PlanarTE) {
        out[1] = S;
        for (int j = 0; j < n; ++j) { out[3][j] = kI * U[j]; out[5][j] = -state.kx[j] * S[j]; }
    } else {
        out[4] = S;
        Vector rhs(n);
        for (int j = 0; j < n; ++j) { out[0][j] = -kI * U[j]; rhs[j] = state.kx[j] * S[j]; }
        if (region >= 0) out[2] = epsilon_solve(state.materials[region], rhs);
        else {
            const cplx eps = region == -1 ? meta.nI * meta.nI : meta.nII * meta.nII;
            for (int j = 0; j < n; ++j) out[2][j] = rhs[j] / eps;
        }
    }
    return out;
}

inline Vector combine(const Matrix& responses, cplx Es, cplx Ep) {
    Vector out(responses.rows());
    for (int j = 0; j < responses.rows(); ++j) out[j] = responses(j, 0) * Es + responses(j, 1) * Ep;
    return out;
}

inline std::pair<Vector, Vector> conical_layer(const rc::ConicalFieldContext& context, int layer,
                                               double d, double z, cplx Es, cplx Ep) {
    const auto& state = dt::ConicalFieldAccess::get(context);
    const auto& mode = state.modes[layer + 1];
    const double k0 = context.metadata().k0;
    Vector p = combine(state.a_plus[layer], Es, Ep), m = combine(state.a_minus[layer], Es, Ep);
    if (!mode.G.empty()) {
        p = dc::triangular_exp(mode.G * cplx(-k0 * z)) * p;
        m = dc::triangular_exp(mode.G * cplx(-k0 * (d - z))) * m;
    } else for (int j = 0; j < static_cast<int>(p.size()); ++j) {
        p[j] *= std::exp(-k0 * z * mode.q[j]);
        m[j] *= std::exp(-k0 * (d - z) * mode.q[j]);
    }
    Vector sum(p.size()), difference(p.size());
    for (int j = 0; j < static_cast<int>(p.size()); ++j) { sum[j] = p[j] + m[j]; difference[j] = -p[j] + m[j]; }
    if (!mode.analytic) return {mode.W * sum, mode.V * difference};
    for (int j = 0; j < static_cast<int>(p.size()); ++j) { sum[j] *= mode.w[j]; difference[j] *= mode.v[j]; }
    return {std::move(sum), std::move(difference)};
}

inline Harmonics conical(const rc::ConicalFieldContext& context, double z, cplx Es = 1.0, cplx Ep = 0.0) {
    const auto& state = dt::ConicalFieldAccess::get(context);
    const auto& meta = context.metadata();
    const double norm = std::sqrt(std::norm(Es) + std::norm(Ep));
    if (!(norm > 0.0) || !std::isfinite(norm)) throw std::invalid_argument("test reconstruction: invalid Jones input");
    Es /= norm; Ep /= norm;
    const int n = static_cast<int>(meta.orders.size()), M = meta.truncation_order;
    const int region = layer_at(meta, z);
    Vector F(2 * n), G(2 * n);
    if (region >= 0) {
        const rc::FieldLayerInfo* geometry = nullptr;
        for (const auto& info : meta.layers) if (info.effective_index == region) geometry = &info;
        const double local_z = (std::max)(0.0, (std::min)(geometry->thickness, z - geometry->z_top));
        auto values = conical_layer(context, region, geometry->thickness, local_z, Es, Ep);
        F = std::move(values.first); G = std::move(values.second);
    } else {
        const auto& mode = region == -1 ? state.modes.front() : state.modes.back();
        const Vector response = combine(region == -1 ? state.reflected : state.transmitted, Es, Ep);
        for (int j = 0; j < 2 * n; ++j) {
            const int order = j % n;
            const cplx incident = region == -1 && order == M ? (j < n ? Es : Ep) * std::exp(kI * meta.kzI[M] * z) : cplx{};
            const cplx reflection = region == -1 ? response[j] * std::exp(-kI * meta.kzI[order] * z) : cplx{};
            const cplx transmitted = region == -2 ? response[j] * std::exp(kI * meta.kzII[order] * (z - meta.z_bottom)) : cplx{};
            F[j] = mode.w[j] * (incident + reflection + transmitted);
            G[j] = mode.v[j] * (-incident + reflection - transmitted);
        }
    }
    Harmonics out;
    for (auto& v : out) v.resize(n);
    Vector rhs(n);
    for (int j = 0; j < n; ++j) {
        const double cs = state.cs[j], sn = state.sn[j];
        out[0][j] = -sn * F[j] + cs * F[n + j];
        out[1][j] = cs * F[j] + sn * F[n + j];
        const cplx hx = -cs * G[j] - sn * G[n + j], hy = -sn * G[j] + cs * G[n + j];
        out[3][j] = -kI * hx; out[4][j] = -kI * hy;
        out[5][j] = state.ky * out[0][j] - state.kx[j] * out[1][j];
        rhs[j] = kI * (state.ky * hx - state.kx[j] * hy);
    }
    if (region >= 0 && !state.materials[region].analytic) out[2] = state.materials[region].inv_eps * rhs;
    else {
        const cplx eps = region >= 0 ? state.materials[region].epsilon :
                         region == -1 ? meta.nI * meta.nI : meta.nII * meta.nII;
        for (int j = 0; j < n; ++j) out[2][j] = rhs[j] / eps;
    }
    return out;
}
}  // namespace field_context_test
