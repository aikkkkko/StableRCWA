// 可选近场快照：旧入口一致性、拥有数据的语义、模式/材料及稳定振幅锚定。
// --reference reference_v1.json 仅在测试内重建六复谐波，供冻结 Python 合同比较。
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

#include "field_context_test_access.hpp"
#include "stablercwa/json.hpp"

using namespace field_context_test;
static int checks = 0, failures = 0;
static void check(bool ok, const char* expr, int line) {
    ++checks;
    if (!ok) { ++failures; std::printf("FAIL line %d: %s\n", line, expr); }
}
#define CHECK(x) check(static_cast<bool>(x), #x, __LINE__)
static void close(cplx a, cplx b, double tolerance = 5e-11) { CHECK(std::abs(a - b) <= tolerance * (1.0 + std::abs(b))); }
static void close(const Vector& a, const Vector& b, double tolerance = 5e-11) {
    CHECK(a.size() == b.size());
    for (std::size_t j = 0; j < (std::min)(a.size(), b.size()); ++j) close(a[j], b[j], tolerance);
}
static double largest(const Vector& a) { double out = 0.0; for (cplx z : a) out = (std::max)(out, std::abs(z)); return out; }
static bool same(const rc::Result& a, const rc::Result& b) {
    return a.orders == b.orders && a.R == b.R && a.T == b.T && a.DEr == b.DEr && a.DEt == b.DEt &&
        a.n_eig == b.n_eig && a.n_dense_solve == b.n_dense_solve && a.eig_info == b.eig_info &&
        a.max_eig_residual == b.max_eig_residual && a.max_solve_backward_error == b.max_solve_backward_error;
}
static bool same(const rc::ConicalResult& a, const rc::ConicalResult& b) {
    if (a.orders != b.orders || a.n_eig != b.n_eig || a.n_dense_solve != b.n_dense_solve ||
        a.n_schur != b.n_schur || a.eig_info != b.eig_info || a.max_eig_residual != b.max_eig_residual ||
        a.max_solve_backward_error != b.max_solve_backward_error) return false;
    for (std::size_t j = 0; j < a.Jr.size(); ++j) {
        for (const auto& pair : {std::make_pair(a.Jr[j], b.Jr[j]), std::make_pair(a.Jt[j], b.Jt[j])})
            if (pair.first.ss != pair.second.ss || pair.first.sp != pair.second.sp ||
                pair.first.ps != pair.second.ps || pair.first.pp != pair.second.pp) return false;
        for (const auto& pair : {std::make_pair(a.DEr[j], b.DEr[j]), std::make_pair(a.DEt[j], b.DEt[j])})
            if (pair.first.ss != pair.second.ss || pair.first.sp != pair.second.sp ||
                pair.first.ps != pair.second.ps || pair.first.pp != pair.second.pp) return false;
    }
    return true;
}
static rc::Stack mixed_stack() {
    rc::Stack s;
    s.nI = 1.2; s.nII = 1.4; s.period = 0.6;
    s.layers = {rc::Layer::grating(0.2, {0.0, 0.21, 0.6}, {1.7, 1.0}),
                rc::Layer::uniform(0.09, cplx(1.46, 0.03)),
                rc::Layer::grating(0.13, {0.0, 0.12, 0.6}, {1.1, cplx(1.9, 0.07)})};
    return s;
}
static void planar_boundaries(const rc::PlanarFieldContext& context) {
    const auto& s = dt::PlanarFieldAccess::get(context);
    const auto& meta = context.metadata();
    const int n = static_cast<int>(meta.orders.size()), M = meta.truncation_order;
    Vector expected_S(n), expected_U(n);
    for (int j = 0; j < n; ++j) {
        const cplx incident = j == M ? 1.0 : 0.0;
        expected_S[j] = incident + s.reflected[j];
        expected_U[j] = s.modes.front().p[j] * (-incident + s.reflected[j]);
    }
    auto top = planar_layer(context, 0, 0.0);
    close(top.first, expected_S); close(top.second, expected_U);
    for (int j = 0; j < static_cast<int>(meta.layers.size()); ++j) {
        auto bottom = planar_layer(context, j, meta.layers[j].thickness);
        if (j + 1 < static_cast<int>(meta.layers.size())) {
            auto next = planar_layer(context, j + 1, 0.0);
            close(bottom.first, next.first); close(bottom.second, next.second);
        } else {
            for (int k = 0; k < n; ++k) expected_U[k] = -s.modes.back().p[k] * s.transmitted[k];
            close(bottom.first, s.transmitted); close(bottom.second, expected_U);
        }
    }
}
static void conical_boundaries(const rc::ConicalFieldContext& context) {
    const auto& s = dt::ConicalFieldAccess::get(context);
    const auto& meta = context.metadata();
    const int orders = static_cast<int>(meta.orders.size()), n = 2 * orders, M = meta.truncation_order;
    for (int col = 0; col < 2; ++col) {
        Vector F(n), G(n);
        for (int j = 0; j < n; ++j) {
            const cplx inc = j == col * orders + M ? 1.0 : 0.0;
            F[j] = s.modes.front().w[j] * (inc + s.reflected(j, col));
            G[j] = s.modes.front().v[j] * (-inc + s.reflected(j, col));
        }
        int effective = 0;
        for (const auto& info : meta.layers) {
            if (info.effective_index < 0) continue;
            auto top = conical_layer(context, effective, info.thickness, 0.0, col == 0 ? 1.0 : 0.0, col == 1 ? 1.0 : 0.0);
            close(top.first, F, 3e-10); close(top.second, G, 3e-10);
            auto bottom = conical_layer(context, effective, info.thickness, info.thickness, col == 0 ? 1.0 : 0.0, col == 1 ? 1.0 : 0.0);
            F = std::move(bottom.first); G = std::move(bottom.second);
            ++effective;
        }
        Vector transmitted_F(n), transmitted_G(n);
        for (int j = 0; j < n; ++j) {
            transmitted_F[j] = s.modes.back().w[j] * s.transmitted(j, col);
            transmitted_G[j] = -s.modes.back().v[j] * s.transmitted(j, col);
        }
        close(F, transmitted_F, 3e-10); close(G, transmitted_G, 3e-10);
    }
}

static void test_same_solve_and_materials() {
    const auto stack = mixed_stack();
    for (auto linear : {rc::LinearSolver::LUPartial, rc::LinearSolver::QRHouseholder}) {
        rc::Options opt; opt.diagnostics = true; opt.linear = linear;
        int calls = 0;
        opt.custom_eig = [&](const Matrix& a) { ++calls; return eig_qr(a); };
        for (auto pol : {rc::Polarization::TE, rc::Polarization::TM}) {
            calls = 0;
            const auto old = rc::solve(stack, 0.83, 27.0, 3, pol, opt);
            const int old_calls = calls; calls = 0;
            const auto out = rc::solve_with_fields(stack, 0.83, 27.0, 3, pol, opt);
            CHECK(out.fields.available()); CHECK(same(old, out.result)); CHECK(calls == old_calls && calls == out.result.n_eig);
            planar_boundaries(out.fields);
            const auto& material = dt::PlanarFieldAccess::get(out.fields).materials.front();
            if (pol == rc::Polarization::TM) {
                CHECK(material.factorized && material.eps_factors.method == linear);
                const Matrix eps = dt::convolution_matrix(stack.layers[0].edges, {1.7 * 1.7, 1.0}, stack.period, 3);
                Vector rhs(7); for (int j = 0; j < 7; ++j) rhs[j] = cplx(0.2 * j + 1.0, -0.13 * j);
                close(eps * epsilon_solve(material, rhs), rhs, 3e-13);
            } else CHECK(material.eps.rows() == 7 && !material.factorized);
        }
        calls = 0; const auto old = rc::solve_conical(stack, 0.83, 27.0, 31.0, 3, opt);
        const int old_calls = calls; calls = 0;
        const auto out = rc::solve_conical_with_fields(stack, 0.83, 27.0, 31.0, 3, opt);
        CHECK(out.fields.available()); CHECK(same(old, out.result)); CHECK(calls == old_calls && calls == out.result.n_eig);
        conical_boundaries(out.fields);
        const auto& material = dt::ConicalFieldAccess::get(out.fields).materials.front();
        Vector rhs(7, cplx(1.2, -0.7));
        const Matrix eps = dt::convolution_matrix(stack.layers[0].edges, {1.7 * 1.7, 1.0}, stack.period, 3);
        const Matrix inverse_eps = dt::convolution_matrix(stack.layers[0].edges, {1.0 / (1.7 * 1.7), 1.0}, stack.period, 3);
        close(eps * (material.inv_eps * rhs), rhs, 3e-13);
        close(inverse_eps * (material.c * rhs), rhs, 3e-13);
    }
    rc::Options dense; dense.homogeneous_fast_path = false; dense.diagnostics = true;
    for (auto pol : {rc::Polarization::TE, rc::Polarization::TM}) {
        const auto out = rc::solve_with_fields(stack, 0.83, 27.0, 2, pol, dense);
        CHECK(out.fields.available()); CHECK(same(rc::solve(stack, 0.83, 27.0, 2, pol, dense), out.result));
        planar_boundaries(out.fields);
    }
    const auto c = rc::solve_conical_with_fields(stack, 0.83, 27.0, 31.0, 2, dense);
    CHECK(c.fields.available()); CHECK(same(rc::solve_conical(stack, 0.83, 27.0, 31.0, 2, dense), c.result));
    conical_boundaries(c.fields);
}

static void test_independent_film_and_zero() {
    rc::Stack stack;
    stack.nI = 1.4; stack.nII = 0.6; stack.period = 0.62;
    stack.layers = {rc::Layer::uniform(0.41, cplx(1.8, 0.13))};
    for (auto pol : {rc::Polarization::TE, rc::Polarization::TM}) {
        const auto out = rc::solve_with_fields(stack, 0.83, 33.0, 2, pol);
        const auto& s = dt::PlanarFieldAccess::get(out.fields);
        const cplx kzI = std::sqrt(stack.nI * stack.nI - s.kx[2] * s.kx[2]);
        const cplx kzL = std::sqrt(stack.layers[0].n * stack.layers[0].n - s.kx[2] * s.kx[2]);
        const cplx kzII = std::sqrt(stack.nII * stack.nII - s.kx[2] * s.kx[2]);
        const cplx pI = pol == rc::Polarization::TE ? -kI * kzI : -kI * kzI / (stack.nI * stack.nI);
        const cplx pL = pol == rc::Polarization::TE ? -kI * kzL : -kI * kzL / (stack.layers[0].n * stack.layers[0].n);
        const cplx pII = pol == rc::Polarization::TE ? -kI * kzII : -kI * kzII / (stack.nII * stack.nII);
        const cplx r01 = (pI - pL) / (pI + pL), r12 = (pL - pII) / (pL + pII);
        const cplx phase = std::exp(kI * out.fields.metadata().k0 * kzL * stack.layers[0].d);
        const cplx A = 2.0 * pI / (pI + pL) / (1.0 + r01 * r12 * phase * phase);
        const cplx B_bottom = r12 * phase * A;
        close(s.a_plus[0][2], A, 4e-14); close(s.a_minus[0][2], B_bottom, 4e-14);
        CHECK(s.modes[1].analytic && s.modes[1].W.empty() && s.modes[1].V.empty());
        CHECK(s.materials[0].analytic && s.materials[0].eps.empty());
        planar_boundaries(out.fields);
    }
    const double kx = 2.0 * std::sin(30.0 * kPi / 180.0);
    const double d = 1.0 / (kPi * std::sqrt(4.0 - kx * kx));
    stack.nI = stack.nII = 2.0; stack.period = 1.0; stack.layers = {rc::Layer::uniform(d, kx)};
    for (auto pol : {rc::Polarization::TE, rc::Polarization::TM}) {
        const auto out = rc::solve_with_fields(stack, 1.0, 30.0, 0, pol);
        CHECK(out.fields.available()); CHECK(same(rc::solve(stack, 1.0, 30.0, 0, pol), out.result));
        CHECK(out.fields.metadata().layers[0].finite_zero_mode_count == 1);
        CHECK(out.fields.metadata().layers[0].propagation == rc::FieldPropagation::FiniteZero);
        const double ratio = pol == rc::Polarization::TE ? 2.0 : 0.5;
        const cplx R = -kI * ratio / (2.0 - kI * ratio), T = 2.0 / (2.0 - kI * ratio);
        close(out.result.R[0], R, 4e-15); close(out.result.T[0], T, 4e-15);
        for (double fraction : {0.0, 0.17, 0.5, 0.83, 1.0}) {
            const auto value = planar_layer(out.fields, 0, d * fraction);
            close(value.first[0], 1.0 + R * (1.0 - 2.0 * fraction), 4e-15);
            close(value.second[0], -2.0 * R / (out.fields.metadata().k0 * d) / (pol == rc::Polarization::TE ? 1.0 : kx * kx), 4e-15);
        }
        planar_boundaries(out.fields);
    }
}

static void test_schur_and_thick_absorption() {
    rc::Stack stack;
    stack.nI = 1.0; stack.nII = 1.46; stack.period = 0.574;
    stack.layers = {rc::Layer::grating(0.38, {0.0, 0.1722, 0.574}, {1.46, 1.0}), rc::Layer::uniform(0.115, 1.96)};
    rc::Options opt; int calls = 0;
    opt.custom_eig = [&](const Matrix& a) { ++calls; return eig_qr(a); };
    const auto out = rc::solve_conical_with_fields(stack, 0.980484673140678, 40.0, 25.0, 8, opt);
    CHECK(out.fields.available() && out.result.n_schur == 1); CHECK(calls == 2 && out.result.n_eig == 3);
    const auto& mode = dt::ConicalFieldAccess::get(out.fields).modes[1];
    CHECK(mode.G.rows() == 34 && !mode.analytic);
    double coupling = 0.0; for (int j = 0; j < 34; ++j) for (int i = 0; i < j; ++i) coupling = (std::max)(coupling, std::abs(mode.G(i, j)));
    CHECK(coupling > 1e-3); CHECK(out.fields.metadata().layers[0].propagation == rc::FieldPropagation::Schur);
    CHECK(out.fields.metadata().layers[1].propagation == rc::FieldPropagation::Exponential);
    CHECK(same(rc::solve_conical(stack, 0.980484673140678, 40.0, 25.0, 8, opt), out.result));
    conical_boundaries(out.fields);
    // 高吸收厚膜下行幅值下溢为零，上行场从底面锚点直接传播，禁止乘增长指数反推。
    stack = mixed_stack(); stack.layers[1] = rc::Layer::uniform(1000.0, cplx(1.8, 0.7));
    for (auto pol : {rc::Polarization::TE, rc::Polarization::TM}) {
        const auto thick = rc::solve_with_fields(stack, 0.83, 27.0, 3, pol);
        CHECK(thick.fields.available()); CHECK(same(rc::solve(stack, 0.83, 27.0, 3, pol), thick.result));
        CHECK(largest(dt::PlanarFieldAccess::get(thick.fields).a_minus[1]) == 0.0);
        for (const auto& f : planar(thick.fields, 500.2)) CHECK(dt::finite_vector(f));
        planar_boundaries(thick.fields);
    }
    const auto thick = rc::solve_conical_with_fields(stack, 0.83, 27.0, 31.0, 3);
    CHECK(thick.fields.available()); CHECK(same(rc::solve_conical(stack, 0.83, 27.0, 31.0, 3), thick.result));
    for (const auto& f : conical(thick.fields, 500.2, cplx(0.3, 0.2), cplx(-0.7, 0.5))) CHECK(dt::finite_vector(f));
    conical_boundaries(thick.fields);
}

static void test_empty_zero_mapping_ownership_and_failure() {
    rc::Stack stack; stack.nI = 1.3; stack.nII = 1.7; stack.period = 0.73;
    const auto p = rc::solve_with_fields(stack, 0.81, 23.0, 1);
    const auto c = rc::solve_conical_with_fields(stack, 0.81, 23.0, 31.0, 1);
    CHECK(p.fields.available() && p.fields.metadata().layers.empty()); CHECK(c.fields.available() && c.fields.metadata().layers.empty());
    CHECK(dt::PlanarFieldAccess::get(p.fields).modes.size() == 2 && dt::PlanarFieldAccess::get(p.fields).a_plus.empty());
    CHECK(dt::ConicalFieldAccess::get(c.fields).modes.size() == 2 && dt::ConicalFieldAccess::get(c.fields).a_plus.empty());
    for (const auto& f : planar(p.fields, -1000.0)) CHECK(dt::finite_vector(f));
    for (const auto& f : conical(c.fields, -1000.0)) CHECK(dt::finite_vector(f));
    const auto no_layers = rc::solve_conical(stack, 0.81, 23.0, 31.0, 1);
    stack.layers = {rc::Layer::uniform(0.0, 1.0), rc::Layer::grating(0.0, {0.0, 0.2, 0.73}, {1.0, 1.8})};
    const auto all_zero = rc::solve_conical_with_fields(stack, 0.81, 23.0, 31.0, 1);
    CHECK(all_zero.fields.available() && same(no_layers, all_zero.result));
    CHECK(all_zero.fields.metadata().layers.size() == 2 && dt::ConicalFieldAccess::get(all_zero.fields).a_plus.empty());
    CHECK(all_zero.fields.metadata().layers[0].effective_index == -1 && all_zero.fields.metadata().layers[1].effective_index == -1);
    // 输入导致物理波数不可表示时仍保留旧圆锥远场结果，但字段快照不能宣称可用。
    stack.layers.clear();
    const auto extreme = rc::solve_conical_with_fields(stack, 1e-320, 23.0, 31.0, 0);
    CHECK(same(rc::solve_conical(stack, 1e-320, 23.0, 31.0, 0), extreme.result));
    CHECK(extreme.fields.status() == rc::FieldContextStatus::Unavailable);
    // 未激发截止级可使旧纯膜标量递推含非有限数，快照应保存严格零的内部/外部振幅。
    stack.nI = stack.nII = 1.0; stack.period = 1.0; stack.layers = {rc::Layer::uniform(0.2, 1.0)};
    const auto unused_p = rc::solve_with_fields(stack, 1.0, 0.0, 1);
    const auto unused_c = rc::solve_conical_with_fields(stack, 1.0, 0.0, 37.0, 1);
    CHECK(unused_p.fields.available() && unused_c.fields.available());
    for (const auto& f : planar(unused_p.fields, 0.1)) CHECK(dt::finite_vector(f));
    for (const auto& f : conical(unused_c.fields, 0.1)) CHECK(dt::finite_vector(f));
    close(dt::PlanarFieldAccess::get(unused_p.fields).a_plus[0][0], 0.0, 0.0);
    close(dt::ConicalFieldAccess::get(unused_c.fields).a_plus[0](0, 0), 0.0, 0.0);
    const double cutoff = 2.0 * std::sin(30.0 * kPi / 180.0);
    stack.nI = stack.nII = 2.0; stack.layers = {rc::Layer::uniform(0.0, cutoff), rc::Layer::uniform(0.2, 1.6),
        rc::Layer::grating(0.0, {0.0, 0.3, 1.0}, {1.4, 1.0}), rc::Layer::uniform(0.1, 1.9)};
    const auto zero_p = rc::solve_with_fields(stack, 1.0, 30.0, 0);
    const auto zero_c = rc::solve_conical_with_fields(stack, 1.0, 30.0, 31.0, 0);
    CHECK(zero_p.fields.available() && zero_c.fields.available());
    const auto& pl = zero_p.fields.metadata().layers; const auto& cl = zero_c.fields.metadata().layers;
    CHECK(pl.size() == 4 && pl[0].effective_index == 0 && pl[0].has_modes && !pl[0].propagates);
    CHECK(cl.size() == 4 && cl[0].effective_index == -1 && !cl[0].has_modes && cl[1].effective_index == 0);
    CHECK(cl[2].original_index == 2 && cl[2].effective_index == -1 && cl[3].effective_index == 1);
    CHECK(zero_c.result.n_eig == 0); close(cl[3].z_top, 0.2, 0.0);
    planar_boundaries(zero_p.fields); conical_boundaries(zero_c.fields);
    rc::PlanarFieldContext owned_p; rc::ConicalFieldContext owned_c;
    {
        auto source = mixed_stack();
        owned_p = rc::solve_with_fields(source, 0.83, 27.0, 3, rc::Polarization::TM).fields;
        owned_c = rc::solve_conical_with_fields(source, 0.83, 27.0, 31.0, 3).fields;
        source.layers[0].n_values[0] = 5.0; source.layers.clear(); source.period = 4.0;
    }
    CHECK(owned_p.metadata().layers[0].n_values[0] == cplx(1.7)); CHECK(owned_c.metadata().period == 0.6);
    const auto before_p = planar(owned_p, 0.08), before_c = conical(owned_c, 0.08);
    auto copy_p = owned_p; auto copy_c = owned_c;
    CHECK(dt::PlanarFieldAccess::get(copy_p).modes[1].W.data() != dt::PlanarFieldAccess::get(owned_p).modes[1].W.data());
    CHECK(dt::ConicalFieldAccess::get(copy_c).modes[1].W.data() != dt::ConicalFieldAccess::get(owned_c).modes[1].W.data());
    auto moved_p = std::move(owned_p); auto moved_c = std::move(owned_c);
    CHECK(owned_p.status() == rc::FieldContextStatus::Empty && owned_p.metadata().layers.empty());
    CHECK(owned_c.status() == rc::FieldContextStatus::Empty && owned_c.metadata().layers.empty());
    for (int j = 0; j < 6; ++j) { close(planar(moved_p, 0.08)[j], before_p[j], 0.0); close(conical(moved_c, 0.08)[j], before_c[j], 0.0); }
    rc::Options bad; bad.custom_eig = [](const Matrix&) { EigResult e; e.info = 7; return e; };
    const auto failed_p = rc::solve_with_fields(mixed_stack(), 0.83, 27.0, 3, rc::Polarization::TE, bad);
    const auto failed_c = rc::solve_conical_with_fields(mixed_stack(), 0.83, 27.0, 31.0, 3, bad);
    CHECK(!failed_p.fields.available() && failed_p.fields.status() == rc::FieldContextStatus::Unavailable && failed_p.result.eig_info == 7);
    CHECK(!failed_c.fields.available() && failed_c.result.eig_info == 7);
    CHECK(std::isnan(failed_p.result.R[0].real()) && std::isnan(failed_c.result.Jr[0].ss.real()));
    bool rejected_p = false, rejected_c = false;
    try { (void)planar(failed_p.fields, 0.1); } catch (const std::logic_error&) { rejected_p = true; }
    try { (void)conical(failed_c.fields, 0.1); } catch (const std::logic_error&) { rejected_c = true; }
    CHECK(rejected_p && rejected_c);
    rc::PlanarFieldContext empty;
    bool rejected_empty = false;
    try { empty.require_available(); } catch (const std::logic_error&) { rejected_empty = true; }
    CHECK(rejected_empty && empty.status() == rc::FieldContextStatus::Empty);
    // 真正被耦合的圆锥 q==0 仍失败，零厚层不引入这个失败。
    stack.nI = stack.nII = 1.0; stack.layers = {rc::Layer::uniform(0.2, 1.0), rc::Layer::grating(0.1, {0.0, 0.3, 1.0}, {1.5, 1.0})};
    const auto cutoff_c = rc::solve_conical_with_fields(stack, 1.0, 0.0, 0.0, 1);
    CHECK(!cutoff_c.fields.available() && std::isnan(cutoff_c.result.Jr[0].ss.real()));
}

using JV = stablercwa::json::Value;
static cplx parse_complex(const JV& value) { return value.is_number() ? cplx(value.as_double()) : cplx(value.at(0).as_double(), value.at(1).as_double()); }
static JV encode_complex(cplx value) { JV out = JV::array(); out.push_back(value.real()); out.push_back(value.imag()); return out; }
static JV encode_vector(const Vector& values) { JV out = JV::array(); for (cplx z : values) out.push_back(encode_complex(z)); return out; }
static rc::Stack parse_stack(const JV& value) {
    rc::Stack s; s.nI = parse_complex(value.at("nI")); s.nII = parse_complex(value.at("nII")); s.period = value.at("period").as_double();
    for (std::size_t i = 0; i < value.at("layers").size(); ++i) {
        const auto& layer = value.at("layers").at(i);
        if (layer.contains("n")) s.layers.push_back(rc::Layer::uniform(layer.at("d").as_double(), parse_complex(layer.at("n"))));
        else {
            RealVector edges; Vector indices;
            for (std::size_t j = 0; j < layer.at("edges").size(); ++j) edges.push_back(layer.at("edges").at(j).as_double());
            for (std::size_t j = 0; j < layer.at("n_values").size(); ++j) indices.push_back(parse_complex(layer.at("n_values").at(j)));
            s.layers.push_back(rc::Layer::grating(layer.at("d").as_double(), std::move(edges), std::move(indices)));
        }
    }
    return s;
}
static int export_reference(const char* path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open reference manifest");
    const auto manifest = json::parse(std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()));
    JV output = JV::array();
    const auto& cases = manifest.at("cases");
    for (std::size_t i = 0; i < cases.size(); ++i) {
        const auto& parameters = cases.at(i).at("parameters");
        const auto stack = parse_stack(parameters.at("stack"));
        const double wavelength = parameters.at("wavelength").as_double(), theta = parameters.at("theta_deg").as_double();
        const int M = static_cast<int>(parameters.at("M").as_double());
        std::vector<Harmonics> samples;
        rc::FieldMetadata meta;
        if (parameters.at("mode").as_string() == "planar") {
            const auto pol = parameters.at("polarization").as_string() == "TE" ? rc::Polarization::TE : rc::Polarization::TM;
            const auto result = rc::solve_with_fields(stack, wavelength, theta, M, pol);
            result.fields.require_available(); meta = result.fields.metadata();
            for (std::size_t z = 0; z < parameters.at("zs").size(); ++z) samples.push_back(planar(result.fields, parameters.at("zs").at(z).as_double()));
        } else {
            const auto result = rc::solve_conical_with_fields(stack, wavelength, theta, parameters.at("phi_deg").as_double(), M);
            result.fields.require_available(); meta = result.fields.metadata();
            cplx Es, Ep;
            if (!parameters.at("jones_in").is_null()) { Es = parse_complex(parameters.at("jones_in").at(0)); Ep = parse_complex(parameters.at("jones_in").at(1)); }
            else { const double psi = parameters.at("psi_deg").as_double() * kPi / 180.0, delta = parameters.at("delta_deg").as_double() * kPi / 180.0;
                Es = std::sin(psi) * std::exp(kI * delta); Ep = std::cos(psi); }
            for (std::size_t z = 0; z < parameters.at("zs").size(); ++z) samples.push_back(conical(result.fields, parameters.at("zs").at(z).as_double(), Es, Ep));
        }
        JV item = JV::object(), components = JV::array();
        item.set("id", parameters.at("id"));
        item.set("kxm", encode_vector(meta.kxm)); item.set("kzI", encode_vector(meta.kzI)); item.set("kzII", encode_vector(meta.kzII)); item.set("ky", meta.ky);
        int zero_modes = 0, schur_layers = 0;
        for (const auto& layer : meta.layers) { zero_modes += layer.finite_zero_mode_count; schur_layers += layer.propagation == rc::FieldPropagation::Schur; }
        item.set("n_zero_modes", zero_modes); item.set("n_schur", schur_layers);
        JV amplitudes = JV::array(); amplitudes.push_back(meta.normalization.electric_incident_amplitude);
        amplitudes.push_back(meta.normalization.magnetic_incident_amplitude); item.set("incident_amplitudes", std::move(amplitudes));
        for (int component = 0; component < 6; ++component) { JV depths = JV::array(); for (const auto& sample : samples) depths.push_back(encode_vector(sample[component])); components.push_back(std::move(depths)); }
        item.set("harmonics", std::move(components)); output.push_back(std::move(item));
    }
    std::puts(json::dump(output).c_str());
    return 0;
}
int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--reference") return export_reference(argv[2]);
        test_same_solve_and_materials(); test_independent_film_and_zero(); test_schur_and_thick_absorption(); test_empty_zero_mapping_ownership_and_failure();
        std::printf("Field context: %d checks, %d failures\n", checks, failures);
        return failures ? 1 : 0;
    } catch (const std::exception& e) { std::printf("FAIL exception: %s\n", e.what()); return 1; }
}
