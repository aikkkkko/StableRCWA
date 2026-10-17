// 生产 field_at/field_map/phase_data 的公开接口验收。冻结数据导出不依赖测试内重建器。
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>

#include "stablercwa/rcwa1d_fields.hpp"
#include "stablercwa/rcwa1d_conical_fields.hpp"
#include "stablercwa/rcwa_phase.hpp"
#include "stablercwa/json.hpp"

using namespace stablercwa;
namespace rc = stablercwa::rcwa;
static int checks = 0, failures = 0;
static void check(bool ok, const char* expr, int line) {
    ++checks;
    if (!ok) { ++failures; std::printf("FAIL line %d: %s\n", line, expr); }
}
#define CHECK(x) check(static_cast<bool>(x), #x, __LINE__)
static void close(cplx a, cplx b, double tolerance = 5e-12) { CHECK(std::abs(a - b) <= tolerance * (1.0 + std::abs(b))); }
template<class F> static bool invalid(F f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; }
template<class F> static bool unavailable(F f) { try { f(); } catch (const std::logic_error&) { return true; } return false; }
template<class F> static bool numeric_failure(F f) { try { f(); } catch (const std::runtime_error&) { return true; } return false; }
static rc::Stack mixed_stack() {
    rc::Stack s; s.nI = 1.2; s.nII = 1.4; s.period = .6;
    s.layers = {rc::Layer::grating(.2, {0., .21, .6}, {1.7, 1.}), rc::Layer::uniform(.09, cplx(1.46, .03)),
                rc::Layer::grating(.13, {0., .12, .6}, {1.1, cplx(1.9, .07)})};
    return s;
}
static rc::Stack schur_stack() {
    rc::Stack s; s.nII = 1.46; s.period = .574;
    s.layers = {rc::Layer::grating(.38, {0., .1722, .574}, {1.46, 1.}), rc::Layer::uniform(.115, 1.96)};
    return s;
}
static void close(const rc::HarmonicFields& a, const rc::HarmonicFields& b, double tol = 5e-12) {
    CHECK(a.orders == b.orders && a.z == b.z);
    for (std::size_t c = 0; c < 6; ++c) for (std::size_t iz = 0; iz < a.nz(); ++iz)
        for (std::size_t j = 0; j < a.norders(); ++j) close(a.components[c](j, iz), b.components[c](j, iz), tol);
}
static void check_map(const rc::HarmonicFields& h, const rc::FieldGridXZ& g, double tol = 5e-12) {
    CHECK(h.z == g.z && h.nz() == g.nz());
    for (std::size_t c = 0; c < 6; ++c) {
        CHECK(g.components[c].rows() == static_cast<int>(g.nx()) && g.components[c].cols() == static_cast<int>(g.nz()));
        for (std::size_t iz = 0; iz < g.nz(); ++iz) for (std::size_t ix = 0; ix < g.nx(); ++ix) {
            cplx expected{};
            for (std::size_t j = 0; j < h.norders(); ++j)
                expected += h.components[c](j, iz) * std::exp(-kI * (h.metadata.kxm[j] * g.x[ix] + h.metadata.ky * g.y));
            close(g.at(static_cast<rc::FieldComponent>(c), iz, ix), expected, tol);
            close(g.components[c].data()[iz * g.nx() + ix], expected, tol);
        }
    }
}
static void test_sampling_order_interfaces_and_ownership() {
    const RealVector zs{.083, -.12, .42, .2, .2+5e-13, .42+5e-13, .083, .245, 0., 100., -100.};
    for (auto pol : {rc::Polarization::TE, rc::Polarization::TM}) {
        rc::Options options; int calls = 0;
        options.custom_eig = [&](const Matrix& a) { ++calls; return eig_qr(a); };
        auto result = rc::solve_with_fields(mixed_stack(), .83, 27., 3, pol, options);
        const int initial = calls;
        const auto h = rc::field_at(result, zs); const auto g = rc::field_map(result.fields, zs, 11);
        CHECK(h.z == zs && g.z == zs && calls == initial);
        CHECK(h.component(rc::FieldComponent::Ey).rows() == 7);
        for (std::size_t ix = 0; ix < g.nx(); ++ix) close(g.x[ix], static_cast<double>(ix) * .6 / 11., 1e-16);
        check_map(h, g);
        for (std::size_t iz = 0; iz < zs.size(); ++iz) {
            const auto one = rc::field_at(result.fields, zs[iz]);
            for (std::size_t c = 0; c < 6; ++c) for (int j = 0; j < 7; ++j) close(h.components[c](j, iz), one.components[c](j, 0));
        }
        for (std::size_t c = 0; c < 6; ++c) for (int j = 0; j < 7; ++j) {
            close(h.components[c](j, 0), h.components[c](j, 6), 0.);
            close(h.components[c](j, 3), h.components[c](j, 4), 0.);  // 内界面上层 + clip
        }
        for (double surface : {0., .2, .29, .42}) {
            const auto sides = rc::field_at(result, RealVector{surface-1e-9, surface+1e-9});
            for (int c : {0, 1, 3, 4}) for (int j = 0; j < 7; ++j) close(sides.components[c](j, 0), sides.components[c](j, 1), 3e-7);
        }
        auto copy = result.fields;
        auto moved = std::move(result.fields);
        close(rc::field_at(copy, zs), rc::field_at(moved, zs), 0.);
        CHECK(unavailable([&] { (void)rc::field_at(result.fields, RealVector{}); }));
        const auto empty = rc::field_at(copy, RealVector{}); const auto eg = rc::field_map(copy, RealVector{}, 7);
        CHECK(empty.nz() == 0 && empty.norders() == 7 && eg.nx() == 7 && eg.nz() == 0);
        for (const auto& component : eg.components) CHECK(component.rows() == 7 && component.cols() == 0 && component.empty());
        CHECK(rc::phase_data(eg, rc::FieldComponent::Ey).phase.empty());
        CHECK(calls == initial);
        CHECK(invalid([&] { (void)rc::field_at(copy, std::numeric_limits<double>::infinity()); }));
        CHECK(invalid([&] { (void)rc::field_at(copy, RealVector{0., std::numeric_limits<double>::quiet_NaN()}); }));
        CHECK(invalid([&] { (void)rc::field_map(copy, zs, 0); }));
        CHECK(invalid([&] { (void)rc::field_map(copy, zs, -1); }));
        CHECK(invalid([&] { (void)rc::field_map(copy, zs, 3, std::numeric_limits<double>::infinity()); }));
    }
    const auto result = rc::solve_conical_with_fields(mixed_stack(), .83, 27., 31., 3);
    const rc::JonesVector jones{{cplx(.3,.2),cplx(-.7,.5)}};
    const auto h = rc::field_at(result, zs, jones); const auto g = rc::field_map(result, zs, 11, .11, jones);
    check_map(h, g);
    for (double surface : {0., .2, .29, .42}) {
        const auto sides = rc::field_at(result, RealVector{surface-1e-9, surface+1e-9}, jones);
        for (int c : {0, 1, 3, 4}) for (int j = 0; j < 7; ++j) close(sides.components[c](j, 0), sides.components[c](j, 1), 3e-7);
    }
    const cplx carrier = std::exp(kI * .73);
    const auto shifted = rc::field_at(result, zs, {jones[0]*carrier, jones[1]*carrier});
    for (std::size_t c = 0; c < 6; ++c) for (std::size_t iz = 0; iz < zs.size(); ++iz) for (int j = 0; j < 7; ++j)
        close(shifted.components[c](j, iz), carrier*h.components[c](j, iz), 2e-12);
    CHECK(invalid([&] { (void)rc::field_at(result, zs, {{0.,0.}}); }));
    CHECK(invalid([&] { (void)rc::field_at(result, zs, {{std::numeric_limits<double>::infinity(),0.}}); }));
    CHECK(rc::field_map(result, RealVector{}, 3).nz() == 0);
    // 多右端 Eps 的 QR/LU 保留分解与关闭解析模式快速路径，都通过真正的采样器检验。
    const RealVector material_z{-.02,0.,.07,.2,.245,.36,.51};
    for(auto pol:{rc::Polarization::TE,rc::Polarization::TM}) {
        const auto ordinary=rc::solve_with_fields(mixed_stack(),.83,27.,2,pol);
        for(auto linear:{rc::LinearSolver::LUPartial,rc::LinearSolver::QRHouseholder}) {
            rc::Options option;option.linear=linear;option.homogeneous_fast_path=false;
            const auto dense=rc::solve_with_fields(mixed_stack(),.83,27.,2,pol,option);
            close(rc::field_at(dense,material_z),rc::field_at(ordinary,material_z),8e-12);
        }
        // 无主元求解在任意模基上本就可能病态；用良态标量检验保留的无主元 Eps 分解分支。
        const auto scalar=rc::solve_with_fields(mixed_stack(),.83,27.,0,pol);
        rc::Options no_pivot;no_pivot.linear=rc::LinearSolver::LUNoPivot;
        const auto nonpivot=rc::solve_with_fields(mixed_stack(),.83,27.,0,pol,no_pivot);
        close(rc::field_at(nonpivot,material_z),rc::field_at(scalar,material_z),8e-12);
    }
    rc::Options dense_option;dense_option.homogeneous_fast_path=false;dense_option.linear=rc::LinearSolver::QRHouseholder;
    const auto dense_conical=rc::solve_conical_with_fields(mixed_stack(),.83,27.,31.,3,dense_option);
    close(rc::field_at(dense_conical,material_z,jones),rc::field_at(result,material_z,jones),8e-12);
}

// 独立 Fresnel/薄膜解析验证；普通传播波的磁场符号、TM 归一化和纵向材料选侧均在六场中检验。
static void test_analytic_film_and_zero() {
    rc::Stack s; s.nI = 1.4; s.nII = .6; s.period = .62;
    const cplx nl{1.8,.13}; const double d = .41, wavelength = .83, theta = 33., k0 = 2*kPi/wavelength;
    const double kt = s.nI.real()*std::sin(theta*kPi/180.);
    const cplx ki = std::sqrt(s.nI*s.nI-kt*kt), kl = std::sqrt(nl*nl-kt*kt), ko = std::sqrt(s.nII*s.nII-kt*kt);
    s.layers = {rc::Layer::uniform(d,nl)};
    const RealVector zs{-.09,0.,.07,.205,d,.55,.73};
    for (auto pol : {rc::Polarization::TE,rc::Polarization::TM}) {
        const bool te = pol == rc::Polarization::TE;
        const cplx pi = ki/(te ? cplx(1.) : s.nI*s.nI), pl = kl/(te ? cplx(1.) : nl*nl), po = ko/(te ? cplx(1.) : s.nII*s.nII);
        const cplx r01 = (pi-pl)/(pi+pl), r12 = (pl-po)/(pl+po), e = std::exp(kI*k0*kl*d);
        const cplx a = (2.*pi/(pi+pl))/(1.+r01*r12*e*e), b = r12*a*e*e;
        const cplx r = a+b-1., t = a*e+b/e;
        const auto result = rc::solve_with_fields(s,wavelength,theta,2,pol);
        const auto values = rc::field_at(result,zs);
        for (std::size_t iz = 0; iz < zs.size(); ++iz) {
            const double z = zs[iz]; cplx p,m,n;
            if (z<0.) { n=s.nI; p=std::exp(kI*k0*ki*z); m=r*std::exp(-kI*k0*ki*z); }
            else if (z>d) { n=s.nII; p=t*std::exp(kI*k0*ko*(z-d)); }
            else { n=nl; p=a*std::exp(kI*k0*kl*z); m=b*std::exp(-kI*k0*kl*z); }
            const cplx adm = z<0. ? pi : z>d ? po : pl, f=p+m, u=kI*adm*(p-m);
            const std::array<cplx,6> expected = te ? std::array<cplx,6>{{0.,f,0.,kI*u,0.,-kt*f}} :
                std::array<cplx,6>{{-kI*u,0.,kt*f/(n*n),0.,f,0.}};
            for (int c=0;c<6;++c) for (int j=0;j<5;++j) close(values.components[c](j,iz),j==2 ? expected[c] : cplx{},4e-14);
        }
    }
    const double kx = 2.*std::sin(30.*kPi/180.), dz = 1./(kPi*std::sqrt(4.-kx*kx));
    s.nI=s.nII=2.; s.period=1.; s.layers={rc::Layer::uniform(dz,kx)};
    for (auto pol : {rc::Polarization::TE,rc::Polarization::TM}) {
        const bool te=pol==rc::Polarization::TE; const double ratio=te?2.:.5;
        const cplx r=-kI*ratio/(2.-kI*ratio);
        const auto result=rc::solve_with_fields(s,1.,30.,0,pol);
        CHECK(result.fields.metadata().layers[0].finite_zero_mode_count==1);
        for (double f : {0.,.17,.5,.83,1.}) {
            const auto value=rc::field_at(result,dz*f); const cplx expected=1.+r*(1.-2.*f);
            const cplx u=-2.*r/(2.*kPi*dz)/(te?1.:kx*kx);
            close(value.at(te?rc::FieldComponent::Ey:rc::FieldComponent::Hy,0,0),expected,5e-15);
            close(value.at(te?rc::FieldComponent::Hx:rc::FieldComponent::Ex,0,0),(te?kI:-kI)*u,5e-15);
            if (!te) close(value.at(rc::FieldComponent::Ez,0,0),expected/kx,5e-15);
        }
    }
    // h>2^53 时 (1+h)-h 会丢掉底面的顶锚点常量项；使用相对误差检验这个微小透射场。
    s.layers={rc::Layer::uniform(2e16,kx)};
    for(auto pol:{rc::Polarization::TE,rc::Polarization::TM}) {
        const auto result=rc::solve_with_fields(s,1.,30.,0,pol);CHECK(result.fields.available());
        const auto bottom=rc::field_at(result,2e16);
        const cplx field=bottom.at(pol==rc::Polarization::TE?rc::FieldComponent::Ey:rc::FieldComponent::Hy,0,0);
        CHECK(std::abs(field-result.result.T[0]) <= 2e-14*std::abs(result.result.T[0]));
    }
}

static void test_empty_extension_zero_thickness_phi0_and_failure() {
    rc::Stack s; s.nI=1.3; s.nII=1.7; s.period=.73;
    for (auto pol : {rc::Polarization::TE,rc::Polarization::TM}) {
        const auto result=rc::solve_with_fields(s,.81,23.,1,pol);
        const auto point=rc::field_at(result,0.), below=rc::field_at(result,1e-13);
        for(int c=0;c<6;++c)for(int j=0;j<3;++j)close(point.components[c](j,0),below.components[c](j,0),4e-12);
        if(pol==rc::Polarization::TM)close(point.at(rc::FieldComponent::Ez,0,1),result.fields.metadata().kxm[1]/result.fields.metadata().k0*result.result.T[1]/(s.nII*s.nII),2e-15);
        CHECK(numeric_failure([&]{(void)rc::field_at(result,-1e308);}));
        auto only_zero=s;only_zero.layers={rc::Layer::uniform(0.,2.6)};
        const auto zerop=rc::solve_with_fields(only_zero,.81,23.,1,pol);
        close(rc::field_at(zerop,0.),point,2e-15);
    }
    const auto no_layers=rc::solve_conical_with_fields(s,.81,23.,31.,1);
    CHECK(numeric_failure([&]{(void)rc::field_at(no_layers,-1e308);}));
    CHECK(numeric_failure([&]{(void)rc::field_map(no_layers,RealVector{.1},3,1e308);}));
    CHECK(rc::field_map(no_layers,RealVector{},3,1e308).nz()==0);  // 空网格不求无用 carrier
    const auto expected=rc::field_at(no_layers,0.,{{.3,kI}});
    s.layers={rc::Layer::uniform(0.,1.),rc::Layer::grating(0.,{0.,.2,.73},{1.,1.8})};
    const auto zeros=rc::solve_conical_with_fields(s,.81,23.,31.,1);
    CHECK(zeros.fields.metadata().layers.size()==2 && zeros.fields.metadata().layers[0].effective_index==-1);
    close(rc::field_at(zeros,0.,{{.3,kI}}),expected,0.);
    s=mixed_stack(); const RealVector zs{-.12,0.,.083,.2,.245,.42,.51};
    for(auto pol : {rc::Polarization::TE,rc::Polarization::TM}) {
        const auto p=rc::solve_with_fields(s,.83,27.,3,pol);
        const auto c=rc::solve_conical_with_fields(s,.83,27.,0.,3);
        const auto ph=rc::field_at(p,zs), ch=rc::field_at(c,zs,pol==rc::Polarization::TE?rc::JonesVector{{1.,0.}}:rc::JonesVector{{0.,1.}});
        const double factor=pol==rc::Polarization::TE?1.:s.nI.real();
        for(int comp=0;comp<6;++comp)for(std::size_t iz=0;iz<zs.size();++iz)for(int j=0;j<7;++j)close(ch.components[comp](j,iz),factor*ph.components[comp](j,iz),4e-12);
    }
    // 圆锥零厚输入跳过，不进入采样层；正厚层使用原始/有效几何索引。
    const auto original=rc::solve_conical_with_fields(s,.83,27.,31.,3);
    s.layers.insert(s.layers.begin(),rc::Layer::uniform(0.,4.)); s.layers.insert(s.layers.begin()+2,rc::Layer::uniform(0.,2.));
    const auto zero_added=rc::solve_conical_with_fields(s,.83,27.,31.,3);
    close(rc::field_at(original,zs),rc::field_at(zero_added,zs),0.);
    CHECK(zero_added.fields.metadata().layers[1].original_index==1 && zero_added.fields.metadata().layers[1].effective_index==0);
    rc::Options bad;bad.custom_eig=[](const Matrix&){EigResult e;e.info=7;return e;};
    const auto failed=rc::solve_with_fields(mixed_stack(),.83,27.,3,rc::Polarization::TE,bad);
    const auto failedc=rc::solve_conical_with_fields(mixed_stack(),.83,27.,31.,3,bad);
    CHECK(unavailable([&]{(void)rc::field_map(failed,zs,3);}));
    CHECK(unavailable([&]{(void)rc::field_at(failedc,RealVector{});}));
    CHECK(unavailable([]{(void)rc::field_at(rc::PlanarFieldContext{},0.);}));
    s.nI=s.nII=1.;s.period=1.;s.layers={rc::Layer::uniform(.2,1.),rc::Layer::grating(.1,{0.,.3,1.},{1.5,1.})};
    const auto cutoff=rc::solve_conical_with_fields(s,1.,0.,0.,1);
    CHECK(!cutoff.fields.available() && unavailable([&]{(void)rc::field_at(cutoff,0.1);}));
}

static void test_schur_and_thick_absorption() {
    rc::Options options;int calls=0;options.custom_eig=[&](const Matrix&a){++calls;return eig_qr(a);};
    const auto result=rc::solve_conical_with_fields(schur_stack(),.980484673140678,40.,25.,8,options);
    CHECK(result.fields.available() && result.result.n_schur==1); const int initial=calls;
    const rc::JonesVector jones{{cplx(.3,.2),cplx(-.7,.5)}};
    RealVector zs(257); for(std::size_t j=0;j<zs.size();++j)zs[j]=.38*static_cast<double>(j)/256.;
    const auto h=rc::field_at(result,zs,jones); const auto grid=rc::field_map(result,zs,37,.11,jones);
    check_map(h,grid,4e-11);
    // 深度批次的步进传播与各点直接从两端锚点传播一致。
    for(std::size_t iz=0;iz<zs.size();iz+=13) {
        const auto direct=rc::field_at(result,zs[iz],jones);
        for(int c=0;c<6;++c)for(int j=0;j<17;++j)close(h.components[c](j,iz),direct.components[c](j,0),2e-11);
    }
    const RealVector arbitrary{.38,.19,0.,.071,.19,.003,.38,.277};
    const auto a=rc::field_at(result,arbitrary,jones);
    for(std::size_t iz=0;iz<arbitrary.size();++iz) {
        const auto direct=rc::field_at(result,arbitrary[iz],jones);
        for(int c=0;c<6;++c)for(int j=0;j<17;++j)close(a.components[c](j,iz),direct.components[c](j,0),2e-11);
    }
    CHECK(calls==initial);
    auto s=mixed_stack();s.layers[1]=rc::Layer::uniform(1000.,cplx(1.8,.7));
    const RealVector thick_z{-1000.,0.,.2,100.2,500.2,1000.29,1000.42,2000.};
    for(auto pol:{rc::Polarization::TE,rc::Polarization::TM}) {
        const auto p=rc::solve_with_fields(s,.83,27.,3,pol);CHECK(p.fields.available());
        for(const auto& f:rc::field_map(p,thick_z,7).components)CHECK(f.all_finite());
    }
    const auto c=rc::solve_conical_with_fields(s,.83,27.,31.,3);CHECK(c.fields.available());
    for(const auto& f:rc::field_map(c,thick_z,7,.11,jones).components)CHECK(f.all_finite());
    // 人工 Jordan 耦合测试包含非零底锚点，阻止只用 diag(G) 的错误实现。
    Matrix generator{{2.,.75},{0.,2.}};const Vector anchor{cplx(1.,1.),.3};
    const RealVector distance{.8,0.,.3,.3,1.};
    const auto action=rc::detail::field_schur_action(generator,anchor,distance,1.);
    for(std::size_t j=0;j<distance.size();++j) {
        close(action(0,j),(anchor[0]-.75*distance[j]*anchor[1])*std::exp(-2.*distance[j]),4e-15);
        close(action(1,j),anchor[1]*std::exp(-2.*distance[j]),4e-15);
    }
    // 绝对深度不能放宽步长复用容差，否则约1e-7的步差经大波数后会产生约0.1rad误差。
    Matrix high_frequency(1,1);high_frequency(0,0)=-kI*1e6;
    const RealVector far_distance{1e8,1e8+1.,1e8+2.0000001};
    const auto exact=rc::detail::field_schur_action(high_frequency,Vector{1.},far_distance,1.);
    for(std::size_t j=0;j<far_distance.size();++j)close(exact(0,j),std::exp(-far_distance[j]*high_frequency(0,0)),0.);
    Matrix coupled_frequency{{-kI*1e6,1e-8},{0.,-kI*1e6}};
    const auto coupled=rc::detail::field_schur_action(coupled_frequency,Vector{0.,1.},far_distance,1.);
    for(std::size_t j=0;j<far_distance.size();++j) {
        const auto direct=rc::detail::conical::triangular_exp(coupled_frequency*cplx(-far_distance[j]))*Vector{0.,1.};
        close(coupled(0,j),direct[0],0.);close(coupled(1,j),direct[1],0.);
    }
}

static rc::FieldGridXZ synthetic(std::size_t nx,std::size_t nz) {
    rc::FieldGridXZ grid;grid.x.resize(nx);grid.z.resize(nz);
    for(std::size_t j=0;j<nx;++j)grid.x[j]=.13*j;
    for(std::size_t j=0;j<nz;++j)grid.z[j]=.17*j;
    grid.metadata.orders={0};grid.metadata.kxm={cplx(2.3,.8)};grid.metadata.kzI={cplx(3.1,.5)};grid.metadata.ky=.7;grid.y=.11;
    for(auto& f:grid.components)f=Matrix(static_cast<int>(nx),static_cast<int>(nz));
    return grid;
}
static void test_phase_contract() {
    auto grid=synthetic(4,2);auto& f=grid.component(rc::FieldComponent::Ey);
    const auto carrier=rc::incident_reference(grid);
    for(std::size_t j=0;j<carrier.size();++j){f.data()[j]=2.*std::exp(kI*.37)*carrier[j];close(std::abs(carrier[j]),1.,2e-16);}
    rc::PhaseOptions options;options.reference=rc::PhaseReference::Incident;
    const auto incident=rc::phase_data(grid,"ey",options);
    for(std::size_t j=0;j<carrier.size();++j){close(incident.phase[j],.37,2e-15);close(incident.amplitude[j],2.,2e-15);CHECK(!incident.mask[j]);}
    options.reference=rc::PhaseReference::Custom;options.custom_reference=Matrix(1,1,cplx(0.,3.));
    const auto custom=rc::phase_data(grid,rc::FieldComponent::Ey,options);
    for(std::size_t j=0;j<carrier.size();++j){close(custom.reference[j],cplx(0.,3.),0.);close(std::exp(kI*custom.phase[j]),f.data()[j]/std::abs(f.data()[j])*-kI,3e-15);}
    options.custom_reference=Matrix(4,1,1.);options.custom_reference(2,0)=0.;
    const auto row=rc::phase_data(grid,rc::FieldComponent::Ey,options);CHECK(row.mask[2]&&row.mask[6]&&!row.mask[1]);
    options.custom_reference=Matrix(1,2,1.);options.custom_reference(0,1)=std::numeric_limits<double>::infinity();
    const auto column=rc::phase_data(grid,rc::FieldComponent::Ey,options);CHECK(!column.mask[0]&&column.mask[4]&&column.mask[7]);
    options.custom_reference=Matrix(3,3,1.);CHECK(invalid([&]{(void)rc::phase_data(grid,rc::FieldComponent::Ey,options);}));
    options={};options.phase_floor=.25;f(0,0)=1.;f(1,0)=.25;f(2,0)=.2500001;f(3,0)=0.;
    for(int j=0;j<4;++j)f(j,1)=std::numeric_limits<double>::quiet_NaN();
    const auto floor=rc::phase_data(grid,rc::FieldComponent::Ey,options);
    CHECK(!floor.mask[0]&&floor.mask[1]&&!floor.mask[2]&&floor.mask[3]&&floor.mask[4]);
    const auto zero=rc::phase_data(grid,rc::FieldComponent::Ez);for(std::size_t j=0;j<zero.mask.size();++j)CHECK(zero.mask[j]&&std::isnan(zero.phase[j]));
    options.phase_floor=0.;const auto zero_floor=rc::phase_data(grid,rc::FieldComponent::Ez,options);for(bool b:zero_floor.mask)CHECK(b);
    options.phase_floor=-1.;CHECK(invalid([&]{(void)rc::phase_data(grid,rc::FieldComponent::Ey,options);}));
    options.phase_floor=std::numeric_limits<double>::quiet_NaN();CHECK(invalid([&]{(void)rc::phase_data(grid,rc::FieldComponent::Ey,options);}));
    CHECK(invalid([&]{(void)rc::phase_data(grid,"E2");}));CHECK(invalid([&]{(void)rc::phase_data(grid," H2 ");}));
    auto vector=synthetic(2,1);vector.components[0]=Matrix(2,1,cplx(1.,2.));vector.components[1]=Matrix(2,1,cplx(3.,-.5));
    const std::array<cplx,3> projection{{cplx(2.,1.),cplx(-.3,.7),0.}};
    const auto projected=rc::phase_data(vector,rc::FieldVectorFamily::E,projection);
    const cplx expected=std::conj(projection[0])*cplx(1.,2.)+std::conj(projection[1])*cplx(3.,-.5);
    close(projected.field[0],expected,2e-16);close(projected.amplitude[0],std::abs(expected),2e-16);
    CHECK(invalid([&]{(void)rc::phase_data(vector,rc::FieldVectorFamily::E,{{0.,0.,0.}});}));
    auto line=synthetic(7,1);const RealVector angles{2.9,-2.9,-2.6,0.,2.8,-2.8,-2.5};
    for(int j=0;j<7;++j)line.components[1](j,0)=j==3?cplx{}:std::exp(kI*angles[j]);
    options={};options.unwrap=rc::PhaseUnwrap::X;
    const auto unwrap=rc::phase_data(line,rc::FieldComponent::Ey,options);
    CHECK(unwrap.mask[3]);close(unwrap.phase[1],-2.9+2.*kPi,2e-15);close(unwrap.phase[4],2.8,2e-15);close(unwrap.phase[5],-2.8+2.*kPi,2e-15);
}

using JV=stablercwa::json::Value;
static cplx parse_complex(const JV& value) {
    if(value.is_null())return std::numeric_limits<double>::quiet_NaN();
    if(value.is_number())return value.as_double();
    const double a=value.at(0).is_null()?std::numeric_limits<double>::quiet_NaN():value.at(0).as_double();
    const double b=value.at(1).is_null()?std::numeric_limits<double>::quiet_NaN():value.at(1).as_double();
    return {a,b};
}
static JV encode_complex(cplx value){JV out=JV::array();out.push_back(value.real());out.push_back(value.imag());return out;}
static JV encode_vector(const Vector& values){JV out=JV::array();for(cplx z:values)out.push_back(encode_complex(z));return out;}
static JV encode_real(const RealVector& values){JV out=JV::array();for(double v:values)out.push_back(v);return out;}
static RealVector parse_real(const JV& values){RealVector out;for(std::size_t j=0;j<values.size();++j)out.push_back(values.at(j).as_double());return out;}
static rc::Stack parse_stack(const JV& value){
    rc::Stack s;s.nI=parse_complex(value.at("nI"));s.nII=parse_complex(value.at("nII"));s.period=value.at("period").as_double();
    for(std::size_t j=0;j<value.at("layers").size();++j){const auto&layer=value.at("layers").at(j);
        if(layer.contains("n"))s.layers.push_back(rc::Layer::uniform(layer.at("d").as_double(),parse_complex(layer.at("n"))));
        else{RealVector edges=parse_real(layer.at("edges"));Vector indices;
            for(std::size_t k=0;k<layer.at("n_values").size();++k)indices.push_back(parse_complex(layer.at("n_values").at(k)));
            s.layers.push_back(rc::Layer::grating(layer.at("d").as_double(),std::move(edges),std::move(indices)));}}
    return s;
}
static JV encode_fields(const std::array<Matrix,6>& fields){
    JV all=JV::array();for(const auto&f:fields){JV depths=JV::array();for(int iz=0;iz<f.cols();++iz)depths.push_back(encode_vector(f.column(iz)));all.push_back(std::move(depths));}return all;
}
static JV encode_phase(const rc::PhaseGrid& phase,bool mask){
    JV out=JV::array();for(std::size_t iz=0;iz<phase.nz;++iz){JV row=JV::array();for(std::size_t ix=0;ix<phase.nx;++ix){
        if(mask)row.push_back(static_cast<bool>(phase.mask[iz*phase.nx+ix]));else row.push_back(phase.phase[iz*phase.nx+ix]);}out.push_back(std::move(row));}return out;
}
static JV reference_case(const JV& parameters){
    const auto stack=parse_stack(parameters.at("stack"));const auto zs=parse_real(parameters.at("zs"));
    const double wavelength=parameters.at("wavelength").as_double(),theta=parameters.at("theta_deg").as_double();
    const int M=static_cast<int>(parameters.at("M").as_double()),nx=static_cast<int>(parameters.at("nx").as_double());
    const double y=parameters.at("y").as_double();rc::HarmonicFields harmonics;rc::FieldGridXZ grid;
    if(parameters.at("mode").as_string()=="planar"){
        const auto pol=parameters.at("polarization").as_string()=="TE"?rc::Polarization::TE:rc::Polarization::TM;
        const auto result=rc::solve_with_fields(stack,wavelength,theta,M,pol);
        harmonics=rc::field_at(result,zs);grid=rc::field_map(result,zs,nx,y);
    }else{
        const auto result=rc::solve_conical_with_fields(stack,wavelength,theta,parameters.at("phi_deg").as_double(),M);
        rc::JonesVector jones;
        if(!parameters.at("jones_in").is_null())jones={{parse_complex(parameters.at("jones_in").at(0)),parse_complex(parameters.at("jones_in").at(1))}};
        else jones=rc::jones_from_angles(parameters.at("psi_deg").as_double(),parameters.at("delta_deg").as_double());
        harmonics=rc::field_at(result,zs,jones);grid=rc::field_map(result,zs,nx,y,jones);
    }
    const auto&meta=grid.metadata;JV item=JV::object();item.set("id",parameters.at("id"));
    item.set("kxm",encode_vector(meta.kxm));item.set("kzI",encode_vector(meta.kzI));item.set("kzII",encode_vector(meta.kzII));item.set("ky",meta.ky);
    item.set("x",encode_real(grid.x));item.set("z",encode_real(grid.z));item.set("y",grid.y);
    int zeros=0,schur=0;for(const auto&l:meta.layers){zeros+=l.finite_zero_mode_count;schur+=l.propagation==rc::FieldPropagation::Schur;}
    item.set("n_zero_modes",zeros);item.set("n_schur",schur);
    item.set("incident_amplitudes",encode_real({meta.normalization.electric_incident_amplitude,meta.normalization.magnetic_incident_amplitude}));
    item.set("harmonics",encode_fields(harmonics.components));item.set("fields",encode_fields(grid.components));
    JV absolute=JV::array(),incident=JV::array(),masks=JV::array();rc::PhaseOptions options;
    for(int c=0;c<6;++c){options.reference=rc::PhaseReference::Absolute;const auto a=rc::phase_data(grid,static_cast<rc::FieldComponent>(c),options);
        options.reference=rc::PhaseReference::Incident;const auto b=rc::phase_data(grid,static_cast<rc::FieldComponent>(c),options);
        absolute.push_back(encode_phase(a,false));incident.push_back(encode_phase(b,false));masks.push_back(encode_phase(a,true));}
    item.set("phase_absolute",std::move(absolute));item.set("phase_incident",std::move(incident));item.set("phase_mask",std::move(masks));return item;
}
static int export_reference(const char* path){
    std::ifstream input(path);if(!input)throw std::runtime_error("cannot open reference manifest");
    const auto manifest=json::parse(std::string(std::istreambuf_iterator<char>(input),std::istreambuf_iterator<char>()));JV output=JV::array();
    for(std::size_t j=0;j<manifest.at("cases").size();++j)output.push_back(reference_case(manifest.at("cases").at(j).at("parameters")));
    std::puts(json::dump(output).c_str());return 0;
}
static int phase_case(){
    const auto request=json::parse(std::string(std::istreambuf_iterator<char>(std::cin),std::istreambuf_iterator<char>()));
    auto grid=synthetic(request.at("x").size(),request.at("z").size());grid.x=parse_real(request.at("x"));grid.z=parse_real(request.at("z"));
    if(request.contains("y"))grid.y=request.at("y").as_double();
    if(request.contains("wavevector")){grid.metadata.kxm={parse_complex(request.at("wavevector").at(0))};grid.metadata.ky=parse_complex(request.at("wavevector").at(1)).real();grid.metadata.kzI={parse_complex(request.at("wavevector").at(2))};}
    for(int c=0;c<6;++c){const std::string name=rc::field_component_name(static_cast<rc::FieldComponent>(c));if(!request.at("fields").contains(name))continue;
        const auto&values=request.at("fields").at(name);for(std::size_t iz=0;iz<grid.nz();++iz)for(std::size_t ix=0;ix<grid.nx();++ix)grid.components[c](ix,iz)=parse_complex(values.at(iz).at(ix));}
    rc::PhaseOptions options;options.phase_floor=request.contains("phase_floor")?request.at("phase_floor").as_double():1e-6;
    if(request.contains("reference")){const std::string name=request.at("reference").as_string();options.reference=name=="incident"?rc::PhaseReference::Incident:name=="custom"?rc::PhaseReference::Custom:rc::PhaseReference::Absolute;}
    if(request.contains("custom_reference")){const auto&ref=request.at("custom_reference");const int nz=static_cast<int>(ref.size()),nx=nz?static_cast<int>(ref.at(0).size()):0;options.custom_reference=Matrix(nx,nz);
        for(int iz=0;iz<nz;++iz)for(int ix=0;ix<nx;++ix)options.custom_reference(ix,iz)=parse_complex(ref.at(iz).at(ix));}
    if(request.contains("unwrap")){const std::string name=request.at("unwrap").as_string();options.unwrap=name=="x"?rc::PhaseUnwrap::X:name=="z"?rc::PhaseUnwrap::Z:name=="both"?rc::PhaseUnwrap::Both:rc::PhaseUnwrap::None;}
    rc::PhaseGrid phase;
    if(request.contains("projection")){std::array<cplx,3> p;for(int c=0;c<3;++c)p[c]=parse_complex(request.at("projection").at(c));
        phase=rc::phase_data(grid,request.at("component").as_string()=="H"?rc::FieldVectorFamily::H:rc::FieldVectorFamily::E,p,options);}
    else phase=rc::phase_data(grid,request.at("component").as_string(),options);
    JV output=JV::object();output.set("phase",encode_phase(phase,false));output.set("mask",encode_phase(phase,true));
    output.set("field",encode_vector(phase.field));output.set("amplitude",encode_real(phase.amplitude));output.set("reference",encode_vector(phase.reference));
    output.set("unwrapped",phase.unwrapped);output.set("reference_name",phase.reference_name);std::puts(json::dump(output).c_str());return 0;
}
static int benchmark(){
    const auto p=rc::solve_with_fields(mixed_stack(),.83,27.,12,rc::Polarization::TM);
    const auto c=rc::solve_conical_with_fields(schur_stack(),.980484673140678,40.,25.,8);
    RealVector zs(401);for(std::size_t j=0;j<zs.size();++j)zs[j]=.38*static_cast<double>(j)/400.;
    using Clock=std::chrono::steady_clock;auto start=Clock::now();const auto pg=rc::field_map(p,zs,256);
    auto mid=Clock::now();const auto cg=rc::field_map(c,zs,256,.11,{{cplx(.3,.2),cplx(-.7,.5)}});auto end=Clock::now();
    auto direct_start=Clock::now();for(double z:zs)(void)rc::field_at(c,z,{{cplx(.3,.2),cplx(-.7,.5)}});auto direct_end=Clock::now();
    std::printf("grid 401x256: planar M=12 %.3f ms; Schur M=8 %.3f ms; 401 direct Schur samples %.3f ms\n",
        std::chrono::duration<double,std::milli>(mid-start).count(),std::chrono::duration<double,std::milli>(end-mid).count(),std::chrono::duration<double,std::milli>(direct_end-direct_start).count());
    return pg.components[0].all_finite()&&cg.components[0].all_finite()?0:1;
}
int main(int argc,char**argv){
    try{
        if(argc==3&&std::string(argv[1])=="--reference")return export_reference(argv[2]);
        if(argc==2&&std::string(argv[1])=="--phase-case")return phase_case();
        if(argc==2&&std::string(argv[1])=="--benchmark")return benchmark();
        test_sampling_order_interfaces_and_ownership();test_analytic_film_and_zero();test_empty_extension_zero_thickness_phi0_and_failure();
        test_schur_and_thick_absorption();test_phase_contract();
        std::printf("Public fields and phase: %d checks, %d failures\n",checks,failures);return failures?1:0;
    }catch(const std::exception&e){std::fprintf(stderr,"FAIL exception: %s\n",e.what());return 1;}
}
