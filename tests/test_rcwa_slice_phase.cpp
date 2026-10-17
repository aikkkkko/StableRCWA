// 切面网格（x-z / y-z）与相位的验收：全部用合成数据，不调用任何求解器。
//
//   c++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -I include tests/test_rcwa_slice_phase.cpp -o build/test_rcwa_slice_phase
//   c++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -I include tests/test_rcwa_slice_phase.cpp -o build/test_rcwa_slice_phase_san
//
// 覆盖：FieldGridSlice 的轴序与布局（xz、yz）；二维元数据表；逐点 carrier 在 yz 面上的取值（人造单平面波的相对入射相位恒定）；
// 掩膜与零场；跨掩膜不连接的展开、先水平后 z 的 both 展开；一维 FieldGridXZ 转成切面网格后与旧入口逐位相同；错误输入抛 invalid_argument。
// 数值误差汇总在最后一行之前打印。
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "stablercwa/rcwa_phase.hpp"

using namespace stablercwa;
namespace rc = stablercwa::rcwa;

static int checks = 0, failures = 0;
static void check(bool ok, const char* expr, int line) {
    ++checks;
    if (!ok) { ++failures; std::printf("FAIL line %d: %s\n", line, expr); }
}
#define CHECK(x) check(static_cast<bool>(x), #x, __LINE__)

template<class F> static bool invalid(F f) { try { f(); } catch (const std::invalid_argument&) { return true; } catch (...) { return false; } return false; }
template<class F> static bool no_throw(F f) { try { f(); } catch (...) { return false; } return true; }
template<class F> static bool out_of_range(F f) { try { f(); } catch (const std::out_of_range&) { return true; } catch (...) { return false; } return false; }

// 误差统计：每个类别记录最大值，最后打印。
static double err_brute = 0, err_carrier = 0, err_relative = 0, err_unwrap = 0, err_orders = 0;
static std::size_t bit_compared = 0, bit_mismatch = 0;
static void track(double& slot, double v) { if (!(v <= slot)) slot = v; }
static void close(cplx a, cplx b, double tol, double* slot = nullptr) {
    const double e = std::abs(a - b);
    if (slot) track(*slot, e);
    CHECK(e <= tol * (1.0 + std::abs(b)));
}

// 逐位相同（NaN 也按位模式比较）。
static void same(double a, double b) { ++bit_compared; if (std::memcmp(&a, &b, sizeof a) != 0) { ++bit_mismatch; CHECK(false); } else CHECK(true); }
static void same(cplx a, cplx b) { same(a.real(), b.real()); same(a.imag(), b.imag()); }

struct Lcg {  // 确定性伪随机，保证测试可复现
    unsigned long long s;
    explicit Lcg(unsigned long long seed) : s(seed) {}
    double next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return static_cast<double>(s >> 11) * (1.0 / 9007199254740992.0); }
    double signed_unit() { return 2.0 * next() - 1.0; }
};

// ------------------------------------------------------------------ 二维元数据（公式与 rcwa2d_types.hpp 的约定一致）
struct Setup2D {
    rc::FieldMetadata meta;
    int M1 = 0, M2 = 0, N = 0, i0 = 0;
    double nI = 1.3, theta = 35.0, phi = 40.0, lambda = 0.8, Lx = 0.9, Ly = 0.55, k0 = 0.0;
    double kx0 = 0.0, ky0 = 0.0, kz0 = 0.0;   // 入射零级波矢，从第一性原理直接算，不读元数据
};
static int harmonic_index(int m, int n, int M1, int M2) { return (m + M1) * (2 * M2 + 1) + (n + M2); }  // 与 harmonic_index_2d 同式

static Setup2D make_setup(int M1 = 2, int M2 = 1) {
    Setup2D s; s.M1 = M1; s.M2 = M2; s.N = (2 * M1 + 1) * (2 * M2 + 1);
    s.k0 = 2.0 * kPi / s.lambda;
    const double th = s.theta * kPi / 180.0, ph = s.phi * kPi / 180.0;
    auto& meta = s.meta;
    meta.kind = rc::FieldSolveKind::Periodic2D;
    meta.wavelength = s.lambda; meta.k0 = s.k0; meta.theta_deg = s.theta; meta.phi_deg = s.phi;
    meta.period = s.Lx; meta.period_y = s.Ly; meta.nI = s.nI; meta.nII = 1.5;
    rc::detail::fill_harmonic_orders_2d(meta, M1, M2);
    meta.kxm.resize(s.N); meta.kyn.resize(s.N); meta.kzI.resize(s.N); meta.kzII.resize(s.N);
    for (int m = -M1; m <= M1; ++m) for (int n = -M2; n <= M2; ++n) {
        const int j = harmonic_index(m, n, M1, M2);
        const double Kx = s.nI * std::sin(th) * std::cos(ph) - m * s.lambda / s.Lx;
        const double Ky = s.nI * std::sin(th) * std::sin(ph) - n * s.lambda / s.Ly;
        meta.kxm[j] = s.k0 * Kx; meta.kyn[j] = s.k0 * Ky;
        for (int side = 0; side < 2; ++side) {
            const double index = side == 0 ? s.nI : 1.5;
            cplx kz = std::sqrt(cplx(index * index - Kx * Kx - Ky * Ky));
            if (kz.imag() < 0.0) kz = -kz;
            (side == 0 ? meta.kzI : meta.kzII)[j] = s.k0 * kz;
        }
    }
    s.i0 = harmonic_index(0, 0, M1, M2);
    meta.ky = meta.kyn[s.i0];
    s.kx0 = s.k0 * s.nI * std::sin(th) * std::cos(ph);
    s.ky0 = s.k0 * s.nI * std::sin(th) * std::sin(ph);
    s.kz0 = s.k0 * s.nI * std::cos(th);
    meta.normalization = {1.0, 1.3};
    return s;
}

static rc::HarmonicFields random_harmonics(const rc::FieldMetadata& meta, const RealVector& zs, std::uint64_t seed) {
    rc::HarmonicFields h;
    h.orders = meta.orders; h.z = zs; h.metadata = meta;
    Lcg rng(seed);
    for (auto& component : h.components) {
        component = Matrix(static_cast<int>(meta.orders.size()), static_cast<int>(zs.size()));
        for (int iz = 0; iz < component.cols(); ++iz) for (int j = 0; j < component.rows(); ++j)
            component(j, iz) = cplx(rng.signed_unit(), rng.signed_unit());
    }
    return h;
}

// ------------------------------------------------------------------ 1. 二维谐波表与元数据
static void test_orders_table() {
    rc::FieldMetadata meta;
    rc::detail::fill_harmonic_orders_2d(meta, 2, 1);
    CHECK(meta.orders.size() == 15 && meta.orders_m.size() == 15 && meta.orders_n.size() == 15);
    CHECK(meta.truncation_order == 2 && meta.truncation_order_y == 1);
    for (int m = -2; m <= 2; ++m) for (int n = -1; n <= 1; ++n) {
        const int j = harmonic_index(m, n, 2, 1);
        CHECK(meta.orders[j] == j && meta.orders_m[j] == m && meta.orders_n[j] == n);
    }
    CHECK(meta.orders_m[0] == -2 && meta.orders_n[0] == -1 && meta.orders_m[14] == 2 && meta.orders_n[14] == 1);   // m 外层、n 内层
    rc::detail::fill_harmonic_orders_2d(meta, 0, 0);                                                              // 退化：只剩零级，且覆盖旧内容
    CHECK(meta.orders == std::vector<int>{0} && meta.orders_m == std::vector<int>{0} && meta.orders_n == std::vector<int>{0});
    CHECK(invalid([&] { rc::detail::fill_harmonic_orders_2d(meta, -1, 0); }));
    CHECK(invalid([&] { rc::detail::fill_harmonic_orders_2d(meta, 0, -2); }));
    CHECK(invalid([&] { rc::detail::fill_harmonic_orders_2d(meta, 40000, 40000); }));    // 谐波总数超出 Matrix 维数

    // 一维元数据的默认值：二维专用字段全空，一维的有限性检查不受影响。
    rc::FieldMetadata plain;
    CHECK(plain.kind == rc::FieldSolveKind::PlanarTE && plain.period_y == 0.0 && plain.kyn.empty() && plain.orders_m.empty());
    CHECK(rc::detail::finite_field_metadata(plain));
    auto s = make_setup();
    CHECK(rc::detail::finite_field_metadata(s.meta));
    auto bad = s.meta; bad.kyn[3] = std::numeric_limits<double>::quiet_NaN();
    CHECK(!rc::detail::finite_field_metadata(bad));
    bad = s.meta; bad.period_y = std::numeric_limits<double>::infinity();
    CHECK(!rc::detail::finite_field_metadata(bad));

    // 第一性原理的零级波矢与元数据一致，ky 显示字段等于零级 ky。
    CHECK(std::abs(s.meta.kxm[s.i0].real() - s.kx0) <= 1e-13 && std::abs(s.meta.kyn[s.i0] - s.ky0) <= 1e-13);
    CHECK(std::abs(s.meta.kzI[s.i0].real() - s.kz0) <= 1e-13 && s.meta.ky == s.meta.kyn[s.i0]);
    // 逐列 ky 随 n 变（不随 m 变），kx 随 m 变（不随 n 变）。
    for (int m = -2; m <= 2; ++m) for (int n = -1; n <= 1; ++n) {
        const int j = harmonic_index(m, n, 2, 1);
        track(err_orders, std::abs(s.meta.kxm[j].real() - (s.kx0 - s.k0 * m * s.lambda / s.Lx)));
        track(err_orders, std::abs(s.meta.kyn[j] - (s.ky0 - s.k0 * n * s.lambda / s.Ly)));
    }
    CHECK(err_orders <= 1e-12);
    bool has_evanescent = false, has_propagating = false;
    for (cplx v : s.meta.kzI) { has_evanescent = has_evanescent || v.imag() > 0.0; has_propagating = has_propagating || v.imag() == 0.0; }
    CHECK(has_evanescent && has_propagating);   // 测试数据里两种级次都有
}

// ------------------------------------------------------------------ 2. 轴序与布局（xz 与 yz）
static void test_layout_and_brute_force() {
    const auto s = make_setup(2, 1);
    const RealVector zs{0.1, -0.2, 0.35, 0.1};   // 保留输入顺序与重复
    const int nt = 7, nz = static_cast<int>(zs.size());      // nt ≠ nz，转置错误会被发现
    for (int pass = 0; pass < 2; ++pass) {
        const rc::SlicePlane plane = pass == 0 ? rc::SlicePlane::XZ : rc::SlicePlane::YZ;
        const double at = pass == 0 ? 0.123 : -0.31, period = pass == 0 ? s.Lx : s.Ly;
        const auto harmonics = random_harmonics(s.meta, zs, 17 + pass);
        const auto grid = rc::detail::reconstruct_field_slice(harmonics, plane, nt, at);
        CHECK(grid.plane == plane && grid.slice_at == at);
        CHECK(grid.nt() == static_cast<std::size_t>(nt) && grid.nz() == static_cast<std::size_t>(nz));
        CHECK(grid.z == zs);
        CHECK(grid.t.size() == static_cast<std::size_t>(nt) && grid.x.size() == grid.t.size() && grid.y.size() == grid.t.size());
        for (int it = 0; it < nt; ++it) {
            CHECK(grid.t[it] == (static_cast<double>(it) / nt) * period);
            CHECK(std::abs(grid.t[it] - it * period / nt) <= 1e-15 * period);       // t_j = j L / nt
            CHECK(grid.x[it] == (pass == 0 ? grid.t[it] : at));
            CHECK(grid.y[it] == (pass == 0 ? at : grid.t[it]));
        }
        CHECK(grid.t.front() == 0.0 && grid.t.back() < period);                       // 不重复周期终点
        CHECK(std::abs(grid.t.back() - period * (nt - 1) / nt) <= 1e-15 * period);
        CHECK(rc::slice_period(grid.metadata, plane) == period);
        for (std::size_t c = 0; c < 6; ++c) {
            const auto cc = static_cast<rc::FieldComponent>(c);
            CHECK(grid.component(cc).rows() == nt && grid.component(cc).cols() == nz);   // Matrix [nt, nz]
            for (int iz = 0; iz < nz; ++iz) for (int it = 0; it < nt; ++it) {
                cplx expected{};
                for (int j = 0; j < s.N; ++j)
                    expected += harmonics.components[c](j, iz) * std::exp(-kI * (s.meta.kxm[j] * grid.x[it] + s.meta.kyn[j] * grid.y[it]));
                close(grid.at(cc, iz, it), expected, 1e-13, &err_brute);
                CHECK(&grid.at(cc, iz, it) == &grid.component(cc)(it, iz));              // at(iz, it) 就是 Matrix(it, iz)
                CHECK(&grid.at(cc, iz, it) == grid.component(cc).data() + iz * nt + it); // 等价 Python (nz, nt) C 顺序
            }
        }
        auto writable = grid;                                                                       // 非 const 访问器同序
        for (int iz = 0; iz < nz; ++iz) for (int it = 0; it < nt; ++it) {
            CHECK(&writable.at(rc::FieldComponent::Hz, iz, it) == writable.component(rc::FieldComponent::Hz).data() + iz * nt + it);
            CHECK(&writable.at(rc::FieldComponent::Hz, iz, it) == &writable.component(rc::FieldComponent::Hz)(it, iz));
        }
        writable.at(rc::FieldComponent::Hy, 1, 2) = cplx(7.0, 8.0);
        CHECK(writable.component(rc::FieldComponent::Hy)(2, 1) == cplx(7.0, 8.0) && grid.at(rc::FieldComponent::Hy, 1, 2) != cplx(7.0, 8.0));
        CHECK(out_of_range([&] { (void)grid.at(rc::FieldComponent::Ex, static_cast<std::size_t>(nz), 0); }));
        CHECK(out_of_range([&] { (void)writable.at(rc::FieldComponent::Ex, 0, static_cast<std::size_t>(nt)); }));
        CHECK(out_of_range([&] { (void)grid.at(rc::FieldComponent::Ex, 0, static_cast<std::size_t>(nt)); }));
        CHECK(invalid([&] { (void)grid.at(static_cast<rc::FieldComponent>(9), 0, 0); }));            // 非法分量下标
    }

    // 单个谐波 (m, n) = (1, -1)：该列的 kx、ky 必须各用各自的下标，场恰为 H·exp[-i(kx(m) x + ky(n) y)]。
    auto h = random_harmonics(s.meta, RealVector{0.0, 0.2}, 5);
    for (auto& component : h.components) component = Matrix(s.N, 2);
    const int j = harmonic_index(1, -1, 2, 1);
    for (auto& component : h.components) { component(j, 0) = cplx(0.7, -0.2); component(j, 1) = cplx(-0.3, 0.9); }
    for (int pass = 0; pass < 2; ++pass) {
        const auto g = rc::detail::reconstruct_field_slice(h, pass == 0 ? rc::SlicePlane::XZ : rc::SlicePlane::YZ, 5, 0.17);
        for (int iz = 0; iz < 2; ++iz) for (int it = 0; it < 5; ++it) {
            const cplx expected = h.components[2](j, iz) * std::exp(-kI * (s.meta.kxm[j] * g.x[it] + s.meta.kyn[j] * g.y[it]));
            close(g.at(rc::FieldComponent::Ez, iz, it), expected, 1e-13, &err_brute);
        }
    }

    // 空深度：每个分量 [nt, 0]。
    const auto empty = rc::detail::reconstruct_field_slice(random_harmonics(s.meta, {}, 1), rc::SlicePlane::YZ, 4, 0.0);
    CHECK(empty.nz() == 0 && empty.nt() == 4);
    for (const auto& component : empty.components) CHECK(component.rows() == 4 && component.cols() == 0);
}

// ------------------------------------------------------------------ 3. carrier：人造单平面波的相对入射相位恒定
// 场 = A·exp[i(-kx0 x - ky0 y + kz0 z)]，只在零级 (0,0) 列上激发。kz0 可取复数（吸收/倏逝参考）：carrier 只取实部。
static rc::HarmonicFields plane_wave_harmonics(const Setup2D& s, const RealVector& zs, const std::array<cplx, 6>& amplitude, cplx kz) {
    rc::HarmonicFields h;
    h.orders = s.meta.orders; h.z = zs; h.metadata = s.meta;
    h.metadata.kzI[s.i0] = kz;
    for (std::size_t c = 0; c < 6; ++c) {
        h.components[c] = Matrix(s.N, static_cast<int>(zs.size()));
        for (std::size_t iz = 0; iz < zs.size(); ++iz) h.components[c](s.i0, static_cast<int>(iz)) = amplitude[c] * std::exp(kI * kz * zs[iz]);
    }
    return h;
}
static double wrap_diff(double a, double b) { return std::remainder(a - b, 2.0 * kPi); }

static void test_carrier_on_slices() {
    const auto s = make_setup(2, 1);
    RealVector zs(6); for (std::size_t j = 0; j < zs.size(); ++j) zs[j] = 0.11 * static_cast<double>(j) - 0.05;
    const std::array<cplx, 6> amplitude{{std::polar(1.7, 0.37), std::polar(0.9, -2.9), std::polar(0.5, 3.0),
                                         std::polar(1.1, 1.2), std::polar(2.0, -0.05), std::polar(0.7, 2.4)}};
    for (int variant = 0; variant < 2; ++variant) {
        const cplx kz = variant == 0 ? cplx(s.kz0, 0.0) : cplx(s.kz0, 0.6);   // 第二组：参考波带衰减，幅值不进 carrier
        const auto h = plane_wave_harmonics(s, zs, amplitude, kz);
        for (int pass = 0; pass < 2; ++pass) {
            const rc::SlicePlane plane = pass == 0 ? rc::SlicePlane::XZ : rc::SlicePlane::YZ;
            const double at = pass == 0 ? 0.21 : 0.37;
            const auto grid = rc::detail::reconstruct_field_slice(h, plane, 24, at);
            const auto carrier = rc::incident_reference(grid);
            CHECK(carrier.size() == 24 * zs.size());
            // 与第一性原理的 exp(i[-kx0 x - ky0 y + kz0 z])（x、y 用切面坐标的闭式）逐点比较。
            double ky_spread = 0.0;
            for (std::size_t iz = 0; iz < zs.size(); ++iz) for (int it = 0; it < 24; ++it) {
                const double t = (static_cast<double>(it) / 24) * (pass == 0 ? s.Lx : s.Ly);
                const double x = pass == 0 ? t : at, y = pass == 0 ? at : t;
                const cplx expected = std::exp(kI * (-s.kx0 * x - s.ky0 * y + s.kz0 * zs[iz]));
                close(carrier[iz * 24 + it], expected, 1e-13, &err_carrier);
                close(std::abs(carrier[iz * 24 + it]), 1.0, 1e-14);
                ky_spread = (std::max)(ky_spread, std::abs(s.ky0 * y));
            }
            if (pass == 1) CHECK(ky_spread > 1.0);     // yz 面里 ky0·y 扫过多个弧度：丢掉这一项测试会失败
            rc::PhaseOptions rel; rel.reference = rc::PhaseReference::Incident;
            rc::PhaseOptions abs_options;
            for (std::size_t c = 0; c < 6; ++c) {
                const auto cc = static_cast<rc::FieldComponent>(c);
                const auto relative = rc::phase_data(grid, cc, rel);
                CHECK(relative.plane == plane && relative.nt() == 24 && relative.nz == zs.size() && relative.reference_name == "incident");
                for (std::size_t j = 0; j < relative.phase.size(); ++j) {
                    CHECK(!relative.mask[j]);
                    track(err_relative, std::abs(wrap_diff(relative.phase[j], std::arg(amplitude[c]))));
                    CHECK(std::abs(wrap_diff(relative.phase[j], std::arg(amplitude[c]))) <= 5e-13);
                }
                const auto absolute = rc::phase_data(grid, cc, abs_options);   // 对照：绝对相位随位置变化
                double lo = 1e9, hi = -1e9;
                for (double v : absolute.phase) { lo = (std::min)(lo, v); hi = (std::max)(hi, v); }
                CHECK(hi - lo > 1.0);
            }
            // 幅值衰减时的相对相位同样恒定，且幅值保留（carrier 不乘进显示值）。
            const auto ey = rc::phase_data(grid, rc::FieldComponent::Ey, rel);
            for (std::size_t iz = 0; iz < zs.size(); ++iz)
                for (int it = 0; it < 24; ++it)
                    close(ey.amplitude[iz * 24 + it], std::abs(amplitude[1]) * std::exp(-kz.imag() * zs[iz]), 1e-13);
        }
    }

    // 对比：XZ 面上二维元数据 + FieldGridXZ（旧网格类型）给出同一个 carrier，逐位相同。
    const auto h = plane_wave_harmonics(s, zs, amplitude, cplx(s.kz0, 0.0));
    const auto xz_grid = rc::detail::reconstruct_field_grid(h, 24, 0.21);
    const auto slice_grid = rc::detail::reconstruct_field_slice(h, rc::SlicePlane::XZ, 24, 0.21);
    const auto a = rc::incident_reference(xz_grid), b = rc::incident_reference(slice_grid);
    CHECK(a.size() == b.size());
    for (std::size_t j = 0; j < a.size(); ++j) same(a[j], b[j]);
    for (std::size_t c = 0; c < 6; ++c) for (std::size_t j = 0; j < 24 * zs.size(); ++j)
        same(xz_grid.components[c].data()[j], slice_grid.components[c].data()[j]);

    // 零级必须按 (m, n) = (0, 0) 找，而不是 orders 里的 0：orders[0] = 0 是 (-M1, -M2) 那一列。
    CHECK(s.meta.orders[0] == 0 && s.meta.orders_m[0] == -2 && s.meta.orders_n[0] == -1);
    auto moved = rc::detail::reconstruct_field_slice(h, rc::SlicePlane::YZ, 6, 0.0);
    const auto original = rc::incident_reference(moved);
    moved.metadata.kxm[0] = 1234.5; moved.metadata.kyn[0] = -777.0; moved.metadata.kzI[0] = 99.0;   // 改 j=0 列不影响 carrier
    const auto after = rc::incident_reference(moved);
    for (std::size_t j = 0; j < original.size(); ++j) same(original[j], after[j]);
    moved.metadata.kyn[s.i0] += 0.5;                                                       // 改零级列则必须变化
    const auto changed = rc::incident_reference(moved);
    bool differs = false;
    for (std::size_t j = 0; j < original.size(); ++j) differs = differs || std::abs(changed[j] - original[j]) > 1e-3;
    CHECK(differs);
}

// ------------------------------------------------------------------ 手工构造的小切面（掩膜 / 展开测试用）
static rc::FieldGridSlice manual_slice(rc::SlicePlane plane, std::size_t nt, std::size_t nz) {
    rc::FieldGridSlice g;
    g.plane = plane; g.slice_at = 0.11;
    g.t.resize(nt); g.x.resize(nt); g.y.resize(nt); g.z.resize(nz);
    for (std::size_t j = 0; j < nt; ++j) {
        g.t[j] = 0.13 * static_cast<double>(j);
        g.x[j] = plane == rc::SlicePlane::XZ ? g.t[j] : g.slice_at;
        g.y[j] = plane == rc::SlicePlane::XZ ? g.slice_at : g.t[j];
    }
    for (std::size_t j = 0; j < nz; ++j) g.z[j] = 0.17 * static_cast<double>(j);
    auto& meta = g.metadata;
    meta.kind = rc::FieldSolveKind::Periodic2D;
    meta.orders = {0}; meta.orders_m = {0}; meta.orders_n = {0};
    meta.kxm = {cplx(2.3, 0.8)}; meta.kyn = {0.7}; meta.kzI = {cplx(3.1, 0.5)}; meta.ky = 0.7;
    for (auto& component : g.components) component = Matrix(static_cast<int>(nt), static_cast<int>(nz));
    return g;
}

// ------------------------------------------------------------------ 4. 掩膜与零场
static void test_masks_and_zero_fields() {
    const double qnan = std::numeric_limits<double>::quiet_NaN(), inf = std::numeric_limits<double>::infinity();
    auto grid = manual_slice(rc::SlicePlane::YZ, 4, 2);
    auto& f = grid.component(rc::FieldComponent::Ey);
    rc::PhaseOptions options; options.phase_floor = 0.25;
    f(0, 0) = 1.0; f(1, 0) = 0.25; f(2, 0) = 0.2500001; f(3, 0) = 0.0;           // 峰值 1：恰等于阈值被掩膜，略大于不掩膜，零被掩膜
    f(0, 1) = qnan; f(1, 1) = cplx(inf, 0.0); f(2, 1) = cplx(0.0, -inf); f(3, 1) = cplx(qnan, 1.0);   // 非有限一律掩膜
    auto phase = rc::phase_data(grid, rc::FieldComponent::Ey, options);
    CHECK(phase.plane == rc::SlicePlane::YZ && phase.nt() == 4 && phase.nx == 4 && phase.nz == 2 && !phase.unwrapped);
    CHECK(!phase.mask[0] && phase.mask[1] && !phase.mask[2] && phase.mask[3]);
    for (int it = 0; it < 4; ++it) CHECK(phase.mask[4 + it]);
    for (std::size_t j = 0; j < phase.mask.size(); ++j) CHECK(phase.mask[j] == std::isnan(phase.phase[j]));
    CHECK(phase.phase[0] == 0.0 && std::abs(phase.phase[2]) == 0.0);
    CHECK(phase.amplitude[1] == 0.25 && phase.amplitude[3] == 0.0 && phase.field.size() == 8);   // 被掩膜点的幅值与原场保留
    CHECK(phase.at(0, 2) == phase.phase[2] && phase.masked(0, 1) && !phase.masked(0, 0));
    CHECK(out_of_range([&] { (void)phase.at(2, 0); }));

    // 峰值只取有限幅值：含 Inf 的场里，有限点仍按有限峰值判断。
    f(0, 0) = cplx(inf, 0.0); f(1, 0) = 2.0; f(2, 0) = 0.5; f(3, 0) = cplx(0.0, 0.5);
    f(0, 1) = 0.1; f(1, 1) = 0.0; f(2, 1) = -2.0; f(3, 1) = cplx(0.0, 0.1);
    options.phase_floor = 0.1;     // 阈值 0.2：0.1 的两个被掩膜，0.5 与 2.0 保留，Inf 掩膜
    phase = rc::phase_data(grid, rc::FieldComponent::Ey, options);
    const bool expected_mask[8] = {true, false, false, false, true, true, false, true};
    for (int j = 0; j < 8; ++j) CHECK(phase.mask[j] == expected_mask[j]);
    CHECK(std::abs(phase.phase[6] - kPi) <= 1e-15 || std::abs(phase.phase[6] + kPi) <= 1e-15);   // arg(-2) = π

    // 零场（该分量全零）：全部掩膜，即使 floor = 0。
    for (double floor : {1e-6, 0.0, 0.5}) {
        options.phase_floor = floor;
        const auto zero = rc::phase_data(grid, rc::FieldComponent::Ez, options);
        for (std::size_t j = 0; j < zero.mask.size(); ++j) CHECK(zero.mask[j] && std::isnan(zero.phase[j]) && zero.amplitude[j] == 0.0);
        options.reference = rc::PhaseReference::Incident;
        const auto zero_incident = rc::phase_data(grid, rc::FieldComponent::Ez, options);
        for (std::size_t j = 0; j < zero_incident.mask.size(); ++j) CHECK(zero_incident.mask[j] && std::isnan(zero_incident.phase[j]));
        options.reference = rc::PhaseReference::Absolute;
    }
    // floor = 0 且场非零：只有零值（和非有限）被掩膜。
    options.phase_floor = 0.0;
    phase = rc::phase_data(grid, rc::FieldComponent::Ey, options);
    for (int j = 0; j < 8; ++j) CHECK(phase.mask[j] == (j == 0 || j == 5));

    // 参考无效：非有限或为零的参考点被掩膜；[nt,1] 横向行、[1,nz] 纵向列、[1,1] 标量的广播。
    f(0, 0) = 1.0; f(1, 0) = 2.0; f(2, 0) = cplx(0.0, 1.0); f(3, 0) = -1.0;
    f(0, 1) = cplx(1.0, 1.0); f(1, 1) = 1.5; f(2, 1) = -2.0; f(3, 1) = cplx(0.0, -3.0);
    options = {}; options.reference = rc::PhaseReference::Custom;
    options.custom_reference = Matrix(1, 1, cplx(0.0, 3.0));
    phase = rc::phase_data(grid, rc::FieldComponent::Ey, options);
    CHECK(phase.reference_name == "custom");
    for (std::size_t j = 0; j < phase.phase.size(); ++j) {
        CHECK(!phase.mask[j] && phase.reference[j] == cplx(0.0, 3.0));
        close(std::exp(kI * phase.phase[j]), phase.field[j] / std::abs(phase.field[j]) * cplx(0.0, -1.0), 3e-15);
    }
    options.custom_reference = Matrix(4, 1, 1.0); options.custom_reference(2, 0) = 0.0;
    phase = rc::phase_data(grid, rc::FieldComponent::Ey, options);
    CHECK(phase.mask[2] && phase.mask[6] && !phase.mask[1] && !phase.mask[5]);
    options.custom_reference = Matrix(1, 2, 1.0); options.custom_reference(0, 1) = inf;
    phase = rc::phase_data(grid, rc::FieldComponent::Ey, options);
    CHECK(!phase.mask[0] && !phase.mask[3] && phase.mask[4] && phase.mask[7]);
    options.custom_reference = Matrix(4, 2, 1.0); options.custom_reference(3, 1) = cplx(qnan, 0.0);
    phase = rc::phase_data(grid, rc::FieldComponent::Ey, options);
    for (int j = 0; j < 8; ++j) CHECK(phase.mask[j] == (j == 7));
    options.custom_reference = Matrix(3, 3, 1.0);
    CHECK(invalid([&] { (void)rc::phase_data(grid, rc::FieldComponent::Ey, options); }));
    options.custom_reference = Matrix(4, 3, 1.0);
    CHECK(invalid([&] { (void)rc::phase_data(grid, rc::FieldComponent::Ey, options); }));

    // 矢量分析器：projection^H @ field，不归一化。
    auto vec = manual_slice(rc::SlicePlane::XZ, 2, 1);
    vec.components[0] = Matrix(2, 1, cplx(1.0, 2.0)); vec.components[1] = Matrix(2, 1, cplx(3.0, -0.5));
    const std::array<cplx, 3> projection{{cplx(2.0, 1.0), cplx(-0.3, 0.7), 0.0}};
    const auto projected = rc::phase_data(vec, rc::FieldVectorFamily::E, projection);
    const cplx expected = std::conj(projection[0]) * cplx(1.0, 2.0) + std::conj(projection[1]) * cplx(3.0, -0.5);
    close(projected.field[0], expected, 2e-16); close(projected.amplitude[0], std::abs(expected), 2e-16);
    CHECK(projected.component == "E_projection" && rc::phase_data(vec, rc::FieldVectorFamily::H, projection).component == "H_projection");
}

// ------------------------------------------------------------------ 5. 展开
// np.unwrap 的独立复写（period = 2π，discont = π）：与被测实现用不同的取模写法。
static std::vector<double> ref_unwrap(const std::vector<double>& p) {
    std::vector<double> up = p;
    double cumulative = 0.0;
    for (std::size_t j = 1; j < p.size(); ++j) {
        const double dd = p[j] - p[j - 1];
        double ddmod = (dd + kPi) - 2.0 * kPi * std::floor((dd + kPi) / (2.0 * kPi)) - kPi;
        if (ddmod == -kPi && dd > 0.0) ddmod = kPi;
        if (std::abs(dd) >= kPi) cumulative += ddmod - dd;
        up[j] = p[j] + cumulative;
    }
    return up;
}
// 沿一个轴对每条线的连续有效段做展开；phase 是 [nz, nt] C 顺序。
static void ref_unwrap_axis(std::vector<double>& phase, const std::vector<bool>& mask, std::size_t nt, std::size_t nz, bool along_t) {
    const std::size_t lines = along_t ? nz : nt, length = along_t ? nt : nz;
    for (std::size_t line = 0; line < lines; ++line) {
        auto at = [&](std::size_t k) { return along_t ? line * nt + k : k * nt + line; };
        std::size_t k = 0;
        while (k < length) {
            if (mask[at(k)]) { ++k; continue; }
            std::size_t end = k;
            std::vector<double> run;
            while (end < length && !mask[at(end)]) run.push_back(phase[at(end++)]);
            const auto up = ref_unwrap(run);
            for (std::size_t q = 0; q < run.size(); ++q) phase[at(k + q)] = up[q];
            k = end;
        }
    }
}

static void test_unwrap() {
    // 7 个点沿 t：2.9, -2.9, -2.6 | 掩膜 | 2.8, -2.8, -2.5。掩膜两侧各自展开，不互相连接。
    const RealVector angles{2.9, -2.9, -2.6, 0.0, 2.8, -2.8, -2.5};
    auto line = manual_slice(rc::SlicePlane::YZ, 7, 1);
    for (int j = 0; j < 7; ++j) line.components[1](j, 0) = j == 3 ? cplx{} : std::exp(kI * angles[j]);
    rc::PhaseOptions options; options.unwrap = rc::PhaseUnwrap::X;
    for (auto axis : {rc::PhaseUnwrap::X, rc::PhaseUnwrap::Horizontal, rc::PhaseUnwrap::T}) {
        options.unwrap = axis;
        const auto up = rc::phase_data(line, rc::FieldComponent::Ey, options);
        CHECK(up.unwrapped && up.mask[3] && std::isnan(up.phase[3]));
        const double expected[7] = {2.9, -2.9 + 2 * kPi, -2.6 + 2 * kPi, 0.0, 2.8, -2.8 + 2 * kPi, -2.5 + 2 * kPi};
        for (int j = 0; j < 7; ++j) if (j != 3) { track(err_unwrap, std::abs(up.phase[j] - expected[j])); close(up.phase[j], expected[j], 2e-15); }
    }
    // 没有掩膜时第二段会接着第一段的修正继续展开，说明上面“不连接”是掩膜造成的。
    line.components[1](3, 0) = std::exp(kI * 0.0);
    options.unwrap = rc::PhaseUnwrap::X;
    const auto joined = rc::phase_data(line, rc::FieldComponent::Ey, options);
    CHECK(!joined.mask[3] && std::abs(joined.phase[4] - 2.8) > 1.0);

    // 同一条序列沿 z（nt = 1）：Z 与 Vertical 别名，结果与沿 t 相同。
    auto column = manual_slice(rc::SlicePlane::YZ, 1, 7);
    for (int j = 0; j < 7; ++j) column.components[1](0, j) = j == 3 ? cplx{} : std::exp(kI * angles[j]);
    for (auto axis : {rc::PhaseUnwrap::Z, rc::PhaseUnwrap::Vertical}) {
        options.unwrap = axis;
        const auto up = rc::phase_data(column, rc::FieldComponent::Ey, options);
        const double expected[7] = {2.9, -2.9 + 2 * kPi, -2.6 + 2 * kPi, 0.0, 2.8, -2.8 + 2 * kPi, -2.5 + 2 * kPi};
        for (int j = 0; j < 7; ++j) if (j != 3) close(up.phase[j], expected[j], 2e-15);
    }
    // 沿 t 展开不动 z 方向：列方向的序列在 X 展开下保持包裹值。
    options.unwrap = rc::PhaseUnwrap::X;
    const auto untouched = rc::phase_data(column, rc::FieldComponent::Ey, options);
    for (int j = 0; j < 7; ++j) if (j != 3) close(untouched.phase[j], std::arg(std::exp(kI * angles[j])), 2e-15);

    // both：先 t 后 z。随机相位网格，包含掩膜节点；期望值由独立复写按同样次序算出。
    const std::size_t nt = 9, nz = 7;
    auto grid = manual_slice(rc::SlicePlane::YZ, nt, nz);
    Lcg rng(20261010);
    std::vector<double> wrapped(nt * nz);
    std::vector<bool> expected_mask(nt * nz, false);
    for (std::size_t iz = 0; iz < nz; ++iz) for (std::size_t it = 0; it < nt; ++it) {
        const std::size_t j = iz * nt + it;
        // 沿两个方向都有较大跳变：相位 = 缓变量 + 随机 ±π 附近的扰动，保证两种次序在同一个点上给出不同答案。
        wrapped[j] = std::remainder(2.4 * static_cast<double>(it) + 1.9 * static_cast<double>(iz) + 1.3 * rng.signed_unit(), 2.0 * kPi);
        expected_mask[j] = (it == 4 && iz == 3) || (it == 7 && iz == 1) || (it == 0 && iz == 5);
        grid.component(rc::FieldComponent::Hy)(static_cast<int>(it), static_cast<int>(iz)) = expected_mask[j] ? cplx{} : std::polar(0.5 + rng.next(), wrapped[j]);
    }
    auto tz = wrapped; ref_unwrap_axis(tz, expected_mask, nt, nz, true); ref_unwrap_axis(tz, expected_mask, nt, nz, false);
    auto zt = wrapped; ref_unwrap_axis(zt, expected_mask, nt, nz, false); ref_unwrap_axis(zt, expected_mask, nt, nz, true);
    auto t_only = wrapped; ref_unwrap_axis(t_only, expected_mask, nt, nz, true);
    auto z_only = wrapped; ref_unwrap_axis(z_only, expected_mask, nt, nz, false);
    bool orders_differ = false;
    for (std::size_t j = 0; j < tz.size(); ++j) orders_differ = orders_differ || (!expected_mask[j] && std::abs(tz[j] - zt[j]) > 1e-9);
    CHECK(orders_differ);   // 测试数据确实区分“先 t 后 z”与“先 z 后 t”
    options = {}; options.unwrap = rc::PhaseUnwrap::Both;
    const auto both = rc::phase_data(grid, rc::FieldComponent::Hy, options);
    options.unwrap = rc::PhaseUnwrap::X; const auto only_t = rc::phase_data(grid, rc::FieldComponent::Hy, options);
    options.unwrap = rc::PhaseUnwrap::Z; const auto only_z = rc::phase_data(grid, rc::FieldComponent::Hy, options);
    double checksum = 0.0;
    for (std::size_t j = 0; j < tz.size(); ++j) {
        CHECK(both.mask[j] == expected_mask[j] && only_t.mask[j] == expected_mask[j] && only_z.mask[j] == expected_mask[j]);
        if (expected_mask[j]) { CHECK(std::isnan(both.phase[j])); continue; }
        track(err_unwrap, std::abs(both.phase[j] - tz[j]));
        track(err_unwrap, std::abs(only_t.phase[j] - t_only[j]));
        track(err_unwrap, std::abs(only_z.phase[j] - z_only[j]));
        CHECK(std::abs(both.phase[j] - tz[j]) <= 1e-12);
        CHECK(std::abs(only_t.phase[j] - t_only[j]) <= 1e-12 && std::abs(only_z.phase[j] - z_only[j]) <= 1e-12);
        checksum += both.phase[j];
    }
    std::printf("  both 展开校验和 %.12f（供 numpy 复算）\n", checksum);
    // 有一个点 t-then-z 与 z-then-t 不同，被测实现必须落在 t-then-z 那一边。
    bool matches_tz = true, matches_zt = true;
    for (std::size_t j = 0; j < tz.size(); ++j) if (!expected_mask[j]) {
        matches_tz = matches_tz && std::abs(both.phase[j] - tz[j]) <= 1e-12;
        matches_zt = matches_zt && std::abs(both.phase[j] - zt[j]) <= 1e-12;
    }
    CHECK(matches_tz && !matches_zt);
    // 展开后的相位在每个连续有效段内相邻两点相差不超过 π；段与段之间（跨掩膜）不做这个要求。
    for (std::size_t iz = 0; iz < nz; ++iz) for (std::size_t it = 1; it < nt; ++it) {
        const std::size_t j = iz * nt + it;
        if (!expected_mask[j] && !expected_mask[j - 1]) CHECK(std::abs(only_t.phase[j] - only_t.phase[j - 1]) <= kPi + 1e-12);
    }
    // 展开选项非法
    options.unwrap = static_cast<rc::PhaseUnwrap>(9);
    CHECK(invalid([&] { (void)rc::phase_data(grid, rc::FieldComponent::Hy, options); }));
}

// ------------------------------------------------------------------ 6. 一维 FieldGridXZ → 切面网格：与旧入口逐位相同
static void test_legacy_conversion_bit_identical() {
    // 一维圆锥式元数据（ky ≠ 0、固定 y ≠ 0），三个级次，不均匀的 x 与重复的 z。
    rc::FieldMetadata meta;
    meta.kind = rc::FieldSolveKind::Conical; meta.period = 0.6; meta.k0 = 7.5; meta.truncation_order = 1;
    meta.orders = {-1, 0, 1};
    meta.kxm = {cplx(5.1, 0.0), cplx(3.9, 0.0), cplx(2.4, 0.2)}; meta.kzI = {cplx(2.0, 1.0), cplx(8.9, 0.0), cplx(4.5, 0.4)};
    meta.kzII = {cplx(3.0), cplx(9.0), cplx(1.0, 2.0)}; meta.ky = 1.7;
    meta.normalization = {1.0, 1.4};
    const RealVector zs{0.0, 0.05, 0.05, 0.31, -0.12, 0.2};
    rc::HarmonicFields h;
    h.orders = meta.orders; h.z = zs; h.metadata = meta;
    Lcg rng(99);
    for (std::size_t c = 0; c < 6; ++c) {
        h.components[c] = Matrix(3, static_cast<int>(zs.size()));
        for (int iz = 0; iz < h.components[c].cols(); ++iz) for (int j = 0; j < 3; ++j)
            h.components[c](j, iz) = c == 2 ? cplx{} : cplx(rng.signed_unit(), rng.signed_unit()) * (iz == 2 ? 1e-9 : 1.0);   // Ez 全零；一个深度几乎为零
    }
    auto legacy = rc::detail::reconstruct_field_grid(h, 11, 0.11);
    // 掺入非有限值和精确零，使掩膜、峰值、展开都有东西可走。
    legacy.component(rc::FieldComponent::Ex)(3, 1) = std::numeric_limits<double>::quiet_NaN();
    legacy.component(rc::FieldComponent::Ex)(5, 2) = cplx(std::numeric_limits<double>::infinity(), 0.0);
    legacy.component(rc::FieldComponent::Ey)(0, 3) = 0.0;
    const auto converted = rc::to_field_slice(legacy);
    CHECK(converted.plane == rc::SlicePlane::XZ && converted.slice_at == 0.11 && converted.nt() == 11 && converted.nz() == zs.size());
    CHECK(converted.t == legacy.x && converted.x == legacy.x && converted.z == legacy.z);
    for (double y : converted.y) CHECK(y == 0.11);

    const std::size_t compared_before = bit_compared;
    const auto rebuilt = rc::detail::reconstruct_field_slice(h, rc::SlicePlane::XZ, 11, 0.11);   // 一维谐波直接在切面上重建
    const auto fresh_legacy = rc::detail::reconstruct_field_grid(h, 11, 0.11);
    for (std::size_t c = 0; c < 6; ++c) {
        CHECK(fresh_legacy.components[c].rows() == rebuilt.components[c].rows() && fresh_legacy.components[c].cols() == rebuilt.components[c].cols());
        for (std::size_t j = 0; j < static_cast<std::size_t>(11 * zs.size()); ++j) same(fresh_legacy.components[c].data()[j], rebuilt.components[c].data()[j]);
    }
    for (std::size_t j = 0; j < fresh_legacy.x.size(); ++j) same(fresh_legacy.x[j], rebuilt.x[j]);

    const auto compare = [&](const rc::PhaseGrid& a, const rc::PhaseGrid& b) {
        CHECK(a.nx == b.nx && a.nz == b.nz && a.component == b.component && a.reference_name == b.reference_name);
        CHECK(a.unwrapped == b.unwrapped && a.mask == b.mask && b.plane == rc::SlicePlane::XZ);
        CHECK(a.phase_floor == b.phase_floor && a.phase.size() == b.phase.size() && a.reference.size() == b.reference.size());
        for (std::size_t j = 0; j < a.phase.size(); ++j) { same(a.phase[j], b.phase[j]); same(a.amplitude[j], b.amplitude[j]); same(a.field[j], b.field[j]); }
        for (std::size_t j = 0; j < a.reference.size(); ++j) same(a.reference[j], b.reference[j]);
    };
    const rc::PhaseUnwrap unwraps[] = {rc::PhaseUnwrap::None, rc::PhaseUnwrap::X, rc::PhaseUnwrap::Z, rc::PhaseUnwrap::Both};
    const rc::PhaseReference references[] = {rc::PhaseReference::Absolute, rc::PhaseReference::Incident, rc::PhaseReference::Custom};
    std::size_t cases = 0;
    for (double floor : {1e-6, 0.0, 0.3}) for (auto reference : references) for (auto unwrap : unwraps) for (std::size_t c = 0; c < 6; ++c) {
        rc::PhaseOptions options; options.phase_floor = floor; options.reference = reference; options.unwrap = unwrap;
        if (reference == rc::PhaseReference::Custom) {
            options.custom_reference = Matrix(11, 6, cplx(0.6, -0.8));
            options.custom_reference(4, 2) = 0.0; options.custom_reference(7, 0) = cplx(std::numeric_limits<double>::infinity(), 0.0);
        }
        const auto cc = static_cast<rc::FieldComponent>(c);
        compare(rc::phase_data(legacy, cc, options), rc::phase_data(converted, cc, options));
        ++cases;
    }
    // 字符串入口与矢量分析器入口。
    rc::PhaseOptions options; options.reference = rc::PhaseReference::Incident; options.unwrap = rc::PhaseUnwrap::Both;
    compare(rc::phase_data(legacy, "ey", options), rc::phase_data(converted, "ey", options));
    compare(rc::phase_data(legacy, " Hz ", options), rc::phase_data(converted, " Hz ", options));
    const std::array<cplx, 3> projection{{cplx(0.4, 0.3), cplx(-1.0, 0.2), cplx(0.0, 0.8)}};
    compare(rc::phase_data(legacy, rc::FieldVectorFamily::E, projection, options), rc::phase_data(converted, rc::FieldVectorFamily::E, projection, options));
    compare(rc::phase_data(legacy, rc::FieldVectorFamily::H, projection), rc::phase_data(converted, rc::FieldVectorFamily::H, projection));
    const auto va = rc::complex_component(legacy, rc::FieldComponent::Ey), vb = rc::complex_component(converted, rc::FieldComponent::Ey);
    for (std::size_t j = 0; j < va.size(); ++j) same(va[j], vb[j]);
    const auto ia = rc::incident_reference(legacy), ib = rc::incident_reference(converted);
    for (std::size_t j = 0; j < ia.size(); ++j) same(ia[j], ib[j]);
    CHECK(cases == 3 * 3 * 4 * 6);
    std::printf("  一维网格 → 切面网格：%zu 组相位参数（含六分量），%zu 个标量逐位比较，累计不同 %zu 个\n", cases, bit_compared - compared_before, bit_mismatch);
}

// ------------------------------------------------------------------ 7. 二维层几何复制（模板，只访问 Lx/Ly/nI/nII/layers[].d/homogeneous/n）
struct MockLayer2D { double d = 0.0; bool homogeneous = true; cplx n = 1.0; };
struct MockStack2D { double nI = 1.0; cplx nII = 1.0; std::vector<MockLayer2D> layers; double Lx = 1.0, Ly = 1.0; };
static void test_geometry_copy() {
    MockStack2D stack; stack.nI = 1.2; stack.nII = cplx(1.5, 0.1); stack.Lx = 0.9; stack.Ly = 0.55;
    stack.layers = {{0.3, false, 1.0}, {0.0, true, 2.0}, {0.2, true, cplx(1.6, 0.05)}};
    for (bool skip : {false, true}) {
        rc::FieldMetadata meta;
        rc::detail::copy_field_geometry_2d(meta, stack, skip);
        CHECK(meta.kind == rc::FieldSolveKind::Periodic2D && meta.period == 0.9 && meta.period_y == 0.55);
        CHECK(meta.nI == cplx(1.2) && meta.nII == cplx(1.5, 0.1) && std::abs(meta.z_bottom - 0.5) <= 1e-15);
        CHECK(meta.layers.size() == 3);
        CHECK(meta.layers[0].effective_index == 0 && meta.layers[1].effective_index == (skip ? -1 : 1) && meta.layers[2].effective_index == (skip ? 1 : 2));
        CHECK(meta.layers[0].z_top == 0.0 && meta.layers[0].z_bottom == 0.3 && meta.layers[1].thickness == 0.0);
        CHECK(meta.layers[2].z_top == 0.3 && std::abs(meta.layers[2].z_bottom - 0.5) <= 1e-15 && meta.layers[2].uniform_index == cplx(1.6, 0.05));
        CHECK(!meta.layers[0].homogeneous && meta.layers[1].homogeneous && meta.layers[0].edges.empty() && meta.layers[0].n_values.empty());
        CHECK(meta.layers[0].original_index == 0 && meta.layers[2].original_index == 2);
    }
}

// ------------------------------------------------------------------ 8. 错误输入
static void test_errors() {
    const double qnan = std::numeric_limits<double>::quiet_NaN(), inf = std::numeric_limits<double>::infinity();
    const auto s = make_setup(1, 1);
    const RealVector zs{0.0, 0.1};
    const auto good = random_harmonics(s.meta, zs, 3);

    // 切面坐标与周期
    CHECK(invalid([&] { (void)rc::slice_coordinates(rc::SlicePlane::XZ, 0, 1.0, 0.0); }));
    CHECK(invalid([&] { (void)rc::slice_coordinates(rc::SlicePlane::XZ, -2, 1.0, 0.0); }));
    CHECK(invalid([&] { (void)rc::slice_coordinates(rc::SlicePlane::YZ, 4, 0.0, 0.0); }));
    CHECK(invalid([&] { (void)rc::slice_coordinates(rc::SlicePlane::YZ, 4, -1.0, 0.0); }));
    CHECK(invalid([&] { (void)rc::slice_coordinates(rc::SlicePlane::YZ, 4, qnan, 0.0); }));
    CHECK(invalid([&] { (void)rc::slice_coordinates(rc::SlicePlane::YZ, 4, 1.0, inf); }));
    CHECK(invalid([&] { (void)rc::slice_coordinates(rc::SlicePlane::YZ, 4, 1.0, qnan); }));
    CHECK(invalid([&] { (void)rc::slice_coordinates(static_cast<rc::SlicePlane>(5), 4, 1.0, 0.0); }));
    const auto coordinates = rc::slice_coordinates(rc::SlicePlane::YZ, 4, 2.0, 0.5);
    CHECK(coordinates.t == RealVector({0.0, 0.5, 1.0, 1.5}) && coordinates.x == RealVector(4, 0.5) && coordinates.y == coordinates.t);
    CHECK(invalid([&] { (void)rc::slice_period(rc::FieldMetadata{}, rc::SlicePlane::XZ); }));          // 周期为 0
    CHECK(invalid([&] { (void)rc::slice_period(rc::FieldMetadata{}, rc::SlicePlane::YZ); }));
    CHECK(invalid([&] { (void)rc::slice_period(s.meta, static_cast<rc::SlicePlane>(7)); }));

    // 切面名字
    CHECK(rc::parse_slice_plane("xz") == rc::SlicePlane::XZ && rc::parse_slice_plane(" YZ ") == rc::SlicePlane::YZ);
    CHECK(std::string(rc::slice_plane_name(rc::SlicePlane::XZ)) == "xz" && std::string(rc::slice_plane_name(rc::SlicePlane::YZ)) == "yz");
    CHECK(invalid([&] { (void)rc::parse_slice_plane("xy"); }));
    CHECK(invalid([&] { (void)rc::parse_slice_plane(""); }));
    CHECK(invalid([&] { (void)rc::slice_plane_name(static_cast<rc::SlicePlane>(3)); }));

    // 重建
    using rc::detail::reconstruct_field_slice;
    CHECK(invalid([&] { (void)reconstruct_field_slice(good, rc::SlicePlane::XZ, 0, 0.0); }));
    CHECK(invalid([&] { (void)reconstruct_field_slice(good, rc::SlicePlane::YZ, -1, 0.0); }));
    CHECK(invalid([&] { (void)reconstruct_field_slice(good, rc::SlicePlane::YZ, 4, qnan); }));
    CHECK(invalid([&] { (void)reconstruct_field_slice(good, rc::SlicePlane::XZ, 4, inf); }));
    CHECK(invalid([&] { (void)reconstruct_field_slice(good, static_cast<rc::SlicePlane>(4), 4, 0.0); }));
    { auto h = good; h.metadata.period_y = 0.0; CHECK(invalid([&] { (void)reconstruct_field_slice(h, rc::SlicePlane::YZ, 4, 0.0); }));
      CHECK(reconstruct_field_slice(h, rc::SlicePlane::XZ, 4, 0.0).nt() == 4); }                       // xz 不需要 y 周期
    { auto h = good; h.metadata.period = qnan; CHECK(invalid([&] { (void)reconstruct_field_slice(h, rc::SlicePlane::XZ, 4, 0.0); })); }
    { auto h = good; h.metadata.kyn.pop_back(); CHECK(invalid([&] { (void)reconstruct_field_slice(h, rc::SlicePlane::XZ, 4, 0.0); }));
      CHECK(invalid([&] { (void)rc::detail::reconstruct_field_grid(h, 4, 0.0); })); }               // 旧网格入口对二维元数据同样检查
    { auto h = good; h.metadata.orders_n.pop_back(); CHECK(invalid([&] { (void)reconstruct_field_slice(h, rc::SlicePlane::XZ, 4, 0.0); })); }
    { auto h = good; h.metadata.kxm.pop_back(); CHECK(invalid([&] { (void)reconstruct_field_slice(h, rc::SlicePlane::XZ, 4, 0.0); })); }
    { auto h = good; h.components[3] = Matrix(s.N - 1, 2); CHECK(invalid([&] { (void)reconstruct_field_slice(h, rc::SlicePlane::XZ, 4, 0.0); })); }
    { auto h = good; h.components[5] = Matrix(s.N, 3); CHECK(invalid([&] { (void)reconstruct_field_slice(h, rc::SlicePlane::XZ, 4, 0.0); })); }
    { auto h = good; h.metadata.kxm[2] = cplx(qnan, 0.0);                                             // 非有限波矢 → 非有限 carrier：runtime_error
      bool threw = false; try { (void)reconstruct_field_slice(h, rc::SlicePlane::XZ, 4, 0.0); } catch (const std::runtime_error&) { threw = true; } CHECK(threw); }
    { rc::HarmonicFields h;                                                                         // 一维元数据没有 y 周期：yz 面无法重建
      h.orders = {0}; h.z = zs; h.metadata.orders = {0}; h.metadata.kxm = {cplx(1.0)}; h.metadata.period = 1.0;
      for (auto& c : h.components) c = Matrix(1, 2, 1.0);
      CHECK(invalid([&] { (void)reconstruct_field_slice(h, rc::SlicePlane::YZ, 4, 0.0); }));
      CHECK(reconstruct_field_slice(h, rc::SlicePlane::XZ, 4, 0.0).nt() == 4); }

    // 相位入口
    auto grid = reconstruct_field_slice(good, rc::SlicePlane::YZ, 5, 0.2);
    CHECK(invalid([&] { (void)rc::phase_data(grid, "E2"); }));
    CHECK(invalid([&] { (void)rc::phase_data(grid, " h2 "); }));
    CHECK(invalid([&] { (void)rc::phase_data(grid, "Bx"); }));
    CHECK(invalid([&] { (void)rc::phase_data(grid, ""); }));
    for (double bad : {-1.0, qnan, inf, -inf}) {
        rc::PhaseOptions options; options.phase_floor = bad;
        CHECK(invalid([&] { (void)rc::phase_data(grid, rc::FieldComponent::Ey, options); }));
    }
    { rc::PhaseOptions options; options.reference = static_cast<rc::PhaseReference>(9);
      CHECK(invalid([&] { (void)rc::phase_data(grid, rc::FieldComponent::Ey, options); })); }
    CHECK(invalid([&] { (void)rc::phase_data(grid, rc::FieldVectorFamily::E, {{0.0, 0.0, 0.0}}); }));
    CHECK(invalid([&] { (void)rc::phase_data(grid, rc::FieldVectorFamily::E, {{cplx(qnan, 0.0), 1.0, 0.0}}); }));
    CHECK(invalid([&] { (void)rc::phase_data(grid, static_cast<rc::FieldVectorFamily>(5), {{1.0, 0.0, 0.0}}); }));
    CHECK(invalid([&] { (void)rc::phase_data(grid, static_cast<rc::FieldComponent>(8)); }));
    { auto bad = grid; bad.components[1] = Matrix(3, 2); CHECK(invalid([&] { (void)rc::phase_data(bad, rc::FieldComponent::Ey); })); }      // 形状与 nt、nz 不符
    { auto bad = grid; bad.components[1] = Matrix(5, 3); CHECK(invalid([&] { (void)rc::phase_data(bad, rc::FieldComponent::Ey); })); }
    { auto bad = grid; bad.x.pop_back(); CHECK(invalid([&] { (void)rc::phase_data(bad, rc::FieldComponent::Ey); })); }                  // t、x、y 长度不一致
    { auto bad = grid; bad.y.push_back(0.0); CHECK(invalid([&] { (void)rc::phase_data(bad, rc::FieldComponent::Ey); })); }
    { auto bad = grid; bad.plane = static_cast<rc::SlicePlane>(6); CHECK(invalid([&] { (void)rc::phase_data(bad, rc::FieldComponent::Ey); })); }
    { rc::PhaseOptions options; options.reference = rc::PhaseReference::Incident;
      auto bad = grid; bad.x[2] = qnan; CHECK(invalid([&] { (void)rc::phase_data(bad, rc::FieldComponent::Ey, options); }));
      bad = grid; bad.y[1] = inf; CHECK(invalid([&] { (void)rc::phase_data(bad, rc::FieldComponent::Ey, options); }));
      bad = grid; bad.z[0] = qnan; CHECK(invalid([&] { (void)rc::phase_data(bad, rc::FieldComponent::Ey, options); }));
      bad = grid; bad.x.pop_back(); CHECK(invalid([&] { (void)rc::incident_reference(bad); }));
      // 零级找不到 / 不唯一 / 波矢非有限 / 数组不够长（相对入射相位需要入射波矢）
      bad = grid; for (auto& m : bad.metadata.orders_m) m = 1;
      CHECK(invalid([&] { (void)rc::phase_data(bad, rc::FieldComponent::Ey, options); }));
      bad = grid; bad.metadata.orders_m[0] = 0; bad.metadata.orders_n[0] = 0;                       // 两个 (0,0)
      CHECK(invalid([&] { (void)rc::incident_reference(bad); }));
      bad = grid; bad.metadata.kzI[s.i0] = cplx(qnan, 0.0);
      CHECK(invalid([&] { (void)rc::incident_reference(bad); }));
      bad = grid; bad.metadata.kyn[s.i0] = inf;
      CHECK(invalid([&] { (void)rc::incident_reference(bad); }));
      bad = grid; bad.metadata.kxm.resize(1);
      CHECK(invalid([&] { (void)rc::incident_reference(bad); }));
      bad = grid; bad.metadata.orders_n.pop_back();
      CHECK(invalid([&] { (void)rc::incident_reference(bad); }));
      // 绝对相位不需要入射波矢：同一个坏元数据下仍可计算。
      bad = grid; for (auto& m : bad.metadata.orders_m) m = 1;
      CHECK(no_throw([&] { (void)rc::phase_data(bad, rc::FieldComponent::Ey); })); }
    // 旧入口：一维元数据缺零级仍抛 invalid_argument（原行为）。
    { rc::FieldGridXZ legacy; legacy.x = {0.0}; legacy.z = {0.0}; legacy.metadata.orders = {1, 2}; legacy.metadata.kxm = {1.0, 2.0}; legacy.metadata.kzI = {1.0, 2.0};
      CHECK(invalid([&] { (void)rc::incident_reference(legacy); }));
      legacy.metadata.orders = {0, 0};
      CHECK(invalid([&] { (void)rc::incident_reference(legacy); }));
      legacy.metadata.orders = {0}; legacy.metadata.kxm = {1.0}; legacy.metadata.kzI = {1.0};
      legacy.y = qnan; CHECK(invalid([&] { (void)rc::incident_reference(legacy); })); }
}

int main() {
    try {
        std::printf("[二维谐波表与元数据]\n");          test_orders_table();
        std::printf("[切面轴序与布局（xz、yz）]\n");     test_layout_and_brute_force();
        std::printf("[carrier 与相对入射相位]\n");       test_carrier_on_slices();
        std::printf("[掩膜与零场]\n");                   test_masks_and_zero_fields();
        std::printf("[展开：不跨掩膜，先 t 后 z]\n");     test_unwrap();
        std::printf("[一维网格转切面：逐位相同]\n");      test_legacy_conversion_bit_identical();
        std::printf("[二维层几何复制]\n");               test_geometry_copy();
        std::printf("[错误输入]\n");                     test_errors();
        std::printf("最大误差：重建与逐项求和 %.3e；carrier 与闭式 %.3e；人造平面波相对相位 %.3e；展开与 np.unwrap 复写 %.3e；波矢表 %.3e\n",
                    err_brute, err_carrier, err_relative, err_unwrap, err_orders);
        std::printf("逐位比较 %zu 个标量，不同 %zu 个\n", bit_compared, bit_mismatch);
        std::printf("Slice grid and phase: %d checks, %d failures\n", checks, failures);
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL exception: %s\n", e.what());
        return 1;
    }
}
