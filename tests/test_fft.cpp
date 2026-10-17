#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

#include "stablercwa/rcwa_json.hpp"

using namespace stablercwa;
namespace rc = stablercwa::rcwa;
static int failures = 0;
#define CHECK(x) do { if (!(x)) { std::printf("line %d: %s\n", __LINE__, #x); ++failures; } } while (0)
template<class F> static void rejects(F f) {
    bool caught = false;
    try { f(); } catch (const std::invalid_argument&) { caught = true; }
    CHECK(caught);
}

// Independent O(N*hmax) quadrature, with phases evaluated at absolute sample
// coordinates. This checks the FFT sign, scaling, bins and midpoint/origin phase.
static Vector direct_dft(const std::vector<double>& edges, const Vector& values,
                         double period, int hmax, int N) {
    Vector out(2 * hmax + 1);
    for (int j = 0; j < N; ++j) {
        const double u = (j + .5) / N * period;
        std::size_t k = 0;
        while (k + 1 < values.size() && u >= edges[k + 1] - edges.front()) ++k;
        for (int h = -hmax; h <= hmax; ++h) {
            const double a = -2 * kPi * h * (edges.front() + u) / period;
            out[h + hmax] += values[k] * cplx(std::cos(a), std::sin(a)) / double(N);
        }
    }
    return out;
}

static void coefficients() {
    for (double origin : {0., -.3, .2, 1.2}) {
        const std::vector<double> edges{origin, origin + .3125, origin + .7, origin + 1.};
        const Vector values{{2.3, .4}, {1.1, -.2}, {3., .1}};
        for (int N : {9, 17, 60, 257, 8192}) {
            const Vector actual = rc::fourier_coefficients_fft(edges, values, 1., 4, N);
            const Vector expected = direct_dft(edges, values, 1., 4, N);
            for (std::size_t k = 0; k < actual.size(); ++k) CHECK(std::abs(actual[k] - expected[k]) < 3e-12);
        }
    }
    // A discontinuity exactly at a midpoint belongs to the segment on its right.
    const auto boundary = rc::fourier_coefficients_fft({0., .5 / 9, 1.}, {4., 1.}, 1., 4, 9);
    CHECK(std::abs(boundary[4] - 1.) < 1e-14);
    const auto constant = rc::fourier_coefficients_fft({0., 1.}, {{2., .3}}, 1., 0, 1);
    CHECK(std::abs(constant[0] - cplx(2., .3)) < 1e-14);

    const std::vector<double> edges{-.3, -.05, .7};
    const Vector values{{2., .3}, {1., -.1}};
    const auto exact = rc::detail::fourier_coefficients(edges, values, 1., 4);
    double previous = 1.;
    for (int N : {64, 256, 1024}) {
        auto c = rc::fourier_coefficients_fft(edges, values, 1., 4, N);
        double error = 0.;
        for (std::size_t j = 0; j < c.size(); ++j) error = (std::max)(error, std::abs(c[j] - exact[j]));
        CHECK(error < previous / 10);
        previous = error;
    }
    CHECK(rc::resolve_fft_samples(0) == 8192);
    CHECK(rc::resolve_fft_samples(100) == 16384);
    CHECK(rc::resolve_fft_samples(2, 9) == 9);
    CHECK(rc::resolve_fft_samples(2, rc::kMaxFFTSamples) == rc::kMaxFFTSamples);
    rejects([] { rc::resolve_fft_samples(2, 8); });
    rejects([] { rc::resolve_fft_samples(-1); });
    rejects([] { rc::resolve_fft_samples((std::numeric_limits<int>::max)()); });
    rejects([] { rc::resolve_fft_samples(32768); });
    rejects([] { rc::resolve_fft_samples(0, rc::kMaxFFTSamples + 1); });
    rejects([] { rc::validate_fourier_settings(static_cast<rc::FourierMethod>(99), 0); });
    rejects([] { rc::fourier_coefficients_fft({0., 1.}, {1.}, 1., -1, 16); });
    rejects([] { rc::fourier_coefficients_fft({0., 1.}, {1.}, 1., 1, 0); });
    rejects([] { rc::fourier_coefficients_fft({}, {}, 1., 1, 16); });
    rejects([] { rc::fourier_coefficients_fft({0., .7}, {1.}, 1., 1, 16); });
    rejects([] { rc::fourier_coefficients_fft({0., 0., 1.}, {1., 2.}, 1., 1, 16); });
    rejects([] { rc::fourier_coefficients_fft({0., 1.}, {1.}, 0., 1, 16); });
    rejects([] { rc::fourier_coefficients_fft({0., 1.}, {{NAN, 0.}}, 1., 1, 16); });
}

static void solvers() {
    rc::Stack analytic;
    analytic.period = .9; analytic.nI = 1.2; analytic.nII = 1.5;
    analytic.layers = {rc::Layer::grating(.2, {-.1, .125, .8}, {cplx(2., .02), 1.2}),
                       rc::Layer::uniform(.12, {1.7, .05})};
    auto sampled = analytic;
    sampled.layers[0].fourier_method = rc::FourierMethod::FFT;
    sampled.layers[0].fft_samples = 65536;
    for (auto pol : {rc::Polarization::TE, rc::Polarization::TM}) {
        const auto a = rc::solve(analytic, .83, 27., 3, pol);
        const auto b = rc::solve(sampled, .83, 27., 3, pol);
        CHECK(a.eig_info == 0 && b.eig_info == 0);
        for (std::size_t j = 0; j < a.R.size(); ++j) {
            CHECK(std::abs(a.R[j] - b.R[j]) < 2e-7);
            CHECK(std::abs(a.T[j] - b.T[j]) < 2e-7);
        }
    }
    const auto a = rc::solve_conical(analytic, .83, 27., 41., 3);
    const auto b = rc::solve_conical(sampled, .83, 27., 41., 3);
    CHECK(a.eig_info == 0 && b.eig_info == 0);
    for (std::size_t j = 0; j < a.Jr.size(); ++j) {
        for (auto member : {&rc::Jones::ss, &rc::Jones::sp, &rc::Jones::ps, &rc::Jones::pp}) {
            CHECK(std::abs(a.Jr[j].*member - b.Jr[j].*member) < 2e-7);
            CHECK(std::abs(a.Jt[j].*member - b.Jt[j].*member) < 2e-7);
        }
    }
    // Uniform layers retain the analytic path, even with FFT settings attached.
    auto uniform = rc::Layer::uniform(.2, 1.7);
    const auto before = rc::layer_matrix(uniform, .9, .83, 1.2, 27., 3, rc::Polarization::TM);
    uniform.fourier_method = rc::FourierMethod::FFT; uniform.fft_samples = 1;
    const auto after = rc::layer_matrix(uniform, .9, .83, 1.2, 27., 3, rc::Polarization::TM);
    for (int col = 0; col < before.cols(); ++col)
        for (int row = 0; row < before.rows(); ++row) CHECK(before(row, col) == after(row, col));
    sampled.layers[0].fft_samples = 12;
    rejects([&] { rc::solve(sampled, .83, 27., 3); });
    rejects([&] { rc::solve_conical(sampled, .83, 27., 41., 3); });

    // Derived epsilon/reciprocal overflow is a numerical failure, not invalid input.
    for (double extreme : {1e200, 1e-200}) {
        auto overflow = sampled;
        overflow.layers[0].fft_samples = 17;
        overflow.layers[0].n_values[0] = extreme;
        for (auto pol : {rc::Polarization::TE, rc::Polarization::TM}) {
            if (extreme < 1. && pol == rc::Polarization::TE) continue; // TE does not use 1/epsilon.
            const auto result = rc::solve(overflow, .83, 27., 3, pol);
            for (double v : result.DEr) CHECK(std::isnan(v));
            for (double v : result.DEt) CHECK(std::isnan(v));
        }
        const auto result = rc::solve_conical(overflow, .83, 27., 41., 3);
        for (auto j : result.Jr) CHECK(std::isnan(j.ss.real()) && std::isnan(j.pp.real()));
    }

    sampled.layers[0].fft_samples = 257;
    sampled.layers.push_back(rc::Layer::grating(.09, {0., .3, .9}, {1.7, 1.2}, rc::FourierMethod::FFT, 1009));
    const std::vector<rc::Point> points{{.83, 27.}, {.91, -15.}};
    set_num_threads(1);
    const auto serial = rc::sweep(sampled, points, 3, rc::Polarization::TM);
    set_num_threads(3);
    const auto parallel = rc::sweep(sampled, points, 3, rc::Polarization::TM);
    set_num_threads(1);
    for (std::size_t j = 0; j < serial.size(); ++j) {
        CHECK(serial[j].R == parallel[j].R && serial[j].T == parallel[j].T);
        CHECK(serial[j].DEr == parallel[j].DEr && serial[j].DEt == parallel[j].DEt);
    }
}

static void json_interface() {
    const std::string base = R"({"mode":"planar","polarization":"TM","M":2,"wavelength":0.83,"theta_deg":27,
        "stack":{"period":0.9,"nI":1.2,"nII":1.5,"layers":[{"d":0.2,"edges":[-0.1,0.2,0.8],
        "n_values":[[2,0.02],1.2],"fourier_method":"fft","fft_samples":17}]}})";
    const auto request = rc::json_detail::read_request(json::parse(base));
    CHECK(request.stack.layers[0].fourier_method == rc::FourierMethod::FFT);
    CHECK(request.stack.layers[0].fft_samples == 17);
    const auto reply = json::parse(rc::solve_json(base));
    CHECK(reply.at("ok").as_bool());
    auto bad = base;
    bad.replace(bad.find("\"fft_samples\":17"), 16, "\"fft_samples\":8");
    auto error = json::parse(rc::solve_json(bad));
    CHECK(!error.at("ok").as_bool());
    CHECK(error.at("where").as_string() == "stack.layers[0].fft_samples");
    auto overflow = base;
    overflow.replace(overflow.find("[2,0.02]"), std::string("[2,0.02]").size(), "1e200");
    const auto failed = json::parse(rc::solve_json(overflow));
    CHECK(failed.at("ok").as_bool());
    CHECK(failed.at("results").at(0).at("DEr").at(0).is_null());
}

int main() {
    coefficients(); solvers(); json_interface();
    std::printf("FFT tests: %d failures\n", failures);
    return failures ? 1 : 0;
}
