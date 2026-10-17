// Sampled material Fourier coefficients, using the bundled BSD-3-Clause pocketfft.
#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "matrix.hpp"
#include "detail/pocketfft.hpp"

namespace stablercwa {
namespace rcwa {

enum class FourierMethod { Analytic, FFT };
inline constexpr int kMaxFFTSamples = 4194304;

inline void validate_fourier_settings(FourierMethod method, int samples) {
    if (method != FourierMethod::Analytic && method != FourierMethod::FFT)
        throw std::invalid_argument("rcwa: invalid Fourier method");
    if (samples < 0 || samples > kMaxFFTSamples)
        throw std::invalid_argument("rcwa: fft_samples must be 0..4194304 (0 for automatic)");
}

// Match gui/spectral1d.py: independent coefficients -2M..2M require N > 4M.
inline int resolve_fft_samples(int M, int samples = 0) {
    validate_fourier_settings(FourierMethod::FFT, samples);
    if (M < 0) throw std::invalid_argument("rcwa: FFT truncation order M must be nonnegative");
    if (M > (kMaxFFTSamples - 1) / 4)
        throw std::invalid_argument("rcwa: FFT truncation requires more than 4194304 samples");
    if (samples == 0) {
        const long long target = (std::max)(8192LL, 64LL * (2LL * M + 1));
        if (target > kMaxFFTSamples)
            throw std::invalid_argument("rcwa: automatic FFT exceeds 4194304 samples");
        samples = 1;
        while (samples < target) samples *= 2;
    }
    if (samples <= 4 * M)
        throw std::invalid_argument("rcwa: fft_samples must be at least 4*M+1");
    return samples;
}

// c[h+hmax] multiplies exp(+i*2*pi*h*x/period). Sample at absolute cell
// midpoints; correct both the origin and half-cell phase after the forward FFT.
// values may be complex epsilon or its independently sampled reciprocal.
inline Vector fourier_coefficients_fft(const std::vector<double>& edges, const Vector& values,
                                       double period, int hmax, int samples) {
    validate_fourier_settings(FourierMethod::FFT, samples);
    if (hmax < 0 || samples <= 2LL * hmax)
        throw std::invalid_argument("rcwa: FFT sample count must exceed 2*hmax");
    if (!std::isfinite(period) || period <= 0 || values.empty() || edges.size() != values.size() + 1)
        throw std::invalid_argument("rcwa: FFT needs a positive period, K values and K+1 edges");
    for (std::size_t j = 0; j < edges.size(); ++j)
        if (!std::isfinite(edges[j]) || (j && edges[j] <= edges[j - 1]))
            throw std::invalid_argument("rcwa: FFT edges must be finite and strictly increasing");
    if (std::abs((edges.back() - edges.front()) - period) > 1e-10 * period)
        throw std::invalid_argument("rcwa: FFT edges must span exactly one period");
    for (const cplx& v : values)
        if (!std::isfinite(v.real()) || !std::isfinite(v.imag()))
            throw std::invalid_argument("rcwa: FFT material values must be finite");

    Vector spectrum(static_cast<std::size_t>(samples));
    std::size_t segment = 0;
    for (int j = 0; j < samples; ++j) {
        const double u = (static_cast<double>(j) + 0.5) / samples * period;
        while (segment + 1 < values.size() && u >= edges[segment + 1] - edges.front()) ++segment;
        spectrum[j] = values[segment];
    }
    const stablercwa_pocketfft::shape_t shape{static_cast<std::size_t>(samples)}, axes{0};
    const stablercwa_pocketfft::stride_t stride{static_cast<std::ptrdiff_t>(sizeof(cplx))};
    // One FFT thread per call: layer/sweep parallelism belongs to StableRCWA.
    stablercwa_pocketfft::c2c(shape, stride, stride, axes, true, spectrum.data(), spectrum.data(), 1.0 / samples, 1);
    double origin = std::fmod(edges.front(), period);
    if (origin < 0) origin += period;
    const double offset = origin / period + 0.5 / samples;
    Vector c(static_cast<std::size_t>(2 * hmax + 1));
    for (int h = -hmax; h <= hmax; ++h) {
        const int bin = h < 0 ? samples + h : h;
        const double arg = -2.0 * kPi * h * offset;
        c[h + hmax] = spectrum[bin] * cplx{std::cos(arg), std::sin(arg)};
    }
    return c;
}

} // namespace rcwa
} // namespace stablercwa
