// An application includes its configured pocketfft before StableRCWA.
#define POCKETFFT_NAMESPACE application_fft
#define POCKETFFT_CACHE_SIZE 2
#define POCKETFFT_NO_VECTORS
// The simulated application silences the upstream header's own warnings, as an application would.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4127 4324)
#elif defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#include "stablercwa/detail/pocketfft/pocketfft_hdronly.h"
#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
#include "stablercwa/matrix.hpp"
#define min(a, b) windows_min_macro(a, b)
#define max(a, b) windows_max_macro(a, b)
#include "stablercwa/fourier.hpp"

#if !defined(POCKETFFT_HDRONLY_H) || POCKETFFT_CACHE_SIZE != 2 || !defined(POCKETFFT_NO_VECTORS)
#error Application pocketfft configuration was changed
#endif
#if !defined(min) || !defined(max) || defined(POCKETFFT_NO_MULTITHREADING)
#error Application macro state was changed
#endif
#undef min
#undef max

int main() {
    // POCKETFFT_NAMESPACE must still name the application's namespace.
    const POCKETFFT_NAMESPACE::shape_t shape{1}, axes{0};
    const POCKETFFT_NAMESPACE::stride_t stride{sizeof(stablercwa::cplx)};
    stablercwa::cplx value{2., .3};
    application_fft::c2c(shape, stride, stride, axes, true, &value, &value, 1.);
    const auto c = stablercwa::rcwa::fourier_coefficients_fft({0., 1.}, {value}, 1., 0, 1);
    return std::abs(c[0] - stablercwa::cplx{2., .3}) < 1e-14 ? 0 : 1;
}
