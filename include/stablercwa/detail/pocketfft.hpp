// Isolate the bundled pocketfft from an application's own pocketfft version and
// configuration. The upstream header remains unmodified, including its license.
#pragma once

#pragma push_macro("min")
#pragma push_macro("max")
#pragma push_macro("POCKETFFT_HDRONLY_H")
#pragma push_macro("POCKETFFT_NAMESPACE")
#pragma push_macro("POCKETFFT_CACHE_SIZE")
#pragma push_macro("POCKETFFT_NO_MULTITHREADING")
#pragma push_macro("POCKETFFT_NO_VECTORS")
#undef min
#undef max
#undef POCKETFFT_HDRONLY_H
#undef POCKETFFT_NAMESPACE
#undef POCKETFFT_CACHE_SIZE
#undef POCKETFFT_NO_MULTITHREADING
#undef POCKETFFT_NO_VECTORS
#define POCKETFFT_NAMESPACE stablercwa_pocketfft
#define POCKETFFT_CACHE_SIZE 0
#define POCKETFFT_NO_MULTITHREADING
// Upstream code that is clean in its own build trips a few warning levels used by applications; keep them
// local to the bundled header instead of editing it.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4127)  // conditional expression is constant
#elif defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#include "pocketfft/pocketfft_hdronly.h"
#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
#pragma pop_macro("POCKETFFT_NO_VECTORS")
#pragma pop_macro("POCKETFFT_NO_MULTITHREADING")
#pragma pop_macro("POCKETFFT_CACHE_SIZE")
#pragma pop_macro("POCKETFFT_NAMESPACE")
#pragma pop_macro("POCKETFFT_HDRONLY_H")
#pragma pop_macro("max")
#pragma pop_macro("min")
