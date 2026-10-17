# pocketfft (vendored)

Upstream: https://github.com/mreineck/pocketfft (branch `cpp`)

Pinned commit: `c90e55b3d529f8efa40ed01a20de22405f45fc65`

`pocketfft_hdronly.h` and `LICENSE.md` are unmodified upstream files.
License: BSD-3-Clause; retain the license when redistributing source or binaries.
No download, separate build, or linked FFT library is required.

StableRCWA uses complex-to-complex forward transforms in `fourier.hpp`, with
explicit normalization and a single FFT thread per call. Arbitrary sample counts
(including non-powers of two) are supported.

`../pocketfft.hpp` isolates the vendored header's namespace, include guard and
configuration macros from applications that also use pocketfft. Internal FFT
threading and caching are disabled; StableRCWA owns layer/sweep parallelism.
