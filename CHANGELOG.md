# Changelog

All notable changes to StableRCWA are listed here. Version numbers follow [Semantic Versioning](https://semver.org/);
while the version is 0.x, the interfaces may still change.

## [Unreleased]

- Add `standalone/MatrixEigenValueAndEigenVector_MKL.cpp`, a single-file drop-in for C++ DLL projects that computes
  eigenvalues and eigenvectors with oneMKL (`dgeev` for real and `zgeev` for complex matrices, 64-bit integer
  interface). It needs oneMKL 2025.0.1 or newer and checks the version at run time to avoid the `dgeev` bug of
  2024.2.2 and 2025.0.0 on AMD processors under Windows. Add `standalone/install_mkl.bat`, which downloads the oneMKL
  runtime from PyPI, and an opt-in CMake test (`STABLERCWA_BUILD_STANDALONE_MKL`).
- Bundle BSD-3-Clause pocketfft at commit `c90e55b3d529f8efa40ed01a20de22405f45fc65` without external build/link dependencies.
- Add per-layer FFT material coefficients for C++ 1-D TE/TM and conical solvers, including fields and sweeps;
  retain analytic coefficients by default. Match Python midpoint sampling, origin phase, normalization, independent
  ε/1/ε transforms, automatic sample counts and support for non-power-of-two counts.
- Accept `fourier_method` and `fft_samples` in 1-D JSON layers and route eligible GUI FFT models to C++.
  Include vendored `.h` files in GUI rebuild detection.
- Add direct-DFT, analytic-convergence, input-validation and C++/Python FFT regression coverage.
- Isolate bundled pocketfft from application namespaces/configuration and preserve the NaN numerical-failure
  result when finite refractive indices overflow during material-operator construction.

## [0.1.0] - 2026-10-17

First public release.

### C++ library (header-only, C++17, standard library only)

- 1-D RCWA for TE and TM polarization, with S-matrix propagation, absorbing (complex-index) layers and
  an analytic fast path for homogeneous layers.
- Exact handling of modes with zero propagation constant in finite planar layers.
- Conical incidence: a full 2×2 Jones matrix and four polarization-channel efficiencies per order; layers whose
  reduced eigenproblem is degenerate switch automatically to a Schur basis.
- 2-D crossed gratings on rectangular cells, with Li's Fourier factorization rules and separate truncation in x and y.
- Six-component near fields and phase maps for 1-D (TE/TM and conical) and 2-D (x-z and y-z slices),
  sampled repeatedly from one solve; absolute, incident-relative and custom phase references, weak-field masks, unwrapping.
- Eigen solvers: unitary Hessenberg-QR iteration (default, after EISPACK CORTH/COMQR2) and the elimination-based
  LR iteration (after EISPACK COMHES/COMLR2, kept for comparison).
- Supporting numerics: LU with and without pivoting, Householder QR, one-sided Jacobi SVD and condition numbers.
- Multi-threading with the standard library only: parallel wavelength/angle sweeps, parallel layers, and a parallel
  `eig_qr` for large matrices that is bitwise identical to the single-threaded result when FMA contraction is off.
- Optional, experimental CUDA backend (cuSOLVER/cuBLAS) for the eigen decomposition.
- Numerical failures are reported (NaN results with a status or eigen-solver code) instead of thrown; invalid input
  throws `std::invalid_argument`.
- CMake target `stablercwa` with alias `StableRCWA::stablercwa`; options `STABLERCWA_BUILD_EXAMPLES` and `STABLERCWA_BUILD_TESTS`.

### Command-line programs

- `rcwa_json`: JSON in, JSON out, for the `planar`, `conical` and `periodic2d` modes, with parallel sweeps over several points.
- `rcwa_fields`: efficiencies, six-component fields and phase from one solve, written as JSON plus NumPy `.npy` files.
- Demonstration programs `eig_minimal`, `rcwa_demo` and `conical_demo`.

### Graphical interface

- Local web GUI served on `127.0.0.1`, for modelling, solving and viewing fields without writing code.
- Uses the C++ solver for supported 1-D models and falls back to the bundled Python solver, with the reason shown.
- The Python solver also covers finite structures with lateral PML, uniaxial layers and FFT-sampled Fourier
  coefficients (1-D); magnetic media and general 6×6 constitutive tensors are included as experimental Python code.

### Validation

- 21 C++ test programs and the GUI's Python test suite.
- Independent comparisons against COMSOL finite-element models (2-D for conical incidence, 3-D for crossed gratings),
  summarized in `docs/validation.zh-CN.md`.
