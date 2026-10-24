# MatrixEigenValueAndEigenVector — oneMKL eigensolver drop-in

[中文说明](README.zh-CN.md)

`MatrixEigenValueAndEigenVector_MKL.cpp` is a single source file for C++ DLL projects on Windows x64. It computes
all eigenvalues and right eigenvectors of a general real or complex matrix with Intel oneMKL's LAPACK
(`dgeev` for real matrices, `zgeev` for complex ones). Add it to the project in place of an existing implementation
of the two functions below; their declarations stay unchanged.

```cpp
int MatrixEigenValueAndEigenVector(double** A1, std::complex<double>* EigenValue, double** EigenVector, int mn);
int MatrixEigenValueAndEigenVector(std::complex<double>** A1, std::complex<double>* EigenValue,
                                   std::complex<double>** EigenVector, int mn);
```

It is separate from the header-only StableRCWA library, which still needs nothing but the C++ standard library.
The file does not include any MKL header and needs no other StableRCWA file.

## Required oneMKL version: 2025.0.1 or newer

| oneMKL | Status |
| --- | --- |
| 2026.2.0 | tested; installed by default by `install_mkl.bat` |
| 2025.0.1, 2025.1.0, 2025.2.0, 2025.3.1 | tested, fine |
| **2025.0.0, 2024.2.2** | **do not use**: `dgeev` is broken on AMD processors under Windows |

What goes wrong in 2024.2.2 and 2025.0.0 (measured on an AMD Ryzen 9 5950X under Windows):
- `dgeev` writes n² values into the left-eigenvector array even when it was told not to compute left eigenvectors.
- It also writes past the end of the workspace size it reported itself, so the calling program's memory is corrupted.
- Single-threaded `LAPACKE_dgeev` can hang.

The same bug shows up in other software linked to these versions: SciPy linked to oneMKL 2025.0.0 returns NaN or wrong
eigenvalues for real matrices and then crashes. `zgeev` (complex matrices) works in every version listed.
Intel's 2025.0.1 release notes list "fixed issues in some BLAS functions on AMD hardware in Windows",
and none of the versions from 2025.0.1 on show the problem.

As a safeguard, the file reads MKL's version string on its first call. If the build is older than 2025.0.1 (build date
20241031), or the version cannot be read, real matrices are solved with `zgeev` instead of `dgeev`. That is safe but
slower and has the two limitations listed under "Accuracy" below. Use 2025.0.1 or newer.

## One-click install of the MKL runtime

Double-click `install_mkl.bat`, or run it from `cmd`:

```bat
standalone\install_mkl.bat
standalone\install_mkl.bat -Version 2025.3.1
```

The script (`install_mkl.ps1`, Windows PowerShell 5.1 or later):
- downloads Intel's official `mkl` and `intel-openmp` wheels for Windows x64 from PyPI, about 200 MB;
- checks each download against the SHA-256 digest that PyPI publishes, and for the default version also against
  digests pinned in the script;
- extracts the 9 runtime DLLs (about 420 MB) into `standalone/mkl/bin` and the Intel license texts into
  `standalone/mkl/licenses`;
- replaces a previous install only after everything has been downloaded and checked, so a failed download leaves
  the old one working;
- refuses versions older than 2025.0.1.

It needs no administrator rights and no Python, and it installs nothing system-wide. The import library
`standalone/mkl/lib/mkl_rt.lib` is already in the repository; it links against `mkl_rt.3.dll`, the name used from
oneMKL 2026 on. For a 2025.x runtime (`mkl_rt.2.dll`), change the first line of `mkl/lib/mkl_rt.def` and regenerate the
import library in a Visual Studio developer prompt: `lib /def:mkl_rt.def /machine:x64 /out:mkl_rt.lib`.
The script prints this reminder when it is needed.

The runtime can also come from an Intel oneAPI installation: the DLLs are in
`C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin`, `libiomp5md.dll` is in
`C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin`, and `mkl_rt.lib` is in `mkl\latest\lib`.

## Adding it to a Visual Studio project

1. Remove the old implementation of the two functions and add `MatrixEigenValueAndEigenVector_MKL.cpp` to the project.
   If the project uses precompiled headers, either add `#include "pch.h"` as the first line of the file or set the file
   to "Not Using Precompiled Headers". Do not include `mkl.h` before this file: its declarations of the same functions
   use different types.
2. Project properties → Linker → General → Additional Library Directories: add `standalone\mkl\lib`.
   Linker → Input → Additional Dependencies: add `mkl_rt.lib`.
3. Make the DLLs in `standalone\mkl\bin` available at run time, for example with a post-build event:

   ```bat
   xcopy /y /d "<path to standalone>\mkl\bin\*.dll" "$(OutDir)"
   ```

   Windows looks for them next to the host program's `.exe` and on `PATH`, not next to your DLL. If your DLL is loaded
   by another program (MATLAB, Python, LabVIEW, …), put the MKL DLLs next to that program's `.exe` or add
   `standalone\mkl\bin` to `PATH`.

The file is UTF-8 with a BOM, so MSVC reads it correctly without `/utf-8`. It builds with the default C++14 of MSVC
(any C++11 compiler should do); `/W4 /WX` builds cleanly, also with `<windows.h>` included first.

### Runtime DLLs (`standalone/mkl/bin`, oneMKL 2026.2.0)

| DLL | Size | Needed |
| --- | ---: | --- |
| `mkl_rt.3.dll` | 28 MB | always (entry point; `mkl_rt.lib` links to it) |
| `mkl_core.3.dll` | 73 MB | always |
| `mkl_intel_thread.3.dll` | 41 MB | always (threading layer) |
| `libiomp5md.dll` | 1.6 MB | always (Intel OpenMP, used by the threading layer) |
| `mkl_def.3.dll` | 40 MB | always (generic kernels; loaded on AMD processors too) |
| `mkl_mc3.3.dll`, `mkl_avx2.3.dll`, `mkl_avx512.3.dll`, `mkl_avx10.3.dll` | 50, 50, 76, 66 MB | on Intel processors MKL picks one of these by instruction set |

- On the AMD test machine the first five were enough (checked in an empty folder with only these DLLs and the system
  directories on `PATH`). Intel processors were not tested, which is why the script installs all nine.
- **A missing DLL is fatal.** It is not reported through the return value: on the first calculation MKL prints, for
  example, `Intel oneMKL FATAL ERROR: Cannot load mkl_def.3.dll.` and the whole process exits. Run one calculation on the
  target machine after deploying.
- The Visual C++ runtime (`vcruntime140.dll`, `msvcp140.dll`) is needed as for any MSVC-built DLL.
- The two parts come under different Intel licenses; both texts end up in `mkl/licenses`:
  - oneMKL (the `mkl_*.dll` files): Intel Simplified Software License, which allows redistributing the DLLs unchanged
    together with its copyright notice and terms.
  - Intel OpenMP (`libiomp5md.dll`): Intel End User License Agreement for Developer Tools. It allows distributing the
    files it calls Redistributables only as part of your own product, in executable form and under terms that prohibit
    reverse engineering. Read `intel-openmp-LICENSE.txt` before you ship `libiomp5md.dll`.

## Interface and conventions

- `mn` is the order n. `A1` and `EigenVector` are arrays of n row pointers with n elements per row; `EigenValue` has n
  elements. The caller allocates all memory.
- `A1[i][j]` is row i, column j. `A1` is not modified. The outputs may share storage with `A1`: the input is read
  completely before anything is written.
- Eigenvector k is column k: `EigenVector[i][k]` is its i-th component. `A v_k = λ_k v_k` and `||v_k||_2 = 1`.
- Eigenvalues are sorted by real part, largest first, then by imaginary part, largest first. The eigenvector columns
  follow the same order.
- Each eigenvector is multiplied by a unit-modulus factor so that its largest component (the lowest row index on a tie)
  is real and positive.
- **Real version** (`double**` output). Complex eigenvectors are packed as in LAPACK `dgeev`:
  - a real eigenvalue (imaginary part exactly 0) has its real eigenvector in column k;
  - a complex pair λ_k, λ_{k+1} = conj(λ_k), positive imaginary part first, exactly conjugate and always adjacent:
    column k = x and column k+1 = y; the eigenvector of λ_k is x + i·y, that of λ_{k+1} is x − i·y,
    and `||x + i·y||_2 = 1`;
  - sorting treats a real eigenvalue or a conjugate pair as one unit (real part first, then the positive imaginary part),
    so a pair is never split by another eigenvalue.

### Return values

| Value | Meaning |
| --- | --- |
| 0 | success |
| −1 | invalid argument: `mn < 1` or a null pointer (the outputs are not touched) |
| −2 | `A1` contains NaN or Inf |
| −3 | out of memory, or another unexpected exception |
| −4 | the computed eigenvalues or eigenvectors contain NaN/Inf (only for extreme element magnitudes) |
| k > 0 | LAPACK's QR iteration did not converge (k is the `info` returned by LAPACK) |

After −2, −3, −4 or k > 0, `EigenValue` and `EigenVector` are filled with NaN.

### Minimal example

```cpp
#include <complex>
#include <cstdio>
#include <vector>
typedef std::complex<double> cplx;

int MatrixEigenValueAndEigenVector(double** A1, cplx* EigenValue, double** EigenVector, int mn);
int MatrixEigenValueAndEigenVector(cplx** A1, cplx* EigenValue, cplx** EigenVector, int mn);

int main() {
    const int n = 3;
    std::vector<double> b = {0, -1, 0,
                             1,  0, 0,
                             0,  0, 2};                 // eigenvalues 2, +i, -i
    std::vector<cplx> w(n);
    std::vector<double> v(n * n);
    std::vector<double*> B(n), V(n);                    // row pointers: B[i][j] is row i, column j
    for (int i = 0; i < n; ++i) { B[i] = &b[i * n]; V[i] = &v[i * n]; }
    const int status = MatrixEigenValueAndEigenVector(B.data(), w.data(), V.data(), n);
    if (status != 0) { std::printf("failed: %d\n", status); return 1; }
    for (int k = 0; k < n; ++k) {
        if (w[k].imag() == 0.0) {                       // real eigenvalue: column k is the eigenvector
            std::printf("lambda_%d = %g        v = (%.4f, %.4f, %.4f)\n", k, w[k].real(), V[0][k], V[1][k], V[2][k]);
        } else {                                        // conjugate pair: columns k, k+1 are x, y; vectors x +- i*y
            std::printf("lambda_%d = %g%+gi  v = x + i*y,  lambda_%d = %g%+gi  v = x - i*y\n",
                        k, w[k].real(), w[k].imag(), k + 1, w[k + 1].real(), w[k + 1].imag());
            ++k;
        }
    }
    return 0;
}
```

Output with this file and oneMKL 2026.2.0:

```text
lambda_0 = 2        v = (0.0000, 0.0000, 1.0000)
lambda_1 = 0+1i  v = x + i*y,  lambda_2 = 0-1i  v = x - i*y
```

## Threads

The macro `MATRIX_EIG_THREADS` at the top of the file sets how many threads one call may use; the function interface
does not change.
- `0` (default): MKL's own thread count (the number of physical cores unless `MKL_NUM_THREADS` says otherwise). If
  another call is already running, this call uses 1 thread, so a host that already runs several matrices in parallel
  does not oversubscribe the cores.
- `n >= 1`: at most n threads per call.

Only the calling thread's MKL setting is changed (`MKL_Set_Num_Threads_Local`) and it is restored afterwards. The
functions are reentrant and may be called from several threads at once.

Results agree with other solvers to rounding error, not bit for bit. MKL's own results also differ in the last bits
between thread counts and processors; for repeatable results set `MKL_CBWR=AVX2` and a fixed `MATRIX_EIG_THREADS`.

## Performance

Random matrices, seconds, best of several runs. AMD Ryzen 9 5950X (16 cores), Windows, MSVC 14.34 `/O2`,
oneMKL 2026.2.0 with its default 16 threads.

| n | StableRCWA `eig_qr`, 8 threads | this file, complex | this file, real | NumPy 1.26.4 (OpenBLAS), complex | NumPy, real |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | 0.141 | 0.082 | 0.045 | 0.217 | 0.135 |
| 578 | 0.657 | 0.443 | 0.215 | 0.837 | 0.467 |
| 882 | 1.91 | 0.926 | 0.457 | 1.72 | 0.908 |
| 1250 | 4.98 | 1.87 | 0.893 | 3.07 | 1.59 |
| 2000 | 20.1 | 5.82 | 2.52 | 8.50 | 3.59 |
| 4000 | 199 | 36.2 | 16.0 | 57.2 | 21.8 |

- Against `eig_qr` (the header-only library's QR solver; measured on 2026-10-21 through a drop-in build of the same
  algorithm): complex matrices are 1.5 to 5.5 times faster, real matrices 3.1 to 12.4 times faster.
  `eig_qr` solves real matrices as complex ones, so its real and complex times are the same.
- Real matrices take `dgeev`, which is about twice as fast as `zgeev`.
- Against `numpy.linalg.eig` with OpenBLAS: 1.5 to 2.6 times faster for complex matrices, 1.4 to 3.0 times for real ones.
  SciPy linked to MKL runs the same `zgeev` and is as fast as this file for complex matrices.
- The first call in a process also loads MKL (0.01 to 0.25 s measured).
- There is no 46340-order limit: the file calls the 64-bit-integer (ILP64) `zgeev_64_` and `dgeev_64_`, which every
  oneMKL from 2024.2.2 to 2026.2.0 exports, whatever `MKL_INTERFACE_LAYER` is set to. Memory is the practical limit:
  one call needs about 80·n² bytes for a complex matrix and 64·n² bytes for a real one, including the caller's arrays.

## Accuracy

- `dgeev` and `zgeev` use unitary transformations and are backward stable. Over all eigenpairs of 2000 × 2000 random
  complex, random real, highly non-normal (Grcar) and badly scaled matrices, the largest relative residual
  `||A v − λ v|| / (||A||_F ||v||)` was at most 5.1e-16.
- Real matrices are reduced to real Schur form, so real eigenvalues have an imaginary part of exactly 0 and conjugate
  pairs are exact.
- LAPACK balances the matrix first. For badly scaled matrices (entries spanning many orders of magnitude) this gives
  much more accurate eigenvalues than an unbalanced QR iteration such as `eig_qr`.
- Only when `dgeev` is not used (MKL older than 2025.0.1) are real matrices solved as complex ones and then paired into
  conjugates. Two limitations apply then: triple or higher defective real roots can leave residuals around 1e-12, and
  two or more defective blocks on the same real eigenvalue can give a numerically singular eigenvector basis.

## Tests

`standalone/tests/test_standalone_mkl.cpp` checks convergence and residuals, sorting, phase normalisation, the real
packing format, conjugate pairing, return codes, NaN filling, aliasing of input and output and concurrent calls.
It needs oneMKL, so it is not built by default or in CI. On Windows with MSVC, after running `install_mkl.bat`:

```bat
cmake -S . -B build -DSTABLERCWA_BUILD_STANDALONE_MKL=ON
cmake --build build --config Release --target test_standalone_mkl
build\Release\test_standalone_mkl.exe
```

Use `-DSTABLERCWA_MKL_RT_LIB=<path>` to link another oneMKL import library.

Verified on Windows (MSVC 14.34, AMD Ryzen 9 5950X):
- the test passes (42125 checks) with oneMKL 2025.3.1 and 2026.2.0, built both through the CMake option above and
  directly with `cl`; an earlier version of the test also passed with `MKL_INTERFACE_LAYER` set to `LP64` or `ILP64`;
- the DLL builds with `/W4 /WX`, exports only the two functions, and survives 30 rounds of
  `LoadLibrary` → both overloads at n = 300 → `FreeLibrary`;
- the runtime installed by `install_mkl.bat` into an empty folder is complete;
- with oneMKL 2025.0.0, where the file falls back to `zgeev`, 27 checks fail, all in the two degenerate families
  described under "Accuracy". This is another reason to use 2025.0.1 or newer.

## License

The drop-in, the install script and the import library are part of StableRCWA and are MIT-licensed. The runtime is
Intel's and is not in this repository: oneMKL is covered by the Intel Simplified Software License and Intel OpenMP by
the Intel End User License Agreement for Developer Tools. `install_mkl.bat` puts both license texts in `mkl/licenses`.
