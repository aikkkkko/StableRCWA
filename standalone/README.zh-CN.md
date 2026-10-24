# MatrixEigenValueAndEigenVector：oneMKL 本征分解，直接替换的单文件

[English](README.md)

`MatrixEigenValueAndEigenVector_MKL.cpp` 是给 Windows x64 上 C++ DLL 项目用的单个源文件。它调用 Intel oneMKL 的
LAPACK（实矩阵用 `dgeev`，复矩阵用 `zgeev`），求一般实矩阵、复矩阵的全部本征值和右本征向量。把它加进项目，
代替原来这两个函数的实现即可，原来头文件里的声明不用改：

```cpp
int MatrixEigenValueAndEigenVector(double** A1, std::complex<double>* EigenValue, double** EigenVector, int mn);
int MatrixEigenValueAndEigenVector(std::complex<double>** A1, std::complex<double>* EigenValue,
                                   std::complex<double>** EigenVector, int mn);
```

它与只有头文件的 StableRCWA 库是分开的：库本身仍然只依赖 C++ 标准库。这个文件不包含任何 MKL 头文件，
也不需要 StableRCWA 的其他文件。

## 需要的 oneMKL 版本：2025.0.1 或更新

| oneMKL | 情况 |
| --- | --- |
| 2026.2.0 | 测过；`install_mkl.bat` 默认装这个版本 |
| 2025.0.1、2025.1.0、2025.2.0、2025.3.1 | 测过，正常 |
| **2025.0.0、2024.2.2** | **不要用**：AMD 处理器、Windows 上 `dgeev` 有毛病 |

2024.2.2 和 2025.0.0 的问题（在 AMD Ryzen 9 5950X、Windows 上实测）：
- 不要左本征向量时，`dgeev` 仍往左本征向量的数组里写 n² 个数。
- 它还写出自己报的工作区的末尾，调用程序的内存被改坏。
- 单线程时 `LAPACKE_dgeev` 会卡死。

链接这两个版本的别的软件也会碰到：链接 oneMKL 2025.0.0 的 SciPy，对实矩阵求本征值时得到 NaN 或错的结果，
随后进程直接退出。`zgeev`（复矩阵）在上表所有版本上都正常。Intel 2025.0.1 的发布说明写着
“修正了 AMD 硬件、Windows 上若干 BLAS 函数的问题”，2025.0.1 及以后的版本都没有再出现。

为防万一，文件第一次调用时读 MKL 的版本字符串。构建日期早于 2025.0.1（20241031）或读不出来时，实矩阵改走
`zgeev`：这样安全，但慢一些，而且有下面“精度”一节说的两条局限。请用 2025.0.1 或更新的版本。

## 一键安装 MKL 运行库

双击 `install_mkl.bat`，或在 `cmd` 里运行：

```bat
standalone\install_mkl.bat
standalone\install_mkl.bat -Version 2025.3.1
```

这个脚本（`install_mkl.ps1`，Windows PowerShell 5.1 或更新）：
- 从 PyPI 下载 Intel 官方发布的 `mkl` 和 `intel-openmp` 两个 Windows x64 wheel，约 200 MB；
- 用 PyPI 公布的 SHA-256 核对每个下载，默认版本还要与脚本里写死的 SHA-256 一致；
- 把运行时要的 9 个 DLL（约 420 MB）解到 `standalone/mkl/bin`，Intel 的许可证解到 `standalone/mkl/licenses`；
- 全部下载、核对完才替换原来装的那一套，下载中途失败时原来的照样能用；
- 拒绝 2025.0.1 以前的版本。

不需要管理员权限，不需要 Python，也不往系统里装任何东西。导入库 `standalone/mkl/lib/mkl_rt.lib` 已经在仓库里，
它链接的是 `mkl_rt.3.dll`，这是 oneMKL 2026 起的名字。用 2025.x 的运行库（`mkl_rt.2.dll`）时，要把
`mkl/lib/mkl_rt.def` 第一行改成对应的名字，在 Visual Studio 开发者命令行里重新生成导入库：
`lib /def:mkl_rt.def /machine:x64 /out:mkl_rt.lib`。需要这样做时脚本会提示。

运行库也可以从 Intel oneAPI 的安装里拿：MKL 的 DLL 在 `C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin`，
`libiomp5md.dll` 在 `C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin`，`mkl_rt.lib` 在 `mkl\latest\lib`。

## 加进 Visual Studio 项目

1. 删掉原来这两个函数的实现，把 `MatrixEigenValueAndEigenVector_MKL.cpp` 加进项目。
   项目用了预编译头时，在文件第一行加 `#include "pch.h"`，或者把这个文件设成“不使用预编译头”。
   不要在这个文件之前包含 `mkl.h`：它对同名函数的声明用的是别的类型。
2. 项目属性 → 链接器 → 常规 → 附加库目录：加上 `standalone\mkl\lib`；输入 → 附加依赖项：加上 `mkl_rt.lib`。
3. 让 `standalone\mkl\bin` 里的 DLL 在运行时能被找到，例如后期生成事件：

   ```bat
   xcopy /y /d "<standalone 的路径>\mkl\bin\*.dll" "$(OutDir)"
   ```

   Windows 在宿主程序 `.exe` 所在的文件夹和 `PATH` 里找它们，不在你的 DLL 自己的文件夹里找。你的 DLL 是被别的
   程序（MATLAB、Python、LabVIEW 等）加载的话，把 MKL 的 DLL 放在那个程序的 `.exe` 旁边，或者把
   `standalone\mkl\bin` 加进 `PATH`。

文件是带 BOM 的 UTF-8，MSVC 不加 `/utf-8` 也能正确读。MSVC 默认的 C++14 即可（C++11 及以上的编译器都应当可以），
`/W4 /WX` 零警告，先包含 `<windows.h>` 也没关系。

### 运行时要带的 DLL（`standalone/mkl/bin`，oneMKL 2026.2.0）

| DLL | 大小 | 什么时候要 |
| --- | ---: | --- |
| `mkl_rt.3.dll` | 28 MB | 总要（入口，`mkl_rt.lib` 链接的就是它） |
| `mkl_core.3.dll` | 73 MB | 总要 |
| `mkl_intel_thread.3.dll` | 41 MB | 总要（多线程层） |
| `libiomp5md.dll` | 1.6 MB | 总要（Intel OpenMP，多线程层要它） |
| `mkl_def.3.dll` | 40 MB | 总要（通用内核，AMD 处理器上也会加载） |
| `mkl_mc3.3.dll`、`mkl_avx2.3.dll`、`mkl_avx512.3.dll`、`mkl_avx10.3.dll` | 50、50、76、66 MB | Intel 处理器上 MKL 按指令集挑其中一个 |

- 在 AMD 测试机上，只放前五个就够了（在只放了这几个 DLL、`PATH` 里只有系统目录的空文件夹里实测）。
  Intel 处理器没有测过，所以脚本把九个都装上。
- **缺 DLL 会让进程直接退出**，不会变成返回值：第一次计算时 MKL 打印例如
  `Intel oneMKL FATAL ERROR: Cannot load mkl_def.3.dll.`，随后整个进程退出。部署之后要在目标机器上实际算一次。
- 另外要有 VC++ 运行库（`vcruntime140.dll`、`msvcp140.dll`），与别的 MSVC 编出的 DLL 一样。
- 两部分的许可证不同，原文都在 `mkl/licenses` 里：
  - oneMKL（`mkl_*.dll`）：Intel Simplified Software License，允许原样再分发这些 DLL，条件是附上它的版权声明和条款。
  - Intel OpenMP（`libiomp5md.dll`）：Intel End User License Agreement for Developer Tools。它所说的
    Redistributables 只能作为你自己产品的一部分、以可执行形式分发，而且分发时的条款要禁止反向工程。分发
    `libiomp5md.dll` 之前请读 `intel-openmp-LICENSE.txt`。

## 接口与约定

- `mn` 是阶数 n。`A1`、`EigenVector` 各是 n 个行指针、每行 n 个元素；`EigenValue` 有 n 个元素。内存都由调用方分配。
- `A1[i][j]` 是第 i 行第 j 列。`A1` 不被修改。输出可以与 `A1` 共用存储：输入先整个读进来，算完才写输出。
- 第 k 个本征向量在第 k 列：`EigenVector[i][k]` 是它的第 i 个分量。`A v_k = λ_k v_k`，`||v_k||_2 = 1`。
- 本征值按实部从大到小排列，实部相同时按虚部从大到小；本征向量的列跟着同样排列。
- 每个本征向量乘了一个单位模的因子，使模最大的分量（并列时取行号小的）成为正实数。
- **实数版**（输出是 `double**`）：复本征值的向量按 LAPACK `dgeev` 的格式打包。
  - 实本征值（虚部恰为 0）：第 k 列就是实的本征向量。
  - 一对共轭复数 λ_k、λ_{k+1} = conj(λ_k)（虚部为正的在前，严格共轭，总是相邻）：第 k 列是 x，第 k+1 列是 y；
    λ_k 的本征向量是 x + i·y，λ_{k+1} 的是 x − i·y，`||x + i·y||_2 = 1`。
  - 排序以“一个实本征值”或“一对共轭本征值”为单位（先比实部，再比取正的虚部），所以一对共轭本征值不会被别的
    本征值隔开。

### 返回值

| 返回值 | 含义 |
| --- | --- |
| 0 | 成功 |
| −1 | 参数不合法：`mn < 1`，或某个指针为空（此时不碰任何输出） |
| −2 | `A1` 里有 NaN 或 Inf |
| −3 | 内存不足，或其他意外的异常 |
| −4 | 算出的本征值或本征向量里有 NaN/Inf（只在元素量级极端时出现） |
| k > 0 | LAPACK 的 QR 迭代没有收敛（k 是 LAPACK 返回的 `info`） |

返回 −2、−3、−4、k > 0 时，`EigenValue`、`EigenVector` 全部填成 NaN。

### 最小例子

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

用这个文件和 oneMKL 2026.2.0 运行的输出：

```text
lambda_0 = 2        v = (0.0000, 0.0000, 1.0000)
lambda_1 = 0+1i  v = x + i*y,  lambda_2 = 0-1i  v = x - i*y
```

## 线程

文件开头的宏 `MATRIX_EIG_THREADS` 决定一次调用最多用几个线程，函数接口不变。
- `0`（默认）：用 MKL 自己的线程数（没有用 `MKL_NUM_THREADS` 改过就是物理核数）。如果已经有别的调用在算，
  这次只用 1 个线程，免得宿主程序已经在几个矩阵之间并行时互相抢核。
- `n >= 1`：每次调用最多 n 个线程。

只改调用线程自己的 MKL 设定（`MKL_Set_Num_Threads_Local`），调用结束就恢复。函数可重入，可以在多个线程里同时调用。

结果与别的求解器只在舍入误差以内一致，不逐位相同。MKL 在不同线程数、不同处理器上的结果末几位也会不同；
要可重复，设环境变量 `MKL_CBWR=AVX2`，并把 `MATRIX_EIG_THREADS` 设成固定的值。

## 用时

随机矩阵，单位秒，取几次中最快的一次。AMD Ryzen 9 5950X（16 核）、Windows、MSVC 14.34 `/O2`，oneMKL 2026.2.0
默认 16 线程。

| n | StableRCWA 的 `eig_qr`，8 线程 | 本文件，复矩阵 | 本文件，实矩阵 | NumPy 1.26.4（OpenBLAS），复矩阵 | NumPy，实矩阵 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | 0.141 | 0.082 | 0.045 | 0.217 | 0.135 |
| 578 | 0.657 | 0.443 | 0.215 | 0.837 | 0.467 |
| 882 | 1.91 | 0.926 | 0.457 | 1.72 | 0.908 |
| 1250 | 4.98 | 1.87 | 0.893 | 3.07 | 1.59 |
| 2000 | 20.1 | 5.82 | 2.52 | 8.50 | 3.59 |
| 4000 | 199 | 36.2 | 16.0 | 57.2 | 21.8 |

- 与 `eig_qr`（只有头文件的库里的 QR 求解器；2026-10-21 用同一算法的单文件版本测）比：复矩阵快 1.5 到 5.5 倍，
  实矩阵快 3.1 到 12.4 倍。`eig_qr` 把实矩阵当复矩阵算，所以它的实矩阵、复矩阵用时相同。
- 实矩阵走 `dgeev`，比 `zgeev` 快一倍左右。
- 与用 OpenBLAS 的 `numpy.linalg.eig` 比：复矩阵快 1.5 到 2.6 倍，实矩阵快 1.4 到 3.0 倍。链接 MKL 的 SciPy
  跑的是同一个 `zgeev`，复矩阵与本文件差不多快。
- 进程里第一次调用还要加载 MKL（实测 0.01 到 0.25 秒）。
- 没有 46340 阶的上限：调的是 64 位整数（ILP64）的 `zgeev_64_`、`dgeev_64_`，2024.2.2 到 2026.2.0 的 oneMKL 都有，
  与 `MKL_INTERFACE_LAYER` 怎么设无关。实际的上限是内存：一次调用约需 80·n² 字节（复矩阵）或 64·n² 字节（实矩阵），
  含调用方自己的数组。

## 精度

- `dgeev`、`zgeev` 用酉变换，后向稳定。2000 阶的随机复矩阵、随机实矩阵、强非正规（Grcar）矩阵和量级悬殊的矩阵上，
  全部本征对的最大相对残差 `||A v − λ v|| / (||A||_F ||v||)` 不超过 5.1e-16。
- 实矩阵化成实 Schur 形，所以实本征值的虚部严格为 0，共轭对严格共轭。
- LAPACK 先对矩阵做平衡。对量级悬殊（元素跨很多个数量级）的矩阵，这比不做平衡的 QR 迭代（例如 `eig_qr`）
  算出的本征值准得多。
- 只有不用 `dgeev` 时（MKL 早于 2025.0.1），实矩阵才当作复矩阵求解再配成共轭对。那时有两条局限：
  三重以上的亏损实根，残差可能到 1e-12 量级；同一个实本征值上有两个以上亏损块时，本征向量组可能数值上奇异。

## 测试

`standalone/tests/test_standalone_mkl.cpp` 检查收敛与残差、排序、相位规范、实数版的打包格式、共轭配对、返回码、
NaN 填充、输入输出共用存储和多线程同时调用。它要有 oneMKL，所以默认不编，CI 里也不编。Windows + MSVC 上，
先运行 `install_mkl.bat`，然后：

```bat
cmake -S . -B build -DSTABLERCWA_BUILD_STANDALONE_MKL=ON
cmake --build build --config Release --target test_standalone_mkl
build\Release\test_standalone_mkl.exe
```

链接别的 oneMKL 导入库时加 `-DSTABLERCWA_MKL_RT_LIB=<路径>`。

在 Windows（MSVC 14.34、AMD Ryzen 9 5950X）上验证过：
- oneMKL 2025.3.1、2026.2.0 下测试全部通过（42125 项），经上面的 CMake 选项编译和直接用 `cl` 编译都试过；
  上一版测试在 `MKL_INTERFACE_LAYER` 设成 `LP64` 或 `ILP64` 时也全部通过；
- DLL 用 `/W4 /WX` 编译零警告，只导出这两个函数，“`LoadLibrary` → 两个重载各算一个 300 阶矩阵 → `FreeLibrary`”
  30 轮正常；
- `install_mkl.bat` 装进空文件夹的运行库是完整的；
- oneMKL 2025.0.0 下（文件退回 `zgeev`）有 27 项失败，全在“精度”一节说的两族退化情形里。这也是要用 2025.0.1
  或更新版本的原因之一。

## 许可证

这个文件、安装脚本和导入库属于 StableRCWA，按 MIT 许可证发布。运行库属于 Intel，不在本仓库里：oneMKL 按 Intel
Simplified Software License 发布，Intel OpenMP 按 Intel End User License Agreement for Developer Tools 发布。
`install_mkl.bat` 会把两份许可证都放进 `mkl/licenses`。
