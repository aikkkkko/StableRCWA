# StableRCWA

**只有头文件的 C++17 RCWA 求解器，用于一维、二维周期光栅，高截断阶数下依然算得准。**
RCWA 即严格耦合波分析，也叫 Fourier 模态法。

[English](README.md)

![界面操作演示：载入示例、开始计算、看近场、切换分量与相位、二维模型的双视图](docs/images/gui-demo.gif)

## 为什么用它

- **高阶稳定。** 每一层的本征问题都用酉变换的 Hessenberg-QR 迭代求解，截断阶数加大也不会失准。传播用 S 矩阵递推。
  一维 TE/TM 下，传播常数严格为零的模式有精确处理；圆锥入射下，退化的层自动换用 Schur 基。
  下表把同一个结构换成消元型的 LR 迭代来对照：

  | 截断阶数 M（2M+1 个谐波） | QR（默认）：Littrow 级效率 | LR：Littrow 级效率 | LR：R + T | LR 的 `eig_info` |
  | --- | --- | --- | --- | --- |
  | 60 | 98.586939 % | 98.586939 % | 100.000000 % | 0 |
  | 100 | 98.586941 % | 98.587496 % | 100.000506 % | 0 |
  | 150 | 98.586942 % | **86.618778 %** | **86.900177 %** | 0 |

  LR 每一次都报告成功，唯一的异常迹象是 `rcwa_demo` 打印的本征对残差：三个 M 下依次是 1.4e-4、1.4e-1、1.2e-1。
  QR 的残差始终不超过 8.5e-16，R + T 始终是 100.000000 %。

  <sub>算例：20 层介质高反膜上的单周期介质光栅，无吸收，TE，λ = 1.053 µm，θ = 66.36°。
  命令：`./rcwa_demo --case single --M 150 --eig lr`。环境：Apple M1 Pro，Apple clang 17。
  LR 的数字随编译器和处理器变化，QR 的不变。</sub>

- **零依赖。** 包含头文件就能调用，只用 C++ 标准库，不用先编译或链接任何库。
- **功能完整：**
  - 一维光栅 TE/TM；
  - 圆锥入射，每一级给出完整的 2×2 Jones 矩阵；
  - 二维交叉光栅；
  - 六分量近场与相位图；
  - 吸收介质（复折射率）。
- **独立验证：**
  - 与 COMSOL 有限元对照，圆锥入射用二维模型，交叉光栅用三维模型；比的是复振幅，不只是效率；
  - 与闭式解、严格成立的物理关系对照：通量幺正性、互易性、镜像、平移与周期重复等价。
- **三种用法：**
  - C++ 头文件；
  - JSON 进、JSON 出的命令行程序，可以从 Python、MATLAB 或脚本里调用；
  - 本机网页界面，不写代码也能建模、计算、看场。

## 一分钟上手

只需要一个 C++17 编译器。

```sh
git clone https://github.com/aikkkkko/StableRCWA.git
cd StableRCWA
c++ -std=c++17 -O2 -I include -I examples examples/rcwa_demo.cpp -o rcwa_demo
./rcwa_demo --case single --M 20
```

输出的前三行：

```text
结构 single，TE，M = 20（N = 41），本征分解用 QR 迭代
  第 1 级反射效率  98.586865 %
  R = 99.794843 %   T = 0.205157 %   R + T = 100.000000 %
```

后面还有残差和用时两行。结构没有吸收，所以 `R + T` 应当是 100 %。

- **Linux：** 编译命令另加 `-pthread`。还没有在 Linux 上实测过。
- **Windows：** 用 `g++ ... -static`，或者 `cl /std:c++17 /O2 /EHsc /utf-8 /I include /I examples examples\rcwa_demo.cpp`。运行前先执行 `chcp 65001`，否则中文输出会显示成乱码。

## 在 C++ 里调用

```cpp
#include <cstdio>
#include "stablercwa/rcwa1d.hpp"
using namespace stablercwa;

int main() {
    rcwa::Stack stack;                       // 自上而下：入射介质、各层、出射介质
    stack.nI = 1.0;  stack.nII = 1.46;  stack.period = 0.574;          // 长度单位任取，全程一致
    // 光栅层：厚 0.38；一个周期里 [0, 0.172] 是折射率 1.46 的齿，[0.172, 0.574] 是空气
    stack.layers.push_back(rcwa::Layer::grating(0.38, {0.0, 0.172, 0.574}, {1.46, 1.0}));
    stack.layers.push_back(rcwa::Layer::uniform(0.115, 1.96));        // 均匀膜

    const int M = 20;                        // 保留 −M … M 级
    rcwa::Result r = rcwa::solve(stack, 1.053, 66.36, M, rcwa::Polarization::TE);  // 波长、入射角（度）

    for (int m = -1; m <= 1; ++m)
        std::printf("m = %+d   R = %.6f   T = %.6f\n", m, r.DEr[m + M], r.DEt[m + M]);
    std::printf("R + T = %.12f\n", r.reflectance() + r.transmittance());
}
```

```sh
c++ -std=c++17 -O2 -I include my_grating.cpp -o my_grating && ./my_grating
```

```text
m = -1   R = 0.000000   T = 0.000000
m = +0   R = 0.026034   T = 0.497012
m = +1   R = 0.258135   T = 0.218818
R + T = 1.000000000000
```

常见的改法：

- **吸收介质：** 折射率写成 `cplx(n, k)`，`k > 0`。
- **圆锥入射：** 用 `rcwa1d_conical.hpp` 的 `rcwa::solve_conical`。
- **二维交叉光栅：** 用 `rcwa2d.hpp` 的 `rcwa::solve2d`。
- **近场：** 用各个 `*_fields.hpp`。
- **数值失败**（方程奇异、本征迭代不收敛）：不抛异常，效率全部返回 NaN。一维由 `eig_info` 给出本征求解的返回码，二维由 `status`、`message` 说明原因。
- **参数非法**（边界不递增、|θ| ≥ 90° 等）：抛 `std::invalid_argument`。

### 在 CMake 项目里使用

```cmake
set(STABLERCWA_BUILD_EXAMPLES OFF)
set(STABLERCWA_BUILD_TESTS OFF)
add_subdirectory(StableRCWA)
target_link_libraries(你的程序 PRIVATE StableRCWA::stablercwa)
```

## JSON 进，JSON 出

```sh
c++ -std=c++17 -O2 -I include -I examples examples/rcwa_json.cpp -o rcwa_json
./rcwa_json request.json --out reply.json
```

```json
{
  "mode": "planar",
  "stack": {"nI": 1.0, "nII": 1.46, "period": 0.574,
            "layers": [{"d": 0.38, "edges": [0.0, 0.172, 0.574], "n_values": [1.46, 1.0]},
                       {"d": 0.115, "n": 1.96}]},
  "M": 20, "polarization": "TE",
  "points": [{"wavelength": 1.053, "theta_deg": 66.36}]
}
```

应答里有各级效率，这个例子的 `R_total = 0.28416918…`，`T_total = 0.71583081…`。

- 模式有 `planar`、`conical`、`periodic2d` 三种。
- `points` 里写多个点就是扫描；`options.threads` 大于 1 时各点并行求解。
- 请求写错时应答 `{"ok": false, "error": …, "where": …}`，拼错的键也算错。

## 图形界面

在本机 `127.0.0.1` 上运行，不联网。需要 Python 3 和 NumPy、SciPy、Matplotlib。

```sh
python3 -m pip install -r gui/requirements.txt
python3 gui/gui.py
```

- **macOS：** 双击 `gui/启动GUI.command`。
- **Windows：** 第一次先运行 `gui\install_windows.bat`（Python 3.11 或 3.12，x64），以后运行 `gui\start_gui.bat`。

启动时，界面会用系统里的编译器自动编译 C++ 求解程序。找不到编译器时，改用自带的 Python 求解器，结果页会注明。

<details>
<summary>截图</summary>

**建模页**

![本机网页界面的建模页](docs/images/gui-model.webp)

**结果页**

![本机网页界面的结果页：各级效率与近场](docs/images/gui-result.webp)

**二维交叉光栅：俯视图与联动剖面**

![二维模型视图：矩形柱的俯视图，以及贯穿全部层的 x-z 剖面](docs/images/gui-2d.webp)

</details>

## 支持什么

| | C++ 头文件 | Python 求解器 |
| --- | --- | --- |
| 一维光栅 TE/TM | ✓ | ✓ |
| 一维圆锥入射（Jones 矩阵） | ✓ | ✓ |
| 二维交叉光栅（矩形分块，Li 因式分解） | ✓ | ✓ |
| 吸收介质（复折射率） | ✓ | ✓ |
| 六分量近场与相位 | ✓ | ✓ |
| 有限（非周期）结构，横向 PML ¹ | – | 一维，入射面内入射 |
| 单轴晶体层（光轴沿 x、y 或 z）¹ | – | 一维，TE/TM |
| FFT 采样的 Fourier 系数（解析系数之外的另一种算法）¹ | – | 一维 |
| 磁性介质（μ ≠ 1） | – | 实验性，未接入界面 |
| 一般 6×6 本构张量 | – | 一维，实验性，未接入界面 |
| GPU 本征分解（CUDA cuSOLVER） | 可选，实验性 | – |

<sub>¹ 来自作者早先的 Python 求解器包，验证报告之后补进 `docs/`。</sub>

使用时要注意：

- 截断阶数要自己做收敛检查。二维比一维收敛慢。
- 入射介质必须无吸收。圆锥入射时，出射介质也必须无吸收。
- 只有关掉 FMA 融合（`-ffp-contract=off`）时，多线程本征分解的结果才验证过与单线程逐位相同。

## 验证

| 检查 | 结果 |
| --- | --- |
| 一维圆锥入射对 COMSOL 二维有限元（8 个算例，从裸界面到光栅加膜系） | 各偏振通道效率 ≤ 3.3e-6，复 Jones 矩阵 ≤ 1.1e-5，相位 ≤ 0.013° |
| 二维交叉光栅对 COMSOL 三维周期有限元（矩形柱） | 复振幅 1.2e-4 到 4.1e-4，随网格加密继续减小；效率 ≤ 2.1e-4 |
| 传播常数严格为零（一维），对 COMSOL | 振幅 2.1e-9，效率 8.3e-11 |
| C++ 二维求解器对它所移植的 Python 实现 | 远场 ≤ 3.1e-13（21 个算例），近场切面 ≤ 1.1e-12 |
| 单元与回归测试 | 21 个 C++ 测试程序，另有 4 项检查 JSON 模式开关的 CTest 条目（共 25 项），在 macOS arm64（Apple clang 17）上全部通过；21 个测试程序在 Windows 上用 MinGW g++ 13.1 和 MSVC 19.34 也全部通过；Linux 只由持续集成覆盖 |

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
cd build && ctest -C Release
```

详见 [docs/validation.zh-CN.md](docs/validation.zh-CN.md)。

## 讲解页

两份 COMSOL 对照的图文讲解，推导和图都嵌在页面里。在 GitHub 上浏览仓库时 `.html` 只显示源码，请在项目网站（GitHub Pages）上看，或者在本地克隆里直接用浏览器打开。

- **圆锥入射与 COMSOL 对照：** 逐通道对照，以及两边约定的对应关系。
  [docs/pages/圆锥入射-COMSOL对照.html](docs/pages/圆锥入射-COMSOL对照.html)
  · [项目网站上的页面](https://aikkkkko.github.io/StableRCWA/pages/圆锥入射-COMSOL对照.html)
- **截止零根、Schur 传播与 COMSOL 对拍：** 本征子空间与病态性、零根散射块的完整推导、圆锥退化层的 Schur 传播、近场。
  [docs/pages/截止零根-Schur传播与COMSOL对拍.html](docs/pages/截止零根-Schur传播与COMSOL对拍.html)
  · [项目网站上的页面](https://aikkkkko.github.io/StableRCWA/pages/截止零根-Schur传播与COMSOL对拍.html)

## 文档

- [快速上手：每一步都写了应当看到的结果](docs/quickstart.zh-CN.md)
- [使用说明：约定、圆锥入射、二维、近场、JSON 协议、图形界面、本征算法、GPU](docs/user-guide.zh-CN.md)
- [验证报告](docs/validation.zh-CN.md)
- `include/stablercwa/` 下每个头文件开头的接口说明

## 引用

如果在发表的工作里用到 StableRCWA，请按 [CITATION.cff](CITATION.cff) 引用。

## 许可证

[MIT](LICENSE)。本征分解按 EISPACK 的算法重新编写（见致谢），仓库里没有打包任何第三方代码。

## 致谢

- **本征分解：** 按 EISPACK 的 CORTH/COMQR2（QR）和 COMHES/COMLR2（LR，只作对照）编写。
- **二维 Fourier 因式分解：** 采用 L. Li 的规则。
- **独立参照：** COMSOL Multiphysics® 有限元模型。
- **界面示例 `meent_conical`：** 结构取自 Meent 项目的教程。
