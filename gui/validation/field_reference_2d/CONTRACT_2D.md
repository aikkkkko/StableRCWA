# 二维（交叉光栅）RCWA 的 C++ 实现：约定

合同版本 1。数值权威：`gui/2drcwa/rcwa2d.py`（与作者早先的 Python 求解器包里的 rcwa2d.py 逐字相同）。
独立证据：COMSOL 三维有限元对 rcwa2d 的远场复振幅与相位、近场六分量的对照（对照报告与原始数据没有随仓库发布）。
公共类型：`include/stablercwa/rcwa2d_types.hpp`（几何、选项、结果、谐波下标约定）。
一维近场的设计与验收见 `gui/validation/field_reference/CONTRACT.md`，二维沿用同一套思路。

## 1. 范围

非磁性、各向同性（折射率可为复数，入射介质必须无吸收）、矩形格子、解析 Fourier 系数（无 FFT）、Li 因式分解（`Factorization2D::Li`，生产默认）和纯 Laurent（仅作对照）。
层 = 均匀层，或 y 分段 × 每段内 x 分段的分段常数层。入射：θ、φ、Jones [Es, Ep]。输出：各 (m, n) 级的 s/p 复振幅与效率；后续步骤再加六复场、切面、相位。
**不做**：横向 PML、各向异性、FFT 系数、磁性扩展（`gui/mag/rcwa2d_mag.py`）、非矩形格子、圆形截断。截断固定为矩形（−M1…M1）×（−M2…M2）。

## 2. 数值约定（逐项照 rcwa2d.py 实现）

下面列的是容易移错的地方，行号是 rcwa2d.py 的。

- 谐波下标、Kx、Ky：见 `rcwa2d_types.hpp` 文件头（rcwa2d.py:75–84、428–429）。`kzI`、`kzII` 存的是 kz/k0。
- `_kz_over_k0`（:205）：先对 `n² − Kx² − Ky²` 开**复**平方根，再把 Im<0 的翻成 −kz；传播级是实正，倏逝级是 +i|kz|，吸收介质保留复数。
- `_branch(q)`（:199）：先把 Re q<0 的整体取负；再把 `Re q < 1e-10 且 Im q > 0` 的取负。两步顺序和阈值都不能改。
- 均匀介质（入射区、出射区、均匀层，:375）：`W = I`，`q = [q1; q1]`（q1 = branch(−i kz)），`V` 是每个级次一个 2×2 块（`Qb/q1`）。**kz 精确等于 0 → 失败**（见 §5），不是输出 NaN。
- 结构层（:394）：`Ei = inv(Eps)`；`P`、`Q` 按 :403–404 装配（`Kx`、`Ky` 是对角阵，用行列缩放，不要真的乘对角阵）；对 `P·Q`（2N 阶，复，非 Hermite）做本征分解；`q = branch(sqrt(w2))`；`V = (Q·W)/q`。
- 界面 S 块（:271–290）：`Tp = ½(W2⁻¹W1 + V2⁻¹V1)`、`Tm = ½(W2⁻¹W1 − V2⁻¹V1)`；`Tdd = Tp − Tm·Tp⁻¹·Tm`、`Rdu = Tm·Tp⁻¹`、`Rud = −Tp⁻¹·Tm`、`Tuu = Tp⁻¹`。传播块 `X = exp(−k0·q·d)`（|X| ≤ 1），`Rdu = Rud = 0`。
- 2×2 分块对角快路径（:217–256、:271–285、:320–346、:348–364）：均匀层的 W、V、传播块、均匀–均匀界面块都存成 `(2,2,N)`，乘法与求逆退化成 N 维逐元素运算。
  这是性能的关键（20 层膜堆几乎免费），必须保留；`Options2D::homogeneous_fast_path=false` 时把均匀层当稠密层走，用来核对快路径。
- 自下而上只递推 `Rud` 一条链（`_rud2`），再自上而下推进（`_step2`）；`star2` 在 rcwa2d.py 里没有被调用，**不要移植**。
- 入射（:466–474）：`inc[i0] = −Es sinφ + Ep cosθ cosφ`（Ex），`inc[N+i0] = Es cosφ + Ep cosθ sinφ`（Ey）；Jones 不归一。
- 效率与 s/p（:481–506）：传播判据 `Re kz > 1e-12`；`Ez = −sign·(Kx Ex + Ky Ey)/kz`（反射 sign=−1，透射 sign=+1，只在传播级）；`DE = Re(kz)·(|Ex|²+|Ey|²+|Ez|²)/(nI cosθ)`；
  s/p 分量在 `kt = hypot(Kx, Ky) > 1e-12` 时按 :497–502 的公式，`kt ≤ 1e-12`（θ=0 的零级）时用入射方位角 φ：`s = −sinφ Ex + cosφ Ey`，`p = sign·(cosφ Ex + sinφ Ey)/cth`。
- Fourier 算子（:109–188）：`fourier_1d` 解析积分；`Cx_h = Σ_k wseg[k]·inv(toeplitz(ci_k))`（x 逆规则沿 y 做 Fourier）；`Eps_h = Σ_k wseg[k]·toeplitz(ce_k)`；
  `Cy` 先沿 y 做逆规则（对每个 x 分段 j，列剖面 `ecol` 来自 `_eps_at`，`wsegx` 是 x 切点集合的 `_seg_fourier`），再沿 x 做 Laurent；Laurent 模式 `Cx = Cy = Eps`。
  x 切点集合的合并去重要照 :144–152 逐步（取模 Lx、`round(…, 12)`、夹回 [0, Lx]、补 0 与 Lx），周期回绕 `_eps_at`（:169）：`xe[0] + (x − xe[0]) mod Lx`、`searchsorted(right) − 1`、夹到 [0, len−1]。
  装配成 N×N：`_assemble_xfirst` / `_assemble_yfirst`（:177–188）。**Matrix 是列主序，下标要显式写，不要依赖线性下标。**

已知的 Python 坑，**C++ 不要照抄**：`rcwa2d.field_at` 区域 I 先算 `exp(+k0 q |z|)` 再乘 0（入射阶里只有 (0,0) 被激发），高阶倏逝波在高 M 且离界面远时 `0×inf = NaN`；
C++ 只对被激发的入射零阶求入射指数（一维合同第 2 节同一条规则）。冻结 Python 参照时不得把这样的 NaN 点当参照值。

## 3. 相关文件

| 部分 | 文件 |
| --- | --- |
| 算子 | `include/stablercwa/rcwa2d_operators.hpp`、`tests/test_rcwa2d_operators.cpp` |
| 求解核心 | `include/stablercwa/rcwa2d.hpp`、`tests/test_rcwa2d.cpp`、`tests/rcwa2d_core_fixture.inc` |
| 近场 | `include/stablercwa/rcwa2d_fields.hpp`、`tests/test_rcwa2d_fields.cpp`、`tests/rcwa2d_fields_fixture.inc` |
| 切面与相位 | `include/stablercwa/rcwa_fields.hpp`、`rcwa_phase.hpp`、`tests/test_rcwa_slice_phase.cpp` |
| 冻结参照 | `gui/validation/field_reference_2d/`、`gui/validation/field_reference_2d_cases.py`、`generate_field_reference_2d.py`、`gui/2drcwa/selftest2d.py`、`tests/rcwa2d_reference.inc`、`tests/test_rcwa2d_reference.cpp` |

## 4. 公共接口

新函数放 `namespace stablercwa::rcwa::detail2d`（内部）或 `stablercwa::rcwa`（公共，名字带 2D 后缀）。头文件仅头文件库，`#pragma once`，注释用中文，风格与一维代码一致。

```cpp
// rcwa2d_operators.hpp
namespace detail2d {
struct LayerOperators2D { Matrix eps, cx, cy; };      // N×N，N=(2M1+1)(2M2+1)，行列下标 = harmonic_index_2d
LayerOperators2D layer_operators(const Layer2D&, double Lx, double Ly, int M1, int M2, Factorization2D);   // 均匀层返回 n²·I 三份
}
// rcwa2d.hpp
Result2D solve2d_with_operators(const Stack2D&, double wavelength, double theta_deg, double phi_deg, int M1, int M2,
                                const Jones2D&, const std::vector<detail2d::LayerOperators2D>& ops,   // 与 layers 同长；均匀层的项忽略
                                const Options2D& = {}, detail2d::FieldCapture2D* capture = nullptr);
Result2D solve2d(const Stack2D&, double wavelength, double theta_deg, double phi_deg, int M1, int M2, const Jones2D&, const Options2D& = {});
```
`solve2d` = 对每个结构层调用 `layer_operators`（可 `parallel_for`），再调用 `solve2d_with_operators`。

`FieldCapture2D`（在 rcwa2d.hpp 里定义，近场计算使用）。`capture != nullptr` 时在同一次求解里保留：
- 区域 I、II：q（2N）、V（2×2 分块对角）、kz/k0；入射矢量 `inc`（2N）、反射 `R`（2N）、透射 `T`（2N）。
- 每层 j：`z_top`、`d`、`homogeneous`、`q`（2N）、`W`（结构层稠密 2N×2N；均匀层不存，W=I）、`V`（结构层稠密；均匀层 2×2 分块对角）、
  `Eps` 的求解表示（结构层：`Ei = Eps⁻¹` 或其 LU，二选一并在注释里写明；均匀层：n²）、`a_plus`（层顶下行，2N）、`a_minus`（层底上行 = 该面以下的反射链作用在层底下行上，2N）。
- `capture == nullptr` 时不得多花内存，也不得改变数值。**不能用 `exp(+k0 q d)` 反推层底振幅**（厚吸收/倏逝层会增长失稳）。

## 5. 失败策略

- 参数非法（非有限、M 为负、层厚为负、边界不递增、折射率非有限、nI 非实或有吸收、θ 非有限…）：抛 `std::invalid_argument`。
- 均匀介质（入射区、出射区、均匀层）里某级 kz 精确为 0：`status = RayleighAnomaly`，`message` 写区域（"入射区"/"出射区"/"第 j 层"）和 (m, n) 列表，结果数组填 NaN。与 rcwa2d.py 的 `RayleighAnomalyError` 一致，与一维 C++ 允许入射区掠射**不同**——这是有意的。
  结构层里近零的 q 不特殊处理（rcwa2d.py 也没有），也不加小量。
- 本征迭代未收敛、线性方程组奇异、中途 NaN/Inf：`status = NumericalFailure`，NaN 填充，`eig_info` 记录第一个非零值。

## 6. 数值验收

逐复数 `|a − b| ≤ atol + rtol·|b|`，不能只比模平方，不能先重新生成基线再验证自己。
- 远场复振幅（Rs、Rp、Ts、Tp）与效率：N ≤ 289 的算例 `atol = rtol = 1e-9`（实测最大差写进测试输出）；N 更大时放宽并说明。
- 算子 Eps、Cx、Cy：`1e-12`（按元素，相对该矩阵的最大模）。
- 不比较 W、V：简并子空间里本征基不唯一，只比较可观测量（振幅、效率、场）。
- 能量守恒（无吸收）`|ΣDE − 1| ≤ 1e-9` 是必要非充分条件，不能替代对参照的比较。
- 独立检验（不依赖 Python 输出）：一维极限（y 不变 / x 不变的结构，对一维 C++ `solve` 与 `solve_conical`）、均匀多层（Airy）、2×2 超胞等价性、x↔y 互换结构并同步改 φ。这些来自 `gui/2drcwa/selftest2d.py` 的 T1–T4，是一维以外唯一不依赖 rcwa2d.py 自身的判据。

## 7. 冻结参照的算例集

沿用一维的格式：JSON 记录输入、容差、源码 SHA-256、Python/NumPy/SciPy 版本；NPZ 存数据；另生成 `tests/rcwa2d_reference.inc`（C++ 测试用，格式参照 `tests/conical_reference.inc`，≤ 2 MB）。
每例保存：Result2D 里的全部数组（mn、Kx、Ky、kzI、kzII、Ex_r…Ez_t、Rs、Rp、Ts、Tp、DEr、DEt、prop），以及（在有限的几个 z 处）六个分量的谐波列。小算例另存 Eps、Cx、Cy（算子单元测试用）。
M 选小（总谐波 N ≤ 169），使 C++ 测试在秒级跑完；至少一个算例 N≈600 用来看大尺度精度。算例（参数写进生成器，不随机）：

1. 无层单界面；均匀膜（θ=42°、φ=−25°）。
2. y 不变光栅（T1，M2=0，TE 与 TM）；x 不变光栅（T1′，M1=0，φ=90°）。
3. 方柱（0.8×0.8 µm、0.3×0.3 µm、n=1.8、厚 0.3 µm，θ=42°、φ=−25°）；偏心矩形柱 0.3×0.2（中心 (0.3, 0.55)）θ=0、φ=30°（kt=0 分支）。
4. 吸收方柱 n=1.8+0.13i；吸收均匀膜（n=1.65+0.12i）。
5. Lx≠Ly；多个 y 分段且各段 x 边界互不相同（专测 Cy 的 x 切点并集）；结构层 + 膜堆 + 第二个结构层。
6. 椭圆偏振（psi、delta）；Laurent 因式分解；M2=0、M1=0 的退化。
7. C4 对称方柱正入射（本征值精确简并，对本征分解是压力）。
8. 一个 Rayleigh 异常（kz 精确为 0）的失败算例：记录期望的 status 与 message 中的 (m, n)。
