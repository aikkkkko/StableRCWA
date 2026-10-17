# 二维（交叉光栅）RCWA 的固定 Python 参照（合同 v1）

数值合同见 [CONTRACT_2D.md](CONTRACT_2D.md)；数值权威是 `gui/2drcwa/rcwa2d.py`（与原求解器包逐字相同，未修改）。
做法沿用一维的 `../field_reference/`：JSON 记录输入、容差、来源哈希、版本；NPZ 存数据；测试只读冻结数据，**普通测试与生成器都不会自动替换它**。

| 文件 | 内容 |
| --- | --- |
| `reference_2d_v1.json` | 25 个算例的显式输入、容差、结构零分量、数组清单（形状 / dtype）、源码与生成器的 SHA-256、Python / NumPy / SciPy 版本、容差依据 |
| `reference_2d_v1.npz` | 数据，键为 `算例id__数组名`（复数 complex128，实数 float64，掩膜 bool，级次 int64，C 连续序） |
| `../../../tests/rcwa2d_reference.inc` | C++ 测试用的参照表（N ≤ 169 的算例；头部写着生成脚本与源 NPZ 的 SHA-256；≤ 2 MB） |
| `../field_reference_2d_cases.py` | 算例的显式输入（不随机、不大扫描）与采样函数 |
| `../generate_field_reference_2d.py` | 生成器（只写候选目录；写冻结目录必须显式 `--refresh-baseline`） |
| `../test_field_contract_2d.py` | 校验：完整性、同环境重算复现、独立检验 |

SHA-256 与字节数记录在 JSON 里（`archive`、`cpp_table`）；`.inc` 头部另写源 NPZ 的 SHA-256。

## 复核

从 `gui/` 目录运行（Python 3.9 / NumPy 1.21.5 / SciPy 1.9.1；BLAS 线程数设 1 可逐位复现）：

```sh
cd gui
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m unittest validation.test_field_contract_2d -v      # 约 45 s，其中 N=625 的算例约 40 s
RCWA2D_SKIP_BIG=1 python3 -m unittest validation.test_field_contract_2d -v                                              # 跳过 N=625 的重算（仍检查它的冻结数据）
```

测试内容：(a) NPZ / `.inc` 的 SHA 与字节数与 JSON 一致，JSON 里的输入与 `field_reference_2d_cases.py` 相同，数组形状 / dtype / 布局正确；
(b) 用当前 `rcwa2d.py` 重算全部数组，在 JSON 的 `tolerance.reproduce` 内复现（同一环境、线程数 1 时逐位相同）；
(c) 独立检验：均匀结构的 Airy / Fresnel 六分量解析场（含吸收膜 + 吸收基底）；y 不变 / x 不变光栅对 `rcwa1d.solve` 与 `conical.solve_conical`（复振幅与六个谐波分量，x 不变的按 `x_1D=y, y_1D=−x, φ_1D=φ−90°` 映射）；
能量守恒与 |E|² = |Rs|²+|Rp|²；界面切向场连续（selftest2d 的 T5，z_i ∓ 1e-10）；C4 方柱的 x↔y 对称；Eps 对矩形柱闭式 Fourier 积分；GUI `gui_fields.view_data` 的相位显示与冻结相位一致；`.inc` 回读后与 NPZ 逐位相同（唯一例外是符号零：NPZ 里 22 个数组共 2339 个 −0.0，生成器的 `number()` 有意把 −0 写成 0，所以在 `.inc` 里是 +0.0）。

生成候选数据（不碰冻结文件）以便比对：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 gui/validation/generate_field_reference_2d.py --output /tmp/rcwa2d-candidate
```

同一环境下重复生成得到逐字节相同的 NPZ、JSON、`.inc`（NPZ 的 zip 时间戳固定）。任何基线更新都要先审查来源哈希、复场差异和独立检验，再显式加 `--refresh-baseline`
（它会同时覆盖 `tests/rcwa2d_reference.inc`）。**不要为了让回归测试变绿而刷新数据。**

## 数组布局（与一维 NPZ 的交换格式一致，分量顺序恒为 `[Ex, Ey, Ez, Hx, Hy, Hz]`）

场的约定：时间因子 exp(−iωt)，横向因子 exp[−i k0 (Kx x + Ky y)]；E 对入射 |E| = 1 归一，H 存的是 **Z0·H**（入射 |Z0H| = nI）；Jones 不归一（算例里都是单位范数）。
长度单位 µm。`Kx, Ky, kzI, kzII` 以 k0 为单位（`k0` 另存，1/µm）。

每个成功算例（键为 `id__名字`）：

| 名字 | 形状 | 含义 |
| --- | --- | --- |
| `mn` | `[N,2]` int64 | 各列的 (m, n)；**列 i = harmonic_index_2d(m,n) = (m+M1)(2M2+1) + (n+M2)**，m 在外层、n 在内层 |
| `Kx, Ky, DEr, DEt` | `[N]` float64 | 归一化横向波矢；反射 / 透射效率（非传播级为 0） |
| `kzI, kzII, Ex_r, Ey_r, Ez_r, Ex_t, Ey_t, Ez_t, Rs, Rp, Ts, Tp` | `[N]` complex128 | Result2D 的全部数组；`Ez_*` 只在传播级非零 |
| `prop_r, prop_t` | `[N]` bool | 传播级（Re kz > 1e-12） |
| `inc` | `[2N]` complex128 | 入射矢量（Ex 块后接 Ey 块；只有零级非零） |
| `i0, k0, theta, phi` | 0 维 | (0,0) 级下标；k0；θ、φ（弧度） |
| `z` | `[nz]` | 取样深度，升序；与 JSON 的 `zs` 相同 |
| `harmonics` | `[6,nz,N]` complex128 | 各 z 处六个分量的谐波列：`[component, z, order]`，order 最快；等价于 `rcwa2d.field_at(res, z)` |
| `eps_L{j}, cx_L{j}, cy_L{j}` | `[N,N]` complex128 | 结构层 j（0 起算的层号）的 Eps、Cx、Cy；行 i、列 j' 是谐波下标；只在 N ≤ 49 的算例里有（`save_operators`）。Laurent 算例里 `cx = cy = eps` |
| `fields_xz`, `fields_yz` | `[6,nz,nt]` complex128 | 两个切面的复场，`[component, z, t]`，t 最快。xz：x = `t_xz` = arange(nt)·Lx/nt，**y = `fixed_xz`**；yz：y = `t_yz` = arange(nt)·Ly/nt，**x = `fixed_yz`**。直接求和（`rcwa2d.field_slice`，GUI 用同一做法） |
| `E2_xz`, `E2_yz` | `[nz,nt]` float64 | Σ\|E\|²（原始复场，不再归一） |
| `t_*`, `fixed_*` | `[nt]` / 0 维 | 切面坐标 |
| `phase_absolute_*`, `phase_incident_*` | `[6,nz,nt]` float64 | arg F 与 arg(F·conj(carrier))，弧度，[−π, π]；掩膜处为 NaN |
| `phase_mask_*` | `[6,nz,nt]` bool | 相位无定义：非有限或 \|F\| ≤ 1e-6 × 该分量在本切面的峰值（GUI 与 `field_phase.phase_data` 的同一规则） |
| `phase_comparison_mask_*` | `[6,nz,nt]` bool | 跨后端比较时排除：`phase_mask`、结构零分量、\|F\| ≤ 10·(near atol + near rtol·\|F\|) |
| `incident_amplitudes` | `[2]` | [\|E_inc\|, \|Z0H_inc\|] = [1, nI] |

**载波**（CONTRACT.md 第 5 节）：`carrier = exp(i[−Re(kx0)·x − Re(ky0)·y + Re(kzI0)·z])`，`kx0 = k0·Kx[i0]`，`ky0 = k0·Ky[i0]`，`kzI0 = k0·kzI[i0]`；
二维里 x、y 按切面取值：xz 切面 x = t、y = 固定值；yz 切面 x = 固定值、y = t。相对入射相位不扣掉偏振分量的方向符号。

失败算例（`status = RayleighAnomaly`）没有数组，JSON 的 `expect` 给出期望的区域（"入射区" / "出射区" / "第j层"）与 (m, n) 列表（按谐波下标顺序），`python_failure` 是当前 Python 的消息。

## 算例（25 个；编号 = JSON 里 `cases` 的位置，也是 `.inc` 里的 id）

| id | 合同项 | M1,M2 (N) | 要点 |
| --- | --- | --- | --- |
| 1 `interface` | 1 | 2,2 (25) | 无层单界面，θ=42°、φ=−25°，线偏振 ψ=40° |
| 2 `film` | 1 | 2,2 (25) | 无吸收均匀膜 n=1.8 厚 0.3，基底 1.5 |
| 3 `y_inv_te` / 4 `y_inv_tm` | 2 | 12,0 (25) | y 不变光栅（selftest2d T1），φ=0，s / p；零分量见下 |
| 5 `y_inv_conical` | 2 | 8,0 (17) | 同一光栅，φ=30°，椭圆偏振（ψ=33°、δ=50°） |
| 6 `x_inv_te` / 7 `x_inv_tm` | 2 | 0,12 (25) | x 不变光栅（T1′），φ=90°，s / p |
| 8 `x_inv_conical` | 2 | 0,8 (17) | 同一光栅，φ=50° |
| 9 `pillar_sq` | 3 | 4,4 (81) | 方柱 0.3×0.3、n=1.8、厚 0.3，0.8×0.8 元胞，基底 1.5，λ=1.053，θ=42°、φ=−25°，s（COMSOL 台阶 3 的几何） |
| 10 `pillar_sq_m6` | 3 | 6,6 (169) | 同上，M=6（C++ 测试的 N 上限），p |
| 11 `pillar_ecc` | 3 | 4,4 (81) | 偏心柱 0.3×0.2、中心 (0.3, 0.55)，θ=0、φ=30°（kt=0 的 s/p 分支），p |
| 12 `pillar_abs` | 4 | 3,3 (49) | 吸收方柱 n=1.8+0.13i，θ=42°、φ=−25°，p |
| 13 `film_abs` | 4 | 2,2 (25) | 吸收膜 n=1.65+0.12i，吸收基底 nII=1.38+0.05i（复 nII），椭圆偏振 |
| 14 `lx_ne_ly` | 5 | 2,3 (35) | Lx=0.62≠Ly=0.93，M1≠M2，nI=1.2，n=2.3 的矩形柱在 n=1.46 中 |
| 15 `multi_yseg` | 5 | 2,2 (25) | 三个 y 分段、各段 x 边界互不相同、起点平移（专测 Cy 的 x 切点并集与 `% Lx` 回绕） |
| 16 `stack_sfs` | 5 | 3,3 (49) | 结构层 + 两层均匀膜 + 第二个结构层（4 层；`homogeneous_fast_path` 开 / 关对照） |
| 17 `ellip_pillar` | 6 | 3,3 (49) | 方柱，θ=30°、φ=18°，ψ=55°、δ=35°（selftest2d T5 的偏振） |
| 18 `laurent_pillar` | 6 | 3,3 (49) | 纯 Laurent 因式分解（Cx = Cy = Eps） |
| 19 `m2_zero` / 20 `m1_zero` | 6 | 5,0 / 0,5 (11) | 真二维方柱，一个方向截断到零级 |
| 21 `c4_normal` | 7 | 4,4 (81) | C4 对称方柱正入射（本征值精确简并），s |
| 22 `pillar_big` | 3, 7 | 12,12 (625) | 同 9 的几何，GUI 的 N 上限；**不在 `.inc` 里**；容差放宽到 1e-8 |
| 23–25 `fail_rayleigh_*` | 8 | 1,1 | kz 精确为 0：入射区（λ=Lx=Ly=1、θ=0）、出射区（nII=1.25、λ/Lx=1.25）、第 2 层（n=1.25）；浮点值取二进制精确数，使 `n²−Kx²−Ky² == 0` |

## 容差与结构零分量

JSON 每例的 `tolerance`：

* `far`、`near`：C++ 对冻结 Python 的逐复数比较 `|a−b| ≤ atol + rtol·|b|`（不能只比模平方）。N ≤ 289 取 1e-9（合同第 6 节），N=625 放宽到 1e-8。
  **实测依据**：把 2N 阶本征问题加 1e-16 相对噪声（对任何后端的本征求解差异都是保守的）后，远场最大变化 5.2e-13（N ≤ 169）/ 9.1e-13（N=625），
  六分量谐波 ≤ 1.9e-12，切面和（625 项求和）≤ 1.1e-11（N=625）；所以 1e-9 / 1e-8 有 ≥ 600 倍余量，又足以抓出真实错误（突变检验：把 Ez 放大 0.1%、Hz 变号、Cx 与 Cy 互换、Laurent 当 Li 用，都被 1e-9 的阈值检出）。
* `reproduce`：同环境重算（5e-12，N=625 为 5e-11）；实测 BLAS 线程数改变引起的差 ≤ 7.1e-13（N ≤ 169）/ 1.0e-11（N=625）。
* `near_independent`：**对独立求解器（如 COMSOL 有限元）的近场比较**用，不用于 C++ 对 Python 的回归：COMSOL 对照里近场的有限元噪声下限（Ez、H 的 rms 1.6e-3；离界面 ≥ 0.15 µm 的 Ex、Ey 为 6.9e-3）的 10 倍，
  即绝对 1.6e-2（Ez、H）/ 7e-2（Ex、Ey），入射 |E|=1；只适用于离材料界面 ≥ 0.05 µm 的点（界面附近是 RCWA 的 Gibbs 区，误差 1e-2 量级）。

不比较 W、V（简并子空间里本征基不唯一），只比较可观测量。

`structural_zero_components`（场的这几个分量在整个算例里严格为零——只在上下限 ≤ near atol 的意义上检验，原始相位是舍入噪声，**不得当相位参照**；`phase_comparison_mask` 已把它们排除）：

| 算例 | 零分量 | 远场里为零的数组 |
| --- | --- | --- |
| `y_inv_te`（φ=0，s） | Ex, Ez, Hy | Ex_r, Ez_r, Ex_t, Ez_t, Rp, Tp |
| `y_inv_tm`（φ=0，p） | Ey, Hx, Hz | Ey_r, Ey_t, Rs, Ts |
| `x_inv_te`（φ=90°，s） | Ey, Ez, Hx | Ey_r, Ez_r, Ey_t, Ez_t, Rp, Tp |
| `x_inv_tm`（φ=90°，p） | Ex, Hy, Hz | Ex_r, Ex_t, Rs, Ts |
| `interface`、`film`、`film_abs`（`uniform_stack`） | 无（一般偏振） | **(0,0) 以外的所有级次**的 Ex_r…Tp、DE、谐波列恒为 0 |

注意 `jones` 在纯 s / p 算例里是精确的 [1,0] / [0,1]（不经 `cos 90°`），所以零分量只剩本征求解的舍入噪声。φ=90° 时 `cos φ = 6e-17`，入射矢量的“零”分量约 6e-17，仍在容差内。
其余算例没有严格零分量（C4 的零级交叉偏振只是对称性给出的 ~1e-15，不是结构零，没有列入）。

## 冻结时避开的坑

* `rcwa2d.field_at` 区域 I 的 `exp(+k0 q|z|)×0`：深度只取 |z| ≤ 0.43 µm，N=625 时最大指数 ≲ 45，不溢出；生成器要求所有非相位数组有限，**没有 NaN 参照值**。
* 层界面归属：取样深度在每个界面两侧取 ±1e-6 µm（远大于 1e-12 的归属带）；无层算例不取 z = 0（Python 此时无层可选，会抛 StopIteration）。
* 切面取样点（t = arange(nt)·L/nt，与 GUI 相同）与固定坐标离**物理**材料边界（指数真的变化的边，含周期回绕）≥ 6e-3 µm（测试断言 > 5e-3）；nt 取 14–16。边界上的 Fourier 部分和是 Gibbs 值，只对同一截断的实现可比。
* `phase_absolute` 等相位数组对零场分量没有意义（全部被掩膜）；掩膜的峰值取本切面，所以不同切面的同一分量掩膜可能不同。

## C++ 参照表 `tests/rcwa2d_reference.inc`

先包含 `stablercwa/rcwa2d_types.hpp`（或 `rcwa2d.hpp`），再包含本文件（全局作用域）。内容：

* `Rcwa2dRefCase kRcwa2dRefCases[]`、`kRcwa2dRefCaseCount`、`rcwa2d_ref_case(id)`、`rcwa2d_ref_stack(id)`（建 `Stack2D`；id 不在表里抛 `std::invalid_argument`）；
  每例有 λ、θ、φ（度）、M1、M2、显式 Jones `[Es 实, Es 虚, Ep 实, Ep 虚]`（直接作为 `Jones2D`，不要再用 `jones_from_angles_2d` 重算）、`laurent`、远场 / 近场 atol、rtol。
* `Rcwa2dRefOrder orders[N]`：m、n、Kx、Ky、kzI、kzII、Ex_r…Ez_t、Rs、Rp、Ts、Tp、DEr、DEt、prop_r、prop_t（复数是 `{实部, 虚部}`），与 Python 同序（下标 = `harmonic_index_2d`）。
* `Rcwa2dRefDepth depths[]`：若干 z（N ≤ 25 取 3 个，其余 2 个：区域 I、层内、区域 II 之一）及其 N 个谐波列 `columns[i].c[分量][0/1]`。
* `Rcwa2dRefOperators operators[]`：只有 N ≤ 35 的结构层算例（`y_inv_*`、`x_inv_*`、`lx_ne_ly`、`multi_yseg`、`m2_zero`、`m1_zero`）：N×N 的 Eps、Cx、Cy，**按行存放**（元素 (i, j) 的实、虚部在 `[(i*N+j)*2 + 0/1]`；C++ Matrix 是列主序，装配时下标要显式写）。更大算例的算子在 NPZ。
* `status = 1`（23–25）：`failure_region` 与 `failure_orders[failure_count][2]`。
* 数字用最短可往返的十进制写出，读回得到相同的 double；表只含 N ≤ 169 的算例，N=625 的算例与全部切面 / 相位数组只在 NPZ。

## 已知限制

* 参照的正确性来自三条互相独立的证据：一维极限对 `rcwa1d` / `conical`（逐位 ~1e-13）、均匀结构对 Airy 解析式（~1e-15）、真二维柱对 COMSOL 三维有限元（对照报告没有随仓库发布；复振幅 1.3e-4，近场 4e-4–1.6e-3，**不是**逐位）。
  对真二维图案，本目录的冻结数据是**回归锚点**，不是独立真值；C++ 通过它说明与 Python 参照一致，物理正确性依赖上述三条。
* 真二维算例的近场没有独立的逐位判据（只有界面连续、能量守恒与 FEM 的统计比较）。
* `gui/2drcwa/README.md` 在本仓库不存在（原求解器包里有），不属于本交付。
