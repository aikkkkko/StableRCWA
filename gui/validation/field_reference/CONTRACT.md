# 一维 / 圆锥 C++ 近场移植：Python 数值合同与模块设计

合同版本 1，数值源码基线是发布前的内部开发版本（不在公开的 git 历史里）。
第一步固定已有行为和参照。第二步已实现可实际调用的 **`solve_with_fields` / `solve_conical_with_fields` 与拥有数据的场上下文**，
同一次求解保存模式、材料算符和层顶 / 层底振幅。正式 `field_at` / `field_map` / `phase_data`、NPY 协议和单次 C++ GUI 路由也已实现。冻结的 Python 数值合同和数据保持原始版本。

## 1. 合同范围与依据

本合同覆盖非磁性、各向同性材料、分段解析 Fourier 算符的周期一维 TE/TM 与圆锥入射。
Python 平面各向异性、FFT 算符、PML、二维和磁性扩展不属于此次 C++ 移植验收范围。
长度单位任取但必须一致；GUI 的长度单位是 μm。`n=n_real+i*k`、`k>0` 表示吸收，ε=n²，μ=1。
圆锥 Python 要求两侧半无限介质的折射率为实数。

权威实现入口：

- `gui/rcwa1d.py`：`solve`、`field_at`、`field_map`、`_eps_ops_at`。
- `gui/conical/rcwa_conical.py`、`modes_conical.py`、`spbasis.py`、`polarization.py`：圆锥模式、Jones 与坐标变换。
- `gui/conical/fields_conical.py`：圆锥六分量谐波和 x-z 场图。
- `gui/mode_propagation.py`：普通、严格零根、Schur 的共同场组合。
- `gui/field_phase.py`：通用相位；`gui/gui_fields.py`：保存结果的 GUI 显示。

参照的 JSON 保存这些文件的 SHA-256，另记 Python / NumPy / SciPy 版本、输入参数、数组结构与容差。
源码 hash 是来源证据，不代替场数值与独立解析验证。

## 2. 坐标、相量、传播与界面归属

时间因子为 `exp(-iωt)`。z 从入射侧向下，第一层顶面为 0，最后一层底面为 `z_bot=Σd_j`。
衍射级按 `m=-M,…,+M` 排列，零级下标 M。令 k0=2π/λ，

```text
平面： kxm = k0(nI sinθ - m λ/P)，ky=0
圆锥： kxm = k0(nI sinθ cosφ - m λ/P)，ky=k0 nI sinθ sinφ
横向实空间因子： exp[-i(kxm x + ky y)]
```

因此物理传播波矢的横向分量是 `(-kxm,-ky)`；θ>0、φ=0 时入射指向 -x。
均匀介质 `kz=sqrt(k0²n²-kxm²-ky²)`，传播取实正、倏逝取正虚部。
q=-i kz/k0，常规下行传播为 `exp(-k0 q Δz)=exp(+i kz Δz)`。
非均匀模式的分支按生产 `_branch` 选择，不把近零根擅自截成零。

- 区域 I：`z<0`，入射 `exp(+i kzI z)`，反射 `exp(-i kzI z)`。
- 区域 II：`z>z_bot`，透射 `exp(+i kzII(z-z_bot))`；T 锚定在结构底面。
- 层内：a_plus 锚定层顶、a_minus 锚定同一层底。两者分别向下、向上用非增长距离传播。
  不能把 a_minus 当作层顶振幅，不能从 R/T 单独恢复内场。
- 外部区域优先判断；否则从第一层开始选择 `z <= z_top+d+1e-12` 的第一层。
  存在正厚有限层时，正好在内界面、或其下方不超过 1e-12 的采样，归属上层；`field_combinations` 再将层内距离夹到 `[0,d]`。
  `z=0` 归第一层，`z=z_bot` 归最后满足条件的层；`z>z_bot` 即使只多 1e-13 也归区域 II。
- 保留任意 z 输入顺序、重复深度与不均匀深度；场图行与输入 z 一一对应。
  x_j=j P/nx，j=0,…,nx-1，不重复周期终点。场图为 `[nz,nx]`，C 顺序中 x 最快。
  圆锥的固定 y 贡献公共因子 `exp(-i ky y)`。

仅对激发的入射零阶求入射指数。未激发的倏逝入射阶不能先求增长指数再乘零；在远离上表面的采样会产生 `0×Inf`。
反射倏逝波在 z→-∞ 衰减，透射倏逝波在 z→+∞ 衰减。

## 3. 六个复分量与入射归一化

统一复分量顺序为 **Ex、Ey、Ez、Hx、Hy、Hz**。存储的 H 都是 `Z0 H_physical`，
与电场使用同一振幅尺度，并非 SI 的 A/m。若要恢复 SI，应另给电场振幅 A：`E_SI=A E`、`H_SI=A H_stored/Z0`。
`rcwa1d.py` 顶部“E 以 Z0 归一”的文句不够明确，以以下公式及 `gui_fields.normalization` 的实际实现为准。

令 `Kx=diag(kxm/k0)`，ε 的卷积矩阵记 Eps。

| 求解 | 入射振幅 | 原始复场谐波 | 严格为零的分量 |
| --- | --- | --- | --- |
| 平面 TE | Ey_inc=1；\|E_inc\|=1，\|Z0H_inc\|=nI | Ey=S，Hx=iU，Hz=-Kx S | Ex、Ez、Hy |
| 平面 TM | (Z0Hy)_inc=1；\|E_inc\|=1/nI，\|Z0H_inc\|=1 | Hy=S，Ex=-iU，Ez=solve(Eps,Kx S) | Ey、Hx、Hz |
| 圆锥 | Jones 电场范数为 1；\|E_inc\|=1，\|Z0H_inc\|=nI | 见下式 | 一般无恒零分量 |

平面 `field_at` 返回 `(S,U)` 谐波，**没有直接返回六个 Cartesian 分量**。
层内 S=W·sum，U=V·difference。区域 I 为
`S=inc+R·e_re`，`U=iP(inc-R·e_re)`；区域 II 为 `S=T·e_t`、`U=iP S`。
P= kz/k0（TE）、kz/(k0 n²)（TM）。对实数入射介质、θ>0 的单位下行波，TE Hx 为负、TM Ex/Ez 为正；θ<0 时 TM Ez 反号。

非均匀 TM 的 Ex 通过 V 中的 `⟦1/ε⟧` 逆规则，Ez 通过 `solve(⟦ε⟧,Kx S)`；
不得用逐点 ε(x) 除谐波和替代，不能把这两个算符混为一谈。
原始平面 `field_map` 只返回三个非零分量与 E2；GUI 用 `complete_six_components` 补零。
平面 TE 的 E2=\|Ey\|²，TM 的 E2=nI²(\|Ex\|²+\|Ez\|²)，均已按入射电场强度归一。

圆锥内部用 `h=i Z0H`，物理切向状态为 `F=[Es;Et]`、`G_phys=[-ht;hs]`。
逐级的 `c=cosφ_m`、`s=sinφ_m`，`φ_m=atan2(ky,kxm)`；`kt<=1e-14` 时延拓入射方位角。
按前 N / 后 N 两块取 `F=[es;et]`、`G_phys=[g1;g2]`，

```text
Ex=-s es+c et， Ey=c es+s et
hx=-c g1-s g2，hy=-s g1+c g2
Ez=i solve(Eps, kappa hx-Kx hy)，kappa=ky/k0
hz=i(kappa Ex-Kx Ey)
(Hx,Hy,Hz)=-i(hx,hy,hz)
Dx=C Ex，C=solve(⟦1/ε⟧,I)；均匀层 C=ε
```

圆锥 `field_at` 返回上述六谐波及 Dx/F/G；`field_map` 返回六复场、Dx、E2/H2 和 x/z/y。
圆锥 E2/H2 原始值分别为六场的平方和；GUI H2 显示再除 nI²，E2 显示不再缩放。

物理 Maxwell 系统是 `dF/d(k0z)=+P G_phys`、`dG_phys/d(k0z)=+Q F`。
`test_degenerate_fields.direct_layer` 的块指数含 `-P,-Q`，其辅助状态是 `[F;G_aux]`、`G_aux=-G_phys`；
`direct_field` 返回的第二项已转回 `G_phys=+i(a-b)`，不能据内部反号系统把生产场的 G 再翻一次符号。

## 4. 圆锥偏振、φ=0 与模式传播

入射 Jones 次序 `[Es,Ep]`；角度入口为 `[sinψ exp(iδ),cosψ]`，δ 是 s 相对 p 的相位。
ψ=90° 为 s（浮点 cos90° 不一定精确为零）；ψ=0° 为 p。
显式非零 Jones 输入除以二范数，**保留整体复相位**。严格 s/p 的参照用 `[1,0]` / `[0,1]`。
入射模式两列 `[s,p]`，s 零级切向振幅为 1，p 零级 Et 为 cosθ；两列的范数均对应单位完整电场。
Jones 矩阵布局为 `[order,out_s_or_p,in_s_or_p]`。反射 p 的切向方向多一个负号：
Jr_p=-R_tangential_p/cosθ_m，Jt_p=T_tangential_p/cosθ_m。

φ=0 且入射 kx0>0 时，圆锥 s 六 Cartesian 复场等于平面 TE 六场；圆锥 p 六场等于 **nI 倍**平面 TM 六场。
衍射 Jones 的逐级 s/p gauge 还带 `σ_m σ_0`，σ_m 是逐级 kx 的符号（法线级延拓 +1）。
不得把效率相等误写为原始 TM 全场振幅直接相等；不必对 Cartesian 场额外乘 σ_m。

普通模式：令 `p=exp(-k0 q z_local)a_plus`、`m=exp(-k0 q(d-z_local))a_minus`，
则 sum=p+m、difference=-p+m。

平面有限层 **严格 q==0** 用辅助界面基，q 本身仍为零；V 的相应 q 因子取 1（TM 再含逆规则）。
令 h=k0d、ξ=k0z_local，则该零根的组合为

```text
sum = 2/(2+h) [(1+h-ξ)a_plus+(1+ξ)a_minus]
difference = 2/(2+h) (-a_plus+a_minus)
```

它保留常量与线性解；这些辅助振幅不能解释为两束指数波。近零但非零走普通指数，不任意改本征值。
圆锥 Schur 层则使用完整生成元 B：`p=exp(-k0 z_local B)a_plus`、
`m=exp(-k0(d-z_local)B)a_minus`，B 的上三角非对角项必须保留。
每列是 Schur 子空间而非独立模式，只保留 `diag(B)` 会丢 Jordan 耦合。

## 5. 两套相位入口

相位必须在全部谐波重建实空间复场后计算。实强度 E2/H2 没有复相位。
绝对相位为 `arg(F)`，包裹范围沿 NumPy angle/atan2 为 [-π,π]。
相对入射相位仅去除 **空间 carrier**：
`arg(F conj(exp(i[-Re(kx0)x-Re(ky)y+Re(kzI0)z])))`。
不会扣掉偏振分量方向负号、Jones 的整体相位或偏振间 δ，故纯 TE Hx 的相对相位仍可为 ±π。
参考波只有单位模，不把吸收/倏逝参考波幅值乘进显示值。

| 行为 | 通用 `field_phase.phase_data` | GUI `gui_fields.view_data` |
| --- | --- | --- |
| 场尺度 | 原始复场，不自动归一 | 先分别除 \|E_inc\|、\|H_inc\| |
| 绝对 / 相对入射 | absolute / incident（别名 none / input） | phase / phase_relative |
| carrier 来源 | Result 的物理 kxm、ky、kzI/kIz 零级 | config 的 k0、nI、θ、φ（约定一致，当前范围中数值在容差内一致） |
| 掩膜 | 非有限 F；\|F\| <= floor·本分量有限幅值峰值；参考非有限或为零 | 非有限 F；\|F\| <= 1e-6·本分量峰值 |
| 零场 | 全部掩膜，即使 floor=0 | 全部 NaN 并标 phase_undefined |
| 自定义参考 | 允许广播；`arg(F conj(R))`，返回原参考幅值；R=0 无效 | 无此入口 |
| 矢量分析器 | `projection^H @ field`，三复数，不归一化 projection | 无此入口 |
| 展开 | 可选 x/z/y/horizontal/vertical/both | 无此入口 |
| 返回 | phase 是 masked array，amplitude/field/mask 等单独返回 | 普通实数组，以 NaN 表示掩膜 |

通用 unwrap 对每条线的连续有效段调用 NumPy 默认 unwrap；跳变绝对值大于 π 时加减 2π，不跨掩膜节点。
both 在本次 x-z 图上先 x 后 z；这不是全局二维相位展开。对转置网格按坐标匹配识别轴，方形网格有歧义时优先 `[z,x]`。
投影不除向量范数，故它的幅值也按原投影尺度返回。自定义参考只保证输入值的有限/非零掩膜，不保证极端乘积不溢出。

## 6. 已确认边界差异与待裁决

这些是当前行为，第一步不修改生产代码，也不把已知数值夹取升格为物理定义。

1. 无层 stack 的 `z=0`：平面与圆锥 `field_at/field_map` 都无法选择层并抛 ValueError。
   单界面冻结参照只采 z<0、z>0。C++ 正式采样明确扩展为：无正厚有限层时，z=0 取出射侧，纵向分量使用出射介质材料。
2. Python 圆锥均匀层（包括 d=0）只要 q==0 就抛 RayleighAnomalyError；平面有限层已支持严格零根。
   C++ 圆锥先过滤 d=0 层，因此不一定复现该失败。两侧外部截止也不能用平面有限层辅助基替代。
3. Python 圆锥均匀层 `0<|q|<1e-13` 时发 RuntimeWarning，q 保持不变，但 p 基分母夹为 **1e-13+0j**，
   即 `V_p=-n²/qsafe`，会改变极近截止的幅值及相位。C++ 采用有限 p 基 `w=kz/n,v=-i n`。
   未来保留 C++ 自有有限基；这些极近截止点不属于本批逐数值对拍域，不能用 Python 夹取结果作物理真值。
   本批圆锥均匀根均远离该阈值，明确零根失败另测；后续若扩大域需独立 Maxwell/解析验证和单独容差。
4. 圆锥 Python 接受 need_fields 后保留所有原始层；C++ 过滤零厚层。C++ 上下文已同时保存原始层索引与有效层索引，
   零厚条目的 effective_index 为 -1、has_modes/propagates 为 false，不为它新增本征分解。
   平面保留原有零厚层与模式；有正厚层时继续上层优先规则，全部层厚为零时 z=0 取出射侧。
5. C++ 效率-only 路径在自底向上级联后释放模式 W/V，且 g[b] 只留下行推进。
   原 `Result/ConicalResult` 无法事后直接恢复内场；可选新入口已在同一次求解中保留状态，旧入口继续释放模式。

## 7. C++17 header-only 模块边界与当前状态

不引入第三方库。保留现有 `solve/solve_conical` 签名与效率-only 路径；新入口已共享求解实现，按需保留近场状态。
场与相位头文件直接包含使用；GUI 默认完整 C++ 路由已接入，原聚合头继续用于核心与远场。

| 新模块 | 职责 | 依赖 |
| --- | --- | --- |
| `rcwa_fields.hpp` | 六分量枚举、状态、元数据、层映射、HarmonicFields 与 FieldGridXZ | matrix.hpp 与标准库 |
| `rcwa1d_fields.hpp` | solve_with_fields、拥有的模式/材料/零根状态、field_at 与 field_map | rcwa1d.hpp、rcwa_fields.hpp |
| `rcwa1d_conical_fields.hpp` | solve_conical_with_fields、两列入射响应、Schur 耦合与 Jones 后选采样 | rcwa1d_conical.hpp、rcwa_fields.hpp |
| `rcwa_phase.hpp` | 已实现 E/H 分量与复投影、参考 carrier/custom、弱场 mask、x/z/both 顺序分段 unwrap | rcwa_fields.hpp 与标准库；不依赖求解器 |

已实现 `FieldComponent`、`FieldNormalization`、`FieldMetadata`、`FieldLayerInfo` 与 `FieldContextStatus`。
`metadata()` 给出物理 kxm/ky/kzI/kzII、完整入射矢量范数、原始层几何及有效层映射。
已实现的谐波、网格与相位容器使用以下语义和数组序：

- `FieldComponent {Ex,Ey,Ez,Hx,Hy,Hz}`，六序列；复杂数据用项目 cplx，坐标用 double。
- `FieldNormalization`：`electric_incident_amplitude` / `magnetic_incident_amplitude`，H 明确为 Z0H；
  原始复场不默默改尺度，另提供 GUI 对齐的强度/显示转换。
- `HarmonicFields`：orders、六分量谐波。每分量 Matrix `[order,depth]`，项目列存储使单深度各谐波连续。
  Python NPZ 的 `[component,depth,order]` 是交换格式，转换时不能按相同线性下标照抄。
- `FieldGridXZ`：x/z/y；每分量 Matrix `[nx,nz]`，列存储等价 Python `[nz,nx]` 的 C 顺序；
  公开访问器 `at(component,iz,ix)` 明确轴序，网格布局在序列化时写入。
- `PhaseGrid`：同一 `[nz,nx]` 逻辑布局的 phase/amplitude 与显式 mask（true/1=无定义或被掩膜），单位 rad；
  无定义值显示时转 NaN。mask 必须保留，不能让零场变成“0 相位”。
- 求解结果拥有近场快照，不持有调用者 Stack 的裸指针；保留波矢、界面位置、材料算符、层映射和诊断，
  采样时不再做本征分解。圆锥近场保存两列入射基，后选非零 Jones 归一到范数 1 并保留整体相位。

当前可调用的求解入口：

```cpp
PlanarFieldResult solve_with_fields(stack, wavelength, theta_deg, M, pol, options)
ConicalFieldResult solve_conical_with_fields(stack, wavelength, theta_deg, phi_deg, M, options)
// .result 为旧 Result / ConicalResult；.fields 为拥有缓存的上下文。
// fields.available() / status() / metadata() / require_available()
```

context 可深拷贝，移动后源对象为 Empty 且元数据重置。数值失败时为 Unavailable，require_available() 抛 logic_error；
无层成功上下文为 Ready、layers 为空，上下介质及响应齐全；z=0 按出射侧采样。
内部矩阵和振幅是实现细节，仅测试通过 friend 访问，不是稳定的公开 API。

已实现的采样与相位入口（可传结果对象或其 fields 上下文）：

```text
HarmonicFields field_at(field_result, z_or_zs [, jones])
FieldGridXZ field_map(field_result, zs, nx [, y, jones])
PhaseGrid phase_data(grid, component, phase_options)
PhaseGrid phase_data(grid, FieldVectorFamily::E_or_H, projection, phase_options)
```

C++ `field_at` 统一返回六谐波；Python 平面 `(S,U)` 与圆锥 dict 的差异继续记录在参考格式中。
错误输入抛 invalid_argument；数值失败保持现有求解结果 NaN/诊断惯例，并让近场快照标为不可采样，禁止返回貌似有效零场。

共享求解采用可选内部 capture 指针，保留、采样和释放流程均已实现：

1. 构造模式时按需保留 W/V（解析区域保留对角，不膨胀成稠密矩阵）、q、严格零根标志或完整 Schur 生成元。
   平面 TM 同时保留构造层算符时原 Eps 求解的 LU/QR 分解；平面 TE 保留 Eps，均匀快速路径用标量。
   圆锥移动已有 Eps 逆表示与 C；不额外本征分解或求逆，不独立归一 Schur 列。
2. 自底向上时，在每层底面保存 **该面以下的反射 Op r_below** 或等价恢复数据。
   g[b] 仅给下行推进，不够恢复 a_minus。只复制需要的每层数据，不把 Python 全部 blocks/rud/downs 照搬到公开结果。
3. 自顶向下时捕获层顶 a_plus；通过传播块得到层底下行后，`a_minus=r_below·down_at_bottom`。
   圆锥同时捕获两列；不能用 `exp(+k0 q d)` 反推层底振幅，厚吸收/倏逝会增长失稳。
4. 普通 / 零根 / Schur 的层内组合与各自散射块共用传播定义。采样按层分组，保留 z 输入顺序；
   Schur 按距锚点的距离排序，固定八槽缓存只复用完全相同的步长；大绝对传播相位时逐点从锚点传播。输出内存 O(N·nz)，步进缓存 O(N²)，不随深度数增加矩阵缓存。
5. retain_fields=false 时仍按现有时机释放模式、只保留 g 和当前 r；不额外保存 Eps、r_below、振幅或场图。
   本阶段可用分配计数与同结构前后基准验证；不把新增缓存的额外成本算作旧入口成本。

## 8. 参照与验收

本批 12 个确定性小参照保存 **完整复场和六复谐波**，还保存绝对/相对相位及 mask，见 [README](README.md)。
普通 TE/TM、吸收+倏逝、φ=0 s/p、普通圆锥线偏振、Schur 椭圆偏振、严格有限层零根均有覆盖。

- 固定 NPZ 是回归锚点；测试只读，普通测试与生成器都不会自动替换它。
- 独立 Fresnel 单界面、薄膜边界线性系统验证六复场，检验符号、幅值、传播相位与 z 锚点。
- φ=0 逐分量复数比对时包含 TM 的 nI 归一化，不仅比较 E2 或效率。
- 复用 `test_degenerate_fields` 的零根解析解与 4N 一阶指数/散射参考，Schur 验证完整六分量。
- 合成 phase 测试独立固定 carrier、阈值等号、全零 mask、投影共轭、自定义参考、跨 mask 禁止连接和先 x 后 z 展开。
- 正常、零根、Schur 分别使用 JSON 指定容差；近场重建的独立传播误差和相位差不可拿远场效率一致替代。
- 后续 C++ 通过同一数据中每个复数的 `|a-b|<=atol+rtol|b|`；不能只比模平方，不能先重新生成“基线”再验自己。

原始相位 mask 依本分量峰值，φ=0 圆锥的结构零分量可能因约 1e-17 的舍入噪声被视为有效。
这属于原始快照，不能要求 C++ 复现噪声相位。JSON 明确每例的结构零分量，仅检验其复场最大幅值不超过场 atol。
单独的 `phase_comparison_mask` 中 true 表示跨后端排除：原 phase_mask、结构零分量，或
`|F|<=10·(atol+rtol|F|)` 的点。其余点比较单位相位因子的弦差
`|exp(iφ_actual)-exp(iφ_ref)|`，按模 2π 等价；容许量为 `2·(atol+rtol|F_ref|)/|F_ref|+5e-14`。
这样正常场的相位容差随复场误差与幅值关联，不把弱场的随机相位当作跨后端真值；原始相位数据不被改写。

第二步增加 `tests/test_rcwa_field_context.cpp`，直接验证材料 LU/QR 表示、层顶/层底锚点、严格零根、混合 Schur 层、
无层、零厚映射、厚吸收、Stack 生命周期独立性、深拷贝/移动和失败拒绝访问。
`test_cpp_field_context.py` 调用该 C++ 测试的仅测试导出入口，用快照重建全部 12 例六复谐波及六复场，并按冻结容差比较。
`test_rcwa_public_fields` / `test_cpp_public_fields.py` 另外通过正式 API 验证六复场、相位和独立一阶参考。GUI 使用同一正式采样与相位入口。

本工作区第二步的复核结果：13 个 C++ CTest 与原有 25 项 Python 合同/退化测试通过；新增冻结对拍通过全部 12 例。
逐分量最大差：普通六复场 `3.338e-15`，Schur 六复谐波 `2.195e-12`、六复场 `2.813e-12`，
严格零根 TE/TM 六复场分量差分别 `6.753e-16` / `4.450e-16`。新上下文目标及新旧入口比较另经 ASan/UBSan 验证。
用同一 3 层混合结构、M=12 的独立分配计数程序比较旧入口：平面 TM 修改前后均为 176 次、898596 字节；
圆锥均为 230 次、3213216 字节。该计数为此基准每次求解的累计分配。


## 9. 完整 CLI、GUI 与复核

`rcwa_fields` 一次求解获得远场与上下文，再由正式采样/相位接口输出六复场及 absolute/incident 两套包裹相位。
JSON 传参数、版本与诊断，五个 NPY 文件传数组；临时目录完整写入后统一发布。Python 校验请求标记、类型、尺寸、布局、文件长度和掩膜后使用。
GUI 的 C++ 成功路径跳过 Python 数值求解、场重建和相位计算；`RCWA_VERIFY_PYTHON=1` 仅为显式效率/Jones 核对模式。
界面保留现有两种包裹相位显示；通用 C++ API 额外支持 custom 参考、复投影和展开。

`RCWA_FIELDS_EXE` 指定完整程序；`RCWA_FIELD_CONTEXT_EXE` 与 `RCWA_PUBLIC_FIELDS_EXE` 分别指定两种测试导出器。
二维、PML、各向异性和 FFT 路径继续使用 Python；完整 C++ 输出失败时整次回退，原因写入结果。
旧归档按原相位规则显示，新归档保留后端、相位版本和掩膜；相位切换不触发求解。

曾修正两处可复现的采样问题：严格零根在大厚度处改用分别距两面的系数，避免 `(1+h)-h` 丢失常量项；
Schur 取消随绝对坐标增长的近似步长匹配，精确步长缓存和大相位直接传播通过专门反例。
401×256 网格的本机采样测试约为平面 TM M12 4.4 ms、Schur M8 9.8 ms；逐次调用 401 个 Schur 单点约 202 ms。
这些计时针对场采样，完整 GUI 还包含进程启动、相位、传输和保存。

复核：在真实浏览器里完成圆锥求解、相位切换及六分量下载，界面显示的效率、场图、相位均来自一次 C++ 求解。
把 Python 求解、场重建、相位函数和 numpy.angle 都替换为抛错函数后，两个完整 GUI worker 仍然成功；
对原 Python 结果的六复场最大差为 3.36e-14 与 9.48e-13。公开 API 与完整 CLI 的 12 冻结例最大差为 2.813e-12。
