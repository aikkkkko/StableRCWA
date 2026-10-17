# C++ 近场移植的固定 Python 参照（合同 v1）

[数值约定、已知边界和 C++ 模块设计](CONTRACT.md)。第一步固定合同、数据和验证；C++ 场上下文、正式采样、相位与单次求解的 GUI 路由均已实现。

`reference_v1.npz` 是提交到源码库的小型冻结数据；`reference_v1.json` 是来源、输入、容差和布局清单。
生成器不会在测试中运行，不会自动覆盖已冻结的数据。参照正确性同时受独立解析与现有退化场测试约束。
`source_sha256` 记录的是冻结时的数值源码；此后 `gui/rcwa1d.py`（只改了文档字符串）与 `gui/gui_fields.py` 有改动，
测试不比较这一项，用现在的代码重新生成仍逐位得到同一份数据。

从仓库根目录运行（Python 需 NumPy、SciPy、Matplotlib）：

```sh
MPLCONFIGDIR=/tmp/stablercwa-mpl /usr/bin/python3 -m unittest discover -s gui/validation -p test_field_contract.py -v
MPLCONFIGDIR=/tmp/stablercwa-mpl /usr/bin/python3 -m unittest discover -s gui/validation -p test_degenerate_fields.py -v
```

可给第二条命令加 `RCWA_JSON_EXE=/tmp/stablercwa-implementation-build/rcwa_json`，检验既有 C++ Schur Jones 远场。
正式近场另由下方公共 API 与完整协议测试覆盖。

验证第二步的 C++ 上下文（测试内访问缓存并重建谐波，不发布采样 API）：

```sh
cmake -S . -B /tmp/stablercwa-fields-context-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/stablercwa-fields-context-build --target test_rcwa_field_context
ctest --test-dir /tmp/stablercwa-fields-context-build -R test_rcwa_field_context --output-on-failure
RCWA_FIELDS_EXE=/tmp/stablercwa-fields-context-build/test_rcwa_field_context MPLCONFIGDIR=/tmp/stablercwa-mpl /usr/bin/python3 -m unittest discover -s gui/validation -p test_cpp_field_context.py -v
```

最后一条命令只读本目录的冻结数据，把 C++ 快照重建出的六复谐波、物理波矢与全部 12 例参照逐值比较；
再使用固定 x/y 和 carrier 验证六复场。不会重新生成或覆盖基线。

显式生成候选数据，以便比对或验证可复现性：

```sh
MPLCONFIGDIR=/tmp/stablercwa-mpl /usr/bin/python3 gui/validation/generate_field_reference.py --output /tmp/stablercwa-field-candidate
```

任何未来基线更新都需要审查来源 hash、复场差异与独立验证结果，然后显式使用 `--refresh-baseline`。
不应为了让回归测试变绿而刷新数据。冻结文件的 hash 写入 JSON，读取时检查内容未被意外改变。

每个 case 的键为 `case_id__array_name`。复数为 complex128，实数为 float64，mask 为 bool，级次为 int64。
`fields` 为 `[component,z,x]`，`harmonics` 为 `[component,z,order]`，均为 C 连续序；
分量顺序恒为 `[Ex,Ey,Ez,Hx,Hy,Hz]`。`phase_absolute/phase_incident` 使用同样的 `[component,z,x]`，
掩膜位置填 NaN，`phase_mask` 明确保留无定义位置。
`phase_comparison_mask=true` 的点不参加跨后端相位比较：包括结构零分量的舍入噪声，以及弱于复场误差地板的点。
原始 phase/mask 仍完整保存。比较相位用模 2π 等价的单位相位因子，容差与场误差/幅值相联系，规则写在 JSON。
`kx/kzI/kzII/ky` 是有长度倒数单位的物理波数，其中场的横向因子为 `exp(-i(kx x+ky y))`。
`incident_amplitudes=[|E_inc|,|Z0H_inc|]` 只说明尺度，保存的复场不统一重缩放。

固定数据用于数值回归；单界面/薄膜解析、φ=0 映射、GUI 与通用 phase 合约测试属于独立验证。
严格零根解析、Schur 4N 一阶指数、厚吸收、远离结构的未激发倏逝保护继续由 `test_degenerate_fields.py` 负责。
本批不包含外部精确截止、圆锥均匀层 `0<|q|<1e-13` 的夹取区，也不连接 COMSOL。


验证正式 C++ 采样、相位和 GUI 完整协议（从仓库根目录）：

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
RCWA_PUBLIC_FIELDS_EXE="$PWD/build/test_rcwa_public_fields" MPLCONFIGDIR=/tmp/stablercwa-mpl python3 -m unittest discover -s gui/validation -p test_cpp_public_fields.py -v
RCWA_FIELDS_EXE="$PWD/build/rcwa_fields" MPLCONFIGDIR=/tmp/stablercwa-mpl python3 -m unittest discover -s gui/validation -p test_cpp_fields_backend.py -v
```

公共接口测试直接调用生产 `field_at` / `field_map` / `phase_data`，包含固定数据、独立解析和 4N 一阶参照。
完整协议测试检查单次 C++ worker、禁用 Python 数值函数后仍成功、坏应答回退、中文路径、超时与取消、旧归档兼容。
