# 二维 RCWA 的 JSON / 命令行 / GUI 协议（合同 v1 的延伸）

本文把一维已有的两个协议扩展到 `periodic2d`：**远场协议**（`rcwa_json`，见 `include/stablercwa/rcwa_json.hpp` 文件头）与**场协议**（`rcwa_fields`，
见 `include/stablercwa/rcwa_fields_json.hpp`）。一维的请求/应答/文件格式一律不变；二维只增加新的 `mode` 与新的采样键。
C++ 一侧用 `include/stablercwa/rcwa2d.hpp`（`solve2d`）与 `rcwa2d_fields.hpp`（`solve2d_with_fields`、`field_at`、`field_slice`、`phase_slice`）；
数值合同见 `CONTRACT_2D.md`，冻结参照见 `README.md`（本目录）。风格与一维一致：**不认识的键一律报错**，不得悄悄忽略；中文错误信息，带出错位置（`where`）。

## 1. 远场请求（`rcwa_json`，`"mode": "periodic2d"`）

```json
{"mode": "periodic2d",
 "stack": {"nI": 1.0, "nII": 1.46, "Lx": 0.8, "Ly": 0.8,
           "layers": [{"d": 0.3, "y_edges": [0.0, 0.25, 0.55, 0.8],
                       "x_profiles": [{"edges": [0.0, 0.8],              "n_values": [1.0]},
                                      {"edges": [0.0, 0.25, 0.55, 0.8], "n_values": [1.0, 1.8, 1.0]},
                                      {"edges": [0.0, 0.8],              "n_values": [1.0]}]},
                      {"d": 0.115, "n": [1.96, 0.0]}]},
 "M": 4, "My": 4,
 "points": [{"wavelength": 1.053, "theta_deg": 42.0, "phi_deg": -25.0}],
 "input": {"psi_deg": 90.0, "delta_deg": 0.0},
 "options": {"eig": "qr", "factorization": "li", "homogeneous_fast_path": true,
             "parallel_layers": false, "diagnostics": false, "threads": 1}}
```

- 折射率写法、`points` / 最外层单点的两种写法、`threads` 的含义与范围、`input` 的两种写法（`psi_deg`/`delta_deg` 或 `jones`）都与一维 conical 相同；
  **二维必须给 `input`**（不写不能默认，免得悄悄取 s）；点里的 `phi_deg` 不写取 0。`nI` 必须是正实数；`nII` 可以是 [实部, 虚部]（吸收出射介质合法）。
- `M` 是 x 方向截断（m = −M…M），`My` 是 y 方向截断（n = −My…My）；两者都不写时报错（二维没有"默认 M"）；`My` 可以为 0。
  总谐波数 N = (2M+1)(2My+1) 必须 ≤ **2048**（`kJsonMax2dHarmonics`，在分配任何东西之前检查；GUI 自己的上限是 625，见 gui_models_2d.py）。
- 层：均匀层 `{"d", "n"}`；结构层 `{"d", "y_edges": [K+1 个递增边界，首尾差一个 Ly], "x_profiles": [K 个 {"edges": [递增，首尾差一个 Lx], "n_values": [...]}]}`。
  两种写法不能混用；零厚度层合法（保留，与 Python 一致）。几何非法（边界不递增、不恰好覆盖一个周期、个数不符、折射率非有限…）→ 请求错误，信息来自 C++ 的 `invalid_argument` 并带层号。
- `options`：`eig` = "qr"（默认）| "lr"（接受；二维上 LR 通常因残差检查返回数值失败，见 rcwa2d_types.hpp）；`factorization` = "li"（默认）| "laurent"；
  `homogeneous_fast_path`、`parallel_layers`、`diagnostics`、`threads` 同一维。**没有 `linear`**（二维求解没有这个选项，写了就报错，说明"二维不适用"）。
- `mode: "periodic2d"` 与一维的 `period`、`polarization` 同时出现 → 请求错误。

## 2. 远场应答

一维的头部不变：`{"ok": true, "version": 1, "mode": "periodic2d", "M": M, "My": My, "orders": [[m, n], ...], "threads": t, "results": [...]}`，
`orders` 是长 N 的 [m, n] 数组，**次序就是 `harmonic_index_2d`（m 在外层、n 在内层）**；每个入射点一个结果：

```json
{"wavelength": 1.053, "theta_deg": 42.0, "phi_deg": -25.0,
 "status": "ok", "message": "",
 "Kx": [...], "Ky": [...], "kzI": [[re, im], ...], "kzII": [[re, im], ...],
 "Rs": [[re, im], ...], "Rp": [...], "Ts": [...], "Tp": [...],
 "Ex_r": [...], "Ey_r": [...], "Ez_r": [...], "Ex_t": [...], "Ey_t": [...], "Ez_t": [...],
 "DEr": [...], "DEt": [...], "prop_r": [0, 1, ...], "prop_t": [...],
 "eig_info": 0, "n_eig": 1, "n_dense_solve": 2, "max_eig_residual": 0.0, "max_solve_backward_error": 0.0,
 "timing": {"total": 0.1, "operators": 0.01, "eig": 0.05, "linear": 0.02, "product": 0.01}}
```

- 数组长度都是 N，次序同 `orders`；复数是 [实部, 虚部]；数值与直接调用 `solve2d` 读回来**逐位相同**（一维的同一条性质）；`Kx`、`Ky`、`kzI`、`kzII` 是以 k0 为单位的值。
- **失败**：`ok` 仍为 true（与一维"数值失败写成 null，ok 仍为 true"一致），`status` = "rayleigh_anomaly" | "numerical_failure"，`message` 原样来自 `Result2D::message`，
  所有数组里的数值为 null（`prop_*` 照常），`eig_info` 等照常填写。参数非法（几何、M、N 超限、角度…）才是 `ok: false` 的请求错误。
- `max_eig_residual` 与 `max_solve_backward_error`：`diagnostics=false` 时写 0（与一维相同的约定）；二维当前没有求解后向误差，写 0。
- `Ez_r`、`Ez_t` 是**远场约定的 Ez**（`−sign·(Kx·Ex+Ky·Ey)/kz`，只在传播级，其余为 0），**与近场 Ez 符号相反**（见 `rcwa2d_fields.hpp` 文件头），不要当近场 Ez 用。

## 3. 场请求与应答（`rcwa_fields`）

请求：`{"solve": <上面的二维请求，恰好一个入射点>, "sampling": {"z": [...], "plane": "xz" | "yz", "nt": 128, "at": 0.4}, "output_dir": "新目录", "request_id": "令牌"}`。

- 二维的 `sampling` 只认 `z`、`plane`、`nt`、`at`（`plane` 与 `nt` 必填，`at` 缺省 0）；一维的 `nx`、`y` 在二维里是未知键 → 报错；一维的 `sampling` 里写 `plane`/`nt`/`at` 同样报错。
  `plane = "xz"`：固定 y = `at`，沿 x 取 `nt` 个点（x_j = j·Lx/nt，不重复周期终点）；`plane = "yz"`：固定 x = `at`，沿 y 取（y_j = j·Ly/nt）。
- `nt·len(z)` ≤ `kFieldsJsonMaxSamples`（4000000，与一维相同）；`at` 必须有限；`nt ≥ 1`；`z` 非空且有限，保留输入顺序与重复值。
- 其余（`output_dir` 必须是本次新建的子目录、`request_id` 1–128 字节、临时目录写完再整体重命名发布、已存在则拒绝）与一维完全一致。
- **GUI 发出的 `z` 总是比用户要的多两行（隐藏探针行，见第 4、6 节）**：不论用户的 z 范围是什么，在最前面加 z = −h（叠层之前），在最后面加 z = 叠层底面 + h（叠层之后）；
  h = max(0.02 × 波长, 1e-8 µm)。协议本身不知道探针：它们只是 `z` 里的普通行，`shape`、`z.npy`、`fields.npy` 等都按发出的行数。
  每行是独立计算的，所以用户各行的场、E2 与没有探针时逐位相同（弱场掩膜重新取峰值的那些点上相位只在 1e-14 rad 以内相等，这是唯一的例外，见 6.4）。**程序发布的原始目录 `cpp_fields-<令牌>/` 保留这两行**；
  GUI 校验完就把它们从每个按行的数组里去掉，`fields.npz`、`metrics.json` 与界面里看到的 z 轴、形状都与没有探针时一模一样。
  加上两行会超过 4000000 个采样点时不加探针，并在 `backend.note` 里写明（界面的上限 1600×1200 到不了）。

应答头部同远场（`results` 里一个结果），再加 `"protocol": "stablercwa-fields"` 与 `field_data`：

```json
"field_data": {"version": 1, "producer": "stablercwa.cpp.rcwa_fields", "request_id": "...", "directory": "...",
  "layout": "C", "shape": [nz, nt], "components": ["Ex","Ey","Ez","Hx","Hy","Hz"],
  "slice": {"plane": "xz", "at": 0.4, "period": 0.8},
  "harmonics": {"M": 4, "My": 4, "N": 81},
  "phase": {"references": ["absolute","incident"], "floor": 1e-06, "unwrap": "none", "unit": "rad", "mask_true": "undefined"},
  "normalization": {"E": 1.0, "H": 1.0, "H_convention": "Z0 H_physical"},
  "files": {"fields": {"name": "fields.npy", "dtype": "<c16", "shape": [6, nz, nt]},
            "phase": {"name": "phase.npy", "dtype": "<f8", "shape": [2, 6, nz, nt]},
            "phase_mask": {"name": "phase_mask.npy", "dtype": "|u1", "shape": [2, 6, nz, nt]},
            "t": {"name": "t.npy", "dtype": "<f8", "shape": [nt]}, "x": {"name": "x.npy", "dtype": "<f8", "shape": [nt]},
            "y": {"name": "y.npy", "dtype": "<f8", "shape": [nt]}, "z": {"name": "z.npy", "dtype": "<f8", "shape": [nz]}},
  "layers": [{"original_index": 0, "effective_index": 0, "z_top": 0.0, "z_bottom": 0.3, "finite_zero_modes": 0, "schur": false}, ...],
  "timing": {"solve": ..., "sample": ..., "phase": ..., "write": ...}}
```

- 与一维的差别只有：`shape` 的末维是 `nt`；多 `slice`、`harmonics`；文件多 `t.npy`、`y.npy`，**`x.npy` 在二维是逐点的 x 坐标**（长 `nt`：xz 时 = t，yz 时全为 `at`），`y.npy` 同理；
  没有顶层的 `y` 键；`layers` 每层都有（二维保留零厚层，`effective_index` = `original_index`），`finite_zero_modes` 恒为 0、`schur` 恒为 false（二维没有这两种机制）。
- `fields.npy`：`[6, nz, nt]` C 序复数，等于 `FieldGridSlice::components` 的 `data()` 按 iz 在前、it 在后写出；`phase.npy`、`phase_mask.npy`：`[2, 6, nz, nt]`，第 0 维 0 = 绝对相位、1 = 相对入射载波，
  与一维相同：掩膜规则 = 分量自己的峰值 × floor（1e-6），无定义处相位写 NaN、掩膜写 1，不展开（unwrap none）。C++ 相位用 `phase_slice` / 切面版 `phase_data`（二维 carrier 含 ky、按每个采样点的实际 x、y）。
- `normalization`：E = |Jones|（用 `psi_deg`/`delta_deg` 时为 1），H = nI·|Jones|，`H_convention` 同一维。
- **失败即错误**：求解失败（Rayleigh、数值失败）、上下文不可用、采样产生非有限值时与一维一样返回 `ok: false` 的错误应答且不发布目录（场协议没有"带 null 的成功应答"）。

## 4. Python / GUI 一侧

- **默认路由：二维模型默认由 Python 求解器计算，不走 C++。** 实测 C++ 的二维本征分解在每个 N 上都比 Python 慢
  （Mac、界面默认的 8 个线程、预设那样一个结构层加两层均匀膜、nx=360、nz=220，两次取较小值：N = 169 / 289 / 625，C++ 0.93 / 1.81 / 13.5 s，Python 0.69 / 1.72 / 9.24 s；单线程慢 1.2–2.1 倍），
  所以在 C++ 的本征分解并行化、并在界面默认线程数下实测不慢于 Python 之前，不把用户交给更慢的路径。开关是环境变量 `RCWA_2D_BACKEND`（`cpp_backend.backend_2d`）：
  不设或留空 = `python`（常量 `cpp_backend.DEFAULT_BACKEND_2D`，注释里写着理由与改回的条件）；`cpp` = 走下面的 C++ 路径；其它值（含还没有实现的 `auto`）按默认处理，`backend.note` 说明“无法识别”。
  默认时 `eligibility` 返回 `(False, NOTE_2D_PYTHON)`，note 是一句中性的话（“二维周期模型默认由 Python 求解器计算（环境变量 RCWA_2D_BACKEND=cpp 改用 C++）”），不当作故障报告；
  C++ 程序连查找都不做（`RCWA_FIELDS_EXE` 指向不存在的文件也不影响）。开关**只管二维**：一维的路由、请求、应答、`fields.npz` 与它无关。
  `RCWA_BACKEND` 仍只有 `python` 一个有效取值（一维、二维全部强制 Python，优先于 `RCWA_2D_BACKEND=cpp`）；`RCWA_BACKEND=cpp` 不是它认识的值，不会替 `RCWA_2D_BACKEND` 打开二维的 C++。
  启动脚本（`启动GUI.command`、`start_gui.bat`）不改环境变量，`gui.py` 把自己的环境原样交给工作进程，所以先在同一个终端设好再启动即可：macOS `RCWA_2D_BACKEND=cpp ./启动GUI.command`，
  Windows 命令行 `set RCWA_2D_BACKEND=cpp` 之后运行 `start_gui.bat`（双击启动时用 `setx` 或系统环境变量）。以下各条写的是**开关打开（`cpp`）之后**的 C++ 路径。
- `gui/cpp_backend.py` 的 `eligibility`（`RCWA_2D_BACKEND=cpp` 时）：`periodic2d` 在 isotropic、解析 Fourier（`normalize_2d` 已保证）、N ≤ 625 时可以走 C++。二维**没有**横向 PML、单轴晶体、FFT 系数（含背景覆盖）、磁性：
  `normalize_2d` 在入口就拒绝它们（界面没有二维 PML 模式，磁性介质无法输入），它们是“二维不支持”，不是“由 Python 求解”。因此 `eligibility` 里除“可以走 C++”之外的分支
  （N>625、M/My 为负、吸收入射区、单轴、FFT、背景覆盖、层类型不明）从界面不可达，只有绕过 `normalize_2d` 直接传入未规范化模型的调用者才碰得到，保留作纵深防御。
  `RCWA_BACKEND=python` 强制 Python 的行为不变。**C++ 任何一步不通（没有编译器、编译失败、超时、返回码非零、`status` 非 ok、应答校验失败）都整体退回 Python**，把原因写进 `backend.note`，与一维一致；
  Rayleigh 异常退回 Python 后由 Python 抛出它那条中文提示（用户看到的文字不变）。
- 请求由 `gui_models_2d.build_stack_2d(cfg)`（得到 `Stack2D`/`Layer2D`）转成上面的 JSON，不要另写一套几何解析；`incident.M`、`incident.My`、`angle_deg`、`phi_deg`、`psi_deg`、`delta_deg`、`wavelength` 来自 cfg。
  采样的 `z` = 用户的各行 + 前后各一行隐藏探针行（`cpp_fields_backend.sampling_plan_2d`，第 3、6 节）；`extract_fields` 按同一个计划核对请求里的 `z`，清单的 `shape`、文件描述与 `z.npy` 都按含探针的行数严格比较。
- `cpp_fields_backend.extract_fields` 的二维分支按本文第 3 节逐项校验（版本、producer、request_id、directory、layout、shape、components、slice、harmonics、phase 约定、normalization、layers、文件描述、
  NPY 头与字节数、有限性、坐标与请求一致、掩膜规则与 GUI 弱场规则一致），**校验方式沿用一维的严格程度**（失败即退回 Python，不得半信半疑地使用）。
  在此之外还做物理自洽检查（第 6 节）：形式正确、数值被改过的应答同样被拒绝。
- 界面（`gui_models_2d._edges`）接受首尾边界之差与周期相差 ≤ 1e-8·max(周期, 1) 的几何，本协议与 C++ 求解器只接受 ≤ 1e-9·周期（第 1 节，有意保持严格）。
  请求构造（`cpp_backend._snap_edges`）因此把偏差超过 5e-10·周期的最后一个边界贴齐到“首个边界+周期”，界面认可的模型不再整体退回 Python。
  移动量就是那个偏差，不超过界面的容限 1e-8·max(周期, 1) µm（周期不超过 1 µm 时至多 1e-8 µm = 0.01 pm）。**偏差不超过 5e-10·周期的几何（含边界与周期精确相符的，这是常态）一个字节也不改**；
  偏差在 5e-10·周期与 C++ 限度 1e-9·周期之间的，C++ 本来会接受，也照样被贴齐（移动量 ≤ 1e-9·周期）——阈值只取一半余量，不逐个判断 C++ 会不会接受；
  贴齐后不再严格递增的几何不改，仍由 C++ 拒绝。已知局限（不处理）：边界离原点极远而周期极小（例如首个边界 1e5 µm、周期 1e-6 µm）时，“首个边界+周期”的取整误差本身就超过 C++ 的限度，
  贴齐也过不了，这样的模型仍整体退回 Python。
- 一次调用没有得到被采用的结果时的清理（一维与二维相同）：
  · **超时、程序被杀、应答被拒绝、中途异常、POSIX 上的取消（SIGTERM）**：工作进程自己活着（SIGTERM 由处理函数接管，校验期间也在管辖之内），`cpp_fields_backend.attempt` 的 `finally`
    删掉这一次调用建的 `cpp_fields-<令牌>` 与 `cpp_fields-<令牌>.tmp-*`，别的一概不碰；
  · **Windows 上的取消**：界面服务器的 `cancel()` 用 `proc.terminate()`，在 Windows 上是 `TerminateProcess`，工作进程没有机会运行 `finally`（实测：1600×1200 时留下 391 MB 的 `cpp_fields-<令牌>/`，
    取消发生在 C++ 运行、写结果或校验的任何时刻都一样）。所以 `cancel()` 在工作进程结束之后调用 `cpp_fields_backend.discard_job_outputs(该任务自己的结果目录)`：删掉其中所有名字是
    `cpp_fields-<32 位十六进制令牌>` 或 `cpp_fields-<令牌>.tmp-*` 的项（每个任务一个新目录，里面的都是它的工作进程建的），别的名字、别的目录一概不碰，符号链接与目录联接只删链接本身；
    C++ 子进程由作业对象在工作进程消失时收拾，但这是异步的，所以删不掉的（文件还被占用）会重试，删过东西之后再扫一遍（正常的取消 0.1–0.2 s；文件被别的进程长期占住时实测 4–6 s 才返回）。
    POSIX 上 C++ 调用期间被取消时这一步什么也找不到（工作进程已经清理过）；应答被采用之后（保存六分量数据期间）被取消的任务，被采用的目录两个平台都由这一步删掉；
    已经写完 `metrics.json` 与 `fields.npz` 的任务（结果完整，工作进程正在退出）不清理；
  · **没有清理的**：已被采用的调用的 `cpp_fields-<令牌>/`（成功的结果保留，是“速度与磁盘”里的 391 MB）；被采用之后 worker 才失败（`error.json`，例如重建的场全是非有限值）时它也保留；
    取消只在任务还在运行时有效，已结束的任务不动；界面服务器被硬杀（不经 `shutdown`）时没有人替正在运行的任务取消。
- GUI 的二维结果对象需要的东西与 Python 路径一致（见 `gui/gui_worker.py` 的 `two_d` 分支与 `gui/gui_fields.py`）：六个分量 `(nz, nt)`、`E2`、`x`、`y`、`t`、`z`、`slice_plane`、`slice_at`、
  远场的 `mn`（每行 m, n）、`DEr`、`DEt`、`Rs`、`Rp`、`Ts`、`Tp`、`kzI`、`kzII`、`k0`、`Kx`、`Ky`；`metrics.json` 的二维 `orders` 行含 m、n、R、T、R_s、R_p、T_s、T_p；
  `backend` 段与一维同构（`fields`/`phase` = "cpp"，`python_solve_seconds` = 0 等）。

## 5. 测试与验收

- C++：`tests/test_rcwa2d_json.cpp`——请求解析与全部错误路径；远场应答读回与直接 `solve2d` 逐位相同；对冻结参照（`tests/rcwa2d_reference.inc`）的成功算例经 JSON 入口得到的数 ≤ 冻结容差；
  场协议写出的 NPY 读回与直接 `field_slice` / `phase_slice` 逐位相同，manifest 与文件描述逐项符合本文；Rayleigh 与数值失败的两种协议行为；`output_dir` 规则；ASan/UBSan。
  现有的一维协议测试（`tests/test_rcwa_json.cpp` 等）输出一个字不变；用改动之前的头文件另编译一份，对一批一维请求逐字节比较应答。
- Python：`gui/validation/test_cpp_2d_backend.py`——eligibility、`build_request`、`extract_fields` 的二维校验（含逐项篡改应答应被拒绝）、`gui_worker.compute` 的端到端（有 C++ 程序时对冻结参照/Python 路径比较六个分量与相位；没有时 `skipUnless`）、
  回退路径（强制 Python、程序缺失、`status` 非 ok、应答被篡改）；现有 `test_cpp_backend.py`、`test_cpp_fields_backend.py`、`test_gui_2d.py`、`test_gui_models_2d.py` 等必须继续通过。

## 6. 应答的物理自洽检查与误报研究

只核对键、形状、有限性与“效率对 |E|²”，查不出形式正确、数值被改过的应答（例如传播级 kz 取负实数、某一级的振幅被改、入射偏振读错、伪造的计时与线程数）。
所以 GUI 另做下面这些物理自洽检查。
当前 C++ 程序是对的（远场与 Python 差 7e-13、近场 5e-12），这些检查是对以后回归的加固：任何一项不符，整体退回 Python，`backend.note` 写出原因。

### 6.1 检查一览

| 检查 | 位置 | 容限 | 诚实应答的最大残差（实测） | 篡改的最小残差 |
|---|---|---|---|---|
| Kx、Ky 与请求（级次、入射方向）解析值 | `cpp_backend._extract_2d` | 1e-11·max(1, \|K\|) | ≤ 3e-16 | — |
| kz² = n²−Kx²−Ky²；Im kz ≥ 0；**Re kz ≥ 0**；kz 与独立重算的 `_kz_over_k0` 逐级比较（含正负号）；**prop 标志取自解析 kz** | `_extract_2d` | kz²：1e-9（未变）；kz 比较：**1e-12**；标志：解析 kz 的实部在 1e-12 附近或 \|n²−Kx²−Ky²\| ≤ 1e-12·尺度（kz < 1e-6）时只要求与应答自己的 kz 一致 | kz² 7.4e-16；kz 比较 7.4e-16 | kz 取负实数：任意大；kz×(1+1e-10)（效率、Ez、Rp 同步改过）：2e-10 |
| 各级效率 = Re kz·(abs(Ex)²+abs(Ey)²+abs(Ez)²)/(nI cosθ)，相对 max(1, 期望值) | `_extract_2d` | 1e-12 | 3.3e-16 | 1e-10（DE×(1+1e-10)） |
| **逐级**：Rs、Rp、Ts、Tp 与远场 Ez 由同一应答的 Ex、Ey、Kx、Ky、kz 按 rcwa2d.py:481–506 重算，相对**该级自己的**幅度（max(\|Ex\|,\|Ey\|,\|Ez\|,\|s\|,\|p\|)，不小于 1e-200）；NaN 记作 inf | `cpp_backend.far_field_residual_2d` | 1e-9 | 4.9e-16（按逐级尺度） | 1.0（Ts/Tp、Rs/Rp 对调、Rs 共轭）；Ez 差 1%：2.4e-3 |
| 应答头 `ok` 是布尔 `true`（不是 1、"yes"） | `_extract_2d` 与 `_extract_fields_2d` **各自**检查 | 精确 | — | — |
| fields.npy 在叠层之外的**每一行**（z < −1e-9、z > 叠层底面 + 1e-9，含隐藏探针行）= 只由远场数组重建的场，按 E、H 两类、每区域分别取峰值（不小于 1e-200） | `cpp_fields_backend.near_field_check_2d` | 1e-9 **+ 取整允许量**（见 6.3） | 差 ÷ (容限 + 允许量) ≤ **0.011**，即余量 ≥ **89.9 倍**（540 个随机配置，Windows；其中 λ ≥ 0.05 µm 的 464 个 ≥ 114 倍）；差 ÷ 峰值本身：普通配置 ≲ 1e-13，\|z\| ~ 1e3–1e5 µm 的配置到 8.8e-7，λ = 1 fm 的极端配置到 3.9e-6（这些配置里取整允许量占主导，比值仍小） | 6.8e-7（Ex×(1+1e-6)）；Ez 变号、H 差 Z0、陈旧文件等 ≥ 0.05 |
| phase.npy 与 fields.npy：abs(F)·e^{iφ₀}=F，abs(F)·e^{iφ₁}=F·conj(入射载波)，未掩膜处，相对该分量的峰值（不小于 1e-200） | `phase_residual_2d` | 1e-9 **+ 幅角允许量**（6.3） | 差 ÷ (容限 + 允许量) ≤ **0.0098**，即余量 ≥ **102 倍**（540 个随机配置；其中 λ ≥ 0.05 µm 的 464 个 ≥ 368 倍） | 1.8（陈旧相位） |
| 应答里的元数据：threads = 请求的线程数；timing.total 与 manifest 各段之和 ≤ 1.05×墙上时间 + 1 s（operators、eig、linear、product 是各线程用时之和，按线程数放宽）；n_eig 在结构层数与总层数之间；n_eig = 0 时 max_eig_residual 恰为 0；max_solve_backward_error = 0；status 为 ok 时 message = "" | `cpp_fields_backend._check_reply_metadata_2d` | 精确 / 见左 | 全部诚实应答通过（含 threads = 3、8、256 与重载的机器） | 伪造的 timing（1e6 s）、threads = 256、n_eig = 0、message 非空：全部拒绝 |

误报研究的“余量”指 容限（含允许量）÷ 诚实应答的最大残差；检出下限指不会被看出的最小改动。远场、效率、kz：诚实余量分别是 2.0e6、3.0e3、1.35e3（kz 比较）与 1.35e6（kz²）；
近场、相位的诚实余量是 ≥ 89.9、≥ 102（含取整允许量，见下）。数字来自 6.2 的 540 个随机配置。

### 6.2 误报研究（对诚实应答）

- 程序：Windows 上用 MinGW g++ 13.1 编出的 rcwa_fields.exe / rcwa_json.exe。
- **540 个随机配置**（6 个种子 × 90）全部被接受，没有一个被拒绝。覆盖：z 范围起止恰在 0 或叠层底面、只在一侧、叠层内、远处与极远处、离界面 ±5e-10 µm 的带里；
  Rayleigh 边缘（在斜入射下按精确截止解出 λ，偏离 1e-6…3e-16）；θ = ±89.9°；λ 从 1e-6 µm 到 300 µm（76 个 λ < 0.05 µm，95 个 λ > 20 µm）；薄层与不透明层；N 从 1 到 625（33 个 625）；线程数 1–4。
  最大诚实残差：远场 4.9e-16、效率 3.3e-16、kz² 7.4e-16、kz 比较 7.4e-16；近场判据比值最大 0.0111（余量 90 倍，λ = 4e-6 µm、θ = 41° 的默认 z 范围），相位判据比值最大 0.0098（102 倍）；
  λ ≥ 0.05 µm 的 464 个：近场余量 ≥ 114 倍、相位 ≥ 368 倍。没有一个配置因取整允许量过大跳过了行或相位检查。`extract_fields` 的用时：中位数 0.018 s，最大 0.157 s。
- **另一组 321 个协议级配置**（Lx≠Ly 与 Lx=Ly、xz/yz、s/p/jones、实/有耗/强吸收/金属折射率、0 到 6 层、掠入射、Rayleigh 边缘两侧、极端幅角 10 个）全部被接受；
  近场余量 ≥ 118 倍，相位余量 ≥ 131 倍。极端幅角配置里近场的差 ÷ 峰值到 3.9e-6、相位到 9.5e-7，由取整允许量吸收；其中 2 个的相位检查因允许量超过 1e-3 而被明确跳过。
- 端到端（经 `gui_worker.compute`）由 `test_cpp_2d_backend` 的 `WorkerEndToEnd*`、`FrozenEndToEnd*` 用例承担。
- 误报最多的一项是 6.3 的取整允许量：不加它，掠入射 θ=89.9° 的诚实残差到 5.4e-11，极端幅角（λ=1 fm 或 |z|=1e6 µm）到 5e-7、1.5e-9，都在 1e-9 附近或以上。

### 6.3 取整允许量：来源、数字与诚实的边界

掠入射时诚实残差随 1/cos²θ 增长。原因（用 50 位精度重算一个位置验证过）是**条件数**，不是算错：入射阶与反射阶在 hx、hy 里几乎相消（真值 ~cosθ），而 V = Q/q 的元素含 1/q ~ 1/cosθ，
两个程序各自舍入的误差都是“各项绝对值之和”乘机器精度（本例：C++ 的误差 9.1e-14，Python 重建的误差 4.9e-14，真值 5e-3）。固定容限不能反映这一点，所以：

- **近场**：允许量 = 96 倍机器精度 × S，S = Σ_谐波 [重建链路上每一项的绝对值之和（含反射阶与入射阶相消的 v、乘 V、合成 Ez/Hz），每项再乘指数因子的取整权重 1 + \|k0 q z\|]
  + 傅里叶相位幅角 × 该谐波的值（`_Region2D.build`）。实测 最大差 ÷ (机器精度 × S) 在各配置里是 0.02–0.9（容限起主导的配置里偶尔到 1–6，那里允许量可以忽略）。
  判据 最大差 ≤ 1e-9 × 峰值 + 允许量，逐（区域，分量，行）。允许量超过该类峰值的 1e-3 的（分量，行）没有鉴别力：**明确跳过**并在 `backend.note` 写“近场自洽检查跳过了 n 行”。
- **相位**：允许量 = 128 倍机器精度 × 入射载波幅角的上界 k0(\|kz₀\|·\|z\|max + \|Kx₀\|·\|x\|max + \|Ky₀\|·\|y\|max)；超过 1e-3 时整项**明确跳过**并写进 `backend.note`。
- 相位允许量的大小：幅角 1e3 时是 2.8e-11（容限的 2.8%），幅角 3.5e4 时与容限相等，1e7 时 2.8e-7，3.5e10 时到 1e-3 而跳过。
  λ = 1e-9 µm、|z| = 1e3 µm（幅角 6e12）时相位检查被**跳过并说明**；λ = 1e-9 µm（1 fm）、|z| ≲ 1 µm 的诚实应答照样被检查（幅角 ~5e9，允许量 7e-5）。
- **检出下限**（最容易藏改动的位置上的 (容限×峰值 + 允许量) ÷ 峰值；相对该族峰值小于它的改动在那里看不出）。540 个随机配置：中位数 1.0e-9（就是容限）；
  λ ≥ 0.05 µm 的 464 个：p90 2.9e-9，p99 7.8e-8，最大 **1.35e-6**（z 范围极远的 `huge` 类，λ = 0.057 µm、N = 91）；其中 z 范围不远的 356 个：最大 **3.4e-7**（λ = 71 µm、N = 625、θ = −85°，z 在界面附近的窄带里）；
  λ < 0.05 µm 的 76 个（亚波长到亚纳米，|z| 相对波长极大）：p90 1.8e-6，最大 **7.5e-4**。对最小的篡改 Ex×(1+1e-6)（残差 6.8e-7）的余量：λ ≥ 0.05 µm 的 446/464 个 ≥ 50 倍，其余 18 个里 10 个是 z 范围极远（`huge`、`far_below`，|z| 达 1e3–1e5 µm），
  8 个是掠入射（θ = ±89.9°）、N = 625 或 λ = 71–117 µm 的配置（检出下限 1.4e-8–3.4e-7）；最差的 1.35e-6 比这个篡改（6.8e-7）还大——在那个配置里 Ex×(1+1e-6) 看不出。
  **诚实余量 ≥ 100 与篡改余量 ≥ 50 在掠入射 θ=89.9°、|z| ~ 1e4 µm 与极端幅角下同时满足不了**（诚实噪声 5e-11…2e-10 的 100 倍加上 50 倍的篡改下限已经超过 1e-6）；本检查在这些配置里保证诚实余量，篡改下限随之上移，上面给出了数字。
- **逐分量的检出下限**（用二分法找“把某个分量在这个区域的全部被检查的行乘以 1+δ 刚好被拒绝”的最小 δ；143 个随机配置、N ≤ 300；某分量在那些行上恒为 0 的不计）：
  Ex 中位数 1.3e-9，p90 2.3e-8，最大 0.11；Ey 1.3e-9、1.6e-8、3.3e-6；**Ez 3.7e-9、7.3e-8、0.15**；Hx 1.2e-9、1.5e-8、6.5e-3；Hy 1.1e-9、2.8e-8、0.18；Hz 4.0e-9、6.9e-8、0.13。
  只改一行（该区域里这个分量最弱的那一行）的下限更高：Ez 中位数 5.0e-9、p90 4.3e-7、最大 0.49。
  **这是一个警告**：判据相对的是 E 或 H **整族**的峰值，所以较弱的分量（法向入射时的 Ez、Hx 等，在大多数配置里比主分量小几个数量级）只受较弱的保护——它们被改动到整族峰值的相应比例以下是看不出的。
  “检出下限 ≈ 容限 1e-9”只对强分量成立，对弱分量应读上面的分布；最坏的情形（法向入射、出射区的 Hy、Ez，下限 0.1–0.2）等于没有保护。

### 6.4 检查范围：哪些行、哪些改动

- **叠层之外的每一行都检查**（逐批 64 行，额外内存与行数无关）。离界面不足 1e-9 µm 的行不检查（界面两侧的 Ez、Hz 本来不连续）。
- **叠层之内的行没有独立的参照，不检查**（z 在 0 与叠层底面之间；“Ez 变号”“整体乘 1.05”“置零”这类只改叠层之内的篡改看不出）。
- **隐藏探针行（总是加两行）**：不论用户的 z 范围是什么，请求里在最前面加 z = −h（叠层之前）、在最后面加 z = 叠层底面 + h（叠层之后）；h = max(0.02·λ, 1e-8 µm)
  （0.02 个波长：低阶倏逝级 |kz| ≲ 30 在探针行上仍有 ≥ 2% 的幅度，改动看得见；下限 1e-8 µm 是为了让探针离开界面处 1e-9 µm 的“不检查带”一个数量级以上，波长小于 5e-7 µm 时 0.02 个波长小于它）。
  用户的叠层之外的行可以全被“取整允许量超过 1e-3”跳过（例如 λ = 1e-9 µm、z ∈ [−1e3, 1e3] µm）；探针行离界面很近，它们的取整允许量即使在那种参数下也很小，
  所以入射偏振（例如 psi 误读成 90−psi）总能在探针行上核对。
  每行独立计算，用户各行的场、E2 与没有探针时**逐位相同**（`test_probe_rows_leave_the_users_data_unchanged` 对真实程序与替身各比较 fields.npz 的全部键，覆盖叠层内/上/下、两侧都有行、很多行、没有结构层等 7 种 z 范围）。
  校验通过后探针行从每个按行的数组（场、相位、掩膜、z）里去掉。
  **唯一的例外：弱场掩膜**（|F| ≤ 1e-6×该分量峰值）要按用户自己的行重新取峰值——探针行的幅度可以比用户的行大（用户的行深在强吸收膜里时，
  场比紧挨着膜的探针行小几个数量级），C++ 按含探针行的峰值写的掩膜就在用户的网格上多掩了一些点（几百到几千个），这些点由 Python 补上相位（`_rederive_weak_field_mask_2d`，公式与 `rcwa_phase.hpp` 相同，但不是同一份代码，末位可以不同）。
  这些点上的相位与不加探针的 C++ 运行只在 1e-14 rad 以内相等（8 个模型，其中 6 个触发了重新取峰值、2 个——电介质、探针行不是峰值——没有）：
  macOS 第一个参考（arg F）逐位相同，第二个参考（arg F·conj 载波）在 197–6364 个点上不同、最大差 2.2e-15；
  Windows（MinGW libm 与 numpy 的 atan2 末位不同）第一个参考在 3–13 个点上不同、第二个参考在 478–1890 个点上不同，两者都 ≤ 4.4e-16。
  六个场、E2、掩膜与补上的点以外的全部相位在两个平台上都逐位相同；没有这样的点时一个数也不动。测试 `test_the_rederive_branch_changes_the_phase_only_at_the_rederived_points_and_by_less_than_1e_14` 断言的就是这个保证。
  **程序发布的原始目录 `cpp_fields-<令牌>/` 保留探针行**（`fields.npy` 等多两行，`manifest["shape"]` 也是含探针的行数）。
  nz×nt 加上两行会超过协议的 4000000 个采样点时不加探针，并在 `backend.note` 写明（“没有加隐藏探针行”），此外若叠层之外一侧没有行能参加检查，note 另写“入射偏振与入射相位没有与请求核对”或“透射振幅没有与请求核对”
  （界面的上限 1600×1200 = 1.92e6 远低于它）。只用于二维。
- 因此，入射偏振与相位（psi 误读成 90−psi、delta 符号、整体相位）与某一级反射/透射振幅的篡改，不论用户的 z 范围在哪里，都被叠层之外的行（至少是探针行）钉住；
  只有探针加不上的情形（nz×nt 超限，库用法）才会漏，那时 `backend.note` 明说哪一侧没有核对。
- **离界面不足 1e-9 µm 的行（z = 0 与 z = 叠层底面两侧各 1e-9 µm 的两个窄带）不检查**，也不在“叠层之内”的参照里：界面两侧的 Ez、Hz 本来不连续，哪一侧由取整决定。在这样的行上把 Ez 变号、H×1.1 是看不出的（6.5 第 2 条）。
  曾评估过在这些行上检查切向分量（Ex、Ey、Hx、Hy）的连续性：诚实应答里恰在界面上的行（40 个随机配置、52 个窄带区域）有 50 个的切向分量与由远场重建的值相差 ≤ 峰值的 3.8e-11，
  但有一个诚实配置（λ = 0.41 µm、N = 33、五层、z 恰在 0 与叠层底面，两个窄带）到 9.1e-9 与 7.3e-9，比近场的容限 1e-9 大 7–9 倍；冻结参照自己的“切向连续性”量也是 4.3e-9 量级（`test_field_contract_2d`）。
  这个残差来自线性求解的条件数，现有的取整允许量模型不覆盖它；窄带里不恰在界面上的行还有 k0·|q|·离界面距离 的额外差。所以这项检查没有实现。

- **计算开销**（N=625、nt=1600、nz=1200，Windows 5950X、一个线程，该机器同时有别的负载，计时只是参考）：默认 z 范围（−0.2 … 叠层 + 0.2 µm，632 行在叠层之外）：近场检查 0.71 s，相位检查 0.55 s，整个 `extract_fields`（读文件、格式与掩膜核对、两项检查）1.7 s；
  z = −3 … 3 µm（1128 行在叠层之外，是近场检查最坏的情形）：近场检查 1.12 s，整个 `extract_fields` 2.14 s。对比：C++ 求解 29–30 s（一个线程）。
  额外内存：近场检查 +61 MB、相位检查 +60 MB，与行数无关（逐批 64 行）；峰值驻留内存 797 MB，其中 fields.npy、phase.npy、phase_mask.npy 三个数组读进来就占 508 MB。

### 6.5 看不出的情形（完整清单，与代码一致）

1. 一套自洽却物理不同的应答：`thickness_swap`（求解器内部对调两层厚度而应答照抄请求）、层折射率读错（`index_misread`）、镜像结构（`mirror_x`）、
   某级反射振幅整体旋转/缩放并同步重算叠层之外的场（`R_phase_rot_resynth`、`R_scale_resynth`）：远场、叠层之外的场、相位、元数据全都自洽，不做一次独立求解无从发现，
   只有 `RCWA_VERIFY_PYTHON=1` 能发现（测试里照实断言“被接受”，并断言显式核对给出 > 1e-3 的差与“相差超过”的提示）。入射点（波长、角度）、Kx、Ky、kz、入射偏振与相位不在此列：它们由请求决定，被逐位或按解析式核对。
2. 叠层之内（0 ≤ z ≤ 叠层底面）各行的任何改动，**以及 z = 0 与 z = 叠层底面两侧各 1e-9 µm 窄带里的行**（6.4）：`interface_rows_only` 这类改动（Ez 变号、H×1.1）看不出。
3. 取整允许量以内的改动（6.3 的检出下限：一般小于峰值的 1e-9；掠入射与极端 |z| 下更大）；被明确跳过的行与相位检查。
4. nz×nt 接近 4000000（加两行探针会超限，界面到不了，库用法才会碰到）时没有探针行：只剩用户自己的叠层之外的行；psi 误读这类改动在用户 z 范围里没有叠层之前的行时又会漏掉——`backend.note` 明说“没有加隐藏探针行”与哪一侧“没有与请求核对”。
5. 应答里只报告的诊断量：`n_dense_solve`（随层结构变化，没有可靠的界）、有结构层时的 `max_eig_residual`（诚实应答里可以是任意非负数）、`eig_info` 之外的 `status` 文字；**计时只检查上界**
   （不超过 1.05×墙上时间 + 1 s），伪造得比实际小（例如全为 0，`timing_all_zero`）看不出；`results[0]` 里多出的未知键被忽略（`extra_result_key`，无任何可见影响）。
6. 极端参数（λ = 1 fm 且 |z| ~ 1e3 µm）：用户的行因取整允许量过大被跳过、相位检查整项跳过（6.3，note 写明）。入射偏振与透射振幅仍由探针行钉住；用户的行本身在那里没有检查。
7. 检查的判据相对的是 E 或 H **整族**的峰值（6.3）：族里较弱的分量只受较弱的保护，改动了才大到整族峰值的相应比例才会被看出。

### 6.6 篡改检验

`test_cpp_2d_backend.py` 内置一组形式正确、数值被改过的应答。已知会被接受的篡改全部在 6.5 列出的范围内：
- 自洽却物理不同的整套应答；
- 只改叠层之内或界面窄带的行；
- 容限以内的改动；
- 只报告的诊断量与计时下限。

其余全部被拒绝，并退回 Python。例如：
- 传播级 kz 取负实数（C++ 里一个字符的回归）：3 个模型全部拒绝，退回 Python，结果与诚实运行相差 ≤ 5e-14；
- 只在没有入射区行的 z 范围里改动入射偏振或相位：由探针行拒绝；
- 伪造的计时、线程数、本征分解次数：全部拒绝。

另对 gui/ 的副本逐个做一行代码变异，共 80 个，检查测试能否发现：78 个被测试发现；存活的 2 个，一个是无操作的对照，一个是等价变异。
