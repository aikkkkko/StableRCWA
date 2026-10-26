"""One C++ solve supplies eligible GUI efficiencies, six complex fields and phase.

rcwa_json's old far-field API remains in cpp_backend. The GUI uses rcwa_fields;
RCWA_FIELDS_EXE overrides that executable. RCWA_JSON_EXE alone routes the GUI
to Python with a compatibility note. RCWA_VERIFY_PYTHON=1 explicitly enables
an optional second solve for comparison; ordinary C++ runs never solve in Python.

Periodic 2D models (periodic2d, isotropic analytic Fourier, N <= 625) default to the Python solver (cpp_backend.DEFAULT_BACKEND_2D:
the C++ route is only faster at larger N); when the model asks for it (backend_2d = "cpp", the GUI's 2-D solver choice) or
RCWA_2D_BACKEND=cpp sets that default, they use the same executable with
PROTOCOL_2D.md: the request carries sampling {z, plane, nt, at}; the reply manifest describes
[6,nz,nt] fields, [2,6,nz,nt] phase/mask and t/x/y/z coordinate files. Every manifest item and
every file header is validated as strictly as in 1D; any mismatch falls back to Python as a whole.
Beyond the format, a 2D reply is also checked for physical self-consistency (far_field_residual_2d in cpp_backend,
near_field_check_2d and phase_residual_2d below): plausible-but-wrong numbers are refused as well. Every row outside the
stack is checked against the far-field arrays. Two hidden probe rows are ALWAYS added, one in front of the stack and one behind it
(sampling_plan_2d), whatever the user's z range is, so the incident polarisation/phase and the reflected/transmitted
amplitudes are always tied to the request, also when the user's rows are all inside the stack or all skipped as too noisy to check;
the probe rows are removed again before anything reaches the GUI.

A call that does not end in an accepted C++ result removes everything it created in the result folder
(cpp_fields-<token> and cpp_fields-<token>.tmp-*; see discard_call_outputs): timeout, cancel and rejected replies leave nothing behind.
"""
from __future__ import annotations

import cmath
import json
import math
import os
import re
import shutil
import stat
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import cpp_backend as common

COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
SOURCE = common.STABLE_ROOT / "examples" / "rcwa_fields.cpp"
EXE_NAME = "rcwa_fields.exe" if os.name == "nt" else "rcwa_fields"
PRODUCER = "stablercwa.cpp.rcwa_fields"
FIELDS_VERSION = 1
PHASE_VERSION = 1
BackendError = common.BackendError


def verify_requested():
    return os.environ.get("RCWA_VERIFY_PYTHON", "").strip() == "1"


def ensure_executable(progress=None, retry_failed=False):
    override = os.environ.get("RCWA_FIELDS_EXE", "").strip()
    if override:
        exe = Path(os.path.abspath(os.path.expanduser(override)))
        return (exe, "") if exe.is_file() else (None, f"找不到命令行程序：RCWA_FIELDS_EXE 指向的 {override} 不存在")
    if os.environ.get("RCWA_JSON_EXE", "").strip():
        return None, "RCWA_JSON_EXE 使用旧效率协议；GUI 完整计算已回退 Python，完整 C++ 覆写请使用 RCWA_FIELDS_EXE"
    exe = common.BIN / EXE_NAME
    stamp = common._source_stamp(SOURCE)
    if stamp is None:
        return (exe, "") if exe.is_file() else (None, "找不到 rcwa_fields 程序或 examples/rcwa_fields.cpp 源码")
    if common._up_to_date(exe, stamp, SOURCE):
        return exe, ""
    try:
        return common._build(exe, stamp, progress, retry_failed, source=SOURCE, program="rcwa_fields")
    except (BackendError, OSError) as error:
        return None, f"编译 rcwa_fields 时出错：{error}"


def warmup():
    if common.forced_python():
        message = "C++ 后端：RCWA_BACKEND=python，全部计算使用 Python 求解器。"
    else:
        exe, note = ensure_executable(retry_failed=True)
        if common.backend_2d()[0] == "cpp":
            covered = "周期平面 TE/TM、圆锥入射与二维周期的效率、六复场和相位使用一次 C++ 求解。"
        else:
            covered = "周期平面 TE/TM 与圆锥入射的效率、六复场和相位使用一次 C++ 求解（二维周期默认用 Python，界面的“二维求解”选 C++ 或设环境变量 RCWA_2D_BACKEND=cpp 改用 C++）。"
        message = f"C++ 后端：{exe}；{covered}" if exe else f"C++ 后端不可用，使用 Python：{note}"
        if exe and note:
            message += f"（{note}）"
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    sys.stdout.write(message.encode(encoding, "replace").decode(encoding) + "\n")
    sys.stdout.flush()


def build_request(cfg, directory, request_id, threads=1):
    solve = common.build_request(cfg, threads)
    solve["options"]["diagnostics"] = True
    sampling = cfg["sampling"]
    if cfg["mode"] == "periodic2d":
        # PROTOCOL_2D.md 第 3 节：二维的 sampling 只认 z、plane、nt、at；nt 沿切面取点，nx、y 是一维的键。
        # z 是用户要的各行，前后各加 sampling_plan_2d 给的一行隐藏探针行（总是两行，加不上才不加），探针行在 extract_fields 里去掉。
        shape = {"z": sent_z_2d(cfg), "plane": sampling["slice_plane"], "nt": int(sampling["nx"]), "at": float(sampling["slice_at"])}
    else:
        z = np.linspace(sampling["z_min"], sampling["z_max"], sampling["nz"]).tolist()
        shape = {"z": z, "nx": int(sampling["nx"]), "y": 0.}
    return {"solve": solve, "sampling": shape,
            "output_dir": str(Path(directory).resolve()), "request_id": request_id}


def _load_npy(directory, manifest, key, dtype, shape):
    """Validate the small header and exact byte count before NumPy allocates an array."""
    descriptor = manifest["files"][key]
    expected_name = {"fields": "fields.npy", "phase": "phase.npy", "phase_mask": "phase_mask.npy",
                     "x": "x.npy", "z": "z.npy", "t": "t.npy", "y": "y.npy"}[key]
    if descriptor != {"name": expected_name, "dtype": dtype, "shape": list(shape)}:
        raise BackendError(f"C++ {key} 文件描述、dtype 或 shape 不符")
    path = directory / expected_name
    if not path.is_file() or path.is_symlink():
        raise BackendError(f"C++ {key} NPY 文件不存在或不是普通结果文件")
    try:
        with path.open("rb") as stream:
            if np.lib.format.read_magic(stream) != (1, 0):
                raise BackendError(f"C++ {key} NPY 版本不符")
            actual_shape, fortran, actual_dtype = np.lib.format.read_array_header_1_0(stream)
            expected_dtype = np.dtype(dtype)
            if actual_shape != tuple(shape) or fortran or actual_dtype.str != dtype or actual_dtype.hasobject:
                raise BackendError(f"C++ {key} NPY 的 dtype、endian、layout 或 shape 不符")
            expected_bytes = math.prod(shape) * expected_dtype.itemsize
            if path.stat().st_size != stream.tell() + expected_bytes:
                raise BackendError(f"C++ {key} NPY 文件不完整或含多余数据")
            stream.seek(0)
            array = np.load(stream, allow_pickle=False)
    except BackendError:
        raise
    except (OSError, ValueError, EOFError) as error:
        raise BackendError(f"C++ {key} NPY 读取失败：{error}") from None
    except Exception as error:
        # 文件头被改坏时，NumPy 解析它用的是 ast.literal_eval 与 tokenize：除了上面三种，还会抛 tokenize.TokenError、SyntaxError、
        # MemoryError、RecursionError 等（模糊测试里 1500 个损坏的文件头有 10 个抛 TokenError）。
        # 不论哪一种，都是“这个结果文件不能用”，不是界面的故障：一律成为带文件名与异常类型的 BackendError，整体退回 Python。
        raise BackendError(f"C++ {key} NPY 读取失败（{type(error).__name__}）：{error}") from None
    if not array.flags.c_contiguous:
        raise BackendError(f"C++ {key} NPY 不是 C layout")
    return array


def _same(value, expected):
    """JSON value equals the expected one item by item. bool / int / float / str do not impersonate each other
    (an integral float may be written as 0 or 0.0); dicts must carry exactly the expected keys."""
    if isinstance(expected, bool):
        return type(value) is bool and value == expected
    if isinstance(expected, int):
        return type(value) is int and value == expected
    if isinstance(expected, float):
        return type(value) in (int, float) and value == expected
    if isinstance(expected, str):
        return type(value) is str and value == expected
    if isinstance(expected, list):
        return type(value) is list and len(value) == len(expected) and all(_same(a, b) for a, b in zip(value, expected))
    if isinstance(expected, dict):
        return type(value) is dict and set(value) == set(expected) and all(_same(value[k], expected[k]) for k in expected)
    return False


FILES_2D = ("fields", "phase", "phase_mask", "t", "x", "y", "z")
MANIFEST_KEYS_2D = {"version", "producer", "request_id", "directory", "layout", "shape", "components", "slice",
                    "harmonics", "phase", "normalization", "files", "layers", "timing"}
LAYER_KEYS_2D = {"original_index", "effective_index", "z_top", "z_bottom", "finite_zero_modes", "schur"}
SOLVE_TIMING_2D = {"total", "operators", "eig", "linear", "product"}


# --------------------------------------------------------------------------- physical self-consistency of a 2D reply
# 容限的依据见 PROTOCOL_2D.md 第 6 节（误报研究）：普通配置的诚实残差比容限小几个数量级，篡改应答里最小的残差比容限大几个数量级；
# 极端配置（|z| ~ 1e3–1e5 µm、λ 到 1 fm）的诚实残差随条件数、幅角增长，由下面的取整允许量吸收。
NEAR_FIELD_TOLERANCE_2D = 1e-9    # 叠层之外的行：fields.npy 与由远场数组重建的场，最大差 ÷ 同一区域、同一类（E 或 H）的峰值（不小于 AMPLITUDE_FLOOR_2D）；诚实应答的差÷峰值：中位数 7e-15，p99 2e-8，极端配置到 3.9e-6（有允许量）；篡改 ≥ 6.8e-7
PHASE_TOLERANCE_2D = 1e-9         # phase.npy 与 fields.npy 的一致性：|F|·e^{iφ} 与 F（或 F·conj(入射载波)）的最大差 ÷ 该分量的峰值；诚实应答的差÷峰值：中位数 1e-15，p99 2e-9，极端配置到 7e-7（有允许量），篡改 ≥ 1.8
REGION_MARGIN_2D = 1e-9           # z 离界面（0 与叠层底面）不足这个距离（µm）的行不检查：边界上两侧的 Ez、Hz 并不连续，取哪一侧由取整决定
# 取整允许量 = 若干倍机器精度 × 噪声尺度 S（近场：_Region2D.build 的逐项绝对值之和，96 倍）或 × 载波幅角的上界（相位：128 倍）。
# 取法（PROTOCOL_2D.md 第 6 节）：诚实应答的实测误差是 S（或幅角）乘机器精度的几分之一到一两倍。实测的诚实余量（容限含允许量 ÷ 诚实残差）最小：
# 近场 90 倍、相位 102 倍（540 个随机配置）。
# “允许量不大于 Ex×(1+1e-6) 这样最小的篡改的 1/50”：λ ≥ 0.05 µm 的 464 个配置里 446 个成立，其余的（|z| 极远、掠入射、N=625、λ ~ 70–120 µm）不成立，
# 最差的检出下限是 1.35e-6（λ < 0.05 µm 时可到 7.5e-4）；而且判据相对的是 E 或 H 整族的峰值，较弱的分量只受较弱的保护（逐分量的分布见文档 6.3）。
NEAR_ROUNDING_FACTOR_2D = 96 * 2.220446049250313e-16
PHASE_ROUNDING_FACTOR_2D = 128 * 2.220446049250313e-16
CHECK_LIMIT_2D = 1e-3             # 取整允许量超过峰值的这个比例时，检查已经没有鉴别力（连 0.1% 的改动都看不出）：该行（相位检查则整项）明确跳过并写进 note
NEAR_ROWS_PER_CHUNK_2D = 64       # 近场重建每批的行数：额外内存与行数无关（一批 6×nt×64 个复数，nt=1600 时约 10 MB）
PROBE_FRACTION_2D = 0.02          # 隐藏探针行离界面的距离 = 它 × 波长，见 sampling_plan_2d
PROBE_MIN_DISTANCE_2D = 1e-8      # 探针行离界面至少这么远（µm）：必须远大于 REGION_MARGIN_2D，否则探针落在“不检查”的带里
MAX_SAMPLES_2D = 4000000          # 协议的采样点上限（rcwa_fields_json.hpp 的 kFieldsJsonMaxSamples）：加上探针行会超过时不加探针


def _branch(q):
    """rcwa2d._branch：先把 Re q < 0 的整体取负，再把 Re q < 1e-10 且 Im q > 0 的取负（顺序与阈值不能改）。"""
    q = np.where(q.real < 0, -q, q)
    return np.where((q.real < 1e-10) & (q.imag > 0), -q, q)


def _incident_jones_2d(solve):
    """请求里的入射 Jones 矢量 (Es, Ep)：psi_deg/delta_deg 写法按 rcwa2d.solve2d，jones 写法原样（PROTOCOL_2D.md 第 1 节）。"""
    given = solve["input"]
    if "jones" in given:
        number = lambda v: complex(v[0], v[1]) if isinstance(v, list) else complex(v)
        return number(given["jones"][0]), number(given["jones"][1])
    psi, delta = math.radians(given["psi_deg"]), math.radians(given.get("delta_deg", 0.))
    return math.sin(psi) * cmath.exp(1j * delta), math.cos(psi)


def far_arrays_2d(far, N):
    """应答里的远场数组 → NumPy（长度、有限性已在 cpp_backend.extract 里核对过；这里再转换一次，使本模块可单独使用）。"""
    reals = {key: np.asarray(common._reals(far[key], N, key), float) for key in ("Kx", "Ky")}
    complexes = {key: np.asarray(common._complexes(far[key], N, key), complex)
                 for key in ("kzI", "kzII", "Ex_r", "Ey_r", "Ex_t", "Ey_t")}
    return {**reals, **complexes}


def stack_bottom_2d(cfg):
    """叠层底面的 z：各层厚度按顺序累加（与 rcwa2d.solve2d 里 ztop 的累加同一个次序）。"""
    z_bot = 0.
    for layer in cfg["structure"]["layers"]:
        z_bot += float(layer["thickness"])
    return z_bot


# --------------------------------------------------------------------------- hidden probe rows
def probe_distance_2d(wavelength):
    """探针行离界面的距离（µm）：0.02 个波长，但不小于 1e-8 µm。
    取 0.02 个波长的理由：足够近，低阶倏逝级（|kz| ≲ 30，即 e^{−2π·0.02·|kz|} ≳ 0.02）在探针行上仍有可观的幅度，对它们的篡改看得见；
    又足够远，离开界面的“不检查带”（REGION_MARGIN_2D = 1e-9 µm）几个数量级，也不碰到任何层。波长小于 5e-7 µm 时 0.02 个波长小于 1e-8，取下限。"""
    return max(PROBE_FRACTION_2D * float(wavelength), PROBE_MIN_DISTANCE_2D)


def sampling_plan_2d(cfg):
    """二维场采样的 z 计划：(用户要的 z, 入射区探针的 z 或 None, 出射区探针的 z 或 None)。

    近场检查（near_field_check_2d）只看叠层之外的行。入射偏振与相位（psi、delta、整体相位）与各级反射/透射振幅的篡改只有叠层之外的行看得见，
    而用户的 z 范围可以全在叠层之内、只在一侧，或者叠层之外的行全因取整允许量太大（波长相对 |z| 极小）被跳过——那样就无处可查。
    所以不论用户的 z 范围是什么，**总是**各加一行：在 z = −h 处（入射区）和叠层底面 + h 处（出射区），h = probe_distance_2d(波长)。
    探针行离界面很近，它们的取整允许量即使在用户的行全被跳过的极端参数下也很小。探针行各自独立计算（C++ 逐行求场，与别的行无关），
    所以用户的各行的场、E2 与没有探针时逐位相同（相位的第二个参考与弱场掩膜的例外见 _rederive_weak_field_mask_2d）；
    探针行只在 extract_fields 里参加检查，随后从每个按行的数组里去掉。
    加上两行会超过协议的采样点上限（4000000）时两行都不加（界面的 nx ≤ 1600、nz ≤ 1200 到不了这个上限，只有直接调用的库用法才会碰到），
    这时 extract_fields 往 notes 里写明，近场检查只看用户自己的行。"""
    sampling = cfg["sampling"]
    z = np.linspace(sampling["z_min"], sampling["z_max"], sampling["nz"])
    if (len(z) + 2) * int(sampling["nx"]) > MAX_SAMPLES_2D:
        return z, None, None
    h = probe_distance_2d(cfg["incident"]["wavelength"])
    return z, -h, stack_bottom_2d(cfg) + h


def sent_z_2d(cfg):
    """请求里发给 C++ 的 z 列表：用户的各行，前后加上 sampling_plan_2d 决定的探针行。"""
    z, below, above = sampling_plan_2d(cfg)
    return ([] if below is None else [below]) + z.tolist() + ([] if above is None else [above])


# --------------------------------------------------------------------------- near field outside the stack
class _Region2D:
    """入射区（z < 0）或出射区（z > 叠层底面）里只由远场数组决定的场（rcwa2d.field_at 的区域 I、II），以及重建时各项的取整噪声尺度。

    区域 I 是入射阶（只有被激发的 (0,0) 级，指数 e^{−k0 q z}）加反射阶 e^{+k0 q z}，区域 II 是透射阶 e^{−k0 q (z−z_bot)}；
    V 是每级一个 2×2 块 Q/q，Ez = i(Ky hx − Kx hy)/n²，Hz = Ky Ex − Kx Ey。
    入射指数只对 (0,0) 级求（不对全部级次先算 e^{+k0 q|z|} 再乘 0：那是 Python 参照里 0×inf = NaN 的隐患，合同第 2 节）。
    """

    def __init__(self, incident_side, n, kz, ex, ey, far, k0, i0, z_bot, inc_xy, x_max, y_max):
        self.incident_side, self.k0, self.i0, self.z_bot, self.inc_xy = incident_side, k0, i0, z_bot, inc_xy
        self.kx, self.ky, self.ex, self.ey = far["Kx"], far["Ky"], ex, ey
        kx, ky = self.kx, self.ky
        self.n2 = n * n
        self.q = _branch(-1j * kz)
        q, n2 = self.q, self.n2
        self.v = (kx * ky / q, (n2 - kx * kx) / q, (ky * ky - n2) / q, -ky * kx / q)       # V = Q/q 的 2×2 块
        # 傅里叶相位 e^{−i k0 (Kx x + Ky y)} 的幅角上界（逐级）：它的取整误差正比于幅角
        self.line_phase = k0 * (np.abs(kx) * x_max + np.abs(ky) * y_max)
        self.q_abs = np.abs(q)

    def build(self, zr):
        """一批行 zr 上的六个分量的谐波列（各 (N, 行数)）与对应的取整噪声尺度 S（(6, 行数)）。

        S = Σ_谐波 [ 逐项绝对值之和 + 傅里叶相位幅角 × |该谐波的值| ]。
        · 逐项绝对值之和：把“反射阶 − 入射阶 → 乘 V → 乘 ky、kx 合成 Ez、Hz”这条链上的每一项都换成它的绝对值（反射阶与入射阶相消的 v 也取绝对值之和），
          再乘该项指数因子 e^{±k0 q z} 的相对取整误差权重 1 + |k0 q z|（exp 的取整误差正比于自变量）。掠入射时 V 的元素含 1/q，入射项与反射项在 hx、hy 里
          几乎相消（真值 ~cosθ，各项 ~1/cosθ），两个程序的取整误差就是这个尺度乘机器精度——固定容限不能反映它（实测 θ=89.9° 时诚实残差 5.4e-11），这一项反映。
        · 傅里叶相位 e^{−i k0 (Kx x + Ky y)} 乘在每个谐波的最终值上（相消之后），它的取整误差是 幅角 × |谐波的值|，不是 幅角 × 相消之前的各项。
        |傅里叶相位因子| = 1，所以 S 与采样点无关，只对谐波求和。"""
        k0, q, i0 = self.k0, self.q, self.i0
        n2, kx, ky = self.n2, self.kx, self.ky
        v00, v01, v10, v11 = self.v
        inc_x, inc_y = self.inc_xy
        if self.incident_side:
            depth = np.abs(zr)
            growth = np.exp(k0 * q[:, None] * zr[None, :])                      # 反射阶 e^{+k0 q z}
            modulus = np.exp(k0 * q.real[:, None] * zr[None, :])
            Fx, Fy = self.ex[:, None] * growth, self.ey[:, None] * growth
            vx, vy = Fx.copy(), Fy.copy()
            decay0 = np.exp(-k0 * q[i0] * zr)                                    # 入射阶，只有 (0,0) 级
            Fx[i0] += inc_x * decay0
            Fy[i0] += inc_y * decay0
            vx[i0] -= inc_x * decay0
            vy[i0] -= inc_y * decay0
        else:
            depth = np.abs(zr - self.z_bot)
            decay = np.exp(-k0 * q[:, None] * (zr - self.z_bot)[None, :])        # 透射阶 e^{−k0 q (z−z_bot)}
            modulus = np.exp(-k0 * q.real[:, None] * (zr - self.z_bot)[None, :])
            Fx, Fy = self.ex[:, None] * decay, self.ey[:, None] * decay
            vx, vy = -Fx, -Fy
        hx = v00[:, None] * vx + v01[:, None] * vy
        hy = v10[:, None] * vx + v11[:, None] * vy
        harmonics = (Fx, Fy, 1j * (ky[:, None] * hx - kx[:, None] * hy) / n2, -1j * hx, -1j * hy,
                     ky[:, None] * Fx - kx[:, None] * Fy)
        weight = 1. + k0 * self.q_abs[:, None] * depth[None, :]
        ax, ay = np.abs(self.ex)[:, None] * modulus * weight, np.abs(self.ey)[:, None] * modulus * weight
        if self.incident_side:
            weight0 = 1. + k0 * self.q_abs[i0] * depth
            incident0 = np.exp(-k0 * q.real[i0] * zr) * weight0
            ax[i0] += abs(inc_x) * incident0
            ay[i0] += abs(inc_y) * incident0
        hxa = np.abs(v00)[:, None] * ax + np.abs(v01)[:, None] * ay
        hya = np.abs(v10)[:, None] * ax + np.abs(v11)[:, None] * ay
        magnitudes = (ax, ay, (np.abs(ky)[:, None] * hxa + np.abs(kx)[:, None] * hya) / abs(n2), hxa, hya,
                      np.abs(ky)[:, None] * ax + np.abs(kx)[:, None] * ay)
        phase_error = self.line_phase[:, None]
        return harmonics, np.stack([(m + phase_error * np.abs(h)).sum(axis=0) for m, h in zip(magnitudes, harmonics)])


class NearFieldCheck2D:
    """near_field_check_2d 的结果。
    ratio：各（区域，分量，行）的“差 ÷ 允许量”的最大者（允许量 = 容限×该类峰值 + 取整允许量），≤ 1 为通过，NaN/inf 记作 inf；
    residual：差 ÷ 该类峰值 的最大者；allowance：最坏位置上的取整允许量（相对峰值）；rows：参加检查的行数；
    skipped：因取整允许量超过 CHECK_LIMIT_2D 而没有检查的行数；where：最坏位置的说明；
    noise：最大差 ÷ S（诊断量：诚实应答里是机器精度的几分之一到一两倍，NEAR_ROUNDING_FACTOR_2D 是机器精度的 96 倍）；
    detect：这次检查里“最容易藏改动的位置”的判据 = (容限×峰值 + 取整允许量) ÷ 峰值——相对峰值小于它的改动，在该位置上看不出来（诊断量，写进文档的检出下限）；
    incident_rows、transmitted_rows：入射区、出射区里至少有一个分量真正参加了判据的行数（为 0 就是那一侧完全没有核对：入射区没有核对时，
    入射偏振、入射相位与反射振幅没有与请求钉在一起，出射区同理对透射振幅；调用方据此写 note）。"""

    def __init__(self, ratio=0., residual=0., allowance=0., rows=0, skipped=0, where="", noise=0., detect=0., incident_rows=0, transmitted_rows=0):
        self.ratio, self.residual, self.allowance, self.rows, self.skipped, self.where = ratio, residual, allowance, rows, skipped, where
        self.noise, self.detect = noise, detect
        self.incident_rows, self.transmitted_rows = incident_rows, transmitted_rows


def near_field_check_2d(six, x, y, z, far, request, cfg):
    """叠层之外的**全部**行（z < −1e-9 的入射区、z > 叠层底面 + 1e-9 的出射区，含隐藏探针行）：fields.npy 与只由远场数组重建的场的比较。

    重建见 _Region2D；六个分量再按 Σ_级次 F_h e^{−i k0 (Kx x + Ky y)} 合成到切面上的 nt 个点。逐批（NEAR_ROWS_PER_CHUNK_2D 行）处理，
    额外内存与行数无关。判据：每个（区域，分量，行）上 最大差 ≤ 容限 × 该区域该类（E 或 H）六分量里这些行的峰值 + 取整允许量，
    峰值不小于 AMPLITUDE_FLOOR_2D（次正规数量级的场没有相对精度可言）；取整允许量 = NEAR_ROUNDING_FACTOR_2D × 逐项绝对值之和（_Region2D.build）。
    允许量超过该类峰值的 CHECK_LIMIT_2D 的（分量，行）没有鉴别力，不参加判据，计入 skipped。
    没有行落在叠层之外（z 范围全在叠层之内，又没有探针——只有探针行加不上时才会这样）时返回全 0 的结果，此时这项检查不适用。非有限一律记作 inf。"""
    inc, st = cfg["incident"], cfg["structure"]
    z_bot = stack_bottom_2d(cfg)
    point = request["solve"]["points"][0]
    k0 = 2 * np.pi / point["wavelength"]
    theta, phi = math.radians(point["theta_deg"]), math.radians(point["phi_deg"])
    Es, Ep = _incident_jones_2d(request["solve"])
    inc_xy = (-Es * math.sin(phi) + Ep * math.cos(theta) * math.cos(phi), Es * math.cos(phi) + Ep * math.cos(theta) * math.sin(phi))
    i0 = int(inc["M"]) * (2 * int(inc["My"]) + 1) + int(inc["My"])
    nI, nII = float(st["n_incident"]), common._index(st["n_substrate"])
    x_max, y_max = float(np.abs(x).max()), float(np.abs(y).max())
    regions = (("入射区", np.flatnonzero(z < -REGION_MARGIN_2D), _Region2D(True, nI, far["kzI"], far["Ex_r"], far["Ey_r"], far, k0, i0, z_bot, inc_xy, x_max, y_max)),
               ("出射区", np.flatnonzero(z > z_bot + REGION_MARGIN_2D), _Region2D(False, nII, far["kzII"], far["Ex_t"], far["Ey_t"], far, k0, i0, z_bot, inc_xy, x_max, y_max)))
    if not any(len(rows) for _, rows, _ in regions):
        return NearFieldCheck2D()
    nt = len(x)
    line = np.exp(-1j * k0 * (np.outer(x, far["Kx"]) + np.outer(y, far["Ky"])))        # (nt, N)：谐波 → 切面上的点
    result = NearFieldCheck2D(ratio=-1.)
    for label, rows, region in regions:
        if not len(rows):
            continue
        gaps, scales = np.empty((6, len(rows))), np.empty((6, len(rows)))
        peaks = {"E": 0., "H": 0.}
        for start in range(0, len(rows), NEAR_ROWS_PER_CHUNK_2D):
            chosen = rows[start:start + NEAR_ROWS_PER_CHUNK_2D]
            harmonics, scale = region.build(z[chosen])
            count = len(chosen)
            rebuilt = (line @ np.concatenate(harmonics, axis=1)).reshape(nt, 6, count).transpose(1, 2, 0)     # (6, 行, nt)
            stored = six[:, chosen, :]
            gaps[:, start:start + count] = np.abs(rebuilt - stored).max(axis=2)
            scales[:, start:start + count] = scale
            magnitude = np.abs(stored)
            peaks["E"], peaks["H"] = (max(peaks["E"], float(magnitude[:3].max())), max(peaks["H"], float(magnitude[3:].max())))
        if not (np.isfinite(gaps).all() and np.isfinite(scales).all()):
            return NearFieldCheck2D(ratio=math.inf, residual=math.inf, rows=len(rows), where=f"{label}（重建出非有限数）")
        peak = np.array([max(peaks["E"], common.AMPLITUDE_FLOOR_2D)] * 3 + [max(peaks["H"], common.AMPLITUDE_FLOOR_2D)] * 3)[:, None]
        allowance = NEAR_ROUNDING_FACTOR_2D * scales                                # 绝对量，(分量, 行)
        usable = allowance <= CHECK_LIMIT_2D * peak
        result.skipped += int((~usable).any(axis=0).sum())
        result.rows += int(usable.all(axis=0).sum())
        if label == "入射区":
            result.incident_rows = int(usable.any(axis=0).sum())
        else:
            result.transmitted_rows = int(usable.any(axis=0).sum())
        if not usable.any():
            continue
        ratio = np.where(usable, gaps / (NEAR_FIELD_TOLERANCE_2D * peak + allowance), 0.)
        worst = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
        result.residual = max(result.residual, float(np.where(usable, gaps / peak, 0.).max()))
        result.detect = max(result.detect, float(np.where(usable, NEAR_FIELD_TOLERANCE_2D + allowance / peak, 0.).max()))
        result.noise = max(result.noise, float(np.where(usable & (scales > 0.), gaps / np.where(scales > 0., scales, 1.), 0.).max()))
        if float(ratio[worst]) > result.ratio or not result.where:
            result.ratio, result.allowance = float(ratio[worst]), float(allowance[worst] / peak[worst[0], 0])
            result.where = f"{label}的 {'E' if worst[0] < 3 else 'H'} 场"
    result.ratio = max(result.ratio, 0.)
    return result


def near_field_residual_2d(six, x, y, z, far, request, cfg):
    """near_field_check_2d 的旧接口：(相对残差 差÷峰值 的最大者, 检查的行数, 最坏位置)。没有可检查的行时是 (0., 0, "")。"""
    check = near_field_check_2d(six, x, y, z, far, request, cfg)
    return check.residual, check.rows, check.where


def phase_rounding_allowance_2d(far, request, x, y, z):
    """相位检查在固定容限之外再允许的取整误差（相对峰值）：PHASE_ROUNDING_FACTOR_2D × 入射载波幅角的上界。

    相位检查含 e^{i·幅角}，幅角 = k0·(位置 × 波数)；幅角很大时，同一个数在两个程序里先乘后加的次序不同，取整误差随幅角线性增长
    （实测约 1e-16 × 幅角）。取幅角上界 k0·(|kz₀|·|z|max + |Kx₀|·|x|max + |Ky₀|·|y|max)。
    允许量超过 CHECK_LIMIT_2D 时（例如 λ = 1e-9 µm 且 |z| ~ 1e3 µm：幅角 6e12，允许量 5e-3）检查没有鉴别力，由调用方明确跳过并写进 note。"""
    k0 = 2 * np.pi / request["solve"]["points"][0]["wavelength"]
    i0 = len(far["Kx"]) // 2
    carrier = k0 * (abs(far["kzI"][i0].real) * float(np.abs(z).max()) + abs(far["Kx"][i0]) * float(np.abs(x).max())
                    + abs(far["Ky"][i0]) * float(np.abs(y).max()))
    return PHASE_ROUNDING_FACTOR_2D * carrier


def phase_residual_2d(six, amplitude, phase, mask, x, y, z, far, request):
    """phase.npy 与 fields.npy 的一致性：未掩膜处 |F|·e^{iφ₀} = F，|F|·e^{iφ₁} = F·conj(入射载波)。

    载波 = exp(i k0 (−Kx₀ x − Ky₀ y + kz₀ z))，(Kx₀, Ky₀, kz₀) 取自被激发的 (0,0) 级（kz₀ = nI cos θ 是实数）。
    不用 np.angle：只做 cos、sin 与乘法，O(nz·nt)。返回“最大差 ÷ 该分量的峰值幅度（不小于 AMPLITUDE_FLOOR_2D）”的最大者；非有限记作 inf。
    """
    point = request["solve"]["points"][0]
    k0 = 2 * np.pi / point["wavelength"]
    i0 = len(far["Kx"]) // 2                         # (0,0) 在 m 外层、n 内层的次序里恰在中间
    conj_carrier = (np.exp(-1j * k0 * far["kzI"][i0].real * z)[:, None]
                    * np.exp(1j * k0 * (far["Kx"][i0] * x + far["Ky"][i0] * y))[None, :])
    worst = 0.
    for c in range(6):
        peak = float(amplitude[c].max())
        if peak == 0.:
            continue
        peak = max(peak, common.AMPLITUDE_FLOOR_2D)
        for reference, field in ((0, six[c]), (1, six[c] * conj_carrier)):
            angle = phase[reference, c]
            gap = np.maximum(np.abs(amplitude[c] * np.cos(angle) - field.real), np.abs(amplitude[c] * np.sin(angle) - field.imag))
            gap[mask[reference, c]] = 0.             # 掩膜处相位是 NaN，无从比较
            residual = float(gap.max()) / peak
            worst = max(worst, residual if math.isfinite(residual) else math.inf)
    return worst


def _check_reply_metadata_2d(response, far, manifest, request, wall_seconds):
    """应答里“只报告、不参与计算”的元数据：不能参与物理自洽检查，但可以和请求、和调用者自己量到的墙上时间对一下，伪造的数明显对不上。

    · threads 必须等于请求的线程数（C++ 如实回报）；
    · 计时：全是有限的非负数；知道墙上时间（wall_seconds：调用者从启动程序到读完应答量到的）时，C++ 自己报告的求解总用时与场结果各段用时之和
      都不能超过它——程序是在这段时间里跑的。上限取 1.05×墙上时间 + 1 s（两个时钟只会有毫秒级的差别；这里宽松取，重载的机器上也不会误拒）；
      并行时 operators、eig、linear、product 是各线程用时之和，可以超过墙上时间，按线程数放宽；
    · n_eig（做了几次稠密本征分解）在“结构层数”与“总层数”之间：请求里每个结构层一次（均匀层走快路径不分解），均匀层不会多于总层数；
      没有本征分解（n_eig = 0）时 max_eig_residual 必须恰好是 0；
    · 二维没有求解后向误差：max_solve_backward_error 恒为 0；status 为 ok 时 message 是空串（PROTOCOL_2D.md 第 2 节）。
    n_dense_solve 只核对是非负整数（它随层结构变化，没有可靠的界）。"""
    options = request["solve"]["options"]
    if response["threads"] != options["threads"]:
        raise BackendError(f"C++ 应答的线程数 {response['threads']} 与请求的 {options['threads']} 不符")
    layers = request["solve"]["stack"]["layers"]
    structured = sum("x_profiles" in layer for layer in layers)
    if not structured <= far["n_eig"] <= len(layers):
        raise BackendError(f"C++ 应答的 n_eig = {far['n_eig']} 不在结构层数 {structured} 与总层数 {len(layers)} 之间")
    if far["n_eig"] == 0 and far["max_eig_residual"] != 0:
        raise BackendError("C++ 应答没有做本征分解（n_eig = 0），max_eig_residual 却不是 0")
    if far["max_solve_backward_error"] != 0:
        raise BackendError("C++ 二维应答的 max_solve_backward_error 应恒为 0")
    if far.get("message") != "":
        raise BackendError(f"C++ 应答 status 为 ok，message 却不是空串（{far.get('message')!r}）")
    if wall_seconds is not None:
        limit = 1.05 * wall_seconds + 1.
        threads = options["threads"]
        total = far["timing"]["total"]
        if not total <= limit:
            raise BackendError(f"C++ 应答报告的求解用时 {total:g} s 超过了调用的墙上时间 {wall_seconds:g} s")
        for key in ("operators", "eig", "linear", "product"):
            if not far["timing"][key] <= limit * threads:
                raise BackendError(f"C++ 应答报告的 {key} 用时 {far['timing'][key]:g} s 超过了墙上时间 {wall_seconds:g} s × 线程数")
        if not sum(manifest["timing"].values()) <= limit:
            raise BackendError(f"C++ 场结果报告的各段用时之和 {sum(manifest['timing'].values()):g} s 超过了调用的墙上时间 {wall_seconds:g} s")


def _extract_fields_2d(response, cfg, request, wall_seconds=None, notes=None):
    """Validate a periodic2d reply (PROTOCOL_2D.md section 3) and load its NPY files; BackendError on any mismatch.

    All checks run on the rows exactly as the program was asked for them (user rows plus hidden probe rows, sampling_plan_2d);
    afterwards the probe rows are removed from every per-row array, so what is returned is exactly the user's grid.
    `notes`, a list, receives the sentences that say a check was not applicable (rounding allowance too large)."""
    from gui_fields import normalization
    inc, st, sam = cfg["incident"], cfg["structure"], cfg["sampling"]
    M, My = int(inc["M"]), int(inc["My"])
    N = (2 * M + 1) * (2 * My + 1)
    nt, nz = int(sam["nx"]), int(sam["nz"])
    plane, at = sam["slice_plane"], float(sam["slice_at"])
    period = float(st["period"]) if plane == "xz" else float(st["period_y"])
    z_user, probe_below, probe_above = sampling_plan_2d(cfg)
    n_below, n_above = int(probe_below is not None), int(probe_above is not None)
    sent_z = ([] if probe_below is None else [probe_below]) + z_user.tolist() + ([] if probe_above is None else [probe_above])
    if not _same(request["sampling"]["z"], sent_z):
        raise BackendError("请求里的 z 与采样计划（用户的行加隐藏探针行）不符")
    nz_sent = nz + n_below + n_above
    if response.get("ok") is not True:               # 真值（1、"yes"）不算：协议写的是布尔 true
        raise BackendError(f"C++ 应答的 ok 是 {response.get('ok')!r}，应为 true")
    manifest = response["field_data"]
    if (response.get("protocol") != "stablercwa-fields" or type(manifest.get("version")) is not int
            or manifest.get("version") != FIELDS_VERSION):
        raise BackendError("C++ 场协议或格式版本不符，无法使用完整场结果")
    if (not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS_2D):
        raise BackendError("C++ 二维场清单的键与协议不符（不应有顶层 y，也不应缺少 slice、harmonics）")
    if (type(response.get("M")) is not int or response.get("M") != M or type(response.get("My")) is not int
            or response.get("My") != My or response.get("mode") != "periodic2d" or len(response["results"]) != 1):
        raise BackendError("C++ 求解点数、模式或截断阶数与请求不符")
    far = response["results"][0]
    point = request["solve"]["points"][0]
    if any(far.get(key) != value for key, value in point.items()) or far.get("status") != "ok":
        raise BackendError("C++ 求解入射点与请求不符，或求解状态不是 ok")
    if type(far.get("eig_info")) is not int or far["eig_info"] != 0:
        raise BackendError("C++ 求解出现数值失败（eig_info 非零或缺失）")
    for key in ("n_eig", "n_dense_solve"):
        if type(far.get(key)) is not int or far[key] < 0:
            raise BackendError("C++ 求解诊断计数无效")
    for key in ("max_eig_residual", "max_solve_backward_error"):
        value = far.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise BackendError("C++ 求解诊断误差无效")
    solve_timing = far["timing"]
    if (not isinstance(solve_timing, dict) or set(solve_timing) != SOLVE_TIMING_2D
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value < 0 for value in solve_timing.values())):
        raise BackendError("C++ 求解计时无效")
    if type(response.get("threads")) is not int or not 1 <= response["threads"] <= 256:
        raise BackendError("C++ 求解线程数无效")
    if (manifest["producer"] != PRODUCER or manifest["request_id"] != request["request_id"]
            or manifest["directory"] != request["output_dir"]):
        raise BackendError("C++ 场结果的生产者、请求标记或输出目录不符")
    if (manifest["layout"] != "C" or not _same(manifest["shape"], [nz_sent, nt])
            or not _same(manifest["components"], list(COMPONENTS))):
        raise BackendError("C++ 场结果的 layout、shape 或分量次序不符")
    if not _same(manifest["slice"], {"plane": plane, "at": at, "period": period}):
        raise BackendError("C++ 场结果的切面（平面、固定坐标或周期）与请求不符")
    if not _same(manifest["harmonics"], {"M": M, "My": My, "N": N}):
        raise BackendError("C++ 场结果的谐波截断与请求不符")
    expected_phase = {"references": ["absolute", "incident"], "floor": 1e-6,
                      "unwrap": "none", "unit": "rad", "mask_true": "undefined"}
    if not _same(manifest["phase"], expected_phase):
        raise BackendError("C++ 相位约定与 GUI 不符")
    timing = manifest["timing"]
    if (not isinstance(timing, dict) or set(timing) != {"solve", "sample", "phase", "write"}
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value < 0 for value in timing.values())):
        raise BackendError("C++ 场结果的计时无效")
    _check_reply_metadata_2d(response, far, manifest, request, wall_seconds)
    layers = manifest["layers"]
    geometry = st["layers"]
    if not isinstance(layers, list) or len(layers) != len(geometry):
        raise BackendError("C++ 场层元数据与模型不符")
    top = 0.
    for index, (layer, source) in enumerate(zip(layers, geometry)):
        bottom = top + float(source["thickness"])
        if (not isinstance(layer, dict) or set(layer) != LAYER_KEYS_2D
                or not _same(layer, {"original_index": index, "effective_index": index, "z_top": top,
                                     "z_bottom": bottom, "finite_zero_modes": 0, "schur": False})):
            raise BackendError("C++ 场层位置、映射或传播计数无效（二维保留每一层，没有零根与 Schur）")
        top = bottom
    norm = normalization(cfg)
    actual_norm = manifest["normalization"]
    if (not isinstance(actual_norm, dict) or set(actual_norm) != {"E", "H", "H_convention"}
            or actual_norm["H_convention"] != "Z0 H_physical"
            or any(isinstance(actual_norm[key], bool) or not isinstance(actual_norm[key], (int, float))
                   or not math.isfinite(actual_norm[key]) or actual_norm[key] <= 0
                   or not math.isclose(actual_norm[key], norm[key], rel_tol=1e-14) for key in ("E", "H"))):
        raise BackendError("C++ 入射场归一化与 GUI 不符")
    expected_files = {"fields": ("fields.npy", "<c16", [6, nz_sent, nt]), "phase": ("phase.npy", "<f8", [2, 6, nz_sent, nt]),
                      "phase_mask": ("phase_mask.npy", "|u1", [2, 6, nz_sent, nt]), "t": ("t.npy", "<f8", [nt]),
                      "x": ("x.npy", "<f8", [nt]), "y": ("y.npy", "<f8", [nt]), "z": ("z.npy", "<f8", [nz_sent])}
    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != set(FILES_2D):
        raise BackendError("C++ 场结果的文件清单与协议不符")
    for key, (name, dtype, shape) in expected_files.items():
        if not _same(files[key], {"name": name, "dtype": dtype, "shape": shape}):
            raise BackendError(f"C++ {key} 文件描述、dtype 或 shape 不符")
    directory = Path(request["output_dir"])
    if directory.is_symlink() or not directory.is_dir():
        raise BackendError("C++ 输出目录不存在或不是本次结果目录")
    six = _load_npy(directory, manifest, "fields", "<c16", (6, nz_sent, nt))
    phase = _load_npy(directory, manifest, "phase", "<f8", (2, 6, nz_sent, nt))
    mask_bytes = _load_npy(directory, manifest, "phase_mask", "|u1", (2, 6, nz_sent, nt))
    t = _load_npy(directory, manifest, "t", "<f8", (nt,))
    x = _load_npy(directory, manifest, "x", "<f8", (nt,))
    y = _load_npy(directory, manifest, "y", "<f8", (nt,))
    z = _load_npy(directory, manifest, "z", "<f8", (nz_sent,))
    if (not np.isfinite(six).all() or not np.isfinite(t).all() or not np.isfinite(x).all()
            or not np.isfinite(y).all() or not np.isfinite(z).all()):
        raise BackendError("C++ 六复场或坐标含非有限值")
    expected_t = np.arange(nt) * period / nt
    fixed = np.full(nt, at)
    along_x = plane == "xz"
    if (not np.allclose(t, expected_t, rtol=1e-14, atol=1e-14)
            or not np.array_equal(x, t if along_x else fixed) or not np.array_equal(y, fixed if along_x else t)
            or not np.array_equal(z, np.asarray(request["sampling"]["z"]))):
        raise BackendError("C++ 采样坐标与请求不符")
    if not np.isin(mask_bytes, (0, 1)).all():
        raise BackendError("C++ 相位 mask 含非法值")
    mask = mask_bytes.astype(bool)
    if (not np.isfinite(phase[~mask]).all() or not np.isnan(phase[mask]).all()
            or np.any(abs(phase[~mask]) > np.pi + 1e-14)):
        raise BackendError("C++ 相位非有限、未正确掩膜或超出包裹范围")
    amplitude = abs(six)
    expected_mask = amplitude <= 1e-6 * amplitude.max(axis=(1, 2), keepdims=True)
    if not np.array_equal(mask[0], expected_mask) or not np.array_equal(mask[1], expected_mask):
        raise BackendError("C++ 相位 mask 与 GUI 弱场规则不符")
    # 物理自洽（形式正确、数值被改过的文件到这里才露馅）：近场对远场、相位对场。容限与依据见本节常量的注释。
    # 近场检查看叠层之外的全部行（含探针行）：入射偏振与相位、各级反射/透射振幅由它与请求钉在一起。叠层之内的行没有独立的参照，不检查。
    far_values = far_arrays_2d(far, N)
    with np.errstate(all="ignore"):          # 被改坏的数组可能溢出：得到 inf/NaN，下面按“不自洽”拒绝，不必在终端里吐警告
        near = near_field_check_2d(six, x, y, z, far_values, request, cfg)
        phase_allowance = phase_rounding_allowance_2d(far_values, request, x, y, z)
        phase_checked = phase_allowance <= CHECK_LIMIT_2D
        phase_residual = phase_residual_2d(six, amplitude, phase, mask, x, y, z, far_values, request) if phase_checked else 0.
    if not near.ratio <= 1.:
        raise BackendError(f"C++ 六复场在叠层之外的行与远场应答重建的场不一致（{near.where}，检查了 {near.rows} 行，"
                           f"相对残差 {near.residual:.1e}，容限 {NEAR_FIELD_TOLERANCE_2D:g} + 取整允许量 {near.allowance:.1e}）")
    if not phase_residual <= PHASE_TOLERANCE_2D + phase_allowance:
        raise BackendError(f"C++ 相位文件与六复场不自洽（|F|e^{{iφ}} 对 F 或对 F·conj(入射载波)，"
                           f"相对残差 {phase_residual:.1e}，容限 {PHASE_TOLERANCE_2D:g}）")
    if notes is not None:
        if probe_below is None and probe_above is None:
            notes.append(f"没有加隐藏探针行：{nz}×{nt} 个采样点再加两行会超过协议的上限 {MAX_SAMPLES_2D}；近场自洽检查只看 z 范围里叠层之外的行")
        if near.skipped:
            notes.append(f"近场自洽检查跳过了 {near.skipped} 行：取整允许量超过该类峰值的 {CHECK_LIMIT_2D:g}（波长相对 |z| 极小），这些行没有鉴别力")
        if not near.incident_rows:
            notes.append("入射偏振与入射相位没有与请求核对：入射区（z < 0）没有一行能参加近场自洽检查，反射振幅也没有核对")
        if not near.transmitted_rows:
            notes.append("透射振幅没有与请求核对：出射区（z > 叠层底面）没有一行能参加近场自洽检查")
        if not phase_checked:
            notes.append(f"相位自洽检查未执行：载波幅角太大，取整允许量 {phase_allowance:.1e} 超过 {CHECK_LIMIT_2D:g}，没有鉴别力")
    if n_below or n_above:
        # 去掉探针行：每个按行的数组（场、相位、掩膜、z）只留用户的 nz 行。探针行只参加了上面的检查。
        keep = slice(n_below, n_below + nz)
        if not np.array_equal(z[keep], z_user):
            raise BackendError("C++ 采样坐标去掉探针行后与用户的 z 不符")
        _rederive_weak_field_mask_2d(six, amplitude, phase, mask, x, y, z, far_values, 2 * np.pi / point["wavelength"], keep, n_below)
        six, phase, mask, z = six[:, keep, :], phase[:, :, keep, :], mask[:, :, keep, :], z[keep]
    fields = {key: six[j] for j, key in enumerate(COMPONENTS)}
    fields.update(x=x, y=y, t=t, z=z, slice_plane=np.asarray(plane), slice_at=np.asarray(at),
                  E2=sum(abs(six[j])**2 for j in range(3))/norm["E"]**2,
                  field_archive_version=np.asarray(FIELDS_VERSION), field_backend=np.asarray("cpp"),
                  phase_archive_version=np.asarray(PHASE_VERSION), phase_backend=np.asarray("cpp"),
                  phase_floor=np.asarray(1e-6), phase_unwrap=np.asarray("none"),
                  phase_components=np.asarray(COMPONENTS), phase_references=np.asarray(("absolute", "incident")),
                  phase_values=phase, phase_masks=mask)
    if not np.isfinite(fields["E2"]).all():
        raise BackendError("C++ 复场强度溢出，无法显示")
    return fields, manifest


def _rederive_weak_field_mask_2d(six, amplitude, phase, mask, x, y, z, far, k0, keep, offset):
    """去掉探针行之后，弱场掩膜规则（|F| ≤ 1e-6·该分量的峰值）要按用户自己的行重新取峰值：探针行的幅度可以比用户的行大，
    C++ 按“含探针行”的峰值掩膜的点，在用户的网格上可能不该被掩膜（反过来不会发生：去掉行只会降低峰值）。
    原地改 phase、mask：规则下不再掩膜、但 C++ 写了 NaN 的点，在 Python 里补上相位——公式与 rcwa_phase.hpp 相同（载波幅角
    −k0Kx₀·x − k0Ky₀·y + k0kz₀·z，reference 0 取 arg F，reference 1 取 arg(F·conj(载波))），但**不是同一份代码，运算次序也不同**：
    补上的点上，两个 reference 与不加探针的 C++ 运行都只在 1e-14 rad 以内相等，不是逐位相同：
    实测（8 个模型里 6 个触发这个分支）：macOS 上 reference 0 逐位相同（arg F 只取决于 F，两边的 atan2 相同）、reference 1 在 197–6364 个点上相差、最大 2.2e-15；
    Windows（MinGW 的 libm 与 numpy 的 atan2 末位不同）上 reference 0 在 3–13 个点相差 ≤ 4.4e-16，reference 1 在 478–1890 个点相差 ≤ 4.4e-16。保证的上界取 1e-14。
    所以“用户的数据与没有探针时逐位相同”在这个分支里只对场、掩膜、E2、以及补上的点以外的全部相位成立；补上的那些点上的相位在 1e-14 rad 以内相等。
    没有这样的点（探针行的幅度不超过用户行，这是常态）时一个数也不动，全部逐位相同。keep 是用户行在含探针行的数组里的切片，offset 是它的起点。"""
    region = amplitude[:, keep, :]
    now_weak = region <= 1e-6 * region.max(axis=(1, 2), keepdims=True)
    was_weak = mask[0][:, keep, :]
    if np.array_equal(was_weak, now_weak):
        return
    i0 = len(far["Kx"]) // 2
    kx0, ky0, kz0 = k0 * far["Kx"][i0], k0 * far["Ky"][i0], k0 * far["kzI"][i0].real
    for newly, strong in ((was_weak & ~now_weak, True), (~was_weak & now_weak, False)):
        c, iz, it = np.nonzero(newly)
        if not len(c):
            continue
        row = iz + offset
        if strong:
            field = six[c, row, it]
            carrier = np.exp(1j * (-kx0 * x[it] - ky0 * y[it] + kz0 * z[row]))
            phase[0, c, row, it] = np.angle(field)
            phase[1, c, row, it] = np.angle(field * np.conj(carrier))
        else:
            phase[0, c, row, it] = phase[1, c, row, it] = np.nan
        mask[0, c, row, it] = mask[1, c, row, it] = not strong


def extract_fields(response, cfg, request, wall_seconds=None, notes=None):
    """wall_seconds：调用者从启动程序到读完应答量到的墙上时间（二维用来核对应答里的计时，不给就不核对）；
    notes：列表，二维的某项检查因取整允许量过大而没有执行时把说明追加进去（PROTOCOL_2D.md 第 6 节）。"""
    try:
        if cfg["mode"] == "periodic2d":
            return _extract_fields_2d(response, cfg, request, wall_seconds, notes)
        manifest = response["field_data"]
        nx, nz = int(cfg["sampling"]["nx"]), int(cfg["sampling"]["nz"])
        if (response.get("protocol") != "stablercwa-fields" or type(manifest.get("version")) is not int
                or manifest.get("version") != FIELDS_VERSION):
            raise BackendError("C++ 场协议或格式版本不符，无法使用完整场结果")
        if response.get("M") != cfg["incident"]["M"] or len(response["results"]) != 1:
            raise BackendError("C++ 求解点数或截断阶数与请求不符")
        far = response["results"][0]
        point = request["solve"]["points"][0]
        if any(far.get(key) != value for key, value in point.items()):
            raise BackendError("C++ 求解入射点与请求不符")
        if type(far.get("eig_info")) is not int or far["eig_info"] != 0:
            raise BackendError("C++ 求解出现数值失败（eig_info 非零或缺失）")
        for key in ("n_eig", "n_dense_solve"):
            if type(far.get(key)) is not int or far[key] < 0:
                raise BackendError("C++ 求解诊断计数无效")
        for key in ("max_eig_residual", "max_solve_backward_error"):
            value = far.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise BackendError("C++ 求解诊断误差无效")
        solve_timing = far["timing"]
        if (not isinstance(solve_timing, dict) or set(solve_timing) != {"total", "eig", "linear", "product"}
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) or value < 0 for value in solve_timing.values())):
            raise BackendError("C++ 求解计时无效")
        if type(response.get("threads")) is not int or not 1 <= response["threads"] <= 256:
            raise BackendError("C++ 求解线程数无效")
        if cfg["mode"] == "conical":
            if (type(far.get("n_schur")) is not int or not 0 <= far["n_schur"] <= len(cfg["structure"]["layers"])
                    or type(far.get("degenerate_mode_warning")) is not bool):
                raise BackendError("C++ 圆锥传播诊断无效")
            minimum = far.get("min_abs_lambda")
            if minimum is not None and (isinstance(minimum, bool) or not isinstance(minimum, (int, float))
                                        or not math.isfinite(minimum) or minimum < 0):
                raise BackendError("C++ 圆锥最小本征值诊断无效")
        if (manifest.get("producer") != PRODUCER or manifest.get("request_id") != request["request_id"]
                or manifest.get("directory") != request["output_dir"]):
            raise BackendError("C++ 场结果的生产者、请求标记或输出目录不符")
        if manifest.get("layout") != "C" or manifest.get("shape") != [nz, nx] or manifest.get("components") != list(COMPONENTS):
            raise BackendError("C++ 场结果的 layout、shape 或分量次序不符")
        expected_phase = {"references": ["absolute", "incident"], "floor": 1e-6,
                          "unwrap": "none", "unit": "rad", "mask_true": "undefined"}
        if manifest.get("phase") != expected_phase or manifest.get("y") != 0.:
            raise BackendError("C++ 相位约定或剖面位置与 GUI 不符")
        timing = manifest["timing"]
        if (set(timing) != {"solve", "sample", "phase", "write"}
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) or value < 0 for value in timing.values())):
            raise BackendError("C++ 场结果的计时无效")
        layers = manifest["layers"]
        geometry = cfg["structure"]["layers"]
        if not isinstance(layers, list) or len(layers) != len(geometry):
            raise BackendError("C++ 场层元数据与模型不符")
        top = 0.; effective = 0
        for index, (layer, source) in enumerate(zip(layers, geometry)):
            bottom = top + float(source["thickness"])
            expected_index = -1 if cfg["mode"] == "conical" and source["thickness"] == 0. else effective
            if expected_index >= 0:effective += 1
            if (type(layer["original_index"]) is not int or layer["original_index"] != index
                    or type(layer["effective_index"]) is not int or layer["effective_index"] != expected_index
                    or layer["z_top"] != top or layer["z_bottom"] != bottom
                    or type(layer["finite_zero_modes"]) is not int or not 0 <= layer["finite_zero_modes"] <= 2*(2*cfg["incident"]["M"]+1)
                    or type(layer["schur"]) is not bool):
                raise BackendError("C++ 场层位置、映射或传播计数无效")
            top = bottom
        if cfg["mode"] == "conical" and sum(layer["schur"] for layer in layers) != far["n_schur"]:
            raise BackendError("C++ Schur 层计数与场元数据不符")
        directory = Path(request["output_dir"])
        if directory.is_symlink() or not directory.is_dir():
            raise BackendError("C++ 输出目录不存在或不是本次结果目录")
        six = _load_npy(directory, manifest, "fields", "<c16", (6, nz, nx))
        phase = _load_npy(directory, manifest, "phase", "<f8", (2, 6, nz, nx))
        mask_bytes = _load_npy(directory, manifest, "phase_mask", "|u1", (2, 6, nz, nx))
        x = _load_npy(directory, manifest, "x", "<f8", (nx,))
        z = _load_npy(directory, manifest, "z", "<f8", (nz,))
        if not np.isfinite(six).all() or not np.isfinite(x).all() or not np.isfinite(z).all():
            raise BackendError("C++ 六复场或坐标含非有限值")
        expected_x = np.arange(nx) * float(cfg["structure"]["period"]) / nx
        if not np.allclose(x, expected_x, rtol=1e-14, atol=1e-14) or not np.array_equal(z, np.asarray(request["sampling"]["z"])):
            raise BackendError("C++ 采样坐标与请求不符")
        if not np.isin(mask_bytes, (0, 1)).all():
            raise BackendError("C++ 相位 mask 含非法值")
        mask = mask_bytes.astype(bool)
        if (not np.isfinite(phase[~mask]).all() or not np.isnan(phase[mask]).all()
                or np.any(abs(phase[~mask]) > np.pi + 1e-14)):
            raise BackendError("C++ 相位非有限、未正确掩膜或超出包裹范围")
        # Masks use each component's own peak; no Python angle or phase reconstruction.
        amplitude = abs(six)
        expected_mask = amplitude <= 1e-6 * amplitude.max(axis=(1, 2), keepdims=True)
        if not np.array_equal(mask[0], expected_mask) or not np.array_equal(mask[1], expected_mask):
            raise BackendError("C++ 相位 mask 与 GUI 弱场规则不符")
        from gui_fields import normalization
        norm = normalization(cfg)
        actual_norm = manifest["normalization"]
        if (actual_norm.get("H_convention") != "Z0 H_physical"
                or any(isinstance(actual_norm.get(key), bool) or not isinstance(actual_norm.get(key), (int, float))
                       or not math.isfinite(actual_norm[key]) or actual_norm[key] <= 0
                       or not math.isclose(actual_norm[key], norm[key], rel_tol=1e-14) for key in ("E", "H"))):
            raise BackendError("C++ 入射场归一化与 GUI 不符")
        fields = {key: six[j] for j, key in enumerate(COMPONENTS)}
        fields.update(x=x, z=z, y=np.asarray(0.), E2=sum(abs(six[j])**2 for j in range(3))/norm["E"]**2,
                      field_archive_version=np.asarray(FIELDS_VERSION), field_backend=np.asarray("cpp"),
                      phase_archive_version=np.asarray(PHASE_VERSION), phase_backend=np.asarray("cpp"),
                      phase_floor=np.asarray(1e-6), phase_unwrap=np.asarray("none"),
                      phase_components=np.asarray(COMPONENTS), phase_references=np.asarray(("absolute", "incident")),
                      phase_values=phase, phase_masks=mask)
        if not np.isfinite(fields["E2"]).all():
            raise BackendError("C++ 复场强度溢出，无法显示")
        return fields, manifest
    except (KeyError, IndexError, TypeError, AttributeError, OverflowError) as error:
        raise BackendError(f"C++ 场应答缺少或含非法内容（{type(error).__name__}：{error}）") from None


class Attempt(common.Attempt):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.fields = None
        self.manifest = None
        self.response = None
        self.load_seconds = None
        self.verification_status = "not_requested"

    def result_2d(self, stack, cfg):
        """Periodic 2D adapter: a Result2D-shaped dict of the C++ far-field arrays (no modes, no amplitudes)."""
        from gui_models_2d import load_rcwa2d
        rc2 = load_rcwa2d()
        inc, values = cfg["incident"], self.values
        M, My = inc["M"], inc["My"]
        m, n = np.meshgrid(np.arange(-M, M + 1), np.arange(-My, My + 1), indexing="ij")
        mn = np.column_stack([m.ravel(), n.ravel()])      # same construction (and dtype) as Lattice.harmonics
        if not np.array_equal(mn, np.asarray(values["mn"])):
            raise BackendError("C++ 级次表与 GUI 的谐波次序不一致")
        arrays = {key: np.asarray(values[key], float) for key in ("Kx", "Ky", "DEr", "DEt")}
        arrays.update({key: np.asarray(values[key], complex) for key in
                       ("kzI", "kzII", "Ex_r", "Ey_r", "Ez_r", "Ex_t", "Ey_t", "Ez_t", "Rs", "Rp", "Ts", "Tp")})
        arrays.update({key: np.asarray(values[key], bool) for key in ("prop_r", "prop_t")})
        theta, phi = np.radians(inc["angle_deg"]), np.radians(inc["phi_deg"])
        return rc2.Result2D(mn=mn, M1=M, M2=My, k0=2 * np.pi / inc["wavelength"], lat=stack.lat, stack=stack,
                            i0=int(np.flatnonzero((mn[:, 0] == 0) & (mn[:, 1] == 0))[0]), theta=theta, phi=phi,
                            nI=stack.nI, nII=stack.nII, N=len(mn), **arrays)

    def result(self, stack, cfg):
        """GUI result adapter. It owns far-field arrays and never constructs Python modes."""
        if cfg["mode"] == "periodic2d":
            return self.result_2d(stack, cfg)
        from rcwa1d import Result
        from spectral1d import layer_fourier_info
        far = self.response["results"][0]
        values = self.values
        result = Result({key: np.asarray(value) if isinstance(value, list) else value for key, value in values.items()})
        result.update(ms=np.arange(-cfg["incident"]["M"], cfg["incident"]["M"] + 1),
                      fourier_layers=[layer_fourier_info(layer, stack.period, cfg["incident"]["M"]) for layer in stack.layers],
                      mode_residual=None, n_schur=far.get("n_schur", 0),
                      n_zero_modes=sum(layer["finite_zero_modes"] for layer in self.manifest["layers"]))
        if self.mode == "conical":
            for key in ("Jr", "Jt"):
                result[key] = result[key].reshape(-1, 2, 2)
        return result

    def verify(self, reference):
        comparison = common.Attempt({"mode": self.mode})
        comparison.used, comparison.values = True, self.values
        comparison.merge(reference)
        self.python_n_schur = comparison.python_n_schur
        self.python_zero_modes = comparison.python_zero_modes
        if comparison.used:
            self.difference, self.jones_difference = comparison.difference, comparison.jones_difference
            self.note = common._join(self.note, comparison.note)
            self.verification_status = "compared"
        else:
            self.note = common._join(self.note, "Python 核对未完成：" + comparison.note)
            self.verification_status = "failed"

    def report(self, python_solve_seconds=0.):
        block = super().report(python_solve_seconds)
        block.update(fields="cpp" if self.used else "python", phase="cpp" if self.used else "python",
                     protocol="stablercwa-fields", protocol_version=FIELDS_VERSION,
                     phase_version=PHASE_VERSION if self.used else None,
                     python_solve_ran=bool(python_solve_seconds > 0), verification_status=self.verification_status)
        if self.mode == "conical":block["max_jones_difference"] = self.jones_difference if self.used else None
        if self.used:
            far = self.response["results"][0]
            block.update(producer=PRODUCER, cpp_load_seconds=self.load_seconds, cpp_field_seconds=self.manifest["timing"]["sample"],
                         cpp_phase_seconds=self.manifest["timing"]["phase"], cpp_write_seconds=self.manifest["timing"]["write"],
                         cpp_zero_modes=sum(layer["finite_zero_modes"] for layer in self.manifest["layers"]))
            for key in ("eig_info", "n_eig", "n_dense_solve", "max_eig_residual", "max_solve_backward_error"):
                block["cpp_" + key] = far[key]
        return block


_LINK_REPARSE_TAGS = (getattr(stat, "IO_REPARSE_TAG_SYMLINK", 0xA000000C), getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", 0xA0000003))


def _is_link(path):
    """符号链接，或 Windows 的联接点（目录联接，也是一种重解析点）：删除时只能删链接本身，不能跟进去。

    重解析点不都是链接：OneDrive 按需同步目录里的真目录也带 FILE_ATTRIBUTE_REPARSE_POINT（云文件标签），那是普通目录，要整棵删。
    所以只认符号链接与挂载点（联接）两种标签；读不到标签（旧版 Python）时保守地当作链接，不跟进去。
    """
    try:
        info = os.lstat(path)
    except OSError:
        return False
    if stat.S_ISLNK(info.st_mode):
        return True
    if not getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        return False
    tag = getattr(info, "st_reparse_tag", None)
    return tag is None or tag in _LINK_REPARSE_TAGS


def _remove_entry(entry):
    """删掉一个目录项：真目录整棵删，链接只删链接本身，普通文件直接删。Windows 上文件还被占用时重试（_retry_while_in_use）。"""
    path = Path(entry.path)
    if _is_link(path):
        try:
            os.unlink(path)
        except OSError:
            common._retry_while_in_use(path.rmdir)          # Windows 的目录联接、目录符号链接：unlink 不认，rmdir 删的是链接本身（不进入目标）
    elif entry.is_dir(follow_symlinks=False):
        common._retry_while_in_use(lambda: shutil.rmtree(path))
    else:
        common._retry_while_in_use(path.unlink)


def _discard_matching(outdir, matches):
    """删掉 outdir 的直接子项里名字满足 matches 的；返回 (找到的个数, 没删掉的个数)（Windows 上文件还被占用，等到重试用完）。不抛异常。"""
    try:
        entries = list(os.scandir(outdir))
    except OSError:
        return 0, 0
    matched = failed = 0
    for entry in entries:
        if not matches(entry.name):
            continue
        matched += 1
        try:
            _remove_entry(entry)
        except OSError:
            failed += 1
    return matched, failed


def discard_call_outputs(outdir, token):
    """删掉这一次调用在结果目录里建的东西：cpp_fields-<token> 与 cpp_fields-<token>.tmp-*（C++ 先在 .tmp-* 里写、写完再改名发布）。

    只认这次调用自己的令牌（uuid4 的 32 位十六进制）：名字必须正好是 cpp_fields-<token> 或以 cpp_fields-<token>.tmp- 开头，
    别的令牌、别的文件一概不碰；符号链接只删链接本身，不跟进去。删不掉的（Windows 上文件还被占用）静默放过，
    不让清理的失败盖过原来的错误。超时、取消、被拒绝的应答、中途异常之后调用；成功的调用不调用它，结果目录保持原样。
    工作进程被 TerminateProcess 结束（Windows 上的取消）时这里的 finally 不会运行：那种情况由界面服务器调用 discard_job_outputs。
    """
    base = "cpp_fields-" + token
    _discard_matching(outdir, lambda name: name == base or name.startswith(base + ".tmp-"))


OUTPUT_NAME = re.compile(r"cpp_fields-[0-9a-f]{32}(\.tmp-.*)?", re.DOTALL)       # attempt 起的名字：令牌是 uuid4().hex，.tmp-* 是程序的发布前缀


def discard_job_outputs(folder, seconds=3.0):
    """一个已被取消的任务的结果目录：删掉里面所有 cpp_fields-<令牌> 与 cpp_fields-<令牌>.tmp-*（只认 32 位十六进制令牌的名字，别的一概不碰）。

    界面服务器取消任务时调用，在工作进程已经结束（wait 返回）之后。POSIX 上取消是 SIGTERM，工作进程自己的 finally 会清理（discard_call_outputs），
    C++ 调用期间被取消时这里什么也找不到；应答被采用之后（保存六分量数据期间）被取消时，被采用的目录两个平台都由这里删掉。
    Windows 上取消是 TerminateProcess，工作进程没有机会清理——取消发生在 C++ 程序运行、写结果或校验期间时，目录（nx=1600、nz=1200 时 391 MB）
    会留下。C++ 子进程由作业对象在工作进程消失时收拾，但这是异步的：它可能还握着 .tmp-* 里的文件一会儿，所以删不掉的重试到 seconds 秒，
    并在删掉过东西之后再扫一遍（抓住刚被发布的目录）。seconds 只管“再扫一遍”：每一项自己还有 common._retry_while_in_use 的重试（约 2 s），
    文件被别的进程长期占住时实测 4–6 s 才返回（占住 1–2 项 4.0 s，3 项 6.0 s）；正常的取消 0.1–0.2 s。
    folder 是这个任务自己的目录（每个任务一个新目录，里面的 cpp_fields-* 都是它的工作进程建的）；不是路径（None、数字……）或空串时什么也不做
    （os.scandir(None) 会去扫当前目录）。不抛异常；返回最后一遍还没删掉的个数。符号链接、目录联接只删链接本身。
    """
    if not isinstance(folder, (str, os.PathLike)) or not str(folder):
        return 0
    deadline = time.monotonic() + seconds
    matches = lambda name: OUTPUT_NAME.fullmatch(name) is not None
    while True:
        matched, failed = _discard_matching(folder, matches)
        if not matched:
            return 0                                        # 没有（了）：工作进程自己清理过时，这里一进来就走
        if time.monotonic() >= deadline:
            return failed
        time.sleep(0.05 if failed else 0.15)                # 删过东西：稍等再扫一遍（抓住刚被发布的目录、等占着文件的子进程退出）


def attempt(cfg, outdir, progress=None):
    result = Attempt(cfg)
    started = time.perf_counter()
    token = None
    try:
        if common.forced_python():
            result.note = "环境变量 RCWA_BACKEND=python：只用 Python 求解器"
            return result
        fits, reason = common.eligibility(cfg)
        if not fits:
            result.note = reason
            return result
        with common._children_die_with_us():
            common._STATE["compiled"] = False
            exe, note = ensure_executable(progress)
            if common._STATE["compiled"]:
                result.build_seconds = time.perf_counter() - started
            if exe is None:
                result.note = note
                return result
            if progress:
                progress("正在用 C++ 计算衍射效率、六分量场与相位")
            threads = os.environ.get("RCWA_BLAS_THREADS", "")
            token = uuid.uuid4().hex
            directory = Path(outdir) / ("cpp_fields-" + token)
            request = build_request(cfg, directory, token, int(threads) if threads.isdigit() else 1)
            called = time.perf_counter()
            try:
                response = common.call(exe, request, outdir)
            finally:
                result.seconds = time.perf_counter() - called
            # 校验也在 SIGTERM 的处理函数管辖之内：程序返回后、校验结束前被取消（几百 MB 的结果文件要校验好几秒）时，
            # 处理函数抛出 SystemExit，下面的 finally 照样删掉这次调用建的目录；处理函数只在 with 块里装着，块外的 SIGTERM 会直接结束进程。
            if result.mode == "conical":
                result.note_conical(response)
            result.values = common.extract(response, cfg)
            loaded = time.perf_counter()
            notes = []
            result.fields, result.manifest = extract_fields(response, cfg, request, result.seconds, notes)
            result.load_seconds = time.perf_counter() - loaded
            result.response, result.used, result.note = response, True, note
            for sentence in notes:
                result.note = common._join(result.note, sentence)
    except BackendError as error:
        result.note = str(error)
    except Exception as error:
        result.note = f"C++ 场后端出错（{type(error).__name__}：{error}），已退回 Python"
    finally:
        if token is not None and not result.used:       # 超时、取消（SystemExit、KeyboardInterrupt 也走这里）、被拒绝的应答
            discard_call_outputs(outdir, token)
        result.elapsed = time.perf_counter() - started
    return result
