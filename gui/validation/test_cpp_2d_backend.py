"""Periodic 2D RCWA through the C++ protocol (PROTOCOL_2D.md): Python / GUI side.

Run from gui/:  OPENBLAS_NUM_THREADS=1 python3 -m unittest validation.test_cpp_2d_backend -v

Two programs speak the protocol here.

* STAND-IN (always available): this very file, started as ``python test_cpp_2d_backend.py --standin``. It reads
  a request exactly as PROTOCOL_2D.md sections 1-3 describe (unknown keys are errors), solves with
  ``gui/2drcwa/rcwa2d.py`` (``solve2d`` / ``field_at``, the same expressions as the Python GUI path) and writes the
  reply and hand-made NPY files (C order, explicit dtypes). It is a *reference implementation of the contract*,
  not the C++ program: it lets the whole Python path (request, validation, fallback, worker, saved results,
  HTTP) be tested without a compiler, and its numbers must equal the Python path bit for bit.
  RCWA_STANDIN_PIDFILE makes it write its process id; RCWA_STANDIN_HANG=<file> makes it leave a half-finished
  ``.tmp-`` result directory like the real program, write the directory's name into <file> and never answer
  (timeout, kill and cancel tests of the leftover-directory cleanup).
* REAL program: ``RCWA_FIELDS_EXE`` if set, else the program ``cpp_fields_backend.ensure_executable()`` finds or builds
  in gui/bin. A minimal periodic2d request probes whether it speaks the 2D protocol; if not, every real-program test
  is skipped and the skip message says why (so a program delivered later is picked up automatically).
  Its numbers are compared with Python and with the frozen reference at the frozen tolerance (1e-9).
"""
import copy
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

GUI = Path(__file__).resolve().parents[1]
STANDIN_FLAG = "--standin"
COMPONENT_NAMES = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


# =====================================================================================================================
# The stand-in program. Only numpy and rcwa2d.py are needed, so it is defined (and dispatched) before the GUI imports.
# =====================================================================================================================
class _RequestError(Exception):
    def __init__(self, message, where=""):
        super().__init__(message)
        self.message, self.where = message, where


def _standin_main():
    import importlib.util
    import numpy as np

    spec = importlib.util.spec_from_file_location("standin_rcwa2d", GUI / "2drcwa" / "rcwa2d.py")
    rc2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc2)

    def need(condition, message, where):
        if not condition:
            raise _RequestError(message, where)

    def child(where, key):
        return f"{where}.{key}" if where else key

    def only(obj, where, allowed):
        need(isinstance(obj, dict), "必须是 JSON 对象", where)
        for key in obj:
            need(key in allowed, f"不认识的键 {key!r}", child(where, key))

    def real(value, where):
        need(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), "必须是有限实数", where)
        return float(value)

    def whole(value, where):
        need(isinstance(value, int) and not isinstance(value, bool), "必须是整数", where)
        return value

    def index(value, where):
        if isinstance(value, list):
            need(len(value) == 2, "折射率写成实数或 [实部, 虚部]", where)
            return complex(real(value[0], where + "[0]"), real(value[1], where + "[1]"))
        return complex(real(value, where))

    def edges(value, period, where):
        need(isinstance(value, list), "必须是数组", where)
        values = [real(v, f"{where}[{i}]") for i, v in enumerate(value)]
        need(len(values) >= 2 and all(b > a for a, b in zip(values, values[1:])), "边界必须严格递增", where)
        need(abs((values[-1] - values[0]) - period) <= 1e-9 * period, "首尾之差必须等于一个周期", where)     # 与 C++ 相同：1e-9 倍周期
        return values

    def parse_solve(solve):
        only(solve, "", {"mode", "stack", "M", "My", "points", "wavelength", "theta_deg", "phi_deg",
                              "input", "options"})
        need(solve.get("mode") == "periodic2d", "mode 必须是 periodic2d", "mode")
        need("stack" in solve and "M" in solve and "My" in solve, "缺少 stack、M 或 My（二维没有默认 M）", "")
        need("polarization" not in solve and "period" not in solve, "二维不用 polarization、period", "")
        stack = solve["stack"]
        only(stack, "stack", {"nI", "nII", "Lx", "Ly", "layers"})
        nI = real(stack.get("nI"), "stack.nI")
        need(nI > 0, "nI 必须是正实数", "stack.nI")
        nII = index(stack.get("nII"), "stack.nII")
        Lx, Ly = real(stack.get("Lx"), "stack.Lx"), real(stack.get("Ly"), "stack.Ly")
        need(Lx > 0 and Ly > 0, "周期必须为正", "stack")
        layers = []
        need(isinstance(stack.get("layers"), list), "layers 必须是数组", "stack.layers")
        for j, row in enumerate(stack["layers"]):
            where = f"stack.layers[{j}]"
            need(isinstance(row, dict), "每层必须是对象", where)
            thickness = real(row.get("d"), where + ".d")
            need(thickness >= 0, "层厚不能为负", where + ".d")
            if "n" in row:
                only(row, where, {"d", "n"})
                layers.append(rc2.Layer2D(thickness, n=index(row["n"], where + ".n"), name=f"第{j + 1}层"))
                continue
            only(row, where, {"d", "y_edges", "x_profiles"})
            y_edges = edges(row.get("y_edges"), Ly, where + ".y_edges")
            profiles = row.get("x_profiles")
            need(isinstance(profiles, list) and len(profiles) == len(y_edges) - 1, "x_profiles 的个数必须比 y_edges 少 1",
                 where + ".x_profiles")
            rows = []
            for k, profile in enumerate(profiles):
                at = f"{where}.x_profiles[{k}]"
                only(profile, at, {"edges", "n_values"})
                x_edges = edges(profile.get("edges"), Lx, at + ".edges")
                values = profile.get("n_values")
                need(isinstance(values, list) and len(values) == len(x_edges) - 1, "n_values 的个数必须比 edges 少 1", at + ".n_values")
                rows.append((x_edges, [index(v, f"{at}.n_values[{i}]") for i, v in enumerate(values)]))
            layers.append(rc2.Layer2D(thickness, y_edges=y_edges, x_profiles=rows, name=f"第{j + 1}层"))
        M, My = whole(solve["M"], "M"), whole(solve["My"], "My")
        need(M >= 0 and My >= 0 and (2 * M + 1) * (2 * My + 1) <= 2048, "M、My 不能为负，谐波总数不得超过 2048", "M")
        if "points" in solve:
            need(not any(k in solve for k in ("wavelength", "theta_deg", "phi_deg")), "points 与最外层单点只能取一种", "points")
            raw_points = solve["points"]
            need(isinstance(raw_points, list) and raw_points, "points 至少一个点", "points")
        else:
            raw_points = [{k: solve[k] for k in ("wavelength", "theta_deg", "phi_deg") if k in solve}]
        points = []
        for j, point in enumerate(raw_points):
            where = f"points[{j}]"
            only(point, where, {"wavelength", "theta_deg", "phi_deg"})
            wavelength = real(point.get("wavelength"), where + ".wavelength")
            need(wavelength > 0, "波长必须为正", where + ".wavelength")
            points.append((wavelength, real(point.get("theta_deg"), where + ".theta_deg"),
                           real(point.get("phi_deg", 0.0), where + ".phi_deg")))
        need("input" in solve, "二维必须给 input（不写不能默认，免得悄悄取 s）", "input")
        given = solve["input"]
        only(given, "input", {"psi_deg", "delta_deg", "jones"})
        if "jones" in given:
            need("psi_deg" not in given and "delta_deg" not in given, "jones 与 psi_deg/delta_deg 只能取一种", "input")
            jones = [index(v, f"input.jones[{i}]") for i, v in enumerate(given["jones"])]
            need(len(jones) == 2, "jones 是 [Es, Ep]", "input.jones")
            angles, jones_in = None, jones
        else:
            angles = (real(given.get("psi_deg"), "input.psi_deg"), real(given.get("delta_deg", 0.0), "input.delta_deg"))
            jones_in = None
        options = solve.get("options", {})
        only(options, "options", {"eig", "factorization", "homogeneous_fast_path", "parallel_layers", "diagnostics", "threads"})
        need(options.get("eig", "qr") in ("qr", "lr"), "eig 必须是 qr 或 lr", "options.eig")
        factorization = options.get("factorization", "li")
        need(factorization in ("li", "laurent"), "factorization 必须是 li 或 laurent", "options.factorization")
        threads = whole(options.get("threads", 1), "options.threads")
        need(1 <= threads <= 256, "threads 为 1 ~ 256", "options.threads")
        return dict(stack=rc2.Stack2D(nI, layers, nII, rc2.Lattice(Lx, Ly)), M=M, My=My, points=points, angles=angles,
                    jones=jones_in, factorization=factorization, threads=threads,
                    diagnostics=bool(options.get("diagnostics", False)), nI=nI, nII=nII, Lx=Lx, Ly=Ly)

    def pairs(values):
        return [[float(complex(v).real), float(complex(v).imag)] for v in values]

    def solve_point(model, point):
        wavelength, theta, phi = point
        started = time.perf_counter()
        kwargs = dict(factorization=model["factorization"])
        if model["jones"] is not None:
            kwargs["jones_in"] = model["jones"]
        else:
            kwargs.update(psi_deg=model["angles"][0], delta_deg=model["angles"][1])
        res = rc2.solve2d(model["stack"], wavelength, theta, phi, model["M"], model["My"], **kwargs)
        return res, time.perf_counter() - started

    def head(model):
        orders = [[m, n] for m in range(-model["M"], model["M"] + 1) for n in range(-model["My"], model["My"] + 1)]
        return {"ok": True, "version": 1, "mode": "periodic2d", "M": model["M"], "My": model["My"], "orders": orders,
                "threads": model["threads"]}

    def result_json(model, point, res, seconds, failure=None):
        wavelength, theta, phi = point
        N = (2 * model["M"] + 1) * (2 * model["My"] + 1)
        out = {"wavelength": wavelength, "theta_deg": theta, "phi_deg": phi,
               "status": "ok" if failure is None else failure[0], "message": "" if failure is None else failure[1]}
        names = ("Kx", "Ky", "kzI", "kzII", "Rs", "Rp", "Ts", "Tp", "Ex_r", "Ey_r", "Ez_r", "Ex_t", "Ey_t", "Ez_t", "DEr", "DEt")
        if failure is not None:
            for key in names:
                out[key] = [None] * N if key in ("Kx", "Ky", "DEr", "DEt") else [[None, None]] * N
            out.update(prop_r=[0] * N, prop_t=[0] * N, eig_info=0 if failure[0] == "rayleigh_anomaly" else 1)
        else:
            for key in names:
                out[key] = [float(v) for v in res[key]] if key in ("Kx", "Ky", "DEr", "DEt") else pairs(res[key])
            out.update(prop_r=[int(v) for v in res["prop_r"]], prop_t=[int(v) for v in res["prop_t"]], eig_info=0)
        structured = sum(not layer.homogeneous for layer in model["stack"].layers)
        out.update(n_eig=structured, n_dense_solve=len(model["stack"].layers) + 1, max_eig_residual=0.0,
                   max_solve_backward_error=0.0,
                   timing={"total": seconds, "operators": 0.0, "eig": 0.0, "linear": 0.0, "product": seconds})
        return out

    def solve_failure(model, point):
        try:
            res, seconds = solve_point(model, point)
            return res, seconds, None
        except rc2.RayleighAnomalyError as error:
            return None, 0.0, ("rayleigh_anomaly", str(error))
        except np.linalg.LinAlgError as error:
            return None, 0.0, ("numerical_failure", str(error))

    def npy(path, array, descr):
        array = np.ascontiguousarray(np.asarray(array).astype(descr))
        shape = "(" + ", ".join(str(n) for n in array.shape) + (",)" if array.ndim == 1 else ")")
        header = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}"
        padding = -(10 + len(header) + 1) % 64                 # magic 6 + version 2 + length 2, then header, newline
        header = (header + " " * padding + "\n").encode("latin1")
        with open(path, "wb") as stream:
            stream.write(b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header + array.tobytes())

    def run_fields(request):
        only(request, "", {"solve", "sampling", "output_dir", "request_id"})
        model = parse_solve(request.get("solve"))
        need(len(model["points"]) == 1, "fields 协议每次只接受一个入射点", "solve.points")
        sampling = request.get("sampling")
        only(sampling, "sampling", {"z", "plane", "nt", "at"})
        need("plane" in sampling and "nt" in sampling and "z" in sampling, "plane、nt、z 必填", "sampling")
        plane = sampling["plane"]
        need(plane in ("xz", "yz"), "plane 必须是 xz 或 yz", "sampling.plane")
        nt = whole(sampling["nt"], "sampling.nt")
        need(nt >= 1, "必须是正整数", "sampling.nt")
        at = real(sampling.get("at", 0.0), "sampling.at")
        need(isinstance(sampling["z"], list) and sampling["z"], "z 必须是非空数组", "sampling.z")
        z = np.asarray([real(v, f"sampling.z[{i}]") for i, v in enumerate(sampling["z"])])
        need(nt * len(z) <= 4000000, "采样网格超过 4000000 个点", "sampling")
        path = request.get("output_dir")
        need(isinstance(path, str) and path and "\0" not in path, "output_dir 不能为空或含 NUL", "output_dir")
        token = request.get("request_id")
        need(isinstance(token, str) and 1 <= len(token.encode("utf-8")) <= 128, "request_id 必须是 1 到 128 字节", "request_id")
        final = Path(os.path.abspath(path))
        need(final.name != "", "必须是本次新建的子目录", "output_dir")
        if final.exists():
            raise RuntimeError("fields 输出目录已存在，拒绝覆盖历史文件")
        if not final.parent.is_dir():
            raise RuntimeError("fields 输出目录的父目录不存在")
        if os.environ.get("RCWA_STANDIN_HANG"):
            # 演练超时与取消：像真程序那样先建临时目录、写下半截文件，把目录名告诉测试（环境变量给的是通知文件），然后不再返回
            partial = final.parent / (final.name + f".tmp-{os.getpid()}-{time.time_ns()}")
            partial.mkdir()
            (partial / "fields.npy").write_bytes(b"\x93NUMPY partial")
            Path(os.environ["RCWA_STANDIN_HANG"]).write_text(str(partial), encoding="utf-8")
            time.sleep(600)
        point = model["points"][0]
        res, solve_seconds, failure = solve_failure(model, point)
        if failure is not None:
            raise RuntimeError(f"{failure[0]}：{failure[1]}")
        started = time.perf_counter()
        # The same expressions as the GUI's Python path (gui_worker.compute), so the result is bit-identical.
        period = model["Lx"] if plane == "xz" else model["Ly"]
        t = np.arange(nt) * period / nt
        xs, ys = (t, np.full_like(t, at)) if plane == "xz" else (np.full_like(t, at), t)
        phase_matrix = np.exp(-1j * res.k0 * (np.outer(xs, res.Kx) + np.outer(ys, res.Ky)))
        harmonics = [rc2.field_at(res, float(value)) for value in z]
        six = np.stack([np.asarray([h[k] for h in harmonics]) @ phase_matrix.T for k in COMPONENT_NAMES])
        if not np.isfinite(six).all():
            raise RuntimeError("fields 采样产生非有限复场，拒绝发布")
        sample_seconds = time.perf_counter() - started
        started = time.perf_counter()
        kx0, ky0, kz0 = res.k0 * res.Kx[res.i0], res.k0 * res.Ky[res.i0], res.k0 * res.kzI[res.i0].real
        carrier = np.exp(1j * (-kx0 * xs[None, :] - ky0 * ys[None, :] + kz0 * z[:, None]))
        phase = np.full((2, 6) + six.shape[1:], np.nan)
        mask = np.zeros(phase.shape, np.uint8)
        for c in range(6):
            amplitude = abs(six[c])
            weak = amplitude <= 1e-6 * amplitude.max()
            for r, field in enumerate((six[c], six[c] * np.conj(carrier))):
                values = np.angle(field)
                values[weak] = np.nan
                phase[r, c], mask[r, c] = values, weak
        phase_seconds = time.perf_counter() - started
        started = time.perf_counter()
        temporary = final.parent / (final.name + f".tmp-{os.getpid()}-{time.time_ns()}")
        temporary.mkdir()
        try:
            npy(temporary / "fields.npy", six, "<c16")
            npy(temporary / "phase.npy", phase, "<f8")
            npy(temporary / "phase_mask.npy", mask, "|u1")
            npy(temporary / "t.npy", t, "<f8")
            npy(temporary / "x.npy", xs, "<f8")
            npy(temporary / "y.npy", ys, "<f8")
            npy(temporary / "z.npy", z, "<f8")
            write_seconds = time.perf_counter() - started
            nz = len(z)
            top, layers = 0.0, []
            for j, layer in enumerate(model["stack"].layers):
                layers.append({"original_index": j, "effective_index": j, "z_top": top, "z_bottom": top + layer.d,
                               "finite_zero_modes": 0, "schur": False})
                top += layer.d
            norm_e = 1.0 if model["jones"] is None else float(math.hypot(abs(model["jones"][0]), abs(model["jones"][1])))
            reply = head(model)
            reply["results"] = [result_json(model, point, res, solve_seconds)]
            reply["protocol"] = "stablercwa-fields"
            reply["field_data"] = {
                "version": 1, "producer": "stablercwa.cpp.rcwa_fields", "request_id": token, "directory": str(final),
                "layout": "C", "shape": [nz, nt], "components": list(COMPONENT_NAMES),
                "slice": {"plane": plane, "at": at, "period": period},
                "harmonics": {"M": model["M"], "My": model["My"], "N": (2 * model["M"] + 1) * (2 * model["My"] + 1)},
                "phase": {"references": ["absolute", "incident"], "floor": 1e-6, "unwrap": "none", "unit": "rad",
                          "mask_true": "undefined"},
                "normalization": {"E": norm_e, "H": model["nI"] * norm_e, "H_convention": "Z0 H_physical"},
                "files": {"fields": {"name": "fields.npy", "dtype": "<c16", "shape": [6, nz, nt]},
                          "phase": {"name": "phase.npy", "dtype": "<f8", "shape": [2, 6, nz, nt]},
                          "phase_mask": {"name": "phase_mask.npy", "dtype": "|u1", "shape": [2, 6, nz, nt]},
                          "t": {"name": "t.npy", "dtype": "<f8", "shape": [nt]},
                          "x": {"name": "x.npy", "dtype": "<f8", "shape": [nt]},
                          "y": {"name": "y.npy", "dtype": "<f8", "shape": [nt]},
                          "z": {"name": "z.npy", "dtype": "<f8", "shape": [nz]}},
                "layers": layers,
                "timing": {"solve": solve_seconds, "sample": sample_seconds, "phase": phase_seconds, "write": write_seconds}}
            os.rename(temporary, final)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return reply

    def run_far(request):
        model = parse_solve(request)
        reply = head(model)
        reply["results"] = []
        for point in model["points"]:
            res, seconds, failure = solve_failure(model, point)
            reply["results"].append(result_json(model, point, res, seconds, failure))
        return reply

    code = 0
    try:
        request = json.loads(sys.stdin.buffer.read().decode("utf-8"))
        if os.environ.get("RCWA_STANDIN_PIDFILE"):
            Path(os.environ["RCWA_STANDIN_PIDFILE"]).write_text(str(os.getpid()))
        reply = run_fields(request) if isinstance(request, dict) and "solve" in request else run_far(request)
    except _RequestError as error:
        reply, code = {"ok": False, "error": error.message, "where": error.where}, 1
    except Exception as error:
        reply, code = {"ok": False, "error": str(error), "where": "fields"}, 1
    sys.stdout.buffer.write(json.dumps(reply, ensure_ascii=False).encode("utf-8"))
    sys.stdout.flush()
    return code


if __name__ == "__main__" and sys.argv[1:2] == [STANDIN_FLAG]:
    raise SystemExit(_standin_main())


# =====================================================================================================================
# The tests
# =====================================================================================================================
# Bit-for-bit agreement of the stand-in with the in-process Python path is asserted only when BLAS runs single-threaded:
# OPENBLAS_NUM_THREADS=1 given by the caller, or unset and numpy not yet imported by anyone (then it is set here).
EXACT_BITS = os.environ.get("OPENBLAS_NUM_THREADS") == "1" or (
    os.environ.get("OPENBLAS_NUM_THREADS") is None and "numpy" not in sys.modules)
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("MPLBACKEND", "Agg")
if str(GUI) not in sys.path:
    sys.path.insert(0, str(GUI))

import contextlib  # noqa: E402
import io  # noqa: E402
from contextlib import contextmanager  # noqa: E402

import numpy as np  # noqa: E402

import cpp_backend as common  # noqa: E402
import cpp_fields_backend as backend  # noqa: E402
import gui_fields  # noqa: E402
import gui_models  # noqa: E402
import gui  # noqa: E402
import gui_worker  # noqa: E402
from gui_models_2d import build_stack_2d, load_rcwa2d, normalize_2d, preset_2d  # noqa: E402
from validation import field_reference_2d_cases as frozen_cases  # noqa: E402
from validation.test_cpp_backend import EnvPatch, Server, make_fake_executable, process_alive  # noqa: E402

STATE = {"standin": None, "real": None, "real_note": "", "real_far": None, "real_far_note": "", "dir": None}
BackendError = backend.BackendError
COMPONENTS = gui_fields.COMPONENTS
# Extra keys the C++ result carries (versioned phase arrays, as in the 1-D C++ path); everything else is the Python key set.
CPP_EXTRA_KEYS = {"field_archive_version", "field_backend", "phase_archive_version", "phase_backend", "phase_floor",
                  "phase_unwrap", "phase_components", "phase_references", "phase_values", "phase_masks"}
FROZEN_TOLERANCE = 1e-9


def say(text):
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(text.encode(encoding, "replace").decode(encoding))


WINDOWS_UTF8_HEADER = "@chcp 65001 >nul\r\n"     # cmd.exe reads a .bat in the console code page (936 on Chinese Windows): without this
                                                  # line a UTF-8 path with Chinese characters inside the shim turns into mojibake


def make_fake(folder):
    """test_cpp_backend.make_fake_executable, with the Windows shim switched to UTF-8 first (the test folders have Chinese names)."""
    exe = make_fake_executable(folder)
    if os.name == "nt":
        exe.write_text(WINDOWS_UTF8_HEADER + exe.read_text(encoding="utf-8"), encoding="utf-8")
    return exe


def make_standin(folder):
    script = Path(__file__).resolve()
    if os.name == "nt":
        exe = folder / "替身 rcwa_fields.bat"
        exe.write_text(WINDOWS_UTF8_HEADER + f'@"{sys.executable}" "{script}" {STANDIN_FLAG} %*\r\n', encoding="utf-8")
    else:
        exe = folder / "替身 rcwa_fields"
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" {STANDIN_FLAG} "$@"\n', encoding="utf-8")
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return exe


def probe_2d(exe, far=False):
    """Does this program speak the 2-D protocol (fields envelope, or the plain far-field request with far=True)?"""
    folder = Path(tempfile.mkdtemp(prefix="rcwa 2d probe "))
    try:
        solve = {"mode": "periodic2d", "stack": {"nI": 1.0, "nII": 1.5, "Lx": 1.0, "Ly": 1.0, "layers": [{"d": 0.2, "n": 1.4}]},
                 "M": 0, "My": 0, "points": [{"wavelength": 1.0, "theta_deg": 10.0, "phi_deg": 20.0}],
                 "input": {"psi_deg": 90.0, "delta_deg": 0.0}, "options": {}}
        request = solve if far else {"solve": solve, "sampling": {"z": [0.1], "plane": "xz", "nt": 4, "at": 0.0},
                                     "output_dir": str(folder / "out"), "request_id": "probe"}
        try:
            code, out, err = common._run_child([str(exe)], 120, text=json.dumps(request))
            reply = json.loads(out)
        except Exception as error:
            return False, f"{type(error).__name__}：{error}"
        if not isinstance(reply, dict):
            return False, "应答不是 JSON 对象"
        if reply.get("ok") is True and reply.get("mode") == "periodic2d" and (far or "field_data" in reply):
            return True, ""
        return False, str(reply.get("error", "应答不是二维协议"))
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def setUpModule():
    STATE["dir"] = Path(tempfile.mkdtemp(prefix="rcwa 中文 2d "))
    STATE["standin"] = make_standin(STATE["dir"])
    exe, note = None, ""
    try:
        with EnvPatch(RCWA_BACKEND=None, RCWA_JSON_EXE=None):
            exe, note = backend.ensure_executable()
    except Exception as error:
        note = f"{type(error).__name__}：{error}"
    if exe is None:
        STATE["real_note"] = "没有可用的 rcwa_fields 程序：" + note
    else:
        ok, why = probe_2d(exe)
        STATE["real"], STATE["real_note"] = (exe, "") if ok else (None, f"{exe} 不支持 periodic2d（{why}）")
    far, far_note = None, ""
    try:
        with EnvPatch(RCWA_BACKEND=None):
            far, far_note = common.ensure_executable()
    except Exception as error:
        far_note = f"{type(error).__name__}：{error}"
    if far is None:
        STATE["real_far_note"] = "没有可用的 rcwa_json 程序：" + far_note
    else:
        ok, why = probe_2d(far, far=True)
        STATE["real_far"], STATE["real_far_note"] = (far, "") if ok else (None, f"{far} 不支持 periodic2d（{why}）")
    say("\n    real fields program: " + (str(STATE["real"]) if STATE["real"] else "NOT USED, real-program tests skipped — " + STATE["real_note"])
        + "\n    real far-field program: " + (str(STATE["real_far"]) if STATE["real_far"] else "NOT USED — " + STATE["real_far_note"]))


def tearDownModule():
    shutil.rmtree(STATE["dir"], ignore_errors=True)


@contextmanager
def cpp_env(exe, **extra):
    """The environment of a run that is meant to take the 2-D C++ route. RCWA_2D_BACKEND=cpp is part of it: 2-D models go to Python by
    default (cpp_backend.DEFAULT_BACKEND_2D), so every test that exercises the C++ route has to ask for it (class Routing2D tests the default)."""
    values = dict(RCWA_FIELDS_EXE=exe, RCWA_BACKEND=None, RCWA_JSON_EXE=None, RCWA_VERIFY_PYTHON=None, RCWA_CPP_TIMEOUT=None, RCWA_2D_BACKEND="cpp")
    values.update(extra)
    with EnvPatch(**values):
        yield


# ----------------------------------------------------------------------------------------------------- models
def model_pillar(plane="xz", pol="s", **incident):
    """The GUI preset (pillar + two films) at N = 25 and a small section."""
    cfg = preset_2d()
    cfg["incident"].update(M=2, My=2, polarization=pol, psi_deg=33., delta_deg=50.)
    cfg["incident"].update(incident)
    cfg["sampling"].update(nx=40, nz=20, slice_plane=plane, slice_at=.4 if plane == "xz" else .5)
    return normalize_2d(cfg)


def model_absorbing():
    """Lx != Ly, M != My, negative angle, elliptical Jones, yz section; an x-only layer, a shifted 2-D layer, lossy film."""
    return normalize_2d({
        "schema_version": 1, "name": "吸收二维", "mode": "periodic2d",
        "incident": {"wavelength": .83, "angle_deg": -27., "phi_deg": -40., "polarization": "jones", "M": 2, "My": 3,
                     "psi_deg": 37., "delta_deg": 71.},
        "structure": {"period": .62, "period_y": .93, "n_incident": 1.2, "n_substrate": 1.45, "layers": [
            {"name": "x条纹", "kind": "patterned", "thickness": .12, "edges": [0., .2, .62], "n_values": [1.9, "1.4+0.02j"]},
            {"name": "吸收柱", "kind": "patterned2d", "thickness": .27, "y_edges": [-.1, .3, .6, .83],
             "x_profiles": [{"edges": [0., .62], "n_values": [1.46]},
                            {"edges": [.1, .35, .72], "n_values": [1.46, "2.3+0.13j"]},
                            {"edges": [0., .62], "n_values": [1.46]}]},
            {"name": "吸收膜", "kind": "uniform", "thickness": .09, "n": "1.65+0.12j"}]},
        "sampling": {"nx": 40, "nz": 20, "z_min": -.15, "z_max": .6, "slice_plane": "yz", "slice_at": .37}})


def model_y_invariant():
    """phi = 0, s polarisation, nothing varies along y: Ex, Ez, Hy vanish identically, so the phase mask is populated."""
    return normalize_2d({
        "schema_version": 1, "name": "沿y不变", "mode": "periodic2d",
        "incident": {"wavelength": 1.053, "angle_deg": 31., "phi_deg": 0., "polarization": "s", "M": 3, "My": 0},
        "structure": {"period": .574, "period_y": .8, "n_incident": 1., "n_substrate": 1.46, "layers": [
            {"name": "光栅", "kind": "patterned", "thickness": .38, "edges": [-.086, .086, .488], "n_values": [1.46, 1.]},
            {"name": "膜", "kind": "uniform", "thickness": .115, "n": 1.96}]},
        "sampling": {"nx": 40, "nz": 20, "z_min": -.1, "z_max": .6, "slice_plane": "xz", "slice_at": .21}})


def model_uniform():
    """No structured layer at all: every layer takes the 2x2 block-diagonal fast path."""
    return normalize_2d({
        "schema_version": 1, "name": "均匀膜", "mode": "periodic2d",
        "incident": {"wavelength": 1., "angle_deg": 27., "phi_deg": 33., "polarization": "p", "M": 1, "My": 1},
        "structure": {"period": 1.1, "period_y": .8, "n_incident": 1., "n_substrate": 1.38, "layers": [
            {"name": "A", "kind": "uniform", "thickness": .27, "n": "1.65+0.12j"},
            {"name": "B", "kind": "uniform", "thickness": .1, "n": 1.46}]},
        "sampling": {"nx": 40, "nz": 20, "z_min": -.1, "z_max": .5, "slice_plane": "xz", "slice_at": .3}})


def model_degenerate():
    """My = 0 truncation, normal incidence with phi = 30 (the kt = 0 s/p branch), eccentric pillar."""
    cfg = preset_2d()
    cfg["incident"].update(M=3, My=0, angle_deg=0., phi_deg=30., polarization="p", wavelength=.93)
    cfg["sampling"].update(nx=40, nz=20, slice_plane="xz", slice_at=.31)
    return normalize_2d(cfg)


def model_square(plane="xz", period=.9, **incident):
    """Lx = Ly with an asymmetric pillar: both sections have the same period and shape [nz, nt], so one section's files can
    pass for the other's, while the fields differ."""
    f = period / .9
    cfg = {
        "schema_version": 1, "name": "方格", "mode": "periodic2d",
        "incident": {"wavelength": .93, "angle_deg": 24., "phi_deg": 33., "polarization": "jones", "M": 2, "My": 2,
                     "psi_deg": 41., "delta_deg": 63.},
        "structure": {"period": period, "period_y": period, "n_incident": 1., "n_substrate": 1.45, "layers": [
            {"name": "偏心柱", "kind": "patterned2d", "thickness": .3, "y_edges": [0., .25 * f, .55 * f, period],
             "x_profiles": [{"edges": [0., period], "n_values": [1.]},
                            {"edges": [0., .1 * f, .45 * f, period], "n_values": [1., 1.9, 1.]},
                            {"edges": [0., period], "n_values": [1.]}]},
            {"name": "膜", "kind": "uniform", "thickness": .16, "n": 1.46}]},
        "sampling": {"nx": 40, "nz": 20, "z_min": -.2, "z_max": .77, "slice_plane": plane, "slice_at": .35}}
    cfg["incident"].update(incident)
    return normalize_2d(cfg)


def model_edge(span_error, period=.8, plane="xz"):
    """The pillar of the GUI preset on a period `period` whose edge lists (the middle x profile and the layer's y list) end
    `span_error` um beyond one period. gui_models_2d accepts up to 1e-8·max(period, 1); the C++ protocol only 1e-9·period."""
    return normalize_2d({
        "schema_version": 1, "name": "边界偏差", "mode": "periodic2d",
        "incident": {"wavelength": 1., "angle_deg": 20., "phi_deg": 30., "polarization": "jones", "M": 2, "My": 2,
                     "psi_deg": 33., "delta_deg": 50.},
        "structure": {"period": period, "period_y": .8, "n_incident": 1., "n_substrate": 1.45, "layers": [
            {"name": "柱", "kind": "patterned2d", "thickness": .3, "y_edges": [0., .2, .6, .8 + span_error],
             "x_profiles": [{"edges": [0., period], "n_values": [1.]},
                            {"edges": [0., .3 * period, .7 * period, period + span_error], "n_values": [1., 1.7, 1.]},
                            {"edges": [0., period], "n_values": [1.]}]},
            {"name": "膜", "kind": "uniform", "thickness": .16, "n": 1.46}]},
        "sampling": {"nx": 40, "nz": 20, "z_min": -.2, "z_max": .77, "slice_plane": plane, "slice_at": .4 if plane == "xz" else .5}})


def model_rows(z_min, z_max, nz=20, plane="xz", **incident):
    """The GUI preset (pillar + two films, stack 0 ... .57 um) with a chosen z range: rows only inside the stack, only above, only below ..."""
    cfg = model_pillar(plane, "jones", **incident)
    cfg["sampling"].update(z_min=z_min, z_max=z_max, nz=nz)
    return normalize_2d(cfg)


def model_metal_inside(plane="xz"):
    """z rows deep inside a 0.3 um film of 0.2+6j (skin depth ~13 nm): the field there is 1e-4 ... 1e-9 of the field next to the film, i.e. of the
    hidden probe rows, so the weak-field mask C++ writes (peak over ALL rows) is not the one of the user's own rows (the re-derivation branch)."""
    pillar = {"name": "柱", "kind": "patterned2d", "thickness": .08, "y_edges": [0., .15, .5, .7],
              "x_profiles": [{"edges": [0., .6], "n_values": [1.46]}, {"edges": [0., .12, .36, .6], "n_values": [1.46, "2.3+0.13j", 1.46]},
                             {"edges": [0., .6], "n_values": [1.46]}]}
    return normalize_2d({
        "schema_version": 1, "name": "金属内部", "mode": "periodic2d",
        "incident": {"wavelength": .5, "angle_deg": 23., "phi_deg": 31., "polarization": "jones", "M": 2, "My": 2, "psi_deg": 37., "delta_deg": 71.},
        "structure": {"period": .6, "period_y": .7, "n_incident": 1., "n_substrate": 1.45, "layers": [
            pillar, {"name": "金属", "kind": "uniform", "thickness": .3, "n": "0.2+6j"}, {"name": "膜", "kind": "uniform", "thickness": .1, "n": 1.5}]},
        "sampling": {"nx": 64, "nz": 40, "z_min": .13, "z_max": .33, "slice_plane": plane, "slice_at": .37 * (.7 if plane == "xz" else .6)}})


def model_cutoff(eps):
    """Normal incidence on a 1 x 1 um cell with lambda = 1 + eps: the orders (+-1, 0) and (0, +-1) sit at n^2 - K^2 = -2 eps (eps > 0:
    evanescent, 2 eps below cutoff; eps < 0: propagating, grazing). The checks' ambiguity band for the propagation flag is |n^2 - Kx^2 - Ky^2| <= 1e-12 scale."""
    pillar = {"name": "柱", "kind": "patterned2d", "thickness": .3, "y_edges": [0., .25, .6, 1.],
              "x_profiles": [{"edges": [0., 1.], "n_values": [1.46]}, {"edges": [0., .2, .6, 1.], "n_values": [1.46, "2.3+0.13j", 1.46]},
                             {"edges": [0., 1.], "n_values": [1.46]}]}
    return normalize_2d({
        "schema_version": 1, "name": "截止附近", "mode": "periodic2d",
        "incident": {"wavelength": 1. + eps, "angle_deg": 0., "phi_deg": 0., "polarization": "jones", "M": 1, "My": 1, "psi_deg": 37., "delta_deg": 71.},
        "structure": {"period": 1., "period_y": 1., "n_incident": 1., "n_substrate": 1.45, "layers": [
            pillar, {"name": "膜", "kind": "uniform", "thickness": .16, "n": 1.46}]},
        "sampling": {"nx": 40, "nz": 20, "z_min": -.3, "z_max": .8, "slice_plane": "xz", "slice_at": .37}})


def no_probes(cfg):
    """sampling_plan_2d as it was before the hidden probe rows: exactly the user's rows."""
    sampling = cfg["sampling"]
    return np.linspace(sampling["z_min"], sampling["z_max"], sampling["nz"]), None, None


WORKER_MODELS = (("pillar xz s", lambda: model_pillar("xz", "s")), ("pillar yz p", lambda: model_pillar("yz", "p")),
                 ("pillar xz jones", lambda: model_pillar("xz", "jones")), ("absorbing yz jones", model_absorbing),
                 ("uniform only", model_uniform), ("My=0, normal incidence", model_degenerate))


def run_program(exe, cfg, folder, threads=1, token="rcwa2d-token"):
    """One request/response against `exe` the way cpp_fields_backend.attempt does it; returns (request, response)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    request = backend.build_request(cfg, folder / "out", token, threads)
    with cpp_env(exe):
        return request, common.call(exe, request, folder)


def run_compute(cfg, folder, **env):
    """gui_worker.compute in this process; returns (metrics, saved NPZ as a dict). Takes the 2-D C++ route unless the caller says otherwise
    (RCWA_2D_BACKEND is `cpp` here; pass RCWA_2D_BACKEND=None for the product default)."""
    env.setdefault("RCWA_2D_BACKEND", "cpp")
    with EnvPatch(**env):
        metrics = gui_worker.compute(copy.deepcopy(cfg), folder)
    assert metrics == json.loads((Path(folder) / "metrics.json").read_text(encoding="utf-8"))
    with np.load(Path(folder) / "fields.npz", allow_pickle=False) as archive:
        return metrics, {key: archive[key] for key in archive.files}


# ----------------------------------------------------------------------------------------------------- base class
class Case(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rcwa 中文 2d case "))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    @staticmethod
    def program(name):
        exe = STATE[name]
        if exe is None:
            raise unittest.SkipTest(STATE["real_note"] if name == "real" else "stand-in unavailable")
        return exe


# ===================================================================================================== 1. eligibility
class Eligibility(Case):
    """Which 2-D models the C++ route can take at all (with RCWA_2D_BACKEND=cpp; the switch itself is tested in Routing2D)."""

    def setUp(self):
        super().setUp()
        patch_env = EnvPatch(RCWA_2D_BACKEND="cpp", RCWA_BACKEND=None)
        patch_env.__enter__()
        self.addCleanup(patch_env.__exit__, None, None, None)

    def test_builtin_preset_and_variants_fit(self):
        for name, build in WORKER_MODELS:
            with self.subTest(model=name):
                self.assertEqual(common.eligibility(build()), (True, ""))
        self.assertEqual(common.eligibility(gui_models.preset("pillar_2d")), (True, ""))
        for My, M in ((0, 0), (12, 12), (30, 0), (0, 30)):          # N = 1, 625, 61, 61
            cfg = model_pillar()
            cfg["incident"].update(M=M, My=My)
            self.assertEqual(common.eligibility(cfg), (True, ""), (M, My))

    def test_too_many_harmonics_is_refused_with_the_number(self):
        for M, My, N in ((13, 12, 675), (12, 13, 675), (30, 30, 3721)):
            cfg = model_pillar()
            cfg["incident"].update(M=M, My=My)
            fits, why = common.eligibility(cfg)
            self.assertFalse(fits)
            self.assertIn(str(N), why)
            self.assertIn("625", why)

    def test_unsupported_models_say_why(self):
        def refused(change, *words):
            cfg = model_pillar()
            change(cfg)
            fits, why = common.eligibility(cfg)
            self.assertFalse(fits)
            for word in words:
                self.assertIn(word, why)

        layers = lambda cfg: cfg["structure"]["layers"]
        refused(lambda c: layers(c)[0].update(fourier_method="fft"), "第 1 层", "FFT")
        refused(lambda c: layers(c)[0].update(fft_samples=512), "FFT")
        refused(lambda c: layers(c)[1].update(n="1.658/1.486@x"), "第 2 层", "单轴")
        refused(lambda c: layers(c)[0]["x_profiles"][1]["n_values"].__setitem__(1, "1.7/1.5@z"), "第 1 层", "单轴")
        refused(lambda c: layers(c)[0].update(background_n=1.), "背景")
        refused(lambda c: layers(c)[0].update(kind="spiral"), "类型")
        refused(lambda c: c["structure"].update(n_incident="1.2+0.1j"), "入射区", "吸收")
        refused(lambda c: c["structure"].update(n_substrate="1.5/1.4@x"), "出射区", "单轴")
        refused(lambda c: c["incident"].update(M=-1), "M")
        refused(lambda c: c["incident"].pop("My"), "My")

    def test_other_modes_keep_their_reasons(self):
        self.assertIn("PML", common.eligibility(gui_models.preset("single_bar"))[1])
        self.assertTrue(common.eligibility(gui_models.preset("trapezoid_analytic"))[0])

    def test_environment_switches(self):
        cfg = model_pillar()
        with cpp_env(self.program("standin"), RCWA_BACKEND="python"):
            attempt = backend.attempt(cfg, self.tmp)
        self.assertFalse(attempt.used)
        self.assertIn("RCWA_BACKEND", attempt.note)
        self.assertIsNone(attempt.seconds)
        self.assertFalse((self.tmp / "cpp_request.json").exists())
        with cpp_env(self.tmp / "不存在"):
            attempt = backend.attempt(cfg, self.tmp)
        self.assertFalse(attempt.used)
        self.assertIn("不存在", attempt.note)
        with cpp_env(None, RCWA_FIELDS_EXE=None, RCWA_JSON_EXE=self.tmp / "old_far"):
            attempt = backend.attempt(cfg, self.tmp)
        self.assertFalse(attempt.used)
        self.assertIn("旧效率协议", attempt.note)
        for value, expected in ((None, False), ("0", False), ("1", True)):
            with EnvPatch(RCWA_VERIFY_PYTHON=value):
                self.assertEqual(backend.verify_requested(), expected)
        fft = model_pillar()
        fft["structure"]["layers"][0]["fourier_method"] = "fft"
        with cpp_env(self.program("standin")):
            attempt = backend.attempt(fft, self.tmp)
        self.assertFalse(attempt.used)
        self.assertIn("FFT", attempt.note)


# ===================================================================================================== 1b. the 2-D route switch (round 3, M2)
class Routing2D(Case):
    """2-D models go to the PYTHON solver by default (cpp_backend.DEFAULT_BACKEND_2D: C++ is only faster at larger N); RCWA_2D_BACKEND=cpp makes
    C++ the default, and the GUI's 二维求解 option (backend_2d in the model) decides per model, over the variable. 1-D routing ignores both."""

    UNRECOGNISED = ("auto", "gpu", "1", "true", "c++", "cpp2", "none")

    def test_the_default_is_python_and_the_note_is_neutral(self):
        cfg = model_pillar()
        self.assertEqual(common.DEFAULT_BACKEND_2D, "python")
        for value in (None, "", "   ", "python", "PYTHON", " Python "):
            with self.subTest(value=value), EnvPatch(RCWA_2D_BACKEND=value, RCWA_BACKEND=None):
                self.assertEqual(common.backend_2d(), ("python", ""))
                self.assertEqual(common.eligibility(cfg), (False, common.NOTE_2D_PYTHON))
        self.assertIn("RCWA_2D_BACKEND=cpp", common.NOTE_2D_PYTHON)               # says how to turn C++ on
        for alarm in ("失败", "错误", "不可用", "警告", "注意", "无法"):             # one neutral sentence, no alarm
            self.assertNotIn(alarm, common.NOTE_2D_PYTHON)
        # the switch comes before the structural checks: a model C++ could not take anyway gets the same sentence
        with EnvPatch(RCWA_2D_BACKEND=None, RCWA_BACKEND=None):
            unsupported = model_pillar()
            unsupported["structure"]["layers"][0].update(fourier_method="fft")
            self.assertEqual(common.eligibility(unsupported), (False, common.NOTE_2D_PYTHON))

    def test_the_switch_values(self):
        cfg = model_pillar()
        for value in ("cpp", "CPP", " cpp ", "Cpp"):
            with self.subTest(value=value), EnvPatch(RCWA_2D_BACKEND=value, RCWA_BACKEND=None):
                self.assertEqual(common.backend_2d(), ("cpp", ""))
                self.assertEqual(common.eligibility(cfg), (True, ""))
                unsupported = model_pillar()
                unsupported["structure"]["layers"][0].update(fourier_method="fft")      # with the switch on, the structural reasons come back
                fits, why = common.eligibility(unsupported)
                self.assertFalse(fits)
                self.assertIn("FFT", why)
        for value in self.UNRECOGNISED:                  # not a silent C++ (nor a silent Python): the note says the value was not understood
            with self.subTest(value=value), EnvPatch(RCWA_2D_BACKEND=value, RCWA_BACKEND=None):
                choice, problem = common.backend_2d()
                self.assertEqual(choice, "python")
                fits, why = common.eligibility(cfg)
                self.assertFalse(fits)
                self.assertEqual(why, problem)
                self.assertIn("无法识别", why)
                self.assertIn(repr(value), why)
                self.assertIn(common.NOTE_2D_PYTHON, why)

    def test_rcwa_backend_does_not_decide_the_2d_route(self):
        """RCWA_BACKEND has one meaning, python (everything off, 1-D included); `cpp` is not a value it knows, so it cannot open the 2-D route."""
        cfg = model_pillar()
        with EnvPatch(RCWA_BACKEND="cpp", RCWA_2D_BACKEND=None):
            self.assertEqual(common.eligibility(cfg), (False, common.NOTE_2D_PYTHON))
        with EnvPatch(RCWA_BACKEND="cpp", RCWA_2D_BACKEND="cpp"):
            self.assertEqual(common.eligibility(cfg), (True, ""))
        with EnvPatch(RCWA_BACKEND="python", RCWA_2D_BACKEND="cpp"):                # python wins over the 2-D switch ...
            for module in (common, backend):
                attempt = module.attempt(cfg, self.tmp)
                self.assertFalse(attempt.used)
                self.assertIn("RCWA_BACKEND=python", attempt.note)
            attempt = backend.attempt(gui_models.preset("trapezoid_analytic"), self.tmp)       # ... and for 1-D, as always
            self.assertIn("RCWA_BACKEND=python", attempt.note)

    def test_one_dimensional_routing_does_not_depend_on_the_switch(self):
        presets = [gui_models.preset(row["id"]) for row in gui_models.PRESETS]
        one_d = [cfg for cfg in presets if cfg["mode"] != "periodic2d"]
        self.assertGreater(len(one_d), 5)
        with EnvPatch(RCWA_2D_BACKEND=None, RCWA_BACKEND=None):
            baseline = [common.eligibility(cfg) for cfg in one_d]
        self.assertTrue(any(fits for fits, _ in baseline) and not all(fits for fits, _ in baseline))     # C++ for some, Python (with a reason) for others
        for value in ("python", "cpp", "auto", "banana", "", "  CPP "):
            with self.subTest(value=value), EnvPatch(RCWA_2D_BACKEND=value, RCWA_BACKEND=None):
                self.assertEqual([common.eligibility(cfg) for cfg in one_d], baseline)
        # and the 2-D presets follow the switch
        with EnvPatch(RCWA_2D_BACKEND="cpp", RCWA_BACKEND=None):
            self.assertTrue(common.eligibility(gui_models.preset("pillar_2d"))[0])
        with EnvPatch(RCWA_2D_BACKEND=None, RCWA_BACKEND=None):
            self.assertFalse(common.eligibility(gui_models.preset("pillar_2d"))[0])

    def test_the_default_runs_python_and_the_switch_runs_cpp_through_the_worker(self):
        """gui_worker.compute with RCWA_2D_BACKEND unset: no C++ program is even looked for (RCWA_FIELDS_EXE points nowhere), the result is the Python
        one with backend 'python' in metrics.json; with RCWA_2D_BACKEND=cpp the same model runs the (stand-in) C++ program and says 'cpp'."""
        cfg = model_pillar("xz", "jones")
        env = dict(RCWA_BACKEND=None, RCWA_JSON_EXE=None, RCWA_VERIFY_PYTHON=None, RCWA_CPP_TIMEOUT=None)
        folder = self.tmp / "default"
        metrics, fields = run_compute(cfg, folder, RCWA_FIELDS_EXE=self.tmp / "不存在", RCWA_2D_BACKEND=None, **env)
        block = metrics["backend"]
        self.assertEqual((block["efficiency"], block["fields"], block["phase"]), ("python", "python", "python"))
        self.assertEqual(block["note"], common.NOTE_2D_PYTHON)
        self.assertIsNone(block["cpp_seconds"])
        self.assertGreater(block["python_solve_seconds"], 0.)
        self.assertEqual([entry.name for entry in folder.iterdir() if entry.name.startswith("cpp_")], [])      # no request, no result folder
        self.assertNotIn("field_backend", fields)
        folder = self.tmp / "switched"
        metrics, fields = run_compute(cfg, folder, RCWA_FIELDS_EXE=self.program("standin"), RCWA_2D_BACKEND="cpp", **env)
        block = metrics["backend"]
        self.assertEqual((block["efficiency"], block["fields"], block["phase"], block["note"]), ("cpp", "cpp", "cpp", ""))
        self.assertEqual(str(fields["field_backend"]), "cpp")
        self.assertTrue((folder / "cpp_request.json").is_file())

    def test_the_start_up_line_says_which_route_two_d_takes(self):
        """The line the GUI prints at start-up (cpp_fields_backend.warmup, and the old cpp_backend.warmup) must not claim the 2-D C++ route while it is off."""
        standin = self.program("standin")
        for module in (backend, common):
            for value, on in ((None, False), ("python", False), ("cpp", True)):
                with self.subTest(module=module.__name__, RCWA_2D_BACKEND=value):
                    output = io.StringIO()
                    with EnvPatch(RCWA_FIELDS_EXE=standin, RCWA_JSON_EXE=standin if module is common else None, RCWA_BACKEND=None, RCWA_2D_BACKEND=value), \
                            contextlib.redirect_stdout(output):
                        module.warmup()
                    line = output.getvalue()
                    self.assertEqual(line.count("\n"), 1, line)
                    self.assertEqual("与二维周期" in line, on, line)
                    self.assertEqual("RCWA_2D_BACKEND=cpp" in line, not on, line)

    def test_the_gui_server_follows_the_switch(self):
        """The GUI start path: gui.py passes its environment on to the worker (the start scripts add nothing), so RCWA_2D_BACKEND set before
        启动GUI.command / start_gui.bat reaches the solver. Same 2-D job through two servers."""
        cfg = model_pillar("yz", "p")
        standin = self.program("standin")
        for value, expected in ((None, "python"), ("cpp", "cpp")):
            with self.subTest(RCWA_2D_BACKEND=value):
                server = Server(RCWA_FIELDS_EXE=standin, RCWA_BACKEND="", RCWA_JSON_EXE="", RCWA_VERIFY_PYTHON="", RCWA_CPP_TIMEOUT="",
                                RCWA_2D_BACKEND="" if value is None else value)
                self.addCleanup(server.close)
                job, status = server.run(cfg)
                self.assertEqual(status["status"], "done", status.get("error"))
                block = status["metrics"]["backend"]
                self.assertEqual((block["efficiency"], block["fields"], block["phase"]), (expected,) * 3, block["note"])
                self.assertEqual(block["note"], common.NOTE_2D_PYTHON if value is None else "")


    def test_the_model_choice_wins_over_the_switch(self):
        """The GUI's 二维求解 option is written into the model as backend_2d; when present it decides the 2-D route whatever RCWA_2D_BACKEND
        says (the variable only sets the default, i.e. the option's initial value). Without it the variable decides, as before."""
        for value in (None, "python", "cpp", "banana"):
            with self.subTest(RCWA_2D_BACKEND=value), EnvPatch(RCWA_2D_BACKEND=value, RCWA_BACKEND=None):
                cfg = model_pillar()
                cfg["backend_2d"] = "cpp"
                self.assertEqual(common.backend_2d_for(cfg), ("cpp", "", True))
                self.assertEqual(common.eligibility(cfg), (True, ""))
                cfg["backend_2d"] = "python"
                self.assertEqual(common.backend_2d_for(cfg), ("python", "", True))
                self.assertEqual(common.eligibility(cfg), (False, common.NOTE_2D_PYTHON_CHOSEN))
                del cfg["backend_2d"]
                self.assertEqual(common.backend_2d_for(cfg), common.backend_2d() + (False,))
        for alarm in ("失败", "错误", "不可用", "警告", "注意", "无法"):           # a neutral sentence, like NOTE_2D_PYTHON
            self.assertNotIn(alarm, common.NOTE_2D_PYTHON_CHOSEN)
        self.assertIn("二维求解", common.NOTE_2D_PYTHON)                            # the default note names the option ...
        self.assertIn("RCWA_2D_BACKEND=cpp", common.NOTE_2D_PYTHON)                 # ... and the variable
        with EnvPatch(RCWA_BACKEND="python", RCWA_2D_BACKEND=None):                # RCWA_BACKEND=python still wins over the option
            cfg = model_pillar()
            cfg["backend_2d"] = "cpp"
            attempt = backend.attempt(cfg, self.tmp)
            self.assertFalse(attempt.used)
            self.assertIn("RCWA_BACKEND=python", attempt.note)
        presets = [gui_models.preset(row["id"]) for row in gui_models.PRESETS]
        one_d = [cfg for cfg in presets if cfg["mode"] != "periodic2d"]
        with EnvPatch(RCWA_2D_BACKEND=None, RCWA_BACKEND=None):                    # 1-D models ignore the key
            baseline = [common.eligibility(cfg) for cfg in one_d]
            for choice in ("python", "cpp"):
                marked = [dict(cfg, backend_2d=choice) for cfg in one_d]
                self.assertEqual([common.eligibility(cfg) for cfg in marked], baseline)

    def test_normalize_keeps_a_valid_choice_and_rejects_others(self):
        base = model_pillar()
        self.assertNotIn("backend_2d", base)                                       # absent stays absent: the variable decides at run time
        for raw, expected in (("cpp", "cpp"), ("CPP", "cpp"), (" Python ", "python"), ("python", "python")):
            with self.subTest(raw=raw):
                cfg = copy.deepcopy(base)
                cfg["backend_2d"] = raw
                self.assertEqual(normalize_2d(cfg)["backend_2d"], expected)
                self.assertEqual(gui_models.normalize(cfg)["backend_2d"], expected)       # the GUI entry point (preview, run)
        cfg = copy.deepcopy(base)
        cfg["backend_2d"] = None
        self.assertNotIn("backend_2d", normalize_2d(cfg))
        for raw in ("auto", "gpu", "", "c++", 1, True):
            with self.subTest(raw=raw):
                cfg = copy.deepcopy(base)
                cfg["backend_2d"] = raw
                with self.assertRaises(ValueError):
                    normalize_2d(cfg)

    def test_the_model_choice_reaches_the_worker(self):
        """Through gui_worker.compute: backend_2d = cpp runs C++ with the variable unset, backend_2d = python runs Python with the variable at cpp,
        and the choice is recorded in the job's model.json."""
        env = dict(RCWA_BACKEND=None, RCWA_JSON_EXE=None, RCWA_VERIFY_PYTHON=None, RCWA_CPP_TIMEOUT=None)
        cfg = model_pillar("xz", "jones")
        cfg["backend_2d"] = "cpp"
        folder = self.tmp / "chosen_cpp"
        metrics, fields = run_compute(cfg, folder, RCWA_FIELDS_EXE=self.program("standin"), RCWA_2D_BACKEND=None, **env)
        block = metrics["backend"]
        self.assertEqual((block["efficiency"], block["fields"], block["phase"], block["note"]), ("cpp", "cpp", "cpp", ""))
        self.assertEqual(json.loads((folder / "model.json").read_text(encoding="utf-8"))["backend_2d"], "cpp")
        cfg["backend_2d"] = "python"
        folder = self.tmp / "chosen_python"
        metrics, fields = run_compute(cfg, folder, RCWA_FIELDS_EXE=self.program("standin"), RCWA_2D_BACKEND="cpp", **env)
        block = metrics["backend"]
        self.assertEqual((block["efficiency"], block["fields"], block["phase"]), ("python",) * 3)
        self.assertEqual(block["note"], common.NOTE_2D_PYTHON_CHOSEN)
        self.assertEqual(json.loads((folder / "model.json").read_text(encoding="utf-8"))["backend_2d"], "python")
        self.assertEqual([entry.name for entry in folder.iterdir() if entry.name.startswith("cpp_")], [])

    def test_the_server_reports_the_default_and_honours_the_choice(self):
        """/api/bootstrap tells the page which choice to show first (RCWA_2D_BACKEND; python when unset or not understood), and a job that
        carries backend_2d runs where it says, whatever the server's variable is."""
        standin = self.program("standin")
        cfg = model_pillar("xz", "s")
        for value, default, choice in (("", "python", "cpp"), ("cpp", "cpp", "python"), ("banana", "python", "cpp")):
            with self.subTest(RCWA_2D_BACKEND=value):
                server = Server(RCWA_FIELDS_EXE=standin, RCWA_BACKEND="", RCWA_JSON_EXE="", RCWA_VERIFY_PYTHON="", RCWA_CPP_TIMEOUT="",
                                RCWA_2D_BACKEND=value)
                self.addCleanup(server.close)
                self.assertEqual(json.loads(server.get("/api/bootstrap"))["backend_2d_default"], default)
                job, status = server.run(dict(cfg, backend_2d=choice))
                self.assertEqual(status["status"], "done", status.get("error"))
                block = status["metrics"]["backend"]
                self.assertEqual((block["efficiency"], block["fields"], block["phase"]), (choice,) * 3, block["note"])

# ===================================================================================================== 2. request
EXPECTED_OPTIONS = {"eig": "qr", "factorization": "li", "homogeneous_fast_path": True}


class RequestBuilding(Case):
    def test_far_request_of_the_preset(self):
        cfg = model_pillar("xz", "jones")
        request = common.build_request(cfg)
        self.assertEqual(set(request), {"mode", "stack", "M", "My", "points", "input", "options"})
        self.assertEqual((request["mode"], request["M"], request["My"]), ("periodic2d", 2, 2))
        self.assertEqual(request["points"], [{"wavelength": 1., "theta_deg": 20., "phi_deg": 30.}])
        self.assertEqual(request["input"], {"psi_deg": 33., "delta_deg": 50.})
        self.assertEqual(request["options"], dict(EXPECTED_OPTIONS, parallel_layers=False, threads=1))
        self.assertEqual(request["stack"], {
            "nI": 1., "nII": 1.45, "Lx": 1., "Ly": .8,
            "layers": [{"d": .3, "y_edges": [0., .2, .6, .8],
                        "x_profiles": [{"edges": [0., 1.], "n_values": [1.]},
                                       {"edges": [0., .3, .7, 1.], "n_values": [1., 1.7, 1.]},
                                       {"edges": [0., 1.], "n_values": [1.]}]},
                       {"d": .16, "n": 1.46}, {"d": .11, "n": 1.9}]})
        json.dumps(request, allow_nan=False)

    def test_polarization_is_always_psi_delta(self):
        for pol, expected in (("s", {"psi_deg": 90., "delta_deg": 0.}), ("p", {"psi_deg": 0., "delta_deg": 0.}),
                              ("TE", {"psi_deg": 90., "delta_deg": 0.}), ("TM", {"psi_deg": 0., "delta_deg": 0.}),
                              ("jones", {"psi_deg": 33., "delta_deg": 50.})):
            cfg = model_pillar("xz", pol)
            self.assertEqual(common.build_request(cfg)["input"], expected, pol)

    def test_complex_indices_and_x_only_layers_come_from_the_stack(self):
        stack = common.build_request(model_absorbing())["stack"]
        self.assertEqual((stack["nI"], stack["nII"], stack["Lx"], stack["Ly"]), (1.2, 1.45, .62, .93))
        first, second, third = stack["layers"]
        self.assertEqual(first, {"d": .12, "y_edges": [0., .93], "x_profiles": [{"edges": [0., .2, .62], "n_values": [1.9, [1.4, .02]]}]})
        self.assertEqual(second["y_edges"], [-.1, .3, .6, .83])
        self.assertEqual(second["x_profiles"][1], {"edges": [.1, .35, .72], "n_values": [1.46, [2.3, .13]]})
        self.assertEqual(third, {"d": .09, "n": [1.65, .12]})
        self.assertEqual(common.build_request(model_absorbing())["points"], [{"wavelength": .83, "theta_deg": -27., "phi_deg": -40.}])

    def test_threads_and_parallel_layers(self):
        two = model_pillar()
        two["structure"]["layers"].append(copy.deepcopy(two["structure"]["layers"][0]))      # a second structured layer
        for cfg, threads, expected in ((model_pillar(), 1, False), (model_pillar(), 8, False), (two, 1, False), (two, 8, True),
                                       (two, 4000, True), (two, 0, False)):
            options = common.build_request(cfg, threads)["options"]
            self.assertEqual(options["parallel_layers"], expected, (threads, expected))
            self.assertEqual(options["threads"], min(256, max(1, threads)))
            self.assertEqual({k: options[k] for k in EXPECTED_OPTIONS}, EXPECTED_OPTIONS)
            self.assertNotIn("linear", options)

    def test_fields_request(self):
        for plane, at in (("xz", .4), ("yz", .5)):
            cfg = model_pillar(plane)
            request = backend.build_request(cfg, self.tmp / "结果 目录", "token-1", 3)
            self.assertEqual(set(request), {"solve", "sampling", "output_dir", "request_id"})
            self.assertEqual(request["output_dir"], str((self.tmp / "结果 目录").resolve()))
            self.assertEqual(request["request_id"], "token-1")
            self.assertEqual(request["solve"]["options"]["diagnostics"], True)
            self.assertEqual(request["solve"]["options"]["threads"], 3)
            sampling = request["sampling"]
            self.assertEqual(set(sampling), {"z", "plane", "nt", "at"})           # no nx, no y: those are the 1-D keys
            self.assertEqual((sampling["plane"], sampling["nt"], sampling["at"]), (plane, 40, at))
            # the user's 20 rows, between the two hidden probe rows (always added: 0.02 wavelength in front of the stack, 0.02 behind it)
            self.assertEqual(sampling["z"][1:-1], np.linspace(-.2, .77, 20).tolist())
            self.assertEqual(sampling["z"][0], -.02)
            self.assertAlmostEqual(sampling["z"][-1], .57 + .02, places=14)
            self.assertEqual(len(sampling["z"]), 22)
            self.assertIsInstance(sampling["at"], float)
            json.dumps(request, allow_nan=False)
        one_d = backend.build_request(gui_models.preset("trapezoid_analytic"), self.tmp / "x", "t")
        self.assertEqual(set(one_d["sampling"]), {"z", "nx", "y"})              # 1-D untouched

    def test_requests_are_accepted_by_a_strict_reader_of_the_protocol(self):
        """The stand-in rejects unknown keys and 1-D leftovers: every request this module builds must pass it."""
        exe = self.program("standin")
        for name, build in WORKER_MODELS:
            with self.subTest(model=name):
                request, response = run_program(exe, build(), self.tmp / name.replace(" ", "_"))
                self.assertEqual(response["mode"], "periodic2d")
        # Sanity of the strict reader itself (so that the line above means something).
        request = backend.build_request(model_pillar(), self.tmp / "strict", "t")
        for label, edit, where in (("nx", lambda r: r["sampling"].update(nx=40), "sampling.nx"),
                                   ("y", lambda r: r["sampling"].update(y=0.), "sampling.y"),
                                   ("linear", lambda r: r["solve"]["options"].update(linear="lu"), "options.linear"),
                                   ("period", lambda r: r["solve"].update(period=1.), None),
                                   ("name", lambda r: r["solve"]["stack"]["layers"][1].update(name="x"), "stack.layers[1].name"),
                                   ("no input", lambda r: r["solve"].pop("input"), None)):
            bad = copy.deepcopy(request)
            bad["output_dir"] = str(self.tmp / ("strict " + label))
            edit(bad)
            code, out, _ = common._run_child([str(exe)], 60, text=json.dumps(bad))
            reply = json.loads(out)
            self.assertEqual((code, reply["ok"]), (1, False), label)
            if where:
                self.assertEqual(reply["where"], where, label)
            self.assertFalse((self.tmp / ("strict " + label)).exists())


def spec_index(value):
    return value if not isinstance(value, list) else f"{value[0]!r}+{value[1]!r}j"


def spec_cfg(spec, plane="xz"):
    """The GUI model of a frozen case (the GUI cannot express: no layer, complex exit medium, Laurent factorisation)."""
    st, layers = spec["stack"], []
    for j, layer in enumerate(st["layers"]):
        row = {"name": f"层{j + 1}", "thickness": layer["d"]}
        if "n" in layer:
            row.update(kind="uniform", n=spec_index(layer["n"]))
        else:
            row.update(kind="patterned2d", y_edges=list(layer["y_edges"]),
                       x_profiles=[{"edges": list(p["edges"]), "n_values": [spec_index(v) for v in p["n_values"]]}
                                   for p in layer["x_profiles"]])
        layers.append(row)
    if spec["psi_deg"] is None:
        pol, psi, delta = ("s" if spec["jones"][1] == [0., 0.] else "p"), 90., 0.
    else:
        pol, psi, delta = "jones", spec["psi_deg"], spec["delta_deg"]
    at = spec["slice"]["xz_fixed_y"] if plane == "xz" else spec["slice"]["yz_fixed_x"]
    return normalize_2d({
        "schema_version": 1, "name": spec["id"], "mode": "periodic2d",
        "incident": {"wavelength": spec["wavelength"], "angle_deg": spec["theta_deg"], "phi_deg": spec["phi_deg"],
                     "polarization": pol, "psi_deg": psi, "delta_deg": delta, "M": spec["M1"], "My": spec["M2"]},
        "structure": {"period": st["Lx"], "period_y": st["Ly"], "n_incident": st["nI"], "n_substrate": st["nII"], "layers": layers},
        "sampling": {"nx": 40, "nz": 20, "z_min": -.2, "z_max": 1., "slice_plane": plane, "slice_at": at}})


def expressible(spec):
    st = spec["stack"]
    return bool(st["layers"]) and not isinstance(st["nII"], list) and spec["factorization"] == "li"


class FrozenRequests(Case):
    def test_every_frozen_input_becomes_the_request_the_protocol_describes(self):
        checked = []
        for spec in frozen_cases.case_specs():
            if not expressible(spec):
                continue
            with self.subTest(case=spec["id"]):
                cfg = spec_cfg(spec)
                request = common.build_request(cfg, 4)
                st = spec["stack"]
                self.assertEqual(set(request), {"mode", "stack", "M", "My", "points", "input", "options"})
                self.assertEqual((request["M"], request["My"]), (spec["M1"], spec["M2"]))
                self.assertEqual(request["points"], [{"wavelength": spec["wavelength"], "theta_deg": spec["theta_deg"],
                                                      "phi_deg": spec["phi_deg"]}])
                if spec["psi_deg"] is None:
                    self.assertEqual(request["input"], {"psi_deg": 90. if spec["jones"][1] == [0., 0.] else 0., "delta_deg": 0.})
                else:
                    self.assertEqual(request["input"], {"psi_deg": spec["psi_deg"], "delta_deg": spec["delta_deg"]})
                self.assertEqual({k: request["stack"][k] for k in ("nI", "nII", "Lx", "Ly")},
                                 {k: st[k] for k in ("nI", "nII", "Lx", "Ly")})
                self.assertEqual(len(request["stack"]["layers"]), len(st["layers"]))
                for got, want in zip(request["stack"]["layers"], st["layers"]):
                    if "n" in want:
                        self.assertEqual(got, {"d": want["d"], "n": want["n"]})
                    else:
                        self.assertEqual(set(got), {"d", "y_edges", "x_profiles"})
                        self.assertEqual((got["d"], got["y_edges"]), (want["d"], want["y_edges"]))
                        self.assertEqual(got["x_profiles"], want["x_profiles"])
                self.assertEqual(request["options"]["parallel_layers"],
                                 sum("y_edges" in layer for layer in st["layers"]) >= 2)
                checked.append(spec["id"])
        self.assertGreaterEqual(len(checked), 18, checked)

    def test_frozen_cases_not_expressible_in_the_gui_are_exactly_the_documented_ones(self):
        skipped = {spec["id"] for spec in frozen_cases.case_specs() if not expressible(spec)}
        self.assertEqual(skipped, {"interface", "film_abs", "laurent_pillar"})


# ===================================================================================================== 3. validation of replies
FILE_KEYS = ("fields", "phase", "phase_mask", "t", "x", "y", "z")
FILE_NAMES = {"fields": "fields.npy", "phase": "phase.npy", "phase_mask": "phase_mask.npy", "t": "t.npy",
              "x": "x.npy", "y": "y.npy", "z": "z.npy"}


def edit_npy(directory, key, change, dtype=None):
    """Load a result file, let `change` edit (or replace) the array, write it back (np.save: always a valid header)."""
    path = Path(directory) / FILE_NAMES[key]
    array = np.load(path, allow_pickle=False)
    changed = change(array)
    array = array if changed is None else changed
    np.save(path, array if dtype is None else array.astype(dtype), allow_pickle=False)


def recompute_phase(directory, far, wavelength):
    """Write phase.npy and phase_mask.npy exactly as the protocol prescribes (angle of F and of F·conj(carrier), floor 1e-6
    of each component's own peak) from the fields.npy now in `directory`: a tamperer who is careful keeps them consistent."""
    fields = np.load(directory / "fields.npy")
    x, y, z = (np.load(directory / name) for name in ("x.npy", "y.npy", "z.npy"))
    i0, k0 = len(far["Kx"]) // 2, 2 * np.pi / wavelength
    carrier = np.exp(1j * k0 * (-far["Kx"][i0] * x[None, :] - far["Ky"][i0] * y[None, :] + far["kzI"][i0][0] * z[:, None]))
    phase, mask = np.empty((2,) + fields.shape), np.empty((2,) + fields.shape, np.uint8)
    for c in range(6):
        amplitude = abs(fields[c])
        weak = amplitude <= 1e-6 * amplitude.max()
        for reference, field in ((0, fields[c]), (1, fields[c] * np.conj(carrier))):
            angle = np.angle(field)
            angle[weak] = np.nan
            phase[reference, c], mask[reference, c] = angle, weak
    np.save(directory / "phase.npy", phase, allow_pickle=False)
    np.save(directory / "phase_mask.npy", mask, allow_pickle=False)


class TamperTests:
    """Corrupt one item of a valid reply (JSON or file) at a time: the GUI must refuse it (BackendError -> Python).

    A valid reply is produced once per class by PROGRAM; every case works on a private copy of the result directory.
    """
    PROGRAM = "standin"

    @classmethod
    def setUpClass(cls):
        cls.exe = Case.program(cls.PROGRAM)
        cls.tmp = Path(tempfile.mkdtemp(prefix="rcwa 2d tamper "))
        cls.counter = 0
        cls.bundles = {}
        # "other": an earlier, different request (stale files come from here); "square ...": Lx = Ly (plane masquerade)
        # "inside / above / below": the z range has no row outside the stack, only above it, only below it (hidden probe rows);
        # "dense": many rows outside the stack (every one of them is checked); "uniform": no structured layer at all (n_eig = 0)
        for label, cfg in (("xz", model_pillar("xz", "jones")), ("yz", model_pillar("yz", "jones")),
                           ("masked", model_pillar("xz", "jones")), ("other", model_pillar("xz", "jones", wavelength=1.1)),
                           ("square xz", model_square("xz")), ("square yz", model_square("yz")),
                           ("inside", model_rows(.05, .5)), ("above", model_rows(.7, 1.6)), ("below", model_rows(-1., -.1)),
                           ("dense", model_rows(-1.2, 1.8, nz=160)), ("uniform", model_uniform())):
            request, response = run_program(cls.exe, cfg, cls.tmp / ("base " + label), token="tamper-" + label)
            cls.bundles[label] = (cfg, request, response, Path(request["output_dir"]))
        cls.force_weak_points(cls.bundles["masked"][3])

    @staticmethod
    def force_weak_points(directory):
        """Make a valid reply with masked phase: a few field points are exactly 0 (phase NaN, mask 1 at both references).
        The mask rule is relative to each component's own peak, so only genuinely vanishing points are masked."""
        fields, phase, mask = (np.load(directory / name) for name in ("fields.npy", "phase.npy", "phase_mask.npy"))
        # 行取在叠层之内（z = -.2 … .77，叠层 0 … .57 对应用户的第 4 … 15 行，含最前面的探针行是原始数组的第 5 … 16 行）：叠层之外的行会被拿去与远场应答重建的场对照，
        # 手工改过的场点在那里是不自洽的，会被拒绝——这里要的是“弱场被正确掩膜”的合法应答。
        for component, iz, points in ((0, 6, slice(5, 9)), (3, 8, slice(0, fields.shape[2]))):
            fields[component, iz, points] = 0
            phase[:, component, iz, points] = np.nan
            mask[:, component, iz, points] = 1
        for name, array in (("fields.npy", fields), ("phase.npy", phase), ("phase_mask.npy", mask)):
            np.save(directory / name, array, allow_pickle=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # -- machinery
    def prepare(self, plane="xz"):
        """A private copy of one valid reply: 'xz' / 'yz' (pillar, all six components nonzero) or 'masked' (y-invariant)."""
        cfg, request, response, directory = self.bundles[plane]
        type(self).counter += 1
        work = self.tmp / f"work{self.counter}"
        shutil.copytree(directory, work)
        request, response = copy.deepcopy(request), copy.deepcopy(response)
        request["output_dir"] = response["field_data"]["directory"] = str(work)
        return copy.deepcopy(cfg), request, response, work

    def accepted(self, cfg, request, response, wall_seconds=None, notes=None):
        values = common.extract(response, cfg)
        fields, manifest = backend.extract_fields(response, cfg, request, wall_seconds, notes)
        return values, fields, manifest

    def refuse(self, label, edit=None, files=None, plane="xz", cfg_edit=None, words=None, wall=None):
        cfg, request, response, work = self.prepare(plane)
        if files:
            files(work)
        if edit:
            edit(response, response["results"][0], response["field_data"])
        if cfg_edit:
            cfg_edit(cfg)
        with self.assertRaises(BackendError, msg=label) as caught:
            self.accepted(cfg, request, response, wall)
        if words:
            self.assertRegex(str(caught.exception), words, label)
        if os.environ.get("RCWA_TEST_VERBOSE"):
            say(f"      refused [{label}]: {caught.exception}")

    # -- the untouched reply is taken
    def test_pristine_reply_is_accepted_and_complete(self):
        for plane in ("xz", "yz", "masked"):
            cfg, request, response, work = self.prepare(plane)
            values, fields, manifest = self.accepted(cfg, request, response)
            nz, nt = cfg["sampling"]["nz"], cfg["sampling"]["nx"]
            plane = cfg["sampling"]["slice_plane"]
            self.assertEqual(len(values["DEr"]), 25)
            self.assertEqual(values["mn"][:2], [[-2, -2], [-2, -1]])
            for key in COMPONENTS:
                self.assertEqual((fields[key].dtype, fields[key].shape), (np.dtype("complex128"), (nz, nt)))
            self.assertEqual(fields["phase_values"].shape, (2, 6, nz, nt))
            self.assertEqual(fields["phase_masks"].dtype, np.dtype(bool))
            self.assertEqual(str(fields["slice_plane"]), plane)
            self.assertEqual(float(fields["slice_at"]), cfg["sampling"]["slice_at"])
            t = np.arange(nt) * (cfg["structure"]["period"] if plane == "xz" else cfg["structure"]["period_y"]) / nt
            np.testing.assert_allclose(fields["t"], t, rtol=1e-14, atol=1e-14)
            fixed = np.full(nt, cfg["sampling"]["slice_at"])
            np.testing.assert_array_equal(fields["y" if plane == "xz" else "x"], fixed)
            np.testing.assert_array_equal(fields["x" if plane == "xz" else "y"], fields["t"])
            np.testing.assert_array_equal(fields["z"], request["sampling"]["z"][1:-1])      # the raw request holds the two probe rows
            self.assertEqual(set(manifest), set(backend.MANIFEST_KEYS_2D))
            self.assertNotIn("y", manifest)
            self.assertEqual(manifest["slice"]["plane"], plane)
        masks = self.accepted(*self.prepare("masked")[:3])[1]["phase_masks"]      # weak points are accepted when masked as the rule says
        # force_weak_points edits raw rows 6 and 8 (inside the stack); the user's rows are the raw ones minus the leading probe row: 5 and 7
        self.assertTrue(masks[:, 0, 5, 5:9].all() and masks[:, 3, 7].all() and masks.sum() == 2 * (4 + masks.shape[-1]))

    # -- far-field part of the reply
    def test_far_field_items(self):
        k_prop = lambda far: far["prop_r"].index(1)
        k_evan = lambda far: far["prop_r"].index(0)

        def setitem(key, index, value):
            return lambda r, far, fd: far[key].__setitem__(index, value)

        def scale(key, index, factor):
            return lambda r, far, fd: far[key].__setitem__(index, [far[key][index][0] * factor, far[key][index][1] * factor])

        edits = {
            "version": lambda r, far, fd: r.update(version=2), "version_bool": lambda r, far, fd: r.update(version=True),
            "version_missing": lambda r, far, fd: r.pop("version"),
            "mode": lambda r, far, fd: r.update(mode="planar"), "mode_missing": lambda r, far, fd: r.pop("mode"),
            "M": lambda r, far, fd: r.update(M=3), "M_float": lambda r, far, fd: r.update(M=2.0),
            "My": lambda r, far, fd: r.update(My=1), "My_missing": lambda r, far, fd: r.pop("My"),
            "orders_swapped": lambda r, far, fd: r["orders"].insert(0, r["orders"].pop(1)),
            "orders_short": lambda r, far, fd: r["orders"].pop(),
            "orders_float": lambda r, far, fd: r["orders"].__setitem__(0, [-2.0, -2]),
            "orders_wrong": lambda r, far, fd: r["orders"].__setitem__(5, [9, 9]),
            "orders_as_1d": lambda r, far, fd: r.update(orders=list(range(-2, 3))),
            "results_empty": lambda r, far, fd: r.update(results=[]),
            "results_two": lambda r, far, fd: r["results"].append(copy.deepcopy(far)),
            "point_wavelength": lambda r, far, fd: far.update(wavelength=far["wavelength"] + 1e-9),
            "point_theta": lambda r, far, fd: far.update(theta_deg=far["theta_deg"] + 1e-9),
            "point_phi": lambda r, far, fd: far.update(phi_deg=far["phi_deg"] + 1e-9),
            "phi_missing": lambda r, far, fd: far.pop("phi_deg"),
            "status_rayleigh": lambda r, far, fd: far.update(status="rayleigh_anomaly", message="入射区 (1, 0) 级掠射"),
            "status_numerical": lambda r, far, fd: far.update(status="numerical_failure", message="本征迭代未收敛"),
            "status_missing": lambda r, far, fd: far.pop("status"),
            "status_unknown": lambda r, far, fd: far.update(status="fine"),
            "null_in_DEr": setitem("DEr", 3, None), "null_in_Kx": setitem("Kx", 3, None),
            "null_in_Rs": setitem("Rs", 4, [None, None]), "nan_in_Ex_t": setitem("Ex_t", 4, [float("nan"), 0.]),
            "inf_in_kzII": setitem("kzII", 4, [float("inf"), 0.]),
            "DEr_short": lambda r, far, fd: far["DEr"].pop(), "Rs_long": lambda r, far, fd: far["Rs"].append([0., 0.]),
            "kzI_not_a_pair": setitem("kzI", 2, [1.0]), "Tp_scalar": setitem("Tp", 2, 1.0),
            "Ey_r_string": setitem("Ey_r", 2, ["a", "b"]), "Kx_bool": setitem("Kx", 2, True),
            "Kx_changed": lambda r, far, fd: far["Kx"].__setitem__(7, far["Kx"][7] + 1e-6),
            "Ky_changed": lambda r, far, fd: far["Ky"].__setitem__(7, far["Ky"][7] - 1e-6),
            "Kx_missing": lambda r, far, fd: far.pop("Kx"),
            "kzI_changed": lambda r, far, fd: far["kzI"].__setitem__(k_prop(far), [far["kzI"][k_prop(far)][0] + 1e-3, 0.]),
            "kzII_changed": lambda r, far, fd: far["kzII"].__setitem__(3, [far["kzII"][3][0] * 1.001, far["kzII"][3][1] * 1.001]),
            "kzI_swapped_with_kzII": lambda r, far, fd: far.update(kzI=far["kzII"], kzII=far["kzI"]),
            "prop_flipped": lambda r, far, fd: far["prop_r"].__setitem__(0, 1 - far["prop_r"][0]),
            "prop_t_flipped": lambda r, far, fd: far["prop_t"].__setitem__(12, 1 - far["prop_t"][12]),
            "prop_float": lambda r, far, fd: far["prop_r"].__setitem__(0, 0.0),
            "prop_bool": lambda r, far, fd: far["prop_r"].__setitem__(0, False),
            "prop_two": lambda r, far, fd: far["prop_r"].__setitem__(0, 2),
            "prop_short": lambda r, far, fd: far["prop_t"].pop(),
            "DE_negative": lambda r, far, fd: far["DEr"].__setitem__(k_prop(far), -1e-3),
            "DE_evanescent_nonzero": lambda r, far, fd: far["DEr"].__setitem__(k_evan(far), 1e-3),
            "DE_scaled": lambda r, far, fd: far["DEt"].__setitem__(12, far["DEt"][12] * 1.01 + 1e-6),
            "DE_swapped": lambda r, far, fd: far.update(DEr=far["DEt"], DEt=far["DEr"]),
            "amplitude_scaled": scale("Ex_r", 12, 1.01), "amplitude_Ez_scaled": scale("Ez_t", 12, 1.01),
            "arrays_missing_Tp": lambda r, far, fd: far.pop("Tp"),
            "eig_info": lambda r, far, fd: far.update(eig_info=3), "eig_info_missing": lambda r, far, fd: far.pop("eig_info"),
            "eig_info_bool": lambda r, far, fd: far.update(eig_info=False),
            "n_eig_negative": lambda r, far, fd: far.update(n_eig=-1), "n_dense_solve_string": lambda r, far, fd: far.update(n_dense_solve="2"),
            "residual_negative": lambda r, far, fd: far.update(max_eig_residual=-1e-16),
            "residual_nan": lambda r, far, fd: far.update(max_eig_residual=float("nan")),
            "backward_error_bool": lambda r, far, fd: far.update(max_solve_backward_error=True),
            "timing_without_operators": lambda r, far, fd: far["timing"].pop("operators"),
            "timing_is_1d_set": lambda r, far, fd: far.update(timing={"total": 0., "eig": 0., "linear": 0., "product": 0.}),
            "timing_extra": lambda r, far, fd: far["timing"].update(extra=0.),
            "timing_negative": lambda r, far, fd: far["timing"].update(eig=-1.),
            "timing_string": lambda r, far, fd: far["timing"].update(eig="x"), "timing_list": lambda r, far, fd: far.update(timing=[1, 2]),
            "threads_zero": lambda r, far, fd: r.update(threads=0), "threads_big": lambda r, far, fd: r.update(threads=257),
            "threads_missing": lambda r, far, fd: r.pop("threads"), "threads_bool": lambda r, far, fd: r.update(threads=True),
            "threads_float": lambda r, far, fd: r.update(threads=1.0),
        }
        for label, edit in edits.items():
            with self.subTest(case=label):
                self.refuse(label, edit)

    # -- manifest
    def test_manifest_items(self):
        cfg = self.bundles["xz"][0]
        nz, nt = cfg["sampling"]["nz"], cfg["sampling"]["nx"]
        at, period = cfg["sampling"]["slice_at"], cfg["structure"]["period"]

        def fd_set(path, value):
            def edit(r, far, fd):
                target = fd
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
            return edit

        def fd_del(*path):
            def edit(r, far, fd):
                target = fd
                for key in path[:-1]:
                    target = target[key]
                del target[path[-1]]
            return edit

        edits = {
            "protocol": lambda r, far, fd: r.update(protocol="stablercwa-1d"),
            "protocol_missing": lambda r, far, fd: r.pop("protocol"),
            "field_data_missing": lambda r, far, fd: r.pop("field_data"),
            "field_data_not_object": lambda r, far, fd: r.update(field_data=[1]),
            "version": fd_set(("version",), 2), "version_float": fd_set(("version",), 1.0), "version_bool": fd_set(("version",), True),
            "top_level_y": fd_set(("y",), 0.0), "top_level_extra": fd_set(("extra",), 1),
            "producer": fd_set(("producer",), "somebody.else"), "request_id": fd_set(("request_id",), "stale"),
            "directory": fd_set(("directory",), "/elsewhere/result"), "layout": fd_set(("layout",), "F"),
            "shape_swapped": fd_set(("shape",), [nt, nz]), "shape_1d": fd_set(("shape",), [nz]),
            "shape_float": fd_set(("shape",), [float(nz), nt]), "shape_one_off": fd_set(("shape",), [nz, nt + 1]),
            "components_order": fd_set(("components",), list(reversed(COMPONENTS))),
            "components_short": fd_set(("components",), list(COMPONENTS[:5])),
            "slice_missing": fd_del("slice"), "slice_plane": fd_set(("slice", "plane"), "yz"),
            "slice_at": fd_set(("slice", "at"), at + 1e-9), "slice_at_string": fd_set(("slice", "at"), str(at)),
            "slice_period": fd_set(("slice", "period"), period * 1.001), "slice_period_missing": fd_del("slice", "period"),
            "slice_extra": fd_set(("slice", "unit"), "um"), "slice_bool_at": fd_set(("slice", "at"), True),
            "harmonics_missing": fd_del("harmonics"), "harmonics_M": fd_set(("harmonics", "M"), 3),
            "harmonics_My": fd_set(("harmonics", "My"), 1), "harmonics_N": fd_set(("harmonics", "N"), 24),
            "harmonics_extra": fd_set(("harmonics", "Mz"), 0), "harmonics_float": fd_set(("harmonics", "M"), 2.0),
            "phase_floor": fd_set(("phase", "floor"), 1e-5), "phase_unwrap": fd_set(("phase", "unwrap"), "unwrap"),
            "phase_references": fd_set(("phase", "references"), ["incident", "absolute"]),
            "phase_references_short": fd_set(("phase", "references"), ["absolute"]),
            "phase_unit": fd_set(("phase", "unit"), "deg"), "phase_mask_true": fd_set(("phase", "mask_true"), "valid"),
            "phase_extra": fd_set(("phase", "carrier"), "x"), "phase_missing": fd_del("phase"),
            "norm_E": fd_set(("normalization", "E"), 2.0), "norm_H": lambda r, far, fd: fd["normalization"].update(H=fd["normalization"]["H"] * (1 + 1e-9)),
            "norm_convention": fd_set(("normalization", "H_convention"), "H_physical"),
            "norm_extra": fd_set(("normalization", "extra"), 1), "norm_E_zero": fd_set(("normalization", "E"), 0.0),
            "norm_E_bool": fd_set(("normalization", "E"), True), "norm_E_string": fd_set(("normalization", "E"), "1"),
            "norm_missing": fd_del("normalization"),
            "files_extra": fd_set(("files", "extra"), {"name": "extra.npy", "dtype": "<f8", "shape": [1]}),
            "files_not_object": fd_set(("files",), [1]),
            "layers_empty": fd_set(("layers",), []), "layers_not_list": fd_set(("layers",), {}),
            "layers_extra": lambda r, far, fd: fd["layers"].append(copy.deepcopy(fd["layers"][-1])),
            "layer_original_index": fd_set(("layers", 1, "original_index"), 5),
            "layer_effective_index": fd_set(("layers", 1, "effective_index"), -1),
            "layer_effective_shifted": fd_set(("layers", 2, "effective_index"), 1),
            "layer_z_top": fd_set(("layers", 1, "z_top"), 0.3 + 1e-12), "layer_z_bottom": fd_set(("layers", 2, "z_bottom"), 0.58),
            "layer_zero_modes": fd_set(("layers", 0, "finite_zero_modes"), 1),
            "layer_zero_modes_bool": fd_set(("layers", 0, "finite_zero_modes"), False),
            "layer_schur": fd_set(("layers", 0, "schur"), True), "layer_schur_int": fd_set(("layers", 0, "schur"), 0),
            "layer_extra_key": fd_set(("layers", 0, "propagation"), "direct"), "layer_key_missing": fd_del("layers", 0, "schur"),
            "layer_not_object": fd_set(("layers", 0), [1]),
            "timing_missing": fd_del("timing", "write"), "timing_extra": fd_set(("timing", "extra"), 0.),
            "timing_string": fd_set(("timing", "solve"), "bad"), "timing_negative": fd_set(("timing", "sample"), -1.),
            "timing_nan": fd_set(("timing", "phase"), float("nan")), "timing_bool": fd_set(("timing", "write"), True),
            "timing_missing_all": fd_del("timing"),
        }
        for key in FILE_KEYS:
            name, dtype, shape = FILE_NAMES[key], {"fields": "<c16", "phase": "<f8", "phase_mask": "|u1"}.get(key, "<f8"), None
            edits[f"files_{key}_missing"] = fd_del("files", key)
            edits[f"files_{key}_name"] = fd_set(("files", key, "name"), "other.npy")
            edits[f"files_{key}_path_escape"] = fd_set(("files", key, "name"), "../" + name)
            edits[f"files_{key}_dtype"] = fd_set(("files", key, "dtype"), ">f8" if dtype != ">f8" else "<f8")
            edits[f"files_{key}_shape"] = lambda r, far, fd, key=key: fd["files"][key].update(shape=fd["files"][key]["shape"][1:])
            edits[f"files_{key}_extra_key"] = fd_set(("files", key, "order"), "C")
        for label, edit in edits.items():
            with self.subTest(case=label):
                self.refuse(label, edit)

    # -- files
    def test_file_headers_and_sizes(self):
        import numpy.lib.format as npy_format

        def rewrite(key, version):
            def change(directory):
                path = Path(directory) / FILE_NAMES[key]
                array = np.load(path, allow_pickle=False)
                with path.open("wb") as stream:
                    npy_format.write_array(stream, array, version=version, allow_pickle=False)
            return change

        def mutate_bytes(key, function):
            def change(directory):
                path = Path(directory) / FILE_NAMES[key]
                path.write_bytes(function(path.read_bytes()))
            return change

        def symlink(key):
            def change(directory):
                path = Path(directory) / FILE_NAMES[key]
                target = path.with_name("real_" + path.name)
                path.rename(target)
                path.symlink_to(target)
            return change

        def directory_instead(key):
            def change(directory):
                path = Path(directory) / FILE_NAMES[key]
                path.unlink()
                path.mkdir()
            return change

        other_dtype = {"fields": "<c8", "phase": "<f4", "phase_mask": "|i1", "t": "<f4", "x": "<f4", "y": "<f4", "z": "<f4"}
        for key in FILE_KEYS:
            name = FILE_NAMES[key]
            cases = {
                "missing": lambda d, name=name: (Path(d) / name).unlink(),
                "directory": directory_instead(key),
                "truncated": mutate_bytes(key, lambda b: b[:-1]), "extra_byte": mutate_bytes(key, lambda b: b + b"\0"),
                "empty": mutate_bytes(key, lambda b: b""), "magic": mutate_bytes(key, lambda b: b"\x00" + b[1:]),
                "header_only": mutate_bytes(key, lambda b: b[:b.index(b"\n") + 1]),
                "version_2_0": rewrite(key, (2, 0)),
                "dtype": lambda d, key=key: edit_npy(d, key, lambda a: a.astype(other_dtype[key])),
                "wrong_shape": lambda d, key=key: edit_npy(d, key, lambda a: a.reshape(-1)[:-1]),
                "transposed_layout": lambda d, key=key: edit_npy(d, key, lambda a: np.swapaxes(a, -1, -2).copy() if a.ndim > 1 else a[::-1].copy()),
            }
            if key == "y":                      # y is constant on an xz section: reversing it changes nothing
                del cases["transposed_layout"]
            if key not in ("phase_mask",):
                cases["big_endian"] = lambda d, key=key: edit_npy(d, key, lambda a: a.astype(a.dtype.newbyteorder(">")))
            if key not in ("t", "x", "y", "z"):
                cases["fortran"] = lambda d, key=key: np.save(Path(d) / FILE_NAMES[key],
                                                             np.asfortranarray(np.load(Path(d) / FILE_NAMES[key])), allow_pickle=False)
            if os.name != "nt":
                cases["symlink"] = symlink(key)
            for label, change in cases.items():
                with self.subTest(file=key, case=label):
                    self.refuse(f"{key} {label}", files=change)

    def test_file_values(self):
        def put(key, index, value):
            return lambda d: edit_npy(d, key, lambda a: a.__setitem__(index, value))

        def first_unmasked(d, reference=0, component=1):         # Ey is nonzero in both models
            mask = np.load(Path(d) / "phase_mask.npy")[reference, component]
            return (reference, component) + tuple(np.argwhere(mask == 0)[0])

        def first_masked(d):                                      # needs the y-invariant model (Ex, Ez, Hy vanish)
            mask = np.load(Path(d) / "phase_mask.npy")
            return tuple(np.argwhere(mask == 1)[0])

        def set_phase_unmasked(value):
            return lambda d: edit_npy(d, "phase", lambda a: a.__setitem__(first_unmasked(d), value))

        def flip_mask(both):
            def change(d):
                where = first_unmasked(d, 0, 1)

                def edit(a):
                    a[where] = 1
                    if both:
                        a[(1,) + where[1:]] = 1
                edit_npy(d, "phase_mask", edit)
            return change

        def spike_field():
            return lambda d: edit_npy(d, "fields", lambda a: a.__setitem__((0, 3, 3), 1e6))

        cases = {
            "fields_nan": put("fields", (0, 0, 0), np.nan), "fields_inf": put("fields", (3, 1, 1), complex(np.inf, 0)),
            "fields_mask_rule_broken_by_spike": spike_field(),
            "phase_nan_unmasked": set_phase_unmasked(np.nan), "phase_inf_unmasked": set_phase_unmasked(np.inf),
            "phase_out_of_range": set_phase_unmasked(3.5),
            "phase_masked_not_nan": lambda d: edit_npy(d, "phase", lambda a: a.__setitem__(first_masked(d), 0.0)),
            "mask_two": lambda d: edit_npy(d, "phase_mask", lambda a: a.__setitem__(first_masked(d), 2)),
            "mask_flipped_both_references": flip_mask(True), "mask_flipped_one_reference": flip_mask(False),
            "mask_all_set": lambda d: edit_npy(d, "phase_mask", lambda a: a.fill(1)),
            "t_nan": put("t", 2, np.nan), "t_shifted": put("t", 3, 1e-9 + np.load(Path(self.bundles["xz"][3]) / "t.npy")[3]),
            "t_reversed": lambda d: edit_npy(d, "t", lambda a: a[::-1].copy()),
            "x_not_t": put("x", 3, 1e-9 + np.load(Path(self.bundles["xz"][3]) / "x.npy")[3]),
            "x_inf": put("x", 0, np.inf),
            "y_not_fixed": put("y", 2, self.bundles["xz"][0]["sampling"]["slice_at"] + 1e-12),
            "y_is_t": lambda d: edit_npy(d, "y", lambda a: np.load(Path(d) / "t.npy")),
            "z_reordered": lambda d: edit_npy(d, "z", lambda a: a[::-1].copy()),
            "z_one_ulp": lambda d: edit_npy(d, "z", lambda a: a.__setitem__(4, np.nextafter(a[4], 1.0))),
            "z_nan": put("z", 0, np.nan),
        }
        masked_only = {"phase_masked_not_nan", "mask_two", "mask_flipped_both_references", "mask_flipped_one_reference"}
        for label, change in cases.items():
            with self.subTest(case=label):
                self.refuse(label, files=change, plane="masked" if label in masked_only else "xz")

    def test_yz_section_coordinates(self):
        xz_x, xz_y = None, None

        def swap_xy(d):
            x, y = np.load(Path(d) / "x.npy"), np.load(Path(d) / "y.npy")
            np.save(Path(d) / "x.npy", y)
            np.save(Path(d) / "y.npy", x)

        for label, change in (("x_and_y_swapped", swap_xy),
                              ("x_not_fixed", lambda d: edit_npy(d, "x", lambda a: a.__setitem__(1, a[1] + 1e-12))),
                              ("y_not_t", lambda d: edit_npy(d, "y", lambda a: a.__setitem__(5, a[5] + 1e-12))),
                              ("t_wrong_period", lambda d: edit_npy(d, "t", lambda a: a * 1.25))):
            with self.subTest(case=label):
                self.refuse(label, files=change, plane="yz")
        self.refuse("manifest says xz for a yz request", lambda r, far, fd: fd["slice"].update(plane="xz", period=fd["slice"]["period"]), plane="yz")
        self.refuse("period of the other axis", lambda r, far, fd: fd["slice"].update(period=1.0), plane="yz")

    def test_model_and_reply_must_agree(self):
        for label, edit in (("nx", lambda c: c["sampling"].update(nx=c["sampling"]["nx"] + 1)),
                            ("nz", lambda c: c["sampling"].update(nz=c["sampling"]["nz"] + 1)),
                            ("plane", lambda c: c["sampling"].update(slice_plane="yz", slice_at=.5)),
                            ("at", lambda c: c["sampling"].update(slice_at=c["sampling"]["slice_at"] + 1e-9)),
                            ("M", lambda c: c["incident"].update(M=1)), ("My", lambda c: c["incident"].update(My=3)),
                            ("layer thickness", lambda c: c["structure"]["layers"][0].update(thickness=.31)),
                            ("layer count", lambda c: c["structure"]["layers"].pop()),
                            ("wavelength", lambda c: c["incident"].update(wavelength=1.0001)),
                            ("angle", lambda c: c["incident"].update(angle_deg=20.001)),
                            ("n_incident", lambda c: c["structure"].update(n_incident=1.1)),
                            ("period_y", lambda c: c["structure"].update(period_y=.9))):
            with self.subTest(case=label):
                self.refuse(label, cfg_edit=edit)

    def test_request_must_agree_with_reply(self):
        cfg, request, response, work = self.prepare()
        request["request_id"] = "another request"
        with self.assertRaises(BackendError):
            backend.extract_fields(response, cfg, request)
        cfg, request, response, work = self.prepare()
        request["sampling"]["z"][3] += 1e-12
        with self.assertRaises(BackendError):
            backend.extract_fields(response, cfg, request)
        cfg, request, response, work = self.prepare()
        request["solve"]["points"][0]["theta_deg"] += 1e-9
        with self.assertRaises(BackendError):
            backend.extract_fields(response, cfg, request)


    # -- plausible but wrong replies: formally perfect, numerically wrong (physical self-consistency)
    def test_plausible_but_wrong_replies_are_refused(self):
        """The reviewer's catalogue of changes that keep every key, shape, dtype and mask rule intact. Each used to be accepted
        silently (the GUI then showed wrong plots with backend = cpp and an empty note); each is now refused with its reason.
        Built from the one honest reply per bundle, in process, so the solver does not run again."""
        def far_edit(change):
            return lambda work, far, response, cfg: change(far, response)

        def field_edit(change, phase=False):
            """change(fields array [6, nz, nt] -> array); `phase` recomputes phase and mask from the new fields."""
            def edit(work, far, response, cfg):
                fields = np.load(work / "fields.npy")
                np.save(work / "fields.npy", np.ascontiguousarray(change(fields)), allow_pickle=False)
                if phase:
                    recompute_phase(work, far, cfg["incident"]["wavelength"])
            return edit

        def scale(component, factor):
            def change(fields):
                fields = fields.copy()
                fields[component] *= factor
                return fields
            return change

        def permute(order):
            return lambda fields: fields[order]

        def roll(fields):
            return np.roll(fields, 1, axis=-1)

        def replace_from(bundle, *names):
            def edit(work, far, response, cfg):
                for name in names:
                    shutil.copy(self.bundles[bundle][3] / name, work / name)
            return edit

        def roll_all(work, far, response, cfg):
            for name in ("fields.npy", "phase.npy", "phase_mask.npy"):
                np.save(work / name, np.roll(np.load(work / name), 1, axis=-1), allow_pickle=False)

        def swap_exey_all(work, far, response, cfg):
            order = [1, 0, 2, 3, 4, 5]
            for name in ("fields.npy", "phase.npy", "phase_mask.npy"):
                array = np.load(work / name)
                np.save(work / name, np.ascontiguousarray(array[order] if name == "fields.npy" else array[:, order]), allow_pickle=False)

        def phase_incident_is_absolute(work, far, response, cfg):
            phase = np.load(work / "phase.npy")
            phase[1] = phase[0]
            np.save(work / "phase.npy", phase, allow_pickle=False)

        def times(key, factor):
            return lambda far, response: far.update({key: [v * factor for v in far[key]]})

        def swap(first, second):
            return lambda far, response: far.update({first: far[second], second: far[first]})

        def conjugate_evanescent(far, response):
            far["kzI"] = [[a, -b] if b > 0 else [a, b] for a, b in far["kzI"]]

        def conjugate_Rs(far, response):
            far["Rs"] = [[a, -b] for a, b in far["Rs"]]

        def orders_transposed(far, response):
            response["orders"] = [[m, n] for n in range(-2, 3) for m in range(-2, 3)]

        def arrays_transposed(far, response):           # every array reordered as if n were the outer index
            perm = [(m + 2) * 5 + (n + 2) for n in range(-2, 3) for m in range(-2, 3)]
            for key, value in list(far.items()):
                if isinstance(value, list) and len(value) == 25:
                    far[key] = [value[j] for j in perm]

        def masquerade(work, far, response, cfg):             # fields of the other section of an Lx = Ly model, with consistent phase
            shutil.copy(self.bundles["square yz"][3] / "fields.npy", work / "fields.npy")
            recompute_phase(work, far, cfg["incident"]["wavelength"])

        far_words, de_words, near, phase_ = "Rs、Rp、Ts、Tp 或远场 Ez", "效率", "叠层之外", "相位文件与六复场不自洽"
        mixed = "叠层之外|相位"          # fields only: whichever of the mask rule, the near field or the phase check sees it first
        cases = {
            "orders transposed": (far_edit(orders_transposed), "级次表", "xz"),
            "arrays transposed": (far_edit(arrays_transposed), "Kx、Ky", "xz"),
            "swap Rs/Rp": (far_edit(swap("Rs", "Rp")), far_words, "xz"),
            "swap Ts/Tp": (far_edit(swap("Ts", "Tp")), far_words, "xz"),
            "conjugate Rs": (far_edit(conjugate_Rs), far_words, "xz"),
            "DE x (1 + 1e-9)": (far_edit(lambda far, r: (times("DEr", 1 + 1e-9)(far, r), times("DEt", 1 + 1e-9)(far, r))), de_words, "xz"),
            "DE x (1 + 1e-10)": (far_edit(lambda far, r: (times("DEr", 1 + 1e-10)(far, r), times("DEt", 1 + 1e-10)(far, r))), de_words, "xz"),
            "evanescent kz conjugated": (far_edit(conjugate_evanescent), "虚部为负", "xz"),
            "Ex x (1 + 1e-6)": (field_edit(scale(0, 1 + 1e-6)), near, "xz"),
            "Hy x (1 + 1e-3)": (field_edit(scale(4, 1 + 1e-3)), near, "xz"),
            "Ez negated, phase untouched": (field_edit(scale(2, -1)), mixed, "xz"),
            "Ez negated, phase consistent": (field_edit(scale(2, -1), phase=True), near, "xz"),
            "Ex/Ey swapped, phase untouched": (field_edit(permute([1, 0, 2, 3, 4, 5])), mixed, "xz"),
            "Ex/Ey swapped, phase consistent": (swap_exey_all, near, "xz"),
            "samples rolled along t, fields only": (field_edit(roll), mixed, "xz"),
            "samples rolled along t, with phase": (roll_all, near, "xz"),
            "H / Z0": (field_edit(lambda fields: np.concatenate([fields[:3], fields[3:] / 376.730313668])), near, "xz"),
            "conjugated fields, phase consistent": (field_edit(np.conj, phase=True), near, "xz"),
            "incident phase = absolute phase": (phase_incident_is_absolute, phase_, "xz"),
            "stale phase": (replace_from("other", "phase.npy"), "相位", "xz"),
            "stale fields": (replace_from("other", "fields.npy"), mixed, "xz"),
            "stale fields, phase and mask": (replace_from("other", "fields.npy", "phase.npy", "phase_mask.npy"), near, "xz"),
            "other section's fields (Lx = Ly)": (masquerade, near, "square xz"),
            "ok is 1, not true": (lambda work, far, response, cfg: response.update(ok=1), "ok", "xz"),
            "ok is a string": (lambda work, far, response, cfg: response.update(ok="true"), "ok", "xz"),
        }
        for label, (change, words, plane) in cases.items():
            with self.subTest(case=label):
                cfg, request, response, work = self.prepare(plane)
                change(work, response["results"][0], response, cfg)
                with self.assertRaises(BackendError, msg=label) as caught:
                    self.accepted(cfg, request, response)
                self.assertRegex(str(caught.exception), words, label)
                if os.environ.get("RCWA_TEST_VERBOSE"):
                    say(f"      refused [{label}]: {caught.exception}")

    def test_each_check_sees_its_own_kind_of_error(self):
        """The three residual functions on their own (the reply validation stops at the first one that objects, so the
        catalogue above does not show that each of them works): honest data ~1e-15, a wrong one far above the tolerance."""
        cfg, request, response, work = self.prepare("xz")
        far = backend.far_arrays_2d(response["results"][0], 25)
        six, phase = np.load(work / "fields.npy"), np.load(work / "phase.npy")
        mask = np.load(work / "phase_mask.npy").astype(bool)
        x, y, z = (np.load(work / name) for name in ("x.npy", "y.npy", "z.npy"))
        values = common.extract(response, cfg)
        st = cfg["structure"]
        inc = cfg["incident"]
        residual, rows, where = backend.near_field_residual_2d(six, x, y, z, far, request, cfg)
        self.assertLess(residual, 1e-12)
        self.assertEqual(rows, 10)          # z = -.2 ... .77 in 20 steps, stack 0 ... .57: 4 rows below it, 4 above, plus the two probe rows
        self.assertLess(backend.phase_residual_2d(six, abs(six), phase, mask, x, y, z, far, request), 1e-12)
        self.assertLess(common.far_field_residual_2d(values, float(st["n_incident"]), float(st["n_substrate"]),
                                                     math.radians(inc["phi_deg"])), 1e-12)
        # near field alone: a fields-only change outside the stack; the same change inside the stack is invisible to it
        below = np.flatnonzero(z < 0)
        changed = six.copy()
        changed[1, below[1]] *= 1 + 1e-6
        residual, rows, where = backend.near_field_residual_2d(changed, x, y, z, far, request, cfg)
        self.assertGreater(residual, 1e-8)
        self.assertIn("入射区", where)
        inside = int(np.flatnonzero((z > 0) & (z < .57))[3])
        changed = six.copy()
        changed[1, inside] *= 1 + 1e-3
        self.assertLess(backend.near_field_residual_2d(changed, x, y, z, far, request, cfg)[0], 1e-12)       # not detectable there
        self.assertEqual(backend.near_field_residual_2d(six[:, 5:17], x, z[5:17] * 1, z[5:17], far, request, cfg)[1:], (0, ""))     # rows 5..16: inside
        # phase alone: negating Ez without touching the phase files; an absolute phase passed off as the incident one
        negated = six.copy()
        negated[2] *= -1
        self.assertGreater(backend.phase_residual_2d(negated, abs(negated), phase, mask, x, y, z, far, request), .5)
        swapped = phase.copy()
        swapped[1] = swapped[0]
        self.assertGreater(backend.phase_residual_2d(six, abs(six), swapped, mask, x, y, z, far, request), .5)
        # far field alone
        broken = copy.deepcopy(values)
        broken["Tp"] = broken["Ts"]
        self.assertGreater(common.far_field_residual_2d(broken, float(st["n_incident"]), float(st["n_substrate"]),
                                                        math.radians(inc["phi_deg"])), .1)

    def test_the_checks_do_not_refuse_honest_replies_of_hard_models(self):
        """No false rejections: grazing incidence, normal incidence (the kt = 0 branch), a grazing diffraction order, strongly
        absorbing and opaque stacks, a complex exit medium, N = 1, My = 0, and z ranges with rows on one side only, inside only, far away."""
        def pillar(**incident):
            cfg = model_pillar("xz", "jones", **incident)
            return cfg

        def sampled(cfg, **sampling):
            cfg["sampling"].update(sampling)
            return cfg

        def stack(cfg, layers):
            cfg["structure"]["layers"] = layers
            return normalize_2d(cfg)

        rayleigh = model_square("xz", period=1., wavelength=1. - 1e-7, angle_deg=0., phi_deg=0.)
        opaque = model_pillar("xz", "jones")
        opaque["structure"]["layers"] = [{"name": "金属", "kind": "uniform", "thickness": 2., "n": "0.2+6j"}]
        thick_absorbing = model_pillar("yz", "jones")
        thick_absorbing["structure"]["layers"][0].update(thickness=1.5)
        thick_absorbing["structure"]["layers"][0]["x_profiles"][1]["n_values"][1] = "2+3j"
        models = {
            "grazing 80 deg": pillar(angle_deg=80., phi_deg=-40., wavelength=.5),
            "normal incidence, phi = 30": pillar(angle_deg=0., phi_deg=30., wavelength=.93),
            "a grazing order (lambda = L (1 - 1e-7))": rayleigh,
            "opaque metal film": normalize_2d(opaque),
            "strongly absorbing pillar": normalize_2d(thick_absorbing),
            "N = 1": pillar(M=0, My=0),
            "My = 0": pillar(M=3, My=0),
            "rows inside the stack only": sampled(pillar(), z_min=.02, z_max=.5),
            "rows below the stack only": sampled(pillar(), z_min=-.8, z_max=.4),
            "rows above the stack only": sampled(pillar(), z_min=.1, z_max=2.),
            "rows far from the stack": sampled(pillar(), z_min=-6., z_max=6.),
            # the phase arguments k0·z reach 1e7 and 1e10: the two programs' rounding differs by ~1e-16 × argument (5e-7 here),
            # which the checks allow for (NEAR_ROUNDING_FACTOR_2D, PHASE_ROUNDING_FACTOR_2D) instead of refusing an honest reply
            "rows 1e6 um away": sampled(pillar(), z_min=-1e6, z_max=1e6),
            "wavelength 1e-9 um (1 fm)": pillar(wavelength=1e-9),
            "grazing 89.9 deg, generic phi": pillar(angle_deg=89.9, phi_deg=33., wavelength=.83),
            "inside the stack only (hidden probe rows)": sampled(pillar(), z_min=.05, z_max=.5),
            "above the stack only (hidden probe row in front)": sampled(pillar(), z_min=.7, z_max=1.6),
            "below the stack only (hidden probe row behind)": sampled(pillar(), z_min=-1., z_max=-.1),
            "transmitted amplitudes subnormal (metal 0.2+6j, 9.7 um, lambda 0.5 um)": stack(sampled(pillar(wavelength=.5, angle_deg=20., phi_deg=10.), z_min=-.3, z_max=10.2),
                                                                                           [{"name": "金属", "kind": "uniform", "thickness": 9.7, "n": "0.2+6j"}]),
            "only rows behind the subnormal film": stack(sampled(pillar(wavelength=.5, angle_deg=20., phi_deg=10.), z_min=9.8, z_max=11.),
                                                          [{"name": "金属", "kind": "uniform", "thickness": 9.7, "n": "0.2+6j"}]),
            "thin layers": stack(pillar(), [{"name": "薄", "kind": "uniform", "thickness": 1e-3, "n": 1.46},
                                              {"name": "更薄", "kind": "uniform", "thickness": 1e-9, "n": 1.9}]),
        }
        if self.PROGRAM != "real":      # the stand-in solves in Python (~1.5 s a model): a representative subset is enough
            # (the stand-in's model solve is Python; "needs the rounding allowance" cases are in the subset: a stand-in mutant without it must fail)
            models = {label: models[label] for label in ("grazing 80 deg", "normal incidence, phi = 30", "opaque metal film",
                                                         "rows far from the stack", "wavelength 1e-9 um (1 fm)",
                                                         "grazing 89.9 deg, generic phi", "inside the stack only (hidden probe rows)",
                                                         "transmitted amplitudes subnormal (metal 0.2+6j, 9.7 um, lambda 0.5 um)")}
        for index, (label, cfg) in enumerate(models.items()):
            with self.subTest(model=label):
                cfg = normalize_2d(cfg)
                request, response = run_program(self.exe, cfg, self.tmp / f"hard {index}", token="hard")
                values, fields, manifest = self.accepted(cfg, request, response)
                self.assertEqual(len(values["DEr"]), (2 * cfg["incident"]["M"] + 1) * (2 * cfg["incident"]["My"] + 1))
        # a complex exit medium cannot be entered in the GUI (normalize_2d), but the protocol and the checks handle it
        cfg = model_pillar("xz", "jones")
        request = backend.build_request(cfg, self.tmp / "hard nII", "hard-nII")
        request["solve"]["stack"]["nII"] = [1.5, .3]
        with cpp_env(self.exe):
            response = common.call(self.exe, request, self.tmp)
        cfg["structure"]["n_substrate"] = "1.5+0.3j"
        self.accepted(cfg, request, response)

    # ------------------------------------------------------------------------------------------------ second review round
    # -- H1: sign and branch of kz
    def _honest_far(self, label="xz"):
        cfg, request, response, work = self.prepare(label)
        return cfg, request, response, work, response["results"][0]

    def test_kz_must_be_the_analytic_branch(self):
        """A kz with a NEGATIVE real part is as self-consistent as the right one (kz² unchanged, prop = 0, DE = 0, Ez = 0, Rp changes sign):
        the C++ one-character regression `kz.imag() <= 0` made every propagating kz negative and the GUI showed R = T = 0, absorption = 1
        with backend = cpp. It is refused for one order and for every propagating order of both media; a kz that is off by 1e-10 (inside
        the kz² tolerance) with efficiency, Ez and s/p made consistent only the independent analytic kz can see; and the propagation flag
        must come from the analytic kz, not from the reply's own."""
        def negate(which):
            def edit(work, far, response, cfg):
                for key, flag, de, ez, amplitude in (("kzI", "prop_r", "DEr", "Ez_r", "Rp"), ("kzII", "prop_t", "DEt", "Ez_t", "Tp")):
                    orders = [j for j, on in enumerate(far[flag]) if on]
                    for j in (orders[:1] if which == "one" else orders):
                        far[key][j] = [-far[key][j][0], -far[key][j][1]]
                        far[flag][j], far[de][j], far[ez][j] = 0, 0., [0., 0.]
                        far[amplitude][j] = [-far[amplitude][j][0], -far[amplitude][j][1]]
                    if which == "one":
                        break
            return edit
        for which in ("one", "all"):
            with self.subTest(which=which):
                cfg, request, response, work, far = self._honest_far()
                negate(which)(work, far, response, cfg)
                with self.assertRaisesRegex(BackendError, "实部为负"):
                    common.extract(response, cfg)

        def tiny_shift(work, far, response, cfg):
            """kzI of one propagating order x (1 + 1e-10); DE, Ez, Rp recomputed from it, so every other relation still holds."""
            j = far["prop_r"].index(1)
            kz = complex(*far["kzI"][j]) * (1 + 1e-10)
            ex, ey, kx, ky = complex(*far["Ex_r"][j]), complex(*far["Ey_r"][j]), far["Kx"][j], far["Ky"][j]
            nI, theta = float(cfg["structure"]["n_incident"]), math.radians(cfg["incident"]["angle_deg"])
            kt = math.hypot(kx, ky)
            ez = (kx * ex + ky * ey) / kz
            far["kzI"][j], far["Ez_r"][j] = [kz.real, kz.imag], [ez.real, ez.imag]
            far["DEr"][j] = kz.real * (abs(ex) ** 2 + abs(ey) ** 2 + abs(ez) ** 2) / (nI * math.cos(theta))
            rp = -(kx * ex + ky * ey) / (kt * kz / nI)
            far["Rp"][j] = [rp.real, rp.imag]
        cfg, request, response, work, far = self._honest_far()
        tiny_shift(work, far, response, cfg)
        with self.assertRaisesRegex(BackendError, "独立重算"):
            common.extract(response, cfg)

        def flag_from_own_kz(work, far, response, cfg):
            """An evanescent order given kz = 5e-12 + i|kz|: kz² still matches to 1e-11, the reply's own kz now says 'propagating', and
            flag, DE, Ez and Rp are all made consistent with that kz. Only the flag derived from the analytic kz (Re = 0) disagrees."""
            j = far["prop_r"].index(0)
            kz = complex(5e-12, far["kzI"][j][1])
            ex, ey, kx, ky = complex(*far["Ex_r"][j]), complex(*far["Ey_r"][j]), far["Kx"][j], far["Ky"][j]
            nI, theta = float(cfg["structure"]["n_incident"]), math.radians(cfg["incident"]["angle_deg"])
            ez = (kx * ex + ky * ey) / kz
            far["kzI"][j], far["Ez_r"][j], far["prop_r"][j] = [kz.real, kz.imag], [ez.real, ez.imag], 1
            far["DEr"][j] = kz.real * (abs(ex) ** 2 + abs(ey) ** 2 + abs(ez) ** 2) / (nI * math.cos(theta))
            rp = -(kx * ex + ky * ey) / (math.hypot(kx, ky) * kz / nI)
            far["Rp"][j] = [rp.real, rp.imag]
        cfg, request, response, work, far = self._honest_far()
        flag_from_own_kz(work, far, response, cfg)
        with patch.object(common, "KZ_TOLERANCE_2D", 1.):             # the kz comparison alone would not see it: the flag check must
            with self.assertRaisesRegex(BackendError, "解析 kz 的传播判据"):
                common.extract(response, cfg)

    # -- M1: hidden probe rows
    def test_probe_rows_are_always_added_on_both_sides_and_are_removed_again(self):
        """Two probe rows, whatever the user's z range is: one in front of the stack (z = -h), one behind it (z = z_bot + h)."""
        for label in ("xz", "yz", "inside", "above", "below", "dense", "uniform"):
            with self.subTest(label=label):
                cfg, request, response, work = self.prepare(label)
                z_user, lower, upper = backend.sampling_plan_2d(cfg)
                z_bot = backend.stack_bottom_2d(cfg)
                h = backend.probe_distance_2d(cfg["incident"]["wavelength"])
                self.assertEqual((lower, upper), (-h, z_bot + h))
                sent = request["sampling"]["z"]
                nz, nt = cfg["sampling"]["nz"], cfg["sampling"]["nx"]
                self.assertEqual(len(sent), nz + 2)
                self.assertEqual((sent[0], sent[-1]), (-h, z_bot + h))
                self.assertEqual(sent[1:-1], z_user.tolist())
                notes = []
                _, fields, manifest = self.accepted(cfg, request, response, 60., notes)
                self.assertEqual(notes, [], label)                     # an ordinary honest reply has nothing to say
                for key in COMPONENTS + ("E2",):
                    self.assertEqual(fields[key].shape, (nz, nt), key)
                self.assertEqual((fields["z"].shape, fields["phase_values"].shape, fields["phase_masks"].shape),
                                 ((nz,), (2, 6, nz, nt), (2, 6, nz, nt)))
                np.testing.assert_array_equal(fields["z"], z_user)
                self.assertEqual(manifest["shape"], [nz + 2, nt])              # the raw reply and the published directory keep the probes
                self.assertEqual(np.load(work / "fields.npy").shape, (6, nz + 2, nt))
                self.assertEqual(np.load(work / "z.npy").shape, (nz + 2,))
        # the distance: 0.02 wavelength, never inside the 1e-9 um band that is not checked
        self.assertEqual(backend.probe_distance_2d(.83), 0.02 * .83)
        self.assertEqual(backend.probe_distance_2d(1e-9), backend.PROBE_MIN_DISTANCE_2D)
        self.assertGreaterEqual(backend.PROBE_MIN_DISTANCE_2D, 10 * backend.REGION_MARGIN_2D)
        # whatever the range: only rows inside the 1e-9 um band next to the stack, or a single row, or rows on both sides
        for z_min, z_max, rows in ((-5e-10, .3, 20), (.3, .30001, 20), (-1., 1., 20), (-100., 100., 1200)):
            cfg = model_rows(z_min, z_max, nz=rows)
            plan = backend.sampling_plan_2d(cfg)
            self.assertEqual((plan[1], plan[2]), (-.02, .57 + .02), (z_min, z_max))
        # the 4,000,000-sample limit of the protocol: a plan that would exceed it sends no probe rows (and says so in the note, see below)
        cfg = model_rows(.05, .5)
        cfg["sampling"].update(nx=1600, nz=2498)           # 2498 + 2 rows = 2500 x 1600 = 4,000,000: still allowed
        self.assertEqual(backend.sampling_plan_2d(cfg)[1:], (-.02, .59))
        cfg["sampling"].update(nx=1600, nz=2499)           # 2499 + 2 rows exceeds it
        self.assertEqual(backend.sampling_plan_2d(cfg)[1:], (None, None))
        self.assertEqual(len(backend.sent_z_2d(cfg)), 2499)
        cfg = model_pillar()                               # rows on both sides of the stack: the limit applies to them as well
        cfg["sampling"].update(nx=1600, nz=2499)
        self.assertEqual(backend.sampling_plan_2d(cfg)[1:], (None, None))

    def users_data_of(self, cfg, request, response, label):
        """The fields the GUI gets from this (probe-carrying) reply, the fields of the same request without probe rows, and the points whose weak-field
        mask the re-derivation changed from 'weak' to 'strong' (a boolean array, user's rows only; None when it changed nothing)."""
        changed = []
        original = backend._rederive_weak_field_mask_2d

        def spy(six, amplitude, phase, mask, *args, **kwargs):
            before = mask.copy()
            original(six, amplitude, phase, mask, *args, **kwargs)
            changed.append(int((before != mask).sum()))
        with patch.object(backend, "_rederive_weak_field_mask_2d", spy):
            _, probed, _ = self.accepted(cfg, request, response)
        folder = self.tmp / f"plain {label}"
        folder.mkdir()
        with patch.object(backend, "sampling_plan_2d", no_probes), cpp_env(self.exe):
            plain_request = backend.build_request(cfg, folder / "out", "plain-" + label, 1)
            self.assertEqual(len(plain_request["sampling"]["z"]), cfg["sampling"]["nz"])
            plain_response = common.call(self.exe, plain_request, folder)
            _, plain, _ = self.accepted(cfg, plain_request, plain_response)
        rederived = None
        if any(changed):
            raw = np.load(Path(request["output_dir"]) / "phase_mask.npy").astype(bool)[0][:, 1:-1]       # C++'s mask (peak over all rows), user's rows
            rederived = raw & ~probed["phase_masks"][0]
        return probed, plain, rederived

    def assert_same_user_data(self, probed, plain, label, rederived):
        """The true guarantee (round 3, L1): everything is bit for bit the same as in the run without probe rows -- the six fields, E2, the masks, every
        other key, and the phase of every point -- except the phase of the points the weak-field re-derivation added (`rederived`): Python computes arg F and
        arg(F conj(carrier)) there, and its libm / operation order is not C++'s, so both references agree to 1e-14 rad, not bit for bit (observed: macOS
        reference 0 identical and reference 1 <= 2.2e-15; Windows/MinGW both references <= 4.4e-16, reference 0 at 3-13 points)."""
        self.assertEqual(set(probed), set(plain))
        for key in probed:
            if key != "phase_values":
                np.testing.assert_array_equal(probed[key], plain[key], err_msg=f"{label} {key}")
        a, b = probed["phase_values"], plain["phase_values"]
        self.assertEqual((a.shape, a.dtype), (b.shape, b.dtype))
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b), err_msg=f"{label}: NaN pattern of the phase")
        allowed = np.zeros(a.shape, bool) if rederived is None else np.broadcast_to(rederived, a.shape)
        differing = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        self.assertFalse((differing & ~allowed).any(), f"{label}: the phase differs outside the re-derived points")
        gap = np.abs(a - b)[differing]
        self.assertLessEqual(float(gap.max()) if gap.size else 0., 1e-14, label)
        return int(differing.sum())

    def test_probe_rows_leave_the_users_data_unchanged(self):
        """Every row is computed on its own, so the user's rows of a reply that also holds probe rows are those of the same request without
        them, bit for bit -- with ONE documented exception (test_the_rederive_branch...): the weak-field mask is re-derived from the user's own
        rows (a probe row can hold the component's peak), and only where that changes the mask does the second phase reference differ (1e-14 rad)."""
        for label in ("xz", "yz", "inside", "above", "below", "dense", "uniform"):
            with self.subTest(label=label):
                cfg, request, response, work = self.prepare(label)
                probed, plain, rederived = self.users_data_of(cfg, request, response, label)
                self.assert_same_user_data(probed, plain, label, rederived)

    def test_the_rederive_branch_changes_the_phase_only_at_the_rederived_points_and_by_less_than_1e_14(self):
        """The one place where 'the user's data is bit for bit the same with probe rows' is NOT true (round 3): the user's rows lie deep inside a metal
        film, where the field is orders of magnitude below the probe rows' (next to the film). The mask C++ wrote (peak including the probe rows) then
        hides points that are not weak on the user's own grid; they get their phase from Python. The six fields, E2, the mask and the phase of every other
        point are still bit for bit those of the run without probes; at the re-derived points the phase agrees to 1e-14 rad (assert_same_user_data)."""
        cfg = model_metal_inside()
        request, response = run_program(self.exe, cfg, self.tmp / "rederive base", token="rederive")
        probed, plain, rederived = self.users_data_of(cfg, request, response, "rederive")
        self.assertIsNotNone(rederived, "the model must trigger the re-derivation (the probe rows hold the peak)")
        self.assertGreater(int(rederived.sum()), 100)
        differing = self.assert_same_user_data(probed, plain, "metal inside", rederived)
        self.assertLessEqual(differing, 2 * int(rederived.sum()))
        self.assertFalse(np.isnan(probed["phase_values"][~probed["phase_masks"]]).any())

    def test_the_weak_field_mask_is_rederived_from_the_users_rows(self):
        """Synthetic reply: the probe row holds a peak much larger than any user row, so C++ masked points that are not weak on the user's grid.
        They get the phase the program would have written (arg F; arg F·conj(carrier)); nothing else changes."""
        rng = np.random.default_rng(5)
        nz, nt, offset = 6, 9, 1
        six = (rng.normal(size=(6, nz + 2, nt)) + 1j * rng.normal(size=(6, nz + 2, nt))) * 10. ** rng.uniform(-3, 0, size=(6, nz + 2, nt))
        six[:, 0, :] *= 400.                               # the probe row below the stack
        amplitude = abs(six)
        weak = amplitude <= 1e-6 * amplitude.max(axis=(1, 2), keepdims=True)
        # place user-row points just between the two thresholds: weak with the probe's peak, not weak without it
        peak_user = abs(six[:, 1:nz + 1]).max(axis=(1, 2))
        for c in range(6):
            six[c, 3, 4] = 3e-6 * peak_user[c] * np.exp(.3j * (c + 1))
        amplitude = abs(six)
        weak = amplitude <= 1e-6 * amplitude.max(axis=(1, 2), keepdims=True)
        self.assertTrue(weak[:, 3, 4].all())
        x, y = np.linspace(0, .6, nt), np.full(nt, .2)
        z = np.concatenate([[-.02], np.linspace(.1, .5, nz), [.6]])[:nz + 2]
        far = {"Kx": np.array([0., .3, .1]), "Ky": np.array([0., .2, -.1]), "kzI": np.array([.8 + 0j, .7 + 0j, .9 + 0j])}
        k0 = 2 * np.pi / .83
        mask = np.stack([weak, weak])
        phase = np.where(mask, np.nan, np.stack([np.angle(six), np.angle(six)]))
        keep = slice(offset, offset + nz)
        before_phase, before_mask = phase.copy(), mask.copy()
        backend._rederive_weak_field_mask_2d(six, amplitude, phase, mask, x, y, z, far, k0, keep, offset)
        region = amplitude[:, keep]
        expected = region <= 1e-6 * region.max(axis=(1, 2), keepdims=True)
        np.testing.assert_array_equal(mask[0][:, keep], expected)
        np.testing.assert_array_equal(mask[1][:, keep], expected)
        changed = before_mask != mask
        self.assertTrue(changed[:, :, keep].any() and not changed[:, :, 0].any() and not changed[:, :, -1].any())
        i0 = 1
        kx0, ky0, kz0 = k0 * far["Kx"][i0], k0 * far["Ky"][i0], k0 * far["kzI"][i0].real
        for c, row, point in zip(*np.nonzero(changed[0])):
            carrier = np.exp(1j * (-kx0 * x[point] - ky0 * y[point] + kz0 * z[row]))
            same_angle = lambda a, b: abs(np.exp(1j * a) - np.exp(1j * b)) < 1e-13
            self.assertTrue(same_angle(phase[0, c, row, point], np.angle(six[c, row, point])))
            self.assertTrue(same_angle(phase[1, c, row, point], np.angle(six[c, row, point] * np.conj(carrier))))
        np.testing.assert_array_equal(phase[~changed], before_phase[~changed])        # every other point (NaN included) is untouched
        self.assertFalse(np.isnan(phase[~mask]).any())

    def test_probe_rows_tie_the_reply_to_the_requested_incident_field(self):
        """A solver that misreads psi (90 - psi), the sign of delta or the phase of the Jones vector answers consistently for the wrong input:
        its far field, fields and phase agree with each other. Only the incident term in a row in front of the stack gives it away; with
        the z range inside the stack (or above it) that row is the hidden probe. Without probe rows the same replies are accepted."""
        def misread(kind):
            def edit(solve):
                if kind == "psi":
                    solve["input"]["psi_deg"] = 90. - solve["input"]["psi_deg"]
                elif kind == "delta":
                    solve["input"]["delta_deg"] = -solve["input"]["delta_deg"]
                else:
                    solve["input"]["delta_deg"] = solve["input"]["delta_deg"] + 40.
            return edit
        for label in ("inside", "above"):
            cfg = self.bundles[label][0]
            for kind in ("psi", "delta", "phase"):
                with self.subTest(rows=label, misread=kind):
                    folder = self.tmp / f"misread {label} {kind}"
                    folder.mkdir()
                    request = backend.build_request(cfg, folder / "out", "m-" + kind, 1)
                    sent = copy.deepcopy(request)
                    misread(kind)(sent["solve"])
                    with cpp_env(self.exe):
                        response = common.call(self.exe, sent, folder)
                    with self.assertRaisesRegex(BackendError, "叠层之外"):
                        self.accepted(cfg, request, response)
                    # the same misreading with no probe row: nothing is left that sees it
                    plain = self.tmp / f"misread plain {label} {kind}"
                    plain.mkdir()
                    with patch.object(backend, "sampling_plan_2d", no_probes), cpp_env(self.exe):
                        request = backend.build_request(cfg, plain / "out", "p-" + kind, 1)
                        sent = copy.deepcopy(request)
                        misread(kind)(sent["solve"])
                        response = common.call(self.exe, sent, plain)
                        self.accepted(cfg, request, response)

    def test_probe_row_behind_the_stack_ties_the_transmitted_amplitudes(self):
        """The user's z range lies in front of the stack: no row sees the transmitted orders. One transmitted order made 1% too large
        (amplitudes, efficiency, Ez and s/p consistent) is refused thanks to the probe row behind the stack; without it, accepted."""
        def amplify(work, far, response, cfg):
            j = [k for k, on in enumerate(far["prop_t"]) if on][1]
            for key in ("Ex_t", "Ey_t", "Ez_t", "Ts", "Tp"):
                far[key][j] = [far[key][j][0] * 1.01, far[key][j][1] * 1.01]
            far["DEt"][j] *= 1.01 ** 2
        cfg, request, response, work = self.prepare("below")
        amplify(work, response["results"][0], response, cfg)
        with self.assertRaisesRegex(BackendError, "出射区"):
            self.accepted(cfg, request, response)
        with patch.object(backend, "sampling_plan_2d", no_probes):
            folder = self.tmp / "behind plain"
            folder.mkdir()
            with cpp_env(self.exe):
                request = backend.build_request(cfg, folder / "out", "behind", 1)
                response = common.call(self.exe, request, folder)
            amplify(None, response["results"][0], response, cfg)
            self.accepted(cfg, request, response)

    # -- M2: every row outside the stack
    def test_every_row_outside_the_stack_is_checked(self):
        """The check used to visit 32 evenly spread rows per side: a change in any other row passed (82 of 114 rows in the reviewer's model)."""
        cfg, request, response, work = self.prepare("dense")
        far = backend.far_arrays_2d(response["results"][0], 25)
        six = np.load(work / "fields.npy")
        x, y, z = (np.load(work / name) for name in ("x.npy", "y.npy", "z.npy"))
        z_bot = backend.stack_bottom_2d(cfg)
        outside = np.flatnonzero((z < -1e-9) | (z > z_bot + 1e-9))
        self.assertGreater(len(outside), 100)
        honest = backend.near_field_check_2d(six, x, y, z, far, request, cfg)
        self.assertLessEqual(honest.ratio, 1e-3)
        self.assertEqual((honest.rows, honest.skipped), (len(outside), 0))
        peak_e, peak_h = float(abs(six[:3]).max()), float(abs(six[3:]).max())
        for row in outside:
            changed = six.copy()
            changed[2, row] += 1e-3 * peak_e                       # Ez of this single row is off by 0.1% of the peak
            self.assertGreater(backend.near_field_check_2d(changed, x, y, z, far, request, cfg).ratio, 1., f"row {row}")
            changed = six.copy()
            changed[3:, row] += 1e-5 * peak_h                      # or the H family of this row by 1e-5
            self.assertGreater(backend.near_field_check_2d(changed, x, y, z, far, request, cfg).ratio, 1., f"row {row}")
        inside = np.setdiff1d(np.arange(len(z)), outside)
        changed = six.copy()
        changed[2, inside] += 1e-3 * peak_e                        # inside the stack there is no independent reference: not visible
        self.assertLessEqual(backend.near_field_check_2d(changed, x, y, z, far, request, cfg).ratio, 1e-3)
        # the chunks bound the extra memory: the result does not depend on the chunk size
        with patch.object(backend, "NEAR_ROWS_PER_CHUNK_2D", 7):
            self.assertEqual(backend.near_field_check_2d(six, x, y, z, far, request, cfg).rows, len(outside))
            changed = six.copy()
            changed[1, outside[-1]] += 1e-5 * peak_e
            self.assertGreater(backend.near_field_check_2d(changed, x, y, z, far, request, cfg).ratio, 1.)
        # a non-finite far-field array is never "close to" the stored field
        broken = {key: value.copy() for key, value in far.items()}
        broken["Ex_t"][3] = complex(np.nan, 0.)
        self.assertFalse(backend.near_field_check_2d(six, x, y, z, broken, request, cfg).ratio <= 1.)

    # -- M3: subnormal amplitudes
    def test_subnormal_amplitudes_do_not_refuse_an_honest_reply(self):
        """Metal film 0.2+6j, d = 9.7 um, lambda = 0.5 um: the transmitted side peaks at 4e-319 (subnormal, ~1e-5 relative resolution): the
        far-field check divided by it and refused the honest reply (residual 2.3e-5) with a misleading note. Every relative scale has an
        absolute floor now (cpp_backend.AMPLITUDE_FLOOR_2D = 1e-200: below it amplitudes are physically zero)."""
        cfg, request, response, work, far = self._honest_far()
        values = common.extract(response, cfg)
        st, inc = cfg["structure"], cfg["incident"]
        residual = lambda v: common.far_field_residual_2d(v, float(st["n_incident"]), float(st["n_substrate"]), math.radians(inc["phi_deg"]))
        subnormal = copy.deepcopy(values)
        for key in ("Ex_t", "Ey_t", "Ez_t", "Ts", "Tp"):
            subnormal[key] = [v * 1e-318 for v in subnormal[key]]       # |v| ~ 1e-318: a few bits of precision
        self.assertLess(residual(subnormal), 1e-12)
        with patch.object(common, "AMPLITUDE_FLOOR_2D", 5e-324):       # (practically) no floor: this is the reviewer's failure
            self.assertGreater(residual(subnormal), 1e-9)
        tiny = copy.deepcopy(values)                                    # a side that is tiny but normal (1e-100) is still compared with its own size
        for key in ("Ex_t", "Ey_t", "Ez_t", "Ts", "Tp"):
            tiny[key] = [v * 1e-100 for v in tiny[key]]
        self.assertLess(residual(tiny), 1e-12)
        tiny["Ts"], tiny["Tp"] = tiny["Tp"], tiny["Ts"]
        self.assertGreater(residual(tiny), .1)
        # the near-field check has the same floor: the rows behind the stack and the transmitted far-field amplitudes scaled to 1e-318
        cfg, request, response, work = self._honest_far()[:4]
        far_arrays = backend.far_arrays_2d(response["results"][0], 25)
        six = np.load(work / "fields.npy")
        x, y, z = (np.load(work / name) for name in ("x.npy", "y.npy", "z.npy"))
        behind = np.flatnonzero(z > backend.stack_bottom_2d(cfg) + 1e-9)
        self.assertGreaterEqual(len(behind), 3)
        six[:, behind, :] *= 1e-318
        for key in ("Ex_t", "Ey_t"):
            far_arrays[key] = far_arrays[key] * 1e-318
        self.assertLessEqual(backend.near_field_check_2d(six, x, y, z, far_arrays, request, cfg).ratio, 1.)
        with patch.object(common, "AMPLITUDE_FLOOR_2D", 5e-324):
            self.assertGreater(backend.near_field_check_2d(six, x, y, z, far_arrays, request, cfg).ratio, 1.)

    # -- M4: per-order scales, NaN
    def test_the_far_field_check_works_per_order_and_does_not_swallow_nan(self):
        """One evanescent order inflated 1e150 (consistently) used to set the scale of its whole side: every other order, with Rs/Rp swapped,
        was below the tolerance. Each order is compared with its own size now. NaN made by an overflow is reported as inf (the old built-in max
        returned 0.0)."""
        cfg, request, response, work, far = self._honest_far()
        values = common.extract(response, cfg)
        st, inc = cfg["structure"], cfg["incident"]
        residual = lambda v: common.far_field_residual_2d(v, float(st["n_incident"]), float(st["n_substrate"]), math.radians(inc["phi_deg"]))
        self.assertLess(residual(values), 1e-12)
        j = max(range(25), key=lambda k: values["kzI"][k].imag)
        inflated = copy.deepcopy(values)
        for key in ("Ex_r", "Ey_r", "Rs", "Rp"):
            inflated[key][j] *= 1e150
        self.assertLess(residual(inflated), 1e-12)                       # self-consistent: nothing to see in the far field alone
        for k in range(25):
            if inflated["prop_r"][k]:
                inflated["Rs"][k], inflated["Rp"][k] = inflated["Rp"][k], inflated["Rs"][k]
        self.assertGreater(residual(inflated), .1)
        ez_only = copy.deepcopy(values)                                   # the Ez term on its own
        k = max(range(25), key=lambda index: abs(values["Ez_t"][index]))
        ez_only["Ez_t"][k] *= 1.01
        self.assertGreater(residual(ez_only), 1e-3)
        # NaN: finite but huge amplitudes whose products overflow to inf - inf
        kx, ky, big = [3.0, .1], [-2.5, .2], 1e308
        nan_values = {"Kx": kx, "Ky": ky, "kzI": [2.9j, .9 + 0j], "kzII": [2.5j, 1.2 + 0j],
                      "Ex_r": [complex(big, big), .1 + 0j], "Ey_r": [complex(-big, big), .2 + 0j], "Ez_r": [0j, 0j], "Rs": [0j, 0j], "Rp": [0j, 0j],
                      "Ex_t": [0j, .1 + 0j], "Ey_t": [0j, .1 + 0j], "Ez_t": [0j, 0j], "Ts": [0j, 0j], "Tp": [0j, 0j]}
        for sign, n, kz_key, ex, ey, ez, s, p in ((-1., 1., "kzI", "Ex_r", "Ey_r", "Ez_r", "Rs", "Rp"), (1., 1.45, "kzII", "Ex_t", "Ey_t", "Ez_t", "Ts", "Tp")):
            kz, kt = nan_values[kz_key][1], math.hypot(kx[1], ky[1])
            nan_values[s][1] = (-ky[1] * nan_values[ex][1] + kx[1] * nan_values[ey][1]) / kt
            nan_values[p][1] = sign * (kx[1] * nan_values[ex][1] + ky[1] * nan_values[ey][1]) / (kt * kz / n)
            nan_values[ez][1] = -sign * (kx[1] * nan_values[ex][1] + ky[1] * nan_values[ey][1]) / kz
        self.assertEqual(common.far_field_residual_2d(nan_values, 1.0, complex(1.45), .3), math.inf)

    # -- L3: metadata the reply only reports
    def test_reply_metadata_must_match_the_request_and_the_clock(self):
        cfg, request, response, work = self.prepare("xz")
        layers = len(cfg["structure"]["layers"])
        self.accepted(cfg, request, response, 60.)                                     # honest, with a (generous) wall time
        self.accepted(*self.prepare("xz")[:3])                                         # and without one
        set_timing = lambda **values: (lambda r, far, fd: far["timing"].update(**values))
        edits = {
            "threads 2 for a 1-thread request": lambda r, far, fd: r.update(threads=2),
            "threads 256": lambda r, far, fd: r.update(threads=256),
            "n_eig = 0 although the model has structured layers": lambda r, far, fd: far.update(n_eig=0),
            "n_eig above the number of layers": lambda r, far, fd: far.update(n_eig=layers + 1),
            "message although status is ok": lambda r, far, fd: far.update(message="全部正常"),
            "message missing": lambda r, far, fd: far.pop("message"),
            "backward error": lambda r, far, fd: far.update(max_solve_backward_error=1e-16),
            "solve time 1e6 s": set_timing(total=1e6),
            "operators time beyond wall x threads": set_timing(operators=1e6),
            "eig time beyond wall x threads": set_timing(eig=1e6),
            "reply sections 1e6 s": lambda r, far, fd: fd["timing"].update(solve=1e6, sample=2e6, phase=3e6, write=4e6),
            "one section 1e6 s": lambda r, far, fd: fd["timing"].update(write=1e6),
        }
        for label, edit in edits.items():
            with self.subTest(case=label):
                self.refuse(label, edit, wall=2.)
        # the timing cases are only judged against the clock when the caller measured it
        for label in ("solve time 1e6 s", "reply sections 1e6 s"):
            cfg, request, response, work = self.prepare("xz")
            edits[label](response, response["results"][0], response["field_data"])
            self.accepted(cfg, request, response)
        # a model without structured layers: no eigen decomposition, so no eigen residual
        cfg, request, response, work = self.prepare("uniform")
        self.assertEqual(response["results"][0]["n_eig"], 0)
        self.accepted(cfg, request, response, 60.)
        self.refuse("max_eig_residual without an eigen decomposition", lambda r, far, fd: far.update(max_eig_residual=1e-3), plane="uniform", wall=60.)
        self.refuse("n_eig above the number of layers (model without structured layers)", lambda r, far, fd: far.update(n_eig=99),
                    plane="uniform", wall=60.)

    # -- the two `ok` checks are independent
    def test_ok_must_be_true_in_each_extractor_on_its_own(self):
        """Both extractors check `ok is True`; each alone must refuse a truthy 1 even if everything else is valid."""
        for value in (1, "true", "yes"):
            cfg, request, response, work = self.prepare("xz")
            response["ok"] = value
            with self.assertRaisesRegex(BackendError, "ok"):
                common.extract(response, cfg)
            with self.assertRaisesRegex(BackendError, "ok"):
                backend.extract_fields(response, cfg, request)


    # ------------------------------------------------------------------------------------------------ third review round
    # -- L2: what the notes say when a check could not be performed
    def test_notes_say_when_a_check_could_not_be_performed(self):
        """An honest reply has no notes. When the rounding allowance makes rows (or the phase check) meaningless, the checks are skipped AND the note
        says so: the number of rows, and -- separately and explicitly -- that the incident polarisation / transmitted amplitudes were not tied to the request."""
        cfg, request, response, work = self.prepare("xz")
        notes = []
        self.accepted(cfg, request, response, 60., notes)
        self.assertEqual(notes, [])
        outside = int(np.sum((np.load(work / "z.npy") < -1e-9) | (np.load(work / "z.npy") > backend.stack_bottom_2d(cfg) + 1e-9)))
        self.assertEqual(outside, 10)
        with patch.object(backend, "CHECK_LIMIT_2D", 1e-15):         # every allowance exceeds it: nothing can be checked
            cfg, request, response, work = self.prepare("xz")
            notes = []
            self.accepted(cfg, request, response, 60., notes)
        self.assertEqual(len(notes), 4, notes)
        self.assertRegex(notes[0], rf"近场自洽检查跳过了 {outside} 行")
        self.assertRegex(notes[1], "入射偏振与入射相位没有与请求核对")
        self.assertRegex(notes[2], "透射振幅没有与请求核对")
        self.assertRegex(notes[3], "相位自洽检查未执行")
        # only the rows of one side skipped: only that side's sentence (patch the check so that the incident side is usable)
        real_check = backend.near_field_check_2d

        def blind_behind(*args, **kwargs):
            check = real_check(*args, **kwargs)
            check.transmitted_rows = 0
            return check
        cfg, request, response, work = self.prepare("xz")
        notes = []
        with patch.object(backend, "near_field_check_2d", blind_behind):
            self.accepted(cfg, request, response, 60., notes)
        self.assertEqual(len(notes), 1)
        self.assertRegex(notes[0], "透射振幅没有与请求核对")

    def test_without_probe_rows_a_range_inside_the_stack_is_not_tied_to_the_request_and_the_note_says_so(self):
        """The protocol's 4,000,000-sample limit is the only reason left for sending no probe rows. Then a z range inside the stack has no row
        outside it: the incident polarisation / phase and the transmitted amplitudes are NOT checked -- accepted, with two explicit sentences."""
        cfg = model_rows(.05, .5)
        nz, nt = cfg["sampling"]["nz"], cfg["sampling"]["nx"]
        with patch.object(backend, "MAX_SAMPLES_2D", (nz + 1) * nt):          # room for the user's rows, not for two more
            self.assertEqual(backend.sampling_plan_2d(cfg)[1:], (None, None))
            request, response = run_program(self.exe, cfg, self.tmp / "no room", token="no-room")
            self.assertEqual(len(request["sampling"]["z"]), nz)
            notes = []
            self.accepted(cfg, request, response, 60., notes)
            self.assertEqual(len(notes), 3, notes)
            self.assertRegex(notes[0], "没有加隐藏探针行")
            self.assertRegex(notes[1], "入射偏振与入射相位没有与请求核对")
            self.assertRegex(notes[2], "透射振幅没有与请求核对")
            # rows on both sides: still checked, so only the first sentence
            cfg = model_pillar("xz", "jones")
            request, response = run_program(self.exe, cfg, self.tmp / "no room 2", token="no-room-2")
            notes = []
            self.accepted(cfg, request, response, 60., notes)
            self.assertEqual(len(notes), 1, notes)
            self.assertRegex(notes[0], "没有加隐藏探针行")

    def test_rows_that_are_all_too_noisy_to_check_still_tie_the_incident_field_through_the_probe_rows(self):
        """lambda = 1 fm (1e-9 um) with the z range +-1000 um: every row of the user's is 1e12 wavelengths away, the rounding allowance of each exceeds
        1e-3 of the peak and the row is skipped (a note says so). With the probe rows (0.02 wavelength -> 1e-8 um from the interface) the incident field is
        still tied to the request: a solver that misreads psi (90 - psi) is refused; without the probe rows it was accepted with only the 'rows skipped' note.
        Needs the real program (the stand-in's Python field_at overflows at that distance for lambda = 1 fm)."""
        if self.PROGRAM != "real":
            self.skipTest("the stand-in's Python reference overflows (0 x inf) at |z| = 1000 um for lambda = 1 fm")
        layers = [{"name": "柱", "kind": "patterned2d", "thickness": 2e-9, "y_edges": [0., 1e-9, 2e-9, 3e-9],
                   "x_profiles": [{"edges": [0., 3e-9], "n_values": [1.46]}, {"edges": [0., 1e-9, 2e-9, 3e-9], "n_values": [1.46, 2.3, 1.46]},
                                  {"edges": [0., 3e-9], "n_values": [1.46]}]},
                  {"name": "膜", "kind": "uniform", "thickness": 1e-9, "n": 1.6}]
        cfg = normalize_2d({"schema_version": 1, "name": "飞米", "mode": "periodic2d",
                            "incident": {"wavelength": 1e-9, "angle_deg": 20., "phi_deg": 30., "polarization": "jones", "M": 1, "My": 1, "psi_deg": 30., "delta_deg": 40.},
                            "structure": {"period": 3e-9, "period_y": 3e-9, "n_incident": 1., "n_substrate": 1.5, "layers": layers},
                            "sampling": {"nx": 40, "nz": 20, "z_min": -1e3, "z_max": 1e3, "slice_plane": "xz", "slice_at": 1.1e-9}})
        folder = self.tmp / "fm"
        folder.mkdir()
        request = backend.build_request(cfg, folder / "out", "fm-honest", 1)
        with cpp_env(self.exe):
            response = common.call(self.exe, request, folder)
        notes = []
        self.accepted(cfg, request, response, 60., notes)                       # honest: accepted ...
        self.assertTrue(any("近场自洽检查跳过了" in note for note in notes), notes)       # ... with the 'rows skipped' note
        self.assertFalse(any("没有与请求核对" in note for note in notes), notes)           # but the incident side WAS checked (by the probe row)
        folder = self.tmp / "fm misread"
        folder.mkdir()
        request = backend.build_request(cfg, folder / "out", "fm-misread", 1)
        sent = copy.deepcopy(request)
        sent["solve"]["input"]["psi_deg"] = 90. - sent["solve"]["input"]["psi_deg"]
        with cpp_env(self.exe):
            response = common.call(self.exe, sent, folder)
        with self.assertRaisesRegex(BackendError, "叠层之外"):
            self.accepted(cfg, request, response)
        # and the same misreading without probe rows, as the code was before this round: accepted, but now with the explicit sentence
        with patch.object(backend, "sampling_plan_2d", no_probes):
            folder = self.tmp / "fm plain"
            folder.mkdir()
            plain = backend.build_request(cfg, folder / "out", "fm-plain", 1)
            sent = copy.deepcopy(plain)
            sent["solve"]["input"]["psi_deg"] = 90. - sent["solve"]["input"]["psi_deg"]
            with cpp_env(self.exe):
                response = common.call(self.exe, sent, folder)
            notes = []
            self.accepted(cfg, plain, response, 60., notes)
            self.assertTrue(any("入射偏振与入射相位没有与请求核对" in note for note in notes), notes)

    # -- L4: the forms of the comparisons, and the absolute floor
    def test_a_nan_residual_is_never_accepted(self):
        """All three acceptance tests are written 'not residual <= tolerance', so that NaN refuses. Honest code paths never produce a NaN there (the far-field
        residual maps it to inf, the near-field check returns inf for a non-finite reconstruction, the phase residual maps it to inf), which made the
        reviewer's mutants 'residual > tolerance' equivalent in the product and survivors of the suite; this pins the form."""
        cfg, request, response, work = self.prepare("xz")
        with patch.object(common, "far_field_residual_2d", return_value=math.nan):
            with self.assertRaisesRegex(BackendError, "Rs、Rp、Ts、Tp"):
                common.extract(copy.deepcopy(response), cfg)
        with patch.object(backend, "near_field_check_2d", return_value=backend.NearFieldCheck2D(ratio=math.nan, residual=math.nan, where="入射区的 E 场")):
            with self.assertRaisesRegex(BackendError, "叠层之外"):
                self.accepted(cfg, request, copy.deepcopy(response))
        with patch.object(backend, "phase_residual_2d", return_value=math.nan):
            with self.assertRaisesRegex(BackendError, "相位文件与六复场不自洽"):
                self.accepted(cfg, request, copy.deepcopy(response))

    def test_the_amplitude_floor_is_1e_200_not_smaller(self):
        """Every 'relative to a peak' residual has an absolute floor of 1e-200 (cpp_backend.AMPLITUDE_FLOOR_2D): below it an amplitude is physically zero and a
        100% change of it is not an error; above it the change is seen. A floor of 1e-300 (the reviewer's mutant) would flag the 1e-250 case.
        (test_subnormal_amplitudes_... cannot tell 1e-200 from 1e-300: both floors lie far above its 4e-319.)"""
        cfg, request, response, work = self.prepare("xz")
        values = common.extract(response, cfg)
        st, inc = cfg["structure"], cfg["incident"]
        args = (float(st["n_incident"]), float(st["n_substrate"]), math.radians(inc["phi_deg"]))
        amplitudes = ("Ex_r", "Ey_r", "Ez_r", "Rs", "Rp", "Ex_t", "Ey_t", "Ez_t", "Ts", "Tp")
        j = values["prop_t"].index(True)
        for factor, floored in ((1e-250, True), (1e-150, False)):
            scaled = {key: ([v * factor for v in value] if key in amplitudes else value) for key, value in values.items()}
            self.assertLess(common.far_field_residual_2d(scaled, *args), 1e-12)               # honest, scaled: consistent at any size
            broken = {key: list(value) if isinstance(value, list) else value for key, value in scaled.items()}
            broken["Ts"][j] = broken["Ts"][j] * 2                                              # +100% on one amplitude
            residual = common.far_field_residual_2d(broken, *args)
            with self.subTest(amplitude=factor):
                if floored:
                    self.assertLess(residual, 1e-40)
                else:
                    self.assertGreater(residual, .01)

    # -- L4 (ii): the clock
    def test_the_timing_bound_is_the_documented_one(self):
        """total and the sum of the reply's sections may not exceed 1.05 x wall + 1 s, the per-thread sums threads x that. Just inside accepted, just outside
        refused: a bound relaxed by a factor (or loosened by seconds) is caught on either side."""
        wall = 10.
        limit = 1.05 * wall + 1.
        cases = {
            "total": lambda r, far, fd, delta: far["timing"].update(total=limit + delta),
            "sections": lambda r, far, fd, delta: fd["timing"].update(solve=limit + delta - .03, sample=.01, phase=.01, write=.01),
            "operators": lambda r, far, fd, delta: far["timing"].update(operators=limit + delta),
            "eig": lambda r, far, fd, delta: far["timing"].update(eig=limit + delta),
            "linear": lambda r, far, fd, delta: far["timing"].update(linear=limit + delta),
            "product": lambda r, far, fd, delta: far["timing"].update(product=limit + delta),
        }
        for label, edit in cases.items():
            with self.subTest(case=label, side="inside"):
                cfg, request, response, work = self.prepare("xz")
                edit(response, response["results"][0], response["field_data"], -.02)
                self.accepted(cfg, request, response, wall)
            with self.subTest(case=label, side="outside"):
                self.refuse(label + " just beyond the bound", lambda r, far, fd, edit=edit: edit(r, far, fd, +.02), wall=wall, words="用时")
        for threads in (3, 64):                      # the four per-thread sums scale with the number of threads
            for label in ("operators", "eig", "linear", "product"):
                for delta, ok in ((-.02, True), (+.02, False)):
                    with self.subTest(threads=threads, case=label, accepted=ok):
                        cfg, request, response, work = self.prepare("xz")
                        request["solve"]["options"]["threads"] = response["threads"] = threads
                        response["results"][0]["timing"][label] = threads * limit + delta
                        if ok:
                            self.accepted(cfg, request, response, wall)
                        else:
                            with self.assertRaisesRegex(BackendError, "用时"):
                                self.accepted(cfg, request, response, wall)

    # -- L4 (iii): Kx, Ky
    def test_kx_ky_must_match_the_request_to_1e_11(self):
        """Kx and Ky are recomputed from the request (1e-11 x max(1, |K|)). The reviewer's mutant that loosened the tolerance to 1e-3 survived: 1e-9 and
        1e-7 relative changes (and 3e-11, just beyond) are refused, 2e-12 (just inside) is not."""
        for key in ("Kx", "Ky"):
            cfg, request, response, work = self.prepare("xz")
            far = response["results"][0]
            j = max(range(len(far[key])), key=lambda k: abs(far[key][k]))
            self.assertGreater(abs(far[key][j]), 1.)
            for relative, ok in ((2e-12, True), (3e-11, False), (1e-9, False), (1e-7, False), (1e-3, False)):
                with self.subTest(key=key, relative=relative):
                    cfg, request, response, work = self.prepare("xz")
                    response["results"][0][key][j] *= 1 + relative
                    if ok:
                        self.accepted(cfg, request, response, 60.)
                    else:
                        with self.assertRaisesRegex(BackendError, "Kx、Ky"):
                            common.extract(response, cfg)

    # -- L4 (iv): the |z| term of the phase rounding allowance
    def test_the_z_term_of_the_phase_allowance_is_what_accepts_far_away_rows(self):
        """Rows ~1e7 um away (carrier argument k0 kz0 |z| ~ 3e7): the two programs round k0 kz0 z differently and the honest phase residual is
        1e-8 -- ten times the fixed tolerance. Only the |z| term of the allowance (PHASE_ROUNDING_FACTOR_2D x k0 |kz0| |z|max) accepts it; without that term
        (x and y alone: 6e-14) the honest reply would be refused. A tamper just beyond the allowance is refused, one just inside is not.
        Real program only (the stand-in's Python reference overflows that far away)."""
        if self.PROGRAM != "real":
            self.skipTest("the stand-in's Python reference overflows (0 x inf) 1e7 um away")
        lay = [{"name": "柱", "kind": "patterned2d", "thickness": .3, "y_edges": [0., .2, .6, .8],
                "x_profiles": [{"edges": [0., .8], "n_values": [1.46]}, {"edges": [0., .2, .6, .8], "n_values": [1.46, "2.3+0.13j", 1.46]},
                               {"edges": [0., .8], "n_values": [1.46]}]},
               {"name": "膜", "kind": "uniform", "thickness": .16, "n": 1.46}]
        cfg = normalize_2d({"schema_version": 1, "name": "远处", "mode": "periodic2d",
                            "incident": {"wavelength": .83, "angle_deg": 20., "phi_deg": 30., "polarization": "jones", "M": 2, "My": 2, "psi_deg": 37., "delta_deg": 71.},
                            "structure": {"period": .8, "period_y": .8, "n_incident": 1., "n_substrate": 1.45, "layers": lay},
                            "sampling": {"nx": 40, "nz": 20, "z_min": -1e7, "z_max": 1e7, "slice_plane": "xz", "slice_at": .3}})
        request, response = run_program(self.exe, cfg, self.tmp / "far away", token="far-away")
        work = Path(request["output_dir"])
        far = backend.far_arrays_2d(response["results"][0], 25)
        six, phase = np.load(work / "fields.npy"), np.load(work / "phase.npy")
        mask = np.load(work / "phase_mask.npy").astype(bool)
        x, y, z = (np.load(work / name) for name in ("x.npy", "y.npy", "z.npy"))
        amplitude = abs(six)
        residual = backend.phase_residual_2d(six, amplitude, phase, mask, x, y, z, far, request)
        allowance = backend.phase_rounding_allowance_2d(far, request, x, y, z)
        k0, i0 = 2 * np.pi / request["solve"]["points"][0]["wavelength"], 12
        without_z = backend.PHASE_ROUNDING_FACTOR_2D * k0 * (abs(far["Kx"][i0]) * float(np.abs(x).max()) + abs(far["Ky"][i0]) * float(np.abs(y).max()))
        self.assertGreater(residual, backend.PHASE_TOLERANCE_2D + without_z, "this configuration must need the |z| term")
        self.assertLessEqual(residual, backend.PHASE_TOLERANCE_2D + allowance)
        self.assertGreater(allowance, 1e-7)                                         # the term: k0 kz0 |z| is ~3e7
        self.accepted(cfg, request, response, 60.)
        # tamper one phase value of the component with the largest amplitude, by an angle just beyond / just inside the tolerance + allowance
        c = int(np.argmax(amplitude.max(axis=(1, 2))))
        iz, it = np.unravel_index(int(np.argmax(amplitude[c])), amplitude[c].shape)
        total = backend.PHASE_TOLERANCE_2D + allowance
        for factor, ok in ((.5, True), (3., False)):
            type(self).counter += 1
            copied = self.tmp / f"far{self.counter}"                 # a private copy of the reply (prepare() only copies the class bundles)
            shutil.copytree(work, copied)
            tampered_request, tampered_response = copy.deepcopy(request), copy.deepcopy(response)
            tampered_request["output_dir"] = tampered_response["field_data"]["directory"] = str(copied)
            edited = np.load(copied / "phase.npy")
            edited[0, c, iz, it] += np.sign(-edited[0, c, iz, it] or 1.) * factor * total      # towards zero: stays inside [-pi, pi]
            np.save(copied / "phase.npy", edited, allow_pickle=False)
            with self.subTest(tamper_times_allowance=factor):
                if ok:
                    self.accepted(cfg, tampered_request, tampered_response, 60.)
                else:
                    with self.assertRaisesRegex(BackendError, "相位文件与六复场不自洽"):
                        self.accepted(cfg, tampered_request, tampered_response, 60.)

    # -- L4 (v): the ambiguity band of the propagation flag
    def test_the_propagation_flag_band_is_as_narrow_as_the_documented_one(self):
        """An order within |n^2 - Kx^2 - Ky^2| <= 1e-12 x scale of cutoff may be called propagating by one program and evanescent by the other (rounding),
        so a flag that disagrees with the analytic kz is tolerated THERE and nowhere else (the reviewer's mutant that widened the band to 1e-2 survived).
        The same self-consistent claim 'this order propagates' (kz = 5e-12 + i|kz|, flag, efficiency, Ez, Rp made consistent) is accepted 4e-13 below cutoff
        and refused 1e-8 and 1e-6 below it."""
        def claim_propagating(response, j, cfg):
            far = response["results"][0]
            kz = complex(5e-12, far["kzI"][j][1])
            ex, ey, kx, ky = complex(*far["Ex_r"][j]), complex(*far["Ey_r"][j]), far["Kx"][j], far["Ky"][j]
            nI, theta = float(cfg["structure"]["n_incident"]), math.radians(cfg["incident"]["angle_deg"])
            ez = (kx * ex + ky * ey) / kz
            far["kzI"][j], far["Ez_r"][j], far["prop_r"][j] = [kz.real, kz.imag], [ez.real, ez.imag], 1
            far["DEr"][j] = kz.real * (abs(ex) ** 2 + abs(ey) ** 2 + abs(ez) ** 2) / (nI * math.cos(theta))
            rp = -(kx * ex + ky * ey) / (math.hypot(kx, ky) * kz / nI)
            far["Rp"][j] = [rp.real, rp.imag]
        for eps, inside in ((2e-13, True), (5e-9, False), (5e-7, False)):
            with self.subTest(eps=eps):
                cfg = model_cutoff(eps)
                request, response = run_program(self.exe, cfg, self.tmp / f"cutoff {eps:g}", token="cutoff")
                j = response["orders"].index([1, 0])
                far = response["results"][0]
                self.assertEqual(far["prop_r"][j], 0)
                self.assertAlmostEqual(far["kzI"][j][1] ** 2 / (2 * eps), 1., places=3)          # |s| = 2 eps, the evanescent order of the model
                common.extract(response, cfg)                                                      # honest: accepted
                claim_propagating(response, j, cfg)
                if inside:
                    common.extract(response, cfg)
                else:
                    with self.assertRaisesRegex(BackendError, "解析 kz 的传播判据"):
                        common.extract(response, cfg)

    # -- L5: damaged NPY headers
    def test_a_damaged_npy_header_is_a_backend_error_whatever_numpy_raises(self):
        """numpy parses the header with ast.literal_eval and tokenize: besides OSError / ValueError / EOFError a corrupted header can raise
        tokenize.TokenError (10 of 1500 random corruptions in the round-3 fuzz run), SyntaxError, MemoryError, RecursionError. All of them are
        'this result file cannot be used': a BackendError that names the file, not an exception that leaks out of the validation."""
        import tokenize

        def damaged(key, text):
            def change(directory):
                path = Path(directory) / FILE_NAMES[key]
                raw = path.read_bytes()
                length = int.from_bytes(raw[8:10], "little")
                body = text.encode("latin-1")
                self.assertLess(len(body), length)
                path.write_bytes(raw[:10] + body + b" " * (length - len(body) - 1) + b"\n" + raw[10 + length:])
            return change
        texts = {"unbalanced tuple": "{'descr': '<f8', 'fortran_order': False, 'shape': (3, ",
                 "open brackets": "{'descr': ((((((((((((((((((((",
                 "just a brace": "{", "junk": "\x00\x01 not python at all (",
                 "missing keys": "{'descr': '<f8'}", "a call instead of a literal": "{'descr': '<f8', 'fortran_order': False, 'shape': __import__('os')}",
                 "huge shape": "{'descr': '<f8', 'fortran_order': False, 'shape': (1e999,)}"}
        for key in FILE_KEYS:
            for label, text in texts.items():
                with self.subTest(file=key, header=label):
                    self.refuse(f"{key} header {label}", files=damaged(key, text), words="NPY")
        # whatever numpy raises while it reads the header (so that this test does not depend on the numpy version's parser)
        cfg, request, response, work = self.prepare("xz")
        manifest = response["field_data"]
        for error in (tokenize.TokenError("EOF in multi-line statement", (2, 0)), SyntaxError("bad"), MemoryError(), RecursionError("deep"),
                      UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad"), KeyError("descr"), OverflowError("big")):
            with self.subTest(raised=type(error).__name__), patch.object(np.lib.format, "read_array_header_1_0", side_effect=error):
                with self.assertRaisesRegex(BackendError, "t NPY 读取失败") as caught:
                    backend._load_npy(work, manifest, "t", "<f8", (cfg["sampling"]["nx"],))

    def test_self_consistent_wrong_stack_is_accepted_and_only_python_verification_flags_it(self):
        """The one change the self-consistency checks cannot see, stated honestly: a solver that swaps two layers' thicknesses
        internally and echoes the request in the manifest produces a reply that is consistent with itself (the far field, the
        fields outside the stack, the phase, the metadata). Only an independent solve (RCWA_VERIFY_PYTHON=1) tells the difference."""
        cfg, request, _, _ = self.bundles["xz"]
        folder = self.tmp / "swapped"
        request = backend.build_request(cfg, folder / "out", "swap-token", 1)
        sent = copy.deepcopy(request)
        layers = sent["solve"]["stack"]["layers"]
        layers[0]["d"], layers[1]["d"] = layers[1]["d"], layers[0]["d"]
        folder.mkdir()
        with cpp_env(self.exe):
            response = common.call(self.exe, sent, folder)
        top = 0.0
        for row, layer in zip(response["field_data"]["layers"], request["solve"]["stack"]["layers"]):      # the manifest echoes the request
            row["z_top"], row["z_bottom"] = top, top + layer["d"]
            top += layer["d"]
        values, fields, manifest = self.accepted(cfg, request, response)                      # accepted: nothing inside contradicts itself
        rc2 = load_rcwa2d()
        inc = cfg["incident"]
        reference = rc2.solve2d(build_stack_2d(cfg), inc["wavelength"], inc["angle_deg"], inc["phi_deg"], inc["M"], inc["My"],
                                psi_deg=inc["psi_deg"], delta_deg=inc["delta_deg"])
        attempt = backend.Attempt(cfg)
        attempt.used, attempt.values = True, values
        attempt.verify(reference)                                                              # the optional independent solve
        self.assertEqual(attempt.verification_status, "compared")
        self.assertGreater(attempt.difference, 1e-3)
        self.assertIn("相差超过", attempt.note)


class TamperStandIn(TamperTests, unittest.TestCase):
    PROGRAM = "standin"


class TamperReal(TamperTests, unittest.TestCase):
    PROGRAM = "real"


# ===================================================================================================== 4. fallback paths
class Fallback(Case):
    """Whatever is wrong on the C++ side, the result is the pure Python one, with the reason in backend.note."""

    def python_reference(self):
        cfg = model_pillar("xz", "jones")
        metrics, fields = run_compute(cfg, self.tmp / "python", RCWA_BACKEND="python")
        return cfg, metrics, fields

    def assert_fell_back(self, metrics, fields, reference, *words):
        block = metrics["backend"]
        for key in ("efficiency", "fields", "phase"):
            self.assertEqual(block[key], "python", block["note"])
        self.assertTrue(block["note"])
        for word in words:
            self.assertIn(word, block["note"])
        self.assertIsNone(block["max_efficiency_difference"])
        self.assertTrue(block["python_solve_seconds"] > 0)
        self.assertNotIn("phase_backend", fields)                      # the Python archive carries no C++ phase marks
        self.assertEqual(metrics["orders"], reference[0]["orders"])      # and its numbers are the Python numbers, bit for bit
        for key in COMPONENTS + ("E2", "phase_relative"):
            np.testing.assert_array_equal(fields[key], reference[1][key])

    def test_forced_python_missing_old_override_and_unsupported(self):
        cfg, metrics, fields = self.python_reference()
        reference = (metrics, fields)
        standin = self.program("standin")
        cases = (("forced", dict(RCWA_FIELDS_EXE=standin, RCWA_BACKEND="python"), ("RCWA_BACKEND",)),
                 ("missing", dict(RCWA_FIELDS_EXE=self.tmp / "不存在"), ("不存在",)),
                 ("old override", dict(RCWA_FIELDS_EXE=None, RCWA_JSON_EXE=self.tmp / "old_far"), ("旧效率协议",)))
        for name, env, words in cases:
            with self.subTest(case=name):
                values = dict(RCWA_BACKEND=None, RCWA_JSON_EXE=None, RCWA_VERIFY_PYTHON=None)
                values.update(env)
                got, got_fields = run_compute(cfg, self.tmp / name.replace(" ", "_"), **values)
                self.assert_fell_back(got, got_fields, reference, *words)
                self.assertFalse((self.tmp / name.replace(" ", "_") / "cpp_request.json").exists())

    def test_programs_that_fail_in_every_way(self):
        cfg, metrics, fields = self.python_reference()
        fake = make_fake(self.tmp)
        pidfile = self.tmp / "fake.pid"
        cases = (("refuse", {}, ("不接受", "没有编入圆锥入射")), ("crash", {}, ("返回码 3",)), ("garbage", {}, ("应答",)),
                 ("null", {}, ("截断阶数",)), ("sleep", {"RCWA_CPP_TIMEOUT": ".4", "RCWA_FAKE_PIDFILE": pidfile}, ("没有返回", "已终止")))
        for mode, extra, words in cases:
            with self.subTest(mode=mode):
                with cpp_env(fake, RCWA_FAKE_MODE=mode, **extra):
                    got, got_fields = run_compute(cfg, self.tmp / mode)
                self.assert_fell_back(got, got_fields, (metrics, fields), *words)
                self.assertIsNotNone(got["backend"]["cpp_seconds"])
                self.assertTrue((self.tmp / mode / "cpp_request.json").is_file())
        self.assertTrue(process_alive(int(pidfile.read_text())) is False)

    def test_standin_failures_after_a_valid_answer(self):
        """Reply status not ok, or any item of the reply wrong: one Python solve, Python numbers."""
        cfg, metrics, fields = self.python_reference()
        exe, real_call = self.program("standin"), common.call
        rc2 = load_rcwa2d()

        def failing(change):
            def call(exe, request, folder, timeout=None):
                response = real_call(exe, request, folder, timeout)
                change(response)
                return response
            return call

        edits = {"status": lambda r: r["results"][0].update(status="numerical_failure", message="本征迭代未收敛"),
                 "orders": lambda r: r["orders"].reverse(),
                 "request_id": lambda r: r["field_data"].update(request_id="stale"),
                 "slice": lambda r: r["field_data"]["slice"].update(plane="yz"),
                 "file": lambda r: (Path(r["field_data"]["directory"]) / "fields.npy").write_bytes(b"junk")}
        for name, change in edits.items():
            with self.subTest(case=name), cpp_env(exe), patch.object(common, "call", side_effect=failing(change)), \
                    patch.object(rc2, "solve2d", wraps=rc2.solve2d) as solver:
                got, got_fields = run_compute(cfg, self.tmp / ("tampered " + name))
                self.assertEqual(solver.call_count, 1, "fallback performs exactly one Python solve")
                self.assert_fell_back(got, got_fields, (metrics, fields))
                self.assertEqual(list((self.tmp / ("tampered " + name)).glob("cpp_fields-*")), [])   # the rejected result is not kept

    def test_rayleigh_anomaly_falls_back_and_python_reports_it(self):
        """A grazing order: C++ refuses (protocol error), Python then raises its own Chinese message unchanged."""
        cfg = normalize_2d({
            "schema_version": 1, "name": "Rayleigh", "mode": "periodic2d",
            "incident": {"wavelength": 1., "angle_deg": 0., "phi_deg": 0., "polarization": "s", "M": 1, "My": 1},
            "structure": {"period": 1., "period_y": 1., "n_incident": 1., "n_substrate": 1.5,
                          "layers": [{"name": "膜", "kind": "uniform", "thickness": .3, "n": 1.8}]},
            "sampling": {"nx": 40, "nz": 20, "z_min": -.1, "z_max": .5, "slice_plane": "xz", "slice_at": .3}})
        rc2 = load_rcwa2d()
        with EnvPatch(RCWA_BACKEND="python"), self.assertRaises(rc2.RayleighAnomalyError) as python_error:
            gui_worker.compute(copy.deepcopy(cfg), self.tmp / "python")
        for name in ("standin", "real"):
            if STATE[name] is None:
                continue
            with self.subTest(program=name), cpp_env(STATE[name]):
                (self.tmp / ("attempt " + name)).mkdir()
                attempt = backend.attempt(copy.deepcopy(cfg), self.tmp / ("attempt " + name))
                self.assertFalse(attempt.used)
                self.assertIn("不接受", attempt.note)
                self.assertIn("入射区", attempt.note)
                self.assertEqual(list((self.tmp / ("attempt " + name)).glob("cpp_fields-*")), [])
                with self.assertRaises(rc2.RayleighAnomalyError) as raised:
                    gui_worker.compute(copy.deepcopy(cfg), self.tmp / ("compute " + name))
                self.assertEqual(str(raised.exception), str(python_error.exception))
                self.assertIn("Rayleigh", str(raised.exception))
                self.assertFalse((self.tmp / ("compute " + name) / "metrics.json").exists())
                self.assertEqual(list((self.tmp / ("compute " + name)).glob("cpp_fields-*")), [])    # no half-published directory

    def test_farfield_status_of_the_far_protocol(self):
        """The far-field (rcwa_json) protocol reports Rayleigh as status/null with ok true; extract turns it into a refusal."""
        cfg = model_pillar()
        request = common.build_request(cfg)
        request["points"][0].update(theta_deg=0., phi_deg=0., wavelength=1.)       # lambda = Lx: orders (+-1, n) graze
        cfg["structure"].update(period=1.)
        cfg["incident"].update(angle_deg=0., phi_deg=0., wavelength=1.)
        code, out, _ = common._run_child([str(self.program("standin"))], 120, text=json.dumps(request))
        reply = json.loads(out)
        self.assertEqual((code, reply["ok"], reply["results"][0]["status"]), (0, True, "rayleigh_anomaly"))
        self.assertIsNone(reply["results"][0]["DEr"][0])
        with self.assertRaisesRegex(BackendError, "rayleigh_anomaly"):
            common.extract(reply, cfg)

    def test_legacy_far_api_handles_2d_and_verifies(self):
        """cpp_backend.attempt / merge (the old efficiency API) now also route periodic2d."""
        cfg = model_pillar("xz", "jones")
        exe = self.program("standin")
        with EnvPatch(RCWA_JSON_EXE=exe, RCWA_BACKEND=None, RCWA_2D_BACKEND="cpp"):
            attempt = common.attempt(cfg, self.tmp)
        self.assertTrue(attempt.used, attempt.note)
        reference = load_rcwa2d().solve2d(build_stack_2d(cfg), cfg["incident"]["wavelength"], cfg["incident"]["angle_deg"],
                                          cfg["incident"]["phi_deg"], 2, 2, psi_deg=33., delta_deg=50.)
        merged = attempt.merge(reference)
        self.assertTrue(attempt.used, attempt.note)
        self.assertEqual(attempt.difference, 0.0)
        np.testing.assert_array_equal(merged["DEr"], reference["DEr"])
        reversed_reference = type(reference)(reference)
        reversed_reference["mn"] = reference["mn"][::-1]
        attempt.used = True
        attempt.merge(reversed_reference)
        self.assertFalse(attempt.used)
        self.assertIn("级次排列", attempt.note)


# ===================================================================================================== 5. worker end to end
def _angle_equal(a, b, tolerance):
    both = np.isfinite(a) & np.isfinite(b)
    np.testing.assert_array_equal(np.isfinite(a), np.isfinite(b))
    if both.any():
        np.testing.assert_allclose(np.exp(1j * a[both]), np.exp(1j * b[both]), rtol=0, atol=tolerance)


def assert_isomorphic(test, python, cpp, exact, label=""):
    """The saved result of the C++ path has the Python path's keys, dtypes and shapes (plus the versioned phase marks)."""
    (metrics_py, fields_py), (metrics_cpp, fields_cpp) = python, cpp
    test.assertEqual(set(fields_cpp) - set(fields_py), CPP_EXTRA_KEYS, label)
    test.assertEqual(set(fields_py) - set(fields_cpp), set(), label)
    tolerance = dict(rtol=0, atol=0) if exact else dict(rtol=FROZEN_TOLERANCE, atol=FROZEN_TOLERANCE)
    worst = {}
    for key, expected in fields_py.items():
        actual = fields_cpp[key]
        test.assertEqual((actual.dtype, actual.shape), (expected.dtype, expected.shape), f"{label} {key}")
        if key == "phase_relative":
            continue
        if key == "orders" or expected.dtype.kind in "US":
            np.testing.assert_array_equal(actual, expected, err_msg=f"{label} {key}")
        elif exact:
            np.testing.assert_array_equal(actual, expected, err_msg=f"{label} {key}")
        else:
            np.testing.assert_allclose(actual, expected, err_msg=f"{label} {key}", **tolerance)
            if expected.dtype.kind in "fc" and expected.ndim:
                worst[key] = float(np.max(abs(actual - expected) / (1 + abs(expected))))
    # phase: same mask pattern except where a weak point sits at the rule's edge; angles agree wherever the field is not weak
    py, cp = fields_py["phase_relative"], fields_cpp["phase_relative"]
    if exact:
        np.testing.assert_array_equal(np.isnan(py), np.isnan(cp))
        _angle_equal(py, cp, 1e-12)
    else:
        field = np.stack([fields_py[k] for k in COMPONENTS])
        main = int(np.argmax([np.nanmax(abs(fields_py[k])) for k in COMPONENTS[:3]]))
        strong = abs(field[main]) > 100 * (FROZEN_TOLERANCE + FROZEN_TOLERANCE * abs(field[main]))
        both = np.isfinite(py) & np.isfinite(cp) & strong
        magnitude = abs(field[main][both])
        gap = abs(np.exp(1j * py[both]) - np.exp(1j * cp[both]))
        test.assertTrue(np.all(gap <= 3 * FROZEN_TOLERANCE * (1 + 1 / magnitude) + 5e-14),
                        f"{label}: phase differs by {float(gap.max()):.2e}")
        test.assertFalse((np.isnan(cp) & strong).any(), label)
    # metrics
    skip = {"backend", "timings", "total_seconds", "solve_seconds", "orders", "R_total", "T_total", "absorption",
            "sampled_E2_peak", "peak_x", "peak_y", "peak_z"}
    for key in set(metrics_py) - skip:
        test.assertEqual(metrics_cpp[key], metrics_py[key], f"{label} metrics[{key}]")
    rows_py = {(r["m"], r["n"]): r for r in metrics_py["orders"]}
    rows_cpp = {(r["m"], r["n"]): r for r in metrics_cpp["orders"]}
    test.assertEqual(set(metrics_cpp["orders"][0]) if metrics_cpp["orders"] else set(),
                     set(metrics_py["orders"][0]) if metrics_py["orders"] else set())
    for key in set(rows_py) | set(rows_cpp):
        if key in rows_py and key in rows_cpp:
            if exact:
                test.assertEqual(rows_cpp[key], rows_py[key], f"{label} order {key}")
            else:
                for name in ("R", "T", "R_s", "R_p", "T_s", "T_p"):
                    test.assertAlmostEqual(rows_cpp[key][name], rows_py[key][name], delta=2 * FROZEN_TOLERANCE, msg=f"{label} {key} {name}")
        else:        # a row at the 1e-9 listing threshold may appear on one side only
            row = rows_py.get(key) or rows_cpp.get(key)
            test.assertLess(row["R"] + row["T"], 1e-9 + 4 * FROZEN_TOLERANCE, f"{label} order {key}")
    for key in ("R_total", "T_total", "absorption", "sampled_E2_peak"):
        if exact:
            test.assertEqual(metrics_cpp[key], metrics_py[key], f"{label} {key}")
        else:
            test.assertAlmostEqual(metrics_cpp[key], metrics_py[key], delta=1e-8 * max(1., abs(metrics_py[key])), msg=f"{label} {key}")
    block = metrics_cpp["backend"]
    for key in ("efficiency", "fields", "phase"):
        test.assertEqual(block[key], "cpp", block["note"])
    test.assertEqual(block["note"], "")
    test.assertEqual((block["python_solve_seconds"], block["python_solve_ran"]), (0., False))
    test.assertIsNone(block["max_efficiency_difference"])
    test.assertEqual(block["verification_status"], "not_requested")
    test.assertEqual((block["protocol"], block["protocol_version"], block["phase_version"]), ("stablercwa-fields", 1, 1))
    test.assertEqual((block["cpp_zero_modes"], block["cpp_eig_info"]), (0, 0))
    test.assertGreater(block["cpp_seconds"], 0.)
    test.assertEqual(set(metrics_py["backend"]) - {"efficiency", "fields", "cpp_seconds", "python_solve_seconds",
                                                    "max_efficiency_difference", "note", "python_n_schur",
                                                    "python_zero_modes"}, {"phase", "protocol", "protocol_version",
                                                                           "phase_version", "python_solve_ran",
                                                                           "verification_status"})
    test.assertEqual(metrics_cpp["timings"]["python_solve_seconds"], 0.)
    return worst


class WorkerEndToEnd:
    PROGRAM = "standin"
    EXACT = True
    RESULTS = {}

    @classmethod
    def setUpClass(cls):
        cls.exe = Case.program(cls.PROGRAM)
        cls.tmp = Path(tempfile.mkdtemp(prefix="rcwa 中文 e2e "))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def both(self, name, cfg):
        python = run_compute(cfg, self.tmp / (name.replace(" ", "_") + " py"), RCWA_BACKEND="python")
        with cpp_env(self.exe):
            cpp = run_compute(cfg, self.tmp / (name.replace(" ", "_") + " cpp"))
        return python, cpp

    def test_worker_models_match_the_python_path(self):
        overall = {}
        for name, build in WORKER_MODELS + (("y-invariant s, phi=0", model_y_invariant),):
            with self.subTest(model=name):
                cfg = build()
                python, cpp = self.both(name, cfg)
                for key, value in assert_isomorphic(self, python, cpp, self.EXACT, name).items():
                    overall[key] = max(overall.get(key, 0.), value)
                self.assertGreaterEqual(len(cpp[0]["orders"]), 1)
                for row in cpp[0]["orders"]:
                    self.assertAlmostEqual(row["R"], row["R_s"] + row["R_p"], places=10)
                    self.assertAlmostEqual(row["T"], row["T_s"] + row["T_p"], places=10)
                self.assertEqual(cpp[0]["field_plane"], cfg["sampling"]["slice_plane"])
                self.assertEqual(cpp[0]["harmonic_count"], (2 * cfg["incident"]["M"] + 1) * (2 * cfg["incident"]["My"] + 1))
        if overall:
            say(f"\n    [{self.PROGRAM}] largest |a-b|/(1+|b|) over six models, per saved array: "
                + ", ".join(f"{k} {v:.1e}" for k, v in sorted(overall.items()) if v > 0))

    def test_no_python_numerics_on_the_success_path(self):
        import field_phase
        rc2 = load_rcwa2d()
        with cpp_env(self.exe), patch.object(rc2, "solve2d", side_effect=AssertionError("Python solve")), \
                patch.object(rc2, "field_at", side_effect=AssertionError("Python field_at")), \
                patch.object(rc2, "field_slice", side_effect=AssertionError("Python field_slice")), \
                patch.object(field_phase, "phase_data", side_effect=AssertionError("Python phase")), \
                patch.object(np, "angle", side_effect=AssertionError("np.angle")):
            for plane in ("xz", "yz"):
                metrics, fields = run_compute(model_pillar(plane, "jones"), self.tmp / plane)
                self.assertEqual(metrics["backend"]["fields"], "cpp", metrics["backend"]["note"])
                self.assertEqual(metrics["backend"]["python_solve_seconds"], 0.)
                self.assertEqual(metrics["timings"]["python_solve_seconds"], 0.)
                self.assertEqual(metrics["timings"]["solve_seconds"], metrics["solve_seconds"])
                for component in COMPONENTS:
                    for quantity in ("phase", "phase_relative"):
                        shown, info = gui_fields.view_data(fields, model_pillar(plane, "jones"), component, quantity)
                        self.assertEqual(info["phase_backend"], "cpp")
                        np.testing.assert_array_equal(shown, np.where(fields["phase_masks"][int(quantity == "phase_relative"), COMPONENTS.index(component)],
                                                                      np.nan, fields["phase_values"][int(quantity == "phase_relative"), COMPONENTS.index(component)]))

    def test_explicit_verification(self):
        cfg = model_pillar("yz", "p")
        with cpp_env(self.exe, RCWA_VERIFY_PYTHON="1"):
            metrics, fields = run_compute(cfg, self.tmp / "verify")
        block = metrics["backend"]
        self.assertEqual((block["fields"], block["efficiency"], block["verification_status"]), ("cpp", "cpp", "compared"), block["note"])
        self.assertTrue(block["python_solve_ran"])
        self.assertGreater(block["python_solve_seconds"], 0.)
        self.assertLessEqual(block["max_efficiency_difference"], 1e-12 if self.EXACT else FROZEN_TOLERANCE)
        if self.EXACT:
            self.assertEqual(block["max_efficiency_difference"], 0.)
        self.assertEqual(block["note"], "")
        # a failing comparison keeps the valid C++ result
        rc2 = load_rcwa2d()
        with cpp_env(self.exe, RCWA_VERIFY_PYTHON="1"), patch.object(rc2, "solve2d", side_effect=rc2.RayleighAnomalyError("oracle cannot solve")):
            metrics, fields = run_compute(cfg, self.tmp / "verify failed")
        block = metrics["backend"]
        self.assertEqual((block["fields"], block["verification_status"]), ("cpp", "failed"))
        self.assertIn("保留 C++", block["note"])
        self.assertIsNone(block["max_efficiency_difference"])
        self.assertTrue(np.isfinite(fields["E2"]).all())

    def test_saved_result_is_redisplayed_and_exported_without_solving(self):
        cfg = model_absorbing()
        with cpp_env(self.exe):
            metrics, fields = run_compute(cfg, self.tmp / "saved")
        rc2 = load_rcwa2d()
        with patch.object(rc2, "solve2d", side_effect=AssertionError("solve")), patch.object(np, "angle", side_effect=AssertionError("angle")):
            for component, quantity in (("E2", "abs2"), ("Hz", "phase_relative"), ("Ex", "phase"), ("Ey", "log_magnitude"), ("Hy", "real")):
                info = gui_fields.render_view(self.tmp / "saved", component, quantity)
                self.assertTrue((self.tmp / "saved" / info["filename"]).stat().st_size > 1000, (component, quantity))
                self.assertIn("固定x=0.37", info["note"])
                if quantity.startswith("phase"):
                    self.assertEqual(info["phase_backend"], "cpp")
            exported = gui_fields.export_six_fields(self.tmp / "saved")
        with np.load(exported, allow_pickle=False) as archive:
            self.assertEqual(int(archive["phase_archive_version"]), 1)
            np.testing.assert_array_equal(archive["phase_masks"], fields["phase_masks"])
            np.testing.assert_array_equal(archive["orders"], fields["orders"])
            self.assertEqual(set(archive.files) - set(fields), {"H2", "E_incident_amplitude", "H_incident_amplitude"} - set(fields))
        with np.load(self.tmp / "saved" / "fields.npz", allow_pickle=False) as archive:       # the file is closed and complete
            self.assertEqual(sorted(archive.files), sorted(fields))
        kept = sorted(path.name for path in (self.tmp / "saved").iterdir() if path.name.startswith("cpp_fields-"))
        self.assertEqual(len(kept), 1, kept)                                                    # the accepted call keeps its raw NPY directory, as before
        self.assertEqual(sorted(path.name for path in (self.tmp / "saved" / kept[0]).iterdir()),
                         ["fields.npy", "phase.npy", "phase_mask.npy", "t.npy", "x.npy", "y.npy", "z.npy"])
        self.assertEqual(json.loads((self.tmp / "saved" / "model.json").read_text(encoding="utf-8"))["mode"], "periodic2d")

    def test_sections_equal_field_slice_and_conserve_energy(self):
        """The C++ twin of test_gui_2d's section test: both slice directions against rcwa2d.field_slice, energy, power channels."""
        rc2 = load_rcwa2d()
        for plane, at in (("xz", .4), ("yz", .5)):
            with self.subTest(plane=plane):
                cfg = model_pillar(plane, "s")
                cfg["sampling"].update(nx=45, nz=25, slice_at=at)
                with cpp_env(self.exe):
                    metrics, fields = run_compute(cfg, self.tmp / ("section " + plane))
                self.assertEqual(metrics["backend"]["fields"], "cpp", metrics["backend"]["note"])
                inc, sam = cfg["incident"], cfg["sampling"]
                res = rc2.solve2d(build_stack_2d(cfg), inc["wavelength"], inc["angle_deg"], inc["phi_deg"], inc["M"], inc["My"], psi_deg=inc["psi_deg"])
                reference = rc2.field_slice(res, np.linspace(sam["z_min"], sam["z_max"], sam["nz"]), axis=plane[0], at=at, npts=sam["nx"])
                for key in COMPONENTS:
                    np.testing.assert_allclose(fields[key], reference[key], atol=3e-12 if self.EXACT else FROZEN_TOLERANCE,
                                               rtol=0 if self.EXACT else FROZEN_TOLERANCE)
                self.assertAlmostEqual(metrics["R_total"] + metrics["T_total"], 1., places=10)
                for row in metrics["orders"]:
                    self.assertAlmostEqual(row["R"], row["R_s"] + row["R_p"], places=10)
                    self.assertAlmostEqual(row["T"], row["T_s"] + row["T_p"], places=10)
                if plane == "yz":
                    np.testing.assert_allclose(fields["y"], np.arange(45) * .8 / 45)
                    np.testing.assert_allclose(fields["x"], at)

    def test_uniform_medium_incident_normalisation_and_phase(self):
        """test_gui_2d's analytic check through the C++ path: |E|^2 = |H|^2/n^2 = 1 and the carrier-free phase of each Jones component."""
        rc2 = load_rcwa2d()
        n, th, ph = 1.4, np.radians(20.), np.radians(30.)
        stack = rc2.Stack2D(n, [rc2.Layer2D(.3, n=n)], n, rc2.Lattice(1., .8))
        cfg = gui_models.from_stack(stack, lam0=1., theta_deg=20, phi_deg=30, pol="jones", psi_deg=45, delta_deg=90, M=1, My=1)
        cfg["sampling"].update(nx=40, nz=20, z_min=-.1, z_max=.4, slice_plane="yz", slice_at=.37)
        with cpp_env(self.exe):
            metrics, fields = run_compute(cfg, self.tmp / "uniform")
        self.assertEqual(metrics["backend"]["fields"], "cpp", metrics["backend"]["note"])
        tolerance = 4e-12 if self.EXACT else 1e-9
        for key in ("E2", "H2"):
            shown, _ = gui_fields.view_data(fields, cfg, key)
            np.testing.assert_allclose(shown, 1., atol=tolerance)
        e = (1j * np.array([-np.sin(ph), np.cos(ph), 0.]) + np.array([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), np.sin(th)])) / np.sqrt(2)
        for key, expected in zip(COMPONENTS[:3], e):
            shown, info = gui_fields.view_data(fields, cfg, key, "phase_relative")
            self.assertEqual(info["phase_backend"], "cpp")
            np.testing.assert_allclose(np.exp(1j * shown), np.exp(1j * np.angle(expected)), atol=tolerance)

    def test_thread_count_does_not_change_the_result(self):
        """The GUI passes the worker's thread count; two structured layers also switch parallel_layers on. Same bits."""
        cfg = model_pillar("xz", "jones")
        cfg["structure"]["layers"].append(copy.deepcopy(cfg["structure"]["layers"][0]))
        cfg = normalize_2d(cfg)
        runs = {}
        for threads in (1, 4):
            request, response = run_program(self.exe, cfg, self.tmp / f"threads {threads}", threads=threads, token=f"threads-{threads}")
            self.assertEqual(response["threads"], threads)
            self.assertEqual(request["solve"]["options"]["parallel_layers"], threads > 1)
            fields, _ = backend.extract_fields(response, cfg, request)
            runs[threads] = (common.extract(response, cfg), fields)
        for key in ("DEr", "DEt", "Rs", "Tp", "Ez_t"):
            self.assertEqual(runs[1][0][key], runs[4][0][key], key)
        for key in COMPONENTS + ("E2", "phase_values", "phase_masks"):
            np.testing.assert_array_equal(runs[1][1][key], runs[4][1][key], err_msg=key)

    def test_legacy_python_archive_still_displays(self):
        """A 2-D NPZ saved by the Python path (no phase marks) is displayed by the same code, as before."""
        cfg = model_pillar("yz", "s")
        _, fields = run_compute(cfg, self.tmp / "legacy", RCWA_BACKEND="python")
        self.assertNotIn("phase_backend", fields)
        main = max(COMPONENTS[:3], key=lambda k: float(np.nanmax(abs(fields[k]))))
        shown, info = gui_fields.view_data(fields, cfg, main, "phase_relative")
        self.assertEqual(info["phase_backend"], "python")
        np.testing.assert_array_equal(shown, fields["phase_relative"])      # what the worker stored is what the display rule gives
        self.assertTrue(gui_fields.render_view(self.tmp / "legacy", "Hz", "phase")["filename"])


class WorkerEndToEndStandIn(WorkerEndToEnd, unittest.TestCase):
    PROGRAM, EXACT = "standin", EXACT_BITS          # same solver code, same inputs: identical bits (single-threaded BLAS)


class WorkerEndToEndReal(WorkerEndToEnd, unittest.TestCase):
    PROGRAM, EXACT = "real", False


# ===================================================================================================== 6. frozen reference
RAYLEIGH_IDS = ("fail_rayleigh_incident", "fail_rayleigh_exit", "fail_rayleigh_layer")


class FrozenEndToEnd:
    """The frozen 2-D corpus (gui/validation/field_reference_2d) through the GUI request/validation path.

    Inputs: the frozen case as a GUI model (cases the GUI cannot express are skipped, see FrozenRequests), z list of the
    case, nt = the case's nt, both sections. Outputs: far-field complex amplitudes, six components, E2 and the two phases
    compared with the frozen NPZ at the frozen tolerance (|a-b| <= atol + rtol |b|, complex, not just |.|^2).
    N = 625 (pillar_big) runs only with RCWA2D_BIG=1 (about a minute per section for the Python stand-in)."""
    PROGRAM = "standin"

    @classmethod
    def setUpClass(cls):
        cls.exe = Case.program(cls.PROGRAM)
        cls.tmp = Path(tempfile.mkdtemp(prefix="rcwa 中文 frozen "))
        cls.baseline = np.load(frozen_cases.FROZEN_DIR / "reference_2d_v1.npz", allow_pickle=False)
        cls.worst = {"far": 0., "near": 0., "phase": 0.}

    @classmethod
    def tearDownClass(cls):
        cls.baseline.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)
        say(f"\n    [{cls.PROGRAM}] frozen reference, largest |a-b|: far {cls.worst['far']:.2e}, fields {cls.worst['near']:.2e}, "
            f"phase-phasor {cls.worst['phase']:.2e}")

    def run_case(self, spec, plane):
        cfg = spec_cfg(spec, plane)
        request, _ = None, None
        folder = self.tmp / f"{spec['id']} {plane}"
        folder.mkdir()
        request = backend.build_request(cfg, folder / "out", f"frozen-{spec['id']}-{plane}", 1)
        zs, nt = spec["zs"], spec["slice"]["nt"]
        request["sampling"].update(z=list(zs), nt=nt)                       # the corpus samples its own explicit depths and nt
        cfg["sampling"].update(nx=nt, nz=len(zs), z_min=zs[0], z_max=zs[-1])
        with cpp_env(self.exe):
            response = common.call(self.exe, request, folder)
        return cfg, request, response

    def test_frozen_successes(self):
        base = self.baseline
        count = 0
        for spec in frozen_cases.case_specs():
            if not expressible(spec) or spec["status"] != "ok" or (spec["id"] == "pillar_big" and not os.environ.get("RCWA2D_BIG")):
                continue
            case, tol = spec["id"], spec["tolerance"]
            for plane in ("xz", "yz"):
                with self.subTest(case=case, plane=plane):
                    cfg, request, response = self.run_case(spec, plane)
                    far = common.extract(response, cfg)
                    # the corpus samples its own explicit depths (not a linspace of the model): no hidden probe rows here
                    with patch.object(backend, "sampling_plan_2d", lambda c, zs=spec["zs"]: (np.asarray(zs, float), None, None)):
                        fields, manifest = backend.extract_fields(response, cfg, request)
                    count += 1
                    # far field: complex amplitudes and efficiencies
                    np.testing.assert_array_equal(np.asarray(far["mn"]), base[case + "__mn"])
                    for key in ("Rs", "Rp", "Ts", "Tp", "kzI", "kzII", "Ex_r", "Ey_r", "Ez_r", "Ex_t", "Ey_t", "Ez_t", "DEr", "DEt", "Kx", "Ky"):
                        actual, expected = np.asarray(far[key]), base[f"{case}__{key}"]
                        np.testing.assert_allclose(actual, expected, atol=tol["far"]["atol"], rtol=tol["far"]["rtol"], err_msg=f"{case} {key}")
                        type(self).worst["far"] = max(self.worst["far"], float(np.max(abs(actual - expected))))
                    np.testing.assert_array_equal(np.asarray(far["prop_r"]), base[case + "__prop_r"])
                    np.testing.assert_array_equal(np.asarray(far["prop_t"]), base[case + "__prop_t"])
                    # near field on the section
                    expected = base[f"{case}__fields_{plane}"]
                    actual = np.stack([fields[key] for key in COMPONENTS])
                    np.testing.assert_allclose(actual, expected, atol=tol["near"]["atol"], rtol=tol["near"]["rtol"], err_msg=case)
                    type(self).worst["near"] = max(self.worst["near"], float(np.max(abs(actual - expected))))
                    np.testing.assert_allclose(fields["E2"], base[f"{case}__E2_{plane}"], atol=10 * tol["near"]["atol"],
                                               rtol=10 * tol["near"]["rtol"], err_msg=case)
                    np.testing.assert_allclose(fields["t"], base[f"{case}__t_{plane}"], rtol=1e-14, atol=1e-14)
                    fixed = float(base[f"{case}__fixed_{plane}"])
                    np.testing.assert_array_equal(fields["y" if plane == "xz" else "x"], np.full(len(fields["t"]), fixed))
                    np.testing.assert_array_equal(fields["z"], base[case + "__z"])
                    # phase: absolute and incident carrier, wherever the frozen comparison mask allows it
                    comparable = ~base[f"{case}__phase_comparison_mask_{plane}"]
                    phases, masks = fields["phase_values"], fields["phase_masks"]
                    self.assertFalse(masks[:, comparable].any(), case)
                    allowed = tol["near"]["atol"] + tol["near"]["rtol"] * abs(expected)
                    for index, name in enumerate(("phase_absolute", "phase_incident")):
                        gap = abs(np.exp(1j * phases[index][comparable]) - np.exp(1j * base[f"{case}__{name}_{plane}"][comparable]))
                        self.assertTrue(np.all(gap <= 2 * allowed[comparable] / abs(expected[comparable]) + 5e-14), (case, name))
                        type(self).worst["phase"] = max(self.worst["phase"], float(gap.max()))
                    # the frozen weak-field mask is the GUI rule: the program's mask equals it where the frozen field is not at the edge
                    np.testing.assert_array_equal(masks[0][comparable], base[f"{case}__phase_mask_{plane}"][comparable])
                    # the manifest is exactly what the protocol says for this case
                    self.assertEqual(manifest["harmonics"], {"M": spec["M1"], "My": spec["M2"], "N": spec["N"]})
                    self.assertEqual(manifest["slice"], {"plane": plane, "at": fixed,
                                                         "period": spec["stack"]["Lx"] if plane == "xz" else spec["stack"]["Ly"]})
        self.assertGreaterEqual(count, 36)

    def test_frozen_rayleigh_failures(self):
        for spec in frozen_cases.case_specs():
            if spec["id"] not in RAYLEIGH_IDS:
                continue
            with self.subTest(case=spec["id"]):
                cfg = spec_cfg(spec)
                folder = self.tmp / spec["id"]
                folder.mkdir()
                with cpp_env(self.exe):
                    attempt = backend.attempt(cfg, folder)
                self.assertFalse(attempt.used)
                self.assertIn("不接受", attempt.note)
                self.assertIn(spec["expect"]["region"].replace("第", "第"), attempt.note.replace(" ", ""))
                self.assertEqual(list(folder.glob("cpp_fields-*")), [])
                # the far-field protocol's own answer: ok true, status rayleigh_anomaly, orders listed in the message
                far_exe = STATE["standin"] if self.PROGRAM == "standin" else STATE["real_far"]
                if far_exe is None:
                    say(f"      [{spec['id']}] far-field protocol check skipped: {STATE['real_far_note']}")
                    continue
                request = common.build_request(cfg)
                code, out, _ = common._run_child([str(far_exe)], 60, text=json.dumps(request))
                reply = json.loads(out)
                self.assertEqual((reply["ok"], reply["results"][0]["status"]), (True, "rayleigh_anomaly"))
                with self.assertRaisesRegex(BackendError, "rayleigh_anomaly"):
                    common.extract(reply, cfg)
                message = reply["results"][0]["message"].replace(" ", "")
                for m, n in spec["expect"]["orders"]:
                    self.assertIn(f"({m},{n})", message)


class FrozenEndToEndStandIn(FrozenEndToEnd, unittest.TestCase):
    PROGRAM = "standin"


class FrozenEndToEndReal(FrozenEndToEnd, unittest.TestCase):
    PROGRAM = "real"


# ===================================================================================================== 7. GUI server: run, save, redisplay
class GuiServer:
    """/api/run with a 2-D model, status, saved files, plots, history and re-opening a saved result in a fresh server."""
    PROGRAM = "standin"

    @classmethod
    def setUpClass(cls):
        cls.exe = Case.program(cls.PROGRAM)
        cls.server = Server(RCWA_FIELDS_EXE=cls.exe, RCWA_BACKEND="", RCWA_JSON_EXE="", RCWA_VERIFY_PYTHON="", RCWA_CPP_TIMEOUT="", RCWA_2D_BACKEND="cpp")
        cls.addClassCleanup(cls.server.close)

    def test_run_plot_save_and_reopen(self):
        cfg = model_pillar("yz", "p")
        job, status = self.server.run(cfg)
        self.assertEqual(status["status"], "done", status.get("error"))
        block = status["metrics"]["backend"]
        for key in ("efficiency", "fields", "phase"):
            self.assertEqual(block[key], "cpp", block["note"])
        self.assertEqual(block["python_solve_seconds"], 0.)
        self.assertEqual(status["metrics"]["field_plane"], "yz")
        self.assertEqual(status["metrics"]["harmonic_count"], 25)
        self.assertEqual(status["config"]["mode"], "periodic2d")
        self.assertEqual(status["metrics"], json.loads(self.server.get(f"/api/jobs/{job}/files/metrics.json")))
        # the saved archive and the model that produced it
        with np.load(io.BytesIO(self.server.get(f"/api/jobs/{job}/files/fields.npz")), allow_pickle=False) as archive:
            self.assertEqual(archive["Hy"].shape, (20, 40))
            self.assertEqual(str(archive["slice_plane"]), "yz")
            self.assertEqual(archive["phase_values"].shape, (2, 6, 20, 40))
            np.testing.assert_allclose(archive["y"], np.arange(40) * .8 / 40)
            np.testing.assert_array_equal(archive["x"], np.full(40, .5))
            saved = {key: archive[key] for key in archive.files}
        self.assertEqual(json.loads(self.server.get(f"/api/jobs/{job}/files/model.json"))["incident"]["polarization"], "p")
        # views are drawn from the saved complex fields (no solver): phase from the stored C++ arrays
        for component, quantity in (("Hz", "phase_relative"), ("Ey", "phase"), ("E2", "abs2"), ("Hx", "magnitude")):
            info = self.server.post("/api/plot", {"id": job, "component": component, "quantity": quantity, "source": "total"})
            self.assertIn("固定x=0.5", info["note"])
            if quantity.startswith("phase"):
                self.assertEqual(info["phase_backend"], "cpp")
            png = self.server.get(f"/api/jobs/{job}/files/{info['filename']}")
            self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
            self.assertGreater(len(png), 1000)
        again = self.server.post("/api/plot", {"id": job, "component": "Hz", "quantity": "phase_relative", "source": "total"})
        self.assertTrue(again["cache_hit"])
        six = self.server.get(f"/api/jobs/{job}/files/fields_six.npz")
        with np.load(io.BytesIO(six), allow_pickle=False) as archive:
            self.assertEqual(int(archive["phase_archive_version"]), 1)
            np.testing.assert_array_equal(archive["phase_masks"], saved["phase_masks"])
            self.assertIn("H_incident_amplitude", archive.files)
        self.assertIn(job, [row["id"] for row in json.loads(self.server.get("/api/history"))])
        row = next(row for row in json.loads(self.server.get("/api/history")) if row["id"] == job)
        self.assertEqual((row["mode"], row["incident"]["My"]), ("periodic2d", 2))
        # a second server process re-opens the saved result from disk and redraws it
        second = Server(RCWA_FIELDS_EXE=self.exe, RCWA_BACKEND="", RCWA_JSON_EXE="", RCWA_2D_BACKEND="cpp")
        self.addCleanup(second.close)
        reopened = second.status(job)
        self.assertEqual(reopened["status"], "done")
        self.assertEqual(reopened["metrics"]["backend"], block)
        self.assertEqual(reopened["config"], status["config"])
        info = second.post("/api/plot", {"id": job, "component": "Hz", "quantity": "phase", "source": "total", "palette": "turbo"})
        self.assertEqual(info["phase_backend"], "cpp")
        self.assertEqual(second.get(f"/api/jobs/{job}/files/{info['filename']}")[:4], b"\x89PNG")

    def test_unsupported_and_failing_programs_through_the_server(self):
        """A program that refuses: the server still produces a complete Python result and says why."""
        fake = make_fake(Path(tempfile.mkdtemp(prefix="rcwa 中文 fake ")))
        self.addCleanup(shutil.rmtree, fake.parent, True)
        server = Server(RCWA_FIELDS_EXE=fake, RCWA_FAKE_MODE="refuse", RCWA_BACKEND="", RCWA_JSON_EXE="", RCWA_2D_BACKEND="cpp")
        self.addCleanup(server.close)
        job, status = server.run(model_pillar("xz", "s"))
        self.assertEqual(status["status"], "done", status.get("error"))
        block = status["metrics"]["backend"]
        self.assertEqual((block["efficiency"], block["fields"], block["phase"]), ("python", "python", "python"))
        self.assertIn("不接受", block["note"])
        self.assertGreater(block["python_solve_seconds"], 0.)
        info = server.post("/api/plot", {"id": job, "component": "Hz", "quantity": "phase_relative", "source": "total"})
        self.assertEqual(info["phase_backend"], "python")

    def test_rayleigh_error_text_is_the_python_one(self):
        cfg = normalize_2d({
            "schema_version": 1, "name": "Rayleigh", "mode": "periodic2d",
            "incident": {"wavelength": 1., "angle_deg": 0., "phi_deg": 0., "polarization": "s", "M": 1, "My": 1},
            "structure": {"period": 1., "period_y": 1., "n_incident": 1., "n_substrate": 1.5,
                          "layers": [{"name": "膜", "kind": "uniform", "thickness": .3, "n": 1.8}]},
            "sampling": {"nx": 40, "nz": 20, "z_min": -.1, "z_max": .5, "slice_plane": "xz", "slice_at": .3}})
        job, status = self.server.run(cfg)
        self.assertEqual(status["status"], "error")
        self.assertIn("Rayleigh", status["error"]["message"])
        self.assertIn("入射区", status["error"]["message"])
        self.assertNotIn("rayleigh_anomaly", status["error"]["message"])        # the C++ wording does not leak into the user's message


class GuiServerStandIn(GuiServer, unittest.TestCase):
    PROGRAM = "standin"


class GuiServerReal(GuiServer, unittest.TestCase):
    PROGRAM = "real"


class Cancellation2D(Case):
    def test_cancel_kills_the_child_of_a_2d_job(self):
        if os.name == "nt":
            self.skipTest("Windows job lifetime needs a native executable")
        fake = make_fake(self.tmp)
        pidfile = self.tmp / "fake.pid"
        server = Server(RCWA_FIELDS_EXE=fake, RCWA_FAKE_MODE="sleep", RCWA_FAKE_PIDFILE=pidfile, RCWA_BACKEND="", RCWA_JSON_EXE="", RCWA_2D_BACKEND="cpp")
        self.addCleanup(server.close)
        job = server.start(model_pillar("xz", "s"))
        deadline = time.time() + 60
        while not pidfile.exists() or not pidfile.read_text().strip():
            self.assertLess(time.time(), deadline, "the fake C++ program did not start")
            time.sleep(.05)
        child = int(pidfile.read_text())
        self.assertTrue(process_alive(child))
        self.assertEqual(server.status(job)["stage"], "正在用 C++ 计算衍射效率、六分量场与相位")
        self.assertEqual(server.post("/api/cancel", {"id": job})["status"], "cancelled")
        deadline = time.time() + 10
        while process_alive(child) and time.time() < deadline:
            time.sleep(.05)
        self.assertFalse(process_alive(child))


# ===================================================================================================== 8. edge-list tolerance
class EdgeTolerance:
    """The GUI accepts edge lists whose span differs from the period by up to 1e-8·max(period, 1) um (gui_models_2d._edges);
    the C++ protocol and solver only 1e-9·period, kept strict on purpose. The request builder snaps the last edge onto
    first edge + period, so such a model is no longer refused wholesale (it used to fall back to Python with a note)."""
    PROGRAM = "standin"
    EXACT = EXACT_BITS

    @classmethod
    def setUpClass(cls):
        cls.exe = Case.program(cls.PROGRAM)
        cls.tmp = Path(tempfile.mkdtemp(prefix="rcwa 中文 edge "))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_the_gui_limit_is_unchanged_and_the_request_carries_the_snapped_edges(self):
        for error in (4.9e-9, -4.9e-9, 1e-9, 9.9e-9):
            with self.subTest(span_error=error):
                cfg = model_edge(error)
                layer = common.build_request(cfg)["stack"]["layers"][0]
                self.assertEqual(layer["y_edges"], [0., .2, .6, .8])                  # last edge = first + period, exactly
                self.assertEqual(layer["x_profiles"][1]["edges"], [0., .3 * .8, .7 * .8, .8])
                self.assertEqual(layer["x_profiles"][0]["edges"], [0., .8])         # lists that were right are untouched
                self.assertEqual(layer["x_profiles"][1]["n_values"], [1., 1.7, 1.])
        small = model_edge(3e-10)                                                       # within half of the C++ limit: not touched
        layer = common.build_request(small)["stack"]["layers"][0]
        self.assertEqual(layer["x_profiles"][1]["edges"][-1], .8 + 3e-10)
        self.assertEqual(model_edge(0.)["structure"]["layers"][0]["y_edges"], [0., .2, .6, .8])
        self.assertEqual(common.build_request(model_edge(0.))["stack"]["layers"][0]["y_edges"], [0., .2, .6, .8])
        for period, error in ((5., 2e-8), (.8, 1e-8 * .9)):                          # the limit grows with the period above 1 um
            cfg = model_edge(0., period=period)
            cfg["structure"]["layers"][0]["x_profiles"][1]["edges"][-1] += error
            cfg = normalize_2d(cfg)
            self.assertEqual(common.build_request(cfg)["stack"]["layers"][0]["x_profiles"][1]["edges"][-1], period)
        with self.assertRaisesRegex(ValueError, "首尾边界之差必须等于周期"):          # above the GUI's own limit: refused as before
            model_edge(1.5e-8)
        with self.assertRaisesRegex(ValueError, "首尾边界之差必须等于周期"):
            model_edge(-1.5e-8)

    def test_snap_edges_directly(self):
        snap = common._snap_edges
        self.assertEqual(snap([0., .5, .8 + 4.9e-9], .8), [0., .5, .8])
        self.assertEqual(snap([0., .8 + 4.9e-9], .8), [0., .8])
        self.assertEqual(snap([-.1, .3, .7 + 4.9e-9], .8), [-.1, .3, -.1 + .8])        # a list that does not start at 0
        self.assertEqual(snap([0., .5, .8 - 4.9e-9], .8), [0., .5, .8])
        self.assertEqual(snap([0., .5, .8], .8), [0., .5, .8])
        self.assertEqual(snap([0., .5, .8 + 3e-10], .8), [0., .5, .8 + 3e-10])
        # snapping must not break the strict increase: the C++ program then refuses it, as before
        self.assertEqual(snap([0., .8000000001, .8 + 4.9e-9], .8), [0., .8000000001, .8 + 4.9e-9])
        self.assertEqual(snap([0, 1], 1.), [0., 1.])                                        # ints in, floats out
        # what "not changed" means: deviations up to 5e-10 x period leave every edge byte-identical ...
        for edges, period in (([0., .3, .8], .8), ([0., .3, .8 * (1 + 4e-10)], .8), ([0., .3, .8 * (1 - 4e-10)], .8),
                              ([.1, .3, .9], .8), ([-.3, .1, .5], .8)):
            self.assertEqual(snap(edges, period), edges)
        # ... but a deviation between 5e-10 and 1e-9 x period (the C++ program would still accept it) is snapped as well, by at most 1e-9 x period
        for relative in (6e-10, 9e-10):
            edges = [0., .3, .8 * (1 + relative)]
            snapped = snap(edges, .8)
            self.assertNotEqual(snapped, edges)
            self.assertEqual(snapped, [0., .3, .8])
            self.assertLessEqual(abs(snapped[-1] - edges[-1]), 1e-9 * .8)
        # far from the origin with a tiny period the sum edge0 + period is itself rounded beyond the C++ limit: still beyond it after snapping
        edges = [1e5, 1e5 + .4e-6, 1e5 + 1e-6 + 4.9e-9]
        snapped = snap(edges, 1e-6)
        self.assertGreater(abs((snapped[-1] - snapped[0]) - 1e-6), 1e-9 * 1e-6)

    def test_a_model_the_gui_accepted_goes_to_cpp_and_agrees_with_python(self):
        cfg = model_edge(4.9e-9)
        snapped = copy.deepcopy(cfg)
        layer = snapped["structure"]["layers"][0]
        layer["y_edges"][-1] = .8
        layer["x_profiles"][1]["edges"][-1] = .8
        python_as_entered = run_compute(cfg, self.tmp / "py as entered", RCWA_BACKEND="python")
        python_snapped = run_compute(snapped, self.tmp / "py snapped", RCWA_BACKEND="python")
        with cpp_env(self.exe):
            cpp = run_compute(cfg, self.tmp / "cpp")
        block = cpp[0]["backend"]
        self.assertEqual((block["efficiency"], block["fields"], block["phase"], block["note"]), ("cpp", "cpp", "cpp", ""))
        # (1) the C++ result is the correct solution of the request it was sent (the snapped geometry): the usual tolerance
        assert_isomorphic(self, python_snapped, cpp, self.EXACT, "snapped")
        # (2) it differs from the as-entered Python result by the snap only: ~ snap / period = 6e-9 relative, far below
        #     anything physical (the edge moved 4.9e-9 um = 0.005 pm); the bound here is 5e-8, a tenth of the GUI's own tolerance 1e-8 um on 1 um
        scale = max(float(np.max(abs(python_as_entered[1][key]))) for key in COMPONENTS)
        worst = max(float(np.max(abs(cpp[1][key] - python_as_entered[1][key]))) for key in COMPONENTS + ("E2",)) / scale
        self.assertLess(worst, 5e-8)
        self.assertGreater(worst, 0.)                      # the snapped geometry is not bit-identical to the entered one: say so
        for row, reference in zip(cpp[0]["orders"], python_as_entered[0]["orders"]):
            for name in ("R", "T", "R_s", "R_p", "T_s", "T_p"):
                self.assertAlmostEqual(row[name], reference[name], delta=5e-8)
        say(f"\n    [{self.PROGRAM}] span error 4.9e-9 um: C++ (snapped) vs Python (as entered), largest field difference {worst:.1e} of the peak")


class EdgeToleranceStandIn(EdgeTolerance, unittest.TestCase):
    PROGRAM = "standin"


class EdgeToleranceReal(EdgeTolerance, unittest.TestCase):
    PROGRAM, EXACT = "real", False


# ===================================================================================================== 9. leftover directories
class TempDirectories(Case):
    """A call that does not end in an accepted C++ result removes everything it created in the result folder
    (cpp_fields-<token> and cpp_fields-<token>.tmp-*, a few hundred MB at nx = 1600, nz = 1200) and nothing else;
    a successful call leaves the result as it always did. The stand-in with RCWA_STANDIN_HANG writes a half-finished
    temporary directory like the real program and then never answers."""

    def hang_env(self, **extra):
        marker = self.tmp / "hang.marker"
        pidfile = self.tmp / "hang.pid"
        return marker, pidfile, dict(RCWA_STANDIN_HANG=marker, RCWA_STANDIN_PIDFILE=pidfile, **extra)

    BYSTANDER = "cpp_fields-" + "0" * 32

    def bystanders(self, folder):
        """Entries that look similar but belong to nobody this call knows: another token, a plain file, a look-alike name."""
        folder.mkdir(parents=True, exist_ok=True)
        for name in (self.BYSTANDER, self.BYSTANDER + ".tmp-1-2", "cpp_fields.tmp-3"):
            (folder / name).mkdir()
        (folder / self.BYSTANDER / "fields.npy").write_bytes(b"kept")
        (folder / "cpp_fields-note.txt").write_text("kept")
        (folder / "metrics.json").write_text("{}")
        return {self.BYSTANDER, self.BYSTANDER + ".tmp-1-2", "cpp_fields.tmp-3", "cpp_fields-note.txt", "metrics.json"}

    def assert_only(self, folder, names):
        """Nothing is left in `folder` but the bystanders (and what the worker itself writes: request/reply copies, model, progress)."""
        own = {"cpp_request.json", "cpp_response.json", "model.json", "progress.json"}
        self.assertEqual({entry.name for entry in folder.iterdir()} - own, names)
        self.assertEqual((folder / self.BYSTANDER / "fields.npy").read_bytes(), b"kept")

    def test_timeout_leaves_nothing_behind(self):
        marker, pidfile, env = self.hang_env(RCWA_CPP_TIMEOUT="8")
        folder = self.tmp / "folder"
        names = self.bystanders(folder)
        with cpp_env(self.program("standin"), **env):
            attempt = backend.attempt(model_pillar(), folder)
        self.assertFalse(attempt.used)
        self.assertIn("没有返回", attempt.note)
        self.assertTrue(marker.exists(), "the stand-in did not reach its hang point before the timeout (very slow machine?)")
        partial = Path(marker.read_text(encoding="utf-8"))
        self.assertEqual(partial.parent.resolve(), folder.resolve())       # the program did write its half-finished directory here
        self.assertFalse(partial.exists())
        self.assertFalse(process_alive(int(pidfile.read_text())))
        self.assert_only(folder, names)

    def test_a_killed_program_leaves_nothing_behind(self):
        import signal
        import threading
        if os.name == "nt":
            self.skipTest("POSIX signals")
        marker, pidfile, env = self.hang_env()
        folder = self.tmp / "folder"
        names = self.bystanders(folder)
        outcome = {}
        with cpp_env(self.program("standin"), **env):
            worker = threading.Thread(target=lambda: outcome.update(attempt=backend.attempt(model_pillar(), folder)))
            worker.start()
            deadline = time.time() + 120
            while not marker.exists() or not marker.read_text(encoding="utf-8"):
                self.assertLess(time.time(), deadline, "the stand-in did not start")
                time.sleep(.05)
            partial = Path(marker.read_text(encoding="utf-8"))
            self.assertTrue(partial.is_dir())                                      # half-finished, as a real crash leaves it
            os.kill(int(pidfile.read_text()), signal.SIGKILL)
            worker.join(60)
        attempt = outcome["attempt"]
        self.assertFalse(attempt.used)
        self.assertIn("返回码", attempt.note)
        self.assertFalse(partial.exists())
        self.assert_only(folder, names)

    def test_cancelling_the_worker_leaves_nothing_behind(self):
        import signal
        if os.name == "nt":
            self.skipTest("POSIX signals; Windows terminates the worker without letting it clean up")
        marker, pidfile, env = self.hang_env()
        folder = self.tmp / "result"
        names = self.bystanders(folder)
        config = self.tmp / "model.json"
        config.write_text(json.dumps(model_pillar()), encoding="utf-8")
        environment = os.environ.copy()
        environment.update({key: str(value) for key, value in env.items()})
        environment.update(RCWA_FIELDS_EXE=str(self.program("standin")), RCWA_BACKEND="", RCWA_JSON_EXE="", RCWA_VERIFY_PYTHON="", RCWA_2D_BACKEND="cpp")
        worker = subprocess.Popen([sys.executable, str(GUI / "gui_worker.py"), str(config), str(folder)], cwd=str(GUI), env=environment,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.time() + 120
            while not marker.exists() or not marker.read_text(encoding="utf-8"):
                self.assertLess(time.time(), deadline, "the stand-in did not start")
                self.assertIsNone(worker.poll())
                time.sleep(.05)
            partial = Path(marker.read_text(encoding="utf-8"))
            self.assertTrue(partial.is_dir())
            worker.send_signal(signal.SIGTERM)                                       # what the GUI server does on "cancel"
            worker.wait(60)
        finally:
            if worker.poll() is None:
                worker.kill()
        self.assertEqual(worker.returncode, 128 + signal.SIGTERM)
        self.assertFalse(partial.exists())
        self.assertFalse(process_alive(int(pidfile.read_text())))
        self.assert_only(folder, names)

    def test_cancelling_while_the_reply_is_validated_leaves_nothing_behind(self):
        """The program has returned and published cpp_fields-<token>; the validation (seconds, for a big reply) is under way; the GUI cancels
        (SIGTERM). The handler used to be installed only around the subprocess call, so the worker died on the spot with the directory in place."""
        import signal
        if os.name == "nt":
            self.skipTest("POSIX signals; Windows terminates the worker without letting it clean up")
        folder = self.tmp / "result"
        names = self.bystanders(folder)
        marker, config, script = self.tmp / "validating.marker", self.tmp / "model.json", self.tmp / "child.py"
        config.write_text(json.dumps(model_pillar()), encoding="utf-8")
        script.write_text(
            "import json, sys, time\n"
            f"sys.path.insert(0, {str(GUI)!r})\n"
            "from pathlib import Path\n"
            "import cpp_fields_backend as backend\n"
            "original = backend.extract_fields\n"
            "def slow(*args, **kwargs):\n"
            f"    Path({str(marker)!r}).write_text('x')\n"
            "    time.sleep(300)\n"
            "    return original(*args, **kwargs)\n"
            "backend.extract_fields = slow\n"
            f"cfg = json.loads(Path({str(config)!r}).read_text(encoding='utf-8'))\n"
            f"backend.attempt(cfg, Path({str(folder)!r}))\n", encoding="utf-8")
        environment = os.environ.copy()
        environment.update(RCWA_FIELDS_EXE=str(self.program("standin")), RCWA_BACKEND="", RCWA_JSON_EXE="", RCWA_VERIFY_PYTHON="", RCWA_2D_BACKEND="cpp")
        child = subprocess.Popen([sys.executable, str(script)], cwd=str(GUI), env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.time() + 180
            while not marker.exists():
                self.assertLess(time.time(), deadline, "the child did not reach the validation")
                self.assertIsNone(child.poll())
                time.sleep(.05)
            published = [entry for entry in folder.iterdir() if entry.name.startswith("cpp_fields-") and entry.name not in names]
            self.assertEqual(len(published), 1)                                       # the program's result is in place, validation not finished
            self.assertTrue((published[0] / "fields.npy").is_file())
            child.send_signal(signal.SIGTERM)
            child.wait(60)
        finally:
            if child.poll() is None:
                child.kill()
        self.assertEqual(child.returncode, 128 + signal.SIGTERM)
        self.assert_only(folder, names)

    def test_a_rejected_reply_removes_the_published_directory(self):
        """The program finished and published its directory; the reply or a file is then wrong: the whole directory goes."""
        exe, real_call = self.program("standin"), common.call
        for label, change in (("ok is 1", lambda r: r.update(ok=1)),
                              ("file is junk", lambda r: (Path(r["field_data"]["directory"]) / "fields.npy").write_bytes(b"junk")),
                              ("far field arrays wrong", lambda r: r["results"][0].update(Rs=r["results"][0]["Rp"]))):
            with self.subTest(case=label):
                def call(exe, request, folder, timeout=None, change=change):
                    response = real_call(exe, request, folder, timeout)
                    change(response)
                    return response
                folder = self.tmp / label.replace(" ", "_")
                names = self.bystanders(folder)
                with cpp_env(exe), patch.object(common, "call", side_effect=call):
                    attempt = backend.attempt(model_pillar(), folder)
                self.assertFalse(attempt.used)
                self.assertTrue(attempt.note)
                self.assert_only(folder, names)

    def test_one_dimensional_calls_clean_up_the_same_way(self):
        """The 1-D path shares attempt(): a request that never produced a valid reply leaves nothing either."""
        cfg = gui_models.preset("trapezoid_analytic")
        folder = self.tmp / "one"
        names = self.bystanders(folder)

        def call(exe, request, workdir, timeout=None):
            final = Path(request["output_dir"])
            (final.parent / (final.name + ".tmp-5-6")).mkdir()
            final.mkdir()
            (final / "fields.npy").write_bytes(b"x" * 1000)
            raise BackendError("C++ 程序超过 1 s 没有返回，已终止")

        with cpp_env(self.program("standin")), patch.object(common, "call", side_effect=call):
            attempt = backend.attempt(cfg, folder)
        self.assertFalse(attempt.used)
        self.assertIn("没有返回", attempt.note)
        self.assert_only(folder, names)

    def test_a_successful_call_keeps_its_directory_and_only_that(self):
        for name in ("standin", "real"):
            if STATE[name] is None:
                continue
            with self.subTest(program=name):
                folder = self.tmp / ("ok " + name)
                names = self.bystanders(folder)
                with cpp_env(STATE[name]):
                    attempt = backend.attempt(model_pillar(), folder)
                self.assertTrue(attempt.used, attempt.note)
                kept = [entry for entry in folder.iterdir() if entry.name.startswith("cpp_fields-") and entry.name not in names]
                self.assertEqual(len(kept), 1)
                self.assertEqual(sorted(f.name for f in kept[0].iterdir()),
                                 ["fields.npy", "phase.npy", "phase_mask.npy", "t.npy", "x.npy", "y.npy", "z.npy"])
                self.assertEqual(sorted(entry.name for entry in folder.iterdir() if ".tmp-" in entry.name and entry.name not in names), [])
                self.assertEqual(kept[0].name, "cpp_fields-" + attempt.manifest["request_id"])
                self.assertEqual(Path(attempt.manifest["directory"]).resolve(), kept[0].resolve())

    def test_discard_only_touches_this_calls_names(self):
        folder = self.tmp / "direct"
        folder.mkdir()
        token = "a" * 32
        for name in (f"cpp_fields-{token}", f"cpp_fields-{token}.tmp-1-2", f"cpp_fields-{token}.tmp-3-4",
                     f"cpp_fields-{token}x", f"cpp_fields-{'b' * 32}", f"cpp_fields-{'b' * 32}.tmp-1-2", f"other-{token}"):
            (folder / name).mkdir()
            (folder / name / "f").write_bytes(b"1")
        (folder / f"cpp_fields-{token}.tmp-file").write_bytes(b"1")                # a plain file with a matching name goes as well
        outside = self.tmp / "outside"
        outside.mkdir()
        (outside / "precious").write_bytes(b"1")
        if os.name != "nt":
            (folder / f"cpp_fields-{token}.tmp-link").symlink_to(outside, target_is_directory=True)     # the link goes, its target stays
        backend.discard_call_outputs(folder, token)
        self.assertEqual(sorted(entry.name for entry in folder.iterdir()),
                         sorted([f"cpp_fields-{token}x", f"cpp_fields-{'b' * 32}", f"cpp_fields-{'b' * 32}.tmp-1-2", f"other-{token}"]))
        self.assertTrue((outside / "precious").is_file())
        backend.discard_call_outputs(self.tmp / "does not exist", token)                  # no folder: nothing to do, no error


@unittest.skipUnless(os.name == "nt", "taskkill is a Windows tool")
class KillTreeWindows(Case):
    """cpp_backend._kill_tree ends a Windows process tree with taskkill /T. The tool is addressed by its full path under %SystemRoot%\\System32
    (a program called taskkill earlier in PATH must not be picked up); a missing file is not an error: child.kill() still runs."""

    class Child:
        pid = 4242
        killed = False

        def poll(self):
            return None

        def kill(self):
            self.killed = True

    def test_taskkill_is_called_by_its_system32_path(self):
        calls = []
        child = self.Child()
        with patch.object(common.subprocess, "run", side_effect=lambda command, **kwargs: calls.append(command)):
            common._kill_tree(child)
        self.assertEqual(len(calls), 1)
        command = calls[0]
        self.assertTrue(os.path.isabs(command[0]))
        self.assertEqual(os.path.normcase(command[0]), os.path.normcase(os.path.join(os.environ["SystemRoot"], "System32", "taskkill.exe")))
        self.assertEqual(command[1:], ["/F", "/T", "/PID", "4242"])
        self.assertTrue(child.killed)

    def test_a_missing_taskkill_is_not_an_error(self):
        calls = []
        child = self.Child()
        with EnvPatch(SystemRoot=self.tmp / "no such windows"), patch.object(common.subprocess, "run", side_effect=lambda command, **kwargs: calls.append(command)):
            common._kill_tree(child)
        self.assertEqual(calls, [])
        self.assertTrue(child.killed)


# ===================================================================================================== 9b. cancelling a job (round 3, M1)
class CancelCleanup(Case):
    """gui.AppState.cancel: the worker is terminated (SIGTERM; TerminateProcess on Windows, where its own cleanup never runs), and the C++ result
    folders of THAT job -- cpp_fields-<token>, cpp_fields-<token>.tmp-* in the job's own folder -- are removed afterwards (cpp_fields_backend.discard_job_outputs)."""

    A, B = "a" * 32, "b" * 32

    def populate(self, folder):
        """What a worker killed in the middle of a C++ call leaves, plus things that must survive."""
        folder.mkdir(parents=True)
        for name in (f"cpp_fields-{self.A}", f"cpp_fields-{self.A}.tmp-12-34", f"cpp_fields-{self.B}.tmp-5-6"):
            (folder / name / "deeper").mkdir(parents=True)
            (folder / name / "fields.npy").write_bytes(b"x" * 1000)
            (folder / name / "deeper" / "t.npy").write_bytes(b"y")
        (folder / f"cpp_fields-{self.A}.tmp-file").write_bytes(b"1")              # a plain file with such a name goes as well
        survivors = {"model.json", "worker.log", "progress.json", "cpp_request.json", "cpp_fields-note.txt", f"cpp_fields-{self.A}x",
                     f"cpp_fields-{self.A[:-1]}", f"xcpp_fields-{self.A}", "cpp_fields"}
        for name in survivors:
            (folder / name).write_text("kept")
        (folder / "cpp_fields-lookalike-dir").mkdir()
        (folder / "cpp_fields-lookalike-dir" / "f").write_text("kept")
        return survivors | {"cpp_fields-lookalike-dir"}

    def link(self, folder, outside):
        """A symbolic link (POSIX) or a directory junction (Windows) named like a result folder, pointing at `outside`; None if it cannot be made."""
        path = folder / f"cpp_fields-{'c' * 32}"
        if os.name != "nt":
            path.symlink_to(outside, target_is_directory=True)
            return path
        made = subprocess.run(["cmd", "/c", "mklink", "/J", str(path), str(outside)], capture_output=True, creationflags=0x08000000)
        return path if made.returncode == 0 and path.exists() else None

    def test_discard_job_outputs_removes_the_jobs_cpp_results_and_nothing_else(self):
        folder = self.tmp / "job"
        survivors = self.populate(folder)
        outside = self.tmp / "outside"
        outside.mkdir()
        (outside / "precious").write_bytes(b"1")
        link = self.link(folder, outside)                  # the link is removed, its target is not entered
        self.assertEqual(backend.discard_job_outputs(folder), 0)
        self.assertEqual({entry.name for entry in folder.iterdir()}, survivors)
        self.assertTrue((outside / "precious").is_file())
        if link is not None:
            self.assertFalse(os.path.lexists(link))
        self.assertEqual((folder / "cpp_fields-lookalike-dir" / "f").read_text(), "kept")
        self.assertEqual(backend.discard_job_outputs(folder), 0)                 # nothing left: nothing to do
        self.assertEqual(backend.discard_job_outputs(self.tmp / "does not exist"), 0)

    def test_discard_job_outputs_waits_for_busy_files_and_looks_again(self):
        """On Windows the C++ child dies asynchronously after the worker was terminated: it may still hold a file in cpp_fields-<token>.tmp-* (the removal
        is refused for a moment) or publish its directory just after the first look. Busy entries are retried until the deadline, and after anything was
        removed the folder is scanned once more; a folder that cannot be cleaned is given up on after `seconds`, not waited for forever."""
        folder = self.tmp / "busy"
        survivors = self.populate(folder)
        real_remove, attempts = backend._remove_entry, {}

        def busy_at_first(entry):
            attempts[entry.name] = attempts.get(entry.name, 0) + 1
            if attempts[entry.name] <= 2:
                raise PermissionError("in use")
            return real_remove(entry)
        with patch.object(backend, "_remove_entry", busy_at_first):
            self.assertEqual(backend.discard_job_outputs(folder), 0)
        self.assertEqual({entry.name for entry in folder.iterdir()}, survivors)
        self.assertEqual(len(attempts), 4)                              # the four entries that are the job's own
        self.assertTrue(all(count >= 3 for count in attempts.values()), attempts)
        # a directory published just after the first pass is found by the second look
        folder = self.tmp / "late"
        self.populate(folder)
        real_pass, calls, late = backend._discard_matching, [0], f"cpp_fields-{'d' * 32}"

        def late_arrival(outdir, matches):
            result = real_pass(outdir, matches)
            if not calls[0]:
                (Path(outdir) / late).mkdir()
            calls[0] += 1
            return result
        with patch.object(backend, "_discard_matching", late_arrival):
            self.assertEqual(backend.discard_job_outputs(folder), 0)
        self.assertFalse((folder / late).exists())
        self.assertGreaterEqual(calls[0], 3)
        # something that can never be removed: gives up after `seconds` and reports it
        folder = self.tmp / "stuck"
        self.populate(folder)
        started = time.time()
        with patch.object(backend, "_remove_entry", side_effect=PermissionError("in use")):
            left = backend.discard_job_outputs(folder, seconds=.6)
        self.assertGreater(left, 0)
        self.assertLess(time.time() - started, 30)

    def test_cancel_removes_what_the_terminated_worker_could_not_clean_up(self):
        """Both platforms: a stand-in worker that is simply terminated (no cleanup runs in it, as on Windows) with C++ result folders in its job folder."""
        state = gui.AppState(model_pillar(), result_root=self.tmp / "runs")
        cfg = model_pillar()
        survivors = self.populate(state.result_root / "20261011-120000-aaaaaa")
        finished = state.result_root / "20261011-120001-bbbbbb"                   # another job, already finished: its result is not touched
        self.populate(finished)
        running = state.result_root / "20261011-120002-cccccc"
        self.populate(running)
        sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
        done = subprocess.Popen([sys.executable, "-c", "pass"])
        done.wait()
        try:
            state.jobs["20261011-120000-aaaaaa"] = {"process": sleeper, "folder": state.result_root / "20261011-120000-aaaaaa", "config": cfg, "cancelled": False}
            state.jobs["20261011-120001-bbbbbb"] = {"process": done, "folder": finished, "config": cfg, "cancelled": False}
            state.jobs["20261011-120002-cccccc"] = {"process": None, "folder": running, "config": cfg, "cancelled": False, "saved": True}
            status = state.cancel("20261011-120000-aaaaaa")
            self.assertEqual(status["status"], "cancelled")
            self.assertIsNotNone(sleeper.poll())
            self.assertEqual({entry.name for entry in (state.result_root / "20261011-120000-aaaaaa").iterdir()}, survivors)
            for other in (finished, running):                                    # a finished or reopened job is not a cancel target: nothing removed
                self.assertEqual(len([entry for entry in other.iterdir() if entry.name.startswith("cpp_fields-" + self.A)]), 4)
                self.assertTrue((other / f"cpp_fields-{self.A}" / "fields.npy").is_file())
            state.cancel("20261011-120001-bbbbbb")
            state.cancel("20261011-120002-cccccc")
            self.assertTrue((finished / f"cpp_fields-{self.A}" / "fields.npy").is_file())
            self.assertTrue((running / f"cpp_fields-{self.A}" / "fields.npy").is_file())
        finally:
            if sleeper.poll() is None:
                sleeper.kill()

    def test_cancel_keeps_the_raw_folder_of_a_result_that_was_already_complete(self):
        """The worker has written fields.npz and metrics.json and is only exiting (a window of some tens of ms): that job shows in the history after a
        restart, so its accepted cpp_fields-<token>/ is kept. With only one of the two files the result is not complete and the folder is cleaned."""
        state = gui.AppState(model_pillar(), result_root=self.tmp / "runs")
        cfg = model_pillar()
        cases = (("20261011-120003-dddddd", ("metrics.json", "fields.npz"), True), ("20261011-120004-eeeeee", ("metrics.json",), False),
                 ("20261011-120005-ffffff", ("fields.npz",), False))
        sleepers = []
        try:
            for jid, written, kept in cases:
                folder = state.result_root / jid
                survivors = self.populate(folder)
                for name in written:
                    (folder / name).write_text("{}")
                before = {entry.name for entry in folder.iterdir()}
                sleepers.append(subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"]))
                state.jobs[jid] = {"process": sleepers[-1], "folder": folder, "config": cfg, "cancelled": False}
                state.cancel(jid)
                self.assertIsNotNone(sleepers[-1].poll())
                self.assertEqual({entry.name for entry in folder.iterdir()}, before if kept else survivors | set(written), jid)
        finally:
            for sleeper in sleepers:
                if sleeper.poll() is None:
                    sleeper.kill()

    def test_discard_job_outputs_does_nothing_for_an_argument_that_is_not_a_path(self):
        """os.scandir(None) lists the current directory: a None folder must not clean the working directory, and nothing else may raise."""
        here = self.tmp / "cwd"
        self.populate(here)
        before = {entry.name for entry in here.iterdir()}
        old = os.getcwd()
        os.chdir(here)
        try:
            for bad in (None, "", 5, 2.5, b"job", [], {}, object()):
                self.assertEqual(backend.discard_job_outputs(bad), 0, repr(bad))
            self.assertEqual({entry.name for entry in here.iterdir()}, before)
        finally:
            os.chdir(old)

    def test_only_symlinks_and_junctions_count_as_links(self):
        """A reparse point that is neither a symbolic link nor a mount point (an OneDrive on-demand directory carries one) is an ordinary directory:
        _is_link must say False so that it is removed as a tree; an unknown tag (old Python) stays on the safe side."""
        class Info:
            st_mode = 0o040755
            st_file_attributes = backend.stat.FILE_ATTRIBUTE_REPARSE_POINT | 0x10

            def __init__(self, tag):
                if tag is not None:
                    self.st_reparse_tag = tag
        cloud, symlink, junction = 0x9000001A, 0xA000000C, 0xA0000003
        for tag, expected in ((cloud, False), (symlink, True), (junction, True), (None, True)):
            with patch.object(backend.os, "lstat", return_value=Info(tag)):
                self.assertIs(backend._is_link("anything"), expected, tag)

    def test_cancelling_a_real_job_through_the_gui_leaves_no_cpp_output(self):
        """Both platforms, the real worker with the stand-in program that hangs after writing a half-finished .tmp- directory: the GUI's own cancel
        (AppState.launch, then AppState.cancel) leaves no cpp_fields-* in the job's folder, and the stand-in process is gone."""
        marker, pidfile = self.tmp / "hang.marker", self.tmp / "hang.pid"
        env = dict(RCWA_STANDIN_HANG=marker, RCWA_STANDIN_PIDFILE=pidfile, RCWA_FIELDS_EXE=self.program("standin"), RCWA_BACKEND=None,
                   RCWA_JSON_EXE=None, RCWA_VERIFY_PYTHON=None, RCWA_CPP_TIMEOUT=None, RCWA_2D_BACKEND="cpp")
        state = gui.AppState(model_pillar(), result_root=self.tmp / "runs")
        with EnvPatch(**env):
            info = state.launch(model_pillar())
        jid = info["id"]
        try:
            folder = state.jobs[jid]["folder"]
            deadline = time.time() + 180
            while not marker.exists() or not marker.read_text(encoding="utf-8"):
                self.assertLess(time.time(), deadline, "the stand-in did not start")
                self.assertIsNone(state.jobs[jid]["process"].poll())
                time.sleep(.05)
            partial = Path(marker.read_text(encoding="utf-8"))
            self.assertTrue(partial.is_dir())
            self.assertEqual(partial.parent.resolve(), folder.resolve())
            self.assertEqual(state.cancel(jid)["status"], "cancelled")
            self.assertEqual([entry.name for entry in folder.iterdir() if entry.name.startswith("cpp_fields-")], [])
            self.assertFalse(partial.exists())
            self.assertTrue((folder / "model.json").is_file())                    # the rest of the job's folder is the GUI's to keep
            pid = int(pidfile.read_text())
            deadline = time.time() + 20
            while process_alive(pid) and time.time() < deadline:
                time.sleep(.1)
            self.assertFalse(process_alive(pid))
        finally:
            process = state.jobs[jid]["process"]
            if process.poll() is None:
                process.kill()


if __name__ == "__main__":
    unittest.main()
