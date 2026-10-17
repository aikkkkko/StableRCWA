"""C++ 后端的验证：与 Python 求解器逐级对拍、路由与退回、圆锥入射的奇点提示、端到端、结果页截图。

在 gui/ 目录里运行（只用标准库的 unittest，不依赖 pytest）：

    python -m unittest validation.test_cpp_backend -v
    python validation/test_cpp_backend.py -v

找不到命令行程序（没有编译器、也没有编好的 gui/bin/rcwa_json）时，需要 C++ 的测试标为跳过，
退回 Python 的各项测试照常进行。无头 Chrome 不存在时截图一项跳过。

环境变量：
    RCWA_JSON_EXE     用指定的可执行文件代替 gui/bin/rcwa_json（见 cpp_backend.py）
    RCWA_TEST_KEEP    一个目录：把结果页截图、导出的 DOM 留在那里；不设时用临时目录，测试完删除
    CHROME            无头 Chrome 的路径（默认找 macOS 的 Google Chrome，再找 PATH 里的 chrome / chromium）
"""
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

GUI = Path(__file__).resolve().parents[1]
if str(GUI) not in sys.path:
    sys.path.insert(0, str(GUI))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402

import cpp_backend  # noqa: E402
import gui_models  # noqa: E402
import gui_worker  # noqa: E402
import rcwa1d as rc  # noqa: E402
from conical.rcwa_conical import solve_conical  # noqa: E402

EFFICIENCY_TOL = 1e-9          # C++ 与 Python 各级效率之差
JONES_TOL = 1e-9               # 传播级 Jones 矩阵元：|ΔJ| ≤ 1e-9·max(1, |J|)
SMALL_M = 20                   # 对拍时把 M 降到这个数，以便快跑

# 内置预设里哪些该交给 C++（其余的走 Python）
CPP_PRESETS = {"trapezoid_analytic", "triangle_analytic", "trapezoid_fft", "triangle_fft", "meent_conical", "slab"}
# 二维预设现在也交给 C++（其效率、场与相位的对拍在 test_cpp_2d_backend.py；这里的逐级对拍只覆盖一维求解器）
CPP_PRESETS_2D = {"pillar_2d"}

_STATE = {"exe": None, "note": "", "fields_exe": None, "fields_note": ""}


def setUpModule():
    # 需要时在这里现场编译（只编一次）；编不出来时相关测试跳过
    if not cpp_backend.forced_python():
        _STATE["exe"], _STATE["note"] = cpp_backend.ensure_executable()
        _STATE["fields_exe"], _STATE["fields_note"] = gui_worker.cpp_backend.ensure_executable()


def say(text):
    """打印一段说明。控制台的编码写不出的字符（中文 Windows 的 GBK 里没有“−”）换成 ?，不让打印本身出错。"""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(text.encode(encoding, "replace").decode(encoding))


def need_cpp(test, fields=False):
    if cpp_backend.forced_python():
        test.skipTest("RCWA_BACKEND=python：不测 C++ 一路")
    if _STATE["fields_exe" if fields else "exe"] is None:
        test.skipTest("没有可用的 " + ("rcwa_fields：" + _STATE["fields_note"] if fields else "rcwa_json：" + _STATE["note"]))


def all_examples():
    """内置示例：界面里的预设，加上 examples/models/ 下的模型文件。返回 [(名字, 模型)]。"""
    items = [("preset:" + p["id"], gui_models.preset(p["id"])) for p in gui_models.PRESETS]
    for path in sorted((GUI / "examples" / "models").glob("*.json")):
        items.append(("model:" + path.name, json.loads(path.read_text(encoding="utf-8"))))
    return items


def planar_model(pol="TE", **incident):
    """一个带吸收、边界起点不在 0 的两层光栅，外加一层吸收膜。"""
    stack = rc.Stack(1.2, [rc.Layer(.25, edges=[0., .4, .9], n_values=[2.0, 1.2], name="光栅 A"),
                           rc.Layer(.12, n=1.7 + .05j, name="吸收膜"),
                           rc.Layer(.2, edges=[.1, .35, .7, 1.0], n_values=[1.5, 2.2 + .02j, 1.5], name="光栅 B")], 1.5, .9)
    options = dict(name="验证用平面模型", lam0=.83, theta_deg=27., M=12, pol=pol)
    options.update(incident)
    return gui_models.from_stack(stack, **options)


def conical_model(pol="jones", phi_deg=63., **incident):
    cfg = planar_model()
    stack = gui_models.build_stack(cfg)[1]
    options = dict(name="验证用圆锥模型", lam0=.83, theta_deg=27., M=12, pol=pol, phi_deg=phi_deg, psi_deg=35., delta_deg=70.)
    options.update(incident)
    return gui_models.from_stack(stack, **options)


# 圆锥入射特有的奇点（说明见 StableRCWA 的 rcwa1d_conical.hpp）：光栅层的层矩阵 A = Kx² − ⟦ε⟧ 有一个本征值过零的
# 波长、角度上，两个模式重合。下面这个无吸收光栅在 θ = 40°、φ = 25°、M = 8 时，奇点波长 t0 ≈ 0.98048 μm；
# C++ 在相对偏离约 4e-4 以内置 degenerate_mode_warning，并对退化层使用 Schur 传播。
DEGENERATE = {"theta": 40., "phi": 25., "M": 8, "period": .574, "edges": [0., .1722, .574], "n_values": [1.46, 1.0]}
DEGENERATE_KY15 = (np.sin(np.radians(40.)) * np.sin(np.radians(25.))) ** 1.5      # |ky|^1.5，ky = nI·sin θ·sin φ
SENTENCE = "两个模式几乎重合"          # 结果页上那句提示里的几个字


def singular_wavelength():
    """对波长二分，找到 A 的负本征值个数变化的地方（A 是厄米的，用 NumPy 直接算）。"""
    if "t0" not in DEGENERATE:
        d = DEGENERATE
        m = np.arange(-d["M"], d["M"] + 1)
        h = m[:, None] - m[None, :]
        eps = np.zeros(h.shape, complex)          # ⟦ε⟧：分段常数 ε 的 Fourier 系数排成的 Toeplitz 矩阵
        for a, b, n in zip(d["edges"][:-1], d["edges"][1:], d["n_values"]):
            phase = lambda x: np.exp(-2j * np.pi * h * x / d["period"])
            eps += n ** 2 * np.where(h == 0, (b - a) / d["period"], (phase(a) - phase(b)) / (2j * np.pi * np.where(h == 0, 1, h)))
        kx0 = np.sin(np.radians(d["theta"])) * np.cos(np.radians(d["phi"]))

        def negative(wavelength):
            kx = kx0 - m * wavelength / d["period"]
            return int(np.sum(np.linalg.eigvalsh(np.diag(kx ** 2) - eps) < 0))

        lo, hi = .95, 1.0
        at_lo = negative(lo)
        assert at_lo - negative(hi) == 1
        while True:
            mid = .5 * (lo + hi)
            if mid in (lo, hi):
                break
            lo, hi = (mid, hi) if negative(mid) == at_lo else (lo, mid)
        d["t0"] = .5 * (lo + hi)
    return DEGENERATE["t0"]


def degenerate_model(offset):
    """波长取在奇点旁边 t0·(1 + offset) 的圆锥模型：一个无吸收光栅层，下面一层膜。"""
    d = DEGENERATE
    stack = rc.Stack(1., [rc.Layer(.38, edges=d["edges"], n_values=d["n_values"], name="光栅"),
                          rc.Layer(.115, n=1.96, name="膜")], 1.46, d["period"])
    return gui_models.from_stack(stack, name="奇点附近的圆锥模型", lam0=singular_wavelength() * (1. + offset), theta_deg=d["theta"],
                                 M=d["M"], pol="jones", phi_deg=d["phi"], psi_deg=45., delta_deg=90.)


def python_solve(cfg):
    cfg, stack, pml, bg = gui_models.build_stack(cfg)
    inc = cfg["incident"]
    if cfg["mode"] == "conical":
        res = solve_conical(stack, inc["wavelength"], inc["angle_deg"], inc["phi_deg"], M=inc["M"],
                            psi_deg=inc["psi_deg"], delta_deg=inc["delta_deg"])
    else:
        res = rc.solve(stack, inc["wavelength"], inc["angle_deg"], inc["M"], inc["polarization"], need_fields=False)
    return cfg, res


def listed_orders(res):
    """界面的级次表只列效率不为零的级次（与 gui_worker.py 的筛选相同）。"""
    return [int(m) for m, r, t in zip(res.ms, res.DEr, res.DEt) if abs(r) + abs(t) > 1e-9]


class EnvPatch:
    """临时改环境变量，退出时恢复。"""

    def __init__(self, **values):
        self.values = values

    def __enter__(self):
        self.saved = {k: os.environ.get(k) for k in self.values}
        for k, v in self.values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


FAKE_SOURCE = r'''
import json, os, sys, time
request = json.loads(sys.stdin.buffer.read().decode("utf-8"))      # 请求走标准输入，应答走标准输出
request = request.get("solve", request)                         # GUI fields envelope / legacy far API
mode = os.environ.get("RCWA_FAKE_MODE", "refuse")
if os.environ.get("RCWA_FAKE_PIDFILE"):
    open(os.environ["RCWA_FAKE_PIDFILE"], "w").write(str(os.getpid()))
if mode == "sleep":
    time.sleep(float(os.environ.get("RCWA_FAKE_SLEEP", "120")))
if mode == "crash":
    sys.stderr.write("segmentation fault (simulated)\n")
    raise SystemExit(3)
M = request["M"]; N = 2 * M + 1
if mode == "null":      # 数值失败：效率写成 null，ok 仍为真
    reply = {"ok": True, "version": 1, "mode": request["mode"], "M": M, "orders": list(range(-M, M + 1)), "threads": 1,
             "results": [{"wavelength": 1.0, "theta_deg": 0.0, "R": [[None, None]] * N, "T": [[None, None]] * N,
                          "DEr": [None] * N, "DEt": [None] * N, "R_total": None, "T_total": None, "eig_info": 7,
                          "n_eig": 1, "n_dense_solve": 0, "max_eig_residual": 0.0, "max_solve_backward_error": 0.0,
                          "timing": {"total": 0.0, "eig": 0.0, "linear": 0.0, "product": 0.0}}]}
elif mode == "degenerate":      # 圆锥入射的奇点：效率是 null，min_abs_lambda 与 degenerate_mode_warning 照常给出
    reply = {"ok": True, "version": 1, "mode": "conical", "M": M, "orders": list(range(-M, M + 1)), "threads": 1,
             "results": [{"polarized": {"DEr": [None] * N}, "eig_info": 0, "min_abs_lambda": 3e-9,
                          "degenerate_mode_warning": True}]}
elif mode == "garbage":
    reply = None
else:
    reply = {"ok": False, "error": "这个程序没有编入圆锥入射", "where": "mode"}
text = "这不是 JSON" if reply is None else json.dumps(reply, ensure_ascii=False)
sys.stdout.buffer.write(text.encode("utf-8"))
raise SystemExit(0 if reply and reply.get("ok") else 1)
'''


def make_fake_executable(folder):
    """一个行为可由环境变量 RCWA_FAKE_MODE 指定的假 rcwa_json，用来演练各种失败。"""
    folder = Path(folder)
    script = folder / "fake_rcwa_json.py"
    script.write_text(FAKE_SOURCE, encoding="utf-8")
    if os.name == "nt":
        exe = folder / "fake_rcwa_json.bat"
        exe.write_text(f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
    else:
        exe = folder / "fake_rcwa_json"
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return exe


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rcwa_cpp_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def run_worker(self, cfg, name="job"):
        """在本进程里走一遍 gui_worker.compute，返回 (metrics, 结果目录)。"""
        out = self.tmp / name
        with EnvPatch(RCWA_VERIFY_PYTHON="1"):
            metrics = gui_worker.compute(cfg, out)
        self.assertEqual(metrics, json.loads((out / "metrics.json").read_text(encoding="utf-8")))
        return metrics, out

    def check_backend_block(self, block):
        for key in ("efficiency", "fields", "cpp_seconds", "python_solve_seconds", "max_efficiency_difference", "note"):
            self.assertIn(key, block)
        self.assertIn(block["efficiency"], ("cpp", "python"))
        self.assertEqual(block["fields"], "cpp" if block["efficiency"] == "cpp" else "python")
        self.assertGreaterEqual(block["python_solve_seconds"], 0.)
        self.assertIsInstance(block["note"], str)


# --------------------------------------------------------------------------- 1. 与 Python 逐级对拍
class Comparison(TempDirCase):
    def test_fft_models(self):
        need_cpp(self)
        for mode, cfg in (("TE", planar_model("TE")), ("TM", planar_model("TM")),
                          ("conical", conical_model("jones"))):
            # Mix analytic and FFT layers, absorption and a translated origin.
            for samples in (0, 1009, 8192):
                with self.subTest(mode=mode, samples=samples):
                    cfg["structure"]["layers"][2].update(fourier_method="fft", fft_samples=samples)
                    self.compare(f"{mode} FFT N={samples}", cfg)

    def compare(self, name, cfg):
        cfg, res = python_solve(cfg)
        attempt = cpp_backend.attempt(cfg, self.tmp)
        self.assertTrue(attempt.used, f"{name}：C++ 没有用上：{attempt.note}")
        eff = attempt.merge(res)
        self.assertTrue(attempt.used, f"{name}：{attempt.note}")
        worst = max(float(np.max(abs(eff[k] - res[k]))) for k in ("DEr", "DEt"))
        self.assertLessEqual(worst, EFFICIENCY_TOL, f"{name}：各级效率相差 {worst:.3e}")
        self.assertLessEqual(attempt.difference, EFFICIENCY_TOL, f"{name}：效率（含 s/p 分量）相差 {attempt.difference:.3e}")
        jones = 0.
        if cfg["mode"] == "conical":
            for key, flux in (("Jr", "flux_r"), ("Jt", "flux_t")):
                live = res[flux] > 0
                self.assertTrue(live.any() or key == "Jt")
                if live.any():
                    jones = max(jones, float(np.max(abs(eff[key][live] - res[key][live]) / np.maximum(1., abs(res[key][live])))))
                np.testing.assert_allclose(eff[flux], res[flux], rtol=0, atol=1e-12, err_msg=name)
            self.assertLessEqual(jones, JONES_TOL, f"{name}：Jones 矩阵相差 {jones:.3e}")
            for key in ("Rs", "Rp", "Ts", "Tp"):     # 给定入射偏振后的各级 s/p 振幅（传播级）
                live = res["flux_r" if key[0] == "R" else "flux_t"] > 0
                if live.any():
                    gap = float(np.max(abs(eff[key][live] - res[key][live]) / np.maximum(1., abs(res[key][live]))))
                    self.assertLessEqual(gap, JONES_TOL, f"{name}：{key} 相差 {gap:.3e}")
        return worst, jones

    def test_builtin_examples(self):
        need_cpp(self)
        compared, report = 0, []
        for name, raw in all_examples():
            cfg = gui_models.normalize(raw)
            if not cpp_backend.eligibility(cfg)[0] or cfg["mode"] == "periodic2d":    # 二维由 test_cpp_2d_backend.py 对拍
                continue
            cfg["incident"]["M"] = SMALL_M
            with self.subTest(example=name):
                try:
                    python_solve(cfg)
                except rc.RayleighAnomalyError:
                    # 例如 examples/models/slab.json（周期 2 μm、正入射、λ = 1 μm：±2 级恰好掠射）。
                    # Python 求解器在这样的点上拒绝求解，界面也照旧报出这条提示，没有可比的数。
                    report.append(f"{name}: Python 求解器报 Rayleigh 异常点，不比较")
                    continue
                worst, jones = self.compare(name, cfg)
                report.append(f"{name}: 效率差 {worst:.2e}" + (f"，Jones 差 {jones:.2e}" if cfg["mode"] == "conical" else ""))
                compared += 1
        self.assertGreaterEqual(compared, len(CPP_PRESETS))
        say("\n    " + "\n    ".join(report))

    def test_extra_models(self):
        need_cpp(self)
        cases = [("平面 TE，吸收，边界起点不在 0", planar_model("TE")),
                 ("平面 TM，吸收", planar_model("TM")),
                 ("平面 TM，负入射角", planar_model("TM", theta_deg=-41.)),
                 ("平面 TE，正入射", planar_model("TE", theta_deg=0.)),
                 ("圆锥 Jones，φ=63°", conical_model("jones")),
                 ("圆锥 s，φ=−40°", conical_model("s", phi_deg=-40.)),
                 ("圆锥 p，负入射角", conical_model("p", phi_deg=20., theta_deg=-35.)),
                 ("圆锥 Jones，φ=0", conical_model("jones", phi_deg=0.)),
                 ("圆锥 Jones，φ=90°", conical_model("jones", phi_deg=90.))]
        report = []
        for name, cfg in cases:
            with self.subTest(case=name):
                worst, jones = self.compare(name, cfg)
                report.append(f"{name}: 效率差 {worst:.2e}" + (f"，Jones 差 {jones:.2e}" if cfg["mode"] == "conical" else ""))
        say("\n    " + "\n    ".join(report))


# --------------------------------------------------------------------------- 2. 路由与退回
class Routing(TempDirCase):
    def test_presets(self):
        # 二维默认交给 Python（cpp_backend.DEFAULT_BACKEND_2D），RCWA_2D_BACKEND=cpp 才走 C++；一维的路由与这个开关无关（test_cpp_2d_backend.Routing2D 细查）
        for value, cpp_ids in ((None, CPP_PRESETS), ("cpp", CPP_PRESETS | CPP_PRESETS_2D)):
            with EnvPatch(RCWA_2D_BACKEND=value):
                for p in gui_models.PRESETS:
                    fits, reason = cpp_backend.eligibility(gui_models.preset(p["id"]))
                    self.assertEqual(fits, p["id"] in cpp_ids, (value, p["id"]))
                    self.assertEqual(bool(reason), not fits, (value, p["id"]))

    def test_reasons(self):
        reason = lambda key: cpp_backend.eligibility(gui_models.preset(key))[1]
        self.assertIn("PML", reason("single_bar"))
        with EnvPatch(RCWA_2D_BACKEND="cpp"):
            self.assertEqual(reason("pillar_2d"), "")             # 二维可以走 C++（要设 RCWA_2D_BACKEND=cpp）；不能走的情形见 test_cpp_2d_backend.py
            too_many = gui_models.preset("pillar_2d")
            too_many["incident"].update(M=13, My=12)              # 二维谐波总数 675 > 625
            self.assertIn("二维", cpp_backend.eligibility(too_many)[1])
        with EnvPatch(RCWA_2D_BACKEND=None):
            self.assertEqual(reason("pillar_2d"), cpp_backend.NOTE_2D_PYTHON)
        self.assertEqual(reason("trapezoid_fft"), "")
        uniaxial = gui_models.preset("slab")
        uniaxial["structure"]["layers"][0]["n"] = "1.658/1.486@x"
        fits, why = cpp_backend.eligibility(gui_models.normalize(uniaxial))
        self.assertFalse(fits)
        self.assertIn("单轴", why)
        segmented = planar_model()
        segmented["structure"]["layers"][2]["n_values"][1] = "2.2/2.4@z"
        fits, why = cpp_backend.eligibility(gui_models.normalize(segmented))
        self.assertFalse(fits)
        self.assertIn("第 3 层", why)

    def test_request(self):
        request = cpp_backend.build_request(planar_model("TM"), threads=4)
        self.assertEqual(set(request), {"mode", "stack", "M", "polarization", "points", "options"})
        self.assertEqual((request["mode"], request["polarization"], request["M"]), ("planar", "TM", 12))
        self.assertEqual(request["points"], [{"wavelength": .83, "theta_deg": 27.}])
        self.assertEqual(request["options"], {"threads": 4, "parallel_layers": True})
        stack = request["stack"]
        self.assertEqual((stack["nI"], stack["nII"], stack["period"]), (1.2, 1.5, .9))
        self.assertEqual(stack["layers"][0], {"d": .25, "edges": [0., .4, .9], "n_values": [2.0, 1.2]})
        self.assertEqual(stack["layers"][1], {"d": .12, "n": [1.7, .05]})
        self.assertEqual(stack["layers"][2]["edges"], [.1, .35, .7, 1.0])
        self.assertEqual(stack["layers"][2]["n_values"], [1.5, [2.2, .02], 1.5])
        conical = cpp_backend.build_request(conical_model("jones"))
        self.assertEqual(set(conical), {"mode", "stack", "M", "points", "input", "options"})
        self.assertEqual(conical["points"], [{"wavelength": .83, "theta_deg": 27., "phi_deg": 63.}])
        self.assertEqual(conical["input"], {"psi_deg": 35., "delta_deg": 70.})
        self.assertEqual(conical["options"], {"threads": 1, "parallel_layers": False})
        self.assertEqual(cpp_backend.build_request(conical_model("s"))["input"], {"psi_deg": 90., "delta_deg": 0.})
        fft = planar_model()
        fft["structure"]["layers"][0].update(fourier_method="fft", fft_samples=1009)
        self.assertEqual(cpp_backend.build_request(fft)["stack"]["layers"][0]["fourier_method"], "fft")
        self.assertEqual(cpp_backend.build_request(fft)["stack"]["layers"][0]["fft_samples"], 1009)
        json.dumps(request), json.dumps(conical)

    def assert_python(self, metrics, *words):
        block = metrics["backend"]
        self.check_backend_block(block)
        self.assertEqual(block["efficiency"], "python")
        self.assertIsNone(block["max_efficiency_difference"])
        self.assertTrue(block["note"])
        for word in words:
            self.assertIn(word, block["note"])

    def test_worker_routes_to_python(self):
        uniaxial = gui_models.preset("slab")
        uniaxial["structure"]["layers"][0]["n"] = "1.658/1.486@x"
        # 二维模型不再在这里：它现在走 C++，路由与退回由 test_cpp_2d_backend.py 检查
        cases = [("pml", gui_models.preset("single_bar"), "PML"),
                 ("uniaxial", uniaxial, "单轴")]
        for name, cfg, word in cases:
            with self.subTest(case=name), EnvPatch(RCWA_BACKEND=None):
                metrics, out = self.run_worker(cfg, name)
                self.assert_python(metrics, word)
                self.assertIsNone(metrics["backend"]["cpp_seconds"])
                self.assertFalse((out / "cpp_request.json").exists())

    def test_forced_python(self):
        with EnvPatch(RCWA_BACKEND="python"):
            for name, cfg in (("planar", planar_model()), ("conical", conical_model()), ("pml", gui_models.preset("single_bar"))):
                with self.subTest(case=name):
                    metrics, out = self.run_worker(cfg, name)
                    self.assert_python(metrics, "RCWA_BACKEND")
                    self.assertFalse((out / "cpp_request.json").exists())

    def test_missing_executable(self):
        with EnvPatch(RCWA_FIELDS_EXE=str(self.tmp / "no_such_rcwa_json"), RCWA_BACKEND=None):
            metrics, _ = self.run_worker(planar_model())
            self.assert_python(metrics, "找不到命令行程序")
            self.assertEqual([row["m"] for row in metrics["orders"]], listed_orders(python_solve(planar_model())[1]))
            self.assertAlmostEqual(metrics["R_total"] + metrics["T_total"] + metrics["absorption"], 1., places=12)

    def test_failing_executable(self):
        fake = make_fake_executable(self.tmp)
        with EnvPatch(RCWA_BACKEND="python"):
            reference, _ = self.run_worker(planar_model(), "reference")
        cases = [("refuse", "没有编入圆锥入射", conical_model()), ("refuse", "不接受", planar_model()),
                 ("crash", "返回码 3", planar_model()), ("garbage", "应答", planar_model()),
                 ("null", "数值失败", planar_model())]
        for mode, word, cfg in cases:
            with self.subTest(mode=mode, model=cfg["mode"]):
                with EnvPatch(RCWA_FIELDS_EXE=str(fake), RCWA_FAKE_MODE=mode, RCWA_BACKEND=None):
                    metrics, out = self.run_worker(cfg, f"{mode}_{cfg['mode']}")
                self.assert_python(metrics, word)
                self.assertIsNotNone(metrics["backend"]["cpp_seconds"])
                self.assertTrue((out / "cpp_request.json").is_file())
                if cfg["mode"] == "periodic":      # 退回之后的数就是纯 Python 的数
                    self.assertEqual(metrics["orders"], reference["orders"])
                    self.assertEqual(metrics["R_total"], reference["R_total"])

    def test_real_program_refuses_a_model(self):
        """真的程序不接受一个模型时（应答 ok 为假、返回码 1）退回 Python，note 里有它的说明和出错的字段。

        例子：分段边界的首尾之差与周期相差 5e-9（相对）。界面允许到 1e-8，StableRCWA 要求不超过 1e-10。
        """
        need_cpp(self, fields=True)
        cfg = planar_model()
        layer, period = cfg["structure"]["layers"][0], cfg["structure"]["period"]
        layer["edges"][-1] = layer["edges"][0] + period * (1 + 5e-9)
        with EnvPatch(RCWA_BACKEND=None):
            metrics, out = self.run_worker(cfg)
        self.assert_python(metrics, "不接受", "stack.layers[0].edges")
        reply = json.loads((out / "cpp_response.json").read_text(encoding="utf-8"))
        self.assertIs(reply["ok"], False)
        self.assertEqual(reply["where"], "stack.layers[0].edges")
        self.assertGreaterEqual(len(metrics["orders"]), 3)          # 页面上是 Python 的数

    def test_degenerate_fields_come_from_the_reply(self):
        """圆锥应答里的 min_abs_lambda、degenerate_mode_warning 记进 backend；数值失败、退回 Python 时也一样。"""
        fake = make_fake_executable(self.tmp)
        with EnvPatch(RCWA_FIELDS_EXE=str(fake), RCWA_FAKE_MODE="degenerate", RCWA_BACKEND=None):
            metrics, _ = self.run_worker(conical_model(), "degenerate")
        self.assert_python(metrics, "数值失败")
        self.assertIs(metrics["backend"]["degenerate_mode_warning"], True)
        self.assertEqual(metrics["backend"]["min_abs_lambda"], 3e-9)
        # C++ 没有给出应答（程序拒绝、强制只用 Python）时，backend 里没有这两项
        with EnvPatch(RCWA_FIELDS_EXE=str(fake), RCWA_FAKE_MODE="refuse", RCWA_BACKEND=None):
            refused, _ = self.run_worker(conical_model(), "refused")
        with EnvPatch(RCWA_BACKEND="python"):
            forced, _ = self.run_worker(conical_model(), "forced")
        for metrics in (refused, forced):
            self.assert_python(metrics)
            self.assertNotIn("degenerate_mode_warning", metrics["backend"])
            self.assertNotIn("min_abs_lambda", metrics["backend"])

    def test_timeout(self):
        fake = make_fake_executable(self.tmp)
        pidfile = self.tmp / "fake.pid"
        with EnvPatch(RCWA_FIELDS_EXE=str(fake), RCWA_FAKE_MODE="sleep", RCWA_FAKE_PIDFILE=str(pidfile),
                      RCWA_CPP_TIMEOUT="1.5", RCWA_BACKEND=None):
            started = time.perf_counter()
            metrics, _ = self.run_worker(planar_model())
        self.assertLess(time.perf_counter() - started, 30.)
        self.assert_python(metrics, "没有返回")
        self.assertGreaterEqual(metrics["backend"]["cpp_seconds"], 1.4)
        # 超时的子进程已被终止。Windows 上这个假程序是批处理：被终止的是运行它的 cmd.exe，它下面的 Python 进程要到
        # 本进程退出时才由作业对象收走（真的 rcwa_json.exe 没有下一层进程），所以这一条只在 POSIX 上查。
        if os.name != "nt":
            self.assertFalse(process_alive(int(pidfile.read_text())))

    def test_progress_file_being_read(self):
        """写 progress.json 时界面服务恰好开着它在读：Windows 上这一刻不让替换文件，要等对方读完再换，不能报错。"""
        target = self.tmp / "progress.json"
        gui_worker.write_json(target, {"stage": "old"})
        reader = target.open("rb")                 # Windows 上这样打开的文件别的进程、别的句柄都换不掉它
        timer = threading.Timer(.3, reader.close)
        timer.start()
        self.addCleanup(timer.join)
        started = time.perf_counter()
        gui_worker.write_json(target, {"stage": "new"})
        waited = time.perf_counter() - started
        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"stage": "new"})
        self.assertFalse(target.with_suffix(".json.tmp").exists())
        if os.name == "nt":
            self.assertGreater(waited, .2)         # 确实等到了对方关闭文件


def process_alive(pid):
    """进程是否还在（POSIX 上僵尸进程不算）。"""
    if os.name == "nt":       # tasklist 找到时输出一行 CSV，第二栏是带引号的进程号；找不到时输出一句提示
        listing = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True).stdout
        return f'"{pid}"'.encode() in listing
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(state) and not state.startswith("Z")


# --------------------------------------------------------------------------- 结果页与 metrics 用的是 C++ 的数
class WorkerUsesCpp(TempDirCase):
    def test_planar(self):
        need_cpp(self, fields=True)
        cfg = planar_model("TM")
        metrics, out = self.run_worker(cfg)
        block = metrics["backend"]
        self.check_backend_block(block)
        self.assertEqual(block["efficiency"], "cpp", block["note"])
        self.assertEqual(block["note"], "")
        self.assertGreater(block["cpp_seconds"], 0.)
        self.assertLessEqual(block["max_efficiency_difference"], EFFICIENCY_TOL)
        self.assertNotIn("degenerate_mode_warning", block)     # 圆锥入射才有的两项
        self.assertNotIn("min_abs_lambda", block)
        result = json.loads((out / "cpp_response.json").read_text(encoding="utf-8"))["results"][0]
        M = cfg["incident"]["M"]
        self.assertEqual([row["m"] for row in metrics["orders"]], listed_orders(python_solve(cfg)[1]))
        self.assertGreaterEqual(len(metrics["orders"]), 3)
        for row in metrics["orders"]:              # 级次表逐位就是 C++ 应答里的数
            self.assertEqual(row["R"], result["DEr"][row["m"] + M])
            self.assertEqual(row["T"], result["DEt"][row["m"] + M])
        self.assertAlmostEqual(metrics["R_total"], result["R_total"], places=14)
        self.assertAlmostEqual(metrics["T_total"], result["T_total"], places=14)
        self.assertAlmostEqual(metrics["absorption"], 1 - result["R_total"] - result["T_total"], places=14)
        self.assertGreater(metrics["absorption"], 1e-3)       # 有吸收层
        with np.load(out / "fields.npz") as fields:             # 六场由完整 C++ 协议提供
            self.assertEqual(fields["Hy"].shape, (cfg["sampling"]["nz"], cfg["sampling"]["nx"]))
            self.assertTrue(np.isfinite(fields["E2"]).all())

    def test_conical(self):
        need_cpp(self, fields=True)
        cfg = conical_model("jones")
        metrics, out = self.run_worker(cfg)
        block = metrics["backend"]
        self.check_backend_block(block)
        self.assertEqual(block["efficiency"], "cpp", block["note"])
        self.assertLessEqual(block["max_efficiency_difference"], EFFICIENCY_TOL)
        self.assertLessEqual(block["max_jones_difference"], JONES_TOL)
        result = json.loads((out / "cpp_response.json").read_text(encoding="utf-8"))["results"][0]
        self.assertIs(block["degenerate_mode_warning"], False)
        self.assertEqual(block["min_abs_lambda"], result["min_abs_lambda"])
        self.assertGreater(block["min_abs_lambda"], 1e-2)
        pol, M = result["polarized"], cfg["incident"]["M"]
        self.assertGreaterEqual(len(metrics["orders"]), 3)
        for row in metrics["orders"]:
            k = row["m"] + M
            self.assertEqual(row["R"], pol["DEr"][k])
            self.assertEqual(row["T"], pol["DEt"][k])
            for key, source in (("R_s", "DEr_s"), ("R_p", "DEr_p"), ("T_s", "DEt_s"), ("T_p", "DEt_p")):
                self.assertAlmostEqual(row[key], pol[source][k], places=14)
        with np.load(out / "fields.npz") as fields:             # NPZ 里的 Jones 矩阵逐位是 C++ 的数
            for key in ("Jr", "Jt"):
                expected = np.array([[complex(*c) for c in row] for row in result[key]]).reshape(-1, 2, 2)
                self.assertTrue(np.array_equal(fields[key], expected), key)
            self.assertTrue(np.array_equal(fields["flux_r"], np.array(result["flux_r"])))
            self.assertTrue(np.array_equal(fields["flux_t"], np.array(result["flux_t"])))
            self.assertEqual(fields["orders"].tolist(), list(range(-M, M + 1)))
            for key in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
                self.assertTrue(np.isfinite(fields[key]).all(), key)


# --------------------------------------------------------------------------- 圆锥入射的奇点：真的程序给出的提示
class DegenerateModes(TempDirCase):
    """波长逐步靠近奇点：窗口之外不提示；窗口里及奇点上使用 C++ 和 Python Schur 传播，远近场均保持一致。"""

    def run_at(self, offset, name):
        metrics, out = self.run_worker(degenerate_model(offset), name)
        block = metrics["backend"]
        self.check_backend_block(block)
        result = json.loads((out / "cpp_response.json").read_text(encoding="utf-8"))["results"][0]
        self.assertEqual(block["min_abs_lambda"], result["min_abs_lambda"])
        self.assertEqual(block["degenerate_mode_warning"], result["degenerate_mode_warning"])
        self.assertEqual(block["n_schur"], result["n_schur"])
        return metrics, block, block["min_abs_lambda"] / DEGENERATE_KY15       # 第三项是 η = min|λ| / |ky|^1.5

    def test_outside_the_window(self):
        need_cpp(self, fields=True)
        for offset in (2e-3, -2e-3):
            with self.subTest(offset=offset):
                _, block, eta = self.run_at(offset, f"outside{offset:+g}")
                self.assertEqual(block["efficiency"], "cpp", block["note"])
                self.assertIs(block["degenerate_mode_warning"], False)
                self.assertGreater(eta, 1e-2)
                self.assertLessEqual(block["max_efficiency_difference"], EFFICIENCY_TOL)
                self.assertLessEqual(block["max_jones_difference"], JONES_TOL)

    def test_inside_the_window(self):
        need_cpp(self, fields=True)
        report = []
        for offset in (2e-4, -2e-4):
            with self.subTest(offset=offset):
                metrics, block, eta = self.run_at(offset, f"inside{offset:+g}")
                self.assertEqual(block["efficiency"], "cpp", block["note"])       # 结果有限，仍取 C++ 的数
                self.assertIs(block["degenerate_mode_warning"], True)
                self.assertTrue(1e-4 < eta < 1e-2, eta)
                self.assertEqual(block["python_n_schur"], 1)
                self.assertLessEqual(block["max_efficiency_difference"], EFFICIENCY_TOL)
                self.assertAlmostEqual(metrics["R_total"] + metrics["T_total"], 1., places=7)
                report.append(f"相对偏离 {offset:+g}: η = {eta:.2e}，效率差 {block['max_efficiency_difference']:.2e}，"
                              f"Jones 差 {block['max_jones_difference']:.2e}")
        say("\n    " + "\n    ".join(report))

    def test_on_the_singular_point(self):
        need_cpp(self, fields=True)
        metrics, block, eta = self.run_at(0., "singular")
        self.assertEqual(block["efficiency"], "cpp", block["note"])
        self.assertEqual(block["n_schur"], 1)
        self.assertIsNotNone(block["max_efficiency_difference"])
        self.assertIs(block["degenerate_mode_warning"], True)
        self.assertLess(eta, 1e-4)
        self.assertAlmostEqual(metrics["R_total"] + metrics["T_total"], 1., places=10)
        self.assertEqual(block["fields"], "cpp" if block["efficiency"] == "cpp" else "python")
        self.assertEqual(block["python_n_schur"], 1)
        self.assertLessEqual(block["max_efficiency_difference"], EFFICIENCY_TOL)
        self.assertLessEqual(block["max_jones_difference"], JONES_TOL)
        say(f"\n    奇点上：η = {eta:.1e}，Schur 传播给出的 R + T = {metrics['R_total'] + metrics['T_total']:.12f}")

    def test_no_decomposed_layer(self):
        need_cpp(self, fields=True)
        films = rc.Stack(1., [rc.Layer(.3, n=1.5, name="介质膜"), rc.Layer(.1, n=2.0 + .01j, name="吸收膜")], 1.5, .8)
        cfg = gui_models.from_stack(films, name="圆锥入射的膜系", lam0=1., theta_deg=30., M=6, pol="s", phi_deg=20.)
        metrics, out = self.run_worker(cfg)
        block = metrics["backend"]
        self.assertEqual(block["efficiency"], "cpp", block["note"])
        self.assertLessEqual(block["max_efficiency_difference"], EFFICIENCY_TOL)
        self.assertIn("min_abs_lambda", block)
        self.assertIsNone(block["min_abs_lambda"])                 # 没有做本征分解的层：应答里是 null
        self.assertIs(block["degenerate_mode_warning"], False)
        result = json.loads((out / "cpp_response.json").read_text(encoding="utf-8"))["results"][0]
        self.assertIsNone(result["min_abs_lambda"])
        self.assertEqual(result["n_eig"], 0)


# --------------------------------------------------------------------------- 现场编译的逻辑
HELLO = r'''
#include <cstdio>
int main() {
    std::puts("{\"ok\": false, \"error\": \"hello VERSION\", \"where\": \"\"}");
    return 1;
}
'''


class Building(TempDirCase):
    """把 cpp_backend 的源码、头文件、输出目录临时指到一棵小树上，检验“缺了就编、旧了重编、失败不重试”。"""

    def setUp(self):
        super().setUp()
        if not cpp_backend.find_compilers():
            self.skipTest("这台机器上找不到 C++ 编译器")
        (self.tmp / "examples").mkdir()
        (self.tmp / "include" / "stablercwa").mkdir(parents=True)
        self.source = self.tmp / "examples" / "rcwa_json.cpp"
        self.header = self.tmp / "include" / "stablercwa" / "dummy.hpp"
        self.header.write_text("// dummy\n", encoding="utf-8")
        self.write_source("1")
        saved = {k: getattr(cpp_backend, k) for k in ("SOURCE", "INCLUDE", "BIN")}
        self.addCleanup(lambda: [setattr(cpp_backend, k, v) for k, v in saved.items()])
        cpp_backend.SOURCE, cpp_backend.INCLUDE, cpp_backend.BIN = self.source, self.tmp / "include", self.tmp / "bin"
        self.env = EnvPatch(RCWA_JSON_EXE=None, RCWA_BACKEND=None)
        self.env.__enter__()
        self.addCleanup(self.env.__exit__)

    def write_source(self, version, broken=False):
        self.source.write_text(HELLO.replace("VERSION", version) + ("\n#error broken on purpose\n" if broken else ""), encoding="utf-8")

    def answer(self, exe):
        code, out, _ = cpp_backend._run_child([str(exe)], 30)
        self.assertEqual(code, 1)
        return json.loads(out)["error"]

    def later(self, path, seconds):
        stamp = time.time() + seconds
        os.utime(path, (stamp, stamp))

    def test_build_and_rebuild(self):
        stages = []
        exe, note = cpp_backend.ensure_executable(stages.append)
        self.assertIsNotNone(exe, note)
        self.assertEqual((exe, note), (self.tmp / "bin" / cpp_backend.EXE_NAME, ""))
        self.assertEqual(self.answer(exe), "hello 1")
        self.assertTrue(any("编译" in s for s in stages))
        self.assertTrue((self.tmp / "bin" / "build.log").is_file())
        self.assertFalse((self.tmp / "bin" / "build.lock").exists())
        built = exe.stat().st_mtime_ns
        stages.clear()
        self.assertEqual(cpp_backend.ensure_executable(stages.append), (exe, ""))       # 已是最新：不重编
        self.assertEqual((exe.stat().st_mtime_ns, stages), (built, []))
        self.write_source("2")
        self.later(self.source, 5)                                                      # 源码比程序新：重编
        self.assertEqual(cpp_backend.ensure_executable(), (exe, ""))
        self.assertEqual(self.answer(exe), "hello 2")
        self.later(exe, 10)
        self.later(self.header, 20)                                                     # 头文件更新也算
        stages.clear()
        cpp_backend.ensure_executable(stages.append)
        self.assertTrue(any("编译" in s for s in stages))
        vendor_header = self.header.with_name("vendor.h")
        vendor_header.write_text("// vendored FFT header\n", encoding="utf-8")
        self.assertIn(vendor_header, cpp_backend._source_files())
        self.later(vendor_header, 30)
        stages.clear()
        cpp_backend.ensure_executable(stages.append)
        self.assertTrue(any("编译" in s for s in stages))
        self.assertEqual(sorted(p.name for p in (self.tmp / "bin").iterdir()), ["build.log", cpp_backend.EXE_NAME, cpp_backend.EXE_NAME + ".sources.json"])

    def test_failed_build_is_reported_once(self):
        self.write_source("3", broken=True)
        exe, note = cpp_backend.ensure_executable()
        self.assertIsNone(exe)
        self.assertIn("编译 rcwa_json 失败", note)
        self.assertIn("broken on purpose", (self.tmp / "bin" / "build.log").read_text(encoding="utf-8"))
        log_stamp = (self.tmp / "bin" / "build.log").stat().st_mtime_ns
        stages = []
        self.assertEqual(cpp_backend.ensure_executable(stages.append), (None, note))    # 同一份源码不再重试
        self.assertEqual((stages, (self.tmp / "bin" / "build.log").stat().st_mtime_ns), ([], log_stamp))
        attempt = cpp_backend.attempt(planar_model(), self.tmp)                         # 界面一侧：退回 Python，原因写明
        self.assertFalse(attempt.used)
        self.assertIn("编译 rcwa_json 失败", attempt.note)
        self.write_source("4")
        self.later(self.source, 5)                                                      # 改好之后重新编
        exe, note = cpp_backend.ensure_executable()
        self.assertIsNotNone(exe, note)
        self.assertEqual(self.answer(exe), "hello 4")
        self.assertFalse((self.tmp / "bin" / "build_failed.json").exists())

    def test_no_compiler(self):
        with EnvPatch(RCWA_CXX="no-such-compiler-xyz"):
            exe, note = cpp_backend.ensure_executable()
            self.assertIsNone(exe)
            self.assertIn("找不到 C++ 编译器", note)
            attempt = cpp_backend.attempt(planar_model(), self.tmp)
            self.assertFalse(attempt.used)
            self.assertIn("找不到 C++ 编译器", attempt.note)

    def test_no_compiler_keeps_existing_program(self):
        exe, note = cpp_backend.ensure_executable()
        self.assertIsNotNone(exe, note)
        self.later(self.source, 5)                                                      # 程序比源码旧，又没有编译器
        with EnvPatch(RCWA_CXX="no-such-compiler-xyz"):
            again, note = cpp_backend.ensure_executable()
        self.assertEqual(again, exe)
        self.assertIn("沿用现有的程序", note)
        self.assertEqual(self.answer(exe), "hello 1")

    def test_no_source_no_executable(self):
        self.source.unlink()
        exe, note = cpp_backend.ensure_executable()
        self.assertIsNone(exe)
        self.assertIn("找不到命令行程序", note)

    def test_source_missing_uses_existing_program(self):
        exe, note = cpp_backend.ensure_executable()
        self.assertIsNotNone(exe, note)
        self.source.unlink()                                                            # 只带了编好的程序、没有源码
        self.assertEqual(cpp_backend.ensure_executable(), (exe, ""))

    def test_lock_left_by_a_dead_process(self):
        """编译途中进程被终止（取消计算、关掉界面）会留下锁文件和半成品：下一次不等它，照常编译，并把半成品清掉。"""
        gone = subprocess.Popen([sys.executable, "-c", "pass"])
        gone.wait()
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir()
        (bin_dir / "build.lock").write_text(str(gone.pid), encoding="utf-8")
        (bin_dir / f"rcwa_json.build-{gone.pid}.exe").write_bytes(b"half written")
        (bin_dir / f"rcwa_json.build-{gone.pid}.obj").write_bytes(b"half written")
        started = time.perf_counter()
        exe, note = cpp_backend.ensure_executable()
        self.assertIsNotNone(exe, note)
        self.assertLess(time.perf_counter() - started, 120.)                           # 没有去等那把作废的锁
        self.assertEqual(self.answer(exe), "hello 1")
        self.assertEqual(sorted(p.name for p in bin_dir.iterdir()), ["build.log", cpp_backend.EXE_NAME, cpp_backend.EXE_NAME + ".sources.json"])

    def test_concurrent_builds(self):
        results, stages = [], []
        def work():
            results.append(cpp_backend.ensure_executable(stages.append))
        threads = [threading.Thread(target=work) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(120)
        exe = self.tmp / "bin" / cpp_backend.EXE_NAME
        self.assertEqual(results, [(exe, "")] * 3)
        self.assertEqual(sum("正在编译" in s for s in stages), 1)                        # 只编了一次，其余的在等
        self.assertEqual(self.answer(exe), "hello 1")
        self.assertFalse((self.tmp / "bin" / "build.lock").exists())


class CompilerMessages(unittest.TestCase):
    """编译器的报错进日志之前要解码：UTF-8 的原样，本机代码页的（中文 Windows 上 MSVC 的报错是 GBK）按本机代码页。"""

    def test_utf8(self):
        self.assertEqual(cpp_backend._decode("错误 error: x".encode("utf-8")), "错误 error: x")
        self.assertEqual(cpp_backend._decode(b""), "")

    @unittest.skipUnless(os.name == "nt", "本机代码页只在 Windows 上有")
    def test_local_code_page_in_utf8_mode(self):
        """界面的启动脚本和工作进程都开着 Python 的 UTF-8 模式；这时也要按本机代码页解，不能解成乱码。"""
        text = "未声明的标识符"
        try:
            raw = text.encode("mbcs")
        except UnicodeEncodeError:
            self.skipTest("本机代码页写不出中文")
        code = "import sys, cpp_backend; sys.stdout.buffer.write(cpp_backend._decode(bytes.fromhex(sys.argv[1])).encode('utf-8'))"
        for mode in ("utf8", "utf8=0"):
            with self.subTest(mode=mode):
                done = subprocess.run([sys.executable, "-X", mode, "-c", code, raw.hex()], cwd=str(GUI), capture_output=True, timeout=120)
                self.assertEqual(done.stdout.decode("utf-8"), text, done.stderr)


# --------------------------------------------------------------------------- 3、4. 端到端与结果页
class Server:
    """起一个界面服务（--no-browser，随机端口），取令牌；close() 时结束它并删掉本次产生的结果目录。"""

    def __init__(self, **env):
        environment = os.environ.copy()
        environment.update({k: str(v) for k, v in env.items()})
        self.process = subprocess.Popen([sys.executable, "-u", str(GUI / "gui.py"), "--no-browser"], cwd=str(GUI), env=environment,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.jobs, self.url = [], None
        try:
            deadline = time.time() + 120
            while time.time() < deadline:
                line = self.process.stdout.readline().decode("utf-8", "replace")
                match = re.match(r"RCWA_GUI_URL=(\S+)", line)
                if match:
                    self.url = match.group(1)
                    break
                if not line and self.process.poll() is not None:
                    break
            if not self.url:
                raise RuntimeError("界面服务没有启动")
            self.token = re.search(r'name="rcwa-token" content="([^"]+)"', self.get("/").decode("utf-8")).group(1)
        except Exception:
            self.close()
            raise

    def get(self, path):
        with urllib.request.urlopen(self.url.rstrip("/") + path, timeout=60) as reply:
            return reply.read()

    def post(self, path, data):
        request = urllib.request.Request(self.url.rstrip("/") + path, data=json.dumps(data).encode("utf-8"),
                                         headers={"X-RCWA-Token": self.token, "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as reply:
            return json.loads(reply.read())

    def status(self, job):
        return json.loads(self.get("/api/jobs/" + job))

    def start(self, cfg):
        job = self.post("/api/run", cfg)["id"]
        self.jobs.append(job)
        return job

    def run(self, cfg, timeout=300):
        job = self.start(cfg)
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self.status(job)
            if status["status"] != "running":
                return job, status
            time.sleep(.1)
        raise AssertionError("计算超时")

    def close(self):
        if self.process.poll() is None:
            if os.name == "nt" and self.url:      # Windows 上 terminate() 是直接终止，服务来不及取消还在算的任务
                for job in self.jobs:
                    try:
                        self.post("/api/cancel", {"id": job})
                    except (OSError, ValueError):
                        pass
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=15)
        self.process.stdout.close()
        for job in self.jobs:
            shutil.rmtree(GUI / "results" / "gui_runs" / job, ignore_errors=True)


def find_chrome():
    candidates = [os.environ.get("CHROME"), "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
    candidates += [shutil.which(name) for name in ("google-chrome", "chromium", "chromium-browser", "chrome")]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def run_chrome(arguments, finished, attempts=3, timeout=60):
    """运行一次无头 Chrome，返回 (标准输出, 标准错误)。

    它偶尔在结果已经写出之后不肯退出：限时等待，结果到手（finished(标准输出) 为真）就不再等；
    没到手则重试。
    """
    for _ in range(attempts):
        try:
            done = subprocess.run(arguments, capture_output=True, timeout=timeout)
            out, err = done.stdout, done.stderr
        except subprocess.TimeoutExpired as late:
            out, err = late.stdout or b"", late.stderr or b""
        if finished(out):
            return out, err
    raise AssertionError("无头 Chrome 没有给出结果")


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = Server(RCWA_VERIFY_PYTHON="1")
        cls.addClassCleanup(cls.server.close)
        cls.cpp = _STATE["fields_exe"] is not None and not cpp_backend.forced_python()
        cls.done = {}

    def finished(self, name):
        """各模型只算一次，后面的检查共用。"""
        if name not in self.done:
            cfg = {"planar": planar_model("TE"), "conical": conical_model("jones"), "pml": gui_models.preset("single_bar"),
                   "warned": degenerate_model(2e-4), "singular": degenerate_model(0.)}[name]
            job, status = self.server.run(cfg)
            self.assertEqual(status["status"], "done", status.get("error"))
            self.done[name] = (job, status, cfg)
        return self.done[name]

    def metrics_file(self, job):
        return json.loads(self.server.get(f"/api/jobs/{job}/files/metrics.json"))

    def check(self, name, expect_cpp):
        job, status, cfg = self.finished(name)
        metrics = self.metrics_file(job)
        block = metrics["backend"]
        self.assertEqual(status["metrics"]["backend"], block)
        for key in ("efficiency", "fields", "cpp_seconds", "python_solve_seconds", "max_efficiency_difference", "note"):
            self.assertIn(key, block)
        self.assertEqual(block["fields"], "cpp" if block["efficiency"] == "cpp" else "python")
        self.assertEqual(block["efficiency"], "cpp" if expect_cpp else "python", block["note"])
        if expect_cpp:
            self.assertLessEqual(block["max_efficiency_difference"], EFFICIENCY_TOL)
            if name == "singular":
                self.assertEqual(block["python_n_schur"], 1)
                self.assertAlmostEqual(metrics["R_total"] + metrics["T_total"], 1., places=10)
            self.assertGreater(block["cpp_seconds"], 0.)
        else:
            self.assertTrue(block["note"])
            self.assertIsNone(block["max_efficiency_difference"])
        return metrics, block, cfg

    def test_planar(self):
        metrics, block, cfg = self.check("planar", self.cpp)
        _, reference = python_solve(cfg)                     # 级次表与独立算的 Python 结果一致
        self.assertEqual([row["m"] for row in metrics["orders"]], listed_orders(reference))
        self.assertGreaterEqual(len(metrics["orders"]), 3)
        for row in metrics["orders"]:
            self.assertEqual(set(row), {"m", "R", "T"})
            self.assertTrue(0 <= row["R"] <= 1 and 0 <= row["T"] <= 1)
        self.assertAlmostEqual(sum(row["R"] for row in metrics["orders"]), metrics["R_total"], places=8)
        self.assertAlmostEqual(sum(row["T"] for row in metrics["orders"]), metrics["T_total"], places=8)
        self.assertTrue(0 < metrics["absorption"] < 1)
        for row in metrics["orders"]:
            k = row["m"] + cfg["incident"]["M"]
            self.assertAlmostEqual(row["R"], float(reference.DEr[k]), delta=EFFICIENCY_TOL)
            self.assertAlmostEqual(row["T"], float(reference.DEt[k]), delta=EFFICIENCY_TOL)

    def test_conical(self):
        metrics, block, cfg = self.check("conical", self.cpp)
        if self.cpp:
            self.assertLessEqual(block["max_jones_difference"], JONES_TOL)
        _, reference = python_solve(cfg)
        self.assertEqual([row["m"] for row in metrics["orders"]], listed_orders(reference))
        self.assertGreaterEqual(len(metrics["orders"]), 3)
        for row in metrics["orders"]:
            self.assertEqual(set(row), {"m", "R", "T", "R_s", "R_p", "T_s", "T_p"})
            self.assertAlmostEqual(row["R_s"] + row["R_p"], row["R"], places=12)
            self.assertAlmostEqual(row["T_s"] + row["T_p"], row["T"], places=12)
            k = row["m"] + cfg["incident"]["M"]
            self.assertAlmostEqual(row["R"], float(reference.DEr[k]), delta=EFFICIENCY_TOL)
            self.assertAlmostEqual(row["T"], float(reference.DEt[k]), delta=EFFICIENCY_TOL)

    def test_degenerate_warning(self):
        """奇点附近与奇点上都使用 C++ Schur 传播，诊断经服务原样传到 metrics。"""
        if not self.cpp:
            self.skipTest("没有可用的 rcwa_json")
        _, block, _ = self.check("warned", True)
        self.assertIs(block["degenerate_mode_warning"], True)
        self.assertTrue(1e-4 < block["min_abs_lambda"] / DEGENERATE_KY15 < 1e-2)
        _, block, _ = self.check("singular", True)
        self.assertIs(block["degenerate_mode_warning"], True)
        self.assertEqual(block["n_schur"], 1)
        _, block, _ = self.check("conical", True)
        self.assertIs(block["degenerate_mode_warning"], False)

    def test_pml(self):
        metrics, block, _ = self.check("pml", False)
        self.assertIn("RCWA_BACKEND" if cpp_backend.forced_python() else "PML", block["note"])
        self.assertEqual(metrics["orders"], [])
        self.assertIsNone(metrics["R_total"])

    def test_result_page(self):
        chrome = find_chrome()
        if not chrome:
            self.skipTest("找不到无头 Chrome")
        keep = os.environ.get("RCWA_TEST_KEEP")
        folder = Path(keep) if keep else Path(tempfile.mkdtemp(prefix="rcwa_cpp_shots_"))
        folder.mkdir(parents=True, exist_ok=True)
        if not keep:
            self.addCleanup(shutil.rmtree, folder, True)
        common = [chrome, "--headless=new", "--disable-gpu", "--window-size=1400,1000", "--virtual-time-budget=8000",
                  "--enable-logging=stderr", "--v=0"]
        for name in ("planar", "conical", "pml") + (("warned", "singular") if self.cpp else ()):
            with self.subTest(page=name):
                job, status, _ = self.finished(name)
                block = status["metrics"]["backend"]
                self.assertEqual(bool(block.get("degenerate_mode_warning")), name in ("warned", "singular"))
                page = f"{self.server.url}?result={job}"
                png = folder / f"result_{name}.png"
                if png.exists():
                    png.unlink()
                _, shot_log = run_chrome(common + [f"--screenshot={png}", page],
                                         lambda out: png.is_file() and png.stat().st_size > 20000)
                dom, dom_log = run_chrome(common + ["--dump-dom", page], lambda out: b"</html>" in out)
                html = dom.decode("utf-8", "replace")
                (folder / f"result_{name}.dom.html").write_text(html, encoding="utf-8")
                console = [line for line in (shot_log + dom_log).decode("utf-8", "replace").splitlines()
                           if ":CONSOLE" in line and ("Uncaught" in line or "Error" in line)]
                self.assertEqual(console, [], "结果页有脚本错误")
                self.assertRegex(html, r'<div id="errorBox"[^>]*\bhidden\b')
                self.assertRegex(html, r'<div id="resultContent"(?![^>]*\bhidden\b)[^>]*>')
                line = re.search(r'<p id="resultBackend"([^>]*)>(.*?)</p>', html, re.S)
                self.assertIsNotNone(line)
                self.assertNotIn("hidden", line.group(1))
                text = line.group(2)
                self.assertIn("场图：C++" if block["fields"] == "cpp" else "场图：Python", text)
                if block["efficiency"] == "cpp":
                    self.assertIn("衍射效率：C++", text)
                    self.assertIn("最大相差", text)
                    self.assertIn("C++ 用时", text)
                else:
                    self.assertIn("衍射效率：Python", text)
                    self.assertIn("未用 C++", text)
                    self.assertIn(block["note"].replace("&", "&amp;"), text)
                if block.get("degenerate_mode_warning") and block.get("n_schur", 0) > 0 and block["efficiency"] == "cpp":
                    self.assertIn("Schur 传播", text)
                    self.assertIn("场图保留完整矩阵耦合", text)
                elif block.get("degenerate_mode_warning"):
                    self.assertIn(SENTENCE, text)
                    self.assertIn("稍微挪动一点", text)
                else:
                    self.assertNotIn(SENTENCE, text)
                rows = re.search(r'<tbody id="orderRows">(.*?)</tbody>', html, re.S).group(1)
                self.assertEqual(rows.count("<tr>"), len(status["metrics"]["orders"]))
        if keep:
            say(f"\n    截图留在 {folder}")


class Cancellation(TempDirCase):
    """取消计算时，正在运行的 C++ 子进程不能留在后台。"""

    def test_cancel_kills_child(self):
        fake = make_fake_executable(self.tmp)
        pidfile = self.tmp / "fake.pid"
        server = Server(RCWA_FIELDS_EXE=fake, RCWA_FAKE_MODE="sleep", RCWA_FAKE_PIDFILE=pidfile, RCWA_BACKEND="")
        self.addCleanup(server.close)
        job = server.start(planar_model())
        deadline = time.time() + 60
        while not pidfile.exists() or not pidfile.read_text().strip():
            self.assertLess(time.time(), deadline, "假的 C++ 程序没有启动")
            time.sleep(.05)
        child = int(pidfile.read_text())
        self.assertTrue(process_alive(child))
        self.assertEqual(server.status(job)["stage"], "正在用 C++ 计算衍射效率、六分量场与相位")
        started = time.time()
        self.assertEqual(server.post("/api/cancel", {"id": job})["status"], "cancelled")
        deadline = time.time() + 10
        while process_alive(child) and time.time() < deadline:
            time.sleep(.05)
        self.assertFalse(process_alive(child), "取消之后 C++ 子进程还在运行")
        self.assertLess(time.time() - started, 5.)
        self.assertEqual(server.status(job)["status"], "cancelled")
        _, status = server.run(gui_models.preset("single_bar"))      # 取消之后还能接着算
        self.assertEqual(status["status"], "done")


if __name__ == "__main__":
    unittest.main()
