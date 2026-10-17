"""Full C++ GUI path: frozen fields/phase, no Python numerical calls, protocol rejection and cleanup.

Run from gui/: python -m unittest validation.test_cpp_fields_backend -v
RCWA_FIELDS_EXE may select a built production rcwa_fields executable. No corpus is regenerated.
"""
import copy
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

GUI = Path(__file__).resolve().parents[1]
if str(GUI) not in sys.path:sys.path.insert(0, str(GUI))
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import cpp_backend as common
import cpp_fields_backend as backend
import gui_fields
import gui_models
import gui_worker
import rcwa1d as rc
import field_phase
from conical import rcwa_conical, fields_conical
from validation.field_reference_cases import build_stack as reference_stack

STATE = {}
REFERENCE = Path(__file__).parent / "field_reference"


@contextmanager
def environment(**values):
    saved = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:os.environ.pop(key, None)
            else:os.environ[key] = str(value)
        yield
    finally:
        for key, value in saved.items():
            if value is None:os.environ.pop(key, None)
            else:os.environ[key] = value


def setUpModule():
    with environment(RCWA_BACKEND=None, RCWA_JSON_EXE=None):
        STATE["exe"], STATE["note"] = backend.ensure_executable()


def model(pol="TE", conical=False):
    stack = rc.Stack(1.2, [rc.Layer(.2, edges=[0., .21, .6], n_values=[1.7, 1.]),
                           rc.Layer(.09, n=1.46+.02j)], 1.4, .6)
    cfg = gui_models.from_stack(stack, name="中文 场测试", lam0=.83, theta_deg=27., M=3,
                                pol="jones" if conical else pol, phi_deg=31. if conical else None,
                                psi_deg=37., delta_deg=71.)
    cfg["sampling"].update(nx=41, nz=23, z_min=-.12, z_max=.39)
    return cfg


class TemporaryCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rcwa 中文 fields "))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def need_cpp(self):
        if STATE.get("exe") is None:self.skipTest("rcwa_fields unavailable: " + STATE.get("note", ""))

    @contextmanager
    def cpp_environment(self, **extra):
        with environment(RCWA_FIELDS_EXE=STATE["exe"], RCWA_JSON_EXE=None, RCWA_BACKEND=None,
                         RCWA_VERIFY_PYTHON=None, **extra):yield

    def run_worker(self, cfg, name="result"):
        result = self.tmp / name
        metrics = gui_worker.compute(cfg, result)
        self.assertEqual(metrics, json.loads((result / "metrics.json").read_text(encoding="utf-8")))
        with np.load(result / "fields.npz", allow_pickle=False) as archive:
            fields = {key: archive[key] for key in archive.files}
        return metrics, fields, result


class FrozenFields(TemporaryCase):
    def test_actual_cli_all_twelve_complex_fields_and_masked_phase(self):
        self.need_cpp()
        manifest = json.loads((REFERENCE / "reference_v1.json").read_text(encoding="utf-8"))
        archive = REFERENCE / manifest["archive"]["filename"]
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), manifest["archive"]["sha256"])
        worst = 0.
        with np.load(archive, allow_pickle=False) as baseline:
            for record in manifest["cases"]:
                spec = record["parameters"];case = spec["id"]
                with self.subTest(case=case):
                    solve = {"mode":spec["mode"], "stack":spec["stack"], "M":spec["M"],
                             "wavelength":spec["wavelength"], "theta_deg":spec["theta_deg"]}
                    if spec["mode"] == "conical":
                        solve.update(phi_deg=spec["phi_deg"], input={"jones":spec["jones_in"]} if spec["jones_in"]
                                     else {"psi_deg":spec["psi_deg"], "delta_deg":spec["delta_deg"]})
                    else:solve["polarization"] = spec["polarization"]
                    directory = self.tmp / case
                    request = {"solve":solve, "sampling":{"z":spec["zs"], "nx":spec["nx"], "y":spec["y"]},
                               "output_dir":str(directory), "request_id":case}
                    code, out, err = common._run_child([str(STATE["exe"])], 90, text=json.dumps(request))
                    response = json.loads(out)
                    self.assertEqual(code, 0, response if response else err)
                    self.assertLess(len(out), 50000, "field grids must never go through stdout JSON")
                    actual = np.load(directory / "fields.npy", allow_pickle=False)
                    expected = baseline[case+"__fields"]
                    self.assertEqual(actual.dtype.str, "<c16")
                    self.assertTrue(actual.flags.c_contiguous)
                    np.testing.assert_allclose(actual, expected, **spec["tolerance"])
                    worst = max(worst, float(np.max(abs(actual-expected))))
                    phases = np.load(directory / "phase.npy", allow_pickle=False)
                    masks = np.load(directory / "phase_mask.npy", allow_pickle=False).astype(bool)
                    valid = ~baseline[case+"__phase_comparison_mask"]
                    self.assertFalse(masks[:,valid].any())
                    error = spec["tolerance"]["atol"]+spec["tolerance"]["rtol"]*abs(expected)
                    for index, reference in enumerate(("phase_absolute", "phase_incident")):
                        difference = abs(np.exp(1j*phases[index][valid])-np.exp(1j*baseline[case+"__"+reference][valid]))
                        self.assertTrue(np.all(difference <= 2*error[valid]/abs(expected[valid])+5e-14), reference)
                    self.assertTrue(np.isnan(phases[masks]).all())
                    meta = response["field_data"]
                    self.assertEqual(sum(layer["finite_zero_modes"] for layer in meta["layers"]), spec["expected_zero_modes"])
                    self.assertEqual(sum(layer["schur"] for layer in meta["layers"]), spec["expected_schur_layers"])
        print("\n    full CLI twelve frozen cases max complex-field difference: %.3e" % worst)


class FullCppWorker(TemporaryCase):
    def test_success_never_calls_python_solver_sampler_or_phase(self):
        self.need_cpp()
        manifest = json.loads((REFERENCE / "reference_v1.json").read_text(encoding="utf-8"))
        schur = next(record["parameters"] for record in manifest["cases"] if record["parameters"]["id"] == "schur_ellipse")
        schur_cfg = gui_models.from_stack(reference_stack(schur["stack"]), name="Schur", lam0=schur["wavelength"],
            theta_deg=schur["theta_deg"], M=schur["M"], pol="jones", phi_deg=schur["phi_deg"], psi_deg=35., delta_deg=80.)
        schur_cfg["sampling"].update(nx=41, nz=23)
        fft_models = [model("TE"), model("TM"), model(conical=True)]
        for cfg in fft_models:
            cfg["structure"]["layers"][0].update(fourier_method="fft", fft_samples=1009)
        with self.cpp_environment(), ExitStack() as patches:
            for module, name in ((rc,"solve"), (rc,"field_map"), (rcwa_conical,"solve_conical"),
                                 (fields_conical,"field_map"), (field_phase,"phase_data"), (np,"angle")):
                patches.enter_context(patch.object(module, name, side_effect=AssertionError("Python numerical call: "+name)))
            for index, cfg in enumerate((model("TE"), model("TM"), model(conical=True), schur_cfg, *fft_models)):
                with self.subTest(model=index):
                    metrics, fields, folder = self.run_worker(cfg, str(index))
                    block = metrics["backend"]
                    for key in ("efficiency","fields","phase"):self.assertEqual(block[key], "cpp", block["note"])
                    self.assertFalse(block["python_solve_ran"])
                    self.assertEqual(block["python_solve_seconds"], 0.)
                    self.assertIsNone(block["max_efficiency_difference"])
                    if index >= 4:
                        info = metrics["fourier_layers"][0]
                        self.assertEqual((info["method"], info["fft_samples"]), ("fft", 1009))
                    for component in gui_fields.COMPONENTS:
                        self.assertTrue(np.isfinite(fields[component]).all())
                        for quantity in ("phase","phase_relative"):
                            displayed, info = gui_fields.view_data(fields, cfg, component, quantity)
                            self.assertEqual(info["phase_backend"], "cpp")
                            expected = fields["phase_values"][int(quantity=="phase_relative"),gui_fields.COMPONENTS.index(component)]
                            np.testing.assert_allclose(displayed, expected, atol=0, rtol=0, equal_nan=True)
                    if cfg["mode"] == "conical":
                        for key in ("Jr","Jt"):self.assertEqual(fields[key].shape, (2*cfg["incident"]["M"]+1,2,2))
                        self.assertAlmostEqual(float(np.linalg.norm(fields["jones_in"])),1.)
                    exported = gui_fields.export_six_fields(folder)
                    with np.load(exported, allow_pickle=False) as archive:
                        self.assertEqual(archive["phase_archive_version"], 1)
                        np.testing.assert_array_equal(archive["phase_masks"], fields["phase_masks"])

    def test_fft_six_fields_match_python(self):
        self.need_cpp()
        worst = 0.
        for pol, conical in (("TE", False), ("TM", False), ("TE", True)):
            cfg = model(pol, conical=conical)
            cfg["structure"]["layers"][0].update(fourier_method="fft", fft_samples=1009)
            # Full workers independently reconstruct all six fields on the same grid.
            with self.cpp_environment():
                cpp_metrics, cpp_fields, _ = self.run_worker(cfg, pol + str(conical) + "_cpp")
            with self.cpp_environment(), environment(RCWA_BACKEND="python"):
                _, py_fields, _ = self.run_worker(cfg, pol + str(conical) + "_python")
            self.assertEqual(cpp_metrics["backend"]["fields"], "cpp")
            for component in gui_fields.COMPONENTS:
                np.testing.assert_allclose(cpp_fields[component], py_fields[component], atol=2e-9, rtol=2e-9)
                worst = max(worst, float(np.max(abs(cpp_fields[component] - py_fields[component]))))
        print("\n    FFT six-field C++/Python max difference: %.3e" % worst)

    def test_explicit_verification_failure_preserves_cpp_success(self):
        self.need_cpp()
        with self.cpp_environment(), environment(RCWA_VERIFY_PYTHON="1"), patch.object(rc, "solve", side_effect=rc.RayleighAnomalyError("oracle cannot solve")):
            metrics, fields, _ = self.run_worker(model())
        block = metrics["backend"]
        self.assertEqual(block["fields"], "cpp")
        self.assertEqual(block["verification_status"], "failed")
        self.assertTrue(block["python_solve_ran"])
        self.assertIn("保留 C++", block["note"])
        self.assertIsNone(block["max_efficiency_difference"])
        self.assertTrue(np.isfinite(fields["E2"]).all())

    def test_explicit_verification_reports_actual_difference(self):
        self.need_cpp()
        with self.cpp_environment(), environment(RCWA_VERIFY_PYTHON="1"):
            metrics, _, _ = self.run_worker(model(conical=True))
        block = metrics["backend"]
        self.assertEqual(block["verification_status"], "compared")
        self.assertTrue(block["python_solve_ran"])
        self.assertLess(block["max_efficiency_difference"], 1e-10)
        self.assertLess(block["max_jones_difference"], 1e-10)

    def test_old_archive_and_invalid_new_phase_version(self):
        self.need_cpp()
        cfg = model("TM")
        with self.cpp_environment():_, fields, _ = self.run_worker(cfg)
        legacy = {key:value for key,value in fields.items() if not key.startswith(("phase_","field_"))}
        for quantity in ("phase","phase_relative"):
            expected, _ = gui_fields.view_data(legacy, cfg, "Hy", quantity)
            actual, _ = gui_fields.view_data(fields, cfg, "Hy", quantity)
            valid = np.isfinite(expected) & np.isfinite(actual)
            np.testing.assert_allclose(np.exp(1j*actual[valid]), np.exp(1j*expected[valid]), atol=2e-12, rtol=0)
        fields["phase_archive_version"] = np.asarray(99)
        with self.assertRaisesRegex(ValueError, "版本"):
            gui_fields.view_data(fields, cfg, "Hy", "phase")

    def test_unicode_space_executable_and_output_paths(self):
        self.need_cpp()
        executable = self.tmp / ("场 程序.exe" if os.name == "nt" else "场 程序")
        shutil.copy2(STATE["exe"], executable)
        with self.cpp_environment(), environment(RCWA_FIELDS_EXE=executable):
            metrics, _, _ = self.run_worker(model(), "六复场 中文 路径")
        self.assertEqual(metrics["backend"]["fields"], "cpp", metrics["backend"]["note"])


class Fallback(TemporaryCase):
    def test_forced_unsupported_missing_and_old_override(self):
        cases = []
        forced = model();cases.append(("forced", forced, {"RCWA_BACKEND":"python"}, "RCWA_BACKEND"))
        uniaxial = model();uniaxial["structure"]["layers"][0]["n_values"][0] = "1.658/1.486@x"
        cases.append(("uniaxial", uniaxial, {}, "单轴"))
        cases.append(("missing", model(), {"RCWA_FIELDS_EXE":self.tmp/"不存在"}, "不存在"))
        cases.append(("old", model(), {"RCWA_FIELDS_EXE":None,"RCWA_JSON_EXE":self.tmp/"old_far"}, "旧效率协议"))
        for name, cfg, extra, reason in cases:
            with self.subTest(case=name), environment(RCWA_FIELDS_EXE=STATE.get("exe"), RCWA_JSON_EXE=None,
                                                     RCWA_BACKEND=None, RCWA_VERIFY_PYTHON=None), environment(**extra):
                metrics, fields, _ = self.run_worker(cfg, name)
                for key in ("efficiency","fields","phase"):self.assertEqual(metrics["backend"][key], "python")
                self.assertIn(reason, metrics["backend"]["note"])
                self.assertNotIn("phase_backend", fields)

    def test_valid_npy_with_corrupt_protocol_falls_back_once(self):
        self.need_cpp()
        real_call = common.call
        mutations = ("dtype","endian","fortran","shape","truncated","nonfinite","phase","mask",
                     "version","request_id","old_far_only","timing","layer_count","layer_type","solve_diagnostic")
        for index, mutation in enumerate(mutations):
            def corrupt(exe, request, folder, timeout=None):
                response = real_call(exe, request, folder, timeout)
                directory = Path(request["output_dir"])
                meta = response["field_data"]
                path = directory / "fields.npy"
                if mutation in ("dtype","endian","fortran","shape","nonfinite"):
                    value = np.load(path, allow_pickle=False)
                    if mutation == "dtype":value = value.real
                    if mutation == "endian":value = value.astype(">c16")
                    if mutation == "fortran":value = np.asfortranarray(value)
                    if mutation == "shape":value = value[:1]
                    if mutation == "nonfinite":value[0,0,0] = np.nan
                    np.save(path, value, allow_pickle=False)
                elif mutation == "truncated":path.write_bytes(path.read_bytes()[:-1])
                elif mutation in ("phase","mask"):
                    path = directory / ("phase.npy" if mutation == "phase" else "phase_mask.npy")
                    value = np.load(path, allow_pickle=False)
                    value.flat[41*23] = np.inf if mutation == "phase" else 2
                    np.save(path, value, allow_pickle=False)
                elif mutation == "version":meta["version"] = 99
                elif mutation == "request_id":meta["request_id"] = "stale"
                elif mutation == "old_far_only":response.pop("field_data");response.pop("protocol")
                elif mutation == "timing":meta["timing"]["solve"] = "bad"
                elif mutation == "layer_count":meta["layers"] = []
                elif mutation == "layer_type":meta["layers"][0]["finite_zero_modes"] = "bad"
                elif mutation == "solve_diagnostic":response["results"][0]["timing"] = [1,2]
                return response
            with self.subTest(case=mutation), self.cpp_environment(), patch.object(common,"call",side_effect=corrupt), patch.object(rc,"solve",wraps=rc.solve) as solver:
                metrics, _, _ = self.run_worker(model(), "corrupt"+str(index))
                self.assertEqual(metrics["backend"]["fields"], "python", mutation)
                self.assertTrue(metrics["backend"]["note"])
                self.assertEqual(solver.call_count, 1, "fallback must perform exactly one Python solve")

    def test_atomic_cli_failure_and_existing_directory_rejection(self):
        self.need_cpp()
        cfg = gui_models.normalize(model())
        directory = self.tmp / "already there"
        directory.mkdir();(directory/"marker").write_text("keep")
        request = backend.build_request(cfg, directory, "unique")
        code, out, _ = common._run_child([str(STATE["exe"])], 30, text=json.dumps(request))
        self.assertEqual(code, 1);self.assertFalse(json.loads(out)["ok"])
        self.assertEqual((directory/"marker").read_text(), "keep")
        request["output_dir"] = str(self.tmp/"failed sampling")
        # Finite input whose propagated phase overflows; staging already exists when sampling fails.
        request["sampling"]["z"] = [1e308]
        code, out, _ = common._run_child([str(STATE["exe"])], 30, text=json.dumps(request))
        self.assertEqual(code, 1);self.assertFalse(json.loads(out)["ok"])
        self.assertFalse(Path(request["output_dir"]).exists())
        self.assertEqual(list(self.tmp.glob("*.tmp-*")), [])


class ChildLifetime(TemporaryCase):
    def fake_sleep(self):
        script = self.tmp / "sleep.py"
        script.write_text("import os,sys,time\nopen(os.environ['RCWA_FIELDS_TEST_PID'],'w').write(str(os.getpid()))\n"
                          "sys.stdin.buffer.read()\ntime.sleep(120)\n", encoding="utf-8")
        executable = self.tmp / ("sleep.bat" if os.name == "nt" else "sleep")
        executable.write_text((f'@"{sys.executable}" "{script}" %*\r\n' if os.name == "nt" else
                               f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n'), encoding="utf-8")
        if os.name != "nt":executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
        return executable

    def test_timeout_falls_back_and_terminates_child(self):
        executable = self.fake_sleep();pidfile = self.tmp / "timeout.pid"
        with environment(RCWA_FIELDS_EXE=executable, RCWA_FIELDS_TEST_PID=pidfile, RCWA_CPP_TIMEOUT=".3",
                         RCWA_BACKEND=None, RCWA_JSON_EXE=None, RCWA_VERIFY_PYTHON=None):
            metrics, _, _ = self.run_worker(model())
        self.assertEqual(metrics["backend"]["fields"], "python")
        self.assertIn("已终止", metrics["backend"]["note"])
        if os.name != "nt":self.assertTrue(common._process_gone(int(pidfile.read_text())))

    def test_cancel_worker_terminates_cpp_child(self):
        if os.name == "nt":self.skipTest("Windows job lifetime requires native executable; existing cancellation integration covers it")
        executable = self.fake_sleep();pidfile = self.tmp / "cancel.pid"
        config = self.tmp / "config.json";config.write_text(json.dumps(model()), encoding="utf-8")
        env = os.environ.copy()
        env.update(RCWA_FIELDS_EXE=str(executable), RCWA_FIELDS_TEST_PID=str(pidfile), RCWA_BACKEND="", RCWA_VERIFY_PYTHON="")
        worker = subprocess.Popen([sys.executable, str(GUI/"gui_worker.py"), str(config), str(self.tmp/"cancelled")],
                                  cwd=GUI, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.addCleanup(lambda: worker.kill() if worker.poll() is None else None)
        deadline = time.monotonic()+15
        while not pidfile.exists() or not pidfile.read_text().strip():
            self.assertLess(time.monotonic(), deadline, "C++ child did not start")
            time.sleep(.02)
        child = int(pidfile.read_text());worker.terminate();worker.communicate(timeout=10)
        self.assertTrue(common._process_gone(child), "cancel left a C++ process running")
        self.assertFalse((self.tmp/"cancelled"/"metrics.json").exists())


if __name__ == "__main__":unittest.main()
