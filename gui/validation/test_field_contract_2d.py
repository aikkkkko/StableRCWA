"""Frozen 2-D RCWA reference: integrity, same-environment reproduction and independent checks.

Reads gui/validation/field_reference_2d/reference_2d_v1.{json,npz} and tests/rcwa2d_reference.inc;
never regenerates or replaces them. Run from the gui directory:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m unittest validation.test_field_contract_2d -v

RCWA2D_SKIP_BIG=1 skips re-solving the N=625 case (about 40 s); its frozen data are still integrity-checked.

(a) SHA-256 / sizes / layout of the frozen files;
(b) recomputation with the current gui/2drcwa/rcwa2d.py reproduces the frozen values (JSON tolerance.reproduce);
(c) independent checks that do not use rcwa2d.py as the oracle: Airy / Fresnel six-component fields of uniform
    stacks, rcwa1d.solve and conical.solve_conical (far fields and six-component harmonics) for y- and x-invariant
    gratings, energy conservation, tangential continuity at the layer interfaces (selftest2d T5), x<->y symmetry
    of the C4 pillar, the GUI phase display, and the C++ table tests/rcwa2d_reference.inc.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import unittest

GUI = Path(__file__).resolve().parents[1]
for _path in (GUI, GUI / "2drcwa"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
import rcwa1d as rc1
from conical.rcwa_conical import solve_conical
from conical.fields_conical import field_at as conical_at
from gui_fields import view_data
from validation.field_reference_2d_cases import (COMPONENTS, CONTRACT_VERSION, FAR_ARRAYS, FROZEN_DIR, ROOT,
                                                 case_specs, complex_value, failure_info, jones_vector,
                                                 sample_case, slice_edge_distances, solve_case)

SKIP_BIG = bool(os.environ.get("RCWA2D_SKIP_BIG"))
INC = ROOT / "tests" / "rcwa2d_reference.inc"
SPECS = {spec["id"]: spec for spec in case_specs()}
OK_SPECS = [spec for spec in case_specs() if spec["status"] == "ok"]
_CACHE = {}
_MAX = {}            # largest error seen per check, printed at the end (report numbers)


def note_max(name, value):
    _MAX[name] = max(_MAX.get(name, 0.0), float(value))


def sampled(case_id):
    if case_id not in _CACHE:
        _CACHE[case_id] = sample_case(SPECS[case_id])
    return _CACHE[case_id]


def wanted(spec):
    return not (SKIP_BIG and spec["N"] > 289)


def allclose_max(actual, reference, atol, rtol, label=""):
    """assert_allclose that also returns max |a-b| (for the printed summary)."""
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, err_msg=label)
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(reference)))) if np.size(actual) else 0.0


# ---------------------------------------------------------------- independent analytic model of uniform stacks
def kz_over_k0(n, kt):
    """sqrt(n^2 - kt^2) with Im >= 0 (propagating: positive real, evanescent: +i|.|)."""
    kz = np.sqrt(complex(n * n - kt * kt))
    return -kz if kz.imag < 0 else kz


def stratified(spec, zs, pol):
    """Scalar TE / TM zero-order solution of a stack of uniform layers from the boundary linear system.

    Returns (r, t, S(z), U(z), n(z)) with TE: S = E_perp, TM: S = Z0 H_perp (incident amplitude 1) and U = i P (down - up).
    Amplitudes are anchored at the layer tops / z = 0 / z_bot so no exponential grows.
    """
    st = spec["stack"]
    nI, nII = float(st["nI"]), complex_value(st["nII"])
    layers = [(layer["d"], complex_value(layer["n"])) for layer in st["layers"]]
    k0 = 2 * np.pi / spec["wavelength"]
    kt = nI * np.sin(np.radians(spec["theta_deg"]))
    media = [nI] + [n for _, n in layers] + [nII]
    kz = [k0 * kz_over_k0(n, kt) for n in media]
    adm = [kz[i] / k0 / (1 if pol == "TE" else media[i] ** 2) for i in range(len(media))]
    L = len(layers)
    thick = [0.0] + [d for d, _ in layers]
    # unknowns x = [r, a_1, b_1, ..., a_L, b_L, t]; interface i joins medium i (above) and i+1 (below);
    # rows 2i / 2i+1 state  S_above - S_below = 0  and  U_above - U_below = 0.
    A = np.zeros((2 * (L + 1), 2 * L + 2), complex)
    rhs = np.zeros(2 * (L + 1), complex)
    for i in range(L + 1):
        row = 2 * i
        if i == 0:                     # S = 1 + r, U = i P0 (1 - r) at z = 0 (the incident wave is the known constant)
            A[row, 0], rhs[row] = 1.0, -1.0
            A[row + 1, 0], rhs[row + 1] = -1j * adm[0], -1j * adm[0]
        else:                          # layer i, evaluated at its bottom
            ia = 1 + 2 * (i - 1)
            down, up = np.exp(1j * kz[i] * thick[i]), np.exp(-1j * kz[i] * thick[i])
            A[row, ia], A[row, ia + 1] = down, up
            A[row + 1, ia], A[row + 1, ia + 1] = 1j * adm[i] * down, -1j * adm[i] * up
        if i == L:                     # region II: S = t, U = i P t
            A[row, 2 * L + 1] -= 1.0
            A[row + 1, 2 * L + 1] -= 1j * adm[i + 1]
        else:                          # layer i+1, evaluated at its top
            ib = 1 + 2 * i
            A[row, ib] -= 1.0
            A[row, ib + 1] -= 1.0
            A[row + 1, ib] -= 1j * adm[i + 1]
            A[row + 1, ib + 1] += 1j * adm[i + 1]
    solution = np.linalg.solve(A, rhs)
    r, t = solution[0], solution[-1]
    zb = np.concatenate([[0.0], np.cumsum([d for d, _ in layers])])
    S = np.zeros(len(zs), complex)
    U = np.zeros(len(zs), complex)
    nz = np.zeros(len(zs), complex)
    for iz, z in enumerate(zs):
        if z < 0:
            f, b = np.exp(1j * kz[0] * z), r * np.exp(-1j * kz[0] * z)
            S[iz], U[iz], nz[iz] = f + b, 1j * adm[0] * (f - b), nI
        elif z > zb[-1]:
            s = t * np.exp(1j * kz[-1] * (z - zb[-1]))
            S[iz], U[iz], nz[iz] = s, 1j * adm[-1] * s, nII
        else:
            j = next(j for j in range(L) if z <= zb[j + 1])
            a, b = solution[1 + 2 * j], solution[2 + 2 * j]
            zl = z - zb[j]
            f, bk = a * np.exp(1j * kz[j + 1] * zl), b * np.exp(-1j * kz[j + 1] * zl)
            S[iz], U[iz], nz[iz] = f + bk, 1j * adm[j + 1] * (f - bk), layers[j][1]
    return r, t, S, U, nz


def uniform_stack_fields(spec, zs):
    """Independent six-component zero-order harmonics [6,nz] and far-field amplitudes (Rs, Rp, Ts, Tp)."""
    nI, nII = float(spec["stack"]["nI"]), complex_value(spec["stack"]["nII"])
    kt = nI * np.sin(np.radians(spec["theta_deg"]))
    phi = np.radians(spec["phi_deg"])
    s, c = np.sin(phi), np.cos(phi)
    es, ep = jones_vector(spec)
    r_s, t_s, S, U, nz = stratified(spec, zs, "TE")
    r_h, t_h, Sm, Um, _ = stratified(spec, zs, "TM")
    te = np.array([0 * S, S, 0 * S, 1j * U, 0 * S, -kt * S])
    tm = nI * np.array([-1j * Um, 0 * Sm, kt * Sm / nz ** 2, 0 * Sm, Sm, 0 * Sm])
    basis_s = np.array([-s * te[1], c * te[1], te[2], c * te[3], s * te[3], te[5]])
    basis_p = np.array([c * tm[0], s * tm[0], tm[2], -s * tm[4], c * tm[4], tm[5]])
    far = dict(Rs=es * r_s, Rp=ep * r_h, Ts=es * t_s, Tp=ep * t_h * nI / nII)
    return es * basis_s + ep * basis_p, far


def one_d_stack(spec, axis):
    """rcwa1d stack of a y-invariant (axis x) or x-invariant (axis y, period Ly) single-layer grating."""
    layer = spec["stack"]["layers"][0]
    if axis == "x":
        profile = layer["x_profiles"][0]
        edges, values, period = profile["edges"], profile["n_values"], spec["stack"]["Lx"]
    else:
        edges, period = layer["y_edges"], spec["stack"]["Ly"]
        values = [p["n_values"][0] for p in layer["x_profiles"]]
    return rc1.Stack(float(spec["stack"]["nI"]),
                     [rc1.Layer(layer["d"], edges=edges, n_values=[complex_value(v) for v in values])],
                     complex_value(spec["stack"]["nII"]), period)


# ---------------------------------------------------------------- (a) integrity and layout
class FrozenIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = json.loads((FROZEN_DIR / "reference_2d_v1.json").read_text(encoding="utf-8"))
        with np.load(FROZEN_DIR / "reference_2d_v1.npz", allow_pickle=False) as archive:
            cls.baseline = {key: archive[key] for key in archive.files}

    def test_file_hashes_sizes_and_provenance(self):
        meta = self.metadata
        archive = FROZEN_DIR / meta["archive"]["filename"]
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), meta["archive"]["sha256"])
        self.assertEqual(archive.stat().st_size, meta["archive"]["bytes"])
        self.assertEqual(meta["contract_version"], CONTRACT_VERSION)
        self.assertEqual(meta["components"], list(COMPONENTS))
        # source_sha256 records the frozen solver provenance (not enforced: the reproduction test is the guard)
        self.assertTrue(all(len(value) == 64 for value in meta["source_sha256"].values()))
        self.assertIn("gui/2drcwa/rcwa2d.py", meta["source_sha256"])
        for key in ("python", "numpy", "scipy"):
            self.assertIn(key, meta["versions"])
        table = INC
        self.assertTrue(table.is_file(), "tests/rcwa2d_reference.inc is missing")
        self.assertEqual(hashlib.sha256(table.read_bytes()).hexdigest(), meta["cpp_table"]["sha256"])
        self.assertEqual(table.stat().st_size, meta["cpp_table"]["bytes"])
        self.assertLessEqual(table.stat().st_size, 2 * 1024 * 1024)
        header = table.read_text(encoding="utf-8").split("struct Rcwa2dRefOrder", 1)[0]
        self.assertIn(meta["archive"]["sha256"], header)
        self.assertIn("generate_field_reference_2d.py", header)

    def test_recorded_inputs_equal_the_case_module(self):
        """The recorded inputs are authoritative.  The case module recomputes the Jones vectors from psi/delta with
        np.sin/np.cos/np.exp, whose last bit differs between libm implementations (found on arm64 + Accelerate: five
        Jones entries one ulp off), so floats are compared with a few-ulp tolerance and everything else exactly."""
        def same(a, b, path="parameters"):
            if isinstance(a, dict) and isinstance(b, dict):
                self.assertEqual(sorted(a), sorted(b), path)
                for key in a:
                    same(a[key], b[key], path + "." + key)
            elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                self.assertEqual(len(a), len(b), path)
                for index, (x, y) in enumerate(zip(a, b)):
                    same(x, y, "%s[%d]" % (path, index))
            elif isinstance(a, float) and isinstance(b, float):
                self.assertTrue(a == b or abs(a - b) <= 4e-16 * max(abs(a), abs(b)), "%s: %r vs %r" % (path, a, b))
            else:
                self.assertEqual(a, b, path)
        recorded, current = [record["parameters"] for record in self.metadata["cases"]], case_specs()
        self.assertEqual(len(recorded), len(current))
        for a, b in zip(recorded, current):
            same(a, b, a["id"])

    def test_harmonic_order_is_consistent_with_the_wavevectors(self):
        """Independent of the stored Kx, Ky and of the solver: Hz = Ky*Ex - Kx*Ey must hold column by column when
        Kx_m = nI sin(theta) cos(phi) - m lambda/Lx, Ky_n likewise, with (m, n) taken from the index convention
        index = (m + M1)(2 M2 + 1) + (n + M2) (m outer, n inner).  A harmonic array in any other order fails this for
        every case with M1 != 0 or M2 != 0, including the N = 625 case that the C++ tables do not contain."""
        for record in self.metadata["cases"]:
            spec = record["parameters"]
            if spec["status"] != "ok":
                continue
            with self.subTest(case=spec["id"]):
                a = {key: self.baseline[spec["id"] + "__" + key] for key in ("harmonics", "mn")}
                M1, M2, st = spec["M1"], spec["M2"], spec["stack"]
                m = np.repeat(np.arange(-M1, M1 + 1), 2 * M2 + 1)
                n = np.tile(np.arange(-M2, M2 + 1), 2 * M1 + 1)
                np.testing.assert_array_equal(a["mn"][:, 0], m)
                np.testing.assert_array_equal(a["mn"][:, 1], n)
                th, ph, lam = np.radians(spec["theta_deg"]), np.radians(spec["phi_deg"]), spec["wavelength"]
                Kx = st["nI"] * np.sin(th) * np.cos(ph) - m * lam / st["Lx"]
                Ky = st["nI"] * np.sin(th) * np.sin(ph) - n * lam / st["Ly"]
                h = a["harmonics"]                                  # [component, depth, order]
                Ex, Ey, Hz = h[0], h[1], h[5]
                scale = max(float(np.max(abs(h))), 1e-300)
                err = float(np.max(abs(Hz - (Ky * Ex - Kx * Ey)))) / scale
                self.assertLessEqual(err, 1e-12, "Hz = Ky Ex - Kx Ey relative error %.2e" % err)

    def test_array_manifest_dtype_shape_and_c_order(self):
        keys = []
        for record in self.metadata["cases"]:
            for item in record["arrays"].values():
                value = self.baseline[item["key"]]
                keys.append(item["key"])
                self.assertEqual(list(value.shape), item["shape"])
                self.assertEqual(str(value.dtype), item["dtype"])
                self.assertTrue(value.flags.c_contiguous)
        self.assertEqual(set(keys), set(self.baseline))

    def test_layout_of_every_array(self):
        for record in self.metadata["cases"]:
            spec = record["parameters"]
            if spec["status"] != "ok":
                self.assertEqual(record["arrays"], {})
                continue
            with self.subTest(case=spec["id"]):
                a = {key: self.baseline[spec["id"] + "__" + key] for key in record["arrays"]}
                N, nz, nt = spec["N"], len(spec["zs"]), spec["slice"]["nt"]
                M1, M2 = spec["M1"], spec["M2"]
                self.assertEqual(N, (2 * M1 + 1) * (2 * M2 + 1))
                self.assertEqual(a["mn"].shape, (N, 2))
                self.assertEqual(a["mn"].dtype, np.int64)
                index = (a["mn"][:, 0] + M1) * (2 * M2 + 1) + (a["mn"][:, 1] + M2)
                np.testing.assert_array_equal(index, np.arange(N))            # m outer, n inner = harmonic_index_2d
                self.assertEqual(int(a["i0"]), M1 * (2 * M2 + 1) + M2)
                np.testing.assert_array_equal(a["mn"][int(a["i0"])], [0, 0])
                for key in FAR_ARRAYS:
                    self.assertEqual(a[key].shape, (N,), key)
                self.assertEqual(a["inc"].shape, (2 * N,))
                for key in ("Kx", "Ky", "DEr", "DEt"):
                    self.assertEqual(a[key].dtype, np.float64)
                for key in ("kzI", "kzII", "Ex_r", "Ey_r", "Ez_r", "Ex_t", "Ey_t", "Ez_t", "Rs", "Rp", "Ts", "Tp"):
                    self.assertEqual(a[key].dtype, np.complex128)
                self.assertEqual(a["prop_r"].dtype, bool)
                self.assertEqual(a["harmonics"].shape, (6, nz, N))
                np.testing.assert_array_equal(a["z"], spec["zs"])
                self.assertTrue(np.all(np.diff(a["z"]) > 0))
                for name, period in (("xz", spec["stack"]["Lx"]), ("yz", spec["stack"]["Ly"])):
                    for key in ("fields", "phase_absolute", "phase_incident", "phase_mask", "phase_comparison_mask"):
                        self.assertEqual(a[f"{key}_{name}"].shape, (6, nz, nt), key + name)
                    self.assertEqual(a[f"E2_{name}"].shape, (nz, nt))
                    np.testing.assert_allclose(a[f"t_{name}"], np.arange(nt) * period / nt, rtol=0, atol=1e-15)
                    # x fastest in the sense of [component, z, t]: t is the last axis of every slice array
                    self.assertEqual(a[f"fields_{name}"].dtype, np.complex128)
                    self.assertEqual(a[f"phase_mask_{name}"].dtype, bool)
                np.testing.assert_array_equal(a["incident_amplitudes"], [1.0, spec["stack"]["nI"]])
                np.testing.assert_allclose(abs(sum(abs(complex(*v)) ** 2 for v in spec["jones"])), 1.0, atol=1e-15)
                for j, layer in enumerate(spec["stack"]["layers"]):
                    has = f"eps_L{j}" in a
                    self.assertEqual(has, spec["save_operators"] and "n" not in layer)
                    if has:
                        for name in ("eps", "cx", "cy"):
                            self.assertEqual(a[f"{name}_L{j}"].shape, (N, N))
                if spec["factorization"] == "laurent":
                    for j in range(len(spec["stack"]["layers"])):
                        if f"eps_L{j}" in a:
                            np.testing.assert_array_equal(a[f"cx_L{j}"], a[f"eps_L{j}"])
                            np.testing.assert_array_equal(a[f"cy_L{j}"], a[f"eps_L{j}"])
                # sample coordinates stay clear of every physical material edge (the Fourier sum is a Gibbs sum there)
                self.assertGreater(min(slice_edge_distances(spec).values()), 5e-3)
                # the harmonic depths avoid the +-1e-12 ownership band of the interfaces (and z == 0 for a bare interface)
                tops = np.concatenate([[0.0], np.cumsum([layer["d"] for layer in spec["stack"]["layers"]])])
                for z in spec["zs"]:
                    self.assertGreater(min(abs(z - tops)), 5e-7)
                # no NaN/inf in anything except the phases (NaN exactly where masked)
                for key, value in a.items():
                    if value.dtype.kind in "fc" and not key.startswith("phase_"):
                        self.assertTrue(np.isfinite(value).all(), key)
                for name in ("xz", "yz"):
                    for key in ("phase_absolute", "phase_incident"):
                        np.testing.assert_array_equal(np.isnan(a[f"{key}_{name}"]), a[f"phase_mask_{name}"])

    def test_structural_zero_claims_and_uniform_stacks(self):
        for record in self.metadata["cases"]:
            spec = record["parameters"]
            if spec["status"] != "ok":
                continue
            a = {key: self.baseline[spec["id"] + "__" + key] for key in record["arrays"]}
            near, far = spec["tolerance"]["near"], spec["tolerance"]["far"]
            with self.subTest(case=spec["id"]):
                for component in spec["structural_zero_components"]:
                    ic = COMPONENTS.index(component)
                    for key in ("harmonics", "fields_xz", "fields_yz"):
                        self.assertLessEqual(float(np.max(abs(a[key][ic]))), near["atol"], component + key)
                    self.assertTrue(a["phase_comparison_mask_xz"][ic].all())
                    self.assertTrue(a["phase_comparison_mask_yz"][ic].all())
                for key in spec["far_structural_zero_arrays"]:
                    self.assertLessEqual(float(np.max(abs(a[key]))), far["atol"], key)
                if spec["uniform_stack"]:
                    others = np.arange(spec["N"]) != int(a["i0"])
                    for key in FAR_ARRAYS:
                        if key not in ("Kx", "Ky", "kzI", "kzII"):
                            self.assertLessEqual(float(np.max(abs(a[key][others]))), 1e-14, key)
                    for name in ("harmonics",):
                        self.assertLessEqual(float(np.max(abs(a[name][:, :, others]))), 1e-14)
        # the documented zero patterns of the four 1-D-limit cases
        self.assertEqual(SPECS["y_inv_te"]["structural_zero_components"], ["Ex", "Ez", "Hy"])
        self.assertEqual(SPECS["y_inv_tm"]["structural_zero_components"], ["Ey", "Hx", "Hz"])
        self.assertEqual(SPECS["x_inv_te"]["structural_zero_components"], ["Ey", "Ez", "Hx"])
        self.assertEqual(SPECS["x_inv_tm"]["structural_zero_components"], ["Ex", "Hy", "Hz"])

    def test_phase_mask_rule_and_phase_values_from_the_stored_fields(self):
        """Absolute / incident-carrier phase and masks recomputed from the stored complex fields, with the carrier built from
        the *inputs* (k0 nI sin(theta) cos/sin(phi), k0 nI cos(theta)), not from solver output."""
        for record in self.metadata["cases"]:
            spec = record["parameters"]
            if spec["status"] != "ok":
                continue
            a = {key: self.baseline[spec["id"] + "__" + key] for key in record["arrays"]}
            nI, k0 = float(spec["stack"]["nI"]), 2 * np.pi / spec["wavelength"]
            th, ph = np.radians(spec["theta_deg"]), np.radians(spec["phi_deg"])
            kx0, ky0, kz0 = k0 * nI * np.sin(th) * np.cos(ph), k0 * nI * np.sin(th) * np.sin(ph), k0 * nI * np.cos(th)
            z = np.asarray(spec["zs"])[:, None]
            for name in ("xz", "yz"):
                t, at = a[f"t_{name}"][None, :], float(a[f"fixed_{name}"])
                x, y = (t, at) if name == "xz" else (at, t)
                carrier = np.exp(1j * (-kx0 * x - ky0 * y + kz0 * z))
                fields = a[f"fields_{name}"]
                np.testing.assert_allclose(a[f"E2_{name}"], sum(abs(fields[i]) ** 2 for i in range(3)), atol=1e-14, rtol=1e-14)
                for ic in range(6):
                    with self.subTest(case=spec["id"], plane=name, component=COMPONENTS[ic]):
                        field = fields[ic]
                        peak = float(np.max(abs(field)))
                        mask = abs(field) <= 1e-6 * peak
                        np.testing.assert_array_equal(a[f"phase_mask_{name}"][ic], mask)
                        valid = ~mask
                        np.testing.assert_allclose(np.exp(1j * a[f"phase_absolute_{name}"][ic][valid]),
                                                   np.exp(1j * np.angle(field[valid])), atol=1e-14, rtol=0)
                        np.testing.assert_allclose(np.exp(1j * a[f"phase_incident_{name}"][ic][valid]),
                                                   np.exp(1j * np.angle(field * np.conj(carrier))[valid]), atol=1e-14, rtol=0)

    def test_gui_phase_display_agrees_with_the_frozen_phases(self):
        """gui_fields.view_data (what the GUI shows for a 2-D run) uses its own carrier and mask code."""
        for case_id in ("pillar_sq", "pillar_ecc", "stack_sfs", "film_abs"):
            spec = SPECS[case_id]
            a = {key.split("__", 1)[1]: value for key, value in self.baseline.items() if key.startswith(case_id + "__")}
            for name in ("xz", "yz"):
                t, at = a[f"t_{name}"], float(a[f"fixed_{name}"])
                x, y = (t, np.full_like(t, at)) if name == "xz" else (np.full_like(t, at), t)
                fields = {key: a[f"fields_{name}"][i] for i, key in enumerate(COMPONENTS)}
                fields.update(E2=a[f"E2_{name}"], x=x, y=y, z=np.asarray(spec["zs"]))
                config = dict(mode="periodic2d", structure=dict(n_incident=float(spec["stack"]["nI"])),
                              incident=dict(polarization="jones", wavelength=spec["wavelength"],
                                            angle_deg=spec["theta_deg"], phi_deg=spec["phi_deg"]),
                              sampling=dict(slice_plane=name, slice_at=at))
                for ic, component in enumerate(COMPONENTS):
                    for quantity, key in (("phase", "phase_absolute"), ("phase_relative", "phase_incident")):
                        shown, _ = view_data(fields, config, component, quantity)
                        mask = a[f"phase_mask_{name}"][ic]
                        with self.subTest(case=case_id, plane=name, component=component, quantity=quantity):
                            np.testing.assert_array_equal(np.isnan(shown), mask)
                            valid = ~mask
                            if valid.any():
                                np.testing.assert_allclose(np.exp(1j * shown[valid]), np.exp(1j * a[f"{key}_{name}"][ic][valid]),
                                                           atol=2e-13, rtol=0)

    def test_failure_cases_record_the_expected_rayleigh_anomaly(self):
        for record in self.metadata["cases"]:
            spec = record["parameters"]
            if spec["status"] == "ok":
                continue
            with self.subTest(case=spec["id"]):
                expect = spec["expect"]
                info = record["python_failure"]
                self.assertEqual(info["error"], "RayleighAnomalyError")
                self.assertTrue(info["message"].startswith(expect["region"]))
                found = [[int(m), int(n)] for m, n in re.findall(r"\((-?\d+), (-?\d+)\)", info["message"])]
                self.assertEqual(found, expect["orders"])
                self.assertEqual(failure_info(spec), info)          # the current solver fails the same way


# ---------------------------------------------------------------- (b) same-environment reproduction
class Reproduction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = json.loads((FROZEN_DIR / "reference_2d_v1.json").read_text(encoding="utf-8"))
        with np.load(FROZEN_DIR / "reference_2d_v1.npz", allow_pickle=False) as archive:
            cls.baseline = {key: archive[key] for key in archive.files}

    def test_current_solver_reproduces_every_frozen_array(self):
        for record in self.metadata["cases"]:
            spec = record["parameters"]
            if spec["status"] != "ok" or not wanted(spec):
                continue
            with self.subTest(case=spec["id"]):
                _, actual, diagnostics = sampled(spec["id"])
                tol = spec["tolerance"]["reproduce"]
                self.assertEqual(set(diagnostics), set(record["diagnostics"]))
                for key, value in diagnostics.items():
                    if isinstance(value, int):
                        self.assertEqual(value, record["diagnostics"][key], key)
                    else:
                        self.assertAlmostEqual(value, record["diagnostics"][key], delta=tol["atol"], msg=key)
                reference = {key: self.baseline[item["key"]] for key, item in record["arrays"].items()}
                self.assertEqual(set(actual), set(reference))
                for key, value in actual.items():
                    ref = reference[key]
                    self.assertEqual(value.shape, ref.shape, key)
                    if key.startswith("phase_") and "mask" not in key:
                        continue                                                  # NaN-bearing phases: below
                    if value.dtype.kind in "iub":
                        np.testing.assert_array_equal(value, ref, err_msg=key)
                    else:
                        note_max("reproduce", np.max(np.abs(value - ref)) if value.size else 0.0)
                        np.testing.assert_allclose(value, ref, **tol, err_msg=spec["id"] + "/" + key)
                near = spec["tolerance"]["near"]
                for name in ("xz", "yz"):
                    amplitude = abs(reference[f"fields_{name}"])
                    error = near["atol"] + near["rtol"] * amplitude
                    expected_mask = reference[f"phase_mask_{name}"] | (amplitude <= 10 * error)
                    for component in spec["structural_zero_components"]:
                        expected_mask[COMPONENTS.index(component)] = True
                    np.testing.assert_array_equal(reference[f"phase_comparison_mask_{name}"], expected_mask)
                    valid = ~expected_mask
                    self.assertTrue(valid.any())
                    self.assertFalse(actual[f"phase_mask_{name}"][valid].any())
                    for key in ("phase_absolute", "phase_incident"):
                        difference = abs(np.exp(1j * actual[f"{key}_{name}"][valid]) - np.exp(1j * reference[f"{key}_{name}"][valid]))
                        self.assertTrue(np.all(difference <= 2 * error[valid] / amplitude[valid] + 5e-14), key + name)


# ---------------------------------------------------------------- (c) independent checks
class IndependentChecks(unittest.TestCase):
    def test_uniform_stacks_against_airy_six_component_fields(self):
        """interface / film / absorbing film on an absorbing substrate: far amplitudes and all six field components."""
        for case_id in ("interface", "film", "film_abs"):
            spec = SPECS[case_id]
            with self.subTest(case=case_id):
                res, arrays, _ = sampled(case_id)
                zs = np.asarray(spec["zs"])
                expected, far = uniform_stack_fields(spec, zs)
                i0 = int(arrays["i0"])
                for key, value in far.items():
                    note_max("airy_far", abs(arrays[key][i0] - value))
                    self.assertAlmostEqual(arrays[key][i0], value, delta=3e-15)
                harmonics = arrays["harmonics"]
                note_max("airy_harmonics", np.max(np.abs(harmonics[:, :, i0] - expected)))
                np.testing.assert_allclose(harmonics[:, :, i0], expected, atol=4e-15, rtol=4e-15)
                others = np.arange(spec["N"]) != i0
                np.testing.assert_array_equal(harmonics[:, :, others], 0.0)
                nI = float(spec["stack"]["nI"])
                th, ph = np.radians(spec["theta_deg"]), np.radians(spec["phi_deg"])
                kx0, ky0 = 2 * np.pi / spec["wavelength"] * nI * np.sin(th) * np.array([np.cos(ph), np.sin(ph)])
                for name, axis in (("xz", "x"), ("yz", "y")):
                    t, at = arrays[f"t_{name}"][None, :], float(arrays[f"fixed_{name}"])
                    x, y = (t, at) if name == "xz" else (at, t)
                    carrier = np.exp(-1j * (kx0 * x + ky0 * y))
                    expected_slice = expected[:, :, None] * carrier[None]
                    note_max("airy_slices", np.max(np.abs(arrays[f"fields_{name}"] - expected_slice)))
                    np.testing.assert_allclose(arrays[f"fields_{name}"], expected_slice, atol=4e-15, rtol=4e-15)
                # energy in the lossless / absorbing cases
                total = float(arrays["DEr"].sum() + arrays["DEt"].sum())
                if case_id != "film_abs":
                    self.assertAlmostEqual(total, 1.0, delta=1e-13)
                else:
                    self.assertTrue(0 < total < 1)

    def test_y_and_x_invariant_gratings_against_rcwa1d_and_conical(self):
        """Complex far-field amplitudes, efficiencies and the six harmonic components of the six 1-D-limit cases."""
        for case_id, axis in (("y_inv_te", "x"), ("y_inv_tm", "x"), ("y_inv_conical", "x"),
                              ("x_inv_te", "y"), ("x_inv_tm", "y"), ("x_inv_conical", "y")):
            spec = SPECS[case_id]
            with self.subTest(case=case_id):
                res, arrays, _ = sampled(case_id)
                jones = np.asarray(jones_vector(spec))
                M = spec["M1"] if axis == "x" else spec["M2"]
                phi1 = spec["phi_deg"] if axis == "x" else spec["phi_deg"] - 90.0   # x_1D = y, y_1D = -x
                stack = one_d_stack(spec, axis)
                conical = solve_conical(stack, spec["wavelength"], spec["theta_deg"], phi1, M=M, jones_in=jones, need_fields=True)
                far = max(float(np.max(np.abs(arrays["Rs"] - conical["Jr"][:, 0, :] @ jones))),
                          float(np.max(np.abs(arrays["Rp"] - conical["Jr"][:, 1, :] @ jones))),
                          float(np.max(np.abs(arrays["Ts"] - conical["Jt"][:, 0, :] @ jones))),
                          float(np.max(np.abs(arrays["Tp"] - conical["Jt"][:, 1, :] @ jones))))
                note_max("oned_conical_far", far)
                self.assertLess(far, 1e-12)
                self.assertLess(float(np.max(np.abs(arrays["DEr"] - conical["DEr"]))), 1e-12)
                self.assertLess(float(np.max(np.abs(arrays["DEt"] - conical["DEt"]))), 1e-12)
                worst = 0.0
                for iz, z in enumerate(spec["zs"]):
                    f = conical_at(conical, float(z), jones_in=jones)
                    if axis == "x":
                        mapped = [f["Ex"], f["Ey"], f["Ez"], f["Hx"], f["Hy"], f["Hz"]]
                    else:     # rotate the 1-D frame back: V_x = -V_1y, V_y = V_1x
                        mapped = [-f["Ey"], f["Ex"], f["Ez"], -f["Hy"], f["Hx"], f["Hz"]]
                    for ic in range(6):
                        worst = max(worst, float(np.max(np.abs(arrays["harmonics"][ic, iz] - mapped[ic]))))
                note_max("oned_conical_harmonics", worst)
                self.assertLess(worst, 1e-11)
                if case_id.endswith(("_te", "_tm")):          # also the scalar rcwa1d.solve (T1 / T1')
                    pol = "TE" if case_id.endswith("_te") else "TM"
                    scalar = rc1.solve(stack, spec["wavelength"], spec["theta_deg"], M, pol, need_fields=False)
                    note_max("oned_scalar_efficiency", max(np.max(np.abs(arrays["DEr"] - scalar["DEr"])),
                                                         np.max(np.abs(arrays["DEt"] - scalar["DEt"]))))
                    np.testing.assert_allclose(arrays["DEr"], scalar["DEr"], atol=1e-11, rtol=0)
                    np.testing.assert_allclose(arrays["DEt"], scalar["DEt"], atol=1e-11, rtol=0)

    def test_energy_conservation_and_polarisation_decomposition(self):
        """sum(DE) = 1 without absorption (a necessary condition only); |E|^2 = |Rs|^2 + |Rp|^2 on propagating orders."""
        for spec in OK_SPECS:
            a = None
            if wanted(spec):
                a = sampled(spec["id"])[1]
            else:
                with np.load(FROZEN_DIR / "reference_2d_v1.npz", allow_pickle=False) as archive:
                    a = {key.split("__", 1)[1]: archive[key] for key in archive.files if key.startswith(spec["id"] + "__")}
            with self.subTest(case=spec["id"]):
                total = float(a["DEr"].sum() + a["DEt"].sum())
                lossless = all(np.imag(complex_value(layer["n"])) == 0 for layer in spec["stack"]["layers"] if "n" in layer) and \
                    all(np.imag(complex_value(v)) == 0 for layer in spec["stack"]["layers"] if "n" not in layer
                        for p in layer["x_profiles"] for v in p["n_values"]) and np.imag(complex_value(spec["stack"]["nII"])) == 0
                note_max("energy" if lossless else "energy_absorbing_margin", abs(total - 1.0) if lossless else 0.0)
                if lossless:
                    self.assertLess(abs(total - 1.0), 1e-9)
                else:
                    self.assertTrue(0.0 < total < 1.0, total)
                if np.imag(complex_value(spec["stack"]["nII"])) == 0:
                    nI, th = float(spec["stack"]["nI"]), np.radians(spec["theta_deg"])
                    for de, kz, rs, rp, prop in ((a["DEr"], a["kzI"], a["Rs"], a["Rp"], a["prop_r"]),
                                                 (a["DEt"], a["kzII"], a["Ts"], a["Tp"], a["prop_t"])):
                        expect = np.where(prop, kz.real * (abs(rs) ** 2 + abs(rp) ** 2) / (nI * np.cos(th)), 0.0)
                        np.testing.assert_allclose(de, expect, atol=1e-12, rtol=1e-10)

    def test_tangential_fields_are_continuous_across_every_layer_interface(self):
        """selftest2d T5 on each case: Ex, Ey, Hx, Hy, Hz harmonic columns at z_i -+ 1e-10 (N <= 169)."""
        import rcwa2d as rc2
        for spec in OK_SPECS:
            if spec["N"] > 169 or not spec["stack"]["layers"]:
                continue
            with self.subTest(case=spec["id"]):
                res = sampled(spec["id"])[0]
                z, interfaces = 0.0, [0.0]
                for layer in spec["stack"]["layers"]:
                    z += layer["d"]
                    interfaces.append(z)
                for zi in interfaces:
                    up, down = rc2.field_at(res, zi - 1e-10), rc2.field_at(res, zi + 1e-10)
                    for key in ("Ex", "Ey", "Hx", "Hy", "Hz"):
                        scale = max(float(np.max(np.abs(up[key]))), 1e-30)
                        error = float(np.max(np.abs(up[key] - down[key]))) / scale
                        note_max("tangential_continuity", error)
                        self.assertLess(error, 1e-7, f"{key} at z={zi}")
        # N = 625: continuity of the frozen +-1e-6 columns, relative to the largest harmonic (a coarser, derivative-limited check)
        spec = SPECS["pillar_big"]
        with np.load(FROZEN_DIR / "reference_2d_v1.npz", allow_pickle=False) as archive:
            h, z = archive["pillar_big__harmonics"], archive["pillar_big__z"]
        for lo, hi in ((1, 2), (5, 6)):
            self.assertAlmostEqual(z[hi] - z[lo], 2e-6, delta=1e-12)
            for ic in (0, 1, 3, 4, 5):
                scale = float(np.max(np.abs(h[ic, lo])))
                self.assertLess(float(np.max(np.abs(h[ic, hi] - h[ic, lo]))) / scale, 5e-3)

    def test_c4_pillar_is_symmetric_under_x_y_exchange(self):
        """Diagonal mirror (x,y) -> (y,x): the s (Ey) response at order (m,n) equals the p (Ex) response at (n,m) with Ex<->Ey."""
        spec = SPECS["c4_normal"]
        res_s = sampled("c4_normal")[0]
        res_p = solve_case(dict(spec, jones=[[0.0, 0.0], [1.0, 0.0]]))
        mn = res_s["mn"]
        swap = np.array([int(np.where((mn[:, 0] == n) & (mn[:, 1] == m))[0][0]) for m, n in mn])
        worst = 0.0
        for a, b in (("Ex_t", "Ey_t"), ("Ey_t", "Ex_t"), ("Ex_r", "Ey_r"), ("Ey_r", "Ex_r"), ("Ez_t", "Ez_t"), ("Ez_r", "Ez_r")):
            worst = max(worst, float(np.max(np.abs(res_s[a] - res_p[b][swap]))))
        note_max("c4_symmetry", worst)
        self.assertLess(worst, 1e-11)
        np.testing.assert_allclose(res_s["DEt"], res_p["DEt"][swap], atol=1e-12, rtol=0)

    def test_laurent_operators_and_li_difference(self):
        """Laurent: Cx = Cy = Eps exactly (frozen); Li differs from Laurent on a 2-D pillar (so the switch is exercised)."""
        a, b = sampled("laurent_pillar")[1], sampled("ellip_pillar")[1]
        np.testing.assert_array_equal(a["cx_L0"], a["eps_L0"])
        np.testing.assert_array_equal(a["cy_L0"], a["eps_L0"])
        self.assertGreater(float(np.max(np.abs(b["cx_L0"] - b["eps_L0"]))), 1e-3)
        self.assertGreater(float(np.max(np.abs(b["cy_L0"] - b["eps_L0"]))), 1e-3)
        np.testing.assert_allclose(a["eps_L0"], b["eps_L0"], atol=1e-14)    # same pillar and same M: Eps does not depend on the rule

    def test_operators_are_independent_fourier_coefficients(self):
        """Eps of a pillar from the closed-form 2-D Fourier integral of the rectangle (independent of rcwa2d's Toeplitz code)."""
        spec = SPECS["ellip_pillar"]
        eps = sampled("ellip_pillar")[1]["eps_L0"]
        mn = sampled("ellip_pillar")[1]["mn"]
        layer = spec["stack"]["layers"][0]
        Lx, Ly = spec["stack"]["Lx"], spec["stack"]["Ly"]
        x0, x1 = layer["x_profiles"][1]["edges"][1:3]
        y0, y1 = layer["y_edges"][1:3]
        n_in, n_bg = complex_value(layer["x_profiles"][1]["n_values"][1]), complex_value(layer["x_profiles"][1]["n_values"][0])

        def coefficient(h, edge0, edge1, period):
            if h == 0:
                return (edge1 - edge0) / period
            return (np.exp(-2j * np.pi * h * edge0 / period) - np.exp(-2j * np.pi * h * edge1 / period)) / (2j * np.pi * h)

        # eps(x,y) = bg^2 + (in^2 - bg^2) * 1_rect;  c(h, k) = bg^2 delta + (in^2 - bg^2) * cx(h) * cy(k)
        N = len(mn)
        expected = np.zeros((N, N), complex)
        for i in range(N):
            for j in range(N):
                dm, dn = mn[i, 0] - mn[j, 0], mn[i, 1] - mn[j, 1]
                expected[i, j] = (n_in ** 2 - n_bg ** 2) * coefficient(dm, x0, x1, Lx) * coefficient(dn, y0, y1, Ly) + \
                    (n_bg ** 2 if (dm == 0 and dn == 0) else 0.0)
        note_max("eps_fourier", np.max(np.abs(eps - expected)))
        np.testing.assert_allclose(eps, expected, atol=1e-13, rtol=0)

    def test_cpp_table_matches_the_frozen_archive(self):
        """tests/rcwa2d_reference.inc parsed back and compared with the NPZ (no stale or hand-edited table)."""
        text = INC.read_text(encoding="utf-8")
        number = re.compile(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?|-?\d+e[-+]?\d+")

        def block(name):
            match = re.search(r"static const \w+ " + name + r"\[\]\s*=\s*\{(.*?)\n\};", text, re.S)
            self.assertIsNotNone(match, name)
            body = re.sub(r"//[^\n]*", "", match.group(1))
            return [float(v) for v in number.findall(body)]

        with np.load(FROZEN_DIR / "reference_2d_v1.npz", allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        for cid, spec in enumerate(case_specs(), start=1):
            if spec["status"] != "ok" or spec["N"] > 169:
                if spec["status"] != "ok":
                    self.assertNotRegex(text, rf"kRcwa2dRef{cid}\[\] = ")
                continue
            with self.subTest(case=spec["id"]):
                a = {key.split("__", 1)[1]: value for key, value in data.items() if key.startswith(spec["id"] + "__")}
                table = np.asarray(block(f"kRcwa2dRef{cid}")).reshape(spec["N"], -1)
                N = spec["N"]
                self.assertEqual(table.shape[1], 4 + 24 + 4)
                np.testing.assert_array_equal(table[:, 0], a["mn"][:, 0])
                np.testing.assert_array_equal(table[:, 1], a["mn"][:, 1])
                np.testing.assert_array_equal(table[:, 2], a["Kx"])
                np.testing.assert_array_equal(table[:, 3], a["Ky"])
                column = 4
                for key in ("kzI", "kzII", "Ex_r", "Ey_r", "Ez_r", "Ex_t", "Ey_t", "Ez_t", "Rs", "Rp", "Ts", "Tp"):
                    np.testing.assert_array_equal(table[:, column], a[key].real, err_msg=key)
                    np.testing.assert_array_equal(table[:, column + 1], a[key].imag, err_msg=key)
                    column += 2
                np.testing.assert_array_equal(table[:, column], a["DEr"])
                np.testing.assert_array_equal(table[:, column + 1], a["DEt"])
                np.testing.assert_array_equal(table[:, column + 2], a["prop_r"])
                np.testing.assert_array_equal(table[:, column + 3], a["prop_t"])
                columns = 0
                while re.search(rf"kRcwa2dRef{cid}Z{columns}\[\]", text):
                    columns += 1
                self.assertGreaterEqual(columns, 2)
                for k in range(columns):
                    values = np.asarray(block(f"kRcwa2dRef{cid}Z{k}")).reshape(N, 6, 2)
                    z = float(re.search(rf"kRcwa2dRef{cid}Z{k}\[\] = \{{\s*// z = (\S+)", text).group(1))
                    iz = int(np.argmin(abs(a["z"] - z)))
                    self.assertEqual(a["z"][iz], z)
                    np.testing.assert_array_equal(values[:, :, 0].T, a["harmonics"][:, iz, :].real)
                    np.testing.assert_array_equal(values[:, :, 1].T, a["harmonics"][:, iz, :].imag)
                for j, layer in enumerate(spec["stack"]["layers"]):
                    if f"kRcwa2dRef{cid}EpsL{j}[]" in text:
                        for name in ("Eps", "Cx", "Cy"):
                            flat = np.asarray(block(f"kRcwa2dRef{cid}{name}L{j}")).reshape(N, N, 2)
                            ref = a[f"{name.lower()}_L{j}"]
                            np.testing.assert_array_equal(flat[:, :, 0], ref.real)
                            np.testing.assert_array_equal(flat[:, :, 1], ref.imag)
        # every structure in the table builds from the case inputs
        for cid, spec in enumerate(case_specs(), start=1):
            if spec["status"] == "ok" and spec["N"] > 169:
                continue
            self.assertIn(f"    case {cid}: {{   // {spec['id']}", text)
        for spec in case_specs():
            if spec["status"] != "ok":
                self.assertIn(spec["expect"]["region"], text)


def tearDownModule():
    if _MAX:
        print("\nmax errors seen in this run:")
        for key in sorted(_MAX):
            print(f"  {key:28s} {_MAX[key]:.3e}")


if __name__ == "__main__":
    unittest.main()
