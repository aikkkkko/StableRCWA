"""Production C++ field_at/field_map/phase_data versus frozen and independent oracles.

Set RCWA_PUBLIC_FIELDS_EXE to test_rcwa_public_fields. This test only reads the
frozen corpus and never calls its generator or the test-only context sampler.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

GUI = Path(__file__).resolve().parents[1]
if str(GUI) not in sys.path:
    sys.path.insert(0, str(GUI))

from field_phase import phase_data
from validation.field_reference_cases import COMPONENTS, solve_case
from validation.test_degenerate_fields import direct_field
from validation.test_field_contract import scalar_analytic

REFERENCE = Path(__file__).resolve().parent / "field_reference"
EXE = os.environ.get("RCWA_PUBLIC_FIELDS_EXE")


def complex_array(pairs):
    values = np.asarray(pairs, dtype=np.float64)
    if not values.size:
        return np.empty(0, dtype=complex)
    return values[..., 0] + 1j * values[..., 1]


def encode_complex(value):
    value = np.asarray(value, dtype=complex)
    pairs = np.stack((value.real, value.imag), axis=-1).astype(object)
    pairs[~np.isfinite(np.asarray(pairs, dtype=float))] = None
    return pairs.tolist()


@unittest.skipUnless(EXE, "set RCWA_PUBLIC_FIELDS_EXE to the C++ public field test executable")
class CppPublicFieldReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((REFERENCE / "reference_v1.json").read_text(encoding="utf-8"))
        cls.archive_path = REFERENCE / cls.manifest["archive"]["filename"]
        with np.load(cls.archive_path, allow_pickle=False) as archive:
            cls.baseline = {key: archive[key] for key in archive.files}
        reply = subprocess.run([EXE, "--reference", str(REFERENCE / "reference_v1.json")],
                               check=True, capture_output=True, text=True, timeout=90)
        cls.actual = json.loads(reply.stdout)

    def test_all_twelve_public_harmonics_maps_metadata_and_phase(self):
        self.assertEqual(hashlib.sha256(self.archive_path.read_bytes()).hexdigest(), self.manifest["archive"]["sha256"])
        self.assertEqual([case["id"] for case in self.actual],
                         [case["parameters"]["id"] for case in self.manifest["cases"]])
        for actual, record in zip(self.actual, self.manifest["cases"]):
            spec = record["parameters"]
            case = spec["id"]
            with self.subTest(case=case):
                self.assertEqual(actual["n_zero_modes"], spec["expected_zero_modes"])
                self.assertEqual(actual["n_schur"], spec["expected_schur_layers"])
                for key in ("harmonics", "fields"):
                    values = complex_array(actual[key])
                    self.assertTrue(np.isfinite(values).all())
                    np.testing.assert_allclose(values, self.baseline[case + "__" + key], **spec["tolerance"])
                for source, key in (("kxm", "kx"), ("kzI", "kzI"), ("kzII", "kzII")):
                    np.testing.assert_allclose(complex_array(actual[source]), self.baseline[case + "__" + key], **spec["tolerance"])
                for key in ("x", "z", "y", "ky", "incident_amplitudes"):
                    np.testing.assert_allclose(actual[key], self.baseline[case + "__" + key], atol=2e-15, rtol=2e-15)
                expected = self.baseline[case + "__fields"]
                valid = ~self.baseline[case + "__phase_comparison_mask"]
                error = spec["tolerance"]["atol"] + spec["tolerance"]["rtol"] * abs(expected)
                actual_mask = np.asarray(actual["phase_mask"], dtype=bool)
                self.assertFalse(actual_mask[valid].any())
                for key in ("phase_absolute", "phase_incident"):
                    phase = np.asarray(actual[key], dtype=float)
                    self.assertTrue(np.isfinite(phase[valid]).all())
                    diff = abs(np.exp(1j * phase[valid]) - np.exp(1j * self.baseline[case + "__" + key][valid]))
                    self.assertTrue(np.all(diff <= 2 * error[valid] / abs(expected[valid]) + 5e-14), key)
                for component in spec["structural_zero_components"]:
                    ic = COMPONENTS.index(component)
                    self.assertLessEqual(float(abs(complex_array(actual["fields"])[ic]).max()), spec["tolerance"]["atol"])

    def test_public_six_fields_against_independent_interface_and_film(self):
        for actual, record in zip(self.actual, self.manifest["cases"]):
            spec = record["parameters"]
            if not spec["id"].startswith(("interface_", "absorbing_film_")):
                continue
            with self.subTest(case=spec["id"]):
                expected = scalar_analytic(spec, spec["zs"])
                kt = spec["stack"]["nI"] * np.sin(np.radians(spec["theta_deg"]))
                carrier = np.exp(-1j * (2 * np.pi / spec["wavelength"]) * kt * np.asarray(actual["x"]))
                np.testing.assert_allclose(complex_array(actual["fields"]), expected[:, :, None] * carrier, atol=4e-14, rtol=4e-14)
                np.testing.assert_allclose(complex_array(actual["harmonics"])[:, :, spec["M"]], expected, atol=4e-14, rtol=4e-14)

    def test_public_schur_six_harmonics_against_four_n_first_order(self):
        index = next(i for i, case in enumerate(self.actual) if case["id"] == "schur_ellipse")
        actual = self.actual[index]
        spec = self.manifest["cases"][index]["parameters"]
        result = solve_case(spec)
        vin = complex_array(spec["jones_in"])
        vin /= np.linalg.norm(vin)
        eps, _ = result.stack.layers[0].operators(result.stack.period, result.M)
        n = len(result.ms)
        cs, sn, kx = result.cos_phi_m, result.sin_phi_m, result.kx_over_k0
        got = complex_array(actual["harmonics"])
        for z in (0., .19, .38):
            with self.subTest(z=z):
                f, g = direct_field(result, z, vin)
                ex, ey = -sn * f[:n] + cs * f[n:], cs * f[:n] + sn * f[n:]
                hx, hy = -cs * g[:n] - sn * g[n:], -sn * g[:n] + cs * g[n:]
                ez = 1j * np.linalg.solve(eps, result.kappa * hx - kx * hy)
                expected = np.array([ex, ey, ez, -1j * hx, -1j * hy, result.kappa * ex - kx * ey])
                np.testing.assert_allclose(got[:, spec["zs"].index(z)], expected, atol=3e-10, rtol=3e-10)


@unittest.skipUnless(EXE, "set RCWA_PUBLIC_FIELDS_EXE to the C++ public field test executable")
class CppPhaseReference(unittest.TestCase):
    def compare(self, fields, *, component="Ey", reference="absolute", projection=None, floor=1e-6, unwrap=None):
        nx, nz = len(fields["x"]), len(fields["z"])
        wavevector = (2.3 + .8j, .7, 3.1 + .5j)
        result = dict(i0=0, kxm=np.array([wavevector[0]]), ky=wavevector[1], kzI=np.array([wavevector[2]]))
        request = dict(x=np.asarray(fields["x"]).tolist(), z=np.asarray(fields["z"]).tolist(), y=fields.get("y", .11),
                       fields={c: encode_complex(fields[c]) for c in COMPONENTS if c in fields}, component=component,
                       wavevector=encode_complex(wavevector), phase_floor=floor, unwrap=unwrap or "none")
        if isinstance(reference, str):
            request["reference"] = reference
        else:
            request["reference"] = "custom"
            request["custom_reference"] = encode_complex(np.atleast_2d(reference))
        if projection is not None:
            request["projection"] = encode_complex(projection)
        reply = subprocess.run([EXE, "--phase-case"], input=json.dumps(request), check=True,
                               capture_output=True, text=True, timeout=30)
        actual = json.loads(reply.stdout)
        expected = phase_data(result, fields, component, reference=reference, projection=projection,
                              phase_floor=floor, unwrap=unwrap)
        actual_phase = np.asarray(actual["phase"], dtype=float).reshape(nz, nx)
        actual_mask = np.asarray(actual["mask"], dtype=bool).reshape(nz, nx)
        np.testing.assert_array_equal(actual_mask, expected["mask"])
        np.testing.assert_allclose(actual_phase[~actual_mask], expected["phase"].data[~actual_mask], atol=8e-14, rtol=3e-15)
        self.assertTrue(np.isnan(actual_phase[actual_mask]).all())
        np.testing.assert_allclose(np.asarray(actual["amplitude"], dtype=float).reshape(nz, nx), expected["amplitude"], atol=2e-14, rtol=3e-15)
        np.testing.assert_allclose(complex_array(actual["field"]).reshape(nz, nx), expected["field"], atol=2e-14, rtol=3e-15)
        if expected["reference"] is not None:
            np.testing.assert_allclose(complex_array(actual["reference"]).reshape(nz, nx), expected["reference"], atol=2e-14, rtol=3e-15)
        self.assertEqual(actual["reference_name"], expected["reference_name"])
        self.assertEqual(actual["unwrapped"], expected["unwrapped"])

    def test_carrier_absolute_incident_and_global_phase(self):
        x, z, y = np.array([0., .13, .31, .7]), np.array([-.1, .2, .2, .7]), .11
        carrier = np.exp(1j * (-2.3 * x[None, :] - .7 * y + 3.1 * z[:, None]))
        fields = dict(x=x, z=z, y=y, Ey=2 * carrier * np.exp(.37j))
        for ref in ("absolute", "incident"):
            with self.subTest(reference=ref):
                self.compare(fields, reference=ref)

    def test_synthetic_nodes_threshold_equality_zero_and_nonfinite(self):
        values = np.array([[1., .25, .250001, 0.], [np.nan, 1j, -.5j, -.2]], complex)
        fields = dict(x=np.arange(4), z=np.arange(2), y=0., Ey=values)
        for floor in (0., .25, 1e-6, 2.):
            with self.subTest(floor=floor):
                self.compare(fields, floor=floor)
        fields["Ey"] = np.zeros((2, 4), complex)
        self.compare(fields, floor=0.)

    def test_custom_reference_scalar_row_column_and_grid_broadcast(self):
        rng = np.random.default_rng(12)
        fields = dict(x=np.arange(7) * .13, z=np.arange(5) * .17, y=.11,
                      Ey=rng.normal(size=(5, 7)) + 1j * rng.normal(size=(5, 7)))
        row = np.array([1 + .3j, 0., 2j, 3., np.nan, -.2, .7])
        column = np.array([1., 0., np.nan, 2j, -.5])[:, None]
        full = np.exp(1j * rng.uniform(-np.pi, np.pi, (5, 7)))
        full[1, 3] = 0.
        for reference in (2 + 3j, row, column, full):
            with self.subTest(shape=np.shape(reference)):
                self.compare(fields, reference=reference)

    def test_complex_projection_has_conjugation_and_original_scale(self):
        rng = np.random.default_rng(87)
        fields = dict(x=np.arange(4) * .1, z=np.arange(3) * .3, y=.11)
        fields.update({c: rng.normal(size=(3, 4)) + 1j * rng.normal(size=(3, 4)) for c in COMPONENTS})
        projection = np.array([2 + 3j, -.7 + .2j, 1.4j])
        for family in ("E", "H"):
            for reference in ("absolute", "incident"):
                with self.subTest(family=family, reference=reference):
                    self.compare(fields, component=family, reference=reference, projection=projection)

    def test_segmented_x_z_both_unwrap_and_pi_ties(self):
        rng = np.random.default_rng(92)
        x, z = np.arange(9) * .3, np.arange(6) * .4
        angles = 4.3 * x[None, :] + 4.6 * z[:, None] + rng.uniform(-.1, .1, (6, 9))
        values = np.exp(1j * angles)
        values[1, 3] = values[2, 7] = values[4, 4] = 0.
        fields = dict(x=x, z=z, y=0., Ey=values)
        for unwrap in (None, "x", "z", "both"):
            with self.subTest(unwrap=unwrap):
                self.compare(fields, unwrap=unwrap)
        fields = dict(x=np.arange(5), z=np.arange(2), Ey=np.array([[1, -1+0j, 1, -1-0j, 1],
                                                               [-1+0j, 1, -1+0j, 1, -1-0j]], complex))
        self.compare(fields, unwrap="both")

    def test_empty_depth_has_defined_empty_phase_layout(self):
        fields = dict(x=np.arange(4), z=np.empty(0), Ey=np.zeros((0, 4), complex))
        self.compare(fields)
        self.compare(fields, reference="incident", unwrap="both")


if __name__ == "__main__":
    unittest.main()
