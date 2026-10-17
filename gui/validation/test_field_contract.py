"""Frozen complex-field regression plus independent analytic/phase contracts.

Never regenerates the baseline. Degenerate propagation physics remains in
test_degenerate_fields, including its independent first-order scatterers.
"""
import hashlib
import json
from pathlib import Path
import sys
import unittest

GUI = Path(__file__).resolve().parents[1]
if str(GUI) not in sys.path:
    sys.path.insert(0, str(GUI))

import numpy as np
import rcwa1d as rc
from conical.rcwa_conical import solve_conical
from conical.fields_conical import field_at as conical_at, field_map as conical_map
from field_phase import complex_component, incident_reference, phase_data
from gui_fields import complete_six_components, normalization, view_data
from validation.field_reference_cases import (COMPONENTS, CONTRACT_VERSION, FROZEN_DIR,
                                              build_stack, case_specs, gui_config, sample_case, solve_case)


def scalar_analytic(spec, zs):
    """One interface or one film, from independent scalar boundary equations.

    Film a/b are both referenced to its top; no production mode or scattering
    code is used. Return all six plane-incidence harmonics of the zero order.
    """
    st = build_stack(spec["stack"])
    nI, nO = st.nI, st.nII
    k0 = 2*np.pi/spec["wavelength"]
    kt = nI*np.sin(np.radians(spec["theta_deg"]))
    pol = spec["polarization"]
    kz = lambda n: k0*np.sqrt(complex(n*n-kt*kt))
    admittance = lambda n: kz(n)/k0/(1 if pol == "TE" else n*n)
    pi, po = admittance(nI), admittance(nO)
    if not st.layers:
        reflection, transmission = (pi-po)/(pi+po), 2*pi/(pi+po)
        d = 0.
    else:
        if len(st.layers) != 1 or not st.layers[0].homogeneous:
            raise ValueError("Analytic oracle supports one uniform film only")
        layer = st.layers[0]
        d, nl = layer.d, layer.n
        pl = admittance(nl)
        down, up = np.exp(1j*kz(nl)*d), np.exp(-1j*kz(nl)*d)
        system = np.array([[1., -1., -1., 0.], [-pi, -pl, pl, 0.],
                           [0., down, up, -1.], [0., pl*down, -pl*up, -po]], complex)
        reflection, a, b, transmission = np.linalg.solve(system, [-1., -pi, 0., 0.])
    harmonics = np.zeros((6, len(zs)), complex)
    for iz, z in enumerate(zs):
        if z < 0:
            n = nI
            forward, backward = np.exp(1j*kz(n)*z), reflection*np.exp(-1j*kz(n)*z)
            s, u = forward+backward, 1j*pi*(forward-backward)
        elif z > d:
            n = nO
            s = transmission*np.exp(1j*kz(n)*(z-d))
            u = 1j*po*s
        else:
            n = nl
            forward, backward = a*np.exp(1j*kz(n)*z), b*np.exp(-1j*kz(n)*z)
            s, u = forward+backward, 1j*pl*(forward-backward)
        if pol == "TE":
            harmonics[1, iz], harmonics[3, iz], harmonics[5, iz] = s, 1j*u, -kt*s
        else:
            harmonics[0, iz], harmonics[2, iz], harmonics[4, iz] = -1j*u, kt*s/(n*n), s
    return harmonics


class FrozenFieldReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = json.loads((FROZEN_DIR/"reference_v1.json").read_text(encoding="utf-8"))
        with np.load(FROZEN_DIR/"reference_v1.npz", allow_pickle=False) as archive:
            cls.baseline = {key: archive[key] for key in archive.files}

    def test_integrity_parameters_and_layout(self):
        meta = self.metadata
        archive = FROZEN_DIR/meta["archive"]["filename"]
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), meta["archive"]["sha256"])
        self.assertEqual(archive.stat().st_size, meta["archive"]["bytes"])
        self.assertEqual(meta["contract_version"], CONTRACT_VERSION)
        self.assertEqual(meta["components"], list(COMPONENTS))
        self.assertEqual([record["parameters"] for record in meta["cases"]], case_specs())
        keys = []
        for record in meta["cases"]:
            for item in record["arrays"].values():
                value = self.baseline[item["key"]]
                keys.append(item["key"])
                self.assertEqual(list(value.shape), item["shape"])
                self.assertEqual(str(value.dtype), item["dtype"])
                self.assertTrue(value.flags.c_contiguous)
        self.assertEqual(set(keys), set(self.baseline))
        # source_sha256 records the frozen solver provenance. Changing current
        # source while retaining correct outputs must not itself fail regression.
        self.assertTrue(all(len(value) == 64 for value in meta["source_sha256"].values()))

    def test_current_six_complex_fields_and_stable_phases(self):
        for record in self.metadata["cases"]:
            spec = record["parameters"]
            with self.subTest(case=spec["id"]):
                _, _, actual, diagnostics = sample_case(spec)
                self.assertEqual(diagnostics, record["diagnostics"])
                reference = {key: self.baseline[item["key"]] for key, item in record["arrays"].items()}
                tol = spec["tolerance"]
                for key in ("fields", "harmonics", "kx", "ky", "kzI", "kzII"):
                    np.testing.assert_allclose(actual[key], reference[key], **tol, err_msg=spec["id"]+"/"+key)
                for key in ("x", "z", "y", "orders", "incident_amplitudes"):
                    np.testing.assert_array_equal(actual[key], reference[key])
                amplitude = abs(reference["fields"])
                error = tol["atol"]+tol["rtol"]*amplitude
                expected_mask = reference["phase_mask"] | (amplitude <= 10*error)
                for component in spec["structural_zero_components"]:
                    ic = COMPONENTS.index(component)
                    expected_mask[ic] = True
                    self.assertLessEqual(float(np.max(abs(actual["fields"][ic]))), tol["atol"])
                np.testing.assert_array_equal(reference["phase_comparison_mask"], expected_mask)
                valid = ~expected_mask
                self.assertTrue(valid.any())
                self.assertFalse(actual["phase_mask"][valid].any())
                for key in ("phase_absolute", "phase_incident"):
                    difference = abs(np.exp(1j*actual[key][valid])-np.exp(1j*reference[key][valid]))
                    self.assertTrue(np.all(difference <= 2*error[valid]/amplitude[valid]+5e-14), key)

    def test_phi0_noise_is_excluded_from_phase_oracle(self):
        for case, zeros, active in (("phi0_s", ("Ex", "Ez", "Hy"), "Ey"),
                                    ("phi0_p", ("Ey", "Hx", "Hz"), "Hy")):
            for component in zeros:
                ic = COMPONENTS.index(component)
                self.assertTrue(self.baseline[case+"__phase_comparison_mask"][ic].all())
                self.assertLessEqual(np.max(abs(self.baseline[case+"__fields"][ic])), 3e-12)
            self.assertTrue((~self.baseline[case+"__phase_comparison_mask"][COMPONENTS.index(active)]).any())


class AnalyticFieldContract(unittest.TestCase):
    def test_interface_and_absorbing_evanescent_film_six_complex_fields(self):
        for spec in case_specs():
            if not spec["id"].startswith(("interface_", "absorbing_film_")):
                continue
            with self.subTest(case=spec["id"]):
                res, fields, arrays, _ = sample_case(spec)
                expected = scalar_analytic(spec, arrays["z"])
                kt = spec["stack"]["nI"]*np.sin(np.radians(spec["theta_deg"]))
                carrier = np.exp(-1j*res.k0*kt*arrays["x"])
                np.testing.assert_allclose(arrays["fields"], expected[:, :, None]*carrier, atol=4e-14, rtol=4e-14)
                np.testing.assert_allclose(arrays["harmonics"][:, :, spec["M"]], expected, atol=4e-14, rtol=4e-14)
                if spec["M"]:
                    off_zero = np.arange(2*spec["M"]+1) != spec["M"]
                    np.testing.assert_array_equal(arrays["harmonics"][:, :, off_zero], 0.)
                    self.assertGreater(res.kIIz[spec["M"]].imag, 0.)
                    self.assertAlmostEqual(float(res.DEt[spec["M"]]), 0.)
                norm = normalization(gui_config(spec))
                e2 = sum(abs(fields["E"+axis])**2 for axis in "xyz")/norm["E"]**2
                np.testing.assert_allclose(fields["E2"], e2, atol=4e-14, rtol=4e-14)

    def test_phi0_cartesian_mapping_includes_tm_nI(self):
        specs = {spec["id"]: spec for spec in case_specs()}
        for scalar, conical, scale in (("grating_te", "phi0_s", 1.),
                                       ("grating_tm", "phi0_p", specs["grating_tm"]["stack"]["nI"])):
            a = sample_case(specs[scalar])[2]
            b = sample_case(specs[conical])[2]
            for key in ("fields", "harmonics"):
                np.testing.assert_allclose(b[key], scale*a[key], atol=2e-12, rtol=2e-12)

    def test_conical_uniform_film_analytic_s_p_and_ellipse(self):
        spec = next(s for s in case_specs() if s["id"] == "absorbing_film_te")
        st = build_stack(spec["stack"])
        z = np.asarray(spec["zs"])
        te = scalar_analytic(spec, z)
        tm = scalar_analytic(dict(spec, polarization="TM"), z)*st.nI
        phi, y = np.radians(29.), .11
        s, c = np.sin(phi), np.cos(phi)
        basis_s = np.array([-s*te[1], c*te[1], te[2], c*te[3], s*te[3], te[5]])
        basis_p = np.array([c*tm[0], s*tm[0], tm[2], -s*tm[4], c*tm[4], tm[5]])
        for vin in ([1., 0.], [0., 1.], [.3+.2j, -.7+.5j]):
            with self.subTest(vin=vin):
                result = solve_conical(st, spec["wavelength"], spec["theta_deg"], 29., 0,
                                       jones_in=vin, need_fields=True)
                fields = conical_map(result, z, nx=7, y=y)
                v = np.asarray(vin, complex)/np.linalg.norm(vin)
                carrier = np.exp(-1j*(result.kxm[0]*fields["x"]+result.ky*y))
                expected = (basis_s*v[0]+basis_p*v[1])[:, :, None]*carrier
                np.testing.assert_allclose(np.stack([fields[key] for key in COMPONENTS]), expected, atol=8e-14, rtol=8e-14)

    def test_same_conical_result_postselection_and_global_phase(self):
        spec = next(s for s in case_specs() if s["id"] == "conical_linear")
        result = solve_case(spec)
        sample = lambda vin: conical_map(result, spec["zs"], nx=spec["nx"], y=spec["y"], jones_in=vin)
        field_s, field_p = sample([1., 0.]), sample([0., 1.])
        vin = np.array([.3+.2j, -.7+.5j])
        ellipse, shifted = sample(vin), sample(vin*np.exp(-.73j))
        v = vin/np.linalg.norm(vin)
        for key in COMPONENTS:
            np.testing.assert_allclose(ellipse[key], v[0]*field_s[key]+v[1]*field_p[key], atol=3e-13, rtol=3e-13)
            np.testing.assert_allclose(shifted[key], np.exp(-.73j)*ellipse[key], atol=3e-13, rtol=3e-13)

    def test_interface_ownership_tolerance_and_exit_priority(self):
        st = rc.Stack(1.1, [rc.Layer(.12, n=1.3), rc.Layer(.17, n=1.8)], 1.4, .6)
        result = rc.solve(st, .83, 23., 0, "TM")
        zs = [.12, .12+5e-13, .12+2e-12, result.z_bot, result.z_bot+5e-13]
        fields = rc.field_map(result, zs, nx=3)
        eps = np.array([1.3**2, 1.3**2, 1.8**2, 1.8**2, 1.4**2])
        np.testing.assert_allclose(fields["Ez"][:, 0], result.Kx[0, 0]*fields["Hy"][:, 0]/eps, atol=1e-15)
        np.testing.assert_array_equal(fields["Ez"][0], fields["Ez"][1])

    def test_known_no_layer_and_conical_zero_root_boundaries(self):
        st = rc.Stack(1.3, [], 1.7, .73)
        for pol in ("TE", "TM"):
            r = rc.solve(st, .81, 23., 0, pol)
            with self.assertRaises(ValueError):
                rc.field_at(r, 0.)
            with self.assertRaises(ValueError):
                rc.field_map(r, [0.], nx=3)
        r = solve_conical(st, .81, 23., 29., 0, need_fields=True)
        with self.assertRaises(ValueError):
            conical_at(r, 0.)
        with self.assertRaises(ValueError):
            conical_map(r, [0.], nx=3)
        kx = 2*np.sin(np.radians(30.))
        for d in (0., .2):
            with self.assertRaises(rc.RayleighAnomalyError):
                solve_conical(rc.Stack(2., [rc.Layer(d, n=kx)], 2., 1.), 1., 30., 0., 0, need_fields=True)


class PhaseContract(unittest.TestCase):
    def test_incident_carrier_sign_real_part_and_transposed_layout(self):
        x, z, y = np.array([-.2, .1, .8]), np.array([-.3, .4]), .12
        result = dict(ms=np.array([0]), kxm=np.array([1.7+.3j]), ky=-.4+.2j, kIz=np.array([2.1+.5j]))
        carrier = np.exp(1j*(-1.7*x[None, :]+.4*y+2.1*z[:, None]))
        fields = dict(x=x, z=z, y=y, Ex=(2-.3j)*carrier)
        np.testing.assert_allclose(incident_reference(result, fields), carrier, atol=2e-15)
        relative = phase_data(result, fields, "ex", reference="incident")
        np.testing.assert_allclose(relative["phase"], np.angle(2-.3j), atol=2e-15)
        transposed = dict(fields, Ex=fields["Ex"].T)
        np.testing.assert_allclose(incident_reference(result, transposed), carrier.T, atol=2e-15)

    def test_threshold_equality_zero_nonfinite_and_invalid_floor(self):
        field = np.array([[1., 1e-6, 0., np.nan], [1.00000001e-6, .7, 0., 1.]], complex)
        fields = dict(Ex=field)
        expected = [[False, True, True, True], [False, False, True, False]]
        np.testing.assert_array_equal(phase_data(None, fields, "Ex")["mask"], expected)
        self.assertTrue(phase_data(None, dict(Ex=np.zeros((2, 4))), "Ex", phase_floor=0)["mask"].all())
        for floor in (-1., np.inf, np.nan, "bad"):
            with self.assertRaises(ValueError):
                phase_data(None, fields, "Ex", phase_floor=floor)

    def test_projection_conjugation_preserves_scale_and_rejects_intensity(self):
        fields = dict(Ex=np.full((2, 3), 1.+.4j), Ey=np.full((2, 3), .3-.2j),
                      Ez=np.zeros((2, 3)), E2=np.ones((2, 3)))
        projected = phase_data(None, fields, "E", projection=[1., 1j, 0.])
        expected = fields["Ex"]-1j*fields["Ey"]
        np.testing.assert_array_equal(projected["field"], expected)
        np.testing.assert_array_equal(projected["amplitude"], abs(expected))
        with self.assertRaises(ValueError):
            complex_component(fields, "E2")

    def test_custom_reference_broadcast_masks_zero_and_nonfinite(self):
        field = np.full((2, 3), np.exp(.7j))
        reference = np.array([2*np.exp(.2j), 0., np.nan+1j*np.nan])
        phase = phase_data(None, dict(Ex=field), "Ex", reference=reference)
        np.testing.assert_array_equal(phase["mask"], [[False, True, True], [False, True, True]])
        np.testing.assert_allclose(phase["phase"][:, 0], .5, atol=1e-15)
        np.testing.assert_allclose(phase["amplitude"], 1., atol=1e-15)

    def test_unwrap_valid_segments_and_both_axis_order(self):
        target = np.array([[0., 2.8, 5.5, 0., .3, 3.2, 6.1],
                           [2.9, 5.6, 8.3, 0., 3.2, 6.1, 8.9],
                           [5.8, 8.4, 11., 0., 6., 8.9, 11.7]])
        field = np.exp(1j*target)
        field[:, 3] = 0.
        fields = dict(x=np.arange(7.), z=np.arange(3.), Ex=field)
        expected_x = target.copy()
        expected_x[2, :3] -= 2*np.pi
        expected_x[1:, 4:] -= 2*np.pi
        valid = abs(field) > 0
        along_x = phase_data(None, fields, "Ex", unwrap="horizontal")
        np.testing.assert_allclose(along_x["phase"].data[valid], expected_x[valid], atol=3e-15)
        both = phase_data(None, fields, "Ex", unwrap="both")
        np.testing.assert_allclose(both["phase"].data[valid], target[valid], atol=3e-15)
        self.assertTrue(both["mask"][:, 3].all())
        # This vortex-like pattern distinguishes x-then-z from z-then-x.
        target = np.array([[0., 2.], [-2., 3.]])
        fields = dict(x=np.arange(2.), z=np.arange(2.), Ex=np.exp(1j*target))
        np.testing.assert_allclose(phase_data(None, fields, "Ex", unwrap="both")["phase"], target, atol=1e-15)

    def test_gui_vector_normalization_and_direction_phase(self):
        st = rc.Stack(1.3, [rc.Layer(.17, n=1.3)], 1.3, .73)
        for pol in ("TE", "TM"):
            result = rc.solve(st, .81, 23., 0, pol)
            fields = rc.field_map(result, [-.1, .08, .27], nx=5)
            config = dict(mode="planar", structure=dict(n_incident=1.3),
                          incident=dict(polarization=pol, wavelength=.81, angle_deg=23.))
            for key in ("E2", "H2"):
                displayed, _ = view_data(fields, config, key)
                np.testing.assert_allclose(displayed, 1., atol=2e-15)
            component = "Hx" if pol == "TE" else "Hy"
            displayed, _ = view_data(fields, config, component, "phase_relative")
            expected_phasor = -1. if pol == "TE" else 1.
            np.testing.assert_allclose(np.exp(1j*displayed), expected_phasor, atol=2e-15)
            completed = complete_six_components(fields, pol)
            generic = phase_data(result, completed, component, reference="incident")["phase"].filled(np.nan)
            np.testing.assert_allclose(np.exp(1j*displayed), np.exp(1j*generic), atol=2e-15)
            zero_component = "Ex" if pol == "TE" else "Ey"
            zero, info = view_data(fields, config, zero_component, "phase")
            self.assertTrue(np.isnan(zero).all())
            self.assertTrue(info["phase_undefined"])


if __name__ == "__main__":
    unittest.main()
