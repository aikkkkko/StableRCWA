"""Exact-cutoff and Schur near fields, checked against analytic/first-order solutions."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

GUI = Path(__file__).resolve().parents[1]
if str(GUI) not in sys.path:
    sys.path.insert(0, str(GUI))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from scipy import linalg
import rcwa1d as rc
from conical.rcwa_conical import solve_conical
from conical.modes_conical import maxwell_operators, homogeneous_modes
from conical.fields_conical import field_at, field_map
from mode_propagation import field_combinations, outgoing_triangular_root


def cascade(a, b):
    """Independent reference star product; order dd, du, ud, uu."""
    dd, du, ud, uu = a
    bd, bu, br, bt = b
    d = np.eye(len(dd)) - du @ br
    x = np.linalg.solve(d, dd)
    y = np.linalg.solve(d, du @ bt)
    return bd @ x, bu + bd @ y, ud + uu @ br @ x, uu @ (bt + br @ y)


def boundary(up, down):
    den = up + down
    return tuple(np.diag(x) for x in (2 * up / den, (down - up) / den,
                                      (up - down) / den, 2 * down / den))


def direct_layer(layer, res, distance):
    """4N first-order exponential + S doubling, with no eigenproblem/square root.

    The transfer state is [F; G_aux] with G_aux = -G_physical, hence -P/-Q.
    direct_field converts its returned second state back to physical G.
    """
    n = len(res.ms)
    if layer.homogeneous:
        eps = layer.n ** 2 * np.eye(n)
        c = eps
    else:
        eps, inv = layer.operators(res.stack.period, res.M)
        c = np.linalg.solve(inv, np.eye(n))
    p, q, _ = maxwell_operators(eps, c, np.diag(res.kx_over_k0), res.kappa)
    cs, sn = np.diag(res.cos_phi_m), np.diag(res.sin_phi_m)
    re = np.block([[-sn, cs], [cs, sn]])
    rv = np.block([[-cs, -sn], [-sn, cs]])
    p, q = re @ p @ rv, rv @ q @ re
    n = 2 * n
    zero, eye = np.zeros((n, n), complex), np.eye(n, dtype=complex)
    m = np.block([[zero, -p], [-q, zero]]) * (res.k0 * distance)
    norm = np.linalg.norm(m, 1)
    steps = max(0, int(np.ceil(np.log2(norm / .25)))) if norm > .25 else 0
    u = np.block([[eye, eye], [-1j * eye, 1j * eye]])
    ui = .5 * np.block([[eye, 1j * eye], [eye, -1j * eye]])
    t = ui @ linalg.expm(m / 2. ** steps) @ u
    a, b, c, d = t[:n, :n], t[:n, n:], t[n:, :n], t[n:, n:]
    uu, ud = np.linalg.solve(d, eye), -np.linalg.solve(d, c)
    s = (a + b @ ud, b @ uu, ud, uu)
    for _ in range(steps):
        s = cascade(s, s)
    return s


def direct_field(res, z, vin):
    """Tangential F,G from independent scatterers above/below a point."""
    n = 2 * len(res.ms)
    pref = np.full(n, -1j)
    eye, zero = np.eye(n, dtype=complex), np.zeros((n, n), complex)
    left = boundary(res.mode_I.vdiag, pref)
    right = (eye, zero, zero, eye)
    top = 0.
    for layer in res.stack.layers:
        cut = np.clip(z - top, 0., layer.d)
        if cut > 0:
            left = cascade(left, direct_layer(layer, res, cut))
        if cut < layer.d:
            right = cascade(right, direct_layer(layer, res, layer.d - cut))
        top += layer.d
    right = cascade(right, boundary(pref, res.mode_II.vdiag))
    a = np.linalg.solve(eye - left[1] @ right[2], left[0] @ (res.incident_modes @ vin))
    b = right[2] @ a
    return a + b, 1j * (a - b)


class DegenerateFields(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stack = rc.Stack(1., [rc.Layer(.38, edges=[0., .1722, .574], n_values=[1.46, 1.]),
                                  rc.Layer(.115, n=1.96)], 1.46, .574)
        cls.wavelength = .980484673140678
        cls.res = solve_conical(cls.stack, cls.wavelength, 40., 25., 8, need_fields=True)

    def test_planar_exact_cutoff_field(self):
        kx = 2 * np.sin(np.radians(30.))
        d = 1 / (np.pi * np.sqrt(4 - kx * kx))
        st = rc.Stack(2., [rc.Layer(d, n=kx)], 2., 1.)
        for pol, ratio in (("TE", 2.), ("TM", .5)):
            with self.subTest(pol=pol):
                r = rc.solve(st, 1., 30., 0, pol)
                reflection = -1j * ratio / (2 - 1j * ratio)
                transmission = 2 / (2 - 1j * ratio)
                self.assertEqual(r.n_zero_modes, 1)
                np.testing.assert_allclose(r.R, reflection, atol=2e-15)
                np.testing.assert_allclose(r.T, transmission, atol=2e-15)
                zs = np.linspace(0, d, 9)
                fields = rc.field_map(r, zs, nx=8)
                for j, z in enumerate(zs):
                    f, u = rc.field_at(r, z)
                    expected = 1 + reflection * (1 - 2*z/d)
                    np.testing.assert_allclose(f, expected, atol=2e-15)
                    np.testing.assert_allclose(u, -2*reflection/(r.k0*d) / (1 if pol == "TE" else kx*kx), atol=2e-15)
                    phase = np.exp(-1j * np.outer(fields["x"], r.kxm))
                    np.testing.assert_allclose(fields["Ey" if pol == "TE" else "Hy"][j], phase @ f, atol=3e-15)
                    np.testing.assert_allclose(fields["Hx" if pol == "TE" else "Ex"][j],
                                               phase @ ((1j if pol == "TE" else -1j)*u), atol=3e-15)
                # Both fields are continuous across either horizontal interface.
                for surface in (0., d):
                    a = rc.field_at(r, surface - 1e-8)
                    b = rc.field_at(r, surface + 1e-8)
                    for av, bv in zip(a, b):
                        np.testing.assert_allclose(av, bv, atol=3e-7, rtol=0)

    def test_zero_thickness_and_external_cutoff(self):
        kx = 2 * np.sin(np.radians(30.))
        st = rc.Stack(2., [rc.Layer(0., n=kx)], 2., 1.)
        for pol in ("TE", "TM"):
            r = rc.solve(st, 1., 30., 0, pol)
            np.testing.assert_allclose(r.R, 0, atol=1e-15)
            np.testing.assert_allclose(r.T, 1, atol=1e-15)
            self.assertTrue(np.isfinite(rc.field_at(r, 0)[0]).all())
        with self.assertRaises(rc.RayleighAnomalyError):
            rc.solve(rc.Stack(1., [rc.Layer(.2, n=1.4)], 1.5, 1.), 1., 0., 1)

    def test_zero_helper_reference_anchors(self):
        # a_minus is at the bottom, not the top. Non-unit values catch wrong anchoring/signs.
        a, b = np.array([2+1j]), np.array([-.3+.2j])
        z = np.array([0., .4, 1.])
        f, u = field_combinations(np.array([0j]), a, b, z, 1., 3., zero_basis=True)
        np.testing.assert_allclose(f, .4*((4-3*z)*a[:, None]+(1+3*z)*b[:, None]))
        np.testing.assert_allclose(u, np.broadcast_to(.4*(-a+b)[:, None], f.shape))

    def test_schur_jordan_coupling_and_zero_rejection(self):
        t = np.array([[4., 3.], [0., 4.]], complex)
        g = outgoing_triangular_root(t, rc._branch)
        np.testing.assert_allclose(g, [[2., .75], [0., 2.]], atol=1e-15)
        zs = np.array([0., .3, .8])
        f, u = field_combinations(np.diag(g), np.array([0., 1.]), np.zeros(2), zs, 1., 1., generator=g)
        expected = np.array([-.75*zs*np.exp(-2*zs), np.exp(-2*zs)])
        np.testing.assert_allclose(f, expected, atol=1e-15)
        np.testing.assert_allclose(u, -expected, atol=1e-15)
        for grid in (np.linspace(0, 1, 11), np.array([.8, 0., .3, .3])):
            a, b = np.array([0., 1.]), np.array([1.+1j, .3])
            f, u = field_combinations(np.diag(g), a, b, grid, 1., 1., generator=g)
            plus = np.array([-.75*grid*np.exp(-2*grid), np.exp(-2*grid)])
            distance = 1-grid
            minus = np.array([(b[0]-.75*distance*b[1])*np.exp(-2*distance), b[1]*np.exp(-2*distance)])
            np.testing.assert_allclose(f, plus+minus, atol=2e-15)
            np.testing.assert_allclose(u, -plus+minus, atol=2e-15)
        with self.assertRaises(np.linalg.LinAlgError):
            outgoing_triangular_root(np.zeros((1, 1)), rc._branch)
        with self.assertRaises(ValueError):
            field_combinations(np.ones(1), np.ones(1), np.ones(1), [-.1], 1, 1)

    def test_conical_first_order_reference(self):
        r = self.res
        self.assertEqual(r.n_schur, 1)
        self.assertTrue(r.degenerate_mode_warning)
        self.assertLess(r.mode_residual, 1e-11)
        eps, _ = r.stack.layers[0].operators(r.stack.period, r.M)
        n = len(r.ms)
        cs, sn, kx = r.cos_phi_m, r.sin_phi_m, r.kx_over_k0
        for vin in (np.array([1., 0.]), np.array([0., 1.]), np.array([1j, 1.])/np.sqrt(2)):
            # Verify a point inside the Schur layer, in addition to both interfaces.
            for z in (0., .19, .38):
                got = field_at(r, z, vin)
                f, g = direct_field(r, z, vin)
                np.testing.assert_allclose(got["F"], f, atol=2e-10, rtol=2e-10)
                np.testing.assert_allclose(got["G"], g, atol=2e-10, rtol=2e-10)
                # Independent Cartesian and longitudinal reconstruction extends
                # this same first-order oracle to every complex field component.
                # z=.38 belongs to the first layer, matching the public contract.
                ex, ey = -sn*f[:n]+cs*f[n:], cs*f[:n]+sn*f[n:]
                hx, hy = -cs*g[:n]-sn*g[n:], -sn*g[:n]+cs*g[n:]
                ez = 1j*np.linalg.solve(eps, r.kappa*hx-kx*hy)
                hz = r.kappa*ex-kx*ey  # already Z0H, unlike auxiliary h=iZ0H
                expected = dict(Ex=ex, Ey=ey, Ez=ez, Hx=-1j*hx, Hy=-1j*hy, Hz=hz)
                for key, value in expected.items():
                    np.testing.assert_allclose(got[key], value, atol=3e-10, rtol=3e-10)

    def test_conical_map_and_horizontal_continuity(self):
        r = self.res
        vin = np.array([.3+.2j, -.7+.5j]); vin /= np.linalg.norm(vin)
        zs = np.array([-.1, 0., .07, .19, .38, .45, r.z_bot, r.z_bot+.1])
        fields = field_map(r, zs, nx=13, y=.11, jones_in=vin)
        phase = np.exp(-1j*(np.outer(fields["x"], r.kxm)+r.ky*.11))
        for j, z in enumerate(zs):
            point = field_at(r, z, vin)
            for key in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Dx"):
                np.testing.assert_allclose(fields[key][j], phase @ point[key], atol=3e-12, rtol=3e-12)
        for surface in (0., .38, r.z_bot):
            above, below = field_at(r, surface-1e-8, vin), field_at(r, surface+1e-8, vin)
            for key in ("Ex", "Ey", "Hx", "Hy"):
                np.testing.assert_allclose(above[key], below[key], atol=2e-6, rtol=0)

    def test_conical_maxwell_equations(self):
        r = self.res
        eps, inv = r.stack.layers[0].operators(r.stack.period, r.M)
        c = np.linalg.solve(inv, np.eye(len(inv)))
        p, q, _ = maxwell_operators(eps, c, np.diag(r.kx_over_k0), r.kappa)
        # Fourth-order differences keep truncation small without amplifying expm
        # roundoff by an unnecessarily tiny depth interval.
        z, step = .19, 1e-3
        for vin in (np.array([1., 0.]), np.array([0., 1.])):
            samples = [field_at(r, z+k*step, vin) for k in (-2, -1, 0, 1, 2)]
            center = samples[2]
            electric = lambda f: np.r_[f["Ex"], f["Ey"]]
            magnetic = lambda f: 1j*np.r_[f["Hx"], f["Hy"]]  # h=i Z0 H
            derivative = lambda fun: (fun(samples[0])-8*fun(samples[1])+8*fun(samples[3])-fun(samples[4]))/(12*step*r.k0)
            de, dh = derivative(electric), derivative(magnetic)
            for actual, expected in ((de, p @ magnetic(center)), (dh, q @ electric(center))):
                error = np.linalg.norm(actual-expected)/max(1., np.linalg.norm(expected))
                self.assertLess(error, 2e-8)

    def test_non_degenerate_and_thick_absorbing_layers(self):
        for absorbing in (False, True):
            st = rc.Stack(1., [rc.Layer(3. if absorbing else .2, edges=[0., .2, .6],
                         n_values=[1.7+.1j if absorbing else 1.7, 1.])], 1.4, .6)
            ordinary = solve_conical(st, .83, 27., 40., 3, need_fields=True)
            forced = solve_conical(st, .83, 27., 40., 3, need_fields=True, mode_method="schur")
            self.assertEqual(ordinary.n_schur, 0)
            self.assertEqual(forced.n_schur, 1)
            np.testing.assert_allclose(ordinary.Jr, forced.Jr, atol=1e-11, rtol=1e-11)
            for z in (0., st.layers[0].d/2, st.layers[0].d):
                a, b = field_at(ordinary, z), field_at(forced, z)
                for key in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
                    np.testing.assert_allclose(a[key], b[key], atol=2e-11, rtol=2e-11)

    def test_unexcited_incident_evanescent_modes(self):
        # Far above the structure, incoming evanescent exponentials would overflow;
        # their amplitudes are exactly zero and must not be evaluated.
        stack = rc.Stack(1., [rc.Layer(.2, n=1.5)], 1.4, .6)
        with np.errstate(over="raise", invalid="raise"):
            for pol in ("TE", "TM"):
                r = rc.solve(stack, .83, 20., 3, pol)
                self.assertTrue(all(np.isfinite(v).all() for v in rc.field_at(r, -100.)))
                f = rc.field_map(r, [-100., -90.], nx=8)
                self.assertTrue(np.isfinite(f["E2"]).all())
            r = self.res
            self.assertTrue(all(np.isfinite(v).all() for v in field_at(r, -100.).values()))
            f = field_map(r, [-100., -90.], nx=8)
            self.assertTrue(np.isfinite(f["E2"]).all())

    def test_cpp_jones_at_degenerate_point(self):
        exe = os.environ.get("RCWA_JSON_EXE")
        if not exe:
            self.skipTest("Set RCWA_JSON_EXE to verify the C++ interface")
        r = self.res
        request = dict(mode="conical", M=8, stack=dict(nI=1., nII=1.46, period=.574,
            layers=[dict(d=.38, edges=[0., .1722, .574], n_values=[1.46, 1.]), dict(d=.115, n=1.96)]),
            points=[dict(wavelength=self.wavelength, theta_deg=40., phi_deg=25.)])
        reply = json.loads(subprocess.check_output([exe], input=json.dumps(request).encode()))
        self.assertTrue(reply["ok"])
        item = reply["results"][0]
        self.assertEqual(item["n_schur"], r.n_schur)
        for key in ("Jr", "Jt"):
            values = np.array(item[key]); values = values[..., 0]+1j*values[..., 1]
            np.testing.assert_allclose(values.reshape(-1, 2, 2), r[key], atol=2e-11, rtol=2e-11)


if __name__ == "__main__":
    unittest.main()
