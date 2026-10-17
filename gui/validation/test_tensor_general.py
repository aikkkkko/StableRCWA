"""General-tensor (6x6 constitutive) grating solver: limits, analytic slabs, symmetries, frozen references.

The checks are (a) reduction to the existing isotropic, diagonal-tensor and magnetic
solvers, (b) uniform slabs against closed forms, an independently coded Berreman 4x4
transfer matrix and an independent plane-wave solution for bianisotropic media,
(c) exact identities of the grating problem (energy, mirror, C2, reciprocity, duality,
Maxwell's equations for the reconstructed fields), and (d) numbers from outside this
code base: COMSOL finite-element results and a published table.
"""
import os
from pathlib import Path
import sys
import unittest

GUI = Path(__file__).resolve().parents[1]
if str(GUI) not in sys.path:
    sys.path.insert(0, str(GUI))
os.environ.setdefault("MPLBACKEND", "Agg")

import warnings

import numpy as np
from scipy.linalg import eig as generalized_eig, expm
import rcwa1d as rc
from conical.rcwa_conical import solve_conical
from conical.fields_conical import field_at as conical_field_at, field_map as conical_field_map
from mag import rcwa1d_mag as mag1d
from mag.rcwa_conical_mag import solve_conical as solve_conical_magnetic
from tensor import rcwa_tensor as rt

LAM, PERIOD = 1.053, 0.8
K0 = 2 * np.pi / LAM
EDGES = [0.0, 0.2, 0.45, 0.8]                       # three segments, no mirror symmetry
TILTED = rt.uniaxial(1.658, 1.486, (0.4, -0.7, 0.6))
BIAXIAL = rt.biaxial(1.4, 1.5, 1.6, rt.rotation_matrix((1, 2, 3), 37.0))
FILM = rt.uniaxial(2.0, 2.2, (1, 1, 1))
TIGHT = 1e-10


def passive_nonsymmetric_eps():
    """Hermitian part with complex off-diagonals plus a positive semidefinite loss part."""
    hermitian = (np.diag([2.2, 2.6, 3.0]) + 0.2 * np.array([[0, 1, -0.5], [1, 0, 0.7], [-0.5, 0.7, 0]])
                 + 0.15j * np.array([[0, -1, 0.4], [1, 0, -0.8], [-0.4, 0.8, 0]]))
    b = 0.15 * np.array([[1, 0.5j, 0.2], [-0.3, 1, 0.4j], [0.1j, -0.6, 1]])
    return hermitian + 1j * (b @ b.conj().T)


def tensor_grating(media=(TILTED, BIAXIAL, 1.0), edges=EDGES, exit_medium=1.5, film=FILM):
    layers = [rt.TensorLayer(0.3, edges=edges, media=list(media))]
    if film is not None:
        layers.append(rt.TensorLayer(0.15, medium=film))
    return rt.TensorStack(1.0, layers, exit_medium, PERIOD)


def channels(res):
    return np.array([res["DEr_" + k] for k in ("ss", "sp", "ps", "pp")]
                    + [res["DEt_" + k] for k in ("ss", "sp", "ps", "pp")])


def zero_order_xy(res, e_t):
    """Reflected and transmitted (Ex, Ey) of the zero order for an incident tangential field e_t."""
    n, i0 = len(res.ms), res.i0
    rows = [i0, n + i0]
    weights = np.linalg.solve(res.incident[rows], np.asarray(e_t, complex))
    return res.R_xy[rows] @ weights, res.T_xy[rows] @ weights


def airy(n1, n2, n3, d):
    """Normal-incidence amplitude reflection and transmission of a slab (E convention)."""
    r12, r23 = (n1 - n2) / (n1 + n2), (n2 - n3) / (n2 + n3)
    t12, t23 = 2 * n1 / (n1 + n2), 2 * n2 / (n2 + n3)
    e = np.exp(1j * K0 * n2 * d)
    den = 1 + r12 * r23 * e * e
    return (r12 + r23 * e * e) / den, t12 * t23 * e / den


def berreman_slab(eps, d, n_in, n_out, theta_deg, phi_deg):
    """Independent reference: Berreman's 4x4 matrix and a matrix-exponential transfer matrix.

    State (Ex, Hy, Ey, -Hx) in a frame whose x' axis is the physical in-plane wave vector,
    fields exp(i k0 K x' - i w t).  No eigenmodes, no mode sorting, no S-matrix.  Returns,
    for the solver's unit s and unit p incidence, the lab-frame (Ex, Ey) reflected at z = 0
    and transmitted at z = d.
    """
    th, ph = np.radians(theta_deg), np.radians(phi_deg)
    frame = np.array([[-np.cos(ph), -np.sin(ph), 0], [np.sin(ph), -np.cos(ph), 0], [0, 0, 1]])
    e = frame @ eps @ frame.T
    K, zz = n_in * np.sin(th), e[2, 2]
    delta = np.array([
        [-K * e[2, 0] / zz, 1 - K * K / zz, -K * e[2, 1] / zz, 0],
        [e[0, 0] - e[0, 2] * e[2, 0] / zz, -K * e[0, 2] / zz, e[0, 1] - e[0, 2] * e[2, 1] / zz, 0],
        [0, 0, 0, 1],
        [e[1, 0] - e[1, 2] * e[2, 0] / zz, -K * e[1, 2] / zz, e[1, 1] - K * K - e[1, 2] * e[2, 1] / zz, 0],
    ], complex)
    transfer = expm(1j * K0 * d * delta)

    def plane_waves(n, sign):                 # s: E = y';  p: E = (kz, 0, -K) / n;  H = k x E
        kz = sign * np.sqrt(complex(n * n - K * K))
        return np.array([0, 0, 1, kz]), np.array([kz / n, n, 0, 0])

    s_in, p_in = plane_waves(n_in, +1)
    s_ref, p_ref = plane_waves(n_in, -1)
    s_tra, p_tra = plane_waves(n_out, +1)
    out = []
    for incident in (-s_in, -p_in):           # the solver's s is -y', its p is -(kz, 0, -K) / n
        system = np.column_stack((transfer @ s_ref, transfer @ p_ref, -s_tra, -p_tra))
        rs, rp, ts, tp = np.linalg.solve(system, -transfer @ incident)
        ref, tra = rs * s_ref + rp * p_ref, ts * s_tra + tp * p_tra
        back = frame.T[:2, :2]
        out.append((back @ ref[[0, 2]], back @ tra[[0, 2]]))
    return out


class ReductionToExistingSolvers(unittest.TestCase):
    def test_isotropic_gratings_match_conical_solver_including_fields(self):
        cases = {
            "symmetric": [rc.Layer(0.3, edges=[0, 0.3, 0.8], n_values=[1.8, 1.0])],
            "asymmetric": [rc.Layer(0.3, edges=EDGES, n_values=[1.8, 1.4, 1.0])],
            "absorbing with films": [rc.Layer(0.3, edges=EDGES, n_values=[1.8 + 0.13j, 1.4, 1.0]),
                                     rc.Layer(0.13, n=1.96), rc.Layer(0.18, n=1.46)],
        }
        for name, layers in cases.items():
            with self.subTest(name=name):
                ref = solve_conical(rc.Stack(1.0, layers, 1.5, PERIOD), LAM, 42.0, -40.0, M=15,
                                    psi_deg=33, delta_deg=20, need_fields=True)
                mine = [rt.TensorLayer(L.d, medium=L.n) if L.homogeneous
                        else rt.TensorLayer(L.d, edges=L.edges, media=list(L.n_values)) for L in layers]
                res = rt.solve_tensor(rt.TensorStack(1.0, mine, 1.5, PERIOD), LAM, 42.0, -40.0, M=15,
                                      psi_deg=33, delta_deg=20, need_fields=True)
                for key in ("Jr", "Jt", "DEr", "DEt", "Rs", "Rp", "Ts", "Tp"):
                    self.assertLess(np.abs(res[key] - ref[key]).max(), TIGHT, key)
                self.assertLess(np.abs(channels(res) - channels(ref)).max(), TIGHT)
                for z in (-0.2, 0.0, 0.1, 0.3, ref["z_bot"] - 0.01, ref["z_bot"] + 0.25):
                    a, b = conical_field_at(ref, z), rt.field_at(res, z)
                    for key in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Dx"):
                        self.assertLess(np.abs(a[key] - b[key]).max(), TIGHT, f"{key} at z={z}")

    def test_axis_aligned_crystals_match_planar_te_tm(self):
        for axis in "xyz":
            tooth, film = rc.uniaxial(1.658, 1.486 + 0.02j, axis), rc.uniaxial(2.2, 2.4, axis)
            ref_stack = rc.Stack(1.0, [rc.Layer(0.3, edges=EDGES, n_values=[tooth, 1.4, 1.0]), rc.Layer(0.2, n=film)], 1.5, PERIOD)
            stack = rt.TensorStack(1.0, [rt.TensorLayer(0.3, edges=EDGES, media=[tooth, 1.4, 1.0]),
                                         rt.TensorLayer(0.2, medium=film)], 1.5, PERIOD)
            for pol, psi in (("TE", 90.0), ("TM", 0.0)):
                with self.subTest(axis=axis, pol=pol):
                    ref = rc.solve(ref_stack, LAM, 42.0, 15, pol, need_fields=False)
                    res = rt.solve_tensor(stack, LAM, 42.0, 0.0, M=15, psi_deg=psi)
                    self.assertLess(np.abs(res.DEr - ref.DEr).max(), TIGHT)
                    self.assertLess(np.abs(res.DEt - ref.DEt).max(), TIGHT)
                    cross = res["DEr_ps" if pol == "TE" else "DEr_sp"]
                    self.assertLess(np.abs(cross).max(), 1e-20)       # TE and TM stay decoupled

    def test_isotropic_magnetic_grating_matches_mag_solver(self):
        eps, mu = [3.2 + 0.1j, 2.0, 1.0], [1.5, 1.0 + 0.05j, 1.0]
        ref_stack = mag1d.Stack(1.0, [mag1d.Layer(0.3, edges=EDGES, eps_values=eps, mu_values=mu)], 1.5, PERIOD)
        stack = rt.TensorStack(1.0, [rt.TensorLayer(0.3, edges=EDGES, media=[rt.Medium(e, m) for e, m in zip(eps, mu)])], 1.5, PERIOD)
        for pol, psi in (("TE", 90.0), ("TM", 0.0)):
            with self.subTest(pol=pol):
                ref = mag1d.solve(ref_stack, LAM, 42.0, 15, pol, need_fields=False)
                res = rt.solve_tensor(stack, LAM, 42.0, 0.0, M=15, psi_deg=psi)
                self.assertLess(np.abs(res.DEr - ref.DEr).max(), TIGHT)
                self.assertLess(np.abs(res.DEt - ref.DEt).max(), TIGHT)


class UniformSlab(unittest.TestCase):
    def test_general_tensors_match_berreman_transfer_matrix(self):
        tensors = {
            "tilted uniaxial": TILTED.eps,
            "absorbing rotated biaxial": rt.biaxial(1.5 + 0.01j, 1.7, 1.9 + 0.03j, rt.rotation_matrix((1, 2, 3), 37.0)).eps,
            "gyrotropic": rt.gyrotropic(2.3, (0.1, -0.2, 0.3)).eps,
            "passive nonsymmetric": passive_nonsymmetric_eps(),
        }
        d = 0.37
        for name, eps in tensors.items():
            for theta, phi in ((42.0, -40.0), (0.0, 25.0), (65.0, 110.0)):
                with self.subTest(name=name, theta=theta, phi=phi):
                    stack = rt.TensorStack(1.0, [rt.TensorLayer(d, medium=rt.Medium(eps=eps))], 1.5, PERIOD)
                    res = rt.solve_tensor(stack, LAM, theta, phi, M=1, need_fields=True)
                    ref = berreman_slab(eps, d, 1.0, 1.5, theta, phi)
                    rows = [res.i0, len(res.ms) + res.i0]
                    for col in range(2):
                        self.assertLess(np.abs(res.R_xy[rows, col] - ref[col][0]).max(), TIGHT)
                        self.assertLess(np.abs(res.T_xy[rows, col] - ref[col][1]).max(), TIGHT)
                    # the same slab through the dense 4N eigenproblem used for gratings
                    dense = rt.solve_tensor(stack, LAM, theta, phi, M=1, _force_dense=True)
                    self.assertEqual(dense.n_dense, 1)
                    self.assertLess(np.abs(dense.Jr - res.Jr).max(), TIGHT)
                    self.assertLess(np.abs(dense.Jt - res.Jt).max(), TIGHT)

    def test_wave_plate_at_normal_incidence(self):
        no, ne, d = 1.658, 1.486 + 0.01j, 0.41
        for alpha in (0.0, 27.0, 45.0, 90.0):
            a = np.radians(alpha)
            along, across = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
            stack = rt.TensorStack(1.0, [rt.TensorLayer(d, medium=rt.uniaxial(no, ne, (along[0], along[1], 0)))], 1.5, PERIOD)
            res = rt.solve_tensor(stack, LAM, 0.0, 10.0, M=1, need_fields=True)
            for vector, index in ((along, ne), (across, no)):
                with self.subTest(alpha=alpha, index=index):
                    r, t = zero_order_xy(res, vector)
                    r_ref, t_ref = airy(1.0, index, 1.5, d)
                    self.assertLess(np.abs(r - r_ref * vector).max(), TIGHT)
                    self.assertLess(np.abs(t - t_ref * vector).max(), TIGHT)

    def test_faraday_and_chiral_slabs_have_circular_eigenpolarizations(self):
        eps, g, kappa, d = 2.3 + 0.02j, 0.25, 0.07, 0.41
        faraday = rt.solve_tensor(rt.TensorStack(1.0, [rt.TensorLayer(d, medium=rt.gyrotropic(eps, (0, 0, g)))], 1.5, PERIOD),
                                  LAM, 0.0, 0.0, M=1, need_fields=True)
        pasteur = rt.solve_tensor(rt.TensorStack(1.0, [rt.TensorLayer(d, medium=rt.chiral(eps, kappa))], 1.5, PERIOD),
                                  LAM, 0.0, 0.0, M=1, need_fields=True)
        for sign in (+1, -1):
            circular = np.array([1, sign * 1j]) / np.sqrt(2)
            with self.subTest(medium="gyrotropic", sign=sign):       # n^2 = eps +- g
                r, t = zero_order_xy(faraday, circular)
                r_ref, t_ref = airy(1.0, np.sqrt(eps + sign * g), 1.5, d)
                self.assertLess(np.abs(r - r_ref * circular).max(), TIGHT)
                self.assertLess(np.abs(t - t_ref * circular).max(), TIGHT)
            with self.subTest(medium="chiral", sign=sign):           # same impedance, index sqrt(eps) +- kappa one way
                r, t = zero_order_xy(pasteur, circular)
                r_ref, t_ref = airy(1.0, np.sqrt(eps), 1.5, d)
                self.assertLess(np.abs(r - r_ref * circular).max(), TIGHT)
                self.assertLess(np.abs(t - t_ref * np.exp(1j * K0 * sign * kappa * d) * circular).max(), TIGHT)

    def test_rotating_tensor_and_azimuth_together_changes_nothing(self):
        medium = rt.Medium(eps=passive_nonsymmetric_eps(), mu=rt.biaxial(1.0, 1.1, 1.2, rt.rotation_matrix((2, -1, 1), 50.0)).eps)
        base = rt.solve_tensor(rt.TensorStack(1.0, [rt.TensorLayer(0.37, medium=medium)], 1.5, PERIOD), LAM, 42.0, -40.0, M=1)
        for alpha in (17.0, 90.0, 203.0):
            with self.subTest(alpha=alpha):
                turned = medium.rotated(rt.rotation_matrix("z", alpha))
                res = rt.solve_tensor(rt.TensorStack(1.0, [rt.TensorLayer(0.37, medium=turned)], 1.5, PERIOD), LAM, 42.0, -40.0 + alpha, M=1)
                self.assertLess(np.abs(res.Jr[1] - base.Jr[1]).max(), TIGHT)
                self.assertLess(np.abs(res.Jt[1] - base.Jt[1]).max(), TIGHT)

    def test_degenerate_ordinary_and_extraordinary_waves(self):
        # c-cut crystal at normal incidence, and a wave vector exactly along a tilted optic axis:
        # both waves see the ordinary index, and the two modes of an order share one eigenvalue
        plain = rt.TensorStack(1.0, [rt.TensorLayer(0.3, medium=1.658)], 1.5, PERIOD)
        c_cut = rt.TensorStack(1.0, [rt.TensorLayer(0.3, medium=rt.uniaxial(1.658, 1.486, "z"))], 1.5, PERIOD)
        a, b = rt.solve_tensor(c_cut, LAM, 0.0, 0.0, M=2), rt.solve_tensor(plain, LAM, 0.0, 0.0, M=2)
        self.assertLess(np.abs(a.Jr[2] - b.Jr[2]).max(), TIGHT)
        self.assertLess(np.abs(a.Jt[2] - b.Jt[2]).max(), TIGHT)
        theta = 30.0
        s = np.sin(np.radians(theta)) / 1.658
        along_k = (-s, 0.0, np.sqrt(1 - s * s))                      # physical wave vector is (-kx, -ky, +kz)
        tilted = rt.TensorStack(1.0, [rt.TensorLayer(0.3, medium=rt.uniaxial(1.658, 1.486, along_k))], 1.5, PERIOD)
        a, b = rt.solve_tensor(tilted, LAM, theta, 0.0, M=0), rt.solve_tensor(plain, LAM, theta, 0.0, M=0)
        self.assertLess(abs(a.Jr[0][0, 0] - b.Jr[0][0, 0]), TIGHT)
        self.assertLess(abs(a.energy - 1), 1e-12)

    def test_anisotropic_half_space(self):
        for alpha in (0.0, 30.0):
            a = np.radians(alpha)
            along, across = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
            res = rt.solve_tensor(rt.TensorStack(1.0, [], rt.uniaxial(1.658, 1.486, (along[0], along[1], 0)), PERIOD),
                                  LAM, 0.0, 0.0, M=1, need_fields=True)
            self.assertNotIn("Jt", res)                              # no s/p Jones matrix in an anisotropic exit medium
            self.assertLess(abs(res.energy - 1), 1e-12)
            for vector, index in ((along, 1.486), (across, 1.658)):
                with self.subTest(alpha=alpha, index=index):
                    r, _ = zero_order_xy(res, vector)
                    self.assertLess(np.abs(r - (1 - index) / (1 + index) * vector).max(), TIGHT)


class TensorGrating(unittest.TestCase):
    M = 12

    def solve(self, stack, phi=-40.0, **kwargs):
        for key, value in (("M", self.M), ("psi_deg", 33.0), ("delta_deg", 20.0)):
            kwargs.setdefault(key, value)
        return rt.solve_tensor(stack, LAM, 42.0, phi, **kwargs)

    def test_lossless_tensors_conserve_energy_and_flux_at_every_depth(self):
        for name, media in (("symmetric tensors", (TILTED, BIAXIAL, 1.0)),
                            ("gyrotropic tooth", (rt.gyrotropic(2.6, (0.2, 0.3, -0.25)), BIAXIAL, 1.0)),
                            ("tensor permeability", (rt.Medium(eps=TILTED.eps, mu=BIAXIAL.eps / 2), BIAXIAL, 1.0))):
            with self.subTest(name=name):
                res = self.solve(tensor_grating(media), need_fields=True)
                self.assertLess(abs(res.energy - 1), 1e-11)
                self.assertLess(res.mode_residual, 1e-11)
                transmitted = res.DEt.sum()
                incident = rt._order_flux(res.top.Fp @ res.incident @ res.jones_in).sum()
                for z in (-0.3, 0.0, 0.07, 0.3, 0.38, 0.45, 0.9):
                    flux = rt._order_flux(rt.field_at(res, z)["psi"]).sum() / incident
                    self.assertLess(abs(flux - transmitted), 1e-11, f"z={z}")

    def test_mirror_image_of_the_whole_problem(self):
        # x -> -x: structure reversed, polar tensors S X S, magnetoelectric blocks change sign, phi -> 180 - phi.
        # The s unit vector is a pseudovector under the mirror, so the image of [Es, Ep] is [-Es, Ep].
        def mirrored(medium):
            s = np.diag([-1.0, 1, 1, -1, 1, 1])
            out = rt.Medium()
            out.C = s @ medium.C @ s
            out.C[:3, 3:] *= -1
            out.C[3:, :3] *= -1
            return out

        tooth = rt.Medium(eps=TILTED.eps, mu=[1.0, 1.1, 1.2], xi=0.05j * np.eye(3), zeta=-0.05j * np.eye(3))
        res = self.solve(tensor_grating((tooth, BIAXIAL, 1.0)))
        flipped = rt.TensorStack(1.0, [rt.TensorLayer(0.3, edges=[0.0, 0.35, 0.6, 0.8], media=[1.0, mirrored(BIAXIAL), mirrored(tooth)]),
                                       rt.TensorLayer(0.15, medium=mirrored(FILM))], 1.5, PERIOD)
        other = self.solve(flipped, phi=180.0 + 40.0, delta_deg=20.0 + 180.0)
        self.assertLess(np.abs(channels(res) - channels(other)[:, ::-1]).max(), TIGHT)
        self.assertLess(np.abs(res.DEr - other.DEr[::-1]).max(), TIGHT)
        self.assertLess(np.abs(res.DEt - other.DEt[::-1]).max(), TIGHT)

    def test_half_turn_about_the_normal(self):
        # a proper rotation, so the complete Jones matrices must agree, order m <-> -m
        turn = np.diag([-1.0, -1, 1])
        res = self.solve(tensor_grating())
        turned = rt.TensorStack(1.0, [rt.TensorLayer(0.3, edges=[0.0, 0.35, 0.6, 0.8], media=[1.0, BIAXIAL.rotated(turn), TILTED.rotated(turn)]),
                                      rt.TensorLayer(0.15, medium=FILM.rotated(turn))], 1.5, PERIOD)
        other = self.solve(turned, phi=-40.0 + 180.0)
        self.assertLess(np.abs(res.Jr - other.Jr[::-1]).max(), TIGHT)
        self.assertLess(np.abs(res.Jt - other.Jt[::-1]).max(), TIGHT)

    def test_reciprocity_of_the_specular_order_and_its_violation(self):
        i0 = self.M
        lossy = rt.Medium(eps=TILTED.eps + 0.05j * np.eye(3))        # symmetric, absorbing
        forward, backward = self.solve(tensor_grating((lossy, BIAXIAL, 1.0))), self.solve(tensor_grating((lossy, BIAXIAL, 1.0)), phi=140.0)
        a, b = forward.Jr[i0], backward.Jr[i0]
        self.assertLess(abs(a[0, 0] - b[0, 0]), TIGHT)
        self.assertLess(abs(a[1, 1] - b[1, 1]), TIGHT)
        self.assertLess(abs(a[0, 1] + b[1, 0]), TIGHT)
        self.assertLess(abs(a[1, 0] + b[0, 1]), TIGHT)
        self.assertGreater(abs(a[0, 1]), 1e-3)                       # the cross terms are not trivially zero
        magneto = (rt.gyrotropic(2.6, (0.2, 0.3, -0.25)), BIAXIAL, 1.0)
        forward, backward = self.solve(tensor_grating(magneto)), self.solve(tensor_grating(magneto), phi=140.0)
        self.assertGreater(abs(forward.Jr[i0][0, 0] - backward.Jr[i0][0, 0]), 1e-3)

    def test_duality_between_permittivity_and_permeability(self):
        # in vacuum, swapping eps <-> mu everywhere swaps the roles of s and p
        a, b = TILTED.eps, rt.biaxial(1.2, 1.3, 1.1, rt.rotation_matrix((0, 1, 1), 20.0)).eps

        def stack(swap):
            first = rt.Medium(eps=b, mu=a) if swap else rt.Medium(eps=a, mu=b)
            second = rt.Medium(eps=1.0, mu=2.25) if swap else rt.Medium(eps=2.25, mu=1.0)
            return rt.TensorStack(1.0, [rt.TensorLayer(0.3, edges=EDGES, media=[first, second, 1.0])], 1.0, PERIOD)

        one, two = self.solve(stack(False)), self.solve(stack(True))
        swapped = np.array([two["DEr_pp"], two["DEr_ps"], two["DEr_sp"], two["DEr_ss"],
                            two["DEt_pp"], two["DEt_ps"], two["DEt_sp"], two["DEt_ss"]])
        self.assertLess(np.abs(channels(one) - swapped).max(), TIGHT)

    def test_anisotropic_exit_medium(self):
        grating = rt.TensorLayer(0.3, edges=EDGES, media=[TILTED, BIAXIAL, 1.0])
        bare = self.solve(rt.TensorStack(1.0, [grating], FILM, PERIOD))
        padded = self.solve(rt.TensorStack(1.0, [grating, rt.TensorLayer(0.77, medium=FILM)], FILM, PERIOD))
        self.assertLess(abs(bare.energy - 1), 1e-11)
        self.assertGreater(bare.DEt.sum(), 0.1)
        self.assertLess(np.abs(bare.DEr - padded.DEr).max(), TIGHT)   # a slab of the exit material is no interface
        self.assertLess(np.abs(bare.DEt - padded.DEt).max(), TIGHT)

    def test_li_and_laurent_rules_converge_to_the_same_limit(self):
        def efficiencies(M, rule):
            res = self.solve(tensor_grating(), M=M, factorization=rule)
            return np.r_[res.DEr[res.prop_r], res.DEt[res.prop_t]]

        reference = efficiencies(60, "li")
        li = {M: np.abs(efficiencies(M, "li") - reference).max() for M in (10, 20)}
        laurent = {M: np.abs(efficiencies(M, "laurent") - reference).max() for M in (20, 40)}
        self.assertLess(li[20], li[10] / 3)                           # about M^-2
        self.assertLess(laurent[40], laurent[20] / 1.6)               # about M^-1, towards the same numbers
        self.assertLess(laurent[40], 5e-4)
        self.assertLess(li[20], laurent[20] / 10)

    def test_small_period_limit_is_the_laminate_tensor(self):
        # period -> 0: uniform slab whose tensor is the zeroth Fourier coefficient of the factorization
        # (the classical laminate formula).  The remaining difference is an interface correction that
        # is first order in the period, as it also is for isotropic constituents.
        fill, cut, rest = 0.4, [0, 3], [1, 2, 4, 5]
        laminate = rt.Medium()
        laminate.C = rt._pivot(fill * rt._pivot(TILTED.C, cut, rest) + (1 - fill) * rt._pivot(BIAXIAL.C, cut, rest), cut, rest)
        uniform = rt.solve_tensor(rt.TensorStack(1.0, [rt.TensorLayer(0.3, medium=laminate)], 1.5, 1.0), LAM, 42.0, -40.0, M=0)
        errors = []
        for period in (0.05, 0.025):
            layer = rt.TensorLayer(0.3, edges=[0, fill * period, period], media=[TILTED, BIAXIAL])
            res = rt.solve_tensor(rt.TensorStack(1.0, [layer], 1.5, period), LAM, 42.0, -40.0, M=8)
            errors.append(max(np.abs(res.Jr[8] - uniform.Jr[0]).max(), np.abs(res.Jt[8] - uniform.Jt[0]).max()))
        self.assertLess(errors[1], 3e-4)
        self.assertTrue(0.4 < errors[1] / errors[0] < 0.6, errors)

    def test_field_map_shapes(self):
        res = self.solve(tensor_grating(), M=6, need_fields=True)
        fields = rt.field_map(res, np.linspace(-0.2, 0.7, 7), nx=32)
        self.assertEqual(fields["Ex"].shape, (7, 32))
        self.assertTrue(np.isfinite(fields["E2"]).all())


class InputsAndFailures(unittest.TestCase):
    def test_rayleigh_point_is_reported(self):
        with self.assertRaises(rc.RayleighAnomalyError):
            rt.solve_tensor(rt.TensorStack(1.0, [], 1.5, LAM), LAM, 0.0, 0.0, M=1)

    def test_incident_medium_must_be_lossless_isotropic(self):
        with self.assertRaises(NotImplementedError):
            rt.solve_tensor(rt.TensorStack(1.5 + 0.1j, [], 1.5, PERIOD), LAM, 10.0, 0.0, M=1)

    def test_singular_normal_block_is_rejected(self):
        layer = rt.TensorLayer(0.3, edges=[0, 0.3, 0.8], media=[rt.Medium(eps=[0.0, 2.0, 2.0]), 1.0])
        with self.assertRaises(ValueError):
            rt.solve_tensor(rt.TensorStack(1.0, [layer], 1.5, PERIOD), LAM, 10.0, 0.0, M=2)

    def test_material_helpers(self):
        self.assertEqual(rt.isotropic(1.5).scalar, (2.25 + 0j, 1 + 0j))
        self.assertIsNone(TILTED.scalar)
        self.assertTrue(TILTED.lossless and rt.gyrotropic(2.0, (0, 0, 0.2)).lossless)
        self.assertFalse(rt.isotropic(1.5 + 0.1j).lossless)
        turned = rt.uniaxial(1.6, 1.5, "x").rotated(rt.rotation_matrix("z", 90.0))
        self.assertLess(np.abs(turned.eps - rt.uniaxial(1.6, 1.5, "y").eps).max(), 1e-14)
        with self.assertRaises(ValueError):
            TILTED.rotated(np.diag([-1.0, 1, 1]))
        with self.assertRaises(ValueError):
            rt.TensorLayer(0.3, edges=[0, 0.5, 0.4], media=[1.0, 2.0])


# =============================================================================
#  Added after the independent review: the blind spots it listed, and regressions for the defects it confirmed
# =============================================================================
# COMSOL 6.4 finite-element channel efficiencies for the tilted-tooth case t3_grating_tilt (model and raw output not included)
# (cubic elements, 16 per local wavelength): tooth = biaxial crystal (1.5, 1.8, 2.1) rotated 37 deg about (1, 2, 3),
# tooth [0, 0.3] of a 0.8 um period, depth 0.3 um, air / glass 1.5, 1.053 um, theta 42 deg, phi -40 deg.
COMSOL_TILTED_TOOTH = {
    ("r", 0, "pp"): 0.007756557, ("r", 0, "ps"): 0.005964254, ("r", 0, "sp"): 0.001235904, ("r", 0, "ss"): 0.001479295,
    ("r", 1, "pp"): 0.019467050, ("r", 1, "ps"): 0.020957469, ("r", 1, "sp"): 0.039047499, ("r", 1, "ss"): 0.010500284,
    ("t", 0, "pp"): 0.699320223, ("t", 0, "ps"): 0.005737286, ("t", 0, "sp"): 0.006300905, ("t", 0, "ss"): 0.874334260,
    ("t", 1, "pp"): 0.013475964, ("t", 1, "ps"): 0.040357467, ("t", 1, "sp"): 0.213395898, ("t", 1, "ss"): 0.040669685,
}
# Demesy, Zolla, Nicolet, Vial, arXiv:1302.1032, Table 5.2 (finite elements; the sums are off by up to 8e-5).
# Aragonite trapezoid on silica; "TE" there means H along the grooves.
ARAGONITE_TABLE = {
    ("TE", 0):  {"T-1": 0.322510, "T0": 0.538165, "T1": 0.124722, "R0": 0.014683},
    ("TE", 20): {"T-1": 0.538727, "T0": 0.444403, "T1": 0.000369, "R-1": 0.005372, "R0": 0.011180},
    ("TE", 40): {"T-2": 0.012058, "T-1": 0.434191, "T0": 0.541090, "R-1": 0.005032, "R0": 0.007686},
    ("TM", 0):  {"T-1": 0.203133, "T0": 0.585235, "T1": 0.203138, "R0": 0.008473},
    ("TM", 20): {"T-1": 0.399719, "T0": 0.575625, "T1": 0.004643, "R-1": 0.004412, "R0": 0.015630},
    ("TM", 40): {"T-2": 0.025047, "T-1": 0.420714, "T0": 0.493491, "R-1": 0.002541, "R0": 0.058238},
}


def bianisotropic(seed, loss=0.0):
    """A deterministic general medium: Hermitian eps, mu, a full non-symmetric xi and zeta = xi^H, plus isotropic loss."""
    rng = np.random.default_rng(seed)

    def hermitian():
        a = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        return (a + a.conj().T) / 2

    eps, mu = 2.5 * np.eye(3) + 0.4 * hermitian(), 1.3 * np.eye(3) + 0.2 * hermitian()
    xi = 0.2 * (rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
    return rt.Medium(eps=eps + 1j * loss * np.eye(3), mu=mu + 1j * loss * np.eye(3), xi=xi, zeta=xi.conj().T)


def plane_waves(C, kx, ky):
    """The four plane waves exp[i k0 (-kx x - ky y + g z)] of a uniform medium and their six-vectors [E; H].

    From Maxwell's equations as a 6x6 pencil, (C + [[0, kX], [-kX, 0]]) f = 0 with k = (-kx, -ky, g); nothing
    is shared with the solver's elimination of Ez and Hz.  Returns (g down, f down, g up, f up).
    """
    def cross(k):
        return np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], complex)

    a0 = np.array(C, complex)
    a0[:3, 3:] += cross((-kx, -ky, 0.0))
    a0[3:, :3] -= cross((-kx, -ky, 0.0))
    a1 = np.zeros((6, 6), complex)
    a1[:3, 3:], a1[3:, :3] = cross((0, 0, 1.0)), -cross((0, 0, 1.0))
    g, f = generalized_eig(a0, -a1)
    keep = np.isfinite(g) & (np.abs(g) < 1e8)
    assert np.count_nonzero(keep) == 4
    g, f = g[keep], f[:, keep] / np.linalg.norm(f[:, keep], axis=0)
    flux = np.real(f[0] * f[4].conj() - f[1] * f[3].conj())
    down = np.where(np.abs(g.imag) > 1e-9 * (1 + np.abs(g)), g.imag > 0, flux > 0)
    assert np.count_nonzero(down) == 2
    return g[down], f[:, down], g[~down], f[:, ~down]


def slab_reference(media, thicknesses, n_in, exit_medium, theta_deg, phi_deg):
    """Reflected (z = 0) and transmitted (bottom) tangential [Ex, Ey, Hx, Hy] of a stack of uniform general
    layers, for the solver's unit s and unit p incidence: one linear system in plane-wave amplitudes, no recursion."""
    th, ph = np.radians(theta_deg), np.radians(phi_deg)
    kx, ky = n_in * np.sin(th) * np.cos(ph), n_in * np.sin(th) * np.sin(ph)
    tangential = [0, 1, 3, 4]
    _, f_in_down, _, f_in_up = plane_waves(rt.isotropic(n_in).C, kx, ky)
    _, f_out_down, _, _ = plane_waves(exit_medium.C, kx, ky)
    layers = [plane_waves(m.C, kx, ky) for m in media]
    varphi = ph if np.hypot(kx, ky) <= 1e-14 else np.arctan2(ky, kx)
    c, s = np.cos(varphi), np.sin(varphi)
    cos_theta = np.sqrt(complex(n_in ** 2 - kx ** 2 - ky ** 2)) / n_in
    out = []
    for ex, ey in ((-s, c), (c * cos_theta, s * cos_theta)):
        incident = f_in_down @ np.linalg.solve(f_in_down[[0, 1]], np.array([ex, ey], complex))
        n_layers = len(media)
        system = np.zeros((4 * (n_layers + 1), 4 + 4 * n_layers), complex)
        rhs = np.zeros(4 * (n_layers + 1), complex)
        system[:4, :2], rhs[:4] = f_in_up[tangential], -incident[tangential]
        row, col = 0, 2
        for (gd, fd, gu, fu), d in zip(layers, thicknesses):
            g, f = np.r_[gd, gu], np.hstack((fd, fu))
            system[row:row + 4, col:col + 4] = -f[tangential]
            row += 4
            system[row:row + 4, col:col + 4] = f[tangential] * np.exp(1j * K0 * g * d)[None, :]
            col += 4
        system[row:row + 4, col:col + 2] = -f_out_down[tangential]
        solution = np.linalg.solve(system, rhs)
        out.append(((f_in_up @ solution[:2])[tangential], (f_out_down @ solution[-2:])[tangential]))
    return out


def order_rows(res, m=0):
    n = len(res.ms)
    return res.i0 + m + n * np.arange(4)


class OtherSolversAgain(unittest.TestCase):
    def test_dense_incident_medium_negative_angle_and_five_layers(self):
        films = [(0.13, 1.96), (0.18, 1.46), (0.13, 1.96), (0.18, 1.46)]
        for name, n_in, n_out, theta in (("oblique from n = 1.4", 1.4, 1.5, -25.0), ("total internal reflection", 1.5, 1.0, 50.0)):
            with self.subTest(name=name):
                ref_layers = [rc.Layer(0.3, edges=EDGES, n_values=[1.8, 1.4, 1.0])] + [rc.Layer(d, n=n) for d, n in films]
                layers = [rt.TensorLayer(0.3, edges=EDGES, media=[1.8, 1.4, 1.0])] + [rt.TensorLayer(d, medium=n) for d, n in films]
                ref = solve_conical(rc.Stack(n_in, ref_layers, n_out, PERIOD), LAM, theta, 35.0, M=10, psi_deg=33, delta_deg=20)
                res = rt.solve_tensor(rt.TensorStack(n_in, layers, n_out, PERIOD), LAM, theta, 35.0, M=10, psi_deg=33, delta_deg=20)
                for key in ("Jr", "Jt", "DEr", "DEt", "kzI_over_k0", "kzII_over_k0", "kx_over_k0"):
                    self.assertLess(np.abs(res[key] - ref[key]).max(), TIGHT, key)
                self.assertLess(np.abs(channels(res) - channels(ref)).max(), TIGHT)
                self.assertTrue(np.array_equal(res.prop_t, ref.prop_t) and np.array_equal(res.prop_r, ref.prop_r))

    def test_magnetic_film_and_magnetic_exit_match_magnetic_conical_solver(self):
        eps, mu = [3.2, 2.0, 1.0], [1.5, 1.2, 1.0]
        eps_out, mu_out = 2.0, 1.4
        ref_stack = mag1d.Stack(1.0, [mag1d.Layer(0.3, edges=EDGES, eps_values=eps, mu_values=mu), mag1d.Layer(0.2, eps=2.6, mu=1.7)],
                                np.sqrt(eps_out * mu_out), PERIOD, muII=mu_out)
        stack = rt.TensorStack(1.0, [rt.TensorLayer(0.3, edges=EDGES, media=[rt.Medium(e, m) for e, m in zip(eps, mu)]),
                                     rt.TensorLayer(0.2, medium=rt.isotropic(np.sqrt(2.6 * 1.7), mu=1.7))],
                               rt.Medium(eps_out, mu_out), PERIOD)
        ref = solve_conical_magnetic(ref_stack, LAM, 42.0, -40.0, M=8, psi_deg=33, delta_deg=20)
        res = rt.solve_tensor(stack, LAM, 42.0, -40.0, M=8, psi_deg=33, delta_deg=20)
        for key in ("Jr", "Jt", "DEr", "DEt"):
            self.assertLess(np.abs(res[key] - ref[key]).max(), TIGHT, key)
        self.assertLess(np.abs(channels(res) - channels(ref)).max(), TIGHT)
        self.assertLess(abs(res.energy - 1), 1e-11)

    def test_absorbing_exit_matches_planar_solver(self):
        n_out = 1.5 + 0.2j
        ref_stack = rc.Stack(1.0, [rc.Layer(0.3, edges=EDGES, n_values=[1.8, 1.4, 1.0])], n_out, PERIOD)
        stack = rt.TensorStack(1.0, [rt.TensorLayer(0.3, edges=EDGES, media=[1.8, 1.4, 1.0])], n_out, PERIOD)
        for pol, psi in (("TE", 90.0), ("TM", 0.0)):
            with self.subTest(pol=pol):
                ref = rc.solve(ref_stack, LAM, 42.0, 12, pol, need_fields=False)
                res = rt.solve_tensor(stack, LAM, 42.0, 0.0, M=12, psi_deg=psi)
                self.assertLess(np.abs(res.DEr - ref.DEr).max(), TIGHT)
                self.assertLess(np.abs(res.DEt - ref.DEt).max(), TIGHT)
                self.assertFalse(res.prop_t.any())              # nothing propagates undamped in an absorbing medium
                self.assertNotIn("DEt_ss", res)                 # and the lossless channel formula is not offered
                self.assertGreater(res.DEt.sum(), 0.5)


class BackwardWavesAndCutoffs(unittest.TestCase):
    def test_double_negative_medium(self):
        # eps and mu both negative: the outgoing wave has a negative propagation constant; |r| equals that of n = 1.5
        ordinary = rt.solve_tensor(rt.TensorStack(1.0, [], 1.5, PERIOD), LAM, 30.0, 20.0, M=0, psi_deg=33, delta_deg=20)
        negative = rt.solve_tensor(rt.TensorStack(1.0, [], rt.Medium(eps=-2.25, mu=-1.0), PERIOD), LAM, 30.0, 20.0, M=0, psi_deg=33, delta_deg=20)
        self.assertLess(abs(negative.DEr[0] - ordinary.DEr[0]), TIGHT)
        self.assertLess(abs(negative.DEt[0] - ordinary.DEt[0]), TIGHT)
        self.assertGreater(negative.DEt[0], 0.5)
        self.assertLess(negative.kzII_over_k0[0].real, 0)
        # matched slab eps = mu = -1 in vacuum: no reflection, unit transmission, negative phase advance
        d = 0.4
        slab = rt.solve_tensor(rt.TensorStack(1.0, [rt.TensorLayer(d, medium=rt.Medium(eps=-1.0, mu=-1.0))], 1.0, PERIOD),
                               LAM, 30.0, 20.0, M=0)
        self.assertLess(slab.DEr[0], 1e-20)
        self.assertLess(abs(slab.Jt[0][0, 0] - np.exp(-1j * K0 * np.cos(np.radians(30.0)) * d)), TIGHT)
        self.assertLess(abs(slab.Jt[0][1, 1] - np.exp(-1j * K0 * np.cos(np.radians(30.0)) * d)), TIGHT)
        # with evanescent orders present each interface carries a lossless surface mode: reported, not garbage
        with self.assertRaises(np.linalg.LinAlgError):
            rt.solve_tensor(rt.TensorStack(1.0, [rt.TensorLayer(d, medium=rt.Medium(eps=-1.0, mu=-1.0))], 1.0, PERIOD), LAM, 30.0, 20.0, M=2)

    def test_backward_extraordinary_wave_in_a_hyperbolic_medium(self):
        # eps = diag(-3, -3, 1): for kt > 1 the only propagating wave carries energy against its phase velocity
        def run(loss, turn):
            medium = rt.Medium(eps=np.diag([-3.0, -3.0, 1.0]) + 1j * loss * np.eye(3)).rotated(rt.rotation_matrix((1, 0.3, 0), turn))
            exit_only = rt.solve_tensor(rt.TensorStack(1.5, [], medium, PERIOD), LAM, 60.0, 25.0, M=0, psi_deg=33, delta_deg=20)
            film = rt.solve_tensor(rt.TensorStack(1.5, [rt.TensorLayer(0.2, medium=medium)], 1.5, PERIOD), LAM, 60.0, 25.0, M=0, psi_deg=33, delta_deg=20)
            return exit_only, film

        for turn in (0.0, 12.0):
            with self.subTest(turn=turn):
                (exit_only, film), (exit_lossy, film_lossy) = run(0.0, turn), run(1e-7, turn)
                self.assertLess(abs(exit_only.energy - 1), 1e-11)
                self.assertGreater(exit_only.DEt[0], 0.05)
                self.assertLess(abs(exit_only.DEr[0] - exit_lossy.DEr[0]), 1e-5)
                self.assertLess(abs(film.energy - 1), 1e-11)
                self.assertLess(abs(film.DEr[0] - film_lossy.DEr[0]), 1e-5)

    def test_degenerate_forward_and_backward_waves_follow_the_small_loss_limit(self):
        # the same physical problem for every turn angle; at this direction a forward and a backward wave of the
        # exit medium share one propagation constant, so the eigenvectors alone do not separate them
        values = []
        for turn in (0.0, 10.0, 45.0, 123.4):
            medium = rt.Medium(eps=[-3.363636363636, 4, 1]).rotated(rt.rotation_matrix("z", turn))
            res = rt.solve_tensor(rt.TensorStack(1.5, [], medium, PERIOD), LAM, 60.0, turn, M=0, psi_deg=40, delta_deg=30)
            values.append(res.DEr[0])
            self.assertLess(abs(res.energy - 1), 1e-11)
        self.assertLess(np.ptp(values), 1e-9)
        lossy = rt.Medium(eps=np.diag([-3.363636363636, 4, 1]) + 1e-7j * np.eye(3)).rotated(rt.rotation_matrix("z", 45.0))
        limit = rt.solve_tensor(rt.TensorStack(1.5, [], lossy, PERIOD), LAM, 60.0, 45.0, M=0, psi_deg=40, delta_deg=30)
        self.assertLess(abs(values[2] - limit.DEr[0]), 1e-6)

    def test_isotropic_film_at_an_order_cutoff_in_a_conical_mount(self):
        # n = 1.3 film between n = 1.5 at the critical angle: the film's wave is at cutoff.  Characteristic matrix.
        def exact(theta):
            kt = 1.5 * np.sin(np.radians(theta))
            kz = [np.sqrt(complex(n * n - kt * kt)) for n in (1.5, 1.3, 1.5)]
            delta = K0 * kz[1] * 0.2
            sinc = np.sinc(delta / np.pi) * K0 * 0.2
            b, c = np.cos(delta) - 1j * sinc * kz[2], -1j * kz[1] ** 2 * sinc + np.cos(delta) * kz[2]
            return abs((kz[0] * b - c) / (kz[0] * b + c)) ** 2

        critical = np.degrees(np.arcsin(1.3 / 1.5))
        solved = 0
        for offset in (0.0, 1e-12, 1e-9, 1e-6):
            for phi in (0.0, 45.0):
                with self.subTest(offset=offset, phi=phi):
                    try:
                        res = rt.solve_tensor(rt.TensorStack(1.5, [rt.TensorLayer(0.2, medium=1.3)], 1.5, PERIOD),
                                              LAM, critical + offset, phi, M=0, psi_deg=90.0)
                    except rc.RayleighAnomalyError:          # the propagation constant came out as exactly zero
                        continue
                    solved += 1
                    self.assertLess(abs(res.DEr[0] - exact(critical + offset)), 1e-8)
        self.assertGreaterEqual(solved, 6)

    def test_exact_cutoff_in_an_anisotropic_exit_medium_is_reported(self):
        # normal incidence, lambda / period = 1.5 exactly: the first orders graze in a crystal of ordinary index 1.5
        with self.assertRaises(rc.RayleighAnomalyError):
            rt.solve_tensor(rt.TensorStack(1.0, [], rt.uniaxial(1.5, 1.7, "z"), 1.0), 1.5, 0.0, 0.0, M=1)


class Magnetoelectric(unittest.TestCase):
    def test_bianisotropic_stacks_match_plane_wave_solution(self):
        media, thicknesses = [bianisotropic(1), bianisotropic(2, loss=0.05)], [0.21, 0.33]
        for name, exit_medium in (("glass exit", rt.isotropic(1.5)), ("bianisotropic exit", bianisotropic(3))):
            for theta, phi in ((35.0, -50.0), (0.0, 20.0)):
                with self.subTest(name=name, theta=theta):
                    stack = rt.TensorStack(1.4, [rt.TensorLayer(d, medium=m) for m, d in zip(media, thicknesses)], exit_medium, PERIOD)
                    ref = slab_reference(media, thicknesses, 1.4, exit_medium, theta, phi)
                    for dense in (False, True):
                        res = rt.solve_tensor(stack, LAM, theta, phi, M=1, _force_dense=dense)
                        rows = order_rows(res)
                        for col in range(2):
                            self.assertLess(np.abs(res.psi_ref[rows, col] - ref[col][0]).max(), TIGHT)
                            self.assertLess(np.abs(res.psi_tra[rows, col] - ref[col][1]).max(), TIGHT)

    def test_duality_rotation_of_a_magnetoelectric_grating(self):
        # (E, H) -> (E cos a + H sin a, -E sin a + H cos a) with (D, B) likewise leaves Maxwell's equations unchanged.
        # A grating with eps and mu tensors becomes one with full xi and zeta; vacuum and a matched exit (eps = mu) stay
        # as they are; the incident polarization turns by a.  The two problems must give rotated copies of one field.
        a = np.radians(27.0)
        c, s = np.cos(a), np.sin(a)
        turn6 = np.block([[c * np.eye(3), s * np.eye(3)], [-s * np.eye(3), c * np.eye(3)]])
        turn4 = lambda psi: np.concatenate((c * psi[:len(psi) // 2] + s * psi[len(psi) // 2:], -s * psi[:len(psi) // 2] + c * psi[len(psi) // 2:]))

        def turned(medium):
            out = rt.Medium()
            out.C = turn6 @ rt.as_medium(medium).C @ turn6.T
            return out

        tooth = rt.Medium(eps=TILTED.eps, mu=rt.biaxial(1.0, 1.2, 1.4, rt.rotation_matrix((2, -1, 1), 50.0)).eps)
        media = [tooth, rt.isotropic(1.4), rt.isotropic(1.0)]
        film = rt.Medium(eps=FILM.eps)
        matched = rt.Medium(eps=1.5, mu=1.5)
        plain = rt.TensorStack(1.0, [rt.TensorLayer(0.3, edges=EDGES, media=media), rt.TensorLayer(0.15, medium=film)], matched, PERIOD)
        other = rt.TensorStack(1.0, [rt.TensorLayer(0.3, edges=EDGES, media=[turned(m) for m in media]),
                                     rt.TensorLayer(0.15, medium=turned(film))], matched, PERIOD)
        self.assertGreater(np.abs(turned(tooth).xi).max(), 0.1)          # the turned tooth really is magnetoelectric
        jones = np.array([0.6 * np.exp(0.4j), 0.8])
        res = rt.solve_tensor(plain, LAM, 42.0, -40.0, M=10, jones_in=jones)
        dual = rt.solve_tensor(other, LAM, 42.0, -40.0, M=10, jones_in=np.array([[c, s], [-s, c]]) @ jones)
        self.assertLess(np.abs(res.DEr - dual.DEr).max(), TIGHT)
        self.assertLess(np.abs(res.DEt - dual.DEt).max(), TIGHT)
        self.assertLess(np.abs(turn4(res.psi_ref @ res.jones_in) - dual.psi_ref @ dual.jones_in).max(), TIGHT)
        self.assertLess(np.abs(turn4(res.psi_tra @ res.jones_in) - dual.psi_tra @ dual.jones_in).max(), TIGHT)

    def test_rotation_about_the_normal_with_magnetoelectric_blocks(self):
        medium = bianisotropic(4, loss=0.02)
        base = rt.solve_tensor(rt.TensorStack(1.0, [rt.TensorLayer(0.37, medium=medium)], 1.5, PERIOD), LAM, 42.0, -40.0, M=1)
        turned = medium.rotated(rt.rotation_matrix("z", 71.0))
        res = rt.solve_tensor(rt.TensorStack(1.0, [rt.TensorLayer(0.37, medium=turned)], 1.5, PERIOD), LAM, 42.0, -40.0 + 71.0, M=1)
        self.assertLess(np.abs(res.Jr[1] - base.Jr[1]).max(), TIGHT)
        self.assertLess(np.abs(res.Jt[1] - base.Jt[1]).max(), TIGHT)


class Fields(unittest.TestCase):
    def test_reconstructed_fields_satisfy_maxwell_equations(self):
        # a grating with a bianisotropic and a crystal segment, on a bianisotropic film, on an isotropic film
        stack = rt.TensorStack(1.0, [rt.TensorLayer(0.3, edges=EDGES, media=[bianisotropic(1), TILTED, 1.0]),
                                     rt.TensorLayer(0.2, medium=bianisotropic(2)), rt.TensorLayer(0.15, medium=1.46)], 1.5, PERIOD)
        res = rt.solve_tensor(stack, LAM, 42.0, -40.0, M=5, psi_deg=33, delta_deg=20, need_fields=True)
        kx, ky, h = res.kx_over_k0, res.kappa, 1e-3

        def ddz(name, z):                     # d/d(k0 z), fourth-order central difference
            f = lambda zz: rt.field_at(res, zz)[name]
            return (-f(z + 2 * h) + 8 * f(z + h) - 8 * f(z - h) + f(z - 2 * h)) / (12 * h * K0)

        for z in (-0.2, 0.11, 0.4, 0.58, 0.8):
            f = rt.field_at(res, z)
            scale = max(np.abs(f[k]).max() for k in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"))
            residuals = {
                "curl E, x": -1j * ky * f["Ez"] - ddz("Ey", z) - 1j * f["Bx"],
                "curl E, y": ddz("Ex", z) + 1j * kx * f["Ez"] - 1j * f["By"],
                "curl E, z": -1j * kx * f["Ey"] + 1j * ky * f["Ex"] - 1j * f["Bz"],
                "curl H, x": -1j * ky * f["Hz"] - ddz("Hy", z) + 1j * f["Dx"],
                "curl H, y": ddz("Hx", z) + 1j * kx * f["Hz"] + 1j * f["Dy"],
                "curl H, z": -1j * kx * f["Hy"] + 1j * ky * f["Hx"] + 1j * f["Dz"],
            }
            for name, value in residuals.items():
                self.assertLess(np.abs(value).max(), 1e-6 * scale, f"{name} at z={z}")

    def test_field_map_matches_conical_field_map(self):
        layers = [rc.Layer(0.3, edges=EDGES, n_values=[1.8, 1.4, 1.0]), rc.Layer(0.13, n=1.96)]
        mine = [rt.TensorLayer(0.3, edges=EDGES, media=[1.8, 1.4, 1.0]), rt.TensorLayer(0.13, medium=1.96)]
        ref = solve_conical(rc.Stack(1.0, layers, 1.5, PERIOD), LAM, 42.0, -40.0, M=8, psi_deg=33, delta_deg=20, need_fields=True)
        res = rt.solve_tensor(rt.TensorStack(1.0, mine, 1.5, PERIOD), LAM, 42.0, -40.0, M=8, psi_deg=33, delta_deg=20, need_fields=True)
        zs = np.array([-0.15, 0.1, 0.35, 0.6])
        a, b = conical_field_map(ref, zs, nx=24, y=0.07), rt.field_map(res, zs, nx=24, y=0.07)
        for key in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            self.assertLess(np.abs(a[key] - b[key]).max(), TIGHT, key)
        self.assertLess(np.abs(b["E2"] - (np.abs(b["Ex"]) ** 2 + np.abs(b["Ey"]) ** 2 + np.abs(b["Ez"]) ** 2)).max(), 1e-13)
        # an explicit Jones vector overrides the one used for the solve
        p_only = rt.solve_tensor(rt.TensorStack(1.0, mine, 1.5, PERIOD), LAM, 42.0, -40.0, M=8, psi_deg=0.0, need_fields=True)
        for key in ("Ex", "Hy", "Ez"):
            self.assertLess(np.abs(rt.field_at(res, 0.1, jones_in=[0, 1])[key] - rt.field_at(p_only, 0.1)[key]).max(), TIGHT)

    def test_bare_interface_has_fields_at_the_interface(self):
        res = rt.solve_tensor(rt.TensorStack(1.0, [], 1.5, PERIOD), LAM, 30.0, 20.0, M=2, need_fields=True)
        above, at = rt.field_at(res, -1e-9), rt.field_at(res, 0.0)
        self.assertLess(np.abs(above["Ex"] - at["Ex"]).max(), 1e-6)
        self.assertEqual(rt.field_map(res, [-0.1, 0.0, 0.1], nx=8)["Ey"].shape, (3, 8))


class NumbersFromOutside(unittest.TestCase):
    def test_tilted_crystal_tooth_matches_comsol(self):
        tooth = rt.biaxial(1.5, 1.8, 2.1, rt.rotation_matrix((1, 2, 3), 37.0))
        stack = rt.TensorStack(1.0, [rt.TensorLayer(0.3, edges=[0.0, 0.3, 0.8], media=[tooth, 1.0])], 1.5, PERIOD)
        res = rt.solve_tensor(stack, LAM, 42.0, -40.0, M=30)
        worst = 0.0
        for (side, m, channel), value in COMSOL_TILTED_TOOTH.items():
            ours = res["DE%s_%s" % (side, channel)][res.i0 + m]
            worst = max(worst, abs(ours - value))
        self.assertLess(worst, 1e-5)          # truncation at M = 30 is about 5e-6, the finite elements about 1e-6

    def test_aragonite_grating_matches_published_table(self):
        # trapezoid base 500 nm, top 300 nm, height 600 nm, period 600 nm, 633 nm; staircase of 16 slices
        ridge = rt.Medium(eps=[[2.592, 0.0, 0.251], [0.0, 2.829, 0.0], [0.251, 0.0, 2.592]])
        layers = []
        for j in range(16):
            w = 0.3 + 0.2 * (j + 0.5) / 16
            layers.append(rt.TensorLayer(0.6 / 16, edges=[0.0, (0.6 - w) / 2, (0.6 + w) / 2, 0.6], media=[1.0, ridge, 1.0]))
        stack = rt.TensorStack(1.0, layers, 1.5, 0.6)
        worst = {"TE": 0.0, "TM": 0.0}
        for (pol, theta), row in ARAGONITE_TABLE.items():
            res = rt.solve_tensor(stack, 0.633, float(theta), 180.0, M=10, psi_deg=0.0 if pol == "TE" else 90.0)
            ours = {"R%d" % m: res.DEr[i] for i, m in enumerate(res.ms) if res.prop_r[i]}
            ours.update({"T%d" % m: res.DEt[i] for i, m in enumerate(res.ms) if res.prop_t[i]})
            self.assertEqual(set(ours), set(row))
            worst[pol] = max(worst[pol], max(abs(ours[k] - v) for k, v in row.items()))
        # at this coarse discretization the differences are 2.6e-4 and 1.9e-3; they shrink with more slices and
        # orders (5e-4 for the second at 64 slices, M = 25).  The mirrored tilt gives 0.2.
        self.assertLess(worst["TM"], 5e-4)          # E along the grooves
        self.assertLess(worst["TE"], 3e-3)          # H along the grooves: the rows that see the off-diagonal element


class MoreInputsAndHelpers(unittest.TestCase):
    def test_material_constructors(self):
        R = rt.rotation_matrix((1, 2, 3), 37.0)
        crystal = rt.biaxial(1.5, 1.8, 2.1, R)
        for i, n in enumerate((1.5, 1.8, 2.1)):                     # the columns of R are the principal axes
            self.assertLess(np.abs(crystal.eps @ R[:, i] - n ** 2 * R[:, i]).max(), 1e-13)
        self.assertLess(np.abs(rt.uniaxial(1.6, 1.5, (2, 0, 0)).eps - rt.uniaxial(1.6, 1.5, "x").eps).max(), 1e-15)
        tensor_eps = np.diag([2.0, 2.2, 2.4]) + 0.1 * (np.ones((3, 3)) - np.eye(3))
        gyro = rt.gyrotropic(tensor_eps, (0.1, 0.2, 0.3)).eps
        self.assertLess(np.abs(0.5 * (gyro + gyro.T) - tensor_eps).max(), 1e-15)
        self.assertLess(abs((gyro[0, 1] - gyro[1, 0]) / 2 - (-0.3j)), 1e-15)
        already_gyrotropic = tensor_eps + 0.05j * np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
        self.assertTrue(np.array_equal(rt.gyrotropic(already_gyrotropic, (0, 0, 0)).eps, already_gyrotropic))
        self.assertTrue(np.array_equal(rt.Medium(xi=[1, 2, 3]).xi, np.diag([1, 2, 3])))
        zeta = np.arange(9.0).reshape(3, 3)
        self.assertTrue(np.array_equal(rt.Medium(zeta=zeta).zeta, zeta))
        magnetic = rt.isotropic(1.5, mu=2.0)
        self.assertEqual(magnetic.scalar, (1.125 + 0j, 2 + 0j))
        self.assertTrue(TILTED.passive and rt.isotropic(1.5 + 0.1j).passive)
        self.assertFalse(rt.isotropic(1.5 - 0.1j).passive)
        self.assertFalse(rt.Medium(eps=2.5, xi=0.3j).passive)       # xi without zeta = xi^H is an active medium

    def test_invalid_input_is_rejected(self):
        air_glass = rt.TensorStack(1.0, [], 1.5, PERIOD)
        with self.assertRaises(ValueError):
            rt.solve_tensor(air_glass, LAM, 90.0, 0.0, M=1)
        with self.assertRaises(ValueError):
            rt.solve_tensor(air_glass, LAM, 95.0, 0.0, M=1)
        with self.assertRaises(ValueError):
            rt.solve_tensor(rt.TensorStack(1.0, [], 1.5, 0.0), LAM, 10.0, 0.0, M=1)
        with self.assertRaises(ValueError):                           # edges that do not span one period
            rt.solve_tensor(rt.TensorStack(1.0, [rt.TensorLayer(0.3, edges=[0, 0.3, 0.7], media=[1.8, 1.0])], 1.5, PERIOD), LAM, 10.0, 0.0, M=2)
        with self.assertRaises(ValueError):                           # gain in the exit medium
            rt.solve_tensor(rt.TensorStack(1.0, [], 1.5 - 0.1j, PERIOD), LAM, 10.0, 0.0, M=1)
        with warnings.catch_warnings(record=True) as caught:          # gain in a layer is solved, with a warning
            warnings.simplefilter("always")
            rt.solve_tensor(rt.TensorStack(1.0, [rt.TensorLayer(0.3, medium=rt.Medium(eps=2.5, xi=0.3j))], 1.5, PERIOD), LAM, 10.0, 0.0, M=1)
        self.assertTrue(any(issubclass(w.category, RuntimeWarning) for w in caught))

    def test_exit_crystal_that_passes_one_polarization_only(self):
        # optic axis along the grooves, planar incidence: E along the grooves sees n = 1.9 and is transmitted,
        # the other polarization sees n = 1.2 and is totally reflected.  The order still counts as propagating.
        stack = rt.TensorStack(1.5, [], rt.uniaxial(1.2, 1.9, "y"), PERIOD)
        s_in = rt.solve_tensor(stack, LAM, 65.0, 0.0, M=0, psi_deg=90.0, need_fields=True)
        p_in = rt.solve_tensor(stack, LAM, 65.0, 0.0, M=0, psi_deg=0.0)
        self.assertTrue(s_in.prop_t[0])
        self.assertGreater(s_in.DEt[0], 0.5)
        self.assertLess(abs(p_in.DEr[0] - 1), 1e-12)
        self.assertLess(abs(p_in.DEt[0]), 1e-12)
        # T_xy is the tangential electric field just below the last interface, whatever basis the exit modes use
        below = rt.field_at(s_in, 1e-12)
        self.assertLess(abs(s_in.T_xy[0] @ s_in.jones_in - below["Ex"][0]), 1e-9)
        self.assertLess(abs(s_in.T_xy[1] @ s_in.jones_in - below["Ey"][0]), 1e-9)
        self.assertGreater(abs(below["Ey"][0]), 0.5)

    def test_anisotropic_exit_medium_with_many_orders(self):
        # strongly evanescent orders in an anisotropic exit medium must still be sorted by their decay
        grating = rt.TensorLayer(0.3, edges=EDGES, media=[TILTED, BIAXIAL, 1.0])
        bare = rt.solve_tensor(rt.TensorStack(1.0, [grating], FILM, PERIOD), LAM, 42.0, -40.0, M=45)
        padded = rt.solve_tensor(rt.TensorStack(1.0, [grating, rt.TensorLayer(0.77, medium=FILM)], FILM, PERIOD), LAM, 42.0, -40.0, M=45)
        self.assertLess(abs(bare.energy - 1), 1e-10)
        self.assertLess(np.abs(bare.DEr - padded.DEr).max(), 1e-9)
        self.assertLess(np.abs(bare.DEt[~bare.prop_t]).max(), 1e-12)     # a lossless exit carries power in propagating orders only
        self.assertGreater(bare.DEt[bare.prop_t].sum(), 0.5)


if __name__ == "__main__":
    unittest.main()
