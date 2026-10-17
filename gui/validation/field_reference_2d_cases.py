"""Explicit inputs for the frozen v1 two-dimensional (crossed grating) RCWA corpus.

Cases follow CONTRACT_2D.md section 7. Everything is deterministic: no random
sampling and no scans. This module samples the *current production* solver
``gui/2drcwa/rcwa2d.py`` (the numerical authority of the port); it is not an
independent oracle. Independent checks (Airy, rcwa1d / conical limits, energy,
tangential continuity) live in ``test_field_contract_2d.py``.

Layer notation (JSON friendly; a refractive index is a float or ``[re, im]``):

    {"d": 0.3, "n": 1.8}                                    uniform layer
    {"d": 0.3, "y_edges": [...], "x_profiles": [{"edges": [...], "n_values": [...]}, ...]}
                                                            patterned layer (y segments x x segments)
"""
from copy import deepcopy
from pathlib import Path
import sys

GUI = Path(__file__).resolve().parents[1]
ROOT = GUI.parent
for path in (GUI, GUI / "2drcwa"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import rcwa2d as rc2
from field_phase import phase_data

COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
FROZEN_DIR = Path(__file__).resolve().parent / "field_reference_2d"
CONTRACT_VERSION = 1
# Layout / dtype of the Result2D arrays stored for every successful case.
FAR_ARRAYS = ("Kx", "Ky", "kzI", "kzII", "Ex_r", "Ey_r", "Ez_r", "Ex_t", "Ey_t", "Ez_t",
              "Rs", "Rp", "Ts", "Tp", "DEr", "DEt")
NUMERIC_SOURCES = (
    "gui/2drcwa/rcwa2d.py", "gui/field_phase.py", "gui/gui_fields.py",
)
# Operators Eps, Cx, Cy of structured layers are stored only for small cases (N = (2M1+1)(2M2+1) <= this limit).
OPERATOR_LIMIT_N = 49
LAMBDA_PILLAR = 1.053


# ---------------------------------------------------------------- geometry helpers
def uniform(d, n):
    return dict(d=d, n=n)


def patterned(d, y_edges, rows):
    """rows: list of (x_edges, n_values), one per y segment."""
    return dict(d=d, y_edges=list(y_edges),
                x_profiles=[dict(edges=list(e), n_values=list(v)) for e, v in rows])


def pillar(Lx, Ly, wx, wy, cx, cy, n, d, bg=1.0):
    """Rectangular pillar of index ``n`` in ``bg`` (the COMSOL 3-D geometry of台阶 3/4)."""
    x0, x1 = round(cx - wx / 2, 12), round(cx + wx / 2, 12)
    y0, y1 = round(cy - wy / 2, 12), round(cy + wy / 2, 12)
    return patterned(d, [0.0, y0, y1, Ly], [([0.0, Lx], [bg]), ([0.0, x0, x1, Lx], [bg, n, bg]),
                                            ([0.0, Lx], [bg])])


def jones(psi_deg=None, delta_deg=0.0, exact=None):
    """Explicit Jones [Es, Ep] as [[re, im], [re, im]]. ``exact`` is "s" or "p" (exact zeros)."""
    if exact == "s":
        return [[1.0, 0.0], [0.0, 0.0]]
    if exact == "p":
        return [[0.0, 0.0], [1.0, 0.0]]
    psi, delta = np.radians(psi_deg), np.radians(delta_deg)
    es = np.sin(psi) * np.exp(1j * delta)
    return [[float(es.real), float(es.imag)], [float(np.cos(psi)), 0.0]]


def z_samples(thicknesses, outside=(.43, .07), after=(.09, .41), gap=1e-6):
    """Sorted explicit depths: region I, both sides of every interface (|dz| = 1e-6, well outside the
    1e-12 ownership band), two interior points per layer, region II. Never z == 0 exactly."""
    zs, top = [-outside[0], -outside[1]], 0.0
    bounds = [0.0]
    for d in thicknesses:
        bounds.append(top + d)
        zs += [top + 0.3 * d, top + 0.7 * d]
        top += d
    for b in bounds:
        zs += [b - gap, b + gap]
    zs += [top + after[0], top + after[1]]
    return sorted(round(float(z), 12) for z in set(round(z, 12) for z in zs))


# ---------------------------------------------------------------- tolerances
# far / near: cross-backend (C++ vs frozen Python) comparison |a-b| <= atol + rtol*|b| (CONTRACT_2D section 6).
# reproduce: same-environment recomputation by this Python solver (test_field_contract_2d).
# near_independent: 10 x the noise floor of the COMSOL 3-D FEM near-field comparison (report not included in this repository:
#   Ez / H rms 1.6e-3 -> 1.6e-2; Ex, Ey at >= 0.15 um from an edge 6.9e-3 -> 7e-2), absolute, field units |E_inc| = 1.
#   For comparing against an independent solver only, away from material edges; NOT used by the regression tests.
def tolerance(N, far=None, near=None, reproduce=None):
    big = N > 289
    far = far or dict(atol=1e-8 if big else 1e-9, rtol=1e-8 if big else 1e-9)
    near = near or dict(atol=1e-8 if big else 1e-9, rtol=1e-8 if big else 1e-9)
    reproduce = reproduce or dict(atol=5e-11 if big else 5e-12, rtol=5e-11 if big else 5e-12)
    return dict(far=far, near=near, reproduce=reproduce,
                near_independent=dict(atol=1.6e-2, atol_Ex_Ey=7e-2, rtol=0.0))


def case_specs():
    """JSON-serializable parameters of every case."""
    cases = []

    def add(case_id, item, description, *, nI=1.0, nII=1.5, Lx=.8, Ly=.8, layers, wavelength=LAMBDA_PILLAR,
            theta, phi, M1, M2, jones_in, psi_deg=None, delta_deg=None, factorization="li",
            slice_xz_y=.37, slice_yz_x=.46, nt=15, zero=(), far_zero=(), oracle="", uniform_stack=False,
            status="ok", expect=None, zs=None, near=None, far=None):
        N = (2 * M1 + 1) * (2 * M2 + 1)
        spec = dict(id=case_id, contract_item=item, description=description,
                    stack=dict(nI=nI, nII=nII, Lx=Lx, Ly=Ly, layers=deepcopy(layers)),
                    wavelength=wavelength, theta_deg=theta, phi_deg=phi, M1=M1, M2=M2, N=N,
                    jones=jones_in, psi_deg=psi_deg, delta_deg=delta_deg, factorization=factorization,
                    uniform_stack=uniform_stack, status=status, expect=expect,
                    zs=zs if zs is not None else z_samples([layer["d"] for layer in layers]),
                    slice=dict(nt=nt, xz_fixed_y=slice_xz_y, yz_fixed_x=slice_yz_x),
                    independent_validation=oracle,
                    structural_zero_components=list(zero), far_structural_zero_arrays=list(far_zero),
                    save_operators=bool(status == "ok" and N <= OPERATOR_LIMIT_N and
                                        any("n" not in layer for layer in layers)),
                    tolerance=tolerance(N, far=far, near=near))
        cases.append(spec)

    # ---- 1. no layer / uniform film at theta = 42, phi = -25
    add("interface", "1", "single interface, no layers, generic linear Jones",
        nII=1.5, layers=[], theta=42., phi=-25., M1=2, M2=2, jones_in=jones(40.), psi_deg=40., delta_deg=0.,
        uniform_stack=True, oracle="Fresnel (Airy with zero layers)")
    add("film", "1", "lossless uniform film n=1.8 d=0.3 on n=1.5", layers=[uniform(.3, 1.8)],
        theta=42., phi=-25., M1=2, M2=2, jones_in=jones(40.), psi_deg=40., delta_deg=0.,
        uniform_stack=True, oracle="Airy characteristic matrix, s and p")

    # ---- 2. y-invariant (T1, M2 = 0) and x-invariant (T1', M1 = 0) gratings
    P0, F = .574, .30
    xe = [-F * P0 / 2, F * P0 / 2, P0 - F * P0 / 2]
    ye_inv = [0.0, .8]
    y_inv = [patterned(.38, ye_inv, [(xe, [1.46, 1.0])])]
    x_inv = [patterned(.38, xe, [([0.0, .8], [1.46]), ([0.0, .8], [1.0])])]
    y_args = dict(nI=1.0, nII=1.46, Lx=P0, Ly=.8, layers=y_inv, wavelength=1.053, theta=66.36)
    x_args = dict(nI=1.0, nII=1.46, Lx=.8, Ly=P0, layers=x_inv, wavelength=1.053, theta=66.36)
    add("y_inv_te", "2", "y-invariant grating, phi=0, s (TE) = rcwa1d TE; selftest2d T1",
        phi=0., M1=12, M2=0, jones_in=jones(exact="s"), slice_xz_y=.31, slice_yz_x=.21,
        zero=("Ex", "Ez", "Hy"), far_zero=("Ex_r", "Ez_r", "Ex_t", "Ez_t", "Rp", "Tp"),
        oracle="rcwa1d.solve TE and conical.solve_conical (complex amplitudes)", **y_args)
    add("y_inv_tm", "2", "y-invariant grating, phi=0, p (TM) = rcwa1d TM; selftest2d T1",
        phi=0., M1=12, M2=0, jones_in=jones(exact="p"), slice_xz_y=.31, slice_yz_x=.21,
        zero=("Ey", "Hx", "Hz"), far_zero=("Ey_r", "Ey_t", "Rs", "Ts"),
        oracle="rcwa1d.solve TM and conical.solve_conical (complex amplitudes)", **y_args)
    add("y_inv_conical", "2", "y-invariant grating, phi=30, elliptical Jones (conical incidence)",
        phi=30., M1=8, M2=0, jones_in=jones(33., 50.), psi_deg=33., delta_deg=50.,
        slice_xz_y=.31, slice_yz_x=.21,
        oracle="conical.solve_conical Jones matrices and six-component field_at", **y_args)
    add("x_inv_te", "2", "x-invariant grating, phi=90, s (TE) = rcwa1d TE with swapped axes; selftest2d T1'",
        phi=90., M1=0, M2=12, jones_in=jones(exact="s"), slice_xz_y=.21, slice_yz_x=.31,
        zero=("Ey", "Ez", "Hx"), far_zero=("Ey_r", "Ez_r", "Ey_t", "Ez_t", "Rp", "Tp"),
        oracle="rcwa1d.solve TE (period P0 along y)", **x_args)
    add("x_inv_tm", "2", "x-invariant grating, phi=90, p (TM) = rcwa1d TM with swapped axes",
        phi=90., M1=0, M2=12, jones_in=jones(exact="p"), slice_xz_y=.21, slice_yz_x=.31,
        zero=("Ex", "Hy", "Hz"), far_zero=("Ex_r", "Ex_t", "Rs", "Ts"),
        oracle="rcwa1d.solve TM (period P0 along y)", **x_args)
    add("x_inv_conical", "2", "x-invariant grating, phi=50, elliptical Jones (maps to conical phi-90)",
        phi=50., M1=0, M2=8, jones_in=jones(33., 50.), psi_deg=33., delta_deg=50.,
        slice_xz_y=.21, slice_yz_x=.31,
        oracle="conical.solve_conical with x_1D=y, y_1D=-x, phi_1D=phi-90 (COMSOL step 2y mapping)", **x_args)

    # ---- 3. square pillar (0.3 x 0.3, n=1.8, d=0.3 in a 0.8 x 0.8 cell) and eccentric 0.3 x 0.2 pillar
    sq = [pillar(.8, .8, .3, .3, .4, .4, 1.8, .3)]
    add("pillar_sq", "3", "square pillar, theta=42, phi=-25, s; COMSOL step 3 geometry (M=4)", layers=sq,
        theta=42., phi=-25., M1=4, M2=4, jones_in=jones(exact="s"),
        oracle="COMSOL 3-D FEM; energy; interface continuity")
    add("pillar_sq_m6", "3", "square pillar as above, p, M=6 (N=169, upper end of the C++ test size)", layers=sq,
        theta=42., phi=-25., M1=6, M2=6, jones_in=jones(exact="p"),
        oracle="COMSOL 3-D FEM; energy; interface continuity")
    ecc = [pillar(.8, .8, .3, .2, .3, .55, 1.8, .3)]
    add("pillar_ecc", "3", "eccentric 0.3x0.2 pillar centred (0.3, 0.55), normal incidence, phi=30: kt=0 s/p branch",
        layers=ecc, theta=0., phi=30., M1=4, M2=4, jones_in=jones(exact="p"),
        slice_xz_y=.53, slice_yz_x=.31,
        oracle="COMSOL 3-D FEM step 4 (theta=0); energy; interface continuity")

    # ---- 4. absorbing pillar and absorbing film
    add("pillar_abs", "4", "absorbing square pillar n=1.8+0.13i, theta=42, phi=-25, p (M=3)",
        layers=[pillar(.8, .8, .3, .3, .4, .4, [1.8, .13], .3)], theta=42., phi=-25., M1=3, M2=3,
        jones_in=jones(exact="p"), oracle="COMSOL 3-D FEM step 4; absorption 0 < 1-R-T < 1")
    add("film_abs", "4", "absorbing film n=1.65+0.12i on an absorbing substrate n=1.38+0.05i, elliptical Jones",
        nI=1.0, nII=[1.38, .05], Lx=1.1, Ly=.8, layers=[uniform(.27, [1.65, .12])], wavelength=1.0,
        theta=27., phi=33., M1=2, M2=2, jones_in=jones(70., 40.), psi_deg=70., delta_deg=40.,
        uniform_stack=True, oracle="Airy characteristic matrix with complex indices (test_gui_2d)")

    # ---- 5. Lx != Ly; several y segments with different x boundaries; structured + films + structured
    add("lx_ne_ly", "5", "Lx=0.62, Ly=0.93, M1=2 != M2=3, nI=1.2, high-index rectangle in silica",
        nI=1.2, nII=1.45, Lx=.62, Ly=.93, layers=[pillar(.62, .93, .31, .37, .3, .5, 2.3, .27, bg=1.46)],
        wavelength=.83, theta=33., phi=17., M1=2, M2=3, jones_in=jones(60.), psi_deg=60., delta_deg=0.,
        slice_xz_y=.46, slice_yz_x=.27, nt=14, oracle="energy; interface continuity")
    multi = patterned(.22, [-.2, .1, .5, .9], [
        ([.1, .35, 1.0], [2.1, 1.0]),                 # origin 0.1 (period 0.9), cut 0.35
        ([.05, .2, .5, .95], [1.0, 1.6, 2.2]),        # origin 0.05, cuts 0.2, 0.5
        ([-.15, .45, .75], [1.5, 1.0]),               # origin -0.15 (wraps to 0.75), cut 0.45
    ])
    add("multi_yseg", "5", "three y segments with different x boundaries, shifted origins (union of x cuts for Cy)",
        nII=1.5, Lx=.9, Ly=1.1, layers=[multi], wavelength=.97, theta=25., phi=-40., M1=2, M2=2,
        jones_in=jones(75., 20.), psi_deg=75., delta_deg=20., slice_xz_y=.2, slice_yz_x=.28,
        oracle="energy; interface continuity")
    stack3 = [pillar(.7, .55, .3, .2, .35, .3, 2.0, .2),
              uniform(.31, 1.46), uniform(.115, 1.96),
              patterned(.25, [.06, .23, .61], [([.05, .25, .75], [1.0, 1.7]), ([.05, .45, .75], [1.7, 1.0])])]
    add("stack_sfs", "5", "structured layer + two uniform films + second structured layer",
        nII=1.52, Lx=.7, Ly=.55, layers=stack3, wavelength=.85, theta=35., phi=62., M1=3, M2=3,
        jones_in=jones(25., 100.), psi_deg=25., delta_deg=100., slice_xz_y=.12, slice_yz_x=.6, nt=16,
        oracle="energy; interface continuity at five interfaces; homogeneous_fast_path on/off")

    # ---- 6. elliptical polarisation; Laurent factorisation; M2 = 0 and M1 = 0 degenerate truncations
    add("ellip_pillar", "6", "square pillar, theta=30, phi=18, elliptical (psi=55, delta=35)", layers=sq,
        theta=30., phi=18., M1=3, M2=3, jones_in=jones(55., 35.), psi_deg=55., delta_deg=35.,
        oracle="energy; interface continuity (selftest2d T5 settings)")
    add("laurent_pillar", "6", "square pillar with pure Laurent factorisation (Cx = Cy = Eps), s", layers=sq,
        theta=42., phi=-25., M1=3, M2=3, jones_in=jones(exact="s"), factorization="laurent",
        oracle="energy; Laurent operators equal")
    add("m2_zero", "6", "genuinely 2-D pillar, M2=0 (y truncated to the zero harmonic), theta=42, phi=-25, p",
        layers=sq, theta=42., phi=-25., M1=5, M2=0, jones_in=jones(exact="p"),
        oracle="energy; interface continuity")
    add("m1_zero", "6", "genuinely 2-D pillar, M1=0 (x truncated to the zero harmonic), theta=42, phi=-25, s",
        layers=sq, theta=42., phi=-25., M1=0, M2=5, jones_in=jones(exact="s"),
        oracle="energy; interface continuity")

    # ---- 7. C4-symmetric pillar at normal incidence (exactly degenerate eigenvalues) and the N~600 case
    add("c4_normal", "7", "C4-symmetric square pillar, normal incidence: exactly degenerate eigenvalues",
        layers=sq, theta=0., phi=0., M1=4, M2=4, jones_in=jones(exact="s"),
        slice_xz_y=.37, slice_yz_x=.46, oracle="energy; interface continuity; x<->y symmetry of the response")
    add("pillar_big", "3,7", "square pillar, M=12 (N=625, the GUI limit): large-scale accuracy", layers=sq,
        theta=42., phi=-25., M1=12, M2=12, jones_in=jones(exact="s"),
        oracle="COMSOL 3-D FEM; energy; interface continuity",
        zs=[-.07, -1e-6, 1e-6, .09, .21, .3 - 1e-6, .3 + 1e-6, .39],
        far=dict(atol=1e-8, rtol=1e-8), near=dict(atol=1e-8, rtol=1e-8))

    # ---- 8. Rayleigh anomalies: kz exactly 0 in a uniform medium (failure cases, no numerical reference)
    add("fail_rayleigh_incident", "8", "lambda = Lx = Ly = 1, normal incidence, nI=1: (+-1,0),(0,+-1) graze in the incident region",
        nII=1.5, Lx=1.0, Ly=1.0, layers=[uniform(.3, 1.8)], wavelength=1.0, theta=0., phi=0.,
        M1=1, M2=1, jones_in=jones(exact="s"), status="RayleighAnomaly",
        expect=dict(region="入射区", orders=[[-1, 0], [0, -1], [0, 1], [1, 0]]), zs=[])
    add("fail_rayleigh_exit", "8", "nII=1.25, lambda/Lx = 1.25: (+-1,0) graze in the exit region",
        nI=1.0, nII=1.25, Lx=.8, Ly=2.0, layers=[uniform(.3, 1.8)], wavelength=1.0, theta=0., phi=0.,
        M1=1, M2=1, jones_in=jones(exact="s"), status="RayleighAnomaly",
        expect=dict(region="出射区", orders=[[-1, 0], [1, 0]]), zs=[])
    add("fail_rayleigh_layer", "8", "second (uniform) layer n=1.25, lambda/Lx = 1.25: (+-1,0) graze in layer 2",
        nI=1.0, nII=2.0, Lx=.8, Ly=2.0, layers=[uniform(.2, 1.5), uniform(.3, 1.25)], wavelength=1.0,
        theta=0., phi=0., M1=1, M2=1, jones_in=jones(exact="s"), status="RayleighAnomaly",
        expect=dict(region="第2层", orders=[[-1, 0], [1, 0]]), zs=[])
    return cases


# ---------------------------------------------------------------- building and sampling
def complex_value(value):
    return complex(*value) if isinstance(value, list) else value


def build_stack(spec):
    st = spec["stack"]
    layers = []
    for layer in st["layers"]:
        if "n" in layer:
            layers.append(rc2.Layer2D(layer["d"], n=complex_value(layer["n"])))
        else:
            profiles = [(p["edges"], [complex_value(v) for v in p["n_values"]]) for p in layer["x_profiles"]]
            layers.append(rc2.Layer2D(layer["d"], y_edges=layer["y_edges"], x_profiles=profiles))
    return rc2.Stack2D(complex_value(st["nI"]), layers, complex_value(st["nII"]),
                       rc2.Lattice(st["Lx"], st["Ly"]))


def jones_vector(spec):
    return [complex(*value) for value in spec["jones"]]


def solve_case(spec):
    """Production solve (may raise rc2.RayleighAnomalyError for the failure cases)."""
    return rc2.solve2d(build_stack(spec), spec["wavelength"], spec["theta_deg"], spec["phi_deg"],
                       spec["M1"], spec["M2"], jones_in=jones_vector(spec), factorization=spec["factorization"])


def incident_carrier(res, x, y, z):
    """exp(i[-Re(kx0) x - Re(ky0) y + Re(kzI0) z]); arrays broadcast (CONTRACT.md section 5)."""
    i0, k0 = res["i0"], res["k0"]
    kx, ky, kz = k0 * res["Kx"][i0], k0 * res["Ky"][i0], k0 * res["kzI"][i0]
    return np.exp(1j * (-np.real(kx) * x - np.real(ky) * y + np.real(kz) * z))


def phase_arrays(res, comps, axis, t, at, zs):
    """Absolute and incident-carrier phase of the six slice fields, plus masks, [component,z,t]."""
    zcol = zs[:, None]
    if axis == "x":
        carrier = incident_carrier(res, t[None, :], at, zcol)
    else:
        carrier = incident_carrier(res, at, t[None, :], zcol)
    fields = {key: comps[key] for key in COMPONENTS}
    absolute = [phase_data(None, fields, key) for key in COMPONENTS]
    incident = [phase_data(None, fields, key, reference=carrier) for key in COMPONENTS]
    values = np.stack([comps[key] for key in COMPONENTS])
    return dict(
        phase_absolute=np.stack([p["phase"].filled(np.nan) for p in absolute]),
        phase_incident=np.stack([p["phase"].filled(np.nan) for p in incident]),
        phase_mask=np.stack([p["mask"] for p in absolute]), values=values)


def sample_case(spec):
    """Production snapshots in the portable exchange layout. Returns (res, arrays, diagnostics)."""
    res = solve_case(spec)
    zs = np.asarray(spec["zs"], dtype=np.float64)
    N = res["N"]
    arrays = dict(mn=np.asarray(res["mn"], np.int64), i0=np.asarray(res["i0"], np.int64),
                  k0=np.asarray(res["k0"], np.float64), theta=np.asarray(res["theta"], np.float64),
                  phi=np.asarray(res["phi"], np.float64), inc=np.asarray(res["inc"], np.complex128),
                  prop_r=np.asarray(res["prop_r"], bool), prop_t=np.asarray(res["prop_t"], bool))
    for key in FAR_ARRAYS:
        arrays[key] = np.asarray(res[key], np.float64 if key in ("Kx", "Ky", "DEr", "DEt") else np.complex128)
    # --- six-component harmonic columns [component, z, order] (order = harmonic_index_2d)
    cols = [rc2.field_at(res, float(z)) for z in zs]
    arrays["z"] = zs
    arrays["harmonics"] = np.stack([[col[key] for col in cols] for key in COMPONENTS]).astype(np.complex128)
    # --- operators of the structured layers (small cases only)
    lat = res["lat"]
    if spec["save_operators"]:
        for j, layer in enumerate(res["stack"].layers):
            if not layer.homogeneous:
                eps, cx, cy = layer.operators(lat, res["mn"], res["M1"], res["M2"], spec["factorization"])
                arrays[f"eps_L{j}"], arrays[f"cx_L{j}"], arrays[f"cy_L{j}"] = eps, cx, cy
    # --- x-z (fixed y) and y-z (fixed x) slices by direct summation: rc2.field_slice (what the GUI uses)
    near = spec["tolerance"]["near"]
    nt = spec["slice"]["nt"]
    for name, axis, at in (("xz", "x", spec["slice"]["xz_fixed_y"]), ("yz", "y", spec["slice"]["yz_fixed_x"])):
        comps = rc2.field_slice(res, zs, axis=axis, at=at, npts=nt)
        ph = phase_arrays(res, comps, axis, comps["t"], at, zs)
        values = ph.pop("values")
        error = near["atol"] + near["rtol"] * np.abs(values)
        mask = ph["phase_mask"] | (np.abs(values) <= 10 * error)
        for component in spec["structural_zero_components"]:
            mask[COMPONENTS.index(component)] = True
        arrays.update({f"fields_{name}": values, f"E2_{name}": np.asarray(comps["E2"], np.float64),
                       f"t_{name}": np.asarray(comps["t"], np.float64),
                       f"fixed_{name}": np.asarray(at, np.float64),
                       f"phase_comparison_mask_{name}": mask})
        arrays.update({f"{key}_{name}": value for key, value in ph.items()})
    arrays["incident_amplitudes"] = np.asarray([1.0, float(spec["stack"]["nI"])])
    arrays = {key: np.ascontiguousarray(value) if value.ndim else value for key, value in arrays.items()}
    absorption = 1.0 - float(res["DEr"].sum() + res["DEt"].sum())
    diagnostics = dict(N=int(N), n_prop_r=int(res["prop_r"].sum()), n_prop_t=int(res["prop_t"].sum()),
                       sum_DEr=float(res["DEr"].sum()), sum_DEt=float(res["DEt"].sum()), one_minus_R_minus_T=absorption,
                       i0=int(res["i0"]))
    return res, arrays, diagnostics


def failure_info(spec):
    """Run a failure case and return what the production solver reports."""
    try:
        solve_case(spec)
    except rc2.RayleighAnomalyError as exc:
        return dict(error="RayleighAnomalyError", message=str(exc))
    return None


def material_edges(spec):
    """Physical material edges (where the index really changes, periodic wrap included) of all
    patterned layers: (x_edges, y_edges), each reduced modulo the period."""
    lx, ly = spec["stack"]["Lx"], spec["stack"]["Ly"]
    xedges, yedges = set(), set()
    for layer in spec["stack"]["layers"]:
        if "n" in layer:
            continue
        profiles = layer["x_profiles"]
        for p in profiles:
            e, v = p["edges"], [complex_value(a) for a in p["n_values"]]
            for k in range(len(v)):
                if v[k - 1] != v[k]:                       # k = 0 compares with the last segment (wrap)
                    xedges.add(round(e[k] % lx, 12))

        def value_at(p, x):
            e = p["edges"]
            xx = e[0] + (x - e[0]) % lx
            k = int(np.clip(np.searchsorted(e, xx, side="right") - 1, 0, len(p["n_values"]) - 1))
            return complex_value(p["n_values"][k])

        cuts = sorted({round(c % lx, 12) for p in profiles for c in p["edges"]} | {0.0, lx})
        mids = [(a + b) / 2 for a, b in zip(cuts[:-1], cuts[1:])]
        for k, y in enumerate(layer["y_edges"][:-1]):
            if any(value_at(profiles[k - 1], x) != value_at(profiles[k], x) for x in mids):
                yedges.add(round(y % ly, 12))
    return xedges, yedges


def slice_edge_distances(spec):
    """Smallest distance of the slice sample / fixed coordinates to a physical material edge."""
    lx, ly = spec["stack"]["Lx"], spec["stack"]["Ly"]
    nt = spec["slice"]["nt"]
    xedges, yedges = material_edges(spec)

    def dist(values, edges, period):
        if not edges:
            return float("inf")
        return min(abs((v - e + period / 2) % period - period / 2) for v in values for e in edges)

    return dict(xz_along_x=dist(np.arange(nt) * lx / nt, xedges, lx),
                xz_fixed_y=dist([spec["slice"]["xz_fixed_y"]], yedges, ly),
                yz_along_y=dist(np.arange(nt) * ly / nt, yedges, ly),
                yz_fixed_x=dist([spec["slice"]["yz_fixed_x"]], xedges, lx))
