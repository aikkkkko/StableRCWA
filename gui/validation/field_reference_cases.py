"""Small, explicit inputs for the frozen v1 six-component field corpus.

This module samples current production code. It is not an independent oracle;
analytic and first-order validation lives in the test modules.
"""
from copy import deepcopy
from pathlib import Path
import sys

GUI = Path(__file__).resolve().parents[1]
ROOT = GUI.parent
if str(GUI) not in sys.path:
    sys.path.insert(0, str(GUI))

import numpy as np
import rcwa1d as rc
from conical.rcwa_conical import solve_conical
from conical.fields_conical import field_at as conical_at, field_map as conical_map
from field_phase import phase_data
from gui_fields import complete_six_components, normalization

COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
FROZEN_DIR = Path(__file__).resolve().parent / "field_reference"
CONTRACT_VERSION = 1
ANCHOR_REVISION = "pre-release development revision (not in the public git history)"
NUMERIC_SOURCES = (
    "gui/rcwa1d.py", "gui/mode_propagation.py", "gui/spectral1d.py",
    "gui/conical/rcwa_conical.py", "gui/conical/fields_conical.py",
    "gui/conical/modes_conical.py", "gui/conical/smatrix_adapt.py",
    "gui/conical/spbasis.py", "gui/conical/polarization.py",
    "gui/field_phase.py", "gui/gui_fields.py",
)


def case_specs():
    """JSON-serializable parameters; no random sampling or large scans."""
    interface = dict(nI=1.3, nII=1.7, period=.73, layers=[])
    absorbing = dict(nI=1.4, nII=.6, period=.62,
                     layers=[dict(d=.41, n=[1.8, .13])])
    grating = dict(nI=1.2, nII=1.4, period=.6, layers=[
        dict(d=.2, edges=[0., .21, .6], n_values=[1.7, 1.]), dict(d=.09, n=1.46)])
    schur = dict(nI=1., nII=1.46, period=.574, layers=[
        dict(d=.38, edges=[0., .1722, .574], n_values=[1.46, 1.]), dict(d=.115, n=1.96)])
    # Match the input floating-point kx exactly: this is q == 0, not a sweep
    # near cutoff. Existing test_degenerate_fields checks its analytic solution.
    kx = float(2 * np.sin(np.radians(30.)))
    d = float(1 / (np.pi * np.sqrt(4 - kx * kx)))
    cutoff = dict(nI=2., nII=2., period=1., layers=[dict(d=d, n=kx)])
    cases = []

    def add(case_id, stack, wavelength, theta, M, zs, nx, *, mode="planar", pol="TE",
            phi=0., jones=None, psi=90., delta=0., y=0., oracle="", zero=0, n_schur=0):
        zeros = (("Ex", "Ez", "Hy") if pol == "TE" or case_id == "phi0_s" else
                 ("Ey", "Hx", "Hz") if pol == "TM" or case_id == "phi0_p" else ())
        cases.append(dict(id=case_id, mode=mode, stack=deepcopy(stack), wavelength=wavelength,
                          theta_deg=theta, phi_deg=phi, M=M, polarization=pol,
                          jones_in=jones, psi_deg=psi, delta_deg=delta,
                          zs=list(zs), nx=nx, y=y, independent_validation=oracle,
                          structural_zero_components=list(zeros),
                          expected_zero_modes=zero, expected_schur_layers=n_schur,
                          tolerance=dict(atol=3e-10 if n_schur else 3e-12,
                                         rtol=3e-10 if n_schur else 3e-12)))

    for pol in ("TE", "TM"):
        suffix = pol.lower()
        add("interface_"+suffix, interface, .81, 23., 0, [-.19, -.03, .07, .21], 5,
            pol=pol, oracle="Fresnel single interface, six complex components")
        add("absorbing_film_"+suffix, absorbing, .83, 33., 2,
            [-.09, 0., .07, .205, .41, .55, .73], 7, pol=pol,
            oracle="Independent film boundary system; absorbing film + evanescent zero-order exit")
        add("grating_"+suffix, grating, .83, 27., 3,
            [-.12, 0., .083, .2, .245, .29, .39], 9, pol=pol,
            oracle="phi=0 Cartesian fields; scalar/vector normalization")
    for pol, vin in (("s", [[1., 0.], [0., 0.]]), ("p", [[0., 0.], [1., 0.]])):
        add("phi0_"+pol, grating, .83, 27., 3,
            [-.12, 0., .083, .2, .245, .29, .39], 9, mode="conical", pol=pol,
            jones=vin, oracle="Full complex field mapping to grating_te/tm")
    add("conical_linear", grating, .83, 27., 3,
        [-.12, 0., .083, .2, .245, .29, .39], 9, mode="conical", pol="jones",
        phi=31., psi=37., y=.11, oracle="Map versus harmonics and Jones superposition")
    add("schur_ellipse", schur, .980484673140678, 40., 8,
        [-.08, 0., .07, .19, .38, .435, .495, .58], 9, mode="conical", pol="jones",
        phi=25., jones=[[.3, .2], [-.7, .5]], y=.11, n_schur=1,
        oracle="test_degenerate_fields: 4N first-order propagation + six-component reconstruction")
    for pol in ("TE", "TM"):
        add("zero_"+pol.lower(), cutoff, 1., 30., 0,
            [-.1, 0., .25*d, .5*d, .75*d, d, d+.1], 7, pol=pol, zero=1,
            oracle="test_degenerate_fields: exact constant/linear finite-layer solution")
    return cases


def complex_value(value):
    return complex(*value) if isinstance(value, list) else value


def build_stack(spec):
    layers = []
    for layer in spec["layers"]:
        kwargs = dict(n=complex_value(layer["n"])) if "n" in layer else dict(
            edges=layer["edges"], n_values=[complex_value(v) for v in layer["n_values"]])
        layers.append(rc.Layer(layer["d"], **kwargs))
    return rc.Stack(complex_value(spec["nI"]), layers, complex_value(spec["nII"]), spec["period"])


def solve_case(spec):
    stack = build_stack(spec["stack"])
    if spec["mode"] == "planar":
        return rc.solve(stack, spec["wavelength"], spec["theta_deg"], spec["M"],
                        spec["polarization"], need_fields=True)
    vin = None if spec["jones_in"] is None else [complex_value(v) for v in spec["jones_in"]]
    return solve_conical(stack, spec["wavelength"], spec["theta_deg"], spec["phi_deg"], spec["M"],
                         psi_deg=spec["psi_deg"], delta_deg=spec["delta_deg"],
                         jones_in=vin, need_fields=True)


def gui_config(spec):
    return dict(mode=spec["mode"], structure=dict(n_incident=float(spec["stack"]["nI"])),
                incident=dict(polarization=spec["polarization"], wavelength=spec["wavelength"],
                              angle_deg=spec["theta_deg"], phi_deg=spec["phi_deg"]))


def sample_case(spec):
    """Production snapshots in a portable, explicit six-component exchange layout."""
    res = solve_case(spec)
    zs = np.asarray(spec["zs"], dtype=np.float64)
    if spec["mode"] == "conical":
        fields = conical_map(res, zs, nx=spec["nx"], y=spec["y"])
        points = [conical_at(res, float(z)) for z in zs]
        harmonics = np.stack([[point[key] for point in points] for key in COMPONENTS])
        kzI, kzII = res.kzI, res.kzII
    else:
        fields = complete_six_components(rc.field_map(res, zs, nx=spec["nx"]), spec["polarization"])
        harmonics = np.zeros((6, len(zs), len(res.ms)), dtype=np.complex128)
        kx = np.diag(res.Kx)
        for iz, z in enumerate(zs):
            sv, uv = rc.field_at(res, float(z))
            if res.pol == "TE":
                harmonics[1, iz], harmonics[3, iz], harmonics[5, iz] = sv, 1j*uv, -kx*sv
            else:
                eps, _ = rc._eps_ops_at(res, float(z))
                harmonics[0, iz], harmonics[2, iz], harmonics[4, iz] = -1j*uv, np.linalg.solve(eps, kx*sv), sv
        kzI, kzII = res.kIz, res.kIIz
    norm = normalization(gui_config(spec))
    absolute = [phase_data(res, fields, key) for key in COMPONENTS]
    incident = [phase_data(res, fields, key, reference="incident") for key in COMPONENTS]
    arrays = dict(x=np.asarray(fields["x"], np.float64), z=zs, y=np.asarray(spec["y"], np.float64),
                  orders=np.asarray(res.ms, np.int64), kx=np.asarray(res.kxm, np.complex128),
                  ky=np.asarray(res.get("ky", 0.), np.complex128),
                  kzI=np.asarray(kzI, np.complex128), kzII=np.asarray(kzII, np.complex128),
                  fields=np.stack([fields[key] for key in COMPONENTS]), harmonics=harmonics,
                  phase_absolute=np.stack([p["phase"].filled(np.nan) for p in absolute]),
                  phase_incident=np.stack([p["phase"].filled(np.nan) for p in incident]),
                  phase_mask=np.stack([p["mask"] for p in absolute]),
                  incident_amplitudes=np.asarray([norm["E"], norm["H"]], np.float64))
    # Keep the raw phase/mask snapshot above. Roundoff in a symmetry-zero
    # component must not become a physical phase oracle for another backend.
    error_floor = spec["tolerance"]["atol"] + spec["tolerance"]["rtol"] * np.abs(arrays["fields"])
    comparison_mask = arrays["phase_mask"] | (np.abs(arrays["fields"]) <= 10*error_floor)
    for component in spec["structural_zero_components"]:
        comparison_mask[COMPONENTS.index(component)] = True
    arrays["phase_comparison_mask"] = comparison_mask
    arrays = {key: np.ascontiguousarray(value) if value.ndim else value for key, value in arrays.items()}
    diagnostics = dict(n_zero_modes=int(res.get("n_zero_modes", 0)), n_schur=int(res.get("n_schur", 0)))
    return res, fields, arrays, diagnostics
