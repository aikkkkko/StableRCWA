"""Mixed (Jones) incident polarization for the lateral-PML branch.

Geometry is invariant along y and every material is isotropic; the x-stretched
PML only adds diagonal tensors.  With the incident wave vector in the x-z plane,
Maxwell's equations therefore split exactly into TE (Ey,Hx,Hz) and TM
(Hy,Ex,Ez).  afmm1d solves precisely these two scalar problems, so

    F(Es,Ep) = w_TE * F_TE + w_TM * F_TM

is the discrete mixed-polarization solution to rounding, not an approximation.
Its truncation error is the same weighted combination of the TE and TM errors,
so M and the PML must be adequate for BOTH polarizations.

Jones convention is that of the conical/2D solvers: [Es,Ep] = [sin(psi) e^{i
delta}, cos(psi)], normalized, in the s/p basis of conical.spbasis at phi=0.
That basis follows the in-plane incident wave vector, so it flips sign for
theta<0 (sigma=-1).  F_TE has Ey_inc=1; F_TM has Z0*Hy_inc=1, hence
|E_inc| = sqrt(muI/epsI).  The weights are

    w_TE = sigma*Es,   w_TM = sigma*sqrt(epsI/muI)*Ep.

Both factors were verified component-by-component against the vector conical
solver at phi=0 (a test in the author's earlier Python package, not included here).
"""
from __future__ import annotations

import math
import time

import numpy as np

from .afmm1d import solve_pml, field_map as _field_map, field_points as _field_points

COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_OWN = {"TE": ("Ey", "Hx", "Hz"), "TM": ("Hy", "Ex", "Ez")}
# Polarization-independent keys copied from the TE result so generic helpers
# (incident phase reference, geometry, adaptive GUI bookkeeping) keep working.
_SHARED = ("is_pml", "formulation", "stack", "lam0", "k0", "theta", "M", "ms", "kxm", "Kx",
           "kappa", "pml", "period", "layers", "z_bot", "epsI", "epsII", "muI", "muII",
           "incident_kxm", "incident_kz", "fourier_layers", "need_fields", "background_materials")


class JonesPMLResult(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def jones_vector(psi_deg=45.0, delta_deg=0.0, jones_in=None):
    """Normalized [Es,Ep]; identical to conical.polarization.jones_vector."""
    if jones_in is None:
        psi = math.radians(float(psi_deg))
        jones_in = [math.sin(psi) * np.exp(1j * math.radians(float(delta_deg))), math.cos(psi)]
    v = np.asarray(jones_in, complex).reshape(2)
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Jones 输入必须是有限的非零矢量")
    return v / norm


def jones_weights(theta_deg, epsI=1.0, muI=1.0, *, psi_deg=45.0, delta_deg=0.0, jones_in=None):
    """Return (w_TE, w_TM, jones) such that F = w_TE*F_TE + w_TM*F_TM."""
    jones = jones_vector(psi_deg, delta_deg, jones_in)
    epsI, muI = complex(epsI), complex(muI)
    if abs(epsI.imag) > 1e-12 or abs(muI.imag) > 1e-12 or epsI.real <= 0 or muI.real <= 0:
        raise ValueError("混合偏振要求无损、正折射率的入射区")
    sigma = -1.0 if float(theta_deg) < 0 else 1.0
    return sigma * jones[0], sigma * math.sqrt(epsI.real / muI.real) * jones[1], jones


def _package(te, tm, theta_deg, psi_deg, delta_deg, jones_in):
    if te["pol"] != "TE" or tm["pol"] != "TM":
        raise ValueError("需要一对 TE、TM 结果")
    w_te, w_tm, jones = jones_weights(theta_deg, te["epsI"], te["muI"], psi_deg=psi_deg,
                                      delta_deg=delta_deg, jones_in=jones_in)
    out = JonesPMLResult({k: te[k] for k in _SHARED if k in te})
    out.update(pol="jones", is_pml_jones=True, TE=te, TM=tm, jones=jones,
               jones_in=jones, weights=(w_te, w_tm),
               psi_deg=float(np.degrees(np.arctan2(abs(jones[0]), abs(jones[1])))),
               delta_deg=float(np.degrees(np.angle(jones[0] * np.conj(jones[1])))),
               modal_residual=max(te["modal_residual"], tm["modal_residual"]))
    return out


def solve_pml_jones(stack, lam0, theta_deg, M, *, pml, background=None, psi_deg=45.0,
                    delta_deg=0.0, jones_in=None, need_fields=True):
    """Solve TE and TM with identical discretization and bundle them with Jones weights.

    Costs two scalar PML solves. Use field_map/field_points from this module
    (or rcwa1d.field_map) to obtain the superposed six-component fields.
    """
    te = solve_pml(stack, lam0, theta_deg, M, "TE", pml=pml, background=background,
                   need_fields=need_fields)
    tm = solve_pml(stack, lam0, theta_deg, M, "TM", pml=pml, background=background,
                   need_fields=need_fields)
    return _package(te, tm, theta_deg, psi_deg, delta_deg, jones_in)


def combine_fields(te, tm, w_te, w_tm):
    """Superpose TE and TM field dictionaries sampled on the same points."""
    for a, b in (("x", "x"), ("z", "z"), ("physical_mask", "physical_mask")):
        if a in te and not np.array_equal(te[a], tm[b]):
            raise ValueError("TE 与 TM 场必须在同一组采样点上")
    out = {}
    for pol, source, weight in (("TE", te, w_te), ("TM", tm, w_tm)):
        for name in _OWN[pol]:
            for key in (name, name + "_contrast"):
                if key in source:
                    out[key] = weight * np.asarray(source[key], complex)
    missing = [k for k in COMPONENTS if k not in out]
    if missing:
        raise ValueError("缺少分量：" + ", ".join(missing))
    E, H = [out[k] for k in COMPONENTS[:3]], [out[k] for k in COMPONENTS[3:]]
    # |E_inc| = 1 for a normalized Jones vector. Cross terms vanish in |E|^2 because
    # TE and TM electric fields are orthogonal; they survive only in Sy.
    out["E2"] = sum(abs(e) ** 2 for e in E)
    out["Sx"] = 0.5 * np.real(E[1] * np.conj(H[2]) - E[2] * np.conj(H[1]))
    out["Sy"] = 0.5 * np.real(E[2] * np.conj(H[0]) - E[0] * np.conj(H[2]))
    out["Sz"] = 0.5 * np.real(E[0] * np.conj(H[1]) - E[1] * np.conj(H[0]))
    for key in ("physical_mask", "x", "z"):
        if key in te:
            out[key] = te[key]
    return out


def field_map(res, zs, nx=256, xs=None):
    """Six-component superposed x-z map; PML masking follows afmm1d.field_map."""
    w_te, w_tm = res["weights"]
    return combine_fields(_field_map(res["TE"], zs, nx=nx, xs=xs),
                          _field_map(res["TM"], zs, nx=nx, xs=xs), w_te, w_tm)


def field_points(res, x, z):
    w_te, w_tm = res["weights"]
    return combine_fields(_field_points(res["TE"], x, z), _field_points(res["TM"], x, z), w_te, w_tm)


def _same_setup(a, b):
    """Two PML results share window, PML and truncation (to floating tolerance)."""
    pa, pb = a["pml"], b["pml"]
    scale = max(1.0, abs(a["period"]))
    close = lambda u, v: abs(u - v) <= 1e-12 * scale
    return (a["M"] == b["M"] and close(a["period"], b["period"]) and close(pa.x0, pb.x0)
            and close(pa.left, pb.left) and close(pa.right, pb.right)
            and pa.strength == pb.strength and pa.order == pb.order)


_RANK = {"verified": 0, "attenuation_met": 1, "time_limit": 2, "limit_reached": 3}


def solve_adaptive_pml_jones(stack, lam0, theta_deg, M, *, pml, background=None, settings=None,
                             on_progress=None, psi_deg=45.0, delta_deg=0.0, jones_in=None,
                             max_rounds=6):
    """Adaptive PML for mixed polarization: TE and TM must both pass at one common setup.

    The TE/TM adaptive searches alternate, each starting from the setup the
    other one selected. Searches only enlarge the PML/M, so the loop stops once
    a search keeps the setup the other polarization already accepted. Each
    search uses the given settings and budget on its own.
    """
    from .adaptive import solve_adaptive_pml
    if int(max_rounds) < 2:
        raise ValueError("max_rounds 至少为 2（TE、TM 各一次）")
    jones_vector(psi_deg, delta_deg, jones_in)            # validate before spending time
    began = time.perf_counter()
    results, rounds = {}, []
    cur_stack, cur_pml, cur_M = stack, pml, int(M)
    pol, stopped = "TE", None
    for index in range(int(max_rounds)):
        def progress(event, pol=pol):
            if on_progress is not None:
                on_progress(dict(event, polarization=pol))
        res = solve_adaptive_pml(cur_stack, lam0, theta_deg, cur_M, pol, pml=cur_pml,
                                 background=background, settings=settings, on_progress=progress)
        rounds.append(dict(polarization=pol, status=res.adaptive["status"], M=res.M,
                           period=res.period, left=res.pml.left, right=res.pml.right,
                           solves=res.adaptive["max_solves_used"]))
        results[pol] = res
        other = "TM" if pol == "TE" else "TE"
        if other in results and _same_setup(results[other], res):
            break
        cur_stack, cur_pml, cur_M = res.stack, res.pml, res.M
        pol = other
    else:
        stopped = f"TE/TM 交替选参 {max_rounds} 轮后仍未取得共同参数。"
        pol = rounds[-1]["polarization"]
    last = results[pol]
    other = "TM" if pol == "TE" else "TE"
    if stopped or not _same_setup(results[other], last):
        # Keep one common discretization for superposition: re-solve the other
        # polarization at the last setup without claiming it was verified there.
        plain = solve_pml(last.stack, lam0, theta_deg, last.M, other, pml=last.pml,
                          background=background)
        meta = dict(results[other].adaptive)
        meta.update(status="limit_reached", verified=False,
                    message=f"{other} 在最终参数上未单独完成自动选参验证。")
        plain["adaptive"] = meta
        results[other] = plain
        stopped = stopped or "两种偏振未收敛到同一组参数。"
    te, tm = results["TE"], results["TM"]
    out = _package(te, tm, theta_deg, psi_deg, delta_deg, jones_in)
    metas = {"TE": te["adaptive"], "TM": tm["adaptive"]}
    worst = max(metas.values(), key=lambda m: _RANK.get(m["status"], 9))
    status = "limit_reached" if stopped else worst["status"]
    message = "；".join(f"{p}：{str(m.get('message', m['status'])).rstrip('。')}" for p, m in metas.items()) + "。"
    if stopped:
        message = stopped + message
    combined = dict(worst)
    combined.update(
        method="jones_alternating_TE_TM", status=status, verified=status == "verified",
        message=message, polarizations=metas, rounds=rounds,
        tail_ratio=max(m["tail_ratio"] for m in metas.values()),
        tail_goal_met=all(m.get("tail_goal_met", False) for m in metas.values()),
        max_solves_used=sum(r["solves"] for r in rounds) + (1 if stopped else 0),
        elapsed_seconds=time.perf_counter() - began,
        final=dict(period=out["period"], x0=out["pml"].x0, left=out["pml"].left,
                   right=out["pml"].right, strength=out["pml"].strength,
                   order=out["pml"].order, M=out["M"]))
    combined.pop("history", None)
    combined.pop("selected_solve_index", None)
    out["adaptive"] = combined
    return out
