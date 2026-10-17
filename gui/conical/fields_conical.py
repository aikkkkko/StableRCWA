# -*- coding: utf-8 -*-
"""圆锥 RCWA 的六分量层内场重建。"""
from __future__ import annotations

import numpy as np
from mode_propagation import field_combinations

try:
    from .polarization import jones_vector
    from .spbasis import to_cartesian
except ImportError:
    from polarization import jones_vector
    from spbasis import to_cartesian

__all__ = ["field_at", "field_map", "eps_profile"]


def _combine(matrix, vector):
    return matrix @ vector if np.ndim(matrix) == 2 else matrix * vector


def _input_vector(res, value):
    return res["jones_in"] if value is None else jones_vector(value=value)


def _layer_at(res, z):
    for amp in res["amps"]:
        if z <= amp["z_top"] + amp["d"] + 1e-12:
            return amp
    raise ValueError("z 不在结构层内")


def _eps_operators(res, z):
    if z < 0:
        eps = res["stack"].nI ** 2
        return eps, eps
    if z > res["z_bot"]:
        eps = res["stack"].nII ** 2
        return eps, eps
    amp = _layer_at(res, z)
    mode = amp["mode"]
    if mode.homogeneous:
        eps = mode.n ** 2
        return eps, eps
    return mode.Eps, mode.C


def field_at(res, z, jones_in=None):
    """返回 z 处各 Fourier 谐波的六分量。

    E 以入射电场归一；H 返回 ``Z0 H``。结果还含 ``Dx``，用于侧壁
    连续性验证。调用求解器时必须 ``need_fields=True``。
    """
    if "amps" not in res:
        raise ValueError("请用 solve_conical(..., need_fields=True) 重新求解")
    vin = _input_vector(res, jones_in)
    n = len(res["ms"])
    k0 = res["k0"]

    if z < 0:
        ere = np.exp(-1j * res["kzI"] * z)
        inc = res["incident_modes"] @ vin
        ref = res["R_tangential"] @ vin
        incoming = np.zeros_like(inc)
        active = inc != 0
        # Only excited incident modes need an exponential. Unexcited evanescent
        # modes would grow toward z=-infinity and otherwise produce 0*inf.
        incoming[active] = inc[active] * np.exp(1j * np.tile(res["kzI"], 2)[active] * z)
        F = incoming + ref * np.r_[ere, ere]
        V = res["mode_I"].vdiag
        G = -V * incoming + V * ref * np.r_[ere, ere]
    elif z > res["z_bot"]:
        et = np.exp(1j * res["kzII"] * (z - res["z_bot"]))
        tra = res["T_tangential"] @ vin
        F = tra * np.r_[et, et]
        G = -res["mode_II"].vdiag * tra * np.r_[et, et]
    else:
        amp = _layer_at(res, z)
        zl = z - amp["z_top"]
        summed, difference = field_combinations(
            amp["q"], amp["a_plus"] @ vin, amp["a_minus"] @ vin, zl, amp["d"], k0,
            generator=amp.get("schur_generator"))
        F = amp["W"] @ summed[:, 0]
        G = amp["V"] @ difference[:, 0]

    E, h = to_cartesian(F, G, res["cos_phi_m"], res["sin_phi_m"])
    Ex, Ey = E[:n], E[n:]
    hx, hy = h[:n], h[n:]
    Kx = np.diag(res["kx_over_k0"])
    kap = res["kappa"]
    Eps, C = _eps_operators(res, z)
    rhs = kap * hx - Kx @ hy
    if np.ndim(Eps) == 0:
        Ez = 1j * rhs / Eps
    else:
        Ez = 1j * np.linalg.solve(Eps, rhs)
    hz = 1j * (kap * Ex - Kx @ Ey)
    # h=i Z0 H -> Z0 H=-i h
    Hx, Hy, Hz = -1j * hx, -1j * hy, -1j * hz
    Dx = C * Ex if np.ndim(C) == 0 else C @ Ex
    return dict(Ex=Ex, Ey=Ey, Ez=Ez, Hx=Hx, Hy=Hy, Hz=Hz, Dx=Dx,
                F=F, G=G)


def field_map(res, zs, nx=256, y=0.0, jones_in=None):
    """在 ``(x,z)`` 网格重建场；结构沿 y 不变，y 只贡献公共相位。"""
    if "amps" not in res:
        raise ValueError("请用 solve_conical(..., need_fields=True) 重新求解")
    zs = np.atleast_1d(zs).astype(float)
    if int(nx) != nx or nx < 2:
        raise ValueError("nx 必须是不小于 2 的整数")
    nx = int(nx)
    period = res["stack"].period
    x = np.arange(nx) / nx * period
    phase = np.exp(-1j * (np.outer(x, res["kxm"]) + res["ky"] * y))
    keys = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Dx")
    # 先收集 z 向 Fourier 系数，再对每个分量作一次 BLAS 矩阵乘。
    # 这与逐 z、逐分量执行 ``phase @ harmonics`` 数学等价，
    # 但在高分辨率二维场图上快得多。
    coeff = {key: np.zeros((len(zs), len(res["ms"])), complex) for key in keys}
    vin = _input_vector(res, jones_in)
    n = len(res["ms"])
    c = res["cos_phi_m"][:, None]
    s = res["sin_phi_m"][:, None]
    kx = res["kx_over_k0"][:, None]
    remaining = np.ones(len(zs), bool)
    groups = []
    for mask, amp in ((zs < 0, None), (zs > res["z_bot"], None)):
        selected = remaining & mask
        groups.append((np.flatnonzero(selected), amp))
        remaining[selected] = False
    for amp in res["amps"]:
        selected = remaining & (zs <= amp["z_top"] + amp["d"] + 1e-12)
        groups.append((np.flatnonzero(selected), amp))
        remaining[selected] = False
    if np.any(remaining):
        raise ValueError("z 不在结构层内")

    for group, (indices, amp) in enumerate(groups):
        if not len(indices):
            continue
        z = zs[indices]
        if group == 0:
            ere = np.exp(-1j * res["kzI"][:, None] * z)
            inc = res["incident_modes"] @ vin
            ref = res["R_tangential"] @ vin
            e_re = np.tile(ere, (2, 1))
            incoming = np.zeros((len(inc), len(z)), complex)
            active = inc != 0
            incoming[active] = inc[active, None] * np.exp(1j * np.tile(res["kzI"], 2)[active, None] * z)
            F = incoming + ref[:, None] * e_re
            V = res["mode_I"].vdiag[:, None]
            G = -V * incoming + V * ref[:, None] * e_re
            eps = res["stack"].nI ** 2
            C = eps
        elif group == 1:
            et = np.exp(1j * res["kzII"][:, None] * (z - res["z_bot"]))
            tra = res["T_tangential"] @ vin
            F = tra[:, None] * np.tile(et, (2, 1))
            G = -res["mode_II"].vdiag[:, None] * F
            eps = res["stack"].nII ** 2
            C = eps
        else:
            zl = z - amp["z_top"]
            summed, difference = field_combinations(
                amp["q"], amp["a_plus"] @ vin, amp["a_minus"] @ vin, zl, amp["d"], res["k0"],
                generator=amp.get("schur_generator"))
            F = amp["W"] @ summed
            G = amp["V"] @ difference
            mode = amp["mode"]
            eps = mode.n ** 2 if mode.homogeneous else mode.Eps
            C = eps if mode.homogeneous else mode.C

        es, et = F[:n], F[n:]
        g1, g2 = G[:n], G[n:]
        Ex, Ey = -s * es + c * et, c * es + s * et
        hx, hy = -c * g1 - s * g2, -s * g1 + c * g2
        rhs = res["kappa"] * hx - kx * hy
        Ez = 1j * (rhs / eps if np.ndim(eps) == 0 else np.linalg.solve(eps, rhs))
        hz = 1j * (res["kappa"] * Ex - kx * Ey)
        values = (Ex, Ey, Ez, -1j * hx, -1j * hy, -1j * hz,
                  C * Ex if np.ndim(C) == 0 else C @ Ex)
        for key, value in zip(keys, values):
            coeff[key][indices] = value.T
    out = {key: coeff[key] @ phase.T for key in keys}
    out["E2"] = np.abs(out["Ex"]) ** 2 + np.abs(out["Ey"]) ** 2 + np.abs(out["Ez"]) ** 2
    out["H2"] = np.abs(out["Hx"]) ** 2 + np.abs(out["Hy"]) ** 2 + np.abs(out["Hz"]) ** 2
    out["x"], out["z"], out["y"] = x, zs, float(y)
    return out


def eps_profile(res, zs, nx=256):
    """返回与 ``field_map`` 对齐的折射率剖面。"""
    zs = np.atleast_1d(zs).astype(float)
    period = res["stack"].period
    x = np.arange(nx) / nx * period
    out = np.zeros((len(zs), nx), complex)
    for iz, z in enumerate(zs):
        if z < 0:
            out[iz] = res["stack"].nI
        elif z > res["z_bot"]:
            out[iz] = res["stack"].nII
        else:
            amp = _layer_at(res, z)
            layer = amp["layer"]
            if layer.homogeneous:
                out[iz] = layer.n
            else:
                edges = np.asarray(layer.edges, float)
                xx = edges[0] + (x - edges[0]) % period
                idx = np.searchsorted(edges, xx, side="right") - 1
                out[iz] = layer.n_values[np.clip(idx, 0, len(layer.n_values) - 1)]
    return x, out
