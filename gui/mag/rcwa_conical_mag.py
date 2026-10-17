# -*- coding: utf-8 -*-
"""一维周期磁性结构的圆锥入射 RCWA。

这是 ``conical/`` 的磁性实验分支。圆锥入射的一维周期问题严格等价于
二维矢量 RCWA 的 ``M_y=0`` 极限，因此本实现复用同目录已经验证的
``rcwa2d_mag`` 全矢量 Maxwell 方程，而不是把原来的非磁性 Moharam
约化式做不可靠的替换。

优点：支持任意各向同性 ``eps(x), mu(x)``、端口磁导率、s/p Jones 矩阵、
电磁对偶及完整 E/H/D/B 场；代价：目前使用 2N 阶完整矢量本征问题，
尚未实现原 ``conical`` 的两次 N 阶快速约化。
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from . import rcwa2d_mag as rc2
except ImportError:  # 允许直接执行/导入本文件
    import rcwa2d_mag as rc2

try:
    from conical.polarization import jones_vector
    from conical.spbasis import order_angles, sigma_gauge
except ImportError:
    from polarization import jones_vector
    from spbasis import order_angles, sigma_gauge

__all__ = [
    "ResultConicalMag", "solve_conical", "sweep_phi", "order_index",
    "field_at", "field_map", "eps_mu_profile", "eps_profile",
]


class ResultConicalMag(dict):
    __getattr__ = dict.get


def _complex_attr(value, name, default):
    attr = getattr(value, name, default)
    return complex(attr)


def _layer_eps_mu(layer):
    """从 mag 1D 或旧 rcwa1d 的 Layer 读取 eps/mu。"""
    homogeneous = bool(getattr(layer, "homogeneous", getattr(layer, "n", None) is not None))
    if homogeneous:
        mu = _complex_attr(layer, "mu", 1.0)
        eps = getattr(layer, "eps", None)
        eps = complex(eps) if eps is not None else complex(layer.n) ** 2 / mu
        return True, eps, mu

    mu_values = getattr(layer, "mu_values", None)
    n_values = getattr(layer, "n_values", None)
    eps_values = getattr(layer, "eps_values", None)
    if eps_values is None:
        if n_values is None:
            raise ValueError("周期层缺少 n_values 或 eps_values")
        if mu_values is None:
            mu_values = np.ones(len(n_values), dtype=complex)
        eps_values = np.asarray(n_values, dtype=complex) ** 2 / np.asarray(mu_values, dtype=complex)
    if mu_values is None:
        mu_values = np.ones(len(eps_values), dtype=complex)
    return False, np.asarray(eps_values, dtype=complex), np.asarray(mu_values, dtype=complex)


def _to_2d_layer(layer, Ly):
    if not layer.homogeneous and getattr(layer,"fourier_method","analytic")!="analytic":
        raise ValueError("磁性圆锥分支尚未接入FFT系数，请使用analytic或一维TE/TM分支")
    homogeneous, eps, mu = _layer_eps_mu(layer)
    if homogeneous:
        return rc2.Layer2D(layer.d, eps=eps, mu=mu, name=getattr(layer, "name", ""))
    edges = np.asarray(layer.edges, dtype=float)
    return rc2.Layer2D(layer.d, y_edges=[0.0, Ly],
                       eps_profiles=[(edges, eps)], mu_profiles=[(edges, mu)],
                       name=getattr(layer, "name", ""))


def _to_2d_stack(stack):
    """把 1D 堆栈转换为 y 不变的 2D 磁性堆栈。"""
    period = float(stack.period)
    nI, nII = complex(stack.nI), complex(stack.nII)
    muI, muII = _complex_attr(stack, "muI", 1.0), _complex_attr(stack, "muII", 1.0)
    epsI = getattr(stack, "epsI", None)
    epsII = getattr(stack, "epsII", None)
    epsI = complex(epsI) if epsI is not None else nI ** 2 / muI
    epsII = complex(epsII) if epsII is not None else nII ** 2 / muII
    if abs(nI.imag) > 1e-13 or abs(nII.imag) > 1e-13:
        raise NotImplementedError("当前磁性圆锥分支要求入射/出射端口折射率为实数")
    lat = rc2.Lattice(period, period)
    layers = [_to_2d_layer(layer, lat.Ly) for layer in stack.layers]
    return rc2.Stack2D(nI, layers, nII, lat, epsI=epsI, muI=muI, epsII=epsII, muII=muII)


def _flux_factor(kz, mu, denominator):
    value = np.real(np.asarray(kz, dtype=complex) / mu / denominator)
    value[np.abs(value) < 1e-14] = 0.0
    return np.maximum(value, 0.0)


def _factorization_2d(value):
    if value == "li":
        return "li"
    if value in ("moharam", "laurent"):
        return "laurent"
    raise ValueError("factorization 必须为 'li'、'moharam' 或 'laurent'")


def solve_conical(stack, lam0, theta_deg, phi_deg=0.0, M=20,
                  psi_deg=90.0, delta_deg=0.0, jones_in=None,
                  factorization="li", need_fields=False, need_jones=True,
                  mode_method="2d"):
    """求解磁性一维周期结构的圆锥入射。

    参数与 ``conical.solve_conical`` 基本一致；磁性分支固定使用
    ``mode_method='2d'``（完整 2N 阶 Maxwell 方程）。输入 stack 可以是
    ``mag.rcwa1d_mag.Stack``，也兼容旧 ``rcwa1d.Stack`` 的非磁性结构。
    """
    if int(M) != M or M < 0:
        raise ValueError("M 必须为非负整数")
    if mode_method not in ("2d", "full", "direct"):
        raise ValueError("磁性圆锥分支仅支持 mode_method='2d'（完整矢量方程）")
    M = int(M)
    stack2 = _to_2d_stack(stack)
    factor2 = _factorization_2d(factorization)
    vin = jones_vector(psi_deg, delta_deg, jones_in)

    # 两列单位 s/p 右端，直接构造物理 Jones 矩阵。
    raw_s = rc2.solve2d(stack2, lam0, theta_deg, phi_deg, M, 0,
                         jones_in=np.array([1.0, 0.0]), factorization=factor2)
    raw_p = rc2.solve2d(stack2, lam0, theta_deg, phi_deg, M, 0,
                         jones_in=np.array([0.0, 1.0]), factorization=factor2)
    ms = raw_s["mn"][:, 0].astype(int)
    N = len(ms)
    Jr = np.empty((N, 2, 2), dtype=complex)
    Jt = np.empty_like(Jr)
    for column, raw in enumerate((raw_s, raw_p)):
        Jr[:, 0, column], Jr[:, 1, column] = raw["Rs"], raw["Rp"]
        Jt[:, 0, column], Jt[:, 1, column] = raw["Ts"], raw["Tp"]

    kx = np.asarray(raw_s["Kx"], dtype=complex)
    if np.max(np.abs(kx.imag)) > 1e-12:
        raise NotImplementedError("当前磁性圆锥分支要求实数横向波矢")
    kx = kx.real
    kappa = float(np.real(raw_s["Ky"][0]))
    varphi, cphi, sphi, kt = order_angles(kx, kappa, phi_deg)
    kzI, kzII = raw_s["kzI"], raw_s["kzII"]
    i0 = int(np.where(ms == 0)[0][0])
    denominator = float(raw_s["incoming_flux"])
    flux_r = _flux_factor(kzI, stack2.muI, denominator)
    flux_t = _flux_factor(kzII, stack2.muII, denominator)
    prop_r, prop_t = flux_r > 0, flux_t > 0

    reflected = np.einsum("mab,b->ma", Jr, vin)
    transmitted = np.einsum("mab,b->ma", Jt, vin)
    Rs, Rp = reflected[:, 0], reflected[:, 1]
    Ts, Tp = transmitted[:, 0], transmitted[:, 1]
    DEr = flux_r * (np.abs(Rs) ** 2 + np.abs(Rp) ** 2)
    DEt = flux_t * (np.abs(Ts) ** 2 + np.abs(Tp) ** 2)

    nI, nII = stack2.nI, stack2.nII
    cthI, cthII = kzI / nI, kzII / nII
    incident_modes = np.zeros((2 * N, 2), dtype=complex)
    incident_modes[i0, 0] = 1.0
    incident_modes[N + i0, 1] = cthI[i0]
    R_tangential = np.vstack([Jr[:, 0, :], -cthI[:, None] * Jr[:, 1, :]])
    T_tangential = np.vstack([Jt[:, 0, :], cthII[:, None] * Jt[:, 1, :]])

    out = ResultConicalMag(
        ms=ms, M=M, k0=raw_s["k0"], kxm=raw_s["k0"] * kx,
        kx_over_k0=kx, ky=raw_s["k0"] * kappa, kappa=kappa,
        phi_m=np.degrees(varphi), varphi=varphi, kt_over_k0=kt,
        cos_phi_m=cphi, sin_phi_m=sphi,
        sigma=sigma_gauge(kx), theta=np.radians(theta_deg), phi=np.radians(phi_deg),
        stack=stack, stack2d=stack2, lam0=lam0, factorization=factorization,
        mode_method="2d", backend="mag.rcwa2d_mag (M2=0)",
        kzI=raw_s["k0"] * kzI, kzII=raw_s["k0"] * kzII,
        kzI_over_k0=kzI, kzII_over_k0=kzII,
        prop_r=prop_r, prop_t=prop_t, flux_r=flux_r, flux_t=flux_t,
        Jr=Jr, Jt=Jt, Rs=Rs, Rp=Rp, Ts=Ts, Tp=Tp,
        DEr=DEr, DEt=DEt, jones_in=vin, i0=i0,
        R_tangential=R_tangential, T_tangential=T_tangential,
        incident_modes=incident_modes, incoming_flux=denominator,
        energy=float(np.sum(DEr) + np.sum(DEt)),
        epsI=stack2.epsI, muI=stack2.muI, epsII=stack2.epsII, muII=stack2.muII,
        mode_residual=0.0,
    )
    for a, out_name in enumerate(("s", "p")):
        for b, in_name in enumerate(("s", "p")):
            out[f"DEr_{out_name}{in_name}"] = flux_r * np.abs(Jr[:, a, b]) ** 2
            out[f"DEt_{out_name}{in_name}"] = flux_t * np.abs(Jt[:, a, b]) ** 2

    unity = np.zeros((2, 2), dtype=complex)
    for m in range(N):
        unity += flux_r[m] * Jr[m].conj().T @ Jr[m]
        unity += flux_t[m] * Jt[m].conj().T @ Jt[m]
    out["flux_unitarity"] = unity

    if need_fields:
        out["raw_columns"] = (raw_s, raw_p)
        out["z_bot"] = raw_s["z_bot"]
        out["amps"] = raw_s["amps"]
    if not need_jones:
        out.pop("flux_unitarity", None)
    return out


def order_index(result, m):
    idx = np.where(np.asarray(result["ms"]) == int(m))[0]
    if len(idx) != 1:
        raise KeyError(f"级次 m={m} 不在当前截断范围内")
    return int(idx[0])


def _input_jones(result, value):
    return result["jones_in"] if value is None else jones_vector(value=value)


def _sp_state_from_cartesian(fields, cos_phi, sin_phi):
    Ex, Ey = fields["Ex"], fields["Ey"]
    # rcwa2d_mag 返回的是 Z0 H；而圆锥内部状态使用 h=i Z0 H。
    hx, hy = 1j * fields["Hx"], 1j * fields["Hy"]
    F = np.concatenate([-sin_phi * Ex + cos_phi * Ey,
                        cos_phi * Ex + sin_phi * Ey])
    G = np.concatenate([-cos_phi * hx - sin_phi * hy,
                        -sin_phi * hx + cos_phi * hy])
    return F, G


def field_at(result, z, jones_in=None):
    """返回 z 处磁性圆锥场的 Fourier 谐波。

    键包括 ``Ex,Ey,Ez,Hx,Hy,Hz,Dx,Dy,Dz,Bx,By,Bz``，并提供与原圆锥
    接口一致的逐级次基状态 ``F,G``。求解时须传 ``need_fields=True``。
    """
    if "raw_columns" not in result:
        raise ValueError("请用 solve_conical(..., need_fields=True) 重新求解")
    vin = _input_jones(result, jones_in)
    fs = rc2.field_at(result["raw_columns"][0], z)
    fp = rc2.field_at(result["raw_columns"][1], z)
    fields = {key: vin[0] * fs[key] + vin[1] * fp[key] for key in fs}
    F, G = _sp_state_from_cartesian(fields, result["cos_phi_m"], result["sin_phi_m"])
    fields.update(F=F, G=G)
    return fields


def field_map(result, zs, nx=256, y=0.0, jones_in=None):
    """在 x-z 平面重建磁性圆锥场，y 只贡献公共相位。"""
    zs = np.atleast_1d(zs).astype(float)
    if int(nx) != nx or nx < 2:
        raise ValueError("nx 必须是不小于 2 的整数")
    nx = int(nx)
    period = float(result["stack"].period)
    x = np.arange(nx) / nx * period
    phase = np.exp(-1j * (np.outer(x, result["kxm"]) + result["ky"] * float(y)))
    keys = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
            "Dx", "Dy", "Dz", "Bx", "By", "Bz")
    coeff = {key: np.zeros((len(zs), len(result["ms"])), dtype=complex) for key in keys}
    for iz, z in enumerate(zs):
        fields = field_at(result, z, jones_in)
        for key in keys:
            coeff[key][iz] = fields[key]
    out = {key: coeff[key] @ phase.T for key in keys}
    out["E2"] = np.abs(out["Ex"]) ** 2 + np.abs(out["Ey"]) ** 2 + np.abs(out["Ez"]) ** 2
    out["H2"] = np.abs(out["Hx"]) ** 2 + np.abs(out["Hy"]) ** 2 + np.abs(out["Hz"]) ** 2
    out["x"], out["z"], out["y"] = x, zs, float(y)
    return out


def _layer_profile(layer, x, period):
    homogeneous, eps, mu = _layer_eps_mu(layer)
    xx = np.asarray(x, dtype=float)
    if homogeneous:
        return np.full(xx.shape, eps, dtype=complex), np.full(xx.shape, mu, dtype=complex)
    edges = np.asarray(layer.edges, dtype=float)
    xp = edges[0] + np.mod(xx - edges[0], period)
    index = np.searchsorted(edges, xp, side="right") - 1
    index = np.clip(index, 0, len(eps) - 1)
    return eps[index], mu[index]


def eps_mu_profile(result, zs, nx=256):
    """返回与 ``field_map`` 对齐的 ``eps(x,z),mu(x,z)``。"""
    if "z_bot" not in result:
        raise ValueError("请用 solve_conical(..., need_fields=True) 重新求解")
    zs = np.atleast_1d(zs).astype(float)
    period = float(result["stack"].period)
    x = np.arange(nx) / nx * period
    eps = np.zeros((len(zs), nx), dtype=complex)
    mu = np.zeros_like(eps)
    ztops, ztop = [], 0.0
    for layer in result["stack"].layers:
        ztops.append((ztop, ztop + layer.d, layer))
        ztop += layer.d
    for i, z in enumerate(zs):
        if z < 0:
            eps[i], mu[i] = result["epsI"], result["muI"]
        elif z > result["z_bot"]:
            eps[i], mu[i] = result["epsII"], result["muII"]
        else:
            _, _, layer = next(item for item in ztops if z <= item[1] + 1e-12)
            eps[i], mu[i] = _layer_profile(layer, x, period)
    return x, zs, eps, mu


def eps_profile(result, zs, nx=256):
    """兼容旧接口：返回等效局部折射率 ``sqrt(eps*mu)``。"""
    x, zs, eps, mu = eps_mu_profile(result, zs, nx)
    return x, np.sqrt(eps * mu + 0j)


def sweep_phi(stack, lam0, theta_deg, phi_values, M, fix="theta", **kwargs):
    """圆锥方位角扫描；接口与原 conical 分支一致。"""
    values = np.asarray(phi_values, dtype=float)
    results = []
    s0 = np.sin(np.radians(theta_deg))
    for phi in values:
        theta = float(theta_deg)
        if fix == "kx0":
            value = s0 / np.cos(np.radians(phi))
            if abs(value) > 1:
                raise ValueError(f"fix='kx0' 在 phi={phi:g}° 时没有实入射角")
            theta = float(np.degrees(np.arcsin(value)))
        elif fix != "theta":
            raise ValueError("fix 必须为 'theta' 或 'kx0'")
        results.append(solve_conical(stack, lam0, theta, phi, M=M, **kwargs))
    return results
