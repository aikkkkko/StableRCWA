# -*- coding: utf-8 -*-
"""一维圆锥入射 RCWA 主求解器。

约定与顶层 ``rcwa1d.py`` 相同：时间因子 ``exp(-i wt)``，z 向下，
空间谐波为 ``exp[-i(kxm x + ky y)]``，层内只保留模长不增长的指数。
对每个入射点同时求两列物理 Jones 基 ``[s,p]``。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from rcwa1d import DiagonalEps, Result, s_propagate
from spectral1d import layer_fourier_info

try:
    from .modes_conical import grating_modes, homogeneous_modes
    from .polarization import jones_vector
    from .smatrix_adapt import apply, interface_modes, rud_chain, step_down
    from .spbasis import order_angles, sigma_gauge
except ImportError:  # 允许 ``python conical/rcwa_conical.py`` 式导入
    from modes_conical import grating_modes, homogeneous_modes
    from polarization import jones_vector
    from smatrix_adapt import apply, interface_modes, rud_chain, step_down
    from spbasis import order_angles, sigma_gauge

__all__ = ["ResultConical", "solve_conical", "sweep_phi", "order_index"]


class ResultConical(Result):
    pass


def _safe_row_divide(values, denominator, tol=1e-13):
    out = np.zeros_like(values, dtype=complex)
    good = np.abs(denominator) > tol
    out[good] = values[good] / denominator[good, None]
    return out


def _flux_factor(kz_over_k0, denominator):
    out = np.real(np.asarray(kz_over_k0, complex) / denominator)
    out[np.abs(out) < 1e-14] = 0.0
    return np.maximum(out, 0.0)


def solve_conical(stack, lam0, theta_deg, phi_deg=0.0, M=20,
                  psi_deg=90.0, delta_deg=0.0, jones_in=None,
                  factorization="li", need_fields=False, need_jones=True,
                  mode_method="reduced", diagnostics=False):
    """求解一维周期结构的圆锥入射。

    Parameters
    ----------
    stack : rcwa1d.Stack
        可直接传入 ``demo_comb.build_stack()`` 的返回值。
    psi_deg, delta_deg : float
        ``[Es,Ep]=[sin(psi) exp(i delta), cos(psi)]``；90° 为 s，0° 为 p。
    jones_in : array-like, optional
        直接给 ``[Es,Ep]`` 时优先，并归一化到单位电场幅值。
    factorization : {'li','moharam'}
        Li 逆规则或原始 Laurent/Moharam 形式。
    mode_method : {'reduced','direct','schur'}
        两次 N 阶本征分解（默认，退化层自动改用 Schur）、独立的 2N 阶本征分解（诊断用），
        或强制非均匀层 Schur 传播。
    diagnostics : bool
        True 时另给出各光栅层模式矩阵的最大条件数 ``max_cond_W``/``max_cond_V``。
        需要 2N 阶 SVD，约占一次求解的 1/4，故默认不算。
    """
    if int(M) != M or M < 0:
        raise ValueError("M 必须是非负整数")
    M = int(M)
    if (any(getattr(layer, "anisotropic", False) for layer in stack.layers)
            or any(isinstance(v, DiagonalEps) for v in (stack.nI, stack.nII))):
        raise NotImplementedError("圆锥模式暂不支持各向异性材料；主轴沿坐标轴的张量目前只用于一维平面 TE/TM（rcwa1d.solve）")
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)
    nI, nII = complex(stack.nI), complex(stack.nII)
    if abs(nI.imag) > 1e-14 or abs(nII.imag) > 1e-14:
        raise NotImplementedError("当前版本要求入射区和出射区折射率为实数")
    nI, nII = float(nI.real), float(nII.real)
    if abs(np.cos(theta)) < 1e-12:
        raise ValueError("入射角不能精确等于 90°")

    k0 = 2 * np.pi / lam0
    ms = np.arange(-M, M + 1)
    n_order = len(ms)
    kx = nI * np.sin(theta) * np.cos(phi) - ms * lam0 / stack.period
    kappa = float(nI * np.sin(theta) * np.sin(phi))
    kxm = k0 * kx
    ky = k0 * kappa
    varphi, cphi, sphi, kt = order_angles(kx, kappa, phi_deg)
    Kx = np.diag(kx)

    mode_I = homogeneous_modes(nI, kx, kappa, where="入射区", orders=ms)
    mode_II = homogeneous_modes(nII, kx, kappa, where="出射区", orders=ms)
    modes = []
    for j, layer in enumerate(stack.layers):
        if layer.homogeneous:
            where = f"第{j + 1}层" + (f"「{layer.name}」" if layer.name else "")
            modes.append(homogeneous_modes(layer.n, kx, kappa, where=where, orders=ms))
        else:
            modes.append(grating_modes(
                layer, stack.period, M, Kx, kappa, cphi, sphi,
                factorization=factorization, method=mode_method))

    # 界面、传播、界面、传播……；均匀膜堆始终走 O(N) 对角快路径。
    blocks = []
    previous = mode_I
    for mode, layer in zip(modes, stack.layers):
        blocks.append(interface_modes(previous, mode))
        blocks.append(s_propagate(mode.q, layer.d, k0, generator=mode.schur_generator))
        previous = mode
    blocks.append(interface_modes(previous, mode_II))

    rud = rud_chain(blocks)
    kzI = 1j * mode_I.q[:n_order]
    kzII = 1j * mode_II.q[:n_order]
    cthI = kzI / nI
    cthII = kzII / nII
    i0 = int(np.where(ms == 0)[0][0])

    # 模态状态是 [Es, Et]；单位物理 p 电场的切向分量为 cos(theta)。
    inc = np.zeros((2 * n_order, 2), complex)
    inc[i0, 0] = 1.0
    inc[n_order + i0, 1] = cthI[i0]
    downs = [inc]
    for k, block in enumerate(blocks):
        downs.append(step_down(block, rud[k + 1], downs[-1]))
    reflected_tan = apply(rud[0], inc)
    transmitted_tan = downs[-1]

    # 切向 p -> 完整物理 p 电场。上行 p 的切向方向带一个负号。
    Jr = np.empty((n_order, 2, 2), complex)
    Jt = np.empty_like(Jr)
    Jr[:, 0, :] = reflected_tan[:n_order]
    Jr[:, 1, :] = -_safe_row_divide(reflected_tan[n_order:], cthI)
    Jt[:, 0, :] = transmitted_tan[:n_order]
    Jt[:, 1, :] = _safe_row_divide(transmitted_tan[n_order:], cthII)

    vin = jones_vector(psi_deg, delta_deg, jones_in)
    reflected = np.einsum("mab,b->ma", Jr, vin)
    transmitted = np.einsum("mab,b->ma", Jt, vin)
    Rs, Rp = reflected[:, 0], reflected[:, 1]
    Ts, Tp = transmitted[:, 0], transmitted[:, 1]

    den = nI * np.cos(theta)
    flux_r = _flux_factor(kzI, den)
    flux_t = _flux_factor(kzII, den)
    prop_r = flux_r > 0
    prop_t = flux_t > 0
    DEr = flux_r * (np.abs(Rs) ** 2 + np.abs(Rp) ** 2)
    DEt = flux_t * (np.abs(Ts) ** 2 + np.abs(Tp) ** 2)

    out = ResultConical(
        ms=ms, M=M, k0=k0, kxm=kxm, kx_over_k0=kx, ky=ky, kappa=kappa,
        phi_m=np.degrees(varphi), varphi=varphi, kt_over_k0=kt,
        sigma=sigma_gauge(kx), theta=theta, phi=phi,
        stack=stack, lam0=lam0, factorization=factorization,
        fourier_layers=[layer_fourier_info(layer,stack.period,M) for layer in stack.layers],
        mode_method=mode_method, kzI=k0 * kzI, kzII=k0 * kzII,
        kzI_over_k0=kzI, kzII_over_k0=kzII,
        prop_r=prop_r, prop_t=prop_t, flux_r=flux_r, flux_t=flux_t,
        Jr=Jr, Jt=Jt, Rs=Rs, Rp=Rp, Ts=Ts, Tp=Tp,
        DEr=DEr, DEt=DEt, jones_in=vin, i0=i0,
        R_tangential=reflected_tan, T_tangential=transmitted_tan,
        energy=float(DEr.sum() + DEt.sum()),
        mode_residual=max([m.ode_residual for m in modes] or [0.0]),
        n_schur=sum(m.schur_generator is not None for m in modes),
        min_abs_lambda=min([m.min_abs_lambda for m in modes] or [float("inf")]),
        degenerate_mode_warning=any(m.degenerate_mode_warning for m in modes),
    )
    if diagnostics:
        out.update(max_cond_W=max([m.cond_W for m in modes] or [1.0]),
                   max_cond_V=max([m.cond_V for m in modes] or [1.0]))

    names = ("s", "p")
    for a, out_name in enumerate(names):
        for b, in_name in enumerate(names):
            out[f"DEr_{out_name}{in_name}"] = flux_r * np.abs(Jr[:, a, b]) ** 2
            out[f"DEt_{out_name}{in_name}"] = flux_t * np.abs(Jt[:, a, b]) ** 2

    unity = np.zeros((2, 2), complex)
    for m in range(n_order):
        unity += flux_r[m] * Jr[m].conj().T @ Jr[m]
        unity += flux_t[m] * Jt[m].conj().T @ Jt[m]
    out["flux_unitarity"] = unity

    if need_fields:
        amps = []
        ztop = 0.0
        for j, (layer, mode) in enumerate(zip(stack.layers, modes)):
            a_plus = downs[2 * j + 1]
            a_minus = apply(rud[2 * j + 2], downs[2 * j + 2])
            amps.append(dict(a_plus=a_plus, a_minus=a_minus, mode=mode,
                             q=mode.q, W=mode.W, V=mode.V, d=layer.d,
                             z_top=ztop, layer=layer, schur_generator=mode.schur_generator))
            ztop += layer.d
        out.update(amps=amps, z_bot=ztop, modes=modes, mode_I=mode_I,
                   mode_II=mode_II, blocks=blocks, rud=rud, downs=downs,
                   cos_phi_m=cphi, sin_phi_m=sphi, incident_modes=inc)
    if not need_jones:
        # 计算仍是双右端（代价几乎相同），但尊重 API 的轻量返回意图。
        out.pop("flux_unitarity", None)
    return out


def order_index(result, m):
    """返回衍射级 ``m`` 的数组下标。"""
    idx = np.where(np.asarray(result["ms"]) == int(m))[0]
    if len(idx) != 1:
        raise KeyError("级次 m=%s 不在当前截断范围内" % m)
    return int(idx[0])


def sweep_phi(stack, lam0, theta_deg, phi_values, M, fix="theta", **kwargs):
    """方位角扫描。

    ``fix='theta'`` 是真实绕法线装调；``fix='kx0'`` 保持横向 x 波矢
    不变（圆锥利特罗快扫的物理路径，当前仅改变角度，不缓存本征分解）。
    """
    phi_values = np.asarray(phi_values, float)
    results = []
    s0 = np.sin(np.radians(theta_deg))
    for ph in phi_values:
        th = float(theta_deg)
        if fix == "kx0":
            value = s0 / np.cos(np.radians(ph))
            if abs(value) > 1:
                raise ValueError("fix='kx0' 在 phi=%g° 时没有实入射角" % ph)
            th = float(np.degrees(np.arcsin(value)))
        elif fix != "theta":
            raise ValueError("fix 必须是 'theta' 或 'kx0'")
        results.append(solve_conical(stack, lam0, th, ph, M=M, **kwargs))
    return results
