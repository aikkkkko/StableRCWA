# -*- coding: utf-8 -*-
"""逐级次衍射面基。

电场切向量按 ``[Ex; Ey] -> [Es; Et]`` 旋转；磁场使用
``h = i Z0 H``，并按 ``[hx; hy] -> [-ht; hs]`` 旋转。这样 Maxwell
的一阶系统仍保持 ``dF=P G, dG=Q F`` 的奇偶结构，而均匀层的模式矩阵
严格为对角形式。这一点比只对 E、H 做同一个几何旋转更重要。
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "order_angles", "sigma_gauge", "transform_modes", "to_cartesian",
    "diffraction_plane_spread",
]


def order_angles(kx_over_k0, ky_over_k0, incident_phi_deg=0.0, atol=1e-14):
    """返回每级 ``(varphi, cos(varphi), sin(varphi), kt/k0)``。

    ``varphi = atan2(ky,kx)``。当 ``kt=0`` 时按 PLAN 的 G3 规范用入射
    方位角延拓，避免法向入射零级的 0/0。
    """
    kx = np.asarray(kx_over_k0, float)
    ky = np.broadcast_to(np.asarray(ky_over_k0, float), kx.shape)
    kt = np.hypot(kx, ky)
    varphi = np.arctan2(ky, kx)
    varphi = np.where(kt <= atol, np.radians(incident_phi_deg), varphi)
    return varphi, np.cos(varphi), np.sin(varphi), kt


def sigma_gauge(kx_over_k0):
    """PLAN G2 的 ``sigma_m``；零点取 +1。"""
    kx = np.asarray(kx_over_k0, float)
    return np.where(kx < 0.0, -1.0, 1.0)


def transform_modes(Wxy, Vxy, cos_phi, sin_phi):
    """把直角切向模式转到稳定的逐级次衍射面基。

    ``Wxy`` 的行是 ``[Ex;Ey]``；``Vxy`` 的行是 ``[hx;hy]``，其中
    ``h=i Z0 H``，且层内约定为 ``F=W(a+ + a-)``、
    ``G=V(-a+ + a-)``。返回的行分别是 ``[Es;Et]`` 与 ``[-ht;hs]``。
    """
    Wxy = np.asarray(Wxy, complex)
    Vxy = np.asarray(Vxy, complex)
    n = len(cos_phi)
    c = np.asarray(cos_phi)[:, None]
    s = np.asarray(sin_phi)[:, None]
    ex, ey = Wxy[:n], Wxy[n:]
    hx, hy = Vxy[:n], Vxy[n:]
    W = np.vstack((-s * ex + c * ey, c * ex + s * ey))
    V = np.vstack((-c * hx - s * hy, -s * hx + c * hy))
    return W, V


def to_cartesian(F, G, cos_phi, sin_phi):
    """``[Es;Et],[-ht;hs]`` 逆变换为 ``[Ex;Ey],[hx;hy]``。"""
    F = np.asarray(F, complex)
    G = np.asarray(G, complex)
    n = len(cos_phi)
    c = np.asarray(cos_phi)
    s = np.asarray(sin_phi)
    es, et = F[:n], F[n:]
    g1, g2 = G[:n], G[n:]
    ex = -s * es + c * et
    ey = c * es + s * et
    hx = -c * g1 - s * g2
    hy = -s * g1 + c * g2
    return np.concatenate((ex, ey)), np.concatenate((hx, hy))


def diffraction_plane_spread(varphi, propagating=None):
    """返回 PLAN 图 A 所用的衍射面方位角展开（弧度）。

    在严格平面入射时 0/180 度代表同一个平面，返回 0；圆锥入射时按
    ``atan2`` 主值的最大值减最小值，复现文档的 170.7577° 定义。
    """
    a = np.asarray(varphi, float)
    if propagating is not None:
        a = a[np.asarray(propagating, bool)]
    if a.size < 2:
        return 0.0
    if np.max(np.abs(np.sin(a))) < 1e-13:
        return 0.0
    return float(a.max() - a.min())
