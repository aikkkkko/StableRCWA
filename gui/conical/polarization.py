# -*- coding: utf-8 -*-
"""Jones、Stokes、Mueller 与偏振椭圆后处理。"""
from __future__ import annotations

import numpy as np

__all__ = ["jones_vector", "stokes", "mueller", "ellipse", "dop", "cross_ratio"]

_SIGMA = np.array([
    [[1, 0], [0, 1]],
    [[1, 0], [0, -1]],
    [[0, 1], [1, 0]],
    [[0, -1j], [1j, 0]],
], dtype=complex)


def jones_vector(psi_deg=90.0, delta_deg=0.0, value=None):
    """返回归一化 ``[Es,Ep]``；``delta`` 是 s 相对 p 的相位。"""
    if value is None:
        psi = np.radians(psi_deg)
        value = np.array([np.sin(psi) * np.exp(1j * np.radians(delta_deg)),
                          np.cos(psi)], complex)
    v = np.asarray(value, complex).reshape(2)
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Jones 输入不能是零矢量")
    return v / norm


def stokes(jones):
    """最后一轴为 2 的 Jones 矢量 -> ``(...,4)`` Stokes 矢量。"""
    v = np.asarray(jones, complex)
    return np.real(np.einsum("...a,iab,...b->...i", v.conj(), _SIGMA, v))


def mueller(J):
    """Jones 矩阵 -> Mueller 矩阵，满足 ``S(Jv)=M S(v)``。"""
    J = np.asarray(J, complex)
    Jd = np.swapaxes(J.conj(), -1, -2)
    out = np.empty(J.shape[:-2] + (4, 4), float)
    for i in range(4):
        for j in range(4):
            out[..., i, j] = 0.5 * np.real(np.trace(
                _SIGMA[i] @ J @ _SIGMA[j] @ Jd, axis1=-2, axis2=-1))
    return out


def dop(stokes_vector):
    s = np.asarray(stokes_vector, float)
    return np.sqrt(np.sum(s[..., 1:] ** 2, axis=-1)) / np.maximum(np.abs(s[..., 0]), 1e-300)


def ellipse(jones):
    """返回偏振方位角、椭圆率角（度）、手性与 DOP。"""
    s = stokes(jones)
    ratio = np.clip(s[..., 3] / np.maximum(np.abs(s[..., 0]), 1e-300), -1.0, 1.0)
    az = 0.5 * np.arctan2(s[..., 2], s[..., 1])
    ell = 0.5 * np.arcsin(ratio)
    return {
        "azimuth_deg": np.degrees(az),
        "ellipticity_deg": np.degrees(ell),
        "handedness": np.sign(s[..., 3]),
        "dop": dop(s),
        "stokes": s,
    }


def cross_ratio(J):
    J = np.asarray(J, complex)
    cross = np.abs(J[..., 0, 1]) ** 2 + np.abs(J[..., 1, 0]) ** 2
    co = np.abs(J[..., 0, 0]) ** 2 + np.abs(J[..., 1, 1]) ** 2
    return cross / np.maximum(co, 1e-300)
