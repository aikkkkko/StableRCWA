# -*- coding: utf-8 -*-
"""一维圆锥 RCWA 的模式求解。

光栅层使用 Moharam et al. (JOSA A 12, 1068, 1995) 的一维圆锥约化：
一次 ``A`` 与一次 ``B C`` 的 N 阶本征分解，代替 4N 阶一阶系统。
``C=inv(<<1/eps>>)`` 为 Li 逆规则；``C=<<eps>>`` 为原 Moharam 形式。
"""
from __future__ import annotations

from dataclasses import dataclass
import pathlib
import sys
import warnings
from typing import Optional

import numpy as np
from scipy import linalg

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from rcwa1d import (RayleighAnomalyError, _branch, _eig_hermitian_definite, _is_hermitian,
                    rayleigh_message)
from mode_propagation import outgoing_triangular_root

try:
    from .spbasis import transform_modes
except ImportError:  # 允许直接运行本文件
    from spbasis import transform_modes

__all__ = ["ModeSet", "homogeneous_modes", "grating_modes", "maxwell_operators"]


@dataclass
class ModeSet:
    W: np.ndarray
    V: np.ndarray
    q: np.ndarray
    homogeneous: bool
    n: Optional[complex] = None
    vdiag: Optional[np.ndarray] = None
    Eps: Optional[np.ndarray] = None
    Aeps: Optional[np.ndarray] = None
    C: Optional[np.ndarray] = None
    Ei: Optional[np.ndarray] = None
    ode_residual: float = 0.0
    schur_generator: Optional[np.ndarray] = None
    min_abs_lambda: float = float("inf")
    degenerate_mode_warning: bool = False

    # 条件数只作诊断，按需计算：2N 阶 SVD 约占一次圆锥求解的 1/4。
    @property
    def cond_W(self) -> float:
        return 1.0 if self.homogeneous else float(np.linalg.cond(self.W))

    @property
    def cond_V(self) -> float:
        return 1.0 if self.homogeneous else float(np.linalg.cond(self.V))


def _kz_branch(n, kx, ky):
    kz = np.sqrt((complex(n) ** 2 - np.asarray(kx) ** 2 - ky ** 2).astype(complex))
    kz = np.where(kz.imag < 0, -kz, kz)
    kz = np.where((np.abs(kz.imag) < 1e-12) & (kz.real < 0), -kz, kz)
    return kz


def homogeneous_modes(n, kx_over_k0, kappa, grazing_tol=1e-13, where="均匀层", orders=None):
    """均匀层解析模式，内部基为 ``[Es,Et]`` / ``[-ht,hs]``。

    ``W=I``，``V=diag(q,-n^2/q)``。第二个对角元对应切向 p 电场，
    与常见以 ``H_s`` 为状态量的 ``q/n^2`` 互为阻抗倒数。
    kz 恰为 0 时 s 支也退化，抛出 RayleighAnomalyError；仅接近 0 时告警并继续。
    """
    kx = np.asarray(kx_over_k0, float)
    kz = _kz_branch(n, kx, float(kappa))
    q0 = _branch(-1j * kz)
    grazing = np.flatnonzero(q0 == 0)
    if grazing.size:
        labels = grazing if orders is None else np.asarray(orders)[grazing]
        raise RayleighAnomalyError(rayleigh_message(where, n, labels))
    if np.any(np.abs(q0) < grazing_tol):
        warnings.warn(
            "存在精确掠射/Rayleigh 级；该级的 p Jones 基在截止点无唯一归一化。",
            RuntimeWarning,
            stacklevel=2,
        )
    qsafe = np.where(np.abs(q0) < grazing_tol, grazing_tol + 0j, q0)
    vdiag = np.concatenate((q0, -(complex(n) ** 2) / qsafe))
    q = np.concatenate((q0, q0))
    size = 2 * len(kx)
    return ModeSet(np.eye(size, dtype=complex), np.diag(vdiag), q,
                   homogeneous=True, n=n, vdiag=vdiag)


def maxwell_operators(Eps, C, Kx, kappa):
    """返回直角切向基 ``[Ex;Ey]`` / ``[hx;hy]`` 的 P、Q。"""
    n = Kx.shape[0]
    eye = np.eye(n, dtype=complex)
    Ei = np.linalg.solve(Eps, eye)
    kap = float(kappa)
    P = np.block([
        [kap * Kx @ Ei, eye - Kx @ Ei @ Kx],
        [kap ** 2 * Ei - eye, -kap * Ei @ Kx],
    ])
    Q = np.block([
        [kap * Kx, Eps - Kx @ Kx],
        [kap ** 2 * eye - C, -kap * Kx],
    ])
    return P, Q, Ei


def _normalize_columns(W, V):
    scale = np.sqrt(np.sum(np.abs(W) ** 2, axis=0) + np.sum(np.abs(V) ** 2, axis=0))
    if np.any(scale < 1e-28):
        raise np.linalg.LinAlgError("圆锥模式列在截止点退化；请把角度移离精确 Rayleigh 点")
    return W / scale[None, :], V / scale[None, :]


def _direct_modes(Eps, C, Kx, kappa, cos_phi, sin_phi):
    """独立的 2N 阶二阶系统实现；用于验证约化式。"""
    P, Q, Ei = maxwell_operators(Eps, C, Kx, kappa)
    w2, Wxy = np.linalg.eig(P @ Q)
    q = _branch(np.sqrt(w2.astype(complex)))
    if np.any(np.abs(q) < 1e-13):
        raise np.linalg.LinAlgError("direct 模式在精确截止点包含 1/q")
    Vxy = (Q @ Wxy) / q[None, :]
    W, V = transform_modes(Wxy, Vxy, cos_phi, sin_phi)
    W, V = _normalize_columns(W, V)
    den = max(np.linalg.norm(P @ Q @ Wxy), 1e-30)
    residual = np.linalg.norm(P @ Q @ Wxy - Wxy * (q ** 2)[None, :]) / den
    return W, V, q, Ei, float(residual)


def _schur_modes(Eps, Aeps, C, Kx, kappa, cos_phi, sin_phi,
                 min_abs_lambda, warned):
    """Complete 2N Schur basis; retain triangular coupling in propagation and fields."""
    P, Q, Ei = maxwell_operators(Eps, C, Kx, kappa)
    T, Z = linalg.schur(P @ Q, output="complex")
    generator = outgoing_triangular_root(T, _branch)
    # Right solve Vxy * generator = Q * Z. No inverse or eigenvector back substitution.
    Vxy = linalg.solve_triangular(generator.T, (Q @ Z).T, lower=True)
    Vxy = Vxy.T
    if not np.isfinite(Vxy).all():
        raise np.linalg.LinAlgError("Nonfinite Schur magnetic basis")
    residuals = []
    for a, b in ((P @ Vxy, Z @ generator), (Q @ Z, Vxy @ generator)):
        residuals.append(np.linalg.norm(a - b) / max(np.linalg.norm(a), np.linalg.norm(b), np.finfo(float).tiny))
    residual = float(max(residuals))
    if not np.isfinite(residual) or residual > 1e-10:
        raise np.linalg.LinAlgError("Schur mode equations have excessive residual")
    W, V = transform_modes(Z, Vxy, cos_phi, sin_phi)
    # Do not normalize columns independently: that would also require D^-1 G D.
    return ModeSet(W, V, np.diag(generator).copy(), False, Eps=Eps, Aeps=Aeps,
                   C=C, Ei=Ei, ode_residual=residual, schur_generator=generator,
                   min_abs_lambda=min_abs_lambda, degenerate_mode_warning=warned)


def grating_modes(layer, period, M, Kx, kappa, cos_phi, sin_phi,
                  factorization="li", method="reduced"):
    """返回一层光栅的 2N 个圆锥模式。

    ``method='reduced'`` 是生产路径（两次 N 阶 eig）；``'direct'`` 是
    独立的 2N 阶二阶系统，仅供自检和诊断。接近约化退化点时默认切换为
    完整 Schur 传播；``schur`` 可强制对非均匀层使用该路径。
    """
    if factorization not in ("li", "moharam"):
        raise ValueError("factorization 必须是 'li' 或 'moharam'")
    if method not in ("reduced", "direct", "schur"):
        raise ValueError("method 必须是 'reduced'、'direct' 或 'schur'")

    # 用户用分段结构描述严格均匀层时，避免两支简并导致的病态本征基。
    nv = np.asarray(layer.n_values, complex)
    if np.max(np.abs(nv - nv[0])) < 1e-14:
        return homogeneous_modes(nv[0], np.diag(Kx).real, kappa)

    Eps, Aeps = layer.operators(period, M)
    n = Eps.shape[0]
    eye = np.eye(n, dtype=complex)
    Ei = np.linalg.solve(Eps, eye)
    C = np.linalg.solve(Aeps, eye) if factorization == "li" else Eps.copy()

    if method == "direct":
        W, V, q, Ei, residual = _direct_modes(
            Eps, C, Kx, kappa, cos_phi, sin_phi)
        return ModeSet(W, V, q, False, Eps=Eps, Aeps=Aeps, C=C, Ei=Ei,
                       ode_residual=residual)

    kx = np.diagonal(Kx)                                   # Kx 对角：矩阵乘改为行列缩放
    A = np.diag(kx ** 2) - Eps
    B = kx[:, None] * Ei * kx[None, :] - eye
    # 无损介质层：A Hermitian；B C w = λ w 令 u = C w 化为 B u = λ inv(C) u，
    # inv(C) 为 Aeps（li）或 Ei（moharam），Hermitian 正定时走 eigh，否则回退 eig。
    lam1 = lam2 = None
    if _is_hermitian(Eps) and _is_hermitian(B):
        lam1, W1 = np.linalg.eigh(0.5 * (A + A.conj().T))
        metric = Aeps if factorization == "li" else Ei
        modes = _eig_hermitian_definite(B, metric) if _is_hermitian(metric) else None
        if modes is not None:
            lam2, U = modes
            W2 = metric @ U
    if lam1 is None:
        lam1, W1 = np.linalg.eig(A)
    if lam2 is None:
        lam2, W2 = np.linalg.eig(B @ C)
    smallest = float(min(np.min(np.abs(lam1)), np.min(np.abs(lam2))))
    warned = bool(kappa != 0 and smallest < 1e-2 * abs(kappa) ** 1.5)
    if warned or method == "schur":
        return _schur_modes(Eps, Aeps, C, Kx, kappa, cos_phi, sin_phi, smallest, warned)
    q1 = _branch(np.sqrt((lam1 + kappa ** 2).astype(complex)))
    q2 = _branch(np.sqrt((lam2 + kappa ** 2).astype(complex)))

    # PLAN 式 (2.3-44') 的无逆列缩放，先写到直角切向基。
    Wxy = np.block([
        [np.zeros((n, n), complex), W2 * lam2[None, :]],
        [W1 * (q1 ** 2)[None, :], kappa * (Ei @ Kx @ C @ W2)],
    ])
    Vxy = np.block([
        [-W1 * (lam1 * q1)[None, :], np.zeros((n, n), complex)],
        [-kappa * (Kx @ W1) * q1[None, :], -(C @ W2) * q2[None, :]],
    ])
    q = np.concatenate((q1, q2))

    P, Q, _ = maxwell_operators(Eps, C, Kx, kappa)
    PQW = P @ Q @ Wxy
    ref = max(np.linalg.norm(PQW), 1e-30)
    residual = np.linalg.norm(PQW - Wxy * (q ** 2)[None, :]) / ref
    W, V = transform_modes(Wxy, Vxy, cos_phi, sin_phi)
    W, V = _normalize_columns(W, V)
    return ModeSet(W, V, q, False, Eps=Eps, Aeps=Aeps, C=C, Ei=Ei,
                   ode_residual=float(residual), min_abs_lambda=smallest)
