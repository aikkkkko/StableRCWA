# -*- coding: utf-8 -*-
"""把 rcwa1d 的 S 矩阵递归适配到两列 Jones 右端。"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from rcwa1d import _rud_below, s_interface

__all__ = ["apply", "step_down", "rud_chain", "interface_modes"]


def apply(matrix_or_diag, rhs):
    """一维对角块同时支持向量和 ``(N,nrhs)`` 右端。"""
    if np.ndim(matrix_or_diag) == 1:
        return matrix_or_diag * rhs if np.ndim(rhs) == 1 else matrix_or_diag[:, None] * rhs
    return matrix_or_diag @ rhs


def step_down(block, rud_next, down):
    if block.diag and (rud_next is None or np.ndim(rud_next) == 1):
        r = 0.0 if rud_next is None else rud_next
        den = 1.0 - block.Rdu * r
        return apply(block.Tdd / den, down)
    if block.diag:
        rhs = apply(block.Tdd, down)          # 对角块 + 稠密 Rud（见 rcwa1d._rud_below）
        if not np.any(block.Rdu):
            return rhs
        n = rud_next.shape[0]
        return np.linalg.solve(np.eye(n, dtype=complex) - block.Rdu[:, None] * rud_next, rhs)
    dense = block.dense()
    n = dense.Tdd.shape[0]
    r = np.zeros((n, n), complex) if rud_next is None else (
        np.diag(rud_next) if np.ndim(rud_next) == 1 else rud_next)
    return np.linalg.solve(np.eye(n, dtype=complex) - dense.Rdu @ r,
                           dense.Tdd @ down)


def rud_chain(blocks):
    out = [None] * (len(blocks) + 1)
    for k in range(len(blocks) - 1, -1, -1):
        out[k] = _rud_below(blocks[k], out[k + 1])
    return out


def interface_modes(upper, lower):
    """均匀层之间保持标量对角快路径；其余界面走稠密路径。"""
    if upper.homogeneous and lower.homogeneous:
        return s_interface(upper.W, upper.V, lower.W, lower.V,
                           both_homog=True, q1=upper.vdiag, q2=lower.vdiag,
                           n1=1.0, n2=1.0, pol="TE")
    return s_interface(upper.W, upper.V, lower.W, lower.V)
