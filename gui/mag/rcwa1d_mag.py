# -*- coding: utf-8 -*-
"""一维各向同性磁性介质 RCWA（实验分支）。

本文件是一个独立的 1D RCWA 求解器，约定与上级目录的 ``rcwa1d.py``
一致，但把每一层的材料参数从仅有 ``epsilon=n**2, mu=1`` 扩展为
``epsilon(x), mu(x)``。支持：

* 均匀层：``eps``、``mu``，或保持旧接口的 ``n``；
* 分段常数周期层：``eps_values``/``mu_values``，或 ``n_values``/``mu_values``；
* TE/TM、任意多层、S 矩阵级联、层内场重建；
* 外部端口也可以给出非单位 ``muI``/``muII``。

时间因子、波矢和场的符号约定保持旧求解器：
``kx_m = k0 * (nI*sin(theta) - m*lam0/P)``，场的横向展开为
``sum_m F_m exp(-i*kx_m*x)``。本分支暂只覆盖 1D 标量、各向同性介质；
二维实验实现见同目录 ``rcwa2d_mag.py``；一维周期圆锥入射见
``rcwa_conical_mag.py``。
"""

from __future__ import annotations

import numpy as np
from spectral1d import convmat_fft, validate_fourier_settings, layer_fourier_info

__all__ = [
    "Layer", "Stack", "solve", "field_at", "field_map",
    "eps_mu_profile", "eps_profile", "fourier_piecewise", "convmat",
]


# =============================================================================
# Fourier 系数与卷积矩阵
# =============================================================================
def fourier_piecewise(edges, values, period, hmax):
    """分段常数剖面的解析 Fourier 系数，返回 h=-hmax..hmax。"""
    edges = np.asarray(edges, dtype=float)
    values = np.asarray(values, dtype=complex)
    if edges.ndim != 1 or values.ndim != 1 or len(edges) != len(values) + 1:
        raise ValueError("edges 必须比 values 多一个元素")
    if not np.isfinite(edges).all() or period <= 0:
        raise ValueError("edges/period 无效")
    if abs((edges[-1] - edges[0]) - period) > 1e-9 * max(1.0, abs(period)):
        raise ValueError("edges 首尾必须相差一个 period")

    h = np.arange(-hmax, hmax + 1)
    c = np.zeros(len(h), dtype=complex)
    c[hmax] = np.sum(values * np.diff(edges)) / period
    hn = h[h != 0]
    ph = np.exp(-2j * np.pi * np.outer(hn, edges) / period)
    c[h != 0] = (values * (ph[:, 1:] - ph[:, :-1])).sum(axis=1) / (-2j * np.pi * hn)
    return c


def convmat(edges, values, period, M):
    """构造 Fourier 卷积矩阵 ``C[m,p] = c_(m-p)``。"""
    c = fourier_piecewise(edges, values, period, 2 * M)
    ms = np.arange(-M, M + 1)
    return c[np.subtract.outer(ms, ms) + 2 * M]


def _as_values(values, length, label):
    arr = np.asarray(values, dtype=complex)
    if arr.ndim != 1 or len(arr) != length:
        raise ValueError(f"{label} 必须是一维数组，长度应为 {length}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{label} 含有非有限数")
    return arr


# =============================================================================
# 层与堆栈
# =============================================================================
class Layer:
    """一层各向同性介质。

    均匀层的材料参数可以写成：

    ``Layer(d, n=n0)``
        旧接口；等价于 ``eps=n0**2/mu, mu=mu``。
    ``Layer(d, eps=eps0, mu=mu0)``
        直接给相对介电常数和相对磁导率。

    周期层可以写成 ``eps_values``/``mu_values``，也可以给局部折射率
    ``n_values``，此时按 ``eps_values=n_values**2/mu_values`` 解释。
    ``edges`` 长度必须比材料数组多 1，且首尾相差一个周期。
    """

    def __init__(self, thickness, n=None, mu=1.0, eps=None,
                 edges=None, n_values=None, eps_values=None,
                 mu_values=None, name="", *, fourier_method="analytic", fft_samples=0):
        self.d = float(thickness)
        if self.d < 0:
            raise ValueError("层厚度不能为负")
        self.name = name
        self.fourier_method, self.fft_samples = validate_fourier_settings(fourier_method,fft_samples)

        homogeneous_args = (n is not None or eps is not None)
        periodic_args = (edges is not None or n_values is not None or
                         eps_values is not None or mu_values is not None)
        if homogeneous_args and periodic_args:
            raise ValueError("均匀层参数与周期层参数不能同时给出")

        self.homogeneous = homogeneous_args
        if self.homogeneous:
            if n is not None and eps is not None:
                raise ValueError("均匀层请在 n 与 eps 中二选一")
            self.mu = complex(mu)
            self.eps = complex(eps) if eps is not None else complex(n) ** 2 / self.mu
            if self.eps == 0 or self.mu == 0:
                raise ValueError("均匀层 eps 和 mu 不能为零")
            self.n = complex(n) if n is not None else np.sqrt(self.eps * self.mu + 0j)
            self.edges = None
            self.eps_values = None
            self.mu_values = None
            self.n_values = None
            return

        if edges is None or (n_values is None and eps_values is None):
            raise ValueError("周期层必须给出 edges，以及 n_values 或 eps_values")
        if n_values is not None and eps_values is not None:
            raise ValueError("周期层请在 n_values 与 eps_values 中二选一")

        self.edges = np.asarray(edges, dtype=float)
        if self.edges.ndim != 1 or len(self.edges) < 2:
            raise ValueError("edges 必须是一维且至少包含两个端点")
        if np.any(np.diff(self.edges) <= 0):
            raise ValueError("edges 必须严格递增")
        nseg = len(self.edges) - 1
        if eps_values is None:
            nv = _as_values(n_values, nseg, "n_values")
            mv = np.full(nseg, complex(mu), dtype=complex) if mu_values is None else \
                _as_values(mu_values, nseg, "mu_values")
            if np.any(mv == 0):
                raise ValueError("mu_values 不能包含零")
            ev = nv ** 2 / mv
            self.n_values = nv
        else:
            ev = _as_values(eps_values, nseg, "eps_values")
            mv = np.full(nseg, complex(mu), dtype=complex) if mu_values is None else \
                _as_values(mu_values, nseg, "mu_values")
            self.n_values = np.sqrt(ev * mv + 0j)

        if np.any(ev == 0) or np.any(mv == 0):
            raise ValueError("周期层 eps_values 和 mu_values 不能包含零")
        self.eps_values = ev
        self.mu_values = mv
        self.n = None
        self.eps = None
        self.mu = None

    def operators(self, period, M):
        """返回 ``(<eps>, <1/eps>, <mu>, <1/mu>)``。"""
        if self.homogeneous:
            return None, None, None, None
        if abs((self.edges[-1] - self.edges[0]) - period) > 1e-9 * max(1.0, abs(period)):
            raise ValueError("周期层 edges 的周期与 Stack.period 不一致")
        if self.fourier_method == "fft":
            return tuple(convmat_fft(self.edges, values, period, M, self.fft_samples)
                         for values in (self.eps_values,1/self.eps_values,
                                        self.mu_values,1/self.mu_values))
        return (
            convmat(self.edges, self.eps_values, period, M),
            convmat(self.edges, 1.0 / self.eps_values, period, M),
            convmat(self.edges, self.mu_values, period, M),
            convmat(self.edges, 1.0 / self.mu_values, period, M),
        )

    def profile(self, x, period):
        """返回给定 x 上的 ``eps(x), mu(x)``，用于几何/材料检查。"""
        if self.homogeneous:
            xx = np.asarray(x)
            return np.full(xx.shape, self.eps, dtype=complex), np.full(xx.shape, self.mu, dtype=complex)
        xx = np.asarray(x, dtype=float)
        x0 = self.edges[0]
        xp = x0 + np.mod(xx - x0, period)
        idx = np.searchsorted(self.edges, xp, side="right") - 1
        idx = np.clip(idx, 0, len(self.eps_values) - 1)
        return self.eps_values[idx], self.mu_values[idx]


class Stack:
    """``nI | layers... | nII``，端口可分别指定 ``eps`` 与 ``mu``。

    ``nI``/``nII`` 仍用于确定端口的切向波矢，默认按
    ``eps=n**2/mu`` 反推介电常数；显式 ``epsI``/``epsII`` 可覆盖该值。
    """

    def __init__(self, nI, layers, nII, period, muI=1.0, muII=1.0,
                 epsI=None, epsII=None):
        self.nI = complex(nI)
        self.nII = complex(nII)
        self.muI = complex(muI)
        self.muII = complex(muII)
        self.epsI = complex(epsI) if epsI is not None else self.nI ** 2 / self.muI
        self.epsII = complex(epsII) if epsII is not None else self.nII ** 2 / self.muII
        if self.muI == 0 or self.muII == 0 or self.epsI == 0 or self.epsII == 0:
            raise ValueError("端口 eps/mu 不能为零")
        self.layers = list(layers)
        self.period = float(period)
        if self.period <= 0:
            raise ValueError("period 必须为正")


# =============================================================================
# 模式与 S 矩阵
# =============================================================================
def _branch(q):
    """选择稳定分支：倏逝模 Re(q)>=0，传播模 Im(q)<=0。"""
    q = np.where(q.real < 0, -q, q)
    return np.where((q.real < 1e-10) & (q.imag > 0), -q, q)


def _homog_modes(eps, mu, kx_over_k0, pol):
    N = len(kx_over_k0)
    kz_k0 = np.sqrt((eps * mu - np.asarray(kx_over_k0) ** 2).astype(complex))
    kz_k0 = np.where(kz_k0.imag < 0, -kz_k0, kz_k0)
    kz_k0 = np.where((abs(kz_k0.imag) < 1e-12) & (kz_k0.real < 0), -kz_k0, kz_k0)
    q = -1j * kz_k0
    W = np.eye(N, dtype=complex)
    if pol == "TE":
        V = np.diag(q / mu)
    else:
        V = np.diag(q / eps)
    return W, V, q


def _grating_modes(Eps, Aeps, Mu, Amu, Kx, pol):
    N = Kx.shape[0]
    if pol == "TE":
        # 与 TM 的 Li 因子化保持电磁对偶：横向磁场分量使用
        # <mu>^{-1}，z 向导数使用 <1/mu>。先写物理二阶矩阵，
        # 再左乘 <1/mu>^{-1}，保证截断后的状态方程自洽。
        Mui = np.linalg.inv(Mu)
        A0 = Kx @ Mui @ Kx - Eps
        A = np.linalg.solve(Amu, A0)
        w2, W = np.linalg.eig(A)
        q = _branch(np.sqrt(w2.astype(complex)))
        V = Amu @ W @ np.diag(q)
    else:
        # 与非磁性分支相同的 Li 因子化，并将 mu 项推广为 <mu>。
        Ei = np.linalg.inv(Eps)
        A = np.linalg.solve(Aeps, Kx @ Ei @ Kx - Mu)
        w2, W = np.linalg.eig(A)
        q = _branch(np.sqrt(w2.astype(complex)))
        V = Aeps @ W @ np.diag(q)
    return W, V, q


class S:
    """[out_down; out_up] = S [in_down; in_up]。"""

    __slots__ = ("Tdd", "Rdu", "Rud", "Tuu", "diag")

    def __init__(self, Tdd, Rdu, Rud, Tuu, diag=False):
        self.Tdd, self.Rdu, self.Rud, self.Tuu, self.diag = Tdd, Rdu, Rud, Tuu, diag

    def dense(self):
        if not self.diag:
            return self
        return S(np.diag(self.Tdd), np.diag(self.Rdu), np.diag(self.Rud), np.diag(self.Tuu))


def s_interface(W1, V1, W2, V2, both_homog=False, q1=None, q2=None,
                eps1=None, mu1=None, eps2=None, mu2=None, pol="TE"):
    """层 1（上）到层 2（下）的界面散射矩阵。"""
    if both_homog:
        p1 = q1 / (mu1 if pol == "TE" else eps1)
        p2 = q2 / (mu2 if pol == "TE" else eps2)
        Nn = p1 / p2
        Tp, Tm = 0.5 * (1 + Nn), 0.5 * (1 - Nn)
        Tpi = 1.0 / Tp
        return S(Tp - Tm * Tpi * Tm, Tm * Tpi, -Tpi * Tm, Tpi, diag=True)

    Mm = np.linalg.solve(W2, W1)
    Nn = np.linalg.solve(V2, V1)
    Tp, Tm = 0.5 * (Mm + Nn), 0.5 * (Mm - Nn)
    Tpi = np.linalg.inv(Tp)
    return S(Tp - Tm @ Tpi @ Tm, Tm @ Tpi, -Tpi @ Tm, Tpi)


def s_propagate(q, d, k0):
    X = np.exp(-k0 * q * d)
    Z = np.zeros_like(X)
    return S(X, Z, Z, X, diag=True)


def star(A, B):
    """A 在上、B 在下的 Redheffer 星乘。"""
    if A.diag and B.diag:
        G = 1.0 / (1.0 - A.Rdu * B.Rud)
        return S(B.Tdd * G * A.Tdd,
                 B.Rdu + B.Tdd * G * A.Rdu * B.Tuu,
                 A.Rud + A.Tuu * B.Rud * G * A.Tdd,
                 A.Tuu * G * B.Tuu, diag=True)
    A, B = A.dense(), B.dense()
    N = A.Tdd.shape[0]
    I = np.eye(N, dtype=complex)
    G = np.linalg.inv(I - A.Rdu @ B.Rud)
    H = np.linalg.inv(I - B.Rud @ A.Rdu)
    return S(B.Tdd @ G @ A.Tdd,
             B.Rdu + B.Tdd @ G @ A.Rdu @ B.Tuu,
             A.Rud + A.Tuu @ B.Rud @ G @ A.Tdd,
             A.Tuu @ H @ B.Tuu)


def _apply(m, v):
    return m * v if np.ndim(m) == 1 else m @ v


def _rud_below(blk, rud):
    if blk.diag and (rud is None or np.ndim(rud) == 1):
        r = 0.0 if rud is None else rud
        G = 1.0 / (1.0 - blk.Rdu * r)
        return blk.Rud + blk.Tuu * r * G * blk.Tdd
    A = blk.dense()
    N = A.Tdd.shape[0]
    r = np.zeros((N, N), dtype=complex) if rud is None else (
        np.diag(rud) if np.ndim(rud) == 1 else rud)
    G = np.linalg.inv(np.eye(N, dtype=complex) - A.Rdu @ r)
    return A.Rud + A.Tuu @ r @ G @ A.Tdd


def _step_down(blk, rud_next, down):
    if blk.diag and (rud_next is None or np.ndim(rud_next) == 1):
        r = 0.0 if rud_next is None else rud_next
        return blk.Tdd * down / (1.0 - blk.Rdu * r)
    A = blk.dense()
    N = A.Tdd.shape[0]
    r = np.zeros((N, N), dtype=complex) if rud_next is None else (
        np.diag(rud_next) if np.ndim(rud_next) == 1 else rud_next)
    return np.linalg.solve(np.eye(N, dtype=complex) - A.Rdu @ r, A.Tdd @ down)


class Result(dict):
    __getattr__ = dict.get


def _port_p(kz, eps, mu, pol):
    """与 S 状态振幅对应的无量纲 z 向功率因子。"""
    return kz / (mu if pol == "TE" else eps)


def solve(stack, lam0, theta_deg, M, pol="TE", need_fields=True):
    """求解并返回 R/T 与各衍射级效率。"""
    pol = pol.upper()
    if pol not in ("TE", "TM"):
        raise ValueError("pol 必须为 TE 或 TM")
    if lam0 <= 0 or int(M) != M or M < 0:
        raise ValueError("lam0 必须为正，M 必须为非负整数")
    M = int(M)

    k0 = 2 * np.pi / lam0
    th = np.radians(theta_deg)
    ms = np.arange(-M, M + 1)
    N = len(ms)
    nI, nII, P = stack.nI, stack.nII, stack.period
    kxm = k0 * (nI * np.sin(th) - ms * lam0 / P)
    kx_k0 = kxm / k0
    Kx = np.diag(kx_k0)

    WI, VI, qI = _homog_modes(stack.epsI, stack.muI, kx_k0, pol)
    WII, VII, qII = _homog_modes(stack.epsII, stack.muII, kx_k0, pol)

    mods, homog, mats = [], [], []
    for L in stack.layers:
        if L.homogeneous:
            mods.append(_homog_modes(L.eps, L.mu, kx_k0, pol))
            homog.append(True)
            mats.append((L.eps, L.mu))
        else:
            ops = L.operators(P, M)
            mods.append(_grating_modes(*ops, Kx, pol))
            homog.append(False)
            mats.append((None, None))

    blocks = []
    Wp, Vp, qp = WI, VI, qI
    ep, mup, hp = stack.epsI, stack.muI, True
    for (W, V, q), L, hg, (e2, m2) in zip(mods, stack.layers, homog, mats):
        blocks.append(s_interface(Wp, Vp, W, V, both_homog=(hp and hg),
                                  q1=qp, q2=q, eps1=ep, mu1=mup,
                                  eps2=e2, mu2=m2, pol=pol))
        blocks.append(s_propagate(q, L.d, k0))
        Wp, Vp, qp, ep, mup, hp = W, V, q, e2, m2, hg
    blocks.append(s_interface(Wp, Vp, WII, VII, both_homog=hp,
                              q1=qp, q2=qII, eps1=ep, mu1=mup,
                              eps2=stack.epsII, mu2=stack.muII, pol=pol))

    nb = len(blocks)
    rud = [None] * (nb + 1)
    for k in range(nb - 1, -1, -1):
        rud[k] = _rud_below(blocks[k], rud[k + 1])

    dlt = (ms == 0).astype(complex)
    downs = [dlt]
    for k in range(nb):
        downs.append(_step_down(blocks[k], rud[k + 1], downs[-1]))
    R = _apply(rud[0], dlt)
    T = downs[nb]

    kIz, kIIz = 1j * k0 * qI, 1j * k0 * qII
    pI = _port_p(kIz / k0, stack.epsI, stack.muI, pol)
    pII = _port_p(kIIz / k0, stack.epsII, stack.muII, pol)
    den = pI[ms == 0][0]
    DEr = np.abs(R) ** 2 * np.real(pI / den)
    DEt = np.abs(T) ** 2 * np.real(pII / den)

    out = Result(ms=ms, R=R, T=T, DEr=DEr, DEt=DEt, kxm=kxm,
                 kIz=kIz, kIIz=kIIz, k0=k0, pol=pol, Kx=Kx, stack=stack,
                 M=M, theta=th, pI=pI, pII=pII, epsI=stack.epsI,
                 muI=stack.muI, epsII=stack.epsII, muII=stack.muII)
    out["fourier_layers"] = [layer_fourier_info(L,P,M) for L in stack.layers]
    if not need_fields:
        return out

    amps, ztop = [], 0.0
    for j, L in enumerate(stack.layers):
        ap = downs[2 * j + 1]
        am = _apply(rud[2 * j + 2], downs[2 * j + 2])
        amps.append(dict(a_plus=ap, a_minus=am, W=mods[j][0], V=mods[j][1],
                         q=mods[j][2], d=L.d, z_top=ztop, layer=L))
        ztop += L.d
    out["amps"] = amps
    out["z_bot"] = ztop
    out["WI"], out["VI"], out["qI"] = WI, VI, qI
    out["WII"], out["VII"], out["qII"] = WII, VII, qII
    return out


# =============================================================================
# 场重建
# =============================================================================
def field_at(res, z):
    """返回 z 处的谐波状态列 ``(S,U)``。"""
    k0 = res["k0"]
    dlt = (res["ms"] == 0).astype(complex)
    if z < 0:
        e_in = np.exp(1j * res["kIz"] * z)
        e_re = np.exp(-1j * res["kIz"] * z)
        Sv = dlt * e_in + res["R"] * e_re
        p = res["kIz"] / k0 / (res["muI"] if res["pol"] == "TE" else res["epsI"])
        return Sv, 1j * p * (dlt * e_in - res["R"] * e_re)
    if z > res["z_bot"]:
        e = np.exp(1j * res["kIIz"] * (z - res["z_bot"]))
        p = res["kIIz"] / k0 / (res["muII"] if res["pol"] == "TE" else res["epsII"])
        return res["T"] * e, 1j * p * res["T"] * e

    for a in res["amps"]:
        if z <= a["z_top"] + a["d"] + 1e-12:
            zl = z - a["z_top"]
            e1 = np.exp(-k0 * a["q"] * zl)
            e2 = np.exp(-k0 * a["q"] * (a["d"] - zl))
            Sv = a["W"] @ (e1 * a["a_plus"] + e2 * a["a_minus"])
            Uv = a["V"] @ (-e1 * a["a_plus"] + e2 * a["a_minus"])
            return Sv, Uv
    # 允许空层堆栈在 z=0 处直接返回端口场。
    return dlt, 1j * res["pI"] * dlt


def _eps_mu_ops_at(res, z):
    """返回 ``(<eps>, <1/eps>, <mu>, <1/mu>)`` 的局部算符。"""
    M, P = res["M"], res["stack"].period
    N = 2 * M + 1
    if z < 0:
        I = np.eye(N, dtype=complex)
        return res["epsI"] * I, I / res["epsI"], res["muI"] * I, I / res["muI"]
    if z > res["z_bot"]:
        I = np.eye(N, dtype=complex)
        return res["epsII"] * I, I / res["epsII"], res["muII"] * I, I / res["muII"]
    for a in res["amps"]:
        if z <= a["z_top"] + a["d"] + 1e-12:
            L = a["layer"]
            if L.homogeneous:
                I = np.eye(N, dtype=complex)
                return L.eps * I, I / L.eps, L.mu * I, I / L.mu
            return L.operators(P, M)
    I = np.eye(N, dtype=complex)
    return res["epsII"] * I, I / res["epsII"], res["muII"] * I, I / res["muII"]


def field_map(res, zs, nx=256):
    """在 (x,z) 网格重建场。

    TE 返回 ``Ey,Hx,Hz``；TM 返回 ``Hy,Ex,Ez``。这些分量均使用与
    原求解器相同的无量纲场约定，另附 ``E2``、``x``、``z``。
    """
    P = res["stack"].period
    x = np.arange(nx) / nx * P
    ph = np.exp(-1j * np.outer(x, res["kxm"]))
    zs = np.atleast_1d(zs).astype(float)
    keys = ("Ey", "Hx", "Hz") if res["pol"] == "TE" else ("Hy", "Ex", "Ez")
    comp = {k: np.zeros((len(zs), nx), dtype=complex) for k in keys}
    for i, z in enumerate(zs):
        Sv, Uv = field_at(res, z)
        if res["pol"] == "TE":
            _, _, Mu, _ = _eps_mu_ops_at(res, z)
            comp["Ey"][i] = ph @ Sv
            comp["Hx"][i] = ph @ (1j * Uv)
            # Hz=-<mu>^{-1}Kx Ey；相应的法向 Bz 恢复为 -Kx Ey。
            comp["Hz"][i] = ph @ (-np.linalg.solve(Mu, res["Kx"] @ Sv))
        else:
            Eps, _, _, _ = _eps_mu_ops_at(res, z)
            comp["Hy"][i] = ph @ Sv
            comp["Ex"][i] = ph @ (-1j * Uv)
            comp["Ez"][i] = ph @ np.linalg.solve(Eps, res["Kx"] @ Sv)
    if res["pol"] == "TE":
        comp["E2"] = np.abs(comp["Ey"]) ** 2
    else:
        comp["E2"] = np.abs(comp["Ex"]) ** 2 + np.abs(comp["Ez"]) ** 2
    comp["x"], comp["z"] = x, zs
    return comp


def eps_mu_profile(res, zs, nx=256):
    """返回与 ``field_map`` 对齐的 ``eps(x,z)`` 与 ``mu(x,z)``。"""
    P = res["stack"].period
    x = np.arange(nx) / nx * P
    zs = np.atleast_1d(zs).astype(float)
    eps_out = np.zeros((len(zs), nx), dtype=complex)
    mu_out = np.zeros_like(eps_out)
    for i, z in enumerate(zs):
        if z < 0:
            eps_out[i] = res["epsI"]
            mu_out[i] = res["muI"]
        elif z > res["z_bot"]:
            eps_out[i] = res["epsII"]
            mu_out[i] = res["muII"]
        else:
            for a in res["amps"]:
                if z <= a["z_top"] + a["d"] + 1e-12:
                    e, m = a["layer"].profile(x, P)
                    eps_out[i], mu_out[i] = e, m
                    break
    return x, zs, eps_out, mu_out


def eps_profile(res, zs, nx=256):
    """兼容旧调用：返回等效局部折射率 ``sqrt(eps*mu)``。"""
    x, zs, eps, mu = eps_mu_profile(res, zs, nx)
    return x, np.sqrt(eps * mu + 0j)
