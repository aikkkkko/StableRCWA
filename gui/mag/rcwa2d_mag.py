# -*- coding: utf-8 -*-
"""二维各向同性磁性介质 RCWA（实验分支）。

矩形二维周期结构，材料可为分段常数 ``eps(x,y)``、``mu(x,y)``。采用
切向状态 ``[Ex, Ey, hx, hy]``，其中 ``h=i Z0 H``；S 矩阵级联和场重建
与上级 ``2drcwa/rcwa2d.py`` 兼容，但不修改其非磁性实现。

对于一般磁性介质，二维 Li 因子化需要两组材料算符：

* ``Eps, Cx, Cy``：介电常数的 Laurent / x、y 方向逆规则；
* ``Mu, Mx, My``：磁导率的完全对偶的一组算符。

这使得在 ``mu=1`` 时严格退化到原 2D 方程，并满足
``TE(eps,mu) <-> TM(mu,eps)`` 的电磁对偶。当前范围是标量、各向同性
``eps``/``mu``；不包含各向异性、磁电耦合或非互易张量材料。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "Lattice", "Layer2D", "Stack2D", "solve2d", "field_at", "field_xy",
    "eps_mu_map", "eps_map", "index_of", "fourier_1d", "toeplitz_1d",
]


# =============================================================================
# Fourier 与材料几何
# =============================================================================
def fourier_1d(edges, values, period, hmax):
    """分段常数一维剖面的解析 Fourier 系数。"""
    edges = np.asarray(edges, dtype=float)
    values = np.asarray(values, dtype=complex)
    if edges.ndim != 1 or values.ndim != 1 or len(edges) != len(values) + 1:
        raise ValueError("edges 必须比 values 多一个元素")
    if period <= 0 or not np.isfinite(edges).all():
        raise ValueError("period/edges 无效")
    if abs((edges[-1] - edges[0]) - period) > 1e-9 * max(1.0, abs(period)):
        raise ValueError("edges 首尾必须相差一个 period")
    h = np.arange(-hmax, hmax + 1)
    c = np.zeros(len(h), dtype=complex)
    c[hmax] = np.sum(values * np.diff(edges)) / period
    hn = h[h != 0]
    ph = np.exp(-2j * np.pi * np.outer(hn, edges) / period)
    c[h != 0] = (values * (ph[:, 1:] - ph[:, :-1])).sum(axis=1) / (-2j * np.pi * hn)
    return c


def toeplitz_1d(c, M):
    """由 ``c[h+2M]`` 生成 ``(2M+1)`` 阶 Fourier 卷积阵。"""
    idx = np.arange(-M, M + 1)
    return c[np.subtract.outer(idx, idx) + 2 * M]


def _seg_fourier(edges, hmax, period):
    """每个分段对 Fourier 系数的贡献，形状 ``(segments, harmonics)``。"""
    edges = np.asarray(edges, dtype=float)
    h = np.arange(-hmax, hmax + 1)
    out = np.zeros((len(edges) - 1, len(h)), dtype=complex)
    out[:, hmax] = np.diff(edges) / period
    hn = h[h != 0]
    ph = np.exp(-2j * np.pi * np.outer(hn, edges) / period)
    out[:, h != 0] = ((ph[:, 1:] - ph[:, :-1]) / (-2j * np.pi * hn)[:, None]).T
    return out


def _profile_value(profile, x, period):
    edges, values = profile
    x0 = edges[0]
    xp = x0 + np.mod(np.asarray(x, dtype=float) - x0, period)
    idx = np.searchsorted(edges, xp, side="right") - 1
    idx = np.clip(idx, 0, len(values) - 1)
    return values[idx]


def _coerce_profiles(profiles, rows, label):
    if len(profiles) != rows:
        raise ValueError(f"{label} 的长度必须与 y 分段数相同")
    out = []
    for j, item in enumerate(profiles):
        if len(item) != 2:
            raise ValueError(f"{label}[{j}] 必须是 (x_edges, values)")
        edges = np.asarray(item[0], dtype=float)
        values = np.asarray(item[1], dtype=complex)
        if edges.ndim != 1 or values.ndim != 1 or len(edges) != len(values) + 1:
            raise ValueError(f"{label}[{j}] 的 x_edges/values 长度不匹配")
        if len(edges) < 2 or np.any(np.diff(edges) <= 0):
            raise ValueError(f"{label}[{j}] 的 x_edges 必须严格递增")
        if not np.isfinite(edges).all() or not np.isfinite(values).all() or np.any(values == 0):
            raise ValueError(f"{label}[{j}] 含有零或非有限材料参数")
        out.append((edges, values))
    return out


def _constant_profiles_like(profiles, value):
    return [(edges.copy(), np.full(len(values), complex(value), dtype=complex))
            for edges, values in profiles]


def _n_mu_to_eps(n_profiles, mu_profiles):
    """将任意 x 分段的 n、mu 剖面相交，生成 eps=n²/mu 剖面。"""
    out = []
    for (ne, nv), (me, mv) in zip(n_profiles, mu_profiles):
        pn, pm = ne[-1] - ne[0], me[-1] - me[0]
        if abs(pn - pm) > 1e-9 * max(1.0, abs(pn), abs(pm)):
            raise ValueError("n_profiles 与 mu_profiles 的 x 周期不一致")
        cuts = np.unique(np.round(np.concatenate([ne, me]), 12))
        if len(cuts) < 2:
            raise ValueError("空的 x 剖面")
        mid = .5 * (cuts[:-1] + cuts[1:])
        vals = _profile_value((ne, nv), mid, pn) ** 2 / _profile_value((me, mv), mid, pm)
        out.append((cuts, vals))
    return out


def _validate_periodic_profiles(y_edges, profiles, lat, label):
    if abs((y_edges[-1] - y_edges[0]) - lat.Ly) > 1e-9 * max(1.0, abs(lat.Ly)):
        raise ValueError("y_edges 的周期必须等于 Lattice.Ly")
    for edges, _ in profiles:
        if abs((edges[-1] - edges[0]) - lat.Lx) > 1e-9 * max(1.0, abs(lat.Lx)):
            raise ValueError(f"{label} 的 x_edges 周期必须等于 Lattice.Lx")


class Lattice:
    """矩形格子。"""

    def __init__(self, Lx, Ly):
        self.Lx, self.Ly = float(Lx), float(Ly)
        if self.Lx <= 0 or self.Ly <= 0:
            raise ValueError("Lx 和 Ly 必须为正")

    def harmonics(self, M1, M2):
        if int(M1) != M1 or int(M2) != M2 or M1 < 0 or M2 < 0:
            raise ValueError("M1/M2 必须为非负整数")
        M1, M2 = int(M1), int(M2)
        m, n = np.meshgrid(np.arange(-M1, M1 + 1), np.arange(-M2, M2 + 1), indexing="ij")
        return np.column_stack([m.ravel(), n.ravel()]), M1, M2


class Layer2D:
    """二维各向同性层。

    均匀层：

    ``Layer2D(d, n=1.5, mu=1.2)`` 或 ``Layer2D(d, eps=2.0, mu=1.3)``。

    周期层使用 ``y_edges`` 与每个 y 分段对应的 x 剖面。旧接口
    ``x_profiles=[(x_edges, n_values), ...]`` 仍然支持。磁性材料建议直接给：

    ``eps_profiles=[(x_edges, eps_values), ...]``
    ``mu_profiles =[(x_edges, mu_values ), ...]``

    若给 ``x_profiles``，则按局部 ``eps=n**2/mu`` 推导；``mu_profiles``
    可使用不同的 x 分段，代码会自动取分段交集。
    """

    def __init__(self, thickness, n=None, mu=1.0, eps=None,
                 y_edges=None, x_profiles=None, eps_profiles=None,
                 mu_profiles=None, name=""):
        self.d = float(thickness)
        if self.d < 0:
            raise ValueError("层厚度不能为负")
        self.name = name
        homogeneous_args = n is not None or eps is not None
        periodic_args = any(v is not None for v in (y_edges, x_profiles, eps_profiles, mu_profiles))
        if homogeneous_args and periodic_args:
            raise ValueError("均匀层参数与周期层参数不能混用")

        self.homogeneous = homogeneous_args
        if self.homogeneous:
            if n is not None and eps is not None:
                raise ValueError("均匀层请在 n 与 eps 中二选一")
            self.mu = complex(mu)
            if self.mu == 0:
                raise ValueError("mu 不能为零")
            self.eps = complex(eps) if eps is not None else complex(n) ** 2 / self.mu
            if self.eps == 0 or self.mu == 0:
                raise ValueError("eps/mu 不能为零")
            self.n = complex(n) if n is not None else np.sqrt(self.eps * self.mu + 0j)
            self.y_edges = self.x_profiles = self.eps_profiles = self.mu_profiles = None
            return

        if y_edges is None or (x_profiles is None and eps_profiles is None):
            raise ValueError("周期层必须给 y_edges 与 x_profiles 或 eps_profiles")
        if x_profiles is not None and eps_profiles is not None:
            raise ValueError("周期层请在 x_profiles(n) 与 eps_profiles 中二选一")
        self.y_edges = np.asarray(y_edges, dtype=float)
        if self.y_edges.ndim != 1 or len(self.y_edges) < 2 or np.any(np.diff(self.y_edges) <= 0):
            raise ValueError("y_edges 必须严格递增")
        rows = len(self.y_edges) - 1

        if eps_profiles is not None:
            self.eps_profiles = _coerce_profiles(eps_profiles, rows, "eps_profiles")
            self.x_profiles = None
            if complex(mu) == 0 and mu_profiles is None:
                raise ValueError("mu 不能为零")
            self.mu_profiles = (_constant_profiles_like(self.eps_profiles, mu)
                                if mu_profiles is None else _coerce_profiles(mu_profiles, rows, "mu_profiles"))
        else:
            self.x_profiles = _coerce_profiles(x_profiles, rows, "x_profiles")
            if complex(mu) == 0 and mu_profiles is None:
                raise ValueError("mu 不能为零")
            self.mu_profiles = (_constant_profiles_like(self.x_profiles, mu)
                                if mu_profiles is None else _coerce_profiles(mu_profiles, rows, "mu_profiles"))
            self.eps_profiles = _n_mu_to_eps(self.x_profiles, self.mu_profiles)
        self.eps = self.mu = self.n = None

    def operators(self, lat, mn, M1, M2, factorization="li"):
        """返回 ``Eps,Cx,Cy,Mu,Mx,My`` 六个二维材料算符。"""
        if factorization not in ("li", "laurent"):
            raise ValueError("factorization 必须为 li 或 laurent")
        if self.homogeneous:
            N = len(mn)
            I = np.eye(N, dtype=complex)
            return self.eps * I, self.eps * I, self.eps * I, self.mu * I, self.mu * I, self.mu * I
        _validate_periodic_profiles(self.y_edges, self.eps_profiles, lat, "eps_profiles")
        _validate_periodic_profiles(self.y_edges, self.mu_profiles, lat, "mu_profiles")
        Eps, Cx, Cy = _scalar_operators(self.y_edges, self.eps_profiles, lat, mn, M1, M2, factorization)
        Mu, Mx, My = _scalar_operators(self.y_edges, self.mu_profiles, lat, mn, M1, M2, factorization)
        return Eps, Cx, Cy, Mu, Mx, My

    def profile(self, x, y, lat):
        """返回实空间 ``eps(x,y),mu(x,y)``。"""
        xx = np.asarray(x, dtype=float)
        yy = float(y)
        if self.homogeneous:
            return (np.full(xx.shape, self.eps, dtype=complex),
                    np.full(xx.shape, self.mu, dtype=complex))
        y0 = self.y_edges[0]
        yp = y0 + np.mod(yy - y0, lat.Ly)
        row = int(np.clip(np.searchsorted(self.y_edges, yp, side="right") - 1,
                          0, len(self.y_edges) - 2))
        return (_profile_value(self.eps_profiles[row], xx, lat.Lx),
                _profile_value(self.mu_profiles[row], xx, lat.Lx))


def _scalar_operators(y_edges, profiles, lat, mn, M1, M2, factorization):
    """对一个标量材料函数构建 Laurent 与嵌套 Li 算符。"""
    Lx, Ly = lat.Lx, lat.Ly
    wseg = _seg_fourier(y_edges, 2 * M2, Ly)
    n1 = 2 * M1 + 1
    eps_h = np.zeros((4 * M2 + 1, n1, n1), dtype=complex)
    cx_h = np.zeros_like(eps_h)

    for k, (xe, values) in enumerate(profiles):
        ce = fourier_1d(xe, values, Lx, 2 * M1)
        lau = toeplitz_1d(ce, M1)
        eps_h += wseg[k][:, None, None] * lau
        if factorization == "li":
            ci = fourier_1d(xe, 1.0 / values, Lx, 2 * M1)
            cx_h += wseg[k][:, None, None] * np.linalg.inv(toeplitz_1d(ci, M1))
        else:
            cx_h += wseg[k][:, None, None] * lau

    Eps = _assemble_xfirst(eps_h, mn, M1, M2)
    Cx = _assemble_xfirst(cx_h, mn, M1, M2)
    if factorization == "laurent":
        return Eps, Eps.copy(), Eps.copy()

    xcuts = np.unique(np.concatenate([np.asarray(xe, dtype=float) % Lx for xe, _ in profiles]
                                      + [np.array([0.0, Lx])]))
    xcuts = np.unique(np.round(xcuts, 12))
    if xcuts[0] > 1e-12:
        xcuts = np.concatenate([[0.0], xcuts])
    if xcuts[-1] < Lx - 1e-12:
        xcuts = np.concatenate([xcuts, [Lx]])
    xmid = .5 * (xcuts[:-1] + xcuts[1:])
    wsegx = _seg_fourier(xcuts, 2 * M1, Lx)
    n2 = 2 * M2 + 1
    cy_h = np.zeros((4 * M1 + 1, n2, n2), dtype=complex)
    for j, xm in enumerate(xmid):
        column = np.array([_profile_value(profile, xm, Lx) for profile in profiles], dtype=complex)
        cy = fourier_1d(y_edges, 1.0 / column, Ly, 2 * M2)
        cy_h += wsegx[j][:, None, None] * np.linalg.inv(toeplitz_1d(cy, M2))
    Cy = _assemble_yfirst(cy_h, mn, M1, M2)
    return Eps, Cx, Cy


def _assemble_xfirst(blocks, mn, M1, M2):
    dn = mn[:, 1][:, None] - mn[:, 1][None, :] + 2 * M2
    im = mn[:, 0] + M1
    return blocks[dn, im[:, None], im[None, :]]


def _assemble_yfirst(blocks, mn, M1, M2):
    dm = mn[:, 0][:, None] - mn[:, 0][None, :] + 2 * M1
    inn = mn[:, 1] + M2
    return blocks[dm, inn[:, None], inn[None, :]]


class Stack2D:
    """``nI | layers... | nII``，端口可显式给相对 eps/mu。"""

    def __init__(self, nI, layers, nII, lattice, muI=1.0, muII=1.0,
                 epsI=None, epsII=None):
        self.nI, self.nII = complex(nI), complex(nII)
        self.muI, self.muII = complex(muI), complex(muII)
        if self.muI == 0 or self.muII == 0:
            raise ValueError("端口 mu 不能为零")
        self.epsI = complex(epsI) if epsI is not None else self.nI ** 2 / self.muI
        self.epsII = complex(epsII) if epsII is not None else self.nII ** 2 / self.muII
        if any(v == 0 for v in (self.epsI, self.epsII, self.muI, self.muII)):
            raise ValueError("端口 eps/mu 不能为零")
        self.layers, self.lat = list(layers), lattice


# =============================================================================
# Maxwell 模式与 S 矩阵
# =============================================================================
def _branch(q):
    q = np.where(q.real < 0, -q, q)
    return np.where((q.real < 1e-10) & (q.imag > 0), -q, q)


def _kz_over_k0(eps, mu, Kx, Ky):
    kz = np.sqrt((eps * mu - np.asarray(Kx) ** 2 - np.asarray(Ky) ** 2).astype(complex))
    kz = np.where(kz.imag < 0, -kz, kz)
    return np.where((np.abs(kz.imag) < 1e-12) & (kz.real < 0), -kz, kz)


def _blk(a11, a12, a21, a22):
    return np.block([[a11, a12], [a21, a22]])


def bmul(A, B):
    return np.einsum("ikn,kjn->ijn", A, B)


def binv(A):
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    return np.array([[A[1, 1], -A[0, 1]], [-A[1, 0], A[0, 0]]]) / det


def beye(N):
    z, o = np.zeros(N, dtype=complex), np.ones(N, dtype=complex)
    return np.array([[o, z], [z, o]])


def bzero(N):
    z = np.zeros(N, dtype=complex)
    return np.array([[z, z], [z, z]])


def bdense(A):
    D = np.diag
    return _blk(D(A[0, 0]), D(A[0, 1]), D(A[1, 0]), D(A[1, 1]))


def bapply(A, v):
    N = A.shape[2]
    vx, vy = v[:N], v[N:]
    if v.ndim == 2:
        return np.concatenate([A[0, 0][:, None] * vx + A[0, 1][:, None] * vy,
                               A[1, 0][:, None] * vx + A[1, 1][:, None] * vy])
    return np.concatenate([A[0, 0] * vx + A[0, 1] * vy,
                           A[1, 0] * vx + A[1, 1] * vy])


def _isbd(value):
    return isinstance(value, np.ndarray) and value.ndim == 3


class S2:
    __slots__ = ("Tdd", "Rdu", "Rud", "Tuu", "bd")

    def __init__(self, Tdd, Rdu, Rud, Tuu, bd=False):
        self.Tdd, self.Rdu, self.Rud, self.Tuu, self.bd = Tdd, Rdu, Rud, Tuu, bd

    def dense(self):
        if not self.bd:
            return self
        return S2(bdense(self.Tdd), bdense(self.Rdu), bdense(self.Rud), bdense(self.Tuu))


def s_interface2(W1, V1, W2, V2, bd=False):
    if bd:
        Mm, Nn = bmul(binv(W2), W1), bmul(binv(V2), V1)
        Tp, Tm = .5 * (Mm + Nn), .5 * (Mm - Nn)
        Tpi = binv(Tp)
        return S2(Tp - bmul(bmul(Tm, Tpi), Tm), bmul(Tm, Tpi),
                  -bmul(Tpi, Tm), Tpi, bd=True)
    Mm, Nn = np.linalg.solve(W2, W1), np.linalg.solve(V2, V1)
    Tp, Tm = .5 * (Mm + Nn), .5 * (Mm - Nn)
    Tpi = np.linalg.inv(Tp)
    return S2(Tp - Tm @ Tpi @ Tm, Tm @ Tpi, -Tpi @ Tm, Tpi)


def s_propagate2(q, d, k0):
    N = len(q) // 2
    x = np.exp(-k0 * q * d)
    z = np.zeros(N, dtype=complex)
    X = np.array([[x[:N], z], [z, x[N:]]])
    return S2(X, bzero(N), bzero(N), X, bd=True)


def _rud2(blk, rud):
    if blk.bd and (rud is None or _isbd(rud)):
        N = blk.Tdd.shape[2]
        r = bzero(N) if rud is None else rud
        G = binv(beye(N) - bmul(blk.Rdu, r))
        return blk.Rud + bmul(bmul(bmul(blk.Tuu, r), G), blk.Tdd)
    A = blk.dense()
    n = A.Tdd.shape[0]
    r = (np.zeros((n, n), dtype=complex) if rud is None
         else (bdense(rud) if _isbd(rud) else rud))
    G = np.linalg.inv(np.eye(n, dtype=complex) - A.Rdu @ r)
    return A.Rud + A.Tuu @ r @ G @ A.Tdd


def _step2(blk, rud_next, down):
    if blk.bd and (rud_next is None or _isbd(rud_next)):
        N = blk.Tdd.shape[2]
        r = bzero(N) if rud_next is None else rud_next
        G = binv(beye(N) - bmul(blk.Rdu, r))
        return bapply(bmul(G, blk.Tdd), down)
    A = blk.dense()
    n = A.Tdd.shape[0]
    r = (np.zeros((n, n), dtype=complex) if rud_next is None
         else (bdense(rud_next) if _isbd(rud_next) else rud_next))
    return np.linalg.solve(np.eye(n, dtype=complex) - A.Rdu @ r, A.Tdd @ down)


def _applyR(rud, value):
    return bapply(rud, value) if _isbd(rud) else rud @ value


def _homog_modes2d(eps, mu, Kx, Ky):
    """均匀端口/层的解析模式，保持 2x2 分块对角快路径。"""
    N = len(Kx)
    kz = _kz_over_k0(eps, mu, Kx, Ky)
    q1 = _branch(-1j * kz)
    q = np.concatenate([q1, q1])
    mui = 1.0 / mu
    Qb = np.array([[Kx * mui * Ky, eps - Kx * mui * Kx],
                   [Ky * mui * Ky - eps, -Ky * mui * Kx]], dtype=complex)
    Vb = Qb / q1[None, None, :]
    return beye(N), Vb, q, kz


def _grating_modes2d(Eps, Cx, Cy, Mu, Mx, My, Kx, Ky):
    """一般 2D 磁性层的 2N 阶二阶本征问题。"""
    D = np.diag
    KxD, KyD = D(Kx), D(Ky)
    Ei, Mui = np.linalg.inv(Eps), np.linalg.inv(Mu)
    P = _blk(KxD @ Ei @ KyD, My - KxD @ Ei @ KxD,
             KyD @ Ei @ KyD - Mx, -KyD @ Ei @ KxD)
    Q = _blk(KxD @ Mui @ KyD, Cy - KxD @ Mui @ KxD,
             KyD @ Mui @ KyD - Cx, -KyD @ Mui @ KxD)
    w2, W = np.linalg.eig(P @ Q)
    q = _branch(np.sqrt(w2.astype(complex)))
    V = (Q @ W) / q[None, :]
    return W, V, q


class Result2DMag(dict):
    __getattr__ = dict.get


def _vertical_components(F, G, Ei, Mui, Kx, Ky):
    N = len(Kx)
    Ex, Ey, hx, hy = F[:N], F[N:], G[:N], G[N:]
    hz = 1j * (Mui @ (Ky * Ex - Kx * Ey))
    Ez = 1j * (Ei @ (Ky * hx - Kx * hy))
    return Ez, hz


def _z_flux(F, G, N):
    """每个谐波的 z 向时间平均 Poynting 因子；比例常数会在效率中消掉。"""
    Ex, Ey, hx, hy = F[:N], F[N:], G[:N], G[N:]
    Hx, Hy = -1j * hx, -1j * hy
    return np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx))


def solve2d(stack, lam0, theta_deg, phi_deg, M1, M2,
            psi_deg=90.0, delta_deg=0.0, jones_in=None,
            factorization="li", need_modes=False):
    """求解二维周期磁性结构。

    ``psi=90`` 为 s 入射、``psi=0`` 为 p 入射；当给出 ``jones_in`` 时，
    效率自动按该 Jones 向量的实际入射功率归一。
    """
    if lam0 <= 0:
        raise ValueError("lam0 必须为正")
    k0 = 2 * np.pi / lam0
    th, ph = np.radians(theta_deg), np.radians(phi_deg)
    lat, nI, nII = stack.lat, stack.nI, stack.nII
    mn, M1, M2 = lat.harmonics(M1, M2)
    N = len(mn)

    Kx = nI * np.sin(th) * np.cos(ph) - mn[:, 0] * lam0 / lat.Lx
    Ky = nI * np.sin(th) * np.sin(ph) - mn[:, 1] * lam0 / lat.Ly
    WI, VI, qI, kzI = _homog_modes2d(stack.epsI, stack.muI, Kx, Ky)
    WII, VII, qII, kzII = _homog_modes2d(stack.epsII, stack.muII, Kx, Ky)

    mods, homo, material = [], [], []
    for layer in stack.layers:
        if layer.homogeneous:
            mods.append(_homog_modes2d(layer.eps, layer.mu, Kx, Ky)[:3])
            homo.append(True)
            material.append((layer.eps, layer.mu, None, None, None, None, None, None))
        else:
            Eps, Cx, Cy, Mu, Mx, My = layer.operators(lat, mn, M1, M2, factorization)
            mods.append(_grating_modes2d(Eps, Cx, Cy, Mu, Mx, My, Kx, Ky))
            homo.append(False)
            material.append((None, None, Eps, Mu, Cx, Cy, Mx, My))

    def _mk(W1, V1, h1, W2, V2, h2):
        if h1 and h2:
            return s_interface2(W1, V1, W2, V2, bd=True)
        return s_interface2(bdense(W1) if h1 else W1, bdense(V1) if h1 else V1,
                            bdense(W2) if h2 else W2, bdense(V2) if h2 else V2)

    blocks = []
    Wp, Vp, hp = WI, VI, True
    for (W, V, q), layer, hg in zip(mods, stack.layers, homo):
        blocks.append(_mk(Wp, Vp, hp, W, V, hg))
        blocks.append(s_propagate2(q, layer.d, k0))
        Wp, Vp, hp = W, V, hg
    blocks.append(_mk(Wp, Vp, hp, WII, VII, True))

    rud = [None] * (len(blocks) + 1)
    for k in range(len(blocks) - 1, -1, -1):
        rud[k] = _rud2(blocks[k], rud[k + 1])

    if jones_in is not None:
        Es, Ep = complex(jones_in[0]), complex(jones_in[1])
    else:
        psi, delta = np.radians(psi_deg), np.radians(delta_deg)
        Es, Ep = np.sin(psi) * np.exp(1j * delta), np.cos(psi)
    i0 = int(np.where((mn[:, 0] == 0) & (mn[:, 1] == 0))[0][0])
    inc = np.zeros(2 * N, dtype=complex)
    inc[i0] = -Es * np.sin(ph) + Ep * np.cos(th) * np.cos(ph)
    inc[N + i0] = Es * np.cos(ph) + Ep * np.cos(th) * np.sin(ph)

    downs = [inc]
    for k, block in enumerate(blocks):
        downs.append(_step2(block, rud[k + 1], downs[-1]))
    R, T = _applyR(rud[0], inc), downs[-1]

    Ginc = bapply(VI, -inc)
    Gr = bapply(VI, R)
    Gt = bapply(VII, -T)
    incoming_flux = float(np.sum(_z_flux(inc, Ginc, N)))
    if not np.isfinite(incoming_flux) or incoming_flux <= 1e-13:
        raise ValueError("入射 Jones 向量在当前端口没有正向功率")
    propR = np.real(kzI) > 1e-12
    propT = np.real(kzII) > 1e-12
    fluxR, fluxT = _z_flux(R, Gr, N), _z_flux(T, Gt, N)
    DEr, DEt = np.zeros(N), np.zeros(N)
    DEr[propR] = -fluxR[propR] / incoming_flux
    DEt[propT] = fluxT[propT] / incoming_flux

    I = np.eye(N, dtype=complex)
    EiI, MuiI = I / stack.epsI, I / stack.muI
    EiII, MuiII = I / stack.epsII, I / stack.muII
    EzR, hzR = _vertical_components(R, Gr, EiI, MuiI, Kx, Ky)
    EzT, hzT = _vertical_components(T, Gt, EiII, MuiII, Kx, Ky)

    def sp(F, kz, n_med, sign):
        Ex, Ey = F[:N], F[N:]
        kt = np.sqrt(Kx ** 2 + Ky ** 2 + 0j)
        nonzero = np.abs(kt) > 1e-12
        safe_kt = np.where(nonzero, kt, 1.0)
        s = np.where(nonzero, (-Ky * Ex + Kx * Ey) / safe_kt,
                     -np.sin(ph) * Ex + np.cos(ph) * Ey)
        cth = np.where(np.abs(kz) > 1e-12, kz / n_med, 1.0)
        p = np.where(nonzero, sign * (Kx * Ex + Ky * Ey) / (safe_kt * cth),
                     sign * (np.cos(ph) * Ex + np.sin(ph) * Ey) / cth)
        return s, p

    Rs, Rp = sp(R, kzI, nI, -1.0)
    Ts, Tp = sp(T, kzII, nII, +1.0)
    out = Result2DMag(
        mn=mn, M1=M1, M2=M2, Kx=Kx, Ky=Ky, k0=k0, lat=lat, stack=stack,
        kzI=kzI, kzII=kzII, Ex_r=R[:N], Ey_r=R[N:], Ez_r=EzR,
        Hx_r=-1j * Gr[:N], Hy_r=-1j * Gr[N:], Hz_r=-1j * hzR,
        Ex_t=T[:N], Ey_t=T[N:], Ez_t=EzT,
        Hx_t=-1j * Gt[:N], Hy_t=-1j * Gt[N:], Hz_t=-1j * hzT,
        Rs=Rs, Rp=Rp, Ts=Ts, Tp=Tp, DEr=DEr, DEt=DEt,
        prop_r=propR, prop_t=propT, i0=i0, theta=th, phi=ph,
        nI=nI, nII=nII, epsI=stack.epsI, epsII=stack.epsII,
        muI=stack.muI, muII=stack.muII, N=N, inc=inc, R=R, T=T,
        VI=VI, VII=VII, qI=qI, qII=qII, incoming_flux=incoming_flux,
        energy=float(np.sum(DEr) + np.sum(DEt)),
    )

    amps, ztop = [], 0.0
    for j, layer in enumerate(stack.layers):
        eps, mu, Eps, Mu, Cx, Cy, Mx, My = material[j]
        if homo[j]:
            Ei, Mui = I / eps, I / mu
            Eps, Mu = eps * I, mu * I
            Cx = Cy = Eps
            Mx = My = Mu
        else:
            Ei, Mui = np.linalg.inv(Eps), np.linalg.inv(Mu)
        W, V, q = mods[j]
        amps.append(dict(a_plus=downs[2 * j + 1],
                         a_minus=_applyR(rud[2 * j + 2], downs[2 * j + 2]),
                         W=W, V=V, q=q, d=layer.d, z_top=ztop, layer=layer,
                         homogeneous=homo[j], Eps=Eps, Mu=Mu, Ei=Ei, Mui=Mui,
                         Cx=Cx, Cy=Cy, Mx=Mx, My=My))
        ztop += layer.d
    out["amps"], out["z_bot"] = amps, ztop
    if need_modes:
        out["mods"], out["blocks"], out["rud"], out["downs"] = mods, blocks, rud, downs
    return out


# =============================================================================
# 场与材料图
# =============================================================================
def _material_ops_at(res, z):
    N = res["N"]
    I = np.eye(N, dtype=complex)
    if z < 0:
        Eps, Mu = res["epsI"] * I, res["muI"] * I
        return Eps, Mu, I / res["epsI"], I / res["muI"], Eps, Eps, Mu, Mu
    if z > res["z_bot"]:
        Eps, Mu = res["epsII"] * I, res["muII"] * I
        return Eps, Mu, I / res["epsII"], I / res["muII"], Eps, Eps, Mu, Mu
    for amp in res["amps"]:
        if z <= amp["z_top"] + amp["d"] + 1e-12:
            return (amp["Eps"], amp["Mu"], amp["Ei"], amp["Mui"],
                    amp["Cx"], amp["Cy"], amp["Mx"], amp["My"])
    Eps, Mu = res["epsII"] * I, res["muII"] * I
    return Eps, Mu, I / res["epsII"], I / res["muII"], Eps, Eps, Mu, Mu


def field_at(res, z):
    """返回 z 处谐波场与因式分解后的 ``D/B`` 分量。"""
    k0, N = res["k0"], res["N"]
    if z < 0:
        e_in = np.exp(-k0 * res["qI"] * z)
        e_re = np.exp(k0 * res["qI"] * z)
        F = res["inc"] * e_in + res["R"] * e_re
        G = bapply(res["VI"], -res["inc"] * e_in + res["R"] * e_re)
    elif z > res["z_bot"]:
        e = np.exp(-k0 * res["qII"] * (z - res["z_bot"]))
        F = res["T"] * e
        G = bapply(res["VII"], -res["T"] * e)
    else:
        if not res["amps"]:
            F, G = res["inc"] + res["R"], bapply(res["VI"], -res["inc"] + res["R"])
        else:
            amp = next(a for a in res["amps"] if z <= a["z_top"] + a["d"] + 1e-12)
            zl = z - amp["z_top"]
            e1 = np.exp(-k0 * amp["q"] * zl)
            e2 = np.exp(-k0 * amp["q"] * (amp["d"] - zl))
            cp, cm = e1 * amp["a_plus"], e2 * amp["a_minus"]
            W, V = amp["W"], amp["V"]
            F = bapply(W, cp + cm) if _isbd(W) else W @ (cp + cm)
            G = bapply(V, -cp + cm) if _isbd(V) else V @ (-cp + cm)

    Eps, Mu, Ei, Mui, Cx, Cy, Mx, My = _material_ops_at(res, z)
    Ez, hz = _vertical_components(F, G, Ei, Mui, res["Kx"], res["Ky"])
    Ex, Ey, hx, hy = F[:N], F[N:], G[:N], G[N:]
    Hx, Hy, Hz = -1j * hx, -1j * hy, -1j * hz
    return dict(Ex=Ex, Ey=Ey, Ez=Ez, Hx=Hx, Hy=Hy, Hz=Hz,
                Dx=Cx @ Ex, Dy=Cy @ Ey, Dz=Eps @ Ez,
                Bx=Mx @ Hx, By=My @ Hy, Bz=Mu @ Hz)


def _to_xy(harm, res, nx, ny):
    mn, M1, M2 = res["mn"], res["M1"], res["M2"]
    if nx < 2 * M1 + 1 or ny < 2 * M2 + 1:
        raise ValueError("nx/ny 必须不小于相应的 Fourier 阶数")
    grid = np.zeros((nx, ny), dtype=complex)
    grid[mn[:, 0] % nx, mn[:, 1] % ny] = harm
    field = np.fft.ifft2(grid) * nx * ny
    lat, k0 = res["lat"], res["k0"]
    x = np.arange(nx) * lat.Lx / nx
    y = np.arange(ny) * lat.Ly / ny
    kx0 = res["nI"] * np.sin(res["theta"]) * np.cos(res["phi"])
    ky0 = res["nI"] * np.sin(res["theta"]) * np.sin(res["phi"])
    field *= np.exp(-1j * k0 * kx0 * x)[:, None] * np.exp(-1j * k0 * ky0 * y)[None, :]
    return x, y, field


def field_xy(res, z, nx=64, ny=64):
    """一个 z 平面上的实空间场及 ``|E|²``。"""
    harm = field_at(res, z)
    out = {}
    for key, values in harm.items():
        x, y, out[key] = _to_xy(values, res, nx, ny)
    out["E2"] = np.abs(out["Ex"]) ** 2 + np.abs(out["Ey"]) ** 2 + np.abs(out["Ez"]) ** 2
    out["x"], out["y"], out["z"] = x, y, float(z)
    return out


def eps_mu_map(layer, lat, nx=200, ny=200):
    """一个元胞上的实空间 ``eps``、``mu`` 图。"""
    x = (np.arange(nx) + .5) / nx * lat.Lx
    y = (np.arange(ny) + .5) / ny * lat.Ly
    eps = np.empty((ny, nx), dtype=complex)
    mu = np.empty_like(eps)
    for j, yy in enumerate(y):
        eps[j], mu[j] = layer.profile(x, yy, lat)
    return x, y, eps, mu


def eps_map(layer, lat, nx=200, ny=200):
    """兼容旧图形接口：返回局部等效折射率 ``sqrt(eps*mu)``。"""
    x, y, eps, mu = eps_mu_map(layer, lat, nx, ny)
    return x, y, np.sqrt(eps * mu + 0j)


def index_of(res, m, n):
    return int(np.where((res["mn"][:, 0] == m) & (res["mn"][:, 1] == n))[0][0])
