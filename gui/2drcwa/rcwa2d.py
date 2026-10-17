# -*- coding: utf-8 -*-
"""
二维（交叉光栅）RCWA 求解器
============================
矩形格子、任意"y 分段 × 每段内 x 分段"的分段常数剖面；S 矩阵级联；
输出各级 s/p 复振幅与效率。二维天然是矢量问题，不存在标量 TE/TM 分裂。

约定（与 ../rcwa1d.py 完全一致，可逐位对拍）
--------------------------------------------
* 时间因子 exp(-i w t)；z 自入射侧指向出射侧，结构上表面 z = 0。
* 归一化横向波矢（以 k0 计）
      Kx_mn = nI sin(th) cos(ph) - m*lam0/Lx
      Ky_mn = nI sin(th) sin(ph) - n*lam0/Ly
  场按  F(x,y,z) = sum_mn S_mn(z) exp(-i(kx_mn x + ky_mn y))
* Kz：传播模取实正，倏逝模取 +i|kz|；层内模指数 q = -i kz/k0，
  故 Re q >= 0，exp(-k0 q z) 模长恒 <= 1，S 矩阵级联天然稳定。

场与算符
--------
以 z' = k0 z、E 为电场、h = i*Z0*H，Maxwell 化为
      d/dz' [Ex;Ey] = P [hx;hy],      d/dz' [hx;hy] = Q [Ex;Ey]
      P = [[Kx Ei Ky,      I - Kx Ei Kx],
           [Ky Ei Ky - I, -Ky Ei Kx   ]]
      Q = [[Kx Ky,        Cy - Kx^2   ],
           [Ky^2 - Cx,   -Ky Kx       ]]
其中 Ei = inv(<<eps>>)（Ez 走 Laurent 阵的逆），而

      Cy 作用在 Ey 上 -> y 方向法向用逆规则、x 方向法向用 Laurent
      Cx 作用在 Ex 上 -> x 方向法向用逆规则、y 方向法向用 Laurent

即 Li 1997 的嵌套因式分解。取 Cx = Cy = <<eps>> 则退化为纯 Laurent 规则
（收敛差，仅作对照）。一维极限下本式严格复现 ../rcwa1d.py 的 TE 与 TM。
"""
import numpy as np

__all__ = ["Lattice", "Layer2D", "Stack2D", "solve2d", "eps_map", "RayleighAnomalyError"]


# =============================================================================
#  1. 一维 Fourier 工具（沿用 rcwa1d.fourier_piecewise 的解析积分，无 FFT 混叠）
# =============================================================================
def fourier_1d(edges, values, period, hmax):
    """分段常数剖面的 Fourier 系数 c[h+hmax]，f(x) = sum_h c_h exp(i 2pi h x/P)。"""
    edges = np.asarray(edges, float)
    values = np.asarray(values, complex)
    h = np.arange(-hmax, hmax + 1)
    c = np.zeros(len(h), complex)
    c[hmax] = np.sum(values * np.diff(edges)) / period
    hn = h[h != 0]
    ph = np.exp(-2j * np.pi * np.outer(hn, edges) / period)
    c[h != 0] = (values * (ph[:, 1:] - ph[:, :-1])).sum(axis=1) / (-2j * np.pi * hn)
    return c


def toeplitz_1d(c, M):
    """由 c[h+2M]（h = -2M..2M）生成 (2M+1)^2 的 Toeplitz 阵，(i,j) 元 = c[i-j]。"""
    idx = np.arange(-M, M + 1)
    return c[np.subtract.outer(idx, idx) + 2 * M]


def _seg_fourier(y_edges, hmax, period):
    """每个 y 分段对第 n 个 Fourier 分量的贡献系数，形状 (n_seg, 2*hmax+1)。"""
    h = np.arange(-hmax, hmax + 1)
    out = np.zeros((len(y_edges) - 1, len(h)), complex)
    out[:, hmax] = np.diff(y_edges) / period
    hn = h[h != 0]
    ph = np.exp(-2j * np.pi * np.outer(hn, y_edges) / period)
    out[:, h != 0] = ((ph[:, 1:] - ph[:, :-1]) / (-2j * np.pi * hn)[:, None]).T
    return out


# =============================================================================
#  2. 格子与谐波集
# =============================================================================
class Lattice:
    """矩形格子：x 周期 Lx、y 周期 Ly。"""

    def __init__(self, Lx, Ly):
        self.Lx, self.Ly = float(Lx), float(Ly)

    def harmonics(self, M1, M2):
        m, n = np.meshgrid(np.arange(-M1, M1 + 1), np.arange(-M2, M2 + 1), indexing="ij")
        return np.column_stack([m.ravel(), n.ravel()]), M1, M2


# =============================================================================
#  3. 层
# =============================================================================
class Layer2D:
    """一层。

    均匀层：Layer2D(d, n=1.46)
    结构层：Layer2D(d, y_edges=[...], x_profiles=[(x_edges, n_values), ...])
        y_edges 长度 K+1，覆盖一个 y 周期；x_profiles 长度 K，
        每个是该 y 分段内沿 x 的分段常数剖面（覆盖一个 x 周期）。
    """

    def __init__(self, thickness, n=None, y_edges=None, x_profiles=None, name=""):
        self.d = float(thickness)
        self.n = n
        self.y_edges = y_edges
        self.x_profiles = x_profiles
        self.name = name
        self.homogeneous = n is not None
        if not self.homogeneous and (y_edges is None or x_profiles is None):
            raise ValueError("结构层必须给出 y_edges 与 x_profiles")

    # ---------- Fourier 算符 ----------
    def operators(self, lat, mn, M1, M2, factorization="li"):
        """返回 (Eps, Cx, Cy)：Laurent 阵、作用于 Ex 的阵、作用于 Ey 的阵。"""
        if self.homogeneous:
            N = len(mn)
            I = np.eye(N, dtype=complex)
            return self.n ** 2 * I, self.n ** 2 * I, self.n ** 2 * I

        Lx, Ly = lat.Lx, lat.Ly
        ys, xps = np.asarray(self.y_edges, float), self.x_profiles
        wseg = _seg_fourier(ys, 2 * M2, Ly)              # (K, 4M2+1)

        # --- 每个 y 分段内的三个一维 x 算符 ---
        eps_x, invx_x, lau_x = [], [], []
        for (xe, nv) in xps:
            e = np.asarray(nv, complex) ** 2
            ce = fourier_1d(xe, e, Lx, 2 * M1)
            ci = fourier_1d(xe, 1.0 / e, Lx, 2 * M1)
            eps_x.append(ce)                                    # Laurent 系数
            invx_x.append(np.linalg.inv(toeplitz_1d(ci, M1)))    # <<1/eps>>^-1
            lau_x.append(toeplitz_1d(ce, M1))                    # <<eps>>

        # --- 沿 y 做 Fourier，得到以 (n-n') 为指标的一维 x 算符族 ---
        n1 = 2 * M1 + 1
        Cx_h = np.zeros((4 * M2 + 1, n1, n1), complex)   # Ex：x 逆规则
        Eps_h = np.zeros((4 * M2 + 1, n1, n1), complex)  # 纯 Laurent
        for k in range(len(xps)):
            Cx_h += wseg[k][:, None, None] * invx_x[k]
            Eps_h += wseg[k][:, None, None] * lau_x[k]

        if factorization == "laurent":
            Cx_h = Eps_h.copy()
            Cy_h = Eps_h.copy()
        else:
            # Ey：y 方向法向用逆规则 -> 先沿 y 做一维逆规则，再沿 x 做 Laurent。
            # 对每个 x 分段（由所有 y 段的 x 边界合并而成）建立一维 y 剖面。
            xcuts = np.unique(np.concatenate([np.asarray(xe, float) % Lx for xe, _ in xps]
                                             + [np.array([0.0, Lx])]))
            xcuts = np.unique(np.round(xcuts, 12))
            xcuts = xcuts[(xcuts >= -1e-12) & (xcuts <= Lx + 1e-12)]
            if xcuts[0] > 1e-12:
                xcuts = np.concatenate([[0.0], xcuts])
            if xcuts[-1] < Lx - 1e-12:
                xcuts = np.concatenate([xcuts, [Lx]])
            xmid = 0.5 * (xcuts[:-1] + xcuts[1:])
            wsegx = _seg_fourier(xcuts, 2 * M1, Lx)      # (Kx, 4M1+1)
            n2 = 2 * M2 + 1
            Cy_yh = np.zeros((4 * M1 + 1, n2, n2), complex)
            for j, xm in enumerate(xmid):
                ecol = np.array([_eps_at(xps[k], xm, Lx) for k in range(len(xps))], complex)
                cy = fourier_1d(ys, 1.0 / ecol, Ly, 2 * M2)
                Cy_yh += wsegx[j][:, None, None] * np.linalg.inv(toeplitz_1d(cy, M2))
            Cy_h = None
        # --- 组装成 (m,n) 全阵 ---
        Eps = _assemble_xfirst(Eps_h, mn, M1, M2)
        Cx = _assemble_xfirst(Cx_h, mn, M1, M2)
        Cy = (Eps.copy() if factorization == "laurent"
              else _assemble_yfirst(Cy_yh, mn, M1, M2))
        return Eps, Cx, Cy


def _eps_at(profile, x, Lx):
    xe, nv = profile
    xe = np.asarray(xe, float)
    xx = xe[0] + (x - xe[0]) % Lx
    k = np.searchsorted(xe, xx, side="right") - 1
    return np.asarray(nv, complex)[np.clip(k, 0, len(nv) - 1)] ** 2


def _assemble_xfirst(blocks, mn, M1, M2):
    """blocks[n-n'] 是 (2M1+1)^2 的 x 算符 -> 全阵。"""
    dn = mn[:, 1][:, None] - mn[:, 1][None, :] + 2 * M2
    im = mn[:, 0] + M1
    return blocks[dn, im[:, None], im[None, :]]


def _assemble_yfirst(blocks, mn, M1, M2):
    """blocks[m-m'] 是 (2M2+1)^2 的 y 算符 -> 全阵。"""
    dm = mn[:, 0][:, None] - mn[:, 0][None, :] + 2 * M1
    inn = mn[:, 1] + M2
    return blocks[dm, inn[:, None], inn[None, :]]


class Stack2D:
    def __init__(self, nI, layers, nII, lattice):
        self.nI, self.nII, self.layers, self.lat = nI, nII, list(layers), lattice


# =============================================================================
#  4. 模式求解
# =============================================================================
def _branch(q):
    """Re q >= 0（倏逝衰减）；传播模统一到 Im q <= 0，与 rcwa1d._branch 一致。"""
    q = np.where(q.real < 0, -q, q)
    return np.where((q.real < 1e-10) & (q.imag > 0), -q, q)


def _kz_over_k0(n, Kx, Ky):
    s = n ** 2 - Kx ** 2 - Ky ** 2
    # Preserve both real and imaginary parts for passive absorbing layers.
    # exp(+i*kz*z) must decay along +z; the real-index limit is unchanged.
    kz = np.sqrt(np.asarray(s, dtype=complex))
    return np.where(kz.imag < 0, -kz, kz)


def _blk(a11, a12, a21, a22):
    return np.block([[a11, a12], [a21, a22]])


# --- 2x2 分块对角矩阵（每块都是 N 维对角阵），存成 (2,2,N) ---------------------
#  均匀层的 W、V、传播阵、以及均匀-均匀界面块都属于这一类，乘法与求逆退化成
#  N 维逐元素运算。MDG 的 20 层膜堆整段因此几乎免费，真正 O((2N)^3) 的稠密运算
#  只剩「入射区↔光栅层」「光栅层↔膜堆」两个界面 + 每个光栅层一次本征分解。
def bmul(A, B):
    return np.einsum("ikn,kjn->ijn", A, B)


def binv(A):
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    return np.array([[A[1, 1], -A[0, 1]], [-A[1, 0], A[0, 0]]]) / det


def beye(N):
    z, o = np.zeros(N, complex), np.ones(N, complex)
    return np.array([[o, z], [z, o]])


def bzero(N):
    z = np.zeros(N, complex)
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
    return np.concatenate([A[0, 0] * vx + A[0, 1] * vy, A[1, 0] * vx + A[1, 1] * vy])


def _isbd(x):
    return isinstance(x, np.ndarray) and x.ndim == 3


class S2:
    """[out_down; out_up] = S [in_down; in_up]；bd=True 时四块存 (2,2,N)。"""
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
        Tp, Tm = 0.5 * (Mm + Nn), 0.5 * (Mm - Nn)
        Tpi = binv(Tp)
        return S2(Tp - bmul(bmul(Tm, Tpi), Tm), bmul(Tm, Tpi),
                  -bmul(Tpi, Tm), Tpi, bd=True)
    return _s_dense(np.linalg.solve(W2, W1), np.linalg.solve(V2, V1))


def s_interface2_to_homog(W1, V1, V2):
    """稠密层 → 均匀层（W2 = I，V2 为 (2,2,N) 分块对角）。与 s_interface2 相同，
    但两次 2N 阶 solve 退化为 O(N^2) 的分块对角逆作用。"""
    return _s_dense(W1, bapply(binv(V2), V1))


def _s_dense(Mm, Nn):
    Tp, Tm = 0.5 * (Mm + Nn), 0.5 * (Mm - Nn)
    Tpi = np.linalg.inv(Tp)
    return S2(Tp - Tm @ Tpi @ Tm, Tm @ Tpi, -Tpi @ Tm, Tpi)


def s_propagate2(q, d, k0):
    """任何对角阵都是 2x2 分块对角的特例。|X| <= 1 由 Re q >= 0 保证。"""
    N = len(q) // 2
    x = np.exp(-k0 * q * d)
    z = np.zeros(N, complex)
    X = np.array([[x[:N], z], [z, x[N:]]])
    return S2(X, bzero(N), bzero(N), X, bd=True)


def star2(A, B):
    if A.bd and B.bd:
        N = A.Tdd.shape[2]
        G = binv(beye(N) - bmul(A.Rdu, B.Rud))
        H = binv(beye(N) - bmul(B.Rud, A.Rdu))
        return S2(bmul(bmul(B.Tdd, G), A.Tdd),
                  B.Rdu + bmul(bmul(bmul(B.Tdd, G), A.Rdu), B.Tuu),
                  A.Rud + bmul(bmul(bmul(A.Tuu, B.Rud), G), A.Tdd),
                  bmul(bmul(A.Tuu, H), B.Tuu), bd=True)
    A, B = A.dense(), B.dense()
    n = A.Tdd.shape[0]
    I = np.eye(n, dtype=complex)
    G = np.linalg.inv(I - A.Rdu @ B.Rud)
    H = np.linalg.inv(I - B.Rud @ A.Rdu)
    return S2(B.Tdd @ G @ A.Tdd, B.Rdu + B.Tdd @ G @ A.Rdu @ B.Tuu,
              A.Rud + A.Tuu @ B.Rud @ G @ A.Tdd, A.Tuu @ H @ B.Tuu)


def _bd_rmul(M, B):
    """M @ bdense(B)，O(n^2)：右乘分块对角 = 对转置做分块对角左乘。"""
    return bapply(B.transpose(1, 0, 2), M.T).T


def _rud2(blk, rud):
    """自下而上只递推 Rud 一条链，不存整条前后缀 S 链。"""
    if blk.bd and (rud is None or _isbd(rud)):
        N = blk.Tdd.shape[2]
        r = bzero(N) if rud is None else rud
        G = binv(beye(N) - bmul(blk.Rdu, r))
        return blk.Rud + bmul(bmul(bmul(blk.Tuu, r), G), blk.Tdd)
    if blk.bd:
        # 分块对角块（传播块、均匀层间界面）遇到下方稠密 Rud：左右乘只是 O(n^2) 的
        # 分块缩放；传播块 Rdu = 0 时 G = I，连 2N 阶求逆也省掉
        rG = rud
        if np.any(blk.Rdu):
            n = rud.shape[0]
            rG = rud @ np.linalg.inv(np.eye(n, dtype=complex) - bapply(blk.Rdu, rud))
        return bdense(blk.Rud) + _bd_rmul(bapply(blk.Tuu, rG), blk.Tdd)
    A = blk.dense()
    n = A.Tdd.shape[0]
    r = (np.zeros((n, n), complex) if rud is None
         else (bdense(rud) if _isbd(rud) else rud))
    G = np.linalg.inv(np.eye(n, dtype=complex) - A.Rdu @ r)
    return A.Rud + A.Tuu @ r @ G @ A.Tdd


def _step2(blk, rud_next, down):
    if blk.bd and (rud_next is None or _isbd(rud_next)):
        N = blk.Tdd.shape[2]
        r = bzero(N) if rud_next is None else rud_next
        G = binv(beye(N) - bmul(blk.Rdu, r))
        return bapply(bmul(G, blk.Tdd), down)
    if blk.bd:
        rhs = bapply(blk.Tdd, down)                      # 分块对角块 + 稠密 Rud（见 _rud2）
        if not np.any(blk.Rdu):
            return rhs
        n = rud_next.shape[0]
        return np.linalg.solve(np.eye(n, dtype=complex) - bapply(blk.Rdu, rud_next), rhs)
    A = blk.dense()
    n = A.Tdd.shape[0]
    r = (np.zeros((n, n), complex) if rud_next is None
         else (bdense(rud_next) if _isbd(rud_next) else rud_next))
    return np.linalg.solve(np.eye(n, dtype=complex) - A.Rdu @ r, A.Tdd @ down)


def _applyR(rud, v):
    return bapply(rud, v) if _isbd(rud) else rud @ v


class RayleighAnomalyError(np.linalg.LinAlgError, ValueError):
    """某均匀介质中有衍射级恰好掠射（kz = 0），V = Q/q 无定义（与 rcwa1d 同名同义）。"""


def _homog_modes2d(n, Kx, Ky, where="均匀层", mn=None):
    """均匀层：W = I（bd），q 二重简并，V = Q/q（bd）。kz 恰为 0 时报错而非输出 NaN。"""
    N = len(Kx)
    kz = _kz_over_k0(n, Kx, Ky)
    grazing = np.flatnonzero(kz == 0)
    if grazing.size:
        labels = ", ".join(f"({m}, {k})" for m, k in (mn[grazing] if mn is not None else
                                                      [(i, "?") for i in grazing]))
        raise RayleighAnomalyError(
            f"{where}（n={n}）的 (m, n) = {labels} 级恰好沿界面掠射（kz=0，Rayleigh 异常点），"
            f"无法求解。请把入射角、方位角或周期略作改动（例如入射角改动 0.01°）后重算。")
    q1 = _branch(-1j * kz)
    q = np.concatenate([q1, q1])
    Qb = np.array([[Kx * Ky, n ** 2 - Kx ** 2],
                   [Ky ** 2 - n ** 2, -Ky * Kx]]).astype(complex)
    Vb = Qb / q1[None, None, :]
    return beye(N), Vb, q, kz


def _grating_modes2d(Eps, Cx, Cy, Kx, Ky, Ei=None):
    """Ei = inv(Eps) 可由调用方传入，避免同一 N 阶逆算两次。"""
    N = len(Kx)
    D = np.diag
    if Ei is None:
        Ei = np.linalg.inv(Eps)
    I = np.eye(N)
    # Kx、Ky 是对角阵：左右乘改为行列缩放，与原矩阵乘逐元素相同
    kx, ky = Kx[:, None], Ky[:, None]
    P = _blk(kx * Ei * Ky, I - kx * Ei * Kx, ky * Ei * Ky - I, -(ky * Ei * Kx))
    Q = _blk(D(Kx * Ky), Cy - D(Kx * Kx), D(Ky * Ky) - Cx, -D(Ky * Kx))
    w2, W = np.linalg.eig(P @ Q)
    q = _branch(np.sqrt(w2.astype(complex)))
    V = (Q @ W) / q[None, :]
    return W, V, q


# =============================================================================
#  5. 求解
# =============================================================================
class Result2D(dict):
    __getattr__ = dict.get


def solve2d(stack, lam0, theta_deg, phi_deg, M1, M2,
            psi_deg=90.0, delta_deg=0.0, jones_in=None,
            factorization="li", need_modes=False):
    """psi=90 为 s、psi=0 为 p；delta 为 s 相对 p 的相位（可给椭圆偏振）。"""
    k0 = 2 * np.pi / lam0
    th, ph = np.radians(theta_deg), np.radians(phi_deg)
    lat, nI, nII = stack.lat, stack.nI, stack.nII
    mn, M1, M2 = lat.harmonics(M1, M2)
    N = len(mn)

    Kx = nI * np.sin(th) * np.cos(ph) - mn[:, 0] * lam0 / lat.Lx
    Ky = nI * np.sin(th) * np.sin(ph) - mn[:, 1] * lam0 / lat.Ly

    WI, VI, qI, kzI = _homog_modes2d(nI, Kx, Ky, "入射区", mn)
    WII, VII, qII, kzII = _homog_modes2d(nII, Kx, Ky, "出射区", mn)

    mods, homo, Eis = [], [], []
    for L in stack.layers:
        if L.homogeneous:
            where = f"第{len(mods) + 1}层" + (f"「{L.name}」" if L.name else "")
            mods.append(_homog_modes2d(L.n, Kx, Ky, where, mn)[:3]); homo.append(True)
            Eis.append(None)
        else:
            Eps, Cx, Cy = L.operators(lat, mn, M1, M2, factorization)
            Ei = np.linalg.inv(Eps)
            mods.append(_grating_modes2d(Eps, Cx, Cy, Kx, Ky, Ei)); homo.append(False)
            Eis.append(Ei)

    def _mk(W1, V1, h1, W2, V2, h2):
        if h1 and h2:
            return s_interface2(W1, V1, W2, V2, bd=True)
        if h2:                                   # 均匀层的 W 恒为 beye(N)
            return s_interface2_to_homog(W1, V1, V2)
        return s_interface2(bdense(W1) if h1 else W1, bdense(V1) if h1 else V1, W2, V2)

    blocks = []
    Wp, Vp, hp = WI, VI, True
    for (W, V, q), L, hg in zip(mods, stack.layers, homo):
        blocks.append(_mk(Wp, Vp, hp, W, V, hg))
        blocks.append(s_propagate2(q, L.d, k0))
        Wp, Vp, hp = W, V, hg
    blocks.append(_mk(Wp, Vp, hp, WII, VII, True))

    nb = len(blocks)
    rud = [None] * (nb + 1)
    for k in range(nb - 1, -1, -1):
        rud[k] = _rud2(blocks[k], rud[k + 1])

    if jones_in is not None:
        Es, Ep = complex(jones_in[0]), complex(jones_in[1])
    else:
        psi, dl = np.radians(psi_deg), np.radians(delta_deg)
        Es, Ep = np.sin(psi) * np.exp(1j * dl), np.cos(psi)
    i0 = int(np.where((mn[:, 0] == 0) & (mn[:, 1] == 0))[0][0])
    inc = np.zeros(2 * N, complex)
    inc[i0] = -Es * np.sin(ph) + Ep * np.cos(th) * np.cos(ph)        # Ex
    inc[N + i0] = Es * np.cos(ph) + Ep * np.cos(th) * np.sin(ph)     # Ey

    downs = [inc]
    for k in range(nb):
        downs.append(_step2(blocks[k], rud[k + 1], downs[-1]))
    Rf, Tf = _applyR(rud[0], inc), downs[nb]

    def efficiency(F, kz, sign):
        Ex, Ey = F[:N], F[N:]
        prop = np.real(kz) > 1e-12
        Ez = np.zeros(N, complex)
        Ez[prop] = -sign * (Kx[prop] * Ex[prop] + Ky[prop] * Ey[prop]) / kz[prop]
        E2 = np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2
        de = np.zeros(N)
        de[prop] = np.real(kz[prop]) * E2[prop] / (nI * np.cos(th))
        return de, prop, Ez

    DEr, propR, EzR = efficiency(Rf, kzI, -1.0)
    DEt, propT, EzT = efficiency(Tf, kzII, +1.0)

    def sp(F, kz, n_med, sign):
        Ex, Ey = F[:N], F[N:]
        kt = np.hypot(Kx, Ky)
        big = kt > 1e-12
        ktz = np.where(big, kt, 1.0)
        s_ = np.where(big, (-Ky * Ex + Kx * Ey) / ktz, -np.sin(ph) * Ex + np.cos(ph) * Ey)
        cth = np.where(np.abs(kz) > 1e-12, kz / n_med, 1.0)
        p_ = np.where(big, sign * (Kx * Ex + Ky * Ey) / (ktz * cth),
                      sign * (np.cos(ph) * Ex + np.sin(ph) * Ey) / cth)
        return s_, p_

    Rs, Rp = sp(Rf, kzI, nI, -1.0)
    Ts, Tp = sp(Tf, kzII, nII, +1.0)

    out = Result2D(mn=mn, M1=M1, M2=M2, Kx=Kx, Ky=Ky, k0=k0, lat=lat, stack=stack,
                   kzI=kzI, kzII=kzII, Ex_r=Rf[:N], Ey_r=Rf[N:], Ez_r=EzR,
                   Ex_t=Tf[:N], Ey_t=Tf[N:], Ez_t=EzT,
                   Rs=Rs, Rp=Rp, Ts=Ts, Tp=Tp, DEr=DEr, DEt=DEt,
                   prop_r=propR, prop_t=propT, i0=i0, theta=th, phi=ph)
    # --- 逐层模式系数：a+ 取层顶下行、a- 取层底上行，两者的指数因子模长均 <= 1 ---
    amps, ztop = [], 0.0
    for j, L in enumerate(stack.layers):
        W, V, q = mods[j]
        amps.append(dict(a_plus=downs[2 * j + 1],
                         a_minus=_applyR(rud[2 * j + 2], downs[2 * j + 2]),
                         W=W, V=V, q=q, d=L.d, z_top=ztop, layer=L,
                         homog=homo[j], Ei=Eis[j]))
        ztop += L.d
    out["amps"], out["z_bot"] = amps, ztop
    out["inc"], out["R"], out["T"] = inc, Rf, Tf
    out["VI"], out["qI"], out["VII"], out["qII"] = VI, qI, VII, qII
    out["nI"], out["nII"], out["N"] = nI, nII, N
    if need_modes:
        out["mods"], out["rud"], out["downs"] = mods, rud, downs
    return out


def index_of(res, m, n):
    return int(np.where((res["mn"][:, 0] == m) & (res["mn"][:, 1] == n))[0][0])


def eps_map(layer, lat, nx=200, ny=200):
    """实空间折射率图，用于把结构与级次谱套准。"""
    x = (np.arange(nx) + 0.5) / nx * lat.Lx
    y = (np.arange(ny) + 0.5) / ny * lat.Ly
    out = np.ones((ny, nx), complex)
    if layer.homogeneous:
        return x, y, out * layer.n
    ye = np.asarray(layer.y_edges, float)
    for j, yy in enumerate(y):
        k = np.clip(np.searchsorted(ye, ye[0] + (yy - ye[0]) % lat.Ly, side="right") - 1,
                    0, len(layer.x_profiles) - 1)
        xe, nv = layer.x_profiles[k]
        xe = np.asarray(xe, float)
        idx = np.clip(np.searchsorted(xe, xe[0] + (x - xe[0]) % lat.Lx, side="right") - 1,
                      0, len(nv) - 1)
        out[j] = np.asarray(nv, complex)[idx]
    return x, y, out


# =============================================================================
#  6. 层内场重建
# =============================================================================
#  层内   [Ex;Ey] = W [a+ e^{-k0 q (z-z_top)} + a- e^{-k0 q (z_bot-z)}]
#         [hx;hy] = V [-a+ e^{...}            + a- e^{...}          ]
#  区域 I 以 z=0 为参考：入射 e^{-k0 q z}（即 e^{+i kz z}）、反射 e^{+k0 q z}
#  区域 II 以 z=z_bot 为参考：透射 e^{-k0 q (z-z_bot)}
#
#  纵向分量（归一化后）：
#         hz = i (Ky Ex - Kx Ey)
#         Ez = i inv(<<eps>>) (Ky hx - Kx hy)
#  Ez 走 Laurent 阵的逆——它沿竖直侧壁是切向、连续。用错算符时衍射效率照样
#  收敛、场却是错的，这一条只能靠界面连续性检验（selftest2d T5）把关。
# =============================================================================
def _dense(A):
    return bdense(A) if _isbd(A) else A


def field_at(res, z):
    """返回 z 处的谐波列 dict(Ex,Ey,Ez,Hx,Hy,Hz)。

    E 对入射 |E|=1 归一；H 以 Z0 归一（即 Z0*H），二者数值可直接比较。
    """
    k0, N = res["k0"], res["N"]
    Kx, Ky = res["Kx"], res["Ky"]
    I = np.eye(N, dtype=complex)

    if z < 0.0:                                   # 区域 I
        q = res["qI"]
        e_in, e_re = np.exp(-k0 * q * z), np.exp(k0 * q * z)
        F = res["inc"] * e_in + res["R"] * e_re
        G = bapply(res["VI"], -res["inc"] * e_in + res["R"] * e_re)
        Ei = I / res["nI"] ** 2
    elif z > res["z_bot"]:                        # 区域 II
        q = res["qII"]
        e = np.exp(-k0 * q * (z - res["z_bot"]))
        F = res["T"] * e
        G = bapply(res["VII"], -res["T"] * e)
        Ei = I / res["nII"] ** 2
    else:                                         # 层内
        a = next(a for a in res["amps"] if z <= a["z_top"] + a["d"] + 1e-12)
        zl = z - a["z_top"]
        e1 = np.exp(-k0 * a["q"] * zl)
        e2 = np.exp(-k0 * a["q"] * (a["d"] - zl))
        cp, cm = e1 * a["a_plus"], e2 * a["a_minus"]
        W, V = a["W"], a["V"]
        F = bapply(W, cp + cm) if _isbd(W) else W @ (cp + cm)
        G = bapply(V, -cp + cm) if _isbd(V) else V @ (-cp + cm)
        Ei = (I / a["layer"].n ** 2) if a["homog"] else a["Ei"]

    Ex, Ey, hx, hy = F[:N], F[N:], G[:N], G[N:]
    hz = 1j * (Ky * Ex - Kx * Ey)
    Ez = 1j * (Ei @ (Ky * hx - Kx * hy))
    return dict(Ex=Ex, Ey=Ey, Ez=Ez, Hx=-1j * hx, Hy=-1j * hy, Hz=-1j * hz)


def _to_xy(harm, res, nx, ny):
    """谐波列 -> 一个元胞上的实空间场（逆 FFT，O(nx ny log)）。"""
    mn, M1, M2 = res["mn"], res["M1"], res["M2"]
    k0, lat = res["k0"], res["lat"]
    if nx < 2 * M1 + 1 or ny < 2 * M2 + 1:
        raise ValueError("nx/ny 必须大于 2M+1，否则混叠")
    G = np.zeros((nx, ny), complex)
    G[mn[:, 0] % nx, mn[:, 1] % ny] = harm
    f = np.fft.ifft2(G) * nx * ny
    x = np.arange(nx) * lat.Lx / nx
    y = np.arange(ny) * lat.Ly / ny
    kx0 = res["nI"] * np.sin(res["theta"]) * np.cos(res["phi"])
    ky0 = res["nI"] * np.sin(res["theta"]) * np.sin(res["phi"])
    f *= np.exp(-1j * k0 * kx0 * x)[:, None] * np.exp(-1j * k0 * ky0 * y)[None, :]
    return x, y, f


def field_xy(res, z, nx=64, ny=256):
    """z 处一个元胞上的六个场分量与 |E|^2（对入射 |E|^2 归一）。"""
    h = field_at(res, z)
    out = {}
    for k, v in h.items():
        x, y, out[k] = _to_xy(v, res, nx, ny)
    out["E2"] = sum(np.abs(out[k]) ** 2 for k in ("Ex", "Ey", "Ez"))
    out["x"], out["y"], out["z"] = x, y, float(z)
    return out


def _to_line(harm, res, x, y):
    """任意 (x,y) 点列上的直接求和（用于剖面，点数不多时最省事）。"""
    k0 = res["k0"]
    ph = np.exp(-1j * k0 * (np.outer(x, res["Kx"]) + np.outer(y, res["Ky"])))
    return ph @ harm


def field_slice(res, zs, axis="x", at=0.0, npts=240):
    """沿 x（固定 y=at）或沿 y（固定 x=at）的纵向剖面，返回 |E|^2 与各分量。"""
    zs = np.atleast_1d(zs)
    lat = res["lat"]
    if axis == "x":
        t = np.arange(npts) * lat.Lx / npts
        xs, ys = t, np.full(npts, at)
    else:
        t = np.arange(npts) * lat.Ly / npts
        xs, ys = np.full(npts, at), t
    comp = {k: np.zeros((len(zs), npts), complex) for k in
            ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
    for i, z in enumerate(zs):
        h = field_at(res, z)
        for k in comp:
            comp[k][i] = _to_line(h[k], res, xs, ys)
    comp["E2"] = sum(np.abs(comp[k]) ** 2 for k in ("Ex", "Ey", "Ez"))
    comp["t"], comp["z"] = t, zs
    return comp
