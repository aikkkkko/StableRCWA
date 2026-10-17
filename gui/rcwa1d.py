# -*- coding: utf-8 -*-
"""
一维周期结构的多层 RCWA 求解器（S 矩阵级联 + 层内场重建）
==========================================================
面向多层介质膜光栅（MDG）等一维周期结构：任意层数、任意分段常数
折射率剖面，输出各级衍射效率与结构内部任意点的场。

约定
----
* 时间因子 exp(-i w t)；z 轴自入射侧指向出射侧，结构上表面为 z = 0。
* 横向波矢    kx_m = k0 * (nI * sin(theta) - m * lam0 / Lam)
* 纵向波矢    kz   = sqrt(k0^2 n^2 - kx^2)，传播模取实正，
                    倏逝模取 +i|kz|（即 exp(+i kz z) 沿 +z 衰减）。
* 场按         F(x, z) = sum_m S_m(z) * exp(-i kx_m x)
* 区域 I  入射 exp(+i kIz0 z)，反射 exp(-i kIz_m z)
  区域 II 透射 exp(+i kIIz_m (z - z_bot))
  （该符号已由界面处 z 向导数 U(0) = i P_I (delta - R) 数值定死，误差 1e-16）
* 层内模式    F(z) = W [ a+ exp(-k0 q (z - z_top)) + a- exp(-k0 q (z_bot - z)) ]
  a+ 以层顶为参考、a- 以层底为参考，两者的指数因子模长均 <= 1，天然稳定。

场分量（单位：E 以 Z0 = sqrt(mu0/eps0) 归一，入射主分量振幅为 1）
--------------------------------------------------------------------
  TE: S = Ey,  U = -i * Hx_tilde,  Hz_tilde = -Kx S
  TM: S = Hy,  Ex_tilde = -i U,    Ez_tilde = inv(<<eps>>) Kx S
  TM 的两个电场分量必须用不同算符：Ex 走 <<1/eps>>（已含在 V 里，Dx 连续、
  Ex 不连续），Ez 走 inv(<<eps>>)（Ez 沿竖直侧壁切向连续）。用错算符时
  衍射效率照样收敛，场却是错的。
"""
import numpy as np
from mode_propagation import field_combinations, matrix_propagator
from spectral1d import convmat_fft, validate_fourier_settings, layer_fourier_info

__all__ = ["Layer", "Stack", "solve", "solve_pml", "solve_adaptive_pml", "field_at", "field_map", "eps_profile",
           "RayleighAnomalyError", "DiagonalEps", "uniaxial"]


# =============================================================================
#  1. Fourier 系数与卷积矩阵
# =============================================================================
def fourier_piecewise(edges, values, period, hmax):
    """分段常数剖面的 Fourier 系数（解析积分，无 FFT 混叠）。

    edges  : 长度 K+1 的分界坐标，edges[0] 与 edges[-1] 相差一个周期
    values : 长度 K 的各段取值
    返回   : c[h + hmax]，h = -hmax..hmax，满足 f(x) = sum_h c_h exp(i 2pi h x / P)
    """
    edges = np.asarray(edges, float)
    values = np.asarray(values, complex)
    h = np.arange(-hmax, hmax + 1)
    c = np.zeros(len(h), complex)

    c[hmax] = np.sum(values * np.diff(edges)) / period          # h = 0
    hn = h[h != 0]
    ph = np.exp(-2j * np.pi * np.outer(hn, edges) / period)     # (len(hn), K+1)
    c[h != 0] = (values * (ph[:, 1:] - ph[:, :-1])).sum(axis=1) / (-2j * np.pi * hn)
    return c


def convmat(edges, values, period, M):
    """由分段常数剖面生成 (2M+1)x(2M+1) 的 Toeplitz 卷积矩阵，(m,p) 元 = c_{m-p}。"""
    c = fourier_piecewise(edges, values, period, 2 * M)
    ms = np.arange(-M, M + 1)
    return c[np.subtract.outer(ms, ms) + 2 * M]


# =============================================================================
#  2. 材料与层
# =============================================================================
class DiagonalEps:
    """主轴沿 x、y、z 的对角相对介电张量 diag(εxx, εyy, εzz)（非磁性）。

    一维平面入射下 TE/TM 仍严格解耦：TE（Ey）只感受 εyy；TM 的 Ex 是光栅侧壁
    的法向分量，Dx = εxx Ex 连续，走 <<1/εxx>> 逆规则；Ez 沿侧壁切向，走 <<εzz>>。
    光轴不沿坐标轴（有非对角元）的张量不属于此类。
    """
    __slots__ = ("xx", "yy", "zz")

    def __init__(self, xx, yy, zz):
        values = [complex(v) for v in (xx, yy, zz)]
        if not all(np.isfinite(v) and v != 0 for v in values):
            raise ValueError("介电张量分量必须是有限的非零数")
        self.xx, self.yy, self.zz = values

    def __repr__(self):
        return f"DiagonalEps(xx={self.xx}, yy={self.yy}, zz={self.zz})"

    def __eq__(self, other):
        return isinstance(other, DiagonalEps) and (self.xx, self.yy, self.zz) == (other.xx, other.yy, other.zz)

    def __hash__(self):
        return hash((self.xx, self.yy, self.zz))


def uniaxial(no, ne, axis="z"):
    """单轴晶体：寻常光折射率 no、非常光折射率 ne，光轴沿 'x'、'y' 或 'z'。"""
    if axis not in ("x", "y", "z"):
        raise ValueError("单轴晶体的光轴须为 'x'、'y' 或 'z'")
    o, e = complex(no) ** 2, complex(ne) ** 2
    return DiagonalEps(*(e if a == axis else o for a in "xyz"))


def _eps_triple(value):
    """标量折射率 n 或 DiagonalEps → (εxx, εyy, εzz)。"""
    if isinstance(value, DiagonalEps):
        return value.xx, value.yy, value.zz
    e = value ** 2
    return e, e, e


class Layer:
    """一层。均匀层给 n；光栅层给 (edges, n_values)，坐标为绝对长度。

    n 或 n_values 的条目可以是标量折射率，也可以是 DiagonalEps（例如 uniaxial(no, ne, 'x')）；
    含张量的层 anisotropic=True，目前只用于一维平面 TE/TM 周期求解。
    fourier_method='analytic' 为分段解析积分；'fft' 为实际均匀采样FFT。
    fft_samples=0 自动选点，否则须大于4M。均匀层始终使用解析模式。
    """

    def __init__(self, thickness, n=None, edges=None, n_values=None, name="", *,
                 fourier_method="analytic", fft_samples=0):
        self.d = float(thickness)
        self.n = n
        self.edges = edges
        values = [] if n_values is None else list(n_values)
        self.anisotropic = isinstance(n, DiagonalEps) or any(isinstance(v, DiagonalEps) for v in values)
        if self.anisotropic:
            # 保留原始条目（标量与张量可混合）；eps_diag 为各段 (εxx, εyy, εzz)
            self.n_values = None if n_values is None else np.array(values, dtype=object)
            self.eps_diag = (np.array(_eps_triple(n), complex) if n is not None
                             else np.array([_eps_triple(v) for v in values], complex))
        else:
            self.n_values = None if n_values is None else np.asarray(n_values, complex)
            self.eps_diag = None
        self.name = name
        self.fourier_method, self.fft_samples = validate_fourier_settings(fourier_method,fft_samples)
        self.homogeneous = n is not None
        if not self.homogeneous and (edges is None or n_values is None):
            raise ValueError("光栅层必须给出 edges 与 n_values")

    def operators(self, period, M, pol=None):
        """返回 (<<eps>>, <<1/eps>>)；均匀层返回 None 表示走解析支路。

        各向异性层须给出 pol：TE 返回 (<<εyy>>, <<1/εyy>>)；TM 返回 (<<εzz>>, <<1/εxx>>)，
        即 Ez 所用的 Laurent 阵与 Ex 所用的逆规则阵。各向同性层忽略 pol。
        """
        if self.homogeneous:
            return None, None
        if self.anisotropic:
            if pol not in ("TE", "TM"):
                raise ValueError("各向异性层的 Fourier 算符依赖偏振，请给出 pol='TE' 或 'TM'")
            exx, eyy, ezz = self.eps_diag.T
            laurent, normal = (eyy, eyy) if pol == "TE" else (ezz, exx)
            return self._convmat(laurent, period, M), self._convmat(1.0 / normal, period, M)
        eps = self.n_values ** 2
        # Li factorization requires the Fourier series of 1/eps separately.
        return self._convmat(eps, period, M), self._convmat(1.0 / eps, period, M)

    def _convmat(self, values, period, M):
        if self.fourier_method == "fft":
            return convmat_fft(self.edges, values, period, M, self.fft_samples)
        return convmat(self.edges, values, period, M)


class Stack:
    """nI | layers... | nII"""

    def __init__(self, nI, layers, nII, period):
        self.nI, self.nII, self.layers, self.period = nI, nII, list(layers), float(period)


# =============================================================================
#  3. 模式求解
# =============================================================================
def _branch(q):
    """统一分支：Re q >= 0（倏逝模衰减）；传播模（Re q ~ 0）取 Im q <= 0，
       使 exp(-k0 q z) 对应 exp(+i kz z)，与均匀层的解析支一致。"""
    q = np.where(q.real < 0, -q, q)
    return np.where((q.real < 1e-10) & (q.imag > 0), -q, q)


class RayleighAnomalyError(np.linalg.LinAlgError, ValueError):
    """某均匀介质中有衍射级恰好掠射（kz = 0）：上下行模式重合，模式基退化。

    同时是 LinAlgError（稠密路径在此报奇异矩阵）与 ValueError（参数落在奇异点）。
    """


def rayleigh_message(where, n, orders):
    medium = (f"ε=diag({n.xx:g}, {n.yy:g}, {n.zz:g})" if isinstance(n, DiagonalEps) else f"n={n}")
    return (f"{where}（{medium}）的第 {', '.join(str(int(m)) for m in orders)} 级恰好沿界面掠射"
            f"（kz=0，Rayleigh 异常点），均匀介质的上下行模式在此重合，无法求解。"
            f"请把入射角或周期略作改动（例如入射角改动 0.01°）后重算；"
            f"异常点附近效率随参数变化剧烈，需要时请加密扫描。")


def _homog_modes(n, kx_over_k0, pol, where="均匀层", orders=None, finite_zero_basis=False):
    """均匀层解析模式：W = I，q = -i kz / k0。不做本征分解，既快又稳。

    n 为标量折射率或 DiagonalEps。kz 恰为 0（精确 Rayleigh 异常点）时抛出
    RayleighAnomalyError，而不是输出 NaN；finite_zero_basis=True 仅用于有限层的零根辅助基。
    """
    N = len(kx_over_k0)
    if isinstance(n, DiagonalEps):
        # TE: kz^2 = εyy - kx^2；TM（非常光）: kz^2 = εxx (1 - kx^2/εzz)，V 用 q/εxx
        kz2 = n.yy - kx_over_k0 ** 2 if pol == "TE" else n.xx - (n.xx / n.zz) * kx_over_k0 ** 2
        kz_k0 = np.sqrt(np.asarray(kz2, complex))
        tm_eps = n.xx
    else:
        kz_k0 = np.sqrt((n ** 2 - kx_over_k0 ** 2).astype(complex))
        tm_eps = n ** 2
    grazing = np.flatnonzero(kz_k0 == 0)
    if grazing.size and not finite_zero_basis:
        labels = grazing if orders is None else np.asarray(orders)[grazing]
        raise RayleighAnomalyError(rayleigh_message(where, n, labels))
    kz_k0 = np.where(kz_k0.imag < 0, -kz_k0, kz_k0)      # 倏逝取 +i|kz|
    kz_k0 = np.where((abs(kz_k0.imag) < 1e-12) & (kz_k0.real < 0), -kz_k0, kz_k0)
    q = -1j * kz_k0
    W = np.eye(N, dtype=complex)
    basis_q = np.where(q == 0, 1., q) if finite_zero_basis else q
    V = np.diag(basis_q) if pol == "TE" else np.diag(basis_q / tm_eps)
    return W, V, q


def _is_hermitian(A, tol=1e-13):
    """无损介质的 Fourier 算符是 Hermitian（仅差舍入）；有耗/增益材料不满足。"""
    scale = np.max(np.abs(A))
    return scale == 0 or np.max(np.abs(A - A.conj().T)) <= tol * scale


def _eig_hermitian_definite(B, S):
    """B W = S W diag(w)，B Hermitian、S Hermitian 正定：Cholesky 化为标准 eigh。
    S 非正定时返回 None，由调用方回退到一般 eig。"""
    try:
        L = np.linalg.cholesky(0.5 * (S + S.conj().T))
    except np.linalg.LinAlgError:
        return None
    Li = np.linalg.inv(L)
    C = Li @ (0.5 * (B + B.conj().T)) @ Li.conj().T
    w, U = np.linalg.eigh(0.5 * (C + C.conj().T))
    return w, Li.conj().T @ U


def _grating_modes(Eps, Aeps, Kx, pol, finite_zero_basis=False):
    """光栅层模式。无损介质层（算符 Hermitian）走 eigh 快路径，本征值/场与 eig 一致，
    本征基更好条件；有耗层或负介电常数层回退到原来的一般 eig。"""
    N = Kx.shape[0]
    kx = np.diagonal(Kx)                                   # Kx 对角：矩阵乘改为行列缩放
    if pol == "TE":
        A = np.diag(kx ** 2) - Eps                          # 式 (2.3-21)
        if _is_hermitian(A):
            w2, W = np.linalg.eigh(0.5 * (A + A.conj().T))
        else:
            w2, W = np.linalg.eig(A)
        q = _branch(np.sqrt(w2.astype(complex)))
        basis_q = np.where(q == 0, 1., q) if finite_zero_basis else q
        V = W * basis_q[None, :]                             # 精确零根采用辅助基
    else:                                                   # 式 (2.3-38)(2.3-39)
        Ei = np.linalg.inv(Eps)
        B = kx[:, None] * Ei * kx[None, :] - np.eye(N)
        # inv(Aeps) B W = W diag(w2)  <=>  B W = Aeps W diag(w2)
        modes = (_eig_hermitian_definite(B, Aeps)
                 if _is_hermitian(Eps) and _is_hermitian(Aeps) else None)
        w2, W = modes if modes is not None else np.linalg.eig(np.linalg.solve(Aeps, B))
        q = _branch(np.sqrt(w2.astype(complex)))
        basis_q = np.where(q == 0, 1., q) if finite_zero_basis else q
        V = (Aeps @ W) * basis_q[None, :]
    return W, V, q


# =============================================================================
#  4. S 矩阵：界面块、传播块、Redheffer 星乘（支持对角块的快路径）
# =============================================================================
#  关键优化：均匀层的模式矩阵 W = I，因此
#    * 传播块永远是对角的；
#    * 均匀层之间的界面块也是对角的。
#  于是 MDG 里那 19 层膜堆整段都是对角的，星乘退化成逐元素运算 O(N)，
#  真正 O(N^3) 的稠密运算只剩「入射区↔光栅层」「光栅层↔膜堆」这两个界面。
#  M 上到 100+ 时，这个区分决定了能不能算得动。
# =============================================================================
class S:
    """[out_down; out_up] = S [in_down; in_up]。diag=True 时四个属性存一维对角元。"""
    __slots__ = ("Tdd", "Rdu", "Rud", "Tuu", "diag")

    def __init__(self, Tdd, Rdu, Rud, Tuu, diag=False):
        self.Tdd, self.Rdu, self.Rud, self.Tuu, self.diag = Tdd, Rdu, Rud, Tuu, diag

    def dense(self):
        if not self.diag:
            return self
        return S(np.diag(self.Tdd), np.diag(self.Rdu),
                 np.diag(self.Rud), np.diag(self.Tuu), diag=False)


def _is_identity_diag(W, V):
    """均匀层模式：W 恰为单位阵、V 恰为可逆对角阵（逐元素精确判断，O(N^2)）。
    V 对角含 0（精确 Rayleigh 点）时返回 False，保留原 solve 的奇异报错。"""
    n = W.shape[0]
    d = np.diagonal(V)
    return (np.array_equal(W, np.eye(n)) and np.all(d != 0) and
            not np.count_nonzero(V - np.diag(d)))


def s_interface(W1, V1, W2, V2, both_homog=False, q1=None, q2=None,
                n1=None, n2=None, pol="TE", p1=None, p2=None):
    """层 1（上）→ 层 2（下）的界面散射矩阵。两侧均匀时走对角快路径。

    p1/p2 可直接给出均匀层 V 的对角元（各向异性 TM 为 q/εxx）；否则由 q、n 计算。
    """
    if both_homog:
        if p1 is None:
            p1 = q1 if pol == "TE" else q1 / n1 ** 2
        if p2 is None:
            p2 = q2 if pol == "TE" else q2 / n2 ** 2
        if not (np.all(p1 != 0) and np.all(p2 != 0)):
            # 否则 p1/p2 静默给出 inf/NaN；_homog_modes 已在求解入口拦截同一情形
            raise RayleighAnomalyError("均匀层界面存在 kz=0 的衍射级（Rayleigh 异常点），无法求解；"
                                       "请把入射角或周期略作改动后重算。")
        Nn = p1 / p2                                     # Mm = 1（W = I）
        Tp, Tm = 0.5 * (1 + Nn), 0.5 * (1 - Nn)
        Tpi = 1.0 / Tp
        return S(Tp - Tm * Tpi * Tm, Tm * Tpi, -Tpi * Tm, Tpi, diag=True)
    if _is_identity_diag(W2, V2):
        # 下侧均匀层（含圆锥的均匀层）：W2 = I、V2 对角，两次稠密 solve 退化为行缩放
        Mm = W1
        Nn = V1 / np.diagonal(V2)[:, None]
    else:
        Mm = np.linalg.solve(W2, W1)
        Nn = np.linalg.solve(V2, V1)
    Tp, Tm = 0.5 * (Mm + Nn), 0.5 * (Mm - Nn)
    Tpi = np.linalg.inv(Tp)
    return S(Tp - Tm @ Tpi @ Tm, Tm @ Tpi, -Tpi @ Tm, Tpi)


def s_propagate(q, d, k0, finite_zero_basis=False, generator=None):
    """传播与模式基一致：普通指数、有限层零根辅助基，或完整 Schur 块。"""
    if generator is not None:
        X = matrix_propagator(generator, d, k0)
        Z = np.zeros_like(X)
        return S(X, Z, Z, X, diag=False)
    X = np.exp(-k0 * q * d)
    R = np.zeros_like(X)
    if finite_zero_basis and np.any(q == 0):
        h = k0 * d
        if not np.isfinite(h):
            raise np.linalg.LinAlgError("Nonfinite zero-root layer thickness")
        X[q == 0] = 1. / (1. + .5 * h)
        R[q == 0] = (.5 * h) / (1. + .5 * h)
    return S(X, R, R, X, diag=True)


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


def _rud_below(blk, rud):
    """只递推整段的 Rud：star(blk, below).Rud，够用来闭合场的回代，
       比存整条前后缀 S 链省一个数量级的内存。"""
    if blk.diag and (rud is None or np.ndim(rud) == 1):
        r = 0.0 if rud is None else rud
        G = 1.0 / (1.0 - blk.Rdu * r)
        return blk.Rud + blk.Tuu * r * G * blk.Tdd
    if blk.diag:
        # 对角块（传播块、均匀层间界面）遇到下方稠密 Rud：左右乘对角阵只是行列缩放；
        # 传播块 Rdu = 0 时 G = I，连求逆也省掉
        rG = rud
        if np.any(blk.Rdu):
            N = rud.shape[0]
            rG = rud @ np.linalg.inv(np.eye(N, dtype=complex) - blk.Rdu[:, None] * rud)
        return np.diag(blk.Rud) + blk.Tuu[:, None] * rG * blk.Tdd[None, :]
    A = blk.dense()
    N = A.Tdd.shape[0]
    r = np.zeros((N, N), complex) if rud is None else (np.diag(rud) if np.ndim(rud) == 1 else rud)
    G = np.linalg.inv(np.eye(N, dtype=complex) - A.Rdu @ r)
    return A.Rud + A.Tuu @ r @ G @ A.Tdd


def _apply(m, v):
    return m * v if np.ndim(m) == 1 else m @ v


def _step_down(blk, rud_next, down):
    """由接缝 k 的下行振幅推到接缝 k+1。"""
    if blk.diag and (rud_next is None or np.ndim(rud_next) == 1):
        r = 0.0 if rud_next is None else rud_next
        return blk.Tdd * down / (1.0 - blk.Rdu * r)
    if blk.diag:
        rhs = (blk.Tdd * down if np.ndim(down) == 1      # 对角块 + 稠密 Rud（见 _rud_below）
               else blk.Tdd[:, None] * down)
        if not np.any(blk.Rdu):
            return rhs
        N = rud_next.shape[0]
        return np.linalg.solve(np.eye(N, dtype=complex) - blk.Rdu[:, None] * rud_next, rhs)
    A = blk.dense()
    N = A.Tdd.shape[0]
    r = np.zeros((N, N), complex) if rud_next is None else (
        np.diag(rud_next) if np.ndim(rud_next) == 1 else rud_next)
    return np.linalg.solve(np.eye(N, dtype=complex) - A.Rdu @ r, A.Tdd @ down)


# =============================================================================
#  5. 求解
# =============================================================================
class Result(dict):
    __getattr__ = dict.get


def solve(stack, lam0, theta_deg, M, pol="TE", need_fields=True):
    """返回衍射效率；need_fields=True 时同时给出各层模式与模式系数。"""
    k0 = 2 * np.pi / lam0
    th = np.radians(theta_deg)
    ms = np.arange(-M, M + 1)
    N = len(ms)
    nI, nII, P = stack.nI, stack.nII, stack.period
    if isinstance(nI, DiagonalEps) or isinstance(nII, DiagonalEps):
        raise ValueError("入射区与出射区须为各向同性介质（标量折射率）；各向异性材料请放在层内")

    kxm = k0 * (nI * np.sin(th) - ms * lam0 / P)
    kx_k0 = kxm / k0
    Kx = np.diag(kx_k0)

    WI, VI, qI = _homog_modes(nI, kx_k0, pol, "入射区", ms)
    WII, VII, qII = _homog_modes(nII, kx_k0, pol, "出射区", ms)

    mods, homog, idx = [], [], []
    for j, L in enumerate(stack.layers):
        if L.homogeneous:
            where = f"第{j + 1}层" + (f"「{L.name}」" if L.name else "")
            mods.append(_homog_modes(L.n, kx_k0, pol, where, ms, finite_zero_basis=True)); homog.append(True); idx.append(L.n)
        else:
            Eps, Aeps = L.operators(P, M, pol)
            mods.append(_grating_modes(Eps, Aeps, Kx, pol, finite_zero_basis=True)); homog.append(False); idx.append(None)

    # 块序列：界面, 层, 界面, 层, ..., 界面
    blocks = []
    Wp, Vp, qp, np_, hp = WI, VI, qI, nI, True
    for (W, V, q), L, hg, nn in zip(mods, stack.layers, homog, idx):
        # 均匀层 V 为对角阵，其对角元即界面所需的 p（各向异性时含 1/εxx）
        blocks.append(s_interface(Wp, Vp, W, V, both_homog=(hp and hg),
                                  q1=qp, q2=q, n1=np_, n2=nn, pol=pol,
                                  p1=np.diagonal(Vp) if hp else None,
                                  p2=np.diagonal(V) if hg else None))
        blocks.append(s_propagate(q, L.d, k0, finite_zero_basis=True))
        Wp, Vp, qp, np_, hp = W, V, q, nn, hg
    blocks.append(s_interface(Wp, Vp, WII, VII, both_homog=hp,
                              q1=qp, q2=qII, n1=np_, n2=nII, pol=pol,
                              p1=np.diagonal(Vp) if hp else None, p2=np.diagonal(VII)))

    # 自下而上只递推 Rud（对角段保持一维，内存与时间都省一个数量级）
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
    den = k0 * nI * np.cos(th)
    DEr = np.abs(R) ** 2 * np.real(kIz / den)
    if pol == "TE":
        DEt = np.abs(T) ** 2 * np.real(kIIz / den)
    else:
        DEt = np.abs(T) ** 2 * np.real((kIIz / nII ** 2) / (k0 * np.cos(th) / nI))

    out = Result(ms=ms, R=R, T=T, DEr=DEr, DEt=DEt, kxm=kxm, kIz=kIz, kIIz=kIIz,
                 k0=k0, pol=pol, Kx=Kx, stack=stack, M=M, theta=th)
    out["fourier_layers"] = [layer_fourier_info(L,P,M) for L in stack.layers]
    out["n_zero_modes"] = sum(int(np.count_nonzero(q == 0)) for _, _, q in mods)
    if not need_fields:
        return out

    amps, ztop = [], 0.0
    for j, L in enumerate(stack.layers):
        ap = downs[2 * j + 1]                                  # 层顶下行 = a+
        am = _apply(rud[2 * j + 2], downs[2 * j + 2])          # 层底上行 = a-
        amps.append(dict(a_plus=ap, a_minus=am, W=mods[j][0], V=mods[j][1],
                         q=mods[j][2], d=L.d, z_top=ztop, layer=L, finite_zero_basis=True))
        ztop += L.d
    out["amps"] = amps
    out["z_bot"] = ztop
    out["WI"], out["VI"], out["qI"] = WI, VI, qI
    out["WII"], out["VII"], out["qII"] = WII, VII, qII
    return out


# =============================================================================
#  6. 场重建
# =============================================================================
def field_at(res, z):
    """返回 z 处的谐波列 (S, U)。z<0 为区域 I，z>z_bot 为区域 II。"""
    if res.get("is_pml_jones", False):
        raise ValueError("混合偏振没有单一的 (S, U)；请分别对 res['TE']、res['TM'] 调用 field_at")
    if res.get("is_pml", False):
        from pml.afmm1d import field_at as pml_field_at
        return pml_field_at(res, z)
    k0, dlt = res["k0"], (res["ms"] == 0).astype(complex)
    if z < 0:
        incoming = np.zeros_like(dlt)
        active = dlt != 0
        incoming[active] = np.exp(1j * res["kIz"][active] * z)
        e_re = np.exp(-1j * res["kIz"] * z)
        Sv = incoming + res["R"] * e_re
        P = res["kIz"] / k0 if res["pol"] == "TE" else res["kIz"] / (k0 * res["stack"].nI ** 2)
        return Sv, 1j * P * (incoming - res["R"] * e_re)
    if z > res["z_bot"]:
        e = np.exp(1j * res["kIIz"] * (z - res["z_bot"]))
        P = res["kIIz"] / k0 if res["pol"] == "TE" else res["kIIz"] / (k0 * res["stack"].nII ** 2)
        return res["T"] * e, 1j * P * res["T"] * e
    for a in res["amps"]:
        if z <= a["z_top"] + a["d"] + 1e-12:
            zl = z - a["z_top"]
            summed, difference = field_combinations(
                a["q"], a["a_plus"], a["a_minus"], zl, a["d"], k0,
                zero_basis=a.get("finite_zero_basis", False))
            Sv = a["W"] @ summed[:, 0]
            Uv = a["V"] @ difference[:, 0]
            return Sv, Uv
    raise ValueError("z 超出结构范围")


def _eps_ops_at(res, z):
    """z 处的 <<eps>> 与 <<1/eps>>（TM 的 Ez 需要 inv(<<eps>>)）。

    各向异性层按 Layer.operators 的约定：TE 为 (εyy, 1/εyy)，TM 为 (εzz, 1/εxx)。
    """
    M, P = res["M"], res["stack"].period
    N = 2 * M + 1
    if z < 0:
        n = res["stack"].nI
    elif z > res["z_bot"]:
        n = res["stack"].nII
    else:
        for a in res["amps"]:
            if z <= a["z_top"] + a["d"] + 1e-12:
                L = a["layer"]
                if not L.homogeneous:
                    return L.operators(P, M, res["pol"])
                n = L.n
                break
    I = np.eye(N, dtype=complex)
    if isinstance(n, DiagonalEps):
        laurent, normal = (n.yy, n.yy) if res["pol"] == "TE" else (n.zz, n.xx)
        return laurent * I, I / normal
    return n ** 2 * I, I / n ** 2


def field_map(res, zs, nx=256):
    """在 (x, z) 网格上重建实空间场。返回 dict，含 |E|^2（对入射 |E|^2 归一）。"""
    if res.get("is_pml_jones", False):
        from pml.jones import field_map as jones_field_map
        return jones_field_map(res, zs, nx=nx)
    if res.get("is_pml", False):
        from pml.afmm1d import field_map as pml_field_map
        return pml_field_map(res, zs, nx=nx)
    P = res["stack"].period
    x = np.arange(nx) / nx * P
    ph = np.exp(-1j * np.outer(x, res["kxm"]))              # (nx, N)
    zs = np.atleast_1d(zs)
    comp = {k: np.zeros((len(zs), nx), complex) for k in
            (("Ey", "Hx", "Hz") if res["pol"] == "TE" else ("Hy", "Ex", "Ez"))}
    # Keep the first layer owning an interface, including the historical tolerance.
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
        raise ValueError("z 超出结构范围")

    k0 = res["k0"]
    kx = np.diag(res["Kx"])
    for group, (indices, amp) in enumerate(groups):
        if not len(indices):
            continue
        z = zs[indices]
        if group == 0:
            ere = np.exp(-1j * res["kIz"][:, None] * z)
            incoming = np.zeros((len(res["ms"]), len(z)), complex)
            active = res["ms"] == 0
            incoming[active] = np.exp(1j * res["kIz"][active, None] * z)
            reflected = res["R"][:, None] * ere
            Sv = incoming + reflected
            n = res["stack"].nI
            p = res["kIz"] / (k0 if res["pol"] == "TE" else k0 * n ** 2)
            Uv = 1j * p[:, None] * (incoming - reflected)
        elif group == 1:
            e = np.exp(1j * res["kIIz"][:, None] * (z - res["z_bot"]))
            Sv = res["T"][:, None] * e
            n = res["stack"].nII
            p = res["kIIz"] / (k0 if res["pol"] == "TE" else k0 * n ** 2)
            Uv = 1j * p[:, None] * Sv
        else:
            zl = z - amp["z_top"]
            summed, difference = field_combinations(
                amp["q"], amp["a_plus"], amp["a_minus"], zl, amp["d"], k0,
                zero_basis=amp.get("finite_zero_basis", False))
            Sv = amp["W"] @ summed
            Uv = amp["V"] @ difference
            n = amp["layer"].n if amp["layer"].homogeneous else None

        if res["pol"] == "TE":
            comp["Ey"][indices] = (ph @ Sv).T
            comp["Hx"][indices] = (ph @ (1j * Uv)).T
            comp["Hz"][indices] = (ph @ (-kx[:, None] * Sv)).T
        else:
            rhs = kx[:, None] * Sv
            if n is not None:                           # Ez 由 Dz = εzz Ez 给出
                ez = rhs / (n.zz if isinstance(n, DiagonalEps) else n ** 2)
            else:
                Eps, _ = amp["layer"].operators(P, res["M"], "TM")
                ez = np.linalg.solve(Eps, rhs)
            comp["Hy"][indices] = (ph @ Sv).T
            comp["Ex"][indices] = (ph @ (-1j * Uv)).T
            comp["Ez"][indices] = (ph @ ez).T
    if res["pol"] == "TE":
        comp["E2"] = np.abs(comp["Ey"]) ** 2                 # 入射 |Ey| = 1
    else:
        comp["E2"] = (np.abs(comp["Ex"]) ** 2 + np.abs(comp["Ez"]) ** 2) * res["stack"].nI ** 2
    comp["x"], comp["z"] = x, zs
    return comp


def eps_profile(res, zs, nx=256, component=None):
    """与 field_map 同网格的 n(x, z)，用于把场图与几何套准。

    各向异性层返回 sqrt(ε_component)，component 为 'xx'、'yy' 或 'zz'；默认取主场
    感受的分量（TE 为 'yy'，TM 为 'xx'）。各向同性层不受 component 影响。
    """
    if res.get("is_pml_jones", False):
        res = res["TE"]                                  # 几何与偏振无关
    if res.get("is_pml", False):
        P = res["period"]
        x = res["pml"].x0 + np.arange(nx) / nx * P
        out = np.empty((len(np.atleast_1d(zs)), nx), complex)
        from pml.afmm1d import _at
        for i, z in enumerate(np.atleast_1d(zs)):
            _, _, mode, _ = _at(res, float(z))
            ep, mu = mode["layer"].profile(x, P)
            out[i] = np.sqrt(ep * mu + 0j)
        return x, out
    P = res["stack"].period
    x = np.arange(nx) / nx * P
    out = np.zeros((len(np.atleast_1d(zs)), nx), complex)
    for i, z in enumerate(np.atleast_1d(zs)):
        if z < 0:
            out[i] = res["stack"].nI
        elif z > res["z_bot"]:
            out[i] = res["stack"].nII
        else:
            for a in res["amps"]:
                if z <= a["z_top"] + a["d"] + 1e-12:
                    L = a["layer"]
                    if L.anisotropic:
                        comp = component or ("yy" if res["pol"] == "TE" else "xx")
                        column = np.sqrt(np.atleast_2d(L.eps_diag)[:, "xyz".index(comp[0])])
                        if L.homogeneous:
                            out[i] = column[0]
                        else:
                            idx = np.searchsorted(np.asarray(L.edges), x % P, side="right") - 1
                            out[i] = column[np.clip(idx, 0, len(column) - 1)]
                    elif L.homogeneous:
                        out[i] = L.n
                    else:
                        idx = np.searchsorted(np.asarray(L.edges), x % P, side="right") - 1
                        out[i] = L.n_values[np.clip(idx, 0, len(L.n_values) - 1)]
                    break
    return x, out


def solve_pml(stack, lam0, theta_deg, M, pol="TE", *, pml, background=None,
              need_fields=True, psi_deg=45.0, delta_deg=0.0, jones_in=None):
    """横向 PML 非周期分支。pml 用 pml.LateralPML；background 指定各层背景。

    支持一维 TE/TM 与斜入射。返回物理总场及对比场，不返回周期衍射级效率。
    pol="jones" 为混合偏振：同一离散下分别求 TE、TM 并按 Jones 矢量严格叠加
    （psi/delta 或 jones_in=[Es,Ep]，约定与圆锥模式相同），耗时约为两倍。
    详细限制与独立解析验证写在作者早先的 Python 包的 PML 说明里，尚未收入本仓库。
    """
    if pol == "jones":
        from pml.jones import solve_pml_jones
        return solve_pml_jones(stack, lam0, theta_deg, M, pml=pml, background=background,
                               psi_deg=psi_deg, delta_deg=delta_deg, jones_in=jones_in,
                               need_fields=need_fields)
    from pml.afmm1d import solve_pml as implementation
    return implementation(stack, lam0, theta_deg, M, pol, pml=pml,
                          background=background, need_fields=need_fields)


def solve_adaptive_pml(stack,lam0,theta_deg,M,pol="TE",*,pml,background=None,settings=None,on_progress=None,
                       psi_deg=45.0,delta_deg=0.0,jones_in=None):
    """Automatic lateral PML sizing; inspect result.adaptive for achieved status.

    pol="jones" alternates TE/TM searches until both accept one common setup.
    """
    if pol=="jones":
        from pml.jones import solve_adaptive_pml_jones
        return solve_adaptive_pml_jones(stack,lam0,theta_deg,M,pml=pml,background=background,
            settings=settings,on_progress=on_progress,psi_deg=psi_deg,delta_deg=delta_deg,jones_in=jones_in)
    from pml.adaptive import solve_adaptive_pml as implementation
    return implementation(stack,lam0,theta_deg,M,pol,pml=pml,background=background,
                          settings=settings,on_progress=on_progress)
