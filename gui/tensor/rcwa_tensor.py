# -*- coding: utf-8 -*-
"""一维光栅、一般线性介质（6×6 本构）的圆锥入射 RCWA（实验分支，纯 NumPy）。

约定与 ``rcwa1d.py``、``conical/`` 相同：时间因子 exp(-iωt)，z 自入射侧指向出射侧，
结构上表面为 z = 0；谐波为 exp[-i k0 (kx_m x + ky y)]，
kx_m = nI sinθ cosφ − m λ/Λ，ky = nI sinθ sinφ（均以 k0 为单位）。
磁场一律写成 Z0·H，下面记作 H。

材料
----
每段材料是一个 6×6 复矩阵 C：

    [D/ε0; c·B] = C [E; Z0 H]，    C = [[ε, ξ], [ζ, μ]]

ε、μ 是任意 3×3 复矩阵（不要求对称，可表示任意取向的单轴/双轴晶体、磁各向异性
和磁光材料）；ξ、ζ 是磁电耦合（手性、双各向异性）。无损对应 C 为 Hermitian。

方法
----
1. Fourier 因式分解。侧壁法向为 x，跨侧壁连续的是 Dx、Bx 与 Ey、Ez、Hy、Hz。逐段
   对 {Ex, Hx} 做部分求逆（principal pivot transform），使乘积都是“间断系数 × 连续
   场”，逐元素取 Laurent 卷积阵，再在块矩阵层面部分求逆回来，得到 6N×6N 的 Ĉ。
   各向同性时退化为 ⟦1/ε⟧⁻¹（Ex）与 ⟦ε⟧（Ey、Ez），即现有求解器的 Li 规则。
2. 模式方程。状态 ψ = [Ex; Ey; Hx; Hy]。Maxwell 方程的两个 z 分量不含 ∂z，
   用它们消去 [Ez; Hz]：

       dψ/dz = i k0 M ψ，   J M = (Ĉ − K) 对法向块的 Schur 补

   K 只含 kx、ky；J 是 z 向能流的二次型，ψ†Jψ = 2 Re(Ex Hy* − Ey Hx*)。
   无损时 J M 为 Hermitian，截断后的系统严格守恒能量。
3. 模式与级联。对 M 做 4N 阶一般本征分解，模式 ∝ exp(i k0 γ z)。倏逝模按 Im γ 的
   符号、传播模按 z 向能流的符号分成下行、上行各 2N 个（一般介质里上下行不成
   ±γ 对）。界面用一次 4N 阶线性求解，自下而上递推反射阵，只出现模长 ≤ 1 的指数。

范围：入射区为各向同性无损介质（实折射率）；各层与出射区可以是任意 Medium。
出射区各向异性时不给 s/p Jones 矩阵，只给按 z 向能流计算的各级效率。

已知的限制：
· 出射区必须无源（有增益时辐射条件没有定义，直接报错）；层内材料有增益时给出告警，结果是频域方程的形式解。
· 无损的双曲材料（主介电常数异号）作光栅齿时，结果随截断阶数 M 不收敛，能量却始终守恒；这是截断问题本身的
  性质（主轴沿坐标轴时与 rcwa1d 逐位一致），加一点吸收后恢复收敛。均匀的双曲膜与基底不受影响。
· 某一级在某个均匀层或半空间里恰好截止（传播常数为零）时报 RayleighAnomalyError。
"""
from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from rcwa1d import DiagonalEps, RayleighAnomalyError, Result, convmat, rayleigh_message

__all__ = [
    "Medium", "isotropic", "uniaxial", "biaxial", "gyrotropic", "chiral", "rotation_matrix",
    "TensorLayer", "TensorStack", "ResultTensor", "solve_tensor", "field_at", "field_map",
    "constitutive_operator", "maxwell_generator",
]

# 六分量的次序：Ex, Ey, Ez, Hx, Hy, Hz（本构的输出侧相应为 Dx, Dy, Dz, Bx, By, Bz）
_T = (0, 1, 3, 4)       # 切向状态 ψ
_Z = (2, 5)             # 法向 Ez, Hz
_P = (0, 3)             # 侧壁法向 x 的分量：Ex、Hx 间断，Dx、Bx 连续
_R = (1, 2, 4, 5)


# =============================================================================
#  1. 材料
# =============================================================================
def rotation_matrix(axis, angle_deg):
    """绕 axis（'x'、'y'、'z' 或三维矢量）按右手定则转 angle_deg 的 3×3 旋转矩阵。"""
    if isinstance(axis, str):
        axis = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    a = np.asarray(axis, float)
    norm = np.linalg.norm(a)
    if a.shape != (3,) or not np.isfinite(norm) or norm == 0:
        raise ValueError("旋转轴须为 'x'、'y'、'z' 或非零三维矢量")
    a = a / norm
    t = np.radians(angle_deg)
    cross = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.cos(t) * np.eye(3) + np.sin(t) * cross + (1 - np.cos(t)) * np.outer(a, a)


def _as_tensor(value, what):
    a = np.asarray(value, complex)
    if a.ndim == 0:
        tensor = a * np.eye(3)
    elif a.shape == (3,):
        tensor = np.diag(a)
    elif a.shape == (3, 3):
        tensor = a.copy()
    else:
        raise ValueError(f"{what} 须为标量、三个主值或 3×3 矩阵")
    if not np.isfinite(tensor).all():
        raise ValueError(f"{what} 含非有限值")
    return tensor


class Medium:
    """一般线性介质：[D/ε0; c·B] = C [E; Z0 H]，C = [[eps, xi], [zeta, mu]]。

    eps、mu 可给标量、三个主值或 3×3 矩阵；xi、zeta 缺省为零。全部是相对量，
    按 exp(-iωt) 约定（吸收对应 Im ε > 0）。
    """
    __slots__ = ("C", "name")

    def __init__(self, eps=1.0, mu=1.0, xi=None, zeta=None, name=""):
        C = np.zeros((6, 6), complex)
        C[:3, :3] = _as_tensor(eps, "eps")
        C[3:, 3:] = _as_tensor(mu, "mu")
        if xi is not None:
            C[:3, 3:] = _as_tensor(xi, "xi")
        if zeta is not None:
            C[3:, :3] = _as_tensor(zeta, "zeta")
        self.C, self.name = C, name

    eps = property(lambda self: self.C[:3, :3])
    xi = property(lambda self: self.C[:3, 3:])
    zeta = property(lambda self: self.C[3:, :3])
    mu = property(lambda self: self.C[3:, 3:])

    def rotated(self, R):
        """把材料整体按旋转矩阵 R 转动（四个 3×3 块都按 R·X·Rᵀ 变换）。R 须为正常旋转。"""
        R = np.asarray(R, float)
        if R.shape != (3, 3) or abs(np.linalg.det(R) - 1) > 1e-10 or not np.allclose(R @ R.T, np.eye(3), atol=1e-10):
            raise ValueError("R 须为 3×3 正常旋转矩阵（行列式为 +1）")
        T = np.zeros((6, 6))
        T[:3, :3] = T[3:, 3:] = R
        out = Medium(name=self.name)
        out.C = T @ self.C @ T.T
        return out

    @property
    def scalar(self):
        """各向同性且无磁电耦合时返回 (ε, μ)，否则 None。"""
        e, m = self.C[0, 0], self.C[3, 3]
        ideal = np.zeros((6, 6), complex)
        ideal[:3, :3], ideal[3:, 3:] = e * np.eye(3), m * np.eye(3)
        if np.max(np.abs(self.C - ideal)) <= 1e-13 * np.max(np.abs(self.C)):
            return complex(e), complex(m)
        return None

    @property
    def lossless(self):
        return bool(np.max(np.abs(self.C - self.C.conj().T)) <= 1e-13 * np.max(np.abs(self.C)))

    @property
    def passive(self):
        """无源（不含增益）：(C − C†)/(2i) 半正定。只给 ξ 不给 ζ = ξ† 的材料通常不满足。"""
        loss = (self.C - self.C.conj().T) / 2j
        return bool(np.linalg.eigvalsh(loss).min() >= -1e-12 * np.max(np.abs(self.C)))

    def __repr__(self):
        s = self.scalar
        if s is not None:
            return f"Medium(eps={s[0]:g}, mu={s[1]:g})"
        return "Medium(" + (self.name or "tensor") + ")"


def isotropic(n, mu=1.0):
    """折射率 n、相对磁导率 mu 的各向同性介质（ε = n²/μ）。"""
    return Medium(eps=complex(n) ** 2 / complex(mu), mu=mu)


def uniaxial(no, ne, axis="z"):
    """单轴晶体：ε = no²·I + (ne² − no²)·a aᵀ，a 为光轴方向（'x'、'y'、'z' 或任意三维矢量）。"""
    if isinstance(axis, str):
        axis = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    a = np.asarray(axis, float)
    if a.shape != (3,) or np.linalg.norm(a) == 0:
        raise ValueError("光轴须为 'x'、'y'、'z' 或非零三维矢量")
    a = a / np.linalg.norm(a)
    o, e = complex(no) ** 2, complex(ne) ** 2
    return Medium(eps=o * np.eye(3) + (e - o) * np.outer(a, a))


def biaxial(n1, n2, n3, R=None):
    """双轴晶体：主折射率 n1、n2、n3，主轴为 R 的三列（缺省为坐标轴）。"""
    medium = Medium(eps=[complex(n1) ** 2, complex(n2) ** 2, complex(n3) ** 2])
    return medium if R is None else medium.rotated(R)


def gyrotropic(eps, g):
    """磁光（回旋）介质：D = ε·E + i g × E，g 为回旋矢量；g 为实数时无损但非互易。"""
    g = np.asarray(g, complex)
    if g.shape != (3,):
        raise ValueError("回旋矢量 g 须有三个分量")
    cross = np.array([[0, -g[2], g[1]], [g[2], 0, -g[0]], [-g[1], g[0], 0]])
    return Medium(eps=_as_tensor(eps, "eps") + 1j * cross)


def chiral(eps, kappa, mu=1.0):
    """各向同性手性（Pasteur）介质：D = εE + iκH，B = −iκE + μH；本征折射率 √(εμ) ± κ。"""
    return Medium(eps=eps, mu=mu, xi=1j * complex(kappa), zeta=-1j * complex(kappa))


def as_medium(value):
    if isinstance(value, Medium):
        return value
    if isinstance(value, DiagonalEps):
        return Medium(eps=[value.xx, value.yy, value.zz])
    return isotropic(value)


# =============================================================================
#  2. 层与堆栈
# =============================================================================
class TensorLayer:
    """一层。均匀层给 medium；光栅层给 edges 与 media（每段一个），坐标为绝对长度。

    条目可以是 Medium、标量折射率或 rcwa1d.DiagonalEps。
    """

    def __init__(self, thickness, medium=None, edges=None, media=None, name=""):
        self.d, self.name = float(thickness), name
        if not np.isfinite(self.d) or self.d < 0:
            raise ValueError("层厚须为非负有限数")
        if medium is not None:
            if edges is not None or media is not None:
                raise ValueError("均匀层只给 medium；光栅层只给 edges 与 media")
            self.edges, self.media = None, [as_medium(medium)]
        else:
            if edges is None or media is None:
                raise ValueError("光栅层必须给出 edges 与 media")
            self.edges = np.asarray(edges, float)
            self.media = [as_medium(m) for m in media]
            if self.edges.ndim != 1 or len(self.media) != len(self.edges) - 1 or np.any(np.diff(self.edges) <= 0):
                raise ValueError("edges 须严格递增，且段数与 media 个数一致")
        self.homogeneous = all(np.array_equal(m.C, self.media[0].C) for m in self.media)


class TensorStack:
    """nI | layers… | nII。nI 为实折射率；nII 为标量折射率或 Medium。"""

    def __init__(self, nI, layers, nII, period):
        self.nI, self.nII, self.layers, self.period = nI, nII, list(layers), float(period)


# =============================================================================
#  3. 算符
# =============================================================================
def _blocks(components, N):
    return np.concatenate([np.arange(c * N, (c + 1) * N) for c in components])


def _pivot(A, p, r):
    """对指标集 p 做部分求逆：把 [g_p; g_r] = A [f_p; f_r] 改写成 [f_p; g_r] = A' [g_p; f_r]。

    该变换是对合的（再做一次回到 A），标量逐段与块矩阵两处共用。
    """
    pp, pr, rp, rr = np.ix_(p, p), np.ix_(p, r), np.ix_(r, p), np.ix_(r, r)
    inv = np.linalg.inv(A[pp])
    out = np.empty_like(A)
    out[pp] = inv
    out[pr] = -inv @ A[pr]
    out[rp] = A[rp] @ inv
    out[rr] = A[rr] - A[rp] @ inv @ A[pr]
    return out


def constitutive_operator(layer, period, M, factorization="li"):
    """光栅层的 6N×6N 本构算符 Ĉ（块 (i, j) 为 N×N，i、j 按 Ex…Hz 排）。

    'li'：对侧壁法向分量部分求逆后取 Laurent 卷积阵，再逆回来；
    'laurent'：逐元素直接取卷积阵（收敛差，仅作对照）。
    """
    if factorization not in ("li", "laurent"):
        raise ValueError("factorization 必须是 'li' 或 'laurent'")
    N = 2 * M + 1
    mats = np.array([m.C for m in layer.media])
    if factorization == "li":
        for k, c in enumerate(mats):
            block = c[np.ix_(_P, _P)]
            if abs(np.linalg.det(block)) <= 1e-14 * max(np.max(np.abs(block)) ** 2, 1e-300):
                raise ValueError(f"第 {k + 1} 段材料的法向块 [[εxx, ξxx], [ζxx, μxx]] 奇异，无法做逆规则")
        mats = np.array([_pivot(c, list(_P), list(_R)) for c in mats])
    op = np.zeros((6 * N, 6 * N), complex)
    for i in range(6):
        for j in range(6):
            values = mats[:, i, j]
            if np.any(values != 0):
                op[i * N:(i + 1) * N, j * N:(j + 1) * N] = convmat(layer.edges, values, period, M)
    return _pivot(op, _blocks(_P, N), _blocks(_R, N)) if factorization == "li" else op


def maxwell_generator(Chat, kx, ky):
    """由本构算符得到 dψ/dz = i k0 M ψ 的 M（4N×4N）与法向重建阵 Z（[Ez; Hz] = Z ψ）。

    Maxwell 的 z 分量给出 Dz = kx·Hy − ky·Hx、Bz = ky·Ex − kx·Ey，即 [Dz; Bz] = Kn ψ。
    记 H = Ĉ − K（K 的非零块为 Kn 与 Knᵀ），则 J M = H_tt − H_tn H_nn⁻¹ H_nt，Z = −H_nn⁻¹ H_nt。
    """
    kx = np.atleast_1d(np.asarray(kx, float))
    N = len(kx)
    t, n = _blocks(_T, N), _blocks(_Z, N)
    Kx, Ky, O = np.diag(kx).astype(complex), ky * np.eye(N, dtype=complex), np.zeros((N, N), complex)
    Kn = np.block([[O, O, -Ky, Kx], [Ky, -Kx, O, O]])
    Htt = Chat[np.ix_(t, t)]
    Htn = Chat[np.ix_(t, n)] - Kn.T
    Hnt = Chat[np.ix_(n, t)] - Kn
    X = np.linalg.solve(Chat[np.ix_(n, n)], Hnt)
    JM = Htt - Htn @ X
    # 左乘 J：四个方程依次取 +By、−Bx、−Dy、+Dx 一行
    Mgen = np.vstack((JM[3 * N:], -JM[2 * N:3 * N], -JM[N:2 * N], JM[:N]))
    return Mgen, -X


# =============================================================================
#  4. 模式
# =============================================================================
class _Modes:
    """一层的模式：ψ = Fp·(a ∘ e^{i k0 γp z}) + Fm·(b ∘ e^{i k0 γm (z − d)})，a 以层顶、b 以层底为参考。"""
    __slots__ = ("Fp", "gp", "Fm", "gm", "kind", "medium", "Chat", "residual", "normal")

    def __init__(self, Fp, gp, Fm, gm, kind, medium=None, Chat=None, residual=0.0):
        self.Fp, self.gp, self.Fm, self.gm = Fp, gp, Fm, gm
        self.kind, self.medium, self.Chat, self.residual = kind, medium, Chat, residual
        self.normal = None          # 场重建用的 (Ĉ, Z)，第一次用到时才算


def _order_flux(psi):
    """各级次的 z 向能流因子 Re(Ex Hy* − Ey Hx*)；psi 为 (4N,) 或 (4N, k)。"""
    n = psi.shape[0] // 4
    ex, ey, hx, hy = psi[:n], psi[n:2 * n], psi[2 * n:3 * n], psi[3 * n:]
    return np.real(ex * hy.conj() - ey * hx.conj())


def _flux_form(A, B):
    """z 向能流的二次型 A†·J·B；对角元是 2 Re(Ex Hy* − Ey Hx*)。"""
    n = B.shape[0] // 4
    return A.conj().T @ np.vstack((B[3 * n:], -B[2 * n:3 * n], -B[n:2 * n], B[:n]))


def _split_modes(gamma, F, Mgen, Z, where):
    """把本征模分成下行（沿 +z 衰减或输运能量）与上行两组，各占一半。Z 把 ψ 映到法向分量 [Ez; Hz]。"""
    half = len(gamma) // 2
    gamma, F = gamma.copy(), F.copy()
    scale = 1 + np.abs(gamma)
    evanescent = np.abs(gamma.imag) > 1e-9 * scale
    # 传播常数相同的一组传播模：本征矢只是这个子空间的任意一组基，可能把上行和下行混在一起
    # （无损各向异性介质里前向波与后向波恰好简并时）。按极限吸收原理取基：给介质加一点各向同性的损耗，
    # 简并一阶劈裂后的本征矢，就是把能流二次型与六分量场的模方同时对角化的那一组。
    order = np.flatnonzero(~evanescent)
    order = order[np.argsort(gamma[order].real)]
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and abs(gamma[order[stop]] - gamma[order[stop - 1]]) < 1e-7 * scale[order[stop]]:
            stop += 1
        group = order[start:stop]
        start = stop
        if len(group) > 1:
            sub = F[:, group]
            form = _flux_form(sub, sub)
            normal = Z @ sub
            L = np.linalg.cholesky(sub.conj().T @ sub + normal.conj().T @ normal)
            Li = np.linalg.inv(L)
            w, U = np.linalg.eigh(Li @ (0.5 * (form + form.conj().T)) @ Li.conj().T)
            if w.min() < 0 < w.max():
                sub = sub @ (Li.conj().T @ U)
                sub /= np.linalg.norm(sub, axis=0)
                F[:, group] = sub
                gamma[group] = np.einsum("ij,ij->j", sub.conj(), Mgen @ sub)      # 各列自己的 Rayleigh 商
    flux = np.real(np.diagonal(_flux_form(F, F))) / np.sum(np.abs(F) ** 2, axis=0)
    if np.any(~evanescent & (np.abs(flux) < 1e-9)):
        raise RayleighAnomalyError(
            f"{where}有模式恰在截止点（传播常数为实数而 z 向能流为零），上下行无法区分；"
            f"请把入射角、波长或周期略作改动后重算。")
    down = np.where(evanescent, gamma.imag > 0, flux > 0)
    if np.count_nonzero(down) != half:
        raise np.linalg.LinAlgError(
            f"{where}的下行模式有 {np.count_nonzero(down)} 个，应为 {half} 个，上下行分不开；"
            f"常见原因是模式基退化。")
    return gamma[down], F[:, down], gamma[~down], F[:, ~down]


def _isotropic_modes(eps, mu, kx, ky, where, orders, basis="xy"):
    """各向同性介质（标量 ε、μ）的解析模式。

    basis='xy'：取 [Ex; Ey] 的谐波作模式振幅，F± = [I; ±Y]。用于入射区和出射区，外面按这个含义读振幅。
    basis='sp'：每级一个 s 模、一个 p 模，列里不出现 1/γ。用于层：某一级接近截止（γ → 0）时 Y 的两列都按
    1/γ 变大而且几乎平行，圆锥入射下 [I; ±Y] 会把有效数字丢光，s/p 基没有这个问题。
    """
    N = len(kx)
    g = np.sqrt((eps * mu - kx ** 2 - ky ** 2).astype(complex))
    g = np.where(g.imag < 0, -g, g)
    # 传播波按能流定向：z 向能流 ∝ Re(γ/μ)。ε、μ 同为负的介质里，朝 +z 输运能量的波 γ 是负的。
    g = np.where((np.abs(g.imag) < 1e-12) & (np.real(g * np.conj(mu)) < 0), -g, g)
    grazing = np.flatnonzero(np.abs(g) < 1e-12)
    if grazing.size:
        raise RayleighAnomalyError(rayleigh_message(where, np.sqrt(eps * mu), np.asarray(orders)[grazing]))
    D = np.diag
    gp = np.concatenate((g, g))
    if basis == "sp":
        c, s = _order_angles(kx, ky, 0.0)
        E = np.block([[D(-s), D(g * c)], [D(c + 0j), D(g * s)]])                    # 列：N 个 s 模，N 个 p 模
        H = np.block([[D(-(g / mu) * c), D(-eps * s)], [D(-(g / mu) * s), D(eps * c)]])
        return _Modes(np.vstack((E, H)), gp, np.vstack((E, -H)), -gp, "isotropic")
    Y = np.block([[D(-kx * ky / (mu * g)), D(-(eps * mu - kx ** 2) / (mu * g))],
                  [D((eps * mu - ky ** 2) / (mu * g)), D(kx * ky / (mu * g))]])
    I = np.eye(2 * N, dtype=complex)
    return _Modes(np.vstack((I, Y)), gp, np.vstack((I, -Y)), -gp, "isotropic")


def _uniform_modes(medium, kx, ky, where, orders):
    """均匀各向异性介质：M 按级次分块，每级一个 4×4 本征问题（两个下行、两个上行）。"""
    N = len(kx)
    Fp, Fm = np.zeros((4 * N, 2 * N), complex), np.zeros((4 * N, 2 * N), complex)
    gp, gm = np.zeros(2 * N, complex), np.zeros(2 * N, complex)
    for m in range(N):
        Mm, Zm = maxwell_generator(medium.C, kx[m:m + 1], ky)
        label = f"{where}的第 {int(orders[m])} 级"
        gamma, vectors = np.linalg.eig(Mm)
        if not np.isfinite(vectors).all() or np.linalg.cond(vectors) > 1e12:
            raise RayleighAnomalyError(
                f"{label}的四个模式不独立：这一级恰在截止点，或两个模式合并成了一个；"
                f"请把入射角、波长或材料取向略作改动后重算。")
        a, A, b, B = _split_modes(gamma, vectors, Mm, Zm, label)
        rows, cols = np.ix_(m + N * np.arange(4), [m, N + m])
        Fp[rows, cols], Fm[rows, cols] = A, B
        gp[[m, N + m]], gm[[m, N + m]] = a, b
    return _Modes(Fp, gp, Fm, gm, "uniform", medium=medium)


def _grating_modes(layer, period, M, kx, ky, factorization, where):
    Chat = constitutive_operator(layer, period, M, factorization)
    Mgen, Z = maxwell_generator(Chat, kx, ky)
    gamma, F = np.linalg.eig(Mgen)
    MF = Mgen @ F
    residual = float(np.linalg.norm(MF - F * gamma[None, :]) / max(np.linalg.norm(MF), 1e-300))
    gp, Fp, gm, Fm = _split_modes(gamma, F, Mgen, Z, where)
    modes = _Modes(Fp, gp, Fm, gm, "grating", Chat=Chat, residual=residual)
    modes.normal = (Chat, Z)
    return modes


def _layer_modes(layer, period, M, kx, ky, factorization, where, orders, force_dense=False):
    if layer.homogeneous and not force_dense:
        medium = layer.media[0]
        scalar = medium.scalar
        if scalar is not None:
            modes = _isotropic_modes(scalar[0], scalar[1], kx, ky, where, orders, basis="sp")
            modes.medium = medium
            return modes
        return _uniform_modes(medium, kx, ky, where, orders)
    if layer.edges is None:
        layer = TensorLayer(layer.d, edges=[0.0, period], media=layer.media, name=layer.name)
    if abs((layer.edges[-1] - layer.edges[0]) - period) > 1e-9 * period:
        raise ValueError(f"{where}的 edges 须恰好覆盖一个周期")
    return _grating_modes(layer, period, M, kx, ky, factorization, where)


# =============================================================================
#  5. 求解
# =============================================================================
class ResultTensor(Result):
    pass


def _jones(psi_deg, delta_deg, value):
    """归一化 [Es, Ep]，与 conical.polarization.jones_vector 相同。"""
    if value is None:
        psi = np.radians(psi_deg)
        value = [np.sin(psi) * np.exp(1j * np.radians(delta_deg)), np.cos(psi)]
    v = np.asarray(value, complex).reshape(2)
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Jones 输入不能是零矢量")
    return v / norm


def _order_angles(kx, ky, phi):
    """各级衍射面的方位角，与 conical.spbasis.order_angles 相同（kt = 0 时取入射方位角）。"""
    kt = np.hypot(kx, ky)
    varphi = np.where(kt <= 1e-14, phi, np.arctan2(np.broadcast_to(ky, kx.shape), kx))
    return np.cos(varphi), np.sin(varphi)


def _safe_divide(values, denominator, tol=1e-13):
    out = np.zeros_like(values, dtype=complex)
    good = np.abs(denominator) > tol
    out[good] = values[good] / denominator[good, None]
    return out


def _clean(values):
    values = np.array(values, float)
    values[np.abs(values) < 1e-14] = 0.0
    return values


def solve_tensor(stack, lam0, theta_deg, phi_deg=0.0, M=20, psi_deg=90.0, delta_deg=0.0,
                 jones_in=None, factorization="li", need_fields=False, _force_dense=False):
    """求解一般介质一维光栅的圆锥入射。

    psi_deg、delta_deg、jones_in 的含义与 conical.solve_conical 相同：
    [Es, Ep] = [sinψ·exp(iδ), cosψ]，90° 为 s，0° 为 p；jones_in 给出时优先。
    返回各级的 s/p Jones 矩阵 Jr、Jt（ab = 出射 a ← 入射 b）、指定入射偏振下的效率
    DEr、DEt（按 z 向能流计算）以及反射、透射的切向电场谐波 R_xy、T_xy（[Ex; Ey]，两列对应 s、p 入射）。
    prop_r、prop_t 标出两侧有传播模式的级次。出射区各向异性时没有 Jt、Ts、Tp；出射区有吸收时 Jt 是按复的
    cosθ 换算的振幅，效率以 DEt 为准。T_modes 是出射区模式的内部振幅，归一方式随模式基而定。
    """
    if int(M) != M or M < 0:
        raise ValueError("M 必须是非负整数")
    M = int(M)
    if not (np.isfinite(lam0) and lam0 > 0 and np.isfinite(stack.period) and stack.period > 0):
        raise ValueError("波长与周期须为正的有限数")
    if not np.isfinite(theta_deg) or abs(theta_deg) >= 90:
        raise ValueError("入射角须满足 |θ| < 90°")
    nI = complex(stack.nI)
    if abs(nI.imag) > 1e-14 or nI.real <= 0:
        raise NotImplementedError("入射区须为各向同性无损介质（正的实折射率）")
    nI = float(nI.real)
    exit_medium = as_medium(stack.nII)
    if not exit_medium.passive:
        raise ValueError("出射区介质有增益（(C − C†)/(2i) 不是半正定），辐射条件没有定义")
    active = [str(j + 1) for j, layer in enumerate(stack.layers) if not all(m.passive for m in layer.media)]
    if active:
        warnings.warn("第 " + "、".join(active) + " 层含有带增益的材料：结果是频域方程的形式解，"
                      "只给 ξ 而没有给 ζ = ξ† 的材料也属于这一类。", RuntimeWarning, stacklevel=2)
    theta, phi = np.radians(theta_deg), np.radians(phi_deg)
    period = stack.period
    k0 = 2 * np.pi / lam0
    ms = np.arange(-M, M + 1)
    N, i0 = len(ms), M
    kx = nI * np.sin(theta) * np.cos(phi) - ms * lam0 / period
    ky = float(nI * np.sin(theta) * np.sin(phi))

    top = _isotropic_modes(nI ** 2, 1.0, kx, ky, "入射区", ms)
    top.medium = isotropic(nI)
    scalar = exit_medium.scalar
    if scalar is not None:
        bottom = _isotropic_modes(scalar[0], scalar[1], kx, ky, "出射区", ms)
        bottom.medium = exit_medium
    else:
        bottom = _uniform_modes(exit_medium, kx, ky, "出射区", ms)
    modes = [top]
    for j, layer in enumerate(stack.layers):
        where = f"第{j + 1}层" + (f"「{layer.name}」" if layer.name else "")
        modes.append(_layer_modes(layer, period, M, kx, ky, factorization, where, ms, _force_dense))
    modes.append(bottom)
    thick = [0.0] + [layer.d for layer in stack.layers] + [0.0]
    xp = [np.exp(1j * k0 * m.gp * d) for m, d in zip(modes, thick)]      # 下行：层顶 → 层底
    xm = [np.exp(-1j * k0 * m.gm * d) for m, d in zip(modes, thick)]     # 上行：层底 → 层顶

    # 自下而上递推。界面 j（第 j 区与第 j+1 区之间）处，记 u 为到达界面的下行振幅：
    #   Fp_j u + Fm_j b_j = (Fp_{j+1} + Fm_{j+1} Rtop_{j+1}) a_{j+1}
    # 一次 4N 阶求解同时给出 b_j = Rb_j u 与 a_{j+1} = Tj u。
    last = len(modes) - 1
    Rb, Tj = [None] * last, [None] * last
    r_top = np.zeros((2 * N, 2 * N), complex)
    for j in range(last - 1, -1, -1):
        up, low = modes[j], modes[j + 1]
        try:
            sol = np.linalg.solve(np.hstack((-up.Fm, low.Fp + low.Fm @ r_top)), up.Fp)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError(
                f"第 {j + 1} 个界面（自上而下）的匹配方程奇异：这个界面在当前波矢下支持无损的表面模或导模共振，"
                f"例如 ε、μ 恰好反号的无损介质贴着普通介质。给材料加一点吸收，或把参数略作改动后重算。") from None
        Rb[j], Tj[j] = sol[:2 * N], sol[2 * N:]
        r_top = xm[j][:, None] * Rb[j] * xp[j][None, :]

    # 入射：两列分别是单位电场的 s、p（约定同 solve_conical）
    cos_m, sin_m = _order_angles(kx, ky, phi)
    cthI = top.gp[:N] / nI
    inc = np.zeros((2 * N, 2), complex)
    inc[i0, 0], inc[N + i0, 0] = -sin_m[i0], cos_m[i0]
    inc[i0, 1], inc[N + i0, 1] = cos_m[i0] * cthI[i0], sin_m[i0] * cthI[i0]
    downs, ups = [inc], []
    for j in range(last):
        arriving = xp[j][:, None] * downs[j]
        ups.append(Rb[j] @ arriving)
        downs.append(Tj[j] @ arriving)
    ups.append(np.zeros((2 * N, 2), complex))

    psi_inc, psi_ref, psi_tra = top.Fp @ inc, top.Fm @ ups[0], bottom.Fp @ downs[last]
    vin = _jones(psi_deg, delta_deg, jones_in)
    incident_flux = float(_order_flux(psi_inc @ vin).sum())
    DEr = _clean(-_order_flux(psi_ref @ vin) / incident_flux)
    DEt = _clean(_order_flux(psi_tra @ vin) / incident_flux)

    def to_sp(psi, cth, sign):
        ex, ey = psi[:N], psi[N:2 * N]
        s = -sin_m[:, None] * ex + cos_m[:, None] * ey
        p = sign * _safe_divide(cos_m[:, None] * ex + sin_m[:, None] * ey, cth)
        return np.stack((s, p), axis=1)                 # (N, 2 出射, 2 入射)

    Jr = to_sp(psi_ref, cthI, -1.0)
    out = ResultTensor(
        ms=ms, M=M, k0=k0, kxm=k0 * kx, kx_over_k0=kx, ky=k0 * ky, kappa=ky, theta=theta, phi=phi,
        stack=stack, lam0=lam0, factorization=factorization, i0=i0, jones_in=vin,
        kzI=k0 * top.gp[:N], kzI_over_k0=top.gp[:N], cos_phi_m=cos_m, sin_phi_m=sin_m,
        Jr=Jr, Rs=Jr[:, 0] @ vin, Rp=Jr[:, 1] @ vin, DEr=DEr, DEt=DEt,
        R_xy=psi_ref[:2 * N], T_xy=psi_tra[:2 * N], T_modes=downs[last], psi_ref=psi_ref, psi_tra=psi_tra,
        energy=float(DEr.sum() + DEt.sum()),
        mode_residual=max([m.residual for m in modes] or [0.0]),
        n_dense=sum(m.kind == "grating" for m in modes),
    )
    flux_r = np.maximum(_clean(np.real(top.gp[:N] / (nI * np.cos(theta)))), 0.0)
    out["prop_r"] = flux_r > 0
    # 出射区里有无衰减（传播）模式的级次。出射区有吸收时没有这样的级次，prop_t 全为假；
    # 那时 DEt 仍是各级次穿过最后一个界面的功率。
    undamped = np.abs(bottom.gp.imag) <= 1e-9 * (1 + np.abs(bottom.gp))
    out["prop_t"] = undamped.reshape(2, N).any(axis=0)
    for a, name_a in enumerate("sp"):
        for b, name_b in enumerate("sp"):
            out[f"DEr_{name_a}{name_b}"] = flux_r * np.abs(Jr[:, a, b]) ** 2
    if scalar is not None:
        # 出射区各向同性：模式振幅就是 [Ex; Ey]，可以写成 s/p Jones 矩阵
        eps2, mu2 = scalar
        n2 = np.sqrt(eps2 * mu2)
        Jt = to_sp(psi_tra, bottom.gp[:N] / n2, +1.0)
        flux_t = np.maximum(_clean(np.real(bottom.gp[:N] / mu2) / (nI * np.cos(theta))), 0.0)
        out.update(Jt=Jt, Ts=Jt[:, 0] @ vin, Tp=Jt[:, 1] @ vin,
                   kzII=k0 * bottom.gp[:N], kzII_over_k0=bottom.gp[:N])
        if abs(np.imag(n2)) < 1e-14:
            for a, name_a in enumerate("sp"):
                for b, name_b in enumerate("sp"):
                    out[f"DEt_{name_a}{name_b}"] = flux_t * np.abs(Jt[:, a, b]) ** 2

    if need_fields:
        amps, ztop = [], 0.0
        for j, layer in enumerate(stack.layers, start=1):
            amps.append(dict(modes=modes[j], a=downs[j], b=ups[j], d=layer.d, z_top=ztop, layer=layer))
            ztop += layer.d
        out.update(amps=amps, z_bot=ztop, top=top, bottom=bottom, incident=inc, modes=modes)
    return out


# =============================================================================
#  6. 场重建
# =============================================================================
def _normal_matrix(modes, kx, ky):
    """(Ĉ, Z)，[Ez; Hz] = Z ψ。均匀介质的本构算符是 C ⊗ I。每层只算一次。"""
    if modes.normal is None:
        Chat = modes.Chat if modes.Chat is not None else np.kron(modes.medium.C, np.eye(len(kx)))
        modes.normal = (Chat, maxwell_generator(Chat, kx, ky)[1])
    return modes.normal


def field_at(res, z, jones_in=None):
    """z 处各 Fourier 谐波的六个场分量（E 以入射电场归一，H 为 Z0·H），外加 D/ε0 与 c·B 的六个分量。

    调用求解器时须 need_fields=True。
    """
    if "amps" not in res:
        raise ValueError("请用 solve_tensor(..., need_fields=True) 重新求解")
    vin = res["jones_in"] if jones_in is None else _jones(0.0, 0.0, jones_in)
    k0, N = res["k0"], len(res["ms"])
    if z < 0 or (z == 0 and not res["amps"]):
        modes, a, b = res["top"], res["incident"] @ vin, res["R_xy"] @ vin
        down = np.zeros_like(a)
        active = a != 0                      # 未激发的倏逝入射模在 z → −∞ 方向增长，不能乘指数
        down[active] = a[active] * np.exp(1j * k0 * modes.gp[active] * z)
        up = b * np.exp(1j * k0 * modes.gm * z)
    elif z > res["z_bot"]:
        modes = res["bottom"]
        down = (res["T_modes"] @ vin) * np.exp(1j * k0 * modes.gp * (z - res["z_bot"]))
        up = np.zeros_like(down)
    else:
        amp = next((a for a in res["amps"] if z <= a["z_top"] + a["d"] + 1e-12), None)
        if amp is None:
            raise ValueError("z 超出结构范围")
        modes, zl = amp["modes"], z - amp["z_top"]
        down = (amp["a"] @ vin) * np.exp(1j * k0 * modes.gp * zl)
        up = (amp["b"] @ vin) * np.exp(1j * k0 * modes.gm * (zl - amp["d"]))
    psi = modes.Fp @ down + modes.Fm @ up
    Chat, Z = _normal_matrix(modes, res["kx_over_k0"], res["kappa"])
    normal = Z @ psi
    six = np.concatenate((psi[:2 * N], normal[:N], psi[2 * N:], normal[N:]))
    flux = Chat @ six
    out = {name: six[i * N:(i + 1) * N] for i, name in enumerate(("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"))}
    out.update({name: flux[i * N:(i + 1) * N] for i, name in enumerate(("Dx", "Dy", "Dz", "Bx", "By", "Bz"))})
    out["psi"] = psi
    return out


def field_map(res, zs, nx=256, y=0.0, jones_in=None):
    """在 (x, z) 网格上重建六个场分量；结构沿 y 不变，y 只贡献公共相位。"""
    zs = np.atleast_1d(zs).astype(float)
    x = np.arange(int(nx)) / int(nx) * res["stack"].period
    phase = np.exp(-1j * (np.outer(x, res["kxm"]) + res["ky"] * y))
    names = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    out = {name: np.zeros((len(zs), len(x)), complex) for name in names}
    for i, z in enumerate(zs):
        harmonics = field_at(res, float(z), jones_in)
        for name in names:
            out[name][i] = phase @ harmonics[name]
    out["E2"] = sum(np.abs(out[name]) ** 2 for name in names[:3])
    out["x"], out["z"] = x, zs
    return out
