# -*- coding: utf-8 -*-
"""用另写的整体矩阵法重算 tests/test_rcwa1d.cpp 的全部参考效率表，逐级比较

这是第二条独立的路线：只用 numpy，不调用 gui/ 的任何代码，算法也不同。
  · gui/rcwa1d.py 与 C++ 都是逐层级联 S 矩阵；这里把所有界面的边界条件一次装成一个大的线性方程组
    （未知量是反射、透射振幅和每一层上行、下行的模式系数），一次解出；
  · 光栅层一律用一般复矩阵的 numpy.linalg.eig（gui/rcwa1d.py 对无吸收层用 eigh）；
  · Fourier 系数按 v·(b − a)/Λ·exp(−iπh(a + b)/Λ)·sinc(h(b − a)/Λ) 计算（与另两处的写法不同，舍入也不同）。
结构与入射条件取自同目录的 rcwa1d_cases.py。

    python tests/reference_gen/crosscheck_rcwa1d_tables.py         # 打印每张表的最大偏差；超过 --tol（默认 1e-11）以状态 1 退出

公式（时间因子 exp(−iωt)，z 向下，z' = k0·z，Kx = diag(kx_m / k0)，kx_m = k0·(nI·sin θ − m·λ/Λ)）：
  场按 Σ_m S_m(z)·exp(−i·kx_m·x) 展开；⟦f⟧ 是卷积矩阵，(m, p) 元是 (1/Λ)∫ f(x)·exp(−i2π(m − p)x/Λ) dx。
  主场 S（TE：Ey；TM：Hy）与切向量 U = F·dS/dz'（TE：F = I；TM：F = ⟦1/ε⟧，即 Ex 的 Fourier 分量，Li 的逆规则）
  在每个界面连续。层内 d²S/dz'² = A·S，TE：A = Kx² − ⟦ε⟧；TM：A = ⟦1/ε⟧⁻¹·(Kx·⟦ε⟧⁻¹·Kx − I)。
  A·W = W·diag(q²)，取 Re q ≥ 0（传播模取 Im q ≤ 0），层内 S = W·[exp(−q(z' − z'顶))·a + exp(−q(z'底 − z'))·b]，
  两个指数的模都不超过 1，方程组的条件数与层厚无关。均匀层（含入射、出射介质）W = I，q = sqrt(kx² − n²)。
"""
import argparse
import os
import sys

import numpy as np

sys.dont_write_bytecode = True     # 不在 tests/ 里留下 __pycache__

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import rcwa1d_cases as cases       # noqa: E402


def toeplitz_of(edges, values, period, M):
    """分段常数函数的卷积矩阵，(m, p) 元是第 m − p 个 Fourier 系数"""
    h = np.arange(-2 * M, 2 * M + 1)
    a, b = np.asarray(edges[:-1]), np.asarray(edges[1:])
    v = np.asarray(values, complex)
    w = (b - a) / period
    centre = (a + b) / period
    coeff = (v[None, :] * w[None, :] * np.exp(-1j * np.pi * np.outer(h, centre)) * np.sinc(np.outer(h, w))).sum(axis=1)
    m = np.arange(-M, M + 1)
    return coeff[np.subtract.outer(m, m) + 2 * M]


def root(z):
    """q = sqrt(z)：Re q ≥ 0；Re q 为零（传播模）时取 Im q ≤ 0，使 exp(−q·z') 沿 +z 传播"""
    q = np.sqrt(np.asarray(z, complex))
    q = np.where(q.real < 0, -q, q)
    tiny = np.abs(q.real) <= 1e-12 * np.maximum(np.abs(q), 1e-300)
    return np.where(tiny & (q.imag > 0), -q, q)


class Region:
    """一段介质的模式：W（S 的模式矩阵）、Fq = F·W·diag(q)（U 的模式矩阵，未带正负号）、q"""

    def __init__(self, W, Fq, q):
        self.W, self.Fq, self.q = W, Fq, q


def homogeneous(n, kx, pol):
    eps = complex(n) ** 2
    q = root(kx ** 2 - eps)
    N = len(kx)
    return Region(np.eye(N, dtype=complex), np.diag(q if pol == "TE" else q / eps), q)


def patterned(layer, period, kx, M, pol):
    eps = [complex(v) ** 2 for v in layer.values]
    E = toeplitz_of(layer.edges, eps, period, M)
    N = len(kx)
    if pol == "TE":
        A = np.diag(kx ** 2) - E
        F = np.eye(N, dtype=complex)
    else:
        F = toeplitz_of(layer.edges, [1.0 / e for e in eps], period, M)          # ⟦1/ε⟧
        A = np.linalg.solve(F, kx[:, None] * np.linalg.solve(E, np.diag(kx)) - np.eye(N))
    w2, W = np.linalg.eig(A)
    q = root(w2)
    return Region(W, F @ W * q[None, :], q)


def efficiencies(stack, wavelength, theta_deg, M, pol):
    k0 = 2 * np.pi / wavelength
    th = np.radians(theta_deg)
    m = np.arange(-M, M + 1)
    N = len(m)
    nI, nII = complex(stack.nI), complex(stack.nII)
    kx = (nI * np.sin(th) - m * wavelength / stack.period).real.astype(complex)
    top, bottom = homogeneous(nI, kx, pol), homogeneous(nII, kx, pol)
    layers = [homogeneous(L.n, kx, pol) if L.n is not None else patterned(L, stack.period, kx, M, pol)
              for L in stack.layers]
    X = [np.exp(-r.q * k0 * L.d) for r, L in zip(layers, stack.layers)]

    # 未知量的次序：r（N 个），各层 (a, b)（各 N 个），t（N 个）
    nL = len(layers)
    size = 2 * N * (nL + 1)
    G = np.zeros((size, size), complex)
    rhs = np.zeros(size, complex)
    col_r, col_t = 0, N + 2 * N * nL
    col_a = [N + 2 * N * j for j in range(nL)]
    col_b = [N + 2 * N * j + N for j in range(nL)]
    inc = (m == 0).astype(complex)

    for k in range(nL + 1):                 # 第 k 个界面：上面是入射介质（k = 0）或第 k − 1 层，下面是第 k 层或出射介质
        rs, ru = 2 * N * k, 2 * N * k + N   # S、U 两组方程所在的行
        # 上侧，取它的底面：入射介质里 S = δ + r，U = Fq·(−δ + r)；第 j 层里 S = W(X a + b)，U = Fq(−X a + b)
        if k == 0:
            G[rs:rs + N, col_r:col_r + N] += top.W
            G[ru:ru + N, col_r:col_r + N] += top.Fq
            rhs[rs:rs + N] -= top.W @ inc
            rhs[ru:ru + N] += top.Fq @ inc
        else:
            j, R = k - 1, layers[k - 1]
            G[rs:rs + N, col_a[j]:col_a[j] + N] += R.W * X[j][None, :]
            G[rs:rs + N, col_b[j]:col_b[j] + N] += R.W
            G[ru:ru + N, col_a[j]:col_a[j] + N] -= R.Fq * X[j][None, :]
            G[ru:ru + N, col_b[j]:col_b[j] + N] += R.Fq
        # 下侧，取它的顶面（移到等号左边，带负号）：第 j 层里 S = W(a + X b)，U = Fq(−a + X b)；出射介质里 S = t，U = −Fq·t
        if k == nL:
            G[rs:rs + N, col_t:col_t + N] -= bottom.W
            G[ru:ru + N, col_t:col_t + N] += bottom.Fq
        else:
            j, R = k, layers[k]
            G[rs:rs + N, col_a[j]:col_a[j] + N] -= R.W
            G[rs:rs + N, col_b[j]:col_b[j] + N] -= R.W * X[j][None, :]
            G[ru:ru + N, col_a[j]:col_a[j] + N] += R.Fq
            G[ru:ru + N, col_b[j]:col_b[j] + N] -= R.Fq * X[j][None, :]

    x = np.linalg.solve(G, rhs)
    r, t = x[col_r:col_r + N], x[col_t:col_t + N]
    kzI, kzII = 1j * top.q, 1j * bottom.q            # kz / k0（q = −i·kz/k0）
    cos_in = nI * np.cos(th)
    der = np.abs(r) ** 2 * (kzI / cos_in).real
    if pol == "TE":
        det = np.abs(t) ** 2 * (kzII / cos_in).real
    else:
        det = np.abs(t) ** 2 * ((kzII / nII ** 2) / (np.cos(th) / nI)).real
    return [float(v) for v in der], [float(v) for v in det]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cpp", default=os.path.join(ROOT, "tests", "test_rcwa1d.cpp"), help="要核对的 C++ 文件")
    ap.add_argument("--tol", type=float, default=1e-11, help="允许的最大偏差")
    args = ap.parse_args()
    tables = cases.read_cpp_tables(args.cpp)
    worst = 0.0
    for case in cases.CASES:
        e = cases.max_deviation(*efficiencies(case.stack(), case.wavelength, case.theta, case.M, case.pol),
                                tables[case.name])
        worst = max(worst, e)
        print(f"{case.name:20s} {case.pol}  M = {case.M:2d}  最大偏差 {e:.1e}")
    print(f"共 {len(cases.CASES)} 张表，整体矩阵法与 {os.path.relpath(args.cpp, ROOT)} 的最大偏差 {worst:.1e}")
    return 0 if worst <= args.tol else 1


if __name__ == "__main__":
    sys.exit(main())
