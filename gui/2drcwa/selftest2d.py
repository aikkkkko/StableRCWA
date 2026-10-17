# -*- coding: utf-8 -*-
"""rcwa2d.py 自检套件。任何一条不过就不要用它出图。"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import rcwa1d as rc1
import rcwa2d as rc2

ok_all = True


def check(name, err, tol):
    global ok_all
    good = bool(err < tol)
    ok_all &= good
    print(f"  [{'通过' if good else '不通过'}] {name}:  误差 = {err:.3e}  (阈值 {tol:.0e})")


LAM0, THETA = 1.053, 66.36
P0, F, DG = 0.574, 0.30, 0.38
NR, NG, NI, NII = 1.46, 1.0, 1.0, 1.46
XE = [-F * P0 / 2, F * P0 / 2, P0 - F * P0 / 2]

# =================================================================== T1
print("\nT1  沿 y 不变的二维结构 → 必须逐位复现一维 TE / TM")
M1, LY = 12, 0.8
lat = rc2.Lattice(P0, LY)
lay = rc2.Layer2D(DG, y_edges=[0.0, LY], x_profiles=[(XE, [NR, NG])])
st2 = rc2.Stack2D(NI, [lay], NII, lat)
for psi, pol in ((90.0, "TE"), (0.0, "TM")):
    r1 = rc1.solve(rc1.Stack(NI, [rc1.Layer(DG, edges=XE, n_values=[NR, NG])], NII, P0),
                   LAM0, THETA, M1, pol, need_fields=False)
    r2 = rc2.solve2d(st2, LAM0, THETA, 0.0, M1, 0, psi_deg=psi)
    check(f"{pol} 反射级效率", np.abs(r2["DEr"] - r1["DEr"]).max(), 1e-11)
    check(f"{pol} 透射级效率", np.abs(r2["DEt"] - r1["DEt"]).max(), 1e-11)

# =================================================================== T1'
print("\nT1' 沿 x 不变的二维结构 → 同上（坐标轴互换，抓 x<->y 块转置错误）")
lat_s = rc2.Lattice(LY, P0)
ye = [-F * P0 / 2, F * P0 / 2, P0 - F * P0 / 2]
lay_s = rc2.Layer2D(DG, y_edges=ye,
                    x_profiles=[([0.0, LY], [NR]), ([0.0, LY], [NG])])
st2s = rc2.Stack2D(NI, [lay_s], NII, lat_s)
for psi, pol in ((90.0, "TE"), (0.0, "TM")):
    # 结构沿 y 周期、槽沿 x：phi=90 时 s 矢量仍平行于槽，故 psi=90 仍是 TE
    r1 = rc1.solve(rc1.Stack(NI, [rc1.Layer(DG, edges=ye, n_values=[NR, NG])], NII, P0),
                   LAM0, THETA, M1, pol, need_fields=False)
    r2 = rc2.solve2d(st2s, LAM0, THETA, 90.0, 0, M1, psi_deg=psi)
    check(f"{pol} 反射级效率", np.abs(r2["DEr"] - r1["DEr"]).max(), 1e-11)
    check(f"{pol} 透射级效率", np.abs(r2["DEt"] - r1["DEt"]).max(), 1e-11)

# =================================================================== T2
print("\nT2  全均匀多层（k_y != 0 的圆锥）→ 与一维解在 phi=0 一致，且非零级严格为 0")
films = [rc2.Layer2D(0.31, n=1.46), rc2.Layer2D(0.115, n=1.96), rc2.Layer2D(0.31, n=1.46)]
st_h = rc2.Stack2D(NI, films, NII, rc2.Lattice(P0, LY))
f1 = [rc1.Layer(0.31, n=1.46), rc1.Layer(0.115, n=1.96), rc1.Layer(0.31, n=1.46)]
for psi, pol in ((90.0, "TE"), (0.0, "TM")):
    r1 = rc1.solve(rc1.Stack(NI, f1, NII, P0), LAM0, THETA, 6, pol, need_fields=False)
    r2 = rc2.solve2d(st_h, LAM0, THETA, 0.0, 6, 3, psi_deg=psi)
    i0 = rc2.index_of(r2, 0, 0)
    j0 = int(np.where(r1["ms"] == 0)[0][0])
    check(f"{pol} 零级反射", abs(r2["DEr"][i0] - r1["DEr"][j0]), 1e-13)
    check(f"{pol} 零级透射", abs(r2["DEt"][i0] - r1["DEt"][j0]), 1e-13)
    m = np.ones(len(r2["mn"]), bool); m[i0] = False
    check(f"{pol} 非零级应为 0", max(r2["DEr"][m].max(), r2["DEt"][m].max()), 1e-14)

# =================================================================== T3
print("\nT3  真二维结构：能量守恒（无吸收）")
YE = [0.0, 3.0, 6.0]
lay2 = rc2.Layer2D(DG, y_edges=YE,
                   x_profiles=[(XE, [NR, NG]), ([0.0, 6.0], [NG])])
lat2 = rc2.Lattice(P0, 6.0)
st_2d = rc2.Stack2D(NI, [lay2, rc2.Layer2D(0.46, n=1.46), rc2.Layer2D(0.115, n=1.96)],
                    NII, lat2)
for phi in (0.0, 12.0, 40.0):
    for psi in (90.0, 45.0, 0.0):
        r = rc2.solve2d(st_2d, LAM0, THETA, phi, 8, 8, psi_deg=psi)
        check(f"phi={phi:>4}° psi={psi:>4}°  sum(DE)=1",
              abs(r["DEr"].sum() + r["DEt"].sum() - 1), 1e-9)

# =================================================================== T4
print("\nT4  二维超胞等价性：2x2 倍胞必须复现单胞，且非整除级严格为 0")
lat_s1 = rc2.Lattice(P0, 3.0)
lay_s1 = rc2.Layer2D(DG, y_edges=[0.0, 1.5, 3.0],
                     x_profiles=[(XE, [NR, NG]), ([0.0, P0], [NG])])
st_s1 = rc2.Stack2D(NI, [lay_s1], NII, lat_s1)
lat_s2 = rc2.Lattice(2 * P0, 6.0)
xe2 = [-F * P0 / 2, F * P0 / 2, P0 - F * P0 / 2, P0 + F * P0 / 2, 2 * P0 - F * P0 / 2]
lay_s2 = rc2.Layer2D(DG, y_edges=[0.0, 1.5, 3.0, 4.5, 6.0],
                     x_profiles=[(xe2, [NR, NG, NR, NG]), ([0.0, 2 * P0], [NG]),
                                 (xe2, [NR, NG, NR, NG]), ([0.0, 2 * P0], [NG])])
st_s2 = rc2.Stack2D(NI, [lay_s2], NII, lat_s2)
Mc = 5
r_1 = rc2.solve2d(st_s1, LAM0, THETA, 15.0, Mc, Mc, psi_deg=90.0)
r_2 = rc2.solve2d(st_s2, LAM0, THETA, 15.0, 2 * Mc, 2 * Mc, psi_deg=90.0)
d, bad = 0.0, 0.0
for k, (m, n) in enumerate(r_1["mn"]):
    d = max(d, abs(r_2["DEr"][rc2.index_of(r_2, 2 * m, 2 * n)] - r_1["DEr"][k]))
sel = (r_2["mn"][:, 0] % 2 == 0) & (r_2["mn"][:, 1] % 2 == 0)
bad = max(r_2["DEr"][~sel].max(), r_2["DEt"][~sel].max())
check("超胞 vs 单胞（对应级）", d, 1e-9)
check("非整除级应为 0", bad, 1e-13)

# =================================================================== T5
print("\nT5  层内场：各界面切向分量连续（Ex,Ey,Hx,Hy 及 Hz）")
res_f = rc2.solve2d(st_2d, LAM0, THETA, 18.0, 7, 7, psi_deg=55.0, delta_deg=35.0)
zc, worst = 0.0, {k: 0.0 for k in ("Ex", "Ey", "Hx", "Hy", "Hz")}
zs_if = [0.0] + [(zc := zc + a["d"]) for a in res_f["amps"]]
for zi in zs_if:
    up, dn = rc2.field_at(res_f, zi - 1e-10), rc2.field_at(res_f, zi + 1e-10)
    for k in worst:
        sc = max(np.abs(up[k]).max(), 1e-30)
        worst[k] = max(worst[k], np.abs(up[k] - dn[k]).max() / sc)
for k, v in worst.items():
    check(f"{k} 跨 {len(zs_if)} 个界面连续", v, 1e-7)

# =================================================================== T6
print("\nT6  层内场：沿 y 不变时必须复现一维 rcwa1d 的场")
r1f = rc1.solve(rc1.Stack(NI, [rc1.Layer(DG, edges=XE, n_values=[NR, NG])], NII, P0),
                LAM0, THETA, M1, "TE")
r2f = rc2.solve2d(st2, LAM0, THETA, 0.0, M1, 0, psi_deg=90.0)
n0 = r2f["mn"][:, 1] == 0
e_ey, e_hz = 0.0, 0.0
for z in (-0.25, 0.0, 0.05, 0.19, DG - 1e-9, DG + 0.3):
    S1, U1 = rc1.field_at(r1f, z)
    f2 = rc2.field_at(r2f, z)
    e_ey = max(e_ey, np.abs(f2["Ey"][n0] - S1).max())
    e_hz = max(e_hz, np.abs(f2["Hz"][n0] - (-r1f["Kx"] @ S1)).max())
check("Ey 谐波（含区域 I/II 与层内 6 个深度）", e_ey, 1e-11)
check("Hz 谐波（= -Kx·Ey）", e_hz, 1e-11)

# =================================================================== T7
print("\nT7  实空间重建：逆 FFT 与直接求和一致；均匀层 |E|^2 与解析值一致")
fx = rc2.field_xy(res_f, 0.15, nx=32, ny=64)
h = rc2.field_at(res_f, 0.15)
X, Y = np.meshgrid(fx["x"], fx["y"], indexing="ij")
direct = rc2._to_line(h["Ey"], res_f, X.ravel(), Y.ravel()).reshape(X.shape)
check("ifft2 vs 直接求和", np.abs(fx["Ey"] - direct).max(), 1e-11)

st_hh = rc2.Stack2D(NI, films, NII, rc2.Lattice(P0, LY))
rh = rc2.solve2d(st_hh, LAM0, THETA, 0.0, 4, 2, psi_deg=90.0)
fh = rc2.field_xy(rh, 0.2, nx=16, ny=16)
check("均匀多层内 |E|^2 沿 x,y 为常数", np.ptp(fh["E2"]), 1e-12)

print("\n" + ("全部通过" if ok_all else "有测试未通过"))
sys.exit(0 if ok_all else 1)
