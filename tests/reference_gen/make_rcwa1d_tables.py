# -*- coding: utf-8 -*-
"""用 gui/rcwa1d.py 算 tests/test_rcwa1d.cpp 的参考效率表

gui/rcwa1d.py 是另外用 Python 写的一维 RCWA（S 矩阵级联，无吸收层用 eigh），与 C++ 的 rcwa1d.hpp 不共用代码。
结构与入射条件见同目录的 rcwa1d_cases.py。

    python tests/reference_gen/make_rcwa1d_tables.py                       # 重算全部表，与 test_rcwa1d.cpp 里的逐级比较
    python tests/reference_gen/make_rcwa1d_tables.py --emit kRidgesSilver50TE,kRidgesSilver50TM   # 写出 C++ 表

第一种用法打印每张表的最大偏差；任何一张超过 --tol（默认 1e-11）就以状态 1 退出。
第二种把表写到标准输出，格式与 test_rcwa1d.cpp 相同（每个数用 Python repr 的最短写法，读回来与算出的 double
逐位相同），可以直接贴回去。
numpy 用多线程 BLAS 时请限制线程数，例如 OMP_NUM_THREADS=2。
"""
import argparse
import os
import sys

import numpy as np

sys.dont_write_bytecode = True     # 不在 tests/ 和 gui/ 里留下 __pycache__

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "gui"))

import rcwa1d                      # noqa: E402  gui/rcwa1d.py
import rcwa1d_cases as cases       # noqa: E402


def to_gui(stack):
    layers = []
    for L in stack.layers:
        if L.n is not None:
            layers.append(rcwa1d.Layer(L.d, n=L.n))
        else:
            layers.append(rcwa1d.Layer(L.d, edges=L.edges, n_values=L.values))
    return rcwa1d.Stack(stack.nI, layers, stack.nII, stack.period)


def efficiencies(case):
    """第 −M … +M 级的反射、透射效率"""
    r = rcwa1d.solve(to_gui(case.stack()), case.wavelength, case.theta, case.M, case.pol, need_fields=False)
    return [float(x) for x in np.asarray(r.DEr)], [float(x) for x in np.asarray(r.DEt)]


def cpp_table(case, der, det):
    if case.kind == "orders":
        lines = [f"// {case.title}", f"static const OrderRef {case.name}[] = {{"]
        for j, m in enumerate(range(-case.M, case.M + 1)):
            if der[j] != 0.0 or det[j] != 0.0:
                lines.append(f"    {{{m}, {der[j]!r}, {det[j]!r}}},")
        lines.append("};")
        return "\n".join(lines)
    out = []
    for side, values in (("Der", der), ("Det", det)):
        rows = [", ".join(repr(v) for v in values[k:k + 4]) for k in range(0, len(values), 4)]
        out.append(f"static const double {case.name}{side}[] = {{  // 第 −M … +M 级\n    " + ",\n    ".join(rows) + ",\n};")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--emit", help="逗号分隔的表名：把这些表写成 C++ 代码")
    ap.add_argument("--cpp", default=os.path.join(ROOT, "tests", "test_rcwa1d.cpp"), help="要比较的 C++ 文件")
    ap.add_argument("--tol", type=float, default=1e-11, help="比较模式下允许的最大偏差")
    args = ap.parse_args()
    by_name = {c.name: c for c in cases.CASES}

    if args.emit:
        for name in args.emit.split(","):
            case = by_name[name.strip()]
            print(cpp_table(case, *efficiencies(case)))
            print()
        return 0

    tables = cases.read_cpp_tables(args.cpp)
    worst = 0.0
    for case in cases.CASES:
        e = cases.max_deviation(*efficiencies(case), tables[case.name])
        worst = max(worst, e)
        print(f"{case.name:20s} {case.pol}  M = {case.M:2d}  最大偏差 {e:.1e}")
    print(f"共 {len(cases.CASES)} 张表，gui/rcwa1d.py 与 {os.path.relpath(args.cpp, ROOT)} 的最大偏差 {worst:.1e}")
    return 0 if worst <= args.tol else 1


if __name__ == "__main__":
    sys.exit(main())
