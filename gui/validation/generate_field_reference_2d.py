"""Generate a candidate 2-D RCWA reference corpus explicitly; ordinary tests only read the frozen one.

    python3 gui/validation/generate_field_reference_2d.py --output /some/candidate/dir

writes reference_2d_v1.json, reference_2d_v1.npz and rcwa2d_reference.inc into the candidate directory.
Writing into the frozen directory (gui/validation/field_reference_2d, which also refreshes
tests/rcwa2d_reference.inc) needs the explicit --refresh-baseline flag; nothing else ever replaces frozen data.
Use OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 for a reproducible run.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import zipfile

import numpy as np
import scipy

from field_reference_2d_cases import (COMPONENTS, CONTRACT_VERSION, FAR_ARRAYS, FROZEN_DIR, NUMERIC_SOURCES,
                                      OPERATOR_LIMIT_N, ROOT, case_specs, failure_info, sample_case)

NPZ_NAME = "reference_2d_v1.npz"
JSON_NAME = "reference_2d_v1.json"
INC_NAME = "rcwa2d_reference.inc"
INC_FROZEN = ROOT / "tests" / INC_NAME
INC_MAX_BYTES = 1_900_000          # contract: <= 2 MB
INC_MAX_N = 169                    # only small cases go to the C++ table


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # pragma: no cover - git is optional for provenance
        return ""


def write_npz(path, arrays):
    """Deterministic .npz (sorted keys, fixed zip timestamps): equal arrays give equal bytes."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for key in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asanyarray(arrays[key]), allow_pickle=False)
            info = zipfile.ZipInfo(key + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compresslevel=9)


# ---------------------------------------------------------------- sanity gates (not oracles)
def check_case(spec, values):
    """Fail generation on a nonfinite reference or a violated structural-zero claim."""
    case = spec["id"]
    for key, value in values.items():
        if not key.startswith("phase_") and value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise RuntimeError(f"{case}/{key}: nonfinite reference value")
    near, far = spec["tolerance"]["near"], spec["tolerance"]["far"]
    for component in spec["structural_zero_components"]:
        ic = COMPONENTS.index(component)
        for key in ("harmonics", "fields_xz", "fields_yz"):
            peak = float(np.max(np.abs(values[key][ic])))
            if peak > near["atol"]:
                raise RuntimeError(f"{case}: component {component} is not structurally zero ({key}: {peak:.3e})")
    for key in spec["far_structural_zero_arrays"]:
        peak = float(np.max(np.abs(values[key])))
        if peak > far["atol"]:
            raise RuntimeError(f"{case}: far-field array {key} is not structurally zero ({peak:.3e})")
    if spec["uniform_stack"]:
        others = np.arange(spec["N"]) != int(values["i0"])
        for key in FAR_ARRAYS:
            if key in ("Kx", "Ky", "kzI", "kzII"):
                continue
            if float(np.max(np.abs(values[key][others]))) > 1e-14:
                raise RuntimeError(f"{case}: uniform stack but order != (0,0) is excited in {key}")
    # every harmonic z must lie off the Python 0*inf hazard: the check above (finite) already enforces it.


def check_failure(spec, info):
    expect = spec["expect"]
    if info is None or info["error"] != "RayleighAnomalyError":
        raise RuntimeError(f"{spec['id']}: expected a RayleighAnomalyError")
    message = info["message"]
    if not message.startswith(expect["region"]):
        raise RuntimeError(f"{spec['id']}: region {expect['region']!r} not found in {message!r}")
    found = [[int(a), int(b)] for a, b in re.findall(r"\((-?\d+), (-?\d+)\)", message)]
    if found != expect["orders"]:
        raise RuntimeError(f"{spec['id']}: orders {found} differ from expected {expect['orders']}")


# ---------------------------------------------------------------- C++ table
def number(value):
    value = float(value)
    if not np.isfinite(value):
        raise RuntimeError("nonfinite value in the C++ table")
    return repr(0.0 if value == 0.0 else value)       # shortest round-trip text; -0 is written as 0


def cpp_complex(value):
    return f"cplx({number(value.real)}, {number(value.imag)})"


def cpp_index(value):
    return cpp_complex(complex(*value) if isinstance(value, list) else complex(value))


def depth_selection(spec, count):
    """Indices into spec['zs'] for the C++ harmonic columns, most informative first."""
    zs = spec["zs"]
    ds = [layer["d"] for layer in spec["stack"]["layers"]]
    tops = np.concatenate([[0.0], np.cumsum(ds)]) if ds else np.zeros(1)
    z_bot = float(tops[-1])
    wanted = [-0.07]
    if ds:
        wanted += [tops[0] + 0.7 * ds[0], z_bot + 0.09, tops[0] + 1e-6, tops[-2] + 0.7 * ds[-1], z_bot - 1e-6]
    else:
        wanted += [0.09, -1e-6, 1e-6]
    wanted += [-0.43, z_bot + 0.41]
    chosen = []
    for z in wanted:
        hits = [i for i, value in enumerate(zs) if abs(value - z) < 2e-12]
        if hits and hits[0] not in chosen:
            chosen.append(hits[0])
    return sorted(chosen[:count])


def depth_count(N):
    return 3 if N <= 25 else 2


def render_inc(specs, arrays, failures, npz_sha, generator_name):
    """tests/rcwa2d_reference.inc: inputs, far fields, some harmonic columns (and operators of tiny cases)."""
    out = []
    w = out.append
    w("// tests/rcwa2d_reference.inc —— 二维（交叉光栅）RCWA 的 C++ 参照表（由 gui/validation/" + generator_name +
      " 生成，请勿手改）")
    w("//")
    w("// 数据来源：权威 Python 求解器 gui/2drcwa/rcwa2d.py 的冻结输出 gui/validation/field_reference_2d/reference_2d_v1.npz。")
    w("//   源 NPZ 的 SHA-256 = " + npz_sha)
    w("//   JSON（输入、容差、源码哈希）：gui/validation/field_reference_2d/reference_2d_v1.json；校验：gui/validation/test_field_contract_2d.py。")
    w("// 长度单位 µm，时间因子 exp(−iωt)，约定见 include/stablercwa/rcwa2d_types.hpp 与 gui/validation/field_reference_2d/CONTRACT_2D.md。")
    w("// 数字用最短可往返的十进制写出（读回得到完全相同的 double），复数写成（实部, 虚部）。")
    w("//")
    w("// 内容（只含 N = (2M1+1)(2M2+1) ≤ " + str(INC_MAX_N) + " 的算例；N≈600 的大算例与全部切面 / 相位数组只在 NPZ 里）：")
    w("//   · Rcwa2dRefCase：几何（经 rcwa2d_ref_stack(id) 建出 Stack2D）、λ、θ、φ（度）、M1、M2、Jones [Es, Ep]（显式，不再归一）、")
    w("//     因式分解（0 = Li，1 = Laurent）、远场与近场的容差（|a−b| ≤ atol + rtol·|b|，逐复数）；")
    w("//   · Rcwa2dRefOrder：Result2D 的逐级数组，与 Python 同序（m 在外层、n 在内层；下标 = harmonic_index_2d），")
    w("//     Kx、Ky、kzI、kzII 以 k0 为单位；Ex_r…Tp 是入射 Jones 的响应（入射电场矢量范数 = Jones 范数 = 1）；")
    w("//   · Rcwa2dRefDepth：若干 z 处六个分量的谐波列（Ex, Ey, Ez, Hx, Hy, Hz；H 是 Z0·H，入射 |E| = 1、|Z0H| = nI）；")
    w("//     columns[i].c[分量][0/1] = 实部 / 虚部，i = 谐波下标。z 的选取避开层界面 ±1e-12 的归属带（紧靠界面取 ±1e-6）；")
    w("//   · Rcwa2dRefOperators：最小几个结构层算例的 Eps、Cx、Cy（N×N，按行存放：元素 (i, j) 的实、虚部在 [(i*N + j)*2 + 0/1]）；")
    w("//   · status = 1（Rayleigh 异常）的算例没有数值，只有期望的失败区域（\"入射区\" / \"出射区\" / \"第j层\"）与 (m, n) 列表，")
    w("//     与 rcwa2d.py 的 RayleighAnomalyError 消息一致。")
    w("// 用法：先包含 stablercwa/rcwa2d_types.hpp（或 rcwa2d.hpp），再包含本文件；本文件不依赖其他测试头文件。")
    w("")
    w("struct Rcwa2dRefOrder {")
    w("    int m, n;")
    w("    double Kx, Ky;")
    w("    double kzI[2], kzII[2];")
    w("    double Ex_r[2], Ey_r[2], Ez_r[2], Ex_t[2], Ey_t[2], Ez_t[2];")
    w("    double Rs[2], Rp[2], Ts[2], Tp[2];")
    w("    double DEr, DEt;")
    w("    int prop_r, prop_t;")
    w("};")
    w("struct Rcwa2dRefColumn { double c[6][2]; };")
    w("struct Rcwa2dRefDepth { double z; const Rcwa2dRefColumn* columns; };")
    w("struct Rcwa2dRefOperators { int layer; int N; const double* eps; const double* cx; const double* cy; };")
    w("struct Rcwa2dRefCase {")
    w("    int id;")
    w("    const char* name;")
    w("    double wavelength, theta_deg, phi_deg;")
    w("    int M1, M2;")
    w("    double jones[4];                       // Es 实部, Es 虚部, Ep 实部, Ep 虚部")
    w("    int laurent;                           // 0 = Li（生产默认），1 = Laurent")
    w("    double far_atol, far_rtol, near_atol, near_rtol;")
    w("    int status;                            // 0 = 成功，1 = RayleighAnomaly（此时只有 failure_*）")
    w("    const char* failure_region;")
    w("    int failure_count;")
    w("    const int (*failure_orders)[2];")
    w("    const Rcwa2dRefOrder* orders;")
    w("    int count;                             // N")
    w("    const Rcwa2dRefDepth* depths;")
    w("    int depth_count;")
    w("    const Rcwa2dRefOperators* operators;")
    w("    int operator_count;")
    w("};")
    w("")
    table = []
    for index, spec in enumerate(specs, start=1):
        cid = index
        if spec["status"] == "ok" and spec["N"] > INC_MAX_N:
            continue
        if spec["status"] == "ok":
            a = arrays[spec["id"]]
            N = spec["N"]
            w(f"// 算例 {cid}：{spec['id']} —— {spec['description']}")
            w(f"[[maybe_unused]] static const Rcwa2dRefOrder kRcwa2dRef{cid}[] = {{")
            for i in range(N):
                m, n = (int(v) for v in a["mn"][i])
                fields = [number(a["Kx"][i]), number(a["Ky"][i])]
                fields_c = []
                for key in ("kzI", "kzII", "Ex_r", "Ey_r", "Ez_r", "Ex_t", "Ey_t", "Ez_t", "Rs", "Rp", "Ts", "Tp"):
                    fields_c.append("{" + number(a[key][i].real) + ", " + number(a[key][i].imag) + "}")
                w(f"    {{{m}, {n}, {fields[0]}, {fields[1]},")
                w("     " + ", ".join(fields_c[:2]) + ",")
                w("     " + ", ".join(fields_c[2:8]) + ",")
                w("     " + ", ".join(fields_c[8:]) + ",")
                w(f"     {number(a['DEr'][i])}, {number(a['DEt'][i])}, {int(a['prop_r'][i])}, {int(a['prop_t'][i])}}},")
            w("};")
            chosen = depth_selection(spec, depth_count(N))
            for k, iz in enumerate(chosen):
                w(f"[[maybe_unused]] static const Rcwa2dRefColumn kRcwa2dRef{cid}Z{k}[] = {{   // z = {number(spec['zs'][iz])}")
                for i in range(N):
                    cells = ", ".join("{" + number(a["harmonics"][c, iz, i].real) + ", " + number(a["harmonics"][c, iz, i].imag) + "}"
                                      for c in range(6))
                    w(f"    {{{{{cells}}}}},")
                w("};")
            w(f"[[maybe_unused]] static const Rcwa2dRefDepth kRcwa2dRef{cid}Depths[] = {{")
            for k, iz in enumerate(chosen):
                w(f"    {{{number(spec['zs'][iz])}, kRcwa2dRef{cid}Z{k}}},")
            w("};")
            ops = [(int(key.split("_L")[1]), key) for key in a if key.startswith("eps_L")]
            ops_ok = spec["save_operators"] and N <= 35
            op_names = []
            if ops_ok:
                for layer, _ in sorted(ops):
                    for name in ("eps", "cx", "cy"):
                        data = a[f"{name}_L{layer}"]
                        w(f"[[maybe_unused]] static const double kRcwa2dRef{cid}{name.capitalize()}L{layer}[] = {{")
                        row = []
                        for i in range(N):
                            row.append(", ".join(number(data[i, j].real) + ", " + number(data[i, j].imag) for j in range(N)))
                        w("    " + ",\n    ".join(row) + ",")
                        w("};")
                w(f"[[maybe_unused]] static const Rcwa2dRefOperators kRcwa2dRef{cid}Ops[] = {{")
                for layer, _ in sorted(ops):
                    w(f"    {{{layer}, {N}, kRcwa2dRef{cid}EpsL{layer}, kRcwa2dRef{cid}CxL{layer}, kRcwa2dRef{cid}CyL{layer}}},")
                w("};")
                op_names = [f"kRcwa2dRef{cid}Ops", len(ops)]
            else:
                op_names = ["nullptr", 0]
            w("")
            table.append((cid, spec, f"kRcwa2dRef{cid}", N, f"kRcwa2dRef{cid}Depths", len(chosen), op_names, None))
        else:
            expect = spec["expect"]
            w(f"// 算例 {cid}：{spec['id']} —— {spec['description']}（期望 Rayleigh 异常，无数值）")
            w(f"[[maybe_unused]] static const int kRcwa2dRef{cid}Fail[][2] = {{")
            w("    " + ", ".join("{" + f"{m}, {n}" + "}" for m, n in expect["orders"]) + ",")
            w("};")
            w("")
            table.append((cid, spec, "nullptr", 0, "nullptr", 0, ["nullptr", 0], f"kRcwa2dRef{cid}Fail"))
    w("[[maybe_unused]] static const Rcwa2dRefCase kRcwa2dRefCases[] = {")
    for cid, spec, orders, count, depths, depth_n, ops, fail in table:
        tol = spec["tolerance"]
        jn = spec["jones"]
        failure_region = "\"" + spec["expect"]["region"] + "\"" if fail else "nullptr"
        failure_n = len(spec["expect"]["orders"]) if fail else 0
        w(f"    {{{cid}, \"{spec['id']}\", {number(spec['wavelength'])}, {number(spec['theta_deg'])}, {number(spec['phi_deg'])}, "
          f"{spec['M1']}, {spec['M2']},")
        w(f"     {{{number(jn[0][0])}, {number(jn[0][1])}, {number(jn[1][0])}, {number(jn[1][1])}}}, "
          f"{1 if spec['factorization'] == 'laurent' else 0},")
        w(f"     {number(tol['far']['atol'])}, {number(tol['far']['rtol'])}, {number(tol['near']['atol'])}, {number(tol['near']['rtol'])},")
        w(f"     {0 if spec['status'] == 'ok' else 1}, {failure_region}, {failure_n}, {fail or 'nullptr'},")
        w(f"     {orders}, {count}, {depths}, {depth_n}, {ops[0]}, {ops[1]}}},")
    w("};")
    w(f"[[maybe_unused]] static const int kRcwa2dRefCaseCount = {len(table)};")
    w("")
    w("// 按编号取算例；编号不在表里抛 std::invalid_argument。")
    w("[[maybe_unused]] static const Rcwa2dRefCase& rcwa2d_ref_case(int id) {")
    w("    for (int i = 0; i < kRcwa2dRefCaseCount; ++i)")
    w("        if (kRcwa2dRefCases[i].id == id) return kRcwa2dRefCases[i];")
    w("    throw std::invalid_argument(\"rcwa2d_ref_case: unknown case id\");")
    w("}")
    w("")
    w("// 按编号建参照算例的结构（自上而下：入射介质 | 各层 | 出射介质；Stack2D 的 nI 为实数）。")
    w("[[maybe_unused]] static stablercwa::rcwa::Stack2D rcwa2d_ref_stack(int id) {")
    w("    using stablercwa::cplx;")
    w("    using stablercwa::rcwa::Layer2D;")
    w("    using stablercwa::rcwa::Stack2D;")
    w("    using stablercwa::rcwa::XProfile2D;")
    w("    switch (id) {")
    for cid, spec, *_ in table:
        st = spec["stack"]
        w(f"    case {cid}: {{   // {spec['id']}")
        w("        Stack2D stack;")
        w(f"        stack.nI = {number(st['nI'])};")
        w(f"        stack.nII = {cpp_index(st['nII'])};")
        w(f"        stack.Lx = {number(st['Lx'])};")
        w(f"        stack.Ly = {number(st['Ly'])};")
        for layer in st["layers"]:
            if "n" in layer:
                w(f"        stack.layers.push_back(Layer2D::uniform({number(layer['d'])}, {cpp_index(layer['n'])}));")
            else:
                ye = ", ".join(number(v) for v in layer["y_edges"])
                w(f"        stack.layers.push_back(Layer2D::patterned({number(layer['d'])}, {{{ye}}}, {{")
                for p in layer["x_profiles"]:
                    xe = ", ".join(number(v) for v in p["edges"])
                    nv = ", ".join(cpp_index(v) for v in p["n_values"])
                    w(f"            XProfile2D{{{{{xe}}}, {{{nv}}}}},")
                w("        }));")
        w("        return stack;")
        w("    }")
    w("    default: break;")
    w("    }")
    w("    throw std::invalid_argument(\"rcwa2d_ref_stack: unknown case id\");")
    w("}")
    text = "\n".join(out) + "\n"
    if len(text.encode("utf-8")) > INC_MAX_BYTES:
        raise RuntimeError(f"{INC_NAME} is {len(text.encode('utf-8'))} bytes, over the {INC_MAX_BYTES} budget")
    return text


# ---------------------------------------------------------------- driver
def tolerance_policy():
    return dict(
        far="CONTRACT_2D.md section 6: |a-b| <= atol + rtol*|b| per complex number; atol = rtol = 1e-9 for N <= 289 "
            "(relaxed to 1e-8 for the N=625 case, see near). Measured eigen-solver sensitivity: perturbing the 2N x 2N eigenproblem "
            "by 1e-16 relative noise changes far fields by <= 5.2e-13 (N <= 169) and <= 9.1e-13 (N = 625).",
        near="C++ vs frozen Python (regression): harmonic columns and slice fields |a-b| <= atol + rtol*|b|, atol = rtol = 1e-9 "
             "(N <= 289; 1e-8 at N=625). Basis: the same 1e-16 eigenproblem perturbation changes six-component harmonics and slice sums by <= 1.5e-12 "
             "(N <= 169) and <= 1.1e-11 (N = 625, slice sums of 625 terms), i.e. the tolerance is >= 600x the backward-error sensitivity of the reference itself.",
        reproduce="Same environment recomputation by this Python solver (test_field_contract_2d): atol = rtol = 5e-12 for N <= 289 "
                  "(measured dependence on the BLAS thread count <= 7.1e-13), 5e-11 at N=625 (measured 1.0e-11).",
        near_independent=dict(
            atol=1.6e-2, atol_Ex_Ey=7e-2, rtol=0.0,
            basis="Per-case tolerance.near_independent = 10 x the largest rms near-field error between rcwa2d and COMSOL 3-D FEM "
                  "(comparison report not included in this repository): Ez and H columns 1.6e-3 (cubic elements k=6, edges x3, S and P) -> atol 1.6e-2; "
                  "Ex, Ey at >= 0.15 um from an edge 6.9e-3 -> atol_Ex_Ey 7e-2 (their RCWA Gibbs error is 5-10x larger and falls with the distance to "
                  "the edge). Absolute, in units of |E_inc| = 1 (H as Z0 H, |Z0 H_inc| = nI). Meant for comparing against an independent solver (FEM), "
                  "NOT for the C++-vs-Python regression (that is tolerance.near); valid only at points >= 0.05 um from material edges."),
        phase="phase_comparison_mask=true excludes: undefined phase, structural-zero components, and |F| <= 10*(atol+rtol|F|) with the near "
              "tolerance; remaining points are compared as |exp(i*phi_a) - exp(i*phi_b)| <= 2*(atol+rtol|F_ref|)/|F_ref| + 5e-14.")


def generate(output, install_inc=False):
    output.mkdir(parents=True, exist_ok=True)
    specs = case_specs()
    arrays, records, by_case, failures = {}, [], {}, {}
    for spec in specs:
        if spec["status"] == "ok":
            _, values, diagnostics = sample_case(spec)
            check_case(spec, values)
            by_case[spec["id"]] = values
            for key, value in values.items():
                arrays[spec["id"] + "__" + key] = value
            records.append(dict(parameters=spec, diagnostics=diagnostics,
                                arrays={key: dict(shape=list(value.shape), dtype=str(value.dtype), order="C",
                                                  key=spec["id"] + "__" + key) for key, value in values.items()}))
        else:
            info = failure_info(spec)
            check_failure(spec, info)
            failures[spec["id"]] = info
            records.append(dict(parameters=spec, diagnostics=None, python_failure=info, arrays={}))
    archive = output / NPZ_NAME
    write_npz(archive, arrays)
    npz_sha = sha256(archive)
    inc_bytes = render_inc(specs, by_case, failures, npz_sha, Path(__file__).name).encode("utf-8")
    # The frozen C++ table lives in tests/; a candidate directory gets its own copy next to the other candidate files.
    inc_path = INC_FROZEN if install_inc else output / INC_NAME
    inc_path.write_bytes(inc_bytes)
    head = git("rev-parse", "HEAD")
    metadata = dict(
        contract_version=CONTRACT_VERSION, contract="gui/validation/field_reference_2d/CONTRACT_2D.md",
        anchor_revision=git("log", "-1", "--format=%H", "--", "gui/2drcwa/rcwa2d.py") or head,
        generator_revision=head,
        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        environment=dict(platform=platform.platform(), machine=platform.machine(),
                         threads={name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}),
        source_sha256={path: sha256(ROOT / path) for path in NUMERIC_SOURCES},
        builder_sha256={path: sha256(ROOT / path) for path in (
            "gui/validation/field_reference_2d_cases.py", "gui/validation/generate_field_reference_2d.py")},
        archive=dict(filename=archive.name, sha256=npz_sha, bytes=archive.stat().st_size),
        cpp_table=dict(path="tests/" + INC_NAME, sha256=hashlib.sha256(inc_bytes).hexdigest(), bytes=len(inc_bytes),
                       contains="cases with N <= %d (inputs, far fields, harmonic columns at a few z, operators of N <= 35 cases)" % INC_MAX_N),
        components=list(COMPONENTS), complex_encoding="numpy complex128 (real + i imaginary)",
        layout=dict(
            harmonic_order="column i <-> (m, n) = mn[i]; i = harmonic_index_2d = (m + M1)*(2*M2 + 1) + (n + M2); m outer, n inner (C order)",
            mn="[N,2] int64", far_fields="Kx, Ky, kzI, kzII, Ex_r..Tp, DEr, DEt, prop_r, prop_t: [N]; Kx, Ky, kzI, kzII in units of k0; inc: [2N] (Ex block, then Ey block)",
            scalars="k0 (1/um), theta, phi (radians), i0 (index of the (0,0) order): 0-d arrays",
            harmonics="[component,z,order]; order fastest; z = zs sorted ascending; H = Z0*H",
            operators="eps_L{j}, cx_L{j}, cy_L{j}: [N,N] complex, row i / column j are harmonic indices (only structured layers j, 0-based layer index, only cases with N <= 49)",
            slices="fields_xz / fields_yz: [component,z,t]; t fastest; x-z slice: x = t_xz = arange(nt)*Lx/nt, y = fixed_xz; "
                   "y-z slice: y = t_yz = arange(nt)*Ly/nt, x = fixed_yz; z = zs; direct Fourier sum (rcwa2d.field_slice, as in gui_worker)",
            E2="E2_xz / E2_yz: [z,t] float64, sum |E|^2 of the raw complex fields",
            phase="phase_absolute_* / phase_incident_*: [component,z,t] radians in [-pi,pi], NaN where phase_mask_* is true",
            phase_mask="true = undefined (non-finite or |F| <= 1e-6 * the component's peak over that slice)",
            phase_comparison_mask="true = excluded from cross-backend phase comparison (see comparison.phase_exclusion)",
            incident_amplitudes="[|E_inc|, |Z0H_inc|] = [1, nI] (Jones vectors have unit norm)"),
        carrier="phase_incident = arg(F * conj(exp(i*(-Re(kx0)*x - Re(ky0)*y + Re(kzI0)*z)))), kx0 = k0*Kx[i0], ky0 = k0*Ky[i0], kzI0 = k0*kzI[i0]; "
                "x, y take the varying coordinate t and the fixed coordinate of the slice (CONTRACT.md section 5)",
        phase_options=dict(phase_floor=1e-6, unwrap=None),
        comparison=dict(complex_fields="abs(actual-reference) <= atol+rtol*abs(reference)",
                        structural_zero="Only abs(field)<=field atol; raw phase is not an oracle (roundoff-noise phases)",
                        phase_exclusion="phase_mask OR abs(field)<=10*(atol+rtol*abs(field)) OR structural zero component",
                        phase_metric="abs(exp(i*actual_phase)-exp(i*reference_phase))",
                        phase_tolerance="2*(atol+rtol*abs(reference_field))/abs(reference_field)+5e-14",
                        not_compared="W, V (eigenbasis is not unique inside degenerate subspaces)"),
        tolerance_policy=tolerance_policy(),
        cases=records)
    (output / JSON_NAME).write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="Candidate directory; frozen data requires --refresh-baseline")
    parser.add_argument("--refresh-baseline", action="store_true",
                        help="Explicitly allow updating the checked-in baseline (and tests/rcwa2d_reference.inc)")
    args = parser.parse_args()
    output = args.output.resolve()
    frozen = output == FROZEN_DIR.resolve()
    if frozen and not args.refresh_baseline:
        parser.error("Frozen corpus is read-only by default; use a candidate directory or explicitly --refresh-baseline")
    if not frozen and args.refresh_baseline:
        parser.error("--refresh-baseline applies only to gui/validation/field_reference_2d")
    manifest = generate(output, install_inc=frozen)
    print(f"{len(manifest['cases'])} cases, {manifest['archive']['bytes']} bytes NPZ, "
          f"{manifest['cpp_table']['bytes']} bytes .inc: {output}")


if __name__ == "__main__":
    main()
