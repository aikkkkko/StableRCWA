"""JSON interchange and exact geometry for the periodic 2D RCWA GUI."""
from __future__ import annotations

import copy
import importlib.util
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np

from gui_models import _number, complex_number, encode_n


@lru_cache(maxsize=1)
def load_rcwa2d():
    """Load the sibling 2drcwa module under a stable, importable module name."""
    name = "_rcwasolver_gui_rcwa2d"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parent / "2drcwa" / "rcwa2d.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载二维求解器：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _edges(values, count, period, label):
    try:
        edges = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(label + "边界数据无效") from None
    if edges.ndim != 1 or len(edges) != count + 1 or not 2 <= len(edges) <= 2001:
        raise ValueError(label + "边界坐标数必须比分段数多1")
    if not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError(label + "边界必须严格递增")
    if abs((edges[-1] - edges[0]) - period) > 1e-8 * max(period, 1.):
        raise ValueError(label + "首尾边界之差必须等于周期")
    return edges.tolist()


def normalize_2d(cfg):
    """Validate and normalize a complete schema-1 periodic2d configuration."""
    if not isinstance(cfg, dict):
        raise ValueError("模型需要是JSON对象")
    cfg = copy.deepcopy(cfg)
    if cfg.get("schema_version", 1) != 1:
        raise ValueError("不支持此模型文件版本")
    if cfg.get("mode") != "periodic2d":
        raise ValueError("二维模型求解方式应为 periodic2d")
    cfg["schema_version"] = 1
    cfg["name"] = str(cfg.get("name", "自定义二维结构"))[:120]
    inc, st, sam = cfg.get("incident"), cfg.get("structure"), cfg.get("sampling")
    if not all(isinstance(v, dict) for v in (inc, st, sam)):
        raise ValueError("模型参数格式不完整")
    # 可选：这个模型用哪个二维求解器（界面的“二维求解”选项）。不写时由环境变量 RCWA_2D_BACKEND 与默认决定（cpp_backend.backend_2d_for）。
    if cfg.get("backend_2d") is None:
        cfg.pop("backend_2d", None)
    else:
        choice = str(cfg["backend_2d"]).strip().lower()
        if choice not in ("python", "cpp"):
            raise ValueError("二维求解应为 python 或 cpp")
        cfg["backend_2d"] = choice
    inc["wavelength"] = _number(inc.get("wavelength"), "波长", 1e-9, 1e6)
    inc["angle_deg"] = _number(inc.get("angle_deg"), "入射角", -89.9, 89.9)
    inc["phi_deg"] = _number(inc.get("phi_deg", 0.), "方位角", -360., 360.)
    inc["M"] = _number(inc.get("M"), "x方向Fourier截断M", 0, 30, integer=True)
    inc["My"] = _number(inc.get("My"), "y方向Fourier截断My", 0, 30, integer=True)
    if (2 * inc["M"] + 1) * (2 * inc["My"] + 1) > 625:
        raise ValueError("二维谐波总数不得超过625")
    inc["psi_deg"] = _number(inc.get("psi_deg", 45.), "偏振混合角", 0., 90.)
    inc["delta_deg"] = _number(inc.get("delta_deg", 0.), "s相对p的相位差", -360., 360.)
    inc["polarization"] = {"TE": "s", "TM": "p"}.get(inc.get("polarization"), inc.get("polarization"))
    if inc["polarization"] not in ("s", "p", "jones"):
        raise ValueError("二维偏振请选择s、p或jones")
    if inc["polarization"] in ("s", "p"):
        inc["psi_deg"] = 90. if inc["polarization"] == "s" else 0.
        inc["delta_deg"] = 0.
    lx = st["period"] = _number(st.get("period"), "x周期", 1e-9, 1e6)
    ly = st["period_y"] = _number(st.get("period_y"), "y周期", 1e-9, 1e6)
    for key, label in (("n_incident", "入射区折射率"), ("n_substrate", "出射区折射率")):
        n = complex_number(st.get(key), label)
        if n.imag:
            raise ValueError(label + "在图形入口中要求为无损实数")
        st[key] = encode_n(n)
    layers = st.get("layers")
    if not isinstance(layers, list) or not 1 <= len(layers) <= 256:
        raise ValueError("层数需为1至256")
    for j, layer in enumerate(layers):
        label = f"第{j + 1}层"
        if not isinstance(layer, dict):
            raise ValueError(label + "格式不正确")
        layer["name"] = str(layer.get("name", label))[:120]
        layer["thickness"] = _number(layer.get("thickness"), label + "厚度", 1e-9, 1e6)
        kind = layer.get("kind")
        if kind == "uniform":
            layer["n"] = encode_n(complex_number(layer.get("n"), label + "折射率"))
            for key in ("edges", "n_values", "y_edges", "x_profiles"):
                layer.pop(key, None)
        elif kind == "patterned":
            values = layer.get("n_values")
            if not isinstance(values, list):
                raise ValueError(label + "分段折射率数据无效")
            layer["n_values"] = [encode_n(complex_number(v, label + "分段折射率")) for v in values]
            layer["edges"] = _edges(layer.get("edges"), len(values), lx, label + "x方向")
            for key in ("n", "y_edges", "x_profiles"):
                layer.pop(key, None)
        elif kind == "patterned2d":
            profiles = layer.get("x_profiles")
            if not isinstance(profiles, list) or not profiles:
                raise ValueError(label + "x剖面数据无效")
            layer["y_edges"] = _edges(layer.get("y_edges"), len(profiles), ly, label + "y方向")
            normalized = []
            for k, profile in enumerate(profiles):
                if not isinstance(profile, dict) or not isinstance(profile.get("n_values"), list):
                    raise ValueError(label + f"第{k + 1}段x剖面无效")
                vals = [encode_n(complex_number(v, label + "分段折射率")) for v in profile["n_values"]]
                edges = _edges(profile.get("edges"), len(vals), lx, label + f"第{k + 1}段x方向")
                normalized.append({"edges": edges, "n_values": vals})
            layer["x_profiles"] = normalized
            for key in ("n", "edges", "n_values"):
                layer.pop(key, None)
        else:
            raise ValueError(label + "类型应为uniform、patterned或patterned2d")
        if layer.get("fourier_method", "analytic") != "analytic":
            raise ValueError(label + "二维求解器仅支持解析Fourier系数，不支持FFT")
        if _number(layer.get("fft_samples", 0), label + "FFT采样数", 0, integer=True):
            raise ValueError(label + "二维求解器不支持FFT采样")
        layer["fourier_method"] = "analytic"
        layer["fft_samples"] = 0
        if layer.get("background_n") is not None:
            raise ValueError(label + "二维求解器不支持背景材料覆盖")
        layer["background_n"] = None
    sam["z_min"] = _number(sam.get("z_min", -.2), "显示z下限")
    sam["z_max"] = _number(sam.get("z_max", 1.4), "显示z上限")
    if sam["z_max"] <= sam["z_min"]:
        raise ValueError("显示z上限必须大于下限")
    sam["nx"] = _number(sam.get("nx", 360), "横向采样数", 40, 1600, integer=True)
    sam["nz"] = _number(sam.get("nz", 220), "纵向采样数", 20, 1200, integer=True)
    sam["slice_plane"] = sam.get("slice_plane", "xz")
    if sam["slice_plane"] not in ("xz", "yz"):
        raise ValueError("切面请选择xz或yz")
    fixed_period = ly if sam["slice_plane"] == "xz" else lx
    sam["slice_at"] = _number(sam.get("slice_at", 0.), "切面位置", 0., fixed_period)
    pml = cfg.get("pml", {})
    if not isinstance(pml, dict):
        raise ValueError("PML参数格式不正确")
    pml.setdefault("left", .15 * lx)
    pml.setdefault("right", .15 * lx)
    pml.setdefault("strength", 3.)
    pml.setdefault("order", 3)
    pml.setdefault("x0", 0.)
    cfg.update(incident=inc, structure=st, pml=pml, sampling=sam)
    return cfg


def from_stack_2d(stack, **model_options):
    """Export a nonmagnetic Stack2D to a complete schema-1 GUI model."""
    opts = dict(model_options)
    name = opts.pop("name", "自定义二维结构")
    wavelength = opts.pop("lam0", opts.pop("wavelength", 1.))
    angle = opts.pop("theta_deg", opts.pop("angle_deg", 0.))
    phi = opts.pop("phi_deg", 0.)
    mx = opts.pop("M", 4)
    my = opts.pop("My", 4)
    pol = opts.pop("pol", opts.pop("polarization", "s"))
    psi = opts.pop("psi_deg", 45.)
    delta = opts.pop("delta_deg", 0.)
    sampling = opts.pop("sampling", {})
    sampling_options = {key: opts.pop(key) for key in ("nx", "nz", "z_min", "z_max",
                                                   "slice_plane", "slice_at") if key in opts}
    if opts:
        raise TypeError("未知二维模型选项：" + ", ".join(sorted(opts)))
    for attr in ("muI", "muII"):
        if abs(complex(getattr(stack, attr, 1)) - 1) > 1e-12:
            raise ValueError("图形入口仅支持非磁性介电结构")
    layers = []
    for j, layer in enumerate(stack.layers):
        if abs(complex(getattr(layer, "mu", 1) or 1) - 1) > 1e-12:
            raise ValueError("图形入口仅支持非磁性介电结构")
        row = {"name": layer.name or f"层 {j + 1}", "thickness": layer.d,
               "fourier_method": "analytic", "fft_samples": 0}
        if layer.homogeneous:
            row.update(kind="uniform", n=encode_n(layer.n))
        else:
            row.update(kind="patterned2d", y_edges=np.asarray(layer.y_edges, float).tolist(),
                       x_profiles=[{"edges": np.asarray(edges, float).tolist(),
                                    "n_values": [encode_n(n) for n in values]}
                                   for edges, values in layer.x_profiles])
        layers.append(row)
    depth = sum(layer.d for layer in stack.layers)
    cfg = {"schema_version": 1, "name": name, "mode": "periodic2d",
           "incident": {"wavelength": wavelength, "angle_deg": angle, "phi_deg": phi,
                        "M": mx, "My": my, "polarization": pol, "psi_deg": psi,
                        "delta_deg": delta},
           "structure": {"period": stack.lat.Lx, "period_y": stack.lat.Ly,
                         "n_incident": encode_n(stack.nI), "n_substrate": encode_n(stack.nII),
                         "layers": layers},
           "sampling": {"z_min": -.2, "z_max": depth + .2, "nx": 80, "nz": 60,
                        "slice_plane": "xz", "slice_at": 0.}}
    if not isinstance(sampling, dict):
        raise ValueError("采样参数格式不正确")
    cfg["sampling"].update(sampling)
    cfg["sampling"].update(sampling_options)
    return normalize_2d(cfg)


def _build_stack_normalized(cfg):
    rc = load_rcwa2d()
    st = cfg["structure"]
    layers = []
    for row in st["layers"]:
        if row["kind"] == "uniform":
            layer = rc.Layer2D(row["thickness"], n=complex_number(row["n"], "折射率"), name=row["name"])
        else:
            if row["kind"] == "patterned":
                y_edges = [0., st["period_y"]]
                profiles = [(row["edges"], row["n_values"])]
            else:
                y_edges = row["y_edges"]
                profiles = [(p["edges"], p["n_values"]) for p in row["x_profiles"]]
            layer = rc.Layer2D(row["thickness"], y_edges=y_edges,
                               x_profiles=[(edges, [complex_number(n, "折射率") for n in values])
                                           for edges, values in profiles], name=row["name"])
        layers.append(layer)
    return rc.Stack2D(float(st["n_incident"]), layers, float(st["n_substrate"]),
                      rc.Lattice(st["period"], st["period_y"]))


def build_stack_2d(cfg):
    """Return a validated Stack2D suitable for solve2d."""
    return _build_stack_normalized(normalize_2d(cfg))


def _cuts(edges, period):
    return np.unique(np.r_[0., period, np.mod(np.asarray(edges, float), period)])


def _profile_values(edges, values, coordinates, period):
    edges = np.asarray(edges, float)
    points = edges[0] + np.mod(np.asarray(coordinates, float) - edges[0], period)
    indices = np.clip(np.searchsorted(edges, points, side="right") - 1, 0, len(values) - 1)
    return np.asarray([complex_number(v, "折射率") for v in values])[indices]


def _layer_values(row, x, y, lx, ly):
    if row["kind"] == "uniform":
        return np.full(np.broadcast_shapes(np.shape(x), np.shape(y)),
                       complex_number(row["n"], "折射率"), dtype=complex)
    if row["kind"] == "patterned":
        return np.broadcast_to(_profile_values(row["edges"], row["n_values"], x, lx),
                               np.broadcast_shapes(np.shape(x), np.shape(y)))
    y_edges = np.asarray(row["y_edges"], float)
    yy = y_edges[0] + np.mod(np.asarray(y, float) - y_edges[0], ly)
    indices = np.clip(np.searchsorted(y_edges, yy, side="right") - 1, 0,
                      len(row["x_profiles"]) - 1)
    x, indices = np.broadcast_arrays(x, indices)
    out = np.empty(x.shape, complex)
    for k, profile in enumerate(row["x_profiles"]):
        mask = indices == k
        if np.any(mask):
            out[mask] = _profile_values(profile["edges"], profile["n_values"], x[mask], lx)
    return out


def preview_2d(cfg):
    """Exact material rectangles covering one x-y cell of every layer."""
    cfg = normalize_2d(cfg)
    st = cfg["structure"]
    lx, ly = st["period"], st["period_y"]
    z = 0.
    layers = []
    for row in st["layers"]:
        ycuts = _cuts(row["y_edges"], ly) if row["kind"] == "patterned2d" else np.array([0., ly])
        rectangles = []
        for ya, yb in zip(ycuts[:-1], ycuts[1:]):
            ym = (ya + yb) / 2
            if row["kind"] == "patterned2d":
                ye = np.asarray(row["y_edges"], float)
                yp = ye[0] + (ym - ye[0]) % ly
                k = np.clip(np.searchsorted(ye, yp, side="right") - 1, 0, len(row["x_profiles"]) - 1)
                xcuts = _cuts(row["x_profiles"][k]["edges"], lx)
            elif row["kind"] == "patterned":
                xcuts = _cuts(row["edges"], lx)
            else:
                xcuts = np.array([0., lx])
            for xa, xb in zip(xcuts[:-1], xcuts[1:]):
                n = complex(_layer_values(row, np.array([(xa + xb) / 2]), np.array([ym]), lx, ly)[0])
                e = n * n
                rectangles.append({"x0": float(xa), "x1": float(xb), "y0": float(ya),
                                   "y1": float(yb), "n": encode_n(n),
                                   "epsilon_real": float(e.real), "epsilon_imag": float(e.imag)})
        layers.append({"name": row["name"], "z0": z, "z1": z + row["thickness"],
                       "rectangles": rectangles,
                       "fourier": {"method": "uniform" if row["kind"] == "uniform" else "analytic",
                                   "name": row["name"]}})
        z += row["thickness"]
    return {"config": cfg, "x0": 0., "x1": lx, "y0": 0., "y1": ly,
            "depth": z, "layers": layers}


def material_slice_2d(cfg, t, z, plane="xz", at=0):
    """Return exact n[z,t] on an xz or yz section, including exterior media."""
    cfg = normalize_2d(cfg)
    if plane not in ("xz", "yz"):
        raise ValueError("切面请选择xz或yz")
    t = np.asarray(t, float)
    z = np.asarray(z, float)
    if t.ndim != 1 or z.ndim != 1 or not np.isfinite(t).all() or not np.isfinite(z).all():
        raise ValueError("切面坐标必须是一维有限数组")
    at = _number(at, "切面位置")
    st = cfg["structure"]
    lx, ly = st["period"], st["period_y"]
    x, y = (t, np.full(t.shape, at)) if plane == "xz" else (np.full(t.shape, at), t)
    ends = np.cumsum([row["thickness"] for row in st["layers"]])
    out = np.empty((len(z), len(t)), complex)
    for j, zz in enumerate(z):
        if zz < 0:
            out[j] = float(st["n_incident"])
        elif zz > ends[-1]:
            out[j] = float(st["n_substrate"])
        else:
            k = min(int(np.searchsorted(ends, zz, side="left")), len(ends) - 1)
            out[j] = _layer_values(st["layers"][k], x, y, lx, ly)
    return out


def preset_2d():
    """A small rectangular pillar with two uniform films."""
    cfg = {"schema_version": 1, "name": "二维矩形柱与双层膜", "mode": "periodic2d",
           "incident": {"wavelength": 1., "angle_deg": 20., "phi_deg": 30.,
                        "polarization": "s", "M": 4, "My": 4,
                        "psi_deg": 90., "delta_deg": 0.},
           "structure": {"period": 1., "period_y": .8, "n_incident": 1., "n_substrate": 1.45,
                         "layers": [
                             {"name": "矩形介质柱", "kind": "patterned2d", "thickness": .3,
                              "y_edges": [0., .2, .6, .8],
                              "x_profiles": [
                                  {"edges": [0., 1.], "n_values": [1.]},
                                  {"edges": [0., .3, .7, 1.], "n_values": [1., 1.7, 1.]},
                                  {"edges": [0., 1.], "n_values": [1.]}]},
                             {"name": "第一膜", "kind": "uniform", "thickness": .16, "n": 1.46},
                             {"name": "第二膜", "kind": "uniform", "thickness": .11, "n": 1.9}]},
           "sampling": {"nx": 80, "nz": 60, "z_min": -.2, "z_max": .77,
                        "slice_plane": "xz", "slice_at": .4}}
    return normalize_2d(cfg)
