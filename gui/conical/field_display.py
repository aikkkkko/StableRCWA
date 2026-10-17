# -*- coding: utf-8 -*-
"""圆锥 RCWA 的通用二维 ``x-z`` 场显示。

Matplotlib 只在调用绘图函数时导入，因此纯数值求解和自检仍只依赖
NumPy。结构沿 y 方向不变；``y`` 参数只改变公共相位。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from field_phase import phase_quantity as _phase_quantity

try:
    from .fields_conical import field_map
except ImportError:
    from fields_conical import field_map

__all__ = [
    "field_quantity", "draw_structure_outline", "plot_field_2d",
    "plot_field_dashboard",
]


_COMPONENT_LABELS = {
    "E2": r"$|\mathbf{E}|^2$",
    "H2": r"$|Z_0\mathbf{H}|^2$",
    "Ex": r"$E_x$", "Ey": r"$E_y$", "Ez": r"$E_z$",
    "Hx": r"$Z_0H_x$", "Hy": r"$Z_0H_y$", "Hz": r"$Z_0H_z$",
    "Dx": r"$D_x/\epsilon_0$", "Dy": r"$D_y/\epsilon_0$", "Dz": r"$D_z/\epsilon_0$",
    "Bx": r"$B_x/\mu_0$", "By": r"$B_y/\mu_0$", "Bz": r"$B_z/\mu_0$",
    "E": r"$\mathbf{u}^\dagger\mathbf{E}$",
    "H": r"$\mathbf{u}^\dagger Z_0\mathbf{H}$",
    "D": r"$\mathbf{u}^\dagger\mathbf{D}/\epsilon_0$",
    "B": r"$\mathbf{u}^\dagger\mathbf{B}/\mu_0$",
}

_REPRESENTATION_LABELS = {
    "value": "", "intensity": r"$|\,{name}\,|^2$",
    "amplitude": r"$|\,{name}\,|$", "real": r"$\Re({name})$",
    "imag": r"$\Im({name})$", "phase": r"$\arg({name})$ (rad)",
}


def _canonical_component(component):
    text = str(component).strip()
    aliases = {
        "e2": "E2", "|e|2": "E2", "|e|^2": "E2",
        "h2": "H2", "|h|2": "H2", "|h|^2": "H2",
        "ex": "Ex", "ey": "Ey", "ez": "Ez",
        "hx": "Hx", "hy": "Hy", "hz": "Hz",
        "dx": "Dx", "dy": "Dy", "dz": "Dz",
        "bx": "Bx", "by": "By", "bz": "Bz",
        "e": "E", "h": "H", "d": "D", "b": "B",
    }
    key = aliases.get(text.lower())
    if key is None:
        raise ValueError("未知场分量 %r；可选 %s" %
                         (component, ", ".join(_COMPONENT_LABELS)))
    return key


def _canonical_representation(component, representation):
    rep = str(representation).strip().lower()
    aliases = {"abs2": "intensity", "power": "intensity", "abs": "amplitude",
               "magnitude": "amplitude", "re": "real", "im": "imag",
               "angle": "phase", "raw": "value"}
    rep = aliases.get(rep, rep)
    if rep == "auto":
        return "value" if component in ("E2", "H2") else "intensity"
    if rep not in _REPRESENTATION_LABELS:
        raise ValueError("representation 必须是 auto/value/intensity/amplitude/real/imag/phase")
    if rep == "value" and component not in ("E2", "H2"):
        raise ValueError("复数场分量不能直接显示 value；请选 real/imag/amplitude/intensity/phase")
    if rep == "phase" and component in ("E2", "H2"):
        raise ValueError("E2/H2 是实强度，不能显示相位；请选择复场分量")
    return rep


def field_quantity(fields, component="E2", representation="auto", *, res=None,
                   phase_reference="absolute", phase_floor=1e-6,
                   projection=None, unwrap=None):
    """从 :func:`fields_conical.field_map` 结果提取可绘制的实数二维量。

    ``E2/H2`` 本身已是六分量强度和；因此对它们选
    ``intensity`` 不会再平方。
    """
    component = _canonical_component(component)
    if component not in fields:
        if component not in ("E", "H", "D", "B") or projection is None:
            raise KeyError("场网格中没有分量 %s" % component)
    rep = _canonical_representation(component, representation)
    if rep == "phase":
        return _phase_quantity(
            res, fields, component, reference=phase_reference,
            phase_floor=phase_floor, projection=projection, unwrap=unwrap)
    if component in ("E", "H", "D", "B"):
        raise ValueError("矢量投影目前仅支持 representation='phase'")
    value = np.asarray(fields[component])
    if value.ndim != 2:
        raise ValueError("场分量必须是 (nz,nx) 二维数组")

    if rep in ("value", "intensity") and component in ("E2", "H2"):
        data = np.real(value)
    elif rep == "intensity":
        data = np.abs(value) ** 2
    elif rep == "amplitude":
        data = np.sqrt(np.maximum(np.real(value), 0.0)) if component in ("E2", "H2") else np.abs(value)
    elif rep == "real":
        data = np.real(value)
    elif rep == "imag":
        data = np.imag(value)
    else:  # pragma: no cover - 上面的规范化已经封闭全部分支
        raise AssertionError(rep)
    return np.asarray(data, float)


def _unit_scale(units):
    key = str(units).strip().lower().replace("μ", "u").replace("µ", "u")
    if key in ("um", "micron", "microns"):
        return 1.0, r"$\mu$m"
    if key in ("nm", "nanometer", "nanometers"):
        return 1e3, "nm"
    raise ValueError("units 只支持 'nm' 或 'um'")


def _quantity_label(component, representation):
    name = _COMPONENT_LABELS[component]
    if representation == "value":
        return name
    # 去掉分量名本身的数学模式分隔符，便于嵌入外层公式。
    bare = name[1:-1] if name.startswith("$") and name.endswith("$") else name
    return _REPRESENTATION_LABELS[representation].format(name=bare)


def draw_structure_outline(ax, res, units="nm", color="white", linewidth=.45,
                           alpha=.82, draw_period_edges=True):
    """在场图上叠加实际多层界面与各光栅层的横向分界。

    此函数不假定任何特定的光栅几何。优先读取原始 ``stack.layers``，因此也
    兼容磁性圆锥分支内部为 2D 后端转换过的层对象。
    """
    if "amps" not in res:
        raise ValueError("请用 solve_conical(..., need_fields=True) 重新求解")
    factor, _ = _unit_scale(units)
    period = float(res["stack"].period)
    x0, x1 = 0.0, period * factor
    z_interfaces = [0.0]
    for amp in res["amps"]:
        z_interfaces.append(float(amp["z_top"] + amp["d"]))
    ax.hlines(np.asarray(z_interfaces) * factor, x0, x1, colors=color,
              linewidths=linewidth, alpha=alpha)

    source_layers = list(getattr(res["stack"], "layers", ()))
    if len(source_layers) != len(res["amps"]):
        source_layers = [amp["layer"] for amp in res["amps"]]
    for amp, layer in zip(res["amps"], source_layers):
        if bool(getattr(layer, "homogeneous", False)):
            continue
        edges = getattr(layer, "edges", None)
        if edges is None:
            continue
        edges = np.asarray(edges, float)
        eps_values = getattr(layer, "eps_values", None)
        mu_values = getattr(layer, "mu_values", None)
        n_values = getattr(layer, "n_values", None)
        if eps_values is not None and mu_values is not None:
            values = np.column_stack([np.asarray(eps_values), np.asarray(mu_values)])
        elif n_values is not None:
            values = np.asarray(n_values)[:, None]
        else:
            continue
        if len(edges) != len(values) + 1:
            continue
        # 只画材料参数真正改变的边界，包括周期首尾。
        positions = []
        for j in range(len(values)):
            if not np.all(np.isclose(values[j - 1], values[j], rtol=1e-10, atol=1e-12)):
                positions.append(float(np.mod(edges[j], period)))
        positions = np.unique(np.round(positions, 13))
        if draw_period_edges and any(abs(p) < 1e-12 for p in positions):
            positions = np.r_[positions, period]
        z0 = amp["z_top"] * factor
        z1 = (amp["z_top"] + amp["d"]) * factor
        ax.vlines(np.asarray(positions) * factor, z0, z1, colors=color,
                  linewidths=linewidth, alpha=alpha)
    return ax


def _normalization(data, representation, scale, vmin, vmax, dynamic_range):
    from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm

    scale = str(scale).strip().lower()
    if scale == "auto":
        scale = "phase" if representation == "phase" else (
            "symmetric" if representation in ("real", "imag") else "linear")
    if scale not in ("linear", "log", "symmetric", "phase"):
        raise ValueError("scale 必须是 auto/linear/log/symmetric/phase")
    if representation == "phase" and scale not in ("auto", "phase", "linear"):
        raise ValueError("相位场不支持 log/symmetric 归一化")
    masked = np.ma.asarray(data)
    raw = np.asarray(masked)
    valid = ~np.ma.getmaskarray(masked) & np.isfinite(raw)
    finite = raw[valid]
    if finite.size == 0:
        raise ValueError("场图中没有有限数值")

    if scale == "phase" or representation == "phase":
        lo = -np.pi if vmin is None else float(vmin)
        hi = np.pi if vmax is None else float(vmax)
        return data, Normalize(lo, hi), "twilight_shifted", "phase"
    if scale == "log":
        if np.any(finite < 0):
            raise ValueError("对数显示要求非负数据；复场请先选 amplitude/intensity")
        hi = float(np.max(finite)) if vmax is None else float(vmax)
        if not np.isfinite(hi) or hi <= 0:
            hi = 1.0
        lo = hi * 10.0 ** (-float(dynamic_range)) if vmin is None else float(vmin)
        if lo <= 0 or hi <= lo:
            raise ValueError("对数显示要求 0 < vmin < vmax")
        return np.ma.maximum(masked, lo), LogNorm(lo, hi, clip=True), "magma", "log"
    if scale == "symmetric":
        lim = max(abs(float(np.min(finite))), abs(float(np.max(finite))))
        if vmin is not None or vmax is not None:
            lim = max(abs(float(vmin if vmin is not None else -lim)),
                      abs(float(vmax if vmax is not None else lim)))
        lim = lim if lim > 0 else 1.0
        return data, TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim), "RdBu_r", "symmetric"
    lo = float(np.min(finite)) if vmin is None else float(vmin)
    hi = float(np.max(finite)) if vmax is None else float(vmax)
    if lo >= 0 and vmin is None:
        lo = 0.0
    if not hi > lo:
        hi = lo + max(abs(lo), 1.0)
    return data, Normalize(lo, hi), "viridis", "linear"


def plot_field_2d(res, zs=None, nx=512, component="E2", representation="auto",
                  scale="auto", y=0.0, jones_in=None, field_data=None, ax=None,
                  phase_reference="absolute", phase_floor=1e-6, projection=None,
                  unwrap=None,
                  units="nm", cmap=None, vmin=None, vmax=None, dynamic_range=6,
                  overlay_structure=True, outline_color="white", colorbar=True,
                  title=None, rasterized=True):
    """绘制一个场分量的二维 ``x-z`` 分布。

    Parameters
    ----------
    component : {'E2','H2','Ex','Ey','Ez','Hx','Hy','Hz','Dx','Dy','Dz','Bx','By','Bz'}
    representation : {'auto','intensity','amplitude','real','imag','phase'}
    scale : {'auto','linear','log','symmetric','phase'}
        ``dynamic_range=6`` 表示对数色标默认保留最大值下 6 个数量级。
    field_data : dict, optional
        已经计算的 ``field_map``；用于多子图共享一次场重建。

    Returns
    -------
    fig, ax, image, data
        ``data`` 是未被对数下限截断的物理显示量。
    """
    import matplotlib.pyplot as plt

    if field_data is None:
        if zs is None:
            raise ValueError("未给 field_data 时必须给出 zs")
        field_data = field_map(res, zs, nx=nx, y=y, jones_in=jones_in)
    component = _canonical_component(component)
    representation = _canonical_representation(component, representation)
    data = field_quantity(
        field_data, component, representation, res=res,
        phase_reference=phase_reference, phase_floor=phase_floor,
        projection=projection, unwrap=unwrap)
    plot_data, norm, default_cmap, resolved_scale = _normalization(
        data, representation, scale, vmin, vmax, dynamic_range)
    factor, unit_label = _unit_scale(units)
    x = np.asarray(field_data["x"], float) * factor
    z = np.asarray(field_data["z"], float) * factor
    if data.shape != (len(z), len(x)):
        raise ValueError("场分量尺寸与 x/z 坐标不一致")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
    else:
        fig = ax.figure
    selected_cmap = plt.get_cmap(default_cmap if cmap is None else cmap)
    if np.ma.isMaskedArray(plot_data) and np.any(np.ma.getmaskarray(plot_data)):
        try:
            selected_cmap = selected_cmap.copy()
        except AttributeError:  # 兼容旧 Matplotlib
            pass
        selected_cmap.set_bad((0.55, 0.55, 0.55, 0.0))
    image = ax.pcolormesh(x, z, plot_data, shading="auto", cmap=selected_cmap,
                          norm=norm, rasterized=rasterized)
    if overlay_structure:
        draw_structure_outline(ax, res, units=units, color=outline_color)
    ax.set(xlabel=rf"$x$ ({unit_label})", ylabel=rf"$z$ ({unit_label})",
           xlim=(0.0, float(res["stack"].period) * factor))
    ax.invert_yaxis()
    if title is None:
        title = _quantity_label(component, representation)
        if representation == "phase" and isinstance(phase_reference, str):
            if phase_reference.lower() in ("incident", "input"):
                title += "（相对入射参考）"
            if unwrap not in (None, False, "none", "None"):
                title += "（unwrap）"
        if resolved_scale == "log":
            title += "（对数色标）"
    ax.set_title(title)
    if colorbar:
        cbar = fig.colorbar(image, ax=ax, pad=.018, aspect=18)
        cbar.set_label(_quantity_label(component, representation))
        if representation == "phase":
            cbar.set_ticks([-np.pi, 0, np.pi])
            cbar.set_ticklabels([r"$-\pi$", "0", r"$\pi$"])
    return fig, ax, image, data


def _broadcast(value, count, name):
    if isinstance(value, str) or value is None:
        return [value] * count
    seq = list(value)
    if len(seq) != count:
        raise ValueError("%s 的长度必须与 components 相同" % name)
    return seq


def plot_field_dashboard(res, zs, nx=512,
                         components=("E2", "Ex", "Ey", "Ez"),
                         representations="auto", scales="auto", titles=None,
                         y=0.0, jones_in=None, phase_reference="absolute",
                         phase_floor=1e-6, projection=None, unwrap=None,
                         units="nm", ncols=2,
                         figsize=None, cmaps=None, dynamic_range=6,
                         overlay_structure=True, outline_color="white"):
    """一次重建场，绘制多分量二维面板。

    ``representations/scales/cmaps/titles`` 可以是单个值，也可以是与
    ``components`` 等长的序列。返回 ``(fig, axes, fields)``，其中
    ``fields`` 可继续传给 :func:`plot_field_2d`。
    """
    import matplotlib.pyplot as plt

    components = tuple(components)
    if not components:
        raise ValueError("components 不能为空")
    if int(ncols) != ncols or ncols < 1:
        raise ValueError("ncols 必须是正整数")
    ncols = min(int(ncols), len(components))
    nrows = int(np.ceil(len(components) / ncols))
    reps = _broadcast(representations, len(components), "representations")
    scl = _broadcast(scales, len(components), "scales")
    ttl = _broadcast(titles, len(components), "titles")
    cmap_values = _broadcast(cmaps, len(components), "cmaps")
    if figsize is None:
        figsize = (6.8 * ncols, 3.4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False,
                             sharex=True, sharey=True)
    fields = field_map(res, zs, nx=nx, y=y, jones_in=jones_in)
    for i, (component, rep, scale_value, title_value, cmap_value) in enumerate(
            zip(components, reps, scl, ttl, cmap_values)):
        plot_field_2d(
            res, field_data=fields, component=component, representation=rep,
            scale=scale_value, ax=axes.flat[i], units=units, cmap=cmap_value,
            dynamic_range=dynamic_range, overlay_structure=overlay_structure,
            outline_color=outline_color, title=title_value,
            phase_reference=phase_reference, phase_floor=phase_floor,
            projection=projection, unwrap=unwrap)
    for ax in axes.flat[len(components):]:
        ax.set_visible(False)
    fig.tight_layout(pad=.7)
    return fig, axes, fields
