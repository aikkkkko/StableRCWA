# -*- coding: utf-8 -*-
"""RCWA 重建复场的相位后处理与二维显示。

求解器给出的 ``field_map`` / ``field_xy`` 结果已经是频域复相量，故相位
必须在所有 Fourier 级次重建到实空间之后计算。这里提供两种常用定义：

``reference='absolute'``
    ``arg(F)``，保留入射平面波的传播条纹；
``reference='incident'``
    ``arg(F * conj(F_inc))``，去除求解器约定下的入射参考波相位，显示局部
    散射相移。

本模块只依赖 NumPy；Matplotlib 仅在调用 :func:`plot_phase_2d` 时延迟导入。
它适用于顶层 1D、``2drcwa``、``conical`` 及 ``mag`` 分支已有的实空间场图。
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

__all__ = [
    "complex_component", "incident_reference", "phase_data", "phase_quantity",
    "plot_phase_2d",
]


_MISSING = object()
_VECTOR_COMPONENTS = {
    "E": ("Ex", "Ey", "Ez"),
    "H": ("Hx", "Hy", "Hz"),
    "D": ("Dx", "Dy", "Dz"),
    "B": ("Bx", "By", "Bz"),
}
_REAL_INTENSITIES = {"E2", "H2"}


def _value(obj, name, default=_MISSING):
    """同时兼容 dict 式 Result 与普通对象属性。"""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _field_key(fields, component):
    requested = str(component).strip()
    if requested in fields:
        return requested
    lookup = {str(key).lower(): key for key in fields}
    key = lookup.get(requested.lower())
    if key is None:
        raise KeyError(f"场图中没有分量 {component!r}")
    return key


def complex_component(fields, component, projection=None):
    """提取一个可定义相位的复标量场。

    ``component='Ex'`` 这类逐分量选择最直接。若想观察特定分析器方向上的
    矢量场相位，可传 ``component='E'``（或 ``H/D/B``）以及三分量
    ``projection``；返回的是 ``projection^H @ field``。

    ``E2`` / ``H2`` 是实强度，没有可定义的复相位，会明确拒绝。
    """
    label = str(component).strip()
    family = label.upper()
    if projection is not None:
        if family not in _VECTOR_COMPONENTS:
            raise ValueError("projection 仅可与 component='E'/'H'/'D'/'B' 一起使用")
        vector = np.asarray(projection, dtype=complex).reshape(-1)
        if vector.shape != (3,) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
            raise ValueError("projection 必须是非零、有限的三个复数")
        keys = tuple(_field_key(fields, key) for key in _VECTOR_COMPONENTS[family])
        values = [np.asarray(fields[key], dtype=complex) for key in keys]
        if any(value.ndim != 2 for value in values):
            raise ValueError("投影的每个场分量都必须是二维实空间场图")
        if len({value.shape for value in values}) != 1:
            raise ValueError("投影的三个场分量尺寸不一致")
        return sum(np.conj(vector[i]) * values[i] for i in range(3)), f"{family}_projection"

    key = _field_key(fields, label)
    if str(key).upper() in _REAL_INTENSITIES:
        raise ValueError(f"{key} 是实强度，不能显示相位；请选择 Ex/Ey/Ez 等复场分量")
    value = np.asarray(fields[key], dtype=complex)
    if value.ndim != 2:
        raise ValueError("相位显示要求二维实空间场图")
    return value, str(key)


def _zero_order_index(result):
    value = _value(result, "i0")
    if value is not _MISSING:
        return int(value)
    ms = _value(result, "ms")
    if ms is not _MISSING:
        found = np.where(np.asarray(ms) == 0)[0]
        if len(found) == 1:
            return int(found[0])
    mn = _value(result, "mn")
    if mn is not _MISSING:
        found = np.where(np.all(np.asarray(mn) == 0, axis=1))[0]
        if len(found) == 1:
            return int(found[0])
    raise ValueError("结果中找不到零级次，无法构造入射参考波")


def _order_value(value, index, label):
    array = np.asarray(value, dtype=complex)
    if array.ndim == 0:
        return complex(array)
    if index < 0 or index >= array.size:
        raise ValueError(f"{label} 的零级次下标无效")
    return complex(array.reshape(-1)[index])


def _incident_wavevector(result):
    """按各求解器的已有字段返回物理 ``(kx, ky, kz)``。"""
    if _value(result, "is_pml", False):
        # PML 端口是数值本征模，不能把它们当作 Rayleigh 衍射级。
        return (complex(_value(result, "incident_kxm")), 0.0,
                complex(_value(result, "incident_kz")))
    index = _zero_order_index(result)
    kxm = _value(result, "kxm")
    if kxm is not _MISSING:  # 1D / conical：kxm、kzI/kIz 已带 k0
        kx = _order_value(kxm, index, "kxm")
        ky = complex(_value(result, "ky", 0.0))
        kz_values = _value(result, "kzI")
        if kz_values is _MISSING:
            kz_values = _value(result, "kIz")
        if kz_values is _MISSING:
            raise ValueError("结果中缺少 kzI/kIz，无法构造入射参考波")
        kz = _order_value(kz_values, index, "kzI/kIz")
        return kx, ky, kz

    kx_values, ky_values = _value(result, "Kx"), _value(result, "Ky")
    kz_values, k0 = _value(result, "kzI"), _value(result, "k0")
    if any(value is _MISSING for value in (kx_values, ky_values, kz_values, k0)):
        raise ValueError("结果不包含可识别的入射波矢字段")
    kx = float(k0) * _order_value(kx_values, index, "Kx")
    ky = float(k0) * _order_value(ky_values, index, "Ky")
    kz = float(k0) * _order_value(kz_values, index, "kzI")
    return kx, ky, kz


def _plane_layout(fields, shape):
    """识别 ``x-z`` 或 ``x-y`` 实空间数组的轴顺序。"""
    x_value = _value(fields, "x")
    if x_value is _MISSING:
        raise ValueError("场图缺少 x 坐标")
    x = np.asarray(x_value, dtype=float)
    if x.ndim != 1:
        raise ValueError("x 坐标必须是一维")

    z_value = _value(fields, "z")
    if z_value is not _MISSING:
        z = np.asarray(z_value, dtype=float)
        if z.ndim == 1:
            if shape == (len(z), len(x)):
                return dict(horizontal_name="x", vertical_name="z", horizontal=x, vertical=z,
                            axis_of={"x": 1, "z": 0}, transpose_for_plot=False)
            if shape == (len(x), len(z)):
                return dict(horizontal_name="x", vertical_name="z", horizontal=x, vertical=z,
                            axis_of={"x": 0, "z": 1}, transpose_for_plot=True)

    y_value = _value(fields, "y")
    if y_value is not _MISSING:
        y = np.asarray(y_value, dtype=float)
        if y.ndim == 1:
            if shape == (len(y), len(x)):
                return dict(horizontal_name="x", vertical_name="y", horizontal=x, vertical=y,
                            axis_of={"x": 1, "y": 0}, transpose_for_plot=False)
            if shape == (len(x), len(y)):
                return dict(horizontal_name="x", vertical_name="y", horizontal=x, vertical=y,
                            axis_of={"x": 0, "y": 1}, transpose_for_plot=True)
    raise ValueError("无法根据 x/z 或 x/y 坐标识别场图网格尺寸")


def incident_reference(result, fields, shape=None):
    """返回与 ``fields`` 对齐的单位模入射平面波相位因子。

    求解器采用 ``exp(-i kx x - i ky y + i kz z)`` 的入射空间因子；这里
    仅保留该因子的相位，避免有损或倏逝参考波的幅值改变待显示场。
    """
    if result is None:
        raise ValueError("reference='incident' 时必须给出求解结果 result")
    if shape is None:
        candidates = []
        for key, value in fields.items():
            if str(key).upper() not in _REAL_INTENSITIES:
                array = np.asarray(value)
                if array.ndim == 2:
                    candidates.append(array.shape)
        if not candidates:
            raise ValueError("场图不含二维复场分量")
        shape = candidates[0]
    layout = _plane_layout(fields, tuple(shape))
    kx, ky, kz = _incident_wavevector(result)
    x = layout["horizontal"]

    if layout["vertical_name"] == "z":
        z = layout["vertical"]
        if layout["axis_of"]["x"] == 1:
            X, Z = x[None, :], z[:, None]
        else:
            X, Z = x[:, None], z[None, :]
        y = _value(fields, "y", 0.0)
        y = np.asarray(y, dtype=float)
        if y.ndim != 0:
            raise ValueError("x-z 场图的 y 必须是一个固定标量")
        argument = -np.real(kx) * X - np.real(ky) * float(y) + np.real(kz) * Z
    else:
        y = layout["vertical"]
        if layout["axis_of"]["x"] == 1:
            X, Y = x[None, :], y[:, None]
        else:
            X, Y = x[:, None], y[None, :]
        z = _value(fields, "z", 0.0)
        z = np.asarray(z, dtype=float)
        if z.ndim != 0:
            raise ValueError("x-y 场图的 z 必须是一个固定标量")
        argument = -np.real(kx) * X - np.real(ky) * Y + np.real(kz) * float(z)
    return np.exp(1j * argument)


def _unwrap_valid_segments(values, mask, axis):
    """只在连续有效区间内 unwrap，绝不跨越节点或掩膜。"""
    moved_values = np.moveaxis(np.asarray(values, dtype=float).copy(), axis, -1)
    moved_mask = np.moveaxis(np.asarray(mask, dtype=bool), axis, -1)
    for index in np.ndindex(moved_values.shape[:-1]):
        line, blocked = moved_values[index], moved_mask[index]
        start = None
        for position, valid in enumerate(~blocked):
            if valid and start is None:
                start = position
            elif not valid and start is not None:
                line[start:position] = np.unwrap(line[start:position])
                start = None
        if start is not None:
            line[start:] = np.unwrap(line[start:])
    return np.moveaxis(moved_values, -1, axis)


def _unwrap_phase(phase, mask, fields, unwrap):
    if unwrap in (None, False, "none", "None"):
        return phase, False
    layout = _plane_layout(fields, phase.shape)
    value = str(unwrap).lower()
    if value == "both":
        axes = (layout["axis_of"][layout["horizontal_name"]],
                layout["axis_of"][layout["vertical_name"]])
    else:
        aliases = {"horizontal": layout["horizontal_name"], "vertical": layout["vertical_name"]}
        value = aliases.get(value, value)
        if value not in layout["axis_of"]:
            options = "/".join((layout["horizontal_name"], layout["vertical_name"], "both"))
            raise ValueError(f"unwrap 必须为 None、{options} 或 'both'")
        axes = (layout["axis_of"][value],)
    out = np.asarray(phase, dtype=float)
    for axis in axes:
        out = _unwrap_valid_segments(out, mask, axis)
    return out, True


def _resolve_reference(result, fields, shape, reference):
    if reference is None:
        reference = "absolute"
    if isinstance(reference, str):
        name = reference.strip().lower()
        if name in ("absolute", "none"):
            return None, "absolute"
        if name in ("incident", "input"):
            return incident_reference(result, fields, shape), "incident"
        raise ValueError("reference 必须为 'absolute'、'incident' 或同尺寸复参考场")
    value = np.asarray(reference, dtype=complex)
    try:
        value = np.broadcast_to(value, shape)
    except ValueError as exc:
        raise ValueError("自定义参考场不能广播到所选场分量的尺寸") from exc
    return value, "custom"


def phase_data(result, fields, component, *, reference="absolute", phase_floor=1e-6,
               projection=None, unwrap=None):
    """返回选定复场的相位、幅值与节点掩膜。

    ``phase_floor`` 是相对本图最大幅值的阈值；低于阈值的相位物理上不稳定，
    会被标为掩膜。``unwrap`` 支持 ``'x'``、``'z'``、``'y'`` 或 ``'both'``，
    仅在每个连续的未掩膜区间内执行，默认保持 ``[-pi, pi]`` 的包裹相位。
    """
    try:
        floor = float(phase_floor)
    except (TypeError, ValueError) as exc:
        raise ValueError("phase_floor 必须是非负实数") from exc
    if not np.isfinite(floor) or floor < 0:
        raise ValueError("phase_floor 必须是非负有限数")

    field, label = complex_component(fields, component, projection=projection)
    reference_field, reference_name = _resolve_reference(result, fields, field.shape, reference)
    amplitude = np.abs(field)
    finite_amplitude = amplitude[np.isfinite(amplitude)]
    peak = float(np.max(finite_amplitude)) if finite_amplitude.size else 0.0
    mask = ~np.isfinite(field) | (amplitude <= floor * peak)
    phasor = field
    if reference_field is not None:
        mask |= ~np.isfinite(reference_field) | (np.abs(reference_field) == 0)
        phasor = field * np.conj(reference_field)
    phase = np.angle(phasor)
    phase, did_unwrap = _unwrap_phase(phase, mask, fields, unwrap)
    return dict(
        phase=np.ma.array(phase, mask=mask, copy=False),
        amplitude=amplitude,
        field=field,
        reference=reference_field,
        reference_name=reference_name,
        component=label,
        mask=mask,
        phase_floor=floor,
        unwrapped=did_unwrap,
    )


def phase_quantity(result, fields, component, **kwargs):
    """只返回 :func:`phase_data` 中用于绘图的 masked 相位数组。"""
    return phase_data(result, fields, component, **kwargs)["phase"]


def _unit_scale(units):
    key = str(units).strip().lower().replace("μ", "u").replace("µ", "u")
    if key in ("um", "micron", "microns"):
        return 1.0, r"$\mu$m"
    if key in ("nm", "nanometer", "nanometers"):
        return 1e3, "nm"
    raise ValueError("units 只支持 'nm' 或 'um'")


def plot_phase_2d(result, fields, component, *, reference="absolute", phase_floor=1e-6,
                  projection=None, unwrap=None, ax=None, units="um", cmap=None,
                  colorbar=True, title=None, invert_vertical=None):
    """绘制任一已有二维实空间场图的选定分量相位。

    对 1D/conical 的 ``field_map``，横纵坐标是 ``x-z``；对 2D 的
    ``field_xy``，横纵坐标是 ``x-y``。返回 ``(fig, ax, image, info)``，
    其中 ``info`` 是 :func:`phase_data` 的完整返回值。
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    info = phase_data(result, fields, component, reference=reference,
                      phase_floor=phase_floor, projection=projection, unwrap=unwrap)
    layout = _plane_layout(fields, info["phase"].shape)
    data = info["phase"].T if layout["transpose_for_plot"] else info["phase"]
    factor, unit_label = _unit_scale(units)
    x = layout["horizontal"] * factor
    y = layout["vertical"] * factor

    if info["unwrapped"]:
        finite = np.ma.compressed(np.ma.masked_invalid(data))
        if finite.size:
            lo, hi = float(np.min(finite)), float(np.max(finite))
            if not hi > lo:
                lo, hi = lo - np.pi, hi + np.pi
        else:
            lo, hi = -np.pi, np.pi
        norm = Normalize(lo, hi)
        default_cmap = "viridis"
    else:
        norm = Normalize(-np.pi, np.pi)
        default_cmap = "twilight_shifted"
    base_cmap = plt.get_cmap(default_cmap if cmap is None else cmap)
    try:
        base_cmap = base_cmap.copy()
    except AttributeError:  # 兼容旧 Matplotlib
        pass
    base_cmap.set_bad((0.55, 0.55, 0.55, 0.0))

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
    else:
        fig = ax.figure
    image = ax.pcolormesh(x, y, data, shading="auto", cmap=base_cmap, norm=norm)
    vertical = layout["vertical_name"]
    if invert_vertical is None:
        invert_vertical = vertical == "z"
    if invert_vertical:
        ax.invert_yaxis()
    ax.set(xlabel=rf"${layout['horizontal_name']}$ ({unit_label})",
           ylabel=rf"${vertical}$ ({unit_label})",
           xlim=(float(np.min(x)), float(np.max(x))))
    if title is None:
        title = rf"$\arg({info['component']})$"
        if info["reference_name"] == "incident":
            title += "（相对入射参考）"
        elif info["reference_name"] == "custom":
            title += "（相对自定义参考）"
        if info["unwrapped"]:
            title += "（unwrap）"
    ax.set_title(title)
    if colorbar:
        bar = fig.colorbar(image, ax=ax, pad=.018, aspect=18)
        bar.set_label(r"phase (rad)")
        if not info["unwrapped"]:
            bar.set_ticks([-np.pi, 0, np.pi])
            bar.set_ticklabels([r"$-\pi$", "0", r"$\pi$"])
    return fig, ax, image, info
