"""Local one-dimensional translation builder for the GUI.

The source unit is stored in metadata. Rebuilding verifies the current
expanded patterned geometry before replacing it, so hand edits are not lost.
"""
from __future__ import annotations

import copy
import math
import numpy as np


_TOL = 1e-10
DEFAULT_OPTIONS = dict(translations=0, intercell_gap=0., trailing_gap=0.,
                       buffer_left=1.053, buffer_right=1.053, gap_n=1.)


def _num(value, label, low=0., high=1e6, integer=False):
    if isinstance(value, bool):
        raise ValueError(f"{label}必须是数字")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{label}必须是数字") from None
    if not math.isfinite(value) or value < low or value > high:
        raise ValueError(f"{label}超出允许范围")
    if integer and value != int(value):
        raise ValueError(f"{label}必须是整数")
    return int(value) if integer else value


def _same(a, b):
    try:
        return abs(complex(a) - complex(b)) <= _TOL * max(1., abs(complex(b)))
    except (TypeError, ValueError, OverflowError):
        return False


def _edge_tol(*values):
    """Only merge cuts indistinguishable at their floating-point scale."""
    return 8 * math.ulp(max(abs(float(v)) for v in values))


def _same_edge(a, b, scale):
    try:
        a, b = float(a), float(b)
        return math.isfinite(a) and math.isfinite(b) and abs(a-b) <= _edge_tol(a, b, scale)
    except (TypeError, ValueError, OverflowError):
        return False


def _options(raw):
    if not isinstance(raw, dict):
        raise ValueError("平移参数必须是对象")
    vals = dict(DEFAULT_OPTIONS)
    vals.update(raw)
    return dict(translations=_num(vals["translations"], "平移次数", high=999, integer=True),
                intercell_gap=_num(vals["intercell_gap"], "单元间隙"),
                trailing_gap=_num(vals["trailing_gap"], "末尾空白"),
                buffer_left=_num(vals["buffer_left"], "左缓冲宽度"),
                buffer_right=_num(vals["buffer_right"], "右缓冲宽度"),
                gap_n=_num(vals["gap_n"], "间隙折射率", low=1e-9, high=1e4))


def _unit(config):
    st = config["structure"]
    period = _num(st["period"], "原单元宽度", low=1e-9)
    layers = copy.deepcopy(st["layers"])
    if not isinstance(layers, list) or not 1 <= len(layers) <= 256:
        raise ValueError("模板层数需为1至256")
    for layer in layers:
        if layer["kind"] == "patterned":
            source_edges = np.asarray(layer["edges"], float)
            if not _same(source_edges[-1] - source_edges[0], period):
                raise ValueError("原单元首尾宽度不一致")
            # Existing layers can have different edge origins. Sample their
            # periodic profiles in the same physical window [0, period].
            cuts = sorted([0., period] + [float(x % period) for x in source_edges])
            unique = []
            for x in cuts:
                if not unique or x - unique[-1] > _edge_tol(x, unique[-1], period):
                    unique.append(x)
            cuts = unique
            cuts[0], cuts[-1] = 0., period
            mids = (np.asarray(cuts[:-1]) + np.asarray(cuts[1:])) / 2
            mapped = source_edges[0] + np.mod(mids - source_edges[0], period)
            indices = np.searchsorted(source_edges, mapped, side="right") - 1
            indices = np.clip(indices, 0, len(layer["n_values"]) - 1)
            source_values = layer["n_values"]
            layer["edges"] = cuts
            layer["n_values"] = [source_values[int(i)] for i in indices]
    return period, layers


def _render(unit_period, unit_layers, opts, mode, left, right):
    copies = opts["translations"] + 1
    physical_width = copies * unit_period + (copies - 1) * opts["intercell_gap"] + opts["trailing_gap"]
    if mode == "pml":
        x0 = -opts["buffer_left"] - left
        x1 = physical_width + opts["buffer_right"] + right
    else:
        x0, x1 = 0., physical_width
    width = x1 - x0
    if not math.isfinite(width) or width > 1e6:
        raise ValueError("横向计算宽度超出范围")
    expanded = []
    for source in unit_layers:
        if source["kind"] != "patterned":
            expanded.append(None)
            continue
        gap_n = source.get("background_n")
        if gap_n is None:
            gap_n = opts["gap_n"]
        edges, values = [x0], []
        if mode == "pml" and x0 < 0:
            edges.append(0.)
            values.append(gap_n)
        base_edges = source["edges"]
        base_values = source["n_values"]
        for j in range(copies):
            offset = j * (unit_period + opts["intercell_gap"])
            if offset > edges[-1] + _edge_tol(offset, edges[-1], unit_period):
                edges.append(offset)
                values.append(gap_n)
            edges.extend([offset + e for e in base_edges[1:-1]] +
                         [offset + unit_period if opts["intercell_gap"] else (j + 1) * unit_period])
            values.extend(base_values)
        if physical_width > edges[-1]:
            edges.append(physical_width)
            values.append(gap_n)
        if x1 > edges[-1]:
            edges.append(x1)
            values.append(gap_n)
        if len(edges) > 2001:
            raise ValueError("平移后分段边界数超过GUI上限2001")
        if not all(b > a for a, b in zip(edges, edges[1:])):
            raise ValueError("平移后边界必须严格递增")
        expanded.append((edges, values, gap_n))
    return expanded, width, x0


def _marker(unit_period, unit_layers, opts, mode, left, right):
    _, width, x0 = _render(unit_period, unit_layers, opts, mode, left, right)
    return dict(kind="translated_1d", unit_period=unit_period,
                unit_layers=copy.deepcopy(unit_layers), options=copy.deepcopy(opts),
                built_mode=mode, built_left=left if mode == "pml" else 0.,
                built_right=right if mode == "pml" else 0.,
                built_x0=x0, built_period=width)


def _verified(config, check_active):
    meta = config.get("repeat_geometry")
    if not isinstance(meta, dict) or meta.get("kind") != "translated_1d":
        return None
    try:
        unit_period = _num(meta["unit_period"], "原单元宽度", low=1e-9)
        units = meta["unit_layers"]
        layers = config["structure"]["layers"]
        if not isinstance(units, list) or len(units) != len(layers) or not 1 <= len(units) <= 256:
            return None
        opts = _options(meta["options"])
        mode = meta["built_mode"]
        if mode not in ("periodic", "conical", "pml"):
            return None
        left = _num(meta["built_left"], "已建左PML", low=1e-9 if mode == "pml" else 0.)
        right = _num(meta["built_right"], "已建右PML", low=1e-9 if mode == "pml" else 0.)
        if mode != "pml" and (left or right):
            return None
        expanded, width, x0 = _render(unit_period, units, opts, mode, left, right)
        if not (_same_edge(meta["built_period"], width, width) and
                _same_edge(meta["built_x0"], x0, width) and
                _same_edge(config["structure"]["period"], width, width)):
            return None
        for actual, unit, expected in zip(layers, units, expanded):
            if actual["kind"] != unit["kind"]:
                return None
            if expected is None:
                continue  # Uniform films remain editable.
            edges, values, bg = expected
            if len(actual["edges"]) != len(edges) or len(actual["n_values"]) != len(values):
                return None
            if any(not _same_edge(a, b, width) for a, b in zip(actual["edges"], edges)):
                return None
            if any(not _same(a, b) for a, b in zip(actual["n_values"], values)):
                return None
            actual_bg = actual.get("background_n")
            if mode == "pml" and not _same(actual_bg, bg):
                return None
            if mode != "pml":
                unit_bg = unit.get("background_n")
                if not ((actual_bg is None and unit_bg is None) or _same(actual_bg, unit_bg)):
                    return None
        if check_active:
            if config["mode"] != mode:
                return None
            if mode == "pml":
                p = config["pml"]
                if not (_same(p["left"], left) and _same(p["right"], right)
                        and _same(p["x0"], x0)):
                    return None
        return _marker(unit_period, units, opts, mode, left, right)
    except (KeyError, IndexError, TypeError, ValueError, OverflowError):
        return None


def validated_repeat(config):
    """Return canonical metadata while its expanded geometry is still real."""
    return _verified(config, True)


def initialize_repeat(config):
    """Capture a periodic 1D structure as one editable translation unit."""
    from gui_models import normalize

    cfg = normalize(config)
    if cfg["mode"] not in ("periodic", "conical"):
        raise ValueError("请先选择一维周期或圆锥模式作为原单元")
    period, units = _unit(cfg)
    opts = _options(DEFAULT_OPTIONS)
    expanded, width, x0 = _render(period, units, opts, cfg["mode"], 0., 0.)
    for layer, shape in zip(cfg["structure"]["layers"], expanded):
        if shape is not None:
            layer["edges"], layer["n_values"] = shape[:2]
    cfg["structure"]["period"] = width
    cfg["pml"]["x0"] = x0
    cfg.pop("profile_geometry", None)
    cfg["repeat_geometry"] = _marker(period, units, opts, cfg["mode"], 0., 0.)
    return normalize(cfg)


def rebuild_repeat(config, options):
    """Regenerate every patterned layer from the verified original unit."""
    old = _verified(config, False) if isinstance(config, dict) else None
    if old is None:
        raise ValueError("平移模板与当前分段几何不符，请重新初始化模板")
    cfg = copy.deepcopy(config)
    opts = _options(options)
    mode = cfg.get("mode")
    if mode not in ("periodic", "conical", "pml"):
        raise ValueError("求解方式应为 periodic、conical 或 pml")
    if mode == "pml":
        p = cfg.get("pml")
        if not isinstance(p, dict):
            raise ValueError("PML参数格式不正确")
        left = _num(p.get("left"), "左PML厚度", low=1e-9)
        right = _num(p.get("right"), "右PML厚度", low=1e-9)
    else:
        left = right = 0.
    period, units = old["unit_period"], old["unit_layers"]
    expanded, width, x0 = _render(period, units, opts, mode, left, right)
    for j, (layer, shape) in enumerate(zip(cfg["structure"]["layers"], expanded)):
        if shape is None:
            continue
        layer["edges"], layer["n_values"] = shape[:2]
        layer["background_n"] = shape[2] if mode == "pml" else units[j].get("background_n")
    cfg["structure"]["period"] = width
    cfg["pml"]["x0"] = x0
    cfg["repeat_geometry"] = _marker(period, units, opts, mode, left, right)
    from gui_models import normalize
    return normalize(cfg)
