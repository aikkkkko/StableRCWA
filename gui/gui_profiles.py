"""Validate and reslice the continuous profiles attached to GUI configurations."""
from __future__ import annotations

import copy
import math

from examples.sloped_mld import build_stack, profile_metadata


_TOL = 1e-11


def _number(value):
    if isinstance(value, bool):
        raise ValueError("boolean is not a dimension")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite dimension")
    return number


def _same(a, b):
    return math.isclose(a, b, rel_tol=_TOL, abs_tol=_TOL)


def validated_profile(cfg):
    """Return canonical metadata only while the first slices still match it."""
    try:
        meta = cfg.get("profile_geometry")
        if not isinstance(meta, dict) or meta.get("kind") != "linear_ridge":
            return None
        shape = meta["shape"]
        if shape not in ("trapezoid", "triangle"):
            return None
        ns = meta["slices"]
        if isinstance(ns, bool) or not isinstance(ns, int) or ns < 1:
            return None
        period = _number(meta["period"])
        depth = _number(meta["depth"])
        top = _number(meta["top_width"])
        bottom = _number(meta["bottom_width"])
        center = _number(meta["center"])
        ridge = _number(meta["ridge_index"])
        groove = _number(meta["groove_index"])
        if not (period > 0 and depth > 0 and 0 <= top <= bottom < period
                and bottom > 0 and groove > 0 and ridge > 0):
            return None
        if shape == "triangle" and top != 0:
            return None
        if not (bottom / 2 < center < period - bottom / 2):
            return None
        structure = cfg["structure"]
        if not _same(period, _number(structure["period"])):
            return None
        layers = structure["layers"]
        if not isinstance(layers, list) or len(layers) < ns:
            return None
        for j, layer in enumerate(layers[:ns]):
            if layer["kind"] != "patterned":
                return None
            if not _same(_number(layer["thickness"]), depth / ns):
                return None
            z = (j + .5) * depth / ns
            width = top + (bottom - top) * z / depth
            expected = (0., center - width / 2, center + width / 2, period)
            edges = layer["edges"]
            if len(edges) != 4 or any(not _same(_number(a), b)
                                          for a, b in zip(edges, expected)):
                return None
            actual_n = layer["n_values"]
            if len(actual_n) != 3 or any(not _same(_number(a), b)
                                         for a, b in zip(actual_n,
                                                         (groove, ridge, groove))):
                return None
        return dict(kind="linear_ridge", shape=shape, period=period,
                    depth=depth, top_width=top, bottom_width=bottom,
                    center=center, slices=ns, ridge_index=ridge,
                    groove_index=groove)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def rebuild_profile(config, slices):
    """Reslice a validated ridge, preserving the film tail and GUI settings."""
    from gui_models import from_stack, normalize

    if isinstance(slices, bool) or not isinstance(slices, int):
        raise ValueError("slices must be an integer")
    # Keep the original metadata for validation even if normalize removes a
    # stale marker as part of its GUI import handling.
    original_meta = copy.deepcopy(config.get("profile_geometry"))
    cfg = normalize(config)
    check_cfg = dict(cfg, profile_geometry=original_meta)
    meta = validated_profile(check_cfg)
    if meta is None:
        raise ValueError("profile geometry no longer matches the ridge slices")
    if cfg["mode"] == "pml":
        raise ValueError("profile reslicing is not supported for PML models")
    if cfg["mode"] not in ("periodic", "conical"):
        raise ValueError("profile reslicing requires a one-dimensional periodic model")
    old_ns = meta["slices"]
    old_layers = cfg["structure"]["layers"]
    films = old_layers[old_ns:]
    maximum = min(236, 256 - len(films))
    if not 1 <= slices <= maximum:
        raise ValueError(f"slices must be between 1 and {maximum}")
    if not (_same(meta["ridge_index"], 1.46)
            and _same(meta["groove_index"], 1.0)):
        raise ValueError("this ridge builder supports indices 1.46 and 1.0")
    stack = build_stack(meta["shape"], slices, top_width=meta["top_width"],
                        bottom_width=meta["bottom_width"], period=meta["period"],
                        depth=meta["depth"], center=meta["center"])
    generated = from_stack(stack, M=cfg["incident"]["M"])["structure"]["layers"][:slices]
    for j, layer in enumerate(generated):
        old = old_layers[min(old_ns - 1, (j * 2 + 1) * old_ns // (2 * slices))]
        layer["fourier_method"] = old["fourier_method"]
        layer["fft_samples"] = old["fft_samples"]
        layer["background_n"] = old["background_n"]
    cfg["structure"]["layers"] = generated + films
    cfg["profile_geometry"] = profile_metadata(
        meta["shape"], slices, top_width=meta["top_width"],
        bottom_width=meta["bottom_width"], period=meta["period"],
        depth=meta["depth"], center=meta["center"])
    result = normalize(cfg)
    # Retain canonical metadata for callers even before normalize integrates
    # its marker verification.
    result["profile_geometry"] = cfg["profile_geometry"]
    return result
