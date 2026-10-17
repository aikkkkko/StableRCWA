"""Geometry expansion for adaptive lateral PML calculations.

The existing physical interval stays at its absolute x coordinates.  A
patterned layer is unfolded over the *old* window before background material
is added outside it, so a translated layer origin cannot introduce another
copy of the object into the enlarged window.
"""
from __future__ import annotations

import math

import numpy as np

import rcwa1d
from .afmm1d import (Background, LateralPML, _background_material,
                     _canonical_layer, _check_pml_background)


def _unfold(layer, x0, period):
    """Return exact segment boundaries and indices over one chosen window."""
    edges = np.asarray(layer.edges, dtype=float)
    values = np.asarray(layer.n_values, dtype=complex)
    if (edges.ndim != 1 or values.ndim != 1 or len(edges) != len(values) + 1
            or not np.isfinite(edges).all() or not np.isfinite(values).all()
            or np.any(np.diff(edges) <= 0)
            or not np.isclose(edges[-1] - edges[0], period, rtol=1e-10,
                                  atol=1e-12 * period)):
        raise ValueError("Invalid patterned layer segments for the old period")
    # The cyclic breakpoints may straddle x0; profile chooses the material
    # between them using the layer's own periodic coordinate convention.
    breaks = x0 + np.mod(edges[:-1] - x0, period)
    cuts = np.unique(np.r_[x0, breaks[(breaks > x0) &
                                      (breaks < x0 + period)], x0 + period])
    mid = (cuts[:-1] + cuts[1:]) / 2
    wrapped = edges[0] + np.mod(mid - edges[0], period)
    indices = np.searchsorted(edges, wrapped, side="right") - 1
    return cuts, values[np.clip(indices, 0, len(values) - 1)]


def expand_pml_window(stack, pml, background, *, left, right, M):
    """Extend PML widths while keeping the old physical structure fixed.

    Returns ``(new_stack, new_pml, new_M, new_background)``.  Only the
    nonmagnetic ``rcwa1d`` geometry is supported by this adaptive helper.
    """
    if not isinstance(stack, rcwa1d.Stack) or any(
            not isinstance(layer, rcwa1d.Layer) for layer in stack.layers):
        raise TypeError("Adaptive PML expansion requires nonmagnetic rcwa1d.Stack and Layer")
    if not isinstance(pml, LateralPML):
        raise TypeError("pml must be LateralPML")
    if int(M) != M or M < 1:
        raise ValueError("M must be an integer >= 1")
    if not np.isfinite([left, right]).all() or left < pml.left or right < pml.right:
        raise ValueError("Expanded PML widths must be finite and at least the old widths")
    P = float(stack.period)
    pml.validate(P)
    entries = [None] * len(stack.layers) if background is None else list(background)
    if len(entries) != len(stack.layers):
        raise ValueError("background length must equal the number of layers")
    if not stack.layers:
        raise ValueError("At least one layer is required")

    canonical = [_canonical_layer(layer) for layer in stack.layers]
    bgs = [_background_material(value, layer)
           for value, layer in zip(entries, canonical)]
    for layer, bg in zip(canonical, bgs):
        if not np.isclose(bg.mu, 1, rtol=0, atol=1e-12):
            raise ValueError("Adaptive PML requires mu=1 in every background")
        if (not np.isfinite([bg.eps.real, bg.eps.imag]).all()
                or abs(bg.eps) < 1e-14):
            raise ValueError("Background epsilon must be finite and nonzero")
        if layer.homogeneous:
            if not np.isclose(layer.mu, 1, rtol=0, atol=1e-12):
                raise ValueError("Adaptive PML requires mu=1 in every layer")
        elif np.any(np.abs(layer.mu_values - 1) > 1e-12):
            raise ValueError("Adaptive PML requires mu=1 in every layer")
        _check_pml_background(layer, bg, pml, P)

    extra_left, extra_right = float(left - pml.left), float(right - pml.right)
    new_x0 = pml.x0 - extra_left
    new_P = P + extra_left + extra_right
    new_M = max(int(M), math.ceil(((2 * M + 1) * new_P / P - 1) / 2))
    quadrature = (0 if pml.quadrature_order == 0 else
                  max(pml.quadrature_order, 4 * new_M + 16))
    new_pml = LateralPML(float(left), float(right), pml.strength, pml.order,
                         new_x0, quadrature)

    new_layers = []
    new_background = []
    for original, layer, bg, entry in zip(stack.layers, canonical, bgs, entries):
        if original.homogeneous:
            new_layers.append(rcwa1d.Layer(original.d, n=original.n,
                                            name=original.name))
            new_background.append(None if entry is None else
                                  complex(np.sqrt(complex(bg.eps))))
            continue
        cuts, vals = _unfold(original, pml.x0, P)
        bg_n = complex(np.sqrt(complex(bg.eps)))
        pieces = []
        if extra_left > 0:
            pieces.append((new_x0, bg_n))
        pieces.extend(zip(cuts[:-1], vals))
        if extra_right > 0:
            pieces.append((pml.x0 + P, bg_n))
        edges = np.array([point for point, _ in pieces] + [new_x0 + new_P],
                         dtype=float)
        n_values = np.array([value for _, value in pieces], dtype=complex)
        fft_samples = original.fft_samples
        if original.fourier_method == "fft" and fft_samples:
            fft_samples = max(math.ceil(fft_samples * new_P / P),
                              4 * new_M + 1)
        new_layers.append(rcwa1d.Layer(original.d, edges=edges,
                                       n_values=n_values, name=original.name,
                                       fourier_method=original.fourier_method,
                                       fft_samples=fft_samples))
        new_background.append(bg_n)
    new_stack = rcwa1d.Stack(stack.nI, new_layers, stack.nII, new_P)
    return new_stack, new_pml, new_M, new_background
