"""Shared layer propagation for scalar modes, exact zero roots and Schur blocks.

The two input amplitudes are anchored at opposite layer faces.  Return their
field sum and signed difference, suitable for ``W @ sum`` and ``V @ difference``.
Neither zero-root auxiliary amplitudes nor Schur columns are individual waves.
"""
import numpy as np
from scipy import linalg
from scipy.sparse.linalg import expm_multiply


def outgoing_triangular_root(t, branch):
    """Square root with the solver's outgoing diagonal branches; no root clipping."""
    t = np.asarray(t, dtype=complex)
    if t.ndim != 2 or t.shape[0] != t.shape[1] or not np.isfinite(t).all():
        raise np.linalg.LinAlgError("Schur matrix must be finite and square")
    n = t.shape[0]
    g = np.zeros_like(t)
    diagonal = branch(np.sqrt(np.diag(t)))
    if np.any(diagonal == 0) or not np.isfinite(diagonal).all():
        raise np.linalg.LinAlgError("Schur propagation has an exact zero propagation constant")
    np.fill_diagonal(g, diagonal)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for gap in range(1, n):
            for i in range(n - gap):
                j = i + gap
                denominator = g[i, i] + g[j, j]
                if denominator == 0:
                    raise np.linalg.LinAlgError("Incompatible Schur square-root branches")
                g[i, j] = (t[i, j] - g[i, i + 1:j] @ g[i + 1:j, j]) / denominator
        scale = max(float(linalg.norm(t, 1)), np.finfo(float).tiny) if n else 1.
        error = float(linalg.norm(g @ g - t, 1)) if n else 0.
    if not np.isfinite(g).all() or (not np.isfinite(scale) or not np.isfinite(error)) or error > 1e-10 * scale:
        raise np.linalg.LinAlgError("Schur square-root residual is too large")
    return g


def matrix_propagator(generator, distance, k0):
    """Finite, full matrix exp(-k0 * distance * generator)."""
    g = np.asarray(generator, dtype=complex)
    if g.ndim != 2 or g.shape[0] != g.shape[1] or not np.isfinite(g).all():
        raise np.linalg.LinAlgError("Propagation generator must be finite and square")
    with np.errstate(over="raise", invalid="raise"):
        argument = (-k0 * distance) * g
        if not np.isfinite(argument).all():
            raise np.linalg.LinAlgError("Nonfinite propagation argument")
        result = linalg.expm(argument)
    if not np.isfinite(result).all():
        raise np.linalg.LinAlgError("Nonfinite matrix exponential")
    return result


def field_combinations(q, a_plus, a_minus, z_local, d, k0, *,
                       zero_basis=False, generator=None):
    """Return two (mode, depth) arrays, using layer-top/layer-bottom amplitudes.

    For an exact zero root, h=k0*d and xi=k0*z give the exact coefficients
    t*((1+h-xi)*a_plus+(1+xi)*a_minus), t*(-a_plus+a_minus), t=2/(2+h).
    Distances from each face are evaluated separately to avoid cancellation.
    """
    q = np.asarray(q, dtype=complex)
    ap, am = np.asarray(a_plus, complex), np.asarray(a_minus, complex)
    z = np.atleast_1d(np.asarray(z_local, float))
    if q.ndim != 1 or ap.shape != q.shape or am.shape != q.shape or z.ndim != 1:
        raise ValueError("Incompatible mode, amplitude or depth shapes")
    if not np.isfinite(z).all() or not np.isfinite(d) or d < 0 or not np.isfinite(k0) or k0 <= 0:
        raise ValueError("Invalid layer depth or wave number")
    if np.any(z < -1e-12) or np.any(z > d + 1e-12):
        raise ValueError("Depth lies outside the layer")
    z = np.clip(z, 0., d)  # same interface tolerance as the public field routines
    if generator is not None:
        if np.shape(generator) != (len(q), len(q)):
            raise ValueError("Generator and modes have different sizes")
        # Apply the exponential to amplitudes, not to a cached dense matrix for
        # every depth. Memory stays O(N * nz), even for high-order field maps.
        g = np.asarray(generator, complex)
        if not np.isfinite(g).all():
            raise np.linalg.LinAlgError("Nonfinite propagation generator")
        operator = -g
        trace = np.trace(operator)

        def action(amplitude, distances):
            with np.errstate(over="raise", invalid="raise"):
                times = k0 * distances
            if not np.isfinite(times).all():
                raise np.linalg.LinAlgError("Nonfinite propagation distance")
            if len(times) == 0:
                return np.empty((len(q), 0), complex)
            unique, inverse = np.unique(times, return_inverse=True)
            # Use the interval API only when it reproduces the requested depths
            # exactly; arbitrary grids and duplicate/interface points also work.
            if len(unique) > 1 and np.array_equal(unique, np.linspace(unique[0], unique[-1], len(unique))):
                values = expm_multiply(operator, amplitude, start=unique[0], stop=unique[-1],
                                       num=len(unique), endpoint=True, traceA=trace)
            else:
                values = np.stack([amplitude if t == 0 else expm_multiply(
                    operator*t, amplitude, traceA=trace*t) for t in unique])
            if not np.isfinite(values).all():
                raise np.linalg.LinAlgError("Nonfinite propagated field")
            return values[inverse].T

        plus = action(ap, z)
        minus = action(am, d - z)
        return plus + minus, -plus + minus

    plus = np.exp(-k0 * q[:, None] * z) * ap[:, None]
    minus = np.exp(-k0 * q[:, None] * (d - z)) * am[:, None]
    summed, difference = plus + minus, -plus + minus
    zero = q == 0 if zero_basis else np.zeros(q.shape, bool)
    if np.any(zero):
        h = k0 * d
        if not np.isfinite(h):
            raise np.linalg.LinAlgError("Nonfinite zero-root layer thickness")
        denominator = 1. + .5 * h
        left = (1. + k0 * (d - z)) / denominator
        right = (1. + k0 * z) / denominator
        summed[zero] = ap[zero, None] * left + am[zero, None] * right
        difference[zero] = ((-ap[zero] + am[zero]) / denominator)[:, None]
    return summed, difference
