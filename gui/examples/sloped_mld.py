"""One-period sloped MLD ridge on a 20-layer SiO2/HfO2 high-reflection stack.

Lengths are in micrometres.  The ridge is represented by equal-height,
midpoint-width slices; its top is z=0 and its base is z=DEPTH.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rcwa1d as rc

LAM0 = 1.053
THETA = 66.36
PERIOD = .574
DEPTH = .380
BOTTOM_WIDTH = .172
TRAPEZOID_TOP_WIDTH = .1032
RIDGE_INDEX = 1.46
GROOVE_INDEX = 1.0


def _geometry(shape, slices, top_width, bottom_width, period, depth, center):
    if shape not in ("trapezoid", "triangle"):
        raise ValueError("shape must be 'trapezoid' or 'triangle'")
    if isinstance(slices, bool) or not isinstance(slices, int) or slices < 1:
        raise ValueError("slices must be a positive integer")
    if top_width is None:
        top_width = TRAPEZOID_TOP_WIDTH if shape == "trapezoid" else 0.
    if center is None:
        center = period / 2
    values = (top_width, bottom_width, period, depth, center)
    try:
        values = tuple(float(v) for v in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("geometry values must be finite numbers") from exc
    if not all(math.isfinite(v) for v in values):
        raise ValueError("geometry values must be finite numbers")
    top_width, bottom_width, period, depth, center = values
    if not (period > 0 and depth > 0 and 0 <= top_width <= bottom_width < period):
        raise ValueError("require period, depth > 0 and 0 <= top_width <= bottom_width < period")
    if shape == "triangle" and top_width != 0:
        raise ValueError("triangle requires top_width=0")
    if bottom_width <= 0:
        raise ValueError("ridge width must be positive")
    if not (bottom_width / 2 < center < period - bottom_width / 2):
        raise ValueError("ridge base must lie strictly inside the period")
    return top_width, bottom_width, period, depth, center


def profile_metadata(shape="trapezoid", slices=64, *, top_width=None,
                     bottom_width=BOTTOM_WIDTH, period=PERIOD, depth=DEPTH,
                     center=None):
    """Return the exact continuous profile behind the midpoint slices."""
    top, bottom, period, depth, center = _geometry(
        shape, slices, top_width, bottom_width, period, depth, center)
    return dict(kind="linear_ridge", shape=shape, period=period, depth=depth,
                top_width=top, bottom_width=bottom, center=center, slices=slices,
                ridge_index=RIDGE_INDEX, groove_index=GROOVE_INDEX)


def build_stack(shape="trapezoid", slices=64, fourier_method="analytic",
                fft_samples=0, *, top_width=None, bottom_width=BOTTOM_WIDTH,
                period=PERIOD, depth=DEPTH, center=None):
    """Build one sloped ridge followed by the 20 uniform films of the stack."""
    top, bottom, period, depth, center = _geometry(
        shape, slices, top_width, bottom_width, period, depth, center)
    dz = depth / slices
    layers = []
    for j in range(slices):
        z_mid = (j + .5) * dz
        width = top + (bottom - top) * z_mid / depth
        left, right = center - width / 2, center + width / 2
        layers.append(rc.Layer(
            dz, edges=[0., left, right, period],
            n_values=[GROOVE_INDEX, RIDGE_INDEX, GROOVE_INDEX],
            name=f"渐变刻线 · {j + 1}/{slices}",
            fourier_method=fourier_method, fft_samples=fft_samples))

    layers.extend([
        rc.Layer(.46438, n=1.46, name="SiO₂ 匹配层 3L"),
        rc.Layer(.115306, n=1.96, name="HfO₂ H"),
    ])
    for j in range(9):
        layers.extend([
            rc.Layer(.30958, n=1.46, name=f"SiO₂ 2L · {j + 1}"),
            rc.Layer(.115306, n=1.96, name=f"HfO₂ H · {j + 1}"),
        ])
    return rc.Stack(1., layers, 1.46, period)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=("trapezoid", "triangle"), default="trapezoid")
    parser.add_argument("--slices", type=int, default=64)
    parser.add_argument("--fourier", choices=("analytic", "fft"), default="analytic")
    parser.add_argument("--fft-samples", type=int, default=0)
    parser.add_argument("--M", type=int, default=30)
    parser.add_argument("--pol", choices=("TE", "TM"), default="TE")
    parser.add_argument("--output", type=Path, help="Save GUI model JSON (defaults to stdout)")
    args = parser.parse_args(argv)
    stack = build_stack(args.shape, args.slices, args.fourier, args.fft_samples)
    # Keep the model module importable without a GUI-model import cycle.
    from gui_models import from_stack
    cfg = from_stack(stack, name=f"单周期 MLD {args.shape} 刻线",
                     lam0=LAM0, theta_deg=THETA, M=args.M, pol=args.pol)
    cfg["profile_geometry"] = profile_metadata(args.shape, args.slices)
    payload = json.dumps(cfg, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return cfg


if __name__ == "__main__":
    main()
