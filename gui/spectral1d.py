"""Actual sampled FFT coefficients for the 1D material convolution operators.

Analytic integration remains the default in the existing solvers. This module
samples each material function at cell midpoints and performs numpy.fft.fft;
it does not substitute analytic coefficients or average interface cells.
"""
import numpy as np


def validate_fourier_settings(method="analytic",samples=0):
    if method not in ("analytic","fft"):
        raise ValueError("fourier_method must be 'analytic' or 'fft'")
    try:n=int(samples)
    except (ValueError,TypeError,OverflowError):
        raise ValueError("fft_samples must be an integer (0 for automatic)")
    if n!=samples or n<0 or n>4_194_304:
        raise ValueError("fft_samples must be 0..4194304 (0 for automatic)")
    return method,n


def resolve_fft_samples(M,samples=0):
    """Need independent coefficients -2M..2M, hence N strictly greater than 4M."""
    if int(M)!=M or M<0:raise ValueError("M must be a nonnegative integer")
    _,n=validate_fourier_settings("fft",samples)
    if n==0:
        target=max(8192,64*(2*int(M)+1))
        n=1<<(target-1).bit_length()
    if n<4*M+1:
        raise ValueError(f"FFT采样点数 N={n} 不足：M={M} 至少需要 N≥{4*M+1}，建议使用0自动选择")
    if n>4_194_304:raise ValueError("Requested automatic FFT exceeds 4194304 samples")
    return n


def fourier_fft_piecewise(edges,values,period,hmax,samples):
    """Midpoint sampled Fourier coefficients c_h for exp(+i*2*pi*h*x/P).

    FFT normalization and the absolute origin/midpoint phase are both explicit.
    values may be eps, 1/eps, mu or 1/mu, and may be complex.
    """
    edges=np.asarray(edges,float);values=np.asarray(values,complex)
    if edges.ndim!=1 or values.ndim!=1 or len(edges)!=len(values)+1:
        raise ValueError("edges must contain one more entry than values")
    if not np.isfinite(period) or period<=0 or not np.isfinite(edges).all() or not np.isfinite(values).all():
        raise ValueError("Fourier geometry/material values must be finite")
    if np.any(np.diff(edges)<=0) or not np.isclose(edges[-1]-edges[0],period,rtol=1e-10,atol=1e-12*period):
        raise ValueError("edges must increase across exactly one period")
    _,N=validate_fourier_settings("fft",samples)
    if int(hmax)!=hmax or hmax<0 or N<=2*hmax:
        raise ValueError("FFT sample count must exceed 2*hmax")
    # Work in coordinates relative to edges[0] to avoid loss of precision for
    # translated geometries. Avoid sampling a material discontinuity at a grid edge.
    u=(np.arange(N,dtype=float)+.5)/N*period
    idx=np.searchsorted(edges-edges[0],u,side="right")-1
    sampled=values[np.clip(idx,0,len(values)-1)]
    spectrum=np.fft.fft(sampled)/N
    h=np.arange(-int(hmax),int(hmax)+1)
    offset=np.remainder(edges[0],period)/period+.5/N
    return spectrum[np.mod(h,N)]*np.exp(-2j*np.pi*h*offset)


def convmat_fft(edges,values,period,M,samples=0):
    N=resolve_fft_samples(M,samples)
    c=fourier_fft_piecewise(edges,values,period,2*M,N)
    ms=np.arange(-M,M+1)
    return c[np.subtract.outer(ms,ms)+2*M]


def layer_fourier_info(layer,period,M):
    if layer.homogeneous:return {"name":layer.name,"method":"uniform","fft_samples":0}
    method=getattr(layer,"fourier_method","analytic")
    n=resolve_fft_samples(M,getattr(layer,"fft_samples",0)) if method=="fft" else 0
    return {"name":layer.name,"method":method,"fft_samples":n,
            "sample_spacing":period/n if n else None}
