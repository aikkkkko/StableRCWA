"""Aperiodic 1D RCWA with lateral coordinate-stretched PML and contrast fields.

Geometry is two-dimensional (x,z), invariant along y.  The Fourier basis and
exp(-i omega t) convention are those of rcwa1d.py.  Only the contrast field is
periodized; a stratified, laterally uniform background supplies the incident
plane wave.  The source must vanish in both lateral PMLs.

PML: s(x)=1+i*strength*(distance/width)**order, F=C[1/s], D=F Kx.
For TE, chi=mu, eta=eps; for TM, chi=eps, eta=mu.  The modal state obeys
    dS/dz = k0 B U + k0 g1 U_bg,
    dU/dz = k0 C S + k0 g2 S_bg,
    B = C[1/chi]^-1, C = D C[chi]^-1 D - C[eta].
Here S,U are CONTRAST fields and the forcing is confined to material contrast.
The physical transverse basis is exp(-i*kxm*x), so positive theta means a
negative physical x component of the incident wave vector.

Sources are integrated analytically in the forward/backward modal basis.
The resulting affine scattering blocks reuse the original stable interface
and propagation routines; no transfer matrix or resonant particular-solution
inverse is used.  PML port eigenmodes are not free-space diffraction orders.

References and the derivation are in the author's earlier Python package (not included in this repository).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import rcwa1d as legacy
from mag import rcwa1d_mag as materials
from spectral1d import layer_fourier_info


@dataclass(frozen=True)
class Background:
    """Relative epsilon and mu of one laterally uniform background layer."""
    eps: complex
    mu: complex = 1.0


@dataclass(frozen=True)
class LateralPML:
    """PMLs inside [x0,x0+period]. All lengths use the Stack's length unit.

    strength is dimensionless Im(s) at the outside edge, not a material loss.
    order=3 gives zero first and second derivatives at the physical interface.
    strength=0 is provided for periodic-limit regression only.
    """
    left: float
    right: float
    strength: float = 3.0
    order: int = 3
    x0: float = 0.0
    quadrature_order: int = 0

    def __post_init__(self):
        if not np.isfinite([self.left, self.right, self.strength, self.x0]).all():
            raise ValueError("PML parameters must be finite")
        if self.left <= 0 or self.right <= 0 or self.strength < 0:
            raise ValueError("PML widths must be positive; strength must be nonnegative")
        if int(self.order) != self.order or self.order < 2:
            raise ValueError("PML polynomial order must be an integer >= 2")
        if int(self.quadrature_order) != self.quadrature_order or self.quadrature_order < 0:
            raise ValueError("quadrature_order must be a nonnegative integer")

    def validate(self, period):
        if not np.isfinite(period) or period <= self.left + self.right:
            raise ValueError("period must exceed the sum of both PML widths")

    def physical_mask(self, x, period):
        self.validate(period)
        x = np.asarray(x, float)
        return ((x >= self.x0 + self.left) &
                (x <= self.x0 + period - self.right))

    def stretch(self, x, period):
        self.validate(period)
        u = np.asarray(x, float) - self.x0
        dl = np.clip((self.left - u) / self.left, 0, 1)
        dr = np.clip((u - period + self.right) / self.right, 0, 1)
        return 1.0 + 1j * self.strength * (dl**self.order + dr**self.order)

    def operator(self, period, M):
        """Fourier convolution of 1/s using interface-split Gaussian quadrature.

        Only the smooth PML correction is integrated numerically here. Material
        coefficients are constructed separately using each Layer's method.
        """
        self.validate(period)
        N = 2 * M + 1
        if self.strength == 0:
            return np.eye(N, dtype=complex)
        nq = int(self.quadrature_order or max(80, 4 * M + 16))
        if nq < 2 * M + 1:
            raise ValueError("quadrature_order is too small for this Fourier truncation")
        t, w = np.polynomial.legendre.leggauss(nq)
        hs = np.arange(-2 * M, 2 * M + 1)
        coeff = np.zeros(len(hs), complex)
        coeff[2 * M] = 1.0
        for a, b in ((self.x0, self.x0 + self.left),
                     (self.x0 + period - self.right, self.x0 + period)):
            xx = a + (t + 1) * (b - a) / 2
            ww = w * (b - a) / (2 * period)
            correction = 1.0 / self.stretch(xx, period) - 1.0
            coeff += np.exp(-2j * np.pi * np.outer(hs, xx) / period) @ (ww * correction)
        ms = np.arange(-M, M + 1)
        return coeff[np.subtract.outer(ms, ms) + 2 * M]


class PMLResult(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _canonical_layer(layer):
    if isinstance(layer, materials.Layer):
        return layer
    if getattr(layer, "anisotropic", False):
        raise NotImplementedError("横向PML暂不支持各向异性材料；主轴沿坐标轴的张量目前只用于一维周期 TE/TM")
    if layer.homogeneous:
        return materials.Layer(layer.d, n=layer.n, name=layer.name)
    return materials.Layer(layer.d, edges=layer.edges, n_values=layer.n_values,
                           name=layer.name,
                           fourier_method=getattr(layer,"fourier_method","analytic"),
                           fft_samples=getattr(layer,"fft_samples",0))


def _background_material(value, layer):
    if value is None:
        if not layer.homogeneous:
            raise ValueError("Every patterned layer needs a uniform background index or Background(eps,mu)")
        return Background(layer.eps, layer.mu)
    if isinstance(value, Background):
        return value
    n = complex(value)
    return Background(n * n, 1.0)


def _validate_material(eps, mu):
    vals = np.asarray([eps, mu], complex)
    if not np.isfinite(vals).all() or np.any(np.abs(vals) < 1e-14):
        raise ValueError("epsilon/mu must be finite and nonzero")
    if eps.imag < -1e-12 or mu.imag < -1e-12 or mu.real <= 0:
        raise ValueError("This PML branch supports passive materials with positive Re(mu)")


def _check_pml_background(layer, bg, pml, period):
    """Check every material segment, not merely a few sampled boundary points."""
    if layer.homogeneous:
        cuts = [pml.x0, pml.x0 + period]
    else:
        cuts = np.r_[pml.x0, pml.x0 + period,
                     pml.x0 + np.mod(layer.edges - pml.x0, period)]
    cuts = np.unique(np.r_[cuts, pml.x0 + pml.left, pml.x0 + period - pml.right])
    xx = (cuts[1:] + cuts[:-1]) / 2
    ep, mu = layer.profile(xx, period)
    in_pml = ~pml.physical_mask(xx, period)
    if (np.any(np.abs(ep[in_pml] - bg.eps) > 1e-10 * max(1, abs(bg.eps))) or
            np.any(np.abs(mu[in_pml] - bg.mu) > 1e-10 * max(1, abs(bg.mu)))):
        raise ValueError("Material contrast reaches a lateral PML; leave uniform background on both sides")


def _material_ops(layer, period, M):
    I = np.eye(2 * M + 1, dtype=complex)
    if layer.homogeneous:
        return layer.eps * I, I / layer.eps, layer.mu * I, I / layer.mu
    return layer.operators(period, M)


def _mode(layer, period, M, D, Kx, pol, unstretched=False):
    E, Ae, Mu, Am = _material_ops(layer, period, M)
    chi, achi, eta = (Mu, Am, E) if pol == "TE" else (E, Ae, Mu)
    I = np.eye(2 * M + 1, dtype=complex)
    chi_inv = np.linalg.solve(chi, I)
    B = np.linalg.solve(achi, I)
    C = D @ chi_inv @ D - eta
    A = B @ C
    if unstretched and layer.homogeneous:
        W, V, q = materials._homog_modes(layer.eps, layer.mu, np.diag(Kx), pol)
    else:
        ev, W = np.linalg.eig(A)
        q = legacy._branch(np.sqrt(ev.astype(complex)))
        V = achi @ (W * q[None, :])
    residual = np.linalg.norm(A @ W - W * q[None, :]**2) / max(np.linalg.norm(A) * np.linalg.norm(W), 1e-300)
    return dict(W=W, V=V, q=q, B=B, C=C, eta=eta, chi_inv=chi_inv,
                achi=achi, layer=layer, modal_residual=float(residual))


def _same_integral(q, qb, length, k0):
    """k0 integral exp(-k0*q*(L-t))*exp(-k0*qb*t) dt; finite at q=qb."""
    q = np.asarray(q, complex)
    out = np.empty_like(q)
    delta = q - qb
    close = np.abs(delta * k0 * length) < 1e-5
    y = delta[close] * k0 * length
    # expm1(y)/y through third order gives <1e-22 error in this branch.
    out[close] = k0 * length * np.exp(-k0 * q[close] * length) * (1 + y/2 + y*y/6 + y*y*y/24)
    out[~close] = (np.exp(-k0 * qb * length) - np.exp(-k0 * q[~close] * length)) / delta[~close]
    return out


def _opposite_integral(q, qb, length, k0):
    """k0 integral exp(-k0*(q+qb)*t) dt; finite at q=-qb."""
    v = np.asarray(q, complex) + qb
    out = np.empty_like(v)
    close = np.abs(v * k0 * length) < 1e-5
    y = v[close] * k0 * length
    out[close] = k0 * length * (1 - y/2 + y*y/6 - y*y*y/24)
    out[~close] = -np.expm1(-k0 * v[~close] * length) / v[~close]
    return out


def _source_data(mode, bg, bg_amp, D, kappa, d0, pol):
    chi_b, eta_b = (bg.mu, bg.eps) if pol == "TE" else (bg.eps, bg.mu)
    g1 = mode["B"] @ d0 - chi_b * d0
    g2 = kappa * D @ (mode["chi_inv"] @ d0 - d0 / chi_b) - (mode["eta"] @ d0 - eta_b * d0)
    # Exact zero for a uniform background avoids roundoff source amplification.
    L = mode["layer"]
    if L.homogeneous and abs(L.eps-bg.eps) < 1e-13 and abs(L.mu-bg.mu) < 1e-13:
        g1[:] = 0; g2[:] = 0
    v1 = np.linalg.solve(mode["W"], g1)
    v2 = np.linalg.solve(mode["V"], g2)
    qb = complex(bg_amp["q"][0])
    t = qb / chi_b
    return dict(qb=qb, ap=complex(bg_amp["a_plus"][0]),
                am=complex(bg_amp["a_minus"][0]),
                dp=(-t*v1-v2)/2, dm=(t*v1-v2)/2,
                up=(-t*v1+v2)/2, um=(t*v1+v2)/2)


def _source_integrals(mode, src, d, k0, z):
    """Contributions to d(z) anchored at top and u(z) anchored at bottom."""
    q, qb = mode["q"], src["qb"]
    jd = (src["ap"] * src["dp"] * _same_integral(q, qb, z, k0) +
          src["am"] * np.exp(-k0*qb*(d-z)) * src["dm"] * _opposite_integral(q, qb, z, k0))
    ju = -(src["ap"] * np.exp(-k0*qb*z) * src["up"] * _opposite_integral(q, qb, d-z, k0) +
           src["am"] * src["um"] * _same_integral(q, qb, d-z, k0))
    return jd, ju


def _interface(a, b):
    if a is b:
        one = np.ones(len(a["q"]), complex); zero = np.zeros_like(one)
        return legacy.S(one, zero, zero, one, diag=True)
    return legacy.s_interface(a["W"], a["V"], b["W"], b["V"])


def _affine_cascade(blocks, N):
    """Compose outgoing-wave relations plus source vectors; no growing exponentials."""
    count = len(blocks)
    rs = [None] * (count + 1)
    bs = [None] * (count + 1)
    rs[-1] = np.zeros((N, N), complex)
    bs[-1] = np.zeros(N, complex)
    I = np.eye(N, dtype=complex)
    for k in range(count-1, -1, -1):
        S, jd, ju = blocks[k]
        R, b = rs[k+1], bs[k+1]
        if S.diag:
            # Propagation blocks have no reflection.
            if np.any(S.Rdu) or np.any(S.Rud):
                raise ValueError("Unexpected diagonal reflective block")
            rs[k] = S.Tuu[:, None] * R * S.Tdd[None, :]
            bs[k] = S.Tuu * (R @ jd + b) + ju
        else:
            rhs0 = S.Rdu @ b + jd
            solved = np.linalg.solve(I - S.Rdu @ R, np.column_stack((S.Tdd, rhs0)))
            rs[k] = S.Rud + S.Tuu @ R @ solved[:, :N]
            bs[k] = S.Tuu @ (R @ solved[:, N] + b) + ju
    down = [np.zeros(N, complex)]
    for k, (S, jd, ju) in enumerate(blocks):
        if S.diag:
            dnext = S.Tdd * down[-1] + jd
        else:
            dnext = np.linalg.solve(I-S.Rdu @ rs[k+1],
                                   S.Tdd @ down[-1]+S.Rdu @ bs[k+1]+jd)
        down.append(dnext)
    up = [r @ d + b for r, d, b in zip(rs, down, bs)]
    return down, up


def solve_pml(stack, lam0, theta_deg, M, pol="TE", *, pml, background=None,
              need_fields=True):
    """Solve an isolated finite pattern against a stratified uniform background.

    stack: original rcwa1d.Stack or mag.rcwa1d_mag.Stack.
    background: one entry per layer, either refractive index n or
      Background(eps,mu). A None entry infers a homogeneous physical layer.
      Every patterned layer requires an explicit entry.
    pml: LateralPML with widths in the same units as wavelength/geometry.

    No DEr/DEt are returned: finite objects have continuous radiation spectra,
    and PML eigenmodes cannot be interpreted as free-space diffraction orders.
    field_map masks the artificial PML in physical total-field outputs.
    """
    pol = pol.upper()
    if pol not in ("TE", "TM"):
        raise ValueError("pol must be TE or TM (no conical/2D-periodic PML in this version)")
    if not np.isfinite([lam0, theta_deg]).all() or lam0 <= 0 or abs(theta_deg) >= 90:
        raise ValueError("Require lam0>0 and finite |theta_deg|<90")
    if int(M) != M or M < 1:
        raise ValueError("M must be an integer >=1")
    if not isinstance(pml, LateralPML):
        raise TypeError("pml must be LateralPML")
    M = int(M); P = float(stack.period); pml.validate(P)
    layers = [_canonical_layer(L) for L in stack.layers]
    if not layers or any(not np.isfinite(L.d) or L.d <= 0 for L in layers):
        raise ValueError("At least one positive-thickness layer is required")
    values = [None]*len(layers) if background is None else list(background)
    if len(values) != len(layers):
        raise ValueError("background length must equal the number of physical layers")
    bgs = [_background_material(v, L) for v, L in zip(values, layers)]
    for L, bg in zip(layers, bgs):
        _validate_material(complex(bg.eps), complex(bg.mu))
        if L.homogeneous:
            _validate_material(L.eps, L.mu)
        else:
            for ep, mu in zip(L.eps_values, L.mu_values):
                _validate_material(ep, mu)
        _check_pml_background(L, bg, pml, P)
    if any(isinstance(v, legacy.DiagonalEps) for v in (stack.nI, stack.nII)):
        raise NotImplementedError("横向PML的入射区与出射区须为各向同性介质")
    nI, nII = complex(stack.nI), complex(stack.nII)
    eI = complex(getattr(stack, "epsI", nI*nI))
    eII = complex(getattr(stack, "epsII", nII*nII))
    mI = complex(getattr(stack, "muI", 1.0)); mII = complex(getattr(stack, "muII", 1.0))
    for n, ep, mu in ((nI,eI,mI),(nII,eII,mII)):
        _validate_material(ep,mu)
        if abs(n.imag)>1e-12 or n.real<=0 or abs(ep.imag)>1e-12 or abs(mu.imag)>1e-12:
            raise ValueError("This version requires lossless positive-index exterior ports")
        if abs(n*n-ep*mu)>1e-10*max(1,abs(n*n)):
            raise ValueError("Port index must satisfy n**2=eps*mu")
    bg_layers = [materials.Layer(L.d, eps=b.eps, mu=b.mu, name="background") for L,b in zip(layers,bgs)]
    bg_stack = materials.Stack(nI,bg_layers,nII,P,epsI=eI,epsII=eII,muI=mI,muII=mII)
    bg_res = materials.solve(bg_stack,lam0,theta_deg,0,pol,need_fields=True)
    ms = np.arange(-M,M+1); N=len(ms); k0=2*np.pi/lam0
    kappa = float(nI.real*np.sin(np.radians(theta_deg)))
    kxm = k0*kappa - 2*np.pi*ms/P
    Kx = np.diag(kxm/k0); F=pml.operator(P,M); D=F @ Kx
    d0=np.zeros(N,complex); d0[M]=1
    cache={}
    def getmode(L):
        if L.homogeneous:
            key=("h",L.eps,L.mu)
        else:
            key=("p",L.edges.tobytes(),L.eps_values.tobytes(),L.mu_values.tobytes(),
                 L.fourier_method,L.fft_samples)
        if key not in cache:
            cache[key]=_mode(L,P,M,D,Kx,pol,pml.strength==0)
        return cache[key]
    portI=getmode(materials.Layer(0,eps=eI,mu=mI))
    portII=getmode(materials.Layer(0,eps=eII,mu=mII))
    mods=[getmode(L) for L in layers]
    srcs=[_source_data(mod,bg,amp,D,kappa,d0,pol)
          for mod,bg,amp in zip(mods,bgs,bg_res.amps)]
    blocks=[]; prev=portI; zero=np.zeros(N,complex)
    for mod,src,L in zip(mods,srcs,layers):
        blocks.append((_interface(prev,mod),zero,zero))
        jd,_=_source_integrals(mod,src,L.d,k0,L.d)
        _,ju=_source_integrals(mod,src,L.d,k0,0)
        blocks.append((legacy.s_propagate(mod["q"],L.d,k0),jd,ju))
        prev=mod
    blocks.append((_interface(prev,portII),zero,zero))
    down,up=_affine_cascade(blocks,N)
    out=PMLResult(is_pml=True, formulation="contrast_field", pol=pol, stack=stack,
                  lam0=lam0, k0=k0, theta=np.radians(theta_deg), M=M, ms=ms,
                  kxm=kxm,Kx=Kx,F=F,D=D,kappa=kappa,pml=pml,period=P,
                  background=bg_res,background_materials=bgs,layers=layers,
                  reflected_modes=up[0],transmitted_modes=down[-1],
                  portI=portI,portII=portII,z_bot=sum(L.d for L in layers),
                  epsI=eI,epsII=eII,muI=mI,muII=mII,
                  incident_kxm=k0*kappa,incident_kz=complex(bg_res.kIz[0]),
                  modal_residual=max(m["modal_residual"] for m in cache.values()),
                  need_fields=need_fields)
    if need_fields:
        amps=[];ztop=0.0
        for j,(mod,src,L,bg) in enumerate(zip(mods,srcs,layers,bgs)):
            amps.append(dict(mode=mod,source=src,layer=L,bg=bg,z_top=ztop,d=L.d,
                             a_plus=down[2*j+1],a_minus=up[2*j+2]))
            ztop+=L.d
        out["amps"]=amps
    out["fourier_layers"]=[layer_fourier_info(L,P,M) for L in layers]
    return out


def _at(res,z):
    if not res["need_fields"]:
        raise ValueError("Recompute with need_fields=True for field reconstruction")
    if not np.isfinite(z):
        raise ValueError("z must be finite")
    k0=res["k0"]
    if z<0:
        mode=res["portI"]
        a=res["reflected_modes"]*np.exp(k0*mode["q"]*z)
        return mode["W"]@a,mode["V"]@a,mode,Background(res["epsI"],res["muI"])
    if z>res["z_bot"]:
        mode=res["portII"]
        a=res["transmitted_modes"]*np.exp(-k0*mode["q"]*(z-res["z_bot"]))
        return mode["W"]@a,-mode["V"]@a,mode,Background(res["epsII"],res["muII"])
    for a in res["amps"]:
        if z<=a["z_top"]+a["d"]+1e-12:
            t=np.clip(z-a["z_top"],0,a["d"])
            mode=a["mode"];q=mode["q"]
            jd,ju=_source_integrals(mode,a["source"],a["d"],k0,t)
            dp=np.exp(-k0*q*t)*a["a_plus"]+jd
            um=np.exp(-k0*q*(a["d"]-t))*a["a_minus"]+ju
            return mode["W"]@(dp+um),mode["V"]@(-dp+um),mode,a["bg"]
    raise RuntimeError("Layer lookup failed")


def field_at(res,z,kind="total"):
    """Main scalar S and tangential U coefficients at z.

    kind='contrast' is the globally periodic PML field. kind='total' adds the
    real-coordinate background: this extension is physically valid only outside
    PML, and is NOT the analytically continued total field inside PML.
    """
    S,U,_,_=_at(res,float(z))
    if kind=="contrast":
        return S,U
    if kind!="total":
        raise ValueError("kind must be total or contrast")
    sb,ub=materials.field_at(res["background"],float(z))
    S=S.copy();U=U.copy();S[res["M"]]+=sb[0];U[res["M"]]+=ub[0]
    return S,U


def _coefficients(res,z):
    S,U,mode,bg=_at(res,z)
    sb,ub=materials.field_at(res["background"],z);sb,ub=sb[0],ub[0]
    d0=np.zeros(len(S),complex);d0[res["M"]]=1
    chi_b=bg.mu if res["pol"]=="TE" else bg.eps
    normal_c=mode["chi_inv"] @ (res["D"]@S) + res["kappa"]*(mode["chi_inv"]@d0-d0/chi_b)*sb
    if res["pol"]=="TE":
        contrast=dict(Ey=S,Hx=1j*U,Hz=-normal_c)
        total=dict(Ey=S+d0*sb,Hx=1j*(U+d0*ub),Hz=-normal_c-res["kappa"]*d0*sb/chi_b)
    else:
        contrast=dict(Hy=S,Ex=-1j*U,Ez=normal_c)
        total=dict(Hy=S+d0*sb,Ex=-1j*(U+d0*ub),Ez=normal_c+res["kappa"]*d0*sb/chi_b)
    return total,contrast


def field_points(res,x,z):
    """Fields at broadcast-compatible paired coordinates; total fields masked in PML.

    *_contrast arrays contain auxiliary PML fields outside the physical region.
    E2 is normalized by incident electric intensity; raw components follow legacy
    conventions (Ey=1 for TE, Hy=1 for TM). Sx/Sz are raw time-average fluxes.
    """
    xx,zz=np.broadcast_arrays(np.asarray(x,float),np.asarray(z,float))
    if not np.isfinite(xx).all() or not np.isfinite(zz).all():
        raise ValueError("Sample coordinates must be finite")
    if np.any(xx<res["pml"].x0) or np.any(xx>res["pml"].x0+res["period"]):
        raise ValueError("Sample x must lie inside the computational cell")
    shape=xx.shape;xf=xx.ravel();zf=zz.ravel()
    keys=("Ey","Hx","Hz") if res["pol"]=="TE" else ("Hy","Ex","Ez")
    out={k:np.empty(len(xf),complex) for k in keys}
    out.update({k+"_contrast":np.empty(len(xf),complex) for k in keys})
    unique,inverse=np.unique(zf,return_inverse=True)
    for j,zj in enumerate(unique):
        sel=inverse==j
        ph=np.exp(-1j*np.outer(xf[sel],res["kxm"]))
        total,contrast=_coefficients(res,float(zj))
        for key in keys:
            out[key][sel]=ph@total[key]
            out[key+"_contrast"][sel]=ph@contrast[key]
    mask=res["pml"].physical_mask(xf,res["period"])
    for key in keys:
        out[key][~mask]=np.nan+1j*np.nan
    if res["pol"]=="TE":
        out["E2"]=abs(out["Ey"])**2
        out["Sx"]=0.5*np.real(out["Ey"]*np.conj(out["Hz"]))
        out["Sz"]=-0.5*np.real(out["Ey"]*np.conj(out["Hx"]))
    else:
        einc2=float(np.real(res["muI"]/res["epsI"]))
        out["E2"]=(abs(out["Ex"])**2+abs(out["Ez"])**2)/einc2
        out["Sx"]=-0.5*np.real(out["Ez"]*np.conj(out["Hy"]))
        out["Sz"]=0.5*np.real(out["Ex"]*np.conj(out["Hy"]))
    out={k:v.reshape(shape) for k,v in out.items()}
    out["physical_mask"]=mask.reshape(shape);out["x"]=xx;out["z"]=zz
    return out


def field_map(res,zs,nx=256,xs=None):
    """Tensor-product x-z sampling; *_contrast shows the PML damping."""
    xs=(res["pml"].x0+np.arange(nx)*res["period"]/nx) if xs is None else np.asarray(xs,float)
    zs=np.atleast_1d(zs).astype(float)
    if np.ndim(xs)!=1 or np.ndim(zs)!=1:
        raise ValueError("xs/zs must be one-dimensional")
    out=field_points(res,xs[None,:],zs[:,None])
    out["x"]=xs;out["z"]=zs
    out["physical_mask"]=res["pml"].physical_mask(xs,res["period"])
    return out
