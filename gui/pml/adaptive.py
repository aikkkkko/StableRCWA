"""Measured-tail PML sizing with separately optional physical-field checks.

This is engineering adaptive parameter control, not a rigorous a posteriori
error estimator. A small auxiliary contrast-field tail is not a reflection
coefficient, nor a bound on the error in the physical solution.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict,replace
import math,time
import numpy as np
from .afmm1d import solve_pml,field_map,field_at,LateralPML
from .adaptive_geometry import expand_pml_window


@dataclass(frozen=True)
class AdaptivePMLSettings:
    target_tail: float = .001
    verify: bool = True
    verification_rtol: float = .01
    verification_quantity: str = 'main'
    max_solves: int = 8
    max_M: int = 300
    max_width_factor: float = 4.
    max_seconds: float = 300.
    width_growth: float = 1.5
    spectral_growth: float = 1.35
    diagnostic_nx: int = 65
    diagnostic_nz: int = 25
    incident_floor: float = 1e-8
    roi: tuple | None = None

    def __post_init__(self):
        for name,lo,hi in [('target_tail',1e-10,.5),('verification_rtol',1e-10,.5),
                           ('max_width_factor',1.,32.),('max_seconds',.001,86400.),
                           ('width_growth',1.05,3.),('spectral_growth',1.05,3.),
                           ('incident_floor',1e-14,.01)]:
            v=getattr(self,name)
            if isinstance(v,bool) or not np.isfinite(v) or not lo<=v<=hi:
                raise ValueError(f'{name} must be finite in [{lo}, {hi}]')
        for name,lo,hi in [('max_solves',1,50),('max_M',1,600),('diagnostic_nx',17,513),('diagnostic_nz',9,257)]:
            v=getattr(self,name)
            if isinstance(v,bool) or int(v)!=v or not lo<=v<=hi:raise ValueError(f'{name} must be an integer in [{lo}, {hi}]')
        if not isinstance(self.verify,(bool,np.bool_)):raise ValueError('verify must be boolean')
        if self.verification_quantity not in ('main','electric'):raise ValueError('verification_quantity must be main or electric')
        if self.roi is not None:
            r=np.asarray(self.roi,float)
            if r.shape!=(4,) or not np.isfinite(r).all() or r[1]<=r[0] or r[3]<=r[2]:
                raise ValueError('roi must be (xmin,xmax,zmin,zmax) with increasing finite bounds')


def settings_from_dict(value=None):
    if isinstance(value,AdaptivePMLSettings):return value
    raw={} if value is None else dict(value)
    raw.pop('enabled',None)
    try:return AdaptivePMLSettings(**raw)
    except (TypeError,OverflowError) as exc:raise ValueError('Invalid adaptive PML settings: '+str(exc)) from None


def _grid(stack,pml,lam0,options):
    left=pml.x0+pml.left;right=pml.x0+stack.period-pml.right
    if options.roi is None:
        roi=(left,right,-.25*lam0,sum(L.d for L in stack.layers)+.25*lam0)
    else:roi=tuple(float(v) for v in options.roi)
    tol=1e-10*max(1.,stack.period)
    if roi[0]<left-tol or roi[1]>right+tol:raise ValueError('Adaptive ROI must lie in the original physical interval, outside both PMLs')
    roi=(max(left,roi[0]),min(right,roi[1]),roi[2],roi[3])
    # Interior samples avoid floating-point classification exactly at the PML interface.
    margin=min((roi[1]-roi[0])*1e-9,lam0*1e-10)
    xs=np.linspace(roi[0]+margin,roi[1]-margin,options.diagnostic_nx)
    zs=list(np.linspace(roi[2],roi[3],options.diagnostic_nz))
    z=0.
    for L in stack.layers:
        mid=z+.5*L.d
        if roi[2]<mid<roi[3]:zs.append(mid)
        z+=L.d
    return roi,xs,np.unique(zs)


def _diagnostic(res,xs,zs,options):
    """RMS of three outer traces on each side / physical main contrast RMS."""
    p=res.pml;end=p.x0+res.period
    left=p.x0+p.left*np.array([0.,.01,.02])
    right=end-p.right*np.array([0.,.01,.02])
    allx=np.r_[xs,left,right]
    phase=np.exp(-1j*np.outer(allx,res.kxm))
    coeff=np.asarray([field_at(res,float(z),kind='contrast')[0] for z in zs])
    f=coeff@phase.T
    def rms(a):return float(np.sqrt(np.mean(abs(a)**2)))
    physical=rms(f[:,:len(xs)]);den=max(physical,options.incident_floor)
    a=rms(f[:,len(xs):len(xs)+3]);b=rms(f[:,len(xs)+3:])
    ratio=max(a,b)/den
    if not np.isfinite(ratio):raise FloatingPointError('Non-finite adaptive PML diagnostic')
    return dict(tail_ratio=ratio,tail_ratio_left=a/den,tail_ratio_right=b/den,
                tail_rms_left=a,tail_rms_right=b,physical_contrast_rms=physical,
                normalization_floor=options.incident_floor)


def _physical_electric(res,xs,zs):
    f=field_map(res,zs,xs=xs)
    keys=('Ey',) if res.pol=='TE' else ('Ex','Ez')
    einc=1. if res.pol=='TE' else 1./float(np.real(res.stack.nI))
    E=np.stack([f[k]/einc for k in keys],axis=-1)
    C=np.stack([f[k+'_contrast']/einc for k in keys],axis=-1)
    if not np.isfinite(E).all() or not np.isfinite(C).all():
        raise FloatingPointError('Physical ROI contains invalid fields')
    return E,C


def _change(a,b,xs,zs,options):
    if options.verification_quantity=='electric':
        Ea,Ca=_physical_electric(a,xs,zs);Eb,Cb=_physical_electric(b,xs,zs)
    else:
        # Background is identical in physical coordinates, so its subtraction
        # leaves the same inter-solution difference, without loss of weak signals.
        def main(res):
            coefficients=np.asarray([field_at(res,float(z),kind='contrast')[0] for z in zs])
            return coefficients@np.exp(-1j*np.outer(xs,res.kxm)).T
        Ea=Ca=main(a);Eb=Cb=main(b)
    # Use the contrast E scale: a large incident/background wave must not hide
    # a poorly converged weak scattered field. Near zero use an incident floor.
    norm=np.linalg.norm(Eb-Ea)
    denominator=max(np.linalg.norm(Ca),np.linalg.norm(Cb),options.incident_floor*np.sqrt(Ea.size))
    return float(norm/denominator)


def solve_adaptive_pml(stack,lam0,theta_deg,M,pol='TE',*,pml,background=None,
                       settings=None,on_progress=None):
    """Automatically expand lateral PMLs; keep physical geometry/ROI fixed.

    Returns a normal PMLResult with additional JSON-safe ``adaptive`` metadata.
    ``verify=False`` stops at the measured tail target. ``verify=True`` also
    probes higher M and a thicker PML separately. Limits return the last result
    with an explicit non-success status. Time budget is checked between solves;
    a single numerical solve is not preempted (GUI cancellation can terminate it).
    """
    options=settings_from_dict(settings)
    if not isinstance(pml,LateralPML) or pml.strength<=0:raise ValueError('Adaptive PML requires a positive stretching strength')
    if not np.isfinite([lam0,theta_deg,M]).all() or lam0<=0 or abs(theta_deg)>=90 or int(M)!=M or M<1:
        raise ValueError('Require positive wavelength, integer M>=1 and |theta|<90 degrees')
    if M>options.max_M:raise ValueError('自适应最大M不能小于初始M')
    # Validate and canonicalize the old window once without mutating inputs.
    current_stack,current_pml,current_M,current_bg=expand_pml_window(
        stack,pml,background,left=pml.left,right=pml.right,M=int(M))
    roi,xs,zs=_grid(stack,pml,lam0,options)
    initial=dict(period=float(stack.period),x0=pml.x0,left=pml.left,right=pml.right,M=int(M))
    history=[];began=time.perf_counter();last=None;last_d=None
    spectral_change=None;window_change=None

    def notify(stage,**extra):
        if on_progress:on_progress(dict(stage=stage,solves=len(history),**extra))

    def room():
        if len(history)>=options.max_solves:return 'limit_reached','达到最大求解次数，尚未完成所选判据。'
        if history and time.perf_counter()-began>=options.max_seconds:return 'time_limit','达到时间预算，尚未完成所选判据。'
        return None

    def finish(status,message,selected=None,diagnostic=None):
        nonlocal last,last_d
        if selected is not None:last,last_d=selected,diagnostic
        assert last is not None
        meta=dict(enabled=True,method='measured_tail_with_optional_physical_checks',status=status,message=message,
            target_tail=options.target_tail,verify=options.verify,verification_rtol=options.verification_rtol,
            verification_quantity=options.verification_quantity,
            tail_ratio=last_d['tail_ratio'],tail_goal_met=bool(last_d['tail_ratio']<=options.target_tail),
            verified=status=='verified',max_solves_used=len(history),elapsed_seconds=time.perf_counter()-began,
            initial=initial,final=dict(period=last.period,x0=last.pml.x0,left=last.pml.left,right=last.pml.right,
                strength=last.pml.strength,order=last.pml.order,M=last.M),roi=list(roi),
            diagnostic_grid=[len(zs),len(xs)],settings=asdict(options),history=history,
            spectral_change=spectral_change,window_change=window_change,
            selected_solve_index=next(row['solve_index'] for row in history if row['M']==last.M and row['period']==last.period and row['x0']==last.pml.x0),
            diagnostic_note='Tail is auxiliary contrast-field RMS, not reflectance or a rigorous error bound. Checks are finite-grid stability tests, not peak/geometry-convergence guarantees.')
        last['adaptive']=meta
        notify('finished',adaptive=meta)
        return last

    def evaluate(st,pm,m,bg,stage):
        nonlocal last,last_d
        notify(stage,M=m,left=pm.left,right=pm.right)
        if pm.quadrature_order and pm.quadrature_order<4*m+16:
            pm=replace(pm,quadrature_order=4*m+16)
        t=time.perf_counter()
        result=solve_pml(st,lam0,theta_deg,m,pol,pml=pm,background=bg,need_fields=True)
        solve_seconds=time.perf_counter()-t
        diagnostic=_diagnostic(result,xs,zs,options)
        row=dict(solve_index=len(history)+1,stage=stage,M=m,period=st.period,x0=pm.x0,
                 left=pm.left,right=pm.right,strength=pm.strength,order=pm.order,
                 solve_seconds=solve_seconds,field_change=None,**diagnostic)
        history.append(row);last=result;last_d=diagnostic
        notify('evaluated',iteration=row)
        return result,diagnostic

    def wider(result,bg,factor,require_full=False):
        pm=result.pml
        if require_full and (pm.left*factor>pml.left*options.max_width_factor*(1+1e-12)
                             or pm.right*factor>pml.right*options.max_width_factor*(1+1e-12)):
            return None,'PML厚度上限不足以执行完整的独立加厚验证。'
        left=min(pm.left*factor,pml.left*options.max_width_factor)
        right=min(pm.right*factor,pml.right*options.max_width_factor)
        if left<=pm.left*(1+1e-12) and right<=pm.right*(1+1e-12):
            return None,'达到PML厚度上限，尚未完成所选判据。'
        new=expand_pml_window(result.stack,pm,bg,left=left,right=right,M=result.M)
        if new[2]>options.max_M:
            return None,f'外扩PML需M={new[2]}以保持谱带宽，超过上限{options.max_M}。'
        return new,None

    current,diag=evaluate(current_stack,current_pml,current_M,current_bg,'initial')
    while True:
        stopped=room()
        # A final absorption-only result may reach its goal on the last budgeted solve.
        if diag['tail_ratio']<=options.target_tail and not options.verify:
            return finish('attenuation_met','边界对比场残余达标；未执行物理区精化验证。')
        if stopped:return finish(*stopped)
        if diag['tail_ratio']>options.target_tail:
            proposal,why=wider(current,current_bg,options.width_growth)
            if proposal is None:return finish('limit_reached',why)
            current_stack,current_pml,current_M,current_bg=proposal
            current,diag=evaluate(current_stack,current_pml,current_M,current_bg,'absorption')
            continue
        # Tail target reached; optional physical checks are independent of that target.
        higher=max(current.M+1,math.ceil((2*current.M+1)*options.spectral_growth/2-.5))
        if higher>options.max_M:return finish('limit_reached','边界残余已达标，但提高M的物理区验证受到最大M限制。')
        probe,probe_diag=evaluate(current.stack,current.pml,higher,current_bg,'spectral_check')
        spectral_change=_change(current,probe,xs,zs,options)
        history[-1]['field_change']=spectral_change
        if spectral_change>options.verification_rtol or probe_diag['tail_ratio']>options.target_tail:
            current,diag=probe,probe_diag
            continue
        stopped=room()
        if stopped:return finish(*stopped)
        proposal,why=wider(current,current_bg,options.width_growth,require_full=True)
        if proposal is None:return finish('limit_reached',why)
        current_stack,current_pml,current_M,current_bg=proposal
        probe,probe_diag=evaluate(current_stack,current_pml,current_M,current_bg,'window_check')
        window_change=_change(current,probe,xs,zs,options)
        history[-1]['field_change']=window_change
        if window_change<=options.verification_rtol and probe_diag['tail_ratio']<=options.target_tail:
            quantity='主复场' if options.verification_quantity=='main' else '电场矢量'
            return finish('verified',f'边界残余达标，提高M与加厚PML的物理区{quantity}稳定性检查均通过。',current,diag)
        current,diag=probe,probe_diag
