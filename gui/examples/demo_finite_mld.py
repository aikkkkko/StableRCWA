"""Finite MLD grating with a central gap, TE at 66.36 degrees.

A finite 8-ridge demonstration: four SiO2 ridges on each side of a central gap,
on a 20-layer SiO2/HfO2 high-reflection stack (a 3L matching layer and H, then
nine 2L/H pairs) over a SiO2 substrate, with lateral PML on both sides.
"""
from pathlib import Path
import sys,time,json,argparse,csv
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import rcwa1d as rc
from pml import LateralPML,solve_pml,field_map

LAM=1.053;THETA=66.36;PITCH=.574;DUTY=.30;DEPTH=.380
NS=1.46;NH=1.96;D3L=.46438;DH=.115306;D2L=.30958


def build(pml_width=1.5795,padding=1.053,ridges_per_side=4,gap=1.148):
    w=PITCH*DUTY
    half=gap/2+(ridges_per_side-1)*PITCH+w
    period=2*(half+padding+pml_width);xc=period/2
    spans=[]
    for j in range(ridges_per_side):
        spans.append((xc-gap/2-w-j*PITCH,xc-gap/2-j*PITCH))
        spans.append((xc+gap/2+j*PITCH,xc+gap/2+w+j*PITCH))
    spans.sort();edges=[0.];vals=[]
    for a,b in spans:
        edges.extend([a,b]);vals.extend([1.,NS])
    edges.append(period);vals.append(1.)
    layers=[rc.Layer(DEPTH,edges=edges,n_values=vals,name="finite SiO2 ridges"),
            rc.Layer(D3L,n=NS,name="3L"),rc.Layer(DH,n=NH,name="H")]
    for _ in range(9):layers.extend([rc.Layer(D2L,n=NS,name="2L"),rc.Layer(DH,n=NH,name="H")])
    return rc.Stack(1.,layers,NS,period),spans


def main():
    a=argparse.ArgumentParser();a.add_argument("--output",default="results/finite_mld")
    a.add_argument("--orders",nargs="+",type=int,default=[120,180,240]);args=a.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    xs0=np.linspace(-2.6,2.6,601);zs=np.linspace(-.15,1.10,241)
    records=[];previous=None;last=None
    configs=[(M,1.5795) for M in args.orders]
    # Same scatterer and buffer, thicker PML, similar Fourier resolution.
    st0,_=build();st1,_=build(pml_width=2.106)
    configs.append((int(np.ceil(args.orders[-1]*st1.period/st0.period)),2.106))
    for M,pw in configs:
        st,spans=build(pml_width=pw);p=LateralPML(pw,pw,strength=3.,order=3)
        t=time.perf_counter()
        res=solve_pml(st,LAM,THETA,M,"TE",pml=p,background=[1.]+[None]*(len(st.layers)-1))
        seconds=time.perf_counter()-t
        fm=field_map(res,zs,xs=st.period/2+xs0)
        ey=fm["Ey"]*np.exp(1j*res.k0*res.kappa*st.period/2)
        ep,_=res.layers[0].profile(fm["x"],st.period)
        silica=(ep.real>1.5)[None,:]&((zs[:,None]>0)&(zs[:,None]<DEPTH))
        near=abs(xs0[None,:])<1.8
        mask=np.broadcast_to(near,ey.shape)
        rec=dict(M=M,period_um=st.period,pml_width_um=pw,solve_seconds=seconds,
                 central_complex_field_change=(None if previous is None else float(np.linalg.norm((ey-previous)[mask])/np.linalg.norm(ey[mask]))),
                 ridge_sampled_E2_peak=float(np.max(fm["E2"][silica])),
                 modal_residual=res.modal_residual)
        records.append(rec);print(json.dumps(rec),flush=True)
        previous=ey;last=(res,fm,ey,spans)
    (out/"metrics.json").write_text(json.dumps(records,indent=2), encoding="utf-8")
    with (out/"metrics.csv").open("w",newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    res,fm,ey,spans=last
    np.savez_compressed(out/"fields.npz",x_centered_um=xs0,z_um=zs,Ey_center_phase=ey,E2=fm["E2"])
    import matplotlib;matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
    im=ax.pcolormesh(xs0,zs,fm["E2"],shading="auto",cmap="inferno")
    for l,r in spans:
        l-=res.period/2;r-=res.period/2
        ax.plot([l,l,r,r],[DEPTH,0,0,DEPTH],color="cyan",lw=.7)
    zz=DEPTH
    for d in (D3L,DH,D2L):
        ax.axhline(zz,color="cyan",lw=.6);zz+=d
    ax.invert_yaxis();ax.set(xlabel="x from central gap / um",ylabel="z / um",
        title=f"Finite MLD: 8 ridges, gap=1.148 um, TE 66.36°, M={res.M}")
    fig.colorbar(im,ax=ax,label="|E|² / |E_inc|²")
    fig.savefig(out/"finite_mld_field.png",dpi=180);plt.close(fig)
    passed=(records[-1]["central_complex_field_change"]<.02 and
            records[-2]["central_complex_field_change"]<.02)
    (out/"validation_status.json").write_text(json.dumps({"passed":passed,"central_field_change_tolerance":.02,
        "scope":"Numerical stability trial; not an independent physical reference or laser damage validation"},indent=2), encoding="utf-8")
    if not passed:raise SystemExit("MLD Fourier/PML-width stability gate failed")


if __name__=="__main__":main()
