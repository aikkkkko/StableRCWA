"""Actual isolated-cylinder scattering, compared with an independent Mie series.

python examples/benchmark_cylinder.py --quick
python examples/benchmark_cylinder.py --output results/cylinder
"""
from pathlib import Path
import sys,time,json,csv,argparse,platform
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import rcwa1d as rc
from pml import LateralPML,solve_pml,field_map,field_points
from validation.cylinder_reference import cylinder_field


def cylinder_stack(period,slices,radius=.25,n=1.5):
    layers=[]
    for j in range(slices):
        z=-radius+(j+.5)*2*radius/slices
        half=round(float(np.sqrt(radius*radius-z*z)),14)
        layers.append(rc.Layer(2*radius/slices,edges=[0,period/2-half,period/2+half,period],
                               n_values=[1.,n,1.],name="cylinder slice"))
    return rc.Stack(1.,layers,1.,period)


def contour_flux(res,extent=.72,count=501):
    xc=res.period/2;zc=.25
    t=np.linspace(-extent,extent,count)
    x=np.r_[xc+t,xc+t,np.full(count,xc-extent),np.full(count,xc+extent)]
    z=np.r_[np.full(count,zc-extent),np.full(count,zc+extent),zc+t,zc+t]
    f=field_points(res,x,z)
    if res.pol=="TE":
        ex=f["Ey_contrast"];hx=f["Hx_contrast"];hz=f["Hz_contrast"]
        sx=.5*np.real(ex*np.conj(hz));sz=-.5*np.real(ex*np.conj(hx))
    else:
        hy=f["Hy_contrast"];ex=f["Ex_contrast"];ez=f["Ez_contrast"]
        sx=-.5*np.real(ez*np.conj(hy));sz=.5*np.real(ex*np.conj(hy))
    outward=(-np.trapz(sz[:count],t)+np.trapz(sz[count:2*count],t)
             -np.trapz(sx[2*count:3*count],t)+np.trapz(sx[3*count:],t))
    net=(-np.trapz(f["Sz"][:count],t)+np.trapz(f["Sz"][count:2*count],t)
         -np.trapz(f["Sx"][2*count:3*count],t)+np.trapz(f["Sx"][3*count:],t))
    return float(2*outward),float(net)


def run_case(M,slices,pol,period=3.6,width=.85,strength=3.,theta=30.,save=False):
    st=cylinder_stack(period,slices)
    start=time.perf_counter()
    res=solve_pml(st,1.,theta,M,pol,pml=LateralPML(width,width,strength=strength,order=3),
                  background=[1.]*slices)
    elapsed=time.perf_counter()-start
    xs=period/2+np.linspace(-.70,.70,61)
    zs=.25+np.linspace(-.69,.71,62)
    got=field_map(res,zs,xs=xs)
    ref=cylinder_field(xs[None,:],zs[:,None],center=(period/2,.25),pol=pol,theta_deg=theta)
    rr=np.hypot(xs[None,:]-period/2,zs[:,None]-.25)
    # Separate external scattered field and physical total field away from the
    # curved staircase interface; near-boundary error is reported separately.
    outside=rr>.32;away=abs(rr-.25)>.055
    key="Ey" if pol=="TE" else "Hy"
    def rel(a,b,mask):return float(np.linalg.norm((a-b)[mask])/np.linalg.norm(b[mask]))
    err=rel(got[key],ref["main"],away)
    errsc=rel(got[key+"_contrast"],ref["scattered"],outside)
    errall=rel(got[key],ref["main"],np.ones_like(away,dtype=bool))
    csca,net=contour_flux(res)
    pmlf=field_map(res,[-.35,.13,.63,.85],nx=601)
    cfield=pmlf[key+"_contrast"]
    peak=np.max(abs(cfield[:,pmlf["physical_mask"]]))
    boundary=float(np.max(abs(cfield[:,[0,-1]]))/peak)
    rec=dict(pol=pol,M=M,slices=slices,period=period,pml_width=width,strength=strength,
             theta_deg=theta,solve_seconds=elapsed,total_field_relative_l2=err,
             scattered_field_relative_l2=errsc,total_field_all_points_l2=errall,
             scattering_width=csca,analytic_scattering_width=ref["scattering_width"],
             scattering_width_relative_error=abs(csca/ref["scattering_width"]-1),
             closed_contour_net_power=net,pml_edge_to_physical_peak=boundary,
             modal_residual=res.modal_residual)
    print(json.dumps(rec),flush=True)
    return rec,(res,got,ref) if save else None


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--quick",action="store_true")
    ap.add_argument("--output",default="results/cylinder");args=ap.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    records=[];last={}
    configs=[(30,32)] if args.quick else [(30,32),(50,64),(70,96),(90,128)]
    for pol in ("TE","TM"):
        for M,slices in configs:
            rec,data=run_case(M,slices,pol,save=(M==configs[-1][0]))
            records.append(rec)
            if data:last[pol]=data
        if not args.quick:
            # Same discretization with unattenuated periodic copies.
            rec,_=run_case(90,128,pol,strength=0)
            records.append(rec)
            # Enlarge exterior/PML and retain similar physical Fourier resolution.
            rec,_=run_case(110,128,pol,period=4.4,width=1.1,strength=3.)
            records.append(rec)
    with (output/"metrics.json").open("w", encoding="utf-8") as f:json.dump(records,f,indent=2)
    with (output/"metrics.csv").open("w",newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    (output/"environment.json").write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,platform=platform.platform()),indent=2), encoding="utf-8")
    import matplotlib;matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for pol,(res,got,ref) in last.items():
        key="Ey" if pol=="TE" else "Hy"
        x=got["x"]-res.period/2;z=got["z"]-.25
        fig,axs=plt.subplots(1,3,figsize=(12,3.6),constrained_layout=True)
        extent=[x[0],x[-1],z[-1],z[0]]
        for ax,arr,title in zip(axs,[abs(got[key])**2,abs(ref["main"])**2,abs(got[key]-ref["main"])],
                               ["PML-FMM |field|²","Independent Mie |field|²","Absolute complex-field error"]):
            im=ax.imshow(arr,extent=extent,aspect="equal",cmap="magma",vmin=0)
            if "error" not in title:im.set_clim(0,max(np.max(abs(got[key])**2),np.max(abs(ref["main"])**2)))
            ax.add_patch(plt.Circle((0,0),.25,fill=False,color="white",lw=.8))
            ax.set(title=title,xlabel="x / wavelength",ylabel="z / wavelength")
            fig.colorbar(im,ax=ax,shrink=.8)
        fig.suptitle(f"Isolated dielectric cylinder: {pol}, n=1.5, radius=0.25 wavelength, incidence=30°")
        fig.savefig(output/f"cylinder_{pol}.png",dpi=180);plt.close(fig)
        full=field_map(res,np.linspace(-.4,.9,151),nx=501)
        fig,ax=plt.subplots(figsize=(7.4,3.6),constrained_layout=True)
        im=ax.imshow(np.log10(np.maximum(abs(full[key+"_contrast"]),1e-8)),
                     extent=[0,res.period,.9,-.4],aspect="auto",cmap="viridis",vmin=-5,vmax=0)
        ax.axvline(res.pml.left,color="w",ls="--");ax.axvline(res.period-res.pml.right,color="w",ls="--")
        ax.set(title=f"{pol}: log10 |contrast field| (dashed: PML interfaces)",xlabel="x / wavelength",ylabel="z / wavelength")
        fig.colorbar(im,ax=ax);fig.savefig(output/f"pml_decay_{pol}.png",dpi=180);plt.close(fig)
        np.savez_compressed(output/f"fields_{pol}.npz",x=got["x"],z=got["z"],computed=got[key],reference=ref["main"],contrast=got[key+"_contrast"])
    if not args.quick:
        # This gate concerns physically independent accuracy, not just residuals.
        finest=[r for r in records if r["M"]==90 and r["strength"]==3]
        passed=all(r["scattered_field_relative_l2"]<.01 and r["scattering_width_relative_error"]<.01 for r in finest)
        (output/"validation_status.json").write_text(json.dumps({"passed":passed,"field_and_cross_section_tolerance":.01},indent=2), encoding="utf-8")
        if not passed:raise SystemExit("Independent Mie accuracy gate failed; inspect metrics")


if __name__=="__main__":main()
