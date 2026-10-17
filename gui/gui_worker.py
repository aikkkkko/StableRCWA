"""Isolated numerical process used by gui.py. Writes results atomically."""
from pathlib import Path
from dataclasses import asdict
import sys,json,time,traceback,os,copy
_PROCESS_START=time.perf_counter()
os.environ.setdefault("MPLBACKEND","Agg")
import numpy as np
import rcwa1d as rc
from gui_models import build_stack,material_grid,from_stack,normalize
from gui_fields import complete_six_components,normalization,COMPONENTS,ACTIVE
import cpp_fields_backend as cpp_backend
_IMPORT_SECONDS=time.perf_counter()-_PROCESS_START


def write_json(path,data):
    tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
    # Windows refuses to replace a file while another process has it open (the GUI server reads progress.json on
    # every status poll), so a refusal is retried for up to 2 s before it counts as an error.
    for attempt in range(200):
        try:tmp.replace(path);return
        except PermissionError:
            if os.name!="nt" or attempt==199:raise
            time.sleep(.01)


def _order_index(res,m,n):
    return int(np.where((res["mn"][:,0]==m)&(res["mn"][:,1]==n))[0][0])


def compute(config,outdir):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    cfg,stack,pml,bg=build_stack(config);inc=cfg["incident"];sam=cfg["sampling"]
    write_json(outdir/"model.json",cfg)
    # Eligible models use one C++ solve for far fields, six complex fields and phase.
    # Python numerical work runs only for unsupported models, fallback, or explicit verification.
    cpp=cpp_backend.attempt(cfg,outdir,lambda stage:write_json(outdir/"progress.json",{"stage":stage}))
    write_json(outdir/"progress.json",{"stage":"正在整理 C++ 场结果" if cpp.used else "正在求解电磁场"})
    start=time.perf_counter();timings={"worker_import_seconds":_IMPORT_SECONDS}
    if os.environ.get("RCWA_BLAS_THREADS","").isdigit():timings["blas_threads"]=int(os.environ["RCWA_BLAS_THREADS"])
    conical=cfg["mode"]=="conical"
    two_d=cfg["mode"]=="periodic2d"
    adaptive=cfg.get("adaptive_pml",{}).get("enabled",False)
    python_solve_seconds=0.
    if cpp.used:
        res=cpp.result(stack,cfg)
        solve_time=float(cpp.manifest["timing"]["solve"])
        eff=res
        if cpp_backend.verify_requested():
            write_json(outdir/"progress.json",{"stage":"正在用 Python 核对 C++ 效率（显式验证模式）"})
            verification_started=time.perf_counter()
            try:
                if two_d:
                    from gui_models_2d import load_rcwa2d
                    reference=load_rcwa2d().solve2d(stack,inc["wavelength"],inc["angle_deg"],inc["phi_deg"],inc["M"],inc["My"],
                        psi_deg=inc["psi_deg"],delta_deg=inc["delta_deg"])
                elif conical:
                    from conical.rcwa_conical import solve_conical
                    reference=solve_conical(stack,inc["wavelength"],inc["angle_deg"],inc["phi_deg"],M=inc["M"],
                        psi_deg=inc["psi_deg"],delta_deg=inc["delta_deg"],need_fields=False)
                else:
                    reference=rc.solve(stack,inc["wavelength"],inc["angle_deg"],inc["M"],inc["polarization"],need_fields=False)
                cpp.verify(reference)
            except Exception as exc:
                cpp.verification_status="failed"
                cpp.note=cpp_backend.common._join(cpp.note,f"Python 核对失败（{type(exc).__name__}：{exc}），保留 C++ 结果")
            finally:
                python_solve_seconds=time.perf_counter()-verification_started
    else:
        if two_d:
            from gui_models_2d import load_rcwa2d
            rc2=load_rcwa2d()
            res=rc2.solve2d(stack,inc["wavelength"],inc["angle_deg"],inc["phi_deg"],inc["M"],inc["My"],
                            psi_deg=inc["psi_deg"],delta_deg=inc["delta_deg"])
        elif conical:
            from conical.rcwa_conical import solve_conical
            res=solve_conical(stack,inc["wavelength"],inc["angle_deg"],inc["phi_deg"],M=inc["M"],
                              psi_deg=inc["psi_deg"],delta_deg=inc["delta_deg"],need_fields=True)
        elif adaptive:
            requested=copy.deepcopy(cfg)
            write_json(outdir/"requested_model.json",requested)
            stages={'initial':'初始PML试算','absorption':'自动加厚PML','spectral_check':'独立提高M验证','window_check':'独立加厚PML验证'}
            def progress(event):
                if event['stage'] in stages:
                    label=f"{stages[event['stage']]} · 第{event['solves']+1}次 · M={event['M']}"
                    if event.get('polarization'):label=f"{event['polarization']} · {label}"
                    write_json(outdir/"progress.json",{"stage":label})
            res=rc.solve_adaptive_pml(stack,inc["wavelength"],inc["angle_deg"],inc["M"],inc["polarization"],
                pml=pml,background=bg,settings=cfg["adaptive_pml"],on_progress=progress,
                psi_deg=inc["psi_deg"],delta_deg=inc["delta_deg"])
            old_period=stack.period;stack=res.stack;pml=res.pml
            cfg=from_stack(stack,name=requested["name"],lam0=inc["wavelength"],theta_deg=inc["angle_deg"],
                M=res.M,pol=inc["polarization"],pml=pml,background=bg,adaptive_pml=requested["adaptive_pml"],
                psi_deg=inc["psi_deg"],delta_deg=inc["delta_deg"])
            cfg["pml"]=asdict(pml);cfg["sampling"]=copy.deepcopy(requested["sampling"])
            cfg["sampling"]["nx"]=min(1600,max(cfg["sampling"]["nx"],int(np.ceil(cfg["sampling"]["nx"]*stack.period/old_period))))
            cfg=normalize(cfg);inc=cfg["incident"];sam=cfg["sampling"]
            write_json(outdir/"model.json",cfg)
            write_json(outdir/"adaptive_pml.json",res.adaptive)
        elif pml is None:res=rc.solve(stack,inc["wavelength"],inc["angle_deg"],inc["M"],inc["polarization"])
        else:res=rc.solve_pml(stack,inc["wavelength"],inc["angle_deg"],inc["M"],inc["polarization"],pml=pml,background=bg,
                              psi_deg=inc["psi_deg"],delta_deg=inc["delta_deg"])
        solve_time=time.perf_counter()-start
        python_solve_seconds=solve_time
        eff=cpp.merge(res)
    timings["solve_seconds"]=solve_time
    timings["python_solve_seconds"]=python_solve_seconds
    field_start=time.perf_counter()
    write_json(outdir/"progress.json",{"stage":"正在整理场分布与相位" if cpp.used else "正在重建场分布与相位"})
    zs=np.linspace(sam["z_min"],sam["z_max"],sam["nz"])
    if cpp.used:
        fields=cpp.fields
    else:
        if two_d:
            # Each z's six harmonic vectors are evaluated once; one phase matrix
            # reconstructs the requested section, avoiding repeated exponentials.
            plane,at=sam["slice_plane"],sam["slice_at"]
            period=stack.lat.Lx if plane=="xz" else stack.lat.Ly
            t=np.arange(sam["nx"])*period/sam["nx"]
            xs,ys=(t,np.full_like(t,at)) if plane=="xz" else (np.full_like(t,at),t)
            phase=np.exp(-1j*res.k0*(np.outer(xs,res.Kx)+np.outer(ys,res.Ky)))
            harmonics=[rc2.field_at(res,z) for z in zs]
            fields={k:np.asarray([h[k] for h in harmonics])@phase.T for k in COMPONENTS}
            fields.update(E2=sum(abs(fields[k])**2 for k in COMPONENTS[:3]),x=xs,y=ys,t=t,z=zs,
                          slice_plane=np.asarray(plane),slice_at=np.asarray(at))
        elif conical:
            from conical.fields_conical import field_map
            fields=field_map(res,zs,nx=sam["nx"],y=0.)
        else:fields=rc.field_map(res,zs,nx=sam["nx"])
    timings["field_reconstruction_seconds"]=(float(cpp.manifest["timing"]["sample"]) if cpp.used else time.perf_counter()-field_start)
    post_start=time.perf_counter()
    fields=complete_six_components(fields,inc["polarization"],pml is not None)
    norms=normalization(cfg)
    fields["H2"]=sum(abs(fields["H"+axis])**2 for axis in "xyz")/norms["H"]**2
    fields["E_incident_amplitude"]=np.asarray(norms["E"])
    fields["H_incident_amplitude"]=np.asarray(norms["H"])
    x=fields["t"] if two_d else fields["x"]
    if two_d:
        from gui_models_2d import material_slice_2d
        nmap=material_slice_2d(cfg,x,zs,sam["slice_plane"],sam["slice_at"])
    else:nmap=material_grid(cfg,x,zs)
    main="Ey" if inc["polarization"]=="TE" else "Hy"
    if conical or two_d or inc["polarization"]=="jones":
        main=max(COMPONENTS[:3],key=lambda k:float(np.nanmax(abs(fields[k]))))
    finite=np.isfinite(fields["E2"])
    if not np.any(finite):
        if pml is not None:raise ValueError("物理区没有有效采样点，请调整PML和采样范围")
        raise ValueError("重建的场全部为非有限值（NaN/inf），请检查参数是否落在奇异点附近")
    peak=float(np.nanmax(fields["E2"]))
    iz,ix=np.unravel_index(np.nanargmax(fields["E2"]),fields["E2"].shape)
    metrics={"name":cfg["name"],"mode":cfg["mode"],"incident":inc,"sampling":sam,
             "layer_count":len(stack.layers),"solve_seconds":solve_time,
             "sampled_E2_peak":peak,"peak_x":float(x[ix]),"peak_z":float(zs[iz]),
             "fourier_layers":res.get("fourier_layers",[]),
             "components_available":list(COMPONENTS),
             "known_zero_components":[c for c in COMPONENTS if c not in ACTIVE[inc["polarization"]]],
             "component_plot_normalization":"E/|E_inc| and H/|H_inc| respectively",
             "orders":[],"R_total":None,"T_total":None,"absorption":None}
    if pml is None:
        metrics["R_total"]=float(np.sum(eff.DEr));metrics["T_total"]=float(np.sum(eff.DEt))
        metrics["absorption"]=1-metrics["R_total"]-metrics["T_total"]
        orders=res.mn if two_d else res.ms
        for m,r,t in zip(orders,eff.DEr,eff.DEt):
            if abs(r)+abs(t)>1e-9:
                row={"m":int(m[0]) if two_d else int(m),"R":float(r),"T":float(t)}
                if two_d:row["n"]=int(m[1])
                metrics["orders"].append(row)
    else:metrics["modal_residual"]=res.modal_residual
    if res.get("is_pml_jones"):
        weights=[[float(w.real),float(w.imag)] for w in res.weights]
        metrics["jones_superposition"]={"method":"exact TE+TM superposition on one discretization","solves":2,
            "jones_in":[[float(v.real),float(v.imag)] for v in res.jones],"TE_weight":weights[0],"TM_weight":weights[1]}
    if cfg.get("profile_geometry"):metrics["profile_geometry"]=cfg["profile_geometry"]
    if adaptive:metrics["adaptive_pml"]=res.adaptive
    if conical:
        metrics["modal_residual"]=res.mode_residual
        metrics["field_y"]=0.
        for row in metrics["orders"]:
            k=int(np.flatnonzero(res.ms==row["m"])[0])
            row.update(R_s=float(eff.flux_r[k]*abs(eff.Rs[k])**2),R_p=float(eff.flux_r[k]*abs(eff.Rp[k])**2),
                       T_s=float(eff.flux_t[k]*abs(eff.Ts[k])**2),T_p=float(eff.flux_t[k]*abs(eff.Tp[k])**2))
        fields.update(Jr=eff.Jr,Jt=eff.Jt,orders=res.ms,jones_in=res.jones_in,
                      flux_r=eff.flux_r,flux_t=eff.flux_t)
    if two_d:
        metrics["fourier_layers"]=[dict(name=L.name,method="uniform" if L.homogeneous else "analytic") for L in stack.layers]
        metrics.update(field_plane=sam["slice_plane"],field_at=sam["slice_at"],harmonic_count=int(res.N))
        metrics.update(peak_x=float(fields["x"][ix]),peak_y=float(fields["y"][ix]))
        den=float(stack.nI)*np.cos(np.radians(inc["angle_deg"]))
        for row in metrics["orders"]:
            k=_order_index(res,row["m"],row["n"])      # same lookup as rcwa2d.index_of; res may be C++'s far-field arrays
            fr=max(0.,float(np.real(res.kzI[k])/den));ft=max(0.,float(np.real(res.kzII[k])/den))
            row.update(R_s=fr*float(abs(res.Rs[k])**2),R_p=fr*float(abs(res.Rp[k])**2),
                       T_s=ft*float(abs(res.Ts[k])**2),T_p=ft*float(abs(res.Tp[k])**2))
        fields.update(orders=res.mn,Rs=res.Rs,Rp=res.Rp,Ts=res.Ts,Tp=res.Tp)
        from gui_fields import view_data
        phase=view_data(fields,cfg,main,"phase_relative")[0]
    elif cpp.used:
        phase=fields["phase_values"][1,COMPONENTS.index(main)].copy()
    else:
        from field_phase import phase_data
        phase=phase_data(res,fields,main,reference="incident")["phase"].filled(np.nan)
    fields["phase_relative"]=phase;fields["epsilon"]=nmap*nmap
    timings["postprocess_seconds"]=time.perf_counter()-post_start
    write_json(outdir/"progress.json",{"stage":"正在保存六分量数据"})
    save_start=time.perf_counter()
    temporary_fields=outdir/"fields.npz.tmp"
    with temporary_fields.open("wb") as stream:np.savez_compressed(stream,**fields)
    temporary_fields.replace(outdir/"fields.npz")
    timings["save_seconds"]=time.perf_counter()-save_start
    # Plot only the requested view in the long-lived GUI server. This avoids
    # importing Matplotlib in every solver process and drawing unused images.
    metrics["backend"]=cpp.report(python_solve_seconds)
    metrics["timings"]=timings
    metrics["total_seconds"]=time.perf_counter()-start+cpp.elapsed
    write_json(outdir/"metrics.json",metrics)
    status="计算完成"
    if adaptive and res.adaptive['status'] not in ('verified','attenuation_met'):status="计算结束 · 自动PML未完成验证"
    write_json(outdir/"progress.json",{"stage":status})
    return metrics


if __name__=="__main__":
    out=Path(sys.argv[2])
    try:compute(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")),out)
    except Exception as exc:
        write_json(out/"error.json",{"message":str(exc),"details":traceback.format_exc()})
        traceback.print_exc();raise SystemExit(1)
