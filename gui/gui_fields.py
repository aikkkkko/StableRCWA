"""Six-component post-processing for saved GUI results; never reruns RCWA.

For plane nonmagnetic 1D TE/TM problems, three components are
identically zero. Legacy archives stored only the three nonzero families;
those omitted components are completed explicitly, preserving the PML mask.
Conical s/p/Jones results require all six components from the vector solver;
PML Jones results store all six from the exact TE+TM superposition.
"""
from pathlib import Path
import json,threading,time
from collections import OrderedDict
import numpy as np

COMPONENTS=("Ex","Ey","Ez","Hx","Hy","Hz")
ACTIVE={"TE":("Ey","Hx","Hz"),"TM":("Hy","Ex","Ez")}
ACTIVE.update({p:COMPONENTS for p in ("s","p","jones")})
QUANTITIES=("abs2","magnitude","real","imag","phase","phase_relative","log_magnitude")
PLOT_VERSION="v6"
PALETTES=("auto","turbo","inferno","viridis","plasma")
_PLOT_LOCK=threading.RLock()
_FIELD_CACHE=OrderedDict()
_CACHE_BYTES=128*1024*1024


def complete_six_components(fields,polarization,is_pml=False):
    if polarization not in ACTIVE:raise ValueError("不支持此偏振的六分量显示")
    result=dict(fields)
    shape=np.asarray(fields["E2"]).shape
    if len(shape)!=2:raise ValueError("场数据必须是二维x-z网格")
    physical=np.isfinite(fields["E2"])
    for name in ACTIVE[polarization]:
        if name not in fields or np.shape(fields[name])!=shape:
            raise ValueError(f"原始结果缺少有效的{name}分量，不能补作零场")
    for name in COMPONENTS:
        if name not in result:
            result[name]=np.where(physical,0.+0j,np.nan+1j*np.nan)
        elif np.shape(result[name])!=shape:raise ValueError(f"{name}网格尺寸与E²不符")
        if is_pml:
            key=name+"_contrast"
            if key not in result:
                if name in ACTIVE[polarization]:raise ValueError(f"原始PML结果缺少{key}")
                result[key]=np.zeros(shape,complex)
    return result


def normalization(config):
    """Stored TM uses Hy_inc=1; stored TE uses Ey_inc=1, with mu=1.

    Vector results (conical, 2D, PML Jones superposition) store |E_inc|=1."""
    pol=config["incident"]["polarization"]
    n=float(config["structure"]["n_incident"])
    if config["mode"] in ("conical","periodic2d") or pol=="jones":return {"E":1.,"H":n}
    return {"E":1. if pol=="TE" else 1./n,"H":n if pol=="TE" else 1.}


def _stored_cpp_phase(fields,component,quantity):
    """Versioned C++ phase arrays are authoritative; older archives use the existing display rule."""
    if "phase_backend" not in fields:return None
    if np.asarray(fields["phase_backend"]).shape!=() or str(fields["phase_backend"])!="cpp":
        raise ValueError("保存结果的相位后端标记无效")
    required={"field_archive_version":1,"field_backend":"cpp","phase_archive_version":1,
              "phase_floor":1e-6,"phase_unwrap":"none"}
    for key,expected in required.items():
        value=np.asarray(fields.get(key))
        if value.shape!=() or type(value.item()) is not type(expected) or value.item()!=expected:
            raise ValueError("保存结果的 C++ 相位版本或约定不符")
    if (np.asarray(fields.get("phase_components")).tolist()!=list(COMPONENTS)
            or np.asarray(fields.get("phase_references")).tolist()!=["absolute","incident"]):
        raise ValueError("保存结果的 C++ 相位分量或参考次序不符")
    phase=np.asarray(fields.get("phase_values"));mask=np.asarray(fields.get("phase_masks"))
    shape=(2,6)+np.shape(fields[component])
    if phase.shape!=shape or phase.dtype!=np.dtype("float64") or mask.shape!=shape or mask.dtype!=np.dtype("bool"):
        raise ValueError("保存结果的 C++ 相位或掩膜格式不符")
    reference=1 if quantity=="phase_relative" else 0;index=COMPONENTS.index(component)
    data=phase[reference,index].copy();invalid=mask[reference,index]
    if not np.isfinite(data[~invalid]).all():raise ValueError("保存结果的 C++ 未掩膜相位无效")
    data[invalid]=np.nan
    return data


def view_data(fields,config,component="E2",quantity="abs2",source="total"):
    if component not in COMPONENTS+("E2","H2"):raise ValueError("请选择Ex/Ey/Ez/Hx/Hy/Hz或总场强度")
    if quantity not in QUANTITIES:raise ValueError("不支持该显示量")
    if source not in ("total","contrast"):raise ValueError("场类型应为total或contrast")
    is_pml=config["mode"]=="pml"
    if source=="contrast" and not is_pml:raise ValueError("当前周期结果没有单独保存对比场")
    f=complete_six_components(fields,config["incident"]["polarization"],is_pml)
    suffix="_contrast" if source=="contrast" else ""
    norm=normalization(config);notes=[];phase_backend=None
    if component in ("E2","H2"):
        if quantity!="abs2":raise ValueError("总矢量场只能显示强度；查看相位请先选择单个分量")
        family=component[0]
        data=sum(abs(f[family+axis+suffix])**2 for axis in "xyz")/norm[family]**2
        label=f"|{family}|² / |{family}_inc|²"
        title="Total electric intensity" if family=="E" else "Total magnetic intensity"
        zero=False
    else:
        value=np.asarray(f[component+suffix],complex)/norm[component[0]]
        finite=np.isfinite(value);amplitude=abs(value)
        peak=float(np.max(amplitude[finite])) if finite.any() else 0.
        zero=bool(peak==0.)
        title=component
        family=component[0]
        if quantity=="abs2":data=amplitude**2;label=f"|{component}|² / |{family}_inc|²"
        elif quantity=="magnitude":data=amplitude;label=f"|{component}| / |{family}_inc|"
        elif quantity=="real":data=value.real.copy();label=f"Re({component} / |{family}_inc|)"
        elif quantity=="imag":data=value.imag.copy();label=f"Im({component} / |{family}_inc|)"
        elif quantity=="log_magnitude":
            data=np.full(value.shape,np.nan) if zero else np.log10(np.maximum(amplitude,1e-12))
            label=f"log10 |{component} / {family}_inc|"
            notes.append("对数幅值下限为10⁻¹²。" if not zero else "零场的对数幅值为−∞，图中留空。")
        else:
            stored=_stored_cpp_phase(f,component,quantity) if source=="total" else None
            if quantity=="phase_relative":
                if stored is None:
                    inc=config["incident"];k0=2*np.pi/float(inc["wavelength"])
                    n=float(config["structure"]["n_incident"]);theta=np.radians(float(inc["angle_deg"]))
                    phi=np.radians(float(inc.get("phi_deg",0))) if config["mode"] in ("conical","periodic2d") else 0.
                    xx,yy=np.asarray(f["x"])[None,:],0.
                    if config["mode"]=="periodic2d":
                        yy=np.asarray(f["y"])[None,:]
                    else:yy=float(f.get("y",0.))
                    carrier=np.exp(1j*k0*n*(-np.sin(theta)*np.cos(phi)*xx
                        -np.sin(theta)*np.sin(phi)*yy
                        +np.cos(theta)*np.asarray(f["z"])[:,None]))
                    phasor=value*np.conj(carrier)
                title+= " · incident carrier removed"
                notes.append("已去除入射波的空间载波相位，未减去各矢量分量的方向符号。")
            else:phasor=value;title+=" · absolute phase"
            if stored is not None:data=stored;phase_backend="cpp"
            else:
                data=np.angle(phasor);phase_backend="python"
                data[~finite | (amplitude<=1e-6*peak)]=np.nan
            label="Phase / rad"
            if zero:notes.append("该分量为零，相位没有定义；不以0相位代替。")
        if zero:
            pol=config["incident"]["polarization"]
            notes.insert(0,f"{component}=0。"+(f"当前一维{pol}偏振下该分量恒为零。" if component not in ACTIVE[pol] else "当前算例中该分量为零。"))
    if source=="contrast":
        title+=" · contrast field"
        notes.append("显示总场减背景场所得的复对比场；PML内为辅助坐标数据，仅用于检查吸收。")
    else:notes.append("PML侧区保留掩膜，不作为真实材料中的场。" if is_pml else "显示本次保存的物理总场。")
    if config["mode"]=="periodic2d":
        sam=config["sampling"];axis="y" if sam["slice_plane"]=="xz" else "x"
        notes.append(f"本次保存为{sam['slice_plane']}剖面，固定{axis}={sam['slice_at']:g} μm。结构页移动剖切线仅更新几何预览；场图仍使用本次计算的切面。")
    values=np.asarray(data)[np.isfinite(data)]
    info={"component":component,"quantity":quantity,"source":source,"title":title,"colorbar":label,
          "zero_component":zero,"phase_undefined":zero and quantity in ("phase","phase_relative"),
          "minimum":float(values.min()) if len(values) else None,"maximum":float(values.max()) if len(values) else None,
          "note":" ".join(notes),"normalization":"E/|E_inc| and H/|H_inc| respectively",
          "phase_backend":phase_backend}
    return np.asarray(data),info


def _load(folder):
    folder=Path(folder).resolve()
    key=(str(folder),(folder/"fields.npz").stat().st_mtime_ns,(folder/"model.json").stat().st_mtime_ns)
    with _PLOT_LOCK:
        if key in _FIELD_CACHE:
            fields,config,size=_FIELD_CACHE.pop(key);_FIELD_CACHE[key]=(fields,config,size)
            return dict(fields),config
        config=json.loads((folder/"model.json").read_text(encoding="utf-8"))
        with np.load(folder/"fields.npz",allow_pickle=False) as archive:
            fields={name:archive[name] for name in archive.files}
        fields=complete_six_components(fields,config["incident"]["polarization"],config["mode"]=="pml")
        size=sum(getattr(v,"nbytes",0) for v in fields.values())
        if size<=_CACHE_BYTES:
            while _FIELD_CACHE and (len(_FIELD_CACHE)>=3 or sum(v[2] for v in _FIELD_CACHE.values())+size>_CACHE_BYTES):
                _FIELD_CACHE.popitem(last=False)
            _FIELD_CACHE[key]=(fields,config,size)
        return dict(fields),config


def warmup_plotting():
    """Load the renderer once in the GUI process, while the user edits inputs."""
    with _PLOT_LOCK:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg


def export_six_fields(folder):
    folder=Path(folder);target=folder/"fields_six.npz"
    with _PLOT_LOCK:
        if target.exists() and target.stat().st_mtime_ns >= max((folder/"fields.npz").stat().st_mtime_ns,(folder/"model.json").stat().st_mtime_ns):return target
        fields,config=_load(folder)
        norm=normalization(config)
        fields["H2"]=sum(abs(fields["H"+a])**2 for a in "xyz")/norm["H"]**2
        fields["E_incident_amplitude"]=np.asarray(norm["E"])
        fields["H_incident_amplitude"]=np.asarray(norm["H"])
        # Preserve the original dataset. A separate export completes old files.
        np.savez_compressed(target,**fields)
    return target


def render_view(folder,component="E2",quantity="abs2",source="total",palette="auto"):
    """Generate/cache a plot from an immutable result; no solver import or call."""
    if component not in COMPONENTS+("E2","H2") or quantity not in QUANTITIES or source not in ("total","contrast") or palette not in PALETTES:
        raise ValueError("显示选项无效")
    default_cmap=("twilight" if quantity in ("phase","phase_relative") else
                  "RdBu_r" if quantity in ("real","imag") else
                  "viridis" if quantity=="log_magnitude" else "inferno")
    cmap=default_cmap if palette=="auto" else palette
    folder=Path(folder);stem=f"view_{PLOT_VERSION}_{component}_{quantity}_{source}"
    if palette!="auto":stem+="_"+palette
    png=folder/(stem+".png");metadata=folder/(stem+".json")
    requested=time.perf_counter()
    with _PLOT_LOCK:
        stamp=max((folder/"fields.npz").stat().st_mtime_ns,(folder/"model.json").stat().st_mtime_ns)
        if png.exists() and metadata.exists() and min(png.stat().st_mtime_ns,metadata.stat().st_mtime_ns)>=stamp:
            info=json.loads(metadata.read_text(encoding="utf-8"));info.update(cache_hit=True,request_seconds=time.perf_counter()-requested,palette=palette,colormap=cmap)
            return info
        t=time.perf_counter();f,cfg=_load(folder);load_time=time.perf_counter()-t
        t=time.perf_counter();array,info=view_data(f,cfg,component,quantity,source);data_time=time.perf_counter()-t
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        plot_start=time.perf_counter()
        fig=Figure(figsize=(10,4.8),dpi=145);FigureCanvasAgg(fig)
        ax=fig.add_axes([.08,.12,.78,.80])
        finite=array[np.isfinite(array)]
        if quantity in ("phase","phase_relative"):
            vmin=-np.pi;vmax=np.pi
        elif quantity in ("real","imag"):
            limit=float(np.max(abs(finite))) if len(finite) else 0.
            if limit==0:limit=1.
            vmin=-limit;vmax=limit
        elif quantity=="log_magnitude":
            vmin=min(-1.,float(finite.min())) if len(finite) else -12.;vmax=max(0.,float(finite.max())) if len(finite) else 0.
        else:
            vmin=0.;vmax=float(finite.max()) if len(finite) else 1.
            if vmax==0:vmax=1.
        horizontal=f["t"] if cfg["mode"]=="periodic2d" else f["x"]
        image=ax.pcolormesh(horizontal,f["z"],array,shading="auto",cmap=cmap,vmin=vmin,vmax=vmax,rasterized=True)
        if "epsilon" in f:
            n=np.sqrt(f["epsilon"].astype(complex)).real;unique=np.unique(np.round(n,6))
            if 1<len(unique)<25:ax.contour(horizontal,f["z"],n,levels=(unique[:-1]+unique[1:])/2,colors="#73c3d4",linewidths=.4,alpha=.7)
        if cfg["mode"]=="pml":
            p=cfg["pml"];x0=p.get("x0",0);period=cfg["structure"]["period"]
            for edge in (x0+p["left"],x0+period-p["right"]):ax.axvline(edge,color="#748a9e",ls="--",lw=.8)
        axis="y" if cfg["mode"]=="periodic2d" and cfg["sampling"]["slice_plane"]=="yz" else "x"
        ax.set(xlabel=axis+" (μm)",ylabel="z (μm)",title=info["title"]);ax.invert_yaxis();ax.set_facecolor("#e8ecf1")
        if info["zero_component"]:
            label=component+" = 0"
            if info["phase_undefined"]:label+="\nPhase is undefined"
            elif quantity=="log_magnitude":label+="\nLog magnitude = -infinity"
            ax.text(.5,.5,label,transform=ax.transAxes,ha="center",va="center",color="#142b42",fontsize=15,
                    bbox=dict(facecolor="white",edgecolor="none",alpha=.9,pad=8))
        fig.colorbar(image,cax=fig.add_axes([.89,.12,.02,.80]),label=info["colorbar"])
        temp=png.with_suffix(".tmp.png");fig.savefig(temp);temp.replace(png)
        info["filename"]=png.name
        info.update(palette=palette,colormap=cmap)
        info.update(cache_hit=False,load_seconds=load_time,data_seconds=data_time,
                    render_seconds=time.perf_counter()-plot_start,request_seconds=time.perf_counter()-requested)
        temp_meta=metadata.with_suffix(".tmp");temp_meta.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding="utf-8");temp_meta.replace(metadata)
    return info
