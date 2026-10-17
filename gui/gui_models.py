"""JSON model interchange for the local 1D RCWA GUI. No GUI dependencies."""
from __future__ import annotations
import copy
from dataclasses import asdict
import math
import re
import numpy as np
import rcwa1d as rc
from pml import LateralPML
from spectral1d import resolve_fft_samples,layer_fourier_info

SCHEMA_VERSION=1
PRESETS=[
    {"id":"trapezoid_analytic","name":"MLD 梯形单周期 · 解析","description":"下底172 nm、上底103.2 nm、刻深380 nm；保留原膜系，斜壁分层近似。"},
    {"id":"trapezoid_fft","name":"MLD 梯形单周期 · FFT","description":"相同梯形与膜系；所有刻线切片使用FFT材料系数。"},
    {"id":"triangle_analytic","name":"MLD 三角单周期 · 解析","description":"尖顶宽0、下底172 nm、刻深380 nm；保留原膜系，斜壁分层近似。"},
    {"id":"triangle_fft","name":"MLD 三角单周期 · FFT","description":"相同三角形与膜系；所有刻线切片使用FFT材料系数。"},
    {"id":"adaptive_bar","name":"自动PML · 孤立介质条","description":"自动外扩PML，独立检查主复场；可查看逐次参数和停止原因。"},
    {"id":"pillar_2d","name":"二维刻线与膜系 · 双视图","description":"x、y均有变化的矩形刻线结构，可联动查看俯视图和剖面。"},
    {"id":"meent_conical","name":"Meent 已核对算例 · 圆锥","description":"官方教程的高折射率分段光栅；原方位角30弧度，已转换为本程序的角度与偏振约定。"},
    {"id":"finite_mld","name":"有限 MLD 光栅 · PML","description":"20 层介质高反膜，左右各4条刻线，中间间隔1.148 μm。"},
    {"id":"single_bar","name":"孤立介质条 · PML","description":"简单且计算快，适合第一次使用。"},
    {"id":"cylinder","name":"介质圆柱 · PML","description":"以32个薄层近似圆形截面，配套解析验证算例。"},
    {"id":"slab","name":"均匀介质膜 · 周期","description":"空气中的单层介质膜，可检验反射和透射。"},
]


def _number(value,name,lo=None,hi=None,integer=False):
    try:
        v=float(value)
    except (TypeError,ValueError):
        raise ValueError(name+"必须是数字")
    if not math.isfinite(v) or (lo is not None and v<lo) or (hi is not None and v>hi):
        raise ValueError(name+"超出允许范围")
    if integer and v!=int(v):raise ValueError(name+"必须是整数")
    return int(v) if integer else v


_UNIAXIAL=re.compile(r"^\s*([^/@]+?)\s*/\s*([^/@]+?)\s*@\s*([xyzXYZ])\s*$")


def complex_number(value,name):
    if isinstance(value,str) and _UNIAXIAL.match(value):
        raise ValueError(name+"：单轴晶体（no/ne@轴）只能用于“周期 RCWA · 平面入射”（TE/TM）中的层；"
                         "入射区、出射区、PML背景及其他求解方式须为各向同性折射率")
    try:
        if isinstance(value,dict):
            n=complex(value.get("real",0),value.get("imag",0))
        else:n=complex(str(value).strip().replace("i","j").replace(" ",""))
    except (TypeError,ValueError):raise ValueError(name+"格式应为 1.46 或 1.5+0.02j")
    if not np.isfinite([n.real,n.imag]).all() or n.real<=0 or n.imag<0:
        raise ValueError(name+"需要正实部、非负虚部的有限折射率")
    return n


def encode_n(n):
    n=complex(n)
    return float(n.real) if abs(n.imag)<1e-15 else f"{n.real:.15g}+{n.imag:.15g}j"


def material_value(value,name,allow_uniaxial=False):
    """Scalar refractive index, or a uniaxial crystal written 'no/ne@axis' (optic axis x, y or z)."""
    m=_UNIAXIAL.match(value) if allow_uniaxial and isinstance(value,str) else None
    if m is None:return complex_number(value,name)
    return rc.uniaxial(complex_number(m.group(1),name+"寻常光折射率no"),
                       complex_number(m.group(2),name+"非常光折射率ne"),m.group(3).lower())


def encode_material(value):
    """Inverse of material_value; a DiagonalEps must be uniaxial (two equal principal values)."""
    if not isinstance(value,rc.DiagonalEps):return encode_n(value)
    comps=dict(zip("xyz",(value.xx,value.yy,value.zz)))
    for axis in "xyz":
        ordinary=[comps[a] for a in "xyz" if a!=axis]
        if ordinary[0]==ordinary[1]:
            no=encode_n(np.sqrt(ordinary[0]))
            return no if comps[axis]==ordinary[0] else f"{no}/{encode_n(np.sqrt(comps[axis]))}@{axis}"
    raise ValueError("图形入口只支持单轴晶体（三个主介电常数中须有两个相等）")


def _display_eps(value,pol):
    """Permittivity shown in previews: the tensor component the main field sees (TE: yy, TM: xx)."""
    if isinstance(value,rc.DiagonalEps):return value.yy if pol=="TE" else value.xx
    return complex(value)**2


def from_stack(stack,*,name="自定义一维结构",lam0=1.053,theta_deg=66.36,M=None,
               pol="TE",pml=None,background=None,phi_deg=None,psi_deg=45.,delta_deg=0.,My=4,adaptive_pml=None):
    """Export a current nonmagnetic 1D Stack without executing model source files."""
    if hasattr(stack,"lat"):
        if pml is not None or background is not None:raise ValueError("二维周期入口不支持横向PML")
        from gui_models_2d import from_stack_2d
        cfg=from_stack_2d(stack,name="自定义二维结构" if name=="自定义一维结构" else name,
            lam0=lam0,theta_deg=theta_deg,M=4 if M is None else M,My=My,pol=pol,
            phi_deg=0. if phi_deg is None else phi_deg,psi_deg=psi_deg,delta_deg=delta_deg)
        if adaptive_pml is not None:cfg["adaptive_pml"]=adaptive_pml
        return normalize(cfg)
    M=100 if M is None else M
    for attr in ("muI","muII"):
        if abs(complex(getattr(stack,attr,1))-1)>1e-12:
            raise ValueError("图形入口当前仅支持非磁性介电结构")
    if phi_deg is not None and pml is not None:
        raise ValueError("当前圆锥入射支持周期结构，尚不支持横向PML")
    layers=[]
    backgrounds=[None]*len(stack.layers) if background is None else list(background)
    if len(backgrounds)!=len(stack.layers):raise ValueError("背景层数与结构不一致")
    for j,L in enumerate(stack.layers):
        mus=getattr(L,"mu_values",None)
        if mus is None:mus=[getattr(L,"mu",1) or 1]
        if np.any(abs(np.asarray(mus,complex)-1)>1e-12):
            raise ValueError("图形入口当前仅支持非磁性介电结构")
        row={"name":L.name or f"层 {j+1}","thickness":L.d,
             "fourier_method":getattr(L,"fourier_method","analytic"),
             "fft_samples":getattr(L,"fft_samples",0)}
        if L.homogeneous:row.update(kind="uniform",n=encode_material(L.n))
        else:
            row.update(kind="patterned",edges=np.asarray(L.edges,float).tolist(),
                       n_values=[encode_material(n) for n in L.n_values])
        bg=backgrounds[j]
        if bg is not None and hasattr(bg,"eps"):
            if abs(complex(bg.mu)-1)>1e-12:raise ValueError("GUI背景磁导率必须为1")
            bg=np.sqrt(complex(bg.eps))
        row["background_n"]=None if bg is None else encode_n(bg)
        layers.append(row)
    d=sum(L.d for L in stack.layers)
    config={"schema_version":SCHEMA_VERSION,"name":name,
            "mode":"conical" if phi_deg is not None else ("pml" if pml is not None else "periodic"),
            "incident":{"wavelength":lam0,"angle_deg":theta_deg,"polarization":pol,"M":M,
                        "phi_deg":0. if phi_deg is None else phi_deg,"psi_deg":psi_deg,"delta_deg":delta_deg},
            "structure":{"period":stack.period,"n_incident":encode_n(stack.nI),
                         "n_substrate":encode_n(stack.nII),"layers":layers},
            "pml":{"left":.15*stack.period,"right":.15*stack.period,"strength":3.,"order":3,"x0":0.},
            "sampling":{"z_min":-.2,"z_max":min(d+.2,1.4),"nx":360,"nz":220}}
    if pml is not None:
        config["pml"]={k:getattr(pml,k) for k in ("left","right","strength","order","x0")}
    if adaptive_pml is not None:config["adaptive_pml"]=adaptive_pml
    return normalize(config)


def preset(key):
    if key in ("trapezoid_analytic","trapezoid_fft","triangle_analytic","triangle_fft"):
        from examples.sloped_mld import build_stack,profile_metadata,LAM0,THETA
        shape,method=key.split('_');slices=64
        st=build_stack(shape,slices,fourier_method=method,fft_samples=65536 if method=='fft' else 0)
        label='梯形' if shape=='trapezoid' else '三角'
        cfg=from_stack(st,name=f"MLD {label}单周期 · {'FFT' if method=='fft' else '解析'}",lam0=LAM0,
                       theta_deg=THETA,M=30,pol="TE")
        cfg['sampling'].update(z_min=-.1,z_max=.85,nx=480,nz=280)
        cfg['profile_geometry']=profile_metadata(shape,slices)
        return normalize(cfg)
    if key=="adaptive_bar":
        cfg=preset("single_bar");cfg["name"]="孤立介质条 · 自动PML"
        cfg["adaptive_pml"]["enabled"]=True
        return cfg
    if key=="pillar_2d":
        from gui_models_2d import preset_2d
        return preset_2d()
    if key=="meent_conical":
        st=rc.Stack(1.,[rc.Layer(.5,edges=np.linspace(0,1,11),
            n_values=[1,5,1,5,5,1,5,1,5,5],name="Meent 10 段介质光栅")],1.,1.)
        cfg=from_stack(st,name="Meent 官方圆锥算例",lam0=.9,theta_deg=-20,M=200,
                       phi_deg=float(np.degrees(30)%360),pol="jones",psi_deg=45,delta_deg=180)
        cfg["sampling"].update(z_min=0.,z_max=.5,nx=240,nz=100)
        return cfg
    if key=="finite_mld":
        from examples.demo_finite_mld import build,LAM,THETA
        st,_=build();cfg=from_stack(st,name="有限 MLD 光栅",lam0=LAM,theta_deg=THETA,M=120,
            pml=LateralPML(1.5795,1.5795),background=[1.]+[None]*(len(st.layers)-1))
        cfg["structure"]["layers"][0]["name"]="有限 SiO₂ 刻线区"
        for j,L in enumerate(cfg["structure"]["layers"][1:],start=1):
            L["name"]=("SiO₂ 匹配层 3L" if j==1 else ("HfO₂ H" if L["n"]==1.96 else "SiO₂ 2L"))
        return cfg
    if key=="single_bar":
        st=rc.Stack(1.,[rc.Layer(.2,edges=[0.,1.3,1.7,3.],n_values=[1.,1.5,1.],name="孤立介质条")],1.,3.)
        return from_stack(st,name="孤立介质条",lam0=.628,theta_deg=30,M=40,
                          pml=LateralPML(.6,.6),background=[1.])
    if key=="cylinder":
        from examples.benchmark_cylinder import cylinder_stack
        st=cylinder_stack(3.6,32)
        return from_stack(st,name="介质圆柱截面",lam0=1.,theta_deg=30,M=30,
                          pml=LateralPML(.85,.85),background=[1.]*32)
    if key=="slab":
        # Period 0.8 um < lambda: only m=0 propagates, and no order has kz=0 in air or n=1.5
        # at normal incidence (the former 2 um period put m=+-2 exactly on a Rayleigh anomaly).
        return from_stack(rc.Stack(1.,[rc.Layer(.3,n=1.5,name="介质膜")],1.,.8),
                          name="均匀介质膜",lam0=1.,theta_deg=0,M=10)
    raise ValueError("未知预设结构")


def new_custom_model():
    """A small editable starting layer rather than a no-op 'custom' option."""
    return from_stack(rc.Stack(1.,[rc.Layer(.2,n=1.46,name="第1层")],1.,1.),
                      name="自定义模型",lam0=1.053,theta_deg=0,M=30)


def normalize(raw):
    if not isinstance(raw,dict):raise ValueError("模型需要是JSON对象")
    if raw.get("mode")=="periodic2d":
        from gui_models_2d import normalize_2d
        cfg=normalize_2d(raw)
        cfg.pop("repeat_geometry",None)
        if "profile_geometry" in cfg:
            from gui_profiles import validated_profile
            profile=validated_profile(cfg)
            if profile is None:cfg.pop("profile_geometry",None)
            else:cfg["profile_geometry"]=profile
        return _adaptive_config(cfg)
    cfg=copy.deepcopy(raw)
    if cfg.get("schema_version",1)!=1:raise ValueError("不支持此模型文件版本")
    cfg["schema_version"]=1;cfg["name"]=str(cfg.get("name","自定义结构"))[:120]
    if cfg.get("mode") not in ("pml","periodic","conical"):raise ValueError("求解方式应为 periodic、conical 或 pml")
    inc=cfg.get("incident",{});st=cfg.get("structure",{});sam=cfg.get("sampling",{})
    if not all(isinstance(v,dict) for v in (inc,st,sam)):raise ValueError("模型参数格式不完整")
    inc["wavelength"]=_number(inc.get("wavelength"),"波长",1e-9,1e6)
    inc["angle_deg"]=_number(inc.get("angle_deg"),"入射角",-89.9,89.9)
    inc["M"]=_number(inc.get("M"),"Fourier截断M",1,600,integer=True)
    inc["phi_deg"]=_number(inc.get("phi_deg",0.),"方位角",-360.,360.)
    inc["psi_deg"]=_number(inc.get("psi_deg",45.),"偏振混合角",0.,90.)
    inc["delta_deg"]=_number(inc.get("delta_deg",0.),"s相对p的相位差",-360.,360.)
    if cfg["mode"]=="conical":
        inc["polarization"]={"TE":"s","TM":"p"}.get(inc.get("polarization"),inc.get("polarization"))
        if inc["polarization"] not in ("s","p","jones"):raise ValueError("圆锥偏振请选择s、p或jones")
        if inc["polarization"] in ("s","p"):
            inc["psi_deg"]=90. if inc["polarization"]=="s" else 0.
            inc["delta_deg"]=0.
    elif cfg["mode"]=="pml":
        # PML mixed polarization is an exact TE+TM superposition (pml/jones.py).
        if inc.get("polarization") not in ("TE","TM","jones"):raise ValueError("PML偏振请选择TE、TM或混合偏振jones")
        if inc["phi_deg"]!=0.:raise ValueError("横向PML暂不支持圆锥入射，方位角须为0")
    else:
        if inc.get("polarization") not in ("TE","TM"):raise ValueError("平面周期入射请选择TE或TM；混合偏振请用圆锥模式（φ=0）")
        if inc["phi_deg"]!=0.:raise ValueError("非零方位角请使用圆锥周期模式")
    P=_number(st.get("period"),"横向计算宽度",1e-9,1e6);st["period"]=P
    for key,label in (("n_incident","入射区折射率"),("n_substrate","出射区折射率")):
        n=complex_number(st.get(key),label)
        if n.imag:raise ValueError(label+"在图形入口中要求为无损实数")
        st[key]=encode_n(n)
    layers=st.get("layers")
    if not isinstance(layers,list) or not 1<=len(layers)<=256:raise ValueError("层数需为1至256")
    uniaxial_ok=cfg["mode"]=="periodic"
    for j,L in enumerate(layers):
        if not isinstance(L,dict):raise ValueError(f"第{j+1}层格式不正确")
        label=f"第{j+1}层";L["name"]=str(L.get("name",label))[:120]
        L["thickness"]=_number(L.get("thickness"),label+"厚度",1e-9,1e6)
        if L.get("kind")=="uniform":
            L["n"]=encode_material(material_value(L.get("n"),label+"折射率",uniaxial_ok))
            L.pop("edges",None);L.pop("n_values",None)
        elif L.get("kind")=="patterned":
            try:
                edges=np.asarray(L.get("edges"),float)
                ns=[encode_material(material_value(v,label+"分段折射率",uniaxial_ok)) for v in L.get("n_values",[])]
            except (TypeError,ValueError) as exc:raise ValueError(label+"分段数据无效："+str(exc))
            if edges.ndim!=1 or not 2<=len(edges)<=2001 or len(edges)!=len(ns)+1:
                raise ValueError(label+"边界坐标数必须比分段折射率数多1")
            if not np.isfinite(edges).all() or np.any(np.diff(edges)<=0):raise ValueError(label+"边界必须严格递增")
            if abs(edges[-1]-edges[0]-P)>1e-8*max(P,1):raise ValueError(label+"首尾边界之差必须等于横向计算宽度")
            L["edges"]=edges.tolist();L["n_values"]=ns;L.pop("n",None)
        else:raise ValueError(label+"类型应为uniform或patterned")
        L["fourier_method"]=L.get("fourier_method","analytic")
        L["fft_samples"]=_number(L.get("fft_samples",0),label+"FFT采样数",0,4_194_304,integer=True)
        if L["fourier_method"] not in ("analytic","fft"):
            raise ValueError(label+"系数计算方式应为 analytic 或 fft")
        if L["kind"]=="uniform":
            L["fourier_method"]="analytic";L["fft_samples"]=0
        elif L["fourier_method"]=="fft":
            resolve_fft_samples(inc["M"],L["fft_samples"])
        if L.get("background_n") is not None:
            L["background_n"]=encode_n(complex_number(L["background_n"],label+"背景折射率"))
        else:L["background_n"]=None
    p=cfg.get("pml",{})
    if not isinstance(p,dict):raise ValueError("PML参数格式不正确")
    p.setdefault("left",.15*P);p.setdefault("right",.15*P);p.setdefault("strength",3.);p.setdefault("order",3);p.setdefault("x0",0.)
    if cfg["mode"]=="pml":
        p["left"]=_number(p["left"],"左PML厚度",1e-9)
        p["right"]=_number(p["right"],"右PML厚度",1e-9)
        p["strength"]=_number(p["strength"],"PML强度",0,100)
        p["order"]=_number(p["order"],"PML渐变次数",2,12,integer=True)
        p["x0"]=_number(p["x0"],"计算窗起点")
        pm=LateralPML(**p);pm.validate(P)
        # Same physical support check as the numerical kernel, before launching.
        from pml.afmm1d import _canonical_layer,_background_material,_check_pml_background
        for L in layers:
            layer=_canonical_layer(_layer(L))
            bg=_background_material(None if L["background_n"] is None else complex_number(L["background_n"],"背景"),layer)
            try:_check_pml_background(layer,bg,pm,P)
            except ValueError:raise ValueError(f"{L['name']}的结构进入了PML区：请在两侧留出与背景一致的缓冲区域")
    sam["z_min"]=_number(sam.get("z_min",-.2),"显示z下限")
    sam["z_max"]=_number(sam.get("z_max",1.4),"显示z上限")
    if sam["z_max"]<=sam["z_min"]:raise ValueError("显示z上限必须大于下限")
    sam["nx"]=_number(sam.get("nx",360),"横向采样数",40,1600,integer=True)
    sam["nz"]=_number(sam.get("nz",220),"纵向采样数",20,1200,integer=True)
    cfg.update(incident=inc,structure=st,pml=p,sampling=sam)
    if "repeat_geometry" in cfg:
        from gui_repeat import validated_repeat
        repeat=validated_repeat(cfg)
        if repeat is None:cfg.pop("repeat_geometry",None)
        else:cfg["repeat_geometry"]=repeat
    if "profile_geometry" in cfg:
        from gui_profiles import validated_profile
        profile=validated_profile(cfg)
        if profile is None:cfg.pop("profile_geometry",None)
        else:cfg["profile_geometry"]=profile
    return _adaptive_config(cfg)


def _adaptive_config(cfg):
    from pml.adaptive import settings_from_dict
    raw=cfg.get("adaptive_pml",{})
    if not isinstance(raw,dict):raise ValueError("自适应PML参数必须为对象")
    enabled=raw.get("enabled",False)
    if not isinstance(enabled,bool):raise ValueError("自适应PML开关必须为布尔值")
    options=settings_from_dict(raw)
    if enabled:
        if cfg["mode"]!="pml":raise ValueError("自动PML仅支持一维TE/TM横向PML模式")
        if options.max_M<cfg["incident"]["M"]:raise ValueError("自适应最大M不能小于初始M")
        if cfg["pml"]["strength"]<=0:raise ValueError("自动PML需要正的拉伸强度")
        if options.roi is not None:
            x0=cfg["pml"]["x0"];P=cfg["structure"]["period"]
            if options.roi[0]<x0+cfg["pml"]["left"] or options.roi[1]>x0+P-cfg["pml"]["right"]:
                raise ValueError("自动PML的观察区域必须位于两侧PML之间")
    cfg["adaptive_pml"]={"enabled":enabled,**asdict(options)}
    return cfg


def _layer(L):
    if L["kind"]=="uniform":return rc.Layer(L["thickness"],n=material_value(L["n"],"折射率",True),name=L["name"])
    return rc.Layer(L["thickness"],edges=L["edges"],n_values=[material_value(v,"折射率",True) for v in L["n_values"]],name=L["name"],
                    fourier_method=L.get("fourier_method","analytic"),fft_samples=L.get("fft_samples",0))


def build_stack(config):
    cfg=normalize(config);s=cfg["structure"]
    if cfg["mode"]=="periodic2d":
        from gui_models_2d import build_stack_2d
        return cfg,build_stack_2d(cfg),None,None
    stack=rc.Stack(float(s["n_incident"]),[_layer(L) for L in s["layers"]],float(s["n_substrate"]),s["period"])
    pml=LateralPML(**cfg["pml"]) if cfg["mode"]=="pml" else None
    bg=[None if L["background_n"] is None else complex_number(L["background_n"],"背景") for L in s["layers"]]
    return cfg,stack,pml,bg


def preview(config):
    """Exact material rectangles in one computational cell, not Fourier-smoothed."""
    if config.get("mode")=="periodic2d":
        from gui_models_2d import preview_2d
        return preview_2d(normalize(config))
    cfg=normalize(config);s=cfg["structure"];P=s["period"];pol=cfg["incident"]["polarization"]
    x0=cfg["pml"]["x0"] if cfg["mode"]=="pml" else 0.
    z=0.;layers=[]
    for L in s["layers"]:
        if L["kind"]=="uniform":cuts=np.array([x0,x0+P])
        else:cuts=np.unique(np.r_[x0,x0+P,x0+np.mod(np.asarray(L["edges"])-x0,P)])
        mid=(cuts[:-1]+cuts[1:])/2
        if L["kind"]=="uniform":vals=[material_value(L["n"],"折射率",True)]*len(mid)
        else:
            edges=np.asarray(L["edges"]);xx=edges[0]+np.mod(mid-edges[0],P)
            idx=np.clip(np.searchsorted(edges,xx,side="right")-1,0,len(L["n_values"])-1)
            table=[material_value(v,"折射率",True) for v in L["n_values"]]
            vals=[table[i] for i in idx]
        segments=[{"x0":float(a),"x1":float(b),"epsilon_real":float(_display_eps(n,pol).real),
                   "epsilon_imag":float(_display_eps(n,pol).imag),"n":encode_material(n)} for a,b,n in zip(cuts[:-1],cuts[1:],vals)]
        layers.append({"name":L["name"],"z0":z,"z1":z+L["thickness"],"segments":segments,
                       "fourier":layer_fourier_info(_layer(L),P,cfg["incident"]["M"])})
        z+=L["thickness"]
    return {"config":cfg,"x0":x0,"x1":x0+P,"depth":z,"layers":layers,
            "profile_geometry":cfg.get("profile_geometry")}


def material_grid(config,x,z):
    """Exact refractive index grid, including shifted segment origins.

    Uniaxial layers report sqrt of the component the main field sees (TE: εyy, TM: εxx).
    """
    cfg=normalize(config);st=cfg["structure"];P=st["period"];pol=cfg["incident"]["polarization"]
    index=lambda v:np.sqrt(_display_eps(v,pol)) if isinstance(v,rc.DiagonalEps) else v
    x=np.asarray(x,float);z=np.asarray(z,float)
    ends=np.cumsum([L["thickness"] for L in st["layers"]])
    out=np.empty((len(z),len(x)),complex)
    for j,zz in enumerate(z):
        if zz<0:out[j]=complex_number(st["n_incident"],"入射折射率")
        elif zz>ends[-1]:out[j]=complex_number(st["n_substrate"],"出射折射率")
        else:
            L=st["layers"][min(int(np.searchsorted(ends,zz,side="left")),len(ends)-1)]
            if L["kind"]=="uniform":out[j]=index(material_value(L["n"],"折射率",True))
            else:
                edges=np.asarray(L["edges"]);xx=edges[0]+np.mod(x-edges[0],P)
                idx=np.clip(np.searchsorted(edges,xx,side="right")-1,0,len(L["n_values"])-1)
                out[j]=np.asarray([index(material_value(v,"折射率",True)) for v in L["n_values"]])[idx]
    return out
