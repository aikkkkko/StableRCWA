"""Local-browser GUI entry for rcwasolver. Run: python gui.py.

Existing Python models can call launch_gui(stack, lam0=..., pml=..., background=...).
The HTTP server binds only 127.0.0.1. Calculations run in a cancellable process.
"""
from __future__ import annotations
from pathlib import Path
import argparse,copy,datetime,json,mimetypes,os,re,secrets,signal,subprocess,sys,threading,webbrowser
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlparse
from gui_models import PRESETS,preset,normalize,preview,from_stack,new_custom_model
from gui_fields import COMPONENTS,ACTIVE,PLOT_VERSION,PALETTES,render_view,export_six_fields,warmup_plotting
import cpp_fields_backend as cpp_backend

ROOT=Path(__file__).resolve().parent
ASSETS=ROOT/"gui_assets"
BLAS_THREAD_VARS=("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS")


def _physical_cores():
    if sys.platform=="darwin":
        # Apple Silicon: performance cores only; Intel Macs fall back to physical cores.
        for key in ("hw.perflevel0.physicalcpu","hw.physicalcpu"):
            try:
                n=int(subprocess.run(["sysctl","-n",key],capture_output=True,text=True,timeout=2).stdout)
                if n>0:return n
            except (OSError,ValueError,subprocess.SubprocessError):pass
    n=os.cpu_count() or 1
    # Most x86 desktops use SMT; dense LAPACK scales with physical cores, not logical ones.
    return max(1,n//2) if n>=4 else n


def blas_threads():
    """Worker BLAS/LAPACK threads: RCWA_THREADS=N, or physical cores (at most 16) by default.

    Only one solve runs at a time, so the worker may use the whole CPU.
    """
    raw=os.environ.get("RCWA_THREADS","auto").strip().lower()
    if raw in ("","auto"):return min(16,_physical_cores())
    try:n=int(raw)
    except ValueError:n=0
    if n<1:raise ValueError(f"RCWA_THREADS 须为正整数或 auto，当前为 {raw!r}")
    return n


class AppState:
    def __init__(self,initial,result_root=None):
        self.initial=normalize(initial);self.token=secrets.token_urlsafe(24)
        self.result_root=Path(result_root or ROOT/"results/gui_runs").resolve()
        self.jobs={};self.lock=threading.RLock();self.active=None
        self.threads=blas_threads()

    def get_job(self,jid):
        with self.lock:
            if not isinstance(jid,str) or not re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{6}",jid):
                raise KeyError("结果编号无效")
            if jid not in self.jobs:
                folder=self.result_root/jid
                if folder.resolve().parent!=self.result_root or not all((folder/f).is_file() for f in ("model.json","metrics.json","fields.npz")):
                    raise KeyError("找不到已完成的计算结果")
                cfg=json.loads((folder/"model.json").read_text(encoding="utf-8"))
                self.jobs[jid]={"process":None,"folder":folder,"config":cfg,"cancelled":False,"saved":True}
            return self.jobs[jid]

    def history(self):
        records=[]
        if not self.result_root.exists():return records
        for folder in sorted(self.result_root.iterdir(),reverse=True):
            if len(records)>=100:break
            if not folder.is_dir():continue
            try:
                job=self.get_job(folder.name)
                if not (folder/"metrics.json").is_file() or not (folder/"fields.npz").is_file():continue
                m=json.loads((folder/"metrics.json").read_text(encoding="utf-8"))
                records.append({"id":folder.name,"name":m["name"],"mode":m["mode"],"incident":m["incident"]})
            except (KeyError,ValueError,OSError):continue
        return records

    def launch(self,config):
        cfg=normalize(config)
        with self.lock:
            if self.active is not None and self.jobs[self.active]["process"] is not None and self.jobs[self.active]["process"].poll() is None:
                raise ValueError("已有计算正在运行，请等待完成或先取消")
            jid=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"-"+secrets.token_hex(3)
            folder=self.result_root/jid;folder.mkdir(parents=True)
            path=folder/"model.json";path.write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding="utf-8")
            env=os.environ.copy()
            for key in BLAS_THREAD_VARS:env[key]=str(self.threads)
            env["RCWA_BLAS_THREADS"]=str(self.threads)
            env["PYTHONUTF8"]="1";env["PYTHONIOENCODING"]="utf-8"
            env["MPLCONFIGDIR"]=str(self.result_root/".matplotlib")
            log=(folder/"worker.log").open("w",encoding="utf-8")
            proc=subprocess.Popen([sys.executable,"-u",str(ROOT/"gui_worker.py"),str(path),str(folder)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            log.close()
            self.jobs[jid]={"process":proc,"folder":folder,"config":cfg,"cancelled":False}
            self.active=jid
            return {"id":jid,"config":cfg}

    def status(self,jid):
        with self.lock:
            job=self.get_job(jid);folder=job["folder"]
            code=job["process"].poll() if job["process"] is not None else 0
            result={"id":jid,"folder":str(folder),"config":job["config"]}
            if job["cancelled"]:result.update(status="cancelled",stage="计算已取消")
            elif code is None:
                stage="正在启动求解器"
                p=folder/"progress.json"
                if p.exists():stage=json.loads(p.read_text(encoding="utf-8"))["stage"]
                result.update(status="running",stage=stage)
            elif code==0 and (folder/"metrics.json").exists():
                result.update(status="done",stage="计算完成",metrics=json.loads((folder/"metrics.json").read_text(encoding="utf-8")))
                if result["metrics"].get("adaptive_pml"):
                    # Adaptive sizing can change the artificial cell and M.
                    # Results/downloads must describe the solve actually selected.
                    job["config"]=json.loads((folder/"model.json").read_text(encoding="utf-8"))
                    result["config"]=job["config"]
                    if result["metrics"]["adaptive_pml"]["status"] not in ("verified","attenuation_met"):
                        result["stage"]="计算结束 · 自动PML未完成验证"
                result["metrics"]["components_available"]=list(COMPONENTS)
                pol=job["config"]["incident"]["polarization"]
                result["metrics"]["known_zero_components"]=[c for c in COMPONENTS if c not in ACTIVE[pol]]
            else:
                ep=folder/"error.json"
                error=json.loads(ep.read_text(encoding="utf-8")) if ep.exists() else {"message":f"求解进程退出（状态{code}）"}
                result.update(status="error",stage="计算未完成",error=error)
            return result

    def cancel(self,jid):
        with self.lock:
            job=self.get_job(jid);proc=job["process"]
            if proc is not None and proc.poll() is None:
                job["cancelled"]=True;proc.terminate()
                try:proc.wait(timeout=3)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=3)
                # Windows: terminate() is TerminateProcess, so the worker's own cleanup never runs and its C++ result folder
                # (cpp_fields-<token>, 391 MB at 1600x1200) would stay. The worker is gone now; remove what it left in THIS job's folder.
                # A result that was already complete (both files written, worker just exiting) is listed in the history: keep its raw folder.
                if not all((job["folder"]/f).is_file() for f in ("metrics.json","fields.npz")):cpp_backend.discard_job_outputs(job["folder"])
            return self.status(jid)

    def plot(self,jid,component,quantity,source,palette="auto"):
        if self.status(jid)["status"]!="done":raise ValueError("请等待计算完成后再选择分量")
        return render_view(self.get_job(jid)["folder"],component,quantity,source,palette)

    def shutdown(self):
        with self.lock:
            for jid in self.jobs:
                if self.jobs[jid]["process"] is not None and self.jobs[jid]["process"].poll() is None:self.cancel(jid)


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,format,*args):pass

        def send_data(self,data,status=200,mime="application/json; charset=utf-8",filename=None):
            if not isinstance(data,bytes):data=json.dumps(data,ensure_ascii=False).encode()
            self.send_response(status);self.send_header("Content-Type",mime)
            self.send_header("Content-Length",str(len(data)));self.send_header("Cache-Control","no-store")
            self.send_header("X-Content-Type-Options","nosniff")
            self.send_header("Content-Security-Policy","default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'")
            if filename:self.send_header("Content-Disposition",f'attachment; filename="{filename}"')
            self.end_headers()
            try:self.wfile.write(data)
            except (BrokenPipeError,ConnectionResetError):pass

        def do_GET(self):
            path=urlparse(self.path).path
            try:
                if path=="/":
                    html=(ASSETS/"index.html").read_text(encoding="utf-8").replace("__RCWA_TOKEN__",state.token)
                    return self.send_data(html.encode(),mime="text/html; charset=utf-8")
                if path in ("/app.js","/results.js","/style.css","/geometry2d.js","/geometry2d.css","/models2d.js","/adaptive-pml.js","/adaptive-pml.css"):
                    p=ASSETS/path.lstrip("/");return self.send_data(p.read_bytes(),mime="text/javascript" if p.suffix==".js" else "text/css")
                if path=="/api/bootstrap":return self.send_data({"presets":PRESETS,"config":state.initial,"history":state.history(),
                                                                 "backend_2d_default":cpp_backend.common.backend_2d()[0]})
                if path=="/api/new_model":return self.send_data(new_custom_model())
                if path=="/api/history":return self.send_data(state.history())
                if path.startswith("/api/preset/"):return self.send_data(preset(path.split("/")[-1]))
                bits=path.strip("/").split("/")
                if len(bits)==3 and bits[:2]==["api","jobs"]:return self.send_data(state.status(bits[2]))
                if len(bits)==5 and bits[:2]==["api","jobs"] and bits[3]=="files":
                    allowed={"field.png","phase.png","contrast.png","fields.npz","fields_six.npz","metrics.json","model.json","worker.log","requested_model.json","adaptive_pml.json"}
                    palettes="|".join(re.escape(p) for p in PALETTES if p!="auto")
                    viewfile=re.fullmatch(rf"view_{PLOT_VERSION}_(Ex|Ey|Ez|Hx|Hy|Hz|E2|H2)_(abs2|magnitude|real|imag|phase|phase_relative|log_magnitude)_(total|contrast)(?:_({palettes}))?\.png",bits[4])
                    if bits[4] not in allowed and not viewfile:raise KeyError("文件不存在")
                    job=state.get_job(bits[2]);p=job["folder"]/bits[4]
                    if not p.exists() and bits[4] in ("field.png","phase.png","contrast.png"):
                        if state.status(bits[2])["status"]!="done":raise ValueError("计算尚未完成")
                        inc=job["config"]["incident"];main="Ey" if inc["polarization"]=="TE" else "Hy"
                        args={"field.png":("E2","abs2","total"),"phase.png":(main,"phase_relative","total"),
                              "contrast.png":(main,"log_magnitude","contrast")}[bits[4]]
                        p=job["folder"]/render_view(job["folder"],*args)["filename"]
                    if bits[4]=="fields_six.npz":
                        if state.status(bits[2])["status"]!="done":raise ValueError("计算尚未完成")
                        p=export_six_fields(job["folder"])
                    if not p.exists():raise KeyError("文件尚未生成")
                    return self.send_data(p.read_bytes(),mime=mimetypes.guess_type(p.name)[0] or "application/octet-stream",
                                          filename=p.name if p.suffix not in (".png",) else None)
                self.send_data({"error":"页面不存在"},404)
            except (ValueError,KeyError,OSError) as exc:self.send_data({"error":str(exc)},400)

        def do_POST(self):
            path=urlparse(self.path).path
            if self.headers.get("X-RCWA-Token")!=state.token:return self.send_data({"error":"请刷新本地界面后重试"},403)
            origin=self.headers.get("Origin")
            if origin and origin!=f"http://{self.headers.get('Host')}":return self.send_data({"error":"请求来源无效"},403)
            try:
                length=int(self.headers.get("Content-Length",0))
                if not 0<length<=2_000_000:raise ValueError("模型文件过大或为空")
                data=json.loads(self.rfile.read(length).decode("utf-8"))
                if path=="/api/preview":return self.send_data(preview(data))
                if path=="/api/repeat/initialize":
                    from gui_repeat import initialize_repeat
                    return self.send_data(initialize_repeat(data))
                if path=="/api/repeat/rebuild":
                    from gui_repeat import rebuild_repeat
                    return self.send_data(rebuild_repeat(data.get("config"),data.get("options")))
                if path=="/api/profile/refine":
                    from gui_profiles import rebuild_profile
                    return self.send_data(rebuild_profile(data.get("config"),data.get("slices")))
                if path=="/api/run":return self.send_data(state.launch(data))
                if path=="/api/cancel":return self.send_data(state.cancel(data.get("id")))
                if path=="/api/plot":return self.send_data(state.plot(data.get("id"),data.get("component","E2"),data.get("quantity","abs2"),data.get("source","total"),data.get("palette","auto")))
                self.send_data({"error":"接口不存在"},404)
            except (ValueError,KeyError,TypeError,AttributeError,OSError) as exc:self.send_data({"error":str(exc)},400)

    return Handler


def launch_gui(stack=None,*,config=None,port=0,open_browser=True,**model_options):
    initial=config if config is not None else (from_stack(stack,**model_options) if stack is not None else preset("finite_mld"))
    state=AppState(initial);server=ThreadingHTTPServer(("127.0.0.1",port),make_handler(state))
    threading.Thread(target=warmup_plotting,daemon=True).start()
    server.daemon_threads=True;url=f"http://127.0.0.1:{server.server_port}/"
    print(f"RCWA_GUI_URL={url}",flush=True)
    print("界面已启动。关闭此进程可退出；计算结果保存到 results/gui_runs。",flush=True)
    print(f"求解进程 BLAS/LAPACK 线程数：{state.threads}（可用环境变量 RCWA_THREADS 指定）",flush=True)
    # Build/locate the C++ program ahead of the first run; started after the lines above so its status line cannot split them.
    threading.Thread(target=cpp_backend.warmup,daemon=True).start()
    handlers={}
    if threading.current_thread() is threading.main_thread():
        def stop(signum,frame):raise KeyboardInterrupt
        for sig in (signal.SIGTERM,getattr(signal,"SIGHUP",signal.SIGTERM)):
            if sig not in handlers:
                handlers[sig]=signal.getsignal(sig);signal.signal(sig,stop)
    if open_browser:webbrowser.open(url)
    try:server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:pass
    finally:
        state.shutdown();server.server_close()
        for sig,handler in handlers.items():signal.signal(sig,handler)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="RCWA本地图形入口")
    parser.add_argument("--model",type=Path,help="载入JSON模型")
    parser.add_argument("--port",type=int,default=0)
    parser.add_argument("--no-browser",action="store_true")
    args=parser.parse_args()
    initial=json.loads(args.model.read_text(encoding="utf-8")) if args.model else None
    launch_gui(config=initial,port=args.port,open_browser=not args.no_browser)
