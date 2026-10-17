"""兼容的效率专用 C++ API 与共用编译、进程管理工具。

GUI 的生产完整路径位于 cpp_fields_backend.py，调用 rcwa_fields 一次得到
效率、六复场与相位；本模块 attempt/merge 保留旧 rcwa_json 效率协议的语义。
下面描述的是这个旧 API，独立回归对拍与已有调用者可继续使用它。

哪些计算交给 C++
    求解方式是“周期 RCWA · 平面入射”（TE/TM）或“周期 RCWA · 圆锥入射”，所有材料各向同性，
    分段层的 Fourier 系数可取解析积分或 FFT。这时各级衍射效率（圆锥还有复 Jones 矩阵）由 C++ 算。
    二维周期（periodic2d）：材料各向同性、Fourier 系数取解析积分、谐波总数 N=(2M+1)(2My+1) 不超过 625 的模型 C++ 也能算，
    但**默认仍由 Python 求解器计算**：实测 C++ 的二维本征分解在每个 N 上都比 Python 慢（DEFAULT_BACKEND_2D 的注释有数字与改回的条件）。
    环境变量 RCWA_2D_BACKEND=cpp 打开二维的 C++ 路径（一维完全不受这个开关影响）。
    二维没有横向 PML、单轴晶体、FFT 系数与磁性：界面入口（gui_models_2d.normalize_2d）就拒绝这几类模型，
    不会交给任何求解器；一维的 PML、单轴晶体模型仍由 Python 求解器完成。
    这一版两边都算：C++ 给出结果页上的效率，Python 照常求解并重建场，两组效率之差记进 metrics.json。
    圆锥入射时 C++ 还报告 min_abs_lambda、degenerate_mode_warning 与 n_schur（使用完整 Schur 传播的层数）。
    它们记进 metrics.json 的 backend 段；Python 的 Schur/零根场传播计数也记录在这里。

命令行程序
    位置是 gui/bin/rcwa_json（Windows 上是 rcwa_json.exe）。不存在或比源码旧时，用系统里的编译器
    从 ../include 和 ../examples/rcwa_json.cpp 现场编译。请求、应答的格式见 StableRCWA 的说明。
    任何一步走不通（没有编译器、编译失败、超时、返回码非零、应答 ok 为假、数值失败）都不报错，
    只是退回全 Python，并把原因写进结果的 backend.note。

环境变量
    RCWA_BACKEND=python    强制只用 Python 求解器（一维、二维都是；RCWA_BACKEND 取别的值，包括 cpp，都不起作用）
    RCWA_2D_BACKEND=       二维周期模型用哪个求解器：python（不设或留空时的默认）或 cpp；别的值按默认处理并在 note 里说明。
                           只管二维：不设它时二维走 Python；RCWA_BACKEND=cpp 不会替它打开二维的 C++ 路径（它本来就不是有效的取值）
    RCWA_JSON_EXE=路径      直接用这个可执行文件，不再查找和编译
    RCWA_CXX=编译器         指定编译器（名字或完整路径）；不设时依次找 c++、g++、clang++，Windows 上再找 cl
    RCWA_CPP_TIMEOUT=秒     单次求解的时限，默认 1800

本文件只用标准库（merge 里才用到 NumPy），gui.py 与 gui_worker.py 都可以直接导入。
"""
from __future__ import annotations

import atexit
import cmath
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent              # gui/
STABLE_ROOT = ROOT.parent                          # StableRCWA/
INCLUDE = STABLE_ROOT / "include"
SOURCE = STABLE_ROOT / "examples" / "rcwa_json.cpp"
BIN = ROOT / "bin"
EXE_NAME = "rcwa_json.exe" if os.name == "nt" else "rcwa_json"
BUILD_TIMEOUT = 600.0                               # 编译的时限，秒
SOLVE_TIMEOUT = 1800.0                              # 单次求解的默认时限，秒
LARGE_DIFFERENCE = 1e-6                             # 两个求解器的效率相差超过它时在 note 里提醒


class BackendError(Exception):
    """C++ 这一路走不通的原因；调用方据此退回 Python。"""


class _NumericalFailure(BackendError):
    """应答里出现了 null（C++ 把 NaN、Inf 写成 null）。"""


_STATE = {"compiled": False}                        # 本进程里是否动过编译器（用来报告编译用时）


# --------------------------------------------------------------------------- 子进程的善后
# 界面取消计算时会终止工作进程。正在运行的 C++ 子进程（或编译器）不能留在后台：
#   · POSIX：子进程运行期间给 SIGTERM 装一个处理函数，先杀子进程，再让工作进程退出；
#   · Windows：TerminateProcess 不给被终止的进程任何机会，所以把子进程放进一个
#     “句柄关闭即终止”的作业对象，工作进程一死，系统替它收拾（尽力而为）。
_CHILDREN = set()
_CHILD_LOCK = threading.Lock()
_WINDOWS_JOB = None


def _kill_tree(child):
    """杀掉还在运行的子进程。POSIX 上连它自己起的进程一起杀（编译器的驱动程序另起 cc1、ld，只杀驱动程序的话
    它们会留在后台，还占着管道）：子进程是用 start_new_session 起的，进程组号就是它的进程号。
    只在子进程尚未被回收时才按组发信号，这时这个号不会已经换了主人。Windows 上用 taskkill /T 结束整棵进程树。"""
    if child.poll() is not None:
        return
    if os.name != "nt":
        try:
            os.killpg(child.pid, signal.SIGKILL)
            return
        except OSError:
            pass
    else:
        # Windows：child.kill() 只结束这一个进程。.bat 外壳（测试用的替身）、MinGW 的 g++ 驱动程序（另起 cc1plus、ld）留下的孙进程会继续跑，
        # 所以先按进程树结束；taskkill 不可用或失败时，下面的 child.kill() 照常兜底。
        # taskkill 用 System32 下的完整路径，不靠 PATH 里的名字（PATH 前面若有一个叫 taskkill 的程序就会被它截走）；找不到这个文件时不试，直接兜底。
        # 已知的理论缺口（不处理）：/T 按“父进程号”找后代，中间的某个后代若在这一刻之前已经退出，它的子孙就找不到父亲而成了孤儿，杀不到；
        # 进程号也可能被系统复用（刚好在 taskkill 开始前这个号换了主人）。彻底的办法是给每个子进程单独建一个作业对象再整体终止，这里有意不做
        # （_tie_to_this_process 的作业对象只管“工作进程自己没了”这一种情况）。
        taskkill = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "taskkill.exe")
        if os.path.isfile(taskkill):
            try:
                subprocess.run([taskkill, "/F", "/T", "/PID", str(child.pid)], capture_output=True, timeout=15, creationflags=0x08000000)
            except (OSError, subprocess.SubprocessError):
                pass
    child.kill()


def _kill_children():
    with _CHILD_LOCK:
        children = list(_CHILDREN)
    for child in children:
        try:
            _kill_tree(child)
        except OSError:
            pass


atexit.register(_kill_children)


@contextmanager
def _children_die_with_us():
    """运行子进程期间接管 SIGTERM；只在主线程里有效，结束后恢复原来的处理方式。"""
    if os.name == "nt" or threading.current_thread() is not threading.main_thread():
        yield
        return

    def handler(signum, frame):
        _kill_children()
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _windows_job_handle():
    """建一个 KILL_ON_JOB_CLOSE 的作业对象（只建一次）；失败返回 None。"""
    global _WINDOWS_JOB
    if _WINDOWS_JOB is not None:
        return _WINDOWS_JOB or None
    _WINDOWS_JOB = 0
    try:
        import ctypes
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in
                        ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                         "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", IoCounters),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.SetInformationJobObject.restype = wintypes.BOOL
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        job = kernel.CreateJobObjectW(None, None)
        if not job:
            return None
        info = ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = 0x2000          # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):   # 9 = 扩展限制信息
            return None
        _WINDOWS_JOB = (kernel, job)
    except Exception:
        return None
    return _WINDOWS_JOB


def _tie_to_this_process(child):
    if os.name != "nt":
        return
    try:
        tied = _windows_job_handle()
        if tied:
            tied[0].AssignProcessToJobObject(tied[1], int(child._handle))
    except Exception:
        pass


def _run_child(args, timeout, cwd=None, env=None, text=None):
    """运行一个子进程，返回 (返回码, 标准输出, 标准错误)；超时就杀掉它并抛 subprocess.TimeoutExpired。

    text 不是 None 时作为标准输入（UTF-8）送给子进程。
    """
    flags = 0x08000000 if os.name == "nt" else 0               # CREATE_NO_WINDOW：不弹出黑窗口
    child = subprocess.Popen(args, cwd=cwd, env=env, stdin=subprocess.DEVNULL if text is None else subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags,
                             start_new_session=os.name != "nt")
    with _CHILD_LOCK:
        _CHILDREN.add(child)
    _tie_to_this_process(child)
    try:
        out, err = child.communicate(None if text is None else text.encode("utf-8"), timeout=timeout)
    except BaseException:
        try:
            _kill_tree(child)
            child.communicate(timeout=5)
        except (OSError, subprocess.SubprocessError):
            pass
        raise
    finally:
        with _CHILD_LOCK:
            _CHILDREN.discard(child)
    return child.returncode, _decode(out), _decode(err)


def _decode(data):
    """rcwa_json 的输出是 UTF-8；编译器的报错可能是本机代码页（中文 Windows 上的 MSVC 是 GBK）。"""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    if os.name == "nt":
        # 本机代码页就是 "mbcs"。这里不能去问 locale：界面的启动脚本和工作进程都开着 Python 的 UTF-8 模式，
        # 那时 locale 只会回答 utf-8，GBK 的报错就成了一串乱码。
        return data.decode("mbcs", "replace")
    import locale
    return data.decode(locale.getpreferredencoding(False) or "utf-8", "replace")


# --------------------------------------------------------------------------- 找到或构建可执行文件
def forced_python():
    return os.environ.get("RCWA_BACKEND", "").strip().lower() == "python"


def executable_path():
    """命令行程序的路径。RCWA_JSON_EXE 里的相对路径相对于当前目录，这里换成绝对路径：
    不带目录的名字原样交给 Popen 的话，它会到 PATH 里去找，而不是用当前目录下的那个文件。"""
    override = os.environ.get("RCWA_JSON_EXE", "").strip()
    return Path(os.path.abspath(os.path.expanduser(override))) if override else BIN / EXE_NAME


def _source_files(source=None):
    source = SOURCE if source is None else source
    files = [source] + sorted(source.parent.glob("*.hpp")) + sorted(INCLUDE.rglob("*.hpp")) + sorted(INCLUDE.rglob("*.h"))
    return [f for f in files if f.is_file()]


def _source_stamp(source=None):
    """源码里最新的修改时间（纳秒）；主程序的源码不存在时返回 None。"""
    source = SOURCE if source is None else source
    if not source.is_file():
        return None
    return max(f.stat().st_mtime_ns for f in _source_files(source))


def _built_from(exe, source=None):
    """眼前这个程序和这份源码的“指纹”：程序的（大小, 修改时间），各源文件的（名字, 大小, 修改时间）。"""
    def row(f):
        info = f.stat()
        return [info.st_size, info.st_mtime_ns]
    names = [("include/" + f.relative_to(INCLUDE).as_posix()) if INCLUDE in f.parents else ("examples/" + f.name)
             for f in _source_files(source)]
    return {"program": row(exe), "sources": [[name] + row(f) for name, f in zip(names, _source_files(source))]}


def _up_to_date(exe, stamp, source=None):
    """程序是不是不必重编。

    先比修改时间（程序不比最新的源文件旧）。光比时间看不出“内容换了、日期却更早”的源文件（解压一份旧包、
    从备份还原、云同步送来一份早先改的文件），所以编译成功时把指纹记在程序旁边，这里再对照一遍。
    没有这份记录，或者它记的不是眼前这个程序（手工编好放进来的），就只比时间。
    """
    if not exe.is_file() or exe.stat().st_mtime_ns < stamp:
        return False
    try:
        record = json.loads(exe.with_name(exe.name + ".sources.json").read_text(encoding="utf-8"))
        current = _built_from(exe, source)
    except (OSError, ValueError):
        return True
    return not isinstance(record, dict) or record.get("program") != current["program"] or record.get("sources") == current["sources"]


def find_compilers():
    """可用的编译器，按先后顺序：[(种类, 路径)]，种类是 "gnu"（g++ / clang++ 的写法）或 "msvc"。"""
    names = [os.environ["RCWA_CXX"].strip()] if os.environ.get("RCWA_CXX", "").strip() else \
        ["c++", "g++", "clang++"] + (["cl"] if os.name == "nt" else [])
    found, seen = [], set()
    for name in names:
        path = shutil.which(name)
        if not path:
            continue
        if sys.platform == "darwin" and os.path.dirname(path) == "/usr/bin":
            # 没装命令行工具时 /usr/bin 下的编译器只是会弹安装窗的占位程序，跳过
            try:
                if subprocess.run(["/usr/bin/xcode-select", "-p"], capture_output=True, timeout=10).returncode != 0:
                    continue
            except (OSError, subprocess.SubprocessError):
                continue
        kind = "msvc" if Path(path).stem.lower() == "cl" else "gnu"
        try:      # c++、g++、clang++ 常常是同一个编译器：按版本信息去重，失败时不必重复试
            probe = subprocess.run([path] if kind == "msvc" else [path, "--version"], capture_output=True, timeout=30)
            ident = (probe.stdout + probe.stderr).strip().splitlines()[:1]
        except (OSError, subprocess.SubprocessError):
            continue
        # GCC 的版本信息以程序自己的名字开头（“g++ (GCC) 13.1.0”、“c++.EXE (…) 13.1.0”）：去掉名字再比
        words = ident[0].split(None, 1) if ident else []
        if len(words) == 2 and words[0].lower().startswith(Path(path).stem.lower().encode()):
            ident = [words[1]]
        key = (kind, tuple(ident))
        if key not in seen:
            seen.add(key)
            found.append((kind, path))
    return found


def _compile_command(kind, compiler, target, source=None):
    source = SOURCE if source is None else source
    examples = source.parent
    if kind == "msvc":
        return [compiler, "/nologo", "/std:c++17", "/O2", "/EHsc", "/utf-8", "/I", str(INCLUDE), "/I", str(examples),
                str(source), "/Fe:" + str(target), "/Fo:" + str(target.with_suffix(".obj"))]
    command = [compiler, "-std=c++17", "-O2", "-I", str(INCLUDE), "-I", str(examples), str(source), "-o", str(target)]
    if os.name == "nt":
        command.append("-static")       # MinGW：静态链接，程序运行时不依赖编译器目录里的运行库
    elif sys.platform != "darwin":
        command.append("-pthread")
    return command


def _retry_while_in_use(action, seconds=2.0):
    """执行 action()（给文件改名或删除文件）。

    Windows 上文件正被别的进程开着的那一刻（等锁的进程在读锁文件、杀毒软件在扫描刚生成的程序），改名和删除会被
    拒绝（PermissionError）：稍等再试，过了 seconds 秒仍不行才算出错。别的系统上没有这回事，直接抛出。
    """
    deadline = time.monotonic() + seconds
    while True:
        try:
            return action()
        except PermissionError:
            if os.name != "nt" or time.monotonic() > deadline:
                raise
            time.sleep(0.01)


def _process_gone(pid):
    """进程 pid 是否已经不在了；拿不准时返回 False。"""
    if os.name != "nt":
        try:
            os.kill(pid, 0)                                      # 信号 0：只探测，不发信号
        except ProcessLookupError:
            return True
        except OSError:
            pass
        return False
    # Windows 上 os.kill(pid, 0) 会真的终止那个进程，所以直接问系统
    try:
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x00100000, False, pid)      # SYNCHRONIZE：只为了能等它
        if not handle:
            return ctypes.get_last_error() == 87                 # ERROR_INVALID_PARAMETER：没有这个进程
        try:
            return kernel.WaitForSingleObject(handle, 0) == 0    # 等得到（WAIT_OBJECT_0）说明它已经结束
        finally:
            kernel.CloseHandle(handle)
    except Exception:
        return False


@contextmanager
def _build_lock(progress, binary_dir=None, program="rcwa_json"):
    """同一时刻只让一个进程编译（界面启动时的预编译与工作进程可能撞在一起）。

    锁文件里记着持锁进程的进程号：那个进程已经不在了（编译途中计算被取消、界面被关掉），锁就作废。
    """
    binary_dir = BIN if binary_dir is None else binary_dir
    lock = binary_dir / "build.lock"
    deadline = time.monotonic() + BUILD_TIMEOUT
    announced = False
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            stale = False
            try:
                stale = time.time() - lock.stat().st_mtime > BUILD_TIMEOUT + 30
                if not stale:
                    try:
                        owner = int(lock.read_text().strip() or 0)
                        stale = owner > 0 and _process_gone(owner)
                    except ValueError:
                        stale = True
            except OSError:
                continue                                         # 锁刚被释放，重试
            if stale:
                try:
                    lock.unlink()
                except OSError:
                    pass
                continue
            if time.monotonic() > deadline:
                raise BackendError(f"另一个进程正在编译 {program}，等待超时")
            if progress and not announced:
                progress("正在等待 C++ 后端编译完成")
                announced = True
            time.sleep(0.2)
    try:
        yield
    finally:
        try:
            _retry_while_in_use(lock.unlink)
        except OSError:
            pass


def _build(exe, stamp, progress, retry_failed=False, source=None, program="rcwa_json"):
    source = SOURCE if source is None else source
    binary_dir = exe.parent
    prefix = "" if program == "rcwa_json" else program + "."
    compilers = find_compilers()
    if not compilers:
        wanted = os.environ.get("RCWA_CXX", "").strip() or ("c++ / g++ / clang++" + (" / cl" if os.name == "nt" else ""))
        if exe.is_file():
            return exe, f"命令行程序比源码旧，但找不到 C++ 编译器（{wanted}），沿用现有的程序"
        return None, f"找不到 C++ 编译器（{wanted}），无法编译 {program}"
    failure = binary_dir / (prefix + "build_failed.json")
    signature = [stamp, [path for _, path in compilers]]
    try:        # 同一份源码、同样的编译器已经失败过：不必每次计算都重试（界面启动时的预编译除外，见 warmup）
        record = json.loads(failure.read_text(encoding="utf-8"))
        if not retry_failed and record.get("signature") == signature:
            return None, record.get("message", f"编译 {program} 失败")
    except (OSError, ValueError, AttributeError):
        pass
    binary_dir.mkdir(parents=True, exist_ok=True)
    with _build_lock(progress, binary_dir, program):
        if _up_to_date(exe, stamp, source):                       # 等锁期间别的进程已经编好了
            return exe, ""
        if progress:
            progress("正在编译 C++ 后端（只在首次使用或源码更新后）")
        _STATE["compiled"] = True
        for leftover in binary_dir.glob(program + ".build-*"):    # 以前的编译被中途终止（取消计算）时留下的半成品
            try:
                leftover.unlink()
            except OSError:
                pass
        target = binary_dir / f"{program}.build-{os.getpid()}{'.exe' if os.name == 'nt' else ''}"
        log, message = [], ""
        try:
            for kind, compiler in compilers:
                command = _compile_command(kind, compiler, target, source)
                env = os.environ.copy()
                # 给出完整路径的编译器（例如 MinGW 的 g++.exe）要靠自己的目录找到配套的程序和运行库
                env["PATH"] = os.path.dirname(compiler) + os.pathsep + env.get("PATH", "")
                started = time.perf_counter()
                try:
                    code, out, err = _run_child(command, BUILD_TIMEOUT, cwd=str(binary_dir), env=env)
                except subprocess.TimeoutExpired:
                    code, out, err = -1, "", f"超过 {BUILD_TIMEOUT:g} s 未完成"
                except OSError as exc:
                    code, out, err = -1, "", str(exc)
                log.append(f"$ {' '.join(command)}\n返回码 {code}，{time.perf_counter() - started:.1f} s\n{out}{err}")
                if code == 0 and target.is_file():
                    _retry_while_in_use(lambda: os.replace(str(target), str(exe)))
                    try:        # 记下这个程序是由哪些源文件编出来的（见 _up_to_date）
                        exe.with_name(exe.name + ".sources.json").write_text(json.dumps(_built_from(exe, source)), encoding="utf-8")
                    except OSError:
                        pass
                    message = ""
                    break
                tail = (err or out).strip().splitlines()
                # 编译器输出的最后一行多半只是“1 error generated.”：取第一条报错（留它的末尾 200 个字符，开头是很长的文件路径），找不到再取最后一行
                detail = next((line for line in tail if "error" in line.lower() or "错误" in line), tail[-1] if tail else "")
                message = f"编译 {program} 失败（{Path(compiler).name}，返回码 {code}）" + (f"：{detail.strip()[-200:]}" if detail else "")
        finally:
            for leftover in (target, target.with_suffix(".obj")):
                try:
                    leftover.unlink()
                except OSError:
                    pass
            try:
                (binary_dir / (prefix + "build.log")).write_text("\n".join(log), encoding="utf-8")
            except OSError:
                pass
        if message:
            message += f"；详见 gui/bin/{prefix}build.log（排除原因后重新启动界面会再试一次）"
            try:
                failure.write_text(json.dumps({"signature": signature, "message": message}, ensure_ascii=False), encoding="utf-8")
            except OSError:
                pass
            return None, message
        try:
            failure.unlink()
        except OSError:
            pass
        return exe, ""


def ensure_executable(progress=None, retry_failed=False):
    """返回 (可执行文件的路径或 None, 说明)。需要时现场编译；不抛异常以外的错误都写在说明里。

    retry_failed：上次编译失败的记录不作数，再编一次（界面每次启动时用，免得许可证没接受、磁盘满了、
    编译超时这类与源码无关的失败被永远记住）。
    """
    override = os.environ.get("RCWA_JSON_EXE", "").strip()
    if override:
        if executable_path().is_file():
            return executable_path(), ""
        return None, f"找不到命令行程序：RCWA_JSON_EXE 指向的 {override} 不存在"
    exe = BIN / EXE_NAME
    stamp = _source_stamp()
    if stamp is None:                 # 没有源码：有现成的程序就用，没有就只能退回 Python
        if exe.is_file():
            return exe, ""
        return None, "找不到命令行程序 gui/bin/" + EXE_NAME + "，也找不到可供编译的 examples/rcwa_json.cpp"
    if _up_to_date(exe, stamp):
        return exe, ""
    try:
        return _build(exe, stamp, progress, retry_failed)
    except BackendError as exc:
        return None, str(exc)
    except OSError as exc:
        return None, f"编译 rcwa_json 时出错：{exc}"


def warmup():
    """界面启动时在后台线程里调用：提前把程序编好，并在终端里说明后端的状态。"""
    if forced_python():
        message = "C++ 后端：已由环境变量 RCWA_BACKEND=python 关闭，全部计算使用 Python 求解器。"
    else:
        exe, note = ensure_executable(retry_failed=True)
        if exe is None:
            message = f"C++ 后端不可用，全部计算使用 Python 求解器：{note}"
        else:
            message = (f"C++ 后端：{exe}" + (f"（{note}）" if note else "")
                       + ("；平面 TE/TM、圆锥入射与二维周期的衍射效率由它计算。" if backend_2d()[0] == "cpp"
                          else "；平面 TE/TM 与圆锥入射的衍射效率由它计算（二维周期默认用 Python，RCWA_2D_BACKEND=cpp 改用 C++）。"))
    # 路径里可能有控制台的编码（中文 Windows 上是 GBK）写不出的字符：换成 ?，不让这一行出错
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    message = message.encode(encoding, "replace").decode(encoding)
    sys.stdout.write(message + "\n")      # 一次写出整行，不与别的线程的输出交错
    sys.stdout.flush()


# --------------------------------------------------------------------------- 模型 → 请求
def _index(value):
    """界面模型里的各向同性折射率（数，或 "1.5+0.02j" 这样的字符串）→ complex；单轴晶体的写法返回 None。"""
    if isinstance(value, bool):
        raise BackendError("折射率格式无法识别")
    if isinstance(value, (int, float)):
        return complex(value)
    if isinstance(value, dict):
        return complex(value.get("real", 0), value.get("imag", 0))
    text = str(value)
    if "/" in text or "@" in text:
        return None
    try:
        return complex(text.strip().replace("i", "j").replace(" ", ""))
    except ValueError:
        raise BackendError("折射率格式无法识别：" + text) from None


MAX_HARMONICS_2D = 625          # 二维谐波总数的上限（与 gui_models_2d.normalize_2d 的入口限制相同）

# 二维周期模型默认交给哪个求解器（环境变量 RCWA_2D_BACKEND 可以改，取值见 BACKENDS_2D）。
# 默认是 "python"，理由是实测：C++ 的二维本征分解在每个 N 上都比 Python 慢——Mac、界面默认的 8 个线程、预设那样一个结构层加两层均匀膜、nx=360、nz=220，
# 取两次的较小值：N = 169 / 289 / 625 用 C++ 0.93 / 1.81 / 13.5 s，用 Python 0.69 / 1.72 / 9.24 s；单线程时 C++ 慢 1.2–2.1 倍（这些数字测于 eig_qr 多线程化之前）。
# C++ 路径另有一个代价：一次要写几百 MB 的结果文件（nx=1600、nz=1200 时 391 MB）。
# 改回 "cpp" 的条件（两条都要满足）：① C++ 的本征分解已经并行化；② 在界面默认线程数下，N = 169、289、625 三档的实测用时都不比 Python 慢。
# 满足之前，只有明确设了 RCWA_2D_BACKEND=cpp 的人才走二维的 C++ 路径；这个开关只管二维，一维的路由与这个常量无关。
DEFAULT_BACKEND_2D = "python"
BACKENDS_2D = ("python", "cpp")
NOTE_2D_PYTHON = "二维周期模型默认由 Python 求解器计算（环境变量 RCWA_2D_BACKEND=cpp 改用 C++）"


def backend_2d():
    """二维周期模型的求解器选择：(取值, 说明)。取值是 BACKENDS_2D 之一；说明在 RCWA_2D_BACKEND 不是有效取值时非空。
    不设或留空取 DEFAULT_BACKEND_2D；大小写和两端空白不计；无法识别的值（包括还没有实现的 auto）按默认处理，并在说明里写出来，
    免得拼错了的人以为自己已经打开了 C++。RCWA_BACKEND 不参与：它只有 python 一个有效取值（forced_python），取 cpp 也不会打开二维的 C++。"""
    raw = os.environ.get("RCWA_2D_BACKEND", "").strip()
    value = raw.lower()
    if not value:
        return DEFAULT_BACKEND_2D, ""
    if value in BACKENDS_2D:
        return value, ""
    return DEFAULT_BACKEND_2D, (f"环境变量 RCWA_2D_BACKEND={raw!r} 无法识别（可取 {' 或 '.join(BACKENDS_2D)}），按默认的 {DEFAULT_BACKEND_2D} 处理；"
                                + NOTE_2D_PYTHON)


def _eligibility_2d(cfg):
    """二维周期模型：RCWA_2D_BACKEND=cpp（默认不是，见 DEFAULT_BACKEND_2D），并且各向同性、解析 Fourier、N=(2M+1)(2My+1) ≤ 625 时可以交给 C++。

    开关先于下面的结构检查：默认（python）时一律不交给 C++，说明是 NOTE_2D_PYTHON 一句中性的话，不当作故障报告。

    从界面进来的二维模型一律先过 gui_models_2d.normalize_2d（gui_models.normalize），它在入口就拒绝：M、My 为负或 N>625、
    入射区吸收、单轴晶体、FFT 系数、背景材料覆盖、层类型不明，所以界面里的模型走到这里总是返回 (True, "")——
    二维没有“退给 Python 求解”的情形，这些模型是“二维不支持”，不会被算出来。下面的各个分支因此从界面不可达，
    只有绕过 normalize_2d、直接传入未规范化模型的调用者（库用法、测试）才会碰到；保留它们是纵深防御，
    免得规范化的约束以后放宽时把不支持的模型悄悄交给 C++。
    """
    choice, problem = backend_2d()
    if choice != "cpp":
        return False, problem or NOTE_2D_PYTHON
    inc, st = cfg["incident"], cfg["structure"]
    try:
        M, My = int(inc["M"]), int(inc["My"])
    except (KeyError, TypeError, ValueError):
        return False, "二维模型的截断阶数 M、My 无效，由 Python 求解器处理"
    if M < 0 or My < 0:
        return False, "二维截断阶数 M、My 不能为负，由 Python 求解器处理"
    harmonics = (2 * M + 1) * (2 * My + 1)
    if harmonics > MAX_HARMONICS_2D:
        return False, f"二维谐波总数 N=(2M+1)(2My+1)={harmonics} 超过 C++ 路径的上限 {MAX_HARMONICS_2D}，由 Python 求解器计算"
    for key, label in (("n_incident", "入射区"), ("n_substrate", "出射区")):
        n = _index(st[key])
        if n is None:
            return False, label + "含单轴晶体（各向异性材料），由 Python 求解器计算"
        if key == "n_incident" and n.imag != 0:
            return False, "入射区折射率有吸收，由 Python 求解器计算"
    for j, layer in enumerate(st["layers"]):
        name = f"第 {j + 1} 层“{layer.get('name', '')}”"
        kind = layer.get("kind")
        if kind == "uniform":
            values = [layer.get("n")]
        elif kind == "patterned":
            values = list(layer.get("n_values", []))
        elif kind == "patterned2d":
            values = [v for profile in layer.get("x_profiles", []) for v in profile.get("n_values", [])]
        else:
            return False, name + f"的类型 {kind!r} 不在 C++ 二维路径的范围内，由 Python 求解器计算"
        if any(_index(v) is None for v in values):
            return False, name + "含单轴晶体（各向异性材料），由 Python 求解器计算"
        if kind != "uniform" and (layer.get("fourier_method", "analytic") != "analytic" or layer.get("fft_samples", 0)):
            return False, name + "的 Fourier 系数取 FFT，由 Python 求解器计算"
        if layer.get("background_n") is not None:
            return False, name + "设有背景材料覆盖，由 Python 求解器计算"
    return True, ""


def eligibility(cfg):
    """这次计算能不能交给 C++：返回 (能否, 不能时的原因)。cfg 是界面规范化之后的模型。"""
    mode = cfg.get("mode")
    if mode == "pml":
        return False, "横向 PML 模型（不输出衍射效率）全部由 Python 求解器计算"
    if mode == "periodic2d":
        return _eligibility_2d(cfg)
    if mode not in ("periodic", "conical"):
        return False, f"求解方式 {mode} 由 Python 求解器计算"
    for j, layer in enumerate(cfg["structure"]["layers"]):
        name = f"第 {j + 1} 层“{layer.get('name', '')}”"
        values = [layer.get("n")] if layer.get("kind") == "uniform" else list(layer.get("n_values", []))
        if any(_index(v) is None for v in values):
            return False, name + "含单轴晶体（各向异性材料），由 Python 求解器计算"
    return True, ""


def _json_index(value):
    n = _index(value)
    return float(n.real) if n.imag == 0 else [float(n.real), float(n.imag)]


def _json_complex(value):
    n = complex(value)
    return float(n.real) if n.imag == 0 else [float(n.real), float(n.imag)]


EDGE_SNAP_FRACTION = 5e-10      # 首尾边界之差与周期的相对偏差超过它才贴齐（C++ 的限是 1e-9，留一半余量）


def _snap_edges(edges, period):
    """边界列表按周期贴齐：首尾之差偏离周期超过 EDGE_SNAP_FRACTION·周期时，把最后一个边界挪到“首个边界 + 周期”。

    界面（gui_models_2d._edges）接受 |首尾之差 − 周期| ≤ 1e-8·max(周期, 1)，C++ 协议与求解器只接受 ≤ 1e-9·周期
    （有意保持严格，不在 C++ 一侧放宽）。界面认可的模型因此不能整体退回 Python：这里只在请求里把最后一个边界贴齐。
    · 移动量就是那个偏差，不超过界面的容限 1e-8·max(周期, 1) µm（周期不超过 1 µm 时至多 1e-8 µm = 0.01 pm），远低于任何物理意义；
      C++ 与 Python 两条路径因此相差约“贴齐量/周期”的相对量级（实测：贴齐 4.9e-9 µm、周期 0.8 µm 时，六个场分量的最大差是峰值的 1.2e-8，
      效率差 4e-9；见 test_cpp_2d_backend.EdgeTolerance），而非逐位相同。
    · 偏差不超过 5e-10·周期的（含边界与周期精确相符的几何，这是常态）一个字节也不改。
      偏差在 5e-10·周期与 C++ 自己的限度 1e-9·周期之间的，C++ 本来会接受，这里也照样贴齐（移动量 ≤ 1e-9·周期，更小）：阈值取一半余量，
      不去逐个判断“C++ 会不会接受”。
    · 贴齐后边界不再严格递增（最后一段比贴齐量还窄）的不改，仍由 C++ 拒绝、退回 Python。
    · 已知的局限（不处理）：边界离原点极远而周期极小（例如首个边界 1e5 µm、周期 1e-6 µm，实际几乎不会出现）时，“首个边界 + 周期”本身的
      取整误差就超过 C++ 的 1e-9·周期，贴齐也过不了 C++ 的检查，这样的模型仍整体退回 Python（snap_probe 实测）。
    """
    edges = [float(e) for e in edges]
    if abs((edges[-1] - edges[0]) - period) > EDGE_SNAP_FRACTION * period:
        snapped = edges[0] + period
        if snapped > edges[-2]:
            edges[-1] = snapped
    return edges


def _build_request_2d(cfg, threads):
    """二维周期模型 → rcwa_json 的 periodic2d 请求（PROTOCOL_2D.md 第 1 节）。

    几何一律取自 gui_models_2d.build_stack_2d 得到的 Stack2D（与 Python 求解器吃的是同一个对象），这里不另写解析：
    折射率在界面里存成字符串（"1.8+0.13j"），到了 Stack2D 才是复数，请求里的数就是 Python 求解器实际使用的那一个。
    边界首尾与周期的微小偏差见 _snap_edges。
    """
    from gui_models_2d import build_stack_2d

    stack = build_stack_2d(cfg)
    inc = cfg["incident"]
    lx, ly = float(stack.lat.Lx), float(stack.lat.Ly)
    layers, structured = [], 0
    for layer in stack.layers:
        if layer.homogeneous:
            layers.append({"d": float(layer.d), "n": _json_complex(layer.n)})
        else:
            structured += 1
            layers.append({"d": float(layer.d), "y_edges": _snap_edges(layer.y_edges, ly),
                           "x_profiles": [{"edges": _snap_edges(edges, lx),
                                           "n_values": [_json_complex(n) for n in values]}
                                          for edges, values in layer.x_profiles]})
    threads = min(256, max(1, int(threads)))
    # 每个结构层一个 2N 阶本征分解；有两个以上互不相干的本征分解时让它们并行（结果与线程数逐位无关）
    return {"mode": "periodic2d",
            "stack": {"nI": float(stack.nI), "nII": _json_complex(stack.nII),
                      "Lx": float(stack.lat.Lx), "Ly": float(stack.lat.Ly), "layers": layers},
            "M": int(inc["M"]), "My": int(inc["My"]),
            "points": [{"wavelength": float(inc["wavelength"]), "theta_deg": float(inc["angle_deg"]),
                        "phi_deg": float(inc.get("phi_deg", 0.0))}],
            "input": {"psi_deg": float(inc["psi_deg"]), "delta_deg": float(inc["delta_deg"])},
            "options": {"eig": "qr", "factorization": "li", "homogeneous_fast_path": True,
                        "parallel_layers": bool(threads > 1 and structured >= 2), "threads": threads}}


def build_request(cfg, threads=1):
    """把界面的模型转成 rcwa_json 的请求（一个入射点）。"""
    if cfg.get("mode") == "periodic2d":
        return _build_request_2d(cfg, threads)
    inc, st = cfg["incident"], cfg["structure"]
    layers, gratings = [], 0
    for layer in st["layers"]:
        if layer["kind"] == "uniform":
            layers.append({"d": float(layer["thickness"]), "n": _json_index(layer["n"])})
        else:
            gratings += 1
            layers.append({"d": float(layer["thickness"]), "edges": [float(e) for e in layer["edges"]],
                           "n_values": [_json_index(v) for v in layer["n_values"]]})
            if layer.get("fourier_method", "analytic") == "fft":
                layers[-1].update(fourier_method="fft", fft_samples=int(layer.get("fft_samples", 0)))
    conical = cfg["mode"] == "conical"
    request = {"mode": "conical" if conical else "planar",
               "stack": {"nI": _json_index(st["n_incident"]), "nII": _json_index(st["n_substrate"]),
                         "period": float(st["period"]), "layers": layers},
               "M": int(inc["M"])}
    point = {"wavelength": float(inc["wavelength"]), "theta_deg": float(inc["angle_deg"])}
    if conical:
        point["phi_deg"] = float(inc.get("phi_deg", 0.0))
        request["input"] = {"psi_deg": float(inc.get("psi_deg", 90.0)), "delta_deg": float(inc.get("delta_deg", 0.0))}
    else:
        request["polarization"] = inc["polarization"]
    request["points"] = [point]
    threads = min(256, max(1, int(threads)))            # rcwa_json 接受 1 ~ 256
    # 只有一个入射点：线程用在求解内部；有两个以上互不相干的本征分解时让它们并行（结果与线程数逐位无关）。
    # 平面入射每个分段层一个本征分解，圆锥入射每个分段层两个。
    request["options"] = {"threads": threads, "parallel_layers": bool(threads > 1 and gratings >= (1 if conical else 2))}
    return request


# --------------------------------------------------------------------------- 调用与应答
def solve_timeout():
    try:
        value = float(os.environ.get("RCWA_CPP_TIMEOUT", "") or SOLVE_TIMEOUT)
    except ValueError:
        value = SOLVE_TIMEOUT
    # 上限 2e6 s（二十多天）：再大的数（包括 inf）标准库的等待函数接不住，会抛 OverflowError
    return min(value, 2e6) if value > 0 else SOLVE_TIMEOUT


def call(exe, request, workdir, timeout=None):
    """调用命令行程序，返回应答（ok 为真）；请求和应答另存在 workdir 里备查。

    请求走标准输入、应答走标准输出：这样与结果目录的路径里有什么字符无关。
    """
    workdir = Path(workdir)
    text = json.dumps(request)
    (workdir / "cpp_request.json").write_text(text, encoding="utf-8")
    response_file = workdir / "cpp_response.json"
    try:
        response_file.unlink()
    except OSError:
        pass
    timeout = solve_timeout() if timeout is None else timeout
    try:
        code, out, err = _run_child([str(exe)], timeout, text=text)
    except subprocess.TimeoutExpired:
        raise BackendError(f"C++ 程序超过 {timeout:g} s 没有返回，已终止") from None
    except OSError as exc:
        raise BackendError(f"无法运行 C++ 程序 {exe}：{exc}") from None
    try:
        if out.strip():
            response_file.write_text(out, encoding="utf-8")
        response = json.loads(out)
        if not isinstance(response, dict):
            raise ValueError
    except (OSError, ValueError):
        detail = (err or out).strip().splitlines()
        raise BackendError(f"C++ 程序没有给出可用的应答（返回码 {code}）" + (f"：{detail[-1][:200]}" if detail else "")) from None
    if not response.get("ok"):
        where = response.get("where")
        raise BackendError("C++ 程序不接受这次计算：" + str(response.get("error", "没有说明")) + (f"（{where}）" if where else ""))
    if code != 0:
        raise BackendError(f"C++ 程序的返回码是 {code}")
    return response


def _reals(values, count, what):
    if not isinstance(values, list) or len(values) != count:
        raise BackendError(f"C++ 应答里的 {what} 长度不对")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise _NumericalFailure(what)
    return [float(v) for v in values]


def _complexes(values, count, what):
    if not isinstance(values, list) or len(values) != count:
        raise BackendError(f"C++ 应答里的 {what} 长度不对")
    flat = []
    for pair in values:
        if not isinstance(pair, list) or len(pair) != 2:
            raise BackendError(f"C++ 应答里的 {what} 不是 [实部, 虚部]")
        flat.extend(pair)
    flat = _reals(flat, 2 * count, what)
    return [complex(flat[2 * j], flat[2 * j + 1]) for j in range(count)]


def _is_int(value):
    return type(value) is int


def _equal_number(value, expected):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and value == expected


# 二维应答自洽检查的容限（依据与数字见 validation/field_reference_2d/PROTOCOL_2D.md 第 6 节）。
# 取法：诚实应答的最大残差（见该节的误报研究：随机界面配置、随机协议级配置、命名压力配置、掠入射专项、Rayleigh 边缘、极端幅角）
# 与篡改应答里残差最小的一项之间，两侧各留几个数量级的余量。
EFFICIENCY_TOLERANCE_2D = 1e-12   # DE 与 Re kz·|E|²/(nI cosθ) 的相对差（以 max(1, 期望值) 为分母）；诚实应答最大 5.2e-16，DE×(1+1e-10) 的篡改是 1e-10
FAR_FIELD_TOLERANCE_2D = 1e-9     # s/p 振幅与远场 Ez 由 Ex、Ey 重算的最大差，相对“该级自己”的幅度（逐级，见 far_field_residual_2d）
KZ_TOLERANCE_2D = 1e-12           # kz 与解析式 sqrt(n²−Kx²−Ky²) 之差：|Δkz|·(|kz|+|kz_解析|) ≤ 它 × max(1, |n|², Kx²+Ky²)；诚实应答最大 7e-16。比 kz² 检查（1e-9，保持不变）紧三个数量级：kz 差 1e-10 而效率、Ez 同步改过的篡改只有它挡得住
PROPAGATING_THRESHOLD_2D = 1e-12  # 传播级的判据 Re kz > 它（合同第 2 节、rcwa2d.py:476）
# 所有“相对某个峰值”的残差，分母都不小于这个绝对下限。理由：双精度的最小正规数是 2.2e-308，低于约 1e-292 的数只剩不足 16 位有效数字
# （次正规数只剩几位：4e-319 的相对分辨率只有 1e-5），在它们上面做的“相对差”是量化噪声，不是误差。1e-200 比它大 90 多个数量级、
# 比入射振幅（界面里是 1）小 200 个数量级（e^{-460}），低于它的振幅在物理上就是 0。实例：金属膜“0.2+6j”、d=9.7 µm、λ=0.5 µm 的透射侧峰值是 4e-319，
# 没有下限时远场残差 2.3e-5，诚实应答被误拒。
AMPLITUDE_FLOOR_2D = 1e-200


def _kz_over_k0_2d(n, kx, ky):
    """rcwa2d._kz_over_k0 的纯 Python 版（顺序不能改）：先对复数 n²−Kx²−Ky² 开平方的主值，再把 Im < 0 的翻成 −kz。
    传播级是实正、倏逝级是 +i|kz|、吸收介质是 Re、Im 都 ≥ 0 的复数。（rcwa2d._branch 只作用在 q = −i·kz 上，对这一支是恒等变换。）"""
    kz = cmath.sqrt(complex(n * n - kx * kx - ky * ky))
    return -kz if kz.imag < 0. else kz


def _largest(*items):
    """非负数的最大值；其中有 NaN 就返回 NaN（内置 max 会因比较次序悄悄吞掉 NaN）。"""
    top = 0.
    for item in items:
        if item != item:
            return math.nan
        if item > top:
            top = item
    return top


def far_field_residual_2d(values, nI, nII, phi):
    """远场应答自洽性的相对残差：Rs、Rp、Ts、Tp 与远场 Ez 能不能由同一应答里的 Ex、Ey、Kx、Ky、kz 重算出来。

    公式与 rcwa2d.py:481–506 相同（合同第 2 节）：kt = hypot(Kx, Ky) > 1e-12 时 s = (−Ky Ex + Kx Ey)/kt、
    p = sign·(Kx Ex + Ky Ey)/(kt·cθ)；kt ≤ 1e-12（θ = 0 的零级）时改用入射方位角 φ；cθ = kz/n（|kz| ≤ 1e-12 时取 1）；
    传播级的远场 Ez = −sign·(Kx Ex + Ky Ey)/kz，其余级为 0。sign 在反射一侧是 −1、透射一侧是 +1。
    残差 = 逐级（每一侧的每个衍射级各算一次）的“s、p、Ez 三项差的最大者 ÷ 该级自己的幅度”，取所有级、两侧的最大者；
    “该级自己的幅度”是该级的 |Ex|、|Ey|、|Ez|、|s|、|p| 的最大者，且不小于 AMPLITUDE_FLOOR_2D。
    为什么逐级：同一侧的峰值会被一个被放大 1e150 倍的倏逝级（自洽地放大）抬高，其余各级的篡改就落到了容限以下而看不见。
    取各级自己的幅度不会增加误报：这些公式的舍入噪声正比于各项的绝对值（|Kx Ex|、|Ky Ey| 等），而它们不超过 |Ex|、|Ey| 乘 O(1)；
    掠射级（cθ → 0）里 p 本身就含 1/cθ，|p| 在分母里。实测见 PROTOCOL_2D.md 第 6 节。
    这是纯 Python 的 O(N)：cpp_backend 只用标准库。返回 inf 表示重算出非有限数（NaN 也按 inf 计，不被 max 吞掉）。
    """
    kx, ky = values["Kx"], values["Ky"]
    sin_phi, cos_phi = math.sin(phi), math.cos(phi)
    worst = 0.
    for sign, medium, kz_key, names in ((-1., nI, "kzI", ("Ex_r", "Ey_r", "Ez_r", "Rs", "Rp")),
                                        (1., nII, "kzII", ("Ex_t", "Ey_t", "Ez_t", "Ts", "Tp"))):
        kz_list, ex_list, ey_list, ez_list, s_list, p_list = (values[key] for key in (kz_key,) + names)
        for j in range(len(kx)):
            kz, ex, ey = kz_list[j], ex_list[j], ey_list[j]
            kt = math.hypot(kx[j], ky[j])
            cth = kz / medium if abs(kz) > 1e-12 else 1.
            if kt > 1e-12:
                s, p = (-ky[j] * ex + kx[j] * ey) / kt, sign * (kx[j] * ex + ky[j] * ey) / (kt * cth)
            else:
                s, p = -sin_phi * ex + cos_phi * ey, sign * (cos_phi * ex + sin_phi * ey) / cth
            ez = -sign * (kx[j] * ex + ky[j] * ey) / kz if kz.real > PROPAGATING_THRESHOLD_2D else 0.
            difference = _largest(abs(s_list[j] - s), abs(p_list[j] - p), abs(ez_list[j] - ez))
            scale = _largest(abs(ex), abs(ey), abs(ez_list[j]), abs(s_list[j]), abs(p_list[j]), AMPLITUDE_FLOOR_2D)
            residual = difference / scale
            if not math.isfinite(residual):
                return math.inf
            if residual > worst:
                worst = residual
    return worst


def _extract_2d(response, cfg):
    """二维应答（PROTOCOL_2D.md 第 2 节）：版本、模式、级次、状态、各数组的长度与有限性，以及数组之间必须自洽的关系。

    返回的 values 是纯 Python 列表：mn（[m, n] 的列表）、Kx、Ky、DEr、DEt、prop_r、prop_t，
    kzI、kzII、Rs、Rp、Ts、Tp、Ex_r…Ez_t（complex 的列表）。成员的次序就是 harmonic_index_2d（m 在外层、n 在内层）。

    除格式外还核对应答自己的物理自洽：Kx、Ky 与 kz 都按请求独立重算（kz 的取支与 Re、Im ≥ 0，传播标志取自解析 kz，不取自应答自己的 kz）、
    效率与振幅的关系（容限 EFFICIENCY_TOLERANCE_2D）、逐级的 Rs、Rp、Ts、Tp 与远场 Ez 对 Ex、Ey 的关系（far_field_residual_2d）。
    它们对“形式正确、数值被悄悄改过”的应答也有效；但自洽的整套错误（例如求解器内部把两层的厚度对调、或把某层折射率读错，而应答照抄请求）
    看不出来，那要靠 RCWA_VERIFY_PYTHON=1 的独立求解。
    """
    inc, st = cfg["incident"], cfg["structure"]
    M, My = int(inc["M"]), int(inc["My"])
    N = (2 * M + 1) * (2 * My + 1)
    try:
        if response.get("ok") is not True:           # 真值（1、"yes"）不算：协议写的是布尔 true
            raise BackendError(f"C++ 应答的 ok 是 {response.get('ok')!r}，应为 true")
        if not _is_int(response.get("version")) or response.get("version") != 1:
            raise BackendError(f"C++ 应答的格式版本是 {response.get('version')!r}，本界面只认 1（命令行程序与界面不是同一版）")
        if response.get("mode") != "periodic2d" or not _is_int(response.get("M")) or response.get("M") != M \
                or not _is_int(response.get("My")) or response.get("My") != My:
            raise BackendError("C++ 应答的模式或截断阶数与请求不符")
        orders = response.get("orders")
        expected_orders = [[m, n] for m in range(-M, M + 1) for n in range(-My, My + 1)]
        if (not isinstance(orders, list) or len(orders) != N
                or any(not isinstance(row, list) or len(row) != 2 or not all(_is_int(v) for v in row) for row in orders)
                or orders != expected_orders):
            raise BackendError("C++ 应答的级次表与请求的截断不符（次序应为 m 在外层、n 在内层）")
        results = response["results"]
        if not isinstance(results, list) or len(results) != 1:
            raise BackendError("C++ 应答的结果个数不是 1")
        result = results[0]
        point = {"wavelength": float(inc["wavelength"]), "theta_deg": float(inc["angle_deg"]),
                 "phi_deg": float(inc.get("phi_deg", 0.0))}
        if any(not _equal_number(result.get(key), value) for key, value in point.items()):
            raise BackendError("C++ 应答里的入射点与请求不符")
        status = result.get("status")
        if status != "ok":
            raise BackendError(f"C++ 二维求解未成功（status={status!r}：{result.get('message', '')}）")
        values = {"timing": result.get("timing") or {}, "threads": response.get("threads"),
                  "mn": [list(row) for row in orders]}
        try:
            for key in ("Kx", "Ky", "DEr", "DEt"):
                values[key] = _reals(result[key], N, key)
            for key in ("kzI", "kzII", "Rs", "Rp", "Ts", "Tp", "Ex_r", "Ey_r", "Ez_r", "Ex_t", "Ey_t", "Ez_t"):
                values[key] = _complexes(result[key], N, key)
        except _NumericalFailure as exc:
            raise BackendError(f"C++ 求解出现数值失败（{exc} 含非有限数，eig_info={result.get('eig_info')}）") from None
        for key in ("prop_r", "prop_t"):
            flags = result[key]
            if not isinstance(flags, list) or len(flags) != N or any(not _is_int(v) or v not in (0, 1) for v in flags):
                raise BackendError(f"C++ 应答里的 {key} 不是长度为 N 的 0/1 数组")
            values[key] = [bool(v) for v in flags]
        # 数组之间的关系是合同写死的：Kx、Ky 由入射方向与级次决定（这里按请求重算，不取应答里的数），kz 是 n²−Kx²−Ky² 的那一支平方根
        # （传播级实正、倏逝级 +i|kz|、吸收介质 Re、Im ≥ 0），传播判据 Re kz > 1e-12，非传播级效率为 0，
        # 传播级效率 = Re kz·(|Ex|²+|Ey|²+|Ez|²)/(nI cos θ)。对不上说明数组错位或被改动。
        # kz 的检查有四层：kz² 对 n²−Kx²−Ky²（只定到正负号）；Im kz ≥ 0 与 Re kz ≥ 0（定下是哪一支：正负号都对不上的“自洽”改动，
        # 例如传播级 kz 取负实数、同时把 prop、DE、Ez 改得与它一致，只能由这一层挡住）；kz 与独立重算的解析式 _kz_over_k0_2d 逐级比较（含正负号）；
        # prop 标志与“解析 kz 的实部是否 > 1e-12”比较——标志不取自应答自己的 kz。解析 kz 的实部恰在 1e-12 附近，或 |n²−Kx²−Ky²| 小到与舍入
        # 噪声同量级（kz 不足 1e-6，掠射到 Rayleigh 边缘）时，两个程序的取整可以落在阈值两侧：这时只要求标志与应答自己的 kz 一致。
        nI, nII = float(st["n_incident"]), _index(st["n_substrate"])
        theta, phi = math.radians(point["theta_deg"]), math.radians(point["phi_deg"])
        kx0, ky0 = nI * math.sin(theta) * math.cos(phi), nI * math.sin(theta) * math.sin(phi)
        wavelength, lx, ly = point["wavelength"], float(st["period"]), float(st["period_y"])
        den = nI * math.cos(theta)
        for j, (m, n) in enumerate(expected_orders):
            kx, ky = kx0 - m * wavelength / lx, ky0 - n * wavelength / ly
            if abs(values["Kx"][j] - kx) > 1e-11 * max(1., abs(kx)) or abs(values["Ky"][j] - ky) > 1e-11 * max(1., abs(ky)):
                raise BackendError("C++ 应答里的 Kx、Ky 与级次、入射方向不符")
            for kz_key, n_medium, flag_key, amp_keys, de_key in (
                    ("kzI", nI, "prop_r", ("Ex_r", "Ey_r", "Ez_r"), "DEr"),
                    ("kzII", nII, "prop_t", ("Ex_t", "Ey_t", "Ez_t"), "DEt")):
                kz = values[kz_key][j]
                scale = max(1., abs(n_medium) ** 2, kx * kx + ky * ky)
                if abs(kz * kz - (n_medium * n_medium - kx * kx - ky * ky)) > 1e-9 * scale:
                    raise BackendError(f"C++ 应答里的 {kz_key} 与 n²−Kx²−Ky² 不符")
                if kz.imag < 0.:      # kz² 只定到正负号：倏逝级取 +i|kz|（衰减）。共轭过来的 −i|kz| 过得了上面的平方检查，必须单独拒绝
                    raise BackendError(f"C++ 应答里的 {kz_key} 虚部为负（合同取 Im kz ≥ 0 的一支：倏逝级是 +i|kz|）")
                analytic = _kz_over_k0_2d(n_medium, kx, ky)
                if kz.real < 0. <= analytic.real:    # 负实部的 kz 同样满足平方检查；传播级的合同取支是实正（Re kz ≥ 0）
                    raise BackendError(f"C++ 应答里的 {kz_key} 实部为负（合同取 Re kz ≥ 0 的一支：传播级是实正数）")
                if not abs(kz - analytic) * (abs(kz) + abs(analytic)) <= KZ_TOLERANCE_2D * scale:
                    raise BackendError(f"C++ 应答里的 {kz_key} 与独立重算的 sqrt(n²−Kx²−Ky²) 不符（容限 {KZ_TOLERANCE_2D:g}）")
                propagating = kz.real > PROPAGATING_THRESHOLD_2D
                if values[flag_key][j] != propagating:
                    raise BackendError(f"C++ 应答里的 {flag_key} 与 Re({kz_key}) > 1e-12 的传播判据不符")
                s = n_medium * n_medium - kx * kx - ky * ky
                near_threshold = (abs(analytic.real - PROPAGATING_THRESHOLD_2D) <= 1e-13 * max(1., abs(analytic))
                                  or abs(s) <= 1e-12 * scale)
                if not near_threshold and values[flag_key][j] != (analytic.real > PROPAGATING_THRESHOLD_2D):
                    raise BackendError(f"C++ 应答里的 {flag_key} 与解析 kz 的传播判据不符（标志不能取自应答自己的 kz）")
                efficiency = values[de_key][j]
                if not propagating:
                    if efficiency != 0.:
                        raise BackendError(f"C++ 应答里非传播级的 {de_key} 不为 0")
                else:
                    expected = kz.real * sum(abs(values[a][j]) ** 2 for a in amp_keys) / den
                    if efficiency < 0. or abs(efficiency - expected) > EFFICIENCY_TOLERANCE_2D * max(1., abs(expected)):
                        raise BackendError(f"C++ 应答里的 {de_key} 与各级振幅、kz 算出的效率不符（容限 {EFFICIENCY_TOLERANCE_2D:g}）")
        residual = far_field_residual_2d(values, nI, nII, phi)
        if not residual <= FAR_FIELD_TOLERANCE_2D:       # 写成取反，NaN 也算不符
            raise BackendError(f"C++ 应答里的 Rs、Rp、Ts、Tp 或远场 Ez 与 Ex、Ey 重算的结果不符"
                               f"（相对残差 {residual:.1e}，容限 {FAR_FIELD_TOLERANCE_2D:g}）")
        return values
    except (KeyError, IndexError, TypeError, AttributeError, ZeroDivisionError, OverflowError) as exc:
        raise BackendError(f"C++ 应答缺少需要的内容或含无法计算的数（{type(exc).__name__}：{exc}）") from None


def extract(response, cfg):
    """从应答里取出界面要用的量（纯 Python 的列表）；数值失败（null）或格式不符抛 BackendError。"""
    if cfg.get("mode") == "periodic2d":
        return _extract_2d(response, cfg)
    M = int(cfg["incident"]["M"])
    N = 2 * M + 1
    conical = cfg["mode"] == "conical"
    try:
        if response.get("version") != 1:
            raise BackendError(f"C++ 应答的格式版本是 {response.get('version')!r}，本界面只认 1（命令行程序与界面不是同一版）")
        if response.get("mode") != ("conical" if conical else "planar") or response.get("orders") != list(range(-M, M + 1)):
            raise BackendError("C++ 应答的模式或级次与请求不符")
        result = response["results"][0]
        values = {"timing": result.get("timing") or {}, "threads": response.get("threads")}
        try:
            if conical:
                pol = result["polarized"]
                for key in ("DEr", "DEt", "DEr_s", "DEr_p", "DEt_s", "DEt_p"):
                    values[key] = _reals(pol[key], N, key)
                for key in ("Rs", "Rp", "Ts", "Tp"):
                    values[key] = _complexes(pol[key], N, key)
                values["jones_in"] = _complexes(pol["jones_in"], 2, "jones_in")
                for key in ("flux_r", "flux_t"):
                    values[key] = _reals(result[key], N, key)
                for key in ("Jr", "Jt"):        # 每级 [ss, sp, ps, pp]
                    rows = result[key]
                    if not isinstance(rows, list) or len(rows) != N:
                        raise BackendError(f"C++ 应答里的 {key} 长度不对")
                    values[key] = [_complexes(row, 4, key) for row in rows]
            else:
                values["DEr"] = _reals(result["DEr"], N, "DEr")
                values["DEt"] = _reals(result["DEt"], N, "DEt")
        except _NumericalFailure as exc:
            raise BackendError(f"C++ 求解出现数值失败（{exc} 含非有限数，eig_info={result.get('eig_info')}）") from None
        return values
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise BackendError(f"C++ 应答缺少需要的内容（{type(exc).__name__}：{exc}）") from None


# --------------------------------------------------------------------------- 给 gui_worker.py 用的入口
class Attempt:
    """一次计算里 C++ 这一路的经过与结果。used 为真时效率取自 C++。"""

    def __init__(self, cfg):
        self.mode = cfg.get("mode")
        self.used = False
        self.note = ""
        self.values = None
        self.seconds = None           # 调用命令行程序的墙上时间（含进程启动与读写 JSON）
        self.build_seconds = None     # 这次计算里花在查找、编译程序上的时间（有编译时才值得看）
        self.elapsed = 0.0            # 这一路总共花的时间
        self.difference = None        # 两个求解器各级效率之差的最大值
        self.jones_difference = None  # 圆锥：传播级 Jones 矩阵元之差的最大值，|ΔJ| / max(1, |J|)
        # 圆锥入射特有的两项，C++ 给出了应答才有（数值失败、退回 Python 时也照样记下）：
        self.answered = False                 # 应答里有没有这两项
        self.min_abs_lambda = None            # 光栅层全部本征值的模的最小值；没有做本征分解的层时是 None
        self.degenerate_mode_warning = None   # True：约化模式接近退化
        self.n_schur = None                   # 新版 C++ 使用 Schur 传播的层数；旧版应答不含此项
        self.python_n_schur = 0
        self.python_zero_modes = 0

    def note_conical(self, response):
        """记下圆锥应答里的 min_abs_lambda 与 degenerate_mode_warning。

        含义见 StableRCWA 的 rcwa1d_conical.hpp：圆锥入射时，某个光栅层的层矩阵出现接近零的本征值，
        两个模式就几乎重合；新版程序可用 Schur 传播恢复远场。旧版应答没有 n_schur 时保留原有告警语义。
        """
        try:
            result = response["results"][0]
            warning, smallest = result["degenerate_mode_warning"], result["min_abs_lambda"]
        except (KeyError, IndexError, TypeError):
            return
        if not isinstance(warning, bool):
            return
        self.answered = True
        self.degenerate_mode_warning = warning
        n_schur = result.get("n_schur")
        if isinstance(n_schur, int) and not isinstance(n_schur, bool) and n_schur >= 0:
            self.n_schur = n_schur
        if isinstance(smallest, (int, float)) and not isinstance(smallest, bool) and math.isfinite(smallest):
            self.min_abs_lambda = float(smallest)             # 应答里的 null（+∞）记为 None

    def merge(self, res):
        """返回一份把效率（圆锥还有 Jones 矩阵、s/p 振幅、通量因子）换成 C++ 数值的 res 浅拷贝，并算出两边的差。

        res 本身不动：层内场图仍按 Python 的解重建。C++ 没用上时原样返回 res。
        """
        self.python_n_schur = int(res.get("n_schur", 0))
        self.python_zero_modes = int(res.get("n_zero_modes", 0))
        if not self.used:
            return res
        try:
            import numpy as np
            v = self.values
            if self.mode == "periodic2d":
                return self._merge_2d(res, v)
            N = len(v["DEr"])
            if len(res["ms"]) != N or int(res["ms"][0]) != -(N // 2):
                raise BackendError("两个求解器的级次排列不一致")
            eff = type(res)(res)
            pairs = []
            for key in ("DEr", "DEt"):
                eff[key] = np.asarray(v[key], float)
                pairs.append((eff[key], np.asarray(res[key], float)))
            if self.mode == "conical":
                for key in ("flux_r", "flux_t"):
                    eff[key] = np.asarray(v[key], float)
                for key in ("Rs", "Rp", "Ts", "Tp"):
                    eff[key] = np.asarray(v[key], complex)
                jones = 0.0
                for key, flux in (("Jr", "flux_r"), ("Jt", "flux_t")):
                    eff[key] = np.asarray(v[key], complex).reshape(N, 2, 2)      # [级次, 出射 s/p, 入射 s/p]
                    ref = np.asarray(res[key], complex)
                    live = np.asarray(res[flux]) > 0                              # 只比传播级
                    if live.any():
                        jones = max(jones, float(np.max(abs(eff[key][live] - ref[live]) / np.maximum(1.0, abs(ref[live])))))
                self.jones_difference = jones if math.isfinite(jones) else None
                for key, flux, amp in (("DEr_s", "flux_r", "Rs"), ("DEr_p", "flux_r", "Rp"),
                                       ("DEt_s", "flux_t", "Ts"), ("DEt_p", "flux_t", "Tp")):
                    pairs.append((np.asarray(v[key], float), np.asarray(res[flux]) * abs(np.asarray(res[amp])) ** 2))
            worst = max(float(np.max(abs(a - b))) for a, b in pairs)
            self.difference = worst if math.isfinite(worst) else None
            if self.difference is None:
                self.note = _join(self.note, "Python 求解器的效率含非有限数，无法互相核对")
            elif self.difference > LARGE_DIFFERENCE:
                self.note = _join(self.note, f"两个求解器的效率相差超过 {LARGE_DIFFERENCE:g}，请检查 M 是否足够、参数是否靠近 Rayleigh 异常点")
            return eff
        except Exception as exc:
            self.used = False
            self.note = _join(self.note, f"C++ 的结果没能并入（{type(exc).__name__}：{exc}），已退回 Python")
            return res

    def _merge_2d(self, res, v):
        """二维：把 res（Python 的 Result2D）的各级效率、s/p 振幅、kz 换成 C++ 的数，并算出两边各级效率之差。

        比较的量与结果页上的一致：各级 DEr、DEt，以及 s、p 通道各自的效率 fr·|Rs|² 等（fr = max(0, Re kzI/(nI cos θ))）。
        """
        import numpy as np
        mn = np.asarray(v["mn"])
        if mn.shape != np.shape(res["mn"]) or not np.array_equal(mn, res["mn"]):
            raise BackendError("两个求解器的级次排列不一致")
        eff = type(res)(res)
        for key in ("Kx", "Ky", "DEr", "DEt"):
            eff[key] = np.asarray(v[key], float)
        for key in ("kzI", "kzII", "Rs", "Rp", "Ts", "Tp", "Ex_r", "Ey_r", "Ez_r", "Ex_t", "Ey_t", "Ez_t"):
            eff[key] = np.asarray(v[key], complex)
        for key in ("prop_r", "prop_t"):
            eff[key] = np.asarray(v[key], bool)
        pairs = [(eff["DEr"], np.asarray(res["DEr"], float)), (eff["DEt"], np.asarray(res["DEt"], float))]
        den = float(res["nI"]) * np.cos(res["theta"])
        for kz, amplitude in (("kzI", "Rs"), ("kzI", "Rp"), ("kzII", "Ts"), ("kzII", "Tp")):
            channel = lambda r: np.maximum(0., np.real(r[kz]) / den) * abs(r[amplitude]) ** 2
            pairs.append((channel(eff), channel(res)))
        worst = max(float(np.max(abs(a - b))) for a, b in pairs)
        self.difference = worst if math.isfinite(worst) else None
        if self.difference is None:
            self.note = _join(self.note, "Python 求解器的效率含非有限数，无法互相核对")
        elif self.difference > LARGE_DIFFERENCE:
            self.note = _join(self.note, f"两个求解器的效率相差超过 {LARGE_DIFFERENCE:g}，请检查 M、My 是否足够、参数是否靠近 Rayleigh 异常点")
        return eff

    def report(self, python_solve_seconds):
        """metrics.json 里的 backend 一段。"""
        block = {"efficiency": "cpp" if self.used else "python", "fields": "python",
                 "cpp_seconds": self.seconds, "python_solve_seconds": float(python_solve_seconds),
                 "max_efficiency_difference": self.difference if self.used else None, "note": self.note,
                 "python_n_schur": self.python_n_schur, "python_zero_modes": self.python_zero_modes}
        if self.used:
            timing = self.values.get("timing") or {}
            total = timing.get("total")
            if isinstance(total, (int, float)) and not isinstance(total, bool) and math.isfinite(total):
                block["cpp_solve_seconds"] = float(total)      # C++ 自己报告的求解用时（不含进程启动与读写 JSON）
            threads = self.values.get("threads")
            if isinstance(threads, (int, float)) and not isinstance(threads, bool) and math.isfinite(threads):
                block["cpp_threads"] = int(threads)
            if self.jones_difference is not None:
                block["max_jones_difference"] = self.jones_difference
        if self.answered:                 # 圆锥：C++ 给出了应答（用上了，或者数值失败退回了 Python）
            block["min_abs_lambda"] = self.min_abs_lambda
            block["degenerate_mode_warning"] = self.degenerate_mode_warning
            if self.n_schur is not None:
                block["n_schur"] = self.n_schur
        if self.build_seconds is not None:
            block["build_seconds"] = self.build_seconds
        return block


def _join(first, second):
    return f"{first}；{second}" if first else second


def attempt(cfg, outdir, progress=None):
    """符合条件时调用 C++，得到一个 Attempt；任何失败都只体现在 note 里，不向外抛。

    progress：可选的回调，参数是一句进度说明（编译、求解时各调一次）。
    """
    result = Attempt(cfg)
    started = time.perf_counter()
    try:
        if forced_python():
            result.note = "环境变量 RCWA_BACKEND=python：只用 Python 求解器"
            return result
        fits, reason = eligibility(cfg)
        if not fits:
            result.note = reason
            return result
        with _children_die_with_us():
            _STATE["compiled"] = False
            exe, note = ensure_executable(progress)
            if _STATE["compiled"]:
                result.build_seconds = time.perf_counter() - started
            if exe is None:
                result.note = note
                return result
            if progress:
                progress("正在用 C++ 计算衍射效率")
            threads = os.environ.get("RCWA_BLAS_THREADS", "")
            request = build_request(cfg, int(threads) if threads.isdigit() else 1)
            called = time.perf_counter()
            try:
                response = call(exe, request, outdir)
            finally:
                result.seconds = time.perf_counter() - called
        if result.mode == "conical":
            result.note_conical(response)
        result.values = extract(response, cfg)
        result.used = True
        result.note = note
    except BackendError as exc:
        result.note = str(exc)
    except Exception as exc:          # 后端的意外不应当让整次计算失败
        result.note = f"C++ 后端出错（{type(exc).__name__}：{exc}），已退回 Python"
    finally:
        result.elapsed = time.perf_counter() - started
    return result
