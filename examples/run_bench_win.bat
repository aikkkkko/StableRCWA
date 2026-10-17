@echo off
setlocal
rem ---------------------------------------------------------------------------
rem Build examples\bench_accel.cpp on Windows and run it, keeping the machine
rem load before and after the run next to the numbers.
rem This file is plain ASCII on purpose: cmd.exe reads .bat files in the console
rem code page, so non-ASCII comments would break it.
rem
rem   examples\run_bench_win.bat [compiler] [placement] [bench_accel options]
rem       compiler    mingw (default) or msvc
rem       placement   default (default), percore or both - see below
rem
rem   examples\run_bench_win.bat                    MinGW g++ -O2, default placement, full benchmark
rem   examples\run_bench_win.bat msvc               the same with MSVC (cl /O2)
rem   examples\run_bench_win.bat mingw percore      one thread per physical core (see below)
rem   examples\run_bench_win.bat mingw both         default placement, then percore
rem   examples\run_bench_win.bat mingw both --quick            everything after these words
rem   examples\run_bench_win.bat msvc --only breakdown         goes to bench_accel unchanged
rem
rem Thread placement (the second word):
rem   default   Windows places the threads itself. With SMT (two logical
rem             processors per physical core) it often puts two busy threads on
rem             the two logical processors of ONE core while other cores idle,
rem             and it does so differently from run to run. A thread count up to
rem             the number of cores then scales worse, and less repeatably, than
rem             the hardware allows. Thread counts: 1, 2, 4, ... up to the number
rem             of logical processors.
rem   percore   The benchmark process is started with "start /affinity <mask>",
rem             the mask selecting one logical processor of every physical core,
rem             so no two threads can share a core. Thread counts: 1, 2, 4, ...
rem             up to the number of physical cores (more threads than that would
rem             only queue up on the same processors).
rem             The mask is derived from the core and logical-processor counts,
rem             assuming the logical processors of core k are numbered 2k and
rem             2k+1 (the usual Windows numbering; checked on the Ryzen 9 5950X
rem             this was written on: 16 cores, mask 55555555). If the counts do
rem             not fit that picture (e.g. a CPU with two kinds of cores, or
rem             more than 64 logical processors) set AFFINITY yourself.
rem   both      runs default first, then percore.
rem This is a way of starting the program only: the library never sets
rem affinities itself.
rem
rem Output in build\bench\ :
rem   <TAG>.json   the JSON object bench_accel writes to stdout
rem   <TAG>.log    load before the run, the placement used, progress and raw
rem                readings of every round (stderr of bench_accel), load after
rem                the run, exit code
rem TAG defaults to bench_win (MinGW) or bench_win_msvc (MSVC); the percore run
rem writes <TAG>_percore.json and <TAG>_percore.log.
rem
rem Required environment (no defaults; only the one for the chosen compiler):
rem   MINGW_BIN           the MinGW-w64 bin directory that holds g++.exe (mingw)
rem   VCVARS              the full path of vcvars64.bat from Visual Studio (msvc)
rem
rem Environment overrides:
rem   THREADS             thread counts for the scaling tests, e.g. 1,2,4,8
rem                       (default: see above; the percore run drops the counts
rem                       larger than the number of physical cores)
rem   MACHINE             CPU name written into the JSON (default: asked from Windows)
rem   TAG                 base name of the output files
rem   IDLE_BELOW          the run starts once the CPU load is below this percentage
rem                       (default 10; it gives up waiting after about 3 minutes
rem                       and runs anyway - the log shows every sample)
rem   AFFINITY            hexadecimal processor mask as in "start /affinity", used
rem                       by the percore run instead of the derived one. When it
rem                       is set and no placement word is given, the script does
rem                       one run with that mask and the plain <TAG> names.
rem   GXX_OPT             optimisation options for g++ (default -O2); several are
rem                       allowed, e.g.  set "GXX_OPT=-O3 -march=native"
rem
rem Exit code: that of bench_accel (of the last run that failed, with "both"),
rem 1 if the build failed, 2 for a usage error or when the compiler's MINGW_BIN /
rem VCVARS is not set.
rem Needs powershell.exe (part of Windows) for the CPU name, topology and load.
rem Limitations: tried from a short ASCII path and from a directory whose name
rem contains a space. Paths with non-ASCII characters, or with percent signs,
rem carets, ampersands or quotes, are not supported.
rem ---------------------------------------------------------------------------

if "%IDLE_BELOW%"=="" set "IDLE_BELOW=10"

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "SRC=%ROOT%\examples\bench_accel.cpp"
set "OUT=%ROOT%\build\bench"
if not exist "%OUT%" mkdir "%OUT%"

rem ---- command line: leading words pick the compiler and the placement, the
rem      rest (kept as typed, commas included) goes to bench_accel
set "MODE=mingw"
set "PLACE="
set "ARGS=%*"
:strip
if not defined ARGS goto stripped
set "WORD="
set "REST="
for /f "tokens=1*" %%A in ("%ARGS%") do set "WORD=%%A"& set "REST=%%B"
set "KNOWN="
if /i "%WORD%"=="mingw"   set "MODE=mingw"& set "KNOWN=1"
if /i "%WORD%"=="msvc"    set "MODE=msvc"& set "KNOWN=1"
if /i "%WORD%"=="default" set "PLACE=default"& set "KNOWN=1"
if /i "%WORD%"=="percore" set "PLACE=percore"& set "KNOWN=1"
if /i "%WORD%"=="both"    set "PLACE=both"& set "KNOWN=1"
if not defined KNOWN goto stripped
set "ARGS=%REST%"
goto strip
:stripped
rem The first word left over must be an option of bench_accel.
if defined ARGS if not "%ARGS:~0,2%"=="--" (
    echo usage: run_bench_win.bat [mingw^|msvc] [default^|percore^|both] [bench_accel options]
    exit /b 2
)
set "LEGACY_MASK="
if not defined PLACE if not "%AFFINITY%"=="" set "LEGACY_MASK=1"
if not defined PLACE set "PLACE=default"
if /i "%MODE%"=="mingw" if not defined MINGW_BIN goto no_mingw_bin
if /i "%MODE%"=="msvc" if not defined VCVARS goto no_vcvars

rem ---- machine: CPU name, cores, logical processors, one-per-core mask, thread lists
if "%MACHINE%"=="" for /f "usebackq delims=" %%N in (`powershell -NoProfile -Command "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name.Trim()"`) do set "MACHINE=%%N"
if "%MACHINE%"=="" set "MACHINE=unknown CPU"
set "CORES=?"
set "LOGICAL=?"
set "MASK=0"
set "THREADS_DEFAULT=%THREADS%"
set "THREADS_PERCORE=%THREADS%"
for /f "usebackq tokens=1-5" %%A in (`powershell -NoProfile -Command "$p = @(Get-CimInstance Win32_Processor); $c = ($p | Measure-Object NumberOfCores -Sum).Sum; $l = ($p | Measure-Object NumberOfLogicalProcessors -Sum).Sum; $step = 0; if ($l -eq 2 * $c) { $step = 2 } elseif ($l -eq $c) { $step = 1 }; $m = [uint64]0; if ($step -gt 0 -and $l -le 64) { for ($k = 0; $k -lt $c; $k++) { $m = $m -bor ([uint64]1 -shl ($step * $k)) } }; $td = @(); for ($n = 1; $n -lt $l; $n *= 2) { $td += $n }; $td += $l; $tp = @(); for ($n = 1; $n -lt $c; $n *= 2) { $tp += $n }; $tp += $c; $u = '%THREADS%'; if ($u) { $td = $u.Split(','); $tp = @($u.Split(',') | Where-Object { [int]$_ -le $c }) }; '{0} {1} {2:X} {3} {4}' -f $c, $l, $m, ($td -join ','), ($tp -join ',')"`) do set "CORES=%%A"& set "LOGICAL=%%B"& set "MASK=%%C"& set "THREADS_DEFAULT=%%D"& set "THREADS_PERCORE=%%E"
set "MASK_FROM=derived: one logical processor per physical core"
if not "%AFFINITY%"=="" set "MASK=%AFFINITY%"
if not "%AFFINITY%"=="" set "MASK_FROM=given in AFFINITY"
if "%THREADS_DEFAULT%"=="" set "THREADS_DEFAULT=1,2,4,8"
if "%THREADS_PERCORE%"=="" set "THREADS_PERCORE=1"
if defined LEGACY_MASK set "THREADS_PERCORE=%THREADS_DEFAULT%"
if /i "%PLACE%"=="default" goto build
if not "%MASK%"=="0" goto build
echo cannot derive a one-per-core mask from %CORES% cores / %LOGICAL% logical processors: set AFFINITY
exit /b 2

:build
if /i "%MODE%"=="msvc" goto build_msvc
if "%TAG%"=="" set "TAG=bench_win"
set "EXE=%OUT%\bench_accel.exe"
rem g++ needs its own bin directory on PATH (the compiler proper loads DLLs from
rem there). -static: the exe itself does not need the MinGW runtime DLLs.
set "PATH=%MINGW_BIN%;%PATH%"
if "%GXX_OPT%"=="" set "GXX_OPT=-O2"
"%MINGW_BIN%\g++.exe" -std=c++17 %GXX_OPT% -Wall -Wextra -Wpedantic "-DBENCH_OPT=%GXX_OPT%" -I "%ROOT%\include" -I "%ROOT%\examples" "%SRC%" -o "%EXE%" -static
if errorlevel 1 (
    echo [mingw] build FAILED
    exit /b 1
)
goto run_all

:build_msvc
if "%TAG%"=="" set "TAG=bench_win_msvc"
set "EXE=%OUT%\bench_accel_msvc.exe"
if not exist "%VCVARS%" (
    echo [msvc] "%VCVARS%" not found
    exit /b 1
)
call "%VCVARS%" >nul
cl /nologo /std:c++17 /O2 /EHsc /W4 /utf-8 /DBENCH_OPT=/O2 /I "%ROOT%\include" /I "%ROOT%\examples" "%SRC%" /Fe:"%EXE%" /Fo:"%OUT%\bench_accel_msvc.obj"
if errorlevel 1 (
    echo [msvc] build FAILED
    exit /b 1
)

:run_all
set "FINAL_RC=0"
if defined LEGACY_MASK (
    call :run "%TAG%" "%MASK%" "%THREADS_PERCORE%"
    goto finish
)
if /i not "%PLACE%"=="percore" call :run "%TAG%" "" "%THREADS_DEFAULT%"
if /i not "%PLACE%"=="default" call :run "%TAG%_percore" "%MASK%" "%THREADS_PERCORE%"
:finish
exit /b %FINAL_RC%

:no_mingw_bin
echo MINGW_BIN is not set. Set it to the MinGW-w64 bin directory that holds g++.exe,
echo for example:  set "MINGW_BIN=C:\mingw64\bin"
exit /b 2
:no_vcvars
echo VCVARS is not set. Set it to the full path of vcvars64.bat from Visual Studio,
echo for example:  set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
exit /b 2

rem ---------------------------------------------------------------------------
rem :run <tag> <affinity mask or ""> <thread counts>     one run of bench_accel
:run
set "RTAG=%~1"
set "RMASK=%~2"
set "RTHREADS=%~3"
set "LOG=%OUT%\%RTAG%.log"
call :load "before the run" wait > "%LOG%" 2>&1
>> "%LOG%" echo machine: %MACHINE%, %CORES% cores, %LOGICAL% logical processors
if "%RMASK%"=="" goto run_plain
>> "%LOG%" echo placement: affinity mask %RMASK% - %MASK_FROM%; threads %RTHREADS%
start "" /b /wait /affinity %RMASK% "%EXE%" --threads "%RTHREADS%" --machine "%MACHINE%" %ARGS% > "%OUT%\%RTAG%.json" 2>> "%LOG%"
set "RC=%errorlevel%"
goto run_done
:run_plain
>> "%LOG%" echo placement: default - Windows places the threads; threads %RTHREADS%
"%EXE%" --threads "%RTHREADS%" --machine "%MACHINE%" %ARGS% > "%OUT%\%RTAG%.json" 2>> "%LOG%"
set "RC=%errorlevel%"
:run_done
call :load "after the run" >> "%LOG%" 2>&1
>> "%LOG%" echo exit code %RC%
echo wrote %OUT%\%RTAG%.json and %LOG% - exit code %RC%
if not "%RC%"=="0" set "FINAL_RC=%RC%"
exit /b 0

rem Print the time, the total CPU load (a few samples) and the busiest processes.
rem With "wait" as second argument: keep sampling every 15 s until the load is
rem below IDLE_BELOW percent, at most 12 times.
:load
echo ==== load %~1
if "%~2"=="wait" goto load_wait
powershell -NoProfile -Command "Get-Date -Format s; 1..3 | ForEach-Object { 'CPU load percent: ' + (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average; Start-Sleep -Milliseconds 700 }"
goto load_top
:load_wait
powershell -NoProfile -Command "Get-Date -Format s; for ($i = 0; $i -lt 12; $i++) { $l = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average; 'CPU load percent: ' + $l; if ($l -lt %IDLE_BELOW%) { break }; Start-Sleep -Seconds 15 }"
:load_top
powershell -NoProfile -Command "'busiest processes (percent of one logical core):'; Get-CimInstance Win32_PerfFormattedData_PerfProc_Process | Where-Object { $_.Name -ne '_Total' -and $_.Name -ne 'Idle' } | Sort-Object PercentProcessorTime -Descending | Select-Object -First 5 | ForEach-Object { '  ' + $_.PercentProcessorTime + '  ' + $_.Name }"
exit /b 0
