@echo off
setlocal
rem ---------------------------------------------------------------------------
rem Build and run examples\bench_gpu.cpp on Windows (CPU baseline + CUDA backend).
rem This file is plain ASCII on purpose: cmd.exe reads .bat files in the console
rem code page, so non-ASCII comments would break it.
rem
rem   examples\build_gpu_win.bat            MinGW g++ first; MSVC only if g++ fails
rem   examples\build_gpu_win.bat mingw      MinGW g++ only   -> build\bench_gpu.exe
rem   examples\build_gpu_win.bat msvc       MSVC only        -> build\bench_gpu_msvc.exe
rem   examples\build_gpu_win.bat all        both
rem   examples\build_gpu_win.bat run  [bench_gpu options]    run build\bench_gpu.exe
rem   examples\build_gpu_win.bat runmsvc [bench_gpu options] run build\bench_gpu_msvc.exe
rem
rem Needs CUDA Toolkit 12.6 or newer (its installer sets CUDA_PATH). The backend
rem only calls cuSOLVER / cuBLAS, so an ordinary C++ compiler is enough: no nvcc.
rem There are no default compiler paths: set MINGW_BIN (the MinGW-w64 bin
rem directory holding g++.exe) for mingw and all, and VCVARS (the full path of
rem vcvars64.bat) for msvc and all. auto uses g++ when MINGW_BIN is set (then
rem MSVC only if g++ fails, which needs VCVARS too), MSVC when only VCVARS is set.
rem ---------------------------------------------------------------------------

if "%CUDA_PATH%"=="" (
    echo CUDA_PATH is not set: install the CUDA Toolkit or set CUDA_PATH by hand.
    exit /b 1
)
set "ROOT=%~dp0.."
set "SRC=%ROOT%\examples\bench_gpu.cpp"
set "OUT=%ROOT%\build"
set "MODE=%~1"
if "%MODE%"=="" set "MODE=auto"
if /i "%MODE%"=="auto" if not defined MINGW_BIN if defined VCVARS set "MODE=msvc"
if /i "%MODE%"=="mingw" if not defined MINGW_BIN goto no_mingw_bin
if /i "%MODE%"=="auto"  if not defined MINGW_BIN goto no_mingw_bin
if /i "%MODE%"=="all"   if not defined MINGW_BIN goto no_mingw_bin
if /i "%MODE%"=="all"   if not defined VCVARS goto no_vcvars
if /i "%MODE%"=="msvc"  if not defined VCVARS goto no_vcvars
if not exist "%OUT%" mkdir "%OUT%"

rem The CUDA runtime libraries sit in bin\x64 (toolkit 13) or bin (toolkit 12).
set "PATH=%CUDA_PATH%\bin\x64;%CUDA_PATH%\bin;%PATH%"

if /i "%MODE%"=="run"     goto run_mingw
if /i "%MODE%"=="runmsvc" goto run_msvc
if /i "%MODE%"=="mingw"   goto build_mingw
if /i "%MODE%"=="msvc"    goto build_msvc
if /i "%MODE%"=="all"     goto build_mingw
if /i "%MODE%"=="auto"    goto build_mingw
echo unknown mode "%MODE%" (use mingw, msvc, all, run or runmsvc)
exit /b 1

:build_mingw
echo [mingw] g++ -std=c++17 -O2, static runtime
rem cusolver.lib and cublas.lib are plain import libraries and link as they are.
rem cudart.lib also carries an MSVC-compiled loader object that g++ cannot link,
rem so the CUDA runtime is linked straight from its DLL instead.
set "CUDART_DLL="
for %%F in ("%CUDA_PATH%\bin\cudart64_*.dll" "%CUDA_PATH%\bin\x64\cudart64_*.dll") do set "CUDART_DLL=%%~fF"
if "%CUDART_DLL%"=="" (
    echo [mingw] cannot find the CUDA runtime DLL under "%CUDA_PATH%\bin"
    if /i "%MODE%"=="mingw" exit /b 1
    goto build_msvc
)
"%MINGW_BIN%\g++.exe" -std=c++17 -O2 -Wall -Wextra -Wpedantic -DBENCH_OPT=-O2 ^
    -I "%ROOT%\include" -isystem "%CUDA_PATH%\include" "%SRC%" -o "%OUT%\bench_gpu.exe" ^
    -L "%CUDA_PATH%\lib\x64" -lcusolver -lcublas "%CUDART_DLL%" -static
if errorlevel 1 (
    echo [mingw] build FAILED
    if /i "%MODE%"=="mingw" exit /b 1
    goto build_msvc
)
echo [mingw] built %OUT%\bench_gpu.exe
if /i "%MODE%"=="all" goto build_msvc
exit /b 0

:build_msvc
if not defined VCVARS goto no_vcvars
echo [msvc] cl /std:c++17 /O2
call "%VCVARS%" >nul
if errorlevel 1 (
    echo [msvc] could not run "%VCVARS%"
    exit /b 1
)
cl /nologo /std:c++17 /O2 /EHsc /utf-8 /W4 /DBENCH_OPT=/O2 ^
    /I "%ROOT%\include" /external:I "%CUDA_PATH%\include" /external:W0 "%SRC%" ^
    /Fe:"%OUT%\bench_gpu_msvc.exe" /Fo:"%OUT%\bench_gpu_msvc.obj" ^
    /link /LIBPATH:"%CUDA_PATH%\lib\x64" cusolver.lib cublas.lib cudart.lib
if errorlevel 1 (
    echo [msvc] build FAILED
    exit /b 1
)
echo [msvc] built %OUT%\bench_gpu_msvc.exe
exit /b 0

:run_mingw
set "EXE=%OUT%\bench_gpu.exe"
set "ARGS=%*"
set "ARGS=%ARGS:*run=%"
goto launch
:run_msvc
set "EXE=%OUT%\bench_gpu_msvc.exe"
set "ARGS=%*"
set "ARGS=%ARGS:*runmsvc=%"
:launch
rem ARGS is the command line with the mode word cut off (commas in --sizes survive).
"%EXE%"%ARGS%
exit /b %errorlevel%

:no_mingw_bin
echo MINGW_BIN is not set. Set it to the MinGW-w64 bin directory that holds g++.exe,
echo for example:  set "MINGW_BIN=C:\mingw64\bin"
exit /b 1
:no_vcvars
echo VCVARS is not set. Set it to the full path of vcvars64.bat from Visual Studio,
echo for example:  set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
exit /b 1
