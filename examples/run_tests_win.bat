@echo off
setlocal
rem ---------------------------------------------------------------------------
rem The whole Windows check in one command: build every tests\test_*.cpp and the
rem five CPU examples with MinGW g++ and with MSVC, run the tests, and write a
rem summary with the warning count of every build and the pass count of every
rem test.
rem This file is plain ASCII on purpose: cmd.exe reads .bat files in the console
rem code page, so non-ASCII comments would break it.
rem
rem   examples\run_tests_win.bat            both compilers
rem   examples\run_tests_win.bat mingw      MinGW g++ only
rem   examples\run_tests_win.bat msvc       MSVC only
rem
rem Flags (the warning level the project is developed with):
rem   g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -I include [-I examples] -static
rem   cl  /std:c++17 /O2 /EHsc /W4 /utf-8 /I include [/I examples]
rem
rem What is built and run:
rem   tests     every tests\test_*.cpp; each one is run, and its pass count is
rem             read from the last line it prints ("passed x / total y", in
rem             Chinese). A test counts as passed when its exit code is 0 and
rem             x equals y.
rem   examples  eig_minimal, rcwa_demo, conical_demo, rcwa_json, bench_accel
rem             are built;
rem             eig_minimal and "rcwa_demo --M 60" are also run (exit code only).
rem             bench_gpu*.cpp need the CUDA Toolkit: see build_gpu_win.bat.
rem
rem Output, all under build\tests_win\ :
rem   summary.txt                   the table that is also printed at the end:
rem                                 compiler, target, build result, number of
rem                                 warning lines, test result; then the totals
rem   <compiler>\<name>.build.txt   everything the compiler printed (console code
rem                                 page, e.g. GBK from a Chinese-language MSVC)
rem   <compiler>\<name>.run.txt     everything the program printed (UTF-8)
rem "Warning lines" = lines of the build log that contain the word "warning".
rem
rem Exit code: 0 only if every build succeeded, every test passed and no build
rem printed a warning; otherwise the number of failed builds and tests (plus one
rem if there were warnings). 2 for a usage error, or when MINGW_BIN or VCVARS is
rem needed but not set (nothing is built then).
rem
rem Required environment (no defaults; mingw needs MINGW_BIN, msvc needs VCVARS,
rem both compilers need both):
rem   MINGW_BIN           the MinGW-w64 bin directory that holds g++.exe
rem   VCVARS              the full path of vcvars64.bat from Visual Studio
rem
rem Environment overrides:
rem   GXX_EXTRA, CL_EXTRA extra compiler options, appended to the lines above
rem   TESTS_OUT           output directory (default build\tests_win)
rem
rem Limitations: tried from a short ASCII path and from a directory whose name
rem contains a space. Paths with non-ASCII characters, or with percent signs,
rem carets, ampersands, quotes or commas, are not supported. Needs
rem powershell.exe (part of Windows) to read the UTF-8 last line of each test.
rem ---------------------------------------------------------------------------

set "EXAMPLES=eig_minimal rcwa_demo conical_demo rcwa_json bench_accel"
set "FIND=%SystemRoot%\System32\find.exe"

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "OUT=%ROOT%\build\tests_win"
if not "%TESTS_OUT%"=="" set "OUT=%TESTS_OUT%"
set "MODE=%~1"
if "%MODE%"=="" set "MODE=all"
set "MODE_OK="
if /i "%MODE%"=="all" set "MODE_OK=1"
if /i "%MODE%"=="mingw" set "MODE_OK=1"
if /i "%MODE%"=="msvc" set "MODE_OK=1"
if not defined MODE_OK (
    echo usage: run_tests_win.bat [all^|mingw^|msvc]
    exit /b 2
)
if /i not "%MODE%"=="msvc" if not defined MINGW_BIN goto no_mingw_bin
if /i not "%MODE%"=="mingw" if not defined VCVARS goto no_vcvars
if not exist "%OUT%\mingw" mkdir "%OUT%\mingw"
if not exist "%OUT%\msvc" mkdir "%OUT%\msvc"
set "SUMMARY=%OUT%\summary.txt"
set /a BUILD_FAILED=0
set /a TEST_FAILED=0
set /a WARN_TOTAL=0
set "NOW="
for /f "usebackq delims=" %%T in (`powershell -NoProfile -Command "Get-Date -Format s"`) do set "NOW=%%T"
> "%SUMMARY%" echo run_tests_win summary: %MODE%, %NOW%
>> "%SUMMARY%" echo.
>> "%SUMMARY%" echo compiler target                           build   warnings  result

if /i "%MODE%"=="msvc" goto msvc

rem ------------------------------------------------------------------ MinGW
set "COMPILER=mingw"
if not exist "%MINGW_BIN%\g++.exe" (
    >> "%SUMMARY%" echo mingw    g++.exe not found in "%MINGW_BIN%"
    set /a BUILD_FAILED+=1
    goto after_mingw
)
"%MINGW_BIN%\g++.exe" --version > "%OUT%\mingw\version.txt" 2>&1
rem g++ needs its own bin directory on PATH (the compiler proper loads DLLs from
rem there). The programs themselves are linked with -static and need nothing.
set "PATH=%MINGW_BIN%;%PATH%"
rem Keep the compiler messages in English so that "warning" can be counted.
set "LC_ALL=C"
for %%F in ("%ROOT%\tests\test_*.cpp") do call :one test %%~nF
for %%N in (%EXAMPLES%) do call :one example %%N
:after_mingw
if /i "%MODE%"=="mingw" goto finish

rem ------------------------------------------------------------------ MSVC
:msvc
set "COMPILER=msvc"
if not exist "%VCVARS%" (
    >> "%SUMMARY%" echo msvc     "%VCVARS%" not found
    set /a BUILD_FAILED+=1
    goto finish
)
call "%VCVARS%" > "%OUT%\msvc\vcvars.txt" 2>&1
cl > "%OUT%\msvc\version.txt" 2>&1
for %%F in ("%ROOT%\tests\test_*.cpp") do call :one test %%~nF
for %%N in (%EXAMPLES%) do call :one example %%N

:finish
set /a FAILED=BUILD_FAILED+TEST_FAILED
if not "%WARN_TOTAL%"=="0" set /a FAILED+=1
>> "%SUMMARY%" echo.
>> "%SUMMARY%" echo totals: builds failed %BUILD_FAILED%, tests failed %TEST_FAILED%, warning lines %WARN_TOTAL%
type "%SUMMARY%"
exit /b %FAILED%

:no_mingw_bin
echo MINGW_BIN is not set. Set it to the MinGW-w64 bin directory that holds g++.exe,
echo for example:  set "MINGW_BIN=C:\mingw64\bin"
exit /b 2
:no_vcvars
echo VCVARS is not set. Set it to the full path of vcvars64.bat from Visual Studio,
echo for example:  set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
exit /b 2

rem ---------------------------------------------------------------------------
rem :one KIND NAME   (KIND is test or example) build one program with the
rem                 current compiler, run it, append one line to the summary
:one
set "KIND=%~1"
set "NAME=%~2"
set "DIR=%OUT%\%COMPILER%"
set "EXE=%DIR%\%NAME%.exe"
set "BLOG=%DIR%\%NAME%.build.txt"
set "RLOG=%DIR%\%NAME%.run.txt"
if exist "%EXE%" del "%EXE%"
if exist "%RLOG%" del "%RLOG%"
if "%KIND%"=="example" goto one_example
set "SRC=%ROOT%\tests\%NAME%.cpp"
set GXX_INC=-I "%ROOT%\include"
set CL_INC=/I "%ROOT%\include"
goto one_build
:one_example
set "SRC=%ROOT%\examples\%NAME%.cpp"
set GXX_INC=-I "%ROOT%\include" -I "%ROOT%\examples"
set CL_INC=/I "%ROOT%\include" /I "%ROOT%\examples"

:one_build
if "%COMPILER%"=="msvc" goto one_build_msvc
"%MINGW_BIN%\g++.exe" -std=c++17 -O2 -Wall -Wextra -Wpedantic %GXX_EXTRA% %GXX_INC% "%SRC%" -o "%EXE%" -static > "%BLOG%" 2>&1
goto one_built
:one_build_msvc
cl /nologo /std:c++17 /O2 /EHsc /W4 /utf-8 %CL_EXTRA% %CL_INC% "%SRC%" /Fe:"%EXE%" /Fo:"%DIR%\%NAME%.obj" > "%BLOG%" 2>&1
:one_built
if errorlevel 1 (set "BUILD=FAILED") else (set "BUILD=ok")
set "WARN=0"
for /f %%C in ('%FIND% /c /i "warning" ^< "%BLOG%"') do set "WARN=%%C"
set /a WARN_TOTAL+=WARN
set "RESULT=not run"
if "%BUILD%"=="ok" goto one_run
set /a BUILD_FAILED+=1
goto one_report

:one_run
if "%KIND%"=="test" goto one_run_test
set "RUNARGS="
if "%NAME%"=="eig_minimal" goto one_run_example
if "%NAME%"=="rcwa_demo" set "RUNARGS=--M 60"
if "%NAME%"=="rcwa_demo" goto one_run_example
set "RESULT=built only"
goto one_report
:one_run_example
"%EXE%" %RUNARGS% > "%RLOG%" 2>&1
set "RC=%errorlevel%"
if not "%RC%"=="0" set /a TEST_FAILED+=1
set "RESULT=exit code %RC% (run %NAME% %RUNARGS%)"
goto one_report

:one_run_test
"%EXE%" > "%RLOG%" 2>&1
set "RC=%errorlevel%"
rem The last line of a test is UTF-8 text with two numbers: passed and total.
set "PASSED=?"
set "TOTAL=?"
for /f "usebackq tokens=1,2" %%A in (`powershell -NoProfile -Command "$t = Get-Content -LiteralPath '%RLOG%' -Encoding UTF8 | Where-Object { $_.Trim() } | Select-Object -Last 1; if ($t -match '(\d+)\D+(\d+)\s*$') { $Matches[1] + ' ' + $Matches[2] }"`) do set "PASSED=%%A"& set "TOTAL=%%B"
set "GOOD=1"
if not "%RC%"=="0" set "GOOD="
if not "%PASSED%"=="%TOTAL%" set "GOOD="
if "%PASSED%"=="?" set "GOOD="
if not defined GOOD set /a TEST_FAILED+=1
set "RESULT=passed %PASSED% / %TOTAL%, exit code %RC%"
if not defined GOOD set "RESULT=%RESULT%  FAILED"

:one_report
set "C1=%COMPILER%            "
set "C2=%NAME%                                 "
set "C3=%BUILD%            "
set "C4=%WARN%            "
>> "%SUMMARY%" echo %C1:~0,9%%C2:~0,33%%C3:~0,8%%C4:~0,10%%RESULT%
exit /b 0
