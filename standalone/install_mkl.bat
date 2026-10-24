@echo off
rem Downloads the oneMKL runtime for MatrixEigenValueAndEigenVector_MKL.cpp into mkl\bin (see install_mkl.ps1).
rem Double-click it, or run it from cmd; arguments are passed on, for example: install_mkl.bat -Version 2025.3.1
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_mkl.ps1" %*
set status=%errorlevel%
if not "%status%"=="0" echo Installation failed.
rem Keep the window open only when the file was double-clicked (Explorer starts it with "cmd /c").
if not "%~1"=="" goto done
echo %cmdcmdline% | find /i "/c" >nul
if not errorlevel 1 pause
:done
exit /b %status%
