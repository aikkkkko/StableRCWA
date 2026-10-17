@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "OPENBLAS_NUM_THREADS=1"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
cd /d "%~dp0"
if errorlevel 1 goto path_error

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: This release is not installed in this folder.
    echo Double-click install_windows.bat first, then try again.
    goto failed
)

".venv\Scripts\python.exe" -c "import numpy, scipy, matplotlib, gui" >nul 2>&1
if errorlevel 1 (
    echo ERROR: The local Python environment is incomplete or incompatible.
    echo Double-click install_windows.bat to repair it, then try again.
    goto failed
)

echo Starting the RCWA GUI.
echo Keep this window open while using the GUI. Press Ctrl+C here to stop it.
".venv\Scripts\python.exe" -u "gui.py" %*
if errorlevel 1 (
    echo ERROR: The GUI stopped with an error. Review the messages above.
    goto failed
)
exit /b 0

:path_error
echo ERROR: Could not open the folder containing this launcher.
:failed
echo.
if not "%RCWA_NO_PAUSE%"=="1" pause
exit /b 1
