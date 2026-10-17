@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "OPENBLAS_NUM_THREADS=1"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
cd /d "%~dp0"
if errorlevel 1 goto path_error

if not exist "requirements-release.txt" (
    echo ERROR: requirements-release.txt is missing from this folder.
    echo Download and extract the complete Windows release, then try again.
    goto failed
)

if exist ".venv\Scripts\python.exe" goto install_packages
if exist ".venv\" (
    echo ERROR: .venv exists but does not contain a Windows Python interpreter.
    echo Rename or remove .venv in this folder, then run this installer again.
    goto failed
)

if defined RCWA_PYTHON goto custom_python

py -3.12 -c "import sys,struct; assert sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8" >nul 2>&1
if not errorlevel 1 goto create_312
py -3.11 -c "import sys,struct; assert sys.version_info[:2] == (3,11) and struct.calcsize('P') == 8" >nul 2>&1
if not errorlevel 1 goto create_311
echo ERROR: Python 3.11 or 3.12 for Windows x64 was not found by the py launcher.
echo Install Python 3.11 or 3.12 x64 yourself, then run this installer again.
echo Or set RCWA_PYTHON to the full path of a compatible python.exe.
goto failed

:custom_python
set "RCWA_PYTHON_EXE=%RCWA_PYTHON:"=%"
"%RCWA_PYTHON_EXE%" -c "import sys,struct; assert sys.version_info[:2] in ((3,11),(3,12)) and struct.calcsize('P') == 8" >nul 2>&1
if errorlevel 1 (
    echo ERROR: RCWA_PYTHON must point to a working Python 3.11 or 3.12 x64 executable.
    goto failed
)
echo Creating local environment with RCWA_PYTHON...
"%RCWA_PYTHON_EXE%" -m venv ".venv"
if errorlevel 1 goto venv_error
goto install_packages

:create_312
echo Creating local environment with Python 3.12 x64...
py -3.12 -m venv ".venv"
if errorlevel 1 goto venv_error
goto install_packages

:create_311
echo Creating local environment with Python 3.11 x64...
py -3.11 -m venv ".venv"
if errorlevel 1 goto venv_error
goto install_packages

:venv_error
echo ERROR: Could not create .venv in this folder.
echo Check that this folder is writable and that Python includes venv.
goto failed

:install_packages
".venv\Scripts\python.exe" -c "import sys,struct; assert sys.version_info[:2] in ((3,11),(3,12)) and struct.calcsize('P') == 8" >nul 2>&1
if errorlevel 1 (
    echo ERROR: .venv contains an incompatible Python interpreter.
    echo Rename or remove .venv in this folder, then run this installer again.
    goto failed
)
echo Installing release dependencies into this folder's .venv...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --only-binary=:all: -r "requirements-release.txt"
if errorlevel 1 (
    echo ERROR: Dependency installation failed. Check the messages above.
    echo Confirm that your internet connection is available, then rerun this installer.
    goto failed
)
".venv\Scripts\python.exe" -c "import numpy, scipy, matplotlib, gui" >nul 2>&1
if errorlevel 1 (
    echo ERROR: The GUI dependency check failed after installation.
    echo Rerun this installer and check its output for errors.
    goto failed
)
echo.
echo Installation complete. Double-click start_gui.bat to open the GUI.
if not "%RCWA_NO_PAUSE%"=="1" pause
exit /b 0

:path_error
echo ERROR: Could not open the folder containing this installer.
:failed
echo.
echo Installation was not completed.
if not "%RCWA_NO_PAUSE%"=="1" pause
exit /b 1
