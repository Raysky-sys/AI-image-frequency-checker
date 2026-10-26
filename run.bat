@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "EXE=%CD%\dist\AIImageFrequencyChecker.exe"
set "SRC=%CD%\AI_image_frequency_checker.py"

echo.
echo ==================================================
echo   AI Image Frequency Checker  v1.1.0
echo ==================================================
echo.

if exist "%EXE%" (
    echo   Launching the compiled executable...
    echo.
    start "" "%EXE%" %*
    exit /b 0
)

echo   No compiled executable found in the "dist" folder.
echo   Falling back to running from source.
echo.

set "PYEXE="

where py >nul 2>&1
if not errorlevel 1 (
    py -3 -c "import sys" >nul 2>&1
    if not errorlevel 1 set "PYEXE=py -3"
)

if not defined PYEXE (
    where python >nul 2>&1
    if not errorlevel 1 (
        python -c "import sys" >nul 2>&1
        if not errorlevel 1 set "PYEXE=python"
    )
)

if not defined PYEXE (
    echo.
    echo   [ERROR] Python was not found on this machine.
    echo.
    echo   Either:
    echo     1. Install Python 3.10+ from https://www.python.org/downloads/
    echo        and tick "Add python.exe to PATH", or
    echo     2. Run build_exe.bat to create a standalone executable.
    echo.
    pause
    exit /b 1
)

echo   Checking dependencies...
%PYEXE% -c "import numpy, cv2, matplotlib, tkinter" >nul 2>&1
if errorlevel 1 (
    echo   Installing dependencies, please wait...
    %PYEXE% -m pip install --upgrade -r requirements.txt
    if errorlevel 1 (
        echo.
        echo   [ERROR] Dependency installation failed.
        pause
        exit /b 1
    )
)

echo.
echo   Launching...
%PYEXE% "%SRC%" %*

if errorlevel 1 (
    echo.
    echo   [ERROR] The application exited with an error.
    pause
    exit /b 1
)

endlocal
