@echo off
REM ============================================================================
REM BIZRA Unified Node Installer - Windows One-Click Entry
REM ============================================================================
REM Just double-click this file. That's it.
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo  ============================================================
echo                    BIZRA NODE INSTALLER
echo          Your Gateway to the Decentralized Future
echo  ============================================================
echo.

REM Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Python not found. Installing Python...
    echo.
    echo Downloading Python installer...

    REM Download Python installer
    powershell -Command "& {Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe' -OutFile '%TEMP%\python_installer.exe'}"

    REM Install Python silently
    %TEMP%\python_installer.exe /quiet InstallAllUsers=0 PrependPath=1 Include_test=0

    echo [OK] Python installed. Please restart this installer.
    pause
    exit /b 0
)

echo [OK] Python found
python --version

REM Check for pip
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] pip not found. Installing pip...
    python -m ensurepip --upgrade
)

echo [OK] pip available

REM Set BIZRA home
set BIZRA_HOME=%USERPROFILE%\.bizra
if not exist "%BIZRA_HOME%" mkdir "%BIZRA_HOME%"

echo [OK] BIZRA home: %BIZRA_HOME%

REM Download and run the installer
echo.
echo [*] Starting BIZRA installer...
echo.

REM If install.py exists locally, use it
set INSTALLER_SCRIPT=%~dp0..\..\bootstrap\install.py
if exist "%INSTALLER_SCRIPT%" (
    python "%INSTALLER_SCRIPT%"
) else (
    REM Download from network (placeholder URL)
    echo [*] Downloading installer script...
    powershell -Command "& {Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/bizra-ai/unified-node/main/bootstrap/install.py' -OutFile '%BIZRA_HOME%\install.py'}"
    python "%BIZRA_HOME%\install.py"
)

echo.
echo  ============================================================
echo                  Installation Complete!
echo  ============================================================
echo.
echo  Your BIZRA node is ready.
echo.
echo  To start: Run start-bizra.bat or open http://localhost:8888
echo.
echo  Welcome to the decentralized AI civilization.
echo.

pause
