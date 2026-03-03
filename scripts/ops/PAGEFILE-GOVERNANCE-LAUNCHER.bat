@echo off
setlocal

set "SCRIPT=C:\BIZRA-DATA-LAKE\scripts\ops\pagefile_governance.ps1"

if not exist "%SCRIPT%" (
    echo ERROR: Missing script: %SCRIPT%
    exit /b 2
)

if "%~1"=="" (
    echo Usage:
    echo   PAGEFILE-GOVERNANCE-LAUNCHER.bat -Mode Analyze
    echo   PAGEFILE-GOVERNANCE-LAUNCHER.bat -Mode Apply -DryRun:$true
    echo   PAGEFILE-GOVERNANCE-LAUNCHER.bat -Mode Apply -DryRun:$false
    exit /b 2
)

set "NEED_ELEVATION=0"
set "DRYRUN_HINT=0"
set "SKIP_ELEVATION=0"
echo %* | findstr /I /C:"-SkipElevation" >nul && set "SKIP_ELEVATION=1"
if /I "%~1"=="-Mode" (
    if /I "%~2"=="Apply" (
        echo %* | findstr /I /C:"-DryRun:$true" /C:"-DryRun:true" /C:"-DryRun true" /C:"-DryRun 1" >nul && set "DRYRUN_HINT=1"
        if "%DRYRUN_HINT%"=="0" set "NEED_ELEVATION=1"
    )
)
if "%SKIP_ELEVATION%"=="1" set "NEED_ELEVATION=0"

if "%NEED_ELEVATION%"=="1" (
    set "IS_ADMIN=False"
    for /f %%A in ('powershell.exe -NoProfile -Command "(New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)"') do set "IS_ADMIN=%%A"
    if /I not "%IS_ADMIN%"=="True" (
        echo Requesting Administrator elevation for apply mode...
        powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$arg='-NoProfile -ExecutionPolicy Bypass -File ""%SCRIPT%"" %*'; $p=Start-Process -FilePath 'powershell.exe' -Verb RunAs -PassThru -Wait -ArgumentList $arg; exit $p.ExitCode"
        exit /b %errorlevel%
    )
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%" %*
exit /b %errorlevel%
