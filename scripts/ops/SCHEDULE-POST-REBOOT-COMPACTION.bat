@echo off
setlocal

set "SCRIPT=C:\BIZRA-DATA-LAKE\scripts\ops\schedule_post_reboot_vhdx_compaction.ps1"

if not exist "%SCRIPT%" (
    echo ERROR: Missing script: %SCRIPT%
    exit /b 2
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%" %*
exit /b %errorlevel%
