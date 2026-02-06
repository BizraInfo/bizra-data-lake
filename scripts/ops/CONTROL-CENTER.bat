@echo off
color 0A
title BIZRA DATA LAKE CONTROL CENTER

:MENU
cls
echo.
echo  ======================================================================
echo                  BIZRA DATA LAKE CONTROL CENTER
echo  ======================================================================
echo.
echo   1. Process Files Once (Batch Mode)
echo   2. Start Continuous Monitoring (Watch Mode)
echo   3. Ingest from Cloud Drives (OneDrive + Google Drive)
echo   4. View Processing Log (Last 50 entries)
echo   5. View Statistics
echo   6. Open Data Lake Folder
echo   7. Quick Start Guide
echo   8. Exit
echo.
echo  ======================================================================
echo.

set /p choice="Select option (1-8): "

if "%choice%"=="1" goto PROCESS_ONCE
if "%choice%"=="2" goto WATCH_MODE
if "%choice%"=="3" goto INGEST_CLOUD
if "%choice%"=="4" goto VIEW_LOG
if "%choice%"=="5" goto STATS
if "%choice%"=="6" goto OPEN_FOLDER
if "%choice%"=="7" goto GUIDE
if "%choice%"=="8" goto EXIT
goto MENU

:PROCESS_ONCE
cls
echo Processing files in INTAKE folder...
echo.
powershell.exe -ExecutionPolicy Bypass -File "C:\BIZRA-DATA-LAKE\DataLakeProcessor.ps1" -ProcessOnce
echo.
pause
goto MENU

:WATCH_MODE
cls
echo Starting continuous monitoring...
echo Press Ctrl+C to stop.
echo.
powershell.exe -ExecutionPolicy Bypass -File "C:\BIZRA-DATA-LAKE\DataLakeProcessor.ps1" -Watch
pause
goto MENU

:INGEST_CLOUD
cls
echo.
echo Cloud Drive Ingestion Options:
echo  1. Dry Run (Test without copying)
echo  2. Ingest OneDrive
echo  3. Ingest Google Drive
echo  4. Ingest Both
echo  5. Back to Main Menu
echo.
set /p cloud_choice="Select option (1-5): "

if "%cloud_choice%"=="1" (
    powershell.exe -ExecutionPolicy Bypass -File "C:\BIZRA-DATA-LAKE\CloudIngestion.ps1" -DryRun
)
if "%cloud_choice%"=="2" (
    powershell.exe -ExecutionPolicy Bypass -File "C:\BIZRA-DATA-LAKE\CloudIngestion.ps1" -Source OneDrive
)
if "%cloud_choice%"=="3" (
    powershell.exe -ExecutionPolicy Bypass -File "C:\BIZRA-DATA-LAKE\CloudIngestion.ps1" -Source GoogleDrive
)
if "%cloud_choice%"=="4" (
    powershell.exe -ExecutionPolicy Bypass -File "C:\BIZRA-DATA-LAKE\CloudIngestion.ps1" -Source Both
)
if "%cloud_choice%"=="5" goto MENU

echo.
pause
goto MENU

:VIEW_LOG
cls
echo ====== PROCESSING LOG (Last 50 Entries) ======
echo.
powershell.exe -Command "Get-Content 'C:\BIZRA-DATA-LAKE\processing.log' -Tail 50"
echo.
pause
goto MENU

:STATS
cls
echo ====== DATA LAKE STATISTICS ======
echo.
powershell.exe -Command "$intake = (Get-ChildItem 'C:\BIZRA-DATA-LAKE\00_INTAKE' -File -Recurse -ErrorAction SilentlyContinue).Count; $processed = (Get-ChildItem 'C:\BIZRA-DATA-LAKE\02_PROCESSED' -File -Recurse -ErrorAction SilentlyContinue).Count; $quarantine = (Get-ChildItem 'C:\BIZRA-DATA-LAKE\99_QUARANTINE' -File -Recurse -ErrorAction SilentlyContinue).Count; Write-Host 'Intake Queue: '$intake' files'; Write-Host 'Processed: '$processed' files'; Write-Host 'Duplicates Quarantined: '$quarantine' files'; Write-Host ''; Write-Host 'Type Distribution:'; Get-ChildItem 'C:\BIZRA-DATA-LAKE\02_PROCESSED\*' -Directory | ForEach-Object { $count = (Get-ChildItem $_.FullName -File -Recurse -ErrorAction SilentlyContinue).Count; Write-Host '  '$_.Name': '$count' files' }"
echo.
pause
goto MENU

:OPEN_FOLDER
explorer "C:\BIZRA-DATA-LAKE"
goto MENU

:GUIDE
start notepad "C:\BIZRA-DATA-LAKE\QUICK-START.md"
goto MENU

:EXIT
cls
echo.
echo Goodbye! Data Lake operations complete.
echo.
timeout /t 2 >nul
exit

