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
echo   8. Node0 Performance Snapshot (Analyze)
echo   9. Node0 Performance Recovery (Dry Run)
echo  10. Node0 Performance Recovery (Execute)
echo  11. Docker Volume Governance (Inventory)
echo  12. Docker Volume Governance (Dry Run Reclaim)
echo  13. Docker Volume Governance (Execute k3d Reclaim)
echo  14. VHDX Compaction Snapshot (Analyze)
echo  15. VHDX Compaction (Dry Run)
echo  16. VHDX Compaction (Execute)
echo  17. Pagefile Governance Snapshot (Analyze)
echo  18. Pagefile Governance (Dry Run Apply)
echo  19. Pagefile Governance (Execute Apply)
echo  20. Schedule Post-Reboot VHDX Compact (One-Time)
echo  21. Exit
echo.
echo  ======================================================================
echo.

set /p choice="Select option (1-21): "

if "%choice%"=="1" goto PROCESS_ONCE
if "%choice%"=="2" goto WATCH_MODE
if "%choice%"=="3" goto INGEST_CLOUD
if "%choice%"=="4" goto VIEW_LOG
if "%choice%"=="5" goto STATS
if "%choice%"=="6" goto OPEN_FOLDER
if "%choice%"=="7" goto GUIDE
if "%choice%"=="8" goto PERF_ANALYZE
if "%choice%"=="9" goto PERF_REMEDIATE_DRYRUN
if "%choice%"=="10" goto PERF_REMEDIATE_EXEC
if "%choice%"=="11" goto VOL_GOV_INVENTORY
if "%choice%"=="12" goto VOL_GOV_DRYRUN
if "%choice%"=="13" goto VOL_GOV_EXEC
if "%choice%"=="14" goto VHDX_ANALYZE
if "%choice%"=="15" goto VHDX_DRYRUN
if "%choice%"=="16" goto VHDX_EXEC
if "%choice%"=="17" goto PAGEFILE_ANALYZE
if "%choice%"=="18" goto PAGEFILE_DRYRUN
if "%choice%"=="19" goto PAGEFILE_EXEC
if "%choice%"=="20" goto SCHEDULE_REBOOT_COMPACT
if "%choice%"=="21" goto EXIT
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
    powershell.exe -ExecutionPolicy Bypass -File "C:\BIZRA-DATA-LAKE\scripts\ops\CloudIngestion.ps1" -DryRun
)
if "%cloud_choice%"=="2" (
    powershell.exe -ExecutionPolicy Bypass -File "C:\BIZRA-DATA-LAKE\scripts\ops\CloudIngestion.ps1" -Source OneDrive
)
if "%cloud_choice%"=="3" (
    powershell.exe -ExecutionPolicy Bypass -File "C:\BIZRA-DATA-LAKE\scripts\ops\CloudIngestion.ps1" -Source GoogleDrive
)
if "%cloud_choice%"=="4" (
    powershell.exe -ExecutionPolicy Bypass -File "C:\BIZRA-DATA-LAKE\scripts\ops\CloudIngestion.ps1" -Source Both
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
start notepad "C:\BIZRA-DATA-LAKE\docs\QUICK-START.md"
goto MENU

:PERF_ANALYZE
cls
echo Running Node0 performance snapshot (Analyze mode)...
echo.
powershell.exe -ExecutionPolicy Bypass -File "C:\BIZRA-DATA-LAKE\scripts\ops\node0_performance_recovery.ps1" -Mode Analyze
echo.
pause
goto MENU

:PERF_REMEDIATE_DRYRUN
cls
echo Running Node0 recovery plan in Dry-Run mode...
echo.
powershell.exe -ExecutionPolicy Bypass -File "C:\BIZRA-DATA-LAKE\scripts\ops\node0_performance_recovery.ps1" -Mode Remediate -DryRun:$true
echo.
pause
goto MENU

:PERF_REMEDIATE_EXEC
cls
echo Running Node0 recovery plan in Execute mode...
echo This will apply cleanup operations and asks for explicit confirmation.
echo.
powershell.exe -ExecutionPolicy Bypass -File "C:\BIZRA-DATA-LAKE\scripts\ops\node0_performance_recovery.ps1" -Mode Remediate -DryRun:$false
echo.
pause
goto MENU

:VOL_GOV_INVENTORY
cls
echo Running Docker volume governance inventory...
echo.
python "C:\BIZRA-DATA-LAKE\scripts\ops\docker_volume_governance.py" inventory
echo.
pause
goto MENU

:VOL_GOV_DRYRUN
cls
echo Running Docker volume governance reclaim in Dry-Run mode...
echo.
python "C:\BIZRA-DATA-LAKE\scripts\ops\docker_volume_governance.py" --dry-run reclaim-k3d
echo.
pause
goto MENU

:VOL_GOV_EXEC
cls
echo Running Docker volume governance reclaim (k3d cache prune)...
echo This applies cache cleanup and includes operator confirmation.
echo.
python "C:\BIZRA-DATA-LAKE\scripts\ops\docker_volume_governance.py" reclaim-k3d --restart-cluster
echo.
pause
goto MENU

:VHDX_ANALYZE
cls
echo Running VHDX compaction governance snapshot (Analyze mode)...
echo.
call "C:\BIZRA-DATA-LAKE\scripts\ops\VHDX-COMPACTION-LAUNCHER.bat" -Mode Analyze
echo.
pause
goto MENU

:VHDX_DRYRUN
cls
echo Running VHDX compaction governance in Dry-Run mode...
echo.
call "C:\BIZRA-DATA-LAKE\scripts\ops\VHDX-COMPACTION-LAUNCHER.bat" -Mode Compact -DryRun:$true -Target docker
echo.
pause
goto MENU

:VHDX_EXEC
cls
echo Running VHDX compaction governance in Execute mode...
echo This performs Windows-side offline VHDX compaction and asks for explicit confirmation.
echo.
call "C:\BIZRA-DATA-LAKE\scripts\ops\VHDX-COMPACTION-LAUNCHER.bat" -Mode Compact -DryRun:$false -Target docker
echo.
pause
goto MENU

:PAGEFILE_ANALYZE
cls
echo Running Pagefile governance snapshot (Analyze mode)...
echo.
call "C:\BIZRA-DATA-LAKE\scripts\ops\PAGEFILE-GOVERNANCE-LAUNCHER.bat" -Mode Analyze
echo.
pause
goto MENU

:PAGEFILE_DRYRUN
cls
echo Running Pagefile governance in Dry-Run apply mode...
echo.
call "C:\BIZRA-DATA-LAKE\scripts\ops\PAGEFILE-GOVERNANCE-LAUNCHER.bat" -Mode Apply -DryRun:$true -NoPrompt
echo.
pause
goto MENU

:PAGEFILE_EXEC
cls
echo Running Pagefile governance in Execute apply mode...
echo This updates pagefile sizing and may require reboot for full effect.
echo.
call "C:\BIZRA-DATA-LAKE\scripts\ops\PAGEFILE-GOVERNANCE-LAUNCHER.bat" -Mode Apply -DryRun:$false
echo.
pause
goto MENU

:SCHEDULE_REBOOT_COMPACT
cls
echo Registering one-time post-reboot VHDX compaction task...
echo This task runs once at next user logon (interactive elevated), executes compaction, then removes itself.
echo.
call "C:\BIZRA-DATA-LAKE\scripts\ops\SCHEDULE-POST-REBOOT-COMPACTION.bat" -Target docker
echo.
pause
goto MENU

:EXIT
cls
echo.
echo Goodbye! Data Lake operations complete.
echo.
timeout /t 2 >nul
exit
