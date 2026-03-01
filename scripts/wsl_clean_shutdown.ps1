# BIZRA DevOps: Clean WSL Shutdown Script
# Register as a Windows scheduled task to run on system shutdown/sleep
#
# Prevents EXT4 corruption by cleanly shutting down WSL before Windows sleep/shutdown
#
# To register as a scheduled task (run once as Admin):
#   $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -File C:\BIZRA-DATA-LAKE\scripts\wsl_clean_shutdown.ps1"
#   $trigger = New-ScheduledTaskTrigger -AtLogOff
#   Register-ScheduledTask -TaskName "BIZRA-WSL-CleanShutdown" -Action $action -Trigger $trigger -User "SYSTEM" -RunLevel Highest

$logFile = "C:\BIZRA-DATA-LAKE\logs\wsl_shutdown.log"
$logDir = Split-Path $logFile
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content $logFile "$timestamp - WSL clean shutdown initiated"

try {
    # Graceful shutdown
    wsl --shutdown 2>&1 | Out-Null
    Start-Sleep -Seconds 3
    Add-Content $logFile "$timestamp - WSL shutdown completed"
} catch {
    Add-Content $logFile "$timestamp - WSL shutdown failed: $_"
}
