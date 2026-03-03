# BIZRA Schedule Post-Reboot VHDX Compaction
# ---------------------------------------------------------------------------
# Registers a one-time-at-startup scheduled task that runs VHDX compaction
# and then self-unregisters.

param(
    [string]$TaskName = "BIZRA-PostReboot-VHDX-Compact",
    [ValidateSet("docker", "ubuntu", "both")]
    [string]$Target = "docker",
    [string]$RunAsUser = $env:USERNAME,
    [switch]$NoPrompt
)

$ErrorActionPreference = "Stop"

function Test-IsAdministrator {
    try {
        $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
        $principal = New-Object Security.Principal.WindowsPrincipal($identity)
        return $principal.IsInRole(
            [Security.Principal.WindowsBuiltInRole]::Administrator
        )
    }
    catch {
        return $false
    }
}

function Confirm-OrDie {
    param([string]$PromptText)
    if ($NoPrompt) { return }
    $answer = Read-Host "$PromptText (type YES to continue)"
    if ($answer -ne "YES") {
        Write-Host "Aborted by operator." -ForegroundColor Red
        exit 1
    }
}

if (-not (Test-IsAdministrator)) {
    Write-Host "Administrator privileges are required to register startup tasks." -ForegroundColor Red
    exit 2
}

$runner = "C:\BIZRA-DATA-LAKE\scripts\ops\post_reboot_vhdx_compact_once.ps1"
if (-not (Test-Path $runner)) {
    Write-Host "Missing runner script: $runner" -ForegroundColor Red
    exit 3
}

Confirm-OrDie -PromptText "Register post-reboot compaction task '$TaskName' for target '$Target' as user '$RunAsUser'"

$arg = "-NoProfile -ExecutionPolicy Bypass -File `"$runner`" -TaskName `"$TaskName`" -Target `"$Target`""
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $RunAsUser
$principal = New-ScheduledTaskPrincipal -UserId $RunAsUser -RunLevel Highest -LogonType Interactive
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue | Out-Null
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null

$logDir = "C:\BIZRA-DATA-LAKE\logs"
if (-not (Test-Path $logDir)) {
    New-Item -Path $logDir -ItemType Directory -Force | Out-Null
}

$report = @{
    timestamp = (Get-Date).ToString("o")
    task_name = $TaskName
    target = $Target
    run_as_user = $RunAsUser
    action = "schedule_post_reboot_vhdx_compaction"
    runner_script = $runner
}

$path = Join-Path $logDir ("schedule_post_reboot_vhdx_compaction_{0}.json" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
$report | ConvertTo-Json -Depth 6 | Set-Content -Path $path -Encoding UTF8

Write-Host "Scheduled task registered: $TaskName" -ForegroundColor Green
Write-Host "Report written to: $path" -ForegroundColor Green
Write-Host "Reboot system to execute compaction automatically once." -ForegroundColor Yellow

exit 0
