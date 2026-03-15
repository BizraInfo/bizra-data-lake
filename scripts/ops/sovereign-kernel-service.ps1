# ═══════════════════════════════════════════════════════════════════════
#  BIZRA SOVEREIGN KERNEL — Windows Task Scheduler Auto-Start Service
#
#  Elite-grade: Task Scheduler (not Startup folder) — survives updates,
#  has retry logic, runs at logon + workstation unlock.
#
#  Usage:
#    .\sovereign-kernel-service.ps1 -Register     # Install auto-start
#    .\sovereign-kernel-service.ps1 -Unregister   # Remove auto-start
#    .\sovereign-kernel-service.ps1 -Status       # Check current state
#    .\sovereign-kernel-service.ps1 -Start        # Start daemon now
#    .\sovereign-kernel-service.ps1 -Stop         # Stop running daemon
# ═══════════════════════════════════════════════════════════════════════

param(
    [switch]$Register,
    [switch]$Unregister,
    [switch]$Status,
    [switch]$Start,
    [switch]$Stop
)

$ErrorActionPreference = "Stop"

# ── Paths ──
$scriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$bizraRoot  = (Get-Item "$scriptDir\..\.." ).FullName
$daemonPy   = Join-Path $bizraRoot "core\sovereign\kernel_daemon.py"
$stateDir   = Join-Path $bizraRoot "sovereign_state"
$pidFile    = Join-Path $stateDir "kernel.pid"
$taskName   = "BIZRA Sovereign Kernel"
$taskPath   = "\BIZRA\"

# ── Find Python (prefer pythonw.exe for no-console) ──
function Find-Python {
    # 1. Project venv
    $venvPythonW = Join-Path $bizraRoot ".venv\Scripts\pythonw.exe"
    $venvPython  = Join-Path $bizraRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPythonW) { return $venvPythonW }
    if (Test-Path $venvPython)  { return $venvPython }

    # 2. System pythonw
    $sysW = Get-Command pythonw.exe -ErrorAction SilentlyContinue
    if ($sysW) { return $sysW.Source }

    # 3. System python
    $sys = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($sys) { return $sys.Source }

    Write-Host "[BIZRA] ERROR: Python not found" -ForegroundColor Red
    exit 1
}

# ═══════════════════════════════════════════════════════
#  REGISTER — Create Task Scheduler entry
# ═══════════════════════════════════════════════════════
if ($Register) {
    $python = Find-Python

    # Verify daemon exists
    if (-not (Test-Path $daemonPy)) {
        Write-Host "[BIZRA] ERROR: Daemon not found: $daemonPy" -ForegroundColor Red
        exit 1
    }

    # Remove existing task if present
    $existing = Get-ScheduledTask -TaskName $taskName -TaskPath $taskPath -ErrorAction SilentlyContinue
    if ($existing) {
        Unregister-ScheduledTask -TaskName $taskName -TaskPath $taskPath -Confirm:$false
        Write-Host "[BIZRA] Removed existing task" -ForegroundColor Yellow
    }

    # Action: pythonw.exe core/sovereign/kernel_daemon.py
    $action = New-ScheduledTaskAction `
        -Execute $python `
        -Argument """$daemonPy""" `
        -WorkingDirectory $bizraRoot

    # Triggers: At Logon (current user) + On Workstation Unlock
    $triggerLogon  = New-ScheduledTaskTrigger -AtLogOn
    $triggerLogon.UserId = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

    # Settings: resilient, battery-friendly, auto-restart
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -RestartCount 3 `
        -RestartInterval (New-TimeSpan -Minutes 1) `
        -ExecutionTimeLimit (New-TimeSpan -Days 0) `
        -MultipleInstances IgnoreNew `
        -Priority 4

    # Register
    Register-ScheduledTask `
        -TaskName $taskName `
        -TaskPath $taskPath `
        -Action $action `
        -Trigger $triggerLogon `
        -Settings $settings `
        -Description "BIZRA Sovereign Kernel Daemon — auto-start, self-healing, constitutional AI runtime" `
        -RunLevel Limited | Out-Null

    # Save registration state
    if (-not (Test-Path $stateDir)) { New-Item -ItemType Directory -Path $stateDir -Force | Out-Null }
    @{
        registered = $true
        taskName   = $taskName
        taskPath   = $taskPath
        python     = $python
        daemon     = $daemonPy
        timestamp  = (Get-Date -Format "o")
        user       = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    } | ConvertTo-Json | Set-Content (Join-Path $stateDir "autostart.json") -Encoding UTF8

    Write-Host ""
    Write-Host "  ╔══════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "  ║   BIZRA SOVEREIGN KERNEL — AUTO-START REGISTERED ║" -ForegroundColor Cyan
    Write-Host "  ╠══════════════════════════════════════════════════╣" -ForegroundColor Cyan
    Write-Host "  ║  Task:    $taskName" -ForegroundColor White
    Write-Host "  ║  Python:  $($python | Split-Path -Leaf)" -ForegroundColor Gray
    Write-Host "  ║  Trigger: At Logon (current user)" -ForegroundColor Gray
    Write-Host "  ║  Restart: 3x with 1-min interval" -ForegroundColor Gray
    Write-Host "  ║  Battery: Yes (won't stop on battery)" -ForegroundColor Gray
    Write-Host "  ║                                                  ║" -ForegroundColor Cyan
    Write-Host "  ║  Next login → http://127.0.0.1:9740 auto-opens   ║" -ForegroundColor Green
    Write-Host "  ╚══════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
    exit 0
}

# ═══════════════════════════════════════════════════════
#  UNREGISTER — Remove Task Scheduler entry
# ═══════════════════════════════════════════════════════
if ($Unregister) {
    $existing = Get-ScheduledTask -TaskName $taskName -TaskPath $taskPath -ErrorAction SilentlyContinue
    if ($existing) {
        Unregister-ScheduledTask -TaskName $taskName -TaskPath $taskPath -Confirm:$false
        Write-Host "[BIZRA] Auto-start REMOVED: $taskName" -ForegroundColor Yellow

        # Clean up state
        $autoFile = Join-Path $stateDir "autostart.json"
        if (Test-Path $autoFile) { Remove-Item $autoFile -Force }
    } else {
        Write-Host "[BIZRA] No auto-start task found." -ForegroundColor Gray
    }
    exit 0
}

# ═══════════════════════════════════════════════════════
#  STATUS — Show current state
# ═══════════════════════════════════════════════════════
if ($Status) {
    Write-Host ""
    Write-Host "  BIZRA Sovereign Kernel Status" -ForegroundColor Cyan
    Write-Host "  $('─' * 40)" -ForegroundColor DarkGray

    # Task Scheduler
    $task = Get-ScheduledTask -TaskName $taskName -TaskPath $taskPath -ErrorAction SilentlyContinue
    if ($task) {
        $info = $task | Get-ScheduledTaskInfo
        Write-Host "  Task:         REGISTERED" -ForegroundColor Green
        Write-Host "  State:        $($task.State)" -ForegroundColor White
        Write-Host "  Last Run:     $($info.LastRunTime)" -ForegroundColor Gray
        Write-Host "  Next Run:     $($info.NextRunTime)" -ForegroundColor Gray
        Write-Host "  Last Result:  $($info.LastTaskResult)" -ForegroundColor Gray
    } else {
        Write-Host "  Task:         NOT REGISTERED" -ForegroundColor Yellow
    }

    # Daemon process
    if (Test-Path $pidFile) {
        $pid = [int](Get-Content $pidFile -ErrorAction SilentlyContinue)
        $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Host "  Daemon PID:   $pid (running)" -ForegroundColor Green
            Write-Host "  Memory:       $([math]::Round($proc.WorkingSet64 / 1MB, 1)) MB" -ForegroundColor Gray
        } else {
            Write-Host "  Daemon PID:   $pid (stale PID file)" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  Daemon:       NOT RUNNING" -ForegroundColor Yellow
    }

    # Init state
    $initFile = Join-Path $stateDir "kernel_initialized.json"
    if (Test-Path $initFile) {
        $init = Get-Content $initFile -Raw | ConvertFrom-Json
        Write-Host "  Initialized:  YES ($($init.userName))" -ForegroundColor Green
        Write-Host "  Version:      $($init.version)" -ForegroundColor Gray
    } else {
        Write-Host "  Initialized:  NO (installer will show)" -ForegroundColor Yellow
    }

    Write-Host ""
    exit 0
}

# ═══════════════════════════════════════════════════════
#  START — Launch daemon now
# ═══════════════════════════════════════════════════════
if ($Start) {
    $python = Find-Python

    # Check if already running
    if (Test-Path $pidFile) {
        $pid = [int](Get-Content $pidFile -ErrorAction SilentlyContinue)
        $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Host "[BIZRA] Daemon already running (PID $pid)" -ForegroundColor Yellow
            exit 0
        }
    }

    Write-Host "[BIZRA] Starting Sovereign Kernel Daemon..." -ForegroundColor Cyan
    Start-Process -FilePath $python -ArgumentList """$daemonPy""" -WorkingDirectory $bizraRoot -WindowStyle Hidden
    Start-Sleep -Seconds 2

    # Verify
    if (Test-Path $pidFile) {
        $pid = Get-Content $pidFile
        Write-Host "[BIZRA] Daemon started (PID $pid) — http://127.0.0.1:9740" -ForegroundColor Green
    } else {
        Write-Host "[BIZRA] Daemon may have failed to start. Check $stateDir\kernel.log" -ForegroundColor Red
    }
    exit 0
}

# ═══════════════════════════════════════════════════════
#  STOP — Kill running daemon
# ═══════════════════════════════════════════════════════
if ($Stop) {
    if (Test-Path $pidFile) {
        $pid = [int](Get-Content $pidFile -ErrorAction SilentlyContinue)
        $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
        if ($proc) {
            Stop-Process -Id $pid -Force
            Write-Host "[BIZRA] Daemon stopped (PID $pid)" -ForegroundColor Yellow
        } else {
            Write-Host "[BIZRA] No process at PID $pid (stale)" -ForegroundColor Gray
        }
        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    } else {
        Write-Host "[BIZRA] No daemon running" -ForegroundColor Gray
    }
    exit 0
}

# ── No flag provided ──
Write-Host "Usage: .\sovereign-kernel-service.ps1 [-Register|-Unregister|-Status|-Start|-Stop]" -ForegroundColor Yellow
