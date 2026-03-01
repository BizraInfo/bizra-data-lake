#Requires -RunAsAdministrator
<#
.SYNOPSIS
    BIZRA Resource Guardian v1.0 - Automated Resource Management for NODE0
.DESCRIPTION
    Continuously monitors system resources and automatically:
    - Kills non-essential processes when thresholds are exceeded
    - Manages disk I/O by controlling Windows services
    - Protects BIZRA-critical processes (WSL, Docker, LM Studio, NVIDIA)
    - Logs all actions for audit trail
.AUTHOR
    BIZRA Cognitive Architecture - Built for mumo's NODE0
#>

param(
    [int]$CpuThreshold = 80,
    [int]$MemoryThreshold = 75,
    [int]$DiskThreshold = 70,
    [int]$CheckIntervalSeconds = 30,
    [switch]$DaemonMode,
    [switch]$DryRun
)

# ============================================================
# CONFIGURATION
# ============================================================

$LogDir = "$env:USERPROFILE\.bizra\logs"
$ConfigPath = "$env:USERPROFILE\.bizra\guardian-config.json"

# BIZRA-critical processes - NEVER touch these
$ProtectedProcesses = @(
    "wsl", "wslhost", "wslservice",
    "docker", "com.docker.*", "Docker Desktop",
    "lms", "LM Studio",
    "nvidia*", "nvcontainer", "nvrla",
    "chrome", "msedge",
    "explorer", "csrss", "lsass", "svchost", "System",
    "code", "cursor", "WindowsTerminal", "pwsh", "powershell",
    "MsMpEng", "SecurityHealthService",
    "conhost", "dwm", "fontdrvhost", "sihost",
    "claude"
)

# Non-essential processes - safe to kill when resources are tight
$KillableProcesses = @(
    @{ Name = "MSPCManagerCore";          Priority = 1; SavesMB = 95  },
    @{ Name = "Teams";                    Priority = 1; SavesMB = 175 },
    @{ Name = "OneDrive";                 Priority = 2; SavesMB = 62  },
    @{ Name = "*Xbox*";                   Priority = 1; SavesMB = 20  },
    @{ Name = "GameBar*";                 Priority = 1; SavesMB = 6   },
    @{ Name = "*Gaming*";                 Priority = 1; SavesMB = 16  },
    @{ Name = "SearchHost";              Priority = 2; SavesMB = 41  },
    @{ Name = "SearchIndexer";            Priority = 2; SavesMB = 30  },
    @{ Name = "StartMenuExperienceHost";  Priority = 3; SavesMB = 10  },
    @{ Name = "Widgets";                  Priority = 1; SavesMB = 50  },
    @{ Name = "PhoneExperienceHost";      Priority = 1; SavesMB = 15  },
    @{ Name = "YourPhone";                Priority = 1; SavesMB = 20  },
    @{ Name = "SkypeApp";                 Priority = 2; SavesMB = 40  },
    @{ Name = "Cortana";                  Priority = 1; SavesMB = 30  },
    @{ Name = "HelpPane";                 Priority = 1; SavesMB = 5   },
    @{ Name = "Video.UI";                 Priority = 2; SavesMB = 15  },
    @{ Name = "Music.UI";                 Priority = 2; SavesMB = 15  }
)

# Services that hammer disk I/O - stop when disk is stressed
$DiskHeavyServices = @(
    @{ Name = "defragsvc";        Display = "Disk Defragmenter" },
    @{ Name = "WSearch";          Display = "Windows Search Indexer" },
    @{ Name = "SysMain";          Display = "Superfetch/SysMain" },
    @{ Name = "DiagTrack";        Display = "Diagnostics Tracking" },
    @{ Name = "dmwappushservice"; Display = "WAP Push Service" },
    @{ Name = "WbioSrvc";        Display = "Biometric Service" }
)

# ============================================================
# LOGGING
# ============================================================

function Initialize-Logging {
    if (-not (Test-Path $LogDir)) {
        New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
    }
}

function Write-GuardianLog {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logLine = "[$timestamp] [$Level] $Message"
    $logFile = Join-Path $LogDir "guardian-$(Get-Date -Format 'yyyy-MM-dd').log"
    Add-Content -Path $logFile -Value $logLine

    $color = switch ($Level) {
        "INFO"    { "Cyan" }
        "ACTION"  { "Green" }
        "WARNING" { "Yellow" }
        "ERROR"   { "Red" }
        "METRIC"  { "DarkGray" }
        default   { "White" }
    }
    Write-Host $logLine -ForegroundColor $color
}

# ============================================================
# RESOURCE MONITORING
# ============================================================

function Get-SystemMetrics {
    $cpu = (Get-CimInstance -ClassName Win32_Processor |
            Measure-Object -Property LoadPercentage -Average).Average

    $os = Get-CimInstance -ClassName Win32_OperatingSystem
    $totalMem = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
    $freeMem = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
    $usedMemPct = [math]::Round((($totalMem - $freeMem) / $totalMem) * 100, 1)

    # Disk I/O via performance counter
    $diskPct = try {
        [math]::Round((Get-Counter '\PhysicalDisk(_Total)\% Disk Time' -ErrorAction Stop).CounterSamples[0].CookedValue, 1)
    } catch { 0 }

    # Cap at 100
    if ($diskPct -gt 100) { $diskPct = 100 }

    return @{
        CPU         = [math]::Round($cpu, 1)
        MemoryPct   = $usedMemPct
        MemoryUsed  = [math]::Round($totalMem - $freeMem, 1)
        MemoryTotal = $totalMem
        MemoryFree  = $freeMem
        DiskPct     = $diskPct
    }
}

# ============================================================
# PROCESS MANAGEMENT
# ============================================================

function Test-IsProtected {
    param([string]$ProcessName)
    foreach ($pattern in $ProtectedProcesses) {
        if ($ProcessName -like $pattern) { return $true }
    }
    return $false
}

function Stop-NonEssentialProcesses {
    param([int]$Urgency = 1)  # 1=low priority only, 2=medium, 3=aggressive

    $totalSaved = 0

    foreach ($proc in ($KillableProcesses | Where-Object { $_.Priority -le $Urgency })) {
        $running = Get-Process -Name $proc.Name -ErrorAction SilentlyContinue
        if ($running) {
            if ($DryRun) {
                Write-GuardianLog "[DRY RUN] Would kill: $($proc.Name) (~$($proc.SavesMB) MB)" "WARNING"
            } else {
                try {
                    $running | Stop-Process -Force -ErrorAction Stop
                    $totalSaved += $proc.SavesMB
                    Write-GuardianLog "Killed: $($proc.Name) -- freed ~$($proc.SavesMB) MB" "ACTION"
                } catch {
                    Write-GuardianLog "Failed to kill $($proc.Name): $($_.Exception.Message)" "ERROR"
                }
            }
        }
    }

    return $totalSaved
}

function Stop-DiskHeavyServices {
    foreach ($svc in $DiskHeavyServices) {
        $service = Get-Service -Name $svc.Name -ErrorAction SilentlyContinue
        if ($service -and $service.Status -eq 'Running') {
            if ($DryRun) {
                Write-GuardianLog "[DRY RUN] Would stop service: $($svc.Display)" "WARNING"
            } else {
                try {
                    Stop-Service -Name $svc.Name -Force -ErrorAction Stop
                    Write-GuardianLog "Stopped service: $($svc.Display)" "ACTION"
                } catch {
                    Write-GuardianLog "Failed to stop $($svc.Display): $($_.Exception.Message)" "ERROR"
                }
            }
        }
    }
}

function Find-ResourceHogs {
    # Find unexpected high-resource processes not in our protected list
    $hogs = Get-Process | Where-Object {
        $_.WorkingSet64 -gt 500MB -and
        -not (Test-IsProtected $_.ProcessName)
    } | Sort-Object WorkingSet64 -Descending | Select-Object -First 5

    foreach ($hog in $hogs) {
        $memMB = [math]::Round($hog.WorkingSet64 / 1MB, 0)
        Write-GuardianLog "Resource hog detected: $($hog.ProcessName) using $memMB MB" "WARNING"
    }

    return $hogs
}

# ============================================================
# PRIORITY MANAGEMENT - Boost BIZRA processes
# ============================================================

function Set-BIZRAPriorities {
    # Boost WSL and Docker priority
    $boostTargets = @("vmmem", "wsl", "wslhost", "docker", "lms")
    foreach ($target in $boostTargets) {
        Get-Process -Name $target -ErrorAction SilentlyContinue | ForEach-Object {
            try {
                $_.PriorityClass = 'AboveNormal'
            } catch { }
        }
    }

    # Lower non-essential process priority
    $lowerTargets = @("SearchIndexer", "OneDrive", "Teams", "MSPCManagerCore")
    foreach ($target in $lowerTargets) {
        Get-Process -Name $target -ErrorAction SilentlyContinue | ForEach-Object {
            try {
                $_.PriorityClass = 'BelowNormal'
            } catch { }
        }
    }
}

# ============================================================
# MAIN GUARDIAN LOOP
# ============================================================

function Invoke-GuardianCheck {
    $metrics = Get-SystemMetrics

    Write-GuardianLog ("CPU: $($metrics.CPU)% | RAM: $($metrics.MemoryPct)% ($($metrics.MemoryUsed)/$($metrics.MemoryTotal) GB) | Disk: $($metrics.DiskPct)%") "METRIC"

    $actionsNeeded = $false

    # --- DISK CRITICAL (most impactful on your machine) ---
    if ($metrics.DiskPct -gt $DiskThreshold) {
        Write-GuardianLog "Disk I/O at $($metrics.DiskPct)% -- stopping disk-heavy services" "WARNING"
        Stop-DiskHeavyServices
        $actionsNeeded = $true
    }

    # --- MEMORY PRESSURE ---
    if ($metrics.MemoryPct -gt $MemoryThreshold) {
        $urgency = if ($metrics.MemoryPct -gt 90) { 3 }
                   elseif ($metrics.MemoryPct -gt 85) { 2 }
                   else { 1 }

        Write-GuardianLog "Memory at $($metrics.MemoryPct)% -- killing non-essentials (urgency: $urgency)" "WARNING"
        $saved = Stop-NonEssentialProcesses -Urgency $urgency
        Write-GuardianLog "Freed approximately $saved MB" "ACTION"
        $actionsNeeded = $true
    }

    # --- CPU PRESSURE ---
    if ($metrics.CPU -gt $CpuThreshold) {
        Write-GuardianLog "CPU at $($metrics.CPU)% -- scanning for hogs" "WARNING"
        Find-ResourceHogs | Out-Null
        $actionsNeeded = $true
    }

    # --- Always boost BIZRA priorities ---
    Set-BIZRAPriorities

    if (-not $actionsNeeded) {
        Write-GuardianLog "System healthy -- no action needed" "INFO"
    }

    return $metrics
}

# ============================================================
# STARTUP OPTIMIZATION (run once)
# ============================================================

function Invoke-StartupCleanup {
    Write-GuardianLog "=== BIZRA Startup Cleanup ===" "INFO"

    # Kill all non-essential immediately
    Stop-NonEssentialProcesses -Urgency 3

    # Stop disk-heavy services
    Stop-DiskHeavyServices

    # Set services to Manual/Disabled
    $disableServices = @(
        @{ Name = "XblAuthManager";    StartType = "Disabled" },
        @{ Name = "XblGameSave";       StartType = "Disabled" },
        @{ Name = "XboxGipSvc";        StartType = "Disabled" },
        @{ Name = "XboxNetApiSvc";     StartType = "Disabled" },
        @{ Name = "defragsvc";         StartType = "Manual" },
        @{ Name = "WSearch";           StartType = "Manual" },
        @{ Name = "SysMain";           StartType = "Manual" },
        @{ Name = "DiagTrack";         StartType = "Disabled" },
        @{ Name = "dmwappushservice";  StartType = "Disabled" },
        @{ Name = "MapsBroker";        StartType = "Disabled" },
        @{ Name = "lfsvc";             StartType = "Disabled" },
        @{ Name = "RetailDemo";        StartType = "Disabled" }
    )

    foreach ($svc in $disableServices) {
        try {
            Set-Service -Name $svc.Name -StartupType $svc.StartType -ErrorAction Stop
            Write-GuardianLog "Service '$($svc.Name)' set to $($svc.StartType)" "ACTION"
        } catch {
            # Service may not exist
        }
    }

    # Boost BIZRA priorities
    Set-BIZRAPriorities

    Write-GuardianLog "=== Startup Cleanup Complete ===" "INFO"
}

# ============================================================
# ENTRY POINT
# ============================================================

Initialize-Logging
Write-GuardianLog "========================================" "INFO"
Write-GuardianLog " BIZRA Resource Guardian v1.0 -- NODE0" "INFO"
Write-GuardianLog "========================================" "INFO"
Write-GuardianLog "Thresholds -- CPU: $CpuThreshold% | RAM: $MemoryThreshold% | Disk: $DiskThreshold%" "INFO"

if ($DryRun) {
    Write-GuardianLog "*** DRY RUN MODE -- no actions will be taken ***" "WARNING"
}

# Always run startup cleanup first
Invoke-StartupCleanup

if ($DaemonMode) {
    Write-GuardianLog "Entering daemon mode (checking every ${CheckIntervalSeconds}s)..." "INFO"
    Write-GuardianLog "Press Ctrl+C to stop" "INFO"

    while ($true) {
        try {
            Invoke-GuardianCheck | Out-Null
        } catch {
            Write-GuardianLog "Error in check cycle: $($_.Exception.Message)" "ERROR"
        }
        Start-Sleep -Seconds $CheckIntervalSeconds
    }
} else {
    # Single check
    Invoke-GuardianCheck | Out-Null
    Write-GuardianLog "Single check complete. Use -DaemonMode for continuous monitoring." "INFO"
}
