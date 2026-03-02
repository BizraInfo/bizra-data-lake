# BIZRA Node0 Performance Recovery Orchestrator
# ---------------------------------------------------------------------------
# Goal:
#   1) Diagnose Windows+WSL performance bottlenecks with evidence.
#   2) Execute a safe, ordered remediation plan (opt-in).
#
# Usage (PowerShell as Administrator recommended):
#   .\node0_performance_recovery.ps1 -Mode Analyze
#   .\node0_performance_recovery.ps1 -Mode Remediate -DryRun
#   .\node0_performance_recovery.ps1 -Mode Remediate -DryRun:$false -NoPrompt
#
# Standing on Giants:
#   - WSL2 architecture (Microsoft)
#   - Docker Desktop disk lifecycle
#   - Fail-closed operations: DryRun default, explicit opt-in for mutation.

param(
    [ValidateSet("Analyze", "Remediate")]
    [string]$Mode = "Analyze",
    [bool]$DryRun = $true,
    [switch]$NoPrompt,
    [switch]$IncludeCompaction,
    [switch]$CleanHuggingFaceCache,
    [int]$DockerVhdxWarningGB = 200,
    [int]$WslVhdxWarningGB = 120
)

$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host "=== $Title ===" -ForegroundColor Cyan
}

function Get-SizeGB {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return 0.0 }
    $item = Get-Item $Path -ErrorAction SilentlyContinue
    if (-not $item) { return 0.0 }
    if ($item.PSIsContainer) {
        $bytes = (Get-ChildItem $Path -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
        if (-not $bytes) { return 0.0 }
        return [math]::Round(($bytes / 1GB), 2)
    }
    return [math]::Round(($item.Length / 1GB), 2)
}

function Parse-WslConfig {
    param([string]$Path)
    $result = @{
        exists = $false
        memory = $null
        processors = $null
        swap = $null
    }
    if (-not (Test-Path $Path)) { return $result }

    $result.exists = $true
    $lines = Get-Content $Path -ErrorAction SilentlyContinue
    foreach ($line in $lines) {
        if ($line -match "^\s*memory\s*=\s*(.+)\s*$") { $result.memory = $Matches[1].Trim() }
        if ($line -match "^\s*processors\s*=\s*(.+)\s*$") { $result.processors = $Matches[1].Trim() }
        if ($line -match "^\s*swap\s*=\s*(.+)\s*$") { $result.swap = $Matches[1].Trim() }
    }
    return $result
}

function Get-CounterValue {
    param([string]$Path)
    try {
        $sample = Get-Counter -Counter $Path -ErrorAction Stop
        $value = $sample.CounterSamples | Select-Object -First 1 -ExpandProperty CookedValue
        if ($null -eq $value) { return $null }
        return [math]::Round([double]$value, 2)
    }
    catch {
        return $null
    }
}

function Get-TopProcesses {
    param([int]$Top = 8)
    $topCpu = @()
    $topMemory = @()
    try {
        $topCpu = Get-Process |
            Sort-Object CPU -Descending |
            Select-Object -First $Top `
                Name, Id,
                @{Name = "cpu_seconds"; Expression = {
                    if ($null -eq $_.CPU) { 0.0 } else { [math]::Round([double]$_.CPU, 2) }
                }}
    }
    catch {}

    try {
        $topMemory = Get-Process |
            Sort-Object WorkingSet64 -Descending |
            Select-Object -First $Top `
                Name, Id,
                @{Name = "working_set_gb"; Expression = { [math]::Round(($_.WorkingSet64 / 1GB), 2) }}
    }
    catch {}

    return @{
        top_cpu = $topCpu
        top_memory = $topMemory
    }
}

function Add-Recommendation {
    param(
        [System.Collections.Generic.List[object]]$List,
        [hashtable]$Seen,
        [string]$Id,
        [int]$Priority,
        [string]$Reason,
        [string]$CommandHint
    )
    if ($Seen.ContainsKey($Id)) { return }
    $Seen[$Id] = $true
    $List.Add([ordered]@{
        id = $Id
        priority = $Priority
        reason = $Reason
        command_hint = $CommandHint
    })
}

function Run-Step {
    param(
        [string]$Label,
        [string]$Command
    )
    Write-Host ""
    Write-Host "[STEP] $Label" -ForegroundColor Yellow
    if ($DryRun) {
        Write-Host "  [DRY-RUN] $Command" -ForegroundColor DarkYellow
        return
    }
    Write-Host "  [EXEC] $Command" -ForegroundColor Gray
    Invoke-Expression $Command
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

Write-Section "Node0 Performance Snapshot"

$localAppData = $env:LOCALAPPDATA
$userProfile = $env:USERPROFILE
$dockerVhdx = Join-Path $localAppData "Docker\wsl\disk\docker_data.vhdx"
$ubuntuVhdx = Join-Path $localAppData "Packages\CanonicalGroupLimited.Ubuntu_79rhkp1fndgsc\LocalState\ext4.vhdx"
$huggingFaceHub = Join-Path $userProfile ".cache\huggingface\hub"
$localTemp = Join-Path $localAppData "Temp"
$localPackages = Join-Path $localAppData "Packages"
$wslConfigPath = Join-Path $userProfile ".wslconfig"

$diskC = Get-PSDrive -Name C
$diskUsedPct = if ($diskC.Used + $diskC.Free -gt 0) {
    [math]::Round(($diskC.Used / ($diskC.Used + $diskC.Free)) * 100, 1)
} else { 0.0 }

$wslCfg = Parse-WslConfig -Path $wslConfigPath
$os = Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue
$pf = Get-CimInstance Win32_PageFileUsage -ErrorAction SilentlyContinue | Select-Object -First 1
$cpuTotalPct = Get-CounterValue '\Processor(_Total)\% Processor Time'
$diskQueueLen = Get-CounterValue '\PhysicalDisk(_Total)\Avg. Disk Queue Length'
$diskBusyPct = Get-CounterValue '\PhysicalDisk(_Total)\% Disk Time'
$processHotspots = Get-TopProcesses -Top 8

$snapshot = [ordered]@{
    timestamp = (Get-Date).ToString("o")
    host = $env:COMPUTERNAME
    mode = $Mode
    dry_run = $DryRun
    disk = @{
        c_used_percent = $diskUsedPct
        c_used_gb = [math]::Round(($diskC.Used / 1GB), 2)
        c_free_gb = [math]::Round(($diskC.Free / 1GB), 2)
    }
    virtual_memory = @{
        total_virtual_gb = if ($os) { [math]::Round(([double]$os.TotalVirtualMemorySize / 1MB), 2) } else { 0.0 }
        free_virtual_gb = if ($os) { [math]::Round(([double]$os.FreeVirtualMemory / 1MB), 2) } else { 0.0 }
        total_physical_gb = if ($os) { [math]::Round(([double]$os.TotalVisibleMemorySize / 1MB), 2) } else { 0.0 }
        free_physical_gb = if ($os) { [math]::Round(([double]$os.FreePhysicalMemory / 1MB), 2) } else { 0.0 }
        pagefile_allocated_gb = if ($pf) { [math]::Round(([double]$pf.AllocatedBaseSize / 1024), 2) } else { 0.0 }
        pagefile_current_gb = if ($pf) { [math]::Round(([double]$pf.CurrentUsage / 1024), 2) } else { 0.0 }
        pagefile_peak_gb = if ($pf) { [math]::Round(([double]$pf.PeakUsage / 1024), 2) } else { 0.0 }
    }
    telemetry = @{
        cpu_total_percent = $cpuTotalPct
        disk_queue_length = $diskQueueLen
        disk_busy_percent = $diskBusyPct
    }
    process_hotspots = $processHotspots
    wsl_config = $wslCfg
    large_paths_gb = @{
        docker_data_vhdx = Get-SizeGB $dockerVhdx
        ubuntu_ext4_vhdx = Get-SizeGB $ubuntuVhdx
        huggingface_hub = Get-SizeGB $huggingFaceHub
        local_temp = Get-SizeGB $localTemp
        local_packages = Get-SizeGB $localPackages
    }
}

$findings = New-Object System.Collections.Generic.List[object]
if ($snapshot.disk.c_used_percent -ge 80) {
    $findings.Add([ordered]@{
        severity = "high"
        id = "DISK_C_PRESSURE"
        message = "C: disk usage is above 80%; paging and random IO latency likely increase."
    })
}
if ($snapshot.large_paths_gb.docker_data_vhdx -ge $DockerVhdxWarningGB) {
    $findings.Add([ordered]@{
        severity = "critical"
        id = "DOCKER_VHDX_BLOAT"
        message = "Docker Desktop VHDX is oversized and likely causing IO pressure."
    })
}
if ($snapshot.large_paths_gb.ubuntu_ext4_vhdx -ge $WslVhdxWarningGB) {
    $findings.Add([ordered]@{
        severity = "high"
        id = "WSL_VHDX_BLOAT"
        message = "Ubuntu ext4.vhdx is large; compaction and cleanup recommended."
    })
}
if ($wslCfg.exists -and $wslCfg.memory -match "32GB|24GB|16GB") {
    $findings.Add([ordered]@{
        severity = "high"
        id = "WSL_MEMORY_CAP"
        message = "WSL memory cap appears constrained for high-end hardware."
    })
}
if ($wslCfg.exists -and $wslCfg.processors -match "16|12|8") {
    $findings.Add([ordered]@{
        severity = "medium"
        id = "WSL_CPU_CAP"
        message = "WSL processor cap appears lower than expected for Node0 hardware."
    })
}
if ($snapshot.large_paths_gb.huggingface_hub -ge 30) {
    $findings.Add([ordered]@{
        severity = "medium"
        id = "HF_CACHE_PRESSURE"
        message = "HuggingFace cache is large; cleanup can recover significant space."
    })
}
if ($snapshot.virtual_memory.pagefile_allocated_gb -lt 8) {
    $findings.Add([ordered]@{
        severity = "high"
        id = "PAGEFILE_UNDERSIZED"
        message = "Pagefile is undersized; elevated operations may fail under pressure."
    })
}
if ($snapshot.virtual_memory.free_virtual_gb -lt 2) {
    $findings.Add([ordered]@{
        severity = "critical"
        id = "LOW_VIRTUAL_MEMORY"
        message = "Free virtual memory is below 2 GB; heavy Windows operations can fail."
    })
}
if (($null -ne $snapshot.telemetry.cpu_total_percent) -and ($snapshot.telemetry.cpu_total_percent -ge 85)) {
    $findings.Add([ordered]@{
        severity = "high"
        id = "CPU_PRESSURE"
        message = "CPU saturation is above 85%; foreground and build latency likely degraded."
    })
}
if (($null -ne $snapshot.telemetry.disk_queue_length) -and ($snapshot.telemetry.disk_queue_length -ge 2)) {
    $findings.Add([ordered]@{
        severity = "high"
        id = "DISK_QUEUE_PRESSURE"
        message = "Disk queue length is elevated; storage contention likely the primary slowdown."
    })
}
if ($snapshot.virtual_memory.free_physical_gb -lt 8) {
    $findings.Add([ordered]@{
        severity = "high"
        id = "LOW_FREE_RAM"
        message = "Free physical RAM is below 8 GB; memory pressure and paging likely."
    })
}

$recommendations = New-Object System.Collections.Generic.List[object]
$recommendationSeen = @{}
foreach ($f in $findings) {
    switch ($f.id) {
        "DOCKER_VHDX_BLOAT" {
            Add-Recommendation -List $recommendations -Seen $recommendationSeen -Id "RUN_DOCKER_VOLUME_GOVERNANCE" -Priority 1 -Reason $f.message -CommandHint "python scripts/ops/docker_volume_governance.py reclaim-k3d --restart-cluster"
            Add-Recommendation -List $recommendations -Seen $recommendationSeen -Id "RUN_VHDX_COMPACTION" -Priority 2 -Reason "Logical reclaim completed but host disk still high." -CommandHint "scripts/ops/VHDX-COMPACTION-LAUNCHER.bat -Mode Compact -DryRun:$false -Target docker"
        }
        "WSL_VHDX_BLOAT" {
            Add-Recommendation -List $recommendations -Seen $recommendationSeen -Id "RUN_VHDX_COMPACTION" -Priority 2 -Reason $f.message -CommandHint "scripts/ops/VHDX-COMPACTION-LAUNCHER.bat -Mode Compact -DryRun:$false -Target both"
        }
        "PAGEFILE_UNDERSIZED" {
            Add-Recommendation -List $recommendations -Seen $recommendationSeen -Id "APPLY_PAGEFILE_GOVERNANCE" -Priority 1 -Reason $f.message -CommandHint "scripts/ops/PAGEFILE-GOVERNANCE-LAUNCHER.bat -Mode Apply -DryRun:$false"
        }
        "LOW_VIRTUAL_MEMORY" {
            Add-Recommendation -List $recommendations -Seen $recommendationSeen -Id "APPLY_PAGEFILE_GOVERNANCE" -Priority 1 -Reason $f.message -CommandHint "scripts/ops/PAGEFILE-GOVERNANCE-LAUNCHER.bat -Mode Apply -DryRun:$false"
        }
        "DISK_C_PRESSURE" {
            Add-Recommendation -List $recommendations -Seen $recommendationSeen -Id "PRUNE_TEMP_AND_CACHES" -Priority 3 -Reason $f.message -CommandHint "scripts/ops/node0_performance_recovery.ps1 -Mode Remediate -DryRun:$false"
        }
        "HF_CACHE_PRESSURE" {
            Add-Recommendation -List $recommendations -Seen $recommendationSeen -Id "CLEAN_HF_CACHE" -Priority 3 -Reason $f.message -CommandHint "scripts/ops/node0_performance_recovery.ps1 -Mode Remediate -DryRun:$false -CleanHuggingFaceCache"
        }
        "CPU_PRESSURE" {
            Add-Recommendation -List $recommendations -Seen $recommendationSeen -Id "REDUCE_BACKGROUND_LOAD" -Priority 2 -Reason $f.message -CommandHint "Close high-CPU processes from report.process_hotspots.top_cpu"
        }
        "DISK_QUEUE_PRESSURE" {
            Add-Recommendation -List $recommendations -Seen $recommendationSeen -Id "REDUCE_IO_CONTENTION" -Priority 2 -Reason $f.message -CommandHint "Pause active indexing/builds, then rerun compaction + docker reclaim"
        }
        "LOW_FREE_RAM" {
            Add-Recommendation -List $recommendations -Seen $recommendationSeen -Id "REDUCE_RAM_PRESSURE" -Priority 2 -Reason $f.message -CommandHint "Close memory-heavy processes from report.process_hotspots.top_memory"
        }
    }
}

$severityCounts = [ordered]@{
    critical = @($findings | Where-Object { $_.severity -eq "critical" }).Count
    high = @($findings | Where-Object { $_.severity -eq "high" }).Count
    medium = @($findings | Where-Object { $_.severity -eq "medium" }).Count
    low = @($findings | Where-Object { $_.severity -eq "low" }).Count
}
$severityWeights = @{ critical = 4; high = 3; medium = 2; low = 1 }
$dominant = $null
$dominantScore = -1
foreach ($f in $findings) {
    $score = $severityWeights[$f.severity]
    if ($score -gt $dominantScore) {
        $dominant = $f
        $dominantScore = $score
    }
}
$recommendationsSorted = @($recommendations | Sort-Object priority, id)
$dominantActionByFinding = @{
    DOCKER_VHDX_BLOAT = "RUN_DOCKER_VOLUME_GOVERNANCE"
    WSL_VHDX_BLOAT = "RUN_VHDX_COMPACTION"
    PAGEFILE_UNDERSIZED = "APPLY_PAGEFILE_GOVERNANCE"
    LOW_VIRTUAL_MEMORY = "APPLY_PAGEFILE_GOVERNANCE"
    LOW_FREE_RAM = "REDUCE_RAM_PRESSURE"
    DISK_QUEUE_PRESSURE = "REDUCE_IO_CONTENTION"
    CPU_PRESSURE = "REDUCE_BACKGROUND_LOAD"
    HF_CACHE_PRESSURE = "CLEAN_HF_CACHE"
    DISK_C_PRESSURE = "PRUNE_TEMP_AND_CACHES"
}
if (($null -ne $dominant) -and $dominantActionByFinding.ContainsKey($dominant.id)) {
    $preferred = $dominantActionByFinding[$dominant.id]
    $preferredItems = @($recommendationsSorted | Where-Object { $_.id -eq $preferred })
    if ($preferredItems.Count -gt 0) {
        $otherItems = @($recommendationsSorted | Where-Object { $_.id -ne $preferred })
        $recommendationsSorted = @($preferredItems + $otherItems)
    }
}
$recommendedNextStep = if ($recommendationsSorted.Count -gt 0) {
    $recommendationsSorted[0].id
} else {
    "MONITOR_ONLY"
}

$snapshot.recommendations = $recommendationsSorted
$snapshot.summary = [ordered]@{
    finding_count = $findings.Count
    severity_counts = $severityCounts
    dominant_bottleneck = if ($null -ne $dominant) { $dominant.id } else { $null }
    recommended_next_step = $recommendedNextStep
}

$snapshot.findings = $findings

Write-Host ("C: used: {0}% ({1} GB used / {2} GB free)" -f `
    $snapshot.disk.c_used_percent, $snapshot.disk.c_used_gb, $snapshot.disk.c_free_gb)
Write-Host ("Docker VHDX: {0} GB" -f $snapshot.large_paths_gb.docker_data_vhdx)
Write-Host ("Ubuntu VHDX: {0} GB" -f $snapshot.large_paths_gb.ubuntu_ext4_vhdx)
Write-Host ("HuggingFace cache: {0} GB" -f $snapshot.large_paths_gb.huggingface_hub)
Write-Host ("Virtual memory free/total: {0} / {1} GB" -f $snapshot.virtual_memory.free_virtual_gb, $snapshot.virtual_memory.total_virtual_gb)
Write-Host ("Pagefile allocated/current/peak: {0}/{1}/{2} GB" -f $snapshot.virtual_memory.pagefile_allocated_gb, $snapshot.virtual_memory.pagefile_current_gb, $snapshot.virtual_memory.pagefile_peak_gb)
Write-Host ("CPU total: {0}% | Disk queue: {1} | Disk busy: {2}%" -f `
    $snapshot.telemetry.cpu_total_percent, $snapshot.telemetry.disk_queue_length, $snapshot.telemetry.disk_busy_percent)

if ($findings.Count -eq 0) {
    Write-Host "No critical pressure signals detected." -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Findings:" -ForegroundColor Yellow
    foreach ($f in $findings) {
        Write-Host ("  [{0}] {1}: {2}" -f $f.severity.ToUpper(), $f.id, $f.message)
    }
}
if ($snapshot.summary.dominant_bottleneck) {
    Write-Host ("Dominant bottleneck: {0}" -f $snapshot.summary.dominant_bottleneck) -ForegroundColor Magenta
}
if ($recommendationsSorted.Count -gt 0) {
    Write-Host ""
    Write-Host "Top recommended actions:" -ForegroundColor Yellow
    foreach ($r in $recommendationsSorted | Select-Object -First 3) {
        Write-Host ("  [P{0}] {1} - {2}" -f $r.priority, $r.id, $r.reason)
    }
}

$logDir = "C:\BIZRA-DATA-LAKE\logs"
if (-not (Test-Path $logDir)) {
    New-Item -Path $logDir -ItemType Directory -Force | Out-Null
}
$reportPath = Join-Path $logDir ("node0_performance_recovery_{0}.json" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
$snapshot | ConvertTo-Json -Depth 8 | Set-Content -Path $reportPath -Encoding UTF8
Write-Host ""
Write-Host "Report written to: $reportPath" -ForegroundColor Green

if ($Mode -eq "Analyze") {
    Write-Host "Analyze mode complete. No changes applied." -ForegroundColor Cyan
    exit 0
}

Write-Section "Remediation Plan"
Confirm-OrDie -PromptText "This will run cleanup commands that may remove cached data"

Run-Step -Label "Stop Docker Desktop processes" -Command @"
Get-Process 'Docker Desktop' -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process 'com.docker.backend' -ErrorAction SilentlyContinue | Stop-Process -Force
"@

Run-Step -Label "Docker cleanup (containers/images/volumes/build cache)" -Command "docker system prune -a --volumes -f"
Run-Step -Label "Builder cache cleanup" -Command "docker builder prune -a -f"

if ($CleanHuggingFaceCache) {
    Run-Step -Label "Clean HuggingFace cache" -Command "if (Test-Path '$huggingFaceHub') { Remove-Item '$huggingFaceHub\\*' -Recurse -Force -ErrorAction SilentlyContinue }"
}

Run-Step -Label "Shutdown WSL" -Command "wsl --shutdown"

if ($IncludeCompaction) {
    Run-Step -Label "Compact Docker VHDX" -Command "Optimize-VHD -Path `"$dockerVhdx`" -Mode Full"
    Run-Step -Label "Compact Ubuntu VHDX" -Command "Optimize-VHD -Path `"$ubuntuVhdx`" -Mode Full"
}

Write-Host ""
Write-Host "Remediation sequence complete." -ForegroundColor Green
Write-Host "If you changed .wslconfig, run: wsl --shutdown and restart WSL/Docker Desktop." -ForegroundColor Yellow
