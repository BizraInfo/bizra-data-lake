# BIZRA VHDX Compaction Governance (Windows-side)
# ---------------------------------------------------------------------------
# Goal:
#   1) Snapshot current VHDX pressure (Docker + Ubuntu).
#   2) Execute deterministic offline compaction with explicit safety gates.
#   3) Emit JSON evidence report with before/after deltas.
#
# Usage (run from Windows PowerShell, Admin recommended):
#   .\vhdx_compaction_governance.ps1 -Mode Analyze
#   .\vhdx_compaction_governance.ps1 -Mode Compact -DryRun
#   .\vhdx_compaction_governance.ps1 -Mode Compact -DryRun:$false -NoPrompt -Target docker
#
# Notes:
#   - Compact mode requires an offline WSL window (script runs wsl --shutdown).
#   - Do not run this from inside an interactive WSL shell.

param(
    [ValidateSet("Analyze", "Compact")]
    [string]$Mode = "Analyze",
    [object]$DryRun = $true,
    [ValidateSet("docker", "ubuntu", "both")]
    [string]$Target = "docker",
    [switch]$NoPrompt,
    [switch]$SkipElevation,
    [switch]$RestartDockerDesktop,
    [switch]$RequireAdminForCompact = $true,
    [double]$MinFreeVirtualMemoryGB = 1.0,
    [double]$MinPagefileAllocatedGB = 8.0
)

$ErrorActionPreference = "Stop"

function Resolve-Bool {
    param(
        [object]$Value,
        [bool]$Default = $true
    )
    if ($null -eq $Value) {
        return $Default
    }
    if ($Value -is [bool]) {
        return [bool]$Value
    }
    if ($Value -is [int] -or $Value -is [long]) {
        return ([int]$Value -ne 0)
    }

    $txt = "$Value".Trim().ToLowerInvariant()
    if ($txt -in @("true", "$true", "1", "yes", "y")) {
        return $true
    }
    if ($txt -in @("false", "$false", "0", "no", "n")) {
        return $false
    }

    throw "Invalid DryRun value '$Value'. Use true/false or 1/0."
}

$DryRun = Resolve-Bool -Value $DryRun -Default $true

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host "=== $Title ===" -ForegroundColor Cyan
}

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

function Get-SizeGB {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return 0.0 }
    $item = Get-Item $Path -ErrorAction SilentlyContinue
    if (-not $item) { return 0.0 }
    return [math]::Round(($item.Length / 1GB), 2)
}

function Get-VirtualMemorySnapshot {
    $os = Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue
    $pf = Get-CimInstance Win32_PageFileUsage -ErrorAction SilentlyContinue | Select-Object -First 1

    $totalVirtualGb = if ($os) { [math]::Round(([double]$os.TotalVirtualMemorySize / 1MB), 2) } else { 0.0 }
    $freeVirtualGb = if ($os) { [math]::Round(([double]$os.FreeVirtualMemory / 1MB), 2) } else { 0.0 }
    $totalPhysicalGb = if ($os) { [math]::Round(([double]$os.TotalVisibleMemorySize / 1MB), 2) } else { 0.0 }
    $freePhysicalGb = if ($os) { [math]::Round(([double]$os.FreePhysicalMemory / 1MB), 2) } else { 0.0 }
    $pageAllocGb = if ($pf) { [math]::Round(([double]$pf.AllocatedBaseSize / 1024), 2) } else { 0.0 }
    $pageCurrentGb = if ($pf) { [math]::Round(([double]$pf.CurrentUsage / 1024), 2) } else { 0.0 }
    $pagePeakGb = if ($pf) { [math]::Round(([double]$pf.PeakUsage / 1024), 2) } else { 0.0 }

    return @{
        total_virtual_gb = $totalVirtualGb
        free_virtual_gb = $freeVirtualGb
        total_physical_gb = $totalPhysicalGb
        free_physical_gb = $freePhysicalGb
        pagefile_allocated_gb = $pageAllocGb
        pagefile_current_gb = $pageCurrentGb
        pagefile_peak_gb = $pagePeakGb
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

function Get-WslStateRaw {
    $raw = (& wsl -l -v --all 2>&1 | Out-String)
    return $raw.Replace("`0", "")
}

function Test-WslDistroExists {
    param([string]$Name)
    try {
        $list = & wsl -l -q 2>$null
        if ($LASTEXITCODE -ne 0) {
            return $false
        }
        $trimmed = @($list | ForEach-Object { "$_".Trim() } | Where-Object { $_ })
        return $trimmed -contains $Name
    }
    catch {
        return $false
    }
}

function Get-ServiceState {
    param([string]$Name)
    try {
        $svc = Get-Service -Name $Name -ErrorAction Stop
        return @{
            name = $svc.Name
            status = "$($svc.Status)"
            start_type = "$($svc.StartType)"
        }
    }
    catch {
        return @{
            name = $Name
            status = "missing"
            start_type = "unknown"
        }
    }
}

function Run-Step {
    param(
        [string]$Label,
        [scriptblock]$Action,
        [switch]$CheckExitCode
    )
    Write-Host ""
    Write-Host "[STEP] $Label" -ForegroundColor Yellow
    if ($DryRun) {
        Write-Host "  [DRY-RUN] skipped mutation" -ForegroundColor DarkYellow
        return @{
            label = $Label
            dry_run = $true
            success = $true
            output = "dry-run"
        }
    }

    try {
        $previousExit = $LASTEXITCODE
        $output = & $Action 2>&1 | Out-String
        $exitCode = if ($null -ne $LASTEXITCODE) { [int]$LASTEXITCODE } else { 0 }
        $isSuccess = $true
        if ($CheckExitCode -and ($exitCode -ne 0)) {
            $isSuccess = $false
        }
        $LASTEXITCODE = $previousExit
        return @{
            label = $Label
            dry_run = $false
            success = $isSuccess
            exit_code = $exitCode
            output = $output
        }
    }
    catch {
        return @{
            label = $Label
            dry_run = $false
            success = $false
            exit_code = -1
            output = $_.Exception.Message
        }
    }
}

function Invoke-DiskpartCompact {
    param([string]$VhdxPath)

    $tmp = Join-Path $env:TEMP ("compact_" + [guid]::NewGuid().ToString("N") + ".txt")
    $script = @"
select vdisk file="$VhdxPath"
attach vdisk readonly
compact vdisk
detach vdisk
exit
"@
    Set-Content -Path $tmp -Value $script -Encoding ASCII
    try {
        $output = (& diskpart /s $tmp 2>&1 | Out-String)
        $ok = (
            ($LASTEXITCODE -eq 0) -and
            ($output -match "(?i)successfully compacted|successfully completed")
        )
        return @{
            path = $VhdxPath
            success = $ok
            exit_code = $LASTEXITCODE
            output = $output
        }
    }
    finally {
        Remove-Item -Path $tmp -Force -ErrorAction SilentlyContinue
    }
}

# Auto-elevate for mutating compaction runs unless explicitly bypassed.
if ((-not $DryRun) -and $RequireAdminForCompact -and (-not (Test-IsAdministrator)) -and (-not $SkipElevation)) {
    $scriptPath = $MyInvocation.MyCommand.Path
    if (-not $scriptPath) {
        throw "Unable to determine script path for elevation relaunch."
    }

    $argList = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "`"$scriptPath`"",
        "-Mode", $Mode,
        "-DryRun", "$DryRun",
        "-Target", $Target,
        "-MinFreeVirtualMemoryGB", "$MinFreeVirtualMemoryGB",
        "-MinPagefileAllocatedGB", "$MinPagefileAllocatedGB",
        "-SkipElevation"
    )
    if ($NoPrompt) { $argList += "-NoPrompt" }
    if ($RestartDockerDesktop) { $argList += "-RestartDockerDesktop" }
    if ($RequireAdminForCompact) { $argList += "-RequireAdminForCompact" }

    Write-Host "Requesting Administrator elevation for compaction..." -ForegroundColor Yellow
    try {
        $proc = Start-Process -FilePath "powershell.exe" -ArgumentList $argList -Verb RunAs -Wait -PassThru
        exit $proc.ExitCode
    }
    catch {
        Write-Host "Elevation was declined or failed." -ForegroundColor Red
        exit 2
    }
}

Write-Section "VHDX Compaction Snapshot"

$localAppData = $env:LOCALAPPDATA
$dockerVhdx = Join-Path $localAppData "Docker\wsl\disk\docker_data.vhdx"
$ubuntuVhdx = Join-Path $localAppData "Packages\CanonicalGroupLimited.Ubuntu_79rhkp1fndgsc\LocalState\ext4.vhdx"
$dockerDesktopExe = "C:\Program Files\Docker\Docker\Docker Desktop.exe"

$diskC = Get-PSDrive -Name C
$snapshotBefore = @{
    timestamp = (Get-Date).ToString("o")
    mode = $Mode
    dry_run = $DryRun
    is_admin = (Test-IsAdministrator)
    target = $Target
    disk = @{
        c_used_gb = [math]::Round(($diskC.Used / 1GB), 2)
        c_free_gb = [math]::Round(($diskC.Free / 1GB), 2)
    }
    virtual_memory = (Get-VirtualMemorySnapshot)
    vhdx_gb = @{
        docker_data = Get-SizeGB $dockerVhdx
        ubuntu_ext4 = Get-SizeGB $ubuntuVhdx
    }
    services = @{
        docker = (Get-ServiceState -Name "com.docker.service")
        vmcompute = (Get-ServiceState -Name "vmcompute")
        wsl = (Get-ServiceState -Name "WSLService")
    }
    wsl_state = Get-WslStateRaw
}

Write-Host ("Docker VHDX: {0} GB" -f $snapshotBefore.vhdx_gb.docker_data)
Write-Host ("Ubuntu VHDX: {0} GB" -f $snapshotBefore.vhdx_gb.ubuntu_ext4)
Write-Host ("C: used/free: {0} / {1} GB" -f $snapshotBefore.disk.c_used_gb, $snapshotBefore.disk.c_free_gb)
Write-Host ("Virtual memory free/total: {0} / {1} GB" -f $snapshotBefore.virtual_memory.free_virtual_gb, $snapshotBefore.virtual_memory.total_virtual_gb)
Write-Host ("Pagefile allocated/current/peak: {0}/{1}/{2} GB" -f $snapshotBefore.virtual_memory.pagefile_allocated_gb, $snapshotBefore.virtual_memory.pagefile_current_gb, $snapshotBefore.virtual_memory.pagefile_peak_gb)

$logDir = "C:\BIZRA-DATA-LAKE\logs"
if (-not (Test-Path $logDir)) {
    New-Item -Path $logDir -ItemType Directory -Force | Out-Null
}

$report = @{
    before = $snapshotBefore
    steps = @()
    compact_results = @()
    after = $null
    preflight_failure = $null
    reclaimed_gb = @{
        docker_data = 0.0
        ubuntu_ext4 = 0.0
    }
}

if ($Mode -eq "Analyze") {
    $path = Join-Path $logDir ("vhdx_compaction_governance_{0}.json" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
    $report | ConvertTo-Json -Depth 8 | Set-Content -Path $path -Encoding UTF8
    Write-Host ""
    Write-Host "Analyze mode complete. No changes applied." -ForegroundColor Cyan
    Write-Host "Report written to: $path" -ForegroundColor Green
    exit 0
}

if ((-not $DryRun) -and $RequireAdminForCompact -and (-not $snapshotBefore.is_admin)) {
    $report.preflight_failure = @{
        code = "ADMIN_REQUIRED"
        message = "Compact mode requires Administrator privileges."
    }
    $path = Join-Path $logDir ("vhdx_compaction_governance_{0}.json" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
    $report | ConvertTo-Json -Depth 10 | Set-Content -Path $path -Encoding UTF8
    Write-Host ""
    Write-Host "Compact mode requires Administrator privileges." -ForegroundColor Red
    Write-Host "Run from elevated PowerShell or use an elevated launcher." -ForegroundColor Yellow
    Write-Host "Report written to: $path" -ForegroundColor Green
    exit 2
}

if ((-not $DryRun) -and ($snapshotBefore.virtual_memory.pagefile_allocated_gb -lt $MinPagefileAllocatedGB)) {
    $report.preflight_failure = @{
        code = "PAGEFILE_TOO_SMALL"
        message = "Pagefile allocation is below safe compaction threshold."
        pagefile_allocated_gb = $snapshotBefore.virtual_memory.pagefile_allocated_gb
        required_min_gb = $MinPagefileAllocatedGB
    }
    $path = Join-Path $logDir ("vhdx_compaction_governance_{0}.json" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
    $report | ConvertTo-Json -Depth 10 | Set-Content -Path $path -Encoding UTF8
    Write-Host ""
    Write-Host "Pagefile allocation is below safe compaction threshold." -ForegroundColor Red
    Write-Host ("Current pagefile allocation: {0} GB (minimum {1} GB)" -f $snapshotBefore.virtual_memory.pagefile_allocated_gb, $MinPagefileAllocatedGB) -ForegroundColor Yellow
    Write-Host "Increase pagefile size and retry." -ForegroundColor Yellow
    Write-Host "Report written to: $path" -ForegroundColor Green
    exit 4
}

if ((-not $DryRun) -and ($snapshotBefore.virtual_memory.free_virtual_gb -lt $MinFreeVirtualMemoryGB)) {
    $report.preflight_failure = @{
        code = "LOW_VIRTUAL_MEMORY"
        message = "Insufficient free virtual memory for safe compaction."
        free_virtual_gb = $snapshotBefore.virtual_memory.free_virtual_gb
        required_min_gb = $MinFreeVirtualMemoryGB
    }
    $path = Join-Path $logDir ("vhdx_compaction_governance_{0}.json" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
    $report | ConvertTo-Json -Depth 10 | Set-Content -Path $path -Encoding UTF8
    Write-Host ""
    Write-Host "Insufficient free virtual memory for safe compaction." -ForegroundColor Red
    Write-Host ("Current free virtual memory: {0} GB (minimum {1} GB)" -f $snapshotBefore.virtual_memory.free_virtual_gb, $MinFreeVirtualMemoryGB) -ForegroundColor Yellow
    Write-Host "Increase pagefile size and close heavy apps, then retry." -ForegroundColor Yellow
    Write-Host "Report written to: $path" -ForegroundColor Green
    exit 3
}

Write-Section "Compaction Plan"
Write-Host "This will stop Docker services, run wsl --shutdown, and compact VHDX files."
Write-Host "Target: $Target"
Confirm-OrDie -PromptText "Proceed with offline compaction"

$steps = New-Object System.Collections.Generic.List[object]
$compactResults = New-Object System.Collections.Generic.List[object]

$steps.Add((Run-Step -Label "Stop Docker Desktop processes" -Action {
    Get-Process "Docker Desktop" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Get-Process "com.docker.backend" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}))

$steps.Add((Run-Step -Label "Stop Docker service" -Action {
    Stop-Service -Name "com.docker.service" -Force -ErrorAction SilentlyContinue
}))

$steps.Add((Run-Step -Label "Terminate docker-desktop distro" -Action {
    if (Test-WslDistroExists -Name "docker-desktop") {
        wsl --terminate docker-desktop 2>$null
    }
    else {
        Write-Host "  [SKIP] docker-desktop distro not present" -ForegroundColor DarkYellow
    }
}))

$steps.Add((Run-Step -Label "Terminate docker-desktop-data distro (if present)" -Action {
    if (Test-WslDistroExists -Name "docker-desktop-data") {
        wsl --terminate docker-desktop-data 2>$null
    }
    else {
        Write-Host "  [SKIP] docker-desktop-data distro not present" -ForegroundColor DarkYellow
    }
}))

$steps.Add((Run-Step -Label "Shutdown WSL utility VM" -Action {
    wsl --shutdown
    Start-Sleep -Seconds 3
} -CheckExitCode))

if ($Target -in @("docker", "both")) {
    Write-Host ""
    Write-Host "[STEP] Compact Docker VHDX" -ForegroundColor Yellow
    if ($DryRun) {
        Write-Host "  [DRY-RUN] diskpart compact skipped" -ForegroundColor DarkYellow
        $compactResults.Add(@{
            path = $dockerVhdx
            success = $true
            dry_run = $true
            output = "dry-run"
        })
    }
    elseif (Test-Path $dockerVhdx) {
        $result = Invoke-DiskpartCompact -VhdxPath $dockerVhdx
        $compactResults.Add($result)
    }
    else {
        $compactResults.Add(@{
            path = $dockerVhdx
            success = $false
            exit_code = 404
            output = "docker_data.vhdx not found"
        })
    }
}

if ($Target -in @("ubuntu", "both")) {
    Write-Host ""
    Write-Host "[STEP] Compact Ubuntu VHDX" -ForegroundColor Yellow
    if ($DryRun) {
        Write-Host "  [DRY-RUN] diskpart compact skipped" -ForegroundColor DarkYellow
        $compactResults.Add(@{
            path = $ubuntuVhdx
            success = $true
            dry_run = $true
            output = "dry-run"
        })
    }
    elseif (Test-Path $ubuntuVhdx) {
        $result = Invoke-DiskpartCompact -VhdxPath $ubuntuVhdx
        $compactResults.Add($result)
    }
    else {
        $compactResults.Add(@{
            path = $ubuntuVhdx
            success = $false
            exit_code = 404
            output = "ubuntu ext4.vhdx not found"
        })
    }
}

$steps.Add((Run-Step -Label "Start Docker service" -Action {
    Start-Service -Name "com.docker.service" -ErrorAction SilentlyContinue
}))

if ($RestartDockerDesktop) {
    $steps.Add((Run-Step -Label "Launch Docker Desktop UI" -Action {
        if (Test-Path $dockerDesktopExe) {
            Start-Process $dockerDesktopExe
        }
    }))
}

$snapshotAfter = @{
    timestamp = (Get-Date).ToString("o")
    virtual_memory = (Get-VirtualMemorySnapshot)
    vhdx_gb = @{
        docker_data = Get-SizeGB $dockerVhdx
        ubuntu_ext4 = Get-SizeGB $ubuntuVhdx
    }
    services = @{
        docker = (Get-ServiceState -Name "com.docker.service")
        vmcompute = (Get-ServiceState -Name "vmcompute")
        wsl = (Get-ServiceState -Name "WSLService")
    }
    wsl_state = Get-WslStateRaw
}

$report.steps = $steps
$report.compact_results = $compactResults
$report.after = $snapshotAfter
$report.reclaimed_gb = @{
    docker_data = [math]::Round(($snapshotBefore.vhdx_gb.docker_data - $snapshotAfter.vhdx_gb.docker_data), 2)
    ubuntu_ext4 = [math]::Round(($snapshotBefore.vhdx_gb.ubuntu_ext4 - $snapshotAfter.vhdx_gb.ubuntu_ext4), 2)
}

$failedCompacts = @($compactResults | Where-Object { -not $_.success }).Count
$failedSteps = @($steps | Where-Object { -not $_.success }).Count
$report.summary = @{
    failed_steps = $failedSteps
    failed_compacts = $failedCompacts
    success = (($failedCompacts -eq 0) -and ($failedSteps -eq 0))
}

$path = Join-Path $logDir ("vhdx_compaction_governance_{0}.json" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
$report | ConvertTo-Json -Depth 10 | Set-Content -Path $path -Encoding UTF8

Write-Host ""
Write-Host ("Docker VHDX delta: {0} GB" -f $report.reclaimed_gb.docker_data) -ForegroundColor Cyan
Write-Host ("Ubuntu VHDX delta: {0} GB" -f $report.reclaimed_gb.ubuntu_ext4) -ForegroundColor Cyan
Write-Host "Report written to: $path" -ForegroundColor Green

if (($failedCompacts -gt 0) -or ($failedSteps -gt 0)) {
    Write-Host "One or more compaction steps failed. Review report output." -ForegroundColor Red
    Write-Host ("failed_steps={0}, failed_compacts={1}" -f $failedSteps, $failedCompacts) -ForegroundColor Red
    exit 1
}

exit 0
