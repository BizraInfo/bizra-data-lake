# BIZRA Pagefile Governance (Windows-side)
# ---------------------------------------------------------------------------
# Goal:
#   1) Snapshot virtual memory and pagefile pressure with evidence.
#   2) Safely apply deterministic pagefile sizing for heavy Node0 workloads.
#   3) Emit JSON reports for analyze/apply runs.
#
# Usage:
#   .\pagefile_governance.ps1 -Mode Analyze
#   .\pagefile_governance.ps1 -Mode Apply -DryRun
#   .\pagefile_governance.ps1 -Mode Apply -DryRun:$false -NoPrompt

param(
    [ValidateSet("Analyze", "Apply")]
    [string]$Mode = "Analyze",
    [object]$DryRun = $true,
    [int]$TargetInitialMB = 16384,
    [int]$TargetMaximumMB = 32768,
    [switch]$NoPrompt,
    [switch]$SkipElevation,
    [switch]$RequireAdminForApply = $true
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

function Run-Step {
    param(
        [string]$Label,
        [scriptblock]$Action
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
        $output = & $Action 2>&1 | Out-String
        return @{
            label = $Label
            dry_run = $false
            success = $true
            output = $output
        }
    }
    catch {
        return @{
            label = $Label
            dry_run = $false
            success = $false
            output = $_.Exception.Message
        }
    }
}

function Get-PagefileSnapshot {
    $os = Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue
    $cs = Get-CimInstance Win32_ComputerSystem -ErrorAction SilentlyContinue
    $settings = @(Get-CimInstance Win32_PageFileSetting -ErrorAction SilentlyContinue)
    $usage = @(Get-CimInstance Win32_PageFileUsage -ErrorAction SilentlyContinue)

    return @{
        timestamp = (Get-Date).ToString("o")
        is_admin = (Test-IsAdministrator)
        automatic_managed_pagefile = if ($cs) { [bool]$cs.AutomaticManagedPagefile } else { $null }
        virtual_memory = @{
            total_virtual_gb = if ($os) { [math]::Round(([double]$os.TotalVirtualMemorySize / 1MB), 2) } else { 0.0 }
            free_virtual_gb = if ($os) { [math]::Round(([double]$os.FreeVirtualMemory / 1MB), 2) } else { 0.0 }
            total_physical_gb = if ($os) { [math]::Round(([double]$os.TotalVisibleMemorySize / 1MB), 2) } else { 0.0 }
            free_physical_gb = if ($os) { [math]::Round(([double]$os.FreePhysicalMemory / 1MB), 2) } else { 0.0 }
        }
        pagefile_settings = @(
            $settings | ForEach-Object {
                @{
                    name = "$($_.Name)"
                    initial_mb = [int]$_.InitialSize
                    maximum_mb = [int]$_.MaximumSize
                }
            }
        )
        pagefile_usage = @(
            $usage | ForEach-Object {
                @{
                    name = "$($_.Name)"
                    allocated_mb = [int]$_.AllocatedBaseSize
                    current_mb = [int]$_.CurrentUsage
                    peak_mb = [int]$_.PeakUsage
                }
            }
        )
    }
}

$DryRun = Resolve-Bool -Value $DryRun -Default $true

$logDir = "C:\BIZRA-DATA-LAKE\logs"
if (-not (Test-Path $logDir)) {
    New-Item -Path $logDir -ItemType Directory -Force | Out-Null
}

Write-Host ""
Write-Host "=== Pagefile Governance Snapshot ===" -ForegroundColor Cyan
$before = Get-PagefileSnapshot
Write-Host ("Admin: {0}" -f $before.is_admin)
Write-Host ("Automatic managed pagefile: {0}" -f $before.automatic_managed_pagefile)
Write-Host ("Virtual memory free/total: {0} / {1} GB" -f $before.virtual_memory.free_virtual_gb, $before.virtual_memory.total_virtual_gb)
if ($before.pagefile_settings.Count -gt 0) {
    foreach ($s in $before.pagefile_settings) {
        Write-Host ("Pagefile setting: {0} (Initial {1} MB, Max {2} MB)" -f $s.name, $s.initial_mb, $s.maximum_mb)
    }
}

$report = @{
    mode = $Mode
    dry_run = $DryRun
    target_mb = @{
        initial = $TargetInitialMB
        maximum = $TargetMaximumMB
    }
    before = $before
    steps = @()
    after = $null
    preflight_failure = $null
    reboot_required = $false
    changes_applied = @()
}

if ($Mode -eq "Analyze") {
    $path = Join-Path $logDir ("pagefile_governance_{0}.json" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
    $report | ConvertTo-Json -Depth 10 | Set-Content -Path $path -Encoding UTF8
    Write-Host ""
    Write-Host "Analyze mode complete. No changes applied." -ForegroundColor Cyan
    Write-Host "Report written to: $path" -ForegroundColor Green
    exit 0
}

if ((-not $DryRun) -and $RequireAdminForApply -and (-not $before.is_admin)) {
    $report.preflight_failure = @{
        code = "ADMIN_REQUIRED"
        message = "Apply mode requires Administrator privileges."
    }
    $path = Join-Path $logDir ("pagefile_governance_{0}.json" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
    $report | ConvertTo-Json -Depth 10 | Set-Content -Path $path -Encoding UTF8
    Write-Host ""
    Write-Host "Apply mode requires Administrator privileges." -ForegroundColor Red
    Write-Host "Run from elevated PowerShell or use an elevated launcher." -ForegroundColor Yellow
    Write-Host "Report written to: $path" -ForegroundColor Green
    exit 2
}

if ($TargetInitialMB -lt 1024 -or $TargetMaximumMB -lt $TargetInitialMB) {
    $report.preflight_failure = @{
        code = "INVALID_TARGET_SIZING"
        message = "Invalid target pagefile sizing."
        target_initial_mb = $TargetInitialMB
        target_maximum_mb = $TargetMaximumMB
    }
    $path = Join-Path $logDir ("pagefile_governance_{0}.json" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
    $report | ConvertTo-Json -Depth 10 | Set-Content -Path $path -Encoding UTF8
    Write-Host "Invalid target pagefile sizing. Ensure max >= initial and both >= 1024 MB." -ForegroundColor Red
    Write-Host "Report written to: $path" -ForegroundColor Green
    exit 3
}

Write-Host ""
Write-Host "=== Apply Plan ===" -ForegroundColor Cyan
Write-Host ("Target pagefile size: initial={0} MB, max={1} MB" -f $TargetInitialMB, $TargetMaximumMB)
Confirm-OrDie -PromptText "Proceed with pagefile governance apply"

$steps = New-Object System.Collections.Generic.List[object]
$changesApplied = New-Object System.Collections.Generic.List[object]

if ($before.automatic_managed_pagefile) {
    $steps.Add((Run-Step -Label "Disable automatic managed pagefile" -Action {
        $cs = Get-CimInstance Win32_ComputerSystem
        Set-CimInstance -InputObject $cs -Property @{ AutomaticManagedPagefile = $false } | Out-Null
    }))
    $changesApplied.Add(@{ change = "disable_auto_managed_pagefile" })
}
else {
    $steps.Add(@{
        label = "Disable automatic managed pagefile"
        dry_run = $DryRun
        success = $true
        output = "already_disabled"
    })
}

$pfName = "C:\pagefile.sys"
$existing = Get-CimInstance Win32_PageFileSetting -ErrorAction SilentlyContinue | Where-Object {
    "$($_.Name)".ToLowerInvariant() -eq $pfName.ToLowerInvariant()
} | Select-Object -First 1

if ($null -eq $existing) {
    $steps.Add((Run-Step -Label "Create pagefile setting" -Action {
        New-CimInstance -ClassName Win32_PageFileSetting -Property @{
            Name = $pfName
            InitialSize = $TargetInitialMB
            MaximumSize = $TargetMaximumMB
        } | Out-Null
    }))
    $changesApplied.Add(@{
        change = "create_pagefile_setting"
        name = $pfName
        initial_mb = $TargetInitialMB
        maximum_mb = $TargetMaximumMB
    })
}
else {
    $currentInitial = [int]$existing.InitialSize
    $currentMaximum = [int]$existing.MaximumSize
    if (($currentInitial -ne $TargetInitialMB) -or ($currentMaximum -ne $TargetMaximumMB)) {
        $steps.Add((Run-Step -Label "Update pagefile setting" -Action {
            Set-CimInstance -InputObject $existing -Property @{
                InitialSize = $TargetInitialMB
                MaximumSize = $TargetMaximumMB
            } | Out-Null
        }))
        $changesApplied.Add(@{
            change = "update_pagefile_setting"
            name = $pfName
            before_initial_mb = $currentInitial
            before_maximum_mb = $currentMaximum
            after_initial_mb = $TargetInitialMB
            after_maximum_mb = $TargetMaximumMB
        })
    }
    else {
        $steps.Add(@{
            label = "Update pagefile setting"
            dry_run = $DryRun
            success = $true
            output = "already_compliant"
        })
    }
}

$after = Get-PagefileSnapshot

$report.steps = $steps
$report.after = $after
$report.changes_applied = $changesApplied
$report.reboot_required = ((-not $DryRun) -and ($changesApplied.Count -gt 0))

$path = Join-Path $logDir ("pagefile_governance_{0}.json" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
$report | ConvertTo-Json -Depth 10 | Set-Content -Path $path -Encoding UTF8

Write-Host ""
Write-Host "Report written to: $path" -ForegroundColor Green
if ($report.reboot_required) {
    Write-Host "Reboot required to fully apply pagefile changes." -ForegroundColor Yellow
}

$failedSteps = @($steps | Where-Object { -not $_.success }).Count
if ($failedSteps -gt 0) {
    Write-Host "One or more steps failed. Review report output." -ForegroundColor Red
    exit 1
}

exit 0
