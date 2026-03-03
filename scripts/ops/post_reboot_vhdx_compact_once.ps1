# BIZRA Post-Reboot VHDX Compaction Runner (single-use)
# ---------------------------------------------------------------------------
# This script is intended to be invoked by a scheduled task at startup.
# It executes VHDX compaction once, writes evidence, and then unregisters itself.

param(
    [string]$TaskName = "BIZRA-PostReboot-VHDX-Compact",
    [string]$Target = "docker"
)

$ErrorActionPreference = "Stop"

$logDir = "C:\BIZRA-DATA-LAKE\logs"
if (-not (Test-Path $logDir)) {
    New-Item -Path $logDir -ItemType Directory -Force | Out-Null
}

$runnerReport = @{
    timestamp = (Get-Date).ToString("o")
    task_name = $TaskName
    target = $Target
    action = "run_vhdx_compaction_once"
    compaction_exit_code = $null
    self_unregistered = $false
    error = $null
}

try {
    $compactScript = "C:\BIZRA-DATA-LAKE\scripts\ops\vhdx_compaction_governance.ps1"
    & $compactScript -Mode Compact -DryRun:$false -Target $Target -NoPrompt -SkipElevation
    $runnerReport.compaction_exit_code = if ($null -ne $LASTEXITCODE) { [int]$LASTEXITCODE } elseif ($?) { 0 } else { 1 }
}
catch {
    $runnerReport.error = $_.Exception.Message
    $runnerReport.compaction_exit_code = 1
}
finally {
    try {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction Stop
        $runnerReport.self_unregistered = $true
    }
    catch {
        $runnerReport.self_unregistered = $false
    }

    $path = Join-Path $logDir ("post_reboot_vhdx_compact_once_{0}.json" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
    $runnerReport | ConvertTo-Json -Depth 6 | Set-Content -Path $path -Encoding UTF8
}

exit ([int]$runnerReport.compaction_exit_code)
