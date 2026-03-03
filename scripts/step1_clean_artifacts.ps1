# step1_clean_artifacts.ps1 — Run as Administrator
# Deletes known large build artifacts to free space on C:
# All targets are regenerable (cargo build / pip install / npm install)

Write-Host "=== STEP 1: Clean Build Artifacts ===" -ForegroundColor Cyan
Write-Host "Scanning known artifact directories..." -ForegroundColor White
Write-Host ""

$targets = @(
    "C:\BIZRA-DATA-LAKE\bizra-omega\target",
    "C:\BIZRA-PROJECTS\00-GENESIS\genesis-node\target",
    "C:\BIZRA-Dual-Agentic-system--main\target",
    "C:\bizra-voice\moshi\.venv",
    "C:\BIZRA-Dual-Agentic-system--main\.venv",
    "C:\BIZRA-DATA-LAKE\.venv-wsl",
    "C:\BIZRA-NODE0\rust\target",
    "C:\BIZRA-DATA-LAKE\.claude\worktrees\zealous-goldberg\bizra-omega\target",
    "C:\BIZRA-DATA-LAKE\personaplex\.venv-wsl",
    "C:\BIZRA-DATA-LAKE\personaplex\.venv",
    "C:\BIZRA-PROJECTS\01-INFRASTRUCTURE\bizra-os\src\ai\src\ai\.venv_cuda",
    "C:\BIZRA-TaskMaster\opencode-cli-review\node_modules",
    "C:\BIZRA-NODE0\target",
    "C:\BIZRA-PROJECTS\bizra-synthesis-orchestrator\target",
    "C:\award-winner-design\bizra-genesis-node\backend\target",
    "C:\BIZRA-DATA-LAKE\BIZRA-DATA-LAKE\xenodochial-shaw\bizra-omega\target",
    "C:\BIZRA-PROJECTS\01-INFRASTRUCTURE\bizra-os\src-tauri\target",
    "C:\BIZRA-PROJECTS\00-GENESIS\synthesis-orchestrator\bizra-moe\target",
    "C:\bizra-genesis-node-repaired\bizra-moe\target",
    "C:\BIZRA-PROJECTS\00-GENESIS\genesis-node\bizra-moe\target",
    "C:\BIZRA-DATA-LAKE\.venv",
    "C:\BIZRA-DATA-LAKE\.mypy_cache",
    "C:\award-winner-design\node_modules",
    "C:\BIZRA-NODE0\node_modules"
)

$totalSize = 0
$found = @()

foreach ($path in $targets) {
    if (Test-Path $path) {
        $size = (Get-ChildItem $path -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
        $sizeGB = [math]::Round($size / 1GB, 1)
        if ($sizeGB -ge 0.01) {
            Write-Host ("  FOUND: {0,-65} {1,8} GB" -f $path, $sizeGB) -ForegroundColor Yellow
            $totalSize += $size
            $found += [PSCustomObject]@{ Path=$path; Size=$size; SizeGB=$sizeGB }
        }
    }
}

$totalGB = [math]::Round($totalSize / 1GB, 1)
Write-Host ""
Write-Host "Total recoverable: $totalGB GB across $($found.Count) directories" -ForegroundColor Cyan
Write-Host ""

# Get current disk state
$vol = Get-Volume -DriveLetter C
$beforeFree = [math]::Round($vol.SizeRemaining / 1GB, 1)

Write-Host "C: free BEFORE cleanup: $beforeFree GB" -ForegroundColor White
Write-Host ""
Write-Host "WARNING: This deletes $totalGB GB of build artifacts." -ForegroundColor Red
Write-Host "All can be regenerated with: cargo build / pip install / npm install" -ForegroundColor Gray
Write-Host ""
$confirm = Read-Host "Type YES to delete all artifacts"

if ($confirm -ne "YES") {
    Write-Host "Aborted." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "Deleting..." -ForegroundColor Yellow
$deletedSize = 0
$failCount = 0

foreach ($item in ($found | Sort-Object SizeGB -Descending)) {
    Write-Host ("  Removing: {0} ({1} GB)..." -f $item.Path, $item.SizeGB) -ForegroundColor White -NoNewline
    try {
        Remove-Item -Path $item.Path -Recurse -Force -ErrorAction Stop
        Write-Host " DONE" -ForegroundColor Green
        $deletedSize += $item.Size
    } catch {
        Write-Host " FAILED" -ForegroundColor Red
        $failCount++
    }
}

$deletedGB = [math]::Round($deletedSize / 1GB, 1)
Write-Host ""

# Get new disk state
$vol = Get-Volume -DriveLetter C
$afterFree = [math]::Round($vol.SizeRemaining / 1GB, 1)
$actualFreed = [math]::Round($afterFree - $beforeFree, 1)

Write-Host "=== CLEANUP COMPLETE ===" -ForegroundColor Green
Write-Host "  Deleted: $deletedGB GB" -ForegroundColor Cyan
Write-Host "  Failed: $failCount directories" -ForegroundColor $(if ($failCount -gt 0) { "Yellow" } else { "Green" })
Write-Host "  C: free BEFORE: $beforeFree GB" -ForegroundColor White
Write-Host "  C: free AFTER:  $afterFree GB" -ForegroundColor Cyan
Write-Host "  Actual freed:   $actualFreed GB" -ForegroundColor Green
Write-Host ""
Write-Host "Next: Run step2_reduce_pagefile.ps1 then restart" -ForegroundColor Yellow
Write-Host ""
Read-Host "Press Enter to close"
