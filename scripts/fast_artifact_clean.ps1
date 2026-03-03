# fast_artifact_clean.ps1 — Run as Administrator
# Directly deletes known large build artifacts to free disk space
# Much faster than the Python walker because it targets exact paths

param(
    [switch]$Execute,
    [switch]$Force
)

Write-Host "=== BIZRA Fast Artifact Cleanup ===" -ForegroundColor Cyan
Write-Host ""

# Known artifact directories to delete
$targets = @(
    @{ Path = "C:\BIZRA-DATA-LAKE\bizra-omega\target"; Desc = "Rust build cache (DATA-LAKE)" },
    @{ Path = "C:\BIZRA-DATA-LAKE\.venv"; Desc = "Windows Python venv (redundant)" },
    @{ Path = "C:\BIZRA-DATA-LAKE\.venv-wsl"; Desc = "WSL Python venv (redundant, keep .venv-linux)" },
    @{ Path = "C:\BIZRA-Dual-Agentic-system--main\bizra-omega\target"; Desc = "Rust build cache (Dual-Agentic)" },
    @{ Path = "C:\BIZRA-NODE0\bizra-omega\target"; Desc = "Rust build cache (NODE0)" },
    @{ Path = "C:\BIZRA-NODE0\target"; Desc = "Rust build cache (NODE0 root)" },
    @{ Path = "C:\BIZRA-PROJECTS\bizra-omega\target"; Desc = "Rust build cache (PROJECTS)" },
    @{ Path = "C:\BIZRA-PROJECTS\target"; Desc = "Rust build cache (PROJECTS root)" },
    @{ Path = "C:\bizra-genesis-node-repaired\target"; Desc = "Rust build cache (genesis repaired)" },
    @{ Path = "C:\bizra-genesis-node\target"; Desc = "Rust build cache (genesis)" },
    @{ Path = "C:\bizra-genesis-node-fresh\target"; Desc = "Rust build cache (genesis fresh)" },
    @{ Path = "C:\BIZRA-TaskMaster\node_modules"; Desc = "Node modules (TaskMaster)" },
    @{ Path = "C:\award-winner-design\node_modules"; Desc = "Node modules (award-winner)" },
    @{ Path = "C:\BIZRA-DATA-LAKE\node_modules"; Desc = "Node modules (DATA-LAKE)" },
    @{ Path = "C:\BIZRA-NODE0\node_modules"; Desc = "Node modules (NODE0)" },
    @{ Path = "C:\BIZRA-PROJECTS\node_modules"; Desc = "Node modules (PROJECTS)" },
    @{ Path = "C:\Advanced Windows Desktop App with Multi-Agent System Features\node_modules"; Desc = "Node modules (Desktop App)" },
    @{ Path = "C:\HERMES project\node_modules"; Desc = "Node modules (HERMES)" },
    @{ Path = "C:\BIZRA-DATA-LAKE\.mypy_cache"; Desc = "MyPy cache" },
    @{ Path = "C:\BIZRA-DATA-LAKE\bizra-omega\target"; Desc = "Rust build cache" }
)

$totalSize = 0
$found = @()

foreach ($t in $targets) {
    if (Test-Path $t.Path) {
        $size = (Get-ChildItem $t.Path -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
        $sizeGB = [math]::Round($size / 1GB, 1)
        if ($sizeGB -ge 0.01) {
            Write-Host ("{0,-70} {1,8} GB  {2}" -f $t.Path, $sizeGB, $t.Desc) -ForegroundColor Yellow
            $totalSize += $size
            $found += @{ Path = $t.Path; Size = $size; SizeGB = $sizeGB; Desc = $t.Desc }
        }
    }
}

Write-Host ""
$totalGB = [math]::Round($totalSize / 1GB, 1)
Write-Host "Total recoverable: $totalGB GB across $($found.Count) directories" -ForegroundColor Cyan

if (-not $Execute) {
    Write-Host ""
    Write-Host "DRY RUN — no files deleted." -ForegroundColor DarkYellow
    Write-Host "To execute: .\fast_artifact_clean.ps1 -Execute" -ForegroundColor White
    Write-Host ""
    Read-Host "Press Enter to close"
    exit 0
}

if (-not $Force) {
    Write-Host ""
    Write-Host "WARNING: This will permanently delete $totalGB GB of build artifacts." -ForegroundColor Red
    Write-Host "These can all be regenerated with 'cargo build' or 'npm install'." -ForegroundColor Gray
    $confirm = Read-Host "Type YES to confirm deletion"
    if ($confirm -ne "YES") {
        Write-Host "Aborted." -ForegroundColor Yellow
        exit 0
    }
}

Write-Host ""
Write-Host "Deleting..." -ForegroundColor Yellow
$deleted = 0
$failed = 0

foreach ($f in $found) {
    Write-Host "  Removing: $($f.Path) ($($f.SizeGB) GB)..." -ForegroundColor White -NoNewline
    try {
        Remove-Item -Path $f.Path -Recurse -Force -ErrorAction Stop
        Write-Host " DONE" -ForegroundColor Green
        $deleted += $f.Size
    } catch {
        Write-Host " FAILED: $($_.Exception.Message)" -ForegroundColor Red
        $failed++
    }
}

$deletedGB = [math]::Round($deleted / 1GB, 1)
Write-Host ""
Write-Host "=== CLEANUP COMPLETE ===" -ForegroundColor Green
Write-Host "Freed: $deletedGB GB" -ForegroundColor Cyan
if ($failed -gt 0) {
    Write-Host "Failed: $failed directories (may be in use)" -ForegroundColor Yellow
}

# Show new disk state
$vol = Get-Volume -DriveLetter C
$freeGB = [math]::Round($vol.SizeRemaining / 1GB, 1)
Write-Host "C: now has $freeGB GB free" -ForegroundColor Cyan

Write-Host ""
Read-Host "Press Enter to close"
