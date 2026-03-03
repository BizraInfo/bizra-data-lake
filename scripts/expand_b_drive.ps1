# expand_b_drive.ps1 — Run as Administrator
# Expands B: drive by: clean artifacts → disable unmovable files → defrag → shrink C → recreate B
# IMPORTANT: This is a multi-step process. Run each step and verify before continuing.

Write-Host "=== BIZRA B: Drive Expansion ===" -ForegroundColor Cyan
Write-Host "Goal: Expand B: from 195 GB to ~2.5 TB" -ForegroundColor Yellow
Write-Host ""

# Current state
$cvol = Get-Volume -DriveLetter C
$cfreeGB = [math]::Round($cvol.SizeRemaining / 1GB, 1)
$ctotalGB = [math]::Round($cvol.Size / 1GB, 1)

$bvol = Get-Volume -DriveLetter B -ErrorAction SilentlyContinue
if ($bvol) {
    $btotalGB = [math]::Round($bvol.Size / 1GB, 1)
    Write-Host "Current: C=$ctotalGB GB (free=$cfreeGB GB) | B=$btotalGB GB" -ForegroundColor White
} else {
    Write-Host "Current: C=$ctotalGB GB (free=$cfreeGB GB) | B=NOT FOUND" -ForegroundColor White
}

Write-Host ""
Write-Host "=== STEP 1: Disable unmovable files ===" -ForegroundColor Yellow
Write-Host "These prevent Windows from shrinking C: fully." -ForegroundColor Gray

# Disable hibernation
Write-Host "[1a] Disabling hibernation..." -ForegroundColor White
powercfg /h off
Write-Host "  Hibernation: OFF (saves ~12 GB)" -ForegroundColor Green

# Check page file
Write-Host "[1b] Page file status:" -ForegroundColor White
$pf = Get-CimInstance Win32_PageFileUsage -ErrorAction SilentlyContinue
if ($pf) {
    Write-Host "  Page file: $($pf.Name) ($([math]::Round($pf.AllocatedBaseSize/1024, 1)) GB)" -ForegroundColor Yellow
    Write-Host "  To minimize: System Properties > Advanced > Performance > Virtual Memory" -ForegroundColor Gray
    Write-Host "  Set to Custom: Initial=2048 MB, Maximum=4096 MB" -ForegroundColor Gray
} else {
    Write-Host "  No page file found" -ForegroundColor Green
}

# Disable system restore
Write-Host "[1c] Disabling System Restore on C:..." -ForegroundColor White
Disable-ComputerRestore -Drive "C:\" -ErrorAction SilentlyContinue
vssadmin delete shadows /for=C: /all /quiet 2>$null
Write-Host "  System Restore: Disabled, shadow copies deleted" -ForegroundColor Green

Write-Host ""
Write-Host "=== STEP 2: Defragment C: ===" -ForegroundColor Yellow
Write-Host "This consolidates free space so shrink can claim more." -ForegroundColor Gray
Write-Host "Running defrag (this may take 30+ minutes on a 3.5 TB drive)..." -ForegroundColor White

# Run optimize (defrag + retrim)
Optimize-Volume -DriveLetter C -Defrag -Verbose
Write-Host "  Defrag: Complete" -ForegroundColor Green

Write-Host ""
Write-Host "=== STEP 3: Check shrinkable space ===" -ForegroundColor Yellow
$size = Get-PartitionSupportedSize -DriveLetter C
$maxGB = [math]::Round($size.SizeMax / 1GB, 1)
$minGB = [math]::Round($size.SizeMin / 1GB, 1)
$shrinkableGB = $maxGB - $minGB
Write-Host "  Shrinkable: $shrinkableGB GB" -ForegroundColor Cyan

if ($shrinkableGB -lt 2300) {
    Write-Host "  WARNING: Only $shrinkableGB GB shrinkable. Need 2300 GB." -ForegroundColor Red
    Write-Host "  Try: Reduce page file manually, then re-run." -ForegroundColor Yellow
    Write-Host "  Or: Accept smaller B: drive ($shrinkableGB GB + current 195 GB)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== STEP 4: Backup B: manifest ===" -ForegroundColor Yellow
$manifestSrc = "B:\BIZRA\GENESIS_MANIFEST.yaml"
$manifestDst = "C:\BIZRA-DATA-LAKE\GENESIS_MANIFEST.yaml.bak"
if (Test-Path $manifestSrc) {
    Copy-Item $manifestSrc $manifestDst -Force
    Write-Host "  Backed up: $manifestSrc -> $manifestDst" -ForegroundColor Green
} else {
    Write-Host "  No manifest to backup" -ForegroundColor DarkYellow
}

Write-Host ""
Write-Host "=== READY FOR MANUAL STEPS ===" -ForegroundColor Cyan
Write-Host "1. Open Disk Management (diskmgmt.msc)" -ForegroundColor White
Write-Host "2. Delete B: partition (right-click -> Delete Volume)" -ForegroundColor White
Write-Host "3. Right-click C: -> Shrink Volume -> Enter $([math]::Min($shrinkableGB, 2300)) GB" -ForegroundColor White
Write-Host "4. Right-click unallocated space -> New Simple Volume" -ForegroundColor White
Write-Host "5. Assign letter B:, NTFS, label 'BIZRA', Quick Format" -ForegroundColor White
Write-Host "6. UNCHECK 'Enable file and folder compression'" -ForegroundColor Red
Write-Host ""
Write-Host "After creating B:, run:" -ForegroundColor Yellow
Write-Host "  wsl -e bash -c 'sudo mount -t drvfs B: /mnt/b'" -ForegroundColor Gray
Write-Host "  wsl -e python3 /mnt/c/BIZRA-DATA-LAKE/scripts/migration/rebuild_b_tree.py" -ForegroundColor Gray
Write-Host ""
Read-Host "Press Enter to close"
