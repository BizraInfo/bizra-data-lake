# step3_expand_b.ps1 — Run as Administrator
# After cleanup + page file reduction + restart:
# Delete B: → Defrag C: → Shrink C: → Recreate B: larger

Write-Host "=== STEP 3: Expand B: Drive ===" -ForegroundColor Cyan
Write-Host ""

# Current state
$cvol = Get-Volume -DriveLetter C
$cfreeGB = [math]::Round($cvol.SizeRemaining / 1GB, 1)
$ctotalGB = [math]::Round($cvol.Size / 1GB, 1)

$bvol = Get-Volume -DriveLetter B -ErrorAction SilentlyContinue
if ($bvol) {
    $btotalGB = [math]::Round($bvol.Size / 1GB, 1)
    $bfreeGB = [math]::Round($bvol.SizeRemaining / 1GB, 1)
    $busedGB = [math]::Round($btotalGB - $bfreeGB, 1)
    Write-Host "Current: C=$ctotalGB GB (free=$cfreeGB GB) | B=$btotalGB GB (used=$busedGB GB)" -ForegroundColor White
} else {
    Write-Host "Current: C=$ctotalGB GB (free=$cfreeGB GB) | B=NOT FOUND" -ForegroundColor White
}

# Shrinkable check
Write-Host ""
Write-Host "Checking shrinkable space..." -ForegroundColor Yellow
$pss = Get-PartitionSupportedSize -DriveLetter C
$shrinkableGB = [math]::Round(($pss.SizeMax - $pss.SizeMin) / 1GB, 1)
Write-Host "  C: shrinkable: $shrinkableGB GB" -ForegroundColor Cyan

if ($shrinkableGB -lt 200) {
    Write-Host ""
    Write-Host "  WARNING: Only $shrinkableGB GB shrinkable." -ForegroundColor Red
    Write-Host "  Try running defrag first:" -ForegroundColor Yellow
    Write-Host "    Optimize-Volume -DriveLetter C -Defrag -Verbose" -ForegroundColor Gray
    Write-Host "  Then re-run this script." -ForegroundColor Yellow
    Write-Host ""
    $defrag = Read-Host "Run defrag now? (yes/no)"
    if ($defrag -eq "yes") {
        Write-Host "  Running defrag (may take 30+ minutes)..." -ForegroundColor Yellow
        Optimize-Volume -DriveLetter C -Defrag -Verbose
        Write-Host "  Defrag complete. Re-checking..." -ForegroundColor Green
        $pss = Get-PartitionSupportedSize -DriveLetter C
        $shrinkableGB = [math]::Round(($pss.SizeMax - $pss.SizeMin) / 1GB, 1)
        Write-Host "  C: shrinkable now: $shrinkableGB GB" -ForegroundColor Cyan
    }
}

# Calculate target
$targetBGB = 500
if ($bvol) {
    $targetShrinkGB = $targetBGB - $btotalGB   # Only need to shrink by the difference
} else {
    $targetShrinkGB = $targetBGB
}

$actualShrinkGB = [math]::Min($shrinkableGB, $targetShrinkGB)
$newBGB = if ($bvol) { $btotalGB + $actualShrinkGB } else { $actualShrinkGB }

Write-Host ""
Write-Host "=== PLAN ===" -ForegroundColor Cyan
Write-Host "  Target B: size:   $targetBGB GB" -ForegroundColor White
Write-Host "  Shrinkable from C: $shrinkableGB GB" -ForegroundColor White
Write-Host "  Will shrink C: by: $actualShrinkGB GB" -ForegroundColor White
Write-Host "  New B: will be:   ~$newBGB GB" -ForegroundColor Green
Write-Host ""
Write-Host "  This will:" -ForegroundColor Yellow
Write-Host "    1. Delete current B: partition ($btotalGB GB)" -ForegroundColor White
Write-Host "    2. Shrink C: by $actualShrinkGB GB" -ForegroundColor White
Write-Host "    3. Create new B: from all unallocated space" -ForegroundColor White
Write-Host ""
Write-Host "  WARNING: If B: has data, back it up first!" -ForegroundColor Red

if ($busedGB -gt 1) {
    Write-Host "  B: has $busedGB GB of data. Back up before proceeding!" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== MANUAL STEPS (Disk Management) ===" -ForegroundColor Cyan
Write-Host "  Since diskpart shrink has the same limitations, use Disk Management:" -ForegroundColor Gray
Write-Host ""
Write-Host "  1. Open: diskmgmt.msc" -ForegroundColor White
Write-Host "  2. Right-click B: partition -> Delete Volume" -ForegroundColor White
Write-Host "  3. Right-click C: partition -> Shrink Volume" -ForegroundColor White
Write-Host "     Enter amount to shrink: $actualShrinkGB GB ($([math]::Round($actualShrinkGB * 1024, 0)) MB)" -ForegroundColor White
Write-Host "  4. Right-click on ALL unallocated space -> New Simple Volume" -ForegroundColor White
Write-Host "  5. Assign letter B:, NTFS format, label 'BIZRA'" -ForegroundColor White
Write-Host "  6. UNCHECK 'Enable file and folder compression'" -ForegroundColor Red
Write-Host ""
Write-Host "  After creating B:, run in WSL:" -ForegroundColor Yellow
Write-Host "    sudo mount -t drvfs B: /mnt/b" -ForegroundColor Gray
Write-Host "    mkdir -p /mnt/b/BIZRA/{00_CONSTITUTION,01_CORE,02_DATA_PIPELINE,03_ASSETS,04_ARCHIVE,05_IMPORTS,06_INDEX}" -ForegroundColor Gray
Write-Host ""

$open = Read-Host "Open Disk Management now? (yes/no)"
if ($open -eq "yes") {
    Start-Process diskmgmt.msc
}

Read-Host "Press Enter to close"
