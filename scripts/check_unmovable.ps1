# check_unmovable.ps1 — Check status of unmovable files blocking C: shrink

Write-Host "=== UNMOVABLE FILE CHECK ===" -ForegroundColor Cyan

# Hibernation
Write-Host ""
Write-Host "[1] HIBERNATION:" -ForegroundColor Yellow
$hiberPath = "C:\hiberfil.sys"
if (Test-Path $hiberPath) {
    $hSize = [math]::Round((Get-Item $hiberPath -Force).Length / 1GB, 1)
    Write-Host "  hiberfil.sys: EXISTS ($hSize GB) - BLOCKING SHRINK" -ForegroundColor Red
    Write-Host "  Fix: powercfg /h off" -ForegroundColor Gray
} else {
    Write-Host "  hiberfil.sys: REMOVED" -ForegroundColor Green
}

# System Restore
Write-Host ""
Write-Host "[2] SYSTEM RESTORE:" -ForegroundColor Yellow
try {
    $rp = Get-ComputerRestorePoint -ErrorAction Stop
    $rpCount = $rp.Count
    Write-Host "  Restore points: $rpCount found - BLOCKING SHRINK" -ForegroundColor Red
} catch {
    Write-Host "  System Restore: DISABLED (no restore points)" -ForegroundColor Green
}

# Page File
Write-Host ""
Write-Host "[3] PAGE FILE:" -ForegroundColor Yellow
$pf = Get-CimInstance Win32_PageFileUsage -ErrorAction SilentlyContinue
if ($pf) {
    $pfSizeGB = [math]::Round($pf.AllocatedBaseSize / 1024, 1)
    $pfName = $pf.Name
    Write-Host "  Page file: $pfName ($pfSizeGB GB)" -ForegroundColor White
    if ($pfSizeGB -gt 4) {
        Write-Host "  LARGE - blocks shrink. Reduce to 2-4 GB" -ForegroundColor Red
    } else {
        Write-Host "  Size OK" -ForegroundColor Green
    }
} else {
    Write-Host "  No page file found" -ForegroundColor Green
}

# Swap file
Write-Host ""
Write-Host "[4] SWAP FILE:" -ForegroundColor Yellow
$swapPath = "C:\swapfile.sys"
if (Test-Path $swapPath) {
    $swapMB = [math]::Round((Get-Item $swapPath -Force).Length / 1MB, 0)
    Write-Host "  swapfile.sys: EXISTS ($swapMB MB)" -ForegroundColor Yellow
} else {
    Write-Host "  swapfile.sys: NOT FOUND" -ForegroundColor Green
}

# Summary
Write-Host ""
Write-Host "[5] SHRINK ANALYSIS:" -ForegroundColor Yellow
$pss = Get-PartitionSupportedSize -DriveLetter C
$shrinkableGB = [math]::Round(($pss.SizeMax - $pss.SizeMin) / 1GB, 1)
$freeGB = [math]::Round((Get-Volume -DriveLetter C).SizeRemaining / 1GB, 1)
$blockedGB = [math]::Round($freeGB - $shrinkableGB, 1)
Write-Host "  Free space:     $freeGB GB" -ForegroundColor White
Write-Host "  Shrinkable:     $shrinkableGB GB" -ForegroundColor White
Write-Host "  Blocked by unmovable: $blockedGB GB" -ForegroundColor Red
Write-Host ""
if ($blockedGB -gt 50) {
    Write-Host "  ACTION NEEDED: $blockedGB GB blocked by unmovable files." -ForegroundColor Red
    Write-Host "  Run these as admin:" -ForegroundColor Yellow
    Write-Host "    powercfg /h off" -ForegroundColor Gray
    Write-Host "    Disable-ComputerRestore -Drive 'C:\'" -ForegroundColor Gray
    Write-Host "    vssadmin delete shadows /for=C: /all /quiet" -ForegroundColor Gray
    Write-Host "    Then: Optimize-Volume -DriveLetter C -Defrag" -ForegroundColor Gray
    Write-Host "    Then: Restart and re-check" -ForegroundColor Gray
} else {
    Write-Host "  Unmovable files under control." -ForegroundColor Green
}
