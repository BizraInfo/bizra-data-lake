# step2_reduce_pagefile.ps1 — Run as Administrator
# Reduces page file from 60 GB to 2-4 GB to free unmovable space

Write-Host "=== STEP 2: Reduce Page File ===" -ForegroundColor Cyan

# Show current
$pf = Get-CimInstance Win32_PageFileUsage -ErrorAction SilentlyContinue
if ($pf) {
    $currentGB = [math]::Round($pf.AllocatedBaseSize / 1024, 1)
    Write-Host "  Current page file: $($pf.Name) ($currentGB GB)" -ForegroundColor Yellow
} else {
    Write-Host "  No page file found." -ForegroundColor Green
    Write-Host "  Nothing to do." -ForegroundColor Green
    Read-Host "Press Enter to close"
    exit 0
}

# Disable automatic management
Write-Host "  Disabling automatic page file management..." -ForegroundColor White
$cs = Get-CimInstance Win32_ComputerSystem
$cs | Set-CimInstance -Property @{AutomaticManagedPagefile = $false}
Write-Host "  AutomaticManagedPagefile: OFF" -ForegroundColor Green

# Remove current settings
Write-Host "  Removing current page file settings..." -ForegroundColor White
$pfs = Get-CimInstance Win32_PageFileSetting -ErrorAction SilentlyContinue
if ($pfs) {
    $pfs | Remove-CimInstance
    Write-Host "  Old settings removed." -ForegroundColor Green
}

# Set new fixed size (2-4 GB)
Write-Host "  Setting new page file: 2048 MB initial, 4096 MB max..." -ForegroundColor White
New-CimInstance -ClassName Win32_PageFileSetting -Property @{
    Name = "C:\pagefile.sys"
    InitialSize = 2048
    MaximumSize = 4096
} | Out-Null
Write-Host "  Page file configured: 2-4 GB" -ForegroundColor Green

Write-Host ""
Write-Host "=== DONE ===" -ForegroundColor Green
Write-Host "  Before: $currentGB GB page file (unmovable, blocking shrink)" -ForegroundColor White
Write-Host "  After:  2-4 GB page file (after restart)" -ForegroundColor Cyan
Write-Host "  Freed:  ~$([math]::Round($currentGB - 4, 0)) GB of unmovable space" -ForegroundColor Green
Write-Host ""
Write-Host "  >>> RESTART REQUIRED for changes to take effect <<<" -ForegroundColor Red
Write-Host ""
Write-Host "After restart, run: C:\BIZRA-DATA-LAKE\scripts\check_unmovable.ps1" -ForegroundColor Yellow
Write-Host "Then: C:\BIZRA-DATA-LAKE\scripts\step3_expand_b.ps1" -ForegroundColor Yellow
Write-Host ""
Read-Host "Press Enter to close"
