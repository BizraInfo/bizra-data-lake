# fix_docker_wsl.ps1 — Run as Administrator
# Fixes: Wsl/Service/RegisterDistro/0x80072746
# Safe: does NOT delete any data or VHDs
#
# NOTE: Canary Build 28020 renamed LxssManager → WSLService.
# This script handles BOTH names for forward/backward compatibility.

Write-Host "=== BIZRA Docker/WSL Fix ===" -ForegroundColor Cyan

# Detect WSL service name (LxssManager on stable, WSLService on Canary 28020+)
$WslSvcName = if (Get-Service WSLService -ErrorAction SilentlyContinue) { "WSLService" }
              elseif (Get-Service LxssManager -ErrorAction SilentlyContinue) { "LxssManager" }
              else { $null }
Write-Host "WSL service detected: $(if ($WslSvcName) { $WslSvcName } else { 'NONE' })" -ForegroundColor DarkGray

# Step 1: Kill Docker Desktop gracefully
Write-Host "`n[1/5] Stopping Docker Desktop..." -ForegroundColor Yellow
Get-Process "Docker Desktop" -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process "com.docker.*" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep 2

# Step 2: Terminate all WSL instances
Write-Host "[2/5] Shutting down WSL..." -ForegroundColor Yellow
wsl --shutdown
Start-Sleep 3

# Step 3: Restart Host Compute Service
Write-Host "[3/5] Restarting Hyper-V Host Compute Service..." -ForegroundColor Yellow
Restart-Service vmcompute -Force
Start-Sleep 2

# Step 4: Restart WSL service
Write-Host "[4/5] Restarting WSL service ($WslSvcName)..." -ForegroundColor Yellow
if ($WslSvcName) {
    Restart-Service $WslSvcName -Force -ErrorAction SilentlyContinue
} else {
    Write-Host "  WARNING: No WSL service found" -ForegroundColor Red
}
Start-Sleep 2

# Step 5: Verify WSL is clean
Write-Host "[5/5] Verifying WSL state..." -ForegroundColor Yellow
$distros = wsl --list --verbose 2>&1
Write-Host $distros

# Report
Write-Host "`n=== Fix complete ===" -ForegroundColor Green
Write-Host "Now start Docker Desktop normally." -ForegroundColor Green
Write-Host "If the error persists, run: wsl --unregister docker-desktop" -ForegroundColor DarkYellow
Write-Host "  then restart Docker Desktop (it will re-create the distro automatically)." -ForegroundColor DarkYellow

Read-Host "`nPress Enter to exit"
