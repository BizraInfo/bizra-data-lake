# fix_docker_wsl_force.ps1 — Run as Administrator
# FORCE fix: bypasses hung wsl --shutdown by killing processes directly
#
# NOTE: Canary Build 28020 renamed LxssManager → WSLService.
# This script handles BOTH names for forward/backward compatibility.

Write-Host "=== BIZRA Docker/WSL FORCE Fix ===" -ForegroundColor Cyan

# Detect WSL service name (LxssManager on stable, WSLService on Canary 28020+)
$WslSvcName = if (Get-Service WSLService -ErrorAction SilentlyContinue) { "WSLService" }
              elseif (Get-Service LxssManager -ErrorAction SilentlyContinue) { "LxssManager" }
              else { $null }
Write-Host "WSL service detected: $(if ($WslSvcName) { $WslSvcName } else { 'NONE' })" -ForegroundColor DarkGray

# Step 1: Kill everything Docker
Write-Host "`n[1/7] Killing Docker processes..." -ForegroundColor Yellow
Get-Process "Docker Desktop" -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process "com.docker*" -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process "docker" -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process "docker-sandbox" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep 1

# Step 2: Kill WSL processes directly (skip wsl --shutdown which hangs)
Write-Host "[2/7] Force-killing WSL processes..." -ForegroundColor Yellow
Get-Process "wsl" -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process "wslhost" -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process "wslservice" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep 2

# Step 3: Stop services
Write-Host "[3/7] Stopping HCS + WSL services..." -ForegroundColor Yellow
if ($WslSvcName) { Stop-Service $WslSvcName -Force -ErrorAction SilentlyContinue }
Stop-Service vmcompute -Force -ErrorAction SilentlyContinue
Start-Sleep 3

# Step 4: Verify everything is down
Write-Host "[4/7] Verifying clean state..." -ForegroundColor Yellow
$staleWsl = Get-Process "wsl" -ErrorAction SilentlyContinue
if ($staleWsl) {
    Write-Host "  WARNING: $($staleWsl.Count) wsl.exe still alive — force killing" -ForegroundColor Red
    $staleWsl | Stop-Process -Force
    Start-Sleep 1
}

# Step 5: Restart services
Write-Host "[5/7] Starting HCS..." -ForegroundColor Yellow
Start-Service vmcompute
Start-Sleep 2

Write-Host "[6/7] Starting WSL service ($WslSvcName)..." -ForegroundColor Yellow
if ($WslSvcName) {
    Start-Service $WslSvcName
} else {
    Write-Host "  ERROR: No WSL service found — WSL may need reinstalling" -ForegroundColor Red
}
Start-Sleep 3

# Step 7: Verify
Write-Host "[7/7] Checking WSL state..." -ForegroundColor Yellow
$result = wsl --list --verbose 2>&1
Write-Host $result

Write-Host "`n=== Force fix complete ===" -ForegroundColor Green
Write-Host "Now start Docker Desktop." -ForegroundColor Green

Read-Host "`nPress Enter to exit"
