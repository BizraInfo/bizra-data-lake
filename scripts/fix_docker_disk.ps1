# BIZRA DevOps: Docker Data Disk EXT4 Repair Script
# Run from Windows PowerShell (Admin)
#
# Problem: /dev/sde has 366,894 EXT4 errors (inode checksum invalid)
# Cause: Unclean WSL shutdowns corrupting Docker Desktop's data VHDX
# Solution: Unmount cleanly, run e2fsck, restart

param(
    [switch]$Force,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

Write-Host "=== BIZRA Docker Data Disk Repair ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Verify Docker Desktop is running
$dockerProc = Get-Process "Docker Desktop" -ErrorAction SilentlyContinue
if (-not $dockerProc) {
    Write-Host "[WARN] Docker Desktop is not running" -ForegroundColor Yellow
}

# Step 2: Show current state
Write-Host "[1/5] Stopping Docker Desktop..." -ForegroundColor Yellow
Stop-Process -Name "Docker Desktop" -Force -ErrorAction SilentlyContinue
Stop-Process -Name "com.docker.backend" -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3

# Step 3: Shutdown WSL cleanly
Write-Host "[2/5] Shutting down WSL..." -ForegroundColor Yellow
if (-not $DryRun) {
    wsl --shutdown
    Start-Sleep -Seconds 5
} else {
    Write-Host "  [DRY RUN] Would run: wsl --shutdown"
}

# Step 4: Run e2fsck via docker-desktop distro
Write-Host "[3/5] Running filesystem repair (e2fsck)..." -ForegroundColor Yellow
Write-Host "  This may take 5-15 minutes for a 1TB disk with 366K errors" -ForegroundColor Gray

if (-not $DryRun) {
    # Start only docker-desktop distro to run e2fsck
    $result = wsl -d docker-desktop -u root -- e2fsck -p /dev/sde 2>&1
    Write-Host $result

    if ($LASTEXITCODE -gt 1) {
        Write-Host ""
        Write-Host "[WARN] e2fsck returned code $LASTEXITCODE" -ForegroundColor Yellow
        Write-Host "  Code 1 = errors corrected (OK)"
        Write-Host "  Code 2 = errors corrected, reboot needed"
        Write-Host "  Code 4+ = uncorrected errors remain"
        Write-Host ""

        if ($Force -or $LASTEXITCODE -le 2) {
            Write-Host "Continuing..." -ForegroundColor Yellow
        } else {
            Write-Host "[ACTION REQUIRED] Run manually with -y flag:" -ForegroundColor Red
            Write-Host "  wsl -d docker-desktop -u root -- e2fsck -f -y /dev/sde"
            Write-Host ""
            Write-Host "Or reset Docker data (loses volumes but fixes disk):" -ForegroundColor Red
            Write-Host "  wsl --unregister docker-desktop-data"
            exit 1
        }
    } else {
        Write-Host "[OK] Filesystem repair completed successfully" -ForegroundColor Green
    }

    # Shutdown again after repair
    wsl --shutdown
    Start-Sleep -Seconds 3
} else {
    Write-Host "  [DRY RUN] Would run: wsl -d docker-desktop -u root -- e2fsck -p /dev/sde"
}

# Step 5: Restart Docker Desktop
Write-Host "[4/5] Restarting Docker Desktop..." -ForegroundColor Yellow
if (-not $DryRun) {
    Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    Write-Host "  Waiting 30 seconds for Docker to initialize..." -ForegroundColor Gray
    Start-Sleep -Seconds 30
} else {
    Write-Host "  [DRY RUN] Would start Docker Desktop"
}

# Step 6: Verify
Write-Host "[5/5] Verifying..." -ForegroundColor Yellow
if (-not $DryRun) {
    $wslStatus = wsl -l -v 2>&1
    Write-Host $wslStatus
    Write-Host ""

    # Check if docker works
    $dockerVer = wsl -- docker version --format '{{.Server.Version}}' 2>&1
    if ($dockerVer -match '\d+\.\d+') {
        Write-Host "[OK] Docker server responding: v$dockerVer" -ForegroundColor Green
    } else {
        Write-Host "[WARN] Docker not responding yet - wait 30 more seconds and retry" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "=== Repair Complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "To prevent future corruption:" -ForegroundColor White
Write-Host "  1. Always use 'wsl --shutdown' before sleep/hibernate" -ForegroundColor Gray
Write-Host "  2. Avoid force-killing Docker Desktop" -ForegroundColor Gray
Write-Host "  3. Consider adding a shutdown hook to Windows Task Scheduler" -ForegroundColor Gray
