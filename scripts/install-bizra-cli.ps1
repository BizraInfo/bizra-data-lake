# ═══════════════════════════════════════════════════════════════
# BIZRA CLI Installer — Windows PowerShell
# Makes 'bizra' available from any terminal
#
# Usage: .\install-bizra-cli.ps1
#
# After install, open new terminal and type: bizra
# ═══════════════════════════════════════════════════════════════

$ErrorActionPreference = "Stop"

$BIZRA_HOME = "$env:LOCALAPPDATA\BIZRA"
$BIZRA_BIN = "$BIZRA_HOME\bin"

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   BIZRA CLI Installer — Windows                  ║" -ForegroundColor Cyan
Write-Host "║   Sovereign Mission Operating System             ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Step 1: Create directories
Write-Host "→ Creating BIZRA directories..." -ForegroundColor Gray
$dirs = @("bin", "sovereign_state", "logs", "models")
foreach ($d in $dirs) {
    New-Item -ItemType Directory -Force -Path "$BIZRA_HOME\$d" | Out-Null
}
Write-Host "  ✓ Directories created at $BIZRA_HOME" -ForegroundColor Green

# Step 2: Copy CLI
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$cliSource = Join-Path $scriptDir "bizra-cli.py"

if (Test-Path $cliSource) {
    Copy-Item $cliSource "$BIZRA_BIN\bizra_cli.py" -Force
    Write-Host "  ✓ CLI installed" -ForegroundColor Green
} else {
    Write-Host "  ✗ Cannot find bizra-cli.py" -ForegroundColor Red
    exit 1
}

# Step 3: Create .bat launcher
$batContent = @"
@echo off
REM BIZRA CLI launcher
set BIZRA_HOME=$BIZRA_HOME

REM Find Python
if exist "%BIZRA_ROOT%\.venv\Scripts\python.exe" (
    set PY=%BIZRA_ROOT%\.venv\Scripts\python.exe
) else if exist "%BIZRA_ROOT%\.venv-linux\bin\python" (
    set PY=%BIZRA_ROOT%\.venv-linux\bin\python
) else (
    set PY=python
)

"%PY%" "$BIZRA_BIN\bizra_cli.py" %*
"@

Set-Content -Path "$BIZRA_BIN\bizra.bat" -Value $batContent
Write-Host "  ✓ Launcher created at $BIZRA_BIN\bizra.bat" -ForegroundColor Green

# Step 4: Auto-detect roots
$detectedRoot = $null
$candidates = @(
    "C:\BIZRA-DATA-LAKE",
    "$env:USERPROFILE\BIZRA-DATA-LAKE",
    "$env:USERPROFILE\BIZRA"
)
foreach ($c in $candidates) {
    if (Test-Path "$c\core\sovereign\api.py") {
        $detectedRoot = $c
        break
    }
}

$detectedFrontend = $null
$feCandidates = @(
    "C:\award-winner-design",
    "$env:USERPROFILE\award-winner-design"
)
foreach ($c in $feCandidates) {
    if ((Test-Path "$c\next.config.mjs") -or (Test-Path "$c\next.config.js")) {
        $detectedFrontend = $c
        break
    }
}

# Step 5: Add to PATH
$currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($currentPath -notlike "*$BIZRA_BIN*") {
    [Environment]::SetEnvironmentVariable("PATH", "$BIZRA_BIN;$currentPath", "User")
    Write-Host "  ✓ Added to user PATH" -ForegroundColor Green
} else {
    Write-Host "  ✓ Already in PATH" -ForegroundColor Green
}

# Set env vars
[Environment]::SetEnvironmentVariable("BIZRA_HOME", $BIZRA_HOME, "User")
if ($detectedRoot) {
    [Environment]::SetEnvironmentVariable("BIZRA_ROOT", $detectedRoot, "User")
}
if ($detectedFrontend) {
    [Environment]::SetEnvironmentVariable("BIZRA_FRONTEND", $detectedFrontend, "User")
}

# Also set for current session
$env:PATH = "$BIZRA_BIN;$env:PATH"
$env:BIZRA_HOME = $BIZRA_HOME
if ($detectedRoot) { $env:BIZRA_ROOT = $detectedRoot }
if ($detectedFrontend) { $env:BIZRA_FRONTEND = $detectedFrontend }

Write-Host ""
Write-Host "─────────────────────────────────────────────────" -ForegroundColor Gray
Write-Host ""
Write-Host "  ✓ BIZRA CLI installed!" -ForegroundColor Green
Write-Host ""
if ($detectedRoot) {
    Write-Host "  BIZRA_ROOT:     $detectedRoot" -ForegroundColor White
}
if ($detectedFrontend) {
    Write-Host "  BIZRA_FRONTEND: $detectedFrontend" -ForegroundColor White
}
Write-Host "  BIZRA_HOME:     $BIZRA_HOME" -ForegroundColor White
Write-Host ""
Write-Host "  Open a NEW terminal, then type: " -NoNewline
Write-Host "bizra" -ForegroundColor Cyan
Write-Host ""
Write-Host "  `"One mission, one proof, remembered forever.`"" -ForegroundColor DarkYellow
Write-Host ""
