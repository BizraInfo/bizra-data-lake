# ═══════════════════════════════════════════════════════════════
#  BIZRA Sovereign Kernel — Windows Auto-Start Registration
#  Adds kernel launcher to Windows Startup (current user)
#  Run: powershell -ExecutionPolicy Bypass -File register-kernel-autostart.ps1
# ═══════════════════════════════════════════════════════════════

param(
    [switch]$Unregister
)

$ErrorActionPreference = "Stop"

# ── Paths ──
$scriptDir    = Split-Path -Parent $MyInvocation.MyCommand.Path
$bizraRoot    = Split-Path -Parent $scriptDir
$launcherVbs  = Join-Path $scriptDir "bizra-kernel-launcher.vbs"
$startupDir   = [Environment]::GetFolderPath("Startup")
$shortcutPath = Join-Path $startupDir "BIZRA Sovereign Kernel.lnk"

# ── Unregister ──
if ($Unregister) {
    if (Test-Path $shortcutPath) {
        Remove-Item $shortcutPath -Force
        Write-Host "[BIZRA] Auto-start REMOVED from: $shortcutPath" -ForegroundColor Yellow
    } else {
        Write-Host "[BIZRA] No auto-start entry found." -ForegroundColor Gray
    }
    exit 0
}

# ── Verify launcher exists ──
if (-not (Test-Path $launcherVbs)) {
    Write-Host "[BIZRA] ERROR: Launcher not found at: $launcherVbs" -ForegroundColor Red
    exit 1
}

# ── Create shortcut in Startup folder ──
$wshShell = New-Object -ComObject WScript.Shell
$shortcut = $wshShell.CreateShortcut($shortcutPath)
$shortcut.TargetPath       = "wscript.exe"
$shortcut.Arguments        = """$launcherVbs"""
$shortcut.WorkingDirectory = $scriptDir
$shortcut.Description      = "BIZRA Sovereign Kernel - Auto-start on boot"
$shortcut.IconLocation     = "shell32.dll,13"
$shortcut.WindowStyle      = 7  # Minimized
$shortcut.Save()

Write-Host ""
Write-Host "  ╔══════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "  ║   BIZRA SOVEREIGN KERNEL — AUTO-START SET    ║" -ForegroundColor Cyan
Write-Host "  ╠══════════════════════════════════════════════╣" -ForegroundColor Cyan
Write-Host "  ║  Shortcut: $($shortcutPath | Split-Path -Leaf)" -ForegroundColor White
Write-Host "  ║  Location: $startupDir" -ForegroundColor Gray
Write-Host "  ║  Launcher: bizra-kernel-launcher.vbs" -ForegroundColor Gray
Write-Host "  ║                                              ║" -ForegroundColor Cyan
Write-Host "  ║  Next OS boot → BIZRA starts automatically   ║" -ForegroundColor Green
Write-Host "  ╚══════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "  To remove: .\register-kernel-autostart.ps1 -Unregister" -ForegroundColor DarkGray
Write-Host ""

# ── Also ensure sovereign_state dir exists for the flag file ──
$stateDir = Join-Path $bizraRoot "sovereign_state"
if (-not (Test-Path $stateDir)) {
    New-Item -ItemType Directory -Path $stateDir -Force | Out-Null
    Write-Host "[BIZRA] Created sovereign_state directory" -ForegroundColor Gray
}
