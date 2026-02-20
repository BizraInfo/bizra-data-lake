# ============================================================
# BIZRA Alpha-100 Installer Bootstrap (Windows)
# ============================================================
# Usage:
#   irm https://install.bizra.ai/alpha100/win | iex
#   OR
#   .\bizra-install.ps1 [-Provider local] [-LocalBackend ollama]
#                       [-Model "llama3.1:8b"] [-ReflexMode shadow]
#                       [-PolicyFile <path>] [-StateDir <path>]
# ============================================================

param(
    [string]$Provider = "",
    [string]$LocalBackend = "",
    [string]$Model = "",
    [string]$ReflexMode = "",
    [string]$PolicyFile = "",
    [string]$StateDir = ""
)

$ErrorActionPreference = "Stop"
$BIZRA_VERSION = if ($env:BIZRA_VERSION) { $env:BIZRA_VERSION } else { "0.1.0" }
$BIZRA_REPO = "https://github.com/bizra-ai/bizra-omega"
$INSTALL_DIR = Join-Path $env:USERPROFILE ".bizra"
$BIN_DIR = Join-Path $INSTALL_DIR "bin"

function Write-Info($msg)  { Write-Host "  [ok]  $msg" -ForegroundColor Green }
function Write-Warn($msg)  { Write-Host "  [!!]  $msg" -ForegroundColor Yellow }
function Write-Fail($msg)  { Write-Host "  [ERR] $msg" -ForegroundColor Red; exit 1 }

# ── Detect Architecture ──────────────────────────────────────
function Get-Target {
    $arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
    switch ($arch) {
        "X64"   { return "windows-x86_64" }
        "Arm64" { return "windows-aarch64" }
        default { Write-Fail "Unsupported architecture: $arch" }
    }
}

# ── Check prerequisites ─────────────────────────────────────
function Test-Prerequisites {
    $nodeCmd = Get-Command node -ErrorAction SilentlyContinue
    if ($nodeCmd) {
        $nodeVer = & node --version 2>$null
        Write-Info "Node.js: $nodeVer"
    } else {
        Write-Warn "Node.js not found. Required for LLM bridge."
    }
}

# ── Download binary ──────────────────────────────────────────
function Get-Binary {
    $target = Get-Target
    Write-Info "Platform: $target"

    $artifact = "bizra-install-$BIZRA_VERSION-$target"
    $url = "$BIZRA_REPO/releases/download/v$BIZRA_VERSION/$artifact.zip"
    $checksumUrl = "$BIZRA_REPO/releases/download/v$BIZRA_VERSION/SHA256SUMS"
    $tempZip = Join-Path $env:TEMP "$artifact.zip"

    New-Item -ItemType Directory -Force -Path $BIN_DIR | Out-Null

    Write-Host ""
    Write-Host "  Downloading BIZRA installer..."

    try {
        Invoke-WebRequest -Uri $url -OutFile $tempZip -UseBasicParsing -ErrorAction Stop

        # Verify checksum if available
        try {
            $checksums = Invoke-WebRequest -Uri $checksumUrl -UseBasicParsing -ErrorAction Stop
            $expected = ($checksums.Content -split "`n" | Where-Object { $_ -match "$artifact.zip" }) -replace '\s+.*',''
            if ($expected) {
                $actual = (Get-FileHash -Algorithm SHA256 $tempZip).Hash.ToLower()
                if ($expected.ToLower() -ne $actual) {
                    Write-Fail "Checksum mismatch! Expected: $expected, Got: $actual"
                }
                Write-Info "Checksum verified"
            }
        } catch {
            Write-Warn "Checksum file not available. Skipping verification."
        }

        Expand-Archive -Path $tempZip -DestinationPath $BIN_DIR -Force
        $script:INSTALLER = Join-Path $BIN_DIR "bizra-install.exe"
        Write-Info "Installed: $($script:INSTALLER)"
        return $true
    } catch {
        Write-Warn "Release artifact not available. Trying source build..."
        return $false
    }
}

# ── Source build fallback ────────────────────────────────────
function Build-FromSource {
    $cargo = Get-Command cargo -ErrorAction SilentlyContinue
    if (-not $cargo) {
        Write-Fail "No prebuilt binary available and Rust toolchain not found.`n  Install Rust: https://rustup.rs/"
    }

    $rustVer = & rustc --version 2>$null
    Write-Info "Rust toolchain: $rustVer"

    $workspaceDir = $null
    if (Test-Path ".\bizra-omega\Cargo.toml") { $workspaceDir = ".\bizra-omega" }
    elseif (Test-Path "..\bizra-omega\Cargo.toml") { $workspaceDir = "..\bizra-omega" }
    else { Write-Fail "Cannot find bizra-omega workspace for source build." }

    Write-Host ""
    Write-Host "  Building from source (this may take a few minutes)..."
    Push-Location $workspaceDir
    try {
        & cargo build --release -p bizra-installer -p bizra-node
        if ($LASTEXITCODE -ne 0) { Write-Fail "Source build failed." }
    } finally {
        Pop-Location
    }

    New-Item -ItemType Directory -Force -Path $BIN_DIR | Out-Null
    Copy-Item "$workspaceDir\target\release\bizra-install.exe" $BIN_DIR -Force -ErrorAction SilentlyContinue
    Copy-Item "$workspaceDir\target\release\bizra-node.exe" $BIN_DIR -Force -ErrorAction SilentlyContinue

    $script:INSTALLER = Join-Path $BIN_DIR "bizra-install.exe"
    Write-Info "Built from source: $($script:INSTALLER)"
}

# ── Run installer ────────────────────────────────────────────
function Invoke-Installer {
    if (-not (Test-Path $script:INSTALLER)) {
        Write-Fail "Installer binary not found at $($script:INSTALLER)"
    }

    $args = @("alpha100", "install")
    if ($Provider)     { $args += "--provider"; $args += $Provider }
    if ($LocalBackend) { $args += "--local-backend"; $args += $LocalBackend }
    if ($Model)        { $args += "--model"; $args += $Model }
    if ($ReflexMode)   { $args += "--reflex-mode"; $args += $ReflexMode }
    if ($PolicyFile)   { $args += "--policy-file"; $args += $PolicyFile }
    if ($StateDir)     { $args += "--state-dir"; $args += $StateDir }

    Write-Host ""
    Write-Host "  Running: bizra-install $($args -join ' ')"
    Write-Host ""
    & $script:INSTALLER @args
}

# ── Add to PATH ──────────────────────────────────────────────
function Update-Path {
    $currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    if ($currentPath -notlike "*$BIN_DIR*") {
        [Environment]::SetEnvironmentVariable("PATH", "$BIN_DIR;$currentPath", "User")
        $env:PATH = "$BIN_DIR;$env:PATH"
        Write-Info "Added $BIN_DIR to user PATH"
    }
}

# ── Main ─────────────────────────────────────────────────────
Write-Host ""
Write-Host "  BIZRA Alpha-100 Installer"
Write-Host "  ========================="
Write-Host ""

Test-Prerequisites

$script:INSTALLER = ""
$downloaded = Get-Binary
if (-not $downloaded) { Build-FromSource }

Update-Path
Invoke-Installer

Write-Host ""
Write-Host "  ========================="
Write-Host "  Installation complete."
Write-Host ""
Write-Host "  Next steps:"
Write-Host "    1. bizra-install alpha100 doctor    # Verify installation"
Write-Host "    2. bizra-install alpha100 launch    # Start Node0 in shadow mode"
Write-Host '    3. node "$env:USERPROFILE\.bizra\alpha100\llm_bridge.js"  # Connect LLM bridge'
Write-Host ""
Write-Host "  Your knowledge stays on YOUR device."
Write-Host ""
