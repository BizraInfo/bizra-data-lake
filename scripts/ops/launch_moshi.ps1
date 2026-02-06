# BIZRA Sovereign Moshi Launcher (BETA)
# Usage: ./launch_moshi.ps1

$ErrorActionPreference = "Stop"

# 1. Load Environment Variables from .env
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^(?<key>[^#\s=]+)=(?<value>.*)$") {
            $key = $Matches.key
            $value = $Matches.value.Trim()
            [System.Environment]::SetEnvironmentVariable($key, $value, "Process")
            Write-Host "✅ Loaded $key" -ForegroundColor Cyan
        }
    }
} else {
    Write-Error ".env file not found. Please ensure HF_TOKEN is configured."
}

# 2. Setup Temporary SSL Directory
$SSL_DIR = Join-Path $env:TEMP "moshi_ssl_$(Get-Random)"
if (-not (Test-Path $SSL_DIR)) { New-Item -ItemType Directory -Path $SSL_DIR | Out-Null }
Write-Host "🔐 SSL Directory: $SSL_DIR" -ForegroundColor Yellow

# 3. Ensure voices and dist exist
if (-not (Test-Path "./voices")) { New-Item -ItemType Directory -Path "./voices" | Out-Null }
if (-not (Test-Path "./dist")) { Write-Warning "./dist directory not found. Static files might fail to load." }

# 4. Launch Moshi Server
Write-Host "🚀 Launching Moshi Server (PersonaPlex)..." -ForegroundColor Green
python -m moshi.server --ssl "$SSL_DIR" --voice-prompt-dir ./voices --static ./dist
