# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  BIZRA GENESIS NODE - UNIFIED STARTUP SCRIPT                              ║
# ║  Node Zero: The First Heartbeat                                            ║
# ║                                                                            ║
# ║  Components:                                                               ║
# ║  1. Docker Services (PostgreSQL 5433, Redis 6379)                         ║
# ║  2. Rust API Server (port 3001)                                           ║
# ║  3. WebSocket Telemetry Bridge (port 8080)                                ║
# ║  4. React Dashboard (port 5173)                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

param(
    [switch]$SkipDocker,
    [switch]$ApiOnly,
    [switch]$FullStack,
    [switch]$Help
)

$ErrorActionPreference = "Continue"

# Colors for output
function Write-Banner {
    Write-Host ""
    Write-Host "╔═══════════════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║          BIZRA GENESIS NODE - NODE ZERO ACTIVATION                        ║" -ForegroundColor Cyan
    Write-Host "║                    بسم الله الرحمن الرحيم                                   ║" -ForegroundColor Yellow
    Write-Host "╚═══════════════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Status {
    param([string]$Message, [string]$Status = "INFO")
    $color = switch ($Status) {
        "OK"      { "Green" }
        "WARN"    { "Yellow" }
        "ERROR"   { "Red" }
        "INFO"    { "Cyan" }
        default   { "White" }
    }
    $icon = switch ($Status) {
        "OK"      { "✅" }
        "WARN"    { "⚠️" }
        "ERROR"   { "❌" }
        "INFO"    { "📍" }
        default   { "•" }
    }
    Write-Host "$icon $Message" -ForegroundColor $color
}

function Show-Help {
    Write-Host @"
BIZRA Genesis Node - Unified Startup Script

Usage: .\start_genesis_zero.ps1 [options]

Options:
    -SkipDocker     Skip starting Docker containers (if already running)
    -ApiOnly        Start only the Rust API server
    -FullStack      Start all components (default if no flags)
    -Help           Show this help message

Components Started:
    1. Docker Compose: postgres (5432), redis (6379)
    2. Rust API: http://localhost:3001
    3. WebSocket Bridge: ws://localhost:8080
    4. Dashboard: http://localhost:5173

Quick Start:
    .\start_genesis_zero.ps1 -FullStack

"@
}

function Start-DockerServices {
    Write-Status "Starting Docker services (Compose)..." "INFO"
    
    # Check if Docker is running
    $dockerStatus = docker info 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Status "Docker is not running! Please start Docker Desktop." "ERROR"
        return $false
    }

    $infraRoot = $env:INFRA_ROOT
    if (-not $infraRoot) { $infraRoot = (Get-Location).Path }

    $composeFile = Join-Path $infraRoot "docker-compose.database.yml"
    if (-not (Test-Path $composeFile)) {
        Write-Status "Compose file not found: $composeFile" "ERROR"
        return $false
    }

    if (-not $env:COMPOSE_PROJECT_NAME) { $env:COMPOSE_PROJECT_NAME = "bizra_node0" }
    Write-Status "COMPOSE_PROJECT_NAME=$env:COMPOSE_PROJECT_NAME" "INFO"

    docker compose -f $composeFile up -d
    if ($LASTEXITCODE -ne 0) {
        Write-Status "docker compose up failed" "ERROR"
        return $false
    }

    $pgId = (docker compose -f $composeFile ps -q postgres 2>$null)
    $redisId = (docker compose -f $composeFile ps -q redis 2>$null)

    if ($pgId) {
        $pgHealth = docker inspect $pgId --format '{{.State.Health.Status}}' 2>$null
        if ($pgHealth -eq "healthy") { Write-Status "PostgreSQL is healthy" "OK" }
        else { Write-Status "PostgreSQL health: $pgHealth (continuing anyway)" "WARN" }
    } else {
        Write-Status "PostgreSQL container id not found (continuing anyway)" "WARN"
    }

    if ($redisId) {
        $redisHealth = docker inspect $redisId --format '{{.State.Health.Status}}' 2>$null
        if ($redisHealth -eq "healthy") { Write-Status "Redis is healthy" "OK" }
        else { Write-Status "Redis health: $redisHealth (continuing anyway)" "WARN" }
    } else {
        Write-Status "Redis container id not found (continuing anyway)" "WARN"
    }

    return $true
}

function Start-RustAPI {
    Write-Status "Starting Rust API server on port 3001..." "INFO"
    
    # Set environment variables
    $env:DATABASE_URL = "postgres://bizra_user:bizra_password@localhost:5433/bizra_genesis"
    $env:PORT = "3001"
    $env:SQLX_OFFLINE = "true"
    $env:RUST_LOG = "info,bizra_genesis_node=debug"
    
    # Check if binary exists
    $apiPath = ".\target\release\api_server.exe"
    if (-not (Test-Path $apiPath)) {
        Write-Status "API binary not found. Building..." "WARN"
        cargo build --release --bin api_server
        if ($LASTEXITCODE -ne 0) {
            Write-Status "Build failed!" "ERROR"
            return $false
        }
    }
    
    # Start API in new window
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c set DATABASE_URL=postgres://bizra_user:bizra_password@localhost:5433/bizra_genesis && set PORT=3001 && set SQLX_OFFLINE=true && .\target\release\api_server.exe" -WindowStyle Normal
    
    Write-Status "API server starting..." "OK"
    Start-Sleep -Seconds 3
    
    # Health check
    try {
        $health = Invoke-RestMethod -Uri "http://localhost:3001/health" -TimeoutSec 5
        if ($health.status -eq "healthy") {
            Write-Status "API health check PASSED" "OK"
            return $true
        }
    } catch {
        Write-Status "API health check pending (may still be starting)" "WARN"
    }
    
    return $true
}

function Start-WebSocketBridge {
    Write-Status "Starting WebSocket Telemetry Bridge on port 8080..." "INFO"
    
    Push-Location ".\backend"
    
    # Check if node_modules exists
    if (-not (Test-Path "node_modules")) {
        Write-Status "Installing backend dependencies..." "INFO"
        npm install
    }
    
    # Start bridge in new window
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c node websocket.js" -WindowStyle Normal
    
    Pop-Location
    
    Write-Status "WebSocket Bridge starting..." "OK"
    Start-Sleep -Seconds 2
    
    # Health check
    try {
        $bridgeHealth = Invoke-RestMethod -Uri "http://localhost:8080/health" -TimeoutSec 5
        Write-Status "WebSocket Bridge health: $($bridgeHealth.status)" "OK"
    } catch {
        Write-Status "WebSocket Bridge health check pending" "WARN"
    }
    
    return $true
}

function Start-Dashboard {
    Write-Status "Starting React Dashboard on port 5173..." "INFO"
    
    Push-Location ".\apps\dashboard"
    
    # Check if node_modules exists
    if (-not (Test-Path "node_modules")) {
        Write-Status "Installing dashboard dependencies..." "INFO"
        npm install
    }
    
    # Start dashboard in new window
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c npm run dev" -WindowStyle Normal
    
    Pop-Location
    
    Write-Status "Dashboard starting at http://localhost:5173" "OK"
    
    return $true
}

function Show-Summary {
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  GENESIS ZERO - SYSTEM STATUS" -ForegroundColor Green
    Write-Host "═══════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  🗄️  PostgreSQL:       http://localhost:5433" -ForegroundColor White
    Write-Host "  📦 Redis:            http://localhost:6379" -ForegroundColor White
    Write-Host "  🦀 Rust API:         http://localhost:3001" -ForegroundColor Green
    Write-Host "     └─ Health:        http://localhost:3001/health" -ForegroundColor Gray
    Write-Host "     └─ Telemetry:     http://localhost:3001/telemetry" -ForegroundColor Gray
    Write-Host "     └─ SLO:           http://localhost:3001/telemetry/slo" -ForegroundColor Gray
    Write-Host "  🔌 WebSocket Bridge: ws://localhost:8080" -ForegroundColor Green
    Write-Host "     └─ Health:        http://localhost:8080/health" -ForegroundColor Gray
    Write-Host "  🎨 Dashboard:        http://localhost:5173" -ForegroundColor Green
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  Press Ctrl+C to stop components (close their windows individually)" -ForegroundColor Yellow
    Write-Host "═══════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
}

# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

if ($Help) {
    Show-Help
    exit 0
}

Write-Banner

# Load workspace contract if present (sets INFRA_ROOT / COMPOSE_PROJECT_NAME)
try {
    $resolver = Resolve-Path (Join-Path $PSScriptRoot "..\\scripts\\resolve-bizra-root.ps1") -ErrorAction Stop
    & $resolver.Path | Out-Null
} catch {
    Write-Status "Workspace resolver unavailable (continuing in local-only mode)" "WARN"
}

# Change to infra directory (default: C:\bizra-genesis-node)
$infraRoot = $env:INFRA_ROOT
if (-not $infraRoot) { $infraRoot = "C:\bizra-genesis-node" }
Set-Location $infraRoot

if ($ApiOnly) {
    # API only mode
    if (-not $SkipDocker) {
        Start-DockerServices
    }
    Start-RustAPI
    Write-Host ""
    Write-Host "API server running at http://localhost:3001" -ForegroundColor Green
    Write-Host ""
} else {
    # Full stack mode (default)
    if (-not $SkipDocker) {
        $dockerOk = Start-DockerServices
        if (-not $dockerOk) {
            Write-Status "Docker services failed to start!" "ERROR"
            exit 1
        }
    }
    
    Start-RustAPI
    Start-Sleep -Seconds 2
    
    Start-WebSocketBridge
    Start-Sleep -Seconds 1
    
    Start-Dashboard
    
    Show-Summary
}

Write-Host "🤲 Alhamdulillah - Node Zero is ALIVE!" -ForegroundColor Green
Write-Host ""
