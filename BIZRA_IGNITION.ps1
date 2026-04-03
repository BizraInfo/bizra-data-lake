# BIZRA IGNITION SEQUENCE v2.0 [SELF-HEALING]
# Automates the "Cold-to-Hot" transition of the Sovereign Knowledge Base.

$ErrorActionPreference = "Continue"

function Print-Status($msg) { Write-Host "
[BIZRA-IGNITION] $msg" -ForegroundColor Cyan }
function Print-Success($msg) { Write-Host "[SUCCESS] $msg
" -ForegroundColor Green }
function Print-Error($msg) { Write-Host "[ERROR] $msg
" -ForegroundColor Red }

Print-Status "Checking Container Runtime Health..."

$proc = Get-Process "Docker Desktop" -ErrorAction SilentlyContinue
$pipePath = "\\.\pipe\dockerDesktopLinuxEngine"
$pipeExists = Test-Path $pipePath

$restartNeeded = $false

if ($null -eq $proc) {
    Print-Status "Docker Desktop process NOT found."
    $restartNeeded = $true
} elseif (-not $pipeExists) {
    Print-Status "Docker Desktop process FOUND but Engine Pipe MISSING (Zombie/Stuck state)."
    $restartNeeded = $true
    Print-Status "Initiating Hard Kill..."
    Stop-Process -Name "Docker Desktop" -Force -ErrorAction SilentlyContinue
    Stop-Process -Name "com.docker.backend" -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
} else {
    $info = docker info 2>&1
    if ($LASTEXITCODE -ne 0 -or "$info".Contains("failed to connect")) {
        Print-Status "Docker Pipe exists but Engine unresponsive."
        $restartNeeded = $true
    }
}

if ($restartNeeded) {
    Print-Status "Igniting Docker Desktop..."
    if (Test-Path "C:\Program Files\Docker\Docker\Docker Desktop.exe") {
        Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    } else {
        Print-Error "Docker Desktop executable not found."
        exit 1
    }

    Print-Status "Waiting for Engine warmup (Up to 120s)..."
    $ready = $false
    $retries = 0
    do {
        Start-Sleep -Seconds 5
        if (Test-Path $pipePath) {
            $info = docker info 2>&1
            if ($LASTEXITCODE -eq 0 -and -not "$info".Contains("failed to connect")) {
                $ready = $true
            }
        }
        Write-Host "." -NoNewline
        $retries++
    } while (-not $ready -and $retries -lt 24)
    Write-Host ""
    
    if (-not $ready) {
        Print-Error "Docker Engine failed to stabilize. Manual intervention required."
        exit 1
    }
}

Print-Success "Container Runtime Verified & Active."

$ErrorActionPreference = "Stop"

Print-Status "Activating 'Wisdom' (Neo4j) and 'Synapse' (Redis) Nodes..."
docker compose up -d wisdom synapse
if ($LASTEXITCODE -ne 0) { exit 1 }

Print-Status "Waiting for Wisdom Node (Port 7687) to accept connections..."
$neo4jReady = $false
$retries = 0
do {
    $conn = Test-NetConnection -ComputerName localhost -Port 7687 -WarningAction SilentlyContinue
    if ($conn.TcpTestSucceeded) { $neo4jReady = $true }
    else { Start-Sleep -Seconds 3; Write-Host "." -NoNewline }
    $retries++
} while (-not $neo4jReady -and $retries -lt 30)
Write-Host ""

if (-not $neo4jReady) {
    Print-Error "Wisdom Node failed to stabilize on port 7687. Check logs: docker compose logs wisdom"
    exit 1
}
Print-Success "Wisdom Node Online."

Print-Status "Injecting 104,908 Verified Artifacts into Knowledge Graph..."
if ([string]::IsNullOrWhiteSpace($env:NEO4J_PASSWORD)) {
    Print-Error "NEO4J_PASSWORD is not set. Export it securely before running ignition."
    exit 1
}
Print-Status "Using NEO4J_PASSWORD from environment."

python bizra_synaptic_loader.py --receipt-out docs/evidence/receipts/ignition_receipt.json

if ($LASTEXITCODE -eq 0) {
    Print-Success "NEURAL INJECTION COMPLETE."
    Print-Status "System State: PEAK MASTERPIECE [ACTIVE]"
} else {
    Print-Error "Synaptic Loader encountered an issue."
    exit 1
}
