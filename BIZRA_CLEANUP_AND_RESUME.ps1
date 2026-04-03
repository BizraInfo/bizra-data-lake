# BIZRA CLEANUP AND RESUME v1.0
# Removes conflicting containers and restarts the main stack.

$ErrorActionPreference = "SilentlyContinue"

Write-Host "[CLEANUP] Stopping conflicting legacy/rogue containers..." -ForegroundColor Yellow

# List of containers that hog ports 6379, 8080, 8000, 5432, 7474
$rogues = @(
    "bizra-redis",
    "bizra-api",
    "bizra-postgres",
    "bizra-grafana",
    "k3d-hermes-node0-serverlb",
    "bizra-node0-db",
    "bizra-jaeger"
)

# Also removing the 'genesis' stack which seems to be a duplicate run
$genesis = docker ps -a --format "{{.Names}}" | Select-String "bizra-genesis-"
foreach ($g in $genesis) { $rogues += $g.ToString() }

foreach ($r in $rogues) {
    Write-Host "   -> Killing $r..." -NoNewline
    docker rm -f $r 2>$null
    if ($?) { Write-Host "DONE" -ForegroundColor Green } else { Write-Host "NOT FOUND" -ForegroundColor Gray }
}

Write-Host "`n[RESUME] Restarting Main Stack (Wisdom/Synapse)..." -ForegroundColor Cyan
docker compose up -d synapse wisdom postgres
if ($LASTEXITCODE -ne 0) { 
    Write-Host "[ERROR] Failed to restart stack." -ForegroundColor Red
    exit 1 
}

Write-Host "[WAIT] Waiting for stabilization (10s)..."
Start-Sleep -Seconds 10

# Check status
$synapse = docker inspect --format="{{.State.Status}}" bizra-dual-agentic-system--main-synapse-1
$wisdom = docker inspect --format="{{.State.Status}}" bizra-dual-agentic-system--main-wisdom-1

Write-Host "Synapse Status: $synapse"
Write-Host "Wisdom Status: $wisdom"

if ($synapse -eq "running" -and $wisdom -eq "running") {
    Write-Host "[READY] Cleanup Complete. Running Activation..." -ForegroundColor Green
    ./BIZRA_ACTIVATION.ps1
} else {
    Write-Host "[FAIL] Stack still unhealthy. Check logs." -ForegroundColor Red
    docker logs bizra-dual-agentic-system--main-synapse-1
}
