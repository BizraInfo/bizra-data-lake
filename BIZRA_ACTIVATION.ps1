# BIZRA ACTIVATION SEQUENCE v1.0 [DUAL-BRAIN LINK]
# Brings the Application Layer (Kernel + Elite) online, connecting to the now-active Knowledge Graph.
#
# LOGIC:
# 1. Prerequisite Check (Wisdom/Synapse)
# 2. Vector Store Activation (ChromaDB)
# 3. Dual-Brain Ignition (Python Kernel + Rust Elite)
# 4. System Health Verification

$ErrorActionPreference = "Stop"

function Print-Status($msg) { Write-Host "`n[BIZRA-ACTIVATION] $msg" -ForegroundColor Cyan }
function Print-Success($msg) { Write-Host "[SUCCESS] $msg`n" -ForegroundColor Green }
function Print-Error($msg) { Write-Host "[ERROR] $msg`n" -ForegroundColor Red }

Print-Status "Verifying Substrate Health (Wisdom & Synapse)..."

$wisdom = docker inspect --format="{{.State.Status}}" bizra-dual-agentic-system--main-wisdom-1 2>$null
$synapse = docker inspect --format="{{.State.Status}}" bizra-dual-agentic-system--main-synapse-1 2>$null

if ($wisdom -ne "running" -or $synapse -ne "running") {
    Print-Error "Substrate Inactive. Please run BIZRA_IGNITION.ps1 first."
    exit 1
}
Print-Success "Substrate Active."

# 2. APPLICATION LAUNCH
Print-Status "Igniting Cognitive Layers: Vectors, Kernel, Elite..."

# We bring up vectors first as it's a dependency
docker compose up -d vectors
if ($LASTEXITCODE -ne 0) { exit 1 }

# Then the dual brains
docker compose up -d kernel elite
if ($LASTEXITCODE -ne 0) { exit 1 }

# 3. HEALTH MONITORING
Print-Status "Waiting for Neural Synchronization (Health Checks)..."
$services = @("bizra-dual-agentic-system--main-vectors-1", "bizra-dual-agentic-system--main-kernel-1", "bizra-dual-agentic-system--main-elite-1")

$timeout = 60 # seconds
$startTime = Get-Date

foreach ($svc in $services) {
    Write-Host -NoNewline "Checking $svc "
    $healthy = $false
    do {
        $status = docker inspect --format="{{.State.Health.Status}}" $svc 2>$null
        # vectors might not have a healthcheck in compose, check status=running
        if ([string]::IsNullOrEmpty($status)) {
            $status = docker inspect --format="{{.State.Status}}" $svc 2>$null
            if ($status -eq "running") { $status = "healthy" }
        }

        if ($status -eq "healthy") { 
            $healthy = $true 
            Write-Host " [OK]" -ForegroundColor Green
        } else {
            Start-Sleep -Seconds 2
            Write-Host "." -NoNewline
        }
    } while (-not $healthy -and ((Get-Date) - $startTime).TotalSeconds -lt $timeout)
    
    if (-not $healthy) {
        Write-Host " [TIMEOUT/FAILED]" -ForegroundColor Red
        Print-Error "Service $svc failed to stabilize."
        docker logs --tail 20 $svc
        exit 1
    }
}

# 4. FINAL VERIFICATION
Print-Status "Testing Public Health Interface (Port 8080)..."
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8080/health" -Method Get -ErrorAction Stop
    if ($response.StatusCode -eq 200) {
        Print-Success "BIZRA SYSTEM ONLINE. DUAL-AGENTIC CORE ACTIVE."
        Print-Status "Metrics: http://localhost:9090"
        Print-Status "Elite API: http://localhost:8080"
        Print-Status "Kernel API: http://localhost:8010"
    } else {
        throw "Status check returned $($response.StatusCode)"
    }
} catch {
    Print-Error "Health check failed: $_"
    Print-Status "Dumping Elite Logs..."
    docker logs --tail 50 bizra-dual-agentic-system--main-elite-1
    exit 1
}
