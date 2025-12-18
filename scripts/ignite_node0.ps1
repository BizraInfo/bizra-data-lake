param(
  [string]$ComposeFile = "docker-compose.yml",
  [string]$HealthUrl = "http://127.0.0.1:8010/healthz",
  [int]$TimeoutSec = 120,
  [int]$PollIntervalSec = 2
)

$ErrorActionPreference = "Stop"

function Fail($msg) {
  throw $msg
}

function Info($msg) {
  Write-Host "[INFO] $msg" -ForegroundColor Cyan
}

function Ok($msg) {
  Write-Host "[OK]  $msg" -ForegroundColor Green
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  Fail "docker not found on PATH."
}

try {
  docker info | Out-Null
} catch {
  Fail "Docker is not running (docker info failed). Start Docker Desktop / daemon and retry."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

try {
  $composePath = (Resolve-Path (Join-Path $repoRoot $ComposeFile)).Path
} catch {
  Fail "Compose file not found: $ComposeFile (repoRoot=$repoRoot)"
}

$composeDir = Split-Path $composePath -Parent
$composeName = Split-Path $composePath -Leaf

try {
  Push-Location $composeDir
  try {
    Info "Igniting Node0 stack via Docker Compose..."
    docker compose -f $composeName up -d --build

    Info "Polling readiness: $HealthUrl (timeout=${TimeoutSec}s, interval=${PollIntervalSec}s)"
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    $lastError = $null

    while ((Get-Date) -lt $deadline) {
      try {
        $resp = Invoke-RestMethod -Method Get -Uri $HealthUrl -TimeoutSec 1
        if ($resp -and $resp.status -eq "ok") {
          Ok "Kernel is healthy."
          return
        }
        $lastError = "healthz returned non-ok status"
      } catch {
        $lastError = $_.Exception.Message
      }
      Start-Sleep -Seconds $PollIntervalSec
    }

    Write-Host ""
    Info "Kernel did not become healthy before timeout. Last error: $lastError"

    Write-Host ""
    Info "docker compose ps"
    docker compose -f $composeName ps

    Write-Host ""
    Info "Last 200 lines of kernel logs"
    docker compose -f $composeName logs kernel --tail 200

    Write-Host ""
    Info "Last 200 lines of wisdom logs"
    docker compose -f $composeName logs wisdom --tail 200

    Fail "Kernel did not become healthy before timeout. Last error: $lastError"
  } finally {
    Pop-Location
  }
} catch {
  Write-Host ""
  Write-Host "[FAIL] $($_.Exception.Message)" -ForegroundColor Red
  exit 1
}
