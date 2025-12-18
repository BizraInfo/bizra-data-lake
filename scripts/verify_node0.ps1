param(
  [string]$ComposeFile = "docker-compose.yml",
  [string]$BaseUrl = "http://127.0.0.1:8010",
  [string]$EnvFile = ".env",
  [int]$HealthSamples = 10,
  [int]$HealthSleepMs = 300,
  [int]$RestartStormCount = 10,
  [int]$RestartRecoverySec = 10
)

$ErrorActionPreference = "Stop"

function Info($msg) {
  Write-Host "[INFO] $msg" -ForegroundColor Cyan
}

function Ok($msg) {
  Write-Host "[OK]  $msg" -ForegroundColor Green
}

function Warn($msg) {
  Write-Host "[WARN] $msg" -ForegroundColor Yellow
}

function Get-EnvValue($path, $key) {
  if (-not (Test-Path $path)) { return $null }
  $lines = Get-Content $path -ErrorAction SilentlyContinue
  foreach ($l in $lines) {
    $line = $l.Trim()
    if (-not $line -or $line.StartsWith("#")) { continue }
    if ($line -match "^(?<k>[^=]+)=(?<v>.*)$") {
      if ($Matches["k"].Trim() -eq $key) {
        return $Matches["v"].Trim()
      }
    }
  }
  return $null
}

function Get-Healthz($url) {
  try {
    return Invoke-RestMethod -Method Get -Uri $url -TimeoutSec 1
  } catch {
    return $null
  }
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  throw "docker not found on PATH."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$gatesOutFile = Join-Path $repoRoot "docs/evidence/gates/node0_gates_latest.json"
$allGates = @("A","B","C","D","E")
$gateStatuses = New-Object System.Collections.Specialized.OrderedDictionary
$gateDetails = New-Object System.Collections.Specialized.OrderedDictionary
foreach ($g in $allGates) {
  $gateStatuses[$g] = "NOT_RUN"
}

function Set-GateStatus([string]$gateId, [string]$status, [string]$detail = "") {
  $gateStatuses[$gateId] = $status
  if ($detail) {
    $gateDetails[$gateId] = $detail
  }
}

$overallPassed = $false
$failureMessage = ""
$healthUrl = ($BaseUrl.TrimEnd("/") + "/healthz")

try {
  $composePath = (Resolve-Path (Join-Path $repoRoot $ComposeFile)).Path
  $composeDir = Split-Path $composePath -Parent
  $composeName = Split-Path $composePath -Leaf
  Push-Location $composeDir

  Info "Gate A: Health stability ($HealthSamples samples, sleep=${HealthSleepMs}ms)"
  for ($i = 1; $i -le $HealthSamples; $i++) {
    $h = Get-Healthz $healthUrl
    if (-not $h -or $h.status -ne "ok") {
      Set-GateStatus "A" "FAIL" "Sample ${i} returned non-ok healthz"
      throw "Gate A failed at sample ${i}: /healthz not OK."
    }
    Start-Sleep -Milliseconds $HealthSleepMs
  }
  Set-GateStatus "A" "PASS" "Stable readiness confirmed"
  Ok "Gate A passed."

  Info "Gate B: Restart resilience (kernel x$RestartStormCount, recover <= ${RestartRecoverySec}s each)"
  for ($i = 1; $i -le $RestartStormCount; $i++) {
    docker compose -f $composeName restart kernel | Out-Null
    $deadline = (Get-Date).AddSeconds($RestartRecoverySec)
    $okRecover = $false
    while ((Get-Date) -lt $deadline) {
      $h = Get-Healthz $healthUrl
      if ($h -and $h.status -eq "ok") {
        $okRecover = $true
        break
      }
      Start-Sleep -Milliseconds 250
    }
    if (-not $okRecover) {
      Set-GateStatus "B" "FAIL" "Restart ${i} failed to recover within ${RestartRecoverySec}s"
      throw "Gate B failed on restart ${i}: /healthz did not recover in time."
    }
  }
  Set-GateStatus "B" "PASS" "Kernel recovers after restart storms"
  Ok "Gate B passed."

  Info "Gate C: Sovereign mount is read-only (/mnt/bizra_projects)"
  $out = docker compose -f $composeName exec -T kernel sh -lc "if echo test > /mnt/bizra_projects/__ro_test 2>/dev/null; then rm -f /mnt/bizra_projects/__ro_test; echo RO_FAIL; else echo RO_OK; fi"
  if ($out -notmatch "RO_OK") {
    Set-GateStatus "C" "FAIL" "Kernel mount accepted write: $out"
    throw "Gate C failed: expected RO_OK, got: $out"
  }
  Set-GateStatus "C" "PASS" "Kernel mount rejected writes"
  Ok "Gate C passed."

  Info "Gate D: LLM reachability observable (DEGRADED allowed, must be explicit)"
  $h = Get-Healthz $healthUrl
  if (-not $h -or $h.status -ne "ok") {
    Set-GateStatus "D" "FAIL" "/healthz not OK ($($h.status))"
    throw "Gate D failed: /healthz not OK."
  }

  $checks = $h.checks
  if ($null -eq $checks) {
    Set-GateStatus "D" "FAIL" "checks payload missing"
    throw "Gate D failed: checks payload missing."
  }

  $llm = $checks.llm
  if ($null -eq $llm) {
    Set-GateStatus "D" "FAIL" "checks.llm missing"
    throw "Gate D failed: checks.llm missing."
  }

  if ($llm.enabled -eq $true) {
    if ($llm.ok -eq $false -and $llm.status -ne "DEGRADED") {
      Set-GateStatus "D" "FAIL" "LLM flagged not ok but status=$($llm.status)"
      throw "Gate D failed: LLM not OK but status != DEGRADED."
    }
    if ($llm.ok -eq $true -and $llm.status -ne "OK") {
      Set-GateStatus "D" "FAIL" "LLM ok but status=$($llm.status)"
      throw "Gate D failed: LLM OK but status != OK."
    }

    $ollama = $checks.ollama
    $lmstudio = $checks.lmstudio
    if ($null -eq $ollama) {
      Set-GateStatus "D" "FAIL" "checks.ollama missing"
      throw "Gate D failed: checks.ollama missing."
    }
    if ($null -eq $lmstudio) {
      Set-GateStatus "D" "FAIL" "checks.lmstudio missing"
      throw "Gate D failed: checks.lmstudio missing."
    }

    if ($ollama.enabled -eq $true) {
      if ($ollama.ok -eq $true -and $ollama.status -ne "ok") {
        Set-GateStatus "D" "FAIL" "Ollama ok but status=$($ollama.status)"
        throw "Gate D failed: ollama OK but checks.ollama.status != ok."
      }
      if ($ollama.ok -eq $false -and $ollama.status -ne "degraded") {
        Set-GateStatus "D" "FAIL" "Ollama degraded but status=$($ollama.status)"
        throw "Gate D failed: ollama not OK but checks.ollama.status != degraded."
      }
    } else {
      if ($ollama.status -ne "disabled") {
        Warn "Gate D: ollama disabled but status=$($ollama.status)"
      }
    }

    if ($lmstudio.enabled -eq $true) {
      if ($lmstudio.ok -eq $true -and $lmstudio.status -ne "ok") {
        Set-GateStatus "D" "FAIL" "LMStudio ok but status=$($lmstudio.status)"
        throw "Gate D failed: lmstudio OK but checks.lmstudio.status != ok."
      }
      if ($lmstudio.ok -eq $false -and $lmstudio.status -ne "degraded") {
        Set-GateStatus "D" "FAIL" "LMStudio degraded but status=$($lmstudio.status)"
        throw "Gate D failed: lmstudio not OK but checks.lmstudio.status != degraded."
      }
    } else {
      if ($lmstudio.status -ne "disabled") {
        Warn "Gate D: lmstudio disabled but status=$($lmstudio.status)"
      }
    }
  } else {
    if ($llm.status -ne "disabled") {
      Set-GateStatus "D" "FAIL" "LLM enabled flag false but status=$($llm.status)"
      throw "Gate D failed: LLM endpoints report inconsistent disabled status."
    }
    Warn "Gate D: LLM endpoints not configured (checks.llm.enabled=false)."
  }
  Set-GateStatus "D" "PASS" "LLM observability verified"
  Ok "Gate D passed."

  Info "Gate E: Evidence fail-closed correctness (H-stakes)"
  $token = $env:BIZRA_API_TOKEN
  if (-not $token) {
    $envPath = $EnvFile
    if (-not [System.IO.Path]::IsPathRooted($EnvFile)) {
      $envPath = Join-Path $repoRoot $EnvFile
    }
    $token = Get-EnvValue $envPath "BIZRA_API_TOKEN"
  }
  if (-not $token) {
    Set-GateStatus "E" "FAIL" "Missing BIZRA_API_TOKEN"
    throw "Gate E failed: Missing BIZRA_API_TOKEN (set env var or add to $EnvFile)."
  }

  $sapeUrl = ($BaseUrl.TrimEnd("/") + "/v1/sape/plan")
  $payload = @{
    domain = "Meta-Cognitive Audit"
    objective = "Verify fail-closed evidence gating for H-stakes SAPE requests."
    stakes = "H"
    constraints = "No web. Evidence required. Must BLOCK if evidence is missing."
    success_criteria = "Return BLOCKED_BY_EVIDENCE when evidence kernels cannot be retrieved."
    require_graph_evidence = $true
    evidence_topics = @("__SAPE_GATE_E_NON_EXISTENT_TOPIC__")
    evidence_limit = 1
  } | ConvertTo-Json -Depth 6

  try {
    $resp = Invoke-RestMethod -Method Post -Uri $sapeUrl -Headers @{ "X-BIZRA-TOKEN" = $token } -ContentType "application/json" -Body $payload -TimeoutSec 10
  } catch {
    Set-GateStatus "E" "FAIL" "SAPE plan request failed: $($_.Exception.Message)"
    throw "Gate E failed: SAPE plan request failed: $($_.Exception.Message)"
  }

  if (-not $resp -or $resp.status -ne "BLOCKED_BY_EVIDENCE") {
    Set-GateStatus "E" "FAIL" "Expected BLOCKED_BY_EVIDENCE, got $($resp.status)"
    throw "Gate E failed: expected status=BLOCKED_BY_EVIDENCE, got: $($resp.status)"
  }

  Set-GateStatus "E" "PASS" "E1 confirmed fail-closed when evidence missing"
  Ok "Gate E passed (E1)."

  $overallPassed = $true
  Ok "All acceptance gates passed."
} catch {
  $failureMessage = $_.Exception.Message
  if (-not $failureMessage) {
    $failureMessage = "verify_node0.ps1: unknown failure"
  }
  Write-Host "[FAIL] $failureMessage" -ForegroundColor Red
} finally {
  try {
    Pop-Location -ErrorAction SilentlyContinue
  } catch {
    # already popped
  }

  $payload = [ordered]@{
    passed = $overallPassed
    timestamp_utc = (Get-Date).ToUniversalTime().ToString("o")
    base_url = $BaseUrl
    gates = $gateStatuses
  }
  if ($gateDetails.Count -gt 0) {
    $payload["details"] = $gateDetails
  }
  if ($failureMessage) {
    $payload["failure_message"] = $failureMessage
  }

  $payloadJson = $payload | ConvertTo-Json -Depth 6
  $payloadDir = Split-Path $gatesOutFile -Parent
  if (-not (Test-Path $payloadDir)) {
    New-Item -ItemType Directory -Path $payloadDir -Force | Out-Null
  }
  Set-Content -Path $gatesOutFile -Value $payloadJson -Encoding UTF8
  Info "Gate results written to $gatesOutFile"
}

if (-not $overallPassed) {
  exit 1
}
exit 0
