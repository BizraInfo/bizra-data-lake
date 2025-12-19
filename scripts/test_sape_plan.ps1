param(
  [string]$BaseUrl = "http://127.0.0.1:8010",
  [string]$EnvFile = ".env"
)

$ErrorActionPreference = "Stop"

function Fail($msg) {
  Write-Host "[FAIL] $msg" -ForegroundColor Red
  exit 1
}

function Info($msg) {
  Write-Host "[INFO] $msg" -ForegroundColor Cyan
}

function Ok($msg) {
  Write-Host "[OK]  $msg" -ForegroundColor Green
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

$token = $env:BIZRA_API_TOKEN
if (-not $token) {
  $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
  $envPath = $EnvFile
  if (-not [System.IO.Path]::IsPathRooted($EnvFile)) {
    $envPath = Join-Path $repoRoot $EnvFile
  }
  $token = Get-EnvValue $envPath "BIZRA_API_TOKEN"
}
if (-not $token) {
  Fail "Missing BIZRA_API_TOKEN (set env var or add to $EnvFile)."
}

$url = ($BaseUrl.TrimEnd("/") + "/v1/sape/plan")
Info "POST $url"

$payload = @{
  domain = "Economic Sovereignty"
  objective = "Validate Proof of Impact tokenomics model against inflation risks."
  stakes = "H"
  constraints = "No web. Use graph evidence only. Fail-closed if evidence is missing."
  success_criteria = "Identify inflation risk vectors, propose mitigations, cite evidence artifacts."
  require_graph_evidence = $true
  evidence_topics = @("PoI","tokenomics","inflation","supply_cap")
  evidence_limit = 8
} | ConvertTo-Json -Depth 6

try {
  $resp = Invoke-RestMethod -Method Post -Uri $url -Headers @{ "X-BIZRA-TOKEN" = $token } -ContentType "application/json" -Body $payload -TimeoutSec 10
} catch {
  Fail $_.Exception.Message
}

$status = $resp.status
$auditId = $resp.audit_id
$missing = @()
if ($resp.PSObject.Properties.Name -contains "missing_artifacts" -and $resp.missing_artifacts) {
  $missing = @($resp.missing_artifacts | ForEach-Object { $_ })
}
$promptSha = $resp.prompt_sha256

Write-Host "status: $status"
if ($auditId) {
  Write-Host "audit_id: $auditId"
}
if ($missing.Count -gt 0) {
  Write-Host "missing_artifacts: $($missing -join ',')"
}
if ($promptSha) {
  Write-Host "prompt_sha256: $promptSha"
}

$result = [PSCustomObject]@{
  status = $status
  audit_id = $auditId
  missing_artifacts = $missing
  prompt_sha256 = $promptSha
  slot = $resp.slot
}
$resultJson = $result | ConvertTo-Json -Depth 6 -Compress
Write-Output $resultJson

if ($status -eq "PLANNED" -or $status -eq "BLOCKED_BY_EVIDENCE") {
  exit 0
}
exit 2
