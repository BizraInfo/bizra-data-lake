param(
  [string]$ComposeFile = "docker-compose.yml",
  [string]$EnvFile = ".env",
  [string]$BaseUrl = "http://127.0.0.1:8010",
  [string]$Neo4jHttp = "http://127.0.0.1:7474",
  [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

function Write-Stamp($message) {
  $ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd HH:mm:ss 'UTC'")
  Write-Host "[$ts] $message" -ForegroundColor Cyan
}

function Write-Ok($message) {
  Write-Host "[OK]  $message" -ForegroundColor Green
}

function Print-Diagnostics($composeName) {
  Write-Stamp "docker compose ps"
  docker compose -f $composeName ps
  Write-Stamp "Last 200 kernel logs"
  docker compose -f $composeName logs --tail 200 kernel
}

function Read-EnvFile($path) {
  $map = @{}
  if (-not (Test-Path $path)) { return $map }
  $lines = Get-Content $path -ErrorAction SilentlyContinue
  foreach ($raw in $lines) {
    $line = ($raw -as [string]).Trim()
    if (-not $line -or $line.StartsWith('#')) { continue }
    if ($line -notmatch '^(?<k>[^=]+)=(?<v>.*)$') { continue }
    $key = $Matches['k'].Trim()
    $val = $Matches['v']
    $map[$key] = $val
  }
  return $map
}

function Parse-Neo4jAuth($raw) {
  if (-not $raw) { return $null }
  $parts = $raw -split '[:/]', 2, [System.StringSplitOptions]::RemoveEmptyEntries
  if ($parts.Count -ne 2) { return $null }
  return @{ user = $parts[0]; pass = $parts[1] }
}

function Get-Neo4jCounts($httpUrl, $auth) {
  $result = @{ nodes = $null; relationships = $null; error = $null }
  $creds = Parse-Neo4jAuth $auth
  if (-not $creds) {
    $result.error = "NEO4J_AUTH missing or invalid"
    return $result
  }
  $payload = @{ statements = @(@{ statement = "MATCH (n) RETURN count(n) AS c" }, @{ statement = "MATCH ()-[r]->() RETURN count(r) AS c" }) } | ConvertTo-Json -Depth 4
  $authHeader = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes("$($creds.user):$($creds.pass)"))
  $uri = ($httpUrl.TrimEnd('/') + "/db/neo4j/tx/commit")
  try {
    $resp = Invoke-RestMethod -Method Post -Uri $uri -Headers @{ Authorization = "Basic $authHeader" } -Body $payload -ContentType "application/json" -TimeoutSec 5
  } catch {
    $result.error = $_.Exception.Message
    return $result
  }
  try {
    $rows = $resp.results[0].data
    if ($rows -and $rows[0].row) { $result.nodes = [int]$rows[0].row[0] }
    $rows2 = $resp.results[1].data
    if ($rows2 -and $rows2[0].row) { $result.relationships = [int]$rows2[0].row[0] }
  } catch {
    $result.error = "unexpected neo4j response"
  }
  return $result
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  throw "docker not found on PATH."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$composePath = (Resolve-Path (Join-Path $repoRoot $ComposeFile)).Path
$composeDir = Split-Path $composePath -Parent
$composeName = Split-Path $composePath -Leaf
$igniteScript = Join-Path $repoRoot "scripts/ignite_node0.ps1"
$verifyScript = Join-Path $repoRoot "scripts/verify_node0.ps1"
$planScript = Join-Path $repoRoot "scripts/test_sape_plan.ps1"
$receiptScript = Join-Path $repoRoot "scripts/genesis_receipt.py"
$seedScript = Join-Path $repoRoot "scripts/seed_min_evidence.py"
$loaderScript = Join-Path $repoRoot "bizra_synaptic_loader.py"
$envPath = if ([System.IO.Path]::IsPathRooted($EnvFile)) { $EnvFile } else { Join-Path $composeDir $EnvFile }
$envMap = Read-EnvFile $envPath
$neo4jAuth = $env:NEO4J_AUTH
if (-not $neo4jAuth -and $envMap.ContainsKey("NEO4J_AUTH")) {
  $neo4jAuth = $envMap["NEO4J_AUTH"].Trim()
}

Push-Location $composeDir
try {
  Write-Stamp "Genesis ignition start"

  Write-Stamp "ACT 0: Safe clean (docker compose down --remove-orphans)"
  docker compose -f $composeName down --remove-orphans
  Write-Ok "Citadel reset complete."

  Write-Stamp "ACT I: Ignite stack"
  & pwsh -File $igniteScript -ComposeFile $ComposeFile
  $igniteExit = $LASTEXITCODE
  if ($igniteExit -ne 0) {
    Write-Host "[FAIL] ignite_node0.ps1 exited with code $igniteExit" -ForegroundColor Red
    Print-Diagnostics $composeName
    exit 1
  }
  Write-Ok "Ignition completed."

  Write-Stamp "ACT II: Acceptance gates"
  & pwsh -File $verifyScript -ComposeFile $ComposeFile -BaseUrl $BaseUrl -EnvFile $EnvFile
  $verifyExit = $LASTEXITCODE
  if ($verifyExit -ne 0) {
    Write-Host "[FAIL] verify_node0.ps1 exited with code $verifyExit" -ForegroundColor Red
    Print-Diagnostics $composeName
    exit 2
  }
  Write-Ok "All gates passed."

  Write-Stamp "ACT III: First SAPE thought"
  $planOutput = & pwsh -File $planScript -BaseUrl $BaseUrl -EnvFile $EnvFile
  $planExit = $LASTEXITCODE
  if ($planExit -ne 0) {
    Write-Host "[FAIL] test_sape_plan.ps1 exited with code $planExit" -ForegroundColor Red
    exit 2
  }
  $planJson = $planOutput | Select-Object -Last 1
  try {
    $planResult = $planJson | ConvertFrom-Json -ErrorAction Stop
  } catch {
    Write-Host "[FAIL] Unable to parse SAPE plan output: $planJson" -ForegroundColor Red
    exit 2
  }

  $planStatus = $planResult.status
  Write-Host "SAPE status: $planStatus"
  if ($planResult.audit_id) {
    Write-Host "SAPE audit_id: $($planResult.audit_id)"
  }
  if ($planResult.prompt_sha256) {
    Write-Host "SAPE prompt_sha256: $($planResult.prompt_sha256)"
  }

  if ($planStatus -eq "BLOCKED_BY_EVIDENCE") {
    Write-Stamp "Evidence branch activated (seeding minimal evidence)"
    if (Test-Path $seedScript) {
      & $Python $seedScript --env-file $EnvFile --neo4j-http $Neo4jHttp
      $seedExit = $LASTEXITCODE
      if ($seedExit -ne 0) {
        Write-Host "[FAIL] seed_min_evidence.py exited with code $seedExit" -ForegroundColor Red
        exit 3
      }
    } else {
      & $Python $loaderScript
      $seedExit = $LASTEXITCODE
      if ($seedExit -ne 0) {
        Write-Host "[FAIL] bizra_synaptic_loader.py exited with code $seedExit" -ForegroundColor Red
        exit 3
      }
    }
    Start-Sleep -Seconds 2
    $planOutput = & pwsh -File $planScript -BaseUrl $BaseUrl -EnvFile $EnvFile
    $planExit = $LASTEXITCODE
    if ($planExit -ne 0) {
      Write-Host "[FAIL] test_sape_plan.ps1 (after seed) exited with code $planExit" -ForegroundColor Red
      exit 3
    }
    $planJson = $planOutput | Select-Object -Last 1
    try {
      $planResult = $planJson | ConvertFrom-Json -ErrorAction Stop
    } catch {
      Write-Host "[FAIL] Unable to parse SAPE plan output after seeding: $planJson" -ForegroundColor Red
      exit 3
    }
    $planStatus = $planResult.status
    Write-Host "SAPE status after seeding: $planStatus"
    if ($planStatus -ne "PLANNED") {
      $counts = Get-Neo4jCounts $Neo4jHttp $neo4jAuth
      Write-Host "Neo4j snapshot (nodes=$($counts.nodes), relationships=$($counts.relationships), error=$($counts.error))"
      exit 3
    }
  }
  elseif ($planStatus -ne "PLANNED") {
    Write-Host "[FAIL] Unexpected SAPE status: $planStatus" -ForegroundColor Red
    exit 2
  }
  Write-Ok "SAPE plan ready."

  Write-Stamp "ACT IV: Receipt minting"
  $receiptTs = (Get-Date).ToUniversalTime().ToString("yyyyMMdd_HHmmssZ")
  $receiptRelPath = "docs/evidence/receipts/genesis_receipt_v1_${receiptTs}.json"
  & $Python $receiptScript --repo-root $repoRoot --env-file $EnvFile --compose-file $ComposeFile --healthz-url ($BaseUrl.TrimEnd('/') + "/healthz") --sape-url ($BaseUrl.TrimEnd('/') + "/v1/sape/plan") --neo4j-http $Neo4jHttp --out $receiptRelPath
  $receiptExit = $LASTEXITCODE
  if ($receiptExit -ne 0) {
    Write-Host "[FAIL] genesis_receipt.py exited with code $receiptExit" -ForegroundColor Red
    exit 4
  }
  $receiptPath = Join-Path $repoRoot $receiptRelPath
  $receiptJson = Get-Content -Path $receiptPath -Raw | ConvertFrom-Json
  Write-Host "Receipt path: $receiptPath"
  Write-Host "Receipt sha256: $($receiptJson.receipt_sha256)"
  Write-Host "Git commit sha: $($receiptJson.git_commit_sha)"
  Write-Ok "Genesis ignition complete."
} finally {
  Pop-Location
}
