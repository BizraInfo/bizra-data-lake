# scripts/genesis_ignite_all.ps1
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

$repoRoot = (Resolve-Path (Join