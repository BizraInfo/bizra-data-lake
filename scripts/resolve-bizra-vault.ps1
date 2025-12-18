# Resolve BIZRA Vault contract (.bizra/vault.yaml) and export process env vars.
# Usage:
#   .\scripts\resolve-bizra-vault.ps1
#   . .\scripts\resolve-bizra-vault.ps1  # (dot-source) to keep env vars in current session

[CmdletBinding()]
param(
  [string]$VaultFile
)

$ErrorActionPreference = "Stop"

function ConvertFrom-SimpleYaml {
  param([Parameter(Mandatory = $true)][string]$YamlText)

  function Parse-Scalar([string]$raw) {
    $v = $raw.Trim()
    if ($v.StartsWith("'") -and $v.EndsWith("'") -and $v.Length -ge 2) { $v = $v.Substring(1, $v.Length - 2) }
    elseif ($v.StartsWith('"') -and $v.EndsWith('"') -and $v.Length -ge 2) { $v = $v.Substring(1, $v.Length - 2) }

    if ($v -match '^(?i:true|false)$') { return [bool]::Parse($v) }
    if ($v -match '^-?\d+$') { return [int]$v }
    return $v
  }

  $root = @{}
  $stack = New-Object System.Collections.Generic.List[object]
  $stack.Add(@{ indent = -1; map = $root }) | Out-Null

  foreach ($line in ($YamlText -split "`r?`n")) {
    if (-not $line) { continue }
    $trimmed = $line.Trim()
    if (-not $trimmed -or $trimmed.StartsWith("#")) { continue }

    $indent = $line.Length - $line.TrimStart().Length

    if ($trimmed -notmatch '^([A-Za-z0-9_]+)\s*:\s*(.*)$') { continue }
    $key = $matches[1]
    $valueRaw = $matches[2]

    while ($stack.Count -gt 0 -and $stack[$stack.Count - 1].indent -ge $indent) {
      $stack.RemoveAt($stack.Count - 1)
    }
    if ($stack.Count -eq 0) { throw "Invalid YAML indentation near: $trimmed" }

    $parent = $stack[$stack.Count - 1].map
    if ($valueRaw -eq '') {
      $child = @{}
      $parent[$key] = $child
      $stack.Add(@{ indent = $indent; map = $child }) | Out-Null
    } else {
      $parent[$key] = Parse-Scalar $valueRaw
    }
  }

  return $root
}

function Get-Deep([hashtable]$map, [string[]]$path) {
  $cur = $map
  foreach ($p in $path) {
    if (-not ($cur -is [hashtable])) { return $null }
    if (-not $cur.ContainsKey($p)) { return $null }
    $cur = $cur[$p]
  }
  return $cur
}

$repoRoot = & "$PSScriptRoot\\bizra-root.ps1"
$env:BIZRA_REPO_ROOT = $repoRoot

if (-not $VaultFile) { $VaultFile = $env:BIZRA_VAULT_FILE }
if (-not $VaultFile) { $VaultFile = (Join-Path $repoRoot ".bizra\\vault.yaml") }
$env:BIZRA_VAULT_FILE = $VaultFile

if (-not (Test-Path $VaultFile)) {
  throw "Vault contract not found: $VaultFile"
}

$vault = ConvertFrom-SimpleYaml -YamlText (Get-Content -Raw $VaultFile)

$nodeId = (Get-Deep $vault @("node_id"))
$dataLake = (Get-Deep $vault @("data_lake_root"))
$roots = (Get-Deep $vault @("roots"))

if ($nodeId) { $env:BIZRA_NODE_ID = $nodeId }
if ($dataLake) { $env:BIZRA_DATA_LAKE_ROOT = $dataLake }

if ($env:BIZRA_DATA_LAKE_ROOT) {
  $env:BIZRA_DATALAKE_INTAKE = Join-Path $env:BIZRA_DATA_LAKE_ROOT "00_INTAKE"
  $env:BIZRA_DATALAKE_RAW = Join-Path $env:BIZRA_DATA_LAKE_ROOT "01_RAW"
  $env:BIZRA_DATALAKE_PROCESSED = Join-Path $env:BIZRA_DATA_LAKE_ROOT "02_PROCESSED"
  $env:BIZRA_DATALAKE_INDEXED = Join-Path $env:BIZRA_DATA_LAKE_ROOT "03_INDEXED"
  $env:BIZRA_DATALAKE_GOLD = Join-Path $env:BIZRA_DATA_LAKE_ROOT "04_GOLD"
  $env:BIZRA_DATALAKE_QUARANTINE = Join-Path $env:BIZRA_DATA_LAKE_ROOT "99_QUARANTINE"
}

if ($roots -is [hashtable]) {
  foreach ($k in $roots.Keys) {
    $v = $roots[$k]
    if (-not $v) { continue }
    $name = ("BIZRA_VAULT_" + $k).ToUpperInvariant()
    Set-Item -Path ("Env:" + $name) -Value "$v"
  }
}

Write-Host "BIZRA_NODE_ID=$env:BIZRA_NODE_ID"
Write-Host "BIZRA_DATA_LAKE_ROOT=$env:BIZRA_DATA_LAKE_ROOT"
Write-Host "BIZRA_DATALAKE_INDEXED=$env:BIZRA_DATALAKE_INDEXED"
Write-Host "BIZRA_VAULT_FILE=$env:BIZRA_VAULT_FILE"

foreach ($p in @(
  @{ name = "BIZRA_DATA_LAKE_ROOT"; value = $env:BIZRA_DATA_LAKE_ROOT },
  @{ name = "BIZRA_DATALAKE_INDEXED"; value = $env:BIZRA_DATALAKE_INDEXED },
  @{ name = "BIZRA_VAULT_DUAL_AGENTIC_REPO"; value = $env:BIZRA_VAULT_DUAL_AGENTIC_REPO },
  @{ name = "BIZRA_VAULT_TASKMASTER_REPO"; value = $env:BIZRA_VAULT_TASKMASTER_REPO },
  @{ name = "BIZRA_VAULT_GENESIS_NODE_REPO"; value = $env:BIZRA_VAULT_GENESIS_NODE_REPO },
  @{ name = "BIZRA_VAULT_CRASH_TEMP"; value = $env:BIZRA_VAULT_CRASH_TEMP },
  @{ name = "BIZRA_VAULT_CRASH_TMP"; value = $env:BIZRA_VAULT_CRASH_TMP }
)) {
  if ($p.value -and -not (Test-Path $p.value)) {
    Write-Warning "$($p.name) points to missing path: $($p.value)"
  }
}
