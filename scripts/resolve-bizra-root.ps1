# Resolve BIZRA repo root and apply the Workspace Contract (.bizra/workspace.yaml).
# Sets process env vars for all downstream tooling (PowerShell, Node, Docker, Rust).

[CmdletBinding()]
param(
  # Optional explicit workspace file path (overrides BIZRA_WORKSPACE env var).
  [string]$WorkspaceFile,
  # If set, do not create a workspace contract when missing.
  [switch]$NoCreate
)

$ErrorActionPreference = "Stop"

function Get-GitRoot {
  param([string]$StartDir)

  if (-not $StartDir) { $StartDir = $PSScriptRoot }

  try {
    $root = (git -C $StartDir rev-parse --show-toplevel 2>$null)
    if ($root) { return ($root.Trim() -replace '/', '\') }
  } catch {}

  $here = (Resolve-Path $StartDir).Path
  while ($true) {
    if (Test-Path (Join-Path $here ".git")) { return $here }
    $parent = Split-Path $here -Parent
    if (-not $parent -or $parent -eq $here) { return $null }
    $here = (Resolve-Path $parent).Path
  }
}

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

$repoRoot = Get-GitRoot -StartDir $PSScriptRoot
if (-not $repoRoot) {
  $repoRoot = $PSScriptRoot
  if ((Split-Path $repoRoot -Leaf) -ieq "scripts") {
    $repoRoot = Split-Path $repoRoot -Parent
  }
}
$env:BIZRA_REPO_ROOT = $repoRoot

if (-not $WorkspaceFile) { $WorkspaceFile = $env:BIZRA_WORKSPACE }
if (-not $WorkspaceFile) { $WorkspaceFile = (Join-Path $repoRoot ".bizra\\workspace.yaml") }

if (-not (Test-Path $WorkspaceFile)) {
  if ($NoCreate) { throw "Workspace contract not found: $WorkspaceFile" }

  $wsDir = Split-Path $WorkspaceFile -Parent
  New-Item -ItemType Directory -Force -Path $wsDir | Out-Null

  @"
workspace_version: 1
workspace_id: local_only

paths:
  bizra_root: '$repoRoot'
  infra_root: '$repoRoot'
  kernel_root: '$repoRoot\\.bizra-kernel'
  evidence_root: '$repoRoot\\evidence'

runtime:
  ollama_host: 'http://127.0.0.1:11434'
  # SECURITY: do not commit DB passwords; use passwordless local auth or a secret manager/.pgpass.
  postgres_dsn: 'postgresql://postgres@localhost:5432/bizra'
  redis_host: 'localhost'
  redis_port: 6379
"@ | Set-Content -Path $WorkspaceFile -Encoding UTF8
}

$env:BIZRA_WORKSPACE_FILE = $WorkspaceFile
$ws = ConvertFrom-SimpleYaml -YamlText (Get-Content -Raw $WorkspaceFile)

function Get-Deep([hashtable]$map, [string[]]$path) {
  $cur = $map
  foreach ($p in $path) {
    if (-not ($cur -is [hashtable])) { return $null }
    if (-not $cur.ContainsKey($p)) { return $null }
    $cur = $cur[$p]
  }
  return $cur
}

$bizraRoot = (Get-Deep $ws @("paths","bizra_root"))
$infraRoot = (Get-Deep $ws @("paths","infra_root"))
$kernelRoot = (Get-Deep $ws @("paths","kernel_root"))
$evidenceRoot = (Get-Deep $ws @("paths","evidence_root"))

if (-not $bizraRoot) { $bizraRoot = $repoRoot }
if (-not $infraRoot) { $infraRoot = $bizraRoot }
if (-not $kernelRoot) { $kernelRoot = (Join-Path $bizraRoot ".bizra-kernel") }
if (-not $evidenceRoot) { $evidenceRoot = (Join-Path $bizraRoot "evidence") }

$ollamaHost = (Get-Deep $ws @("runtime","ollama_host"))
$postgresDsn = (Get-Deep $ws @("runtime","postgres_dsn"))
$redisHost = (Get-Deep $ws @("runtime","redis_host"))
$redisPort = (Get-Deep $ws @("runtime","redis_port"))
$composeProject = (Get-Deep $ws @("docker","compose_project_name"))

if ($ollamaHost) { $env:OLLAMA_HOST = $ollamaHost }
if ($postgresDsn) { $env:POSTGRES_DSN = $postgresDsn }
if ($redisHost) { $env:REDIS_HOST = $redisHost }
if ($redisPort) { $env:REDIS_PORT = "$redisPort" }
if ($composeProject) { $env:COMPOSE_PROJECT_NAME = $composeProject }

$env:BIZRA_ROOT = $bizraRoot
$env:INFRA_ROOT = $infraRoot
$env:KERNEL_ROOT = $kernelRoot
$env:EVIDENCE_ROOT = $evidenceRoot
$env:EVIDENCE_DIR = $evidenceRoot

Write-Host "BIZRA_REPO_ROOT=$env:BIZRA_REPO_ROOT"
Write-Host "BIZRA_ROOT=$env:BIZRA_ROOT"
Write-Host "KERNEL_ROOT=$env:KERNEL_ROOT"
Write-Host "INFRA_ROOT=$env:INFRA_ROOT"
Write-Host "EVIDENCE_ROOT=$env:EVIDENCE_ROOT"
if ($env:COMPOSE_PROJECT_NAME) { Write-Host "COMPOSE_PROJECT_NAME=$env:COMPOSE_PROJECT_NAME" }
if ($env:OLLAMA_HOST) { Write-Host "OLLAMA_HOST=$env:OLLAMA_HOST" }

foreach ($p in @(
  @{ name = "BIZRA_ROOT"; value = $env:BIZRA_ROOT },
  @{ name = "KERNEL_ROOT"; value = $env:KERNEL_ROOT },
  @{ name = "INFRA_ROOT"; value = $env:INFRA_ROOT },
  @{ name = "EVIDENCE_ROOT"; value = $env:EVIDENCE_ROOT }
)) {
  if ($p.value -and -not (Test-Path $p.value)) {
    Write-Warning "$($p.name) points to missing path: $($p.value)"
  }
}
