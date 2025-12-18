# capture_evidence.ps1
# Captures: host/GPU, Ollama digests + modelfile hashes, LM Studio model list, and app-layer lockfiles.

$ErrorActionPreference = "Stop"

function Sha256File([string]$path) {
  (Get-FileHash -Algorithm SHA256 -Path $path).Hash.ToLower()
}

function Sha256Text([string]$text) {
  $bytes = [System.Text.Encoding]::UTF8.GetBytes($text)
  $ms = [System.IO.MemoryStream]::new($bytes)
  (Get-FileHash -InputStream $ms -Algorithm SHA256).Hash.ToLower()
}

function CanonicalizeOllamaModelfile([string]$text) {
  $lines = $text -split "`r?`n"
  $out = @()
  foreach ($line in $lines) {
    $trim = $line.TrimEnd()
    if ($trim -match '^\s*FROM\s+.*sha256-([a-f0-9]{64})\s*$') {
      $blob = $matches[1].ToLower()
      $out += "FROM sha256:$blob"
      continue
    }
    if ($trim -match '^\s*FROM\s+sha256:([a-f0-9]{64})\s*$') {
      $blob = $matches[1].ToLower()
      $out += "FROM sha256:$blob"
      continue
    }
    $out += $trim
  }
  return ($out -join "`n")
}

# Resolve workspace/repo roots (works even if invoked from a subfolder)
$repoRoot = $PSScriptRoot
$resolver = Join-Path $repoRoot "scripts\\resolve-bizra-root.ps1"
if (Test-Path $resolver) {
  try { & $resolver | Out-Null } catch { Write-Warning "Workspace resolver failed: $($_.Exception.Message)" }
}
if ($env:BIZRA_REPO_ROOT) { $repoRoot = $env:BIZRA_REPO_ROOT }

$outDir = Join-Path $repoRoot "evidence"
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Force -Path $outDir | Out-Null }
$outFile = Join-Path $outDir "audit-results-node0.json"

$now = (Get-Date).ToUniversalTime().ToString("o")

$result = [ordered]@{
  schema_version  = 2
  captured_at_utc = $now
  host            = [ordered]@{
    machine = $env:COMPUTERNAME
    cpu     = (Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)
    os      = (Get-CimInstance Win32_OperatingSystem | Select-Object -ExpandProperty Caption)
    ram_gb  = [math]::Round(((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB), 1)
    gpu     = $null
  }
  application_layer = [ordered]@{
    npm_project_detected = $false
    lockfile_present     = $false
    lockfile_sha256      = "N/A"
  }
  ollama = [ordered]@{
    reachable = $false
    host      = $null
    models    = @()
  }
  lm_studio = [ordered]@{
    reachable = $false
    base_url  = $null
    models    = @()
  }
}

# 1) GPU State
try {
  $gpuCsv = nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,nounits,noheader
  if ($gpuCsv) {
    $p = ($gpuCsv -split ",") | ForEach-Object { $_.Trim() }
    $result.host.gpu = [ordered]@{
      name      = $p[0]
      total_mb  = [int]$p[1]
      used_mb   = [int]$p[2]
      free_mb   = [int]$p[3]
    }
  }
} catch {
  Write-Warning "nvidia-smi not available (CPU-only mode) or failed."
}

# 2) Ollama tags (preferred)
$ollamaHost = $env:OLLAMA_URL
if (-not $ollamaHost) { $ollamaHost = $env:OLLAMA_HOST }
if (-not $ollamaHost) { $ollamaHost = "http://127.0.0.1:11434" }
$result.ollama.host = $ollamaHost

try {
  $raw = (Invoke-WebRequest -Uri "$ollamaHost/api/tags" -Method GET -TimeoutSec 5 -UseBasicParsing).Content
  $tags = $raw | ConvertFrom-Json
  if ($tags.models) {
    foreach ($m in $tags.models) {
      $name = $m.name
      $digest = $m.digest
      $digestValue = if ($digest) { "sha256:$digest" } else { $null }
      $mfHash = $null
      try {
        $mfText = (ollama show $name --modelfile 2>$null) -join "`n"
        if ($mfText) { $mfHash = (Sha256Text (CanonicalizeOllamaModelfile $mfText)) }
      } catch {}

      $result.ollama.models += [ordered]@{
        name            = $name
        digest          = $digestValue
        size_bytes      = $m.size
        modelfile_sha256 = $mfHash
      }
    }
    $result.ollama.reachable = $true
  }
} catch {
  Write-Warning "Ollama /api/tags unreachable ($($_.Exception.Message)); falling back to 'ollama list' parsing."
  try {
    $models = ollama list
    $lines = $models -split "`n" | Select-Object -Skip 1
    foreach ($line in $lines) {
      if (-not $line.Trim()) { continue }
      $parts = $line -split "\s{2,}" | Where-Object { $_ -ne "" }
      $name = $parts[0]
      $id = $parts[1]
      $mfHash = $null
      try {
        $mfText = (ollama show $name --modelfile 2>$null) -join "`n"
        if ($mfText) { $mfHash = (Sha256Text (CanonicalizeOllamaModelfile $mfText)) }
      } catch {}
      $result.ollama.models += [ordered]@{
        name             = $name
        digest_short     = $id
        modelfile_sha256 = $mfHash
      }
    }
    $result.ollama.reachable = $true
  } catch {
    Write-Warning "Ollama unreachable."
    $result.ollama.reachable = $false
  }
}

# 3) LM Studio models via OpenAI-compatible API
$lmBase = $env:LMSTUDIO_URL
if (-not $lmBase) { $lmBase = "http://127.0.0.1:1234/v1" }
$result.lm_studio.base_url = $lmBase
try {
  $ms = Invoke-RestMethod -Uri "$lmBase/models" -Method GET -TimeoutSec 5
  if ($ms.data) {
    foreach ($m in $ms.data) {
      $result.lm_studio.models += [ordered]@{
        id = $m.id
        object = $m.object
        owned_by = $m.owned_by
      }
    }
    $result.lm_studio.reachable = $true
  }
} catch {
  Write-Warning "LM Studio /v1/models unreachable."
  $result.lm_studio.reachable = $false
}

# 4) Application Layer (NPM Supply Chain)
$npmFile = Join-Path $repoRoot "package.json"
$lockFile = Join-Path $repoRoot "package-lock.json"

if (Test-Path $npmFile) {
  $result.application_layer.npm_project_detected = $true
  if (Test-Path $lockFile) {
    $result.application_layer.lockfile_present = $true
    $result.application_layer.lockfile_sha256 = (Sha256File $lockFile)
  } else {
    Write-Warning "CRITICAL: package.json found but package-lock.json MISSING. Supply chain is OPEN."
    $result.application_layer.lockfile_present = $false
    $result.application_layer.lockfile_sha256 = "MISSING_CRITICAL"
  }
} else {
  $deepPkg = Get-ChildItem -Path $repoRoot -Filter "package.json" -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch "\\\\node_modules\\\\" } |
    Select-Object -First 1
  if ($deepPkg) {
    $result.application_layer.npm_project_detected = $true
    $lockPath = Join-Path $deepPkg.DirectoryName "package-lock.json"
    if (Test-Path $lockPath) {
      $result.application_layer.lockfile_present = $true
      $result.application_layer.lockfile_sha256 = (Sha256File $lockPath)
    } else {
      Write-Warning "CRITICAL: package.json found but package-lock.json MISSING. Supply chain is OPEN."
      $result.application_layer.lockfile_present = $false
      $result.application_layer.lockfile_sha256 = "MISSING_CRITICAL_DEEP"
    }
  }
}

$result | ConvertTo-Json -Depth 10 | Set-Content -Path $outFile -Encoding UTF8
Write-Host "Evidence captured to: $outFile" -ForegroundColor Green
