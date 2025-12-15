# capture_evidence.ps1
$ErrorActionPreference = "Stop"

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

$result = @{
  schema_version    = 1
  captured_at_utc   = $now
  host              = $env:COMPUTERNAME
  ollama            = @{
    reachable = $false
    models    = @()
    env       = @{
      OLLAMA_HOST         = $env:OLLAMA_HOST
      OLLAMA_KEEP_ALIVE   = $env:OLLAMA_KEEP_ALIVE
      OLLAMA_NUM_PARALLEL = $env:OLLAMA_NUM_PARALLEL
    }
  }
  gpu               = @{}
  application_layer = @{
    npm_project_detected = $false
    lockfile_present     = $false
    lockfile_sha256      = "N/A"
  }
}

# 1. Capture GPU State
try {
  $gpuCsv = nvidia-smi --query-gpu=name, memory.total, memory.used --format=csv, nounits, noheader
  if ($gpuCsv) {
    $p = $gpuCsv -split ","
    $result.gpu = @{ name = $p[0].Trim(); total_mb = [int]$p[1]; used_mb = [int]$p[2] }
  }
}
catch { Write-Warning "NVIDIA-SMI not found or failed." }

# 2. Capture Ollama Models & Digests
try {
  $models = ollama list
  $lines = $models -split "`n" | Select-Object -Skip 1
  foreach ($line in $lines) {
    if (-not $line.Trim()) { continue }
    # Parse "NAME ID SIZE MODIFIED"
    $parts = $line -split "\s{2,}"
    $name = $parts[0]
    $id = $parts[1]
    
    # Capture Modelfile Hash for extra integrity
    $modelfile = ollama show $name --modelfile
    $mfStream = [System.IO.MemoryStream]::new([System.Text.Encoding]::UTF8.GetBytes(($modelfile -join "`n")))
    $mfHash = (Get-FileHash -InputStream $mfStream -Algorithm SHA256).Hash.ToLower()

    $result.ollama.models += @{
      name             = $name
      digest_short     = $id
      modelfile_sha256 = $mfHash
    }
  }
  $result.ollama.reachable = $true
}
catch { 
  Write-Warning "Ollama unreachable."
  $result.ollama.reachable = $false 
}

# 3. Capture Application Layer (NPM Supply Chain)
$npmFile = Join-Path $repoRoot "package.json"
$lockFile = Join-Path $repoRoot "package-lock.json"

if (Test-Path $npmFile) {
  if (Test-Path $lockFile) {
    $result.application_layer.npm_project_detected = $true
    $result.application_layer.lockfile_present = $true
    $result.application_layer.lockfile_sha256 = (Get-FileHash -Path $lockFile -Algorithm SHA256).Hash.ToLower()
  }
  else {
    Write-Warning "CRITICAL: package.json found but package-lock.json MISSING. Supply chain is OPEN."
    $result.application_layer.npm_project_detected = $true
    $result.application_layer.lockfile_present = $false
    $result.application_layer.lockfile_sha256 = "MISSING_CRITICAL"
  }
}
else {
  # If not in root, try to find deeper project
  $deepPkg = Get-ChildItem -Path $repoRoot -Filter "package.json" -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch "\\\\node_modules\\\\" } |
    Select-Object -First 1
  if ($deepPkg) {
    $result.application_layer.npm_project_detected = $true
    $lockPath = Join-Path $deepPkg.DirectoryName "package-lock.json"
    if (Test-Path $lockPath) {
      $result.application_layer.lockfile_present = $true
      $result.application_layer.lockfile_sha256 = (Get-FileHash -Path $lockPath -Algorithm SHA256).Hash.ToLower()
    }
    else {
      $result.application_layer.lockfile_present = $false
      $result.application_layer.lockfile_sha256 = "MISSING_CRITICAL_DEEP"
    }
  }
}

$result | ConvertTo-Json -Depth 5 | Set-Content -Path $outFile -Encoding UTF8
Write-Host "Evidence captured to: $outFile" -ForegroundColor Green
