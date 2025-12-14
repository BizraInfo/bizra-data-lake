# capture_evidence.ps1
# Runs local model + GPU audit and writes evidence/audit-results-node0.json

$ErrorActionPreference = "Stop"
$outDir = ".\evidence"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$outFile = Join-Path $outDir "audit-results-node0.json"

$now = (Get-Date).ToUniversalTime().ToString("o")

$result = @{
  schema_version  = 1
  captured_at_utc = $now
  host            = @{
    os      = (Get-CimInstance Win32_OperatingSystem).Caption
    machine = $env:COMPUTERNAME
    cpu     = (Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)
    ram_gb  = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
    gpu     = @{}
  }
  ollama          = @{
    reachable  = $false
    host       = "http://localhost:11434"
    env        = @{
      OLLAMA_HOST              = $env:OLLAMA_HOST
      OLLAMA_KEEP_ALIVE        = $env:OLLAMA_KEEP_ALIVE
      OLLAMA_MAX_LOADED_MODELS = $env:OLLAMA_MAX_LOADED_MODELS
      OLLAMA_NUM_PARALLEL      = $env:OLLAMA_NUM_PARALLEL
    }
    models     = @()
    smoke_test = @{
      prompt     = "What is the SHA-256 output size in bits? Return ONLY the number."
      response   = $null
      ok         = $false
      latency_ms = $null
    }
  }
  llm_studio      = @{ models = @() }
}

# GPU info (nvidia-smi optional)
try {
  $gpuCsv = nvidia-smi --query-gpu=name, memory.total, memory.used, memory.free --format=csv, nounits | Select-Object -Skip 1
  if ($gpuCsv) {
    $parts = $gpuCsv -split "," | ForEach-Object { $_.Trim() }
    $result.host.gpu = @{
      name          = $parts[0]
      vram_total_mb = [int]$parts[1]
      vram_used_mb  = [int]$parts[2]
      vram_free_mb  = [int]$parts[3]
    }
  }
}
catch { }

# Ollama model list + modelfiles (captures digests if present)
try {
  $list = ollama list
  $lines = $list | Select-Object -Skip 1
  foreach ($line in $lines) {
    if (-not $line.Trim()) { continue }
    $cols = ($line -split "\s{2,}")
    $name = $cols[0]
    if (-not $name) { continue }

    $modelfile = ollama show $name --modelfile
    $tmp = New-TemporaryFile
    Set-Content -Path $tmp.FullName -Value ($modelfile -join "`n") -Encoding UTF8
    $mfHash = (Get-FileHash -Algorithm SHA256 -Path $tmp.FullName).Hash.ToLower()

    $digest = ($modelfile | Select-String -Pattern "sha256:" -AllMatches | Select-Object -First 1).ToString()
    if ($digest) {
      $digest = ($digest | Select-String -Pattern "sha256:[0-9a-f]{64}" -AllMatches).Matches.Value | Select-Object -First 1
    }

    $result.ollama.models += @{
      name             = $name
      digest           = $digest
      modelfile_sha256 = $mfHash
    }
  }
  $result.ollama.reachable = $true
}
catch {
  $result.ollama.reachable = $false
}

# Smoke test via Ollama HTTP API
try {
  $body = @{
    model  = "deepseek-8b-instruct"
    prompt = "What is the SHA-256 output size in bits? Return ONLY the number."
    stream = $false
  } | ConvertTo-Json

  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $resp = Invoke-RestMethod -Uri "http://localhost:11434/api/generate" -Method POST -Body $body -ContentType "application/json" -TimeoutSec 15
  $sw.Stop()

  $result.ollama.smoke_test.response = $resp.response
  $result.ollama.smoke_test.latency_ms = [int]$sw.ElapsedMilliseconds
  $result.ollama.smoke_test.ok = ($resp.response -match "^256\b")
}
catch { }

# 3. Capture Application Layer (NPM Supply Chain)
$npmFile = "package.json"
$lockFile = "package-lock.json"

if (Test-Path $npmFile) {
  $result.application_layer.npm_project_detected = $true
  if (Test-Path $lockFile) {
    $result.application_layer.lockfile_present = $true
    $result.application_layer.lockfile_sha256 = (Get-FileHash -Path $lockFile -Algorithm SHA256).Hash.ToLower()
  }
  else {
    Write-Warning "CRITICAL: package.json found but package-lock.json MISSING. Supply chain is OPEN."
    $result.application_layer.lockfile_present = $false
    $result.application_layer.lockfile_sha256 = "MISSING_CRITICAL"
  }
}

$result | ConvertTo-Json -Depth 5 | Set-Content -Path $outFile -Encoding UTF8
Write-Host "Evidence captured to: $outFile" -ForegroundColor Green
