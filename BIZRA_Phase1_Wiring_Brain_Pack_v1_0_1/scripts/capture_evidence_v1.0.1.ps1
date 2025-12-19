# capture_evidence_v1.0.1.ps1
# Captures: GPU, Ollama model digests + modelfile hashes, LM Studio model list
# Output: .\evidence\audit-results-node0.json
$ErrorActionPreference = "Stop"
$outDir = ".\evidence"
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Force -Path $outDir | Out-Null }
$outFile = Join-Path $outDir "audit-results-node0.json"
$now = (Get-Date).ToUniversalTime().ToString("o")

$result = [ordered]@{
  schema_version = 1
  captured_at_utc = $now
  host = $env:COMPUTERNAME
  gpu = $null
  ollama = [ordered]@{
    reachable = $false
    base_url = ($env:OLLAMA_HOST); if (-not $env:OLLAMA_HOST) { $env:OLLAMA_HOST = "http://localhost:11434" }
    models = @()
    env = [ordered]@{
      OLLAMA_HOST = $env:OLLAMA_HOST
      OLLAMA_KEEP_ALIVE = $env:OLLAMA_KEEP_ALIVE
      OLLAMA_NUM_PARALLEL = $env:OLLAMA_NUM_PARALLEL
      OLLAMA_MAX_LOADED_MODELS = $env:OLLAMA_MAX_LOADED_MODELS
    }
  }
  lm_studio = [ordered]@{
    reachable = $false
    base_url = "http://localhost:1234/v1"
    models = @()
  }
}

# ---- GPU ----
try {
  $gpuCsv = nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,nounits,noheader
  if ($gpuCsv) {
    $p = ($gpuCsv -split ",") | ForEach-Object { $_.Trim() }
    $result.gpu = [ordered]@{ name=$p[0]; total_mb=[int]$p[1]; used_mb=[int]$p[2]; free_mb=[int]$p[3] }
  }
} catch {
  Write-Warning "nvidia-smi not available (CPU-only mode) or failed."
}

# ---- Ollama digests via REST tags (preferred) ----
try {
  $tags = Invoke-RestMethod -Uri "$($env:OLLAMA_HOST)/api/tags" -Method GET -TimeoutSec 5
  if ($tags.models) {
    foreach ($m in $tags.models) {
      # m.name, m.digest (may exist), m.model, m.size
      $name = $m.name
      $digest = $m.digest
      if (-not $digest -and $m.model) { $digest = $m.model } # some versions use 'model' as digest-like
      # Modelfile hash (best-effort)
      $mfHash = $null
      try {
        $modelfile = ollama show $name --modelfile 2>$null
        if ($modelfile) {
          $mfBytes = [System.Text.Encoding]::UTF8.GetBytes(($modelfile -join "`n"))
          $mfStream = [System.IO.MemoryStream]::new($mfBytes)
          $mfHash = (Get-FileHash -InputStream $mfStream -Algorithm SHA256).Hash.ToLower()
        }
      } catch {}
      $result.ollama.models += [ordered]@{
        name = $name
        digest = $digest
        size_bytes = $m.size
        modelfile_sha256 = $mfHash
      }
    }
    $result.ollama.reachable = $true
  }
} catch {
  Write-Warning "Ollama REST /api/tags unreachable; falling back to 'ollama list' parsing."
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
        $modelfile = ollama show $name --modelfile 2>$null
        if ($modelfile) {
          $mfBytes = [System.Text.Encoding]::UTF8.GetBytes(($modelfile -join "`n"))
          $mfStream = [System.IO.MemoryStream]::new($mfBytes)
          $mfHash = (Get-FileHash -InputStream $mfStream -Algorithm SHA256).Hash.ToLower()
        }
      } catch {}
      $result.ollama.models += [ordered]@{
        name = $name
        digest_short = $id
        modelfile_sha256 = $mfHash
      }
    }
    $result.ollama.reachable = $true
  } catch {
    Write-Warning "Ollama unreachable."
    $result.ollama.reachable = $false
  }
}

# ---- LM Studio models via OpenAI-compatible API ----
try {
  $ms = Invoke-RestMethod -Uri "$($result.lm_studio.base_url)/models" -Method GET -TimeoutSec 5
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

$result | ConvertTo-Json -Depth 8 | Set-Content -Path $outFile -Encoding UTF8
Write-Host "Evidence captured to: $outFile" -ForegroundColor Green
