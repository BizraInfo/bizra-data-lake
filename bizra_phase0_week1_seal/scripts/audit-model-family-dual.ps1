################################################################################
# BIZRA Dual-Provider Model Audit (Ollama + LM Studio) — PowerShell
# Run (from pack root):
#   powershell -ExecutionPolicy Bypass -File .\scripts\audit-model-family-dual.ps1
################################################################################

$ErrorActionPreference = "Stop"

function Write-Section($title) {
  Write-Host ""
  Write-Host "============================================================" -ForegroundColor Cyan
  Write-Host $title -ForegroundColor Cyan
  Write-Host "============================================================" -ForegroundColor Cyan
}

$PackRoot = (Get-Item $PSScriptRoot).Parent.FullName
$OutDir = Join-Path $PackRoot "evidence\\model_family"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
"TIMESTAMP_LOCAL: $stamp" | Out-File (Join-Path $OutDir "audit_timestamp.txt")

# ---------------- [0] Host Info ----------------
Write-Section "[0] HOST INFO"
try {
  $cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)
  $ramBytes = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
  $ramGB = [math]::Round($ramBytes / 1GB, 2)
  $host = $env:COMPUTERNAME

  "HOSTNAME: $host" | Out-File (Join-Path $OutDir "host_info.txt")
  "CPU: $cpu" | Out-File (Join-Path $OutDir "host_info.txt") -Append
  "RAM_GB: $ramGB" | Out-File (Join-Path $OutDir "host_info.txt") -Append

  Write-Host "Hostname: $host" -ForegroundColor Green
  Write-Host "CPU: $cpu" -ForegroundColor Green
  Write-Host "RAM: $ramGB GB" -ForegroundColor Green
} catch {
  Write-Host "Failed host info: $($_.Exception.Message)" -ForegroundColor Yellow
}

# ---------------- [1] Ollama list ----------------
Write-Section "[1] OLLAMA LIST"
try {
  "COMMAND: ollama list" | Out-File (Join-Path $OutDir "ollama_list.txt")
  ollama list | Out-File (Join-Path $OutDir "ollama_list.txt") -Append
  Write-Host "Saved: $OutDir\\ollama_list.txt" -ForegroundColor Green
} catch {
  Write-Host "Ollama list failed: $($_.Exception.Message)" -ForegroundColor Red
}

# ---------------- [2] Ollama show modelfile (try key models) ----------------
Write-Section "[2] OLLAMA SHOW --MODELFILE"
$OllamaModels = @("bizra-planner:latest","deepseek-r1:8b","llama3.2:latest","mistral:latest","qwen2.5:7b","deepseek-llm:latest","nomic-embed-text:latest")
foreach ($m in $OllamaModels) {
  $safe = $m -replace '[:/]', '_'
  $out = Join-Path $OutDir ("ollama_show_modelfile_{0}.txt" -f $safe)
  try {
    "COMMAND: ollama show $m --modelfile" | Out-File $out
    ollama show $m --modelfile | Out-File $out -Append
    Write-Host "Saved: $out" -ForegroundColor Green
  } catch {
    Write-Host "Not available: $m" -ForegroundColor DarkGray
  }
}

# ---------------- [3] Ollama store snapshot (manifests/blobs) ----------------
Write-Section "[3] OLLAMA STORE SNAPSHOT"
$OllamaRoot = Join-Path $env:USERPROFILE ".ollama\\models"
"OLLAMA_ROOT: $OllamaRoot" | Out-File (Join-Path $OutDir "ollama_store_snapshot.txt")
if (Test-Path $OllamaRoot) {
  try {
    "== MANIFESTS ==" | Out-File (Join-Path $OutDir "ollama_store_snapshot.txt") -Append
    Get-ChildItem (Join-Path $OllamaRoot "manifests") -Recurse -ErrorAction SilentlyContinue |
      Select-Object FullName, Length, LastWriteTime | Out-String |
      Out-File (Join-Path $OutDir "ollama_store_snapshot.txt") -Append

    "== BLOBS (LATEST 40) ==" | Out-File (Join-Path $OutDir "ollama_store_snapshot.txt") -Append
    Get-ChildItem (Join-Path $OllamaRoot "blobs") -ErrorAction SilentlyContinue |
      Sort-Object LastWriteTime -Descending |
      Select-Object -First 40 Name, Length, LastWriteTime | Out-String |
      Out-File (Join-Path $OutDir "ollama_store_snapshot.txt") -Append

    Write-Host "Saved: $OutDir\\ollama_store_snapshot.txt" -ForegroundColor Green
  } catch {
    Write-Host "Snapshot failed: $($_.Exception.Message)" -ForegroundColor Yellow
  }
} else {
  Write-Host "Ollama root not found: $OllamaRoot" -ForegroundColor Yellow
}

# ---------------- [4] LM Studio /v1/models ----------------
Write-Section "[4] LM STUDIO MODELS"
$LmBase = "http://127.0.0.1:1234"
try {
  $models = Invoke-RestMethod -Uri "$LmBase/v1/models" -Method GET -TimeoutSec 5
  $models | ConvertTo-Json -Depth 8 | Out-File (Join-Path $OutDir "lmstudio_models.json")
  Write-Host "Saved: $OutDir\\lmstudio_models.json" -ForegroundColor Green
} catch {
  Write-Host "LM Studio not responding at $LmBase/v1/models : $($_.Exception.Message)" -ForegroundColor Yellow
}

# ---------------- [5] LM Studio file hashes (best-effort scan) ----------------
Write-Section "[5] LM STUDIO FILE HASHES (BEST-EFFORT)"
$Candidates = @(
  Join-Path $env:USERPROFILE ".lmstudio\\models",
  Join-Path $env:LOCALAPPDATA "LMStudio\\models",
  Join-Path $env:APPDATA "LMStudio\\models"
)

$hashes = @()
foreach ($dir in $Candidates) {
  if (Test-Path $dir) {
    Write-Host "Scanning: $dir" -ForegroundColor Gray
    Get-ChildItem $dir -Recurse -File -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -match "\\.gguf$|\\.bin$|\\.safetensors$" } |
      ForEach-Object {
        try {
          $h = Get-FileHash -Algorithm SHA256 -Path $_.FullName
          $hashes += [pscustomobject]@{
            path = $_.FullName
            sha256 = $h.Hash.ToLower()
            size_bytes = $_.Length
            last_write = $_.LastWriteTime.ToString("o")
          }
        } catch {}
      }
  }
}

$hashes | ConvertTo-Json -Depth 4 | Out-File (Join-Path $OutDir "lmstudio_file_hashes.json")
Write-Host "Saved: $OutDir\\lmstudio_file_hashes.json (count=$($hashes.Count))" -ForegroundColor Green

# ---------------- [6] GPU memory ----------------
Write-Section "[6] GPU MEMORY"
try {
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,nounits |
    Out-File (Join-Path $OutDir "gpu_memory.txt")
  Write-Host "Saved: $OutDir\\gpu_memory.txt" -ForegroundColor Green
} catch {
  "GPU not detected (CPU-only mode)" | Out-File (Join-Path $OutDir "gpu_memory.txt")
  Write-Host "nvidia-smi not available (CPU-only ok)" -ForegroundColor Yellow
}

Write-Section "AUDIT COMPLETE"
Write-Host "Collect these files to seal the artifacts manifest:" -ForegroundColor Yellow
Write-Host "  - evidence/model_family/ollama_list.txt" -ForegroundColor Gray
Write-Host "  - evidence/model_family/ollama_show_modelfile_*.txt" -ForegroundColor Gray
Write-Host "  - evidence/model_family/lmstudio_models.json" -ForegroundColor Gray
Write-Host "  - evidence/model_family/lmstudio_file_hashes.json" -ForegroundColor Gray
Write-Host "  - evidence/model_family/gpu_memory.txt" -ForegroundColor Gray

