################################################################################
# BIZRA Phase0 Audit — Node0 Live Evidence Capture
# Version: 1.0.0-LIVE
# Node: node_0000_genesis_momo_dubai
#
# Purpose: Capture immutable evidence of current system state
# Output: docs\evidence\phase0_week1\<timestamp>\*
#
# Run: powershell -ExecutionPolicy Bypass -File .\tools\phase0_audit.ps1
################################################################################

$ErrorActionPreference = "Stop"

# Resolve BIZRA_ROOT dynamically
$BIZRA_ROOT = & "$PSScriptRoot\..\scripts\bizra-root.ps1"
Write-Host "BIZRA_ROOT: $BIZRA_ROOT" -ForegroundColor Cyan

# Generate timestamped evidence directory
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$ev = Join-Path $BIZRA_ROOT "docs\evidence\phase0_week1\$ts"
New-Item -ItemType Directory -Force -Path $ev | Out-Null

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host " BIZRA Phase0 Audit — Evidence Capture" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Node: node_0000_genesis_momo_dubai" -ForegroundColor White
Write-Host "  Output: $ev" -ForegroundColor White
Write-Host ""

function Save($name, $cmd) {
    $out = Join-Path $ev $name
    Write-Host "[CAPTURE] $name" -ForegroundColor Yellow
    
    "# BIZRA Phase0 Evidence" | Out-File -FilePath $out -Encoding utf8
    "# File: $name" | Out-File -FilePath $out -Append -Encoding utf8
    "# Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss K')" | Out-File -FilePath $out -Append -Encoding utf8
    "# Command: $cmd" | Out-File -FilePath $out -Append -Encoding utf8
    "" | Out-File -FilePath $out -Append -Encoding utf8
    
    try {
        Invoke-Expression $cmd 2>&1 | Out-File -FilePath $out -Append -Encoding utf8
        Write-Host "  ✅ Captured" -ForegroundColor Green
    } catch {
        "ERROR: $($_.Exception.Message)" | Out-File -FilePath $out -Append -Encoding utf8
        Write-Host "  ⚠️  Error (logged)" -ForegroundColor Yellow
    }
}

# ═══════════════════════════════════════════════════════════════════════════
# [E0] NODE IDENTITY
# ═══════════════════════════════════════════════════════════════════════════
Write-Host "[E0] Node Identity..." -ForegroundColor Magenta

Save "00_node_identity.txt" "whoami"
Save "01_hostname.txt" "hostname"
Save "02_os_version.txt" "ver"
Save "03_bizra_root.txt" "echo $BIZRA_ROOT"
Save "04_timestamp.txt" "Get-Date -Format 'yyyy-MM-dd HH:mm:ss K'"
Save "05_timezone.txt" "Get-TimeZone | Format-List"

# ═══════════════════════════════════════════════════════════════════════════
# [E1] GPU & CUDA
# ═══════════════════════════════════════════════════════════════════════════
Write-Host "[E1] GPU & CUDA..." -ForegroundColor Magenta

Save "10_gpu_nvidia_smi.txt" "nvidia-smi"
Save "11_gpu_wmi.txt" "Get-CimInstance Win32_VideoController | Format-List Name,DriverVersion,AdapterRAM"
Save "12_cuda_path.txt" "where.exe nvcc"

# ═══════════════════════════════════════════════════════════════════════════
# [E2] OLLAMA RUNTIME
# ═══════════════════════════════════════════════════════════════════════════
Write-Host "[E2] Ollama Runtime..." -ForegroundColor Magenta

Save "20_ollama_version.txt" "ollama --version"
Save "21_ollama_list.txt" "ollama list"
Save "22_ollama_ps.txt" "ollama ps"
Save "23_ollama_env.txt" "Get-ChildItem Env: | Where-Object { $_.Name -like '*OLLAMA*' } | Format-List"

# ═══════════════════════════════════════════════════════════════════════════
# [E3] MODEL MANIFESTS
# ═══════════════════════════════════════════════════════════════════════════
Write-Host "[E3] Model Manifests..." -ForegroundColor Magenta

Save "30_modelfile_bizra_planner.txt" "ollama show bizra-planner:latest --modelfile"
Save "31_modelfile_deepseek_r1.txt" "ollama show deepseek-r1:8b --modelfile"
Save "32_modelfile_qwen.txt" "ollama show qwen2.5:7b --modelfile"
Save "33_modelfile_mistral.txt" "ollama show mistral:latest --modelfile"
Save "34_modelfile_llama.txt" "ollama show llama3.2:latest --modelfile"
Save "35_modelfile_nomic_embed.txt" "ollama show nomic-embed-text:latest --modelfile"

# ═══════════════════════════════════════════════════════════════════════════
# [E4] CONTAINERS
# ═══════════════════════════════════════════════════════════════════════════
Write-Host "[E4] Containers..." -ForegroundColor Magenta

Save "40_docker_version.txt" "docker --version"
Save "41_docker_ps_all.txt" "docker ps -a"
Save "42_docker_compose_ls.txt" "docker compose ls"
Save "43_docker_images.txt" "docker images"

# ═══════════════════════════════════════════════════════════════════════════
# [E5] DATABASE QUICK CHECK
# ═══════════════════════════════════════════════════════════════════════════
Write-Host "[E5] Database..." -ForegroundColor Magenta

# PostgreSQL check (if psql is available on host)
Save "50_postgres_tables.txt" "docker exec bizra-postgres psql -U postgres -d postgres -c '\dt'"
Save "51_redis_info.txt" "docker exec bizra-redis redis-cli INFO server"

# ═══════════════════════════════════════════════════════════════════════════
# [E6] INFERENCE PROBES
# ═══════════════════════════════════════════════════════════════════════════
Write-Host "[E6] Inference Probes..." -ForegroundColor Magenta

# Probe 1: bizra-planner (deterministic-like request)
$probe1 = @"
Return ONLY valid JSON with these keys:
{
  "node_id": "node_0000_genesis_momo_dubai",
  "status": "operational",
  "slots": [
    {"name": "primary_reasoning", "model": "bizra-planner", "purpose": "orchestration"}
  ]
}
"@

$probe1 | Out-File -FilePath (Join-Path $ev "60_probe1_request.txt") -Encoding utf8 -NoNewline

try {
    $result = $probe1 | ollama run bizra-planner:latest 2>&1
    $result | Out-File -FilePath (Join-Path $ev "61_probe1_response.txt") -Encoding utf8
    Write-Host "  ✅ bizra-planner probe complete" -ForegroundColor Green
} catch {
    "ERROR: $($_.Exception.Message)" | Out-File -FilePath (Join-Path $ev "61_probe1_response.txt") -Encoding utf8
    Write-Host "  ⚠️  bizra-planner probe failed" -ForegroundColor Yellow
}

# Probe 2: nomic-embed determinism check
try {
    $embedTest = "BIZRA seed phrase for determinism check"
    $result = $embedTest | ollama run nomic-embed-text:latest 2>&1
    $result | Out-File -FilePath (Join-Path $ev "62_embed_output.txt") -Encoding utf8
    
    # Hash the embedding output
    $hash = ($result | Out-String).Trim() | 
        ForEach-Object { [System.Security.Cryptography.SHA256]::Create().ComputeHash([System.Text.Encoding]::UTF8.GetBytes($_)) } |
        ForEach-Object { [BitConverter]::ToString($_) -replace '-','' }
    
    "Input: $embedTest" | Out-File -FilePath (Join-Path $ev "63_embed_hash.txt") -Encoding utf8
    "SHA256: $hash" | Out-File -FilePath (Join-Path $ev "63_embed_hash.txt") -Append -Encoding utf8
    
    Write-Host "  ✅ Embedding probe complete" -ForegroundColor Green
} catch {
    "ERROR: $($_.Exception.Message)" | Out-File -FilePath (Join-Path $ev "62_embed_output.txt") -Encoding utf8
    Write-Host "  ⚠️  Embedding probe failed" -ForegroundColor Yellow
}

# ═══════════════════════════════════════════════════════════════════════════
# [E7] SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
Write-Host "[E7] Generating Summary..." -ForegroundColor Magenta

$summary = @"
# BIZRA Phase0 Audit Summary
# Node: node_0000_genesis_momo_dubai
# Timestamp: $ts
# Evidence Directory: $ev

## Files Captured
$(Get-ChildItem $ev -File | ForEach-Object { "- $($_.Name) ($([math]::Round($_.Length/1KB,2)) KB)" } | Out-String)

## Status
Evidence capture: COMPLETE
Total files: $(Get-ChildItem $ev -File).Count
Total size: $([math]::Round((Get-ChildItem $ev -File | Measure-Object -Property Length -Sum).Sum / 1MB, 2)) MB

## Next Steps
1. Review evidence files in: $ev
2. Create deterministic model variants (optional)
3. Generate slot manifests
4. Run golden set validation
5. Seal + commit
"@

$summary | Out-File -FilePath (Join-Path $ev "ZZ_SUMMARY.txt") -Encoding utf8

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host " ✅ PHASE0 EVIDENCE CAPTURE COMPLETE" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""
Write-Host "Output: $ev" -ForegroundColor Cyan
Write-Host ""
Write-Host "Files captured:" -ForegroundColor Yellow
Get-ChildItem $ev -File | ForEach-Object {
    Write-Host "  ✓ $($_.Name) ($([math]::Round($_.Length/1KB,2)) KB)" -ForegroundColor White
}
Write-Host ""
