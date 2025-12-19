# setup_intelligence.ps1
# Ensures Ollama models exist + endpoints respond.
$ErrorActionPreference = "Stop"

Write-Host "== BIZRA Phase 1: Intelligence Setup ==" -ForegroundColor Cyan

# 1) Check Ollama
try {
  $v = Invoke-RestMethod -Uri "http://localhost:11434/api/version" -Method GET -TimeoutSec 3
  Write-Host "Ollama OK: $($v.version)" -ForegroundColor Green
} catch {
  Write-Host "Ollama not reachable at localhost:11434. Start Ollama first." -ForegroundColor Red
  exit 1
}

# 2) Ensure models
$required = @("deepseek-r1:8b", "mistral:latest")
$installed = (ollama list | Select-Object -Skip 1 | ForEach-Object { ($_ -split "\s{2,}")[0] }) | Where-Object { $_ }
foreach ($m in $required) {
  if ($installed -notcontains $m) {
    Write-Host "Pulling $m ..." -ForegroundColor Yellow
    ollama pull $m
  } else {
    Write-Host "Model present: $m" -ForegroundColor Green
  }
}

# 3) LM Studio health (optional)
try {
  $models = Invoke-RestMethod -Uri "http://localhost:1234/v1/models" -Method GET -TimeoutSec 3
  Write-Host "LM Studio OK. Models: $($models.data.Count)" -ForegroundColor Green
} catch {
  Write-Host "LM Studio not reachable at localhost:1234 (optional for Phase 1)." -ForegroundColor Yellow
}

Write-Host "Setup complete." -ForegroundColor Cyan
