# tools/run_master.ps1
# One-command, evidence-first activation for the solo-dev "personal agentic team".
# Safe defaults: no deletions; indexes + receipts only.

[CmdletBinding()]
param(
  [string]$ChatInputDir,
  [string]$ModelTarget,
  [string]$OllamaHost,
  [switch]$SkipChatIngest,
  [switch]$IngestCrashReports,
  [switch]$RunLLMTeam,
  [switch]$StoreFullText
)

$ErrorActionPreference = "Stop"

if ($ModelTarget) { $env:MODEL_TARGET = $ModelTarget }
if ($OllamaHost) { $env:OLLAMA_HOST = $OllamaHost }

Write-Host "BIZRA Master Run" -ForegroundColor Cyan
Write-Host "MODEL_TARGET=$($env:MODEL_TARGET)" -ForegroundColor DarkCyan
Write-Host "OLLAMA_HOST=$($env:OLLAMA_HOST)" -ForegroundColor DarkCyan

$activateParams = @{}
if ($ChatInputDir) { $activateParams.ChatInputDir = $ChatInputDir }
if ($SkipChatIngest) { $activateParams.SkipChatIngest = $true }
if ($StoreFullText) { $activateParams.StoreFullText = $true }
if ($IngestCrashReports) { $activateParams.IngestCrashReports = $true }
if ($RunLLMTeam) { $activateParams.RunLLMTeam = $true }

Write-Host ("Flags: SkipChatIngest=$SkipChatIngest StoreFullText=$StoreFullText IngestCrashReports=$IngestCrashReports RunLLMTeam=$RunLLMTeam") -ForegroundColor DarkCyan
Write-Host ("Forwarding: tools\\activate_team.ps1 " + (($activateParams.Keys | Sort-Object) -join ", ")) -ForegroundColor DarkCyan

& (Join-Path $PSScriptRoot "activate_team.ps1") @activateParams
