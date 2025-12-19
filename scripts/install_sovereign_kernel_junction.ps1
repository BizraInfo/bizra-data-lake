$ErrorActionPreference = "Stop"

param(
  [string]$TargetRoot = "C:\\BIZRA-PROJECTS\\00-GENESIS",
  [string]$TargetName = "core"
)

$repoRoot = (Get-Item $PSScriptRoot).Parent.FullName
$source = Join-Path $repoRoot "core"
$target = Join-Path $TargetRoot $TargetName

if (-not (Test-Path $TargetRoot)) {
  Write-Error "Target root not found: $TargetRoot"
}

if (-not (Test-Path $source)) {
  Write-Error "Source 'core' package not found: $source"
}

if (Test-Path $target) {
  Write-Host "Target already exists (skipping): $target"
  Write-Host "If you want to replace it, remove it manually then re-run."
  exit 0
}

Write-Host "Creating junction:"
Write-Host "  $target -> $source"
New-Item -ItemType Junction -Path $target -Target $source | Out-Null

Write-Host "OK. You can now run from $TargetRoot:"
Write-Host "  python -m core.main"

