# seal_evidence.ps1
# Computes SHA256 for key artifacts and creates an annotated or signed git tag.

param(
  [string]$Tag = "evidence-seal-v1",
  [string]$Message = "Sealed evidence pack v1",
  [switch]$Sign
)

$ErrorActionPreference = "Stop"

function Resolve-RepoRoot {
  # Prefer script location so this works even if invoked from a different CWD/repo.
  if ($PSScriptRoot) { return $PSScriptRoot }

  try {
    $root = (git rev-parse --show-toplevel 2>$null)
    if ($root) { return ($root.Trim() -replace '/', '\') }
  } catch {}

  return (Resolve-Path ".").Path
}

function Sha256File($path) {
  (Get-FileHash -Algorithm SHA256 -Path $path).Hash.ToLower()
}

$repoRoot = Resolve-RepoRoot
Push-Location $repoRoot
try {

$files = @(
  "model-family-genesis-v1-SEALED.yaml",
  "golden-set-genesis-v1-DETERMINISTIC.json",
  "evidence\audit-results-node0.json"
)

# Optional Application Layer Seal (NPM)
if (Test-Path "ace-framework/package.json") {
  if (Test-Path "ace-framework/package-lock.json") {
    $files += "ace-framework/package-lock.json"
    Write-Host "  + Included: package-lock.json (Supply Chain Locked)" -ForegroundColor Green
  }
  else {
    Write-Warning "CRITICAL: ace-framework/package.json found but package-lock.json MISSING. Sealing FAILED."
    exit 1
  }
}

$lines = @()
foreach ($f in $files) {
  if (-not (Test-Path $f)) { throw "Missing file: $f" }
  $h = Sha256File $f
  $lines += "$f  sha256:$h"
}

$sealNote = @"
$Message

$(($lines -join "`n"))
"@

git add $files
# Commit if changes exist, ignore error if clean
git commit -m "chore(evidence): update sealed evidence inputs" --allow-empty 2>$null

# Tag with annotation (default) or signature (if GPG configured)
if ($Sign) {
  git tag -f -s $Tag -m $sealNote
} else {
  git tag -f -a $Tag -m $sealNote
}

Write-Host "Sealed tag created: $Tag (Local)" -ForegroundColor Green
Write-Host "NOTE: Execute 'git push origin $Tag' manually to publish." -ForegroundColor Gray

} finally {
  Pop-Location
}
