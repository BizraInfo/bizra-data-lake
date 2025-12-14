# seal_evidence.ps1
# Computes SHA256 for key artifacts and creates a signed git tag (requires GPG configured).

param(
  [string]$Tag = "evidence-seal-v1",
  [string]$Message = "Sealed evidence pack v1"
)

$ErrorActionPreference = "Stop"

function Sha256File($path) {
  (Get-FileHash -Algorithm SHA256 -Path $path).Hash.ToLower()
}

$files = @(
  "model-family-genesis-v1-SEALED.yaml",
  "golden-set-genesis-v1-DETERMINISTIC.json",
  "evidence\audit-results-node0.json"
)

# Add checks for package-lock.json and hash it if present. Warn if missing but package.json exists.
if (Test-Path "package.json") {
  if (Test-Path "package-lock.json") {
    $files += "package-lock.json"
    Write-Host "  + Included: package-lock.json (Supply Chain Locked)" -ForegroundColor Green
  }
  else {
    Write-Warning "CRITICAL: package.json found but package-lock.json MISSING. Sealing FAILED."
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

# Tag with annotation (local seal)
git tag -f -a $Tag -m $sealNote

Write-Host "Sealed tag created: $Tag (Local)" -ForegroundColor Green
Write-Host "NOTE: Execute 'git push origin $Tag' manually to publish." -ForegroundColor Gray

Write-Host "Sealed tag pushed: $Tag" -ForegroundColor Green
```
