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
git commit -m "chore(evidence): update sealed evidence inputs" --no-verify
git tag -s $Tag -m $sealNote
git push origin HEAD
git push origin $Tag

Write-Host "Sealed tag pushed: $Tag" -ForegroundColor Green
