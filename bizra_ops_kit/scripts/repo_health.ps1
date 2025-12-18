param(
  [Parameter(Mandatory=$true)][string]$Root,
  [Parameter(Mandatory=$true)][string]$OutDir
)

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$outPath = Join-Path $OutDir "repo_health.txt"
"Repo health run at $(Get-Date -AsUTC -Format o)" | Out-File -FilePath $outPath -Encoding UTF8

Push-Location $Root

function Run($cmd) {
  "---- $cmd" | Out-File -FilePath $outPath -Append -Encoding UTF8
  try {
    Invoke-Expression $cmd 2>&1 | Out-File -FilePath $outPath -Append -Encoding UTF8
  } catch {
    $_ | Out-File -FilePath $outPath -Append -Encoding UTF8
  }
}

if (Test-Path "Cargo.toml") {
  Run "cargo --version"
  Run "cargo test --all --locked"
  if (Get-Command cargo-audit -ErrorAction SilentlyContinue) { Run "cargo audit" }
}

if (Test-Path "package.json") {
  Run "node --version"
  Run "npm --version"
  Run "npm ci"
  Run "npm test"
  Run "npm audit --audit-level=high"
}

if (Test-Path "pyproject.toml" -or Test-Path "requirements.txt") {
  Run "python --version"
  Run "python -m pip --version"
  if (Get-Command pip-audit -ErrorAction SilentlyContinue) { Run "pip-audit" }
  if (Test-Path "pytest.ini" -or Test-Path "tests") { Run "python -m pytest -q" }
}

Pop-Location
Write-Host "Wrote: $outPath"
