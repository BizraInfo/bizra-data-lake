param(
  [Parameter(Mandatory=$true)][string]$Root,
  [Parameter(Mandatory=$true)][string]$OutDir
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$outPath = Join-Path $OutDir "secrets_scan.txt"

# Prefer trufflehog if present
$truffle = Get-Command trufflehog -ErrorAction SilentlyContinue
if ($truffle) {
  "Running trufflehog filesystem scan..." | Out-File -FilePath $outPath -Encoding UTF8
  trufflehog filesystem --no-update $Root 2>&1 | Out-File -FilePath $outPath -Append -Encoding UTF8
  Write-Host "Wrote: $outPath"
  exit 0
}

"trufflehog not found. Running lightweight regex scan (best-effort)..." | Out-File -FilePath $outPath -Encoding UTF8

$patterns = @(
  "AKIA[0-9A-Z]{16}",                     # AWS key
  "-----BEGIN(.*)PRIVATE KEY-----",       # private key blocks
  "sk-[A-Za-z0-9]{20,}",                  # OpenAI-like
  "AIza[0-9A-Za-z\\-_]{35}",              # Google API key
  "xox[baprs]-[0-9A-Za-z\\-]{10,}"        # Slack tokens
)

Get-ChildItem -LiteralPath $Root -Recurse -File -Force | ForEach-Object {
  $p = $_.FullName
  try {
    $c = Get-Content -LiteralPath $p -Raw -ErrorAction Stop
    foreach ($pat in $patterns) {
      if ($c -match $pat) {
        "POSSIBLE SECRET: $pat in $p" | Out-File -FilePath $outPath -Append -Encoding UTF8
      }
    }
  } catch { }
}

Write-Host "Wrote: $outPath"
