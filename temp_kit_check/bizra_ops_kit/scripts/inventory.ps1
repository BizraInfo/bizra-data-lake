param(
  [Parameter(Mandatory=$true)][string]$Root,
  [Parameter(Mandatory=$true)][string]$OutDir,
  [switch]$IncludeHash = $false,
  [int]$HashMaxMB = 10,
  [int]$MaxFiles = 0
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$manifestPath = Join-Path $OutDir "manifest.jsonl"
if (Test-Path $manifestPath) { Remove-Item $manifestPath -Force }

function Get-Sha256($path) {
  $sha = [System.Security.Cryptography.SHA256]::Create()
  $stream = [System.IO.File]::OpenRead($path)
  try {
    $hash = $sha.ComputeHash($stream)
    return ([BitConverter]::ToString($hash) -replace "-", "").ToLowerInvariant()
  } finally {
    $stream.Dispose()
    $sha.Dispose()
  }
}

$count = 0
Get-ChildItem -LiteralPath $Root -Recurse -File -Force | ForEach-Object {
  if ($MaxFiles -gt 0 -and $count -ge $MaxFiles) { return }

  $full = $_.FullName
  $size = $_.Length
  $mtime = $_.LastWriteTimeUtc.ToString("o")

  $rec = [ordered]@{
    path = $full
    size_bytes = $size
    mtime_utc = $mtime
    ext = $_.Extension.ToLowerInvariant()
  }

  if ($IncludeHash) {
    $maxBytes = $HashMaxMB * 1024 * 1024
    if ($size -le $maxBytes) {
      try { $rec.sha256 = Get-Sha256 $full } catch { $rec.sha256 = $null }
    } else {
      $rec.sha256 = $null
    }
  }

  ($rec | ConvertTo-Json -Compress) | Add-Content -Path $manifestPath -Encoding UTF8
  $count++
}

Write-Host "Wrote manifest: $manifestPath"
Write-Host "Files indexed: $count"
