$roots = @(
    'C:\BIZRA-DATA-LAKE',
    'C:\BIZRA-Dual-Agentic-system--main',
    'C:\BIZRA-NODE0',
    'C:\Users\BIZRA-OS\Downloads',
    'B:\BIZRA-SOVEREIGN'
)

Write-Host '=== PAT DATA SOURCE CENSUS ==='
Write-Host ''

$grandFiles = 0
$grandMB = 0

foreach ($root in $roots) {
    if (-not (Test-Path $root)) {
        Write-Host "  $root : NOT FOUND"
        Write-Host ''
        continue
    }
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    # Fast count using robocopy /L (list only, no copy)
    $count = 0
    $totalBytes = 0
    $extCounts = @{}
    $extBytes = @{}
    
    $skipDirs = @('.git','node_modules','.venv','.venv-linux','__pycache__','.mypy_cache','.pytest_cache','.ruff_cache','.cache','.benchmarks','.fastembed_cache','.venv-apex','coverage','.codex')
    $skipExts = @('.exe','.dll','.so','.pyc','.pyo','.whl','.lock','.idx','.pack')
    
    Get-ChildItem -Path $root -Recurse -File -ErrorAction SilentlyContinue | ForEach-Object {
        $dir = $_.DirectoryName
        $skip = $false
        foreach ($sd in $skipDirs) {
            if ($dir -match [regex]::Escape("\$sd\") -or $dir -match [regex]::Escape("\$sd$")) {
                $skip = $true
                break
            }
        }
        if (-not $skip) {
            $ext = $_.Extension.ToLower()
            if ($ext -notin $skipExts) {
                $count++
                $totalBytes += $_.Length
                if (-not $extCounts.ContainsKey($ext)) { $extCounts[$ext] = 0; $extBytes[$ext] = 0 }
                $extCounts[$ext]++
                $extBytes[$ext] += $_.Length
            }
        }
    }
    $sw.Stop()
    $mb = [math]::Round($totalBytes / 1MB, 1)
    $gb = [math]::Round($totalBytes / 1GB, 3)
    $grandFiles += $count
    $grandMB += $mb
    Write-Host ("  {0}" -f $root)
    Write-Host ("    {0:N0} files | {1:N3} GB | {2:N1}s" -f $count, $gb, $sw.Elapsed.TotalSeconds)
