# Scan top-level C:\ directories
Write-Host "=== C:\ TOP-LEVEL ===" -ForegroundColor Cyan
Get-ChildItem C:\ -Directory -Force -ErrorAction SilentlyContinue | ForEach-Object {
    $size = (Get-ChildItem $_.FullName -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    $sizeGB = [math]::Round($size / 1GB, 2)
    if ($sizeGB -ge 0.01) {
        Write-Host ("{0,-50} {1,10} GB" -f $_.Name, $sizeGB)
    }
}

# User profile breakdown
Write-Host "`n=== USER PROFILE (C:\Users\BIZRA-OS) ===" -ForegroundColor Cyan
Get-ChildItem "C:\Users\BIZRA-OS" -Directory -Force -ErrorAction SilentlyContinue | ForEach-Object {
    $size = (Get-ChildItem $_.FullName -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    $sizeGB = [math]::Round($size / 1GB, 2)
    if ($sizeGB -ge 0.1) {
        Write-Host ("{0,-50} {1,10} GB" -f $_.Name, $sizeGB)
    }
}

# Key heavy hitters
Write-Host "`n=== KNOWN HEAVY DIRS ===" -ForegroundColor Cyan
$heavyPaths = @(
    "C:\Users\BIZRA-OS\AppData\Local\Docker",
    "C:\Users\BIZRA-OS\AppData\Local\npm-cache",
    "C:\Users\BIZRA-OS\AppData\Local\NuGet",
    "C:\Users\BIZRA-OS\AppData\Roaming\npm",
    "C:\Users\BIZRA-OS\Downloads",
    "C:\Users\BIZRA-OS\Documents",
    "C:\Users\BIZRA-OS\Desktop",
    "C:\Users\BIZRA-OS\.cargo",
    "C:\Users\BIZRA-OS\.rustup",
    "C:\Users\BIZRA-OS\.ollama",
    "C:\Users\BIZRA-OS\.cache",
    "C:\Program Files\Docker",
    "C:\Program Files\nodejs",
    "C:\ProgramData\Docker",
    "C:\Windows\Installer"
)
foreach ($p in $heavyPaths) {
    if (Test-Path $p) {
        $size = (Get-ChildItem $p -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
        $sizeGB = [math]::Round($size / 1GB, 2)
        if ($sizeGB -ge 0.1) {
            Write-Host ("{0,-50} {1,10} GB" -f $p.Replace("C:\Users\BIZRA-OS\","~\"), $sizeGB)
        }
    }
}

# Total C: usage
Write-Host "`n=== DISK SUMMARY ===" -ForegroundColor Cyan
$vol = Get-Volume -DriveLetter C
$totalGB = [math]::Round($vol.Size / 1GB, 1)
$freeGB = [math]::Round($vol.SizeRemaining / 1GB, 1)
$usedGB = $totalGB - $freeGB
Write-Host "Total: $totalGB GB | Used: $usedGB GB | Free: $freeGB GB"
