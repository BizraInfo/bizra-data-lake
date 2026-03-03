Get-ChildItem C:\BIZRA-DATA-LAKE -Directory | ForEach-Object {
    $size = (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    $sizeMB = [math]::Round($size / 1MB, 1)
    $sizeGB = [math]::Round($size / 1GB, 2)
    Write-Host ("{0,-40} {1,10} MB  ({2,6} GB)" -f $_.Name, $sizeMB, $sizeGB)
} | Out-Null

# Total
$total = (Get-ChildItem C:\BIZRA-DATA-LAKE -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
Write-Host "`n--- TOTAL: $([math]::Round($total / 1GB, 2)) GB ---"
