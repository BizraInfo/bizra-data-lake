$vol = Get-Volume -DriveLetter C
$totalGB = [math]::Round($vol.Size / 1GB, 1)
$freeGB = [math]::Round($vol.SizeRemaining / 1GB, 1)
$usedGB = $totalGB - $freeGB
Write-Host "C: Total=$totalGB GB | Used=$usedGB GB | Free=$freeGB GB"

$bvol = Get-Volume -DriveLetter B
$btotalGB = [math]::Round($bvol.Size / 1GB, 1)
$bfreeGB = [math]::Round($bvol.SizeRemaining / 1GB, 1)
Write-Host "B: Total=$btotalGB GB | Free=$bfreeGB GB"

Write-Host ""
Write-Host "=== SHRINK QUERY ==="
$size = (Get-PartitionSupportedSize -DriveLetter C)
$minGB = [math]::Round($size.SizeMin / 1GB, 1)
$maxGB = [math]::Round($size.SizeMax / 1GB, 1)
$shrinkableGB = $maxGB - $minGB
Write-Host "C: Min size=$minGB GB | Max size=$maxGB GB | Shrinkable=$shrinkableGB GB"
Write-Host "Target: Shrink C by 2300 GB, expand B to ~2500 GB"
