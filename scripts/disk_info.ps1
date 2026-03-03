Write-Host "=== DISKS ===" -ForegroundColor Cyan
Get-Disk | ForEach-Object {
    $sizeGB = [math]::Round($_.Size / 1GB, 1)
    Write-Host "Disk $($_.Number): $($_.FriendlyName) | $sizeGB GB | $($_.PartitionStyle)" -ForegroundColor White
}

Write-Host "`n=== PARTITIONS ===" -ForegroundColor Cyan
Get-Partition | ForEach-Object {
    $sizeGB = [math]::Round($_.Size / 1GB, 1)
    $letter = if ($_.DriveLetter) { "$($_.DriveLetter):" } else { "---" }
    Write-Host "  Disk$($_.DiskNumber) Part$($_.PartitionNumber): $letter  $sizeGB GB  ($($_.Type))" -ForegroundColor White
}

Write-Host "`n=== VOLUMES ===" -ForegroundColor Cyan
Get-Volume | Where-Object { $_.DriveLetter } | ForEach-Object {
    $sizeGB = [math]::Round($_.Size / 1GB, 1)
    $freeGB = [math]::Round($_.SizeRemaining / 1GB, 1)
    Write-Host "  $($_.DriveLetter):  $freeGB GB free / $sizeGB GB total  ($($_.FileSystemLabel))" -ForegroundColor White
}
