# Fixes Redis Certs by restoring the key file
$path = "config/redis/redis-server-key.pem"
$source = "config/redis/redis-server-key.pem.EXPIRED"

if (Test-Path $source) {
    Copy-Item $source $path -Force
    Write-Host "Restored redis-server-key.pem" -ForegroundColor Green
} else {
    Write-Host "Source key not found!" -ForegroundColor Red
}

# Also ensure local permissions (conceptually)
