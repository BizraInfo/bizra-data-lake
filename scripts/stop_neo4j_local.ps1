$ErrorActionPreference = "Stop"

param(
  [string]$ContainerName = "bizra-node0-graph"
)

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  Write-Error "docker not found on PATH."
}

$running = docker ps --filter "name=^/${ContainerName}$" --format "{{.Names}}"
if ($running -and $running.Trim() -eq $ContainerName) {
  Write-Host "Stopping Neo4j container: $ContainerName"
  docker stop $ContainerName | Out-Null
  exit 0
}

Write-Host "Neo4j container not running: $ContainerName"

