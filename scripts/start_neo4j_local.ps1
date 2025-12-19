$ErrorActionPreference = "Stop"

param(
  [string]$ContainerName = "bizra-node0-graph",
  [string]$Image = "neo4j:5.15-community",
  [string]$DataVolume = "bizra-node0-neo4j-data",
  [int]$HttpPort = 7474,
  [int]$BoltPort = 7687
)

if (-not $env:GRAPH_PASSWORD -or -not $env:GRAPH_PASSWORD.Trim()) {
  Write-Error "GRAPH_PASSWORD is required. Set `$env:GRAPH_PASSWORD then re-run."
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  Write-Error "docker not found on PATH."
}

$existing = docker ps -a --filter "name=^/${ContainerName}$" --format "{{.Names}}"
if ($existing -and $existing.Trim() -eq $ContainerName) {
  $running = docker ps --filter "name=^/${ContainerName}$" --format "{{.Names}}"
  if ($running -and $running.Trim() -eq $ContainerName) {
    Write-Host "Neo4j already running: $ContainerName"
  } else {
    Write-Host "Starting existing Neo4j container: $ContainerName"
    docker start $ContainerName | Out-Null
  }
} else {
  Write-Host "Creating Neo4j container: $ContainerName"
  docker volume create $DataVolume | Out-Null

  docker run -d --name $ContainerName `
    --restart unless-stopped `
    -p "127.0.0.1:${HttpPort}:7474" `
    -p "127.0.0.1:${BoltPort}:7687" `
    -v "${DataVolume}:/data" `
    -e "NEO4J_AUTH=neo4j/$env:GRAPH_PASSWORD" `
    $Image | Out-Null
}

$boltOk = (Test-NetConnection -ComputerName localhost -Port $BoltPort).TcpTestSucceeded
if (-not $boltOk) {
  Write-Warning "Neo4j Bolt port did not open yet. Wait a few seconds and re-run: Test-NetConnection localhost -Port $BoltPort"
} else {
  Write-Host "Neo4j is reachable:"
  Write-Host "  Browser: http://localhost:$HttpPort"
  Write-Host "  Bolt:    bolt://localhost:$BoltPort"
  Write-Host ""
  Write-Host "Next step:"
  Write-Host "  python .\\bizra_synaptic_loader.py"
}

