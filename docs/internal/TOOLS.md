# TOOLS.md - Local Notes

## Environment
- Workspace: C:\BIZRA-DATA-LAKE
- Shell: PowerShell

## Common Commands
- Install dev deps: pip install -e ".[dev]"
- Pinned deps: pip install -r requirements.lock
- Run intake once: .\DataLakeProcessor.ps1 -ProcessOnce
- Watch intake: .\DataLakeProcessor.ps1 -Watch
- Cloud ingestion dry run: .\CloudIngestion.ps1 -DryRun

## Notes
Add environment-specific details here (keys, hosts, device names, etc.).
