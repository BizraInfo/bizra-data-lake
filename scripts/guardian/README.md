# BIZRA Resource Guardian v1.0

Automated resource management for NODE0 -- keeps your machine dedicated to BIZRA.

## What It Does

| Feature | How |
|---|---|
| **Auto-kills bloatware** | Teams, Xbox, GameBar, OneDrive, PC Manager -- gone on startup and when resources spike |
| **Disk I/O protection** | Stops defrag, search indexer, SysMain when disk usage exceeds 70% |
| **WSL2 memory cap** | Configurable limit (default 32GB for 128GB machine) |
| **Process priority boost** | WSL, Docker, LM Studio get AboveNormal priority automatically |
| **Continuous monitoring** | Daemon checks every 60 seconds, escalates response based on severity |
| **Startup optimization** | Disables telemetry, Xbox services, unnecessary scheduled tasks |
| **Power plan** | Sets High Performance, disables USB suspend and disk timeouts |

## Quick Install

```powershell
# Run as Administrator
cd C:\BIZRA-DATA-LAKE\scripts\guardian
.\Install-Guardian.ps1

# Then restart WSL
wsl --shutdown
```

## Thresholds (configurable)

| Resource | Threshold | Action |
|---|---|---|
| Disk I/O | > 70% | Stop defrag, search indexer, SysMain |
| Memory | > 75% | Kill Priority 1 (bloatware) |
| Memory | > 85% | Kill Priority 1+2 (+ OneDrive, Search) |
| Memory | > 90% | Kill Priority 1+2+3 (aggressive cleanup) |
| CPU | > 80% | Detect and report resource hogs |

## Files

```
~\.bizra\
├── BIZRA-ResourceGuardian.ps1   # Main guardian script
├── status.ps1                    # Quick system status
├── run-guardian.ps1              # Manual trigger
├── stop-guardian.ps1             # Stop daemon
├── logs\                         # Daily log files
│   └── guardian-YYYY-MM-DD.log
└── config\                       # Future: JSON config
```

## Usage

```powershell
# Check system status
. ~/.bizra/status.ps1

# Run guardian once (manual)
. ~/.bizra/run-guardian.ps1

# Run in daemon mode
powershell -ExecutionPolicy Bypass -File ~/.bizra/BIZRA-ResourceGuardian.ps1 -DaemonMode

# Dry run (see what it would do)
powershell -ExecutionPolicy Bypass -File ~/.bizra/BIZRA-ResourceGuardian.ps1 -DryRun

# Custom thresholds
powershell -ExecutionPolicy Bypass -File ~/.bizra/BIZRA-ResourceGuardian.ps1 -DaemonMode -CpuThreshold 70 -MemoryThreshold 70 -DiskThreshold 50
```

## Protected Processes (never killed)

WSL, Docker, LM Studio, NVIDIA, Chrome, VS Code, Cursor, Windows Terminal, Windows Defender, Claude, core OS processes.

## Customization

Edit the `$KillableProcesses` and `$ProtectedProcesses` arrays in `BIZRA-ResourceGuardian.ps1` to adjust what gets killed or protected.
