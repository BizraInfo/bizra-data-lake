# BIZRA Sovereign Partition Design

**Date:** 2026-02-26
**Status:** Approved
**Author:** Node0 Operator + Claude

## Context

Since Ramadan 2023, every day and night — 3 years of continuous work — has produced data serving BIZRA. This data is scattered across C:\ root (630 GB in repos), Downloads (400 GB), Desktop (128 GB), OneDrive (345 GB), mobile devices, and cloud storage. Build caches, duplicate venvs, stale worktrees, and MagicMock test artifacts add ~100+ GB of waste.

A single 3.8 TB C: partition holds everything — OS, programs, Docker (753 GB VHD), and all BIZRA data. Free space: 541 GB.

## Decision

Create a dedicated **B:\BIZRA** partition (2.5 TB) as the unified sovereign data home. C: retains Windows, programs, Docker, and user profile.

## Partition Layout

| Drive | Size | Purpose |
|-------|------|---------|
| C: | ~1290 GB | Windows + Programs + Docker + User Profile |
| B: | ~2500 GB | All BIZRA data, repos, assets, archives |

## B:\BIZRA Directory Tree

```
B:\BIZRA\
├── 00_CONSTITUTION/        SOUL.md, IDENTITY.md, Universal Constitution, NODE0_IDENTITY
├── 01_CORE/                Consolidated source code (git repos)
│   ├── data-lake/          BIZRA-DATA-LAKE
│   ├── node0/              BIZRA-NODE0
│   ├── dual-agentic/       Dual-Agentic-system
│   ├── projects/           BIZRA-PROJECTS
│   └── genesis/            Single consolidated genesis node
├── 02_DATA_PIPELINE/       Pipeline stages (deduplicated)
│   ├── 00_INTAKE/
│   ├── 01_RAW/
│   ├── 02_PROCESSED/
│   ├── 03_INDEXED/
│   ├── 04_GOLD/
│   └── 99_QUARANTINE/
├── 03_ASSETS/              Non-code BIZRA artifacts
│   ├── voice/              bizra-voice (12.4 GB)
│   ├── design/             award-winner-design, UI/UX
│   ├── models/             ML models, checkpoints
│   ├── documents/          PDFs, specs, research papers
│   ├── conversations/      Chat exports, Claude sessions
│   ├── media/              Screenshots, recordings, diagrams
│   └── mobile/             Phone backups
├── 04_ARCHIVE/             Verified old versions (compressed)
│   ├── backups/            NODE0-BACKUP, TaskMaster backup
│   ├── quarantine_2025/    Old quarantine
│   └── legacy/             Superseded but worth keeping
├── 05_IMPORTS/             Staging area for new data
│   ├── downloads/          From C:\Users\Downloads
│   ├── desktop/            From C:\Users\Desktop
│   ├── cloud/              From Google Drive / OneDrive
│   └── mobile/             From phone
├── 06_INDEX/               Master catalog
│   ├── manifest.parquet    SHA-256 hash of every file
│   ├── duplicates.log      All duplicates found during migration
│   ├── classification.jsonl AI-classified file metadata
│   └── timeline.jsonl      Every file mapped to creation date
└── GENESIS_MANIFEST.yaml   Root manifest
```

## Migration Phases

### Phase 0: Create Partition
- Shrink C: by 2500 GB using diskpart/PowerShell
- Create new partition, format NTFS, assign letter B:
- Label: "BIZRA"

### Phase 1: Deduplicate
- SHA-256 hash scan across all C:\ BIZRA directories
- Build master index before moving anything
- Identify duplicates across repos, backups, worktrees

### Phase 2: Migrate Repos
- Move 6 major repos to B:\BIZRA\01_CORE\
- Consolidate 4 genesis node copies into 1
- Delete stale worktrees (*-hexhash directories)

### Phase 3: Clean Build Artifacts
- Delete bizra-omega target/ (~85 GB)
- Remove redundant venvs (.venv, .venv-linux, .venv-wsl → keep 1)
- Clean .mypy_cache, __pycache__, MagicMock junk files
- Expected recovery: ~100+ GB

### Phase 4: Import Scattered Files
- Pull Downloads, Desktop into B:\BIZRA\05_IMPORTS\
- AI-classify each file (content type, BIZRA relevance, date)
- Move to correct destination in the tree

### Phase 5: Mobile + Cloud
- Import phone data into 05_IMPORTS\mobile\
- Import cloud drive into 05_IMPORTS\cloud\
- Same classification pipeline

### Phase 6: Master Index
- Generate manifest.parquet: SHA-256, file type, AI classification, creation date, source origin
- Every file on B: cataloged and searchable

## What Gets Deleted (After Verification)
- Duplicate genesis nodes (4 copies -> 1)
- Stale worktrees (*-hexhash directories)
- Build artifacts (target/, .venv-*, __pycache__)
- MagicMock junk files leaked into repo root
- Nested junk (BIZRA-DATA-LAKE/C:/, BIZRA-DATA-LAKE/BIZRA-DATA-LAKE/)

## What NEVER Gets Deleted Without Operator Confirmation
- Any file without an identical duplicate elsewhere
- Any git history
- Personal files

## Current C:\ Inventory (for reference)

### BIZRA Repos on C:\ Root
| Directory | Size |
|-----------|------|
| BIZRA-PROJECTS | 162.9 GB |
| BIZRA-NODE0 | 143.7 GB |
| BIZRA-DATA-LAKE | 137.7 GB |
| BIZRA-Dual-Agentic-system--main | 85.3 GB |
| bizra-genesis-node-repaired | 17.7 GB |
| BIZRA-NODE0-BACKUP | 17.5 GB |
| award-winner-design | 17.5 GB |
| bizra-voice | 12.4 GB |
| BIZRA-TaskMaster | 10.1 GB |
| momo phone | 7.4 GB |
| HERMES project | 6.4 GB |
| BIZRA-GENESIS-CLEAN | 4.5 GB |
| bizra-genesis-node-fresh | 2.1 GB |

### User Profile
| Directory | Size |
|-----------|------|
| AppData (Docker VHD 753 GB) | 1424.5 GB |
| Downloads | 400.6 GB |
| OneDrive | 345.4 GB |
| Desktop | 127.8 GB |
| .lmstudio | 124.9 GB |
| .cache | 59.9 GB |
| .ollama | 29.4 GB |
| .gemini | 25.2 GB |
| miniconda3 | 25.7 GB |

## Pre-requisites Completed
- [x] Docker/WSL fix (0x80072746)
- [x] Windows Insider exit (Canary -> ReleasePreview, services disabled)
- [x] Privacy lockdown (8 layers, telemetry dark)
- [ ] System restart (recommended before partition)
