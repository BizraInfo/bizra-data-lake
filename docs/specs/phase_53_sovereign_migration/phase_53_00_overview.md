# Phase 53.0: Sovereign Data Lake Migration -- Master Index

**Status:** SPEC DRAFT
**Author:** BIZRA Node0 Engineering
**Date:** 2026-02-26
**Giants:** Shannon (entropy-based classification), Deming (PDCA quality cycles), Lamport (hash chains for integrity verification)

---

## Problem Statement

Three years of work since Ramadan 2023 have produced ~1.9 TB of BIZRA artifacts
scattered across C:\ with no canonical organization. The C: drive contains:

| Source | Size | Description |
|--------|------|-------------|
| BIZRA-PROJECTS | 162.9 GB | Project workspace aggregate |
| BIZRA-NODE0 | 143.7 GB | Node0 runtime + configs |
| BIZRA-DATA-LAKE | 137.7 GB | Data pipeline (85 GB is target/cache) |
| BIZRA-Dual-Agentic-system | 85.3 GB | Multi-agent reasoning |
| 4x genesis copies | ~20 GB | Redundant genesis node clones |
| bizra-voice | 12.4 GB | Voice assets |
| Downloads | 400.6 GB | Unsorted inbound files |
| Desktop | 127.8 GB | Working surface debris |
| OneDrive | 345.4 GB | Cloud sync mirror |
| Build artifacts | ~103 GB | target/, venvs, caches |
| Nested junk | ~4 GB | Self-referential directory loops |
| Stale worktrees | ~2 GB | Orphaned git worktrees |

**Core tension:** C: is a 1 TB system drive under pressure. B: is a dedicated 195 GB
BIZRA volume (expandable) with a clean constitutional directory tree already built.

**Goal:** Migrate all BIZRA-relevant data to B:\BIZRA with zero data loss, full
deduplication, and a searchable master index -- while freeing ~500+ GB on C:.

---

## Migration Phases

| Phase | File | Script | Purpose |
|-------|------|--------|---------|
| 0 | This file | -- | Master index and architecture |
| 1 | `phase_53_01_dedup_scanner.md` | `dedup_scanner.py` | SHA-256 manifest + duplicate detection |
| 2 | `phase_53_02_artifact_cleaner.md` | `artifact_cleaner.py` | Build artifact purge (~103 GB) |
| 3 | `phase_53_03_repo_migrator.md` | `repo_migrator.py` | Git repo migration with integrity checks |
| 4 | `phase_53_04_file_classifier.md` | `file_classifier.py` | AI-powered file classification |
| 5 | `phase_53_05_import_pipeline.md` | `import_pipeline.py` | Downloads/Desktop/Cloud import |
| 6 | `phase_53_06_master_index.md` | `master_index.py` | Master index + timeline generation |

**Execution order:** 1 -> 2 -> 3 -> 5 -> 4 (within 5) -> 6
Phase 2 must precede Phase 3 to reduce transfer volume.
Phase 6 runs last as the final integrity seal.

---

## Architecture

```
    C:\ SOURCE TOPOLOGY                    MIGRATION ENGINE                    B:\BIZRA TARGET
    ==================                    ================                    ================

    BIZRA-DATA-LAKE/  ----+
    BIZRA-NODE0/      ----+
    BIZRA-PROJECTS/   ----+               +------------------+
    BIZRA-Dual-Ag*/   ----+----+--------> | dedup_scanner.py |---> 06_INDEX/manifest.parquet
    bizra-genesis-*4  ----+    |          | (SHA-256 stream) |---> 06_INDEX/duplicates.log
    bizra-voice/      ----+    |          +------------------+
                               |                  |
                               |                  v
                               |          +--------------------+
                               +--------> | artifact_cleaner.py|---> Receipt + ~103 GB freed
                               |          | (dry-run default)  |
                               |          +--------------------+
                               |                  |
                               |                  v
                               |          +-------------------+       00_CONSTITUTION/
                               +--------> | repo_migrator.py  |---->  01_CORE/
                               |          | (robocopy + SHA)  |         data-lake/
                               |          +-------------------+         node0/
                               |                  |                     dual-agentic/
                               |                  v                     projects/
    Downloads/ -------+        |          +-------------------+         genesis/
    Desktop/   -------+--------+--------> | import_pipeline.py|---->  02_DATA_PIPELINE/
    OneDrive/  -------+        |          | (copy -> classify)|---->  03_ASSETS/
    Mobile/    -------+        |          +-------------------+       04_ARCHIVE/
                               |                  |                   05_IMPORTS/ (staging)
                               |                  v
                               |          +-------------------+
                               +--------> | file_classifier.py|---> classification.jsonl
                               |          | (rules + LLM)     |
                               |          +-------------------+
                               |                  |
                               |                  v
                               |          +-------------------+
                               +--------> | master_index.py   |---> 06_INDEX/manifest.parquet
                                          | (final integrity) |---> 06_INDEX/timeline.jsonl
                                          +-------------------+---> 06_INDEX/stats_report.json
```

---

## Safety Invariants

These are hard gates. Violation of any invariant aborts the operation.

1. **NEVER delete without duplicate verification.** Every file targeted for deletion must
   have its SHA-256 verified against at least one surviving copy at the destination.

2. **NEVER lose git history.** Repository migrations verify `.git/HEAD` SHA-256 match
   post-transfer. If mismatch, abort and alert.

3. **Operator confirmation for ambiguous files.** Any file the classifier scores
   `relevance < 0.5` or categorizes as `unknown` is routed to human review queue.

4. **Dry-run by default.** Every destructive script requires explicit `--execute` flag.
   Without it, the script reports what it WOULD do and exits cleanly.

5. **Receipt chain.** Every operation generates a timestamped receipt (JSON) recording
   what was done, enabling full rollback audit.

6. **Source markers.** Migrated source directories receive a `.MIGRATED_TO_B` marker
   file containing the destination path and migration timestamp. This prevents
   double-migration and enables symlink creation.

7. **Ihsan threshold >= 0.95.** Migration quality score (files verified / files migrated)
   must meet the production Ihsan threshold from `core/integration/constants.py`.

---

## B: Drive Specifications

```
Volume Label:    BIZRA
File System:     NTFS
Total Capacity:  195 GB (current)
Expandable:      Yes (external SSD, additional partitions)
Mount in WSL:    /mnt/b/BIZRA
Windows Path:    B:\BIZRA
```

**Target tree (pre-built):**

```
B:\BIZRA\
+-- 00_CONSTITUTION/          # Foundational documents, covenants, ADRs
+-- 01_CORE/                  # Active git repositories
|   +-- data-lake/            # <- C:\BIZRA-DATA-LAKE
|   +-- node0/                # <- C:\BIZRA-NODE0
|   +-- dual-agentic/         # <- C:\BIZRA-Dual-Agentic-system--main
|   +-- projects/             # <- C:\BIZRA-PROJECTS
|   +-- genesis/              # <- Consolidated from 4 copies
+-- 02_DATA_PIPELINE/         # Medallion architecture data flow
|   +-- 00_INTAKE/
|   +-- 01_RAW/
|   +-- 02_PROCESSED/
|   +-- 03_INDEXED/
|   +-- 04_GOLD/
|   +-- 99_QUARANTINE/
+-- 03_ASSETS/                # Non-code creative assets
|   +-- voice/                # <- bizra-voice
|   +-- design/
|   +-- models/
|   +-- documents/
|   +-- conversations/
|   +-- media/
|   +-- mobile/
+-- 04_ARCHIVE/               # Cold storage
|   +-- backups/
|   +-- quarantine_2025/
|   +-- legacy/
+-- 05_IMPORTS/               # Staging area for inbound files
|   +-- downloads/
|   +-- desktop/
|   +-- cloud/
|   +-- mobile/
+-- 06_INDEX/                 # Master index, manifests, search
+-- GENESIS_MANIFEST.yaml     # Root manifest
```

---

## Environment Configuration

All scripts resolve paths via environment variable:

```python
import os
BIZRA_SOVEREIGN_ROOT = os.environ.get("BIZRA_SOVEREIGN_ROOT", "/mnt/b/BIZRA")
```

**WSL path translation:**

| Windows | WSL |
|---------|-----|
| `B:\BIZRA` | `/mnt/b/BIZRA` |
| `C:\BIZRA-DATA-LAKE` | `/mnt/c/BIZRA-DATA-LAKE` |
| `C:\Users\BIZRA-OS\Downloads` | `/mnt/c/Users/BIZRA-OS/Downloads` |

---

## Cross-References

- **Phase 52:** Lifecycle emulation (node0 systemd integration)
- **Phase 41-51:** Prior phases building the infrastructure this migration serves
- **`core/integration/constants.py`:** Ihsan/SNR/ADL thresholds used as quality gates
- **`GENESIS_MANIFEST.yaml`:** Root manifest on B: drive documenting the tree structure
- **ADR-004:** Swarm engine strategies (relevant for parallel migration workers)

---

## Quality Gates (from `core/integration/constants.py`)

| Gate | Threshold | Application |
|------|-----------|-------------|
| Ihsan (production) | >= 0.95 | Files verified / files migrated ratio |
| SNR (minimum) | >= 0.85 | Dedup scanner signal-to-noise |
| ADL Gini | <= 0.35 | Distribution fairness of storage across categories |

---

## Estimated Impact

| Metric | Before | After |
|--------|--------|-------|
| C: free space | ~120 GB | ~620 GB |
| B: used space | ~2 GB | ~150 GB |
| Duplicate files | Unknown | 0 (within tolerance) |
| Searchable index | None | Full manifest + timeline |
| Genesis copies | 4 | 1 (canonical) |
| Build artifacts | ~103 GB | 0 (cleaned) |
| Git integrity | Unverified | SHA-256 verified |

---

## Deming PDCA Cycle

Each migration phase follows Plan-Do-Check-Act:

1. **Plan:** Dry-run generates report of intended operations
2. **Do:** Operator reviews, approves with `--execute`
3. **Check:** Post-migration integrity verification (SHA-256 match)
4. **Act:** Generate receipt, update master index, create symlinks

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Data loss during transfer | Low | Critical | SHA-256 pre/post verification |
| B: drive failure | Low | Critical | Receipts enable re-migration from C: |
| Git history corruption | Low | High | .git/HEAD hash verification |
| Misclassified files | Medium | Medium | Human review queue for low-confidence |
| WSL/NTFS permission issues | Medium | Low | robocopy preserves ACLs, rsync --perms |
| B: capacity exceeded | Low | Medium | Artifact cleaning runs first, monitor usage |

---

*"The fundamental problem of communication is that of reproducing at one point
either exactly or approximately a message selected at another point."*
-- Claude Shannon, 1948

Phase 53 is Shannon's problem applied to a developer's filesystem: reproduce the
essential signal (3 years of work) at a new location (B:), while eliminating the
noise (duplicates, build artifacts, junk). The entropy of the source is high; the
entropy of the destination must be low and well-structured.
