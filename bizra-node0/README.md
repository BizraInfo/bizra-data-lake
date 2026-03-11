# BIZRA Genesis Node0

> Sovereign Lifecycle Runtime — the canonical production surface for Node0.

## What This Is

This is the **production extraction** of Node0 from `bizra-data-lake`.
It contains only the dependency closure needed to boot, prove MVSA,
execute tasks, and pass the genesis ceremony.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Activate Node0
python scripts/node0_standalone.py activate --architect "MoMo"

# Prove MVSA
python scripts/node0_standalone.py prove-mvsa

# Execute a task
python scripts/node0_standalone.py task "write file missions/genesis.txt :: Node0 birth proof"

# Health check
python scripts/node0_standalone.py health

# Serve API
python scripts/node0_standalone.py serve --host 127.0.0.1 --port 8091

# Genesis ceremony
bash scripts/node0_genesis_ceremony.sh --full
```

## Canonical Commands

| Command | Purpose |
|---------|---------|
| `activate --architect "MoMo"` | Birth Node0 with genesis identity |
| `prove-mvsa` | Run Rust MVSA proof binary |
| `task "..."` | Execute mission with evidence receipt |
| `health` | Report lifecycle status and gate states |
| `serve` | Start sovereign API server |

## Document Hierarchy

| Document | Role |
|----------|------|
| `docs/NODE0_STANDALONE_READINESS.md` | MVSA specification |
| `docs/constitutional/BIZRA-Node0-Definition-of-Done-v1.0-LOCKED.md` | Verification gate (v1.2, 19 hard gates) |
| `docs/constitutional/NODE0_DOD_CORRECTION_MATRIX.md` | Audit trail |
| `docs/OPERATIONS_RUNBOOK.md` | Operator procedures |

## Upstream

Extracted from [bizra-data-lake](https://github.com/BizraInfo/bizra-data-lake)
at commit `274396c`. See `UPSTREAM_IMPORT_MANIFEST.yaml` in the parent repo
for the complete dependency closure analysis.

## License

AGPL-3.0-or-later
