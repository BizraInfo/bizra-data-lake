# Claude-Flow Memory Map (BIZRA Crosswalk)

## Overview
This repository includes two Claude-Flow memory locations:
- `.swarm/memory.db` (SQLite, v3.0.0 schema in `.swarm/schema.sql`)
- `.claude-flow/memory/*.json` (CLI/session artifacts and summaries)

Treat `.swarm/schema.sql` as the canonical schema for the DB-backed memory system.

## Storage Layout
```
.swarm/
  memory.db    # SQLite database (WAL)
  schema.sql   # DDL for all tables and indexes
  state.json   # Swarm runtime state
.claude-flow/memory/
  *.json       # Session state, integration summaries, patterns, metrics
```

## Table Map (from schema.sql)
- `memory_entries`: Main key/value store with `namespace`, `type` (semantic, episodic, procedural, working, pattern), optional embeddings, tags/metadata, and access tracking.
- `patterns` / `pattern_history`: Learned patterns with confidence, decay, and versioning.
- `trajectories` / `trajectory_steps`: Multi-step learning traces with rewards and outcomes.
- `sessions`: Persistent session state, project path, and lifecycle metrics.
- `migration_state`: Migration progress and recovery tracking.
- `vector_indexes`: HNSW configuration and index stats.
- `metadata`: Schema version and feature flags.

## Crosswalk to BIZRA Memory
- `memory_entries.type=working` -> TemporalMemoryHierarchy L2 (working).
- `memory_entries.type=episodic` -> TemporalMemoryHierarchy L3 (episodic).
- `memory_entries.type=semantic` -> TemporalMemoryHierarchy L4 (semantic).
- `memory_entries.type=procedural` -> TemporalMemoryHierarchy L5 (expertise).
- `memory_entries.type=pattern` and `patterns` -> Federation `PatternStore` and ARTE pattern pools.
- `sessions` -> `core/sovereign/bridge.py` session memory (persisted state).
- `trajectories` -> Pinnacle/SAPE trace inputs (potential UnifiedStalk conversion).
- `vector_indexes` -> 03_INDEXED metadata (HNSW settings).

## MCP-First Access (Primary)
- MCP `filesystem` server already points at `/mnt/c/BIZRA-DATA-LAKE`; use it to read `.claude-flow/memory/*.json`.
- MCP `claude-flow-sqlite` (in `.mcp.json`) exposes `.swarm/memory.db` for SQL queries over `memory_entries`, `patterns`, and `sessions`.
- The MCP `memory` server is a separate runtime key/value store; it does not replace the Claude-Flow DB.

## Adapter and CLI
Use the adapter to inspect, query, or export Claude-Flow memory:

```
python tools/claude_flow_memory_adapter.py stats
python tools/claude_flow_memory_adapter.py query "auth" --namespace project --limit 10
python tools/claude_flow_memory_adapter.py export --out 04_GOLD/claude_flow_memory.jsonl
python tools/claude_flow_memory_adapter.py export --as-sovereign --snr 0.0 \
  --out 04_GOLD/claude_flow_sovereign.jsonl
```

Notes:
- `--as-sovereign` maps entries into a Sovereign catalog style record so they can be merged later.
- SNR is not stored in Claude-Flow; provide `--snr` explicitly if you want ranking behavior.

## UnifiedMemory (Recommended)
`sovereign_memory.UnifiedMemory` merges the Sovereign Catalog with Claude-Flow memory sources (`.swarm/memory.db` and `.claude-flow/memory/*.json`). `bizra_prime.py` now uses it automatically when available.

Example:
```
from sovereign_memory import UnifiedMemory

mem = UnifiedMemory()
results = mem.search_across_domains("architecture", top_k=5)
```

## Integration Path (Recommended)
1. Export JSONL via the adapter.
2. Store exports under `04_GOLD/` or `02_PROCESSED/data/`.
3. If needed, merge into `sovereign_catalog.parquet` using a separate enrichment pass.
