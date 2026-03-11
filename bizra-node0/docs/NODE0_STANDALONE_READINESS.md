# Node0 Standalone Readiness — MVSA Sign-Off

This document describes the **Minimum Viable Sovereign Architecture (MVSA)** for Node0.

## Canonical Commands

```bash
python scripts/node0_standalone.py activate --architect "MoMo"
python scripts/node0_standalone.py prove-mvsa
python scripts/node0_standalone.py task "write file missions/mvsa.txt :: node0 mvsa proof"
python scripts/node0_standalone.py health
python scripts/node0_standalone.py health   # fresh process observes persisted restart recovery
```

No alternate MVSA completion path is allowed through `node0_activate.py` or ad hoc scripts.

## Activation Flow (v2)

1. **Authority resolution** — canonical `sovereign_state/node0_genesis.json` + `genesis_hash.txt` (fail-closed)
2. PAT/SAT identity derived from authority artifacts (not onboarding defaults)
3. Optional helper signer provision/reuse under `sovereign_state/identity/credentials.json` (non-authoritative)
4. Hardware scan and asset registry (`sovereign_state/node0_assets.json`)
5. URP signed pledge + verification (`sovereign_state/urp_pledge.json`)
6. PAT awareness publication (`sovereign_state/pat_awareness.json`)
7. **Rust MVSA proof** — loopback bootstrap + self-validation (`sovereign_state/node0_mvsa_proof.json`)
8. Lifecycle v2 gate update (`sovereign_state/node0_lifecycle.json`, schema 2.0.0)

## Lifecycle v2 Status Semantics

| Status | Meaning |
|--------|---------|
| `blocked` | Authority invalid/missing, Rust MVSA failed, or structural gates not met |
| `degraded` | Authority + MVSA valid, but mission receipt or restart recovery incomplete |
| `ready` | All 11 status-determining gates satisfied |

### 11 Gates

| Gate | Set by |
|------|--------|
| `genesis_authority_valid` | `activate` |
| `identity_ready` | `activate` |
| `pat_sat_ready` | `activate` |
| `urp_signed` | `activate` |
| `urp_verified` | `activate` |
| `assets_written` | `activate` |
| `awareness_written` | `activate` |
| `mvsa_network_bootstrap_ok` | `activate` / `prove-mvsa` |
| `mvsa_self_validation_ok` | `activate` / `prove-mvsa` |
| `mission_path_receipted` | `task` (on evidence receipt) |
| `restart_recovery_ready` | mutating commands persist it when all artifacts reload cleanly; `health` only reports it |

### Availability Gates (informational — NOT status-determining)

The runtime lifecycle JSON also stores availability gates for operational health monitoring.
These do NOT affect `_compute_status()` and are NOT required for `ready` status.

| Gate | Set by | Purpose |
|------|--------|---------|
| `desktop_bridge_reachable` | `health` | AHK HDA TCP probe (port 9743) |
| `mcp_available` | `health` | MCP server availability |
| `a2a_available` | `health` | Agent-to-Agent protocol |
| `telescript_available` | `health` | TeleScript mobile agent |

Total entries in `node0_lifecycle.json`: 15 (11 status-determining + 4 availability).
The "All 11 gates satisfied" criterion above refers only to the status-determining gates.

### CLI Exit Codes

| Code | Meaning |
|------|---------|
| `0` | `ready` |
| `2` | `degraded` |
| `3` | `blocked` or failure |

## Local API

```bash
python scripts/node0_standalone.py serve --host 127.0.0.1 --port 8091
```

Endpoints:

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | No (loopback) | Lifecycle health |
| POST | `/activate` | API key | Full activation |
| POST | `/prove-mvsa` | API key | Run Rust MVSA proof |
| GET | `/mvsa` | API key | Read persisted proof |
| POST | `/task` | API key | Mission execution |
| GET | `/assets` | API key | Node0 assets |
| GET | `/lifecycle` | API key | Lifecycle v2 state |

`GET /health` remains read-only. It reports persisted MVSA state and does not rewrite lifecycle files.

## Authority Resolution Order

1. Canonical: `sovereign_state/node0_genesis.json` + `genesis_hash.txt`
2. Legacy ceremony: `sovereign_state/genesis.json` (migratable)
3. Legacy ceremony: `bizra-storage/genesis.json` (migratable)
4. Reference only: `04_GOLD/genesis.json` (NOT sufficient for MVSA)

Migration persists: `sovereign_state/node0_authority_migration.json`

`04_GOLD/genesis.json` remains provenance only. It is not sufficient by itself for Node0 MVSA authority.

## Rust MVSA Proof

Binary resolution: `BIZRA_NODE0_MVSA_BIN` → release → debug → `cargo run` → fail closed.

Output: `sovereign_state/node0_mvsa_proof.json` with genesis validation, loopback bootstrap, and self-validation.

If no operational signer exists yet, activation provisions a local helper credential under `sovereign_state/identity/credentials.json`. This signer is for URP and receipts only; Node0 authority still comes exclusively from canonical genesis artifacts.

## Acceptance Criteria

- Canonical authority validates from `sovereign_state/`
- Rust MVSA proof: `genesis_hash_valid=true`, `bootstrap_ok=true`, `self_validation_ok=true`
- Task emits evidence receipt ID
- A fresh-process `health` call reports `restart_recovery_ready=true`
- Lifecycle v2 status: `ready`
