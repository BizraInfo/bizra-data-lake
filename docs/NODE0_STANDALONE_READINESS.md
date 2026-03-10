# Node0 Standalone Readiness — MVSA Sign-Off

This document describes the **Minimum Viable Sovereign Architecture (MVSA)** for Node0.

## Canonical Commands

```bash
python scripts/node0_standalone.py activate --architect "MoMo"
python scripts/node0_standalone.py prove-mvsa
python scripts/node0_standalone.py task "write file missions/mvsa.txt :: node0 mvsa proof"
python scripts/node0_standalone.py health
python scripts/node0_standalone.py health   # second call validates restart recovery
```

No alternate MVSA completion path is allowed through `node0_activate.py` or ad hoc scripts.

## Activation Flow (v2)

1. **Authority resolution** — canonical `sovereign_state/node0_genesis.json` + `genesis_hash.txt` (fail-closed)
2. PAT/SAT identity derived from authority artifacts (not onboarding defaults)
3. Hardware scan and asset registry (`sovereign_state/node0_assets.json`)
4. URP signed pledge + verification (`sovereign_state/urp_pledge.json`)
5. PAT awareness publication (`sovereign_state/pat_awareness.json`)
6. **Rust MVSA proof** — loopback bootstrap + self-validation (`sovereign_state/node0_mvsa_proof.json`)
7. Lifecycle v2 gate update (`sovereign_state/node0_lifecycle.json`, schema 2.0.0)

## Lifecycle v2 Status Semantics

| Status | Meaning |
|--------|---------|
| `blocked` | Authority invalid/missing, Rust MVSA failed, or structural gates not met |
| `degraded` | Authority + MVSA valid, but mission receipt or restart recovery incomplete |
| `ready` | All 11 gates satisfied |

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
| `restart_recovery_ready` | `health` (second call, when all artifacts present) |

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

## Authority Resolution Order

1. Canonical: `sovereign_state/node0_genesis.json` + `genesis_hash.txt`
2. Legacy ceremony: `sovereign_state/genesis.json` (migratable)
3. Legacy ceremony: `bizra-storage/genesis.json` (migratable)
4. Reference only: `04_GOLD/genesis.json` (NOT sufficient for MVSA)

Migration persists: `sovereign_state/node0_authority_migration.json`

## Rust MVSA Proof

Binary resolution: `BIZRA_NODE0_MVSA_BIN` → release → debug → `cargo run` → fail closed.

Output: `sovereign_state/node0_mvsa_proof.json` with genesis validation, loopback bootstrap, and self-validation.

## Acceptance Criteria

- Canonical authority validates from `sovereign_state/`
- Rust MVSA proof: `genesis_hash_valid=true`, `bootstrap_ok=true`, `self_validation_ok=true`
- Task emits evidence receipt ID
- Second `health` call: `restart_recovery_ready=true`
- Lifecycle v2 status: `ready`
