# Node0 DoD Correction Matrix

**Date:** 2026-03-11
**Context:** DevOps ground-truth review of `BIZRA-Node0-Definition-of-Done-v1.0-LOCKED.md`
**Canonical truth sources:**
- `docs/NODE0_STANDALONE_READINESS.md` — MVSA contract
- `sovereign_state/node0_lifecycle.json` — Lifecycle v2 runtime state (14 gates)
- `scripts/node0_standalone.py` — Canonical CLI surface

## The Core Problem

The DoD v1.0 mixes four categories:
1. **Canonical MVSA truth** (lifecycle v2 gates, authority resolution)
2. **Machine-specific local state** (ceremony_signer.key, .keystore)
3. **Historical genesis artifacts** (00_GENESIS/genesis.json, pat/sat manifests)
4. **Future ceremony design** (Node0GenesisReceipt schema)

Only category 1 should be hard gates. Categories 2-4 are supporting evidence.

## Correction Matrix

### Layer 1: GENESIS INTEGRITY

| Gate | Current | Verdict | Reason |
|------|---------|---------|--------|
| 1.1 | Check `sovereign_state/node0_genesis.json` has `identity` | **KEEP** | This IS the canonical authority file |
| 1.2 | Check `00_GENESIS/genesis.json` block_number | **DEMOTE** → supporting | `00_GENESIS/` is reference-only per authority resolution order. Canonical authority is `sovereign_state/node0_genesis.json` |
| 1.3 | Check `genesis_hash.txt` non-empty | **KEEP** | Part of canonical authority pair |
| 1.4 | Check `ceremony_signer.key` + `identity.json` | **REWRITE** → Check lifecycle gate `identity_ready` | Current gate overfits to local file names |
| 1.5 | Check `ceremony_signer.key` exists | **REMOVE** → redundant with 1.4 | Duplicates 1.4 |
| 1.6 | Check `pat_manifest.json` | **REWRITE** → Check lifecycle `pat_sat_ready` | PAT/SAT identity derived from authority, not manifest files |
| 1.7 | Check `sat_manifest.json` | **REMOVE** → merged into 1.6 | Same gate as 1.6 |
| 1.8 | Check `genesis/identity.json` | **REMOVE** → covered by 1.4 rewrite | Redundant |
| 1.9 | `resolve_authority()` | **KEEP** | Direct canonical API call |

**After correction: 4 hard gates** (1.1, 1.3, 1.4→lifecycle, 1.9)

### Layer 2: PERSONAL SOVEREIGN ACTIVATION

| Gate | Current | Verdict | Reason |
|------|---------|---------|--------|
| 2.1 | `PAT_AGENT_COUNT == 7` | **KEEP** | Constitutional constant verification |
| 2.2 | `len(PAT_ROLES) == 7` | **KEEP** | Ceremony definition integrity |
| 2.3 | `SAT_AGENTS_PER_NODE == 5` | **KEEP** | Constitutional constant |
| 2.4 | `len(SAT_ROLES) == 5` | **KEEP** | Ceremony definition integrity |
| 2.5 | PAT bootstrap importable | **REWRITE** → Check lifecycle `pat_sat_ready` | Import check is weak; lifecycle gate is runtime truth |
| 2.6 | MOE routing works | **KEEP** | Functional verification beyond import |
| 2.7 | Receipt schema loads | **DEMOTE** → supporting | Import check only |

**After correction: 5 hard gates + 1 supporting**

### Layer 3: STAND-ALONE CAPABILITY (MVSA)

| Gate | Current | Verdict | Reason |
|------|---------|---------|--------|
| 3.1 | Check block_number in `00_GENESIS/` | **REMOVE** → already demoted in L1 | Same artifact as 1.2 |
| 3.2 | PBFT consensus importable | **DEMOTE** → supporting | Import check, not runtime proof |
| 3.3 | Federation gossip importable | **DEMOTE** → supporting | Import check, not runtime proof |
| 3.4 | MVSA self-validation via `prove-mvsa` | **REWRITE** → Read lifecycle `mvsa_self_validation_ok` from persisted JSON | Current grep-scraping is fragile. The truth is in `node0_lifecycle.json` |
| 3.5 | Authority resolution | **KEEP** (merge with 1.9) | Already covered |
| 3.6 | Ihsan = 0.95 | **KEEP** | Constitutional invariant |
| 3.7 | SNR = 0.85 | **KEEP** | Constitutional invariant |
| 3.8 | Gini = 0.35 | **KEEP** | Constitutional invariant |

**After correction: 4 hard gates** (3.4→lifecycle read, 3.6, 3.7, 3.8)

### Layer 4: DEVICE CONSECRATION

| Gate | Current | Verdict | Reason |
|------|---------|---------|--------|
| 4.1 | Dirs `sovereign_state/` + `00_GENESIS/` exist | **REWRITE** → only `sovereign_state/` | `00_GENESIS/` is reference-only |
| 4.2 | Health via `node0_standalone.py health` | **REWRITE** → Read lifecycle status from JSON, assert != "blocked" | Shell grep is fragile; JSON is canonical |
| 4.3 | Atomic I/O importable | **DEMOTE** → supporting | Import check |
| 4.4 | Lifecycle state persisted | **KEEP** | Core runtime truth |
| 4.5 | Evidence ledger importable | **DEMOTE** → supporting | Import check |
| 4.6 | Mission orchestrator importable | **DEMOTE** → supporting | Import check |
| 4.7 | `node0_baseline.json` exists | **DEMOTE** → supporting | Not part of MVSA contract |

**After correction: 3 hard gates** (4.1→sovereign_state, 4.2→lifecycle JSON, 4.4)

### Layer 5: REPLICATION READINESS

| Gate | Current | Verdict | Reason |
|------|---------|---------|--------|
| 5.1 | NodeTemplate loads | **KEEP** | Functional verification |
| 5.2 | Ceremony deterministic | **KEEP** | Critical architectural property |
| 5.3 | PAT/SAT specs exist | **DEMOTE** → supporting | File presence, not runtime |
| 5.4 | Cross-repo constants sync | **KEEP** | CI-level canonical check |
| 5.5 | Installer exists | **DEMOTE** → supporting | File presence |
| 5.6 | CLI entry works | **DEMOTE** → supporting | Not MVSA core |
| 5.7 | Upgrade path docs | **DEMOTE** → supporting | File presence |

**After correction: 3 hard gates** (5.1, 5.2, 5.4)

## Summary

| Layer | Current Gates | Hard Gates After | Supporting After | Removed |
|-------|--------------|-----------------|-----------------|---------|
| L1 | 9 | 4 | 1 | 4 (redundant) |
| L2 | 7 | 5 | 2 | 0 |
| L3 | 8 | 4 | 2 | 2 (redundant) |
| L4 | 7 | 3 | 4 | 0 |
| L5 | 7 | 3 | 4 | 0 |
| **Total** | **38** | **19 hard** | **13 supporting** | **6 removed** |

## The Canonical Hard Gate Set (19 gates)

These read runtime truth, not file presence:

```
L1.1  sovereign_state/node0_genesis.json has identity key
L1.3  sovereign_state/genesis_hash.txt non-empty
L1.4  lifecycle.gates.identity_ready == true
L1.9  resolve_authority() succeeds

L2.1  PAT_AGENT_COUNT == 7
L2.2  len(PAT_ROLES) == 7
L2.3  SAT_AGENTS_PER_NODE == 5
L2.4  len(SAT_ROLES) == 5
L2.5  lifecycle.gates.pat_sat_ready == true
L2.6  MOEEngine().route() returns experts

L3.4  lifecycle.gates.mvsa_self_validation_ok == true
L3.6  UNIFIED_IHSAN_THRESHOLD == 0.95
L3.7  UNIFIED_SNR_THRESHOLD == 0.85
L3.8  ADL_GINI_THRESHOLD == 0.35

L4.1  sovereign_state/ directory exists
L4.2  lifecycle.status != "blocked"
L4.4  node0_lifecycle.json exists and has schema_version "2.0.0"

L5.1  NodeTemplate.default() loads
L5.2  run_ceremony(seed, config) is deterministic
L5.4  validate_cross_repo_consistency() passes
```

## Node0GenesisReceipt

**Verdict: DEMOTE to appendix.** No code currently emits or validates this schema. When `node0_standalone.py` gains a `ceremony` command that produces this receipt, promote to hard gate.

## Document Hierarchy After Fix

1. `docs/NODE0_STANDALONE_READINESS.md` — **Canonical MVSA contract** (unchanged)
2. `docs/constitutional/BIZRA-Node0-Definition-of-Done-v1.0-LOCKED.md` — **Birth gate** (rebased onto lifecycle v2)
3. `docs/constitutional/BIZRA-Definition-of-Done-Genesis-100.md` — **Scale gate** (unchanged)

## Pre-Commit Checklist

- [x] Rewrite DoD v1.0 with 19 hard gates + 13 supporting (clearly separated) — v1.1
- [x] All hard gates read `node0_lifecycle.json` or canonical authority files — v1.1
- [x] Remove grep/timeout scraping in favor of JSON reads — v1.1
- [x] Node0GenesisReceipt moved to appendix — v1.1
- [ ] README.md references both NODE0_STANDALONE_READINESS.md and DoD v1.2
- [x] `node0_genesis_ceremony.sh` updated to match — v1.1, tightened v1.2
- [x] git commit includes doc + script in same commit — v1.1

## v1.2 Correction (Ready Only)

**Date:** 2026-03-11
**Trigger:** Second DevOps ground-truth review identified 3 HIGH findings in v1.1

### Changes

| Finding | v1.1 State | v1.2 Fix |
|---------|-----------|----------|
| Gate 4.2 too weak | `!= "blocked"` allows `degraded` to pass | `== "ready"` — Ready Only locked by Mumo |
| Lifecycle coverage gap | DoD reads 3 of 14 gates | Gate 4.2 now implies all 11 status gates; availability gates documented as non-birth-critical |
| STANDALONE_READINESS drift | Doc says 11 gates, JSON has 14 | Doc updated to document 3 availability gates as informational addendum |
| Competing truth | Ambiguous DoD↔Readiness governance | Clear hierarchy: Readiness=spec, DoD=verification, Matrix=audit |
| Certification path | Undeclared | Linux/WSL2 declared in DoD header |

### Current Ceremony Result (v1.2)

```
Hard Gates: 18/19 passed, 1 failed
Failed: 4.2 (lifecycle.status == "degraded", not "ready")
Reason: mission_path_receipted not yet set — requires `node0_standalone.py task`
Score: 94.7%
```

This is **correct behavior**. Gate 4.2 will pass only when the full MVSA lifecycle reaches `ready`.
