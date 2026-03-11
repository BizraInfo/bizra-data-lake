# BIZRA Definition of Done — Genesis Node0

## Version 1.1 · LOCKED

> **Date:** March 11, 2026 · Dubai
> **Author:** MoMo (محمد) / System Architect
> **Scope:** Node0 only — MoMo's machine as the Genesis Node
> **Status:** LOCKED — changes require SAT-5 consensus
> **Canonical truth:** `docs/NODE0_STANDALONE_READINESS.md` + `sovereign_state/node0_lifecycle.json`
> **Rule:** If any hard gate fails, Node0 is NOT declared complete.

---

## 1. Scope Boundary

**IN SCOPE:** Genesis identity, PAT-7 activation, SAT-5 activation, MVSA standalone proof, device consecration, replication readiness.

**OUT OF SCOPE:** Genesis-100 invitation, Alpha-10 rollout, multi-validator federation, full token economy, public launch.

---

## 2. Truth Sources

This document reads from two canonical sources:

1. **`sovereign_state/node0_lifecycle.json`** — Lifecycle v2 runtime state (14 gates, computed by `node0_standalone.py`)
2. **`sovereign_state/node0_genesis.json`** + **`genesis_hash.txt`** — Canonical authority pair

All hard gates below read these sources. No gate depends on shell scraping, grep, or timeout heuristics.

---

## 3. The Five Layers

```
L1: GENESIS INTEGRITY            → Canonical authority validates
L2: PERSONAL SOVEREIGN ACTIVATION → PAT-7 / SAT-5 constitutionally alive
L3: STAND-ALONE CAPABILITY       → MVSA lifecycle gates satisfied
L4: DEVICE CONSECRATION          → Lifecycle not blocked, state persisted
L5: REPLICATION READINESS        → Template deterministic, constants synced
```

---

## 4. Hard Gates (19)

These must ALL pass. Each reads runtime truth, not file presence.

### Layer 1 — Genesis Integrity (4 gates)

| # | Gate | Verification | Pass |
|---|------|-------------|------|
| 1.1 | Canonical manifest exists | `node0_genesis.json` has `identity` key | JSON valid |
| 1.2 | Genesis hash anchored | `genesis_hash.txt` is non-empty | Non-empty |
| 1.3 | Identity ready | `lifecycle.gates.identity_ready == true` | true |
| 1.4 | Authority resolves | `resolve_authority(state_dir, project_root)` succeeds | Not None |

### Layer 2 — Personal Sovereign Activation (6 gates)

| # | Gate | Verification | Pass |
|---|------|-------------|------|
| 2.1 | PAT count = 7 | `constants.PAT_AGENT_COUNT == 7` | Exact |
| 2.2 | PAT roles defined | `len(genesis_ceremony.PAT_ROLES) == 7` | 7 roles |
| 2.3 | SAT count = 5 | `constants.SAT_AGENTS_PER_NODE == 5` | Exact |
| 2.4 | SAT roles defined | `len(genesis_ceremony.SAT_ROLES) == 5` | 5 roles |
| 2.5 | PAT/SAT ready | `lifecycle.gates.pat_sat_ready == true` | true |
| 2.6 | MOE routing functional | `MOEEngine().route(prompt)` returns experts | len > 0 |

### Layer 3 — Stand-Alone MVSA (4 gates)

| # | Gate | Verification | Pass |
|---|------|-------------|------|
| 3.1 | MVSA self-validation | `lifecycle.gates.mvsa_self_validation_ok == true` | true |
| 3.2 | Ihsan threshold | `UNIFIED_IHSAN_THRESHOLD == 0.95` | 0.95 |
| 3.3 | SNR threshold | `UNIFIED_SNR_THRESHOLD == 0.85` | 0.85 |
| 3.4 | Gini threshold | `ADL_GINI_THRESHOLD == 0.35` | 0.35 |

### Layer 4 — Device Consecration (3 gates)

| # | Gate | Verification | Pass |
|---|------|-------------|------|
| 4.1 | State directory exists | `sovereign_state/` is a directory | Exists |
| 4.2 | Lifecycle not blocked | `lifecycle.status != "blocked"` | degraded or ready |
| 4.3 | Lifecycle schema v2 | `lifecycle.schema_version == "2.0.0"` | 2.0.0 |

### Layer 5 — Replication Readiness (2 gates)

| # | Gate | Verification | Pass |
|---|------|-------------|------|
| 5.1 | Ceremony deterministic | Same seed + config → identical genesis hash | Hash match |
| 5.2 | Cross-repo sync | `validate_cross_repo_consistency()` | true |

---

## 5. Supporting Evidence (13 checks)

These provide additional confidence but do NOT block Node0 birth.

| # | Check | What it verifies |
|---|-------|-----------------|
| S.1 | `00_GENESIS/genesis.json` has block_number 0 | Historical genesis reference |
| S.2 | `ceremony_signer.key` exists | Local signing capability |
| S.3 | PAT/SAT manifest JSONs valid | Bootstrap artifact integrity |
| S.4 | PBFT consensus importable | Federation code present |
| S.5 | Federation gossip importable | Gossip protocol available |
| S.6 | Receipt schema importable | Proof chain infrastructure |
| S.7 | Atomic I/O importable | Crash-safe writes available |
| S.8 | Evidence ledger importable | Audit trail infrastructure |
| S.9 | Mission orchestrator importable | Task execution capability |
| S.10 | `node0_baseline.json` exists | Performance baseline recorded |
| S.11 | `NodeTemplate.default()` loads | Template instantiation works |
| S.12 | CLI entry point works | Operator surface functional |
| S.13 | Upgrade path documented | Evolution path exists |

---

## 6. Node0GenesisReceipt (Future — Appendix)

When `node0_standalone.py` gains a `ceremony` command that emits this receipt, it will be promoted to a hard gate. Until then, it is a design specification.

```python
@dataclass
class Node0GenesisReceipt:
    """Future: Cryptographically signed record of Node0 birth."""
    ceremony: str = "NODE0_GENESIS"
    timestamp_ms: int = 0
    node_id: str = ""
    hard_gates_passed: int = 0
    hard_gates_total: int = 19
    all_passed: bool = False
    receipt_hash: str = ""      # BLAKE3
    signature: str = ""         # Ed25519
    signer_pubkey: str = ""
```

---

## 7. What "Node0 Complete" Means

**When all 19 hard gates pass:**
- MoMo's machine is formally born as Node0
- PAT-7 and SAT-5 are constitutionally bound
- Genesis identity is cryptographically anchored
- MVSA self-validation has proven sovereignty
- Lifecycle v2 is not blocked
- Future nodes can replicate via deterministic ceremony

**Does NOT mean:** Genesis-100 readiness, public launch, multi-validator federation.

---

## 8. Relationship to NODE0_STANDALONE_READINESS.md

This document is the **birth gate**. It reads from the same lifecycle v2 truth that `NODE0_STANDALONE_READINESS.md` defines. It does not introduce new truth — it assembles existing canonical truth into a verifiable checklist.

The MVSA contract in `NODE0_STANDALONE_READINESS.md` remains the authoritative specification. This DoD is its verification instrument.

---

## 9. Document Control

| Field | Value |
|-------|-------|
| Version | 1.1 LOCKED |
| Date | 2026-03-11 |
| Lock policy | Changes require SAT-5 consensus |
| Canonical truth | `node0_lifecycle.json` + `node0_genesis.json` |
| Predecessor | `NODE0_STANDALONE_READINESS.md` (authoritative) |
