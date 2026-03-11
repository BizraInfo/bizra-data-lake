# BIZRA Definition of Done — Genesis Node0

## Version 1.2 · LOCKED

> **Date:** March 11, 2026 · Dubai
> **Author:** MoMo (محمد) / System Architect
> **Scope:** Node0 only — MoMo's machine as the Genesis Node
> **Status:** LOCKED — changes require SAT-5 consensus
> **Canonical truth:** `sovereign_state/node0_lifecycle.json` (runtime) + `sovereign_state/node0_genesis.json` (authority)
> **Governance:** This DoD is the **birth gate** (verification instrument). `NODE0_STANDALONE_READINESS.md` is the **MVSA specification** (definition). The DoD verifies what the spec defines — neither replaces the other.
> **Certification path:** Linux / WSL2 via `bash scripts/node0_genesis_ceremony.sh`
> **Rule:** If any hard gate fails, Node0 is NOT declared complete. Lifecycle must report `status == "ready"` (not merely degraded).

---

## 1. Scope Boundary

**IN SCOPE:** Genesis identity, PAT-7 activation, SAT-5 activation, MVSA standalone proof, device consecration, replication readiness.

**OUT OF SCOPE:** Genesis-100 invitation, Alpha-10 rollout, multi-validator federation, full token economy, public launch.

---

## 2. Truth Sources

This document reads from two canonical sources:

1. **`sovereign_state/node0_lifecycle.json`** — Lifecycle v2 runtime state (computed by `node0_standalone.py`)
   - **11 status-determining gates** — used by `_compute_status()` to derive `blocked` / `degraded` / `ready`
   - **3 availability gates** — informational (desktop_bridge, mcp, a2a, telescript); do NOT affect status
   - Total: 14 entries in runtime JSON, 11 gates determine status
2. **`sovereign_state/node0_genesis.json`** + **`genesis_hash.txt`** — Canonical authority pair

All hard gates below read these sources. No gate depends on shell scraping, grep, or timeout heuristics.

### Lifecycle Gate Coverage

The DoD hard gates verify a **superset** of lifecycle truth:
- Gates 1.3, 2.5, 3.1 → named lifecycle gate checks (identity, PAT/SAT, MVSA)
- Gate 4.2 → requires `lifecycle.status == "ready"`, which implies ALL 11 status-determining gates are true
- Gates 1.1–1.4, 2.1–2.6, 3.2–3.4, 5.1–5.2 → constitutional and ceremony checks beyond lifecycle scope

The 3 availability gates (mcp_available, a2a_available, telescript_available) are NOT birth-critical. `desktop_bridge_reachable` is NOT birth-critical (requires AHK HDA running). These are operational health, not genesis identity.

---

## 3. The Five Layers

```
L1: GENESIS INTEGRITY            → Canonical authority validates
L2: PERSONAL SOVEREIGN ACTIVATION → PAT-7 / SAT-5 constitutionally alive
L3: STAND-ALONE CAPABILITY       → MVSA lifecycle gates satisfied
L4: DEVICE CONSECRATION          → Lifecycle READY, state persisted
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
| 4.2 | Lifecycle ready | `lifecycle.status == "ready"` | ready |
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

**When all 19 hard gates pass AND lifecycle.status == "ready":**
- MoMo's machine is formally born as Node0
- PAT-7 and SAT-5 are constitutionally bound
- Genesis identity is cryptographically anchored
- MVSA self-validation has proven sovereignty
- All 11 status-determining lifecycle gates are satisfied
- Mission path has been receipted (evidence chain operational)
- Restart recovery validated (artifacts survive process restart)
- Future nodes can replicate via deterministic ceremony

**Does NOT mean:** Genesis-100 readiness, public launch, multi-validator federation.

**Prerequisite for "ready":** Run `python scripts/node0_standalone.py task` to receipt a mission and trigger restart recovery. Without this, lifecycle remains `degraded` and gate 4.2 fails.

---

## 8. Document Governance

### Hierarchy (no competing truth)

| Document | Role | Changes |
|----------|------|---------|
| `NODE0_STANDALONE_READINESS.md` | **MVSA specification** — defines what gates exist, how status is computed, what "ready" means | Architect-level change |
| This DoD (v1.2) | **Birth gate** — verifies the spec is satisfied, adds ceremony/constitutional checks | SAT-5 consensus |
| `NODE0_DOD_CORRECTION_MATRIX.md` | **Audit trail** — documents rebase decisions from v1.0→v1.1→v1.2 | Append-only |

This DoD does not introduce new truth. It verifies truth defined by the MVSA spec and constitutional constants. If the spec and DoD ever conflict, the spec wins.

### Lifecycle Gate Reconciliation

`NODE0_STANDALONE_READINESS.md` defines 11 status-determining gates. The runtime JSON (`node0_lifecycle.json`) contains 14 entries because `node0_standalone.py` also writes 3 availability gates for operational health monitoring. These availability gates do NOT affect `_compute_status()` and are NOT birth-critical.

| Category | Gates | Determines status? |
|----------|-------|--------------------|
| MVSA-critical (9) | genesis_authority_valid, identity_ready, pat_sat_ready, urp_signed, urp_verified, assets_written, awareness_written, mvsa_network_bootstrap_ok, mvsa_self_validation_ok | Yes — all must be true or status = "blocked" |
| Completion (2) | mission_path_receipted, restart_recovery_ready | Yes — both must be true for status = "ready" |
| Availability (3) | desktop_bridge_reachable, mcp_available, a2a_available, telescript_available | No — informational only |

---

## 9. Document Control

| Field | Value |
|-------|-------|
| Version | 1.2 LOCKED |
| Date | 2026-03-11 |
| Lock policy | Changes require SAT-5 consensus |
| Canonical truth | `node0_lifecycle.json` (runtime) + `node0_genesis.json` (authority) |
| Birth requirement | `lifecycle.status == "ready"` (Ready Only — locked by Mumo) |
| MVSA spec | `NODE0_STANDALONE_READINESS.md` (defines gates, DoD verifies them) |
| Certification path | Linux / WSL2 (`bash scripts/node0_genesis_ceremony.sh`) |
| Changelog | v1.0→v1.1: rebased 38→19 gates; v1.1→v1.2: Ready Only enforced, governance clarified |
