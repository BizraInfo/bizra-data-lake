# Cycle-5 — G2-hardening Acceptance Note (per g2-patches-abc.md)

بسم الله الرحمن الرحيم

**Cycle:** 5 (Principal Activation)
**Gate:** G2-hardening — three constitutional corrections to the mission-runtime shipped in `80c41602` + `b031fec8`
**Sealed:** 2026-04-17 (Friday) per system `date`
**Commit:** `8b16762a` (bizra-data-lake main)
**Authority:** Founder-authored spec `g2-patches-abc.md` (dropped post-live-curl as the authoritative reference)

---

## Why this gate exists

After G2 (`80c41602`) and G3a (`b031fec8`) landed, the founder dropped the authoritative Patch-A/B/C spec. A direct read revealed that the shipped implementation diverged from the spec in three constitutional ways:

1. **Patch A semantics** — shipped code returned `Err(MissionRuntimeError::Rejected(AdmissibilityResult))`. Spec required `Ok(MissionRuntimeRecord { rejected=true, receipt_id=None, stage=Admissibility })` with the rejection preserved in the registry as derived state per §10. Shipped behavior lost the rejection (registry stayed empty) and appended the mission envelope to the chain before evaluation — which left a rejected claim visible on the chain of source truth.
2. **Patch B semantics** — shipped code unconditionally advanced to `S8 Replayability` after chain append. Spec required S8 advancement ONLY when decode round-trip succeeds (`ReceiptPayloadDecode::from_canonical_bytes` returns equivalent receipt_id). Shipped behavior could over-claim replayability on a corrupted encode/decode.
3. **Patch C completeness** — shipped `manifest_artifact.rs` lacked two identity hardenings: `chain_head_at_generation` + `receipt_count` were operational metadata but NOT bound into `manifest_id` derivation, and `receipt_refs` were sorted but not deduplicated.

G2-hardening addresses all three.

## What changed in `8b16762a`

### `bizra-cognition/src/runtime.rs`

- `submit_mission` signature changed from `Result<Blake3Hash, MissionRuntimeError>` to `Result<MissionRuntimeRecord, MissionRuntimeError>`. Rejection is not an error — it is structured state returned via `Ok(record)`.
- `MissionRuntimeError::Rejected` variant **removed**.
- `MissionRuntimeRecord` gained four fields: `rejected: bool`, `receipt_id: Option<Blake3Hash>`, `stage: MissionStage`, `timestamp_ns: u64`. Existing `mission_payload_hash` became `Option<Blake3Hash>` and `final_receipt` became `Option<ReceiptArtifact>`.
- Eval-first ordering: admissibility is evaluated **before** any chain mutation. On reject, chain is **not touched**. On permit, chain receives the mission envelope, 5 gate verdicts, and 1 final `NodeLifecycle` receipt in that order.
- Reject path stores a record in the `missions` registry with `rejected=true`, `stage=Admissibility`, no chain footprint. Queryable via `mission_by_id`.
- Permit path advances stages S4→S5→S6→S7 unconditionally via `advance_stage()`. S7→S8 `Replayability` is gated on: fetch payload bytes, decode as `ReceiptArtifact`, verify `decoded.receipt_id == receipt_id`. If decode fails, stage stays at S7.

### `bizra-cognition/src/manifest_artifact.rs`

- `from_window(...)` now calls `receipt_refs.dedup()` after `sort()`. Duplicate references no longer inflate `receipt_count` dishonestly.
- `manifest_id` derivation now binds `chain_head_at_generation` (32 bytes) + `receipt_refs.len() as u32` (4 bytes) into the hash input, not just `window_start || window_end || integrity_hash`. All fields that affect identity are now in the identity.
- `timestamp_ns()` override to `window_end` preserved from earlier landing.

### `bizra-cognition/src/admissibility_freeze_v1.rs`

- `AdmissibilityResult` gained `#[derive(Clone)]`. Required so `MissionRuntimeRecord` (now `Clone`) can be inserted into the registry and returned to the caller without moving.

### `bizra-cognition-gateway/src/main.rs`

- `post_mission` handler refactored to branch on `record.rejected` instead of matching `MissionRuntimeError::Rejected(...)`. Permit path returns HTTP 200 with full `SubmitMissionResponse`. Reject path returns HTTP 422 with structured `error.admissibility` + `error.admissibility.rejected` (RejectedClaim with `invariant`, `reason`, `remediationPath`, `escalationAllowed`).

### Tests

- `submit_mission_records_chain_backed_runtime_state` updated to assert new `receipt_id`, `rejected=false`, `stage=Replayability` fields.
- `submit_mission_rejects_without_canonicalizing` replaced with `submit_mission_rejects_without_canonicalizing_and_preserves_in_registry` — asserts chain length unchanged, `rejected=true`, `receipt_id=None`, `stage=Admissibility`, **and** registry lookup returns the rejected record.
- `submit_mission_advances_to_replayability_on_permit` updated for new return type.
- All 5 manifest_artifact tests pass with the hardened identity derivation.

## Live-curl end-to-end verification (hardened contract)

With release binary running from `target/release/bizra-cognition-gateway`:

**PERMIT path — `"activate my dual agentic system"`, qualityScore=0.98:**
```
missionId:  721c713cfaa63402c9fb3a3d4151f4242209bb45e54d39773aeac6a6f281555e
receiptId:  38037484093a2deb62424b9df46c8b39a1ad7266e8141dc9c2fa2646ea9e5c0f
verdict:    Permit
gates:      5 (all Permit)
finalStage: Replayability
chainHead:  == receiptId
chain.length: 7 (1 mission + 5 gate verdicts + 1 final NodeLifecycle receipt)
```

**REJECT path — qualityScore=0.5 (below IHSAN_FLOOR 0.95):**
```
HTTP:       422
error.code: ADMISSIBILITY_REJECTED
verdict:    Reject
rejected.invariant:    IHSAN_FLOOR
rejected.reason:       "IHSAN_FLOOR violation: score 0.5000 below floor 0.9500"
rejected.remediationPath: "Improve claim quality score to ≥ 0.95..."
chain.length AFTER reject: 7 (UNCHANGED — §10 Proof Law)
```

The second assertion is the critical one. **A rejected claim leaves zero footprint on the chain.** This is the constitutional invariant the hardening was built to restore.

## Evidence

| Check | Result |
|---|---|
| `cargo test -p bizra-cognition --lib` | 64/64 green |
| `cargo test -p bizra-cognition-gateway` | 7/7 green |
| `cargo clippy -p bizra-cognition -p bizra-cognition-gateway --no-deps` | 0 errors on session crates |
| Live PERMIT curl | receipt sealed, chain length 7 |
| Live REJECT curl | HTTP 422, chain length stays 7 |
| Integrator cross-artifact cohesion | §10 Proof Law structurally upheld |

## Constitutional fidelity (end-to-end, hardened)

| Anchor | How G2-hardening reinforces it |
|---|---|
| **ZANN_ZERO** | Claim must carry evidence (gate 1). Unchanged. |
| **CLAIM_MUST_BIND** | On permit, mission envelope binds claim to chain before gate verdicts. On reject, no claim enters chain at all — no unbound claims possible. |
| **RIBA_ZERO** | EconomicPattern check (gate 3). Unchanged. |
| **NO_SHADOW_STATE** | Registry holds both permitted and rejected records, but ONLY permitted missions advance the chain. The chain never contains a rejection receipt that could be misread as success. |
| **IHSAN_FLOOR** | 0.95 floor enforced (gate 5). On reject, structured `RejectedClaim.remediationPath` tells the operator exactly how to raise quality. |

## What G2-hardening claims and does not claim

**Claims:**
- The reject-path lie is structurally gone at the canonical runtime level
- The §10 Proof Law (chain = source truth, only lawful completions) holds byte-for-byte
- The S8 Replayability stage can no longer be claimed without decode verification
- The ManifestArtifact identity is tamper-robust against duplicate-ref attacks

**Does not claim:**
- All audit findings are closed — D-1 (partial-commit DegradedPath emission), D-2 (Mission-specific ReceiptKind variant), D-4 (test gaps on DuplicateMission, partial-commit, mission_by_id None) remain deferred
- Cross-language drift resolved — Rust `IhsanFloorGate` still hardcodes 0.95; Python SSOT has 4-tier. Future `cross-lang-sync` arc.

## Chain position

```
Cycle-5
  G1 (D5)                 ✅ cycle-5/d5-acceptance-note.md
  G2 (mission-runtime)    ✅ 80c41602 + cycle-5/g2-acceptance-note.md
  G3a (gateway v0.2)      ✅ b031fec8
  G3b (Next.js proxy)     ✅ 40a6832 + cycle-5/g3-acceptance-note.md
  G2-hardening            ✅ 8b16762a + THIS note
  G4 (browser)            ▶ pending — Mumo's Daughter Test
```

## Why this note belongs in the cycle-5/ directory

G2-hardening is not a separate cycle — it is the constitutional completion of G2's intent. Filing it here (rather than in a retrospective) preserves the truth that the first G2 commit was incomplete and was corrected by the founder's own authoritative spec. Per Cycle-4's `PROVEN` definition, incomplete code shipped under `feat(...)` is legitimate if a subsequent `fix(...)` completes it within the same cycle. The session self-corrected. The chain records both commits. The audit trail is honest.

---

Close it. Prove it. Reveal it. (It was close. Then it was proven. Then it was revealed. Then the spec arrived, and the whole cycle ran again, tighter.)
