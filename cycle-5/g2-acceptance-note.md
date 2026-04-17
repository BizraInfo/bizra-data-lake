# Cycle-5 — Gate G2 (Mission-runtime + Manifest) Acceptance Note

بسم الله الرحمن الرحيم

**Cycle:** 5 (Principal Activation)
**Gate:** G2 — Mission-runtime + ManifestArtifact landing on NODE0
**Sealed:** 2026-04-17 (Friday) per system `date`
**Commit:** `80c41602` (bizra-data-lake main)
**Node:** NODE0

---

## What landed

### Mission-runtime in `CognitionRuntime`
- `submit_mission(envelope, claim) -> Result<Blake3Hash, MissionRuntimeError>`
- `mission_by_id(&Blake3Hash) -> Option<&MissionRuntimeRecord>`
- `rehydrate_mission(&Blake3Hash) -> Result<MissionReplayReport, ..>`
- `mission_count() -> usize`
- Supporting types: `MissionRuntimeError`, `MissionRuntimeRecord`, `MissionReplayResult`, `MissionReplayReport`

### `manifest_artifact.rs` module (§7 Fifth Canonical Contract)
- `ManifestArtifact { manifest_id, window_start, window_end, receipt_refs, integrity_hash, receipt_count, chain_head_at_generation }`
- `from_window(...)` builder with canonical-sorted integrity hash
- `verify_integrity() -> bool`
- `ReceiptPayload` impl with `timestamp_ns() = window_end` override (tracks via `ReceiptChain::latest_timestamp`)
- `ReceiptPayloadDecode` impl for chain replay
- 5 tests (has section-7 fields, integrity verifies, deterministic, roundtrip, empty manifest valid)

## Patches applied before commit (per multi-lens audit)

| Patch | What it fixes | Constitutional impact |
|---|---|---|
| **A** | `submit_mission` now returns `Err(MissionRuntimeError::Rejected(result))` on non-Permit verdicts; no gate receipts, no final receipt | Closes NO_SHADOW_STATE leak where rejected missions sealed a NodeLifecycle receipt identical to a permit path |
| **B** | Permitted missions walk §6 sequence to **S8 Replayability** (was stamping S7 directly) | Envelope state now reflects the full chain journey; aligns with `rehydrate_mission` proof surface |
| **C** | Add `manifest_artifact` module + `timestamp_ns()` override + register in `lib.rs` | Unblocks lawful manifest surface (§7 Fifth Contract) + ReceiptChain latest_timestamp correctness |

## Audit findings deferred to follow-up arc (not blockers for G2)

- D-1: Partial-commit atomicity hole — if gate-append fails mid-loop, orphan mission envelope stays on chain with no DegradedPath compensating receipt. (Requires DegradedPath emission path + test.)
- D-2: `final_receipt.kind = NodeLifecycle` is generic. Ideally `ReceiptKind::MissionCreated` would distinguish from other lifecycle events. Blocked on adding a new ReceiptKind variant (affects serialization).
- D-3: Cross-language drift — Rust `IhsanFloorGate` hardcodes 0.95; Python SSOT has 4-tier (0.90/0.95/0.99/1.0). Track via `cross-lang-sync` scope.
- D-4: Test gaps — no explicit test for `DuplicateMission`, no partial-commit failure injection, no `mission_by_id` None path.

## Evidence

### Test counts — Rust
- `cargo test -p bizra-cognition --lib` → **64/64 green** (was 54/54 pre-G2)
- `cargo test -p bizra-cognition-gateway` → **4/4 green** (unaffected)
- `cargo test --workspace` (bizra-omega) → **all 24+ crates green**, 1,200+ tests, 0 failed

### New tests added by patches
- `submit_mission_rejects_without_canonicalizing` — verifies Patch A: reject path returns Err, chain only advances by the mission envelope (no gate verdicts, no final receipt), `missions` registry stays empty
- `submit_mission_advances_to_replayability_on_permit` — verifies Patch B: permit path ends at `MissionStage::Replayability`
- 5 tests in `manifest_artifact.rs` — verify Patch C surface

### Existing test fixed
- `submit_mission_records_chain_backed_runtime_state` — updated stage assertion from `Canonicalization` → `Replayability` to reflect new correct behavior

## Scope discipline

- Touched files: `bizra-cognition/src/{runtime.rs, manifest_artifact.rs, lib.rs}` only (3 files, +608 / −3)
- Parallel-session dirty state in `bizra-hooks/`, `bizra-node/`, `bizra-python/tests/` and ~200 files across `core/`, `bizra-node0/`, etc. — left **untouched** (Path 1 discipline)
- `Cargo.lock` unchanged (no new dependencies — manifest_artifact uses existing `canonical_hasher` and `receipts`)
- `bizra-omega/Cargo.toml` unchanged (no new crate added — manifest_artifact is a module within bizra-cognition)

## What G2 does and does not claim

**G2 claims:**
- The mission-runtime layer compiles, tests green, and respects 5 constitutional anchors on the **permit** path
- The **reject** path no longer fabricates success-shaped receipts — rejection is structurally visible via `Err(Rejected(..))`
- `ManifestArtifact` surface exists and is serializable through the chain
- No regressions in any workspace crate

**G2 does NOT claim:**
- Principal activation has happened (that's G3)
- All audit findings are closed (D-1 through D-4 are deferred)
- The mission-runtime is battle-tested beyond happy path (coverage gaps noted above)
- The slice is CANONICAL (still trending; canonicality gated on G3 per Cycle-4 definition)

## Next gate

**G3 — First principal-activation receipt:**
1. Construct `MissionEnvelope` from Mumo's exact intent: *"activate my dual agentic system"* via `MissionEnvelope::from_intent(...)`
2. Build `AdmissibilityClaim` with proper evidence_hash, quality_score ≥ 0.95
3. Call `runtime.submit_mission(envelope, claim)` → expect `Ok(mission_id)` (Permit)
4. Verify receipt in `/api/chain` via gateway
5. Render the activation state in Dema — principal sees receipted activation end-to-end

Only after G3 passes can the `bizra-cognition` + `bizra-cognition-gateway` slice be relabeled **CANONICAL** per the canonicality gate defined in Cycle-4 retrospective.

## Chain position

```
Cycle-4 afe9cc30 → Cycle-5 G1 (d5-acceptance-note, uncommitted) → Cycle-5 G2 80c41602 [this]
                                                               └─ G3 pending
```

---

Close it. Prove it. Reveal it.
