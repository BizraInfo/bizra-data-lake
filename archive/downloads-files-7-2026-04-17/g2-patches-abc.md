# G2 Closure Patches — A + B + C

بسم الله الرحمن الرحيم

**Authority:** Manifest v0.2 §3 (NO_SHADOW_STATE), §6 (Canonical Runtime Flow), §7 (ManifestArtifact)
**Target:** bizra-cognition crate on NODE0 (runtime.rs @ 915 lines)
**Gate:** G2 (Cycle-5)

---

## Patch A — Fix reject-path canonicalization (CONSTITUTIONAL)

### The bug

When `submit_mission()` evaluates a claim and the AdmissibilityChain returns
REJECT, the current code may still:
1. Append a receipt to the chain (the GateVerdict with REJECT verdict)
2. Store the mission in the registry with a "sealed" stage marker
3. Return the mission record to the caller

This means a rejected mission produces a chain-persisted receipt that, when
viewed through the gateway or Dema, looks like a "sealed" operation. The Face
layer cannot distinguish "sealed because PERMIT + execution succeeded" from
"sealed because REJECT was recorded."

This is a **NO_SHADOW_STATE violation** — the operator surface would reveal
a sealed receipt for a mission that was actually denied.

### The fix

In `submit_mission()`, when admissibility returns REJECT:

1. Do NOT append a proof-bearing receipt to the canonical chain
2. DO store the RejectedClaim in a separate rejection log (not the receipt chain)
3. Set the mission stage to `MissionStage::Admissibility` (the stage where it stopped)
4. Set a `rejected: bool` flag on MissionRuntimeRecord
5. Return the record with the rejection clearly distinguishable from success

```rust
// PATCH A — apply inside submit_mission(), at the REJECT branch

if adm_result.verdict != Verdict::Permit {
    // CONSTITUTIONAL: rejected missions do NOT produce canonical receipts.
    // The rejection is recorded in the mission registry (derived state),
    // NOT in the receipt chain (source truth). This preserves:
    //   - NO_SHADOW_STATE: chain only contains lawful completions
    //   - CLAIM_MUST_BIND: no receipt exists for unproven claims
    //   - Chain-is-truth (§10): chain reflects what actually happened (rejection)
    //     by ABSENCE, not by presence of a "rejection receipt"
    
    let record = MissionRuntimeRecord {
        envelope: envelope.clone(),
        stage: MissionStage::Admissibility, // stopped here
        admissibility: Some(adm_result),
        receipt_id: None,                    // NO receipt — mission was denied
        rejected: true,                      // explicit flag
        timestamp_ns,
    };
    
    self.missions.insert(envelope.mission_id, record.clone());
    
    return Ok(record); // caller sees rejected=true, receipt_id=None
}
```

### Why rejection receipts are wrong

The intuition "receipt everything for audit trail" is understandable but
constitutionally incorrect for BIZRA:

- §10 (Proof Law): "The chain is source truth." If rejected missions
  produce chain records, the chain asserts things happened that didn't.
- §3 P4 (CLAIM_MUST_BIND): A rejected claim has no binding evidence of
  execution. A receipt for it would be an unbound claim in the chain.
- §11 (UX Law): Dema must reveal truth. A receipt for a rejected mission
  would be revealed as "sealed" — which is a lie.

The rejection IS still recorded — in the mission registry (derived state,
§10: "graph is derived"). It can be queried via `mission_by_id()`. It just
doesn't enter the chain.

### MissionRuntimeRecord needs a `rejected` field

```rust
pub struct MissionRuntimeRecord {
    pub envelope: MissionEnvelope,
    pub stage: MissionStage,
    pub admissibility: Option<AdmissibilityResult>,
    pub receipt_id: Option<Blake3Hash>,  // None for rejected missions
    pub rejected: bool,                  // NEW — explicit rejection flag
    pub timestamp_ns: u64,
}
```

---

## Patch B — Stage advancement truthfulness

### The issue

After a successful submit_mission(), the mission stage should advance to
`MissionStage::Replayability` (S8) — but only if replay verification
actually succeeds. Currently the stage may be set to Canonicalization (S7)
without confirming replay.

### The fix

After appending the receipt artifact to the chain:

```rust
// PATCH B — after chain.append_artifact(receipt) succeeds

// S7: Canonicalization — receipt is in the chain
envelope.stage = MissionStage::Canonicalization;

// S8: Replayability — verify the receipt can be decoded back
let replay_ok = match self.chain.fetch_payload_bytes(&receipt_id) {
    Ok(Some(bytes)) => {
        match <ReceiptArtifact as ReceiptPayloadDecode>::from_canonical_bytes(&bytes) {
            Ok(decoded) => decoded.receipt_id == receipt_id,
            Err(_) => false,
        }
    }
    _ => false,
};

if replay_ok {
    envelope.stage = MissionStage::Replayability; // S8 confirmed
} else {
    // Replay failed — mission stays at S7 (Canonicalization)
    // This is a degraded path, not a failure. The receipt IS in the chain,
    // but we can't confirm decode round-trip. Log it.
    // DO NOT advance to S8 — that would be overclaiming.
}

let record = MissionRuntimeRecord {
    envelope: envelope.clone(),
    stage: envelope.stage,
    admissibility: Some(adm_result),
    receipt_id: Some(receipt_id),
    rejected: false,
    timestamp_ns,
};
```

---

## Patch C — ManifestArtifact landing

### Action

1. Copy `manifest_artifact.rs` from Downloads to `bizra-cognition/src/`
2. Add `pub mod manifest_artifact;` to `lib.rs`
3. The file already contains the three hardenings from the review:
   - chain_head_at_generation bound into manifest_id derivation
   - `fn timestamp_ns(&self) -> u64 { self.window_end }` override
   - `receipt_refs.dedup()` after sort

### Verification

```bash
cargo test -p bizra-cognition --lib manifest_artifact 2>&1 | tail -5
# Expected: 5 tests, all green
```

### Why this is needed now

§16 success condition #5: "one daily manifest includes it." Without
ManifestArtifact, there is no typed contract for this condition. The
gateway cannot project manifest data, and Dema's daily-manifest panel
has nothing to reveal.

---

## Post-ABC verification

After all three patches:

```bash
# Full crate test
cargo test -p bizra-cognition --lib 2>&1 | tail -3
# Expected: 63+ tests (58 existing + 5 manifest), 0 failures

# Gateway test (unchanged but must not regress)
cargo test -p bizra-cognition-gateway 2>&1 | tail -3
# Expected: 4 tests, 0 failures

# Clippy (session crates)
cargo clippy -p bizra-cognition -p bizra-cognition-gateway --no-deps 2>&1 | grep "^error"
# Expected: no errors
```

When all pass: **G2 is GREEN. Commit as one focused commit.**

Suggested commit message:
```
fix(cognition): reject-path canonicalization + stage truthfulness + ManifestArtifact

A: Rejected missions no longer produce canonical receipts (NO_SHADOW_STATE fix)
B: Stage advances to Replayability only after confirmed decode round-trip
C: ManifestArtifact contract (§7) with integrity hash, timestamp override, dedup

Closes G2 of Cycle-5.
```

---

## After G2: the path to G3

With A+B+C landed, the gateway write path becomes safe to open:

1. POST `/missions` → constructs MissionEnvelope from operator intent →
   calls `runtime.submit_mission()` → returns mission record
2. If rejected=true: Dema shows rejection reason + remediation path
3. If receipt_id=Some: Dema shows sealed receipt + chain position

The first mission through this path is:
**"Activate my dual agentic system as Node0 principal"**

That's G3. That's the moment.
