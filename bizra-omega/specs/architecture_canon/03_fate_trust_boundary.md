# 03 — FATE Trust Boundary

> FATE = Fairness, Accountability, Transparency, Ethics.
> The judiciary between user domain (PAT) and system domain (SAT/URP).
> Only proof-carrying requests cross. No raw authority leaks.

## Gate Layer Map

```
FATE gates (this spec, crossing boundary):    PCI gates (bizra-core, local validation):
  Gate::Ihsan    (>= 0.95)                      SchemaGate   (structural validity)
  Gate::Adl      (Gini <= 0.35)                 IhsanGate    (quality/ethics score)
  Gate::Guardian  (constitutional approval)      SNRGate      (signal-to-noise)

FATE runs AFTER PCI. PCI validates locally within PAT. FATE validates at the crossing.
Both are fail-closed, fixed-order, non-skippable.
Code ref: bizra-core/src/pci/gates.rs:232 (default_gate_chain)
Code ref: bizra-core/src/topology_canon.rs:149 (GATE_ORDER = ["Schema","Ihsan","SNR"])
```

## Pseudocode: FATE Gate Chain

```
CONST GATE_ORDER: [Gate; 3] = [
    Gate::Ihsan,      // Excellence threshold (>= 0.95)
    Gate::Adl,        // Justice / fairness (Gini <= 0.35)
    Gate::Guardian,    // Constitutional guardian approval
]

STRUCT ProofCarryingRequest:
    origin_node:     NodeId              // who made this
    proof_trace:     ProofTrace          // from PAT mission
    ihsan_score:     f64                 // computed by P4_Judge
    snr_score:       f64                 // signal-to-noise
    crown_verdict:   Verdict             // P5_Crown user-side check
    signature:       Ed25519Signature    // signed by node's key
    receipt_hash:    BLAKE3Hash          // integrity seal
    timestamp:       u64                 // when created
    chain_link:      Option<BLAKE3Hash>  // previous receipt (chain)

FUNCTION wrap_as_proof_request(
    mission_result: MissionResult,
    signing_key:    Ed25519SigningKey,
    previous:       Option<ReceiptHash>,
) -> ProofCarryingRequest:
    // Only called when local mission wants to cross into URP

    // Step 1: Verify local quality first
    ASSERT mission_result.ihsan >= IHSAN_THRESHOLD  // 0.95
    ASSERT mission_result.snr   >= SNR_THRESHOLD    // 0.85
    ASSERT mission_result.crown_ok == PASS

    // Step 2: Build receipt
    receipt = MissionReceipt::new()
    receipt.set_scores(mission_result.ihsan, mission_result.snr)
    receipt.set_model(mission_result.model_used)
    receipt.set_states(mission_result.states_traversed)
    receipt.chain_to(previous)
    receipt.compute_hash()       // BLAKE3
    receipt.sign(signing_key)    // Ed25519

    // Step 3: Wrap as proof-carrying request
    RETURN ProofCarryingRequest {
        origin_node:   derive_node_id(signing_key),
        proof_trace:   mission_result.proof_trace,
        ihsan_score:   mission_result.ihsan,
        snr_score:     mission_result.snr,
        crown_verdict: mission_result.crown_ok,
        signature:     receipt.signature,
        receipt_hash:  receipt.hash,
        timestamp:     now(),
        chain_link:    previous,
    }
```

## Pseudocode: FATE Admissibility Check

```
FUNCTION fate_admit(request: ProofCarryingRequest) -> FateVerdict:
    // Run gate chain in ORDER. Fail-closed.

    // Gate 1: Ihsan (Excellence)
    IF request.ihsan_score < IHSAN_THRESHOLD:
        RETURN FateVerdict::Reject("Ihsan below threshold")

    // Gate 2: Adl (Justice/Fairness)
    gini = compute_post_transaction_gini(request)
    IF gini > ADL_GINI_THRESHOLD:
        RETURN FateVerdict::Reject("Gini ceiling violated")

    // Gate 3: Guardian (Constitutional)
    IF request.crown_verdict != PASS:
        RETURN FateVerdict::Reject("Crown veto")

    // Cryptographic integrity
    IF NOT verify_signature(request.signature, request.origin_node):
        RETURN FateVerdict::Reject("Invalid Ed25519 signature")

    IF NOT verify_hash(request.receipt_hash):
        RETURN FateVerdict::Reject("BLAKE3 hash mismatch")

    // Chain integrity (if chained)
    IF request.chain_link IS Some(prev_hash):
        IF NOT chain_valid(prev_hash, request.receipt_hash):
            RETURN FateVerdict::Reject("Chain link broken")

    RETURN FateVerdict::Admit(request)

ENUM FateVerdict:
    Admit(ProofCarryingRequest)  // passes to SAT for system validation
    Reject(String)               // halts — no network effect
```

## Fail-Closed Principle

```
INVARIANT fate_fail_closed:
    // If ANY gate is uncertain, the default is REJECT.
    // Constitution wins over convenience.
    // No request crosses without ALL gates passing.

    IF gate_result IS Unknown OR Error:
        RETURN Reject("Fail-closed: uncertain gate")

    // Gate order is fixed. Cannot be reordered or skipped.
    FOR gate IN GATE_ORDER:
        result = gate.evaluate(request)
        IF result != PASS:
            RETURN Reject(gate.name + ": " + result.reason)
```

## What Crosses vs What Stays Local

```
CROSSES FATE BOUNDARY:              STAYS LOCAL (never crosses):
- proof traces (hashed)              - raw user data
- quality scores (ihsan, snr)        - conversation content
- receipt hashes                     - local memory atoms
- Ed25519 signatures                 - model weights
- chain links                        - private keys
- agent attestations                 - PAT internal state
- SEED settlement requests           - local file paths
```

## TDD Anchors

```
TEST fate_rejects_low_ihsan:
    request = make_request(ihsan=0.80)
    verdict = fate_admit(request)
    ASSERT verdict IS Reject
    ASSERT verdict.reason CONTAINS "Ihsan"

TEST fate_rejects_high_gini:
    request = make_request(gini_post=0.50)
    verdict = fate_admit(request)
    ASSERT verdict IS Reject
    ASSERT verdict.reason CONTAINS "Gini"

TEST fate_rejects_invalid_signature:
    request = make_request(signature=FORGED)
    verdict = fate_admit(request)
    ASSERT verdict IS Reject

TEST fate_rejects_broken_chain:
    request = make_request(chain_link=WRONG_HASH)
    verdict = fate_admit(request)
    ASSERT verdict IS Reject

TEST fate_admits_valid_request:
    request = make_valid_request()
    verdict = fate_admit(request)
    ASSERT verdict IS Admit

TEST gate_order_is_fixed:
    ASSERT GATE_ORDER == [Ihsan, Adl, Guardian]
    ASSERT GATE_ORDER.len() == 3

TEST fate_never_leaks_raw_data:
    request = wrap_as_proof_request(mission_result, key, None)
    ASSERT request.proof_trace DOES NOT CONTAIN raw_content
    ASSERT request HAS NO field named "user_data"
    ASSERT request HAS NO field named "conversation"
```
