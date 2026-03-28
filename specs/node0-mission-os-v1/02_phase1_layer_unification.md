# Phase 1: Layer Unification — Days 3-7

**Ihsan Gate:** Excellence (Itqan)
**Objective:** Four cross-layer contracts become the ONLY cross-boundary communication path.

## Task 1.1: MissionEnvelope Contract

**Purpose:** Every mission entering the system must be wrapped in a canonical envelope
that is identical regardless of which layer creates it.

### Schema (Frozen)

```
MissionEnvelope {
    mission_id:             String          # UUID v4, unique per mission
    initiator_id:           String          # Node identity (Ed25519 public key hex)
    payload:                MissionPayload  # Task description + context
    constitutional_context: ConstitutionalContext {
        ihsan_threshold:    f64             # 0.95 (from constitution)
        snr_threshold:      f64             # 0.85
        gini_threshold:     f64             # 0.35
        policy_version:     String          # Semantic version of policy bundle
    }
    created_at:             u64             # Unix timestamp ms
    expires_at:             u64             # TTL enforcement
    canonical_hash:         [u8; 32]        # BLAKE3 of all fields above
}
```

### Pseudocode — Rust Implementation

```
FUNCTION MissionEnvelope::new(initiator, payload, constitution) -> Self:
    envelope = MissionEnvelope {
        mission_id: uuid_v4(),
        initiator_id: initiator.public_key_hex(),
        payload: payload,
        constitutional_context: ConstitutionalContext::from(constitution),
        created_at: now_ms(),
        expires_at: now_ms() + constitution.mission_ttl_ms,
        canonical_hash: [0; 32],  # Placeholder
    }
    envelope.canonical_hash = envelope.compute_hash()
    RETURN envelope

FUNCTION MissionEnvelope::compute_hash() -> [u8; 32]:
    # Deterministic serialization (same as golden vector)
    buf = serialize_canonical(self)
    RETURN blake3(buf)

FUNCTION MissionEnvelope::verify(&self) -> Result<(), EnvelopeError>:
    IF self.expires_at < now_ms():
        RETURN Err(EnvelopeError::Expired)
    IF self.compute_hash() != self.canonical_hash:
        RETURN Err(EnvelopeError::IntegrityFailure)
    RETURN Ok(())
```

### Pseudocode — Python Implementation

```
FUNCTION MissionEnvelope.create(initiator, payload, constitution) -> MissionEnvelope:
    envelope = MissionEnvelope(
        mission_id=str(uuid4()),
        initiator_id=initiator.public_key_hex(),
        payload=payload,
        constitutional_context=ConstitutionalContext.from_constitution(constitution),
        created_at=now_ms(),
        expires_at=now_ms() + constitution.mission_ttl_ms,
    )
    envelope.canonical_hash = envelope.compute_hash()
    RETURN envelope

    # compute_hash MUST use same serialization as Rust
    # Verified by golden-vector CI (Phase 0)
```

### TDD Anchors

```
TEST envelope_creation_produces_valid_hash
TEST envelope_verification_fails_on_tamper
TEST envelope_verification_fails_on_expiry
TEST envelope_rust_python_produce_identical_hashes
TEST envelope_serialization_is_deterministic
```

---

## Task 1.2: GateVerdict Contract

**Purpose:** The output of every gate evaluation is a structured verdict that travels
unchanged across layer boundaries.

### Schema (Frozen)

```
GateVerdict {
    mission_id:          String           # Links to MissionEnvelope
    gate_chain:          Vec<GateResult>  # Ordered results per gate
    overall_status:      VerdictStatus    # ADMITTED | REJECTED | DEFERRED
    proof_status:        ProofStatus      # VERIFIED | UNVERIFIABLE | PENDING
    ihsan_score:         f64              # Measured Ihsan score
    snr_score:           f64              # Measured SNR
    reject_reasons:      Vec<RejectCode>  # Empty if admitted
    policy_version:      String           # Must match envelope's policy
    evaluated_at:        u64              # Timestamp
    verdict_hash:        [u8; 32]         # BLAKE3 of all fields
}

GateResult {
    gate_name:  String       # "Schema" | "Ihsan" | "SNR"
    passed:     bool
    latency_ns: u64
    reason:     Option<RejectCode>
}

VerdictStatus = ADMITTED | REJECTED | DEFERRED
ProofStatus   = VERIFIED | UNVERIFIABLE | PENDING
```

### Pseudocode — Gate Chain Execution

```
FUNCTION evaluate_mission(envelope: MissionEnvelope, constitution: Constitution) -> GateVerdict:
    # Gate chain order: Schema → Ihsan → SNR (ethics-first)
    gates = [SchemaGate, IhsanGate, SNRGate]
    results = []

    FOR gate IN gates:
        start = monotonic_ns()
        result = gate.evaluate(envelope, constitution)
        elapsed = monotonic_ns() - start

        gate_result = GateResult {
            gate_name: gate.name(),
            passed: result.is_ok(),
            latency_ns: elapsed,
            reason: result.err(),
        }
        results.append(gate_result)

        # Fail-closed: stop on first failure
        IF NOT gate_result.passed:
            RETURN GateVerdict {
                mission_id: envelope.mission_id,
                gate_chain: results,
                overall_status: REJECTED,
                reject_reasons: [gate_result.reason],
                ...
            }

    # All gates passed
    RETURN GateVerdict {
        mission_id: envelope.mission_id,
        gate_chain: results,
        overall_status: ADMITTED,
        proof_status: VERIFIED,
        ihsan_score: extract_ihsan(results),
        snr_score: extract_snr(results),
        reject_reasons: [],
        ...
    }
```

### TDD Anchors

```
TEST gate_verdict_admits_valid_mission
TEST gate_verdict_rejects_low_ihsan
TEST gate_verdict_rejects_invalid_schema
TEST gate_verdict_fails_closed_on_first_failure
TEST gate_verdict_hash_is_deterministic
TEST gate_verdict_policy_version_must_match_envelope
```

---

## Task 1.3: ReceiptArtifact Contract

**Purpose:** Every evaluated transition produces an immutable receipt that forms
the evidence chain.

### Schema (Frozen)

```
ReceiptArtifact {
    receipt_id:      String       # UUID v4
    mission_id:      String       # Links to MissionEnvelope
    verdict_hash:    [u8; 32]     # Links to GateVerdict
    state_before:    [u8; 32]     # Hash of system state pre-transition
    state_after:     [u8; 32]     # Hash of system state post-transition
    ihsan_score:     f64          # Constitutional quality score
    channel:         Channel      # Which execution channel
    action_summary:  String       # Human-readable description
    signature:       [u8; 64]     # Ed25519 signature (node identity)
    previous_hash:   [u8; 32]     # Hash-chain link to prior receipt
    timestamp:       u64          # Unix ms
    receipt_hash:    [u8; 32]     # BLAKE3 of all fields above
}
```

### Pseudocode — Receipt Chain

```
FUNCTION emit_receipt(mission, verdict, state_before, state_after, action) -> ReceiptArtifact:
    receipt = ReceiptArtifact {
        receipt_id: uuid_v4(),
        mission_id: mission.mission_id,
        verdict_hash: verdict.verdict_hash,
        state_before: state_before,
        state_after: state_after,
        ihsan_score: verdict.ihsan_score,
        channel: action.channel(),
        action_summary: action.summary(),
        signature: [0; 64],         # Filled after hash
        previous_hash: chain.head_hash(),
        timestamp: now_ms(),
        receipt_hash: [0; 32],      # Filled next
    }

    receipt.receipt_hash = receipt.compute_hash()
    receipt.signature = node_identity.sign(receipt.receipt_hash)

    chain.append(receipt)
    RETURN receipt

FUNCTION verify_receipt(receipt, chain, node_public_key) -> bool:
    # 1. Hash integrity
    IF receipt.compute_hash() != receipt.receipt_hash:
        RETURN false
    # 2. Signature verification
    IF NOT ed25519_verify(receipt.signature, receipt.receipt_hash, node_public_key):
        RETURN false
    # 3. Chain linkage
    IF receipt.previous_hash != chain.expected_previous():
        RETURN false
    RETURN true
```

### TDD Anchors

```
TEST receipt_hash_is_deterministic
TEST receipt_signature_verifies_with_node_key
TEST receipt_chain_linkage_valid
TEST receipt_tamper_detected
TEST receipt_replay_produces_identical_hash
```

---

## Task 1.4: Make Reflex Default-Live

**Purpose:** Remove `BIZRA_CLOSED_LOOP_ENABLED` feature flag.
The reflex system is harness-proven (126.7x speedup). It should be the default.

### Pseudocode

```
# BEFORE:
FUNCTION should_use_reflex() -> bool:
    RETURN env("BIZRA_CLOSED_LOOP_ENABLED") == "true"

# AFTER:
FUNCTION should_use_reflex() -> bool:
    # Reflex is default-live. Disable only for debugging.
    IF env("BIZRA_DISABLE_REFLEX") == "true":
        RETURN false
    RETURN true

# Status semantics tightened:
FUNCTION system_status() -> SystemHealth:
    health = compute_health()
    # Require at least 1 compiled reflex for OPERATIONAL status
    IF health.compiled_reflexes == 0:
        health.status = WARMING_UP
    ELSE:
        health.status = OPERATIONAL
    RETURN health
```

### TDD Anchors

```
TEST reflex_enabled_by_default
TEST reflex_can_be_disabled_via_env
TEST system_status_warming_up_when_no_reflexes
TEST system_status_operational_when_reflexes_compiled
TEST reflex_hit_rate_tracked_in_health
```

---

## Phase 1 Exit Criteria

- [ ] MissionEnvelope contract implemented in Rust and Python
- [ ] GateVerdict contract implemented with fail-closed semantics
- [ ] ReceiptArtifact contract with Ed25519 signatures
- [ ] ManifestArtifact bundles receipts with integrity hash
- [ ] Reflex default-live (feature flag removed)
- [ ] All e2e tests use cross-layer contracts
- [ ] Replay parity verified (positive + negative paths)
