# Phase 0: Symbolic Freeze — Days 1-2

**Ihsan Gate:** Adl (Justice/Consistency)
**Objective:** Lock constitutional invariants. Remove cryptographic fallbacks. Zero drift tolerance.

## Task 0.1: Golden-Vector CI

**Purpose:** Ensure Rust and Python produce identical digests for identical inputs.
Cross-language sealing is the foundation of multi-layer trust.

### Pseudocode

```
FUNCTION create_golden_vector():
    # Canonical test vector — frozen, version-controlled
    vector = {
        mission_id: "golden-vector-v1",
        initiator_id: "node0-genesis",
        payload: b"In the Name of Allah, Most Gracious, Most Merciful",
        ihsan_score: 0.9847,
        timestamp: 1711584000000,  # Fixed epoch for reproducibility
    }
    RETURN vector

FUNCTION serialize_canonical(vector) -> bytes:
    # Deterministic serialization: sorted keys, no floating-point ambiguity
    # ihsan_score serialized as fixed-point integer: round(0.9847 * 10000) = 9847
    buf = []
    buf.append(encode_utf8(vector.mission_id))
    buf.append(encode_utf8(vector.initiator_id))
    buf.append(vector.payload)
    buf.append(encode_u64_le(vector.ihsan_score * 10000))
    buf.append(encode_u64_le(vector.timestamp))
    RETURN concat(buf)

FUNCTION golden_vector_test():
    vector = create_golden_vector()
    serialized = serialize_canonical(vector)

    rust_digest = blake3_hash(serialized)     # From bizra-core
    python_digest = blake3_hash(serialized)   # From core/

    ASSERT rust_digest == python_digest,
        "FATAL: Cross-language sealing broken. Digests diverge."

    # Store canonical digest for regression detection
    ASSERT rust_digest == FROZEN_GOLDEN_DIGEST,
        "FATAL: Golden vector digest changed. Invariant drift detected."
```

### CI Integration

```yaml
# In canonical-validation-gate.yml, add:
- name: "Gate 0: Golden Vector Sealing"
  run: |
    RUST_DIGEST=$(cargo test --package bizra-core golden_vector -- --nocapture | grep DIGEST)
    PYTHON_DIGEST=$(python -m pytest tests/core/test_golden_vector.py -s | grep DIGEST)
    [ "$RUST_DIGEST" = "$PYTHON_DIGEST" ] || exit 1
```

### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `bizra-omega/bizra-core/src/golden_vector.rs` | CREATE | Rust canonical serialization + hash |
| `core/integration/golden_vector.py` | CREATE | Python canonical serialization + hash |
| `tests/core/test_golden_vector.py` | CREATE | Cross-language digest comparison |
| `.github/workflows/canonical-validation-gate.yml` | MODIFY | Add golden vector gate |

### TDD Anchors

```
TEST golden_vector_rust_produces_canonical_digest
TEST golden_vector_python_produces_canonical_digest
TEST golden_vector_digests_match_across_languages
TEST golden_vector_digest_matches_frozen_value
TEST golden_vector_rejects_modified_input
```

---

## Task 0.2: Remove Dilithium Fallback

**Purpose:** The marketing-main repo has a Dilithium signature verification that returns
`true` for ANY signature. This is a critical vulnerability — it means any forged receipt
would pass verification.

### Pseudocode

```
# BEFORE (vulnerable):
FUNCTION verify_dilithium(signature, message, public_key) -> bool:
    TRY:
        native_verify(signature, message, public_key)
        RETURN true
    CATCH:
        RETURN true  # ← CRITICAL: fallback always returns true

# AFTER (fail-closed):
FUNCTION verify_dilithium(signature, message, public_key) -> Result<bool, CryptoError>:
    TRY:
        result = native_verify(signature, message, public_key)
        RETURN Ok(result)
    CATCH NativeNotAvailable:
        # Emit error receipt documenting the failure
        emit_receipt(ReceiptArtifact {
            type: "crypto_failure",
            reason: "dilithium_native_unavailable",
            action: "verification_rejected",
            timestamp: now(),
        })
        RETURN Err(CryptoError::NativeUnavailable)
    CATCH InvalidSignature:
        RETURN Ok(false)
```

### Files to Modify

| File | Action | Purpose |
|------|--------|---------|
| `marketing-main/*/crypto.py` (or equivalent) | MODIFY | Remove `return true` fallback |
| `tests/test_crypto_fallback.py` | CREATE | Verify fallback removal |

### TDD Anchors

```
TEST dilithium_verify_rejects_invalid_signature
TEST dilithium_verify_rejects_when_native_unavailable
TEST dilithium_verify_emits_error_receipt_on_failure
TEST dilithium_verify_accepts_valid_signature
TEST dilithium_fallback_does_not_return_true
```

---

## Task 0.3: Freeze Thresholds Across Layers

**Purpose:** Constitutional invariants must be identical across all 5 layers.
Any drift = proof integrity failure.

### Pseudocode

```
# Canonical values (from BIZRA_CANONICAL.md, frozen 2026-03-26):
IHSAN_THRESHOLD     = 0.95
STRICT_IHSAN        = 0.99
ADL_GINI_THRESHOLD  = 0.35
ADL_GINI_EMERGENCY  = 0.60
SNR_THRESHOLD       = 0.85
ZAKAT_RATE          = 0.025

FUNCTION verify_threshold_sync():
    # Read from each layer
    rust_ihsan   = grep("IHSAN_THRESHOLD.*f64", "bizra-omega/bizra-core/src/lib.rs")
    python_ihsan = grep("IHSAN_THRESHOLD.*Final", "core/integration/constants.py")
    kernel_ihsan = json_read("thresholds.ihsan", ".bizra-kernel/core-context.json")
    ts_ihsan     = grep("ihsan.*0.95", "BIZRA-OS/src/api/config.ts")

    ASSERT rust_ihsan == python_ihsan == kernel_ihsan == ts_ihsan == 0.95,
        "IHSAN threshold misaligned across layers"

    # Repeat for GINI, SNR
    # ... (same pattern)

    PRINT "[ENFORCEMENT: PROVEN] All thresholds synchronized"
```

### CI Integration

This is already implemented in the `canonical-validation-gate.yml` Cross-Layer Coherence job.
Verify it catches drift by temporarily changing one value and confirming CI fails.

### TDD Anchors

```
TEST threshold_ihsan_matches_across_rust_python_kernel
TEST threshold_gini_matches_across_rust_python
TEST threshold_snr_matches_across_python_kernel
TEST threshold_drift_detected_when_one_layer_changes
```

---

## Phase 0 Exit Criteria

All of the following must be true before proceeding to Phase 1:

- [ ] Golden-vector CI passes with identical Rust/Python digests
- [ ] Dilithium fallback removed; error receipt emitted on native failure
- [ ] Redis auth aligned (DONE: 2026-03-28)
- [ ] All thresholds verified identical across layers (DONE: constitutional bridge = 1.00)
- [ ] No constitutional drift detected in CI
