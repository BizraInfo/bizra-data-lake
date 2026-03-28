# Phase 2: Probe Rarely Fired Circuits — Days 8-10

**Ihsan Gate:** Amanah (Trustworthiness/Completeness)
**Objective:** Verify fail-closed behavior under ALL adverse conditions.
No capability may be elevated without surviving the probe set.

## Mandatory SAPE Probe Set

Seven probes that must ALL pass in CI before any claim can be promoted.

---

### Probe 1: Negative Path (Constitutional Rejection)

**Purpose:** Proposals violating constitutional constraints MUST fail closed.

```
FUNCTION probe_negative_path():
    # Create a mission that violates Ihsan threshold
    bad_mission = MissionEnvelope {
        payload: "Transfer all SEED to single account",  # Violates Gini
        ihsan_score: 0.40,  # Below 0.95 threshold
    }

    verdict = evaluate_mission(bad_mission, constitution)

    ASSERT verdict.overall_status == REJECTED
    ASSERT verdict.reject_reasons CONTAINS RejectCode::IhsanBelowThreshold
    ASSERT verdict.gate_chain[1].gate_name == "Ihsan"
    ASSERT verdict.gate_chain[1].passed == false

    # Verify rejection produces a receipt (failures are evidence too)
    receipt = get_latest_receipt()
    ASSERT receipt.action_summary CONTAINS "rejected"
    ASSERT receipt.ihsan_score == 0.40

    PRINT "[PROBE PASSED] Negative path: fail-closed verified"
```

### Probe 2: Proof Engine Timeout

**Purpose:** If the proof engine times out, the mission MUST be rejected (not deferred indefinitely).

```
FUNCTION probe_timeout():
    # Inject artificial delay into proof verification
    WITH mock_proof_engine(latency=30_000ms):  # 30s timeout trigger
        mission = create_valid_mission()
        verdict = evaluate_mission(mission, constitution)

    ASSERT verdict.overall_status == REJECTED
    ASSERT verdict.reject_reasons CONTAINS RejectCode::ProofTimeout
    # System must not hang — verify wall-clock time
    ASSERT elapsed_ms < 35_000  # Grace period above timeout

    PRINT "[PROBE PASSED] Timeout: rejection within bounds"
```

### Probe 3: Dependency Failure (Redis Down)

**Purpose:** If a critical dependency fails, the system degrades gracefully
but does NOT admit unverified missions.

```
FUNCTION probe_dependency_failure():
    # Stop Redis temporarily
    redis_stop()
    DEFER: redis_start()

    mission = create_valid_mission()

    TRY:
        verdict = evaluate_mission(mission, constitution)
        # If evaluation completes, it must be DEFERRED (not admitted)
        ASSERT verdict.overall_status IN [REJECTED, DEFERRED]
        ASSERT verdict.overall_status != ADMITTED
    CATCH ConnectionError:
        # Acceptable: system refuses to process without dependencies
        PASS

    PRINT "[PROBE PASSED] Dependency failure: no false admission"
```

### Probe 4: Replay Divergence

**Purpose:** If the same mission is replayed with different inputs,
the system MUST detect the divergence.

```
FUNCTION probe_replay_divergence():
    # Create and evaluate a mission
    mission_v1 = create_mission(payload="Organize files in ~/Documents")
    verdict_v1 = evaluate_mission(mission_v1, constitution)
    receipt_v1 = get_latest_receipt()

    # Tamper with the mission and replay
    mission_v2 = mission_v1.clone()
    mission_v2.payload = "Delete all files in ~/Documents"  # Tampered
    mission_v2.canonical_hash = mission_v1.canonical_hash   # Keep old hash

    # Verification must catch the tamper
    verify_result = mission_v2.verify()
    ASSERT verify_result.is_err()
    ASSERT verify_result.err() == EnvelopeError::IntegrityFailure

    PRINT "[PROBE PASSED] Replay divergence: tamper detected"
```

### Probe 5: Reflex Promotion with Incomplete Provenance

**Purpose:** A reflex pattern MUST NOT be promoted to the reflex cache
if it lacks complete provenance (fewer than 3 successful executions).

```
FUNCTION probe_reflex_incomplete_provenance():
    reflex_cache = ReflexCache::new()

    # Single execution — not enough for promotion
    task_hash = reflex_cache.task_signature("sort inbox")
    reflex_cache.record_execution(task_hash, now(), duration=50ms)

    # Attempt promotion
    result = reflex_cache.try_promote(task_hash)
    ASSERT result.is_err()
    ASSERT result.err() == ReflexError::InsufficientProvenance

    # Execute 2 more times (total: 3)
    reflex_cache.record_execution(task_hash, now(), duration=45ms)
    reflex_cache.record_execution(task_hash, now(), duration=48ms)

    # Now promotion should succeed
    result = reflex_cache.try_promote(task_hash)
    ASSERT result.is_ok()

    PRINT "[PROBE PASSED] Reflex provenance: 3-execution minimum enforced"
```

### Probe 6: Policy Version Mismatch

**Purpose:** If the mission's policy version doesn't match the current
constitution, the mission MUST be rejected.

```
FUNCTION probe_policy_mismatch():
    # Create mission with outdated policy version
    mission = MissionEnvelope {
        constitutional_context: ConstitutionalContext {
            policy_version: "0.87.0",  # Outdated
        },
    }

    # Current constitution is at 0.90.0+
    verdict = evaluate_mission(mission, constitution)

    ASSERT verdict.overall_status == REJECTED
    ASSERT verdict.reject_reasons CONTAINS RejectCode::PolicyVersionMismatch

    PRINT "[PROBE PASSED] Policy mismatch: outdated version rejected"
```

### Probe 7: Fallback Removal Verification

**Purpose:** Verify that the Dilithium fallback (Phase 0 fix) is truly removed
and cannot be circumvented.

```
FUNCTION probe_fallback_removal():
    # Attempt verification with invalid signature
    invalid_sig = bytes(64)  # All zeros
    message = b"test message"
    public_key = generate_test_key().public()

    result = verify_dilithium(invalid_sig, message, public_key)

    # Must fail — NOT return true
    ASSERT result.is_err() OR result == Ok(false)
    ASSERT result != Ok(true)

    # Attempt with unavailable native library
    WITH mock_native_unavailable():
        result = verify_dilithium(valid_sig, message, public_key)
        ASSERT result.is_err()
        # Verify error receipt was emitted
        receipt = get_latest_receipt()
        ASSERT receipt.type == "crypto_failure"

    PRINT "[PROBE PASSED] Fallback removed: no false positive verification"
```

---

## 24-Hour Heartbeat with Probes

```
FUNCTION heartbeat_with_probes(hours=24, interval_minutes=5):
    FOR tick IN 1..target_ticks:
        # Standard health checks
        health = sovereign_activation_check()

        # Run one probe per tick (round-robin)
        probe_index = tick % 7
        probe = PROBES[probe_index]
        probe_result = probe.execute()

        ASSERT health.ready AND probe_result.passed,
            f"Tick {tick}: health={health.ready}, probe={probe.name}: {probe_result}"

        log_heartbeat(tick, health, probe_result)
        sleep(interval_minutes * 60)

    generate_manifest(include_probe_results=true)
```

---

## Phase 2 Exit Criteria

- [ ] All 7 SAPE probes pass individually
- [ ] 24-hour heartbeat completes with probes enabled, 0 failures
- [ ] Daily manifest generated with integrity hash
- [ ] Probe results included in evidence bundle
- [ ] No false admissions detected in any probe
