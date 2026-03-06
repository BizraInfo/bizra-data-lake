# Phase 00 — Framework Overview: Unified Compliance Kernel

> Source: BIZRA Quality Standards & Certification Framework
> Standards: ISO 25010, CMMI Level 5, SOC 2 Type II, ISO 9001
> Status: SPECIFICATION SEALED | SNR: 0.95

---

## 1. Functional Requirements

### FR-C00: Unified Evidence Model

Every action in BIZRA produces a cryptographically signed **ActionReceipt** (`core/constitutional/types.py`). This receipt is the atomic unit of compliance evidence across all four standards:

| Receipt Field | ISO 25010 | CMMI L5 | SOC 2 | ISO 9001 |
|:---|:---|:---|:---|:---|
| `receipt_id` (BLAKE3) | Reliability | Traceability | Processing Integrity | Evidence-based |
| `intent_score` (Ghazali) | Functional Suitability | Causal Analysis | — | Customer Focus |
| `efficiency_score` | Performance | Quantitative Mgmt | — | Process Approach |
| `impact_score` | — | Process Improvement | — | Engagement |
| `reproducibility_score` | Reliability | Quantitative Mgmt | Processing Integrity | Evidence-based |
| `oracle_signature` (Ed25519) | Security | — | Security | — |
| `metadata_hash` (BLAKE3) | — | — | Confidentiality | — |

### FR-C01: Continuous Audit Pipeline

Unlike traditional annual audits, BIZRA runs continuous compliance verification:

1. **Pre-Execution**: FATE Gate checks formal/alignment/testing/ethical constraints.
2. **Execution**: Action produces ActionReceipt with all scores.
3. **Post-Execution**: Ihsan Wall validates quality floor (0.95 production).
4. **Periodic**: Gini Attractor checks economic homeostasis (every block).
5. **Aggregate**: Compliance Report Generator produces standard-specific evidence packs.

### FR-C02: Fixed-Point Determinism Guarantee

All compliance scores use fixed-point integer arithmetic (`core/constitutional/fixed_point.py`):
- **FP_PRECISION** = 1,000,000 (6 decimal places)
- **FP_ONE** = 1,000,000
- **FP_MAX** = 2^63 - 1
- Zero floating-point dependency — byte-identical results across ARM, x86, RISC-V.
- This is the foundation of ISO 25010 Reliability and SOC 2 Processing Integrity.

### FR-C03: Standard-Specific Evidence Packs

Each standard requires a different evidence format:

| Standard | Evidence Pack | Generation Frequency |
|:---|:---|:---|
| ISO 25010 | Quality Characteristics Report | Per-release |
| CMMI Level 5 | Process Performance Baseline + Improvement Trend | Continuous (weekly roll-up) |
| SOC 2 Type II | Control Effectiveness Over Period | Continuous (quarterly report) |
| ISO 9001 | QMS Audit Report | Continuous (annual certification) |

### FR-C04: Cross-Standard Invariants

Seven constitutional invariants apply to ALL four standards simultaneously:
1. **I-1**: No action without FATE Gate approval.
2. **I-2**: All arithmetic is fixed-point deterministic.
3. **I-3**: All events are append-only with Merkle chain.
4. **I-4**: Gini coefficient must remain <= 0.35.
5. **I-5**: Ihsan floor is enforced (0.95 production).
6. **I-6**: Every node is sovereign over its data.
7. **I-7**: Crown verification (H0/H1/H2) is mandatory for state transitions.

---

## 2. Edge Cases

- **EC-C01**: Standard requirements conflict (e.g., ISO 25010 Performance vs. SOC 2 Security) -> Apply H0/H1/H2 Crown hierarchy: Safety > Ethics > Performance.
- **EC-C02**: Evidence pack generation fails mid-cycle -> Replay from last Merkle checkpoint, regenerate from event log.
- **EC-C03**: Fixed-point overflow in aggregate metrics -> Cap at FP_MAX, flag for human review, log overflow event.
- **EC-C04**: Compliance score drops below threshold during production -> Ihsan Wall triggers automatic degradation to safe mode, alerts governance.
- **EC-C05**: External auditor requests raw data beyond retention window -> Return Merkle proof of existence + summary, not raw data (sovereignty I-6).

---

## 3. Pseudocode: Compliance Kernel

```
FUNCTION compliance_check(action, receipt, standard):
    """Run compliance verification for a specific standard."""

    # Step 1: Validate receipt integrity
    computed_hash = BLAKE3.hash(receipt.content)
    ASSERT computed_hash == receipt.receipt_id, "Receipt tampered"

    # Step 2: Verify oracle signature
    ASSERT Ed25519.verify(
        receipt.oracle_signature,
        receipt.receipt_id,
        oracle_public_key
    ), "Oracle signature invalid"

    # Step 3: Standard-specific checks
    IF standard == "ISO_25010":
        RETURN check_iso25010(receipt)
    ELIF standard == "CMMI_L5":
        RETURN check_cmmi_l5(receipt)
    ELIF standard == "SOC2_T2":
        RETURN check_soc2_t2(receipt)
    ELIF standard == "ISO_9001":
        RETURN check_iso9001(receipt)
    ELSE:
        RETURN Error("Unknown standard: " + standard)


FUNCTION generate_evidence_pack(standard, time_range):
    """Generate a compliance evidence pack for external audit."""

    # Collect all receipts in time range from append-only log
    receipts = EventLog.query(
        start=time_range.start,
        end=time_range.end,
        event_type="action_receipt"
    )

    # Compute aggregate metrics
    metrics = AggregateMetrics()
    FOR receipt IN receipts:
        result = compliance_check(receipt.action, receipt, standard)
        metrics.accumulate(result)

    # Build Merkle proof of completeness
    merkle_root = MerkleTree.build(
        [r.receipt_id FOR r IN receipts]
    ).root

    # Standard-specific formatting
    IF standard == "ISO_25010":
        report = format_iso25010_report(metrics, merkle_root)
    ELIF standard == "CMMI_L5":
        report = format_cmmi_report(metrics, merkle_root)
    ELIF standard == "SOC2_T2":
        report = format_soc2_report(metrics, merkle_root)
    ELIF standard == "ISO_9001":
        report = format_iso9001_report(metrics, merkle_root)

    # Sign the evidence pack
    report.signature = node_keypair.sign(BLAKE3.hash(report.content))
    report.merkle_root = merkle_root
    report.receipt_count = len(receipts)

    RETURN report


FUNCTION continuous_audit_tick():
    """Called every block — checks cross-standard invariants."""

    # I-1: FATE Gate active
    ASSERT fate_gate.is_active, "FATE Gate disabled — HALT"

    # I-2: Fixed-point mode
    ASSERT FP_PRECISION == 1_000_000, "FP precision drift"

    # I-3: Event log integrity
    ASSERT event_log.verify_chain(), "Merkle chain broken"

    # I-4: Gini ceiling
    current_gini = fp_float(compute_gini(all_wallets))
    ASSERT current_gini <= ADL_GINI_THRESHOLD, "Gini breach: " + str(current_gini)

    # I-5: Ihsan floor
    recent_ihsan = compute_recent_ihsan(last_100_receipts)
    ASSERT fp_float(recent_ihsan) >= IHSAN_PRODUCTION, "Ihsan floor breach"

    # I-6: Sovereignty check (no data leaks)
    ASSERT data_egress_log.all_authorized(), "Unauthorized data egress"

    # I-7: Crown verification
    ASSERT crown.h0_pass AND crown.h1_pass AND crown.h2_pass, "Crown verification failed"

    RETURN AuditTick(
        timestamp=now_ms(),
        gini=current_gini,
        ihsan=recent_ihsan,
        chain_valid=True,
        invariants_pass=True
    )
```

---

## 4. TDD Anchors

```
TEST receipt_is_atomic_compliance_evidence:
    receipt = create_test_receipt(intent=0.95, efficiency=0.90, impact=0.85)
    FOR standard IN ["ISO_25010", "CMMI_L5", "SOC2_T2", "ISO_9001"]:
        result = compliance_check(test_action, receipt, standard)
        ASSERT result.verified == True

TEST fixed_point_determinism_across_platforms:
    # Same inputs must produce byte-identical outputs
    score_arm = fp_add(fp(0.95), fp(0.05))  # Simulated ARM
    score_x86 = fp_add(fp(0.95), fp(0.05))  # Simulated x86
    ASSERT score_arm == score_x86 == FP_ONE

TEST evidence_pack_has_merkle_proof:
    pack = generate_evidence_pack("ISO_25010", last_30_days)
    ASSERT pack.merkle_root is not None
    ASSERT len(pack.merkle_root) == 32  # BLAKE3 digest
    ASSERT pack.signature is not None

TEST continuous_audit_detects_gini_breach:
    # Inject inequality
    skewed_wallets = create_skewed_wallets(gini=0.50)
    EXPECT_RAISE continuous_audit_tick()  # Gini > 0.35

TEST continuous_audit_detects_ihsan_breach:
    inject_low_quality_receipts(ihsan=0.80)
    EXPECT_RAISE continuous_audit_tick()  # Ihsan < 0.95

TEST evidence_pack_is_signed:
    pack = generate_evidence_pack("SOC2_T2", last_90_days)
    ASSERT Ed25519.verify(pack.signature, BLAKE3.hash(pack.content), node_pubkey)

TEST cross_standard_invariants_always_checked:
    tick = continuous_audit_tick()
    ASSERT tick.invariants_pass == True
    ASSERT tick.chain_valid == True
```

---

## 5. Cross-References

- `core/constitutional/types.py` — ActionReceipt, WalletState data structures
- `core/constitutional/fixed_point.py` — FP_PRECISION, fp(), fp_float(), fp_add()
- `core/constitutional/algorithms.py` — 15 Native Algorithms (A1-A15)
- `core/integration/constants.py` — All thresholds (authoritative)
- `core/proof_engine/evidence_ledger.py` — Append-only event log with Merkle chain
- `core/governance/` — Shura voting, proposal pipeline
- Phase 01 (ISO 25010), Phase 02 (CMMI L5), Phase 03 (SOC 2), Phase 04 (ISO 9001)
