"""
Tests for Constitutional Security Kernel (CSK)
===============================================

Tests all 7 audit findings:
  α1: Oblivious compute scheduler
  α3: Takaful admission gate
  α4: Conservative FATE fallback
  α6: Evolutionary إحسان fitness interface
  α7: Tiered verification engine
  Performance attestation envelopes (XZ thermodynamic detection)

Standing on: Hypothesis (Maciver, 2013) — property-based testing
"""

import time

import pytest

from core.proof_engine.constitutional_security_kernel import (
    CSKReasonCode,
    CSKReceipt,
    ObliviousComputeScheduler,
    PerformanceAttestationRegistry,
    PerformanceEnvelope,
    StaticIhsanFitness,
    TakafulAdmissionGate,
    TakafulStatus,
    TierDecision,
    TieredVerificationEngine,
    conservative_constraint_check,
)

# ═══════════════════════════════════════════════════════════════════════════════
# §1  Conservative FATE Fallback (α4)
# ═══════════════════════════════════════════════════════════════════════════════


class TestConservativeFallback:
    """The fallback MUST be stricter than Z3, not weaker."""

    def test_zero_risk_passes(self):
        """Zero-risk actions always pass if thresholds met."""
        ctx = {
            "ihsan": 0.99,
            "snr": 0.95,
            "cost": 1.0,
            "autonomy_limit": 10.0,
            "risk_level": 0.0,
        }
        passed, code = conservative_constraint_check(ctx)
        assert passed is True
        assert code == CSKReasonCode.T2_FATE_FALLBACK_CONSERVATIVE_PASS

    def test_ihsan_below_threshold_rejects(self):
        """Below إحسان floor → reject regardless of other factors."""
        ctx = {
            "ihsan": 0.5,
            "snr": 0.99,
            "cost": 0,
            "autonomy_limit": 100,
            "risk_level": 0.0,
        }
        passed, code = conservative_constraint_check(ctx)
        assert passed is False
        assert code == CSKReasonCode.T2_FATE_FALLBACK_CONSERVATIVE_REJECT

    def test_nonzero_risk_without_safe_pattern_rejects(self):
        """Non-zero risk + no safe pattern match → default deny."""
        ctx = {
            "ihsan": 0.99,
            "snr": 0.99,
            "cost": 1.0,
            "autonomy_limit": 10.0,
            "risk_level": 0.5,
            "action_type": "deploy_nuclear_reactor",  # Not a registered safe pattern
        }
        passed, code = conservative_constraint_check(ctx)
        assert passed is False

    def test_read_only_query_passes(self):
        """Read-only query pattern is registered safe by default."""
        ctx = {
            "ihsan": 0.99,
            "snr": 0.95,
            "cost": 0.5,
            "autonomy_limit": 10.0,
            "risk_level": 0.2,
            "action_type": "query",
        }
        passed, code = conservative_constraint_check(ctx)
        assert passed is True

    def test_human_approved_passes(self):
        """Human-approved action passes conservative check."""
        ctx = {
            "ihsan": 0.99,
            "snr": 0.95,
            "cost": 5.0,
            "autonomy_limit": 10.0,
            "risk_level": 0.6,
            "human_approved": True,
        }
        passed, code = conservative_constraint_check(ctx)
        assert passed is True

    def test_high_risk_unapproved_rejects(self):
        """High risk + not approved + not reversible → reject."""
        ctx = {
            "ihsan": 0.99,
            "snr": 0.99,
            "cost": 1.0,
            "autonomy_limit": 10.0,
            "risk_level": 0.8,
            "reversible": False,
            "human_approved": False,
        }
        passed, _ = conservative_constraint_check(ctx)
        assert passed is False

    def test_cost_exceeding_limit_rejects(self):
        """Cost > autonomy_limit → reject."""
        ctx = {
            "ihsan": 0.99,
            "snr": 0.99,
            "cost": 100.0,
            "autonomy_limit": 10.0,
            "risk_level": 0.0,
        }
        passed, _ = conservative_constraint_check(ctx)
        assert passed is False

    def test_missing_fields_default_to_unsafe(self):
        """Missing context fields default to unsafe values."""
        # Empty context → ihsan=0.0, snr=0.0 → reject
        passed, _ = conservative_constraint_check({})
        assert passed is False


# ═══════════════════════════════════════════════════════════════════════════════
# §2  Tiered Verification Engine (α7)
# ═══════════════════════════════════════════════════════════════════════════════


class TestTieredVerification:
    """4-tier pipeline: 50ms / 500ms / 1.6s / async."""

    @pytest.fixture
    def engine(self):
        return TieredVerificationEngine()

    def test_tier1_blocks_dangerous(self, engine):
        """Tier 1 blocks known-dangerous actions immediately."""
        ctx = {
            "action_type": "delete",
            "risk_level": 0.9,
            "human_approved": False,
        }
        result = engine.tier1_precheck(ctx)
        assert result.tier == 1
        assert result.decision == TierDecision.BLOCK
        assert result.receipt.reason_code == CSKReasonCode.T1_KNOWN_DANGEROUS_BLOCKED

    def test_tier1_allows_safe(self, engine):
        """Tier 1 allows known-safe patterns immediately."""
        ctx = {
            "action_type": "query",
            "risk_level": 0.1,
        }
        result = engine.tier1_precheck(ctx)
        assert result.tier == 1
        assert result.decision == TierDecision.ALLOW
        assert result.receipt.reason_code == CSKReasonCode.T1_SAFE_PATTERN_MATCH

    def test_tier1_escalates_unknown(self, engine):
        """Tier 1 escalates unrecognized actions to Tier 2."""
        ctx = {
            "action_type": "synthesize",
            "risk_level": 0.4,
        }
        result = engine.tier1_precheck(ctx)
        assert result.tier == 1
        assert result.decision == TierDecision.ESCALATE

    def test_tier1_completes_under_50ms(self, engine):
        """Tier 1 must complete within 50ms budget."""
        ctx = {"action_type": "query", "risk_level": 0.1}
        result = engine.tier1_precheck(ctx)
        # 50ms = 50,000 µs
        assert result.duration_us < 50_000

    def test_tier2_conservative_fallback_without_z3(self, engine):
        """Tier 2 uses conservative fallback when Z3 unavailable."""
        engine._z3_available = False
        ctx = {
            "ihsan": 0.99,
            "snr": 0.95,
            "cost": 1.0,
            "autonomy_limit": 10.0,
            "risk_level": 0.0,
        }
        result = engine.tier2_formal_verification(ctx)
        assert result.tier == 2
        # Should pass (zero risk, thresholds met)
        assert result.decision == TierDecision.ALLOW

    def test_tier2_rejects_when_z3_unavailable_and_risky(self, engine):
        """Tier 2 rejects risky actions when Z3 unavailable."""
        engine._z3_available = False
        ctx = {
            "ihsan": 0.99,
            "snr": 0.95,
            "cost": 5.0,
            "autonomy_limit": 10.0,
            "risk_level": 0.6,
            "action_type": "deploy",
            "reversible": False,
            "human_approved": False,
        }
        result = engine.tier2_formal_verification(ctx)
        assert result.tier == 2
        assert result.decision == TierDecision.BLOCK

    def test_full_pipeline_blocks_at_tier1(self, engine):
        """Full pipeline stops at Tier 1 for dangerous actions."""
        ctx = {
            "action_type": "execute",
            "risk_level": 0.9,
            "human_approved": False,
        }
        results = engine.verify(ctx)
        assert len(results) == 1  # Only Tier 1 executed
        assert results[0].decision == TierDecision.BLOCK

    def test_full_pipeline_escalates_to_tier2(self, engine):
        """Full pipeline runs Tier 2 when Tier 1 escalates."""
        engine._z3_available = False
        ctx = {
            "action_type": "synthesize",
            "risk_level": 0.4,
            "ihsan": 0.99,
            "snr": 0.95,
            "cost": 1.0,
            "autonomy_limit": 10.0,
            "reversible": False,
            "human_approved": False,
        }
        results = engine.verify(ctx)
        assert len(results) == 2  # Tier 1 + Tier 2

    def test_every_decision_produces_receipt(self, engine):
        """Every tier decision produces a CSKReceipt."""
        ctx = {"action_type": "query", "risk_level": 0.1}
        result = engine.tier1_precheck(ctx)
        assert isinstance(result.receipt, CSKReceipt)
        assert result.receipt.receipt_id.startswith("csk_t1_")
        assert result.receipt.action_digest  # Non-empty


# ═══════════════════════════════════════════════════════════════════════════════
# §3  Performance Attestation Envelopes (XZ thermodynamic detection)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPerformanceAttestation:
    """Physics can't lie — timing anomalies detect compromise."""

    @pytest.fixture
    def registry(self):
        reg = PerformanceAttestationRegistry()
        reg.register(
            PerformanceEnvelope(
                module_name="gate_chain",
                expected_duration_us=500.0,
                duration_stddev_us=50.0,
                sigma_threshold=2.0,
            )
        )
        return reg

    def test_normal_execution_passes(self, registry):
        """Execution within envelope produces PERF_WITHIN_ENVELOPE."""
        receipt = registry.attest("gate_chain", 520.0)
        assert receipt.passed is True
        assert receipt.reason_code == CSKReasonCode.PERF_WITHIN_ENVELOPE

    def test_anomalous_execution_detected(self, registry):
        """Execution far outside envelope produces anomaly receipt."""
        # 500ms when expected 500µs = 1000x deviation
        receipt = registry.attest("gate_chain", 500_000.0)
        assert receipt.passed is False
        assert receipt.reason_code in (
            CSKReasonCode.PERF_ANOMALY_MINOR,
            CSKReasonCode.PERF_ANOMALY_CRITICAL,
        )

    def test_welford_updates_statistics(self, registry):
        """Welford's algorithm tracks running mean and stddev."""
        envelope = registry._envelopes["gate_chain"]
        for _ in range(100):
            envelope.record_observation(500.0 + (time.perf_counter_ns() % 100 - 50))
        assert envelope._count == 100
        assert 400 < envelope.observed_mean < 600

    def test_unregistered_module_not_blocked(self, registry):
        """Unregistered modules are flagged but not blocked."""
        receipt = registry.attest("unknown_module", 1000.0)
        assert receipt.passed is True  # Not blocked
        assert receipt.reason_code == CSKReasonCode.PERF_ANOMALY_MINOR

    def test_anomaly_log_maintained(self, registry):
        """Anomalies are logged for downstream review."""
        registry.attest("gate_chain", 500_000.0)
        anomalies = registry.get_anomalies()
        assert len(anomalies) >= 1
        assert anomalies[0]["module"] == "gate_chain"


# ═══════════════════════════════════════════════════════════════════════════════
# §4  Takaful Admission Gate (α3 — Sybil defense)
# ═══════════════════════════════════════════════════════════════════════════════


class TestTakafulAdmission:
    """Three gates: humanity + impact + إحسان maintenance."""

    @pytest.fixture
    def gate(self):
        return TakafulAdmissionGate(
            min_interactions=10,  # Lower for testing
            min_impact_score=5.0,
            ihsan_floor=0.95,
            ihsan_history_window=5,
        )

    def test_new_node_is_probationary(self, gate):
        """New nodes start as PROBATIONARY."""
        profile = gate.register_node("node_001")
        assert profile.status == TakafulStatus.PROBATIONARY

    def test_probationary_can_receive_not_contribute(self, gate):
        """Probationary nodes can receive but not contribute."""
        gate.register_node("node_001")
        assert gate.can_receive("node_001") is True
        assert gate.can_contribute("node_001") is False

    def test_unverified_humanity_rejects(self, gate):
        """Without humanity verification, admission fails."""
        gate.register_node("node_001")
        receipt = gate.evaluate_admission("node_001")
        assert receipt.passed is False
        assert receipt.reason_code == CSKReasonCode.TAKAFUL_REJECTED_NO_HUMANITY_PROOF

    def test_insufficient_impact_rejects(self, gate):
        """With humanity but insufficient impact, admission fails."""
        gate.register_node("node_001")
        gate.verify_humanity("node_001")
        # Only 3 interactions (need 10)
        for _ in range(3):
            gate.record_interaction("node_001", impact_delta=1.0, ihsan_score=0.99)
        receipt = gate.evaluate_admission("node_001")
        assert receipt.passed is False
        assert receipt.reason_code == CSKReasonCode.TAKAFUL_REJECTED_INSUFFICIENT_IMPACT

    def test_ihsan_below_floor_rejects(self, gate):
        """With humanity + impact but إحسان violations, admission fails."""
        gate.register_node("node_001")
        gate.verify_humanity("node_001")
        # Enough interactions and impact
        for i in range(15):
            gate.record_interaction("node_001", impact_delta=1.0, ihsan_score=0.80)
        receipt = gate.evaluate_admission("node_001")
        assert receipt.passed is False
        assert receipt.reason_code == CSKReasonCode.TAKAFUL_REJECTED_IHSAN_BELOW_FLOOR

    def test_all_three_gates_pass_admits(self, gate):
        """Pass humanity + impact + إحسان → ADMITTED."""
        gate.register_node("node_001")
        gate.verify_humanity("node_001")
        for _ in range(15):
            gate.record_interaction("node_001", impact_delta=1.0, ihsan_score=0.99)
        receipt = gate.evaluate_admission("node_001")
        assert receipt.passed is True
        assert receipt.reason_code == CSKReasonCode.TAKAFUL_ADMITTED
        assert gate.can_contribute("node_001") is True

    def test_suspended_node_cannot_contribute(self, gate):
        """Suspended nodes lose contributor status."""
        gate.register_node("node_001")
        gate.verify_humanity("node_001")
        for _ in range(15):
            gate.record_interaction("node_001", impact_delta=1.0, ihsan_score=0.99)
        gate.evaluate_admission("node_001")
        assert gate.can_contribute("node_001") is True

        gate.suspend_node("node_001", reason="anomaly_detected")
        assert gate.can_contribute("node_001") is False

    def test_sybil_attack_blocked_at_gate2(self, gate):
        """Sybil with zero-impact interactions cannot pass Gate 2."""
        gate.register_node("sybil_001")
        gate.verify_humanity("sybil_001")  # Assume they passed CAPTCHA
        # Fake interactions with zero impact
        for _ in range(100):
            gate.record_interaction("sybil_001", impact_delta=0.0, ihsan_score=0.99)
        receipt = gate.evaluate_admission("sybil_001")
        assert receipt.passed is False
        assert receipt.reason_code == CSKReasonCode.TAKAFUL_REJECTED_INSUFFICIENT_IMPACT


# ═══════════════════════════════════════════════════════════════════════════════
# §5  Oblivious Compute Scheduler (α1 — URP privacy)
# ═══════════════════════════════════════════════════════════════════════════════


class TestObliviousCompute:
    """Compute patterns must be ε-indistinguishable."""

    @pytest.fixture
    def scheduler(self):
        return ObliviousComputeScheduler(epsilon=1.0, dummy_compute_ratio=0.25)

    def test_dummy_operations_injected(self, scheduler):
        """Real operations are padded with dummies."""
        real_ops = [
            {"op_type": "inference", "payload_size": 256},
            {"op_type": "embedding", "payload_size": 512},
            {"op_type": "search", "payload_size": 128},
            {"op_type": "inference", "payload_size": 256},
        ]
        scheduled = scheduler.schedule(real_ops)
        assert len(scheduled) > len(real_ops)
        # Dummies exist
        dummies = [op for op in scheduled if op.get("is_dummy")]
        assert len(dummies) >= 1

    def test_strip_dummies_recovers_real_count(self, scheduler):
        """Stripping dummies returns only real results."""
        real_ops = [
            {"op_type": "inference", "payload_size": 256},
            {"op_type": "inference", "payload_size": 256},
        ]
        scheduled = scheduler.schedule(real_ops)
        # Simulate execution: add result field
        results = [{**op, "result": "ok"} for op in scheduled]
        stripped = scheduler.strip_dummies(results)
        assert len(stripped) == len(real_ops)
        for r in stripped:
            assert r["is_dummy"] is False

    def test_shuffling_randomizes_order(self, scheduler):
        """Scheduled operations are shuffled (non-deterministic order)."""
        real_ops = [{"op_type": f"op_{i}", "payload_size": i * 100} for i in range(10)]
        results = []
        for _ in range(10):
            scheduled = scheduler.schedule(real_ops)
            order = [op.get("op_type") for op in scheduled if not op.get("is_dummy")]
            results.append(tuple(order))
        # At least one permutation should differ (probabilistic but very likely)
        assert len(set(results)) > 1

    def test_empty_operations_handled(self, scheduler):
        """Empty operation list doesn't crash."""
        scheduled = scheduler.schedule([])
        assert isinstance(scheduled, list)


# ═══════════════════════════════════════════════════════════════════════════════
# §6  Static إحسان Fitness (α6 — evolutionary interface)
# ═══════════════════════════════════════════════════════════════════════════════


class TestStaticIhsanFitness:
    """Static v1.0 evaluator — baseline for evolutionary v2.0."""

    @pytest.fixture
    def fitness(self):
        return StaticIhsanFitness()

    def test_perfect_context_scores_high(self, fitness):
        """Perfect action context scores near 1.0."""
        ctx = {
            "ihsan": 1.0,
            "snr": 1.0,
            "impact_delta": 10.0,
            "reversible": True,
            "human_approved": True,
        }
        score = fitness.evaluate(ctx)
        assert score >= 0.95

    def test_empty_context_scores_zero(self, fitness):
        """Empty context scores 0.0."""
        score = fitness.evaluate({})
        assert score == 0.0

    def test_score_bounded_0_1(self, fitness):
        """Score is always in [0.0, 1.0]."""
        for ihsan in [0.0, 0.5, 1.0, 2.0]:
            for snr in [0.0, 0.5, 1.0]:
                score = fitness.evaluate({"ihsan": ihsan, "snr": snr})
                assert 0.0 <= score <= 1.0

    def test_static_does_not_mutate(self, fitness):
        """Static evaluator always returns None for mutations."""
        result = fitness.propose_mutation([{"ihsan": 0.99}])
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# §7  CSK Receipt Integrity
# ═══════════════════════════════════════════════════════════════════════════════


class TestCSKReceipt:
    """Receipts are the proof.  They must be reliable."""

    def test_receipt_is_frozen(self):
        """CSKReceipt is immutable (frozen dataclass)."""
        receipt = CSKReceipt(
            receipt_id="test_001",
            tier=1,
            reason_code=CSKReasonCode.T1_SAFE_PATTERN_MATCH,
            timestamp_ns=time.time_ns(),
            action_digest="abc123",
            passed=True,
        )
        with pytest.raises(AttributeError):
            receipt.passed = False  # type: ignore[misc]

    def test_receipt_digest_deterministic(self):
        """Same receipt content → same digest."""
        r1 = CSKReceipt(
            receipt_id="test_001",
            tier=1,
            reason_code=CSKReasonCode.T1_SAFE_PATTERN_MATCH,
            timestamp_ns=1000,
            action_digest="abc123",
            passed=True,
        )
        r2 = CSKReceipt(
            receipt_id="test_001",
            tier=1,
            reason_code=CSKReasonCode.T1_SAFE_PATTERN_MATCH,
            timestamp_ns=1000,
            action_digest="abc123",
            passed=True,
        )
        assert r1.digest() == r2.digest()

    def test_receipt_to_dict_complete(self):
        """to_dict() includes all fields."""
        receipt = CSKReceipt(
            receipt_id="test_001",
            tier=2,
            reason_code=CSKReasonCode.T2_FATE_Z3_PASSED,
            timestamp_ns=time.time_ns(),
            action_digest="deadbeef",
            passed=True,
            evidence={"proof_id": "p_000001"},
        )
        d = receipt.to_dict()
        assert d["receipt_id"] == "test_001"
        assert d["tier"] == 2
        assert d["passed"] is True
        assert "proof_id" in d["evidence"]
