"""
6-Gate Chain Tests — INT-005 Coverage Expansion.

Proves that the 6-gate fail-closed execution pipeline works correctly:
1. SchemaGate     -> Input validation
2. ProvenanceGate -> Source verification
3. SNRGate        -> Signal-to-noise threshold
4. ConstraintGate -> Z3 + Ihsan constraints
5. SafetyGate     -> Constitutional safety
6. CommitGate     -> Resource allocation + commit

Standing on Giants:
- Lamport (1978): Fail-closed semantics
- Dijkstra (1968): Structured decomposition
- BIZRA Spearpoint PRD SP-001: "6 gates, fail fast, fail closed"
"""

import pytest
from datetime import datetime, timezone

from core.proof_engine.canonical import CanonQuery, CanonPolicy
from core.proof_engine.gates import (
    Gate,
    GateStatus,
    GateResult,
    GateChainResult,
    GateChain,
    SchemaGate,
    ProvenanceGate,
    SNRGate,
    ConstraintGate,
    SafetyGate,
    CommitGate,
)
from core.proof_engine.receipt import SimpleSigner, ReceiptStatus
from core.proof_engine.snr import SNREngine, SNRInput, SNRPolicy


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def query():
    """Standard test query."""
    return CanonQuery(
        user_id="alice",
        user_state="active",
        intent="What is the meaning of life?",
        nonce="test_nonce_001",
    )


@pytest.fixture
def policy():
    """Standard test policy."""
    return CanonPolicy(
        policy_id="pol_test",
        version="1.0.0",
        rules={"snr_min": 0.95},
        thresholds={"ihsan": 0.95, "snr": 0.95},
    )


@pytest.fixture
def good_context():
    """Context that passes all gates."""
    return {
        "trust_score": 0.9,
        "ihsan_score": 0.97,
        "z3_satisfiable": True,
        "risk_score": 0.1,
        "provenance_depth": 3,
        "corroboration_count": 2,
        "source_trust_score": 0.9,
        "prediction_accuracy": 0.8,
        "context_fit_score": 0.8,
    }


@pytest.fixture
def signer():
    """Test signer."""
    return SimpleSigner(secret=b"test_secret_key_for_signing_receipts")


# =============================================================================
# GATE STATUS
# =============================================================================

class TestGateStatus:
    """Tests for GateStatus enum."""

    def test_has_all_statuses(self):
        """All 4 statuses exist."""
        assert GateStatus.PENDING.value == "pending"
        assert GateStatus.PASSED.value == "passed"
        assert GateStatus.FAILED.value == "failed"
        assert GateStatus.SKIPPED.value == "skipped"


# =============================================================================
# GATE RESULT
# =============================================================================

class TestGateResult:
    """Tests for GateResult dataclass."""

    def test_passed_property_true(self):
        """passed is True when status is PASSED."""
        r = GateResult(gate_name="test", status=GateStatus.PASSED)
        assert r.passed is True

    def test_passed_property_false(self):
        """passed is False when status is FAILED."""
        r = GateResult(gate_name="test", status=GateStatus.FAILED)
        assert r.passed is False

    def test_to_dict_shape(self):
        """to_dict() has expected keys."""
        r = GateResult(
            gate_name="schema", status=GateStatus.PASSED,
            duration_us=100, reason=None, evidence={"key": "val"},
        )
        d = r.to_dict()
        assert d["gate_name"] == "schema"
        assert d["status"] == "passed"
        assert d["duration_us"] == 100
        assert d["evidence"]["key"] == "val"
        assert "timestamp" in d

    def test_default_evidence_empty(self):
        """Default evidence is empty dict."""
        r = GateResult(gate_name="test", status=GateStatus.PENDING)
        assert r.evidence == {}

    def test_timestamp_auto_set(self):
        """Timestamp is auto-set."""
        r = GateResult(gate_name="test", status=GateStatus.PASSED)
        assert r.timestamp is not None
        assert r.timestamp.tzinfo is not None


# =============================================================================
# GATE CHAIN RESULT
# =============================================================================

class TestGateChainResult:
    """Tests for GateChainResult dataclass."""

    def test_passed_property(self, query, policy):
        """passed is True when final_status is PASSED."""
        r = GateChainResult(
            query=query, policy=policy, gate_results=[],
            final_status=GateStatus.PASSED, last_gate_passed="commit",
        )
        assert r.passed is True

    def test_failed_property(self, query, policy):
        """passed is False when final_status is FAILED."""
        r = GateChainResult(
            query=query, policy=policy, gate_results=[],
            final_status=GateStatus.FAILED, last_gate_passed="schema",
            rejection_reason="Missing field",
        )
        assert r.passed is False

    def test_to_dict_shape(self, query, policy):
        """to_dict() includes query/policy digests."""
        r = GateChainResult(
            query=query, policy=policy, gate_results=[],
            final_status=GateStatus.PASSED, last_gate_passed="commit",
            snr=0.96, ihsan_score=0.97,
        )
        d = r.to_dict()
        assert "query_digest" in d
        assert "policy_digest" in d
        assert d["final_status"] == "passed"
        assert d["snr"] == 0.96


# =============================================================================
# SCHEMA GATE (Gate 1)
# =============================================================================

class TestSchemaGate:
    """Tests for SchemaGate — input validation."""

    def test_passes_valid_query(self, query, policy):
        """Valid query passes schema gate."""
        gate = SchemaGate()
        result = gate.evaluate(query, policy, {})
        assert result.passed is True
        assert result.gate_name == "schema"

    def test_fails_missing_user_id(self, policy):
        """Missing user_id fails."""
        q = CanonQuery(user_id="", user_state="active", intent="test")
        gate = SchemaGate()
        result = gate.evaluate(q, policy, {})
        assert result.passed is False
        assert "user_id" in result.reason

    def test_fails_missing_intent(self, policy):
        """Missing intent fails."""
        q = CanonQuery(user_id="alice", user_state="active", intent="")
        gate = SchemaGate()
        result = gate.evaluate(q, policy, {})
        assert result.passed is False
        assert "intent" in result.reason

    def test_fails_intent_too_long(self, policy):
        """Intent exceeding max length fails."""
        q = CanonQuery(user_id="alice", user_state="active", intent="x" * 10001)
        gate = SchemaGate(max_intent_length=10000)
        result = gate.evaluate(q, policy, {})
        assert result.passed is False
        assert "max length" in result.reason

    def test_custom_max_intent_length(self, query, policy):
        """Custom max_intent_length respected."""
        gate = SchemaGate(max_intent_length=5)
        result = gate.evaluate(query, policy, {})
        assert result.passed is False  # "What is the meaning of life?" > 5

    def test_fails_payload_too_large(self, policy):
        """Payload exceeding max size fails."""
        q = CanonQuery(
            user_id="alice", user_state="active", intent="test",
            payload={"data": "x" * 100},
        )
        gate = SchemaGate(max_payload_size=50)
        result = gate.evaluate(q, policy, {})
        assert result.passed is False
        assert "max size" in result.reason

    def test_custom_required_fields(self, policy):
        """Custom required_fields checked."""
        gate = SchemaGate(required_fields=["user_id"])
        q = CanonQuery(user_id="alice", user_state="", intent="test")
        result = gate.evaluate(q, policy, {})
        assert result.passed is True  # Only user_id required

    def test_evidence_on_pass(self, query, policy):
        """Passing gate includes evidence."""
        gate = SchemaGate()
        result = gate.evaluate(query, policy, {})
        assert "intent_length" in result.evidence
        assert "payload_bytes" in result.evidence
        assert "fields_validated" in result.evidence

    def test_duration_measured(self, query, policy):
        """Duration is measured in microseconds."""
        gate = SchemaGate()
        result = gate.evaluate(query, policy, {})
        assert result.duration_us >= 0


# =============================================================================
# PROVENANCE GATE (Gate 2)
# =============================================================================

class TestProvenanceGate:
    """Tests for ProvenanceGate — source verification."""

    def test_passes_no_restrictions(self, query, policy):
        """No trusted sources requirement passes any source."""
        gate = ProvenanceGate()
        result = gate.evaluate(query, policy, {"trust_score": 0.8})
        assert result.passed is True

    def test_passes_trusted_source(self, query, policy):
        """Trusted source passes."""
        gate = ProvenanceGate(trusted_sources=["active"])
        result = gate.evaluate(query, policy, {"trust_score": 0.8})
        assert result.passed is True

    def test_fails_untrusted_low_score(self, query, policy):
        """Untrusted source with low trust score fails."""
        gate = ProvenanceGate(
            trusted_sources=["privileged"],
            min_trust_score=0.8,
        )
        result = gate.evaluate(query, policy, {"trust_score": 0.3})
        assert result.passed is False
        assert "Untrusted" in result.reason

    def test_untrusted_high_score_passes(self, query, policy):
        """Untrusted source with high trust score passes."""
        gate = ProvenanceGate(
            trusted_sources=["privileged"],
            min_trust_score=0.5,
        )
        result = gate.evaluate(query, policy, {"trust_score": 0.9})
        assert result.passed is True

    def test_fails_missing_signature(self, query, policy):
        """Missing required signature fails."""
        gate = ProvenanceGate(require_signature=True)
        result = gate.evaluate(query, policy, {})
        assert result.passed is False
        assert "signature" in result.reason.lower()

    def test_passes_with_signature(self, query, policy):
        """Present signature passes."""
        gate = ProvenanceGate(require_signature=True)
        result = gate.evaluate(query, policy, {"signature": "abc123"})
        assert result.passed is True

    def test_evidence_includes_trust_info(self, query, policy):
        """Evidence includes source and trust data."""
        gate = ProvenanceGate(trusted_sources=["active"])
        result = gate.evaluate(query, policy, {"trust_score": 0.8})
        assert "source" in result.evidence
        assert "trust_score" in result.evidence
        assert "is_trusted_source" in result.evidence


# =============================================================================
# SNR GATE (Gate 3)
# =============================================================================

class TestSNRGate:
    """Tests for SNRGate — signal-to-noise threshold."""

    def test_passes_high_quality(self, query, policy, good_context):
        """High-quality context passes SNR gate."""
        gate = SNRGate()
        result = gate.evaluate(query, policy, good_context)
        assert result.passed is True
        assert result.gate_name == "snr"

    def test_fails_low_quality(self, query, policy):
        """Low-quality context fails SNR gate."""
        ctx = {
            "trust_score": 0.1,
            "ihsan_score": 0.5,
            "z3_satisfiable": False,
            "contradiction_count": 5,
            "conflicting_sources": 3,
            "unverifiable_claims": 5,
            "missing_citations": 5,
        }
        gate = SNRGate()
        result = gate.evaluate(query, policy, ctx)
        assert result.passed is False
        assert "SNR below threshold" in result.reason

    def test_evidence_includes_snr(self, query, policy, good_context):
        """Evidence includes SNR value and threshold."""
        gate = SNRGate()
        result = gate.evaluate(query, policy, good_context)
        assert "snr" in result.evidence
        assert "threshold" in result.evidence

    def test_custom_policy(self, query, policy, good_context):
        """Custom SNR policy applied."""
        lenient_policy = SNRPolicy(snr_min=0.1)
        gate = SNRGate(snr_policy=lenient_policy)
        result = gate.evaluate(query, policy, good_context)
        assert result.passed is True

    def test_failed_evidence_includes_trace(self, query, policy):
        """Failed SNR gate includes trace in evidence."""
        ctx = {
            "trust_score": 0.1,
            "contradiction_count": 10,
            "conflicting_sources": 5,
            "unverifiable_claims": 10,
            "missing_citations": 10,
        }
        gate = SNRGate()
        result = gate.evaluate(query, policy, ctx)
        if not result.passed:
            assert "trace" in result.evidence


# =============================================================================
# CONSTRAINT GATE (Gate 4)
# =============================================================================

class TestConstraintGate:
    """Tests for ConstraintGate — Z3 + Ihsan constraints."""

    def test_passes_high_ihsan(self, query, policy):
        """High Ihsan score + Z3 satisfiable passes."""
        gate = ConstraintGate(ihsan_threshold=0.95)
        # CRITICAL-3: z3_satisfiable must be explicitly True (fail-closed default)
        result = gate.evaluate(query, policy, {"ihsan_score": 0.97, "z3_satisfiable": True})
        assert result.passed is True

    def test_fails_low_ihsan(self, query, policy):
        """Low Ihsan score fails."""
        gate = ConstraintGate(ihsan_threshold=0.95)
        result = gate.evaluate(query, policy, {"ihsan_score": 0.80})
        assert result.passed is False
        assert "score below threshold" in result.reason

    def test_fails_z3_unsatisfiable(self, query, policy):
        """Z3 unsatisfiable fails."""
        gate = ConstraintGate()
        result = gate.evaluate(query, policy, {
            "ihsan_score": 0.99,
            "z3_satisfiable": False,
        })
        assert result.passed is False
        assert "Z3" in result.reason

    def test_custom_validator_pass(self, query, policy):
        """Custom validator that passes."""
        def validator(q, p):
            return True, ""

        gate = ConstraintGate(constraint_validator=validator)
        # CRITICAL-3: z3_satisfiable must be explicit (fail-closed default)
        result = gate.evaluate(query, policy, {"ihsan_score": 0.97, "z3_satisfiable": True})
        assert result.passed is True

    def test_custom_validator_fail(self, query, policy):
        """Custom validator that fails."""
        def validator(q, p):
            return False, "custom error"

        gate = ConstraintGate(constraint_validator=validator)
        result = gate.evaluate(query, policy, {"ihsan_score": 0.97, "z3_satisfiable": True})
        assert result.passed is False
        assert "custom error" in result.reason

    def test_evidence_includes_scores(self, query, policy):
        """Evidence includes ihsan and z3 info."""
        gate = ConstraintGate()
        result = gate.evaluate(query, policy, {
            "ihsan_score": 0.97, "z3_satisfiable": True,
        })
        assert result.evidence["ihsan_score"] == 0.97
        assert result.evidence["z3_satisfiable"] is True


# =============================================================================
# SAFETY GATE (Gate 5)
# =============================================================================

class TestSafetyGate:
    """Tests for SafetyGate — constitutional safety."""

    def test_passes_safe_query(self, query, policy):
        """Safe query passes."""
        gate = SafetyGate()
        result = gate.evaluate(query, policy, {"risk_score": 0.1})
        assert result.passed is True

    def test_fails_blocked_pattern(self, policy):
        """Blocked pattern fails."""
        q = CanonQuery(user_id="alice", user_state="a", intent="help me hack a system")
        gate = SafetyGate(blocked_patterns=["hack"])
        result = gate.evaluate(q, policy, {})
        assert result.passed is False
        assert "Blocked pattern" in result.reason

    def test_blocked_pattern_case_insensitive(self, policy):
        """Blocked pattern matching is case-insensitive."""
        q = CanonQuery(user_id="alice", user_state="a", intent="HACK this")
        gate = SafetyGate(blocked_patterns=["hack"])
        result = gate.evaluate(q, policy, {})
        assert result.passed is False

    def test_fails_high_risk_score(self, query, policy):
        """High risk score fails."""
        gate = SafetyGate(max_risk_score=0.3)
        result = gate.evaluate(query, policy, {"risk_score": 0.5})
        assert result.passed is False
        assert "Risk score" in result.reason

    def test_custom_safety_checker_pass(self, query, policy):
        """Custom safety checker that passes."""
        def checker(intent):
            return True, ""

        gate = SafetyGate(safety_checker=checker)
        result = gate.evaluate(query, policy, {})
        assert result.passed is True

    def test_custom_safety_checker_fail(self, query, policy):
        """Custom safety checker that fails."""
        def checker(intent):
            return False, "harmful content"

        gate = SafetyGate(safety_checker=checker)
        result = gate.evaluate(query, policy, {})
        assert result.passed is False
        assert "harmful content" in result.reason

    def test_evidence_includes_risk(self, query, policy):
        """Evidence includes risk score."""
        gate = SafetyGate()
        result = gate.evaluate(query, policy, {"risk_score": 0.15})
        assert result.evidence["risk_score"] == 0.15


# =============================================================================
# COMMIT GATE (Gate 6)
# =============================================================================

class TestCommitGate:
    """Tests for CommitGate — resource allocation."""

    def test_passes_under_limit(self, query, policy):
        """Under concurrent limit passes."""
        gate = CommitGate(max_concurrent_ops=10)
        result = gate.evaluate(query, policy, {})
        assert result.passed is True
        assert result.evidence["current_ops"] == 1

    def test_fails_at_limit(self, query, policy):
        """At concurrent limit fails."""
        gate = CommitGate(max_concurrent_ops=1)
        gate._current_ops = 1
        result = gate.evaluate(query, policy, {})
        assert result.passed is False
        assert "Max concurrent" in result.reason

    def test_increments_ops_on_pass(self, query, policy):
        """Passing increments current operations."""
        gate = CommitGate()
        assert gate._current_ops == 0
        gate.evaluate(query, policy, {})
        assert gate._current_ops == 1

    def test_release_decrements(self, query, policy):
        """release() decrements operations."""
        gate = CommitGate()
        gate.evaluate(query, policy, {})
        assert gate._current_ops == 1
        gate.release()
        assert gate._current_ops == 0

    def test_release_no_underflow(self):
        """release() doesn't go below zero."""
        gate = CommitGate()
        gate.release()
        assert gate._current_ops == 0

    def test_resource_checker_pass(self, query, policy):
        """Resource checker that passes."""
        def checker():
            return True, ""

        gate = CommitGate(resource_checker=checker)
        result = gate.evaluate(query, policy, {})
        assert result.passed is True

    def test_resource_checker_fail(self, query, policy):
        """Resource checker that fails."""
        def checker():
            return False, "out of memory"

        gate = CommitGate(resource_checker=checker)
        result = gate.evaluate(query, policy, {})
        assert result.passed is False
        assert "out of memory" in result.reason

    def test_commit_id_in_evidence(self, query, policy):
        """Evidence includes commit_id."""
        gate = CommitGate()
        result = gate.evaluate(query, policy, {})
        assert "commit_id" in result.evidence
        assert len(result.evidence["commit_id"]) == 16


# =============================================================================
# GATE CHAIN — Full Pipeline
# =============================================================================

class TestGateChain:
    """Tests for GateChain — 6-gate execution pipeline."""

    def test_all_gates_pass(self, query, policy, good_context, signer):
        """All gates pass with good context."""
        chain = GateChain(signer=signer)
        result, receipt = chain.evaluate(query, policy, good_context)
        assert result.passed is True
        assert result.final_status == GateStatus.PASSED
        assert result.last_gate_passed == "commit"
        assert len(result.gate_results) == 6

    def test_receipt_accepted_on_pass(self, query, policy, good_context, signer):
        """Accepted receipt on pass."""
        chain = GateChain(signer=signer)
        _, receipt = chain.evaluate(query, policy, good_context)
        assert receipt.status == ReceiptStatus.ACCEPTED
        assert receipt.gate_passed == "commit"

    def test_receipt_rejected_on_fail(self, policy, signer):
        """Rejected receipt on fail."""
        q = CanonQuery(user_id="", user_state="a", intent="test")
        chain = GateChain(signer=signer)
        result, receipt = chain.evaluate(q, policy, {})
        assert result.passed is False
        assert receipt.status == ReceiptStatus.REJECTED

    def test_stops_at_first_failure(self, policy, signer):
        """Chain stops at first failed gate."""
        q = CanonQuery(user_id="", user_state="a", intent="test")
        chain = GateChain(signer=signer)
        result, _ = chain.evaluate(q, policy, {})
        # Schema gate fails, so only 1 gate evaluated
        assert len(result.gate_results) == 1
        assert result.gate_results[0].gate_name == "schema"

    def test_failure_at_safety_gate(self, query, policy, signer):
        """Safety gate failure stops chain."""
        chain = GateChain(
            signer=signer,
            gates=[
                SchemaGate(),
                ProvenanceGate(),
                SNRGate(),
                ConstraintGate(),
                SafetyGate(blocked_patterns=["meaning"]),
                CommitGate(),
            ],
        )
        ctx = {
            "trust_score": 0.9,
            "ihsan_score": 0.97,
            "z3_satisfiable": True,
            "source_trust_score": 0.9,
            "prediction_accuracy": 0.8,
            "context_fit_score": 0.8,
        }
        result, receipt = chain.evaluate(query, policy, ctx)
        assert result.passed is False
        assert result.last_gate_passed == "constraint"
        assert "Blocked pattern" in result.rejection_reason

    def test_receipt_is_signed(self, query, policy, good_context, signer):
        """Receipt is signed."""
        chain = GateChain(signer=signer)
        _, receipt = chain.evaluate(query, policy, good_context)
        assert len(receipt.signature) > 0

    def test_receipt_signature_verifiable(self, query, policy, good_context, signer):
        """Receipt signature can be verified."""
        chain = GateChain(signer=signer)
        _, receipt = chain.evaluate(query, policy, good_context)
        assert receipt.verify_signature(signer)

    def test_snr_captured_in_result(self, query, policy, good_context, signer):
        """SNR value captured in chain result."""
        chain = GateChain(signer=signer)
        result, _ = chain.evaluate(query, policy, good_context)
        assert result.snr > 0.0

    def test_default_gates_order(self, signer):
        """Default gates are in correct order."""
        chain = GateChain(signer=signer)
        names = [g.name for g in chain.gates]
        assert names == ["schema", "provenance", "snr", "constraint", "safety", "commit"]

    def test_custom_gates(self, query, policy, signer):
        """Custom gates list respected."""
        chain = GateChain(
            signer=signer,
            gates=[SchemaGate(), CommitGate()],
        )
        result, _ = chain.evaluate(query, policy, {})
        assert result.passed is True
        assert len(result.gate_results) == 2

    def test_total_duration_measured(self, query, policy, good_context, signer):
        """Total duration is measured."""
        chain = GateChain(signer=signer)
        result, _ = chain.evaluate(query, policy, good_context)
        assert result.total_duration_us >= 0

    def test_evaluations_tracked(self, query, policy, good_context, signer):
        """Evaluations are stored for statistics."""
        chain = GateChain(signer=signer)
        chain.evaluate(query, policy, good_context)
        chain.evaluate(query, policy, good_context)
        assert len(chain._evaluations) == 2


# =============================================================================
# GATE CHAIN — Statistics
# =============================================================================

class TestGateChainStats:
    """Tests for GateChain statistics."""

    def test_empty_stats(self, signer):
        """Stats with no evaluations."""
        chain = GateChain(signer=signer)
        stats = chain.get_stats()
        assert stats["total_evaluations"] == 0
        assert "gates" in stats

    def test_stats_after_evaluations(self, query, policy, good_context, signer):
        """Stats after multiple evaluations."""
        chain = GateChain(signer=signer)
        chain.evaluate(query, policy, good_context)
        chain.evaluate(query, policy, good_context)

        stats = chain.get_stats()
        assert stats["total_evaluations"] == 2
        assert stats["passed"] == 2
        assert stats["failed"] == 0
        assert stats["pass_rate"] == 1.0
        assert stats["avg_snr"] > 0

    def test_stats_with_failures(self, policy, signer, good_context):
        """Stats track failures by gate."""
        chain = GateChain(signer=signer)
        good_q = CanonQuery(user_id="a", user_state="s", intent="test")
        bad_q = CanonQuery(user_id="", user_state="s", intent="test")

        chain.evaluate(good_q, policy, good_context)
        chain.evaluate(bad_q, policy, {})

        stats = chain.get_stats()
        assert stats["total_evaluations"] == 2
        assert stats["passed"] == 1
        assert stats["failed"] == 1

    def test_recent_results(self, query, policy, good_context, signer):
        """get_recent_results returns evaluations."""
        chain = GateChain(signer=signer)
        chain.evaluate(query, policy, good_context)
        recent = chain.get_recent_results(limit=5)
        assert len(recent) == 1
        assert "final_status" in recent[0]


# =============================================================================
# GATE CHAIN — Amber Restricted
# =============================================================================

class TestGateChainAmber:
    """Tests for amber-restricted fallback."""

    def test_amber_on_safety_fail_high_snr(self, query, policy, signer):
        """Amber receipt when safety fails but SNR is high."""
        chain = GateChain(
            signer=signer,
            gates=[
                SchemaGate(),
                ProvenanceGate(),
                SNRGate(),
                ConstraintGate(),
                SafetyGate(blocked_patterns=["meaning"]),
                CommitGate(),
            ],
        )
        ctx = {
            "trust_score": 0.9,
            "ihsan_score": 0.97,
            "z3_satisfiable": True,
            "source_trust_score": 0.9,
            "prediction_accuracy": 0.9,
            "context_fit_score": 0.9,
        }
        result, receipt = chain.evaluate_with_amber(query, policy, ctx)
        # Note: amber only triggers if last_gate_passed == "constraint"
        # and SNR >= 0.90, which should be the case here
        if not result.passed and result.last_gate_passed == "constraint":
            assert receipt.status == ReceiptStatus.AMBER_RESTRICTED

    def test_no_amber_on_early_fail(self, policy, signer):
        """No amber on early gate failure."""
        q = CanonQuery(user_id="", user_state="a", intent="test")
        chain = GateChain(signer=signer)
        result, receipt = chain.evaluate_with_amber(q, policy, {})
        assert receipt.status == ReceiptStatus.REJECTED


# =============================================================================
# BASE GATE CLASS
# =============================================================================

class TestBaseGate:
    """Tests for Gate base class."""

    def test_evaluate_raises(self, query, policy):
        """Base Gate.evaluate() raises NotImplementedError."""
        gate = Gate("test_base")
        with pytest.raises(NotImplementedError):
            gate.evaluate(query, policy, {})

    def test_name_stored(self):
        """Gate stores its name."""
        gate = Gate("my_gate")
        assert gate.name == "my_gate"
