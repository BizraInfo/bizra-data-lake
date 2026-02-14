"""
Comprehensive tests for the Proof-Carrying Execution Engine.

Tests all 4 deliverables:
1. Deterministic Canonicalization
2. SNR Scoring as Verifiable Function
3. Signed Rejection Receipts
4. Bench-as-Receipt Harness

Plus the 6-Gate Chain implementation.
"""

import pytest
import time
from datetime import datetime, timezone

from core.proof_engine import (
    PROOF_ENGINE_VERSION,
    GATE_CHAIN,
    PROOF_KPIS,
    DEFAULT_SNR_POLICY,
)
from core.proof_engine.canonical import (
    CanonQuery,
    CanonPolicy,
    CanonEnvironment,
    canonical_json,
    canonical_bytes,
    blake3_digest,
    verify_determinism,
)
from core.proof_engine.snr import (
    SNREngine,
    SNRPolicy,
    SNRInput,
    SNRTrace,
)
from core.proof_engine.receipt import (
    Receipt,
    ReceiptStatus,
    ReceiptBuilder,
    ReceiptVerifier,
    SimpleSigner,
    Metrics,
)
from core.proof_engine.gates import (
    GateChain,
    GateResult,
    GateChainResult,
    GateStatus,
    SchemaGate,
    ProvenanceGate,
    SNRGate,
    ConstraintGate,
    SafetyGate,
    CommitGate,
)
from core.proof_engine.bench import (
    BenchHarness,
    BenchResult,
    BenchReceipt,
    BenchSample,
    bench_to_receipt,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def signer():
    """Create a test signer."""
    return SimpleSigner(b"test_secret_key_12345")


@pytest.fixture
def query():
    """Create a test query."""
    return CanonQuery(
        user_id="test_user_001",
        user_state="authenticated",
        intent="What is the capital of France?",
        payload={"context": "geography", "language": "en"},
        nonce="nonce_12345",
    )


@pytest.fixture
def policy():
    """Create a test policy."""
    return CanonPolicy(
        policy_id="policy_001",
        version="1.0.0",
        rules={"allow_general": True, "max_tokens": 1000},
        thresholds={"snr_min": 0.95, "ihsan_min": 0.95},
        constraints=["no_harm", "factual_only"],
    )


@pytest.fixture
def snr_engine():
    """Create an SNR engine."""
    return SNREngine()


@pytest.fixture
def receipt_builder(signer):
    """Create a receipt builder."""
    return ReceiptBuilder(signer)


@pytest.fixture
def gate_chain(signer):
    """Create a gate chain."""
    return GateChain(signer)


# =============================================================================
# DELIVERABLE 1: DETERMINISTIC CANONICALIZATION
# =============================================================================

class TestDeterministicCanonicalization:
    """Tests for deterministic canonicalization."""

    def test_canonical_json_sorts_keys(self):
        """Canonical JSON sorts object keys alphabetically."""
        obj = {"z": 1, "a": 2, "m": 3}
        result = canonical_json(obj)
        assert list(result.keys()) == ["a", "m", "z"]

    def test_canonical_json_normalizes_strings(self):
        """Canonical JSON normalizes Unicode strings."""
        obj = {"text": "  hello world  "}
        result = canonical_json(obj)
        assert result["text"] == "hello world"

    def test_canonical_json_preserves_arrays(self):
        """Canonical JSON preserves array order."""
        obj = {"items": [3, 1, 2]}
        result = canonical_json(obj)
        assert result["items"] == [3, 1, 2]

    def test_canonical_json_nested_objects(self):
        """Canonical JSON handles nested objects."""
        obj = {"outer": {"z": 1, "a": 2}}
        result = canonical_json(obj)
        assert list(result["outer"].keys()) == ["a", "z"]

    def test_canonical_bytes_deterministic(self):
        """Canonical bytes produces same output for same input."""
        obj = {"key": "value", "number": 42}
        bytes1 = canonical_bytes(obj)
        bytes2 = canonical_bytes(obj)
        assert bytes1 == bytes2

    def test_blake3_digest_deterministic(self):
        """BLAKE3 digest is deterministic."""
        data = b"test data for hashing"
        digest1 = blake3_digest(data)
        digest2 = blake3_digest(data)
        assert digest1 == digest2
        assert len(digest1) == 32  # 256 bits

    def test_canon_query_digest_deterministic(self, query):
        """CanonQuery produces deterministic digest."""
        digest1 = query.hex_digest()
        digest2 = query.hex_digest()
        assert digest1 == digest2

    def test_canon_query_different_inputs_different_digests(self):
        """Different inputs produce different digests."""
        query1 = CanonQuery(
            user_id="user1",
            user_state="state1",
            intent="intent1",
        )
        query2 = CanonQuery(
            user_id="user2",
            user_state="state1",
            intent="intent1",
        )
        assert query1.hex_digest() != query2.hex_digest()

    def test_verify_determinism_passes(self, query):
        """Determinism verification passes for valid query."""
        result = verify_determinism(query, iterations=50)
        assert result["deterministic"] is True
        assert result["unique_hashes"] == 1

    def test_canon_policy_digest_deterministic(self, policy):
        """CanonPolicy produces deterministic digest."""
        digest1 = policy.hex_digest()
        digest2 = policy.hex_digest()
        assert digest1 == digest2

    def test_canon_environment_capture(self):
        """Environment capture works."""
        env = CanonEnvironment.capture()
        assert env.platform is not None
        assert env.python_version is not None
        assert env.cpu_count >= 1


# =============================================================================
# DELIVERABLE 2: SNR SCORING AS VERIFIABLE FUNCTION
# =============================================================================

class TestSNRScoring:
    """Tests for SNR scoring with audit trail."""

    def test_snr_engine_creation(self):
        """SNR engine can be created with default policy."""
        engine = SNREngine()
        assert engine.policy.snr_min == DEFAULT_SNR_POLICY["snr_min"]

    def test_snr_engine_custom_policy(self):
        """SNR engine accepts custom policy."""
        policy = SNRPolicy(snr_min=0.90)
        engine = SNREngine(policy)
        assert engine.policy.snr_min == 0.90

    def test_snr_compute_returns_trace(self, snr_engine):
        """SNR compute returns value and trace."""
        inputs = SNRInput(
            provenance_depth=2,
            source_trust_score=0.8,
            z3_satisfiable=True,
            ihsan_score=0.95,
        )
        snr, trace = snr_engine.compute(inputs)

        assert isinstance(snr, float)
        assert 0.0 <= snr <= 1.0
        assert isinstance(trace, SNRTrace)

    def test_snr_trace_contains_all_components(self, snr_engine):
        """SNR trace contains all computation components."""
        inputs = SNRInput()
        _, trace = snr_engine.compute(inputs)

        assert hasattr(trace, "provenance_score")
        assert hasattr(trace, "constraint_score")
        assert hasattr(trace, "prediction_score")
        assert hasattr(trace, "signal_mass")
        assert hasattr(trace, "noise_mass")
        assert hasattr(trace, "snr")
        assert hasattr(trace, "policy_digest")

    def test_snr_check_threshold_pass(self, snr_engine):
        """SNR check passes for high-quality input."""
        inputs = SNRInput(
            provenance_depth=5,
            corroboration_count=3,
            source_trust_score=0.95,
            z3_satisfiable=True,
            ihsan_score=0.99,
            contradiction_count=0,
            unverifiable_claims=0,
        )
        passed, snr, trace = snr_engine.check_threshold(inputs)
        assert passed is True
        assert snr >= snr_engine.policy.snr_min

    def test_snr_check_threshold_fail(self, snr_engine):
        """SNR check fails for low-quality input."""
        inputs = SNRInput(
            provenance_depth=0,
            source_trust_score=0.1,
            z3_satisfiable=False,
            ihsan_score=0.5,
            contradiction_count=10,
            unverifiable_claims=5,
        )
        passed, snr, trace = snr_engine.check_threshold(inputs)
        assert passed is False
        assert snr < snr_engine.policy.snr_min

    def test_snr_trace_verifiable(self, snr_engine):
        """SNR trace can be verified."""
        inputs = SNRInput(
            provenance_depth=3,
            source_trust_score=0.8,
        )
        _, trace = snr_engine.compute(inputs)

        # Verify the trace
        valid = snr_engine.verify_trace(trace)
        assert valid is True

    def test_snr_policy_digest_deterministic(self):
        """SNR policy digest is deterministic."""
        policy = SNRPolicy.default()
        digest1 = policy.hex_digest()
        digest2 = policy.hex_digest()
        assert digest1 == digest2

    def test_snr_compute_simple_api(self, snr_engine):
        """SNR engine has simplified compute API."""
        snr, trace = snr_engine.compute_simple(
            provenance_score=0.8,
            constraint_score=0.9,
            prediction_score=0.7,
        )
        assert isinstance(snr, float)
        assert isinstance(trace, SNRTrace)

    def test_snr_engine_stats(self, snr_engine):
        """SNR engine tracks statistics."""
        inputs = SNRInput()
        for _ in range(5):
            snr_engine.compute(inputs)

        stats = snr_engine.get_stats()
        assert stats["total_computations"] == 5


# =============================================================================
# DELIVERABLE 3: SIGNED REJECTION RECEIPTS
# =============================================================================

class TestSignedReceipts:
    """Tests for signed success/rejection receipts."""

    def test_simple_signer_sign_verify(self, signer):
        """SimpleSigner can sign and verify messages."""
        msg = b"test message"
        signature = signer.sign(msg)
        assert signer.verify(msg, signature) is True

    def test_simple_signer_reject_tampered(self, signer):
        """SimpleSigner rejects tampered messages."""
        msg = b"test message"
        signature = signer.sign(msg)
        tampered = b"tampered message"
        assert signer.verify(tampered, signature) is False

    def test_receipt_builder_accepted(self, receipt_builder, query, policy):
        """Receipt builder creates accepted receipts."""
        receipt = receipt_builder.accepted(
            query=query,
            policy=policy,
            payload=b"test payload",
            snr=0.98,
            ihsan_score=0.99,
        )

        assert receipt.status == ReceiptStatus.ACCEPTED
        assert receipt.snr == 0.98
        assert receipt.ihsan_score == 0.99
        assert len(receipt.signature) > 0

    def test_receipt_builder_rejected(self, receipt_builder, query, policy):
        """Receipt builder creates rejection receipts."""
        receipt = receipt_builder.rejected(
            query=query,
            policy=policy,
            snr=0.5,
            ihsan_score=0.6,
            gate_failed="snr",
            reason="SNR below threshold",
        )

        assert receipt.status == ReceiptStatus.REJECTED
        assert receipt.reason == "SNR below threshold"
        assert receipt.gate_passed == "snr"

    def test_receipt_builder_amber(self, receipt_builder, query, policy):
        """Receipt builder creates amber-restricted receipts."""
        receipt = receipt_builder.amber_restricted(
            query=query,
            policy=policy,
            payload=b"partial payload",
            snr=0.92,
            ihsan_score=0.94,
            restriction_reason="Safety concerns",
        )

        assert receipt.status == ReceiptStatus.AMBER_RESTRICTED
        assert "AMBER:" in receipt.reason

    def test_receipt_signature_valid(self, receipt_builder, query, policy, signer):
        """Receipt signature can be verified."""
        receipt = receipt_builder.accepted(
            query=query,
            policy=policy,
            payload=b"test",
            snr=0.98,
            ihsan_score=0.99,
        )

        assert receipt.verify_signature(signer) is True

    def test_receipt_verifier(self, receipt_builder, query, policy, signer):
        """Receipt verifier validates receipts."""
        receipt = receipt_builder.accepted(
            query=query,
            policy=policy,
            payload=b"test",
            snr=0.98,
            ihsan_score=0.99,
        )

        verifier = ReceiptVerifier(signer)
        valid, error = verifier.verify(receipt)
        assert valid is True
        assert error is None

    def test_receipt_verifier_stats(self, receipt_builder, query, policy, signer):
        """Receipt verifier tracks statistics."""
        verifier = ReceiptVerifier(signer)

        for _ in range(3):
            receipt = receipt_builder.accepted(
                query=query,
                policy=policy,
                payload=b"test",
                snr=0.98,
                ihsan_score=0.99,
            )
            verifier.verify(receipt)

        stats = verifier.get_stats()
        assert stats["total_verified"] == 3
        assert stats["total_failed"] == 0

    def test_receipt_digest_deterministic(self, receipt_builder, query, policy):
        """Receipt digest is deterministic."""
        receipt = receipt_builder.accepted(
            query=query,
            policy=policy,
            payload=b"test",
            snr=0.98,
            ihsan_score=0.99,
        )

        digest1 = receipt.hex_digest()
        digest2 = receipt.hex_digest()
        assert digest1 == digest2

    def test_receipt_to_dict(self, receipt_builder, query, policy):
        """Receipt can be serialized to dict."""
        receipt = receipt_builder.accepted(
            query=query,
            policy=policy,
            payload=b"test",
            snr=0.98,
            ihsan_score=0.99,
        )

        data = receipt.to_dict()
        assert "receipt_id" in data
        assert "status" in data
        assert "signature" in data
        assert data["status"] == "accepted"


# =============================================================================
# DELIVERABLE 4: BENCH-AS-RECEIPT HARNESS
# =============================================================================

class TestBenchHarness:
    """Tests for benchmark-to-receipt harness."""

    def test_bench_harness_creation(self, signer):
        """Bench harness can be created."""
        harness = BenchHarness(signer)
        assert harness.bench_iterations == 100

    def test_bench_fn_execution(self, signer):
        """Bench harness can benchmark a function."""
        harness = BenchHarness(signer, bench_iterations=10)

        result = harness.bench_fn("test_sum", lambda: sum(range(100)))

        assert result.name == "test_sum"
        assert result.iterations == 10
        assert len(result.samples) == 10

    def test_bench_result_statistics(self, signer):
        """Bench result computes statistics."""
        harness = BenchHarness(signer, bench_iterations=20)

        result = harness.bench_fn("test_op", lambda: [i**2 for i in range(10)])

        assert result.min_ns > 0
        assert result.max_ns >= result.min_ns
        assert result.mean_ns > 0
        assert result.p50_ns > 0
        assert result.p95_ns > 0
        assert result.p99_ns > 0

    def test_bench_result_throughput(self, signer):
        """Bench result computes throughput."""
        harness = BenchHarness(signer, bench_iterations=10)

        result = harness.bench_fn("fast_op", lambda: None)

        assert result.throughput_ops > 0

    def test_bench_to_receipt_pass(self, signer):
        """Bench converts to passing receipt."""
        harness = BenchHarness(signer, bench_iterations=10)

        result = harness.bench_fn("fast_op", lambda: None)

        # Claim very generous limits
        receipt = harness.bench_to_receipt(
            result,
            claimed_p99_us=1_000_000,  # 1 second
            claimed_throughput=1,       # 1 op/sec
            claimed_allocs=1_000_000,
        )

        assert receipt.verdict == "PASS"
        assert receipt.claims_verified is True

    def test_bench_to_receipt_fail(self, signer):
        """Bench converts to failing receipt for unmet claims."""
        harness = BenchHarness(signer, bench_iterations=10)

        # Slow operation
        result = harness.bench_fn("slow_op", lambda: time.sleep(0.001))

        # Claim impossible limits
        receipt = harness.bench_to_receipt(
            result,
            claimed_p99_us=1,           # 1 microsecond
            claimed_throughput=1_000_000,  # 1M ops/sec
            claimed_allocs=0,
        )

        assert receipt.verdict == "FAIL"
        assert receipt.claims_verified is False

    def test_bench_receipt_signature(self, signer):
        """Bench receipt has valid signature."""
        harness = BenchHarness(signer, bench_iterations=5)

        result = harness.bench_fn("test_op", lambda: None)
        receipt = harness.bench_to_receipt(
            result,
            claimed_p99_us=1_000_000,
            claimed_throughput=1,
            claimed_allocs=1_000_000,
        )

        assert receipt.verify_signature(signer) is True

    def test_bench_receipt_comparison(self, signer):
        """Bench receipts can be compared for regression."""
        harness = BenchHarness(signer, bench_iterations=10)

        # Baseline
        result1 = harness.bench_fn("baseline", lambda: None)
        receipt1 = harness.bench_to_receipt(
            result1,
            claimed_p99_us=100000,
            claimed_throughput=1,
            claimed_allocs=1000000,
        )

        # Current (same operation, should be similar)
        result2 = harness.bench_fn("current", lambda: None)
        receipt2 = harness.bench_to_receipt(
            result2,
            claimed_p99_us=100000,
            claimed_throughput=1,
            claimed_allocs=1000000,
        )

        comparison = harness.compare_receipts(receipt1, receipt2)

        assert "has_regression" in comparison
        assert "p99" in comparison
        assert "throughput" in comparison

    def test_bench_harness_stats(self, signer):
        """Bench harness tracks statistics."""
        harness = BenchHarness(signer, bench_iterations=5)

        for _ in range(3):
            result = harness.bench_fn("test", lambda: None)
            harness.bench_to_receipt(
                result,
                claimed_p99_us=1_000_000,
                claimed_throughput=1,
                claimed_allocs=1_000_000,
            )

        stats = harness.get_stats()
        assert stats["total_benchmarks"] == 3
        assert stats["total_receipts"] == 3


# =============================================================================
# 6-GATE CHAIN
# =============================================================================

class TestGateChain:
    """Tests for the 6-gate execution chain."""

    def test_gate_chain_creation(self, signer):
        """Gate chain can be created."""
        chain = GateChain(signer)
        assert len(chain.gates) == 6

    def test_gate_chain_order(self, signer):
        """Gate chain has correct order."""
        chain = GateChain(signer)
        gate_names = [g.name for g in chain.gates]
        assert gate_names == GATE_CHAIN

    def test_gate_chain_full_pass(self, gate_chain, query, policy):
        """Gate chain passes valid query."""
        context = {
            "trust_score": 0.9,
            "ihsan_score": 0.98,
            "z3_satisfiable": True,
        }

        result, receipt = gate_chain.evaluate(query, policy, context)

        assert result.passed is True
        assert result.final_status == GateStatus.PASSED
        assert result.last_gate_passed == "commit"
        assert receipt.status == ReceiptStatus.ACCEPTED

    def test_gate_chain_schema_fail(self, gate_chain, policy):
        """Gate chain fails on schema validation."""
        bad_query = CanonQuery(
            user_id="",  # Missing required field
            user_state="test",
            intent="test",
        )

        result, receipt = gate_chain.evaluate(bad_query, policy, {})

        assert result.passed is False
        assert result.final_status == GateStatus.FAILED
        assert receipt.status == ReceiptStatus.REJECTED

    def test_gate_chain_snr_fail(self, gate_chain, query, policy):
        """Gate chain fails on SNR check."""
        context = {
            "trust_score": 0.1,  # Low trust
            "ihsan_score": 0.5,  # Low ihsan
            "contradiction_count": 10,
            "unverifiable_claims": 5,
        }

        result, receipt = gate_chain.evaluate(query, policy, context)

        assert result.passed is False
        assert receipt.status == ReceiptStatus.REJECTED

    def test_gate_chain_constraint_fail(self, gate_chain, query, policy):
        """Gate chain fails on constraint check."""
        context = {
            "trust_score": 0.9,
            "ihsan_score": 0.5,  # Below threshold
            "z3_satisfiable": True,
        }

        result, receipt = gate_chain.evaluate(query, policy, context)

        assert result.passed is False
        assert "Ihsān" in (result.rejection_reason or "")

    def test_gate_chain_stats(self, gate_chain, query, policy):
        """Gate chain tracks statistics."""
        context = {"trust_score": 0.9, "ihsan_score": 0.98, "z3_satisfiable": True}

        for _ in range(5):
            gate_chain.evaluate(query, policy, context)

        stats = gate_chain.get_stats()
        assert stats["total_evaluations"] == 5
        assert stats["passed"] == 5


class TestIndividualGates:
    """Tests for individual gates."""

    def test_schema_gate_valid(self, query, policy):
        """Schema gate passes valid query."""
        gate = SchemaGate()
        result = gate.evaluate(query, policy, {})
        assert result.passed is True

    def test_schema_gate_long_intent(self, policy):
        """Schema gate fails on long intent."""
        gate = SchemaGate(max_intent_length=10)
        query = CanonQuery(
            user_id="test",
            user_state="test",
            intent="This intent is way too long",
        )
        result = gate.evaluate(query, policy, {})
        assert result.passed is False

    def test_provenance_gate_trusted(self, query, policy):
        """Provenance gate passes trusted source."""
        gate = ProvenanceGate(trusted_sources=["authenticated"])
        result = gate.evaluate(query, policy, {})
        assert result.passed is True

    def test_snr_gate_high_quality(self, query, policy):
        """SNR gate passes high-quality input."""
        gate = SNRGate()
        context = {
            "provenance_depth": 5,
            "source_trust_score": 0.9,
            "ihsan_score": 0.98,
        }
        result = gate.evaluate(query, policy, context)
        assert result.passed is True

    def test_constraint_gate_high_ihsan(self, query, policy):
        """Constraint gate passes high Ihsān."""
        gate = ConstraintGate()
        context = {"ihsan_score": 0.99, "z3_satisfiable": True}
        result = gate.evaluate(query, policy, context)
        assert result.passed is True

    def test_safety_gate_clean(self, query, policy):
        """Safety gate passes clean query."""
        gate = SafetyGate()
        result = gate.evaluate(query, policy, {})
        assert result.passed is True

    def test_safety_gate_blocked_pattern(self, policy):
        """Safety gate fails on blocked pattern."""
        gate = SafetyGate(blocked_patterns=["dangerous"])
        query = CanonQuery(
            user_id="test",
            user_state="test",
            intent="This is dangerous content",
        )
        result = gate.evaluate(query, policy, {})
        assert result.passed is False

    def test_commit_gate_resources_available(self, query, policy):
        """Commit gate passes when resources available."""
        gate = CommitGate()
        result = gate.evaluate(query, policy, {})
        assert result.passed is True

    def test_commit_gate_max_concurrent(self, query, policy):
        """Commit gate fails at max concurrent."""
        gate = CommitGate(max_concurrent_ops=1)
        gate._current_ops = 1
        result = gate.evaluate(query, policy, {})
        assert result.passed is False


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the complete proof engine."""

    def test_full_workflow_accepted(self, signer, query, policy):
        """Full workflow for accepted query."""
        # Create chain
        chain = GateChain(signer)

        # Evaluate
        context = {
            "trust_score": 0.95,
            "ihsan_score": 0.99,
            "z3_satisfiable": True,
        }
        result, receipt = chain.evaluate(query, policy, context)

        # Verify
        assert result.passed is True
        assert receipt.status == ReceiptStatus.ACCEPTED

        # Verify signature
        verifier = ReceiptVerifier(signer)
        valid, _ = verifier.verify(receipt)
        assert valid is True

    def test_full_workflow_rejected(self, signer, query, policy):
        """Full workflow for rejected query."""
        chain = GateChain(signer)

        context = {
            "trust_score": 0.1,
            "ihsan_score": 0.3,
            "contradiction_count": 10,
        }
        result, receipt = chain.evaluate(query, policy, context)

        assert result.passed is False
        assert receipt.status == ReceiptStatus.REJECTED
        assert receipt.reason is not None

    def test_determinism_across_runs(self, query):
        """Verify determinism across multiple runs."""
        digests = []
        for _ in range(10):
            # Create fresh query
            q = CanonQuery(
                user_id="test",
                user_state="state",
                intent="intent",
                nonce="fixed_nonce",
            )
            digests.append(q.hex_digest())

        assert len(set(digests)) == 1  # All same

    def test_receipt_chain_audit_trail(self, signer, query, policy):
        """Receipts form a verifiable audit trail."""
        chain = GateChain(signer)
        context = {"trust_score": 0.9, "ihsan_score": 0.98, "z3_satisfiable": True}

        receipts = []
        for i in range(5):
            q = CanonQuery(
                user_id=f"user_{i}",
                user_state="authenticated",
                intent=f"Query {i}",
            )
            _, receipt = chain.evaluate(q, policy, context)
            receipts.append(receipt)

        # All receipts have unique IDs
        receipt_ids = [r.receipt_id for r in receipts]
        assert len(set(receipt_ids)) == 5

        # All signatures valid
        verifier = ReceiptVerifier(signer)
        for receipt in receipts:
            valid, _ = verifier.verify(receipt)
            assert valid is True
