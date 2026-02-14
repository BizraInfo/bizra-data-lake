"""Tests for core.spearpoint.sovereign_spearpoint -- Unified Foundation.

Covers:
- SpearheadStatus, MemoryType, CircuitState enums
- SpearheadConfig, MemoryEntry, Z3Proof, SpearheadResult data classes
- CircuitBreaker: state transitions, request gating
- MemorySystem: encode, retrieve, overflow
- Z3Verifier: constraint checking (simulation mode)
- SovereignSpearpoint: configuration, SNR/Ihsan calculation
"""

import time

import pytest

from core.spearpoint.sovereign_spearpoint import (
    CircuitBreaker,
    CircuitState,
    MemoryEntry,
    MemorySystem,
    MemoryType,
    SpearheadConfig,
    SpearheadResult,
    SpearheadStatus,
    SovereignSpearpoint,
    Z3Proof,
    Z3Verifier,
)


# ---------------------------------------------------------------------------
# ENUM TESTS
# ---------------------------------------------------------------------------


class TestEnums:

    def test_spearhead_statuses(self):
        assert len(SpearheadStatus) == 10

    def test_memory_types(self):
        expected = {"EPISODIC", "SEMANTIC", "PROCEDURAL", "WORKING", "PROSPECTIVE"}
        actual = {mt.name for mt in MemoryType}
        assert actual == expected

    def test_circuit_states(self):
        expected = {"CLOSED", "OPEN", "HALF_OPEN"}
        actual = {cs.name for cs in CircuitState}
        assert actual == expected


# ---------------------------------------------------------------------------
# DATA CLASS TESTS
# ---------------------------------------------------------------------------


class TestSpearheadConfig:

    def test_defaults(self):
        config = SpearheadConfig()
        assert config.snr_floor == 0.85
        assert config.ihsan_threshold == 0.95
        assert config.circuit_failure_threshold == 5
        assert config.max_iterations == 10


class TestMemoryEntry:

    def test_to_dict(self):
        from datetime import datetime, timezone

        entry = MemoryEntry(
            id="mem_test",
            content="Test content that is quite long " * 10,
            memory_type=MemoryType.EPISODIC,
            timestamp=datetime.now(timezone.utc),
            snr_score=0.9,
            ihsan_score=0.95,
            importance=0.7,
        )
        d = entry.to_dict()
        assert d["id"] == "mem_test"
        assert d["type"] == "EPISODIC"
        assert len(d["content"]) <= 104  # Truncated at 100 + "..."


class TestZ3Proof:

    def test_satisfiable_proof(self):
        proof = Z3Proof(
            proof_id="z3_test",
            constraints_checked=["ihsan >= 0.95", "snr >= 0.85"],
            satisfiable=True,
            model={"mode": "simulation"},
            generation_time_ms=5,
        )
        d = proof.to_dict()
        assert d["satisfiable"] is True
        assert d["counterexample"] is None

    def test_unsatisfiable_proof(self):
        proof = Z3Proof(
            proof_id="z3_fail",
            constraints_checked=["ihsan >= 0.95"],
            satisfiable=False,
            model={},
            generation_time_ms=3,
            counterexample="ihsan 0.800 < 0.95",
        )
        assert proof.counterexample is not None


class TestSpearheadResult:

    def test_to_dict(self):
        result = SpearheadResult(
            session_id="sess_test",
            status=SpearheadStatus.COMPLETED,
            output="Test output " * 50,
            snr_score=0.92,
            ihsan_score=0.96,
            iterations=1,
            elapsed_seconds=0.5,
        )
        d = result.to_dict()
        assert d["status"] == "COMPLETED"
        assert len(d["output"]) <= 204


# ---------------------------------------------------------------------------
# CircuitBreaker TESTS
# ---------------------------------------------------------------------------


class TestCircuitBreaker:

    @pytest.fixture
    def breaker(self):
        return CircuitBreaker(name="test", failure_threshold=3, recovery_timeout=0.1)

    def test_initial_state_closed(self, breaker):
        assert breaker.state == CircuitState.CLOSED

    def test_allows_request_when_closed(self, breaker):
        assert breaker.allow_request() is True

    def test_opens_after_threshold_failures(self, breaker):
        for _ in range(3):
            breaker.record_failure()
        assert breaker._state == CircuitState.OPEN

    def test_denies_request_when_open(self, breaker):
        for _ in range(3):
            breaker.record_failure()
        assert breaker.allow_request() is False

    def test_transitions_to_half_open_after_timeout(self, breaker):
        for _ in range(3):
            breaker.record_failure()
        assert breaker._state == CircuitState.OPEN
        # Wait for recovery timeout
        time.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN

    def test_closes_on_success_from_half_open(self, breaker):
        for _ in range(3):
            breaker.record_failure()
        time.sleep(0.15)
        _ = breaker.state  # Trigger transition to HALF_OPEN
        breaker.record_success()
        assert breaker._state == CircuitState.CLOSED

    def test_reopens_on_failure_from_half_open(self, breaker):
        for _ in range(3):
            breaker.record_failure()
        time.sleep(0.15)
        _ = breaker.state  # Trigger transition to HALF_OPEN
        breaker.record_failure()
        assert breaker._state == CircuitState.OPEN

    def test_success_decrements_failure_count(self, breaker):
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()
        assert breaker._failure_count == 1

    def test_get_metrics(self, breaker):
        breaker.record_success()
        metrics = breaker.get_metrics()
        assert metrics["name"] == "test"
        assert metrics["state"] == "CLOSED"
        assert metrics["success_count"] == 1


# ---------------------------------------------------------------------------
# MemorySystem TESTS
# ---------------------------------------------------------------------------


class TestMemorySystem:

    @pytest.fixture
    def memory(self):
        return MemorySystem(working_limit=3, snr_floor=0.5)

    def test_encode_valid_memory(self, memory):
        entry = memory.encode(
            content="Test memory",
            memory_type=MemoryType.SEMANTIC,
            snr_score=0.9,
            ihsan_score=0.95,
        )
        assert entry is not None
        assert entry.id.startswith("mem_")

    def test_encode_rejected_low_snr(self, memory):
        entry = memory.encode(
            content="Low quality",
            memory_type=MemoryType.SEMANTIC,
            snr_score=0.3,
            ihsan_score=0.5,
        )
        assert entry is None

    def test_working_memory_overflow(self, memory):
        # Fill working memory beyond limit
        for i in range(5):
            memory.encode(
                content=f"Working memory entry {i}",
                memory_type=MemoryType.WORKING,
                snr_score=0.9,
                ihsan_score=0.95,
            )
        # Working queue should be capped at limit
        assert len(memory._working_queue) <= 3
        # Overflowed entries should be in EPISODIC
        assert len(memory._memories[MemoryType.EPISODIC]) == 2

    def test_retrieve_by_keyword(self, memory):
        memory.encode(
            content="Python programming language",
            memory_type=MemoryType.SEMANTIC,
            snr_score=0.9,
            ihsan_score=0.95,
            importance=0.8,
        )
        memory.encode(
            content="Rust systems language",
            memory_type=MemoryType.SEMANTIC,
            snr_score=0.9,
            ihsan_score=0.95,
            importance=0.8,
        )
        results = memory.retrieve("Python programming")
        assert len(results) >= 1
        # Python entry should rank higher
        assert "Python" in results[0].content

    def test_retrieve_empty_query(self, memory):
        results = memory.retrieve("")
        assert isinstance(results, list)

    def test_get_working_context(self, memory):
        memory.encode("Recent context", MemoryType.WORKING, 0.9, 0.95)
        context = memory.get_working_context(limit=5)
        assert len(context) == 1

    def test_get_statistics(self, memory):
        memory.encode("Test", MemoryType.SEMANTIC, 0.9, 0.95)
        stats = memory.get_statistics()
        assert stats["total_memories"] == 1
        assert "SEMANTIC" in stats["by_type"]


# ---------------------------------------------------------------------------
# Z3Verifier TESTS (simulation mode)
# ---------------------------------------------------------------------------


class TestZ3Verifier:

    @pytest.fixture
    def verifier(self):
        return Z3Verifier(timeout_ms=1000)

    def test_satisfiable_all_constraints_met(self, verifier):
        proof = verifier.verify(
            ihsan_score=0.96,
            snr_score=0.90,
            risk_level=0.3,
            cost=0.1,
            autonomy_limit=1.0,
            reversible=True,
        )
        assert proof.satisfiable is True
        assert proof.counterexample is None

    def test_unsatisfiable_low_ihsan(self, verifier):
        proof = verifier.verify(
            ihsan_score=0.80,
            snr_score=0.90,
        )
        assert proof.satisfiable is False
        assert "ihsan" in proof.counterexample

    def test_unsatisfiable_low_snr(self, verifier):
        proof = verifier.verify(
            ihsan_score=0.96,
            snr_score=0.70,
        )
        assert proof.satisfiable is False
        assert "snr" in proof.counterexample

    def test_unsatisfiable_high_risk_no_reversibility(self, verifier):
        proof = verifier.verify(
            ihsan_score=0.96,
            snr_score=0.90,
            risk_level=0.9,
            reversible=False,
            human_approved=False,
        )
        assert proof.satisfiable is False
        assert "risk" in proof.counterexample.lower()

    def test_high_risk_with_approval(self, verifier):
        proof = verifier.verify(
            ihsan_score=0.96,
            snr_score=0.90,
            risk_level=0.9,
            reversible=False,
            human_approved=True,
        )
        assert proof.satisfiable is True

    def test_cost_exceeds_limit(self, verifier):
        proof = verifier.verify(
            ihsan_score=0.96,
            snr_score=0.90,
            cost=5.0,
            autonomy_limit=1.0,
        )
        assert proof.satisfiable is False
        assert "cost" in proof.counterexample


# ---------------------------------------------------------------------------
# SovereignSpearpoint TESTS
# ---------------------------------------------------------------------------


class TestSovereignSpearpoint:

    @pytest.fixture
    def spearpoint(self):
        return SovereignSpearpoint(SpearheadConfig(require_z3_proof=False))

    def test_initialization(self, spearpoint):
        assert spearpoint._session_id is not None
        assert len(spearpoint._session_id) == 12

    def test_calculate_snr(self, spearpoint):
        content = "This is a diverse and unique analysis with many different words"
        snr = spearpoint._calculate_snr(content, {})
        assert 0.0 <= snr <= 1.0

    def test_calculate_snr_with_evidence(self, spearpoint):
        content = "Analysis with grounded evidence"
        snr_no_evidence = spearpoint._calculate_snr(content, {})
        snr_with_evidence = spearpoint._calculate_snr(content, {"has_evidence": True})
        assert snr_with_evidence >= snr_no_evidence

    def test_calculate_ihsan(self, spearpoint):
        content = "A thoughtful response however we must consider the implications"
        snr = 0.85
        ihsan = spearpoint._calculate_ihsan(content, snr)
        assert 0.0 <= ihsan <= 1.0

    def test_get_statistics(self, spearpoint):
        stats = spearpoint.get_statistics()
        assert "session_id" in stats
        assert "cycles" in stats
        assert stats["cycles"] == 0


@pytest.mark.timeout(60)
class TestSpearheadExecution:

    @pytest.mark.asyncio
    async def test_execute_simulated(self):
        sp = SovereignSpearpoint(SpearheadConfig(require_z3_proof=False))
        result = await sp.execute("What is the meaning of sovereignty?")
        assert result.status == SpearheadStatus.COMPLETED
        assert result.output != ""
        assert result.elapsed_seconds > 0

    @pytest.mark.asyncio
    async def test_execute_with_z3_verification(self):
        sp = SovereignSpearpoint(SpearheadConfig(require_z3_proof=True))
        result = await sp.execute("Test query", context={"risk_level": 0.2})
        # May be COMPLETED or GATED depending on SNR/Ihsan scores
        assert result.status in {SpearheadStatus.COMPLETED, SpearheadStatus.GATED}
        assert len(result.loop_trace) > 0

    @pytest.mark.asyncio
    async def test_execute_records_memories(self):
        sp = SovereignSpearpoint(SpearheadConfig(require_z3_proof=False))
        result = await sp.execute("Remember this test query")
        if result.status == SpearheadStatus.COMPLETED:
            assert len(result.memories_created) > 0
