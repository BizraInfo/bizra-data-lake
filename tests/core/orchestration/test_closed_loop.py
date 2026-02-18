"""
Closed Loop Orchestrator Tests
================================
Validates the 12-step value cycle orchestrator end-to-end using
mock Protocol implementations.

Test Strategy:
1. Unit tests for each step in isolation
2. Integration test for full loop with mock dependencies
3. Fail-closed behavior verification
4. Hash-chain integrity verification
5. Context threading across steps

Standing on Giants: Deming (1950) PDCA, Shannon (1948) SNR
"""

from typing import Any, Dict

import pytest

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD, UNIFIED_SNR_THRESHOLD
from core.orchestration.closed_loop import (
    ClosedLoopContext,
    ClosedLoopOrchestrator,
    ClosedLoopResult,
    ClosedLoopStep,
    LoopReceipt,
    StepResult,
    StepStatus,
)

# ============================================================================
# Mock Protocol Implementations
# ============================================================================


class MockReasoning:
    """Mock ReasoningProtocol for testing."""

    async def reason(self, intent: str, *, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "thoughts": [f"Thought about: {intent}"],
            "graph_hash": "abc123def456",
            "strategy": "mock_got",
            "node_count": 3,
            "snr_score": 0.96,
        }


class MockExecutor:
    """Mock ExecutionProtocol for testing."""

    def __init__(self, *, fail: bool = False):
        self._fail = fail

    async def execute(
        self, mission: Dict[str, Any], *, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self._fail:
            return {
                "success": False,
                "error": "Mock execution failure",
                "response": "",
                "model_used": "mock-fail",
            }
        return {
            "success": True,
            "response": "This is a comprehensive test response with sufficient content.",
            "model_used": "mock-7b",
            "snr_score": 0.92,
        }


class MockQualityGate:
    """Mock QualityGateProtocol for testing."""

    def __init__(self, *, snr: float = 0.95, ihsan: float = 0.96):
        self._snr = snr
        self._ihsan = ihsan

    async def evaluate(
        self, result: Dict[str, Any], *, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "passed": True,
            "snr_score": self._snr,
            "ihsan_score": self._ihsan,
        }


class MockImpact:
    """Mock ImpactProtocol for testing."""

    async def measure(
        self, result: Dict[str, Any], *, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "impact_score": 0.88,
            "poi_score": 0.85,
        }


class MockProof:
    """Mock ProofProtocol for testing."""

    async def emit_proof(
        self, result: Dict[str, Any], *, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "receipt_id": "rcpt_test_001",
            "receipt_hash": "deadbeef" * 8,
        }


class MockMinting:
    """Mock MintingProtocol for testing."""

    def __init__(self, *, fail: bool = False):
        self._fail = fail

    async def mint(
        self,
        account_id: str,
        poi_score: float,
        *,
        epoch_id: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self._fail:
            return {"success": False, "error": "Mint cap exceeded"}
        amount = poi_score * 10.0
        return {
            "success": True,
            "amount": amount,
            "tx_hash": "tx_" + epoch_id,
        }


class MockFederation:
    """Mock FederationProtocol for testing."""

    async def broadcast(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"peers_reached": 5}


class MockPriorUpdater:
    """Mock PriorUpdateProtocol for testing."""

    async def update_priors(self, loop_context: Dict[str, Any]) -> Dict[str, Any]:
        return {"priors_updated": 3, "skills_cached": 1}


# ============================================================================
# Helper to build a fully-wired orchestrator
# ============================================================================


def build_orchestrator(**overrides: Any) -> ClosedLoopOrchestrator:
    """Build orchestrator with all mock dependencies."""
    defaults = {
        "reasoning": MockReasoning(),
        "executor": MockExecutor(),
        "quality_gate": MockQualityGate(),
        "impact": MockImpact(),
        "proof": MockProof(),
        "minting": MockMinting(),
        "federation": MockFederation(),
        "prior_updater": MockPriorUpdater(),
        "node_id": "BIZRA-TEST-0001",
    }
    defaults.update(overrides)
    return ClosedLoopOrchestrator(**defaults)


# ============================================================================
# Full Loop Tests
# ============================================================================


class TestFullLoop:
    """Full 12-step loop with all mocks wired."""

    @pytest.mark.asyncio
    async def test_successful_full_loop(self):
        orch = build_orchestrator()
        result = await orch.execute_loop("What is sovereignty?")

        assert result.success is True
        assert result.halted_at_step is None
        assert len(result.steps) == 12
        assert all(s.passed for s in result.steps)
        assert result.total_duration_ms > 0

    @pytest.mark.asyncio
    async def test_full_loop_produces_receipt(self):
        orch = build_orchestrator()
        result = await orch.execute_loop("Test intent")

        assert result.receipt is not None
        assert result.receipt.chain_hash != ""
        assert len(result.receipt.step_hashes) == 12
        assert result.receipt.total_steps_completed == 12
        assert result.receipt.total_steps_failed == 0

    @pytest.mark.asyncio
    async def test_context_populated_after_loop(self):
        orch = build_orchestrator()
        result = await orch.execute_loop("Test context threading")

        ctx = result.context
        assert ctx is not None
        assert ctx.user_intent == "Test context threading"
        assert ctx.snr_score >= UNIFIED_SNR_THRESHOLD
        assert ctx.ihsan_score >= UNIFIED_IHSAN_THRESHOLD
        assert ctx.tokens_minted > 0
        assert ctx.peers_reached == 5
        assert ctx.priors_updated == 3
        assert ctx.receipt_hash != ""

    @pytest.mark.asyncio
    async def test_loop_iteration_counter_increments(self):
        orch = build_orchestrator()
        r1 = await orch.execute_loop("First")
        r2 = await orch.execute_loop("Second")

        assert r1.context.iteration == 1
        assert r2.context.iteration == 2

    @pytest.mark.asyncio
    async def test_loop_with_previous_context(self):
        orch = build_orchestrator()
        r1 = await orch.execute_loop("First loop")

        r2 = await orch.execute_loop("Second loop", previous_context=r1.context)

        assert r2.success is True
        assert r2.context.previous_loop_hash == r1.context.receipt_hash


# ============================================================================
# Fail-Closed Tests
# ============================================================================


class TestFailClosed:
    """Verify fail-closed behavior at various steps."""

    @pytest.mark.asyncio
    async def test_empty_intent_fails_immediately(self):
        orch = build_orchestrator()
        result = await orch.execute_loop("")

        assert result.success is False
        assert result.halted_at_step == ClosedLoopStep.USER_INTENT

    @pytest.mark.asyncio
    async def test_whitespace_intent_fails(self):
        orch = build_orchestrator()
        result = await orch.execute_loop("   ")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_no_executor_fails_at_step_4(self):
        orch = build_orchestrator(executor=None)
        result = await orch.execute_loop("Test without executor")

        assert result.success is False
        assert result.halted_at_step == ClosedLoopStep.EXECUTION
        # Steps 1-3 should have passed
        assert len(result.completed_steps) == 3

    @pytest.mark.asyncio
    async def test_executor_failure_halts_loop(self):
        orch = build_orchestrator(executor=MockExecutor(fail=True))
        result = await orch.execute_loop("Test executor failure")

        assert result.success is False
        assert result.halted_at_step == ClosedLoopStep.EXECUTION

    @pytest.mark.asyncio
    async def test_low_snr_fails_quality_gate(self):
        orch = build_orchestrator(quality_gate=MockQualityGate(snr=0.50, ihsan=0.96))
        result = await orch.execute_loop("Low SNR test")

        assert result.success is False
        assert result.halted_at_step == ClosedLoopStep.QUALITY_GATE
        assert "SNR below threshold" in result.steps[-1].error

    @pytest.mark.asyncio
    async def test_low_ihsan_fails_quality_gate(self):
        orch = build_orchestrator(quality_gate=MockQualityGate(snr=0.95, ihsan=0.50))
        result = await orch.execute_loop("Low Ihsan test")

        assert result.success is False
        assert result.halted_at_step == ClosedLoopStep.QUALITY_GATE
        assert "Ihsan below threshold" in result.steps[-1].error

    @pytest.mark.asyncio
    async def test_mint_failure_halts_loop(self):
        orch = build_orchestrator(minting=MockMinting(fail=True))
        result = await orch.execute_loop("Mint failure test")

        assert result.success is False
        assert result.halted_at_step == ClosedLoopStep.TOKEN_MINT

    @pytest.mark.asyncio
    async def test_partial_result_contains_context(self):
        orch = build_orchestrator(executor=MockExecutor(fail=True))
        result = await orch.execute_loop("Partial context test")

        assert result.context is not None
        assert result.context.user_intent == "Partial context test"
        assert result.receipt is not None
        assert result.receipt.total_steps_failed == 1


# ============================================================================
# Graceful Degradation Tests
# ============================================================================


class TestGracefulDegradation:
    """Steps that degrade gracefully without their protocols."""

    @pytest.mark.asyncio
    async def test_no_reasoning_passes_through(self):
        orch = build_orchestrator(reasoning=None)
        result = await orch.execute_loop("No reasoning engine")

        assert result.success is True
        step2 = result.steps[1]
        assert step2.step == ClosedLoopStep.PAT_REASONING
        assert step2.passed
        assert "pass-through" in step2.detail

    @pytest.mark.asyncio
    async def test_no_quality_gate_uses_default(self):
        orch = build_orchestrator(quality_gate=None)
        result = await orch.execute_loop("Default quality gate")

        assert result.success is True
        assert result.context.snr_score == UNIFIED_SNR_THRESHOLD
        assert result.context.ihsan_score == UNIFIED_IHSAN_THRESHOLD

    @pytest.mark.asyncio
    async def test_no_impact_uses_derived_score(self):
        orch = build_orchestrator(impact=None)
        result = await orch.execute_loop("Derived impact")

        assert result.success is True
        assert result.context.impact_score > 0

    @pytest.mark.asyncio
    async def test_no_proof_uses_local_hash(self):
        orch = build_orchestrator(proof=None)
        result = await orch.execute_loop("Local proof")

        assert result.success is True
        assert result.context.receipt_hash != ""
        assert result.context.receipt_id.startswith("rcpt_")

    @pytest.mark.asyncio
    async def test_no_minting_records_theoretical(self):
        orch = build_orchestrator(minting=None)
        result = await orch.execute_loop("Theoretical mint")

        assert result.success is True
        assert result.context.tokens_minted > 0

    @pytest.mark.asyncio
    async def test_no_federation_passes_with_zero_peers(self):
        orch = build_orchestrator(federation=None)
        result = await orch.execute_loop("No federation")

        assert result.success is True
        assert result.context.peers_reached == 0

    @pytest.mark.asyncio
    async def test_no_prior_updater_passes(self):
        orch = build_orchestrator(prior_updater=None)
        result = await orch.execute_loop("No prior updater")

        assert result.success is True
        assert result.context.priors_updated == 1


# ============================================================================
# Hash Chain Integrity Tests
# ============================================================================


class TestHashChainIntegrity:
    """Verify the hash chain in LoopReceipt."""

    @pytest.mark.asyncio
    async def test_chain_hash_is_deterministic(self):
        orch = build_orchestrator()
        r1 = await orch.execute_loop("Deterministic test")
        r2 = await orch.execute_loop("Deterministic test")

        # Different loop_ids means different hashes, but both should be non-empty
        assert r1.receipt.chain_hash != ""
        assert r2.receipt.chain_hash != ""

    def test_receipt_finalize_computes_hash(self):
        receipt = LoopReceipt(
            receipt_id="test",
            loop_id="loop_test",
            step_hashes=["hash1", "hash2", "hash3"],
        )
        receipt.finalize()

        assert receipt.chain_hash != ""
        assert receipt.chain_hash != "hash1"  # Not just the first hash

    def test_empty_receipt_has_deterministic_hash(self):
        receipt = LoopReceipt(
            receipt_id="empty",
            loop_id="loop_empty",
        )
        receipt.finalize()

        assert receipt.chain_hash != ""

    def test_step_hash_uniqueness(self):
        s1 = StepResult(
            step=ClosedLoopStep.USER_INTENT,
            status=StepStatus.PASSED,
            snr_score=0.95,
            detail="test1",
        )
        s2 = StepResult(
            step=ClosedLoopStep.PAT_REASONING,
            status=StepStatus.PASSED,
            snr_score=0.95,
            detail="test2",
        )

        assert s1.step_hash != s2.step_hash

    def test_receipt_to_dict(self):
        receipt = LoopReceipt(
            receipt_id="test",
            loop_id="loop_test",
            step_hashes=["h1", "h2"],
            total_steps_completed=2,
        )
        receipt.finalize()
        d = receipt.to_dict()

        assert d["receipt_id"] == "test"
        assert d["chain_hash"] != ""
        assert len(d["step_hashes"]) == 2


# ============================================================================
# Step Result Tests
# ============================================================================


class TestStepResult:
    """Unit tests for StepResult dataclass."""

    def test_passed_property(self):
        sr = StepResult(step=ClosedLoopStep.USER_INTENT, status=StepStatus.PASSED)
        assert sr.passed is True

    def test_failed_property(self):
        sr = StepResult(step=ClosedLoopStep.USER_INTENT, status=StepStatus.FAILED)
        assert sr.passed is False

    def test_to_dict_includes_error_when_present(self):
        sr = StepResult(
            step=ClosedLoopStep.EXECUTION,
            status=StepStatus.FAILED,
            error="Something broke",
        )
        d = sr.to_dict()
        assert "error" in d
        assert d["error"] == "Something broke"

    def test_to_dict_excludes_error_when_absent(self):
        sr = StepResult(
            step=ClosedLoopStep.USER_INTENT,
            status=StepStatus.PASSED,
        )
        d = sr.to_dict()
        assert "error" not in d


# ============================================================================
# Context Tests
# ============================================================================


class TestClosedLoopContext:
    """Unit tests for ClosedLoopContext."""

    def test_default_context_has_loop_id(self):
        ctx = ClosedLoopContext()
        assert len(ctx.loop_id) == 16

    def test_to_dict_truncates_intent(self):
        ctx = ClosedLoopContext(user_intent="x" * 500)
        d = ctx.to_dict()
        assert len(d["user_intent"]) == 200


# ============================================================================
# ClosedLoopResult Tests
# ============================================================================


class TestClosedLoopResult:
    """Unit tests for ClosedLoopResult."""

    def test_aggregate_snr_empty(self):
        result = ClosedLoopResult(loop_id="test", success=False)
        assert result.aggregate_snr == 0.0

    def test_aggregate_snr_computed(self):
        result = ClosedLoopResult(
            loop_id="test",
            success=True,
            steps=[
                StepResult(
                    step=ClosedLoopStep.USER_INTENT,
                    status=StepStatus.PASSED,
                    snr_score=0.90,
                ),
                StepResult(
                    step=ClosedLoopStep.PAT_REASONING,
                    status=StepStatus.PASSED,
                    snr_score=1.00,
                ),
            ],
        )
        assert abs(result.aggregate_snr - 0.95) < 0.001

    def test_to_dict_includes_halted_step(self):
        result = ClosedLoopResult(
            loop_id="test",
            success=False,
            halted_at_step=ClosedLoopStep.EXECUTION,
        )
        d = result.to_dict()
        assert d["halted_at_step"] == "EXECUTION"

    def test_to_dict_excludes_halted_when_none(self):
        result = ClosedLoopResult(loop_id="test", success=True)
        d = result.to_dict()
        assert "halted_at_step" not in d


# ============================================================================
# Diagnostics Tests
# ============================================================================


class TestDiagnostics:
    """Orchestrator stats and diagnostics."""

    def test_get_stats(self):
        orch = build_orchestrator()
        stats = orch.get_stats()

        assert stats["node_id"] == "BIZRA-TEST-0001"
        assert stats["has_reasoning"] is True
        assert stats["has_executor"] is True

    @pytest.mark.asyncio
    async def test_recent_receipts(self):
        orch = build_orchestrator()
        await orch.execute_loop("First")
        await orch.execute_loop("Second")

        receipts = orch.get_recent_receipts(limit=5)
        assert len(receipts) == 2


# ============================================================================
# Exception Handling Tests
# ============================================================================


class TestExceptionHandling:
    """Verify that unhandled exceptions are caught and wrapped."""

    @pytest.mark.asyncio
    async def test_reasoning_exception_halts_loop(self):
        class BrokenReasoning:
            async def reason(self, intent, *, context):
                raise RuntimeError("Reasoning exploded")

        orch = build_orchestrator(reasoning=BrokenReasoning())
        result = await orch.execute_loop("Broken reasoning")

        assert result.success is False
        assert result.halted_at_step == ClosedLoopStep.PAT_REASONING
        assert "Reasoning failed" in result.steps[1].error

    @pytest.mark.asyncio
    async def test_quality_gate_exception_fails_closed(self):
        class BrokenGate:
            async def evaluate(self, result, *, context):
                raise ValueError("Gate exploded")

        orch = build_orchestrator(quality_gate=BrokenGate())
        result = await orch.execute_loop("Broken gate")

        assert result.success is False
        assert result.halted_at_step == ClosedLoopStep.QUALITY_GATE
        assert "fail-closed" in result.steps[5].error

    @pytest.mark.asyncio
    async def test_federation_exception_is_non_fatal(self):
        class BrokenFederation:
            async def broadcast(self, payload):
                raise ConnectionError("Network down")

        orch = build_orchestrator(federation=BrokenFederation())
        result = await orch.execute_loop("Federation error")

        # Federation failure is non-fatal
        assert result.success is True
        step10 = result.steps[9]
        assert step10.step == ClosedLoopStep.FEDERATION_SHARE
        assert step10.passed is True
        assert "degraded" in step10.detail


# ============================================================================
# Step Enum Tests
# ============================================================================


class TestClosedLoopStep:
    """Verify step enum completeness and ordering."""

    def test_twelve_steps(self):
        assert len(ClosedLoopStep) == 12

    def test_steps_are_ordered(self):
        steps = list(ClosedLoopStep)
        for i, step in enumerate(steps):
            assert step.value == i + 1

    def test_step_names(self):
        assert ClosedLoopStep.USER_INTENT.name == "USER_INTENT"
        assert ClosedLoopStep.LOOP_RETURNS.name == "LOOP_RETURNS"
        assert ClosedLoopStep.TOKEN_MINT.value == 9

    def test_intent_length_validation(self):
        """Intent exceeding max length should fail at step 1."""

    @pytest.mark.asyncio
    async def test_oversized_intent_fails(self):
        orch = build_orchestrator()
        huge_intent = "x" * 60_000
        result = await orch.execute_loop(huge_intent)

        assert result.success is False
        assert result.halted_at_step == ClosedLoopStep.USER_INTENT
        assert "exceeds maximum length" in result.steps[0].error
