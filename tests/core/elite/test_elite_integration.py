"""
Elite Integration Tests — Comprehensive Test Suite

Tests for the four new elite integration patterns:
1. Hook-First Governance (FATE Gate)
2. Session as State Machine (Merkle-DAG)
3. Thinking Budget Allocation (7-3-6-9)
4. Permission as Market (Harberger Tax + Gini)

Target: SNR >= 0.95 (Ihsan threshold)

Standing on Giants: pytest + asyncio + property-based testing
"""

import asyncio
import hashlib
import math
import pytest
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

# Import modules under test
from core.elite.hooks import (
    FATEGate,
    FATEScore,
    HookRegistry,
    HookExecutor,
    HookPhase,
    HookPriority,
    HookContext,
    HookResult,
    FATEDimension,
    register_hook,
    fate_guarded,
    FATEGateError,
)
from core.elite.session_dag import (
    MerkleDAG,
    MerkleNode,
    SessionStateMachine,
    SessionState,
    TransitionType,
    create_session,
    InvalidTransitionError,
    DAGError,
)
from core.elite.cognitive_budget import (
    CognitiveBudgetAllocator,
    BudgetAllocation,
    BudgetTier,
    BudgetTracker,
    BudgetUsage,
    TaskType,
    ComplexitySignal,
    allocate_budget,
    DNA_DEPTH,
    DNA_TRACKS,
    DNA_BALANCE,
    DNA_COMPLETE,
)
from core.elite.compute_market import (
    ComputeMarket,
    ComputeLicense,
    ResourceUnit,
    ResourceType,
    LicenseStatus,
    MarketTransaction,
    create_market,
    create_inference_license,
    DEFAULT_TAX_RATE,
    GINI_THRESHOLD,
)
from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)


# ============================================================================
# FATE GATE TESTS
# ============================================================================

class TestFATEScore:
    """Tests for FATEScore dataclass."""

    def test_overall_score_calculation(self):
        """Test weighted geometric mean calculation."""
        score = FATEScore(
            fidelity=0.9,
            accountability=0.95,
            transparency=0.92,
            ethics=0.98,
        )

        # Weighted geometric mean
        expected = (
            (0.9 ** 0.25) *
            (0.95 ** 0.25) *
            (0.92 ** 0.25) *
            (0.98 ** 0.25)
        )

        assert abs(score.overall - expected) < 0.001

    def test_passed_threshold(self):
        """Test Ihsan threshold check."""
        passing = FATEScore(
            fidelity=0.96,
            accountability=0.96,
            transparency=0.96,
            ethics=0.96,
        )
        assert passing.passed is True

        failing = FATEScore(
            fidelity=0.8,
            accountability=0.8,
            transparency=0.8,
            ethics=0.8,
        )
        assert failing.passed is False

    def test_weakest_dimension(self):
        """Test weakest dimension identification."""
        score = FATEScore(
            fidelity=0.95,
            accountability=0.98,
            transparency=0.70,  # Weakest
            ethics=0.96,
        )
        assert score.weakest_dimension == FATEDimension.TRANSPARENCY

    def test_serialization(self):
        """Test to_dict serialization."""
        score = FATEScore(fidelity=0.9, accountability=0.9, transparency=0.9, ethics=0.9)
        d = score.to_dict()

        assert "fidelity" in d
        assert "overall" in d
        assert "passed" in d
        assert "weakest_dimension" in d


class TestFATEGate:
    """Tests for FATEGate validation."""

    def test_validate_basic(self):
        """Test basic validation."""
        gate = FATEGate()
        context = HookContext(
            operation_name="test_operation",
            operation_type="test",
            input_data={"key": "value"},
        )

        score = gate.validate(context, declared_intent="test operation")

        assert isinstance(score, FATEScore)
        assert 0.0 <= score.overall <= 1.0

    def test_validate_with_snr(self):
        """Test validation with SNR score."""
        gate = FATEGate()
        context = HookContext(operation_name="high_quality_op")

        # High SNR should boost ethics score
        score = gate.validate(context, snr_score=0.98)
        assert score.ethics >= 0.95

    def test_validate_pii_detected(self):
        """Test validation with PII detection."""
        gate = FATEGate()
        context = HookContext(
            operation_name="pii_op",
            metadata={"pii_detected": True},
        )

        score = gate.validate(context)
        # PII should reduce ethics score
        assert score.ethics < 0.95

    def test_statistics_tracking(self):
        """Test gate statistics."""
        gate = FATEGate()
        context = HookContext(operation_name="test")

        # Run multiple validations
        for _ in range(5):
            gate.validate(context, snr_score=0.98)

        stats = gate.get_stats()
        assert stats["total_validations"] == 5
        assert stats["passed"] + stats["blocked"] == 5


class TestHookRegistry:
    """Tests for HookRegistry."""

    def test_register_and_get_hooks(self):
        """Test hook registration and retrieval."""
        registry = HookRegistry()

        def my_hook(data):
            return data

        registry.register(
            name="test_hook",
            phase=HookPhase.PRE_EXECUTE,
            function=my_hook,
            priority=HookPriority.NORMAL,
        )

        hooks = registry.get_hooks(HookPhase.PRE_EXECUTE)
        assert len(hooks) == 1
        assert hooks[0].name == "test_hook"

    def test_priority_ordering(self):
        """Test hooks are sorted by priority."""
        registry = HookRegistry()

        registry.register("low", HookPhase.PRE_EXECUTE, lambda d: d, HookPriority.LOW)
        registry.register("critical", HookPhase.PRE_EXECUTE, lambda d: d, HookPriority.CRITICAL)
        registry.register("normal", HookPhase.PRE_EXECUTE, lambda d: d, HookPriority.NORMAL)

        hooks = registry.get_hooks(HookPhase.PRE_EXECUTE)
        assert hooks[0].name == "critical"
        assert hooks[1].name == "normal"
        assert hooks[2].name == "low"

    def test_enable_disable(self):
        """Test hook enable/disable."""
        registry = HookRegistry()
        registry.register("test", HookPhase.PRE_EXECUTE, lambda d: d)

        registry.disable("test")
        assert len(registry.get_hooks(HookPhase.PRE_EXECUTE)) == 0

        registry.enable("test")
        assert len(registry.get_hooks(HookPhase.PRE_EXECUTE)) == 1


class TestHookExecutor:
    """Tests for HookExecutor."""

    @pytest.mark.asyncio
    async def test_execute_simple_operation(self):
        """Test execution of simple operation."""
        executor = HookExecutor()

        def simple_op(x: int, y: int):
            return x + y

        result = await executor.execute(
            operation=simple_op,
            input_data={"x": 2, "y": 3},
            operation_name="addition",
            declared_intent="addition",  # Aligns with operation_name for high fidelity
            snr_score=0.98,  # Ensure FATE passes
            metadata={"rationale": "adding two numbers"},  # Boost transparency
        )

        assert result.success is True
        assert result.context.output_data["result"] == 5

    @pytest.mark.asyncio
    async def test_fate_gate_blocks(self):
        """Test FATE gate blocking low-quality operations."""
        executor = HookExecutor(fate_gate=FATEGate(ihsan_threshold=0.99))

        def blocked_op():
            return "should not execute"

        result = await executor.execute(
            operation=blocked_op,
            input_data={},
            operation_name="blocked",
            snr_score=0.5,  # Low SNR should block
        )

        assert result.success is False
        assert "fate_gate" in result.blocked_by

    @pytest.mark.asyncio
    async def test_hook_chain_execution(self):
        """Test full hook chain execution."""
        registry = HookRegistry()
        executor = HookExecutor(registry=registry)

        execution_order = []

        registry.register(
            "pre_hook",
            HookPhase.PRE_EXECUTE,
            lambda d: (execution_order.append("pre"), d)[1],
        )
        registry.register(
            "post_hook",
            HookPhase.POST_EXECUTE,
            lambda d: (execution_order.append("post"), d)[1],
        )

        def tracked_op():
            execution_order.append("op")
            return "done"

        result = await executor.execute(
            operation=tracked_op,
            input_data={},
            operation_name="tracked_op",
            operation_type="validated",
            declared_intent="tracked_op",  # Align for high fidelity
            snr_score=0.98,
            metadata={"rationale": "tracking operation execution order"},  # Boost transparency
        )

        assert result.success is True
        assert execution_order == ["pre", "op", "post"]


# ============================================================================
# SESSION DAG TESTS
# ============================================================================

class TestMerkleNode:
    """Tests for MerkleNode."""

    def test_compute_hash(self):
        """Test hash computation is deterministic."""
        ts = datetime.now(timezone.utc)

        hash1 = MerkleNode.compute_hash(
            parents=["parent1", "parent2"],
            state=SessionState.ACTIVE,
            data={"key": "value"},
            timestamp=ts,
        )

        hash2 = MerkleNode.compute_hash(
            parents=["parent2", "parent1"],  # Different order
            state=SessionState.ACTIVE,
            data={"key": "value"},
            timestamp=ts,
        )

        # Same content = same hash (canonical ordering)
        assert hash1 == hash2

    def test_hash_changes_with_content(self):
        """Test hash changes when content changes."""
        ts = datetime.now(timezone.utc)

        hash1 = MerkleNode.compute_hash([], SessionState.ACTIVE, {"a": 1}, ts)
        hash2 = MerkleNode.compute_hash([], SessionState.ACTIVE, {"a": 2}, ts)

        assert hash1 != hash2


class TestMerkleDAG:
    """Tests for MerkleDAG."""

    def test_genesis_creation(self):
        """Test genesis node is created on init."""
        dag = MerkleDAG()

        assert dag._genesis is not None
        assert len(dag._nodes) == 1
        assert dag.get_current_state() == SessionState.INIT

    def test_add_state(self):
        """Test adding states to DAG."""
        dag = MerkleDAG()

        node = dag.add_state(
            state=SessionState.ACTIVE,
            data={"activated": True},
        )

        assert node.state == SessionState.ACTIVE
        assert len(node.parents) == 1  # Links to genesis
        assert dag.get_current_state() == SessionState.ACTIVE

    def test_verify_lineage(self):
        """Test lineage verification."""
        dag = MerkleDAG()

        # Create a chain
        dag.add_state(SessionState.ACTIVE, {})
        dag.add_state(SessionState.COMPUTING, {})
        node = dag.add_state(SessionState.VALIDATED, {}, fate_score=0.98)

        # Verify lineage to genesis
        assert dag.verify_lineage(node.hash) is True

        # Invalid hash should fail
        assert dag.verify_lineage("nonexistent") is False

    def test_branch_and_merge(self):
        """Test branching and merging."""
        dag = MerkleDAG()
        dag.add_state(SessionState.ACTIVE, {})

        # Create branches
        branch1 = dag.branch("experiment_1", {"variant": "A"})
        dag.add_state(SessionState.COMPUTING, {})  # Continue on branch1

        # Go back and create branch2
        branch2 = dag.branch("experiment_2", {"variant": "B"})

        # Merge
        merged = dag.merge(
            [branch1.hash, branch2.hash],
            {"merged": True, "winner": "A"},
            SessionState.VALIDATED,
        )

        assert len(merged.parents) == 2
        assert merged.transition_type == TransitionType.MERGE

    def test_rollback(self):
        """Test state rollback."""
        dag = MerkleDAG()

        active_node = dag.add_state(SessionState.ACTIVE, {"version": 1})
        dag.add_state(SessionState.COMPUTING, {"version": 2})
        dag.add_state(SessionState.FAILED, {"error": "oops"})

        # Rollback to active state
        rollback_node = dag.rollback(active_node.hash, "recovering from failure")

        assert rollback_node.state == SessionState.ACTIVE
        assert rollback_node.transition_type == TransitionType.ROLLBACK

    def test_export_import(self):
        """Test DAG serialization round-trip."""
        dag = MerkleDAG(session_id="test_session")
        dag.add_state(SessionState.ACTIVE, {"data": 1})
        dag.add_state(SessionState.COMPUTING, {"data": 2})

        exported = dag.export()
        imported = MerkleDAG.import_dag(exported)

        assert imported.session_id == dag.session_id
        assert len(imported._nodes) == len(dag._nodes)
        assert imported.get_current_state() == dag.get_current_state()


class TestSessionStateMachine:
    """Tests for SessionStateMachine."""

    def test_lifecycle(self):
        """Test full session lifecycle."""
        session = create_session()

        assert session.current_state == SessionState.INIT

        session.activate()
        assert session.current_state == SessionState.ACTIVE

        session.compute({"task": "process"})
        assert session.current_state == SessionState.COMPUTING

        session.validate({"result": "ok"}, fate_score=0.97)
        assert session.current_state == SessionState.VALIDATED

        session.commit()
        assert session.current_state == SessionState.COMMITTED

    def test_suspend_resume(self):
        """Test suspend/resume flow."""
        session = create_session()
        session.activate()

        session.suspend("user requested")
        assert session.current_state == SessionState.SUSPENDED

        session.resume()
        assert session.current_state == SessionState.RESUMED

    def test_fail_recover(self):
        """Test failure/recovery flow."""
        session = create_session()
        session.activate()
        session.compute({"task": "risky"})

        session.fail("computation error")
        assert session.current_state == SessionState.FAILED

        session.recover({"fix": "applied"})
        assert session.current_state == SessionState.RECOVERED

    def test_invalid_transition(self):
        """Test invalid transition raises error."""
        session = create_session()

        # Cannot go directly from INIT to COMPUTING
        with pytest.raises(InvalidTransitionError):
            session.compute({})

    def test_integrity_verification(self):
        """Test DAG integrity verification."""
        session = create_session()
        session.activate()
        session.compute({})

        assert session.verify_integrity() is True


# ============================================================================
# COGNITIVE BUDGET TESTS
# ============================================================================

class TestDNAConstants:
    """Tests for 7-3-6-9 DNA constants."""

    def test_dna_values(self):
        """Test DNA constant values."""
        assert DNA_DEPTH == 7
        assert DNA_TRACKS == 3
        assert DNA_BALANCE == 6
        assert DNA_COMPLETE == 9

    def test_dna_sum(self):
        """Test DNA sum."""
        assert DNA_DEPTH + DNA_TRACKS + DNA_BALANCE + DNA_COMPLETE == 25


class TestComplexitySignal:
    """Tests for ComplexitySignal."""

    def test_complexity_score_range(self):
        """Test complexity score is in [0, 1]."""
        signal = ComplexitySignal(
            input_length=5000,
            input_entropy=0.8,
            reasoning_depth=5,
            convergence_required=2,
            knowledge_gap=0.5,
        )

        score = signal.compute_complexity_score()
        assert 0.0 <= score <= 1.0

    def test_low_complexity(self):
        """Test low complexity scenario."""
        signal = ComplexitySignal(
            input_length=100,
            input_entropy=0.2,
            reasoning_depth=1,
            convergence_required=1,
        )

        score = signal.compute_complexity_score()
        assert score < 0.4  # Low complexity

    def test_high_complexity(self):
        """Test high complexity scenario."""
        signal = ComplexitySignal(
            input_length=50000,
            input_entropy=0.9,
            reasoning_depth=7,
            convergence_required=3,
            knowledge_gap=0.8,
            previous_attempts=5,
        )

        score = signal.compute_complexity_score()
        assert score > 0.6  # High complexity


class TestCognitiveBudgetAllocator:
    """Tests for CognitiveBudgetAllocator."""

    def test_allocate_nano_task(self):
        """Test allocation for NANO tier task."""
        allocator = CognitiveBudgetAllocator()
        allocation = allocator.allocate(TaskType.ECHO, "Hello")

        assert allocation.tier == BudgetTier.NANO
        assert allocation.time_budget_s <= 5.0
        assert allocation.token_budget <= 500

    def test_allocate_mega_task(self):
        """Test allocation for MEGA tier task."""
        allocator = CognitiveBudgetAllocator()
        allocation = allocator.allocate(TaskType.RESEARCH, "complex research topic")

        assert allocation.tier == BudgetTier.MEGA
        assert allocation.extended_thinking is True
        assert allocation.time_budget_s > 300.0

    def test_complexity_promotes_tier(self):
        """Test that high complexity promotes tier."""
        allocator = CognitiveBudgetAllocator()

        # Simple task
        simple = allocator.allocate(
            TaskType.SUMMARIZE,
            "short text",
        )

        # Same task with complexity context
        complex = allocator.allocate(
            TaskType.SUMMARIZE,
            "very long complex text " * 1000,
            context={"knowledge_gap": 0.9, "retry_count": 3},
        )

        # Complex should have equal or higher tier
        tier_order = list(BudgetTier)
        assert tier_order.index(complex.tier) >= tier_order.index(simple.tier)

    def test_statistics_tracking(self):
        """Test allocator statistics."""
        allocator = CognitiveBudgetAllocator()

        allocator.allocate(TaskType.ECHO, "")
        allocator.allocate(TaskType.ANALYZE, "text")
        allocator.allocate(TaskType.RESEARCH, "topic")

        stats = allocator.get_stats()
        assert stats["total_allocations"] == 3
        assert "tier_distribution" in stats


class TestBudgetTracker:
    """Tests for BudgetTracker context manager."""

    def test_budget_tracking(self):
        """Test budget usage tracking."""
        allocation = BudgetAllocation(
            tier=BudgetTier.MESO,
            time_budget_s=30.0,
            token_budget=2000,
            depth=4,
            parallel_tracks=2,
            max_iterations=6,
            confidence_threshold=0.95,
        )

        with BudgetTracker(allocation) as tracker:
            assert tracker.is_budget_available() is True

            tracker.consume_tokens(500)
            assert tracker.tokens_remaining() == 1500

            tracker.complete_iteration(0.8)
            assert tracker.iterations_remaining() == 5
            assert tracker.should_continue() is True

            tracker.complete_iteration(0.96)  # Exceeds threshold
            assert tracker.should_continue() is False
            assert tracker.early_exit is True

    def test_budget_exhaustion(self):
        """Test budget exhaustion detection."""
        allocation = BudgetAllocation(
            tier=BudgetTier.NANO,
            time_budget_s=1.0,
            token_budget=100,
            depth=1,
            parallel_tracks=1,
            max_iterations=1,
            confidence_threshold=0.9,
        )

        with BudgetTracker(allocation) as tracker:
            tracker.consume_tokens(150)  # Over budget
            assert tracker.exhausted is True
            assert tracker.is_budget_available() is False


# ============================================================================
# COMPUTE MARKET TESTS
# ============================================================================

class TestResourceUnit:
    """Tests for ResourceUnit."""

    def test_serialization(self):
        """Test resource unit serialization."""
        resource = ResourceUnit(
            resource_type=ResourceType.GPU,
            quantity=4.0,
            unit_name="cards",
        )

        d = resource.to_dict()
        assert d["type"] == "gpu"
        assert d["quantity"] == 4.0


class TestComputeLicense:
    """Tests for ComputeLicense."""

    def test_tax_due_calculation(self):
        """Test tax calculation over time."""
        license = ComputeLicense(
            self_assessed_value=100.0,
            tax_rate=0.05,
        )

        # Fast-forward 1 hour (1 period)
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        tax = license.compute_tax_due(future)

        expected = 100.0 * 0.05 * 1.0  # value * rate * periods
        assert abs(tax - expected) < 0.1

    def test_purchasable_status(self):
        """Test license purchasability."""
        license = ComputeLicense(status=LicenseStatus.ACTIVE)
        assert license.is_purchasable() is True

        license.status = LicenseStatus.REVOKED
        assert license.is_purchasable() is False


class TestComputeMarket:
    """Tests for ComputeMarket."""

    def test_issue_license(self):
        """Test license issuance."""
        market = create_market()

        license = market.issue_license(
            holder_id="node_001",
            resource=ResourceUnit(ResourceType.INFERENCE, 10, "slots"),
            initial_value=500.0,
        )

        assert license.holder_id == "node_001"
        assert license.self_assessed_value == 500.0
        assert license.status == LicenseStatus.ACTIVE

    def test_purchase_license(self):
        """Test Harberger purchase mechanism."""
        market = create_market()

        license = market.issue_license(
            "seller",
            ResourceUnit(ResourceType.INFERENCE, 1, "slots"),
            100.0,
        )

        transaction = market.purchase_license(
            license.license_id,
            "buyer",
            buyer_ihsan=0.96,
        )

        assert transaction.success is True
        assert transaction.price == 100.0
        assert license.holder_id == "buyer"

    def test_cannot_purchase_own_license(self):
        """Test cannot buy own license."""
        market = create_market()

        license = market.issue_license(
            "holder",
            ResourceUnit(ResourceType.INFERENCE, 1, "slots"),
            100.0,
        )

        transaction = market.purchase_license(
            license.license_id,
            "holder",  # Same as owner
            buyer_ihsan=0.98,
        )

        assert transaction.success is False

    def test_ihsan_requirement(self):
        """Test buyer must meet Ihsan threshold."""
        market = create_market()

        license = market.issue_license(
            "seller",
            ResourceUnit(ResourceType.INFERENCE, 1, "slots"),
            100.0,
        )

        transaction = market.purchase_license(
            license.license_id,
            "buyer",
            buyer_ihsan=0.5,  # Below threshold
        )

        assert transaction.success is False

    def test_reassess_value(self):
        """Test self-assessment update."""
        market = create_market()

        license = market.issue_license("holder", ResourceUnit(ResourceType.CPU, 1, "cores"), 50.0)

        success = market.reassess_value(license.license_id, 75.0, "holder")
        assert success is True
        assert license.self_assessed_value == 75.0

    def test_gini_calculation(self):
        """Test Gini coefficient calculation."""
        market = create_market()

        # Equal distribution
        market.issue_license("node1", ResourceUnit(ResourceType.CPU, 1, "cores"), 100.0)
        market.issue_license("node2", ResourceUnit(ResourceType.CPU, 1, "cores"), 100.0)
        market.issue_license("node3", ResourceUnit(ResourceType.CPU, 1, "cores"), 100.0)

        gini = market.compute_gini()
        assert gini < 0.1  # Very equal

    def test_gini_inequality(self):
        """Test Gini detects inequality."""
        market = create_market()

        # Unequal distribution
        market.issue_license("whale", ResourceUnit(ResourceType.CPU, 1, "cores"), 1000.0)
        market.issue_license("small1", ResourceUnit(ResourceType.CPU, 1, "cores"), 10.0)
        market.issue_license("small2", ResourceUnit(ResourceType.CPU, 1, "cores"), 10.0)

        gini = market.compute_gini()
        assert gini > 0.5  # High inequality

    def test_gini_enforcement(self):
        """Test Gini enforcement triggers redistribution."""
        market = create_market(gini_threshold=0.3)

        # Create inequality
        market.issue_license("whale", ResourceUnit(ResourceType.CPU, 1, "cores"), 1000.0)
        market.issue_license("small", ResourceUnit(ResourceType.CPU, 1, "cores"), 10.0)

        report = market.enforce_gini()

        assert report["action_taken"] is True
        assert len(report["forced_sales"]) > 0
        assert report["gini_after"] < report["gini_before"]

    def test_tax_collection(self):
        """Test periodic tax collection."""
        market = create_market()

        market.issue_license("holder", ResourceUnit(ResourceType.CPU, 1, "cores"), 100.0)

        # Simulate time passing
        for license in market._licenses.values():
            license.last_tax_payment = datetime.now(timezone.utc) - timedelta(hours=1)

        collected = market.collect_taxes()
        assert collected > 0
        assert market._treasury > 0

    def test_treasury_distribution(self):
        """Test treasury fund distribution."""
        market = create_market()
        market._treasury = 1000.0

        distribution = market.distribute_treasury(500.0, ["node1", "node2"])

        assert len(distribution) == 2
        assert distribution["node1"] == 250.0
        assert market._treasury == 500.0

    def test_market_health(self):
        """Test market health assessment."""
        market = create_market()
        market.issue_license("node1", ResourceUnit(ResourceType.CPU, 1, "cores"), 100.0)

        health = market.get_market_health()

        assert "gini_status" in health
        assert health["adl_compliance"] in [True, False]


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestEliteIntegration:
    """Integration tests across elite modules."""

    @pytest.mark.asyncio
    async def test_fate_with_session(self):
        """Test FATE gate integrated with session tracking."""
        session = create_session()
        gate = FATEGate()

        # Activate session
        session.activate()

        # Validate operation with declared_intent for high fidelity
        context = HookContext(
            operation_name="compute_task",
            operation_type="validated",
            input_data={"task": "compute"},
            metadata={"session_id": session.session_id, "rationale": "session compute task"},
        )
        score = gate.validate(context, declared_intent="compute_task", snr_score=0.97)

        # Transition based on FATE result
        if score.passed:
            session.compute({"fate_score": score.overall})
            session.validate({"fate": score.to_dict()}, score.overall)
        else:
            session.fail("FATE gate blocked")

        assert session.current_state == SessionState.VALIDATED

    def test_budget_with_market(self):
        """Test cognitive budget influences market licensing."""
        market = create_market()
        allocator = CognitiveBudgetAllocator()

        # Allocate budget for a complex task
        budget = allocator.allocate(TaskType.RESEARCH, "complex research")

        # Higher tier = need more resources = higher license value
        value_multiplier = list(BudgetTier).index(budget.tier) + 1

        license = create_inference_license(
            market,
            "researcher",
            slots=budget.parallel_tracks,
            value=50.0 * value_multiplier,
        )

        assert license.self_assessed_value > 50.0  # Scaled up

    def test_session_dag_integrity_with_fate(self):
        """Test session DAG maintains integrity through FATE-validated transitions."""
        session = create_session()
        gate = FATEGate()

        transitions = [
            (SessionState.ACTIVE, "activate"),
            (SessionState.COMPUTING, "process"),
            (SessionState.VALIDATED, "validate"),
            (SessionState.COMMITTED, "commit"),
        ]

        for target_state, reason in transitions:
            context = HookContext(
                operation_name=reason,
                operation_type="validated",
                input_data={"transition": reason},
                metadata={"rationale": f"transition to {target_state.value}"},
            )
            score = gate.validate(context, declared_intent=reason, snr_score=0.97)

            if target_state == SessionState.COMMITTED:
                session.commit()
            else:
                # Use transition() directly to pass fate_score for all states
                session.transition(
                    target_state,
                    data={"fate": score.overall},
                    reason=reason,
                    fate_score=score.overall,
                )

            assert session.current_node.ihsan_achieved or target_state == SessionState.COMMITTED

        # Verify full lineage
        assert session.verify_integrity() is True
        history = session.get_history()
        assert len(history) >= 5  # Genesis + 4 transitions


class TestSNRCompliance:
    """Tests verifying SNR >= 0.95 target."""

    def test_fate_achieves_ihsan(self):
        """Test FATE can achieve Ihsan threshold."""
        gate = FATEGate()
        context = HookContext(
            operation_name="quality_operation",
            operation_type="validated",
            metadata={"rationale": "well documented"},
        )

        score = gate.validate(context, declared_intent="quality operation", snr_score=0.98)

        assert score.overall >= UNIFIED_IHSAN_THRESHOLD

    def test_session_tracks_ihsan(self):
        """Test session properly tracks Ihsan achievement."""
        session = create_session()
        session.activate()
        session.compute({})

        node = session.validate({"quality": "high"}, fate_score=0.97)
        assert node.ihsan_achieved is True

    def test_market_enforces_ihsan_threshold(self):
        """Test market enforces Ihsan for buyers."""
        market = create_market()
        license = market.issue_license(
            "seller",
            ResourceUnit(ResourceType.INFERENCE, 1, "slots"),
            100.0,
        )

        # Below threshold should fail
        tx_fail = market.purchase_license(license.license_id, "buyer", 0.90)
        assert tx_fail.success is False

        # At threshold should succeed
        tx_pass = market.purchase_license(license.license_id, "buyer", 0.95)
        assert tx_pass.success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
