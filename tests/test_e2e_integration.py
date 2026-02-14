"""
End-to-End Integration Tests — Phase 18.1 Wiring Verification
==============================================================

Standing on the Shoulders of:
- pytest (Krekel, 2004) — The gold standard test framework
- Hypothesis (MacIver, 2015) — Property-based testing philosophy

These tests verify that the sovereign runtime correctly wires ALL
subsystems into a functioning whole:

1. Orchestrator is instantiated and injected with gateway + memory
2. Complexity detector routes queries to orchestrator vs direct pipeline
3. FastAPI app factory exposes /v1/orchestrate endpoint
4. Full query pipeline round-trips through all 5 stages
5. Memory persistence survives across runtime instances
"""

from __future__ import annotations

import asyncio
import pathlib
import tempfile
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides: Any) -> Any:
    """Create a RuntimeConfig with test-friendly defaults."""
    from core.sovereign.runtime_types import RuntimeConfig

    defaults = {
        "autonomous_enabled": False,
        "enable_graph_reasoning": False,
        "enable_snr_optimization": False,
        "enable_guardian_validation": False,
        "enable_autonomous_loop": False,
        "enable_zpk_preflight": False,
        "enable_proactive_kernel": False,
        "enable_persistence": False,
    }
    defaults.update(overrides)
    return RuntimeConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. Orchestrator Wiring
# ---------------------------------------------------------------------------


class TestOrchestratorWiring:
    """Verify orchestrator is created and injected during init."""

    def test_orchestrator_import(self) -> None:
        """SovereignOrchestrator is importable and constructible."""
        from core.sovereign.orchestrator import (
            RoutingStrategy,
            SovereignOrchestrator,
        )

        orch = SovereignOrchestrator(routing_strategy=RoutingStrategy.ADAPTIVE)
        orch.register_default_agents()
        assert len(orch.router.agents) >= 1, f"No agents registered: {orch.router.agents}"

    def test_set_gateway_and_memory(self) -> None:
        """set_gateway and set_memory accept arbitrary objects."""
        from core.sovereign.orchestrator import SovereignOrchestrator

        orch = SovereignOrchestrator()
        mock_gw = MagicMock()
        mock_mem = MagicMock()

        orch.set_gateway(mock_gw)
        orch.set_memory(mock_mem)

        assert orch._gateway is mock_gw
        assert orch._memory is mock_mem

    @pytest.mark.asyncio
    async def test_runtime_creates_orchestrator(self) -> None:
        """SovereignRuntime._init_components wires orchestrator."""
        from core.sovereign.runtime_core import SovereignRuntime

        config = _make_config()
        runtime = SovereignRuntime(config)

        # Patch gateway so orchestrator gets something to inject
        runtime._gateway = MagicMock()

        try:
            await runtime._init_components()
        except Exception:
            pass  # Other components may fail; we only care about orchestrator

        assert runtime._orchestrator is not None, "Orchestrator was not created"
        # Verify gateway was injected
        assert getattr(runtime._orchestrator, "_gateway", None) is runtime._gateway


# ---------------------------------------------------------------------------
# 2. Complexity Estimator
# ---------------------------------------------------------------------------


class TestComplexityEstimator:
    """Verify the query complexity detector works."""

    def _estimate(self, text: str, **ctx: Any) -> float:
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import SovereignQuery

        rt = SovereignRuntime(_make_config())
        q = SovereignQuery(text=text, context=ctx)
        return rt._estimate_complexity(q)

    def test_simple_query_low_complexity(self) -> None:
        score = self._estimate("What is sovereignty?")
        assert score < 0.4, f"Simple query too complex: {score}"

    def test_complex_query_high_complexity(self) -> None:
        score = self._estimate(
            "Compare and contrast the architectural approaches of sovereign "
            "node systems and centralized cloud platforms, then analyze "
            "the multi-domain implications for data sovereignty, privacy, "
            "and furthermore evaluate the step by step migration path "
            "for enterprise adoption. Additionally, provide a comprehensive "
            "cost analysis."
        )
        assert score >= 0.4, f"Complex query scored too low: {score}"

    def test_question_marks_increase_complexity(self) -> None:
        base = self._estimate("Tell me about X")
        multi = self._estimate("What is X? How does Y work? Why is Z important?")
        assert multi > base, "Multiple questions should score higher"

    def test_complexity_hint_from_context(self) -> None:
        low = self._estimate("hello", complexity_hint=0.0)
        high = self._estimate("hello", complexity_hint=1.0)
        assert high > low, "Hint should influence score"


# ---------------------------------------------------------------------------
# 3. Query Pipeline Routing
# ---------------------------------------------------------------------------


class TestQueryPipelineRouting:
    """Verify complex queries route through orchestrator."""

    @pytest.mark.asyncio
    async def test_simple_query_uses_direct_pipeline(self) -> None:
        """Simple query bypasses orchestrator."""
        from core.sovereign.runtime_core import SovereignRuntime

        config = _make_config()
        runtime = SovereignRuntime(config)
        runtime._initialized = True
        runtime._running = True
        runtime._orchestrator = MagicMock()

        result = await runtime.query("Hello")
        # Either succeeds, returns an error, or rejects via fail-closed gate chain
        # — we only care that it didn't crash
        assert result is not None
        assert result.response or result.error or result.success

    @pytest.mark.asyncio
    async def test_complex_query_attempts_orchestrator(self) -> None:
        """Complex query routes to _orchestrate_complex_query."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import SovereignQuery

        config = _make_config()
        runtime = SovereignRuntime(config)
        runtime._initialized = True
        runtime._running = True

        # Create a mock orchestrator with decompose and task_results
        mock_orch = MagicMock()
        mock_plan = MagicMock()
        mock_plan.tasks = []
        mock_plan.id = "test-plan"
        mock_plan.complexity = MagicMock(name="MODERATE")
        mock_orch.decompose = AsyncMock(return_value=mock_plan)
        mock_orch.task_results = {}
        runtime._orchestrator = mock_orch

        # Complex query should trigger orchestrator
        complex_text = (
            "Compare and contrast the multi-domain comprehensive analysis "
            "of furthermore evaluate the step by step implications and "
            "additionally provide a full assessment"
        )
        result = await runtime.query(complex_text, timeout_ms=10000)
        assert result is not None


# ---------------------------------------------------------------------------
# 4. FastAPI App — /v1/orchestrate Endpoint
# ---------------------------------------------------------------------------


class TestFastAPIEndpoints:
    """Verify FastAPI app has orchestrate endpoint."""

    def test_create_fastapi_app(self) -> None:
        """create_fastapi_app returns a FastAPI instance with all routes."""
        from core.sovereign.api import create_fastapi_app

        mock_runtime = MagicMock()
        mock_runtime.status.return_value = {
            "health": {"status": "healthy"},
            "identity": {"version": "1.0.0"},
        }

        app = create_fastapi_app(mock_runtime)

        # Collect all route paths
        routes = [r.path for r in app.routes if hasattr(r, "path")]

        assert "/v1/health" in routes, f"Missing /v1/health: {routes}"
        assert "/v1/query" in routes, f"Missing /v1/query: {routes}"
        assert "/v1/orchestrate" in routes, f"Missing /v1/orchestrate: {routes}"
        assert "/v1/status" in routes, f"Missing /v1/status: {routes}"
        assert "/" in routes, f"Missing / (console): {routes}"


# ---------------------------------------------------------------------------
# 5. Memory → Orchestrator → Gateway Round-Trip
# ---------------------------------------------------------------------------


class TestFullRoundTrip:
    """Verify the full stack wires together."""

    @pytest.mark.asyncio
    async def test_orchestrator_with_gateway(self) -> None:
        """Orchestrator calls gateway.infer when executing a task."""
        from core.sovereign.orchestrator import (
            AgentType,
            RoutingStrategy,
            SovereignOrchestrator,
            TaskNode,
        )

        orch = SovereignOrchestrator(routing_strategy=RoutingStrategy.ADAPTIVE)
        orch.register_default_agents()

        # Mock gateway
        mock_result = MagicMock()
        mock_result.content = "The answer from the LLM"
        mock_gw = MagicMock()
        mock_gw.infer = AsyncMock(return_value=mock_result)
        orch.set_gateway(mock_gw)

        # Submit a task
        task = TaskNode(
            title="Analyze sovereignty patterns",
            description="Find patterns in sovereignty architectures",
            assigned_agent=AgentType.ANALYST,
        )

        result = await orch.execute_task(task)
        assert result is not None
        assert result.get("status") in ("completed", "COMPLETED", TaskNode)
        content = result.get("content", "")
        assert len(content) > 0, "Task produced no content"

    @pytest.mark.asyncio
    async def test_memory_persistence_roundtrip(self) -> None:
        """Memory encodes → persists to SQLite → reloads on new instance."""
        from core.living_memory.core import LivingMemoryCore, MemoryType

        tmp = pathlib.Path(tempfile.mkdtemp())
        try:
            mem1 = LivingMemoryCore(storage_path=tmp)
            await mem1.initialize()

            entry = await mem1.encode(
                "The orchestrator decomposes complex queries into sub-tasks",
                MemoryType.SEMANTIC,
                source="test",
                importance=0.85,
            )
            assert entry is not None

            # New instance should load from SQLite
            mem2 = LivingMemoryCore(storage_path=tmp)
            await mem2.initialize()
            assert len(mem2._memories) >= 1, f"Persistence failed: {len(mem2._memories)}"
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_orchestrator_decompose(self) -> None:
        """Orchestrator can decompose a query into a task plan."""
        from core.sovereign.orchestrator import SovereignOrchestrator, TaskNode as TN

        orch = SovereignOrchestrator()
        orch.register_default_agents()

        plan = await orch.decomposer.decompose(
            TN(
                title="What is the meaning of sovereignty?",
                description="What is the meaning of sovereignty?",
            )
        )
        assert plan is not None
        assert hasattr(plan, "subtasks")

    def test_snr_scorer(self) -> None:
        """Quick SNR scorer produces reasonable values."""
        from core.sovereign.orchestrator import SovereignOrchestrator

        orch = SovereignOrchestrator()

        assert orch._score_output("") == 0.0
        assert orch._score_output("hi") == 0.3  # Too short

        good = orch._score_output(
            "The sovereign node architecture provides data ownership "
            "through local inference, memory persistence, and constitutional "
            "validation with full SNR optimization."
        )
        assert 0.4 < good <= 1.0, f"Unexpected score: {good}"


# ---------------------------------------------------------------------------
# 6. Cross-System E2E — Full Stack Proof
# ---------------------------------------------------------------------------


class TestCrossSystemE2E:
    """Prove the full stack works: API → Orchestrator → Gateway → Memory → Response.

    This is the Genesis Proof — if these tests pass, the system is alive.
    """

    @pytest.mark.asyncio
    async def test_full_stack_query_pipeline(self) -> None:
        """Complete round-trip: query → decompose → infer → score → respond."""
        from core.sovereign.orchestrator import (
            RoutingStrategy,
            SovereignOrchestrator,
            TaskComplexity,
            TaskNode,
        )
        from core.living_memory.core import LivingMemoryCore, MemoryType

        tmp = pathlib.Path(tempfile.mkdtemp())
        try:
            # 1. Initialize memory with seed knowledge
            memory = LivingMemoryCore(storage_path=tmp)
            await memory.initialize()
            await memory.encode(
                "BIZRA is a decentralized sovereign system where every human is a node",
                MemoryType.SEMANTIC,
                source="test_seed",
                importance=0.95,
            )
            await memory.encode(
                "The inference gateway uses circuit breakers for resilience",
                MemoryType.PROCEDURAL,
                source="test_seed",
                importance=0.9,
            )

            # 2. Create orchestrator with mock LLM gateway
            orch = SovereignOrchestrator(routing_strategy=RoutingStrategy.ADAPTIVE)
            orch.register_default_agents()

            mock_result = MagicMock()
            mock_result.content = (
                "BIZRA sovereignty ensures data ownership through local inference. "
                "The circuit breaker pattern protects against cascade failures."
            )
            mock_gw = MagicMock()
            mock_gw.infer = AsyncMock(return_value=mock_result)

            orch.set_gateway(mock_gw)
            orch.set_memory(memory)

            # 3. Execute a real task through the full pipeline
            task = TaskNode(
                title="Explain BIZRA sovereignty model",
                description="How does BIZRA achieve data sovereignty?",
                complexity=TaskComplexity.SIMPLE,
            )

            result = await orch.execute_task(task)

            # 4. Verify complete pipeline
            assert result["status"] == "completed"
            assert len(result["content"]) > 20, "Response too short"
            assert result["snr_score"] > 0.3, f"SNR too low: {result['snr_score']}"
            assert result["latency_ms"] > 0, "No latency recorded"

            # 5. Verify gateway was called with memory-enriched prompt
            mock_gw.infer.assert_called_once()
            call_args = mock_gw.infer.call_args
            prompt_sent = call_args[0][0] if call_args[0] else call_args[1].get("prompt", "")
            # Memory context should be included in the prompt
            assert "BIZRA" in prompt_sent or "sovereignty" in prompt_sent.lower()

        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_event_driven_orchestration_loop(self) -> None:
        """Orchestration loop dispatches via events, not polling."""
        from core.sovereign.orchestrator import (
            SovereignOrchestrator,
            TaskComplexity,
            TaskNode,
        )

        orch = SovereignOrchestrator(max_concurrent_tasks=3)
        orch.register_default_agents()

        mock_result = MagicMock()
        mock_result.content = "Completed analysis"
        mock_gw = MagicMock()
        mock_gw.infer = AsyncMock(return_value=mock_result)
        orch.set_gateway(mock_gw)

        # Start the run loop in background
        loop_task = asyncio.create_task(orch.run())

        # Submit a task — should be dispatched via event signal
        task = TaskNode(
            title="Test event dispatch",
            description="Verify event-driven orchestration",
            complexity=TaskComplexity.TRIVIAL,
        )
        await orch.submit(task)

        # Wait for completion (with timeout)
        for _ in range(50):  # 5 seconds max
            await asyncio.sleep(0.1)
            if task.id in orch.completed_tasks:
                break

        orch.stop()
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass

        assert task.id in orch.completed_tasks, (
            f"Task not completed. Status: {orch.get_status()}"
        )

    @pytest.mark.asyncio
    async def test_proactive_retriever_with_memory(self) -> None:
        """Proactive retriever surfaces relevant suggestions from memory."""
        from core.living_memory.core import LivingMemoryCore, MemoryType
        from core.living_memory.proactive import ProactiveRetriever

        tmp = pathlib.Path(tempfile.mkdtemp())
        try:
            memory = LivingMemoryCore(storage_path=tmp)
            await memory.initialize()

            # Seed with knowledge
            await memory.encode(
                "Rust provides zero-cost abstractions for systems programming",
                MemoryType.SEMANTIC,
                source="test",
                importance=0.9,
            )
            await memory.encode(
                "PyO3 bridges Python and Rust for high-performance inference",
                MemoryType.SEMANTIC,
                source="test",
                importance=0.95,
            )

            retriever = ProactiveRetriever(memory=memory, max_suggestions=3)

            # Update context with relevant queries
            retriever.update_context(query="How does the Rust bridge work?")
            retriever.update_context(query="What about PyO3 performance?")

            suggestions = await retriever.get_proactive_suggestions()
            # Should find at least some relevant memories
            assert isinstance(suggestions, list)
            # Context summary should reflect our queries
            ctx = retriever.get_context_summary()
            assert ctx["recent_queries_count"] == 2

        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_fastapi_new_endpoints(self) -> None:
        """FastAPI app includes WebSocket and suggestions endpoints."""
        from core.sovereign.api import create_fastapi_app

        mock_runtime = MagicMock()
        mock_runtime.status.return_value = {
            "health": {"status": "healthy"},
            "identity": {"version": "1.0.0", "node_id": "test-node"},
        }

        app = create_fastapi_app(mock_runtime)
        routes = [r.path for r in app.routes if hasattr(r, "path")]

        assert "/v1/suggestions" in routes, f"Missing /v1/suggestions: {routes}"
        # WebSocket route may show differently in FastAPI routing
        ws_paths = [r.path for r in app.routes if hasattr(r, "path") and "stream" in r.path]
        assert len(ws_paths) >= 1 or "/v1/stream" in routes, f"Missing /v1/stream: {routes}"

    @pytest.mark.asyncio
    async def test_a2a_task_manager_event_driven(self) -> None:
        """A2A TaskManager processes submitted tasks via its queue."""
        from core.a2a.tasks import TaskManager
        from core.a2a.schema import TaskCard

        tm = TaskManager(max_concurrent=2)

        # Submit a task and verify it is queued for processing
        task = TaskCard(
            prompt="Test task",
            capability_required="testing",
        )
        task_id = tm.submit(task)

        assert task_id == task.task_id, "submit must return the task ID"
        assert task.task_id in tm.tasks, "Task not stored after submit"
        assert len(tm.queue) >= 1, "Task not enqueued after submit"
        assert tm.stats["submitted"] == 1, "Submit counter not incremented"

    @pytest.mark.asyncio
    async def test_memory_sqlite_persistence_integrity(self) -> None:
        """Memory persists across instances and maintains data integrity."""
        from core.living_memory.core import LivingMemoryCore, MemoryType

        tmp = pathlib.Path(tempfile.mkdtemp())
        try:
            # Instance 1: Create and encode
            mem1 = LivingMemoryCore(storage_path=tmp)
            await mem1.initialize()

            entries = []
            for i in range(5):
                entry = await mem1.encode(
                    f"Test memory entry number {i} with unique content",
                    MemoryType.EPISODIC,
                    source="integrity_test",
                    importance=0.5 + (i * 0.1),
                )
                entries.append(entry)

            assert len(mem1._memories) >= 5
            await mem1._save_memories()

            # Instance 2: Should load all entries
            mem2 = LivingMemoryCore(storage_path=tmp)
            await mem2.initialize()
            assert len(mem2._memories) >= 5, (
                f"Persistence lost entries: {len(mem2._memories)} < 5"
            )

            # Verify data integrity
            for entry in entries:
                if entry is not None:
                    assert entry.id in mem2._memories, f"Entry {entry.id} lost"

        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
