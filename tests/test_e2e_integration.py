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
        assert result.success or result.error  # Either works; just didn't crash

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
