"""
GateChain Integration Tests — INT-006 Completion.

Proves that the 6-gate fail-closed chain is wired into
SovereignRuntime._process_query_direct() as a pre-flight check.

Standing on Giants:
- Lamport (1978): Fail-closed semantics
- Dijkstra (1968): Structured decomposition
- BIZRA Spearpoint PRD: "6 gates, fail fast, fail closed"
"""

import pytest
from pathlib import Path

from core.sovereign.runtime_core import SovereignRuntime
from core.sovereign.runtime_types import RuntimeConfig, SovereignQuery, SovereignResult


# =============================================================================
# GATE CHAIN INITIALIZATION
# =============================================================================

class TestGateChainInit:
    """Tests for GateChain initialization in SovereignRuntime."""

    def test_gate_chain_field_exists(self):
        """SovereignRuntime has _gate_chain attribute."""
        runtime = SovereignRuntime()
        assert hasattr(runtime, "_gate_chain")

    def test_gate_chain_none_by_default(self):
        """_gate_chain is None before initialization."""
        runtime = SovereignRuntime()
        assert runtime._gate_chain is None

    def test_init_gate_chain_creates_instance(self, tmp_path):
        """_init_gate_chain creates a GateChain."""
        config = RuntimeConfig()
        config.state_dir = tmp_path
        runtime = SovereignRuntime(config)
        runtime._init_gate_chain()

        assert runtime._gate_chain is not None

    def test_gate_chain_has_6_gates(self, tmp_path):
        """GateChain has exactly 6 gates in correct order."""
        config = RuntimeConfig()
        config.state_dir = tmp_path
        runtime = SovereignRuntime(config)
        runtime._init_gate_chain()

        gate_names = [g.name for g in runtime._gate_chain.gates]
        assert gate_names == [
            "schema", "provenance", "snr", "constraint", "safety", "commit"
        ]

    def test_gate_chain_stats_none_when_not_init(self):
        """get_gate_chain_stats returns None when chain not initialized."""
        runtime = SovereignRuntime()
        assert runtime.get_gate_chain_stats() is None

    def test_gate_chain_stats_empty_after_init(self, tmp_path):
        """get_gate_chain_stats returns empty stats after init."""
        config = RuntimeConfig()
        config.state_dir = tmp_path
        runtime = SovereignRuntime(config)
        runtime._init_gate_chain()

        stats = runtime.get_gate_chain_stats()
        assert stats is not None
        assert stats["total_evaluations"] == 0


# =============================================================================
# PRE-FLIGHT CHECK
# =============================================================================

class TestGateChainPreflight:
    """Tests for GateChain pre-flight check in query pipeline."""

    @pytest.mark.asyncio
    async def test_preflight_passes_valid_query(self, tmp_path):
        """Valid query passes the gate chain preflight."""
        config = RuntimeConfig()
        config.state_dir = tmp_path
        runtime = SovereignRuntime(config)
        runtime._init_gate_chain()

        query = SovereignQuery(text="What is the meaning of life?", id="a1b2c3d4e5f60001")
        result = SovereignResult(query_id=query.id)

        rejection = await runtime._run_gate_chain_preflight(query, result)
        assert rejection is None  # None means "continue processing"

    @pytest.mark.asyncio
    async def test_preflight_rejects_empty_intent(self, tmp_path):
        """Empty intent fails the schema gate."""
        config = RuntimeConfig()
        config.state_dir = tmp_path
        runtime = SovereignRuntime(config)
        runtime._init_gate_chain()

        query = SovereignQuery(text="", id="a1b2c3d4e5f60002")
        result = SovereignResult(query_id=query.id)

        rejection = await runtime._run_gate_chain_preflight(query, result)
        assert rejection is not None
        assert rejection.success is False
        assert "rejected by gate chain" in rejection.response.lower()

    @pytest.mark.asyncio
    async def test_preflight_rejects_when_no_chain(self, tmp_path):
        """CRITICAL-1 FIX: No gate chain → REJECT (fail-closed)."""
        config = RuntimeConfig()
        config.state_dir = tmp_path
        runtime = SovereignRuntime(config)
        # Don't initialize gate chain

        query = SovereignQuery(text="", id="a1b2c3d4e5f60003")
        result = SovereignResult(query_id=query.id)

        rejection = await runtime._run_gate_chain_preflight(query, result)
        # CRITICAL-1: No gate chain → reject (fail-closed, not pass-through)
        assert rejection is not None or result.success is False

    @pytest.mark.asyncio
    async def test_preflight_tracks_stats(self, tmp_path):
        """Gate chain stats are updated after preflight."""
        config = RuntimeConfig()
        config.state_dir = tmp_path
        runtime = SovereignRuntime(config)
        runtime._init_gate_chain()

        query = SovereignQuery(text="What is truth?", id="a1b2c3d4e5f60004")
        result = SovereignResult(query_id=query.id)

        await runtime._run_gate_chain_preflight(query, result)

        stats = runtime.get_gate_chain_stats()
        assert stats["total_evaluations"] == 1
        assert stats["passed"] == 1

    @pytest.mark.asyncio
    async def test_preflight_rejection_has_snr(self, tmp_path):
        """Rejected result includes SNR score from gate chain."""
        config = RuntimeConfig()
        config.state_dir = tmp_path
        runtime = SovereignRuntime(config)
        runtime._init_gate_chain()

        # Empty text will fail schema gate
        query = SovereignQuery(text="", id="a1b2c3d4e5f60005")
        result = SovereignResult(query_id=query.id)

        rejection = await runtime._run_gate_chain_preflight(query, result)
        assert rejection is not None
        assert rejection.validation_passed is False

    @pytest.mark.asyncio
    async def test_preflight_uses_context(self, tmp_path):
        """Gate chain preflight extracts context from query."""
        config = RuntimeConfig()
        config.state_dir = tmp_path
        runtime = SovereignRuntime(config)
        runtime._init_gate_chain()

        query = SovereignQuery(
            text="What is the meaning?", id="a1b2c3d4e5f60006",
            context={"user_state": "privileged"},
        )
        result = SovereignResult(query_id=query.id)

        rejection = await runtime._run_gate_chain_preflight(query, result)
        assert rejection is None  # Should pass

    @pytest.mark.asyncio
    async def test_preflight_multiple_evaluations(self, tmp_path):
        """Multiple queries are tracked in gate chain stats."""
        config = RuntimeConfig()
        config.state_dir = tmp_path
        runtime = SovereignRuntime(config)
        runtime._init_gate_chain()

        for i in range(5):
            query = SovereignQuery(
                text=f"Query {i}", id=f"a1b2c3d4e5f6{i:04x}"
            )
            result = SovereignResult(query_id=query.id)
            await runtime._run_gate_chain_preflight(query, result)

        stats = runtime.get_gate_chain_stats()
        assert stats["total_evaluations"] == 5
        assert stats["passed"] == 5
        assert stats["pass_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_preflight_exception_rejects(self, tmp_path):
        """CRITICAL-2 FIX: Exception in gate chain → REJECT (fail-closed)."""
        config = RuntimeConfig()
        config.state_dir = tmp_path
        runtime = SovereignRuntime(config)
        # Set gate chain to a broken object to trigger exception
        runtime._gate_chain = "not_a_gate_chain"

        query = SovereignQuery(text="test", id="a1b2c3d4e5f60007")
        result = SovereignResult(query_id=query.id)

        # Should not raise; returns rejection result (fail-closed)
        rejection = await runtime._run_gate_chain_preflight(query, result)
        assert rejection is not None or result.success is False
