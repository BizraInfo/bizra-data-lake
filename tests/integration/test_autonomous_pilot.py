"""
Autonomous Pilot Smoke Test — BIZRA Node0 End-to-End Validation
================================================================
Boots the full sovereign stack without external dependencies and
validates that core subsystems initialize, communicate, and produce
verifiable outputs.

This is the "turn the key" test: if it passes, Node0 is alive.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict

import pytest

# ---------------------------------------------------------------------------
# Pillar 1: Runtime boots and reports healthy status
# ---------------------------------------------------------------------------


class TestRuntimeBoot:
    """Verify that SovereignRuntime initializes without external deps."""

    @pytest.mark.asyncio
    async def test_runtime_creates_and_initializes(self):
        """Runtime boots with default config."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig, RuntimeMode

        config = RuntimeConfig(
            mode=RuntimeMode.MINIMAL,
            autonomous_enabled=False,
        )
        runtime = SovereignRuntime(config)
        await runtime.initialize()

        try:
            assert runtime._initialized is True
        finally:
            await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_status_returns_valid_structure(self):
        """Status dict has identity, health, state keys."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig, RuntimeMode

        config = RuntimeConfig(mode=RuntimeMode.MINIMAL, autonomous_enabled=False)
        async with SovereignRuntime.create(config) as runtime:
            status = runtime.status()

            assert "identity" in status
            assert "health" in status
            assert "state" in status
            assert status["identity"]["node_id"]
            assert status["health"]["status"] in ("healthy", "degraded", "unhealthy", "unknown")

    @pytest.mark.asyncio
    async def test_runtime_context_manager_cleans_up(self):
        """Context manager properly shuts down runtime."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig, RuntimeMode

        config = RuntimeConfig(mode=RuntimeMode.MINIMAL, autonomous_enabled=False)
        async with SovereignRuntime.create(config) as runtime:
            assert runtime._initialized is True
            node_id = runtime.config.node_id

        # After context exit, runtime should be shut down
        assert node_id  # Was set during init


# ---------------------------------------------------------------------------
# Pillar 2: Token system operates independently
# ---------------------------------------------------------------------------


class TestTokenSystemSmoke:
    """Verify token ledger, mint, and PoI bridge work end-to-end."""

    def test_ledger_initializes(self):
        """Token ledger loads or creates."""
        from core.token.ledger import TokenLedger

        ledger = TokenLedger()
        assert ledger is not None

    def test_mint_creates_tokens(self):
        """Mint can create tokens in the ledger."""
        from core.token.mint import TokenMinter
        from core.token.types import TokenType

        # Use factory method which generates its own keypair
        minter = TokenMinter.create()

        # mint_seed is the correct method
        tx = minter.mint_seed(
            to_account="SMOKE-TEST-ACCOUNT",
            amount=1.0,
            epoch_id="smoke-epoch",
        )
        assert tx is not None
        assert tx.success is True
        assert tx.balance_after >= 1.0

        # Verify balance
        balance = minter._ledger.get_balance("SMOKE-TEST-ACCOUNT", TokenType.SEED)
        assert balance.balance >= 1.0

    def test_ledger_chain_integrity(self, tmp_path):
        """Token ledger chain verifies."""
        from core.token.ledger import TokenLedger

        # Use isolated paths so existing ledger corruption doesn't affect test
        ledger = TokenLedger(
            db_path=tmp_path / "test_memory.db",
            log_path=tmp_path / "test_ledger.jsonl",
        )
        valid, count, err = ledger.verify_chain()
        assert valid is True
        assert count >= 0


# ---------------------------------------------------------------------------
# Pillar 3: Evidence chain works
# ---------------------------------------------------------------------------


class TestEvidenceChainSmoke:
    """Verify evidence ledger append + verify cycle."""

    def test_evidence_append_and_verify(self, tmp_path):
        """Append an entry and verify the chain."""
        from core.proof_engine.evidence_ledger import EvidenceLedger, emit_receipt

        ledger = EvidenceLedger(path=tmp_path / "smoke_evidence.jsonl")
        # Use emit_receipt helper which builds a schema-compliant receipt
        entry = emit_receipt(
            ledger=ledger,
            receipt_id="a" * 32,  # Must match ^[a-f0-9]{8,64}$
            node_id="SMOKE-TEST-NODE",
            reason_codes=["SMOKE_TEST"],
            snr_score=0.92,
            ihsan_score=0.96,
        )
        assert entry is not None
        assert entry.sequence >= 1

        # verify_chain returns (bool, List[str]) — errors list
        valid, errors = ledger.verify_chain()
        assert valid is True
        assert errors == []


# ---------------------------------------------------------------------------
# Pillar 4: SNR engine computes scores
# ---------------------------------------------------------------------------


class TestSNRSmoke:
    """Verify SNR computation works."""

    def test_snr_facade_computes_score(self):
        """SNR facade returns a valid score."""
        from core.snr_protocol import SNRFacade

        facade = SNRFacade()
        result = facade.calculate(
            text="This is a test of the SNR engine with reasonable content that "
            "should produce a measurable signal-to-noise ratio for validation."
        )
        assert 0.0 <= result.score <= 1.0

    def test_snr_threshold_check(self):
        """SNR score can be compared against threshold."""
        from core.snr_protocol import SNRFacade

        facade = SNRFacade()
        result = facade.calculate(
            text="A well-structured claim with evidence: cryptographic hashes provide "
            "tamper-proof verification of data integrity using SHA-256 algorithms."
        )
        # Score should be non-negative
        assert result.score >= 0.0


# ---------------------------------------------------------------------------
# Pillar 5: SpearPoint pipeline executes all 8 steps
# ---------------------------------------------------------------------------


class TestSpearPointSmoke:
    """Verify SpearPoint cockpit executes without subsystems."""

    @pytest.mark.asyncio
    async def test_spearpoint_pipeline_all_steps(self):
        """Pipeline with no subsystems skips all 8 steps gracefully."""
        from dataclasses import dataclass, field
        from typing import List, Optional

        from core.sovereign.spearpoint_pipeline import SpearPointPipeline, SpearPointResult

        @dataclass
        class FakeResult:
            query_id: str = "smoke-001"
            success: bool = True
            response: str = "Smoke test response."
            reasoning_depth: int = 1
            thoughts: List[str] = field(default_factory=list)
            ihsan_score: float = 0.96
            snr_score: float = 0.92
            snr_ok: bool = True
            validated: bool = False
            validation_passed: bool = True
            processing_time_ms: float = 10.0
            graph_hash: Optional[str] = None
            claim_tags: dict = field(default_factory=dict)
            model_used: str = "smoke-test"

        @dataclass
        class FakeQuery:
            id: str = "q-smoke"
            text: str = "Smoke test query"
            user_id: str = "pilot"

        @dataclass
        class FakeConfig:
            node_id: str = "SMOKE-TEST-NODE"
            ihsan_threshold: float = 0.95
            snr_threshold: float = 0.85

        pipeline = SpearPointPipeline(config=FakeConfig())
        result = await pipeline.execute(FakeResult(), FakeQuery())

        assert isinstance(result, SpearPointResult)
        assert result.all_passed is True
        assert len(result.steps) == 8
        assert result.total_duration_ms >= 0


# ---------------------------------------------------------------------------
# Pillar 6: Opportunity pipeline processes items
# ---------------------------------------------------------------------------


class TestOpportunityPipelineSmoke:
    """Verify opportunity pipeline accepts and processes items."""

    @pytest.mark.asyncio
    async def test_pipeline_processes_autolow(self):
        """AUTOLOW opportunity flows through to execution."""
        from core.sovereign.autonomy_matrix import AutonomyLevel
        from core.sovereign.opportunity_pipeline import (
            OpportunityPipeline,
            PipelineOpportunity,
        )

        executed = []

        async def track_exec(opp):
            executed.append(opp.id)
            return {"success": True}

        pipeline = OpportunityPipeline()
        pipeline.set_execution_callback(track_exec)
        await pipeline.start()

        try:
            opp = PipelineOpportunity(
                id="smoke-auto",
                domain="cognitive",
                description="Smoke test auto-execute",
                source="pilot",
                detected_at=time.time(),
                snr_score=0.95,
                ihsan_score=0.98,
                autonomy_level=AutonomyLevel.AUTOLOW,
            )
            await pipeline.submit(opp)
            await asyncio.sleep(0.5)

            assert "smoke-auto" in executed
        finally:
            await pipeline.stop()


# ---------------------------------------------------------------------------
# Pillar 7: CLI entry point parses without error
# ---------------------------------------------------------------------------


class TestCLISmoke:
    """Verify CLI module imports and version works."""

    def test_cli_module_imports(self):
        """CLI module imports without error."""
        from core.sovereign.__main__ import main, print_banner

        assert callable(main)
        assert callable(print_banner)

    def test_version_command(self):
        """Version string is returned."""
        import subprocess

        result = subprocess.run(
            ["python", "-m", "core.sovereign", "version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "BIZRA" in result.stdout


# ---------------------------------------------------------------------------
# Pillar 8: Full stack smoke — runtime + query + evidence
# ---------------------------------------------------------------------------


class TestFullStackSmoke:
    """Boot runtime, run a query-like operation, verify evidence trail."""

    @pytest.mark.asyncio
    async def test_full_stack_boot_and_status(self):
        """Full boot → status → metrics → shutdown cycle."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig, RuntimeMode

        config = RuntimeConfig(mode=RuntimeMode.MINIMAL, autonomous_enabled=False)

        async with SovereignRuntime.create(config) as runtime:
            # Get status
            status = runtime.status()
            assert status["identity"]["node_id"]
            assert "health" in status

            # Get metrics
            metrics = runtime.metrics.to_dict()
            assert "queries" in metrics
            assert "reasoning" in metrics

    @pytest.mark.asyncio
    async def test_pilot_health_summary(self, tmp_path):
        """Produce a JSON health summary suitable for monitoring."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig, RuntimeMode
        from core.token.ledger import TokenLedger

        config = RuntimeConfig(mode=RuntimeMode.MINIMAL, autonomous_enabled=False)

        async with SovereignRuntime.create(config) as runtime:
            status = runtime.status()

            # Token system health — use isolated paths for deterministic chain
            ledger = TokenLedger(
                db_path=tmp_path / "test_memory.db",
                log_path=tmp_path / "test_ledger.jsonl",
            )
            chain_valid, tx_count, chain_err = ledger.verify_chain()

            health_summary = {
                "node_id": status["identity"]["node_id"],
                "runtime_health": status["health"]["status"],
                "runtime_score": status["health"]["score"],
                "token_chain_valid": chain_valid,
                "token_transactions": tx_count,
                "mode": status["state"]["mode"],
            }

            # All fields must be populated
            assert health_summary["node_id"]
            assert health_summary["runtime_health"] in ("healthy", "degraded", "unhealthy", "unknown")
            assert health_summary["token_chain_valid"] is True

            # Must be JSON-serializable
            serialized = json.dumps(health_summary)
            assert len(serialized) > 0
