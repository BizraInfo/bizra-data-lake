"""
Runtime Integration Tests — Verifies SovereignRuntime Component Wiring
=======================================================================
Tests that all real components load (not stubs) when initialized,
and that lifecycle, graph artifacts, and evidence ledger work end-to-end.

Standing on Giants:
- Lamport (event ordering)
- Merkle (hash chains)
- Besta (GoT graph artifacts)
- Shannon (SNR as quality)

Categories:
1. TestComponentImports      — verify importability of real components
2. TestRuntimeInitialization — lifecycle: init, wire, shutdown
3. TestRealComponentsLoad    — stubs are NOT used when real exists
4. TestStubFallback          — stubs ARE used when config disables
5. TestGraphArtifact         — graph hash, artifact schema, to_dict
6. TestEvidenceLedgerIntegration — append, chain, tamper detection
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_valid_receipt(
    receipt_id: str = "",
    node_id: str = "test-node",
    snr_score: float = 0.95,
    ihsan_score: float = 0.96,
) -> Dict[str, Any]:
    """Build a receipt dict that passes schema validation."""
    rid = receipt_id or uuid.uuid4().hex[:32]
    return {
        "receipt_id": rid,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "node_id": node_id,
        "policy_version": "1.0.0",
        "status": "accepted",
        "decision": "APPROVED",
        "reason_codes": [],
        "snr": {"score": snr_score},
        "ihsan": {
            "score": ihsan_score,
            "threshold": 0.95,
            "decision": "APPROVED",
        },
        "seal": {
            "algorithm": "blake3",
            "digest": "0" * 64,
        },
    }


# ===========================================================================
# 1. TestComponentImports — Verify all real components can be imported
# ===========================================================================


class TestComponentImports:
    """Verify that every real component module is importable.

    If any of these fail, the runtime would silently fall back to stubs.
    These tests guard against accidental breakage of import paths.
    """

    def test_graph_of_thoughts_importable(self) -> None:
        """GraphOfThoughts is importable from core.sovereign.graph_reasoner."""
        from core.sovereign.graph_reasoner import GraphOfThoughts

        assert GraphOfThoughts is not None
        # Verify it is a class, not a stub
        got = GraphOfThoughts()
        assert hasattr(got, "reason"), "GraphOfThoughts must have a reason() method"
        assert hasattr(got, "add_thought"), "GraphOfThoughts must have add_thought()"

    def test_snr_maximizer_importable(self) -> None:
        """SNRMaximizer is importable from core.sovereign.snr_maximizer."""
        from core.sovereign.snr_maximizer import SNRMaximizer

        assert SNRMaximizer is not None
        snr = SNRMaximizer(ihsan_threshold=0.95)
        assert hasattr(snr, "optimize"), "SNRMaximizer must have an optimize() method"

    def test_guardian_council_importable(self) -> None:
        """GuardianCouncil is importable from core.sovereign.guardian_council."""
        from core.sovereign.guardian_council import GuardianCouncil

        assert GuardianCouncil is not None
        gc = GuardianCouncil()
        assert hasattr(gc, "validate"), "GuardianCouncil must have a validate() method"

    def test_autonomous_loop_importable(self) -> None:
        """AutonomousLoop is importable from core.sovereign.autonomy."""
        from core.sovereign.autonomy import AutonomousLoop, DecisionGate

        assert AutonomousLoop is not None
        assert DecisionGate is not None
        gate = DecisionGate(ihsan_threshold=0.95)
        loop = AutonomousLoop(decision_gate=gate)
        assert hasattr(loop, "start"), "AutonomousLoop must have a start() method"
        assert hasattr(loop, "stop"), "AutonomousLoop must have a stop() method"

    def test_evidence_ledger_importable(self) -> None:
        """EvidenceLedger is importable from core.proof_engine.evidence_ledger."""
        from core.proof_engine.evidence_ledger import EvidenceLedger, emit_receipt

        assert EvidenceLedger is not None
        assert emit_receipt is not None


# ===========================================================================
# 2. TestRuntimeInitialization — Runtime lifecycle tests
# ===========================================================================


class TestRuntimeInitialization:
    """Test SovereignRuntime lifecycle: creation, init, component wiring, shutdown."""

    def test_runtime_creates_with_defaults(self, tmp_path: Path) -> None:
        """SovereignRuntime can be created with default RuntimeConfig."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig

        config = RuntimeConfig()
        config.state_dir = tmp_path
        config.autonomous_enabled = False
        runtime = SovereignRuntime(config)

        assert runtime is not None
        assert runtime.config.ihsan_threshold == 0.95
        assert runtime.config.enable_graph_reasoning is True
        assert runtime._initialized is False

    @pytest.mark.asyncio
    async def test_runtime_initializes_evidence_ledger(self, tmp_path: Path) -> None:
        """After initialize(), _evidence_ledger is not None."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig

        config = RuntimeConfig()
        config.state_dir = tmp_path
        config.autonomous_enabled = False
        runtime = SovereignRuntime(config)
        await runtime.initialize()
        try:
            assert runtime._evidence_ledger is not None, (
                "Evidence Ledger should be initialized after runtime.initialize()"
            )
        finally:
            await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_runtime_initializes_gate_chain(self, tmp_path: Path) -> None:
        """After initialize(), _gate_chain is not None."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig

        config = RuntimeConfig()
        config.state_dir = tmp_path
        config.autonomous_enabled = False
        runtime = SovereignRuntime(config)
        await runtime.initialize()
        try:
            assert runtime._gate_chain is not None, (
                "GateChain should be initialized after runtime.initialize()"
            )
        finally:
            await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_runtime_initializes_poi_engine(self, tmp_path: Path) -> None:
        """After initialize(), _poi_orchestrator is not None."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig

        config = RuntimeConfig()
        config.state_dir = tmp_path
        config.autonomous_enabled = False
        runtime = SovereignRuntime(config)
        await runtime.initialize()
        try:
            assert runtime._poi_orchestrator is not None, (
                "PoI Orchestrator should be initialized after runtime.initialize()"
            )
        finally:
            await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_runtime_initializes_sat_controller(self, tmp_path: Path) -> None:
        """After initialize(), _sat_controller is not None."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig

        config = RuntimeConfig()
        config.state_dir = tmp_path
        config.autonomous_enabled = False
        runtime = SovereignRuntime(config)
        await runtime.initialize()
        try:
            assert runtime._sat_controller is not None, (
                "SAT Controller should be initialized after runtime.initialize()"
            )
        finally:
            await runtime.shutdown()


# ===========================================================================
# 3. TestRealComponentsLoad — Stubs NOT used when real components exist
# ===========================================================================


class TestRealComponentsLoad:
    """Verify that real components are loaded (not stubs) when feature flags are enabled.

    Each test checks that the component does NOT have `is_stub = True`,
    which is the marker attribute set by StubFactory on all stubs.
    """

    @pytest.mark.asyncio
    async def test_graph_reasoner_not_stub(self, tmp_path: Path) -> None:
        """GraphOfThoughts should be real, not a stub."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig

        config = RuntimeConfig()
        config.state_dir = tmp_path
        config.autonomous_enabled = False
        config.enable_graph_reasoning = True
        runtime = SovereignRuntime(config)
        await runtime.initialize()
        try:
            assert runtime._graph_reasoner is not None
            assert not getattr(runtime._graph_reasoner, "is_stub", False), (
                "Graph reasoner should be real, not a stub"
            )
        finally:
            await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_snr_optimizer_not_stub(self, tmp_path: Path) -> None:
        """SNRMaximizer should be real, not a stub."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig

        config = RuntimeConfig()
        config.state_dir = tmp_path
        config.autonomous_enabled = False
        config.enable_snr_optimization = True
        runtime = SovereignRuntime(config)
        await runtime.initialize()
        try:
            assert runtime._snr_optimizer is not None
            assert not getattr(runtime._snr_optimizer, "is_stub", False), (
                "SNR optimizer should be real, not a stub"
            )
        finally:
            await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_guardian_council_not_stub(self, tmp_path: Path) -> None:
        """GuardianCouncil should be real, not a stub."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig

        config = RuntimeConfig()
        config.state_dir = tmp_path
        config.autonomous_enabled = False
        config.enable_guardian_validation = True
        runtime = SovereignRuntime(config)
        await runtime.initialize()
        try:
            assert runtime._guardian_council is not None
            assert not getattr(runtime._guardian_council, "is_stub", False), (
                "Guardian council should be real, not a stub"
            )
        finally:
            await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_autonomous_loop_not_stub(self, tmp_path: Path) -> None:
        """AutonomousLoop should be real, not a stub."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig

        config = RuntimeConfig()
        config.state_dir = tmp_path
        config.autonomous_enabled = False
        config.enable_autonomous_loop = True
        runtime = SovereignRuntime(config)
        await runtime.initialize()
        try:
            assert runtime._autonomous_loop is not None
            assert not getattr(runtime._autonomous_loop, "is_stub", False), (
                "Autonomous loop should be real, not a stub"
            )
        finally:
            await runtime.shutdown()


# ===========================================================================
# 4. TestStubFallback — Stubs ARE used when config disables components
# ===========================================================================


class TestStubFallback:
    """When feature flags are disabled, verify stubs are used."""

    @pytest.mark.asyncio
    async def test_graph_reasoner_stub_when_disabled(self, tmp_path: Path) -> None:
        """Graph reasoner should be a stub when enable_graph_reasoning=False."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig

        config = RuntimeConfig()
        config.state_dir = tmp_path
        config.autonomous_enabled = False
        config.enable_graph_reasoning = False
        runtime = SovereignRuntime(config)
        await runtime.initialize()
        try:
            assert runtime._graph_reasoner is not None
            assert getattr(runtime._graph_reasoner, "is_stub", False) is True, (
                "Graph reasoner should be a stub when disabled by config"
            )
        finally:
            await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_snr_optimizer_stub_when_disabled(self, tmp_path: Path) -> None:
        """SNR optimizer should be a stub when enable_snr_optimization=False."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig

        config = RuntimeConfig()
        config.state_dir = tmp_path
        config.autonomous_enabled = False
        config.enable_snr_optimization = False
        runtime = SovereignRuntime(config)
        await runtime.initialize()
        try:
            assert runtime._snr_optimizer is not None
            assert getattr(runtime._snr_optimizer, "is_stub", False) is True, (
                "SNR optimizer should be a stub when disabled by config"
            )
        finally:
            await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_guardian_council_stub_when_disabled(self, tmp_path: Path) -> None:
        """Guardian council should be a stub when enable_guardian_validation=False."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig

        config = RuntimeConfig()
        config.state_dir = tmp_path
        config.autonomous_enabled = False
        config.enable_guardian_validation = False
        runtime = SovereignRuntime(config)
        await runtime.initialize()
        try:
            assert runtime._guardian_council is not None
            assert getattr(runtime._guardian_council, "is_stub", False) is True, (
                "Guardian council should be a stub when disabled by config"
            )
        finally:
            await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_autonomous_loop_stub_when_disabled(self, tmp_path: Path) -> None:
        """Autonomous loop should be a stub when enable_autonomous_loop=False."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig

        config = RuntimeConfig()
        config.state_dir = tmp_path
        config.autonomous_enabled = False
        config.enable_autonomous_loop = False
        runtime = SovereignRuntime(config)
        await runtime.initialize()
        try:
            assert runtime._autonomous_loop is not None
            assert getattr(runtime._autonomous_loop, "is_stub", False) is True, (
                "Autonomous loop should be a stub when disabled by config"
            )
        finally:
            await runtime.shutdown()


# ===========================================================================
# 5. TestGraphArtifact — Graph hash determinism, artifact schema, to_dict
# ===========================================================================


class TestGraphArtifact:
    """Test GoT graph artifact production, deterministic hashing, and schema."""

    def test_graph_hash_deterministic(self) -> None:
        """compute_graph_hash() returns the same value on repeated calls."""
        from core.sovereign.graph_core import GraphOfThoughts
        from core.sovereign.graph_types import ThoughtType

        got = GraphOfThoughts()

        # Build a non-trivial graph
        root = got.add_thought("What is sovereignty?", ThoughtType.QUESTION)
        h1 = got.add_thought(
            "Self-governance of data",
            ThoughtType.HYPOTHESIS,
            parent_id=root.id,
        )
        h2 = got.add_thought(
            "Decentralized identity",
            ThoughtType.HYPOTHESIS,
            parent_id=root.id,
        )
        _e1 = got.add_thought(
            "GDPR precedent supports data ownership",
            ThoughtType.EVIDENCE,
            parent_id=h1.id,
        )

        hash_1 = got.compute_graph_hash()
        hash_2 = got.compute_graph_hash()

        assert hash_1 == hash_2, (
            "Graph hash must be deterministic: two calls with same state "
            f"must match. Got {hash_1} vs {hash_2}"
        )
        assert len(hash_1) == 64, "Graph hash should be a SHA-256 hex digest (64 chars)"

    def test_graph_to_artifact_schema(self) -> None:
        """to_artifact() produces a dict with required schema fields."""
        from core.sovereign.graph_core import GraphOfThoughts
        from core.sovereign.graph_types import ThoughtType

        got = GraphOfThoughts()
        root = got.add_thought("Test question", ThoughtType.QUESTION)
        _h = got.add_thought(
            "Test hypothesis",
            ThoughtType.HYPOTHESIS,
            parent_id=root.id,
        )

        artifact = got.to_artifact(build_id="test-build-001")

        # Required top-level keys per reasoning_graph schema
        assert "nodes" in artifact, "Artifact must contain 'nodes'"
        assert "edges" in artifact, "Artifact must contain 'edges'"
        assert "roots" in artifact, "Artifact must contain 'roots'"
        assert "graph_hash" in artifact, "Artifact must contain 'graph_hash'"
        assert "stats" in artifact, "Artifact must contain 'stats'"
        assert "config" in artifact, "Artifact must contain 'config'"
        assert "build_id" in artifact, "Artifact must contain 'build_id' when provided"

        # Node structure
        assert len(artifact["nodes"]) == 2
        node = artifact["nodes"][0]
        assert "id" in node
        assert "content" in node
        assert "type" in node
        assert "content_hash" in node
        assert "confidence" in node
        assert "snr" in node
        assert "ihsan" in node
        assert "depth" in node

    def test_graph_to_dict_includes_hash(self) -> None:
        """to_dict() includes graph_hash field."""
        from core.sovereign.graph_core import GraphOfThoughts
        from core.sovereign.graph_types import ThoughtType

        got = GraphOfThoughts()
        got.add_thought("Root thought", ThoughtType.QUESTION)

        result = got.to_dict()

        assert "graph_hash" in result, "to_dict() must include graph_hash"
        assert isinstance(result["graph_hash"], str)
        assert len(result["graph_hash"]) == 64
        assert "nodes" in result
        assert "edges" in result
        assert "roots" in result
        assert "stats" in result


# ===========================================================================
# 6. TestEvidenceLedgerIntegration — append, chain, tamper detection
# ===========================================================================


class TestEvidenceLedgerIntegration:
    """Test the Evidence Ledger: append, sequence, chain integrity, tamper detection."""

    def test_ledger_append_increments_sequence(self, tmp_path: Path) -> None:
        """Each append() call increments the sequence number."""
        from core.proof_engine.evidence_ledger import EvidenceLedger

        ledger_path = tmp_path / "test_evidence.jsonl"
        ledger = EvidenceLedger(ledger_path, validate_on_append=True)

        assert ledger.sequence == 0, "Fresh ledger starts at sequence 0"

        receipt_1 = _build_valid_receipt(receipt_id="a" * 32)
        ledger.append(receipt_1)
        assert ledger.sequence == 1

        receipt_2 = _build_valid_receipt(receipt_id="b" * 32)
        ledger.append(receipt_2)
        assert ledger.sequence == 2

        receipt_3 = _build_valid_receipt(receipt_id="c" * 32)
        ledger.append(receipt_3)
        assert ledger.sequence == 3

    def test_ledger_hash_chain_valid(self, tmp_path: Path) -> None:
        """After multiple appends, verify_chain() returns True."""
        from core.proof_engine.evidence_ledger import EvidenceLedger

        ledger_path = tmp_path / "chain_test.jsonl"
        ledger = EvidenceLedger(ledger_path, validate_on_append=True)

        # Append several entries
        for i in range(5):
            rid = f"{i:032x}"
            ledger.append(_build_valid_receipt(receipt_id=rid))

        is_valid, errors = ledger.verify_chain()
        assert is_valid is True, f"Chain should be valid, got errors: {errors}"
        assert len(errors) == 0

    def test_ledger_verify_chain_detects_tampering(self, tmp_path: Path) -> None:
        """Modifying a ledger entry causes verify_chain() to detect tampering."""
        from core.proof_engine.evidence_ledger import EvidenceLedger

        ledger_path = tmp_path / "tamper_test.jsonl"
        ledger = EvidenceLedger(ledger_path, validate_on_append=True)

        # Append entries
        for i in range(3):
            rid = f"{i:032x}"
            ledger.append(_build_valid_receipt(receipt_id=rid))

        # Read and tamper with the file: modify the second line's receipt_id
        lines = ledger_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3

        # Tamper with the second entry: change a character in entry_hash
        entry_data = json.loads(lines[1])
        original_hash = entry_data["entry_hash"]
        tampered_hash = "f" * 64 if original_hash != "f" * 64 else "0" * 64
        entry_data["entry_hash"] = tampered_hash
        lines[1] = json.dumps(entry_data, separators=(",", ":"), sort_keys=True)

        ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Re-create ledger from tampered file and verify
        tampered_ledger = EvidenceLedger(ledger_path, validate_on_append=False)
        is_valid, errors = tampered_ledger.verify_chain()

        assert is_valid is False, "Tampered chain should fail verification"
        assert len(errors) > 0, "Should report at least one error"

    def test_emit_receipt_creates_entry(self, tmp_path: Path) -> None:
        """emit_receipt() convenience function creates a valid ledger entry."""
        from core.proof_engine.evidence_ledger import EvidenceLedger, emit_receipt

        ledger_path = tmp_path / "emit_test.jsonl"
        ledger = EvidenceLedger(ledger_path, validate_on_append=True)

        entry = emit_receipt(
            ledger,
            receipt_id="a" * 32,
            node_id="test-node-001",
            policy_version="1.0.0",
            status="accepted",
            decision="APPROVED",
            reason_codes=[],
            snr_score=0.96,
            ihsan_score=0.97,
            ihsan_threshold=0.95,
            seal_digest="0" * 64,
            duration_ms=42.5,
        )

        assert entry is not None
        assert entry.sequence == 1
        assert entry.receipt["receipt_id"] == "a" * 32
        assert entry.receipt["node_id"] == "test-node-001"
        assert entry.receipt["snr"]["score"] == 0.96
        assert entry.receipt["ihsan"]["score"] == 0.97
        assert entry.receipt["ihsan"]["decision"] == "APPROVED"

        # Chain should still be valid
        is_valid, errors = ledger.verify_chain()
        assert is_valid is True, f"Chain should be valid after emit_receipt: {errors}"


# ===========================================================================
# 7. TestRuntimeShutdownSafety — Additional lifecycle edge cases
# ===========================================================================


class TestRuntimeShutdownSafety:
    """Extra lifecycle safety tests for double-init, double-shutdown, etc."""

    @pytest.mark.asyncio
    async def test_double_initialize_is_idempotent(self, tmp_path: Path) -> None:
        """Calling initialize() twice does not break the runtime."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig

        config = RuntimeConfig()
        config.state_dir = tmp_path
        config.autonomous_enabled = False
        runtime = SovereignRuntime(config)
        await runtime.initialize()
        try:
            # Second initialize should be a no-op (guarded by _initialized flag)
            await runtime.initialize()
            assert runtime._initialized is True
        finally:
            await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_double_shutdown_is_safe(self, tmp_path: Path) -> None:
        """Calling shutdown() twice does not raise."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig

        config = RuntimeConfig()
        config.state_dir = tmp_path
        config.autonomous_enabled = False
        runtime = SovereignRuntime(config)
        await runtime.initialize()
        await runtime.shutdown()
        # Second shutdown should not raise
        await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_runtime_initialized_flag_set(self, tmp_path: Path) -> None:
        """After initialize(), _initialized is True; after shutdown(), _running is False."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig

        config = RuntimeConfig()
        config.state_dir = tmp_path
        config.autonomous_enabled = False
        runtime = SovereignRuntime(config)

        assert runtime._initialized is False
        assert runtime._running is False

        await runtime.initialize()
        assert runtime._initialized is True
        assert runtime._running is True

        await runtime.shutdown()
        assert runtime._running is False
