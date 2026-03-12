"""Tests: SovereignOrganism → Node0Heartbeat bridge (P0 closure).

Validates the ONE canonical ingest authority:
  organism.boot() → Node0Heartbeat.boot()
  organism.mission() → Node0Heartbeat.ingest_mission_receipt()
  organism.tick() → Node0Heartbeat.breathe()

Standing on Giants:
  Deming (1950)   — PDCA closure verified end-to-end
  Nakamoto (2008) — Chain integrity across organism→node0 boundary
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ─── Minimal InferenceBackend mock ──────────────────────────────────


class EchoInference:
    """Minimal inference backend for testing (no LLM required)."""

    async def infer(self, prompt: str, **kwargs: Any) -> str:
        return f"[echo] {prompt[:50]}"


# ─── Test: Node0 wired at organism boot ──────────────────────────────


class TestOrganismNode0Boot:
    """Verify Node0Heartbeat is created and booted during organism boot."""

    @pytest.fixture
    def persistence_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "sovereign"

    def test_organism_has_node0_field(self) -> None:
        """SovereignOrganism.__init__ creates _node0 attribute."""
        from core.sovereign.organism import SovereignOrganism

        org = SovereignOrganism()
        assert hasattr(org, "_node0")
        assert org._node0 is None  # Not wired until boot()

    def test_boot_wires_node0(self, persistence_dir: Path) -> None:
        """boot() creates and boots Node0Heartbeat with organism's Helix3."""
        from core.sovereign.organism import SovereignOrganism

        org = asyncio.run(
            SovereignOrganism.boot(
                EchoInference(),
                persistence_dir=persistence_dir,
            )
        )

        assert org._node0 is not None
        assert org._node0._booted is True
        assert org._node0.node_id  # Non-empty node ID
        assert org._node0._boot_receipt is not None
        assert org._node0._boot_receipt.sovereignty_proven is True

    def test_node0_uses_organism_helix3(self, persistence_dir: Path) -> None:
        """Node0 should use organism's Helix3, not create its own."""
        from core.sovereign.organism import SovereignOrganism

        org = asyncio.run(
            SovereignOrganism.boot(
                EchoInference(),
                persistence_dir=persistence_dir,
            )
        )

        # Node0's Helix3 should be the same object as organism's
        assert org._node0._helix3 is org._helix3
        assert org._node0._external_helix3 is True

    def test_node0_in_stats(self, persistence_dir: Path) -> None:
        """Node0 health should appear in organism stats."""
        from core.sovereign.organism import SovereignOrganism

        org = asyncio.run(
            SovereignOrganism.boot(
                EchoInference(),
                persistence_dir=persistence_dir,
            )
        )

        stats = org.stats
        assert "node0" in stats
        assert stats["node0"]["booted"] is True
        assert "node_id" in stats["node0"]
        assert "subsystems" in stats["node0"]

    def test_boot_degraded_without_node0(self) -> None:
        """Boot should succeed even if Node0 wiring fails (degraded mode)."""
        from core.sovereign.organism import SovereignOrganism

        with patch(
            "core.node0.heartbeat.Node0Heartbeat.boot",
            side_effect=RuntimeError("simulated failure"),
        ):
            org = asyncio.run(SovereignOrganism.boot(EchoInference()))

        # Organism booted, Node0 degraded
        assert org.health.alive is True
        assert org._node0 is None


# ─── Test: Mission receipt ingest ─────────────────────────────────────


class TestOrganismMissionIngest:
    """Verify mission() bridges receipts to Node0Heartbeat."""

    @pytest.fixture
    def organism(self, tmp_path: Path) -> Any:
        from core.sovereign.organism import SovereignOrganism

        return asyncio.run(
            SovereignOrganism.boot(
                EchoInference(),
                persistence_dir=tmp_path / "sovereign",
            )
        )

    def test_mission_ingests_to_node0(self, organism: Any) -> None:
        """mission() should call _ingest_to_node0 with the receipt."""
        with patch.object(organism, "_ingest_to_node0") as mock_ingest:
            receipt = asyncio.run(organism.mission("test task"))
            mock_ingest.assert_called_once()
            call_arg = mock_ingest.call_args[0][0]
            assert call_arg.mission_id == receipt.mission_id

    def test_ingest_feeds_node0_heartbeat(self, organism: Any) -> None:
        """_ingest_to_node0 calls Node0Heartbeat.ingest_mission_receipt."""
        with patch.object(organism._node0, "ingest_mission_receipt") as mock_ingest:
            receipt = asyncio.run(organism.mission("test task"))
            mock_ingest.assert_called_once()
            ingest_dict = mock_ingest.call_args[0][0]
            assert ingest_dict["mission_id"] == receipt.mission_id
            assert "ihsan_score" in ingest_dict
            assert "description" in ingest_dict

    def test_mission_still_returns_on_node0_failure(self, organism: Any) -> None:
        """If Node0 ingest fails, mission should still return receipt."""
        organism._node0.ingest_mission_receipt = MagicMock(
            side_effect=RuntimeError("ingest boom")
        )

        receipt = asyncio.run(organism.mission("test despite failure"))
        assert receipt is not None
        assert receipt.mission_id  # Got a valid receipt back


# ─── Test: Tick delegates to Node0 breathe ─────────────────────────────


class TestOrganismTickBreathe:
    """Verify tick() delegates to Node0Heartbeat.breathe()."""

    @pytest.fixture
    def organism(self, tmp_path: Path) -> Any:
        from core.sovereign.organism import SovereignOrganism

        return asyncio.run(
            SovereignOrganism.boot(
                EchoInference(),
                persistence_dir=tmp_path / "sovereign",
            )
        )

    def test_tick_calls_node0_breathe(self, organism: Any) -> None:
        """tick() should delegate to Node0.breathe() when wired."""
        # Ingest a mission first so breathe has something to process
        organism._node0.ingest_mission_receipt(
            {
                "mission_id": "tick-test-001",
                "description": "Test task",
                "ihsan_score": 0.96,
                "agent_id": "P3",
            }
        )

        breath = asyncio.run(organism.tick())

        # Should return a BreathReceipt (from Node0), not HeartbeatReceipt
        assert hasattr(breath, "tick_number")
        assert hasattr(breath, "chain_hash")
        assert hasattr(breath, "evidence_entries") or hasattr(breath, "evidence_hash")

    def test_tick_fallback_without_node0(self, organism: Any) -> None:
        """tick() falls back to direct Helix3 if Node0 is None."""
        organism._node0 = None  # Simulate degraded mode

        receipt = asyncio.run(organism.tick())

        # Should still return something from Helix3
        assert hasattr(receipt, "tick_number")
        assert hasattr(receipt, "ihsan_composite")

    def test_tick_fallback_on_breathe_error(self, organism: Any) -> None:
        """tick() falls back to Helix3 if Node0.breathe() raises."""
        organism._node0.breathe = MagicMock(side_effect=RuntimeError("breathe failed"))

        receipt = asyncio.run(organism.tick())

        # Fallback to direct Helix3 tick
        assert hasattr(receipt, "tick_number")
        assert hasattr(receipt, "ihsan_composite")


# ─── Test: End-to-end chain integrity ──────────────────────────────


class TestOrganismNode0ChainIntegrity:
    """Verify evidence chain flows organism→Node0→evidence correctly."""

    @pytest.fixture
    def organism(self, tmp_path: Path) -> Any:
        from core.sovereign.organism import SovereignOrganism

        return asyncio.run(
            SovereignOrganism.boot(
                EchoInference(),
                persistence_dir=tmp_path / "sovereign",
            )
        )

    def test_chain_hash_updates_after_tick(self, organism: Any) -> None:
        """Node0 chain_hash should change after breathing."""
        hash_before = organism._node0.chain_hash

        # Run a mission → ingest → tick(breathe)
        asyncio.run(organism.mission("chain integrity test"))
        asyncio.run(organism.tick())

        hash_after = organism._node0.chain_hash
        assert hash_after != hash_before

    def test_full_lifecycle_organism_to_node0(self, organism: Any) -> None:
        """Full lifecycle: boot → 3 missions → tick → verify state."""
        # 3 missions
        for i in range(3):
            asyncio.run(organism.mission(f"lifecycle mission {i}"))

        # Tick (triggers breathe)
        asyncio.run(organism.tick())

        # Verify Node0 state
        health = organism._node0.health()
        assert health["booted"] is True
        assert health["tick_number"] >= 1

        # Verify organism stats include Node0
        stats = organism.stats
        assert "node0" in stats
        assert stats["node0"]["booted"] is True
