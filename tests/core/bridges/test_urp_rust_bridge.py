"""
Tests for URP Rust Bridge — both degraded and live paths.

Tests are structured to pass regardless of whether PyO3 is built:
- Degradation tests force-disable the bridge to verify None returns
- Live tests run when PyO3 IS available (skipif not)
- Mocked tests verify the wrapper logic independent of backend

Standing on Giants: Liskov (substitution), Meszaros (test doubles)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from core.bridges.urp_rust_bridge import URPRustBridge

# Detect whether the real Rust backend is available
_RUST_AVAILABLE = URPRustBridge().available


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@dataclass(frozen=True)
class FakePledge:
    """Minimal pledge stub for testing."""

    node_id: str = "test-node-001"
    ram_gb: int = 16
    vram_gb: int = 6
    storage_gb: int = 0
    signed: bool = False
    signature: str = ""
    signer_public_key: str = ""
    pledge_hash: str = "abcd1234"
    pledged_at: str = "2026-03-14T00:00:00Z"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "ram_gb": self.ram_gb,
            "vram_gb": self.vram_gb,
            "storage_gb": self.storage_gb,
            "signed": self.signed,
            "signature": self.signature,
            "signer_public_key": self.signer_public_key,
            "pledge_hash": self.pledge_hash,
            "pledged_at": self.pledged_at,
        }


@pytest.fixture()
def fake_pledge() -> FakePledge:
    return FakePledge()


def _make_disabled_bridge() -> URPRustBridge:
    """Create a bridge with Rust explicitly disabled."""
    bridge = URPRustBridge.__new__(URPRustBridge)
    bridge._available = False
    bridge._pool = None
    return bridge


# -------------------------------------------------------------------------
# Test: Bridge unavailable (Level 0 degradation — forced off)
# -------------------------------------------------------------------------


class TestURPRustBridgeDegradation:
    """All operations return None when Rust is forced unavailable."""

    def test_bridge_not_available(self) -> None:
        bridge = _make_disabled_bridge()
        assert bridge.available is False

    def test_submit_pledge_returns_none(self, fake_pledge: FakePledge) -> None:
        bridge = _make_disabled_bridge()
        assert bridge.submit_pledge(fake_pledge) is None

    def test_contribute_returns_none(self) -> None:
        bridge = _make_disabled_bridge()
        result = bridge.contribute("node1", "cpu", 2.0, 3600000, "hash123")
        assert result is None

    def test_get_rewards_returns_none(self) -> None:
        bridge = _make_disabled_bridge()
        assert bridge.get_rewards("node1") is None

    def test_process_zakat_returns_none(self) -> None:
        bridge = _make_disabled_bridge()
        assert bridge.process_zakat() is None

    def test_check_adl_returns_none(self) -> None:
        bridge = _make_disabled_bridge()
        assert bridge.check_adl() is None

    def test_stats_returns_none(self) -> None:
        bridge = _make_disabled_bridge()
        assert bridge.stats() is None


# -------------------------------------------------------------------------
# Test: Bridge available (mocked Rust backend)
# -------------------------------------------------------------------------


class TestURPRustBridgeMocked:
    """Operations succeed with mocked Rust backend."""

    def _make_available_bridge(self) -> URPRustBridge:
        """Create a bridge that thinks Rust is available."""
        bridge = URPRustBridge.__new__(URPRustBridge)
        bridge._available = True
        bridge._pool = MagicMock()
        return bridge

    def test_submit_pledge_success(self, fake_pledge: FakePledge) -> None:
        bridge = self._make_available_bridge()
        mock_node = MagicMock()
        mock_node.to_dict.return_value = {"id": "node1", "class": "standard"}

        with patch.dict(
            "sys.modules",
            {"bizra": MagicMock(
                PyURPPledge=MagicMock(from_dict=MagicMock()),
                submit_pledge=MagicMock(return_value=mock_node),
            )},
        ):
            result = bridge.submit_pledge(fake_pledge)
            assert result == {"id": "node1", "class": "standard"}

    def test_submit_pledge_runtime_error(self, fake_pledge: FakePledge) -> None:
        bridge = self._make_available_bridge()

        with patch.dict(
            "sys.modules",
            {"bizra": MagicMock(
                PyURPPledge=MagicMock(
                    from_dict=MagicMock(side_effect=RuntimeError("bad pledge")),
                ),
            )},
        ):
            result = bridge.submit_pledge(fake_pledge)
            assert result is None

    def test_contribute_success(self) -> None:
        bridge = self._make_available_bridge()
        mock_receipt = MagicMock()
        mock_receipt.to_dict.return_value = {
            "tokens_earned": 100,
            "resource_type": "cpu",
        }

        with patch.dict(
            "sys.modules",
            {"bizra": MagicMock(
                contribute_resources=MagicMock(return_value=mock_receipt),
            )},
        ):
            result = bridge.contribute("n1", "cpu", 2.0, 3600000, "h")
            assert result is not None
            assert result["tokens_earned"] == 100

    def test_get_rewards_success(self) -> None:
        bridge = self._make_available_bridge()
        expected = {"balance": 500, "total_earned": 600}

        with patch.dict(
            "sys.modules",
            {"bizra": MagicMock(get_rewards=MagicMock(return_value=expected))},
        ):
            result = bridge.get_rewards("n1")
            assert result == expected

    def test_check_adl_success(self) -> None:
        bridge = self._make_available_bridge()
        expected = {"gini_coefficient": 0.25, "compliant": True}

        with patch.dict(
            "sys.modules",
            {"bizra": MagicMock(check_adl=MagicMock(return_value=expected))},
        ):
            result = bridge.check_adl()
            assert result == expected

    def test_stats_success(self) -> None:
        bridge = self._make_available_bridge()
        mock_stats = MagicMock()
        mock_stats.to_dict.return_value = {"total_nodes": 42, "adl_compliant": True}

        with patch.dict(
            "sys.modules",
            {"bizra": MagicMock(pool_stats=MagicMock(return_value=mock_stats))},
        ):
            result = bridge.stats()
            assert result == {"total_nodes": 42, "adl_compliant": True}

    def test_process_zakat_success(self) -> None:
        bridge = self._make_available_bridge()
        expected = {"total_distributed": 250, "eligible_nodes": 10}

        with patch.dict(
            "sys.modules",
            {"bizra": MagicMock(
                process_zakat=MagicMock(return_value=expected),
            )},
        ):
            result = bridge.process_zakat()
            assert result == expected


# -------------------------------------------------------------------------
# Test: Live Rust backend (only when PyO3 built)
# -------------------------------------------------------------------------


@pytest.mark.skipif(not _RUST_AVAILABLE, reason="PyO3 bizra not built")
class TestURPRustBridgeLive:
    """Integration tests using the real Rust ResourcePool."""

    def test_bridge_available(self) -> None:
        bridge = URPRustBridge()
        assert bridge.available is True

    def test_pool_stats_returns_dict(self) -> None:
        bridge = URPRustBridge()
        result = bridge.stats()
        assert result is not None
        assert "total_nodes" in result
        assert "gini_coefficient" in result
        assert "adl_compliant" in result

    def test_check_adl_returns_dict(self) -> None:
        bridge = URPRustBridge()
        result = bridge.check_adl()
        assert result is not None
        assert result["compliant"] is True
        assert result["gini_coefficient"] >= 0.0

    def test_process_zakat_returns_dict(self) -> None:
        bridge = URPRustBridge()
        result = bridge.process_zakat()
        assert result is not None
        assert "total_collected" in result

    def test_unsigned_pledge_returns_none(self, fake_pledge: FakePledge) -> None:
        """Unsigned pledge fails at Rust validation, bridge returns None."""
        bridge = URPRustBridge()
        result = bridge.submit_pledge(fake_pledge)
        # Unsigned pledge → Rust rejects → wrapper catches → None
        assert result is None

    def test_get_rewards_unknown_node(self) -> None:
        bridge = URPRustBridge()
        result = bridge.get_rewards("nonexistent-node-xyz")
        # Unknown node → Rust returns error → wrapper catches → None
        assert result is None


# -------------------------------------------------------------------------
# Test: Pledge conversion helpers
# -------------------------------------------------------------------------


class TestPledgeConversion:
    """Tests for pledge_to_rust / rust_verify_pledge helpers."""

    def test_pledge_to_rust_forced_unavailable(self) -> None:
        """With bizra mocked as ImportError, returns None."""
        from core.genesis.urp import pledge_to_rust

        pledge = FakePledge()
        with patch.dict("sys.modules", {"bizra": None}):
            result = pledge_to_rust(pledge)
            assert result is None

    @pytest.mark.skipif(not _RUST_AVAILABLE, reason="PyO3 bizra not built")
    def test_pledge_to_rust_with_pyO3(self) -> None:
        """With real PyO3, converts successfully."""
        from core.genesis.urp import pledge_to_rust

        pledge = FakePledge()
        result = pledge_to_rust(pledge)
        assert result is not None
        assert result.node_id == "test-node-001"

    def test_rust_verify_pledge_forced_unavailable(self) -> None:
        from core.genesis.urp import rust_verify_pledge

        pledge = FakePledge()
        with patch.dict("sys.modules", {"bizra": None}):
            result = rust_verify_pledge(pledge)
            assert result is None

    @pytest.mark.skipif(not _RUST_AVAILABLE, reason="PyO3 bizra not built")
    def test_rust_verify_unsigned_returns_false(self) -> None:
        """Unsigned pledge → Rust verify returns False."""
        from core.genesis.urp import rust_verify_pledge

        pledge = FakePledge(signed=False)
        result = rust_verify_pledge(pledge)
        assert result is False

    def test_pledge_to_dict_passed_correctly(self, fake_pledge: FakePledge) -> None:
        d = fake_pledge.to_dict()
        assert d["node_id"] == "test-node-001"
        assert d["ram_gb"] == 16
        assert d["vram_gb"] == 6


# -------------------------------------------------------------------------
# Test: Edge cases
# -------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for bridge robustness."""

    def test_submit_raw_dict_pledge_degraded(self) -> None:
        """Disabled bridge handles a raw dict (no to_dict method)."""
        bridge = _make_disabled_bridge()
        result = bridge.submit_pledge({"node_id": "x", "ram_gb": 8})
        assert result is None

    def test_multiple_bridge_instances_independent(self) -> None:
        b1 = _make_disabled_bridge()
        b2 = _make_disabled_bridge()
        assert b1.available is False
        assert b2.available is False
        assert b1 is not b2

    @pytest.mark.skipif(not _RUST_AVAILABLE, reason="PyO3 bizra not built")
    def test_multiple_live_bridges(self) -> None:
        b1 = URPRustBridge()
        b2 = URPRustBridge()
        assert b1.available is True
        assert b2.available is True
        assert b1 is not b2
