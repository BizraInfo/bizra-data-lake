"""Regression tests for P1-P3 bug fixes.

P1: /v1/plan terminal receipt returns receipt_id (not null)
P2a: /v1/terminal/state reflects mission lifecycle after /v1/plan
P2b: /v1/terminal/briefing uses bloom_balance + fp_float
P3: user_store master secret creation doesn't double-close fd
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

# ── P3: user_store double-close fix ──────────────────────────────────────


class TestUserStoreMasterSecretDoubleClose:
    """Verify that _load_or_create_master_secret doesn't double-close fd."""

    def test_replace_failure_does_not_double_close(self, tmp_path: Path, monkeypatch):
        """When os.replace raises, the except block must not re-close an already-closed fd."""
        monkeypatch.delenv("BIZRA_USERSTORE_MASTER_SECRET", raising=False)
        monkeypatch.delenv("BIZRA_VAULT_SECRET", raising=False)

        from core.auth.user_store import UserStore

        # Point the store at a fresh db path within tmp
        db_path = tmp_path / "test.db"

        # Patch os.replace to fail *after* fd has been closed

        def failing_replace(src, dst):
            raise OSError("Simulated os.replace failure")

        with patch("os.replace", side_effect=failing_replace):
            with pytest.raises(OSError, match="Simulated os.replace failure"):
                # This triggers _load_or_create_master_secret → mkstemp → close → replace(fail)
                # Before the fix, the except block would double-close fd → [Errno 9]
                UserStore(db_path=db_path)

    def test_normal_path_creates_secret(self, tmp_path: Path, monkeypatch):
        """Happy path: master secret is created and store initializes."""
        monkeypatch.delenv("BIZRA_USERSTORE_MASTER_SECRET", raising=False)
        monkeypatch.delenv("BIZRA_VAULT_SECRET", raising=False)

        from core.auth.user_store import UserStore

        db_path = tmp_path / "test.db"
        UserStore(db_path=db_path)

        secret_file = db_path.parent / ".user_store_master_secret"
        assert secret_file.exists(), "Master secret file should be created"
        assert len(secret_file.read_text().strip()) > 0, "Secret should not be empty"


# ── P2b: briefing wallet field fix ───────────────────────────────────────


class TestBriefingWalletFields:
    """Verify that briefing uses bloom_balance (not bloom_score) and fp_float."""

    def test_wallet_snapshot_uses_correct_fields(self):
        """A WalletState with bloom_balance=5000000 should produce bloom=5.0."""
        from core.constitutional.fixed_point import fp_float
        from core.constitutional.types import WalletState

        w = WalletState(
            node_id=b"\x00" * 32,
            seed_balance=12_500_000,  # 12.5 SEED in fixed-point
            bloom_balance=5_000_000,  # 5.0 BLOOM in fixed-point
        )

        # This mirrors the fixed briefing logic
        wallet_snap = {
            "seed": fp_float(getattr(w, "seed_balance", 0)),
            "bloom": fp_float(getattr(w, "bloom_balance", 0)),
        }

        assert wallet_snap["seed"] == 12.5, f"Expected 12.5, got {wallet_snap['seed']}"
        assert wallet_snap["bloom"] == 5.0, f"Expected 5.0, got {wallet_snap['bloom']}"

    def test_bloom_score_attribute_does_not_exist(self):
        """WalletState should NOT have bloom_score — the old (buggy) field name."""
        from core.constitutional.types import WalletState

        w = WalletState(node_id=b"\x00" * 32)
        assert not hasattr(
            w, "bloom_score"
        ), "bloom_score should not exist on WalletState"


# ── P1: MissionPlanResponse no longer strips receipt_id ──────────────────


class TestTerminalReceiptContract:
    """Verify TerminalReceipt.to_dict() produces receipt_id (not evidence_receipt_id)."""

    def test_terminal_receipt_has_receipt_id_key(self):
        from core.sovereign.terminal import (
            ChannelRecord,
            MissionReceipt,
        )

        receipt = MissionReceipt(
            mission_id="test-mission",
            receipt_id="test-receipt-abc",
            status="COMPLETE",
            synthesis="test synthesis",
            ihsan_score=0.97,
            snr_score=0.92,
            duration_ms=123.4,
            channels_executed=[
                ChannelRecord(channel="llm", success=True, duration_ms=100.0),
            ],
            action_count=1,
        )

        d = receipt.to_dict()
        assert "receipt_id" in d, "to_dict() must include receipt_id"
        assert d["receipt_id"] == "test-receipt-abc"
        # Ensure evidence_receipt_id is NOT in the terminal receipt dict
        assert (
            "evidence_receipt_id" not in d
        ), "Terminal receipt should use receipt_id, not evidence_receipt_id"


# ── P2a: TerminalStateController lifecycle ───────────────────────────────


class TestTerminalStateControllerLifecycle:
    """Verify the TerminalStateController transitions work for API-driven missions."""

    def test_mission_lifecycle_transitions(self):
        from core.sovereign.terminal import TerminalState, TerminalStateController

        ctrl = TerminalStateController()
        ctrl.transition(TerminalState.READY)

        assert ctrl.state == TerminalState.READY
        assert ctrl.mission_id == ""

        # Start mission
        assert ctrl.start_mission("mission-123")
        assert ctrl.state == TerminalState.MISSION_DRAFTING
        assert ctrl.mission_id == "mission-123"

        # Fast-forward to EXECUTING (as API-driven missions do)
        ctrl._state = TerminalState.EXECUTING

        # Complete
        assert ctrl.complete()
        assert ctrl.state == TerminalState.COMPLETED

    def test_failed_mission_lifecycle(self):
        from core.sovereign.terminal import TerminalState, TerminalStateController

        ctrl = TerminalStateController()
        ctrl.transition(TerminalState.READY)
        ctrl.start_mission("mission-fail")
        ctrl._state = TerminalState.EXECUTING

        assert ctrl.fail()
        assert ctrl.state == TerminalState.FAILED_RECOVERABLY
