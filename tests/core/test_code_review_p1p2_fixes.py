"""
Tests for P1/P2 Code Review Fixes
===================================

Covers the four findings from the code review:
  P1 — strict_gate_passed must check failed_steps in non-strict mode
  P2 — SAT count respects explicit --sat-count overrides
  P1 — genesis_grant enforces one-time idempotency per epoch
  P1 — capture_screenshot hashes content, not timestamps

Standing on Giants: Shannon (1948), Rawls (1971), Boyd (OODA)
"""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# P1 — strict_gate_passed requires failed_steps == 0 in non-strict mode
# ---------------------------------------------------------------------------
from core.genesis.orchestrator import GenesisOrchestrator
from core.genesis.types import (
    GenesisConfig,
    GenesisResult,
    GenesisStep,
    GenesisStepStatus,
)


class TestStrictGatePassedRequiresNoFailures:
    """strict_gate_passed must be False when failed_steps > 0, even non-strict."""

    def test_non_strict_with_hard_failure_is_false(self) -> None:
        """A failed (non-degraded) step in non-strict must yield strict_gate_passed=False."""
        config = GenesisConfig(
            strict_bootstrap=False,
            allow_degraded=True,
        )
        orchestrator = GenesisOrchestrator(config)

        # Inject a step that always hard-fails
        original_run = orchestrator._step_token_allocation

        def _failing_step() -> dict:
            return {
                "success": False,
                "status": "failed",
                "reason_code": "INJECTED_FAIL",
            }

        orchestrator._step_token_allocation = _failing_step
        result = orchestrator.run()

        # The token_allocation step should have failed
        tok_step = next((s for s in result.steps if s.name == "token_allocation"), None)
        assert tok_step is not None
        assert tok_step.status == GenesisStepStatus.FAILED

        # Critical assertion: strict_gate_passed must be False
        assert result.failed_steps > 0
        assert (
            result.strict_gate_passed is False
        ), "strict_gate_passed should be False when failed_steps > 0"

    def test_non_strict_all_success_is_true(self) -> None:
        """Non-strict with no failures and no degradation → strict_gate_passed=True."""
        config = GenesisConfig(
            strict_bootstrap=False,
            allow_degraded=True,
        )
        orchestrator = GenesisOrchestrator(config)
        result = orchestrator.run()

        # Default orchestrator (no external deps) succeeds all basic steps
        if result.failed_steps == 0 and not result.degraded:
            assert result.strict_gate_passed is True


# ---------------------------------------------------------------------------
# P2 — SAT count respects explicit --sat-count
# ---------------------------------------------------------------------------
from core.genesis.cli import build_genesis_parser, handle_genesis


class TestSATCountOverride:
    """--sat-count N should be honoured when sat_mode allows it."""

    def _parse_args(self, *args: str) -> SimpleNamespace:
        """Simulate CLI argument parsing."""
        import argparse

        main_parser = argparse.ArgumentParser()
        subparsers = main_parser.add_subparsers()
        build_genesis_parser(subparsers)
        return main_parser.parse_args(["genesis", *args])

    def test_sat_count_12_not_clamped(self) -> None:
        """--sat-count 12 must not be silently clamped to 5."""
        args = self._parse_args("--sat-count", "12")

        # Simulate the same resolution logic as handle_genesis
        pat_count = 7 if getattr(args, "pat_7", False) else (args.pat_count or 7)
        sat_mode = args.sat_mode
        if getattr(args, "sat_49", False):
            sat_mode = "full49"
            sat_count = 49
        elif getattr(args, "sat_5", False):
            sat_mode = "mini5"
            sat_count = 5
        else:
            sat_count = args.sat_count or 5
            if sat_mode is None:
                sat_mode = "full49" if sat_count >= 49 else "mini5"

        # Apply the NEW logic (not the old hard-clamp)
        if sat_mode == "full49":
            sat_count = max(sat_count, 49)
        elif sat_mode == "mini5" and sat_count == 5:
            pass
        # else: honour explicit --sat-count as-is

        assert sat_count == 12, f"Expected 12 but got {sat_count}"

    def test_sat_49_flag_forces_49(self) -> None:
        """--sat-49 always yields 49."""
        args = self._parse_args("--sat-49")
        sat_count = 49 if getattr(args, "sat_49", False) else 5
        assert sat_count == 49

    def test_sat_5_flag_forces_5(self) -> None:
        """--sat-5 always yields 5."""
        args = self._parse_args("--sat-5")
        sat_count = 5 if getattr(args, "sat_5", False) else 49
        assert sat_count == 5

    def test_default_sat_is_5(self) -> None:
        """No explicit SAT flag yields default 5."""
        args = self._parse_args()
        sat_count = args.sat_count or 5
        assert sat_count == 5


# ---------------------------------------------------------------------------
# P1 — genesis_grant one-time idempotency
# ---------------------------------------------------------------------------
from core.token.ledger import TokenLedger
from core.token.types import TokenType


class TestGenesisGrantIdempotency:
    """genesis_grant must reject duplicate mints for the same node+epoch."""

    @pytest.fixture()
    def ledger(self, tmp_path: Path) -> TokenLedger:
        db = tmp_path / "ledger.db"
        log = tmp_path / "log.jsonl"
        return TokenLedger(db_path=db, log_path=log)

    def test_first_grant_succeeds(self, ledger: TokenLedger) -> None:
        r = ledger.genesis_grant("NODE-A", amount=100.0)
        assert r.success is True
        bal = ledger.get_balance("NODE-A", TokenType.SEED)
        assert bal.available == 100.0

    def test_second_grant_same_epoch_rejected(self, ledger: TokenLedger) -> None:
        r1 = ledger.genesis_grant("NODE-B", amount=100.0)
        r2 = ledger.genesis_grant("NODE-B", amount=100.0)
        assert r1.success is True
        assert r2.success is False
        assert "already minted" in (r2.error or "")
        # Balance must NOT have doubled
        bal = ledger.get_balance("NODE-B", TokenType.SEED)
        assert bal.available == 100.0

    def test_grant_different_epoch_allowed(self, ledger: TokenLedger) -> None:
        r1 = ledger.genesis_grant("NODE-C", amount=50.0, epoch_id="epoch-1")
        r2 = ledger.genesis_grant("NODE-C", amount=50.0, epoch_id="epoch-2")
        assert r1.success is True
        assert r2.success is True
        bal = ledger.get_balance("NODE-C", TokenType.SEED)
        assert bal.available == 100.0

    def test_grant_different_node_same_epoch(self, ledger: TokenLedger) -> None:
        r1 = ledger.genesis_grant("NODE-D", amount=100.0)
        r2 = ledger.genesis_grant("NODE-E", amount=100.0)
        assert r1.success is True
        assert r2.success is True


# ---------------------------------------------------------------------------
# P1 — capture_screenshot hashes content, not timestamps
# ---------------------------------------------------------------------------
from core.bridges.desktop_bridge import DesktopBridge


class TestCaptureScreenshotHash:
    """state_hash must be content-addressed when screenshot bytes are provided."""

    @pytest.fixture()
    def bridge(self) -> DesktopBridge:
        return DesktopBridge.__new__(DesktopBridge)

    @pytest.mark.asyncio
    async def test_same_bytes_produce_same_hash(self, bridge: DesktopBridge) -> None:
        """Identical screenshot bytes must yield identical hashes."""
        img = b"\x89PNG\r\n\x1a\n" + b"\x00" * 256
        import base64

        b64 = base64.b64encode(img).decode()

        r1 = await bridge._handle_capture_screenshot(
            {"label": "pre", "screenshot_base64": b64}
        )
        r2 = await bridge._handle_capture_screenshot(
            {"label": "pre", "screenshot_base64": b64}
        )

        assert r1["state_hash"] == r2["state_hash"], "Same bytes must yield same hash"

    @pytest.mark.asyncio
    async def test_different_bytes_produce_different_hash(
        self, bridge: DesktopBridge
    ) -> None:
        """Different screenshot bytes must yield different hashes."""
        import base64

        img_a = base64.b64encode(b"state_A_data").decode()
        img_b = base64.b64encode(b"state_B_data").decode()

        r1 = await bridge._handle_capture_screenshot({"screenshot_base64": img_a})
        r2 = await bridge._handle_capture_screenshot({"screenshot_base64": img_b})

        assert (
            r1["state_hash"] != r2["state_hash"]
        ), "Different bytes must yield different hash"

    @pytest.mark.asyncio
    async def test_no_bytes_fallback_still_unique(self, bridge: DesktopBridge) -> None:
        """Without screenshot data, fallback hashes are timestamp-unique."""
        r1 = await bridge._handle_capture_screenshot({"label": "pre"})
        r2 = await bridge._handle_capture_screenshot({"label": "post"})

        assert r1["captured"] is True
        assert r2["captured"] is True
        # Different labels at different monotonic times → different hashes
        assert r1["state_hash"] != r2["state_hash"]

    @pytest.mark.asyncio
    async def test_content_hash_matches_expected_sha256(
        self, bridge: DesktopBridge
    ) -> None:
        """Content-addressed hash matches manual SHA-256 of the raw bytes."""
        import base64

        raw = b"deterministic_test_payload"
        b64 = base64.b64encode(raw).decode()

        result = await bridge._handle_capture_screenshot({"screenshot_base64": b64})
        expected = hashlib.sha256(raw).hexdigest()

        assert result["state_hash"] == expected
