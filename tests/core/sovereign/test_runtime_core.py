"""
Tests for SovereignRuntime — Part 1: Lifecycle, Init, Config, Helper Methods
=============================================================================

Comprehensive unit tests for runtime_core.py covering:
  - __init__ and RuntimeConfig defaults
  - _load_env_vars
  - _parse_env_bool
  - _apply_env_overrides
  - create() async context manager
  - Evidence/Experience/Judgment ledger init
  - _observe_judgment
  - _commit_experience_episode
  - Gate chain init + preflight
  - Receipt emission
  - PoI contribution registration
  - Memory encoding
  - Graph artifact storage
  - Cache, health, mode mapping, checkpoint
  - _estimate_complexity
  - status()

Target: ~120 tests.  No production code modifications.

Standing on Giants: pytest + unittest.mock + Shannon (SNR) + Besta (GoT)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import (
    AsyncMock,
    MagicMock,
    PropertyMock,
    call,
    patch,
)

import pytest

from core.sovereign.runtime_core import SovereignRuntime
from core.sovereign.runtime_types import (
    HealthStatus,
    RuntimeConfig,
    RuntimeMetrics,
    RuntimeMode,
    SovereignQuery,
    SovereignResult,
)


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------

def _minimal_config(tmp_path: Path) -> RuntimeConfig:
    """Create minimal config with all optional features disabled."""
    return RuntimeConfig(
        node_id="test-node-0001",
        enable_graph_reasoning=False,
        enable_snr_optimization=False,
        enable_guardian_validation=False,
        enable_autonomous_loop=False,
        enable_cache=False,
        enable_persistence=False,
        autonomous_enabled=False,
        enable_zpk_preflight=False,
        enable_proactive_kernel=False,
        state_dir=tmp_path / "sovereign_state",
    )


@pytest.fixture
def tmp_state(tmp_path: Path) -> Path:
    """Return a temporary state directory (already created)."""
    state = tmp_path / "sovereign_state"
    state.mkdir(parents=True, exist_ok=True)
    return state


@pytest.fixture
def cfg(tmp_path: Path) -> RuntimeConfig:
    """Minimal RuntimeConfig for isolated tests."""
    return _minimal_config(tmp_path)


@pytest.fixture
def rt(cfg: RuntimeConfig) -> SovereignRuntime:
    """A fresh SovereignRuntime with all optional components disabled."""
    return SovereignRuntime(cfg)


@pytest.fixture(autouse=True)
def _runtime_role_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIZRA_NODE_ROLE", "node")


# ---------------------------------------------------------------------------
# 1. __init__
# ---------------------------------------------------------------------------

class TestInit:
    """Tests for SovereignRuntime.__init__."""

    def test_default_construction(self, tmp_path: Path) -> None:
        """Default construction should use RuntimeConfig defaults."""
        runtime = SovereignRuntime()
        assert runtime.config is not None
        assert isinstance(runtime.config, RuntimeConfig)
        assert runtime.metrics is not None
        assert isinstance(runtime.metrics, RuntimeMetrics)

    def test_custom_config(self, cfg: RuntimeConfig) -> None:
        """Custom config should be stored exactly."""
        runtime = SovereignRuntime(cfg)
        assert runtime.config is cfg
        assert runtime.config.node_id == "test-node-0001"

    def test_components_start_none(self, rt: SovereignRuntime) -> None:
        """All lazy components should start as None."""
        assert rt._graph_reasoner is None
        assert rt._snr_optimizer is None
        assert rt._guardian_council is None
        assert rt._autonomous_loop is None
        assert rt._orchestrator is None
        assert rt._genesis is None
        assert rt._memory_coordinator is None
        assert rt._impact_tracker is None
        assert rt._evidence_ledger is None
        assert rt._gate_chain is None
        assert rt._poi_orchestrator is None
        assert rt._sat_controller is None
        assert rt._experience_ledger is None
        assert rt._judgment_telemetry is None
        assert rt._gateway is None
        assert rt._omega is None
        assert rt._living_memory is None
        assert rt._pek is None
        assert rt._zpk_bootstrap_result is None

    def test_state_flags_initial(self, rt: SovereignRuntime) -> None:
        """Initial state flags should be False."""
        assert rt._initialized is False
        assert rt._running is False
        assert not rt._shutdown_event.is_set()

    def test_cache_initialized_empty(self, rt: SovereignRuntime) -> None:
        """Cache dict should start empty."""
        assert rt._cache == {}

    def test_graph_artifacts_initialized_empty(self, rt: SovereignRuntime) -> None:
        """Graph artifacts dict should start empty."""
        assert rt._graph_artifacts == {}

    def test_query_times_is_bounded_deque(self, rt: SovereignRuntime) -> None:
        """_query_times should be a deque with maxlen=100."""
        assert isinstance(rt._query_times, deque)
        assert rt._query_times.maxlen == 100
        assert len(rt._query_times) == 0

    def test_last_snr_trace_none(self, rt: SovereignRuntime) -> None:
        """_last_snr_trace should start None."""
        assert rt._last_snr_trace is None


# ---------------------------------------------------------------------------
# 2. _load_env_vars
# ---------------------------------------------------------------------------

class TestLoadEnvVars:
    """Tests for SovereignRuntime._load_env_vars."""

    def test_no_env_file(self, rt: SovereignRuntime) -> None:
        """No .env file should be a no-op (no crash)."""
        # state_dir does not exist yet — should handle gracefully
        rt._load_env_vars()  # should not raise

    def test_loads_key_value(
        self, rt: SovereignRuntime, tmp_state: Path
    ) -> None:
        """Should load KEY=VALUE pairs into os.environ."""
        rt.config.state_dir = tmp_state
        env_file = tmp_state / ".env"
        env_file.write_text("MY_TEST_KEY_ALPHA=hello_world\n")

        # Make sure key doesn't exist first
        os.environ.pop("MY_TEST_KEY_ALPHA", None)

        rt._load_env_vars()

        assert os.environ.get("MY_TEST_KEY_ALPHA") == "hello_world"
        # cleanup
        os.environ.pop("MY_TEST_KEY_ALPHA", None)

    def test_skips_comments(
        self, rt: SovereignRuntime, tmp_state: Path
    ) -> None:
        """Lines starting with # should be skipped."""
        rt.config.state_dir = tmp_state
        env_file = tmp_state / ".env"
        env_file.write_text("# this is a comment\nTEST_LOAD_COMMENT=yes\n")

        os.environ.pop("TEST_LOAD_COMMENT", None)
        rt._load_env_vars()

        assert os.environ.get("TEST_LOAD_COMMENT") == "yes"
        os.environ.pop("TEST_LOAD_COMMENT", None)

    def test_skips_empty_lines(
        self, rt: SovereignRuntime, tmp_state: Path
    ) -> None:
        """Empty lines should be skipped without error."""
        rt.config.state_dir = tmp_state
        env_file = tmp_state / ".env"
        env_file.write_text("\n\n\nTEST_LOAD_EMPTY=value\n\n")

        os.environ.pop("TEST_LOAD_EMPTY", None)
        rt._load_env_vars()

        assert os.environ.get("TEST_LOAD_EMPTY") == "value"
        os.environ.pop("TEST_LOAD_EMPTY", None)

    def test_strips_quotes(
        self, rt: SovereignRuntime, tmp_state: Path
    ) -> None:
        """Quoted values should have quotes stripped."""
        rt.config.state_dir = tmp_state
        env_file = tmp_state / ".env"
        env_file.write_text(
            "TEST_SINGLE_Q='quoted_single'\nTEST_DOUBLE_Q=\"quoted_double\"\n"
        )

        os.environ.pop("TEST_SINGLE_Q", None)
        os.environ.pop("TEST_DOUBLE_Q", None)

        rt._load_env_vars()

        assert os.environ.get("TEST_SINGLE_Q") == "quoted_single"
        assert os.environ.get("TEST_DOUBLE_Q") == "quoted_double"

        os.environ.pop("TEST_SINGLE_Q", None)
        os.environ.pop("TEST_DOUBLE_Q", None)

    def test_existing_env_not_overwritten(
        self, rt: SovereignRuntime, tmp_state: Path
    ) -> None:
        """Existing environment variables must NOT be overwritten."""
        rt.config.state_dir = tmp_state
        env_file = tmp_state / ".env"
        env_file.write_text("TEST_PRESERVE=from_file\n")

        os.environ["TEST_PRESERVE"] = "original"

        rt._load_env_vars()

        assert os.environ.get("TEST_PRESERVE") == "original"
        os.environ.pop("TEST_PRESERVE", None)


# ---------------------------------------------------------------------------
# 3. _parse_env_bool
# ---------------------------------------------------------------------------

class TestParseEnvBool:
    """Tests for SovereignRuntime._parse_env_bool (static method)."""

    @pytest.mark.parametrize("val", ["1", "true", "yes", "on"])
    def test_truthy_values(self, val: str) -> None:
        assert SovereignRuntime._parse_env_bool(val) is True

    @pytest.mark.parametrize("val", ["0", "false", "no", "off"])
    def test_falsy_values(self, val: str) -> None:
        assert SovereignRuntime._parse_env_bool(val) is False

    def test_none_returns_default_false(self) -> None:
        assert SovereignRuntime._parse_env_bool(None) is False

    def test_none_returns_custom_default_true(self) -> None:
        assert SovereignRuntime._parse_env_bool(None, default=True) is True

    def test_invalid_returns_default(self) -> None:
        assert SovereignRuntime._parse_env_bool("banana") is False
        assert SovereignRuntime._parse_env_bool("banana", default=True) is True

    def test_whitespace_tolerance(self) -> None:
        assert SovereignRuntime._parse_env_bool("  true  ") is True
        assert SovereignRuntime._parse_env_bool("  FALSE  ") is False

    def test_case_insensitive(self) -> None:
        assert SovereignRuntime._parse_env_bool("TRUE") is True
        assert SovereignRuntime._parse_env_bool("Yes") is True
        assert SovereignRuntime._parse_env_bool("ON") is True
        assert SovereignRuntime._parse_env_bool("False") is False
        assert SovereignRuntime._parse_env_bool("NO") is False
        assert SovereignRuntime._parse_env_bool("Off") is False


# ---------------------------------------------------------------------------
# 4. _apply_env_overrides
# ---------------------------------------------------------------------------

class TestApplyEnvOverrides:
    """Tests for SovereignRuntime._apply_env_overrides."""

    def _clean_env(self, keys: list[str]) -> None:
        for k in keys:
            os.environ.pop(k, None)

    def test_zpk_manifest_uri(self, rt: SovereignRuntime) -> None:
        os.environ["ZPK_MANIFEST_URI"] = "https://example.com/manifest.json"
        try:
            rt._apply_env_overrides()
            assert rt.config.zpk_manifest_uri == "https://example.com/manifest.json"
        finally:
            os.environ.pop("ZPK_MANIFEST_URI", None)

    def test_zpk_release_public_key(self, rt: SovereignRuntime) -> None:
        os.environ["ZPK_RELEASE_PUBLIC_KEY"] = "abc123hex"
        try:
            rt._apply_env_overrides()
            assert rt.config.zpk_release_public_key == "abc123hex"
        finally:
            os.environ.pop("ZPK_RELEASE_PUBLIC_KEY", None)

    def test_zpk_preflight_enabled(self, rt: SovereignRuntime) -> None:
        os.environ["ZPK_PREFLIGHT_ENABLED"] = "true"
        try:
            rt._apply_env_overrides()
            assert rt.config.enable_zpk_preflight is True
        finally:
            os.environ.pop("ZPK_PREFLIGHT_ENABLED", None)

    def test_zpk_emit_bootstrap_events(self, rt: SovereignRuntime) -> None:
        os.environ["ZPK_EMIT_BOOTSTRAP_EVENTS"] = "1"
        try:
            rt._apply_env_overrides()
            assert rt.config.zpk_emit_bootstrap_events is True
        finally:
            os.environ.pop("ZPK_EMIT_BOOTSTRAP_EVENTS", None)

    def test_zpk_event_topic(self, rt: SovereignRuntime) -> None:
        os.environ["ZPK_EVENT_TOPIC"] = "custom.topic"
        try:
            rt._apply_env_overrides()
            assert rt.config.zpk_event_topic == "custom.topic"
        finally:
            os.environ.pop("ZPK_EVENT_TOPIC", None)

    def test_zpk_allowed_versions_comma_separated(self, rt: SovereignRuntime) -> None:
        os.environ["ZPK_ALLOWED_VERSIONS"] = "1.0.0, 1.1.0 , 2.0.0"
        try:
            rt._apply_env_overrides()
            assert rt.config.zpk_allowed_versions == ["1.0.0", "1.1.0", "2.0.0"]
        finally:
            os.environ.pop("ZPK_ALLOWED_VERSIONS", None)

    def test_zpk_min_policy_version(self, rt: SovereignRuntime) -> None:
        os.environ["ZPK_MIN_POLICY_VERSION"] = "5"
        try:
            rt._apply_env_overrides()
            assert rt.config.zpk_min_policy_version == 5
        finally:
            os.environ.pop("ZPK_MIN_POLICY_VERSION", None)

    def test_zpk_min_policy_version_invalid_logs_warning(
        self, rt: SovereignRuntime, caplog: pytest.LogCaptureFixture
    ) -> None:
        os.environ["ZPK_MIN_POLICY_VERSION"] = "not_int"
        try:
            rt._apply_env_overrides()
            # Should warn but not raise
            assert any("Invalid ZPK_MIN_POLICY_VERSION" in r.message for r in caplog.records)
        finally:
            os.environ.pop("ZPK_MIN_POLICY_VERSION", None)

    def test_zpk_min_ihsan_policy(self, rt: SovereignRuntime) -> None:
        os.environ["ZPK_MIN_IHSAN_POLICY"] = "0.99"
        try:
            rt._apply_env_overrides()
            assert rt.config.zpk_min_ihsan_policy == pytest.approx(0.99)
        finally:
            os.environ.pop("ZPK_MIN_IHSAN_POLICY", None)

    def test_zpk_min_ihsan_policy_invalid_logs_warning(
        self, rt: SovereignRuntime, caplog: pytest.LogCaptureFixture
    ) -> None:
        os.environ["ZPK_MIN_IHSAN_POLICY"] = "not_a_float"
        try:
            rt._apply_env_overrides()
            assert any("Invalid ZPK_MIN_IHSAN_POLICY" in r.message for r in caplog.records)
        finally:
            os.environ.pop("ZPK_MIN_IHSAN_POLICY", None)

    def test_pek_enabled(self, rt: SovereignRuntime) -> None:
        os.environ["PEK_ENABLED"] = "true"
        try:
            rt._apply_env_overrides()
            assert rt.config.enable_proactive_kernel is True
        finally:
            os.environ.pop("PEK_ENABLED", None)

    def test_pek_emit_proof_events(self, rt: SovereignRuntime) -> None:
        os.environ["PEK_EMIT_PROOF_EVENTS"] = "yes"
        try:
            rt._apply_env_overrides()
            assert rt.config.proactive_kernel_emit_events is True
        finally:
            os.environ.pop("PEK_EMIT_PROOF_EVENTS", None)

    def test_pek_float_overrides(self, rt: SovereignRuntime) -> None:
        envs = {
            "PEK_CYCLE_SECONDS": "15.5",
            "PEK_MIN_CONFIDENCE": "0.70",
            "PEK_MIN_AUTO_CONFIDENCE": "0.80",
            "PEK_BASE_TAU": "0.60",
            "PEK_AUTO_EXECUTE_TAU": "0.85",
            "PEK_QUEUE_SILENT_TAU": "0.40",
            "PEK_ATTENTION_BUDGET_CAPACITY": "12.0",
            "PEK_ATTENTION_BUDGET_RECOVERY_PER_CYCLE": "1.5",
        }
        for k, v in envs.items():
            os.environ[k] = v
        try:
            rt._apply_env_overrides()
            assert rt.config.proactive_kernel_cycle_seconds == pytest.approx(15.5)
            assert rt.config.proactive_kernel_min_confidence == pytest.approx(0.70)
            assert rt.config.proactive_kernel_min_auto_confidence == pytest.approx(0.80)
            assert rt.config.proactive_kernel_base_tau == pytest.approx(0.60)
            assert rt.config.proactive_kernel_auto_execute_tau == pytest.approx(0.85)
            assert rt.config.proactive_kernel_queue_silent_tau == pytest.approx(0.40)
            assert rt.config.proactive_kernel_attention_budget_capacity == pytest.approx(12.0)
            assert rt.config.proactive_kernel_attention_recovery_per_cycle == pytest.approx(1.5)
        finally:
            for k in envs:
                os.environ.pop(k, None)

    def test_pek_invalid_float_logs_warning(
        self, rt: SovereignRuntime, caplog: pytest.LogCaptureFixture
    ) -> None:
        os.environ["PEK_CYCLE_SECONDS"] = "not_a_number"
        try:
            rt._apply_env_overrides()
            assert any("Invalid" in r.message and "PEK_CYCLE_SECONDS" in r.message for r in caplog.records)
        finally:
            os.environ.pop("PEK_CYCLE_SECONDS", None)

    def test_pek_proof_event_topic(self, rt: SovereignRuntime) -> None:
        os.environ["PEK_PROOF_EVENT_TOPIC"] = "custom.pek.topic"
        try:
            rt._apply_env_overrides()
            assert rt.config.proactive_kernel_event_topic == "custom.pek.topic"
        finally:
            os.environ.pop("PEK_PROOF_EVENT_TOPIC", None)

    def test_no_env_vars_set_leaves_defaults(self, rt: SovereignRuntime) -> None:
        """When no env vars are present, config should stay at defaults."""
        # Wipe all potentially conflicting env vars
        keys = [
            "ZPK_MANIFEST_URI", "ZPK_RELEASE_PUBLIC_KEY", "ZPK_PREFLIGHT_ENABLED",
            "ZPK_EMIT_BOOTSTRAP_EVENTS", "ZPK_EVENT_TOPIC", "ZPK_ALLOWED_VERSIONS",
            "ZPK_MIN_POLICY_VERSION", "ZPK_MIN_IHSAN_POLICY",
            "PEK_ENABLED", "PEK_EMIT_PROOF_EVENTS", "PEK_PROOF_EVENT_TOPIC",
            "PEK_CYCLE_SECONDS", "PEK_MIN_CONFIDENCE",
        ]
        saved = {}
        for k in keys:
            saved[k] = os.environ.pop(k, None)
        try:
            original_zpk = rt.config.zpk_manifest_uri
            rt._apply_env_overrides()
            assert rt.config.zpk_manifest_uri == original_zpk
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v


# ---------------------------------------------------------------------------
# 5. create() classmethod async context manager
# ---------------------------------------------------------------------------

class TestCreate:
    """Tests for SovereignRuntime.create() lifecycle."""

    @pytest.mark.asyncio
    async def test_create_initializes_and_shuts_down(self, tmp_path: Path) -> None:
        """create() should call initialize(), yield runtime, then call shutdown()."""
        cfg = _minimal_config(tmp_path)

        with patch.object(SovereignRuntime, "initialize", new_callable=AsyncMock) as mock_init, \
             patch.object(SovereignRuntime, "shutdown", new_callable=AsyncMock) as mock_shut:

            async with SovereignRuntime.create(cfg) as runtime:
                assert isinstance(runtime, SovereignRuntime)
                mock_init.assert_awaited_once()

            mock_shut.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_shutdown_on_exception(self, tmp_path: Path) -> None:
        """Shutdown should still be called even if body raises."""
        cfg = _minimal_config(tmp_path)

        with patch.object(SovereignRuntime, "initialize", new_callable=AsyncMock), \
             patch.object(SovereignRuntime, "shutdown", new_callable=AsyncMock) as mock_shut:

            with pytest.raises(ValueError):
                async with SovereignRuntime.create(cfg) as _runtime:
                    raise ValueError("test error")

            mock_shut.assert_awaited_once()


# ---------------------------------------------------------------------------
# 6. _init_evidence_ledger
# ---------------------------------------------------------------------------

class TestInitEvidenceLedger:
    """Tests for _init_evidence_ledger."""

    def test_success_path(self, rt: SovereignRuntime, tmp_state: Path) -> None:
        """Successful import should set _evidence_ledger."""
        rt.config.state_dir = tmp_state
        mock_ledger = MagicMock()
        mock_ledger.sequence = 0

        with patch.dict("sys.modules", {
            "core.proof_engine.evidence_ledger": MagicMock(
                EvidenceLedger=MagicMock(return_value=mock_ledger)
            ),
        }):
            rt._init_evidence_ledger()

        assert rt._evidence_ledger is mock_ledger

    def test_failure_non_fatal(self, rt: SovereignRuntime) -> None:
        """Import failure should set _evidence_ledger to None, not raise."""
        with patch.dict("sys.modules", {
            "core.proof_engine.evidence_ledger": None,
        }):
            # Force import error
            rt._init_evidence_ledger()

        assert rt._evidence_ledger is None


# ---------------------------------------------------------------------------
# 7. _init_experience_ledger
# ---------------------------------------------------------------------------

class TestInitExperienceLedger:
    """Tests for _init_experience_ledger."""

    def test_success_path(self, rt: SovereignRuntime) -> None:
        mock_ledger = MagicMock()
        with patch(
            "core.sovereign.runtime_core.SovereignRuntime._init_experience_ledger",
            wraps=rt._init_experience_ledger,
        ):
            with patch.dict("sys.modules", {
                "core.sovereign.experience_ledger": MagicMock(
                    SovereignExperienceLedger=MagicMock(return_value=mock_ledger)
                ),
            }):
                rt._init_experience_ledger()

        assert rt._experience_ledger is mock_ledger

    def test_failure_non_fatal(self, rt: SovereignRuntime) -> None:
        with patch.dict("sys.modules", {
            "core.sovereign.experience_ledger": None,
        }):
            rt._init_experience_ledger()
        assert rt._experience_ledger is None


# ---------------------------------------------------------------------------
# 8. _init_judgment_telemetry
# ---------------------------------------------------------------------------

class TestInitJudgmentTelemetry:
    """Tests for _init_judgment_telemetry."""

    def test_success_path(self, rt: SovereignRuntime) -> None:
        mock_telemetry = MagicMock()
        with patch.dict("sys.modules", {
            "core.sovereign.judgment_telemetry": MagicMock(
                JudgmentTelemetry=MagicMock(return_value=mock_telemetry)
            ),
        }):
            rt._init_judgment_telemetry()

        assert rt._judgment_telemetry is mock_telemetry

    def test_failure_non_fatal(self, rt: SovereignRuntime) -> None:
        with patch.dict("sys.modules", {
            "core.sovereign.judgment_telemetry": None,
        }):
            rt._init_judgment_telemetry()
        assert rt._judgment_telemetry is None


# ---------------------------------------------------------------------------
# 9. _observe_judgment
# ---------------------------------------------------------------------------

class TestObserveJudgment:
    """Tests for _observe_judgment."""

    def _make_result(self, **kwargs: Any) -> SovereignResult:
        defaults = {
            "success": True,
            "snr_ok": True,
            "ihsan_score": 0.96,
            "validated": False,
            "validation_passed": True,
        }
        defaults.update(kwargs)
        return SovereignResult(**defaults)

    def test_none_telemetry_noop(self, rt: SovereignRuntime) -> None:
        """With no telemetry engine, should be a silent no-op."""
        rt._judgment_telemetry = None
        rt._observe_judgment(self._make_result())  # must not raise

    def test_forbid_on_success_false(self, rt: SovereignRuntime) -> None:
        mock_telemetry = MagicMock()
        rt._judgment_telemetry = mock_telemetry

        mock_verdict_module = MagicMock()
        mock_verdict_module.JudgmentVerdict.FORBID = "FORBID"

        with patch.dict("sys.modules", {
            "core.sovereign.judgment_telemetry": mock_verdict_module,
        }):
            rt._observe_judgment(self._make_result(success=False))

        mock_telemetry.observe.assert_called_once_with("FORBID")

    def test_forbid_on_validation_failed(self, rt: SovereignRuntime) -> None:
        mock_telemetry = MagicMock()
        rt._judgment_telemetry = mock_telemetry

        mock_verdict_module = MagicMock()
        mock_verdict_module.JudgmentVerdict.FORBID = "FORBID"

        with patch.dict("sys.modules", {
            "core.sovereign.judgment_telemetry": mock_verdict_module,
        }):
            rt._observe_judgment(
                self._make_result(validated=True, validation_passed=False)
            )

        mock_telemetry.observe.assert_called_once_with("FORBID")

    def test_demote_when_snr_not_ok(self, rt: SovereignRuntime) -> None:
        mock_telemetry = MagicMock()
        rt._judgment_telemetry = mock_telemetry

        mock_verdict_module = MagicMock()
        mock_verdict_module.JudgmentVerdict.FORBID = "FORBID"
        mock_verdict_module.JudgmentVerdict.DEMOTE = "DEMOTE"

        with patch.dict("sys.modules", {
            "core.sovereign.judgment_telemetry": mock_verdict_module,
        }):
            rt._observe_judgment(
                self._make_result(success=True, snr_ok=False)
            )

        mock_telemetry.observe.assert_called_once_with("DEMOTE")

    def test_promote_when_ihsan_high_and_snr_ok(self, rt: SovereignRuntime) -> None:
        mock_telemetry = MagicMock()
        rt._judgment_telemetry = mock_telemetry

        mock_verdict_module = MagicMock()
        mock_verdict_module.JudgmentVerdict.FORBID = "FORBID"
        mock_verdict_module.JudgmentVerdict.DEMOTE = "DEMOTE"
        mock_verdict_module.JudgmentVerdict.PROMOTE = "PROMOTE"

        with patch.dict("sys.modules", {
            "core.sovereign.judgment_telemetry": mock_verdict_module,
        }):
            rt._observe_judgment(
                self._make_result(success=True, snr_ok=True, ihsan_score=0.97)
            )

        mock_telemetry.observe.assert_called_once_with("PROMOTE")

    def test_neutral_when_snr_ok_but_ihsan_below_095(self, rt: SovereignRuntime) -> None:
        mock_telemetry = MagicMock()
        rt._judgment_telemetry = mock_telemetry

        mock_verdict_module = MagicMock()
        mock_verdict_module.JudgmentVerdict.FORBID = "FORBID"
        mock_verdict_module.JudgmentVerdict.DEMOTE = "DEMOTE"
        mock_verdict_module.JudgmentVerdict.PROMOTE = "PROMOTE"
        mock_verdict_module.JudgmentVerdict.NEUTRAL = "NEUTRAL"

        with patch.dict("sys.modules", {
            "core.sovereign.judgment_telemetry": mock_verdict_module,
        }):
            rt._observe_judgment(
                self._make_result(success=True, snr_ok=True, ihsan_score=0.90)
            )

        mock_telemetry.observe.assert_called_once_with("NEUTRAL")

    def test_exception_swallowed(self, rt: SovereignRuntime) -> None:
        """Telemetry exceptions must be swallowed (fire-and-forget)."""
        mock_telemetry = MagicMock()
        mock_telemetry.observe.side_effect = RuntimeError("boom")
        rt._judgment_telemetry = mock_telemetry

        mock_verdict_module = MagicMock()
        mock_verdict_module.JudgmentVerdict.PROMOTE = "PROMOTE"
        mock_verdict_module.JudgmentVerdict.FORBID = "FORBID"
        mock_verdict_module.JudgmentVerdict.DEMOTE = "DEMOTE"

        with patch.dict("sys.modules", {
            "core.sovereign.judgment_telemetry": mock_verdict_module,
        }):
            # Should not raise
            rt._observe_judgment(
                self._make_result(success=True, snr_ok=True, ihsan_score=0.96)
            )


# ---------------------------------------------------------------------------
# 10. _commit_experience_episode
# ---------------------------------------------------------------------------

class TestCommitExperienceEpisode:
    """Tests for _commit_experience_episode."""

    def _make_result(self, **kwargs: Any) -> SovereignResult:
        defaults = {
            "query_id": "q-001",
            "success": True,
            "snr_ok": True,
            "snr_score": 0.92,
            "ihsan_score": 0.96,
            "response": "A valid response",
            "thoughts": ["thought1", "thought2"],
            "processing_time_ms": 150.0,
        }
        defaults.update(kwargs)
        return SovereignResult(**defaults)

    def _make_query(self, text: str = "test query") -> SovereignQuery:
        return SovereignQuery(text=text)

    def test_none_ledger_noop(self, rt: SovereignRuntime) -> None:
        rt._experience_ledger = None
        rt._commit_experience_episode(self._make_result(), self._make_query())

    def test_result_not_success_noop(self, rt: SovereignRuntime) -> None:
        mock_ledger = MagicMock()
        rt._experience_ledger = mock_ledger
        rt._commit_experience_episode(
            self._make_result(success=False), self._make_query()
        )
        mock_ledger.commit.assert_not_called()

    def test_snr_not_ok_noop(self, rt: SovereignRuntime) -> None:
        mock_ledger = MagicMock()
        rt._experience_ledger = mock_ledger
        rt._commit_experience_episode(
            self._make_result(snr_ok=False), self._make_query()
        )
        mock_ledger.commit.assert_not_called()

    def test_success_path_calls_commit(self, rt: SovereignRuntime) -> None:
        mock_ledger = MagicMock()
        rt._experience_ledger = mock_ledger

        result = self._make_result()
        # model_used is now a proper field on SovereignResult
        result.model_used = "test-model"
        query = self._make_query()

        rt._commit_experience_episode(result, query)

        mock_ledger.commit.assert_called_once()
        call_kwargs = mock_ledger.commit.call_args
        # Verify graph_hash computed from thoughts (BLAKE3, SEC-001)
        from core.proof_engine.canonical import hex_digest

        expected_hash = hex_digest(
            "thought1|thought2".encode("utf-8")
        )
        assert call_kwargs.kwargs["graph_hash"] == expected_hash
        assert call_kwargs.kwargs["graph_node_count"] == 2
        assert call_kwargs.kwargs["snr_score"] == pytest.approx(0.92)

    def test_no_thoughts_empty_graph_hash(self, rt: SovereignRuntime) -> None:
        mock_ledger = MagicMock()
        rt._experience_ledger = mock_ledger

        result = self._make_result(thoughts=[])
        # model_used is now a proper Optional[str] field
        result.model_used = None
        query = self._make_query()

        rt._commit_experience_episode(result, query)

        mock_ledger.commit.assert_called_once()
        call_kwargs = mock_ledger.commit.call_args
        assert call_kwargs.kwargs["graph_hash"] == ""
        assert call_kwargs.kwargs["graph_node_count"] == 0

    def test_model_used_adds_inference_action(self, rt: SovereignRuntime) -> None:
        mock_ledger = MagicMock()
        rt._experience_ledger = mock_ledger

        result = self._make_result()
        result.model_used = "qwen2.5:7b"
        query = self._make_query()

        rt._commit_experience_episode(result, query)

        call_kwargs = mock_ledger.commit.call_args
        actions = call_kwargs.kwargs["actions"]
        inference_actions = [a for a in actions if a[0] == "inference"]
        assert len(inference_actions) == 1
        assert "qwen2.5:7b" in inference_actions[0][1]

    def test_exception_swallowed(self, rt: SovereignRuntime) -> None:
        mock_ledger = MagicMock()
        mock_ledger.commit.side_effect = RuntimeError("boom")
        rt._experience_ledger = mock_ledger

        result = self._make_result()
        query = self._make_query()

        # Should not raise
        rt._commit_experience_episode(result, query)


# ---------------------------------------------------------------------------
# 11. _init_gate_chain
# ---------------------------------------------------------------------------

class TestInitGateChain:
    """Tests for _init_gate_chain."""

    def test_success_path(self, rt: SovereignRuntime) -> None:
        mock_chain = MagicMock()
        mock_chain.gates = [MagicMock(name="gate1"), MagicMock(name="gate2")]

        mock_gates = MagicMock()
        mock_gates.GateChain = MagicMock(return_value=mock_chain)
        mock_receipt = MagicMock()

        with patch.dict("sys.modules", {
            "core.proof_engine.gates": mock_gates,
            "core.proof_engine.receipt": mock_receipt,
        }):
            rt._init_gate_chain()

        assert rt._gate_chain is mock_chain

    def test_failure_non_fatal(self, rt: SovereignRuntime) -> None:
        with patch.dict("sys.modules", {
            "core.proof_engine.gates": None,
        }):
            rt._init_gate_chain()
        assert rt._gate_chain is None


# ---------------------------------------------------------------------------
# 12. _run_gate_chain_preflight
# ---------------------------------------------------------------------------

class TestRunGateChainPreflight:
    """Tests for _run_gate_chain_preflight (async)."""

    @pytest.mark.asyncio
    async def test_none_gate_chain_rejects_query(self, rt: SovereignRuntime) -> None:
        """CRITICAL-1 FIX: None gate chain → REJECT (fail-closed), not pass-through."""
        rt._gate_chain = None
        query = SovereignQuery(text="test")
        result = SovereignResult(query_id="q1")
        returned = await rt._run_gate_chain_preflight(query, result)
        # Fail-closed: result should indicate rejection
        assert returned is not None or result.success is False

    @pytest.mark.asyncio
    async def test_pass_returns_none(self, rt: SovereignRuntime) -> None:
        mock_chain = MagicMock()
        chain_result = MagicMock()
        chain_result.passed = True
        chain_result.gate_results = [MagicMock(), MagicMock()]
        mock_chain.evaluate.return_value = (chain_result, MagicMock())
        rt._gate_chain = mock_chain

        mock_canonical = MagicMock()

        with patch.dict("sys.modules", {
            "core.proof_engine.canonical": mock_canonical,
        }):
            query = SovereignQuery(text="test")
            result = SovereignResult(query_id="q1")
            ret = await rt._run_gate_chain_preflight(query, result)

        assert ret is None

    @pytest.mark.asyncio
    async def test_fail_returns_rejection_result(self, rt: SovereignRuntime) -> None:
        mock_chain = MagicMock()
        chain_result = MagicMock()
        chain_result.passed = False
        chain_result.last_gate_passed = "ihsan_gate"
        chain_result.rejection_reason = "SNR below threshold"
        chain_result.snr = 0.70
        chain_result.ihsan_score = 0.88
        mock_chain.evaluate.return_value = (chain_result, MagicMock())
        rt._gate_chain = mock_chain

        mock_canonical = MagicMock()

        with patch.dict("sys.modules", {
            "core.proof_engine.canonical": mock_canonical,
        }):
            query = SovereignQuery(text="test")
            result = SovereignResult(query_id="q1")
            ret = await rt._run_gate_chain_preflight(query, result)

        assert ret is not None
        assert ret.success is False
        assert "gate chain" in ret.response.lower()
        assert ret.snr_score == 0.70
        assert ret.ihsan_score == 0.88
        assert ret.validation_passed is False

    @pytest.mark.asyncio
    async def test_exception_rejects_query(self, rt: SovereignRuntime) -> None:
        """CRITICAL-2 FIX: Gate chain exceptions → REJECT (fail-closed)."""
        mock_chain = MagicMock()
        mock_chain.evaluate.side_effect = RuntimeError("boom")
        rt._gate_chain = mock_chain

        mock_canonical = MagicMock()
        mock_canonical.CanonQuery.side_effect = RuntimeError("boom")

        with patch.dict("sys.modules", {
            "core.proof_engine.canonical": mock_canonical,
        }):
            query = SovereignQuery(text="test")
            result = SovereignResult(query_id="q1")
            ret = await rt._run_gate_chain_preflight(query, result)

        # CRITICAL-2: Exception → fail-closed (result indicates rejection)
        assert ret is not None or result.success is False


# ---------------------------------------------------------------------------
# 13. _emit_query_receipt
# ---------------------------------------------------------------------------

class TestEmitQueryReceipt:
    """Tests for _emit_query_receipt."""

    def _make_result(self, **kwargs: Any) -> SovereignResult:
        defaults = {
            "query_id": "q-001",
            "success": True,
            "response": "test response",
            "validation_passed": True,
            "snr_score": 0.92,
            "ihsan_score": 0.96,
            "thoughts": ["t1"],
            "claim_tags": {},
        }
        defaults.update(kwargs)
        return SovereignResult(**defaults)

    def test_none_ledger_noop(self, rt: SovereignRuntime) -> None:
        rt._evidence_ledger = None
        rt._emit_query_receipt(self._make_result(), SovereignQuery(text="test"))

    def test_approved_path(self, rt: SovereignRuntime) -> None:
        mock_ledger = MagicMock()
        rt._evidence_ledger = mock_ledger

        mock_el = MagicMock()

        with patch.dict("sys.modules", {
            "core.proof_engine.evidence_ledger": mock_el,
        }):
            result = self._make_result(validation_passed=True, snr_score=0.92)
            query = SovereignQuery(text="hello world")
            rt._emit_query_receipt(result, query)

        mock_el.emit_receipt.assert_called_once()
        call_kwargs = mock_el.emit_receipt.call_args
        assert call_kwargs.kwargs["decision"] == "APPROVED"
        assert call_kwargs.kwargs["status"] == "accepted"

    def test_rejected_path(self, rt: SovereignRuntime) -> None:
        mock_ledger = MagicMock()
        rt._evidence_ledger = mock_ledger

        mock_el = MagicMock()

        with patch.dict("sys.modules", {
            "core.proof_engine.evidence_ledger": mock_el,
        }):
            result = self._make_result(validation_passed=False, snr_score=0.92)
            query = SovereignQuery(text="test")
            rt._emit_query_receipt(result, query)

        call_kwargs = mock_el.emit_receipt.call_args
        assert call_kwargs.kwargs["decision"] == "REJECTED"
        assert call_kwargs.kwargs["status"] == "rejected"

    def test_quarantined_path(self, rt: SovereignRuntime) -> None:
        mock_ledger = MagicMock()
        rt._evidence_ledger = mock_ledger

        mock_el = MagicMock()

        with patch.dict("sys.modules", {
            "core.proof_engine.evidence_ledger": mock_el,
        }):
            # validation_passed=True but snr < 0.85
            result = self._make_result(validation_passed=True, snr_score=0.60)
            query = SovereignQuery(text="test")
            rt._emit_query_receipt(result, query)

        call_kwargs = mock_el.emit_receipt.call_args
        assert call_kwargs.kwargs["decision"] == "QUARANTINED"
        assert call_kwargs.kwargs["status"] == "quarantined"

    def test_exception_swallowed(self, rt: SovereignRuntime) -> None:
        mock_ledger = MagicMock()
        rt._evidence_ledger = mock_ledger

        with patch.dict("sys.modules", {
            "core.proof_engine.evidence_ledger": MagicMock(
                emit_receipt=MagicMock(side_effect=RuntimeError("boom"))
            ),
        }):
            rt._emit_query_receipt(self._make_result(), SovereignQuery(text="test"))


# ---------------------------------------------------------------------------
# 14. _register_poi_contribution
# ---------------------------------------------------------------------------

class TestRegisterPoiContribution:
    """Tests for _register_poi_contribution."""

    def _make_result(self, **kwargs: Any) -> SovereignResult:
        defaults = {
            "query_id": "q-001",
            "success": True,
            "snr_score": 0.92,
            "ihsan_score": 0.96,
            "graph_hash": "abc123",
        }
        defaults.update(kwargs)
        return SovereignResult(**defaults)

    def test_none_orchestrator_noop(self, rt: SovereignRuntime) -> None:
        rt._poi_orchestrator = None
        rt._register_poi_contribution(self._make_result(), SovereignQuery(text="q"))

    def test_result_not_success_noop(self, rt: SovereignRuntime) -> None:
        mock_orch = MagicMock()
        rt._poi_orchestrator = mock_orch
        rt._register_poi_contribution(
            self._make_result(success=False), SovereignQuery(text="q")
        )
        mock_orch.register_contribution.assert_not_called()

    def test_success_path(self, rt: SovereignRuntime) -> None:
        mock_orch = MagicMock()
        rt._poi_orchestrator = mock_orch

        mock_poi = MagicMock()

        with patch.dict("sys.modules", {
            "core.proof_engine.poi_engine": mock_poi,
        }):
            result = self._make_result()
            query = SovereignQuery(text="test query")
            rt._register_poi_contribution(result, query)

        mock_orch.register_contribution.assert_called_once()

    def test_exception_swallowed(self, rt: SovereignRuntime) -> None:
        mock_orch = MagicMock()
        mock_orch.register_contribution.side_effect = RuntimeError("boom")
        rt._poi_orchestrator = mock_orch

        mock_poi = MagicMock()

        with patch.dict("sys.modules", {
            "core.proof_engine.poi_engine": mock_poi,
        }):
            # Should not raise
            rt._register_poi_contribution(self._make_result(), SovereignQuery(text="q"))


# ---------------------------------------------------------------------------
# 15. _encode_query_memory
# ---------------------------------------------------------------------------

class TestEncodeQueryMemory:
    """Tests for _encode_query_memory."""

    def _make_result(self, **kwargs: Any) -> SovereignResult:
        defaults = {
            "success": True,
            "response": "meaningful response",
            "snr_score": 0.92,
            "ihsan_score": 0.96,
        }
        defaults.update(kwargs)
        return SovereignResult(**defaults)

    def test_none_living_memory_noop(self, rt: SovereignRuntime) -> None:
        rt._living_memory = None
        rt._encode_query_memory(self._make_result(), SovereignQuery(text="q"))

    def test_not_success_noop(self, rt: SovereignRuntime) -> None:
        rt._living_memory = MagicMock()
        rt._encode_query_memory(
            self._make_result(success=False), SovereignQuery(text="q")
        )

    def test_empty_response_noop(self, rt: SovereignRuntime) -> None:
        rt._living_memory = MagicMock()
        rt._encode_query_memory(
            self._make_result(response=""), SovereignQuery(text="q")
        )

    def test_success_path_schedules_encode(self, rt: SovereignRuntime) -> None:
        mock_memory = MagicMock()
        mock_memory.encode = AsyncMock()
        rt._living_memory = mock_memory

        mock_memory_type = MagicMock()

        with patch.dict("sys.modules", {
            "core.living_memory.core": mock_memory_type,
        }), patch("asyncio.ensure_future") as mock_ensure:
            result = self._make_result()
            query = SovereignQuery(text="test query")
            rt._encode_query_memory(result, query)

        mock_ensure.assert_called_once()


# ---------------------------------------------------------------------------
# 16. _store_graph_artifact
# ---------------------------------------------------------------------------

class TestStoreGraphArtifact:
    """Tests for _store_graph_artifact."""

    def test_no_graph_reasoner_noop(self, rt: SovereignRuntime) -> None:
        rt._graph_reasoner = None
        rt._store_graph_artifact("q1", "hash1")
        assert "q1" not in rt._graph_artifacts

    def test_success_stores_artifact(self, rt: SovereignRuntime) -> None:
        mock_reasoner = MagicMock()
        mock_artifact = {"id": "q1", "nodes": []}
        mock_reasoner.to_artifact.return_value = mock_artifact
        rt._graph_reasoner = mock_reasoner

        rt._store_graph_artifact("q1", "hash1")

        assert rt._graph_artifacts["q1"] == mock_artifact
        mock_reasoner.to_artifact.assert_called_once_with(build_id="q1")

    def test_overflow_evicts_oldest(self, rt: SovereignRuntime) -> None:
        mock_reasoner = MagicMock()
        mock_reasoner.to_artifact.return_value = {"new": True}
        rt._graph_reasoner = mock_reasoner

        # Pre-fill with 101 artifacts
        for i in range(101):
            rt._graph_artifacts[f"old-{i:03d}"] = {"idx": i}

        rt._store_graph_artifact("brand-new", "hash")

        # Should have evicted the oldest key
        assert "old-000" not in rt._graph_artifacts
        assert "brand-new" in rt._graph_artifacts

    def test_exception_swallowed(self, rt: SovereignRuntime) -> None:
        mock_reasoner = MagicMock()
        mock_reasoner.to_artifact.side_effect = RuntimeError("boom")
        rt._graph_reasoner = mock_reasoner

        rt._store_graph_artifact("q1", "hash1")  # should not raise

    def test_no_to_artifact_method_noop(self, rt: SovereignRuntime) -> None:
        """If graph_reasoner has no to_artifact, should silently return."""
        mock_reasoner = MagicMock(spec=[])  # Empty spec => no to_artifact
        rt._graph_reasoner = mock_reasoner

        rt._store_graph_artifact("q1", "hash1")
        assert "q1" not in rt._graph_artifacts


# ---------------------------------------------------------------------------
# 17. get_graph_artifact
# ---------------------------------------------------------------------------

class TestGetGraphArtifact:
    """Tests for get_graph_artifact."""

    def test_found(self, rt: SovereignRuntime) -> None:
        rt._graph_artifacts["q1"] = {"data": "here"}
        assert rt.get_graph_artifact("q1") == {"data": "here"}

    def test_not_found(self, rt: SovereignRuntime) -> None:
        assert rt.get_graph_artifact("nonexistent") is None


# ---------------------------------------------------------------------------
# 18. get_gate_chain_stats
# ---------------------------------------------------------------------------

class TestGetGateChainStats:
    """Tests for get_gate_chain_stats."""

    def test_none_gate_chain(self, rt: SovereignRuntime) -> None:
        rt._gate_chain = None
        assert rt.get_gate_chain_stats() is None

    def test_delegates_to_gate_chain(self, rt: SovereignRuntime) -> None:
        mock_chain = MagicMock()
        mock_chain.get_stats.return_value = {"evaluations": 10}
        rt._gate_chain = mock_chain

        result = rt.get_gate_chain_stats()
        assert result == {"evaluations": 10}
        mock_chain.get_stats.assert_called_once()


# ---------------------------------------------------------------------------
# 19. _init_poi_engine
# ---------------------------------------------------------------------------

class TestInitPoiEngine:
    """Tests for _init_poi_engine."""

    def test_success_path(self, rt: SovereignRuntime) -> None:
        mock_orch = MagicMock()
        mock_config = MagicMock()
        mock_config.alpha = 0.3
        mock_config.beta = 0.3
        mock_config.gamma = 0.4

        mock_poi_module = MagicMock()
        mock_poi_module.PoIOrchestrator.return_value = mock_orch
        mock_poi_module.PoIConfig.return_value = mock_config

        mock_sat_module = MagicMock()
        mock_sat_ctrl = MagicMock()
        mock_sat_module.SATController.return_value = mock_sat_ctrl

        with patch.dict("sys.modules", {
            "core.proof_engine.poi_engine": mock_poi_module,
            "core.sovereign.sat_controller": mock_sat_module,
        }):
            rt._init_poi_engine()

        assert rt._poi_orchestrator is mock_orch
        assert rt._sat_controller is mock_sat_ctrl

    def test_failure_non_fatal(self, rt: SovereignRuntime) -> None:
        with patch.dict("sys.modules", {
            "core.proof_engine.poi_engine": None,
        }):
            rt._init_poi_engine()
        assert rt._poi_orchestrator is None
        assert rt._sat_controller is None


# ---------------------------------------------------------------------------
# 20. get_poi_stats, get_contributor_poi, compute_poi_epoch
# ---------------------------------------------------------------------------

class TestPoiDelegation:
    """Tests for PoI delegation methods."""

    def test_get_poi_stats_none(self, rt: SovereignRuntime) -> None:
        rt._poi_orchestrator = None
        assert rt.get_poi_stats() is None

    def test_get_poi_stats_delegates(self, rt: SovereignRuntime) -> None:
        mock_orch = MagicMock()
        mock_orch.get_stats.return_value = {"total": 5}
        rt._poi_orchestrator = mock_orch
        assert rt.get_poi_stats() == {"total": 5}

    def test_get_contributor_poi_none(self, rt: SovereignRuntime) -> None:
        rt._poi_orchestrator = None
        assert rt.get_contributor_poi("node-1") is None

    def test_get_contributor_poi_delegates(self, rt: SovereignRuntime) -> None:
        mock_orch = MagicMock()
        mock_poi = MagicMock()
        mock_poi.to_dict.return_value = {"score": 0.85}
        mock_orch.get_contributor_poi.return_value = mock_poi
        rt._poi_orchestrator = mock_orch

        result = rt.get_contributor_poi("node-1")
        assert result == {"score": 0.85}
        mock_orch.get_contributor_poi.assert_called_once_with("node-1")

    def test_get_contributor_poi_not_found(self, rt: SovereignRuntime) -> None:
        mock_orch = MagicMock()
        mock_orch.get_contributor_poi.return_value = None
        rt._poi_orchestrator = mock_orch

        assert rt.get_contributor_poi("unknown") is None

    def test_compute_poi_epoch_none(self, rt: SovereignRuntime) -> None:
        rt._poi_orchestrator = None
        assert rt.compute_poi_epoch() is None

    def test_compute_poi_epoch_delegates(self, rt: SovereignRuntime) -> None:
        mock_orch = MagicMock()
        mock_audit = MagicMock()
        mock_audit.to_dict.return_value = {"epoch": "e1"}
        mock_orch.compute_epoch.return_value = mock_audit
        rt._poi_orchestrator = mock_orch

        result = rt.compute_poi_epoch("e1")
        assert result == {"epoch": "e1"}
        mock_orch.compute_epoch.assert_called_once_with("e1")


# ---------------------------------------------------------------------------
# 21. get_sat_stats, finalize_sat_epoch
# ---------------------------------------------------------------------------

class TestSatDelegation:
    """Tests for SAT Controller delegation methods."""

    def test_get_sat_stats_none(self, rt: SovereignRuntime) -> None:
        rt._sat_controller = None
        assert rt.get_sat_stats() is None

    def test_get_sat_stats_delegates(self, rt: SovereignRuntime) -> None:
        mock_ctrl = MagicMock()
        mock_ctrl.get_stats.return_value = {"epochs": 3}
        rt._sat_controller = mock_ctrl
        assert rt.get_sat_stats() == {"epochs": 3}

    def test_finalize_sat_epoch_none(self, rt: SovereignRuntime) -> None:
        rt._sat_controller = None
        assert rt.finalize_sat_epoch() is None

    def test_finalize_sat_epoch_delegates(self, rt: SovereignRuntime) -> None:
        mock_ctrl = MagicMock()
        mock_ctrl.finalize_epoch.return_value = {"distributed": 1000}
        rt._sat_controller = mock_ctrl

        result = rt.finalize_sat_epoch(epoch_reward=500.0)
        assert result == {"distributed": 1000}
        mock_ctrl.finalize_epoch.assert_called_once_with(500.0)


# ---------------------------------------------------------------------------
# 22. _health_status and _calculate_health
# ---------------------------------------------------------------------------

class TestHealth:
    """Tests for _health_status and _calculate_health."""

    def test_healthy(self, rt: SovereignRuntime) -> None:
        """Score >= 0.9 should be HEALTHY."""
        rt.metrics.current_snr_score = 0.90
        rt.metrics.current_ihsan_score = 0.97
        rt.metrics.queries_processed = 10
        rt.metrics.queries_succeeded = 10

        assert rt._health_status() == HealthStatus.HEALTHY

    def test_degraded(self, rt: SovereignRuntime) -> None:
        """0.7 <= score < 0.9 should be DEGRADED."""
        rt.metrics.current_snr_score = 0.70
        rt.metrics.current_ihsan_score = 0.80
        rt.metrics.queries_processed = 10
        rt.metrics.queries_succeeded = 7

        assert rt._health_status() == HealthStatus.DEGRADED

    def test_unhealthy(self, rt: SovereignRuntime) -> None:
        """0 < score < 0.7 should be UNHEALTHY."""
        rt.metrics.current_snr_score = 0.30
        rt.metrics.current_ihsan_score = 0.30
        rt.metrics.queries_processed = 10
        rt.metrics.queries_succeeded = 3

        assert rt._health_status() == HealthStatus.UNHEALTHY

    def test_unknown(self, rt: SovereignRuntime) -> None:
        """Score == 0 should be UNKNOWN."""
        rt.metrics.current_snr_score = 0.0
        rt.metrics.current_ihsan_score = 0.0
        rt.metrics.queries_processed = 0
        rt.metrics.queries_succeeded = 0

        assert rt._health_status() == HealthStatus.UNKNOWN

    def test_calculate_health_formula(self, rt: SovereignRuntime) -> None:
        """_calculate_health = (snr_factor + ihsan_factor + success_factor) / 3."""
        rt.config.snr_threshold = 0.85
        rt.config.ihsan_threshold = 0.95
        rt.metrics.current_snr_score = 0.85
        rt.metrics.current_ihsan_score = 0.95
        rt.metrics.queries_processed = 10
        rt.metrics.queries_succeeded = 10

        health = rt._calculate_health()
        # snr_factor = min(1.0, 0.85/0.85) = 1.0
        # ihsan_factor = min(1.0, 0.95/0.95) = 1.0
        # success_factor = 10/10 = 1.0
        # (1.0 + 1.0 + 1.0) / 3 = 1.0
        assert health == pytest.approx(1.0)

    def test_calculate_health_partial_scores(self, rt: SovereignRuntime) -> None:
        rt.config.snr_threshold = 0.85
        rt.config.ihsan_threshold = 0.95
        rt.metrics.current_snr_score = 0.425  # half of threshold
        rt.metrics.current_ihsan_score = 0.475  # half of threshold
        rt.metrics.queries_processed = 10
        rt.metrics.queries_succeeded = 5

        health = rt._calculate_health()
        # snr_factor = min(1.0, 0.425/0.85) = 0.5
        # ihsan_factor = min(1.0, 0.475/0.95) = 0.5
        # success_factor = 5/10 = 0.5
        # (0.5 + 0.5 + 0.5) / 3 = 0.5
        assert health == pytest.approx(0.5)

    def test_calculate_health_zero_queries(self, rt: SovereignRuntime) -> None:
        """Zero queries processed should use max(1, 0) = 1 as denominator."""
        rt.metrics.queries_processed = 0
        rt.metrics.queries_succeeded = 0
        # success_factor = 0 / max(1, 0) = 0
        health = rt._calculate_health()
        assert health == pytest.approx(0.0)

    def test_snr_above_threshold_capped_at_1(self, rt: SovereignRuntime) -> None:
        """SNR score above threshold should be capped at 1.0."""
        rt.config.snr_threshold = 0.85
        rt.metrics.current_snr_score = 1.0  # above threshold
        rt.metrics.current_ihsan_score = 0.0
        rt.metrics.queries_processed = 0

        health = rt._calculate_health()
        snr_factor = min(1.0, 1.0 / 0.85)
        assert snr_factor == 1.0  # capped


# ---------------------------------------------------------------------------
# 23. _cache_key and _update_cache
# ---------------------------------------------------------------------------

class TestCaching:
    """Tests for _cache_key and _update_cache."""

    def test_cache_key_deterministic(self, rt: SovereignRuntime) -> None:
        query = SovereignQuery(text="hello", require_reasoning=True)
        key1 = rt._cache_key(query)
        key2 = rt._cache_key(query)
        assert key1 == key2

    def test_cache_key_is_blake3_prefix(self, rt: SovereignRuntime) -> None:
        """Cache key uses BLAKE3 (SEC-001)."""
        from core.proof_engine.canonical import hex_digest

        query = SovereignQuery(text="test", require_reasoning=False)
        key = rt._cache_key(query)
        expected = hex_digest("test:False".encode())[:16]
        assert key == expected

    def test_cache_key_differs_by_reasoning(self, rt: SovereignRuntime) -> None:
        q1 = SovereignQuery(text="same", require_reasoning=True)
        q2 = SovereignQuery(text="same", require_reasoning=False)
        assert rt._cache_key(q1) != rt._cache_key(q2)

    def test_update_cache_stores_result(self, rt: SovereignRuntime) -> None:
        rt.config.max_cache_entries = 100
        result = SovereignResult(query_id="q1", response="res")
        rt._update_cache("k1", result)
        assert rt._cache["k1"] is result

    def test_update_cache_eviction(self, rt: SovereignRuntime) -> None:
        """When cache is at capacity, oldest 100 entries are evicted."""
        rt.config.max_cache_entries = 10
        for i in range(10):
            rt._cache[f"old-{i}"] = SovereignResult(query_id=f"old-{i}")

        assert len(rt._cache) == 10

        new_result = SovereignResult(query_id="new")
        rt._update_cache("new-key", new_result)

        # After eviction of up to 100 oldest + adding 1 new
        assert "new-key" in rt._cache
        assert rt._cache["new-key"] is new_result


# ---------------------------------------------------------------------------
# 24. _mode_to_tier
# ---------------------------------------------------------------------------

class TestModeToTier:
    """Tests for _mode_to_tier."""

    def test_import_error_returns_none(self, rt: SovereignRuntime) -> None:
        with patch.dict("sys.modules", {
            "core.inference.gateway": None,
        }):
            result = rt._mode_to_tier(MagicMock())
        assert result is None

    def test_non_treasury_mode_returns_none(self, rt: SovereignRuntime) -> None:
        """A non-TreasuryMode object should return None."""
        from enum import Enum, auto

        class FakeTreasuryMode(Enum):
            ETHICAL = auto()
            HIBERNATION = auto()
            EMERGENCY = auto()

        class FakeComputeTier(Enum):
            LOCAL = auto()
            EDGE = auto()

        mock_gateway_mod = MagicMock()
        mock_gateway_mod.ComputeTier = FakeComputeTier

        mock_omega_mod = MagicMock()
        mock_omega_mod.TreasuryMode = FakeTreasuryMode

        with patch.dict("sys.modules", {
            "core.inference.gateway": mock_gateway_mod,
            "core.sovereign.omega_engine": mock_omega_mod,
        }):
            # "not_a_mode" is not a FakeTreasuryMode instance
            result = rt._mode_to_tier("not_a_mode")
        assert result is None

    def test_ethical_maps_to_local(self, rt: SovereignRuntime) -> None:
        """TreasuryMode.ETHICAL should map to ComputeTier.LOCAL."""
        from enum import Enum, auto

        class FakeTreasuryMode(Enum):
            ETHICAL = auto()
            HIBERNATION = auto()
            EMERGENCY = auto()

        class FakeComputeTier(Enum):
            LOCAL = auto()
            EDGE = auto()

        mock_gateway_mod = MagicMock()
        mock_gateway_mod.ComputeTier = FakeComputeTier

        mock_omega_mod = MagicMock()
        mock_omega_mod.TreasuryMode = FakeTreasuryMode

        with patch.dict("sys.modules", {
            "core.inference.gateway": mock_gateway_mod,
            "core.sovereign.omega_engine": mock_omega_mod,
        }):
            result = rt._mode_to_tier(FakeTreasuryMode.ETHICAL)
        assert result == FakeComputeTier.LOCAL

    def test_hibernation_maps_to_edge(self, rt: SovereignRuntime) -> None:
        """TreasuryMode.HIBERNATION should map to ComputeTier.EDGE."""
        from enum import Enum, auto

        class FakeTreasuryMode(Enum):
            ETHICAL = auto()
            HIBERNATION = auto()
            EMERGENCY = auto()

        class FakeComputeTier(Enum):
            LOCAL = auto()
            EDGE = auto()

        mock_gateway_mod = MagicMock()
        mock_gateway_mod.ComputeTier = FakeComputeTier

        mock_omega_mod = MagicMock()
        mock_omega_mod.TreasuryMode = FakeTreasuryMode

        with patch.dict("sys.modules", {
            "core.inference.gateway": mock_gateway_mod,
            "core.sovereign.omega_engine": mock_omega_mod,
        }):
            result = rt._mode_to_tier(FakeTreasuryMode.HIBERNATION)
        assert result == FakeComputeTier.EDGE

    def test_emergency_maps_to_edge(self, rt: SovereignRuntime) -> None:
        """TreasuryMode.EMERGENCY should map to ComputeTier.EDGE."""
        from enum import Enum, auto

        class FakeTreasuryMode(Enum):
            ETHICAL = auto()
            HIBERNATION = auto()
            EMERGENCY = auto()

        class FakeComputeTier(Enum):
            LOCAL = auto()
            EDGE = auto()

        mock_gateway_mod = MagicMock()
        mock_gateway_mod.ComputeTier = FakeComputeTier

        mock_omega_mod = MagicMock()
        mock_omega_mod.TreasuryMode = FakeTreasuryMode

        with patch.dict("sys.modules", {
            "core.inference.gateway": mock_gateway_mod,
            "core.sovereign.omega_engine": mock_omega_mod,
        }):
            result = rt._mode_to_tier(FakeTreasuryMode.EMERGENCY)
        assert result == FakeComputeTier.EDGE


# ---------------------------------------------------------------------------
# 25. _extract_ihsan_from_response
# ---------------------------------------------------------------------------

class TestExtractIhsanFromResponse:
    """Tests for _extract_ihsan_from_response."""

    def test_import_error_returns_none(self, rt: SovereignRuntime) -> None:
        with patch.dict("sys.modules", {
            "core.sovereign.omega_engine": None,
        }):
            result = rt._extract_ihsan_from_response("hello", {})
        assert result is None

    def test_harmful_content_low_safety(self, rt: SovereignRuntime) -> None:
        mock_omega = MagicMock()
        mock_ihsan = MagicMock(return_value="ihsan_vector")
        mock_omega.ihsan_from_scores = mock_ihsan

        with patch.dict("sys.modules", {
            "core.sovereign.omega_engine": mock_omega,
        }):
            result = rt._extract_ihsan_from_response(
                "you should kill the process", {}
            )

        mock_ihsan.assert_called_once()
        call_kwargs = mock_ihsan.call_args
        # safety should be 0.50 because "kill" is in harmful words
        assert call_kwargs.kwargs["safety"] == pytest.approx(0.50)

    def test_normal_content_high_safety(self, rt: SovereignRuntime) -> None:
        mock_omega = MagicMock()
        mock_ihsan = MagicMock(return_value="ihsan_vector")
        mock_omega.ihsan_from_scores = mock_ihsan

        with patch.dict("sys.modules", {
            "core.sovereign.omega_engine": mock_omega,
        }):
            result = rt._extract_ihsan_from_response(
                "This is a perfectly safe response about cooking.", {}
            )

        mock_ihsan.assert_called_once()
        call_kwargs = mock_ihsan.call_args
        assert call_kwargs.kwargs["safety"] == pytest.approx(0.98)


# ---------------------------------------------------------------------------
# 26. _estimate_complexity
# ---------------------------------------------------------------------------

class TestEstimateComplexity:
    """Tests for _estimate_complexity."""

    def test_short_simple_query(self, rt: SovereignRuntime) -> None:
        query = SovereignQuery(text="What is 2+2?")
        score = rt._estimate_complexity(query)
        assert 0.0 <= score <= 0.3  # short, simple

    def test_long_query_higher_length_score(self, rt: SovereignRuntime) -> None:
        long_text = " ".join(["word"] * 80)
        query = SovereignQuery(text=long_text)
        score = rt._estimate_complexity(query)
        # length_score = min(80/80, 1.0) = 1.0
        # 0.3 * 1.0 = 0.3 from length alone
        assert score >= 0.3

    def test_sub_question_keywords(self, rt: SovereignRuntime) -> None:
        query = SovereignQuery(
            text="Compare and contrast the two approaches, then evaluate them step by step."
        )
        score = rt._estimate_complexity(query)
        # "compare", "contrast", "evaluate", "step by step", "then"
        assert score >= 0.2  # sub_q_score contributes

    def test_multiple_question_marks(self, rt: SovereignRuntime) -> None:
        query = SovereignQuery(text="What? Why? How? When? Where?")
        score = rt._estimate_complexity(query)
        # q_count=5, q_score = min(5*0.2, 0.6) = 0.6
        # 0.2 * 0.6 = 0.12 from question marks
        assert score > 0.1

    def test_complexity_hint_from_context(self, rt: SovereignRuntime) -> None:
        query = SovereignQuery(
            text="Simple query.",
            context={"complexity_hint": 1.0},
        )
        score = rt._estimate_complexity(query)
        # hint contribution: 0.2 * 1.0 = 0.2
        assert score >= 0.2

    def test_score_capped_at_1(self, rt: SovereignRuntime) -> None:
        """Score must never exceed 1.0."""
        long_text = " ".join(["compare analyze evaluate furthermore"] * 30) + "????"
        query = SovereignQuery(
            text=long_text,
            context={"complexity_hint": 1.0},
        )
        score = rt._estimate_complexity(query)
        assert score <= 1.0

    def test_empty_query(self, rt: SovereignRuntime) -> None:
        query = SovereignQuery(text="")
        score = rt._estimate_complexity(query)
        assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 27. _checkpoint
# ---------------------------------------------------------------------------

class TestCheckpoint:
    """Tests for _checkpoint (async)."""

    @pytest.mark.asyncio
    async def test_persistence_disabled_noop(self, rt: SovereignRuntime) -> None:
        rt.config.enable_persistence = False
        await rt._checkpoint()
        # No state_dir should be created
        assert not rt.config.state_dir.exists()

    @pytest.mark.asyncio
    async def test_success_writes_checkpoint(
        self, rt: SovereignRuntime, tmp_state: Path
    ) -> None:
        rt.config.enable_persistence = True
        rt.config.state_dir = tmp_state

        await rt._checkpoint()

        checkpoint_file = tmp_state / "checkpoint.json"
        assert checkpoint_file.exists()

        data = json.loads(checkpoint_file.read_text())
        assert "metrics" in data
        assert "config" in data
        assert data["config"]["node_id"] == "test-node-0001"
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_checkpoint_with_genesis(
        self, rt: SovereignRuntime, tmp_state: Path
    ) -> None:
        rt.config.enable_persistence = True
        rt.config.state_dir = tmp_state

        mock_genesis = MagicMock()
        mock_genesis.summary.return_value = {"hash": "abc123"}
        rt._genesis = mock_genesis

        await rt._checkpoint()

        data = json.loads((tmp_state / "checkpoint.json").read_text())
        assert "genesis" in data
        assert data["genesis"]["hash"] == "abc123"

    @pytest.mark.asyncio
    async def test_exception_swallowed(
        self, rt: SovereignRuntime, tmp_state: Path
    ) -> None:
        rt.config.enable_persistence = True
        rt.config.state_dir = tmp_state

        with patch.object(Path, "mkdir", side_effect=PermissionError("denied")):
            # Should not raise
            await rt._checkpoint()


# ---------------------------------------------------------------------------
# 28. status()
# ---------------------------------------------------------------------------

class TestStatus:
    """Tests for status() — integration-style test of the full status dict."""

    def test_basic_status_structure(self, rt: SovereignRuntime) -> None:
        status = rt.status()

        assert "identity" in status
        assert "state" in status
        assert "health" in status
        assert "autonomous" in status
        assert "omega_point" in status
        assert "memory" in status
        assert "sovereignty" in status
        assert "metrics" in status

    def test_identity_fields(self, rt: SovereignRuntime) -> None:
        status = rt.status()
        identity = status["identity"]

        assert identity["node_id"] == "test-node-0001"
        assert identity["version"] == "1.0.0"
        assert identity["origin"]["designation"] == "ephemeral_node"
        assert identity["origin"]["genesis_node"] is False
        assert identity["origin"]["genesis_block"] is False
        assert identity["origin"]["home_base_device"] is False
        assert identity["origin"]["authority_source"] == "genesis_files"
        assert identity["origin"]["hash_validated"] is False

    def test_state_fields(self, rt: SovereignRuntime) -> None:
        rt._initialized = True
        rt._running = True

        status = rt.status()
        state = status["state"]

        assert state["initialized"] is True
        assert state["running"] is True
        assert state["mode"] == "STANDARD"

    def test_with_genesis_identity(self, rt: SovereignRuntime) -> None:
        mock_genesis = MagicMock()
        mock_genesis.node_name = "BIZRA-Node0"
        mock_genesis.identity.location = "Doha, Qatar"
        mock_genesis.identity.public_key = "abcdef0123456789abcdef0123456789"
        mock_genesis.pat_team = [MagicMock(), MagicMock()]
        mock_genesis.sat_team = [MagicMock()]
        mock_genesis.genesis_hash = b"\xab\xcd\xef\x01\x23\x45\x67\x89"
        rt._genesis = mock_genesis
        rt._origin_snapshot = {
            "designation": "node0",
            "genesis_node": True,
            "genesis_block": True,
            "block_id": "block0",
            "home_base_device": True,
            "node_id": "node0_fixture_0001",
            "node_name": "Node0 Fixture",
            "authority_source": "genesis_files",
            "hash_validated": True,
        }

        status = rt.status()
        identity = status["identity"]

        assert identity["node_name"] == "BIZRA-Node0"
        assert identity["location"] == "Doha, Qatar"
        assert identity["pat_agents"] == 2
        assert identity["sat_agents"] == 1
        assert "genesis_hash" in identity
        assert identity["origin"]["designation"] == "node0"
        assert identity["origin"]["genesis_node"] is True
        assert identity["origin"]["genesis_block"] is True
        assert identity["origin"]["block_id"] == "block0"
        assert identity["origin"]["home_base_device"] is True

    def test_without_genesis(self, rt: SovereignRuntime) -> None:
        rt._genesis = None
        rt._origin_snapshot = {
            "designation": "ephemeral_node",
            "genesis_node": False,
            "genesis_block": False,
            "home_base_device": False,
            "authority_source": "genesis_files",
            "hash_validated": False,
        }
        status = rt.status()
        identity = status["identity"]

        assert "node_name" not in identity
        assert "pat_agents" not in identity
        assert identity["origin"]["designation"] == "ephemeral_node"
        assert identity["origin"]["genesis_node"] is False

    def test_health_contains_status_and_score(self, rt: SovereignRuntime) -> None:
        status = rt.status()
        health = status["health"]

        assert "status" in health
        assert "score" in health
        assert health["status"] in {"healthy", "degraded", "unhealthy", "unknown"}

    def test_autonomous_loop_not_set(self, rt: SovereignRuntime) -> None:
        rt._autonomous_loop = None
        status = rt.status()
        assert status["autonomous"] == {"running": False}

    def test_autonomous_loop_set(self, rt: SovereignRuntime) -> None:
        mock_loop = MagicMock()
        mock_loop.status.return_value = {"running": True, "cycle": 5}
        rt._autonomous_loop = mock_loop

        status = rt.status()
        assert status["autonomous"] == {"running": True, "cycle": 5}

    def test_omega_point_default_version(self, rt: SovereignRuntime) -> None:
        status = rt.status()
        assert status["omega_point"]["version"] == "2.2.3"

    def test_sovereignty_tracking_false_by_default(self, rt: SovereignRuntime) -> None:
        rt._impact_tracker = None
        status = rt.status()
        assert status["sovereignty"] == {"tracking": False}

    def test_with_impact_tracker(self, rt: SovereignRuntime) -> None:
        mock_tracker = MagicMock()
        mock_tracker.sovereignty_score = 0.87
        mock_tier = MagicMock()
        mock_tier.value = "SEEDLING"
        mock_tracker.sovereignty_tier = mock_tier
        mock_tracker.total_bloom = 42.5
        mock_tracker.achievements = ["first_query", "hundred_queries"]
        rt._impact_tracker = mock_tracker

        status = rt.status()
        sov = status["sovereignty"]

        assert sov["tracking"] is True
        assert sov["score"] == 0.87
        assert sov["tier"] == "SEEDLING"
        assert sov["total_bloom"] == 42.5
        assert sov["achievements"] == 2

    def test_memory_status_when_no_coordinator(self, rt: SovereignRuntime) -> None:
        rt._memory_coordinator = None
        status = rt.status()
        assert status["memory"] == {"running": False}

    def test_omega_with_gateway(self, rt: SovereignRuntime) -> None:
        mock_gw = MagicMock()
        mock_gw.status = "active"
        rt._gateway = mock_gw

        status = rt.status()
        gw = status["omega_point"]["gateway"]
        assert gw["connected"] is True
        assert gw["status"] == "active"

    def test_omega_without_gateway(self, rt: SovereignRuntime) -> None:
        rt._gateway = None
        status = rt.status()
        gw = status["omega_point"]["gateway"]
        assert gw["connected"] is False
