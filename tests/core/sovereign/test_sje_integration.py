"""
Tests for the Self-Evolving Judgment Engine (SJE) integration.

Covers: runtime wiring, verdict classification, API endpoint handlers,
and epoch distribution simulation via API.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.sovereign.judgment_telemetry import (
    JudgmentTelemetry,
    JudgmentVerdict,
    simulate_epoch_distribution,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def telemetry():
    """Create a fresh JudgmentTelemetry instance."""
    return JudgmentTelemetry()


@pytest.fixture
def populated_telemetry():
    """Create a telemetry with mixed verdicts."""
    t = JudgmentTelemetry()
    for _ in range(10):
        t.observe(JudgmentVerdict.PROMOTE)
    for _ in range(5):
        t.observe(JudgmentVerdict.NEUTRAL)
    for _ in range(2):
        t.observe(JudgmentVerdict.DEMOTE)
    t.observe(JudgmentVerdict.FORBID)
    return t


@pytest.fixture
def mock_runtime(populated_telemetry):
    """Mock runtime with judgment telemetry attached."""
    runtime = MagicMock()
    runtime._judgment_telemetry = populated_telemetry
    return runtime


@pytest.fixture
def mock_runtime_no_jt():
    """Mock runtime without judgment telemetry."""
    runtime = MagicMock()
    runtime._judgment_telemetry = None
    return runtime


# ═══════════════════════════════════════════════════════════════════════════════
# Verdict Classification Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestVerdictClassification:
    """Test the _observe_judgment verdict classification logic.

    These tests simulate the classification logic directly, matching
    the implementation in runtime_core.py::_observe_judgment.
    """

    def _classify(self, success, snr_ok, ihsan_score, validated=False, validation_passed=True):
        """Classify a verdict using the same logic as _observe_judgment."""
        if not success or (validated and not validation_passed):
            return JudgmentVerdict.FORBID
        elif not snr_ok:
            return JudgmentVerdict.DEMOTE
        elif ihsan_score >= 0.95:
            return JudgmentVerdict.PROMOTE
        else:
            return JudgmentVerdict.NEUTRAL

    def test_promote_on_excellence(self):
        assert self._classify(True, True, 0.98) == JudgmentVerdict.PROMOTE

    def test_promote_at_threshold(self):
        assert self._classify(True, True, 0.95) == JudgmentVerdict.PROMOTE

    def test_neutral_on_acceptable(self):
        assert self._classify(True, True, 0.90) == JudgmentVerdict.NEUTRAL

    def test_neutral_at_snr_floor(self):
        assert self._classify(True, True, 0.85) == JudgmentVerdict.NEUTRAL

    def test_demote_on_low_snr(self):
        assert self._classify(True, False, 0.90) == JudgmentVerdict.DEMOTE

    def test_forbid_on_failure(self):
        assert self._classify(False, True, 0.95) == JudgmentVerdict.FORBID

    def test_forbid_on_validation_failure(self):
        assert self._classify(True, True, 0.95, validated=True, validation_passed=False) == JudgmentVerdict.FORBID

    def test_forbid_overrides_snr(self):
        """Failure takes precedence over SNR status."""
        assert self._classify(False, False, 0.50) == JudgmentVerdict.FORBID

    def test_demote_overrides_ihsan(self):
        """Low SNR takes precedence over high Ihsan."""
        assert self._classify(True, False, 0.99) == JudgmentVerdict.DEMOTE


# ═══════════════════════════════════════════════════════════════════════════════
# Runtime Wiring Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestRuntimeWiring:
    """Test SJE runtime integration without starting the full runtime."""

    def test_telemetry_observe_accumulates(self, telemetry):
        """Observations accumulate correctly."""
        telemetry.observe(JudgmentVerdict.PROMOTE)
        telemetry.observe(JudgmentVerdict.PROMOTE)
        telemetry.observe(JudgmentVerdict.DEMOTE)
        assert telemetry.total_observations == 3
        assert telemetry.verdict_counts[JudgmentVerdict.PROMOTE] == 2

    def test_populated_telemetry_stats(self, populated_telemetry):
        """Populated telemetry has correct totals."""
        assert populated_telemetry.total_observations == 18
        assert populated_telemetry.dominant_verdict() == JudgmentVerdict.PROMOTE

    def test_populated_telemetry_entropy(self, populated_telemetry):
        """Mixed verdicts produce non-zero entropy."""
        assert populated_telemetry.entropy() > 0.0

    def test_to_dict_serialization(self, populated_telemetry):
        """to_dict produces a valid JSON-serializable dict."""
        d = populated_telemetry.to_dict()
        assert d["total_observations"] == 18
        assert "entropy" in d
        assert d["dominant_verdict"] == "promote"
        json_str = json.dumps(d)
        assert len(json_str) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# API Handler Tests (asyncio server)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSJEAPIHandlers:
    """Test the asyncio server SJE endpoint handlers."""

    @pytest.fixture
    def api_server(self, mock_runtime):
        """Create a SovereignAPIServer with mocked runtime."""
        from core.sovereign.api import SovereignAPIServer
        return SovereignAPIServer(mock_runtime)

    @pytest.fixture
    def api_server_no_jt(self, mock_runtime_no_jt):
        """Create a SovereignAPIServer without judgment telemetry."""
        from core.sovereign.api import SovereignAPIServer
        return SovereignAPIServer(mock_runtime_no_jt)

    @pytest.mark.asyncio
    async def test_judgment_stats_endpoint(self, api_server):
        """GET /v1/judgment/stats returns telemetry data."""
        resp = await api_server._handle_judgment_stats()
        data = json.loads(resp.split("\r\n\r\n", 1)[1]) if "\r\n\r\n" in resp else json.loads(resp)
        assert data["total_observations"] == 18
        assert data["dominant_verdict"] == "promote"
        assert "entropy" in data

    @pytest.mark.asyncio
    async def test_judgment_stats_not_initialized(self, api_server_no_jt):
        """GET /v1/judgment/stats returns 404 when not initialized."""
        resp = await api_server_no_jt._handle_judgment_stats()
        assert "404" in resp or "not initialized" in resp.lower()

    @pytest.mark.asyncio
    async def test_judgment_stability_endpoint(self, api_server):
        """GET /v1/judgment/stability returns stability data."""
        resp = await api_server._handle_judgment_stability()
        data = json.loads(resp.split("\r\n\r\n", 1)[1]) if "\r\n\r\n" in resp else json.loads(resp)
        assert "is_stable" in data
        assert "entropy" in data
        assert data["total_observations"] == 18

    @pytest.mark.asyncio
    async def test_judgment_simulate_endpoint(self, api_server):
        """POST /v1/judgment/simulate returns allocations."""
        body = json.dumps({"impacts": [100, 200, 300], "epoch_cap": 600}).encode()
        resp = await api_server._handle_judgment_simulate(body)
        data = json.loads(resp.split("\r\n\r\n", 1)[1]) if "\r\n\r\n" in resp else json.loads(resp)
        assert data["allocations"] == [100, 200, 300]
        assert data["dust"] == 0

    @pytest.mark.asyncio
    async def test_judgment_simulate_rounding(self, api_server):
        """Simulate with rounding dust."""
        body = json.dumps({"impacts": [1, 1, 1], "epoch_cap": 100}).encode()
        resp = await api_server._handle_judgment_simulate(body)
        data = json.loads(resp.split("\r\n\r\n", 1)[1]) if "\r\n\r\n" in resp else json.loads(resp)
        assert data["allocations"] == [33, 33, 33]
        assert data["dust"] == 1

    @pytest.mark.asyncio
    async def test_judgment_simulate_empty(self, api_server):
        """Simulate with empty impacts."""
        body = json.dumps({"impacts": [], "epoch_cap": 1000}).encode()
        resp = await api_server._handle_judgment_simulate(body)
        data = json.loads(resp.split("\r\n\r\n", 1)[1]) if "\r\n\r\n" in resp else json.loads(resp)
        assert data["allocations"] == []

    @pytest.mark.asyncio
    async def test_judgment_simulate_invalid_json(self, api_server):
        """Simulate with invalid JSON returns error."""
        resp = await api_server._handle_judgment_simulate(b"not json")
        assert "400" in resp or "Invalid JSON" in resp


# ═══════════════════════════════════════════════════════════════════════════════
# SJE + SEL Cross-Component Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSJESELCrossComponent:
    """Test that SJE and SEL work together correctly."""

    def test_promote_verdict_on_high_quality(self, telemetry):
        """High quality result should produce PROMOTE verdict."""
        telemetry.observe(JudgmentVerdict.PROMOTE)
        assert telemetry.total_observations == 1
        assert telemetry.dominant_verdict() == JudgmentVerdict.PROMOTE

    def test_mixed_verdicts_entropy(self, telemetry):
        """Mixed verdicts produce measurable entropy."""
        for v in JudgmentVerdict:
            telemetry.observe(v)
        # 4 equal verdicts: entropy = log2(4) = 2.0
        assert abs(telemetry.entropy() - 2.0) < 0.01

    def test_stability_with_dominant_verdict(self, telemetry):
        """Strongly dominant verdict should be stable."""
        for _ in range(100):
            telemetry.observe(JudgmentVerdict.PROMOTE)
        telemetry.observe(JudgmentVerdict.NEUTRAL)
        assert telemetry.is_stable()
        assert telemetry.dominant_verdict() == JudgmentVerdict.PROMOTE

    def test_distribution_percentages(self, populated_telemetry):
        """Distribution percentages sum to ~100%."""
        dist = populated_telemetry.distribution()
        total_pct = sum(dist.values())
        assert abs(total_pct - 100.0) < 0.01
