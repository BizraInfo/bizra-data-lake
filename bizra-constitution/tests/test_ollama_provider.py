"""Tests for BIZRA Ollama Provider — Circuit Breaker & Fallback Chain."""

import json
import os
import sys
import time
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ollama_provider import (
    CIRCUIT_BREAKER_THRESHOLD,
    CircuitBreaker,
    CircuitState,
    InferenceResult,
    ModelMetrics,
    OllamaProvider,
)


@pytest.fixture
def provider():
    return OllamaProvider(
        base_url="http://localhost:11434",
        model_chain=["model-a", "model-b", "model-c"],
        default_timeout_s=5.0,
    )


def _mock_response(text="Hello from LLM", eval_count=10):
    """Create a mock urllib response."""
    body = json.dumps(
        {
            "response": text,
            "eval_count": eval_count,
            "eval_duration": 500_000_000,  # 500ms in nanoseconds
        }
    ).encode()
    mock = MagicMock()
    mock.read.return_value = body
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    return mock


def _mock_error():
    """Create a mock that raises an error."""
    import urllib.error

    raise urllib.error.URLError("Connection refused")


# ═══════════════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════════════════════════


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker(model="test")
        assert cb.state == CircuitState.CLOSED
        assert cb.is_available()

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(model="test", threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_available()  # Still closed
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert not cb.is_available()

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(model="test", threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED

    def test_half_open_after_reset_timeout(self):
        cb = CircuitBreaker(model="test", threshold=1, reset_timeout_s=0.01)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.02)
        assert cb.is_available()  # Should transition to HALF_OPEN
        assert cb.state == CircuitState.HALF_OPEN

    def test_open_blocks_before_timeout(self):
        cb = CircuitBreaker(model="test", threshold=1, reset_timeout_s=60)
        cb.record_failure()
        assert not cb.is_available()


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL METRICS
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelMetrics:
    def test_initial_state(self):
        m = ModelMetrics(model="test")
        assert m.success_rate == 0.0
        assert m.avg_latency_ms == 0.0

    def test_success_tracking(self):
        m = ModelMetrics(model="test")
        m.record_success(100.0)
        m.record_success(200.0)
        assert m.successes == 2
        assert m.success_rate == 1.0
        assert m.avg_latency_ms == 150.0

    def test_failure_tracking(self):
        m = ModelMetrics(model="test")
        m.record_success(100.0)
        m.record_failure()
        assert m.success_rate == 0.5

    def test_as_dict(self):
        m = ModelMetrics(model="test")
        m.record_success(100.0)
        d = m.as_dict()
        assert "success_rate" in d
        assert "avg_latency_ms" in d


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE RESULT
# ═══════════════════════════════════════════════════════════════════════════════


class TestInferenceResult:
    def test_as_evidence(self):
        r = InferenceResult(
            text="hello",
            model="test",
            latency_ms=100,
            tokens_generated=10,
            tokens_per_second=100,
            is_fallback=False,
            fallback_chain=["test"],
            success=True,
        )
        ev = r.as_evidence()
        assert ev["model"] == "test"
        assert ev["success"] is True
        assert ev["latency_ms"] == 100

    def test_failed_result(self):
        r = InferenceResult(
            text="",
            model="test",
            latency_ms=5000,
            tokens_generated=0,
            tokens_per_second=0,
            is_fallback=False,
            fallback_chain=["test"],
            success=False,
            error="timeout",
        )
        assert not r.success
        assert r.error == "timeout"


# ═══════════════════════════════════════════════════════════════════════════════
# PROVIDER — GENERATE
# ═══════════════════════════════════════════════════════════════════════════════


class TestGenerate:
    @patch("ollama_provider.urllib.request.urlopen")
    def test_successful_generation(self, mock_urlopen, provider):
        mock_urlopen.return_value = _mock_response("Generated text")
        result = provider.generate("Hello")
        assert result.success
        assert result.text == "Generated text"
        assert result.model == "model-a"

    @patch("ollama_provider.urllib.request.urlopen")
    def test_fallback_on_first_failure(self, mock_urlopen, provider):
        import urllib.error

        mock_urlopen.side_effect = [
            urllib.error.URLError("fail"),
            _mock_response("Fallback response"),
        ]
        result = provider.generate("Hello")
        assert result.success
        assert result.text == "Fallback response"
        assert result.model == "model-b"
        assert result.is_fallback

    @patch("ollama_provider.urllib.request.urlopen")
    def test_all_models_fail(self, mock_urlopen, provider):
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("all down")
        result = provider.generate("Hello")
        assert not result.success
        assert "All" in result.error

    @patch("ollama_provider.urllib.request.urlopen")
    def test_latency_tracked(self, mock_urlopen, provider):
        mock_urlopen.return_value = _mock_response()
        result = provider.generate("Hello")
        assert result.latency_ms > 0

    @patch("ollama_provider.urllib.request.urlopen")
    def test_circuit_breaker_skips_failed_model(self, mock_urlopen, provider):
        import urllib.error

        # Trip circuit breaker on model-a
        for _ in range(CIRCUIT_BREAKER_THRESHOLD):
            provider._breakers["model-a"].record_failure()

        mock_urlopen.return_value = _mock_response("From model-b")
        result = provider.generate("Hello")
        assert result.success
        assert result.model == "model-b"
        # model-a should have been skipped
        assert "model-a(circuit-open)" in result.fallback_chain

    @patch("ollama_provider.urllib.request.urlopen")
    def test_metrics_updated_on_success(self, mock_urlopen, provider):
        mock_urlopen.return_value = _mock_response()
        provider.generate("Hello")
        assert provider._metrics["model-a"].successes == 1

    @patch("ollama_provider.urllib.request.urlopen")
    def test_metrics_updated_on_failure(self, mock_urlopen, provider):
        import urllib.error

        mock_urlopen.side_effect = [
            urllib.error.URLError("fail"),
            _mock_response(),
        ]
        provider.generate("Hello")
        assert provider._metrics["model-a"].failures == 1
        assert provider._metrics["model-b"].successes == 1


# ═══════════════════════════════════════════════════════════════════════════════
# PROVIDER — HEALTH
# ═══════════════════════════════════════════════════════════════════════════════


class TestProviderHealth:
    def test_health_report_structure(self, provider):
        h = provider.health()
        assert "server_available" in h
        assert "model_chain" in h
        assert "circuit_breakers" in h
        assert "metrics" in h

    def test_model_chain_in_health(self, provider):
        h = provider.health()
        assert h["model_chain"] == ["model-a", "model-b", "model-c"]

    def test_breaker_states_in_health(self, provider):
        h = provider.health()
        assert "model-a" in h["circuit_breakers"]
        assert h["circuit_breakers"]["model-a"]["state"] == "closed"

    @patch("ollama_provider.urllib.request.urlopen")
    def test_is_available_when_server_up(self, mock_urlopen, provider):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"models":[]}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp
        assert provider.is_available()

    def test_is_available_when_server_down(self, provider):
        # Default: no mock, real connection to localhost will fail
        # This tests the error handling path
        p = OllamaProvider(base_url="http://localhost:99999")
        assert not p.is_available()
