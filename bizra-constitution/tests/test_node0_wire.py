"""Tests for BIZRA Node0 Integration Wire."""

import os
import sys
import pytest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BIZRA_CONSTITUTION_PATH",
                       str(Path(__file__).parent.parent / "constitution.toml"))

from node0_wire import GenesisWire, WireResult, wire_genesis_engine


@pytest.fixture
def wire(tmp_path):
    return GenesisWire(
        data_dir=tmp_path / "genesis",
        ollama_url="http://localhost:99999",  # No real Ollama
    )


class TestGenesisWireInit:
    def test_initializes_successfully(self, wire):
        result = wire.initialize()
        assert result is True
        assert wire._initialized is True

    def test_pipeline_created(self, wire):
        wire.initialize()
        assert wire._pipeline is not None

    def test_node_id_assigned(self, wire):
        wire.initialize()
        assert len(wire._pipeline.identity.node_id) == 64

    def test_double_init_is_idempotent(self, wire):
        wire.initialize()
        node_id_1 = wire._pipeline.identity.node_id
        wire.initialize()
        # Same pipeline, not re-initialized
        assert wire._pipeline.identity.node_id == node_id_1


class TestGenesisWireExecution:
    def test_execute_returns_wire_result(self, wire):
        result = wire.execute("Hello from wire")
        assert isinstance(result, WireResult)

    def test_wire_result_has_output(self, wire):
        result = wire.execute("Test input")
        assert len(result.output) > 0

    def test_wire_result_has_ihsan(self, wire):
        result = wire.execute("What is AI?")
        assert 0 <= result.ihsan_composite <= 1

    def test_wire_result_has_dimensions(self, wire):
        result = wire.execute("Test")
        assert len(result.ihsan_dimensions) == 6

    def test_wire_result_has_snr(self, wire):
        result = wire.execute("Test")
        assert 0 <= result.snr_normalized <= 1

    def test_wire_result_is_signed(self, wire):
        result = wire.execute("Sign this")
        assert result.signed is True

    def test_wire_result_has_node_id(self, wire):
        result = wire.execute("Test")
        assert result.node_id is not None
        assert len(result.node_id) == 64

    def test_wire_result_has_receipt_id(self, wire):
        result = wire.execute("Test")
        assert result.evidence_receipt_id is not None

    def test_wire_result_has_tier(self, wire):
        result = wire.execute("Test")
        assert result.tier in ["trivial", "simple", "complex", "sovereign"]

    def test_wire_result_has_latency(self, wire):
        result = wire.execute("Test")
        assert result.latency_ms > 0


class TestEventBusPayload:
    def test_payload_structure(self, wire):
        result = wire.execute("Test")
        payload = result.to_event_bus_payload()
        assert payload["type"] == "mission_complete"
        assert "output" in payload
        assert "ihsan" in payload
        assert "snr" in payload
        assert "classification" in payload
        assert "evidence" in payload
        assert "timing" in payload
        assert "agent_trace" in payload

    def test_payload_ihsan_nested(self, wire):
        result = wire.execute("Test")
        payload = result.to_event_bus_payload()
        ihsan = payload["ihsan"]
        assert "composite" in ihsan
        assert "dimensions" in ihsan
        assert "bloom_eligible" in ihsan

    def test_payload_classification_nested(self, wire):
        result = wire.execute("Test")
        payload = result.to_event_bus_payload()
        cls = payload["classification"]
        assert "tier" in cls
        assert "confidence" in cls
        assert "reflex_hit" in cls

    def test_payload_evidence_nested(self, wire):
        result = wire.execute("Test")
        payload = result.to_event_bus_payload()
        ev = payload["evidence"]
        assert "receipt_id" in ev
        assert "signed" in ev
        assert "node_id" in ev


class TestFallbackBehavior:
    def test_disabled_wire_returns_none(self):
        os.environ["BIZRA_GENESIS_WIRE"] = "false"
        try:
            wire = wire_genesis_engine()
            assert wire is None
        finally:
            os.environ.pop("BIZRA_GENESIS_WIRE", None)

    def test_disabled_wire_zero_returns_none(self):
        os.environ["BIZRA_GENESIS_WIRE"] = "0"
        try:
            wire = wire_genesis_engine()
            assert wire is None
        finally:
            os.environ.pop("BIZRA_GENESIS_WIRE", None)

    def test_fallback_callback_called(self, tmp_path):
        fallback_log = []

        def on_fb(input_text, error):
            fallback_log.append((input_text, str(error)))

        wire = GenesisWire(
            data_dir=tmp_path / "genesis",
            ollama_url="http://localhost:99999",
            on_fallback=on_fb,
        )
        # Force pipeline to None to simulate failure
        wire._initialized = True
        wire._pipeline = None

        result = wire.execute("should fallback")
        assert result is None
        assert wire._fallback_missions == 1


class TestWireMetrics:
    def test_mission_count_increments(self, wire):
        wire.execute("one")
        wire.execute("two")
        assert wire._total_missions == 2

    def test_genesis_missions_tracked(self, wire):
        wire.execute("test")
        assert wire._genesis_missions == 1

    def test_health_report(self, wire):
        wire.execute("test")
        h = wire.health()
        assert h["wire_enabled"] is True
        assert h["total_missions"] == 1
        assert h["genesis_missions"] == 1
        assert h["genesis_rate"] == 1.0
        assert h["avg_latency_ms"] > 0
        assert h["pipeline_health"] is not None

    def test_health_when_not_initialized(self, tmp_path):
        wire = GenesisWire(data_dir=tmp_path / "noinit")
        h = wire.health()
        assert h["wire_enabled"] is False
        assert h["initialized"] is False


class TestWireFactory:
    def test_factory_returns_wire(self, tmp_path):
        wire = wire_genesis_engine(
            data_dir=tmp_path / "factory",
            ollama_url="http://localhost:99999",
        )
        assert isinstance(wire, GenesisWire)
        assert wire._initialized is True

    def test_factory_wire_executes(self, tmp_path):
        wire = wire_genesis_engine(
            data_dir=tmp_path / "factory",
            ollama_url="http://localhost:99999",
        )
        result = wire.execute("factory test")
        assert result is not None
        assert result.success is True


class TestShutdown:
    def test_shutdown_no_error(self, wire):
        wire.execute("before shutdown")
        wire.shutdown()
        # No exception = success

    def test_shutdown_without_init(self, tmp_path):
        wire = GenesisWire(data_dir=tmp_path / "noinit")
        wire.shutdown()  # Should not raise
