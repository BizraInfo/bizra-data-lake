"""Tests for core.protocols.degradation — Graceful Degradation Protocol.

Covers:
- DegradationSeverity enum values
- DegradationEvent: construction, immutability, degradation_ratio, to_dict
- DegradationEmitter: check, emit, is_healthy, logging levels
- Integration: FULL vs PARTIAL severity routing

Blueprint Reference: Section 3.2 — P1 Graceful Degradation Protocol
"""

import logging
from datetime import datetime, timezone

import pytest

from core.protocols.degradation import (
    DegradationEmitter,
    DegradationEvent,
    DegradationSeverity,
)

# ═══════════════════════════════════════════════════════════════════════════
# DegradationSeverity
# ═══════════════════════════════════════════════════════════════════════════


class TestDegradationSeverity:

    def test_values(self):
        assert DegradationSeverity.PARTIAL.value == "PARTIAL"
        assert DegradationSeverity.FULL.value == "FULL"

    def test_members(self):
        assert len(DegradationSeverity) == 2


# ═══════════════════════════════════════════════════════════════════════════
# DegradationEvent
# ═══════════════════════════════════════════════════════════════════════════


class TestDegradationEvent:

    def test_construction(self):
        event = DegradationEvent(
            engine="TestEngine",
            missing=["arg_a", "arg_b"],
            available=["arg_c"],
            severity=DegradationSeverity.PARTIAL,
        )
        assert event.engine == "TestEngine"
        assert event.missing == ["arg_a", "arg_b"]
        assert event.available == ["arg_c"]
        assert event.severity == DegradationSeverity.PARTIAL

    def test_frozen_immutable(self):
        event = DegradationEvent(
            engine="X",
            missing=[],
            available=[],
            severity=DegradationSeverity.FULL,
        )
        with pytest.raises(AttributeError):
            event.engine = "Y"  # type: ignore[misc]

    def test_degradation_ratio_full(self):
        event = DegradationEvent(
            engine="E",
            missing=["a", "b", "c"],
            available=[],
            severity=DegradationSeverity.FULL,
        )
        assert event.degradation_ratio == 1.0

    def test_degradation_ratio_partial(self):
        event = DegradationEvent(
            engine="E",
            missing=["a"],
            available=["b", "c"],
            severity=DegradationSeverity.PARTIAL,
        )
        assert abs(event.degradation_ratio - 1 / 3) < 1e-9

    def test_degradation_ratio_healthy(self):
        event = DegradationEvent(
            engine="E",
            missing=[],
            available=["a", "b"],
            severity=DegradationSeverity.PARTIAL,
        )
        assert event.degradation_ratio == 0.0

    def test_degradation_ratio_empty(self):
        event = DegradationEvent(
            engine="E",
            missing=[],
            available=[],
            severity=DegradationSeverity.FULL,
        )
        assert event.degradation_ratio == 0.0

    def test_timestamp_auto(self):
        before = datetime.now(timezone.utc)
        event = DegradationEvent(
            engine="E",
            missing=["a"],
            available=[],
            severity=DegradationSeverity.FULL,
        )
        after = datetime.now(timezone.utc)
        assert before <= event.timestamp <= after

    def test_to_dict(self):
        event = DegradationEvent(
            engine="CognitiveFusionEngine",
            missing=["moe_router", "hrm_engine"],
            available=["rag_engine"],
            severity=DegradationSeverity.PARTIAL,
        )
        d = event.to_dict()
        assert d["engine"] == "CognitiveFusionEngine"
        assert d["severity"] == "PARTIAL"
        assert d["missing"] == ["moe_router", "hrm_engine"]
        assert d["available"] == ["rag_engine"]
        assert isinstance(d["timestamp"], str)
        assert abs(d["degradation_ratio"] - 2 / 3) < 1e-9


# ═══════════════════════════════════════════════════════════════════════════
# DegradationEmitter
# ═══════════════════════════════════════════════════════════════════════════


class TestDegradationEmitter:

    def test_healthy_returns_none(self):
        emitter = DegradationEmitter("Healthy")
        emitter.check("arg_a", object())
        emitter.check("arg_b", object())
        assert emitter.emit() is None

    def test_is_healthy_true(self):
        emitter = DegradationEmitter("H")
        emitter.check("a", "present")
        assert emitter.is_healthy is True

    def test_is_healthy_false(self):
        emitter = DegradationEmitter("H")
        emitter.check("a", None)
        assert emitter.is_healthy is False

    def test_full_degradation(self):
        emitter = DegradationEmitter("CognitiveFusionEngine")
        emitter.check("moe_router", None)
        emitter.check("hrm_engine", None)
        emitter.check("rag_engine", None)
        emitter.check("northstar", None)
        event = emitter.emit()
        assert event is not None
        assert event.severity == DegradationSeverity.FULL
        assert len(event.missing) == 4
        assert len(event.available) == 0

    def test_partial_degradation(self):
        emitter = DegradationEmitter("GoTBridge")
        emitter.check("search_engine", None)
        emitter.check("got_engine", object())
        event = emitter.emit()
        assert event is not None
        assert event.severity == DegradationSeverity.PARTIAL
        assert event.missing == ["search_engine"]
        assert event.available == ["got_engine"]

    def test_total_checked(self):
        emitter = DegradationEmitter("E")
        emitter.check("a", None)
        emitter.check("b", "ok")
        emitter.check("c", None)
        assert emitter.total_checked == 3

    def test_full_logs_warning(self, caplog):
        emitter = DegradationEmitter("TestEngine")
        emitter.check("x", None)
        with caplog.at_level(logging.WARNING, logger="core.protocols.degradation"):
            event = emitter.emit()
        assert event is not None
        assert any("DEGRADATION-FULL" in r.message for r in caplog.records)

    def test_partial_logs_info(self, caplog):
        emitter = DegradationEmitter("TestEngine")
        emitter.check("x", None)
        emitter.check("y", "ok")
        with caplog.at_level(logging.INFO, logger="core.protocols.degradation"):
            event = emitter.emit()
        assert event is not None
        assert any("DEGRADATION-PARTIAL" in r.message for r in caplog.records)

    def test_emitter_engine_name_in_event(self):
        emitter = DegradationEmitter("BicameralReasoningEngine")
        emitter.check("local_endpoint", None)
        event = emitter.emit()
        assert event is not None
        assert event.engine == "BicameralReasoningEngine"


# ═══════════════════════════════════════════════════════════════════════════
# Engine Wiring Integration Tests (P1 validation)
# ═══════════════════════════════════════════════════════════════════════════


class TestCognitiveFusionDegradation:
    """Verify CognitiveFusionEngine emits degradation events."""

    def test_full_degradation_all_none(self):
        from core.cognitive_fusion.fusion_engine import CognitiveFusionEngine

        engine = CognitiveFusionEngine()
        assert engine._degraded is True
        assert engine._degradation_event is not None
        assert engine._degradation_event.severity == DegradationSeverity.FULL
        assert len(engine._degradation_event.missing) == 4

    def test_partial_degradation_some_present(self):
        from core.cognitive_fusion.fusion_engine import CognitiveFusionEngine

        # Provide a mock MoE router
        class FakeMoE:
            def classify(self, query, embedding):
                return None

        engine = CognitiveFusionEngine(moe_router=FakeMoE())
        assert engine._degraded is True
        assert engine._degradation_event.severity == DegradationSeverity.PARTIAL
        assert "moe_router" in engine._degradation_event.available
        assert len(engine._degradation_event.missing) == 3

    def test_healthy_no_degradation(self):
        from core.cognitive_fusion.fusion_engine import CognitiveFusionEngine

        class FakeMoE:
            def classify(self, query, embedding):
                return None

        class FakeHRM:
            def reason(self, query, level, context):
                return None

        class FakeRAG:
            def retrieve(self, query, query_embedding, top_k=10):
                return []

        class FakeNorthStar:
            def run_cycle(self, observation):
                return None

        engine = CognitiveFusionEngine(
            moe_router=FakeMoE(),
            hrm_engine=FakeHRM(),
            hypergraph_rag=FakeRAG(),
            northstar_engine=FakeNorthStar(),
        )
        assert engine._degraded is False
        assert engine._degradation_event is None


class TestGoTBridgeDegradation:
    """Verify GoTBridge emits degradation events."""

    def test_full_degradation(self):
        from core.reasoning.got_bridge import GoTBridge

        bridge = GoTBridge()
        assert bridge._degraded is True
        assert bridge._degradation_event is not None
        assert bridge._degradation_event.severity == DegradationSeverity.FULL

    def test_partial_with_search(self):
        from core.reasoning.got_bridge import GoTBridge

        bridge = GoTBridge(search_engine=object())
        assert bridge._degraded is True
        assert bridge._degradation_event.severity == DegradationSeverity.PARTIAL
        assert "search_engine" in bridge._degradation_event.available

    def test_healthy(self):
        from core.reasoning.got_bridge import GoTBridge

        bridge = GoTBridge(search_engine=object(), got_engine=object())
        assert bridge._degraded is False


class TestBicameralDegradation:
    """Verify BicameralReasoningEngine emits degradation events."""

    def test_full_degradation(self):
        from core.sovereign.bicameral_engine import BicameralReasoningEngine

        engine = BicameralReasoningEngine()
        assert engine._degraded is True
        assert engine._degradation_event is not None
        assert engine._degradation_event.severity == DegradationSeverity.FULL

    def test_partial_with_local(self):
        from core.sovereign.bicameral_engine import BicameralReasoningEngine

        class FakeLocal:
            async def generate(self, prompt, max_tokens, temperature):
                return ""

        engine = BicameralReasoningEngine(local_endpoint=FakeLocal())
        assert engine._degraded is True
        assert engine._degradation_event.severity == DegradationSeverity.PARTIAL

    def test_healthy(self):
        from core.sovereign.bicameral_engine import BicameralReasoningEngine

        class FakeLocal:
            async def generate(self, prompt, max_tokens, temperature):
                return ""

        class FakeAPI:
            async def analyze(self, content, criteria):
                return {}

        engine = BicameralReasoningEngine(
            local_endpoint=FakeLocal(), api_client=FakeAPI()
        )
        assert engine._degraded is False
