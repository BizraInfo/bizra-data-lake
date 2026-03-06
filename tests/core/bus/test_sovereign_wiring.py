"""
Sovereign Bus Wiring Tests — Phase 70.01
=========================================

TDD anchors for bus-to-runtime integration: component initialization,
graceful degradation, and wiring state reporting.

Standing on Giants:
- Beck (2002): TDD by Example
- Fowler (2005): Integration patterns
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from core.bus.sovereign_wiring import (
    BusWiringState,
    wire_action_bus,
    wire_all,
    wire_capsule_runtime,
    wire_config_loader,
    wire_omega_controller,
    wire_telescript_engine,
    wire_topic_registry,
)

# ═══════════════════════════════════════════════════════════
# TopicRegistry
# ═══════════════════════════════════════════════════════════


class TestWireTopicRegistry:
    """TopicRegistry initialization."""

    def test_returns_registry_with_topics(self) -> None:
        registry = wire_topic_registry()
        assert len(registry.active_topics()) >= 10  # immutable tiers always active

    def test_registry_validates_known_topic(self) -> None:
        registry = wire_topic_registry()
        assert registry.validate("action.intent") is True

    def test_registry_rejects_unknown_topic(self) -> None:
        registry = wire_topic_registry()
        assert registry.validate("totally.unknown.topic") is False


# ═══════════════════════════════════════════════════════════
# TeleScript
# ═══════════════════════════════════════════════════════════


class TestWireTeleScript:
    """TeleScript engine initialization."""

    def test_default_policy_denies_shell(self) -> None:
        engine = wire_telescript_engine()
        verdict = engine.check(requested=("shell_execute",))
        assert not verdict.allowed

    def test_default_policy_allows_file_read(self) -> None:
        engine = wire_telescript_engine()
        verdict = engine.check(requested=("file_read",))
        assert verdict.allowed

    def test_custom_policy(self) -> None:
        engine = wire_telescript_engine(
            policy_allow=frozenset(["file_read"]),
            policy_deny=frozenset(["file_write"]),
        )
        assert engine.check(requested=("file_read",)).allowed
        assert not engine.check(requested=("file_write",)).allowed


# ═══════════════════════════════════════════════════════════
# ActionBus
# ═══════════════════════════════════════════════════════════


class TestWireActionBus:
    """ActionBus initialization."""

    def test_creates_bus_with_no_channels(self) -> None:
        bus = wire_action_bus()
        assert bus is not None
        assert bus.receipt_chain == []

    def test_creates_bus_with_event_bus(self) -> None:
        eb = AsyncMock()
        bus = wire_action_bus(event_bus=eb)
        assert bus is not None

    def test_creates_bus_with_channels(self) -> None:
        ch = AsyncMock()
        bus = wire_action_bus(channels={"file": ch})
        assert bus is not None


# ═══════════════════════════════════════════════════════════
# ConfigLoader
# ═══════════════════════════════════════════════════════════


class TestWireConfigLoader:
    """ConfigLoader initialization."""

    def test_creates_loader_without_paths(self) -> None:
        loader = wire_config_loader()
        assert loader is not None

    def test_loads_from_yaml(self, tmp_path: Path) -> None:
        pytest.importorskip("yaml")
        import yaml

        config_file = tmp_path / "bizra.node.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "node": {"name": "test-node"},
                    "policy": {},
                }
            )
        )
        loader = wire_config_loader(config_paths=[config_file])
        assert loader is not None

    def test_missing_path_skipped(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.yaml"
        loader = wire_config_loader(config_paths=[missing])
        assert loader is not None  # no crash


# ═══════════════════════════════════════════════════════════
# CapsuleRuntime
# ═══════════════════════════════════════════════════════════


class TestWireCapsuleRuntime:
    """CapsuleRuntime initialization."""

    def test_creates_registry_and_runtime(self, tmp_path: Path) -> None:
        registry, runtime = wire_capsule_runtime(capsules_dir=tmp_path)
        assert registry is not None
        assert runtime is not None
        assert registry.count == 0  # empty dir

    def test_discovers_capsules(self, tmp_path: Path) -> None:
        pytest.importorskip("yaml")
        import yaml

        cap_dir = tmp_path / "my-cap"
        cap_dir.mkdir()
        (cap_dir / "CAPSULE.yaml").write_text(
            yaml.dump(
                {
                    "name": "my-cap",
                    "version": "1.0.0",
                    "capabilities": {"allow": ["file_read"]},
                    "workflow": [{"step": "s1", "action": "read", "args": {}}],
                }
            )
        )
        registry, runtime = wire_capsule_runtime(capsules_dir=tmp_path)
        assert registry.count == 1


# ═══════════════════════════════════════════════════════════
# OmegaLoopController
# ═══════════════════════════════════════════════════════════


class TestWireOmegaController:
    """OmegaLoopController initialization."""

    def test_creates_controller(self) -> None:
        controller = wire_omega_controller()
        assert controller is not None
        assert controller.active_loops == {}

    def test_creates_with_event_bus(self) -> None:
        eb = AsyncMock()
        controller = wire_omega_controller(event_bus=eb)
        assert controller is not None


# ═══════════════════════════════════════════════════════════
# wire_all — Full integration
# ═══════════════════════════════════════════════════════════


class TestWireAll:
    """Full bus wiring integration."""

    def test_wire_all_succeeds(self, tmp_path: Path) -> None:
        components, state = wire_all(
            state_dir=tmp_path,
            capsules_dir=tmp_path / "capsules",
        )
        assert state.topic_registry_ok
        assert state.action_bus_ok
        assert state.config_loader_ok
        assert state.capsule_runtime_ok
        assert state.omega_controller_ok
        assert state.all_ok
        assert len(state.errors) == 0

    def test_wire_all_returns_all_components(self, tmp_path: Path) -> None:
        components, _ = wire_all(
            state_dir=tmp_path,
            capsules_dir=tmp_path / "capsules",
        )
        assert "topic_registry" in components
        assert "action_bus" in components
        assert "telescript_engine" in components
        assert "config_loader" in components
        assert "capsule_registry" in components
        assert "capsule_runtime" in components
        assert "omega_controller" in components

    def test_wire_all_with_event_bus(self, tmp_path: Path) -> None:
        eb = AsyncMock()
        components, state = wire_all(
            event_bus=eb,
            state_dir=tmp_path,
            capsules_dir=tmp_path / "capsules",
        )
        assert state.all_ok


# ═══════════════════════════════════════════════════════════
# BusWiringState
# ═══════════════════════════════════════════════════════════


class TestBusWiringState:
    """Wiring state tracking."""

    def test_default_state_not_ok(self) -> None:
        state = BusWiringState()
        assert not state.all_ok

    def test_all_true_is_ok(self) -> None:
        state = BusWiringState(
            action_bus_ok=True,
            topic_registry_ok=True,
            config_loader_ok=True,
            capsule_runtime_ok=True,
            omega_controller_ok=True,
        )
        assert state.all_ok

    def test_summary_includes_errors(self) -> None:
        state = BusWiringState(errors=["test error"])
        summary = state.summary
        assert "errors" in summary
        assert "test error" in summary["errors"]

    def test_partial_failure_not_ok(self) -> None:
        state = BusWiringState(
            action_bus_ok=True,
            topic_registry_ok=True,
            config_loader_ok=False,
            capsule_runtime_ok=True,
            omega_controller_ok=True,
        )
        assert not state.all_ok
