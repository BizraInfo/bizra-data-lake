"""
Sovereign Bus Wiring — Connects Bus Infrastructure to Runtime
=============================================================

Bridges the Phase 69 bus modules (ActionBus, TopicRegistry, ConfigLoader,
CapsuleRuntime, OmegaLoopController) into the SovereignRuntime lifecycle.

Each component initializes with graceful fallback: if a dependency is
missing or fails, the runtime continues at reduced capability — never
crashes.

Standing on Giants:
- Fowler (2005): CQRS integration
- Thompson (1984): Capability-based system assembly
- Lamport (1978): Event ordering across subsystems

Phase 70.01 — Sovereign Synthesis Integration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BusWiringState:
    """Tracks which bus components are wired and healthy."""

    action_bus_ok: bool = False
    topic_registry_ok: bool = False
    config_loader_ok: bool = False
    capsule_runtime_ok: bool = False
    omega_controller_ok: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return (
            self.action_bus_ok
            and self.topic_registry_ok
            and self.config_loader_ok
            and self.capsule_runtime_ok
            and self.omega_controller_ok
        )

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "action_bus": self.action_bus_ok,
            "topic_registry": self.topic_registry_ok,
            "config_loader": self.config_loader_ok,
            "capsule_runtime": self.capsule_runtime_ok,
            "omega_controller": self.omega_controller_ok,
            "all_ok": self.all_ok,
            "errors": self.errors,
        }


def wire_topic_registry() -> Any:
    """Initialize the TopicRegistry with all 40 canonical topics."""
    from core.bus.topics import TopicRegistry

    registry = TopicRegistry()
    logger.info("TopicRegistry initialized: %d topics", len(registry.active_topics()))
    return registry


def wire_telescript_engine(
    policy_allow: frozenset[str] | None = None,
    policy_deny: frozenset[str] | None = None,
) -> Any:
    """Initialize TeleScript engine with node's capability policy."""
    from core.bus.telescript import Capability, TeleScriptEngine, TeleScriptPolicy

    if policy_allow is None:
        # Default node policy: allow all capabilities except shell_execute
        policy_allow = frozenset(
            c.value for c in Capability if c != Capability.SHELL_EXECUTE
        )
    if policy_deny is None:
        policy_deny = frozenset()

    policy = TeleScriptPolicy(allow=policy_allow, deny=policy_deny)
    engine = TeleScriptEngine(policy)
    logger.info(
        "TeleScript engine initialized: %d allowed, %d denied",
        len(policy_allow),
        len(policy_deny),
    )
    return engine


def wire_action_bus(
    event_bus: Any = None,
    telescript: Any = None,
    channels: dict[str, Any] | None = None,
    fate_gate: Any = None,
) -> Any:
    """Initialize ActionBus with TeleScript and optional FATE gate."""
    from core.bus.action_bus import ActionBus

    if telescript is None:
        telescript = wire_telescript_engine()

    bus = ActionBus(
        telescript=telescript,
        channels=channels or {},
        fate_gate=fate_gate,
        event_bus=event_bus,
    )
    logger.info("ActionBus initialized (channels=%d)", len(channels or {}))
    return bus


def wire_config_loader(
    config_paths: list[Path] | None = None,
) -> Any:
    """Initialize ConfigLoader with 3-scope config resolution."""
    from core.config.loader import ConfigLoader

    loader = ConfigLoader()

    # Load config files in priority order if provided
    if config_paths:
        for path in config_paths:
            if path.exists():
                try:
                    loader.load(path)
                    logger.info("Config loaded: %s", path)
                except Exception as exc:  # noqa: BLE001 — boundary boundary
                    logger.warning("Config load failed: %s — %s", path, exc)

    return loader


def wire_capsule_runtime(
    action_bus: Any = None,
    capsules_dir: Path | str = "capsules",
) -> tuple[Any, Any]:
    """Initialize CapsuleRegistry + CapsuleRuntime.

    Returns (registry, runtime) tuple.
    """
    from core.bus.capsule import CapsuleRegistry, CapsuleRuntime

    registry = CapsuleRegistry(capsules_dir)
    discovered = registry.discover()
    logger.info(
        "CapsuleRegistry: discovered %d capsules from %s", discovered, capsules_dir
    )

    runtime = CapsuleRuntime(registry, action_bus)
    return registry, runtime


def wire_omega_controller(
    action_bus: Any = None,
    event_bus: Any = None,
) -> Any:
    """Initialize OmegaLoopController with ActionBus and EventBus."""
    from core.bus.omega import OmegaLoopController

    controller = OmegaLoopController(
        action_bus=action_bus,
        event_bus=event_bus,
    )
    logger.info("OmegaLoopController initialized")
    return controller


def wire_all(
    event_bus: Any = None,
    state_dir: Path | str = "sovereign_state",
    config_paths: list[Path] | None = None,
    capsules_dir: Path | str = "capsules",
    fate_gate: Any = None,
) -> tuple[dict[str, Any], BusWiringState]:
    """Wire all bus components into a dict suitable for runtime injection.

    Returns:
        (components_dict, wiring_state)

    The components dict keys match SovereignRuntime attribute names
    (without the leading underscore).
    """
    state = BusWiringState()
    components: dict[str, Any] = {}

    # 1. TopicRegistry
    try:
        components["topic_registry"] = wire_topic_registry()
        state.topic_registry_ok = True
    except Exception as exc:  # noqa: BLE001 — boundary boundary
        state.errors.append(f"TopicRegistry: {exc}")
        logger.warning("TopicRegistry wiring failed: %s", exc)

    # 2. TeleScript + ActionBus
    try:
        telescript = wire_telescript_engine()
        components["telescript_engine"] = telescript
        components["action_bus"] = wire_action_bus(
            event_bus=event_bus,
            telescript=telescript,
            fate_gate=fate_gate,
        )
        state.action_bus_ok = True
    except Exception as exc:  # noqa: BLE001 — boundary boundary
        state.errors.append(f"ActionBus: {exc}")
        logger.warning("ActionBus wiring failed: %s", exc)

    # 3. ConfigLoader
    try:
        components["config_loader"] = wire_config_loader(config_paths)
        state.config_loader_ok = True
    except Exception as exc:  # noqa: BLE001 — boundary boundary
        state.errors.append(f"ConfigLoader: {exc}")
        logger.warning("ConfigLoader wiring failed: %s", exc)

    # 4. CapsuleRuntime
    try:
        registry, capsule_rt = wire_capsule_runtime(
            action_bus=components.get("action_bus"),
            capsules_dir=capsules_dir,
        )
        components["capsule_registry"] = registry
        components["capsule_runtime"] = capsule_rt
        state.capsule_runtime_ok = True
    except Exception as exc:  # noqa: BLE001 — boundary boundary
        state.errors.append(f"CapsuleRuntime: {exc}")
        logger.warning("CapsuleRuntime wiring failed: %s", exc)

    # 5. OmegaLoopController
    try:
        components["omega_controller"] = wire_omega_controller(
            action_bus=components.get("action_bus"),
            event_bus=event_bus,
        )
        state.omega_controller_ok = True
    except Exception as exc:  # noqa: BLE001 — boundary boundary
        state.errors.append(f"OmegaLoopController: {exc}")
        logger.warning("OmegaLoopController wiring failed: %s", exc)

    level = "INFO" if state.all_ok else "WARNING"
    msg = "Bus wiring complete: %s"
    getattr(logger, level.lower())(msg, state.summary)

    return components, state
