"""Plugin runtime for BIZRA CLI spearpoint features.

This module provides a small, fail-closed plugin loader so front-door
capabilities (like onboarding) can be shipped as plugins without changing
core CLI orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol


class PluginRuntimeError(RuntimeError):
    """Raised when plugin registry or execution fails."""


@dataclass(frozen=True)
class PluginSpec:
    name: str
    module: str
    factory: str = "build_plugin"
    description: str = ""
    enabled: bool = True


@dataclass
class PluginContext:
    identity: Any = None
    kernel_factory: Optional[Callable[[], Any]] = None
    console: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BizraPlugin(Protocol):
    name: str

    def run(
        self, action: str, payload: Dict[str, Any], context: PluginContext
    ) -> Dict[str, Any]:
        ...


class PluginRuntime:
    """JSON-registry backed plugin loader for the main Python CLI."""

    def __init__(self, registry_path: Optional[str | Path] = None) -> None:
        env_path = os.getenv("BIZRA_PLUGIN_REGISTRY", "").strip()
        default_path = Path("config/plugins_registry.json")
        self.registry_path = Path(registry_path or env_path or default_path)
        self._specs: Dict[str, PluginSpec] = {}
        self._instances: Dict[str, BizraPlugin] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        if not self.registry_path.exists():
            raise PluginRuntimeError(
                f"Plugin registry not found: {self.registry_path}. "
                "Create config/plugins_registry.json or set BIZRA_PLUGIN_REGISTRY."
            )

        try:
            payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PluginRuntimeError(
                f"Invalid plugin registry JSON at {self.registry_path}: {exc}"
            ) from exc

        raw_plugins = payload.get("plugins", [])
        if not isinstance(raw_plugins, list):
            raise PluginRuntimeError("Registry field 'plugins' must be a list.")

        specs: Dict[str, PluginSpec] = {}
        for raw in raw_plugins:
            spec = PluginSpec(
                name=str(raw["name"]),
                module=str(raw["module"]),
                factory=str(raw.get("factory", "build_plugin")),
                description=str(raw.get("description", "")),
                enabled=bool(raw.get("enabled", True)),
            )
            specs[spec.name] = spec

        self._specs = specs

    def list_plugins(self) -> List[PluginSpec]:
        return sorted(self._specs.values(), key=lambda s: s.name)

    def load(self, name: str) -> BizraPlugin:
        if name in self._instances:
            return self._instances[name]

        spec = self._specs.get(name)
        if spec is None:
            raise PluginRuntimeError(f"Plugin not registered: {name}")
        if not spec.enabled:
            raise PluginRuntimeError(f"Plugin disabled: {name}")

        try:
            module = importlib.import_module(spec.module)
        except Exception as exc:
            raise PluginRuntimeError(
                f"Failed to import plugin module '{spec.module}' for '{name}': {exc}"
            ) from exc

        factory = getattr(module, spec.factory, None)
        if factory is None or not callable(factory):
            raise PluginRuntimeError(
                f"Plugin '{name}' factory '{spec.factory}' not found in {spec.module}"
            )

        plugin = factory()
        if not hasattr(plugin, "run"):
            raise PluginRuntimeError(f"Plugin '{name}' does not expose run().")
        self._instances[name] = plugin
        return plugin

    def invoke(
        self,
        name: str,
        action: str = "start",
        payload: Optional[Dict[str, Any]] = None,
        context: Optional[PluginContext] = None,
    ) -> Dict[str, Any]:
        plugin = self.load(name)
        payload = payload or {}
        context = context or PluginContext()
        try:
            result = plugin.run(action=action, payload=payload, context=context)
        except Exception as exc:
            raise PluginRuntimeError(
                f"Plugin '{name}' failed during action '{action}': {exc}"
            ) from exc

        if not isinstance(result, dict):
            raise PluginRuntimeError(
                f"Plugin '{name}' returned {type(result).__name__}; expected dict."
            )
        return result

