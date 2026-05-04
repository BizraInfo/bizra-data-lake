"""Regression tests for Rust event bus PyO3 import resolution."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def test_pyo3_bridge_import_ignores_scripts_bizra_shadow(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "bizra.py").write_text(
        "SHADOWED = True\n",
        encoding="utf-8",
    )

    pyo3_root = tmp_path / "pyo3"
    package_dir = pyo3_root / "bizra"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text(
        """
class PyEventBridge:
    def __init__(self, production=False):
        self.production = production

    def wire_subscribers(self):
        return 13

    def health(self):
        return {"active_subscriptions": 13, "system_ihsan": 1.0}
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(pyo3_root))
    monkeypatch.syspath_prepend(str(scripts_dir))
    sys.modules.pop("bizra", None)
    sys.modules.pop("core.sovereign.event_bus", None)

    event_bus = importlib.import_module("core.sovereign.event_bus")

    assert event_bus.is_rust_event_bus_available() is True
    bridge = event_bus.create_rust_event_bridge(production=False)
    assert bridge is not None
    assert bridge.wire() == 13
    assert bridge.health()["system_ihsan"] == 1.0
