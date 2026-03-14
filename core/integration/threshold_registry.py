"""ThresholdRegistry — Constitutional Enforcement Singleton.

Sealed-after-boot runtime authority for all BIZRA constitutional thresholds.
Prevents threshold drift by providing a single, immutable source for all
Ihsān, SNR, ADL, and gate values.

Standing on Giants: Lamport (Byzantine agreement) · Al-Ghazali (Ihsān, 1095)
                    Singleton: GoF (1994) · Immutability: Rich Hickey (2007)

Architecture:
    ┌─────────────────────────────────────────────────┐
    │           ThresholdRegistry (singleton)          │
    │                                                  │
    │  BOOT ──► register() ──► seal() ──► get() only  │
    │                                                  │
    │  After seal():                                   │
    │    • register() raises SealedRegistryError        │
    │    • get() returns Final values                   │
    │    • audit() scans for module-level shadows       │
    └─────────────────────────────────────────────────┘

Usage:
    from core.integration.threshold_registry import registry

    # At module level (read-only after boot):
    ihsan = registry.get("UNIFIED_IHSAN_THRESHOLD")  # 0.95
    snr = registry.get("UNIFIED_SNR_THRESHOLD")      # 0.85

    # Gate check:
    if score >= registry.get("IHSAN_GATE_MINIMUM"):
        ...
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Final, Optional


class SealedRegistryError(RuntimeError):
    """Raised when attempting to modify a sealed ThresholdRegistry."""


class ThresholdNotFoundError(KeyError):
    """Raised when requesting an unregistered threshold."""


@dataclass
class ThresholdEntry:
    """A single constitutional threshold with provenance."""

    name: str
    value: float
    category: str  # "ihsan" | "snr" | "adl" | "gate" | "timing" | "economic"
    source: str = "constants.py"
    constitutional: bool = True
    description: str = ""


class ThresholdRegistry:
    """Sealed-after-boot singleton for constitutional threshold enforcement.

    Properties:
        P1 (Uniqueness):  Only one registry instance exists per process.
        P2 (Immutability): After seal(), no threshold can be added or modified.
        P3 (Completeness): All canonical thresholds from constants.py are loaded.
        P4 (Auditability): audit() detects module-level threshold shadows.
    """

    _instance: Optional[ThresholdRegistry] = None
    _lock: Final[threading.Lock] = threading.Lock()

    def __new__(cls) -> ThresholdRegistry:
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._thresholds = {}
                instance._sealed = False
                instance._seal_lock = threading.Lock()
                cls._instance = instance
            return cls._instance

    def register(self, name: str, value: float, category: str = "general",
                 constitutional: bool = True, description: str = "") -> None:
        """Register a threshold. Raises SealedRegistryError if sealed."""
        if self._sealed:
            raise SealedRegistryError(
                f"Cannot register '{name}': registry is sealed after boot"
            )
        self._thresholds[name] = ThresholdEntry(
            name=name,
            value=value,
            category=category,
            constitutional=constitutional,
            description=description,
        )

    def seal(self) -> None:
        """Seal the registry — no further modifications allowed."""
        with self._seal_lock:
            self._sealed = True

    @property
    def is_sealed(self) -> bool:
        return self._sealed

    def get(self, name: str) -> float:
        """Get a threshold value. Raises ThresholdNotFoundError if missing."""
        entry = self._thresholds.get(name)
        if entry is None:
            raise ThresholdNotFoundError(
                f"Threshold '{name}' not registered. "
                f"Available: {sorted(self._thresholds.keys())}"
            )
        return entry.value

    def get_entry(self, name: str) -> ThresholdEntry:
        """Get full threshold entry with metadata."""
        entry = self._thresholds.get(name)
        if entry is None:
            raise ThresholdNotFoundError(f"Threshold '{name}' not registered")
        return entry

    def get_or_default(self, name: str, default: float) -> float:
        """Get a threshold value, returning default if not found."""
        entry = self._thresholds.get(name)
        return entry.value if entry is not None else default

    def has(self, name: str) -> bool:
        """Check if a threshold is registered."""
        return name in self._thresholds

    @property
    def count(self) -> int:
        return len(self._thresholds)

    def all_thresholds(self) -> dict[str, float]:
        """Return all thresholds as a name→value dict (read-only snapshot)."""
        return {name: entry.value for name, entry in self._thresholds.items()}

    def by_category(self, category: str) -> dict[str, float]:
        """Return all thresholds in a category."""
        return {
            name: entry.value
            for name, entry in self._thresholds.items()
            if entry.category == category
        }

    def audit_module_shadows(self) -> list[dict[str, Any]]:
        """Scan core/ modules for threshold shadows not imported from constants.

        Returns list of shadow findings:
            [{"module": "core.autonomous", "name": "SNR_THRESHOLDS", "line": 81, ...}]

        This is a STATIC audit — safe to call at any time.
        """
        import ast
        import os
        from pathlib import Path

        shadows: list[dict[str, Any]] = []
        core_dir = Path(__file__).resolve().parent.parent  # core/
        constants_path = str(Path(__file__).resolve().parent / "constants.py")

        # Patterns that indicate a threshold definition (not import)
        threshold_patterns = {
            "IHSAN", "SNR_THRESHOLD", "ADL_GINI", "GATE_MINIMUM",
        }

        for root, _dirs, files in os.walk(core_dir):
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                fpath = os.path.join(root, fname)
                # Skip constants.py itself and this file
                if os.path.abspath(fpath) in (
                    constants_path,
                    os.path.abspath(__file__),
                ):
                    continue

                try:
                    source = Path(fpath).read_text(encoding="utf-8")
                    tree = ast.parse(source, filename=fpath)
                except (SyntaxError, UnicodeDecodeError):
                    continue

                for node in ast.walk(tree):
                    if not isinstance(node, ast.Assign):
                        continue
                    for target in node.targets:
                        if not isinstance(target, ast.Name):
                            continue
                        name = target.id
                        # Check if this looks like a threshold definition
                        if not any(pat in name for pat in threshold_patterns):
                            continue
                        # Check if it's a numeric assignment (shadow)
                        if isinstance(node.value, (ast.Constant, ast.Dict)):
                            rel_path = os.path.relpath(fpath, core_dir.parent)
                            shadows.append({
                                "file": rel_path.replace("\\", "/"),
                                "name": name,
                                "line": node.lineno,
                                "type": "numeric_shadow" if isinstance(
                                    node.value, ast.Constant
                                ) else "dict_shadow",
                            })

        return shadows

    def validate_against_canonical(self) -> list[dict[str, Any]]:
        """Validate registry values against CANONICAL_THRESHOLDS from constants.

        Returns list of drift findings.
        """
        from core.integration.constants import CANONICAL_THRESHOLDS

        drifts: list[dict[str, Any]] = []
        for name, expected in CANONICAL_THRESHOLDS.items():
            if not self.has(name):
                drifts.append({
                    "name": name,
                    "expected": expected,
                    "actual": None,
                    "status": "missing",
                })
                continue
            actual = self.get(name)
            if abs(actual - expected) > 1e-9:
                drifts.append({
                    "name": name,
                    "expected": expected,
                    "actual": actual,
                    "status": "drift",
                })
        return drifts

    @classmethod
    def _reset_for_testing(cls) -> None:
        """Reset singleton state — TEST USE ONLY."""
        with cls._lock:
            cls._instance = None

    def __repr__(self) -> str:
        state = "sealed" if self._sealed else "open"
        return f"<ThresholdRegistry [{state}] {self.count} thresholds>"


def _boot_registry() -> ThresholdRegistry:
    """Load all canonical thresholds from constants.py and seal.

    Called once at module import time. After this, the registry is immutable.
    """
    from core.integration import constants as C

    reg = ThresholdRegistry()

    # If already sealed (re-import), return as-is
    if reg.is_sealed:
        return reg

    # ── Ihsān thresholds ──
    for name, value, desc in [
        ("UNIFIED_IHSAN_THRESHOLD", C.UNIFIED_IHSAN_THRESHOLD, "Production excellence"),
        ("IHSAN_THRESHOLD", C.IHSAN_THRESHOLD, "Backward compat alias"),
        ("STRICT_IHSAN_THRESHOLD", C.STRICT_IHSAN_THRESHOLD, "Consensus-critical"),
        ("RUNTIME_IHSAN_THRESHOLD", C.RUNTIME_IHSAN_THRESHOLD, "Z3-proven only"),
        ("RUNTIME_IHSAN", C.RUNTIME_IHSAN_THRESHOLD, "Canonical key alias"),
        ("IHSAN_THRESHOLD_PRODUCTION", C.IHSAN_THRESHOLD_PRODUCTION, "Production env"),
        ("IHSAN_THRESHOLD_CI", C.IHSAN_THRESHOLD_CI, "CI environment"),
        ("IHSAN_THRESHOLD_DEV", C.IHSAN_THRESHOLD_DEV, "Development env"),
        ("IHSAN_GATE_MINIMUM", C.IHSAN_GATE_MINIMUM, "Hard floor fail-closed"),
        ("IHSAN_POI_CONSENSUS", C.IHSAN_POI_CONSENSUS, "PoI attestation minimum"),
        ("IHSAN_BLOOM_ELIGIBILITY", C.IHSAN_BLOOM_ELIGIBILITY, "BLOOM minting gate"),
        ("IHSAN_CONFORMANCE_JOIN", C.IHSAN_CONFORMANCE_JOIN, "Network join gate"),
        ("REFLEX_PRECIPITATION_IHSAN", C.REFLEX_PRECIPITATION_IHSAN, "Cache precipitation"),
    ]:
        reg.register(name, value, category="ihsan", description=desc)

    # ── SNR thresholds ──
    for name, value, desc in [
        ("UNIFIED_SNR_THRESHOLD", C.UNIFIED_SNR_THRESHOLD, "Base minimum"),
        ("SNR_THRESHOLD", C.SNR_THRESHOLD, "Backward compat alias"),
        ("SNR_THRESHOLD_MINIMUM", C.UNIFIED_SNR_THRESHOLD, "Canonical minimum key"),
        ("MUSEUM_SNR_FLOOR", C.MUSEUM_SNR_FLOOR, "Museum floor (Pillar 2)"),
        ("SNR_THRESHOLD_T0_ELITE", C.SNR_THRESHOLD_T0_ELITE, "T0 elite tier"),
        ("SNR_THRESHOLD_T1_HIGH", C.SNR_THRESHOLD_T1_HIGH, "T1 high tier"),
        ("SNR_THRESHOLD_T2_STANDARD", C.SNR_THRESHOLD_T2_STANDARD, "T2 standard tier"),
        ("SNR_THRESHOLD_T3_ACCEPTABLE", C.SNR_THRESHOLD_T3_ACCEPTABLE, "T3 acceptable"),
        ("SNR_THRESHOLD_T4_MINIMUM", C.SNR_THRESHOLD_T4_MINIMUM, "T4 minimum"),
    ]:
        reg.register(name, value, category="snr", description=desc)

    # ── ADL (Justice) thresholds ──
    for name, value, desc in [
        ("ADL_GINI_THRESHOLD", C.ADL_GINI_THRESHOLD, "Hard gate operational"),
        ("ADL_GINI_EMERGENCY", C.ADL_GINI_EMERGENCY, "System-wide freeze"),
        ("ADL_HARBERGER_TAX_RATE", C.ADL_HARBERGER_TAX_RATE, "Annual Harberger"),
        ("ADL_MINIMUM_HOLDING", C.ADL_MINIMUM_HOLDING, "Dust attack prevention"),
    ]:
        reg.register(name, value, category="adl", description=desc)

    # ── Gate thresholds ──
    reg.register(
        "PREDICTION_VERIFICATION_AGREEMENT_THRESHOLD",
        C.PREDICTION_VERIFICATION_AGREEMENT_THRESHOLD,
        category="gate",
        description="HMM-FATE agreement minimum",
    )

    # ── Confidence thresholds ──
    for name, value, desc in [
        ("CONFIDENCE_HIGH", C.CONFIDENCE_HIGH, "High confidence"),
        ("CONFIDENCE_MEDIUM", C.CONFIDENCE_MEDIUM, "Medium confidence"),
        ("CONFIDENCE_LOW", C.CONFIDENCE_LOW, "Low confidence"),
        ("CONFIDENCE_MINIMUM", C.CONFIDENCE_MINIMUM, "Minimum confidence"),
    ]:
        reg.register(name, value, category="confidence", description=desc)

    reg.seal()
    return reg


# ── Module-level singleton ──
registry: ThresholdRegistry = _boot_registry()
