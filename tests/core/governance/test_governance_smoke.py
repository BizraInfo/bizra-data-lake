"""
Governance Module Smoke Tests
==============================
Validates that key governance components can be instantiated
and their core interfaces respond correctly.

Created: 2026-02-07 | BIZRA Mastermind Sprint
"""

import pytest


# ============================================================================
# ConstitutionalGate — Z3-proven synthesis admission
# ============================================================================


class TestConstitutionalGate:
    """Constitutional gate must validate against Ihsan constraint."""

    def test_import_and_create(self):
        from core.governance import ConstitutionalGate

        gate = ConstitutionalGate()
        assert gate is not None

    def test_admission_types(self):
        from core.governance import AdmissionResult, AdmissionStatus

        assert AdmissionStatus is not None
        assert AdmissionResult is not None


# ============================================================================
# AutonomyMatrix — Multi-level autonomous operation
# ============================================================================


class TestAutonomyMatrix:
    """Autonomy matrix must support level-based decision control."""

    def test_import_and_create(self):
        from core.governance import AutonomyMatrix

        matrix = AutonomyMatrix()
        assert matrix is not None

    def test_autonomy_types(self):
        from core.governance import AutonomyDecision, AutonomyLevel

        assert AutonomyLevel is not None
        assert AutonomyDecision is not None

    def test_autonomous_loop(self):
        from core.governance import AutonomousLoop, DecisionGate

        assert AutonomousLoop is not None
        assert DecisionGate is not None


# ============================================================================
# IhsanProjector — 8-dimensional excellence scoring
# ============================================================================


class TestIhsanProjector:
    """Ihsan projector must provide excellence scoring."""

    def test_import_and_create(self):
        from core.governance import IhsanProjector

        projector = IhsanProjector()
        assert projector is not None

    def test_ihsan_vector(self):
        from core.governance import IhsanDimension, IhsanVector

        assert IhsanVector is not None
        assert IhsanDimension is not None


# ============================================================================
# CapabilityCard — Model capability validation
# ============================================================================


class TestCapabilityCard:
    """Capability cards must define model tiers and task types."""

    def test_types(self):
        from core.governance import (
            CapabilityCard,
            CardIssuer,
            ModelCapabilities,
            ModelTier,
            TaskType,
        )

        assert CapabilityCard is not None
        assert ModelCapabilities is not None
        assert CardIssuer is not None
        assert ModelTier is not None
        assert TaskType is not None


# ============================================================================
# KeyRegistry — Trusted public key management
# ============================================================================


class TestKeyRegistry:
    """Key registry must support registration and lookup."""

    def test_import(self):
        from core.governance import (
            KeyStatus,
            RegisteredKey,
            TrustedKeyRegistry,
            get_key_registry,
        )

        assert TrustedKeyRegistry is not None
        assert RegisteredKey is not None
        assert KeyStatus is not None
        assert callable(get_key_registry)

    def test_get_registry(self):
        from core.governance import get_key_registry

        registry = get_key_registry()
        assert registry is not None


# ============================================================================
# ModelLicenseGate — Model capability validation chain
# ============================================================================


class TestModelLicenseGate:
    """Model license gate must instantiate."""

    def test_import(self):
        from core.governance import ModelLicenseGate

        assert ModelLicenseGate is not None


# ============================================================================
# __all__ completeness
# ============================================================================


def test_all_exports_resolvable():
    """Every name in __all__ must be accessible."""
    import core.governance as mod

    for name in mod.__all__:
        attr = getattr(mod, name, None)
        assert attr is not None, f"core.governance.__all__ exports '{name}' but it's None"
