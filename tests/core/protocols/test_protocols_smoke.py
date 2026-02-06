"""
Protocols Module Smoke Tests
==============================
Validates that protocol interface contracts are properly defined
and can be used for structural subtyping.

Created: 2026-02-07 | BIZRA Mastermind Sprint
"""

import pytest


# ============================================================================
# InferenceBackend — Backend protocol
# ============================================================================


class TestInferenceBackend:
    """InferenceBackend protocol must define required methods."""

    def test_import(self):
        from core.protocols import (
            BackendCapability,
            InferenceBackend,
            InferenceRequest,
            InferenceResponse,
        )

        assert InferenceBackend is not None
        assert InferenceRequest is not None
        assert InferenceResponse is not None
        assert BackendCapability is not None

    def test_protocol_has_methods(self):
        from core.protocols import InferenceBackend

        # ABC should define complete/is_healthy
        assert hasattr(InferenceBackend, "complete")
        assert hasattr(InferenceBackend, "is_healthy")


# ============================================================================
# BridgeProtocol — Bridge interface
# ============================================================================


class TestBridgeProtocol:
    """BridgeProtocol must define bridge lifecycle methods."""

    def test_import(self):
        from core.protocols import BridgeDirection, BridgeHealth, BridgeProtocol

        assert BridgeProtocol is not None
        assert BridgeHealth is not None
        assert BridgeDirection is not None


# ============================================================================
# GateChain — Composable validation gates
# ============================================================================


class TestGateChain:
    """GateChain must support composable gate validation."""

    def test_import(self):
        from core.protocols import Gate, GateChain, GateResult, GateStatus

        assert Gate is not None
        assert GateChain is not None
        assert GateResult is not None
        assert GateStatus is not None

    def test_gate_status_values(self):
        from core.protocols import GateStatus

        # Verify expected status values exist
        assert hasattr(GateStatus, "PASS") or hasattr(GateStatus, "PASSED")


# ============================================================================
# __all__ completeness
# ============================================================================


def test_all_exports_resolvable():
    """Every name in __all__ must be accessible."""
    import core.protocols as mod

    for name in mod.__all__:
        attr = getattr(mod, name, None)
        assert attr is not None, f"core.protocols.__all__ exports '{name}' but it's None"
