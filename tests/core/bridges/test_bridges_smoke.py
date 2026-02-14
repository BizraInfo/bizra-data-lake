"""
Bridges Module Smoke Tests
============================
Validates that key bridge components can be instantiated
and their core interfaces respond correctly.

Created: 2026-02-07 | BIZRA Mastermind Sprint
"""

import pytest


# ============================================================================
# BridgeProtocol — Interface contract
# ============================================================================


class TestBridgeProtocol:
    """Bridge protocol must be importable from both bridges and protocols."""

    def test_from_bridges(self):
        from core.bridges import BridgeDirection, BridgeHealth, BridgeProtocol

        assert BridgeProtocol is not None
        assert BridgeHealth is not None
        assert BridgeDirection is not None

    def test_from_protocols(self):
        from core.protocols import BridgeDirection, BridgeHealth, BridgeProtocol

        assert BridgeProtocol is not None
        assert BridgeHealth is not None
        assert BridgeDirection is not None


# ============================================================================
# SovereignBridge — Main orchestration bridge
# ============================================================================


class TestSovereignBridge:
    """SovereignBridge must instantiate."""

    def test_import(self):
        from core.bridges import SovereignBridge

        assert SovereignBridge is not None


# ============================================================================
# RustLifecycle — Python <-> Rust bridge
# ============================================================================


class TestRustLifecycle:
    """Rust lifecycle bridge must provide process management."""

    def test_import(self):
        from core.bridges import (
            RustLifecycle,
            RustLifecycleManager,
            RustProcessManager,
        )

        assert RustLifecycle is not None
        assert RustLifecycleManager is not None
        assert RustProcessManager is not None


# ============================================================================
# KnowledgeIntegrator
# ============================================================================


class TestKnowledgeIntegrator:
    """Knowledge integrator must instantiate."""

    def test_import(self):
        from core.bridges import KnowledgeIntegrator

        assert KnowledgeIntegrator is not None


# ============================================================================
# LocalInferenceBridge
# ============================================================================


class TestLocalInferenceBridge:
    """Local inference bridge must instantiate."""

    def test_import(self):
        from core.bridges import LocalInferenceBridge

        assert LocalInferenceBridge is not None


# ============================================================================
# Iceoryx2Bridge — Zero-copy IPC
# ============================================================================


class TestIceoryx2Bridge:
    """Iceoryx2 bridge must instantiate."""

    def test_import(self):
        from core.bridges import Iceoryx2Bridge

        assert Iceoryx2Bridge is not None


# ============================================================================
# Lazy imports
# ============================================================================


class TestLazyImports:
    """Lazy-loaded bridge symbols must resolve."""

    def test_dual_agentic_bridge(self):
        from core.bridges import DualAgenticBridge

        assert DualAgenticBridge is not None

    def test_swarm_knowledge_bridge(self):
        from core.bridges import SwarmKnowledgeBridge

        assert SwarmKnowledgeBridge is not None


# ============================================================================
# __all__ completeness
# ============================================================================


def test_all_exports_resolvable():
    """Every name in __all__ must be accessible."""
    import core.bridges as mod

    for name in mod.__all__:
        attr = getattr(mod, name, None)
        assert attr is not None, f"core.bridges.__all__ exports '{name}' but it's None"
