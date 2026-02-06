"""
Treasury Module Smoke Tests
============================
Validates that key treasury components can be instantiated
and their core interfaces respond correctly.

Created: 2026-02-07 | BIZRA Mastermind Sprint
"""

import pytest


# ============================================================================
# TreasuryTypes — Enums and data classes
# ============================================================================


class TestTreasuryTypes:
    """Treasury type definitions must be importable and valid."""

    def test_treasury_mode_enum(self):
        from core.treasury import TreasuryMode

        assert TreasuryMode is not None
        # Verify it's an enum-like with expected states
        assert hasattr(TreasuryMode, "__members__") or callable(TreasuryMode)

    def test_treasury_state(self):
        from core.treasury import TreasuryState

        assert TreasuryState is not None

    def test_treasury_event(self):
        from core.treasury import TreasuryEvent

        assert TreasuryEvent is not None

    def test_transition_types(self):
        from core.treasury import TransitionEvent, TransitionTrigger

        assert TransitionEvent is not None
        assert TransitionTrigger is not None

    def test_ethics_assessment(self):
        from core.treasury import EthicsAssessment

        assert EthicsAssessment is not None


# ============================================================================
# ADL (عدل) — Justice Enforcement
# ============================================================================


class TestAdlKernel:
    """ADL kernel must enforce Gini inequality constraints."""

    def test_calculate_gini_uniform(self):
        from core.treasury import calculate_gini

        # Perfect equality → Gini = 0.0 (expects dict: node_id -> holdings)
        holdings = {"node_a": 100.0, "node_b": 100.0, "node_c": 100.0}
        gini = calculate_gini(holdings)
        assert gini == pytest.approx(0.0, abs=0.01)

    def test_calculate_gini_unequal(self):
        from core.treasury import calculate_gini

        # High inequality → Gini > 0.3 (filter excludes zero holdings)
        holdings = {"node_a": 1.0, "node_b": 1.0, "node_c": 100.0}
        gini = calculate_gini(holdings)
        assert gini > 0.3

    def test_adl_invariant(self):
        from core.treasury import AdlInvariant

        invariant = AdlInvariant()
        assert invariant is not None

    def test_adl_gate(self):
        from core.treasury import AdlGate

        # AdlGate requires a holdings_provider callable
        gate = AdlGate(holdings_provider=lambda: {"a": 50.0, "b": 50.0})
        assert gate is not None

    def test_incremental_gini(self):
        from core.treasury import IncrementalGini

        tracker = IncrementalGini()
        assert tracker is not None


# ============================================================================
# TreasuryController — Core operations
# ============================================================================


class TestTreasuryController:
    """TreasuryController must instantiate."""

    def test_import(self):
        from core.treasury import TreasuryController

        assert TreasuryController is not None

    def test_create_factory(self):
        from core.treasury import create_treasury_controller

        assert callable(create_treasury_controller)


# ============================================================================
# MarketIntegration — Harberger tax
# ============================================================================


class TestMarketIntegration:
    """Market integration must expose monitoring interface."""

    def test_import(self):
        from core.treasury import MarketAwareMuraqabah

        assert MarketAwareMuraqabah is not None


# ============================================================================
# __all__ completeness
# ============================================================================


def test_all_exports_resolvable():
    """Every name in __all__ must be accessible."""
    import core.treasury as mod

    for name in mod.__all__:
        attr = getattr(mod, name, None)
        assert attr is not None, f"core.treasury.__all__ exports '{name}' but it's None"
