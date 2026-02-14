"""
Reasoning Module Smoke Tests
==============================
Validates that key reasoning components can be instantiated
and their core interfaces respond correctly.

Created: 2026-02-07 | BIZRA Mastermind Sprint
"""

import pytest


# ============================================================================
# GraphOfThoughts — Core reasoning engine
# ============================================================================


class TestGraphOfThoughts:
    """GraphOfThoughts must compose mixins and support basic operations."""

    def test_import_and_create(self):
        from core.reasoning import GraphOfThoughts

        got = GraphOfThoughts()
        assert got is not None

    def test_graph_types(self):
        from core.reasoning import (
            EdgeType,
            ReasoningPath,
            ReasoningResult,
            ReasoningStrategy,
            ThoughtEdge,
            ThoughtNode,
            ThoughtType,
        )

        # Verify all enum/data types exist
        assert ThoughtType is not None
        assert EdgeType is not None
        assert ReasoningStrategy is not None
        assert ThoughtNode is not None
        assert ThoughtEdge is not None
        assert ReasoningPath is not None
        assert ReasoningResult is not None

    def test_mixins_present(self):
        from core.reasoning import (
            GraphOperationsMixin,
            GraphReasoningMixin,
            GraphSearchMixin,
        )

        assert GraphOperationsMixin is not None
        assert GraphSearchMixin is not None
        assert GraphReasoningMixin is not None


# ============================================================================
# GuardianCouncil — Multi-perspective validation
# ============================================================================


class TestGuardianCouncil:
    """Guardian council must validate against Ihsan constraint."""

    def test_import_and_create(self):
        from core.reasoning import GuardianCouncil

        council = GuardianCouncil()
        assert council is not None

    def test_guardian(self):
        from core.reasoning import CouncilVerdict, Guardian

        assert Guardian is not None
        assert CouncilVerdict is not None


# ============================================================================
# SNRMaximizer — Shannon-inspired quality optimization
# ============================================================================


class TestSNRMaximizer:
    """SNR maximizer must import cleanly."""

    def test_import_and_create(self):
        from core.reasoning import SNRMaximizer

        maximizer = SNRMaximizer()
        assert maximizer is not None


# ============================================================================
# BicameralEngine — Dual-hemisphere reasoning
# ============================================================================


class TestBicameralEngine:
    """Bicameral engine must instantiate."""

    def test_import_and_create(self):
        from core.reasoning import BicameralReasoningEngine

        engine = BicameralReasoningEngine()
        assert engine is not None


# ============================================================================
# Lazy imports — Collective intelligence
# ============================================================================


class TestLazyImports:
    """Lazy-loaded collective symbols must resolve."""

    def test_collective_intelligence(self):
        from core.reasoning import CollectiveIntelligence

        assert CollectiveIntelligence is not None

    def test_collective_synthesizer(self):
        from core.reasoning import CollectiveSynthesizer

        assert CollectiveSynthesizer is not None


# ============================================================================
# __all__ completeness
# ============================================================================


def test_all_exports_resolvable():
    """Every name in __all__ must be accessible."""
    import core.reasoning as mod

    for name in mod.__all__:
        attr = getattr(mod, name, None)
        assert attr is not None, f"core.reasoning.__all__ exports '{name}' but it's None"
