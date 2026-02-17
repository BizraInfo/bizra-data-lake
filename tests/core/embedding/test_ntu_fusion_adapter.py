"""
Tests for NTUFusionAdapter — NTU temporal state enrichment for CognitiveFusion.

Covers:
- Context enrichment with NTU state
- High entropy triggers retrieval depth multiplier
- Graceful behavior when bridge is None
- Pattern detection passthrough

Standing on Giants: Takens (1981, embedding theorem) + Shannon (1948, entropy)
Artifact: core/ntu/bridge.py :: NTUFusionAdapter
"""

from __future__ import annotations

import pytest

# NTU requires numpy — skip all tests if unavailable
np = pytest.importorskip("numpy")


class TestNTUFusionAdapterEnrichment:
    """NTUFusionAdapter.enrich_context() behavior."""

    def test_enriches_context_with_ntu_state(self):
        """Context receives ntu_state dict with belief, entropy, potential, iteration."""
        from core.ntu import NTUBridge, NTUFusionAdapter

        bridge = NTUBridge()
        adapter = NTUFusionAdapter(ntu_bridge=bridge)

        context: dict = {"existing_key": "value"}
        enriched = adapter.enrich_context(context)

        assert "ntu_state" in enriched
        ntu = enriched["ntu_state"]
        assert "belief" in ntu
        assert "entropy" in ntu
        assert "potential" in ntu
        assert "iteration" in ntu
        assert "pattern" in ntu

        # All values should be numeric
        assert isinstance(ntu["belief"], float)
        assert isinstance(ntu["entropy"], float)
        assert isinstance(ntu["potential"], float)
        assert isinstance(ntu["iteration"], int)

    def test_preserves_existing_context(self):
        """Enrichment adds to context without removing existing keys."""
        from core.ntu import NTUBridge, NTUFusionAdapter

        bridge = NTUBridge()
        adapter = NTUFusionAdapter(ntu_bridge=bridge)

        original = {"query_type": "search", "user_id": "u42"}
        enriched = adapter.enrich_context(original)

        assert enriched["query_type"] == "search"
        assert enriched["user_id"] == "u42"
        assert "ntu_state" in enriched

    def test_high_entropy_increases_retrieval_depth(self):
        """When NTU entropy > 0.7, retrieval_depth_multiplier is set to 2.0."""
        from core.ntu import NTUBridge, NTUFusionAdapter

        bridge = NTUBridge()
        adapter = NTUFusionAdapter(ntu_bridge=bridge)

        # Feed diverse observations to increase entropy
        for i in range(20):
            value = 0.1 if i % 2 == 0 else 0.9  # alternating → high entropy
            bridge._unified_ntu.observe(value)

        state = bridge.get_state()
        context: dict = {}
        enriched = adapter.enrich_context(context)

        # If entropy is high enough, retrieval depth should be set
        if state.entropy > 0.7:
            assert enriched.get("retrieval_depth_multiplier") == 2.0
        elif state.entropy > 0.4:
            assert enriched.get("retrieval_depth_multiplier") == 1.5

    def test_medium_entropy_moderate_retrieval_depth(self):
        """When NTU entropy is between 0.4 and 0.7, depth multiplier is 1.5."""
        from core.ntu import NTUBridge, NTUFusionAdapter

        bridge = NTUBridge()
        adapter = NTUFusionAdapter(ntu_bridge=bridge)

        # Feed moderately diverse observations
        for i in range(10):
            bridge._unified_ntu.observe(0.4 + (i * 0.02))

        state = bridge.get_state()
        context: dict = {}
        enriched = adapter.enrich_context(context)

        # Verify the logic matches the thresholds
        if state.entropy > 0.7:
            assert enriched.get("retrieval_depth_multiplier") == 2.0
        elif state.entropy > 0.4:
            assert enriched.get("retrieval_depth_multiplier") == 1.5
        # If entropy <= 0.4, no multiplier set

    def test_does_not_overwrite_existing_depth_multiplier(self):
        """setdefault ensures existing multiplier is preserved."""
        from core.ntu import NTUBridge, NTUFusionAdapter

        bridge = NTUBridge()
        adapter = NTUFusionAdapter(ntu_bridge=bridge)

        # Pre-set a custom multiplier
        context: dict = {"retrieval_depth_multiplier": 3.0}
        enriched = adapter.enrich_context(context)

        # Original value should be preserved
        assert enriched["retrieval_depth_multiplier"] == 3.0


class TestNTUFusionAdapterNoBridge:
    """Behavior when bridge is None."""

    def test_returns_context_unchanged_when_no_bridge(self):
        """If bridge is None, context passes through unmodified."""
        from core.ntu import NTUFusionAdapter

        adapter = NTUFusionAdapter(ntu_bridge=None)

        context = {"key": "value"}
        result = adapter.enrich_context(context)

        assert result == {"key": "value"}
        assert "ntu_state" not in result

    def test_bridge_can_be_set_later(self):
        """Bridge can be assigned after construction via property."""
        from core.ntu import NTUBridge, NTUFusionAdapter

        adapter = NTUFusionAdapter(ntu_bridge=None)
        assert adapter.bridge is None

        bridge = NTUBridge()
        adapter.bridge = bridge
        assert adapter.bridge is bridge

        # Now enrichment should work
        result = adapter.enrich_context({})
        assert "ntu_state" in result


class TestNTUFusionAdapterIntegration:
    """Integration with NTU observation pipeline."""

    def test_state_evolves_with_observations(self):
        """After processing observations, NTU state reflects the data."""
        from core.ntu import NTUBridge, NTUFusionAdapter

        bridge = NTUBridge()
        adapter = NTUFusionAdapter(ntu_bridge=bridge)

        # Initial state
        ctx1 = adapter.enrich_context({})
        initial_iteration = ctx1["ntu_state"]["iteration"]

        # Feed observations
        for _ in range(5):
            bridge._unified_ntu.observe(0.8)

        # State should have advanced
        ctx2 = adapter.enrich_context({})
        assert ctx2["ntu_state"]["iteration"] > initial_iteration

    def test_belief_converges_with_consistent_signal(self):
        """Consistent high-quality signal should drive belief upward."""
        from core.ntu import NTUBridge, NTUFusionAdapter

        bridge = NTUBridge()
        adapter = NTUFusionAdapter(ntu_bridge=bridge)

        # Feed consistently high values
        for _ in range(30):
            bridge._unified_ntu.observe(0.9)

        ctx = adapter.enrich_context({})
        # With 30 consistent high observations, belief should be elevated
        assert ctx["ntu_state"]["belief"] > 0.5
