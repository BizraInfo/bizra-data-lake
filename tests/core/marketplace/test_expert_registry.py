"""
Tests for Expert Marketplace — Registry and Query Routing
"""

from core.marketplace.expert_registry import (
    CapabilityVector,
    ExpertListing,
    ExpertMatch,
    ExpertRegistry,
)
from core.marketplace.query_router import (
    MarketplaceRouter,
    PricingEngine,
)


class TestCapabilityVector:
    """Test capability vector operations."""

    def test_cosine_similarity_identical(self):
        v1 = CapabilityVector(dimensions={"a": 1.0, "b": 0.0})
        v2 = CapabilityVector(dimensions={"a": 1.0, "b": 0.0})
        assert abs(v1.cosine_similarity(v2) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        v1 = CapabilityVector(dimensions={"a": 1.0, "b": 0.0})
        v2 = CapabilityVector(dimensions={"a": 0.0, "b": 1.0})
        assert abs(v1.cosine_similarity(v2)) < 1e-6

    def test_cosine_similarity_partial_overlap(self):
        v1 = CapabilityVector(dimensions={"reasoning": 0.8, "code": 0.6})
        v2 = CapabilityVector(dimensions={"reasoning": 0.9, "code": 0.1})
        sim = v1.cosine_similarity(v2)
        assert 0 < sim < 1

    def test_cosine_similarity_empty(self):
        v1 = CapabilityVector(dimensions={})
        v2 = CapabilityVector(dimensions={})
        assert v1.cosine_similarity(v2) == 0.0

    def test_magnitude(self):
        v = CapabilityVector(dimensions={"a": 0.6, "b": 0.8})
        assert abs(v.magnitude - 1.0) < 1e-6

    def test_values_clamped(self):
        v = CapabilityVector(dimensions={"a": 1.5, "b": -0.5})
        assert v.dimensions["a"] == 1.0
        assert v.dimensions["b"] == 0.0


class TestExpertRegistry:
    """Test the expert registry."""

    def _make_experts(self, count: int = 10) -> list:
        experts = []
        domains = ["reasoning", "code", "summarization", "chat", "analysis"]
        for i in range(count):
            caps = {}
            for j, d in enumerate(domains):
                caps[d] = ((i * 7 + j * 3) % 10) / 10.0  # Deterministic variety
            experts.append(
                ExpertListing(
                    expert_id=f"expert-{i}",
                    name=f"Expert {i}",
                    capabilities=CapabilityVector(dimensions=caps),
                    self_assessed_value=100.0 + i * 50.0,
                    node_id=f"BIZRA-{i:08X}",
                    tier=["EDGE", "LOCAL", "POOL"][i % 3],
                )
            )
        return experts

    def test_register_expert(self):
        registry = ExpertRegistry()
        expert = ExpertListing(
            expert_id="test-1",
            name="Test Expert",
            capabilities=CapabilityVector(dimensions={"reasoning": 0.9}),
            self_assessed_value=100.0,
        )
        assert registry.register(expert)
        assert registry.size == 1

    def test_deregister_expert(self):
        registry = ExpertRegistry()
        expert = ExpertListing(
            expert_id="test-1",
            name="Test Expert",
            capabilities=CapabilityVector(dimensions={"reasoning": 0.9}),
            self_assessed_value=100.0,
        )
        registry.register(expert)
        assert registry.deregister("test-1")
        assert registry.size == 0

    def test_reject_zero_value(self):
        registry = ExpertRegistry()
        expert = ExpertListing(
            expert_id="test-1",
            name="Free Expert",
            capabilities=CapabilityVector(dimensions={"reasoning": 0.9}),
            self_assessed_value=0.0,
        )
        assert not registry.register(expert)

    def test_find_matches(self):
        registry = ExpertRegistry()
        experts = self._make_experts(10)
        for e in experts:
            registry.register(e)

        query = CapabilityVector(dimensions={"reasoning": 1.0, "code": 0.5})
        matches = registry.find_matches(query, top_k=3)

        assert len(matches) <= 3
        assert all(isinstance(m, ExpertMatch) for m in matches)
        # Should be sorted by similarity (descending)
        for i in range(len(matches) - 1):
            assert matches[i].similarity >= matches[i + 1].similarity

    def test_find_matches_with_tier_filter(self):
        registry = ExpertRegistry()
        experts = self._make_experts(10)
        for e in experts:
            registry.register(e)

        query = CapabilityVector(dimensions={"reasoning": 1.0})
        matches = registry.find_matches(query, tier_filter="LOCAL")

        for m in matches:
            assert m.expert.tier == "LOCAL"

    def test_register_10_route_query_verify_best(self):
        """SAPE spec: register 10 experts, route query, verify best match."""
        registry = ExpertRegistry()
        experts = self._make_experts(10)
        for e in experts:
            registry.register(e)

        assert registry.size == 10

        query = CapabilityVector(dimensions={"reasoning": 0.9, "code": 0.7})
        matches = registry.find_matches(query, top_k=1)

        assert len(matches) == 1
        assert matches[0].similarity > 0.0
        assert matches[0].estimated_price > 0


class TestPricingEngine:
    """Test Harberger pricing."""

    def test_query_price(self):
        engine = PricingEngine(annual_rate=0.05)
        result = engine.compute_query_price(
            self_assessed_value=100.0,
            similarity=0.8,
            query_duration_seconds=10.0,
        )
        assert result["total"] > 0
        assert result["harberger_tax"] > 0
        assert result["total"] == result["base_cost"] + result["harberger_tax"]

    def test_listing_tax(self):
        engine = PricingEngine(annual_rate=0.05)
        tax = engine.compute_listing_tax(
            self_assessed_value=1000.0,
            duration_seconds=86400,  # 1 day
        )
        # 1000 * 0.05 * (86400 / 31557600) ≈ 0.1369
        assert 0.1 < tax < 0.2


class TestMarketplaceRouter:
    """Test the marketplace router."""

    def test_route_query_success(self):
        router = MarketplaceRouter()

        # Register experts
        for i in range(5):
            router.registry.register(
                ExpertListing(
                    expert_id=f"exp-{i}",
                    name=f"Expert {i}",
                    capabilities=CapabilityVector(
                        dimensions={
                            "reasoning": 0.5 + i * 0.1,
                            "code": 0.3 + i * 0.05,
                        }
                    ),
                    self_assessed_value=100.0 + i * 20.0,
                )
            )

        result = router.route_query({"reasoning": 0.9, "code": 0.5})
        assert result.success
        assert result.selected_expert_id is not None
        assert result.final_price > 0
        assert len(result.matches) > 0

    def test_route_empty_registry(self):
        router = MarketplaceRouter()
        result = router.route_query({"reasoning": 1.0})
        assert not result.success
        assert "No matching" in result.error

    def test_route_with_max_price(self):
        router = MarketplaceRouter()
        router.registry.register(
            ExpertListing(
                expert_id="exp-expensive",
                name="Expensive Expert",
                capabilities=CapabilityVector(dimensions={"reasoning": 0.9}),
                self_assessed_value=1_000_000.0,
            )
        )

        result = router.route_query(
            {"reasoning": 0.9},
            max_price=0.000001,
        )
        assert not result.success

    def test_total_queries_increments(self):
        router = MarketplaceRouter()
        router.registry.register(
            ExpertListing(
                expert_id="exp-1",
                name="Expert 1",
                capabilities=CapabilityVector(dimensions={"reasoning": 0.9}),
                self_assessed_value=100.0,
            )
        )

        assert router.total_queries_routed == 0
        router.route_query({"reasoning": 0.9})
        assert router.total_queries_routed == 1
