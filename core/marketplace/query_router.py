"""
Marketplace Query Router — Capability-Based Routing with Harberger Pricing

Routes incoming queries to the best-matching experts using
capability vectors and applies Harberger-based pricing.

Standing on Giants:
- Harberger (1962): COST — Common Ownership Self-assessed Tax
- Gini (1912): Inequality measurement for marketplace fairness
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.marketplace.expert_registry import (
    CapabilityVector,
    ExpertMatch,
    ExpertRegistry,
)

logger = logging.getLogger(__name__)

# Harberger tax rate for marketplace listings (annual, applied per-query)
HARBERGER_ANNUAL_RATE = 0.07  # 7% annual — aligned with constants.py

# Seconds in a year for pro-rating
SECONDS_PER_YEAR = 365.25 * 24 * 3600

# Maximum price multiplier
MAX_PRICE_MULTIPLIER = 10.0


@dataclass
class RoutingResult:
    """Result of routing a query through the marketplace."""

    success: bool
    query_vector: Optional[CapabilityVector] = None
    matches: List[ExpertMatch] = field(default_factory=list)
    selected_expert_id: Optional[str] = None
    final_price: float = 0.0
    harberger_tax_component: float = 0.0
    routing_time_ms: float = 0.0
    error: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "matches": [m.to_dict() for m in self.matches],
            "selected_expert_id": self.selected_expert_id,
            "final_price": round(self.final_price, 6),
            "harberger_tax_component": round(self.harberger_tax_component, 6),
            "routing_time_ms": round(self.routing_time_ms, 2),
            "error": self.error,
            "timestamp": self.timestamp,
        }


class PricingEngine:
    """
    Harberger-based pricing for marketplace queries.

    Experts self-assess their value. A continuous Harberger tax
    ensures prices stay honest — overpricing costs you in tax,
    underpricing costs you in being bought out.
    """

    def __init__(self, annual_rate: float = HARBERGER_ANNUAL_RATE):
        self._annual_rate = annual_rate

    def compute_query_price(
        self,
        self_assessed_value: float,
        similarity: float,
        query_duration_seconds: float = 10.0,
    ) -> Dict[str, float]:
        """
        Compute the price for a query.

        Price = base_cost + harberger_tax_component

        Where:
        - base_cost = self_assessed_value * similarity * (duration / year)
        - harberger_tax = self_assessed_value * annual_rate * (duration / year)

        Args:
            self_assessed_value: Expert's self-assessed value in SEED
            similarity: Query-expert similarity [0, 1]
            query_duration_seconds: Estimated query duration

        Returns:
            Dict with price breakdown
        """
        time_fraction = query_duration_seconds / SECONDS_PER_YEAR

        base_cost = self_assessed_value * similarity * time_fraction
        harberger_tax = self_assessed_value * self._annual_rate * time_fraction

        total = base_cost + harberger_tax

        return {
            "base_cost": base_cost,
            "harberger_tax": harberger_tax,
            "total": total,
            "time_fraction": time_fraction,
        }

    def compute_listing_tax(
        self,
        self_assessed_value: float,
        duration_seconds: float,
    ) -> float:
        """
        Compute Harberger tax for keeping a listing active.

        Tax = self_assessed_value * annual_rate * (duration / year)
        """
        time_fraction = duration_seconds / SECONDS_PER_YEAR
        return self_assessed_value * self._annual_rate * time_fraction


class MarketplaceRouter:
    """
    Routes queries through the expert marketplace.

    Combines capability matching with Harberger pricing and
    Gini-aware fairness constraints.
    """

    def __init__(
        self,
        registry: Optional[ExpertRegistry] = None,
        pricing: Optional[PricingEngine] = None,
        gini_threshold: float = 0.35,
    ):
        self._registry = registry or ExpertRegistry()
        self._pricing = pricing or PricingEngine()
        self._gini_threshold = gini_threshold
        self._total_queries_routed = 0

    @property
    def registry(self) -> ExpertRegistry:
        return self._registry

    @property
    def pricing(self) -> PricingEngine:
        return self._pricing

    @property
    def total_queries_routed(self) -> int:
        return self._total_queries_routed

    def route_query(
        self,
        query_capabilities: Dict[str, float],
        top_k: int = 5,
        auto_select: bool = True,
        tier_filter: Optional[str] = None,
        max_price: Optional[float] = None,
    ) -> RoutingResult:
        """
        Route a query to the best matching expert.

        Args:
            query_capabilities: Dict of capability_name -> required_score
            top_k: Maximum matches to consider
            auto_select: If True, automatically select the best match
            tier_filter: Optional tier filter
            max_price: Maximum price the requester will pay

        Returns:
            RoutingResult with matches and pricing
        """
        import time

        start = time.monotonic()

        query_vector = CapabilityVector.from_scores(query_capabilities)

        matches = self._registry.find_matches(
            query_vector=query_vector,
            top_k=top_k,
            tier_filter=tier_filter,
        )

        if not matches:
            elapsed_ms = (time.monotonic() - start) * 1000
            return RoutingResult(
                success=False,
                query_vector=query_vector,
                error="No matching experts found",
                routing_time_ms=elapsed_ms,
            )

        # Compute prices for each match
        for match in matches:
            pricing = self._pricing.compute_query_price(
                self_assessed_value=match.expert.self_assessed_value,
                similarity=match.similarity,
            )
            match.estimated_price = pricing["total"]

        # Apply max price filter
        if max_price is not None:
            matches = [m for m in matches if m.estimated_price <= max_price]
            if not matches:
                elapsed_ms = (time.monotonic() - start) * 1000
                return RoutingResult(
                    success=False,
                    query_vector=query_vector,
                    error=f"No experts within price limit {max_price}",
                    routing_time_ms=elapsed_ms,
                )

        selected_id = None
        final_price = 0.0
        harberger_component = 0.0

        if auto_select and matches:
            best = matches[0]
            selected_id = best.expert.expert_id
            pricing = self._pricing.compute_query_price(
                self_assessed_value=best.expert.self_assessed_value,
                similarity=best.similarity,
            )
            final_price = pricing["total"]
            harberger_component = pricing["harberger_tax"]

            # Update query count
            best.expert.total_queries_served += 1

        elapsed_ms = (time.monotonic() - start) * 1000
        self._total_queries_routed += 1

        return RoutingResult(
            success=True,
            query_vector=query_vector,
            matches=matches,
            selected_expert_id=selected_id,
            final_price=final_price,
            harberger_tax_component=harberger_component,
            routing_time_ms=elapsed_ms,
        )


__all__ = [
    "MarketplaceRouter",
    "PricingEngine",
    "RoutingResult",
    "HARBERGER_ANNUAL_RATE",
]
