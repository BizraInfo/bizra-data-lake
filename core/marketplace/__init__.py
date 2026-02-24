"""
BIZRA Expert Marketplace — Discovery, Matching, and Pricing

Provides a capability vector index for expert discovery and
Harberger-priced query routing.

Standing on Giants:
- Arnold Harberger (1962): Self-assessed value taxation
- Salton, Wong & Yang (1975): Vector space model for information retrieval
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
    RoutingResult,
)

__all__ = [
    "ExpertRegistry",
    "ExpertListing",
    "ExpertMatch",
    "CapabilityVector",
    "MarketplaceRouter",
    "PricingEngine",
    "RoutingResult",
]
