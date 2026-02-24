"""
Expert Registry — Capability Vector Index for Expert Discovery

Maintains a registry of expert models/agents with their capability
vectors, enabling cosine-similarity matching for incoming queries.

Standing on Giants:
- Salton, Wong & Yang (1975): Vector Space Model
- Deerwester et al. (1990): Latent Semantic Indexing
- Harberger (1962): Self-assessed value taxation for pricing
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default capability dimensions
DEFAULT_DIMENSIONS = [
    "reasoning",
    "code_generation",
    "summarization",
    "translation",
    "classification",
    "embedding",
    "chat",
    "analysis",
]

# Maximum experts in the registry
MAX_REGISTRY_SIZE = 10_000

# Minimum similarity for a valid match
MIN_MATCH_SIMILARITY = 0.1


@dataclass
class CapabilityVector:
    """
    A normalized vector representing expert capabilities.

    Each dimension corresponds to a capability domain, and the
    value represents proficiency (0.0 to 1.0).
    """

    dimensions: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        # Clamp values to [0, 1]
        self.dimensions = {k: max(0.0, min(1.0, v)) for k, v in self.dimensions.items()}

    @property
    def magnitude(self) -> float:
        """L2 magnitude of the vector."""
        return math.sqrt(sum(v * v for v in self.dimensions.values()))

    def cosine_similarity(self, other: "CapabilityVector") -> float:
        """
        Compute cosine similarity with another capability vector.

        Returns value in [-1, 1], but practically [0, 1] since
        all dimensions are non-negative.
        """
        # Find common dimensions
        all_dims = set(self.dimensions.keys()) | set(other.dimensions.keys())
        if not all_dims:
            return 0.0

        dot = 0.0
        mag_a = 0.0
        mag_b = 0.0

        for dim in all_dims:
            a = self.dimensions.get(dim, 0.0)
            b = other.dimensions.get(dim, 0.0)
            dot += a * b
            mag_a += a * a
            mag_b += b * b

        mag_a = math.sqrt(mag_a)
        mag_b = math.sqrt(mag_b)

        if mag_a == 0 or mag_b == 0:
            return 0.0

        return dot / (mag_a * mag_b)

    def to_dict(self) -> Dict[str, float]:
        return dict(self.dimensions)

    @classmethod
    def from_scores(cls, scores: Dict[str, float]) -> "CapabilityVector":
        """Create from a dictionary of capability scores."""
        return cls(dimensions=scores)


@dataclass
class ExpertListing:
    """A registered expert in the marketplace."""

    expert_id: str
    name: str
    capabilities: CapabilityVector
    self_assessed_value: float  # Harberger self-assessed value (SEED)
    node_id: str = ""  # BIZRA node hosting this expert
    tier: str = "LOCAL"  # EDGE, LOCAL, POOL
    availability: float = 1.0  # 0.0 to 1.0
    total_queries_served: int = 0
    average_rating: float = 0.0
    registered_at: str = ""

    def __post_init__(self):
        if not self.registered_at:
            self.registered_at = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expert_id": self.expert_id,
            "name": self.name,
            "capabilities": self.capabilities.to_dict(),
            "self_assessed_value": self.self_assessed_value,
            "node_id": self.node_id,
            "tier": self.tier,
            "availability": self.availability,
            "total_queries_served": self.total_queries_served,
            "average_rating": self.average_rating,
            "registered_at": self.registered_at,
        }


@dataclass
class ExpertMatch:
    """Result of matching a query to an expert."""

    expert: ExpertListing
    similarity: float
    estimated_price: float  # SEED cost for this query
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expert_id": self.expert.expert_id,
            "name": self.expert.name,
            "similarity": round(self.similarity, 4),
            "estimated_price": round(self.estimated_price, 4),
            "tier": self.expert.tier,
            "rank": self.rank,
        }


class ExpertRegistry:
    """
    Registry of available experts with capability vector indexing.

    Supports:
    - Registration and deregistration
    - Cosine-similarity based query matching
    - Top-k retrieval
    - Harberger value enforcement
    """

    def __init__(self):
        self._experts: Dict[str, ExpertListing] = {}

    @property
    def size(self) -> int:
        return len(self._experts)

    def register(self, listing: ExpertListing) -> bool:
        """
        Register an expert in the marketplace.

        Args:
            listing: Expert listing to register

        Returns:
            True if registered successfully
        """
        if self.size >= MAX_REGISTRY_SIZE:
            logger.warning("Registry full: %d experts", self.size)
            return False

        if listing.self_assessed_value <= 0:
            logger.warning(
                "Expert %s has non-positive self-assessed value", listing.expert_id
            )
            return False

        self._experts[listing.expert_id] = listing
        logger.info("Expert registered: %s (%s)", listing.name, listing.expert_id)
        return True

    def deregister(self, expert_id: str) -> bool:
        """Remove an expert from the registry."""
        if expert_id in self._experts:
            del self._experts[expert_id]
            return True
        return False

    def get(self, expert_id: str) -> Optional[ExpertListing]:
        """Get an expert by ID."""
        return self._experts.get(expert_id)

    def find_matches(
        self,
        query_vector: CapabilityVector,
        top_k: int = 5,
        min_similarity: float = MIN_MATCH_SIMILARITY,
        tier_filter: Optional[str] = None,
    ) -> List[ExpertMatch]:
        """
        Find the best matching experts for a query.

        Args:
            query_vector: Capability vector representing the query needs
            top_k: Maximum number of matches to return
            min_similarity: Minimum cosine similarity threshold
            tier_filter: Optional filter by tier (EDGE, LOCAL, POOL)

        Returns:
            Sorted list of ExpertMatch objects
        """
        candidates = []

        for expert in self._experts.values():
            # Apply tier filter
            if tier_filter and expert.tier != tier_filter:
                continue

            # Skip unavailable experts
            if expert.availability <= 0:
                continue

            similarity = query_vector.cosine_similarity(expert.capabilities)

            if similarity >= min_similarity:
                candidates.append((expert, similarity))

        # Sort by similarity (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Build matches with ranking
        matches = []
        for rank, (expert, similarity) in enumerate(candidates[:top_k], 1):
            # Price = base cost proportional to self-assessed value and similarity
            estimated_price = expert.self_assessed_value * similarity * 0.01
            matches.append(
                ExpertMatch(
                    expert=expert,
                    similarity=similarity,
                    estimated_price=estimated_price,
                    rank=rank,
                )
            )

        return matches

    def list_all(self) -> List[ExpertListing]:
        """List all registered experts."""
        return list(self._experts.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "size": self.size,
            "experts": {eid: e.to_dict() for eid, e in self._experts.items()},
        }


__all__ = [
    "ExpertRegistry",
    "ExpertListing",
    "ExpertMatch",
    "CapabilityVector",
    "DEFAULT_DIMENSIONS",
]
