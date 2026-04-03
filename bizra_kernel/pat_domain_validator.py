"""
PAT Domain Validator — Cross-Pollination Analysis
=================================================
Validates domain diversity and unrelatedness for PAT enforcement.

Constitution: constitution/pat_enforcement_v1.yaml
Thresholds:
- min_domains: 3
- unrelatedness_threshold: 0.70
- min_cross_connections: 2

Integration:
- Semantic clustering for domain distance
- Cross-domain connection mapping
- Evidence receipts
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger("pat.domain_validator")


# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

MIN_DOMAINS = 3
UNRELATEDNESS_THRESHOLD = 0.70
MIN_CROSS_CONNECTIONS = 2


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Domain:
    """Represents a knowledge domain."""
    name: str
    cluster_id: str
    embedding: Optional[List[float]] = None
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "cluster_id": self.cluster_id,
            "keywords": self.keywords,
            "metadata": self.metadata,
        }


@dataclass
class CrossConnection:
    """Represents a connection between domains."""
    domain_a: str
    domain_b: str
    connection_type: str  # e.g., "synthesis", "analogy", "contrast"
    strength: float
    evidence: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain_a": self.domain_a,
            "domain_b": self.domain_b,
            "connection_type": self.connection_type,
            "strength": self.strength,
            "evidence": self.evidence,
        }


@dataclass
class DomainValidationResult:
    """Result of domain validation."""
    passed: bool
    domain_count: int
    unrelatedness_score: float
    cross_connections: List[CrossConnection]
    domain_map: Dict[str, Domain]
    evidence: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "domain_count": self.domain_count,
            "unrelatedness_score": self.unrelatedness_score,
            "cross_connections": [c.to_dict() for c in self.cross_connections],
            "domain_map": {k: v.to_dict() for k, v in self.domain_map.items()},
            "evidence": self.evidence,
            "timestamp": self.timestamp,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PAT DOMAIN VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════════

class PATDomainValidator:
    """
    Validates domain diversity and cross-pollination for PAT.

    Key Functions:
    - Compute semantic distance between domains
    - Detect cross-domain connections
    - Validate unrelatedness threshold
    """

    def __init__(
        self,
        min_domains: int = MIN_DOMAINS,
        unrelatedness_threshold: float = UNRELATEDNESS_THRESHOLD,
        min_cross_connections: int = MIN_CROSS_CONNECTIONS,
    ):
        """Initialize domain validator."""
        self.min_domains = min_domains
        self.unrelatedness_threshold = unrelatedness_threshold
        self.min_cross_connections = min_cross_connections

        logger.info(
            f"PATDomainValidator initialized: "
            f"min_domains={min_domains}, "
            f"unrelatedness_threshold={unrelatedness_threshold}"
        )

    async def validate(
        self, domains: List[Dict[str, Any]], context: Optional[Dict[str, Any]] = None
    ) -> DomainValidationResult:
        """
        Validate domain diversity and cross-pollination.

        Args:
            domains: List of domain dictionaries
            context: Optional context for validation

        Returns:
            DomainValidationResult with validation outcome
        """
        # Parse domains
        domain_objects = [self._parse_domain(d) for d in domains]
        domain_map = {d.name: d for d in domain_objects}

        # Check domain count
        domain_count = len(domain_objects)
        domain_count_ok = domain_count >= self.min_domains

        # Compute unrelatedness score
        unrelatedness_score = await self._compute_unrelatedness(domain_objects)
        unrelatedness_ok = unrelatedness_score >= self.unrelatedness_threshold

        # Detect cross-domain connections
        cross_connections = await self._detect_cross_connections(domain_objects, context)
        cross_connections_ok = len(cross_connections) >= self.min_cross_connections

        # Overall validation
        passed = domain_count_ok and unrelatedness_ok and cross_connections_ok

        evidence = [
            f"Domain count: {domain_count} (required: {self.min_domains})",
            f"Unrelatedness score: {unrelatedness_score:.4f} (required: {self.unrelatedness_threshold})",
            f"Cross connections: {len(cross_connections)} (required: {self.min_cross_connections})",
        ]

        if not passed:
            if not domain_count_ok:
                evidence.append(f"FAIL: Insufficient domains ({domain_count} < {self.min_domains})")
            if not unrelatedness_ok:
                evidence.append(
                    f"FAIL: Domains too similar "
                    f"({unrelatedness_score:.4f} < {self.unrelatedness_threshold})"
                )
            if not cross_connections_ok:
                evidence.append(
                    f"FAIL: Insufficient cross-connections "
                    f"({len(cross_connections)} < {self.min_cross_connections})"
                )

        return DomainValidationResult(
            passed=passed,
            domain_count=domain_count,
            unrelatedness_score=unrelatedness_score,
            cross_connections=cross_connections,
            domain_map=domain_map,
            evidence=evidence,
        )

    def _parse_domain(self, domain_dict: Dict[str, Any]) -> Domain:
        """Parse domain from dictionary."""
        return Domain(
            name=domain_dict.get("name", "unknown"),
            cluster_id=domain_dict.get("cluster_id", "unknown"),
            embedding=domain_dict.get("embedding"),
            keywords=domain_dict.get("keywords", []),
            metadata=domain_dict.get("metadata", {}),
        )

    async def _compute_unrelatedness(self, domains: List[Domain]) -> float:
        """
        Compute unrelatedness score for domains.

        Uses semantic distance between domain embeddings or keyword overlap.

        Returns:
            Score from 0.0 (identical) to 1.0 (completely unrelated)
        """
        if len(domains) < 2:
            return 0.0

        # Check if embeddings are available
        if all(d.embedding for d in domains):
            return await self._compute_embedding_distance(domains)
        else:
            # Fallback to keyword-based distance
            return await self._compute_keyword_distance(domains)

    async def _compute_embedding_distance(self, domains: List[Domain]) -> float:
        """
        Compute average pairwise cosine distance between domain embeddings.

        Returns:
            Average distance (0.0 = identical, 1.0 = orthogonal)
        """
        embeddings = [np.array(d.embedding) for d in domains]

        total_distance = 0.0
        pair_count = 0

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                # Cosine similarity
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )

                # Convert to distance (0 = similar, 1 = dissimilar)
                distance = 1.0 - similarity

                total_distance += distance
                pair_count += 1

        return total_distance / pair_count if pair_count > 0 else 0.0

    async def _compute_keyword_distance(self, domains: List[Domain]) -> float:
        """
        Compute average pairwise Jaccard distance between domain keywords.

        Returns:
            Average distance (0.0 = identical, 1.0 = no overlap)
        """
        keyword_sets = [set(d.keywords) for d in domains]

        total_distance = 0.0
        pair_count = 0

        for i in range(len(keyword_sets)):
            for j in range(i + 1, len(keyword_sets)):
                intersection = len(keyword_sets[i] & keyword_sets[j])
                union = len(keyword_sets[i] | keyword_sets[j])

                # Jaccard distance
                distance = 1.0 - (intersection / union) if union > 0 else 1.0

                total_distance += distance
                pair_count += 1

        return total_distance / pair_count if pair_count > 0 else 0.0

    async def _detect_cross_connections(
        self, domains: List[Domain], context: Optional[Dict[str, Any]]
    ) -> List[CrossConnection]:
        """
        Detect cross-domain synthesis connections.

        Args:
            domains: List of domains
            context: Optional context with synthesis nodes

        Returns:
            List of CrossConnection objects
        """
        connections = []

        # Check if context contains synthesis nodes
        synthesis_nodes = context.get("synthesis_nodes", []) if context else []

        # Detect connections from synthesis nodes
        for node in synthesis_nodes:
            node_domains = node.get("domains", [])

            # Multi-domain nodes indicate cross-connections
            if len(node_domains) >= 2:
                for i in range(len(node_domains)):
                    for j in range(i + 1, len(node_domains)):
                        domain_a = node_domains[i]
                        domain_b = node_domains[j]

                        # Check if both domains are in our domain list
                        if any(d.name == domain_a for d in domains) and any(
                            d.name == domain_b for d in domains
                        ):
                            connection = CrossConnection(
                                domain_a=domain_a,
                                domain_b=domain_b,
                                connection_type=node.get("connection_type", "synthesis"),
                                strength=node.get("strength", 0.8),
                                evidence=node.get("content", "")[:100],
                            )
                            connections.append(connection)

        # Also detect connections from domain metadata
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                domain_a = domains[i]
                domain_b = domains[j]

                # Check if metadata indicates a connection
                if self._has_metadata_connection(domain_a, domain_b):
                    connection = CrossConnection(
                        domain_a=domain_a.name,
                        domain_b=domain_b.name,
                        connection_type="metadata_link",
                        strength=0.7,
                        evidence=f"Metadata connection between {domain_a.name} and {domain_b.name}",
                    )
                    connections.append(connection)

        return connections

    def _has_metadata_connection(self, domain_a: Domain, domain_b: Domain) -> bool:
        """Check if domains have metadata indicating a connection."""
        # Check for shared keywords
        keyword_overlap = set(domain_a.keywords) & set(domain_b.keywords)
        if keyword_overlap:
            return True

        # Check for explicit connections in metadata
        if "connected_domains" in domain_a.metadata:
            if domain_b.name in domain_a.metadata["connected_domains"]:
                return True

        if "connected_domains" in domain_b.metadata:
            if domain_a.name in domain_b.metadata["connected_domains"]:
                return True

        return False

    async def expand_domains(
        self, current_domains: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """
        Expand domain list to meet minimum requirements.

        Correction action for Gate 1 failure.

        Args:
            current_domains: Current domain list
            query: User query for context

        Returns:
            Expanded domain list
        """
        logger.info(f"Expanding domains from {len(current_domains)} to {self.min_domains}")

        # Parse current domains
        domain_names = {d.get("name") for d in current_domains}

        # Generate candidate domains based on query keywords
        candidate_domains = await self._generate_candidate_domains(query, domain_names)

        # Add candidates until we reach min_domains
        expanded = list(current_domains)
        for candidate in candidate_domains:
            if len(expanded) >= self.min_domains:
                break
            expanded.append(candidate)

        logger.info(f"Expanded to {len(expanded)} domains")

        return expanded

    async def _generate_candidate_domains(
        self, query: str, existing_domains: Set[str]
    ) -> List[Dict[str, Any]]:
        """
        Generate candidate domains based on query.

        Args:
            query: User query
            existing_domains: Set of existing domain names

        Returns:
            List of candidate domain dictionaries
        """
        # Mock implementation: Generate domains from query keywords
        # Real implementation would use LLM or domain taxonomy

        # Extract keywords from query
        keywords = query.lower().split()

        # Predefined domain categories
        domain_categories = [
            "Distributed Systems",
            "Machine Learning",
            "Database Systems",
            "Security",
            "Performance Engineering",
            "Software Architecture",
            "Data Science",
            "Cloud Computing",
            "DevOps",
            "Network Engineering",
        ]

        candidates = []
        for category in domain_categories:
            if category not in existing_domains:
                # Check if query relates to this domain
                category_keywords = category.lower().split()
                if any(kw in keywords for kw in category_keywords):
                    candidates.append(
                        {
                            "name": category,
                            "cluster_id": f"cluster_{hashlib.md5(category.encode()).hexdigest()[:8]}",
                            "keywords": category_keywords,
                            "metadata": {"generated": True, "source": "query_expansion"},
                        }
                    )

        return candidates


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    """Test domain validator."""
    validator = PATDomainValidator()

    # Test case 1: Valid domains
    domains_valid = [
        {
            "name": "Distributed Systems",
            "cluster_id": "cluster_1",
            "keywords": ["distributed", "consensus", "replication"],
        },
        {
            "name": "Machine Learning",
            "cluster_id": "cluster_2",
            "keywords": ["neural", "training", "inference"],
        },
        {
            "name": "Database Systems",
            "cluster_id": "cluster_3",
            "keywords": ["sql", "transactions", "indexing"],
        },
    ]

    result = await validator.validate(domains_valid)
    print("Test 1 - Valid Domains:")
    print(f"  Passed: {result.passed}")
    print(f"  Domain Count: {result.domain_count}")
    print(f"  Unrelatedness: {result.unrelatedness_score:.4f}")
    print(f"  Cross Connections: {len(result.cross_connections)}")
    print()

    # Test case 2: Insufficient domains
    domains_insufficient = [
        {
            "name": "Distributed Systems",
            "cluster_id": "cluster_1",
            "keywords": ["distributed", "consensus"],
        },
    ]

    result = await validator.validate(domains_insufficient)
    print("Test 2 - Insufficient Domains:")
    print(f"  Passed: {result.passed}")
    print(f"  Evidence: {result.evidence}")
    print()

    # Test case 3: Domain expansion
    expanded = await validator.expand_domains(
        domains_insufficient, "Optimize distributed database performance using machine learning"
    )
    print("Test 3 - Domain Expansion:")
    print(f"  Original: {len(domains_insufficient)} domains")
    print(f"  Expanded: {len(expanded)} domains")
    for domain in expanded:
        print(f"    - {domain['name']}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
