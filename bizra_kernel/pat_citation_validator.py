"""
PAT Citation Validator — Elite Practitioner Verification
========================================================
Validates practitioner credentials and relevance for PAT enforcement.

Constitution: constitution/pat_enforcement_v1.yaml
Requirements:
- min_per_domain: 3 practitioners per domain
- tier_required: top_1% (only top 1% practitioners)
- relevance_threshold: 0.60 (minimum relevance to query)

Integration:
- Practitioner registry
- Domain mapping
- Citation tracking
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("pat.citation_validator")


# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

MIN_PRACTITIONERS_PER_DOMAIN = 3
REQUIRED_TIER = "top_1%"
RELEVANCE_THRESHOLD = 0.60


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Practitioner:
    """Represents an elite practitioner."""
    practitioner_id: str
    name: str
    tier: str  # top_1%, top_5%, top_10%
    domains: List[str]
    relevance_score: float
    credentials: List[str] = field(default_factory=list)
    publications: int = 0
    h_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "practitioner_id": self.practitioner_id,
            "name": self.name,
            "tier": self.tier,
            "domains": self.domains,
            "relevance_score": self.relevance_score,
            "credentials": self.credentials,
            "publications": self.publications,
            "h_index": self.h_index,
            "metadata": self.metadata,
        }


@dataclass
class CitationValidationResult:
    """Result of citation validation."""
    passed: bool
    practitioners: List[Practitioner]
    practitioners_per_domain: Dict[str, int]
    tier_distribution: Dict[str, int]
    average_relevance: float
    evidence: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "practitioners": [p.to_dict() for p in self.practitioners],
            "practitioners_per_domain": self.practitioners_per_domain,
            "tier_distribution": self.tier_distribution,
            "average_relevance": self.average_relevance,
            "evidence": self.evidence,
            "timestamp": self.timestamp,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PAT CITATION VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════════

class PATCitationValidator:
    """
    Validates elite practitioner credentials and relevance.

    Enforces top 1% tier requirement and domain coverage.
    """

    def __init__(
        self,
        min_per_domain: int = MIN_PRACTITIONERS_PER_DOMAIN,
        required_tier: str = REQUIRED_TIER,
        relevance_threshold: float = RELEVANCE_THRESHOLD,
    ):
        """Initialize citation validator."""
        self.min_per_domain = min_per_domain
        self.required_tier = required_tier
        self.relevance_threshold = relevance_threshold

        # Practitioner registry (would be loaded from database)
        self.practitioner_registry: Dict[str, Practitioner] = {}

        logger.info(
            f"PATCitationValidator initialized: "
            f"min_per_domain={min_per_domain}, "
            f"tier={required_tier}, "
            f"relevance>={relevance_threshold}"
        )

    async def validate(
        self,
        practitioners: List[Dict[str, Any]],
        domains: List[Dict[str, Any]],
        query: Optional[str] = None,
    ) -> CitationValidationResult:
        """
        Validate practitioner credentials and relevance.

        Args:
            practitioners: List of practitioner dictionaries
            domains: List of domain dictionaries
            query: Optional query for relevance scoring

        Returns:
            CitationValidationResult with validation outcome
        """
        # Parse practitioners
        practitioner_objects = [self._parse_practitioner(p) for p in practitioners]

        # Count practitioners per domain
        domain_names = [d.get("name", "unknown") for d in domains]
        practitioners_per_domain = self._count_per_domain(
            practitioner_objects, domain_names
        )

        # Check domain coverage
        domain_coverage_ok = all(
            count >= self.min_per_domain
            for count in practitioners_per_domain.values()
        )

        # Verify all practitioners are top 1%
        tier_distribution = self._compute_tier_distribution(practitioner_objects)
        all_top_1_percent = all(p.tier == self.required_tier for p in practitioner_objects)

        # Verify relevance scores
        if practitioner_objects:
            average_relevance = sum(p.relevance_score for p in practitioner_objects) / len(
                practitioner_objects
            )
        else:
            average_relevance = 0.0

        relevance_ok = all(
            p.relevance_score >= self.relevance_threshold for p in practitioner_objects
        )

        # Overall validation
        passed = domain_coverage_ok and all_top_1_percent and relevance_ok

        evidence = [
            f"Total practitioners: {len(practitioner_objects)}",
            f"Practitioners per domain: {practitioners_per_domain}",
            f"All top 1%: {all_top_1_percent}",
            f"Average relevance: {average_relevance:.4f}",
        ]

        if not passed:
            if not domain_coverage_ok:
                insufficient_domains = [
                    f"{domain}: {count}/{self.min_per_domain}"
                    for domain, count in practitioners_per_domain.items()
                    if count < self.min_per_domain
                ]
                evidence.append(f"FAIL: Insufficient domain coverage: {insufficient_domains}")

            if not all_top_1_percent:
                non_top_1 = [
                    p.name for p in practitioner_objects if p.tier != self.required_tier
                ]
                evidence.append(f"FAIL: Non-top-1% practitioners: {non_top_1}")

            if not relevance_ok:
                low_relevance = [
                    f"{p.name} ({p.relevance_score:.2f})"
                    for p in practitioner_objects
                    if p.relevance_score < self.relevance_threshold
                ]
                evidence.append(
                    f"FAIL: Low relevance scores (<{self.relevance_threshold}): {low_relevance}"
                )

        return CitationValidationResult(
            passed=passed,
            practitioners=practitioner_objects,
            practitioners_per_domain=practitioners_per_domain,
            tier_distribution=tier_distribution,
            average_relevance=average_relevance,
            evidence=evidence,
        )

    def _parse_practitioner(self, prac_dict: Dict[str, Any]) -> Practitioner:
        """Parse practitioner from dictionary."""
        practitioner_id = prac_dict.get(
            "practitioner_id",
            hashlib.sha256(prac_dict.get("name", "unknown").encode()).hexdigest()[:16],
        )

        return Practitioner(
            practitioner_id=practitioner_id,
            name=prac_dict.get("name", "unknown"),
            tier=prac_dict.get("tier", "unknown"),
            domains=prac_dict.get("domains", []),
            relevance_score=prac_dict.get("relevance_score", 0.0),
            credentials=prac_dict.get("credentials", []),
            publications=prac_dict.get("publications", 0),
            h_index=prac_dict.get("h_index", 0),
            metadata=prac_dict.get("metadata", {}),
        )

    def _count_per_domain(
        self, practitioners: List[Practitioner], domain_names: List[str]
    ) -> Dict[str, int]:
        """Count practitioners per domain."""
        counts = {domain: 0 for domain in domain_names}

        for practitioner in practitioners:
            for domain in practitioner.domains:
                if domain in counts:
                    counts[domain] += 1

        return counts

    def _compute_tier_distribution(
        self, practitioners: List[Practitioner]
    ) -> Dict[str, int]:
        """Compute distribution of practitioner tiers."""
        distribution: Dict[str, int] = {}

        for practitioner in practitioners:
            tier = practitioner.tier
            distribution[tier] = distribution.get(tier, 0) + 1

        return distribution

    async def fetch_additional_practitioners(
        self,
        domain: str,
        query: str,
        current_count: int,
    ) -> List[Dict[str, Any]]:
        """
        Fetch additional practitioners to meet minimum requirements.

        Correction action for Gate 4 failure.

        Args:
            domain: Domain name
            query: User query for relevance scoring
            current_count: Current practitioner count for domain

        Returns:
            List of additional practitioner dictionaries
        """
        needed = self.min_per_domain - current_count

        logger.info(f"Fetching {needed} additional practitioners for domain: {domain}")

        # Query practitioner registry
        additional = await self._query_practitioner_registry(domain, query, limit=needed)

        logger.info(f"Fetched {len(additional)} additional practitioners")

        return additional

    async def _query_practitioner_registry(
        self, domain: str, query: str, limit: int
    ) -> List[Dict[str, Any]]:
        """
        Query practitioner registry for domain experts.

        Integration point: Would query practitioner database or API.
        """
        # Mock implementation: Generate practitioners
        mock_practitioners = [
            {
                "name": f"Expert {i+1} - {domain}",
                "tier": "top_1%",
                "domains": [domain],
                "relevance_score": 0.70 + (i * 0.05),
                "credentials": ["PhD", "Industry Experience"],
                "publications": 50 + (i * 10),
                "h_index": 20 + (i * 5),
            }
            for i in range(limit)
        ]

        return mock_practitioners

    async def compute_relevance_score(
        self, practitioner: Practitioner, query: str, context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute relevance score for practitioner to query.

        Uses semantic similarity between practitioner expertise and query.

        Args:
            practitioner: Practitioner object
            query: User query
            context: Optional context

        Returns:
            Relevance score (0.0-1.0)
        """
        # Mock implementation: Would use embeddings
        # Check if practitioner domains overlap with query keywords
        query_lower = query.lower()
        domain_matches = sum(
            1 for domain in practitioner.domains if domain.lower() in query_lower
        )

        # Base relevance on domain matches
        base_relevance = min(domain_matches * 0.3, 0.9)

        # Add bonus for high-tier practitioners
        if practitioner.tier == "top_1%":
            base_relevance = min(base_relevance + 0.1, 1.0)

        return base_relevance

    async def verify_credentials(self, practitioner: Practitioner) -> bool:
        """
        Verify practitioner credentials.

        Integration point: Would verify against credential database.

        Args:
            practitioner: Practitioner to verify

        Returns:
            True if credentials verified
        """
        # Check minimum credential requirements
        required_credentials = {"PhD", "Industry Experience", "Publications"}

        practitioner_creds = set(practitioner.credentials)

        # Must have at least 2 of the required credentials
        matches = len(required_credentials & practitioner_creds)

        # Also check publication and h-index thresholds
        publication_ok = practitioner.publications >= 10
        h_index_ok = practitioner.h_index >= 5

        return matches >= 2 and publication_ok and h_index_ok


# ═══════════════════════════════════════════════════════════════════════════════
# PRACTITIONER REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

class PractitionerRegistry:
    """
    Registry of elite practitioners.

    Would be backed by database in production.
    """

    def __init__(self):
        """Initialize empty registry."""
        self.practitioners: Dict[str, Practitioner] = {}
        logger.info("PractitionerRegistry initialized")

    def register(self, practitioner: Practitioner) -> None:
        """Register a practitioner."""
        self.practitioners[practitioner.practitioner_id] = practitioner
        logger.info(f"Registered practitioner: {practitioner.name}")

    def get(self, practitioner_id: str) -> Optional[Practitioner]:
        """Get practitioner by ID."""
        return self.practitioners.get(practitioner_id)

    def search(
        self, domain: Optional[str] = None, tier: Optional[str] = None, limit: int = 10
    ) -> List[Practitioner]:
        """
        Search practitioners by criteria.

        Args:
            domain: Optional domain filter
            tier: Optional tier filter
            limit: Maximum results

        Returns:
            List of matching practitioners
        """
        results = []

        for practitioner in self.practitioners.values():
            # Apply filters
            if domain and domain not in practitioner.domains:
                continue

            if tier and practitioner.tier != tier:
                continue

            results.append(practitioner)

            if len(results) >= limit:
                break

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    """Test citation validator."""
    validator = PATCitationValidator()

    # Test case 1: Valid practitioners
    practitioners_valid = [
        {
            "name": "Dr. Alice Smith",
            "tier": "top_1%",
            "domains": ["Distributed Systems", "Database Systems"],
            "relevance_score": 0.85,
            "credentials": ["PhD", "Industry Experience"],
            "publications": 50,
            "h_index": 25,
        },
        {
            "name": "Prof. Bob Johnson",
            "tier": "top_1%",
            "domains": ["Distributed Systems"],
            "relevance_score": 0.80,
            "credentials": ["PhD", "Publications"],
            "publications": 100,
            "h_index": 40,
        },
        {
            "name": "Dr. Carol Williams",
            "tier": "top_1%",
            "domains": ["Database Systems"],
            "relevance_score": 0.75,
            "credentials": ["PhD", "Industry Experience", "Publications"],
            "publications": 75,
            "h_index": 30,
        },
    ]

    domains = [
        {"name": "Distributed Systems"},
        {"name": "Database Systems"},
    ]

    result = await validator.validate(practitioners_valid, domains)

    print("Test 1 - Valid Practitioners:")
    print(f"  Passed: {result.passed}")
    print(f"  Total: {len(result.practitioners)}")
    print(f"  Per domain: {result.practitioners_per_domain}")
    print(f"  Average relevance: {result.average_relevance:.4f}")
    print()

    # Test case 2: Insufficient practitioners
    practitioners_insufficient = [
        {
            "name": "Dr. Alice Smith",
            "tier": "top_1%",
            "domains": ["Distributed Systems"],
            "relevance_score": 0.85,
        },
    ]

    result = await validator.validate(practitioners_insufficient, domains)

    print("Test 2 - Insufficient Practitioners:")
    print(f"  Passed: {result.passed}")
    print(f"  Evidence: {result.evidence}")
    print()

    # Test case 3: Fetch additional practitioners
    additional = await validator.fetch_additional_practitioners(
        "Database Systems",
        "Optimize database performance",
        current_count=1,
    )

    print("Test 3 - Fetch Additional:")
    print(f"  Fetched: {len(additional)} practitioners")
    for prac in additional:
        print(f"    - {prac['name']} (relevance: {prac['relevance_score']:.2f})")
    print()

    # Test case 4: Practitioner registry
    registry = PractitionerRegistry()

    for prac_dict in practitioners_valid:
        prac = validator._parse_practitioner(prac_dict)
        registry.register(prac)

    search_results = registry.search(domain="Distributed Systems", tier="top_1%")

    print("Test 4 - Registry Search:")
    print(f"  Query: domain='Distributed Systems', tier='top_1%'")
    print(f"  Results: {len(search_results)}")
    for prac in search_results:
        print(f"    - {prac.name}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
