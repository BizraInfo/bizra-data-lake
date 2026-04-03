"""
Elite Practitioner Protocol - Standing on Giants Validation
============================================================
Validates that synthesis results are grounded in top 1% elite practitioners
from multiple unrelated domains.

Constitution: constitution/pat_enforcement_v1.yaml
Key Thresholds:
- DOMAIN_MIN: 3 (minimum unrelated domains)
- PRACTITIONERS_PER_DOMAIN: 3 (minimum elite per domain)
- UNRELATEDNESS_THRESHOLD: 0.70 (pairwise semantic distance)
- NOVELTY_THRESHOLD: 0.75 (semantic distance from known patterns)

Integration:
- WinterProofEmbedder from core/sovereignty/winter_proof.py
- PAT novelty probe from bizra_kernel/pat_novelty_probe.py
- Domain validation from bizra_kernel/pat_domain_validator.py
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

# Import constitutional thresholds - Genesis v2.2.2 compliance
from core.constants import (
    NOVELTY_THRESHOLD_STANDARD,
    CONFIDENCE_LOW,
)

# Optional numpy for distance calculations
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logger = logging.getLogger("apex.sovereign.elite_practitioner")


# =============================================================================
# THRESHOLDS (from core/constants.py - Genesis v2.2.2 compliance)
# =============================================================================

DOMAIN_MIN = 3                      # Minimum unrelated domains required
PRACTITIONERS_PER_DOMAIN = 3        # Minimum elite practitioners per domain
UNRELATEDNESS_THRESHOLD = CONFIDENCE_LOW  # 0.70 - Cluster distance threshold
NOVELTY_THRESHOLD = NOVELTY_THRESHOLD_STANDARD  # 0.75 - Semantic distance from known patterns

# Domain prefix for receipts
DOMAIN_PREFIX = "bizra-elite-practitioner-v1:"


# =============================================================================
# ENUMS
# =============================================================================


class PractitionerTier(str, Enum):
    """
    Elite practitioner tier classification.

    Based on academic impact, citations, and verified expertise.
    Only TOP_1_PERCENT tier is acceptable for sovereign synthesis.
    """
    TOP_1_PERCENT = "top_1_percent"   # Elite, required for sovereign
    TOP_5_PERCENT = "top_5_percent"   # Expert
    TOP_10_PERCENT = "top_10_percent"  # Proficient
    GENERAL = "general"               # Not verified


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class Practitioner:
    """
    Represents an elite practitioner (researcher, author, expert).

    Attributes:
        name: Full name of the practitioner
        domain: Primary domain of expertise
        tier: Practitioner tier classification
        h_index: Optional h-index for academic practitioners
        citations: Optional total citation count
        verified: Whether the practitioner has been verified
        source: Source of verification (paper, book, institution)
    """
    name: str
    domain: str
    tier: PractitionerTier
    h_index: Optional[int] = None
    citations: Optional[int] = None
    verified: bool = False
    source: str = "unverified"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "domain": self.domain,
            "tier": self.tier.value,
            "h_index": self.h_index,
            "citations": self.citations,
            "verified": self.verified,
            "source": self.source,
        }

    def is_elite(self) -> bool:
        """Check if practitioner meets elite (top 1%) requirements."""
        return self.tier == PractitionerTier.TOP_1_PERCENT and self.verified


@dataclass
class DomainValidation:
    """
    Validation result for a single domain.

    Attributes:
        domain: Domain name
        practitioners: List of practitioners in this domain
        elite_count: Number of elite (top 1%) practitioners
        unrelatedness_scores: Pairwise unrelatedness scores with other domains
        meets_requirements: Whether domain meets all requirements
    """
    domain: str
    practitioners: List[Practitioner]
    elite_count: int
    unrelatedness_scores: Dict[str, float]
    meets_requirements: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "domain": self.domain,
            "practitioners": [p.to_dict() for p in self.practitioners],
            "elite_count": self.elite_count,
            "unrelatedness_scores": self.unrelatedness_scores,
            "meets_requirements": self.meets_requirements,
        }


@dataclass
class ElitePractitionerResult:
    """
    Result from elite practitioner validation.

    Attributes:
        valid: Overall validation passed
        reason: Reason for failure if not valid
        domains_validated: List of domain validation results
        total_domains: Total number of domains found
        total_elite_practitioners: Total elite practitioners across all domains
        novelty_score: Novelty score for cross-domain synthesis
        cross_domain_synthesis_valid: Whether synthesis novelty >= 0.75
        evidence: Additional evidence and metrics
    """
    valid: bool
    reason: Optional[str]
    domains_validated: List[DomainValidation]
    total_domains: int
    total_elite_practitioners: int
    novelty_score: float
    cross_domain_synthesis_valid: bool
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "valid": self.valid,
            "reason": self.reason,
            "domains_validated": [d.to_dict() for d in self.domains_validated],
            "total_domains": self.total_domains,
            "total_elite_practitioners": self.total_elite_practitioners,
            "novelty_score": self.novelty_score,
            "cross_domain_synthesis_valid": self.cross_domain_synthesis_valid,
            "evidence": self.evidence,
            "timestamp": self.timestamp,
        }

    def generate_receipt(self) -> Dict[str, Any]:
        """Generate validation receipt for evidence chain."""
        receipt_id = hashlib.sha256(
            f"{DOMAIN_PREFIX}{self.timestamp}{self.valid}".encode()
        ).hexdigest()[:16]

        return {
            "receipt_id": receipt_id,
            "operation": "elite_practitioner_validation",
            "domain": DOMAIN_PREFIX,
            "timestamp": self.timestamp,
            "valid": self.valid,
            "total_domains": self.total_domains,
            "total_elite_practitioners": self.total_elite_practitioners,
            "novelty_score": self.novelty_score,
            "reason": self.reason,
        }


# =============================================================================
# UNRELATEDNESS MEASURE
# =============================================================================


class UnrelatednessMeasure:
    """
    Computes semantic distance between domains using embeddings.

    Uses WinterProofEmbedder for deterministic offline embeddings.
    """

    def __init__(self, embedder: Optional[Any] = None):
        """
        Initialize unrelatedness measure.

        Args:
            embedder: WinterProofEmbedder instance (lazy loaded if None)
        """
        self._embedder = embedder
        self._embedding_cache: Dict[str, List[float]] = {}

    def _get_embedder(self) -> Any:
        """Get or create WinterProofEmbedder instance."""
        if self._embedder is None:
            try:
                from core.sovereignty.winter_proof import WinterProofEmbedder
                self._embedder = WinterProofEmbedder(dimension=384)
            except ImportError:
                logger.warning("WinterProofEmbedder not available, using mock")
                self._embedder = MockEmbedder()
        return self._embedder

    def get_domain_embedding(self, domain: str) -> List[float]:
        """
        Get embedding vector for a domain.

        Args:
            domain: Domain name

        Returns:
            Embedding vector as list of floats
        """
        if domain not in self._embedding_cache:
            embedder = self._get_embedder()
            self._embedding_cache[domain] = embedder.embed(domain)
        return self._embedding_cache[domain]

    def compute_distance(self, domain_a: str, domain_b: str) -> float:
        """
        Compute semantic distance between two domains.

        Args:
            domain_a: First domain name
            domain_b: Second domain name

        Returns:
            Distance from 0.0 (identical) to 1.0 (completely unrelated)
        """
        emb_a = self.get_domain_embedding(domain_a)
        emb_b = self.get_domain_embedding(domain_b)

        if HAS_NUMPY:
            vec_a = np.array(emb_a)
            vec_b = np.array(emb_b)

            # Cosine similarity
            dot = np.dot(vec_a, vec_b)
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)

            if norm_a == 0 or norm_b == 0:
                return 1.0

            similarity = dot / (norm_a * norm_b)
            # Convert to distance (0 = similar, 1 = dissimilar)
            distance = 1.0 - similarity
        else:
            # Manual computation without numpy
            dot = sum(a * b for a, b in zip(emb_a, emb_b))
            norm_a = sum(x * x for x in emb_a) ** 0.5
            norm_b = sum(x * x for x in emb_b) ** 0.5

            if norm_a == 0 or norm_b == 0:
                return 1.0

            similarity = dot / (norm_a * norm_b)
            distance = 1.0 - similarity

        return float(max(0.0, min(1.0, distance)))

    def is_unrelated(self, domain_a: str, domain_b: str) -> bool:
        """
        Check if two domains are sufficiently unrelated.

        Args:
            domain_a: First domain name
            domain_b: Second domain name

        Returns:
            True if distance >= UNRELATEDNESS_THRESHOLD
        """
        distance = self.compute_distance(domain_a, domain_b)
        return distance >= UNRELATEDNESS_THRESHOLD


class MockEmbedder:
    """Mock embedder for when WinterProofEmbedder is not available."""

    def embed(self, text: str) -> List[float]:
        """Generate deterministic mock embedding."""
        hash_val = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        # Generate 384-dimensional embedding
        import random
        random.seed(hash_val % (2**32))
        embedding = [random.gauss(0, 1) for _ in range(384)]
        # Normalize
        norm = sum(x * x for x in embedding) ** 0.5
        return [x / norm for x in embedding]


# =============================================================================
# PRACTITIONER REGISTRY
# =============================================================================


class PractitionerRegistry:
    """
    Registry of known elite practitioners.

    Maintains a database of verified practitioners indexed by domain.
    """

    def __init__(self):
        """Initialize practitioner registry."""
        self.practitioners: Dict[str, Practitioner] = {}
        self.domain_index: Dict[str, List[str]] = {}  # domain -> practitioner names

        logger.info("PractitionerRegistry initialized")

    def load_from_yaml(self, path: str) -> int:
        """
        Load practitioner data from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Number of practitioners loaded
        """
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)

            count = 0
            for entry in data.get("practitioners", []):
                practitioner = Practitioner(
                    name=entry["name"],
                    domain=entry["domain"],
                    tier=PractitionerTier(entry.get("tier", "general")),
                    h_index=entry.get("h_index"),
                    citations=entry.get("citations"),
                    verified=entry.get("verified", False),
                    source=entry.get("source", "yaml_import"),
                )

                self.practitioners[practitioner.name] = practitioner

                # Index by domain
                if practitioner.domain not in self.domain_index:
                    self.domain_index[practitioner.domain] = []
                self.domain_index[practitioner.domain].append(practitioner.name)

                count += 1

            logger.info(f"Loaded {count} practitioners from {path}")
            return count

        except Exception as e:
            logger.error(f"Failed to load practitioners from {path}: {e}")
            return 0

    def is_top_1_percent(self, practitioner: Practitioner) -> bool:
        """
        Verify if practitioner meets top 1% criteria.

        Args:
            practitioner: Practitioner to verify

        Returns:
            True if top 1% criteria met
        """
        # Check explicit tier
        if practitioner.tier == PractitionerTier.TOP_1_PERCENT:
            return True

        # Check h-index threshold (field-dependent, using 50 as general threshold)
        if practitioner.h_index and practitioner.h_index >= 50:
            return True

        # Check citation threshold (using 10000 as general threshold)
        if practitioner.citations and practitioner.citations >= 10000:
            return True

        return False

    def get_practitioners_for_domain(self, domain: str) -> List[Practitioner]:
        """
        Get all practitioners for a domain.

        Args:
            domain: Domain name

        Returns:
            List of practitioners in the domain
        """
        names = self.domain_index.get(domain, [])
        return [self.practitioners[name] for name in names if name in self.practitioners]

    def verify_citation(
        self, name: str, domain: str
    ) -> Optional[Practitioner]:
        """
        Verify a citation matches a known practitioner.

        Args:
            name: Practitioner name from citation
            domain: Domain context

        Returns:
            Practitioner if found and verified, None otherwise
        """
        # Direct lookup
        if name in self.practitioners:
            practitioner = self.practitioners[name]
            if practitioner.verified:
                return practitioner

        # Fuzzy match by domain
        for p_name, practitioner in self.practitioners.items():
            if self._fuzzy_match(name, p_name):
                if practitioner.domain == domain or not domain:
                    return practitioner

        return None

    def _fuzzy_match(self, query: str, target: str) -> bool:
        """Simple fuzzy name matching."""
        query_parts = set(query.lower().split())
        target_parts = set(target.lower().split())

        # Check for significant overlap
        overlap = len(query_parts & target_parts)
        return overlap >= 2 or (overlap >= 1 and len(query_parts) == 1)

    def register_practitioner(self, practitioner: Practitioner) -> None:
        """
        Register a new practitioner.

        Args:
            practitioner: Practitioner to register
        """
        self.practitioners[practitioner.name] = practitioner

        if practitioner.domain not in self.domain_index:
            self.domain_index[practitioner.domain] = []
        if practitioner.name not in self.domain_index[practitioner.domain]:
            self.domain_index[practitioner.domain].append(practitioner.name)

        logger.debug(f"Registered practitioner: {practitioner.name}")


# =============================================================================
# ELITE PRACTITIONER PROTOCOL
# =============================================================================


class ElitePractitionerProtocol:
    """
    "Standing on Giants" validation protocol.

    Validates that synthesis results are grounded in top 1% elite
    practitioners from multiple unrelated domains.

    Key Requirements (from PAT enforcement):
    - Minimum 3 unrelated domains
    - Minimum 3 elite practitioners per domain
    - Unrelatedness threshold >= 0.70
    - Novelty threshold >= 0.75
    """

    # Configuration constants
    DOMAIN_MIN = DOMAIN_MIN
    PRACTITIONERS_PER_DOMAIN = PRACTITIONERS_PER_DOMAIN
    UNRELATEDNESS_THRESHOLD = UNRELATEDNESS_THRESHOLD
    NOVELTY_THRESHOLD = NOVELTY_THRESHOLD

    def __init__(
        self,
        practitioner_registry: Optional[PractitionerRegistry] = None,
        novelty_probe: Optional[Any] = None,
    ):
        """
        Initialize elite practitioner protocol.

        Args:
            practitioner_registry: Registry of known practitioners
            novelty_probe: PAT novelty probe instance
        """
        self.practitioner_registry = practitioner_registry or PractitionerRegistry()
        self._novelty_probe = novelty_probe
        self._unrelatedness_measure = UnrelatednessMeasure()

        logger.info(
            f"ElitePractitionerProtocol initialized: "
            f"DOMAIN_MIN={self.DOMAIN_MIN}, "
            f"PRACTITIONERS_PER_DOMAIN={self.PRACTITIONERS_PER_DOMAIN}"
        )

    async def validate(self, response: Any) -> ElitePractitionerResult:
        """
        Validate response against elite practitioner requirements.

        Args:
            response: SynthesisResult or similar response object

        Returns:
            ElitePractitionerResult with validation outcome
        """
        evidence: Dict[str, Any] = {
            "validation_steps": [],
            "domain_details": {},
            "unrelatedness_matrix": {},
        }

        # Step 1: Extract practitioner citations from response
        citations = await self._extract_practitioner_citations(response)
        evidence["validation_steps"].append({
            "step": "extract_citations",
            "citations_found": len(citations),
        })

        # Step 2: Group citations by domain
        domains = self._group_by_domain(citations)
        evidence["validation_steps"].append({
            "step": "group_domains",
            "domains_found": list(domains.keys()),
        })

        # Step 3: Validate domain count
        domain_count_valid, domain_count_reason = self._validate_domain_count(
            list(domains.keys())
        )
        evidence["validation_steps"].append({
            "step": "validate_domain_count",
            "valid": domain_count_valid,
            "reason": domain_count_reason,
        })

        # Step 4: Validate unrelatedness between domains
        unrelatedness_valid, unrelatedness_scores = await self._validate_unrelatedness(
            list(domains.keys())
        )
        evidence["unrelatedness_matrix"] = unrelatedness_scores
        evidence["validation_steps"].append({
            "step": "validate_unrelatedness",
            "valid": unrelatedness_valid,
        })

        # Step 5: Validate practitioner tiers per domain
        domains_validated: List[DomainValidation] = []
        total_elite = 0
        all_domains_valid = True

        for domain_name, domain_practitioners in domains.items():
            tier_valid, elite_count = self._validate_practitioner_tier(
                domain_practitioners
            )

            # Get unrelatedness scores for this domain
            domain_unrelatedness = {
                other: unrelatedness_scores.get(f"{domain_name}-{other}", 0.0)
                for other in domains.keys()
                if other != domain_name
            }

            domain_validation = DomainValidation(
                domain=domain_name,
                practitioners=domain_practitioners,
                elite_count=elite_count,
                unrelatedness_scores=domain_unrelatedness,
                meets_requirements=tier_valid and elite_count >= self.PRACTITIONERS_PER_DOMAIN,
            )
            domains_validated.append(domain_validation)
            total_elite += elite_count

            if not domain_validation.meets_requirements:
                all_domains_valid = False

            evidence["domain_details"][domain_name] = {
                "practitioners": len(domain_practitioners),
                "elite_count": elite_count,
                "tier_valid": tier_valid,
            }

        evidence["validation_steps"].append({
            "step": "validate_practitioner_tiers",
            "all_valid": all_domains_valid,
            "total_elite": total_elite,
        })

        # Step 6: Validate cross-domain novelty
        novelty_score = await self._validate_cross_domain_novelty(response)
        cross_domain_valid = novelty_score >= self.NOVELTY_THRESHOLD
        evidence["validation_steps"].append({
            "step": "validate_novelty",
            "novelty_score": novelty_score,
            "valid": cross_domain_valid,
        })

        # Determine overall validity
        valid = (
            domain_count_valid
            and unrelatedness_valid
            and all_domains_valid
            and cross_domain_valid
        )

        # Determine failure reason
        reason: Optional[str] = None
        if not valid:
            reasons = []
            if not domain_count_valid:
                reasons.append(domain_count_reason)
            if not unrelatedness_valid:
                reasons.append(
                    f"Domains not sufficiently unrelated (threshold: {self.UNRELATEDNESS_THRESHOLD})"
                )
            if not all_domains_valid:
                failing_domains = [
                    d.domain for d in domains_validated if not d.meets_requirements
                ]
                reasons.append(
                    f"Insufficient elite practitioners in domains: {failing_domains}"
                )
            if not cross_domain_valid:
                reasons.append(
                    f"Novelty score {novelty_score:.4f} below threshold {self.NOVELTY_THRESHOLD}"
                )
            reason = "; ".join(reasons)

        result = ElitePractitionerResult(
            valid=valid,
            reason=reason,
            domains_validated=domains_validated,
            total_domains=len(domains),
            total_elite_practitioners=total_elite,
            novelty_score=novelty_score,
            cross_domain_synthesis_valid=cross_domain_valid,
            evidence=evidence,
        )

        logger.info(
            f"Elite practitioner validation: valid={valid}, "
            f"domains={len(domains)}, elite={total_elite}, novelty={novelty_score:.4f}"
        )

        return result

    async def _extract_practitioner_citations(
        self, response: Any
    ) -> List[Practitioner]:
        """
        Extract practitioner citations from response.

        Parses response content to identify cited practitioners.

        Args:
            response: SynthesisResult or similar response

        Returns:
            List of Practitioner objects extracted from citations
        """
        practitioners: List[Practitioner] = []

        # Get content from response
        content = ""
        if hasattr(response, "synthesized_content"):
            content = response.synthesized_content
        elif hasattr(response, "content"):
            content = response.content
        elif isinstance(response, str):
            content = response
        elif isinstance(response, dict):
            content = response.get("content", "") or response.get("synthesized_content", "")

        if not content:
            return practitioners

        # Pattern for detecting citations
        # Matches patterns like "Author (Year)", "Author et al.", "[Author, Year]"
        citation_patterns = [
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*(?:et\s+al\.?)?\s*\((\d{4})\)",
            r"\[([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),?\s*(\d{4})\]",
            r"(?:according to|cited by|proposed by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        ]

        found_names: Set[str] = set()

        for pattern in citation_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                name = match[0] if isinstance(match, tuple) else match
                name = name.strip()
                if name and name not in found_names:
                    found_names.add(name)

        # Verify citations against registry
        for name in found_names:
            # Try to find in registry
            verified = self.practitioner_registry.verify_citation(name, "")
            if verified:
                practitioners.append(verified)
            else:
                # Create unverified practitioner
                practitioners.append(Practitioner(
                    name=name,
                    domain="unknown",
                    tier=PractitionerTier.GENERAL,
                    verified=False,
                    source="citation_extraction",
                ))

        # Also check for domain-specific practitioners in metadata
        if hasattr(response, "metadata") and isinstance(response.metadata, dict):
            for domain, domain_practitioners in response.metadata.get(
                "practitioners", {}
            ).items():
                for p_data in domain_practitioners:
                    if isinstance(p_data, dict):
                        practitioner = Practitioner(
                            name=p_data.get("name", "Unknown"),
                            domain=domain,
                            tier=PractitionerTier(p_data.get("tier", "general")),
                            h_index=p_data.get("h_index"),
                            citations=p_data.get("citations"),
                            verified=p_data.get("verified", False),
                            source=p_data.get("source", "metadata"),
                        )
                        practitioners.append(practitioner)

        logger.debug(f"Extracted {len(practitioners)} practitioner citations")
        return practitioners

    def _group_by_domain(
        self, practitioners: List[Practitioner]
    ) -> Dict[str, List[Practitioner]]:
        """Group practitioners by domain."""
        domains: Dict[str, List[Practitioner]] = {}

        for practitioner in practitioners:
            domain = practitioner.domain or "unknown"
            if domain not in domains:
                domains[domain] = []
            domains[domain].append(practitioner)

        return domains

    def _validate_domain_count(
        self, domains: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate minimum domain count requirement.

        Args:
            domains: List of domain names

        Returns:
            Tuple of (valid, reason_if_invalid)
        """
        # Filter out "unknown" domain
        valid_domains = [d for d in domains if d != "unknown"]

        if len(valid_domains) >= self.DOMAIN_MIN:
            return True, None

        return False, (
            f"Insufficient domains: {len(valid_domains)} found, "
            f"{self.DOMAIN_MIN} required"
        )

    async def _validate_unrelatedness(
        self, domains: List[str]
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Validate pairwise unrelatedness between domains.

        Args:
            domains: List of domain names

        Returns:
            Tuple of (all_pairs_valid, unrelatedness_scores)
        """
        scores: Dict[str, float] = {}
        all_valid = True

        # Filter out "unknown" domain
        valid_domains = [d for d in domains if d != "unknown"]

        if len(valid_domains) < 2:
            return False, scores

        for i, domain_a in enumerate(valid_domains):
            for domain_b in valid_domains[i + 1:]:
                distance = self._compute_domain_unrelatedness(domain_a, domain_b)
                key = f"{domain_a}-{domain_b}"
                scores[key] = distance

                if distance < self.UNRELATEDNESS_THRESHOLD:
                    all_valid = False
                    logger.debug(
                        f"Domains too similar: {domain_a} - {domain_b} "
                        f"(distance: {distance:.4f})"
                    )

        return all_valid, scores

    def _compute_domain_unrelatedness(
        self, domain_a: str, domain_b: str
    ) -> float:
        """
        Compute unrelatedness (semantic distance) between two domains.

        Args:
            domain_a: First domain name
            domain_b: Second domain name

        Returns:
            Distance from 0.0 (identical) to 1.0 (unrelated)
        """
        return self._unrelatedness_measure.compute_distance(domain_a, domain_b)

    def _validate_practitioner_tier(
        self, practitioners: List[Practitioner]
    ) -> Tuple[bool, int]:
        """
        Validate practitioner tier requirements.

        Args:
            practitioners: List of practitioners to validate

        Returns:
            Tuple of (has_enough_elite, elite_count)
        """
        elite_count = sum(
            1 for p in practitioners
            if p.is_elite() or self.practitioner_registry.is_top_1_percent(p)
        )

        valid = elite_count >= self.PRACTITIONERS_PER_DOMAIN
        return valid, elite_count

    async def _validate_cross_domain_novelty(self, response: Any) -> float:
        """
        Validate cross-domain synthesis novelty.

        Uses PAT novelty probe to measure semantic distance from known patterns.

        Args:
            response: SynthesisResult or similar response

        Returns:
            Novelty score from 0.0 to 1.0
        """
        # Get novelty probe
        novelty_probe = self._get_novelty_probe()

        # Get content from response
        content = ""
        if hasattr(response, "synthesized_content"):
            content = response.synthesized_content
        elif hasattr(response, "content"):
            content = response.content
        elif isinstance(response, str):
            content = response
        elif isinstance(response, dict):
            content = response.get("content", "") or response.get("synthesized_content", "")

        if not content:
            return 0.0

        try:
            # Use novelty probe
            result = await novelty_probe.probe(content)
            return result.novelty_score
        except Exception as e:
            logger.warning(f"Novelty probe failed: {e}, using fallback")
            # Fallback: compute novelty based on domain diversity
            return self._compute_fallback_novelty(response)

    def _get_novelty_probe(self) -> Any:
        """Get or create PAT novelty probe instance."""
        if self._novelty_probe is None:
            try:
                from bizra_kernel.pat_novelty_probe import PATNoveltyProbe
                self._novelty_probe = PATNoveltyProbe(
                    novelty_threshold=self.NOVELTY_THRESHOLD
                )
            except ImportError:
                logger.warning("PATNoveltyProbe not available, using mock")
                self._novelty_probe = MockNoveltyProbe()
        return self._novelty_probe

    def _compute_fallback_novelty(self, response: Any) -> float:
        """
        Compute fallback novelty score based on domain diversity.

        Args:
            response: SynthesisResult or similar response

        Returns:
            Novelty score from 0.0 to 1.0
        """
        # Check for domains covered
        domains_covered: Set[str] = set()

        if hasattr(response, "domains_covered"):
            domains_covered = response.domains_covered
        elif hasattr(response, "metadata") and isinstance(response.metadata, dict):
            domains_covered = set(response.metadata.get("domains", []))

        # More domains = higher novelty (base 0.5 + 0.1 per domain, max 1.0)
        novelty = min(0.5 + len(domains_covered) * 0.1, 1.0)

        return novelty


class MockNoveltyProbe:
    """Mock novelty probe for when PATNoveltyProbe is not available."""

    async def probe(self, content: str) -> Any:
        """Mock novelty probe returning high novelty."""
        class MockResult:
            novelty_score = 0.8
        return MockResult()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_elite_practitioner_protocol(
    registry_path: Optional[str] = None,
) -> ElitePractitionerProtocol:
    """
    Create an ElitePractitionerProtocol instance.

    Args:
        registry_path: Optional path to practitioner registry YAML

    Returns:
        Configured ElitePractitionerProtocol instance
    """
    registry = PractitionerRegistry()

    if registry_path:
        registry.load_from_yaml(registry_path)

    return ElitePractitionerProtocol(
        practitioner_registry=registry,
    )


# =============================================================================
# TESTING
# =============================================================================


async def main():
    """Test elite practitioner protocol."""
    print("Elite Practitioner Protocol - Standing on Giants")
    print("=" * 60)

    # Create protocol
    protocol = ElitePractitionerProtocol()

    # Register some test practitioners
    test_practitioners = [
        Practitioner(
            name="Geoffrey Hinton",
            domain="Machine Learning",
            tier=PractitionerTier.TOP_1_PERCENT,
            h_index=170,
            citations=500000,
            verified=True,
            source="academic_record",
        ),
        Practitioner(
            name="Yann LeCun",
            domain="Machine Learning",
            tier=PractitionerTier.TOP_1_PERCENT,
            h_index=150,
            citations=400000,
            verified=True,
            source="academic_record",
        ),
        Practitioner(
            name="Yoshua Bengio",
            domain="Machine Learning",
            tier=PractitionerTier.TOP_1_PERCENT,
            h_index=160,
            citations=450000,
            verified=True,
            source="academic_record",
        ),
        Practitioner(
            name="Leslie Lamport",
            domain="Distributed Systems",
            tier=PractitionerTier.TOP_1_PERCENT,
            h_index=80,
            citations=100000,
            verified=True,
            source="academic_record",
        ),
        Practitioner(
            name="Barbara Liskov",
            domain="Distributed Systems",
            tier=PractitionerTier.TOP_1_PERCENT,
            h_index=70,
            citations=80000,
            verified=True,
            source="academic_record",
        ),
        Practitioner(
            name="Nancy Lynch",
            domain="Distributed Systems",
            tier=PractitionerTier.TOP_1_PERCENT,
            h_index=65,
            citations=70000,
            verified=True,
            source="academic_record",
        ),
        Practitioner(
            name="Donald Knuth",
            domain="Algorithms",
            tier=PractitionerTier.TOP_1_PERCENT,
            h_index=90,
            citations=200000,
            verified=True,
            source="academic_record",
        ),
        Practitioner(
            name="Robert Tarjan",
            domain="Algorithms",
            tier=PractitionerTier.TOP_1_PERCENT,
            h_index=85,
            citations=150000,
            verified=True,
            source="academic_record",
        ),
        Practitioner(
            name="Edsger Dijkstra",
            domain="Algorithms",
            tier=PractitionerTier.TOP_1_PERCENT,
            h_index=75,
            citations=180000,
            verified=True,
            source="academic_record",
        ),
    ]

    for p in test_practitioners:
        protocol.practitioner_registry.register_practitioner(p)

    print(f"\nRegistered {len(test_practitioners)} test practitioners")
    print(f"Domains: {list(protocol.practitioner_registry.domain_index.keys())}")

    # Test case 1: Valid response with multiple domains
    class MockSynthesisResult:
        synthesized_content = """
        This synthesis draws on foundational work by Hinton (2012) in deep learning,
        combined with distributed consensus principles from Lamport (1998).
        The algorithmic foundations trace back to Knuth (1973) and Tarjan (1983).
        """
        metadata = {
            "practitioners": {
                "Machine Learning": [
                    {"name": "Geoffrey Hinton", "tier": "top_1_percent", "verified": True},
                    {"name": "Yann LeCun", "tier": "top_1_percent", "verified": True},
                    {"name": "Yoshua Bengio", "tier": "top_1_percent", "verified": True},
                ],
                "Distributed Systems": [
                    {"name": "Leslie Lamport", "tier": "top_1_percent", "verified": True},
                    {"name": "Barbara Liskov", "tier": "top_1_percent", "verified": True},
                    {"name": "Nancy Lynch", "tier": "top_1_percent", "verified": True},
                ],
                "Algorithms": [
                    {"name": "Donald Knuth", "tier": "top_1_percent", "verified": True},
                    {"name": "Robert Tarjan", "tier": "top_1_percent", "verified": True},
                    {"name": "Edsger Dijkstra", "tier": "top_1_percent", "verified": True},
                ],
            }
        }
        domains_covered = {"Machine Learning", "Distributed Systems", "Algorithms"}

    print("\nTest 1: Valid multi-domain synthesis")
    result = await protocol.validate(MockSynthesisResult())
    print(f"  Valid: {result.valid}")
    print(f"  Total domains: {result.total_domains}")
    print(f"  Elite practitioners: {result.total_elite_practitioners}")
    print(f"  Novelty score: {result.novelty_score:.4f}")
    if result.reason:
        print(f"  Reason: {result.reason}")

    # Test case 2: Invalid response with insufficient domains
    class MockInsufficientDomains:
        synthesized_content = "This is a simple response without proper citations."
        metadata = {
            "practitioners": {
                "Machine Learning": [
                    {"name": "Geoffrey Hinton", "tier": "top_1_percent", "verified": True},
                ],
            }
        }
        domains_covered = {"Machine Learning"}

    print("\nTest 2: Insufficient domains")
    result = await protocol.validate(MockInsufficientDomains())
    print(f"  Valid: {result.valid}")
    print(f"  Total domains: {result.total_domains}")
    print(f"  Reason: {result.reason}")

    # Test unrelatedness measure
    print("\nTest 3: Domain unrelatedness")
    measure = UnrelatednessMeasure()

    test_pairs = [
        ("Machine Learning", "Distributed Systems"),
        ("Machine Learning", "Deep Learning"),  # Should be related
        ("Algorithms", "Poetry"),  # Should be unrelated
    ]

    for domain_a, domain_b in test_pairs:
        distance = measure.compute_distance(domain_a, domain_b)
        is_unrelated = measure.is_unrelated(domain_a, domain_b)
        print(f"  {domain_a} - {domain_b}:")
        print(f"    Distance: {distance:.4f}, Unrelated: {is_unrelated}")

    # Generate receipt
    print("\nReceipt:")
    result = await protocol.validate(MockSynthesisResult())
    receipt = result.generate_receipt()
    for key, value in receipt.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
