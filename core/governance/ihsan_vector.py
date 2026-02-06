"""
BIZRA Canonical 8-Dimension Ihsan Vector - Constitutional Excellence Enforcement

Standing on Giants:
- Al-Ghazali (1058-1111): Ihsan (excellence) as spiritual refinement
- Shannon (1948): Information-theoretic signal quality
- de Moura (2008): Z3 SMT solver for formal verification

Constitutional Principle:
"Ihsan (excellence) is to worship Allah as if you see Him,
 and if you do not see Him, He certainly sees you." - Prophetic Hadith

Ihsan is not a soft metric - it is a HARD GATE enforced at multiple levels
based on execution context. The 8 dimensions represent verifiable qualities
that collectively ensure sovereign inference meets constitutional standards.

8 Canonical Dimensions with Verification Methods:
1. correctness (0.22)       - z3_smt_proof
2. safety (0.22)            - aegis_lambda_zero_trust
3. user_benefit (0.14)      - proof_of_impact_receipt
4. efficiency (0.12)        - green_ai_metrics
5. auditability (0.12)      - cryptographic_receipt_completeness
6. anti_centralization (0.08) - gini_coefficient_check (Gini <= 0.35)
7. robustness (0.06)        - sape_9_probe_survival
8. fairness (0.04)          - kl_divergence_bias_check

Execution Context Thresholds:
- Development:  >= 0.85, 4+ dimensions verified
- Staging:      >= 0.90, 6+ dimensions verified
- Production:   >= 0.95, 8 dimensions verified
- Critical:     >= 0.99, 8 dimensions verified + manual_review

Complexity: O(1) weighted sum, O(n) verification where n = number of dimensions
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - CANONICAL WEIGHTS (MUST SUM TO 1.0)
# =============================================================================

# Dimension weights - constitutional values, do not modify
CANONICAL_WEIGHTS: Dict[str, float] = {
    "correctness": 0.22,
    "safety": 0.22,
    "user_benefit": 0.14,
    "efficiency": 0.12,
    "auditability": 0.12,
    "anti_centralization": 0.08,
    "robustness": 0.06,
    "fairness": 0.04,
}

# Verification method identifiers
VERIFY_METHODS: Dict[str, str] = {
    "correctness": "z3_smt_proof",
    "safety": "aegis_lambda_zero_trust",
    "user_benefit": "proof_of_impact_receipt",
    "efficiency": "green_ai_metrics",
    "auditability": "cryptographic_receipt_completeness",
    "anti_centralization": "gini_coefficient_check",
    "robustness": "sape_9_probe_survival",
    "fairness": "kl_divergence_bias_check",
}

# Anti-centralization Gini threshold (stricter than ADL invariant's 0.40)
ANTI_CENTRALIZATION_GINI_THRESHOLD: float = 0.35


# =============================================================================
# EXECUTION CONTEXT
# =============================================================================


class ExecutionContext(str, Enum):
    """
    Execution environment determining verification strictness.

    Each context has specific thresholds reflecting operational requirements:
    - Development: Loose constraints for rapid iteration
    - Staging: Moderate constraints for pre-production validation
    - Production: Strict constraints for live operations
    - Critical: Maximum constraints for high-stakes decisions
    """

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    CRITICAL = "critical"


# Context-specific thresholds: (min_score, min_verified_dims, requires_manual_review)
CONTEXT_THRESHOLDS: Dict[ExecutionContext, Tuple[float, int, bool]] = {
    ExecutionContext.DEVELOPMENT: (0.85, 4, False),
    ExecutionContext.STAGING: (0.90, 6, False),
    ExecutionContext.PRODUCTION: (0.95, 8, False),
    ExecutionContext.CRITICAL: (0.99, 8, True),
}


# =============================================================================
# DIMENSION ID ENUM
# =============================================================================


class DimensionId(IntEnum):
    """
    8-dimensional Ihsan canonical ordering.

    The ordering reflects priority - correctness and safety lead
    because they are non-negotiable prerequisites for excellence.
    """

    CORRECTNESS = 0
    SAFETY = 1
    USER_BENEFIT = 2
    EFFICIENCY = 3
    AUDITABILITY = 4
    ANTI_CENTRALIZATION = 5
    ROBUSTNESS = 6
    FAIRNESS = 7

    @property
    def weight(self) -> float:
        """Get canonical weight for this dimension."""
        return CANONICAL_WEIGHTS[self.name.lower()]

    @property
    def verify_method(self) -> str:
        """Get verification method identifier for this dimension."""
        return VERIFY_METHODS[self.name.lower()]


# =============================================================================
# IHSAN DIMENSION DATACLASS
# =============================================================================


@dataclass
class IhsanDimension:
    """
    A single Ihsan dimension with verification state.

    Each dimension tracks:
    - id: Which dimension (from DimensionId enum)
    - weight: Canonical weight (0.04-0.22)
    - score: Current score [0.0, 1.0]
    - verified: Whether verification has passed
    - verify_method: The method used for verification

    Verification is independent of score - a high score without verification
    is worth less than a moderate score with proof.
    """

    id: DimensionId
    weight: float
    score: float = 0.0
    verified: bool = False
    verify_method: str = ""
    verification_timestamp: Optional[str] = None
    verification_proof: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate dimension constraints."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be in [0, 1], got {self.score}")
        if not 0.0 < self.weight <= 1.0:
            raise ValueError(f"Weight must be in (0, 1], got {self.weight}")
        if not self.verify_method:
            self.verify_method = self.id.verify_method

    @property
    def weighted_score(self) -> float:
        """Return weight * score."""
        return self.weight * self.score

    @property
    def verified_weighted_score(self) -> float:
        """Return weight * score if verified, else 0."""
        return self.weight * self.score if self.verified else 0.0

    def mark_verified(
        self,
        proof: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> "IhsanDimension":
        """
        Mark dimension as verified with optional proof.

        Returns a new IhsanDimension instance (immutability pattern).
        """
        ts = timestamp or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return IhsanDimension(
            id=self.id,
            weight=self.weight,
            score=self.score,
            verified=True,
            verify_method=self.verify_method,
            verification_timestamp=ts,
            verification_proof=proof,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for receipts and persistence."""
        return {
            "id": self.id.name.lower(),
            "weight": self.weight,
            "score": round(self.score, 6),
            "verified": self.verified,
            "verify_method": self.verify_method,
            "verification_timestamp": self.verification_timestamp,
            "verification_proof": self.verification_proof,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IhsanDimension":
        """Reconstruct from serialized form."""
        dim_id = DimensionId[data["id"].upper()]
        return cls(
            id=dim_id,
            weight=data.get("weight", dim_id.weight),
            score=data.get("score", 0.0),
            verified=data.get("verified", False),
            verify_method=data.get("verify_method", dim_id.verify_method),
            verification_timestamp=data.get("verification_timestamp"),
            verification_proof=data.get("verification_proof"),
        )


# =============================================================================
# IHSAN VECTOR CLASS
# =============================================================================


@dataclass
class IhsanVector:
    """
    The canonical 8-dimension Ihsan vector.

    Holds all 8 dimensions with their scores and verification states.
    Provides aggregate scoring and context-aware threshold validation.

    Usage:
        # Create with scores
        vec = IhsanVector.from_scores(
            correctness=0.98,
            safety=0.97,
            user_benefit=0.92,
            efficiency=0.88,
            auditability=0.90,
            anti_centralization=0.85,
            robustness=0.82,
            fairness=0.80,
        )

        # Calculate aggregate score
        score = vec.calculate_score()  # Weighted sum

        # Verify against execution context
        result = vec.verify_thresholds(ExecutionContext.PRODUCTION)

        # Generate receipt
        receipt = vec.to_receipt()
    """

    dimensions: Dict[DimensionId, IhsanDimension] = field(default_factory=dict)
    created_at: str = ""
    context: Optional[ExecutionContext] = None

    def __post_init__(self) -> None:
        """Initialize default dimensions if not provided."""
        if not self.created_at:
            self.created_at = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )

        # Ensure all 8 dimensions exist
        for dim_id in DimensionId:
            if dim_id not in self.dimensions:
                self.dimensions[dim_id] = IhsanDimension(
                    id=dim_id,
                    weight=dim_id.weight,
                    score=0.0,
                    verified=False,
                )

        # Validate weight sum
        total_weight = sum(d.weight for d in self.dimensions.values())
        if abs(total_weight - 1.0) > 1e-6:
            logger.warning(f"Dimension weights sum to {total_weight}, expected 1.0")

    @classmethod
    def from_scores(
        cls,
        correctness: float = 0.0,
        safety: float = 0.0,
        user_benefit: float = 0.0,
        efficiency: float = 0.0,
        auditability: float = 0.0,
        anti_centralization: float = 0.0,
        robustness: float = 0.0,
        fairness: float = 0.0,
        context: Optional[ExecutionContext] = None,
    ) -> "IhsanVector":
        """
        Create IhsanVector from score values.

        All scores must be in [0, 1]. Verification state defaults to False.
        """
        scores = {
            DimensionId.CORRECTNESS: correctness,
            DimensionId.SAFETY: safety,
            DimensionId.USER_BENEFIT: user_benefit,
            DimensionId.EFFICIENCY: efficiency,
            DimensionId.AUDITABILITY: auditability,
            DimensionId.ANTI_CENTRALIZATION: anti_centralization,
            DimensionId.ROBUSTNESS: robustness,
            DimensionId.FAIRNESS: fairness,
        }

        dimensions = {}
        for dim_id, score in scores.items():
            dimensions[dim_id] = IhsanDimension(
                id=dim_id,
                weight=dim_id.weight,
                score=score,
                verified=False,
            )

        return cls(dimensions=dimensions, context=context)

    @classmethod
    def neutral(cls) -> "IhsanVector":
        """Create a neutral vector (all scores 0.5, unverified)."""
        return cls.from_scores(
            correctness=0.5,
            safety=0.5,
            user_benefit=0.5,
            efficiency=0.5,
            auditability=0.5,
            anti_centralization=0.5,
            robustness=0.5,
            fairness=0.5,
        )

    @classmethod
    def perfect(cls) -> "IhsanVector":
        """Create a perfect vector (all scores 1.0, unverified)."""
        return cls.from_scores(
            correctness=1.0,
            safety=1.0,
            user_benefit=1.0,
            efficiency=1.0,
            auditability=1.0,
            anti_centralization=1.0,
            robustness=1.0,
            fairness=1.0,
        )

    # -------------------------------------------------------------------------
    # SCORE CALCULATION
    # -------------------------------------------------------------------------

    def calculate_score(self) -> float:
        """
        Calculate weighted sum of all dimension scores.

        Formula: sum(weight_i * score_i) for i in [0..7]

        This considers all scores regardless of verification state.
        For verified-only scoring, use calculate_verified_score().

        Returns:
            float: Aggregate Ihsan score in [0, 1]
        """
        return sum(d.weighted_score for d in self.dimensions.values())

    def calculate_verified_score(self) -> float:
        """
        Calculate weighted sum of verified dimension scores only.

        Unverified dimensions contribute 0 to the score.

        Returns:
            float: Verified aggregate Ihsan score in [0, 1]
        """
        return sum(d.verified_weighted_score for d in self.dimensions.values())

    @property
    def verified_count(self) -> int:
        """Count of verified dimensions."""
        return sum(1 for d in self.dimensions.values() if d.verified)

    @property
    def unverified_count(self) -> int:
        """Count of unverified dimensions."""
        return 8 - self.verified_count

    @property
    def all_verified(self) -> bool:
        """True if all 8 dimensions are verified."""
        return self.verified_count == 8

    # -------------------------------------------------------------------------
    # THRESHOLD VERIFICATION
    # -------------------------------------------------------------------------

    def verify_thresholds(
        self,
        context: ExecutionContext,
        manual_review_approved: bool = False,
    ) -> "ThresholdResult":
        """
        Verify vector against execution context thresholds.

        Checks:
        1. Aggregate score >= min_score for context
        2. Verified dimension count >= min_verified_dims for context
        3. If context requires manual review, manual_review_approved must be True

        Args:
            context: The execution context (development, staging, production, critical)
            manual_review_approved: Whether manual review has been completed

        Returns:
            ThresholdResult with pass/fail status and detailed breakdown
        """
        min_score, min_verified, requires_manual = CONTEXT_THRESHOLDS[context]

        aggregate_score = self.calculate_score()
        verified_dims = self.verified_count

        # Check conditions
        score_passed = aggregate_score >= min_score
        dims_passed = verified_dims >= min_verified
        manual_passed = not requires_manual or manual_review_approved

        # Determine overall pass/fail
        passed = score_passed and dims_passed and manual_passed

        # Build detailed failures list
        failures = []
        if not score_passed:
            failures.append(
                f"Score {aggregate_score:.4f} < {min_score} required for {context.value}"
            )
        if not dims_passed:
            failures.append(
                f"Verified dimensions {verified_dims}/8 < {min_verified} required for {context.value}"
            )
        if not manual_passed:
            failures.append(
                f"Manual review required for {context.value} but not approved"
            )

        return ThresholdResult(
            passed=passed,
            context=context,
            aggregate_score=aggregate_score,
            verified_count=verified_dims,
            required_score=min_score,
            required_verified=min_verified,
            requires_manual_review=requires_manual,
            manual_review_approved=manual_review_approved,
            failures=failures,
            dimension_summary={
                d.id.name.lower(): {
                    "score": d.score,
                    "verified": d.verified,
                    "weighted": d.weighted_score,
                }
                for d in self.dimensions.values()
            },
        )

    def passes_context(
        self,
        context: ExecutionContext,
        manual_review_approved: bool = False,
    ) -> bool:
        """Quick check if vector passes context thresholds."""
        return self.verify_thresholds(context, manual_review_approved).passed

    # -------------------------------------------------------------------------
    # DIMENSION ACCESS AND UPDATE
    # -------------------------------------------------------------------------

    def get_dimension(self, dim_id: DimensionId) -> IhsanDimension:
        """Get a specific dimension."""
        return self.dimensions[dim_id]

    def set_score(self, dim_id: DimensionId, score: float) -> "IhsanVector":
        """
        Set score for a dimension (returns new vector, immutability pattern).

        This resets verification state for the dimension.
        """
        new_dims = {k: v for k, v in self.dimensions.items()}
        new_dims[dim_id] = IhsanDimension(
            id=dim_id,
            weight=dim_id.weight,
            score=score,
            verified=False,
        )
        return IhsanVector(dimensions=new_dims, context=self.context)

    def verify_dimension(
        self,
        dim_id: DimensionId,
        proof: Optional[str] = None,
    ) -> "IhsanVector":
        """
        Mark a dimension as verified (returns new vector, immutability pattern).
        """
        new_dims = {k: v for k, v in self.dimensions.items()}
        new_dims[dim_id] = self.dimensions[dim_id].mark_verified(proof)
        return IhsanVector(dimensions=new_dims, context=self.context)

    def verify_all(
        self, proofs: Optional[Dict[DimensionId, str]] = None
    ) -> "IhsanVector":
        """
        Mark all dimensions as verified (returns new vector).

        Args:
            proofs: Optional dict mapping dimension IDs to proof strings
        """
        proofs = proofs or {}
        new_dims = {}
        for dim_id, dim in self.dimensions.items():
            proof = proofs.get(dim_id)
            new_dims[dim_id] = dim.mark_verified(proof)
        return IhsanVector(dimensions=new_dims, context=self.context)

    # -------------------------------------------------------------------------
    # SERIALIZATION
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "created_at": self.created_at,
            "context": self.context.value if self.context else None,
            "aggregate_score": round(self.calculate_score(), 6),
            "verified_score": round(self.calculate_verified_score(), 6),
            "verified_count": self.verified_count,
            "dimensions": {
                d.id.name.lower(): d.to_dict() for d in self.dimensions.values()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IhsanVector":
        """Reconstruct from serialized form."""
        dimensions = {}
        for key, dim_data in data.get("dimensions", {}).items():
            dim = IhsanDimension.from_dict(dim_data)
            dimensions[dim.id] = dim

        context = None
        if data.get("context"):
            context = ExecutionContext(data["context"])

        vec = cls(dimensions=dimensions, context=context)
        vec.created_at = data.get("created_at", vec.created_at)
        return vec

    def to_receipt(self) -> "IhsanReceipt":
        """
        Generate canonical receipt for audit trail.

        The receipt contains:
        - All dimension scores and verification states
        - Aggregate metrics
        - Cryptographic hash for integrity verification
        - Timestamp
        """
        receipt_data = self.to_dict()
        receipt_json = json.dumps(receipt_data, sort_keys=True, separators=(",", ":"))
        receipt_hash = hashlib.sha256(receipt_json.encode()).hexdigest()

        return IhsanReceipt(
            vector=self,
            receipt_hash=receipt_hash,
            receipt_json=receipt_json,
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )


# =============================================================================
# THRESHOLD RESULT
# =============================================================================


@dataclass
class ThresholdResult:
    """
    Result of threshold verification against execution context.

    Provides detailed breakdown of pass/fail criteria.
    """

    passed: bool
    context: ExecutionContext
    aggregate_score: float
    verified_count: int
    required_score: float
    required_verified: int
    requires_manual_review: bool
    manual_review_approved: bool
    failures: List[str]
    dimension_summary: Dict[str, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging and reporting."""
        return {
            "passed": self.passed,
            "context": self.context.value,
            "aggregate_score": round(self.aggregate_score, 6),
            "verified_count": self.verified_count,
            "required_score": self.required_score,
            "required_verified": self.required_verified,
            "requires_manual_review": self.requires_manual_review,
            "manual_review_approved": self.manual_review_approved,
            "failures": self.failures,
            "dimension_summary": self.dimension_summary,
        }


# =============================================================================
# IHSAN RECEIPT
# =============================================================================


@dataclass
class IhsanReceipt:
    """
    Cryptographic receipt for Ihsan vector state.

    Provides tamper-evident record of vector state at a point in time.
    Use this for audit trails and compliance verification.
    """

    vector: IhsanVector
    receipt_hash: str
    receipt_json: str
    timestamp: str

    def verify_integrity(self) -> bool:
        """Verify receipt hash matches content."""
        computed_hash = hashlib.sha256(self.receipt_json.encode()).hexdigest()
        return computed_hash == self.receipt_hash

    def to_dict(self) -> Dict[str, Any]:
        """Serialize receipt for persistence."""
        return {
            "receipt_hash": self.receipt_hash,
            "timestamp": self.timestamp,
            "vector": self.vector.to_dict(),
            "integrity_valid": self.verify_integrity(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IhsanReceipt":
        """Reconstruct receipt from serialized form."""
        vector = IhsanVector.from_dict(data["vector"])
        receipt_json = json.dumps(data["vector"], sort_keys=True, separators=(",", ":"))
        return cls(
            vector=vector,
            receipt_hash=data["receipt_hash"],
            receipt_json=receipt_json,
            timestamp=data["timestamp"],
        )


# =============================================================================
# VERIFICATION FUNCTIONS
# =============================================================================


def create_verifier(
    verify_func: Callable[[IhsanDimension], Tuple[bool, Optional[str]]],
) -> Callable[[IhsanVector, DimensionId], IhsanVector]:
    """
    Create a verifier function for a dimension.

    Args:
        verify_func: Function that takes a dimension and returns (passed, proof)

    Returns:
        Function that verifies a dimension in a vector
    """

    def verifier(vector: IhsanVector, dim_id: DimensionId) -> IhsanVector:
        dim = vector.get_dimension(dim_id)
        passed, proof = verify_func(dim)
        if passed:
            return vector.verify_dimension(dim_id, proof)
        return vector

    return verifier


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def quick_ihsan(
    correctness: float,
    safety: float,
    user_benefit: float,
    efficiency: float,
    auditability: float,
    anti_centralization: float,
    robustness: float,
    fairness: float,
) -> float:
    """
    Quick calculation of Ihsan score without creating full vector.

    Args:
        correctness: Correctness score [0, 1]
        safety: Safety score [0, 1]
        user_benefit: User benefit score [0, 1]
        efficiency: Efficiency score [0, 1]
        auditability: Auditability score [0, 1]
        anti_centralization: Anti-centralization score [0, 1]
        robustness: Robustness score [0, 1]
        fairness: Fairness score [0, 1]

    Returns:
        Weighted Ihsan score in [0, 1]
    """
    return (
        correctness * 0.22
        + safety * 0.22
        + user_benefit * 0.14
        + efficiency * 0.12
        + auditability * 0.12
        + anti_centralization * 0.08
        + robustness * 0.06
        + fairness * 0.04
    )


def passes_production(
    correctness: float,
    safety: float,
    user_benefit: float,
    efficiency: float,
    auditability: float,
    anti_centralization: float,
    robustness: float,
    fairness: float,
) -> bool:
    """
    Quick check if scores would pass production threshold (0.95).

    Note: This does not check verification state.
    """
    score = quick_ihsan(
        correctness,
        safety,
        user_benefit,
        efficiency,
        auditability,
        anti_centralization,
        robustness,
        fairness,
    )
    return score >= 0.95


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "CANONICAL_WEIGHTS",
    "VERIFY_METHODS",
    "ANTI_CENTRALIZATION_GINI_THRESHOLD",
    "CONTEXT_THRESHOLDS",
    # Enums
    "ExecutionContext",
    "DimensionId",
    # Classes
    "IhsanDimension",
    "IhsanVector",
    "ThresholdResult",
    "IhsanReceipt",
    # Functions
    "create_verifier",
    "quick_ihsan",
    "passes_production",
]
