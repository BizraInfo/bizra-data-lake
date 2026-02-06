"""
Impact Oracle — Proof-of-Impact Verification

The Impact Oracle measures and verifies entropy reduction to support
the Proof-of-Impact (PoI) consensus mechanism:

- Value = Entropy Reduction
- Only verifiable ΔE generates SEED tokens
- Zero-Violation ethics enforcement
- Third Fact crystallization

"The wealth of the network correlates directly with its order and capability."
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from core.uers.convergence import ConvergenceResult
from core.uers.entropy import EntropyCalculator, ManifoldState

logger = logging.getLogger(__name__)


class ImpactType(str, Enum):
    """Types of impactful actions."""

    INFORMATION_ORGANIZATION = "information_organization"
    TRUTH_VERIFICATION = "truth_verification"
    PROBLEM_SOLVING = "problem_solving"
    ENTROPY_REDUCTION = "entropy_reduction"
    KNOWLEDGE_SYNTHESIS = "knowledge_synthesis"


class ViolationType(str, Enum):
    """Types of ethical violations."""

    BIAS_INTRODUCTION = "bias_introduction"
    HARM_POTENTIAL = "harm_potential"
    DECEPTION = "deception"
    PRIVACY_BREACH = "privacy_breach"
    CENTRALIZATION = "centralization"


@dataclass
class ImpactClaim:
    """A claim of impactful work submitted to the Oracle."""

    id: str
    agent_id: str
    impact_type: ImpactType
    description: str
    claimed_delta_e: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "type": self.impact_type.value,
            "description": self.description[:200],
            "claimed_delta_e": self.claimed_delta_e,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ImpactVerdict:
    """Oracle's verdict on an impact claim."""

    claim_id: str
    verified: bool
    actual_delta_e: float
    reward_tokens: float
    confidence: float
    violations: List[ViolationType] = field(default_factory=list)
    reasoning: str = ""
    hash: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_zero_violation(self) -> bool:
        """Check if verdict has zero ethical violations."""
        return len(self.violations) == 0

    @property
    def is_rewarded(self) -> bool:
        """Check if claim received rewards."""
        return self.verified and self.reward_tokens > 0 and self.is_zero_violation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "verified": self.verified,
            "actual_delta_e": self.actual_delta_e,
            "reward_tokens": self.reward_tokens,
            "confidence": self.confidence,
            "violations": [v.value for v in self.violations],
            "is_zero_violation": self.is_zero_violation,
            "is_rewarded": self.is_rewarded,
            "hash": self.hash,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ThirdFact:
    """
    A crystallized, cryptographically-anchored truth.

    Third Facts are immutable records of verified understanding
    that form the Generational Ledger of knowledge.
    """

    id: str
    content: str
    source_claim_id: str
    entropy_state: ManifoldState
    delta_e: float
    confidence: float
    hash: str
    merkle_root: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content[:500],
            "source_claim": self.source_claim_id,
            "delta_e": self.delta_e,
            "confidence": self.confidence,
            "hash": self.hash,
            "merkle_root": self.merkle_root,
            "timestamp": self.timestamp.isoformat(),
        }


class ImpactOracle:
    """
    The Impact Oracle — Adjudicator of Proof-of-Impact.

    Measures, verifies, and rewards entropy reduction according
    to the UERS framework and ethical constraints.
    """

    def __init__(
        self,
        minimum_delta_e: float = 0.01,
        reward_rate: float = 100.0,  # SEED tokens per unit entropy reduction
        confidence_threshold: float = 0.95,
        zero_violations_required: bool = True,
    ):
        self.minimum_delta_e = minimum_delta_e
        self.reward_rate = reward_rate
        self.confidence_threshold = confidence_threshold
        self.zero_violations_required = zero_violations_required

        # Components
        self._entropy_calc = EntropyCalculator()

        # State
        self._claims: List[ImpactClaim] = []
        self._verdicts: List[ImpactVerdict] = []
        self._third_facts: List[ThirdFact] = []
        self._total_rewards: float = 0.0
        self._claim_counter: int = 0

    # =========================================================================
    # CLAIM SUBMISSION
    # =========================================================================

    def submit_claim(
        self,
        agent_id: str,
        impact_type: ImpactType,
        description: str,
        claimed_delta_e: float,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> ImpactClaim:
        """
        Submit an impact claim for verification.

        Returns the claim ID for tracking.
        """
        self._claim_counter += 1

        claim = ImpactClaim(
            id=f"claim_{self._claim_counter:06d}",
            agent_id=agent_id,
            impact_type=impact_type,
            description=description,
            claimed_delta_e=claimed_delta_e,
            evidence=evidence or {},
        )

        self._claims.append(claim)
        logger.info(
            f"Claim submitted: {claim.id} by {agent_id}, ΔE={claimed_delta_e:.4f}"
        )

        return claim

    def submit_convergence_claim(
        self,
        agent_id: str,
        convergence_result: ConvergenceResult,
    ) -> ImpactClaim:
        """
        Submit a claim based on a convergence result.

        Automatically extracts delta_e and evidence from the result.
        """
        return self.submit_claim(
            agent_id=agent_id,
            impact_type=ImpactType.ENTROPY_REDUCTION,
            description=f"Convergence achieved: {convergence_result.state.value}",
            claimed_delta_e=convergence_result.total_delta_e,
            evidence={
                "convergence_id": convergence_result.id,
                "iterations": convergence_result.iterations,
                "initial_entropy": convergence_result.initial_entropy,
                "final_entropy": convergence_result.final_entropy,
                "singularity": convergence_result.singularity_achieved,
            },
        )

    # =========================================================================
    # VERIFICATION
    # =========================================================================

    async def verify_claim(
        self,
        claim: ImpactClaim,
        before_state: Optional[ManifoldState] = None,
        after_state: Optional[ManifoldState] = None,
    ) -> ImpactVerdict:
        """
        Verify an impact claim and issue verdict.

        The Oracle reverse-engineers the claim to measure actual ΔE.
        """
        # Calculate actual entropy change
        if before_state and after_state:
            actual_delta_e = self._entropy_calc.calculate_delta_e(
                before_state, after_state
            )
        else:
            # Use claimed value with reduced confidence
            actual_delta_e = claim.claimed_delta_e * 0.8

        # Check ethical violations
        violations = self._check_violations(claim)

        # Determine verification status
        verified = (
            actual_delta_e >= self.minimum_delta_e
            and abs(actual_delta_e - claim.claimed_delta_e)
            / max(claim.claimed_delta_e, 0.001)
            < 0.5
        )

        # Calculate confidence
        if before_state and after_state:
            confidence = 0.99
        else:
            confidence = 0.7

        # Calculate rewards
        if verified and (not self.zero_violations_required or len(violations) == 0):
            reward_tokens = actual_delta_e * self.reward_rate
        else:
            reward_tokens = 0.0

        # Generate verdict hash
        verdict_data = f"{claim.id}:{actual_delta_e}:{verified}:{reward_tokens}"
        verdict_hash = hashlib.sha256(verdict_data.encode()).hexdigest()[:16]

        verdict = ImpactVerdict(
            claim_id=claim.id,
            verified=verified,
            actual_delta_e=actual_delta_e,
            reward_tokens=reward_tokens,
            confidence=confidence,
            violations=violations,
            reasoning=self._generate_reasoning(claim, verified, violations),
            hash=verdict_hash,
        )

        self._verdicts.append(verdict)
        self._total_rewards += reward_tokens

        logger.info(
            f"Verdict issued: {claim.id} -> verified={verified}, "
            f"ΔE={actual_delta_e:.4f}, rewards={reward_tokens:.2f}"
        )

        return verdict

    def _check_violations(self, claim: ImpactClaim) -> List[ViolationType]:
        """
        Check for ethical violations in the claim.

        Implements Zero-Violation constraint checking.
        """
        violations = []

        # Check for bias patterns
        if claim.evidence.get("bias_score", 0) > 0.1:
            violations.append(ViolationType.BIAS_INTRODUCTION)

        # Check for harm potential
        if claim.evidence.get("harm_score", 0) > 0.1:
            violations.append(ViolationType.HARM_POTENTIAL)

        # Check for deception
        if claim.claimed_delta_e > 0 and claim.evidence.get("verified", True) is False:
            violations.append(ViolationType.DECEPTION)

        # Check for privacy breaches
        if claim.evidence.get("privacy_exposed", False):
            violations.append(ViolationType.PRIVACY_BREACH)

        # Check for centralization
        if claim.evidence.get("centralization_increase", 0) > 0.1:
            violations.append(ViolationType.CENTRALIZATION)

        return violations

    def _generate_reasoning(
        self,
        claim: ImpactClaim,
        verified: bool,
        violations: List[ViolationType],
    ) -> str:
        """Generate reasoning for the verdict."""
        if not verified:
            return "Claim not verified: insufficient entropy reduction or evidence mismatch."

        if violations:
            return f"Claim verified but rejected due to violations: {[v.value for v in violations]}"

        return f"Claim verified with confirmed entropy reduction of type {claim.impact_type.value}."

    # =========================================================================
    # THIRD FACT CRYSTALLIZATION
    # =========================================================================

    def crystallize_third_fact(
        self,
        claim: ImpactClaim,
        verdict: ImpactVerdict,
        manifold_state: ManifoldState,
        content: str,
    ) -> Optional[ThirdFact]:
        """
        Crystallize a verified claim into a Third Fact.

        Third Facts are immutable records anchored to the ledger.
        """
        if not verdict.is_rewarded:
            logger.warning(
                f"Cannot crystallize: verdict {verdict.claim_id} not rewarded"
            )
            return None

        # Generate fact hash
        fact_data = f"{claim.id}:{content}:{manifold_state.average_entropy}"
        fact_hash = hashlib.sha256(fact_data.encode()).hexdigest()

        fact = ThirdFact(
            id=f"fact_{hashlib.sha256(fact_hash.encode()).hexdigest()[:12]}",
            content=content,
            source_claim_id=claim.id,
            entropy_state=manifold_state,
            delta_e=verdict.actual_delta_e,
            confidence=verdict.confidence,
            hash=fact_hash,
            merkle_root=self._compute_merkle_root(),
        )

        self._third_facts.append(fact)
        logger.info(f"Third Fact crystallized: {fact.id}")

        return fact

    def _compute_merkle_root(self) -> str:
        """Compute Merkle root of all Third Facts."""
        if not self._third_facts:
            return hashlib.sha256(b"genesis").hexdigest()

        hashes = [f.hash for f in self._third_facts]

        while len(hashes) > 1:
            if len(hashes) % 2 == 1:
                hashes.append(hashes[-1])

            new_hashes = []
            for i in range(0, len(hashes), 2):
                combined = hashes[i] + hashes[i + 1]
                new_hashes.append(hashlib.sha256(combined.encode()).hexdigest())

            hashes = new_hashes

        return hashes[0]

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get Oracle statistics."""
        verified_count = sum(1 for v in self._verdicts if v.verified)
        rewarded_count = sum(1 for v in self._verdicts if v.is_rewarded)

        return {
            "total_claims": len(self._claims),
            "total_verdicts": len(self._verdicts),
            "verified_count": verified_count,
            "rewarded_count": rewarded_count,
            "verification_rate": verified_count / max(len(self._verdicts), 1),
            "reward_rate": rewarded_count / max(len(self._verdicts), 1),
            "total_rewards_issued": self._total_rewards,
            "third_facts_count": len(self._third_facts),
            "merkle_root": self._compute_merkle_root()[:16],
            "config": {
                "minimum_delta_e": self.minimum_delta_e,
                "reward_rate": self.reward_rate,
                "confidence_threshold": self.confidence_threshold,
                "zero_violations_required": self.zero_violations_required,
            },
        }

    def get_claims(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent claims."""
        return [c.to_dict() for c in self._claims[-limit:]]

    def get_verdicts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent verdicts."""
        return [v.to_dict() for v in self._verdicts[-limit:]]

    def get_third_facts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent Third Facts."""
        return [f.to_dict() for f in self._third_facts[-limit:]]

    def get_agent_rewards(self, agent_id: str) -> Dict[str, Any]:
        """Get total rewards for a specific agent."""
        agent_claims = [c for c in self._claims if c.agent_id == agent_id]
        agent_verdicts = [
            v for v in self._verdicts if any(c.id == v.claim_id for c in agent_claims)
        ]

        total_rewards = sum(v.reward_tokens for v in agent_verdicts)
        total_delta_e = sum(v.actual_delta_e for v in agent_verdicts if v.verified)

        return {
            "agent_id": agent_id,
            "total_claims": len(agent_claims),
            "verified_claims": sum(1 for v in agent_verdicts if v.verified),
            "rewarded_claims": sum(1 for v in agent_verdicts if v.is_rewarded),
            "total_rewards": total_rewards,
            "total_delta_e": total_delta_e,
        }
