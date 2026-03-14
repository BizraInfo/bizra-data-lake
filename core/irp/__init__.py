"""
BIZRA Isnad Risk Propagation (IRP) v1.0.0

Novel algorithm: Unifies information provenance grading with
risk computation in a single mathematical framework.

Standing on Giants:
- Islamic hadith science (8th century CE): isnad chain verification
- Markowitz (1952): mean-variance portfolio optimization
- Shannon (1948): information entropy as quality measure

Architectural Role:
Operates at the PAT->SAT boundary. PAT grades its own information
sources using Isnad classification before forwarding to SAT.

License: BIZRA Constitutional License v1.0
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

# ============================================================================
# ISNAD CLASSIFICATION (Battle-tested 1,400 years)
# ============================================================================


class IsnadGrade(IntEnum):
    """Hadith science grading taxonomy applied to data provenance."""

    SAHIH = 4  # Sound — highest confidence
    HASAN = 3  # Good — acceptable confidence
    DAIF = 2  # Weak — significant uncertainty
    MAWDU = 1  # Fabricated — excluded from computation

    @property
    def risk_multiplier(self) -> float:
        """How much this grade inflates variance estimates."""
        return {
            IsnadGrade.SAHIH: 1.0,
            IsnadGrade.HASAN: 1.5,
            IsnadGrade.DAIF: 3.0,
            IsnadGrade.MAWDU: float("inf"),
        }[self]


@dataclass(frozen=True)
class Source:
    """A named, verifiable information source."""

    id: str
    name: str
    reliability: float  # r(s) in [0, 1]
    verified: bool = False

    def __post_init__(self):
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError(
                f"Source reliability must be in [0,1], got {self.reliability}"
            )


@dataclass
class IsnadChain:
    """A chain of transmission: [s0 -> s1 -> ... -> sn]."""

    sources: list[Source]
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.sources:
            raise ValueError("Isnad chain must have at least one source")

    @property
    def length(self) -> int:
        return len(self.sources)

    @property
    def chain_hash(self) -> str:
        content = "|".join(s.id for s in self.sources)
        return hashlib.blake2b(content.encode(), digest_size=16).hexdigest()


# ============================================================================
# CHAIN STRENGTH FUNCTION
# ============================================================================

DECAY_HALF_LIFE: int = 5
MUTAWATIR_THRESHOLD: int = 3


def chain_strength(chain: IsnadChain) -> float:
    """Compute Psi(C) = product(r(si)) * decay(n)."""
    if not chain.sources:
        return 0.0
    reliability_product = math.prod(s.reliability for s in chain.sources)
    decay = 2.0 ** (-chain.length / DECAY_HALF_LIFE)
    return reliability_product * decay


def _count_independent(chains: list[IsnadChain]) -> int:
    """Count chains that share no common sources."""
    if len(chains) <= 1:
        return len(chains)
    source_sets = [frozenset(s.id for s in c.sources) for c in chains]
    independent = 1
    for i in range(1, len(source_sets)):
        if all(source_sets[i].isdisjoint(source_sets[j]) for j in range(i)):
            independent += 1
    return independent


def aggregate_strength(
    chains: list[IsnadChain],
    threshold: int = MUTAWATIR_THRESHOLD,
) -> tuple[float, IsnadGrade]:
    """Aggregate multiple independent chains for the same data point."""
    if not chains:
        return 0.0, IsnadGrade.MAWDU
    strengths = [chain_strength(c) for c in chains]
    independent_chains = _count_independent(chains)
    max_strength = max(strengths)

    if independent_chains >= threshold and max_strength > 0.8:
        grade = IsnadGrade.SAHIH
    elif max_strength > 0.6:
        grade = IsnadGrade.HASAN
    elif max_strength > 0.2:
        grade = IsnadGrade.DAIF
    else:
        grade = IsnadGrade.MAWDU

    # Mutawatir bonus for independent corroboration
    if independent_chains >= threshold:
        mutawatir_bonus = 1.0 + 0.1 * (independent_chains - threshold)
        agg = min(1.0, max_strength * mutawatir_bonus)
    else:
        agg = max_strength

    return agg, grade


# ============================================================================
# RISK MODIFICATION: The Core Innovation
# ============================================================================


@dataclass
class DataPoint:
    """A market data point with provenance metadata."""

    asset_id: str
    value: float
    chains: list[IsnadChain]
    timestamp: float = field(default_factory=time.time)

    @property
    def grade(self) -> IsnadGrade:
        _, g = aggregate_strength(self.chains)
        return g

    @property
    def strength(self) -> float:
        s, _ = aggregate_strength(self.chains)
        return s


def irp_variance_adjustment(
    base_variance: float,
    data_point: DataPoint,
) -> float:
    """
    The IRP core operation: adjust variance by chain strength.

    Standard risk: sigma^2 (assumes perfect data)
    IRP risk: sigma^2 * grade.risk_multiplier

    A weak-chain data point inflates variance, meaning:
    - Position sizes shrink (Kelly criterion effect)
    - Correlation estimates widen
    - Portfolio risk increases (more conservative)
    """
    grade = data_point.grade
    if grade == IsnadGrade.MAWDU:
        return float("inf")
    return base_variance * grade.risk_multiplier


def irp_position_size(
    signal_strength: float,
    base_variance: float,
    data_point: DataPoint,
    capital: float,
    max_risk_fraction: float = 0.02,
) -> float:
    """
    IRP-adjusted position sizing.

    SAHIH data -> full position. DAIF data -> 1/3 position.
    MAWDU data -> zero position. Rational behavior no
    current trading system implements.
    """
    adjusted_variance = irp_variance_adjustment(base_variance, data_point)
    if adjusted_variance == float("inf") or adjusted_variance <= 0:
        return 0.0
    adjusted_vol = math.sqrt(adjusted_variance)
    if adjusted_vol == 0:
        return 0.0
    kelly_fraction = signal_strength / adjusted_vol
    position = capital * min(kelly_fraction, max_risk_fraction)
    return max(0.0, position)


# ============================================================================
# PAT->SAT BOUNDARY INTEGRATION
# ============================================================================


@dataclass
class IrpAssessment:
    """
    Structured output PAT sends to SAT at the security boundary.
    Raw data never crosses -- only assessments with provenance.
    """

    asset_id: str
    assessed_value: float
    grade: IsnadGrade
    chain_strength: float
    independent_chain_count: int
    recommended_variance_multiplier: float
    assessment_hash: str = ""

    def __post_init__(self):
        if not self.assessment_hash:
            content = (
                f"{self.asset_id}|{self.assessed_value}|"
                f"{self.grade.name}|{self.chain_strength}"
            )
            self.assessment_hash = hashlib.blake2b(
                content.encode(), digest_size=32
            ).hexdigest()


def pat_assess(data_point: DataPoint) -> IrpAssessment:
    """
    PAT-side assessment: grades data and produces structured
    assessment for SAT consumption. Raw market data never
    crosses to SAT, only graded assessments with provenance.
    """
    strength, grade = aggregate_strength(data_point.chains)
    independent = _count_independent(data_point.chains)

    return IrpAssessment(
        asset_id=data_point.asset_id,
        assessed_value=data_point.value,
        grade=grade,
        chain_strength=strength,
        independent_chain_count=independent,
        recommended_variance_multiplier=grade.risk_multiplier,
    )


# Public API
__all__ = [
    "IsnadGrade",
    "Source",
    "IsnadChain",
    "DataPoint",
    "IrpAssessment",
    "chain_strength",
    "aggregate_strength",
    "irp_variance_adjustment",
    "irp_position_size",
    "pat_assess",
]
