"""
Isnad Risk Propagation (IRP) — Trust model from classical hadith science.

Definition 6 (Isnad Trust):
    Trust(c) = min{T(n_i) : n_i in isnad(c)}

Theorem 4 (Poison Propagation):
    If any T(n_i) = 0, then Trust(c) = 0.

Theorem 4.2 (Chain Strength):
    P(no poison) = f^k for uniform narrator trust f, chain length k.

IRP is strictly stronger than PageRank-style mean aggregation because
a single unverified link zeros the entire chain, whereas PageRank
averages out untrusted sources.

Standing on Giants:
- Al-Bukhari (810-870): Narrator-chain authentication methodology
- Ibn Hajar (1372-1449): Fath al-Bari — systematic trust classification
- Page & Brin (1998): PageRank — what IRP improves upon
- BIZRA Constitution: ZANN_ZERO demands min-trust, not mean-trust
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Narrator:
    """A single narrator in an isnad chain."""

    narrator_id: str
    trust: float  # T(n) in [0, 1]
    verification_method: str = "unknown"  # how trust was established


@dataclass
class IsnadChain:
    """A chain of narrators for a claim."""

    claim: str
    narrators: List[Narrator]


@dataclass
class TrustResult:
    """Result of IRP trust computation."""

    trust: float
    chain_length: int
    weakest_link: Optional[str] = None
    weakest_trust: float = 1.0
    poisoned: bool = False


def isnad_trust(chain: IsnadChain) -> TrustResult:
    """Compute IRP trust: Trust(c) = min{T(n_i)}.

    This is the core IRP algorithm. Unlike PageRank which averages,
    IRP takes the minimum — one weak link zeros everything.
    """
    if not chain.narrators:
        return TrustResult(trust=0.0, chain_length=0, poisoned=True)

    min_trust = 1.0
    weakest_id = chain.narrators[0].narrator_id

    for narrator in chain.narrators:
        if narrator.trust < min_trust:
            min_trust = narrator.trust
            weakest_id = narrator.narrator_id

    return TrustResult(
        trust=min_trust,
        chain_length=len(chain.narrators),
        weakest_link=weakest_id,
        weakest_trust=min_trust,
        poisoned=min_trust == 0.0,
    )


def chain_strength_probability(
    trust_per_narrator: float,
    chain_length: int,
) -> float:
    """P(no poison) = f^k for uniform narrator trust f, chain length k.

    Theorem 4.2: This decays exponentially with chain length,
    providing stronger guarantees than mean aggregation.
    """
    if chain_length <= 0:
        return 0.0
    if not (0.0 <= trust_per_narrator <= 1.0):
        return 0.0
    return trust_per_narrator**chain_length


def poison_decay_probability(
    trust_per_narrator: float,
    chain_length: int,
) -> float:
    """P(poison accepted) = (1-f)^k — exponential decay.

    Corollary 4.3: The probability of accepting a poisoned claim
    decays exponentially with chain length.
    """
    if chain_length <= 0:
        return 1.0
    return (1.0 - trust_per_narrator) ** chain_length


class IsnadTrustModel:
    """IRP trust model with narrator registry.

    Maintains a registry of known narrators and their trust scores.
    When evaluating a chain, looks up each narrator's trust and
    applies the min-trust aggregation.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, float] = {}

    def register_narrator(self, narrator_id: str, trust: float) -> None:
        """Register or update a narrator's trust score."""
        self._registry[narrator_id] = max(0.0, min(trust, 1.0))

    def evaluate_chain(self, narrator_ids: List[str], claim: str = "") -> TrustResult:
        """Evaluate trust for a chain of narrator IDs."""
        narrators = []
        for nid in narrator_ids:
            trust = self._registry.get(nid, 0.0)  # unknown => zero trust
            narrators.append(Narrator(narrator_id=nid, trust=trust))

        return isnad_trust(IsnadChain(claim=claim, narrators=narrators))

    def narrator_count(self) -> int:
        return len(self._registry)
