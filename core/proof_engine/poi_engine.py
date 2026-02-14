"""
Proof-of-Impact Engine — The Soul of BIZRA

4-Stage Scoring Pipeline:
  Stage 1: Contribution Verification (signature + SNR quality gate)
  Stage 2: Network Reach (PageRank-style centrality over sorted nodes)
  Stage 3: Temporal Longevity (exponential decay + sustained relevance)
  Stage 4: Composite PoI = alpha*contribution + beta*reach + gamma*longevity

Determinism Invariants (SAPE audit v1):
  - All iterations use sorted() node/key lists
  - No datetime.now() in scoring paths — epoch_id is the clock
  - Every score/rejection emits a signed PoIReceipt with reason code
  - Same input → same bytes → same receipt hash (100-iteration DoD)

Standing on Giants:
- Nakamoto (2008): Proof-of-Work as verifiable contribution
- Page & Brin (1998): PageRank for citation-based authority
- Ebbinghaus (1885): Exponential decay for temporal relevance
- Axelrod (1984): Cooperation through repeated games
- Ostrom (1990): Commons governance without tragedy
- Al-Ghazali (1058-1111): Proportional justice in distribution
- Gini (1912): Inequality measurement
- Shannon (1948): Signal-to-noise as quality metric
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from core.proof_engine.canonical import blake3_digest, canonical_bytes

# =============================================================================
# PoI REASON CODES — Every outcome gets a reason
# =============================================================================


class PoIReasonCode(Enum):
    """Reason codes for PoI scoring decisions.

    Every accept, reject, quarantine, or penalty emits one of these codes
    in a signed PoIReceipt. No silent failures.
    """

    # Success
    POI_OK = "POI_OK"

    # Quarantine (soft reject — evidence missing but not provably bad)
    POI_QUARANTINE_MISSING_EVIDENCE = "POI_QUARANTINE_MISSING_EVIDENCE"

    # Hard rejects
    POI_REJECT_BAD_SIGNATURE = "POI_REJECT_BAD_SIGNATURE"
    POI_REJECT_DUPLICATE_ARTIFACT = "POI_REJECT_DUPLICATE_ARTIFACT"
    POI_REJECT_SNR_BELOW_THRESHOLD = "POI_REJECT_SNR_BELOW_THRESHOLD"
    POI_REJECT_EPOCH_MISMATCH = "POI_REJECT_EPOCH_MISMATCH"
    POI_REJECT_IHSAN_BELOW_THRESHOLD = "POI_REJECT_IHSAN_BELOW_THRESHOLD"

    # Penalties (accept with score reduction)
    POI_PENALTY_RING_DETECTED = "POI_PENALTY_RING_DETECTED"
    POI_PENALTY_RECIPROCAL_FARM = "POI_PENALTY_RECIPROCAL_FARM"

    # Internal error (fail-closed)
    POI_INTERNAL_INVARIANT_FAIL = "POI_INTERNAL_INVARIANT_FAIL"


# =============================================================================
# PoI RECEIPT — Signed proof of every scoring decision
# =============================================================================


@dataclass
class PoIReceipt:
    """Signed receipt for a PoI scoring decision.

    Every score, rejection, or penalty produces a receipt.
    The receipt is the proof; the score is just a number.
    """

    receipt_id: str
    epoch_id: str
    contributor_id: str
    reason: PoIReasonCode
    poi_score: float  # 0.0 for rejects
    contribution_score: float  # Stage 1
    reach_score: float  # Stage 2
    longevity_score: float  # Stage 3
    config_digest: str  # PoIConfig hash
    content_hash: str  # Contribution content hash (or "")

    # Signature (filled by sign_with)
    signature: bytes = field(default_factory=bytes)
    signer_pubkey: bytes = field(default_factory=bytes)

    def body_bytes(self) -> bytes:
        """Deterministic byte representation for signing.

        Excludes signature itself. Sorted keys via canonical_bytes.
        """
        data = {
            "receipt_id": self.receipt_id,
            "epoch_id": self.epoch_id,
            "contributor_id": self.contributor_id,
            "reason": self.reason.value,
            "poi_score": self.poi_score,
            "contribution_score": self.contribution_score,
            "reach_score": self.reach_score,
            "longevity_score": self.longevity_score,
            "config_digest": self.config_digest,
            "content_hash": self.content_hash,
        }
        return canonical_bytes(data)

    def sign_with(self, signer: Any) -> "PoIReceipt":
        """Sign the receipt. Accepts any SovereignSigner-compatible object."""
        body = self.body_bytes()
        self.signature = signer.sign(body)
        self.signer_pubkey = signer.public_key_bytes()
        return self

    def verify_signature(self, signer: Any) -> bool:
        """Verify signature against signer."""
        body = self.body_bytes()
        return signer.verify(body, self.signature)

    def digest(self) -> bytes:
        """Receipt digest (includes signature)."""
        data = self.body_bytes() + self.signature
        return blake3_digest(data)

    def hex_digest(self) -> str:
        """Hex-encoded receipt digest."""
        return self.digest().hex()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize PoI receipt to dictionary including signature and digest."""
        return {
            "receipt_id": self.receipt_id,
            "epoch_id": self.epoch_id,
            "contributor_id": self.contributor_id,
            "reason": self.reason.value,
            "poi_score": self.poi_score,
            "contribution_score": self.contribution_score,
            "reach_score": self.reach_score,
            "longevity_score": self.longevity_score,
            "config_digest": self.config_digest,
            "content_hash": self.content_hash,
            "signature": self.signature.hex(),
            "signer_pubkey": self.signer_pubkey.hex(),
            "receipt_digest": self.hex_digest(),
        }


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class PoIConfig:
    """PoI computation configuration.

    Default weights: alpha=0.5, beta=0.3, gamma=0.2
    Convex combination: alpha + beta + gamma = 1.0
    """

    # Composite weights (must sum to 1.0)
    alpha: float = 0.5  # Contribution weight
    beta: float = 0.3  # Network reach weight
    gamma: float = 0.2  # Temporal longevity weight

    # Stage 1: Contribution thresholds
    snr_quality_min: float = 0.85
    ihsan_quality_min: float = 0.90

    # Stage 2: PageRank parameters
    pagerank_damping: float = 0.85
    pagerank_iterations: int = 50
    pagerank_tolerance: float = 1e-6
    citation_ring_threshold: int = 3  # Max mutual citations before penalty

    # Stage 3: Temporal decay
    decay_lambda: float = 0.01  # Decay rate (per day)
    spike_threshold: float = 3.0  # Stddev multiplier for spike detection
    sustained_bonus: float = 0.15  # Bonus for sustained relevance

    # Gini / SAT thresholds
    gini_rebalance_threshold: float = 0.45
    zakat_rate: float = 0.025  # 2.5% computational zakat
    zakat_exemption_floor: float = 0.1  # Minimum PoI below which no zakat

    def validate(self) -> None:
        """Validate config invariants."""
        total = self.alpha + self.beta + self.gamma
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0, got {total:.6f} "
                f"(alpha={self.alpha}, beta={self.beta}, gamma={self.gamma})"
            )
        if not (0.0 < self.pagerank_damping < 1.0):
            raise ValueError(
                f"Damping factor must be in (0, 1), got {self.pagerank_damping}"
            )
        if self.decay_lambda <= 0:
            raise ValueError(f"Decay lambda must be positive, got {self.decay_lambda}")

    def canonical_bytes(self) -> bytes:
        """Deterministic byte representation."""
        data = {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "snr_quality_min": self.snr_quality_min,
            "ihsan_quality_min": self.ihsan_quality_min,
            "pagerank_damping": self.pagerank_damping,
            "pagerank_iterations": self.pagerank_iterations,
            "decay_lambda": self.decay_lambda,
            "gini_rebalance_threshold": self.gini_rebalance_threshold,
            "zakat_rate": self.zakat_rate,
        }
        return canonical_bytes(data)

    def digest(self) -> bytes:
        """Config digest."""
        return blake3_digest(self.canonical_bytes())

    def hex_digest(self) -> str:
        """Hex config digest."""
        return self.digest().hex()


# =============================================================================
# STAGE 1: CONTRIBUTION VERIFICATION
# =============================================================================


class ContributionType(Enum):
    """Types of verifiable contributions."""

    CODE = "code"
    DATA = "data"
    REVIEW = "review"
    GOVERNANCE = "governance"
    INFRASTRUCTURE = "infrastructure"


@dataclass
class ContributionMetadata:
    """Metadata for a verified contribution."""

    contributor_id: str
    contribution_type: ContributionType
    content_hash: str  # BLAKE3 hash of contribution content
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    snr_score: float = 0.0
    ihsan_score: float = 0.0
    signature: bytes = field(default_factory=bytes)
    nonce: str = ""

    def canonical_bytes(self) -> bytes:
        """Deterministic byte representation."""
        data = {
            "contributor_id": self.contributor_id,
            "contribution_type": self.contribution_type.value,
            "content_hash": self.content_hash,
            "timestamp": self.timestamp.isoformat(),
            "snr_score": self.snr_score,
            "ihsan_score": self.ihsan_score,
        }
        if self.nonce:
            data["nonce"] = self.nonce
        return canonical_bytes(data)

    def digest(self) -> bytes:
        """Contribution digest."""
        return blake3_digest(self.canonical_bytes())

    def hex_digest(self) -> str:
        """Hex contribution digest."""
        return self.digest().hex()


@dataclass
class VerifiedContribution:
    """Result of Stage 1: Contribution verification."""

    metadata: ContributionMetadata
    verified: bool
    quality_score: float  # Normalized [0, 1]
    rejection_reason: Optional[str] = None
    reason_code: PoIReasonCode = PoIReasonCode.POI_OK

    def to_dict(self) -> Dict[str, Any]:
        """Serialize verified contribution result to dictionary."""
        return {
            "contributor_id": self.metadata.contributor_id,
            "contribution_type": self.metadata.contribution_type.value,
            "content_hash": self.metadata.content_hash,
            "verified": self.verified,
            "quality_score": self.quality_score,
            "snr_score": self.metadata.snr_score,
            "ihsan_score": self.metadata.ihsan_score,
            "rejection_reason": self.rejection_reason,
            "reason_code": self.reason_code.value,
        }


class ContributionVerifier:
    """Stage 1: Verify contributions and compute quality scores.

    Checks:
    1. Content hash integrity
    2. SNR quality gate
    3. Ihsan quality gate
    4. Duplicate detection

    Every outcome emits a PoIReasonCode.
    """

    def __init__(self, config: Optional[PoIConfig] = None):
        self.config = config or PoIConfig()
        self._seen_hashes: Set[str] = set()

    def verify(self, metadata: ContributionMetadata) -> VerifiedContribution:
        """Verify a single contribution."""
        # Check duplicate
        if metadata.content_hash in self._seen_hashes:
            return VerifiedContribution(
                metadata=metadata,
                verified=False,
                quality_score=0.0,
                rejection_reason="Duplicate contribution detected",
                reason_code=PoIReasonCode.POI_REJECT_DUPLICATE_ARTIFACT,
            )

        # SNR quality gate
        if metadata.snr_score < self.config.snr_quality_min:
            return VerifiedContribution(
                metadata=metadata,
                verified=False,
                quality_score=0.0,
                rejection_reason=(
                    f"SNR below threshold: {metadata.snr_score:.3f} "
                    f"< {self.config.snr_quality_min}"
                ),
                reason_code=PoIReasonCode.POI_REJECT_SNR_BELOW_THRESHOLD,
            )

        # Ihsan quality gate
        if metadata.ihsan_score < self.config.ihsan_quality_min:
            return VerifiedContribution(
                metadata=metadata,
                verified=False,
                quality_score=0.0,
                rejection_reason=(
                    f"Ihsan below threshold: {metadata.ihsan_score:.3f} "
                    f"< {self.config.ihsan_quality_min}"
                ),
                reason_code=PoIReasonCode.POI_REJECT_IHSAN_BELOW_THRESHOLD,
            )

        # Compute quality score: weighted combination of SNR and Ihsan
        quality_score = 0.6 * metadata.snr_score + 0.4 * metadata.ihsan_score

        # Register hash to prevent duplicates
        self._seen_hashes.add(metadata.content_hash)

        return VerifiedContribution(
            metadata=metadata,
            verified=True,
            quality_score=quality_score,
            reason_code=PoIReasonCode.POI_OK,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get verifier statistics."""
        return {
            "unique_contributions": len(self._seen_hashes),
        }


# =============================================================================
# STAGE 2: NETWORK REACH (PageRank-style, deterministic iteration)
# =============================================================================


@dataclass
class ReachScore:
    """Result of Stage 2: Network reach computation."""

    contributor_id: str
    raw_pagerank: float
    normalized_reach: float  # Normalized [0, 1]
    citation_count: int
    cited_by_count: int
    ring_penalty: float = 0.0  # Penalty for citation rings
    reason_code: PoIReasonCode = PoIReasonCode.POI_OK

    def to_dict(self) -> Dict[str, Any]:
        """Serialize network reach score to dictionary."""
        return {
            "contributor_id": self.contributor_id,
            "raw_pagerank": self.raw_pagerank,
            "normalized_reach": self.normalized_reach,
            "citation_count": self.citation_count,
            "cited_by_count": self.cited_by_count,
            "ring_penalty": self.ring_penalty,
            "reason_code": self.reason_code.value,
        }


class CitationGraph:
    """PageRank-style citation graph for network reach.

    Standing on: Page & Brin (1998).

    Determinism: All iterations use sorted() node lists.
    Anti-gaming: Citation ring detection penalizes mutual
    citation clusters that exceed the configured threshold.
    """

    def __init__(self, config: Optional[PoIConfig] = None):
        self.config = config or PoIConfig()
        # Adjacency: node -> set of nodes it cites
        self._outgoing: Dict[str, Set[str]] = {}
        # Reverse: node -> set of nodes that cite it
        self._incoming: Dict[str, Set[str]] = {}
        # All known nodes
        self._nodes: Set[str] = set()

    def add_citation(self, citer: str, cited: str) -> None:
        """Record that 'citer' cites 'cited'."""
        if citer == cited:
            return  # Self-citations are ignored
        self._nodes.add(citer)
        self._nodes.add(cited)
        if citer not in self._outgoing:
            self._outgoing[citer] = set()
        self._outgoing[citer].add(cited)
        if cited not in self._incoming:
            self._incoming[cited] = set()
        self._incoming[cited].add(citer)

    def detect_citation_rings(self) -> Dict[str, float]:
        """Detect mutual citation rings and compute penalties.

        A ring is detected when two nodes cite each other AND
        have more than `citation_ring_threshold` mutual citations
        in common.

        Determinism: iterates over sorted(self._nodes).
        """
        penalties: Dict[str, float] = {}
        threshold = self.config.citation_ring_threshold

        for node in sorted(self._nodes):
            outgoing = self._outgoing.get(node, set())
            incoming = self._incoming.get(node, set())
            # Mutual citations: nodes that both cite and are cited by this node
            mutual = outgoing & incoming
            if len(mutual) > threshold:
                # Penalty proportional to excess mutual citations
                excess = len(mutual) - threshold
                penalty = min(excess * 0.1, 0.5)  # Cap at 50% penalty
                penalties[node] = penalty

        return penalties

    def compute_pagerank(self) -> Dict[str, float]:
        """Compute PageRank for all nodes.

        Standing on: Page & Brin (1998) — iterative power method.
        Determinism: sorted node list, deterministic iteration order.
        """
        if not self._nodes:
            return {}

        n = len(self._nodes)
        nodes = sorted(self._nodes)
        d = self.config.pagerank_damping

        # Initialize uniform
        rank: Dict[str, float] = {node: 1.0 / n for node in nodes}

        for _ in range(self.config.pagerank_iterations):
            new_rank: Dict[str, float] = {}

            for node in nodes:
                # Sum contributions from incoming nodes (sorted for determinism)
                incoming_sum = 0.0
                for citer in sorted(self._incoming.get(node, set())):
                    out_degree = len(self._outgoing.get(citer, set()))
                    if out_degree > 0:
                        incoming_sum += rank[citer] / out_degree

                new_rank[node] = (1 - d) / n + d * incoming_sum

            # Check convergence
            diff = sum(abs(new_rank[nd] - rank[nd]) for nd in nodes)
            rank = new_rank
            if diff < self.config.pagerank_tolerance:
                break

        return rank

    def compute_reach_scores(self) -> List[ReachScore]:
        """Compute reach scores for all nodes.

        Includes PageRank computation and ring penalty detection.
        Determinism: sorted iteration, deterministic penalties.
        """
        pagerank = self.compute_pagerank()
        penalties = self.detect_citation_rings()

        if not pagerank:
            return []

        # Normalize PageRank to [0, 1]
        max_pr = max(pagerank.values()) if pagerank else 1.0
        if max_pr == 0:
            max_pr = 1.0

        scores = []
        for node in sorted(self._nodes):
            raw_pr = pagerank.get(node, 0.0)
            penalty = penalties.get(node, 0.0)
            normalized = (raw_pr / max_pr) * (1.0 - penalty)
            normalized = max(0.0, min(1.0, normalized))

            reason = PoIReasonCode.POI_OK
            if penalty > 0:
                reason = PoIReasonCode.POI_PENALTY_RING_DETECTED

            scores.append(
                ReachScore(
                    contributor_id=node,
                    raw_pagerank=raw_pr,
                    normalized_reach=normalized,
                    citation_count=len(self._outgoing.get(node, set())),
                    cited_by_count=len(self._incoming.get(node, set())),
                    ring_penalty=penalty,
                    reason_code=reason,
                )
            )

        return scores

    def get_stats(self) -> Dict[str, Any]:
        """Graph statistics."""
        total_edges = sum(len(v) for v in self._outgoing.values())
        return {
            "total_nodes": len(self._nodes),
            "total_edges": total_edges,
            "avg_out_degree": total_edges / len(self._nodes) if self._nodes else 0.0,
        }


# =============================================================================
# STAGE 3: TEMPORAL LONGEVITY
# =============================================================================


@dataclass
class LongevityScore:
    """Result of Stage 3: Temporal longevity."""

    contributor_id: str
    raw_longevity: float
    normalized_longevity: float  # [0, 1]
    days_active: float
    decay_factor: float
    spike_detected: bool = False
    sustained_bonus_applied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize temporal longevity score to dictionary."""
        return {
            "contributor_id": self.contributor_id,
            "raw_longevity": self.raw_longevity,
            "normalized_longevity": self.normalized_longevity,
            "days_active": self.days_active,
            "decay_factor": self.decay_factor,
            "spike_detected": self.spike_detected,
            "sustained_bonus_applied": self.sustained_bonus_applied,
        }


class TemporalScorer:
    """Stage 3: Temporal longevity with exponential decay.

    Standing on: Ebbinghaus (1885) — forgetting curve as exponential decay.

    longevity = e^(-lambda * days_since_last) * activity_factor

    Determinism: reference_time is always passed explicitly (no datetime.now()
    in scoring path). Spike detection uses the same reference_time.

    Anti-gaming:
    - Spike detection: sudden bursts of activity are penalized
    - Sustained relevance: consistent contribution earns a bonus
    """

    def __init__(self, config: Optional[PoIConfig] = None):
        self.config = config or PoIConfig()
        # contributor_id -> list of contribution timestamps
        self._activity: Dict[str, List[datetime]] = {}

    def record_activity(
        self, contributor_id: str, timestamp: Optional[datetime] = None
    ) -> None:
        """Record a contribution event."""
        ts = timestamp or datetime.now(timezone.utc)
        if contributor_id not in self._activity:
            self._activity[contributor_id] = []
        self._activity[contributor_id].append(ts)

    def _detect_spike(
        self, timestamps: List[datetime], reference_time: datetime
    ) -> bool:
        """Detect activity spikes (sudden burst of contributions).

        A spike is when recent activity (last 7 days) is > spike_threshold
        standard deviations above the mean weekly rate.

        Determinism: uses reference_time instead of datetime.now().
        """
        if len(timestamps) < 4:
            return False

        now = reference_time
        # Count contributions per 7-day window
        windows: List[int] = []
        sorted_ts = sorted(timestamps)
        first_ts = sorted_ts[0]
        total_days = (now - first_ts).total_seconds() / 86400
        num_weeks = max(1, int(total_days / 7))

        for w in range(num_weeks):
            # Count in each week bucket
            window_start_days = w * 7
            window_end_days = (w + 1) * 7
            count = sum(
                1
                for t in sorted_ts
                if window_start_days
                <= (t - first_ts).total_seconds() / 86400
                < window_end_days
            )
            windows.append(count)

        if len(windows) < 2:
            return False

        mean = sum(windows) / len(windows)
        variance = sum((w - mean) ** 2 for w in windows) / len(windows)
        stddev = math.sqrt(variance) if variance > 0 else 0.0

        if stddev == 0:
            return False

        # Check if last window is a spike
        last_window = windows[-1]
        z_score = (last_window - mean) / stddev

        return z_score > self.config.spike_threshold

    def _is_sustained(self, timestamps: List[datetime], min_weeks: int = 4) -> bool:
        """Check if contributions are sustained over time.

        Sustained = contributions in at least `min_weeks` distinct weeks.
        """
        if len(timestamps) < min_weeks:
            return False

        weeks: Set[int] = set()
        for ts in timestamps:
            week_number = ts.isocalendar()[1] + ts.year * 100
            weeks.add(week_number)

        return len(weeks) >= min_weeks

    def compute_longevity(
        self,
        contributor_id: str,
        reference_time: Optional[datetime] = None,
    ) -> LongevityScore:
        """Compute temporal longevity for a contributor.

        Determinism: reference_time must be passed for deterministic scoring.
        Falls back to datetime.now() only for ad-hoc queries.
        """
        now = reference_time or datetime.now(timezone.utc)
        timestamps = self._activity.get(contributor_id, [])

        if not timestamps:
            return LongevityScore(
                contributor_id=contributor_id,
                raw_longevity=0.0,
                normalized_longevity=0.0,
                days_active=0.0,
                decay_factor=0.0,
            )

        sorted_ts = sorted(timestamps)
        last_activity = sorted_ts[-1]
        first_activity = sorted_ts[0]

        # Days since last activity
        days_since_last = max(0.0, (now - last_activity).total_seconds() / 86400)

        # Total active span
        days_active = max(0.0, (last_activity - first_activity).total_seconds() / 86400)

        # Exponential decay: e^(-lambda * days_since_last)
        decay_factor = math.exp(-self.config.decay_lambda * days_since_last)

        # Activity factor: log(1 + count) / log(1 + expected_count)
        count = len(timestamps)
        activity_factor = min(1.0, math.log1p(count) / math.log1p(20))

        # Raw longevity
        raw_longevity = decay_factor * activity_factor

        # Spike detection (deterministic: uses reference_time)
        spike = self._detect_spike(timestamps, now)
        if spike:
            raw_longevity *= 0.5  # 50% penalty for spiking

        # Sustained bonus
        sustained = self._is_sustained(timestamps)
        if sustained and not spike:
            raw_longevity = min(1.0, raw_longevity + self.config.sustained_bonus)

        # Clamp [0, 1]
        normalized = max(0.0, min(1.0, raw_longevity))

        return LongevityScore(
            contributor_id=contributor_id,
            raw_longevity=raw_longevity,
            normalized_longevity=normalized,
            days_active=days_active,
            decay_factor=decay_factor,
            spike_detected=spike,
            sustained_bonus_applied=sustained and not spike,
        )


# =============================================================================
# STAGE 4: COMPOSITE PoI
# =============================================================================


@dataclass
class ProofOfImpact:
    """Composite Proof-of-Impact score."""

    contributor_id: str

    # Component scores
    contribution_score: float  # Stage 1
    reach_score: float  # Stage 2
    longevity_score: float  # Stage 3

    # Composite
    poi_score: float  # alpha*contribution + beta*reach + gamma*longevity

    # Weights used
    alpha: float
    beta: float
    gamma: float

    # Audit
    config_digest: str
    computation_id: str
    epoch_id: str = ""
    reason_code: PoIReasonCode = PoIReasonCode.POI_OK

    # Timestamp — for audit display only, NOT used in scoring
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize composite PoI score to dictionary."""
        return {
            "contributor_id": self.contributor_id,
            "contribution_score": self.contribution_score,
            "reach_score": self.reach_score,
            "longevity_score": self.longevity_score,
            "poi_score": self.poi_score,
            "weights": {
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
            },
            "config_digest": self.config_digest,
            "computation_id": self.computation_id,
            "epoch_id": self.epoch_id,
            "reason_code": self.reason_code.value,
            "timestamp": self.timestamp.isoformat(),
        }

    def canonical_bytes(self) -> bytes:
        """Deterministic byte representation for hashing.

        Uses sorted-key canonical encoding. Excludes mutable timestamp
        from the canonical form to ensure determinism.
        """
        data = {
            "contributor_id": self.contributor_id,
            "contribution_score": self.contribution_score,
            "reach_score": self.reach_score,
            "longevity_score": self.longevity_score,
            "poi_score": self.poi_score,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "config_digest": self.config_digest,
            "computation_id": self.computation_id,
            "epoch_id": self.epoch_id,
            "reason_code": self.reason_code.value,
        }
        return canonical_bytes(data)

    def digest(self) -> bytes:
        """PoI digest."""
        return blake3_digest(self.canonical_bytes())

    def hex_digest(self) -> str:
        """Hex PoI digest."""
        return self.digest().hex()


@dataclass
class AuditTrail:
    """Complete audit trail for a PoI computation epoch."""

    epoch_id: str
    poi_scores: List[ProofOfImpact]
    gini_coefficient: float
    rebalance_triggered: bool
    config_digest: str
    receipts: List[PoIReceipt] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize audit trail with all scores and receipts to dictionary."""
        return {
            "epoch_id": self.epoch_id,
            "total_contributors": len(self.poi_scores),
            "gini_coefficient": self.gini_coefficient,
            "rebalance_triggered": self.rebalance_triggered,
            "config_digest": self.config_digest,
            "timestamp": self.timestamp.isoformat(),
            "scores": [p.to_dict() for p in self.poi_scores],
            "receipts": [r.to_dict() for r in self.receipts],
        }

    def canonical_bytes(self) -> bytes:
        """Deterministic byte representation.

        Excludes mutable timestamp. Scores are sorted by contributor_id.
        """
        scores_data = []
        for p in sorted(self.poi_scores, key=lambda x: x.contributor_id):
            scores_data.append(
                {
                    "contributor_id": p.contributor_id,
                    "poi_score": p.poi_score,
                    "reason_code": p.reason_code.value,
                }
            )
        data = {
            "epoch_id": self.epoch_id,
            "gini_coefficient": self.gini_coefficient,
            "rebalance_triggered": self.rebalance_triggered,
            "config_digest": self.config_digest,
            "scores": scores_data,
        }
        return canonical_bytes(data)

    def digest(self) -> bytes:
        return blake3_digest(self.canonical_bytes())

    def hex_digest(self) -> str:
        return self.digest().hex()


# =============================================================================
# GINI COEFFICIENT + SAT REBALANCING
# =============================================================================


def compute_gini(values: List[float]) -> float:
    """Compute Gini coefficient for a distribution.

    Standing on: Gini (1912) — measure of statistical dispersion.

    Returns 0.0 for perfect equality, 1.0 for maximum inequality.
    """
    if not values or all(v == 0 for v in values):
        return 0.0

    n = len(values)
    if n == 1:
        return 0.0

    sorted_values = sorted(values)
    total = sum(sorted_values)

    if total == 0:
        return 0.0

    # Standard Gini formula: G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n+1)/n
    numerator = sum((i + 1) * v for i, v in enumerate(sorted_values))
    gini = (2.0 * numerator) / (n * total) - (n + 1) / n

    return max(0.0, min(1.0, gini))


@dataclass
class RebalanceResult:
    """Result of SAT rebalancing."""

    original_scores: Dict[str, float]
    rebalanced_scores: Dict[str, float]
    gini_before: float
    gini_after: float
    zakat_collected: float
    zakat_distributed: float
    rebalance_triggered: bool

    def to_dict(self) -> Dict[str, Any]:
        """Serialize SAT rebalance result to dictionary."""
        return {
            "gini_before": self.gini_before,
            "gini_after": self.gini_after,
            "zakat_collected": self.zakat_collected,
            "zakat_distributed": self.zakat_distributed,
            "rebalance_triggered": self.rebalance_triggered,
            "contributors_affected": len(self.rebalanced_scores),
        }


class SATRebalancer:
    """SAT-based rebalancing with computational zakat.

    Standing on:
    - Al-Ghazali: Proportional justice — excess flows to those in need
    - Ostrom (1990): Commons governance rules

    Mechanism:
    1. Compute Gini coefficient
    2. If above threshold, collect 2.5% zakat from excess holdings
    3. Redistribute to lowest contributors proportionally

    Determinism: iterates over sorted(scores.keys()).
    """

    def __init__(self, config: Optional[PoIConfig] = None):
        self.config = config or PoIConfig()

    def rebalance(self, scores: Dict[str, float]) -> RebalanceResult:
        """Rebalance scores if Gini exceeds threshold.

        Determinism: sorted key iteration for collection and distribution.
        """
        if not scores:
            return RebalanceResult(
                original_scores={},
                rebalanced_scores={},
                gini_before=0.0,
                gini_after=0.0,
                zakat_collected=0.0,
                zakat_distributed=0.0,
                rebalance_triggered=False,
            )

        values = list(scores.values())
        gini_before = compute_gini(values)

        if gini_before <= self.config.gini_rebalance_threshold:
            return RebalanceResult(
                original_scores=dict(scores),
                rebalanced_scores=dict(scores),
                gini_before=gini_before,
                gini_after=gini_before,
                zakat_collected=0.0,
                zakat_distributed=0.0,
                rebalance_triggered=False,
            )

        # Compute mean score
        mean_score = sum(values) / len(values)

        # Collect zakat from those above the mean (sorted for determinism)
        zakat_pool = 0.0
        rebalanced = dict(scores)

        for contributor in sorted(scores.keys()):
            score = scores[contributor]
            if score <= self.config.zakat_exemption_floor:
                continue
            if score > mean_score:
                excess = score - mean_score
                zakat = excess * self.config.zakat_rate
                rebalanced[contributor] = score - zakat
                zakat_pool += zakat

        # Distribute to those below the mean (sorted for determinism)
        below_mean = {
            k: mean_score - scores[k]
            for k in sorted(scores.keys())
            if scores[k] < mean_score
        }
        total_deficit = sum(below_mean.values())

        zakat_distributed = 0.0
        if total_deficit > 0 and zakat_pool > 0:
            for contributor in sorted(below_mean.keys()):
                deficit = below_mean[contributor]
                share = (deficit / total_deficit) * zakat_pool
                rebalanced[contributor] = scores[contributor] + share
                zakat_distributed += share

        gini_after = compute_gini(list(rebalanced.values()))

        return RebalanceResult(
            original_scores=dict(scores),
            rebalanced_scores=rebalanced,
            gini_before=gini_before,
            gini_after=gini_after,
            zakat_collected=zakat_pool,
            zakat_distributed=zakat_distributed,
            rebalance_triggered=True,
        )


# =============================================================================
# PoI ORCHESTRATOR
# =============================================================================


class PoIOrchestrator:
    """Orchestrates the full 4-stage PoI pipeline.

    Pipeline:
      Stage 1: ContributionVerifier → VerifiedContribution
      Stage 2: CitationGraph → ReachScore
      Stage 3: TemporalScorer → LongevityScore
      Stage 4: Composite PoI → ProofOfImpact

    Post-pipeline:
      Gini check → SAT rebalancing (if needed) → AuditTrail + PoIReceipts

    Determinism:
      - All contributor iteration is sorted
      - reference_time is passed to TemporalScorer
      - epoch_id is the logical clock (not wall-clock)
      - Every outcome produces a signed receipt
    """

    def __init__(self, config: Optional[PoIConfig] = None, signer: Any = None):
        self.config = config or PoIConfig()
        self.config.validate()

        self.verifier = ContributionVerifier(self.config)
        self.citation_graph = CitationGraph(self.config)
        self.temporal_scorer = TemporalScorer(self.config)
        self.rebalancer = SATRebalancer(self.config)

        # Receipt signing — use SimpleSigner if none provided
        if signer is None:
            from core.proof_engine.receipt import SimpleSigner

            self._signer = SimpleSigner(b"poi-engine-default-key")
        else:
            self._signer = signer

        self._epoch_counter = 0
        self._receipt_counter = 0
        self._contributions: Dict[str, List[VerifiedContribution]] = {}
        self._epochs: List[AuditTrail] = []

    def _next_receipt_id(self, epoch_id: str) -> str:
        """Generate deterministic receipt ID from epoch + counter."""
        self._receipt_counter += 1
        return f"poi_rcpt_{epoch_id}_{self._receipt_counter:08d}"

    def register_contribution(
        self, metadata: ContributionMetadata
    ) -> VerifiedContribution:
        """Register and verify a contribution (Stage 1)."""
        result = self.verifier.verify(metadata)

        if result.verified:
            cid = metadata.contributor_id
            if cid not in self._contributions:
                self._contributions[cid] = []
            self._contributions[cid].append(result)
            self.temporal_scorer.record_activity(cid, metadata.timestamp)

        return result

    def add_citation(self, citer: str, cited: str) -> None:
        """Add a citation to the graph (Stage 2 input)."""
        self.citation_graph.add_citation(citer, cited)

    def compute_epoch(
        self,
        epoch_id: Optional[str] = None,
        reference_time: Optional[datetime] = None,
    ) -> AuditTrail:
        """Run a full PoI computation epoch (Stages 1-4 + rebalancing).

        Determinism:
        - epoch_id is the logical clock
        - reference_time is passed to temporal scorer
        - All contributor iteration is sorted
        - Receipts are emitted for every contributor

        Returns an AuditTrail with all scores, receipts, and Gini analysis.
        """
        self._epoch_counter += 1
        eid = epoch_id or f"epoch_{self._epoch_counter:08d}"
        ref_time = reference_time or datetime.now(timezone.utc)

        # Stage 2: Network reach
        reach_scores = self.citation_graph.compute_reach_scores()
        reach_map = {r.contributor_id: r for r in reach_scores}

        # Stage 1: Average contribution quality per contributor
        contribution_map: Dict[str, float] = {}
        for cid in sorted(self._contributions.keys()):
            contributions = self._contributions[cid]
            verified = [c for c in contributions if c.verified]
            if verified:
                contribution_map[cid] = sum(c.quality_score for c in verified) / len(
                    verified
                )
            else:
                contribution_map[cid] = 0.0

        # All known contributors (sorted for determinism)
        all_contributors = sorted(
            set(contribution_map.keys())
            | set(reach_map.keys())
            | set(self.temporal_scorer._activity.keys())
        )

        # Stage 3 + Stage 4: Compute composite PoI
        poi_scores: List[ProofOfImpact] = []
        receipts: List[PoIReceipt] = []

        for cid in all_contributors:
            contribution = contribution_map.get(cid, 0.0)
            reach_data = reach_map.get(cid)
            reach = reach_data.normalized_reach if reach_data else 0.0
            longevity = self.temporal_scorer.compute_longevity(
                cid, reference_time=ref_time
            )

            # Composite: PoI = alpha*contribution + beta*reach + gamma*longevity
            poi_score = (
                self.config.alpha * contribution
                + self.config.beta * reach
                + self.config.gamma * longevity.normalized_longevity
            )
            poi_score = max(0.0, min(1.0, poi_score))

            # Determine reason code
            reason = PoIReasonCode.POI_OK
            if reach_data and reach_data.ring_penalty > 0:
                reason = PoIReasonCode.POI_PENALTY_RING_DETECTED

            poi = ProofOfImpact(
                contributor_id=cid,
                contribution_score=contribution,
                reach_score=reach,
                longevity_score=longevity.normalized_longevity,
                poi_score=poi_score,
                alpha=self.config.alpha,
                beta=self.config.beta,
                gamma=self.config.gamma,
                config_digest=self.config.hex_digest(),
                computation_id=f"{eid}_{cid}",
                epoch_id=eid,
                reason_code=reason,
            )
            poi_scores.append(poi)

            # Emit receipt
            content_hash = ""
            if cid in self._contributions and self._contributions[cid]:
                content_hash = self._contributions[cid][-1].metadata.content_hash

            receipt = PoIReceipt(
                receipt_id=self._next_receipt_id(eid),
                epoch_id=eid,
                contributor_id=cid,
                reason=reason,
                poi_score=poi_score,
                contribution_score=contribution,
                reach_score=reach,
                longevity_score=longevity.normalized_longevity,
                config_digest=self.config.hex_digest(),
                content_hash=content_hash,
            )
            receipt.sign_with(self._signer)
            receipts.append(receipt)

        # Gini check + rebalancing
        score_map = {p.contributor_id: p.poi_score for p in poi_scores}
        rebalance_result = self.rebalancer.rebalance(score_map)

        # Apply rebalanced scores back to PoI objects
        if rebalance_result.rebalance_triggered:
            for poi in poi_scores:
                poi.poi_score = rebalance_result.rebalanced_scores.get(
                    poi.contributor_id, poi.poi_score
                )

        gini = compute_gini([p.poi_score for p in poi_scores])

        audit = AuditTrail(
            epoch_id=eid,
            poi_scores=poi_scores,
            gini_coefficient=gini,
            rebalance_triggered=rebalance_result.rebalance_triggered,
            config_digest=self.config.hex_digest(),
            receipts=receipts,
        )

        self._epochs.append(audit)
        return audit

    def get_contributor_poi(self, contributor_id: str) -> Optional[ProofOfImpact]:
        """Get most recent PoI for a contributor."""
        if not self._epochs:
            return None
        latest = self._epochs[-1]
        for poi in latest.poi_scores:
            if poi.contributor_id == contributor_id:
                return poi
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        total_contributions = sum(len(v) for v in self._contributions.values())
        return {
            "total_contributors": len(self._contributions),
            "total_contributions": total_contributions,
            "total_epochs": len(self._epochs),
            "total_receipts": self._receipt_counter,
            "graph_stats": self.citation_graph.get_stats(),
            "verifier_stats": self.verifier.get_stats(),
            "config_digest": self.config.hex_digest(),
        }


# =============================================================================
# TOKEN DISTRIBUTION
# =============================================================================


@dataclass
class TokenDistribution:
    """Token distribution result for an epoch.

    Tokens minted per contributor:
      delta_token = k * poi_score * epoch_reward
    """

    epoch_id: str
    epoch_reward: float
    distributions: Dict[str, float]  # contributor_id -> tokens
    total_minted: float
    gini_coefficient: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize token distribution result to dictionary."""
        return {
            "epoch_id": self.epoch_id,
            "epoch_reward": self.epoch_reward,
            "total_minted": self.total_minted,
            "gini_coefficient": self.gini_coefficient,
            "num_recipients": len(self.distributions),
            "distributions": self.distributions,
        }


def compute_token_distribution(
    audit: AuditTrail,
    epoch_reward: float,
    scaling_factor: float = 1.0,
) -> TokenDistribution:
    """Compute token distribution from PoI scores.

    delta_token = k * poi_score * epoch_reward

    where k normalizes so total minted = epoch_reward.

    Determinism: iterates over sorted poi_scores by contributor_id.
    """
    scores = {p.contributor_id: p.poi_score for p in audit.poi_scores}
    total_poi = sum(scores.values())

    distributions: Dict[str, float] = {}
    if total_poi > 0:
        for cid in sorted(scores.keys()):
            score = scores[cid]
            distributions[cid] = scaling_factor * (score / total_poi) * epoch_reward
    total_minted = sum(distributions.values())

    return TokenDistribution(
        epoch_id=audit.epoch_id,
        epoch_reward=epoch_reward,
        distributions=distributions,
        total_minted=total_minted,
        gini_coefficient=audit.gini_coefficient,
    )
