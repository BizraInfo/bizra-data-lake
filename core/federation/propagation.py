"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA PATTERN FEDERATION — PATTERN PROPAGATION                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Manages the lifecycle of SAPE-elevated patterns across the network.        ║
║                                                                              ║
║   Flow:                                                                      ║
║   1. Local SAPE elevates a pattern (>3 repetitions, high SNR)                ║
║   2. Pattern wrapped in PCI Envelope                                         ║
║   3. Broadcast via Gossip to peers                                           ║
║   4. Peers validate (Ihsān gate) and adopt                                   ║
║   5. Impact tracked → feeds back to ImpactScore                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field

logger = logging.getLogger("PROPAGATION")
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Dict, List

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD, UNIFIED_SNR_THRESHOLD

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════


ELEVATION_THRESHOLD = 3  # Min repetitions to elevate
MIN_SNR_FOR_ELEVATION = (
    UNIFIED_SNR_THRESHOLD  # Minimum SNR to consider for elevation (SEC-020 aligned)
)
MIN_SNR_DELTA_FOR_ELEVATION = 0.10  # Minimum SNR improvement required for elevation
MIN_IHSAN_FOR_PROPAGATION = UNIFIED_IHSAN_THRESHOLD  # Ihsān floor for network sharing
PATTERN_TTL_HOURS = 168  # 7 days
MAX_PATTERNS_CACHE = 1000  # Hard limit for pattern cache size

# ═══════════════════════════════════════════════════════════════════════════════
# TYPES
# ═══════════════════════════════════════════════════════════════════════════════


class PatternStatus(str, Enum):
    LOCAL = "LOCAL"  # Only on this node
    PROPOSED = "PROPOSED"  # Submitted to network
    VALIDATED = "VALIDATED"  # Accepted by quorum
    REJECTED = "REJECTED"  # Failed validation
    DEPRECATED = "DEPRECATED"  # Superseded by better pattern


@dataclass
class PatternMetrics:
    """Tracks usage and impact of a pattern."""

    uses: int = 0
    successes: int = 0
    total_snr_delta: float = 0.0
    total_latency_saved_ms: float = 0.0
    first_used: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        return self.successes / max(1, self.uses)

    @property
    def average_snr_boost(self) -> float:
        return self.total_snr_delta / max(1, self.uses)

    def record_use(self, success: bool, snr_delta: float, latency_saved_ms: float = 0):
        self.uses += 1
        if success:
            self.successes += 1
        self.total_snr_delta += snr_delta
        self.total_latency_saved_ms += latency_saved_ms
        self.last_used = time.time()


@dataclass
class ElevatedPattern:
    """A pattern that has been elevated by SAPE."""

    pattern_id: str
    source_node_id: str

    # Pattern definition
    trigger_condition: str  # When to apply (e.g., "snr < 0.8")
    action: str  # What to do (e.g., "apply_refinement")
    domain: str  # Domain (e.g., "snr_optimization")

    # Provenance
    creation_time: str
    elevation_count: int  # How many times triggered before elevation

    # Metrics
    metrics: PatternMetrics = field(default_factory=PatternMetrics)

    # Status
    status: PatternStatus = PatternStatus.LOCAL
    ihsan_score: float = 0.95

    # Validation
    validator_signatures: List[str] = field(default_factory=list)

    def compute_hash(self) -> str:
        """Deterministic hash of pattern content."""
        content = f"{self.trigger_condition}:{self.action}:{self.domain}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def compute_impact_score(self) -> float:
        """
        Calculate the pattern's impact score for consensus.
        Score = success_rate × ihsan × (1 + avg_snr_boost) × age_factor
        """
        age_hours = (time.time() - self.metrics.first_used) / 3600
        age_factor = min(1.0, age_hours / 24)  # Full weight after 24h

        score = (
            self.metrics.success_rate
            * self.ihsan_score
            * (1.0 + self.metrics.average_snr_boost)
            * (0.5 + 0.5 * age_factor)  # 50% base + 50% from age
        )
        return min(1.0, score)

    def to_dict(self) -> Dict:
        return {
            "pattern_id": self.pattern_id,
            "source_node_id": self.source_node_id,
            "trigger_condition": self.trigger_condition,
            "action": self.action,
            "domain": self.domain,
            "creation_time": self.creation_time,
            "elevation_count": self.elevation_count,
            "impact_score": self.compute_impact_score(),
            "ihsan_score": self.ihsan_score,
            "status": self.status.value,
            "metrics": {
                "uses": self.metrics.uses,
                "success_rate": self.metrics.success_rate,
                "avg_snr_boost": self.metrics.average_snr_boost,
            },
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ElevatedPattern":
        metrics = PatternMetrics(
            uses=d.get("metrics", {}).get("uses", 0),
            successes=int(
                d.get("metrics", {}).get("success_rate", 0)
                * d.get("metrics", {}).get("uses", 1)
            ),
        )
        return cls(
            pattern_id=d["pattern_id"],
            source_node_id=d["source_node_id"],
            trigger_condition=d["trigger_condition"],
            action=d["action"],
            domain=d["domain"],
            creation_time=d["creation_time"],
            elevation_count=d.get("elevation_count", ELEVATION_THRESHOLD),
            metrics=metrics,
            status=PatternStatus(d.get("status", "LOCAL")),
            ihsan_score=d.get("ihsan_score", 0.95),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN STORE
# ═══════════════════════════════════════════════════════════════════════════════


class PatternStore:
    """
    Local storage for elevated patterns.
    Manages both locally-created and network-received patterns.
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.local_patterns: Dict[str, ElevatedPattern] = {}
        self.network_patterns: Dict[str, ElevatedPattern] = {}
        self.pending_candidates: Dict[str, int] = {}  # trigger -> count

    def record_pattern_use(self, trigger: str, success: bool, snr_delta: float):
        """
        Record a pattern trigger. If count exceeds threshold, elevate.
        """
        # Check if already elevated
        pattern_id = f"sape_{hashlib.sha256(trigger.encode()).hexdigest()[:16]}"
        if pattern_id in self.local_patterns:
            self.local_patterns[pattern_id].metrics.record_use(success, snr_delta)
            return

        self.pending_candidates[trigger] = self.pending_candidates.get(trigger, 0) + 1

        if self.pending_candidates[trigger] >= ELEVATION_THRESHOLD:
            # SEC-020: Enforce minimum SNR improvement for elevation
            if snr_delta >= MIN_SNR_DELTA_FOR_ELEVATION:
                self._elevate_pattern(trigger, snr_delta)

    def _elevate_pattern(self, trigger: str, snr_delta: float):
        """Promote a candidate to an elevated pattern."""
        pattern_id = f"sape_{hashlib.sha256(trigger.encode()).hexdigest()[:16]}"

        if pattern_id in self.local_patterns:
            # Already elevated, just update metrics
            self.local_patterns[pattern_id].metrics.record_use(True, snr_delta)
            return

        pattern = ElevatedPattern(
            pattern_id=pattern_id,
            source_node_id=self.node_id,
            trigger_condition=trigger,
            action="auto_optimize",  # Default action
            domain="general",
            creation_time=datetime.now(timezone.utc).isoformat(),
            elevation_count=self.pending_candidates[trigger],
            ihsan_score=0.95,
        )
        pattern.metrics.record_use(True, snr_delta)

        self.local_patterns[pattern_id] = pattern
        del self.pending_candidates[trigger]

        # Enforce cache limit
        self._prune_cache_if_needed(self.local_patterns)

        print(f"📈 Pattern elevated: {pattern_id}")

    def add_network_pattern(self, pattern: ElevatedPattern) -> bool:
        """
        Add a pattern received from the network.
        Returns True if accepted.
        """
        # Validate Ihsān
        if pattern.ihsan_score < MIN_IHSAN_FOR_PROPAGATION:
            print(
                f"❌ Rejected pattern {pattern.pattern_id}: Ihsān {pattern.ihsan_score} < {MIN_IHSAN_FOR_PROPAGATION}"
            )
            return False

        # Check for duplicates
        if pattern.pattern_id in self.network_patterns:
            existing = self.network_patterns[pattern.pattern_id]
            # Accept if higher impact score
            if pattern.compute_impact_score() <= existing.compute_impact_score():
                return False

        pattern.status = PatternStatus.VALIDATED
        self.network_patterns[pattern.pattern_id] = pattern

        # Enforce cache limit
        self._prune_cache_if_needed(self.network_patterns)

        print(f"✅ Accepted network pattern: {pattern.pattern_id}")
        return True

    def get_applicable_patterns(self, context: Dict) -> List[ElevatedPattern]:
        """Find patterns that match the current context."""
        # Simplified: return all validated patterns
        # In production, would match trigger_condition against context
        all_patterns = list(self.local_patterns.values()) + list(
            self.network_patterns.values()
        )
        return [
            p
            for p in all_patterns
            if p.status in (PatternStatus.LOCAL, PatternStatus.VALIDATED)
        ]

    def get_patterns_for_sharing(self) -> List[ElevatedPattern]:
        """Get local patterns ready to share with the network."""
        shareable = []
        for p in self.local_patterns.values():
            if (
                p.status == PatternStatus.LOCAL
                and p.ihsan_score >= MIN_IHSAN_FOR_PROPAGATION
            ):
                if (
                    p.metrics.uses >= ELEVATION_THRESHOLD
                    and p.metrics.success_rate >= 0.8
                ):
                    shareable.append(p)
        return shareable

    def _prune_cache_if_needed(self, cache: Dict[str, ElevatedPattern]):
        """
        Enforce MAX_PATTERNS_CACHE by evicting lowest-impact patterns.

        SECURITY: Prevents unbounded memory growth from pattern accumulation.
        """
        if len(cache) <= MAX_PATTERNS_CACHE:
            return

        # Sort by impact score (lowest first) and evict excess
        sorted_patterns = sorted(
            cache.items(), key=lambda x: x[1].compute_impact_score()
        )
        excess = len(cache) - MAX_PATTERNS_CACHE
        for pattern_id, _ in sorted_patterns[:excess]:
            del cache[pattern_id]
            print(f"🗑️ Evicted low-impact pattern: {pattern_id}")

    def get_stats(self) -> Dict:
        return {
            "local_patterns": len(self.local_patterns),
            "network_patterns": len(self.network_patterns),
            "pending_candidates": len(self.pending_candidates),
            "total_pattern_uses": sum(
                p.metrics.uses for p in self.local_patterns.values()
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PROPAGATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════


class PropagationEngine:
    """
    Manages the broadcast and reception of patterns across the network.

    Security (Standing on Giants — Lamport BFT):
    - All patterns are wrapped in PCI envelopes with Ed25519 signatures
    - Patterns without valid signatures are rejected
    - Ihsān threshold enforced before propagation
    """

    def __init__(
        self,
        store: PatternStore,
        broadcast_fn: Callable[[bytes], None] = None,
        node_id: str = "",
        private_key: str = "",
        public_key: str = "",
    ):
        self.store = store
        self.broadcast = broadcast_fn or (lambda x: None)
        self._propagation_queue: List[ElevatedPattern] = []

        # PCI signing credentials
        self._node_id = node_id
        self._private_key = private_key
        self._public_key = public_key

        # PCI verification (lazy import)
        self._pci_gatekeeper = None

    def _get_pci_gatekeeper(self):
        """
        Lazy load PCI gatekeeper for pattern verification.

        Note: Policy enforcement is disabled for pattern propagation.
        Patterns are validated by signature + Ihsān + SNR thresholds.
        The constitution policy applies to inference requests, not pattern sharing.
        """
        if self._pci_gatekeeper is None:
            try:
                from core.pci import PCIGateKeeper

                # Disable policy enforcement for pattern propagation
                # Patterns are self-validating via Ed25519 signature
                self._pci_gatekeeper = PCIGateKeeper(policy_enforcement=False)
            except ImportError:
                logger.warning("PCI module not available, signatures disabled")
        return self._pci_gatekeeper

    def queue_for_propagation(self, pattern: ElevatedPattern):
        """Add a pattern to the propagation queue."""
        if pattern.ihsan_score < MIN_IHSAN_FOR_PROPAGATION:
            logger.warning(
                f"Pattern {pattern.pattern_id} below Ihsān threshold ({pattern.ihsan_score} < {MIN_IHSAN_FOR_PROPAGATION}), not propagating"
            )
            return
        self._propagation_queue.append(pattern)

    def propagate_pending(self) -> int:
        """
        Send all queued patterns to the network with PCI envelopes.

        Security: Each pattern is wrapped in a signed PCI envelope before broadcast.
        """
        count = 0
        while self._propagation_queue:
            pattern = self._propagation_queue.pop(0)
            pattern.status = PatternStatus.PROPOSED

            # Create PCI-enveloped propagation message
            envelope_data = self._create_pci_envelope(pattern)

            msg = json.dumps(
                {
                    "type": "PATTERN_PROPAGATE",
                    "envelope": envelope_data,
                }
            ).encode("utf-8")

            self.broadcast(msg)
            count += 1
            logger.info(f"📡 Propagated pattern: {pattern.pattern_id} (PCI-signed)")

        return count

    def _create_pci_envelope(self, pattern: ElevatedPattern) -> Dict:
        """
        Wrap pattern in PCI envelope with Ed25519 signature.

        Standing on Giants — Shannon + Lamport:
        - Content integrity via cryptographic signature
        - Ihsān/SNR scores embedded in metadata

        SNR Calculation for Patterns:
        - Patterns with positive SNR delta (improvement) get a high SNR score
        - Base SNR score is MIN_SNR_FOR_ELEVATION (0.85) + scaled delta
        - Capped at 1.0
        """
        try:
            from core.pci import EnvelopeBuilder

            if self._private_key and self._public_key:
                # Calculate pattern SNR: base + scaled improvement
                # A pattern with 0.15 avg boost gets 0.85 + 0.10 = 0.95
                snr_boost = pattern.metrics.average_snr_boost
                pattern_snr = min(1.0, MIN_SNR_FOR_ELEVATION + (snr_boost * 0.67))

                # Use pattern hash for policy (patterns are self-validating via signature)
                pattern_hash = pattern.compute_hash()

                builder = EnvelopeBuilder()
                envelope = (
                    builder.with_sender("PAT", self._node_id, self._public_key)
                    .with_payload(
                        "pattern/propagate",
                        pattern.to_dict(),
                        pattern_hash,  # Pattern's own hash as policy
                        "federation",
                    )
                    .with_metadata(pattern.ihsan_score, pattern_snr)
                    .build()
                    .sign(self._private_key)
                )
                return envelope.to_dict()
            else:
                # Fallback: unsigned envelope (log warning)
                logger.warning(
                    f"No signing keys available for pattern {pattern.pattern_id}"
                )
                return {"pattern": pattern.to_dict(), "signed": False}

        except ImportError:
            logger.warning("PCI EnvelopeBuilder not available")
            return {"pattern": pattern.to_dict(), "signed": False}

    def _verify_pci_envelope(self, envelope_data: Dict) -> bool:
        """
        Verify PCI envelope signature using Ed25519.

        Standing on Giants — Lamport BFT:
        - Cryptographic verification prevents message tampering
        - Rejects patterns without valid signatures
        - Enforces Ihsān threshold on incoming patterns

        Security (SEC-016 compliance):
        - All federation messages MUST be signed
        - Unsigned or invalid signatures result in rejection

        Returns:
            True if envelope signature is valid and thresholds met, False otherwise
        """
        try:
            # Check if this is explicitly marked as unsigned
            if envelope_data.get("signed") is False:
                logger.warning("Rejecting unsigned envelope (strict mode)")
                return False

            # Extract signature info
            sig_data = envelope_data.get("signature")
            if isinstance(sig_data, dict):
                signature = sig_data.get("value", "")
            else:
                signature = sig_data

            sender_info = envelope_data.get("sender", {})
            public_key = sender_info.get("public_key", "")

            if not signature or not public_key:
                logger.error(
                    "Missing signature or public key in envelope",
                    extra={
                        "has_signature": bool(signature),
                        "has_public_key": bool(public_key),
                    },
                )
                return False

            # Use PCI gatekeeper for full verification
            gatekeeper = self._get_pci_gatekeeper()
            if gatekeeper:
                from core.pci import PCIEnvelope

                # Reconstruct envelope and verify through gate chain
                envelope = PCIEnvelope.from_dict(envelope_data)
                result = gatekeeper.verify(envelope)

                if not result.passed:
                    logger.error(
                        f"PCI verification failed: {result.reject_code}",
                        extra={
                            "reject_code": str(result.reject_code),
                            "details": result.details,
                            "gate_passed": result.gate_passed,
                        },
                    )
                    return False

                logger.debug(
                    f"PCI envelope verified: {result.gate_passed}",
                    extra={"gate_passed": result.gate_passed},
                )
                return True

            else:
                # Fallback: manual signature verification
                from core.pci.crypto import (
                    canonical_json,
                    domain_separated_digest,
                    verify_signature,
                )

                # Reconstruct signable content (excluding signature)
                signable = {
                    "version": envelope_data.get("version"),
                    "envelope_id": envelope_data.get("envelope_id"),
                    "timestamp": envelope_data.get("timestamp"),
                    "nonce": envelope_data.get("nonce"),
                    "sender": envelope_data.get("sender"),
                    "payload": envelope_data.get("payload"),
                    "metadata": envelope_data.get("metadata"),
                }

                digest = domain_separated_digest(canonical_json(signable))

                if not verify_signature(digest, signature, public_key):
                    logger.error("Signature verification failed (manual)")
                    return False

                # Manual threshold checks (Ihsān ≥ 0.95)
                metadata = envelope_data.get("metadata", {})
                ihsan = metadata.get("ihsan_score", 0.0)
                if ihsan < MIN_IHSAN_FOR_PROPAGATION:
                    logger.error(
                        f"Ihsān below threshold: {ihsan} < {MIN_IHSAN_FOR_PROPAGATION}"
                    )
                    return False

                logger.debug("PCI envelope verified (manual fallback)")
                return True

        except ImportError as e:
            logger.error(f"PCI crypto module not available: {e}")
            return False
        except Exception as e:
            logger.error(
                f"Envelope verification error: {e}",
                extra={"error_type": type(e).__name__},
            )
            return False

    def receive_pattern(self, data: Dict) -> bool:
        """
        Process a received pattern from the network.

        Security: Verifies PCI envelope signature before accepting pattern.
        """
        try:
            envelope_data = data.get("envelope")

            if envelope_data:
                # Verify PCI envelope
                if not self._verify_pci_envelope(envelope_data):
                    logger.error(
                        "Pattern rejected: invalid PCI signature",
                        extra={"envelope_id": envelope_data.get("id", "unknown")},
                    )
                    return False

                # Extract pattern from verified envelope
                # Pattern is in payload.data (not payload.content)
                payload = envelope_data.get("payload", {})
                pattern_data = payload.get("data", {})
                if not pattern_data:
                    # Fallback paths
                    pattern_data = payload.get("content", {})
                if not pattern_data:
                    pattern_data = envelope_data.get(
                        "pattern", data.get("pattern", data)
                    )
            else:
                # Legacy format without envelope
                pattern_data = data.get("pattern", data)
                if os.getenv("BIZRA_ALLOW_LEGACY_UNSIGNED_PATTERNS", "0") == "1":
                    logger.warning(
                        "Received unsigned pattern (legacy format) - allowed by env"
                    )
                else:
                    logger.error(
                        "Rejecting unsigned pattern (legacy format). "
                        "Set BIZRA_ALLOW_LEGACY_UNSIGNED_PATTERNS=1 to allow."
                    )
                    return False

            pattern = ElevatedPattern.from_dict(pattern_data)
            return self.store.add_network_pattern(pattern)

        except Exception as e:
            logger.error(
                f"Failed to process pattern: {e}",
                extra={
                    "pattern_id": data.get("pattern", {}).get("pattern_id", "unknown"),
                    "error_type": type(e).__name__,
                },
            )
            return False

    def auto_share_elevated(self):
        """Automatically queue local patterns that are ready for sharing."""
        for pattern in self.store.get_patterns_for_sharing():
            if pattern.status == PatternStatus.LOCAL:
                self.queue_for_propagation(pattern)


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("BIZRA PATTERN PROPAGATION — Simulation")
    print("=" * 70)

    # Create two nodes
    store_a = PatternStore("node_A")
    store_b = PatternStore("node_B")

    # Simulate Node A discovering a pattern through repeated use
    print("\n[Node A] Discovering pattern through usage...")
    for i in range(5):
        store_a.record_pattern_use("query.snr < 0.7", success=True, snr_delta=0.15)

    print(f"  Stats: {store_a.get_stats()}")

    # Create propagation engines
    received_by_b = []
    engine_a = PropagationEngine(
        store_a, broadcast_fn=lambda x: received_by_b.append(x)
    )
    engine_b = PropagationEngine(store_b)

    # Node A shares its patterns
    print("\n[Node A] Sharing patterns with network...")
    engine_a.auto_share_elevated()
    count = engine_a.propagate_pending()
    print(f"  Propagated {count} patterns")

    # Node B receives the pattern
    print("\n[Node B] Receiving patterns...")
    for msg_bytes in received_by_b:
        msg = json.loads(msg_bytes.decode("utf-8"))
        engine_b.receive_pattern(msg)

    print(f"  Node B Stats: {store_b.get_stats()}")

    # Check network patterns on Node B
    print("\n[Node B] Network patterns:")
    for pid, pattern in store_b.network_patterns.items():
        print(f"  - {pid}: impact={pattern.compute_impact_score():.3f}")

    print("\n" + "=" * 70)
    print("✅ Pattern Propagation Demo Complete")
    print("=" * 70)
