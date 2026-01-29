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
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Any, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

ELEVATION_THRESHOLD = 3  # Min repetitions to elevate
MIN_SNR_FOR_ELEVATION = 0.75  # Minimum SNR to consider for elevation
MIN_IHSAN_FOR_PROPAGATION = 0.95  # Ihsān floor for network sharing
PATTERN_TTL_HOURS = 168  # 7 days
MAX_PATTERNS_CACHE = 1000

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
            if snr_delta >= MIN_SNR_FOR_ELEVATION - 0.75:  # Relative check
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
    """

    def __init__(
        self, store: PatternStore, broadcast_fn: Callable[[bytes], None] = None
    ):
        self.store = store
        self.broadcast = broadcast_fn or (lambda x: None)
        self._propagation_queue: List[ElevatedPattern] = []

    def queue_for_propagation(self, pattern: ElevatedPattern):
        """Add a pattern to the propagation queue."""
        if pattern.ihsan_score < MIN_IHSAN_FOR_PROPAGATION:
            print(
                f"⚠️ Pattern {pattern.pattern_id} below Ihsān threshold, not propagating"
            )
            return
        self._propagation_queue.append(pattern)

    def propagate_pending(self) -> int:
        """Send all queued patterns to the network."""
        count = 0
        while self._propagation_queue:
            pattern = self._propagation_queue.pop(0)
            pattern.status = PatternStatus.PROPOSED

            # Create propagation message
            msg = json.dumps(
                {"type": "PATTERN_PROPAGATE", "pattern": pattern.to_dict()}
            ).encode("utf-8")

            self.broadcast(msg)
            count += 1
            print(f"📡 Propagated pattern: {pattern.pattern_id}")

        return count

    def receive_pattern(self, data: Dict) -> bool:
        """Process a received pattern from the network."""
        try:
            pattern = ElevatedPattern.from_dict(data.get("pattern", data))
            return self.store.add_network_pattern(pattern)
        except Exception as e:
            print(f"❌ Failed to process pattern: {e}")
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
