"""
System Agentic Team (SAT) Controller — Ecosystem Homeostasis Engine.

SAT's mandate: Maintain ecosystem balance through autonomous monitoring
and rebalancing of Proof-of-Impact distributions.

Standing on Giants:
- Ostrom (1990): Commons governance without tragedy
- Axelrod (1984): Cooperation through repeated games
- Al-Ghazali (1097): Proportional justice (zakat)
- Gini (1912): Inequality measurement
- Piketty (2013): Capital concentration dynamics

Mechanisms:
1. Monitor: Track Gini coefficient across PoI distributions
2. Detect: Trigger rebalancing when inequality exceeds threshold
3. Redistribute: Apply progressive or zakat-based rebalancing
4. Audit: Log every rebalancing event with full provenance
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.proof_engine.poi_engine import (
    PoIConfig,
    PoIOrchestrator,
    RebalanceResult,
    SATRebalancer,
    compute_gini,
    compute_token_distribution,
)

logger = logging.getLogger("sovereign.sat")


# =============================================================================
# URP (Universal Resource Pool) Snapshot
# =============================================================================


@dataclass
class URPSnapshot:
    """Universal Resource Pool state at a point in time."""

    total_compute_credits: int = 0
    allocated_credits: int = 0
    available_credits: int = 0
    holder_credits: Dict[str, int] = field(default_factory=dict)
    gini_coefficient: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_compute_credits": self.total_compute_credits,
            "allocated_credits": self.allocated_credits,
            "available_credits": self.available_credits,
            "num_holders": len(self.holder_credits),
            "gini_coefficient": self.gini_coefficient,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# Rebalancing Event (audit record)
# =============================================================================


@dataclass
class RebalancingEvent:
    """Record of a SAT rebalancing action."""

    event_id: str
    timestamp: datetime
    reason: str
    strategy: str
    gini_before: float
    gini_after: float
    credits_redistributed: float
    contributors_affected: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason,
            "strategy": self.strategy,
            "gini_before": self.gini_before,
            "gini_after": self.gini_after,
            "credits_redistributed": self.credits_redistributed,
            "contributors_affected": self.contributors_affected,
        }


# =============================================================================
# SAT Controller
# =============================================================================


class SATController:
    """System Agentic Team (SAT) autonomous controller.

    The SAT is BIZRA's "immune system" — it ensures the ecosystem
    remains healthy, balanced, and aligned with constitutional principles.

    Responsibilities:
    1. Monitor URP credit distribution (Gini coefficient)
    2. Trigger rebalancing when inequality exceeds threshold
    3. Support multiple rebalancing strategies
    4. Maintain audit trail of all actions
    """

    def __init__(
        self,
        poi_orchestrator: Optional[PoIOrchestrator] = None,
        config: Optional[PoIConfig] = None,
    ):
        self.config = config or PoIConfig()
        self.poi_orchestrator = poi_orchestrator
        self.rebalancer = SATRebalancer(self.config)

        # URP state
        self._urp_credits: Dict[str, int] = {}
        self._urp_snapshots: List[URPSnapshot] = []

        # Audit trail
        self._events: List[RebalancingEvent] = []
        self._event_counter = 0

        # Epoch tracking
        self._epochs_finalized: int = 0

    def _next_event_id(self) -> str:
        self._event_counter += 1
        return f"sat_{self._event_counter:08d}_{int(time.time() * 1000)}"

    # ─── URP Credit Management ─────────────────────────────────

    def allocate_credits(self, contributor_id: str, credits: int) -> None:
        """Allocate URP compute credits to a contributor."""
        current = self._urp_credits.get(contributor_id, 0)
        self._urp_credits[contributor_id] = current + credits

    def adjust_credits(self, contributor_id: str, delta: int) -> None:
        """Adjust a contributor's credits (positive or negative)."""
        current = self._urp_credits.get(contributor_id, 0)
        self._urp_credits[contributor_id] = max(0, current + delta)

    def get_credits(self, contributor_id: str) -> int:
        """Get a contributor's current URP credits."""
        return self._urp_credits.get(contributor_id, 0)

    def get_urp_snapshot(self) -> URPSnapshot:
        """Capture current URP state."""
        values = list(self._urp_credits.values())
        total = sum(values)
        gini = compute_gini([float(v) for v in values]) if values else 0.0

        snapshot = URPSnapshot(
            total_compute_credits=total,
            allocated_credits=total,
            available_credits=0,
            holder_credits=dict(self._urp_credits),
            gini_coefficient=gini,
        )
        self._urp_snapshots.append(snapshot)
        return snapshot

    # ─── Rebalancing ───────────────────────────────────────────

    def check_and_rebalance(self) -> Optional[RebalancingEvent]:
        """Check Gini coefficient and trigger rebalancing if needed.

        Returns a RebalancingEvent if rebalancing was performed, None otherwise.
        """
        snapshot = self.get_urp_snapshot()

        if snapshot.gini_coefficient <= self.config.gini_rebalance_threshold:
            return None

        return self.rebalance(
            reason=(
                f"Gini coefficient {snapshot.gini_coefficient:.3f} "
                f"exceeds threshold {self.config.gini_rebalance_threshold}"
            ),
            strategy="computational_zakat",
        )

    def rebalance(
        self,
        reason: str,
        strategy: str = "computational_zakat",
    ) -> RebalancingEvent:
        """Execute a rebalancing operation.

        Strategies:
        - computational_zakat: 2.5% levy on excess, redistributed to neediest
        - progressive_redistribution: Transfer from top to bottom
        """
        logger.warning(f"SAT rebalancing triggered: {reason}")

        # Snapshot before
        gini_before = (
            compute_gini([float(v) for v in self._urp_credits.values()])
            if self._urp_credits
            else 0.0
        )

        if strategy == "computational_zakat":
            result = self._computational_zakat()
        elif strategy == "progressive_redistribution":
            result = self._progressive_redistribution()
        else:
            logger.error(f"Unknown strategy: {strategy}")
            result = RebalanceResult(
                original_scores={},
                rebalanced_scores={},
                gini_before=gini_before,
                gini_after=gini_before,
                zakat_collected=0.0,
                zakat_distributed=0.0,
                rebalance_triggered=False,
            )

        # Snapshot after
        gini_after = (
            compute_gini([float(v) for v in self._urp_credits.values()])
            if self._urp_credits
            else 0.0
        )

        event = RebalancingEvent(
            event_id=self._next_event_id(),
            timestamp=datetime.now(timezone.utc),
            reason=reason,
            strategy=strategy,
            gini_before=gini_before,
            gini_after=gini_after,
            credits_redistributed=result.zakat_collected,
            contributors_affected=len(result.rebalanced_scores),
        )
        self._events.append(event)

        logger.info(
            f"SAT rebalancing complete: Gini {gini_before:.3f} -> {gini_after:.3f}, "
            f"strategy={strategy}, credits_moved={result.zakat_collected:.1f}"
        )
        return event

    def _computational_zakat(self) -> RebalanceResult:
        """Apply 2.5% computational zakat on excess holdings.

        Standing on: Al-Ghazali — wealth must flow from surplus to need.
        """
        scores = {k: float(v) for k, v in self._urp_credits.items()}
        result = self.rebalancer.rebalance(scores)

        if result.rebalance_triggered:
            # Apply rebalanced scores back to URP
            for contributor, new_score in result.rebalanced_scores.items():
                self._urp_credits[contributor] = int(new_score)

        return result

    def _progressive_redistribution(self) -> RebalanceResult:
        """Progressive redistribution from top to bottom.

        Standing on: Piketty — capital concentration requires active redistribution.
        """
        # Use the same SAT rebalancer with a tighter threshold
        temp_config = PoIConfig(
            gini_rebalance_threshold=0.01,  # Always trigger
            zakat_rate=0.10,  # 10% progressive levy (vs 2.5% zakat)
            zakat_exemption_floor=self.config.zakat_exemption_floor,
        )
        temp_rebalancer = SATRebalancer(temp_config)

        scores = {k: float(v) for k, v in self._urp_credits.items()}
        result = temp_rebalancer.rebalance(scores)

        if result.rebalance_triggered:
            for contributor, new_score in result.rebalanced_scores.items():
                self._urp_credits[contributor] = int(new_score)

        return result

    # ─── Epoch Finalization ────────────────────────────────────

    def finalize_epoch(
        self,
        epoch_reward: float = 1000.0,
        epoch_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Finalize a PoI epoch: compute scores, distribute tokens, check balance.

        This is the SAT's primary periodic action. Returns a summary dict.
        """
        if self.poi_orchestrator is None:
            return {"error": "No PoI orchestrator attached"}

        # Compute epoch
        audit = self.poi_orchestrator.compute_epoch(epoch_id)

        # Distribute tokens (informational — token ledger is external)
        distribution = compute_token_distribution(audit, epoch_reward)

        # Allocate URP credits proportionally to PoI scores
        for poi in audit.poi_scores:
            credits = int(poi.poi_score * 10000)  # 0-10k credits per unit PoI
            self.allocate_credits(poi.contributor_id, credits)

        # Check and rebalance if needed
        rebalancing_event = self.check_and_rebalance()

        self._epochs_finalized += 1

        summary = {
            "epoch_id": audit.epoch_id,
            "total_contributors": len(audit.poi_scores),
            "gini_coefficient": audit.gini_coefficient,
            "tokens_distributed": distribution.total_minted,
            "rebalancing_triggered": rebalancing_event is not None,
            "epochs_finalized": self._epochs_finalized,
        }

        if rebalancing_event:
            summary["rebalancing"] = rebalancing_event.to_dict()

        return summary

    # ─── Stats and Audit ───────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Get SAT controller statistics."""
        values = [float(v) for v in self._urp_credits.values()]
        gini = compute_gini(values) if values else 0.0

        return {
            "total_holders": len(self._urp_credits),
            "total_credits": sum(self._urp_credits.values()),
            "gini_coefficient": gini,
            "gini_threshold": self.config.gini_rebalance_threshold,
            "rebalancing_events": len(self._events),
            "epochs_finalized": self._epochs_finalized,
            "zakat_rate": self.config.zakat_rate,
        }

    def get_rebalancing_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent rebalancing events."""
        return [e.to_dict() for e in self._events[-limit:]]

    def get_top_contributors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top contributors by URP credits."""
        sorted_holders = sorted(
            self._urp_credits.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:limit]

        return [
            {"contributor_id": cid, "credits": credits}
            for cid, credits in sorted_holders
        ]
