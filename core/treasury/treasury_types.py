"""
Treasury Types — Data Classes, Enums, and Constants
====================================================
Type definitions for the Treasury Mode system.

Standing on Giants:
- Shannon (1948): SNR for market quality assessment
- Lamport (1982): State machine replication for consensus
- DDAGI Constitution: Ihsan constraint (>= 0.95) as ethical floor
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

# Import unified thresholds from authoritative source
try:
    from core.integration.constants import (
        UNIFIED_IHSAN_THRESHOLD,
        UNIFIED_SNR_THRESHOLD,
    )
except ImportError:
    UNIFIED_IHSAN_THRESHOLD = 0.95
    UNIFIED_SNR_THRESHOLD = 0.85


# =============================================================================
# CONSTANTS
# =============================================================================

# Treasury mode thresholds
ETHICS_THRESHOLD_HIBERNATION: float = 0.60  # Below this -> hibernate
ETHICS_THRESHOLD_RECOVERY: float = 0.75     # Above this -> return to ethical mode
RESERVES_THRESHOLD_EMERGENCY: float = 7.0   # Days - below this -> emergency
RESERVES_THRESHOLD_HIBERNATION: float = 30.0  # Days - below this -> consider hibernation

# Emergency treasury unlock percentage
EMERGENCY_TREASURY_UNLOCK_PERCENT: float = 0.10  # 10%

# Default burn rate (SEED/day)
DEFAULT_BURN_RATE: float = 100.0

# Mode-specific compute multipliers
COMPUTE_MULTIPLIERS: Dict[str, float] = {
    "ethical": 1.0,       # Full URP access
    "hibernation": 0.25,  # EDGE compute only
    "emergency": 0.10,    # Minimal essential operations
}


# =============================================================================
# ENUMS
# =============================================================================

class TreasuryMode(Enum):
    """
    Treasury operational modes.

    ETHICAL: Full operation, ethical trades only
    HIBERNATION: Minimal compute, preserve reserves
    EMERGENCY: Community funding, treasury unlock
    """
    ETHICAL = "ethical"
    HIBERNATION = "hibernation"
    EMERGENCY = "emergency"


class TransitionTrigger(Enum):
    """Triggers for mode transitions."""
    MARKET_ETHICS_DEGRADED = "market_ethics_degraded"
    MARKET_ETHICS_RECOVERED = "market_ethics_recovered"
    RESERVES_DEPLETED = "reserves_depleted"
    RESERVES_REPLENISHED = "reserves_replenished"
    MANUAL_OVERRIDE = "manual_override"
    EMERGENCY_APPEAL_SUCCESSFUL = "emergency_appeal_successful"
    SCHEDULED_REVIEW = "scheduled_review"


class TreasuryEvent(Enum):
    """Events emitted by the treasury controller."""
    MODE_TRANSITION = "mode_transition"
    ETHICS_SCORE_UPDATE = "ethics_score_update"
    RESERVES_UPDATE = "reserves_update"
    BURN_RATE_UPDATE = "burn_rate_update"
    TREASURY_UNLOCK = "treasury_unlock"
    COMMUNITY_APPEAL = "community_appeal"
    RECOVERY_INITIATED = "recovery_initiated"
    HEALTH_CHECK = "health_check"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TreasuryState:
    """Current state of the treasury system."""
    mode: TreasuryMode
    reserves_days: float
    ethical_score: float
    last_transition: datetime
    transition_reason: str
    burn_rate_seed_per_day: float = DEFAULT_BURN_RATE
    total_reserves_seed: float = 0.0
    locked_treasury_seed: float = 0.0
    unlocked_treasury_seed: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "mode": self.mode.value,
            "reserves_days": self.reserves_days,
            "ethical_score": self.ethical_score,
            "last_transition": self.last_transition.isoformat(),
            "transition_reason": self.transition_reason,
            "burn_rate_seed_per_day": self.burn_rate_seed_per_day,
            "total_reserves_seed": self.total_reserves_seed,
            "locked_treasury_seed": self.locked_treasury_seed,
            "unlocked_treasury_seed": self.unlocked_treasury_seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TreasuryState":
        """Deserialize from dictionary."""
        return cls(
            mode=TreasuryMode(data["mode"]),
            reserves_days=data["reserves_days"],
            ethical_score=data["ethical_score"],
            last_transition=datetime.fromisoformat(data["last_transition"]),
            transition_reason=data["transition_reason"],
            burn_rate_seed_per_day=data.get("burn_rate_seed_per_day", DEFAULT_BURN_RATE),
            total_reserves_seed=data.get("total_reserves_seed", 0.0),
            locked_treasury_seed=data.get("locked_treasury_seed", 0.0),
            unlocked_treasury_seed=data.get("unlocked_treasury_seed", 0.0),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "TreasuryState":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class TransitionEvent:
    """Record of a mode transition."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: datetime = field(default_factory=datetime.utcnow)
    from_mode: TreasuryMode = TreasuryMode.ETHICAL
    to_mode: TreasuryMode = TreasuryMode.ETHICAL
    trigger: TransitionTrigger = TransitionTrigger.SCHEDULED_REVIEW
    ethical_score_at_transition: float = 0.0
    reserves_days_at_transition: float = 0.0
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "from_mode": self.from_mode.value,
            "to_mode": self.to_mode.value,
            "trigger": self.trigger.value,
            "ethical_score_at_transition": self.ethical_score_at_transition,
            "reserves_days_at_transition": self.reserves_days_at_transition,
            "reason": self.reason,
            "metadata": self.metadata,
        }


@dataclass
class EthicsAssessment:
    """Result of a market ethics evaluation."""
    overall_score: float
    transparency_score: float = 0.0
    fairness_score: float = 0.0
    sustainability_score: float = 0.0
    compliance_score: float = 0.0
    ihsan_alignment: float = 0.0
    assessment_timestamp: datetime = field(default_factory=datetime.utcnow)
    data_sources: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "overall_score": self.overall_score,
            "transparency_score": self.transparency_score,
            "fairness_score": self.fairness_score,
            "sustainability_score": self.sustainability_score,
            "compliance_score": self.compliance_score,
            "ihsan_alignment": self.ihsan_alignment,
            "assessment_timestamp": self.assessment_timestamp.isoformat(),
            "data_sources": self.data_sources,
            "confidence": self.confidence,
        }


__all__ = [
    # Constants
    "ETHICS_THRESHOLD_HIBERNATION",
    "ETHICS_THRESHOLD_RECOVERY",
    "RESERVES_THRESHOLD_EMERGENCY",
    "RESERVES_THRESHOLD_HIBERNATION",
    "EMERGENCY_TREASURY_UNLOCK_PERCENT",
    "DEFAULT_BURN_RATE",
    "COMPUTE_MULTIPLIERS",
    # Enums
    "TreasuryMode",
    "TransitionTrigger",
    "TreasuryEvent",
    # Data classes
    "TreasuryState",
    "TransitionEvent",
    "EthicsAssessment",
]
