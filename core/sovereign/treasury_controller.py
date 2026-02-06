"""
Treasury Controller — State Machine for Treasury Mode Management
================================================================
Evaluates market conditions and manages transitions between modes:
- ETHICAL: Full operation with ethical trading
- HIBERNATION: Minimal compute to preserve reserves
- EMERGENCY: Community funding and treasury unlock

Standing on Giants:
- Shannon (1948): SNR for market quality assessment
- Lamport (1982): State machine replication for consensus
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .treasury_persistence import TreasuryPersistence
from .treasury_types import (
    COMPUTE_MULTIPLIERS,
    DEFAULT_BURN_RATE,
    EMERGENCY_TREASURY_UNLOCK_PERCENT,
    ETHICS_THRESHOLD_HIBERNATION,
    ETHICS_THRESHOLD_RECOVERY,
    RESERVES_THRESHOLD_EMERGENCY,
    RESERVES_THRESHOLD_HIBERNATION,
    EthicsAssessment,
    TransitionEvent,
    TransitionTrigger,
    TreasuryEvent,
    TreasuryMode,
    TreasuryState,
)

logger = logging.getLogger(__name__)


class TreasuryController:
    """
    State machine controller for treasury mode management.

    Integration points:
    - SNR v2 for market ethics scoring
    - Federation for network-wide awareness
    - Persistence for state recovery
    """

    # Valid state transitions
    VALID_TRANSITIONS: Dict[TreasuryMode, List[TreasuryMode]] = {
        TreasuryMode.ETHICAL: [TreasuryMode.HIBERNATION],
        TreasuryMode.HIBERNATION: [TreasuryMode.ETHICAL, TreasuryMode.EMERGENCY],
        TreasuryMode.EMERGENCY: [TreasuryMode.HIBERNATION, TreasuryMode.ETHICAL],
    }

    def __init__(
        self,
        initial_reserves_seed: float = 10000.0,
        initial_treasury_seed: float = 50000.0,
        db_path: Optional[Path] = None,
        snr_calculator: Optional[Any] = None,
        federation_broadcast: Optional[Callable[[Dict], None]] = None,
    ):
        """
        Initialize the treasury controller.

        Args:
            initial_reserves_seed: Initial operational reserves in SEED
            initial_treasury_seed: Initial locked treasury in SEED
            db_path: Path to SQLite database
            snr_calculator: Optional SNR v2 calculator for ethics scoring
            federation_broadcast: Callback for federation event broadcast
        """
        self._persistence = TreasuryPersistence(db_path)
        self._snr_calculator = snr_calculator
        self._federation_broadcast = federation_broadcast
        self._event_handlers: Dict[TreasuryEvent, List[Callable]] = {
            event: [] for event in TreasuryEvent
        }

        # Load or initialize state
        loaded_state = self._persistence.load_state()
        if loaded_state:
            self._state = loaded_state
            logger.info(f"Loaded treasury state: mode={self._state.mode.value}")
        else:
            self._state = self._create_initial_state(
                initial_reserves_seed, initial_treasury_seed
            )
            self._persistence.save_state(self._state)
            logger.info("Initialized treasury state in ETHICAL mode")

    def _create_initial_state(
        self,
        initial_reserves: float,
        initial_treasury: float,
    ) -> TreasuryState:
        """Create initial treasury state."""
        initial_multiplier = COMPUTE_MULTIPLIERS.get(TreasuryMode.ETHICAL.value, 1.0)
        burn_rate = DEFAULT_BURN_RATE * initial_multiplier
        reserves_days = initial_reserves / burn_rate if burn_rate > 0 else 0

        return TreasuryState(
            mode=TreasuryMode.ETHICAL,
            reserves_days=reserves_days,
            ethical_score=1.0,
            last_transition=datetime.utcnow(),
            transition_reason="Initial state",
            burn_rate_seed_per_day=burn_rate,
            total_reserves_seed=initial_reserves,
            locked_treasury_seed=initial_treasury,
            unlocked_treasury_seed=0.0,
        )

    @property
    def state(self) -> TreasuryState:
        """Get current treasury state."""
        return self._state

    @property
    def mode(self) -> TreasuryMode:
        """Get current mode."""
        return self._state.mode

    # -------------------------------------------------------------------------
    # MARKET ETHICS EVALUATION
    # -------------------------------------------------------------------------

    def evaluate_market_ethics(
        self,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Evaluate market ethics score using SNR-inspired methodology.

        Returns a score from 0.0 (unethical) to 1.0 (fully ethical).
        """
        weights = {
            "transparency": 0.20,
            "fairness": 0.25,
            "sustainability": 0.20,
            "compliance": 0.15,
            "ihsan_alignment": 0.20,
        }

        if market_data:
            transparency = self._assess_transparency(market_data)
            fairness = self._assess_fairness(market_data)
            sustainability = self._assess_sustainability(market_data)
            compliance = self._assess_compliance(market_data)
            ihsan = self._assess_ihsan_alignment(market_data)
        else:
            transparency = fairness = 0.75
            sustainability = 0.80
            compliance = 0.85
            ihsan = 0.80

        overall_score = (
            weights["transparency"] * transparency
            + weights["fairness"] * fairness
            + weights["sustainability"] * sustainability
            + weights["compliance"] * compliance
            + weights["ihsan_alignment"] * ihsan
        )

        assessment = EthicsAssessment(
            overall_score=overall_score,
            transparency_score=transparency,
            fairness_score=fairness,
            sustainability_score=sustainability,
            compliance_score=compliance,
            ihsan_alignment=ihsan,
            confidence=0.85 if market_data else 0.50,
            data_sources=list(market_data.keys()) if market_data else [],
        )
        self._persistence.record_ethics_assessment(assessment)

        self._state.ethical_score = overall_score
        self._persistence.save_state(self._state)

        self._emit_event(
            TreasuryEvent.ETHICS_SCORE_UPDATE,
            {
                "score": overall_score,
                "assessment": assessment.to_dict(),
            },
        )

        logger.info(f"Market ethics score: {overall_score:.3f}")
        return overall_score

    def _assess_transparency(self, market_data: Dict) -> float:
        """Assess market transparency."""
        indicators = [
            "price_discovery_quality",
            "information_availability",
            "disclosure_completeness",
        ]
        scores = [
            market_data.get(ind, 0.75) for ind in indicators if ind in market_data
        ]
        return sum(scores) / len(scores) if scores else 0.75

    def _assess_fairness(self, market_data: Dict) -> float:
        """Assess market fairness."""
        if market_data.get("manipulation_detected", False):
            return 0.20
        spread_fairness = market_data.get("spread_fairness", 0.80)
        access_equality = market_data.get("access_equality", 0.85)
        return (spread_fairness + access_equality) / 2

    def _assess_sustainability(self, market_data: Dict) -> float:
        """Assess market sustainability."""
        volatility = market_data.get("volatility_index", 0.30)
        liquidity = market_data.get("liquidity_score", 0.75)
        volatility_penalty = max(0, volatility - 0.50) * 0.5
        return max(0.0, min(1.0, liquidity - volatility_penalty))

    def _assess_compliance(self, market_data: Dict) -> float:
        """Assess regulatory compliance."""
        compliance_flags = market_data.get("compliance_flags", [])
        base_compliance = market_data.get("compliance_score", 0.90)
        penalty = len(compliance_flags) * 0.10
        return max(0.0, base_compliance - penalty)

    def _assess_ihsan_alignment(self, market_data: Dict) -> float:
        """Assess alignment with Ihsan (excellence) principles."""
        if self._snr_calculator:
            try:
                snr_result = self._snr_calculator.calculate_simple(
                    query="market ethics assessment",
                    texts=[json.dumps(market_data)],
                    iaas_score=0.85,
                )
                return min(1.0, snr_result.snr / 0.95)
            except Exception as e:
                logger.warning(f"SNR calculation failed: {e}")
        return market_data.get("ihsan_alignment", 0.80)

    # -------------------------------------------------------------------------
    # BURN RATE CALCULATION
    # -------------------------------------------------------------------------

    def calculate_burn_rate(self, mode: Optional[TreasuryMode] = None) -> float:
        """Calculate current SEED burn rate per day."""
        if mode is None:
            mode = self._state.mode if self._state else TreasuryMode.ETHICAL

        multiplier = COMPUTE_MULTIPLIERS.get(mode.value, 1.0)
        burn_rate = DEFAULT_BURN_RATE * multiplier

        if self._state and mode == self._state.mode:
            self._state.burn_rate_seed_per_day = burn_rate
            if burn_rate > 0:
                self._state.reserves_days = self._state.total_reserves_seed / burn_rate
            self._persistence.save_state(self._state)
            self._emit_event(
                TreasuryEvent.BURN_RATE_UPDATE,
                {
                    "burn_rate": burn_rate,
                    "mode": mode.value,
                    "reserves_days": self._state.reserves_days,
                },
            )

        return burn_rate

    # -------------------------------------------------------------------------
    # MODE TRANSITION
    # -------------------------------------------------------------------------

    def transition_mode(
        self,
        new_mode: TreasuryMode,
        reason: str,
        trigger: TransitionTrigger = TransitionTrigger.MANUAL_OVERRIDE,
        force: bool = False,
    ) -> Tuple[bool, str]:
        """Transition to a new treasury mode."""
        old_mode = self._state.mode

        if old_mode == new_mode:
            return True, f"Already in {new_mode.value} mode"

        if not force:
            valid_targets = self.VALID_TRANSITIONS.get(old_mode, [])
            if new_mode not in valid_targets:
                msg = f"Invalid transition: {old_mode.value} -> {new_mode.value}"
                logger.warning(msg)
                return False, msg

        event = TransitionEvent(
            from_mode=old_mode,
            to_mode=new_mode,
            trigger=trigger,
            ethical_score_at_transition=self._state.ethical_score,
            reserves_days_at_transition=self._state.reserves_days,
            reason=reason,
            metadata={
                "forced": force,
                "burn_rate_before": self._state.burn_rate_seed_per_day,
            },
        )

        # Execute mode-specific protocols
        if new_mode == TreasuryMode.EMERGENCY:
            self._execute_emergency_protocol()
        elif new_mode == TreasuryMode.HIBERNATION:
            self._execute_hibernation_protocol()
        elif new_mode == TreasuryMode.ETHICAL:
            self._execute_recovery_protocol()

        self._state.mode = new_mode
        self._state.last_transition = datetime.utcnow()
        self._state.transition_reason = reason

        self.calculate_burn_rate()
        self._persistence.save_state(self._state)
        self._persistence.record_transition(event)

        self._emit_event(
            TreasuryEvent.MODE_TRANSITION,
            {
                "from_mode": old_mode.value,
                "to_mode": new_mode.value,
                "reason": reason,
                "trigger": trigger.value,
                "event": event.to_dict(),
            },
        )

        self._broadcast_to_federation(
            {
                "type": "TREASURY_MODE_TRANSITION",
                "from_mode": old_mode.value,
                "to_mode": new_mode.value,
                "timestamp": datetime.utcnow().isoformat(),
                "reason": reason,
            }
        )

        logger.info(f"Treasury mode transition: {old_mode.value} -> {new_mode.value}")
        return True, f"Transitioned from {old_mode.value} to {new_mode.value}"

    def _execute_emergency_protocol(self) -> None:
        """Execute emergency mode protocol."""
        unlock_amount = (
            self._state.locked_treasury_seed * EMERGENCY_TREASURY_UNLOCK_PERCENT
        )
        self._state.locked_treasury_seed -= unlock_amount
        self._state.unlocked_treasury_seed += unlock_amount

        self._emit_event(
            TreasuryEvent.TREASURY_UNLOCK,
            {
                "amount": unlock_amount,
                "reason": "Emergency protocol activation",
                "remaining_locked": self._state.locked_treasury_seed,
            },
        )

        self._emit_event(
            TreasuryEvent.COMMUNITY_APPEAL,
            {
                "message": "Emergency mode activated. Community support requested.",
                "reserves_days": self._state.reserves_days,
                "unlocked_amount": unlock_amount,
            },
        )

        self._broadcast_to_federation(
            {
                "type": "TREASURY_EMERGENCY_APPEAL",
                "timestamp": datetime.utcnow().isoformat(),
                "reserves_days": self._state.reserves_days,
                "appeal": "Community funding requested for node survival",
            }
        )

        logger.warning(f"Emergency protocol: Unlocked {unlock_amount:.2f} SEED")

    def _execute_hibernation_protocol(self) -> None:
        """Execute hibernation mode protocol."""
        logger.info("Hibernation protocol: Reducing compute to EDGE only")

    def _execute_recovery_protocol(self) -> None:
        """Execute recovery to ethical mode."""
        self._emit_event(
            TreasuryEvent.RECOVERY_INITIATED,
            {
                "from_mode": self._state.mode.value,
                "ethical_score": self._state.ethical_score,
                "reserves_days": self._state.reserves_days,
            },
        )
        logger.info("Recovery protocol: Restoring full operations")

    # -------------------------------------------------------------------------
    # CONDITION CHECKS
    # -------------------------------------------------------------------------

    def should_hibernate(self) -> bool:
        """Check if system should enter hibernation mode."""
        if self._state.mode != TreasuryMode.ETHICAL:
            return False
        return self._state.ethical_score < ETHICS_THRESHOLD_HIBERNATION

    def should_emergency(self) -> bool:
        """Check if system should enter emergency mode."""
        if self._state.mode != TreasuryMode.HIBERNATION:
            return False
        return self._state.reserves_days < RESERVES_THRESHOLD_EMERGENCY

    def should_recover(self) -> bool:
        """Check if system can recover to a higher operational mode."""
        if self._state.mode == TreasuryMode.EMERGENCY:
            return self._state.reserves_days >= RESERVES_THRESHOLD_EMERGENCY

        if self._state.mode == TreasuryMode.HIBERNATION:
            return (
                self._state.ethical_score >= ETHICS_THRESHOLD_RECOVERY
                and self._state.reserves_days >= RESERVES_THRESHOLD_HIBERNATION
            )

        return False

    # -------------------------------------------------------------------------
    # AUTONOMOUS EVALUATION
    # -------------------------------------------------------------------------

    async def evaluate_and_transition(
        self,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[TransitionEvent]:
        """Evaluate current conditions and transition if necessary."""
        ethics_score = self.evaluate_market_ethics(market_data)

        if self.should_hibernate():
            success, _ = self.transition_mode(
                TreasuryMode.HIBERNATION,
                f"Market ethics degraded to {ethics_score:.3f}",
                TransitionTrigger.MARKET_ETHICS_DEGRADED,
            )
            if success:
                return self._persistence.get_transition_history(limit=1)[0]

        elif self.should_emergency():
            success, _ = self.transition_mode(
                TreasuryMode.EMERGENCY,
                f"Reserves depleted to {self._state.reserves_days:.1f} days",
                TransitionTrigger.RESERVES_DEPLETED,
            )
            if success:
                return self._persistence.get_transition_history(limit=1)[0]

        elif self.should_recover():
            if self._state.mode == TreasuryMode.EMERGENCY:
                target = TreasuryMode.HIBERNATION
                trigger = TransitionTrigger.RESERVES_REPLENISHED
                reason = f"Reserves recovered to {self._state.reserves_days:.1f} days"
            else:
                target = TreasuryMode.ETHICAL
                trigger = TransitionTrigger.MARKET_ETHICS_RECOVERED
                reason = f"Market ethics recovered to {ethics_score:.3f}"

            success, _ = self.transition_mode(target, reason, trigger)
            if success:
                return self._persistence.get_transition_history(limit=1)[0]

        return None

    # -------------------------------------------------------------------------
    # RESERVES MANAGEMENT
    # -------------------------------------------------------------------------

    def update_reserves(self, amount: float, reason: str = "") -> None:
        """Update reserve levels (add or subtract)."""
        self._state.total_reserves_seed += amount
        self._state.total_reserves_seed = max(0, self._state.total_reserves_seed)

        if self._state.burn_rate_seed_per_day > 0:
            self._state.reserves_days = (
                self._state.total_reserves_seed / self._state.burn_rate_seed_per_day
            )

        self._persistence.save_state(self._state)

        self._emit_event(
            TreasuryEvent.RESERVES_UPDATE,
            {
                "amount": amount,
                "new_total": self._state.total_reserves_seed,
                "reserves_days": self._state.reserves_days,
                "reason": reason,
            },
        )

        logger.info(f"Reserves updated: {amount:+.2f} SEED ({reason})")

    # -------------------------------------------------------------------------
    # EVENT SYSTEM
    # -------------------------------------------------------------------------

    def on_event(
        self, event_type: TreasuryEvent, handler: Callable[[Dict], None]
    ) -> None:
        """Register an event handler."""
        self._event_handlers[event_type].append(handler)

    def _emit_event(self, event_type: TreasuryEvent, data: Dict) -> None:
        """Emit an event to registered handlers."""
        for handler in self._event_handlers.get(event_type, []):
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

    def _broadcast_to_federation(self, message: Dict) -> None:
        """Broadcast message to federation network."""
        if self._federation_broadcast:
            try:
                self._federation_broadcast(message)
            except Exception as e:
                logger.error(f"Federation broadcast error: {e}")

    # -------------------------------------------------------------------------
    # STATUS AND HEALTH
    # -------------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive treasury status."""
        return {
            "state": self._state.to_dict(),
            "mode_multiplier": COMPUTE_MULTIPLIERS.get(self._state.mode.value, 1.0),
            "thresholds": {
                "ethics_hibernation": ETHICS_THRESHOLD_HIBERNATION,
                "ethics_recovery": ETHICS_THRESHOLD_RECOVERY,
                "reserves_emergency": RESERVES_THRESHOLD_EMERGENCY,
                "reserves_hibernation": RESERVES_THRESHOLD_HIBERNATION,
            },
            "conditions": {
                "should_hibernate": self.should_hibernate(),
                "should_emergency": self.should_emergency(),
                "should_recover": self.should_recover(),
            },
            "valid_transitions": [
                t.value for t in self.VALID_TRANSITIONS.get(self._state.mode, [])
            ],
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the treasury system."""
        health = {
            "healthy": True,
            "mode": self._state.mode.value,
            "reserves_days": self._state.reserves_days,
            "ethical_score": self._state.ethical_score,
            "warnings": [],
            "timestamp": datetime.utcnow().isoformat(),
        }

        if self._state.reserves_days < RESERVES_THRESHOLD_HIBERNATION:
            health["warnings"].append(
                f"Low reserves: {self._state.reserves_days:.1f} days"
            )
            if self._state.reserves_days < RESERVES_THRESHOLD_EMERGENCY:
                health["healthy"] = False

        if self._state.ethical_score < ETHICS_THRESHOLD_RECOVERY:
            health["warnings"].append(
                f"Low ethics score: {self._state.ethical_score:.3f}"
            )
            if self._state.ethical_score < ETHICS_THRESHOLD_HIBERNATION:
                health["healthy"] = False

        if self._state.mode == TreasuryMode.EMERGENCY:
            time_in_emergency = datetime.utcnow() - self._state.last_transition
            if time_in_emergency > timedelta(days=7):
                health["healthy"] = False
                health["warnings"].append("Extended emergency mode (>7 days)")

        self._emit_event(TreasuryEvent.HEALTH_CHECK, health)
        return health


__all__ = [
    "TreasuryController",
]
