"""
BIZRA Treasury Mode - Graceful Degradation for the Wealth Engine
================================================================
GAP-C4 Solution: The Wealth Engine operates in single-mode (ethical trading only).
If markets become unethical, the system starves. This module implements graceful
degradation through a state machine that transitions between operational modes.

State Machine:
    ETHICAL (full trading) <--> HIBERNATION (minimal ops) <--> EMERGENCY (treasury unlock)

Module Structure (SPARC refinement):
- treasury_types.py      — Constants, enums, data classes
- treasury_persistence.py — SQLite-based state storage
- treasury_controller.py  — State machine controller
- treasury_mode.py        — Public facade (this file)

Standing on Giants:
- Shannon (1948): SNR for market quality assessment
- Lamport (1982): State machine replication for consensus
- DDAGI Constitution: Ihsan constraint (>= 0.95) as ethical floor

"Every node is sovereign. Every human is a seed."
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

# Controller
from .treasury_controller import TreasuryController

# Persistence layer
from .treasury_persistence import TreasuryPersistence

# Types, constants, enums
from .treasury_types import (  # Constants; Enums; Data classes
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

# =============================================================================
# PUBLIC API - Re-export from modular components
# =============================================================================

logger = logging.getLogger(__name__)

# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_treasury_controller(
    initial_reserves: float = 10000.0,
    initial_treasury: float = 50000.0,
    db_path: Optional[Path] = None,
) -> TreasuryController:
    """
    Create a configured TreasuryController instance.

    Args:
        initial_reserves: Initial operational reserves in SEED
        initial_treasury: Initial locked treasury in SEED
        db_path: Optional custom database path

    Returns:
        Configured TreasuryController
    """
    snr_calculator = None
    try:
        from core.iaas.snr_v2 import SNRCalculatorV2

        snr_calculator = SNRCalculatorV2()
    except ImportError:
        logger.warning("SNR calculator not available")

    return TreasuryController(
        initial_reserves_seed=initial_reserves,
        initial_treasury_seed=initial_treasury,
        db_path=db_path,
        snr_calculator=snr_calculator,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "TreasuryMode",
    "TransitionTrigger",
    "TreasuryEvent",
    # Data classes
    "TreasuryState",
    "TransitionEvent",
    "EthicsAssessment",
    # Controller
    "TreasuryController",
    "TreasuryPersistence",
    # Factory
    "create_treasury_controller",
    # Constants
    "COMPUTE_MULTIPLIERS",
    "DEFAULT_BURN_RATE",
    "ETHICS_THRESHOLD_HIBERNATION",
    "ETHICS_THRESHOLD_RECOVERY",
    "RESERVES_THRESHOLD_EMERGENCY",
    "RESERVES_THRESHOLD_HIBERNATION",
    "EMERGENCY_TREASURY_UNLOCK_PERCENT",
]

# =============================================================================
# DEMO / SELF-TEST
# =============================================================================

if __name__ == "__main__":
    import tempfile

    print("=" * 70)
    print("BIZRA TREASURY MODE - Demo")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_memory.db"

        controller = TreasuryController(
            initial_reserves_seed=5000.0,
            initial_treasury_seed=50000.0,
            db_path=db_path,
        )

        print("\n[Initial State]")
        status = controller.get_status()
        print(f"  Mode: {status['state']['mode']}")
        print(f"  Reserves: {status['state']['reserves_days']:.1f} days")
        print(f"  Ethical Score: {status['state']['ethical_score']:.3f}")
        print(f"  Burn Rate: {status['state']['burn_rate_seed_per_day']:.1f} SEED/day")

        print("\n[Simulating Market Ethics Degradation]")
        market_data = {
            "transparency_score": 0.40,
            "fairness_score": 0.35,
            "manipulation_detected": True,
            "volatility_index": 0.80,
            "liquidity_score": 0.50,
        }

        ethics_score = controller.evaluate_market_ethics(market_data)
        print(f"  New Ethics Score: {ethics_score:.3f}")
        print(f"  Should Hibernate: {controller.should_hibernate()}")

        if controller.should_hibernate():
            print("\n[Transitioning to Hibernation]")
            success, msg = controller.transition_mode(
                TreasuryMode.HIBERNATION,
                "Market ethics below threshold",
                TransitionTrigger.MARKET_ETHICS_DEGRADED,
            )
            print(f"  Result: {msg}")
            print(
                f"  New Burn Rate: {controller.state.burn_rate_seed_per_day:.1f} SEED/day"
            )

        print("\n[Health Check]")
        health = controller.health_check()
        print(f"  Healthy: {health['healthy']}")
        for warning in health.get("warnings", []):
            print(f"  Warning: {warning}")

    print("\n" + "=" * 70)
    print("Treasury Mode Demo Complete")
    print("=" * 70)
