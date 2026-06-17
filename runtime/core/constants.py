"""
BIZRA Constitutional Constants - Single Source of Truth

Genesis Strict Synthesis v2.2.2 mandates all threshold values
originate from this file. Zero threshold drift permitted.

Standing on Giants: Shannon • Lamport • Vaswani • Anthropic
La hawla wa la quwwata illa billah.
"""

import importlib.util
from enum import Enum
from pathlib import Path
from typing import Final


def _load_repo_integration_constants():
    constants_path = (
        Path(__file__).resolve().parents[2] / "core" / "integration" / "constants.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_bizra_repo_integration_constants", constants_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load canonical constants from {constants_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CANONICAL = _load_repo_integration_constants()

# =============================================================================
# IHSĀN (إحسان) CONSTITUTIONAL THRESHOLDS
# =============================================================================
# Source: constitution/ihsan_v1.yaml
# These values are LOCKED and require constitutional amendment to change.

# Production Ihsān threshold - balanced for practical flexibility
IHSAN_THRESHOLD: Final[float] = _CANONICAL.IHSAN_THRESHOLD

# Environment-specific thresholds
IHSAN_THRESHOLD_PRODUCTION: Final[float] = 0.95
IHSAN_THRESHOLD_STAGING: Final[float] = 0.95
IHSAN_THRESHOLD_CI: Final[float] = 0.90
IHSAN_THRESHOLD_DEV: Final[float] = 0.80

# =============================================================================
# IHSĀN DIMENSION WEIGHTS
# =============================================================================
# 8-dimensional ethical scoring (must sum to 1.0)

IHSAN_WEIGHTS: Final[dict] = {
    "correctness": 0.22,  # Is it right?
    "safety": 0.22,  # Is it safe?
    "user_benefit": 0.14,  # Does it help?
    "efficiency": 0.12,  # Is it optimal?
    "auditability": 0.12,  # Can it be reviewed?
    "anti_centralization": 0.08,  # Does it decentralize?
    "robustness": 0.06,  # Is it resilient?
    "adl_fairness": 0.04,  # Is it fair?
}

# =============================================================================
# SNR (Signal-to-Noise Ratio) THRESHOLDS
# =============================================================================

# Base SNR threshold for quality filtering
SNR_THRESHOLD_BASE: Final[float] = 0.95

# Tier-specific SNR thresholds
SNR_THRESHOLD_T0_ELITE: Final[float] = 0.98
SNR_THRESHOLD_T1_HIGH: Final[float] = 0.95
SNR_THRESHOLD_T2_STANDARD: Final[float] = 0.90
SNR_THRESHOLD_T3_ACCEPTABLE: Final[float] = 0.85
SNR_THRESHOLD_T4_MINIMUM: Final[float] = 0.80

# PAT Peak Mode SNR thresholds
SNR_THRESHOLD_PAT_STANDARD: Final[float] = 0.980
SNR_THRESHOLD_PAT_ELEVATED: Final[float] = 0.985
SNR_THRESHOLD_PAT_SOVEREIGN: Final[float] = 0.990
SNR_THRESHOLD_PAT_TRANSCENDENT: Final[float] = 0.995

# =============================================================================
# MUSEUM MODE THRESHOLDS (Pillar 2)
# =============================================================================
# Unproven code awaiting Z3 synthesis

MUSEUM_SNR_FLOOR: Final[float] = 0.85
MUSEUM_PROMOTION_THRESHOLD: Final[float] = 1.0  # Z3-proven = Ihsān 1.0

# =============================================================================
# NOVELTY THRESHOLDS
# =============================================================================

NOVELTY_THRESHOLD_STANDARD: Final[float] = 0.75
NOVELTY_THRESHOLD_ELEVATED: Final[float] = 0.80
NOVELTY_THRESHOLD_SOVEREIGN: Final[float] = 0.85
NOVELTY_THRESHOLD_TRANSCENDENT: Final[float] = 0.90

# =============================================================================
# CONFIDENCE THRESHOLDS
# =============================================================================

CONFIDENCE_HIGH: Final[float] = 0.95
CONFIDENCE_MEDIUM: Final[float] = 0.85
CONFIDENCE_LOW: Final[float] = 0.70
CONFIDENCE_MINIMUM: Final[float] = 0.50

# =============================================================================
# SAPE PROBE THRESHOLDS
# =============================================================================
# 9-probe verification system

SAPE_THRESHOLD_CRITICAL: Final[float] = (
    0.95  # threat_scan, compliance, safety, correctness
)
SAPE_THRESHOLD_HIGH: Final[float] = 0.90  # bias
SAPE_THRESHOLD_STANDARD: Final[float] = 0.85  # user_benefit, groundedness
SAPE_THRESHOLD_MINIMUM: Final[float] = 0.80  # relevance, fluency

SAPE_PROBE_THRESHOLDS: Final[dict] = {
    "threat_scan": SAPE_THRESHOLD_CRITICAL,
    "compliance": SAPE_THRESHOLD_CRITICAL,
    "bias": SAPE_THRESHOLD_HIGH,
    "user_benefit": SAPE_THRESHOLD_STANDARD,
    "correctness": SAPE_THRESHOLD_CRITICAL,
    "safety": SAPE_THRESHOLD_CRITICAL,
    "groundedness": SAPE_THRESHOLD_STANDARD,
    "relevance": SAPE_THRESHOLD_MINIMUM,
    "fluency": SAPE_THRESHOLD_MINIMUM,
}

# =============================================================================
# CONSENSUS THRESHOLDS
# =============================================================================

SAT_CONSENSUS_REQUIRED: Final[int] = 3  # 3/5 guardians must approve
SAT_GUARDIAN_COUNT: Final[int] = 5

# =============================================================================
# FATE ESCALATION LEVELS
# =============================================================================


class FATELevel(Enum):
    """Fail-Safe Agentic Trust Escalation levels."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


# =============================================================================
# GENESIS PROTOCOL CONSTANTS
# =============================================================================

GENESIS_CUTOFF_HOURS: Final[int] = 72
RUNTIME_LOC_LIMIT: Final[int] = 17500  # ≤17,500 LOC hot path

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_ihsan_threshold(environment: str = "production") -> float:
    """Get Ihsān threshold for the specified environment."""
    thresholds = {
        "production": IHSAN_THRESHOLD_PRODUCTION,
        "staging": IHSAN_THRESHOLD_STAGING,
        "ci": IHSAN_THRESHOLD_CI,
        "dev": IHSAN_THRESHOLD_DEV,
    }
    return thresholds.get(environment.lower(), IHSAN_THRESHOLD_PRODUCTION)


def get_snr_threshold(tier: str = "T1") -> float:
    """Get SNR threshold for the specified tier."""
    thresholds = {
        "T0": SNR_THRESHOLD_T0_ELITE,
        "T1": SNR_THRESHOLD_T1_HIGH,
        "T2": SNR_THRESHOLD_T2_STANDARD,
        "T3": SNR_THRESHOLD_T3_ACCEPTABLE,
        "T4": SNR_THRESHOLD_T4_MINIMUM,
    }
    return thresholds.get(tier.upper(), SNR_THRESHOLD_T1_HIGH)


def calculate_ihsan_score(dimension_scores: dict) -> float:
    """
    Calculate weighted Ihsān score from dimension scores.

    Args:
        dimension_scores: Dict mapping dimension names to scores (0.0-1.0)

    Returns:
        Weighted Ihsān score (0.0-1.0)
    """
    total = 0.0
    for dimension, weight in IHSAN_WEIGHTS.items():
        score = dimension_scores.get(dimension, 0.0)
        total += score * weight
    return total


def validate_ihsan_score(score: float, environment: str = "production") -> bool:
    """Check if Ihsān score meets threshold for environment."""
    threshold = get_ihsan_threshold(environment)
    return score >= threshold
