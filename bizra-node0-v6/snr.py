"""
BIZRA SNR Module — Single Source of Truth
═════════════════════════════════════════

Canonical SNR normalization. Every consumer imports from HERE.

The split-brain problem (mission.py vs snr.py defining different
normalization functions) is resolved: this module IS snr.py.
Delete the copy in mission.py. grep -r "snr_linear" should return
ONLY this file.

Theorem 2.1 (SNR Monotonicity):
    For validated evidence e with PoI ≥ threshold:
        SNR(t+1) ≥ SNR(t)
    Signal grows monotonically. Noise is bounded by constitutional gates.

Mathematical basis:
    SNR_linear = signal_power / noise_power
    SNR_normalized = min(SNR_linear / (1 + SNR_linear), 1.0)

    This sigmoid-like normalization maps [0, ∞) → [0, 1) with:
    - SNR_linear = 0  →  normalized = 0.00
    - SNR_linear = 1  →  normalized = 0.50
    - SNR_linear = 9  →  normalized = 0.90
    - SNR_linear = 19 →  normalized = 0.95
    - SNR_linear = ∞  →  normalized = 1.00 (limit)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
# CANONICAL NORMALIZATION — The one function. No copies. No variants.
# ═══════════════════════════════════════════════════════════════════════════════


def normalize_snr(snr_linear: float) -> float:
    """
    Canonical SNR normalization: maps [0, ∞) → [0, 1).

    This is THE normalization function for the entire BIZRA codebase.
    Every module that needs SNR normalization imports this function.
    No other file should define its own normalization.

    Args:
        snr_linear: Raw signal-to-noise ratio (≥ 0).

    Returns:
        Normalized SNR in [0.0, 1.0].

    Verification:
        After applying this change, run:
            grep -r "snr_linear" bizra_omega/
        Result MUST return ONLY this file.
    """
    if snr_linear < 0:
        return 0.0
    return min(snr_linear / (1.0 + snr_linear), 1.0)


def snr_to_db(snr_linear: float) -> float:
    """Convert linear SNR to decibels. For logging/display only."""
    if snr_linear <= 0:
        return float("-inf")
    return 10.0 * math.log10(snr_linear)


def db_to_snr(snr_db: float) -> float:
    """Convert decibels back to linear SNR."""
    return 10.0 ** (snr_db / 10.0)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPOSITE SNR — For SAPE dimensional scoring
# ═══════════════════════════════════════════════════════════════════════════════

# SAPE weights — derived from constitution.toml analysis methodology
# These are the weights for the 8-dimension SAPE composite score.
SAPE_WEIGHTS = {
    "security": 0.15,
    "architecture": 0.20,
    "error_handling": 0.15,
    "scalability": 0.10,
    "testing": 0.15,
    "documentation": 0.10,
    "dependencies": 0.10,
    "performance": 0.05,
}


@dataclass
class SapeScore:
    """SAPE dimensional score with weighted composite."""
    dimensions: dict[str, float]
    composite: float
    t1_threshold: float = 0.950

    @property
    def passes_t1(self) -> bool:
        return self.composite >= self.t1_threshold

    @property
    def gap_to_t1(self) -> float:
        return max(0.0, self.t1_threshold - self.composite)

    def as_evidence(self) -> dict[str, Any]:
        return {
            "sape_dimensions": self.dimensions,
            "sape_composite": self.composite,
            "t1_threshold": self.t1_threshold,
            "passes_t1": self.passes_t1,
            "gap": self.gap_to_t1,
        }


def compute_sape_composite(
    scores: dict[str, float],
    weights: dict[str, float] | None = None,
) -> SapeScore:
    """
    Compute the SAPE weighted composite score.

    Args:
        scores: Dict mapping dimension name → score (0.0 to 1.0).
        weights: Optional custom weights. Defaults to SAPE_WEIGHTS.

    Returns:
        SapeScore with composite and T1 pass/fail.
    """
    weights = weights or SAPE_WEIGHTS

    composite = 0.0
    for dim, weight in weights.items():
        dim_score = scores.get(dim, 0.0)
        composite += weight * dim_score

    return SapeScore(
        dimensions=dict(scores),
        composite=round(composite, 4),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MISSION-LEVEL SNR — Per-mission signal quality measurement
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class MissionSNR:
    """SNR measurement for a single mission execution."""
    signal_power: float     # Useful information in the output
    noise_power: float      # Irrelevant/harmful/redundant content
    snr_linear: float       # signal / noise
    snr_normalized: float   # normalize_snr(snr_linear)
    snr_db: float           # 10 * log10(snr_linear)

    def as_evidence(self) -> dict[str, float]:
        return {
            "signal_power": self.signal_power,
            "noise_power": self.noise_power,
            "snr_linear": self.snr_linear,
            "snr_normalized": self.snr_normalized,
            "snr_db": self.snr_db,
        }


def measure_mission_snr(
    output: str,
    ihsan_composite: float,
    relevance_score: float = 1.0,
    noise_markers: list[str] | None = None,
) -> MissionSNR:
    """
    Measure the SNR of a mission output.

    Signal = ihsan_composite * relevance * output_density
    Noise = (1 - ihsan_composite) + detected_noise_markers

    Args:
        output: The mission output text.
        ihsan_composite: Ihsan gate composite score (0-1).
        relevance_score: How relevant output is to mission intent (0-1).
        noise_markers: Optional list of noise patterns to detect.

    Returns:
        MissionSNR with linear, normalized, and dB measurements.
    """
    noise_markers = noise_markers or []

    # Signal: Ihsan-weighted relevance
    output_density = min(len(output.strip()) / max(len(output), 1), 1.0)
    signal = ihsan_composite * relevance_score * max(output_density, 0.1)

    # Noise: inverse Ihsan + detected noise
    base_noise = 1.0 - ihsan_composite
    marker_noise = 0.0
    if noise_markers and output:
        marker_hits = sum(1 for m in noise_markers if m.lower() in output.lower())
        marker_noise = min(marker_hits * 0.05, 0.3)

    noise = max(base_noise + marker_noise, 0.001)  # Prevent division by zero

    snr_linear = signal / noise
    snr_normalized = normalize_snr(snr_linear)
    snr_db_val = snr_to_db(snr_linear)

    return MissionSNR(
        signal_power=round(signal, 4),
        noise_power=round(noise, 4),
        snr_linear=round(snr_linear, 4),
        snr_normalized=round(snr_normalized, 4),
        snr_db=round(snr_db_val, 2),
    )
