"""Frozen harness constants. Change these = new policy bundle version."""

from __future__ import annotations

HARNESS_VERSION = "1.0.0"
SURFACE_CONTRACT_VERSION = "1.0.0"
RECEIPT_SCHEMA_VERSION = "1.0.0"

# Two distinct thresholds — never conflate them
EXECUTION_IHSAN_FLOOR = 0.85  # Below this: gate blocks EXECUTABLE
FEDERATION_IHSAN_FLOOR = 0.95  # Below this: receipt is not federable

# Reflex precipitation
REFLEX_PRECIPITATION_HITS = 3  # Successful deliberate runs before compile

# Gate chain order (canonical, matches TopologyCanon)
GATE_ORDER = ("Schema", "Ihsan", "SNR")

# Verdict precedence (canonical, matches verdict.rs)
VERDICT_PRECEDENCE = ("RIBA", "ZANN", "FATE", "Ihsan", "SNR")
