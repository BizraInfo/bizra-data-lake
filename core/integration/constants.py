"""
BIZRA Integration Constants — AUTHORITATIVE SOURCE OF TRUTH

╔══════════════════════════════════════════════════════════════════════════════╗
║   ALL MODULES MUST IMPORT THRESHOLDS FROM THIS FILE                          ║
║   Do NOT define IHSAN_THRESHOLD or SNR_THRESHOLD elsewhere.                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Genesis Strict Synthesis v2.2.2 — Cross-Repository Constants

Unified constants across all core modules to ensure consistency.
These values override module-specific constants when using the
IntegrationBridge.

Sovereignty: Single source of truth for quality thresholds.

Canonical Values (v2.2.2):
- IHSAN: 0.95 (standard), 0.99 (strict/consensus), 1.0 (runtime/Z3-proven)
- SNR: 0.85 (minimum/museum floor), 0.95 (T1), 0.98 (T0/elite)

Cross-repo alignment:
- BIZRA-DATA-LAKE: core/integration/constants.py (this file)
- BIZRA-Dual-Agentic-system: core/constants.py
- bizra-omega (Rust): bizra-core/src/lib.rs
- TypeScript: src/core/sovereign/capability-card.ts

Standing on Giants: Shannon • Lamport • Vaswani • Anthropic
"""

from typing import Final

# ═══════════════════════════════════════════════════════════════════════════════
# IHSĀN (إحسان) CONSTITUTIONAL THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════
# These values are LOCKED and require constitutional amendment to change.

# Production Ihsān threshold - balanced for practical flexibility
UNIFIED_IHSAN_THRESHOLD: Final[float] = 0.95
IHSAN_THRESHOLD: Final[float] = 0.95  # Alias for backward compatibility

# Strict threshold for consensus-critical operations
STRICT_IHSAN_THRESHOLD: Final[float] = 0.99

# Runtime threshold - Z3-proven agents only (Four Pillars Pillar 1)
RUNTIME_IHSAN_THRESHOLD: Final[float] = 1.0

# Environment-specific thresholds (aligned with Dual Agentic System)
IHSAN_THRESHOLD_PRODUCTION: Final[float] = 0.95
IHSAN_THRESHOLD_STAGING: Final[float] = 0.95
IHSAN_THRESHOLD_CI: Final[float] = 0.90
IHSAN_THRESHOLD_DEV: Final[float] = 0.80

# ═══════════════════════════════════════════════════════════════════════════════
# IHSĀN DIMENSION WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
# 8-dimensional ethical scoring (must sum to 1.0)

IHSAN_WEIGHTS: Final[dict] = {
    "correctness": 0.22,  # Is it right?
    "safety": 0.22,  # Is it safe?
    "user_benefit": 0.14,  # Does it help?
    "efficiency": 0.12,  # Is it optimal?
    "auditability": 0.12,  # Can it be reviewed?
    "anti_centralization": 0.08,  # Does it decentralize?
    "robustness": 0.06,  # Is it resilient?
    "adl_fairness": 0.04,  # Is it fair (عدل)?
}

# ═══════════════════════════════════════════════════════════════════════════════
# SNR (Signal-to-Noise Ratio) THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

# Base/Minimum SNR threshold - also Museum floor (Pillar 2)
UNIFIED_SNR_THRESHOLD: Final[float] = 0.85
SNR_THRESHOLD: Final[float] = 0.85  # Alias for backward compatibility
MUSEUM_SNR_FLOOR: Final[float] = 0.85

# Tier-specific SNR thresholds (aligned with Dual Agentic System)
SNR_THRESHOLD_T0_ELITE: Final[float] = 0.98
SNR_THRESHOLD_T1_HIGH: Final[float] = 0.95
SNR_THRESHOLD_T2_STANDARD: Final[float] = 0.90
SNR_THRESHOLD_T3_ACCEPTABLE: Final[float] = 0.85
SNR_THRESHOLD_T4_MINIMUM: Final[float] = 0.80

# ═══════════════════════════════════════════════════════════════════════════════
# FOUR PILLARS ARCHITECTURE THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

# Pillar 1: Runtime (The Fortress) - Z3-proven only
PILLAR_1_RUNTIME_IHSAN: Final[float] = 1.0

# Pillar 2: Museum (The Ark) - SNR-scored, awaiting proof
PILLAR_2_MUSEUM_SNR_FLOOR: Final[float] = 0.85

# Pillar 3: Sandbox (The Vestibule) - Simulation only
PILLAR_3_SANDBOX_SNR_FLOOR: Final[float] = 0.70

# Pillar 4: Genesis Cutoff (The Event Horizon)
GENESIS_CUTOFF_HOURS: Final[int] = 72

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIDENCE THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

CONFIDENCE_HIGH: Final[float] = 0.95
CONFIDENCE_MEDIUM: Final[float] = 0.85
CONFIDENCE_LOW: Final[float] = 0.70
CONFIDENCE_MINIMUM: Final[float] = 0.50

# ═══════════════════════════════════════════════════════════════════════════════
# ADL (JUSTICE) INVARIANT THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════
# Standing on Giants: Gini (1912), Harberger (1962), Rawls (1971)
# "Adl (عدل) - Justice is not optional. It is a hard constraint."

# Maximum Gini coefficient - HARD GATE, not warning
# 0.40 represents moderate inequality (below most developed nations)
# Transactions that would push Gini above this are REJECTED
ADL_GINI_THRESHOLD: Final[float] = 0.40

# Harberger tax rate (annual, applied continuously)
# Flows to Universal Basic Compute (UBC) pool
ADL_HARBERGER_TAX_RATE: Final[float] = 0.05

# Minimum holding to be considered a participant
# Prevents dust attacks and ensures meaningful participation
ADL_MINIMUM_HOLDING: Final[float] = 1e-9


# ═══════════════════════════════════════════════════════════════════════════════
# TIMING CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Maximum allowed clock skew for timestamp validation
UNIFIED_CLOCK_SKEW_SECONDS = 120

# Nonce TTL for replay protection
UNIFIED_NONCE_TTL_SECONDS = 300

# Pattern sync interval
UNIFIED_SYNC_INTERVAL_SECONDS = 60

# Consensus check interval
UNIFIED_CONSENSUS_INTERVAL_SECONDS = 30

# Agent timeout
UNIFIED_AGENT_TIMEOUT_MS = 30000

# ═══════════════════════════════════════════════════════════════════════════════
# NETWORK CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Default bind address for federation
DEFAULT_FEDERATION_BIND = "0.0.0.0:7654"

# Default A2A port offset from federation port
A2A_PORT_OFFSET = 100

# Maximum retry attempts for A2A operations
MAX_RETRY_ATTEMPTS = 3

# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Primary LLM backend
LMSTUDIO_URL = "http://192.168.56.1:1234"

# Fallback LLM backend
OLLAMA_URL = "http://localhost:11434"

# Model directory (unified path)
MODEL_DIR = "/mnt/c/BIZRA-DATA-LAKE/models"


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-REPOSITORY SYNC
# ═══════════════════════════════════════════════════════════════════════════════

# Repository paths for threshold synchronization
CROSS_REPO_CONSTANTS = {
    "bizra-data-lake": "/mnt/c/BIZRA-DATA-LAKE/core/integration/constants.py",
    "dual-agentic-system": "/mnt/c/BIZRA-Dual-Agentic-system--main/core/constants.py",
    "bizra-omega-rust": "/mnt/c/BIZRA-DATA-LAKE/bizra-omega/bizra-core/src/lib.rs",
}

# Canonical threshold values for cross-repo validation
CANONICAL_THRESHOLDS = {
    "IHSAN_THRESHOLD": 0.95,
    "SNR_THRESHOLD_MINIMUM": 0.85,
    "SNR_THRESHOLD_T0_ELITE": 0.98,
    "MUSEUM_SNR_FLOOR": 0.85,
    "RUNTIME_IHSAN": 1.0,
    "ADL_GINI_THRESHOLD": 0.40,  # Justice invariant - anti-plutocracy
}


def validate_cross_repo_consistency() -> dict:
    """
    Validate threshold consistency across repositories.

    Returns:
        dict with validation results per repo
    """
    import re
    from pathlib import Path

    results = {}

    for repo, path in CROSS_REPO_CONSTANTS.items():
        p = Path(path)
        if not p.exists():
            results[repo] = {"status": "not_found", "path": path}
            continue

        content = p.read_text()
        drift_count = 0

        # Check for IHSAN threshold
        if "0.95" in content:
            # Verify it's the correct context
            if repo == "bizra-omega-rust":
                match = re.search(r"IHSAN_THRESHOLD.*=.*0\.95", content)
            else:
                match = re.search(r"IHSAN_THRESHOLD.*=.*0\.95", content)
            if not match:
                drift_count += 1
        else:
            drift_count += 1

        results[repo] = {
            "status": "synced" if drift_count == 0 else "drift_detected",
            "drift_count": drift_count,
            "path": path,
        }

    return results
