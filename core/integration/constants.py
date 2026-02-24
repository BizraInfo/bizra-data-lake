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

import os
from pathlib import Path
from typing import Final

# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-LOAD .env — Ensures LM Studio token (and all secrets) are available
# to every module that imports from constants.py.
#
# Root cause: .env defines LM_STUDIO_API_KEY but os.getenv() only reads
# shell environment variables, not .env files. This bridge closes the gap.
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from dotenv import load_dotenv

    # Walk up from this file to find the repo root .env
    _constants_dir = Path(__file__).resolve().parent  # core/integration/
    _repo_root = _constants_dir.parent.parent  # BIZRA-DATA-LAKE/
    _env_path = _repo_root / ".env"
    if _env_path.exists():
        load_dotenv(_env_path, override=False)  # Don't clobber existing env vars
except ImportError:
    pass  # dotenv not installed — rely on shell exports

# ═══════════════════════════════════════════════════════════════════════════════
# LM STUDIO API TOKEN UNIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
# The codebase reads 4 different env var names for the same token:
#   LM_API_TOKEN        — node0_activate.py, scripts, nexus, e2e_pipeline
#   LMSTUDIO_API_KEY    — lmstudio_backend.py (fallback 1), Rust CLI
#   LM_STUDIO_API_KEY   — .env file, bizra_cli_bridge.py
#   LM_STUDIO_TOKEN     — sovereign_command.py
#
# This block resolves the canonical token ONCE and propagates it to all names,
# so every consumer finds it regardless of which name they query.
_lm_token = (
    os.getenv("LM_API_TOKEN")
    or os.getenv("LMSTUDIO_API_KEY")
    or os.getenv("LM_STUDIO_API_KEY")
    or os.getenv("LM_STUDIO_TOKEN")
    or ""
)
if _lm_token:
    os.environ.setdefault("LM_API_TOKEN", _lm_token)
    os.environ.setdefault("LMSTUDIO_API_KEY", _lm_token)
    os.environ.setdefault("LM_STUDIO_API_KEY", _lm_token)
    os.environ.setdefault("LM_STUDIO_TOKEN", _lm_token)

# Canonical export for direct import
LM_API_TOKEN: Final[str] = os.getenv("LM_API_TOKEN", "")


def _env_int(name: str, default: int) -> int:
    """Read integer env var with safe fallback."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


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
# 0.35 represents moderate inequality — aligned with Rust bizra-resourcepool
# Transactions that would push Gini above this are REJECTED
ADL_GINI_THRESHOLD: Final[float] = 0.35

# Harberger tax rate (annual, applied continuously)
# Flows to Universal Basic Compute (UBC) pool
# Aligned with Rust bizra-resourcepool HARBERGER_TAX_RATE = 0.07
ADL_HARBERGER_TAX_RATE: Final[float] = 0.07

# Minimum holding to be considered a participant
# Prevents dust attacks and ensures meaningful participation
ADL_MINIMUM_HOLDING: Final[float] = 1e-9

# Minimum non-pool accounts before Gini enforcement activates
# Gini coefficient is statistically meaningless with < 5 data points.
# During genesis bootstrap, the system must distribute to initial participants
# before equality enforcement can meaningfully apply.
# Standing on Giants — Gini (1912): sample-size requirement for robust estimation
ADL_GINI_MIN_ACCOUNTS: Final[int] = 5


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

# Primary LLM backend (env override: LMSTUDIO_URL or LMSTUDIO_HOST + LMSTUDIO_PORT)
# Single source of truth — all core/ modules must import these instead of hardcoding.
LMSTUDIO_HOST: str = os.getenv("LMSTUDIO_HOST", "172.22.48.1")
LMSTUDIO_PORT: str = os.getenv("LMSTUDIO_PORT", "1234")
LMSTUDIO_URL: str = os.getenv("LMSTUDIO_URL", f"http://{LMSTUDIO_HOST}:{LMSTUDIO_PORT}")

# Fallback LLM backend (env override: OLLAMA_URL or OLLAMA_HOST)
OLLAMA_URL = os.getenv("OLLAMA_URL", os.getenv("OLLAMA_HOST", "http://localhost:11434"))

# Model directory (unified path)
MODEL_DIR = os.getenv("BIZRA_MODELS_DIR", "/mnt/c/BIZRA-DATA-LAKE/models")


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
    "ADL_GINI_THRESHOLD": 0.35,  # Justice invariant - anti-plutocracy
}


# ═══════════════════════════════════════════════════════════════════════════════
# TRI-TEMPORAL INTEGRATION (Golden Gem #10)
# ═══════════════════════════════════════════════════════════════════════════════
# Standing on Giants: Friston (Free Energy, 2006) · Boyd (OODA, 1976) · Besta (GoT, 2024)
#
# Definition (Tri-Temporal Integration):
#   T = T1(ms) ⊗ T2(min) ⊗ T3(days)
# where ⊗ denotes temporal coupling:
#   T1 generates observations that T2 interprets
#   T2 generates plans that T1 executes
#   T2 generates skills that T3 consolidates
#   T3 generates priors that bias T2 and calibrate T1
#
# BIZRA is the first system to integrate all three timescales into a closed loop.

# Timescale 1 — Reactive (Cerebellum): HMM Micro-States, AHK, Receipts
TIMESCALE_T1_CYCLE_MS: Final[int] = 50  # Reactive sensorimotor loop
TIMESCALE_T1_PROACTIVE_MS: Final[int] = 5  # Pre-staged via HMM prediction

# Timescale 2 — Deliberative (Prefrontal): GoT Diffusion, PAT agents
TIMESCALE_T2_CYCLE_SECONDS: Final[float] = 5.0  # OODA cycle interval
TIMESCALE_T2_GOT_HYPOTHESES: Final[int] = 3  # Min GoT hypothesis branches

# Timescale 3 — Adaptive (Hippocampal): Federated Memory, PoI, Adl convergence
TIMESCALE_T3_CONSOLIDATION_HOURS: Final[int] = 24  # Skill consolidation window
TIMESCALE_T3_FEDERATION_DAYS: Final[int] = 7  # Cross-node sync period

# ═══════════════════════════════════════════════════════════════════════════════
# SOVEREIGN EMPOWERMENT LOOP (SEL) — Golden Gem #4
# ═══════════════════════════════════════════════════════════════════════════════
# Standing on Giants: Friston (Active Inference) · Deming (PDCA) · Boyd (OODA)
#
# SEL = Perceive(W) → GoT{h1..hk} → diffusion(p*) → Act(p*) → receipt(r)
#       → verify(r) → Learn(r) → federate → Share(r)
#
# The first digital sensorimotor loop — AI that touches physical reality,
# feels the result through verification, and develops procedural memory.

SEL_STAGES: Final[tuple] = (
    "PERCEIVE",  # OS-level awareness (file system, UI state, user context)
    "THINK",  # Diffusion reasoning (parallel GoT hypotheses)
    "PLAN",  # PAT agent team (7 agents, ranked action plans)
    "ACT",  # Desktop automation (real keystrokes, real file moves)
    "SENSE",  # Receipt pipeline (verify what actually happened)
    "LEARN",  # Layer 2 memory (encrypted personal episodic storage)
    "REMEMBER",  # Layer 3 memory (federated skill aggregation)
    "SHARE",  # PoI consensus (network-wide skill propagation)
)

# ═══════════════════════════════════════════════════════════════════════════════
# ACTIVE INFERENCE CONSTANTS — Golden Gem #13
# ═══════════════════════════════════════════════════════════════════════════════
# Standing on Giants: Friston (Free Energy Principle, 2006) · Shannon (1948)
#
# BIZRA is an Active Inference agent at civilization scale.
# Each node minimizes its own free energy (personal prediction accuracy).
# The network minimizes collective free energy (shared knowledge quality).
# The economic system (PoI + Adl) maintains allostatic balance.
# The Three Facts are the prior beliefs constraining all inference.

# SAT federation sizing for FRONTIER governance routing.
# SAT-5 is per-node. Federation-wide quorum scales with node count.
SAT_VALIDATORS_PER_NODE: Final[int] = 5
FEDERATION_NODE_COUNT_DEFAULT: Final[int] = _env_int("BIZRA_FEDERATION_NODE_COUNT", 10)


def sat_frontier_quorum(
    federation_nodes: int,
    sat_validators_per_node: int = SAT_VALIDATORS_PER_NODE,
) -> int:
    """Compute BFT quorum across federated SAT validators (2f+1)."""
    nodes = federation_nodes if federation_nodes > 0 else 1
    validators_per_node = (
        sat_validators_per_node
        if sat_validators_per_node > 0
        else SAT_VALIDATORS_PER_NODE
    )
    total_sat_validators = nodes * validators_per_node
    faulty = (total_sat_validators - 1) // 3
    return max(1, (2 * faulty) + 1)


SAT_FRONTIER_QUORUM_DEFAULT: Final[int] = sat_frontier_quorum(
    FEDERATION_NODE_COUNT_DEFAULT
)

# Prediction-Verification Duality (Golden Gem #14)
# System acts only when prediction (HMM) and verification (FATE) agree.
# Prediction: P(o_{t+1} | o_{1:t}) — "what WILL happen?"
# Verification: P(safe(a) | C) — "what SHOULD happen?"
PREDICTION_VERIFICATION_AGREEMENT_THRESHOLD: Final[float] = 0.90

# Takaful Bootstrap Protocol (Golden Gem #11)
# New nodes inherit collective intelligence from behaviorally similar peers.
# KL-divergence between HMM transition matrices groups similar users.
TAKAFUL_BOOTSTRAP_OBSERVATION_MINUTES: Final[int] = 10
TAKAFUL_KL_DIVERGENCE_THRESHOLD: Final[float] = 0.5

# Constitutional Immune System (Golden Gem #12)
# HMM anomaly detection = runtime enforcement of Three Invariants
ANOMALY_LOG_LIKELIHOOD_THRESHOLD: Final[float] = -3.0  # Standard deviations

# ═══════════════════════════════════════════════════════════════════════════════
# THREE KERNEL INVARIANTS — Immutable Constitutional Axioms
# ═══════════════════════════════════════════════════════════════════════════════
# Standing on Giants: Al-Ghazali (Ihsan, 1095) · Shannon (1948) · البذرة (2023)

KERNEL_INVARIANTS: Final[tuple] = (
    "RIBA_ZERO",  # No exploitation. No interest. No harm.
    "CLAIM_MUST_BIND",  # No hallucination. Every claim has evidence. (ZANN_ZERO)
    "IHSAN_FLOOR",  # Excellence is the minimum. 0.99 threshold.
)

# ═══════════════════════════════════════════════════════════════════════════════
# DUAL-TOKEN ECONOMY — Golden Gem #3
# ═══════════════════════════════════════════════════════════════════════════════
# SEED = معاملات (muamalat) — worldly transactions, pegged to compute hours
# BLOOM = عبادة (ibadah) — worship/service, minted from verified impact
SEED_COMPUTE_HOUR_PEG: Final[float] = 1.0  # 1 SEED = 1 compute hour
BLOOM_REDISTRIBUTION_RATE: Final[float] = 0.50  # 50% — thermodynamic necessity

# ═══════════════════════════════════════════════════════════════════════════════
# HASH TABLE INFRASTRUCTURE — Phase 44
# Standing on Giants: Bloom (1970), Merkle (1979), Kirsch & Mitzenmacher (2006)
# ═══════════════════════════════════════════════════════════════════════════════
BLOOM_DEFAULT_FPR: Final[float] = 0.01
BLOOM_MAX_BITS: Final[int] = 10_000_000  # ~1.2 MB cap
MERKLE_LEAF_PREFIX: Final[bytes] = b"\x00"  # RFC 6962 domain separation
MERKLE_NODE_PREFIX: Final[bytes] = b"\x01"
SKILL_CACHE_MAX_SIZE: Final[int] = 256
SKILL_CACHE_DEFAULT_TTL: Final[int] = 3600  # seconds

# ═══════════════════════════════════════════════════════════════════════════════
# COGNITIVE RESONANCE — Phase 46
# Standing on Giants: Shannon (1948), Johnson/FAISS (2021), Rabiner (1989)
# ═══════════════════════════════════════════════════════════════════════════════
FAISS_INDEX_PATH: Final[str] = "04_GOLD/node0_faiss.index"
FAISS_META_PATH: Final[str] = "04_GOLD/node0_faiss_meta.json"
FAISS_GOLD_DIR: Final[str] = "04_GOLD"
FAISS_EMBEDDING_DIM: Final[int] = 384
FAISS_DEFAULT_TOP_K: Final[int] = 10
FAISS_SIMILARITY_FLOOR: Final[float] = 0.35
HMM_NUM_HIDDEN_STATES: Final[int] = 6
HMM_OBSERVATION_WINDOW: Final[int] = 50
HMM_CONVERGENCE_THRESHOLD: Final[float] = 1e-4
HMM_MAX_EM_ITERATIONS: Final[int] = 100
GOT_MAX_HYPOTHESES: Final[int] = 5
GOT_CONVERGENCE_SNR: Final[float] = 0.90
GOT_MAX_DEPTH: Final[int] = 4

# ═══════════════════════════════════════════════════════════════════════════════
# SAFE ACTIVATION — Phase 47.1
# Standing on Giants: Fowler (canary, 2010), Nygard (Release It!, 2007)
# ═══════════════════════════════════════════════════════════════════════════════
CANARY_PERCENT_MIN: Final[int] = 0
CANARY_PERCENT_MAX: Final[int] = 100
CANARY_DEFAULT_SALT: Final[str] = "bizra-phase46-canary-v1"
HMM_CALLER_MODE_DEFAULT: Final[str] = "single"
HMM_ALLOWED_CALLER_DEFAULT: Final[str] = "mcp"
ROLLBACK_CONSECUTIVE_BREACHES: Final[int] = 2
ROLLBACK_SEARCH_ERROR_THRESHOLD: Final[float] = 0.02  # 2%
ROLLBACK_GOT_FALLBACK_THRESHOLD: Final[float] = 0.20  # 20%
ROLLBACK_HMM_CONFIDENCE_FLOOR: Final[float] = 0.55
ROLLBACK_SNR_DROP_THRESHOLD: Final[float] = 0.15  # 15% drop from baseline
ROLLBACK_LATENCY_DELTA_THRESHOLD: Final[float] = 0.30  # 30% p95 regression


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
            "drift_count": drift_count,  # type: ignore[dict-item]
            "path": path,
        }

    return results
