"""
BIZRA Integration Constants — AUTHORITATIVE SOURCE OF TRUTH

╔══════════════════════════════════════════════════════════════════════════════╗
║   ALL MODULES MUST IMPORT THRESHOLDS FROM THIS FILE                          ║
║   Do NOT define IHSAN_THRESHOLD or SNR_THRESHOLD elsewhere.                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Genesis Strict Synthesis v3.0.0 — Cross-Repository Constants

Unified constants across all core modules to ensure consistency.
These values override module-specific constants when using the
IntegrationBridge.

Sovereignty: Single source of truth for quality thresholds.

v3.0.0 Convergence (Constitution v5.0.0-GENESIS):
- Absorbs 30+ new constitutional constants from bizra-constitution/
- Ihsan operational thresholds: 0.85 (gate), 0.90 (bloom), 0.95 (excellence)
- Ihsan constitutional tensor: 8-dim canonical, 6-dim operational projection
- SNR: 0.85 (minimum/museum floor), 0.95 (T1), 0.98 (T0/elite)
- Gate configuration: 5 alpha gates, fail-closed
- HHMM: 47-state taxonomy (5 initial live), 4 complexity tiers
- Reflex cache: precipitation at 3 consecutive hits with Ihsan >= 0.90
- Action bus: 100ms GCD tick, 10 concurrent, 100/hour rate limit

Legacy Operational Values (v2.2.2):
- IHSAN: 0.95 (standard), 0.99 (strict/consensus), 1.0 (runtime/Z3-proven)
- ADL_GINI_THRESHOLD: 0.35 (operational, pending constitutional review → 0.45)

Cross-repo alignment:
- BIZRA-DATA-LAKE: core/integration/constants.py (this file)
- BIZRA-Dual-Agentic-system: core/constants.py
- bizra-omega (Rust): bizra-core/src/lib.rs
- bizra-constitution/generated: generated_constants.py (constitutional source)
- TypeScript: src/core/sovereign/capability-card.ts

Standing on Giants: Shannon • Lamport • Vaswani • Anthropic • Al-Ghazali
"""

import os
from pathlib import Path
from typing import Dict, Final

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
# IHSĀN DIMENSION WEIGHTS — Legacy Operational (v2.x)
# ═══════════════════════════════════════════════════════════════════════════════
# 8-dimensional ethical scoring (must sum to 1.0)
# Used by: core/proof_engine/ihsan_gate.py, core/constitutional/omega_engine.py

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
# IHSĀN CONSTITUTIONAL TENSOR — v5.0.0-GENESIS (8-dim canonical)
# ═══════════════════════════════════════════════════════════════════════════════
# Standing on Giants: Al-Ghazali (Ihsan, 1095) · Shannon (information entropy)
# Source: constitution.toml §3 [ihsan_tensor.weights]
# Migration: constitutional tensor supersedes legacy IHSAN_WEIGHTS over time.
# Used by: bizra-constitution/ihsan_gate.py (6-dim operational projection)

IHSAN_CANONICAL_WEIGHTS: Final[dict] = {
    "moral_clarity": 0.1200,  # وضوح أخلاقي — ethical transparency
    "epistemic_humility": 0.1400,  # تواضع معرفي — knowing what you don't know
    "structural_integrity": 0.1300,  # سلامة بنيوية — coherent architecture
    "verifiability": 0.1300,  # قابلية التحقق — provable claims
    "contextual_relevance": 0.1100,  # ملاءمة سياقية — right answer, right time
    "intent_alignment": 0.1400,  # توافق النية — serves the user's true need
    "resilience": 0.1100,  # مرونة — graceful under failure
    "efficiency": 0.1200,  # كفاءة — minimum waste, maximum signal
}

# 6-dim operational projection (excludes contextual_relevance + efficiency, renormalized)
# This is the scoring tensor used by the constitutional IhsanGate at runtime.
IHSAN_OPERATIONAL_WEIGHTS: Final[dict] = {
    "moral_clarity": 0.1558,
    "epistemic_humility": 0.1818,
    "structural_integrity": 0.1688,
    "verifiability": 0.1688,
    "intent_alignment": 0.1818,
    "resilience": 0.1429,
}

IHSAN_DIMENSIONS_CANONICAL: Final[int] = 8
IHSAN_DIMENSIONS_OPERATIONAL: Final[int] = 6
IHSAN_OPERATIONAL_NAMES: Final[list] = [
    "moral_clarity",
    "epistemic_humility",
    "structural_integrity",
    "verifiability",
    "intent_alignment",
    "resilience",
]

# ═══════════════════════════════════════════════════════════════════════════════
# IHSĀN CONSTITUTIONAL THRESHOLDS — v5.0.0-GENESIS
# ═══════════════════════════════════════════════════════════════════════════════
# Source: constitution.toml §3 [ihsan_tensor.thresholds]

IHSAN_GATE_MINIMUM: Final[float] = 0.85  # Hard floor — fail-closed below this
IHSAN_POI_CONSENSUS: Final[float] = 0.85  # PoI attestation minimum
IHSAN_BLOOM_ELIGIBILITY: Final[float] = 0.90  # BLOOM token minting threshold
IHSAN_CONFORMANCE_JOIN: Final[float] = 0.95  # Network join conformance

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
# Aligned with Rust bizra-resourcepool HARBERGER_TAX_RATE = 0.05
# Constitutional: 5% annual — discourages idle hoarding, not punitive
ADL_HARBERGER_TAX_RATE: Final[float] = 0.05

# Emergency Gini threshold — system-wide freeze if exceeded
# Aligned with Rust bizra-core/omega.rs ADL_GINI_EMERGENCY = 0.60
ADL_GINI_EMERGENCY: Final[float] = 0.60

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
#
# WSL2 gateway IP changes between reboots — auto-detect unless env override is set.
# Standing on Giants: Boyd (OODA observe — sense the real environment, not a cached one)


def _detect_wsl_gateway() -> str:
    """Auto-detect the Windows host IP from WSL2.

    Priority: LMSTUDIO_HOST env var > `ip route` gateway > hardcoded fallback.
    This runs ONCE at import time. Fast (~2ms subprocess).
    """
    env_host = os.getenv("LMSTUDIO_HOST")
    if env_host:
        return env_host

    try:
        import subprocess

        result = subprocess.run(
            ["ip", "route", "show", "default"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0 and "via" in result.stdout:
            # "default via 172.22.48.1 dev eth0" → extract the IP
            return result.stdout.split("via")[1].strip().split()[0]
    except Exception:
        pass

    # Fallback: most recent known WSL gateway
    return "172.22.48.1"


LMSTUDIO_HOST: str = _detect_wsl_gateway()
LMSTUDIO_PORT: str = os.getenv("LMSTUDIO_PORT", "1234")
LMSTUDIO_URL: str = os.getenv("LMSTUDIO_URL", f"http://{LMSTUDIO_HOST}:{LMSTUDIO_PORT}")

# Fallback LLM backend (env override: OLLAMA_URL or OLLAMA_HOST)
OLLAMA_URL = os.getenv("OLLAMA_URL", os.getenv("OLLAMA_HOST", "http://localhost:11434"))

# ═══════════════════════════════════════════════════════════════════════════════
# NODE0 MODEL FLEET — Agent-to-Model Routing (§1 The Living Organism)
# ═══════════════════════════════════════════════════════════════════════════════
# 12 agents per node: 7 PAT (user's personal team) + 5 SAT (system gates).
#
# 7 PAT — Personal Agentic Team (owned by user, loyal to user):
#   P1 Planner    — Strategic decomposition, goal breakdown
#   P2 Researcher — Knowledge retrieval, domain learning
#   P3 Coder      — Executable actions (Telescript generation)
#   P4 Evaluator  — Testing, simulation, outcome scoring
#   P5 Ethicist   — Ihsan scoring, constitutional alignment (frozen gate)
#   P6 Publisher   — Communication, formatting, user-facing output
#   P7 Integrator (DEMA) — Synthesis, team coordination, voice persona
#
# 5 SAT — System Agentic Team (owned by BIZRA URP, system-wide):
#   S1 Sentinel   — Real-time threat detection (pure-code, no LLM)
#   S2 Oracle     — Constitutional reasoning, Shura consensus
#   S3 Ledger     — Evidence chain, proof-carrying inference (pure-code)
#   S4 Conductor  — Event bus routing, agent orchestration (pure-code)
#   S5 Ambassador — Federation gossip, inter-node protocol (pure-code)
#
# Ollama defaults (always available). When LM Studio is reachable,
# load_fleet_from_yaml() overrides with config/local_models.yaml IDs.
# Vision and embedding are shared capabilities, not agents — routed via task type.
_OLLAMA_FLEET_DEFAULTS: Dict[str, str] = {
    # 7 PAT agents
    "P1-Planner": "deepseek-r1:14b",  # Deep reasoning for strategic planning
    "P2-Researcher": "qwen2.5:3b",  # Knowledge + multilingual
    "P3-Coder": "mistral:latest",  # 7B code generation (LM Studio: agentflow-7b)
    "P4-Evaluator": "phi3:mini",  # Fast evaluation + scoring
    "P5-Ethicist": "frozen",  # Constitutional gate — no LLM, pure Ihsan logic
    "P6-Publisher": "phi3:mini",  # Communication formatting
    "P7-DEMA": "deephat-v1-7b",  # Integrator + voice (NVIDIA PersonaPlex 7B)
    # 5 SAT agents
    "S1-Sentinel": "pure-code",  # Threat detection — no LLM needed
    "S2-Oracle": "phi3:mini",  # Constitutional reasoning (lightweight)
    "S3-Ledger": "pure-code",  # Evidence chain — no LLM needed
    "S4-Conductor": "pure-code",  # Event routing — no LLM needed
    "S5-Ambassador": "pure-code",  # Federation — no LLM needed
    # Shared capabilities (not agents — routed by task type)
    "vision": "moondream:1.8b",  # Visual analysis capability
    "embedding": "nomic-embed-text:latest",  # Vector embedding capability
    "default": "phi3:mini",  # Fallback for unrouted queries
}

# Agent ID → local_models.yaml lookup key.
# Checked against pat_agents first, then models[] directly.
PAT_ROLE_MAP: Dict[str, str] = {
    "P1-Planner": "strategist",  # pat_agents → planning (agentflow 7B)
    "P2-Researcher": "researcher",  # pat_agents → reasoning (deepseek 8B)
    "P3-Coder": "planning",  # pat_agents → planning (agentflow 7B)
    "P4-Evaluator": "analyst",  # pat_agents → reasoning (deepseek 8B)
    "P5-Ethicist": "frozen",  # No YAML lookup — constitutional gate
    "P6-Publisher": "coordinator",  # pat_agents → planning (agentflow 7B)
    "P7-DEMA": "voice",  # personaplex/engine.py — nvidia/personaplex-7b-v1
    "S2-Oracle": "guardian",  # pat_agents → reasoning (deepseek 8B)
    "vision": "vision_large",  # models[] direct — qwen VL 8B
    "embedding": "embedding",  # models[] direct — nomic embed
    "default": "fast",  # models[] direct — liquid 1.2B
}

# Env-var overrides applied on top of the loaded fleet
_ENV_OVERRIDES: Dict[str, str] = {
    "P1-Planner": "BIZRA_MODEL_PLANNER",
    "P2-Researcher": "BIZRA_MODEL_RESEARCHER",
    "P3-Coder": "BIZRA_MODEL_CODER",
    "P4-Evaluator": "BIZRA_MODEL_EVALUATOR",
    "P6-Publisher": "BIZRA_MODEL_PUBLISHER",
    "P7-DEMA": "BIZRA_MODEL_DEMA",
    "S2-Oracle": "BIZRA_MODEL_ORACLE",
    "vision": "BIZRA_MODEL_VISION",
    "embedding": "BIZRA_MODEL_EMBED",
    "default": "BIZRA_MODEL_DEFAULT",
}


def load_fleet_from_yaml(
    yaml_path: str = "",
) -> Dict[str, str]:
    """Build NODE0_MODEL_FLEET, preferring config/local_models.yaml when present.

    Resolution order per agent:
        1. Environment variable (BIZRA_MODEL_*)  — highest priority
        2. config/local_models.yaml (LM Studio)  — if file exists
        3. Ollama defaults                        — always available
    """
    fleet: Dict[str, str] = dict(_OLLAMA_FLEET_DEFAULTS)

    # Try loading YAML config
    if not yaml_path:
        yaml_path = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "config",
            "local_models.yaml",
        )
    try:
        import yaml  # type: ignore[import-untyped]

        with open(yaml_path, "r") as fh:
            cfg = yaml.safe_load(fh) or {}
        models = cfg.get("models", {})
        pat_agents = cfg.get("pat_agents", {})

        for agent_id, role in PAT_ROLE_MAP.items():
            # First check pat_agents mapping (role → model key)
            model_key = pat_agents.get(role, role)
            model_def = models.get(model_key, {})
            if isinstance(model_def, dict) and "id" in model_def:
                fleet[agent_id] = model_def["id"]
            elif isinstance(model_def, str) and model_def in models:
                # pat_agents value is a model key reference
                resolved = models[model_def]
                if isinstance(resolved, dict) and "id" in resolved:
                    fleet[agent_id] = resolved["id"]
    except FileNotFoundError:
        pass  # No YAML config — Ollama defaults used
    except Exception:  # noqa: BLE001
        pass  # Malformed YAML — fall back safely to Ollama defaults

    # Env overrides always win
    for agent_id, env_key in _ENV_OVERRIDES.items():
        val = os.getenv(env_key)
        if val:
            fleet[agent_id] = val

    return fleet


NODE0_MODEL_FLEET: Dict[str, str] = load_fleet_from_yaml()

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
    "bizra-constitution": "/mnt/c/BIZRA-DATA-LAKE/bizra-constitution/generated/generated_constants.py",
}

# Canonical threshold values for cross-repo validation (numeric only)
CANONICAL_THRESHOLDS = {
    "IHSAN_THRESHOLD": 0.95,
    "SNR_THRESHOLD_MINIMUM": 0.85,
    "SNR_THRESHOLD_T0_ELITE": 0.98,
    "MUSEUM_SNR_FLOOR": 0.85,
    "RUNTIME_IHSAN": 1.0,
    "ADL_GINI_THRESHOLD": 0.35,  # Operational — constitutional target is 0.45
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
HMM_NUM_HIDDEN_STATES: Final[int] = 6  # Operational (legacy Phase 46)
HMM_OBSERVATION_WINDOW: Final[int] = 50
HMM_CONVERGENCE_THRESHOLD: Final[float] = 1e-4
HMM_MAX_EM_ITERATIONS: Final[int] = 100
GOT_MAX_HYPOTHESES: Final[int] = 5
GOT_CONVERGENCE_SNR: Final[float] = 0.90
GOT_MAX_DEPTH: Final[int] = 4

# HHMM Constitutional Taxonomy — v5.0.0-GENESIS
# Source: constitution.toml §7 [hhmm]
# 47 states is the full taxonomy; 5 initial live at genesis.
# HMM_NUM_HIDDEN_STATES (6) is the Phase 46 operational count.
HMM_FULL_TAXONOMY_STATES: Final[int] = 47
HMM_INITIAL_LIVE_STATES: Final[int] = 5
HMM_EXPANSION_TRIGGER: Final[int] = 1000  # Missions before state expansion

# ═══════════════════════════════════════════════════════════════════════════════
# COMPLEXITY TIER BUDGETS — v5.0.0-GENESIS
# ═══════════════════════════════════════════════════════════════════════════════
# Source: constitution.toml §7 [hhmm.tiers]
# Latency budget per complexity tier (HHMM classification output)

TIER_TRIVIAL_BUDGET_MS: Final[int] = 100  # Reflex cache hit (S1)
TIER_SIMPLE_BUDGET_MS: Final[int] = 3000  # Single agent pipeline
TIER_COMPLEX_BUDGET_MS: Final[int] = 15000  # Mission orchestrator (full PAT)
TIER_SOVEREIGN_BUDGET_MS: Final[int] = 60000  # Multi-model sovereign pipeline

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTITUTIONAL GATE CONFIGURATION — v5.0.0-GENESIS
# ═══════════════════════════════════════════════════════════════════════════════
# Source: constitution.toml §5 [gates]
# 5 alpha gates with weights summing to 1.0. Fail mode: closed.

GATE_WEIGHTS: Final[dict] = {
    "alpha_4": 0.15,  # Fallback gate
    "alpha_7": 0.25,  # Verification gate
    "alpha_8": 0.20,  # Dark matter gate
    "alpha_9": 0.25,  # Attestation gate
    "alpha_10": 0.15,  # Binary gate
}
GATE_FAIL_MODE: Final[str] = "closed"
GATE_OVERHEAD_BUDGET_MS: Final[int] = 50

# ═══════════════════════════════════════════════════════════════════════════════
# MOE ENGINE — Phase 68.07
# ═══════════════════════════════════════════════════════════════════════════════
# Source: docs/specs/phase_68_bus_architecture/phase_68_07_moe_engine.md
# Standing on: Shazeer (2017) sparsely-gated MOE, top-K routing

MOE_EXPERT_COUNT: Final[int] = 5
MOE_TOP_K: Final[int] = 2  # experts activated per query
MOE_FALLBACK_EXPERT: Final[str] = "pat_r"  # reasoning is default
MOE_MIN_CONFIDENCE: Final[float] = 0.1  # below this, expert is skipped
MOE_SYNTHESIS_STRATEGY: Final[str] = "weighted"  # "weighted" | "best_of"

# ═══════════════════════════════════════════════════════════════════════════════
# REFLEX CACHE — v5.0.0-GENESIS
# ═══════════════════════════════════════════════════════════════════════════════
# Source: constitution.toml §9 [reflex]
# O(1) HashMap cache with precipitation model (Theorem 2.2)

REFLEX_STORE_TYPE: Final[str] = "HashMap"
REFLEX_MAX_ENTRIES: Final[int] = 500
REFLEX_PRECIPITATION_HITS: Final[int] = 3  # Consecutive high-quality hits
REFLEX_PRECIPITATION_IHSAN: Final[float] = 0.90  # Minimum Ihsan for precipitation
REFLEX_SIMILARITY_THRESHOLD: Final[float] = 0.95  # Template matching threshold
REFLEX_INVALIDATION_INTERVAL: Final[int] = 100  # Hits between validation checks
REFLEX_INVALIDATION_DELTA: Final[float] = 0.05  # Max Ihsan drift before invalidation
REFLEX_STALENESS_DAYS: Final[int] = 30  # Force invalidation after N days

# ═══════════════════════════════════════════════════════════════════════════════
# ACTION BUS — v5.0.0-GENESIS
# ═══════════════════════════════════════════════════════════════════════════════
# Source: constitution.toml §10 [action_bus]

ACTION_BUS_GCD_TICK_MS: Final[int] = 100  # Greatest common divisor tick
ACTION_BUS_MAX_CONCURRENT: Final[int] = 10  # Max parallel missions
ACTION_BUS_MAX_PER_HOUR: Final[int] = 100  # Hourly rate limit

# ═══════════════════════════════════════════════════════════════════════════════
# PAT/SAT AGENT CONFIGURATION — v5.0.0-GENESIS
# ═══════════════════════════════════════════════════════════════════════════════
# Source: constitution.toml §4 [pat], §6 [sat]

PAT_AGENT_COUNT: Final[int] = 7
PAT_AGENT_NAMES: Final[list] = [
    "Planner",
    "Researcher",
    "Coder",
    "Evaluator",
    "Ethicist",
    "Publisher",
    "Integrator",
]
PAT_TRUST_STAGES: Final[list] = [
    "abstracting",
    "gathering",
    "executing",
    "attesting",
    "certifying",
    "publishing",
    "chaining",
]

SAT_AGENTS_PER_NODE: Final[int] = 5
SAT_BOOTSTRAP_ROLES: Final[list] = [
    "ComputeScheduler",
    "SecurityMonitor",
    "PerformanceAnalyzer",
    "ConsensusValidator",
    "NetworkOrchestrator",
]
SAT_INFRASTRUCTURE_FLOOR_PCT: Final[int] = 20  # Minimum % devoted to infra
SAT_REBALANCE_INTERVAL_S: Final[int] = 300
SAT_SERVICE_TYPES: Final[list] = [
    "ComputeAllocation",
    "NetworkRoute",
    "ConsensusVerification",
    "SecurityCheck",
    "TemplatePublish",
    "EconomicSettlement",
]
IDENTITY_AGENTS_PER_NODE: Final[int] = 12  # PAT(7) + SAT(5)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTITUTIONAL ECONOMICS — v5.0.0-GENESIS
# ═══════════════════════════════════════════════════════════════════════════════
# Source: constitution.toml §8 [economics]

SEED_YEARLY_CAP: Final[int] = 1_000_000
ZAKAT_RATE: Final[float] = 0.025  # 2.5% — applied at mint time
NISAB_THRESHOLD: Final[float] = 85.0  # Minimum SEED balance for Zakat obligation
NO_RIBA: Final[bool] = True  # Kernel invariant: zero exploitation
NO_GHARAR: Final[bool] = True  # Kernel invariant: zero deception

# Constitutional Gini threshold (0.45) vs operational (0.35) — pending governance review.
# constitution.toml §8 sets 0.45; existing code enforces 0.35.
# ADL_GINI_THRESHOLD remains 0.35 (operational) until cross-repo alignment completes.
CONSTITUTIONAL_GINI_THRESHOLD: Final[float] = 0.45
GINI_MEASUREMENT_INTERVAL_S: Final[int] = 3600

# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN SEPARATION STRINGS — v5.0.0-GENESIS
# ═══════════════════════════════════════════════════════════════════════════════
# Source: constitution.toml §2 [identity.domain_separation]

DOMAIN_EVIDENCE_RECEIPT: Final[str] = "bizra-evidence-v1"
DOMAIN_URP_LEASE: Final[str] = "bizra-urp-lease-v1"
DOMAIN_POI_ATTESTATION: Final[str] = "bizra-poi-v1"
DOMAIN_IDENTITY_GENESIS: Final[str] = "bizra-identity-genesis-v1"
DOMAIN_TELESCRIPT_PUBLISH: Final[str] = "bizra-telescript-v1"
DOMAIN_BLOOM_MINT: Final[str] = "bizra-bloom-mint-v1"

# ═══════════════════════════════════════════════════════════════════════════════
# CONFORMANCE TARGETS — v5.0.0-GENESIS
# ═══════════════════════════════════════════════════════════════════════════════
# Source: constitution.toml §11 [conformance]

CONFORMANCE_HHMM_ACCURACY: Final[float] = 1.0
CONFORMANCE_POI_VARIANCE: Final[float] = 0.01
CONFORMANCE_CROWN_ENTROPY: Final[float] = 0.95
CONFORMANCE_REFLEX_SEMANTIC: Final[float] = 0.90
CONFORMANCE_POOL_LATENCY_MS: Final[int] = 200

# ═══════════════════════════════════════════════════════════════════════════════
# PRIVACY — v5.0.0-GENESIS
# ═══════════════════════════════════════════════════════════════════════════════
# Source: constitution.toml §12 [privacy]

PRIVACY_CLASSES: Final[list] = ["LOCAL_ONLY", "ABSTRACT_OK", "SHAREABLE"]
PRIVACY_DEFAULT: Final[str] = "LOCAL_ONLY"

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 67 — SOVEREIGN INSTANTIATION CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
# Standing on Giants: Al-Khwarizmi (780-850), Ibn Khaldun (1332-1406), Al-Ghazali (1058-1111)

# Fixed-point precision (6 decimal places, integer-only arithmetic)
FP_PRECISION: Final[int] = 1_000_000

# Al-Ghazali intent pre-gate: receipts below this are discarded before scoring
INTENT_FLOOR: Final[float] = 0.90

# Khaldunian Curve thresholds (progressive minting throttle)
GINI_HEALTHY: Final[float] = 0.30  # Full minting zone
GINI_WARNING: Final[float] = 0.50  # Throttle zone boundary
GINI_CRISIS: Final[float] = 0.70  # Crisis zone (1% minting)

# Demurrage rate on idle balances (per tick)
DEMURRAGE_RATE: Final[float] = 0.001

# BLOOM governance token decay rate per tick
BLOOM_DECAY: Final[float] = 0.01

# Reflex cache TTL (24 hours in seconds)
REFLEX_TTL: Final[int] = 86400

# Ghazali Equity Factor bounds (newcomer advantage multiplier)
EQUITY_FACTOR_MIN: Final[float] = 1.0
EQUITY_FACTOR_MAX: Final[float] = 5.0

# Asabiyyah social cohesion weights (attestations, votes, cooperation)
ASABIYYAH_WEIGHTS: Final[tuple] = (0.4, 0.3, 0.3)

# Asabiyyah-Gini coupling: minting multiplier range based on social cohesion
# Phase 69 Sprint 1 — closes the Khaldunian feedback loop
# Low asabiyyah (fragmented network) → throttle DOWN to 0.80x minting
# High asabiyyah (cohesive network) → boost UP to 1.20x minting
# Neutral point: asabiyyah = 0.50 → multiplier = 1.00x (no effect)
ASABIYYAH_COUPLING_FLOOR: Final[float] = 0.80
ASABIYYAH_COUPLING_CEIL: Final[float] = 1.20
ASABIYYAH_NEUTRAL: Final[float] = 0.50

# Constitution reference
CONSTITUTION_VERSION: Final[str] = "5.0.0-GENESIS"

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


# ═══════════════════════════════════════════════════════════════════════════════
# SEED ENGINE THRESHOLDS — Phase 72
# ═══════════════════════════════════════════════════════════════════════════════
# Standing on Giants: Deming (PDCA) · Kahneman (System 1/2) · Shannon (SNR)
#
# Three distinct gates in the growth pipeline:
#   1. Constitutional acceptance (I-1): UNIFIED_IHSAN_THRESHOLD (0.95)
#   2. Episode qualification: SNR + Ihsan + reward composite
#   3. Reward minimum: composite reward floor for meaningful growth

SEED_REWARD_QUALIFICATION: Final[float] = 0.75
SEED_QUALIFICATION_RATE_VERIFIER: Final[float] = 0.75
SEED_QUALIFICATION_RATE_APPRENTICE: Final[float] = 0.50

# ═══════════════════════════════════════════════════════════════════════════════
# HUMAN LIFECYCLE STAGES — Phase 72
# ═══════════════════════════════════════════════════════════════════════════════
# Standing on Giants: Maslow (1943) · Kohlberg (1958) · Al-Ghazali (1095)
#
# Seven stages of human growth, parallel to agent skill tree.
# Both earned through verified work. Both gated by quality.

HUMAN_STAGE_THRESHOLDS: Final[dict] = {
    "Seed": 0.00,
    "Node": 0.10,
    "Apprentice": 0.20,
    "Builder": 0.35,
    "Verifier": 0.55,
    "Mentor": 0.70,
    "Catalyst": 0.85,
}

HUMAN_STAGE_ORDER: Final[list] = [
    "Seed",
    "Node",
    "Apprentice",
    "Builder",
    "Verifier",
    "Mentor",
    "Catalyst",
]

# ═══════════════════════════════════════════════════════════════════════════════
# NODE VALUE NORMALIZATION — Phase 72
# ═══════════════════════════════════════════════════════════════════════════════
# Standing on Giants: Shannon (bounded information) · Deming (SPC control limits)
#
# Each factor normalized [0, 1]. Composite = geometric mean.

NODE_VALUE_ACTIVATION_REFERENCE: Final[float] = 5.0
NODE_VALUE_COMPOUNDING_REFERENCE_DAYS: Final[int] = 365
NODE_VALUE_STREAK_REFERENCE: Final[int] = 10


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
