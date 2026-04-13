"""
PAT-7 Model Routing — Maps agent roles to local Ollama models.

Canonical routing table for NODE0. Each PAT role gets the best-fit
model based on the role's requirements and VRAM constraints.

Models are loaded on-demand by Ollama (keep_alive=5m).
Only one 26b model fits in VRAM at a time alongside desktop overhead.

Standing on Giants:
- Shannon (1948): Match channel capacity to signal requirements
- BIZRA llm-stack.md: Empirical benchmarks from NODE0 bring-up session
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class PatRole(Enum):
    """PAT-7 agent roles."""

    STRATEGIST = "strategist"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    CREATOR = "creator"
    EXECUTOR = "executor"
    GUARDIAN = "guardian"
    COORDINATOR = "coordinator"


class SatRole(Enum):
    """SAT-5 governance roles."""

    SENTINEL = "sentinel"        # Layer 1: Structural integrity
    ORACLE_S = "oracle_s"        # Layer 2: Constitutional compliance
    LEDGER = "ledger"            # Layer 3: Economic soundness
    CONDUCTOR = "conductor"      # Layer 4: Operational readiness
    AMBASSADOR = "ambassador"    # Layer 5: Human verification


@dataclass
class ModelAssignment:
    """Model assignment for a role."""

    model: str
    tier: str          # "fast" (e4b/7b) | "deep" (26b) | "embed"
    tok_s: float       # Measured throughput on NODE0
    reason: str        # Why this model for this role


# ── PAT-7 Routing Table ──────────────────────────────────────────
# Configurable via env vars: BIZRA_PAT_<ROLE>_MODEL

_PAT_DEFAULTS: Dict[PatRole, ModelAssignment] = {
    PatRole.STRATEGIST: ModelAssignment(
        model="gemma4:26b-bizra-16k",
        tier="deep",
        tok_s=19.83,
        reason="Strategic planning needs deep reasoning + 16K context",
    ),
    PatRole.RESEARCHER: ModelAssignment(
        model="gemma4:e4b",
        tier="fast",
        tok_s=61.0,
        reason="Evidence gathering prioritizes speed over depth",
    ),
    PatRole.ANALYST: ModelAssignment(
        model="qwen2.5-coder:14b",
        tier="fast",
        tok_s=28.5,
        reason="Code/data analysis needs coder-optimized model",
    ),
    PatRole.CREATOR: ModelAssignment(
        model="gemma4:e4b",
        tier="fast",
        tok_s=61.0,
        reason="Content generation at speed, quality gated by SAT",
    ),
    PatRole.EXECUTOR: ModelAssignment(
        model="deepseek-r1:7b",
        tier="fast",
        tok_s=54.0,
        reason="Task execution needs reasoning chain (R1 architecture)",
    ),
    PatRole.GUARDIAN: ModelAssignment(
        model="gemma4:26b-bizra-16k",
        tier="deep",
        tok_s=19.83,
        reason="Guardian needs deepest model for safety evaluation",
    ),
    PatRole.COORDINATOR: ModelAssignment(
        model="gemma4:26b-bizra-16k",
        tier="deep",
        tok_s=19.83,
        reason="Coordination needs full context window for multi-agent synthesis",
    ),
}

# ── SAT-5 Routing Table ──────────────────────────────────────────
# All SAT roles use the canonical governance-lane model

_SAT_DEFAULTS: Dict[SatRole, ModelAssignment] = {
    role: ModelAssignment(
        model="gemma4:26b-bizra-16k",
        tier="deep",
        tok_s=19.83,
        reason="All governance verdicts require the canonical governance model",
    )
    for role in SatRole
}

# Embedding model (used for RAG retrieval, not generation)
EMBEDDING_MODEL = "nomic-embed-text"


def get_pat_model(role: PatRole) -> str:
    """Get the model assigned to a PAT role. Env var override supported."""
    env_key = f"BIZRA_PAT_{role.value.upper()}_MODEL"
    override = os.getenv(env_key)
    if override:
        return override
    return _PAT_DEFAULTS[role].model


def get_sat_model(role: SatRole) -> str:
    """Get the model assigned to a SAT role. Env var override supported."""
    env_key = f"BIZRA_SAT_{role.value.upper()}_MODEL"
    override = os.getenv(env_key)
    if override:
        return override
    return _SAT_DEFAULTS[role].model


def get_pat_assignment(role: PatRole) -> ModelAssignment:
    """Get full model assignment details for a PAT role."""
    return _PAT_DEFAULTS[role]


def get_sat_assignment(role: SatRole) -> ModelAssignment:
    """Get full model assignment details for a SAT role."""
    return _SAT_DEFAULTS[role]


def routing_table_summary() -> Dict[str, Dict[str, str]]:
    """Return the complete routing table as a dict for display/telemetry."""
    table = {}
    for role in PatRole:
        model = get_pat_model(role)
        assignment = _PAT_DEFAULTS[role]
        table[f"pat.{role.value}"] = {
            "model": model,
            "tier": assignment.tier,
            "tok_s": str(assignment.tok_s),
        }
    for role in SatRole:
        model = get_sat_model(role)
        assignment = _SAT_DEFAULTS[role]
        table[f"sat.{role.value}"] = {
            "model": model,
            "tier": assignment.tier,
            "tok_s": str(assignment.tok_s),
        }
    table["embedding"] = {"model": EMBEDDING_MODEL, "tier": "embed", "tok_s": "N/A"}
    return table
