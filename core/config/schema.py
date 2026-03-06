"""
Config Schema — Pydantic Models for BIZRA Configuration
═══════════════════════════════════════════════════════

All sections are optional with sensible defaults. Policy defaults
are pulled from constants.py SSoT.

Phase 68.03 — Sovereign Instantiation
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from core.integration.constants import (
    ADL_GINI_THRESHOLD,
    INTENT_FLOOR,
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)


class NodeConfig(BaseModel):
    """Node identity."""

    id: str = ""
    covenant_hash: str = ""


class PolicyConfig(BaseModel):
    """Constitutional thresholds. Defaults from SSoT."""

    ihsan_floor: float = UNIFIED_IHSAN_THRESHOLD
    intent_floor: float = INTENT_FLOOR
    gini_target: float = ADL_GINI_THRESHOLD
    snr_minimum: float = UNIFIED_SNR_THRESHOLD


class OrchestratorConfig(BaseModel):
    """Mission orchestration settings."""

    routing_model: Literal["hhmm", "keyword", "reflex_first"] = "hhmm"
    max_workers: int = Field(default=7, ge=1, le=32)
    s2_calls_per_hour: int = Field(default=120, ge=0)
    omega_max_iterations: int = Field(default=50, ge=1, le=1000)


class InferenceConfig(BaseModel):
    """LLM inference backend settings."""

    primary: str = "auto"
    fallback: str = "ollama"
    timeout_ms: int = Field(default=30000, ge=1000, le=300000)


class HooksPreExecution(BaseModel):
    """Pre-execution hook configuration."""

    deny_paths: list[str] = Field(
        default_factory=lambda: ["**/.env*", "**/secrets.*", "**/.git/**"]
    )
    require_attestation: list[str] = Field(
        default_factory=lambda: ["network:*", "self_modify:*"]
    )


class HooksPostReceipt(BaseModel):
    """Post-receipt hook configuration."""

    on_write: list[str] = Field(default_factory=lambda: ["format", "lint"])
    on_code_change: list[str] = Field(
        default_factory=lambda: ["typecheck", "tests:related"]
    )


class HooksConfig(BaseModel):
    """Hooks configuration."""

    pre_execution: HooksPreExecution = Field(default_factory=HooksPreExecution)
    post_receipt: HooksPostReceipt = Field(default_factory=HooksPostReceipt)


class BridgeConfig(BaseModel):
    """External bridge configuration."""

    id: str
    enabled: bool = True
    scopes: list[str] = Field(default_factory=list)


class CapsuleConfig(BaseModel):
    """Capsule discovery settings."""

    dir: str = "./capsules/"
    auto_discover: bool = True


class EconomyConfig(BaseModel):
    """Constitutional ticker economy settings."""

    enabled: bool = True
    zakat_cycle_hours: int = 8760
    tick_interval_ms: int = 3600000


class BizraConfig(BaseModel):
    """Unified BIZRA node configuration — 3-scope merged."""

    node: NodeConfig = Field(default_factory=NodeConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    hooks: HooksConfig = Field(default_factory=HooksConfig)
    bridges: list[BridgeConfig] = Field(default_factory=list)
    capsules: CapsuleConfig = Field(default_factory=CapsuleConfig)
    economy: EconomyConfig = Field(default_factory=EconomyConfig)
