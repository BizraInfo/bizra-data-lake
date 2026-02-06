"""
Integration Types — Data Classes and Enums for Integration Module
=================================================================
Type definitions for the Sovereign LLM Integration system.

Standing on Giants: Shannon + Lamport + Vaswani + Anthropic
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .capability_card import ModelTier, TaskType


# =============================================================================
# ENUMS
# =============================================================================

class AdmissionStatus(Enum):
    """Strict Synthesis admission statuses."""
    RUNTIME = "RUNTIME"    # Z3-proven, Ihsan = 1.0, executed
    MUSEUM = "MUSEUM"      # SNR-v2 scored, awaiting Z3 proof, archived
    SANDBOX = "SANDBOX"    # Simulation only, no PCI signing
    REJECTED = "REJECTED"  # Below quality threshold


class NetworkMode(Enum):
    """Network operation modes."""
    OFFLINE = "offline"
    LOCAL_ONLY = "local"
    FEDERATED = "federated"
    HYBRID = "hybrid"


# =============================================================================
# DATA CLASSES - Z3 Certificates
# =============================================================================

@dataclass
class Z3Certificate:
    """Z3 formal verification certificate."""
    hash: str
    valid: bool
    proof_type: str = "z3-smt2"
    verified_at: str = ""


@dataclass
class AdmissionResult:
    """Result of Constitutional Gate admission check."""
    status: AdmissionStatus
    score: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    promotion_path: Optional[str] = None


# =============================================================================
# DATA CLASSES - Configuration
# =============================================================================

@dataclass
class SovereignConfig:
    """Configuration for the Sovereign Runtime."""
    # Network
    network_mode: NetworkMode = NetworkMode.HYBRID
    discovery_timeout_ms: int = 5000
    bootstrap_nodes: List[str] = None

    # Model Store
    model_store_path: Optional[Path] = None
    default_model: Optional[str] = None

    # Inference
    sandbox_enabled: bool = True
    gpu_layers: int = -1
    context_length: int = 4096

    # Federation
    pool_min_peers: int = 1
    pool_quorum: float = 0.67
    pool_timeout_ms: int = 60000

    # Security
    keypair_path: Optional[Path] = None
    post_quantum: bool = False

    def __post_init__(self):
        if self.bootstrap_nodes is None:
            self.bootstrap_nodes = []
        if self.model_store_path is None:
            self.model_store_path = Path.home() / ".bizra" / "models"
        if self.keypair_path is None:
            self.keypair_path = Path.home() / ".bizra" / "keypair.json"

    @classmethod
    def from_env(cls) -> "SovereignConfig":
        """Create config from environment variables."""
        return cls(
            network_mode=NetworkMode(os.getenv("BIZRA_NETWORK_MODE", "hybrid").lower()),
            discovery_timeout_ms=int(os.getenv("BIZRA_DISCOVERY_TIMEOUT_MS", "5000")),
            bootstrap_nodes=os.getenv("BIZRA_BOOTSTRAP_NODES", "").split(",") if os.getenv("BIZRA_BOOTSTRAP_NODES") else [],
            model_store_path=Path(os.path.expanduser(os.getenv("BIZRA_MODEL_STORE", "~/.bizra/models"))),
            default_model=os.getenv("BIZRA_DEFAULT_MODEL"),
            sandbox_enabled=os.getenv("BIZRA_SANDBOX", "1") == "1",
            gpu_layers=int(os.getenv("BIZRA_GPU_LAYERS", "-1")),
            context_length=int(os.getenv("BIZRA_CONTEXT_LENGTH", "4096")),
            pool_min_peers=int(os.getenv("BIZRA_POOL_MIN_PEERS", "1")),
            pool_quorum=float(os.getenv("BIZRA_POOL_QUORUM", "0.67")),
            pool_timeout_ms=int(os.getenv("BIZRA_POOL_TIMEOUT_MS", "60000")),
            keypair_path=Path(os.path.expanduser(os.getenv("BIZRA_KEYPAIR_PATH", "~/.bizra/keypair.json"))),
            post_quantum=os.getenv("BIZRA_POST_QUANTUM", "false").lower() == "true",
        )


# =============================================================================
# DATA CLASSES - Inference
# =============================================================================

@dataclass
class InferenceRequest:
    """Request for inference."""
    prompt: str
    model_id: Optional[str] = None
    task_type: TaskType = TaskType.CHAT
    max_tokens: int = 256
    temperature: float = 0.7


@dataclass
class InferenceResult:
    """Result from inference."""
    content: str
    model_id: str
    tier: ModelTier
    ihsan_score: float
    snr_score: float
    generation_time_ms: int
    gate_passed: bool
    used_pool: bool = False
    used_fallback: bool = False


__all__ = [
    # Enums
    "AdmissionStatus",
    "NetworkMode",
    # Z3 Certificates
    "Z3Certificate",
    "AdmissionResult",
    # Configuration
    "SovereignConfig",
    # Inference
    "InferenceRequest",
    "InferenceResult",
]
