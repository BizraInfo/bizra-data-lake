"""
BIZRA Genesis Types — Bootstrap Configuration & Results
=========================================================

Data structures for the one-command genesis bootstrap pipeline.
GenesisConfig captures all CLI flags; GenesisResult records
every step with timing, forming an auditable genesis receipt.

Standing on Giants:
- Nakamoto (2008): Genesis block concept
- Lamport (1978): Ordered step execution
- Shannon (1948): SNR on genesis quality
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class GenesisStepStatus(Enum):
    """Status of an individual genesis step."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class GenesisConfig:
    """
    Configuration for the genesis bootstrap pipeline.
    Maps 1:1 to CLI flags from the dream command.
    """

    identity_genesis: bool = False
    hardware_scan: bool = False
    pat_count: int = 7
    sat_count: int = 5
    hda_bridge: bool = False
    mobile_pair: Optional[str] = None  # e.g. "Z Fold 6:SM-F956B"
    guild_join: Optional[str] = None  # e.g. "agriculture"
    quest_accept: Optional[str] = None  # e.g. "001-sustainable-water"
    ihsan_target: float = 0.999
    node_dir: Optional[str] = None
    architect_name: str = "MoMo"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity_genesis": self.identity_genesis,
            "hardware_scan": self.hardware_scan,
            "pat_count": self.pat_count,
            "sat_count": self.sat_count,
            "hda_bridge": self.hda_bridge,
            "mobile_pair": self.mobile_pair,
            "guild_join": self.guild_join,
            "quest_accept": self.quest_accept,
            "ihsan_target": self.ihsan_target,
        }


@dataclass
class GenesisStep:
    """A single step in the genesis pipeline with timing."""

    name: str
    status: GenesisStepStatus = GenesisStepStatus.PENDING
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "duration_ms": round(self.duration_ms, 2),
            "details": self.details,
            "error": self.error,
        }


@dataclass
class GenesisResult:
    """
    Complete result of a genesis bootstrap run.
    Forms an auditable receipt with deterministic hash.
    """

    steps: List[GenesisStep] = field(default_factory=list)
    node_id: str = ""
    genesis_hash: str = ""
    total_duration_ms: float = 0.0
    success: bool = False
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )

    @property
    def successful_steps(self) -> int:
        return sum(1 for s in self.steps if s.status == GenesisStepStatus.SUCCESS)

    @property
    def failed_steps(self) -> int:
        return sum(1 for s in self.steps if s.status == GenesisStepStatus.FAILED)

    @property
    def skipped_steps(self) -> int:
        return sum(1 for s in self.steps if s.status == GenesisStepStatus.SKIPPED)

    def compute_hash(self) -> str:
        """Compute deterministic hash of the genesis result."""
        payload = json.dumps(
            {
                "node_id": self.node_id,
                "steps": [
                    s.name for s in self.steps if s.status == GenesisStepStatus.SUCCESS
                ],
                "created_at": self.created_at,
            },
            sort_keys=True,
        )
        self.genesis_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]
        return self.genesis_hash

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "genesis_hash": self.genesis_hash,
            "success": self.success,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "successful_steps": self.successful_steps,
            "failed_steps": self.failed_steps,
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at,
        }


# Terminal output styling constants
CHECKMARK = "\u2713"  # ✓
CROSSMARK = "\u2717"  # ✗
OMEGA = "\u03a9"  # Ω
