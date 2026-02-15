"""
Genesis Data Types — Bootstrap Configuration and Results
=========================================================

Standing on Giants:
- Al-Ghazali (1095): Covenant types for ethical genesis
- Deming (1950): Step-based quality tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GenesisConfig:
    """Configuration for a genesis bootstrap run."""

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
    json_output: bool = False


@dataclass
class GenesisStep:
    """Record of a single genesis step execution."""

    name: str
    status: str = "pending"  # pending | running | success | failed | skipped
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def success(self) -> bool:
        return self.status == "success"


@dataclass
class GenesisResult:
    """Result of a full genesis bootstrap run."""

    steps: List[GenesisStep] = field(default_factory=list)
    node_id: str = ""
    genesis_hash: str = ""
    total_duration_ms: float = 0.0
    success: bool = False

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def success_count(self) -> int:
        return sum(1 for s in self.steps if s.success)

    @property
    def failed_count(self) -> int:
        return sum(1 for s in self.steps if s.status == "failed")

    def get_step(self, name: str) -> Optional[GenesisStep]:
        for s in self.steps:
            if s.name == name:
                return s
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "genesis_hash": self.genesis_hash,
            "success": self.success,
            "total_duration_ms": self.total_duration_ms,
            "steps": [
                {
                    "name": s.name,
                    "status": s.status,
                    "duration_ms": s.duration_ms,
                    "error": s.error,
                }
                for s in self.steps
            ],
        }
