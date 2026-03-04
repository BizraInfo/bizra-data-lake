"""
BIZRA Floor Constraint — Universality Gate.

The floor is the supreme design constraint. Everything else
(4090 acceleration, reverse scaling, URP contribution) is built
ON TOP of this floor. Without the floor, there is no building.

A system that requires a 4090 addresses ~50M users.
A system that runs on a $200 phone addresses 8B users.
The floor is a 160x market expansion.

Standing on Giants:
- Shannon (channel capacity floor)
- Al-Ghazali (Daughter Test — deploy for her)
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .asset_registry import HardwareAsset, NodeBody

logger = logging.getLogger(__name__)


# ── Data model ────────────────────────────────────────────────


@dataclass(frozen=True)
class FloorProfile:
    """Minimum viable hardware profile for a BIZRA node.

    This is the grandmother's $200 phone in Cairo.
    This is the student's 5-year-old laptop in Jakarta.
    If the pipeline doesn't work HERE, it doesn't work.
    """

    min_ram_gb: float = 2.0
    min_storage_gb: float = 4.0
    min_cpu_cores: int = 2
    min_cpu_mhz: int = 1000
    gpu_required: bool = False  # GPU is NEVER required
    network_required: bool = False  # offline-first always
    min_model_params_b: float = 0.5
    max_pipeline_time_s: float = 60.0
    max_memory_usage_mb: float = 512.0


@dataclass
class FloorCheckResult:
    """Result of checking a node against the floor constraint."""

    compliant: bool
    node_body: NodeBody
    floor_profile: FloorProfile
    violations: List[str]
    margin: Dict[str, float]
    pipeline_time_s: Optional[float] = None
    pipeline_memory_mb: Optional[float] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass(frozen=True)
class FloorBenchmark:
    """Recorded benchmark of the constitutional pipeline on this hardware."""

    benchmark_id: str
    hardware_summary: str
    pipeline_time_s: float
    pipeline_memory_mb: float
    model_used: str
    ihsan_score: float
    constitutional_gates_passed: bool
    evidence_hash: str
    benchmarked_at: str


# ── Core class ────────────────────────────────────────────────


class FloorConstraint:
    """Enforces minimum viable node universality.

    The floor is the supreme design constraint. Everything else
    (4090 acceleration, reverse scaling, URP contribution) is built
    ON TOP of this floor. Without the floor, there is no building.
    """

    def __init__(
        self,
        floor_profile: Optional[FloorProfile] = None,
        evidence_ledger: Optional[Any] = None,
    ) -> None:
        self._floor = floor_profile or FloorProfile()
        self._evidence = evidence_ledger
        self._benchmarks: List[FloorBenchmark] = []

    @property
    def floor(self) -> FloorProfile:
        """Expose floor profile for inspection."""
        return self._floor

    # ── Constraint checking ────────────────────────────────────

    def check(self, body: NodeBody) -> FloorCheckResult:
        """Check if a node meets the floor constraint.

        GPU is NEVER checked — GPU is NEVER required.
        Network is NEVER checked — offline-first always.
        """
        violations: List[str] = []
        margin: Dict[str, float] = {}
        total = body.total_capacity
        floor = self._floor

        # RAM check
        ram_total = total.get("ram", 0.0)
        margin["ram"] = round(ram_total - floor.min_ram_gb, 2)
        if ram_total < floor.min_ram_gb:
            violations.append(
                f"RAM below minimum: {ram_total:.1f} GB < {floor.min_ram_gb:.1f} GB"
            )

        # Disk check
        disk_total = total.get("disk", 0.0)
        margin["disk"] = round(disk_total - floor.min_storage_gb, 2)
        if disk_total < floor.min_storage_gb:
            violations.append(
                f"Storage below minimum: {disk_total:.1f} GB < {floor.min_storage_gb:.1f} GB"
            )

        # CPU check
        cpu_total = total.get("cpu", 0.0)
        margin["cpu"] = round(cpu_total - floor.min_cpu_cores, 2)
        if cpu_total < floor.min_cpu_cores:
            violations.append(
                f"CPU cores below minimum: {cpu_total:.0f} < {floor.min_cpu_cores}"
            )

        # GPU — NEVER checked. GPU is NEVER required.
        gpu_total = total.get("gpu", 0.0)
        margin["gpu"] = round(gpu_total, 2)  # all GPU is surplus

        # Network — NEVER checked. Offline-first always.
        margin["network"] = 0.0  # not a constraint

        compliant = len(violations) == 0

        return FloorCheckResult(
            compliant=compliant,
            node_body=body,
            floor_profile=floor,
            violations=violations,
            margin=margin,
        )

    def check_pipeline_time(
        self,
        measured_time_s: float,
        measured_memory_mb: float,
    ) -> bool:
        """Check if the constitutional pipeline meets time/memory floor."""
        if measured_time_s > self._floor.max_pipeline_time_s:
            return False
        if measured_memory_mb > self._floor.max_memory_usage_mb:
            return False
        return True

    # ── Floor simulation ───────────────────────────────────────

    def simulate_floor_node(self) -> FloorProfile:
        """Return the floor profile for testing.

        This profile represents the weakest node BIZRA must support.
        Tests MUST pass under this profile.
        """
        return FloorProfile()

    # ── Dashboard ──────────────────────────────────────────────

    def floor_report(self, body: NodeBody) -> Dict[str, Any]:
        """Human-readable floor compliance report."""
        result = self.check(body)
        total = body.total_capacity
        floor = self._floor

        # Headroom: how far above the floor
        headroom: Dict[str, str] = {}
        cpu = total.get("cpu", 0.0)
        if floor.min_cpu_cores > 0 and cpu > 0:
            headroom["cpu"] = f"{cpu / floor.min_cpu_cores:.0f}x above floor"
        ram = total.get("ram", 0.0)
        if floor.min_ram_gb > 0 and ram > 0:
            headroom["ram"] = f"{ram / floor.min_ram_gb:.0f}x above floor"
        disk = total.get("disk", 0.0)
        if floor.min_storage_gb > 0 and disk > 0:
            headroom["disk"] = f"{disk / floor.min_storage_gb:.0f}x above floor"
        gpu = total.get("gpu", 0.0)
        if gpu > 0:
            headroom["gpu"] = f"bonus ({gpu:.1f} GB, not required)"
        else:
            headroom["gpu"] = "none (not required)"

        # Surplus for URP: what's above the floor
        surplus: Dict[str, str] = {}
        cpu_surplus = max(0.0, cpu - floor.min_cpu_cores)
        surplus["cpu"] = f"{cpu_surplus:.0f} cores available"
        ram_surplus = max(0.0, ram - floor.min_ram_gb)
        surplus["ram"] = f"{ram_surplus:.1f} GB available"
        if gpu > 0:
            surplus["gpu"] = f"{gpu:.1f} GB VRAM (100% surplus)"

        # Daughter Test
        dt = "PASSED" if result.compliant else "FAILED"
        dt_msg = f"{dt} {'— this pipeline works for her' if result.compliant else '— FLOOR VIOLATION'}"

        # Floor summary
        floor_desc = (
            f"{floor.min_cpu_cores}-core, "
            f"{floor.min_ram_gb:.0f}GB RAM, "
            f"{floor.min_storage_gb:.0f}GB disk, "
            f"no GPU"
        )

        # Node summary
        gpu_str = ""
        gpu_assets = [a for a in body.assets.values() if a.asset_type == "gpu"]
        if gpu_assets:
            gpu_str = f", {sum(a.capacity_total for a in gpu_assets):.0f}GB GPU"
        node_desc = (
            f"{cpu:.0f}-core, "
            f"{ram:.0f}GB RAM, "
            f"{disk:.0f}GB disk"
            f"{gpu_str}"
        )

        return {
            "compliant": result.compliant,
            "violations": result.violations,
            "floor": floor_desc,
            "this_node": node_desc,
            "headroom": headroom,
            "surplus_for_urp": surplus,
            "daughter_test": dt_msg,
        }


# ── Public function ────────────────────────────────────────────


def daughter_test(body: NodeBody) -> bool:
    """Would you deploy this for your daughter?

    She has a mid-range phone from 3 years ago,
    unreliable mobile data, no GPU, no cloud subscription.
    But she IS a node. She IS sovereign.

    If the pipeline doesn't work for her,
    the ceiling is irrelevant.
    """
    floor = FloorProfile()
    result = FloorConstraint(floor).check(body)
    return result.compliant
