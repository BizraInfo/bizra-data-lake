# URP - Unified Resource Planner
"""
BIZRA Unified Resource Planner (URP)
=====================================
Hardware-calibrated resource management for Node0.

Fixes F-PERF-001: Resource Blindness
Enforces I1: Total_Allocated_VRAM <= 14GB
"""

from .manager import (
    URPManager,
    URPLease,
    ResourceRequest,
    Lease,
    URPSnapshot,
    LeaseStatus,
    OverCapacityError,
    MaxAgentsError,
    LeaseNotFoundError,
    LeaseExpiredError,
    USABLE_VRAM_GB,
    MAX_CONCURRENT_AGENTS,
    AGENT_VRAM_REQUIREMENTS,
)

__all__ = [
    "URPManager",
    "URPLease", 
    "ResourceRequest",
    "Lease",
    "URPSnapshot",
    "LeaseStatus",
    "OverCapacityError",
    "MaxAgentsError",
    "LeaseNotFoundError",
    "LeaseExpiredError",
    "USABLE_VRAM_GB",
    "MAX_CONCURRENT_AGENTS",
    "AGENT_VRAM_REQUIREMENTS",
]
