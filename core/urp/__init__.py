# URP - Unified Resource Planner
"""
BIZRA Unified Resource Planner (URP)
=====================================
Hardware-calibrated resource management for Node0.

Node0 Hardware Profile:
- GPU: NVIDIA RTX 4090 (16GB VRAM, 14GB usable)
- CPU: Intel Core i9-14900 (24 cores, 32 threads)
- RAM: 128GB DDR5 (112GB usable)
- Storage: 3TB NVMe SSD (2.5TB usable)

Fixes F-PERF-001: Resource Blindness
Invariants:
- I1: Total_Allocated_VRAM <= 14GB
- I2: GPU_Agents <= 3
- I3: CPU_Agents <= 10
"""

from .manager import (
    URPManager,
    URPLease,
    ResourceRequest,
    Lease,
    URPSnapshot,
    LeaseStatus,
    ResourceMode,
    OverCapacityError,
    MaxAgentsError,
    LeaseNotFoundError,
    LeaseExpiredError,
    # GPU Resources
    TOTAL_VRAM_GB,
    USABLE_VRAM_GB,
    MAX_GPU_AGENTS,
    AGENT_VRAM_REQUIREMENTS,
    # RAM Resources
    TOTAL_RAM_GB,
    USABLE_RAM_GB,
    MAX_CPU_AGENTS,
    AGENT_RAM_REQUIREMENTS,
    # CPU
    CPU_CORES,
    CPU_THREADS,
    # Storage
    TOTAL_STORAGE_GB,
    USABLE_STORAGE_GB,
    # Legacy alias
    MAX_CONCURRENT_AGENTS,
)

__all__ = [
    "URPManager",
    "URPLease", 
    "ResourceRequest",
    "Lease",
    "URPSnapshot",
    "LeaseStatus",
    "ResourceMode",
    "OverCapacityError",
    "MaxAgentsError",
    "LeaseNotFoundError",
    "LeaseExpiredError",
    # Hardware profile
    "TOTAL_VRAM_GB",
    "USABLE_VRAM_GB",
    "MAX_GPU_AGENTS",
    "AGENT_VRAM_REQUIREMENTS",
    "TOTAL_RAM_GB",
    "USABLE_RAM_GB",
    "MAX_CPU_AGENTS",
    "AGENT_RAM_REQUIREMENTS",
    "CPU_CORES",
    "CPU_THREADS",
    "TOTAL_STORAGE_GB",
    "USABLE_STORAGE_GB",
    "MAX_CONCURRENT_AGENTS",
]
