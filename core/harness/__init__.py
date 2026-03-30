"""
BIZRA Harness — The One Authoritative Mission-to-Receipt Pipeline.

Every mission enters through run_mission(). Every receipt exits through run_mission().
No surface renders without the harness. No exception.

RUNTIME_CUTOVER_04.
"""

from __future__ import annotations

from core.harness.constants import (
    EXECUTION_IHSAN_FLOOR,
    FEDERATION_IHSAN_FLOOR,
    HARNESS_VERSION,
    SURFACE_CONTRACT_VERSION,
)
from core.harness.pipeline import run_mission

__all__ = [
    "run_mission",
    "HARNESS_VERSION",
    "SURFACE_CONTRACT_VERSION",
    "EXECUTION_IHSAN_FLOOR",
    "FEDERATION_IHSAN_FLOOR",
]
