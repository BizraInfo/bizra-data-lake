"""
URP Resource Pledge — Compute Resource Commitment
===================================================
Stub for Universal Resource Pool pledging. Creates a signed
pledge record; actual URP enforcement is in Rust (bizra-resourcepool).

Standing on Giants:
- Wiener (1948): Resource allocation in cybernetic systems
- Ostrom (1990): Commons resource governance
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class URPPledge:
    """A node's resource pledge to the Universal Resource Pool."""

    node_id: str
    ram_gb: float = 0.0
    vram_gb: float = 0.0
    storage_gb: float = 0.0
    pledge_hash: str = ""

    def __post_init__(self) -> None:
        if not self.pledge_hash:
            pledge_str = f"{self.node_id}|{self.ram_gb}|{self.vram_gb}|{self.storage_gb}"
            self.pledge_hash = hashlib.sha256(pledge_str.encode()).hexdigest()[:32]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "ram_gb": self.ram_gb,
            "vram_gb": self.vram_gb,
            "storage_gb": self.storage_gb,
            "pledge_hash": self.pledge_hash,
        }


def pledge_resources(
    node_id: str,
    hardware_info: Dict[str, Any],
) -> URPPledge:
    """
    Create a resource pledge from hardware scan results.

    Pledges 50% of detected RAM and storage as a conservative default.
    Actual enforcement happens in the Rust bizra-resourcepool crate.

    Args:
        node_id: Node ID making the pledge
        hardware_info: Dict from HardwareScanner.scan()

    Returns:
        URPPledge with computed pledge hash
    """
    ram_gb = hardware_info.get("ram_gb", 0.0)

    return URPPledge(
        node_id=node_id,
        ram_gb=round(ram_gb * 0.5, 1),  # Pledge 50% of RAM
        vram_gb=0.0,  # VRAM detection is GPU-specific
        storage_gb=0.0,  # Storage detection is future work
    )
