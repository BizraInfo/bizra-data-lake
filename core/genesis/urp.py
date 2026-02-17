"""
BIZRA URP Pledge — Universal Resource Pool Contribution
=========================================================

When a node joins the network, it pledges a portion of its
hardware resources to the Universal Resource Pool (URP).
This stub creates a signed pledge record; actual enforcement
and scheduling lives in Rust (bizra-resourcepool crate).

Standing on Giants:
- Baran (1964): Distributed resource sharing
- Ostrom (1990): Commons governance with commitment
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class URPPledge:
    """A node's resource pledge to the Universal Resource Pool."""

    node_id: str
    ram_gb: int
    vram_gb: int
    storage_gb: int = 0
    pledge_hash: str = ""
    pledged_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "ram_gb": self.ram_gb,
            "vram_gb": self.vram_gb,
            "storage_gb": self.storage_gb,
            "pledge_hash": self.pledge_hash,
            "pledged_at": self.pledged_at,
        }


def pledge_resources(
    node_id: str,
    hardware_info: Dict[str, Any],
) -> URPPledge:
    """
    Create a URP resource pledge from hardware info.

    The pledge commits a portion of the node's resources:
    - RAM: full amount detected
    - VRAM: full GPU memory detected
    - Storage: 0 (placeholder for future)

    In production, this pledge is signed with the node's Ed25519
    key and submitted to the federation for validation.

    Args:
        node_id: Pledging node's identity
        hardware_info: Dict with ram_gb, vram_gb keys

    Returns:
        URPPledge with deterministic hash
    """
    ram_gb = hardware_info.get("ram_gb", 0)
    vram_gb = hardware_info.get("vram_gb", 0)
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    # Compute deterministic pledge hash
    payload = json.dumps(
        {"node_id": node_id, "ram_gb": ram_gb, "vram_gb": vram_gb, "at": now},
        sort_keys=True,
    )
    pledge_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]

    pledge = URPPledge(
        node_id=node_id,
        ram_gb=ram_gb,
        vram_gb=vram_gb,
        pledge_hash=pledge_hash,
        pledged_at=now,
    )

    logger.info(
        "URP pledge created: %s — %dGB RAM + %dGB VRAM (hash: %s)",
        node_id, ram_gb, vram_gb, pledge_hash,
    )
    return pledge
