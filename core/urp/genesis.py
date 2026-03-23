"""
URP Genesis — the one-time mint that creates the constitutional membrane.

Called by Node0 at activation. Creates the URP, deploys SAT agents,
and connects Node0 as the first participant. The flywheel starts here.
"""

from __future__ import annotations

import logging
from typing import Optional

from core.urp.service import URPService, URPGenesisReceipt

logger = logging.getLogger("bizra.urp.genesis")

# Module-level singleton — one URP per process
_urp_instance: Optional[URPService] = None


def mint_urp_genesis(
    founder_node_id: str,
    founder_public_key: str,
) -> tuple[URPService, URPGenesisReceipt]:
    """Mint the URP. Called once by Node0 at activation.

    Returns the URP service instance and the genesis receipt.
    Subsequent calls return the existing instance (idempotent).
    """
    global _urp_instance

    if _urp_instance is not None and _urp_instance.genesis_complete:
        logger.info("URP already minted — returning existing instance")
        return _urp_instance, _urp_instance._genesis_receipt

    urp = URPService()
    receipt = urp.mint_genesis(
        founder_node_id=founder_node_id,
        founder_public_key=founder_public_key,
    )

    _urp_instance = urp
    logger.info(
        "URP Genesis complete: id=%s, founder=%s",
        receipt.urp_id[:16],
        founder_node_id,
    )
    return urp, receipt


def get_urp() -> Optional[URPService]:
    """Get the current URP instance (None if not yet minted)."""
    return _urp_instance


def reset_urp() -> None:
    """Reset URP state (testing only)."""
    global _urp_instance
    _urp_instance = None
