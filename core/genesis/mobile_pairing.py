"""
BIZRA Mobile Pairing — Device Bridge Stub
============================================

Stub for mobile device pairing during genesis bootstrap.
Parses device specifications (e.g., "Z Fold 6:SM-F956B")
and creates a pairing record with proximity routing enabled.

Actual BLE/NFC pairing will be implemented in the mobile
bridge project. This stub ensures the genesis pipeline
completes cleanly with a future-ready interface.

Standing on Giants:
- Bluetooth SIG (1998): Short-range device pairing
- IEEE 802.11 (1997): Proximity networking
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class MobilePairResult:
    """Result of a mobile device pairing operation."""

    device_name: str
    model: str
    paired: bool = True
    proximity_routing: bool = True
    protocol: str = "stub-v1"  # Will be "ble-v1" or "nfc-v1" in production

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_name": self.device_name,
            "model": self.model,
            "paired": self.paired,
            "proximity_routing": self.proximity_routing,
            "protocol": self.protocol,
        }


def pair_mobile(device_spec: str) -> MobilePairResult:
    """
    Parse and pair a mobile device.

    Device spec format: "DeviceName:ModelNumber"
    Example: "Z Fold 6:SM-F956B"

    Args:
        device_spec: Device specification string

    Returns:
        MobilePairResult with pairing status
    """
    # Parse device spec
    parts = device_spec.split(":", 1)
    device_name = parts[0].strip()
    model = parts[1].strip() if len(parts) > 1 else "Unknown"

    result = MobilePairResult(
        device_name=device_name,
        model=model,
        paired=True,
        proximity_routing=True,
    )

    logger.info(
        "Mobile paired (stub): %s [%s] — proximity routing enabled",
        device_name,
        model,
    )
    return result
