"""
Mobile Pairing — Device Companion Stub
========================================
Parses mobile device specifications and returns a pairing
confirmation. Actual BLE/NFC pairing is future mobile-bridge work.

Standing on Giants:
- Weiser (1991): Ubiquitous computing vision
- Shannon (1948): Communication channel establishment
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class MobilePairResult:
    """Result of a mobile device pairing attempt."""

    device_name: str = ""
    model: str = ""
    paired: bool = False
    proximity_routing: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_name": self.device_name,
            "model": self.model,
            "paired": self.paired,
            "proximity_routing": self.proximity_routing,
        }


def pair_mobile(device_spec: str) -> MobilePairResult:
    """
    Parse a device specification and return a pairing result.

    Accepts format: "Device Name:Model" (e.g. "Z Fold 6:SM-F956B").
    This is a stub — actual BLE/NFC pairing is future work.

    Args:
        device_spec: Device specification string

    Returns:
        MobilePairResult with parsed device info
    """
    parts = device_spec.split(":", 1)
    device_name = parts[0].strip() if parts else device_spec
    model = parts[1].strip() if len(parts) > 1 else ""

    return MobilePairResult(
        device_name=device_name,
        model=model,
        paired=True,
        proximity_routing=True,
    )
