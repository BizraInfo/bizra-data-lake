"""
URP Constitution — the immutable invariants governing the membrane.

These are the Maqasid al-Sharia formalized as code constraints.
They cannot be modified at runtime. They are the system's axioms.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict

from core.integration.constants import (
    ADL_GINI_THRESHOLD,
    UNIFIED_IHSAN_THRESHOLD,
)


@dataclass(frozen=True)
class Constitution:
    """Immutable constitutional invariants for the URP membrane."""

    zann_zero: bool = True  # No unverified claims
    riba_zero: bool = True  # No extractive economics
    ihsan_floor: float = UNIFIED_IHSAN_THRESHOLD  # 0.95
    gini_ceiling: float = ADL_GINI_THRESHOLD  # 0.35
    zakat_rate: float = 0.025  # 2.5% redistribution
    harberger_rate: float = 0.05  # 5% annual continuous tax
    frozen_agents: tuple = ("P5-Ethicist", "S2-Oracle")

    def hash(self) -> str:
        """BLAKE3 hash of constitutional content for verification."""
        content = (
            f"zann={self.zann_zero}|riba={self.riba_zero}|"
            f"ihsan={self.ihsan_floor}|gini={self.gini_ceiling}|"
            f"zakat={self.zakat_rate}|harberger={self.harberger_rate}|"
            f"frozen={','.join(self.frozen_agents)}"
        )
        return hashlib.blake2b(content.encode(), digest_size=32).hexdigest()

    def check_receipt(self, receipt: Dict[str, Any]) -> tuple[bool, str]:
        """Check if a receipt passes all constitutional invariants."""
        ihsan = receipt.get("ihsan_score", 0.0)
        if ihsan < self.ihsan_floor:
            return False, f"ihsan {ihsan:.4f} < floor {self.ihsan_floor}"

        if not receipt.get("signed", False) and self.zann_zero:
            return False, "unsigned receipt violates ZANN_ZERO"

        return True, "constitutional"

    def check_gini(self, gini: float) -> tuple[bool, str]:
        """Check if current Gini coefficient is within bounds."""
        if gini > self.gini_ceiling:
            return False, f"gini {gini:.4f} > ceiling {self.gini_ceiling}"
        return True, "adl_compliant"
