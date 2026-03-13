"""
URP Rust Bridge — Fail-closed Python wrapper for PyO3 bindings.

Every function returns None when Rust is unavailable (Level 0 degradation).
The node continues working with Python-only pledge verification.

Standing on Giants: Liskov (substitution — None is valid return)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class URPRustBridge:
    """Wrapper providing graceful degradation for URP Rust operations.

    When the PyO3-built ``bizra`` package is available, operations are
    forwarded to the Rust ``ResourcePool`` for authoritative validation.
    When unavailable, every method returns ``None`` (never raises).
    """

    _pool: Any = None
    _available: bool = False

    def __init__(self) -> None:
        try:
            from bizra import PyResourcePool  # type: ignore[import-untyped]

            self._pool = PyResourcePool()
            self._available = True
        except (ImportError, RuntimeError, OSError):
            self._available = False
            logger.info("URP Rust bridge unavailable — Level 0 mode")

    @property
    def available(self) -> bool:
        """Whether the Rust pool backend is loaded."""
        return self._available

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def submit_pledge(self, pledge: Any) -> Optional[Dict[str, Any]]:
        """Submit pledge to Rust pool. Returns node dict or None."""
        if not self._available:
            return None
        try:
            from bizra import PyURPPledge, submit_pledge  # type: ignore[import-untyped]

            pledge_dict = pledge.to_dict() if hasattr(pledge, "to_dict") else pledge
            rust_pledge = PyURPPledge.from_dict(pledge_dict)
            node = submit_pledge(self._pool, rust_pledge)
            return node.to_dict()
        except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
            logger.warning("URP submit failed: %s", exc)
            return None

    def contribute(
        self,
        node_id: str,
        resource_type: str,
        amount: float,
        duration_ms: int,
        proof_hash: str,
    ) -> Optional[Dict[str, Any]]:
        """Record contribution. Returns receipt dict or None."""
        if not self._available:
            return None
        try:
            from bizra import contribute_resources  # type: ignore[import-untyped]

            receipt = contribute_resources(
                self._pool, node_id, resource_type, amount, duration_ms, proof_hash
            )
            return receipt.to_dict()
        except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
            logger.warning("URP contribute failed: %s", exc)
            return None

    def get_rewards(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get SEED balance and history. Returns dict or None."""
        if not self._available:
            return None
        try:
            from bizra import get_rewards  # type: ignore[import-untyped]

            return get_rewards(self._pool, node_id)
        except (RuntimeError, TypeError, ValueError, AttributeError):
            return None

    def process_zakat(self) -> Optional[Dict[str, Any]]:
        """Trigger Zakat distribution. Returns summary dict or None."""
        if not self._available:
            return None
        try:
            from bizra import process_zakat  # type: ignore[import-untyped]

            return process_zakat(self._pool)
        except (RuntimeError, TypeError, ValueError, AttributeError):
            return None

    def check_adl(self) -> Optional[Dict[str, Any]]:
        """Check ADL compliance. Returns dict or None."""
        if not self._available:
            return None
        try:
            from bizra import check_adl  # type: ignore[import-untyped]

            return check_adl(self._pool)
        except (RuntimeError, TypeError, ValueError, AttributeError):
            return None

    def stats(self) -> Optional[Dict[str, Any]]:
        """Get pool stats. Returns dict or None."""
        if not self._available:
            return None
        try:
            from bizra import pool_stats  # type: ignore[import-untyped]

            result = pool_stats(self._pool)
            return result.to_dict() if hasattr(result, "to_dict") else result
        except (RuntimeError, TypeError, ValueError, AttributeError):
            return None
