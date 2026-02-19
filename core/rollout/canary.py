"""Deterministic canary routing for Phase 46 components.

Routes a stable, reproducible percentage of requests through Phase 46
paths using a hash of (salt, component, request_key). Kill switches
(boolean env flags) always take precedence over percentage routing.

Standing on Giants: Fowler (canary releases, 2010)
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Dict, Optional, Tuple

from core.integration.constants import CANARY_DEFAULT_SALT

logger = logging.getLogger(__name__)


class CanaryRouter:
    """Deterministic canary routing using stable hashing.

    Kill switches (boolean flags) ALWAYS take precedence:
    - If BIZRA_PHASE46_SEARCH_ENABLED == "0", search is OFF regardless of percent.
    - Percent routing only applies when boolean flag is "1" or not set.
    """

    _ENV_MAP: Dict[str, str] = {
        "search": "BIZRA_PHASE46_SEARCH_ENABLED",
        "got_bridge": "BIZRA_PHASE46_GOT_BRIDGE_ENABLED",
        "hmm": "BIZRA_PHASE46_HMM_ENABLED",
    }

    _PERCENT_MAP: Dict[str, str] = {
        "search": "BIZRA_PHASE46_SEARCH_PERCENT",
        "got_bridge": "BIZRA_PHASE46_GOT_BRIDGE_PERCENT",
        "hmm": "BIZRA_PHASE46_HMM_PERCENT",
    }

    def __init__(self, salt: Optional[str] = None) -> None:
        self._salt = salt or os.getenv(
            "BIZRA_PHASE46_CANARY_SALT", CANARY_DEFAULT_SALT
        )

    def should_route(
        self, component: str, request_key: str, percent: Optional[int] = None
    ) -> bool:
        """Determine if this request should use the Phase 46 canary path.

        Args:
            component: ``"search"`` | ``"got_bridge"`` | ``"hmm"``
            request_key: Stable identifier (request_id, query hash, caller_id).
            percent: 0-100 routing percentage.  When *None*, reads from env.

        Returns:
            ``True`` if this request should use the Phase 46 component.
        """
        if percent is None:
            percent = self._read_percent(component)

        # Gate 0: Percent bounds
        if percent <= 0:
            return False
        if percent >= 100:
            return True

        # Gate 1: Kill switch precedence
        kill = self._check_kill_switch(component)
        if kill is not None:
            return kill

        # Gate 2: Deterministic hash routing
        hash_input = f"{self._salt}:{component}:{request_key}"
        digest = hashlib.md5(hash_input.encode()).hexdigest()  # noqa: S324
        bucket = int(digest[:8], 16) % 100  # 0-99

        routed = bucket < percent
        logger.debug(
            "canary: component=%s key=%.20s pct=%d bucket=%d routed=%s",
            component,
            request_key,
            percent,
            bucket,
            routed,
        )
        return routed

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    def _check_kill_switch(self, component: str) -> Optional[bool]:
        """Check boolean kill switch.  Returns ``None`` if not explicitly set."""
        env_key = self._ENV_MAP.get(component)
        if env_key is None:
            return None
        value = os.getenv(env_key)
        if value is None:
            return None  # Not set — defer to percent
        low = value.lower()
        if low in ("0", "false", "no"):
            return False
        if low in ("1", "true", "yes"):
            return True
        return None

    # ------------------------------------------------------------------
    # Percent helpers
    # ------------------------------------------------------------------

    def _read_percent(self, component: str) -> int:
        env_key = self._PERCENT_MAP.get(component, "")
        raw = os.getenv(env_key, "0")
        try:
            return max(0, min(100, int(raw)))
        except (ValueError, TypeError):
            return 0

    def get_active_percents(self) -> Dict[str, int]:
        """Return current canary percentages from env."""
        return {c: self._read_percent(c) for c in self._PERCENT_MAP}

    @property
    def salt(self) -> str:
        return self._salt
