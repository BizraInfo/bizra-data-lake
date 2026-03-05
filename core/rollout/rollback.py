"""Strict rollback automation for Phase 46 canary rollout.

Policy: any two consecutive breached evaluation windows triggers rollback.
Rollback sequence (reverse activation): HMM -> GoT -> Search -> hard kill.
Every rollback emits an immutable receipt to ``artifacts/rollback_receipts/``.

Standing on Giants: Nygard (Release It!, 2007) · Fowler (canary, 2010)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from core.integration.constants import ROLLBACK_CONSECUTIVE_BREACHES

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RollbackReceipt:
    """Immutable receipt for every rollback event."""

    timestamp: str
    trigger: str
    breach_count: int
    component: str
    action: str
    previous_config: Dict[str, str]
    metrics_snapshot: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _BreachWindow:
    """Tracks consecutive breaches for a single metric."""

    metric_name: str
    consecutive_count: int = 0
    last_breached: bool = False
    last_evaluation: Optional[datetime] = None


class RollbackEngine:
    """Strict rollback automation for Phase 46.

    - 2 consecutive breached windows -> rollback
    - Rollback order: HMM % -> GoT % -> Search % -> hard kill
    - Every rollback persists a JSON receipt
    """

    _TRACKED_METRICS = (
        "search_error_rate",
        "got_fallback_rate",
        "hmm_confidence",
        "resonance_snr",
        "latency_regression",
    )

    def __init__(
        self,
        receipt_dir: Optional[str] = None,
        metrics: Optional[Any] = None,
    ) -> None:
        self._receipt_dir = Path(receipt_dir or "artifacts/rollback_receipts")
        self._receipt_dir.mkdir(parents=True, exist_ok=True)
        self._metrics = metrics
        self._breach_windows: Dict[str, _BreachWindow] = {
            name: _BreachWindow(metric_name=name) for name in self._TRACKED_METRICS
        }
        self._rollback_in_progress = False

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(self, metric_name: str, breached: bool) -> Optional[RollbackReceipt]:
        """Evaluate a metric window.

        Returns a :class:`RollbackReceipt` if rollback was triggered,
        otherwise ``None``.
        """
        window = self._breach_windows.get(metric_name)
        if window is None:
            logger.warning("Unknown rollback metric: %s", metric_name)
            return None

        window.last_evaluation = datetime.now(timezone.utc)

        if breached:
            window.consecutive_count += 1
            window.last_breached = True
            logger.warning(
                "Rollback eval: %s breached (%d consecutive)",
                metric_name,
                window.consecutive_count,
            )
            if window.consecutive_count >= ROLLBACK_CONSECUTIVE_BREACHES:
                return self._execute_rollback(metric_name, window)
        else:
            if window.consecutive_count > 0:
                logger.info(
                    "Rollback eval: %s clean — reset from %d",
                    metric_name,
                    window.consecutive_count,
                )
            window.consecutive_count = 0
            window.last_breached = False

        return None

    # ------------------------------------------------------------------
    # Rollback execution
    # ------------------------------------------------------------------

    def _execute_rollback(self, trigger: str, window: _BreachWindow) -> RollbackReceipt:
        self._rollback_in_progress = True

        previous_config = self._snapshot_config()
        component, action = self._determine_scope(trigger)
        self._apply(component, action)

        metrics_snap: Dict[str, Any] = {}
        if self._metrics is not None:
            try:
                metrics_snap = self._metrics.snapshot()
            except Exception as exc:
                logger.warning("Metrics snapshot failed during rollback: %s", exc)

        receipt = RollbackReceipt(
            timestamp=datetime.now(timezone.utc).isoformat(),
            trigger=trigger,
            breach_count=window.consecutive_count,
            component=component,
            action=action,
            previous_config=previous_config,
            metrics_snapshot=metrics_snap,
        )

        self._persist_receipt(receipt)
        window.consecutive_count = 0
        self._rollback_in_progress = False

        logger.critical(
            "ROLLBACK EXECUTED: trigger=%s component=%s action=%s",
            trigger,
            component,
            action,
        )
        return receipt

    # ------------------------------------------------------------------
    # Scope determination
    # ------------------------------------------------------------------

    def _determine_scope(self, trigger: str) -> Tuple[str, str]:
        """Determine which component to roll back and how.

        Reverse activation order: HMM -> GoT -> Search -> hard kill.
        """
        hmm_pct = self._env_int("BIZRA_PHASE46_HMM_PERCENT")
        got_pct = self._env_int("BIZRA_PHASE46_GOT_BRIDGE_PERCENT")
        search_pct = self._env_int("BIZRA_PHASE46_SEARCH_PERCENT")

        # Component-specific triggers target that component first
        if trigger == "hmm_confidence" and hmm_pct > 0:
            return ("hmm", "percent_zero")
        if trigger == "got_fallback_rate" and got_pct > 0:
            return ("got_bridge", "percent_zero")
        if trigger == "search_error_rate" and search_pct > 0:
            return ("search", "percent_zero")

        # Cross-cutting: reverse activation order
        if hmm_pct > 0:
            return ("hmm", "percent_zero")
        if got_pct > 0:
            return ("got_bridge", "percent_zero")
        if search_pct > 0:
            return ("search", "percent_zero")

        return ("all", "hard_kill")

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def _apply(self, component: str, action: str) -> None:
        if action == "percent_zero":
            env_map = {
                "search": "BIZRA_PHASE46_SEARCH_PERCENT",
                "got_bridge": "BIZRA_PHASE46_GOT_BRIDGE_PERCENT",
                "hmm": "BIZRA_PHASE46_HMM_PERCENT",
            }
            key = env_map.get(component)
            if key:
                os.environ[key] = "0"
                logger.info("Rollback: set %s=0", key)
        elif action == "hard_kill":
            for key in (
                "BIZRA_PHASE46_SEARCH_ENABLED",
                "BIZRA_PHASE46_GOT_BRIDGE_ENABLED",
                "BIZRA_PHASE46_HMM_ENABLED",
                "BIZRA_PHASE46_SEARCH_PERCENT",
                "BIZRA_PHASE46_GOT_BRIDGE_PERCENT",
                "BIZRA_PHASE46_HMM_PERCENT",
            ):
                os.environ[key] = "0"
            logger.critical("Rollback: HARD KILL — all Phase 46 disabled")

    # ------------------------------------------------------------------
    # Receipt persistence
    # ------------------------------------------------------------------

    def _persist_receipt(self, receipt: RollbackReceipt) -> None:
        filename = f"rollback_{receipt.timestamp.replace(':', '-')}.json"
        path = self._receipt_dir / filename
        path.write_text(json.dumps(asdict(receipt), indent=2, default=str))
        logger.info("Rollback receipt: %s", path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _env_int(key: str) -> int:
        try:
            return int(os.getenv(key, "0"))
        except (ValueError, TypeError):
            return 0

    @staticmethod
    def _snapshot_config() -> Dict[str, str]:
        keys = [
            "BIZRA_PHASE46_SEARCH_ENABLED",
            "BIZRA_PHASE46_SEARCH_PERCENT",
            "BIZRA_PHASE46_GOT_BRIDGE_ENABLED",
            "BIZRA_PHASE46_GOT_BRIDGE_PERCENT",
            "BIZRA_PHASE46_HMM_ENABLED",
            "BIZRA_PHASE46_HMM_PERCENT",
        ]
        return {k: os.getenv(k, "unset") for k in keys}

    @property
    def status(self) -> Dict[str, Any]:
        """Current rollback engine status."""
        return {
            "rollback_in_progress": self._rollback_in_progress,
            "breach_windows": {
                name: {
                    "consecutive": w.consecutive_count,
                    "last_breached": w.last_breached,
                }
                for name, w in self._breach_windows.items()
            },
            "receipts_dir": str(self._receipt_dir),
        }
