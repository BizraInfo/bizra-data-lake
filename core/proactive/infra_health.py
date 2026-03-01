"""
Infrastructure health bridge — exposes guardian probes to Node0 proactive pipeline.

Provides InfraHealthProbe that can be called from ProactiveHarness or
any Node0 component to get real-time infrastructure health status.

Usage:
    from core.proactive.infra_health import InfraHealthProbe

    probe = InfraHealthProbe()
    report = probe.check()           # Quick check, no corrections
    report = probe.check_and_fix()   # Check + auto-correct
    score  = probe.ihsan_score()     # 0.0 - 1.0

Created: 2026-03-01 | BIZRA Infrastructure Health Bridge v1.0
"""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("bizra.proactive.infra_health")

# Add guardian directory to path for import
_GUARDIAN_DIR = Path(__file__).resolve().parents[2] / "scripts" / "guardian"


def _import_guardian() -> Any:
    """Lazy-import infra_guardian to avoid hard dependency."""
    if str(_GUARDIAN_DIR) not in sys.path:
        sys.path.insert(0, str(_GUARDIAN_DIR))
    try:
        return importlib.import_module("infra_guardian")
    except ImportError:
        logger.warning(
            "infra_guardian not found at %s — infra probes disabled", _GUARDIAN_DIR
        )
        return None


class InfraHealthProbe:
    """Bridge between infra_guardian and the Node0 proactive pipeline."""

    def __init__(self) -> None:
        self._guardian = _import_guardian()
        self._last_report: dict[str, Any] | None = None

    @property
    def available(self) -> bool:
        return self._guardian is not None

    def check(self) -> dict[str, Any]:
        """Run all probes without corrections. Returns JSON-serializable report."""
        if not self._guardian:
            return {
                "overall": "UNKNOWN",
                "ihsan": 0.0,
                "error": "guardian not available",
            }

        state = self._guardian.GuardianState()
        results = self._guardian.run_all_probes(correct=False, state=state)
        self._last_report = self._guardian.results_to_report(results)
        state.save()
        return self._last_report

    def check_and_fix(self) -> dict[str, Any]:
        """Run all probes with auto-correction enabled. Returns report."""
        if not self._guardian:
            return {
                "overall": "UNKNOWN",
                "ihsan": 0.0,
                "error": "guardian not available",
            }

        state = self._guardian.GuardianState()
        results = self._guardian.run_all_probes(correct=True, state=state)
        all_ok = all(
            r.severity in (self._guardian.Severity.OK, self._guardian.Severity.FIXED)
            for r in results
        )
        state.record_check(all_ok)
        self._last_report = self._guardian.results_to_report(results)
        state.save()
        return self._last_report

    def ihsan_score(self) -> float:
        """Return the last computed Ihsan score (0.0-1.0). Runs check if needed."""
        if self._last_report is None:
            self.check()
        return self._last_report.get("ihsan", 0.0) if self._last_report else 0.0

    def summary(self) -> str:
        """One-line status summary for Node0 dashboard."""
        if self._last_report is None:
            self.check()
        if not self._last_report:
            return "INFRA: unknown (guardian unavailable)"
        s = self._last_report.get("summary", {})
        overall = self._last_report.get("overall", "?")
        ihsan = self._last_report.get("ihsan", 0)
        return (
            f"INFRA: {overall} | Ihsan {ihsan:.3f} | "
            f"{s.get('ok', 0)} ok, {s.get('warnings', 0)} warn, "
            f"{s.get('critical', 0)} crit, {s.get('auto_fixed', 0)} fixed"
        )
