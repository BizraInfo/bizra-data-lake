"""
Gate Result — Data model for SAT-5 Genesis Gate checks.

Standing on Giants:
- Shannon (1948): Information as measurable quantity
- Dijkstra (1972): Structured verification over ad-hoc testing

Zero external dependencies. This is the foundation for all gate modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class CheckStatus(Enum):
    """Status of an individual gate check."""

    PASS = "PASS"
    FAIL = "FAIL"
    PARTIAL = "PARTIAL"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    SKIPPED = "SKIPPED"


@dataclass
class CheckResult:
    """Result of a single gate check."""

    name: str
    status: CheckStatus
    evidence: str = ""
    is_manual: bool = False

    @property
    def passed(self) -> bool:
        return self.status == CheckStatus.PASS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "passed": self.passed,
            "evidence": self.evidence,
            "is_manual": self.is_manual,
        }


@dataclass
class GateResult:
    """Result of an entire gate layer (one SAT agent)."""

    agent: str
    layer: str
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Gate passes if all non-skipped checks pass."""
        for c in self.checks:
            if c.status == CheckStatus.FAIL:
                return False
        return True

    @property
    def verdict(self) -> CheckStatus:
        if self.passed:
            return CheckStatus.PASS
        return CheckStatus.FAIL

    @property
    def failed(self) -> List[CheckResult]:
        return [c for c in self.checks if c.status == CheckStatus.FAIL]

    @property
    def stats(self) -> Dict[str, int]:
        counts: Dict[str, int] = {
            "pass": 0,
            "fail": 0,
            "partial": 0,
            "not_impl": 0,
            "skipped": 0,
        }
        for c in self.checks:
            if c.status == CheckStatus.PASS:
                counts["pass"] += 1
            elif c.status == CheckStatus.FAIL:
                counts["fail"] += 1
            elif c.status == CheckStatus.PARTIAL:
                counts["partial"] += 1
            elif c.status == CheckStatus.NOT_IMPLEMENTED:
                counts["not_impl"] += 1
            elif c.status == CheckStatus.SKIPPED:
                counts["skipped"] += 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "layer": self.layer,
            "passed": self.passed,
            "verdict": self.verdict.value,
            "stats": self.stats,
            "checks": [c.to_dict() for c in self.checks],
        }
