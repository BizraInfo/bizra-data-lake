"""
FATE Boundary — Constitutional Crossing Gate at PAT↔URP Membrane
=================================================================
Enforces FATE validation whenever PAT agents request URP resources
(knowledge submission, receipt minting, resource access).

Standing on Giants: Membrane Computing (Paun) + Capability Security (Miller)
Constitutional Constraint: Ihsan >= 0.95
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class CrossingResult:
    """Result of a FATE boundary crossing check."""

    allowed: bool
    direction: str  # "pat_to_urp" | "urp_to_pat"
    agent_id: str = ""
    fate_verdict: str = ""  # PASS | FAIL | DEGRADED
    ihsan_score: float = 0.0
    evidence_valid: bool = True
    reason: str = ""
    elapsed_ms: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "direction": self.direction,
            "agent_id": self.agent_id,
            "fate_verdict": self.fate_verdict,
            "ihsan_score": self.ihsan_score,
            "evidence_valid": self.evidence_valid,
            "reason": self.reason,
            "elapsed_ms": self.elapsed_ms,
            "timestamp": self.timestamp,
        }


class FATEBoundary:
    """
    Constitutional membrane gate between PAT agents and URP resources.

    Every crossing (PAT→URP or URP→PAT) passes through:
      1. Ihsan floor check (>= 0.95)
      2. Evidence audit (if FATE gate is available)
      3. Receipt logging

    Degraded mode: If FATE gate is unavailable, allows crossing with
    degraded verdict (logged, not blocked).

    Usage:
        boundary = FATEBoundary(
            fate_gate=runtime._fate_gate,
            ihsan_threshold=0.95,
            receipt_dir=Path("sovereign_state/receipts"),
        )
        result = await boundary.check_crossing(agent_id="PAT-1", content="...")
    """

    def __init__(
        self,
        fate_gate: Optional[Any] = None,
        ihsan_threshold: float = 0.95,
        receipt_dir: Optional[Path] = None,
    ):
        self._fate_gate = fate_gate
        self._ihsan_threshold = ihsan_threshold
        self._receipt_dir = receipt_dir

        # Metrics
        self._crossings_total = 0
        self._crossings_allowed = 0
        self._crossings_blocked = 0
        self._crossings_degraded = 0
        self._prev_hash = "0" * 64

    async def check_crossing(
        self,
        agent_id: str = "",
        content: str = "",
        direction: str = "pat_to_urp",
        ihsan_score: float = 0.95,
        evidence_refs: Optional[list[str]] = None,
    ) -> bool:
        """
        Check if a crossing is allowed. Returns True if allowed.

        This is the fast path — callers get a bool.
        Use check_crossing_detailed() for full CrossingResult.
        """
        result = await self.check_crossing_detailed(
            agent_id=agent_id,
            content=content,
            direction=direction,
            ihsan_score=ihsan_score,
            evidence_refs=evidence_refs,
        )
        return result.allowed

    async def check_crossing_detailed(
        self,
        agent_id: str = "",
        content: str = "",
        direction: str = "pat_to_urp",
        ihsan_score: float = 0.95,
        evidence_refs: Optional[list[str]] = None,
    ) -> CrossingResult:
        """Full crossing check with detailed result."""
        t0 = time.monotonic()
        self._crossings_total += 1

        # 1. Ihsan floor check
        if ihsan_score < self._ihsan_threshold:
            elapsed = (time.monotonic() - t0) * 1000
            self._crossings_blocked += 1
            result = CrossingResult(
                allowed=False,
                direction=direction,
                agent_id=agent_id,
                fate_verdict="FAIL",
                ihsan_score=ihsan_score,
                reason=(
                    f"Ihsan {ihsan_score:.2f} < " f"threshold {self._ihsan_threshold}"
                ),
                elapsed_ms=elapsed,
            )
            self._emit_crossing_receipt(result)
            return result

        # 2. FATE gate check (if available)
        if self._fate_gate is not None:
            try:
                validate_fn = getattr(self._fate_gate, "validate_with_evidence", None)
                if validate_fn and callable(validate_fn):
                    pat_output = {
                        "agent_id": agent_id,
                        "content": content,
                        "evidence_refs": evidence_refs or [],
                        "ihsan_score": ihsan_score,
                    }
                    fate_result = validate_fn(pat_output)
                    passed = getattr(fate_result, "passed", False)
                    verdict_str = str(getattr(fate_result, "verdict", "UNKNOWN"))
                    evidence_audit = getattr(fate_result, "evidence_audit", None)
                    evidence_valid = True
                    if evidence_audit:
                        evidence_valid = getattr(evidence_audit, "valid", True)

                    elapsed = (time.monotonic() - t0) * 1000
                    if passed:
                        self._crossings_allowed += 1
                    else:
                        self._crossings_blocked += 1

                    result = CrossingResult(
                        allowed=passed,
                        direction=direction,
                        agent_id=agent_id,
                        fate_verdict=verdict_str,
                        ihsan_score=ihsan_score,
                        evidence_valid=evidence_valid,
                        reason=(
                            "FATE gate passed"
                            if passed
                            else f"FATE gate blocked: {verdict_str}"
                        ),
                        elapsed_ms=elapsed,
                    )
                    self._emit_crossing_receipt(result)
                    return result
            except (RuntimeError, TypeError, ValueError, OSError) as e:
                logger.warning(f"FATE gate check failed, degrading: {e}")

        # 3. Degraded mode — FATE unavailable, allow with warning
        elapsed = (time.monotonic() - t0) * 1000
        self._crossings_allowed += 1
        self._crossings_degraded += 1
        result = CrossingResult(
            allowed=True,
            direction=direction,
            agent_id=agent_id,
            fate_verdict="DEGRADED",
            ihsan_score=ihsan_score,
            reason="FATE gate unavailable — degraded pass",
            elapsed_ms=elapsed,
        )
        self._emit_crossing_receipt(result)
        return result

    def _emit_crossing_receipt(self, result: CrossingResult) -> None:
        """Emit a receipt for a crossing event."""
        if not self._receipt_dir:
            return
        try:
            import blake3

            content = json.dumps(result.to_dict(), sort_keys=True).encode()
            receipt_hash = blake3.blake3(self._prev_hash.encode() + content).hexdigest()
            self._prev_hash = receipt_hash

            self._receipt_dir.mkdir(parents=True, exist_ok=True)
            receipt_path = (
                self._receipt_dir
                / f"fate_crossing_{result.timestamp.replace(':', '-')}.json"
            )
            receipt_doc = {
                "event": "fate_boundary_crossing",
                "receipt_hash": receipt_hash,
                **result.to_dict(),
            }
            receipt_path.write_text(json.dumps(receipt_doc, indent=2))
        except (ImportError, OSError, ValueError) as e:
            logger.debug(f"Crossing receipt emit failed: {e}")

    def get_status(self) -> Dict[str, Any]:
        return {
            "crossings_total": self._crossings_total,
            "crossings_allowed": self._crossings_allowed,
            "crossings_blocked": self._crossings_blocked,
            "crossings_degraded": self._crossings_degraded,
            "fate_gate_available": self._fate_gate is not None,
            "ihsan_threshold": self._ihsan_threshold,
        }
