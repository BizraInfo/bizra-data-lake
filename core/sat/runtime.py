"""
SAT Runtime Daemon — System Agentic Team Validation Loop
=========================================================
Runs SAT-5 agents in a continuous validation loop, processing
incoming receipts through the 6 constitutional gates.

Standing on Giants: BFT Consensus (Lamport) + Gate Pattern (GoF)
Constitutional Constraint: Ihsan >= 0.95
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationRequest:
    """A receipt or output to validate through SAT gates."""

    request_id: str
    content: str
    source_agent_id: str = ""
    evidence_refs: List[str] = field(default_factory=list)
    ihsan_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class GateVerdict:
    """Result of a single SAT gate check."""

    gate_name: str
    passed: bool
    score: float = 0.0
    reason: str = ""
    elapsed_ms: float = 0.0


@dataclass
class ValidationResult:
    """Aggregate result of SAT-5 validation."""

    request_id: str
    passed: bool
    gate_verdicts: List[GateVerdict] = field(default_factory=list)
    aggregate_score: float = 0.0
    receipt_hash: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def gates_passed(self) -> int:
        return sum(1 for g in self.gate_verdicts if g.passed)

    @property
    def gates_total(self) -> int:
        return len(self.gate_verdicts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "passed": self.passed,
            "gates_passed": self.gates_passed,
            "gates_total": self.gates_total,
            "aggregate_score": self.aggregate_score,
            "gate_verdicts": [
                {
                    "gate": g.gate_name,
                    "passed": g.passed,
                    "score": g.score,
                    "reason": g.reason,
                }
                for g in self.gate_verdicts
            ],
            "receipt_hash": self.receipt_hash,
            "timestamp": self.timestamp,
        }


class SATRuntime:
    """
    Persistent runtime for SAT-5 system agents.

    Runs 5 SAT agents as gate validators. Incoming receipts or PAT
    outputs pass through all 6 constitutional gates. Any gate failure
    triggers a BLOCKED verdict.

    Gates (loaded dynamically):
        1. Sentinel     — security + integrity
        2. Ambassador   — external interface compliance
        3. Conductor    — operational readiness
        4. Oracle       — knowledge verification
        5. Ledger       — financial/receipt integrity
        6. Provenance   — origin + attribution chain

    Usage:
        runtime = SATRuntime(receipt_dir=Path("sovereign_state/receipts"))
        await runtime.start()
        result = await runtime.validate(ValidationRequest(...))
        await runtime.stop()
    """

    # Gate modules — loaded lazily
    GATE_MODULES = {
        "sentinel": "core.sat.sentinel_gate",
        "ambassador": "core.sat.ambassador_gate",
        "conductor": "core.sat.conductor_gate",
        "oracle": "core.sat.oracle_s_gate",
        "ledger": "core.sat.ledger_gate",
        "provenance": "core.sat.provenance_gate",
    }

    # Verification function names (convention: {name}_verify)
    GATE_FUNCTIONS = {
        "sentinel": "sentinel_verify",
        "ambassador": "ambassador_verify",
        "conductor": "conductor_verify",
        "oracle": "oracle_verify",
        "ledger": "ledger_verify",
        "provenance": "provenance_verify",
    }

    def __init__(
        self,
        agents: Optional[List[Any]] = None,
        receipt_dir: Optional[Path] = None,
        ihsan_threshold: float = 0.95,
    ):
        self._agents: List[Any] = agents or []
        self._receipt_dir = receipt_dir
        self._ihsan_threshold = ihsan_threshold

        self._validation_queue: asyncio.Queue[ValidationRequest] = asyncio.Queue()
        self._running = False
        self._loop_task: Optional[asyncio.Task[Any]] = None
        self._results: Dict[str, ValidationResult] = {}
        self._gates_loaded: Dict[str, Any] = {}
        self._prev_hash = "0" * 64

        # Metrics
        self._validations_processed = 0
        self._validations_passed = 0
        self._validations_failed = 0
        self._receipt_count = 0

    @property
    def is_running(self) -> bool:
        return self._running

    def _load_gates(self) -> None:
        """Load gate verification functions."""
        for gate_name, module_path in self.GATE_MODULES.items():
            func_name = self.GATE_FUNCTIONS[gate_name]
            try:
                mod = __import__(module_path, fromlist=[func_name])
                func = getattr(mod, func_name, None)
                if func and callable(func):
                    self._gates_loaded[gate_name] = func
                    logger.debug(f"SAT gate loaded: {gate_name}")
                else:
                    logger.warning(f"SAT gate function not found: {module_path}.{func_name}")
            except (ImportError, AttributeError) as e:
                logger.warning(f"SAT gate load failed: {gate_name}: {e}")

    def _run_gate(self, gate_name: str, gate_fn: Any) -> GateVerdict:
        """Run a single gate, return verdict."""
        t0 = time.monotonic()
        try:
            result = gate_fn()
            elapsed = (time.monotonic() - t0) * 1000

            # GateResult has .passed, .score, .checks
            passed = getattr(result, "passed", False)
            score = getattr(result, "score", 0.0)
            checks = getattr(result, "checks", [])
            total = len(checks)
            passed_count = sum(1 for c in checks if getattr(c, "passed", False))

            return GateVerdict(
                gate_name=gate_name,
                passed=passed,
                score=score if score else (passed_count / max(total, 1)),
                reason=f"{passed_count}/{total} checks passed",
                elapsed_ms=elapsed,
            )
        except (RuntimeError, OSError, ValueError, TypeError) as e:
            elapsed = (time.monotonic() - t0) * 1000
            return GateVerdict(
                gate_name=gate_name,
                passed=False,
                score=0.0,
                reason=f"Gate error: {e}",
                elapsed_ms=elapsed,
            )

    def _emit_receipt(self, result: ValidationResult) -> str:
        """Emit a BLAKE3-chained receipt for a validation result."""
        try:
            import blake3

            content = json.dumps(result.to_dict(), sort_keys=True).encode()
            receipt_hash = blake3.blake3(
                self._prev_hash.encode() + content
            ).hexdigest()
            self._prev_hash = receipt_hash
            self._receipt_count += 1

            if self._receipt_dir:
                self._receipt_dir.mkdir(parents=True, exist_ok=True)
                receipt_path = (
                    self._receipt_dir
                    / f"sat_validation_{result.request_id}.json"
                )
                receipt_doc = {
                    "event": "sat_validation",
                    "receipt_hash": receipt_hash,
                    **result.to_dict(),
                }
                receipt_path.write_text(json.dumps(receipt_doc, indent=2))

            return receipt_hash
        except ImportError:
            import hashlib

            content = json.dumps(result.to_dict(), sort_keys=True).encode()
            receipt_hash = hashlib.sha256(
                self._prev_hash.encode() + content
            ).hexdigest()
            self._prev_hash = receipt_hash
            return receipt_hash

    async def _process_validation(
        self, request: ValidationRequest
    ) -> ValidationResult:
        """Process a single validation through all gates."""
        verdicts: List[GateVerdict] = []

        # Run all gates
        for gate_name, gate_fn in self._gates_loaded.items():
            verdict = self._run_gate(gate_name, gate_fn)
            verdicts.append(verdict)

        # Ihsan threshold check
        ihsan_verdict = GateVerdict(
            gate_name="ihsan_threshold",
            passed=request.ihsan_score >= self._ihsan_threshold,
            score=request.ihsan_score,
            reason=(
                f"Ihsan {request.ihsan_score:.2f} >= {self._ihsan_threshold}"
                if request.ihsan_score >= self._ihsan_threshold
                else f"Ihsan {request.ihsan_score:.2f} < {self._ihsan_threshold}"
            ),
        )
        verdicts.append(ihsan_verdict)

        all_passed = all(v.passed for v in verdicts)
        avg_score = sum(v.score for v in verdicts) / max(len(verdicts), 1)

        result = ValidationResult(
            request_id=request.request_id,
            passed=all_passed,
            gate_verdicts=verdicts,
            aggregate_score=avg_score,
        )

        result.receipt_hash = self._emit_receipt(result)
        return result

    async def _validation_loop(self) -> None:
        """Main validation processing loop."""
        logger.info(
            f"SAT runtime loop started ({len(self._gates_loaded)} gates loaded)"
        )
        while self._running:
            try:
                request = await asyncio.wait_for(
                    self._validation_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            result = await self._process_validation(request)
            self._results[result.request_id] = result
            self._validations_processed += 1
            if result.passed:
                self._validations_passed += 1
            else:
                self._validations_failed += 1
            logger.debug(
                f"Validation {result.request_id}: "
                f"{'PASS' if result.passed else 'FAIL'} "
                f"({result.gates_passed}/{result.gates_total} gates)"
            )

    async def start(self) -> None:
        """Start the SAT runtime."""
        if self._running:
            return

        self._load_gates()
        logger.info(
            f"SAT Runtime: {len(self._gates_loaded)} gates loaded, "
            f"{len(self._agents)} agents"
        )

        self._running = True
        self._loop_task = asyncio.create_task(
            self._validation_loop(), name="sat_runtime_loop"
        )

    async def stop(self) -> None:
        """Stop the SAT runtime."""
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        logger.info(
            f"SAT Runtime stopped: {self._validations_processed} validations "
            f"({self._validations_passed} passed, {self._validations_failed} failed)"
        )

    async def validate(self, request: ValidationRequest) -> ValidationResult:
        """Submit a validation and wait for result."""
        await self._validation_queue.put(request)
        for _ in range(300):  # 30s max
            await asyncio.sleep(0.1)
            if request.request_id in self._results:
                return self._results[request.request_id]
        return ValidationResult(
            request_id=request.request_id,
            passed=False,
        )

    def validate_fire_and_forget(self, request: ValidationRequest) -> None:
        """Submit a validation without waiting."""
        self._validation_queue.put_nowait(request)

    def get_status(self) -> Dict[str, Any]:
        """Get runtime status."""
        return {
            "running": self._running,
            "gates_loaded": list(self._gates_loaded.keys()),
            "agents_count": len(self._agents),
            "validations_processed": self._validations_processed,
            "validations_passed": self._validations_passed,
            "validations_failed": self._validations_failed,
            "validations_queued": self._validation_queue.qsize(),
            "receipt_count": self._receipt_count,
        }
