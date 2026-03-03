# Phase 52.4: Action Execution (Phase 3 -- Telescript + AHK)

> Standing on Giants: General Magic (Telescript mobile agents with permits, 1994) · Boyd (OODA loop for action-perception, 1976) · Nakamoto (receipt as proof of work, 2008) · Shannon (CPVA as information cost, 1948) · Al-Ghazali (permit as ethical gate, 1095)

## 1. Overview

After Ahmed approves the plan, the ActionExecutor takes over. For each sub-task it
runs a 5-step pipeline: check permit, invoke Telescript via AHK bridge, compute
CPVA (Cost Per Verified Action), generate a signed receipt, and reinforce memory
in Engram. Every action is gated by the FATE pipeline (Ihsan, ADL, SNR) and
scoped by a budget-constrained permit.

Cross-reference: [Phase 48](../phase_48_ahk_hda_desktop_automation.md) for AHK
backend, [Phase 50](../phase_50_telescript_mobile_agents.md) for Telescript primitives.

---

## 2. Data Flow

```
  PlanResponse (approved by Ahmed)
       │
  ┌────▼─────────────────────────────────────────────┐
  │  ACTION EXECUTOR                                   │
  │                                                    │
  │  for each task in plan.tasks (topological order):  │
  │                                                    │
  │    1. check_permit(task, permit)                   │
  │    2. fate_gate(task)                              │
  │    3. invoke_telescript(task) → AHK bridge         │
  │    4. compute_cpva(elapsed, tokens)                │
  │    5. generate_receipt(task, result, cpva)          │
  │    6. reinforce_memory(task, receipt)               │
  │                                                    │
  │  If any task fails:                                │
  │    execute rollback for that task + all dependents │
  └────┬─────────────────────────────────────────────┘
       │
       ▼ receipts[] → ReceiptChain (Phase 52.5)
```

---

## 3. Pseudocode

### 3.1 ActionExecutor

```python
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

from core.integration.constants import (
    IHSAN_THRESHOLD,
    SNR_THRESHOLD,
    ADL_GINI_THRESHOLD,
    TIMESCALE_T1_CYCLE_MS,
)

logger = logging.getLogger("bizra.executor")


@dataclass
class ExecutionResult:
    """Result of a single task execution."""
    task_id: str
    success: bool
    elapsed_ms: float = 0.0
    cpva_usd: float = 0.0
    pre_hash: str = ""
    post_hash: str = ""
    error: Optional[str] = None


class ActionExecutor:
    """Executes approved plan tasks via AHK bridge with full audit trail."""

    def __init__(self, ahk: AHKBridgeClient, permit: HDAPermit,
                 guardian: GuardianCouncil, receipt_chain: ReceiptChain,
                 engram: EngramStore) -> None:
        self.ahk = ahk
        self.permit = permit
        self.guardian = guardian
        self.receipt_chain = receipt_chain
        self.engram = engram
        self.perception = PerceptionActionLoop(ahk)

    async def execute_plan(self, plan: PlanResponse) -> list[ExecutionResult]:
        """Execute all tasks in order. Rollback on failure."""
        results: list[ExecutionResult] = []
        completed: list[dict] = []

        for task_dict in plan.tasks:
            result = await self._execute_single(task_dict)
            results.append(result)
            if result.success:
                completed.append(task_dict)
            else:
                logger.error("task.failed id=%s err=%s", task_dict["taskId"], result.error)
                await self._rollback(completed)
                break
        return results

    async def _execute_single(self, task: dict) -> ExecutionResult:
        """Execute one task through the 5-step pipeline."""
        task_id = task["taskId"]
        start = time.monotonic()

        # Step 1: Permit check
        verb = HDA_VERBS.get(task.get("verb", "file_open"))
        if verb is None:
            return ExecutionResult(task_id=task_id, success=False, error="unknown_verb")

        ok, reason = self._check_permit(verb, task.get("estimatedCostUsd", 0.01))
        if not ok:
            return ExecutionResult(task_id=task_id, success=False, error=f"permit:{reason}")

        # Step 2: FATE gate
        gate_ok = await self.guardian.check_fate(ihsan=task.get("confidence", 0.0), snr=0.90)
        if not gate_ok:
            return ExecutionResult(task_id=task_id, success=False, error="fate_gate_blocked")

        # Step 3: AHK invoke with perception-action loop
        response, pre_hash, post_hash = await self.perception.execute_with_verification(
            verb=verb.name, params=task.get("parameters", {}),
            expect_state_change=verb.is_mutating)

        elapsed_ms = (time.monotonic() - start) * 1000
        if not response.success:
            return ExecutionResult(task_id=task_id, success=False, elapsed_ms=elapsed_ms,
                                   pre_hash=pre_hash, post_hash=post_hash, error=str(response.error))

        # Step 4: Compute CPVA
        cpva = self._compute_cpva(elapsed_ms, task)
        self.permit.spend(cpva)

        # Step 5: Generate receipt
        await self.receipt_chain.append(
            action_id=task_id, action_type=task.get("action", "unknown"),
            description=task.get("description", ""), ihsan_score=task.get("confidence", 0.0),
            cpva_usd=cpva, pre_hash=pre_hash, post_hash=post_hash)

        # Step 6: Reinforce memory
        await self.engram.store_episode(task_id=task_id, description=task.get("description", ""),
                                         result="success", cpva=cpva)

        return ExecutionResult(task_id=task_id, success=True, elapsed_ms=elapsed_ms,
                               cpva_usd=cpva, pre_hash=pre_hash, post_hash=post_hash)

    def _check_permit(self, verb: HDAVerb, cost: float) -> tuple[bool, str]:
        if self.permit.is_expired(): return False, "expired"
        for cap in verb.capabilities_required:
            if not self.permit.has_capability(cap): return False, f"missing:{cap}"
        if not self.permit.has_budget(cost): return False, "insufficient_budget"
        return True, "ok"

    def _compute_cpva(self, elapsed_ms: float, task: dict) -> float:
        """CPVA = elapsed_time * power_cost + token_cost."""
        power_cost_per_ms = 0.0000015   # ~$5.40/hr GPU at 200W
        token_cost = task.get("estimatedCostUsd", 0.005)
        return round(elapsed_ms * power_cost_per_ms + token_cost, 6)

    async def _rollback(self, completed: list[dict]) -> None:
        """Rollback completed tasks in reverse order."""
        for task in reversed(completed):
            rollback = task.get("rollback")
            if rollback and rollback.get("action") != "noop":
                logger.info("rollback task=%s", task["taskId"])
                await self.ahk.invoke("file_open", rollback)
```

### 3.2 Telescript Envelope

```python
@dataclass
class TelescriptEnvelope:
    """Wraps a Telescript action with execution metadata and rollback.
    Cross-reference: Phase 50, Telescript primitives."""
    envelope_id: str = field(default_factory=lambda: str(uuid4()))
    action: TelescriptAction = field(default_factory=TelescriptAction)
    permit_id: str = ""
    profile_id: str = ""
    timestamp: float = field(default_factory=time.time)
    rollback_action: Optional[dict] = None
    max_retries: int = 1
    timeout_ms: int = 30000

    def to_ahk_request(self) -> dict:
        return {"method": self.action.verb,
                "params": {**self.action.parameters,
                           "envelope_id": self.envelope_id,
                           "permit_id": self.permit_id}}
```

---

## 4. Ahmed's Execution (Concrete)

```
Task A: Extract vendor names from 200 PDFs
  permit: capabilities=["filesystem"] budget=$0.50 → OK
  fate_gate: ihsan=0.91 snr=0.90 → PASS
  ahk("file_open", {action: "ocr_extract", path: "*.pdf"})
  elapsed: 7200ms, cpva: $0.021, receipt: A appended

Task D: Move 200 PDFs to vendor/month folders
  permit: budget=$0.474 → OK
  perception_pre: 7a3f...b2c1
  ahk("file_open", {action: "move", ...}) x200
  perception_post: d1e9...4f8a (CHANGED)
  elapsed: 7500ms, cpva: $0.016, receipt: D appended

Total: 6 tasks, CPVA: $0.072
```

---

## 5. TDD Anchors

```python
import pytest

class TestActionExecution:
    """Phase 52.4: Action execution tests."""

    def test_permit_check_valid(self):
        executor = make_executor(capabilities=["filesystem"], budget=1.0)
        ok, _ = executor._check_permit(HDA_VERBS["file_open"], 0.01)
        assert ok is True

    def test_permit_check_missing_capability(self):
        executor = make_executor(capabilities=["keyboard"], budget=1.0)
        ok, reason = executor._check_permit(HDA_VERBS["file_open"], 0.01)
        assert ok is False and "missing" in reason

    def test_permit_check_insufficient_budget(self):
        executor = make_executor(capabilities=["filesystem"], budget=0.001)
        ok, _ = executor._check_permit(HDA_VERBS["file_open"], 0.01)
        assert ok is False

    @pytest.mark.asyncio
    async def test_telescript_invoke(self):
        executor = make_executor_with_mock_ahk(success=True)
        result = await executor._execute_single(mock_task_dict())
        assert result.success is True and result.cpva_usd > 0

    def test_cpva_compute(self):
        executor = make_executor()
        cpva = executor._compute_cpva(1000.0, {"estimatedCostUsd": 0.01})
        assert 0.01 < cpva < 0.02

    @pytest.mark.asyncio
    async def test_receipt_generated(self):
        executor = make_executor_with_mock_ahk(success=True)
        await executor._execute_single(mock_task_dict())
        assert executor.receipt_chain.length() > 1

    @pytest.mark.asyncio
    async def test_rollback_on_failure(self):
        executor = make_executor_with_mock_ahk(success=False)
        results = await executor.execute_plan(mock_plan_with_failing_task())
        assert any(not r.success for r in results)

    def test_telescript_envelope_to_ahk(self):
        env = TelescriptEnvelope(
            action=TelescriptAction(action_type="fs", verb="file_open",
                                    parameters={"path": "/test"}),
            permit_id="p-001")
        ahk = env.to_ahk_request()
        assert ahk["method"] == "file_open"
        assert ahk["params"]["permit_id"] == "p-001"

    @pytest.mark.asyncio
    async def test_fate_gate_blocks_low_ihsan(self):
        executor = make_executor_with_mock_ahk(success=True)
        result = await executor._execute_single(mock_task_dict(confidence=0.50))
        assert result.success is False and "fate_gate" in result.error
```
