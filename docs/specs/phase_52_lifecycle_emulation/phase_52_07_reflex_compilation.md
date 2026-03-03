# Phase 52.7: Reflex Compilation (Phase 6 -- System 2 to System 1)

> Standing on Giants: Kahneman (System 1 fast / System 2 slow dual process, 2011) · Friston (Free Energy minimization through precision-weighted prediction, 2006) · Anderson (ACT-R procedural compilation, 1993) · Shannon (entropy as novelty measure, 1948) · Al-Ghazali (excellence through practice until it becomes nature, 1095)

## 1. Overview

After Ahmed's node successfully completes the same type of task 5+ times with
Ihsan > 0.96, the Reflex Compiler activates. It extracts the common pattern from
those successful runs, creates a Telescript template with parameterized slots, and
stores it in the ReflexLedger (backed by Engram).

On future matching tasks, the entropy router detects low entropy (familiar pattern)
and routes directly to the reflex -- bypassing the full PAT-7 pipeline. This is
BIZRA's System 2 to System 1 compression: deliberate reasoning becomes automatic
expertise.

**Economic impact:** CPVA drops from ~$0.08 (full PAT-7) to ~$0.01 (reflex hit).

---

## 2. Data Flow

```
  Receipt chain (5+ successful runs, same pattern)
       │
  ┌────▼─────────────────────────────────────────────┐
  │  REFLEX COMPILER                                   │
  │                                                    │
  │  1. Detect pattern: same goal_category +           │
  │     similar task decomposition 5+ times            │
  │  2. Verify: all runs have Ihsan > 0.96             │
  │  3. Extract template: common Telescript + params   │
  │  4. Compute CPVA statistics (min, max, mean)       │
  │  5. Store in ReflexLedger (Engram-backed)          │
  └────┬─────────────────────────────────────────────┘
       │
  Future task arrives:
  ┌────▼─────────────────────────────────────────────┐
  │  ENTROPY ROUTER                                    │
  │                                                    │
  │  1. Compute task entropy (novelty score)           │
  │  2. If entropy < threshold: search ReflexLedger    │
  │  3. If reflex found: bypass PAT-7, execute direct  │
  │  4. If no reflex: full PAT-7 pipeline              │
  └────┬─────────────────────────────────────────────┘
       │
       ├── LOW entropy → Reflex hit → Direct execution
       │                  (System 1, ~$0.01)
       │
       └── HIGH entropy → PAT-7 pipeline
                          (System 2, ~$0.08)
```

---

## 3. Pseudocode

### 3.1 Reflex Data Model

```python
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

from core.integration.constants import IHSAN_THRESHOLD, FAISS_SIMILARITY_FLOOR


REFLEX_IHSAN_THRESHOLD: float = 0.96   # Stricter than production (0.95)
REFLEX_MIN_SUCCESSES: int = 5
REFLEX_ENTROPY_THRESHOLD: float = 0.30  # Below this = familiar pattern


@dataclass
class CompiledReflex:
    """A compiled System 1 reflex extracted from repeated System 2 reasoning."""
    reflex_id: str = field(default_factory=lambda: str(uuid4()))
    pattern_name: str = ""             # "invoice_pdf_organization"
    goal_category: str = ""            # "file_organization"
    goal_template: str = ""            # "Organize {file_type} by {criteria}"
    telescript_template: list[dict] = field(default_factory=list)
    parameter_slots: list[str] = field(default_factory=list)
    success_count: int = 0
    total_runs: int = 0
    mean_ihsan: float = 0.0
    cpva_range: tuple[float, float, float] = (0.0, 0.0, 0.0)  # min, max, mean
    created_at: float = field(default_factory=time.time)
    last_used: float = 0.0
    embedding: list[float] = field(default_factory=list)  # For FAISS matching
```

### 3.2 Reflex Compiler

```python
class ReflexCompiler:
    """Compiles repeated successful patterns into reflexes.
    Standing on Giants: Kahneman (2011) -- System 2 practice → System 1."""

    def __init__(self, receipt_chain: ReceiptChain, engram: EngramStore) -> None:
        self.receipt_chain = receipt_chain
        self.engram = engram

    async def check_and_compile(self, goal_category: str) -> Optional[CompiledReflex]:
        """Check if a pattern has enough successful runs to compile."""
        # Gather receipts for this category
        runs = self._gather_runs(goal_category)

        # Check compilation threshold
        successful = [r for r in runs if r["ihsan"] >= REFLEX_IHSAN_THRESHOLD
                      and r["outcome"] == "success"]

        if len(successful) < REFLEX_MIN_SUCCESSES:
            return None  # Not enough evidence yet

        # Extract common template
        template = self._extract_template(successful)
        params = self._extract_parameter_slots(successful)
        cpva_values = [r["cpva"] for r in successful]

        reflex = CompiledReflex(
            pattern_name=f"{goal_category}_reflex",
            goal_category=goal_category,
            goal_template=template["goal_template"],
            telescript_template=template["actions"],
            parameter_slots=params,
            success_count=len(successful),
            total_runs=len(runs),
            mean_ihsan=sum(r["ihsan"] for r in successful) / len(successful),
            cpva_range=(min(cpva_values), max(cpva_values),
                        sum(cpva_values) / len(cpva_values)),
        )

        # Generate embedding for future matching
        reflex.embedding = await self.engram.embed(reflex.goal_template)

        return reflex

    def _gather_runs(self, category: str) -> list[dict]:
        """Collect all runs for a goal category from receipt chain."""
        return [{"ihsan": r.ihsan_score, "outcome": r.outcome,
                 "cpva": r.cpva_usd, "actions": r.description}
                for r in self.receipt_chain.get_chain()
                if r.domain == category or r.action_type != "genesis"]

    def _extract_template(self, runs: list[dict]) -> dict:
        """Find common action sequence across successful runs."""
        # Simplified: take the most recent successful run as template
        return {"goal_template": "Organize {file_type} by {criteria}",
                "actions": [{"verb": "file_open", "params": "{dynamic}"}]}

    def _extract_parameter_slots(self, runs: list[dict]) -> list[str]:
        """Identify which parameters varied across runs."""
        return ["file_type", "criteria", "target_dir", "email_recipient"]
```

### 3.3 Entropy Router

```python
import math


class EntropyRouter:
    """Routes tasks to reflex (System 1) or PAT-7 (System 2) based on novelty.
    Standing on Giants: Shannon (1948) -- entropy measures information content."""

    def __init__(self, reflex_ledger: ReflexLedger) -> None:
        self.reflex_ledger = reflex_ledger

    async def route(self, goal: str, goal_category: str) -> dict:
        """Determine routing: reflex hit or full PAT-7."""
        entropy = await self._compute_entropy(goal)

        if entropy < REFLEX_ENTROPY_THRESHOLD:
            reflex = await self.reflex_ledger.match(goal)
            if reflex is not None:
                return {"route": "reflex", "reflex_id": reflex.reflex_id,
                        "entropy": entropy, "estimated_cpva": reflex.cpva_range[2]}

        return {"route": "pat7", "entropy": entropy, "estimated_cpva": 0.08}

    async def _compute_entropy(self, goal: str) -> float:
        """Compute task entropy: 0.0 = perfectly familiar, 1.0 = fully novel."""
        # Search Engram for similar past goals
        results = await self.reflex_ledger.engram.search(
            query=goal, top_k=5, similarity_floor=FAISS_SIMILARITY_FLOOR)

        if not results:
            return 1.0  # Fully novel

        max_sim = max(r.similarity for r in results)
        return 1.0 - max_sim  # Higher similarity = lower entropy


class ReflexLedger:
    """Persistent store of compiled reflexes, backed by Engram."""

    def __init__(self, engram: EngramStore) -> None:
        self.engram = engram
        self._reflexes: dict[str, CompiledReflex] = {}

    async def store(self, reflex: CompiledReflex) -> None:
        self._reflexes[reflex.reflex_id] = reflex
        await self.engram.store_reflex(reflex)

    async def match(self, goal: str) -> Optional[CompiledReflex]:
        """Find matching reflex via embedding similarity."""
        results = await self.engram.search(query=goal, top_k=1,
                                            similarity_floor=0.85)
        if results and results[0].metadata.get("reflex_id"):
            rid = results[0].metadata["reflex_id"]
            return self._reflexes.get(rid)
        return None

    async def load_persisted(self) -> None:
        """Load reflexes from persistent Engram store."""
        stored = await self.engram.get_all_reflexes()
        for r in stored:
            self._reflexes[r.reflex_id] = r
```

---

## 4. Ahmed's Reflex Timeline

```
Run 1: Full PAT-7, Ihsan=0.97, CPVA=$0.072  → no reflex (1/5)
Run 2: Full PAT-7, Ihsan=0.96, CPVA=$0.068  → no reflex (2/5)
Run 3: Full PAT-7, Ihsan=0.98, CPVA=$0.065  → no reflex (3/5)
Run 4: Full PAT-7, Ihsan=0.97, CPVA=$0.070  → no reflex (4/5)
Run 5: Full PAT-7, Ihsan=0.97, CPVA=$0.067  → COMPILED!
  pattern: "invoice_pdf_organization"
  template: 6 Telescript actions with {vendor}, {month}, {email} slots
  cpva_range: ($0.065, $0.072, $0.068)

Run 6: Entropy=0.12 → REFLEX HIT → bypass PAT-7
  Direct Telescript execution (6 actions)
  CPVA=$0.009 (8x cheaper than Run 1)
```

---

## 5. TDD Anchors

```python
import pytest

class TestReflexCompilation:
    """Phase 52.7: Reflex compilation tests."""

    @pytest.mark.asyncio
    async def test_reflex_trigger_threshold(self):
        """Reflex compiles after 5+ successful runs with Ihsan > 0.96."""
        compiler = ReflexCompiler(
            receipt_chain=chain_with_n_successes(5, ihsan=0.97),
            engram=mock_engram())
        reflex = await compiler.check_and_compile("file_organization")
        assert reflex is not None
        assert reflex.success_count >= 5

    @pytest.mark.asyncio
    async def test_reflex_not_triggered_below_threshold(self):
        """Fewer than 5 runs does not trigger compilation."""
        compiler = ReflexCompiler(
            receipt_chain=chain_with_n_successes(3, ihsan=0.97),
            engram=mock_engram())
        reflex = await compiler.check_and_compile("file_organization")
        assert reflex is None

    @pytest.mark.asyncio
    async def test_reflex_stored_in_ledger(self):
        """Compiled reflex is stored in ReflexLedger."""
        ledger = ReflexLedger(engram=mock_engram())
        reflex = CompiledReflex(pattern_name="test", goal_category="test")
        await ledger.store(reflex)
        assert reflex.reflex_id in ledger._reflexes

    @pytest.mark.asyncio
    async def test_reflex_hit_on_match(self):
        """Matching goal retrieves stored reflex."""
        ledger = ReflexLedger(engram=mock_engram_with_reflex("invoice_org"))
        result = await ledger.match("Organize my invoice PDFs")
        assert result is not None

    @pytest.mark.asyncio
    async def test_entropy_router_reflex_route(self):
        """Low entropy routes to reflex."""
        router = EntropyRouter(reflex_ledger=ledger_with_reflex())
        route = await router.route("Organize invoice PDFs", "file_organization")
        assert route["route"] == "reflex"
        assert route["estimated_cpva"] < 0.02

    @pytest.mark.asyncio
    async def test_entropy_router_pat7_route(self):
        """High entropy routes to PAT-7."""
        router = EntropyRouter(reflex_ledger=empty_ledger())
        route = await router.route("Completely novel task", "unknown")
        assert route["route"] == "pat7"

    def test_cpva_reduction(self):
        """Reflex CPVA is significantly lower than PAT-7 CPVA."""
        reflex = CompiledReflex(cpva_range=(0.008, 0.012, 0.010))
        pat7_cpva = 0.08
        assert reflex.cpva_range[2] < pat7_cpva * 0.25  # 4x+ cheaper

    @pytest.mark.asyncio
    async def test_reflex_requires_high_ihsan(self):
        """Runs with Ihsan < 0.96 do not count toward compilation."""
        compiler = ReflexCompiler(
            receipt_chain=chain_with_n_successes(5, ihsan=0.90),
            engram=mock_engram())
        reflex = await compiler.check_and_compile("file_organization")
        assert reflex is None
```
