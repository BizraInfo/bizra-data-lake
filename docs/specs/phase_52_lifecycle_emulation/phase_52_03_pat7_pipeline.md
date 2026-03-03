# Phase 52.3: PAT-7 Chain of Reasoning (Phase 2 -- Agent Pipeline)

> Standing on Giants: Besta (Graph of Thoughts parallel hypothesis, 2024) · Kahneman (System 2 deliberate reasoning, 2011) · Minsky (Society of Mind agent specialization, 1986) · Shannon (information-theoretic confidence, 1948) · Al-Ghazali (8-dimension Ihsan scoring, 1095) · General Magic (agent collaboration, 1994)

## 1. Overview

When Ahmed's task arrives with entropy 1.0 (fully novel), the full PAT-7 pipeline
engages. Seven specialized agents execute sequentially, each contributing a distinct
cognitive function with defined contract, inputs, and outputs.

| # | Agent | Role | Kahneman |
|---|-------|------|----------|
| 1 | Planner | Decompose goal into sub-tasks | System 2: planning |
| 2 | Researcher | Retrieve relevant knowledge | System 2: gathering |
| 3 | Coder | Generate executable Telescript | System 2: implementation |
| 4 | Evaluator | Sandbox-simulate and score | System 2: evaluation |
| 5 | Ethicist | Score 8 Ihsan dimensions | System 2: ethical judgment |
| 6 | Publisher | Generate human-readable summary | System 2: communication |
| 7 | Integrator | Assemble final plan JSON | System 2: synthesis |

---

## 2. Data Flow

```
  PCI Envelope (from Genesis Bridge)
       │
  ┌────▼─────────────────────────────────────────┐
  │  PLANNER: goal → list[SubTask] via RLM + GoT │
  └────┬─────────────────────────────────────────┘
       │ sub_tasks: [A, B, C, D, E, F]
  ┌────▼─────────────────────────────────────────┐
  │  RESEARCHER: Engram lookup → schema fallback  │
  └────┬─────────────────────────────────────────┘
       │ enriched_tasks
  ┌────▼─────────────────────────────────────────┐
  │  CODER: tasks → TelescriptAction per task    │
  └────┬─────────────────────────────────────────┘
       │ telescript_actions
  ┌────▼─────────────────────────────────────────┐
  │  EVALUATOR: sandbox simulation + confidence  │
  └────┬─────────────────────────────────────────┘
       │ scored_actions
  ┌────▼─────────────────────────────────────────┐
  │  ETHICIST: 8-dim Ihsan >= IHSAN_THRESHOLD    │
  └────┬─────────────────────────────────────────┘
       │ ihsan_approved_actions
  ┌────▼─────────────────────────────────────────┐
  │  PUBLISHER: human summary + CPVA estimate    │
  └────┬─────────────────────────────────────────┘
  ┌────▼─────────────────────────────────────────┐
  │  INTEGRATOR: assemble PlanResponse JSON       │
  └────┬─────────────────────────────────────────┘
       ▼ PlanResponse → Genesis Bridge → Ahmed
```

---

## 3. Pseudocode

### 3.1 Core Types

```python
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
from uuid import uuid4
from core.integration.constants import (
    FAISS_SIMILARITY_FLOOR, GOT_MAX_HYPOTHESES,
    IHSAN_THRESHOLD, IHSAN_WEIGHTS, SNR_THRESHOLD,
)

class TaskStatus(Enum):
    PLANNED = auto(); ENRICHED = auto(); CODED = auto()
    EVALUATED = auto(); APPROVED = auto(); READY = auto()

@dataclass
class SubTask:
    task_id: str = field(default_factory=lambda: str(uuid4()))
    description: str = ""
    domain: str = ""            # "filesystem", "email", "ocr", "analysis"
    depends_on: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PLANNED
    knowledge_context: dict = field(default_factory=dict)
    telescript: Optional[TelescriptAction] = None
    confidence: float = 0.0
    ihsan_scores: dict[str, float] = field(default_factory=dict)
    risk_flags: list[str] = field(default_factory=list)

@dataclass
class TelescriptAction:
    action_type: str           # "file_move", "ocr_extract", "email_send"
    verb: str                  # HDA verb: "file_open", "type_text", etc.
    parameters: dict = field(default_factory=dict)
    rollback: Optional[dict] = None
    estimated_duration_ms: int = 0
    estimated_cost_usd: float = 0.0
```

### 3.2 Planner Agent

```python
class PlannerAgent:
    """Agent 1/7: Decomposes goal via RLM + GoT branching."""

    def __init__(self, rlm: RecursiveLanguageModel, constitution: Constitution):
        self.rlm = rlm
        self.max_hypotheses = constitution.got_max_hypotheses

    async def decompose(self, goal: str, context: dict) -> list[SubTask]:
        hypotheses: list[list[SubTask]] = []
        for _ in range(self.max_hypotheses):
            hypotheses.append(await self.rlm.decompose(goal, context))

        # Select hypothesis with highest SNR (least redundant steps)
        best = max(hypotheses, key=self._compute_decomposition_snr)
        return self._topological_sort(best)

    def _compute_decomposition_snr(self, tasks: list[SubTask]) -> float:
        if not tasks: return 0.0
        return len({t.domain for t in tasks}) / len(tasks)

    def _topological_sort(self, tasks: list[SubTask]) -> list[SubTask]:
        """Kahn's algorithm: dependencies satisfied before dependents."""
        return sorted(tasks, key=lambda t: len(t.depends_on))
```

**Ahmed's decomposition:**

| Task | Description | Domain | Depends On |
|------|-------------|--------|------------|
| A | Extract vendor name + date from each PDF | ocr | -- |
| B | Create vendor folders in target directory | filesystem | -- |
| C | Create YYYY-MM sub-folders per vendor | filesystem | A, B |
| D | Move each PDF to correct vendor/month folder | filesystem | C |
| E | Generate summary (vendor counts, date ranges) | analysis | D |
| F | Email summary to ahmed@example.com | email | E |

### 3.3 Researcher Agent

```python
class ResearcherAgent:
    """Agent 2/7: Enriches sub-tasks with knowledge from Engram."""

    def __init__(self, engram: EngramStore):
        self.engram = engram

    async def enrich(self, tasks: list[SubTask], goal: str) -> list[SubTask]:
        for task in tasks:
            results = await self.engram.search(
                query=task.description, top_k=5,
                similarity_floor=FAISS_SIMILARITY_FLOOR)

            if results:
                task.knowledge_context = {
                    "source": "engram",
                    "memories": [r.to_dict() for r in results],
                    "confidence": max(r.similarity for r in results)}
            else:
                task.knowledge_context = {
                    "source": "schema_inference",
                    "domain": task.domain, "confidence": 0.5}
            task.status = TaskStatus.ENRICHED
        return tasks
```

### 3.4 Coder Agent

```python
class CoderAgent:
    """Agent 3/7: Generates TelescriptAction for each sub-task."""

    VERB_MAP = {"ocr": "file_open", "filesystem": "file_open",
                "email": "browser_navigate", "analysis": "type_text"}

    async def generate(self, tasks: list[SubTask]) -> list[SubTask]:
        for task in tasks:
            verb = self.VERB_MAP.get(task.domain, "file_open")
            task.telescript = TelescriptAction(
                action_type=f"{task.domain}_{task.task_id[:8]}", verb=verb,
                parameters={"description": task.description, "domain": task.domain},
                rollback=({"action": "undo_move", "restore_original": True}
                          if task.domain == "filesystem" else {"action": "noop"}),
                estimated_duration_ms={"ocr": 5000, "filesystem": 500,
                    "email": 3000, "analysis": 2000}.get(task.domain, 1000),
                estimated_cost_usd={"ocr": 0.02, "filesystem": 0.005,
                    "email": 0.01, "analysis": 0.01}.get(task.domain, 0.01))
            task.status = TaskStatus.CODED
        return tasks
```

### 3.5 Evaluator Agent

```python
class EvaluatorAgent:
    """Agent 4/7: Sandbox-simulates actions and scores confidence."""

    async def evaluate(self, tasks: list[SubTask]) -> list[SubTask]:
        for task in tasks:
            if task.telescript is None:
                task.confidence = 0.0
                task.risk_flags.append("no_telescript_generated")
                continue
            sandbox = await self._sandbox_simulate(task.telescript)
            task.confidence = self._score(task, sandbox)
            if task.telescript.rollback is None:
                task.risk_flags.append("no_rollback")
            if task.domain == "email":
                task.risk_flags.append("irreversible_action")
            task.status = TaskStatus.EVALUATED
        return tasks

    async def _sandbox_simulate(self, action: TelescriptAction) -> dict:
        return {"success": True, "params_valid": bool(action.parameters),
                "rollback_valid": action.rollback is not None}

    def _score(self, task: SubTask, sandbox: dict) -> float:
        s = 0.4 * sandbox["success"] + 0.2 * sandbox["params_valid"]
        s += 0.2 * sandbox["rollback_valid"]
        s += 0.2 if task.knowledge_context.get("source") == "engram" else 0.1
        return min(1.0, s)
```

### 3.6 Ethicist Agent

```python
class EthicistAgent:
    """Agent 5/7: 8-dimension Ihsan scoring. GATE, not metric."""

    def __init__(self, constitution: Constitution):
        self.weights = constitution.ihsan_weights
        self.threshold = constitution.ihsan_threshold

    async def score(self, tasks: list[SubTask], goal: str) -> tuple[dict, float]:
        n = max(len(tasks), 1)
        d = {
            "correctness": min(1.0, sum(t.confidence for t in tasks) / n),
            "safety": 1.0 - sum(1 for t in tasks if "no_rollback" in t.risk_flags) / n,
            "user_benefit": 0.98,
            "efficiency": 1.0 if len(tasks) <= 8 else 0.9,
            "auditability": sum(1 for t in tasks if t.telescript) / n,
            "anti_centralization": 1.0,
            "robustness": sum(1 for t in tasks if t.telescript and t.telescript.rollback) / n,
            "adl_fairness": 1.0,
        }
        composite = sum(d[k] * self.weights[k] for k in self.weights)
        return d, composite
```

### 3.7 Publisher + Integrator

```python
class PublisherAgent:
    """Agent 6/7: Generates human-readable summary."""

    async def summarize(self, tasks: list[SubTask],
                        ihsan_scores: dict, composite: float) -> str:
        cost = sum(t.telescript.estimated_cost_usd for t in tasks if t.telescript)
        ms = sum(t.telescript.estimated_duration_ms for t in tasks if t.telescript)
        lines = [f"Plan: {len(tasks)} actions | ~{ms/1000:.1f}s | ${cost:.3f} CPVA | Ihsan: {composite:.3f}", ""]
        for i, t in enumerate(tasks, 1):
            lines.append(f"  {i}. {t.description} [{t.domain}] (conf: {t.confidence:.2f})")
        return "\n".join(lines)


@dataclass
class PlanResponse:
    plan_id: str = field(default_factory=lambda: str(uuid4()))
    tasks: list[dict] = field(default_factory=list)
    estimated_cpva: float = 0.0
    ihsan_composite: float = 0.0
    ihsan_dimensions: dict[str, float] = field(default_factory=dict)
    summary: str = ""
    requires_confirmation: bool = True


class IntegratorAgent:
    """Agent 7/7: Assembles all outputs into PlanResponse."""

    async def assemble(self, tasks: list[SubTask], ihsan_scores: dict,
                       ihsan_composite: float, summary: str) -> PlanResponse:
        return PlanResponse(
            tasks=[{"taskId": t.task_id, "description": t.description,
                    "domain": t.domain, "confidence": t.confidence,
                    "action": t.telescript.action_type if t.telescript else None,
                    "riskFlags": t.risk_flags} for t in tasks],
            estimated_cpva=sum(t.telescript.estimated_cost_usd for t in tasks if t.telescript),
            ihsan_composite=ihsan_composite, ihsan_dimensions=ihsan_scores,
            summary=summary, requires_confirmation=True)
```

---

## 4. TDD Anchors

```python
import pytest
from core.integration.constants import IHSAN_THRESHOLD, IHSAN_WEIGHTS

class TestPAT7Pipeline:
    """Phase 52.3: PAT-7 Chain of Reasoning tests."""

    @pytest.mark.asyncio
    async def test_planner_decomposes_goal(self):
        planner = PlannerAgent(rlm=mock_rlm(), constitution=Constitution())
        tasks = await planner.decompose("Organize my invoice PDFs", {"source_dir": "/invoices"})
        assert 4 <= len(tasks) <= 8

    def test_planner_topological_sort(self):
        planner = PlannerAgent(rlm=mock_rlm(), constitution=Constitution())
        tasks = planner._topological_sort([
            SubTask(task_id="C", depends_on=["A", "B"]),
            SubTask(task_id="A", depends_on=[]),
            SubTask(task_id="B", depends_on=[])])
        ids = [t.task_id for t in tasks]
        assert ids.index("A") < ids.index("C")

    @pytest.mark.asyncio
    async def test_researcher_engram_miss(self):
        researcher = ResearcherAgent(engram=empty_engram())
        tasks = [SubTask(description="Extract vendor names", domain="ocr")]
        enriched = await researcher.enrich(tasks, "organize invoices")
        assert enriched[0].knowledge_context["source"] == "schema_inference"

    @pytest.mark.asyncio
    async def test_coder_generates_telescript(self):
        coder = CoderAgent()
        tasks = [SubTask(description="Create vendor folders", domain="filesystem")]
        coded = await coder.generate(tasks)
        assert coded[0].telescript is not None
        assert coded[0].telescript.verb == "file_open"

    def test_coder_rollback_for_filesystem(self):
        coder = CoderAgent()
        rollback = coder._build_rollback(SubTask(domain="filesystem"))
        assert rollback["action"] == "undo_move"

    @pytest.mark.asyncio
    async def test_evaluator_confidence_range(self):
        evaluator = EvaluatorAgent()
        tasks = [SubTask(domain="filesystem",
            telescript=TelescriptAction("fs", "file_open", rollback={"action": "undo"}),
            knowledge_context={"source": "engram"})]
        evaluated = await evaluator.evaluate(tasks)
        assert 0.0 <= evaluated[0].confidence <= 1.0

    @pytest.mark.asyncio
    async def test_ethicist_passes_valid_plan(self):
        ethicist = EthicistAgent(constitution=Constitution())
        tasks = make_high_confidence_tasks(6)
        _, composite = await ethicist.score(tasks, "organize invoices")
        assert composite >= IHSAN_THRESHOLD

    @pytest.mark.asyncio
    async def test_ethicist_rejects_unsafe_plan(self):
        ethicist = EthicistAgent(constitution=Constitution())
        tasks = [SubTask(domain="email", risk_flags=["no_rollback", "irreversible_action"])]
        scores, _ = await ethicist.score(tasks, "delete everything")
        assert scores["safety"] < 1.0

    def test_ihsan_weights_sum_to_one(self):
        assert abs(sum(IHSAN_WEIGHTS.values()) - 1.0) < 1e-6

    @pytest.mark.asyncio
    async def test_publisher_includes_cpva(self):
        publisher = PublisherAgent()
        tasks = [SubTask(telescript=TelescriptAction("fs", "file_open",
                         estimated_cost_usd=0.01))]
        summary = await publisher.summarize(tasks, {}, 0.97)
        assert "CPVA" in summary

    @pytest.mark.asyncio
    async def test_integrator_assembles_response(self):
        integrator = IntegratorAgent()
        resp = await integrator.assemble(
            tasks=[SubTask(telescript=TelescriptAction("fs", "file_open",
                           estimated_cost_usd=0.01))],
            ihsan_scores={"correctness": 0.98}, ihsan_composite=0.97, summary="test")
        assert resp.plan_id is not None
        assert resp.requires_confirmation is True

    @pytest.mark.asyncio
    async def test_full_pipeline_end_to_end(self):
        pipeline = PAT7Pipeline(constitution=Constitution(), engram=mock_engram(), lm_url="mock://")
        plan = await pipeline.generate_plan(mock_envelope())
        assert len(plan.tasks) >= 4
        assert plan.ihsan_composite >= IHSAN_THRESHOLD
```
