# Phase 68.02 — Omega Loop Controller (Proof-Based Iteration)

## Context

Claude's "Ralph Loop" iterates via artifacts until a promise tag is emitted.
BIZRA upgrades this: the loop can only terminate when **proof validates**.
No self-reported completion. The system proves it or keeps going.

This is the MMORPG "game save" — state persists in the Event Log, resumable
after months of inactivity.

---

## 1. Requirements

### FR-1: Proof-Based Termination
The loop MUST NOT terminate until all proof conditions are satisfied:
- FATE gate passes on final state
- Receipts verify state changes
- Ihsan >= floor (0.95 production)
- Validators pass (tests, UIA diff, etc.)
- Ledger commit succeeds

### FR-2: Resumability
Loop state is stored in EventLog (not local files). A node can resume
an Omega loop after arbitrary downtime by replaying events.

### FR-3: Bounded Iterations
Hard limit prevents infinite loops. Default: 50 iterations.
Configurable via `bizra.node.yaml`.

### FR-4: Cancel + Pause
Operator can cancel mid-loop. State is preserved for later resume.

### FR-5: Budget Enforcement
Each iteration consumes budget (time, tokens, actions). Loop stops
if budget exhausted even if proof incomplete (status: BUDGET_EXHAUSTED).

---

## 2. Data Types

```python
@dataclass
class OmegaLoopState:
    """Persistent state of an Omega loop execution."""
    loop_id: str                    # blake3 of mission + params
    mission_id: str                 # correlation to mission
    iteration: int = 0             # current iteration count
    max_iterations: int = 50       # hard stop
    status: OmegaStatus = RUNNING
    budget_remaining: LoopBudget = field(default_factory=LoopBudget)
    proof_conditions: list[ProofCondition] = field(default_factory=list)
    events: list[str] = field(default_factory=list)  # event IDs
    started_at: int = 0
    last_tick_at: int = 0

class OmegaStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PROVED = "proved"           # all proofs satisfied
    BUDGET_EXHAUSTED = "budget_exhausted"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    FAILED = "failed"           # unrecoverable error
    MAX_ITERATIONS = "max_iterations"

@dataclass
class LoopBudget:
    time_ms: int = 300_000      # 5 minutes default
    s2_tokens: int = 50_000     # LLM token budget
    actions: int = 100          # max action proposals

@dataclass(frozen=True)
class ProofCondition:
    """A condition that must be satisfied for loop termination."""
    kind: str                   # "tests_pass" | "ihsan_above" | "uia_verified" | "custom"
    target: str                 # what to check
    threshold: int = 0          # fixed-point threshold (if applicable)
    satisfied: bool = False
```

---

## 3. OmegaLoopController — Pseudocode

```
CLASS OmegaLoopController:
    INIT(action_bus, event_bus, event_log, config):
        self.action_bus = action_bus
        self.event_bus = event_bus
        self.event_log = event_log
        self.config = config
        self._active_loops: dict[str, OmegaLoopState] = {}

    ASYNC run(mission_id, plan, proof_conditions, budget) -> OmegaLoopState:
        """Execute an Omega loop until proof validates or budget exhausted."""

        # Initialize or resume
        loop_id = blake3(mission_id + str(proof_conditions))
        state = self._resume_or_create(loop_id, mission_id, proof_conditions, budget)
        self._active_loops[loop_id] = state

        AWAIT self.event_bus.publish("omega.started", {
            "loop_id": loop_id,
            "mission_id": mission_id,
            "proof_conditions": [pc.kind for pc in proof_conditions],
        })

        WHILE state.status == RUNNING:
            # Check iteration limit
            IF state.iteration >= state.max_iterations:
                state.status = MAX_ITERATIONS
                BREAK

            # Check budget
            IF NOT self._budget_ok(state):
                state.status = BUDGET_EXHAUSTED
                BREAK

            # === One Iteration ===
            state.iteration += 1
            state.last_tick_at = now_ms()

            AWAIT self.event_bus.publish("omega.iteration", {
                "loop_id": loop_id,
                "iteration": state.iteration,
            })

            # Step 1: Plan next actions from current state
            actions = AWAIT self._plan_iteration(state, plan)

            # Step 2: Execute actions via ActionBus
            receipts = []
            FOR action IN actions:
                receipt = AWAIT self.action_bus.propose(action)
                receipts.append(receipt)
                state.budget_remaining.actions -= 1
                IF receipt.status == DENIED:
                    # Don't count denied actions against budget
                    state.budget_remaining.actions += 1

            # Step 3: Log events
            FOR receipt IN receipts:
                event = self.event_log.append("omega.action", state.actor, {
                    "loop_id": loop_id,
                    "iteration": state.iteration,
                    "receipt": receipt.to_dict(),
                })
                state.events.append(event.event_id)

            # Step 4: Check proof conditions
            all_proved = AWAIT self._check_proofs(state, receipts)

            IF all_proved:
                state.status = PROVED
                AWAIT self.event_bus.publish("omega.proved", {
                    "loop_id": loop_id,
                    "iterations": state.iteration,
                    "proof_conditions": [pc.to_dict() for pc in state.proof_conditions],
                })
                BREAK

            # Step 5: Update budget (time consumed)
            elapsed = now_ms() - state.started_at
            state.budget_remaining.time_ms = max(0, budget.time_ms - elapsed)

        # Final state emission
        AWAIT self.event_bus.publish("omega.completed", {
            "loop_id": loop_id,
            "status": state.status.value,
            "iterations": state.iteration,
        })

        RETURN state

    ASYNC cancel(loop_id: str):
        """Operator-initiated cancellation."""
        state = self._active_loops.get(loop_id)
        IF state AND state.status == RUNNING:
            state.status = CANCELLED
            AWAIT self.event_bus.publish("omega.cancelled", {"loop_id": loop_id})

    ASYNC pause(loop_id: str):
        """Pause for later resume."""
        state = self._active_loops.get(loop_id)
        IF state AND state.status == RUNNING:
            state.status = PAUSED
            AWAIT self.event_bus.publish("omega.paused", {
                "loop_id": loop_id,
                "iteration": state.iteration,
            })

    ASYNC _check_proofs(state, receipts) -> bool:
        """Evaluate all proof conditions against current state."""
        all_satisfied = True
        FOR pc IN state.proof_conditions:
            IF pc.kind == "ihsan_above":
                # Check that ALL receipts meet ihsan threshold
                scores = [r.ihsan_score for r in receipts if r.status == COMPLETED]
                pc.satisfied = all(s >= pc.threshold for s in scores) AND len(scores) > 0
            ELIF pc.kind == "tests_pass":
                pc.satisfied = AWAIT self._run_validator(pc.target)
            ELIF pc.kind == "uia_verified":
                pc.satisfied = AWAIT self._check_uia_diff(pc.target)
            ELIF pc.kind == "fate_passes":
                pc.satisfied = self.fate_gate.evaluate_state(state).approved
            ELIF pc.kind == "ledger_committed":
                pc.satisfied = len(state.events) > 0

            IF NOT pc.satisfied:
                all_satisfied = False

        RETURN all_satisfied

    DEF _resume_or_create(loop_id, mission_id, conditions, budget) -> OmegaLoopState:
        """Resume from EventLog or create fresh."""
        # Scan event log for prior loop state
        prior_events = [e for e in self.event_log if e.data.get("loop_id") == loop_id]
        IF prior_events:
            # Replay to reconstruct state
            state = self._replay_state(prior_events)
            state.status = RUNNING  # resume
            RETURN state
        RETURN OmegaLoopState(
            loop_id=loop_id,
            mission_id=mission_id,
            proof_conditions=conditions,
            budget_remaining=budget,
            started_at=now_ms(),
        )
```

---

## 4. Resumability via Event Replay

The key insight: loop state is NOT stored in a file or database. It's
reconstructed by replaying events from the EventLog.

```
FUNCTION replay_state(events) -> OmegaLoopState:
    state = OmegaLoopState()
    FOR event IN sorted(events, key=lambda e: e.event_id):
        MATCH event.event_type:
            "omega.started":
                state.loop_id = event.data["loop_id"]
                state.mission_id = event.data["mission_id"]
            "omega.iteration":
                state.iteration = event.data["iteration"]
            "omega.action":
                state.events.append(event.event_id)
            "omega.proved":
                state.status = PROVED
            "omega.cancelled":
                state.status = CANCELLED
            "omega.paused":
                state.status = PAUSED
    RETURN state
```

This means a node can shut down, reboot 6 months later, replay its event
log, and resume exactly where it left off. MMORPG character persistence.

---

## 5. Integration with MissionOrchestrator

```
CURRENT:
  MissionOrchestrator.execute() -> run 6 phases -> return result

PHASE 68:
  MissionOrchestrator.execute() ->
    OmegaLoop.run(
        mission_id=req.mission_id,
        plan=decomposed_plan,
        proof_conditions=[
            ProofCondition("ihsan_above", threshold=IHSAN_FLOOR),
            ProofCondition("ledger_committed"),
        ],
        budget=LoopBudget(time_ms=60_000),
    )
```

Simple missions: 1 iteration (plan -> execute -> prove -> done).
Complex missions: multiple iterations with progressive refinement.

---

## 6. TDD Anchors (14 tests)

```python
class TestOmegaLoopBasic:
    def test_single_iteration_proves()
    def test_multiple_iterations_to_prove()
    def test_max_iterations_stops_loop()
    def test_budget_exhausted_stops_loop()

class TestOmegaTermination:
    def test_terminates_only_when_all_proofs_satisfied()
    def test_ihsan_below_floor_continues()
    def test_partial_proofs_do_not_terminate()

class TestOmegaCancel:
    def test_cancel_stops_running_loop()
    def test_pause_preserves_state()

class TestOmegaResume:
    def test_resume_from_event_log()
    def test_resume_continues_iteration_count()
    def test_resume_after_pause_runs()

class TestOmegaEvents:
    def test_started_event_emitted()
    def test_proved_event_includes_conditions()
```

---

## 7. Non-Goals

- **No distributed Omega loops.** Single-node only for Phase 68.
- **No LLM-generated proof conditions.** Conditions are specified by the
  operator/capsule, not generated at runtime.
- **No nested Omega loops.** A loop can spawn actions that internally iterate,
  but the OmegaLoop itself is flat.
