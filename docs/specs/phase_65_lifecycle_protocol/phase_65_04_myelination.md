# Phase 65.4: Myelination — Reflex Compilation

> Standing on Giants: Kahneman (System 1/2 boundary, 2011) · Hebb (synaptic strengthening, 1949) · Shannon (compression, 1948)

## 1. Purpose

Convert a verified System-2 action pattern into a System-1 reflex. This is the
**myelination** step — analogous to neural pathways becoming insulated for faster
signal transmission. The compiled reflex bypasses PAT deliberation but NEVER bypasses
the FATE gate. Safety checks are pre-verified and cached, not removed.

**Entry State**: `[LEARNING]` — Pattern in compilation queue with >= 5 verified successes
**Exit State**: `[MYELINATED]` — Reflex compiled and registered, IMPT consumed
**IMPT Cost**: 50-80 IMPT per reflex compilation
**Temperature after**: T reduced by 50% (e.g., 1.2 → 0.6)

---

## 2. Pseudocode

### 2.1 Reflex Compiler

```
FUNCTION compile_reflex(
    pattern: ActionPattern,
    system_state: SystemState,
    impt_cost: float = 50.0
) -> CompiledReflex:
    """Compile a verified pattern into a System-1 reflex."""

    # Pre-condition: pattern meets compilation threshold
    ASSERT pattern.successes >= COMPILATION_THRESHOLD
    ASSERT pattern.successes / pattern.occurrences >= MIN_SUCCESS_RATE

    # Pre-condition: sufficient IMPT balance
    IF system_state.impt_balance < impt_cost:
        RAISE InsufficientIMPTError(
            f"Need {impt_cost} IMPT, have {system_state.impt_balance}"
        )

    # Step 1: Extract invariant features from historical executions
    invariants = extract_invariants(pattern)

    # Step 2: Generate optimized execution script
    # This is a pre-compiled version of what PAT+Executor produced
    optimized_script = generate_optimized_script(
        pattern.action_type,
        pattern.method,
        invariants
    )

    # Step 3: Pre-compute safety gates
    safety_gates = [
        SafetyGate("TeleScriptPermit", pattern.target_paths),
        SafetyGate("FileNotOpen", pattern.target_resources),
        SafetyGate("BackupManifest", True),  # Always create backup
    ]

    # Step 4: Define the compiled reflex
    reflex = CompiledReflex(
        reflex_id=f"reflex_{pattern.pattern_id}_v1",
        trigger_regex=pattern.trigger_regex,
        action_type=pattern.action_type,
        method=pattern.method,
        safety_gates=safety_gates,
        optimized_script_hash=blake3_hash(optimized_script),
        latency_budget_ms=50,  # System-1 target
        compiled_at=utc_now_iso(),
        source_pattern=pattern.pattern_id,
        source_successes=pattern.successes,
        source_avg_reward=pattern.total_reward / pattern.successes,
    )

    # Step 5: Deduct IMPT compilation fee
    system_state.impt_balance -= impt_cost
    LOG f"Reflex compiled: {reflex.reflex_id}, cost: {impt_cost} IMPT"

    RETURN reflex


FUNCTION extract_invariants(pattern: ActionPattern) -> Invariants:
    """Extract stable features across all successful executions."""

    RETURN Invariants(
        action_type=pattern.action_type,
        method=pattern.method,
        target_scope="user_document_folders",  # Inferred from history
        always_reversible=True,                 # All 5 runs were reversible
        avg_execution_ms=pattern.avg_latency_ms,
    )
```

### 2.2 Reflex Registry

```
@dataclass
class ReflexRegistry:
    reflexes: dict[str, CompiledReflex]    # reflex_id -> reflex
    dispatch_order: list[str]               # Ordered by specificity

FUNCTION register_reflex(
    registry: ReflexRegistry,
    reflex: CompiledReflex
) -> ReflexRegistry:
    """Add compiled reflex to the dispatch registry."""

    registry.reflexes[reflex.reflex_id] = reflex

    # Re-sort dispatch order: more specific triggers first
    registry.dispatch_order = sorted(
        registry.reflexes.keys(),
        key=LAMBDA rid: len(registry.reflexes[rid].trigger_regex),
        reverse=True  # Longer regex = more specific
    )

    RETURN registry


FUNCTION match_reflex(
    registry: ReflexRegistry,
    atom: TeachAtom
) -> CompiledReflex | None:
    """Find the best matching reflex for an intent."""

    FOR reflex_id IN registry.dispatch_order:
        reflex = registry.reflexes[reflex_id]
        IF regex_match(reflex.trigger_regex, atom.raw):
            # Verify safety gates still valid
            IF ALL(gate.check(atom) FOR gate IN reflex.safety_gates):
                RETURN reflex
    RETURN None  # No match → fall through to System-2
```

### 2.3 Temperature Adjustment Post-Compilation

```
FUNCTION adjust_temperature_after_compilation(
    system_state: SystemState
) -> float:
    """Compilation event causes significant temperature drop."""

    # Each compilation halves the distance to T_MIN
    T_MIN = config.get("temperature_min", 0.05)
    new_temp = (system_state.temperature + T_MIN) / 2.0

    system_state.temperature = new_temp
    LOG f"Temperature adjusted: {system_state.temperature} → {new_temp}"

    RETURN new_temp
```

### 2.4 Myelination Orchestrator

```
FUNCTION myelinate(
    pattern_id: str,
    pattern_registry: PatternRegistry,
    reflex_registry: ReflexRegistry,
    system_state: SystemState
) -> MyelinationResult:
    """Full myelination: compile + register + update thermodynamics."""

    pattern = pattern_registry.patterns[pattern_id]

    # Step 1: Compile the reflex
    reflex = compile_reflex(pattern, system_state)

    # Step 2: Register in dispatch
    reflex_registry = register_reflex(reflex_registry, reflex)

    # Step 3: Update thermodynamics
    adjust_temperature_after_compilation(system_state)

    # Step 4: Update system state
    system_state.reflexes_compiled += 1
    system_state.state = "MYELINATED"

    # Step 5: Remove from compilation queue
    pattern_registry.compilation_queue.remove(pattern_id)

    # Step 6: Emit compilation receipt
    receipt = {
        "type": "REFLEX_COMPILATION",
        "reflex_id": reflex.reflex_id,
        "source_pattern": pattern_id,
        "source_successes": pattern.successes,
        "impt_cost": 50.0,
        "new_temperature": system_state.temperature,
        "reason_codes": ["MYELINATION", "SYSTEM2_TO_SYSTEM1"]
    }
    system_state.ledger.append(receipt=receipt)

    RETURN MyelinationResult(
        reflex=reflex,
        impt_spent=50.0,
        new_temperature=system_state.temperature,
        total_reflexes=system_state.reflexes_compiled
    )
```

---

## 3. Data Structures

```
@dataclass
class CompiledReflex:
    reflex_id: str                   # "reflex_{pattern}_v1"
    trigger_regex: str               # Pattern matcher
    action_type: str                 # "FILE_ORGANIZATION"
    method: str                      # "TOPIC_EXTRACTION"
    safety_gates: list[SafetyGate]   # Pre-computed safety checks
    optimized_script_hash: str       # BLAKE3 of compiled script
    latency_budget_ms: int           # 50ms target
    compiled_at: str                 # ISO timestamp
    source_pattern: str              # Pattern ID that triggered compilation
    source_successes: int            # How many verified runs
    source_avg_reward: float         # Average reward across runs

@dataclass
class SafetyGate:
    gate_type: str                   # "TeleScriptPermit", etc.
    parameters: Any                  # Gate-specific config
    FUNCTION check(self, atom: TeachAtom) -> bool

@dataclass
class Invariants:
    action_type: str
    method: str
    target_scope: str
    always_reversible: bool
    avg_execution_ms: float

@dataclass
class MyelinationResult:
    reflex: CompiledReflex
    impt_spent: float
    new_temperature: float
    total_reflexes: int
```

---

## 4. Safety Invariant

```
CRITICAL INVARIANT: Reflex compilation NEVER removes FATE gate checks.

System-2 path:  TEACH → PAT → FATE → HDA → UIA → PoI → RLVR
System-1 path:  TEACH → REFLEX_MATCH → FATE(cached) → HDA(optimized) → UIA → PoI → RLVR

The FATE gate is present in BOTH paths. What changes:
  - PAT deliberation: SKIPPED (pre-verified safe by 5+ successful runs)
  - FATE gate: CACHED (pre-computed safety checks, still run but faster)
  - HDA execution: OPTIMIZED (pre-compiled script, fewer branches)
  - UIA verification: TARGETED (knows exactly what to check)
  - PoI emission: COMPILED (pre-computed receipt template)

What NEVER changes:
  - FATE gate runs on every action
  - UIA verification confirms every state change
  - PoI receipt is emitted for every action
  - Ed25519 signature on every receipt
```

---

## 5. TDD Anchors

### Existing Tests
- `tests/core/sovereign/test_hardening_track1.py` — threshold alignment
- `tests/core/treasury/test_token_minter.py` — IMPT balance operations

### New Tests Required

```python
# tests/core/sovereign/test_lifecycle_myelination.py

class TestReflexCompiler:

    def test_compilation_requires_threshold_successes(self):
        """Pattern with < 5 successes cannot compile."""
        pattern = make_pattern(successes=4)
        with pytest.raises(AssertionError):
            compile_reflex(pattern, make_state())

    def test_compilation_deducts_impt(self):
        """Compilation costs exactly impt_cost IMPT."""
        state = make_state(impt_balance=100.0)
        compile_reflex(make_eligible_pattern(), state, impt_cost=50.0)
        assert state.impt_balance == 50.0

    def test_insufficient_impt_raises(self):
        """Cannot compile if IMPT balance too low."""
        state = make_state(impt_balance=10.0)
        with pytest.raises(InsufficientIMPTError):
            compile_reflex(make_eligible_pattern(), state, impt_cost=50.0)

    def test_reflex_has_safety_gates(self):
        """Compiled reflex includes safety gates."""
        reflex = compile_reflex(make_eligible_pattern(), make_state())
        assert len(reflex.safety_gates) > 0

    def test_reflex_latency_budget_is_50ms(self):
        """System-1 target is 50ms."""
        reflex = compile_reflex(make_eligible_pattern(), make_state())
        assert reflex.latency_budget_ms == 50


class TestReflexRegistry:

    def test_registry_matches_by_trigger(self):
        """Registered reflex matches on trigger regex."""
        registry = make_registry_with_reflex("organize .* by topic")
        atom = make_atom("organize my papers by topic")
        match = match_reflex(registry, atom)
        assert match is not None

    def test_registry_returns_none_for_unknown(self):
        """Unmatched intent returns None (falls through to System-2)."""
        registry = make_empty_registry()
        atom = make_atom("send email to boss")
        assert match_reflex(registry, atom) is None

    def test_more_specific_regex_wins(self):
        """Longer (more specific) regex takes priority."""
        registry = make_registry_with_reflexes([
            "organize .*",
            "organize .* by topic"
        ])
        atom = make_atom("organize files by topic")
        match = match_reflex(registry, atom)
        assert "by topic" in match.trigger_regex


class TestMyelinationOrchestrator:

    def test_myelination_transitions_state(self):
        """State becomes MYELINATED after first compilation."""
        state = make_state(state="LEARNING")
        myelinate("pattern_001", make_pattern_reg(), make_reflex_reg(), state)
        assert state.state == "MYELINATED"

    def test_myelination_emits_receipt(self):
        """Compilation event produces a REFLEX_COMPILATION receipt."""
        state = make_state_with_ledger()
        myelinate("pattern_001", make_pattern_reg(), make_reflex_reg(), state)
        last_receipt = state.ledger.last()
        assert last_receipt["type"] == "REFLEX_COMPILATION"

    def test_temperature_drops_after_compilation(self):
        """Temperature decreases significantly after myelination."""
        state = make_state(temperature=1.2)
        myelinate("pattern_001", make_pattern_reg(), make_reflex_reg(), state)
        assert state.temperature < 1.2
```

---

## 6. Error Handling

```
ON compilation failure (e.g., invariant extraction fails):
    Keep pattern in compilation queue
    LOG error with pattern details
    Retry on next successful occurrence of same pattern

ON safety gate pre-computation failure:
    Do NOT compile — safety gates must be computable
    Pattern remains in System-2 path until gates resolve
```
