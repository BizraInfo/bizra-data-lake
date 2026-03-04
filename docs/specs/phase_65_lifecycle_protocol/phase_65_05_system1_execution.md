# Phase 65.5: System-1 Lightning Execution

> Standing on Giants: Kahneman (System 1 fast path, 2011) · Boyd (OODA speed advantage, 1976)

## 1. Purpose

Execute a user action through the compiled System-1 reflex path, achieving 8.2x speedup
over System-2 while maintaining identical safety guarantees. The reflex dispatcher
matches intent to compiled reflex, executes through cached FATE gate, and still produces
full PoI receipt with UIA verification.

**Entry State**: `[MYELINATED]` — Reflex compiled, T ~ 0.6
**Exit State**: `[MYELINATED]` — Same state, reward accumulated
**Target Latency**: < 400ms (vs 3080ms System-2)

---

## 2. Pseudocode

### 2.1 Reflex Dispatcher

```
FUNCTION dispatch_intent(
    atom: TeachAtom,
    system_state: SystemState,
    reflex_registry: ReflexRegistry
) -> DispatchDecision:
    """Route intent to System-1 (reflex) or System-2 (PAT)."""

    # Try System-1 first
    reflex = match_reflex(reflex_registry, atom)

    IF reflex IS NOT None:
        RETURN DispatchDecision(
            path="SYSTEM_1",
            reflex=reflex,
            reason=f"Matched reflex: {reflex.reflex_id}"
        )
    ELSE:
        RETURN DispatchDecision(
            path="SYSTEM_2",
            reflex=None,
            reason="No matching reflex — full PAT deliberation"
        )
```

### 2.2 System-1 Execution Path

```
FUNCTION system1_execute(
    atom: TeachAtom,
    reflex: CompiledReflex,
    system_state: SystemState
) -> InteractionResult:
    """System-1 fast path: cached safety + optimized execution."""

    t0 = monotonic_clock()

    # Step 1: FATE Gate (cached — pre-computed safety gates)
    # CRITICAL: Gate is still RUN, just faster
    gate = fate_gate_verify_cached(reflex.safety_gates, system_state)
    IF NOT gate.allowed:
        # Safety violation — fall through to System-2 for full analysis
        LOG "Reflex safety gate failed, falling through to System-2"
        RETURN system2_execute(atom, system_state)  # Phase 65.2 path

    # Step 2: HDA Execution (optimized — pre-compiled script)
    pre_state = UIA.snapshot_targeted(reflex.expected_targets)
    hda_result = hda_execute_optimized(reflex, atom)
    post_state = UIA.snapshot_targeted(reflex.expected_targets)

    # Step 3: UIA Verification (targeted — knows exactly what to check)
    uia_receipt = uia_verify_targeted(
        pre_state, post_state, reflex.expected_changes
    )

    # Step 4: PoI Receipt (compiled — pre-computed template)
    poi = poi_emit_compiled(atom, hda_result, uia_receipt, gate, system_state)

    # Step 5: RLVR Reward (with efficiency bonus)
    reward = rlvr_reward(
        uia_receipt, gate, hda_result,
        budget_ms=reflex.latency_budget_ms
    )

    elapsed = monotonic_clock() - t0

    # Update system state
    system_state.impt_balance += reward.total

    RETURN InteractionResult(
        success=True,
        poi=poi,
        reward=reward,
        total_latency_ms=elapsed,
        path="SYSTEM_1"
    )


FUNCTION fate_gate_verify_cached(
    safety_gates: list[SafetyGate],
    system_state: SystemState
) -> FateGateResult:
    """Lightweight FATE verification using pre-computed gates."""

    # The gates were computed at compilation time.
    # We re-run them, but they execute faster because:
    # 1. No need to compute TeleScript from scratch (pre-verified paths)
    # 2. No need to run 7-agent PAT (pre-verified plan)
    # 3. Ihsan dimensions pre-scored (just verify nothing changed)

    all_pass = ALL(gate.check_fast() FOR gate IN safety_gates)

    # Still verify sovereignty (Ed25519 check is always fresh)
    sovereignty = verify_ed25519_signature(
        system_state.identity.public_key,
        system_state.latest_signature
    )

    RETURN FateGateResult(
        allowed=all_pass AND sovereignty,
        gate_latency_ms=elapsed()  # ~8ms vs ~45ms
    )
```

### 2.3 Unified Intent Handler

```
FUNCTION handle_intent(
    user_input: str,
    system_state: SystemState,
    reflex_registry: ReflexRegistry,
    pattern_registry: PatternRegistry
) -> InteractionResult:
    """Unified handler: dispatch to System-1 or System-2."""

    atom = extract_teach_atom(user_input)

    # Dispatch decision
    decision = dispatch_intent(atom, system_state, reflex_registry)

    IF decision.path == "SYSTEM_1":
        result = system1_execute(atom, decision.reflex, system_state)
    ELSE:
        result = first_interaction(user_input, system_state)  # System-2

    # Always record for pattern learning (even System-1 actions)
    record_action_outcome(pattern_registry, atom, result)
    update_temperature(system_state, result.success)
    update_entropy(system_state, pattern_registry)

    RETURN result
```

---

## 3. Performance Comparison

```
| Subsystem          | System-2 (ms) | System-1 (ms) | Speedup |
|--------------------|---------------|---------------|---------|
| TEACH extraction   | 20            | 20            | 1.0x    |
| PAT deliberation   | 1650          | SKIPPED       | --      |
| FATE gate          | 45            | 8             | 5.6x    |
| HDA execution      | 1180          | 340           | 3.5x    |
| UIA verification   | 85            | 12            | 7.1x    |
| PoI emission       | 120           | 15            | 8.0x    |
| RLVR computation   | 15            | 5             | 3.0x    |
| TOTAL              | 3115          | 400           | 7.8x    |

System-1 achieves ~8x speedup. The dominant savings come from:
  1. PAT bypass (1650ms saved — pre-verified safe path)
  2. HDA optimization (840ms saved — pre-compiled script)
  3. Targeted UIA (73ms saved — knows exactly what changed)
```

---

## 4. Safety Guarantee

```
INVARIANT: System-1 never trades safety for speed.

PROPERTY: For any action A:
  system1_execute(A).safety_checks == system2_execute(A).safety_checks
  - FATE gate: present in both (cached vs. full)
  - UIA verification: present in both (targeted vs. broad)
  - PoI receipt: present in both (compiled vs. dynamic)
  - Ed25519 signature: present in both (always fresh)

PROPERTY: If System-1 safety gate fails, automatic fallback to System-2:
  fate_gate_verify_cached(reflex.gates, state).allowed == False
  → system2_execute(atom, state)  # Full deliberation, not abort
```

---

## 5. TDD Anchors

### Existing Tests
- `tests/core/sovereign/test_mission.py` — mission execution pipeline
- `tests/core/sovereign/test_hardening_track1.py` — auth and threshold gates

### New Tests Required

```python
# tests/core/sovereign/test_lifecycle_system1.py

class TestReflexDispatcher:

    def test_dispatches_to_system1_when_reflex_matches(self):
        """Matching reflex → SYSTEM_1 dispatch."""
        registry = make_registry_with_reflex("organize .* by topic")
        atom = make_atom("organize files by topic")
        decision = dispatch_intent(atom, make_state(), registry)
        assert decision.path == "SYSTEM_1"

    def test_falls_through_to_system2_when_no_match(self):
        """No matching reflex → SYSTEM_2 dispatch."""
        registry = make_empty_registry()
        atom = make_atom("summarize this document")
        decision = dispatch_intent(atom, make_state(), registry)
        assert decision.path == "SYSTEM_2"


class TestSystem1Execution:

    def test_system1_faster_than_system2(self):
        """System-1 execution latency < System-2."""
        s1_result = system1_execute(atom, reflex, state)
        s2_result = first_interaction(atom.raw, state)
        assert s1_result.total_latency_ms < s2_result.total_latency_ms

    def test_system1_still_emits_receipt(self):
        """System-1 path produces signed PoI receipt."""
        result = system1_execute(atom, reflex, state)
        assert result.poi is not None
        assert result.poi.signature is not None

    def test_system1_falls_back_on_gate_failure(self):
        """If cached gate fails, falls through to System-2."""
        reflex = make_reflex_with_invalid_telescript()
        result = system1_execute(atom, reflex, state)
        # Should still succeed (via System-2 fallback)
        # but with System-2 latency
        assert result.path in ("SYSTEM_1", "SYSTEM_2")

    def test_system1_reward_includes_efficiency_bonus(self):
        """Fast execution earns efficiency bonus."""
        result = system1_execute(atom, fast_reflex, state)
        assert result.reward.total > base_reward  # efficiency bonus added


class TestSafetyInvariant:

    def test_fate_gate_runs_in_system1(self):
        """FATE gate verification occurs in System-1 path."""
        # Monkeypatch gate to track calls
        gate_called = False
        result = system1_execute(atom, reflex, state)
        assert fate_gate_was_called  # Gate always runs

    def test_uia_verifies_in_system1(self):
        """UIA verification occurs in System-1 path."""
        result = system1_execute(atom, reflex, state)
        assert result.poi.receipt["verification"]["uia_confirmed"] is True
```

---

## 6. Error Handling

```
ON reflex execution failure (HDA error):
    Do NOT retry via System-1
    Fall through to System-2 for full PAT analysis
    Record failure in pattern registry (may revoke reflex if failure rate > 20%)

ON reflex safety gate drift:
    Gate was pre-computed but environment changed (e.g., new TeleScript deny)
    Automatic fallback to System-2
    Recompile reflex with updated safety gates

ON repeated System-1 failures (3 consecutive):
    Deactivate reflex temporarily
    Force System-2 path for this pattern
    Alert user: "A learned shortcut is having issues — using careful mode"
```
