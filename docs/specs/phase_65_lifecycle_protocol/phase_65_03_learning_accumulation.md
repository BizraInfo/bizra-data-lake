# Phase 65.3: Learning & Pattern Accumulation

> Standing on Giants: Hebb (synaptic learning, 1949) · Shannon (entropy reduction, 1948) · Deming (PDCA cycle, 1950)

## 1. Purpose

Over 7 days of repeated interactions, the system accumulates verified action patterns
in episodic memory. Each successful action reduces epistemic entropy and cools the
thermodynamic temperature. When a pattern reaches the compilation threshold (5 verified
successes), it becomes eligible for reflex myelination (Phase 65.4).

**Entry State**: `[LEARNING]` — Temperature T = 2.0, Entropy H ~ 4.0 bits
**Exit State**: `[LEARNING]` — Temperature T ~ 1.2, Entropy H ~ 2.1 bits, pattern eligible
**Duration**: ~7 days of natural use

---

## 2. Pseudocode

### 2.1 Pattern Tracker

```
CONSTANT COMPILATION_THRESHOLD = 5  # Verified successes before eligible
CONSTANT MIN_SUCCESS_RATE = 0.80    # 80% to qualify

@dataclass
class ActionPattern:
    pattern_id: str              # "pattern_{hash}"
    trigger_regex: str           # e.g., "organize .* by topic"
    action_type: str             # "FILE_ORGANIZATION"
    method: str                  # "TOPIC_EXTRACTION"
    occurrences: int             # Count of matches
    successes: int               # UIA-verified successes
    total_reward: float          # Accumulated RLVR reward
    avg_latency_ms: float        # Average execution time
    first_seen: str              # ISO timestamp
    last_seen: str               # ISO timestamp

@dataclass
class PatternRegistry:
    patterns: dict[str, ActionPattern]   # pattern_id -> ActionPattern
    compilation_queue: list[str]         # Pattern IDs eligible for compilation

FUNCTION record_action_outcome(
    registry: PatternRegistry,
    atom: TeachAtom,
    result: InteractionResult
) -> PatternRegistry:
    """Record action outcome and check compilation eligibility."""

    # Step 1: Find or create matching pattern
    pattern_id = compute_pattern_hash(atom.structured)
    IF pattern_id NOT IN registry.patterns:
        registry.patterns[pattern_id] = ActionPattern(
            pattern_id=pattern_id,
            trigger_regex=infer_trigger_regex(atom),
            action_type=atom.structured.action,
            method=atom.structured.method,
            occurrences=0,
            successes=0,
            total_reward=0.0,
            avg_latency_ms=0.0,
            first_seen=utc_now_iso(),
            last_seen=utc_now_iso()
        )

    pattern = registry.patterns[pattern_id]

    # Step 2: Update counters
    pattern.occurrences += 1
    pattern.last_seen = utc_now_iso()

    IF result.success AND result.uia_receipt.verified:
        pattern.successes += 1
        pattern.total_reward += result.reward.total
        pattern.avg_latency_ms = (
            (pattern.avg_latency_ms * (pattern.successes - 1) + result.total_latency_ms)
            / pattern.successes
        )

    # Step 3: Check compilation eligibility
    IF (
        pattern.successes >= COMPILATION_THRESHOLD
        AND pattern.successes / pattern.occurrences >= MIN_SUCCESS_RATE
        AND pattern_id NOT IN registry.compilation_queue
    ):
        registry.compilation_queue.append(pattern_id)
        LOG "Pattern eligible for compilation: {pattern_id}"

    RETURN registry
```

### 2.2 Thermodynamic Cooling

```
FUNCTION update_temperature(
    system_state: SystemState,
    action_success: bool
) -> float:
    """Cool temperature based on successful actions."""

    # Exponential cooling: T(t) = T_0 * exp(-lambda * n_success)
    # where lambda = learning_rate / action_space_size
    LAMBDA = config.get("cooling_lambda", 0.05)

    IF action_success:
        system_state.successful_actions += 1

    new_temp = (
        INITIAL_TEMPERATURE
        * exp(-LAMBDA * system_state.successful_actions)
    )

    # Floor: never go below T_min (always allow some exploration)
    T_MIN = config.get("temperature_min", 0.05)
    new_temp = max(new_temp, T_MIN)

    system_state.temperature = new_temp
    RETURN new_temp
```

### 2.3 Entropy Reduction

```
FUNCTION update_entropy(
    system_state: SystemState,
    pattern_registry: PatternRegistry
) -> float:
    """Compute epistemic entropy from pattern coverage."""

    # H = -sum(p_i * log2(p_i)) over action distribution
    # As patterns are learned, distribution sharpens → entropy drops

    total_actions = sum(p.occurrences FOR p IN pattern_registry.patterns.values())
    IF total_actions == 0:
        RETURN system_state.epistemic_entropy  # No change

    entropy = 0.0
    FOR pattern IN pattern_registry.patterns.values():
        p_i = pattern.occurrences / total_actions
        IF p_i > 0:
            entropy -= p_i * log2(p_i)

    system_state.epistemic_entropy = entropy
    RETURN entropy
```

### 2.4 Learning Phase Orchestrator

```
FUNCTION learning_phase_tick(
    user_input: str,
    system_state: SystemState,
    pattern_registry: PatternRegistry
) -> tuple[InteractionResult, PatternRegistry]:
    """One tick of the learning phase: interact + record + cool."""

    # Execute interaction (full System-2 path)
    result = first_interaction(user_input, system_state)

    # Record outcome in pattern registry
    atom = extract_teach_atom(user_input)
    pattern_registry = record_action_outcome(pattern_registry, atom, result)

    # Update thermodynamics
    update_temperature(system_state, result.success)
    update_entropy(system_state, pattern_registry)

    # Check if any patterns ready for compilation
    IF pattern_registry.compilation_queue:
        LOG "Patterns ready for myelination: {pattern_registry.compilation_queue}"
        # Trigger Phase 65.4 (Myelination)

    RETURN result, pattern_registry
```

---

## 3. 7-Day Example Timeline

```
Day 1: "Organize research papers by topic"
  Pattern: FILE_ORGANIZE_BY_TOPIC
  Occurrences: 1, Successes: 1, Reward: +1.996
  Temperature: 2.0 → 1.90, Entropy: 4.2 bits

Day 2: "Organize downloads folder by topic"
  Pattern: FILE_ORGANIZE_BY_TOPIC (same pattern!)
  Occurrences: 2, Successes: 2, Reward: +1.850
  Temperature: 1.90 → 1.81, Entropy: 3.9 bits

Day 3: "Organize project files by topic"
  Occurrences: 3, Successes: 3, Reward: +1.920
  Temperature: 1.81 → 1.72, Entropy: 3.5 bits

Day 4: "Organize email attachments by topic"
  Occurrences: 4, Successes: 4, Reward: +1.780
  Temperature: 1.72 → 1.64, Entropy: 3.1 bits

Day 5: "Organize desktop files by topic"
  Occurrences: 5, Successes: 5, Reward: +1.940
  *** COMPILATION THRESHOLD REACHED ***
  Success rate: 100% (>= 80%)
  Pattern added to compilation_queue
  Temperature: 1.64 → 1.56, Entropy: 2.8 bits

Days 6-7: Additional interactions (other patterns)
  Temperature: 1.56 → 1.20, Entropy: 2.8 → 2.1 bits
```

---

## 4. Data Structures

```
@dataclass
class LearningPhaseState:
    system_state: SystemState          # From Phase 65.1
    pattern_registry: PatternRegistry  # Pattern tracker
    day_count: int                     # Days since genesis
    total_actions: int                 # All actions attempted
    total_successes: int               # UIA-verified successes
    avg_reward: float                  # Rolling average
    compilation_triggered: bool        # Any pattern hit threshold
```

---

## 5. TDD Anchors

### Existing Tests
- `tests/core/sovereign/test_mission.py` — mission pipeline execution
- `tests/core/living_memory/` — episodic memory storage

### New Tests Required

```python
# tests/core/sovereign/test_lifecycle_learning.py

class TestPatternTracker:

    def test_new_pattern_created_on_first_action(self):
        """First occurrence of an intent creates a new pattern."""
        registry = PatternRegistry(patterns={}, compilation_queue=[])
        atom = make_atom("organize files by topic")
        result = make_success_result()
        registry = record_action_outcome(registry, atom, result)
        assert len(registry.patterns) == 1

    def test_same_intent_increments_existing_pattern(self):
        """Repeated intent increments occurrence counter."""
        registry = PatternRegistry(patterns={}, compilation_queue=[])
        atom1 = make_atom("organize papers by topic")
        atom2 = make_atom("organize downloads by topic")
        record_action_outcome(registry, atom1, make_success_result())
        record_action_outcome(registry, atom2, make_success_result())
        # Same pattern (FILE_ORGANIZE_BY_TOPIC)
        assert list(registry.patterns.values())[0].occurrences == 2

    def test_compilation_threshold_triggers_at_5(self):
        """Pattern enters compilation queue after 5 successes."""
        registry = PatternRegistry(patterns={}, compilation_queue=[])
        FOR i IN range(5):
            record_action_outcome(registry, make_atom(f"organize {i}"),
                                  make_success_result())
        assert len(registry.compilation_queue) == 1

    def test_failed_actions_dont_count_toward_threshold(self):
        """Only UIA-verified successes count toward compilation."""
        registry = PatternRegistry(patterns={}, compilation_queue=[])
        FOR i IN range(5):
            record_action_outcome(registry, make_atom(f"organize {i}"),
                                  make_failure_result())
        assert len(registry.compilation_queue) == 0


class TestThermodynamicCooling:

    def test_temperature_decreases_on_success(self):
        """Successful action cools the temperature."""
        state = make_state(temperature=2.0)
        new_t = update_temperature(state, action_success=True)
        assert new_t < 2.0

    def test_temperature_has_floor(self):
        """Temperature never drops below T_MIN."""
        state = make_state(temperature=0.01, successful_actions=10000)
        new_t = update_temperature(state, action_success=True)
        assert new_t >= 0.05  # T_MIN

    def test_entropy_decreases_as_patterns_concentrate(self):
        """Entropy drops as action distribution sharpens."""
        registry = make_registry_with_patterns(n_patterns=1, n_actions=100)
        state = make_state(epistemic_entropy=4.0)
        new_h = update_entropy(state, registry)
        assert new_h < 4.0  # Single dominant pattern → low entropy
```

---

## 6. Error Handling

```
ON pattern hash collision:
    Use full UUID as secondary key
    LOG warning for monitoring

ON compilation queue overflow (> 10 patterns waiting):
    Prioritize by reward/occurrence ratio
    Process top-3 in next myelination cycle

ON entropy computation with zero actions:
    Return current entropy unchanged
    This is a no-op, not an error
```
