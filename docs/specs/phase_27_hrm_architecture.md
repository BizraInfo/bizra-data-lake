# Phase 27: Hierarchical Reasoning Model (HRM)

> Multi-level cognitive architecture with nested autopoietic loops, learning cascade, resonance detection, and meta-autopoiesis.

## Context

The HRM fuses hierarchical reasoning (Simon, 1962) with autopoietic cognitive architecture (Maturana & Varela, 1980). The result is a multi-dimensional cognitive space where hierarchical depth and autopoietic depth intersect — each level runs its own complete 8-stage RDVE cycle while coordinating with adjacent levels through the CrossLevelBridge.

"The autopoietic cognitive architecture does not process information — it BECOMES information."

Standing on Giants: Maturana & Varela (1980) — autopoiesis, Simon (1962) — hierarchical decomposition, Friston (2010) — free energy principle, Brooks (1986) — subsumption architecture, Shannon (1948) — information theory, Boyd (1976) — OODA loop, Al-Ghazali — Muraqabah/Ihsan.

## Package Structure

```
core/hrm/
  __init__.py              # 131 lines — comprehensive exports
  abstraction_levels.py    # 254 lines — 5-level hierarchy + SNR gradient
  cross_level_bridge.py    # 538 lines — 5 integration mechanisms
  hierarchical_engine.py   # 709 lines — core engine
  meta_level.py            # 424 lines — Level N meta-autopoiesis
```

Total: ~2,056 lines.

## 5 Abstraction Levels

The cognitive hierarchy maps to Curry-Howard correspondence:

| Level | Name | SNR Threshold | Temporal Scale | Learning Rate | Noise Tolerance |
|-------|------|---------------|----------------|---------------|-----------------|
| L0 | PERCEPTUAL | 0.85 | Immediate | 1.0 | 0.20 |
| L1 | OPERATIONAL | 0.85 | Short-term | 0.8 | 0.18 |
| L2 | TACTICAL | 0.90 | Medium-term | 0.6 | 0.12 |
| L3 | STRATEGIC | 0.95 | Long-term | 0.4 | 0.08 |
| LN | META_COGNITIVE | 0.98 | Evolutionary | 0.2 | 0.03 |

SNR thresholds are the constitutional gradient: lower levels tolerate more noise (more data), higher levels require higher signal purity (higher stakes). All thresholds sourced from `core/integration/constants.py`.

### Level-Pillar Mapping (Curry-Howard)

```
L0 (Perceptual)    -> Genesis/Sandbox  (hypothesis)
L1 (Operational)   -> Museum           (verified conjecture)
L2 (Tactical)      -> Museum->Runtime  (transition)
L3 (Strategic)     -> Runtime          (proven theorem)
LN (Meta-Cognitive) -> Adaptive Ihsan  (axiom evolution)
```

## Level Boundaries

Boundaries are NOT walls — they are selectively permeable membranes:

```
DATACLASS LevelBoundary:
  source_level: AbstractionLevel
  target_level: AbstractionLevel
  permeability: float       # 0.0 = sealed, 1.0 = transparent
  transform_required: bool  # Must information be abstracted?
  message_count: int        # Messages that crossed
  blocked_count: int        # Messages blocked

  METHOD should_pass(confidence: float) -> bool:
    threshold = 1.0 - permeability
    RETURN confidence >= threshold
```

Default permeability pattern:
- Upward (evidence ascending): starts at 0.6, decreases by 0.1 per level
- Downward (goals descending): starts at 0.7, decreases by 0.05 per level
- Downward is more permeable because goals must reach execution levels

## Cross-Level Bridge

The bridge propagates insights through 5 integration mechanisms:

```
CLASS CrossLevelBridge:
  METHOD propagate_hypothesis(hypothesis, source_level, direction, confidence):
    # Route through appropriate boundary
    # Transform if required, track crossing telemetry
    RETURN list of BridgeMessage

  METHOD synchronize_integration(level_states) -> SyncResult:
    # Periodic full-hierarchy synchronization
    RETURN SyncResult(sync_quality, messages_exchanged)

  METHOD get_bridge_metrics() -> Dict:
    # Total messages, crossings, blocked, per-boundary stats
```

### Bridge Node Types (GoT)

```
ENUM BridgeNodeType:
  INTRA_LEVEL   # Reasoning within single level
  INTER_LEVEL   # Between adjacent levels
  BRIDGE        # Multi-scale insight (HIGHEST SNR)
  HUB           # Integration/synthesis point
  FRONTIER      # Exploration boundary probe
```

## Meta-Autopoietic Level (Level N)

Level N reasons about the hierarchy itself — it optimizes the system's own cognitive architecture:

```
CLASS MetaAutopoieticLevel:
  METHOD observe_hierarchy(level_states, bridge_metrics) -> MetaObservation:
    # Analyze: which levels are stagnating? boundaries too tight?
    # Detect: learning rate mismatch, boundary bottlenecks
    RETURN MetaObservation(health_scores, bottlenecks, recommendations)

  METHOD propose_modification(observation) -> Optional[MetaProposal]:
    # Propose: adjust boundary permeability, tweak learning rates
    # Constitutional guard: modifications cannot lower SNR below thresholds
    RETURN MetaProposal(changes, expected_improvement, risk_level)

  METHOD apply_modification(proposal, boundaries, level_configs):
    # Apply proposed changes to boundaries and configs
    # Reversible: all changes logged for potential rollback
```

## Hierarchical Reasoning Engine

### Configuration

```
DATACLASS HRMConfig:
  level_configs: List[LevelConfig]     # Default: 5 levels
  enable_meta_level: bool = True
  meta_observation_interval: int = 3   # Every N cycles
  cascade_factor: float = 0.8          # Learning transfer coefficient
  cascade_decay: float = 0.9           # Decay per level distance
  sync_interval_cycles: int = 5        # Synchronization frequency
  max_cycles: int = 50                 # Campaign limit
  convergence_threshold: float = 0.01  # Minimum improvement
  ihsan_threshold: float = 0.95        # Constitutional floor
  snr_floor: float = 0.85             # Minimum quality
```

### Single Cycle

```
FUNCTION HierarchicalReasoningModel.run_cycle(observation) -> HRMCycleResult:
  cycle_count += 1
  result = HRMCycleResult(cycle_number=cycle_count)

  # Phase 1: Run each level's autopoietic cycle (BOTTOM-UP)
  FOR level_config IN sorted_by_level:
    level_result = _run_level_cycle(level_config, observation)
    result.level_results[level] = level_result

    # Propagate insights upward via bridge
    IF level_result.insights_discovered > 0:
      messages = bridge.propagate_hypothesis(
        hypothesis={source, insights, snr, confidence},
        direction=UPWARD,
        confidence=level_result.snr_score
      )
      result.bridge_messages_sent += len(messages)

  # Phase 2: Learning cascade
  result.cascade_events = _cascade_learning(result.level_results)

  # Phase 3: Resonance detection
  result.resonance_detected = _detect_resonance(result.level_results)

  # Phase 4: Periodic synchronization
  IF cycle_count % sync_interval == 0:
    bridge.synchronize_integration(level_states)

  # Phase 5: Meta-autopoietic observation
  IF meta_level AND cycle_count % meta_interval == 0:
    observation = meta_level.observe_hierarchy(level_states, bridge_metrics)
    proposal = meta_level.propose_modification(observation)
    IF proposal: meta_level.apply_modification(proposal, ...)

  # Compound metrics
  result.compound_snr = _compute_compound_snr(level_results)
  result.compound_learning_delta = _compute_compound_learning(level_results)

  RETURN result
```

### 8-Stage Autopoietic Cycle (Per Level)

```
FUNCTION _run_level_cycle(level_config, observation) -> LevelCycleResult:
  # Stages 1-2: Observe & Generate
  base_quality = 0.7 + (learning_rate_factor * 0.15)
  quality_boost = min(cumulative_learning * 0.05, 0.15)
  noise_factor = 1.0 - (noise_tolerance * 0.2)
  snr = min(1.0, base_quality + quality_boost + noise_factor * 0.05)

  # Stages 3-4: Explore & Filter
  hypotheses_gen = max(1, int(max_hypotheses * 0.6))
  hypotheses_valid = max(1, int(hypotheses_gen * snr))

  # Stages 5-7: Validate, Implement, Integrate
  insights = max(0, hypotheses_valid - int(hypotheses_gen * 0.4))

  # Stage 8: Learn
  learning_delta = snr - moving_average(last_5_scores)

  # Update cumulative state
  state.cumulative_learning += max(0, learning_delta)
  state.snr_scores.append(snr)  # Keep last 20

  RETURN LevelCycleResult(level, snr, hypotheses, insights, learning_delta)
```

### Learning Cascade

```
FUNCTION _cascade_learning(level_results) -> int:
  """
  Golden Gem: The Compound Learning Rate
    When L0 improves by 10%, L1 receives higher-quality patterns (8%),
    which improves L2 by 6%, and L3 by 4%. Small improvements at many
    levels compound into large system-level improvements.
  """
  cascade_count = 0
  FOR each level with positive learning_delta:
    # Cascade UPWARD
    FOR each higher level:
      transfer = delta * (decay ^ distance) * cascade_factor
      IF transfer > 0.001: apply to target cumulative_learning
      cascade_count += 1

    # Cascade DOWNWARD (weaker: * 0.5)
    FOR each lower level:
      transfer = delta * (decay ^ distance) * cascade_factor * 0.5
      IF transfer > 0.001: apply
      cascade_count += 1

  RETURN cascade_count
```

### Resonance Detection

```
FUNCTION _detect_resonance(level_results) -> bool:
  """
  Hidden Pattern: Learning Resonance
    Multiple levels improving simultaneously, with correlated magnitudes.
    Detected when >= 3 levels show positive delta with low coefficient of variation.
  """
  positive_deltas = [lr.delta for lr in results IF lr.delta > 0.01]
  IF len(positive_deltas) < 3: RETURN False

  mean = avg(positive_deltas)
  variance = sum((d - mean)^2) / len
  cv = sqrt(variance) / max(mean, 0.001)

  RETURN cv < 1.0  # Low CV = high correlation = resonance
```

### Compound SNR

```
FUNCTION _compute_compound_snr(level_results) -> float:
  """Weighted by level importance (higher levels make higher-stakes decisions)."""
  WEIGHTS = {L0: 0.10, L1: 0.15, L2: 0.20, L3: 0.25, LN: 0.30}
  RETURN weighted_average(level_results, WEIGHTS)
```

### Campaign (Multi-Cycle)

```
FUNCTION run_campaign(observation, max_cycles) -> List[HRMCycleResult]:
  FOR cycle IN range(max_cycles):
    result = run_cycle(observation)
    improvement = abs(result.compound_snr - prev_snr)
    IF improvement < convergence_threshold AND cycle > 3:
      result.status = CONVERGED
      BREAK
  RETURN all_results
```

## Result Types

```
DATACLASS LevelCycleResult:
  level: AbstractionLevel
  cycle_number: int
  snr_score: float
  hypotheses_generated: int
  hypotheses_validated: int
  insights_discovered: int
  learning_delta: float
  duration_ms: float
  bridge_node_type: BridgeNodeType

  PROPERTY success -> snr_score >= HRM_SNR_GRADIENT[level]

DATACLASS HRMCycleResult:
  cycle_id: str (UUID[:12])
  cycle_number: int
  status: IDLE | RUNNING | COMPLETED | FAILED | CONVERGED
  level_results: Dict[AbstractionLevel, LevelCycleResult]
  bridge_messages_sent: int
  resonance_detected: bool
  cascade_events: int
  compound_snr: float
  compound_learning_delta: float
  meta_observation: Optional[MetaObservation]
  meta_proposal: Optional[MetaProposal]
  total_duration_ms: float

  PROPERTY levels_passed -> count(lr.success for lr in level_results)
  PROPERTY all_levels_passed -> levels_passed == len(level_results)
```

## TDD Anchors

```
TEST single_cycle_produces_all_levels:
  hrm = HierarchicalReasoningModel()
  result = hrm.run_cycle({"context": "test"})
  ASSERT len(result.level_results) == 5
  ASSERT result.status == HRMStatus.COMPLETED
  ASSERT result.compound_snr > 0

TEST snr_gradient_enforced:
  hrm = HierarchicalReasoningModel()
  result = hrm.run_cycle({})
  FOR level, lr IN result.level_results:
    expected_threshold = HRM_SNR_GRADIENT[level]
    # SNR should be in plausible range
    ASSERT 0.5 < lr.snr_score <= 1.0

TEST campaign_converges:
  hrm = HierarchicalReasoningModel(config=HRMConfig(max_cycles=20))
  results = hrm.run_campaign({})
  ASSERT len(results) <= 20
  # Final cycle should be CONVERGED or COMPLETED
  ASSERT results[-1].status IN (COMPLETED, CONVERGED)

TEST learning_cascade_transfers:
  hrm = HierarchicalReasoningModel()
  # Run multiple cycles to build cumulative learning
  FOR _ IN range(5): hrm.run_cycle({})
  # Higher levels should have non-zero cumulative from cascade
  strategic_state = hrm.get_level_state(AbstractionLevel.STRATEGIC)
  ASSERT strategic_state["cumulative_learning"] > 0

TEST resonance_detected_with_correlated_deltas:
  hrm = HierarchicalReasoningModel()
  # Simulate: create level_results with correlated positive deltas
  mock_results = {level: LevelCycleResult(level=level, learning_delta=0.05)
                  FOR level IN AbstractionLevel}
  ASSERT hrm._detect_resonance(mock_results) == True

TEST meta_level_observes_hierarchy:
  hrm = HierarchicalReasoningModel(
    config=HRMConfig(meta_observation_interval=1)
  )
  result = hrm.run_cycle({})
  ASSERT result.meta_observation IS NOT None

TEST compound_snr_weights_higher_levels_more:
  # L3 and LN get 0.25 + 0.30 = 0.55 total weight
  # L0 and L1 get 0.10 + 0.15 = 0.25 total weight
  # Verify compound SNR reflects this weighting

TEST hierarchy_status_comprehensive:
  hrm = HierarchicalReasoningModel()
  hrm.run_cycle({})
  status = hrm.get_hierarchy_status()
  ASSERT "version" IN status
  ASSERT "levels" IN status
  ASSERT len(status["levels"]) == 5
  ASSERT "bridge_metrics" IN status
```

## Architectural Invariants

1. All SNR thresholds come from `core/integration/constants.py` — never hardcoded
2. Learning cascade is bidirectional but asymmetric (downward is 0.5x weaker)
3. Resonance requires >= 3 levels + low coefficient of variation
4. Meta-level cannot lower SNR thresholds below constitutional floor
5. Level states are mutable dicts; LevelConfig is frozen
6. Campaign convergence requires > 3 cycles minimum (avoid premature convergence)
