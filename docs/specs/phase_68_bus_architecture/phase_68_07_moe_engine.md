# Phase 68.07 — MOE Engine: 5-Expert Routing Engine

## Specification + Pseudocode

**Status:** SPEC-READY
**Priority:** P0 (Week 2 Genesis Sprint)
**Estimated LOC:** ~350 new
**Files created:** 2 (`core/living_model/moe_engine.py`, `core/living_model/__init__.py`)
**Files modified:** 2 (`core/sovereign/api.py`, `core/integration/constants.py`)
**Dependencies:** ReflexCompiler (Phase 80), constants.py
**TDD anchors:** 45+ tests required

---

## 1. Problem Statement

The `/v1/plan` endpoint routes ALL queries through a single monolithic pipeline.
There is no expert specialization — a governance question uses the same path as
a knowledge retrieval or a skill execution request.

The DDAGI Pilot (v2.0) defines 12 agents (7 PAT + 5 SAT), but no routing
engine selects which expert(s) handle a given input. The `time.sleep(0.1)`
diffusion stub in the proof harness confirms this gap.

**What exists:**
- ReflexCompiler (System-1 cache, O(1) lookup) — shipped Week 1
- MissionOrchestrator (6-phase pipeline) — shipped Phase 57
- HHMM state taxonomy (47 states, 5 initial live) — defined in constants.py

**What's missing:**
- Expert selection router (input → expert assignment)
- Per-expert scoring (each expert rates its confidence)
- Synthesis combiner (weighted merge of expert outputs)
- Constitutional gate on combined output (Ihsan >= 0.95)

---

## 2. Architecture

```
                    ┌─────────────────┐
  Input ──────────► │  HHMM Router    │ ──► macro_state prediction
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Expert-R │  │ Expert-K │  │ Expert-S │  ... (up to 5)
        │ Reasoning│  │ Knowledge│  │ Skills   │
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             │              │              │
             ▼              ▼              ▼
        ┌─────────────────────────────────────┐
        │         Synthesis Combiner          │
        │  (weighted by confidence × Ihsan)   │
        └──────────────────┬──────────────────┘
                           │
                    ┌──────▼──────┐
                    │  P5 Gate    │ ──► Ihsan >= 0.95?
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Evidence   │ ──► Receipt
                    └─────────────┘
```

---

## 3. Expert Definitions

| Expert | ID | Domain | Activation Signal |
|--------|----|--------|-------------------|
| **Expert-R** (Reasoning) | `pat_r` | Logic, planning, decomposition | Questions with "how", "why", multi-step |
| **Expert-K** (Knowledge) | `pat_k` | Retrieval, facts, memory lookup | Questions with "what", "who", "when" |
| **Expert-S** (Skills) | `pat_s` | Code gen, file ops, tool use | Action verbs, tool references |
| **Expert-G** (Governance) | `sat_g` | Constitutional checks, policy | Governance terms, threshold refs |
| **Expert-V** (Verification) | `sat_v` | Proof, evidence, validation | "Verify", "prove", "check" |

Each expert is a lightweight scoring function, NOT a full LLM call.
Experts rate their relevance (0.0-1.0) for a given input.
Top-K experts (K=2 default) are activated.

---

## 4. Functional Requirements

### FR-1: Expert Router

```
FUNCTION route(input_text: str, context: dict) -> List[ExpertAssignment]:
    # Step 1: HHMM macro-state prediction (if available)
    macro_state = hhmm.predict_state(input_text) if hhmm else "general"

    # Step 2: Each expert scores its relevance
    scores = {}
    FOR expert IN [Expert_R, Expert_K, Expert_S, Expert_G, Expert_V]:
        scores[expert.id] = expert.score_relevance(input_text, macro_state, context)

    # Step 3: Select top-K experts (K from constants.py)
    top_k = sorted(scores, key=scores.get, reverse=True)[:MOE_TOP_K]

    # Step 4: Normalize weights
    total = sum(scores[e] for e in top_k)
    IF total == 0:
        RETURN [ExpertAssignment("pat_r", weight=1.0)]  # fallback to reasoning

    RETURN [ExpertAssignment(e, weight=scores[e]/total) for e in top_k]

INVARIANT: len(result) >= 1
INVARIANT: sum(weights) == 1.0 (within floating point tolerance)
INVARIANT: all(0.0 <= w <= 1.0 for w in weights)
```

### FR-2: Expert Scoring

Each expert uses keyword + HHMM state matching. No ML model required at V1.

```
CLASS Expert:
    id: str
    keywords: FrozenSet[str]          # activation keywords
    hhmm_states: FrozenSet[str]       # preferred macro-states
    base_weight: float                # prior (from constants.py)

    FUNCTION score_relevance(input_text, macro_state, context) -> float:
        keyword_score = count_keyword_hits(input_text, self.keywords) / max(len(self.keywords), 1)
        state_score = 1.0 IF macro_state IN self.hhmm_states ELSE 0.0
        context_score = self._context_boost(context)

        raw = (keyword_score * 0.5) + (state_score * 0.3) + (context_score * 0.2)
        RETURN clamp(raw * self.base_weight, 0.0, 1.0)

INVARIANT: score_relevance always returns [0.0, 1.0]
INVARIANT: empty input returns base_weight * 0.2 (context default)
```

### FR-3: Synthesis Combiner

```
FUNCTION synthesize(expert_results: List[ExpertResult]) -> SynthesisResult:
    # Weighted combination of expert outputs
    combined_text = ""
    combined_ihsan = 0.0

    FOR result, assignment IN zip(expert_results, assignments):
        combined_text += f"\n[{assignment.expert_id}] {result.text}"
        combined_ihsan += result.ihsan * assignment.weight

    # Constitutional gate
    IF combined_ihsan < UNIFIED_IHSAN_THRESHOLD:
        RETURN SynthesisResult(
            text=combined_text,
            ihsan=combined_ihsan,
            passed_gate=False,
            reason="Combined Ihsan below threshold"
        )

    RETURN SynthesisResult(
        text=combined_text,
        ihsan=combined_ihsan,
        passed_gate=True,
        experts_used=[a.expert_id for a in assignments]
    )

INVARIANT: combined_ihsan is weighted average (not sum)
INVARIANT: gate uses UNIFIED_IHSAN_THRESHOLD from constants.py
```

### FR-4: Integration with ReflexCompiler

The MOE Engine sits BETWEEN the ReflexCompiler cache check and the
MissionOrchestrator. The data flow is:

```
/v1/plan request
  │
  ├─► ReflexCompiler.lookup() → HIT? Return cached (System-1)
  │
  ├─► MOE Router → select experts → execute → synthesize (System-2)
  │
  ├─► ReflexCompiler.record_observation() → feed precipitation
  │
  └─► Evidence Ledger → receipt
```

### FR-5: Backward Compatibility

- `/v1/plan` with no MOE flag uses MOE transparently (default K=2)
- `/v1/plan` with `expert_override` parameter forces specific expert(s)
- All existing tests pass without modification
- ReflexCompiler cache keys unchanged (input_text hash, not expert selection)

---

## 5. Data Structures

```python
@dataclass(frozen=True)
class ExpertAssignment:
    expert_id: str          # "pat_r", "pat_k", "pat_s", "sat_g", "sat_v"
    weight: float           # 0.0 - 1.0, normalized

@dataclass(frozen=True)
class ExpertResult:
    expert_id: str
    text: str
    ihsan: float
    confidence: float       # expert's self-assessed confidence
    latency_ms: float

@dataclass(frozen=True)
class SynthesisResult:
    text: str
    ihsan: float
    passed_gate: bool
    reason: str = ""
    experts_used: tuple = ()
    total_latency_ms: float = 0.0

@dataclass
class MOEEngineStats:
    total_routes: int = 0
    expert_activations: dict = field(default_factory=dict)  # expert_id -> count
    avg_experts_per_query: float = 0.0
    gate_rejections: int = 0
```

---

## 6. Constants (additions to constants.py)

```python
# MOE Engine — Phase 68.07
MOE_EXPERT_COUNT = 5
MOE_TOP_K = 2                          # experts activated per query
MOE_FALLBACK_EXPERT = "pat_r"          # reasoning is default
MOE_MIN_CONFIDENCE = 0.1               # below this, expert is skipped
MOE_SYNTHESIS_STRATEGY = "weighted"    # "weighted" | "best_of" | "consensus"
```

---

## 7. TDD Anchors

### 7.1 Unit Tests (`tests/core/living_model/test_moe_engine.py`)

```
class TestExpertScoring:
    test_reasoning_expert_activates_on_how_questions
    test_knowledge_expert_activates_on_what_questions
    test_skills_expert_activates_on_action_verbs
    test_governance_expert_activates_on_policy_terms
    test_verification_expert_activates_on_prove_terms
    test_empty_input_returns_base_weight
    test_score_always_in_zero_one_range
    test_hhmm_state_boosts_matching_expert

class TestRouter:
    test_route_returns_at_least_one_expert
    test_route_weights_sum_to_one
    test_top_k_selects_highest_scorers
    test_top_k_default_is_two
    test_fallback_to_reasoning_on_zero_scores
    test_expert_override_bypasses_router
    test_all_five_experts_can_be_selected

class TestSynthesis:
    test_weighted_combination_produces_correct_ihsan
    test_gate_rejects_below_threshold
    test_gate_passes_at_threshold
    test_single_expert_result
    test_all_five_experts_combined
    test_empty_expert_results_returns_failure

class TestMOEEngine:
    test_full_pipeline_route_to_synthesis
    test_engine_stats_track_activations
    test_engine_stats_track_rejections
    test_engine_thread_safe
    test_engine_with_hhmm_none_fallback

class TestIntegration:
    test_moe_sits_between_reflex_and_orchestrator
    test_reflex_cache_hit_skips_moe
    test_moe_result_feeds_reflex_observation
    test_backward_compatible_plan_endpoint
```

### 7.2 Property-Based Tests

```
@given(input_text=text(), k=integers(1, 5))
def test_router_invariants(input_text, k):
    result = router.route(input_text, {}, top_k=k)
    assert len(result) >= 1
    assert abs(sum(a.weight for a in result) - 1.0) < 1e-6
    assert all(0.0 <= a.weight <= 1.0 for a in result)

@given(ihsan_scores=lists(floats(0.0, 1.0), min_size=1, max_size=5))
def test_synthesis_ihsan_is_weighted_average(ihsan_scores):
    # weighted average is always between min and max of inputs
    result = synthesize(mock_results(ihsan_scores))
    assert min(ihsan_scores) <= result.ihsan <= max(ihsan_scores)
```

---

## 8. Error Handling

- **No experts score above MOE_MIN_CONFIDENCE:** Fallback to `pat_r` with weight 1.0
- **Expert execution fails:** Skip failed expert, re-normalize remaining weights
- **All experts fail:** Return error result with `passed_gate=False`
- **HHMM unavailable:** Route using keyword scoring only (graceful degradation)
- **Thread safety:** `threading.Lock` around stats updates only (routing is stateless)

---

## 9. Performance Budget

| Operation | Target | Method |
|-----------|--------|--------|
| Router (5 experts scored) | < 1 ms | Keyword counting, no ML |
| Single expert execution | < 500 ms | Depends on downstream |
| Synthesis combiner | < 1 ms | String concat + weighted avg |
| Total MOE overhead | < 5 ms | Excludes expert execution |

---

## 10. Standing on Giants

- **Shazeer (2017):** Sparsely-gated Mixture of Experts — top-K routing
- **Kahneman (2011):** System-2 deliberation uses multiple experts, not one
- **Ibn Khaldun (1377):** Asabiyyah — collective intelligence from specialized roles
- **Boyd (1976):** OODA loop — observe (score), orient (route), decide (synthesize), act (gate)
