# Module 03 — Cognition Engine

> **Domain:** Reasoning, HRM, NorthStar, RLM, entropy routing, SNR
> **Source Specs:** Phase 47 (cognitive resonance), Phase 50 (RLM), Phase 43 (identity)
> **Key Paths:** `core/reasoning/`, `core/inference/`, `core/resonance/`, `core/apex/`

## 3.1 Graph-of-Thoughts (GoT) Reasoning

**Status:** [x] BUILT
**Path:** `core/reasoning/`

Multi-hypothesis reasoning engine. Generates up to `GOT_MAX_HYPOTHESES = 5`
parallel thought branches, evaluates each, converges on best path.

Standing on Giants: Besta et al. — Graph-of-Thoughts framework

**Tests:** `tests/core/reasoning/`

---

## 3.2 Tiered LLM Gateway

**Status:** [x] BUILT
**Path:** `core/inference/`

Three-tier inference with automatic failover:
1. LM Studio at WSL gateway (auto-detected, `_detect_wsl_gateway()`)
2. Ollama at localhost:11434 (5 models loaded)
3. Cloud API (emergency fallback)

**Resilience:** `CircuitBreaker` in `core/inference/_resilience.py`
**Auto-detection:** WSL gateway IP discovered via `ip route show default` (never hardcoded)

**Tests:** `tests/core/inference/` — 11+ test files

---

## 3.3 SNR Apex Engine

**Status:** [x] BUILT
**Path:** `core/apex/snr_apex_engine.py`

Top-level SNR scoring engine. Computes signal-to-noise ratio for any
content or inference output. Thresholds from constants.py.

---

## 3.4 SNR v2 Adapter

**Status:** [x] BUILT
**Path:** `core/iaas/snr_v2_adapter.py`

Adapter layer for SNR v2 protocol. Bridges between old SNR interface
and new apex engine.

---

## 3.5 Dual-Process Cognition (Kahneman)

**Status:** [x] BUILT
**Path:** `core/inference/` (fast/slow routing)

System 1 (fast, cached) and System 2 (slow, deliberate) routing.
Simple queries hit cache; complex queries trigger full reasoning pipeline.

Standing on Giants: Kahneman — Thinking, Fast and Slow

---

## 3.6 Entropy-Based Routing

**Status:** [x] BUILT
**Path:** `core/inference/` (routing logic)

High-entropy queries (novel, uncertain) routed to stronger models.
Low-entropy queries (familiar, cached) handled locally.

---

## 3.7 Active Inference (Friston)

**Status:** [x] BUILT
**Path:** `core/sovereign/` (integrated into OODA loop)

Free Energy Principle applied to decision-making. Node minimizes
surprise by predicting and acting to confirm predictions.

Standing on Giants: Friston — Active Inference

---

## 3.8 OmniKernel Cognitive Cycle (Rust)

**Status:** [x] BUILT
**Path:** `bizra-omega/bizra-agent/`

7-line cognitive cycle: `try_cache_hit(&self)` + `complete_cache_hit(&mut self)`.
Two-phase R/W pattern for lock-free concurrent cognition.

---

## 3.9 Adaptive Prior System

**Status:** [x] BUILT
**Path:** `core/inference/` (adaptive priors)

Bayesian prior updating based on evidence accumulation.
Dict iteration order matters for keyword matching (documented gotcha).

---

## 3.10 Zero-Point Kernel (ZPK)

**Status:** [x] BUILT
**Path:** `core/zpk/kernel.py` (953 LOC)

Foundational axioms and zero-state computations. Full implementation,
not a stub — provides the axiomatic basis for cognition bootstrap.

---

## 3.11 Cognitive Resonance

**Status:** [~] PARTIAL
**Path:** `core/resonance.py` (single file)
**Spec:** Phase 47 — full resonance activation with cross-agent synchronization
**Gap:** Single-file implementation. No cross-agent resonance, no frequency tuning.

### TDD Anchor
```
def test_cross_agent_resonance():
    r = CognitiveResonanceEngine()
    r.register_agent("agent_a", frequency=0.8)
    r.register_agent("agent_b", frequency=0.75)
    sync = r.compute_resonance(["agent_a", "agent_b"])
    assert sync.coherence > 0.5  # Agents in partial sync
    assert sync.interference_pattern is not None
```

---

## 3.12 HRM (Hierarchical Reasoning Model)

**Status:** [x] BUILT
**Path:** `core/hrm/` (~2,014 LOC)

Full module with 4 files:
- `hierarchical_engine.py` (~900 LOC) — multi-level reasoning with cross-layer bridges
- `abstraction_levels.py` (~400 LOC) — abstraction hierarchy definitions
- `meta_level.py` (~450 LOC) — meta-level reasoning over abstractions
- `cross_level_bridge.py` (~264 LOC) — bridge between abstraction levels

---

## 3.13 NorthStar Goal Alignment

**Status:** [~] PARTIAL
**Path:** Mission orchestrator has goal decomposition, no standalone NorthStar
**Gap:** No persistent goal registry, no goal-drift detection

### TDD Anchor
```
def test_northstar_goal_alignment():
    ns = NorthStarEngine()
    ns.set_goal("universal_digital_sovereignty")
    alignment = ns.measure_alignment(current_actions=actions_list)
    assert alignment.score > 0.0
    assert alignment.drift_warning is False  # On track
```

---

## 3.14 RLM Integration (Recursive Language Model)

**Status:** [ ] NOT BUILT
**Path:** `core/inference/rlm_bridge.py` (bridge file exists, no implementation)
**Spec:** Phase 50 — MIT CSAIL RLM for 10M+ token context scaling
**Gap:** Bridge file is placeholder. No REPL integration, no token scaling.

### Pseudocode
```
class RLMBridge:
    """Bridge to Recursive Language Model for 10M+ token contexts"""

    def __init__(self, model_path: str):
        self.model = load_rlm(model_path)
        self.context_window = 10_000_000  # 10M tokens

    def recursive_query(self, query: str, context: LargeContext) -> RLMResult:
        # Chunk context into recursive segments
        segments = self.segment_context(context, chunk_size=100_000)
        # Process recursively — each level summarizes and feeds up
        summaries = [self.model.summarize(seg) for seg in segments]
        meta_summary = self.model.synthesize(summaries)
        return self.model.query(query, context=meta_summary)

    def segment_context(self, ctx: LargeContext, chunk_size: int) -> List[Segment]:
        return [ctx[i:i+chunk_size] for i in range(0, len(ctx), chunk_size)]
```

---

## Completion

| Feature | Status | Coverage |
|---------|--------|----------|
| 3.1 Graph-of-Thoughts | BUILT | Full |
| 3.2 Tiered LLM Gateway | BUILT | 11+ tests |
| 3.3 SNR Apex | BUILT | Full |
| 3.4 SNR v2 Adapter | BUILT | Full |
| 3.5 Dual-Process | BUILT | Routing |
| 3.6 Entropy Routing | BUILT | Full |
| 3.7 Active Inference | BUILT | OODA |
| 3.8 OmniKernel | BUILT | Rust |
| 3.9 Adaptive Prior | BUILT | Full |
| 3.10 ZPK | BUILT | Stub |
| 3.11 Resonance | PARTIAL | Single file |
| 3.12 HRM | BUILT | 2,014 LOC |
| 3.13 NorthStar | PARTIAL | No registry |
| 3.14 RLM | NOT BUILT | Bridge only |
| **TOTAL** | **11/14 + 2P + 1N** | **82%** |
