# 06 — Adaptive Resource Routing

**Status:** SPEC  
**Crate:** `bizra-agent` (extend `omni_kernel.rs`)  
**Depends on:** 03_context_gate (alpha_t), 01_engram_tiered

---

## 1. Motivation

The context gate (alpha_t) reveals a tri-modal distribution across task categories:
- Entity retrieval: mean alpha_t = 0.72 (Engram-dominant, GPU-light)
- Mixed composition: mean alpha_t = 0.48 (balanced)
- Reasoning: mean alpha_t = 0.24 (MoE-dominant, GPU-heavy)

v0.90.0 treats all requests identically — every token goes through the same pipeline
regardless of its knowledge vs reasoning mix. This wastes GPU cycles on entity
retrieval tasks that could be served from Engram.

v0.91.0 uses alpha_t to **route batches** to appropriate execution paths, freeing
GPU capacity for requests that actually need it. Expected impact: +8-12% throughput
on mixed workloads.

## 2. Routing Paths

```
                     ┌──────────────────────┐
                     │  INCOMING REQUEST     │
                     │  (batch of tokens)    │
                     └──────────┬────────────┘
                                │
                     ┌──────────▼────────────┐
                     │  ALPHA_T ESTIMATION   │
                     │  (predict from intent) │
                     └──────────┬────────────┘
                                │
               ┌────────────────┼────────────────┐
               │                │                │
          alpha_t > 0.65   0.35 ≤ α ≤ 0.65  alpha_t < 0.35
               │                │                │
               ▼                ▼                ▼
        ┌────────────┐  ┌────────────┐  ┌────────────┐
        │  GPU-LIGHT  │  │  BALANCED   │  │  GPU-HEAVY  │
        │  Path       │  │  Path       │  │  Path       │
        ├────────────┤  ├────────────┤  ├────────────┤
        │ Engram O(1) │  │ Engram ctx  │  │ Full MoE   │
        │ Minimal MoE │  │ Partial MoE │  │ All experts │
        │ 1-2 GPU ops │  │ 4-8 GPU ops │  │ 16+ GPU ops│
        │ ~50ms       │  │ ~200ms      │  │ ~450ms     │
        └────────────┘  └────────────┘  └────────────┘
```

## 3. Data Structures

```pseudocode
enum RoutingPath:
    GpuLight     # Engram-dominant, minimal GPU
    Balanced     # Mixed Engram + MoE
    GpuHeavy    # MoE-dominant, full GPU

struct RoutingDecision:
    path: RoutingPath
    estimated_alpha_t: f64
    estimated_gpu_ops: u32
    estimated_latency_ms: f64
    batch_id: u64

struct RoutingConfig:
    gpu_light_threshold: f64    # alpha_t above this → GPU-light (default: 0.65)
    gpu_heavy_threshold: f64    # alpha_t below this → GPU-heavy (default: 0.35)
    max_gpu_light_fraction: f64 # max fraction of requests on light path (0.6)
    min_gpu_heavy_fraction: f64 # min fraction reserved for heavy path (0.2)

struct RoutingTelemetry:
    light_count: u64
    balanced_count: u64
    heavy_count: u64
    light_avg_latency_ms: RunningMean
    balanced_avg_latency_ms: RunningMean
    heavy_avg_latency_ms: RunningMean
    gpu_utilization: f64         # EMA of GPU busy fraction
```

## 4. Intent-Based Alpha Estimation

Before the full context gate (which requires a hidden state), we need a fast
**pre-routing** estimate based on the intent string alone:

```pseudocode
struct AlphaEstimator:
    # Historical: intent_hash → observed mean alpha_t
    history: HashMap<[u8;32], RunningMean>
    # Keyword heuristics for cold-start
    entity_keywords: HashSet<String>   # "what is", "who is", "define", "capital of"
    reasoning_keywords: HashSet<String> # "why", "prove", "explain how", "compare"

fn estimate_alpha_t(estimator, intent) -> f64:
    """
    Fast pre-routing estimate. O(1) hash lookup + keyword scan.
    Accuracy improves over time as history accumulates.
    """
    
    # Strategy 1: Historical lookup (most accurate)
    intent_hash = blake3(intent.as_bytes())
    if mean = estimator.history.get(intent_hash):
        if mean.count >= 5:  # enough data to trust
            return mean.value
    
    # Strategy 2: Similar intent lookup (fuzzy match via prefix hash)
    prefix_hash = blake3(intent[..min(20, intent.len())].as_bytes())
    if mean = estimator.history.get(prefix_hash):
        if mean.count >= 10:
            return mean.value
    
    # Strategy 3: Keyword heuristic (cold-start fallback)
    entity_score = count_keyword_matches(intent, estimator.entity_keywords)
    reasoning_score = count_keyword_matches(intent, estimator.reasoning_keywords)
    
    if entity_score > reasoning_score:
        return 0.70  # likely entity retrieval
    elif reasoning_score > entity_score:
        return 0.25  # likely reasoning
    else:
        return 0.48  # balanced default

fn update_alpha_history(estimator, intent, actual_alpha_t):
    """Post-routing feedback: refine estimates with observed alpha_t."""
    intent_hash = blake3(intent.as_bytes())
    estimator.history.get_or_default(intent_hash).update(actual_alpha_t)
```

## 5. Batch Routing

```pseudocode
fn route_batch(router, requests, config) -> Vec<(Request, RoutingDecision)>:
    """
    Route a batch of incoming requests to execution paths.
    Respects capacity constraints (don't overload GPU-light path).
    """
    decisions = []
    light_count = 0
    heavy_count = 0
    total = requests.len()
    
    for request in requests:
        estimated_alpha = estimate_alpha_t(router.estimator, request.intent)
        
        path = match:
            estimated_alpha >= config.gpu_light_threshold => GpuLight
            estimated_alpha <= config.gpu_heavy_threshold => GpuHeavy
            _ => Balanced
        
        # Capacity check: don't overload any path
        if path == GpuLight and (light_count as f64 / total as f64) >= config.max_gpu_light_fraction:
            path = Balanced  # overflow to balanced
        if path == GpuHeavy and (heavy_count as f64 / total as f64) < config.min_gpu_heavy_fraction:
            # Ensure minimum GPU-heavy capacity is maintained
            pass  # keep as GpuHeavy
        
        decisions.push((request, RoutingDecision {
            path,
            estimated_alpha_t: estimated_alpha,
            estimated_gpu_ops: match path {
                GpuLight => 2,
                Balanced => 8,
                GpuHeavy => 16,
            },
            estimated_latency_ms: match path {
                GpuLight => 50.0,
                Balanced => 200.0,
                GpuHeavy => 450.0,
            },
            batch_id: router.next_batch_id(),
        }))
        
        match path:
            GpuLight => light_count += 1
            GpuHeavy => heavy_count += 1
            _ => ()
    
    return decisions
```

## 6. Integration with OmniKernel

```pseudocode
# In OmniKernel.run_cycle():

fn run_cycle_routed(kernel, cycle, routing_decision, ...):
    match routing_decision.path:
        GpuLight =>
            # Skip full MoE routing — go straight to Engram lookup
            result = kernel.engram_cache.lookup(cycle.intent_bytes, min_confidence)
            if result.is_hit():
                # Minimal post-processing (SFE gate only)
                return build_receipt(result, CyclePath::EngramHit)
            else:
                # Misrouted — fallback to balanced path
                routing_decision.path = Balanced
                # fall through
        
        Balanced =>
            # Standard OmniKernel cycle (existing run_cycle)
            return kernel.run_cycle(cycle, ...)
        
        GpuHeavy =>
            # Full MoE with all experts activated
            # Skip Engram lookup entirely (alpha_t too low to be useful)
            return kernel.run_cycle_full_moe(cycle, ...)
```

## 7. TDD Anchors

```
TEST routing_01: high alpha_t routes to GpuLight
    estimate = 0.72
    decision = route_request(request, estimate, config)
    ASSERT decision.path == GpuLight

TEST routing_02: low alpha_t routes to GpuHeavy
    estimate = 0.20
    decision = route_request(request, estimate, config)
    ASSERT decision.path == GpuHeavy

TEST routing_03: mid alpha_t routes to Balanced
    estimate = 0.48
    decision = route_request(request, estimate, config)
    ASSERT decision.path == Balanced

TEST routing_04: GpuLight overflow redirects to Balanced
    SET max_gpu_light_fraction = 0.6
    CREATE batch of 10 requests, all with alpha > 0.65
    decisions = route_batch(batch, config)
    light_count = count(d.path == GpuLight for d in decisions)
    ASSERT light_count <= 6

TEST routing_05: alpha estimator improves with history
    # Cold start: keyword heuristic
    est1 = estimate_alpha_t("what is the capital of France")
    ASSERT est1 ≈ 0.70  # entity keyword match
    
    # After history: more accurate
    update_alpha_history("what is the capital of France", 0.85)
    update_alpha_history("what is the capital of France", 0.82)
    update_alpha_history("what is the capital of France", 0.88)
    update_alpha_history("what is the capital of France", 0.84)
    update_alpha_history("what is the capital of France", 0.86)
    est2 = estimate_alpha_t("what is the capital of France")
    ASSERT abs(est2 - 0.85) < 0.05  # much closer to true alpha

TEST routing_06: GpuLight misroute falls back to Balanced
    route request to GpuLight
    Engram returns Miss
    ASSERT fallback path == Balanced

TEST routing_07: telemetry tracks per-path latency
    RUN 100 cycles across all paths
    ASSERT telemetry.light_avg_latency_ms.count > 0
    ASSERT telemetry.heavy_avg_latency_ms.count > 0

TEST routing_08: reasoning keywords route to GpuHeavy
    estimate = estimate_alpha_t("prove that P implies Q using modus ponens")
    ASSERT estimate < 0.35

TEST routing_09: batch routing respects min_gpu_heavy_fraction
    SET min_gpu_heavy_fraction = 0.2
    CREATE batch of 10 requests, all entity retrieval
    decisions = route_batch(batch, config)
    # At least 2 should still be GPU-heavy (reserve capacity)
    # Note: this only applies if there are heavy requests — pure-entity batches
    # don't force heavy allocation. Test with mixed batch instead.

TEST routing_10: empty batch returns empty decisions
    decisions = route_batch([], config)
    ASSERT decisions.is_empty()
```

## 8. Edge Cases

- **Misrouted requests**: If GpuLight is chosen but Engram misses, the request
  must fall back without double-counting latency. The fallback path adds routing
  overhead (~5ms) but prevents incorrect answers.
- **All requests same category**: A batch of 100% entity retrieval requests should
  not force any to GpuHeavy. The min_gpu_heavy_fraction is a lower bound only
  when heavy requests exist in the batch.
- **Estimator cold start**: First 100 requests use keyword heuristics only. Accept
  higher misroute rate (~15%) during warmup.
- **Workload shift**: If the task mix changes abruptly, the history-based estimator
  lags. Exponential decay (alpha = 0.1) on historical means ensures adaptation
  within ~20 requests.
