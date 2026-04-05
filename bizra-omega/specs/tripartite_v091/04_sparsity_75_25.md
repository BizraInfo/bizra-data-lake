# 04 — 75/25 Sparsity Allocation Law

**Status:** SPEC  
**Crate:** `bizra-core` (new `sparsity_law.rs`)  
**Constitutional:** YES — hard constraint, fail-closed

---

## 1. Motivation

The 75/25 law is the central design principle governing parameter distribution between
MoE experts (computation sparsity) and Engram memory (knowledge sparsity).

Empirical U-curve analysis across six model scales (7B–300B) confirms:

- **< 20% Engram**: compute-bound regime — transformer wastes attention on static retrieval
- **24–26% Engram**: optimal — rho (sparsity efficiency) = 74.3–75.0%
- **> 30% Engram**: memory-bound regime — MoE starved of capacity for dynamic reasoning

The optimal point is remarkably stable: ±2pp variance across all scales and task domains.

## 2. Constants

```pseudocode
# Constitutional constants — source: bizra-core/src/lib.rs
ENGRAM_ALLOCATION_RATIO     = 0.25    # target: 25% of sparse params to Engram
ENGRAM_ALLOCATION_TOLERANCE = 0.02    # ±2pp acceptable variance
ENGRAM_ALLOCATION_MIN       = 0.23    # floor = 0.25 - 0.02
ENGRAM_ALLOCATION_MAX       = 0.27    # ceiling = 0.25 + 0.02

# Derived from evaluation data (Table 6)
MODEL_ALLOCATIONS = {
    "BIZRA-7B":   { total: 7.1B,   moe: 5.3B,   engram: 1.8B,  ratio: 0.254 },
    "BIZRA-13B":  { total: 13.1B,  moe: 9.8B,   engram: 3.3B,  ratio: 0.252 },
    "BIZRA-27B":  { total: 27.1B,  moe: 20.3B,  engram: 6.8B,  ratio: 0.251 },
    "BIZRA-70B":  { total: 70.0B,  moe: 52.5B,  engram: 17.5B, ratio: 0.250 },
    "BIZRA-140B": { total: 140.0B, moe: 105.0B, engram: 35.0B, ratio: 0.250 },
    "BIZRA-300B": { total: 300.0B, moe: 225.0B, engram: 75.0B, ratio: 0.250 },
}
```

## 3. Data Structures

```pseudocode
struct SparsityAllocation:
    total_sparse_params: u64     # total parameters in sparse modules
    moe_params: u64              # parameters allocated to MoE experts
    engram_params: u64           # parameters allocated to Engram tables
    dense_params: u64            # non-sparse parameters (attention, embeddings)

struct SparsityValidation:
    ratio: f64                   # engram_params / total_sparse_params
    rho: f64                     # sparsity efficiency = moe_params / total_sparse_params
    is_compliant: bool           # ratio within [ENGRAM_ALLOCATION_MIN, ENGRAM_ALLOCATION_MAX]
    deviation_pp: f64            # distance from 0.25 in percentage points
    regime: SparsityRegime

enum SparsityRegime:
    ComputeBound      # ratio < ENGRAM_ALLOCATION_MIN (too little Engram)
    Optimal           # ENGRAM_ALLOCATION_MIN <= ratio <= ENGRAM_ALLOCATION_MAX
    MemoryBound       # ratio > ENGRAM_ALLOCATION_MAX (too much Engram)
```

## 4. Validation

```pseudocode
fn validate_allocation(allocation: SparsityAllocation) -> SparsityValidation:
    """
    Validate that the parameter allocation complies with the 75/25 law.
    This is a constitutional gate — violations MUST block deployment.
    """
    
    if allocation.total_sparse_params == 0:
        return SparsityValidation {
            ratio: 0.0,
            rho: 0.0,
            is_compliant: false,
            deviation_pp: 25.0,
            regime: ComputeBound,
        }
    
    ratio = allocation.engram_params as f64 / allocation.total_sparse_params as f64
    rho = allocation.moe_params as f64 / allocation.total_sparse_params as f64
    
    # Sanity: moe + engram should equal total sparse
    assert abs((allocation.moe_params + allocation.engram_params) 
               - allocation.total_sparse_params) < 1000  # tolerance for rounding
    
    deviation_pp = (ratio - ENGRAM_ALLOCATION_RATIO) * 100.0
    
    regime = match:
        ratio < ENGRAM_ALLOCATION_MIN => ComputeBound
        ratio > ENGRAM_ALLOCATION_MAX => MemoryBound
        _ => Optimal
    
    return SparsityValidation {
        ratio,
        rho,
        is_compliant: regime == Optimal,
        deviation_pp,
        regime,
    }
```

## 5. Enforcement Points

The 75/25 law is enforced at three points:

### 5a. Model Compilation (Engram Compiler)

```pseudocode
fn compile_engram_allocation(total_sparse, model_config) -> (u64, u64):
    """
    Given total sparse parameter budget, compute MoE and Engram allocations.
    Called once during model compilation.
    """
    engram_target = (total_sparse as f64 * ENGRAM_ALLOCATION_RATIO) as u64
    moe_target = total_sparse - engram_target
    
    # Validate before returning
    allocation = SparsityAllocation {
        total_sparse_params: total_sparse,
        moe_params: moe_target,
        engram_params: engram_target,
        dense_params: model_config.dense_params,
    }
    validation = validate_allocation(allocation)
    assert validation.is_compliant, 
        f"Allocation violates 75/25 law: ratio={validation.ratio:.4f}"
    
    return (moe_target, engram_target)
```

### 5b. Runtime Monitoring (Metabolic Ledger)

```pseudocode
fn check_runtime_allocation(kernel) -> SparsityValidation:
    """
    Periodically verify that the live allocation hasn't drifted.
    Drift can occur from dynamic expert pruning or Engram table growth.
    """
    live_engram = kernel.engram_cache.total_param_bytes()
    live_moe = kernel.moe_router.total_param_bytes()
    total = live_engram + live_moe
    
    validation = validate_allocation(SparsityAllocation {
        total_sparse_params: total,
        moe_params: live_moe,
        engram_params: live_engram,
        dense_params: 0,  # not tracked at runtime
    })
    
    if not validation.is_compliant:
        emit_event("SPARSITY_DRIFT_DETECTED", {
            ratio: validation.ratio,
            regime: validation.regime,
            deviation_pp: validation.deviation_pp,
        })
        # Trigger corrective action (see 5c)
    
    return validation
```

### 5c. Corrective Rebalancing

```pseudocode
fn rebalance_allocation(kernel, validation):
    """
    If allocation drifts out of bounds, correct it.
    This is a soft correction — not a hard stop. The constitutional gate
    at deployment (5a) is the hard gate.
    """
    
    if validation.regime == ComputeBound:
        # Too little Engram — promote more entries from cold storage
        deficit_params = compute_deficit(validation, ENGRAM_ALLOCATION_RATIO)
        kernel.engram_compiler.schedule_additional_entries(deficit_params)
    
    elif validation.regime == MemoryBound:
        # Too much Engram — evict low-value entries
        surplus_params = compute_surplus(validation, ENGRAM_ALLOCATION_RATIO)
        kernel.engram_cache.evict_lowest_value(surplus_params)
```

## 6. U-Curve Validation

The U-curve relationship between Engram allocation and performance loss:

```pseudocode
fn compute_u_curve_point(allocation_ratio, benchmark_results) -> f64:
    """
    Compute performance loss relative to optimal allocation.
    Used in ablation testing (spec 07) to validate the 75/25 law.
    """
    baseline_score = benchmark_results.at_ratio(ENGRAM_ALLOCATION_RATIO)
    current_score = benchmark_results.at_ratio(allocation_ratio)
    
    # Performance loss as percentage
    return max(0.0, (baseline_score - current_score) / baseline_score * 100.0)

fn validate_u_curve(test_ratios, benchmark_results) -> bool:
    """
    Validate that the U-curve has its minimum within [0.23, 0.27].
    
    test_ratios: [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    """
    losses = [(r, compute_u_curve_point(r, benchmark_results)) for r in test_ratios]
    min_loss_ratio = min(losses, key=lambda x: x[1])[0]
    
    return ENGRAM_ALLOCATION_MIN <= min_loss_ratio <= ENGRAM_ALLOCATION_MAX
```

## 7. Cross-Scale Stability

```pseudocode
fn validate_cross_scale_stability(model_configs) -> bool:
    """
    Confirm that the 75/25 optimum holds across model scales.
    The evaluation shows ±2pp variance — this test enforces that.
    """
    ratios = []
    for config in model_configs:
        optimal = find_optimal_ratio(config)  # from ablation
        ratios.push(optimal)
    
    mean_ratio = mean(ratios)
    max_deviation = max(abs(r - mean_ratio) for r in ratios)
    
    return max_deviation <= ENGRAM_ALLOCATION_TOLERANCE
```

## 8. TDD Anchors

```
TEST sparsity_01: 25% allocation is Optimal regime
    allocation = SparsityAllocation(total=100B, moe=75B, engram=25B)
    result = validate_allocation(allocation)
    ASSERT result.is_compliant == true
    ASSERT result.regime == Optimal
    ASSERT abs(result.ratio - 0.25) < 1e-9

TEST sparsity_02: 20% allocation is ComputeBound (below 23% floor)
    allocation = SparsityAllocation(total=100B, moe=80B, engram=20B)
    result = validate_allocation(allocation)
    ASSERT result.is_compliant == false
    ASSERT result.regime == ComputeBound

TEST sparsity_03: 30% allocation is MemoryBound (above 27% ceiling)
    allocation = SparsityAllocation(total=100B, moe=70B, engram=30B)
    result = validate_allocation(allocation)
    ASSERT result.is_compliant == false
    ASSERT result.regime == MemoryBound

TEST sparsity_04: boundary values 23% and 27% are still Optimal
    alloc_low = SparsityAllocation(total=100B, moe=77B, engram=23B)
    alloc_high = SparsityAllocation(total=100B, moe=73B, engram=27B)
    ASSERT validate_allocation(alloc_low).is_compliant == true
    ASSERT validate_allocation(alloc_high).is_compliant == true

TEST sparsity_05: zero total_sparse is non-compliant
    allocation = SparsityAllocation(total=0, moe=0, engram=0)
    ASSERT validate_allocation(allocation).is_compliant == false

TEST sparsity_06: compile_engram_allocation produces valid split
    (moe, engram) = compile_engram_allocation(100_000_000_000, config)
    ASSERT abs(engram as f64 / 100e9 - 0.25) < 0.001
    ASSERT moe + engram == 100_000_000_000

TEST sparsity_07: all reference model allocations are compliant
    for (name, alloc) in MODEL_ALLOCATIONS:
        result = validate_allocation(to_sparse_allocation(alloc))
        ASSERT result.is_compliant, f"{name} is non-compliant"

TEST sparsity_08: rho matches expected 75% for all scales
    for (name, alloc) in MODEL_ALLOCATIONS:
        result = validate_allocation(to_sparse_allocation(alloc))
        ASSERT abs(result.rho - 0.75) < 0.02, f"{name} rho={result.rho}"

TEST sparsity_09: deviation_pp is correct
    allocation = SparsityAllocation(total=100B, moe=74B, engram=26B)
    result = validate_allocation(allocation)
    ASSERT abs(result.deviation_pp - 1.0) < 0.01  # 26% - 25% = 1pp
```

## 9. Edge Cases

- **Non-integer parameter counts**: Rounding may cause moe + engram to differ from total
  by a few parameters. The assertion allows up to 1000 parameters of rounding error.
- **Dynamic expert pruning**: If MoE experts are pruned at runtime (unused experts), the
  ratio drifts toward MemoryBound. Runtime monitoring (5b) detects and corrects.
- **Engram table growth**: As the Engram compiler adds new entries, the table may grow
  beyond the 25% allocation. The eviction mechanism in spec 01 prevents unbounded growth.
- **Mixed precision**: If MoE uses fp16 and Engram uses fp32, the parameter *count* is the
  same but byte count differs. The 75/25 law operates on parameter count, not bytes.
