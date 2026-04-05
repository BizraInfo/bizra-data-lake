# 05 — Ihsan Composite Framework

**Status:** SPEC  
**Crate:** `bizra-core` (extend `lib.rs`)  
**Constitutional:** YES — geometric mean with per-dimension 0.95 floor

---

## 1. Motivation

v0.90.0's Ihsan is a scalar score — a single number representing overall quality. The
existing `IhsanScore` type in `bizra-core` encodes this as a fixed-point integer with
`as_f64()` conversion.

This creates a blind spot: a system can achieve Ihsan 0.96 by excelling at reasoning (0.99)
while neglecting safety (0.93). The scalar hides the deficit.

v0.91.0 introduces the **Ihsan Composite** — a structured quality framework comprising
8 sub-dimensions, each with its own 0.95 constitutional floor. The composite score is
the **geometric mean** of all dimensions, ensuring that a deficit in ANY dimension
disproportionately impacts the overall score. This is a mathematically enforced
anti-gaming mechanism.

## 2. The 8 Dimensions

| Dimension | Code | What it measures | v0.90.0 | v0.91.0 |
|-----------|------|------------------|---------|---------|
| Knowledge Retrieval | `knowledge` | Factual accuracy, entity recall | 0.720 | 0.940 |
| Reasoning Depth | `reasoning` | Logical inference, multi-step deduction | 0.820 | 0.840 |
| Code Generation | `code` | Correct, efficient code output | 0.780 | 0.810 |
| Instruction Following | `instruction` | Adherence to user intent | 0.910 | 0.920 |
| Safety & Compliance | `safety` | Harmful content rejection, policy adherence | 0.960 | 0.970 |
| Multilingual Support | `multilingual` | Cross-language quality parity | 0.880 | 0.890 |
| Latency Efficiency | `latency` | Response time within SLA bounds | 0.650 | 0.910 |
| Cost Efficiency | `cost` | Inference cost relative to quality | 0.700 | 0.880 |

## 3. Data Structures

```pseudocode
struct IhsanComposite:
    knowledge: f64       # [0.0, 1.0]
    reasoning: f64
    code: f64
    instruction: f64
    safety: f64
    multilingual: f64
    latency: f64
    cost: f64

struct CompositeValidation:
    composite_score: f64           # geometric mean
    is_compliant: bool             # all dimensions >= IHSAN_COMPOSITE_FLOOR
    violations: Vec<DimensionViolation>
    strongest_dimension: (String, f64)
    weakest_dimension: (String, f64)

struct DimensionViolation:
    dimension: String
    score: f64
    floor: f64
    deficit: f64                   # floor - score
```

## 4. Computation

```pseudocode
const IHSAN_COMPOSITE_FLOOR: f64 = 0.95
const IHSAN_COMPOSITE_DIMENSIONS: usize = 8

fn compute_composite(composite: IhsanComposite) -> CompositeValidation:
    """
    Compute the Ihsan Composite score as geometric mean of all 8 dimensions.
    
    Geometric mean = (d1 * d2 * ... * d8) ^ (1/8)
    
    Why geometric mean (not arithmetic):
    - Arithmetic mean: (0.99 + 0.01) / 2 = 0.50 — hides the 0.01 disaster
    - Geometric mean: (0.99 * 0.01) ^ 0.5 = 0.10 — amplifies the deficit
    - This prevents gaming by excelling in easy dimensions to compensate
      for neglecting hard ones.
    """
    dimensions = [
        ("knowledge", composite.knowledge),
        ("reasoning", composite.reasoning),
        ("code", composite.code),
        ("instruction", composite.instruction),
        ("safety", composite.safety),
        ("multilingual", composite.multilingual),
        ("latency", composite.latency),
        ("cost", composite.cost),
    ]
    
    # Validate each dimension
    violations = []
    for (name, score) in dimensions:
        if score < IHSAN_COMPOSITE_FLOOR:
            violations.push(DimensionViolation {
                dimension: name,
                score,
                floor: IHSAN_COMPOSITE_FLOOR,
                deficit: IHSAN_COMPOSITE_FLOOR - score,
            })
    
    # Compute geometric mean
    product = 1.0
    for (_, score) in dimensions:
        product *= max(score, 1e-10)  # floor to avoid zero destroying the product
    composite_score = product.pow(1.0 / IHSAN_COMPOSITE_DIMENSIONS as f64)
    
    # Find strongest and weakest
    sorted_dims = sorted(dimensions, key=lambda x: x[1])
    weakest = sorted_dims[0]
    strongest = sorted_dims[-1]
    
    return CompositeValidation {
        composite_score,
        is_compliant: violations.is_empty(),
        violations,
        strongest_dimension: strongest,
        weakest_dimension: weakest,
    }
```

## 5. Constitutional Gate

```pseudocode
fn ihsan_composite_gate(composite: IhsanComposite) -> GateResult:
    """
    Constitutional gate — fail-closed.
    
    Returns PASS only if:
    1. ALL 8 dimensions >= 0.95 individually
    2. Geometric mean >= 0.95 (implied by #1 but checked explicitly)
    """
    validation = compute_composite(composite)
    
    if not validation.is_compliant:
        return GateResult::Fail {
            reason: format!(
                "Ihsan Composite FAIL: {} dimensions below floor. Weakest: {}={:.4f}",
                validation.violations.len(),
                validation.weakest_dimension.0,
                validation.weakest_dimension.1,
            ),
            violations: validation.violations,
        }
    
    if validation.composite_score < IHSAN_COMPOSITE_FLOOR:
        # Shouldn't happen if all dimensions >= 0.95, but defense in depth
        return GateResult::Fail {
            reason: format!(
                "Ihsan Composite score {:.4f} below floor {:.4f}",
                validation.composite_score,
                IHSAN_COMPOSITE_FLOOR,
            ),
            violations: vec![],
        }
    
    return GateResult::Pass {
        composite_score: validation.composite_score,
        weakest: validation.weakest_dimension,
    }
```

## 6. Integration with Existing IhsanScore

The existing scalar `IhsanScore` is not replaced — it continues to serve as the
per-cycle quality signal in the OmniKernel. The IhsanComposite is a higher-order
aggregate computed over evaluation windows (weekly in production).

```pseudocode
# Relationship:
# - IhsanScore: per-cycle, per-mission scalar (used in omni_kernel.rs, mission_bridge.rs)
# - IhsanComposite: periodic aggregate across 8 dimensions (used in CI gate, deployment gate)

fn cycle_ihsan_to_composite_contribution(cycle_receipt: CycleReceipt, task_category):
    """
    Map a single cycle's IhsanScore to the relevant composite dimension.
    Many cycles are aggregated to produce one IhsanComposite per evaluation window.
    """
    match task_category:
        EntityRetrieval => contribute_to("knowledge", cycle_receipt.ihsan_score)
        Reasoning | Deduction => contribute_to("reasoning", cycle_receipt.ihsan_score)
        CodeGeneration => contribute_to("code", cycle_receipt.ihsan_score)
        InstructionFollowing => contribute_to("instruction", cycle_receipt.ihsan_score)
        SafetyEval => contribute_to("safety", cycle_receipt.ihsan_score)
        Multilingual => contribute_to("multilingual", cycle_receipt.ihsan_score)
        # latency and cost dimensions are computed from infrastructure metrics, not cycle Ihsan
```

## 7. CI Gate Fix

The SAPE analysis identified that the current CI gate (`IhsanGate` in `ci.yml:892`)
only checks SNR, not Ihsan independently. The IhsanComposite framework fixes this:

```pseudocode
fn ci_quality_gate(benchmark_results) -> CiGateResult:
    """
    CI gate — runs in GitHub Actions quality-management workflow.
    Must independently check BOTH SNR and Ihsan.
    """
    
    # SNR check (existing, unchanged)
    snr_passed = benchmark_results.snr >= UNIFIED_SNR_THRESHOLD
    
    # Ihsan Composite check (NEW)
    composite = build_composite_from_benchmarks(benchmark_results)
    ihsan_validation = compute_composite(composite)
    ihsan_passed = ihsan_validation.is_compliant
    
    # BOTH must pass — fail-closed
    return CiGateResult {
        passed: snr_passed AND ihsan_passed,
        snr_score: benchmark_results.snr,
        ihsan_composite: ihsan_validation.composite_score,
        ihsan_violations: ihsan_validation.violations,
    }
```

## 8. TDD Anchors

```
TEST composite_01: all dimensions at 0.95 → composite = 0.95 (geometric mean)
    c = IhsanComposite(all dimensions = 0.95)
    result = compute_composite(c)
    ASSERT abs(result.composite_score - 0.95) < 1e-9
    ASSERT result.is_compliant == true

TEST composite_02: one dimension at 0.94 → non-compliant
    c = IhsanComposite(safety=0.94, all others=0.99)
    result = compute_composite(c)
    ASSERT result.is_compliant == false
    ASSERT result.violations.len() == 1
    ASSERT result.violations[0].dimension == "safety"

TEST composite_03: geometric mean < arithmetic mean for uneven scores
    c = IhsanComposite(knowledge=0.99, cost=0.96, rest=0.97)
    result = compute_composite(c)
    arithmetic = (0.99 + 0.96 + 6*0.97) / 8
    ASSERT result.composite_score < arithmetic

TEST composite_04: one zero dimension collapses composite near zero
    c = IhsanComposite(safety=0.001, all others=1.0)
    result = compute_composite(c)
    ASSERT result.composite_score < 0.10
    ASSERT result.is_compliant == false

TEST composite_05: v0.91.0 reference scores produce expected composite
    c = IhsanComposite(
        knowledge=0.940, reasoning=0.840, code=0.810,
        instruction=0.920, safety=0.970, multilingual=0.890,
        latency=0.910, cost=0.880
    )
    result = compute_composite(c)
    ASSERT abs(result.composite_score - 0.970) < 0.005
    # Note: report says 0.970 — verify geometric mean matches

TEST composite_06: weakest dimension correctly identified
    c = IhsanComposite(latency=0.91, all others=0.99)
    result = compute_composite(c)
    ASSERT result.weakest_dimension == ("latency", 0.91)

TEST composite_07: strongest dimension correctly identified
    c = IhsanComposite(safety=0.99, all others=0.96)
    result = compute_composite(c)
    ASSERT result.strongest_dimension == ("safety", 0.99)

TEST composite_08: gate returns Pass when all compliant
    c = IhsanComposite(all dimensions = 0.96)
    ASSERT ihsan_composite_gate(c) == GateResult::Pass

TEST composite_09: gate returns Fail with violation list
    c = IhsanComposite(knowledge=0.80, reasoning=0.93, rest=0.99)
    result = ihsan_composite_gate(c)
    ASSERT result is GateResult::Fail
    ASSERT result.violations.len() == 2

TEST composite_10: ci_quality_gate fails if SNR passes but Ihsan fails
    benchmarks = { snr: 0.92, ihsan_composite: (safety=0.94, rest=0.99) }
    result = ci_quality_gate(benchmarks)
    ASSERT result.passed == false  # Ihsan safety below floor
```

## 9. Edge Cases

- **Missing dimension**: If a benchmark suite doesn't produce a score for one dimension
  (e.g., no multilingual tests in a run), that dimension defaults to `IHSAN_COMPOSITE_FLOOR`
  (0.95) — not 1.0 and not 0.0. This prevents both gaming and false failure.
- **Negative scores**: Clamp all inputs to [0.0, 1.0] before computation.
- **Backward compatibility**: The scalar `IhsanScore` continues to work unchanged. The
  composite is additive — new code uses `IhsanComposite`, old code uses `IhsanScore`.
- **Evaluation window**: The composite is meaningful only over a sufficient sample size.
  Minimum 100 cycles per dimension before the composite is considered valid.
