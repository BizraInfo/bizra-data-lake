# 07 — Benchmark Validation Suite

**Status:** SPEC  
**Crate:** `bizra-tests` (new `benchmark/` directory)  
**Depends on:** 05_ihsan_composite, 04_sparsity_75_25

---

## 1. Motivation

The v0.91.0 evaluation claims specific quantitative improvements across 7 standard
benchmarks and 8 Ihsan dimensions. These claims must be reproducible through an
automated benchmark suite with deterministic seeding, statistical validation, and
component ablation capability.

## 2. Benchmark Inventory

| Benchmark | Domain | v0.90.0 | v0.91.0 | Delta | Primary Attribution |
|-----------|--------|---------|---------|-------|---------------------|
| NIAH Short | Knowledge | 85.4% | 96.8% | +11.4pp | Direct Engram injection |
| NIAH Long | Knowledge | 68.2% | 90.1% | +21.9pp | O(1) vs attention reconstruction |
| MMLU 5-shot | General | 75.1% | 78.4% | +3.3pp | Freed attention for reasoning |
| GSM8K | Math | 66.8% | 71.2% | +4.4pp | Formula recall from Engram |
| HumanEval | Code | 60.3% | 64.1% | +3.8pp | API pattern retrieval |
| ARC-C | Reasoning | 70.2% | 73.9% | +3.7pp | Reserved attention capacity |
| HellaSwag | Commonsense | 81.5% | 83.2% | +1.7pp | World knowledge injection |

## 3. Data Structures

```pseudocode
struct BenchmarkConfig:
    name: String
    domain: BenchmarkDomain
    dataset_path: String
    num_samples: usize          # minimum samples for statistical significance
    random_seed: u64            # deterministic seeding
    timeout_per_sample_ms: u64
    model_scale: ModelScale     # 7B, 13B, 27B, 70B, 140B, 300B

enum BenchmarkDomain:
    Knowledge
    General
    Math
    Code
    Reasoning
    Commonsense

struct BenchmarkResult:
    name: String
    accuracy: f64               # primary metric (0.0–1.0)
    latency_p50_ms: f64
    latency_p99_ms: f64
    throughput_tokens_per_sec: f64
    samples_evaluated: usize
    samples_correct: usize
    confidence_interval_95: (f64, f64)  # Wilson score interval
    statistical_significance: f64       # p-value vs baseline

struct BenchmarkSuite:
    benchmarks: Vec<BenchmarkConfig>
    baseline_results: HashMap<String, BenchmarkResult>  # v0.90.0 reference
    current_results: HashMap<String, BenchmarkResult>
    ablation_results: HashMap<String, HashMap<String, BenchmarkResult>>
```

## 4. Execution Flow

```pseudocode
fn run_benchmark_suite(suite, model, config) -> SuiteReport:
    """
    Execute all benchmarks in the suite.
    Deterministic: same seed → same results (within floating-point tolerance).
    """
    results = {}
    
    for bench in suite.benchmarks:
        # Set deterministic seed
        set_random_seed(bench.random_seed)
        
        # Load dataset
        dataset = load_benchmark_dataset(bench.dataset_path, bench.num_samples)
        
        # Run inference
        correct = 0
        latencies = []
        for sample in dataset:
            start = now_ms()
            output = model.infer(sample.input, timeout=bench.timeout_per_sample_ms)
            elapsed = now_ms() - start
            latencies.push(elapsed)
            
            if evaluate_correctness(output, sample.expected, bench.domain):
                correct += 1
        
        accuracy = correct as f64 / dataset.len() as f64
        
        results[bench.name] = BenchmarkResult {
            name: bench.name,
            accuracy,
            latency_p50_ms: percentile(latencies, 0.50),
            latency_p99_ms: percentile(latencies, 0.99),
            throughput_tokens_per_sec: compute_throughput(dataset, latencies),
            samples_evaluated: dataset.len(),
            samples_correct: correct,
            confidence_interval_95: wilson_score_interval(correct, dataset.len()),
            statistical_significance: compute_p_value(
                accuracy, suite.baseline_results[bench.name].accuracy, dataset.len()
            ),
        }
    
    return SuiteReport {
        results,
        all_significant: all(r.statistical_significance < 0.01 for r in results.values()),
        ihsan_composite: build_composite_from_benchmarks(results),
    }
```

## 5. Component Ablation

```pseudocode
struct AblationConfig:
    component_to_disable: String  # "engram", "prefetch", "context_gate", "75_25"
    description: String

fn run_ablation(suite, model, ablation) -> AblationReport:
    """
    Disable a single component and re-run benchmarks to measure its contribution.
    
    From evaluation Table 7:
    - Disabling Engram: entity retrieval latency reverts to 312ms (from 48ms)
    - Disabling prefetch: L1 hit rate drops, throughput penalty increases
    - Disabling context gate: all requests treated equally (v0.90.0 behavior)
    - Disabling 75/25: allocation unconstrained (may drift to suboptimal)
    """
    
    # Disable the target component
    modified_model = model.clone()
    match ablation.component_to_disable:
        "engram" => modified_model.engram_cache.disable()
        "prefetch" => modified_model.prefetch_pipeline.disable()
        "context_gate" => modified_model.context_gate.set_constant(0.5)  # neutral
        "75_25" => modified_model.sparsity_constraint.disable()
    
    # Run full suite on modified model
    ablated_results = run_benchmark_suite(suite, modified_model, config)
    
    # Compute impact
    impacts = {}
    for (name, baseline) in suite.current_results:
        ablated = ablated_results.results[name]
        impacts[name] = AblationImpact {
            benchmark: name,
            baseline_accuracy: baseline.accuracy,
            ablated_accuracy: ablated.accuracy,
            delta_pp: (baseline.accuracy - ablated.accuracy) * 100.0,
            attribution: ablation.component_to_disable,
        }
    
    return AblationReport {
        component: ablation.component_to_disable,
        impacts,
        total_accuracy_loss_pp: sum(i.delta_pp for i in impacts.values()),
    }
```

## 6. Statistical Validation

```pseudocode
fn compute_p_value(accuracy_a, accuracy_b, n_samples) -> f64:
    """Two-proportion z-test for comparing benchmark accuracies."""
    p1 = accuracy_a
    p2 = accuracy_b
    p_pooled = (p1 * n_samples + p2 * n_samples) / (2 * n_samples)
    se = sqrt(p_pooled * (1 - p_pooled) * (2.0 / n_samples))
    if se < 1e-10:
        return 1.0  # no difference
    z = abs(p1 - p2) / se
    return 2.0 * (1.0 - normal_cdf(z))  # two-tailed

fn wilson_score_interval(successes, total) -> (f64, f64):
    """95% Wilson score confidence interval for proportions."""
    z = 1.96  # 95% CI
    p = successes as f64 / total as f64
    denominator = 1.0 + z * z / total as f64
    center = (p + z * z / (2.0 * total as f64)) / denominator
    margin = z * sqrt(p * (1.0 - p) / total as f64 + z * z / (4.0 * total as f64 * total as f64)) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))
```

## 7. Regression Detection

```pseudocode
fn check_regression(current, baseline, tolerance_pp) -> Vec<Regression>:
    """
    Detect regressions: any benchmark where current < baseline - tolerance.
    
    tolerance_pp: acceptable regression in percentage points (default: 0.5pp)
    """
    regressions = []
    for (name, current_result) in current:
        if baseline_result = baseline.get(name):
            delta = current_result.accuracy - baseline_result.accuracy
            if delta < -tolerance_pp / 100.0:
                regressions.push(Regression {
                    benchmark: name,
                    baseline: baseline_result.accuracy,
                    current: current_result.accuracy,
                    delta_pp: delta * 100.0,
                    is_significant: current_result.statistical_significance < 0.05,
                })
    return regressions
```

## 8. TDD Anchors

```
TEST bench_01: deterministic seeding produces identical results
    result_a = run_benchmark("NIAH_short", seed=42)
    result_b = run_benchmark("NIAH_short", seed=42)
    ASSERT result_a.accuracy == result_b.accuracy

TEST bench_02: confidence interval contains true proportion
    # Monte Carlo: run 100 times, check CI coverage
    covered = 0
    for i in 0..100:
        (low, high) = wilson_score_interval(75, 100)  # 75%
        if low <= 0.75 <= high:
            covered += 1
    ASSERT covered >= 90  # 95% CI should cover ~95% of the time

TEST bench_03: p-value < 0.01 for large differences
    p = compute_p_value(0.90, 0.68, 1000)
    ASSERT p < 0.01  # 22pp difference on 1000 samples is highly significant

TEST bench_04: p-value > 0.05 for tiny differences
    p = compute_p_value(0.75, 0.74, 100)
    ASSERT p > 0.05  # 1pp difference on 100 samples is not significant

TEST bench_05: ablation disabling Engram increases entity retrieval latency
    baseline = run_benchmark("entity_retrieval", model=v091)
    ablated = run_ablation("engram", model=v091)
    ASSERT ablated.latency_p50_ms > baseline.latency_p50_ms * 5  # ~85% regression

TEST bench_06: regression detection catches 1pp drop
    baseline = { "NIAH_short": 0.968 }
    current = { "NIAH_short": 0.955 }
    regressions = check_regression(current, baseline, tolerance_pp=0.5)
    ASSERT regressions.len() == 1
    ASSERT regressions[0].delta_pp ≈ -1.3

TEST bench_07: no regression when current > baseline
    baseline = { "MMLU": 0.751 }
    current = { "MMLU": 0.784 }
    ASSERT check_regression(current, baseline, 0.5).is_empty()

TEST bench_08: suite report builds valid Ihsan Composite
    results = run_benchmark_suite(all_7_benchmarks)
    composite = results.ihsan_composite
    ASSERT composite.knowledge >= 0.0  # populated from NIAH results
    ASSERT composite.reasoning >= 0.0  # populated from ARC-C results

TEST bench_09: all v0.91.0 reference deltas are statistically significant
    for (name, v090, v091) in REFERENCE_RESULTS:
        p = compute_p_value(v091, v090, 1000)
        ASSERT p < 0.01, f"{name}: improvement not significant (p={p})"

TEST bench_10: timeout kills hung inference without corrupting results
    SET timeout_per_sample_ms = 100
    RUN benchmark with model that sleeps 200ms
    ASSERT result.samples_evaluated == expected  # timed out samples counted as incorrect
```

## 9. Edge Cases

- **Non-deterministic GPU operations**: Some CUDA operations (atomics, reductions) are
  non-deterministic. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` and
  `torch.backends.cudnn.deterministic = True` to maximize reproducibility. Accept
  ±0.1pp variance.
- **Dataset version drift**: Pin benchmark datasets to specific commits/versions. Use
  content-addressed hashes to verify dataset integrity before running.
- **OOM on large models**: 300B benchmarks may require model parallelism. The suite must
  handle multi-GPU configurations transparently.
- **Flaky benchmarks**: If a benchmark result varies by >2pp across runs (same seed),
  flag it as unstable and require 5 runs for averaging.
