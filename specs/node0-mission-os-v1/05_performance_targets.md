# Performance Targets — v1.0.0 Release Gate

**Status:** [ENFORCEMENT: WIRED]
**Reference:** BIZRA_CANONICAL.md (frozen 2026-03-26)

## Stage-Level Requirements

### End-to-End Canonical Loop

| Metric | P50 | P95 | P99 | Gate |
|--------|-----|-----|-----|------|
| Full loop latency | ≤ 120ms | ≤ 250ms | ≤ 400ms | HARD |
| Gate + proof | ≤ 60ms | ≤ 140ms | ≤ 140ms | HARD |
| Receipt emission | ≤ 10ms | ≤ 20ms | ≤ 50ms | HARD |
| Evidence bundle gen | — | — | ≤ 5s | SOFT |
| False admission rate | — | — | 0 | HARD |

### Canonical Benchmarks (from BIZRA_CANONICAL.md)

| Operation | Canonical | Std Dev | P99 |
|-----------|-----------|---------|-----|
| IHSAN check | 90.4 ns | ±12.3 ns | 145 ns |
| BLAKE3 hash (4KB) | 349 ns | ±28.7 ns | 420 ns |
| Ed25519 sign | 396 ns | ±35.2 ns | 480 ns |
| Total membrane | 3.02 us | ±0.89 us | 5.8 us |
| Throughput | 237,199 req/s | — | — |

### Reliability

| Metric | Target | Gate |
|--------|--------|------|
| CI success rate | ≥ 99.5% | SOFT |
| Recovery after crash | ≤ 60s | HARD |
| Missing receipt rate | 0 | HARD |
| Replay parity | 100% | HARD |
| Heartbeat pass rate | 100% (288/288) | HARD |

## Benchmark CI Integration

```
# In canonical-validation-gate.yml PERF job:
FUNCTION run_performance_gate():
    results = run_canonical_benchmarks()

    # Hard gates — fail CI if violated
    ASSERT results.false_admission_rate == 0
    ASSERT results.receipt_p95_ms <= 20
    ASSERT results.gate_proof_p95_ms <= 140

    # Soft gates — warn but don't fail
    IF results.e2e_p50_ms > 120:
        WARN "E2E P50 above target: {results.e2e_p50_ms}ms"
    IF results.ci_success_rate < 0.995:
        WARN "CI success rate below target: {results.ci_success_rate}"

    # Regression detection — compare with previous run
    previous = load_previous_benchmarks()
    IF previous:
        FOR metric IN results:
            regression = (metric.value - previous[metric.name]) / previous[metric.name]
            IF regression > 0.10:  # 10% regression threshold
                FAIL f"Performance regression: {metric.name} degraded by {regression:.1%}"

    save_benchmarks(results)
    RETURN results
```

## Characterization Discipline

All performance claims MUST be characterized as:

> "Validated in reported EV&V environment"

NOT:

> ~~"Universally proven in all real-world conditions"~~

The environment must be documented in the evidence bundle:
- Hardware: CPU model, RAM, GPU
- OS: Ubuntu version, kernel
- Runtime: Rust version, Python version
- Workload: Number of concurrent missions, payload sizes
- Duration: Benchmark runtime, warm-up period
