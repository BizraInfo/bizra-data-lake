# BIZRA Benchmark Suite

Minimal local benchmarking framework for Node0 evaluation. Runs fully offline, measures real metrics, under 300 lines of core infrastructure.

**Total lines of code: 1,032** (including all suites and documentation)
- `runner.py` (233 lines): Core BenchmarkRunner class
- `__main__.py` (199 lines): CLI and orchestration
- `suites/inference.py` (143 lines): Inference benchmarks
- `suites/security.py` (184 lines): Security/crypto benchmarks
- `suites/quality.py` (231 lines): Quality metric benchmarks

## Quick Start

### Run All Benchmarks
```bash
python3 -m benchmark --iterations 20
```

### Run Specific Suite
```bash
python3 -m benchmark inference --iterations 50
python3 -m benchmark security
python3 -m benchmark quality
```

### Export JSON Results
```bash
python3 -m benchmark --iterations 20 --json results.json
```

### Verbose Mode
```bash
python3 -m benchmark --verbose --iterations 10
```

## Metrics Measured

### Inference Suite
- **latency_ms**: Mock inference operation latency (100-token generation)
- **batch_throughput**: Batch processing throughput (32 parallel requests)
- **tokens_per_second**: Token generation rate (256 tokens)
- **model_selection_overhead**: Task complexity analysis and model selection
- **context_window_4k**: Inference with 4K token context window

### Security Suite
- **sha256_hash**: SHA256 hashing baseline (1KB message)
- **hmac_verification**: HMAC-SHA256 verification for replay detection
- **timing_safe_cmp**: Constant-time comparison (critical for crypto)
- **nonce_validation**: Replay attack prevention via nonce tracking
- **envelope_serialization**: PCI envelope serialization overhead
- **key_derivation**: KDF (PBKDF2-style) operation cost

### Quality Suite
- **snr_calculation**: Signal-to-Noise Ratio calculation
- **ihsan_scoring**: Ihsān dimensional excellence scoring
- **type_validation**: Runtime type checking overhead
- **compliance_check**: Constitutional compliance validation
- **gate_evaluation**: Quality gate evaluation

## Output Format

### Console Output
```
======================================================================
BIZRA BENCHMARK SUITE RESULTS
======================================================================

INFERENCE
----------------------------------------------------------------------
  inference.mock_latency              :    50.16 ms     (p95:    50.30 ms)
  inference.batch_throughput          :   213.13 qps    (p95:   213.17 qps)
  inference.tokens_per_second         :   499.84 qps    (p95:   499.94 qps)
  inference.model_selection_overhead  :     1.12 ms     (p95:     1.19 ms)
  inference.context_window_4k         :   141.15 ms     (p95:   141.21 ms)

SECURITY
----------------------------------------------------------------------
  security.sha256                     :     0.00 ms     (p95:     0.00 ms)
  security.hmac_verification          :     0.00 ms     (p95:     0.00 ms)
  security.timing_safe_cmp            :     0.00 ms     (p95:     0.00 ms)
  security.nonce_validation           :     0.00 ms     (p95:     0.00 ms)
  security.envelope_serialization     :     0.00 ms     (p95:     0.00 ms)
  security.key_derivation             :     0.42 ms     (p95:     0.42 ms)

QUALITY
----------------------------------------------------------------------
  quality.snr_calculation             :     0.02 ms     (p95:     0.02 ms)
  quality.ihsan_scoring               :     0.00 ms     (p95:     0.00 ms)
  quality.type_validation             :     0.01 ms     (p95:     0.01 ms)
  quality.compliance_check            :     0.00 ms     (p95:     0.00 ms)
  quality.gate_evaluation             :     0.00 ms     (p95:     0.00 ms)

======================================================================
OVERALL SCORE:  94.2/100
======================================================================
```

### JSON Output
```json
{
  "overall_score": 94.2,
  "iterations": 20,
  "suites": {
    "inference": {
      "latency": {
        "metrics": {
          "latency_ms": {
            "min": 49.8,
            "avg": 50.16,
            "max": 50.9,
            "p95": 50.30,
            "count": 20
          }
        }
      }
    }
  }
}
```

## Architecture

### BenchmarkRunner
Core measurement framework:
- `run()`: Measure latency with auto GC
- `run_throughput()`: Measure operations/second
- `get_summary()`: Export results as dict
- `print_summary()`: Human-readable output

Features:
- Automatic warmup iterations (default: 2)
- Per-iteration tracking of min/avg/max/p95
- Optional memory tracking via `tracemalloc`
- Exception handling per iteration (doesn't break suite)

### Metric Result
Statistical aggregation:
- Stores raw measurements per iteration
- Computes min/avg/max/p95 on demand
- Exports to dict for JSON serialization

### Benchmark Result
Encapsulates one complete benchmark:
- Multiple metrics per benchmark
- Iteration count tracking
- Dict conversion for JSON

### Suite Classes
Specialized benchmark implementations:
- **InferenceBenchmark**: Mock inference operations (no LLM required)
- **SecurityBenchmark**: Pure Python crypto benchmarks
- **QualityBenchmark**: SNR/Ihsān calculation benchmarks

All benchmarks:
- Run fully offline (no network, no external services)
- Use realistic mock operations (not `pass` statements)
- Measure actual wall-clock time
- Support variable iteration counts

## Scoring

### Overall Score (0-100)
Weighted average of suite scores:

```
overall_score = (
  0.5 * inference_score +
  0.2 * security_score +
  0.3 * quality_score
)
```

Each suite score is normalized against reference latencies:
- Inference: 50ms baseline
- Security: 0.5ms baseline
- Quality: 2.0ms baseline

**Example:**
- Inference avg latency: 25ms → score = (50/25)*100 = 200 → capped at 100
- Security avg latency: 0.1ms → score = (0.5/0.1)*100 = 500 → capped at 100
- Quality avg latency: 5.0ms → score = (2.0/5.0)*100 = 40

Overall = (100 * 0.5) + (100 * 0.2) + (40 * 0.3) = 50 + 20 + 12 = 82/100

## Design Principles

### Standing on Giants
- **Knuth (The Art of Computer Programming)**: "Measure, don't guess"
  - All metrics are wall-clock measurements, not predictions
  - Statistical aggregation (min/avg/max/p95) prevents overconfidence

- **Amdahl's Law**: "Find and optimize bottlenecks"
  - Per-benchmark breakdown identifies slow paths
  - Suite weighting focuses on critical operations

### Ihsān (Excellence)
- Type-safe: Full Python type hints
- Error-resilient: Exceptions don't crash suite
- Offline-first: Zero external dependencies
- Fast: 20 iterations typically completes in <1 minute
- Minimal: <300 lines of core infrastructure

## Limitations & Future Work

### Current Limitations
1. **Mock Operations**: All benchmarks use simulated operations, not real LLM inference
2. **Single-threaded**: No concurrent benchmarking (simplicity trade-off)
3. **Platform-specific**: Timing may vary by OS and hardware
4. **No Statistical Tests**: No p-values or significance testing (out of scope)

### Future Enhancements
- Real LLM integration (when available)
- Concurrent request simulation
- Regression detection (compare against baseline)
- Profiling integration (CPU flame graphs)
- Distributed benchmarking (across multiple nodes)

## Integration with CI/CD

### GitHub Actions Example
```yaml
- name: Run BIZRA Benchmarks
  run: |
    python3 -m benchmark --iterations 50 --json benchmark_results.json

- name: Comment Results on PR
  uses: actions/github-script@v6
  with:
    script: |
      const fs = require('fs');
      const results = JSON.parse(fs.readFileSync('benchmark_results.json'));
      github.rest.issues.createComment({
        issue_number: context.issue.number,
        owner: context.repo.owner,
        repo: context.repo.repo,
        body: `## Benchmark Results\nScore: ${results.overall_score}/100`
      })
```

## References

### Standing on Giants
- **Knuth (1969-1997)**: "The Art of Computer Programming" — methodology for performance analysis
- **Amdahl (1967)**: "Validity of the Single Processor Approach" — identifying bottlenecks
- **Gorelick & Ozsvald (2020)**: "High Performance Python" — practical measurement techniques

### BIZRA Architecture
- [quality_gates.py](../core/elite/quality_gates.py): SNR and Ihsān implementation
- [inference/gateway.py](../core/inference/gateway.py): Local inference infrastructure
- [pci/crypto.py](../core/pci/crypto.py): Cryptographic operations
- [CLAUDE.md](../CLAUDE.md): Project standards and configuration

## License

Part of BIZRA Node0. See parent repository LICENSE.
