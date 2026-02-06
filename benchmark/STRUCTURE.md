# BIZRA Benchmark Suite - Code Structure

## Directory Layout

```
benchmark/
│
├── __init__.py                    [28 lines]
│   Package initialization and exports
│   - BenchmarkRunner
│   - BenchmarkResult
│   - MetricResult
│
├── __main__.py                    [199 lines]
│   CLI orchestration and scoring
│   - main() entry point
│   - ArgumentParser configuration
│   - compute_overall_score()
│   - print_summary_table()
│   - JSON export
│
├── runner.py                      [233 lines]
│   Core benchmarking framework
│   - MetricResult: Statistical aggregation
│   - BenchmarkResult: Benchmark encapsulation
│   - BenchmarkRunner: Main measurement class
│     ├── run(func, iterations): Latency measurement
│     ├── run_throughput(func, iterations): QPS measurement
│     ├── get_summary(): Dict export
│     └── print_summary(): Pretty print
│
├── README.md                      [~500 lines]
│   Complete user documentation
│   - Quick start
│   - Metrics reference
│   - Output format
│   - Architecture overview
│   - Design principles
│   - Integration examples
│   - References
│
├── STRUCTURE.md                   [This file]
│   Code organization and flow
│
└── suites/
    │
    ├── __init__.py                [14 lines]
    │   Suite package exports
    │
    ├── inference.py               [143 lines]
    │   Inference performance benchmarks
    │   - InferenceBenchmark class
    │     ├── benchmark_mock_inference_latency()
    │     ├── benchmark_batch_throughput()
    │     ├── benchmark_token_generation()
    │     ├── benchmark_model_selection_overhead()
    │     ├── benchmark_context_window_scaling()
    │     └── run_all()
    │
    ├── security.py                [184 lines]
    │   Cryptographic operation benchmarks
    │   - SecurityBenchmark class
    │     ├── benchmark_sha256_hashing()
    │     ├── benchmark_hmac_verification()
    │     ├── benchmark_timing_safe_comparison()
    │     ├── benchmark_nonce_generation()
    │     ├── benchmark_envelope_serialization()
    │     ├── benchmark_key_derivation()
    │     └── run_all()
    │
    └── quality.py                 [231 lines]
        Quality metric benchmarks
        - QualityBenchmark class
          ├── benchmark_snr_calculation()
          ├── benchmark_ihsan_scoring()
          ├── benchmark_type_validation()
          ├── benchmark_compliance_check()
          ├── benchmark_gate_evaluation()
          └── run_all()
```

## Code Flow

### Command Line Execution

```
python3 -m benchmark [suite] [--iterations N] [--json file] [--verbose]
           ↓
__main__.py::main()
    ├── Parse arguments (argparse)
    ├── Create BenchmarkRunner(warmup=2, verbose=args.verbose)
    ├── Conditionally instantiate suite classes:
    │   ├── InferenceBenchmark(runner)
    │   ├── SecurityBenchmark(runner)
    │   └── QualityBenchmark(runner)
    ├── Run selected suite(s)
    │   └── suite.run_all(iterations)
    │       └── suite.benchmark_*()
    │           └── runner.run() or runner.run_throughput()
    ├── Compute overall_score()
    ├── print_summary_table(all_results, overall_score)
    ├── Export JSON (if requested)
    └── Exit
```

### Benchmark Execution

```
runner.run(name, func, iterations, track_memory=False)
    │
    ├── Warmup iterations (default: 2)
    │   └── func() - exceptions caught, logged, continue
    │
    ├── Benchmark iterations
    │   └── for i in range(iterations):
    │       ├── Start timer: time.perf_counter()
    │       ├── [Optional] tracemalloc.start()
    │       ├── func() - execute benchmark
    │       ├── Measure elapsed_ms
    │       ├── [Optional] Get memory usage
    │       ├── result.add_metric(name, unit, value)
    │       └── [Exception] Log, continue
    │
    └── Return BenchmarkResult
        └── Metrics aggregated: min/avg/max/p95
```

### Metric Aggregation

```
MetricResult
    ├── __init__(name, unit)
    ├── add_value(val) - append to values list
    │
    └── Properties (computed on demand):
        ├── min_val: min(values)
        ├── avg_val: statistics.mean(values)
        ├── max_val: max(values)
        └── p95_val: sorted(values)[idx @ 95%]
        
        └── to_dict(): export for JSON
```

### Scoring

```
compute_overall_score(all_results)
    ├── Extract latencies from all benchmarks
    ├── For each suite:
    │   ├── Calculate avg latency
    │   ├── Normalize: (baseline / actual) * 100
    │   ├── Cap at 100.0
    │   └── Store suite_score
    │
    ├── Weighted average:
    │   overall = (
    │       0.5 * inference_score +
    │       0.2 * security_score +
    │       0.3 * quality_score
    │   )
    │
    └── Return min(100.0, overall)
```

## Class Hierarchy

### BenchmarkRunner

```python
class BenchmarkRunner:
    def __init__(warmup: int = 2, verbose: bool = False)
    def run(name, func, iterations, track_memory) -> BenchmarkResult
    def run_throughput(name, func, iterations) -> BenchmarkResult
    def get_summary() -> Dict[str, Any]
    def print_summary()
```

### BenchmarkResult

```python
@dataclass
class BenchmarkResult:
    name: str
    iterations: int
    metrics: Dict[str, MetricResult] = {}
    
    def add_metric(name, unit, value)
    def to_dict() -> Dict[str, Any]
```

### MetricResult

```python
@dataclass
class MetricResult:
    name: str
    unit: str
    values: List[float] = []
    
    @property min_val -> float
    @property avg_val -> float
    @property max_val -> float
    @property p95_val -> float
    def to_dict() -> Dict[str, Any]
```

### Suite Classes

```python
class InferenceBenchmark:
    def __init__(runner: BenchmarkRunner)
    def benchmark_*() -> Dict[str, Any]
    def run_all(iterations) -> Dict[str, Any]

class SecurityBenchmark:
    [Same structure]

class QualityBenchmark:
    [Same structure]
```

## Data Flow - Complete Example

### Input
```bash
python3 -m benchmark inference --iterations 3 --json /tmp/out.json
```

### Processing

1. **Parse Arguments**
   ```python
   suite = "inference"
   iterations = 3
   json_output = "/tmp/out.json"
   verbose = False
   ```

2. **Create Runner**
   ```python
   runner = BenchmarkRunner(warmup=2, verbose=False)
   ```

3. **Run Inference Suite**
   ```python
   inference = InferenceBenchmark(runner)
   results = inference.run_all(iterations=3)
   ```

4. **Run Each Benchmark**
   ```python
   # Example: mock_latency
   def inference_op():
       time.sleep(0.050)  # 50ms
   
   result = runner.run(
       "inference.mock_latency",
       inference_op,
       iterations=3,
       track_memory=False
   )
   
   # Iteration 1: 50.12 ms
   # Iteration 2: 50.08 ms
   # Iteration 3: 50.15 ms
   
   # Aggregated:
   # min: 50.08, avg: 50.117, max: 50.15, p95: 50.15
   ```

5. **Compute Score**
   ```python
   inference_score = (50.0 / 50.117) * 100 = 99.8
   overall_score = 0.75 * 99.8 = 74.9 / 100
   ```

6. **Export**
   ```json
   {
     "overall_score": 74.9,
     "iterations": 3,
     "suites": {
       "inference": {
         "latency": {
           "metrics": {
             "latency_ms": {
               "min": 50.08,
               "avg": 50.117,
               "max": 50.15,
               "p95": 50.15,
               "count": 3
             }
           }
         }
       }
     }
   }
   ```

### Output
- Console: Pretty table
- JSON: Exported to `/tmp/out.json`

## Key Design Patterns

### 1. Template Method Pattern
```python
class BenchmarkSuite:
    def run_all():
        for benchmark_name in self.benchmarks:
            result = self.benchmark_name()
            self.results[benchmark_name] = result
        return self.results
```

### 2. Strategy Pattern
```python
runner.run(
    name,
    func,  # Strategy: the operation to benchmark
    iterations
)
```

### 3. Data Transfer Object
```python
@dataclass
class MetricResult:
    # Encapsulates all data for one metric
```

### 4. Facade Pattern
```python
main()  # Hides all complexity behind simple CLI
```

## Error Handling

### Iteration-Level
```python
for i in range(iterations):
    try:
        func()
        # Record metric
    except Exception as e:
        if verbose:
            print(f"Iteration {i} failed: {e}")
        # Continue to next iteration
```

### Suite-Level
```python
if suite in ("all", "inference"):
    try:
        inference = InferenceBenchmark(runner)
        results["inference"] = inference.run_all()
    except Exception as e:
        logger.error(f"Inference suite failed: {e}")
        # Continue to next suite
```

## Performance Characteristics

### Time Complexity
- Per benchmark: O(N) where N = iterations
- Per suite: O(M * N) where M = number of benchmarks
- Overall: O(S * M * N) where S = number of suites

### Space Complexity
- Per metric: O(N) to store all measurements
- Per benchmark: O(K * N) where K = number of metrics
- Overall: O(S * M * K * N)

### Typical Execution
- N=20, M=5 benchmarks, S=3 suites
- ~60 seconds total
- Constant memory (~10MB)

## Testing Considerations

### Unit Test Targets
- `MetricResult`: min/avg/max/p95 calculations
- `BenchmarkRunner`: timing accuracy
- `compute_overall_score()`: scoring algorithm
- Suite classes: benchmark correctness

### Integration Test Targets
- Full CLI execution
- JSON export validation
- Error recovery
- Statistical stability

### Performance Test Targets
- Benchmark execution time
- Memory footprint
- Statistical quality (low variance)

## Future Architecture

### Extensibility Points

1. **New Suite**
   ```python
   class StorageBenchmark:
       def __init__(runner)
       def benchmark_io_latency()
       def run_all()
   
   # Register in __main__.py
   if suite in ("all", "storage"):
       storage = StorageBenchmark(runner)
   ```

2. **New Metric Type**
   ```python
   class EnergyMetric(MetricResult):
       unit = "watts"
       
       @property
       def avg_consumption():
           return self.avg_val
   ```

3. **Custom Scoring**
   ```python
   def custom_score_fn(results):
       # Custom weighting
       return score
   ```

## References

- `/mnt/c/BIZRA-DATA-LAKE/benchmark/runner.py` - Implementation details
- `/mnt/c/BIZRA-DATA-LAKE/benchmark/README.md` - User guide
- `/mnt/c/BIZRA-DATA-LAKE/BENCHMARK_IMPLEMENTATION.md` - Design decisions
