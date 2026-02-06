# BIZRA Benchmark Suite

## Overview

This directory contains the comprehensive benchmark suite for BIZRA performance validation.
Targets: 8 billion nodes, 100ns/observation, O(n log n) complexity.

## Directory Structure

```
benchmark_suite/
├── README.md               # This file
├── benches/                # Rust criterion benchmarks
│   ├── ntu_bench.rs        # NTU microbenchmarks
│   ├── fate_bench.rs       # FATE Gate benchmarks
│   ├── gossip_bench.rs     # Federation gossip benchmarks
│   └── snr_bench.rs        # SNR calculation benchmarks
├── e2e/                    # End-to-end Python benchmarks
│   ├── test_planetary_scale.py
│   ├── test_memory_pressure.py
│   └── test_network_simulation.py
├── profiling/              # Profiling utilities
│   ├── python_profile.py
│   └── flame_graph.sh
├── results/                # Benchmark results (gitignored)
└── Cargo.toml              # Rust workspace config
```

## Quick Start

### Python Benchmarks

```bash
cd /mnt/c/BIZRA-DATA-LAKE
source .venv/Scripts/activate  # Windows
# source .venv/bin/activate    # Linux

# Run all benchmarks
python -m pytest benchmark_suite/e2e/ -v --benchmark-only

# Run specific benchmark
python -m pytest benchmark_suite/e2e/test_planetary_scale.py -v -k "ntu"
```

### Rust Benchmarks (after Rust implementation)

```bash
cd /mnt/c/BIZRA-DATA-LAKE/benchmark_suite

# Run all benchmarks
cargo bench

# Run specific benchmark with baseline comparison
cargo bench --bench ntu_bench -- --save-baseline main
cargo bench --bench ntu_bench -- --baseline main

# Generate HTML report
cargo bench -- --verbose --plotting-backend plotters
```

## Target Metrics

| Component | Current (Python) | Target (Rust) | Improvement |
|-----------|------------------|---------------|-------------|
| NTU observe | 8ms | 100ns | 80,000x |
| FATE validate | 1-5ms | <10us | 100-500x |
| Gossip message | 1-5ms | <1ms | 5x |
| SNR batch (1K) | 50s | <100ms | 500x |

## Benchmark Categories

### 1. Microbenchmarks (Criterion)

Single-operation latency measurements:

- `ntu_observe`: Single observation processing
- `ntu_batch`: Batch observation processing
- `fate_validate`: Single FATE gate validation
- `snr_compute`: Single SNR calculation
- `gossip_handle`: Single message handling

### 2. Throughput Benchmarks

Operations per second measurements:

- `ntu_throughput`: Observations/second
- `fate_throughput`: Validations/second
- `gossip_throughput`: Messages/second

### 3. Memory Benchmarks

Memory usage measurements:

- `ntu_memory`: Memory per NTU instance
- `peer_table_memory`: Memory per peer entry
- `window_memory`: Memory for sliding windows

### 4. Scale Benchmarks

Planetary scale simulations:

- `million_nodes`: 1M node gossip simulation
- `billion_observations`: 1B observation processing
- `consensus_quorum`: 10K voter consensus

## Running Benchmarks

### CI Pipeline

Benchmarks run automatically on PR with 5% regression threshold:

```yaml
# .github/workflows/benchmark.yml
- name: Run Benchmarks
  run: cargo bench --bench ntu_bench -- --baseline main --threshold 0.05
```

### Local Development

```bash
# Quick benchmark (subset)
cargo bench -- --sample-size 10

# Full benchmark (production)
cargo bench -- --sample-size 100 --measurement-time 10

# Profile-guided benchmark
RUSTFLAGS="-C target-cpu=native" cargo bench
```

## Interpreting Results

### Criterion Output

```
ntu_observe/single    time:   [95.2 ns 98.1 ns 101.3 ns]
                      change: [-2.1% +0.3% +2.8%] (p = 0.78 > 0.05)
                      No change in performance detected.
```

- **time**: [lower bound, estimate, upper bound] with 95% confidence
- **change**: Comparison to baseline
- **p-value**: Statistical significance

### Performance Thresholds

| Metric | Green | Yellow | Red |
|--------|-------|--------|-----|
| NTU observe | <100ns | <1us | >1us |
| FATE validate | <10us | <100us | >100us |
| Gossip handle | <1ms | <5ms | >5ms |

## Adding New Benchmarks

### Criterion Benchmark Template

```rust
// benches/example_bench.rs
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn example_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("example");

    for size in [10, 100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, &size| {
                // Setup
                let data = setup_data(size);

                // Benchmark
                b.iter(|| {
                    process_data(&data)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, example_benchmark);
criterion_main!(benches);
```

### Python Benchmark Template

```python
# e2e/test_example.py
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

def test_example_benchmark(benchmark: BenchmarkFixture):
    # Setup
    data = setup_data()

    # Benchmark
    result = benchmark(process_data, data)

    # Assertions
    assert result is not None
```

## Results Storage

Results are stored in `results/` with timestamps:

```
results/
├── 2026-02-03_ntu_bench.json
├── 2026-02-03_fate_bench.json
└── history/
    └── ntu_observe_history.csv
```

## Dashboards

Performance dashboards available at:
- CI: GitHub Actions Artifacts
- Local: `target/criterion/report/index.html`

## Contact

Performance Specialist, BIZRA Elite Swarm
