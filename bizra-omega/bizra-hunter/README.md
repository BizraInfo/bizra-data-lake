# bizra-hunter

**BIZRA Hunter** — SNR-Maximized Vulnerability Discovery.

## Overview

Zero-allocation, SIMD-accelerated, lock-free vulnerability hunting engine.

**Target: 47.9M ops/sec sustained throughput.**

## The 7 Quiet Tricks

1. **Two-Lane Pipeline** — Fast heuristics → Expensive proofs
2. **Invariant Deduplication** — O(1) uniqueness check via hash cache
3. **Multi-Axis Entropy** — SIMD-accelerated Shannon entropy calculation
4. **Challenge Bonds** — Economic truth enforcement
5. **Safe PoC** — Non-weaponized proof generation
6. **Harberger Rent** — Spam prevention via economic friction
7. **Critical Cascade** — Fail-safe gate enforcement

## Benchmarks

Run with:
```bash
cargo bench -p bizra-hunter
```

Benchmark groups: entropy, invariant_cache, cascade, lane1, queue, throughput.

## Features

| Feature | Description |
|---------|-------------|
| `simd` | SIMD-accelerated entropy (default) |
| `snr-max` | Full SNR optimization |
| `avx512` | AVX-512 support (CPU-dependent) |

## License

MIT
