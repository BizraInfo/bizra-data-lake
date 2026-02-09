# bizra-tests

**BIZRA End-to-End Integration Tests** — Cross-crate integration and property-based testing.

## Overview

Standalone test crate exercising the full BIZRA stack:

- **Integration Tests** — Cross-crate workflows (identity → PCI → inference → federation)
- **Property-Based Tests** — Proptest-driven invariant verification
- **Performance Benchmarks** — Crypto, PCI, gates, inference, SIMD operations
- **Memory Profiling** — Struct size estimation for key types

## Running

```bash
# Unit + integration tests
cargo test -p bizra-tests

# Performance benchmarks (custom harness)
cargo bench -p bizra-tests
```

## Dependencies

- bizra-core (dev-mode), bizra-inference, bizra-federation, bizra-autopoiesis
- proptest, mockall — Testing utilities

## License

MIT
