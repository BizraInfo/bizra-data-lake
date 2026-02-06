# Phase 01: NTU Rust Implementation — Requirements Specification

## Document Metadata
- **Version**: 1.0.0
- **Created**: 2026-02-03
- **Author**: SPARC Specification Writer
- **Status**: Draft
- **Predecessor**: Python NTU (core/ntu/) — 73 tests passing

---

## 1. Executive Summary

### 1.1 Purpose
Implement the NeuroTemporal Unit (NTU) in Rust for maximum performance, then expose it to Python via PyO3 bindings. This enables:
- **10-100x performance improvement** over pure Python
- **Memory safety guarantees** via Rust ownership
- **SIMD acceleration** on modern CPUs (AVX-512 on RTX 4090 system)
- **WebAssembly compilation** for browser/edge deployment

### 1.2 Success Criteria
| Metric | Target |
|--------|--------|
| Execution time | < 1ms per observation (vs 8ms Python) |
| Memory usage | < 1KB per NTU instance |
| Test coverage | 100% parity with Python tests |
| PyO3 overhead | < 10μs per FFI call |
| WASM binary size | < 100KB |

### 1.3 Constraints
- Must maintain **exact behavioral parity** with Python NTU
- Must pass all 73 existing Python tests via PyO3
- Must integrate with existing `bizra-omega` workspace
- Must support `#![no_std]` for embedded deployment

---

## 2. Functional Requirements

### 2.1 Core NTU State Machine

```
FR-NTU-001: State Representation
  - belief: f64 in [0.0, 1.0] — certainty score
  - entropy: f64 in [0.0, 1.0] — uncertainty measure
  - potential: f64 in [0.0, 1.0] — predictive capacity
  - iteration: u64 — operation counter
```

```
FR-NTU-002: Observation Processing
  INPUT: value ∈ [0.0, 1.0], optional metadata
  OUTPUT: Updated NTUState
  INVARIANT: All state values remain in [0.0, 1.0]
  COMPLEXITY: O(1) per observation, O(window_size) for consistency
```

```
FR-NTU-003: Temporal Consistency
  INPUT: Memory window (VecDeque<Observation>)
  OUTPUT: consistency_score ∈ [0.0, 1.0]
  METHOD: Variance + monotonicity analysis
  FORMULA: consistency = 0.7 * exp(-4*variance) + 0.3 * |mean(sign(diff))|
```

```
FR-NTU-004: Neural Prior Lookup
  INPUT: ObservationType {LOW, MEDIUM, HIGH}
  OUTPUT: [f64; 3] normalized embedding
  METHOD: O(1) lookup from frozen embeddings
  EMBEDDINGS:
    - LOW: [0.1, 0.2, 0.7] normalized
    - MEDIUM: [0.4, 0.4, 0.2] normalized
    - HIGH: [0.8, 0.1, 0.1] normalized
```

```
FR-NTU-005: Bayesian Update
  INPUT: temporal_consistency, neural_prior, current_state
  OUTPUT: posterior_state
  FORMULA:
    belief_new = α*temporal + β*neural[0] + γ*belief_current
    entropy_new = Shannon_entropy(memory_histogram)
    potential_new = belief_new * (1 - entropy_new)
  CONSTRAINT: α + β + γ = 1.0 (convex combination)
```

### 2.2 Configuration

```
FR-CFG-001: NTUConfig
  - window_size: usize (default: 5, range: 1..1000)
  - alpha: f64 (default: 0.4, range: 0.0..1.0)
  - beta: f64 (default: 0.35, range: 0.0..1.0)
  - gamma: f64 (default: 0.25, range: 0.0..1.0)
  - ihsan_threshold: f64 (default: 0.95, range: 0.5..1.0)
  - epsilon: f64 (default: 0.01, convergence criterion)
  - max_iterations: usize (default: 1000)
  INVARIANT: alpha + beta + gamma = 1.0 (auto-normalize if violated)
```

### 2.3 Pattern Detection

```
FR-PAT-001: Single Pattern Detection
  INPUT: Vec<f64> observations
  OUTPUT: (detected: bool, final_state: NTUState)
  detected = state.belief >= config.ihsan_threshold
```

```
FR-PAT-002: Convergence Detection
  INPUT: Vec<f64> observations, epsilon: f64
  OUTPUT: (converged: bool, iterations: u64, final_state: NTUState)
  converged = |belief_new - belief_old| < epsilon
```

```
FR-PAT-003: Multi-Pattern Registry
  - Register named patterns with custom thresholds
  - Detect specific pattern by name
  - Detect all patterns and return results
```

### 2.4 Markov Chain Properties

```
FR-MKV-001: Transition Matrix
  - 3x3 row-stochastic matrix
  - States: [LOW, MEDIUM, HIGH]
  - Default transitions encoding "convergence to Ihsan":
    P = [[0.5, 0.4, 0.1],   // From LOW
         [0.2, 0.4, 0.4],   // From MEDIUM
         [0.05, 0.15, 0.8]] // From HIGH (sticky)
```

```
FR-MKV-002: Stationary Distribution
  OUTPUT: [f64; 3] probability distribution
  METHOD: Left eigenvector with eigenvalue 1
  PROPERTY: π @ P = π
```

---

## 3. Non-Functional Requirements

### 3.1 Performance

```
NFR-PERF-001: Observation Latency
  - Single observation: < 100ns (excluding FFI)
  - Batch of 1000 observations: < 100μs
  - Pattern detection on 1000 values: < 1ms
```

```
NFR-PERF-002: Memory Efficiency
  - NTU instance: < 500 bytes (excluding memory window)
  - Per-observation overhead: 24 bytes (f64 + metadata pointer)
  - Maximum memory window: configurable, default 5 observations
```

```
NFR-PERF-003: SIMD Optimization
  - Utilize AVX-512 for batch operations where available
  - Fallback to AVX2/SSE for older CPUs
  - Automatic detection via `is_x86_feature_detected!`
```

### 3.2 Safety

```
NFR-SAFE-001: Memory Safety
  - No unsafe code except for SIMD intrinsics
  - All unsafe blocks documented with safety invariants
  - Miri clean under `cargo miri test`
```

```
NFR-SAFE-002: Panic Freedom
  - Core NTU operations must not panic
  - Use Result<T, NTUError> for fallible operations
  - #[cfg(not(feature = "panic"))] removes all panic paths
```

```
NFR-SAFE-003: Thread Safety
  - NTU: Send + !Sync (single-threaded observation)
  - NTUConfig: Send + Sync (immutable after construction)
  - PatternDetector: Send + Sync with interior mutability
```

### 3.3 Portability

```
NFR-PORT-001: Platform Support
  - Primary: x86_64-unknown-linux-gnu (RTX 4090 system)
  - Secondary: x86_64-pc-windows-msvc (Windows development)
  - Tertiary: wasm32-unknown-unknown (browser deployment)
  - Future: aarch64-unknown-linux-gnu (ARM servers)
```

```
NFR-PORT-002: no_std Support
  - Core NTU must compile with #![no_std]
  - Use alloc crate for dynamic memory
  - Feature flag: default-features = false
```

---

## 4. Edge Cases and Error Handling

### 4.1 Input Validation

| Input | Edge Case | Handling |
|-------|-----------|----------|
| observation value | < 0.0 | Clamp to 0.0 |
| observation value | > 1.0 | Clamp to 1.0 |
| observation value | NaN | Return Err(NTUError::InvalidValue) |
| observation value | Inf | Return Err(NTUError::InvalidValue) |
| empty observations | detect_pattern([]) | Return (false, initial_state) |
| weights sum != 1.0 | α+β+γ = 1.5 | Auto-normalize to sum = 1.0 |

### 4.2 State Invariants

```
INVARIANT-001: State Bounds
  POST: 0.0 <= belief <= 1.0
  POST: 0.0 <= entropy <= 1.0
  POST: 0.0 <= potential <= 1.0
```

```
INVARIANT-002: Monotonic Iteration
  POST: state.iteration == old.iteration + 1 after observe()
```

```
INVARIANT-003: Consistent Potential
  POST: potential == belief * (1.0 - entropy) (within f64 precision)
```

### 4.3 Numerical Stability

```
STABILITY-001: Division by Zero
  - Avoid division by zero in entropy calculation
  - Use epsilon = 1e-10 for log calculations
  - Formula: -sum(p * log2(p + 1e-10))
```

```
STABILITY-002: Floating Point Comparison
  - Use approx crate for f64 comparisons in tests
  - Tolerance: 1e-10 for state values
  - Tolerance: 1e-6 for Markov chain eigenvector
```

---

## 5. Integration Points

### 5.1 PyO3 Bindings

```
INT-PYO3-001: Python Module
  - Module name: bizra_ntu
  - Classes: NTU, NTUConfig, NTUState, Observation, PatternDetector
  - Functions: minimal_ntu_detect(observations, threshold, window)
```

```
INT-PYO3-002: Type Mappings
  | Rust | Python |
  |------|--------|
  | f64 | float |
  | usize | int |
  | bool | bool |
  | Vec<f64> | List[float] |
  | Option<T> | Optional[T] |
  | Result<T, E> | raises Exception |
```

```
INT-PYO3-003: GIL Management
  - Release GIL during batch operations
  - Use py.allow_threads(|| { ... }) for parallel execution
  - Document thread safety in Python docstrings
```

### 5.2 bizra-omega Integration

```
INT-OMEGA-001: Workspace Member
  - Add bizra-ntu to workspace members in Cargo.toml
  - Share dependencies: serde, thiserror, rayon
```

```
INT-OMEGA-002: Feature Flags
  - default: ["std", "pyo3"]
  - no_std: no-default-features
  - simd: ["std", "packed_simd"]
  - wasm: ["no_std", "wasm-bindgen"]
```

### 5.3 SNR Bridge

```
INT-SNR-001: SNR Adapter
  - Accept SNRComponentsV2 and convert to observation
  - Weighted combination: signal*0.4 + diversity*0.3 + grounding*0.3
  - Return NTUState for quality assessment
```

---

## 6. Test Requirements

### 6.1 Unit Tests (Rust)

```
TEST-UNIT-001: State Clamping
  - Verify values outside [0,1] are clamped
  - Verify NaN/Inf are rejected
```

```
TEST-UNIT-002: Observation Types
  - value=0.1 → LOW
  - value=0.5 → MEDIUM
  - value=0.9 → HIGH
```

```
TEST-UNIT-003: Weight Normalization
  - (1.0, 1.0, 1.0) → (0.333, 0.333, 0.333)
  - Verify sum = 1.0 within tolerance
```

```
TEST-UNIT-004: Markov Properties
  - Transition matrix is row-stochastic
  - Stationary distribution sums to 1.0
  - π @ P ≈ π
```

### 6.2 Integration Tests (PyO3)

```
TEST-PYO3-001: Python Parity
  - Run all 73 Python tests through PyO3 bindings
  - Verify identical results within f64 tolerance
```

```
TEST-PYO3-002: Performance
  - Benchmark Python NTU vs Rust NTU
  - Verify 10x+ speedup
```

### 6.3 Property Tests (proptest)

```
TEST-PROP-001: State Bounds
  - For all observations, state remains in [0,1]³
```

```
TEST-PROP-002: Convergence
  - For stable input, belief converges
```

---

## 7. Dependencies

### 7.1 Required Crates

| Crate | Version | Purpose |
|-------|---------|---------|
| thiserror | 1.0 | Error types |
| serde | 1.0 | Serialization |
| pyo3 | 0.20 | Python bindings |
| approx | 0.5 | Float comparison |

### 7.2 Optional Crates

| Crate | Feature | Purpose |
|-------|---------|---------|
| rayon | parallel | Parallel batch processing |
| proptest | proptest | Property-based testing |
| criterion | bench | Benchmarking |

---

## 8. Acceptance Criteria

- [ ] All 73 Python tests pass via PyO3
- [ ] Benchmark shows 10x+ speedup
- [ ] Memory usage < 1KB per instance
- [ ] `cargo clippy` clean
- [ ] `cargo miri test` passes
- [ ] Documentation complete with examples
- [ ] WASM build succeeds (wasm32-unknown-unknown)

---

## Appendix A: Reference Implementation

See: `/mnt/c/BIZRA-DATA-LAKE/core/ntu/ntu.py` (Python reference)

## Appendix B: Related Specifications

- Phase 02: FATE Gate Rust Implementation
- Phase 03: Session DAG Rust Implementation
- Phase 04: Compute Market Rust Implementation
