# Phase 01: NTU Rust Implementation — TDD Anchors

## Document Metadata
- **Version**: 1.0.0
- **Created**: 2026-02-03
- **Author**: SPARC Specification Writer
- **Purpose**: Test-Driven Development anchors for Rust NTU

---

## TDD Philosophy

> "Write tests first, then write code to make them pass."

Each anchor below defines:
1. **Test name** (snake_case)
2. **Input** specification
3. **Expected output**
4. **Invariants** to verify
5. **Edge cases** to cover

---

## Module: types.rs

### TDD-TYPES-001: test_observation_clamping

```rust
#[test]
fn test_observation_clamping() {
    // Values outside [0,1] should be clamped
    let obs_high = Observation::new(1.5);
    assert_eq!(obs_high.value, 1.0);

    let obs_low = Observation::new(-0.5);
    assert_eq!(obs_low.value, 0.0);

    let obs_valid = Observation::new(0.7);
    assert!((obs_valid.value - 0.7).abs() < 1e-10);
}
```

### TDD-TYPES-002: test_observation_type_boundaries

```rust
#[test]
fn test_observation_type_boundaries() {
    // Boundary at 0.33
    assert_eq!(Observation::new(0.32).observation_type(), ObservationType::LOW);
    assert_eq!(Observation::new(0.34).observation_type(), ObservationType::MEDIUM);

    // Boundary at 0.67
    assert_eq!(Observation::new(0.66).observation_type(), ObservationType::MEDIUM);
    assert_eq!(Observation::new(0.68).observation_type(), ObservationType::HIGH);

    // Extremes
    assert_eq!(Observation::new(0.0).observation_type(), ObservationType::LOW);
    assert_eq!(Observation::new(1.0).observation_type(), ObservationType::HIGH);
}
```

### TDD-TYPES-003: test_state_invariants

```rust
#[test]
fn test_state_invariants() {
    let state = NTUState::default();

    // Default values
    assert!((state.belief - 0.5).abs() < 1e-10);
    assert!((state.entropy - 1.0).abs() < 1e-10);
    assert!((state.potential - 0.5).abs() < 1e-10);
    assert_eq!(state.iteration, 0);

    // Bounds
    assert!(state.belief >= 0.0 && state.belief <= 1.0);
    assert!(state.entropy >= 0.0 && state.entropy <= 1.0);
    assert!(state.potential >= 0.0 && state.potential <= 1.0);
}
```

### TDD-TYPES-004: test_config_weight_normalization

```rust
#[test]
fn test_config_weight_normalization() {
    let mut config = NTUConfig {
        alpha: 1.0,
        beta: 1.0,
        gamma: 1.0,
        ..Default::default()
    };

    config.normalize_weights();

    // Should be normalized to 1/3 each
    assert!((config.alpha - 1.0/3.0).abs() < 1e-10);
    assert!((config.beta - 1.0/3.0).abs() < 1e-10);
    assert!((config.gamma - 1.0/3.0).abs() < 1e-10);

    // Sum should be 1.0
    let sum = config.alpha + config.beta + config.gamma;
    assert!((sum - 1.0).abs() < 1e-10);
}
```

### TDD-TYPES-005: test_config_validation

```rust
#[test]
fn test_config_validation() {
    // Valid config
    let valid = NTUConfig::default();
    assert!(valid.validate().is_ok());

    // Invalid window_size
    let invalid_window = NTUConfig {
        window_size: 0,
        ..Default::default()
    };
    assert!(invalid_window.validate().is_err());

    // Invalid threshold
    let invalid_threshold = NTUConfig {
        ihsan_threshold: 1.5,
        ..Default::default()
    };
    assert!(invalid_threshold.validate().is_err());
}
```

---

## Module: embeddings.rs

### TDD-EMB-001: test_embeddings_normalized

```rust
#[test]
fn test_embeddings_normalized() {
    let embeddings = [EMBEDDING_LOW, EMBEDDING_MEDIUM, EMBEDDING_HIGH];

    for emb in embeddings {
        let norm = (emb[0].powi(2) + emb[1].powi(2) + emb[2].powi(2)).sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "Embedding should be unit vector");
    }
}
```

### TDD-EMB-002: test_embedding_lookup

```rust
#[test]
fn test_embedding_lookup() {
    let low = get_embedding(ObservationType::LOW);
    let med = get_embedding(ObservationType::MEDIUM);
    let high = get_embedding(ObservationType::HIGH);

    // Should be distinct
    assert!(low != med);
    assert!(med != high);
    assert!(low != high);

    // LOW should have highest third component (entropy-heavy)
    assert!(low[2] > low[0]);

    // HIGH should have highest first component (belief-heavy)
    assert!(high[0] > high[2]);
}
```

---

## Module: markov.rs

### TDD-MKV-001: test_transition_matrix_stochastic

```rust
#[test]
fn test_transition_matrix_stochastic() {
    for row in TRANSITION_MATRIX.iter() {
        let sum: f64 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Row sum should be 1.0, got {}",
            sum
        );
    }
}
```

### TDD-MKV-002: test_transition_matrix_nonnegative

```rust
#[test]
fn test_transition_matrix_nonnegative() {
    for row in TRANSITION_MATRIX.iter() {
        for &val in row.iter() {
            assert!(val >= 0.0, "Transition probabilities must be non-negative");
        }
    }
}
```

### TDD-MKV-003: test_stationary_distribution_valid

```rust
#[test]
fn test_stationary_distribution_valid() {
    let pi = compute_stationary_distribution();

    // Should sum to 1
    let sum: f64 = pi.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);

    // All non-negative
    for &p in pi.iter() {
        assert!(p >= 0.0);
    }
}
```

### TDD-MKV-004: test_stationary_is_eigenvector

```rust
#[test]
fn test_stationary_is_eigenvector() {
    let pi = compute_stationary_distribution();

    // π @ P should equal π
    let mut result = [0.0; 3];
    for i in 0..3 {
        for j in 0..3 {
            result[i] += pi[j] * TRANSITION_MATRIX[j][i];
        }
    }

    for i in 0..3 {
        assert!(
            (result[i] - pi[i]).abs() < 1e-6,
            "Stationary distribution should satisfy π = π @ P"
        );
    }
}
```

---

## Module: ntu.rs

### TDD-NTU-001: test_ntu_initialization

```rust
#[test]
fn test_ntu_initialization() {
    let ntu = NTU::default();

    assert_eq!(ntu.memory.len(), 0);
    assert!((ntu.state.belief - 0.5).abs() < 1e-10);
    assert!((ntu.state.entropy - 1.0).abs() < 1e-10);
    assert_eq!(ntu.config.window_size, 5);
}
```

### TDD-NTU-002: test_observe_updates_state

```rust
#[test]
fn test_observe_updates_state() {
    let mut ntu = NTU::default();
    let initial_iteration = ntu.state.iteration;

    ntu.observe(0.9);

    assert_eq!(ntu.memory.len(), 1);
    assert_eq!(ntu.state.iteration, initial_iteration + 1);
}
```

### TDD-NTU-003: test_memory_sliding_window

```rust
#[test]
fn test_memory_sliding_window() {
    let config = NTUConfig {
        window_size: 3,
        ..Default::default()
    };
    let mut ntu = NTU::new(config).unwrap();

    ntu.observe(0.1);
    ntu.observe(0.2);
    ntu.observe(0.3);
    ntu.observe(0.4);
    ntu.observe(0.5);

    // Should have last 3 observations
    assert_eq!(ntu.memory.len(), 3);

    let values: Vec<f64> = ntu.memory.iter().map(|o| o.value).collect();
    assert!((values[0] - 0.3).abs() < 1e-10);
    assert!((values[1] - 0.4).abs() < 1e-10);
    assert!((values[2] - 0.5).abs() < 1e-10);
}
```

### TDD-NTU-004: test_high_observation_increases_belief

```rust
#[test]
fn test_high_observation_increases_belief() {
    let mut ntu = NTU::default();

    // Start with neutral
    for _ in 0..5 {
        ntu.observe(0.5);
    }
    let initial_belief = ntu.state.belief;

    // Feed high observations
    for _ in 0..10 {
        ntu.observe(0.95);
    }

    assert!(ntu.state.belief > initial_belief);
}
```

### TDD-NTU-005: test_low_observation_decreases_belief

```rust
#[test]
fn test_low_observation_decreases_belief() {
    let mut ntu = NTU::default();

    // Start high
    for _ in 0..10 {
        ntu.observe(0.95);
    }
    let high_belief = ntu.state.belief;

    // Feed low observations
    for _ in 0..10 {
        ntu.observe(0.1);
    }

    assert!(ntu.state.belief < high_belief);
}
```

### TDD-NTU-006: test_reset_clears_state

```rust
#[test]
fn test_reset_clears_state() {
    let mut ntu = NTU::default();

    for _ in 0..10 {
        ntu.observe(0.9);
    }

    ntu.reset();

    assert_eq!(ntu.memory.len(), 0);
    assert!((ntu.state.belief - 0.5).abs() < 1e-10);
    assert!((ntu.state.entropy - 1.0).abs() < 1e-10);
    assert_eq!(ntu.state.iteration, 0);
}
```

### TDD-NTU-007: test_temporal_consistency_stable_sequence

```rust
#[test]
fn test_temporal_consistency_stable_sequence() {
    let mut ntu = NTU::default();

    // Monotonic increasing sequence
    for i in 0..10 {
        ntu.observe(0.5 + (i as f64) * 0.05);
    }

    // Should have high belief due to consistency
    assert!(ntu.state.belief > 0.5);
}
```

### TDD-NTU-008: test_temporal_consistency_oscillating_sequence

```rust
#[test]
fn test_temporal_consistency_oscillating_sequence() {
    let mut ntu = NTU::default();

    // Oscillating sequence
    for i in 0..10 {
        let value = if i % 2 == 0 { 0.9 } else { 0.1 };
        ntu.observe(value);
    }

    // Should have higher entropy due to inconsistency
    assert!(ntu.state.entropy > 0.3);
}
```

### TDD-NTU-009: test_state_bounds_maintained

```rust
#[test]
fn test_state_bounds_maintained() {
    let mut ntu = NTU::default();

    // Feed random values
    let values = [0.0, 1.0, 0.5, 0.99, 0.01, 0.7, 0.3, 0.8, 0.2, 0.6];

    for value in values {
        ntu.observe(value);

        // Check bounds after each observation
        assert!(ntu.state.belief >= 0.0 && ntu.state.belief <= 1.0);
        assert!(ntu.state.entropy >= 0.0 && ntu.state.entropy <= 1.0);
        assert!(ntu.state.potential >= 0.0 && ntu.state.potential <= 1.0);
    }
}
```

---

## Module: detector.rs

### TDD-DET-001: test_detect_pattern_high_quality

```rust
#[test]
fn test_detect_pattern_high_quality() {
    let mut ntu = NTU::default();

    let observations = vec![0.95, 0.96, 0.94, 0.97, 0.95, 0.96, 0.95, 0.98];
    let (detected, state) = ntu.detect_pattern(&observations);

    assert!(state.belief > 0.7);
}
```

### TDD-DET-002: test_detect_pattern_low_quality

```rust
#[test]
fn test_detect_pattern_low_quality() {
    let mut ntu = NTU::default();

    let observations = vec![0.1, 0.15, 0.2, 0.1, 0.05, 0.15, 0.1, 0.2];
    let (detected, state) = ntu.detect_pattern(&observations);

    assert!(!detected);
    assert!(state.belief < 0.95);
}
```

### TDD-DET-003: test_detect_pattern_resets_state

```rust
#[test]
fn test_detect_pattern_resets_state() {
    let mut ntu = NTU::default();

    // First run
    ntu.detect_pattern(&[0.1, 0.2, 0.3]);

    // Second run
    let (_, state) = ntu.detect_pattern(&[0.9, 0.95, 0.92]);

    // Should not carry over from first run
    assert_eq!(state.iteration, 3);
}
```

### TDD-DET-004: test_convergence_with_stable_input

```rust
#[test]
fn test_convergence_with_stable_input() {
    let mut ntu = NTU::default();

    let observations: Vec<f64> = vec![0.8; 20];
    let (converged, iterations, _) = ntu.run_until_convergence(&observations, None);

    // Should converge with stable input
    assert!(iterations < 100);
}
```

### TDD-DET-005: test_convergence_max_iterations

```rust
#[test]
fn test_convergence_max_iterations() {
    let config = NTUConfig {
        max_iterations: 50,
        ..Default::default()
    };
    let mut ntu = NTU::new(config).unwrap();

    // Oscillating input won't converge easily
    let observations: Vec<f64> = vec![0.1, 0.9].repeat(50);
    let (_, iterations, _) = ntu.run_until_convergence(&observations, None);

    assert!(iterations <= 50);
}
```

### TDD-DET-006: test_pattern_registry_register

```rust
#[test]
fn test_pattern_registry_register() {
    let mut detector = PatternDetector::new();

    detector.register("high_quality", 0.95, 5).unwrap();
    detector.register("medium_quality", 0.85, 10).unwrap();

    assert!(detector.patterns.contains_key("high_quality"));
    assert!(detector.patterns.contains_key("medium_quality"));
}
```

### TDD-DET-007: test_pattern_registry_detect

```rust
#[test]
fn test_pattern_registry_detect() {
    let mut detector = PatternDetector::new();
    detector.register("test_pattern", 0.7, 3).unwrap();

    let observations = vec![0.8, 0.85, 0.9, 0.88, 0.9];
    let result = detector.detect("test_pattern", &observations);

    assert!(result.is_ok());
    let (detected, confidence, _) = result.unwrap();
    assert!(confidence >= 0.0 && confidence <= 1.0);
}
```

### TDD-DET-008: test_pattern_registry_unknown_pattern

```rust
#[test]
fn test_pattern_registry_unknown_pattern() {
    let mut detector = PatternDetector::new();

    let result = detector.detect("nonexistent", &[0.5, 0.6]);

    assert!(result.is_err());
    match result {
        Err(NTUError::UnknownPattern(_)) => {}
        _ => panic!("Expected UnknownPattern error"),
    }
}
```

---

## Module: python.rs (PyO3)

### TDD-PYO3-001: test_pyntu_creation

```rust
#[test]
fn test_pyntu_creation() {
    Python::with_gil(|py| {
        let ntu = PyNTU::new(None).unwrap();
        assert!(ntu.inner.memory.is_empty());
    });
}
```

### TDD-PYO3-002: test_pyntu_observe

```rust
#[test]
fn test_pyntu_observe() {
    Python::with_gil(|py| {
        let mut ntu = PyNTU::new(None).unwrap();

        let state = ntu.observe(0.9);

        assert!(state.iteration == 1);
    });
}
```

### TDD-PYO3-003: test_minimal_ntu_detect

```rust
#[test]
fn test_minimal_ntu_detect() {
    Python::with_gil(|py| {
        let observations = vec![0.9, 0.92, 0.88, 0.95, 0.91];
        let (detected, confidence) = minimal_ntu_detect(py, observations, None, None);

        assert!(confidence >= 0.0 && confidence <= 1.0);
    });
}
```

---

## Python Parity Tests

### TDD-PARITY-001: test_python_parity_observation_clamping

```python
# Python reference
from core.ntu import Observation
obs = Observation(value=1.5)
assert obs.value == 1.0
```

```rust
// Rust must match
#[test]
fn test_python_parity_observation_clamping() {
    let obs = Observation::new(1.5);
    assert_eq!(obs.value, 1.0);
}
```

### TDD-PARITY-002: test_python_parity_full_sequence

```rust
#[test]
fn test_python_parity_full_sequence() {
    // Test sequence from Python tests
    let observations = vec![0.9, 0.92, 0.88, 0.95, 0.91];

    // Run Rust NTU
    let mut rust_ntu = NTU::default();
    for v in &observations {
        rust_ntu.observe(*v);
    }

    // Expected values from Python (with tolerance)
    // These should be verified against actual Python output
    assert!(rust_ntu.state.belief > 0.7);
    assert!(rust_ntu.state.entropy < 0.5);
}
```

---

## Benchmark Anchors

### BENCH-001: benchmark_single_observation

```rust
fn benchmark_single_observation(c: &mut Criterion) {
    let mut ntu = NTU::default();

    c.bench_function("single_observation", |b| {
        b.iter(|| ntu.observe(black_box(0.9)))
    });
}
```

Target: < 100ns

### BENCH-002: benchmark_batch_observation

```rust
fn benchmark_batch_observation(c: &mut Criterion) {
    let mut ntu = NTU::default();
    let observations: Vec<f64> = (0..1000).map(|i| (i as f64) / 1000.0).collect();

    c.bench_function("batch_1000_observations", |b| {
        b.iter(|| {
            ntu.reset();
            for v in &observations {
                ntu.observe(black_box(*v));
            }
        })
    });
}
```

Target: < 100μs

### BENCH-003: benchmark_pattern_detection

```rust
fn benchmark_pattern_detection(c: &mut Criterion) {
    let observations: Vec<f64> = (0..1000).map(|i| (i as f64) / 1000.0).collect();

    c.bench_function("detect_pattern_1000", |b| {
        b.iter(|| {
            let mut ntu = NTU::default();
            ntu.detect_pattern(black_box(&observations))
        })
    });
}
```

Target: < 1ms
