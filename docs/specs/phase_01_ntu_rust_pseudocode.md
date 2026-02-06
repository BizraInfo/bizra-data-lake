# Phase 01: NTU Rust Implementation — Pseudocode

## Document Metadata
- **Version**: 1.0.0
- **Created**: 2026-02-03
- **Author**: SPARC Specification Writer
- **Status**: Draft
- **TDD Anchors**: Included for each module

---

## Module 1: Core Types (`types.rs`)

```pseudocode
// ============================================================================
// OBSERVATION TYPE ENUMERATION
// ============================================================================

/// TDD Anchor: test_observation_type_boundaries
ENUM ObservationType:
    LOW = 0      // value < 0.33
    MEDIUM = 1   // 0.33 <= value < 0.67
    HIGH = 2     // value >= 0.67

FUNCTION classify_observation(value: f64) -> ObservationType:
    IF value < 0.33:
        RETURN ObservationType::LOW
    ELSE IF value < 0.67:
        RETURN ObservationType::MEDIUM
    ELSE:
        RETURN ObservationType::HIGH

// ============================================================================
// OBSERVATION STRUCT
// ============================================================================

/// TDD Anchor: test_observation_clamping
STRUCT Observation:
    value: f64           // Clamped to [0.0, 1.0]
    metadata: Option<HashMap<String, Value>>

IMPL Observation:
    FUNCTION new(value: f64) -> Self:
        Self {
            value: clamp(value, 0.0, 1.0),
            metadata: None
        }

    FUNCTION with_metadata(value: f64, meta: HashMap) -> Self:
        Self {
            value: clamp(value, 0.0, 1.0),
            metadata: Some(meta)
        }

    FUNCTION observation_type(&self) -> ObservationType:
        classify_observation(self.value)

// ============================================================================
// NTU STATE
// ============================================================================

/// TDD Anchor: test_state_invariants
STRUCT NTUState:
    belief: f64      // [0.0, 1.0] - certainty
    entropy: f64     // [0.0, 1.0] - uncertainty
    potential: f64   // [0.0, 1.0] - predictive capacity
    iteration: u64   // Operation counter

IMPL NTUState:
    FUNCTION default() -> Self:
        Self {
            belief: 0.5,
            entropy: 1.0,
            potential: 0.5,
            iteration: 0
        }

    FUNCTION ihsan_achieved(&self, threshold: f64) -> bool:
        self.belief >= threshold

    FUNCTION as_vector(&self) -> [f64; 3]:
        [self.belief, self.entropy, self.potential]

// ============================================================================
// NTU CONFIGURATION
// ============================================================================

/// TDD Anchor: test_config_weight_normalization
STRUCT NTUConfig:
    window_size: usize
    alpha: f64           // Temporal decay weight
    beta: f64            // Neural prior weight
    gamma: f64           // Symbolic coherence weight
    ihsan_threshold: f64
    epsilon: f64         // Convergence criterion
    max_iterations: usize

IMPL NTUConfig:
    FUNCTION default() -> Self:
        Self {
            window_size: 5,
            alpha: 0.4,
            beta: 0.35,
            gamma: 0.25,
            ihsan_threshold: 0.95,
            epsilon: 0.01,
            max_iterations: 1000
        }

    /// Auto-normalize weights to sum to 1.0
    FUNCTION normalize_weights(&mut self):
        LET total = self.alpha + self.beta + self.gamma
        IF abs(total - 1.0) > 1e-10:
            self.alpha /= total
            self.beta /= total
            self.gamma /= total

    FUNCTION validate(&self) -> Result<(), NTUError>:
        IF self.window_size == 0:
            RETURN Err(NTUError::InvalidConfig("window_size must be > 0"))
        IF self.ihsan_threshold < 0.0 OR self.ihsan_threshold > 1.0:
            RETURN Err(NTUError::InvalidConfig("ihsan_threshold must be in [0,1]"))
        Ok(())
```

---

## Module 2: Embeddings (`embeddings.rs`)

```pseudocode
// ============================================================================
// PRETRAINED EMBEDDINGS (FROZEN)
// ============================================================================

/// TDD Anchor: test_embeddings_normalized
CONST EMBEDDING_LOW: [f64; 3] = normalize([0.1, 0.2, 0.7])
CONST EMBEDDING_MEDIUM: [f64; 3] = normalize([0.4, 0.4, 0.2])
CONST EMBEDDING_HIGH: [f64; 3] = normalize([0.8, 0.1, 0.1])

FUNCTION normalize(v: [f64; 3]) -> [f64; 3]:
    LET norm = sqrt(v[0]² + v[1]² + v[2]²)
    [v[0]/norm, v[1]/norm, v[2]/norm]

/// TDD Anchor: test_embedding_lookup
FUNCTION get_embedding(obs_type: ObservationType) -> [f64; 3]:
    MATCH obs_type:
        LOW => EMBEDDING_LOW
        MEDIUM => EMBEDDING_MEDIUM
        HIGH => EMBEDDING_HIGH
```

---

## Module 3: Markov Chain (`markov.rs`)

```pseudocode
// ============================================================================
// 3x3 TRANSITION MATRIX
// ============================================================================

/// TDD Anchor: test_transition_matrix_stochastic
CONST TRANSITION_MATRIX: [[f64; 3]; 3] = [
    [0.5, 0.4, 0.1],    // From LOW
    [0.2, 0.4, 0.4],    // From MEDIUM
    [0.05, 0.15, 0.8],  // From HIGH (sticky - convergence to Ihsan)
]

FUNCTION verify_stochastic(matrix: &[[f64; 3]; 3]) -> bool:
    FOR row IN matrix:
        LET sum = row[0] + row[1] + row[2]
        IF abs(sum - 1.0) > 1e-10:
            RETURN false
    RETURN true

// ============================================================================
// STATIONARY DISTRIBUTION
// ============================================================================

/// TDD Anchor: test_stationary_distribution_eigenvector
FUNCTION compute_stationary_distribution() -> [f64; 3]:
    // Solve π = π @ P via power iteration or eigendecomposition

    // Method: Power iteration (simple, converges for ergodic chains)
    LET mut pi = [1.0/3.0, 1.0/3.0, 1.0/3.0]  // Initial guess

    FOR _ IN 0..1000:
        LET new_pi = matrix_vector_multiply(transpose(TRANSITION_MATRIX), pi)

        // Check convergence
        IF max_diff(pi, new_pi) < 1e-10:
            BREAK

        pi = new_pi

    // Normalize
    LET sum = pi[0] + pi[1] + pi[2]
    [pi[0]/sum, pi[1]/sum, pi[2]/sum]

/// TDD Anchor: test_stationary_is_eigenvector
FUNCTION verify_stationary(pi: [f64; 3]) -> bool:
    LET result = matrix_vector_multiply(transpose(TRANSITION_MATRIX), pi)
    max_diff(pi, result) < 1e-6
```

---

## Module 4: Core NTU (`ntu.rs`)

```pseudocode
// ============================================================================
// NTU STRUCT
// ============================================================================

/// TDD Anchor: test_ntu_initialization
STRUCT NTU:
    config: NTUConfig
    state: NTUState
    memory: VecDeque<Observation>  // Sliding window

IMPL NTU:
    FUNCTION new(config: NTUConfig) -> Result<Self, NTUError>:
        config.validate()?

        LET mut normalized_config = config
        normalized_config.normalize_weights()

        Ok(Self {
            config: normalized_config,
            state: NTUState::default(),
            memory: VecDeque::with_capacity(config.window_size)
        })

    FUNCTION default() -> Self:
        Self::new(NTUConfig::default()).unwrap()

// ============================================================================
// OBSERVATION PROCESSING
// ============================================================================

/// TDD Anchor: test_observe_updates_state
IMPL NTU:
    FUNCTION observe(&mut self, value: f64) -> &NTUState:
        self.observe_with_metadata(value, None)

    FUNCTION observe_with_metadata(&mut self, value: f64, metadata: Option<HashMap>) -> &NTUState:
        // Validate input
        IF value.is_nan() OR value.is_infinite():
            // Log warning and use clamped value
            value = clamp(value, 0.0, 1.0)

        // Create observation
        LET obs = Observation::with_metadata(value, metadata)

        // Add to memory (sliding window)
        IF self.memory.len() >= self.config.window_size:
            self.memory.pop_front()
        self.memory.push_back(obs)

        // Compute update components
        LET temporal = self.compute_temporal_consistency()
        LET neural = self.compute_neural_prior(&obs)

        // Bayesian update
        self.update_state(temporal, neural)

        // Increment iteration
        self.state.iteration += 1

        &self.state

// ============================================================================
// TEMPORAL CONSISTENCY
// ============================================================================

/// TDD Anchor: test_temporal_consistency_stable_sequence
IMPL NTU:
    FUNCTION compute_temporal_consistency(&self) -> f64:
        IF self.memory.len() < 2:
            RETURN 0.5  // Insufficient data

        // Extract values
        LET values: Vec<f64> = self.memory.iter().map(|o| o.value).collect()

        // Compute variance
        LET mean = values.iter().sum() / values.len()
        LET variance = values.iter().map(|v| (v - mean)²).sum() / values.len()

        // Variance-based consistency: lower variance = higher consistency
        LET variance_score = exp(-4.0 * variance)

        // Monotonicity score: how consistently increasing/decreasing
        LET mut monotonic_sum = 0.0
        FOR i IN 1..values.len():
            LET diff = values[i] - values[i-1]
            monotonic_sum += sign(diff)

        LET monotonic_score = abs(monotonic_sum) / (values.len() - 1)

        // Combined consistency
        0.7 * variance_score + 0.3 * monotonic_score

// ============================================================================
// NEURAL PRIOR
// ============================================================================

/// TDD Anchor: test_neural_prior_lookup
IMPL NTU:
    FUNCTION compute_neural_prior(&self, obs: &Observation) -> [f64; 3]:
        get_embedding(obs.observation_type())

// ============================================================================
// BAYESIAN UPDATE
// ============================================================================

/// TDD Anchor: test_bayesian_update_convex
IMPL NTU:
    FUNCTION update_state(&mut self, temporal: f64, neural: [f64; 3]):
        LET α = self.config.alpha
        LET β = self.config.beta
        LET γ = self.config.gamma

        // Posterior belief (convex combination)
        LET prior_belief = self.state.belief
        LET posterior_belief = α * temporal + β * neural[0] + γ * prior_belief

        // Entropy from memory histogram
        LET posterior_entropy = self.compute_entropy()

        // Potential (predictive capacity)
        LET posterior_potential = posterior_belief * (1.0 - posterior_entropy)

        // Update state with clamping
        self.state.belief = clamp(posterior_belief, 0.0, 1.0)
        self.state.entropy = clamp(posterior_entropy, 0.0, 1.0)
        self.state.potential = clamp(posterior_potential, 0.0, 1.0)

    /// Shannon entropy from memory histogram
    FUNCTION compute_entropy(&self) -> f64:
        IF self.memory.is_empty():
            RETURN 1.0  // Maximum uncertainty

        // Bin observations into 3 categories
        LET mut counts = [0, 0, 0]
        FOR obs IN &self.memory:
            counts[obs.observation_type() as usize] += 1

        // Compute probabilities
        LET total = self.memory.len() as f64
        LET probs: Vec<f64> = counts.iter().map(|c| *c as f64 / total).collect()

        // Shannon entropy: -Σ p * log2(p)
        LET mut entropy = 0.0
        FOR p IN probs:
            IF p > 0.0:
                entropy -= p * log2(p + 1e-10)

        // Normalize by max entropy (log2(3) for 3 bins)
        entropy / log2(3.0)

// ============================================================================
// RESET
// ============================================================================

/// TDD Anchor: test_reset_clears_state
IMPL NTU:
    FUNCTION reset(&mut self):
        self.state = NTUState::default()
        self.memory.clear()
```

---

## Module 5: Pattern Detection (`detector.rs`)

```pseudocode
// ============================================================================
// SINGLE PATTERN DETECTION
// ============================================================================

/// TDD Anchor: test_detect_pattern_high_quality
IMPL NTU:
    FUNCTION detect_pattern(&mut self, observations: &[f64]) -> (bool, NTUState):
        self.reset()

        FOR value IN observations:
            self.observe(*value)

        LET detected = self.state.belief >= self.config.ihsan_threshold

        (detected, self.state.clone())

// ============================================================================
// CONVERGENCE DETECTION
// ============================================================================

/// TDD Anchor: test_convergence_with_stable_input
IMPL NTU:
    FUNCTION run_until_convergence(
        &mut self,
        observations: &[f64],
        epsilon: Option<f64>
    ) -> (bool, u64, NTUState):
        LET eps = epsilon.unwrap_or(self.config.epsilon)
        self.reset()

        LET mut prev_belief = self.state.belief
        LET mut converged = false
        LET mut obs_iter = observations.iter().cycle()

        FOR _ IN 0..self.config.max_iterations:
            LET value = obs_iter.next().unwrap()
            self.observe(*value)

            // Check convergence
            IF abs(self.state.belief - prev_belief) < eps:
                converged = true
                BREAK

            prev_belief = self.state.belief

        (converged, self.state.iteration, self.state.clone())

// ============================================================================
// PATTERN REGISTRY
// ============================================================================

/// TDD Anchor: test_pattern_registry
STRUCT PatternDetector:
    patterns: HashMap<String, NTUConfig>
    ntus: HashMap<String, NTU>

IMPL PatternDetector:
    FUNCTION new() -> Self:
        Self {
            patterns: HashMap::new(),
            ntus: HashMap::new()
        }

    FUNCTION register(
        &mut self,
        name: &str,
        threshold: f64,
        window_size: usize
    ) -> Result<(), NTUError>:
        LET config = NTUConfig {
            ihsan_threshold: threshold,
            window_size,
            ..Default::default()
        }

        self.patterns.insert(name.to_string(), config.clone())
        self.ntus.insert(name.to_string(), NTU::new(config)?)

        Ok(())

    FUNCTION detect(
        &mut self,
        pattern_name: &str,
        observations: &[f64]
    ) -> Result<(bool, f64, NTUState), NTUError>:
        LET ntu = self.ntus.get_mut(pattern_name)
            .ok_or(NTUError::UnknownPattern(pattern_name.to_string()))?

        LET (detected, state) = ntu.detect_pattern(observations)

        Ok((detected, state.belief, state))

    FUNCTION detect_all(
        &mut self,
        observations: &[f64]
    ) -> HashMap<String, (bool, f64)>:
        LET mut results = HashMap::new()

        FOR name IN self.patterns.keys():
            IF let Ok((detected, confidence, _)) = self.detect(name, observations):
                results.insert(name.clone(), (detected, confidence))

        results
```

---

## Module 6: PyO3 Bindings (`python.rs`)

```pseudocode
// ============================================================================
// PYTHON MODULE DEFINITION
// ============================================================================

/// TDD Anchor: test_pyo3_module_import
#[pymodule]
FUNCTION bizra_ntu(py: Python, m: &PyModule) -> PyResult<()>:
    m.add_class::<PyNTU>()?
    m.add_class::<PyNTUConfig>()?
    m.add_class::<PyNTUState>()?
    m.add_class::<PyObservation>()?
    m.add_class::<PyPatternDetector>()?
    m.add_function(wrap_pyfunction!(minimal_ntu_detect, m)?)?
    Ok(())

// ============================================================================
// PyNTU CLASS
// ============================================================================

/// TDD Anchor: test_pyntu_observe
#[pyclass(name = "NTU")]
STRUCT PyNTU:
    inner: NTU

#[pymethods]
IMPL PyNTU:
    #[new]
    FUNCTION new(config: Option<PyNTUConfig>) -> PyResult<Self>:
        LET rust_config = config.map(|c| c.into()).unwrap_or_default()
        LET ntu = NTU::new(rust_config)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
        Ok(Self { inner: ntu })

    FUNCTION observe(&mut self, value: f64) -> PyNTUState:
        self.inner.observe(value).clone().into()

    FUNCTION detect_pattern(&mut self, observations: Vec<f64>) -> (bool, PyNTUState):
        LET (detected, state) = self.inner.detect_pattern(&observations)
        (detected, state.into())

    #[getter]
    FUNCTION state(&self) -> PyNTUState:
        self.inner.state.clone().into()

    FUNCTION reset(&mut self):
        self.inner.reset()

    FUNCTION get_diagnostics(&self) -> PyObject:
        // Convert to Python dict
        ...

// ============================================================================
// MINIMAL DETECT FUNCTION
// ============================================================================

/// TDD Anchor: test_minimal_ntu_detect
#[pyfunction]
FUNCTION minimal_ntu_detect(
    py: Python,
    observations: Vec<f64>,
    threshold: Option<f64>,
    window: Option<usize>
) -> (bool, f64):
    // Release GIL for computation
    py.allow_threads(|| {
        LET config = NTUConfig {
            ihsan_threshold: threshold.unwrap_or(0.95),
            window_size: window.unwrap_or(5),
            ..Default::default()
        }

        LET mut ntu = NTU::new(config).unwrap()
        LET (detected, state) = ntu.detect_pattern(&observations)

        (detected, state.belief)
    })
```

---

## Module 7: Error Types (`error.rs`)

```pseudocode
// ============================================================================
// ERROR ENUMERATION
// ============================================================================

/// TDD Anchor: test_error_display
#[derive(Debug, thiserror::Error)]
ENUM NTUError:
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Invalid observation value: {0}")]
    InvalidValue(f64),

    #[error("Unknown pattern: {0}")]
    UnknownPattern(String),

    #[error("Computation error: {0}")]
    ComputationError(String),
```

---

## Module 8: SIMD Optimizations (`simd.rs`)

```pseudocode
// ============================================================================
// SIMD BATCH OBSERVATION (Optional)
// ============================================================================

/// TDD Anchor: test_simd_batch_observe
#[cfg(target_arch = "x86_64")]
FUNCTION batch_observe_simd(ntu: &mut NTU, values: &[f64]) -> NTUState:
    IF is_x86_feature_detected!("avx512f"):
        batch_observe_avx512(ntu, values)
    ELSE IF is_x86_feature_detected!("avx2"):
        batch_observe_avx2(ntu, values)
    ELSE:
        batch_observe_scalar(ntu, values)

FUNCTION batch_observe_scalar(ntu: &mut NTU, values: &[f64]) -> NTUState:
    FOR value IN values:
        ntu.observe(*value)
    ntu.state.clone()
```

---

## File Structure

```
bizra-omega/bizra-ntu/
├── Cargo.toml
├── src/
│   ├── lib.rs          // Module declarations
│   ├── types.rs        // ObservationType, Observation, NTUState, NTUConfig
│   ├── embeddings.rs   // Pretrained embeddings
│   ├── markov.rs       // Transition matrix, stationary distribution
│   ├── ntu.rs          // Core NTU implementation
│   ├── detector.rs     // PatternDetector
│   ├── error.rs        // NTUError
│   ├── simd.rs         // SIMD optimizations (optional)
│   └── python.rs       // PyO3 bindings
├── tests/
│   ├── test_types.rs
│   ├── test_markov.rs
│   ├── test_ntu.rs
│   ├── test_detector.rs
│   └── test_parity.rs  // Parity with Python tests
└── benches/
    └── ntu_benchmark.rs
```

---

## TDD Test Checklist

| Test Name | Module | Priority |
|-----------|--------|----------|
| test_observation_clamping | types.rs | P0 |
| test_observation_type_boundaries | types.rs | P0 |
| test_state_invariants | types.rs | P0 |
| test_config_weight_normalization | types.rs | P0 |
| test_embeddings_normalized | embeddings.rs | P1 |
| test_embedding_lookup | embeddings.rs | P1 |
| test_transition_matrix_stochastic | markov.rs | P0 |
| test_stationary_distribution_eigenvector | markov.rs | P1 |
| test_ntu_initialization | ntu.rs | P0 |
| test_observe_updates_state | ntu.rs | P0 |
| test_temporal_consistency_stable_sequence | ntu.rs | P1 |
| test_neural_prior_lookup | ntu.rs | P1 |
| test_bayesian_update_convex | ntu.rs | P0 |
| test_reset_clears_state | ntu.rs | P0 |
| test_detect_pattern_high_quality | detector.rs | P0 |
| test_convergence_with_stable_input | detector.rs | P1 |
| test_pattern_registry | detector.rs | P1 |
| test_pyo3_module_import | python.rs | P0 |
| test_pyntu_observe | python.rs | P0 |
| test_minimal_ntu_detect | python.rs | P0 |
| test_python_parity_all | test_parity.rs | P0 |
