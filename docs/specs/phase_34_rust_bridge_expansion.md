# Phase 34: Rust Bridge Expansion — PyO3 Bindings for Omega/PAT/GoT/Federation

> Exposes the remaining Rust-only types to Python via PyO3, enabling the Python Sovereign runtime to leverage Rust-speed constitutional gates, PAT minting, GoT traversal, and federation consensus.

Standing on Giants: Lamport (1978, distributed system interfaces must be formally typed) + Hoare (1969, communicating sequential processes — typed FFI boundaries) + Anthropic (2023, constitutional AI gates must execute at wire speed)

## Context

`bizra-omega/bizra-python/src/lib.rs` (924 lines) exposes 13 PyO3 classes. However, `bizra-core` defines 20+ public types (including `OmegaEngine`, `TreasuryController`, `AgentMintingEngine`, `ThoughtGraph`, `SNREngine`, `AdlInvariant`) that have **no Python bindings**. The federation crate defines `ByzantineConsensus` and gossip types that are Rust-only.

## Gaps Addressed

| Rust Type | Crate | PyO3 Status | Impact |
|-----------|-------|-------------|--------|
| `OmegaEngine` | bizra-core | Missing | Ihsan vector computation stuck in Python |
| `TreasuryController` | bizra-core | Missing | Resource allocation logic duplicated |
| `AgentMintingEngine` | bizra-core | Missing | PAT creation is Python-only |
| `ThoughtGraph` / `GoT` | bizra-core | Missing | Graph-of-Thoughts runs in Python networkx |
| `SNREngine` | bizra-core | Missing | SNR scoring duplicated in Python |
| `AdlInvariant` | bizra-core | Missing | Justice gate is Python-only |
| `ZakatCalculator` | bizra-core | Missing | Islamic finance rules in Rust unreachable |
| `IhsanProjector` | bizra-core | Missing | Ihsan scoring in Rust unreachable |
| `ByzantineConsensus` | bizra-federation | Missing | BFT runs in Python only |

## Binding Strategy

### Phase 34a: Constitutional Gates (Priority 1)

These types are hot-path — called on every query. Rust speed matters most here.

```rust
// New in bizra-python/src/lib.rs

/// PyO3 wrapper for OmegaEngine — the unified constitutional gate
#[pyclass(name = "OmegaEngine")]
pub struct PyOmegaEngine {
    inner: OmegaEngine,
}

#[pymethods]
impl PyOmegaEngine {
    #[new]
    fn new(ihsan_threshold: f64, snr_threshold: f64) -> Self {
        Self { inner: OmegaEngine::new(ihsan_threshold, snr_threshold) }
    }

    /// Score a response against all constitutional dimensions.
    /// Returns dict with ihsan_score, snr_score, adl_gini, passes_gate.
    fn score(&self, py: Python, response: &str, context: &PyDict) -> PyResult<PyObject> {
        let result = self.inner.score(response, &extract_context(context)?);
        Ok(result_to_pydict(py, &result))
    }
}

/// PyO3 wrapper for SNREngine — Shannon entropy-based scoring
#[pyclass(name = "SNREngine")]
pub struct PySNREngine {
    inner: SNREngine,
}

#[pymethods]
impl PySNREngine {
    #[new]
    fn new(threshold: f64) -> Self {
        Self { inner: SNREngine::new(threshold) }
    }

    /// Compute SNR for text content.
    fn compute(&self, text: &str) -> f64 {
        self.inner.compute(text)
    }

    /// Batch SNR computation (SIMD-accelerated).
    fn compute_batch(&self, texts: Vec<String>) -> Vec<f64> {
        self.inner.compute_batch(&texts)
    }
}

/// PyO3 wrapper for AdlInvariant — justice/fairness gate
#[pyclass(name = "AdlInvariant")]
pub struct PyAdlInvariant {
    inner: AdlInvariant,
}

#[pymethods]
impl PyAdlInvariant {
    #[new]
    fn new(max_gini: f64) -> Self {
        Self { inner: AdlInvariant::new(max_gini) }
    }

    /// Check if resource distribution passes the Gini threshold.
    fn check(&self, allocations: Vec<f64>) -> bool {
        self.inner.check(&allocations)
    }

    /// Compute Gini coefficient.
    fn gini(&self, allocations: Vec<f64>) -> f64 {
        self.inner.gini(&allocations)
    }
}

/// PyO3 wrapper for IhsanProjector — multi-dimensional excellence scoring
#[pyclass(name = "IhsanProjector")]
pub struct PyIhsanProjector {
    inner: IhsanProjector,
}

#[pymethods]
impl PyIhsanProjector {
    #[new]
    fn new() -> Self {
        Self { inner: IhsanProjector::default() }
    }

    /// Project response across Ihsan dimensions.
    /// Returns dict of dimension -> score.
    fn project(&self, py: Python, text: &str) -> PyResult<PyObject> {
        let scores = self.inner.project(text);
        Ok(scores_to_pydict(py, &scores))
    }
}
```

### Phase 34b: PAT/Treasury (Priority 2)

Agent minting and resource allocation.

```rust
/// PyO3 wrapper for AgentMintingEngine
#[pyclass(name = "AgentMintingEngine")]
pub struct PyAgentMintingEngine {
    inner: AgentMintingEngine,
}

#[pymethods]
impl PyAgentMintingEngine {
    #[new]
    fn new(treasury: &PyTreasuryController) -> Self {
        Self { inner: AgentMintingEngine::new(treasury.inner.clone()) }
    }

    /// Mint a new PAT agent with given specialization.
    fn mint(&mut self, name: &str, specialization: &str) -> PyResult<String> {
        self.inner.mint(name, specialization)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

/// PyO3 wrapper for TreasuryController
#[pyclass(name = "TreasuryController")]
pub struct PyTreasuryController {
    inner: Arc<TreasuryController>,
}

#[pymethods]
impl PyTreasuryController {
    #[new]
    fn new(initial_budget: f64) -> Self {
        Self { inner: Arc::new(TreasuryController::new(initial_budget)) }
    }

    fn allocate(&self, agent_id: &str, amount: f64) -> PyResult<bool> {
        self.inner.allocate(agent_id, amount)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn balance(&self) -> f64 { self.inner.balance() }
}
```

### Phase 34c: Graph-of-Thoughts (Priority 3)

```rust
/// PyO3 wrapper for ThoughtGraph
#[pyclass(name = "ThoughtGraph")]
pub struct PyThoughtGraph {
    inner: ThoughtGraph,
}

#[pymethods]
impl PyThoughtGraph {
    #[new]
    fn new() -> Self {
        Self { inner: ThoughtGraph::new() }
    }

    fn add_thought(&mut self, content: &str, parent: Option<&str>) -> String {
        self.inner.add_thought(content, parent)
    }

    fn best_path(&self) -> Vec<String> {
        self.inner.best_path().into_iter().map(|t| t.id.to_string()).collect()
    }

    fn prune(&mut self, min_score: f64) -> usize {
        self.inner.prune(min_score)
    }
}
```

### Phase 34d: Federation (Priority 4)

```rust
/// PyO3 wrapper for ByzantineConsensus
#[pyclass(name = "ByzantineConsensus")]
pub struct PyByzantineConsensus {
    inner: ByzantineConsensus,
}

#[pymethods]
impl PyByzantineConsensus {
    #[new]
    fn new(node_count: usize, fault_tolerance: usize) -> Self {
        Self { inner: ByzantineConsensus::new(node_count, fault_tolerance) }
    }

    fn propose(&mut self, value: &[u8]) -> PyResult<()> {
        self.inner.propose(value)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn is_decided(&self) -> bool { self.inner.is_decided() }
}
```

---

## Python Integration Pattern

```python
# core/bridges/rust_bridge.py — add new bindings

FUNCTION load_rust_gates() -> Optional[RustGates]:
  """
  Attempt to import Rust-accelerated constitutional gates.
  Falls back gracefully to Python implementations.
  """
  TRY:
    FROM bizra_python IMPORT (
      OmegaEngine, SNREngine, AdlInvariant, IhsanProjector
    )
    RETURN RustGates(
      omega=OmegaEngine(ihsan=0.95, snr=0.85),
      snr=SNREngine(threshold=0.85),
      adl=AdlInvariant(max_gini=0.40),
      ihsan=IhsanProjector(),
    )
  EXCEPT ImportError:
    RETURN None  # Fall back to Python implementations
```

---

## TDD Anchors

```
TEST test_pyo3_omega_engine_scores_response:
  engine = OmegaEngine(ihsan_threshold=0.95, snr_threshold=0.85)
  result = engine.score("well-reasoned response", {})
  ASSERT "ihsan_score" IN result
  ASSERT "snr_score" IN result
  ASSERT "passes_gate" IN result

TEST test_pyo3_snr_batch_matches_single:
  engine = SNREngine(threshold=0.85)
  texts = ["text1", "text2", "text3"]
  batch = engine.compute_batch(texts)
  singles = [engine.compute(t) for t in texts]
  ASSERT batch == singles  # SIMD batch must match sequential

TEST test_pyo3_adl_gini_gate:
  adl = AdlInvariant(max_gini=0.40)
  ASSERT adl.check([10, 10, 10, 10]) IS True    # Equal = Gini 0
  ASSERT adl.check([100, 0, 0, 0]) IS False     # Monopoly = Gini ~0.75

TEST test_pyo3_agent_minting:
  treasury = TreasuryController(initial_budget=1000.0)
  minter = AgentMintingEngine(treasury)
  agent_id = minter.mint("researcher", "nlp")
  ASSERT agent_id IS NOT None

TEST test_pyo3_thought_graph_best_path:
  graph = ThoughtGraph()
  t1 = graph.add_thought("premise", None)
  t2 = graph.add_thought("reasoning", t1)
  t3 = graph.add_thought("conclusion", t2)
  path = graph.best_path()
  ASSERT len(path) == 3

TEST test_rust_python_snr_parity:
  # Rust SNR must match Python SNR within epsilon
  rust_score = PySNREngine(0.85).compute(sample_text)
  python_score = python_snr_compute(sample_text)
  ASSERT abs(rust_score - python_score) < 0.001

TEST test_fallback_when_rust_unavailable:
  # When bizra_python not installed, Python gates activate
  gates = load_rust_gates()
  # In test env without maturin build, this returns None
  IF gates IS None:
    python_gates = PythonGates()
    ASSERT python_gates.snr.compute("text") > 0
```

## Success Criteria

| Metric | Target |
|--------|--------|
| New PyO3 classes | +9 (Omega, SNR, Adl, Ihsan, Treasury, Minter, ThoughtGraph, ByzConsensus, Zakat) |
| Behavioral parity | Rust output matches Python within epsilon for all shared operations |
| Performance | Constitutional gate scoring < 100us (vs ~5ms Python) |
| Graceful fallback | Python runtime works identically when Rust bindings unavailable |
| Test count | +7 Python integration tests, +9 Rust unit tests |
