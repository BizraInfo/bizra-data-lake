//! BIZRA Python Bindings — PyO3 Bridge
//!
//! Exposes Rust bizra-core to Python for 10-100x performance boost.
//! Giants: PyO3 team, Rust-Python interop pioneers

mod urp_bridge;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use bizra_autopoiesis::{
    pattern_memory::PatternMemory,
    preference_tracker::{PreferenceTracker, PreferenceType},
};
use bizra_core::{
    domain_separated_digest as rust_domain_digest, Constitution as RustConstitution,
    NodeId as RustNodeId, NodeIdentity as RustNodeIdentity, PCIEnvelope as RustPCIEnvelope,
    IHSAN_THRESHOLD, SNR_THRESHOLD,
};

/// Python wrapper for NodeId
#[pyclass(name = "NodeId")]
#[derive(Clone)]
pub struct PyNodeId {
    inner: RustNodeId,
}

#[pymethods]
impl PyNodeId {
    #[new]
    fn new(id: String) -> PyResult<Self> {
        if id.len() != 32 {
            return Err(PyValueError::new_err("NodeId must be 32 hex characters"));
        }
        Ok(Self {
            inner: RustNodeId(id),
        })
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __repr__(&self) -> String {
        format!("NodeId('{}')", self.inner.0)
    }

    #[getter]
    fn id(&self) -> String {
        self.inner.0.clone()
    }
}

/// Python wrapper for NodeIdentity
#[pyclass(name = "NodeIdentity")]
pub struct PyNodeIdentity {
    inner: RustNodeIdentity,
}

#[pymethods]
impl PyNodeIdentity {
    /// Generate a new random identity
    #[new]
    fn new() -> Self {
        Self {
            inner: RustNodeIdentity::generate(),
        }
    }

    /// Create from secret bytes (32 bytes)
    #[staticmethod]
    fn from_secret(secret: &[u8]) -> PyResult<Self> {
        if secret.len() != 32 {
            return Err(PyValueError::new_err("Secret must be 32 bytes"));
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(secret);
        Ok(Self {
            inner: RustNodeIdentity::from_secret_bytes(&arr),
        })
    }

    /// Get the node ID
    #[getter]
    fn node_id(&self) -> PyNodeId {
        PyNodeId {
            inner: self.inner.node_id().clone(),
        }
    }

    /// Get public key as hex string
    #[getter]
    fn public_key(&self) -> String {
        self.inner.public_key_hex()
    }

    /// Get secret bytes (handle with care!)
    fn secret_bytes(&self) -> Vec<u8> {
        self.inner.secret_bytes().to_vec()
    }

    /// Sign a message with domain separation
    fn sign(&self, message: &[u8]) -> String {
        self.inner.sign(message)
    }

    /// Verify a signature
    #[staticmethod]
    fn verify(message: &[u8], signature: &str, public_key: &str) -> bool {
        RustNodeIdentity::verify_with_hex(message, signature, public_key)
    }

    fn __repr__(&self) -> String {
        format!("NodeIdentity(node_id='{}')", self.inner.node_id())
    }
}

/// Python wrapper for Constitution
#[pyclass(name = "Constitution")]
#[derive(Clone)]
pub struct PyConstitution {
    inner: RustConstitution,
}

#[pymethods]
impl PyConstitution {
    /// Create default constitution
    #[new]
    fn new() -> Self {
        Self {
            inner: RustConstitution::default(),
        }
    }

    /// Check if score meets Ihsan threshold
    fn check_ihsan(&self, score: f64) -> bool {
        self.inner.check_ihsan(score)
    }

    /// Check if SNR meets threshold
    fn check_snr(&self, snr: f64) -> bool {
        self.inner.check_snr(snr)
    }

    /// Get Ihsan threshold
    #[getter]
    fn ihsan_threshold(&self) -> f64 {
        self.inner.ihsan.minimum
    }

    /// Get SNR threshold
    #[getter]
    fn snr_threshold(&self) -> f64 {
        self.inner.snr_threshold
    }

    /// Get version
    #[getter]
    fn version(&self) -> String {
        self.inner.version.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Constitution(version='{}', ihsan={}, snr={})",
            self.inner.version, self.inner.ihsan.minimum, self.inner.snr_threshold
        )
    }
}

/// Python wrapper for PCI Envelope
#[pyclass(name = "PCIEnvelope")]
pub struct PyPCIEnvelope {
    id: String,
    sender: PyNodeId,
    content_hash: String,
    signature: String,
    public_key: String,
    payload_json: String,
    ttl: u64,
}

#[pymethods]
impl PyPCIEnvelope {
    /// Create a new PCI envelope
    #[staticmethod]
    fn create(
        identity: &PyNodeIdentity,
        payload: &str, // JSON string
        ttl: u64,
        provenance: Vec<String>,
    ) -> PyResult<Self> {
        // Parse payload as JSON value
        let payload_value: serde_json::Value = serde_json::from_str(payload)
            .map_err(|e| PyValueError::new_err(format!("Invalid JSON: {e}")))?;

        let envelope = RustPCIEnvelope::create(&identity.inner, payload_value, ttl, provenance)
            .map_err(|e| PyRuntimeError::new_err(format!("PCI error: {e}")))?;

        Ok(Self {
            id: envelope.id,
            sender: PyNodeId {
                inner: envelope.sender,
            },
            content_hash: envelope.content_hash,
            signature: envelope.signature,
            public_key: envelope.public_key,
            payload_json: serde_json::to_string(&envelope.payload).map_err(|e| {
                PyRuntimeError::new_err(format!("Payload serialization failed: {e}"))
            })?,
            ttl: envelope.ttl,
        })
    }

    #[getter]
    fn id(&self) -> String {
        self.id.clone()
    }

    #[getter]
    fn sender(&self) -> PyNodeId {
        self.sender.clone()
    }

    #[getter]
    fn content_hash(&self) -> String {
        self.content_hash.clone()
    }

    #[getter]
    fn signature(&self) -> String {
        self.signature.clone()
    }

    #[getter]
    fn public_key(&self) -> String {
        self.public_key.clone()
    }

    #[getter]
    fn payload(&self) -> String {
        self.payload_json.clone()
    }

    #[getter]
    fn ttl(&self) -> u64 {
        self.ttl
    }

    fn __repr__(&self) -> String {
        format!(
            "PCIEnvelope(id='{}', sender={})",
            self.id,
            self.sender.__str__()
        )
    }
}

/// Compute domain-separated BLAKE3 digest
#[pyfunction]
fn domain_separated_digest(message: &[u8]) -> String {
    rust_domain_digest(message)
}

/// Get Ihsan threshold constant
#[pyfunction]
fn get_ihsan_threshold() -> f64 {
    IHSAN_THRESHOLD
}

/// Get SNR threshold constant
#[pyfunction]
fn get_snr_threshold() -> f64 {
    SNR_THRESHOLD
}

/// Task complexity estimation
#[pyclass(name = "TaskComplexity")]
#[derive(Clone)]
pub struct PyTaskComplexity {
    level: String,
}

#[pymethods]
impl PyTaskComplexity {
    /// Estimate complexity from prompt and max_tokens
    #[staticmethod]
    fn estimate(prompt: &str, max_tokens: usize) -> Self {
        use bizra_inference::selector::TaskComplexity;
        let complexity = TaskComplexity::estimate(prompt, max_tokens);
        Self {
            level: format!("{complexity:?}"),
        }
    }

    #[getter]
    fn level(&self) -> String {
        self.level.clone()
    }

    fn __repr__(&self) -> String {
        format!("TaskComplexity(level='{}')", self.level)
    }
}

/// Model tier for inference
#[pyclass(name = "ModelTier")]
#[derive(Clone)]
pub struct PyModelTier {
    tier: String,
}

#[pymethods]
impl PyModelTier {
    #[new]
    fn new(tier: &str) -> PyResult<Self> {
        match tier.to_lowercase().as_str() {
            "edge" | "local" | "pool" => Ok(Self {
                tier: tier.to_lowercase(),
            }),
            _ => Err(PyValueError::new_err(
                "Tier must be 'edge', 'local', or 'pool'",
            )),
        }
    }

    #[getter]
    fn name(&self) -> String {
        self.tier.clone()
    }

    fn __repr__(&self) -> String {
        format!("ModelTier('{}')", self.tier)
    }
}

/// Model selector for tier selection
#[pyclass(name = "ModelSelector")]
pub struct PyModelSelector;

#[pymethods]
impl PyModelSelector {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Select tier based on complexity
    fn select_tier(&self, complexity: &PyTaskComplexity) -> PyModelTier {
        let tier = match complexity.level.as_str() {
            "Simple" | "Medium" => "edge",
            "Complex" => "local",
            "Expert" => "pool",
            _ => "local",
        };
        PyModelTier { tier: tier.into() }
    }
}

/// Gate chain for content validation
#[pyclass(name = "GateChain")]
pub struct PyGateChain;

#[pymethods]
impl PyGateChain {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Verify content through gate chain
    fn verify(
        &self,
        content: &[u8],
        snr_score: Option<f64>,
        ihsan_score: Option<f64>,
    ) -> PyResult<Vec<(String, bool, String)>> {
        use bizra_core::pci::gates::{default_gate_chain, GateContext};

        let chain = default_gate_chain();
        let constitution = RustConstitution::default();

        let ctx = GateContext {
            sender_id: "python_client".into(),
            envelope_id: "py_envelope".into(),
            content: content.to_vec(),
            constitution,
            snr_score,
            ihsan_score,
        };

        let results = chain.verify(&ctx);

        Ok(results
            .iter()
            .map(|r| (r.gate.clone(), r.passed, format!("{:?}", r.code)))
            .collect())
    }

    /// Check if all gates passed
    #[staticmethod]
    fn all_passed(results: Vec<(String, bool, String)>) -> bool {
        results.iter().all(|(_, passed, _)| *passed)
    }
}

// =============================================================================
// Inference Gateway Bindings (Python↔Rust unified inference path)
// =============================================================================

/// Python wrapper for InferenceResponse
#[pyclass(name = "InferenceResponse")]
#[derive(Clone)]
pub struct PyInferenceResponse {
    #[pyo3(get)]
    request_id: String,
    #[pyo3(get)]
    text: String,
    #[pyo3(get)]
    model: String,
    #[pyo3(get)]
    tier: String,
    #[pyo3(get)]
    completion_tokens: usize,
    #[pyo3(get)]
    duration_ms: u64,
    #[pyo3(get)]
    tokens_per_second: f32,
}

#[pymethods]
impl PyInferenceResponse {
    fn __repr__(&self) -> String {
        format!(
            "InferenceResponse(model='{}', tier='{}', tokens={}, {:.1} tok/s)",
            self.model, self.tier, self.completion_tokens, self.tokens_per_second
        )
    }
}

impl From<bizra_inference::InferenceResponse> for PyInferenceResponse {
    fn from(r: bizra_inference::InferenceResponse) -> Self {
        Self {
            request_id: r.request_id,
            text: r.text,
            model: r.model,
            tier: format!("{:?}", r.tier),
            completion_tokens: r.completion_tokens,
            duration_ms: r.duration_ms,
            tokens_per_second: r.tokens_per_second,
        }
    }
}

/// Python wrapper for InferenceGateway — the unified Python↔Rust inference path.
///
/// This bridges the gap: Python code can now call Rust's SIMD-accelerated
/// inference gateway with constitutional gate enforcement.
#[pyclass(name = "InferenceGateway")]
pub struct PyInferenceGateway {
    gateway: std::sync::Arc<tokio::sync::Mutex<bizra_inference::InferenceGateway>>,
    runtime: std::sync::Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl PyInferenceGateway {
    /// Create a new InferenceGateway with identity and constitution.
    ///
    /// The gateway starts with no backends. Register backends with
    /// `register_ollama()` or `register_lmstudio()` before calling `infer()`.
    #[new]
    fn new(identity: &PyNodeIdentity, constitution: &PyConstitution) -> PyResult<Self> {
        // NodeIdentity doesn't implement Clone (Ed25519 secret key security).
        // Reconstruct from secret bytes, matching the pattern in bizra-api/src/main.rs.
        let secret = identity.inner.secret_bytes();
        let gateway_identity = RustNodeIdentity::from_secret_bytes(&secret);
        let gateway =
            bizra_inference::InferenceGateway::new(gateway_identity, constitution.inner.clone());
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("Tokio runtime error: {e}")))?;
        Ok(Self {
            gateway: std::sync::Arc::new(tokio::sync::Mutex::new(gateway)),
            runtime: std::sync::Arc::new(runtime),
        })
    }

    /// Register an Ollama backend at the given URL and tier.
    ///
    /// Args:
    ///     model: Model name (e.g. "llama3.2", "qwen2.5:7b")
    ///     tier: "edge", "local", or "pool"
    ///     base_url: Ollama URL (default: "http://localhost:11434")
    fn register_ollama(&self, model: &str, tier: &str, base_url: Option<&str>) -> PyResult<()> {
        let model_tier = parse_tier(tier)?;
        let config = bizra_inference::BackendConfig {
            name: format!("ollama-{tier}"),
            model: model.to_string(),
            context_length: 4096,
            gpu_layers: -1,
        };
        let backend = std::sync::Arc::new(bizra_inference::backends::ollama::OllamaBackend::new(
            config, base_url,
        ));
        let gw = self.gateway.clone();
        self.runtime.block_on(async move {
            gw.lock().await.register_backend(model_tier, backend).await;
        });
        Ok(())
    }

    /// Register an LM Studio backend with default configuration.
    ///
    /// Args:
    ///     tier: "edge", "local", or "pool"
    ///     host: LM Studio host (default: env LMSTUDIO_HOST or WSL gateway)
    ///     port: LM Studio port (default: env LMSTUDIO_PORT or 1234)
    fn register_lmstudio(&self, tier: &str, host: Option<&str>, port: Option<u16>) -> PyResult<()> {
        let model_tier = parse_tier(tier)?;
        let mut lms_config = bizra_inference::LMStudioConfig::default();
        if let Some(h) = host {
            lms_config.host = h.to_string();
        }
        if let Some(p) = port {
            lms_config.port = p;
        }
        let backend = std::sync::Arc::new(bizra_inference::LMStudioBackend::new(lms_config));
        let gw = self.gateway.clone();
        self.runtime.block_on(async move {
            gw.lock().await.register_backend(model_tier, backend).await;
        });
        Ok(())
    }

    /// Run inference through the Rust gateway with constitutional gate enforcement.
    ///
    /// This is the critical path: Python → Rust gateway → backend → response.
    ///
    /// Args:
    ///     prompt: The input prompt
    ///     system: Optional system message
    ///     max_tokens: Maximum tokens to generate (default: 1024)
    ///     temperature: Sampling temperature (default: 0.7)
    ///     tier: Optional preferred tier ("edge", "local", "pool")
    ///
    /// Returns:
    ///     InferenceResponse with text, model, tier, timing, and token metrics.
    ///
    /// Raises:
    ///     RuntimeError: If no backend is registered for the selected tier,
    ///                   the backend fails, or the request times out.
    #[pyo3(signature = (prompt, system=None, max_tokens=1024, temperature=0.7, tier=None))]
    fn infer(
        &self,
        prompt: &str,
        system: Option<&str>,
        max_tokens: usize,
        temperature: f32,
        tier: Option<&str>,
    ) -> PyResult<PyInferenceResponse> {
        let preferred_tier = tier.map(parse_tier).transpose()?;
        let complexity = bizra_inference::TaskComplexity::estimate(prompt, max_tokens);

        let request = bizra_inference::InferenceRequest {
            id: uuid::Uuid::new_v4().to_string(),
            prompt: prompt.to_string(),
            system: system.map(|s| s.to_string()),
            max_tokens,
            temperature,
            complexity,
            preferred_tier,
        };

        let gw = self.gateway.clone();
        let result = self
            .runtime
            .block_on(async move { gw.lock().await.infer(request).await });

        match result {
            Ok(response) => Ok(PyInferenceResponse::from(response)),
            Err(e) => Err(PyRuntimeError::new_err(format!("Inference error: {e}"))),
        }
    }

    /// Check health of a specific backend tier.
    ///
    /// Returns True if the backend for the given tier is reachable.
    fn health_check(&self, tier: &str) -> PyResult<bool> {
        // Health check goes through the backend directly, not the gateway.
        // For now, we attempt a minimal infer and check for connection errors.
        let preferred_tier = parse_tier(tier)?;
        let request = bizra_inference::InferenceRequest {
            id: "health_check".to_string(),
            prompt: "ping".to_string(),
            system: None,
            max_tokens: 1,
            temperature: 0.0,
            complexity: bizra_inference::TaskComplexity::Simple,
            preferred_tier: Some(preferred_tier),
        };
        let gw = self.gateway.clone();
        let result = self
            .runtime
            .block_on(async move { gw.lock().await.infer(request).await });
        Ok(result.is_ok())
    }

    fn __repr__(&self) -> String {
        "InferenceGateway(rust_native=True)".to_string()
    }
}

/// Parse a tier string into ModelTier. Shared by gateway methods.
fn parse_tier(tier: &str) -> PyResult<bizra_inference::ModelTier> {
    match tier.to_lowercase().as_str() {
        "edge" | "nano" => Ok(bizra_inference::ModelTier::Edge),
        "local" | "medium" => Ok(bizra_inference::ModelTier::Local),
        "pool" | "large" => Ok(bizra_inference::ModelTier::Pool),
        _ => Err(PyValueError::new_err(
            "Tier must be 'edge', 'local', or 'pool'",
        )),
    }
}

// =============================================================================
// SNR Engine Bindings (Rust-native signal quality measurement)
// =============================================================================

/// Python wrapper for SNREngine — Shannon-inspired signal quality measurement.
///
/// Exposes the Rust SNR engine's weighted geometric mean computation to Python,
/// completing the Rust→Python SNR bridge (Gap G-2).
///
/// Standing on Giants: Shannon (information theory, 1948) · Gerganov (SIMD optimization)
#[pyclass(name = "SNREngine")]
pub struct PySNREngine {
    inner: bizra_core::SNREngine,
}

#[pymethods]
impl PySNREngine {
    /// Create a new SNR engine with floor and target thresholds.
    ///
    /// Args:
    ///     snr_floor: Minimum acceptable SNR (default: 0.85)
    ///     ihsan_target: Ihsan excellence target (default: 0.95)
    #[new]
    #[pyo3(signature = (snr_floor=0.85, ihsan_target=0.95))]
    fn new(snr_floor: f64, ihsan_target: f64) -> Self {
        Self {
            inner: bizra_core::SNREngine::new(snr_floor, ihsan_target),
        }
    }

    /// Create with full configuration.
    ///
    /// Args:
    ///     snr_floor: Minimum acceptable SNR
    ///     ihsan_target: Ihsan excellence target
    ///     weight_signal: Signal strength weight (default: 0.30)
    ///     weight_diversity: Diversity weight (default: 0.25)
    ///     weight_grounding: Grounding weight (default: 0.25)
    ///     weight_balance: Balance weight (default: 0.20)
    #[staticmethod]
    #[pyo3(signature = (snr_floor=0.85, ihsan_target=0.95, weight_signal=0.30, weight_diversity=0.25, weight_grounding=0.25, weight_balance=0.20))]
    fn with_config(
        snr_floor: f64,
        ihsan_target: f64,
        weight_signal: f64,
        weight_diversity: f64,
        weight_grounding: f64,
        weight_balance: f64,
    ) -> Self {
        let config = bizra_core::SNRConfig {
            snr_floor,
            ihsan_target,
            weight_signal,
            weight_diversity,
            weight_grounding,
            weight_balance,
            ..Default::default()
        };
        Self {
            inner: bizra_core::SNREngine::with_config(config),
        }
    }

    /// Analyze text and return signal metrics as a dict.
    ///
    /// Returns dict with: snr, signal_strength, noise_level, diversity,
    /// grounding, balance, input_size, word_count, unique_words, analysis_duration_us
    ///
    /// Raises:
    ///     ValueError: If text is empty or exceeds 1MB
    fn analyze_text(&self, text: &str) -> PyResult<pyo3::PyObject> {
        let metrics = self
            .inner
            .analyze_text(text)
            .map_err(|e| PyValueError::new_err(format!("SNR analysis error: {e}")))?;
        Ok(signal_metrics_to_pyobject(&metrics))
    }

    /// Analyze and validate against SNR floor.
    ///
    /// Returns metrics dict if SNR >= floor, raises ValueError otherwise.
    fn validate(&self, text: &str) -> PyResult<pyo3::PyObject> {
        let metrics = self
            .inner
            .validate(text)
            .map_err(|e| PyValueError::new_err(format!("SNR validation failed: {e}")))?;
        Ok(signal_metrics_to_pyobject(&metrics))
    }

    /// Analyze and validate against Ihsan target.
    ///
    /// Returns metrics dict if SNR >= ihsan_target, raises ValueError otherwise.
    fn validate_ihsan(&self, text: &str) -> PyResult<pyo3::PyObject> {
        let metrics = self
            .inner
            .validate_ihsan(text)
            .map_err(|e| PyValueError::new_err(format!("Ihsan validation failed: {e}")))?;
        Ok(signal_metrics_to_pyobject(&metrics))
    }

    /// Get rolling average SNR across the history window.
    fn average_snr(&self) -> f64 {
        self.inner.average_snr()
    }

    /// Get engine statistics as a dict.
    ///
    /// Returns dict with: total_measurements, history_size, average_snr, snr_floor, ihsan_target
    fn stats(&self) -> pyo3::PyObject {
        let s = self.inner.stats();
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            let _ = dict.set_item("total_measurements", s.total_measurements);
            let _ = dict.set_item("history_size", s.history_size);
            let _ = dict.set_item("average_snr", s.average_snr);
            let _ = dict.set_item("snr_floor", s.snr_floor);
            let _ = dict.set_item("ihsan_target", s.ihsan_target);
            dict.into()
        })
    }

    fn __repr__(&self) -> String {
        let s = self.inner.stats();
        format!(
            "SNREngine(measurements={}, avg_snr={:.4}, floor={}, target={})",
            s.total_measurements, s.average_snr, s.snr_floor, s.ihsan_target
        )
    }
}

/// Convert SignalMetrics to a Python dict.
fn signal_metrics_to_pyobject(metrics: &bizra_core::SignalMetrics) -> pyo3::PyObject {
    Python::with_gil(|py| {
        let dict = pyo3::types::PyDict::new(py);
        let _ = dict.set_item("snr", metrics.snr);
        let _ = dict.set_item("signal_strength", metrics.signal_strength);
        let _ = dict.set_item("noise_level", metrics.noise_level);
        let _ = dict.set_item("diversity", metrics.diversity);
        let _ = dict.set_item("grounding", metrics.grounding);
        let _ = dict.set_item("balance", metrics.balance);
        let _ = dict.set_item("input_size", metrics.input_size);
        let _ = dict.set_item("word_count", metrics.word_count);
        let _ = dict.set_item("unique_words", metrics.unique_words);
        let _ = dict.set_item("analysis_duration_us", metrics.analysis_duration_us);
        dict.into()
    })
}

// =============================================================================
// Autopoiesis Bindings (10-100x faster pattern learning)
// =============================================================================

/// Python wrapper for PatternMemory (autopoiesis)
#[pyclass(name = "PatternMemory")]
pub struct PyPatternMemory {
    inner: PatternMemory,
}

#[pymethods]
impl PyPatternMemory {
    /// Create a new in-memory pattern store for a node
    #[new]
    fn new(node_id: &str) -> Self {
        let nid = RustNodeId(node_id.to_string());
        Self {
            inner: PatternMemory::in_memory(nid),
        }
    }

    /// Learn a new pattern from content, embedding, and tags
    ///
    /// Returns the pattern ID on success.
    fn learn(&mut self, content: &str, embedding: Vec<f32>, tags: Vec<String>) -> PyResult<String> {
        self.inner
            .learn(content.to_string(), embedding, tags)
            .map_err(|e| PyRuntimeError::new_err(format!("Pattern learn error: {e}")))
    }

    /// Recall patterns similar to the given embedding
    ///
    /// Returns list of (content, confidence, tags) tuples.
    fn recall(&self, embedding: Vec<f32>, limit: usize) -> Vec<(String, f64, Vec<String>)> {
        self.inner
            .recall(&embedding, limit)
            .into_iter()
            .map(|p| (p.content.clone(), p.confidence, p.tags.clone()))
            .collect()
    }

    /// Get the number of stored patterns
    fn pattern_count(&self) -> usize {
        self.inner.count()
    }

    fn __repr__(&self) -> String {
        format!("PatternMemory(count={})", self.inner.count())
    }
}

/// Python wrapper for PreferenceTracker (autopoiesis)
#[pyclass(name = "PreferenceTracker")]
pub struct PyPreferenceTracker {
    inner: PreferenceTracker,
}

#[pymethods]
impl PyPreferenceTracker {
    #[new]
    fn new() -> Self {
        Self {
            inner: PreferenceTracker::new(),
        }
    }

    /// Observe a user preference (pref_type, key, value)
    ///
    /// pref_type: "style", "verbosity", "code_style", "language", or custom string
    fn observe(&mut self, pref_type: &str, key: &str, value: &str) {
        let pt = match pref_type.to_lowercase().as_str() {
            "style" => PreferenceType::Style,
            "verbosity" => PreferenceType::Verbosity,
            "code_style" => PreferenceType::CodeStyle,
            "language" => PreferenceType::Language,
            other => PreferenceType::Custom(other.to_string()),
        };
        self.inner.observe(pt, key, value);
    }

    /// Get the current value for a preference (returns None if below confidence threshold)
    fn get_strength(&self, pref_type: &str, key: &str) -> Option<String> {
        let pt = match pref_type.to_lowercase().as_str() {
            "style" => PreferenceType::Style,
            "verbosity" => PreferenceType::Verbosity,
            "code_style" => PreferenceType::CodeStyle,
            "language" => PreferenceType::Language,
            other => PreferenceType::Custom(other.to_string()),
        };
        self.inner.get(&pt, key).map(|s| s.to_string())
    }

    /// Apply learned preferences to a prompt
    fn apply_to_prompt(&self, prompt: &str) -> String {
        self.inner.apply_to_prompt(prompt)
    }

    fn __repr__(&self) -> String {
        "PreferenceTracker()".to_string()
    }
}

// =============================================================================
// Sovereign Experience Ledger Bindings (Episodic Memory)
// =============================================================================

/// Python wrapper for ExperienceLedger
#[pyclass(name = "ExperienceLedger")]
pub struct PyExperienceLedger {
    inner: bizra_core::ExperienceLedger,
}

#[pymethods]
impl PyExperienceLedger {
    /// Create a new experience ledger
    #[new]
    fn new() -> Self {
        Self {
            inner: bizra_core::ExperienceLedger::new(),
        }
    }

    /// Commit an episode to the ledger.
    ///
    /// Args:
    ///     context: The query or trigger text
    ///     graph_hash: BLAKE3 hash of the GoT artifact
    ///     graph_node_count: Number of thoughts in the graph
    ///     actions: List of (action_type, description, success, duration_us) tuples
    ///     snr_score: Signal-to-Noise Ratio score (0.0-1.0)
    ///     ihsan_score: Ihsan excellence score (0.0-1.0)
    ///     snr_ok: Whether the SNR gate passed
    ///     context_embedding: Optional embedding vector for semantic retrieval
    ///     response_summary: Optional truncated response text
    ///
    /// Returns:
    ///     The episode content-address hash (hex string)
    #[pyo3(signature = (context, graph_hash, graph_node_count, actions, snr_score, ihsan_score, snr_ok, context_embedding=None, response_summary=None))]
    #[allow(clippy::too_many_arguments)]
    fn commit(
        &mut self,
        context: &str,
        graph_hash: &str,
        graph_node_count: usize,
        actions: Vec<(String, String, bool, u64)>,
        snr_score: f64,
        ihsan_score: f64,
        snr_ok: bool,
        context_embedding: Option<Vec<f32>>,
        response_summary: Option<String>,
    ) -> String {
        let episode_actions: Vec<bizra_core::EpisodeAction> = actions
            .into_iter()
            .map(|(at, desc, ok, dur)| bizra_core::EpisodeAction {
                action_type: at,
                description: desc,
                success: ok,
                duration_us: dur,
            })
            .collect();

        let impact = bizra_core::EpisodeImpact {
            snr_score,
            ihsan_score,
            snr_ok,
            user_feedback: None,
            tokens_used: 0,
            efficiency_score: 0.0,
        };

        self.inner.commit(
            context.to_string(),
            graph_hash.to_string(),
            graph_node_count,
            episode_actions,
            impact,
            context_embedding,
            response_summary,
        )
    }

    /// Retrieve top-K episodes using RIR algorithm.
    ///
    /// Args:
    ///     query_text: The query to match against
    ///     top_k: Maximum number of episodes to return
    ///     query_embedding: Optional embedding for semantic matching
    ///
    /// Returns:
    ///     List of dicts with episode fields
    #[pyo3(signature = (query_text, top_k, query_embedding=None))]
    fn retrieve(
        &self,
        query_text: &str,
        top_k: usize,
        query_embedding: Option<Vec<f32>>,
    ) -> Vec<pyo3::PyObject> {
        let emb_ref = query_embedding.as_deref();
        let episodes = self.inner.retrieve(query_text, emb_ref, top_k);

        Python::with_gil(|py| {
            episodes
                .into_iter()
                .map(|ep| {
                    let dict = pyo3::types::PyDict::new(py);
                    let _ = dict.set_item("sequence", ep.sequence);
                    let _ = dict.set_item("timestamp_secs", ep.timestamp_secs);
                    let _ = dict.set_item("context", &ep.context);
                    let _ = dict.set_item("graph_hash", &ep.graph_hash);
                    let _ = dict.set_item("graph_node_count", ep.graph_node_count);
                    let _ = dict.set_item("snr_score", ep.impact.snr_score);
                    let _ = dict.set_item("ihsan_score", ep.impact.ihsan_score);
                    let _ = dict.set_item("snr_ok", ep.impact.snr_ok);
                    let _ = dict.set_item("episode_hash", &ep.episode_hash);
                    let _ = dict.set_item("chain_hash", &ep.chain_hash);
                    if let Some(ref summary) = ep.response_summary {
                        let _ = dict.set_item("response_summary", summary);
                    }
                    dict.into()
                })
                .collect()
        })
    }

    /// Verify the entire chain integrity.
    fn verify_chain_integrity(&self) -> bool {
        self.inner.verify_chain_integrity()
    }

    /// Get the current chain head hash.
    #[getter]
    fn chain_head(&self) -> String {
        self.inner.chain_head().to_string()
    }

    /// Get the number of episodes.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Get the next sequence number.
    #[getter]
    fn next_sequence(&self) -> u64 {
        self.inner.next_sequence()
    }

    /// Get the distillation count.
    #[getter]
    fn distillation_count(&self) -> u64 {
        self.inner.distillation_count()
    }

    fn __repr__(&self) -> String {
        format!(
            "ExperienceLedger(episodes={}, chain_head='{}')",
            self.inner.len(),
            &self.inner.chain_head()[..8.min(self.inner.chain_head().len())]
        )
    }
}

// =============================================================================
// Cognitive Layer Bindings — Memory Synthesis (bizra-memory)
// =============================================================================

/// Python wrapper for BizraMemory — the soul of "My AI Knows Me".
///
/// Transforms conversations into understanding through a synthesis pipeline:
/// Fragment → Atom → Insight → Profile.
///
/// Standing on Giants: Maturana (autopoiesis) · Shannon (information density)
#[pyclass(name = "BizraMemory")]
pub struct PyBizraMemory {
    inner: bizra_memory::BizraMemory,
}

#[pymethods]
impl PyBizraMemory {
    /// Create a new memory synthesis system.
    #[new]
    fn new() -> Self {
        Self {
            inner: bizra_memory::BizraMemory::new(),
        }
    }

    /// Process a user message through the full pipeline.
    ///
    /// Returns dict with: ingested, atoms_extracted, insights_produced, synthesis_triggered.
    fn process_user_turn(
        &mut self,
        content: &str,
        session_id: u64,
        turn: u32,
        timestamp: u64,
    ) -> pyo3::PyObject {
        let result = self
            .inner
            .process_user_turn(content, session_id, turn, timestamp);
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            let _ = dict.set_item("ingested", result.ingested);
            let _ = dict.set_item("atoms_extracted", result.atoms_extracted);
            let _ = dict.set_item("insights_produced", result.insights_produced);
            let _ = dict.set_item("synthesis_triggered", result.synthesis_triggered);
            dict.into()
        })
    }

    /// Process an assistant message (lower priority, context enrichment).
    fn process_assistant_turn(
        &mut self,
        content: &str,
        session_id: u64,
        turn: u32,
        timestamp: u64,
    ) -> pyo3::PyObject {
        let result = self
            .inner
            .process_assistant_turn(content, session_id, turn, timestamp);
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            let _ = dict.set_item("ingested", result.ingested);
            let _ = dict.set_item("atoms_extracted", result.atoms_extracted);
            let _ = dict.set_item("insights_produced", result.insights_produced);
            let _ = dict.set_item("synthesis_triggered", result.synthesis_triggered);
            dict.into()
        })
    }

    /// "What do I know?" — all reliable facts with confidence scores.
    fn what_do_i_know(&mut self, now: u64) -> Vec<(String, f32)> {
        self.inner
            .what_do_i_know(now)
            .into_iter()
            .map(|(s, c)| (s.to_string(), c))
            .collect()
    }

    /// User preferences with confidence scores.
    fn user_preferences(&mut self, now: u64) -> Vec<(String, f32)> {
        self.inner
            .user_preferences(now)
            .into_iter()
            .map(|(s, c)| (s.to_string(), c))
            .collect()
    }

    /// Active user goals with confidence scores.
    fn user_goals(&mut self, now: u64) -> Vec<(String, f32)> {
        self.inner
            .user_goals(now)
            .into_iter()
            .map(|(s, c)| (s.to_string(), c))
            .collect()
    }

    /// User boundaries and negations with confidence scores.
    fn user_boundaries(&mut self, now: u64) -> Vec<(String, f32)> {
        self.inner
            .user_boundaries(now)
            .into_iter()
            .map(|(s, c)| (s.to_string(), c))
            .collect()
    }

    /// Observed behavioral patterns with confidence scores.
    fn user_patterns(&mut self, now: u64) -> Vec<(String, f32)> {
        self.inner
            .user_patterns(now)
            .into_iter()
            .map(|(s, c)| (s.to_string(), c))
            .collect()
    }

    /// User principles and values with confidence scores.
    fn user_principles(&mut self, now: u64) -> Vec<(String, f32)> {
        self.inner
            .user_principles(now)
            .into_iter()
            .map(|(s, c)| (s.to_string(), c))
            .collect()
    }

    /// Synthesized insights — connected understanding.
    fn insights(&mut self) -> Vec<(String, f32)> {
        self.inner
            .insights()
            .into_iter()
            .map(|(s, c)| (s.to_string(), c))
            .collect()
    }

    /// Force a synthesis pass (regardless of batch threshold).
    fn force_synthesis(&mut self, now: u64) {
        self.inner.force_synthesis(now);
    }

    /// Activate the memory system.
    fn activate(&mut self) {
        self.inner.activate();
    }

    /// Deactivate (pause processing).
    fn deactivate(&mut self) {
        self.inner.deactivate();
    }

    /// Is the system active?
    #[getter]
    fn is_active(&self) -> bool {
        self.inner.is_active()
    }

    /// Full health snapshot as dict.
    fn health(&self) -> pyo3::PyObject {
        let h = self.inner.health();
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            let _ = dict.set_item("active", h.active);
            let _ = dict.set_item("turns_processed", h.turns_processed);
            let _ = dict.set_item("fragments", h.fragments);
            let _ = dict.set_item("atoms", h.atoms);
            let _ = dict.set_item("active_atoms", h.active_atoms);
            let _ = dict.set_item("insights", h.insights);
            let _ = dict.set_item("profile_completeness", h.profile_completeness);
            let _ = dict.set_item("synthesis_passes", h.synthesis_passes);
            let _ = dict.set_item("queries_served", h.queries_served);
            dict.into()
        })
    }

    fn __repr__(&self) -> String {
        let h = self.inner.health();
        format!(
            "BizraMemory(turns={}, atoms={}, insights={}, profile={:.0}%)",
            h.turns_processed,
            h.atoms,
            h.insights,
            h.profile_completeness * 100.0
        )
    }
}

// =============================================================================
// Graph-of-Thoughts Bindings — Besta et al. (2024) 6 Operations
// =============================================================================

/// Python wrapper for ThoughtGraph — Graph-of-Thoughts reasoning engine.
///
/// Implements all 6 GoT operations: GENERATE, AGGREGATE, REFINE, VALIDATE, PRUNE, BACKTRACK.
/// Each thought node carries an SNR score for quality-driven exploration.
///
/// Standing on Giants: Besta et al. (Graph-of-Thoughts, 2024) · Shannon (SNR)
#[pyclass(name = "ThoughtGraph")]
pub struct PyThoughtGraph {
    inner: bizra_core::ThoughtGraph,
}

#[pymethods]
impl PyThoughtGraph {
    /// Create a new empty thought graph.
    #[new]
    fn new() -> Self {
        Self {
            inner: bizra_core::ThoughtGraph::new(),
        }
    }

    /// Create a thought node (GENERATE operation).
    ///
    /// Args:
    ///     description: The thought content.
    ///     parent: Optional parent thought ID for tree structure.
    ///
    /// Returns: The new thought's ID (string).
    #[pyo3(signature = (description, parent=None))]
    fn create_thought(&mut self, description: &str, parent: Option<&str>) -> String {
        self.inner.create_thought(description, parent)
    }

    /// Create a typed thought node.
    ///
    /// thought_type: "hypothesis", "evidence", "reasoning", "synthesis",
    ///               "refinement", "validation", "conclusion", "question", "counterpoint"
    #[pyo3(signature = (description, thought_type, parent=None))]
    fn create_typed_thought(
        &mut self,
        description: &str,
        thought_type: &str,
        parent: Option<&str>,
    ) -> PyResult<String> {
        let tt = match thought_type.to_lowercase().as_str() {
            "hypothesis" => bizra_core::ThoughtType::Hypothesis,
            "evidence" => bizra_core::ThoughtType::Evidence,
            "reasoning" => bizra_core::ThoughtType::Reasoning,
            "synthesis" => bizra_core::ThoughtType::Synthesis,
            "refinement" => bizra_core::ThoughtType::Refinement,
            "validation" => bizra_core::ThoughtType::Validation,
            "conclusion" => bizra_core::ThoughtType::Conclusion,
            "question" => bizra_core::ThoughtType::Question,
            "counterpoint" => bizra_core::ThoughtType::Counterpoint,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown thought type: '{thought_type}'. Use: hypothesis, evidence, reasoning, synthesis, \
                 refinement, validation, conclusion, question, counterpoint"
                )))
            }
        };
        Ok(self.inner.create_thought_with_type(description, parent, tt))
    }

    /// Set a thought's result and confidence (REFINE operation).
    fn complete_thought(&mut self, id: &str, result: bool, confidence: f64) -> PyResult<()> {
        let node = self
            .inner
            .get_thought_mut(id)
            .ok_or_else(|| PyValueError::new_err(format!("Thought '{id}' not found")))?;
        node.complete(result, confidence);
        Ok(())
    }

    /// Set a thought's SNR score.
    fn set_snr(&mut self, id: &str, snr: f64) -> PyResult<()> {
        let node = self
            .inner
            .get_thought_mut(id)
            .ok_or_else(|| PyValueError::new_err(format!("Thought '{id}' not found")))?;
        node.set_snr(snr);
        Ok(())
    }

    /// BACKTRACK — Return to highest-SNR unexplored frontier node.
    ///
    /// The 6th GoT operation (Besta et al., 2024). Enables recovery from dead-ends
    /// by returning to promising unexplored branches.
    ///
    /// Returns dict with thought fields, or None if all explored.
    fn backtrack(&self) -> Option<pyo3::PyObject> {
        self.inner.backtrack().map(thought_to_pyobject)
    }

    /// VALIDATE — Get conclusions that meet the SNR threshold.
    fn get_conclusions(&self, min_snr: f64) -> Vec<pyo3::PyObject> {
        self.inner
            .get_conclusions(min_snr)
            .into_iter()
            .map(thought_to_pyobject)
            .collect()
    }

    /// Get frontier (leaf) nodes — candidates for expansion.
    fn get_frontier(&self) -> Vec<pyo3::PyObject> {
        self.inner
            .get_frontier()
            .into_iter()
            .map(thought_to_pyobject)
            .collect()
    }

    /// AGGREGATE — Aggregate all reasoning paths from a root.
    fn aggregate(&self, root_id: &str) -> pyo3::PyObject {
        let paths = self.inner.explore_parallel(root_id);
        let agg = self.inner.aggregate_paths(&paths);
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            let _ = dict.set_item("total_paths", agg.total_paths);
            let _ = dict.set_item("complete_paths", agg.complete_paths);
            let _ = dict.set_item("successful_paths", agg.successful_paths);
            let _ = dict.set_item("average_confidence", agg.average_confidence);
            let _ = dict.set_item("consensus", agg.consensus);
            dict.into()
        })
    }

    /// Graph statistics.
    fn stats(&self) -> pyo3::PyObject {
        let s = self.inner.stats();
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            let _ = dict.set_item("total_thoughts", s.total_thoughts);
            let _ = dict.set_item("total_paths", s.total_paths);
            let _ = dict.set_item("root_count", s.root_count);
            dict.into()
        })
    }

    /// Explore with automatic backtracking until target SNR reached.
    ///
    /// Returns the best thought found, or None if exhausted.
    fn explore_with_backtrack(
        &self,
        max_iterations: usize,
        target_snr: f64,
    ) -> Option<pyo3::PyObject> {
        self.inner
            .explore_with_backtrack(max_iterations, target_snr)
            .map(thought_to_pyobject)
    }

    fn __len__(&self) -> usize {
        self.inner.stats().total_thoughts
    }

    fn __repr__(&self) -> String {
        let s = self.inner.stats();
        format!(
            "ThoughtGraph(thoughts={}, paths={}, roots={})",
            s.total_thoughts, s.total_paths, s.root_count
        )
    }
}

/// Convert a ThoughtNode to a Python dict.
fn thought_to_pyobject(node: &bizra_core::ThoughtNode) -> pyo3::PyObject {
    Python::with_gil(|py| {
        let dict = pyo3::types::PyDict::new(py);
        let _ = dict.set_item("id", &node.id);
        let _ = dict.set_item("description", &node.description);
        let _ = dict.set_item("thought_type", format!("{:?}", node.thought_type));
        let _ = dict.set_item("result", node.result);
        let _ = dict.set_item("confidence", node.confidence);
        let _ = dict.set_item("snr_score", node.snr_score);
        let _ = dict.set_item("children", &node.children);
        let _ = dict.set_item("parent", &node.parent);
        dict.into()
    })
}

// ─── PyO3 Event Bridge ────────────────────────────────────────────────────

/// Python-callable event bridge into the Rust nervous system.
///
/// Usage from Python:
///   from bizra import PyEventBridge
///   bridge = PyEventBridge(production=False)
///   bridge.wire_subscribers()
///   delivered = bridge.emit("action.intent", "organize_invoices", 1)
///   health = bridge.health()
#[pyclass]
pub struct PyEventBridge {
    system: bizra_hooks::BizraSystem,
    source: Option<bizra_hooks::types::ComponentId>,
}

#[pymethods]
impl PyEventBridge {
    #[new]
    #[pyo3(signature = (production = false))]
    fn new(production: bool) -> Self {
        let system = if production {
            bizra_hooks::BizraSystem::production()
        } else {
            bizra_hooks::BizraSystem::new()
        };
        PyEventBridge {
            system,
            source: None,
        }
    }

    /// Wire all 12 constitutional subscribers. Returns count wired.
    fn wire_subscribers(&mut self) -> PyResult<usize> {
        let (wired, errors) = bizra_hooks::subscribers::wire_all(&mut self.system, 0);
        if !errors.is_empty() {
            return Err(PyRuntimeError::new_err(format!(
                "Failed to wire {} subscribers: {:?}",
                errors.len(),
                errors
            )));
        }

        // Register Python bridge as a source component
        let src = self
            .system
            .register_component("python-bridge", "1.0.0", 1)
            .map_err(|e| PyRuntimeError::new_err(format!("Registration failed: {e}")))?;
        self.system
            .activate_component(&src)
            .map_err(|e| PyRuntimeError::new_err(format!("Activation failed: {e}")))?;
        self.source = Some(src);

        Ok(wired)
    }

    /// Emit an event from Python into the Rust nervous system.
    /// priority: 0=Low, 1=Normal, 2=High, 3=Critical, 4=Emergency
    fn emit(&mut self, topic: &str, payload: &str, priority: u8) -> PyResult<usize> {
        let src = self
            .source
            .ok_or_else(|| PyRuntimeError::new_err("Call wire_subscribers() first"))?;
        let prio = match priority {
            0 => bizra_hooks::types::Priority::Low,
            1 => bizra_hooks::types::Priority::Normal,
            2 => bizra_hooks::types::Priority::High,
            3 => bizra_hooks::types::Priority::Critical,
            4 => bizra_hooks::types::Priority::Emergency,
            _ => return Err(PyValueError::new_err("priority must be 0-4")),
        };
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        self.system
            .emit(
                src,
                topic,
                bizra_hooks::types::Payload::from_text(payload),
                prio,
                now_ns,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Emit failed: {e}")))
    }

    /// Emit an event with receipt reference for cross-boundary trust.
    /// The receipt_id binds this event to a verified proof chain.
    /// This is the identity-aware handoff: Python hooks can verify
    /// that the event came from a governed Rust mission pipeline.
    ///
    /// Standing on: Lamport (1978) — cross-boundary event ordering
    /// Amanah: every cross-boundary event carries its receipt provenance
    fn emit_with_receipt(
        &mut self,
        topic: &str,
        payload: &str,
        receipt_id: &str,
        ihsan_score: f64,
        priority: u8,
    ) -> PyResult<usize> {
        let src = self
            .source
            .ok_or_else(|| PyRuntimeError::new_err("Call wire_subscribers() first"))?;
        let prio = match priority {
            0 => bizra_hooks::types::Priority::Low,
            1 => bizra_hooks::types::Priority::Normal,
            2 => bizra_hooks::types::Priority::High,
            3 => bizra_hooks::types::Priority::Critical,
            4 => bizra_hooks::types::Priority::Emergency,
            _ => return Err(PyValueError::new_err("priority must be 0-4")),
        };

        // Bind receipt reference into payload for cross-boundary verification
        let governed_payload = format!(
            "{{\"payload\":\"{payload}\",\"receipt_id\":\"{receipt_id}\",\"ihsan\":{ihsan_score:.4}}}"
        );

        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        self.system
            .emit(
                src,
                topic,
                bizra_hooks::types::Payload::from_text(&governed_payload),
                prio,
                now_ns,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Emit failed: {e}")))
    }

    /// Poll the subscriber feedback signals (atomic flags).
    /// Returns a dict with pending counts for each feedback type.
    /// Python hooks call this to know what the Rust nervous system detected.
    fn poll_feedback(&self, py: Python<'_>) -> PyResult<PyObject> {
        use bizra_hooks::subscribers::{
            PROMOTE_CHECK_PENDING, QUARANTINE_PENDING, REINFORCE_PENDING,
            SESSION_COMPILE_PENDING,
        };
        use core::sync::atomic::Ordering;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("reinforce_pending", REINFORCE_PENDING.load(Ordering::Relaxed))?;
        dict.set_item("promote_pending", PROMOTE_CHECK_PENDING.load(Ordering::Relaxed))?;
        dict.set_item("quarantine_pending", QUARANTINE_PENDING.load(Ordering::Relaxed))?;
        dict.set_item("compile_pending", SESSION_COMPILE_PENDING.load(Ordering::Relaxed))?;
        Ok(dict.into())
    }

    /// Get system health as a Python dict.
    fn health(&self, py: Python<'_>) -> PyResult<PyObject> {
        let h = self.system.health();
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("events_emitted", h.events_emitted)?;
        dict.set_item("events_delivered", h.events_delivered)?;
        dict.set_item("events_dropped", h.events_dropped)?;
        dict.set_item("delivery_ratio", h.delivery_ratio)?;
        dict.set_item("active_subscriptions", h.active_subscriptions)?;
        dict.set_item("system_ihsan", h.system_ihsan.as_f64())?;
        dict.set_item("gate_evaluations", h.gate_evaluations)?;
        dict.set_item("gate_violations", h.gate_violations)?;
        dict.set_item("gate_stability", h.gate_stability)?;
        Ok(dict.into())
    }
}

// ─── PyO3 Saga Types ─────────────────────────────────────────────────────

/// Python-callable saga phase enum.
///
/// Usage from Python:
///   from bizra import PySagaPhase
///   phase = PySagaPhase.received()
///   print(phase.name)   # "Received"
///   print(phase.value)  # 0
#[pyclass(name = "SagaPhase")]
#[derive(Clone)]
pub struct PySagaPhase {
    inner: bizra_hooks::saga::SagaPhase,
}

#[pymethods]
impl PySagaPhase {
    #[staticmethod]
    fn received() -> Self {
        Self {
            inner: bizra_hooks::saga::SagaPhase::Received,
        }
    }
    #[staticmethod]
    fn planned() -> Self {
        Self {
            inner: bizra_hooks::saga::SagaPhase::Planned,
        }
    }
    #[staticmethod]
    fn executed() -> Self {
        Self {
            inner: bizra_hooks::saga::SagaPhase::Executed,
        }
    }
    #[staticmethod]
    fn evaluated() -> Self {
        Self {
            inner: bizra_hooks::saga::SagaPhase::Evaluated,
        }
    }
    #[staticmethod]
    fn drafted() -> Self {
        Self {
            inner: bizra_hooks::saga::SagaPhase::Drafted,
        }
    }
    #[staticmethod]
    fn gated() -> Self {
        Self {
            inner: bizra_hooks::saga::SagaPhase::Gated,
        }
    }
    #[staticmethod]
    fn attested() -> Self {
        Self {
            inner: bizra_hooks::saga::SagaPhase::Attested,
        }
    }
    #[staticmethod]
    fn completed() -> Self {
        Self {
            inner: bizra_hooks::saga::SagaPhase::Completed,
        }
    }
    #[staticmethod]
    fn failed() -> Self {
        Self {
            inner: bizra_hooks::saga::SagaPhase::Failed,
        }
    }
    #[staticmethod]
    fn compensating() -> Self {
        Self {
            inner: bizra_hooks::saga::SagaPhase::Compensating,
        }
    }

    #[getter]
    fn name(&self) -> &'static str {
        match self.inner {
            bizra_hooks::saga::SagaPhase::Received => "Received",
            bizra_hooks::saga::SagaPhase::Planned => "Planned",
            bizra_hooks::saga::SagaPhase::Executed => "Executed",
            bizra_hooks::saga::SagaPhase::Evaluated => "Evaluated",
            bizra_hooks::saga::SagaPhase::Drafted => "Drafted",
            bizra_hooks::saga::SagaPhase::Gated => "Gated",
            bizra_hooks::saga::SagaPhase::Attested => "Attested",
            bizra_hooks::saga::SagaPhase::Completed => "Completed",
            bizra_hooks::saga::SagaPhase::Failed => "Failed",
            bizra_hooks::saga::SagaPhase::Compensating => "Compensating",
        }
    }

    #[getter]
    fn value(&self) -> u8 {
        self.inner as u8
    }

    #[getter]
    fn topic(&self) -> &'static str {
        self.inner.topic()
    }

    #[getter]
    fn is_terminal(&self) -> bool {
        matches!(
            self.inner,
            bizra_hooks::saga::SagaPhase::Completed | bizra_hooks::saga::SagaPhase::Failed
        )
    }

    fn __repr__(&self) -> String {
        format!("SagaPhase.{}", self.name())
    }
}

/// Python-callable saga registry for tracking request lifecycles.
///
/// Usage from Python:
///   from bizra import PySagaRegistry
///   registry = PySagaRegistry()
///   saga_id = registry.create("orchestrator", "1.0.0")
///   registry.advance(saga_id, 0.98)
///   print(registry.active_count)
#[pyclass(name = "SagaRegistry")]
pub struct PySagaRegistry {
    inner: bizra_hooks::saga::SagaRegistry,
}

#[pymethods]
impl PySagaRegistry {
    #[new]
    fn new() -> Self {
        Self {
            inner: bizra_hooks::saga::SagaRegistry::new(),
        }
    }

    /// Create a new saga. Returns saga_id (u64) or None if registry is full.
    fn create(&mut self, component_name: &str, component_version: &str) -> Option<u64> {
        let owner = bizra_hooks::types::ComponentId::from_name(component_name, component_version);
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        self.inner.create(owner, now_ns).map(|id| id.0)
    }

    /// Advance a saga to the next phase. Returns the action kind as a string.
    fn advance(&mut self, saga_id: u64, ihsan_score: f64) -> PyResult<String> {
        let id = bizra_hooks::saga::SagaId(saga_id);
        let saga = self
            .inner
            .get_mut(id)
            .ok_or_else(|| PyValueError::new_err(format!("Saga {saga_id} not found")))?;
        let ihsan = bizra_hooks::types::IhsanScore::from_f64(ihsan_score);
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        let action = saga.advance(ihsan, now_ns);
        Ok(match action {
            bizra_hooks::saga::SagaAction::Emit { topic, .. } => {
                format!("emit:{topic}")
            }
            bizra_hooks::saga::SagaAction::Complete => "complete".to_string(),
            bizra_hooks::saga::SagaAction::Fail { error_code } => {
                format!("fail:{error_code}")
            }
            bizra_hooks::saga::SagaAction::Aborted => "aborted".to_string(),
            bizra_hooks::saga::SagaAction::None => "none".to_string(),
        })
    }

    /// Fail a saga and begin compensation.
    fn fail(&mut self, saga_id: u64, error_code: u16) -> PyResult<String> {
        let id = bizra_hooks::saga::SagaId(saga_id);
        let saga = self
            .inner
            .get_mut(id)
            .ok_or_else(|| PyValueError::new_err(format!("Saga {saga_id} not found")))?;
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        let action = saga.fail(error_code, now_ns);
        Ok(match action {
            bizra_hooks::saga::SagaAction::Emit { topic, .. } => {
                format!("emit:{topic}")
            }
            bizra_hooks::saga::SagaAction::Fail { error_code } => {
                format!("fail:{error_code}")
            }
            _ => "none".to_string(),
        })
    }

    /// Get the current phase of a saga.
    fn phase(&self, saga_id: u64) -> PyResult<PySagaPhase> {
        let id = bizra_hooks::saga::SagaId(saga_id);
        let saga = self
            .inner
            .get(id)
            .ok_or_else(|| PyValueError::new_err(format!("Saga {saga_id} not found")))?;
        Ok(PySagaPhase { inner: saga.phase })
    }

    /// Remove a terminal saga from the registry.
    fn remove(&mut self, saga_id: u64) -> PyResult<bool> {
        let id = bizra_hooks::saga::SagaId(saga_id);
        Ok(self.inner.remove(id).is_some())
    }

    #[getter]
    fn active_count(&self) -> usize {
        self.inner.active_count()
    }

    #[getter]
    fn total_created(&self) -> u64 {
        self.inner.total_created()
    }

    #[getter]
    fn total_completed(&self) -> u64 {
        self.inner.total_completed()
    }

    #[getter]
    fn total_failed(&self) -> u64 {
        self.inner.total_failed()
    }

    fn __repr__(&self) -> String {
        format!(
            "SagaRegistry(active={}, created={}, completed={}, failed={})",
            self.inner.active_count(),
            self.inner.total_created(),
            self.inner.total_completed(),
            self.inner.total_failed(),
        )
    }
}

/// Python wrapper for ReflexLedger (System 1 cache)
#[pyclass(name = "ReflexLedger")]
pub struct PyReflexLedger {
    inner: bizra_action::ReflexLedger,
}

#[pymethods]
impl PyReflexLedger {
    #[new]
    fn new(capacity: usize) -> Self {
        Self {
            inner: bizra_action::ReflexLedger::new(capacity),
        }
    }

    /// Compile a reflex from successful reasoning (Ihsan >= threshold)
    fn compile_vrg_reflex(
        &mut self,
        task_description: &str,
        ihsan_score: f64,
        timestamp_ns: u64,
        vrg_root: &str,
        branch_certificates: Vec<String>,
    ) -> PyResult<usize> {
        let score = bizra_action::IhsanScore::new(ihsan_score);
        let ts = bizra_action::ActionTimestamp(timestamp_ns);
        let actions = vec![]; // Verified abstract thought
        let mut provenance = vec![vrg_root.to_string()];
        provenance.extend(branch_certificates);

        self.inner
            .compile(task_description, actions, score, ts, provenance)
            .map_err(|e| PyRuntimeError::new_err(format!("Reflex compilation failed: {:?}", e)))
    }
}

/// Python wrapper for Dilithium-5 (ML-DSA-87) Keypair
#[pyclass(name = "DilithiumKeypair")]
#[derive(Clone)]
pub struct PyDilithiumKeypair {
    public_key: Vec<u8>,
    secret_key: Vec<u8>,
}

#[pymethods]
impl PyDilithiumKeypair {
    #[staticmethod]
    fn generate() -> PyResult<Self> {
        use pqcrypto_traits::sign::{PublicKey, SecretKey};
        let (pk, sk) = pqcrypto_mldsa::mldsa87::keypair();
        Ok(Self {
            public_key: pk.as_bytes().to_vec(),
            secret_key: sk.as_bytes().to_vec(),
        })
    }

    #[getter]
    fn public_key_hex(&self) -> String {
        hex::encode(&self.public_key)
    }

    fn sign(&self, message: &[u8]) -> PyResult<Vec<u8>> {
        use pqcrypto_traits::sign::{DetachedSignature, SecretKey};
        let sk = pqcrypto_mldsa::mldsa87::SecretKey::from_bytes(&self.secret_key)
            .map_err(|_| PyValueError::new_err("Invalid secret key"))?;
        let signature = pqcrypto_mldsa::mldsa87::detached_sign(message, &sk);
        Ok(signature.as_bytes().to_vec())
    }

    fn verify(&self, message: &[u8], signature: &[u8]) -> PyResult<bool> {
        use pqcrypto_traits::sign::{DetachedSignature, PublicKey};
        let pk = pqcrypto_mldsa::mldsa87::PublicKey::from_bytes(&self.public_key)
            .map_err(|_| PyValueError::new_err("Invalid public key"))?;
        let sig = pqcrypto_mldsa::mldsa87::DetachedSignature::from_bytes(signature)
            .map_err(|_| PyValueError::new_err("Invalid signature format"))?;
        Ok(pqcrypto_mldsa::mldsa87::verify_detached_signature(&sig, message, &pk).is_ok())
    }
}

#[pyfunction]
fn verify_dilithium_signature(
    public_key: &[u8],
    message: &[u8],
    signature: &[u8],
) -> PyResult<bool> {
    use pqcrypto_traits::sign::{DetachedSignature, PublicKey};
    let pk = pqcrypto_mldsa::mldsa87::PublicKey::from_bytes(public_key)
        .map_err(|_| PyValueError::new_err("Invalid public key"))?;
    let sig = pqcrypto_mldsa::mldsa87::DetachedSignature::from_bytes(signature)
        .map_err(|_| PyValueError::new_err("Invalid signature format"))?;
    Ok(pqcrypto_mldsa::mldsa87::verify_detached_signature(&sig, message, &pk).is_ok())
}

#[pymodule]
fn bizra(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core types
    m.add_class::<PyNodeId>()?;
    m.add_class::<PyNodeIdentity>()?;
    m.add_class::<PyConstitution>()?;
    m.add_class::<PyPCIEnvelope>()?;

    // Inference types
    m.add_class::<PyTaskComplexity>()?;
    m.add_class::<PyModelTier>()?;
    m.add_class::<PyModelSelector>()?;

    // Gate chain
    m.add_class::<PyGateChain>()?;

    // SNR Engine (Rust-native signal quality measurement)
    m.add_class::<PySNREngine>()?;

    // Inference gateway (Python↔Rust unified path)
    m.add_class::<PyInferenceGateway>()?;
    m.add_class::<PyInferenceResponse>()?;

    // Autopoiesis (pattern learning + preference tracking)
    m.add_class::<PyPatternMemory>()?;
    m.add_class::<PyPreferenceTracker>()?;

    // Sovereign Experience Ledger (episodic memory)
    m.add_class::<PyExperienceLedger>()?;

    // Cognitive Layer: Memory Synthesis (bizra-memory)
    m.add_class::<PyBizraMemory>()?;

    // Cognitive Layer: Graph-of-Thoughts (bizra-core::sovereign)
    m.add_class::<PyThoughtGraph>()?;

    // Nervous System: Event Bridge (bizra-hooks)
    m.add_class::<PyEventBridge>()?;

    // Saga: Request lifecycle tracking (bizra-hooks::saga)
    m.add_class::<PySagaPhase>()?;
    m.add_class::<PySagaRegistry>()?;

    // Fast System 1 Reflexes
    m.add_class::<PyReflexLedger>()?;

    // Functions
    m.add_function(wrap_pyfunction!(domain_separated_digest, m)?)?;
    m.add_function(wrap_pyfunction!(get_ihsan_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(get_snr_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(verify_dilithium_signature, m)?)?;

    // Post-Quantum Cryptography
    m.add_class::<PyDilithiumKeypair>()?;

    // URP Bridge: ResourcePool types and operations
    urp_bridge::register_urp_types(m)?;

    // Module metadata
    m.add("__version__", "2.0.0")?;
    m.add("IHSAN_THRESHOLD", IHSAN_THRESHOLD)?;
    m.add("SNR_THRESHOLD", SNR_THRESHOLD)?;

    Ok(())
}
