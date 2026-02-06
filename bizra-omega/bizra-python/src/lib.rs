//! BIZRA Python Bindings — PyO3 Bridge
//!
//! Exposes Rust bizra-core to Python for 10-100x performance boost.
//! Giants: PyO3 team, Rust-Python interop pioneers

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

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
            .map_err(|e| PyValueError::new_err(format!("Invalid JSON: {}", e)))?;

        let envelope = RustPCIEnvelope::create(&identity.inner, payload_value, ttl, provenance)
            .map_err(|e| PyRuntimeError::new_err(format!("PCI error: {}", e)))?;

        Ok(Self {
            id: envelope.id,
            sender: PyNodeId {
                inner: envelope.sender,
            },
            content_hash: envelope.content_hash,
            signature: envelope.signature,
            public_key: envelope.public_key,
            payload_json: serde_json::to_string(&envelope.payload).unwrap(),
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
            level: format!("{:?}", complexity),
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

/// BIZRA Python Module
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

    // Functions
    m.add_function(wrap_pyfunction!(domain_separated_digest, m)?)?;
    m.add_function(wrap_pyfunction!(get_ihsan_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(get_snr_threshold, m)?)?;

    // Module metadata
    m.add("__version__", "1.0.0")?;
    m.add("IHSAN_THRESHOLD", IHSAN_THRESHOLD)?;
    m.add("SNR_THRESHOLD", SNR_THRESHOLD)?;

    Ok(())
}
