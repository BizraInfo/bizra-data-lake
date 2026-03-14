//! URP Bridge — PyO3 wrappers for bizra-resourcepool types.
//!
//! Exposes ResourcePool operations to Python for pledge submission,
//! resource contribution, Zakat processing, and ADL compliance.
//!
//! Standing on Giants: Ostrom (Commons), Weyl & Posner (Harberger)

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};

use bizra_resourcepool::{
    NodeClass, NodeResources, ResourceContribution, ResourcePool, ResourceType,
};
use chrono::Utc;
use rust_decimal::prelude::*;
use std::sync::Arc;
use tokio::runtime::Runtime;

// =============================================================================
// PyResourcePool — Pool Handle
// =============================================================================

/// Python wrapper for ResourcePool. Created once at node boot.
#[pyclass(name = "PyResourcePool")]
pub struct PyResourcePool {
    inner: Arc<ResourcePool>,
    rt: Arc<Runtime>,
}

#[pymethods]
impl PyResourcePool {
    /// Create a new genesis pool with a placeholder node.
    #[new]
    fn new() -> PyResult<Self> {
        let rt = Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Tokio init failed: {e}")))?;

        // Generate a placeholder genesis key for pool initialization
        let genesis_key = ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng);
        let verifying_key = genesis_key.verifying_key();

        let pool = rt
            .block_on(ResourcePool::genesis(
                "genesis-node-0".to_string(),
                "Node0-Genesis".to_string(),
                verifying_key,
            ))
            .map_err(|e| PyRuntimeError::new_err(format!("Pool genesis failed: {e}")))?;

        Ok(Self {
            inner: Arc::new(pool),
            rt: Arc::new(rt),
        })
    }
}

// =============================================================================
// PyURPPledge — Pledge Wrapper
// =============================================================================

/// Python wrapper for a URP pledge (Python→Rust direction).
#[pyclass(name = "PyURPPledge")]
#[derive(Clone)]
pub struct PyURPPledge {
    node_id: String,
    ram_gb: u32,
    vram_gb: u32,
    storage_gb: u32,
    pledge_hash: String,
    pledged_at: String,
    signed: bool,
    signature: String,
    signer_public_key: String,
    payload_digest: String,
    enforcement_mode: String,
    status: String,
}

#[pymethods]
impl PyURPPledge {
    #[new]
    #[pyo3(signature = (node_id, ram_gb=0, vram_gb=0, storage_gb=0, pledge_hash="".to_string(), pledged_at="".to_string(), signed=false, signature="".to_string(), signer_public_key="".to_string(), payload_digest="".to_string(), enforcement_mode="stub".to_string(), status="deferred".to_string()))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        node_id: String,
        ram_gb: u32,
        vram_gb: u32,
        storage_gb: u32,
        pledge_hash: String,
        pledged_at: String,
        signed: bool,
        signature: String,
        signer_public_key: String,
        payload_digest: String,
        enforcement_mode: String,
        status: String,
    ) -> Self {
        Self {
            node_id,
            ram_gb,
            vram_gb,
            storage_gb,
            pledge_hash,
            pledged_at,
            signed,
            signature,
            signer_public_key,
            payload_digest,
            enforcement_mode,
            status,
        }
    }

    #[getter]
    fn node_id(&self) -> &str {
        &self.node_id
    }

    #[getter]
    fn ram_gb(&self) -> u32 {
        self.ram_gb
    }

    #[getter]
    fn vram_gb(&self) -> u32 {
        self.vram_gb
    }

    #[getter]
    fn signed(&self) -> bool {
        self.signed
    }

    #[getter]
    fn status(&self) -> &str {
        &self.status
    }

    /// Reconstruct from a Python dict.
    #[classmethod]
    fn from_dict(_cls: &Bound<'_, PyType>, data: &Bound<'_, PyDict>) -> PyResult<Self> {
        let node_id: String = data
            .get_item("node_id")?
            .ok_or_else(|| PyValueError::new_err("missing node_id"))?
            .extract()?;
        let ram_gb: u32 = data
            .get_item("ram_gb")?
            .map(|v| v.extract().unwrap_or(0))
            .unwrap_or(0);
        let vram_gb: u32 = data
            .get_item("vram_gb")?
            .map(|v| v.extract().unwrap_or(0))
            .unwrap_or(0);
        let storage_gb: u32 = data
            .get_item("storage_gb")?
            .map(|v| v.extract().unwrap_or(0))
            .unwrap_or(0);

        fn get_str(data: &Bound<'_, PyDict>, key: &str) -> String {
            data.get_item(key)
                .ok()
                .flatten()
                .and_then(|v| v.extract::<String>().ok())
                .unwrap_or_default()
        }

        let signed: bool = data
            .get_item("signed")?
            .map(|v| v.extract().unwrap_or(false))
            .unwrap_or(false);

        Ok(Self {
            node_id,
            ram_gb,
            vram_gb,
            storage_gb,
            pledge_hash: get_str(data, "pledge_hash"),
            pledged_at: get_str(data, "pledged_at"),
            signed,
            signature: get_str(data, "signature"),
            signer_public_key: get_str(data, "signer_public_key"),
            payload_digest: get_str(data, "payload_digest"),
            enforcement_mode: get_str(data, "enforcement_mode"),
            status: get_str(data, "status"),
        })
    }

    /// Serialize to Python dict.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("node_id", &self.node_id)?;
        dict.set_item("ram_gb", self.ram_gb)?;
        dict.set_item("vram_gb", self.vram_gb)?;
        dict.set_item("storage_gb", self.storage_gb)?;
        dict.set_item("pledge_hash", &self.pledge_hash)?;
        dict.set_item("pledged_at", &self.pledged_at)?;
        dict.set_item("signed", self.signed)?;
        dict.set_item("signature", &self.signature)?;
        dict.set_item("signer_public_key", &self.signer_public_key)?;
        dict.set_item("payload_digest", &self.payload_digest)?;
        dict.set_item("enforcement_mode", &self.enforcement_mode)?;
        dict.set_item("status", &self.status)?;
        Ok(dict)
    }

    /// Verify Ed25519 signature (Rust-authoritative).
    fn verify_signature(&self) -> bool {
        if !self.signed || self.signature.is_empty() || self.signer_public_key.is_empty() {
            return false;
        }

        let Ok(pk_bytes) = hex::decode(&self.signer_public_key) else {
            return false;
        };
        let Ok(sig_bytes) = hex::decode(&self.signature) else {
            return false;
        };

        let pk_array: [u8; 32] = match pk_bytes.try_into() {
            Ok(a) => a,
            Err(_) => return false,
        };
        let sig_array: [u8; 64] = match sig_bytes.try_into() {
            Ok(a) => a,
            Err(_) => return false,
        };

        let Ok(verifying_key) = ed25519_dalek::VerifyingKey::from_bytes(&pk_array) else {
            return false;
        };
        let signature = ed25519_dalek::Signature::from_bytes(&sig_array);

        // Reconstruct canonical payload digest
        let digest_bytes = match hex::decode(&self.payload_digest) {
            Ok(b) => b,
            Err(_) => return false,
        };

        verifying_key
            .verify_strict(&digest_bytes, &signature)
            .is_ok()
    }
}

// =============================================================================
// PyPoolNode — Node Registration Result
// =============================================================================

/// Read-only view of a registered node in the pool.
#[pyclass(name = "PyPoolNode")]
pub struct PyPoolNode {
    id: String,
    class: String,
    status: String,
    token_balance: u64,
    ihsan_score: f64,
    registered_at: String,
}

#[pymethods]
impl PyPoolNode {
    #[getter]
    fn id(&self) -> &str {
        &self.id
    }

    #[getter]
    fn class(&self) -> &str {
        &self.class
    }

    #[getter]
    fn token_balance(&self) -> u64 {
        self.token_balance
    }

    #[getter]
    fn ihsan_score(&self) -> f64 {
        self.ihsan_score
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("id", &self.id)?;
        dict.set_item("class", &self.class)?;
        dict.set_item("status", &self.status)?;
        dict.set_item("token_balance", self.token_balance)?;
        dict.set_item("ihsan_score", self.ihsan_score)?;
        dict.set_item("registered_at", &self.registered_at)?;
        Ok(dict)
    }

    fn passes_ihsan(&self) -> bool {
        self.ihsan_score >= 0.95
    }
}

// =============================================================================
// PyContributionReceipt — Proof-of-Impact Receipt
// =============================================================================

/// Receipt proving a resource contribution was recorded.
#[pyclass(name = "PyContributionReceipt")]
pub struct PyContributionReceipt {
    contribution_id: String,
    node_id: String,
    resource_type: String,
    amount: f64,
    duration_ms: u64,
    tokens_earned: u64,
    receipt_hash: String,
    timestamp: String,
}

#[pymethods]
impl PyContributionReceipt {
    #[getter]
    fn tokens_earned(&self) -> u64 {
        self.tokens_earned
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("contribution_id", &self.contribution_id)?;
        dict.set_item("node_id", &self.node_id)?;
        dict.set_item("resource_type", &self.resource_type)?;
        dict.set_item("amount", self.amount)?;
        dict.set_item("duration_ms", self.duration_ms)?;
        dict.set_item("tokens_earned", self.tokens_earned)?;
        dict.set_item("receipt_hash", &self.receipt_hash)?;
        dict.set_item("timestamp", &self.timestamp)?;
        Ok(dict)
    }
}

// =============================================================================
// PyPoolStats — Pool-Level Statistics
// =============================================================================

/// Pool statistics exposed to Python.
#[pyclass(name = "PyPoolStats")]
pub struct PyPoolStats {
    total_nodes: usize,
    active_nodes: usize,
    total_services: usize,
    total_compute: u64,
    gini_coefficient: f64,
    adl_compliant: bool,
    avg_ihsan: f64,
}

#[pymethods]
impl PyPoolStats {
    #[getter]
    fn total_nodes(&self) -> usize {
        self.total_nodes
    }

    #[getter]
    fn gini_coefficient(&self) -> f64 {
        self.gini_coefficient
    }

    #[getter]
    fn adl_compliant(&self) -> bool {
        self.adl_compliant
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("total_nodes", self.total_nodes)?;
        dict.set_item("active_nodes", self.active_nodes)?;
        dict.set_item("total_services", self.total_services)?;
        dict.set_item("total_compute", self.total_compute)?;
        dict.set_item("gini_coefficient", self.gini_coefficient)?;
        dict.set_item("adl_compliant", self.adl_compliant)?;
        dict.set_item("avg_ihsan", self.avg_ihsan)?;
        Ok(dict)
    }
}

// =============================================================================
// Free Functions — Operations
// =============================================================================

/// Submit a signed URP pledge to the Rust pool.
#[pyfunction]
pub fn submit_pledge(pool: &PyResourcePool, pledge: &PyURPPledge) -> PyResult<PyPoolNode> {
    if !pledge.signed {
        return Err(PyRuntimeError::new_err("Pledge must be signed"));
    }
    if !pledge.verify_signature() {
        return Err(PyRuntimeError::new_err("Invalid pledge signature"));
    }
    if pledge.ram_gb == 0 && pledge.vram_gb == 0 {
        return Err(PyRuntimeError::new_err(
            "Pledge must include at least RAM or VRAM",
        ));
    }

    // Build a minimal RegistrationRequest from pledge fields
    let request = bizra_resourcepool::RegistrationRequest {
        node_id: pledge.node_id.clone(),
        name: format!("node-{}", &pledge.node_id[..8.min(pledge.node_id.len())]),
        requested_class: NodeClass::Sovereign,
        resources: NodeResources {
            cpu_millicores: 0,
            gpu_tflops: Decimal::ZERO,
            memory_bytes: (pledge.ram_gb as u64) * 1_073_741_824,
            storage_bytes: (pledge.storage_gb as u64) * 1_073_741_824,
            network_bps: 0,
            inference_tps: 0,
            self_assessment: 0,
            availability: Decimal::ONE,
        },
        sponsor_node: None,
        identity_proof: Some(pledge.pledge_hash.clone()),
        requested_at: Utc::now(),
        signature: pledge.signature.clone(),
    };

    let response = pool
        .rt
        .block_on(pool.inner.register_node(request))
        .map_err(|e| PyRuntimeError::new_err(format!("Registration failed: {e}")))?;

    Ok(PyPoolNode {
        id: pledge.node_id.clone(),
        class: if response.approved {
            "sovereign"
        } else {
            "pending"
        }
        .to_string(),
        status: if response.approved {
            "active"
        } else {
            "pending"
        }
        .to_string(),
        token_balance: response.initial_tokens,
        ihsan_score: 1.0,
        registered_at: response.responded_at.to_rfc3339(),
    })
}

/// Record a resource contribution and mint SEED tokens.
#[pyfunction]
pub fn contribute_resources(
    pool: &PyResourcePool,
    node_id: &str,
    resource_type: &str,
    amount: f64,
    duration_ms: u64,
    proof_hash: &str,
) -> PyResult<PyContributionReceipt> {
    let res_type = match resource_type {
        "cpu" => ResourceType::Cpu,
        "ram" | "memory" => ResourceType::Memory,
        "gpu" => ResourceType::Gpu,
        "storage" => ResourceType::Storage,
        "network" => ResourceType::Network,
        "witness" | "inference" => ResourceType::Inference,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Invalid resource type: {resource_type}"
            )))
        }
    };

    let contribution = ResourceContribution {
        resource_type: res_type,
        quantity: amount as u64,
        duration_seconds: duration_ms / 1000,
        utilization: 100,
        proof_block_id: Some(proof_hash.to_string()),
    };

    let mint_proof = pool
        .rt
        .block_on(pool.inner.contribute_resources(node_id, contribution))
        .map_err(|e| PyRuntimeError::new_err(format!("Contribution failed: {e}")))?;

    Ok(PyContributionReceipt {
        contribution_id: mint_proof.proof_id.to_string(),
        node_id: node_id.to_string(),
        resource_type: resource_type.to_string(),
        amount,
        duration_ms,
        tokens_earned: mint_proof.tokens_minted,
        receipt_hash: hex::encode(mint_proof.proof_hash),
        timestamp: mint_proof.timestamp.to_rfc3339(),
    })
}

/// Get SEED token balance and contribution history for a node.
#[pyfunction]
pub fn get_rewards<'py>(
    py: Python<'py>,
    pool: &PyResourcePool,
    node_id: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let node = pool
        .rt
        .block_on(pool.inner.get_node(node_id))
        .ok_or_else(|| PyRuntimeError::new_err("Node not found"))?;

    let dict = PyDict::new(py);
    dict.set_item("node_id", &node.node_id)?;
    dict.set_item("balance", node.token_balance)?;
    dict.set_item("ihsan_score", node.ihsan_score.to_f64().unwrap_or(0.0))?;
    dict.set_item("node_class", node.class.as_registration_label())?;
    dict.set_item("zakat_paid", node.zakat_paid_year)?;
    Ok(dict)
}

/// Trigger Zakat distribution across the pool.
#[pyfunction]
pub fn process_zakat<'py>(py: Python<'py>, pool: &PyResourcePool) -> PyResult<Bound<'py, PyDict>> {
    let dist = pool
        .rt
        .block_on(pool.inner.process_zakat())
        .map_err(|e| PyRuntimeError::new_err(format!("Zakat processing failed: {e}")))?;

    let dict = PyDict::new(py);
    dict.set_item("total_collected", dist.total_collected)?;
    dict.set_item("recipients", dist.distributions.len())?;
    dict.set_item("period", &dist.period)?;
    dict.set_item("timestamp", dist.distributed_at.to_rfc3339())?;
    Ok(dict)
}

/// Check ADL (justice) compliance via Gini coefficient.
#[pyfunction]
pub fn check_adl<'py>(py: Python<'py>, pool: &PyResourcePool) -> PyResult<Bound<'py, PyDict>> {
    let gini = pool.rt.block_on(pool.inner.calculate_gini());
    let compliant = pool.rt.block_on(pool.inner.check_adl()).is_ok();

    let dict = PyDict::new(py);
    dict.set_item("gini_coefficient", gini.to_f64().unwrap_or(0.0))?;
    dict.set_item("threshold", 0.35)?;
    dict.set_item("compliant", compliant)?;
    dict.set_item("action_required", !compliant)?;
    Ok(dict)
}

/// Get current pool statistics.
#[pyfunction]
pub fn pool_stats(pool: &PyResourcePool) -> PyResult<PyPoolStats> {
    let stats = pool.rt.block_on(pool.inner.stats());
    let gini = pool.rt.block_on(pool.inner.calculate_gini());
    let gini_f64 = gini.to_f64().unwrap_or(0.0);

    Ok(PyPoolStats {
        total_nodes: stats.total_nodes,
        active_nodes: stats.active_nodes,
        total_services: stats.total_services,
        total_compute: stats.total_compute,
        gini_coefficient: gini_f64,
        adl_compliant: gini_f64 <= 0.35,
        avg_ihsan: stats.avg_ihsan.to_f64().unwrap_or(0.0),
    })
}

// =============================================================================
// Module Registration
// =============================================================================

/// Register all URP bridge types and functions with the PyO3 module.
pub fn register_urp_types(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyResourcePool>()?;
    m.add_class::<PyURPPledge>()?;
    m.add_class::<PyPoolNode>()?;
    m.add_class::<PyContributionReceipt>()?;
    m.add_class::<PyPoolStats>()?;
    m.add_function(wrap_pyfunction!(submit_pledge, m)?)?;
    m.add_function(wrap_pyfunction!(contribute_resources, m)?)?;
    m.add_function(wrap_pyfunction!(get_rewards, m)?)?;
    m.add_function(wrap_pyfunction!(process_zakat, m)?)?;
    m.add_function(wrap_pyfunction!(check_adl, m)?)?;
    m.add_function(wrap_pyfunction!(pool_stats, m)?)?;
    Ok(())
}
