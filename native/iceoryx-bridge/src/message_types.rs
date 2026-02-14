//! Message Types for IPC Communication

use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde::{Deserialize, Serialize};

/// Inference request from orchestrator to sandbox
#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// Unique request ID
    pub id: String,

    /// Prompt text
    pub prompt: String,

    /// Target model ID
    pub model_id: String,

    /// Maximum tokens to generate
    pub max_tokens: i32,

    /// Temperature (0.0 - 2.0)
    pub temperature: f64,

    /// Top-p sampling
    pub top_p: f64,
}

/// Inference response from sandbox
#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    /// Request ID (matches request)
    pub id: String,

    /// Generated content
    pub content: String,

    /// Model that generated the response
    pub model_id: String,

    /// Tokens generated
    pub tokens_generated: i32,

    /// Generation time in milliseconds
    pub generation_time_ms: i64,

    /// Raw Ihsān score (before gate validation)
    pub ihsan_score: f64,

    /// Raw SNR score (before gate validation)
    pub snr_score: f64,

    /// Whether generation succeeded
    pub success: bool,

    /// Error message if failed
    pub error: Option<String>,
}

/// Gate validation request
#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateRequest {
    /// Request ID
    pub id: String,

    /// Inference output to validate
    pub output: InferenceResponse,

    /// Required gates (empty = all gates)
    pub gates: Vec<String>,
}

/// Gate validation response
#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateResponse {
    /// Request ID
    pub id: String,

    /// Whether all gates passed
    pub passed: bool,

    /// Individual gate results
    pub gate_results: Vec<GateResultDetail>,

    /// Final validated score
    pub final_score: f64,

    /// PCI signature if passed
    pub pci_signature: Option<String>,
}

/// Individual gate result
#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateResultDetail {
    /// Gate name
    pub gate: String,

    /// Whether this gate passed
    pub passed: bool,

    /// Score for this gate
    pub score: f64,

    /// Reason for failure (if failed)
    pub reason: Option<String>,

    /// Time taken in nanoseconds
    pub time_ns: i64,
}

/// Control message for bridge coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlMessage {
    /// Ping/pong for health checks
    Ping { id: u64 },
    Pong { id: u64 },

    /// Shutdown signal
    Shutdown,

    /// Stats request
    StatsRequest,
    StatsResponse(BridgeStatsInternal),

    /// Model loading
    LoadModel { model_id: String, model_path: String },
    ModelLoaded { model_id: String, success: bool },

    /// Configuration update
    UpdateConfig { config_json: String },
    ConfigUpdated { success: bool },
}

/// Internal bridge statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BridgeStatsInternal {
    pub messages_sent: u64,
    pub messages_received: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub latencies_ns: Vec<u64>,
    pub errors: u64,
}

impl BridgeStatsInternal {
    pub fn avg_latency_ns(&self) -> u64 {
        if self.latencies_ns.is_empty() {
            0
        } else {
            self.latencies_ns.iter().sum::<u64>() / self.latencies_ns.len() as u64
        }
    }

    pub fn p99_latency_ns(&self) -> u64 {
        if self.latencies_ns.is_empty() {
            return 0;
        }

        let mut sorted = self.latencies_ns.clone();
        sorted.sort();
        sorted[(sorted.len() as f64 * 0.99) as usize]
    }
}

/// Message envelope for all IPC messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageEnvelope {
    /// Message type discriminator
    pub msg_type: MessageType,

    /// Timestamp (Unix epoch ns)
    pub timestamp_ns: u64,

    /// Sender ID
    pub sender: String,

    /// Target receiver (empty = broadcast)
    pub target: String,

    /// Payload (MessagePack or JSON)
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MessageType {
    InferenceRequest,
    InferenceResponse,
    GateRequest,
    GateResponse,
    Control,
}

impl MessageEnvelope {
    /// Create a new envelope for an inference request
    pub fn inference_request(request: &InferenceRequest, sender: &str) -> Self {
        Self {
            msg_type: MessageType::InferenceRequest,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            sender: sender.to_string(),
            target: "sandbox".to_string(),
            payload: rmp_serde::to_vec(request).unwrap(),
        }
    }

    /// Create a new envelope for an inference response
    pub fn inference_response(response: &InferenceResponse, sender: &str) -> Self {
        Self {
            msg_type: MessageType::InferenceResponse,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            sender: sender.to_string(),
            target: "orchestrator".to_string(),
            payload: rmp_serde::to_vec(response).unwrap(),
        }
    }

    /// Decode payload as specific type
    pub fn decode<T: for<'de> Deserialize<'de>>(&self) -> Result<T> {
        rmp_serde::from_slice(&self.payload)
            .map_err(|e| Error::from_reason(format!("Decode error: {}", e)))
    }
}
