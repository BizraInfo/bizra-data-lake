//! Inference Backends
//!
//! Priority order:
//! 1. LM Studio (primary) — reasoning, agentic, vision, voice
//! 2. Ollama (fallback) — general chat
//! 3. LlamaCpp (embedded) — edge/offline

pub mod lmstudio;
pub mod ollama;
pub mod llamacpp;
pub mod llamacpp_ffi;

pub use lmstudio::{LMStudioBackend, LMStudioConfig, ModelCapability};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::gateway::{InferenceRequest, InferenceResponse};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackendConfig {
    pub name: String,
    pub model: String,
    pub context_length: usize,
    pub gpu_layers: i32,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self { name: "default".into(), model: "".into(), context_length: 4096, gpu_layers: -1 }
    }
}

#[async_trait]
pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
    async fn generate(&self, request: &InferenceRequest) -> Result<InferenceResponse, BackendError>;
    async fn health_check(&self) -> bool;
}

#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("Model not loaded")]
    NotLoaded,
    #[error("Generation failed: {0}")]
    Generation(String),
    #[error("Connection failed: {0}")]
    Connection(String),
}
