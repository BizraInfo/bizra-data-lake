//! BIZRA Inference — Sovereign LLM Gateway
//!
//! Backend Priority:
//! 1. LM Studio (primary) — WSL gateway (auto-detected, env: LMSTUDIO_HOST)
//!    - Reasoning: DeepSeek-R1, Qwen-72B
//!    - Agentic: function calling, tool use
//!    - Vision: LLaVA, Qwen-VL
//!    - Voice: Whisper, Moshi
//! 2. Ollama (fallback) — localhost:11434
//! 3. LlamaCpp (embedded) — edge/offline

use async_trait::async_trait;

pub mod backends;
pub mod gateway;
pub mod selector;

pub use backends::{
    Backend, BackendConfig, BackendError, LMStudioBackend, LMStudioConfig, ModelCapability,
    OllamaBackend,
};
pub use gateway::{GatewayError, InferenceGateway, InferenceRequest, InferenceResponse};
pub use selector::{ModelSelector, ModelTier, TaskComplexity};

/// Default timeout for inference requests
pub const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// LM Studio connection defaults
/// NOTE: In WSL2, the Windows host IP changes between reboots.
/// Use env var LMSTUDIO_HOST to override. Python side auto-detects via `ip route`.
pub const LMSTUDIO_DEFAULT_HOST: &str = "172.22.48.1";
pub const LMSTUDIO_DEFAULT_PORT: u16 = 1234;

#[async_trait]
pub trait InferenceBackend: Send + Sync {
    async fn generate(&self, req: InferenceRequest) -> Result<InferenceResponse, BackendError>;
}

#[async_trait]
impl<T> InferenceBackend for T
where
    T: backends::Backend + Send + Sync,
{
    async fn generate(&self, req: InferenceRequest) -> Result<InferenceResponse, BackendError> {
        backends::Backend::generate(self, &req).await
    }
}
