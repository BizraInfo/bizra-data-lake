//! BIZRA Inference — Sovereign LLM Gateway
//!
//! Backend Priority:
//! 1. LM Studio (primary) — 192.168.56.1:1234
//!    - Reasoning: DeepSeek-R1, Qwen-72B
//!    - Agentic: function calling, tool use
//!    - Vision: LLaVA, Qwen-VL
//!    - Voice: Whisper, Moshi
//! 2. Ollama (fallback) — localhost:11434
//! 3. LlamaCpp (embedded) — edge/offline

pub mod backends;
pub mod gateway;
pub mod selector;

pub use backends::{Backend, BackendConfig, BackendError};
pub use backends::{LMStudioBackend, LMStudioConfig, ModelCapability};
pub use gateway::{InferenceGateway, InferenceRequest, InferenceResponse};
pub use selector::{ModelSelector, ModelTier, TaskComplexity};

/// Default timeout for inference requests
pub const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// LM Studio connection defaults
pub const LMSTUDIO_DEFAULT_HOST: &str = "192.168.56.1";
pub const LMSTUDIO_DEFAULT_PORT: u16 = 1234;
