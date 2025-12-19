//! llm_gateway.rs
//! Phase 1: Wiring the Brain
//! - Routes PAT/SAT requests to Ollama or LM Studio based on docs/runtime/slots.yaml
//! - Keeps providers swappable + testable
use std::collections::HashMap;

use serde::Deserialize;

use crate::providers::{ollama::OllamaClient, openai_compat::OpenAiCompatClient};

#[derive(Debug, Clone, Deserialize)]
pub struct SlotsConfig {
    pub schema_version: u32,
    pub endpoints: Endpoints,
    pub slots: HashMap<String, SlotConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Endpoints {
    pub ollama: String,
    pub lm_studio: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SlotConfig {
    pub provider: ProviderKind,
    pub model: String,
    pub mode: SlotMode,
    #[serde(default)]
    pub params: HashMap<String, serde_yaml::Value>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    Ollama,
    LmStudio,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlotMode {
    Generate,
    Chat,
    Embeddings,
}

/// High-level request used by PAT/SAT
#[derive(Debug, Clone)]
pub struct LlmRequest {
    pub slot: String,
    pub prompt: String,
}

#[derive(Debug, Clone)]
pub struct LlmResponse {
    pub text: String,
    pub provider: ProviderKind,
    pub model: String,
}

#[derive(thiserror::Error, Debug)]
pub enum GatewayError {
    #[error("unknown slot: {0}")]
    UnknownSlot(String),
    #[error("provider error: {0}")]
    Provider(String),
    #[error("http error: {0}")]
    Http(String),
    #[error("parse error: {0}")]
    Parse(String),
}

pub struct LlmGateway {
    cfg: SlotsConfig,
    ollama: OllamaClient,
    openai: OpenAiCompatClient,
}

impl LlmGateway {
    pub fn new(cfg: SlotsConfig) -> Self {
        let ollama = OllamaClient::new(cfg.endpoints.ollama.clone());
        let openai = OpenAiCompatClient::new(cfg.endpoints.lm_studio.clone());
        Self { cfg, ollama, openai }
    }

    pub fn slot(&self, name: &str) -> Result<&SlotConfig, GatewayError> {
        self.cfg
            .slots
            .get(name)
            .ok_or_else(|| GatewayError::UnknownSlot(name.to_string()))
    }

    /// Executes an LLM call for a given slot + prompt.
    pub async fn complete(&self, req: LlmRequest) -> Result<LlmResponse, GatewayError> {
        let slot = self.slot(&req.slot)?.clone();
        let text = match slot.provider {
            ProviderKind::Ollama => {
                self.ollama
                    .generate(&slot.model, &req.prompt, &slot.params)
                    .await
                    .map_err(|e| GatewayError::Provider(e.to_string()))?
            }
            ProviderKind::LmStudio => {
                // For Phase 1 we use OpenAI-compatible chat completions; mode can be extended.
                self.openai
                    .chat_completion(&slot.model, &req.prompt, &slot.params)
                    .await
                    .map_err(|e| GatewayError::Provider(e.to_string()))?
            }
        };

        Ok(LlmResponse {
            text,
            provider: slot.provider,
            model: slot.model,
        })
    }
}
