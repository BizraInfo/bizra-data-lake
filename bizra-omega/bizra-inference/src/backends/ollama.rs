//! Ollama Backend

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use super::{Backend, BackendConfig, BackendError};
use crate::gateway::{InferenceRequest, InferenceResponse};
use crate::selector::ModelTier;

#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    stream: bool,
    options: OllamaOptions,
}

#[derive(Serialize)]
struct OllamaOptions {
    temperature: f32,
    num_predict: i32,
}

#[derive(Deserialize)]
struct OllamaResponse {
    response: String,
    total_duration: Option<u64>,
    eval_count: Option<usize>,
}

pub struct OllamaBackend {
    config: BackendConfig,
    base_url: String,
    client: reqwest::Client,
}

impl OllamaBackend {
    pub fn new(config: BackendConfig, base_url: Option<&str>) -> Self {
        Self {
            config,
            base_url: base_url.unwrap_or("http://localhost:11434").into(),
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl Backend for OllamaBackend {
    fn name(&self) -> &str { &self.config.name }

    async fn generate(&self, request: &InferenceRequest) -> Result<InferenceResponse, BackendError> {
        let url = format!("{}/api/generate", self.base_url);

        let req = OllamaRequest {
            model: self.config.model.clone(),
            prompt: request.prompt.clone(),
            stream: false,
            options: OllamaOptions {
                temperature: request.temperature,
                num_predict: request.max_tokens as i32,
            },
        };

        let response = self.client.post(&url).json(&req).send().await
            .map_err(|e| BackendError::Connection(e.to_string()))?;

        if !response.status().is_success() {
            return Err(BackendError::Generation(format!("Status: {}", response.status())));
        }

        let resp: OllamaResponse = response.json().await
            .map_err(|e| BackendError::Generation(e.to_string()))?;

        Ok(InferenceResponse {
            request_id: request.id.clone(),
            text: resp.response,
            model: self.config.model.clone(),
            tier: ModelTier::Local,
            completion_tokens: resp.eval_count.unwrap_or(0),
            duration_ms: resp.total_duration.unwrap_or(0) / 1_000_000,
            tokens_per_second: 0.0,
        })
    }

    async fn health_check(&self) -> bool {
        self.client.get(format!("{}/api/tags", self.base_url)).send().await.is_ok()
    }
}
