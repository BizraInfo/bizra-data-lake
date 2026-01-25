// src/lmstudio.rs - LM Studio OpenAI-Compatible Client
//
// Provides local LM Studio integration using OpenAI-compatible endpoints.
// Environment variables:
// - LMSTUDIO_URL: Base URL (default: http://localhost:1234)
// - LMSTUDIO_API_KEY: Optional bearer token

use crate::ollama::{ChatMessage, GenerationOptions};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::OnceCell;
use tracing::{info, warn};

const DEFAULT_LMSTUDIO_URL: &str = "http://localhost:1234";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

static LMSTUDIO_CLIENT: OnceCell<Arc<LmStudioClient>> = OnceCell::const_new();

pub async fn get_lmstudio() -> Arc<LmStudioClient> {
    LMSTUDIO_CLIENT
        .get_or_init(|| async {
            let client = LmStudioClient::from_env().await;
            Arc::new(client)
        })
        .await
        .clone()
}

#[derive(Debug, thiserror::Error)]
pub enum LmStudioError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Invalid response: {0}")]
    InvalidResponse(String),
}

#[derive(Debug, Clone, Deserialize)]
struct ModelsResponse {
    data: Vec<ModelItem>,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelItem {
    id: String,
}

#[derive(Debug, Clone, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
    usage: Option<Usage>,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

#[derive(Debug, Clone, Deserialize)]
struct Usage {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct LmStudioChatResponse {
    pub message: ChatMessage,
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
}

pub struct LmStudioClient {
    base_url: String,
    api_key: Option<String>,
    http: Client,
    connected: bool,
}

impl LmStudioClient {
    pub fn new(base_url: String, api_key: Option<String>) -> Self {
        let http = Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .expect("Failed to create HTTP client");

        Self {
            base_url,
            api_key,
            http,
            connected: false,
        }
    }

    pub async fn from_env() -> Self {
        let base_url = std::env::var("LMSTUDIO_URL").unwrap_or_else(|_| DEFAULT_LMSTUDIO_URL.to_string());
        let api_key = std::env::var("LMSTUDIO_API_KEY")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());

        let mut client = Self::new(base_url.clone(), api_key);

        match client.health_check().await {
            Ok(_) => {
                info!("🧪 LM Studio connected at {}", base_url);
                client.connected = true;
            }
            Err(e) => {
                warn!("⚠️ LM Studio not available at {}: {}", base_url, e);
                client.connected = false;
            }
        }

        client
    }

    pub fn is_connected(&self) -> bool {
        self.connected
    }

    async fn health_check(&self) -> Result<(), LmStudioError> {
        self.list_models().await.map(|_| ())
    }

    pub async fn list_models(&self) -> Result<Vec<String>, LmStudioError> {
        let url = format!("{}/v1/models", self.base_url.trim_end_matches('/'));
        let mut request = self.http.get(url);
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }

        let response = request.send().await?;
        if !response.status().is_success() {
            return Err(LmStudioError::InvalidResponse(format!(
                "LM Studio /v1/models HTTP {}",
                response.status()
            )));
        }

        let models: ModelsResponse = response.json().await?;
        Ok(models.data.into_iter().map(|m| m.id).collect())
    }

    pub async fn chat_completion(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        options: GenerationOptions,
    ) -> Result<LmStudioChatResponse, LmStudioError> {
        let url = format!("{}/v1/chat/completions", self.base_url.trim_end_matches('/'));
        let request_body = ChatCompletionRequest {
            model: model.to_string(),
            messages,
            temperature: options.temperature,
            max_tokens: options.num_predict,
            top_p: options.top_p,
            seed: options.seed,
        };

        let mut request = self.http.post(url).json(&request_body);
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }

        let response = request.send().await?;
        if !response.status().is_success() {
            return Err(LmStudioError::InvalidResponse(format!(
                "LM Studio /v1/chat/completions HTTP {}",
                response.status()
            )));
        }

        let parsed: ChatCompletionResponse = response.json().await?;
        let message = parsed
            .choices.first()
            .map(|c| c.message.clone())
            .ok_or_else(|| LmStudioError::InvalidResponse("Missing choices[0]".to_string()))?;

        Ok(LmStudioChatResponse {
            message,
            prompt_tokens: parsed.usage.as_ref().and_then(|u| u.prompt_tokens),
            completion_tokens: parsed.usage.as_ref().and_then(|u| u.completion_tokens),
        })
    }
}
