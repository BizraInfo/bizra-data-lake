// src/lmstudio.rs - LM Studio v1 API Client
//
// Provides local LM Studio integration with support for:
// - Native v1 API (/api/v1/*) - LM Studio 0.4.0+ with MCP, stateful chats
// - OpenAI-compatible endpoints (/v1/*) - Fallback for older versions
//
// Environment variables:
// - LMSTUDIO_URL: Base URL (default: http://localhost:1234)
// - LMSTUDIO_API_KEY: Optional bearer token for authentication
// - LMSTUDIO_API_VERSION: "v1" (native) or "openai" (compatible)

use crate::ollama::{ChatMessage, GenerationOptions};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::OnceCell;
use tracing::{debug, info, warn};

const DEFAULT_LMSTUDIO_URL: &str = "http://localhost:1234";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

// API endpoint paths
const NATIVE_CHAT_ENDPOINT: &str = "/api/v1/chat";
const NATIVE_MODELS_ENDPOINT: &str = "/api/v1/models";
const NATIVE_LOAD_ENDPOINT: &str = "/api/v1/models/load";
const NATIVE_UNLOAD_ENDPOINT: &str = "/api/v1/models/unload";
const OPENAI_CHAT_ENDPOINT: &str = "/v1/chat/completions";
const OPENAI_MODELS_ENDPOINT: &str = "/v1/models";

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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ApiVersion {
    /// Native LM Studio v1 API (/api/v1/*) - LM Studio 0.4.0+
    Native,
    /// OpenAI-compatible API (/v1/*) - Fallback for older versions
    OpenAICompat,
}

impl Default for ApiVersion {
    fn default() -> Self {
        ApiVersion::Native // Prefer native API for new features
    }
}

pub struct LmStudioClient {
    base_url: String,
    api_key: Option<String>,
    api_version: ApiVersion,
    http: Client,
    connected: bool,
}

impl LmStudioClient {
    pub fn new(base_url: String, api_key: Option<String>, api_version: ApiVersion) -> Self {
        let http = Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .expect("Failed to create HTTP client");

        Self {
            base_url,
            api_key,
            api_version,
            http,
            connected: false,
        }
    }

    pub async fn from_env() -> Self {
        let base_url =
            std::env::var("LMSTUDIO_URL").unwrap_or_else(|_| DEFAULT_LMSTUDIO_URL.to_string());
        let api_key = std::env::var("LMSTUDIO_API_KEY")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());

        // Determine API version from env or auto-detect
        let api_version = match std::env::var("LMSTUDIO_API_VERSION")
            .unwrap_or_else(|_| "native".to_string())
            .to_lowercase()
            .as_str()
        {
            "openai" | "compat" | "v1" => ApiVersion::OpenAICompat,
            _ => ApiVersion::Native,
        };

        let mut client = Self::new(base_url.clone(), api_key, api_version);

        // Try native API first, fall back to OpenAI-compat if needed
        match client.health_check().await {
            Ok(_) => {
                info!(
                    "🧪 LM Studio connected at {} (API: {:?})",
                    base_url, api_version
                );
                client.connected = true;
            }
            Err(e) => {
                if api_version == ApiVersion::Native {
                    debug!("Native API failed, trying OpenAI-compat fallback");
                    client.api_version = ApiVersion::OpenAICompat;
                    match client.health_check().await {
                        Ok(_) => {
                            info!(
                                "🧪 LM Studio connected at {} (OpenAI-compat fallback)",
                                base_url
                            );
                            client.connected = true;
                        }
                        Err(e2) => {
                            warn!(
                                "⚠️ LM Studio not available at {}: native={}, compat={}",
                                base_url, e, e2
                            );
                            client.connected = false;
                        }
                    }
                } else {
                    warn!("⚠️ LM Studio not available at {}: {}", base_url, e);
                    client.connected = false;
                }
            }
        }

        client
    }

    /// Get the current API version in use
    pub fn api_version(&self) -> ApiVersion {
        self.api_version
    }

    pub fn is_connected(&self) -> bool {
        self.connected
    }

    async fn health_check(&self) -> Result<(), LmStudioError> {
        self.list_models().await.map(|_| ())
    }

    pub async fn list_models(&self) -> Result<Vec<String>, LmStudioError> {
        let endpoint = match self.api_version {
            ApiVersion::Native => NATIVE_MODELS_ENDPOINT,
            ApiVersion::OpenAICompat => OPENAI_MODELS_ENDPOINT,
        };
        let url = format!("{}{}", self.base_url.trim_end_matches('/'), endpoint);

        let mut request = self.http.get(&url);
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }

        let response = request.send().await?;
        if !response.status().is_success() {
            return Err(LmStudioError::InvalidResponse(format!(
                "LM Studio {} HTTP {}",
                endpoint,
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
        let endpoint = match self.api_version {
            ApiVersion::Native => NATIVE_CHAT_ENDPOINT,
            ApiVersion::OpenAICompat => OPENAI_CHAT_ENDPOINT,
        };
        let url = format!("{}{}", self.base_url.trim_end_matches('/'), endpoint);

        let request_body = ChatCompletionRequest {
            model: model.to_string(),
            messages,
            temperature: options.temperature,
            max_tokens: options.num_predict,
            top_p: options.top_p,
            seed: options.seed,
        };

        let mut request = self.http.post(&url).json(&request_body);
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }

        let response = request.send().await?;
        if !response.status().is_success() {
            return Err(LmStudioError::InvalidResponse(format!(
                "LM Studio {} HTTP {}",
                endpoint,
                response.status()
            )));
        }

        let parsed: ChatCompletionResponse = response.json().await?;
        let message = parsed
            .choices
            .first()
            .map(|c| c.message.clone())
            .ok_or_else(|| LmStudioError::InvalidResponse("Missing choices[0]".to_string()))?;

        Ok(LmStudioChatResponse {
            message,
            prompt_tokens: parsed.usage.as_ref().and_then(|u| u.prompt_tokens),
            completion_tokens: parsed.usage.as_ref().and_then(|u| u.completion_tokens),
        })
    }

    /// Load a model into memory (Native API only, LM Studio 0.4.0+)
    pub async fn load_model(&self, model_id: &str) -> Result<(), LmStudioError> {
        if self.api_version != ApiVersion::Native {
            return Err(LmStudioError::InvalidResponse(
                "load_model requires Native API (LM Studio 0.4.0+)".to_string(),
            ));
        }

        let url = format!(
            "{}{}",
            self.base_url.trim_end_matches('/'),
            NATIVE_LOAD_ENDPOINT
        );

        #[derive(Serialize)]
        struct LoadRequest<'a> {
            model: &'a str,
        }

        let mut request = self.http.post(&url).json(&LoadRequest { model: model_id });
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }

        let response = request.send().await?;
        if !response.status().is_success() {
            return Err(LmStudioError::InvalidResponse(format!(
                "LM Studio {} HTTP {}",
                NATIVE_LOAD_ENDPOINT,
                response.status()
            )));
        }

        info!("✓ Model {} loaded", model_id);
        Ok(())
    }

    /// Unload a model from memory (Native API only, LM Studio 0.4.0+)
    pub async fn unload_model(&self, model_id: &str) -> Result<(), LmStudioError> {
        if self.api_version != ApiVersion::Native {
            return Err(LmStudioError::InvalidResponse(
                "unload_model requires Native API (LM Studio 0.4.0+)".to_string(),
            ));
        }

        let url = format!(
            "{}{}",
            self.base_url.trim_end_matches('/'),
            NATIVE_UNLOAD_ENDPOINT
        );

        #[derive(Serialize)]
        struct UnloadRequest<'a> {
            model: &'a str,
        }

        let mut request = self
            .http
            .post(&url)
            .json(&UnloadRequest { model: model_id });
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }

        let response = request.send().await?;
        if !response.status().is_success() {
            return Err(LmStudioError::InvalidResponse(format!(
                "LM Studio {} HTTP {}",
                NATIVE_UNLOAD_ENDPOINT,
                response.status()
            )));
        }

        info!("✓ Model {} unloaded", model_id);
        Ok(())
    }
}
