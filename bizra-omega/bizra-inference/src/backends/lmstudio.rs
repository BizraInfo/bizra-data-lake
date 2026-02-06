//! LM Studio Backend — Primary inference backend for Node0
//!
//! Supports:
//! - Reasoning models (DeepSeek, Qwen)
//! - Agentic models (function calling, tool use)
//! - Vision models (LLaVA, Qwen-VL)
//! - Voice models (Whisper, TTS)
//!
//! LM Studio runs at 192.168.56.1:1234 with OpenAI-compatible API

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, error, info, warn};

use super::{Backend, BackendError};
use crate::gateway::{InferenceRequest, InferenceResponse};
use crate::selector::ModelTier;

/// LM Studio host configuration
const DEFAULT_LMSTUDIO_HOST: &str = "192.168.56.1";
const DEFAULT_LMSTUDIO_PORT: u16 = 1234;

/// Model capability categories
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelCapability {
    /// Deep reasoning (DeepSeek-R1, Qwen-2.5-72B)
    Reasoning,
    /// Agentic with function calling
    Agentic,
    /// Vision/multimodal (LLaVA, Qwen-VL)
    Vision,
    /// Voice/audio (Whisper, Moshi)
    Voice,
    /// General chat
    Chat,
    /// Code generation
    Code,
    /// Embedding generation
    Embedding,
}

/// Model configuration for LM Studio
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LMStudioModel {
    pub id: String,
    pub name: String,
    pub capabilities: Vec<ModelCapability>,
    pub context_length: usize,
    pub is_loaded: bool,
}

/// LM Studio backend configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LMStudioConfig {
    pub host: String,
    pub port: u16,
    pub timeout_secs: u64,
    /// Prefer these models by capability
    pub model_preferences: ModelPreferences,
    /// Enable streaming responses
    pub streaming: bool,
    /// Maximum retries on failure
    pub max_retries: u32,
}

impl Default for LMStudioConfig {
    fn default() -> Self {
        Self {
            host: std::env::var("LMSTUDIO_HOST")
                .unwrap_or_else(|_| DEFAULT_LMSTUDIO_HOST.to_string()),
            port: std::env::var("LMSTUDIO_PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(DEFAULT_LMSTUDIO_PORT),
            timeout_secs: 120,
            model_preferences: ModelPreferences::default(),
            streaming: true,
            max_retries: 3,
        }
    }
}

/// Model preferences by capability
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelPreferences {
    /// Model for reasoning tasks
    pub reasoning: String,
    /// Model for agentic tasks
    pub agentic: String,
    /// Model for vision tasks
    pub vision: String,
    /// Model for voice/transcription
    pub voice: String,
    /// Model for code
    pub code: String,
    /// Model for embeddings
    pub embedding: String,
    /// Default/fallback model
    pub default: String,
}

impl Default for ModelPreferences {
    fn default() -> Self {
        Self {
            // Your local models - adjust to match your LM Studio setup
            reasoning: "deepseek-r1-distill-qwen-32b".to_string(),
            agentic: "qwen2.5-32b-instruct".to_string(),
            vision: "llava-v1.6-mistral-7b".to_string(),
            voice: "whisper-large-v3".to_string(),
            code: "qwen2.5-coder-32b".to_string(),
            embedding: "nomic-embed-text".to_string(),
            default: "qwen2.5-7b-instruct".to_string(),
        }
    }
}

/// OpenAI-compatible message format
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: MessageContent,
}

/// Content can be text or multimodal
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Multimodal(Vec<ContentPart>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub content_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<ImageUrl>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Tool definition for agentic models
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDef,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Chat completion request
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
}

/// Chat completion response
#[derive(Clone, Debug, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Choice {
    pub index: usize,
    pub message: ResponseMessage,
    pub finish_reason: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ResponseMessage {
    pub role: String,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Clone, Debug, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Models list response
#[derive(Clone, Debug, Deserialize)]
pub struct ModelsResponse {
    pub data: Vec<ModelInfo>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub owned_by: String,
}

/// LM Studio Backend
pub struct LMStudioBackend {
    config: LMStudioConfig,
    client: reqwest::Client,
    loaded_models: tokio::sync::RwLock<Vec<LMStudioModel>>,
}

impl LMStudioBackend {
    pub fn new(config: LMStudioConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            config,
            client,
            loaded_models: tokio::sync::RwLock::new(Vec::new()),
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(LMStudioConfig::default())
    }

    fn base_url(&self) -> String {
        format!("http://{}:{}/v1", self.config.host, self.config.port)
    }

    /// Refresh loaded models from LM Studio
    pub async fn refresh_models(&self) -> Result<Vec<LMStudioModel>, BackendError> {
        let url = format!("{}/models", self.base_url());

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| BackendError::Connection(e.to_string()))?;

        if !response.status().is_success() {
            return Err(BackendError::Connection(format!(
                "LM Studio returned status {}",
                response.status()
            )));
        }

        let models_response: ModelsResponse = response
            .json()
            .await
            .map_err(|e| BackendError::Generation(e.to_string()))?;

        let models: Vec<LMStudioModel> = models_response
            .data
            .iter()
            .map(|m| {
                let capabilities = Self::infer_capabilities(&m.id);
                LMStudioModel {
                    id: m.id.clone(),
                    name: m.id.clone(),
                    capabilities,
                    context_length: Self::infer_context_length(&m.id),
                    is_loaded: true,
                }
            })
            .collect();

        *self.loaded_models.write().await = models.clone();
        info!("Refreshed {} models from LM Studio", models.len());

        Ok(models)
    }

    /// Infer model capabilities from name
    fn infer_capabilities(model_id: &str) -> Vec<ModelCapability> {
        let id_lower = model_id.to_lowercase();
        let mut caps = vec![ModelCapability::Chat];

        if id_lower.contains("deepseek") || id_lower.contains("r1") {
            caps.push(ModelCapability::Reasoning);
        }
        if id_lower.contains("coder")
            || id_lower.contains("codellama")
            || id_lower.contains("starcoder")
        {
            caps.push(ModelCapability::Code);
        }
        if id_lower.contains("llava") || id_lower.contains("qwen-vl") || id_lower.contains("vision")
        {
            caps.push(ModelCapability::Vision);
        }
        if id_lower.contains("whisper") || id_lower.contains("moshi") || id_lower.contains("voice")
        {
            caps.push(ModelCapability::Voice);
        }
        if id_lower.contains("instruct") || id_lower.contains("chat") {
            caps.push(ModelCapability::Agentic);
        }
        if id_lower.contains("embed") || id_lower.contains("nomic") || id_lower.contains("bge") {
            caps.push(ModelCapability::Embedding);
        }

        caps
    }

    /// Infer context length from model name
    fn infer_context_length(model_id: &str) -> usize {
        let id_lower = model_id.to_lowercase();
        if id_lower.contains("128k") || id_lower.contains("128000") {
            131072
        } else if id_lower.contains("32k") || id_lower.contains("32000") {
            32768
        } else if id_lower.contains("16k") || id_lower.contains("16000") {
            16384
        } else if id_lower.contains("8k") || id_lower.contains("8000") {
            8192
        } else {
            4096 // Default
        }
    }

    /// Select best model for capability
    pub async fn select_model(&self, capability: ModelCapability) -> String {
        let models = self.loaded_models.read().await;

        // Try to find a loaded model with the capability
        for model in models.iter() {
            if model.capabilities.contains(&capability) {
                return model.id.clone();
            }
        }

        // Fall back to preferences
        match capability {
            ModelCapability::Reasoning => self.config.model_preferences.reasoning.clone(),
            ModelCapability::Agentic => self.config.model_preferences.agentic.clone(),
            ModelCapability::Vision => self.config.model_preferences.vision.clone(),
            ModelCapability::Voice => self.config.model_preferences.voice.clone(),
            ModelCapability::Code => self.config.model_preferences.code.clone(),
            ModelCapability::Embedding => self.config.model_preferences.embedding.clone(),
            ModelCapability::Chat => self.config.model_preferences.default.clone(),
        }
    }

    /// Chat completion (standard)
    pub async fn chat(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, BackendError> {
        let url = format!("{}/chat/completions", self.base_url());

        debug!("LM Studio chat request: model={}", request.model);

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| BackendError::Connection(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            error!("LM Studio error {}: {}", status, body);
            return Err(BackendError::Generation(format!(
                "HTTP {}: {}",
                status, body
            )));
        }

        response
            .json()
            .await
            .map_err(|e| BackendError::Generation(e.to_string()))
    }

    /// Vision request with image
    pub async fn vision(
        &self,
        prompt: &str,
        image_base64: &str,
        max_tokens: usize,
    ) -> Result<String, BackendError> {
        let model = self.select_model(ModelCapability::Vision).await;

        let request = ChatCompletionRequest {
            model,
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: MessageContent::Multimodal(vec![
                    ContentPart {
                        content_type: "text".to_string(),
                        text: Some(prompt.to_string()),
                        image_url: None,
                    },
                    ContentPart {
                        content_type: "image_url".to_string(),
                        text: None,
                        image_url: Some(ImageUrl {
                            url: format!("data:image/jpeg;base64,{}", image_base64),
                            detail: Some("high".to_string()),
                        }),
                    },
                ]),
            }],
            max_tokens: Some(max_tokens),
            temperature: Some(0.7),
            stream: Some(false),
            tools: None,
            tool_choice: None,
        };

        let response = self.chat(request).await?;
        Ok(response
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default())
    }

    /// Agentic request with tools
    pub async fn agentic(
        &self,
        system: &str,
        prompt: &str,
        tools: Vec<Tool>,
        max_tokens: usize,
    ) -> Result<ChatCompletionResponse, BackendError> {
        let model = self.select_model(ModelCapability::Agentic).await;

        let request = ChatCompletionRequest {
            model,
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: MessageContent::Text(system.to_string()),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: MessageContent::Text(prompt.to_string()),
                },
            ],
            max_tokens: Some(max_tokens),
            temperature: Some(0.3), // Lower for agentic tasks
            stream: Some(false),
            tools: Some(tools),
            tool_choice: Some("auto".to_string()),
        };

        self.chat(request).await
    }

    /// Reasoning request (for complex analysis)
    pub async fn reason(
        &self,
        system: &str,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<String, BackendError> {
        let model = self.select_model(ModelCapability::Reasoning).await;

        let request = ChatCompletionRequest {
            model,
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: MessageContent::Text(system.to_string()),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: MessageContent::Text(prompt.to_string()),
                },
            ],
            max_tokens: Some(max_tokens),
            temperature: Some(0.1), // Very low for reasoning
            stream: Some(false),
            tools: None,
            tool_choice: None,
        };

        let response = self.chat(request).await?;
        Ok(response
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default())
    }

    /// Generate embeddings
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, BackendError> {
        let model = self.select_model(ModelCapability::Embedding).await;
        let url = format!("{}/embeddings", self.base_url());

        #[derive(Serialize)]
        struct EmbedRequest {
            model: String,
            input: String,
        }

        #[derive(Deserialize)]
        struct EmbedResponse {
            data: Vec<EmbedData>,
        }

        #[derive(Deserialize)]
        struct EmbedData {
            embedding: Vec<f32>,
        }

        let response = self
            .client
            .post(&url)
            .json(&EmbedRequest {
                model,
                input: text.to_string(),
            })
            .send()
            .await
            .map_err(|e| BackendError::Connection(e.to_string()))?;

        if !response.status().is_success() {
            return Err(BackendError::Generation(format!(
                "Embedding failed: {}",
                response.status()
            )));
        }

        let embed_response: EmbedResponse = response
            .json()
            .await
            .map_err(|e| BackendError::Generation(e.to_string()))?;

        embed_response
            .data
            .first()
            .map(|d| d.embedding.clone())
            .ok_or(BackendError::Generation("No embedding returned".into()))
    }
}

#[async_trait]
impl Backend for LMStudioBackend {
    fn name(&self) -> &str {
        "lmstudio"
    }

    async fn generate(
        &self,
        request: &InferenceRequest,
    ) -> Result<InferenceResponse, BackendError> {
        // Determine capability from request
        let capability = if request.prompt.contains("[VISION]") {
            ModelCapability::Vision
        } else if request.prompt.contains("[REASON]")
            || request.complexity == crate::selector::TaskComplexity::Expert
        {
            ModelCapability::Reasoning
        } else if request.prompt.contains("[CODE]") {
            ModelCapability::Code
        } else {
            ModelCapability::Chat
        };

        let model = self.select_model(capability).await;

        let messages = if let Some(ref system) = request.system {
            vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: MessageContent::Text(system.clone()),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: MessageContent::Text(request.prompt.clone()),
                },
            ]
        } else {
            vec![ChatMessage {
                role: "user".to_string(),
                content: MessageContent::Text(request.prompt.clone()),
            }]
        };

        let chat_request = ChatCompletionRequest {
            model: model.clone(),
            messages,
            max_tokens: Some(request.max_tokens),
            temperature: Some(request.temperature),
            stream: Some(false),
            tools: None,
            tool_choice: None,
        };

        let response = self.chat(chat_request).await?;

        let text = response
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default();

        let completion_tokens = response
            .usage
            .map(|u| u.completion_tokens)
            .unwrap_or(text.split_whitespace().count());

        Ok(InferenceResponse {
            request_id: request.id.clone(),
            text,
            model,
            tier: request.preferred_tier.unwrap_or(ModelTier::Local),
            completion_tokens,
            duration_ms: 0,         // Set by gateway
            tokens_per_second: 0.0, // Set by gateway
        })
    }

    async fn health_check(&self) -> bool {
        let url = format!("{}/models", self.base_url());
        match self.client.get(&url).send().await {
            Ok(response) => response.status().is_success(),
            Err(e) => {
                warn!("LM Studio health check failed: {}", e);
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capability_inference() {
        assert!(LMStudioBackend::infer_capabilities("deepseek-r1-32b")
            .contains(&ModelCapability::Reasoning));
        assert!(
            LMStudioBackend::infer_capabilities("llava-1.5-7b").contains(&ModelCapability::Vision)
        );
        assert!(LMStudioBackend::infer_capabilities("qwen2.5-coder-7b")
            .contains(&ModelCapability::Code));
        assert!(LMStudioBackend::infer_capabilities("whisper-large-v3")
            .contains(&ModelCapability::Voice));
    }

    #[test]
    fn test_context_length_inference() {
        assert_eq!(LMStudioBackend::infer_context_length("qwen-128k"), 131072);
        assert_eq!(LMStudioBackend::infer_context_length("model-32k"), 32768);
        assert_eq!(LMStudioBackend::infer_context_length("default-model"), 4096);
    }

    #[test]
    fn test_default_config() {
        let config = LMStudioConfig::default();
        assert_eq!(config.host, "192.168.56.1");
        assert_eq!(config.port, 1234);
    }
}
