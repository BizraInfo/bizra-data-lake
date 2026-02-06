//! LM Studio Inference Module
//!
//! Connects to LM Studio's v1 REST API for local inference.
//! Supports both native `/api/v1/chat` and OpenAI-compatible endpoints.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use std::env;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};

/// LM Studio configuration
#[derive(Debug, Clone)]
pub struct LMStudioConfig {
    pub host: String,
    pub port: u16,
    pub timeout_secs: u64,
    pub api_key: Option<String>,
}

impl Default for LMStudioConfig {
    fn default() -> Self {
        let api_key = env::var("LMSTUDIO_API_KEY")
            .ok()
            .or_else(|| env::var("LMSTUDIO_TOKEN").ok());

        Self {
            host: "192.168.56.1".to_string(),
            port: 1234,
            timeout_secs: 120,
            api_key,
        }
    }
}

impl LMStudioConfig {
    pub fn base_url(&self) -> String {
        format!("http://{}:{}", self.host, self.port)
    }
}

/// LM Studio client
pub struct LMStudio {
    config: LMStudioConfig,
    client: reqwest::Client,
}

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Native v1 chat request
#[derive(Debug, Serialize)]
struct V1ChatRequest {
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<i32>,
    stream: bool,
}

/// Native v1 chat response
#[derive(Debug, Deserialize)]
pub struct V1ChatResponse {
    pub message: ChatMessage,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub usage: Option<UsageInfo>,
}

/// OpenAI-compatible chat request
#[derive(Debug, Serialize)]
struct OpenAIChatRequest {
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<i32>,
    stream: bool,
}

/// OpenAI-compatible response
#[derive(Debug, Deserialize)]
pub struct OpenAIChatResponse {
    pub id: String,
    pub choices: Vec<OpenAIChoice>,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub usage: Option<UsageInfo>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIChoice {
    pub index: i32,
    pub message: ChatMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// Usage information
#[derive(Debug, Deserialize)]
pub struct UsageInfo {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
}

/// Model information
#[derive(Debug, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub owned_by: String,
}

/// Models list response
#[derive(Debug, Deserialize)]
pub struct ModelsResponse {
    pub data: Vec<ModelInfo>,
}

impl LMStudio {
    /// Create new LM Studio client with default config
    pub fn new() -> Self {
        Self::with_config(LMStudioConfig::default())
    }

    /// Create with custom config
    pub fn with_config(config: LMStudioConfig) -> Self {
        let mut headers = HeaderMap::new();
        if let Some(ref key) = config.api_key {
            let value = format!("Bearer {}", key);
            if let Ok(header_value) = HeaderValue::from_str(&value) {
                headers.insert(AUTHORIZATION, header_value);
            }
        }

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .default_headers(headers)
            .build()
            .expect("Failed to create HTTP client");

        Self { config, client }
    }

    /// Check if LM Studio is reachable
    pub async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/api/v1/models", self.config.base_url());

        match self.client.get(&url).send().await {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    /// List available models
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let url = format!("{}/api/v1/models", self.config.base_url());

        let resp = self.client.get(&url)
            .send()
            .await
            .map_err(|e| anyhow!("Failed to connect to LM Studio: {}", e))?;

        if !resp.status().is_success() {
            return Err(anyhow!("LM Studio returned error: {}", resp.status()));
        }

        let models: ModelsResponse = resp.json().await
            .map_err(|e| anyhow!("Failed to parse models response: {}", e))?;

        Ok(models.data)
    }

    /// Get currently loaded model
    pub async fn current_model(&self) -> Result<Option<String>> {
        let models = self.list_models().await?;
        Ok(models.first().map(|m| m.id.clone()))
    }

    /// Chat using native v1 API (supports stateful chats, MCP)
    pub async fn chat_v1(
        &self,
        messages: Vec<ChatMessage>,
        model: Option<String>,
        temperature: Option<f32>,
        max_tokens: Option<i32>,
    ) -> Result<V1ChatResponse> {
        let url = format!("{}/api/v1/chat", self.config.base_url());

        let request = V1ChatRequest {
            messages,
            model,
            temperature,
            max_tokens,
            stream: false,
        };

        let resp = self.client.post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| anyhow!("Failed to connect to LM Studio: {}", e))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("LM Studio error {}: {}", status, body));
        }

        let response: V1ChatResponse = resp.json().await
            .map_err(|e| anyhow!("Failed to parse chat response: {}", e))?;

        Ok(response)
    }

    /// Chat using OpenAI-compatible API
    pub async fn chat_openai(
        &self,
        messages: Vec<ChatMessage>,
        model: Option<String>,
        temperature: Option<f32>,
        max_tokens: Option<i32>,
    ) -> Result<OpenAIChatResponse> {
        let url = format!("{}/v1/chat/completions", self.config.base_url());

        let request = OpenAIChatRequest {
            messages,
            model,
            temperature,
            max_tokens,
            stream: false,
        };

        let resp = self.client.post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| anyhow!("Failed to connect to LM Studio: {}", e))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("LM Studio error {}: {}", status, body));
        }

        let response: OpenAIChatResponse = resp.json().await
            .map_err(|e| anyhow!("Failed to parse chat response: {}", e))?;

        Ok(response)
    }

    /// Simple chat helper - sends a single user message
    pub async fn ask(&self, prompt: &str) -> Result<String> {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }];

        // Try v1 API first, fall back to OpenAI-compatible
        match self.chat_v1(messages.clone(), None, Some(0.7), Some(2048)).await {
            Ok(resp) => Ok(resp.message.content),
            Err(_) => {
                let resp = self.chat_openai(messages, None, Some(0.7), Some(2048)).await?;
                Ok(resp.choices.first()
                    .map(|c| c.message.content.clone())
                    .unwrap_or_default())
            }
        }
    }

    /// Chat with system prompt (for PAT agents)
    pub async fn chat_with_system(&self, system_prompt: &str, user_message: &str) -> Result<String> {
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: system_prompt.to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: user_message.to_string(),
            },
        ];

        match self.chat_v1(messages.clone(), None, Some(0.7), Some(2048)).await {
            Ok(resp) => Ok(resp.message.content),
            Err(_) => {
                let resp = self.chat_openai(messages, None, Some(0.7), Some(2048)).await?;
                Ok(resp.choices.first()
                    .map(|c| c.message.content.clone())
                    .unwrap_or_default())
            }
        }
    }
}

/// PAT Agent system prompts
pub fn get_agent_system_prompt(agent: &str) -> String {
    match agent.to_lowercase().as_str() {
        "strategist" => r#"You are the Strategist of a Personal Agentic Team (PAT).
Your role is strategic planning and long-term thinking.
Standing on the shoulders of: Sun Tzu, Clausewitz, Michael Porter.
Focus on objectives, competitive advantage, and strategic positioning.
Be concise but thorough. Think in terms of goals, resources, and outcomes."#.to_string(),

        "researcher" => r#"You are the Researcher of a Personal Agentic Team (PAT).
Your role is knowledge discovery and synthesis.
Standing on the shoulders of: Claude Shannon, Alan Turing, Edsger Dijkstra.
Focus on finding accurate information, synthesizing knowledge, and providing insights.
Be thorough in research but clear in presentation."#.to_string(),

        "developer" => r#"You are the Developer of a Personal Agentic Team (PAT).
Your role is code implementation and technical solutions.
Standing on the shoulders of: Donald Knuth, Dennis Ritchie, Linus Torvalds.
Focus on clean code, efficient algorithms, and robust implementation.
Be precise, practical, and security-conscious."#.to_string(),

        "analyst" => r#"You are the Analyst of a Personal Agentic Team (PAT).
Your role is data analysis and insight extraction.
Standing on the shoulders of: John Tukey, Edward Tufte, William Cleveland.
Focus on patterns, trends, and data-driven insights.
Be quantitative, visual, and clear in presenting findings."#.to_string(),

        "reviewer" => r#"You are the Reviewer of a Personal Agentic Team (PAT).
Your role is quality validation and constructive feedback.
Standing on the shoulders of: Michael Fagan, David Parnas, Fred Brooks.
Focus on correctness, completeness, and improvement opportunities.
Be thorough but constructive. Identify issues and suggest solutions."#.to_string(),

        "executor" => r#"You are the Executor of a Personal Agentic Team (PAT).
Your role is task execution and delivery.
Standing on the shoulders of: Toyota Production System, W. Edwards Deming, Taiichi Ohno.
Focus on efficient execution, continuous improvement, and delivering results.
Be action-oriented, efficient, and results-focused."#.to_string(),

        "guardian" | _ => r#"You are the Guardian of a Personal Agentic Team (PAT).
Your role is ethical oversight and protective guidance.
Standing on the shoulders of: Al-Ghazali, John Rawls, Anthropic.
Focus on beneficial outcomes, ethical considerations, and harm prevention.
Apply the FATE gates: Ihsān (excellence ≥0.95), Adl (fairness), Harm (≤0.30), Confidence (≥0.80).
Be wise, protective, and guide toward beneficial outcomes."#.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = LMStudioConfig::default();
        assert_eq!(config.host, "192.168.56.1");
        assert_eq!(config.port, 1234);
        assert_eq!(config.base_url(), "http://192.168.56.1:1234");
    }

    #[test]
    fn test_agent_prompts() {
        let guardian = get_agent_system_prompt("guardian");
        assert!(guardian.contains("Guardian"));
        assert!(guardian.contains("Al-Ghazali"));

        let developer = get_agent_system_prompt("developer");
        assert!(developer.contains("Developer"));
        assert!(developer.contains("Knuth"));
    }
}
