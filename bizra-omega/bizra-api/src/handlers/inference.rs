//! Inference Handlers — Model generation and tier selection

use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use bizra_inference::selector::{ModelTier, TaskComplexity, ModelSelector};
use bizra_inference::gateway::InferenceRequest;
use crate::{state::AppState, error::ApiError};

#[derive(Deserialize)]
pub struct GenerateApiRequest {
    pub prompt: String,
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub tier: Option<String>,
}

fn default_max_tokens() -> usize { 512 }
fn default_temperature() -> f32 { 0.7 }

#[derive(Serialize)]
pub struct GenerateResponse {
    pub request_id: String,
    pub text: String,
    pub model: String,
    pub tier: String,
    pub completion_tokens: usize,
    pub duration_ms: u64,
    pub tokens_per_second: f32,
}

/// Generate text completion
pub async fn generate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GenerateApiRequest>,
) -> Result<Json<GenerateResponse>, ApiError> {
    // Parse tier from string
    let preferred_tier = req.tier.as_ref().map(|t| match t.to_lowercase().as_str() {
        "edge" | "nano" => ModelTier::Edge,
        "local" | "medium" => ModelTier::Local,
        "pool" | "large" => ModelTier::Pool,
        _ => ModelTier::Local,
    });

    // Estimate complexity
    let complexity = TaskComplexity::estimate(&req.prompt, req.max_tokens);

    let inference = state.inference.read().await;

    // If gateway configured, use it
    if let Some(gateway) = inference.as_ref() {
        let request = InferenceRequest {
            id: uuid::Uuid::new_v4().to_string(),
            prompt: req.prompt,
            system: req.system,
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            complexity,
            preferred_tier,
        };

        let response = gateway.infer(request).await
            .map_err(|e| ApiError::InferenceError(e.to_string()))?;

        return Ok(Json(GenerateResponse {
            request_id: response.request_id,
            text: response.text,
            model: response.model,
            tier: format!("{:?}", response.tier),
            completion_tokens: response.completion_tokens,
            duration_ms: response.duration_ms,
            tokens_per_second: response.tokens_per_second,
        }));
    }

    // Determine tier if no gateway
    let selector = ModelSelector::default();
    let tier = preferred_tier.unwrap_or_else(|| selector.select_tier(&complexity));

    // Return placeholder if no gateway configured
    Ok(Json(GenerateResponse {
        request_id: uuid::Uuid::new_v4().to_string(),
        text: format!(
            "[Inference gateway not configured. Would use {:?} tier for: {}...]",
            tier,
            &req.prompt[..req.prompt.len().min(50)]
        ),
        model: "none".into(),
        tier: format!("{:?}", tier),
        completion_tokens: 0,
        duration_ms: 0,
        tokens_per_second: 0.0,
    }))
}

#[derive(Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub tier: String,
    pub parameters: String,
    pub context_length: usize,
    pub available: bool,
}

#[derive(Serialize)]
pub struct ListModelsResponse {
    pub models: Vec<ModelInfo>,
}

/// List available models
pub async fn list_models() -> Json<ListModelsResponse> {
    // Default model catalog
    let models = vec![
        ModelInfo {
            name: "qwen2.5-0.5b".into(),
            tier: "Edge".into(),
            parameters: "0.5B".into(),
            context_length: 32768,
            available: true,
        },
        ModelInfo {
            name: "qwen2.5-1.5b".into(),
            tier: "Edge".into(),
            parameters: "1.5B".into(),
            context_length: 32768,
            available: true,
        },
        ModelInfo {
            name: "qwen2.5-7b".into(),
            tier: "Local".into(),
            parameters: "7B".into(),
            context_length: 32768,
            available: true,
        },
        ModelInfo {
            name: "llama-3.2-3b".into(),
            tier: "Local".into(),
            parameters: "3B".into(),
            context_length: 128000,
            available: true,
        },
        ModelInfo {
            name: "deepseek-r1-14b".into(),
            tier: "Local".into(),
            parameters: "14B".into(),
            context_length: 65536,
            available: false,
        },
        ModelInfo {
            name: "qwen2.5-72b".into(),
            tier: "Pool".into(),
            parameters: "72B".into(),
            context_length: 32768,
            available: false,
        },
    ];

    Json(ListModelsResponse { models })
}

#[derive(Deserialize)]
pub struct SelectTierRequest {
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    pub latency_sensitive: bool,
}

#[derive(Serialize)]
pub struct SelectTierResponse {
    pub recommended_tier: String,
    pub complexity: String,
    pub reasoning: String,
}

/// Select optimal tier for a task
pub async fn select_tier(
    Json(req): Json<SelectTierRequest>,
) -> Json<SelectTierResponse> {
    let selector = ModelSelector::default();
    let complexity = TaskComplexity::estimate(&req.prompt, req.max_tokens);
    let tier = selector.select_tier(&complexity);

    let reasoning = match (&complexity, req.latency_sensitive) {
        (TaskComplexity::Simple, true) => "Simple task with latency requirement → Edge tier",
        (TaskComplexity::Simple, false) => "Simple task, no latency constraint → Edge tier",
        (TaskComplexity::Medium, true) => "Medium complexity, latency sensitive → Edge tier",
        (TaskComplexity::Medium, false) => "Medium complexity → Edge tier",
        (TaskComplexity::Complex, _) => "Complex task requiring deep reasoning → Local tier",
        (TaskComplexity::Expert, _) => "Expert-level task → Pool tier with federation",
    };

    Json(SelectTierResponse {
        recommended_tier: format!("{:?}", tier),
        complexity: format!("{:?}", complexity),
        reasoning: reasoning.into(),
    })
}
