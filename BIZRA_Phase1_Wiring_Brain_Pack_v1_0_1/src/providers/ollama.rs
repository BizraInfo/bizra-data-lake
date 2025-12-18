//! providers/ollama.rs
use std::collections::HashMap;

use reqwest::Client;
use serde_json::json;

#[derive(thiserror::Error, Debug)]
pub enum OllamaError {
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),
    #[error("invalid response: {0}")]
    Invalid(String),
}

pub struct OllamaClient {
    base: String,
    http: Client,
}

impl OllamaClient {
    pub fn new(base: String) -> Self {
        Self { base, http: Client::new() }
    }

    pub async fn generate(
        &self,
        model: &str,
        prompt: &str,
        params: &HashMap<String, serde_yaml::Value>,
    ) -> Result<String, OllamaError> {
        let url = format!("{}/api/generate", self.base.trim_end_matches('/'));
        let mut options = serde_json::Map::new();

        // Pull a few known params if present
        if let Some(t) = params.get("temperature").and_then(|v| v.as_f64()) {
            options.insert("temperature".to_string(), json!(t));
        }
        if let Some(ctx) = params.get("num_ctx").and_then(|v| v.as_i64()) {
            options.insert("num_ctx".to_string(), json!(ctx));
        }

        let body = json!({
            "model": model,
            "prompt": prompt,
            "stream": false,
            "options": options
        });

        let resp = self.http.post(url).json(&body).send().await?;
        let v: serde_json::Value = resp.json().await?;

        // Ollama returns { response: "...", done: true, ... }
        let text = v
            .get("response")
            .and_then(|x| x.as_str())
            .ok_or_else(|| OllamaError::Invalid(v.to_string()))?;

        Ok(text.trim().to_string())
    }
}
