//! providers/openai_compat.rs
use std::collections::HashMap;

use reqwest::Client;
use serde_json::json;

#[derive(thiserror::Error, Debug)]
pub enum OpenAiCompatError {
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),
    #[error("invalid response: {0}")]
    Invalid(String),
}

pub struct OpenAiCompatClient {
    base_v1: String,
    http: Client,
}

impl OpenAiCompatClient {
    pub fn new(base_v1: String) -> Self {
        Self { base_v1, http: Client::new() }
    }

    pub async fn chat_completion(
        &self,
        model: &str,
        prompt: &str,
        params: &HashMap<String, serde_yaml::Value>,
    ) -> Result<String, OpenAiCompatError> {
        let url = format!("{}/chat/completions", self.base_v1.trim_end_matches('/'));
        let temperature = params.get("temperature").and_then(|v| v.as_f64()).unwrap_or(0.4);

        let body = json!({
            "model": model,
            "messages": [
                {"role": "system", "content": "You are BIZRA Node0. Answer concisely and correctly."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "stream": false
        });

        let resp = self.http.post(url).json(&body).send().await?;
        let v: serde_json::Value = resp.json().await?;

        let text = v
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c0| c0.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|s| s.as_str())
            .ok_or_else(|| OpenAiCompatError::Invalid(v.to_string()))?;

        Ok(text.trim().to_string())
    }
}
