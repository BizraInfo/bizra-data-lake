//! Provider detection
//!
//! Probes local LLM backends (Ollama, LM Studio) to determine the best
//! available inference provider and model for a new Alpha-100 install.

use serde::Deserialize;

/// Result of probing a local inference backend.
#[derive(Debug, Clone)]
pub struct ProbeResult {
    /// Whether the backend is reachable
    pub available: bool,
    /// Model identifiers found on the backend
    pub models: Vec<String>,
    /// The endpoint that was probed
    pub endpoint: String,
}

/// Ollama API response shape for /api/tags
#[derive(Deserialize)]
struct OllamaTagsResponse {
    #[serde(default)]
    models: Vec<OllamaModel>,
}

#[derive(Deserialize)]
struct OllamaModel {
    name: String,
}

/// LM Studio / OpenAI-compatible response shape for /v1/models
#[derive(Deserialize)]
struct LMStudioModelsResponse {
    #[serde(default)]
    data: Vec<LMStudioModel>,
}

#[derive(Deserialize)]
struct LMStudioModel {
    id: String,
}

/// Probe Ollama at localhost:11434 for available models.
pub fn probe_ollama() -> ProbeResult {
    let endpoint = "http://localhost:11434/api/tags".to_string();

    let response = match ureq::get(&endpoint)
        .timeout(std::time::Duration::from_secs(3))
        .call()
    {
        Ok(resp) => resp,
        Err(_) => {
            return ProbeResult {
                available: false,
                models: vec![],
                endpoint,
            };
        }
    };

    let parsed: Result<OllamaTagsResponse, _> = response.into_json();
    match parsed {
        Ok(tags) => ProbeResult {
            available: true,
            models: tags.models.into_iter().map(|m| m.name).collect(),
            endpoint,
        },
        Err(_) => ProbeResult {
            available: false,
            models: vec![],
            endpoint,
        },
    }
}

/// Probe LM Studio at localhost:1234 for available models.
pub fn probe_lmstudio() -> ProbeResult {
    let endpoint = "http://localhost:1234/v1/models".to_string();

    let response = match ureq::get(&endpoint)
        .timeout(std::time::Duration::from_secs(3))
        .call()
    {
        Ok(resp) => resp,
        Err(_) => {
            return ProbeResult {
                available: false,
                models: vec![],
                endpoint,
            };
        }
    };

    let parsed: Result<LMStudioModelsResponse, _> = response.into_json();
    match parsed {
        Ok(data) => ProbeResult {
            available: true,
            models: data.data.into_iter().map(|m| m.id).collect(),
            endpoint,
        },
        Err(_) => ProbeResult {
            available: false,
            models: vec![],
            endpoint,
        },
    }
}

/// Detect the best available provider by probing backends in order:
/// 1. Ollama (preferred — lighter, native model management)
/// 2. LM Studio (fallback — OpenAI-compatible)
/// 3. Default (offline fallback)
///
/// Returns `(provider, backend, model)`.
pub fn detect_best_provider() -> (String, String, String) {
    let ollama = probe_ollama();
    if ollama.available {
        let model = ollama
            .models
            .first()
            .cloned()
            .unwrap_or_else(|| "llama3.1:8b".to_string());
        return ("local".to_string(), "ollama".to_string(), model);
    }

    let lmstudio = probe_lmstudio();
    if lmstudio.available {
        let model = lmstudio
            .models
            .first()
            .cloned()
            .unwrap_or_else(|| "default".to_string());
        return ("local".to_string(), "lmstudio".to_string(), model);
    }

    // Neither available — return safe defaults
    ("local".to_string(), "ollama".to_string(), "llama3.1:8b".to_string())
}

/// Recommend a model size based on available VRAM.
///
/// Tiers:
/// - >= 12 GB VRAM: qwen2.5-14b (high capacity)
/// - >= 6 GB VRAM:  llama3.1:8b (balanced)
/// - >= 3 GB VRAM:  qwen2.5-1.5b (lightweight)
/// - < 3 GB VRAM:   qwen2.5-0.5b (ultra-light)
pub fn recommend_model(vram_gb: f64) -> String {
    if vram_gb >= 12.0 {
        "qwen2.5-14b".to_string()
    } else if vram_gb >= 6.0 {
        "llama3.1:8b".to_string()
    } else if vram_gb >= 3.0 {
        "qwen2.5-1.5b".to_string()
    } else {
        "qwen2.5-0.5b".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recommend_model_high_vram() {
        assert_eq!(recommend_model(16.0), "qwen2.5-14b");
        assert_eq!(recommend_model(12.0), "qwen2.5-14b");
    }

    #[test]
    fn recommend_model_mid_vram() {
        assert_eq!(recommend_model(8.0), "llama3.1:8b");
        assert_eq!(recommend_model(6.0), "llama3.1:8b");
    }

    #[test]
    fn recommend_model_low_vram() {
        assert_eq!(recommend_model(4.0), "qwen2.5-1.5b");
        assert_eq!(recommend_model(3.0), "qwen2.5-1.5b");
    }

    #[test]
    fn recommend_model_minimal_vram() {
        assert_eq!(recommend_model(2.0), "qwen2.5-0.5b");
        assert_eq!(recommend_model(0.0), "qwen2.5-0.5b");
    }

    #[test]
    fn probe_result_defaults() {
        // Verify ProbeResult can be constructed with expected defaults
        let result = ProbeResult {
            available: false,
            models: vec![],
            endpoint: "http://localhost:11434/api/tags".to_string(),
        };
        assert!(!result.available);
        assert!(result.models.is_empty());
    }

    #[test]
    fn detect_best_provider_returns_valid_triple() {
        // This test runs against real network — if nothing is running,
        // we should still get the safe defaults back.
        let (provider, backend, model) = detect_best_provider();
        assert!(!provider.is_empty());
        assert!(!backend.is_empty());
        assert!(!model.is_empty());
        // Provider must be one of the known values
        assert!(
            provider == "local" || provider == "anthropic" || provider == "openai",
            "unexpected provider: {}",
            provider
        );
    }
}
