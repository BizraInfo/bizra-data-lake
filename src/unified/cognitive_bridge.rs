// src/unified/cognitive_bridge.rs - Brain-Body Interface
//
// SAPE v1.∞: The Bridge Contract
// ===============================
// HTTP/gRPC interface to the Python Cognitive Engine.
// Implements the Sidecar Pattern with strict type validation.
//
// The fuzzy outputs of the LLM are validated before entering
// the Rust control logic.

use crate::ollama::{ChatMessage, OllamaClient};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tracing::{info, instrument, warn};

/// Thinking mode for cognitive processing
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub enum ThinkingMode {
    /// System 1: Fast, heuristic, intuitive (PAT)
    FastPat,
    /// System 2: Slow, deliberate, analytical (SAT)
    DeepSat,
    /// Combined: PAT initiates, SAT validates
    #[default]
    HybridSynergy,
    /// Self-improvement through iteration
    Reflexion,
    /// Multi-dimensional synthesis
    GraphOfThought,
}

/// Request to the cognitive engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveRequest {
    /// Agent making the request
    pub agent_id: String,
    /// Task identifier
    pub task_id: String,
    /// Context vector (embeddings)
    pub context_vector: Vec<f32>,
    /// Thinking mode to use
    pub mode: ThinkingMode,
    /// The actual prompt/task
    pub prompt: String,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, String>,

    // SAPE constraints
    /// Minimum SNR required (dB)
    pub min_snr_threshold: f64,
    /// Minimum ethical threshold
    pub min_ihsan_score: f64,
    /// Graph exploration depth limit
    pub max_thinking_depth: u32,
    /// Processing timeout in ms
    pub timeout_ms: u64,
}

impl Default for CognitiveRequest {
    fn default() -> Self {
        Self {
            agent_id: String::new(),
            task_id: String::new(),
            context_vector: Vec::new(),
            mode: ThinkingMode::default(),
            prompt: String::new(),
            metadata: std::collections::HashMap::new(),
            min_snr_threshold: 15.0,
            min_ihsan_score: 0.99,
            max_thinking_depth: 5,
            timeout_ms: 30000,
        }
    }
}

/// Response from the cognitive engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveResponse {
    /// Agent that processed the request
    pub agent_id: String,
    /// Task identifier
    pub task_id: String,

    // Primary output
    /// The synthesized thought/answer
    pub synthesis: String,
    /// Confidence score (0.0-1.0)
    pub confidence: f64,

    // SAPE metrics
    /// Signal-to-Noise Ratio (dB)
    pub snr_score: f64,
    /// Economic/objective alignment
    pub utility_score: f64,
    /// Ethical alignment score
    pub ihsan_score: f64,

    // Thought structure
    /// Serialized thought graph (JSON)
    pub serialized_graph: String,
    /// Individual thought nodes
    pub thought_nodes: Vec<ThoughtNode>,

    // Provenance
    /// Processing time in ms
    pub processing_time_ms: u64,
    /// Model used for generation
    pub model_used: String,
    /// Reasoning steps taken
    pub reasoning_steps: Vec<String>,

    // Status
    /// Whether processing succeeded
    pub success: bool,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Error code
    pub error_code: CognitiveErrorCode,
}

/// Individual thought node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtNode {
    pub id: String,
    pub content: String,
    pub weight: f64,
    pub connections: Vec<String>,
    pub node_type: String,
    pub local_snr: f64,
}

/// Error codes for cognitive failures
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub enum CognitiveErrorCode {
    #[default]
    None,
    LowSnr,
    EthicsViolation,
    Timeout,
    ContextOverflow,
    ModelUnavailable,
    InvalidRequest,
    CircuitBreakerOpen,
}

/// Cognitive Bridge - Interface to the Python Brain
///
/// Implements the Sidecar Pattern:
/// - Rust orchestrator manages lifecycle and resources
/// - Python engine handles heavy tensor/LLM operations
pub struct CognitiveBridge {
    /// HTTP client for Python service (when available)
    http_client: reqwest::Client,
    /// Python cognitive service URL
    python_service_url: Option<String>,
    /// Local Ollama client (fallback)
    ollama: Arc<OllamaClient>,
    /// Concurrency limiter
    semaphore: Arc<Semaphore>,
    /// Circuit breaker state
    circuit_breaker: CircuitBreaker,
    /// Default model to use
    default_model: String,
}

/// Circuit breaker for fault tolerance
struct CircuitBreaker {
    /// Consecutive failures
    failures: std::sync::atomic::AtomicU32,
    /// Threshold to open circuit
    threshold: u32,
    /// Last failure time
    last_failure: std::sync::atomic::AtomicU64,
    /// Cooldown period in seconds
    cooldown_secs: u64,
}

impl CircuitBreaker {
    fn new(threshold: u32, cooldown_secs: u64) -> Self {
        Self {
            failures: std::sync::atomic::AtomicU32::new(0),
            threshold,
            last_failure: std::sync::atomic::AtomicU64::new(0),
            cooldown_secs,
        }
    }

    fn is_open(&self) -> bool {
        let failures = self.failures.load(std::sync::atomic::Ordering::Relaxed);
        if failures >= self.threshold {
            // Check cooldown
            let last = self.last_failure.load(std::sync::atomic::Ordering::Relaxed);
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            if now - last < self.cooldown_secs {
                return true;
            }
            // Reset after cooldown
            self.failures.store(0, std::sync::atomic::Ordering::Relaxed);
        }
        false
    }

    fn record_success(&self) {
        self.failures.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    fn record_failure(&self) {
        self.failures
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.last_failure
            .store(now, std::sync::atomic::Ordering::Relaxed);
    }
}

impl CognitiveBridge {
    /// Create a new cognitive bridge
    pub async fn new(
        python_service_url: Option<String>,
        max_concurrent: usize,
        default_model: &str,
    ) -> anyhow::Result<Self> {
        let ollama = crate::ollama::get_ollama().await;

        info!(
            python_url = ?python_service_url,
            max_concurrent = max_concurrent,
            model = default_model,
            "🧠 CognitiveBridge initialized"
        );

        Ok(Self {
            http_client: reqwest::Client::builder()
                .timeout(Duration::from_secs(60))
                .build()?,
            python_service_url,
            ollama,
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            circuit_breaker: CircuitBreaker::new(5, 30),
            default_model: default_model.to_string(),
        })
    }

    /// Process a cognitive request
    #[instrument(skip(self, request), fields(agent_id = %request.agent_id, task_id = %request.task_id))]
    pub async fn process(&self, request: CognitiveRequest) -> CognitiveResponse {
        let start = Instant::now();

        // Check circuit breaker
        if self.circuit_breaker.is_open() {
            warn!(agent_id = %request.agent_id, "Circuit breaker open - attempting local fallback");

            // Fallback: Return a degraded but valid response to keep the system running
            return CognitiveResponse {
                agent_id: request.agent_id,
                task_id: request.task_id,
                synthesis:
                    "Service degraded: Circuit breaker open. Returning best-effort fallback."
                        .to_string(),
                confidence: 0.1, // Low confidence
                snr_score: 1.0,  // Baseline SNR
                utility_score: 0.1,
                ihsan_score: 0.99, // Assume safe
                serialized_graph: "{}".to_string(),
                thought_nodes: vec![],
                processing_time_ms: start.elapsed().as_millis() as u64,
                model_used: "fallback-circuit-open".to_string(),
                reasoning_steps: vec!["Circuit breaker open, skipped inference".to_string()],
                success: true, // Degraded success to prevent system panic
                error_message: Some("Circuit breaker open - using fallback".to_string()),
                error_code: CognitiveErrorCode::CircuitBreakerOpen, // Client can check this
            };
        }

        // Acquire semaphore permit
        let _permit = match self.semaphore.acquire().await {
            Ok(p) => p,
            Err(_) => {
                return self.error_response(
                    &request,
                    start,
                    "Semaphore closed",
                    CognitiveErrorCode::InvalidRequest,
                );
            }
        };

        // Try Python service first if available
        if let Some(ref url) = self.python_service_url {
            match self.call_python_service(url, &request).await {
                Ok(response) => {
                    self.circuit_breaker.record_success();
                    return response;
                }
                Err(e) => {
                    warn!(error = %e, "Python cognitive service failed, falling back to Ollama");
                    self.circuit_breaker.record_failure();
                }
            }
        }

        // Fallback to local Ollama
        self.process_with_ollama(request, start).await
    }

    /// Call the Python cognitive service
    async fn call_python_service(
        &self,
        url: &str,
        request: &CognitiveRequest,
    ) -> anyhow::Result<CognitiveResponse> {
        let response = self
            .http_client
            .post(format!("{}/process", url))
            .json(request)
            .timeout(Duration::from_millis(request.timeout_ms))
            .send()
            .await?;

        if response.status().is_success() {
            let cognitive_response: CognitiveResponse = response.json().await?;
            Ok(cognitive_response)
        } else {
            anyhow::bail!("Python service returned error: {}", response.status())
        }
    }

    /// Process using local Ollama
    async fn process_with_ollama(
        &self,
        request: CognitiveRequest,
        start: Instant,
    ) -> CognitiveResponse {
        // Build prompt based on thinking mode
        let system_prompt = self.build_system_prompt(&request.mode);
        let user_prompt = self.build_user_prompt(&request);

        let messages = vec![
            ChatMessage::system(system_prompt),
            ChatMessage::user(user_prompt),
        ];

        // Call Ollama
        match self
            .ollama
            .chat(messages, Some(&self.default_model), None)
            .await
        {
            Ok(response) => {
                let processing_time = start.elapsed().as_millis() as u64;
                let content = &response.message.content;

                // Calculate metrics
                let confidence = self.calculate_confidence(content);
                let snr_score = self.calculate_snr(content, &request.prompt);
                let ihsan_score = self.calculate_ihsan(content);
                let utility_score = self.calculate_utility(content, &request);

                // Validate against thresholds
                if snr_score < request.min_snr_threshold {
                    return self.error_response(
                        &request,
                        start,
                        &format!(
                            "SNR {} below threshold {}",
                            snr_score, request.min_snr_threshold
                        ),
                        CognitiveErrorCode::LowSnr,
                    );
                }

                if ihsan_score < request.min_ihsan_score {
                    return self.error_response(
                        &request,
                        start,
                        &format!(
                            "Ihsān {} below threshold {}",
                            ihsan_score, request.min_ihsan_score
                        ),
                        CognitiveErrorCode::EthicsViolation,
                    );
                }

                self.circuit_breaker.record_success();

                CognitiveResponse {
                    agent_id: request.agent_id,
                    task_id: request.task_id,
                    synthesis: content.clone(),
                    confidence,
                    snr_score,
                    utility_score,
                    ihsan_score,
                    serialized_graph: serde_json::json!({
                        "nodes": [{"id": "root", "content": content.chars().take(200).collect::<String>()}],
                        "edges": []
                    }).to_string(),
                    thought_nodes: vec![ThoughtNode {
                        id: "root".to_string(),
                        content: content.chars().take(200).collect(),
                        weight: confidence,
                        connections: Vec::new(),
                        node_type: "synthesis".to_string(),
                        local_snr: snr_score,
                    }],
                    processing_time_ms: processing_time,
                    model_used: self.default_model.clone(),
                    reasoning_steps: vec!["Direct LLM synthesis".to_string()],
                    success: true,
                    error_message: None,
                    error_code: CognitiveErrorCode::None,
                }
            }
            Err(e) => {
                self.circuit_breaker.record_failure();
                self.error_response(
                    &request,
                    start,
                    &e.to_string(),
                    CognitiveErrorCode::ModelUnavailable,
                )
            }
        }
    }

    /// Build system prompt based on thinking mode
    fn build_system_prompt(&self, mode: &ThinkingMode) -> String {
        match mode {
            ThinkingMode::FastPat => {
                "You are a fast, intuitive thinker. Provide quick, heuristic-based responses. \
                 Prioritize speed and practicality over exhaustive analysis.".to_string()
            }
            ThinkingMode::DeepSat => {
                "You are a deep, analytical thinker. Take time to reason through problems step by step. \
                 Consider multiple perspectives and verify your logic carefully.".to_string()
            }
            ThinkingMode::HybridSynergy => {
                "You combine intuitive and analytical thinking. Start with quick insights, \
                 then validate them with careful reasoning. Balance speed with accuracy.".to_string()
            }
            ThinkingMode::Reflexion => {
                "You are a self-improving reasoner. After providing an answer, reflect on it critically. \
                 Identify weaknesses in your reasoning and refine your response.".to_string()
            }
            ThinkingMode::GraphOfThought => {
                "You think in interconnected concepts. Build a web of related ideas, exploring \
                 multiple pathways simultaneously. Synthesize insights from diverse connections.".to_string()
            }
        }
    }

    /// Build user prompt with context
    fn build_user_prompt(&self, request: &CognitiveRequest) -> String {
        let mut prompt = request.prompt.clone();

        // Add metadata context if present
        if !request.metadata.is_empty() {
            prompt.push_str("\n\nContext:\n");
            for (key, value) in &request.metadata {
                prompt.push_str(&format!("- {}: {}\n", key, value));
            }
        }

        prompt
    }

    /// Calculate confidence score from response
    fn calculate_confidence(&self, response: &str) -> f64 {
        // Simple heuristics for confidence
        let len_factor = (response.len() as f64 / 500.0).min(1.0);
        let structure_factor = if response.contains('\n') { 0.1 } else { 0.0 };
        let certainty_factor = if response.to_lowercase().contains("certain")
            || response.to_lowercase().contains("definitely")
        {
            0.1
        } else if response.to_lowercase().contains("maybe")
            || response.to_lowercase().contains("possibly")
        {
            -0.1
        } else {
            0.0
        };

        (0.7 + len_factor * 0.2 + structure_factor + certainty_factor).clamp(0.0, 1.0)
    }

    /// Calculate Signal-to-Noise Ratio
    fn calculate_snr(&self, response: &str, prompt: &str) -> f64 {
        // Relevance score (how much response relates to prompt)
        let prompt_lower = prompt.to_lowercase();
        let prompt_words: std::collections::HashSet<_> = prompt_lower
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();

        let response_lower = response.to_lowercase();
        let response_words: Vec<_> = response_lower
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();

        let relevant_count = response_words
            .iter()
            .filter(|w| prompt_words.contains(*w))
            .count();

        let relevance = if response_words.is_empty() {
            0.0
        } else {
            relevant_count as f64 / response_words.len() as f64
        };

        // Information density (unique words ratio)
        let unique_words: std::collections::HashSet<_> = response_words.iter().collect();
        let density = if response_words.is_empty() {
            0.0
        } else {
            unique_words.len() as f64 / response_words.len() as f64
        };

        // SNR in "dB" (scaled 0-30)
        let raw_snr = (relevance * 0.6 + density * 0.4) * 30.0;
        raw_snr.max(0.0)
    }

    /// Calculate Ihsān (ethical) score
    fn calculate_ihsan(&self, response: &str) -> f64 {
        let lower = response.to_lowercase();

        // Check for harmful patterns
        let harmful_patterns = [
            "harm",
            "attack",
            "exploit",
            "malicious",
            "illegal",
            "unauthorized",
            "bypass",
            "hack",
            "steal",
        ];
        let harm_count = harmful_patterns
            .iter()
            .filter(|p| lower.contains(*p))
            .count();

        // Check for helpful patterns
        let helpful_patterns = [
            "help",
            "assist",
            "support",
            "guide",
            "ensure",
            "safe",
            "secure",
            "ethical",
            "responsible",
        ];
        let help_count = helpful_patterns
            .iter()
            .filter(|p| lower.contains(*p))
            .count();

        // Base score with adjustments
        let base = 0.92;
        let harm_penalty = harm_count as f64 * 0.05;
        let help_bonus = help_count as f64 * 0.02;

        (base - harm_penalty + help_bonus).clamp(0.0, 1.0)
    }

    /// Calculate utility score
    fn calculate_utility(&self, response: &str, request: &CognitiveRequest) -> f64 {
        // Length appropriateness
        let len = response.len();
        let len_score = if len < 50 {
            0.5
        } else if len < 500 {
            0.8
        } else if len < 2000 {
            1.0
        } else {
            0.9
        };

        // Actionability (contains action words)
        let action_words = [
            "implement",
            "create",
            "build",
            "design",
            "develop",
            "execute",
            "run",
        ];
        let actionable = action_words
            .iter()
            .any(|w| response.to_lowercase().contains(w));
        let action_score = if actionable { 0.1 } else { 0.0 };

        // Mode alignment
        let mode_score = match request.mode {
            ThinkingMode::FastPat => {
                if len < 300 {
                    0.1
                } else {
                    0.0
                }
            }
            ThinkingMode::DeepSat => {
                if len > 500 {
                    0.1
                } else {
                    0.0
                }
            }
            _ => 0.05,
        };

        let total: f64 = len_score + action_score + mode_score;
        total.min(1.0)
    }

    /// Create an error response
    fn error_response(
        &self,
        request: &CognitiveRequest,
        start: Instant,
        message: &str,
        code: CognitiveErrorCode,
    ) -> CognitiveResponse {
        CognitiveResponse {
            agent_id: request.agent_id.clone(),
            task_id: request.task_id.clone(),
            synthesis: String::new(),
            confidence: 0.0,
            snr_score: 0.0,
            utility_score: 0.0,
            ihsan_score: 0.0,
            serialized_graph: String::new(),
            thought_nodes: Vec::new(),
            processing_time_ms: start.elapsed().as_millis() as u64,
            model_used: self.default_model.clone(),
            reasoning_steps: Vec::new(),
            success: false,
            error_message: Some(message.to_string()),
            error_code: code,
        }
    }

    /// Check if cognitive service is available
    pub async fn is_available(&self) -> bool {
        if self.circuit_breaker.is_open() {
            return false;
        }

        if let Some(ref url) = self.python_service_url {
            if let Ok(response) = self.http_client.get(format!("{}/health", url)).send().await {
                if response.status().is_success() {
                    return true;
                }
            }
        }

        // Check Ollama fallback
        self.ollama.is_connected()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thinking_mode_default() {
        assert_eq!(ThinkingMode::default(), ThinkingMode::HybridSynergy);
    }

    #[test]
    fn test_cognitive_request_default() {
        let req = CognitiveRequest::default();
        assert_eq!(req.min_snr_threshold, 15.0);
        assert_eq!(req.min_ihsan_score, 0.99);
    }

    #[test]
    fn test_circuit_breaker() {
        let cb = CircuitBreaker::new(3, 10);
        assert!(!cb.is_open());

        cb.record_failure();
        cb.record_failure();
        assert!(!cb.is_open());

        cb.record_failure();
        assert!(cb.is_open());

        cb.record_success();
        assert!(!cb.is_open());
    }
}
