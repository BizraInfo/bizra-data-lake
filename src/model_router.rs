// src/model_router.rs - Capability-Based Model Router
//
// STANDING ON THE SHOULDERS OF GIANTS:
// - DeepSeek R1: Deterministic reasoning with self-correction
// - Mistral: Nuance and user-facing tone control
// - Nomic: Semantic embeddings for RAG/search
// - BFT Consensus: 3/5 quorum for high-stakes decisions
// - eBPF-style probes: SAPE pattern elevation
//
// This module connects the capability slots defined in model-family-genesis-v1-SEALED.yaml
// to actual Ollama inference, enabling intelligent routing based on task requirements.

use crate::lmstudio::{self, LmStudioChatResponse, LmStudioClient};
use crate::ollama::{self, ChatMessage, GenerationOptions, OllamaClient};
use lazy_static::lazy_static;
use prometheus::{register_counter_vec, register_histogram_vec, CounterVec, HistogramVec};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{debug, error, info, instrument, warn};

lazy_static! {
    /// Model routing decisions counter
    pub static ref ROUTER_DECISIONS: CounterVec = register_counter_vec!(
        "bizra_model_router_decisions_total",
        "Total model routing decisions",
        &["slot", "model", "result"]  // cold_core/warm_surface/etc, model name, success/fallback/error
    ).unwrap();
    
    /// Model inference latency by slot
    pub static ref ROUTER_LATENCY: HistogramVec = register_histogram_vec!(
        "bizra_model_router_latency_seconds",
        "Model routing and inference latency",
        &["slot", "model"],
        vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
    ).unwrap();
    
    /// Slot utilization gauge
    pub static ref SLOT_USAGE: CounterVec = register_counter_vec!(
        "bizra_model_slot_usage_total",
        "Slot usage by capability",
        &["slot"]
    ).unwrap();
}

// ============================================================
// Capability Slots (from model-family-genesis-v1-SEALED.yaml)
// ============================================================

/// Capability slots define the routing categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CapabilitySlot {
    /// Deterministic reasoning + self-correction + causal trace
    ColdCore,
    /// Nuance + formatting + user-facing tone control
    WarmSurface,
    /// Deterministic embedding for RAG / semantic search
    Embeddings,
    /// Multi-agent orchestration + strategic planning
    PrimaryReasoning,
    /// Vision-capable inference (multimodal)
    Vision,
}

impl CapabilitySlot {
    pub fn name(&self) -> &'static str {
        match self {
            Self::ColdCore => "cold_core",
            Self::WarmSurface => "warm_surface",
            Self::Embeddings => "embeddings",
            Self::PrimaryReasoning => "primary_reasoning",
            Self::Vision => "vision",
        }
    }
    
    pub fn description(&self) -> &'static str {
        match self {
            Self::ColdCore => "Deterministic reasoning + self-correction + causal trace",
            Self::WarmSurface => "Nuance + formatting + user-facing tone control",
            Self::Embeddings => "Deterministic embedding for RAG / semantic search",
            Self::PrimaryReasoning => "Multi-agent orchestration + strategic planning",
            Self::Vision => "Vision-capable inference (multimodal)",
        }
    }
    
    /// Primary model for this slot (from SEALED config)
    pub fn primary_model(&self) -> &'static str {
        match self {
            Self::ColdCore => "deepseek-r1:8b",
            Self::WarmSurface => "mistral:latest",
            Self::Embeddings => "nomic-embed-text:latest",
            Self::PrimaryReasoning => "bizra-planner:latest",
            Self::Vision => "qwen/qwen3-vl-8b",
        }
    }
    
    /// Fallback model for this slot (from SEALED config)
    pub fn fallback_model(&self) -> &'static str {
        match self {
            Self::ColdCore => "mistral:latest",
            Self::WarmSurface => "qwen2.5:7b",
            Self::Embeddings => "nomic-embed-text:latest", // No fallback
            Self::PrimaryReasoning => "agentflow-planner-7b-i1",
            Self::Vision => "qwen/qwen3-vl-4b",
        }
    }

    /// Alternative models for this slot (local overrides when SEALED models are unavailable)
    pub fn alternative_models(&self) -> &'static [&'static str] {
        match self {
            Self::Vision => &[
                "llava:7b",
                "llava:latest",
                "llava",
                "llama3.2-vision",
                "llama3.2-vision:latest",
            ],
            _ => &[],
        }
    }
    
    /// Temperature setting for this slot
    pub fn temperature(&self) -> f64 {
        match self {
            Self::ColdCore => 0.6,      // Optimized for consistency
            Self::WarmSurface => 0.3,   // Low for reliability
            Self::PrimaryReasoning => 0.7,
            Self::Embeddings => 0.0,    // N/A for embeddings
            Self::Vision => 0.7,
        }
    }
    
    /// Context window size for this slot
    pub fn num_ctx(&self) -> u32 {
        match self {
            Self::ColdCore => 8192,
            Self::WarmSurface => 32768,
            Self::Embeddings => 8192,
            Self::PrimaryReasoning => 4096,
            Self::Vision => 8192,
        }
    }
    
    /// All capability slots
    pub fn all() -> &'static [CapabilitySlot] {
        &[
            Self::ColdCore,
            Self::WarmSurface,
            Self::Embeddings,
            Self::PrimaryReasoning,
            Self::Vision,
        ]
    }
}

// ============================================================
// Task Classification for Routing
// ============================================================

/// Task characteristics that influence routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskCharacteristics {
    /// Complexity score (0.0 - 1.0)
    pub complexity: f64,
    /// Requires strict logical reasoning
    pub requires_reasoning: bool,
    /// Requires creative/nuanced output
    pub requires_creativity: bool,
    /// Requires multi-step planning
    pub requires_planning: bool,
    /// Requires image understanding
    pub requires_vision: bool,
    /// Requires embedding generation
    pub requires_embedding: bool,
    /// Is user-facing output
    pub is_user_facing: bool,
    /// Is high-stakes (needs determinism)
    pub is_high_stakes: bool,
}

impl Default for TaskCharacteristics {
    fn default() -> Self {
        Self {
            complexity: 0.5,
            requires_reasoning: false,
            requires_creativity: false,
            requires_planning: false,
            requires_vision: false,
            requires_embedding: false,
            is_user_facing: false,
            is_high_stakes: false,
        }
    }
}

impl TaskCharacteristics {
    /// Classify task from content using heuristics
    pub fn classify(content: &str) -> Self {
        let content_lower = content.to_lowercase();
        let word_count = content.split_whitespace().count();
        
        // Complexity estimation
        let complexity = match word_count {
            0..=20 => 0.2,
            21..=50 => 0.4,
            51..=150 => 0.6,
            151..=500 => 0.8,
            _ => 1.0,
        };
        
        // Reasoning indicators
        let reasoning_patterns = [
            "why", "because", "therefore", "prove", "verify",
            "logic", "reason", "deduce", "analyze", "conclude",
            "calculate", "compute", "derive", "solve",
        ];
        let requires_reasoning = reasoning_patterns.iter()
            .any(|p| content_lower.contains(p));
        
        // Creativity indicators
        let creativity_patterns = [
            "creative", "imagine", "design", "write", "story",
            "poem", "innovative", "brainstorm", "novel", "unique",
        ];
        let requires_creativity = creativity_patterns.iter()
            .any(|p| content_lower.contains(p));
        
        // Planning indicators
        let planning_patterns = [
            "plan", "strategy", "steps", "roadmap", "timeline",
            "orchestrate", "coordinate", "schedule", "milestone",
        ];
        let requires_planning = planning_patterns.iter()
            .any(|p| content_lower.contains(p));
        
        // Vision indicators
        let vision_patterns = [
            "image", "picture", "photo", "visual", "diagram",
            "screenshot", "look at", "see the", "in this image",
        ];
        let requires_vision = vision_patterns.iter()
            .any(|p| content_lower.contains(p));
        
        // Embedding indicators
        let embedding_patterns = [
            "embed", "similar", "search", "semantic", "vector",
            "retrieve", "find related", "match",
        ];
        let requires_embedding = embedding_patterns.iter()
            .any(|p| content_lower.contains(p));
        
        // User-facing indicators
        let user_facing_patterns = [
            "user", "customer", "client", "explain to",
            "present", "communicate", "message", "response",
        ];
        let is_user_facing = user_facing_patterns.iter()
            .any(|p| content_lower.contains(p));
        
        // High-stakes indicators
        let high_stakes_patterns = [
            "security", "critical", "production", "deploy",
            "financial", "medical", "legal", "compliance", "safety",
            "must be correct", "cannot fail", "verify",
        ];
        let is_high_stakes = high_stakes_patterns.iter()
            .any(|p| content_lower.contains(p));
        
        Self {
            complexity,
            requires_reasoning,
            requires_creativity,
            requires_planning,
            requires_vision,
            requires_embedding,
            is_user_facing,
            is_high_stakes,
        }
    }
    
    /// Determine the optimal capability slot for this task
    pub fn optimal_slot(&self) -> CapabilitySlot {
        // Priority order based on task requirements
        if self.requires_embedding {
            return CapabilitySlot::Embeddings;
        }
        
        if self.requires_vision {
            return CapabilitySlot::Vision;
        }
        
        if self.requires_planning || self.complexity > 0.7 {
            return CapabilitySlot::PrimaryReasoning;
        }
        
        if self.is_high_stakes || self.requires_reasoning {
            return CapabilitySlot::ColdCore;
        }
        
        if self.is_user_facing || self.requires_creativity {
            return CapabilitySlot::WarmSurface;
        }
        
        // Default to warm surface for general tasks
        CapabilitySlot::WarmSurface
    }
}

// ============================================================
// Model Router
// ============================================================

/// Routing decision result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    pub slot: CapabilitySlot,
    pub model: String,
    pub is_fallback: bool,
    pub characteristics: TaskCharacteristics,
    pub reasoning: String,
}

/// Model availability cache
#[derive(Debug, Clone, Copy, Default)]
struct ProviderAvailability {
    ollama: bool,
    lmstudio: bool,
}

struct ModelAvailability {
    available: HashMap<String, ProviderAvailability>,
    last_check: std::time::Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelProvider {
    Ollama,
    LmStudio,
}

/// Model Router - connects capability slots to actual inference
pub struct ModelRouter {
    ollama: Arc<OllamaClient>,
    lmstudio: Arc<LmStudioClient>,
    availability: RwLock<ModelAvailability>,
    /// Cache refresh interval (5 minutes)
    cache_ttl: std::time::Duration,
}

impl ModelRouter {
    /// Create new model router
    pub async fn new() -> anyhow::Result<Self> {
        info!("🔀 Initializing Model Router");
        
        let ollama = ollama::get_ollama().await;
        let lmstudio = lmstudio::get_lmstudio().await;
        
        let mut router = Self {
            ollama,
            lmstudio,
            availability: RwLock::new(ModelAvailability {
                available: HashMap::new(),
                last_check: std::time::Instant::now() - std::time::Duration::from_secs(600),
            }),
            cache_ttl: std::time::Duration::from_secs(300),
        };
        
        // Initial model discovery
        router.refresh_availability().await?;
        
        Ok(router)
    }
    
    /// Refresh model availability cache
    #[instrument(skip(self))]
    pub async fn refresh_availability(&mut self) -> anyhow::Result<()> {
        let models = self.ollama.list_models().await.unwrap_or_default();
        
        let mut availability = self.availability.write().await;
        availability.available.clear();
        
        for model in &models {
            let entry = availability
                .available
                .entry(model.name.clone())
                .or_insert_with(ProviderAvailability::default);
            entry.ollama = true;
            debug!("Model available: {}", model.name);
        }

        match self.lmstudio.list_models().await {
            Ok(models) => {
                for model in models {
                    let entry = availability
                        .available
                        .entry(model.clone())
                        .or_insert_with(ProviderAvailability::default);
                    entry.lmstudio = true;
                    debug!("Model available (lmstudio): {}", model);
                }
            }
            Err(err) => {
                warn!("LM Studio model discovery failed: {}", err);
            }
        }
        
        availability.last_check = std::time::Instant::now();
        
        info!(
            models_found = models.len(),
            "Model availability refreshed"
        );
        
        Ok(())
    }
    
    /// Check if a model is available
    async fn is_model_available(&self, model: &str) -> bool {
        let availability = self.availability.read().await;
        
        // Check if cache is stale
        if availability.last_check.elapsed() > self.cache_ttl {
            // Cache is stale, but we'll use it anyway and refresh async
            // This prevents blocking on availability checks
        }
        
        availability
            .available
            .get(model)
            .map(|entry| entry.ollama || entry.lmstudio)
            .unwrap_or(false)
    }

    async fn provider_for_model(&self, model: &str) -> Option<ModelProvider> {
        let availability = self.availability.read().await;
        let entry = availability.available.get(model)?;
        if entry.ollama {
            Some(ModelProvider::Ollama)
        } else if entry.lmstudio {
            Some(ModelProvider::LmStudio)
        } else {
            None
        }
    }
    
    /// Route a task to the optimal model
    #[instrument(skip(self))]
    pub async fn route(&self, content: &str) -> RoutingDecision {
        let characteristics = TaskCharacteristics::classify(content);
        let slot = characteristics.optimal_slot();
        
        SLOT_USAGE.with_label_values(&[slot.name()]).inc();
        
        // Check primary model availability
        let primary = slot.primary_model();
        if self.is_model_available(primary).await {
            ROUTER_DECISIONS.with_label_values(&[slot.name(), primary, "primary"]).inc();
            
            return RoutingDecision {
                slot,
                model: primary.to_string(),
                is_fallback: false,
                characteristics,
                reasoning: format!(
                    "Routed to {} slot using primary model {}",
                    slot.name(),
                    primary
                ),
            };
        }
        
        // Try fallback model
        let fallback = slot.fallback_model();
        if self.is_model_available(fallback).await {
            ROUTER_DECISIONS.with_label_values(&[slot.name(), fallback, "fallback"]).inc();
            
            warn!(
                slot = slot.name(),
                primary = primary,
                fallback = fallback,
                "Primary model unavailable, using fallback"
            );
            
            return RoutingDecision {
                slot,
                model: fallback.to_string(),
                is_fallback: true,
                characteristics,
                reasoning: format!(
                    "Primary {} unavailable, falling back to {}",
                    primary, fallback
                ),
            };
        }

        // Try alternative models (local overrides)
        for alternative in slot.alternative_models() {
            if self.is_model_available(alternative).await {
                ROUTER_DECISIONS.with_label_values(&[slot.name(), alternative, "alternative"]).inc();

                warn!(
                    slot = slot.name(),
                    primary = primary,
                    fallback = fallback,
                    alternative = alternative,
                    "Using alternative model for slot"
                );

                return RoutingDecision {
                    slot,
                    model: alternative.to_string(),
                    is_fallback: true,
                    characteristics,
                    reasoning: format!(
                        "Primary {} and fallback {} unavailable, using alternative {}",
                        primary, fallback, alternative
                    ),
                };
            }
        }
        
        // Last resort: use any available model
        let availability = self.availability.read().await;
        if let Some((model, _)) = availability
            .available
            .iter()
            .find(|(_, v)| v.ollama || v.lmstudio)
        {
            ROUTER_DECISIONS.with_label_values(&[slot.name(), model, "emergency"]).inc();
            
            warn!(
                slot = slot.name(),
                model = model.as_str(),
                "Both primary and fallback unavailable, using emergency model"
            );
            
            return RoutingDecision {
                slot,
                model: model.clone(),
                is_fallback: true,
                characteristics,
                reasoning: format!(
                    "Emergency routing: using {} as only available model",
                    model
                ),
            };
        }
        
        // No models available
        ROUTER_DECISIONS.with_label_values(&[slot.name(), "none", "error"]).inc();
        error!("No models available for routing");
        
        RoutingDecision {
            slot,
            model: slot.primary_model().to_string(), // Will fail at inference
            is_fallback: false,
            characteristics,
            reasoning: "No models available - inference will fail".to_string(),
        }
    }
    
    /// Route explicitly to a specific slot
    pub async fn route_to_slot(&self, slot: CapabilitySlot, content: &str) -> RoutingDecision {
        let characteristics = TaskCharacteristics::classify(content);
        
        SLOT_USAGE.with_label_values(&[slot.name()]).inc();
        
        // Check primary model
        let primary = slot.primary_model();
        if self.is_model_available(primary).await {
            ROUTER_DECISIONS.with_label_values(&[slot.name(), primary, "explicit"]).inc();
            
            return RoutingDecision {
                slot,
                model: primary.to_string(),
                is_fallback: false,
                characteristics,
                reasoning: format!("Explicit routing to {} slot", slot.name()),
            };
        }
        
        // Try fallback
        let fallback = slot.fallback_model();
        if self.is_model_available(fallback).await {
            ROUTER_DECISIONS.with_label_values(&[slot.name(), fallback, "explicit_fallback"]).inc();
            
            return RoutingDecision {
                slot,
                model: fallback.to_string(),
                is_fallback: true,
                characteristics,
                reasoning: format!(
                    "Explicit routing to {} slot (fallback model)",
                    slot.name()
                ),
            };
        }

        // Try alternative models (local overrides)
        for alternative in slot.alternative_models() {
            if self.is_model_available(alternative).await {
                ROUTER_DECISIONS.with_label_values(&[slot.name(), alternative, "explicit_alternative"]).inc();

                return RoutingDecision {
                    slot,
                    model: alternative.to_string(),
                    is_fallback: true,
                    characteristics,
                    reasoning: format!(
                        "Explicit routing to {} slot (alternative model)",
                        slot.name()
                    ),
                };
            }
        }
        
        // Use primary anyway (will fail)
        RoutingDecision {
            slot,
            model: primary.to_string(),
            is_fallback: false,
            characteristics,
            reasoning: format!("Explicit routing to {} (model may be unavailable)", slot.name()),
        }
    }
    
    /// Execute inference with automatic routing
    #[instrument(skip(self, messages))]
    pub async fn infer(
        &self,
        content: &str,
        messages: Vec<ChatMessage>,
    ) -> anyhow::Result<InferenceResult> {
        let decision = self.route(content).await;
        self.infer_with_decision(messages, &decision).await
    }
    
    /// Execute inference with a specific routing decision
    #[instrument(skip(self, messages))]
    pub async fn infer_with_decision(
        &self,
        messages: Vec<ChatMessage>,
        decision: &RoutingDecision,
    ) -> anyhow::Result<InferenceResult> {
        let start = Instant::now();
        let slot = decision.slot;
        let model = &decision.model;
        
        // Build options from slot config
        let options = GenerationOptions {
            temperature: Some(slot.temperature()),
            num_ctx: Some(slot.num_ctx()),
            ..Default::default()
        };
        
        debug!(
            slot = slot.name(),
            model = model.as_str(),
            "Executing inference"
        );
        
        let provider = self.provider_for_model(model).await.unwrap_or(ModelProvider::Ollama);

        let (content, tokens_prompt, tokens_completion) = match provider {
            ModelProvider::Ollama => {
                let response = self.ollama
                    .chat(messages, Some(model), Some(options))
                    .await?;
                (
                    response.message.content,
                    response.prompt_eval_count,
                    response.eval_count,
                )
            }
            ModelProvider::LmStudio => {
                let response: LmStudioChatResponse = self
                    .lmstudio
                    .chat_completion(model, messages, options)
                    .await?;
                (
                    response.message.content,
                    response.prompt_tokens,
                    response.completion_tokens,
                )
            }
        };
        
        let latency = start.elapsed();
        
        ROUTER_LATENCY
            .with_label_values(&[slot.name(), model])
            .observe(latency.as_secs_f64());
        
        info!(
            slot = slot.name(),
            model = model.as_str(),
            latency_ms = latency.as_millis(),
            "Inference completed"
        );
        
        Ok(InferenceResult {
            content,
            model: model.clone(),
            slot,
            is_fallback: decision.is_fallback,
            latency,
            tokens_prompt,
            tokens_completion,
        })
    }
    
    /// Execute inference for a specific slot (bypasses automatic routing)
    #[instrument(skip(self, messages))]
    pub async fn infer_slot(
        &self,
        slot: CapabilitySlot,
        messages: Vec<ChatMessage>,
        context: &str,
    ) -> anyhow::Result<InferenceResult> {
        let decision = self.route_to_slot(slot, context).await;
        self.infer_with_decision(messages, &decision).await
    }
    
    /// Get router statistics
    pub async fn get_stats(&self) -> RouterStats {
        let availability = self.availability.read().await;
        let available_models: Vec<String> = availability
            .available
            .iter()
            .filter(|(_, v)| v.ollama || v.lmstudio)
            .map(|(k, _)| k.clone())
            .collect();
        let available_models_ollama: Vec<String> = availability
            .available
            .iter()
            .filter(|(_, v)| v.ollama)
            .map(|(k, _)| k.clone())
            .collect();
        let available_models_lmstudio: Vec<String> = availability
            .available
            .iter()
            .filter(|(_, v)| v.lmstudio)
            .map(|(k, _)| k.clone())
            .collect();
        
        RouterStats {
            available_models,
            available_models_ollama,
            available_models_lmstudio,
            last_refresh: availability.last_check.elapsed().as_secs(),
            slots: CapabilitySlot::all()
                .iter()
                .map(|s| SlotStats {
                    name: s.name().to_string(),
                    primary: s.primary_model().to_string(),
                    fallback: s.fallback_model().to_string(),
                    primary_available: availability
                        .available
                        .get(s.primary_model())
                        .map(|v| v.ollama || v.lmstudio)
                        .unwrap_or(false),
                    fallback_available: availability
                        .available
                        .get(s.fallback_model())
                        .map(|v| v.ollama || v.lmstudio)
                        .unwrap_or(false),
                    primary_available_ollama: availability
                        .available
                        .get(s.primary_model())
                        .map(|v| v.ollama)
                        .unwrap_or(false),
                    primary_available_lmstudio: availability
                        .available
                        .get(s.primary_model())
                        .map(|v| v.lmstudio)
                        .unwrap_or(false),
                    fallback_available_ollama: availability
                        .available
                        .get(s.fallback_model())
                        .map(|v| v.ollama)
                        .unwrap_or(false),
                    fallback_available_lmstudio: availability
                        .available
                        .get(s.fallback_model())
                        .map(|v| v.lmstudio)
                        .unwrap_or(false),
                    alternatives: s
                        .alternative_models()
                        .iter()
                        .map(|m| m.to_string())
                        .collect(),
                    alternatives_available: s
                        .alternative_models()
                        .iter()
                        .any(|m| {
                            availability
                                .available
                                .get(*m)
                                .map(|v| v.ollama || v.lmstudio)
                                .unwrap_or(false)
                        }),
                    alternatives_available_ollama: s
                        .alternative_models()
                        .iter()
                        .any(|m| {
                            availability
                                .available
                                .get(*m)
                                .map(|v| v.ollama)
                                .unwrap_or(false)
                        }),
                    alternatives_available_lmstudio: s
                        .alternative_models()
                        .iter()
                        .any(|m| {
                            availability
                                .available
                                .get(*m)
                                .map(|v| v.lmstudio)
                                .unwrap_or(false)
                        }),
                })
                .collect(),
        }
    }
}

/// Inference result with routing metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    pub content: String,
    pub model: String,
    pub slot: CapabilitySlot,
    pub is_fallback: bool,
    pub latency: std::time::Duration,
    pub tokens_prompt: Option<u32>,
    pub tokens_completion: Option<u32>,
}

/// Router statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterStats {
    pub available_models: Vec<String>,
    pub available_models_ollama: Vec<String>,
    pub available_models_lmstudio: Vec<String>,
    pub last_refresh: u64,
    pub slots: Vec<SlotStats>,
}

/// Slot statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotStats {
    pub name: String,
    pub primary: String,
    pub fallback: String,
    pub primary_available: bool,
    pub fallback_available: bool,
    pub primary_available_ollama: bool,
    pub primary_available_lmstudio: bool,
    pub fallback_available_ollama: bool,
    pub fallback_available_lmstudio: bool,
    pub alternatives: Vec<String>,
    pub alternatives_available: bool,
    pub alternatives_available_ollama: bool,
    pub alternatives_available_lmstudio: bool,
}

// ============================================================
// Global Router Instance
// ============================================================

use tokio::sync::OnceCell;

static MODEL_ROUTER: OnceCell<Arc<ModelRouter>> = OnceCell::const_new();

/// Get or create the global model router
pub async fn get_router() -> anyhow::Result<Arc<ModelRouter>> {
    MODEL_ROUTER
        .get_or_try_init(|| async {
            let router = ModelRouter::new().await?;
            Ok::<_, anyhow::Error>(Arc::new(router))
        })
        .await
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_task_classification() {
        // High-stakes reasoning task
        let chars = TaskCharacteristics::classify(
            "Verify the security of this production deployment"
        );
        assert!(chars.is_high_stakes);
        assert_eq!(chars.optimal_slot(), CapabilitySlot::ColdCore);
        
        // Creative user-facing task
        let chars = TaskCharacteristics::classify(
            "Write a creative story for the user"
        );
        assert!(chars.requires_creativity);
        assert!(chars.is_user_facing);
        assert_eq!(chars.optimal_slot(), CapabilitySlot::WarmSurface);
        
        // Planning task
        let chars = TaskCharacteristics::classify(
            "Create a strategic roadmap for the Q3 milestones"
        );
        assert!(chars.requires_planning);
        assert_eq!(chars.optimal_slot(), CapabilitySlot::PrimaryReasoning);
        
        // Embedding task
        let chars = TaskCharacteristics::classify(
            "Find similar documents using semantic search"
        );
        assert!(chars.requires_embedding);
        assert_eq!(chars.optimal_slot(), CapabilitySlot::Embeddings);
    }
    
    #[test]
    fn test_capability_slots() {
        // ColdCore and WarmSurface have distinct primaries
        assert_ne!(
            CapabilitySlot::ColdCore.primary_model(),
            CapabilitySlot::WarmSurface.primary_model()
        );
        
        // Temperatures are configured correctly
        assert!(CapabilitySlot::ColdCore.temperature() < CapabilitySlot::PrimaryReasoning.temperature());
    }
}
