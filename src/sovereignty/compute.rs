// src/sovereignty/compute.rs - Compute Sovereignty (Pillar 3: Runtime)
//
// Principle: Works offline / degraded mode without cloud.
// Models run locally or in a federation you control.

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Compute provider types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComputeProvider {
    /// Ollama (local)
    Ollama,
    /// LM Studio (local)
    LmStudio,
    /// Federation peer node
    Federation,
    /// Cached patterns (no inference)
    Cache,
    /// WASM sandbox
    Wasm,
}

impl ComputeProvider {
    /// Is this provider local?
    pub fn is_local(&self) -> bool {
        matches!(
            self,
            Self::Ollama | Self::LmStudio | Self::Cache | Self::Wasm
        )
    }

    /// Is this provider sovereign (no cloud)?
    pub fn is_sovereign(&self) -> bool {
        // All current providers are sovereign
        true
    }
}

/// Compute capability slot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilitySlot {
    /// Slot name
    pub name: String,
    /// Description
    pub description: String,
    /// Primary provider
    pub primary: ComputeProvider,
    /// Fallback chain
    pub fallbacks: Vec<ComputeProvider>,
    /// Currently active provider
    pub active: Option<ComputeProvider>,
    /// Last health check
    pub last_health_check: Option<chrono::DateTime<chrono::Utc>>,
    /// Health status
    pub is_healthy: bool,
}

impl CapabilitySlot {
    /// Create new slot
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        primary: ComputeProvider,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            primary,
            fallbacks: Vec::new(),
            active: None,
            last_health_check: None,
            is_healthy: false,
        }
    }

    /// Add fallback provider
    pub fn with_fallback(mut self, provider: ComputeProvider) -> Self {
        self.fallbacks.push(provider);
        self
    }

    /// Get next available provider
    pub fn next_provider(&self) -> Option<ComputeProvider> {
        if self.is_healthy && self.active.is_some() {
            return self.active;
        }

        // Try primary first
        Some(self.primary)
    }
}

/// Degradation level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum DegradationLevel {
    /// Full capability (all providers healthy)
    Full = 0,
    /// Reduced (some providers offline)
    Reduced = 1,
    /// Minimal (only cache available)
    Minimal = 2,
    /// Offline (no inference, pattern match only)
    Offline = 3,
}

impl DegradationLevel {
    /// Human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Full => "Full Stack",
            Self::Reduced => "Reduced Capability",
            Self::Minimal => "Minimal Mode",
            Self::Offline => "Offline Mode",
        }
    }

    /// Can perform inference?
    pub fn can_infer(&self) -> bool {
        matches!(self, Self::Full | Self::Reduced | Self::Minimal)
    }
}

/// Health probe result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthProbe {
    pub provider: ComputeProvider,
    pub endpoint: String,
    pub is_healthy: bool,
    pub latency_ms: u64,
    pub models_available: Vec<String>,
    pub error: Option<String>,
    pub probed_at: chrono::DateTime<chrono::Utc>,
}

/// Compute runtime status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeStatus {
    /// Current degradation level
    pub level: DegradationLevel,
    /// Active providers
    pub active_providers: Vec<ComputeProvider>,
    /// Provider health
    pub health: Vec<HealthProbe>,
    /// Total available models
    pub models_available: usize,
    /// Last status update
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// Compute sovereignty manager
pub struct ComputeManager {
    /// Capability slots
    slots: Vec<CapabilitySlot>,
    /// Current degradation level
    degradation: DegradationLevel,
    /// Provider health cache
    health_cache: Vec<HealthProbe>,
    /// Blocked providers (cloud APIs)
    blocked_providers: Vec<String>,
}

impl ComputeManager {
    /// Create with default slots from model-family-genesis
    pub fn new() -> Self {
        let slots = vec![
            CapabilitySlot::new(
                "cold_core",
                "Deterministic reasoning + self-correction",
                ComputeProvider::Ollama,
            )
            .with_fallback(ComputeProvider::LmStudio),
            CapabilitySlot::new(
                "warm_surface",
                "Nuance + formatting + user-facing",
                ComputeProvider::Ollama,
            )
            .with_fallback(ComputeProvider::LmStudio),
            CapabilitySlot::new(
                "embeddings",
                "Deterministic embedding for RAG",
                ComputeProvider::Ollama,
            ),
            CapabilitySlot::new(
                "primary_reasoning",
                "Multi-agent orchestration",
                ComputeProvider::Ollama,
            )
            .with_fallback(ComputeProvider::LmStudio),
            CapabilitySlot::new(
                "vision",
                "Vision-capable inference",
                ComputeProvider::LmStudio,
            )
            .with_fallback(ComputeProvider::Ollama),
        ];

        Self {
            slots,
            degradation: DegradationLevel::Full,
            health_cache: Vec::new(),
            blocked_providers: vec![
                "api.openai.com".to_string(),
                "api.anthropic.com".to_string(),
                "generativelanguage.googleapis.com".to_string(),
            ],
        }
    }

    /// Check if endpoint is blocked (cloud API)
    pub fn is_blocked(&self, endpoint: &str) -> bool {
        self.blocked_providers.iter().any(|b| endpoint.contains(b))
    }

    /// Get current degradation level
    pub fn degradation_level(&self) -> DegradationLevel {
        self.degradation
    }

    /// Get available slots
    pub fn slots(&self) -> &[CapabilitySlot] {
        &self.slots
    }

    /// Get slot by name
    pub fn get_slot(&self, name: &str) -> Option<&CapabilitySlot> {
        self.slots.iter().find(|s| s.name == name)
    }

    /// Update health status for a provider
    pub fn update_health(&mut self, probe: HealthProbe) {
        // Remove old probe for this provider
        self.health_cache.retain(|p| p.provider != probe.provider);
        self.health_cache.push(probe);

        // Recalculate degradation level
        self.recalculate_degradation();
    }

    /// Recalculate degradation level based on health
    fn recalculate_degradation(&mut self) {
        let healthy_count = self.health_cache.iter().filter(|p| p.is_healthy).count();
        let total_providers = 2; // Ollama + LM Studio

        self.degradation = match healthy_count {
            n if n >= total_providers => DegradationLevel::Full,
            1 => DegradationLevel::Reduced,
            0 if !self.health_cache.is_empty() => DegradationLevel::Minimal,
            _ => DegradationLevel::Offline,
        };
    }

    /// Get compute status
    pub fn status(&self) -> ComputeStatus {
        let active_providers: Vec<_> = self
            .health_cache
            .iter()
            .filter(|p| p.is_healthy)
            .map(|p| p.provider)
            .collect();

        let models_available: usize = self
            .health_cache
            .iter()
            .filter(|p| p.is_healthy)
            .map(|p| p.models_available.len())
            .sum();

        ComputeStatus {
            level: self.degradation,
            active_providers,
            health: self.health_cache.clone(),
            models_available,
            updated_at: chrono::Utc::now(),
        }
    }

    /// Check sovereignty invariant (no cloud APIs)
    pub fn is_sovereign(&self) -> bool {
        // Check no cloud API keys in environment
        let has_cloud_keys = std::env::var("OPENAI_API_KEY").is_ok()
            || std::env::var("ANTHROPIC_API_KEY").is_ok()
            || std::env::var("GOOGLE_API_KEY").is_ok();

        !has_cloud_keys
    }
}

impl Default for ComputeManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_is_sovereign() {
        assert!(ComputeProvider::Ollama.is_sovereign());
        assert!(ComputeProvider::LmStudio.is_sovereign());
        assert!(ComputeProvider::Federation.is_sovereign());
    }

    #[test]
    fn test_degradation_levels() {
        assert!(DegradationLevel::Full.can_infer());
        assert!(DegradationLevel::Reduced.can_infer());
        assert!(!DegradationLevel::Offline.can_infer());
    }

    #[test]
    fn test_blocked_endpoints() {
        let manager = ComputeManager::new();

        assert!(manager.is_blocked("https://api.openai.com/v1/chat"));
        assert!(manager.is_blocked("https://api.anthropic.com/v1/messages"));
        assert!(!manager.is_blocked("http://localhost:11434/api/generate"));
    }

    #[test]
    fn test_capability_slots() {
        let manager = ComputeManager::new();

        assert_eq!(manager.slots().len(), 5);
        assert!(manager.get_slot("cold_core").is_some());
        assert!(manager.get_slot("nonexistent").is_none());
    }
}
