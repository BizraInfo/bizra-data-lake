// bizra-mission/src/preflight.rs
// ============================================================
// Model Preflight — constitutional pre-check before queueing
// ============================================================
//
// "No mission contract may require a specific model family."
// Preflight checks CAPABILITIES, not model names.
// ============================================================

use serde::{Deserialize, Serialize};

/// Result of model preflight check.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum PreflightResult {
    /// Model is installed and ready.
    Ready { model: String, vram_used_mb: u64 },
    /// Model is installed but needs loading.
    NeedsWarmup {
        model: String,
        estimated_warmup_ms: u64,
    },
    /// Requested capability not available with preferred model. Fallback selected.
    FallbackUsed {
        requested: String,
        fallback: String,
        reason: String,
    },
    /// No model can serve the required capability.
    NoModelAvailable { reason: String },
}

impl PreflightResult {
    /// Did preflight pass (mission can proceed)?
    pub fn passed(&self) -> bool {
        !matches!(self, Self::NoModelAvailable { .. })
    }

    /// The model that will be used (if any).
    pub fn chosen_model(&self) -> Option<&str> {
        match self {
            Self::Ready { model, .. } => Some(model),
            Self::NeedsWarmup { model, .. } => Some(model),
            Self::FallbackUsed { fallback, .. } => Some(fallback),
            Self::NoModelAvailable { .. } => None,
        }
    }
}

/// Capability that a mission can request.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    Chat,
    Code,
    Reasoning,
    Vision,
    Embedding,
    ToolUse,
    MemoryRetrieval,
}

/// Run preflight against a list of available model names.
/// This is a simplified check — the full implementation will
/// use the ResourceManifest and ModelRegistry.
pub fn run_preflight(
    required_capabilities: &[Capability],
    available_models: &[String],
    preferred_model: Option<&str>,
) -> PreflightResult {
    // If no models available at all, fail immediately
    if available_models.is_empty() {
        return PreflightResult::NoModelAvailable {
            reason: "no models installed on this node".to_string(),
        };
    }

    // Check if preferred model is available
    if let Some(preferred) = preferred_model {
        if available_models.iter().any(|m| m == preferred) {
            return PreflightResult::Ready {
                model: preferred.to_string(),
                vram_used_mb: 0,
            };
        }
        // Preferred not available, use first available as fallback
        return PreflightResult::FallbackUsed {
            requested: preferred.to_string(),
            fallback: available_models[0].clone(),
            reason: format!("preferred model '{}' not installed", preferred),
        };
    }

    // No preference — use first available model
    // For capabilities like Vision, check if any vision model exists
    if required_capabilities.contains(&Capability::Vision) {
        let has_vision = available_models
            .iter()
            .any(|m| m.contains("VL") || m.contains("vision") || m.contains("moondream"));
        if !has_vision {
            return PreflightResult::NoModelAvailable {
                reason: "vision capability required but no vision model installed".to_string(),
            };
        }
    }

    PreflightResult::Ready {
        model: available_models[0].clone(),
        vram_used_mb: 0,
    }
}
