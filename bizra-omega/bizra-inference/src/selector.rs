//! Model Selector — Adaptive tier selection

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelTier {
    Edge,  // 0.5B-1.5B, <1GB VRAM
    Local, // 7B-13B, 4-8GB VRAM
    Pool,  // 70B+, federated
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskComplexity {
    Simple,
    Medium,
    Complex,
    Expert,
}

impl TaskComplexity {
    pub fn estimate(prompt: &str, max_tokens: usize) -> Self {
        let words = prompt.split_whitespace().count();
        let has_code = prompt.contains("```");
        let has_reasoning = prompt.to_lowercase().contains("explain");

        if max_tokens > 2000 || (has_reasoning && words > 200) {
            Self::Expert
        } else if has_code || has_reasoning || words > 100 {
            Self::Complex
        } else if words > 30 || max_tokens > 500 {
            Self::Medium
        } else {
            Self::Simple
        }
    }
}

pub struct ModelSelector {
    forced_tier: Option<ModelTier>,
}

impl ModelSelector {
    pub fn new() -> Self {
        Self { forced_tier: None }
    }

    pub fn force_tier(&mut self, tier: ModelTier) {
        self.forced_tier = Some(tier);
    }

    pub fn select_tier(&self, complexity: &TaskComplexity) -> ModelTier {
        if let Some(t) = self.forced_tier {
            return t;
        }
        match complexity {
            TaskComplexity::Simple | TaskComplexity::Medium => ModelTier::Edge,
            TaskComplexity::Complex => ModelTier::Local,
            TaskComplexity::Expert => ModelTier::Pool,
        }
    }
}

impl Default for ModelSelector {
    fn default() -> Self {
        Self::new()
    }
}
