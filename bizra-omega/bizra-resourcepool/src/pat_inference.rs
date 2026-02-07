//! PAT-LM Studio Integration
//!
//! Routes each PAT agent to the optimal LM Studio model:
//! - Strategist → Reasoning model (DeepSeek-R1)
//! - Researcher → Reasoning + Embedding
//! - Developer → Code model (Qwen-Coder)
//! - Analyst → Reasoning model
//! - Reviewer → Code + Reasoning
//! - Executor → Agentic model (function calling)
//! - Guardian → Reasoning + Vision (security analysis)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// PAT Agent roles
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PATRole {
    Strategist,
    Researcher,
    Developer,
    Analyst,
    Reviewer,
    Executor,
    Guardian,
}

impl PATRole {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "strategist" => Some(Self::Strategist),
            "researcher" => Some(Self::Researcher),
            "developer" => Some(Self::Developer),
            "analyst" => Some(Self::Analyst),
            "reviewer" => Some(Self::Reviewer),
            "executor" => Some(Self::Executor),
            "guardian" => Some(Self::Guardian),
            _ => None,
        }
    }
}

/// Model capability for routing
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InferenceCapability {
    Reasoning,
    Agentic,
    Vision,
    Voice,
    Code,
    Embedding,
    Chat,
}

/// PAT Inference configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PATInferenceConfig {
    /// LM Studio host
    pub lmstudio_host: String,
    /// LM Studio port
    pub lmstudio_port: u16,
    /// Model mappings by capability
    pub models: HashMap<InferenceCapability, String>,
    /// Role-to-capability mapping
    pub role_capabilities: HashMap<PATRole, Vec<InferenceCapability>>,
}

impl Default for PATInferenceConfig {
    fn default() -> Self {
        let mut models = HashMap::new();
        models.insert(
            InferenceCapability::Reasoning,
            std::env::var("LMSTUDIO_MODEL_REASONING")
                .unwrap_or_else(|_| "deepseek-r1-distill-qwen-32b".to_string()),
        );
        models.insert(
            InferenceCapability::Agentic,
            std::env::var("LMSTUDIO_MODEL_AGENTIC")
                .unwrap_or_else(|_| "qwen2.5-32b-instruct".to_string()),
        );
        models.insert(
            InferenceCapability::Vision,
            std::env::var("LMSTUDIO_MODEL_VISION")
                .unwrap_or_else(|_| "llava-v1.6-mistral-7b".to_string()),
        );
        models.insert(
            InferenceCapability::Voice,
            std::env::var("LMSTUDIO_MODEL_VOICE")
                .unwrap_or_else(|_| "whisper-large-v3".to_string()),
        );
        models.insert(
            InferenceCapability::Code,
            std::env::var("LMSTUDIO_MODEL_CODE")
                .unwrap_or_else(|_| "qwen2.5-coder-32b".to_string()),
        );
        models.insert(
            InferenceCapability::Embedding,
            std::env::var("LMSTUDIO_MODEL_EMBEDDING")
                .unwrap_or_else(|_| "nomic-embed-text".to_string()),
        );
        models.insert(
            InferenceCapability::Chat,
            std::env::var("LMSTUDIO_MODEL_DEFAULT")
                .unwrap_or_else(|_| "qwen2.5-7b-instruct".to_string()),
        );

        let mut role_capabilities = HashMap::new();

        // Strategist: Deep reasoning for planning and analysis
        role_capabilities.insert(PATRole::Strategist, vec![InferenceCapability::Reasoning]);

        // Researcher: Reasoning for synthesis + Embedding for search
        role_capabilities.insert(
            PATRole::Researcher,
            vec![
                InferenceCapability::Reasoning,
                InferenceCapability::Embedding,
            ],
        );

        // Developer: Code generation and understanding
        role_capabilities.insert(
            PATRole::Developer,
            vec![InferenceCapability::Code, InferenceCapability::Reasoning],
        );

        // Analyst: Pattern recognition and reasoning
        role_capabilities.insert(
            PATRole::Analyst,
            vec![
                InferenceCapability::Reasoning,
                InferenceCapability::Embedding,
            ],
        );

        // Reviewer: Code review + reasoning for quality
        role_capabilities.insert(
            PATRole::Reviewer,
            vec![InferenceCapability::Code, InferenceCapability::Reasoning],
        );

        // Executor: Agentic for autonomous execution
        role_capabilities.insert(
            PATRole::Executor,
            vec![InferenceCapability::Agentic, InferenceCapability::Code],
        );

        // Guardian: Vision for security analysis + reasoning
        role_capabilities.insert(
            PATRole::Guardian,
            vec![InferenceCapability::Reasoning, InferenceCapability::Vision],
        );

        Self {
            lmstudio_host: std::env::var("LMSTUDIO_HOST")
                .unwrap_or_else(|_| "192.168.56.1".to_string()),
            lmstudio_port: std::env::var("LMSTUDIO_PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(1234),
            models,
            role_capabilities,
        }
    }
}

impl PATInferenceConfig {
    /// Get the primary model for a PAT role
    pub fn model_for_role(&self, role: &PATRole) -> String {
        if let Some(caps) = self.role_capabilities.get(role) {
            if let Some(cap) = caps.first() {
                if let Some(model) = self.models.get(cap) {
                    return model.clone();
                }
            }
        }
        self.models
            .get(&InferenceCapability::Chat)
            .cloned()
            .unwrap_or_default()
    }

    /// Get all capable models for a PAT role
    pub fn models_for_role(&self, role: &PATRole) -> Vec<(InferenceCapability, String)> {
        match self.role_capabilities.get(role) {
            Some(caps) => caps
                .iter()
                .filter_map(|cap| self.models.get(cap).map(|m| (cap.clone(), m.clone())))
                .collect(),
            None => Vec::new(),
        }
    }

    /// Get LM Studio base URL
    pub fn base_url(&self) -> String {
        format!("http://{}:{}/v1", self.lmstudio_host, self.lmstudio_port)
    }
}

/// System prompts for each PAT role
pub fn system_prompt_for_role(role: &PATRole) -> &'static str {
    match role {
        PATRole::Strategist => {
            r#"You are the Strategist of a Personal Agentic Team (PAT).
Your role: Plan, analyze, and decide strategic directions.
Giants you stand on: Sun Tzu, Clausewitz, Porter.
Approach: Think deeply about long-term implications. Consider multiple scenarios.
Always validate decisions against the Ihsān constraint (≥0.95 excellence).
Output format: Structured analysis with clear recommendations."#
        }

        PATRole::Researcher => {
            r#"You are the Researcher of a Personal Agentic Team (PAT).
Your role: Search, synthesize, and cite information.
Giants you stand on: Shannon, Besta, Hinton.
Approach: Gather comprehensive information. Cross-reference sources. Synthesize insights.
Always cite sources and quantify confidence levels.
Output format: Research report with citations and confidence scores."#
        }

        PATRole::Developer => {
            r#"You are the Developer of a Personal Agentic Team (PAT).
Your role: Code, test, and deploy software.
Giants you stand on: Knuth, Dijkstra, Thompson.
Approach: Write clean, efficient, well-documented code. Follow TDD principles.
Always consider edge cases and security implications.
Output format: Code with tests and documentation."#
        }

        PATRole::Analyst => {
            r#"You are the Analyst of a Personal Agentic Team (PAT).
Your role: Pattern recognition, measurement, and prediction.
Giants you stand on: Tukey, Fisher, Bayes.
Approach: Apply statistical rigor. Identify patterns. Quantify uncertainty.
Always show your work and confidence intervals.
Output format: Analysis with metrics and visualizations."#
        }

        PATRole::Reviewer => {
            r#"You are the Reviewer of a Personal Agentic Team (PAT).
Your role: Validate, critique, and improve work.
Giants you stand on: Hoare, Dijkstra, Meyer.
Approach: Apply rigorous quality standards. Identify issues constructively.
Focus on correctness, security, and maintainability.
Output format: Review with specific issues and recommendations."#
        }

        PATRole::Executor => {
            r#"You are the Executor of a Personal Agentic Team (PAT).
Your role: Execute tasks, monitor progress, and report status.
Giants you stand on: Deming, Taylor, Ohno.
Approach: Take autonomous action within bounds. Report progress proactively.
Escalate blockers immediately. Optimize for reliability.
Output format: Execution log with status and next steps."#
        }

        PATRole::Guardian => {
            r#"You are the Guardian of a Personal Agentic Team (PAT).
Your role: Protect, audit, and enforce ethical constraints.
Giants you stand on: Al-Ghazali, Rawls, Anthropic.
Approach: Ensure all actions meet Ihsān (excellence) and Adl (justice) standards.
Protect against harm. Audit for compliance. Enforce the constitution.
FATE Gates: Ihsān ≥ 0.95, Adl Gini ≤ 0.35, Harm ≤ 0.30
Output format: Audit report with compliance status."#
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_parsing() {
        assert_eq!(PATRole::from_str("strategist"), Some(PATRole::Strategist));
        assert_eq!(PATRole::from_str("DEVELOPER"), Some(PATRole::Developer));
        assert_eq!(PATRole::from_str("Guardian"), Some(PATRole::Guardian));
        assert_eq!(PATRole::from_str("unknown"), None);
    }

    #[test]
    fn test_default_config() {
        let config = PATInferenceConfig::default();
        assert!(!config.models.is_empty());
        assert_eq!(config.role_capabilities.len(), 7);
    }

    #[test]
    fn test_model_for_role() {
        let config = PATInferenceConfig::default();

        // Strategist should get reasoning model
        let model = config.model_for_role(&PATRole::Strategist);
        assert!(model.contains("deepseek") || model.contains("qwen"));

        // Developer should get code model
        let model = config.model_for_role(&PATRole::Developer);
        assert!(model.contains("coder") || model.contains("code"));
    }

    #[test]
    fn test_system_prompts() {
        let prompt = system_prompt_for_role(&PATRole::Guardian);
        assert!(prompt.contains("Al-Ghazali"));
        assert!(prompt.contains("Ihsān"));

        let prompt = system_prompt_for_role(&PATRole::Developer);
        assert!(prompt.contains("Knuth"));
        assert!(prompt.contains("Dijkstra"));
    }
}
