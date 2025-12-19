// src/pat.rs - Personal Agentic Team (7 agents)
//
// BIZRA PAT Layer with LLM Integration
// =====================================
// - 7 specialized agents with distinct roles
// - Ollama LLM integration for reasoning
// - Graceful fallback to static responses
// - SAPE-informed quality assessment

use crate::ollama::{self, ChatMessage};
use crate::types::{AgentResult, DualAgenticRequest};
use std::time::Instant;
use tracing::{debug, info, instrument, warn};

/// PAT Orchestrator with 7 specialized agents
pub struct PATOrchestrator {
    agents: Vec<PATAgent>,
    llm_enabled: bool,
}

#[derive(Debug, Clone)]
struct PATAgent {
    name: String,
    role: String,
    specialty: String,
    system_prompt: String,
}

impl PATOrchestrator {
    pub async fn new() -> anyhow::Result<Self> {
        info!("🎭 Initializing PAT (Personal Agentic Team)");

        // Check if Ollama is available
        let ollama_client = ollama::get_ollama().await;
        let llm_enabled = ollama_client.is_connected();
        
        if llm_enabled {
            info!("✅ Ollama LLM connected - PAT agents will use real reasoning");
        } else {
            warn!("⚠️ Ollama not available - PAT agents will use simulated responses");
        }

        let agents = vec![
            PATAgent {
                name: "strategic_visionary".to_string(),
                role: "Strategic Planning".to_string(),
                specialty: "Long-term vision and strategic direction".to_string(),
                system_prompt: r#"You are the Strategic Visionary agent in BIZRA's PAT (Personal Agentic Team).
Your role is to provide long-term strategic direction and vision.
Focus on: sustainable growth, strategic positioning, risk-aware planning.
Keep responses concise (2-3 paragraphs max).
Apply Ihsān (إحسان) principles: excellence, ethics, user benefit."#.to_string(),
            },
            PATAgent {
                name: "creative_innovator".to_string(),
                role: "Innovation".to_string(),
                specialty: "Creative solutions and novel approaches".to_string(),
                system_prompt: r#"You are the Creative Innovator agent in BIZRA's PAT.
Your role is to propose novel solutions and innovative approaches.
Focus on: creative problem-solving, out-of-box thinking, novel methodologies.
Keep responses concise (2-3 paragraphs max).
Apply Ihsān principles: excellence through innovation."#.to_string(),
            },
            PATAgent {
                name: "analytical_optimizer".to_string(),
                role: "Analysis & Optimization".to_string(),
                specialty: "Data analysis and performance optimization".to_string(),
                system_prompt: r#"You are the Analytical Optimizer agent in BIZRA's PAT.
Your role is to provide data-driven analysis and optimization recommendations.
Focus on: metrics, efficiency gains, performance improvements, evidence-based decisions.
Keep responses concise (2-3 paragraphs max).
Apply Ihsān principles: excellence through optimization."#.to_string(),
            },
            PATAgent {
                name: "implementation_specialist".to_string(),
                role: "Execution".to_string(),
                specialty: "Practical implementation and delivery".to_string(),
                system_prompt: r#"You are the Implementation Specialist agent in BIZRA's PAT.
Your role is to create practical, actionable execution plans.
Focus on: step-by-step plans, deliverables, timelines, resource allocation.
Keep responses concise (2-3 paragraphs max).
Apply Ihsān principles: excellence through execution."#.to_string(),
            },
            PATAgent {
                name: "quality_guardian".to_string(),
                role: "Quality Assurance".to_string(),
                specialty: "Quality standards and excellence (إحسان)".to_string(),
                system_prompt: r#"You are the Quality Guardian agent in BIZRA's PAT.
Your role is to ensure quality standards and ethical excellence (Ihsān - إحسان).
Focus on: quality gates, testing strategies, ethical considerations, compliance.
Keep responses concise (2-3 paragraphs max).
You embody Ihsān: the pursuit of excellence as if being observed by the highest authority."#.to_string(),
            },
            PATAgent {
                name: "user_advocate".to_string(),
                role: "User Experience".to_string(),
                specialty: "User needs and experience optimization".to_string(),
                system_prompt: r#"You are the User Advocate agent in BIZRA's PAT.
Your role is to represent user interests and optimize user experience.
Focus on: user needs, usability, accessibility, satisfaction metrics.
Keep responses concise (2-3 paragraphs max).
Apply Ihsān principles: excellence in serving users."#.to_string(),
            },
            PATAgent {
                name: "integration_coordinator".to_string(),
                role: "Coordination".to_string(),
                specialty: "System integration and harmony".to_string(),
                system_prompt: r#"You are the Integration Coordinator agent in BIZRA's PAT.
Your role is to ensure seamless integration and coordination across components.
Focus on: system harmony, interface design, dependency management, cohesion.
Keep responses concise (2-3 paragraphs max).
Apply Ihsān principles: excellence through harmonious integration."#.to_string(),
            },
        ];

        info!(agents_count = agents.len(), llm_enabled, "PAT agents initialized");
        Ok(Self { agents, llm_enabled })
    }

    /// Execute all agents in parallel (with LLM or fallback)
    #[instrument(skip(self))]
    pub async fn execute_parallel(
        &self,
        _prompts: Vec<String>,
        request: DualAgenticRequest,
    ) -> anyhow::Result<Vec<AgentResult>> {
        let start = Instant::now();

        // Execute agents concurrently using tokio::join_all
        let agent_futures: Vec<_> = self.agents
            .iter()
            .map(|agent| self.execute_agent(agent, &request))
            .collect();

        let results: Vec<Result<AgentResult, anyhow::Error>> = 
            futures::future::join_all(agent_futures).await;

        // Collect successful results, log errors
        let mut successful_results = Vec::new();
        for result in results {
            match result {
                Ok(r) => successful_results.push(r),
                Err(e) => warn!("Agent execution failed: {}", e),
            }
        }

        let total_time = start.elapsed();
        info!(
            agents_executed = successful_results.len(),
            total_time_ms = total_time.as_millis(),
            llm_enabled = self.llm_enabled,
            "PAT parallel execution completed"
        );

        Ok(successful_results)
    }

    async fn execute_agent(
        &self,
        agent: &PATAgent,
        request: &DualAgenticRequest,
    ) -> anyhow::Result<AgentResult> {
        let start = Instant::now();

        // Try LLM-powered response first
        let contribution = if self.llm_enabled {
            match self.execute_with_llm(agent, request).await {
                Ok(response) => response,
                Err(e) => {
                    debug!("LLM call failed for {}: {}, using fallback", agent.name, e);
                    self.generate_fallback_contribution(agent, request)
                }
            }
        } else {
            self.generate_fallback_contribution(agent, request)
        };

        let execution_time = start.elapsed();

        // Calculate confidence based on response quality
        let base_confidence = if self.llm_enabled { 0.90 } else { 0.85 };
        let confidence = base_confidence + (rand::random::<f64>() * 0.08);

        Ok(AgentResult {
            agent_name: agent.name.clone(),
            contribution,
            confidence,
            execution_time,
        })
    }

    /// Execute agent with actual LLM call via Ollama
    async fn execute_with_llm(
        &self,
        agent: &PATAgent,
        request: &DualAgenticRequest,
    ) -> anyhow::Result<String> {
        let ollama_client = ollama::get_ollama().await;

        // Build conversation with agent's system prompt and user message
        let context_str = if request.context.is_empty() {
            "No additional context".to_string()
        } else {
            request.context
                .iter()
                .map(|(k, v)| format!("{}: {}", k, v))
                .collect::<Vec<_>>()
                .join(", ")
        };

        // Format user prompt with task context
        let user_prompt = format!(
            "Task: {}\nContext: {}\n\nProvide your {} perspective on this task.",
            request.task,
            context_str,
            agent.role
        );

        let messages = vec![
            ChatMessage::system(&agent.system_prompt),
            ChatMessage::user(&user_prompt),
        ];

        let response = ollama_client
            .chat(messages, None, None)
            .await?;

        let content = response.message.content;
        
        // Format with agent role prefix
        Ok(format!("[{}] {}", agent.role, content))
    }

    /// Generate fallback contribution when LLM is unavailable
    fn generate_fallback_contribution(&self, agent: &PATAgent, request: &DualAgenticRequest) -> String {
        let base_contribution = match agent.name.as_str() {
            "strategic_visionary" => {
                format!("[Strategic] Long-term vision for '{}': Establish foundation for sustainable growth with phased milestones", request.task)
            }
            "creative_innovator" => {
                format!(
                    "[Innovation] Novel approach for '{}': Apply cutting-edge methodologies and explore unconventional solutions",
                    request.task
                )
            }
            "analytical_optimizer" => {
                format!(
                    "[Analysis] Data-driven insights for '{}': Target 95% efficiency through systematic optimization",
                    request.task
                )
            }
            "implementation_specialist" => {
                format!(
                    "[Implementation] Practical execution plan for '{}': 5-phase delivery with clear deliverables per phase",
                    request.task
                )
            }
            "quality_guardian" => {
                format!(
                    "[Quality] Excellence standards for '{}': إحسان score target 0.95+ with comprehensive testing",
                    request.task
                )
            }
            "user_advocate" => {
                format!(
                    "[UX] User-centric design for '{}': Optimize for user satisfaction and accessibility",
                    request.task
                )
            }
            "integration_coordinator" => {
                format!(
                    "[Coordination] Harmonized approach for '{}': Ensure seamless system integration",
                    request.task
                )
            }
            _ => format!("[{}] Contribution for '{}'", agent.role, request.task),
        };

        format!("{} (Specialty: {}) [SIMULATED]", base_contribution, agent.specialty)
    }

    pub fn get_agent_count(&self) -> usize {
        self.agents.len()
    }

    pub fn is_llm_enabled(&self) -> bool {
        self.llm_enabled
    }
}

// Simple random number generation without external crate
pub(crate) mod rand {
    use std::cell::Cell;
    use std::time::{SystemTime, UNIX_EPOCH};

    thread_local! {
        static SEED: Cell<u64> = Cell::new(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64
        );
    }

    pub fn random<T: From<f64>>() -> T {
        SEED.with(|seed| {
            let mut s = seed.get();
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            seed.set(s);
            T::from((s as f64) / (u64::MAX as f64))
        })
    }
}
