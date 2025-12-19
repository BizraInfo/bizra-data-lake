// src/pat_enhanced.rs - PAT with all capabilities

use crate::{
    a2a::A2AServer, ihsan, mcp::MCPClient, pat::PATOrchestrator, reasoning::MultiMethodReasoning,
    types::*,
};
use std::{collections::BTreeMap, sync::Arc, time::Instant};
use tokio::sync::RwLock;
use tracing::{info, instrument};

pub struct EnhancedPATOrchestrator {
    // Base orchestrator
    base: Arc<PATOrchestrator>,

    // Enhanced capabilities
    mcp_client: Arc<RwLock<MCPClient>>,
    a2a_server: Arc<RwLock<A2AServer>>,
    reasoning_engine: Arc<MultiMethodReasoning>,

    // Sub-agent factory
    sub_agent_count: Arc<RwLock<usize>>,
    max_sub_agents: usize,
}

impl EnhancedPATOrchestrator {
    pub async fn new() -> anyhow::Result<Self> {
        info!("🎭 Initializing ENHANCED PAT with full arsenal");

        let mut mcp = MCPClient::new();

        // Register MCP servers
        mcp.register_server(
            "local_tools".to_string(),
            "stdio://local".to_string(),
            crate::mcp::MCPTransport::Stdio,
        )
        .await?;

        let mut a2a = A2AServer::new();

        // Register agent capabilities
        a2a.register_agent(crate::a2a::AgentCard {
            name: "strategic_visionary".to_string(),
            version: "2.0.0".to_string(),
            capabilities: vec![crate::a2a::Capability {
                id: "strategic_planning".to_string(),
                description: "Long-term strategic planning".to_string(),
                inputs: vec!["goals".to_string(), "constraints".to_string()],
                outputs: vec!["strategic_plan".to_string()],
            }],
            protocols: vec!["a2a".to_string()],
            authentication: vec!["oauth2".to_string()],
        });

        Ok(Self {
            base: Arc::new(PATOrchestrator::new().await?),
            mcp_client: Arc::new(RwLock::new(mcp)),
            a2a_server: Arc::new(RwLock::new(a2a)),
            reasoning_engine: Arc::new(MultiMethodReasoning::new(vec![
                ReasoningMethod::ChainOfThought,
                ReasoningMethod::TreeOfThought,
                ReasoningMethod::GraphOfThought,
                ReasoningMethod::ReAct,
                ReasoningMethod::Reflexion,
            ])),
            sub_agent_count: Arc::new(RwLock::new(0)),
            max_sub_agents: 100,
        })
    }

    #[instrument(skip(self))]
    pub async fn execute_enhanced(
        &self,
        request: EnhancedDualAgenticRequest,
    ) -> anyhow::Result<DualAgenticResponse> {
        let start = Instant::now();
        info!("🚀 Enhanced PAT execution with full capabilities");

        // Handle slash commands
        if let Some(cmd) = &request.slash_command {
            return self.handle_slash_command(cmd, &request).await;
        }

        // Select reasoning method
        let method = self.reasoning_engine.select_method(
            "general",
            0.5,
            request.reasoning_preference.clone(),
        );

        info!(?method, "Selected reasoning method");

        // Execute with MCP tools if needed
        let mcp = self.mcp_client.read().await;
        let available_tools = mcp.list_tools();
        info!(tools_count = available_tools.len(), "MCP tools available");

        // Spawn sub-agents if enabled
        if request.enable_sub_agents {
            let mut count = self.sub_agent_count.write().await;
            if *count < self.max_sub_agents {
                *count += 1;
                info!(sub_agents = *count, "Sub-agent spawned");
            }
        }

        // Execute base orchestration
        let base_result = self
            .base
            .execute_parallel(vec![], request.base.clone())
            .await?;

        let pat_avg = avg_confidence(&base_result);
        let (ihsan_score, ihsan_vector) = self.calculate_ihsan_pat_only(&base_result)?;
        let (ihsan_env, ihsan_threshold_applied, ihsan_passes_threshold) =
            self.enforce_ihsan(ihsan_score, "docs")?;
        let latency = start.elapsed();

        // Build enhanced response
        Ok(DualAgenticResponse {
            pat_contributions: base_result.iter().map(|r| r.contribution.clone()).collect(),
            sat_contributions: vec![],
            synergy_score: pat_avg,
            ihsan_score,
            latency,
            meta: serde_json::json!({
                "reasoning_method": format!("{:?}", method),
                "mcp_tools_available": available_tools.len(),
                "sub_agents_spawned": *self.sub_agent_count.read().await,
                "sat_absent": true,
                "synergy_score_source": "pat_avg_confidence_v0",
                "adapter_modes": AdapterModes::current(),
                "ihsan_constitution_id": ihsan::constitution().id(),
                "ihsan_threshold_baseline": ihsan::constitution().threshold(),
                "ihsan_env": ihsan_env,
                "ihsan_artifact_class": "docs",
                "ihsan_threshold_applied": ihsan_threshold_applied,
                "ihsan_passes_threshold": ihsan_passes_threshold,
                "ihsan_vector": ihsan_vector,
                "ihsan_vector_source": "pat_only_confidence_mapping_v0",
            }),
        })
    }

    async fn handle_slash_command(
        &self,
        command: &SlashCommand,
        request: &EnhancedDualAgenticRequest,
    ) -> anyhow::Result<DualAgenticResponse> {
        let start = Instant::now();
        match command {
            SlashCommand::Reason { method } => {
                info!(?method, "Slash command: Force reasoning method");
                let result = self
                    .reasoning_engine
                    .reason(method, &request.base.task, serde_json::json!({}))
                    .await?;

                let (ihsan_score, ihsan_vector) =
                    self.ihsan_from_scalar_confidence(result.confidence)?;
                let (ihsan_env, ihsan_threshold_applied, ihsan_passes_threshold) =
                    self.enforce_ihsan(ihsan_score, "docs")?;

                Ok(DualAgenticResponse {
                    pat_contributions: vec![result.conclusion],
                    sat_contributions: vec![],
                    synergy_score: result.confidence,
                    ihsan_score,
                    latency: start.elapsed(),
                    meta: serde_json::json!({
                        "slash_command": "reason",
                        "method": format!("{:?}", method),
                        "steps": result.steps,
                        "sat_absent": true,
                        "adapter_modes": AdapterModes::current(),
                        "ihsan_constitution_id": ihsan::constitution().id(),
                        "ihsan_threshold_baseline": ihsan::constitution().threshold(),
                        "ihsan_env": ihsan_env,
                        "ihsan_artifact_class": "docs",
                        "ihsan_threshold_applied": ihsan_threshold_applied,
                        "ihsan_passes_threshold": ihsan_passes_threshold,
                        "ihsan_vector": ihsan_vector,
                        "ihsan_vector_source": "reasoning_confidence_v0",
                    }),
                })
            }

            SlashCommand::Tools { filter } => {
                info!(filter, "Slash command: List tools");
                let mcp = self.mcp_client.read().await;
                let tools = mcp.filter_tools(filter);

                let (ihsan_score, ihsan_vector) = self.ihsan_from_scalar_confidence(1.0)?;
                let (ihsan_env, ihsan_threshold_applied, ihsan_passes_threshold) =
                    self.enforce_ihsan(ihsan_score, "docs")?;

                Ok(DualAgenticResponse {
                    pat_contributions: tools
                        .iter()
                        .map(|t| format!("{}: {}", t.name, t.description))
                        .collect(),
                    sat_contributions: vec![],
                    synergy_score: 1.0,
                    ihsan_score,
                    latency: start.elapsed(),
                    meta: serde_json::json!({
                        "slash_command": "tools",
                        "filter": filter,
                        "count": tools.len(),
                        "sat_absent": true,
                        "adapter_modes": AdapterModes::current(),
                        "ihsan_constitution_id": ihsan::constitution().id(),
                        "ihsan_threshold_baseline": ihsan::constitution().threshold(),
                        "ihsan_env": ihsan_env,
                        "ihsan_artifact_class": "docs",
                        "ihsan_threshold_applied": ihsan_threshold_applied,
                        "ihsan_passes_threshold": ihsan_passes_threshold,
                        "ihsan_vector": ihsan_vector,
                        "ihsan_vector_source": "deterministic_tools_listing_v0",
                    }),
                })
            }

            SlashCommand::Spawn { role, task } => {
                info!(role, task, "Slash command: Spawn sub-agent");
                let mut count = self.sub_agent_count.write().await;
                *count += 1;

                let (ihsan_score, ihsan_vector) = self.ihsan_from_scalar_confidence(0.5)?;
                let (ihsan_env, ihsan_threshold_applied, ihsan_passes_threshold) =
                    self.enforce_ihsan(ihsan_score, "docs")?;

                Ok(DualAgenticResponse {
                    pat_contributions: vec![format!(
                        "Spawned sub-agent '{}' for task: {}",
                        role, task
                    )],
                    sat_contributions: vec![],
                    synergy_score: 0.95,
                    ihsan_score,
                    latency: start.elapsed(),
                    meta: serde_json::json!({
                        "slash_command": "spawn",
                        "sub_agent_role": role,
                        "total_sub_agents": *count,
                        "sat_absent": true,
                        "adapter_modes": AdapterModes::current(),
                        "ihsan_constitution_id": ihsan::constitution().id(),
                        "ihsan_threshold_baseline": ihsan::constitution().threshold(),
                        "ihsan_env": ihsan_env,
                        "ihsan_artifact_class": "docs",
                        "ihsan_threshold_applied": ihsan_threshold_applied,
                        "ihsan_passes_threshold": ihsan_passes_threshold,
                        "ihsan_vector": ihsan_vector,
                        "ihsan_vector_source": "simulated_slash_command_v0",
                    }),
                })
            }

            SlashCommand::Delegate { agent, task } => {
                info!(agent, task, "Slash command: Delegate to agent");
                let a2a = self.a2a_server.read().await;
                let result = a2a.delegate(agent, task.clone()).await
                    .map_err(|e| anyhow::anyhow!("Delegation failed: {}", e))?;

                let (ihsan_score, ihsan_vector) = self.ihsan_from_scalar_confidence(0.5)?;
                let (ihsan_env, ihsan_threshold_applied, ihsan_passes_threshold) =
                    self.enforce_ihsan(ihsan_score, "docs")?;

                Ok(DualAgenticResponse {
                    pat_contributions: vec![format!(
                        "Delegated to {}: {}",
                        agent,
                        result.result
                    )],
                    sat_contributions: vec![],
                    synergy_score: 0.93,
                    ihsan_score,
                    latency: start.elapsed(),
                    meta: serde_json::json!({
                        "slash_command": "delegate",
                        "agent": agent,
                        "result": result.result,
                        "execution_time_ms": result.execution_time_ms,
                        "delegation_depth": result.delegation_depth,
                        "sat_absent": true,
                        "adapter_modes": AdapterModes::current(),
                        "ihsan_constitution_id": ihsan::constitution().id(),
                        "ihsan_threshold_baseline": ihsan::constitution().threshold(),
                        "ihsan_env": ihsan_env,
                        "ihsan_artifact_class": "docs",
                        "ihsan_threshold_applied": ihsan_threshold_applied,
                        "ihsan_passes_threshold": ihsan_passes_threshold,
                        "ihsan_vector": ihsan_vector,
                        "ihsan_vector_source": "simulated_slash_command_v0",
                    }),
                })
            }

            _ => {
                // Other slash commands...
                let (ihsan_score, ihsan_vector) = self.ihsan_from_scalar_confidence(0.5)?;
                let (ihsan_env, ihsan_threshold_applied, ihsan_passes_threshold) =
                    self.enforce_ihsan(ihsan_score, "docs")?;

                Ok(DualAgenticResponse {
                    pat_contributions: vec![format!("Slash command executed: {:?}", command)],
                    sat_contributions: vec![],
                    synergy_score: 0.90,
                    ihsan_score,
                    latency: start.elapsed(),
                    meta: serde_json::json!({
                        "slash_command": format!("{:?}", command),
                        "sat_absent": true,
                        "adapter_modes": AdapterModes::current(),
                        "ihsan_constitution_id": ihsan::constitution().id(),
                        "ihsan_threshold_baseline": ihsan::constitution().threshold(),
                        "ihsan_env": ihsan_env,
                        "ihsan_artifact_class": "docs",
                        "ihsan_threshold_applied": ihsan_threshold_applied,
                        "ihsan_passes_threshold": ihsan_passes_threshold,
                        "ihsan_vector": ihsan_vector,
                        "ihsan_vector_source": "simulated_slash_command_v0",
                    }),
                })
            }
        }
    }

    fn enforce_ihsan(
        &self,
        ihsan_score: f64,
        artifact_class: &'static str,
    ) -> anyhow::Result<(String, f64, bool)> {
        let env = ihsan::current_env();
        let threshold = ihsan::constitution().threshold_for(&env, artifact_class);
        let passes = ihsan_score >= threshold;
        if !passes && ihsan::should_enforce() {
            anyhow::bail!(
                "Ihsan gate failed (env={env} artifact_class={artifact} score={score:.4} threshold={threshold:.4}); escalate via FATE",
                env = env,
                artifact = artifact_class,
                score = ihsan_score,
                threshold = threshold,
            );
        }
        Ok((env, threshold, passes))
    }

    fn ihsan_from_scalar_confidence(
        &self,
        confidence: f64,
    ) -> anyhow::Result<(f64, BTreeMap<String, f64>)> {
        fn clamp01(value: f64) -> f64 {
            value.clamp(0.0, 1.0)
        }

        let mut scores = BTreeMap::new();
        scores.insert("correctness".to_string(), clamp01(confidence));
        scores.insert("safety".to_string(), 0.0);
        scores.insert("user_benefit".to_string(), clamp01(confidence));
        scores.insert("efficiency".to_string(), 0.0);
        scores.insert("auditability".to_string(), 0.0);
        scores.insert("anti_centralization".to_string(), 0.0);
        scores.insert("robustness".to_string(), clamp01(confidence));
        scores.insert("adl_fairness".to_string(), 0.0);

        let score = ihsan::score(&scores)?;
        Ok((score, scores))
    }

    fn calculate_ihsan_pat_only(
        &self,
        pat_results: &[AgentResult],
    ) -> anyhow::Result<(f64, BTreeMap<String, f64>)> {
        fn clamp01(value: f64) -> f64 {
            value.clamp(0.0, 1.0)
        }

        fn find(results: &[AgentResult], name: &str) -> Option<f64> {
            results
                .iter()
                .find(|r| r.agent_name == name)
                .map(|r| r.confidence)
        }

        let pat_avg = avg_confidence(pat_results);

        let mut scores = BTreeMap::new();
        scores.insert(
            "correctness".to_string(),
            clamp01(find(pat_results, "quality_guardian").unwrap_or(pat_avg)),
        );
        scores.insert("safety".to_string(), 0.0);
        scores.insert(
            "user_benefit".to_string(),
            clamp01(find(pat_results, "user_advocate").unwrap_or(pat_avg)),
        );
        scores.insert("efficiency".to_string(), 0.0);
        scores.insert("auditability".to_string(), 0.0);
        scores.insert("anti_centralization".to_string(), 0.0);
        scores.insert(
            "robustness".to_string(),
            clamp01(calculate_consistency(pat_results)),
        );
        scores.insert("adl_fairness".to_string(), 0.0);

        let score = ihsan::score(&scores)?;
        Ok((score, scores))
    }
}

fn avg_confidence(results: &[AgentResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    results.iter().map(|r| r.confidence).sum::<f64>() / results.len() as f64
}

fn calculate_consistency(results: &[AgentResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }

    let mean = avg_confidence(results);

    let variance = results
        .iter()
        .map(|r| (r.confidence - mean).powi(2))
        .sum::<f64>()
        / results.len() as f64;

    // High consistency = low variance
    1.0 - variance.sqrt()
}
