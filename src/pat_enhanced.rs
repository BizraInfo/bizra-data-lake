// src/pat_enhanced.rs - PAT with all capabilities

use crate::{
    a2a::A2AServer,
    errors::PolicyError,
    ihsan,
    mcp::MCPClient,
    pat::PATOrchestrator,
    sape::{self, ProbeDimension, ProbeResult},
    reasoning::MultiMethodReasoning,
    types::*,
};
use std::{
    collections::{BTreeMap, HashSet},
    sync::Arc,
    time::Instant,
};
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

        // 🌉 Register BIZRA Data Lake Bridge (Hypergraph RAG)
        mcp.register_server(
            "bizra_data_lake".to_string(),
            "http://host.docker.internal:8000".to_string(),
            crate::mcp::MCPTransport::HttpSse,
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

        let tool_allowlist = Self::normalize_tool_allowlist(&request.mcp_tools_whitelist);

        // Execute with MCP tools if needed
        let mcp = self.mcp_client.read().await;
        let available_tools = Self::apply_tool_allowlist(mcp.list_tools(), &tool_allowlist);
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
        let (ihsan_score, ihsan_vector, sape_flags, sape_probe_count) =
            self.calculate_ihsan_pat_only(&base_result, &request)?;
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
                "mcp_allowlist_provided": request.mcp_tools_whitelist.is_some(),
                "sub_agents_spawned": *self.sub_agent_count.read().await,
                "sat_absent": true,
                "synergy_score_source": "pat_avg_confidence_v0",
                "execution_mode": "PRODUCTION",
                "ihsan_constitution_id": ihsan::constitution().id(),
                "ihsan_threshold_baseline": ihsan::constitution().threshold(),
                "ihsan_env": ihsan_env,
                "ihsan_artifact_class": "docs",
                "ihsan_threshold_applied": ihsan_threshold_applied,
                "ihsan_passes_threshold": ihsan_passes_threshold,
                "ihsan_vector": ihsan_vector,
                "ihsan_vector_source": "sape_probes_v1",
                "sape_probe_flags": sape_flags,
                "sape_probe_count": sape_probe_count,
            }),
        })
    }

    async fn handle_slash_command(
        &self,
        command: &SlashCommand,
        request: &EnhancedDualAgenticRequest,
    ) -> anyhow::Result<DualAgenticResponse> {
        let start = Instant::now();
        let tool_allowlist = Self::normalize_tool_allowlist(&request.mcp_tools_whitelist);
        match command {
            SlashCommand::Reason { method } => {
                info!(?method, "Slash command: Force reasoning method");
                let result = self
                    .reasoning_engine
                    .reason(method, &request.base.task, serde_json::json!({}))
                    .await?;

                let reasoning_content = format!(
                    "Conclusion: {}\nSteps:\n{}",
                    result.conclusion,
                    result.steps.join("\n")
                );
                let (ihsan_score, ihsan_vector, sape_flags, sape_probe_count) =
                    self.ihsan_from_content(&reasoning_content)?;
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
                        "execution_mode": "PRODUCTION",
                        "ihsan_constitution_id": ihsan::constitution().id(),
                        "ihsan_threshold_baseline": ihsan::constitution().threshold(),
                        "ihsan_env": ihsan_env,
                        "ihsan_artifact_class": "docs",
                        "ihsan_threshold_applied": ihsan_threshold_applied,
                        "ihsan_passes_threshold": ihsan_passes_threshold,
                        "ihsan_vector": ihsan_vector,
                        "ihsan_vector_source": "sape_probes_v1",
                        "sape_probe_flags": sape_flags,
                        "sape_probe_count": sape_probe_count,
                    }),
                })
            }

            SlashCommand::Tools { filter } => {
                info!(filter, "Slash command: List tools");
                Self::ensure_tools_allowed(&tool_allowlist)?;
                let mcp = self.mcp_client.read().await;
                let tools = Self::apply_tool_allowlist(mcp.filter_tools(filter), &tool_allowlist);

                let pat_contributions: Vec<String> = tools
                    .iter()
                    .map(|t| format!("{}: {}", t.name, t.description))
                    .collect();

                let tools_content = pat_contributions.join("\n");
                let (ihsan_score, ihsan_vector, sape_flags, sape_probe_count) =
                    self.ihsan_from_content(&tools_content)?;
                let (ihsan_env, ihsan_threshold_applied, ihsan_passes_threshold) =
                    self.enforce_ihsan(ihsan_score, "docs")?;

                Ok(DualAgenticResponse {
                    pat_contributions,
                    sat_contributions: vec![],
                    synergy_score: 1.0,
                    ihsan_score,
                    latency: start.elapsed(),
                    meta: serde_json::json!({
                        "slash_command": "tools",
                        "filter": filter,
                        "count": tools.len(),
                        "mcp_allowlist_provided": request.mcp_tools_whitelist.is_some(),
                        "sat_absent": true,
                        "execution_mode": "PRODUCTION",
                        "ihsan_constitution_id": ihsan::constitution().id(),
                        "ihsan_threshold_baseline": ihsan::constitution().threshold(),
                        "ihsan_env": ihsan_env,
                        "ihsan_artifact_class": "docs",
                        "ihsan_threshold_applied": ihsan_threshold_applied,
                        "ihsan_passes_threshold": ihsan_passes_threshold,
                        "ihsan_vector": ihsan_vector,
                        "ihsan_vector_source": "sape_probes_v1",
                        "sape_probe_flags": sape_flags,
                        "sape_probe_count": sape_probe_count,
                    }),
                })
            }

            SlashCommand::Spawn { role, task } => {
                info!(role, task, "Slash command: Spawn sub-agent");
                let mut count = self.sub_agent_count.write().await;
                *count += 1;

                let spawn_content =
                    format!("Spawned sub-agent '{}' for task: {}", role, task);
                let (ihsan_score, ihsan_vector, sape_flags, sape_probe_count) =
                    self.ihsan_from_content(&spawn_content)?;
                let (ihsan_env, ihsan_threshold_applied, ihsan_passes_threshold) =
                    self.enforce_ihsan(ihsan_score, "docs")?;

                Ok(DualAgenticResponse {
                    pat_contributions: vec![spawn_content],
                    sat_contributions: vec![],
                    synergy_score: 0.95,
                    ihsan_score,
                    latency: start.elapsed(),
                    meta: serde_json::json!({
                        "slash_command": "spawn",
                        "sub_agent_role": role,
                        "total_sub_agents": *count,
                        "sat_absent": true,
                        "execution_mode": "PRODUCTION",
                        "ihsan_constitution_id": ihsan::constitution().id(),
                        "ihsan_threshold_baseline": ihsan::constitution().threshold(),
                        "ihsan_env": ihsan_env,
                        "ihsan_artifact_class": "docs",
                        "ihsan_threshold_applied": ihsan_threshold_applied,
                        "ihsan_passes_threshold": ihsan_passes_threshold,
                        "ihsan_vector": ihsan_vector,
                        "ihsan_vector_source": "sape_probes_v1",
                        "sape_probe_flags": sape_flags,
                        "sape_probe_count": sape_probe_count,
                    }),
                })
            }

            SlashCommand::Delegate { agent, task } => {
                info!(agent, task, "Slash command: Delegate to agent");
                let a2a = self.a2a_server.read().await;
                let result = a2a.delegate(agent, task.clone()).await
                    .map_err(|e| anyhow::anyhow!("Delegation failed: {}", e))?;

                let delegate_content = format!(
                    "Delegated to {} for task '{}': {}",
                    agent, task, result.result
                );
                let (ihsan_score, ihsan_vector, sape_flags, sape_probe_count) =
                    self.ihsan_from_content(&delegate_content)?;
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
                        "execution_mode": "PRODUCTION",
                        "ihsan_constitution_id": ihsan::constitution().id(),
                        "ihsan_threshold_baseline": ihsan::constitution().threshold(),
                        "ihsan_env": ihsan_env,
                        "ihsan_artifact_class": "docs",
                        "ihsan_threshold_applied": ihsan_threshold_applied,
                        "ihsan_passes_threshold": ihsan_passes_threshold,
                        "ihsan_vector": ihsan_vector,
                        "ihsan_vector_source": "sape_probes_v1",
                        "sape_probe_flags": sape_flags,
                        "sape_probe_count": sape_probe_count,
                    }),
                })
            }

            _ => {
                // Other slash commands...
                let default_content = format!("Slash command executed: {:?}", command);
                let (ihsan_score, ihsan_vector, sape_flags, sape_probe_count) =
                    self.ihsan_from_content(&default_content)?;
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
                        "execution_mode": "PRODUCTION",
                        "ihsan_constitution_id": ihsan::constitution().id(),
                        "ihsan_threshold_baseline": ihsan::constitution().threshold(),
                        "ihsan_env": ihsan_env,
                        "ihsan_artifact_class": "docs",
                        "ihsan_threshold_applied": ihsan_threshold_applied,
                        "ihsan_passes_threshold": ihsan_passes_threshold,
                        "ihsan_vector": ihsan_vector,
                        "ihsan_vector_source": "sape_probes_v1",
                        "sape_probe_flags": sape_flags,
                        "sape_probe_count": sape_probe_count,
                    }),
                })
            }
        }
    }

    fn normalize_tool_allowlist(
        raw: &Option<Vec<String>>,
    ) -> Option<HashSet<String>> {
        let Some(list) = raw else {
            return None;
        };

        let mut allowlist = HashSet::new();
        for name in list {
            let trimmed = name.trim();
            if !trimmed.is_empty() {
                allowlist.insert(trimmed.to_string());
            }
        }
        Some(allowlist)
    }

    fn apply_tool_allowlist<'a>(
        tools: Vec<&'a crate::mcp::ToolDefinition>,
        allowlist: &Option<HashSet<String>>,
    ) -> Vec<&'a crate::mcp::ToolDefinition> {
        match allowlist {
            Some(set) => tools
                .into_iter()
                .filter(|tool| set.contains(&tool.name))
                .collect(),
            None => tools,
        }
    }

    fn ensure_tools_allowed(allowlist: &Option<HashSet<String>>) -> Result<(), PolicyError> {
        if let Some(set) = allowlist {
            if set.is_empty() {
                return Err(PolicyError::McpToolsBlocked {
                    message: "MCP allowlist provided but empty".to_string(),
                });
            }
        }
        Ok(())
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
            return Err(PolicyError::IhsanGateFailed {
                env,
                score: ihsan_score,
                threshold,
            }
            .into());
        }
        Ok((env, threshold, passes))
    }

    fn ihsan_from_content(
        &self,
        content: &str,
    ) -> anyhow::Result<(f64, BTreeMap<String, f64>, Vec<String>, usize)> {
        let sape_engine = sape::get_sape();
        let mut engine = sape_engine
            .lock()
            .map_err(|_| anyhow::anyhow!("SAPE engine lock poisoned"))?;

        let probe_results = engine.execute_probes(content);
        let ihsan_vector = Self::map_probes_to_ihsan_vector(&probe_results)?;
        let ihsan_score = ihsan::score(&ihsan_vector)?;
        let flags = probe_results
            .iter()
            .flat_map(|r| r.flags.clone())
            .collect::<Vec<String>>();

        Ok((ihsan_score, ihsan_vector, flags, probe_results.len()))
    }

    fn map_probes_to_ihsan_vector(
        probe_results: &[ProbeResult],
    ) -> anyhow::Result<BTreeMap<String, f64>> {
        fn find(
            results: &[ProbeResult],
            dim: ProbeDimension,
        ) -> Option<&ProbeResult> {
            results.iter().find(|r| r.dimension == dim)
        }

        fn weighted_mean(
            results: &[ProbeResult],
            dims: &[ProbeDimension],
        ) -> anyhow::Result<f64> {
            let mut total = 0.0;
            let mut weight_sum = 0.0;

            for dim in dims {
                let result = find(results, *dim).ok_or_else(|| {
                    anyhow::anyhow!("SAPE probe result missing for dimension {:?}", dim)
                })?;
                let weight = dim.weight();
                total += result.score * weight;
                weight_sum += weight;
            }

            if weight_sum == 0.0 {
                anyhow::bail!("SAPE probe weights summed to zero");
            }

            Ok(total / weight_sum)
        }

        let mut scores = BTreeMap::new();
        scores.insert(
            "correctness".to_string(),
            weighted_mean(probe_results, &[ProbeDimension::Correctness])?,
        );
        scores.insert(
            "safety".to_string(),
            weighted_mean(
                probe_results,
                &[ProbeDimension::ThreatScan, ProbeDimension::Safety],
            )?,
        );
        scores.insert(
            "user_benefit".to_string(),
            weighted_mean(probe_results, &[ProbeDimension::UserBenefit])?,
        );
        scores.insert(
            "efficiency".to_string(),
            weighted_mean(probe_results, &[ProbeDimension::Relevance])?,
        );
        scores.insert(
            "auditability".to_string(),
            weighted_mean(
                probe_results,
                &[ProbeDimension::ComplianceCheck],
            )?,
        );
        scores.insert(
            "anti_centralization".to_string(),
            weighted_mean(probe_results, &[ProbeDimension::Fluency])?,
        );
        scores.insert(
            "robustness".to_string(),
            weighted_mean(probe_results, &[ProbeDimension::Groundedness])?,
        );
        scores.insert(
            "adl_fairness".to_string(),
            weighted_mean(probe_results, &[ProbeDimension::BiasProbe])?,
        );

        Ok(scores)
    }

    fn calculate_ihsan_pat_only(
        &self,
        pat_results: &[AgentResult],
        request: &EnhancedDualAgenticRequest,
    ) -> anyhow::Result<(f64, BTreeMap<String, f64>, Vec<String>, usize)> {
        let mut content_parts = Vec::new();
        content_parts.push(format!("User task: {}", request.base.task));

        if !request.base.requirements.is_empty() {
            content_parts.push(format!(
                "Requirements: {}",
                request.base.requirements.join("; ")
            ));
        }

        if !request.base.context.is_empty() {
            let ctx = request
                .base
                .context
                .iter()
                .map(|(k, v)| format!("{}: {}", k, v))
                .collect::<Vec<_>>()
                .join(", ");
            content_parts.push(format!("Context: {ctx}"));
        }

        for result in pat_results {
            content_parts.push(format!(
                "{}: {}",
                result.agent_name,
                result.contribution
            ));
        }

        let content = content_parts.join("\n");
        self.ihsan_from_content(&content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::ToolDefinition;
    use crate::sape::{ProbeDimension, ProbeResult};

    #[test]
    fn normalize_tool_allowlist_trims_and_dedupes() {
        let raw = Some(vec![
            "calculator".to_string(),
            "  calculator  ".to_string(),
            "".to_string(),
        ]);
        let allowlist = EnhancedPATOrchestrator::normalize_tool_allowlist(&raw)
            .expect("expected allowlist");
        assert_eq!(allowlist.len(), 1);
        assert!(allowlist.contains("calculator"));
    }

    #[test]
    fn apply_tool_allowlist_filters_tools() {
        let tools = vec![
            ToolDefinition {
                name: "calculator".to_string(),
                description: "math".to_string(),
                parameters: vec![],
                server: "local".to_string(),
            },
            ToolDefinition {
                name: "filesystem_read".to_string(),
                description: "fs".to_string(),
                parameters: vec![],
                server: "local".to_string(),
            },
        ];
        let refs: Vec<&ToolDefinition> = tools.iter().collect();

        let mut allow = HashSet::new();
        allow.insert("calculator".to_string());
        let filtered =
            EnhancedPATOrchestrator::apply_tool_allowlist(refs, &Some(allow));

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].name, "calculator");
    }

    #[test]
    fn ensure_tools_allowed_blocks_empty_allowlist() {
        let allowlist = Some(HashSet::<String>::new());
        let err = EnhancedPATOrchestrator::ensure_tools_allowed(&allowlist)
            .expect_err("expected empty allowlist error");
        assert!(err.to_string().contains("MCP allowlist provided but empty"));
    }

    #[test]
    fn normalize_tool_allowlist_with_none_returns_none() {
        let result = EnhancedPATOrchestrator::normalize_tool_allowlist(&None);
        assert!(result.is_none(), "None input should return None");
    }

    #[test]
    fn normalize_tool_allowlist_with_only_empty_strings_returns_empty_set() {
        let raw = Some(vec![
            "".to_string(),
            "   ".to_string(),
            "\t".to_string(),
        ]);
        let allowlist = EnhancedPATOrchestrator::normalize_tool_allowlist(&raw)
            .expect("expected Some with empty set");
        assert!(
            allowlist.is_empty(),
            "Allowlist with only empty/whitespace strings should be empty set"
        );
    }

    #[test]
    fn apply_tool_allowlist_with_none_returns_all_tools() {
        let tools = vec![
            ToolDefinition {
                name: "calculator".to_string(),
                description: "math".to_string(),
                parameters: vec![],
                server: "local".to_string(),
            },
            ToolDefinition {
                name: "filesystem_read".to_string(),
                description: "fs".to_string(),
                parameters: vec![],
                server: "local".to_string(),
            },
        ];
        let refs: Vec<&ToolDefinition> = tools.iter().collect();

        let filtered = EnhancedPATOrchestrator::apply_tool_allowlist(refs.clone(), &None);

        assert_eq!(
            filtered.len(),
            2,
            "None allowlist should return all tools unchanged"
        );
    }

    #[test]
    fn map_probes_to_ihsan_vector_respects_probe_weights() {
        let probes: Vec<ProbeResult> = ProbeDimension::all()
            .iter()
            .map(|dim| ProbeResult {
                dimension: *dim,
                // Safety-related probes get higher score to verify averaging logic
                score: if matches!(dim, ProbeDimension::ThreatScan | ProbeDimension::Safety) {
                    0.8
                } else {
                    0.5
                },
                confidence: 1.0,
                flags: vec![],
                latency_ms: 1.0,
            })
            .collect();

        let vector =
            EnhancedPATOrchestrator::map_probes_to_ihsan_vector(&probes).expect("ihsan vector");

        assert!(
            (vector["correctness"] - 0.5).abs() < 1e-9,
            "Correctness should mirror correctness probe score"
        );
        assert!(
            (vector["safety"] - 0.8).abs() < 1e-9,
            "Safety should average threat_scan and safety probes"
        );
        assert!(
            (vector["adl_fairness"] - 0.5).abs() < 1e-9,
            "Bias probe should map to adl_fairness"
        );
    }

    #[test]
    fn apply_tool_allowlist_with_nonexistent_names_returns_empty() {
        let tools = vec![
            ToolDefinition {
                name: "calculator".to_string(),
                description: "math".to_string(),
                parameters: vec![],
                server: "local".to_string(),
            },
            ToolDefinition {
                name: "filesystem_read".to_string(),
                description: "fs".to_string(),
                parameters: vec![],
                server: "local".to_string(),
            },
        ];
        let refs: Vec<&ToolDefinition> = tools.iter().collect();

        let mut allow = HashSet::new();
        allow.insert("nonexistent_tool".to_string());
        allow.insert("another_missing_tool".to_string());
        let filtered = EnhancedPATOrchestrator::apply_tool_allowlist(refs, &Some(allow));

        assert!(
            filtered.is_empty(),
            "Allowlist with names not present in tools should return empty vec"
        );
    }

    #[test]
    fn ensure_tools_allowed_with_none_returns_ok() {
        let result = EnhancedPATOrchestrator::ensure_tools_allowed(&None);
        assert!(
            result.is_ok(),
            "None allowlist should return Ok, got: {:?}",
            result
        );
    }
}

fn avg_confidence(results: &[AgentResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    results.iter().map(|r| r.confidence).sum::<f64>() / results.len() as f64
}
