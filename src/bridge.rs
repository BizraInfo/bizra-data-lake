// src/bridge.rs - PAT-SAT Bridge Coordinator

use crate::{
    ihsan,
    pat::PATOrchestrator,
    sat::SATOrchestrator,
    types::{AdapterModes, AgentResult, DualAgenticRequest, DualAgenticResponse},
};
use std::{collections::BTreeMap, time::Instant};
use tracing::{info, instrument};

/// Bridge coordinator between PAT and SAT
pub struct BridgeCoordinator {
    pat: PATOrchestrator,
    sat: SATOrchestrator,
}

impl BridgeCoordinator {
    pub async fn new() -> anyhow::Result<Self> {
        info!("dYO% Initializing PAT-SAT Bridge Coordinator");

        let pat = PATOrchestrator::new().await?;
        let sat = SATOrchestrator::new().await?;

        Ok(Self { pat, sat })
    }

    /// Execute full dual-agentic workflow
    #[instrument(skip(self))]
    pub async fn execute(
        &self,
        request: DualAgenticRequest,
    ) -> anyhow::Result<DualAgenticResponse> {
        let start = Instant::now();

        info!("dYs? Starting dual-agentic execution");

        // Step 1: SAT validates the request
        let validation = self.sat.validate_request(&request).await?;

        if !validation.consensus_reached {
            return Err(anyhow::anyhow!(
                "SAT consensus not reached for request validation"
            ));
        }

        info!(
            validation_time_ms = validation.validation_time.as_millis(),
            "SAT validation passed"
        );

        // Step 2: PAT executes the task
        let pat_results = self.pat.execute_parallel(vec![], request.clone()).await?;

        info!(pat_agents = pat_results.len(), "PAT execution completed");

        // Step 3: SAT evaluates PAT results
        let sat_evaluations = self.sat.evaluate_results(&pat_results).await?;

        info!(
            sat_evaluations = sat_evaluations.len(),
            "SAT evaluation completed"
        );

        // Step 4: Calculate synergy and Ihsan scores
        let synergy_score = self.calculate_synergy(&pat_results, &sat_evaluations);
        let (ihsan_score, ihsan_vector) = self.calculate_ihsan(&pat_results, &sat_evaluations)?;

        let ihsan_env = ihsan::current_env();
        let ihsan_artifact_class = "docs";
        let ihsan_threshold_applied =
            ihsan::constitution().threshold_for(&ihsan_env, ihsan_artifact_class);
        let ihsan_passes_threshold = ihsan_score >= ihsan_threshold_applied;

        if !ihsan_passes_threshold && ihsan::should_enforce() {
            return Err(anyhow::anyhow!(
                "Ihsan gate failed (env={env} artifact_class={artifact} score={score:.4} threshold={threshold:.4}); escalate via FATE",
                env = ihsan_env,
                artifact = ihsan_artifact_class,
                score = ihsan_score,
                threshold = ihsan_threshold_applied,
            ));
        }

        let total_latency = start.elapsed();

        info!(
            synergy = synergy_score,
            ihsan = ihsan_score,
            latency_ms = total_latency.as_millis(),
            "Dual-agentic execution completed"
        );

        Ok(DualAgenticResponse {
            pat_contributions: pat_results.iter().map(|r| r.contribution.clone()).collect(),
            sat_contributions: sat_evaluations
                .iter()
                .map(|r| r.contribution.clone())
                .collect(),
            synergy_score,
            ihsan_score,
            latency: total_latency,
            meta: serde_json::json!({
                "pat_agents": self.pat.get_agent_count(),
                "sat_agents": self.sat.get_agent_count(),
                "adapter_modes": AdapterModes::current(),
                "validation_time_ms": validation.validation_time.as_millis(),
                "ihsan_constitution_id": ihsan::constitution().id(),
                "ihsan_threshold_baseline": ihsan::constitution().threshold(),
                "ihsan_env": ihsan_env,
                "ihsan_artifact_class": ihsan_artifact_class,
                "ihsan_threshold_applied": ihsan_threshold_applied,
                "ihsan_passes_threshold": ihsan_passes_threshold,
                "ihsan_vector": ihsan_vector,
                "ihsan_vector_source": "simulated_confidence_mapping_v0",
            }),
        })
    }

    /// Calculate synergy between PAT and SAT
    fn calculate_synergy(&self, pat_results: &[AgentResult], sat_results: &[AgentResult]) -> f64 {
        let pat_avg =
            pat_results.iter().map(|r| r.confidence).sum::<f64>() / pat_results.len() as f64;

        let sat_avg =
            sat_results.iter().map(|r| r.confidence).sum::<f64>() / sat_results.len() as f64;

        // Harmonic mean for synergy
        2.0 * pat_avg * sat_avg / (pat_avg + sat_avg)
    }

    fn calculate_ihsan(
        &self,
        pat_results: &[AgentResult],
        sat_results: &[AgentResult],
    ) -> anyhow::Result<(f64, BTreeMap<String, f64>)> {
        fn clamp01(value: f64) -> f64 {
            value.clamp(0.0, 1.0)
        }

        fn avg(results: &[AgentResult]) -> f64 {
            if results.is_empty() {
                return 0.0;
            }
            results.iter().map(|r| r.confidence).sum::<f64>() / results.len() as f64
        }

        fn find(results: &[AgentResult], name: &str) -> Option<f64> {
            results
                .iter()
                .find(|r| r.agent_name == name)
                .map(|r| r.confidence)
        }

        let pat_avg = avg(pat_results);
        let sat_avg = avg(sat_results);

        let mut scores = BTreeMap::new();
        scores.insert(
            "correctness".to_string(),
            clamp01(find(pat_results, "quality_guardian").unwrap_or(pat_avg)),
        );
        scores.insert(
            "safety".to_string(),
            clamp01(find(sat_results, "security_guardian").unwrap_or(sat_avg)),
        );
        scores.insert(
            "user_benefit".to_string(),
            clamp01(find(pat_results, "user_advocate").unwrap_or(pat_avg)),
        );
        scores.insert(
            "efficiency".to_string(),
            clamp01(find(sat_results, "performance_monitor").unwrap_or(sat_avg)),
        );
        scores.insert(
            "auditability".to_string(),
            clamp01(find(sat_results, "consistency_checker").unwrap_or(sat_avg)),
        );
        scores.insert(
            "anti_centralization".to_string(),
            clamp01(find(sat_results, "resource_optimizer").unwrap_or(sat_avg)),
        );
        scores.insert(
            "robustness".to_string(),
            clamp01(self.calculate_consistency(pat_results)),
        );
        scores.insert(
            "adl_fairness".to_string(),
            clamp01(find(sat_results, "ethics_validator").unwrap_or(sat_avg)),
        );

        let score = ihsan::score(&scores)?;
        Ok((score, scores))
    }

    fn calculate_consistency(&self, results: &[AgentResult]) -> f64 {
        if results.is_empty() {
            return 0.0;
        }

        let mean = results.iter().map(|r| r.confidence).sum::<f64>() / results.len() as f64;

        let variance = results
            .iter()
            .map(|r| (r.confidence - mean).powi(2))
            .sum::<f64>()
            / results.len() as f64;

        // High consistency = low variance
        1.0 - variance.sqrt()
    }
}
