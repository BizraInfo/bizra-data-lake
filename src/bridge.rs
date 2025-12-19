// src/bridge.rs - PAT-SAT Bridge Coordinator

use crate::{
    fate::FATECoordinator,
    ihsan,
    metrics,
    pat::PATOrchestrator,
    receipts::ReceiptEmitter,
    sat::SATOrchestrator,
    types::{AdapterModes, AgentResult, DualAgenticRequest, DualAgenticResponse},
};
use std::{collections::BTreeMap, sync::Mutex, time::Instant};
use tracing::{info, warn, instrument};

/// Bridge coordinator between PAT and SAT
pub struct BridgeCoordinator {
    pat: PATOrchestrator,
    sat: SATOrchestrator,
    fate: Mutex<FATECoordinator>,
    receipts: ReceiptEmitter,
}

impl BridgeCoordinator {
    pub async fn new() -> anyhow::Result<Self> {
        info!("🌉 Initializing PAT-SAT Bridge Coordinator");

        let pat = PATOrchestrator::new().await?;
        let sat = SATOrchestrator::new().await?;
        let fate = Mutex::new(FATECoordinator::new());
        let receipts = ReceiptEmitter::default();

        Ok(Self { pat, sat, fate, receipts })
    }

    /// Execute full dual-agentic workflow with FATE escalation and receipt emission
    #[instrument(skip(self))]
    pub async fn execute(
        &self,
        request: DualAgenticRequest,
    ) -> anyhow::Result<DualAgenticResponse> {
        let start = Instant::now();

        info!("🚀 Starting dual-agentic execution");

        // Step 1: SAT validates the request
        let validation = self.sat.validate_request(&request).await?;
        let sat_validation_time = validation.validation_time;

        // Record SAT validation metrics
        let rejection_codes_str: Vec<String> = validation.rejection_codes.iter()
            .map(|c| c.to_string())
            .collect();
        metrics::record_sat_validation(
            validation.consensus_reached,
            &rejection_codes_str,
            sat_validation_time.as_secs_f64(),
            validation.validations.iter().filter(|v| v.approved).count(),
        );

        if !validation.consensus_reached {
            // FATE escalation for SAT rejection
            let escalation = {
                let mut fate = self.fate.lock().unwrap();
                fate.escalate_rejection(
                    &validation.rejection_codes,
                    &request.task,
                    &request.context,
                )
            };

            // Record FATE escalation metrics
            let fate_pending = self.fate.lock().unwrap().pending_count();
            metrics::record_fate_escalation(&format!("{:?}", escalation.level), fate_pending);

            // Collect rejecting and approving validators
            let rejecting: Vec<String> = validation.validations
                .iter()
                .filter(|v| !v.approved)
                .map(|v| v.agent_name.clone())
                .collect();
            let approving: Vec<String> = validation.validations
                .iter()
                .filter(|v| v.approved)
                .map(|v| v.agent_name.clone())
                .collect();

            // Emit rejection receipt
            let receipt = self.receipts.emit_rejection(
                &request.task,
                &validation.rejection_codes,
                &escalation,
                rejecting,
                approving,
            );

            // Record receipt emission
            metrics::record_receipt_emitted("rejection");

            warn!(
                receipt_id = %receipt.receipt_id,
                escalation_id = %escalation.id,
                escalation_level = ?escalation.level,
                "🚨 Request BLOCKED by SAT - receipt emitted"
            );

            return Err(anyhow::anyhow!(
                "SAT BLOCKED: {} (escalation={}, receipt={})",
                validation.rejection_codes.iter()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>()
                    .join("; "),
                escalation.id,
                receipt.receipt_id,
            ));
        }

        info!(
            validation_time_ms = sat_validation_time.as_millis(),
            "✅ SAT validation passed"
        );

        // Step 2: PAT executes the task
        let pat_start = Instant::now();
        let pat_results = self.pat.execute_parallel(vec![], request.clone()).await?;
        let pat_execution_time = pat_start.elapsed();

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

        // Record Ihsān metrics
        metrics::record_ihsan_result(ihsan_score, ihsan_passes_threshold, &ihsan_env, &ihsan_vector);

        if !ihsan_passes_threshold && ihsan::should_enforce() {
            // FATE escalation for Ihsān failure
            let escalation = {
                let mut fate = self.fate.lock().unwrap();
                fate.escalate_ihsan_failure(
                    &ihsan_env,
                    ihsan_artifact_class,
                    ihsan_score,
                    ihsan_threshold_applied,
                    &request.context,
                )
            };

            // Record FATE escalation for Ihsān failure
            let fate_pending = self.fate.lock().unwrap().pending_count();
            metrics::record_fate_escalation(&format!("{:?}", escalation.level), fate_pending);

            warn!(
                escalation_id = %escalation.id,
                ihsan_score = ihsan_score,
                threshold = ihsan_threshold_applied,
                "⚠️ Ihsān gate failed - escalated via FATE"
            );

            return Err(anyhow::anyhow!(
                "IHSAN GATE FAILED: env={} score={:.4} < threshold={:.4} (escalation={})",
                ihsan_env,
                ihsan_score,
                ihsan_threshold_applied,
                escalation.id,
            ));
        }

        let total_latency = start.elapsed();

        // Record request latency metrics
        metrics::record_request_completion("success", total_latency.as_secs_f64(), synergy_score);

        // Emit execution receipt for successful flow
        let sat_approvers = validation.validations.iter().filter(|v| v.approved).count();
        let _execution_receipt = self.receipts.emit_execution(
            &request.task,
            sat_validation_time.as_millis(),
            pat_execution_time.as_millis(),
            total_latency.as_millis(),
            synergy_score,
            ihsan_score,
            ihsan_threshold_applied,
            pat_results.len(),
            sat_approvers,
        );

        // Record execution receipt emission
        metrics::record_receipt_emitted("execution");

        info!(
            synergy = synergy_score,
            ihsan = ihsan_score,
            latency_ms = total_latency.as_millis(),
            "✅ Dual-agentic execution completed - receipt emitted"
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
                "validation_time_ms": sat_validation_time.as_millis(),
                "pat_execution_time_ms": pat_execution_time.as_millis(),
                "ihsan_constitution_id": ihsan::constitution().id(),
                "ihsan_threshold_baseline": ihsan::constitution().threshold(),
                "ihsan_env": ihsan_env,
                "ihsan_artifact_class": ihsan_artifact_class,
                "ihsan_threshold_applied": ihsan_threshold_applied,
                "ihsan_passes_threshold": ihsan_passes_threshold,
                "ihsan_vector": ihsan_vector,
                "ihsan_vector_source": "simulated_confidence_mapping_v0",
                "fate_pending_escalations": self.fate.lock().unwrap().pending_count(),
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
