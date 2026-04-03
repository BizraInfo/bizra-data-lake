// src/bridge.rs - PAT-SAT Bridge Coordinator
//
// Enhanced with:
// - IdempotentReplayManager for exactly-once semantics
// - EntropyPool integration for cryptographic operations
// - Checkpoint-based crash recovery

use crate::{
    entropy::{self},
    errors::BridgeError,
    fate::FATECoordinator,
    idempotency::{IdempotencyStatus, IdempotentReplayManager},
    ifc::{IntegrityLevel, SecrecyLevel, TaintContext, TaintLabel},
    ihsan, metrics,
    pat::PATOrchestrator,
    receipts::ReceiptEmitter,
    sat::SATOrchestrator,
    types::{AgentResult, DualAgenticRequest, DualAgenticResponse},
};
use std::{collections::BTreeMap, sync::Mutex, time::Instant};
use tracing::{debug, info, instrument, warn};

/// Bridge coordinator between PAT and SAT
pub struct BridgeCoordinator {
    pat: PATOrchestrator,
    sat: SATOrchestrator,
    fate: Mutex<FATECoordinator>,
    receipts: ReceiptEmitter,
    /// Idempotency manager for exactly-once semantics
    idempotency: IdempotentReplayManager,
}

impl BridgeCoordinator {
    pub async fn new() -> anyhow::Result<Self> {
        info!("🌉 Initializing PAT-SAT Bridge Coordinator");

        // Initialize entropy pool first (used by other components)
        let _ = entropy::global_pool();
        info!("🎲 EntropyPool initialized");

        let pat = PATOrchestrator::new().await?;
        let sat = SATOrchestrator::new().await?;
        // Use from_env() for Redis persistence (hard fail if unavailable)
        let fate = Mutex::new(FATECoordinator::from_env().await?);
        let receipts = ReceiptEmitter::from_env("docs/evidence/receipts").await?;
        let idempotency = IdempotentReplayManager::new();

        info!("🔒 IdempotentReplayManager ready (exactly-once semantics enabled)");

        Ok(Self {
            pat,
            sat,
            fate,
            receipts,
            idempotency,
        })
    }

    /// Execute full dual-agentic workflow with FATE escalation and receipt emission
    ///
    /// Features:
    /// - Exactly-once semantics via idempotency checking
    /// - Checkpoint-based crash recovery
    /// - EntropyPool-backed cryptographic operations
    #[instrument(skip(self))]
    pub async fn execute(
        &self,
        request: DualAgenticRequest,
    ) -> anyhow::Result<DualAgenticResponse> {
        let start = Instant::now();
        let request_id = request.context.get("request_id").cloned();

        // Generate idempotency key from request fingerprint
        let idem_key = self.idempotency.fingerprint_structured(&request);

        // Check for duplicate/in-progress requests
        let (idem_status, cached) = self.idempotency.check(&idem_key);

        match idem_status {
            IdempotencyStatus::Duplicate => {
                if let Some(cached_result) = cached {
                    info!(
                        idem_key = %idem_key,
                        "♻️ Returning cached result (exactly-once semantics)"
                    );
                    // Deserialize and return cached response
                    if let Ok(response) =
                        serde_json::from_str::<DualAgenticResponse>(&cached_result.result)
                    {
                        return Ok(response);
                    }
                }
                // Fall through if deserialization fails
            }
            IdempotencyStatus::InProgress => {
                info!(
                    idem_key = %idem_key,
                    "⏳ Request already in progress"
                );
                return Err(BridgeError::RequestInProgress { key: idem_key }.into());
            }
            IdempotencyStatus::Expired | IdempotencyStatus::New => {
                // Continue with normal processing
            }
        }

        // Reserve slot for this request (mark as in-progress)
        let checkpoint_id = self
            .idempotency
            .reserve(&idem_key)
            .map_err(|e| BridgeError::IdempotencyError { message: e })?;

        debug!(
            idem_key = %idem_key,
            checkpoint_id = %checkpoint_id,
            "📍 Checkpoint created for request"
        );

        info!("🚀 Starting dual-agentic execution");

        // IFC: Tag user input as Untrusted/Internal
        let mut taint_ctx = TaintContext::new("user");
        taint_ctx.taint(
            "request.task",
            TaintLabel::new(SecrecyLevel::Internal, IntegrityLevel::Untrusted, "user".into()),
        );

        // Step 1: SAT validates the request
        let validation = self.sat.validate_request(&request).await?;
        let sat_validation_time = validation.validation_time;

        // Record SAT validation metrics
        let rejection_codes_str: Vec<String> = validation
            .rejection_codes
            .iter()
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
            // Use scoped lock to extract both escalation and pending count in one acquisition
            let (escalation, fate_pending) = {
                let mut fate = self
                    .fate
                    .lock()
                    .map_err(|e| BridgeError::FateLockPoisoned {
                        message: format!("FATE mutex poisoned: {}", e),
                    })?;
                let esc = fate.escalate_rejection(
                    &validation.rejection_codes,
                    &request.task,
                    &request.context,
                );
                let pending = fate.pending_count();
                (esc, pending)
            };

            // Record FATE escalation metrics (no second lock needed)
            metrics::record_fate_escalation(&format!("{:?}", escalation.level), fate_pending);

            // Collect rejecting and approving validators
            let rejecting: Vec<String> = validation
                .validations
                .iter()
                .filter(|v| !v.approved)
                .map(|v| v.agent_name.clone())
                .collect();
            let approving: Vec<String> = validation
                .validations
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
                request_id.clone(),
            );

            // Record receipt emission
            metrics::record_receipt_emitted("rejection");

            // Anchor rejection receipt to blockchain (best-effort, don't block on failure)
            if let Err(e) = self.receipts.anchor_rejection_to_chain(&receipt).await {
                warn!(error = %e, receipt_id = %receipt.receipt_id, "Blockchain anchoring failed for rejection receipt");
            }

            // Mark idempotency slot as failed (allows retry)
            self.idempotency.fail(&idem_key, "SAT_REJECTION");

            warn!(
                receipt_id = %receipt.receipt_id,
                escalation_id = %escalation.id,
                escalation_level = ?escalation.level,
                "🚨 Request BLOCKED by SAT - receipt emitted"
            );

            let rejection_message = validation
                .rejection_codes
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join("; ");
            return Err(BridgeError::SatBlocked {
                message: rejection_message,
                escalation_id: escalation.id,
                receipt_id: receipt.receipt_id,
            }
            .into());
        }

        info!(
            validation_time_ms = sat_validation_time.as_millis(),
            "✅ SAT validation passed"
        );

        // IFC: Promote to Validated after SAT consensus
        let _ = taint_ctx.promote("request.task", IntegrityLevel::Validated);

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

        // IFC: Promote to Attested after PAT execution + SAT post-eval
        let _ = taint_ctx.promote("request.task", IntegrityLevel::Attested);

        // Step 4: Calculate synergy and Ihsan scores
        let synergy_score = self.calculate_synergy(&pat_results, &sat_evaluations);
        let (ihsan_score, ihsan_vector) = self.calculate_ihsan(&pat_results, &sat_evaluations)?;

        let ihsan_env = ihsan::current_env();
        let ihsan_artifact_class = "docs";
        let ihsan_threshold_applied =
            ihsan::constitution().threshold_for(&ihsan_env, ihsan_artifact_class);
        let ihsan_passes_threshold = ihsan_score >= ihsan_threshold_applied;

        // Record Ihsān metrics
        metrics::record_ihsan_result(
            ihsan_score,
            ihsan_passes_threshold,
            &ihsan_env,
            &ihsan_vector,
        );

        if !ihsan_passes_threshold && ihsan::should_enforce() {
            // FATE escalation for Ihsān failure
            // Use scoped lock to extract both escalation and pending count in one acquisition
            let (escalation, fate_pending) = {
                let mut fate = self
                    .fate
                    .lock()
                    .map_err(|e| BridgeError::FateLockPoisoned {
                        message: format!("FATE mutex poisoned: {}", e),
                    })?;
                let esc = fate.escalate_ihsan_failure(
                    &ihsan_env,
                    ihsan_artifact_class,
                    ihsan_score,
                    ihsan_threshold_applied,
                    &request.context,
                );
                let pending = fate.pending_count();
                (esc, pending)
            };

            // Record FATE escalation for Ihsān failure (no second lock needed)
            metrics::record_fate_escalation(&format!("{:?}", escalation.level), fate_pending);

            // Mark idempotency slot as failed (allows retry)
            self.idempotency.fail(&idem_key, "IHSAN_GATE_FAILED");

            warn!(
                escalation_id = %escalation.id,
                ihsan_score = ihsan_score,
                threshold = ihsan_threshold_applied,
                "⚠️ Ihsān gate failed - escalated via FATE"
            );

            return Err(BridgeError::IhsanGateFailed {
                env: ihsan_env,
                score: ihsan_score,
                threshold: ihsan_threshold_applied,
                escalation_id: escalation.id,
            }
            .into());
        }

        let total_latency = start.elapsed();

        // Record request latency metrics
        metrics::record_request_completion("success", total_latency.as_secs_f64(), synergy_score);

        // Emit execution receipt for successful flow
        let sat_approvers = validation.validations.iter().filter(|v| v.approved).count();
        let execution_receipt = self.receipts.emit_execution(
            &request.task,
            sat_validation_time.as_millis(),
            pat_execution_time.as_millis(),
            total_latency.as_millis(),
            synergy_score,
            ihsan_score,
            ihsan_threshold_applied,
            pat_results.len(),
            sat_approvers,
            request_id,
        );

        // Record execution receipt emission
        metrics::record_receipt_emitted("execution");

        // Anchor execution receipt to blockchain (best-effort, don't block on failure)
        if let Err(e) = self.receipts.anchor_execution_to_chain(&execution_receipt).await {
            warn!(error = %e, receipt_id = %execution_receipt.receipt_id, "Blockchain anchoring failed for execution receipt");
        }

        // Build response
        let response = DualAgenticResponse {
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
                "execution_mode": "PRODUCTION",
                "validation_time_ms": sat_validation_time.as_millis(),
                "pat_execution_time_ms": pat_execution_time.as_millis(),
                "ihsan_constitution_id": ihsan::constitution().id(),
                "ihsan_threshold_baseline": ihsan::constitution().threshold(),
                "ihsan_env": ihsan_env,
                "ihsan_artifact_class": ihsan_artifact_class,
                "ihsan_threshold_applied": ihsan_threshold_applied,
                "ihsan_passes_threshold": ihsan_passes_threshold,
                "ihsan_vector": ihsan_vector,
                "ihsan_vector_source": "real_confidence_mapping_v1",
                "fate_pending_escalations": self.fate.lock().map(|f| f.pending_count()).unwrap_or(0),
                "idempotency_key": &idem_key,
                "entropy_pool_level": entropy::global_pool().pool_level(),
                "ifc_integrity": taint_ctx.get_label("request.task").integrity.to_string(),
                "ifc_secrecy": taint_ctx.get_label("request.task").secrecy.to_string(),
                "ifc_audit_count": taint_ctx.audit_log().len(),
            }),
        };

        // Cache result for exactly-once semantics
        if let Ok(json) = serde_json::to_string(&response) {
            self.idempotency.complete(&idem_key, &json);
            debug!(
                idem_key = %idem_key,
                "💾 Response cached for exactly-once semantics"
            );
        }

        info!(
            synergy = synergy_score,
            ihsan = ihsan_score,
            latency_ms = total_latency.as_millis(),
            idem_key = %idem_key,
            "✅ Dual-agentic execution completed - receipt emitted"
        );

        Ok(response)
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

        /// Find agent and track whether fallback was used
        fn find_with_tracking(
            results: &[AgentResult],
            name: &str,
            fallback: f64,
            missing_agents: &mut Vec<String>,
        ) -> f64 {
            match results.iter().find(|r| r.agent_name == name) {
                Some(r) => r.confidence,
                None => {
                    missing_agents.push(name.to_string());
                    fallback
                }
            }
        }

        let pat_avg = avg(pat_results);
        let sat_avg = avg(sat_results);

        // Track missing agents for validation logging
        let mut missing_agents: Vec<String> = Vec::new();

        let mut scores = BTreeMap::new();

        // Dimension mappings with explicit agent tracking
        // PAT agents use PascalCase, SAT agents use snake_case
        scores.insert(
            "correctness".to_string(),
            clamp01(find_with_tracking(
                pat_results,
                "EthicsGuardian",
                pat_avg,
                &mut missing_agents,
            )),
        );
        scores.insert(
            "safety".to_string(),
            clamp01(find_with_tracking(
                sat_results,
                "security_guardian",
                sat_avg,
                &mut missing_agents,
            )),
        );
        scores.insert(
            "user_benefit".to_string(),
            clamp01(find_with_tracking(
                pat_results,
                "Communicator",
                pat_avg,
                &mut missing_agents,
            )),
        );
        scores.insert(
            "efficiency".to_string(),
            clamp01(find_with_tracking(
                sat_results,
                "performance_monitor",
                sat_avg,
                &mut missing_agents,
            )),
        );
        scores.insert(
            "auditability".to_string(),
            clamp01(find_with_tracking(
                sat_results,
                "consistency_checker",
                sat_avg,
                &mut missing_agents,
            )),
        );
        scores.insert(
            "anti_centralization".to_string(),
            clamp01(find_with_tracking(
                sat_results,
                "resource_optimizer",
                sat_avg,
                &mut missing_agents,
            )),
        );
        scores.insert(
            "robustness".to_string(),
            clamp01(self.calculate_consistency(pat_results)),
        );
        scores.insert(
            "adl_fairness".to_string(),
            clamp01(find_with_tracking(
                sat_results,
                "ethics_validator",
                sat_avg,
                &mut missing_agents,
            )),
        );

        // FAIL-VISIBLE: Log warnings for missing agents (required for audit trail)
        if !missing_agents.is_empty() {
            warn!(
                missing_agents = ?missing_agents,
                pat_avg = pat_avg,
                sat_avg = sat_avg,
                "⚠️ Ihsān calculation used fallback values for {} missing agent(s). \
                 This may indicate incomplete execution or agent spawn failure.",
                missing_agents.len()
            );

            // Critical agents that MUST be present for valid Ihsān scoring
            let critical_agents = ["EthicsGuardian", "security_guardian"];
            let missing_critical: Vec<&String> = missing_agents
                .iter()
                .filter(|a| critical_agents.contains(&a.as_str()))
                .collect();

            if !missing_critical.is_empty() {
                // FAIL-CLOSED: Critical agents missing — cannot produce valid Ihsan score
                // This enforces "la naftarid" (we do not assume) for safety-critical dimensions
                return Err(BridgeError::CriticalAgentsMissing {
                    missing_agents: missing_critical.iter().map(|s| s.to_string()).collect(),
                }.into());
            }
        }

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
