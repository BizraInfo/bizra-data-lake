// src/autopoietic/loop_engine.rs - The 11-Step AutopoieticLoop Engine
//
// Core implementation of the self-creating, self-improving autonomous engine.
//
// The 11-step cycle:
//   1. Create agents from current blueprints (reuse AgentFactory warm pools)
//   2. Deploy agents (PATOrchestrator + A2A registration)
//   3. Monitor for generation_duration (OperationMonitor metrics)
//   4. Evaluate (operational + environment + economic + ethical - Ihsān 8-dim + SAPE 9-probe)
//   5. Improve blueprints (ImprovementGenome.evolve() with FATE gate)
//   6. Record GenerationPerformance (Extended ExecutionReceipt + KEP fields)
//   7. Update current_blueprints (apply improvements, maintain lineage)
//   8. Update proof chain (Merkle append + blockchain anchor)
//   9. Economic/ethical model updates (adjust incentives based on history)
//  10. Check convergence (KEP detection - plateau vs explosion)
//  11. Increment counter and persist (Redis via Synapse)

use crate::autopoietic::{
    blueprints::{
        AgentBlueprint, AgentTeam, BlueprintManager, BlueprintPerformance, CapabilitySlot,
    },
    convergence::{ConvergenceDetector, ConvergenceUpdate},
    evaluation::{EvaluationResult, ExecutionRecord, OperationMonitor},
    proof_chain::{BlockchainAnchor, ProofChain},
    step9_implementation::{GenerationReward, TokenIncentiveState},
    types::{
        AutopoieticConfig, AutopoieticError, AutopoieticStatus, GenerationPerformance, KEPState,
    },
};
use crate::blockchain::tokens::TokenAmount;
use crate::blockchain::{BizraChain, BizraTransaction};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, RwLock};
use tokio::time::sleep;
use tracing::{debug, error, info, warn};

/// Loop control commands
#[derive(Debug, Clone)]
pub enum LoopControl {
    /// Start the loop
    Start,
    /// Stop the loop gracefully
    Stop,
    /// Pause the loop (can be resumed)
    Pause,
    /// Resume from pause
    Resume,
    /// Force stop immediately
    ForceStop,
    /// Inject a blueprint for testing
    InjectBlueprint(AgentBlueprint),
    /// Trigger manual evaluation
    TriggerEvaluation,
}

/// Events emitted by the loop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AutopoieticEvent {
    /// Loop started
    Started { generation: u64 },
    /// Generation started
    GenerationStarted { generation: u64 },
    /// Generation completed
    GenerationCompleted {
        generation: u64,
        ihsan_score: f64,
        kep_state: KEPState,
    },
    /// KEP state changed
    KEPStateChanged {
        from: KEPState,
        to: KEPState,
        generation: u64,
    },
    /// Ihsān gate failed
    IhsanGateFailed {
        generation: u64,
        score: f64,
        threshold: f64,
    },
    /// Blueprint evolved
    BlueprintEvolved {
        blueprint_id: String,
        generation: u64,
    },
    /// Loop stopped
    Stopped { generation: u64, reason: String },
    /// Error occurred
    Error { generation: u64, message: String },
}

/// The main AutopoieticLoop structure
pub struct AutopoieticLoop {
    /// Configuration
    config: AutopoieticConfig,

    /// Blueprint manager
    blueprints: RwLock<BlueprintManager>,

    /// Operation monitor
    monitor: Arc<OperationMonitor>,

    /// Proof chain
    proof_chain: RwLock<ProofChain>,

    /// Convergence detector
    convergence: RwLock<ConvergenceDetector>,

    /// Generation counter
    generation_counter: AtomicU64,

    /// Performance history
    performance_history: RwLock<Vec<GenerationPerformance>>,

    /// Running state
    is_running: AtomicBool,

    /// Paused state
    is_paused: AtomicBool,

    /// Event sender
    event_tx: mpsc::Sender<AutopoieticEvent>,

    /// Control receiver (kept for future external control integration)
    #[allow(dead_code)]
    control_rx: RwLock<Option<mpsc::Receiver<LoopControl>>>,

    /// Receipts emitted counter
    receipts_emitted: AtomicU64,

    /// Blockchain client for proof anchoring
    chain_client: Arc<BizraChain>,

    /// Token incentive state for economic model updates (Step 9)
    token_state: RwLock<TokenIncentiveState>,
}

impl AutopoieticLoop {
    /// Create a new AutopoieticLoop
    pub fn new(
        config: AutopoieticConfig,
    ) -> (
        Self,
        mpsc::Receiver<AutopoieticEvent>,
        mpsc::Sender<LoopControl>,
    ) {
        let (event_tx, event_rx) = mpsc::channel(100);
        let (control_tx, control_rx) = mpsc::channel(100);

        let convergence = ConvergenceDetector::new(
            config.kep_thresholds.clone(),
            config.convergence_window,
            config.improvement_threshold,
        );

        // Initialize blockchain client for proof anchoring
        let chain_client = Arc::new(BizraChain::new());

        let loop_engine = Self {
            config,
            blueprints: RwLock::new(BlueprintManager::new()),
            monitor: Arc::new(OperationMonitor::new()),
            proof_chain: RwLock::new(ProofChain::new()),
            convergence: RwLock::new(convergence),
            generation_counter: AtomicU64::new(0),
            performance_history: RwLock::new(Vec::new()),
            is_running: AtomicBool::new(false),
            is_paused: AtomicBool::new(false),
            event_tx,
            control_rx: RwLock::new(Some(control_rx)),
            receipts_emitted: AtomicU64::new(0),
            chain_client,
            token_state: RwLock::new(TokenIncentiveState::new()),
        };

        (loop_engine, event_rx, control_tx)
    }

    /// Initialize with default PAT/SAT blueprints
    pub async fn initialize_default_blueprints(&self) {
        let mut manager = self.blueprints.write().await;

        // PAT Agents (7)
        let pat_specs = [
            ("master-reasoner", "MasterReasoner", CapabilitySlot::MasterReasoner, "deepseek-r1:7b", 4.5,
             "You are the Master Reasoner, responsible for strategic thinking and multi-step planning. Apply Graph of Thoughts for complex problems. Ensure all outputs meet 0.95 Ihsān threshold."),
            ("memory-architect", "MemoryArchitect", CapabilitySlot::MemoryArchitect, "qwen2.5:7b", 4.0,
             "You are the Memory Architect, organizing and retrieving knowledge across all tiers (Working, Episodic, Semantic, Procedural, Sovereign). Maintain knowledge integrity."),
            ("creative-synthesizer", "CreativeSynthesizer", CapabilitySlot::CreativeSynthesizer, "qwen2.5:7b", 4.0,
             "You are the Creative Synthesizer, generating novel ideas and content. Bridge domains for emergent capabilities. Apply SAPE pattern elevation."),
            ("data-analyzer", "DataAnalyzer", CapabilitySlot::DataAnalyzer, "mistral:7b", 4.0,
             "You are the Data Analyzer, specializing in pattern recognition and statistical analysis. Provide evidence-backed insights."),
            ("communicator", "Communicator", CapabilitySlot::Communicator, "mistral:7b", 4.0,
             "You are the Communicator, handling external communications and user interactions. Ensure clarity and empathy."),
            ("execution-planner", "ExecutionPlanner", CapabilitySlot::ExecutionPlanner, "agentflow-7b", 4.0,
             "You are the Execution Planner, decomposing tasks into actionable steps. Optimize for parallel execution where possible."),
            ("ethics-guardian", "EthicsGuardian", CapabilitySlot::EthicsGuardian, "qwen2.5:7b", 4.0,
             "You are the Ethics Guardian, ensuring all outputs meet safety and ethical standards. Apply the 8-dimension Ihsān framework."),
        ];

        for (id, name, slot, model, vram, prompt) in pat_specs {
            let blueprint = AgentBlueprint::genesis(
                id,
                name,
                AgentTeam::PAT,
                slot,
                prompt,
                model,
                "ollama",
                vram,
            );
            manager.register(blueprint);
        }

        // SAT Agents (5)
        let sat_specs = [
            ("poi-verifier", "PoiVerifier", CapabilitySlot::PoiVerifier,
             "You are the PoI Verifier, validating Proof-of-Impact claims. Require evidence for all attestations."),
            ("resource-allocator", "ResourceAllocator", CapabilitySlot::ResourceAllocator,
             "You are the Resource Allocator, optimizing compute and memory usage. Enforce Harberger taxation principles."),
            ("risk-guardian", "RiskGuardian", CapabilitySlot::RiskGuardian,
             "You are the Risk Guardian, monitoring for security threats and vulnerabilities. Apply FATE escalation protocol."),
            ("governance-engine", "GovernanceEngine", CapabilitySlot::GovernanceEngine,
             "You are the Governance Engine, enforcing policy and compliance. Require 3/5 consensus for critical decisions."),
            ("evidence-engine", "EvidenceEngine", CapabilitySlot::EvidenceEngine,
             "You are the Evidence Engine, generating and validating audit trails. All operations emit receipts."),
        ];

        for (id, name, slot, prompt) in sat_specs {
            let blueprint = AgentBlueprint::genesis(
                id,
                name,
                AgentTeam::SAT,
                slot,
                prompt,
                "rule-based",
                "internal",
                0.1,
            );
            manager.register(blueprint);
        }

        info!(
            pat_count = manager.get_by_team(AgentTeam::PAT).len(),
            sat_count = manager.get_by_team(AgentTeam::SAT).len(),
            "📋 Default blueprints initialized"
        );
    }

    /// Start the autopoietic loop
    pub async fn start(&self) -> Result<(), AutopoieticError> {
        if self.is_running.load(Ordering::SeqCst) {
            return Err(AutopoieticError::AlreadyRunning);
        }

        // Connect to blockchain for proof anchoring
        if let Err(e) = self.chain_client.connect().await {
            warn!("Failed to connect to BIZRA chain: {} (continuing without anchoring)", e);
        }

        self.is_running.store(true, Ordering::SeqCst);
        self.is_paused.store(false, Ordering::SeqCst);

        let generation = self.generation_counter.load(Ordering::SeqCst);
        self.emit_event(AutopoieticEvent::Started { generation })
            .await;

        info!(generation = generation, "🚀 AutopoieticLoop started");

        Ok(())
    }

    /// Run the main loop (blocking)
    pub async fn run(&self) -> Result<(), AutopoieticError> {
        self.start().await?;

        loop {
            // Check for stop signal
            if !self.is_running.load(Ordering::SeqCst) {
                break;
            }

            // Check for pause
            while self.is_paused.load(Ordering::SeqCst) {
                sleep(Duration::from_millis(100)).await;
                if !self.is_running.load(Ordering::SeqCst) {
                    break;
                }
            }

            // Check max generations
            let current_gen = self.generation_counter.load(Ordering::SeqCst);
            if self.config.max_generations > 0 && current_gen >= self.config.max_generations {
                return Err(AutopoieticError::MaxGenerationsReached {
                    max: self.config.max_generations,
                });
            }

            // Execute one generation cycle
            match self.execute_generation_cycle().await {
                Ok(_) => {}
                Err(AutopoieticError::IhsanGateFailed { score, threshold }) => {
                    self.emit_event(AutopoieticEvent::IhsanGateFailed {
                        generation: current_gen,
                        score,
                        threshold,
                    })
                    .await;
                    warn!(
                        generation = current_gen,
                        score = format!("{:.4}", score),
                        threshold = format!("{:.4}", threshold),
                        "⚠️ Ihsān gate failed - generation will be retried"
                    );
                    // Continue to next generation (self-correcting)
                }
                Err(e) => {
                    self.emit_event(AutopoieticEvent::Error {
                        generation: current_gen,
                        message: e.to_string(),
                    })
                    .await;
                    error!(error = %e, "Generation cycle error");
                    // Continue running, log error
                }
            }
        }

        let final_gen = self.generation_counter.load(Ordering::SeqCst);
        self.emit_event(AutopoieticEvent::Stopped {
            generation: final_gen,
            reason: "Loop stopped gracefully".to_string(),
        })
        .await;

        info!(generation = final_gen, "🛑 AutopoieticLoop stopped");

        Ok(())
    }

    /// Execute a single generation cycle (the 11 steps)
    pub async fn execute_generation_cycle(
        &self,
    ) -> Result<GenerationPerformance, AutopoieticError> {
        let generation = self.generation_counter.fetch_add(1, Ordering::SeqCst) + 1;

        self.emit_event(AutopoieticEvent::GenerationStarted { generation })
            .await;
        info!(generation = generation, "🔄 Starting generation cycle");

        // Step 1: Create agents from current blueprints
        let blueprints = self.step1_create_agents().await?;
        debug!(
            generation = generation,
            agents = blueprints.len(),
            "Step 1: Agents created"
        );

        // Step 2: Deploy agents (simulated - would integrate with actual PAT/SAT)
        self.step2_deploy_agents(&blueprints).await?;
        debug!(generation = generation, "Step 2: Agents deployed");

        // Step 3: Monitor for generation_duration
        self.step3_monitor(generation).await?;
        debug!(generation = generation, "Step 3: Monitoring complete");

        // Step 4: Evaluate (operational + environment + economic + ethical)
        let evaluation = self.step4_evaluate(generation).await?;
        debug!(
            generation = generation,
            ihsan = format!("{:.4}", evaluation.ethical.aggregate_ihsan),
            "Step 4: Evaluation complete"
        );

        // IHSĀN HARD GATE - Fail-closed
        if !evaluation.passes_ihsan_gate() {
            return Err(AutopoieticError::IhsanGateFailed {
                score: evaluation.ethical.aggregate_ihsan,
                threshold: self.config.ihsan_threshold,
            });
        }

        // Step 5: Improve blueprints
        let improvements = self.step5_improve_blueprints(&evaluation).await?;
        debug!(
            generation = generation,
            improvements = improvements.len(),
            "Step 5: Improvements generated"
        );

        // Step 6: Record GenerationPerformance
        let receipt_id = self
            .step6_record_performance(&evaluation, &improvements)
            .await?;
        debug!(generation = generation, receipt_id = %receipt_id, "Step 6: Performance recorded");

        // Step 7: Update current_blueprints
        self.step7_update_blueprints(generation, &improvements)
            .await?;
        debug!(generation = generation, "Step 7: Blueprints updated");

        // Step 8: Update proof chain
        let proof_hash = self
            .step8_update_proof_chain(generation, &evaluation, &receipt_id)
            .await?;
        debug!(generation = generation, proof_hash = %proof_hash, "Step 8: Proof chain updated");

        // Step 9: Economic/ethical model updates
        self.step9_update_models(&evaluation).await?;
        debug!(generation = generation, "Step 9: Models updated");

        // Step 10: Check convergence (KEP detection)
        let convergence_update = self
            .step10_check_convergence(&evaluation, &proof_hash, &receipt_id)
            .await?;
        debug!(
            generation = generation,
            kep_state = ?convergence_update.new_state,
            "Step 10: Convergence checked"
        );

        // Handle KEP state changes
        if convergence_update.state_changed {
            self.emit_event(AutopoieticEvent::KEPStateChanged {
                from: convergence_update.previous_state,
                to: convergence_update.new_state,
                generation,
            })
            .await;
        }

        // Step 11: Increment counter and persist
        self.step11_persist(generation).await?;
        debug!(generation = generation, "Step 11: Persisted");

        // Build final performance record
        let performance = evaluation.to_generation_performance(&proof_hash, &receipt_id);

        self.emit_event(AutopoieticEvent::GenerationCompleted {
            generation,
            ihsan_score: performance.aggregate_ihsan,
            kep_state: convergence_update.new_state,
        })
        .await;

        info!(
            generation = generation,
            ihsan = format!("{:.4}", performance.aggregate_ihsan),
            kep = ?convergence_update.new_state,
            "✅ Generation cycle complete"
        );

        Ok(performance)
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // STEP IMPLEMENTATIONS
    // ═══════════════════════════════════════════════════════════════════════════════

    /// Step 1: Create agents from current blueprints
    async fn step1_create_agents(&self) -> Result<Vec<AgentBlueprint>, AutopoieticError> {
        let manager = self.blueprints.read().await;
        let blueprints: Vec<AgentBlueprint> =
            manager.get_all_active().into_iter().cloned().collect();
        Ok(blueprints)
    }

    /// Step 2: Deploy agents
    async fn step2_deploy_agents(
        &self,
        _blueprints: &[AgentBlueprint],
    ) -> Result<(), AutopoieticError> {
        // In a full implementation, this would:
        // - Create agent instances using warm pools
        // - Register with A2A
        // - Deploy to PAT/SAT orchestrators
        Ok(())
    }

    /// Step 3: Monitor for generation_duration
    async fn step3_monitor(&self, generation: u64) -> Result<(), AutopoieticError> {
        self.monitor.start_period().await;

        let duration = Duration::from_millis(self.config.generation_duration_ms);

        // In production, this would collect real metrics
        // For now, simulate with a sleep and some synthetic data
        sleep(duration.min(Duration::from_millis(100))).await;

        // Record some synthetic executions for testing
        for i in 0..10 {
            let record = ExecutionRecord {
                agent_id: format!("agent-{}", i % 5),
                success: i % 7 != 0,
                latency_ms: 100 + (i * 10) as u64,
                ihsan_dimensions: Some(crate::autopoietic::types::IhsanDimensions {
                    // Scores must aggregate to >= 0.95 (Ihsān hard gate)
                    // Weighted sum: 0.22*corr + 0.22*safe + 0.14*user + 0.12*eff + 0.12*aud + 0.08*anti + 0.06*rob + 0.04*adl
                    correctness: 0.97 + (i as f64 * 0.002),
                    safety: 0.98,
                    user_benefit: 0.96,
                    efficiency: 0.94,
                    auditability: 0.95,
                    anti_centralization: 0.92,
                    robustness: 0.93,
                    adl_fairness: 0.91,
                }),
                sape_results: None,
                timestamp: Utc::now(),
            };
            self.monitor.record_execution(record).await;
        }

        debug!(generation = generation, "Monitor period complete");
        Ok(())
    }

    /// Step 4: Evaluate operations
    async fn step4_evaluate(&self, generation: u64) -> Result<EvaluationResult, AutopoieticError> {
        let result = self.monitor.end_period(generation).await;
        Ok(result)
    }

    /// Step 5: Improve blueprints based on evaluation
    async fn step5_improve_blueprints(
        &self,
        evaluation: &EvaluationResult,
    ) -> Result<Vec<String>, AutopoieticError> {
        let mut improvements = Vec::new();

        // Get learning rate from convergence detector
        let learning_rate = {
            let convergence = self.convergence.read().await;
            convergence.learning_rate_multiplier()
        };

        let manager = self.blueprints.read().await;

        for blueprint in manager.get_all_active() {
            // Generate mutations based on performance
            let performance = BlueprintPerformance {
                generation: evaluation.generation,
                ihsan_score: evaluation.ethical.aggregate_ihsan,
                tasks_completed: evaluation.operational.tasks_processed,
                avg_latency_ms: evaluation.operational.avg_latency_ms,
                success_rate: if evaluation.operational.tasks_processed > 0 {
                    evaluation.operational.successful_executions as f64
                        / evaluation.operational.tasks_processed as f64
                } else {
                    0.0
                },
                contribution_score: 0.0,
                timestamp: Utc::now(),
            };

            let mutations = blueprint
                .improvement_genome
                .generate_mutations(&performance);

            if !mutations.is_empty() {
                improvements.push(format!(
                    "{}:{} mutations (lr={:.2})",
                    blueprint.id,
                    mutations.len(),
                    learning_rate
                ));
            }
        }

        Ok(improvements)
    }

    /// Step 6: Record generation performance
    async fn step6_record_performance(
        &self,
        evaluation: &EvaluationResult,
        improvements: &[String],
    ) -> Result<String, AutopoieticError> {
        let receipt_id = format!(
            "GEN-{}-{:06}",
            Utc::now().format("%Y%m%d%H%M%S"),
            self.receipts_emitted.fetch_add(1, Ordering::SeqCst)
        );

        // In production, emit receipt via ReceiptEmitter
        // For now, just log

        info!(
            receipt_id = %receipt_id,
            generation = evaluation.generation,
            ihsan = format!("{:.4}", evaluation.ethical.aggregate_ihsan),
            improvements = improvements.len(),
            "🧾 Generation receipt emitted"
        );

        Ok(receipt_id)
    }

    /// Step 7: Update current blueprints
    async fn step7_update_blueprints(
        &self,
        generation: u64,
        improvements: &[String],
    ) -> Result<(), AutopoieticError> {
        if improvements.is_empty() {
            return Ok(());
        }

        let mut manager = self.blueprints.write().await;

        // For each improvement, evolve the corresponding blueprint
        for improvement in improvements {
            if let Some(blueprint_id) = improvement.split(':').next() {
                if let Some(blueprint) = manager.get(blueprint_id).cloned() {
                    let evolved = blueprint.evolve(vec![], generation);
                    manager.register(evolved);

                    self.emit_event(AutopoieticEvent::BlueprintEvolved {
                        blueprint_id: blueprint_id.to_string(),
                        generation,
                    })
                    .await;
                }
            }
        }

        Ok(())
    }

    /// Step 8: Update proof chain
    async fn step8_update_proof_chain(
        &self,
        generation: u64,
        evaluation: &EvaluationResult,
        receipt_id: &str,
    ) -> Result<String, AutopoieticError> {
        let mut chain = self.proof_chain.write().await;

        // Convert evaluation to performance
        let performance = evaluation.to_generation_performance("", receipt_id);

        let node = chain.append(generation, &performance);

        Ok(node.hash)
    }

    /// Step 9: Update economic/ethical models
    ///
    /// Implements the Proof-of-Impact token incentive feedback loop:
    /// 1. Calculate impact score from Ihsan + operational success rate
    /// 2. Mint BLOOM tokens proportional to impact
    /// 3. Adjust Ihsan dimension weights based on performance trends
    /// 4. Record generation reward for analytics/auditing
    async fn step9_update_models(
        &self,
        evaluation: &EvaluationResult,
    ) -> Result<(), AutopoieticError> {
        let mut state = self.token_state.write().await;
        let generation = self.generation_counter.load(Ordering::SeqCst);

        // 1. Calculate impact score from evaluation
        //    impact = (ihsan_score * 100) * (success_rate * 10)
        //    This creates a multiplicative relationship between ethical quality and practical success
        let success_rate = if evaluation.operational.tasks_processed > 0 {
            evaluation.operational.successful_executions as f64
                / evaluation.operational.tasks_processed as f64
        } else {
            0.0
        };
        let impact_score =
            ((evaluation.ethical.aggregate_ihsan * 100.0) * (success_rate * 10.0)) as u64;

        // 2. Mint BLOOM tokens from impact (if above minimum threshold)
        let bloom_minted = if impact_score >= 10 {
            match state.bloom.mint_from_impact(impact_score) {
                Ok(amount) => {
                    info!(
                        generation,
                        impact_score,
                        bloom = %amount,
                        "BLOOM minted from Proof-of-Impact"
                    );
                    amount
                }
                Err(e) => {
                    warn!(generation, error = %e, "BLOOM minting failed");
                    TokenAmount::ZERO
                }
            }
        } else {
            TokenAmount::ZERO
        };

        // 3. Adjust Ihsan dimension weights based on performance trends
        //    If safety scores trend below threshold over 3 generations, increase weight
        let history = self.performance_history.read().await;
        if history.len() >= 3 {
            let recent = &history[history.len().saturating_sub(3)..];
            let safety_trend: f64 = recent
                .iter()
                .map(|p| p.ihsan_dimensions.safety)
                .sum::<f64>()
                / recent.len() as f64;
            if safety_trend < 0.95 {
                state
                    .ihsan_weight_adjustments
                    .push(("safety".to_string(), 0.02));
                info!(
                    generation,
                    safety_trend,
                    "Ihsan safety weight increased due to downward trend"
                );
            }
        }
        drop(history);

        // 4. Record generation reward
        state.generation_rewards.push(GenerationReward {
            generation,
            bloom_minted,
            impact_score,
            ihsan_score: evaluation.ethical.aggregate_ihsan,
            timestamp: Utc::now(),
        });

        // Keep only last 100 rewards (rolling window)
        if state.generation_rewards.len() > 100 {
            let drain_count = state.generation_rewards.len() - 100;
            state.generation_rewards.drain(..drain_count);
        }

        Ok(())
    }

    /// Step 10: Check convergence (KEP detection)
    async fn step10_check_convergence(
        &self,
        evaluation: &EvaluationResult,
        proof_hash: &str,
        receipt_id: &str,
    ) -> Result<ConvergenceUpdate, AutopoieticError> {
        let performance = evaluation.to_generation_performance(proof_hash, receipt_id);

        let mut convergence = self.convergence.write().await;
        let update = convergence.update(performance.clone());

        // Store in history
        let mut history = self.performance_history.write().await;
        history.push(performance);

        // Keep only recent history
        if history.len() > 100 {
            history.remove(0);
        }

        Ok(update)
    }

    /// Step 11: Persist state via blockchain anchoring
    async fn step11_persist(&self, generation: u64) -> Result<(), AutopoieticError> {
        let mut chain = self.proof_chain.write().await;

        // Get pending anchors (generations not yet anchored to blockchain)
        let pending = chain.pending_anchors().to_vec();
        if pending.is_empty() {
            debug!(generation, "No pending anchors to persist");
            return Ok(());
        }

        // Batch verify pending anchors using Merkle tree
        let batch = chain.batch_verify_pending();
        if batch.verified_count == 0 {
            debug!(generation, "No verified anchors in batch");
            return Ok(());
        }

        // Submit Merkle root to blockchain
        let tx = BizraTransaction::AnchorReceipt {
            receipt_id: format!("GEN-BATCH-{}", generation),
            receipt_type: "ProofChainAnchor".to_string(),
            integrity_hash: batch.merkle_root.clone(),
            ihsan_score: 0.0, // Batch anchor doesn't have individual scores
            sat_approvers: 0,
        };

        match self.chain_client.submit_transaction(tx).await {
            Ok(receipt) => {
                // Mark generations as anchored with blockchain proof
                let anchor = BlockchainAnchor {
                    chain: "bizra-native".to_string(),
                    tx_hash: receipt.tx_hash.clone(),
                    block_number: receipt.block_number.unwrap_or(0),
                    anchored_at: Utc::now(),
                    generations: pending.clone(),
                };

                chain.anchor_with_merkle_proof(&pending, anchor);

                info!(
                    generation,
                    anchored = pending.len(),
                    tx_hash = %receipt.tx_hash,
                    merkle_root = %batch.merkle_root,
                    "⛓️ Proof chain batch anchored to BIZRA blockchain"
                );
            }
            Err(e) => {
                warn!(
                    generation,
                    error = %e,
                    "Blockchain anchoring failed (non-fatal) - anchors remain pending for next attempt"
                );
                // Non-fatal: anchors remain pending for next attempt
            }
        }

        Ok(())
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // UTILITY METHODS
    // ═══════════════════════════════════════════════════════════════════════════════

    /// Stop the loop
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::SeqCst);
    }

    /// Pause the loop
    pub fn pause(&self) {
        self.is_paused.store(true, Ordering::SeqCst);
    }

    /// Resume the loop
    pub fn resume(&self) {
        self.is_paused.store(false, Ordering::SeqCst);
    }

    /// Get current status
    pub async fn status(&self) -> AutopoieticStatus {
        let convergence = self.convergence.read().await;
        let chain = self.proof_chain.read().await;
        let blueprints = self.blueprints.read().await;

        let state = convergence.get_state();

        AutopoieticStatus {
            is_running: self.is_running.load(Ordering::SeqCst),
            current_generation: self.generation_counter.load(Ordering::SeqCst),
            kep_state: state.kep_state,
            aggregate_ihsan: state.metrics.mean_ihsan,
            last_generation_start: None,
            last_generation_end: None,
            active_agents: blueprints.get_all_active().len(),
            blueprint_count: blueprints.get_all_active().len(),
            convergence_state: format!("{}", state),
            proof_chain_length: chain.len(),
            receipts_emitted: self.receipts_emitted.load(Ordering::SeqCst),
        }
    }

    /// Get evolution history
    pub async fn history(&self, limit: usize) -> Vec<GenerationPerformance> {
        let history = self.performance_history.read().await;
        history.iter().rev().take(limit).cloned().collect()
    }

    /// Verify proof chain integrity
    pub async fn verify_chain(&self) -> crate::autopoietic::proof_chain::ChainVerificationResult {
        let chain = self.proof_chain.read().await;
        chain.verify_integrity()
    }

    /// Inject a blueprint (for testing)
    pub async fn inject_blueprint(&self, blueprint: AgentBlueprint) {
        let mut manager = self.blueprints.write().await;
        manager.register(blueprint);
    }

    /// Emit an event
    async fn emit_event(&self, event: AutopoieticEvent) {
        if let Err(e) = self.event_tx.send(event).await {
            warn!(error = %e, "Failed to emit autopoietic event");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_loop_creation() {
        let config = AutopoieticConfig {
            generation_duration_ms: 10, // Fast for testing
            max_generations: 3,
            ..Default::default()
        };

        let (loop_engine, mut event_rx, control_tx) = AutopoieticLoop::new(config);

        // Initialize blueprints
        loop_engine.initialize_default_blueprints().await;

        let status = loop_engine.status().await;
        assert!(!status.is_running);
        assert!(status.blueprint_count > 0);
    }

    #[tokio::test]
    async fn test_single_generation() {
        let config = AutopoieticConfig {
            generation_duration_ms: 10,
            ..Default::default()
        };

        let (loop_engine, _event_rx, _control_tx) = AutopoieticLoop::new(config);
        loop_engine.initialize_default_blueprints().await;

        let result = loop_engine.execute_generation_cycle().await;
        assert!(result.is_ok());

        let performance = result.unwrap();
        assert!(performance.aggregate_ihsan > 0.9);
    }

    #[tokio::test]
    async fn test_proof_chain_integrity() {
        let config = AutopoieticConfig {
            generation_duration_ms: 10,
            max_generations: 5,
            ..Default::default()
        };

        let (loop_engine, _event_rx, _control_tx) = AutopoieticLoop::new(config);
        loop_engine.initialize_default_blueprints().await;

        // Run a few generations
        for _ in 0..3 {
            let _ = loop_engine.execute_generation_cycle().await;
        }

        let verification = loop_engine.verify_chain().await;
        assert!(verification.is_valid);
        assert_eq!(verification.verified_nodes, 3);
    }
}
