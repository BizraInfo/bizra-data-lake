//! Sovereign Orchestrator — The Unified Intelligence Coordinator
//!
//! Coordinates all BIZRA subsystems into a coherent whole,
//! implementing the peak integration pattern.

use std::sync::Arc;
use tokio::sync::RwLock;

use crate::{NodeIdentity, Constitution, IHSAN_THRESHOLD, SNR_THRESHOLD};
use crate::pci::gates::GateContext;
use super::graph_of_thoughts::{ThoughtGraph, ThoughtNode, ReasoningPath};
use super::snr_engine::{SNREngine, SignalMetrics};

/// Orchestrator configuration
#[derive(Clone, Debug)]
pub struct OrchestratorConfig {
    /// Maximum concurrent operations
    pub max_concurrency: usize,
    /// Enable Graph-of-Thoughts reasoning
    pub enable_got: bool,
    /// Enable autonomous SNR optimization
    pub enable_snr_auto: bool,
    /// Minimum acceptable SNR
    pub snr_floor: f64,
    /// Target Ihsān score
    pub ihsan_target: f64,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            max_concurrency: 100,
            enable_got: true,
            enable_snr_auto: true,
            snr_floor: SNR_THRESHOLD,
            ihsan_target: IHSAN_THRESHOLD,
        }
    }
}

/// Operation result from the orchestrator
#[derive(Debug)]
pub struct OrchestratorResult<T> {
    /// The actual result
    pub value: T,
    /// SNR metrics for this operation
    pub snr: SignalMetrics,
    /// Reasoning path taken (if GoT enabled)
    pub reasoning: Option<ReasoningPath>,
    /// Latency in microseconds
    pub latency_us: u64,
}

/// The Sovereign Orchestrator — Peak integration layer
///
/// Unifies identity, protocol, inference, and federation into
/// a single coherent system with Graph-of-Thoughts reasoning.
pub struct SovereignOrchestrator {
    config: OrchestratorConfig,
    identity: Arc<RwLock<Option<NodeIdentity>>>,
    constitution: Constitution,
    snr_engine: SNREngine,
    thought_graph: ThoughtGraph,
    operation_count: std::sync::atomic::AtomicU64,
}

impl SovereignOrchestrator {
    /// Create new orchestrator with configuration
    pub fn new(config: OrchestratorConfig) -> Self {
        Self {
            config: config.clone(),
            identity: Arc::new(RwLock::new(None)),
            constitution: Constitution::default(),
            snr_engine: SNREngine::new(config.snr_floor, config.ihsan_target),
            thought_graph: ThoughtGraph::new(),
            operation_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Initialize with identity
    pub async fn with_identity(self, identity: NodeIdentity) -> Self {
        *self.identity.write().await = Some(identity);
        self
    }

    /// Execute an operation with full orchestration
    ///
    /// This is the primary entry point for all sovereign operations.
    /// It applies Graph-of-Thoughts reasoning and SNR optimization.
    pub async fn execute<F, T>(&self, operation: F) -> OrchestratorResult<T>
    where
        F: FnOnce() -> T,
    {
        let start = std::time::Instant::now();

        // Increment operation counter
        self.operation_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Build reasoning path if GoT enabled
        let reasoning = if self.config.enable_got {
            let path = self.thought_graph.create_path("execute_operation");
            Some(path)
        } else {
            None
        };

        // Execute the operation
        let value = operation();

        // Calculate SNR metrics
        let snr = if self.config.enable_snr_auto {
            self.snr_engine.measure_operation()
        } else {
            SignalMetrics::default()
        };

        let latency_us = start.elapsed().as_micros() as u64;

        OrchestratorResult {
            value,
            snr,
            reasoning,
            latency_us,
        }
    }

    /// Validate content through the full gate chain with GoT reasoning
    pub fn validate_with_reasoning(&self, ctx: &GateContext) -> (bool, ReasoningPath) {
        let mut path = self.thought_graph.create_path("validate_content");

        // Schema validation thought
        path.add_thought(ThoughtNode::new(
            "schema_check",
            "Validate JSON structure",
        ));

        let schema_valid = serde_json::from_slice::<serde_json::Value>(&ctx.content).is_ok();
        path.record_result("schema_check", schema_valid);

        if !schema_valid {
            return (false, path);
        }

        // SNR validation thought
        path.add_thought(ThoughtNode::new(
            "snr_check",
            "Verify signal-to-noise ratio",
        ));

        let snr_valid = ctx.snr_score
            .map(|s| self.constitution.check_snr(s))
            .unwrap_or(true);
        path.record_result("snr_check", snr_valid);

        if !snr_valid {
            return (false, path);
        }

        // Ihsān validation thought
        path.add_thought(ThoughtNode::new(
            "ihsan_check",
            "Verify excellence threshold",
        ));

        let ihsan_valid = ctx.ihsan_score
            .map(|i| self.constitution.check_ihsan(i))
            .unwrap_or(true);
        path.record_result("ihsan_check", ihsan_valid);

        (ihsan_valid, path)
    }

    /// Get current operation statistics
    pub fn stats(&self) -> OrchestratorStats {
        OrchestratorStats {
            total_operations: self.operation_count.load(std::sync::atomic::Ordering::Relaxed),
            config: self.config.clone(),
            snr_floor: self.config.snr_floor,
            ihsan_target: self.config.ihsan_target,
        }
    }
}

/// Orchestrator statistics
#[derive(Debug)]
pub struct OrchestratorStats {
    pub total_operations: u64,
    pub config: OrchestratorConfig,
    pub snr_floor: f64,
    pub ihsan_target: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_orchestrator_execute() {
        let orch = SovereignOrchestrator::new(OrchestratorConfig::default());

        let result = orch.execute(|| {
            2 + 2
        }).await;

        assert_eq!(result.value, 4);
        assert!(result.latency_us < 1_000_000); // < 1 second
    }

    #[test]
    fn test_validation_with_reasoning() {
        let orch = SovereignOrchestrator::new(OrchestratorConfig::default());
        let constitution = Constitution::default();

        let ctx = GateContext {
            sender_id: "test".into(),
            envelope_id: "pci_test".into(),
            content: br#"{"valid": "json"}"#.to_vec(),
            constitution,
            snr_score: Some(0.90),
            ihsan_score: Some(0.96),
        };

        let (valid, path) = orch.validate_with_reasoning(&ctx);
        assert!(valid);
        assert!(!path.thoughts.is_empty());
    }
}
