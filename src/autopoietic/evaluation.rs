// src/autopoietic/evaluation.rs - Operation Monitoring and Evaluation
//
// Implements Step 4 of the 11-step cycle:
// - Operational metrics collection
// - Environment assessment
// - Economic evaluation
// - Ethical scoring via Ihsān 8-dimension + SAPE 9-probe

use crate::autopoietic::types::{
    GenerationPerformance, IhsanDimensions, KEPProgress, ProbeResult, SAPEResults,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Operation monitor for collecting metrics during generation
pub struct OperationMonitor {
    /// Start time of current monitoring period
    start_time: RwLock<Option<DateTime<Utc>>>,

    /// Task execution counter
    tasks_executed: AtomicU64,

    /// Successful executions
    successful_executions: AtomicU64,

    /// Failed executions (rejections)
    rejections: AtomicU64,

    /// Latency samples (ms)
    latency_samples: RwLock<Vec<u64>>,

    /// Per-agent metrics
    agent_metrics: RwLock<HashMap<String, AgentMetrics>>,

    /// Ihsān dimension accumulators
    ihsan_accumulators: RwLock<IhsanAccumulators>,

    /// SAPE results accumulators
    sape_accumulators: RwLock<SAPEAccumulators>,

    /// Knowledge metrics for KEP
    knowledge_metrics: RwLock<KnowledgeMetrics>,
}

impl OperationMonitor {
    pub fn new() -> Self {
        Self {
            start_time: RwLock::new(None),
            tasks_executed: AtomicU64::new(0),
            successful_executions: AtomicU64::new(0),
            rejections: AtomicU64::new(0),
            latency_samples: RwLock::new(Vec::new()),
            agent_metrics: RwLock::new(HashMap::new()),
            ihsan_accumulators: RwLock::new(IhsanAccumulators::default()),
            sape_accumulators: RwLock::new(SAPEAccumulators::default()),
            knowledge_metrics: RwLock::new(KnowledgeMetrics::default()),
        }
    }

    /// Start a new monitoring period
    pub async fn start_period(&self) {
        let mut start = self.start_time.write().await;
        *start = Some(Utc::now());

        // Reset counters
        self.tasks_executed.store(0, Ordering::SeqCst);
        self.successful_executions.store(0, Ordering::SeqCst);
        self.rejections.store(0, Ordering::SeqCst);

        // Clear samples
        self.latency_samples.write().await.clear();
        self.agent_metrics.write().await.clear();
        *self.ihsan_accumulators.write().await = IhsanAccumulators::default();
        *self.sape_accumulators.write().await = SAPEAccumulators::default();

        info!("📊 Operation monitoring period started");
    }

    /// Record a task execution
    pub async fn record_execution(&self, result: ExecutionRecord) {
        self.tasks_executed.fetch_add(1, Ordering::SeqCst);

        if result.success {
            self.successful_executions.fetch_add(1, Ordering::SeqCst);
        } else {
            self.rejections.fetch_add(1, Ordering::SeqCst);
        }

        // Record latency
        self.latency_samples.write().await.push(result.latency_ms);

        // Record per-agent metrics
        let mut agents = self.agent_metrics.write().await;
        let metrics = agents.entry(result.agent_id.clone()).or_default();
        metrics.record(&result);

        // Accumulate Ihsān dimensions
        if let Some(dims) = &result.ihsan_dimensions {
            self.ihsan_accumulators.write().await.accumulate(dims);
        }

        // Accumulate SAPE results
        if let Some(sape) = &result.sape_results {
            self.sape_accumulators.write().await.accumulate(sape);
        }

        debug!(
            agent = %result.agent_id,
            success = result.success,
            latency_ms = result.latency_ms,
            "Recorded execution"
        );
    }

    /// Record a knowledge discovery (for KEP)
    pub async fn record_knowledge_discovery(&self, discovery: KnowledgeDiscovery) {
        let mut metrics = self.knowledge_metrics.write().await;
        metrics.record(discovery);
    }

    /// End monitoring period and collect results
    pub async fn end_period(&self, generation: u64) -> EvaluationResult {
        let start_time = self.start_time.read().await.unwrap_or(Utc::now());
        let end_time = Utc::now();
        let duration_ms = (end_time - start_time).num_milliseconds() as u64;

        // Collect operational metrics
        let operational = self.collect_operational_metrics().await;

        // Collect environment metrics
        let environment = self.collect_environment_metrics().await;

        // Collect economic metrics
        let economic = self.collect_economic_metrics().await;

        // Collect ethical metrics
        let ethical = self.collect_ethical_metrics().await;

        // Collect KEP progress
        let kep_progress = self.collect_kep_progress().await;

        info!(
            generation = generation,
            tasks = operational.tasks_processed,
            ihsan = format!("{:.4}", ethical.aggregate_ihsan),
            "📊 Operation monitoring period ended"
        );

        EvaluationResult {
            generation,
            start_time,
            end_time,
            duration_ms,
            operational,
            environment,
            economic,
            ethical,
            kep_progress,
        }
    }

    async fn collect_operational_metrics(&self) -> OperationalMetrics {
        let latencies = self.latency_samples.read().await;
        let mut sorted_latencies = latencies.clone();
        sorted_latencies.sort_unstable();

        let avg_latency = if latencies.is_empty() {
            0
        } else {
            latencies.iter().sum::<u64>() / latencies.len() as u64
        };

        let p95_latency = if sorted_latencies.is_empty() {
            0
        } else {
            let idx = (sorted_latencies.len() as f64 * 0.95) as usize;
            sorted_latencies[idx.min(sorted_latencies.len() - 1)]
        };

        let p99_latency = if sorted_latencies.is_empty() {
            0
        } else {
            let idx = (sorted_latencies.len() as f64 * 0.99) as usize;
            sorted_latencies[idx.min(sorted_latencies.len() - 1)]
        };

        let agents = self.agent_metrics.read().await;
        let agent_utilization: HashMap<String, f64> = agents
            .iter()
            .map(|(id, m)| (id.clone(), m.utilization()))
            .collect();

        OperationalMetrics {
            tasks_processed: self.tasks_executed.load(Ordering::SeqCst),
            successful_executions: self.successful_executions.load(Ordering::SeqCst),
            rejections: self.rejections.load(Ordering::SeqCst),
            avg_latency_ms: avg_latency,
            p95_latency_ms: p95_latency,
            p99_latency_ms: p99_latency,
            agent_utilization,
            throughput_per_second: 0.0, // Calculated from duration
        }
    }

    async fn collect_environment_metrics(&self) -> EnvironmentMetrics {
        // These would typically come from system monitoring
        EnvironmentMetrics {
            cpu_usage_percent: 0.0,
            memory_usage_percent: 0.0,
            gpu_utilization_percent: 0.0,
            network_latency_ms: 0,
            active_connections: 0,
            queue_depth: 0,
        }
    }

    async fn collect_economic_metrics(&self) -> EconomicMetrics {
        // These would come from token/resource tracking
        EconomicMetrics {
            tokens_consumed: 0,
            tokens_generated: 0,
            compute_units_used: 0.0,
            cost_estimate_usd: 0.0,
            value_generated: 0.0,
            efficiency_ratio: 0.0,
        }
    }

    async fn collect_ethical_metrics(&self) -> EthicalMetrics {
        let ihsan_acc = self.ihsan_accumulators.read().await;
        let sape_acc = self.sape_accumulators.read().await;

        let ihsan_dimensions = ihsan_acc.aggregate();
        let aggregate_ihsan = ihsan_dimensions.aggregate();
        let sape_results = sape_acc.aggregate();

        EthicalMetrics {
            aggregate_ihsan,
            ihsan_dimensions,
            sape_results,
            ihsan_gate_passed: aggregate_ihsan >= 0.95,
            sape_all_passed: sape_acc.all_passed(),
            fate_escalations: 0,
            human_reviews_required: 0,
        }
    }

    async fn collect_kep_progress(&self) -> KEPProgress {
        let km = self.knowledge_metrics.read().await;

        KEPProgress {
            knowledge_mass: km.total_elements,
            discovery_velocity: km.calculate_velocity(),
            synergy_density: km.synergy_density(),
            learning_rate_multiplier: 1.0, // Will be adjusted by convergence module
            synergies_detected: km.synergies_detected,
            compounds_synthesized: km.compounds_synthesized,
            explosion_duration_seconds: 0,
        }
    }
}

impl Default for OperationMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Record of a single execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRecord {
    /// Agent that performed the execution
    pub agent_id: String,

    /// Whether execution was successful
    pub success: bool,

    /// Latency in milliseconds
    pub latency_ms: u64,

    /// Ihsān dimension scores (if available)
    pub ihsan_dimensions: Option<IhsanDimensions>,

    /// SAPE results (if available)
    pub sape_results: Option<SAPEResults>,

    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Per-agent metrics accumulator
#[derive(Debug, Clone, Default)]
pub struct AgentMetrics {
    pub tasks_executed: u64,
    pub successful: u64,
    pub failed: u64,
    pub total_latency_ms: u64,
    pub active_time_ms: u64,
}

impl AgentMetrics {
    pub fn record(&mut self, record: &ExecutionRecord) {
        self.tasks_executed += 1;
        if record.success {
            self.successful += 1;
        } else {
            self.failed += 1;
        }
        self.total_latency_ms += record.latency_ms;
        self.active_time_ms += record.latency_ms;
    }

    pub fn utilization(&self) -> f64 {
        if self.tasks_executed == 0 {
            return 0.0;
        }
        self.successful as f64 / self.tasks_executed as f64
    }
}

/// Accumulator for Ihsān dimensions
#[derive(Debug, Clone, Default)]
struct IhsanAccumulators {
    count: u64,
    correctness: f64,
    safety: f64,
    user_benefit: f64,
    efficiency: f64,
    auditability: f64,
    anti_centralization: f64,
    robustness: f64,
    adl_fairness: f64,
}

impl IhsanAccumulators {
    fn accumulate(&mut self, dims: &IhsanDimensions) {
        self.count += 1;
        self.correctness += dims.correctness;
        self.safety += dims.safety;
        self.user_benefit += dims.user_benefit;
        self.efficiency += dims.efficiency;
        self.auditability += dims.auditability;
        self.anti_centralization += dims.anti_centralization;
        self.robustness += dims.robustness;
        self.adl_fairness += dims.adl_fairness;
    }

    fn aggregate(&self) -> IhsanDimensions {
        if self.count == 0 {
            return IhsanDimensions::default();
        }
        let n = self.count as f64;
        IhsanDimensions {
            correctness: self.correctness / n,
            safety: self.safety / n,
            user_benefit: self.user_benefit / n,
            efficiency: self.efficiency / n,
            auditability: self.auditability / n,
            anti_centralization: self.anti_centralization / n,
            robustness: self.robustness / n,
            adl_fairness: self.adl_fairness / n,
        }
    }
}

/// Accumulator for SAPE probe results
#[derive(Debug, Clone, Default)]
struct SAPEAccumulators {
    count: u64,
    threat_scan: (u64, f64), // (passed_count, score_sum)
    compliance: (u64, f64),
    bias: (u64, f64),
    user_benefit: (u64, f64),
    correctness: (u64, f64),
    safety: (u64, f64),
    groundedness: (u64, f64),
    relevance: (u64, f64),
    fluency: (u64, f64),
}

impl SAPEAccumulators {
    fn accumulate(&mut self, results: &SAPEResults) {
        self.count += 1;
        Self::accumulate_probe_static(&results.threat_scan, &mut self.threat_scan);
        Self::accumulate_probe_static(&results.compliance, &mut self.compliance);
        Self::accumulate_probe_static(&results.bias, &mut self.bias);
        Self::accumulate_probe_static(&results.user_benefit, &mut self.user_benefit);
        Self::accumulate_probe_static(&results.correctness, &mut self.correctness);
        Self::accumulate_probe_static(&results.safety, &mut self.safety);
        Self::accumulate_probe_static(&results.groundedness, &mut self.groundedness);
        Self::accumulate_probe_static(&results.relevance, &mut self.relevance);
        Self::accumulate_probe_static(&results.fluency, &mut self.fluency);
    }

    fn accumulate_probe_static(probe: &ProbeResult, acc: &mut (u64, f64)) {
        if probe.passed {
            acc.0 += 1;
        }
        acc.1 += probe.score;
    }

    fn aggregate(&self) -> SAPEResults {
        if self.count == 0 {
            return SAPEResults::default();
        }
        SAPEResults {
            threat_scan: self.aggregate_probe(&self.threat_scan),
            compliance: self.aggregate_probe(&self.compliance),
            bias: self.aggregate_probe(&self.bias),
            user_benefit: self.aggregate_probe(&self.user_benefit),
            correctness: self.aggregate_probe(&self.correctness),
            safety: self.aggregate_probe(&self.safety),
            groundedness: self.aggregate_probe(&self.groundedness),
            relevance: self.aggregate_probe(&self.relevance),
            fluency: self.aggregate_probe(&self.fluency),
        }
    }

    fn aggregate_probe(&self, acc: &(u64, f64)) -> ProbeResult {
        if self.count == 0 {
            return ProbeResult::default();
        }
        ProbeResult {
            passed: acc.0 as f64 / self.count as f64 >= 0.5, // Majority passed
            score: acc.1 / self.count as f64,
            evidence: Vec::new(),
        }
    }

    fn all_passed(&self) -> bool {
        if self.count == 0 {
            return false;
        }
        let threshold = self.count as f64 * 0.5;
        self.threat_scan.0 as f64 >= threshold
            && self.compliance.0 as f64 >= threshold
            && self.bias.0 as f64 >= threshold
            && self.user_benefit.0 as f64 >= threshold
            && self.correctness.0 as f64 >= threshold
            && self.safety.0 as f64 >= threshold
            && self.groundedness.0 as f64 >= threshold
            && self.relevance.0 as f64 >= threshold
            && self.fluency.0 as f64 >= threshold
    }
}

/// Knowledge discovery record for KEP
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeDiscovery {
    pub discovery_type: KnowledgeDiscoveryType,
    pub element_id: String,
    pub related_elements: Vec<String>,
    pub impact_score: f64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KnowledgeDiscoveryType {
    NewElement,
    Synergy,
    Compound,
    Pattern,
}

/// Knowledge metrics for KEP calculation
#[derive(Debug, Clone, Default)]
struct KnowledgeMetrics {
    total_elements: u64,
    synergies_detected: u64,
    compounds_synthesized: u64,
    elements_added_this_period: u64,
    first_discovery_time: Option<DateTime<Utc>>,
    last_discovery_time: Option<DateTime<Utc>>,
}

impl KnowledgeMetrics {
    fn record(&mut self, discovery: KnowledgeDiscovery) {
        if self.first_discovery_time.is_none() {
            self.first_discovery_time = Some(discovery.timestamp);
        }
        self.last_discovery_time = Some(discovery.timestamp);

        match discovery.discovery_type {
            KnowledgeDiscoveryType::NewElement => {
                self.total_elements += 1;
                self.elements_added_this_period += 1;
            }
            KnowledgeDiscoveryType::Synergy => {
                self.synergies_detected += 1;
            }
            KnowledgeDiscoveryType::Compound => {
                self.compounds_synthesized += 1;
            }
            KnowledgeDiscoveryType::Pattern => {
                // Patterns contribute to knowledge mass
                self.total_elements += 1;
            }
        }
    }

    fn calculate_velocity(&self) -> f64 {
        if self.compounds_synthesized == 0 {
            return 0.0;
        }

        match (self.first_discovery_time, self.last_discovery_time) {
            (Some(first), Some(last)) => {
                let duration_hours = (last - first).num_seconds() as f64 / 3600.0;
                if duration_hours > 0.0 {
                    self.compounds_synthesized as f64 / duration_hours
                } else {
                    self.compounds_synthesized as f64
                }
            }
            _ => 0.0,
        }
    }

    fn synergy_density(&self) -> f64 {
        if self.total_elements == 0 {
            return 0.0;
        }
        // Synergy density = synergies / (elements * (elements - 1) / 2)
        // Simplified: synergies / elements
        self.synergies_detected as f64 / self.total_elements as f64
    }
}

/// Complete evaluation result for a generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    /// Generation number
    pub generation: u64,

    /// Start time of evaluation period
    pub start_time: DateTime<Utc>,

    /// End time of evaluation period
    pub end_time: DateTime<Utc>,

    /// Duration in milliseconds
    pub duration_ms: u64,

    /// Operational metrics
    pub operational: OperationalMetrics,

    /// Environment metrics
    pub environment: EnvironmentMetrics,

    /// Economic metrics
    pub economic: EconomicMetrics,

    /// Ethical metrics
    pub ethical: EthicalMetrics,

    /// KEP progress
    pub kep_progress: KEPProgress,
}

impl EvaluationResult {
    /// Check if this generation passes the Ihsān hard gate
    pub fn passes_ihsan_gate(&self) -> bool {
        self.ethical.ihsan_gate_passed
    }

    /// Check if all critical checks pass
    pub fn all_checks_passed(&self) -> bool {
        self.ethical.ihsan_gate_passed && self.ethical.sape_all_passed
    }

    /// Convert to GenerationPerformance for storage
    pub fn to_generation_performance(
        &self,
        proof_hash: &str,
        receipt_id: &str,
    ) -> GenerationPerformance {
        GenerationPerformance {
            generation: self.generation,
            started_at: self.start_time,
            ended_at: self.end_time,
            duration_ms: self.duration_ms,
            aggregate_ihsan: self.ethical.aggregate_ihsan,
            ihsan_dimensions: self.ethical.ihsan_dimensions.clone(),
            sape_results: self.ethical.sape_results.clone(),
            tasks_processed: self.operational.tasks_processed,
            successful_executions: self.operational.successful_executions,
            rejections: self.operational.rejections,
            avg_latency_ms: self.operational.avg_latency_ms,
            p95_latency_ms: self.operational.p95_latency_ms,
            kep_progress: self.kep_progress.clone(),
            improvements_applied: Vec::new(),
            proof_hash: proof_hash.to_string(),
            receipt_id: receipt_id.to_string(),
        }
    }
}

/// Operational metrics for a generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationalMetrics {
    /// Total tasks processed
    pub tasks_processed: u64,

    /// Successful executions
    pub successful_executions: u64,

    /// Rejections
    pub rejections: u64,

    /// Average latency
    pub avg_latency_ms: u64,

    /// P95 latency
    pub p95_latency_ms: u64,

    /// P99 latency
    pub p99_latency_ms: u64,

    /// Per-agent utilization
    pub agent_utilization: HashMap<String, f64>,

    /// Throughput
    pub throughput_per_second: f64,
}

/// Environment metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentMetrics {
    pub cpu_usage_percent: f64,
    pub memory_usage_percent: f64,
    pub gpu_utilization_percent: f64,
    pub network_latency_ms: u64,
    pub active_connections: u64,
    pub queue_depth: u64,
}

/// Economic metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EconomicMetrics {
    pub tokens_consumed: u64,
    pub tokens_generated: u64,
    pub compute_units_used: f64,
    pub cost_estimate_usd: f64,
    pub value_generated: f64,
    pub efficiency_ratio: f64,
}

/// Ethical metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EthicalMetrics {
    pub aggregate_ihsan: f64,
    pub ihsan_dimensions: IhsanDimensions,
    pub sape_results: SAPEResults,
    pub ihsan_gate_passed: bool,
    pub sape_all_passed: bool,
    pub fate_escalations: u64,
    pub human_reviews_required: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_operation_monitor() {
        let monitor = OperationMonitor::new();
        monitor.start_period().await;

        // Record some executions
        for i in 0..10 {
            let record = ExecutionRecord {
                agent_id: format!("agent-{}", i % 3),
                success: i % 4 != 0, // 75% success rate
                latency_ms: 100 + (i * 10),
                ihsan_dimensions: Some(IhsanDimensions {
                    correctness: 0.95,
                    safety: 0.96,
                    user_benefit: 0.94,
                    efficiency: 0.92,
                    auditability: 0.93,
                    anti_centralization: 0.90,
                    robustness: 0.91,
                    adl_fairness: 0.89,
                }),
                sape_results: None,
                timestamp: Utc::now(),
            };
            monitor.record_execution(record).await;
        }

        let result = monitor.end_period(1).await;

        assert_eq!(result.operational.tasks_processed, 10);
        assert!(result.ethical.aggregate_ihsan > 0.9);
    }

    #[test]
    fn test_ihsan_dimensions_aggregate() {
        let dims = IhsanDimensions {
            correctness: 1.0,
            safety: 1.0,
            user_benefit: 1.0,
            efficiency: 1.0,
            auditability: 1.0,
            anti_centralization: 1.0,
            robustness: 1.0,
            adl_fairness: 1.0,
        };

        let aggregate = dims.aggregate();
        assert!((aggregate - 1.0).abs() < 0.001);
    }
}
