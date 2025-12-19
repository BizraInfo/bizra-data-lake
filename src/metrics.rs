// src/metrics.rs - Glass Cockpit Prometheus Metrics Exporter
// Provides real-time observability into BIZRA's PAT↔SAT↔FATE flow

use lazy_static::lazy_static;
use prometheus::{
    register_counter_vec, register_gauge, register_gauge_vec, register_histogram,
    register_histogram_vec, CounterVec, Encoder, Gauge, GaugeVec, Histogram, HistogramVec,
    TextEncoder,
};
use std::time::Instant;
use tracing::info;

lazy_static! {
    // ============================================================
    // SAT Metrics - Validation layer observability
    // ============================================================
    
    /// Total SAT validation requests
    pub static ref SAT_REQUESTS_TOTAL: CounterVec = register_counter_vec!(
        "bizra_sat_requests_total",
        "Total number of SAT validation requests",
        &["result"]  // approved, rejected, quarantine
    ).unwrap();

    /// SAT rejections by rejection code
    pub static ref SAT_REJECTIONS_BY_CODE: CounterVec = register_counter_vec!(
        "bizra_sat_rejections_total",
        "Total SAT rejections by rejection code",
        &["code"]  // security_threat, ethics_violation, performance_exceeded, etc.
    ).unwrap();

    /// SAT validation latency histogram
    pub static ref SAT_VALIDATION_LATENCY: Histogram = register_histogram!(
        "bizra_sat_validation_seconds",
        "SAT validation latency in seconds",
        vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
    ).unwrap();

    /// SAT consensus approvals (3/5 required)
    pub static ref SAT_CONSENSUS_APPROVALS: GaugeVec = register_gauge_vec!(
        "bizra_sat_consensus_approvals",
        "Number of SAT validators that approved (last request)",
        &["request_id"]
    ).unwrap();

    // ============================================================
    // FATE Metrics - Escalation layer observability
    // ============================================================

    /// FATE escalations by level
    pub static ref FATE_ESCALATIONS_TOTAL: CounterVec = register_counter_vec!(
        "bizra_fate_escalations_total",
        "Total FATE escalations by severity level",
        &["level"]  // low, medium, high, critical
    ).unwrap();

    /// FATE pending escalations (queue depth)
    pub static ref FATE_PENDING_ESCALATIONS: Gauge = register_gauge!(
        "bizra_fate_pending_escalations",
        "Number of FATE escalations pending human review"
    ).unwrap();

    // ============================================================
    // Ihsān Metrics - Ethics/Quality gate observability
    // ============================================================

    /// Ihsān score histogram
    pub static ref IHSAN_SCORE_HISTOGRAM: Histogram = register_histogram!(
        "bizra_ihsan_score",
        "Ihsān score distribution",
        vec![0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.95, 0.98, 1.0]
    ).unwrap();

    /// Ihsān dimension scores (last request)
    pub static ref IHSAN_DIMENSION_SCORES: GaugeVec = register_gauge_vec!(
        "bizra_ihsan_dimension_score",
        "Ihsān score by dimension (last request)",
        &["dimension"]  // correctness, safety, user_benefit, etc.
    ).unwrap();

    /// Ihsān gate pass/fail
    pub static ref IHSAN_GATE_RESULTS: CounterVec = register_counter_vec!(
        "bizra_ihsan_gate_total",
        "Ihsān gate results",
        &["result", "env"]  // passed/failed, dev/ci/prod
    ).unwrap();

    // ============================================================
    // Request Lifecycle Metrics
    // ============================================================

    /// End-to-end request latency
    pub static ref REQUEST_LATENCY: HistogramVec = register_histogram_vec!(
        "bizra_request_latency_seconds",
        "End-to-end request latency in seconds",
        &["outcome"],  // success, sat_rejected, ihsan_failed
        vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    ).unwrap();

    /// PAT execution latency
    pub static ref PAT_EXECUTION_LATENCY: Histogram = register_histogram!(
        "bizra_pat_execution_seconds",
        "PAT agent execution latency in seconds",
        vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
    ).unwrap();

    /// Synergy score histogram
    pub static ref SYNERGY_SCORE_HISTOGRAM: Histogram = register_histogram!(
        "bizra_synergy_score",
        "PAT-SAT synergy score distribution",
        vec![0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    ).unwrap();

    // ============================================================
    // Receipt Metrics
    // ============================================================

    /// Receipts emitted by type
    pub static ref RECEIPTS_EMITTED_TOTAL: CounterVec = register_counter_vec!(
        "bizra_receipts_emitted_total",
        "Total receipts emitted by type",
        &["type"]  // rejection, execution, quarantine
    ).unwrap();

    // ============================================================
    // Neo4j/HyperGraph Metrics
    // ============================================================

    /// Neo4j query latency
    pub static ref NEO4J_QUERY_LATENCY: HistogramVec = register_histogram_vec!(
        "bizra_neo4j_query_seconds",
        "Neo4j query latency in seconds",
        &["query_type"],  // evidence_retrieval, graph_traversal, etc.
        vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
    ).unwrap();

    /// Neo4j connection status (1 = connected, 0 = disconnected)
    pub static ref NEO4J_CONNECTED: Gauge = register_gauge!(
        "bizra_neo4j_connected",
        "Neo4j connection status (1=connected, 0=disconnected)"
    ).unwrap();

    // ============================================================
    // MCP/A2A Metrics
    // ============================================================

    /// MCP tool calls by tool name
    pub static ref MCP_TOOL_CALLS_TOTAL: CounterVec = register_counter_vec!(
        "bizra_mcp_tool_calls_total",
        "Total MCP tool calls by tool name",
        &["tool", "result"]  // tool name, success/failure/timeout
    ).unwrap();

    /// A2A delegations by agent
    pub static ref A2A_DELEGATIONS_TOTAL: CounterVec = register_counter_vec!(
        "bizra_a2a_delegations_total",
        "Total A2A delegations by agent",
        &["agent", "result"]  // agent name, success/failure/blocked
    ).unwrap();

    /// A2A delegation depth (current max)
    pub static ref A2A_DELEGATION_DEPTH: Gauge = register_gauge!(
        "bizra_a2a_delegation_depth_max",
        "Maximum A2A delegation depth observed"
    ).unwrap();
}

/// Timer guard for measuring operation duration
pub struct MetricsTimer {
    start: Instant,
    histogram: &'static Histogram,
}

impl MetricsTimer {
    pub fn new(histogram: &'static Histogram) -> Self {
        Self {
            start: Instant::now(),
            histogram,
        }
    }
}

impl Drop for MetricsTimer {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        self.histogram.observe(duration.as_secs_f64());
    }
}

/// Record SAT validation result
pub fn record_sat_validation(
    approved: bool,
    rejection_codes: &[String],
    latency_secs: f64,
    _approvals: usize,
) {
    let result = if approved { "approved" } else { "rejected" };
    SAT_REQUESTS_TOTAL.with_label_values(&[result]).inc();
    SAT_VALIDATION_LATENCY.observe(latency_secs);

    for code in rejection_codes {
        let code_label = code.split(':').next().unwrap_or("unknown").to_lowercase();
        SAT_REJECTIONS_BY_CODE
            .with_label_values(&[&code_label])
            .inc();
    }
}

/// Record FATE escalation
pub fn record_fate_escalation(level: &str, pending_count: usize) {
    FATE_ESCALATIONS_TOTAL
        .with_label_values(&[level.to_lowercase().as_str()])
        .inc();
    FATE_PENDING_ESCALATIONS.set(pending_count as f64);
}

/// Record Ihsān score and gate result
pub fn record_ihsan_result(
    score: f64,
    passed: bool,
    env: &str,
    dimension_scores: &std::collections::BTreeMap<String, f64>,
) {
    IHSAN_SCORE_HISTOGRAM.observe(score);

    let result = if passed { "passed" } else { "failed" };
    IHSAN_GATE_RESULTS.with_label_values(&[result, env]).inc();

    for (dimension, score) in dimension_scores {
        IHSAN_DIMENSION_SCORES
            .with_label_values(&[dimension])
            .set(*score);
    }
}

/// Record end-to-end request completion
pub fn record_request_completion(outcome: &str, latency_secs: f64, synergy: f64) {
    REQUEST_LATENCY
        .with_label_values(&[outcome])
        .observe(latency_secs);
    SYNERGY_SCORE_HISTOGRAM.observe(synergy);
}

/// Record receipt emission
pub fn record_receipt_emitted(receipt_type: &str) {
    RECEIPTS_EMITTED_TOTAL
        .with_label_values(&[receipt_type])
        .inc();
}

/// Record Neo4j query
pub fn record_neo4j_query(query_type: &str, latency_secs: f64, connected: bool) {
    NEO4J_QUERY_LATENCY
        .with_label_values(&[query_type])
        .observe(latency_secs);
    NEO4J_CONNECTED.set(if connected { 1.0 } else { 0.0 });
}

/// Generate Prometheus text format output
pub fn gather_metrics() -> String {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    String::from_utf8(buffer).unwrap()
}

/// Initialize metrics (call once at startup)
pub fn init_metrics() {
    info!("📊 Glass Cockpit metrics initialized");
    // Force lazy_static initialization
    let _ = SAT_REQUESTS_TOTAL.with_label_values(&["init"]);
    NEO4J_CONNECTED.set(0.0);
    FATE_PENDING_ESCALATIONS.set(0.0);
    A2A_DELEGATION_DEPTH.set(0.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sat_validation_metrics() {
        record_sat_validation(true, &[], 0.015, 5);
        record_sat_validation(false, &["SECURITY_THREAT: test".to_string()], 0.020, 2);

        let output = gather_metrics();
        assert!(output.contains("bizra_sat_requests_total"));
        assert!(output.contains("bizra_sat_validation_seconds"));
    }

    #[test]
    fn test_fate_escalation_metrics() {
        record_fate_escalation("critical", 3);

        let output = gather_metrics();
        assert!(output.contains("bizra_fate_escalations_total"));
        assert!(output.contains("bizra_fate_pending_escalations"));
    }

    #[test]
    fn test_ihsan_metrics() {
        let mut scores = std::collections::BTreeMap::new();
        scores.insert("correctness".to_string(), 0.95);
        scores.insert("safety".to_string(), 0.98);

        record_ihsan_result(0.93, true, "ci", &scores);

        let output = gather_metrics();
        assert!(output.contains("bizra_ihsan_score"));
        assert!(output.contains("bizra_ihsan_gate_total"));
    }

    #[test]
    fn test_metrics_endpoint_output() {
        init_metrics();
        let output = gather_metrics();

        // Verify it's valid Prometheus format
        assert!(output.contains("# HELP"));
        assert!(output.contains("# TYPE"));
    }
}
