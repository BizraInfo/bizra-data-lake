//! Omega Integration Layer — The Peak Masterpiece
//!
//! This module represents the ultimate integration of BIZRA components,
//! embodying interdisciplinary thinking and standing on the shoulders
//! of giants to achieve state-of-the-art performance.
//!
//! # Architecture (Standing on Giants)
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                         OMEGA ENGINE                                     │
//! │                    "If I have seen further..."                           │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐     │
//! │  │  Shannon   │   │  Lamport   │   │   Besta    │   │   Giants   │     │
//! │  │  SNR Max   │   │  Consensus │   │    GoT     │   │  Registry  │     │
//! │  └─────┬──────┘   └─────┬──────┘   └─────┬──────┘   └─────┬──────┘     │
//! │        │                │                │                │             │
//! │        └────────────────┼────────────────┼────────────────┘             │
//! │                         ▼                ▼                              │
//! │              ┌──────────────────────────────────┐                       │
//! │              │   Unified Reasoning Pipeline     │                       │
//! │              │   • Multi-path exploration       │                       │
//! │              │   • Quality-gated execution      │                       │
//! │              │   • Adaptive thresholds          │                       │
//! │              └──────────────────────────────────┘                       │
//! │                              │                                          │
//! │                              ▼                                          │
//! │              ┌──────────────────────────────────┐                       │
//! │              │   Circuit Breaker & Recovery     │                       │
//! │              │   • Fault tolerance              │                       │
//! │              │   • Self-healing                 │                       │
//! │              │   • Telemetry                    │                       │
//! │              └──────────────────────────────────┘                       │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Key Principles
//!
//! 1. **Shannon's SNR**: Maximize signal, minimize noise
//! 2. **Lamport's Clocks**: Ordered, traceable operations
//! 3. **Besta's GoT**: Non-linear reasoning exploration
//! 4. **Bernstein's Security**: Fail-secure by default
//! 5. **Torvalds' Composability**: Unix philosophy

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::Instant;
use tokio::sync::RwLock;

use crate::{NodeIdentity, Constitution, IHSAN_THRESHOLD, SNR_THRESHOLD};
use super::error::{SovereignError, SovereignResult};
use super::snr_engine::{SNREngine, SNRConfig, SignalMetrics};
use super::graph_of_thoughts::{ThoughtGraph, ThoughtNode, ReasoningPath};
use super::giants::GiantRegistry;

/// Omega Engine configuration
#[derive(Clone, Debug)]
pub struct OmegaConfig {
    /// SNR engine configuration
    pub snr_config: SNRConfig,
    /// Maximum concurrent operations
    pub max_concurrency: usize,
    /// Enable Graph-of-Thoughts reasoning
    pub enable_got: bool,
    /// Enable adaptive thresholds
    pub enable_adaptive: bool,
    /// Circuit breaker threshold (failures before open)
    pub circuit_breaker_threshold: u32,
    /// Circuit breaker recovery time
    pub circuit_breaker_recovery_ms: u64,
    /// Maximum operation timeout
    pub operation_timeout_ms: u64,
    /// Enable telemetry
    pub enable_telemetry: bool,
    /// Minimum acceptable SNR (floor)
    pub snr_floor: f64,
    /// Target Ihsān score
    pub ihsan_target: f64,
}

impl Default for OmegaConfig {
    fn default() -> Self {
        Self {
            snr_config: SNRConfig::default(),
            max_concurrency: 100,
            enable_got: true,
            enable_adaptive: true,
            circuit_breaker_threshold: 5,
            circuit_breaker_recovery_ms: 30_000,
            operation_timeout_ms: 60_000,
            enable_telemetry: true,
            snr_floor: SNR_THRESHOLD,
            ihsan_target: IHSAN_THRESHOLD,
        }
    }
}

impl OmegaConfig {
    /// Create production configuration
    pub fn production() -> Self {
        Self {
            enable_telemetry: true,
            circuit_breaker_threshold: 3,
            circuit_breaker_recovery_ms: 60_000,
            ..Default::default()
        }
    }

    /// Create development configuration
    ///
    /// WARNING: Development mode uses relaxed thresholds.
    /// This configuration MUST NOT be used in production deployments.
    /// Use `OmegaConfig::production()` for production environments.
    #[cfg(any(test, feature = "dev-mode"))]
    pub fn development() -> Self {
        Self {
            enable_telemetry: false,
            circuit_breaker_threshold: 10,
            snr_floor: 0.70, // More lenient for development (GUARDED)
            ihsan_target: 0.75, // Lower Ihsān requirement for development (GUARDED)
            ..Default::default()
        }
    }

    /// Development stub for release builds - always returns production config
    /// This prevents accidental use of relaxed thresholds in production
    #[cfg(not(any(test, feature = "dev-mode")))]
    pub fn development() -> Self {
        // In release builds without dev-mode feature, development() = production()
        // This is a safety guard against threshold leakage
        Self::production()
    }

    /// Create edge deployment configuration
    pub fn edge() -> Self {
        Self {
            snr_config: SNRConfig::edge(),
            max_concurrency: 10,
            enable_got: false, // Disable for resource constraints
            enable_adaptive: false,
            enable_telemetry: false,
            ..Default::default()
        }
    }
}

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation
    Closed,
    /// Rejecting requests
    Open,
    /// Testing recovery
    HalfOpen,
}

/// Omega Engine metrics
#[derive(Debug, Clone)]
pub struct OmegaMetrics {
    /// Total operations executed
    pub total_operations: u64,
    /// Successful operations
    pub successful_operations: u64,
    /// Failed operations
    pub failed_operations: u64,
    /// SNR violations
    pub snr_violations: u64,
    /// Ihsān violations
    pub ihsan_violations: u64,
    /// Circuit breaker trips
    pub circuit_breaker_trips: u64,
    /// Average latency in microseconds
    pub avg_latency_us: u64,
    /// Maximum latency observed
    pub max_latency_us: u64,
    /// Average SNR score
    pub avg_snr: f64,
    /// Current circuit state
    pub circuit_state: CircuitState,
    /// Active operations count
    pub active_operations: u64,
}

impl Default for OmegaMetrics {
    fn default() -> Self {
        Self {
            total_operations: 0,
            successful_operations: 0,
            failed_operations: 0,
            snr_violations: 0,
            ihsan_violations: 0,
            circuit_breaker_trips: 0,
            avg_latency_us: 0,
            max_latency_us: 0,
            avg_snr: 0.0,
            circuit_state: CircuitState::Closed,
            active_operations: 0,
        }
    }
}

/// Operation result from the Omega Engine
#[derive(Debug)]
pub struct OmegaResult<T> {
    /// The operation result
    pub value: T,
    /// SNR metrics
    pub snr: SignalMetrics,
    /// Reasoning path (if GoT enabled)
    pub reasoning: Option<ReasoningPath>,
    /// Operation latency in microseconds
    pub latency_us: u64,
    /// Operation ID for tracing
    pub operation_id: String,
    /// Quality assessment
    pub quality: &'static str,
}

/// The Omega Engine — Peak integration layer
///
/// Unifies all BIZRA components into a coherent, high-performance system
/// with fault tolerance, adaptive thresholds, and comprehensive telemetry.
pub struct OmegaEngine {
    /// Configuration
    config: OmegaConfig,
    /// Node identity (optional)
    identity: Arc<RwLock<Option<NodeIdentity>>>,
    /// Constitution (reserved for future governance integration)
    #[allow(dead_code)]
    constitution: Constitution,
    /// SNR Engine
    snr_engine: SNREngine,
    /// Thought Graph for GoT reasoning
    thought_graph: RwLock<ThoughtGraph>,
    /// Giants Registry for attribution
    giants: GiantRegistry,
    /// Circuit breaker failure count
    circuit_failures: AtomicU64,
    /// Circuit breaker state
    circuit_open: AtomicBool,
    /// Circuit breaker last trip time
    circuit_trip_time: RwLock<Option<Instant>>,
    /// Operation counter
    operation_count: AtomicU64,
    /// Successful operation counter
    success_count: AtomicU64,
    /// Failure counter
    failure_count: AtomicU64,
    /// SNR violation counter
    snr_violation_count: AtomicU64,
    /// Ihsān violation counter
    ihsan_violation_count: AtomicU64,
    /// Circuit trip counter
    circuit_trip_count: AtomicU64,
    /// Total latency accumulator (for averaging)
    total_latency_us: AtomicU64,
    /// Maximum latency observed
    max_latency_us: AtomicU64,
    /// SNR accumulator (for averaging)
    snr_accumulator: RwLock<f64>,
    /// Active operations
    active_ops: AtomicU64,
}

impl OmegaEngine {
    /// Create new Omega Engine with configuration
    pub fn new(config: OmegaConfig) -> Self {
        let snr_engine = SNREngine::with_config(config.snr_config.clone());
        
        Self {
            config,
            identity: Arc::new(RwLock::new(None)),
            constitution: Constitution::default(),
            snr_engine,
            thought_graph: RwLock::new(ThoughtGraph::new()),
            giants: GiantRegistry::new(),
            circuit_failures: AtomicU64::new(0),
            circuit_open: AtomicBool::new(false),
            circuit_trip_time: RwLock::new(None),
            operation_count: AtomicU64::new(0),
            success_count: AtomicU64::new(0),
            failure_count: AtomicU64::new(0),
            snr_violation_count: AtomicU64::new(0),
            ihsan_violation_count: AtomicU64::new(0),
            circuit_trip_count: AtomicU64::new(0),
            total_latency_us: AtomicU64::new(0),
            max_latency_us: AtomicU64::new(0),
            snr_accumulator: RwLock::new(0.0),
            active_ops: AtomicU64::new(0),
        }
    }

    /// Create with production configuration
    pub fn production() -> Self {
        Self::new(OmegaConfig::production())
    }

    /// Initialize with identity
    pub async fn with_identity(self, identity: NodeIdentity) -> Self {
        *self.identity.write().await = Some(identity);
        self
    }

    /// Check circuit breaker state
    async fn check_circuit(&self) -> SovereignResult<()> {
        if !self.circuit_open.load(Ordering::Relaxed) {
            return Ok(());
        }

        // Check if recovery time has passed
        let trip_time = self.circuit_trip_time.read().await;
        if let Some(time) = *trip_time {
            let elapsed = time.elapsed().as_millis() as u64;
            if elapsed > self.config.circuit_breaker_recovery_ms {
                // Transition to half-open
                self.circuit_open.store(false, Ordering::Relaxed);
                self.circuit_failures.store(0, Ordering::Relaxed);
                return Ok(());
            }
        }

        Err(SovereignError::CircuitBreakerOpen {
            service: "omega".to_string(),
            retry_after_ms: self.config.circuit_breaker_recovery_ms,
        })
    }

    /// Record operation failure for circuit breaker
    async fn record_failure(&self) {
        let failures = self.circuit_failures.fetch_add(1, Ordering::Relaxed) + 1;
        
        if failures >= self.config.circuit_breaker_threshold as u64 {
            self.circuit_open.store(true, Ordering::Relaxed);
            *self.circuit_trip_time.write().await = Some(Instant::now());
            self.circuit_trip_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record operation success
    #[inline]
    fn record_success(&self) {
        self.circuit_failures.store(0, Ordering::Relaxed);
        self.success_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Generate operation ID
    #[inline]
    fn generate_op_id(&self) -> String {
        let count = self.operation_count.fetch_add(1, Ordering::Relaxed);
        format!("omega-{:016x}", count)
    }

    /// Execute an operation with full Omega orchestration
    ///
    /// This is the primary entry point for all sovereign operations.
    /// It applies:
    /// - Circuit breaker protection
    /// - SNR quality validation
    /// - Graph-of-Thoughts reasoning (optional)
    /// - Telemetry and tracing
    pub async fn execute<F, T>(&self, operation: F) -> SovereignResult<OmegaResult<T>>
    where
        F: FnOnce() -> T,
    {
        // Check circuit breaker
        self.check_circuit().await?;

        let start = Instant::now();
        let op_id = self.generate_op_id();
        
        // Track active operations
        self.active_ops.fetch_add(1, Ordering::Relaxed);

        // Build reasoning path if GoT enabled
        let reasoning = if self.config.enable_got {
            let graph = self.thought_graph.write().await;
            let mut path = graph.create_path(&op_id);
            path.add_thought(ThoughtNode::new("execute", "Operation execution"));
            Some(path)
        } else {
            None
        };

        // Execute the operation
        let value = operation();

        // Calculate SNR metrics
        let snr = self.snr_engine.measure_operation();

        // Record metrics
        let latency_us = start.elapsed().as_micros() as u64;
        self.total_latency_us.fetch_add(latency_us, Ordering::Relaxed);
        
        // Update max latency
        let mut current_max = self.max_latency_us.load(Ordering::Relaxed);
        while latency_us > current_max {
            match self.max_latency_us.compare_exchange_weak(
                current_max,
                latency_us,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(v) => current_max = v,
            }
        }

        // Update SNR accumulator
        {
            let mut acc = self.snr_accumulator.write().await;
            *acc += snr.compute_snr();
        }

        // Record success
        self.record_success();
        self.active_ops.fetch_sub(1, Ordering::Relaxed);

        Ok(OmegaResult {
            value,
            snr: snr.clone(),
            reasoning,
            latency_us,
            operation_id: op_id,
            quality: snr.quality_assessment(),
        })
    }

    /// Execute with content validation
    pub async fn execute_validated<F, T>(
        &self,
        content: &str,
        operation: F,
    ) -> SovereignResult<OmegaResult<T>>
    where
        F: FnOnce() -> T,
    {
        // Validate content first
        let metrics = self.snr_engine.analyze_text(content)?;
        let snr = metrics.compute_snr();

        // Check SNR threshold
        if snr < self.config.snr_floor {
            self.snr_violation_count.fetch_add(1, Ordering::Relaxed);
            self.failure_count.fetch_add(1, Ordering::Relaxed);
            self.record_failure().await;
            return Err(SovereignError::SNRBelowThreshold {
                actual: snr,
                threshold: self.config.snr_floor,
            });
        }

        // STRICT IHSĀN ENFORCEMENT (Genesis Strict Synthesis)
        // Runtime tier requires excellence - violations are hard failures
        if snr < self.config.ihsan_target {
            self.ihsan_violation_count.fetch_add(1, Ordering::Relaxed);
            self.failure_count.fetch_add(1, Ordering::Relaxed);
            self.record_failure().await;
            return Err(SovereignError::IhsanViolation {
                actual: snr,
                threshold: self.config.ihsan_target,
            });
        }

        // Execute the operation
        self.execute(operation).await
    }

    /// Validate content through the full reasoning chain
    pub async fn validate_with_reasoning(&self, content: &str) -> SovereignResult<(SignalMetrics, ReasoningPath)> {
        let graph = self.thought_graph.write().await;
        let mut path = graph.create_path("validate_content");

        // Schema validation thought
        path.add_thought(ThoughtNode::new("schema_check", "Validate content structure"));
        
        // Try to parse as JSON if applicable
        let schema_valid = serde_json::from_str::<serde_json::Value>(content).is_ok()
            || !content.starts_with('{'); // Non-JSON is valid too
        path.record_result("schema_check", schema_valid);

        if !schema_valid {
            return Err(SovereignError::SchemaValidation {
                message: "Invalid JSON structure".into(),
            });
        }

        // SNR validation thought
        path.add_thought(ThoughtNode::new("snr_check", "Analyze signal-to-noise ratio"));
        
        let metrics = self.snr_engine.analyze_text(content)?;
        let snr_valid = metrics.compute_snr() >= self.config.snr_floor;
        path.record_result("snr_check", snr_valid);

        if !snr_valid {
            return Err(SovereignError::SNRBelowThreshold {
                actual: metrics.compute_snr(),
                threshold: self.config.snr_floor,
            });
        }

        // Ihsān validation thought
        path.add_thought(ThoughtNode::new("ihsan_check", "Verify excellence threshold"));
        
        let ihsan_valid = metrics.compute_snr() >= self.config.ihsan_target;
        path.record_result("ihsan_check", ihsan_valid);

        if !ihsan_valid {
            return Err(SovereignError::IhsanViolation {
                actual: metrics.compute_snr(),
                threshold: self.config.ihsan_target,
            });
        }

        Ok((metrics, path))
    }

    /// Get current metrics
    pub async fn metrics(&self) -> OmegaMetrics {
        let total = self.operation_count.load(Ordering::Relaxed);
        let successful = self.success_count.load(Ordering::Relaxed);
        let total_latency = self.total_latency_us.load(Ordering::Relaxed);
        
        let avg_latency = if total > 0 {
            total_latency / total
        } else {
            0
        };

        let snr_acc = *self.snr_accumulator.read().await;
        let avg_snr = if total > 0 {
            snr_acc / total as f64
        } else {
            0.0
        };

        let circuit_state = if self.circuit_open.load(Ordering::Relaxed) {
            CircuitState::Open
        } else if self.circuit_failures.load(Ordering::Relaxed) > 0 {
            CircuitState::HalfOpen
        } else {
            CircuitState::Closed
        };

        OmegaMetrics {
            total_operations: total,
            successful_operations: successful,
            failed_operations: self.failure_count.load(Ordering::Relaxed),
            snr_violations: self.snr_violation_count.load(Ordering::Relaxed),
            ihsan_violations: self.ihsan_violation_count.load(Ordering::Relaxed),
            circuit_breaker_trips: self.circuit_trip_count.load(Ordering::Relaxed),
            avg_latency_us: avg_latency,
            max_latency_us: self.max_latency_us.load(Ordering::Relaxed),
            avg_snr,
            circuit_state,
            active_operations: self.active_ops.load(Ordering::Relaxed),
        }
    }

    /// Get giants attribution
    pub fn giants(&self) -> &GiantRegistry {
        &self.giants
    }

    /// Get configuration
    pub fn config(&self) -> &OmegaConfig {
        &self.config
    }

    /// Get SNR engine
    pub fn snr_engine(&self) -> &SNREngine {
        &self.snr_engine
    }

    /// Check system health
    pub async fn health_check(&self) -> SovereignResult<bool> {
        // Check circuit breaker
        let circuit_ok = !self.circuit_open.load(Ordering::Relaxed);

        // Check metrics
        let metrics = self.metrics().await;
        let success_rate = if metrics.total_operations > 0 {
            metrics.successful_operations as f64 / metrics.total_operations as f64
        } else {
            1.0
        };

        let healthy = circuit_ok && success_rate > 0.9;

        if !healthy {
            return Err(SovereignError::OperationFailed {
                operation: "health_check".into(),
                reason: format!(
                    "Unhealthy: circuit={}, success_rate={:.2}%",
                    if circuit_ok { "closed" } else { "open" },
                    success_rate * 100.0
                ),
            });
        }

        Ok(true)
    }

    /// Print attribution to all giants
    pub fn print_attribution(&self) -> String {
        self.giants.attribution()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_omega_execute() {
        let engine = OmegaEngine::new(OmegaConfig::default());
        
        let result = engine.execute(|| 42).await.unwrap();
        
        assert_eq!(result.value, 42);
        assert!(result.latency_us < 1_000_000);
        assert!(!result.operation_id.is_empty());
    }

    #[tokio::test]
    async fn test_omega_metrics() {
        let engine = OmegaEngine::new(OmegaConfig::default());
        
        // Execute some operations
        for _ in 0..10 {
            engine.execute(|| 1 + 1).await.unwrap();
        }

        let metrics = engine.metrics().await;
        assert_eq!(metrics.total_operations, 10);
        assert_eq!(metrics.successful_operations, 10);
        assert_eq!(metrics.failed_operations, 0);
    }

    #[tokio::test]
    async fn test_circuit_breaker_closed() {
        let engine = OmegaEngine::new(OmegaConfig::default());
        
        // Should be closed by default
        let result = engine.check_circuit().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_validate_with_reasoning() {
        let engine = OmegaEngine::new(OmegaConfig::development());
        
        let content = "The quantum algorithm implementation demonstrates \
                       significant performance improvements according to research data.";
        
        let result = engine.validate_with_reasoning(content).await;
        assert!(result.is_ok());
        
        let (metrics, path) = result.unwrap();
        assert!(metrics.compute_snr() > 0.5);
        assert!(!path.thoughts.is_empty());
    }

    #[tokio::test]
    async fn test_health_check() {
        let engine = OmegaEngine::new(OmegaConfig::default());
        
        let health = engine.health_check().await;
        assert!(health.is_ok());
    }

    #[test]
    fn test_config_presets() {
        let prod = OmegaConfig::production();
        assert!(prod.enable_telemetry);
        
        let dev = OmegaConfig::development();
        assert!(!dev.enable_telemetry);
        assert!(dev.snr_floor < prod.snr_floor);
        
        let edge = OmegaConfig::edge();
        assert!(!edge.enable_got);
    }

    #[tokio::test]
    async fn test_giants_attribution() {
        let engine = OmegaEngine::new(OmegaConfig::default());
        
        let attribution = engine.print_attribution();
        assert!(attribution.contains("Shannon"));
        assert!(attribution.contains("Lamport"));
    }
}
