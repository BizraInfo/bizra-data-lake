// src/apex/circuit_breaker.rs - Fault Tolerance
//
// Implements the Circuit Breaker pattern for resilient agent execution:
// - Closed: Normal operation, requests flow through
// - Open: Failures exceeded threshold, requests fail fast
// - HalfOpen: Testing recovery, limited requests allowed
//
// Features:
// - Exponential backoff with configurable thresholds
// - Automatic recovery testing
// - Integration with model_router.rs fallback chains

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{Duration, Instant};
use tracing::{debug, info, instrument, warn};

use super::{ApexError, ApexResult};

/// Circuit breaker events for kernel integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CircuitBreakerEvent {
    Tripped { circuit_name: String, reason: String, trip_count: u64 },
    RecoveryStarted { circuit_name: String },
    Recovered { circuit_name: String, recovery_duration_ms: u64 },
    RecoveryFailed { circuit_name: String, reason: String },
}

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitState {
    /// Normal operation - requests flow through
    Closed,
    /// Circuit tripped - requests fail fast
    Open,
    /// Recovery testing - limited requests allowed
    HalfOpen,
}

impl std::fmt::Display for CircuitState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CircuitState::Closed => write!(f, "CLOSED"),
            CircuitState::Open => write!(f, "OPEN"),
            CircuitState::HalfOpen => write!(f, "HALF_OPEN"),
        }
    }
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Number of failures before opening circuit
    pub failure_threshold: u32,
    /// Number of successes in half-open before closing
    pub success_threshold: u32,
    /// Initial timeout before attempting recovery (ms)
    pub initial_timeout_ms: u64,
    /// Maximum timeout (ms)
    pub max_timeout_ms: u64,
    /// Timeout multiplier for exponential backoff
    pub backoff_multiplier: f64,
    /// Time window for counting failures (ms)
    pub failure_window_ms: u64,
    /// Maximum requests allowed in half-open state
    pub half_open_max_requests: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 3,
            initial_timeout_ms: 5_000, // 5 seconds
            max_timeout_ms: 60_000,    // 1 minute
            backoff_multiplier: 2.0,
            failure_window_ms: 60_000, // 1 minute window
            half_open_max_requests: 3,
        }
    }
}

/// Internal state for a single circuit
#[derive(Debug)]
struct CircuitInternalState {
    state: CircuitState,
    failures: Vec<Instant>,
    successes_in_half_open: u32,
    requests_in_half_open: u32,
    last_failure: Option<Instant>,
    last_state_change: Instant,
    current_timeout: Duration,
    trip_count: u64,
}

impl CircuitInternalState {
    fn new(config: &CircuitBreakerConfig) -> Self {
        Self {
            state: CircuitState::Closed,
            failures: Vec::new(),
            successes_in_half_open: 0,
            requests_in_half_open: 0,
            last_failure: None,
            last_state_change: Instant::now(),
            current_timeout: Duration::from_millis(config.initial_timeout_ms),
            trip_count: 0,
        }
    }
}

/// Circuit breaker for a single agent/service
pub struct CircuitBreaker {
    /// Agent/service identifier
    agent_id: String,
    /// Configuration
    config: CircuitBreakerConfig,
    /// Internal state
    state: RwLock<CircuitInternalState>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker for an agent
    pub fn new(agent_id: &str, config: CircuitBreakerConfig) -> Self {
        info!(
            agent = %agent_id,
            failure_threshold = config.failure_threshold,
            timeout_ms = config.initial_timeout_ms,
            "🔌 Circuit breaker initialized"
        );

        Self {
            agent_id: agent_id.to_string(),
            config: config.clone(),
            state: RwLock::new(CircuitInternalState::new(&config)),
        }
    }

    /// Create with default configuration
    pub fn with_defaults(agent_id: &str) -> Self {
        Self::new(agent_id, CircuitBreakerConfig::default())
    }

    /// Check if a request is allowed through the circuit
    #[instrument(skip(self))]
    pub fn allow_request(&self) -> ApexResult<(bool, Option<CircuitBreakerEvent>)> {
        let mut state = self.state.write().map_err(|e| {
            ApexError::Internal(anyhow::anyhow!(
                "Failed to acquire circuit breaker lock: {}",
                e
            ))
        })?;

        match state.state {
            CircuitState::Closed => Ok((true, None)),

            CircuitState::Open => {
                // Check if timeout has elapsed
                let elapsed = state.last_state_change.elapsed();
                if elapsed >= state.current_timeout {
                    // Transition to half-open
                    state.state = CircuitState::HalfOpen;
                    state.last_state_change = Instant::now();
                    state.successes_in_half_open = 0;
                    state.requests_in_half_open = 0;

                    info!(
                        agent = %self.agent_id,
                        "🔄 Circuit breaker transitioning to HALF_OPEN"
                    );

                    let event = CircuitBreakerEvent::RecoveryStarted {
                        circuit_name: self.agent_id.clone(),
                    };

                    Ok((true, Some(event)))
                } else {
                    // Still in timeout
                    debug!(
                        agent = %self.agent_id,
                        remaining_ms = (state.current_timeout - elapsed).as_millis(),
                        "Circuit OPEN - request blocked"
                    );
                    Ok((false, None))
                }
            }

            CircuitState::HalfOpen => {
                // Allow limited requests for recovery testing
                if state.requests_in_half_open < self.config.half_open_max_requests {
                    state.requests_in_half_open += 1;
                    Ok((true, None))
                } else {
                    debug!(
                        agent = %self.agent_id,
                        "Circuit HALF_OPEN - max test requests reached"
                    );
                    Ok((false, None))
                }
            }
        }
    }

    /// Record a successful execution
    #[instrument(skip(self))]
    pub fn record_success(&self) -> ApexResult<Option<CircuitBreakerEvent>> {
        let mut state = self.state.write().map_err(|e| {
            ApexError::Internal(anyhow::anyhow!(
                "Failed to acquire circuit breaker lock: {}",
                e
            ))
        })?;

        match state.state {
            CircuitState::Closed => {
                // Nothing special to do
                Ok(None)
            }

            CircuitState::HalfOpen => {
                state.successes_in_half_open += 1;

                if state.successes_in_half_open >= self.config.success_threshold {
                    // Close the circuit
                    let recovery_duration_ms = state.last_state_change.elapsed().as_millis() as u64;
                    state.state = CircuitState::Closed;
                    state.last_state_change = Instant::now();
                    state.current_timeout = Duration::from_millis(self.config.initial_timeout_ms);
                    state.failures.clear();

                    info!(
                        agent = %self.agent_id,
                        "✅ Circuit breaker CLOSED - recovery successful"
                    );

                    Ok(Some(CircuitBreakerEvent::Recovered {
                        circuit_name: self.agent_id.clone(),
                        recovery_duration_ms,
                    }))
                } else {
                    Ok(None)
                }
            }

            CircuitState::Open => {
                // Shouldn't happen, but log if it does
                warn!(
                    agent = %self.agent_id,
                    "Success recorded while circuit OPEN"
                );
                Ok(None)
            }
        }
    }

    /// Record a failed execution
    #[instrument(skip(self))]
    pub fn record_failure(&self) -> ApexResult<Option<CircuitBreakerEvent>> {
        let mut state = self.state.write().map_err(|e| {
            ApexError::Internal(anyhow::anyhow!(
                "Failed to acquire circuit breaker lock: {}",
                e
            ))
        })?;

        let now = Instant::now();
        state.last_failure = Some(now);

        match state.state {
            CircuitState::Closed => {
                // Add failure and clean old ones
                state.failures.push(now);
                let window = Duration::from_millis(self.config.failure_window_ms);
                state.failures.retain(|&t| now.duration_since(t) < window);

                // Check if threshold exceeded
                if state.failures.len() >= self.config.failure_threshold as usize {
                    let event = self.trip_circuit(&mut state);
                    Ok(Some(event))
                } else {
                    Ok(None)
                }
            }

            CircuitState::HalfOpen => {
                // Any failure in half-open trips the circuit again
                let reason = "Recovery failed - failure during half-open testing".to_string();
                let recovery_failed_event = CircuitBreakerEvent::RecoveryFailed {
                    circuit_name: self.agent_id.clone(),
                    reason,
                };

                self.trip_circuit(&mut state);

                // Increase timeout with exponential backoff
                let new_timeout_ms = (state.current_timeout.as_millis() as f64
                    * self.config.backoff_multiplier) as u64;
                state.current_timeout =
                    Duration::from_millis(new_timeout_ms.min(self.config.max_timeout_ms));

                warn!(
                    agent = %self.agent_id,
                    next_timeout_ms = state.current_timeout.as_millis(),
                    "Circuit re-tripped from HALF_OPEN with increased timeout"
                );

                Ok(Some(recovery_failed_event))
            }

            CircuitState::Open => {
                // Already open, just record the failure
                Ok(None)
            }
        }
    }

    /// Trip the circuit to open state
    fn trip_circuit(&self, state: &mut CircuitInternalState) -> CircuitBreakerEvent {
        state.state = CircuitState::Open;
        state.last_state_change = Instant::now();
        state.trip_count += 1;

        warn!(
            agent = %self.agent_id,
            trip_count = state.trip_count,
            timeout_ms = state.current_timeout.as_millis(),
            "🔴 Circuit breaker OPEN - agent isolated"
        );

        CircuitBreakerEvent::Tripped {
            circuit_name: self.agent_id.clone(),
            reason: format!("Failure threshold exceeded ({} failures)", self.config.failure_threshold),
            trip_count: state.trip_count,
        }
    }

    /// Get current circuit state
    pub fn get_state(&self) -> CircuitState {
        self.state
            .read()
            .map(|s| s.state)
            .unwrap_or(CircuitState::Closed)
    }

    /// Get circuit statistics
    pub fn get_stats(&self) -> CircuitStats {
        let state = self.state.read().ok();

        CircuitStats {
            agent_id: self.agent_id.clone(),
            state: state
                .as_ref()
                .map(|s| s.state)
                .unwrap_or(CircuitState::Closed),
            trip_count: state.as_ref().map(|s| s.trip_count).unwrap_or(0),
            failures_in_window: state.as_ref().map(|s| s.failures.len()).unwrap_or(0),
            current_timeout_ms: state
                .as_ref()
                .map(|s| s.current_timeout.as_millis() as u64)
                .unwrap_or(self.config.initial_timeout_ms),
            time_in_state_ms: state
                .as_ref()
                .map(|s| s.last_state_change.elapsed().as_millis() as u64)
                .unwrap_or(0),
            config: self.config.clone(),
        }
    }

    /// Force reset the circuit to closed state (use with caution)
    pub fn reset(&self) -> ApexResult<()> {
        let mut state = self.state.write().map_err(|e| {
            ApexError::Internal(anyhow::anyhow!(
                "Failed to acquire circuit breaker lock: {}",
                e
            ))
        })?;

        state.state = CircuitState::Closed;
        state.failures.clear();
        state.successes_in_half_open = 0;
        state.requests_in_half_open = 0;
        state.current_timeout = Duration::from_millis(self.config.initial_timeout_ms);
        state.last_state_change = Instant::now();

        info!(agent = %self.agent_id, "Circuit breaker manually reset");
        Ok(())
    }
}

/// Circuit breaker statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitStats {
    pub agent_id: String,
    pub state: CircuitState,
    pub trip_count: u64,
    pub failures_in_window: usize,
    pub current_timeout_ms: u64,
    pub time_in_state_ms: u64,
    pub config: CircuitBreakerConfig,
}

/// Circuit breaker manager for multiple agents
pub struct CircuitBreakerManager {
    /// Circuit breakers by agent ID
    breakers: RwLock<HashMap<String, CircuitBreaker>>,
    /// Default configuration for new breakers
    default_config: CircuitBreakerConfig,
    /// Event log for kernel integration
    event_log: RwLock<Vec<CircuitBreakerEvent>>,
}

impl CircuitBreakerManager {
    /// Create a new manager
    pub fn new() -> Self {
        info!("🔌 Circuit Breaker Manager initialized");
        Self {
            breakers: RwLock::new(HashMap::new()),
            default_config: CircuitBreakerConfig::default(),
            event_log: RwLock::new(Vec::new()),
        }
    }

    /// Create with custom default configuration
    pub fn with_config(config: CircuitBreakerConfig) -> Self {
        Self {
            breakers: RwLock::new(HashMap::new()),
            default_config: config,
            event_log: RwLock::new(Vec::new()),
        }
    }

    /// Drain pending events (retrieve and clear)
    pub fn drain_events(&self) -> Vec<CircuitBreakerEvent> {
        self.event_log
            .write()
            .map(|mut log| {
                let events = log.clone();
                log.clear();
                events
            })
            .unwrap_or_default()
    }

    /// Emit an event to the log
    fn emit_event(&self, event: CircuitBreakerEvent) {
        if let Ok(mut log) = self.event_log.write() {
            log.push(event);
        }
    }

    /// Get or create a circuit breaker for an agent
    pub fn get_or_create(&self, agent_id: &str) -> ApexResult<()> {
        let mut breakers = self.breakers.write().map_err(|e| {
            ApexError::Internal(anyhow::anyhow!("Failed to acquire breakers lock: {}", e))
        })?;

        if !breakers.contains_key(agent_id) {
            let breaker = CircuitBreaker::new(agent_id, self.default_config.clone());
            breakers.insert(agent_id.to_string(), breaker);
        }

        Ok(())
    }

    /// Check if a request is allowed for an agent
    pub fn allow_request(&self, agent_id: &str) -> ApexResult<bool> {
        self.get_or_create(agent_id)?;

        let breakers = self
            .breakers
            .read()
            .map_err(|e| ApexError::Internal(anyhow::anyhow!("Failed to read breakers: {}", e)))?;

        match breakers.get(agent_id) {
            Some(breaker) => {
                let (allowed, event) = breaker.allow_request()?;
                if let Some(event) = event {
                    self.emit_event(event);
                }
                Ok(allowed)
            }
            None => Ok(true), // Allow if no breaker (shouldn't happen)
        }
    }

    /// Record success for an agent
    pub fn record_success(&self, agent_id: &str) -> ApexResult<()> {
        self.get_or_create(agent_id)?;

        let breakers = self
            .breakers
            .read()
            .map_err(|e| ApexError::Internal(anyhow::anyhow!("Failed to read breakers: {}", e)))?;

        if let Some(breaker) = breakers.get(agent_id) {
            if let Some(event) = breaker.record_success()? {
                self.emit_event(event);
            }
        }

        Ok(())
    }

    /// Record failure for an agent
    pub fn record_failure(&self, agent_id: &str) -> ApexResult<()> {
        self.get_or_create(agent_id)?;

        let breakers = self
            .breakers
            .read()
            .map_err(|e| ApexError::Internal(anyhow::anyhow!("Failed to read breakers: {}", e)))?;

        if let Some(breaker) = breakers.get(agent_id) {
            if let Some(event) = breaker.record_failure()? {
                self.emit_event(event);
            }
        }

        Ok(())
    }

    /// Get state for an agent
    pub fn get_state(&self, agent_id: &str) -> CircuitState {
        self.breakers
            .read()
            .ok()
            .and_then(|b| b.get(agent_id).map(|cb| cb.get_state()))
            .unwrap_or(CircuitState::Closed)
    }

    /// Get stats for all agents
    pub fn get_all_stats(&self) -> Vec<CircuitStats> {
        self.breakers
            .read()
            .map(|b| b.values().map(|cb| cb.get_stats()).collect())
            .unwrap_or_default()
    }

    /// Get agents with open circuits
    pub fn get_open_circuits(&self) -> Vec<String> {
        self.breakers
            .read()
            .map(|b| {
                b.iter()
                    .filter(|(_, cb)| cb.get_state() == CircuitState::Open)
                    .map(|(id, _)| id.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Reset a specific agent's circuit
    pub fn reset(&self, agent_id: &str) -> ApexResult<()> {
        let breakers = self
            .breakers
            .read()
            .map_err(|e| ApexError::Internal(anyhow::anyhow!("Failed to read breakers: {}", e)))?;

        if let Some(breaker) = breakers.get(agent_id) {
            breaker.reset()?;
        }

        Ok(())
    }

    /// Reset all circuits
    pub fn reset_all(&self) -> ApexResult<()> {
        let breakers = self
            .breakers
            .read()
            .map_err(|e| ApexError::Internal(anyhow::anyhow!("Failed to read breakers: {}", e)))?;

        for breaker in breakers.values() {
            breaker.reset()?;
        }

        Ok(())
    }
}

impl Default for CircuitBreakerManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_circuit_breaker_starts_closed() {
        let breaker = CircuitBreaker::with_defaults("test_agent");
        assert_eq!(breaker.get_state(), CircuitState::Closed);
        let (allowed, _event) = breaker.allow_request().unwrap();
        assert!(allowed);
    }

    #[test]
    fn test_circuit_trips_after_threshold() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            failure_window_ms: 60_000,
            ..Default::default()
        };
        let breaker = CircuitBreaker::new("test_agent", config);

        // Record failures
        for _ in 0..3 {
            breaker.record_failure().unwrap();
        }

        // Circuit should be open
        assert_eq!(breaker.get_state(), CircuitState::Open);
        let (allowed, _event) = breaker.allow_request().unwrap();
        assert!(!allowed);
    }

    #[test]
    fn test_circuit_recovers_after_timeout() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            initial_timeout_ms: 50, // Very short for testing
            success_threshold: 1,
            half_open_max_requests: 5,
            ..Default::default()
        };
        let breaker = CircuitBreaker::new("test_agent", config);

        // Trip the circuit
        breaker.record_failure().unwrap();
        breaker.record_failure().unwrap();
        assert_eq!(breaker.get_state(), CircuitState::Open);

        // Wait for timeout
        thread::sleep(Duration::from_millis(100));

        // Should transition to half-open
        let (allowed, _event) = breaker.allow_request().unwrap();
        assert!(allowed);
        assert_eq!(breaker.get_state(), CircuitState::HalfOpen);

        // Record success to close
        breaker.record_success().unwrap();
        assert_eq!(breaker.get_state(), CircuitState::Closed);
    }

    #[test]
    fn test_exponential_backoff() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            initial_timeout_ms: 100,
            backoff_multiplier: 2.0,
            max_timeout_ms: 1000,
            half_open_max_requests: 5,
            ..Default::default()
        };
        let breaker = CircuitBreaker::new("test_agent", config);

        // First trip
        breaker.record_failure().unwrap();
        let stats1 = breaker.get_stats();

        // Wait and recover to half-open
        thread::sleep(Duration::from_millis(150));
        let (_allowed, _event) = breaker.allow_request().unwrap();

        // Fail again in half-open
        breaker.record_failure().unwrap();
        let stats2 = breaker.get_stats();

        // Timeout should have doubled
        assert!(stats2.current_timeout_ms > stats1.current_timeout_ms);
    }

    #[test]
    fn test_circuit_breaker_manager() {
        let manager = CircuitBreakerManager::new();

        // Multiple agents
        assert!(manager.allow_request("agent_1").unwrap());
        assert!(manager.allow_request("agent_2").unwrap());

        // Record failures for one agent
        for _ in 0..5 {
            manager.record_failure("agent_1").unwrap();
        }

        // Agent 1 should be blocked
        assert_eq!(manager.get_state("agent_1"), CircuitState::Open);

        // Agent 2 should still work
        assert!(manager.allow_request("agent_2").unwrap());
        assert_eq!(manager.get_state("agent_2"), CircuitState::Closed);
    }

    #[test]
    fn test_get_open_circuits() {
        let manager = CircuitBreakerManager::with_config(CircuitBreakerConfig {
            failure_threshold: 2,
            ..Default::default()
        });

        manager.record_failure("agent_1").unwrap();
        manager.record_failure("agent_1").unwrap();

        manager.record_failure("agent_2").unwrap();
        manager.record_failure("agent_2").unwrap();

        manager.allow_request("agent_3").unwrap(); // Just create it

        let open = manager.get_open_circuits();
        assert_eq!(open.len(), 2);
        assert!(open.contains(&"agent_1".to_string()));
        assert!(open.contains(&"agent_2".to_string()));
    }

    #[test]
    fn test_reset() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            ..Default::default()
        };
        let breaker = CircuitBreaker::new("test_agent", config);

        // Trip the circuit
        breaker.record_failure().unwrap();
        breaker.record_failure().unwrap();
        assert_eq!(breaker.get_state(), CircuitState::Open);

        // Reset
        breaker.reset().unwrap();
        assert_eq!(breaker.get_state(), CircuitState::Closed);
        let (allowed, _event) = breaker.allow_request().unwrap();
        assert!(allowed);
    }
}
