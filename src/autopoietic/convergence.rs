// src/autopoietic/convergence.rs - KEP Detection and Convergence Metrics
//
// Implements Step 10 of the 11-step cycle:
// - Knowledge Explosion Point (KEP) detection
// - Plateau vs Explosion state transitions
// - Convergence metrics tracking
// - Learning rate modulation

use crate::autopoietic::types::{
    GenerationPerformance, KEPState, KEPThresholds,
};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tracing::{debug, info, warn};

/// Convergence detector for KEP state management
pub struct ConvergenceDetector {
    /// KEP thresholds configuration
    thresholds: KEPThresholds,

    /// Window size for convergence analysis
    window_size: usize,

    /// Improvement threshold for plateau detection
    improvement_threshold: f64,

    /// Recent generation performances
    history: VecDeque<GenerationPerformance>,

    /// Current KEP state
    current_state: KEPState,

    /// Time when explosion mode was entered
    explosion_start: Option<DateTime<Utc>>,

    /// Time when last explosion mode ended
    last_explosion_end: Option<DateTime<Utc>>,

    /// Total time spent in explosion mode
    total_explosion_time_seconds: u64,
}

impl ConvergenceDetector {
    pub fn new(thresholds: KEPThresholds, window_size: usize, improvement_threshold: f64) -> Self {
        Self {
            thresholds,
            window_size,
            improvement_threshold,
            history: VecDeque::with_capacity(window_size + 1),
            current_state: KEPState::Normal,
            explosion_start: None,
            last_explosion_end: None,
            total_explosion_time_seconds: 0,
        }
    }

    /// Add a generation performance and check for state transitions
    pub fn update(&mut self, performance: GenerationPerformance) -> ConvergenceUpdate {
        // Add to history
        self.history.push_back(performance.clone());
        if self.history.len() > self.window_size {
            self.history.pop_front();
        }

        // Calculate convergence metrics
        let metrics = self.calculate_metrics();

        // Check for KEP state transitions
        let previous_state = self.current_state;
        let new_state = self.check_kep(&performance, &metrics);

        // Handle state transition
        if new_state != previous_state {
            self.handle_state_transition(previous_state, new_state);
        }

        self.current_state = new_state;

        ConvergenceUpdate {
            previous_state,
            new_state,
            metrics: metrics.clone(),
            state_changed: new_state != previous_state,
            explosion_duration_seconds: self.current_explosion_duration(),
        }
    }

    /// Check KEP state based on current performance and metrics
    fn check_kep(&self, current: &GenerationPerformance, metrics: &ConvergenceMetrics) -> KEPState {
        let kep = &current.kep_progress;

        // Check if we're in cooldown after explosion
        if self.in_cooldown() {
            debug!("KEP: In cooldown period after explosion");
            return KEPState::Normal;
        }

        // Explosion entry conditions
        let mass_ok = kep.knowledge_mass >= self.thresholds.min_knowledge_mass;
        let velocity_ok = kep.discovery_velocity >= self.thresholds.min_velocity;
        let synergy_ok = kep.synergy_density >= self.thresholds.min_synergy_density;
        let ihsan_ok = current.aggregate_ihsan >= 0.95;

        // Currently in explosion mode
        if self.current_state == KEPState::InExplosion {
            // Check exit conditions
            if !velocity_ok || !ihsan_ok {
                return KEPState::ExitingExplosion;
            }
            return KEPState::InExplosion;
        }

        // Check for explosion entry
        if mass_ok && velocity_ok && synergy_ok && ihsan_ok {
            if self.current_state == KEPState::EnteringExplosion {
                return KEPState::InExplosion;
            }
            return KEPState::EnteringExplosion;
        }

        // Check for approaching explosion
        let approaching_mass = kep.knowledge_mass >= self.thresholds.approaching_mass;
        let approaching_velocity = kep.discovery_velocity >= self.thresholds.approaching_velocity;

        if approaching_mass && approaching_velocity && ihsan_ok {
            return KEPState::Approaching;
        }

        // Check for plateau (convergence)
        if self.is_plateaued(metrics) {
            return KEPState::Plateau;
        }

        // Normal operation
        KEPState::Normal
    }

    /// Check if system has plateaued
    fn is_plateaued(&self, metrics: &ConvergenceMetrics) -> bool {
        if self.history.len() < self.window_size / 2 {
            return false;
        }

        // Check if all recent improvement deltas are below threshold
        metrics.improvement_deltas.iter().all(|d| d.abs() < self.improvement_threshold)
    }

    /// Handle state transition
    fn handle_state_transition(&mut self, from: KEPState, to: KEPState) {
        info!(
            from = ?from,
            to = ?to,
            "KEP state transition"
        );

        match (from, to) {
            (_, KEPState::InExplosion) => {
                self.explosion_start = Some(Utc::now());
                info!("🚀 Entering Knowledge Explosion Mode!");
            }
            (KEPState::InExplosion, _) | (_, KEPState::ExitingExplosion) => {
                if let Some(start) = self.explosion_start {
                    let duration = (Utc::now() - start).num_seconds() as u64;
                    self.total_explosion_time_seconds += duration;
                    info!(
                        duration_seconds = duration,
                        total_explosion_seconds = self.total_explosion_time_seconds,
                        "📉 Exiting Knowledge Explosion Mode"
                    );
                }
                self.explosion_start = None;
                self.last_explosion_end = Some(Utc::now());
            }
            (_, KEPState::Plateau) => {
                warn!("⏸️ System has plateaued - consider external intervention");
            }
            _ => {}
        }
    }

    /// Check if in cooldown period after explosion
    fn in_cooldown(&self) -> bool {
        if let Some(end) = self.last_explosion_end {
            let cooldown_duration = Duration::seconds(self.thresholds.explosion_cooldown_seconds as i64);
            return Utc::now() < end + cooldown_duration;
        }
        false
    }

    /// Get current explosion duration if in explosion mode
    fn current_explosion_duration(&self) -> u64 {
        match (self.current_state, self.explosion_start) {
            (KEPState::InExplosion, Some(start)) => (Utc::now() - start).num_seconds() as u64,
            _ => 0,
        }
    }

    /// Calculate convergence metrics from history
    fn calculate_metrics(&self) -> ConvergenceMetrics {
        if self.history.is_empty() {
            return ConvergenceMetrics::default();
        }

        let ihsan_scores: Vec<f64> = self.history.iter().map(|p| p.aggregate_ihsan).collect();
        let latencies: Vec<u64> = self.history.iter().map(|p| p.avg_latency_ms).collect();
        let success_rates: Vec<f64> = self.history.iter().map(|p| {
            if p.tasks_processed > 0 {
                p.successful_executions as f64 / p.tasks_processed as f64
            } else {
                0.0
            }
        }).collect();

        // Calculate improvement deltas (difference from previous generation)
        let improvement_deltas: Vec<f64> = ihsan_scores
            .windows(2)
            .map(|w| w[1] - w[0])
            .collect();

        // Calculate trend
        let trend = if improvement_deltas.is_empty() {
            0.0
        } else {
            improvement_deltas.iter().sum::<f64>() / improvement_deltas.len() as f64
        };

        // Calculate stability (inverse of variance)
        let mean_ihsan = ihsan_scores.iter().sum::<f64>() / ihsan_scores.len() as f64;
        let variance = ihsan_scores.iter()
            .map(|s| (s - mean_ihsan).powi(2))
            .sum::<f64>() / ihsan_scores.len() as f64;
        let stability = 1.0 / (1.0 + variance.sqrt());

        // Best/worst values
        let best_ihsan = ihsan_scores.iter().cloned().fold(0.0, f64::max);
        let worst_ihsan = ihsan_scores.iter().cloned().fold(1.0, f64::min);
        let best_latency = *latencies.iter().min().unwrap_or(&0);
        let worst_latency = *latencies.iter().max().unwrap_or(&0);

        ConvergenceMetrics {
            generations_analyzed: self.history.len(),
            improvement_deltas,
            trend,
            stability,
            mean_ihsan,
            best_ihsan,
            worst_ihsan,
            mean_latency_ms: latencies.iter().sum::<u64>() / latencies.len().max(1) as u64,
            best_latency_ms: best_latency,
            worst_latency_ms: worst_latency,
            mean_success_rate: success_rates.iter().sum::<f64>() / success_rates.len().max(1) as f64,
            total_explosion_time_seconds: self.total_explosion_time_seconds,
        }
    }

    /// Get current convergence state
    pub fn get_state(&self) -> ConvergenceState {
        ConvergenceState {
            kep_state: self.current_state,
            metrics: self.calculate_metrics(),
            in_cooldown: self.in_cooldown(),
            explosion_duration_seconds: self.current_explosion_duration(),
        }
    }

    /// Get current KEP state
    pub fn kep_state(&self) -> KEPState {
        self.current_state
    }

    /// Get learning rate multiplier based on current state
    pub fn learning_rate_multiplier(&self) -> f64 {
        match self.current_state {
            KEPState::Normal => 1.0,
            KEPState::Approaching => 1.5,
            KEPState::EnteringExplosion => 2.0,
            KEPState::InExplosion => 3.0,
            KEPState::ExitingExplosion => 1.5,
            KEPState::Plateau => 0.5, // Slow down when plateaued
        }
    }

    /// Reset the detector state
    pub fn reset(&mut self) {
        self.history.clear();
        self.current_state = KEPState::Normal;
        self.explosion_start = None;
        self.last_explosion_end = None;
        // Keep total_explosion_time_seconds for historical tracking
    }
}

impl Default for ConvergenceDetector {
    fn default() -> Self {
        Self::new(KEPThresholds::default(), 10, 0.001)
    }
}

/// Convergence metrics computed from generation history
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConvergenceMetrics {
    /// Number of generations analyzed
    pub generations_analyzed: usize,

    /// Improvement deltas between consecutive generations
    pub improvement_deltas: Vec<f64>,

    /// Overall trend (positive = improving)
    pub trend: f64,

    /// Stability score (higher = more stable)
    pub stability: f64,

    /// Mean Ihsān score
    pub mean_ihsan: f64,

    /// Best Ihsān score
    pub best_ihsan: f64,

    /// Worst Ihsān score
    pub worst_ihsan: f64,

    /// Mean latency
    pub mean_latency_ms: u64,

    /// Best latency
    pub best_latency_ms: u64,

    /// Worst latency
    pub worst_latency_ms: u64,

    /// Mean success rate
    pub mean_success_rate: f64,

    /// Total time spent in explosion mode
    pub total_explosion_time_seconds: u64,
}

impl ConvergenceMetrics {
    /// Check if system is converging (stable with high Ihsān)
    pub fn is_converging(&self) -> bool {
        self.stability > 0.8 && self.mean_ihsan >= 0.95
    }

    /// Check if system is improving
    pub fn is_improving(&self) -> bool {
        self.trend > 0.0 && !self.improvement_deltas.is_empty()
    }

    /// Check if system is degrading
    pub fn is_degrading(&self) -> bool {
        self.trend < -(self.improvement_deltas.len() as f64) * 0.001
    }
}

/// Update result from convergence check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceUpdate {
    /// Previous KEP state
    pub previous_state: KEPState,

    /// New KEP state
    pub new_state: KEPState,

    /// Current metrics
    pub metrics: ConvergenceMetrics,

    /// Whether state changed
    pub state_changed: bool,

    /// Duration in explosion mode (if applicable)
    pub explosion_duration_seconds: u64,
}

/// Complete convergence state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceState {
    /// Current KEP state
    pub kep_state: KEPState,

    /// Convergence metrics
    pub metrics: ConvergenceMetrics,

    /// Whether in cooldown after explosion
    pub in_cooldown: bool,

    /// Current explosion duration (if in explosion)
    pub explosion_duration_seconds: u64,
}

impl std::fmt::Display for ConvergenceState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KEP: {:?}, Ihsān: {:.4}, Trend: {:+.4}, Stability: {:.2}",
            self.kep_state,
            self.metrics.mean_ihsan,
            self.metrics.trend,
            self.metrics.stability
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autopoietic::types::{IhsanDimensions, KEPProgress, SAPEResults};

    fn make_performance(generation: u64, ihsan: f64, kep: KEPProgress) -> GenerationPerformance {
        GenerationPerformance {
            generation,
            started_at: Utc::now(),
            ended_at: Utc::now(),
            duration_ms: 60000,
            aggregate_ihsan: ihsan,
            ihsan_dimensions: IhsanDimensions::default(),
            sape_results: SAPEResults::default(),
            tasks_processed: 100,
            successful_executions: 95,
            rejections: 5,
            avg_latency_ms: 150,
            p95_latency_ms: 200,
            kep_progress: kep,
            improvements_applied: Vec::new(),
            proof_hash: "test".to_string(),
            receipt_id: "test".to_string(),
        }
    }

    #[test]
    fn test_normal_state() {
        let mut detector = ConvergenceDetector::default();

        let kep = KEPProgress {
            knowledge_mass: 100,
            discovery_velocity: 1.0,
            synergy_density: 0.1,
            ..Default::default()
        };

        let perf = make_performance(1, 0.96, kep);
        let update = detector.update(perf);

        assert_eq!(update.new_state, KEPState::Normal);
    }

    #[test]
    fn test_explosion_entry() {
        let mut detector = ConvergenceDetector::default();

        let kep = KEPProgress {
            knowledge_mass: 2000,  // Above min_knowledge_mass (1000)
            discovery_velocity: 15.0,  // Above min_velocity (10.0)
            synergy_density: 0.5,  // Above min_synergy_density (0.3)
            ..Default::default()
        };

        let perf = make_performance(1, 0.96, kep);
        let update = detector.update(perf);

        assert_eq!(update.new_state, KEPState::EnteringExplosion);

        // Second update should transition to InExplosion
        let kep2 = KEPProgress {
            knowledge_mass: 2500,
            discovery_velocity: 20.0,
            synergy_density: 0.6,
            ..Default::default()
        };

        let perf2 = make_performance(2, 0.97, kep2);
        let update2 = detector.update(perf2);

        assert_eq!(update2.new_state, KEPState::InExplosion);
    }

    #[test]
    fn test_plateau_detection() {
        let mut detector = ConvergenceDetector::new(KEPThresholds::default(), 5, 0.001);

        // Add multiple generations with nearly identical Ihsān scores
        for i in 0..7 {
            let kep = KEPProgress::default();
            let perf = make_performance(i, 0.95 + (i as f64 * 0.0001), kep);
            detector.update(perf);
        }

        let state = detector.get_state();
        assert_eq!(state.kep_state, KEPState::Plateau);
    }

    #[test]
    fn test_learning_rate_multiplier() {
        let mut detector = ConvergenceDetector::default();

        assert_eq!(detector.learning_rate_multiplier(), 1.0);

        // Force into explosion mode
        let kep = KEPProgress {
            knowledge_mass: 2000,
            discovery_velocity: 15.0,
            synergy_density: 0.5,
            ..Default::default()
        };
        detector.update(make_performance(1, 0.96, kep.clone()));
        detector.update(make_performance(2, 0.97, kep));

        assert_eq!(detector.learning_rate_multiplier(), 3.0);
    }

    #[test]
    fn test_metrics_calculation() {
        let mut detector = ConvergenceDetector::default();

        for i in 0..5 {
            let kep = KEPProgress::default();
            let ihsan = 0.90 + (i as f64 * 0.02);
            detector.update(make_performance(i, ihsan, kep));
        }

        let metrics = detector.calculate_metrics();
        assert_eq!(metrics.generations_analyzed, 5);
        assert!(metrics.trend > 0.0); // Should be improving
        assert!(metrics.mean_ihsan > 0.9);
    }
}
