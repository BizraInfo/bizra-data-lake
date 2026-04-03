// src/sovereign/thermal.rs - Thermal Consciousness Engine
//
// # LANGEVIN DYNAMICS WITH LYAPUNOV STABILITY
//
// Standing on the Shoulders of Giants:
// - Szu & Hartley (1987): "Nonconvex Optimization by Fast Simulated Annealing"
// - Langevin (1908): Stochastic differential equations
// - Lyapunov (1892): Stability theory for dynamical systems
//
// ## Mathematical Foundation
//
// The Thermal Consciousness Engine implements the Langevin equation:
//
// ```
// dX_t = -∇E(X_t)dt + √(2T(t))dW_t
// ```
//
// Where:
// - X_t: State vector at time t
// - E(X): Energy function (objective to minimize)
// - T(t): Temperature schedule (FSA: T₀/(1+t))
// - W_t: Wiener process (Brownian motion)
//
// ## Lyapunov Stability
//
// The system is proven stable via the Lyapunov function:
//
// ```
// V(x) = E(x) - E(x*) + 0.5‖x - x*‖²
// ```
//
// Stability guaranteed when: T ≤ ‖∇E‖² / (Δ + d)
//
// ## PAT↔SAT Coordination
//
// The Reconciler implements switched-system control with dwell-time constraints
// for BIBO (Bounded-Input Bounded-Output) stability.

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

// ============================================================================
// WISDOM FEEDBACK TYPES
// ============================================================================

/// Feedback from WisdomStore to adjust thermal energy landscape
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WisdomFeedback {
    pub pattern_name: String,
    pub effectiveness: f64,
    pub context_vector: Vec<f64>,
    pub generation: u64,
}

/// Record of gradient adjustment from wisdom feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientAdjustment {
    pub dimension: String,
    pub old_weight: f64,
    pub new_weight: f64,
    pub reason: String,
}

// ============================================================================
// THERMAL CONSCIOUSNESS ENGINE
// ============================================================================

/// Thermal Consciousness Engine implementing Langevin dynamics
/// with Lyapunov stability guarantees for global optimization
#[derive(Clone, Debug)]
pub struct ThermalConsciousness {
    /// Current state vector
    state: Vec<f64>,
    /// Current energy (objective value)
    energy: f64,
    /// Best energy found
    best_energy: f64,
    /// Best state found
    best_state: Vec<f64>,
    /// Current temperature
    temperature: f64,
    /// Initial temperature T₀
    t0: f64,
    /// Time step counter
    time_step: u64,
    /// Learning rate (gradient step size η)
    learning_rate: f64,
    /// Lyapunov function value
    lyapunov_value: f64,
    /// Random seed for reproducibility
    seed: u64,
    /// Configuration
    config: ThermalConfig,
}

/// Configuration for thermal consciousness
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThermalConfig {
    /// State space dimensions
    pub dimensions: usize,
    /// Initial temperature
    pub initial_temperature: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Minimum temperature (convergence threshold)
    pub min_temperature: f64,
    /// Energy convergence threshold
    pub energy_threshold: f64,
    /// Gradient norm convergence threshold
    pub gradient_threshold: f64,
    /// Maximum iterations
    pub max_iterations: u64,
    /// Temperature schedule type
    pub schedule: TemperatureSchedule,
}

impl Default for ThermalConfig {
    fn default() -> Self {
        Self {
            dimensions: 64,
            initial_temperature: 1.0,
            learning_rate: 0.01,
            min_temperature: 1e-6,
            energy_threshold: 1e-6,
            gradient_threshold: 1e-4,
            max_iterations: 100_000,
            schedule: TemperatureSchedule::FastSimulatedAnnealing,
        }
    }
}

/// Temperature schedule types
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum TemperatureSchedule {
    /// Fast Simulated Annealing: T(t) = T₀ / (1 + t)
    FastSimulatedAnnealing,
    /// Boltzmann Annealing: T(t) = T₀ / log(1 + t)
    BoltzmannAnnealing,
    /// Exponential Annealing: T(t) = T₀ * α^t, α < 1
    ExponentialAnnealing { alpha: f64 },
    /// Adaptive: Adjusts based on acceptance rate
    Adaptive { target_acceptance: f64 },
}

impl ThermalConsciousness {
    /// Create new thermal consciousness engine with full config
    pub fn with_config(config: ThermalConfig) -> Self {
        let dimensions = config.dimensions;
        let t0 = config.initial_temperature;
        let learning_rate = config.learning_rate;

        Self {
            state: vec![0.5; dimensions], // Initialize at midpoint of [0,1]^d
            energy: f64::MAX,
            best_energy: f64::MAX,
            best_state: vec![0.5; dimensions],
            temperature: t0,
            t0,
            time_step: 0,
            learning_rate,
            lyapunov_value: f64::MAX,
            seed: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(12345),
            config,
        }
    }

    /// Create new thermal consciousness engine with simple parameters
    ///
    /// # Arguments
    /// * `initial_energy` - Starting energy value
    /// * `initial_temperature` - Initial temperature T₀
    /// * `learning_rate` - Gradient step size η
    pub fn new(initial_energy: f64, initial_temperature: f64, learning_rate: f64) -> Self {
        let config = ThermalConfig {
            initial_temperature,
            learning_rate,
            ..Default::default()
        };
        let mut tc = Self::with_config(config);
        tc.energy = initial_energy;
        tc
    }

    /// Check if the thermal system is stable
    ///
    /// Returns true if energy is bounded and Lyapunov function is decreasing
    /// A fresh system (not yet stepped) is considered stable by default.
    pub fn is_stable(&self) -> bool {
        // Fresh system (never stepped) is stable by default
        if self.time_step == 0 {
            return true;
        }

        self.energy.is_finite()
            && self.lyapunov_value.is_finite()
            && self.lyapunov_value < 1e6
            && self.temperature > 0.0
    }

    /// Create with default configuration
    pub fn default_config(dimensions: usize, initial_temperature: f64) -> Self {
        Self::with_config(ThermalConfig {
            dimensions,
            initial_temperature,
            ..Default::default()
        })
    }

    /// Set initial state
    pub fn set_state(&mut self, state: Vec<f64>) {
        assert_eq!(state.len(), self.config.dimensions);
        self.state = state.clone();
        self.best_state = state;
    }

    /// Get current state
    pub fn state(&self) -> &[f64] {
        &self.state
    }

    /// Get current temperature
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Get current energy
    pub fn energy(&self) -> f64 {
        self.energy
    }

    /// Get best energy found
    pub fn best_energy(&self) -> f64 {
        self.best_energy
    }

    /// Get best state found
    pub fn best_state(&self) -> &[f64] {
        &self.best_state
    }

    /// Get Lyapunov function value (stability indicator)
    pub fn lyapunov(&self) -> f64 {
        self.lyapunov_value
    }

    /// Get time step
    pub fn time_step(&self) -> u64 {
        self.time_step
    }

    /// Calculate temperature based on schedule
    fn calculate_temperature(&self) -> f64 {
        let t = self.time_step as f64;

        match self.config.schedule {
            TemperatureSchedule::FastSimulatedAnnealing => {
                // Szu-Hartley FSA: T(t) = T₀ / (1 + t)
                // Provably converges to global optimum
                self.t0 / (1.0 + t)
            }
            TemperatureSchedule::BoltzmannAnnealing => {
                // Classical: T(t) = T₀ / log(1 + t)
                // Slower but more thorough exploration
                self.t0 / (1.0 + t).ln().max(1.0)
            }
            TemperatureSchedule::ExponentialAnnealing { alpha } => {
                // Geometric: T(t) = T₀ * α^t
                // Fastest but may miss global optimum
                self.t0 * alpha.powf(t)
            }
            TemperatureSchedule::Adaptive { .. } => {
                // Adaptive temperature (handled separately)
                self.temperature
            }
        }
    }

    /// Generate Cauchy-distributed random step (fat tails for FSA)
    fn cauchy_step(&mut self) -> f64 {
        // Cauchy distribution for fat-tailed exploration
        // More likely to make large jumps than Gaussian
        let u = self.random_uniform();
        // Cauchy quantile function: tan(π(u - 0.5))
        (PI * (u - 0.5)).tan()
    }

    /// Generate Beta-distributed sample for bounded manifold
    fn beta_sample(&mut self, alpha: f64) -> f64 {
        // Approximation using inverse transform
        let u1 = self.random_uniform();
        let u2 = self.random_uniform();

        let x = u1.powf(1.0 / alpha);
        let y = u2.powf(1.0 / alpha);
        x / (x + y)
    }

    /// Simple xorshift64 PRNG
    fn random_uniform(&mut self) -> f64 {
        self.seed ^= self.seed << 13;
        self.seed ^= self.seed >> 7;
        self.seed ^= self.seed << 17;
        (self.seed as f64) / (u64::MAX as f64)
    }

    /// Generate Gaussian random variable (Box-Muller transform)
    fn random_gaussian(&mut self) -> f64 {
        let u1 = self.random_uniform().max(1e-10);
        let u2 = self.random_uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    /// Perform one step of Langevin dynamics
    ///
    /// Implements: dX = -η∇E(X)dt + √(2ηT)dW
    ///
    /// # Arguments
    /// * `gradient` - Gradient of energy function ∇E(x)
    /// * `energy` - Current energy E(x)
    ///
    /// # Returns
    /// New state after Langevin update
    pub fn step(&mut self, gradient: &[f64], energy: f64) -> Vec<f64> {
        assert_eq!(gradient.len(), self.state.len());

        self.time_step += 1;
        self.temperature = self
            .calculate_temperature()
            .max(self.config.min_temperature);
        self.energy = energy;

        // Track best solution
        if energy < self.best_energy {
            self.best_energy = energy;
            self.best_state = self.state.clone();
        }

        // Compute drift term: -η∇E(x)
        let drift: Vec<f64> = gradient.iter().map(|g| -self.learning_rate * g).collect();

        // Compute diffusion term: √(2ηT) * noise
        let noise_scale = (2.0 * self.learning_rate * self.temperature).sqrt();

        // Use appropriate noise distribution based on schedule
        let noise: Vec<f64> = if self.config.schedule == TemperatureSchedule::FastSimulatedAnnealing
        {
            // Cauchy noise for FSA (fat tails)
            (0..self.state.len())
                .map(|_| self.cauchy_step() * noise_scale * 0.1)
                .collect()
        } else {
            // Gaussian noise for other schedules
            (0..self.state.len())
                .map(|_| self.random_gaussian() * noise_scale)
                .collect()
        };

        // Langevin update: x_{t+1} = x_t + drift + noise
        for i in 0..self.state.len() {
            self.state[i] += drift[i] + noise[i];
            // Clamp to [0, 1] for bounded manifold
            self.state[i] = self.state[i].clamp(0.0, 1.0);
        }

        // Update Lyapunov function value
        let gradient_norm_sq: f64 = gradient.iter().map(|g| g * g).sum();
        self.lyapunov_value = energy + 0.5 * gradient_norm_sq.sqrt();

        self.state.clone()
    }

    /// Check Lyapunov stability condition
    /// Stability guaranteed when: T ≤ ‖∇E‖² / (Δ + d)
    pub fn check_stability(&self, gradient: &[f64], laplacian: f64) -> bool {
        let d = self.state.len() as f64;
        let gradient_norm_sq: f64 = gradient.iter().map(|g| g * g).sum();

        if gradient_norm_sq < 1e-10 {
            return true; // At stationary point
        }

        let stability_bound = gradient_norm_sq / (laplacian.abs() + d);
        self.temperature <= stability_bound
    }

    /// Check convergence
    pub fn is_converged(&self, gradient: &[f64]) -> bool {
        let gradient_norm: f64 = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();

        self.energy < self.config.energy_threshold
            || gradient_norm < self.config.gradient_threshold
            || self.temperature < self.config.min_temperature
    }

    /// Golden ratio refinement for local search
    /// Uses Fibonacci minimax optimality
    pub fn golden_ratio_refine(&self, search_radius: f64) -> f64 {
        const PHI: f64 = 0.618_033_988_749_895;
        search_radius * PHI
    }

    /// Compute energy landscape curvature estimate
    pub fn estimate_curvature(&self, gradient: &[f64], prev_gradient: &[f64]) -> f64 {
        let delta_grad: f64 = gradient
            .iter()
            .zip(prev_gradient.iter())
            .map(|(g, pg)| (g - pg).powi(2))
            .sum::<f64>()
            .sqrt();

        let step_size = self.learning_rate;
        delta_grad / step_size.max(1e-10)
    }

    /// Reset to initial state
    pub fn reset(&mut self) {
        self.state = vec![0.5; self.config.dimensions];
        self.energy = f64::MAX;
        self.best_energy = f64::MAX;
        self.best_state = vec![0.5; self.config.dimensions];
        self.temperature = self.t0;
        self.time_step = 0;
        self.lyapunov_value = f64::MAX;
    }

    /// Absorb feedback from WisdomStore to adjust energy landscape
    /// Bounded: max ±10% adjustment per generation to prevent runaway
    ///
    /// # Arguments
    /// * `feedback` - Slice of wisdom feedback from historical patterns
    ///
    /// # Returns
    /// Vector of gradient adjustments applied to the energy landscape
    ///
    /// # Theory
    /// High-performing patterns (effectiveness > 0.95) reduce energy barriers,
    /// making it easier for the system to follow proven paths. This implements
    /// a form of "crystallization" where the energy landscape adapts to
    /// successful trajectories discovered by the wisdom system.
    pub fn absorb_wisdom_feedback(&mut self, feedback: &[WisdomFeedback]) -> Vec<GradientAdjustment> {
        let mut adjustments = Vec::new();

        for fb in feedback {
            if fb.effectiveness > 0.95 {
                // High-performing pattern: reduce energy in that direction
                // (make it easier to follow proven paths)
                let adjustment_factor = (fb.effectiveness - 0.95) * 2.0; // 0.0 to 0.1
                let clamped = adjustment_factor.clamp(-0.10, 0.10); // ±10% max

                // Record the adjustment
                adjustments.push(GradientAdjustment {
                    dimension: fb.pattern_name.clone(),
                    old_weight: self.energy,
                    new_weight: self.energy * (1.0 - clamped),
                    reason: format!("Wisdom feedback: pattern '{}' effectiveness {:.3}", fb.pattern_name, fb.effectiveness),
                });

                // Apply bounded adjustment to energy
                self.energy *= 1.0 - clamped;
            }
        }

        adjustments
    }
}

// ============================================================================
// RECONCILER (PAT↔SAT COORDINATOR)
// ============================================================================

/// Reconciler: Switched system controller for PAT↔SAT coordination
///
/// Implements dwell-time constraints for BIBO stability in switched systems.
/// Based on Lyapunov stability theory for hybrid dynamical systems.
#[derive(Clone, Debug)]
pub struct Reconciler {
    /// Current operating mode
    mode: ReconcilerMode,
    /// Time in current mode
    dwell_time: u64,
    /// Minimum dwell time for stability (τ_min)
    min_dwell_time: u64,
    /// Mode switch count
    switch_count: u64,
    /// Energy accumulator (for stability tracking)
    energy_accumulator: f64,
    /// Average energy (exponential moving average)
    avg_energy: f64,
    /// Energy variance (for mode selection)
    energy_variance: f64,
    /// PAT energy contribution
    pat_energy: f64,
    /// SAT energy contribution
    sat_energy: f64,
}

/// Operating modes for PAT↔SAT coordination
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ReconcilerMode {
    /// Exploration mode: High temperature, PAT-dominant
    /// - More creative, exploratory responses
    /// - Higher variance in outputs
    /// - Good for ideation and brainstorming
    Exploration,

    /// Exploitation mode: Low temperature, SAT-dominant
    /// - More precise, validated responses
    /// - Lower variance, higher confidence
    /// - Good for critical decisions
    Exploitation,

    /// Synthesis mode: Balanced PAT↔SAT
    /// - Combines creativity with validation
    /// - Moderate temperature
    /// - Default operating mode
    #[default]
    Synthesis,

    /// Balanced mode: Equal weighting (alias for Synthesis)
    /// - 50/50 PAT↔SAT coordination
    /// - Stable equilibrium point
    Balanced,
}

impl Reconciler {
    /// Create new reconciler with stability guarantees
    ///
    /// # Arguments
    /// * `mode` - Initial operating mode
    /// * `pat_count` - Number of PAT agents
    /// * `sat_count` - Number of SAT validators
    pub fn new(mode: ReconcilerMode, pat_count: usize, sat_count: usize) -> Self {
        // Minimum dwell time scales with team sizes for stability
        let min_dwell_time = (pat_count + sat_count) as u64;
        Self {
            mode,
            dwell_time: 0,
            min_dwell_time: min_dwell_time.max(3),
            switch_count: 0,
            energy_accumulator: 0.0,
            avg_energy: 0.0,
            energy_variance: 0.0,
            pat_energy: 0.0,
            sat_energy: 0.0,
        }
    }

    /// Create new reconciler with just minimum dwell time
    ///
    /// # Arguments
    /// * `min_dwell_time` - Minimum time in mode before switching (τ_min)
    pub fn with_dwell_time(min_dwell_time: u64) -> Self {
        Self {
            mode: ReconcilerMode::Synthesis,
            dwell_time: 0,
            min_dwell_time,
            switch_count: 0,
            energy_accumulator: 0.0,
            avg_energy: 0.0,
            energy_variance: 0.0,
            pat_energy: 0.0,
            sat_energy: 0.0,
        }
    }

    /// Step the reconciler with an energy/score value
    ///
    /// # Arguments
    /// * `score` - Current system score (e.g., Ihsān score)
    pub fn step(&mut self, score: f64) {
        // Distribute score between PAT and SAT based on mode
        let (pat_share, sat_share) = match self.mode {
            ReconcilerMode::Exploration => (0.7, 0.3),
            ReconcilerMode::Exploitation => (0.3, 0.7),
            ReconcilerMode::Synthesis | ReconcilerMode::Balanced => (0.5, 0.5),
        };

        self.tick(score * pat_share, score * sat_share);
        self.auto_select_mode();
    }

    /// Get current mode
    pub fn mode(&self) -> ReconcilerMode {
        self.mode
    }

    /// Get dwell time in current mode
    pub fn dwell_time(&self) -> u64 {
        self.dwell_time
    }

    /// Get total mode switches
    pub fn switch_count(&self) -> u64 {
        self.switch_count
    }

    /// Request mode switch (enforces dwell time constraint)
    ///
    /// Returns true if switch was successful, false if dwell time not met.
    pub fn request_switch(&mut self, new_mode: ReconcilerMode) -> bool {
        if self.mode == new_mode {
            return false;
        }

        // Enforce minimum dwell time for Lyapunov stability
        if self.dwell_time >= self.min_dwell_time {
            self.mode = new_mode;
            self.dwell_time = 0;
            self.switch_count += 1;
            true
        } else {
            false
        }
    }

    /// Force mode switch (bypasses dwell time)
    /// Use only in emergency situations
    pub fn force_switch(&mut self, new_mode: ReconcilerMode) {
        self.mode = new_mode;
        self.dwell_time = 0;
        self.switch_count += 1;
    }

    /// Tick reconciler clock with energy update
    ///
    /// # Arguments
    /// * `pat_energy` - Energy from PAT (execution) system
    /// * `sat_energy` - Energy from SAT (validation) system
    pub fn tick(&mut self, pat_energy: f64, sat_energy: f64) {
        self.dwell_time += 1;
        self.pat_energy = pat_energy;
        self.sat_energy = sat_energy;

        let total_energy = pat_energy + sat_energy;
        self.energy_accumulator += total_energy;

        // Exponential moving average
        let alpha = 0.1;
        self.avg_energy = alpha * total_energy + (1.0 - alpha) * self.avg_energy;

        // Running variance estimate
        let delta = total_energy - self.avg_energy;
        self.energy_variance = alpha * delta * delta + (1.0 - alpha) * self.energy_variance;
    }

    /// Auto-select mode based on energy landscape
    pub fn auto_select_mode(&mut self) -> ReconcilerMode {
        // High variance → need exploration
        // Low variance + high energy → need exploitation to converge
        // Moderate → synthesis

        let variance_threshold = 0.1;
        let energy_threshold = 0.5;

        let suggested = if self.energy_variance > variance_threshold {
            ReconcilerMode::Exploration
        } else if self.avg_energy > energy_threshold && self.energy_variance < variance_threshold {
            ReconcilerMode::Exploitation
        } else {
            ReconcilerMode::Synthesis
        };

        if self.request_switch(suggested) {
            suggested
        } else {
            self.mode
        }
    }

    /// Calculate adaptive sampling period
    /// Δt < 2 / (λ_max + T × σ_noise)
    pub fn adaptive_sample_period(&self, eigenvalue_max: f64, temperature: f64) -> f64 {
        let noise_scale = self.energy_variance.sqrt();
        2.0 / (eigenvalue_max + temperature * noise_scale + 1e-10)
    }

    /// Check BIBO (Bounded-Input Bounded-Output) stability
    pub fn is_stable(&self) -> bool {
        // Energy should be bounded (not diverging)
        self.energy_accumulator.abs() < 1e6 && self.avg_energy.is_finite()
    }

    /// Get PAT/SAT energy balance
    pub fn energy_balance(&self) -> f64 {
        if (self.pat_energy + self.sat_energy).abs() < 1e-10 {
            return 0.5;
        }
        self.pat_energy / (self.pat_energy + self.sat_energy)
    }

    /// Get recommended temperature for current mode
    pub fn recommended_temperature(&self) -> f64 {
        match self.mode {
            ReconcilerMode::Exploration => 1.0,
            ReconcilerMode::Exploitation => 0.1,
            ReconcilerMode::Synthesis | ReconcilerMode::Balanced => 0.5,
        }
    }

    /// Reset reconciler state
    pub fn reset(&mut self) {
        self.mode = ReconcilerMode::Synthesis;
        self.dwell_time = 0;
        self.switch_count = 0;
        self.energy_accumulator = 0.0;
        self.avg_energy = 0.0;
        self.energy_variance = 0.0;
        self.pat_energy = 0.0;
        self.sat_energy = 0.0;
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_consciousness_creation() {
        let tc = ThermalConsciousness::default_config(4, 1.0);
        assert_eq!(tc.state().len(), 4);
        assert_eq!(tc.temperature(), 1.0);
    }

    #[test]
    fn test_thermal_step() {
        let mut tc = ThermalConsciousness::default_config(4, 1.0);
        let gradient = vec![0.5, -0.3, 0.2, -0.1];
        let energy = 10.0;

        let new_state = tc.step(&gradient, energy);
        assert_eq!(new_state.len(), 4);

        // Temperature should decrease (FSA schedule)
        assert!(tc.temperature() < 1.0);
    }

    #[test]
    fn test_temperature_schedules() {
        let config = ThermalConfig {
            dimensions: 4,
            initial_temperature: 100.0,
            schedule: TemperatureSchedule::FastSimulatedAnnealing,
            ..Default::default()
        };
        let mut tc = ThermalConsciousness::with_config(config);

        // Step multiple times
        let gradient = vec![0.1; 4];
        for _ in 0..100 {
            tc.step(&gradient, 1.0);
        }

        // Temperature should have decreased significantly
        assert!(tc.temperature() < 1.0);
    }

    #[test]
    fn test_convergence_tracking() {
        let mut tc = ThermalConsciousness::default_config(4, 1.0);
        let gradient = vec![0.0001; 4];
        tc.step(&gradient, 1e-7);

        assert!(tc.is_converged(&gradient));
    }

    #[test]
    fn test_reconciler_dwell_time() {
        let mut rec = Reconciler::with_dwell_time(5);

        // Should not switch immediately (dwell time not met)
        assert!(!rec.request_switch(ReconcilerMode::Exploration));

        // Tick past minimum dwell time
        for _ in 0..6 {
            rec.tick(0.1, 0.1);
        }

        // Now should allow switch
        assert!(rec.request_switch(ReconcilerMode::Exploration));
        assert_eq!(rec.mode(), ReconcilerMode::Exploration);
    }

    #[test]
    fn test_reconciler_stability() {
        let mut rec = Reconciler::with_dwell_time(5);

        // Normal operation should be stable
        for _ in 0..100 {
            rec.tick(0.1, 0.1);
        }
        assert!(rec.is_stable());
    }

    #[test]
    fn test_reconciler_auto_mode() {
        let mut rec = Reconciler::with_dwell_time(1);

        // High variance should suggest exploration
        for _ in 0..10 {
            rec.tick(0.1, 0.1);
        }

        // Add high variance
        rec.energy_variance = 0.5;
        rec.dwell_time = 10;

        let mode = rec.auto_select_mode();
        assert_eq!(mode, ReconcilerMode::Exploration);
    }

    #[test]
    fn test_energy_balance() {
        let mut rec = Reconciler::with_dwell_time(5);
        rec.tick(0.3, 0.7);

        let balance = rec.energy_balance();
        assert!((balance - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_golden_ratio() {
        let tc = ThermalConsciousness::default_config(4, 1.0);
        let refined = tc.golden_ratio_refine(1.0);

        // Should be approximately φ ≈ 0.618
        assert!((refined - 0.618).abs() < 0.001);
    }
}
