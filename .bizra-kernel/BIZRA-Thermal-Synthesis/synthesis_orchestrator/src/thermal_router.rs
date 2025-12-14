//! Thermal Thompson Sampling Router
//!
//! Integrates Thompson sampling with thermal computing principles:
//! - Boltzmann exploration with temperature-controlled annealing
//! - Golden ratio (Φ) cooling schedules for optimal convergence
//! - Energy-based model selection

use crate::types::*;
use rand::Rng;
use rand_distr::{Beta, Distribution};
use std::collections::HashMap;

/// Thermal Thompson Sampling Router
/// 
/// Routes tasks to models using thermally-enhanced Thompson sampling:
/// - High temperature (T→1.0): Broad exploration of model space
/// - Low temperature (T→0.0): Exploitation of best-known models
pub struct ThermalRouter {
    /// Per-model win rate statistics
    model_statistics: HashMap<String, WinRate>,
    
    /// Current system temperature
    temperature: f32,
    
    /// Cooling schedule
    cooling_schedule: CoolingSchedule,
    
    /// Minimum temperature floor
    temperature_floor: f32,
    
    /// Exploration bonus factor
    exploration_factor: f32,
}

impl ThermalRouter {
    /// Create new thermal router with initial temperature
    pub fn new(initial_temperature: f32) -> Self {
        Self {
            model_statistics: HashMap::new(),
            temperature: initial_temperature,
            cooling_schedule: CoolingSchedule::GoldenRatio { t: 0.0 },
            temperature_floor: 0.1,
            exploration_factor: 0.1,
        }
    }
    
    /// Set current temperature
    pub fn set_temperature(&mut self, temperature: f32) {
        self.temperature = temperature.max(self.temperature_floor);
    }
    
    /// Route task to models using thermal Thompson sampling
    pub fn route_thermal(
        &mut self,
        _task: &Task,
        available_models: &[String],
    ) -> Result<Vec<Route>, SynthesisError> {
        let mut routes = Vec::new();
        
        for model in available_models {
            // Get or initialize win rate
            let win_rate = self.model_statistics
                .entry(model.clone())
                .or_insert_with(WinRate::default)
                .clone();
            
            // Sample Thompson score with thermal annealing
            let thompson_score = self.sample_thermal_thompson(win_rate);
            
            routes.push(Route {
                model: model.clone(),
                win_rate: self.model_statistics[model].clone(),
                thompson_score,
            });
        }
        
        // Sort by Thompson score (descending)
        routes.sort_by(|a, b| 
            b.thompson_score.partial_cmp(&a.thompson_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        );
        
        Ok(routes)
    }
    
    /// Sample Thompson score with thermal annealing
    /// 
    /// Integrates:
    /// - Beta distribution Thompson sampling
    /// - Thermal noise injection (temperature-dependent)
    /// - Boltzmann weighting: P ∝ exp(-E/kT)
    fn sample_thermal_thompson(&self, win_rate: WinRate) -> f32 {
        // Handle cold start with exploration bonus
        if win_rate.samples == 0 {
            return 0.5 + self.exploration_factor * self.temperature;
        }
        
        // Beta distribution parameters with thermal noise
        let alpha = (win_rate.wins as f64) + (self.temperature as f64);
        let beta_param = ((win_rate.samples - win_rate.wins) as f64) 
            + (self.temperature as f64);
        
        // Sample from Beta distribution
        let beta_dist = Beta::new(alpha, beta_param)
            .expect("valid beta parameters");
        let sample = beta_dist.sample(&mut rand::thread_rng()) as f32;
        
        // Apply Boltzmann weighting
        self.apply_boltzmann_weight(sample)
    }
    
    /// Apply Boltzmann weight: P ∝ exp(-E/kT)
    /// 
    /// Energy E = 1 - score (lower score = higher energy)
    /// Higher temperature → more uniform distribution
    /// Lower temperature → concentrate on high scores
    fn apply_boltzmann_weight(&self, score: f32) -> f32 {
        let energy = 1.0 - score;
        let boltzmann_factor = (-energy / (BOLTZMANN_K * self.temperature)).exp();
