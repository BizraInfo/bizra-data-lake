// src/bridge.rs - PAT-SAT Bridge Coordinator

use crate::{
    pat::PATOrchestrator,
    sat::SATOrchestrator,
    types::{AgentResult, DualAgenticRequest, DualAgenticResponse},
};
use std::time::Instant;
use tracing::{info, instrument};

/// Bridge coordinator between PAT and SAT
pub struct BridgeCoordinator {
    pat: PATOrchestrator,
    sat: SATOrchestrator,
}

impl BridgeCoordinator {
    pub async fn new() -> anyhow::Result<Self> {
        info!("🌉 Initializing PAT-SAT Bridge Coordinator");
        
        let pat = PATOrchestrator::new().await?;
        let sat = SATOrchestrator::new().await?;
        
        Ok(Self { pat, sat })
    }
    
    /// Execute full dual-agentic workflow
    #[instrument(skip(self))]
    pub async fn execute(
        &self,
        request: DualAgenticRequest,
    ) -> anyhow::Result<DualAgenticResponse> {
        let start = Instant::now();
        
        info!("🚀 Starting dual-agentic execution");
        
        // Step 1: SAT validates the request
        let validation = self.sat.validate_request(&request).await?;
        
        if !validation.consensus_reached {
            return Err(anyhow::anyhow!("SAT consensus not reached for request validation"));
        }
        
        info!(
            validation_time_ms = validation.validation_time.as_millis(),
            "SAT validation passed"
        );
        
        // Step 2: PAT executes the task
        let pat_results = self.pat.execute_parallel(vec![], request.clone()).await?;
        
        info!(
            pat_agents = pat_results.len(),
            "PAT execution completed"
        );
        
        // Step 3: SAT evaluates PAT results
        let sat_evaluations = self.sat.evaluate_results(&pat_results).await?;
        
        info!(
            sat_evaluations = sat_evaluations.len(),
            "SAT evaluation completed"
        );
        
        // Step 4: Calculate synergy and quality scores
        let synergy_score = self.calculate_synergy(&pat_results, &sat_evaluations);
        let ihsan_score = self.calculate_ihsan(&pat_results);
        
        let total_latency = start.elapsed();
        
        info!(
            synergy = synergy_score,
            ihsan = ihsan_score,
            latency_ms = total_latency.as_millis(),
            "Dual-agentic execution completed"
        );
        
        Ok(DualAgenticResponse {
            pat_contributions: pat_results.iter()
                .map(|r| r.contribution.clone())
                .collect(),
            sat_contributions: sat_evaluations.iter()
                .map(|r| r.contribution.clone())
                .collect(),
            synergy_score,
            ihsan_score,
            latency: total_latency,
            meta: serde_json::json!({
                "pat_agents": self.pat.get_agent_count(),
                "sat_agents": self.sat.get_agent_count(),
                "validation_time_ms": validation.validation_time.as_millis(),
            }),
        })
    }
    
    /// Calculate synergy between PAT and SAT
    fn calculate_synergy(&self, pat_results: &[AgentResult], sat_results: &[AgentResult]) -> f64 {
        let pat_avg = pat_results.iter()
            .map(|r| r.confidence)
            .sum::<f64>() / pat_results.len() as f64;
        
        let sat_avg = sat_results.iter()
            .map(|r| r.confidence)
            .sum::<f64>() / sat_results.len() as f64;
        
        // Harmonic mean for synergy
        2.0 * pat_avg * sat_avg / (pat_avg + sat_avg)
    }
    
    /// Calculate إحسان (excellence) score
    fn calculate_ihsan(&self, results: &[AgentResult]) -> f64 {
        let avg_confidence = results.iter()
            .map(|r| r.confidence)
            .sum::<f64>() / results.len() as f64;
        
        let consistency = self.calculate_consistency(results);
        
        // إحسان = (confidence + consistency) / 2
        (avg_confidence + consistency) / 2.0
    }
    
    fn calculate_consistency(&self, results: &[AgentResult]) -> f64 {
        if results.is_empty() {
            return 0.0;
        }
        
        let mean = results.iter()
            .map(|r| r.confidence)
            .sum::<f64>() / results.len() as f64;
        
        let variance = results.iter()
            .map(|r| (r.confidence - mean).powi(2))
            .sum::<f64>() / results.len() as f64;
        
        // High consistency = low variance
        1.0 - variance.sqrt()
    }
}
