// src/apex/learning.rs - SONA Self-Learning Bridge
//
// Implements the learning loop: Execute -> Evaluate -> Extract -> Update -> Optimize
// Connects to SAPE probes for feedback and triggers pattern elevation when
// occurrences > 3 (matching SAPE's ELEVATION_THRESHOLD).
//
// Integration:
// - SAPE probes for quality feedback (src/sape.rs)
// - Thompson Sampling router for Bayesian updates
// - Ihsan scoring for holistic evaluation

use crate::ihsan;
use crate::sape::{
    get_sape, ProbeDimension, ProbeResult, SAPEEngine, SnrTier, ELEVATION_THRESHOLD,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tracing::{debug, info, instrument, warn};

use super::router::ThompsonSamplingRouter;
use super::{ApexError, ApexResult};

/// Performance record for a single execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRecord {
    /// Unique record ID
    pub record_id: String,
    /// Agent that executed the task
    pub agent_id: String,
    /// Task identifier (hashed)
    pub task_id: String,
    /// Task content summary
    pub task_summary: String,
    /// Execution timestamp (Unix millis)
    pub timestamp: u64,
    /// Execution latency
    pub latency_ms: u64,
    /// Whether execution succeeded
    pub success: bool,
    /// SAPE probe results
    pub probe_results: Vec<ProbeResult>,
    /// Ihsan score achieved
    pub ihsan_score: f64,
    /// SNR tier classification
    pub snr_tier: SnrTier,
    /// Extracted patterns (for elevation)
    pub patterns: Vec<String>,
    /// Error message if failed
    pub error: Option<String>,
}

impl PerformanceRecord {
    /// Calculate aggregate quality score from probe results
    pub fn quality_score(&self) -> f64 {
        if self.probe_results.is_empty() {
            return self.ihsan_score;
        }

        // Weighted average of probe scores
        let total_weight: f64 = self
            .probe_results
            .iter()
            .map(|r| r.dimension.weight())
            .sum();
        let weighted_sum: f64 = self
            .probe_results
            .iter()
            .map(|r| r.score * r.dimension.weight())
            .sum();

        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            self.ihsan_score
        }
    }

    /// Get the reward signal for Thompson Sampling
    pub fn reward(&self) -> f64 {
        if !self.success {
            return 0.0;
        }

        // Combine quality and speed
        let quality = self.quality_score();
        let speed_bonus = if self.latency_ms < 1000 {
            0.1
        } else if self.latency_ms < 5000 {
            0.05
        } else {
            0.0
        };

        (quality + speed_bonus).clamp(0.0, 1.0)
    }
}

/// Pattern occurrence tracker for elevation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternTracker {
    /// Pattern identifier
    pub pattern: String,
    /// Number of occurrences
    pub occurrences: usize,
    /// Success rate when pattern is present
    pub success_rate: f64,
    /// Average quality score
    pub avg_quality: f64,
    /// First seen timestamp
    pub first_seen: u64,
    /// Last seen timestamp
    pub last_seen: u64,
    /// Whether pattern has been elevated
    pub elevated: bool,
}

impl PatternTracker {
    fn new(pattern: &str) -> Self {
        let now = current_timestamp_millis();
        Self {
            pattern: pattern.to_string(),
            occurrences: 1,
            success_rate: 1.0,
            avg_quality: 1.0,
            first_seen: now,
            last_seen: now,
            elevated: false,
        }
    }

    fn update(&mut self, success: bool, quality: f64) {
        self.occurrences += 1;

        // Rolling average for success rate
        let n = self.occurrences as f64;
        self.success_rate = ((n - 1.0) * self.success_rate + if success { 1.0 } else { 0.0 }) / n;
        self.avg_quality = ((n - 1.0) * self.avg_quality + quality) / n;
        self.last_seen = current_timestamp_millis();
    }

    /// Check if pattern should be elevated (>3 occurrences with good performance)
    fn should_elevate(&self) -> bool {
        !self.elevated
            && self.occurrences > ELEVATION_THRESHOLD
            && self.success_rate > 0.8
            && self.avg_quality > 0.7
    }
}

/// Learning statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningStats {
    pub total_records: usize,
    pub successful_records: usize,
    pub avg_quality: f64,
    pub avg_latency_ms: f64,
    pub patterns_tracked: usize,
    pub patterns_elevated: usize,
    pub top_patterns: Vec<(String, usize)>,
    pub tier_distribution: HashMap<String, usize>,
}

/// SONA Learning Loop - Execute -> Evaluate -> Extract -> Update -> Optimize
pub struct LearningLoop {
    /// Performance records history
    records: RwLock<VecDeque<PerformanceRecord>>,
    /// Pattern trackers
    patterns: RwLock<HashMap<String, PatternTracker>>,
    /// Reference to Thompson Sampling router for updates
    router: Option<Arc<ThompsonSamplingRouter>>,
    /// Maximum history size
    max_history: usize,
    /// SAPE engine reference for probe execution and elevation
    sape: Arc<std::sync::Mutex<SAPEEngine>>,
}

impl LearningLoop {
    /// Create a new learning loop
    pub fn new() -> Self {
        info!("📚 Initializing SONA Learning Loop");
        Self {
            records: RwLock::new(VecDeque::with_capacity(1000)),
            patterns: RwLock::new(HashMap::new()),
            router: None,
            max_history: 1000,
            sape: get_sape(),
        }
    }

    /// Create with router integration for Bayesian updates
    pub fn with_router(router: Arc<ThompsonSamplingRouter>) -> Self {
        let mut loop_instance = Self::new();
        loop_instance.router = Some(router);
        loop_instance
    }

    /// Execute the full learning cycle for a completed task
    ///
    /// Learning Loop Phases:
    /// 1. Execute: Task already executed, we receive results
    /// 2. Evaluate: Run SAPE probes on the output
    /// 3. Extract: Identify patterns from the execution
    /// 4. Update: Update Thompson Sampling posteriors
    /// 5. Optimize: Trigger pattern elevation if threshold met
    #[instrument(skip(self, output))]
    pub fn learn(
        &self,
        agent_id: &str,
        task_id: &str,
        task_summary: &str,
        output: &str,
        latency: Duration,
        success: bool,
        error: Option<String>,
    ) -> ApexResult<PerformanceRecord> {
        let start = Instant::now();

        // Phase 2: Evaluate - Run SAPE probes
        let probe_results = {
            let mut sape = self.sape.lock().map_err(|e| ApexError::LearningError {
                message: format!("Failed to acquire SAPE lock: {}", e),
            })?;
            sape.execute_probes(output)
        };

        // Calculate Ihsan score
        let ihsan_scores = self.probe_results_to_ihsan(&probe_results);
        let ihsan_score = ihsan::score(&ihsan_scores).unwrap_or(0.0);

        // Classify SNR tier
        let snr_tier = SnrTier::from_ihsan_score(ihsan_score);

        // Phase 3: Extract - Identify patterns
        let patterns = self.extract_patterns(task_summary, output, &probe_results);

        // Create performance record
        let record = PerformanceRecord {
            record_id: generate_record_id(),
            agent_id: agent_id.to_string(),
            task_id: task_id.to_string(),
            task_summary: truncate(task_summary, 100),
            timestamp: current_timestamp_millis(),
            latency_ms: latency.as_millis() as u64,
            success,
            probe_results,
            ihsan_score,
            snr_tier,
            patterns: patterns.clone(),
            error,
        };

        // Phase 4: Update - Thompson Sampling posteriors
        if let Some(ref router) = self.router {
            let reward = record.reward();
            if let Err(e) = router.update(agent_id, success, reward) {
                warn!(error = %e, "Failed to update router posteriors");
            }
        }

        // Store record
        self.store_record(record.clone())?;

        // Phase 5: Optimize - Track patterns and trigger elevation
        self.update_patterns(&patterns, success, record.quality_score())?;

        debug!(
            agent = %agent_id,
            ihsan = ihsan_score,
            snr_tier = ?snr_tier,
            patterns = patterns.len(),
            learning_latency_ms = start.elapsed().as_millis(),
            "Learning cycle completed"
        );

        Ok(record)
    }

    /// Convert probe results to Ihsan dimension scores
    fn probe_results_to_ihsan(&self, probes: &[ProbeResult]) -> BTreeMap<String, f64> {
        let mut scores = BTreeMap::new();

        // Map probe dimensions to Ihsan dimensions
        for probe in probes {
            let dimension = match probe.dimension {
                ProbeDimension::Correctness => "correctness",
                ProbeDimension::Safety | ProbeDimension::ThreatScan => "safety",
                ProbeDimension::UserBenefit => "user_benefit",
                ProbeDimension::Relevance => "efficiency",
                ProbeDimension::ComplianceCheck => "auditability",
                ProbeDimension::BiasProbe => "adl_fairness",
                ProbeDimension::Groundedness => "robustness",
                ProbeDimension::Fluency => "anti_centralization",
            };

            // Take the max if multiple probes map to same dimension
            let current = scores.entry(dimension.to_string()).or_insert(0.0_f64);
            *current = (*current).max(probe.score);
        }

        // Fill in missing dimensions with neutral values
        let required = [
            "correctness",
            "safety",
            "user_benefit",
            "efficiency",
            "auditability",
            "anti_centralization",
            "robustness",
            "adl_fairness",
        ];
        for dim in required {
            scores.entry(dim.to_string()).or_insert(0.7);
        }

        scores
    }

    /// Extract patterns from execution for tracking
    fn extract_patterns(&self, task: &str, output: &str, probes: &[ProbeResult]) -> Vec<String> {
        let mut patterns = Vec::new();

        // Task type patterns
        let task_lower = task.to_lowercase();
        if task_lower.contains("security") || task_lower.contains("verify") {
            patterns.push("task:security_verification".to_string());
        }
        if task_lower.contains("generate") || task_lower.contains("create") {
            patterns.push("task:content_generation".to_string());
        }
        if task_lower.contains("analyze") || task_lower.contains("review") {
            patterns.push("task:analysis".to_string());
        }

        // Probe result patterns
        let high_quality_count = probes.iter().filter(|p| p.score > 0.9).count();
        if high_quality_count >= 7 {
            patterns.push("quality:excellent".to_string());
        } else if high_quality_count >= 5 {
            patterns.push("quality:good".to_string());
        }

        // Flag patterns
        for probe in probes {
            for flag in &probe.flags {
                patterns.push(format!("flag:{}", flag));
            }
        }

        // Output length pattern
        let output_tokens = (output.len() + 3) / 4;
        if output_tokens < 100 {
            patterns.push("output:short".to_string());
        } else if output_tokens < 500 {
            patterns.push("output:medium".to_string());
        } else {
            patterns.push("output:long".to_string());
        }

        patterns
    }

    /// Store a performance record
    fn store_record(&self, record: PerformanceRecord) -> ApexResult<()> {
        let mut records = self.records.write().map_err(|e| ApexError::LearningError {
            message: format!("Failed to acquire records lock: {}", e),
        })?;

        // Evict oldest if at capacity
        while records.len() >= self.max_history {
            records.pop_front();
        }

        records.push_back(record);
        Ok(())
    }

    /// Update pattern trackers and trigger elevation if needed
    fn update_patterns(&self, patterns: &[String], success: bool, quality: f64) -> ApexResult<()> {
        let mut trackers = self
            .patterns
            .write()
            .map_err(|e| ApexError::LearningError {
                message: format!("Failed to acquire patterns lock: {}", e),
            })?;

        let mut to_elevate = Vec::new();

        for pattern in patterns {
            let tracker = trackers
                .entry(pattern.clone())
                .or_insert_with(|| PatternTracker::new(pattern));

            tracker.update(success, quality);

            if tracker.should_elevate() {
                to_elevate.push(pattern.clone());
                tracker.elevated = true;
            }
        }

        // Trigger SAPE elevation for qualified patterns
        drop(trackers); // Release lock before elevating

        for pattern in to_elevate {
            self.elevate_pattern(&pattern)?;
        }

        Ok(())
    }

    /// Elevate a pattern to SAPE kernel shortcut
    fn elevate_pattern(&self, pattern: &str) -> ApexResult<()> {
        info!(pattern = %pattern, "📈 Elevating pattern to SAPE kernel");

        // The SAPE engine handles the actual elevation
        // We notify it that this pattern should be compiled into a shortcut
        if let Ok(mut sape) = self.sape.lock() {
            // Add as threat concept if it's a security pattern
            if pattern.starts_with("flag:") && pattern.contains("threat") {
                let concept = pattern.replace("flag:", "").replace("_", " ");
                if let Err(e) = sape.add_threat_concept(concept) {
                    warn!(error = %e, "Failed to add threat concept");
                }
            }
        }

        Ok(())
    }

    /// Get learning statistics
    pub fn get_stats(&self) -> ApexResult<LearningStats> {
        let records = self.records.read().map_err(|e| ApexError::LearningError {
            message: format!("Failed to read records: {}", e),
        })?;

        let patterns = self.patterns.read().map_err(|e| ApexError::LearningError {
            message: format!("Failed to read patterns: {}", e),
        })?;

        let total_records = records.len();
        let successful_records = records.iter().filter(|r| r.success).count();

        let avg_quality = if total_records > 0 {
            records.iter().map(|r| r.quality_score()).sum::<f64>() / total_records as f64
        } else {
            0.0
        };

        let avg_latency = if total_records > 0 {
            records.iter().map(|r| r.latency_ms as f64).sum::<f64>() / total_records as f64
        } else {
            0.0
        };

        let patterns_elevated = patterns.values().filter(|p| p.elevated).count();

        // Top patterns by occurrence
        let mut top_patterns: Vec<_> = patterns
            .iter()
            .map(|(k, v)| (k.clone(), v.occurrences))
            .collect();
        top_patterns.sort_by(|a, b| b.1.cmp(&a.1));
        top_patterns.truncate(10);

        // SNR tier distribution
        let mut tier_distribution: HashMap<String, usize> = HashMap::new();
        for record in records.iter() {
            *tier_distribution
                .entry(record.snr_tier.name().to_string())
                .or_insert(0) += 1;
        }

        Ok(LearningStats {
            total_records,
            successful_records,
            avg_quality,
            avg_latency_ms: avg_latency,
            patterns_tracked: patterns.len(),
            patterns_elevated,
            top_patterns,
            tier_distribution,
        })
    }

    /// Get recent performance records
    pub fn recent_records(&self, limit: usize) -> Vec<PerformanceRecord> {
        self.records
            .read()
            .map(|r| r.iter().rev().take(limit).cloned().collect())
            .unwrap_or_default()
    }

    /// Get pattern tracker by pattern name
    pub fn get_pattern(&self, pattern: &str) -> Option<PatternTracker> {
        self.patterns.read().ok()?.get(pattern).cloned()
    }
}

impl Default for LearningLoop {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate unique record ID
fn generate_record_id() -> String {
    format!(
        "REC-{}-{:06x}",
        chrono::Utc::now().format("%Y%m%d%H%M%S"),
        rand::random::<u32>() & 0xFFFFFF
    )
}

/// Get current timestamp in milliseconds
fn current_timestamp_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Truncate string to max length
fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_record_quality_score() {
        let record = PerformanceRecord {
            record_id: "test".to_string(),
            agent_id: "agent_1".to_string(),
            task_id: "task_1".to_string(),
            task_summary: "Test task".to_string(),
            timestamp: 0,
            latency_ms: 100,
            success: true,
            probe_results: vec![],
            ihsan_score: 0.95,
            snr_tier: SnrTier::T5,
            patterns: vec![],
            error: None,
        };

        // With no probe results, should return ihsan_score
        assert!((record.quality_score() - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_performance_record_reward() {
        let mut record = PerformanceRecord {
            record_id: "test".to_string(),
            agent_id: "agent_1".to_string(),
            task_id: "task_1".to_string(),
            task_summary: "Test task".to_string(),
            timestamp: 0,
            latency_ms: 500,
            success: true,
            probe_results: vec![],
            ihsan_score: 0.9,
            snr_tier: SnrTier::T5,
            patterns: vec![],
            error: None,
        };

        // Successful with fast latency should get speed bonus
        let reward = record.reward();
        assert!(reward > 0.9);

        // Failure should return 0
        record.success = false;
        assert_eq!(record.reward(), 0.0);
    }

    #[test]
    fn test_pattern_tracker_should_elevate() {
        let mut tracker = PatternTracker::new("test_pattern");

        // Not enough occurrences
        assert!(!tracker.should_elevate());

        // Add more occurrences with good performance
        for _ in 0..5 {
            tracker.update(true, 0.9);
        }

        // Should now be ready for elevation
        assert!(tracker.should_elevate());

        // After elevation, should not re-elevate
        tracker.elevated = true;
        assert!(!tracker.should_elevate());
    }

    #[test]
    fn test_learning_loop_creation() {
        let loop_instance = LearningLoop::new();
        let stats = loop_instance.get_stats().unwrap();

        assert_eq!(stats.total_records, 0);
        assert_eq!(stats.patterns_tracked, 0);
    }

    #[test]
    fn test_pattern_extraction() {
        let loop_instance = LearningLoop::new();

        let patterns = loop_instance.extract_patterns(
            "Security verification of the system",
            "The system passed all security checks.",
            &[],
        );

        assert!(patterns.iter().any(|p| p.contains("security")));
    }
}
