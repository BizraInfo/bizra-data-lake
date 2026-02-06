//! Core Hunter runtime loop

use std::time::{Duration, Instant};

use crate::config::HunterConfig;
use crate::entropy::EntropyCalculator;
use crate::pipeline::{PipelineStats, SNRPipeline};

#[derive(Debug, Clone)]
pub struct HunterResult {
    pub lane1_processed: u64,
    pub lane1_filtered: u64,
    pub lane2_submitted: u64,
    pub duplicates_filtered: u64,
    pub cascade_blocked: u64,
}

impl From<&PipelineStats> for HunterResult {
    fn from(s: &PipelineStats) -> Self {
        Self {
            lane1_processed: s.lane1_processed.load(std::sync::atomic::Ordering::Relaxed),
            lane1_filtered: s.lane1_filtered.load(std::sync::atomic::Ordering::Relaxed),
            lane2_submitted: s.lane2_submitted.load(std::sync::atomic::Ordering::Relaxed),
            duplicates_filtered: s
                .duplicates_filtered
                .load(std::sync::atomic::Ordering::Relaxed),
            cascade_blocked: s.cascade_blocked.load(std::sync::atomic::Ordering::Relaxed),
        }
    }
}

pub struct Hunter<const N: usize> {
    pub config: HunterConfig,
    pub pipeline: SNRPipeline<N>,
    pub entropy: EntropyCalculator,
    last_tick: Instant,
}

impl<const N: usize> Hunter<N> {
    pub fn new(config: HunterConfig) -> Self {
        let pipeline =
            SNRPipeline::<N>::new().with_snr_config(config.snr_threshold, config.min_axes);
        let entropy = EntropyCalculator::new();
        Self {
            config,
            pipeline,
            entropy,
            last_tick: Instant::now(),
        }
    }

    /// Health check: pipeline gates open + recent tick
    pub fn health_check(&self) -> bool {
        
        self
            .pipeline
            .cascade
            .is_open(crate::cascade::GateType::Technical)
            && self
                .pipeline
                .cascade
                .is_open(crate::cascade::GateType::Ethics)
            && self
                .pipeline
                .cascade
                .is_open(crate::cascade::GateType::Legal)
    }

    /// Run a lightweight loop (no external input). Returns stats snapshot.
    pub fn run_loop(&mut self, iterations: u32) -> HunterResult {
        for _ in 0..iterations {
            std::thread::sleep(Duration::from_millis(self.config.loop_sleep_ms));
            self.last_tick = Instant::now();
        }
        HunterResult::from(&self.pipeline.stats)
    }
}
