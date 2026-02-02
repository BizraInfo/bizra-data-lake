//! Configuration for BIZRA Hunter

use crate::{LANE1_SNR_THRESHOLD, MIN_CONSISTENT_AXES};

#[derive(Debug, Clone)]
pub struct HunterConfig {
    /// Pipeline capacity (lane1 queue size)
    pub pipeline_capacity: usize,
    /// Lane1 SNR threshold
    pub snr_threshold: f32,
    /// Minimum consistent axes
    pub min_axes: usize,
    /// Loop sleep millis (health loop)
    pub loop_sleep_ms: u64,
}

impl Default for HunterConfig {
    fn default() -> Self {
        Self {
            pipeline_capacity: 65_536,
            snr_threshold: LANE1_SNR_THRESHOLD,
            min_axes: MIN_CONSISTENT_AXES,
            loop_sleep_ms: 25,
        }
    }
}
