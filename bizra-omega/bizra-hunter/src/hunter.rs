//! Core Hunter Runtime
//!
//! Orchestrates the full vulnerability discovery pipeline:
//!
//! ```text
//! BytecodeSource → EVM Decode → Entropy Filter → Sequence Detect
//!                                 (Lane 1)           (Lane 2)
//!                                                       ↓
//!                                              PoC Generate → Finding
//! ```
//!
//! The Hunter is the conductor — it owns the pipeline, entropy calculator,
//! and drives contracts from ingestion through to actionable findings.

use std::time::{Duration, Instant};

use crate::{
    cascade::GateType,
    config::HunterConfig,
    entropy::EntropyCalculator,
    ingestion::BytecodeSource,
    pipeline::{Complexity, PipelineStats, SNRPipeline, VulnType},
    poc::SafePoC,
    submission::{BondedSubmission, SubmissionResult},
};

/// A complete vulnerability finding produced by the pipeline.
#[derive(Debug, Clone)]
pub struct Finding {
    /// Contract address
    pub contract_addr: [u8; 20],
    /// Detected vulnerability type
    pub vuln_type: VulnType,
    /// Bytecode offset where the vulnerability pattern starts
    pub location: u32,
    /// Complexity tier
    pub complexity: Complexity,
    /// Multi-axis entropy scores
    pub entropy: crate::entropy::MultiAxisEntropy,
    /// Estimated bounty (USD cents)
    pub bounty_estimate: u64,
    /// Generated safe PoC (if address is a valid 0x-prefixed hex string)
    pub poc: Option<String>,
    /// Submission validation result
    pub submission: Option<SubmissionResult>,
}

impl Finding {
    /// Format the contract address as a 0x-prefixed hex string.
    pub fn address_hex(&self) -> String {
        format!("0x{}", hex::encode(self.contract_addr))
    }
}

/// Aggregate scan results.
#[derive(Debug, Clone)]
pub struct HunterResult {
    pub lane1_processed: u64,
    pub lane1_filtered: u64,
    pub lane2_submitted: u64,
    pub duplicates_filtered: u64,
    pub cascade_blocked: u64,
    /// Number of actionable findings produced.
    pub findings_count: u64,
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
            findings_count: 0,
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

    /// Health check: all cascade gates open.
    pub fn health_check(&self) -> bool {
        self.pipeline.cascade.is_open(GateType::Technical)
            && self.pipeline.cascade.is_open(GateType::Ethics)
            && self.pipeline.cascade.is_open(GateType::Legal)
    }

    // ─── Core Scanning API ─────────────────────────────────────────────

    /// Scan a single contract through the full pipeline.
    ///
    /// Returns `Some(Finding)` if the contract passes all gates and
    /// exhibits a detectable vulnerability pattern. Returns `None` if
    /// filtered by SNR threshold, deduplication, or cascade gates.
    pub fn scan_one(&mut self, address: [u8; 20], bytecode: &[u8]) -> Option<Finding> {
        // Lane 1: fast heuristic filter
        let heuristic = self
            .pipeline
            .process_lane1(address, bytecode, &mut self.entropy)?;

        // Lane 2: detailed analysis — create proof job
        let proof_job = self
            .pipeline
            .create_proof_job(&heuristic, bytecode.to_vec());

        // Update stats
        self.pipeline
            .stats
            .lane2_processed
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Generate safe PoC
        let addr_hex = format!("0x{}", hex::encode(address));
        let poc = SafePoC::generate(proof_job.vuln_type, &addr_hex, proof_job.location).ok();

        // Validate submission
        let submission = poc.as_ref().map(|poc_code| {
            let sub = BondedSubmission {
                contract_addr: address,
                vuln_type: proof_job.vuln_type,
                bond_cents: proof_job.bounty_estimate / 10, // 10% bond
                poc: poc_code.clone(),
            };
            let result = sub.validate();
            if result.accepted {
                self.pipeline
                    .stats
                    .lane2_submitted
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            result
        });

        self.last_tick = Instant::now();

        Some(Finding {
            contract_addr: address,
            vuln_type: proof_job.vuln_type,
            location: proof_job.location,
            complexity: heuristic.complexity,
            entropy: heuristic.entropy,
            bounty_estimate: heuristic.bounty_estimate,
            poc,
            submission,
        })
    }

    /// Scan contracts from a `BytecodeSource` through the full pipeline.
    ///
    /// Drains the source, processes each contract through Lane 1 → Lane 2,
    /// and returns all findings sorted by bounty estimate (descending).
    pub fn scan(&mut self, source: &mut dyn BytecodeSource) -> Vec<Finding> {
        let mut findings = Vec::new();

        tracing::info!(source = source.describe(), "Starting scan");

        let contracts = source.drain();
        let total = contracts.len();

        for contract in &contracts {
            if let Some(finding) = self.scan_one(contract.address, &contract.bytecode) {
                tracing::info!(
                    addr = %finding.address_hex(),
                    vuln = ?finding.vuln_type,
                    bounty = finding.bounty_estimate,
                    offset = finding.location,
                    "Finding detected"
                );
                findings.push(finding);
            }
        }

        // Sort by bounty estimate (highest first)
        findings.sort_by(|a, b| b.bounty_estimate.cmp(&a.bounty_estimate));

        tracing::info!(
            total_contracts = total,
            findings = findings.len(),
            "Scan complete"
        );

        findings
    }

    /// Legacy health-loop for backward compatibility.
    /// Runs `iterations` ticks with no external input.
    pub fn run_loop(&mut self, iterations: u32) -> HunterResult {
        for _ in 0..iterations {
            std::thread::sleep(Duration::from_millis(self.config.loop_sleep_ms));
            self.last_tick = Instant::now();
        }
        HunterResult::from(&self.pipeline.stats)
    }

    /// Get a snapshot of pipeline statistics.
    pub fn stats(&self) -> HunterResult {
        HunterResult::from(&self.pipeline.stats)
    }
}
