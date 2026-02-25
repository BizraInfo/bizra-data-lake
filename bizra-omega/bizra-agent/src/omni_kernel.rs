// bizra-agent/src/omni_kernel.rs
// ============================================================
// BIZRA Ghost Reign Omni-Kernel — 8-Line Sovereign Loop
// ============================================================
//
// Wires all existing BIZRA infrastructure into one coherent cycle:
//
//   EVENT BUS  →  HHMM cortex (intent)
//              →  Chain of Reasoning (decision pivots)
//              →  Tier-1 Reflex Cache   (O(1) system hit)
//              →  Tier-2 Engram Cache   (O(1) model fact hit)
//              →  PAT Agents            (GPU inference, miss path)
//              →  ConstitutionalGate    (İhsān ≥ 0.95)
//              →  SSO stability check   (after TTRL update)
//              →  Receipt Chain append
//              →  Metabolic Ledger      (PoI yield + SEED emission)
//              →  TTRL queue_update     (self-improvement signal)
//              →  EVENT BUS emit cycle_complete
//
// ## The 8-Line Mapping (from the four-paper analysis)
//   Line 1: hhmm_cortex.infer_macro_state()     ← Chain of Reasoning
//   Line 2: reflex_ledger.get(&state_hash)       ← Engram + Tier-1 cache
//   Line 3: context_membrane.distill()           ← RLM small-model filter
//   Line 4: pat_agents.run_parallel()            ← PAT execution
//   Line 5: verify_ihsan_bounds() >= 0.95        ← SSO stability
//   Line 6: receipt_chain.append(receipt)        ← Receipt chain
//   Line 7: metabolic_ledger.mint_poi_yield()    ← TTRL emission decay
//   Line 8: event_bus.emit("cycle_complete")     ← Close the OODA loop
//
// Standing on Giants:
//   Boyd (1976): OODA loop
//   Shannon (1948): SNR / information theory
//   Al-Ghazali (1095): İhsān gate as excellence requirement
//   TTRL paper (2025): on-device self-improvement
//   SSO paper (2025): spectral-sphere stability

use bizra_ttrl::{
    decision_pivot::{HhmmLevel, ReasoningChain, PIVOT_IHSAN_DEFAULT},
    engram::{EngramCache, EngramResult},
    metabolic_ledger::{MetabolicLedger, PoiYield},
    sso::{SpectralNorm, SpectralSphereConstraint},
    ttrl_engine::TtrlEngine,
};

use crate::hash_namespace::TriggerHash;
use crate::reflex_cache::{ReflexCache, ReflexMode, BOOTSTRAP_POLICY_HASH};

// ─── Kernel configuration ────────────────────────────────────────────────────

/// Configuration for the Omni-Kernel.
/// All float thresholds are sourced from `config/proactive_config.yaml`
/// which in turn mirrors `core/integration/constants.py`.
/// **Never hard-code thresholds here.**
#[derive(Debug, Clone)]
pub struct OmniKernelConfig {
    /// Minimum İhsān to accept a completed cycle.  Default: 0.95.
    pub ihsan_threshold: f64,
    /// Minimum Engram confidence to serve a Tier-2 hit.  Default: 0.95.
    pub engram_min_confidence: f64,
    /// Base SEED per verified action (before emission decay).
    pub base_seed_per_action: f64,
    /// Number of PAT agents whose outputs form the majority vote.  Default: 7.
    pub pat_team_size: usize,
    /// SSO constraint epsilon.  Default: `SSO_DEFAULT_EPSILON`.
    pub sso_epsilon: f64,
    /// Current federation network size (updated by federation module).
    pub network_size: u64,
}

impl Default for OmniKernelConfig {
    fn default() -> Self {
        Self {
            ihsan_threshold: PIVOT_IHSAN_DEFAULT, // 0.95
            engram_min_confidence: 0.95,
            base_seed_per_action: 1.0,
            pat_team_size: 7,
            sso_epsilon: bizra_ttrl::SSO_DEFAULT_EPSILON,
            network_size: 1,
        }
    }
}

// ─── Cycle input / output ─────────────────────────────────────────────────────

/// Input to a single Omni-Kernel cycle.
#[derive(Debug, Clone)]
pub struct OmniCycle {
    /// Raw intent string (from HHMM cortex or user input).
    pub intent: String,
    /// Canonical bytes for cache keying.  `BLAKE3("omni/intent/v1:" + intent)`.
    pub intent_bytes: Vec<u8>,
    /// User identifier (opaque 32-bit hash).
    pub user_hash: u32,
    /// UNIX milliseconds.
    pub now_ms: u64,
}

impl OmniCycle {
    pub fn new(intent: impl Into<String>, user_hash: u32, now_ms: u64) -> Self {
        let intent = intent.into();
        let intent_bytes = {
            let mut h = blake3::Hasher::new();
            h.update(b"omni/intent/v1:");
            h.update(intent.as_bytes());
            h.finalize().as_bytes().to_vec()
        };
        Self {
            intent,
            intent_bytes,
            user_hash,
            now_ms,
        }
    }
}

/// Which execution path was taken in this cycle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CyclePath {
    /// Tier-1 Reflex Cache hit — compiled rule executed, no inference.
    ReflexHit,
    /// Tier-2 Engram hit — factual lookup, no GPU inference.
    EngramHit,
    /// Tier-3 full PAT inference — cache miss, GPU involved.
    FullInference,
    /// Guardian vetoed — İhsān gate did not pass.
    GuardianVeto,
    /// Chain-of-Reasoning failed at a decision pivot.
    PivotFailed { at_index: usize },
}

/// The receipt produced at the end of one Omni-Kernel cycle.
#[derive(Debug, Clone)]
pub struct CycleReceipt {
    /// The execution path taken.
    pub path: CyclePath,
    /// İhsān score of this cycle (0–1).
    pub ihsan_score: f64,
    /// The composite BLAKE3 hash of all decision pivots.
    pub pivot_chain_hash: [u8; 32],
    /// Was the İhsān gate passed?
    pub gate_passed: bool,
    /// PoI yield minted for this cycle.
    pub poi_yield: Option<PoiYield>,
    /// TTRL update queued?
    pub ttrl_queued: bool,
    /// Response / payload from this cycle.
    pub response: String,
}

// ─── The Omni-Kernel ─────────────────────────────────────────────────────────

/// The BIZRA Ghost Reign Omni-Kernel.
///
/// Owns the four mutable state machines that power the sovereign node.
/// Designed to be wrapped in `Arc<Mutex<OmniKernel>>` for async use.
pub struct OmniKernel {
    config: OmniKernelConfig,
    reflex_mode: ReflexMode,
    /// Current policy hash used to validate reflex rules.
    /// Defaults to `BOOTSTRAP_POLICY_HASH`; updated by the policy subsystem.
    policy_hash: [u8; 32],
    reflex_cache: ReflexCache,
    engram_cache: EngramCache,
    ttrl_engine: TtrlEngine,
    metabolic_ledger: MetabolicLedger,
}

impl OmniKernel {
    /// Construct a new kernel with default configuration.
    pub fn new(config: OmniKernelConfig) -> Self {
        let sso = SpectralSphereConstraint::new(config.sso_epsilon);
        let mut reflex_cache = ReflexCache::new(2048);
        reflex_cache.load_bootstrap_rules();
        Self {
            metabolic_ledger: MetabolicLedger::new(config.base_seed_per_action),
            engram_cache: EngramCache::new(),
            ttrl_engine: TtrlEngine::new(sso),
            reflex_mode: ReflexMode::Active,
            policy_hash: BOOTSTRAP_POLICY_HASH,
            reflex_cache,
            config,
        }
    }

    // ─── Line 1: HHMM / Chain of Reasoning ───────────────────────────────────

    /// Build a Chain-of-Reasoning for the given intent.
    /// In production this calls the Python HMM engine via PyO3/HTTP.
    /// Here it is modelled: L2→L3→L4 pivots with caller-supplied scores.
    pub fn build_reasoning_chain(
        &self,
        intent: &str,
        level_scores: &[(HhmmLevel, f64)],
    ) -> ReasoningChain {
        let mut chain = ReasoningChain::new();
        for (level, ihsan) in level_scores {
            if level.needs_pivot() {
                chain.push(*level, format!("Reasoning at {level:?}: {intent}"), *ihsan);
            }
        }
        chain
    }

    // ─── The Main Loop: run_cycle ────────────────────────────────────────────

    /// Execute one Omni-Kernel cycle.
    ///
    /// `pat_responses` — candidate answers from PAT agents (pass `&[]` for cache hits).
    /// `ihsan_score`   — caller-measured İhsān of the candidate response.
    /// `level_scores`  — HHMM level → İhsān scores for Chain-of-Reasoning.
    /// `pre_spectral`/`post_spectral` — model spectral norms (for SSO after TTRL).
    pub fn run_cycle(
        &mut self,
        cycle: &OmniCycle,
        pat_responses: &[String],
        ihsan_score: f64,
        level_scores: &[(HhmmLevel, f64)],
        pre_spectral: Option<SpectralNorm>,
        post_spectral: Option<SpectralNorm>,
    ) -> CycleReceipt {
        // ─── Line 1: Chain of Reasoning ──────────────────────────────────────
        let reasoning_chain = self.build_reasoning_chain(&cycle.intent, level_scores);

        // Check every pivot — fail-fast if any pivot's İhsān is below threshold.
        for pivot in reasoning_chain.decision_pivots() {
            if !pivot.passes(self.config.ihsan_threshold) {
                tracing::warn!(
                    pivot_index = pivot.index,
                    ihsan = pivot.ihsan,
                    threshold = self.config.ihsan_threshold,
                    "Omni-Kernel: decision pivot failed — early exit"
                );
                return CycleReceipt {
                    path: CyclePath::PivotFailed {
                        at_index: pivot.index,
                    },
                    ihsan_score: pivot.ihsan,
                    pivot_chain_hash: reasoning_chain.tail_hash(),
                    gate_passed: false,
                    poi_yield: None,
                    ttrl_queued: false,
                    response: String::new(),
                };
            }
        }

        // ─── Line 2: Tier-1 Reflex Cache ──────────────────────────────────────
        let state_hash = TriggerHash(*blake3::hash(&cycle.intent_bytes).as_bytes());

        if let Some(rule) = self.reflex_cache.get_active(
            self.reflex_mode,
            &state_hash,
            Some(self.policy_hash),
            cycle.now_ms,
        ) {
            tracing::debug!(route = %rule.action_template.route_signature, "Omni-Kernel: Tier-1 reflex hit");
            let poi =
                self.metabolic_ledger
                    .mint_poi_yield(true, self.config.network_size, cycle.now_ms);
            return CycleReceipt {
                path: CyclePath::ReflexHit,
                ihsan_score: rule.compile_ihsan as f64,
                pivot_chain_hash: reasoning_chain.tail_hash(),
                gate_passed: (rule.compile_ihsan as f64) >= self.config.ihsan_threshold,
                poi_yield: Some(poi),
                ttrl_queued: false,
                response: rule.action_template.route_signature.clone(),
            };
        }

        // ─── Line 2b: Tier-2 Engram Cache ─────────────────────────────────────
        match self
            .engram_cache
            .lookup(&cycle.intent_bytes, self.config.engram_min_confidence)
        {
            EngramResult::Hit { value, .. } => {
                tracing::debug!("Omni-Kernel: Tier-2 Engram hit");
                let poi = self.metabolic_ledger.mint_poi_yield(
                    true,
                    self.config.network_size,
                    cycle.now_ms,
                );
                return CycleReceipt {
                    path: CyclePath::EngramHit,
                    ihsan_score,
                    pivot_chain_hash: reasoning_chain.tail_hash(),
                    gate_passed: ihsan_score >= self.config.ihsan_threshold,
                    poi_yield: Some(poi),
                    ttrl_queued: false,
                    response: value,
                };
            }
            EngramResult::Miss => {
                tracing::debug!("Omni-Kernel: Tier-2 Engram miss — proceeding to full inference");
            }
        }

        // ─── Lines 3–4: Full PAT inference (caller-supplied responses) ────────
        // In production: PAT agents run in parallel; caller passes their outputs.
        // Here we take the first non-empty response as the synthesised answer.
        let response = pat_responses
            .iter()
            .find(|r| !r.is_empty())
            .cloned()
            .unwrap_or_default();

        // ─── Line 5: İhsān gate + SSO stability check ─────────────────────────
        if ihsan_score < self.config.ihsan_threshold {
            tracing::warn!(
                ihsan = ihsan_score,
                threshold = self.config.ihsan_threshold,
                "Omni-Kernel: Guardian veto — İhsān gate failed"
            );
            return CycleReceipt {
                path: CyclePath::GuardianVeto,
                ihsan_score,
                pivot_chain_hash: reasoning_chain.tail_hash(),
                gate_passed: false,
                poi_yield: None,
                ttrl_queued: false,
                response: String::new(),
            };
        }

        // SSO check after TTRL update (if caller provides spectral norms).
        if let (Some(pre), Some(post)) = (pre_spectral, post_spectral) {
            if self.ttrl_engine.has_pending_update() {
                let sso_result = self.ttrl_engine.apply_pending_update(pre, post);
                if !sso_result.passed() {
                    tracing::warn!(
                        drift = sso_result.drift,
                        epsilon = sso_result.epsilon,
                        "Omni-Kernel: SSO violation — caller must rollback model weights"
                    );
                }
            }
        }

        // ─── Line 6: Receipt chain (hash binds this cycle's evidence) ─────────
        // The pivot_chain_hash serves as the tamper-evident receipt.
        let pivot_chain_hash = reasoning_chain.tail_hash();

        // ─── Line 7: Metabolic Ledger → PoI yield + emission decay ───────────
        let poi =
            self.metabolic_ledger
                .mint_poi_yield(false, self.config.network_size, cycle.now_ms);

        // ─── Line 7b: TTRL queue_update (PAT majority vote → GRPO signal) ─────
        let ttrl_queued = if pat_responses.len() >= 3 {
            let intent_hash = *blake3::hash(&cycle.intent_bytes).as_bytes();
            self.ttrl_engine
                .queue_update(
                    intent_hash,
                    pat_responses,
                    ihsan_score,
                    poi.amount, // cpva_actual approximated by minted yield
                    cycle.now_ms,
                )
                .is_some()
        } else {
            false
        };

        // ─── Line 8: Emit cycle_complete via Event Bus ────────────────────────
        // In production: `event_bus.emit(Topic::CYCLE_COMPLETE, receipt_bytes)`.
        // Modelled here as a tracing span.
        tracing::info!(
            path = "FullInference",
            ihsan = ihsan_score,
            poi_amount = poi.amount,
            emission = poi.emission_multiplier,
            ttrl_queued = ttrl_queued,
            "Omni-Kernel: cycle_complete"
        );

        CycleReceipt {
            path: CyclePath::FullInference,
            ihsan_score,
            pivot_chain_hash,
            gate_passed: true,
            poi_yield: Some(poi),
            ttrl_queued,
            response,
        }
    }

    // ─── Public accessors ─────────────────────────────────────────────────────

    pub fn engram_cache_mut(&mut self) -> &mut EngramCache {
        &mut self.engram_cache
    }

    pub fn metabolic_stats(&self) -> &bizra_ttrl::metabolic_ledger::LedgerStats {
        &self.metabolic_ledger.stats
    }

    pub fn ttrl_stats(&self) -> &bizra_ttrl::ttrl_engine::TtrlStats {
        &self.ttrl_engine.stats
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use bizra_ttrl::decision_pivot::HhmmLevel;

    fn make_kernel() -> OmniKernel {
        OmniKernel::new(OmniKernelConfig::default())
    }

    fn cycle(intent: &str) -> OmniCycle {
        OmniCycle::new(intent, 0xDEAD_BEEF, 1_740_000_000_000)
    }

    #[test]
    fn test_pivot_fail_short_circuits() {
        let mut k = make_kernel();
        let c = cycle("test intent");
        let level_scores = vec![
            (HhmmLevel::L2Cognitive, 0.97),
            (HhmmLevel::L3Memory, 0.80), // below 0.95 → should fail
        ];
        let receipt = k.run_cycle(&c, &[], 0.97, &level_scores, None, None);
        assert!(matches!(receipt.path, CyclePath::PivotFailed { .. }));
        assert!(!receipt.gate_passed);
        assert!(receipt.poi_yield.is_none());
    }

    #[test]
    fn test_guardian_veto_on_low_ihsan() {
        let mut k = make_kernel();
        let c = cycle("low quality response");
        let receipt = k.run_cycle(&c, &["bad response".into()], 0.70, &[], None, None);
        assert_eq!(receipt.path, CyclePath::GuardianVeto);
        assert!(!receipt.gate_passed);
    }

    #[test]
    fn test_full_inference_path_mints_poi() {
        let mut k = make_kernel();
        let c = cycle("novel question about BIZRA");
        let responses = vec!["Answer A".into(), "Answer A".into(), "Answer B".into()];
        let level_scores = vec![(HhmmLevel::L2Cognitive, 0.97), (HhmmLevel::L3Memory, 0.96)];
        let receipt = k.run_cycle(&c, &responses, 0.97, &level_scores, None, None);
        assert_eq!(receipt.path, CyclePath::FullInference);
        assert!(receipt.gate_passed);
        assert!(receipt.poi_yield.is_some());
        assert!(receipt.ttrl_queued, "TTRL should queue with ≥3 responses");
    }

    #[test]
    fn test_engram_hit_returns_cached_value() {
        let mut k = make_kernel();
        let c = cycle("what is the capital of France");

        // Pre-populate Engram cache.
        k.engram_cache_mut()
            .insert(c.intent_bytes.as_slice(), "Paris", 0.99, c.now_ms);

        let receipt = k.run_cycle(&c, &[], 0.97, &[], None, None);
        assert_eq!(receipt.path, CyclePath::EngramHit);
        assert_eq!(receipt.response, "Paris");
    }

    #[test]
    fn test_emission_decay_rises_with_repeated_misses() {
        let mut k = make_kernel();
        // Run 10 full-inference cycles with good İhsān.
        for i in 0..10 {
            let c = cycle(&format!("unique query {i}"));
            let responses = vec!["R".into(); 3];
            k.run_cycle(&c, &responses, 0.97, &[], None, None);
        }
        // After 10 cache misses, hit_rate ≈ 0 → emission multiplier ≈ 1.0
        assert!(k.metabolic_stats().avg_multiplier > 0.9);
    }
}
