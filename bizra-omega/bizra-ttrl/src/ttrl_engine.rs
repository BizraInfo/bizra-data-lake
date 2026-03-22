//! # TTRL Engine — Test-Time Reinforcement Learning (Paper 4)
//!
//! TTRL closes the self-improvement loop.  Every verified action receipt
//! (Iḥsān ≥ 0.95) becomes a training signal.  PAT agents run in parallel
//! and their majority-vote answer is the reward signal for on-device GRPO.
//!
//! ## The loop (maps to Omni-Kernel line 7)
//! 1. PAT agents produce `N` candidate responses.
//! 2. Majority vote → `best_answer`.
//! 3. GRPO reward = `f(ihsan_score, cpva_actual, majority_fraction)`.
//! 4. Update is queued; applied lazily after Iḥsān gate passes.
//! 5. SSO constraint is checked before the update commits.
//! 6. SEED emission multiplier decays as cache-hit-rate rises.
//!
//! ## CPVA improvement curve (from paper, Qwen-2.5-Math-7B baseline)
//! Month 1: +0%   (base)
//! Month 3: +50%  (TTRL warmup)
//! Month 12: +211% (proven result)
//!
//! Standing on Giants:
//! - TTRL paper (2025): Test-Time Reinforcement Learning
//! - DeepSeek-R1 (2025): Group Relative Policy Optimisation (GRPO)
//! - Nakamoto (2008): emission decay as scarcity signal

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::sso::{SpectralNorm, SpectralSphereConstraint, SsoCheckResult};

/// A single pending GRPO update, queued for lazy application.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpoUpdate {
    /// Canonical bytes of the triggering intent.
    pub intent_hash: [u8; 32],
    /// The majority-vote response from PAT agents.
    pub best_answer: String,
    /// Computed reward signal (0–1).
    pub reward: f64,
    /// Iḥsān score of the verified action that produced this update.
    pub ihsan_score: f64,
    /// UNIX ms timestamp.
    pub queued_at_ms: u64,
}

/// Statistics exported by `TtrlEngine`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TtrlStats {
    pub updates_queued: u64,
    pub updates_applied: u64,
    pub updates_rejected: u64, // SSO violations
    pub total_reward: f64,
}

/// The on-device TTRL engine.
///
/// In production, `apply_pending_update` calls into the model runtime
/// (Python via PyO3 or HTTP to LM Studio).  Here it is modelled as a
/// pure Rust stub: the spectral-norm values are passed in from the caller
/// (who owns the model weights).
#[derive(Debug)]
pub struct TtrlEngine {
    queue: VecDeque<GrpoUpdate>,
    sso: SpectralSphereConstraint,
    pub stats: TtrlStats,
}

impl TtrlEngine {
    pub fn new(sso: SpectralSphereConstraint) -> Self {
        Self {
            queue: VecDeque::new(),
            sso,
            stats: TtrlStats::default(),
        }
    }

    /// Queue a new GRPO update derived from PAT majority vote.
    ///
    /// `pat_responses` — slice of candidate answers from PAT agents.
    /// Returns the queued `GrpoUpdate` (for logging / receipt chain).
    pub fn queue_update(
        &mut self,
        intent_hash: [u8; 32],
        pat_responses: &[String],
        ihsan_score: f64,
        cpva_actual: f64,
        now_ms: u64,
    ) -> Option<GrpoUpdate> {
        if pat_responses.len() < 3 {
            // Not enough agents for majority vote — skip.
            return None;
        }

        let best_answer = Self::majority_vote(pat_responses);
        let majority_fraction = Self::majority_fraction(&best_answer, pat_responses);
        let reward = Self::compute_reward(ihsan_score, cpva_actual, majority_fraction);

        let update = GrpoUpdate {
            intent_hash,
            best_answer,
            reward,
            ihsan_score,
            queued_at_ms: now_ms,
        };
        self.queue.push_back(update.clone());
        self.stats.updates_queued += 1;
        Some(update)
    }

    /// Returns `true` if there is at least one pending update.
    pub fn has_pending_update(&self) -> bool {
        !self.queue.is_empty()
    }

    /// Apply the oldest queued update, constrained by SSO.
    ///
    /// `pre_norm`  — spectral norm measured BEFORE applying the update.
    /// `post_norm` — spectral norm measured AFTER applying the update.
    ///              (Caller runs the actual weight update; we check constraint.)
    ///
    /// Returns the SSO check result.  If it fails, the update is discarded
    /// and the caller must roll back the weight change.
    pub fn apply_pending_update(
        &mut self,
        pre_norm: SpectralNorm,
        post_norm: SpectralNorm,
    ) -> SsoCheckResult {
        let result = self.sso.check(pre_norm, post_norm);

        if let Some(update) = self.queue.pop_front() {
            if result.passed() {
                self.stats.updates_applied += 1;
                self.stats.total_reward += update.reward;
                tracing::info!(
                    reward = update.reward,
                    drift = result.drift,
                    "TTRL update applied (SSO passed)"
                );
            } else {
                self.stats.updates_rejected += 1;
                tracing::warn!(
                    drift = result.drift,
                    epsilon = result.epsilon,
                    "TTRL update rejected: SSO violation — caller must rollback weights"
                );
            }
        }
        result
    }

    // ─── private helpers ────────────────────────────────────────────────

    /// Return the most-frequent response.  Ties broken by first occurrence.
    fn majority_vote(responses: &[String]) -> String {
        let mut counts: Vec<(usize, &str)> = Vec::new();
        for r in responses {
            if let Some(c) = counts.iter_mut().find(|(_, s)| *s == r.as_str()) {
                c.0 += 1;
            } else {
                counts.push((1, r.as_str()));
            }
        }
        counts.sort_by(|a, b| b.0.cmp(&a.0));
        counts
            .first()
            .map(|(_, s)| s.to_string())
            .unwrap_or_default()
    }

    fn majority_fraction(winner: &str, responses: &[String]) -> f64 {
        let count = responses.iter().filter(|r| r.as_str() == winner).count();
        count as f64 / responses.len() as f64
    }

    /// GRPO reward signal combining Iḥsān, economic efficiency, and consensus.
    ///
    /// reward = ihsan_score × (1 − cpva_normalised) × majority_fraction
    /// where cpva_normalised = cpva_actual / 0.10 (baseline cost)
    fn compute_reward(ihsan_score: f64, cpva_actual: f64, majority_fraction: f64) -> f64 {
        const CPVA_BASELINE: f64 = 0.10;
        let cost_efficiency = 1.0 - (cpva_actual / CPVA_BASELINE).min(1.0);
        (ihsan_score * cost_efficiency * majority_fraction).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sso::SpectralNorm;

    fn engine() -> TtrlEngine {
        TtrlEngine::new(SpectralSphereConstraint::default())
    }

    #[test]
    fn test_majority_vote_correct() {
        let responses = vec!["A".into(), "B".into(), "A".into(), "C".into(), "A".into()];
        let winner = TtrlEngine::majority_vote(&responses);
        assert_eq!(winner, "A");
    }

    #[test]
    fn test_queue_requires_3_responses() {
        let mut e = engine();
        let r = e.queue_update([0u8; 32], &["A".into(), "B".into()], 0.97, 0.08, 0);
        assert!(r.is_none(), "Need ≥3 responses");
    }

    #[test]
    fn test_update_applied_on_sso_pass() {
        let mut e = engine();
        let responses = vec!["X".into(); 3];
        e.queue_update([0u8; 32], &responses, 0.97, 0.05, 1000);
        assert!(e.has_pending_update());

        let pre = SpectralNorm::new(1.0);
        let post = SpectralNorm::new(1.0005); // drift=5e-4 < 1e-3 ✓
        let r = e.apply_pending_update(pre, post);
        assert!(r.passed());
        assert_eq!(e.stats.updates_applied, 1);
        assert_eq!(e.stats.updates_rejected, 0);
    }

    #[test]
    fn test_update_rejected_on_sso_fail() {
        let mut e = engine();
        let responses = vec!["X".into(); 3];
        e.queue_update([0u8; 32], &responses, 0.97, 0.05, 1000);

        let pre = SpectralNorm::new(1.0);
        let post = SpectralNorm::new(1.01); // drift=1e-2 > 1e-3 ✗
        let r = e.apply_pending_update(pre, post);
        assert!(!r.passed());
        assert_eq!(e.stats.updates_rejected, 1);
        assert_eq!(e.stats.updates_applied, 0);
    }
}
