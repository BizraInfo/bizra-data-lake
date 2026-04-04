// src/federation/consensus.rs - Byzantine Fault Tolerant Pattern Consensus
//
// Implements 3-of-5 (or N-of-M) Byzantine consensus for pattern acceptance.
// Compatible with Python implementation in core/federation/consensus.py

use crate::federation::protocol::{
    verify_signature, ConsensusResult, ConsensusVote, PatternEnvelope, SignedVote, VoteDecision,
    MIN_IHSAN_SCORE,
};
use anyhow::Result;
use chrono::{DateTime, Utc};
use ed25519_dalek::{Signer, SigningKey, VerifyingKey};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Byzantine parameters: N = 5, f = 1, quorum = 2f + 1 = 3
pub const CONSENSUS_N: usize = 5;
pub const CONSENSUS_F: usize = 1;
pub const CONSENSUS_QUORUM: usize = 2 * CONSENSUS_F + 1; // = 3

/// Consensus timeout in seconds
pub const CONSENSUS_TIMEOUT_SEC: u64 = 30;

// ═══════════════════════════════════════════════════════════════════════════════
// CONSENSUS STATE
// ═══════════════════════════════════════════════════════════════════════════════

/// Phases of consensus
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsensusPhase {
    Idle,
    PrePrepare,
    Prepare,
    Commit,
    Finalized,
    Failed,
}

/// State of an ongoing consensus round
#[derive(Debug, Clone)]
pub struct ConsensusState {
    pub pattern_id: String,
    pub phase: ConsensusPhase,
    pub started_at: Instant,

    /// Votes collected in prepare phase
    pub prepare_votes: HashMap<String, ConsensusVote>,
    /// Votes collected in commit phase
    pub commit_votes: HashMap<String, ConsensusVote>,

    /// Final result
    pub result: Option<ConsensusResult>,
}

impl ConsensusState {
    pub fn new(pattern_id: String) -> Self {
        Self {
            pattern_id,
            phase: ConsensusPhase::Idle,
            started_at: Instant::now(),
            prepare_votes: HashMap::new(),
            commit_votes: HashMap::new(),
            result: None,
        }
    }

    pub fn is_expired(&self) -> bool {
        self.started_at.elapsed() > Duration::from_secs(CONSENSUS_TIMEOUT_SEC)
    }

    pub fn prepare_count(&self) -> usize {
        self.prepare_votes
            .values()
            .filter(|v| v.decision == VoteDecision::Accept)
            .count()
    }

    pub fn commit_count(&self) -> usize {
        self.commit_votes
            .values()
            .filter(|v| v.decision == VoteDecision::Accept)
            .count()
    }

    pub fn has_prepare_quorum(&self) -> bool {
        self.prepare_count() >= CONSENSUS_QUORUM
    }

    pub fn has_commit_quorum(&self) -> bool {
        self.commit_count() >= CONSENSUS_QUORUM
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PATTERN VALIDATOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Validates patterns for consensus
pub struct PatternValidator {
    /// Known malicious pattern hashes
    blacklisted_patterns: std::collections::HashSet<String>,
    /// Node reputation cache
    node_reputation: HashMap<String, f64>,
}

impl PatternValidator {
    pub fn new() -> Self {
        Self {
            blacklisted_patterns: std::collections::HashSet::new(),
            node_reputation: HashMap::new(),
        }
    }

    /// Validate pattern and return vote decision
    pub fn validate(
        &self,
        envelope: &PatternEnvelope,
        local_ihsan_score: Option<f64>,
    ) -> (VoteDecision, String, f64) {
        // 1. Check blacklist
        if self
            .blacklisted_patterns
            .contains(&envelope.metadata.pattern_id)
        {
            return (VoteDecision::Reject, "Pattern blacklisted".to_string(), 0.0);
        }

        // 2. Verify cryptographic integrity
        match envelope.verify() {
            Ok(true) => {}
            Ok(false) => {
                return (VoteDecision::Reject, "Verification failed".to_string(), 0.0);
            }
            Err(e) => {
                return (
                    VoteDecision::Reject,
                    format!("Verification error: {}", e),
                    0.0,
                );
            }
        }

        // 3. Check Ihsān score
        let ihsan = envelope.metadata.ihsan_score;
        if ihsan < MIN_IHSAN_SCORE {
            return (
                VoteDecision::Reject,
                format!("Ihsān {:.2} < {:.2}", ihsan, MIN_IHSAN_SCORE),
                ihsan,
            );
        }

        // 4. Check node reputation
        let origin = &envelope.metadata.origin_node_id;
        if let Some(&rep) = self.node_reputation.get(origin) {
            if rep < 0.5 {
                return (
                    VoteDecision::Reject,
                    "Origin node reputation too low".to_string(),
                    ihsan,
                );
            }
        }

        // 5. Check impact score
        if envelope.metadata.impact_score < 0.5 {
            return (
                VoteDecision::Reject,
                format!(
                    "Impact score too low: {:.2}",
                    envelope.metadata.impact_score
                ),
                ihsan,
            );
        }

        // 6. Check repetition count
        if envelope.metadata.repetition_count < 3 {
            return (
                VoteDecision::Reject,
                "Insufficient repetitions".to_string(),
                ihsan,
            );
        }

        // 7. Combine with local evaluation if provided
        let final_ihsan = if let Some(local) = local_ihsan_score {
            let combined = 0.7 * ihsan + 0.3 * local;
            if combined < MIN_IHSAN_SCORE {
                return (
                    VoteDecision::Reject,
                    format!("Combined Ihsān {:.2} too low", combined),
                    combined,
                );
            }
            combined
        } else {
            ihsan
        };

        (
            VoteDecision::Accept,
            "Valid pattern".to_string(),
            final_ihsan,
        )
    }

    /// Update node reputation
    pub fn update_reputation(&mut self, node_id: &str, delta: f64) {
        let current = self.node_reputation.get(node_id).copied().unwrap_or(1.0);
        let new_rep = (current + delta).clamp(0.0, 1.0);
        self.node_reputation.insert(node_id.to_string(), new_rep);
    }

    /// Blacklist a pattern
    pub fn blacklist_pattern(&mut self, pattern_id: &str) {
        self.blacklisted_patterns.insert(pattern_id.to_string());
    }
}

impl Default for PatternValidator {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONSENSUS ENGINE
// ═══════════════════════════════════════════════════════════════════════════════

/// Consensus statistics
#[derive(Debug, Default, Clone)]
pub struct ConsensusStats {
    pub rounds_started: u64,
    pub rounds_accepted: u64,
    pub rounds_rejected: u64,
    pub rounds_timeout: u64,
}

impl ConsensusStats {
    pub fn acceptance_rate(&self) -> f64 {
        if self.rounds_started == 0 {
            0.0
        } else {
            self.rounds_accepted as f64 / self.rounds_started as f64
        }
    }
}

/// Byzantine Fault Tolerant consensus for pattern acceptance
pub struct PatternConsensus {
    node_id: String,
    private_key: Vec<u8>,
    public_key: Vec<u8>,

    /// Active consensus rounds
    active_rounds: HashMap<String, ConsensusState>,
    /// Completed rounds (recent)
    completed_rounds: HashMap<String, ConsensusResult>,
    /// Validator
    validator: PatternValidator,
    /// Statistics
    pub stats: ConsensusStats,
}

impl PatternConsensus {
    pub fn new(node_id: String, private_key: Vec<u8>, public_key: Vec<u8>) -> Self {
        Self {
            node_id,
            private_key,
            public_key,
            active_rounds: HashMap::new(),
            completed_rounds: HashMap::new(),
            validator: PatternValidator::new(),
            stats: ConsensusStats::default(),
        }
    }

    /// Propose a new pattern for consensus (PRE-PREPARE phase)
    pub async fn propose_pattern(&mut self, envelope: &PatternEnvelope) -> Result<ConsensusState> {
        let pattern_id = envelope.metadata.pattern_id.clone();

        // Already in consensus?
        if let Some(state) = self.active_rounds.get(&pattern_id) {
            return Ok(state.clone());
        }

        // Already completed?
        if let Some(result) = self.completed_rounds.get(&pattern_id) {
            let mut state = ConsensusState::new(pattern_id);
            state.phase = ConsensusPhase::Finalized;
            state.result = Some(result.clone());
            return Ok(state);
        }

        // Create new consensus round
        let mut state = ConsensusState::new(pattern_id.clone());
        state.phase = ConsensusPhase::PrePrepare;

        info!(
            "📋 Starting consensus for pattern {}...",
            &pattern_id[..16.min(pattern_id.len())]
        );

        // Validate and cast our vote
        let (decision, reason, ihsan) = self.validator.validate(envelope, None);

        let our_vote = self.create_vote(&pattern_id, decision, &reason, ihsan)?;
        state.prepare_votes.insert(self.node_id.clone(), our_vote);

        // Move to PREPARE phase
        state.phase = ConsensusPhase::Prepare;

        self.active_rounds.insert(pattern_id, state.clone());
        self.stats.rounds_started += 1;

        Ok(state)
    }

    /// Receive a vote from another validator
    pub async fn receive_vote(
        &mut self,
        vote: ConsensusVote,
        phase: &str,
    ) -> Option<ConsensusResult> {
        let pattern_id = vote.pattern_id.clone();

        // Already completed?
        if let Some(result) = self.completed_rounds.get(&pattern_id) {
            return Some(result.clone());
        }

        // Check for timeout first
        let is_expired = self
            .active_rounds
            .get(&pattern_id)
            .map(|s| s.is_expired())
            .unwrap_or(false);

        if is_expired {
            return Some(self.finalize_timeout(&pattern_id));
        }

        // Process the vote and determine next action
        let (need_commit_vote, ihsan_for_commit, should_finalize_accept, should_finalize_commit) = {
            // Get or create state
            let state = self
                .active_rounds
                .entry(pattern_id.clone())
                .or_insert_with(|| {
                    let mut s = ConsensusState::new(pattern_id.clone());
                    s.phase = ConsensusPhase::Prepare;
                    s
                });

            let mut need_commit = false;
            let mut ihsan = 0.0;
            let mut finalize_accept = false;
            let mut finalize_commit = false;

            // Record vote
            match phase {
                "prepare" => {
                    state
                        .prepare_votes
                        .insert(vote.voter_id.clone(), vote.clone());

                    // Check for prepare quorum
                    if state.has_prepare_quorum() && state.phase == ConsensusPhase::Prepare {
                        state.phase = ConsensusPhase::Commit;
                        debug!(
                            "Pattern {} reached prepare quorum",
                            &pattern_id[..16.min(pattern_id.len())]
                        );
                        need_commit = true;
                        ihsan = vote.ihsan_score;
                    }
                }
                "commit" => {
                    state.commit_votes.insert(vote.voter_id.clone(), vote);

                    // Check for commit quorum
                    if state.has_commit_quorum() && state.phase == ConsensusPhase::Commit {
                        finalize_commit = true;
                    }
                }
                _ => {}
            }

            (need_commit, ihsan, finalize_accept, finalize_commit)
        };

        // Now perform actions outside the borrow
        if need_commit_vote {
            if let Ok(commit_vote) = self.create_vote(
                &pattern_id,
                VoteDecision::Accept,
                "Prepare quorum reached",
                ihsan_for_commit,
            ) {
                if let Some(state) = self.active_rounds.get_mut(&pattern_id) {
                    state.commit_votes.insert(self.node_id.clone(), commit_vote);
                }
            }
        }

        if should_finalize_commit {
            return Some(self.finalize_accept(&pattern_id));
        }

        None
    }

    /// Receive a signed vote with Ed25519 verification (CRIT-1)
    ///
    /// Genesis Strict Synthesis v2.2.2: All votes MUST be verified before counting.
    /// This is the preferred method for receiving votes from other nodes.
    pub async fn receive_signed_vote(
        &mut self,
        signed_vote: SignedVote,
        phase: &str,
    ) -> Option<ConsensusResult> {
        // CRIT-1: Verify signature BEFORE processing
        if !signed_vote.verify() {
            warn!(
                "⚠️ REJECTED vote from {}: Ed25519 signature verification FAILED",
                signed_vote.vote.voter_id
            );
            return None;
        }

        debug!(
            "✓ Vote from {} verified (pattern: {})",
            signed_vote.vote.voter_id,
            &signed_vote.vote.pattern_id[..16.min(signed_vote.vote.pattern_id.len())]
        );

        // Extract verified vote and process
        self.receive_vote(signed_vote.into_vote(), phase).await
    }

    /// Create a signed vote
    fn create_vote(
        &self,
        pattern_id: &str,
        decision: VoteDecision,
        reason: &str,
        ihsan: f64,
    ) -> Result<ConsensusVote> {
        let timestamp = Utc::now();

        // Create signature content
        let content = serde_json::json!({
            "pattern_id": pattern_id,
            "voter_id": self.node_id,
            "decision": decision,
            "reason": reason,
            "ihsan_score": ihsan,
            "timestamp": timestamp.to_rfc3339(),
        });

        let content_str = serde_json::to_string(&content)?;

        // Sign (simplified - real implementation would use proper key handling)
        let private_key_bytes: [u8; 32] = self.private_key[..32]
            .try_into()
            .map_err(|_| anyhow::anyhow!("Invalid private key length"))?;
        let signing_key = SigningKey::from_bytes(&private_key_bytes);
        let signature = signing_key.sign(content_str.as_bytes());

        Ok(ConsensusVote {
            pattern_id: pattern_id.to_string(),
            voter_id: self.node_id.clone(),
            decision,
            reason: reason.to_string(),
            ihsan_score: ihsan,
            timestamp,
            signature: hex::encode(signature.to_bytes()),
        })
    }

    /// Create a signed vote (CRIT-1 compliant)
    ///
    /// Genesis Strict Synthesis v2.2.2: Returns SignedVote for transmission.
    /// Recipients MUST call receive_signed_vote() to verify before counting.
    pub fn create_signed_vote(
        &self,
        pattern_id: &str,
        decision: VoteDecision,
        reason: &str,
        ihsan: f64,
    ) -> Result<SignedVote> {
        let vote = self.create_vote(pattern_id, decision, reason, ihsan)?;

        // Get verifying key from private key
        let private_key_bytes: [u8; 32] = self.private_key[..32]
            .try_into()
            .map_err(|_| anyhow::anyhow!("Invalid private key length"))?;
        let signing_key = SigningKey::from_bytes(&private_key_bytes);
        let verifying_key = signing_key.verifying_key();

        Ok(SignedVote::from_vote(vote, &verifying_key))
    }

    /// Get our public key as hex
    pub fn public_key_hex(&self) -> String {
        hex::encode(&self.public_key)
    }

    /// Finalize consensus as accepted
    fn finalize_accept(&mut self, pattern_id: &str) -> ConsensusResult {
        let state = self.active_rounds.remove(pattern_id).unwrap();

        let all_votes: Vec<ConsensusVote> = state
            .prepare_votes
            .values()
            .chain(state.commit_votes.values())
            .cloned()
            .collect();

        let result = ConsensusResult {
            pattern_id: pattern_id.to_string(),
            accepted: true,
            accept_votes: state.commit_count(),
            reject_votes: state
                .commit_votes
                .values()
                .filter(|v| v.decision == VoteDecision::Reject)
                .count(),
            abstain_votes: state
                .commit_votes
                .values()
                .filter(|v| v.decision == VoteDecision::Abstain)
                .count(),
            quorum_reached: true,
            finalized_at: Utc::now(),
            votes: all_votes,
        };

        self.completed_rounds
            .insert(pattern_id.to_string(), result.clone());
        self.stats.rounds_accepted += 1;

        info!(
            "✅ Pattern {} ACCEPTED by consensus ({}/{})",
            &pattern_id[..16.min(pattern_id.len())],
            result.accept_votes,
            CONSENSUS_QUORUM
        );

        result
    }

    /// Finalize consensus as failed due to timeout
    fn finalize_timeout(&mut self, pattern_id: &str) -> ConsensusResult {
        let state = self.active_rounds.remove(pattern_id).unwrap();

        let result = ConsensusResult {
            pattern_id: pattern_id.to_string(),
            accepted: false,
            accept_votes: state.prepare_count(),
            reject_votes: state
                .prepare_votes
                .values()
                .filter(|v| v.decision == VoteDecision::Reject)
                .count(),
            abstain_votes: state
                .prepare_votes
                .values()
                .filter(|v| v.decision == VoteDecision::Abstain)
                .count(),
            quorum_reached: false,
            finalized_at: Utc::now(),
            votes: state.prepare_votes.values().cloned().collect(),
        };

        self.completed_rounds
            .insert(pattern_id.to_string(), result.clone());
        self.stats.rounds_timeout += 1;

        warn!(
            "⏱️ Pattern {} TIMEOUT - no consensus reached",
            &pattern_id[..16.min(pattern_id.len())]
        );

        result
    }

    /// Get consensus result for a pattern
    pub fn get_result(&self, pattern_id: &str) -> Option<&ConsensusResult> {
        self.completed_rounds.get(pattern_id)
    }

    /// Check if pattern was accepted
    pub fn is_accepted(&self, pattern_id: &str) -> bool {
        self.completed_rounds
            .get(pattern_id)
            .map(|r| r.accepted)
            .unwrap_or(false)
    }

    /// Clean up old completed rounds
    pub fn cleanup_old_rounds(&mut self, max_age: Duration) {
        let cutoff = Utc::now() - chrono::Duration::from_std(max_age).unwrap();

        self.completed_rounds
            .retain(|_, result| result.finalized_at > cutoff);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::federation::protocol::{PatternMetadata, PatternPayload, PatternType};

    #[test]
    fn test_consensus_state() {
        let state = ConsensusState::new("test_pattern".to_string());

        assert_eq!(state.phase, ConsensusPhase::Idle);
        assert!(!state.is_expired());
        assert!(!state.has_prepare_quorum());
        assert!(!state.has_commit_quorum());
    }

    #[test]
    fn test_pattern_validator() {
        let validator = PatternValidator::new();

        // Create test envelope
        let (private_key, public_key) = crate::federation::protocol::generate_keypair();

        let metadata = PatternMetadata {
            pattern_id: "test".to_string(),
            pattern_type: PatternType::SapeProbe,
            version: 1,
            origin_node_id: "node".to_string(),
            origin_timestamp: Utc::now(),
            repetition_count: 5,
            success_rate: 0.9,
            impact_score: 0.8,
            ihsan_score: 0.92,
            adoption_count: 0,
            expires_at: Utc::now() + chrono::Duration::days(30),
            tags: vec![],
        };

        let payload = PatternPayload {
            trigger_sequence: vec!["test".to_string()],
            optimization: "test".to_string(),
            latency_reduction_ms: 50,
            token_savings_percent: 20.0,
            snr_improvement: 0.1,
        };

        let envelope = crate::federation::protocol::PatternEnvelope::create(
            metadata,
            payload,
            &private_key,
            &public_key,
        )
        .unwrap();

        let (decision, _, _) = validator.validate(&envelope, None);
        assert_eq!(decision, VoteDecision::Accept);
    }

    #[test]
    fn test_consensus_stats() {
        let mut stats = ConsensusStats::default();

        assert_eq!(stats.acceptance_rate(), 0.0);

        stats.rounds_started = 10;
        stats.rounds_accepted = 8;

        assert!((stats.acceptance_rate() - 0.8).abs() < 0.001);
    }
}
