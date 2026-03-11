//! BFT Consensus — Byzantine Fault-Tolerant Voting
//!
//! SECURITY: All votes MUST be cryptographically signed and verified
//! BEFORE counting. This prevents vote spoofing attacks.
//!
//! Standing on Giants: Lamport (1982) — Byzantine Generals Problem

use bizra_core::{NodeId, NodeIdentity, IHSAN_THRESHOLD};
use chrono::{DateTime, Utc};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Proposal {
    pub id: String,
    pub proposer: NodeId,
    pub pattern: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub ihsan_score: f64,
}

impl Proposal {
    pub fn new(proposer: NodeId, pattern: serde_json::Value, ihsan_score: f64) -> Self {
        Self {
            id: format!(
                "prop_{}",
                &uuid::Uuid::new_v4().to_string().replace("-", "")[..12]
            ),
            proposer,
            pattern,
            created_at: Utc::now(),
            ihsan_score,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Vote {
    pub proposal_id: String,
    pub voter_id: NodeId,
    pub approve: bool,
    pub ihsan_score: f64,
    pub timestamp: DateTime<Utc>,
}

impl Vote {
    /// Serialize vote to canonical bytes for signing
    pub fn to_bytes(&self) -> Vec<u8> {
        // Domain-separated canonical serialization
        let canonical = format!(
            "bizra-vote-v1:{}:{}:{}:{}:{}",
            self.proposal_id,
            self.voter_id,
            self.approve,
            self.ihsan_score,
            self.timestamp.timestamp_millis()
        );
        canonical.into_bytes()
    }
}

/// Signed vote with Ed25519 cryptographic verification
/// SECURITY: Signature MUST be verified before vote is counted
#[derive(Clone, Debug)]
pub struct SignedVote {
    pub vote: Vote,
    pub signature: [u8; 64],
    pub sender_pubkey: [u8; 32],
}

impl SignedVote {
    /// Create a new signed vote
    pub fn new(vote: Vote, signing_key: &SigningKey) -> Self {
        let message_bytes = vote.to_bytes();
        let signature = signing_key.sign(&message_bytes);
        Self {
            vote,
            signature: signature.to_bytes(),
            sender_pubkey: signing_key.verifying_key().to_bytes(),
        }
    }

    /// Verify the Ed25519 signature on this vote
    /// CRITICAL: This MUST be called before counting the vote
    pub fn verify(&self) -> Result<(), ConsensusError> {
        let verifying_key = VerifyingKey::from_bytes(&self.sender_pubkey)
            .map_err(|_| ConsensusError::InvalidPublicKey)?;
        let signature = Signature::from_bytes(&self.signature);
        verifying_key
            .verify(&self.vote.to_bytes(), &signature)
            .map_err(|_| ConsensusError::InvalidSignature)
    }

    /// Get hex-encoded public key for display
    pub fn pubkey_hex(&self) -> String {
        hex::encode(self.sender_pubkey)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConsensusState {
    Idle,
    Voting,
    Committing,
    Committed,
    Rejected,
}

struct Round {
    #[allow(dead_code)] // Stored for audit trail and round inspection
    proposal: Proposal,
    votes: HashMap<NodeId, SignedVote>,
    state: ConsensusState,
    quorum_size: usize,
}

/// Known peer with verified public key
#[derive(Clone, Debug)]
pub struct KnownPeer {
    pub node_id: NodeId,
    pub public_key: [u8; 32],
}

pub struct ConsensusEngine {
    identity: NodeIdentity,
    rounds: HashMap<String, Round>,
    committed: HashSet<String>,
    total_nodes: usize,
    /// Known peers with verified public keys (for voter verification)
    known_peers: HashMap<NodeId, KnownPeer>,
}

impl ConsensusEngine {
    pub fn new(identity: NodeIdentity) -> Self {
        let node_id = identity.node_id().clone();
        let public_key = identity.public_key_bytes();
        let mut known_peers = HashMap::new();
        // Register self as a known peer
        known_peers.insert(
            node_id.clone(),
            KnownPeer {
                node_id,
                public_key,
            },
        );
        Self {
            identity,
            rounds: HashMap::new(),
            committed: HashSet::new(),
            total_nodes: 1,
            known_peers,
        }
    }

    /// Register a known peer with their verified public key
    pub fn register_peer(&mut self, node_id: NodeId, public_key: [u8; 32]) {
        self.known_peers.insert(
            node_id.clone(),
            KnownPeer {
                node_id,
                public_key,
            },
        );
    }

    pub fn set_node_count(&mut self, count: usize) {
        self.total_nodes = count.max(1);
    }

    pub fn propose(
        &mut self,
        pattern: serde_json::Value,
        ihsan_score: f64,
    ) -> Result<Proposal, ConsensusError> {
        if ihsan_score < IHSAN_THRESHOLD {
            return Err(ConsensusError::IhsanThreshold(ihsan_score));
        }
        let proposal = Proposal::new(self.identity.node_id().clone(), pattern, ihsan_score);
        let quorum = (2 * self.total_nodes / 3) + 1;
        self.rounds.insert(
            proposal.id.clone(),
            Round {
                proposal: proposal.clone(),
                votes: HashMap::new(),
                state: ConsensusState::Voting,
                quorum_size: quorum.max(1),
            },
        );
        Ok(proposal)
    }

    /// Create a signed vote for a proposal
    pub fn vote(
        &self,
        proposal_id: &str,
        approve: bool,
        ihsan: f64,
    ) -> Result<SignedVote, ConsensusError> {
        if !self.rounds.contains_key(proposal_id) {
            return Err(ConsensusError::ProposalNotFound(proposal_id.into()));
        }
        let vote = Vote {
            proposal_id: proposal_id.into(),
            voter_id: self.identity.node_id().clone(),
            approve,
            ihsan_score: ihsan,
            timestamp: Utc::now(),
        };
        Ok(SignedVote::new(vote, self.identity.signing_key()))
    }

    /// Receive and verify a signed vote
    /// SECURITY: Signature verification happens BEFORE vote counting
    pub fn receive_vote(&mut self, signed_vote: SignedVote) -> Result<bool, ConsensusError> {
        // CRITICAL: Verify signature BEFORE anything else
        signed_vote.verify()?;

        // Verify voter pubkey matches known peer
        let voter_id = &signed_vote.vote.voter_id;
        let expected_peer = self
            .known_peers
            .get(voter_id)
            .ok_or(ConsensusError::UnknownVoter(voter_id.clone()))?;

        if signed_vote.sender_pubkey != expected_peer.public_key {
            return Err(ConsensusError::PubkeyMismatch {
                voter: voter_id.clone(),
                expected: hex::encode(expected_peer.public_key),
                received: hex::encode(signed_vote.sender_pubkey),
            });
        }

        // Verify Ihsān score meets threshold
        if signed_vote.vote.ihsan_score < IHSAN_THRESHOLD {
            return Err(ConsensusError::IhsanThreshold(signed_vote.vote.ihsan_score));
        }

        let round = self
            .rounds
            .get_mut(&signed_vote.vote.proposal_id)
            .ok_or_else(|| {
                ConsensusError::ProposalNotFound(signed_vote.vote.proposal_id.clone())
            })?;

        if round.state != ConsensusState::Voting {
            return Err(ConsensusError::NotVoting);
        }

        // Now safe to count the verified vote
        round.votes.insert(voter_id.clone(), signed_vote);
        let approvals = round.votes.values().filter(|v| v.vote.approve).count();

        if approvals >= round.quorum_size {
            round.state = ConsensusState::Committing;
            return Ok(true);
        }
        Ok(false)
    }

    pub fn commit(&mut self, proposal_id: &str) -> Result<(), ConsensusError> {
        let round = self
            .rounds
            .get_mut(proposal_id)
            .ok_or_else(|| ConsensusError::ProposalNotFound(proposal_id.into()))?;
        if round.state != ConsensusState::Committing {
            return Err(ConsensusError::NotReady);
        }
        round.state = ConsensusState::Committed;
        self.committed.insert(proposal_id.into());
        Ok(())
    }

    pub fn is_committed(&self, proposal_id: &str) -> bool {
        self.committed.contains(proposal_id)
    }
    pub fn committed_count(&self) -> usize {
        self.committed.len()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConsensusError {
    #[error("Proposal not found: {0}")]
    ProposalNotFound(String),
    #[error("Ihsan {0} below threshold {}", IHSAN_THRESHOLD)]
    IhsanThreshold(f64),
    #[error("Not in voting state")]
    NotVoting,
    #[error("Not ready for commit")]
    NotReady,
    #[error("Invalid public key format")]
    InvalidPublicKey,
    #[error("Invalid Ed25519 signature")]
    InvalidSignature,
    #[error("Unknown voter: {0}")]
    UnknownVoter(NodeId),
    #[error("Public key mismatch for voter {voter}: expected {expected}, got {received}")]
    PubkeyMismatch {
        voter: NodeId,
        expected: String,
        received: String,
    },
}
