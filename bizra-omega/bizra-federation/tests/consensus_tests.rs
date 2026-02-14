//! Comprehensive tests for BFT consensus — proposals, votes, signatures, quorum
//!
//! Phase 13: Test Sprint — Covering ConsensusEngine, SignedVote, Proposal

use bizra_core::{NodeId, NodeIdentity};
use bizra_federation::consensus::*;

fn make_identity() -> NodeIdentity {
    NodeIdentity::generate()
}

// ---------------------------------------------------------------------------
// Proposal
// ---------------------------------------------------------------------------

#[test]
fn proposal_new_generates_unique_id() {
    let id = NodeId("proposer".into());
    let p1 = Proposal::new(id.clone(), serde_json::json!({"a": 1}), 0.96);
    let p2 = Proposal::new(id, serde_json::json!({"b": 2}), 0.96);
    assert!(p1.id.starts_with("prop_"));
    assert_ne!(p1.id, p2.id);
}

#[test]
fn proposal_stores_ihsan_score() {
    let p = Proposal::new(NodeId("x".into()), serde_json::json!(null), 0.99);
    assert!((p.ihsan_score - 0.99).abs() < f64::EPSILON);
}

// ---------------------------------------------------------------------------
// Vote serialization
// ---------------------------------------------------------------------------

#[test]
fn vote_to_bytes_deterministic() {
    let v = Vote {
        proposal_id: "prop_abc".into(),
        voter_id: NodeId("abcdef1234567890abcdef1234567890".into()),
        approve: true,
        ihsan_score: 0.96,
        timestamp: chrono::Utc::now(),
    };
    let b1 = v.to_bytes();
    let b2 = v.to_bytes();
    assert_eq!(b1, b2);
}

#[test]
fn vote_to_bytes_differs_for_different_votes() {
    let ts = chrono::Utc::now();
    let v1 = Vote {
        proposal_id: "prop_abc".into(),
        voter_id: NodeId("abcdef1234567890abcdef1234567890".into()),
        approve: true,
        ihsan_score: 0.96,
        timestamp: ts,
    };
    let v2 = Vote {
        proposal_id: "prop_abc".into(),
        voter_id: NodeId("abcdef1234567890abcdef1234567890".into()),
        approve: false,
        ihsan_score: 0.96,
        timestamp: ts,
    };
    assert_ne!(v1.to_bytes(), v2.to_bytes());
}

// ---------------------------------------------------------------------------
// SignedVote
// ---------------------------------------------------------------------------

#[test]
fn signed_vote_sign_and_verify() {
    let identity = make_identity();
    let v = Vote {
        proposal_id: "prop_test".into(),
        voter_id: identity.node_id().clone(),
        approve: true,
        ihsan_score: 0.97,
        timestamp: chrono::Utc::now(),
    };
    let signed = SignedVote::new(v, identity.signing_key());
    assert!(signed.verify().is_ok());
}

#[test]
fn signed_vote_rejects_tampered_signature() {
    let identity = make_identity();
    let v = Vote {
        proposal_id: "prop_tamper".into(),
        voter_id: identity.node_id().clone(),
        approve: true,
        ihsan_score: 0.96,
        timestamp: chrono::Utc::now(),
    };
    let mut signed = SignedVote::new(v, identity.signing_key());
    signed.signature[0] ^= 0xFF;
    assert!(signed.verify().is_err());
}

#[test]
fn signed_vote_rejects_wrong_pubkey() {
    let id1 = make_identity();
    let id2 = make_identity();
    let v = Vote {
        proposal_id: "prop_wrong_key".into(),
        voter_id: id1.node_id().clone(),
        approve: true,
        ihsan_score: 0.96,
        timestamp: chrono::Utc::now(),
    };
    let mut signed = SignedVote::new(v, id1.signing_key());
    signed.sender_pubkey = id2.public_key_bytes();
    assert!(signed.verify().is_err());
}

#[test]
fn signed_vote_pubkey_hex_length() {
    let identity = make_identity();
    let v = Vote {
        proposal_id: "prop_hex".into(),
        voter_id: identity.node_id().clone(),
        approve: true,
        ihsan_score: 0.96,
        timestamp: chrono::Utc::now(),
    };
    let signed = SignedVote::new(v, identity.signing_key());
    assert_eq!(signed.pubkey_hex().len(), 64);
}

// ---------------------------------------------------------------------------
// ConsensusEngine — core flow
// ---------------------------------------------------------------------------

#[test]
fn engine_new_starts_empty() {
    let identity = make_identity();
    let engine = ConsensusEngine::new(identity);
    assert_eq!(engine.committed_count(), 0);
}

#[test]
fn engine_propose_succeeds_above_threshold() {
    let identity = make_identity();
    let mut engine = ConsensusEngine::new(identity);
    let result = engine.propose(serde_json::json!({"pattern": "test"}), 0.96);
    assert!(result.is_ok());
    let proposal = result.unwrap();
    assert!(proposal.id.starts_with("prop_"));
}

#[test]
fn engine_propose_fails_below_ihsan() {
    let identity = make_identity();
    let mut engine = ConsensusEngine::new(identity);
    let result = engine.propose(serde_json::json!({"bad": true}), 0.5);
    assert!(result.is_err());
    match result.unwrap_err() {
        ConsensusError::IhsanThreshold(score) => {
            assert!((score - 0.5).abs() < f64::EPSILON);
        }
        _ => panic!("expected IhsanThreshold"),
    }
}

#[test]
fn engine_vote_fails_for_unknown_proposal() {
    let identity = make_identity();
    let engine = ConsensusEngine::new(identity);
    let result = engine.vote("nonexistent", true, 0.96);
    assert!(result.is_err());
    match result.unwrap_err() {
        ConsensusError::ProposalNotFound(id) => assert_eq!(id, "nonexistent"),
        _ => panic!("expected ProposalNotFound"),
    }
}

#[test]
fn engine_vote_creates_valid_signed_vote() {
    let identity = make_identity();
    let mut engine = ConsensusEngine::new(identity);
    let proposal = engine
        .propose(serde_json::json!({"test": 1}), 0.96)
        .unwrap();
    let signed_vote = engine.vote(&proposal.id, true, 0.96).unwrap();
    assert!(signed_vote.verify().is_ok());
    assert!(signed_vote.vote.approve);
}

#[test]
fn engine_single_node_self_vote_reaches_quorum() {
    // With 1 node, quorum = (2*1/3)+1 = 1
    let identity = make_identity();
    let mut engine = ConsensusEngine::new(identity);
    let proposal = engine
        .propose(serde_json::json!({"quorum": true}), 0.96)
        .unwrap();
    let signed_vote = engine.vote(&proposal.id, true, 0.96).unwrap();
    let reached = engine.receive_vote(signed_vote).unwrap();
    assert!(reached); // quorum reached with 1 vote on 1-node network
}

#[test]
fn engine_commit_after_quorum() {
    let identity = make_identity();
    let mut engine = ConsensusEngine::new(identity);
    let proposal = engine
        .propose(serde_json::json!({"commit": true}), 0.96)
        .unwrap();
    let pid = proposal.id.clone();
    let sv = engine.vote(&pid, true, 0.96).unwrap();
    engine.receive_vote(sv).unwrap();
    assert!(engine.commit(&pid).is_ok());
    assert!(engine.is_committed(&pid));
    assert_eq!(engine.committed_count(), 1);
}

#[test]
fn engine_commit_fails_if_not_committing() {
    let identity = make_identity();
    let mut engine = ConsensusEngine::new(identity);
    let proposal = engine
        .propose(serde_json::json!({"early": true}), 0.96)
        .unwrap();
    // Don't vote → still in Voting state
    let result = engine.commit(&proposal.id);
    assert!(result.is_err());
    match result.unwrap_err() {
        ConsensusError::NotReady => {}
        _ => panic!("expected NotReady"),
    }
}

#[test]
fn engine_commit_fails_for_unknown_proposal() {
    let identity = make_identity();
    let mut engine = ConsensusEngine::new(identity);
    let result = engine.commit("nonexistent");
    assert!(result.is_err());
}

#[test]
fn engine_receive_vote_rejects_below_ihsan() {
    let identity = make_identity();
    let mut engine = ConsensusEngine::new(identity);
    let proposal = engine
        .propose(serde_json::json!({"ihsan": true}), 0.96)
        .unwrap();

    // Use a separate identity registered as peer to forge a low-ihsan vote.
    let peer = make_identity();
    engine.register_peer(peer.node_id().clone(), peer.public_key_bytes());
    engine.set_node_count(2);

    let low_vote = Vote {
        proposal_id: proposal.id.clone(),
        voter_id: peer.node_id().clone(),
        approve: true,
        ihsan_score: 0.3,
        timestamp: chrono::Utc::now(),
    };
    let signed = SignedVote::new(low_vote, peer.signing_key());
    let result = engine.receive_vote(signed);
    assert!(result.is_err());
    match result.unwrap_err() {
        ConsensusError::IhsanThreshold(_) => {}
        e => panic!("expected IhsanThreshold, got {:?}", e),
    }
}

#[test]
fn engine_receive_vote_rejects_unknown_voter() {
    let identity = make_identity();
    let mut engine = ConsensusEngine::new(identity);
    let proposal = engine
        .propose(serde_json::json!({"unknown": true}), 0.96)
        .unwrap();

    let unknown = make_identity();
    let vote = Vote {
        proposal_id: proposal.id.clone(),
        voter_id: unknown.node_id().clone(),
        approve: true,
        ihsan_score: 0.96,
        timestamp: chrono::Utc::now(),
    };
    let signed = SignedVote::new(vote, unknown.signing_key());
    let result = engine.receive_vote(signed);
    assert!(result.is_err());
    match result.unwrap_err() {
        ConsensusError::UnknownVoter(_) => {}
        e => panic!("expected UnknownVoter, got {:?}", e),
    }
}

#[test]
fn engine_receive_vote_rejects_pubkey_mismatch() {
    let identity = make_identity();
    let mut engine = ConsensusEngine::new(identity);
    let proposal = engine
        .propose(serde_json::json!({"mismatch": true}), 0.96)
        .unwrap();

    // Register peer with one key, sign with another
    let peer = make_identity();
    let other_key = NodeIdentity::generate();
    engine.register_peer(peer.node_id().clone(), peer.public_key_bytes());
    engine.set_node_count(2);

    let vote = Vote {
        proposal_id: proposal.id.clone(),
        voter_id: peer.node_id().clone(),
        approve: true,
        ihsan_score: 0.96,
        timestamp: chrono::Utc::now(),
    };
    // Sign with the OTHER key, but the vote's voter_id is peer's id
    let signed = SignedVote::new(vote, other_key.signing_key());
    let result = engine.receive_vote(signed);
    assert!(result.is_err());
    match result.unwrap_err() {
        ConsensusError::PubkeyMismatch { .. } => {}
        e => panic!("expected PubkeyMismatch, got {:?}", e),
    }
}

#[test]
fn engine_multi_node_quorum() {
    let id1 = make_identity();
    let id2 = make_identity();
    let id3 = make_identity();

    let mut engine = ConsensusEngine::new(NodeIdentity::from_secret_bytes(&id1.secret_bytes()));
    engine.register_peer(id2.node_id().clone(), id2.public_key_bytes());
    engine.register_peer(id3.node_id().clone(), id3.public_key_bytes());
    engine.set_node_count(3); // quorum = (2*3/3)+1 = 3

    let proposal = engine
        .propose(serde_json::json!({"multi": true}), 0.96)
        .unwrap();
    let pid = proposal.id.clone();

    // Vote 1: self (approval)
    let sv1 = engine.vote(&pid, true, 0.96).unwrap();
    let reached1 = engine.receive_vote(sv1).unwrap();
    assert!(!reached1); // 1/3, not enough

    // Vote 2: id2 (approval)
    let v2 = Vote {
        proposal_id: pid.clone(),
        voter_id: id2.node_id().clone(),
        approve: true,
        ihsan_score: 0.96,
        timestamp: chrono::Utc::now(),
    };
    let sv2 = SignedVote::new(v2, id2.signing_key());
    let reached2 = engine.receive_vote(sv2).unwrap();
    assert!(!reached2); // 2/3, not enough

    // Vote 3: id3 (approval)
    let v3 = Vote {
        proposal_id: pid.clone(),
        voter_id: id3.node_id().clone(),
        approve: true,
        ihsan_score: 0.96,
        timestamp: chrono::Utc::now(),
    };
    let sv3 = SignedVote::new(v3, id3.signing_key());
    let reached3 = engine.receive_vote(sv3).unwrap();
    assert!(reached3); // 3/3 = quorum
}

#[test]
fn engine_rejection_no_quorum() {
    let id1 = make_identity();
    let id2 = make_identity();

    let mut engine = ConsensusEngine::new(NodeIdentity::from_secret_bytes(&id1.secret_bytes()));
    engine.register_peer(id2.node_id().clone(), id2.public_key_bytes());
    engine.set_node_count(2); // quorum = (2*2/3)+1 = 2

    let proposal = engine
        .propose(serde_json::json!({"reject": true}), 0.96)
        .unwrap();
    let pid = proposal.id.clone();

    // Vote 1: self (reject)
    let sv1 = engine.vote(&pid, false, 0.96).unwrap();
    let reached = engine.receive_vote(sv1).unwrap();
    assert!(!reached);

    // Vote 2: id2 (reject)
    let v2 = Vote {
        proposal_id: pid.clone(),
        voter_id: id2.node_id().clone(),
        approve: false,
        ihsan_score: 0.96,
        timestamp: chrono::Utc::now(),
    };
    let sv2 = SignedVote::new(v2, id2.signing_key());
    let reached2 = engine.receive_vote(sv2).unwrap();
    assert!(!reached2); // 0 approvals, quorum never reached
}

// ---------------------------------------------------------------------------
// ConsensusError display
// ---------------------------------------------------------------------------

#[test]
fn consensus_error_display_messages() {
    assert!(ConsensusError::ProposalNotFound("x".into())
        .to_string()
        .contains("x"));
    assert!(ConsensusError::IhsanThreshold(0.5)
        .to_string()
        .contains("0.5"));
    assert!(ConsensusError::NotVoting.to_string().contains("voting"));
    assert!(ConsensusError::NotReady.to_string().contains("ready"));
    assert!(ConsensusError::InvalidPublicKey
        .to_string()
        .contains("public key"));
    assert!(ConsensusError::InvalidSignature
        .to_string()
        .contains("signature"));
}
